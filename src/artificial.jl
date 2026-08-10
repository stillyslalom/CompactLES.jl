# Cook-style artificial fluid properties (Cook, Phys. Fluids 2007), extended
# with the artificial species diffusivity for multicomponent mixing.
#
# Sensors use an undivided high-pass in the computational indices — Cook's
# explicit fourth difference δ⁴ (1, −4, 6, −4, 1) by default, or the reference
# implementation's compact eighth derivative under
# `ArtParams.detector = :d8` — so the formal grid-spacing powers reduce to
# per-dimension weights: h² for μ*, β* (from |∇⁴S| Δ⁶), h for κ* (|∇⁴e| Δ⁵)
# and for D* (|∇⁴Y| Δ⁵). β* optionally keys on compression, either by gating
# the strain sensor or by rebuilding it from the dilatation, which carries the
# same h² weight (`ArtParams.beta_sensor`; see `gate_beta!` and
# `dilatation_beta!`). The Gaussian test filter is `smooth!`, selected by
# `ArtParams.smoother`. Each species carries its own sensor and its own D*_k;
# Σ_k J_k = 0 is then restored by the correction velocity in the flux assembly
# (rhs.jl) rather than by giving every species the same diffusivity. Indices
# are clamped at closed physical edges; halos cover rank boundaries. On
# curvilinear grids the sensors are computed in computational space — a
# grid-based rather than strictly physical-space regularization, standard
# practice and consistent with resolving power following the mesh.

"""
    ArtParams(; enabled=true, C_mu=0.002, C_beta=1.0, C_kappa=0.01, C_D=0.01,
              beta_sensor=:strain, smoother=:gaussian, detector=:delta4)

Cook-style artificial-property controls.

# Keywords

- `enabled`: compute and apply artificial properties. Set this to `false` to
  skip the complete sensor and artificial-flux calculation.
- `C_mu`: coefficient for artificial shear viscosity generated from the strain
  sensor.
- `C_beta`: coefficient for artificial bulk viscosity; this is the primary
  shock-spreading term.
- `C_kappa`: coefficient for artificial conductivity generated from the
  internal-energy sensor.
- `C_D`: coefficient for per-species artificial diffusivity generated from
  mass-fraction sensors.
- `beta_sensor`: how the β\\* sensor is built. `:strain` is the Cook original,
  Σ_d h_d²|δ⁴_d S|. `:gated_strain` keeps that sensor and multiplies it by a
  Ducros-style compression switch, for one pointwise pass (`gate_beta!`).
  `:dilatation` additionally rebuilds the sensor from ∇·u, for one further
  smoothing pass per RHS evaluation (`dilatation_beta!`). The measured
  differences are in `reference/CALIBRATION.md`.
- `smoother`: which operator stands in for Cook's Gaussian test filter in
  [`smooth!`](@ref). `:gaussian` (default) is [`gaussian_filter`](@ref), the
  explicit nine-point stencil, which carries no line solve and no interface
  collective. `:compact` reuses `Numerics.filt` and is what shipped before this
  option existed; it retains 99% of the amplitude at four points per wavelength
  where the Gaussian retains 19%, which leaves the β\\* field rough enough to
  drop out intermittently at a symmetry cell. The default was changed on that
  measurement: it raises the spherical-origin CFL ceiling from 0.15 to 0.4 and
  the cylindrical from 0.15 to 0.2, and makes the sensor phase 29% cheaper, at
  the cost of about seven points of planar wall heating. The four constants
  above are calibrated per setting. All of it is in `reference/CALIBRATION.md`.
- `detector`: the high-pass that builds every sensor, in
  [`detect_sum!`](@ref). `:delta4` (default) is Cook's undivided fourth
  difference, computed explicitly. `:d8` is the reference implementation's
  compact eighth derivative ([`compact_d8`](@ref)), which is two to three
  orders of magnitude more selective below the Nyquist and costs a
  pentadiagonal line solve per direction per sensor, where `:delta4` costs
  none. Both carry the same per-dimension `h` weights and are normalized to
  the same grid-oscillation response, so the four constants transfer between
  them as starting points.

The coefficients are dimensionless numerical regularization parameters, not
material properties. Their useful values depend on resolution, flow regime,
the scheme supplied as `Numerics.filt`, and filter cadence. The displayed
defaults are documented starting points rather than universal values; larger
values can reduce the explicit diffusive timestep.
"""
Base.@kwdef struct ArtParams{T}
    enabled::Bool = true
    C_mu::T    = 0.002
    C_beta::T  = 1.0
    C_kappa::T = 0.01
    C_D::T     = 0.01
    beta_sensor::Symbol = :strain
    smoother::Symbol = :gaussian
    detector::Symbol = :delta4
end

const D4 = (1.0, -4.0, 6.0, -4.0, 1.0)   # offsets −2:2

"""
    delta4_sum!(out, f, solver, wpow; accumulate=false)

Interior accumulation of Σ_d h_d^wpow |δ⁴_d f| into `out` (added to existing
contents when `accumulate`). Requires current rank-boundary halos of `f`.
"""
function delta4_sum!(out, f, solver, wpow::Int; accumulate::Bool=false)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    accumulate || fill!(out, 0)
    for d in 1:3
        decomp.active[d] || continue
        wd = solver.h[d]^wpow
        n_d = decomp.n_local[d]
        lomin = at_lo_edge(decomp, d) ? 1 : -1
        himax = at_hi_edge(decomp, d) ? n_d : n_d + 2
        e = CartesianIndex(ntuple(k -> k == d ? 1 : 0, 3))
        @threaded nx*ny*nz for jk in outer_indices(ny, nz)
            j, k = Tuple(jk)
            @inbounds for i in 1:nx
                I = CartesianIndex(i + o1, j + o2, k + o3)
                il = (d == 1 ? i : d == 2 ? j : k)
                acc = 0.0
                for m in -2:2
                    ilm = clamp(il + m, lomin, himax)
                    acc += D4[m + 3] * f[I + (ilm - il) * e]
                end
                out[I] += wd * abs(acc)
            end
        end
    end
    return out
end

"""
    ring_sum!(out, f, solver, wpow; accumulate=false)

The [`compact_d8`](@ref) counterpart of [`delta4_sum!`](@ref): accumulates
Σ_d h_d^wpow |d⁸_d f| into `out`, one distributed pentadiagonal line solve per
active dimension. Requires current rank-boundary halos of `f` to depth 4.

`f` is always an even scalar across a coordinate fold — the strain magnitude,
the internal energy, a mass fraction, the dilatation — so the antipodal sign
passed through is `+1`. Closed physical edges are handled by the scheme's own
closure rows, where `delta4_sum!` instead clamps its indices.

`solver.ring_buf` receives the directional result and is scratch belonging to
this function alone. That it is not `tmp_a` or `tmp_b` is deliberate: both of
those hold live inputs at two of the four call sites (the internal energy for
κ\\*, and the dilatation and its compression switch in
[`dilatation_beta!`](@ref)).
"""
function ring_sum!(out, f, solver, wpow::Int; accumulate::Bool=false)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    accumulate || fill!(out, 0)
    # Read out of the solver before the loop, not inside the threaded closure:
    # a closure capturing `solver` allocates once per region in proportion to
    # its size, as recorded on `compute_artificial!`.
    ring_buf = solver.ring_buf
    for d in 1:3
        decomp.active[d] || continue
        wd = solver.h[d]^wpow
        ring_along!(ring_buf, f, solver, d, 1)
        @threaded nx*ny*nz for jk in outer_indices(ny, nz)
            j, k = Tuple(jk)
            @inbounds for i in 1:nx
                I = CartesianIndex(i + o1, j + o2, k + o3)
                out[I] += wd * abs(ring_buf[I])
            end
        end
    end
    return out
end

"""
    detect_sum!(out, f, solver, wpow; accumulate=false)

Apply the detector named by `ArtParams.detector` — [`delta4_sum!`](@ref) or
[`ring_sum!`](@ref) — to build one sensor field.

The choice is made once per sensor rather than inside either kernel: both are
full array passes, and a per-point test would cost more than the difference
between the detectors themselves.

It is also made on a type rather than on the `Symbol`. `solver.ring_plans` is
`nothing` exactly when the detector is `:delta4`, so that case never reaches
`ring_sum!` at all — reaching it in dead code would leave `bench/jetcheck.jl`
reporting the runtime dispatch of an `apply_along!` on an absent plan, charged
to every run whether or not it uses the detector. Both settings are setup-time
constants identical on every rank, so either form of the branch is safe to sit
above `ring_sum!`'s collectives.
"""
detect_sum!(out, f, solver, wpow::Int; accumulate::Bool=false) =
    _detect_sum!(out, f, solver, wpow, accumulate, solver.ring_plans)

_detect_sum!(out, f, solver, wpow::Int, acc::Bool, ::Nothing) =
    delta4_sum!(out, f, solver, wpow; accumulate=acc)
_detect_sum!(out, f, solver, wpow::Int, acc::Bool, ::Tuple) =
    ring_sum!(out, f, solver, wpow; accumulate=acc)

"""
    smooth!(f, solver)

One directional smoother pass per active dimension, standing in for Cook's
Gaussian test filter (even-parity axis fill along r when applicable). Which
operator runs is `ArtParams.smoother`; see [`smooth_along!`](@ref).
"""
function smooth!(f, solver)
    for d in 1:3
        solver.decomp.active[d] || continue
        # Only dimension d: the filter about to run is a 1-D stencil along d,
        # so the other two dimensions' halos are dead. This used to call
        # exchange_halos!, refreshing all three every time round the loop --
        # three times the traffic it needs in a 3-D run, on a routine that
        # runs once per sensor per species per RHS.
        exchange_dim!(f, solver.decomp, d)
        smooth_along!(solver.tmp_a, f, solver, d, 1)
        copy_interior!(f, solver.tmp_a, solver.decomp)
    end
    return f
end

# Guard against 0/0 in the Ducros ratio where the flow is exactly quiescent. It
# is an absolute floor rather than a fraction of a local scale, and the value is
# the one used in the source literature. A relative floor scaled to the local |S|
# was tried and reverted: it moves the twelfth digit on Taylor-Green at 32^3 and
# nothing else, because the points where the gate fails to suppress a solenoidal
# flow are the cusps of |S| — where the strain sensor is large precisely because
# |S| passes through zero, taking any local scale a floor could use with it. See
# `gate_beta!`.
const DUCROS_EPS = 1e-32

"""
    compression_switch(grad_u, I)

The shock switch H(−Δ)·Δ²/(Δ² + |ω|² + ε) at one point, from the dilatation
Δ = ∇·u and the vorticity ω (Ducros et al., JCP 152, 1999; in this form Mani,
Larsson & Moin, JCP 228, 2009). It is zero in expansion and small wherever
vorticity dominates dilatation, so multiplying a sensor by it restricts that
sensor to compression.

`grad_u[d, j]` is ∂u_j/∂x_d, so the trace is the dilatation and the
off-diagonal pairs are the curl. Both are already available wherever the
sensors are built, which is what makes the switch free.
"""
@inline function compression_switch(grad_u, I)
    @inbounds begin
        div = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
        div < 0 || return 0.0
        w1 = grad_u[2, 3][I] - grad_u[3, 2][I]
        w2 = grad_u[3, 1][I] - grad_u[1, 3][I]
        w3 = grad_u[1, 2][I] - grad_u[2, 1][I]
        return div^2 / (div^2 + w1^2 + w2^2 + w3^2 + DUCROS_EPS)
    end
end

"""
    gate_beta!(solver)

Multiply `solver.beta_art` in place by `compression_switch`, leaving the
Cook strain sensor that produced it untouched. Selected by
`ArtParams(beta_sensor = :gated_strain)`.

This is the shock-switch half of the Mani, Larsson & Moin refinement without
the sensor-field half. The two are separable and they do different things: the
switch confines β\\* to compression, while changing the sensor from |S| to Δ
also changes *how much* β\\* a given compression produces. The second half is
what loses the coordinate folds — see `dilatation_beta!` — so this variant
exists to take the first half alone.

Cost is one pointwise pass over the interior. No sensor, no smoothing, no line
solve, in contrast to `:dilatation`.

The switch cannot suppress β\\* at a cusp of |S|. The strain sensor is a fourth
difference, so it peaks where |S| passes through zero with a kink, and the
vorticity generally vanishes there too, leaving only ε in the denominator. On a
solenoidal Taylor–Green field this leaves 71 of 32768 points carrying β\\* — 0.6%
of the summed total, but the full maximum. `reference/CALIBRATION.md` has the
measurement and why a relative ε does not help.
"""
function gate_beta!(solver)
    o1, o2, o3 = solver.decomp.n_halo_d
    nx, ny, nz = solver.decomp.n_local
    grad_u = solver.grad_u
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        @inbounds for i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            solver.beta_art[I] *= compression_switch(grad_u, I)
        end
    end
    return solver
end

"""
    dilatation_beta!(solver, C_beta)

Rebuild `solver.beta_art` from a compression-keyed sensor, replacing the strain
form. Selected by `ArtParams(beta_sensor = :dilatation)`.

The sensor is Σ_d h_d² |δ⁴_d Δ| built from the dilatation Δ = ∇·u, smoothed as
the strain sensor is, and then multiplied by `compression_switch`
(Mani, Larsson & Moin, JCP 228, 2009; Kawai, Shankar & Lele, JCP 229, 2010).
The strain form fires on any fourth difference of |S|, so it also spreads
vortical structures and expansion fans; this form restricts β\\* to compression,
which is where bulk viscosity has a physical interpretation. `gate_beta!`
applies the same switch to the unchanged strain sensor, which is the other half
of the same refinement and a great deal cheaper.

Cost is one extra `delta4_sum!` and one extra `smooth!` per RHS evaluation —
the latter is a compact filter solve per active dimension — because μ\\* still
needs the strain sensor. `grad_u` supplies both Δ and ω, so the switch itself
is free.

Unlike the strain form, this one does not reproduce bit-for-bit under a changed
decomposition. H(−Δ) is discontinuous at Δ = 0 and, with ε at the literature
value, the ratio has not yet decayed there, so a point whose dilatation is zero
to round-off carries either no β\\* or the full C_β·ρ·sensor depending on the
last bit of a cancelling sum. The measured spread over three split axes is
2e-7 relative, against 1e-14 for the strain sensor; `test/mpi_tests.jl` holds
the guard.

`solver.tmp_a` and `solver.tmp_b` are scratch here, under the invariant
recorded on `compute_artificial!`.
"""
function dilatation_beta!(solver, C_beta)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    grad_u = solver.grad_u
    # Δ into tmp_a, the switch into tmp_b.
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        @inbounds for i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            solver.tmp_a[I] = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
            solver.tmp_b[I] = compression_switch(grad_u, I)
        end
    end
    exchange_halos!(solver.tmp_a, decomp)
    detect_sum!(solver.sensor, solver.tmp_a, solver, 2)
    smooth!(solver.sensor, solver)      # clobbers tmp_a, which is spent by now
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        @inbounds for i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            solver.beta_art[I] = C_beta * solver.rho[I] *
                                 max(solver.sensor[I], 0.0) * solver.tmp_b[I]
        end
    end
    return solver
end

"""
    compute_artificial!(solver, Q)

Fill solver.mu_art, solver.beta_art, solver.kappa_art and solver.D_art from
the current primitives and (metric-corrected) velocity gradients. No-op if
disabled.

`solver.sensor`, `solver.sensor_sp`, `solver.tmp_a` and `solver.tmp_b` are
scratch here, and their contents are dead once this function returns: every
value read out of them has been folded into the four coefficient arrays above.
Later phases of the same `compute_rhs!` may therefore reuse them, and
`NSCBCOutflowBC` does, for transverse pressure derivatives. A diagnostic, source
term, or boundary condition that adds a reader of a sensor after this point
invalidates the invariant and must take its own storage. One reader already
exists outside the RHS: `scalar_field(solver, :sensor)` returns `solver.sensor`,
which is why the NSCBC correction borrows `tmp_b` instead.
"""
function compute_artificial!(solver, Q)
    art = solver.art
    if !art.enabled
        return solver   # arrays stay zero from allocation
    end
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    grad_u = solver.grad_u
    # The coefficients are read into locals rather than off `art` inside the
    # loops. A threaded closure that captures the struct allocates once per
    # region in proportion to the closure's size: 768 B per RHS call when this
    # was written the obvious way. The driver is that size, not `beta_sensor`
    # making ArtParams non-isbits, since an isbits struct of the same shape
    # measures the same +128 B per region when captured. Unpacking is therefore
    # the fix, and a further ArtParams field costs nothing so long as no loop
    # closes over the struct. A field consulted *inside* a loop is the case to
    # avoid: keep the test at a function barrier, as `beta_sensor` is below.
    C_mu, C_beta = art.C_mu, art.C_beta
    C_kappa, C_D = art.C_kappa, art.C_D

    # Strain-rate magnitude |S| = sqrt(S_ij S_ij) in the interior (physical
    # components — the metric corrections are already in grad_u).
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        @inbounds for i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ss = 0.0
            for b in 1:3, a in 1:3
                Sab = 0.5 * (grad_u[a, b][I] + grad_u[b, a][I])
                ss += Sab * Sab
            end
            solver.strain_mag[I] = sqrt(ss)
        end
    end

    # μ*, β* sensor: Σ_d h_d² |D_d S| for the selected detector D, smoothed.
    exchange_halos!(solver.strain_mag, decomp)
    detect_sum!(solver.sensor, solver.strain_mag, solver, 2)
    smooth!(solver.sensor, solver)
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        @inbounds for i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ρsensor = solver.rho[I] * max(solver.sensor[I], 0.0)
            solver.mu_art[I]   = C_mu   * ρsensor
            solver.beta_art[I] = C_beta * ρsensor
        end
    end
    # β* may instead be keyed on compression, either by gating what the block
    # above just wrote into beta_art or by rebuilding it from a different
    # sensor. Both leave mu_art alone.
    art.beta_sensor === :gated_strain && gate_beta!(solver)
    art.beta_sensor === :dilatation   && dilatation_beta!(solver, C_beta)

    # κ* sensor: Σ_d h_d |D_d e|, smoothed; κ* = C_κ · scale(EOS) · sensor,
    # with the scale ρc/T_ion for the gas models. Internal energy comes straight
    # from Q, so the sensor itself is EOS-agnostic; the scale is an EOS query
    # (see the note in physics.jl on why it is the weaker of the two
    # abstractions and what it is still singular in).
    i_energy = solver.equations.i_energy
    m1, m2, m3 = solver.equations.i_mom
    nxf, nyf, nzf = size(solver.tmp_a)
    @threaded nxf*nyf*nzf for jk in outer_indices(nyf, nzf)
        j, k = Tuple(jk)
        @inbounds for i in 1:nxf
            ρ = max(solver.rho[i, j, k], 1e-300)
            ke = 0.5 * (Q[i,j,k,m1]^2 + Q[i,j,k,m2]^2 + Q[i,j,k,m3]^2) / ρ
            solver.tmp_a[i, j, k] = (Q[i, j, k, i_energy] - ke) / ρ
        end
    end
    exchange_halos!(solver.tmp_a, decomp)
    detect_sum!(solver.sensor, solver.tmp_a, solver, 1)
    smooth!(solver.sensor, solver)
    eos = solver.eos
    @threaded nx*ny*nz for jk in outer_indices(ny, nz)
        j, k = Tuple(jk)
        @inbounds for i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            scale = art_conductivity_scale(eos, solver.rho[I], solver.c[I],
                                           solver.T_ion[I], solver.cp_mix[I])
            solver.kappa_art[I] = C_kappa * scale * max(solver.sensor[I], 0.0)
        end
    end

    # Per-species D*_k sensors: Σ_d h_d |D_d Y_k|, each smoothed;
    # D*_k = C_D c · sensor_k. Costs n_species filter sweeps per RHS; the flux
    # assembly's correction velocity keeps Σ_k J_k = 0 despite unequal D_k.
    # Only meaningful with more than one species.
    if solver.equations.n_species > 1
        for sp in 1:solver.equations.n_species
            detect_sum!(solver.sensor_sp, solver.Y[sp], solver, 1)
            smooth!(solver.sensor_sp, solver)
            @threaded nx*ny*nz for jk in outer_indices(ny, nz)
                j, k = Tuple(jk)
                @inbounds for i in 1:nx
                    I = CartesianIndex(i + o1, j + o2, k + o3)
                    solver.D_art[sp][I] = C_D * solver.c[I] * max(solver.sensor_sp[I], 0.0)
                end
            end
        end
    end
    return solver
end
