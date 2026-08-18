# Cook-style artificial fluid properties (Cook, Phys. Fluids 2007), extended
# with the artificial species diffusivity for multicomponent mixing.
#
# Sensors use an undivided high-pass in the computational indices — Cook's
# explicit fourth difference δ⁴ (1, −4, 6, −4, 1) by default, or the reference
# implementation's compact eighth derivative under
# `ArtParams.detector = :d8` — so the formal grid-spacing powers reduce to
# per-dimension weights: h² for a field carrying one velocity derivative (|S|,
# ∇·u) and h for a field carrying none (the velocity components themselves, the
# internal energy, a mass fraction). The two weights coincide at the grid scale,
# so one set of constants covers both.
#
# `ArtParams.mu_sensor` and `ArtParams.beta_sensor` select the field each of the
# two viscosities is built from. Cook takes both from the strain magnitude |S|;
# the reference implementation takes μ* from the velocity components and β* from
# the dilatation, neither of which carries an absolute value (see `velocity_mu!`
# and `dilatation_beta!`). β* may also key on compression through a Ducros
# switch (`gate_beta!`). Directions combine by Σ_d or by MAX, under
# `ArtParams.reduction`. `smooth!` stands in for Cook's Gaussian test filter,
# and `ArtParams.smoother` selects the operator it applies. With more than one
# species, each carries its own sensor and its own D*_k; Σ_k J_k = 0 is then
# restored by the correction velocity in the flux assembly (rhs.jl) rather than
# by giving every species the same diffusivity. Under `:delta4`, indices are
# clamped at closed physical edges, except for an odd field across a fold, which
# takes the half-offset mirror; `:d8` uses the scheme's own closure rows there
# instead. Halos cover rank boundaries. On
# curvilinear grids the sensors are computed in computational space — a
# grid-based rather than strictly physical-space regularization, standard
# practice and consistent with resolving power following the mesh.

"""
    ArtParams(; enabled=true, C_mu=0.002, C_beta=1.0, C_kappa=0.01, C_D=0.01,
              mu_sensor=:strain, beta_sensor=:strain, reduction=:sum,
              smoother=:gaussian, detector=:delta4)

Cook-style artificial-property controls.

# Keywords

- `enabled`: compute the artificial properties. Set this to `false` to skip the
  sensor and coefficient calculation entirely; the four coefficient arrays then
  stay zero and contribute nothing to the assembled fluxes.
- `C_mu`: coefficient for artificial shear viscosity, multiplying whichever
  sensor `mu_sensor` selects.
- `C_beta`: coefficient for artificial bulk viscosity; this is the primary
  shock-spreading term.
- `C_kappa`: coefficient for artificial conductivity generated from the
  internal-energy sensor.
- `C_D`: coefficient for per-species artificial diffusivity generated from
  mass-fraction sensors.
- `mu_sensor`: how the μ\\* sensor is built. `:strain` is the Cook original,
  h_d²|D_d S| from the strain magnitude reduced over directions, for the
  detector D and the reduction the two fields below name. `:velocity` is the
  reference implementation's, built from the velocity components themselves and
  costing three detector applications per direction instead of one
  (`velocity_mu!`).
  The two differ in the absolute value taken by |S|, which places a cusp
  wherever the strain passes through zero; a cusp is grid-scale structure at
  any resolution, so a sensor built from |S| responds to smooth flow.
- `beta_sensor`: how the β\\* sensor is built. `:strain` is the Cook original
  and shares its sensor with `mu_sensor = :strain` when both are selected.
  `:gated_strain` keeps that sensor and multiplies it by a Ducros-style
  compression switch, for one pointwise pass (`gate_beta!`).
  `:ungated_dilatation` rebuilds the sensor from ∇·u, which is the reference
  implementation's form, and `:dilatation` applies the compression switch to
  that as well; both cost one further smoothing pass per RHS evaluation
  (`dilatation_beta!`). The measured differences are in
  `reference/CALIBRATION.md`.
- `reduction`: how the per-direction detector outputs combine into one sensor.
  `:sum` is Σ_d, which is Cook's form and grows with the number of active
  dimensions; `:max` is the reference implementation's MAX, which does not. The
  two are identical in a one-dimensional calculation.
- `smoother`: which operator stands in for Cook's Gaussian test filter in
  `smooth!`. `:gaussian` (default) is [`gaussian_filter`](@ref), the
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
  `detect_sum!`. `:delta4` (default) is Cook's undivided fourth
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
    mu_sensor::Symbol = :strain
    beta_sensor::Symbol = :strain
    reduction::Symbol = :sum
    smoother::Symbol = :gaussian
    detector::Symbol = :delta4
end

const D4 = (1.0, -4.0, 6.0, -4.0, 1.0)   # offsets −2:2

"""
    delta4_sum!(out, f, solver, wpow; accumulate=false, parity=(1, 1, 1))

Interior reduction of h_d^wpow |δ⁴_d f| over the active directions into `out`,
combined by Σ_d or by MAX according to `ArtParams.reduction` and combined with
the existing contents when `accumulate`. Requires current rank-boundary halos
of `f` to depth 2; the kernel itself communicates nothing.

`parity[d]` is the field's sign across a coordinate fold on dimension `d`,
which is −1 only for a velocity component (`velocity_mu!`). It is applied at a
folded edge and nowhere else: an outer wall carries no reflection symmetry to
exploit, and the fold is also the only edge whose grid is half-offset.

Indices at a closed edge are clamped, a zeroth-order extension of the field
past it, except across a fold with `parity[d] == -1`, where the half-offset
mirror (ghost j ↔ interior j, negated) is used instead. The two differ in
order. For an even field the clamp misplaces one δ⁴ tap by a term that the
vanishing edge derivative makes O(h²); for an odd field the edge derivative is
the largest quantity there, the same tap is wrong at O(h), and the result is a
nonzero sensor on a field as regular as u_r = r, which the mirror annihilates
exactly. Putting the even path on the mirror as well would move guarded numbers
for an effect expected to be small, and nobody has measured it;
`reference/CALIBRATION.md` carries it as an open item.
"""
function delta4_sum!(out, f, solver, wpow::Int; accumulate::Bool=false,
                     parity::NTuple{3,Int}=(1, 1, 1))
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    maxred = solver.art.reduction === :max
    accumulate || fill!(out, 0)
    for d in 1:3
        decomp.active[d] || continue
        wd = solver.h[d]^wpow
        n_d = decomp.n_local[d]
        lomin = at_lo_edge(decomp, d) ? 1 : -1
        himax = at_hi_edge(decomp, d) ? n_d : n_d + 2
        fold = solver.folds[d]
        odd_lo = parity[d] == -1 && at_lo_edge(decomp, d) &&
                 fold !== nothing && fold.lo
        odd_hi = parity[d] == -1 && at_hi_edge(decomp, d) &&
                 fold !== nothing && fold.hi
        pointwise!(_delta4_point!, out, nx, ny, nz,
                   out, f, wd, d, n_d, lomin, himax, odd_lo, odd_hi, maxred,
                   o1, o2, o3)
    end
    return out
end

@inline function _delta4_point!(out, f, wd, d, n_d, lomin, himax,
                                odd_lo, odd_hi, maxred, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        e = CartesianIndex(ntuple(q -> q == d ? 1 : 0, 3))
        il = (d == 1 ? i : d == 2 ? j : k)
        acc = 0.0
        # The parity test sits outside the stencil loop rather than on each
        # tap: both flags are invariant over the whole sweep, and the clamped
        # branch is the one every sensor but a velocity component across a
        # fold takes.
        if !(odd_lo | odd_hi)
            for m in -2:2
                ilm = clamp(il + m, lomin, himax)
                acc += D4[m + 3] * f[I + (ilm - il) * e]
            end
        else
            for m in -2:2
                q = il + m
                if odd_lo && q < 1              # ghost 1−q, negated
                    acc -= D4[m + 3] * f[I + (1 - q - il) * e]
                elseif odd_hi && q > n_d        # ghost 2n+1−q, negated
                    acc -= D4[m + 3] * f[I + (2n_d + 1 - q - il) * e]
                else
                    ilm = clamp(q, lomin, himax)
                    acc += D4[m + 3] * f[I + (ilm - il) * e]
                end
            end
        end
        v = wd * abs(acc)
        out[I] = maxred ? max(out[I], v) : out[I] + v
    end
    return nothing
end

"""
    ring_sum!(out, f, solver, wpow; accumulate=false, parity=(1, 1, 1))

The [`compact_d8`](@ref) counterpart of [`delta4_sum!`](@ref): reduces
h_d^wpow |d⁸_d f| over the active directions into `out`, one distributed
pentadiagonal line solve per direction, combined with the existing contents of
`out` when `accumulate`, exactly as `delta4_sum!` does. Requires current
rank-boundary halos of `f` to depth 4. Collective: every rank must call it.

`parity[d]` is the field's antipodal sign across a fold on dimension `d`, and
is `+1` for every even scalar the sensors are built from: the strain magnitude,
the internal energy, a mass fraction, the dilatation. Only
`velocity_mu!` passes anything else. A closed physical edge that is not
a fold takes the scheme's own closure rows, which mirror symmetrically whatever
the parity of the field; the reference implementation instead carries a second,
antisymmetric closure for that case, and this is the one part of its sensor
construction not reproduced here.

`solver.ring_buf` receives the directional result and is scratch belonging to
this function alone. That it is not `tmp_a` or `tmp_b` is deliberate: both of
those hold live inputs at two of the call sites (the internal energy for
κ\\*, and the dilatation and its compression switch in
[`dilatation_beta!`](@ref)).
"""
function ring_sum!(out, f, solver, wpow::Int; accumulate::Bool=false,
                   parity::NTuple{3,Int}=(1, 1, 1))
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    maxred = solver.art.reduction === :max
    accumulate || fill!(out, 0)
    # Read out of the solver before the loop, not inside the threaded closure:
    # a closure capturing `solver` allocates once per region in proportion to
    # its size, as recorded on `compute_artificial!`.
    ring_buf = solver.ring_buf
    for d in 1:3
        decomp.active[d] || continue
        wd = solver.h[d]^wpow
        ring_along!(ring_buf, f, solver, d, parity[d])
        @threaded nx*ny*nz for jk in outer_indices(ny, nz)
            j, k = Tuple(jk)
            @inbounds for i in 1:nx
                I = CartesianIndex(i + o1, j + o2, k + o3)
                v = wd * abs(ring_buf[I])
                out[I] = maxred ? max(out[I], v) : out[I] + v
            end
        end
    end
    return out
end

"""
    detect_sum!(out, f, solver, wpow; accumulate=false, parity=(1, 1, 1))

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
detect_sum!(out, f, solver, wpow::Int; accumulate::Bool=false,
            parity::NTuple{3,Int}=(1, 1, 1)) =
    _detect_sum!(out, f, solver, wpow, accumulate, parity, solver.ring_plans)

_detect_sum!(out, f, solver, wpow::Int, acc::Bool, par, ::Nothing) =
    delta4_sum!(out, f, solver, wpow; accumulate=acc, parity=par)
_detect_sum!(out, f, solver, wpow::Int, acc::Bool, par, ::Tuple) =
    ring_sum!(out, f, solver, wpow; accumulate=acc, parity=par)

"""
    smooth!(f, solver)

One directional smoother pass per active dimension, standing in for Cook's
Gaussian test filter, applied to `f` in place and returned. A fold is crossed
with even parity, which is correct for every field smoothed here: each is a
detector output, and each detector ends in an absolute value. Which operator
runs is `ArtParams.smoother`; see [`smooth_along!`](@ref).

`solver.tmp_a` is scratch and is overwritten. Collective: the halos of `f` are
exchanged along each active dimension, and under `smoother = :compact` the pass
is itself a distributed line solve, so every rank must call this.
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

@inline function _rho_sensor_point!(dest, rho, sensor, C, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        dest[I] = C * rho[I] * max(sensor[I], 0.0)
    end
    return nothing
end

"""
    rho_sensor!(dest, solver, C)

Write `C · ρ · max(sensor, 0)` over the interior of `dest` from the smoothed
sensor `solver.sensor`, the form μ\\* and β\\* take. κ\\* and D\\* carry their
own scale in place of ρ, an EOS query and the sound speed respectively, and are
assembled in `compute_artificial!` rather than here. The clamp is against a
smoother that undershoots, not against a negative detector output: |δ⁴f| and
|d⁸f| are non-negative by construction.
"""
function rho_sensor!(dest, solver, C)
    o1, o2, o3 = solver.decomp.n_halo_d
    nx, ny, nz = solver.decomp.n_local
    pointwise!(_rho_sensor_point!, dest, nx, ny, nz,
               dest, solver.rho, solver.sensor, C, o1, o2, o3)
    return dest
end

"""
    velocity_mu!(solver, C_mu)

Rebuild `solver.mu_art` from the velocity components, replacing the strain
form. Selected by `ArtParams(mu_sensor = :velocity)`.

The sensor is the reduction of h_d |D_d u_j| over the (direction, component)
pairs, three velocity components by each active direction and so nine pairs in
a three-dimensional run, smoothed as the strain sensor is; this is Miranda's
`ringV`, and `ArtParams.reduction` chooses between its MAX and Cook's Σ. The
weight is h rather than the strain form's h² because u carries one derivative
fewer than |S|, and the two agree at the grid scale: a grid-to-grid oscillation
of amplitude A gives 16Ah either way, so `C_mu` transfers between the settings
as a starting point.

The motivation is the absolute value in |S| = sqrt(S_ij S_ij), which places a
cusp wherever the strain passes through zero, and a cusp is grid-scale
structure at any resolution. Applied to |S|, the two detectors stay within a
factor of 1.8 of each other at every wavelength; applied to the velocity, they
reproduce their designed separation of 569× at eight points per wavelength.
`reference/CALIBRATION.md` has the response table.

Cost is one detector application per component per active direction, three
times the strain form's, which under `detector = :d8` is three pentadiagonal
line solves per direction instead of one. The strain magnitude itself is still
computed, since `scalar_field(solver, :strain_mag)` exposes it as a diagnostic.

Each call carries a parity, which distinguishes this from three further
`detect_sum!` calls on scalars. A velocity component is odd across the fold
that reverses its axis, and `vel_parity` supplies the sign for it. At a closed
edge that is not a fold, such as a slip wall, neither detector applies a parity
at all, so a wall-normal velocity is differenced as though it were even. That
is a gap against the reference implementation, which carries an antisymmetric
closure for the case, and it bounds what this sensor can do at the planar Noh
wall.
"""
function velocity_mu!(solver, C_mu)
    vel = (solver.u, solver.v, solver.w)
    fill!(solver.sensor, 0)
    for j in 1:3
        detect_sum!(solver.sensor, vel[j], solver, 1; accumulate=true,
                    parity=ntuple(d -> vel_parity(solver, d, j), 3))
    end
    smooth!(solver.sensor, solver)
    rho_sensor!(solver.mu_art, solver, C_mu)
    return solver
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
sensors are built, so the switch evaluates no further derivatives. `I` is a
`CartesianIndex` into the padded arrays.
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
also changes *how much* β\\* a given compression produces. The second half
loses the coordinate folds (see `dilatation_beta!`), so this variant takes the
first half alone.

Cost is one pointwise pass over the interior of `solver.beta_art`, reading
`solver.grad_u`. It builds no sensor, smooths nothing, runs no line solve and
communicates nothing, unlike `:dilatation`.

The switch cannot suppress β\\* at a cusp of |S|. The strain sensor is a
high-pass, so it peaks where |S| passes through zero with a kink, and the
vorticity generally vanishes there too, leaving only ε in the denominator. On a
solenoidal Taylor–Green field this leaves 71 of 32768 points carrying β\\*:
0.6% of the summed total, but the full maximum. `reference/CALIBRATION.md` has the
measurement and why a relative ε does not help.
"""
function gate_beta!(solver)
    o1, o2, o3 = solver.decomp.n_halo_d
    nx, ny, nz = solver.decomp.n_local
    pointwise!(_gate_beta_point!, solver.beta_art, nx, ny, nz,
               solver.beta_art, solver.field_tuples.grad_u, o1, o2, o3)
    return solver
end

@inline function _gate_beta_point!(beta_art, grad_u, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        beta_art[I] *= compression_switch(grad_u, I)
    end
    return nothing
end

"""
    dilatation_beta!(solver, C_beta, gated)

Rebuild `solver.beta_art` from a sensor built on the dilatation, replacing the
strain form. Selected by `ArtParams(beta_sensor = :ungated_dilatation)` for
`gated = false` and `= :dilatation` for `gated = true`.

The sensor is the reduction of h_d² |D_d Δ| over the active directions, for the
detector D named by `ArtParams.detector`, built from the dilatation Δ = ∇·u and
smoothed as the strain sensor is. The ungated form is the one the reference
implementation ships. The gated form multiplies it by `compression_switch`
(Mani, Larsson & Moin, JCP 228, 2009; Kawai, Shankar &
Lele, JCP 229, 2010), which restricts β\\* to compression, where bulk viscosity
has a physical interpretation; `gate_beta!` applies that same switch to the
unchanged strain sensor, which is the other half of the same refinement and a
great deal cheaper.

Either form applies the same detector, weights and smoothing as the strain
sensor and differs only in the field underneath: Δ, the signed sum of the three
diagonal strain components, in place of |S|, the root-sum-square of all nine.
The two therefore respond differently in two ways. Δ carries no off-diagonal
component and is signed, so pure shear leaves it at zero while reaching |S|, and
compression on one axis against expansion on another cancels in Δ while adding
in |S|. That is the sense in which the strain form places β\\* where nothing is
being compressed, and the switch above corrects it. Δ also grows with
the number of axes compressing at once where |S| grows as its square root:
isotropic compression at rate a along n axes gives |Δ| = na against |S| = a√n.
The converging geometries compress along two or three axes at once, so the
second difference is largest there.

Cost is one extra `detect_sum!` and one extra `smooth!` per RHS evaluation
whenever μ\\* still needs the strain sensor. The smoothing is one pass per
active dimension, and a line solve only under `smoother = :compact`. `grad_u`
supplies both Δ and ω, so the switch itself adds no derivatives.

Collective: the dilatation is halo-exchanged and both `detect_sum!` and
`smooth!` run over every active dimension, so every rank must call this.

Unlike the strain form, the gated one does not reproduce bit-for-bit under a
changed decomposition. H(−Δ) is discontinuous at Δ = 0 and, with ε at the
literature value, the ratio has not yet decayed there, so a point whose
dilatation is zero to round-off carries either no β\\* or the full C_β·ρ·sensor
depending on the last bit of a cancelling sum. The measured spread over three
split axes is 2e-7 relative, against 1e-14 for the strain sensor;
`test/mpi_tests.jl` holds the guard. The ungated form has no switch and no such
discontinuity, and reproduces as the strain sensor does.

`solver.sensor` and `solver.tmp_a` are scratch here, as is `solver.tmp_b` in
the gated form, under the invariant recorded on `compute_artificial!`.
"""
function dilatation_beta!(solver, C_beta, gated::Bool)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    if gated
        # Δ into tmp_a, the switch into tmp_b.
        pointwise!(_dilatation_switch_point!, solver.tmp_a, nx, ny, nz,
                   solver.tmp_a, solver.tmp_b, solver.field_tuples.grad_u,
                   o1, o2, o3)
    else
        pointwise!(_dilatation_point!, solver.tmp_a, nx, ny, nz,
                   solver.tmp_a, solver.field_tuples.grad_u, o1, o2, o3)
    end
    exchange_halos!(solver.tmp_a, decomp)
    detect_sum!(solver.sensor, solver.tmp_a, solver, 2)
    smooth!(solver.sensor, solver)      # clobbers tmp_a, which is spent by now
    if !gated
        rho_sensor!(solver.beta_art, solver, C_beta)
        return solver
    end
    pointwise!(_gated_beta_point!, solver.beta_art, nx, ny, nz,
               solver.beta_art, solver.rho, solver.sensor, solver.tmp_b,
               C_beta, o1, o2, o3)
    return solver
end

@inline function _dilatation_switch_point!(tmp_a, tmp_b, grad_u, o1, o2, o3,
                                           i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        tmp_a[I] = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
        tmp_b[I] = compression_switch(grad_u, I)
    end
    return nothing
end

@inline function _dilatation_point!(tmp_a, grad_u, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        tmp_a[I] = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
    end
    return nothing
end

@inline function _gated_beta_point!(beta_art, rho, sensor, switch, C_beta,
                                    o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        beta_art[I] = C_beta * rho[I] * max(sensor[I], 0.0) * switch[I]
    end
    return nothing
end

"""
    compute_artificial!(solver, Q)

Fill `solver.mu_art`, `solver.beta_art` and `solver.kappa_art`, and
`solver.D_art` when the equation set carries more than one species, from the
current primitives and (metric-corrected) velocity gradients. A single-species
run never enters the per-species sweep, so its `D_art` keeps the zeros it was
allocated with. `solver.strain_mag` is written whichever sensors are selected,
since `scalar_field(solver, :strain_mag)` exposes it. `Q` is the padded
conserved array, read only for the internal energy behind the κ\\* sensor.
No-op if disabled.

The caller supplies the primitives and the velocity gradients: `compute_rhs!`
runs `compute_primitives_and_gradients!` on the same `Q` immediately before
this. Collective, since `smooth!` exchanges halos along every active dimension
and both it and `detect_sum!` may run a distributed line solve; every rank must
call it. The `enabled` early return sits above all of that and is safe only
because `ArtParams` is a setup-time constant identical on every rank.

`solver.sensor`, `solver.sensor_sp`, `solver.tmp_a` and `solver.tmp_b` are
scratch here, as is `solver.ring_buf` under `detector = :d8`, and their
contents are dead once this function returns: every value read out of them has
been folded into the coefficient arrays above.
Later phases of the same `compute_rhs!` may therefore reuse them, and
`NSCBCOutflowBC` does, for transverse pressure derivatives. A diagnostic, source
term, or boundary condition that adds a reader of a sensor after this point
invalidates the invariant and must take its own storage. One reader already
exists outside the RHS: `scalar_field(solver, :sensor)` returns `solver.sensor`,
which is why the NSCBC correction borrows `tmp_b` and `sensor_sp` instead.
"""
function compute_artificial!(solver, Q)
    art = solver.art
    if !art.enabled
        return solver   # arrays stay zero from allocation
    end
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
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
    pointwise!(_strain_mag_point!, solver.strain_mag, nx, ny, nz,
               solver.strain_mag, solver.field_tuples.grad_u, o1, o2, o3)

    # Strain sensor: h_d² |D_d S| reduced over directions for the selected
    # detector D, smoothed. One pass serves μ* and β* where both are built from
    # it, which is the shipped configuration; a setting that rebuilds either
    # channel from its own field skips the corresponding write here rather than
    # overwriting it afterwards. The strain magnitude above is computed either
    # way, since `scalar_field(solver, :strain_mag)` exposes it.
    strain_mu = art.mu_sensor === :strain
    strain_beta = art.beta_sensor === :strain || art.beta_sensor === :gated_strain
    if strain_mu || strain_beta
        exchange_halos!(solver.strain_mag, decomp)
        detect_sum!(solver.sensor, solver.strain_mag, solver, 2)
        smooth!(solver.sensor, solver)
        if strain_mu && strain_beta
            pointwise!(_mu_beta_point!, solver.mu_art, nx, ny, nz,
                       solver.mu_art, solver.beta_art, solver.rho,
                       solver.sensor, C_mu, C_beta, o1, o2, o3)
        elseif strain_mu
            rho_sensor!(solver.mu_art, solver, C_mu)
        else
            rho_sensor!(solver.beta_art, solver, C_beta)
        end
    end
    # μ* may instead come from the velocity components, and β* may be keyed on
    # compression, rebuilt from the dilatation, or both. Each rebuild writes one
    # coefficient array and leaves the other untouched.
    art.mu_sensor === :velocity && velocity_mu!(solver, C_mu)
    art.beta_sensor === :gated_strain && gate_beta!(solver)
    art.beta_sensor === :dilatation && dilatation_beta!(solver, C_beta, true)
    art.beta_sensor === :ungated_dilatation && dilatation_beta!(solver, C_beta, false)

    # κ* sensor: Σ_d h_d |D_d e|, smoothed; κ* = C_κ · scale(EOS) · sensor,
    # with the scale ρc/T_ion for the gas models. Internal energy comes straight
    # from Q, so the sensor itself is EOS-agnostic; the scale is an EOS query
    # (see the note in physics.jl on why it is the weaker of the two
    # abstractions and what it is still singular in).
    i_energy = solver.equations.i_energy
    m1, m2, m3 = solver.equations.i_mom
    nxf, nyf, nzf = size(solver.tmp_a)
    pointwise!(_internal_energy_point!, solver.tmp_a, nxf, nyf, nzf,
               solver.tmp_a, Q, solver.rho, m1, m2, m3, i_energy)
    exchange_halos!(solver.tmp_a, decomp)
    detect_sum!(solver.sensor, solver.tmp_a, solver, 1)
    smooth!(solver.sensor, solver)
    pointwise!(_kappa_point!, solver.kappa_art, nx, ny, nz,
               solver.kappa_art, solver.eos, solver.rho, solver.c,
               solver.T_ion, solver.cp_mix, solver.sensor, C_kappa,
               o1, o2, o3)

    # Per-species D*_k sensors: Σ_d h_d |D_d Y_k|, each smoothed;
    # D*_k = C_D c · sensor_k. Costs n_species filter sweeps per RHS; the flux
    # assembly's correction velocity keeps Σ_k J_k = 0 despite unequal D_k.
    # Only meaningful with more than one species.
    if solver.equations.n_species > 1
        for sp in 1:solver.equations.n_species
            detect_sum!(solver.sensor_sp, solver.Y[sp], solver, 1)
            smooth!(solver.sensor_sp, solver)
            pointwise!(_species_diffusivity_point!, solver.sensor_sp,
                       nx, ny, nz, solver.D_art[sp], solver.c,
                       solver.sensor_sp, C_D, o1, o2, o3)
        end
    end
    return solver
end

# Per-point bodies of the sensor-to-coefficient loops above, in order of use.
@inline function _strain_mag_point!(strain_mag, grad_u, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ss = 0.0
        for b in 1:3, a in 1:3
            Sab = 0.5 * (grad_u[a, b][I] + grad_u[b, a][I])
            ss += Sab * Sab
        end
        strain_mag[I] = sqrt(ss)
    end
    return nothing
end

@inline function _mu_beta_point!(mu_art, beta_art, rho, sensor, C_mu, C_beta,
                                 o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρsensor = rho[I] * max(sensor[I], 0.0)
        mu_art[I]   = C_mu   * ρsensor
        beta_art[I] = C_beta * ρsensor
    end
    return nothing
end

@inline function _internal_energy_point!(tmp_a, Q, rho, m1, m2, m3, i_energy,
                                         i, j, k)
    @inbounds begin
        ρ = max(rho[i, j, k], 1e-300)
        ke = 0.5 * (Q[i,j,k,m1]^2 + Q[i,j,k,m2]^2 + Q[i,j,k,m3]^2) / ρ
        tmp_a[i, j, k] = (Q[i, j, k, i_energy] - ke) / ρ
    end
    return nothing
end

@inline function _kappa_point!(kappa_art, eos, rho, c, T_ion, cp_mix, sensor,
                               C_kappa, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        scale = art_conductivity_scale(eos, rho[I], c[I], T_ion[I], cp_mix[I])
        kappa_art[I] = C_kappa * scale * max(sensor[I], 0.0)
    end
    return nothing
end

@inline function _species_diffusivity_point!(D_sp, c, sensor_sp, C_D,
                                             o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        D_sp[I] = C_D * c[I] * max(sensor_sp[I], 0.0)
    end
    return nothing
end
