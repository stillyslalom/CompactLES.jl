# Cook-style artificial fluid properties (Cook, Phys. Fluids 2007), extended
# with the artificial species diffusivity for multicomponent mixing.
#
# Sensors use *undivided* explicit fourth differences δ⁴ (1, −4, 6, −4, 1) in
# the computational indices, so the formal grid-spacing powers reduce to
# per-dimension weights: h² for μ*, β* (from |∇⁴S| Δ⁶), h for κ* (|∇⁴e| Δ⁵)
# and for D* (|∇⁴Y| Δ⁵). The Gaussian test filter is approximated by one
# compact-filter pass. Each species carries its own sensor and its own D*_k;
# Σ_k J_k = 0 is then restored by the correction velocity in the flux assembly
# (rhs.jl) rather than by giving every species the same diffusivity. Indices
# are clamped at closed physical edges; halos cover rank boundaries. On
# curvilinear grids the sensors are computed in computational space — a
# grid-based rather than strictly physical-space regularization, standard
# practice and consistent with resolving power following the mesh.

"""
    ArtParams(; enabled=true, C_mu=0.002, C_beta=1.0,
              C_kappa=0.01, C_D=0.01)

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
        @threaded nx*ny*nz for k in 1:nz
            @inbounds for j in 1:ny, i in 1:nx
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

"One directional compact-filter pass per active dimension as a
Gaussian-filter proxy (even-parity axis fill along r when applicable)."
function smooth!(f, solver)
    for d in 1:3
        solver.decomp.active[d] || continue
        # Only dimension d: the filter about to run is a 1-D stencil along d,
        # so the other two dimensions' halos are dead. This used to call
        # exchange_halos!, refreshing all three every time round the loop --
        # three times the traffic it needs in a 3-D run, on a routine that
        # runs once per sensor per species per RHS.
        exchange_dim!(f, solver.decomp, d)
        filt_along!(solver.tmp_a, f, solver, d, 1)
        copy_interior!(f, solver.tmp_a, solver.decomp)
    end
    return f
end

"""
    compute_artificial!(solver, Q)

Fill solver.mu_art, solver.beta_art, solver.kappa_art and solver.D_art from
the current primitives and (metric-corrected) velocity gradients. No-op if
disabled.
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

    # Strain-rate magnitude |S| = sqrt(S_ij S_ij) in the interior (physical
    # components — the metric corrections are already in grad_u).
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ss = 0.0
            for b in 1:3, a in 1:3
                Sab = 0.5 * (grad_u[a, b][I] + grad_u[b, a][I])
                ss += Sab * Sab
            end
            solver.strain_mag[I] = sqrt(ss)
        end
    end

    # μ*, β* sensor: Σ_d h_d² |δ⁴_d S|, smoothed.
    exchange_halos!(solver.strain_mag, decomp)
    delta4_sum!(solver.sensor, solver.strain_mag, solver, 2)
    smooth!(solver.sensor, solver)
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            ρsensor = solver.rho[I] * max(solver.sensor[I], 0.0)
            solver.mu_art[I]   = art.C_mu   * ρsensor
            solver.beta_art[I] = art.C_beta * ρsensor
        end
    end

    # κ* sensor: Σ_d h_d |δ⁴_d e|, smoothed; κ* = C_κ · scale(EOS) · sensor,
    # with the scale ρc/T_ion for the gas models. Internal energy comes straight
    # from Q, so the sensor itself is EOS-agnostic; the scale is an EOS query
    # (see the note in physics.jl on why it is the weaker of the two
    # abstractions and what it is still singular in).
    i_energy = solver.equations.i_energy
    m1, m2, m3 = solver.equations.i_mom
    @threaded length(solver.tmp_a) for k in 1:size(solver.tmp_a, 3)
        @inbounds for j in 1:size(solver.tmp_a, 2), i in 1:size(solver.tmp_a, 1)
            ρ = max(solver.rho[i, j, k], 1e-300)
            ke = 0.5 * (Q[i,j,k,m1]^2 + Q[i,j,k,m2]^2 + Q[i,j,k,m3]^2) / ρ
            solver.tmp_a[i, j, k] = (Q[i, j, k, i_energy] - ke) / ρ
        end
    end
    exchange_halos!(solver.tmp_a, decomp)
    delta4_sum!(solver.sensor, solver.tmp_a, solver, 1)
    smooth!(solver.sensor, solver)
    eos = solver.eos
    @threaded nx*ny*nz for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            scale = art_conductivity_scale(eos, solver.rho[I], solver.c[I],
                                           solver.T_ion[I], solver.cp_mix[I])
            solver.kappa_art[I] = art.C_kappa * scale * max(solver.sensor[I], 0.0)
        end
    end

    # Per-species D*_k sensors: Σ_d h_d |δ⁴_d Y_k|, each smoothed;
    # D*_k = C_D c · sensor_k. Costs n_species filter sweeps per RHS; the flux
    # assembly's correction velocity keeps Σ_k J_k = 0 despite unequal D_k.
    # Only meaningful with more than one species.
    if solver.equations.n_species > 1
        for sp in 1:solver.equations.n_species
            delta4_sum!(solver.sensor_sp, solver.Y[sp], solver, 1)
            smooth!(solver.sensor_sp, solver)
            @threaded nx*ny*nz for k in 1:nz
                @inbounds for j in 1:ny, i in 1:nx
                    I = CartesianIndex(i + o1, j + o2, k + o3)
                    solver.D_art[sp][I] = art.C_D * solver.c[I] * max(solver.sensor_sp[I], 0.0)
                end
            end
        end
    end
    return solver
end
