# Taylor–Green vortex with the kinetic-energy budget split by which mechanism
# removes the energy. Built for the C_mu question: `test/convergence.jl` already
# runs TGV, but with `art=ArtParams(enabled=false)`, so it tests the physical
# viscous term and says nothing about the artificial one.
#
# The split is the point. -dKE/dt is TOTAL energy loss; the viscous stress
# accounts for only part of it, and the residual is the compact filter (plus
# numerical loss and, early on, a small pressure-dilatation exchange with
# internal energy that shows up as a slightly negative residual while the flow
# is still smooth). Three channels, and only two of them have a coefficient.
#
# WHAT THIS MEASURED, so the next person does not have to rediscover it:
#
#   32³, Re=1600, filter_interval=1, C_mu=0.002 default, at the dissipation peak
#   (t ≈ 6.3): molecular 12%, artificial shear 5%, artificial bulk ~0%,
#   FILTER ≈ 83%. Peak -dKE/dt 1.46e-2 at t = 6.5 against the reference 1.2e-2
#   at t = 9 — over-dissipating, too early, and the excess is mostly filter.
#
#   β* is four orders below μ* despite C_beta = 1.0: at Ma 0.1 there is almost
#   no dilatation for it to act on. This is not a bug, it is the case.
#
#   Filter sweep at 32³, art on: interval 1 gives the above; interval 4 diverges
#   (9.5e-2 still rising at t = 10); interval 0 fails outright with
#   SolverFailure(:negative_density) at t = 5.32. The compact filter is the
#   primary STABILIZER here, not just a smoother — the Cook properties alone do
#   not hold this case together at this resolution.
#
#   Consequence for calibration: C_mu cannot be fitted against this dissipation
#   curve while the filter owns ~83% of the sink, and the filter cannot be
#   switched off to isolate it. They set the subgrid dissipation jointly.
#
#   Resolution caveat, and it is the important one: 32³ is coarse for Re = 1600
#   (DNS wants ~256³; 64³ is the usual coarse-LES point). Across 16³ → 32³ the
#   artificial share falls (50% → 28% of the viscous budget) but the filter share
#   does NOT (≈87% → 83%). Two coarse points is not a trend. Filter dominance is
#   established at 32³ and UNESTABLISHED at a resolution worth quoting.
#
# Cost, measured on a 24-thread desktop: 32³ to t = 10 is ~3.3 min per
# configuration. 64³ is 8× the points and 2× the steps, so ~13 min per
# configuration — the reason this lives in bench/ and wants a cluster.
#
# Usage:
#   julia --project=. bench/tgv_energy.jl [N] [tfinal]
#   CL_TGV_CONFIGS="on:1,on:4,off:1" julia --project=. bench/tgv_energy.jl 64
#
# CL_TGV_CONFIGS is a comma-separated list of <art>:<filter_interval>, where
# <art> is on|off and filter_interval 0 disables filtering. Default is the pair
# that answers the first-order question: art off and art on, both filtered every
# step. Runs under mpiexec unchanged; every reduction here is collective.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

"""
Volume-averaged resolved dissipation split into molecular, artificial-shear and
artificial-bulk contributions. Same tensor contraction as `dissipation_rate` in
`diagnostics.jl`, which sums them; here they are kept apart.

Accumulates scalars rather than filling fields: TGV is a uniform periodic
Cartesian box, so `volume_integral`'s quadrature weights are all equal and a
plain sum is the same number without borrowing scratch arrays.
"""
function diss_split(solver, Q)
    CL.compute_primitives_and_gradients!(solver, Q)
    CL.compute_artificial!(solver, Q)
    o1, o2, o3 = solver.decomp.n_halo_d
    nx, ny, nz = solver.decomp.n_local
    g = solver.grad_u
    mu0 = solver.transport.mu0
    s_mol, s_shear, s_bulk, s_rho = 0.0, 0.0, 0.0, 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        divu = g[1, 1][I] + g[2, 2][I] + g[3, 3][I]
        mu_a = solver.mu_art[I]
        beta_a = solver.beta_art[I]
        for b in 1:3, a in 1:3
            S2 = g[a, b][I] + g[b, a][I]
            trace = a == b ? divu : 0.0
            s_mol += (mu0 * S2 - 2mu0 / 3 * trace) * g[a, b][I]
            s_shear += (mu_a * S2 - 2mu_a / 3 * trace) * g[a, b][I]
            s_bulk += beta_a * trace * g[a, b][I]
        end
        s_rho += solver.rho[I]
    end
    v = MPI.Allreduce([s_mol, s_shear, s_bulk, s_rho], +, solver.decomp.comm)
    return (v[1] / v[4], v[2] / v[4], v[3] / v[4])
end

"Kinetic energy per unit volume, globally reduced."
function kinetic_energy(solver, Q, cellvol)
    ke = 0.0
    m1, m2, m3 = solver.equations.i_mom
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
        i in 1:solver.decomp.n_local[1]
        I = gidx(solver, i, j, k)
        ke += 0.5 * (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / Q[I, 1]
    end
    return MPI.Allreduce(ke * cellvol, +, solver.decomp.comm) / (2π)^3
end

function taylor_green(N, art_on; tfinal=10.0, Re=1600.0, C_mu=0.002,
                      filter_interval=1, sample=100)
    γ = 1.4
    c0 = 10.0                      # Ma ≈ 0.1 at |u|max = 1
    p0 = c0^2 / γ
    prob = Problem(eos=single_species(gamma=γ), transport=Transport(mu0=1 / Re),
                   domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                   ic=(x, y, z) -> Prim(
                       u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
                       p=p0 + (1 / 16) * (cos(2x) + cos(2y)) * (cos(2z) + 2),
                       rho=1.0))
    solver, Q = setup(prob, Numerics(n_global=(N, N, N), cfl=0.6,
                                     filter_interval=filter_interval,
                                     art=ArtParams(enabled=art_on, C_mu=C_mu)))
    cellvol = prod(solver.h)
    ts = Float64[]
    kes = Float64[]
    samples = Tuple{Float64,Float64,Float64,Float64,Int}[]
    record = Callback(EveryStep(1), (s, Q) -> begin
        push!(ts, s.t)
        push!(kes, kinetic_energy(s, Q, cellvol))
        nothing
    end)
    # diss_split costs an extra gradient pass, so it is sampled, not stepwise.
    split = Callback(EveryStep(sample), (s, Q) -> begin
        mol, shear, bulk = diss_split(s, Q)
        push!(samples, (s.t, mol, shear, bulk, length(ts)))
        nothing
    end)
    run!(solver, Q; tfinal=tfinal, callback=(record, split))
    return solver, ts, kes, samples
end

function parse_configs()
    spec = get(ENV, "CL_TGV_CONFIGS", "off:1,on:1")
    configs = Tuple{Bool,Int}[]
    for item in split(spec, ',')
        parts = split(strip(item), ':')
        length(parts) == 2 || error("bad CL_TGV_CONFIGS entry '$item', want art:interval")
        art = parts[1] == "on" ? true :
              parts[1] == "off" ? false : error("art must be on|off, got '$(parts[1])'")
        push!(configs, (art, parse(Int, parts[2])))
    end
    return configs
end

function main()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
    tfinal = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 10.0
    if rank == 0
        @printf("=== Taylor-Green %d^3, Re=1600, tfinal=%.1f, %d rank(s), %d thread(s)\n",
                N, tfinal, MPI.Comm_size(MPI.COMM_WORLD), Threads.nthreads())
        println("    reference peak -dKE/dt = 1.2e-2 at t = 9 (van Rees et al. 2011)")
    end
    for (art, filt) in parse_configs()
        elapsed = @elapsed begin
            solver, ts, kes, samples = taylor_green(N, art; tfinal=tfinal,
                                                    filter_interval=filt)
        end
        rank == 0 || continue
        eps = -diff(kes) ./ diff(ts)
        imax = argmax(eps)
        @printf("\n--- art %s, filter_interval %d   (%d steps, %.1f s)\n",
                art ? "ON " : "OFF", filt, solver.step, elapsed)
        @printf("peak -dKE/dt = %.4e at t = %5.2f\n", eps[imax], ts[imax+1])
        ts[imax+1] >= tfinal - 1e-9 &&
            println("    NOTE: still rising at the last step, so this is not a " *
                    "resolved peak — either the run was cut short of t = 9 or " *
                    "the configuration is diverging.")
        println("     t     eps_mol     eps_mu*    eps_beta*    -dKE/dt   " *
                "mu*/visc  filter")
        for (t, mol, shear, bulk, idx) in samples
            (idx < 2 || idx >= length(ts)) && continue
            total = -(kes[idx+1] - kes[idx-1]) / (ts[idx+1] - ts[idx-1])
            visc = mol + shear + bulk
            @printf("  %5.2f  %.4e  %.4e  %.4e  %.4e  %6.1f%%  %6.1f%%\n",
                    t, mol, shear, bulk, total,
                    100 * shear / max(visc, 1e-300),
                    100 * (total - visc) / max(abs(total), 1e-300))
        end
    end
end

main()
