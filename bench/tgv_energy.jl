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
# Usage — positional grid and end time, then `key=value` options:
#
#   julia --project=. bench/tgv_energy.jl [N] [tfinal] [key=value ...]
#   julia --project=. bench/tgv_energy.jl 64 configs=on:1,on:4,off:1
#   srun -n 224 --cpu-bind=threads julia --project=. -t 1 \
#       -e 'using CompactLES; include(joinpath(pkgdir(CompactLES), "bench",
#           "tgv_energy.jl"))' 128 10.0 configs=on:1:0.002,on:1:0.008 progress=200
#
# Options, all optional:
#   configs   comma-separated <art>:<filter_interval>[:<C_mu>], where <art> is
#             on|off, filter_interval 0 disables filtering, and C_mu defaults to
#             0.002. Both knobs in one entry because they are the pair that has
#             to be calibrated together — see the filter-dominance note above.
#             Default "off:1,on:1", the pair that answers the first-order
#             question: art off and art on, both filtered every step.
#   progress  ProgressLog interval in steps, 0 (default) to disable. Set it for
#             anything long enough to look hung — at 256³ a configuration is
#             ~21,500 steps and the sample table below is the only other output
#             for hours.
#   sample    steps between diss_split samples (default 100). Scale it with the
#             step count or a long run prints hundreds of rows.
#   nmax      step cap per configuration (default none). A sweep that may visit
#             bad configurations wants one: a run that loses positivity does not
#             crash, it grinds — CLAUDE.md, Conventions.
#
# Parsed by `script_args` (src/scriptargs.jl), shared with the cluster scripts;
# the reasoning for ARGS over environment variables is there. An unknown key is
# an error, so a typo costs a message rather than an hour at the default.
#
# Runs under mpiexec unchanged; every reduction here is collective.
#
# A configuration that raises SolverFailure is reported and the sweep continues
# to the next one. That is safe only because the failure is raised off a reduced
# quantity (`max_rate` reduces, `check_step` reads the result), so every rank
# throws at the same step and every rank moves on together. Anything else
# escapes to `mpi_main`, which aborts the job — do not widen that catch.

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
                      filter_interval=1, sample=100, progress=0,
                      nmax=typemax(Int))
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
    callbacks = (record, split)
    if progress > 0
        # `record` runs first in this tuple, so `kes[end]` is already this step's
        # energy and the progress line costs no second Allreduce. The value came
        # out of one, so it is identical on every rank and reading it here breaks
        # no collective-ordering rule.
        callbacks = (callbacks...,
                     ProgressLog(every=progress, tfinal=tfinal, label="KE",
                                 quantity=(s, Q) -> isempty(kes) ? NaN : kes[end]))
    end
    run!(solver, Q; tfinal=tfinal, nmax=nmax, callback=callbacks)
    return solver, ts, kes, samples
end

const DEFAULTS = (N = 32, tfinal = 10.0, configs = "off:1,on:1",
                  progress = 0, sample = 100, nmax = typemax(Int))

function parse_configs(spec)
    configs = NamedTuple{(:art, :filt, :C_mu),Tuple{Bool,Int,Float64}}[]
    for item in split(spec, ',')
        parts = split(strip(item), ':')
        2 <= length(parts) <= 3 ||
            error("bad configs entry '$item', want art:interval[:C_mu]")
        art = parts[1] == "on" ? true :
              parts[1] == "off" ? false : error("art must be on|off, got '$(parts[1])'")
        C_mu = length(parts) == 3 ? parse(Float64, parts[3]) : 0.002
        art || C_mu == 0.002 ||
            error("C_mu given with art off in '$item'; it would have no effect")
        push!(configs, (art=art, filt=parse(Int, parts[2]), C_mu=C_mu))
    end
    return configs
end

function main()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    opt = script_args(ARGS, DEFAULTS; positional = (:N, :tfinal))
    N, tfinal, sample, progress, nmax =
        opt.N, opt.tfinal, opt.sample, opt.progress, opt.nmax
    configs = parse_configs(opt.configs)
    if rank == 0
        @printf("=== Taylor-Green %d^3, Re=1600, tfinal=%.1f, %d rank(s), %d thread(s)\n",
                N, tfinal, MPI.Comm_size(MPI.COMM_WORLD), Threads.nthreads())
        println("    reference peak -dKE/dt = 1.2e-2 at t = 9 (van Rees et al. 2011)")
    end
    for cfg in configs
        result, failure = nothing, nothing
        elapsed = @elapsed begin
            try
                result = taylor_green(N, cfg.art; tfinal=tfinal, C_mu=cfg.C_mu,
                                      filter_interval=cfg.filt, sample=sample,
                                      progress=progress, nmax=nmax)
            catch err
                # Collective by construction, so every rank lands here together
                # and the sweep stays in step. See the header note.
                err isa SolverFailure || rethrow()
                failure = err
            end
        end
        rank == 0 || continue
        label = @sprintf("art %s, filter_interval %d, C_mu %.4g",
                         cfg.art ? "ON " : "OFF", cfg.filt, cfg.C_mu)
        if failure !== nothing
            @printf("\n--- %s   (FAILED after %.1f s)\n", label, elapsed)
            @printf("    SolverFailure(:%s) at step %d, t = %.4f, dt = %.3e\n",
                    failure.reason, failure.step, failure.t, failure.dt)
            continue
        end
        solver, ts, kes, samples = result
        eps = -diff(kes) ./ diff(ts)
        imax = argmax(eps)
        @printf("\n--- %s   (%d steps, %.1f s)\n", label, solver.step, elapsed)
        @printf("peak -dKE/dt = %.4e at t = %5.2f\n", eps[imax], ts[imax+1])
        # Peak at the last recorded step, whatever ended the run. Testing `t`
        # against `tfinal` missed the case `nmax=` creates, where the run
        # stops early and every t is below tfinal.
        imax + 1 >= length(ts) &&
            println("    NOTE: still rising at the last step, so this is not a " *
                    "resolved peak — either the run was cut short of t = 9 " *
                    "(check nmax=) or the configuration is diverging.")
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
        # The calibration readout, one line per configuration. Comparing filter
        # share across a sweep by eye over several hundred table rows is not
        # something anyone does reliably, and filter share is the whole question.
        usable = filter(s -> 2 <= s[5] < length(ts), samples)
        isempty(usable) && continue
        t, mol, shear, bulk, idx =
            usable[argmin(abs.(getindex.(usable, 5) .- (imax + 1)))]
        total = -(kes[idx+1] - kes[idx-1]) / (ts[idx+1] - ts[idx-1])
        share(x) = 100 * x / max(abs(total), 1e-300)
        # Nearest sample to the peak, not the peak step itself — diss_split only
        # runs every `sample=` steps.
        @printf("at peak t=%5.2f:  mol %5.1f%%  mu* %5.1f%%  beta* %5.1f%%  FILTER %5.1f%%\n",
                t, share(mol), share(shear), share(bulk),
                share(total - (mol + shear + bulk)))
    end
end

mpi_main(main)
