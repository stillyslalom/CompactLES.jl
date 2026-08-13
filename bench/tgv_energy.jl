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
#   AT RESOLUTION — 128³, 224 ranks over 2 rzhound nodes, t = 10. This is what
#   the 32³ numbers above were waiting for, and it moves three of them:
#
#     config              peak -dKE/dt      mol    mu*   filter    wall
#     art ON,  filter 1   1.2065e-2 @ 8.77  60.4%  2.3%   37.3%   1520 s
#     art OFF, filter 1   1.2248e-2 @ 8.87  62.5%  0      37.5%   1063 s
#     art ON,  filter 0   SolverFailure(:negative_density) at t = 4.66
#
#   Filter dominance does NOT survive resolution: 87% → 83% → 37%. Two coarse
#   points were not a trend.
#
#   The filter is still REQUIRED, and its stabilizing role is decoupled from its
#   energy share. At 37% of the sink, removing it kills the run *earlier* than at
#   32³ (t = 4.66 vs 5.32) through a clean energy blow-up — KE turns upward at
#   t ≈ 4.4 and triples before positivity goes, dt collapsing to 2e-60. TGV is
#   unforced, so that is unambiguously numerical: the filter owns ~100% of the
#   grid-scale sink whatever its share of the total. Art-off/filter-on runs to
#   completion. The filter is necessary and sufficient here; the Cook properties
#   are neither.
#
#   Refinement does not disentangle mu* from the filter — their ratio is
#   invariant across the 4× refinement (5.0/83 = 0.060 at 32³, 2.3/37.3 = 0.062
#   at 128³), both shrinking together as the molecular term takes over. So
#   "refine until the filter lets go, then fit C_mu" does not work. Holding the
#   filter FIXED and fitting against the peak does, because the peak is resolved
#   far below the effect size. Numbers and the open C_mu sweep are in
#   reference/CALIBRATION.md.
#
# Cost: 32³ to t = 10 is ~3.3 min per configuration on a 24-thread desktop, 64³
# ~13 min. On a cluster, 128³ is ~20–25 min per configuration at 224 ranks over
# two nodes (0.10–0.12 s/step, ~11k–13k steps) — the reason this lives in bench/.
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
#   smoother, mu_sensor, beta_sensor, reduction
#             `ArtParams` settings applied to every configuration in the sweep,
#             so a comparison across them is one invocation each rather than one
#             entry each. `mu_sensor` is the setting this case answers: TGV is
#             the only case in the repository where the μ* channel carries a
#             measurable share of the sink, and no case in the 1-D battery of
#             bench/artcal.jl exercises a shear sensor at all. Note that
#             `smoother` still defaults to `:compact`, which the solver no longer
#             ships, because the 128³ numbers above were measured under it. Pass
#             `smoother=gaussian` for a comparison against the shipped
#             configuration.
#   cfl       timestep multiplier (default 0.6). Exposed because the filter's
#             share of the sink depends on it: unrelaxed, the filter removes
#             energy per APPLICATION, so halving the CFL doubles the number of
#             applications covering the same interval and doubles what the
#             filter takes. Sweep it to measure that, not to tune anything.
#   filter_cfl  reference CFL for a full-strength filter pass, 0 (default) for
#             the unrelaxed formulation. Positive makes the filter dissipation
#             a rate rather than a per-application amount, so the sweep above
#             flattens. See `filter_weight`.
#   window    steps either side for every -dKE/dt reported (default 250, i.e. a
#             501-step window, clamped to length(ts)/8). Do not lower it towards
#             1 to "see more detail" — the one-step rate is contaminated by dt
#             jitter at several times the effect size. See `windowed_rate`.
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

"""
−dKE/dt averaged over `w` steps either side of step `i`, rather than the
one-step centred difference this used to print.

The filter removes energy per *application*, not per unit time, so the
instantaneous rate is `(filter loss)/dt + physical` and carries the full
step-to-step `dt` jitter divided into it. Measured at 128³: with the artificial
properties on, the sensor feeds `compute_dt` and `dt` swings ±12% step to step,
which against a filter supplying ~37% of the sink predicts ∓4.4% on the total —
and that is exactly the scatter the sampled table showed. With `art` off, `dt` is
smooth to under a percent and so is the curve. Since `C_mu` is being ranked on
differences of well under 1%, the instantaneous rate cannot be the estimator.

This is the same mechanism as the truncated-final-step artifact below; widening
the window fixes both, but that step is still excluded because one clipped `dt`
inside a window is a bias rather than noise.

A boxcar over a curved peak reads slightly low — ~0.2% at 128³ with the default
501-step window, growing as the window widens. That is common-mode across
configurations run at the same `window=`, so it cancels in a `C_mu` comparison
and only matters when quoting the peak against an external reference. The window
is clamped to `length(ts) ÷ 8`, so hold step counts within ~8x of each other or
the clamp will hand two configurations different windows.
"""
function windowed_rate(ts, kes, i, w)
    lo, hi = max(i - w, 1), min(i + w, length(ts))
    hi > lo || return NaN
    return -(kes[hi] - kes[lo]) / (ts[hi] - ts[lo])
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
                      nmax=typemax(Int), smoother=:compact,
                      cfl=0.6, filter_cfl=0.0,
                      mu_sensor=:strain, beta_sensor=:strain, reduction=:sum)
    γ = 1.4
    c0 = 10.0                      # Ma ≈ 0.1 at |u|max = 1
    p0 = c0^2 / γ
    prob = Problem(eos=single_species(gamma=γ), transport=Transport(mu0=1 / Re),
                   domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                   ic=(x, y, z) -> Prim(
                       u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
                       p=p0 + (1 / 16) * (cos(2x) + cos(2y)) * (cos(2z) + 2),
                       rho=1.0))
    solver, Q = setup(prob, Numerics(n_global=(N, N, N), cfl=cfl,
                                     filter_interval=filter_interval,
                                     filter_cfl=filter_cfl,
                                     art=ArtParams(enabled=art_on, C_mu=C_mu,
                                                   smoother=smoother,
                                                   mu_sensor=mu_sensor,
                                                   beta_sensor=beta_sensor,
                                                   reduction=reduction)))
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
                  progress = 0, sample = 100, nmax = typemax(Int),
                  window = 250, smoother = :compact,
                  cfl = 0.6, filter_cfl = 0.0,
                  mu_sensor = :strain, beta_sensor = :strain, reduction = :sum)

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
    N, tfinal, sample, progress, nmax, window =
        opt.N, opt.tfinal, opt.sample, opt.progress, opt.nmax, opt.window
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
                                      progress=progress, nmax=nmax,
                                      smoother=opt.smoother,
                                      cfl=opt.cfl, filter_cfl=opt.filter_cfl,
                                      mu_sensor=opt.mu_sensor,
                                      beta_sensor=opt.beta_sensor,
                                      reduction=opt.reduction)
            catch err
                # Collective by construction, so every rank lands here together
                # and the sweep stays in step. See the header note.
                err isa SolverFailure || rethrow()
                failure = err
            end
        end
        rank == 0 || continue
        label = @sprintf("art %s, filter_interval %d, C_mu %.4g, %s/%s/%s/%s",
                         cfg.art ? "ON " : "OFF", cfg.filt, cfg.C_mu,
                         opt.smoother, opt.mu_sensor, opt.beta_sensor, opt.reduction)
        if failure !== nothing
            @printf("\n--- %s   (FAILED after %.1f s)\n", label, elapsed)
            @printf("    SolverFailure(:%s) at step %d, t = %.4f, dt = %.3e\n",
                    failure.reason, failure.step, failure.t, failure.dt)
            continue
        end
        solver, ts, kes, samples = result
        # Window for every rate reported below — see `windowed_rate`. Clamped so
        # a short run (a low `nmax`, or a smoke test) still gets a window it can
        # fit rather than one spanning the whole history.
        w = max(min(window, length(ts) ÷ 8), 1)
        # The last step is excluded even so: `run!` truncates it to land exactly
        # on `tfinal`, and one clipped dt inside a window biases rather than
        # jitters. Before this was handled, 128³ reported a spurious 1.4226e-2 at
        # t = 10.00 against a true peak of 1.2065e-2, i.e. +18%, on both filtered
        # configurations. A StepControl retry could clip a step the same way and
        # is not handled here.
        last_full = max(length(ts) - 1, 2)
        rates = [windowed_rate(ts, kes, i, w) for i in 2:last_full]
        imax = argmax(rates) + 1
        @printf("\n--- %s   (%d steps, %.1f s)\n", label, solver.step, elapsed)
        @printf("peak -dKE/dt = %.4e at t = %5.2f   (%d-step window)\n",
                rates[imax-1], ts[imax], 2w + 1)
        # Peak at the last usable step, whatever ended the run. Testing `t`
        # against `tfinal` missed the case `nmax=` creates, where the run
        # stops early and every t is below tfinal.
        imax >= last_full &&
            println("    NOTE: still rising at the last step, so this is not a " *
                    "resolved peak — either the run was cut short of t = 9 " *
                    "(check nmax=) or the configuration is diverging.")
        println("     t     eps_mol     eps_mu*    eps_beta*    -dKE/dt   " *
                "mu*/visc  filter")
        for (t, mol, shear, bulk, idx) in samples
            (idx < 2 || idx >= length(ts)) && continue
            total = windowed_rate(ts, kes, idx, w)
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
            usable[argmin(abs.(getindex.(usable, 5) .- imax))]
        total = windowed_rate(ts, kes, idx, w)
        share(x) = 100 * x / max(abs(total), 1e-300)
        # Nearest sample to the peak, not the peak step itself — diss_split only
        # runs every `sample=` steps.
        @printf("at peak t=%5.2f:  mol %5.1f%%  mu* %5.1f%%  beta* %5.1f%%  FILTER %5.1f%%\n",
                t, share(mol), share(shear), share(bulk),
                share(total - (mol + shear + bulk)))
    end
end

mpi_main(main)
