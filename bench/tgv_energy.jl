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
# --- What this measured, so it is not rediscovered ---------------------------
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
#   primary stabilizer here, beyond its smoothing role — the Cook properties do
#   not hold this case together at this resolution.
#
#   At resolution — 128³, 224 ranks over 2 rzhound nodes, t = 10. The 32³ trends
#   above change in three ways:
#
#     config              peak -dKE/dt      mol    mu*   filter    wall
#     art ON,  filter 1   1.2065e-2 @ 8.77  60.4%  2.3%   37.3%   1520 s
#     art OFF, filter 1   1.2248e-2 @ 8.87  62.5%  0      37.5%   1063 s
#     art ON,  filter 0   SolverFailure(:negative_density) at t = 4.66
#
#   Filter dominance does not survive resolution: 87% → 83% → 37% → 12.8% at
#   16³, 32³, 128³, 256³. Two coarse points were not a trend, and the fall
#   continues at 256³.
#
#   The filter is still required, and its stabilizing role is decoupled from its
#   energy share. At 37% of the sink, removing it kills the run *earlier* than at
#   32³ (t = 4.66 vs 5.32) through a clean energy blow-up — KE turns upward at
#   t ≈ 4.4 and triples before positivity goes, dt collapsing to 2e-60. TGV is
#   unforced, so that is unambiguously numerical: the filter contributes ~100% of
#   grid-scale sink whatever its share of the total. Art-off/filter-on runs to
#   completion. The filter is necessary and sufficient here; the Cook properties
#   are neither.
#
#   Refinement does not disentangle mu* from the filter — their ratio is
#   invariant across an 8× refinement in linear resolution (5.0/83 = 0.060 at
#   32³, 2.3/37.3 = 0.062 at 128³, 0.8/12.8 = 0.0625 at 256³), both shrinking
#   together as the molecular term takes over. So "refine until the filter lets
#   go, then fit C_mu" does not work. Holding the filter FIXED and fitting
#   against the peak does, because the peak is resolved far below the effect
#   size. Numbers and the open C_mu sweep are in reference/CALIBRATION.md.
#
#   256³, art ON, filter 1, 4 MI300A APUs on rzadams (backend=amdgpu), t = 10:
#   peak -dKE/dt 1.3043e-2 @ 8.84, mol 86.4%, mu* 0.8%, beta* 0.0%, filter 12.8%.
#   The peak sits ~4% above the rounded 1.2e-2 banner, within the well-resolved
#   DNS range (~1.25-1.28e-2 near t = 9), with its time converged toward 9; the
#   coarse-grid early overprediction is gone. Only the art-ON leg was run, so the
#   filter necessary-and-sufficient test (art OFF completes, filter 0 fails) is
#   measured at 128³ but not yet at 256³.
#
# Cost: 32³ to t = 10 is ~3.3 min per configuration on a 24-thread desktop, 64³
# ~13 min. On a cluster, 128³ is ~20–25 min per configuration at 224 ranks over
# two nodes (0.10–0.12 s/step, ~11k–13k steps) — the reason this lives in bench/.
# 256³ art on over 4 MI300A APUs on rzadams at -t 1 is ~3.7 h (24.5k steps,
# 0.35 s/step baseline inflated ~1.5× by device stall episodes; AMR_GPU.md).
#
#   Precision measurement (24-thread desktop, art off, filter every step):
#
#     N   type       peak -dKE/dt @ t    solver wall   footprint    ρ̄ drift
#     32  Float64    1.4714e-2 @ 6.86      149.77 s      37.3 MiB   1.67e-13
#     32  Float32    1.4714e-2 @ 6.86      141.78 s      18.7 MiB   7.62e-5
#     64  Float64    1.2471e-2 @ 8.93      621.41 s     219.5 MiB   7.49e-13
#     64  Float32    1.2472e-2 @ 8.93      563.52 s     109.8 MiB   1.39e-4
#
# At 64³ the published reference is 1.2e-2 near t=9: both precisions are about
# 4% high and land at the same time. Float32 halves resident solver/state/RK
# memory but buys only 1.10x CPU throughput here; its conservation drift is a
# real 1e-4 tradeoff, not hidden by the Float64 diagnostic reductions.
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
#             bad configurations should set one: a run that loses positivity does
#             crash, it grinds — CLAUDE.md, Conventions.
#   smoother, mu_sensor, beta_sensor, reduction
#             `ArtParams` settings applied to every configuration in the sweep,
#             so a comparison across them is one invocation each rather than one
#             entry each. `mu_sensor` is the setting this case answers: TGV is
#             the only case in the repository where the μ* channel carries a
#             measurable share of the sink, and no case in the 1-D battery of
#             bench/artcal.jl exercises a shear sensor at all. Note that
#             `smoother` still defaults to `:compact`, which is no longer the
#             solver default, because the 128³ numbers above were measured under it. Pass
#             `smoother=gaussian` for a comparison against the default
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
#   precision Float storage and arithmetic to measure: float64 (default),
#             float32, or both. Diagnostics intentionally accumulate in
#             Float64 in either mode; this option changes the solver state,
#             schemes, EOS, transport, artificial controls, and RK workspace.
#   backend   cpu (default), amdgpu, or cuda: where the solver lives. A device
#             backend needs an environment carrying the device package (see
#             bench/device_bringup.jl); the solver and state are
#             device-resident and the energy diagnostics read a host copy
#             of the state, downloaded per callback — the
#             documented I/O-gathers-to-host path, excluded from solver wall
#             time by the same accounting that excludes callbacks on the CPU.
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
import CompactLES: AbstractBackend, PatchSolver
using Printf

const CL = CompactLES
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

# Host view of one field: the identity on host storage, a download on device
# storage. Diagnostics-only; nothing in the stepped path calls it.
_host(a::Array) = a
_host(a) = Array(a)

function make_backend(spec::AbstractString)
    spec == "cpu" && return CPUBackend()
    if spec == "amdgpu"
        @eval using AMDGPU
        @eval AMDGPU.functional() || error("AMDGPU is not functional here")
        return DeviceBackend(@eval ROCBackend())
    elseif spec == "cuda"
        @eval using CUDA
        @eval CUDA.functional() || error("CUDA is not functional here")
        return DeviceBackend(@eval CUDABackend())
    end
    error("backend must be cpu, amdgpu, or cuda; got '$spec'")
end

"""
Volume-averaged resolved dissipation split into molecular, artificial-shear and
artificial-bulk contributions. Same tensor contraction as `dissipation_rate` in
`diagnostics.jl`, which sums them; here they are kept apart.

Accumulates scalars rather than filling fields. On the single periodic grid
every quadrature weight is the cell volume; on a refined run the sum runs
over every held patch with that patch's cell volume, its own edge weights,
and the covered mask (`uncovered_fraction`), so a coarse node under the
refined level is counted once.
"""
function diss_split(solver, states)
    s_mol, s_shear, s_bulk, s_rho = 0.0, 0.0, 0.0, 0.0
    for (ps, Q) in CL.eachpatch(solver, _as_vector(states))
        CL.compute_primitives_and_gradients!(ps, Q)
        CL.compute_artificial!(ps, Q)
        o1, o2, o3 = ps.decomp.n_halo_d
        nx, ny, nz = ps.decomp.n_local
        # `_host` aliases host arrays and downloads device ones, so the same
        # scalar accumulation serves a device-resident solver (sampled, so the
        # transfer runs every `sample=` steps, not every step).
        g = [_host(ps.grad_u[a, b]) for a in 1:3, b in 1:3]
        mu_art = _host(ps.mu_art)
        beta_art = _host(ps.beta_art)
        rho = _host(ps.rho)
        mu0 = solver.transport.mu0
        dv = prod(ps.h)
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            w = _node_weight(ps, i, j, k, I) * dv
            divu = g[1, 1][I] + g[2, 2][I] + g[3, 3][I]
            mu_a = mu_art[I]
            beta_a = beta_art[I]
            for b in 1:3, a in 1:3
                S2 = g[a, b][I] + g[b, a][I]
                trace = a == b ? divu : 0.0
                s_mol += w * (mu0 * S2 - 2mu0 / 3 * trace) * g[a, b][I]
                s_shear += w * (mu_a * S2 - 2mu_a / 3 * trace) * g[a, b][I]
                s_bulk += w * beta_a * trace * g[a, b][I]
            end
            s_rho += w * rho[I]
        end
    end
    v = MPI.Allreduce([s_mol, s_shear, s_bulk, s_rho], +, solver.comm)
    return (v[1] / v[4], v[2] / v[4], v[3] / v[4])
end

# The state vector a refined run carries, or the one-element vector of a
# single-patch run's state, so one patch loop serves both.
_as_vector(states::Vector) = states
_as_vector(Q) = [Q]

# One node's composite quadrature weight relative to the patch cell volume:
# the edge weights of the patch's own quadrature (one everywhere on the
# periodic root, a half on a refined patch's boundary planes) times the
# fraction of the node's cell no child level covers.
@inline function _node_weight(ps, i, j, k, I)
    w = CL.quad_weight(ps, 1, i) * CL.quad_weight(ps, 2, j) *
        CL.quad_weight(ps, 3, k)
    m = ps.covered[I]
    m == 0 || (w *= CL.uncovered_fraction(m))
    return w
end

"""
−dKE/dt averaged over `w` steps either side of step `i`, rather than the
one-step centred difference this used to print.

The filter removes energy per *application*, not per unit time, so the
instantaneous rate is `(filter loss)/dt + physical` and carries the full
step-to-step `dt` jitter divided into it. Measured at 128³: with the artificial
properties on, the sensor feeds `compute_dt` and `dt` swings ±12% step to step,
which against a filter supplying ~37% of the sink predicts ∓4.4% on the total —
matching the scatter in the sampled table. With `art` off, `dt` is
smooth to under a percent and so is the curve. Since `C_mu` is being ranked on
differences of well under 1%, the instantaneous rate cannot be the estimator.

This is the same mechanism as the truncated-final-step artifact below; widening
the window fixes both, but that step is still excluded because one clipped `dt`
inside a window is a bias rather than noise.

A boxcar over a curved peak reads slightly low — ~0.2% at 128³ with the default
501-step window, growing as the window widens. That is common-mode across
configurations run at the same `window=`, so it cancels in a `C_mu` comparison
and affects only comparisons with an external peak reference. The window
is clamped to `length(ts) ÷ 8`, so hold step counts within ~8x of each other or
the clamp will hand two configurations different windows.
"""
function windowed_rate(ts, kes, i, w)
    lo, hi = max(i - w, 1), min(i + w, length(ts))
    hi > lo || return NaN
    return -(kes[hi] - kes[lo]) / (ts[hi] - ts[lo])
end

"Kinetic energy per unit volume, globally reduced. `hosts` holds one host
array per held patch, aligned with `solver.patches`; on a refined run the
sum is the composite quadrature under the covered masks."
function kinetic_energy(solver, hosts::Vector)
    ke = 0.0
    m1, m2, m3 = solver.equations.i_mom
    for (n, p) in enumerate(solver.patches)
        ps = PatchSolver(solver, p)
        Q = hosts[n]
        dv = prod(ps.h)
        for k in 1:ps.decomp.n_local[3], j in 1:ps.decomp.n_local[2],
            i in 1:ps.decomp.n_local[1]
            I = gidx(ps, i, j, k)
            ke += _node_weight(ps, i, j, k, I) * dv *
                  0.5 * (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / Q[I, 1]
        end
    end
    return MPI.Allreduce(ke, +, solver.comm) / (2π)^3
end

"Volume-averaged mixture density, accumulated and reduced in Float64."
function mean_density(solver, hosts::Vector)
    mass = 0.0
    for (n, p) in enumerate(solver.patches)
        ps = PatchSolver(solver, p)
        Q = hosts[n]
        dv = Float64(prod(ps.h))
        for k in 1:ps.decomp.n_local[3], j in 1:ps.decomp.n_local[2],
            i in 1:ps.decomp.n_local[1]
            I = gidx(ps, i, j, k)
            w = _node_weight(ps, i, j, k, I) * dv
            for sp in 1:solver.equations.n_species
                mass += w * Float64(Q[I, sp])
            end
        end
    end
    return MPI.Allreduce(mass, +, solver.comm) / (2π)^3
end

function taylor_green(N, art_on; tfinal=10.0, Re=1600.0, C_mu=0.002,
                      filter_interval=1, sample=100, progress=0,
                      nmax=typemax(Int), smoother=:compact,
                      cfl=0.6, filter_cfl=0.0,
                      mu_sensor=:strain, beta_sensor=:strain, reduction=:sum,
                      T::Type{<:AbstractFloat}=Float64,
                      backend::AbstractBackend=CPUBackend(),
                      refine::Int=0, tile::Int=0, subcycle::Bool=false)
    # A refined run carries a centered cube of `refine` root nodes on one
    # level below the root, host-only: its energy history is the composite
    # quadrature and is the covered-mask check against the single-level
    # history (reference/AMR_GPU.md, diagnostics).
    refine == 0 || backend isa CPUBackend ||
        error("a refined run takes the host backend")
    region = refine == 0 ? nothing :
             BlockRegion(ntuple(_ -> (N - refine) ÷ 2, 3), ntuple(_ -> refine, 3))
    γ = T(1.4)
    c0 = T(10)                     # Ma ≈ 0.1 at |u|max = 1
    p0 = c0^2 / γ
    prob = Problem(eos=IdealSpecies("gas"; R=one(T), gamma=γ),
                   transport=Transport{T}(mu0=one(T) / T(Re)),
                   domain=((zero(T), T(2π)), (zero(T), T(2π)),
                           (zero(T), T(2π))), bcs=per3,
                   ic=(x, y, z) -> Prim(
                       u=(sin(x) * cos(y) * cos(z),
                          -cos(x) * sin(y) * cos(z), zero(T)),
                       p=p0 + one(T) / T(16) *
                           (cos(T(2) * x) + cos(T(2) * y)) *
                           (cos(T(2) * z) + T(2)),
                       rho=one(T)))
    solver = Q = nothing
    setup_elapsed = @elapsed begin
        solver, Q = setup(
            prob,
            Numerics(n_global=(N, N, N), cfl=cfl,
                     filter_interval=filter_interval, filter_cfl=filter_cfl,
                     deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T),
                     art=ArtParams{T}(enabled=art_on, C_mu=T(C_mu),
                                      smoother=smoother,
                                      mu_sensor=mu_sensor,
                                      beta_sensor=beta_sensor,
                                      reduction=reduction),
                     backend=backend, refine=region, tile=tile,
                     subcycle=subcycle))
    end
    workspace = Workspace(Q)
    footprint = (solver=Base.summarysize(solver), state=Base.summarysize(Q),
                 workspace=Base.summarysize(workspace))
    # Device-resident state reads through one reused host buffer: the scalar
    # energy/mass loops index this, not Q. On the CPU it aliases Q and
    # the download is a no-op. A refined run is host-only and hands the
    # loops its patches' arrays directly.
    devres = refine == 0 && !(parent(Q) isa Array)
    Qhost = devres ? Array(parent(Q)) : nothing
    sync_host!(Q) = devres ? (copyto!(Qhost, parent(Q)); [Qhost]) :
                    [parent(q) for q in _as_vector(Q)]
    mass0 = mean_density(solver, sync_host!(Q))

    # Compile the precision-specific hot path before timing. The warm state is
    # disposable; the measured state and solver time remain at t = 0.
    Qwarm = refine == 0 ? copy(Q) : [copy(q) for q in Q]
    warmspace = Workspace(Qwarm)
    dtwarm = compute_dt(solver, Qwarm)
    step!(solver, Qwarm, warmspace, dtwarm)
    filter_interval > 0 && CL.filter_state!(solver, Qwarm)
    kinetic_energy(solver, sync_host!(Qwarm))
    Qwarm = warmspace = nothing
    GC.gc()

    ts = Float64[]
    kes = Float64[]
    samples = Tuple{Float64,Float64,Float64,Float64,Int}[]
    record = Callback(EveryStep(1), (s, Q) -> begin
        push!(ts, s.t)
        push!(kes, kinetic_energy(s, sync_host!(Q)))
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
    run_elapsed = @elapsed begin
        run!(solver, Q, workspace; tfinal=T(tfinal), nmax=nmax,
             callback=callbacks)
    end
    mass1 = mean_density(solver, sync_host!(Q))
    return (; solver, Q, workspace, ts, kes, samples, setup_elapsed,
            run_elapsed, footprint, mass0, mass1)
end

const DEFAULTS = (N = 32, tfinal = 10.0, configs = "off:1,on:1",
                  progress = 0, sample = 100, nmax = typemax(Int),
                  window = 250, smoother = :compact,
                  cfl = 0.6, filter_cfl = 0.0,
                  mu_sensor = :strain, beta_sensor = :strain, reduction = :sum,
                  precision = "float64", backend = "cpu",
                  refine = 0, tile = 0, subcycle = false)

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

function parse_precisions(spec)
    lowercase(spec) == "both" && return (Float64, Float32)
    out = DataType[]
    for item in split(lowercase(spec), ',')
        p = strip(item)
        push!(out, p == "float64" ? Float64 :
                   p == "float32" ? Float32 :
                   error("precision must be float64, float32, or both; got '$p'"))
    end
    isempty(out) && error("precision list must not be empty")
    return Tuple(out)
end

function main(opt, backend)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    N, tfinal, sample, progress, nmax, window =
        opt.N, opt.tfinal, opt.sample, opt.progress, opt.nmax, opt.window
    configs = parse_configs(opt.configs)
    precisions = parse_precisions(opt.precision)
    summaries = NamedTuple[]
    if rank == 0
        @printf("=== Taylor-Green %d^3, Re=1600, tfinal=%.1f, %d rank(s), ",
                N, tfinal, MPI.Comm_size(MPI.COMM_WORLD))
        @printf("%d thread(s), backend %s\n", Threads.nthreads(), opt.backend)
        println("    reference peak -dKE/dt = 1.2e-2 at t = 9 (van Rees et al. 2011)")
    end
    for T in precisions, cfg in configs
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
                                      reduction=opt.reduction, T=T,
                                      backend=backend, refine=opt.refine,
                                      tile=opt.tile, subcycle=opt.subcycle)
            catch err
                # Collective by construction, so every rank lands here together
                # and the sweep stays in step. See the header note.
                err isa SolverFailure || rethrow()
                failure = err
            end
        end
        rank == 0 || continue
        label = @sprintf("%s, art %s, filter_interval %d, C_mu %.4g, %s/%s/%s/%s",
                         T, cfg.art ? "ON " : "OFF", cfg.filt, cfg.C_mu,
                         opt.smoother, opt.mu_sensor, opt.beta_sensor, opt.reduction)
        opt.refine == 0 ||
            (label *= @sprintf(", refined %d^3 (tile %d, %s)", opt.refine, opt.tile,
                               opt.subcycle ? "subcycled" : "unsubcycled"))
        if failure !== nothing
            @printf("\n--- %s   (FAILED after %.1f s)\n", label, elapsed)
            @printf("    SolverFailure(:%s) at step %d, t = %.4f, dt = %.3e\n",
                    failure.reason, failure.step, failure.t, failure.dt)
            continue
        end
        solver, ts, kes, samples =
            result.solver, result.ts, result.kes, result.samples
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
        mem = result.footprint
        total_mem = mem.solver + mem.state + mem.workspace
        throughput = N^3 * solver.step /
                     max(solver.wall_total, eps(Float64)) / 1e6
        @printf("setup %.2f s; run %.2f s; solver %.2f s; diagnostics/loop %.2f s\n",
                result.setup_elapsed, result.run_elapsed, solver.wall_total,
                max(result.run_elapsed - solver.wall_total, 0.0))
        @printf("footprint %.1f MiB = solver %.1f + state %.1f + RK workspace %.1f",
                total_mem / 2.0^20, mem.solver / 2.0^20,
                mem.state / 2.0^20, mem.workspace / 2.0^20)
        @printf("; %.2f Mpoint-steps/s\n", throughput)
        @printf("mean-density drift %.3e\n",
                abs(result.mass1 - result.mass0) /
                max(abs(result.mass0), eps(Float64)))
        push!(summaries,
              (; T, cfg, peak=rates[imax-1], peak_t=ts[imax],
               wall=solver.wall_total, memory=total_mem,
               drift=abs(result.mass1 - result.mass0) /
                     max(abs(result.mass0), eps(Float64))))
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
    if Float64 in precisions && Float32 in precisions && rank == 0
        println("\n=== precision comparison ===")
        for cfg in configs
            i64 = findfirst(s -> s.T === Float64 && s.cfg == cfg, summaries)
            i32 = findfirst(s -> s.T === Float32 && s.cfg == cfg, summaries)
            (i64 === nothing || i32 === nothing) && continue
            a, b = summaries[i64], summaries[i32]
            @printf("art %s, filter %d: Float32 speedup %.3fx, memory %.3fx smaller; ",
                    cfg.art ? "ON " : "OFF", cfg.filt,
                    a.wall / b.wall, a.memory / b.memory)
            @printf("peak Δ %.3e (Δt %.3e), density drift %.3e vs %.3e\n",
                    b.peak - a.peak, b.peak_t - a.peak_t,
                    b.drift, a.drift)
        end
    end
end

# Argument parsing and the backend load run at top level: a device package
# loaded inside `main` would define its methods in a newer world than the one
# `main` executes in, and every launch would raise a world-age error.
const _opt = CompactLES.script_args(ARGS, DEFAULTS; positional = (:N, :tfinal))
const _backend = make_backend(_opt.backend)
mpi_main(() -> main(_opt, _backend))
