# Taylor–Green vortex with the kinetic-energy budget split by which mechanism
# removes the energy. Built for the C_mu question: `test/convergence.jl` already
# runs TGV, but with `art=ArtParams(enabled=false)`, so it tests the physical
# viscous term and says nothing about the artificial one.
#
# The split is the purpose of the script. -dKE/dt is the total energy loss, and
# the viscous stress accounts for only part of it. Five channels are measured:
# molecular, artificial shear, artificial bulk, the compact filter, and the
# reversible pressure work that moves energy between kinetic and internal. What
# remains after those is printed as `unattr`: numerical error, aliasing and
# time integration.
#
# The filter column is `filter_loss`, a measured pass. Tables recorded under
# the earlier residual definition, -dKE/dt - (mol + mu* + beta*), print no
# `unattr` column beside it and are not comparable, since that residual
# carried the pressure work and every numerical error along with the filter.
#
# --- What this measured, so it is not rediscovered ---------------------------
#
#   The residual definition is an adequate proxy where the filter dominates
#   and degrades as its share falls. At 32³ to t = 10, smoother :compact, at
#   the peak: mol 12.1%, mu* 4.7%, beta* 0.0%, filter 84.1%, pressure work
#   0.4%, unattributed -1.3%, against a residual of 83.1% for the same pass.
#   That is one point low at an 84% share; if the absorbed error is roughly
#   fixed in absolute terms, it is about 8% of a 12.8% channel at 256³.
#
#   Bracketing the artificial coefficients costs nothing at production times:
#   32³ to t = 10 agrees on the peak to five digits and on the KE misfit to
#   0.013% either way, 3515 steps against 3514, and the difference reaches
#   1.9% only at 16³ to t = 1, where the peak is still rising.
#
#   32³, Re=1600, filter_interval=1, C_mu=0.002 default, at the dissipation peak
#   (t ≈ 6.3): molecular 12%, artificial shear 5%, artificial bulk ~0%,
#   filter ≈ 83%. Peak -dKE/dt 1.46e-2 at t = 6.5 against the reference
#   1.286e-2 at t = 8.97: over-dissipating, too early, and the excess is
#   mostly filter.
#
#   β* is four orders below μ* despite C_beta = 1.0: at Ma 0.1 there is almost
#   no dilatation for it to act on. That is the physics of the case, not a
#   defect.
#
#   Filter sweep at 32³, art on: interval 1 gives the above; interval 4 diverges
#   (9.5e-2 still rising at t = 10); interval 0 fails outright with
#   SolverFailure(:negative_density) at t = 5.32. The compact filter is the
#   primary stabilizer here, beyond its smoothing role; the Cook properties
#   alone do not keep this case stable at this resolution.
#
#   At 128³, 224 ranks over 2 rzhound nodes, t = 10, the 32³ trends above
#   change in three ways:
#
#     config              peak -dKE/dt      mol    mu*   filter    wall
#     art ON,  filter 1   1.2065e-2 @ 8.77  60.4%  2.3%   37.3%   1520 s
#     art OFF, filter 1   1.2248e-2 @ 8.87  62.5%  0      37.5%   1063 s
#     art ON,  filter 0   SolverFailure(:negative_density) at t = 4.66
#
#   Filter dominance does not survive resolution: 87% → 83% → 37% → 12.8% at
#   16³, 32³, 128³, 256³. The two coarse points alone did not establish a
#   trend; the fall continues at 256³.
#
#   The filter is still required, and its stabilizing role is decoupled from its
#   energy share. At 37% of the sink, removing it kills the run earlier than at
#   32³ (t = 4.66 vs 5.32) through a clean energy blow-up: KE turns upward at
#   t ≈ 4.4 and triples before positivity is lost, with dt collapsing to 2e-60.
#   TGV is unforced, so the rise is unambiguously numerical, and the filter
#   supplies essentially all of the grid-scale sink whatever its share of the
#   total. Art-off/filter-on runs to completion. The filter is necessary and
#   sufficient here; the Cook properties are neither.
#
#   Refinement does not disentangle mu* from the filter: their ratio is
#   invariant across an 8× refinement in linear resolution (5.0/83 = 0.060 at
#   32³, 2.3/37.3 = 0.062 at 128³, 0.8/12.8 = 0.0625 at 256³), both shrinking
#   together as the molecular term takes over. Refining until the filter's
#   share is negligible and then fitting C_mu is therefore not a workable plan.
#   Holding the filter fixed and fitting against the peak is, because the peak
#   is resolved far below the effect size. Every denominator above is the old
#   residual column rather than a measured filter dissipation, so the ratio
#   itself awaits remeasurement. Numbers and the open C_mu sweep are in
#   reference/CALIBRATION.md.
#
#   256³, art ON, filter 1, 4 MI300A APUs on rzadams (backend=amdgpu), t = 10:
#   peak -dKE/dt 1.3043e-2 @ 8.84, mol 86.4%, mu* 0.8%, beta* 0.0%, filter 12.8%.
#   The peak is 1.6% above the reference seen through the same window
#   (reference/CALIBRATION.md, "Taylor-Green"), with its time converged toward
#   9; the coarse-grid early overprediction is gone. Only the art-ON leg was
#   run, so the filter necessary-and-sufficient test (art OFF completes,
#   filter 0 fails) is measured at 128³ but not yet at 256³.
#
# Cost: 32³ to t = 10 is ~3.3 min per configuration on a 24-thread desktop, 64³
# ~13 min. On a cluster, 128³ is ~20–25 min per configuration at 224 ranks over
# two nodes (0.10–0.12 s/step, ~11k–13k steps), which is why this script lives
# in bench/ rather than test/.
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
# Against the tabulated reference peak of 1.286e-2 at t = 8.97, both 64³
# precisions are about 3% low and land at the same time. Float32 halves
# resident solver/state/RK memory but buys only 1.10x CPU throughput here; its
# conservation drift is a real 1e-4 tradeoff, not hidden by the Float64
# diagnostic reductions.
#
# Usage: positional grid and end time, then `key=value` options:
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
#             0.002. Both settings sit in one entry because they have to be
#             calibrated together (the filter-dominance note above). Default
#             "off:1,on:1": art off and art on, both filtered every step, which
#             is the first-order comparison.
#   progress  ProgressLog interval in steps, 0 (default) to disable. Set it for
#             any run long enough to look hung: at 256³ a configuration is
#             ~21,500 steps, and the sample table below is otherwise the only
#             output for hours.
#   sample    steps between budget samples (default 100). Scale it with the
#             step count or a long run prints hundreds of rows.
#   filter_probe
#             measure the filter's own dissipation (default true). Each budget
#             sample then costs one trial step and two extra reductions on top
#             of the gradient pass, and the warm-up state and workspace are
#             held for the run instead of freed. Set false to recover both.
#   nmax      step cap per configuration (default none). A sweep that may visit
#             bad configurations should set one: a run that loses positivity
#             does not crash but grinds (CLAUDE.md, Conventions).
#   smoother, mu_sensor, beta_sensor, reduction
#             `ArtParams` settings applied to every configuration in the sweep,
#             so a comparison across them is one invocation each rather than one
#             entry each. `mu_sensor` is the setting this case is suited to: TGV is
#             the only case in the repository where the μ* channel carries a
#             measurable share of the sink, and no case in the 1-D battery of
#             bench/artcal.jl exercises a shear sensor at all. Note that
#             `smoother` still defaults to `:compact`, which is no longer the
#             solver default, because the 128³ numbers above were measured under it. Pass
#             `smoother=gaussian` for a comparison against the default
#             configuration.
#   alphaf    comma-separated list of compact-filter alpha values (default
#             "0.45"), the filter strength itself; larger filters more weakly.
#             One of the three axes of the joint filter fit, the other two being
#             `filter_interval` inside `configs` and `filter_cfl` below. The
#             three cross with `configs`, `cfl` and `precision`, so one
#             invocation runs the whole grid and prints one block per point.
#
#   cfl       comma-separated list of timestep multipliers (default "0.6").
#             Exposed because the filter's
#             share of the sink depends on it: unrelaxed, the filter removes
#             energy per APPLICATION, so halving the CFL doubles the number of
#             applications covering the same interval and doubles what the
#             filter takes. Sweep it to measure that, not to tune anything.
#   filter_cfl  comma-separated list of reference CFLs at which one filter pass
#             is applied at full strength, 0 (default) for the unrelaxed
#             formulation. Positive makes the filter dissipation a rate rather
#             than a per-application amount, so the `cfl` sweep flattens. See
#             `filter_weight`.
#   precision Float storage and arithmetic to measure: float64 (default),
#             float32, or both. Diagnostics intentionally accumulate in
#             Float64 in either mode; this option changes the solver state,
#             schemes, EOS, transport, artificial controls, and RK workspace.
#   backend   cpu (default), amdgpu, or cuda: where the solver lives. A device
#             backend needs an environment carrying the device package (see
#             probes/device_bringup.jl); the solver and state are
#             device-resident and the energy diagnostics read a host copy
#             of the state, downloaded per callback on the documented
#             I/O-gathers-to-host path and excluded from solver wall time by
#             the same accounting that excludes callbacks on the CPU.
#   snapshots comma-separated times at which to write an HDF5 checkpoint of the
#             state, empty (default) for none. The spectra under N1 are taken
#             offline from these by `bench/tgv_spectrum.jl`, so nothing here
#             computes a transform and no distributed FFT exists. Requires HDF5
#             to be loadable, which the package environment cannot do (it is a
#             weakdep); run from a project carrying both. Files land in
#             `snapshot_dir` under a stem naming the configuration, so a sweep
#             writes one set per point without collision.
#
#             `run!` shortens a step to land exactly on each instant, so
#             snapshots are comparable across a sweep. Under
#             `filter_cfl = 0` a shortened step still pays a full filter pass,
#             so requesting snapshots perturbs the energy budget slightly at
#             each one; under a positive `filter_cfl` it does not. Use the same
#             snapshot times for every configuration being compared.
#
#   snapshot_dir  directory for those files (default "tgv_snapshots").
#
#   window    steps either side for every -dKE/dt reported (default 250, i.e. a
#             501-step window, clamped to length(ts)/8). Do not lower it towards
#             1 for more detail: the one-step rate is contaminated by dt jitter
#             at several times the effect size. See `windowed_rate`.
#
# Every configuration is compared against the 512^3 spectral reference vendored
# at `data/spectral_Re1600_512.gdiag`: the peak seen through the run's own
# window, and the relative-L2 misfits of the kinetic-energy and dissipation
# histories. The misfits are the quantity the joint alpha/cadence/filter_cfl fit
# minimizes; the peak alone is one scalar and one parameter always fits it.
#
# Parsed by `script_args` (src/scriptargs.jl), shared with the cluster scripts;
# the reasoning for ARGS over environment variables is there. An unknown key is
# an error, so a typo produces a message rather than an hour-long run at the
# default.
#
# Runs under mpiexec unchanged; every reduction here is collective.
#
# A configuration that raises SolverFailure is reported and the sweep continues
# to the next one. That is safe only because the failure is raised off a reduced
# quantity (`max_rate` reduces, `check_step` reads the result), so every rank
# throws at the same step and every rank moves on together. Anything else
# escapes to `mpi_main`, which aborts the job; do not widen that catch.

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
    s_mol, s_shear, s_bulk, s_pdil, s_rho = 0.0, 0.0, 0.0, 0.0, 0.0
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
        pres = _host(ps.p)
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
            # Pressure work, the reversible exchange with internal energy:
            # dKE/dt = ∫p∇·u − ∫τ:∇u over a periodic domain, so this enters
            # −dKE/dt with the opposite sign to the three dissipations. It was
            # previously inside the residual that the table labelled "filter".
            s_pdil += w * pres[I] * divu
            s_rho += w * rho[I]
        end
    end
    v = MPI.Allreduce([s_mol, s_shear, s_bulk, s_pdil, s_rho], +, solver.comm)
    return (v[1] / v[5], v[2] / v[5], v[3] / v[5], v[4] / v[5])
end

# The state vector a refined run carries, or the one-element vector of a
# single-patch run's state, so one patch loop serves both.
_as_vector(states::Vector) = states
_as_vector(Q) = [Q]

_copy_state!(dst::Vector, src::Vector) =
    (for (a, b) in zip(dst, src); copyto!(parent(a), parent(b)); end; dst)
_copy_state!(dst, src) = (copyto!(parent(dst), parent(src)); dst)

# `diss_split` and the filter probe both recompute the artificial coefficients
# into the solver's own arrays, and `max_rate` sizes the next step from those
# arrays without recomputing them. Snapshotting them around a diagnostic keeps
# the instrument from perturbing the run it is measuring, which it otherwise
# does once every `sample=` steps.
_art_snapshot(solver, states) =
    [CL.art_block(ps) for (ps, _) in CL.eachpatch(solver, _as_vector(states))]

function _art_restore!(solver, states, blocks)
    for (n, (ps, _)) in enumerate(CL.eachpatch(solver, _as_vector(states)))
        CL.set_art_block!(ps, blocks[n])
    end
    return solver
end

"""
The compact filter's own energy removal, measured directly.

The residual this replaces, total loss minus the three viscous channels,
carries the pressure work and every numerical error alongside the filter, so it
was never a filter dissipation.

A callback sees a state the filter has already acted on this step, so filtering
it again would measure a second pass: at each wavenumber the first pass's loss
scaled by the square of the transfer function, a severe underestimate exactly
where the filter does its work. The copy is therefore advanced one step first,
and the pass measured on it is the one the run takes next.

Returned on `diss_split`'s normalization, energy per unit mass per unit time.
"""
function filter_loss(solver, Q, probe, dt, sync_host!, mass)
    Qp, wsp = probe
    _copy_state!(Qp, Q)
    step!(solver, Qp, wsp, dt)
    ke_pre = kinetic_energy(solver, sync_host!(Qp))
    CL.filter_state!(solver, Qp)
    ke_post = kinetic_energy(solver, sync_host!(Qp))
    return (ke_pre - ke_post) / (dt * mass)
end

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

A boxcar over a curved peak reads slightly low, about 0.2% at 128³ with the
default 501-step window, growing as the window widens. That is common-mode across
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

# --- The spectral reference history ------------------------------------------
#
# `data/spectral_Re1600_512.gdiag` is the 512^3 dealiased pseudo-spectral
# Taylor-Green solution at Re = 1600 distributed with the International
# Workshops on High-Order CFD Methods, vendored verbatim under its upstream
# name; `data/README.md` carries the provenance. Its columns are time, kinetic
# energy, -dE/dt and enstrophy, from t = 0 to t = 19.99 at dt = 0.01.
#
# It replaces the single digitized scalar this script used to print. The
# tabulated peak is 1.28575e-2 at t = 8.97 against the 1.289e-2 at t = 8.86 read
# off van Rees et al., JCP 230 (2011), Fig. 8: the values agree to 0.3% and the
# times to 0.11, which is figure-reading error, so the table and the figure are
# the same solution. The rounded 1.2e-2 at t = 9 that this script printed for a
# year is 6.7% below the tabulated peak, and comparisons against it read
# correspondingly over-dissipative.
#
# The reference is incompressible and this case is compressible at Ma 0.1, so
# the kinetic energies differ by the pressure-dilatation exchange, which is
# O(Ma^2) and shows up early while the flow is still smooth.

struct SpectralReference
    t::Vector{Float64}
    ke::Vector{Float64}
    eps::Vector{Float64}
end

reference_path() = joinpath(pkgdir(CompactLES), "data", "spectral_Re1600_512.gdiag")

function load_reference(path::AbstractString = reference_path())
    t, ke, eps = Float64[], Float64[], Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, '#')) && continue
        f = split(s)
        length(f) >= 3 || error("short row '$s' in $path")
        push!(t, parse(Float64, f[1]))
        push!(ke, parse(Float64, f[2]))
        push!(eps, parse(Float64, f[3]))
    end
    length(t) > 1 || error("no data rows in $path")
    issorted(t) || error("reference times are not sorted in $path")
    return SpectralReference(t, ke, eps)
end

# Linear interpolation, clamped at both ends. The table is uniform at dt = 0.01
# and a run samples every step, so the interpolation error sits far below the
# differences being ranked.
function _interp(xs, ys, x)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    theta = (x - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + theta * (ys[i+1] - ys[i])
end

reference_ke(ref::SpectralReference, t) = _interp(ref.t, ref.ke, t)

"""
The reference rate over the same interval `windowed_rate` differences, taken
from the reference's energy column rather than its own -dE/dt column so that
both curves carry the identical estimator: the ~0.2% a boxcar reads low over a
curved peak is then common-mode and cancels. `NaN` outside the table.
"""
function reference_rate(ref::SpectralReference, tlo, thi)
    (thi > tlo && tlo >= ref.t[1] && thi <= ref.t[end]) || return NaN
    return -(reference_ke(ref, thi) - reference_ke(ref, tlo)) / (thi - tlo)
end

"The tabulated maximum of the reference dissipation column, and its time."
function raw_reference_peak(ref::SpectralReference)
    v, i = findmax(ref.eps)
    return v, ref.t[i]
end

"""
The reference peak seen through a boxcar of half-width `halfwidth`, which is the
run's own window converted to time. Reported beside the raw peak because a run
compared at the default 501-step window is not being compared with the raw
tabulated maximum.
"""
function windowed_reference_peak(ref::SpectralReference, halfwidth)
    halfwidth > 0 || return raw_reference_peak(ref)
    best, best_t = -Inf, NaN
    for t in ref.t
        r = reference_rate(ref, t - halfwidth, t + halfwidth)
        isfinite(r) && r > best && ((best, best_t) = (r, t))
    end
    return best, best_t
end

"""
Relative L2 misfit of a run's history against the spectral reference, which is
the history fit `C_mu` and the filter constants are ranked on once one scalar
peak stops separating them.

Both quantities are normalized by the reference's own RMS over exactly the steps
compared, so each is a dimensionless number that falls as the fit improves.
Kinetic energy is compared at every step through `last`; the rate only where the
full window fits inside the run, since a clamped window is a different estimator
on one side. `last` excludes the truncated final step for the reason
`windowed_rate` documents.
"""
function reference_misfit(ref::SpectralReference, ts, kes, w, last)
    ske = nke = sdiss = ndiss = 0.0
    n_ke = n_diss = 0
    tmax = 0.0
    for i in 1:min(last, length(ts))
        t = ts[i]
        t > ref.t[end] && break
        r = reference_ke(ref, t)
        ske += (kes[i] - r)^2
        nke += r^2
        n_ke += 1
        tmax = t
        (i - w >= 1 && i + w <= last) || continue
        rr = reference_rate(ref, ts[i-w], ts[i+w])
        isfinite(rr) || continue
        sdiss += (windowed_rate(ts, kes, i, w) - rr)^2
        ndiss += rr^2
        n_diss += 1
    end
    rel(a, b) = b > 0 ? sqrt(a / b) : NaN
    return (ke = rel(ske, nke), diss = rel(sdiss, ndiss),
            n_ke = n_ke, n_diss = n_diss, tmax = tmax)
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
                      cfl=0.6, filter_cfl=0.0, alphaf=0.45,
                      mu_sensor=:strain, beta_sensor=:strain, reduction=:sum,
                      T::Type{<:AbstractFloat}=Float64,
                      backend::AbstractBackend=CPUBackend(),
                      refine::Int=0, tile::Int=0, subcycle::Bool=false,
                      filter_probe::Bool=true,
                      snapshots::Vector{Float64}=Float64[],
                      snapshot_stem::AbstractString="")
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
                     deriv=lele_d1_6(T), filt=compact_filter(T(alphaf), T),
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

    # Compile the precision-specific hot path before timing. The measured state
    # and solver time remain at t = 0, and the artificial coefficients the warm
    # step leaves behind are put back: `max_rate` sizes the run's first step
    # from those arrays.
    probe = filter_interval > 0 && filter_probe
    art_warm = _art_snapshot(solver, Q)
    tstage_warm = solver.tstage
    Qwarm = refine == 0 ? copy(Q) : [copy(q) for q in Q]
    warmspace = Workspace(Qwarm)
    dtwarm = compute_dt(solver, Qwarm)
    step!(solver, Qwarm, warmspace, dtwarm)
    filter_interval > 0 && CL.filter_state!(solver, Qwarm)
    kinetic_energy(solver, sync_host!(Qwarm))
    _art_restore!(solver, Q, art_warm)
    solver.tstage = tstage_warm
    # The warm pair is exactly the state and workspace `filter_loss` requires,
    # so the probe reuses it rather than allocating a second pair.
    probe || (Qwarm = warmspace = nothing)
    GC.gc()

    ts = Float64[]
    kes = Float64[]
    samples = Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Int}[]
    record = Callback(EveryStep(1), (s, Q) -> begin
        push!(ts, s.t)
        push!(kes, kinetic_energy(s, sync_host!(Q)))
        nothing
    end)
    # diss_split costs an extra gradient pass and the filter probe a trial step,
    # so the budget is sampled, not stepwise. Both write solver-owned arrays the
    # next iteration reads, so the pass brackets itself with a snapshot.
    split = Callback(EveryStep(sample), (s, Q) -> begin
        art_saved = _art_snapshot(s, Q)
        tstage_saved = s.tstage
        mol, shear, bulk, pdil = diss_split(s, Q)
        filt = probe && s.dt_prev > 0 ?
               filter_loss(s, Q, (Qwarm, warmspace), s.dt_prev, sync_host!,
                           mass0) : 0.0
        push!(samples, (s.t, mol, shear, bulk, pdil, filt, length(ts)))
        _art_restore!(s, Q, art_saved)
        s.tstage = tstage_saved
        nothing
    end)
    callbacks = (record, split)
    if !isempty(snapshots)
        # One checkpoint per listed instant. `save_checkpoint_hdf5` reads the
        # coefficient arrays rather than recomputing them, so a snapshot leaves
        # the next timestep alone; the step shortened to land here does not, see
        # the header note.
        refine == 0 ||
            error("snapshots take the single-patch state; a refined run has no " *
                  "shared-file checkpoint")
        snap = Callback(AtTime(snapshots), (s, Q) -> begin
            save_checkpoint_hdf5(s, Q, @sprintf("%s_t%08.4f", snapshot_stem, s.t))
            nothing
        end)
        callbacks = (callbacks..., snap)
    end
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
                  cfl = "0.6", filter_cfl = "0.0", alphaf = "0.45",
                  mu_sensor = :strain, beta_sensor = :strain, reduction = :sum,
                  precision = "float64", backend = "cpu",
                  refine = 0, tile = 0, subcycle = false,
                  filter_probe = true,
                  snapshots = "", snapshot_dir = "tgv_snapshots")

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

"""
Load HDF5 if any snapshot was asked for. Done at top level for the reason
`make_backend` gives: a package loaded inside `main` defines its methods in a
newer world than the one `main` runs in.
"""
function load_hdf5(snapshots)
    isempty(snapshots) && return nothing
    @eval using HDF5
    hdf5_available() ||
        error("snapshots= needs HDF5; run from a project carrying it")
    return nothing
end

"""
The stem naming one configuration's snapshot files. Every sweep axis appears,
so a sweep writes one set per point and `bench/tgv_spectrum.jl` can label a
curve from the filename alone.
"""
function snapshot_stem(dir, N, T, cfg, alphaf, cfl, filter_cfl, smoother)
    return joinpath(dir,
        @sprintf("tgv_N%d_%s_art%s_fi%d_cmu%g_a%g_cfl%g_fcfl%g_%s",
                 N, T === Float32 ? "f32" : "f64", cfg.art ? "on" : "off",
                 cfg.filt, cfg.C_mu, alphaf, cfl, filter_cfl, smoother))
end

"""
A comma-separated list of floats: the sweep axes `alphaf`, `filter_cfl` and
`cfl`, which the joint filter fit varies together. A single value is a
one-element list, so every existing invocation reads unchanged.
"""
function parse_floats(spec, key)
    out = Float64[]
    isempty(strip(String(spec))) && return out
    for item in split(String(spec), ',')
        v = tryparse(Float64, strip(item))
        v === nothing && error("bad $key entry '$item', want a float")
        push!(out, v)
    end
    isempty(out) && error("$key list must not be empty")
    return out
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
    alphafs = parse_floats(opt.alphaf, :alphaf)
    filter_cfls = parse_floats(opt.filter_cfl, :filter_cfl)
    cfls = parse_floats(opt.cfl, :cfl)
    snapshots = parse_floats(opt.snapshots, :snapshots)
    ref = load_reference()
    summaries = NamedTuple[]
    if rank == 0
        @printf("=== Taylor-Green %d^3, Re=1600, tfinal=%.1f, %d rank(s), ",
                N, tfinal, MPI.Comm_size(MPI.COMM_WORLD))
        @printf("%d thread(s), backend %s\n", Threads.nthreads(), opt.backend)
        rpk, rpt = raw_reference_peak(ref)
        @printf("    reference: 512^3 spectral, peak -dKE/dt %.4e at t = %.2f, ",
                rpk, rpt)
        @printf("history to t = %.2f\n", ref.t[end])
    end
    for T in precisions, cfg in configs, alphaf in alphafs,
        filter_cfl in filter_cfls, cfl in cfls
        result, failure = nothing, nothing
        elapsed = @elapsed begin
            try
                result = taylor_green(N, cfg.art; tfinal=tfinal, C_mu=cfg.C_mu,
                                      filter_interval=cfg.filt, sample=sample,
                                      progress=progress, nmax=nmax,
                                      smoother=opt.smoother,
                                      cfl=cfl, filter_cfl=filter_cfl,
                                      alphaf=alphaf,
                                      mu_sensor=opt.mu_sensor,
                                      beta_sensor=opt.beta_sensor,
                                      reduction=opt.reduction, T=T,
                                      backend=backend, refine=opt.refine,
                                      tile=opt.tile, subcycle=opt.subcycle,
                                      filter_probe=opt.filter_probe,
                                      snapshots=snapshots,
                                      snapshot_stem=snapshot_stem(
                                          opt.snapshot_dir, N, T, cfg, alphaf,
                                          cfl, filter_cfl, opt.smoother))
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
        label *= @sprintf(", alphaf %.3g, cfl %.3g, filter_cfl %.3g",
                          alphaf, cfl, filter_cfl)
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
        # Window for every rate reported below (see `windowed_rate`). Clamped so
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
              (; T, cfg, alphaf, cfl, filter_cfl,
               peak=rates[imax-1], peak_t=ts[imax],
               wall=solver.wall_total, memory=total_mem,
               drift=abs(result.mass1 - result.mass0) /
                     max(abs(result.mass0), eps(Float64))))
        @printf("peak -dKE/dt = %.4e at t = %5.2f   (%d-step window)\n",
                rates[imax-1], ts[imax], 2w + 1)
        # Against the spectral reference: the peak seen through the same boxcar,
        # then the two history misfits. One scalar peak fitted with one parameter
        # always succeeds, so the misfits, not the peak, separate a filter
        # setting that reproduces the dissipation history from one that lands
        # on its maximum by cancellation.
        halfwidth = (ts[min(imax + w, last_full)] - ts[max(imax - w, 1)]) / 2
        wpk, wpt = windowed_reference_peak(ref, halfwidth)
        mis = reference_misfit(ref, ts, kes, w, last_full)
        @printf("reference through this window: %.4e at t = %5.2f", wpk, wpt)
        @printf("   peak %+.2f%%, peak time %+.2f\n",
                100 * (rates[imax-1] - wpk) / wpk, ts[imax] - wpt)
        @printf("history misfit (rel L2 to t = %5.2f): KE %.4e over %d steps, ",
                mis.tmax, mis.ke, mis.n_ke)
        @printf("-dKE/dt %.4e over %d\n", mis.diss, mis.n_diss)
        # Peak at the last usable step, whatever ended the run. Testing `t`
        # against `tfinal` missed the case `nmax=` creates, where the run
        # stops early and every t is below tfinal.
        imax >= last_full &&
            println("    NOTE: still rising at the last step, so this is not a " *
                    "resolved peak: either the run was cut short of t = 9 " *
                    "(check nmax=) or the configuration is diverging.")
        println("     t      eps_mol      eps_mu*    eps_beta*     eps_filt" *
                "       p*divu      -dKE/dt   unattr")
        for (t, mol, shear, bulk, pdil, filt, idx) in samples
            (idx < 2 || idx >= length(ts)) && continue
            total = windowed_rate(ts, kes, idx, w)
            # What no measured channel accounts for. Pressure work enters with
            # the opposite sign, since it moves energy between kinetic and
            # internal rather than removing it. This column used to be
            # labelled "filter" and carried the filter, the pressure work and
            # every numerical error together.
            unattr = total - (mol + shear + bulk + filt - pdil)
            @printf("  %5.2f %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %6.1f%%\n",
                    t, mol, shear, bulk, filt, pdil, total,
                    100 * unattr / max(abs(total), 1e-300))
        end
        # The calibration readout, one line per configuration, so that the
        # filter share can be compared across a sweep without reading several
        # hundred table rows.
        usable = filter(s -> 2 <= s[7] < length(ts), samples)
        isempty(usable) && continue
        t, mol, shear, bulk, pdil, filt, idx =
            usable[argmin(abs.(getindex.(usable, 7) .- imax))]
        total = windowed_rate(ts, kes, idx, w)
        share(x) = 100 * x / max(abs(total), 1e-300)
        # Nearest sample to the peak, not the peak step itself, since the budget
        # pass only runs every `sample=` steps.
        @printf("at peak t=%5.2f:  mol %5.1f%%  mu* %5.1f%%  beta* %5.1f%%",
                t, share(mol), share(shear), share(bulk))
        @printf("  filter %5.1f%%  pdil %5.1f%%  unattr %5.1f%%\n",
                share(filt), share(-pdil),
                share(total - (mol + shear + bulk + filt - pdil)))
    end
    if Float64 in precisions && Float32 in precisions && rank == 0
        println("\n=== precision comparison ===")
        for cfg in configs, alphaf in alphafs, filter_cfl in filter_cfls,
            cfl in cfls
            same(s) = s.cfg == cfg && s.alphaf == alphaf &&
                      s.filter_cfl == filter_cfl && s.cfl == cfl
            i64 = findfirst(s -> s.T === Float64 && same(s), summaries)
            i32 = findfirst(s -> s.T === Float32 && same(s), summaries)
            (i64 === nothing || i32 === nothing) && continue
            a, b = summaries[i64], summaries[i32]
            @printf("art %s, filter %d, alphaf %.3g, cfl %.3g, filter_cfl %.3g: ",
                    cfg.art ? "ON " : "OFF", cfg.filt, alphaf, cfl, filter_cfl)
            @printf("Float32 speedup %.3fx, memory %.3fx smaller; ",
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
load_hdf5(strip(_opt.snapshots))
mpi_main(() -> main(_opt, _backend))
