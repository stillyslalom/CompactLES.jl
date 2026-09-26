# Collective refreshed-coefficient substep-CFL checks.  This file runs both as
# an include from runtests.jl and directly under the MPI launcher.
using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using Test

module SubstepRateTests
using ..MPI, ..CompactLES, ..Test

const CL = CompactLES
const COMM = MPI.COMM_WORLD
const NP = MPI.Comm_size(COMM)
const PER = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

struct FinestKick end
function CL.add_source!(::FinestKick, ps, dQ, Q, t)
    ps.patch.level == 2 || return dQ
    d = ps.decomp
    m = ps.equations.i_mom[1]
    o1, o2, o3 = d.n_halo_d
    @inbounds for k in 1:d.n_local[3], j in 1:d.n_local[2], i in 1:d.n_local[1]
        dQ[i + o1, j + o2, k + o3, m] += 1e8
    end
    return dQ
end

function nested(; control=StepControl(), cfl=0.2, sources=())
    N = max(96, 24 * NP)
    r1 = BlockRegion((N ÷ 2 - 8, 0, 0), (16, 1, 1))
    r2 = BlockRegion((3 * r1.offset[1] + 11, 0, 0), (8, 1, 1))
    solver = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0), bcs=PER,
                    dims=(NP, 1, 1), cfl=cfl, control=control,
                    filter_interval=0, subcycle=true, refine=[r1, r2],
                    sources=sources)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(0.4, 0, 0), p=1.0, rho=1.0 + 0.1sin(x)))
    return solver, states
end

function collective_substep_failure(f)
    err = try
        f(); nothing
    catch caught
        caught
    end
    ok = err isa SolverFailure && err.reason === :substep_cfl
    @test MPI.Allreduce(Int(ok), +, COMM) == NP
    @test ok
    return err
end

@testset "refreshed refined-level CFL is collective and recoverable" begin
    @test_throws ArgumentError StepControl(substep_cfl=-1.0)
    @test_throws ArgumentError StepControl(substep_cfl=Inf)
    @test_throws ArgumentError StepControl(substep_cfl=NaN)

    # A tiny ceiling makes the guard deterministic while preserving the actual
    # three-level subcycling schedule.  Direct step! reports collectively.
    tiny = StepControl(substep_cfl=1e-12)
    solver, Q = nested(control=tiny)
    collective_substep_failure() do
        step!(solver, Q, Workspace(Q), 1e-4)
    end

    # The kernel max-rate path takes the same collective guard route.
    solver, Q = nested(control=tiny)
    CL.FORCE_KA[] = true
    try
        collective_substep_failure() do
            step!(solver, Q, Workspace(Q), 1e-4)
        end
    finally
        CL.FORCE_KA[] = false
    end

    # With no retry available, the partially attempted hierarchy never
    # advances the root clock or its step counter; state restoration is only a
    # promise when `_rollback!` actually has a retry to take.
    solver, Q = nested(control=tiny)
    collective_substep_failure() do
        run!(solver, Q; tfinal=1.0, nmax=1, control=tiny)
    end
    @test solver.step == 0 && solver.t == 0 && solver.cfl == 0.2

    # A source restricted to level 2 leaves level 1 below this ceiling at
    # stage one, then makes level 2 violate it at the next refreshed stage.
    # Only the deepest level owners see the source update, but the status
    # returns through both recursive parents and every root rank reports it.
    deep = StepControl(substep_cfl=0.5)
    solver, Q = nested(control=deep, sources=(FinestKick(),))
    owners = MPI.Allreduce(Int(solver.levels[3].level_comm.owned), +, COMM)
    @test NP < 4 || owners < NP
    collective_substep_failure() do
        step!(solver, Q, Workspace(Q), 1e-4)
    end

    # A retry restores the entry savepoint before taking its replacement step.
    # The backoff makes that replacement comfortably below this synthetic
    # ceiling, so the run completes and can be compared with a fresh low-CFL
    # reference trajectory.
    solver, Q = nested(control=tiny)
    CL._prime_coefficients!(solver, Q, Workspace(Q))
    retry = StepControl(substep_cfl=0.1, retries=1, savepoint_interval=1,
                        cfl_backoff=0.25)
    hits = Ref(0)
    run!(solver, Q; tfinal=1.0, nmax=1, control=retry,
         callback=Callback(EveryStep(), (_, _) -> (hits[] += 1; false)))
    ref, Qref = nested(cfl=0.05)
    run!(ref, Qref; tfinal=1.0, nmax=1, control=StepControl(substep_cfl=0))
    @test solver.step == ref.step == 1
    @test solver.t == ref.t && hits[] == 1
    @test solver.cfl == ref.cfl == 0.05
    @test all(parent(Q[i]) == parent(Qref[i]) for i in eachindex(Q))
    @test all(all(a == b for (a, b) in zip(CL._art_arrays(p), CL._art_arrays(ref.patches[i])))
              for (i, p) in enumerate(solver.patches))

    # Direct regrid! takes the same priming path as run!'s cadence hook.  This
    # is the moving Sod fixture's deterministic rebuild point from level_tests.
    wall = (SlipWallBC(), SlipWallBC())
    sod = Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
                 bcs=(wall, PER[2], PER[3]), dims=(NP, 1, 1), cfl=0.2,
                 subcycle=true, filter_cfl=0.0, regrid_interval=5,
                 refine=BlockRegion((85, 0, 0), (31, 1, 1)), tile=8)
    Qs = allocate_state(sod)
    initialize!(sod, Qs, (x, y, z) -> x < 0.5 ?
        Prim(u=(0, 0, 0), p=1.0, rho=1.0) : Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    run!(sod, Qs; tfinal=1.0, nmax=30)
    work = Workspace(Qs)
    @test CL.regrid!(sod, Qs, work, nothing)
    r0 = max_rate(sod, Qs)[1]
    CL._presync!(sod, Qs)
    CL._prime_rhs!(sod, Qs, work)
    @test max_rate(sod, Qs)[1] == r0
    @test MPI.Allreduce(Int(any(maximum(abs, p.mu_art) > 0 for p in sod.patches)), +,
                        COMM) > 0

    # A check that keeps the layout must refresh too, otherwise a timing-based
    # ownership change would decide which steps see current coefficients.
    # Suppress all density tags, then make the coefficient cache stale without
    # touching Q. An unchanged check must not replace the existing savepoint.
    # With nothing tagged the tiles are kept by their lifetime alone; without
    # it the check would remove them all.
    saved = CL.Savepoint(CL._snapshot(Qs), CL._art_snapshot(sod), sod.t, sod.step, -1)
    saved_Q, saved_art = saved.Q, saved.art
    sod.regrid.threshold = Inf
    sod.regrid.lifetime = typemax(Int)
    for p in sod.patches, a in CL._art_arrays(p)
        fill!(a, 0)
    end
    @test !CL.regrid!(sod, Qs, work, saved)
    @test max_rate(sod, Qs)[1] == r0
    @test saved.Q === saved_Q && saved.art === saved_art
    @test MPI.Allreduce(Int(any(maximum(abs, p.mu_art) > 0 for p in sod.patches)), +,
                        COMM) > 0

    # A per-run policy overrides the solver's stored one, including the
    # documented opt-out.
    solver, Q = nested(control=tiny)
    CL.FORCE_KA[] = true
    try
        run!(solver, Q; tfinal=1e-4, nmax=1, control=StepControl(substep_cfl=0))
    finally
        CL.FORCE_KA[] = false
    end
    @test solver.step == 1
end

end # module
