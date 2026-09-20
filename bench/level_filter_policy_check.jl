# Standalone N12a harness for bench/level_filter_policy.jl.
#
#   julia --project=. -t 1 bench/level_filter_policy_check.jl
#   $(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))') -n 8 \
#       julia --project=. -t 1 bench/level_filter_policy_check.jl
#
# This stays in bench/ rather than runtests.jl: it loads benchmark-only method
# specializations into CompactLES and belongs in a fresh Julia process.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using Test

include(joinpath(@__DIR__, "level_filter_policy.jl"))
using .LevelFilterPolicy

const CL = CompactLES
const COMM = MPI.COMM_WORLD
const NP = MPI.Comm_size(COMM)
const PER = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

function nested(; subcycle=false, filter_cfl=1e6, filter_interval=1,
                initial=(x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.1, 0, 0)))
    N = max(96, 24 * NP)
    r1 = BlockRegion((div(N, 2) - 8, 0, 0), (16, 1, 1))
    r2 = BlockRegion((3 * r1.offset[1] + 11, 0, 0), (8, 1, 1))
    solver = Solver(n_global=(N, 1, 1), L_domain=(2 * pi, 1.0, 1.0), bcs=PER,
                    dims=(NP, 1, 1), filter_cfl=filter_cfl,
                    filter_interval=filter_interval, subcycle=subcycle,
                    refine=[r1, r2])
    states = allocate_state(solver)
    initialize!(solver, states, initial)
    return solver, states
end

function level_weight(solver, states, level)
    value = 0.0
    count = 0
    for (ps, _) in eachpatch(solver, states)
        ps.patch.level == level || continue
        value = CL.filter_weight(ps, 1)
        count += 1
    end
    total = MPI.Allreduce([value, count], +, COMM)
    total[2] > 0 || error("no rank owns level $level")
    return total[1] / total[2]
end

function check_holder_consistency(solver, states, level)
    local_value = Inf
    for (ps, _) in eachpatch(solver, states)
        ps.patch.level == level || continue
        local_value = CL.filter_weight(ps, 1)
    end
    lo = MPI.Allreduce(local_value, min, COMM)
    hi = MPI.Allreduce(isfinite(local_value) ? local_value : 0.0, max, COMM)
    @test isapprox(lo, hi)
end

function main()
@testset "N12a benchmark-only level filter policies" begin
    @test_throws ArgumentError set_policy!(:unknown)

    # :default delegates to the original multi-patch max_rate method exactly.
    solver, states = nested()
    set_policy!(:default)
    original = invoke(CL.max_rate, Tuple{CL.Solver, Vector{<:CL.ConservedState}},
                      solver, states)
    @test CL.max_rate(solver, states) == original

    # Uniform physical data have raw directional rates in 1:3:9 proportion.
    # The normalized envelope restores that ratio, while :level reaches the
    # same answer here through its per-level envelopes.
    set_policy!(:normalized)
    CL.max_rate(solver, states)
    solver.dt_prev = 1e-5
    wn = [level_weight(solver, states, l) for l in 0:2]
    @test isapprox(wn[2], 3wn[1])
    @test isapprox(wn[3], 3wn[2])
    for l in 0:2
        check_holder_consistency(solver, states, l)
    end

    set_policy!(:level)
    CL.max_rate(solver, states)
    wl = [level_weight(solver, states, l) for l in 0:2]
    @test isapprox(wl, wn)

    # Replacing the state then calling max_rate replaces the solver-identity
    # cache.  This checks replacement directly; moving layouts belong to the
    # benchmark instruments rather than this focused harness.
    old = wl[1]
    initialize!(solver, states,
                (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.4, 0, 0)))
    CL.max_rate(solver, states)
    @test level_weight(solver, states, 0) > old

    # A shorter accepted step is reflected linearly in a relaxed weight.
    set_policy!(:normalized)
    CL.max_rate(solver, states)
    solver.dt_prev = 1e-5
    full = level_weight(solver, states, 1)
    solver.dt_prev /= 2
    @test isapprox(level_weight(solver, states, 1), full / 2)

    # Only the deepest owners see the faster flow. Their normalized rate must
    # still strengthen the root filter on every rank, including nonowners.
    coarse_before = level_weight(solver, states, 0)
    for (ps, Q) in eachpatch(solver, states)
        ps.patch.level == 2 || continue
        momentum = ps.equations.i_mom[1]
        energy = ps.equations.i_energy
        for i in 1:ps.decomp.n_local[1]
            I = gidx(ps, i, 1, 1)
            rho = Q[I, 1]
            old_momentum = Q[I, momentum]
            Q[I, momentum] = 2rho
            Q[I, energy] += (Q[I, momentum]^2 - old_momentum^2) / (2rho)
        end
    end
    CL.max_rate(solver, states)
    @test level_weight(solver, states, 0) > coarse_before
    check_holder_consistency(solver, states, 0)

    # Unrelaxed and subcycled paths invoke CompactLES unchanged.
    solver0, states0 = nested(filter_cfl=0.0)
    set_policy!(:normalized)
    CL.max_rate(solver0, states0)
    solver0.dt_prev = 1e-5
    for (ps, _) in eachpatch(solver0, states0)
        @test CL.filter_weight(ps, 1) == 1.0
    end
    solver_sub, states_sub = nested(subcycle=true)
    set_policy!(:normalized)
    CL.max_rate(solver_sub, states_sub)
    solver_sub.dt_prev = 1e-5
    for (ps, _) in eachpatch(solver_sub, states_sub)
        production = invoke(CL.filter_weight, Tuple{CL.SolverLike{Float64}, Int}, ps, 1)
        @test CL.filter_weight(ps, 1) == production
    end

    # :cadence makes all levels' due-pass weights equal:
    # its stride cancels the 3^level normalized rate.  It is intentionally
    # restricted to filter_interval = 1, because it uses a fixed latest-dt
    # approximation rather than an elapsed-time accumulator.
    sine = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.1sin(6x), 0, 0))
    solver_c, states_c = nested(filter_cfl=0.35, initial=sine)
    set_policy!(:cadence)
    CL.max_rate(solver_c, states_c)
    solver_c.dt_prev = 1e-5
    wc = [level_weight(solver_c, states_c, l) for l in 0:2]
    @test isapprox(wc[1], wc[2]) && isapprox(wc[2], wc[3])
    root = states_c[1]
    solver_c.step = 1
    before = copy(parent(root))
    CL._level_filter!(solver_c, solver_c.levels[1], states_c)
    @test parent(root) == before
    solver_c.step = LevelFilterPolicy._stride(solver_c, 0)
    CL._level_filter!(solver_c, solver_c.levels[1], states_c)
    change = maximum(abs(root[I, c] - before[I, c])
                     for I in CL.interior(solver_c.patches[1].decomp),
                         c in 1:solver_c.equations.n_cons)
    @test MPI.Allreduce(change, max, COMM) > 0
    bad, bad_states = nested(filter_interval=2)
    @test_throws ArgumentError CL.max_rate(bad, bad_states)

    # Deeper levels can be owned by a strict rank subset; their envelope still
    # agrees on every holder and non-owners contribute only neutral zeros.
    owners = MPI.Allreduce(Int(solver.levels[3].level_comm.owned), +, COMM)
    @test owners > 0
    NP < 4 || @test owners < NP

    # Loading this helper cannot alter a subcycled trajectory: its candidate
    # methods delegate before touching either the cache or filter weights.
    ref_solver, ref_states = nested(subcycle=true, initial=sine)
    set_policy!(:default)
    run!(ref_solver, ref_states; tfinal=1.0, nmax=3)
    trial_solver, trial_states = nested(subcycle=true, initial=sine)
    set_policy!(:normalized)
    run!(trial_solver, trial_states; tfinal=1.0, nmax=3)
    @test ref_solver.step == trial_solver.step == 3
    @test all(parent(ref_states[i]) == parent(trial_states[i])
              for i in eachindex(ref_states))
end

set_policy!(:default)
return nothing
end

mpi_main(main)
