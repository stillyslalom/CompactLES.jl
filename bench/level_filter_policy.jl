# N12a benchmark-only level-aware filter policies.
#
# Include this file explicitly in a fresh Julia process before constructing the
# solver (bench/levelfilter.jl does that).  These methods deliberately extend
# CompactLES internals and are not package behavior: do not load this helper
# beside another experiment that also specializes max_rate or filter_weight.
#
# :normalized uses one normalized directional envelope,
#     max_{patches} r_d / 3^level,
# then restores a patch's physical level rate by multiplying by 3^level.
# :level instead uses a separate global directional envelope at each level.
# :cadence uses the normalized weights but filters level l only every
# 3^(L - 1 - l) root steps.  Its weight multiplies the latest dt by that
# stride.  This is intentionally only a fixed-stride cadence diagnostic: a
# landing/final shortened step is not accumulated, so it must not support a
# temporal-invariance or production-policy claim.

module LevelFilterPolicy

using MPI
using CompactLES

const CL = CompactLES
const POLICIES = (:default, :normalized, :level, :cadence)
const POLICY = Ref{Symbol}(:default)

"Select the benchmark-only level filter policy for subsequently constructed runs."
function set_policy!(policy::Symbol)
    policy in POLICIES || throw(ArgumentError(
        "level filter policy must be one of $(join(string.(POLICIES), ", ")), got :$policy"))
    POLICY[] = policy
    empty!(RATE_CACHE)
    return policy
end

policy() = POLICY[]

# A cache is per solver identity.  `max_rate` overwrites its rate payload at
# every call, rather than attempting to recognize a regrid from patch layout.
# That also covers ranks that own no tile of a refined level: their zeros enter
# the one whole-solver reduction below and owners supply the envelope.
mutable struct RateCache
    normalized::NTuple{3,Float64}
    levels::Vector{NTuple{3,Float64}}
end

const RATE_CACHE = IdDict{Any,RateCache}()

# An unrelaxed filter never reads a rate weight, so preserve that path exactly
# too: no cache sweep or added collective when filter_cfl == 0.
_candidate(solver) = POLICY[] !== :default && !getfield(solver, :subcycle) &&
                     solver.filter_cfl > 0
_stride(solver, level::Int) = 3 ^ (CL.nlevels(solver) - 1 - level)

function _store_rate_cache!(solver, states, selected::Symbol)
    nlev = CL.nlevels(solver)
    # The production max_rate call immediately before this sweep exchanged Q
    # and recovered primitives patch by patch.  Read those fresh primitives;
    # this extra diagnostic sweep makes no per-patch collective call.
    if selected === :level
        local_rates = zeros(Float64, 3nlev)
        for (ps, Q) in CL.eachpatch(solver, states)
            _, _, rd = CL._local_max_rate(ps, Q)
            off = 3 * ps.patch.level
            @inbounds for d in 1:3
                local_rates[off + d] = max(local_rates[off + d], Float64(rd[d]))
            end
        end
        global_rates = MPI.Allreduce(local_rates, max, solver.comm)
        levels = [ntuple(d -> global_rates[3 * l + d], 3) for l in 0:nlev-1]
        normalized = ntuple(d -> maximum(r[d] / 3.0^(l - 1)
                                         for (l, r) in enumerate(levels)), 3)
        RATE_CACHE[solver] = RateCache(normalized, levels)
        return nothing
    end

    local_rates = zeros(Float64, 3)
    for (ps, Q) in CL.eachpatch(solver, states)
        _, _, rd = CL._local_max_rate(ps, Q)
        scale = 3.0 ^ ps.patch.level
        @inbounds for d in 1:3
            local_rates[d] = max(local_rates[d], Float64(rd[d]) / scale)
        end
    end
    normalized = Tuple(MPI.Allreduce(local_rates, max, solver.comm))
    RATE_CACHE[solver] = RateCache(normalized, NTuple{3,Float64}[])
    return nothing
end

# This is more specific than CompactLES' Solver/Vector method, so `invoke`
# first preserves the production rate, density, exchange, primitive recovery,
# and its single production reduction exactly.  Candidates then pay one
# additional whole-solver Allreduce to cache the diagnostic filter envelopes.
function CL.max_rate(solver::CL.Solver{Float64}, states::Vector{<:CL.ConservedState})
    result = invoke(CL.max_rate,
                    Tuple{CL.Solver, Vector{<:CL.ConservedState}}, solver, states)
    selected = POLICY[]
    _candidate(solver) || return result
    selected === :cadence && solver.filter_interval != 1 &&
        throw(ArgumentError(":cadence is only defined for filter_interval = 1"))
    _store_rate_cache!(solver, states, selected)
    return result
end

function _cached_rate(ps::CL.PatchSolver{Float64}, d::Int, selected::Symbol)
    cached = get(RATE_CACHE, ps.solver, nothing)
    cached === nothing && return nothing
    level = ps.patch.level
    if selected === :level
        level + 1 <= length(cached.levels) || return nothing
        return cached.levels[level + 1][d]
    end
    return cached.normalized[d] * 3.0^level
end

# A PatchSolver-specific method leaves the single-patch solver form alone and
# confines these trial weights to the refined-state filter path.
function CL.filter_weight(ps::CL.PatchSolver{Float64}, d::Int)
    selected = POLICY[]
    if !_candidate(ps.solver) || ps.filter_cfl <= 0 || ps.dt_prev <= 0
        return invoke(CL.filter_weight, Tuple{CL.SolverLike{Float64}, Int}, ps, d)
    end
    rd = _cached_rate(ps, d, selected)
    rd === nothing &&
        return invoke(CL.filter_weight, Tuple{CL.SolverLike{Float64}, Int}, ps, d)
    nactive = count(ps.decomp.active)
    stride = selected === :cadence ? _stride(ps.solver, ps.patch.level) : 1
    weight = ps.filter_interval * stride * ps.dt_prev * rd * sqrt(Float64(nactive)) /
             ps.filter_cfl
    return min(1.0, weight)
end

# The ordinary outer driver already calls filter_state! at filter_interval.
# For :cadence its per-level hook suppresses non-due passes; filter_weight
# supplies the corresponding fixed-stride strength on due ones.  The phase is
# tied to the root step, so changing hierarchy depth during a run would reset
# neither phase nor a physical elapsed-time accumulator; this is benchmark-only.
function CL._level_filter!(solver::CL.Solver{Float64}, lev::CL.Level, states)
    selected = POLICY[]
    if !_candidate(solver) || selected !== :cadence || solver.filter_cfl <= 0
        return invoke(CL._level_filter!, Tuple{CL.Solver, CL.Level, Any}, solver, lev, states)
    end
    solver.filter_interval == 1 || throw(ArgumentError(
        ":cadence is only defined for filter_interval = 1"))
    solver.step % _stride(solver, lev.index) == 0 || return states
    return invoke(CL._level_filter!, Tuple{CL.Solver, CL.Level, Any}, solver, lev, states)
end

export set_policy!, policy

end # module LevelFilterPolicy
