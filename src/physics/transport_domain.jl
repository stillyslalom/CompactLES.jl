# Polynomial transport is strict about its source domain. Check primitives
# before coefficient kernels run, reducing on the communicator of the caller,
# never on the whole solver communicator from inside a rank-local patch loop.
@inline function _transport_status_point!(out, transport::CeaTransport{T,N}, eos,
                                         temperature, rho, Y, o1, o2, o3,
                                         i, j, k) where {T,N}
    I = CartesianIndex(i + o1, j + o2, k + o3)
    fractions = ntuple(sp -> @inbounds(Y[sp][I]), Val(N))
    @inbounds out[I] = transport_domain_status(transport, eos, temperature[I], rho[I], fractions)
    return nothing
end

function _local_transport_status(solver)
    transport_has_domain(solver.transport) || return UInt8(0)
    d = solver.decomp
    out = solver.tmp_a
    pointwise!(_transport_status_point!, out, d.n_local..., out, solver.transport,
               solver.eos, solver.T_ion, solver.rho, solver.field_tuples.Y,
               d.n_halo_d...)
    return UInt8(maximum(view(out, interior(d))))
end

function _check_transport_status(solver, status)
    status == 0 && return nothing
    throw(SolverFailure(:transport_domain, solver.step, solver.t, 0.0, solver.cfl,
        "neutral binary diffusion requires finite positive temperature and pressure, " *
        "temperature inside every pair's source range, and finite positive coefficients " *
        "(domain status $status). No extrapolation or clamping is permitted."))
end

function _validate_transport_state!(solver::SolverLike, Q; current::Bool=false)
    transport_has_domain(solver.transport) || return nothing
    current || refresh_primitives!(solver, Q)
    status = MPI.Allreduce(_local_transport_status(solver), max, solver.decomp.comm)
    _check_transport_status(solver, status)
    return nothing
end

function _validate_transport_state!(solver::Solver, states::Vector{<:ConservedState};
                                    current::Bool=false)
    transport_has_domain(solver.transport) || return nothing
    status = UInt8(0)
    for (ps, Q) in eachpatch(solver, states)
        current || refresh_primitives!(ps, Q)
        status = max(status, _local_transport_status(ps))
    end
    _check_transport_status(solver, MPI.Allreduce(status, max, solver.comm))
    return nothing
end

# Return a collective status instead of throwing: recursive fine-level stepping
# must first carry a rejection back to ranks waiting on its parent level.
function _prepare_level_transport!(solver, lev, states, prepared, enforce, comm)
    transport_has_domain(solver.transport) || return UInt8(0)
    status = UInt8(0)
    patches = getfield(solver, :patches)
    for pi in lev.patches
        ps = PatchSolver(solver, patches[pi])
        enforce && apply_bcs!(ps, states[pi])
        prepared || refresh_primitives!(ps, states[pi])
        status = max(status, _local_transport_status(ps))
    end
    return MPI.Allreduce(status, max, comm)
end
