# Standalone collective-domain tests for neutral polynomial transport.
#
# Run serially or with the bundled MPI launcher:
#   julia --project=. -t 1 test/neutral_transport_domain_tests.jl
#   mpiexec -n 2 julia --project=. -t 1 test/neutral_transport_domain_tests.jl
#   mpiexec -n 4 julia --project=. -t 1 test/neutral_transport_domain_tests.jl
#   mpiexec -n 8 julia --project=. -t 1 test/neutral_transport_domain_tests.jl

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using Test

module NeutralTransportDomainTests

using ..MPI
using ..CompactLES
using ..Test

const CL = CompactLES
const COMM = MPI.COMM_WORLD
const NP = MPI.Comm_size(COMM)
const RANK = MPI.Comm_rank(COMM)
const PER = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

# The temperature 340 K lies inside the widest pair range, but outside the
# D2--N2 range.  This catches an implementation which checks only the extrema
# over the model instead of every active pair.
function polynomial_transport()
    eos = IdealMixture((IdealSpecies("H2"; R=1.0, gamma=1.4),
                        IdealSpecies("D2"; R=0.8, gamma=1.4),
                        IdealSpecies("N2"; R=0.3, gamma=1.4)))
    Dref = [0.0 7.0e-5 6.0e-5;
            7.0e-5 0.0 5.0e-5;
            6.0e-5 5.0e-5 0.0]
    coeff = zeros(3, 3, 1)
    coeff[:, :, 1] .= 1.75
    Tmin = [0.0 250.0 280.0;
            250.0 0.0 290.0;
            280.0 290.0 0.0]
    Tmax = [0.0 400.0 350.0;
            400.0 0.0 330.0;
            350.0 330.0 0.0]
    binary = BinaryDiffusionPolynomial(("H2", "D2", "N2"), Dref, coeff,
                                       Tmin, Tmax; temperature_ref=300.0)
    transport = CeaTransport(eos; diffusion=:mixture_averaged,
                             binary_diffusion=binary)
    return eos, transport
end

good_prim() = Prim(Y=(0.2, 0.3, 0.5), rho=1.0, T_ion=300.0)
bad_prim() = Prim(Y=(0.2, 0.3, 0.5), rho=1.0, T_ion=340.0)

function base_solver(; kwargs...)
    eos, transport = polynomial_transport()
    return Solver(n_global=(72, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
                  eos=eos, transport=transport, dims=(NP, 1, 1),
                  art=ArtParams(enabled=false), filter_interval=0; kwargs...)
end

function poison!(ps, Q)
    i, j, k = ntuple(d -> cld(ps.decomp.n_local[d], 2), 3)
    CL.write_conserved!(Q, gidx(ps, i, j, k), ps, bad_prim())
    return Q
end

"Assert that an operation rejects collectively with the transport-domain reason."
function collective_domain_failure(f)
    err = try
        f()
        nothing
    catch caught
        caught
    end
    local_ok = err isa SolverFailure && err.reason === :transport_domain
    !local_ok && RANK == 0 && @info "unexpected transport-domain result" err
    @test MPI.Allreduce(Int(local_ok), +, COMM) == NP
    @test local_ok
    return nothing
end

function rank_local_case()
    solver = base_solver()
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> good_prim())
    RANK == NP - 1 && poison!(solver, Q)
    return solver, Q
end

@testset "single rank-local polynomial transport violation" begin
    # Each entry point gets a fresh state because step! and run! may update it
    # before reporting a failure.
    solver, Q = rank_local_case()
    collective_domain_failure() do
        max_rate(solver, Q)
    end

    solver, Q = rank_local_case()
    collective_domain_failure() do
        compute_rhs!(solver, Q, zero(Q))
    end

    solver, Q = rank_local_case()
    work = Workspace(Q)
    collective_domain_failure() do
        step!(solver, Q, work, 1e-6)
    end

    solver, Q = rank_local_case()
    collective_domain_failure() do
        run!(solver, Q; tfinal=1e-6, nmax=1)
    end

    solver, Q = rank_local_case()
    collective_domain_failure() do
        dissipation_rate(solver, Q)
    end

    # Force Array storage through the KernelAbstractions launch route.  The
    # domain scan is a pointwise kernel, so the host loop alone does not cover
    # its launchable body.
    solver, Q = rank_local_case()
    CL.FORCE_KA[] = true
    try
        collective_domain_failure() do
            max_rate(solver, Q)
        end
    finally
        CL.FORCE_KA[] = false
    end
end

@testset "patch-disparate ownership rejects collectively" begin
    eos, transport = polynomial_transport()
    solver = Solver(n_global=(144, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
                    eos=eos, transport=transport, patch_grid=(2, 1, 1),
                    art=ArtParams(enabled=false), filter_interval=0)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> good_prim())
    for (ps, Q) in CL.eachpatch(solver, states)
        ps.patch.id == 2 && poison!(ps, Q)
    end
    collective_domain_failure() do
        max_rate(solver, states)
    end
end

@testset "subcycle fine-only owner rejects collectively" begin
    eos, transport = polynomial_transport()
    solver = Solver(n_global=(72, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
                    eos=eos, transport=transport,
                    refine=BlockRegion((32, 0, 0), (4, 1, 1)), subcycle=true,
                    art=ArtParams(enabled=false), filter_interval=0)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> good_prim())
    poisoned = false
    for (ps, Q) in CL.eachpatch(solver, states)
        if ps.patch.level == 1
            poison!(ps, Q)
            poisoned = true
        end
    end
    # The 10-node child is held by one rank for every supported MPI size.
    @test MPI.Allreduce(Int(poisoned), +, COMM) == 1
    work = Workspace(states)
    collective_domain_failure() do
        step!(solver, states, work.dQ, work.du, 1e-6)
    end


    # The same subset-owned hierarchy remains runnable when every level is in
    # range; this guards against reducing a nonowner's empty level as failure.
    valid = allocate_state(solver)
    initialize!(solver, valid, (x, y, z) -> good_prim())
    valid_work = Workspace(valid)
    @test step!(solver, valid, valid_work.dQ, valid_work.du, 1e-8) === valid
    @test all(all(isfinite, parent(Q)) for Q in valid)
end

MPI.Barrier(COMM)
RANK == 0 && println("neutral transport domain tests passed on $NP rank(s)")

end # module
