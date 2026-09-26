# Molecular-transport allocation comparison on one CPU rank.
# Run with: julia --project=. -t 1 bench/transport.jl
#
# Hold the EOS, state, and grid fixed across the constant model and both CEA
# diffusion modes. Positive allocation deltas reveal per-point inference
# regressions; test/transport_tests.jl supplies the numerical/inference gates.
using CompactLES
using CompactLES: compute_rhs!, padded_index

MPI.Initialized() || MPI.Init(threadlevel=:funneled)

const PER = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
const cea_eos = Nasa9Mixture(["H2", "N2"])

function build(transport)
    solver = Solver(n_global=(16, 12, 1), L_domain=(1.0, 0.75, 1.0),
                    bcs=PER, eos=cea_eos, transport=transport,
                    art=ArtificialProperties(enabled=false), filter_interval=0)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> begin
        Prim(Y=(0.2 + 0.1sinpi(2x), 0.8 - 0.1sinpi(2x)), rho=0.9,
             T_ion=450 + 30cospi(2y/0.75))
    end)
    return solver, Q, zero(Q)
end

function audit(transport)
    solver, Q, dQ = build(transport)
    compute_rhs!(solver, Q, dQ)
    compute_dt(solver, Q)
    rhs = @allocated compute_rhs!(solver, Q, dQ)
    cfl = @allocated compute_dt(solver, Q)
    I = padded_index(solver, 5, 5, 1)
    coefficients() = begin
        total = zero(eltype(Q))
        for _ in 1:1_000
            total += CompactLES.transport_at(solver.transport, solver.eos,
                                              solver.T_ion, solver.rho,
                                              solver.cp_mix,
                                              solver.field_tuples.Y, I).mu
        end
        total
    end
    coefficients()
    coeff = @allocated coefficients()
    return (; rhs, cfl, coeff)
end

constant = audit(ConstantTransport(mu0=1e-5, Pr=0.7, Sc=0.7))
cea = audit(CeaTransport(cea_eos))
# Synthetic reference values exercise the diffusion path, not a fit to H2/N2.
binary = audit(CeaTransport(cea_eos; diffusion=:mixture_averaged,
                            binary_diffusion=BinaryDiffusion([0.0 7.25e-5;
                                                               7.25e-5 0.0])))
println((; constant, cea, delta=(rhs=cea.rhs - constant.rhs,
                                 cfl=cea.cfl - constant.cfl,
                                 coeff=cea.coeff - constant.coeff),
         binary, binary_delta=(rhs=binary.rhs - constant.rhs,
                               cfl=binary.cfl - constant.cfl,
                               coeff=binary.coeff - constant.coeff)))
