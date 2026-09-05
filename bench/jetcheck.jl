# Runtime-dispatch audit with JET.
#
#   julia --project=. -t 1 bench/jetcheck.jl
#
# @report_opt finds call sites inference could not resolve to a concrete
# method -- i.e. real dynamic dispatch, as opposed to the merely-non-concrete
# SSA values a raw code_typed scan turns up. Restricted to CompactLES so Base
# and MPI internals do not drown the signal.
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using JET, Printf
const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

solver = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3,
           transport=Transport(mu0=1e-3), art=ArtParams(enabled=true))
Q = allocate_state(solver); dQ = zero(Q); du = zero(Q)
initialize!(solver, Q, (x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0))

ss = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3,
            sources=(ConstantBodyForce((0.0, -1.0, 0.0)),),
            art=ArtParams(enabled=false))
Qs = allocate_state(ss); dQs = zero(Qs)
initialize!(ss, Qs, (x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0))

# axis-fold solver: exercises the fold path, which the Cartesian one skips
sf = Solver(n_global=(64, 1, 1), L_domain=(1.0, 1.0, 1.0), metric=CylindricalMetric(),
            bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
            art=ArtParams(enabled=true))
Qf = allocate_state(sf); dQf = zero(Qf)
initialize!(sf, Qf, (r, θ, z) -> Prim(u=(0, 0, 0), p=1 + exp(-40(r - 0.4)^2), rho=1.0))

# two-species solver under the bulk species channel: the mole-fraction and
# bulk-flux bodies, and the conserved gradients, which the default skips
eos2 = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                     IdealSpecies{Float64}("b", 0.2, 1.09)])
sb = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3, eos=eos2,
            art=ArtParams(enabled=true, species_flux=:bulk))
Qb = allocate_state(sb); dQb = zero(Qb)
initialize!(sb, Qb, (x, y, z) -> Prim(Y=(0.5 + 0.4tanh(4sin(x)), 0.5 - 0.4tanh(4sin(x))),
                                       u=(0.1sin(x), 0, 0), p=1.0, rho=1 + 0.5cos(x)))

function summarize(name, res)
    reports = JET.get_reports(res)
    buf = IOBuffer()
    show(IOContext(buf, :color => false, :limit => false), res)
    txt = String(take!(buf))
    @printf("\n%-24s %4d runtime-dispatch reports\n", name, length(reports))
    # one line per distinct dispatch site, deduplicated
    seen = Set{String}()
    for ln in split(txt, '\n')
        occursin("runtime dispatch detected", ln) || continue
        t = strip(replace(ln, r"^[│├└─\s]*" => ""))
        t in seen && continue
        push!(seen, t)
        println("    ", first(t, 150))
    end
end

summarize("primitives!",   @report_opt target_modules=(CL,) CL.primitives!(solver, Q))
summarize("deriv_along!",  @report_opt target_modules=(CL,) CL.deriv_along!(solver.tmp_a, solver.rho, solver, 1, 1))
summarize("apply_bcs!",    @report_opt target_modules=(CL,) apply_bcs!(solver, Q))
summarize("compute_rhs!",  @report_opt target_modules=(CL,) compute_rhs!(solver, Q, dQ))
summarize("compute_rhs! (source)", @report_opt target_modules=(CL,) compute_rhs!(ss, Qs, dQs))
summarize("compute_dt",    @report_opt target_modules=(CL,) compute_dt(solver, Q))
summarize("filter_state!", @report_opt target_modules=(CL,) filter_state!(solver, Q))
summarize("step!",         @report_opt target_modules=(CL,) step!(solver, Q, dQ, du, 1e-4))
summarize("compute_rhs! (axis fold)", @report_opt target_modules=(CL,) compute_rhs!(sf, Qf, dQf))
summarize("compute_rhs! (bulk species)", @report_opt target_modules=(CL,) compute_rhs!(sb, Qb, dQb))
summarize("compute_dt (bulk species)", @report_opt target_modules=(CL,) compute_dt(sb, Qb))

println("\njet check complete")
