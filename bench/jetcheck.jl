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

s = Solver(nglob=(32, 32, 32), Ldom=(2π, 2π, 2π), bcs=per3,
           transport=Transport(mu0=1e-3), art=ArtParams(enabled=true))
Q = allocate_state(s); dQ = zero(Q); du = zero(Q)
initialize!(s, Q, (x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0))

# axis-fold solver: exercises the fold path, which the Cartesian one skips
sf = Solver(nglob=(64, 1, 1), Ldom=(1.0, 1.0, 1.0), metric=CylindricalMetric(),
            bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
            art=ArtParams(enabled=true))
Qf = allocate_state(sf); dQf = zero(Qf)
initialize!(sf, Qf, (r, θ, z) -> Prim(u=(0, 0, 0), p=1 + exp(-40(r - 0.4)^2), rho=1.0))

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

summarize("primitives!",   @report_opt target_modules=(CL,) CL.primitives!(s, Q))
summarize("deriv_along!",  @report_opt target_modules=(CL,) CL.deriv_along!(s.tmpA, s.rho, s, 1, 1))
summarize("apply_bcs!",    @report_opt target_modules=(CL,) apply_bcs!(s, Q))
summarize("compute_rhs!",  @report_opt target_modules=(CL,) compute_rhs!(s, Q, dQ))
summarize("compute_dt",    @report_opt target_modules=(CL,) compute_dt(s, Q))
summarize("filter_state!", @report_opt target_modules=(CL,) filter_state!(s, Q))
summarize("step!",         @report_opt target_modules=(CL,) step!(s, Q, dQ, du, 1e-4))
summarize("compute_rhs! (axis fold)", @report_opt target_modules=(CL,) compute_rhs!(sf, Qf, dQf))

println("\njet check complete")
