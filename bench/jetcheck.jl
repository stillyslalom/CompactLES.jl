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
using CompactLES: compute_rhs!, apply_bcs!, filter_state!, step!
# JET is not a dependency of this package. It resolves from the default
# environment, which stays on the load path under `--project=.`, so it is
# installed once per Julia version rather than added to Project.toml. Say that
# rather than failing with "Package JET not found in current path".
if Base.find_package("JET") === nothing
    println("""
    bench/jetcheck.jl needs JET, which is not a dependency of this package.
    Install it in your default environment, which stays on the load path
    under --project=. :

        julia -e 'import Pkg; Pkg.add("JET")'
    """)
    exit(1)
end
using JET, Printf
const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

solver = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3,
           transport=ConstantTransport(mu0=1e-3), art=ArtificialProperties(enabled=true))
Q = allocate_state(solver); dQ = zero(Q); du = zero(Q)
initialize!(solver, Q, (x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0))

ss = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3,
            sources=(ConstantBodyForce((0.0, -1.0, 0.0)),),
            art=ArtificialProperties(enabled=false))
Qs = allocate_state(ss); dQs = zero(Qs)
initialize!(ss, Qs, (x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0))

# axis-fold solver: exercises the fold path, which the Cartesian one skips
sf = Solver(n_global=(64, 1, 1), L_domain=(1.0, 1.0, 1.0), metric=CylindricalMetric(),
            bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
            art=ArtificialProperties(enabled=true))
Qf = allocate_state(sf); dQf = zero(Qf)
initialize!(sf, Qf, (r, θ, z) -> Prim(u=(0, 0, 0), p=1 + exp(-40(r - 0.4)^2), rho=1.0))

# two-species solver under the bulk species channel: the mole-fraction pass,
# the conserved gradients and both shared-D_b flux bodies, since the channel is
# a runtime field and JET analyses the partial-density branch in the same method
eos2 = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                     IdealSpecies{Float64}("b", 0.2, 1.09)])
sb = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3, eos=eos2,
            art=ArtificialProperties(enabled=true, species_flux=:bulk))
Qb = allocate_state(sb); dQb = zero(Qb)
initialize!(sb, Qb, (x, y, z) -> Prim(Y=(0.5 + 0.4tanh(4sin(x)), 0.5 - 0.4tanh(4sin(x))),
                                       u=(0.1sin(x), 0, 0), p=1.0, rho=1 + 0.5cos(x)))

# JET follows a call into its callee and infers the callee at the call's
# argument types. A report is dropped when its stack passes through an
# instance inferred at an abstract type in an argument Julia specializes on
# (`is_compileable_mi` is false): at run time that callee is compiled for the
# concrete types it receives, so the abstract instance never runs. The
# dispatch into it is still reported, in its concrete caller. An argument
# declared `@nospecialize` is compiled at its declared type whatever is
# inferred there, so an instance inferred at an abstract type in it is kept
# and the dispatches inside it are counted (`sensor_mirror` in `_face_mirror`,
# the calls in `_build_reflux!`). Since Julia 1.12 a call whose one matching
# method is compiled for the call's abstract argument type is a type test and
# a static call, with `Core.throw_methoderror` in the fallback branch, so the
# call into such a callee is not itself a dispatch. Where the solver
# dispatches on purpose into a specialized callee (through the plan-operator
# handle of `compute_artificial!` or a `_cold` barrier of timestep.jl), the
# callee is probed at concrete types below.
_runs(vf) = JET.is_compileable_mi(vf.linfo)

function summarize(name, res)
    reports = filter(r -> all(_runs, r.vst[2:end]), JET.get_reports(res))
    @printf("\n%-24s %4d runtime-dispatch reports\n", name, length(reports))
    # one line per distinct dispatch site, deduplicated
    seen = Set{String}()
    for r in reports
        buf = IOBuffer()
        JET.print_report(IOContext(buf, :color => false, :limit => false), r)
        for ln in split(String(take!(buf)), '\n')
            occursin("runtime dispatch detected", ln) || continue
            t = strip(replace(ln, r"^[│├└─\s]*" => ""))
            t in seen && continue
            push!(seen, t)
            println("    ", first(t, 150))
        end
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
# Callees the solver reaches only through an intended dynamic dispatch, probed
# at the concrete types they are compiled for at run time (see `summarize`).
summarize("smooth!", @report_opt target_modules=(CL,) CL.smooth!(solver.sensor, solver))
summarize("_detect!", @report_opt target_modules=(CL,) CL._detect!(solver.sensor, solver.tmp_a, solver, 1, true))
summarize("_bulk_diffusivity!", @report_opt target_modules=(CL,) CL._bulk_diffusivity!(sb))
summarize("_bulk_gradients!", @report_opt target_modules=(CL,) CL._bulk_gradients!(sb, Qb))
summarize("_local_max_rate_launch",
          @report_opt target_modules=(CL,) CL._local_max_rate_launch(solver, Q))

println("\njet check complete")
