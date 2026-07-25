# Locate the runtime-dispatch sites reported by JET: group them by the
# innermost CompactLES frame (file:line + method) rather than dumping SSA IR.
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using JET, Printf
const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

s = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3,
           transport=Transport(mu0=1e-3), art=ArtParams(enabled=true))
Q = allocate_state(s); dQ = zero(Q)
initialize!(s, Q, (x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0))

sc = Solver(n_global=(32, 16, 12), L_domain=(1.0, 2π, 0.5), metric=CylindricalMetric(),
            origin=(0.2, 0.0, 0.0),
            bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
            art=ArtParams(enabled=true))
Qc = allocate_state(sc); dQc = zero(Qc)
initialize!(sc, Qc, (r, θ, z) -> Prim(u=(0, 0.1r, 0), p=1.0, rho=1.0))

sn = Solver(n_global=(32, 12, 12), L_domain=(1.0, 0.4, 0.4),
            bcs=((SlipWallBC(), NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
            art=ArtParams(enabled=true))
Qn = allocate_state(sn); dQn = zero(Qn)
initialize!(sn, Qn, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))

function where_(name, res)
    reports = JET.get_reports(res)
    counts = Dict{String,Int}()
    for r in reports
        frame = nothing
        for f in r.vst                       # innermost CompactLES frame wins
            m = f.linfo
            mod = m isa Core.MethodInstance ? m.def.module : nothing
            mod === CL && (frame = f)
        end
        frame === nothing && (frame = last(r.vst))
        key = string(frame.file, ":", frame.line, "  ", frame.linfo)
        counts[key] = get(counts, key, 0) + 1
    end
    @printf("\n%s -- %d reports across %d sites\n", name, length(reports), length(counts))
    for (k, v) in sort(collect(counts), by=x -> -x[2])
        @printf("  %5d  %s\n", v, first(k, 130))
    end
end

where_("compute_rhs! cartesian",   @report_opt target_modules=(CL,) compute_rhs!(s, Q, dQ))
where_("compute_rhs! cylindrical", @report_opt target_modules=(CL,) compute_rhs!(sc, Qc, dQc))
where_("compute_rhs! NSCBC",       @report_opt target_modules=(CL,) compute_rhs!(sn, Qn, dQn))
println("\ndone")
