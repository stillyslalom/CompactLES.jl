# The artificial bulk viscosity on anisotropic grids: the aspect-ratio
# penalty of the scalar β* in the step, and the relaxed filter's pass count,
# which followed it until the weight became directional (roadmap N5a).
#
#   julia --project=. -t 16 bench/anisotropic.jl                      # both cases
#   julia --project=. -t 16 bench/anisotropic.jl aligned ar=1,4,16
#   julia --project=. -t 16 bench/anisotropic.jl aligned ar=16 filter_cfl=5
#   julia --project=. -t 16 bench/anisotropic.jl cartesian N=64 ar=1,2,4
#   julia --project=. -t 16 bench/anisotropic.jl cartesian N=24 ar=4 p0=1e-2 tfinal=0.06
#
# Sweeps: aligned cartesian. Settings (`key=value`): N (points per unit
# length, or per half-side on the plane), ar (the aspect ratios, a comma
# list), filter_cfl, cfl, p0 (the ambient pressure of the plane case),
# tfinal (its end time).
#
# Scratch tooling, like everything else in bench/: it prints tables, asserts
# nothing, and is not part of the gate. The cases are `noh_aligned` and
# `noh_cartesian` in test/cases.jl. The first is the planar Noh implosion
# along the coarse dimension of a grid of aspect ratio `ar`, on which the
# solution is the one-dimensional profile at every station and the step is
# the only thing the grid's fine direction can change; the second is the
# cylindrical implosion on the Cartesian plane, a curved front oblique to the
# grid at every angle, with the exact solution along every cut. Each row
# prints the step count, the wall time, which rate limited the last step, the
# plateau, the deficit at the symmetry point, the front position, the L1
# density error, the largest transverse variation (the aligned case) or the
# front width along the fine axis, the coarse axis and the diagonal and the
# largest pre-shock vorticity and dilatation at the end (the plane case). The
# width reads NaN while the plateau is under 14.8, the 10% level of the
# 16 → 4 jump it measures, which at N = 24 it is. A run that fails prints the
# failure and its time instead.
#
# The directional forms this script was written to compare against were
# implemented for the measurement and not retained; the record and the two
# forms' definitions are in reference/CALIBRATION_APPENDIX.md, "Directional
# bulk viscosity on anisotropic grids"; the aligned sweep under the
# directional filter weight is recorded in its last subsection. Update that
# section when this is re-run under different settings.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const OPTS = CompactLES.script_args(filter(a -> occursin('=', a), ARGS),
                        (N = 0, ar = "1,2,4", filter_cfl = 0.35, cfl = NC_CFL,
                         p0 = NOH_P0, tfinal = NOH_T))
const NAMES = filter(a -> !occursin('=', a), ARGS)
const WHICH = isempty(NAMES) ? ["aligned", "cartesian"] : NAMES
want(name) = name in WHICH
const ARS = parse.(Int, split(OPTS.ar, ','))

# The largest |ω_z| and |∇·u| over the pre-shock region r > 1.2 R_s at the
# end of a plane run, by centered differences of the primitive velocity: the
# exact pre-shock flow is irrotational, and the bulk force ∇(β*∇·u) is a
# gradient and keeps it so. A form of the artificial stress that is not a
# gradient shows here first.
function preshock_curl(r, tfin)
    s = r.solver
    nx, ny = s.decomp.n_local[1], s.decomp.n_local[2]
    Rs = (NOH_G - 1) / 2 * tfin
    hx = xcoord(s, 1, 2) - xcoord(s, 1, 1)
    hy = xcoord(s, 2, 2) - xcoord(s, 2, 1)
    ex, ey = CartesianIndex(1, 0, 0), CartesianIndex(0, 1, 0)
    wmax = 0.0; dmax = 0.0
    for j in 2:ny-1, i in 2:nx-1
        hypot(xcoord(s, 1, i), xcoord(s, 2, j)) > 1.2Rs || continue
        I = gidx(s, i, j, 1)
        dvdx = (s.v[I + ex] - s.v[I - ex]) / 2hx
        dudy = (s.u[I + ey] - s.u[I - ey]) / 2hy
        dudx = (s.u[I + ex] - s.u[I - ex]) / 2hx
        dvdy = (s.v[I + ey] - s.v[I - ey]) / 2hy
        wmax = max(wmax, abs(dvdx - dudy))
        dmax = max(dmax, abs(dudx + dvdy))
    end
    return wmax, dmax
end

function failed(label, e)
    e isa SolverFailure || rethrow(e)
    @printf("  %-4s FAILED %s at step %d, t = %.4f\n", label, e.reason, e.step, e.t)
end

function aligned_sweep()
    N = OPTS.N == 0 ? Dict(NOH_N)[1] : OPTS.N
    println("\n=== planar Noh along the coarse dimension, N = $N, cfl = $(OPTS.cfl), " *
            "filter_cfl = $(OPTS.filter_cfl) ===")
    println("  AR   steps     wall  limit      plateau  deficit  shock    transverse")
    for AR in ARS
        label = @sprintf("%-4d", AR)
        r = try
            noh_aligned(; N, AR, cfl=OPTS.cfl, filter_cfl=OPTS.filter_cfl)
        catch e
            failed(label, e); continue
        end
        plat, deficit, Rnum, _ = noh_metrics(r.y, r.rho, 1)
        @printf("  %s %6d %7.1fs  %-9s  %.4f  %+.0f%%     %.4f   %.1e\n",
                label, r.steps, r.wall, r.kind, plat, 100deficit, Rnum, r.uniformity)
        flush(stdout)
    end
end

function cartesian_sweep()
    N = OPTS.N == 0 ? NC_N : OPTS.N
    println("\n=== cylindrical Noh on the Cartesian plane, N = $N per half-side, " *
            "p0 = $(OPTS.p0), to t = $(OPTS.tfinal), cfl = $(OPTS.cfl), " *
            "filter_cfl = $(OPTS.filter_cfl) ===")
    println("  AR   steps     wall  limit      plateau  center   front    L1 rho    " *
            "width x/y/diag        pre-shock |omega| |div u|")
    for AR in ARS
        label = @sprintf("%-4d", AR)
        r = try
            noh_cartesian(; N, AR, cfl=OPTS.cfl, filter_cfl=OPTS.filter_cfl,
                          p0=OPTS.p0, tfinal=OPTS.tfinal)
        catch e
            failed(label, e); continue
        end
        w, d = preshock_curl(r, OPTS.tfinal)
        @printf("  %s %6d %7.1fs  %-9s  %.3f   %+.0f%%    %.4f   %.3e  ",
                label, r.steps, r.wall, r.kind, r.plateau, 100r.deficit,
                r.cut_diag.front, r.l1)
        @printf("%.4f %.4f %.4f  %.2e %.2e\n", r.cut_x.width, r.cut_y.width,
                r.cut_diag.width, w, d)
        flush(stdout)
    end
end

want("aligned") && aligned_sweep()
want("cartesian") && cartesian_sweep()
