# The sensor operators at a node-centred wall: how far the artificial-property
# smoother and the ringing detector depart from the periodic mirror of the same
# field near a reflecting boundary.
#
#   julia --project=. -t 1 bench/sensorwall.jl
#   julia --project=. -t 1 bench/sensorwall.jl ns=49,97 nodes=4
#   julia --project=. -t 1 bench/sensorwall.jl wall=folded
#
# Each case applies one operator on a slab between slip walls and the same
# operator on the periodic extension of that slab, 2(N − 1) cells of the same
# spacing whose restriction to [0, 1] is the wall run. A field exactly even
# about both wall nodes, or exactly odd for the wall-normal velocity component
# the detector also sees, continues past the wall as its own reflection, so the
# two runs agree to round-off wherever the wall closure represents that
# continuation and depart from each other where it does not.
#
# `wall=folded` puts a symmetry plane half a cell outside the end node instead,
# against the periodic extension on 2N cells offset by h/2. Nothing is planned
# at that face: the operator's own stencil runs to the edge over the mirror
# halo, so the round-off row is the expected reading at every node, and the
# node-centred run is what the wall rows have to reproduce.
#
# The printed number is |wall − periodic| at a node over the amplitude of the
# input field, and the two order columns are log2 of the ratio between
# successive resolutions, the spacing halving at each step. The input amplitude
# is the scale and not the operator's own output, whose magnitude falls with
# the resolution on a fixed field and would put every round-off row on a
# spurious order. The detector runs at weight power zero for the same reason,
# so no factor of the spacing enters. Round-off in this measure is near 1e-16.
#
# Cases: the detector `detect_sum!` under `:delta4` and under `:d8`, on the even
# field and on the odd one, and the smoother `smooth!` under `:gaussian` and
# under `:compact`, on the even field alone, every field the smoother sees being
# a detector output past an absolute value and so even at a wall.

using CompactLES
using Printf
const CL = CompactLES

const PER3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
const WALL = ((SlipWallBC(), SlipWallBC()), PER3[2], PER3[3])
const PLANE = ((SymmetryPlaneBC(), SymmetryPlaneBC()), PER3[2], PER3[3])

# Three modes each, so the operators are exercised well below and near the
# grid scale at once. Both are periodic with period 2 and reflect about the
# nodes at x = 0 and x = 1.
even_field(x) = cospi(x) + 0.5cospi(5x) + 0.1cospi(13x)
odd_field(x) = sinpi(x) + 0.1sinpi(13x)

# At a node-centred wall the periodic extension is 2(N − 1) cells sharing the
# wall run's origin. At a symmetry plane the mirror lies half a cell outside
# node 1, so the extension is 2N cells of the same spacing with its origin at
# h/2, and node i of the wall run is again node i of the periodic run.
build(N, art, folded) = folded ?
    (Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), art=art, bcs=PLANE),
     Solver(n_global=(2N, 1, 1), L_domain=(2.0, 1.0, 1.0), art=art, bcs=PER3,
            origin=(0.5 / N, 0.0, 0.0))) :
    (Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), art=art, bcs=WALL),
     Solver(n_global=(2(N - 1), 1, 1), L_domain=(2.0, 1.0, 1.0), art=art,
            bcs=PER3))

function load(s, fn)
    f = CL.field(s.decomp)
    pad = s.decomp.n_halo_d[1]
    for i in 1:s.decomp.n_local[1]
        f[i+pad, 1, 1] = fn(CL.xcoord(s, 1, i))
    end
    CL.exchange_halos!(f, s.decomp)
    return f
end

line(s, a) = (pad = s.decomp.n_halo_d[1];
              [a[i+pad, 1, 1] for i in 1:s.decomp.n_local[1]])

# The scaled departure at the first `nodes` nodes of each wall. Node N of the
# wall run is node N of the periodic run, the two grids sharing their origin
# and spacing.
windows(N, w, p, scale, nodes) =
    ([abs(w[i] - p[i]) / scale for i in 1:nodes],
     [abs(w[N+1-i] - p[N+1-i]) / scale for i in 1:nodes])

# The mirror sign reaches a node-centred wall as `wall_parity`, which plans the
# detector's own reflected rows, and a symmetry plane as `parity`, the sign the
# fold continues the field with; the two are the same statement about the field
# at boundaries of different kinds.
function detector_case(N, detector, fn, σw, nodes, folded)
    art = ArtificialProperties(detector=detector)
    sw, sp = build(N, art, folded)
    gw, gp = load(sw, fn), load(sp, fn)
    ow, op = CL.field(sw.decomp), CL.field(sp.decomp)
    folded ? CL.detect_sum!(ow, gw, sw, 0; parity=(σw, 1, 1)) :
             CL.detect_sum!(ow, gw, sw, 0; wall_parity=(σw, 1, 1))
    CL.detect_sum!(op, gp, sp, 0)
    return windows(N, line(sw, ow), line(sp, op),
                   maximum(abs, line(sp, gp)), nodes)
end

function smoother_case(N, smoother, nodes, folded)
    art = ArtificialProperties(smoother=smoother)
    sw, sp = build(N, art, folded)
    fw, fp = load(sw, even_field), load(sp, even_field)
    scale = maximum(abs, line(sp, fp))
    CL.smooth!(fw, sw)
    CL.smooth!(fp, sp)
    return windows(N, line(sw, fw), line(sp, fp), scale, nodes)
end

order(a, b) = (a > 0 && b > 0) ? log2(a / b) : NaN

function report(name, ns, walls, nodes)
    println(name)
    print("  node")
    for N in ns
        @printf("%13s", "N=$N")
    end
    println("      orders")
    for i in 1:nodes
        @printf("  %4d", i)
        vals = [w[i] for w in walls]
        for v in vals
            @printf("%13.3e", v)
        end
        print("   ")
        for k in 1:(length(vals)-1)
            @printf(" %6.2f", order(vals[k], vals[k+1]))
        end
        println()
    end
end

function main()
    opt = CL.script_args(ARGS, (ns="49,97,193", nodes=6, wall="node"))
    ns = parse.(Int, split(opt.ns, ','))
    nodes = opt.nodes
    opt.wall in ("node", "folded") || error("wall must be node or folded")
    folded = opt.wall == "folded"
    cases = (("detector :delta4, even field", N -> detector_case(N, :delta4,
                                                                 even_field, 1,
                                                                 nodes, folded)),
             ("detector :delta4, odd field", N -> detector_case(N, :delta4,
                                                                odd_field, -1,
                                                                nodes, folded)),
             ("detector :d8, even field", N -> detector_case(N, :d8, even_field,
                                                             1, nodes, folded)),
             ("detector :d8, odd field", N -> detector_case(N, :d8, odd_field,
                                                            -1, nodes, folded)),
             ("smoother :gaussian, even field",
              N -> smoother_case(N, :gaussian, nodes, folded)),
             ("smoother :compact, even field",
              N -> smoother_case(N, :compact, nodes, folded)))
    face = folded ? "a symmetry plane" : "a node-centred wall"
    println("relative departure from the periodic mirror at $face, ns = ",
            join(ns, ", "))
    for (name, run) in cases
        results = [run(N) for N in ns]
        println()
        report(name * ", low wall", ns, [r[1] for r in results], nodes)
        report(name * ", high wall", ns, [r[2] for r in results], nodes)
    end
end

main()
