# The sensor operators at a node-centred wall: how far the artificial-property
# smoother and the ringing detector depart from the periodic mirror of the same
# field near a reflecting boundary.
#
#   julia --project=. -t 1 bench/sensorwall.jl
#   julia --project=. -t 1 bench/sensorwall.jl ns=49,97 nodes=4
#
# Each case applies one operator on a slab between slip walls and the same
# operator on the periodic extension of that slab, 2(N − 1) cells of the same
# spacing whose restriction to [0, 1] is the wall run. A field exactly even
# about both wall nodes, or exactly odd for the wall-normal velocity component
# the detector also sees, continues past the wall as its own reflection, so the
# two runs agree to round-off wherever the wall closure represents that
# continuation and depart from each other where it does not.
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

# Three modes each, so the operators are exercised well below and near the
# grid scale at once. Both are periodic with period 2 and reflect about the
# nodes at x = 0 and x = 1.
even_field(x) = cospi(x) + 0.5cospi(5x) + 0.1cospi(13x)
odd_field(x) = sinpi(x) + 0.1sinpi(13x)

build(N, art) =
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

function detector_case(N, detector, fn, σw, nodes)
    art = ArtParams(detector=detector)
    sw, sp = build(N, art)
    gw, gp = load(sw, fn), load(sp, fn)
    ow, op = CL.field(sw.decomp), CL.field(sp.decomp)
    CL.detect_sum!(ow, gw, sw, 0; wall_parity=(σw, 1, 1))
    CL.detect_sum!(op, gp, sp, 0)
    return windows(N, line(sw, ow), line(sp, op),
                   maximum(abs, line(sp, gp)), nodes)
end

function smoother_case(N, smoother, nodes)
    art = ArtParams(smoother=smoother)
    sw, sp = build(N, art)
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
    opt = CL.script_args(ARGS, (ns="49,97,193", nodes=6))
    ns = parse.(Int, split(opt.ns, ','))
    nodes = opt.nodes
    cases = (("detector :delta4, even field", N -> detector_case(N, :delta4,
                                                                 even_field, 1,
                                                                 nodes)),
             ("detector :delta4, odd field", N -> detector_case(N, :delta4,
                                                                odd_field, -1,
                                                                nodes)),
             ("detector :d8, even field", N -> detector_case(N, :d8, even_field,
                                                             1, nodes)),
             ("detector :d8, odd field", N -> detector_case(N, :d8, odd_field,
                                                            -1, nodes)),
             ("smoother :gaussian, even field",
              N -> smoother_case(N, :gaussian, nodes)),
             ("smoother :compact, even field",
              N -> smoother_case(N, :compact, nodes)))
    println("relative departure from the periodic mirror, ns = ", join(ns, ", "))
    for (name, run) in cases
        results = [run(N) for N in ns]
        println()
        report(name * ", low wall", ns, [r[1] for r in results], nodes)
        report(name * ", high wall", ns, [r[2] for r in results], nodes)
    end
end

main()
