# Disturbances a moving refined level carries away from the feature it follows,
# against the uniform grid at the refined spacing. Two scaled-down tutorial
# configurations, each run as
#
#   coarse    the root grid alone
#   fine      the uniform grid at the refined spacing
#   moving    the root with a regridded level following the feature
#   fixed     the root with a level fixed over the feature's whole path
#
# and, under `mask=both`, the two refined runs again with the parent's filter
# pass taking the residual of the covered nodes as well
# (`CompactLES.MASK_CHILD_RESIDUAL[] = false`, suffix `-plain`). The refined
# runs take the parent's derivative mask, which the package leaves off
# (`CompactLES.MASK_CHILD_DERIVATIVE[]`), and under `derivative_mask=both`
# run again with the plain derivatives (suffix `-plainD`); `derivative_mask=off`
# gives the package default alone.
#
#   bubble  the Advected bubbles tutorial reduced to one helium bubble in a
#           periodic box of `bubble_n`² root nodes 10 mm apart, the edge width
#           `bubble_w` (one root spacing by default, which the root does not
#           resolve), lattice tiles of edge `bubble_tile`, carried
#           `bubble_dist` along each axis. The exact solution is the initial
#           state shifted by the wind. Reported: the largest |X − exact| of the
#           helium mole fraction over composite nodes at the edge (within 4w of
#           the radius) and away from it (beyond 4w, split into root and
#           refined nodes), and the relative change of the helium mass over
#           the run and across the regrids alone.
#   shock   the Imploding shock tutorial on a root of `shock_n` nodes with one
#           subcycled box placed by the density sensor. The gas ahead of the
#           shock is at rest at density 1, so the disturbance is the largest
#           |ρ − 1| over composite nodes more than `ahead` (and more than 0.05)
#           inside the shock radius, over `shock_rmin` < R < 0.3. The box
#           leads the shock by `shock_buffer` root nodes (`tag_buffer`);
#           `limiter=true` turns the positivity limiter on, and `fixed_inner`
#           starts the fixed level at that radius, so that its inner face
#           stands ahead of the shock. Reported with the count of
#           inadmissible nodes at the end.
#
#   julia --project=. -t 1 bench/movinglevel.jl
#   julia --project=. -t 1 bench/movinglevel.jl problems=bubble bubble_w=0.03
#
# One process takes about a minute after the package loads. Serial only. The
# measurements are in reference/CALIBRATION_APPENDIX.md under this script's
# name.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES

const opt = CL.script_args(ARGS, (
    problems = "bubble,shock", configs = "coarse,fine,moving,fixed", mask = "both",
    derivative_mask = "on",
    bubble_n = 48, bubble_w = 0.01, bubble_tile = 12, bubble_dist = 0.06,
    shock_n = 256, shock_t = 0.221, ahead = 0.02, shock_buffer = 4, limiter = false,
    fixed_inner = 0.0, shock_rmin = 0.02))

# ---------------------------------------------------------------------------
# Bubble

function bubble_case()
    n = opt.bubble_n
    L = n * 0.01
    U, R, w = 300.0, 0.04, opt.bubble_w
    c0 = (0.5L - 0.03, 0.5L - 0.03)
    wrapd(d) = d - L * round(d / L)
    dist(x, y, t) = hypot(wrapd(x - c0[1] - U * t), wrapd(y - c0[2] - U * t))
    bubble(x, y, t) = (1 - tanh((dist(x, y, t) - R) / w)) / 2
    eos = IdealMixture(["Air", "He"])
    state(x, y, t) = Prim(Y=mass_fractions(eos, "Air" => 1 - bubble(x, y, t),
                                           "He" => bubble(x, y, t); basis=:mole),
                          p=101_325.0, T_ion=300.0, u=(U, U, 0.0))
    problem = Problem(name="bubble", eos=eos, domain=((0.0, L), (0.0, L), (0.0, 1.0)),
                      bcs=(PeriodicBC(), PeriodicBC(), PeriodicBC()),
                      ic=(x, y, z) -> state(x, y, 0.0))
    tiles = AMR(tile=opt.bubble_tile)
    # The path: within R + 6w of the segment the center travels.
    path(x, y, z) = begin
        s = clamp(((x - c0[1]) + (y - c0[2])) / (2 * opt.bubble_dist), 0, 1)
        hypot(x - c0[1] - s * opt.bubble_dist, y - c0[2] - s * opt.bubble_dist) < R + 6w
    end
    numerics(config) = config == "coarse" ? Numerics(n_global=(n, n, 1)) :
                       config == "fine" ? Numerics(n_global=(3n, 3n, 1)) :
                       config == "moving" ? Numerics(n_global=(n, n, 1), amr=tiles) :
                       Numerics(n_global=(n, n, 1), amr=AMR(tiles; initial=path))
    he_mass(solver, states::Vector) =
        volume_integral(solver, [view(parent(Q), :, :, :, 2) for Q in states])
    he_mass(solver, Q) = volume_integral(solver, view(parent(Q), :, :, :, 2))
    function errors(solver, states)
        snaps = field_snapshot(solver, states; fields=(:X,))
        snaps isa Vector || (snaps = [snaps])
        e = Dict(:edge => 0.0, :root => 0.0, :level => 0.0)
        for snap in snaps
            x, y, X = snap.coords[1], snap.coords[2], snap[:X]
            for j in eachindex(y), i in eachindex(x)
                snap.covered[i, j, 1] && continue
                err = abs(X[i, j, 1, 2] - bubble(x[i], y[j], solver.t))
                k = dist(x[i], y[j], solver.t) < R + 4w ? :edge :
                    snap.level == 0 ? :root : :level
                e[k] = max(e[k], err)
            end
        end
        return e
    end
    function run(config)
        solver, states = setup(problem, numerics(config))
        m0 = he_mass(solver, states)
        layout = Ref(states isa Vector ? level_regions(solver, 1) : BlockRegion[])
        last = Ref(m0)
        regrid = Ref(0.0)
        changes = Ref(0)
        # `run!` has no hook between a regrid check and the step after it, so
        # a change of layout is detected after that step, and the mass change
        # of the step is counted whole as the regrid's.
        cb = Callback(EveryStep(1), function (s, st)
            m = he_mass(s, st)
            if st isa Vector
                now = level_regions(s, 1)
                if now != layout[]
                    regrid[] += m - last[]
                    changes[] += 1
                    layout[] = now
                end
            end
            last[] = m
            nothing
        end)
        run!(solver, states; tfinal=opt.bubble_dist / U, nmax=5000, callback=cb)
        e = errors(solver, states)
        return (; steps=solver.step, e, dm=he_mass(solver, states) / m0 - 1,
                regrid=regrid[] / m0, changes=changes[])
    end
    header = @sprintf("bubble: root %d², w = %.3f m (%.2f root spacings), tile %d, %.3f m",
                      n, w, w / 0.01, opt.bubble_tile, opt.bubble_dist)
    function report(label, r)
        @printf("  %-14s %5d steps  edge %.2e  away: root %.2e  level %.2e  \
                 He mass %+.1e (layout-change steps %+.1e, %d)\n", label, r.steps,
                r.e[:edge], r.e[:root], r.e[:level], r.dm, r.regrid, r.changes)
        @printf("row,bubble,%s,away_level,%.3e\n", label, r.e[:level])
    end
    return (; header, run, report)
end

# ---------------------------------------------------------------------------
# Shock

function shock_case()
    n = opt.shock_n
    problem = Problem(name="imploding shock",
        eos=IdealSpecies("gas"; R=1.0, gamma=1.4), metric=CylindricalMetric(),
        domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        bcs=((AxisBC(), SlipWallBC()), PeriodicBC(), PeriodicBC()),
        ic=(r, θ, z) -> begin
            drive = tanh_blend(r, 0.7, 0.012)
            Prim(rho=1.0 + 3.0 * drive, p=0.1 + 19.9 * drive)
        end)
    box = AMR(initial=:sensor, subcycle=true, tag_buffer=opt.shock_buffer)
    # h = 1/(N − 1/2) with node 1 at h/2, so the grid at h/3 has 3N − 1 nodes.
    # The converging shock's state at the axis near t_c can leave the internal
    # energy negative at a few nodes, which the default validity rejects; the
    # disturbance ahead is measured before that.
    control = StepControl(validity=:permissive)
    positivity_limiter = opt.limiter
    numerics(config) =
        config == "coarse" ? Numerics(n_global=(n, 1, 1); control, positivity_limiter) :
        config == "fine" ? Numerics(n_global=(3n - 1, 1, 1); control, positivity_limiter) :
        config == "moving" ? Numerics(n_global=(n, 1, 1), amr=box; control,
                                      positivity_limiter) :
        Numerics(n_global=(n, 1, 1),
                 amr=AMR(box; initial=(r, θ, z) -> opt.fixed_inner < r < 0.76);
                 control, positivity_limiter)
    function radius(snaps)
        for snap in Iterators.reverse(snaps)
            r, p = snap.coords[1], vec(snap[:p])
            i = findfirst(>(3.0), p)
            (i === nothing || i == 1) && continue
            return r[i-1] + (r[i] - r[i-1]) * (3.0 - p[i-1]) / (p[i] - p[i-1])
        end
        return NaN
    end
    function run(config)
        solver, states = setup(problem, numerics(config))
        e = Dict(:root => 0.0, :level => 0.0, :root_far => 0.0, :level_far => 0.0)
        changes = Ref(0)
        layout = Ref(states isa Vector ? level_regions(solver, 1) : BlockRegion[])
        cb = Callback(EveryStep(1), function (s, st)
            snaps = field_snapshot(s, st; fields=(:rho, :p))
            snaps isa Vector || (snaps = [snaps])
            if st isa Vector && level_regions(s, 1) != layout[]
                changes[] += 1
                layout[] = level_regions(s, 1)
            end
            R = radius(snaps)
            opt.shock_rmin < R < 0.3 || return nothing
            for snap in snaps
                r, rho = snap.coords[1], vec(snap[:rho])
                for i in eachindex(r)
                    (snap.covered[i, 1, 1] || r[i] >= R - opt.ahead) && continue
                    k = snap.level == 0 ? :root : :level
                    e[k] = max(e[k], abs(rho[i] - 1))
                    r[i] < R - 0.05 && (e[Symbol(k, :_far)] =
                                            max(e[Symbol(k, :_far)], abs(rho[i] - 1)))
                end
            end
            nothing
        end)
        run!(solver, states; tfinal=opt.shock_t, nmax=20_000, callback=cb)
        bad = state_report(solver, states).inadmissible
        return (; steps=solver.step, e, changes=changes[], bad)
    end
    header = @sprintf("shock: root %d, to t = %.3f; |ρ − 1| beyond %.2f and 0.05 \
                       inside the shock, %.2f < R < 0.3; tag_buffer %d, limiter %s",
                      n, opt.shock_t, opt.ahead, opt.shock_rmin, opt.shock_buffer,
                      opt.limiter)
    function report(label, r)
        @printf("  %-14s %5d steps  beyond %.2f: root %.2e  level %.2e   \
                 beyond 0.05: root %.2e  level %.2e   layout changes %d  \
                 inadmissible %d\n",
                label, r.steps, opt.ahead, r.e[:root], r.e[:level], r.e[:root_far],
                r.e[:level_far], r.changes, r.bad)
        @printf("row,shock,%s,ahead_level,%.3e\n", label, r.e[:level])
    end
    return (; header, run, report)
end

function main()
    for name in split(opt.problems, ",")
        case = name == "bubble" ? bubble_case() : name == "shock" ? shock_case() :
               error("unknown problem '$name', want bubble or shock")
        println(case.header)
        choices(o) = o == "both" ? (true, false) : o == "on" ? (true,) : (false,)
        for config in split(opt.configs, ",")
            refined = config in ("moving", "fixed")
            masks = refined ? choices(opt.mask) : (true,)
            dmasks = refined ? choices(opt.derivative_mask) : (true,)
            for mask in masks, dmask in dmasks
                CL.MASK_CHILD_RESIDUAL[] = mask
                CL.MASK_CHILD_DERIVATIVE[] = dmask
                label = String(config) * (mask ? "" : "-plain") * (dmask ? "" : "-plainD")
                case.report(label, case.run(String(config)))
                flush(stdout)
            end
        end
        CL.MASK_CHILD_RESIDUAL[] = true
        CL.MASK_CHILD_DERIVATIVE[] = false
    end
end

main()
