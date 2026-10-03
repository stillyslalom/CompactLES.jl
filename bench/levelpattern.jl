# Grid-scale pattern a refined level leaves in a shocked or sharpened field,
# against the uniform grid on the level's nodes. Two problems
# (`problems=`, comma-separated), each run in the configurations
# (`configs=`, comma-separated)
#
#   uniform  the uniform grid at the refined spacing, whose nodes coincide
#            with the level's
#   box      one fixed level over a region as a single patch
#   tiles    the same cover as lattice tiles of edge `tile`
#   regrid   (shock only) tiles of edge `tile` placed and moved by the
#            artificial-diffusivity tag (`sensor`), the density tag off,
#            `buffer` nodes of tag buffer
#
#   shock      a spherical shock converging on an r-z quadrant (the axis at
#              r = 0, a symmetry plane at z = 0, slip walls at r = 1 and
#              z = 1) from a tanh-blended high-pressure shell at radius 0.7
#              of width `width`, on a root of `n`² nodes, to `t` (shock
#              radius R ≈ 0.17 at the defaults); the level covers r, z < 0.8
#              and the uniform grid has 3n − 1 nodes per dimension, which the
#              half-cell offset of the axis and the plane puts on the
#              level's. The measure is the mean over the shell
#              R + 0.02 < ϱ < R + 0.08 behind the shock of
#              (|δ⁴_r ρ| + |δ⁴_z ρ|)/ρ, the undivided fourth differences
#              along r and z at one fine spacing, on the composite lattice of
#              the finest level's nodes (a node no fine patch holds is left
#              out; the axis and the plane mirror the stencil). R is the
#              first node along the diagonal r = z whose pressure exceeds 3.
#   interface  a helium bubble of radius 40 mm in air, edge width
#              `bubble_w`, carried at (300, 300) m/s through a periodic box
#              of `bubble_n`² root nodes 10 mm apart for 60 mm along each
#              axis, with the interface sharpening flux at `C_sharpen`; the
#              level covers the bubble's path and the uniform grid has
#              3 `bubble_n` nodes per dimension. The measures, over the
#              level's nodes where the uniform run's helium mass fraction is
#              between 0.02 and 0.98: the mean of (|δ⁴_x Y| + |δ⁴_y Y|) for
#              the helium mass fraction Y, and the mean and largest
#              |Y − Y_uniform|.
#
# A smooth field gives a pattern value set by its own curvature, the uniform
# run's; a checkerboard at tile faces or at a moving cover's edge raises it.
# Each refined configuration's pattern is reported as a ratio to the uniform
# run's value, which the process runs first.
#
# `mask=false` filters the parent's covered nodes with their residual as well
# (`MASK_CHILD_RESIDUAL[] = false`); `dsnap > 0` prints the shock's measure
# every `dsnap` of simulated time as well as at the end. The shock runs
# accept a state with a few negative internal energies at the axis near the
# focus (`validity = :permissive`) and report the count.
#
#   julia --project=. -t 8 bench/levelpattern.jl
#   julia --project=. -t 8 bench/levelpattern.jl configs=uniform,regrid buffer=8
#   julia --project=. -t 8 bench/levelpattern.jl problems=interface
#
# At the defaults the shock takes about two minutes after the package loads
# on the workstation, the tiles most of it, and the interface about one.
# Serial only. The measurements are in reference/CALIBRATION_APPENDIX.md
# under this script's name.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES.Regions
using Printf
const CL = CompactLES

const opt = CL.script_args(ARGS, (
    problems = "shock", configs = "uniform,box,tiles,regrid", n = 64, t = 0.165,
    width = 0.024, tile = 8, sensor = 0.5, buffer = 4, mask = true, dsnap = 0.0,
    bubble_n = 48, bubble_w = 0.01, bubble_tile = 8, C_sharpen = 1.0))

# The finest level's values of `value(snapshot, i, j)` on its node lattice
# over the dimensions `dims` (NaN where no patch of that level holds the
# node), and the lattice coordinates.
function finest_lattice(solver, states, fields, dims, value)
    snaps = states isa Vector ? field_snapshot(solver, states; fields) :
            [field_snapshot(solver, states; fields)]
    top = maximum(s.level for s in snaps)
    fine = filter(s -> s.level == top, snaps)
    a, b = dims
    h = fine[1].coords[a][2] - fine[1].coords[a][1]
    x0 = mod(fine[1].coords[a][1], h)
    x0 > h * (1 - 1e-6) && (x0 = 0.0)
    m = round(Int, maximum(maximum(s.coords[a]) for s in fine) / h) + 1
    F = fill(NaN, m, m)
    index(x) = round(Int, (x - x0) / h) + 1
    for s in fine
        is, js = index.(s.coords[a]), index.(s.coords[b])
        maximum(abs, x0 .+ (is .- 1) .* h .- s.coords[a]) < 1e-6h ||
            error("a patch's nodes are off the lattice")
        for (q, j) in enumerate(js), (p, i) in enumerate(is)
            1 <= i <= m && 1 <= j <= m && (F[i, j] = value(s, p, q))
        end
    end
    return F, x0 .+ (0:m-1) .* h
end

# The undivided fourth difference of `F` along each lattice dimension at
# (i, j), reading node (i', j') through `at`.
function fourth_differences(at, i, j)
    di = at(i - 2, j) - 4at(i - 1, j) + 6at(i, j) - 4at(i + 1, j) + at(i + 2, j)
    dj = at(i, j - 2) - 4at(i, j - 1) + 6at(i, j) - 4at(i, j + 1) + at(i, j + 2)
    return abs(di) + abs(dj)
end

# ---------------------------------------------------------------------------
# Shock

function shock_case()
    ic = (r, theta, z) -> begin
        d = tanh_blend(hypot(r, z), 0.7, opt.width)
        Prim(rho = 1.0 + 3.0 * d, p = 0.1 + 19.9 * d)
    end
    problem = Problem(name = "converging shock",
                      eos = IdealSpecies("gas"; R = 1.0, gamma = 1.4),
                      metric = CylindricalMetric(),
                      domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                      bcs = ((AxisBC(), SlipWallBC()), PeriodicBC(),
                             (SymmetryPlaneBC(), SlipWallBC())), ic = ic)
    n = opt.n
    function numerics(config)
        config == "uniform" && return Numerics(n_global = (3n - 1, 1, 3n - 1))
        box = Box((0.0, 0.0, 0.0), (0.8, 1.0, 0.8))
        amr = config == "box" ? AMR(initial = box) :
              config == "tiles" ? AMR(initial = box, tile = opt.tile) :
              config == "regrid" ? AMR(initial = :sensor, tile = opt.tile,
                                       tag_threshold = Inf,
                                       tag_sensor_threshold = opt.sensor,
                                       tag_buffer = opt.buffer) :
              error("unknown configuration $config")
        return Numerics(n_global = (n, 1, n), amr = amr)
    end
    function measure(solver, states)
        rho, x = finest_lattice(solver, states, (:rho,), (1, 3),
                                (s, p, q) -> s[:rho][p, 1, q])
        p, _ = finest_lattice(solver, states, (:p,), (1, 3),
                              (s, a, b) -> s[:p][a, 1, b])
        m = length(x)
        R = NaN
        for i in 1:m
            isfinite(p[i, i]) && p[i, i] > 3 && (R = hypot(x[i], x[i]); break)
        end
        # Node j < 1 is node 1 − j across the axis or the plane.
        at(i, j) = (ii = i < 1 ? 1 - i : i; jj = j < 1 ? 1 - j : j;
                    ii > m || jj > m ? NaN : rho[ii, jj])
        total, count = 0.0, 0
        for j in 1:m, i in 1:m
            R + 0.02 < hypot(x[i], x[j]) < R + 0.08 || continue
            value = fourth_differences(at, i, j) / rho[i, j]
            isfinite(value) || continue
            total += value
            count += 1
        end
        return (; pattern = total / max(count, 1), count, R)
    end
    function run(config, reference)
        solver, states = setup(problem, numerics(config))
        mass0 = volume_integral(solver, states, :rho)
        callback = opt.dsnap > 0 ?
            Callback(EveryTime(opt.dsnap), (s, Q) -> begin
                r = measure(s, Q)
                @printf("  %-8s t = %.3f  R = %.4f  pattern %.3e over %d nodes\n",
                        config, s.t, r.R, r.pattern, r.count)
                false
            end) : nothing
        wall = @elapsed run!(solver, states; tfinal = opt.t, callback = callback,
                             control = StepControl(validity = :permissive))
        r = measure(solver, states)
        bad = state_report(solver, states).inadmissible
        ratio = reference === nothing ? 1.0 : r.pattern / reference.pattern
        npatch = states isa Vector ? length(states) : 1
        @printf("shock      %-7s %4d steps  R = %.4f  pattern %.3e over %4d nodes  \
                 ratio %.2f  patches %3d  mass change % .1e  inadmissible %d  %.0f s\n",
                config, solver.step, r.R, r.pattern, r.count, ratio, npatch,
                volume_integral(solver, states, :rho) / mass0 - 1, bad, wall)
        println("row,shock,", config, ",ratio,", ratio)
        return r
    end
    return run
end

# ---------------------------------------------------------------------------
# Interface

function interface_case()
    n = opt.bubble_n
    L = n * 0.01
    U, R, w, dist = 300.0, 0.04, opt.bubble_w, 0.06
    c0 = (0.5L - 0.03, 0.5L - 0.03)
    eos = IdealMixture(["Air", "He"])
    function ic(x, y, z)
        b = (1 - tanh((hypot(x - c0[1], y - c0[2]) - R) / w)) / 2
        return Prim(Y = mass_fractions(eos, "Air" => 1 - b, "He" => b; basis = :mole),
                    p = 101_325.0, T_ion = 300.0, u = (U, U, 0.0))
    end
    problem = Problem(name = "sharpened bubble", eos = eos,
                      domain = ((0.0, L), (0.0, L), (0.0, 1.0)),
                      bcs = (PeriodicBC(), PeriodicBC(), PeriodicBC()), ic = ic)
    art = ArtificialProperties(C_sharpen = opt.C_sharpen)
    function numerics(config)
        config == "uniform" && return Numerics(n_global = (3n, 3n, 1), art = art)
        reach = R + 6w
        box = Box((c0[1] - reach, c0[2] - reach, 0.0),
                  (c0[1] + dist + reach, c0[2] + dist + reach, 1.0))
        amr = config == "box" ? AMR(initial = box) :
              config == "tiles" ? AMR(initial = box, tile = opt.bubble_tile) :
              error("configuration $config does not apply to the interface")
        return Numerics(n_global = (n, n, 1), amr = amr, art = art)
    end
    helium(solver, states) = finest_lattice(solver, states, (:Y,), (1, 2),
                                            (s, p, q) -> s[:Y][p, q, 1, 2])
    function run(config, reference)
        solver, states = setup(problem, numerics(config))
        wall = @elapsed run!(solver, states; tfinal = dist / U)
        Y, _ = helium(solver, states)
        Yu = reference === nothing ? Y : reference.Y
        at(i, j) = checkbounds(Bool, Y, i, j) ? Y[i, j] : NaN
        total, count, diff, worst = 0.0, 0, 0.0, 0.0
        for j in axes(Y, 2), i in axes(Y, 1)
            0.02 < Yu[i, j] < 0.98 || continue
            value = fourth_differences(at, i, j)
            isfinite(value) || continue
            total += value
            count += 1
            diff += abs(Y[i, j] - Yu[i, j])
            worst = max(worst, abs(Y[i, j] - Yu[i, j]))
        end
        pattern = total / max(count, 1)
        ratio = reference === nothing ? 1.0 : pattern / reference.pattern
        @printf("interface  %-7s %4d steps  pattern %.3e over %4d nodes  ratio %.2f  \
                 |Y - Y_uniform| mean %.2e max %.2e  %.0f s\n",
                config, solver.step, pattern, count, ratio, diff / max(count, 1), worst,
                wall)
        println("row,interface,", config, ",ratio,", ratio)
        return (; Y, pattern)
    end
    return run
end

function main()
    CL.MASK_CHILD_RESIDUAL[] = opt.mask
    for name in split(opt.problems, ',')
        run = name == "shock" ? shock_case() : name == "interface" ? interface_case() :
              error("unknown problem $name")
        configs = split(opt.configs, ',')
        name == "interface" && (configs = filter(!=("regrid"), configs))
        reference = nothing
        for config in configs
            r = run(config, config == "uniform" ? nothing : reference)
            config == "uniform" && (reference = r)
        end
    end
end

main()
