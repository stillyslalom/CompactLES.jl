# The live level transfer by itself, apart from any divergence: the
# point-sample interpolation that fills a refined patch's shell and a freshly
# created fine region, the restriction, and repeated regrids, at each
# `level_interpolation_order`.
#
#   julia --project=. -t 1 bench/leveltransfer.jl [study=all] [ns=48,96,192]
#                                                 [orders=2,4,6,8] [regrids=16]
#
# `study` is one of, or a comma-separated list of,
#
#   values        exactness on a tensor polynomial of the order's degree less
#                 one, then the value error of a smooth 2-D field in the
#                 imposed shell (ghost ring and boundary planes) and in a
#                 freshly filled fine region
#   derivatives   the first derivative across the shell through the fine
#                 patch's gradient plans (extended interface rows), and the
#                 second derivative through the divergence plans applied to
#                 it, each against the same operator on exact shell data (the
#                 part the interpolation contributes) and against the exact
#                 derivative; the same after a fresh fill
#   restriction   the injection write-back against the fine coincident nodes,
#                 and the filtered restriction on point samples
#   regrid        a 1-D region moved back and forth `regrids` times with no
#                 evolution, the fine error after the first and the last
#   positivity    a fresh fill of a tanh density step one coarse cell wide,
#                 the undershoot below the low state and the overshoot above
#                 the high one, as fractions of the jump; then the moving-
#                 region Sod gate of test/level_tests.jl
#
# Every field is analytic and the root is exact, so each number is the
# transfer's alone. Serial Float64; the 2-D fields are periodic on
# [0, 2π)² with the level fixed in physical space at 5L/12..7L/12 per
# dimension. Scratch tooling: it prints tables and asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES, Printf
using CompactLES: interior_index, xcoord
const CL = CompactLES
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this study on one rank")

const OPTS = CL.script_args(ARGS, (study="all", ns="48,96,192", orders="2,4,6,8",
                                   regrids=16))
const NS = parse.(Int, split(OPTS.ns, ','))
const ORDERS = parse.(Int, split(OPTS.orders, ','))
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
const W = 4                      # interface window, fine nodes

selected(name) = OPTS.study == "all" || name in split(OPTS.study, ',')

orders_of(hs, es) = [log(es[i] / es[i+1]) / log(hs[i] / hs[i+1]) for i in 1:length(es)-1]
fmt_orders(v) = join((@sprintf("%.2f", x) for x in v), " / ")

# --- the 2-D two-level layout ---------------------------------------------------

function layout2d(N, order; kw...)
    N % 12 == 0 || error("N = $N must be a multiple of 12")
    r = BlockRegion((5N ÷ 12, 5N ÷ 12, 0), (N ÷ 6 + 1, N ÷ 6 + 1, 1))
    solver = Solver(n_global=(N, N, 1), L_domain=(2pi, 2pi, 1.0), bcs=per3,
                    art=ArtificialProperties(enabled=false), filter_interval=0, refine=r,
                    level_interpolation_order=order; kw...)
    return solver, allocate_state(solver)
end

patch_solver(solver, k) = CL.PatchSolver(solver, getfield(solver, :patches)[k])

# Physical coordinates of a padded slot of patch `ps` (collapsed dims: 0 pad).
function slot_xy(ps, I)
    pad = ps.decomp.n_halo_d
    return (xcoord(ps, 1, I[1] - pad[1]), xcoord(ps, 2, I[2] - pad[2]))
end

# Region class of a padded slot of the fine patch: :ghost (outside the patch
# along some dimension), :plane (on an imposed boundary plane), :window
# (within W nodes of an end along `dim`, off the planes), :interior.
function slot_class(ps, I; dim=1)
    pad = ps.decomp.n_halo_d
    n = ps.decomp.n_local
    i, j = I[1] - pad[1], I[2] - pad[2]
    (i < 1 || i > n[1] || j < 1 || j > n[2]) && return :ghost
    (i == 1 || i == n[1] || j == 1 || j == n[2]) && return :plane
    g = dim == 1 ? i : j
    (g <= W || g > n[dim] - W) && return :window
    return :interior
end

"Write f(x, y) into component `c` of every padded slot of every patch."
function set_all!(solver, states, c, f)
    for k in eachindex(states)
        ps = patch_solver(solver, k)
        Q = states[k]
        for I in CartesianIndices(size(Q)[1:3])
            Q[I, c] = f(slot_xy(ps, I)...)
        end
    end
end

"Max |component c − f| on the fine patch, by slot class."
function fine_errors(solver, states, c, f)
    ps = patch_solver(solver, 2)
    Q = states[2]
    e = Dict(:ghost => 0.0, :plane => 0.0, :window => 0.0, :interior => 0.0)
    for I in CartesianIndices(size(Q)[1:3])
        cl = slot_class(ps, I)
        e[cl] = max(e[cl], abs(Q[I, c] - f(slot_xy(ps, I)...)))
    end
    return e
end

# A non-constant background for the other components keeps the state valid;
# only component 1 is measured, component 2 carries derivative data.
function base_state!(solver, states)
    set_all!(solver, states, 1, (x, y) -> 1.0)
    for c in 2:size(states[1], 4)
        set_all!(solver, states, c, (x, y) -> c == size(states[1], 4) ? 2.5 : 0.0)
    end
end

smooth(x, y) = 1 + 0.2 * sin(3x + 0.37) * cos(2y - 0.1)
smooth_x(x, y) = 0.6 * cos(3x + 0.37) * cos(2y - 0.1)
smooth_xx(x, y) = -1.8 * sin(3x + 0.37) * cos(2y - 0.1)

# --- values -----------------------------------------------------------------------

function values_study()
    println("\n=== values: the imposed shell and a fresh fill, 2-D, point samples ===")
    println("\nexactness on ((x − 3)(y − 2.5)/2)^d, N = 48, relative to its max on the box")
    println("  order   d = p−1: shell   fill        d = p: shell / fill")
    for p in ORDERS
        row = Float64[]
        for deg in (p - 1, p)
            solver, states = layout2d(48, p)
            base_state!(solver, states)
            f(x, y) = ((x - 3) * (y - 2.5) / 2)^deg
            set_all!(solver, states, 1, f)
            scale = maximum(abs(f(slot_xy(patch_solver(solver, 2), I)...))
                            for I in CartesianIndices(size(states[2])[1:3]))
            CL.prolong_level_ghosts!(solver, states)
            e = fine_errors(solver, states, 1, f)
            push!(row, max(e[:ghost], e[:plane]) / scale)
            CL._fill_fine_from_coarse!(solver, states,
                                       getfield(solver, :levels)[2].transfers[1])
            push!(row, maximum(values(fine_errors(solver, states, 1, f))) / scale)
        end
        @printf("  %4d    %.2e    %.2e    %.2e / %.2e\n", p, row[1], row[2], row[3], row[4])
    end
    println("\nsmooth field 1 + 0.2 sin(3x + 0.37) cos(2y − 0.1)")
    println("  order     N     ghosts      planes      fill")
    for p in ORDERS
        hs = Float64[]; eg = Float64[]; ep = Float64[]; ef = Float64[]
        for N in NS
            solver, states = layout2d(N, p)
            base_state!(solver, states)
            set_all!(solver, states, 1, smooth)
            CL.prolong_level_ghosts!(solver, states)
            e = fine_errors(solver, states, 1, smooth)
            CL._fill_fine_from_coarse!(solver, states,
                                       getfield(solver, :levels)[2].transfers[1])
            f = fine_errors(solver, states, 1, smooth)
            push!(hs, 2pi / N); push!(eg, e[:ghost]); push!(ep, e[:plane])
            push!(ef, maximum(values(f)))
            @printf("  %4d  %5d    %.3e   %.3e   %.3e\n", p, N, e[:ghost], e[:plane],
                    ef[end])
        end
        @printf("        orders   %s | %s | %s\n", fmt_orders(orders_of(hs, eg)),
                fmt_orders(orders_of(hs, ep)), fmt_orders(orders_of(hs, ef)))
    end
    flush(stdout)
end

# --- derivatives --------------------------------------------------------------------

function fine_derivative!(out, solver, states, c, dim; divergence=false)
    ps = patch_solver(solver, 2)
    a = CL.field(ps.decomp)
    a .= view(states[2], :, :, :, c)
    divergence ? CL.div_along!(out, a, ps, dim, 1) : CL.deriv_along!(out, a, ps, dim, 1)
    CL._scale_grad!(out, ps, dim)
    return out
end

# Max |a − ref| over the fine patch's own nodes, by class, the window taken
# along `dim`.
function field_errors(solver, a, ref; dim=1)
    ps = patch_solver(solver, 2)
    e = Dict(:plane => 0.0, :window => 0.0, :interior => 0.0)
    for I in CartesianIndices(a)
        cl = slot_class(ps, I; dim=dim)
        cl === :ghost && continue
        r = ref isa AbstractArray ? ref[I] : ref(slot_xy(ps, I)...)
        e[cl] = max(e[cl], abs(a[I] - r))
    end
    return e
end

# Put the fine field `vals` into component c on the fine patch's own nodes,
# boundary planes included, as the flux of a viscous term is formed there
# from the plane's own gradient. A divergence reads no ghosts at an
# interface end, so the ghosts are left as they are.
function carry!(solver, states, c, vals)
    ps = patch_solver(solver, 2)
    Q = states[2]
    for I in CartesianIndices(size(Q)[1:3])
        slot_class(ps, I) === :ghost || (Q[I, c] = vals[I])
    end
    return states
end

function derivatives_study()
    println("\n=== derivatives through the shell on the fine patch, 2-D ===")
    println("first:  d/dx through the gradient plans (the extended rows read the ghosts)")
    println("second: the divergence plans applied to the first, along x (xx) and along")
    println("        y (xy); the flux path a viscous term takes")
    println("interp: against the same operators on exact shell data (the transfer's part)")
    println("total:  against the exact derivative (includes the closure rows' own error)")
    println("window: within $W nodes of a fine end along the differentiated dimension,")
    println("        off the planes; plane: the imposed boundary planes")
    println("fill:   d/dx after a fresh fill, against the exact-data operator")
    println("\n  order     N   1st interp window/plane   1st total window  |" *
            " xx interp   xy interp   xx total   | fill 1st")
    for p in ORDERS
        hs = Float64[]
        cols = [Float64[] for _ in 1:7]
        for N in NS
            solver, states = layout2d(N, p)
            base_state!(solver, states)
            ps = patch_solver(solver, 2)
            D_imp = CL.field(ps.decomp)
            D_ex, S_imp, S_ex, X_imp, X_ex, D_fill = (similar(D_imp) for _ in 1:6)
            set_all!(solver, states, 1, smooth)
            fine_derivative!(D_ex, solver, states, 1, 1)
            CL.prolong_level_ghosts!(solver, states)
            fine_derivative!(D_imp, solver, states, 1, 1)
            e1i = field_errors(solver, D_imp, D_ex)
            e1t = field_errors(solver, D_imp, smooth_x)
            carry!(solver, states, 2, D_ex)
            fine_derivative!(S_ex, solver, states, 2, 1; divergence=true)
            fine_derivative!(X_ex, solver, states, 2, 2; divergence=true)
            carry!(solver, states, 2, D_imp)
            fine_derivative!(S_imp, solver, states, 2, 1; divergence=true)
            fine_derivative!(X_imp, solver, states, 2, 2; divergence=true)
            e2i = field_errors(solver, S_imp, S_ex)
            e2t = field_errors(solver, S_imp, smooth_xx)
            exi = field_errors(solver, X_imp, X_ex; dim=2)
            CL._fill_fine_from_coarse!(solver, states,
                                       getfield(solver, :levels)[2].transfers[1])
            fine_derivative!(D_fill, solver, states, 1, 1)
            ef = field_errors(solver, D_fill, D_ex)
            vals = (e1i[:window], e1i[:plane], e1t[:window],
                    e2i[:window], exi[:window], e2t[:window],
                    max(ef[:window], ef[:interior]))
            push!(hs, 2pi / N)
            for (col, v) in zip(cols, vals)
                push!(col, v)
            end
            @printf("  %4d  %5d   %.3e / %.3e     %.3e      |  %.3e   %.3e   %.3e  |  %.3e\n",
                    p, N, vals...)
        end
        @printf("        orders %s\n",
                join((fmt_orders(orders_of(hs, col)) for col in cols), " | "))
    end
    flush(stdout)
end

# --- restriction -----------------------------------------------------------------------

function restriction_study()
    println("\n=== restriction onto the covered root nodes, 2-D ===")
    println("inject: a fine-only perturbation 1e-3 sin(7x) is added; the written root")
    println("        nodes must equal the fine coincident values bitwise")
    println("filter: exact fine data, |coarse − f| over the written nodes")
    println("  N     inject: written  max |coarse − fine|    filter: written  error")
    hs = Float64[]; ef = Float64[]
    for N in NS
        out = Any[]
        for mode in (:inject, :filter)
            solver, states = layout2d(N, 6; level_restriction=mode)
            base_state!(solver, states)
            set_all!(solver, states, 1, smooth)
            pert(x) = mode === :inject ? 1e-3 * sin(7x) : 0.0
            Qf = states[2]
            psf = patch_solver(solver, 2)
            for I in CartesianIndices(size(Qf)[1:3])
                Qf[I, 1] += pert(slot_xy(psf, I)[1])
            end
            CL.restrict_level!(solver, states)
            ps = patch_solver(solver, 1)
            Q = states[1]
            written = 0; e = 0.0
            for I in CartesianIndices(size(Q)[1:3])
                ps.covered[I] == 0 && continue
                x, y = slot_xy(ps, I)
                Q[I, 1] == smooth(x, y) && continue     # held back by the margin
                written += 1
                e = max(e, abs(Q[I, 1] - (smooth(x, y) + pert(x))))
            end
            push!(out, written, e)
        end
        push!(hs, 2pi / N); push!(ef, out[4])
        @printf("  %4d   %6d   %.3e              %6d   %.3e\n", N, out...)
    end
    @printf("        filter orders %s\n", fmt_orders(orders_of(hs, ef)))
    flush(stdout)
end

# --- repeated regrids ------------------------------------------------------------------

wave1(x) = 1 + 0.2 * sin(3x + 0.37)

function regrid_study()
    println("\n=== repeated regrids, 1-D periodic, no evolution ===")
    println("the region is moved ±N/24 root nodes $(OPTS.regrids) times, " *
            "restricted each time")
    println("  order     N    fine error: first regrid   last regrid   ratio")
    for p in ORDERS
        hs = Float64[]; e1s = Float64[]; eKs = Float64[]
        for N in NS
            center = Ref(Float64(pi))
            halfwidth = 2pi / 12
            predicate = (patch, I) -> begin
                i, _, _ = interior_index(patch, I)
                abs(xcoord(patch, 1, i) - center[]) < halfwidth
            end
            r0 = BlockRegion((5N ÷ 12, 0, 0), (N ÷ 6 + 1, 1, 1))
            solver = Solver(n_global=(N, 1, 1), L_domain=(2pi, 1.0, 1.0), bcs=per3,
                            art=ArtificialProperties(enabled=false), filter_interval=0,
                            refine=r0,
                            regrid_interval=1, tag_threshold=1e6, tag_buffer=2,
                            tag_predicate=predicate, level_interpolation_order=p)
            states = allocate_state(solver)
            initialize!(solver, states, (x, y, z) -> Prim(rho=wave1(x), u=(0.5, 0, 0), p=1.0))
            workspace = Workspace(states)
            shift = 2pi / 24
            errs = Float64[]
            for k in 1:OPTS.regrids
                center[] = pi + (isodd(k) ? shift : -shift)
                getfield(solver, :regrid).checks += 1     # as run!'s cadence hook
                changed = CL.regrid!(solver, states, workspace, nothing)
                changed || error("regrid $k did not move the region")
                CL.restrict_level!(solver, states)
                ps = patch_solver(solver, 2)
                Q = states[2]
                n = ps.decomp.n_local[1]
                pad = ps.decomp.n_halo_d[1]
                push!(errs, maximum(abs(Q[i + pad, 1, 1, 1] - wave1(xcoord(ps, 1, i)))
                                    for i in 1:n))
            end
            push!(hs, 2pi / N); push!(e1s, errs[1]); push!(eKs, errs[end])
            @printf("  %4d  %5d    %.3e                  %.3e     %.2f\n", p, N, errs[1],
                    errs[end], errs[end] / errs[1])
        end
        @printf("        orders %s | %s\n", fmt_orders(orders_of(hs, e1s)),
                fmt_orders(orders_of(hs, eKs)))
    end
    flush(stdout)
end

# --- positivity ------------------------------------------------------------------------

function positivity_study()
    println("\n=== positivity: a fresh fill of a density step, 1-D ===")
    println("rho = 0.5625 + 0.4375 tanh((x0 − x)/w), 1 → 0.125, w in root cells")
    println("  order   w     undershoot/jump   overshoot/jump")
    N = 96
    for p in ORDERS, wcells in (0.5, 1.0, 2.0)
        r0 = BlockRegion((5N ÷ 12, 0, 0), (N ÷ 6 + 1, 1, 1))
        solver = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3,
                        art=ArtificialProperties(enabled=false), filter_interval=0, refine=r0,
                        level_interpolation_order=p)
        states = allocate_state(solver)
        h = 1 / N
        x0 = 0.5 + 0.37 * h
        rho(x) = 0.5625 + 0.4375 * tanh((x0 - x) / (wcells * h))
        initialize!(solver, states, (x, y, z) -> Prim(rho=rho(x), u=(0, 0, 0), p=1.0))
        CL._fill_fine_from_coarse!(solver, states, getfield(solver, :levels)[2].transfers[1])
        ps = patch_solver(solver, 2)
        pad = ps.decomp.n_halo_d[1]
        vals = [states[2][i + pad, 1, 1, 1] for i in 1:ps.decomp.n_local[1]]
        @printf("  %4d  %.1f   %.3e         %.3e\n", p, wcells,
                max(0.0, 0.125 - minimum(vals)) / 0.875,
                max(0.0, maximum(vals) - 1.0) / 0.875)
    end
    println("\nmoving-region Sod (test/level_tests.jl), N = 201, t = 0.15, subcycled,")
    println("regrid every 5 root steps; composite density error against the 601-node")
    println("uniform-fine run, and the composite minima of rho and p at the end")
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    N = 201; Nf = 3N - 2; tf = 0.15
    sf = Solver(n_global=(Nf, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=(wall2, per, per),
                cfl=0.2)
    Qf = allocate_state(sf)
    initialize!(sf, Qf, ic)
    run!(sf, Qf; tfinal=tf, nmax=40000)
    padF = sf.decomp.n_halo_d[1]
    rho_ref = [Qf[i + padF, 1, 1, 1] for i in 1:Nf]
    println("  order   error      min rho    min p      steps")
    for p in ORDERS
        sa = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=(wall2, per, per),
                    cfl=0.2, subcycle=true, regrid_interval=5,
                    refine=BlockRegion((85, 0, 0), (31, 1, 1)), level_interpolation_order=p)
        states = allocate_state(sa)
        initialize!(sa, states, ic)
        try
            run!(sa, states; tfinal=tf, nmax=40000)
        catch err
            err isa SolverFailure || rethrow()
            @printf("  %4d   FAILED: %s at step %d\n", p, err.reason, err.step)
            continue
        end
        region = CL.refined_region(sa)
        lo = region.offset[1] + 1
        hi = region.offset[1] + region.extent[1]
        padc = sa.patches[1].decomp.n_halo_d[1]
        padf = sa.patches[2].decomp.n_halo_d[1]
        e = 0.0; rmin = Inf; pmin = Inf
        for (k, Q) in enumerate(states)
            ps = patch_solver(sa, k)
            pad = ps.decomp.n_halo_d[1]
            for i in 1:ps.decomp.n_local[1]
                r = Q[i + pad, 1, 1, 1]; m = Q[i + pad, 1, 1, 2]
                E = Q[i + pad, 1, 1, size(Q, 4)]
                rmin = min(rmin, r)
                pmin = min(pmin, 0.4 * (E - 0.5 * m^2 / r))
            end
        end
        for i in 1:N
            v = lo <= i <= hi ? states[2][3 * (i - lo) + 1 + padf, 1, 1, 1] :
                                states[1][i + padc, 1, 1, 1]
            e = max(e, abs(v - rho_ref[3i - 2]))
        end
        @printf("  %4d   %.3e  %.4e  %.4e  %d\n", p, e, rmin, pmin, sa.step)
    end
    flush(stdout)
end

function main()
    t0 = time()
    selected("values") && values_study()
    selected("derivatives") && derivatives_study()
    selected("restriction") && restriction_study()
    selected("regrid") && regrid_study()
    selected("positivity") && positivity_study()
    @printf("\ndone in %.1f s\n", time() - t0)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
