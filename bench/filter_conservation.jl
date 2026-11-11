# Discrete conservation of the state filter on uniform and non-uniform cell
# volumes: the unweighted operator (`filter_weighting = :none`) against the
# volume-weighted form of the public Pyranda implementation (`:volume`), which
# filters J·q and divides by the volume passed through the same filter.
#
#   julia --project=. bench/filter_conservation.jl                  # everything
#   julia --project=. bench/filter_conservation.jl parts=operator   # the assembled lines only
#   julia --project=. bench/filter_conservation.jl parts=noh,sedov alphaf=0.49
#   julia --project=. bench/filter_conservation.jl parts=shared N=48
#
# Parts: operator (each 1-D line operator assembled column by column: constant
# preservation, the per-row conservation defect, the wall and fold profiles),
# noh (the three Noh geometries under both weightings, with the filter's own
# mass and energy tally over the run), sedov (the same through the spherical
# origin), interface (the shock/interface case on a clustered grid) and shared
# (a periodic line as two patches sharing two nodes, each patch's line closed
# by the filter's interface rows, the identity row and the conservative rows of
# `CONSERVATIVE_INTERFACE_FILTER`: the composite operator's defect about a
# shared node, the spread of the node's two copies, and its departure from the
# single line's pass on a smooth wave).
#
# What is measured. A pass is a linear operator M on each conserved component
# along a line, and the quadrature `volume_integral` applies weights each node
# by V_i = w_i J_i h. The pass conserves ∑ V_i q_i for every q exactly when
# Mᵀ V = V, so the row vector d = Mᵀ V − V is the defect of one pass: d·q is
# the mass the pass creates on q, and d_i / V_i is the fraction of node i's
# content it creates or destroys. Constant preservation is the other property,
# M 1 = 1, and the two coincide only for a symmetric M on a uniform V. The
# unweighted filter preserves constants on any volume by construction, since
# it never reads the volume; the weighted form does so through F(J)/F(J).
#
# Scratch tooling, like everything else in bench/: it prints tables and asserts
# nothing. The conclusions drawn from a run of it are written up in
# reference/CALIBRATION_APPENDIX.md (the filter on non-uniform volumes).

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: filter_state!, padded_index
using Printf

const CL = CompactLES
include(joinpath(@__DIR__, "..", "test", "references.jl"))
include(joinpath(@__DIR__, "..", "test", "cases.jl"))

const WEIGHTINGS = (:none, :volume)

# --- the assembled line operator --------------------------------------------

# The operator of one directional pass along `d` on component `c`, assembled
# from unit impulses at the first interior slot of the other dimensions, and
# the quadrature volumes along that line.
function line_operator(solver, d::Int, c::Int, N::Int)
    Q = allocate_state(solver)
    σ = CL.cons_parity(solver, d, c)
    idx(n) = d == 1 ? padded_index(solver, n, 1, 1) :
             d == 2 ? padded_index(solver, 6, n, 1) : padded_index(solver, 1, 1, n)
    M = zeros(N, N)
    weighted = CL._weighted_filter(solver)
    for n in 1:N
        fill!(parent(Q), 0.0)
        Q[idx(n), c] = 1.0
        q = view(Q, :, :, :, c)
        CL.exchange_dim_batch!([q], solver.decomp, d)
        if weighted
            CL._filter_volume!(solver, d)
            CL._filter_weighted!(q, solver, d, σ, 1.0)
        else
            CL.filt_along!(solver.tmp_a, q, solver, d, σ)
            CL.copy_interior!(q, solver.tmp_a, solver.decomp)
        end
        for k in 1:N
            M[k, n] = Q[idx(k), c]
        end
    end
    V = [CL.quad_weight(solver, d, n) / solver.inv_J[idx(n)] for n in 1:N]
    return M, V
end

fmt(v) = join((@sprintf("%+.1e", x) for x in v), " ")

# `@printf` takes a literal format string only; the long formats below are
# concatenated, so they go through the runtime form.
printf(fmt::String, args...) = print(Printf.format(Printf.Format(fmt), args...))

function operator_part(opt)
    N = opt.N
    filt(cl) = compact_filter(opt.alphaf; closures=cl)
    walls = ((SlipWallBC(), SlipWallBC()), per3[2], per3[3])
    line(wt; kw...) = Solver(; n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                             art=ArtificialProperties(enabled=false), filter_weighting=wt,
                             kw...)
    configs = [
        ("cartesian periodic", 1, wt -> line(wt; bcs=per3, filt=filt(:cascade))),
        ("cartesian walls, cascade", 1, wt -> line(wt; bcs=walls, filt=filt(:cascade))),
        ("cartesian walls, onesided", 1, wt -> line(wt; bcs=walls, filt=filt(:onesided))),
        ("stretched a=0.5, cascade", 1,
         wt -> line(wt; bcs=walls, filt=filt(:cascade),
                    stretch=(sine_cluster(0.0, 1.0, 0.5, 0.5), nothing, nothing))),
        ("cylindrical axis, cascade", 1,
         wt -> line(wt; bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                    metric=CylindricalMetric(), filt=filt(:cascade))),
        ("spherical origin, cascade", 1,
         wt -> line(wt; bcs=((OriginBC(), SlipWallBC()), per3[2], per3[3]),
                    metric=SphericalMetric(), origin=(0.0, π / 2, 0.0),
                    filt=filt(:cascade))),
        ("spherical poles (θ line)", 2,
         wt -> Solver(; n_global=(12, N, 1), L_domain=(1.0, π, 1.0),
                      metric=SphericalMetric(), origin=(0.2, 0.0, 0.0),
                      bcs=((SlipWallBC(), SlipWallBC()), (PoleBC(), PoleBC()),
                           per3[3]),
                      art=ArtificialProperties(enabled=false), filter_weighting=wt,
                      filt=filt(:cascade))),
    ]
    println("\n=== line operators, N = $N, alphaf = $(opt.alphaf) ===")
    println("d = Mᵀ V − V per source row, relative to V; interior is rows 9..N−8")
    for (name, d, mk) in configs
        println("\n--- $name ---")
        for wt in WEIGHTINGS
            solver = mk(wt)
            for (c, lab) in ((1, "rho"), (solver.equations.i_mom[d], "mom"))
                M, V = line_operator(solver, d, c, N)
                const_defect = maximum(abs, M * ones(N) .- 1)
                rel = (M' * V .- V) ./ V
                imax = argmax(abs.(rel))
                printf("  %-7s %-4s constant %.1e | max |d|/V %.2e at row %2d | " *
                        "interior %.1e | Σ|d|/ΣV %.2e\n", wt, lab, const_defect,
                        abs(rel[imax]), imax, maximum(abs, rel[9:N-8]),
                        sum(abs, M' * V .- V) / sum(V))
                lab == "rho" || continue
                println("          rows 1..8   ", fmt(rel[1:8]))
                println("          rows N-7..N ", fmt(rel[N-7:N]))
                if wt === :none
                    cs = vec(sum(M; dims=1)) .- 1
                    println("          column sums 1ᵀM − 1ᵀ, rows 1..8 ", fmt(cs[1:8]))
                end
            end
        end
    end
end

# --- the filter's tally over a run ------------------------------------------

# Total mass, dimension-1 momentum and energy of the state.
function totals(solver, Q)
    eq = solver.equations
    mass = sum(volume_integral(solver, view(Q, :, :, :, sp)) for sp in 1:eq.n_species)
    mom = volume_integral(solver, view(Q, :, :, :, eq.i_mom[1]))
    E = volume_integral(solver, view(Q, :, :, :, eq.i_energy))
    return [mass, mom, E]
end

# The run filters through this callback instead of inside `run!`
# (`filter_interval = 0` in the numerics), so the change of the totals across
# each pass is the filter's own and nothing else's. The interval is raised to
# one for the pass so that `filter_weight` reads the cadence the solver's own
# pass would; with no positivity floor configured the trajectory is the one
# `run!` produces filtering itself, since the pass then sits at the same
# point of the step.
function tallying_filter(acc::Vector{Float64}, scale::Vector{Float64})
    return (solver, Q) -> begin
        before = totals(solver, Q)
        solver.filter_interval = 1
        filter_state!(solver, Q)
        solver.filter_interval = 0
        after = totals(solver, Q)
        acc .+= after .- before
        scale .= max.(scale, abs.(after))
        nothing
    end
end

function run_tallied(prob, numerics, tfinal)
    solver, Q = setup(prob, numerics)
    acc = zeros(3)
    scale = zeros(3)
    run!(solver, Q; tfinal=tfinal, nmax=NMAX, callback=tallying_filter(acc, scale))
    return solver, Q, acc ./ max.(scale, 1e-300)
end

function noh_part(opt)
    println("\n=== Noh, both weightings: the filter's running tally ===")
    println("mass/energy: the filter's accumulated change over the run, relative " *
            "to the largest total seen")
    for ν in 1:3
        N = Dict(NOH_N)[ν]
        t0 = Dict(NOH_T0)[ν]
        for wt in WEIGHTINGS
            num = Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=true),
                           cfl=NOH_CFL, deriv=lele_d1_6(),
                           filter=StateFilter(compact_filter(opt.alphaf); interval=0,
                                              weighting=wt),
                           control=StepControl(validity=:permissive))
            solver, Q, defect = run_tallied(noh_problem(ν; N, t0), num, NOH_T - t0)
            xs, ρ, u, p = case_line_profile(solver, Q)
            plat, deficit, Rnum, epre = noh_metrics(xs, ρ, ν)
            printf("  nu=%d %-7s plateau %.4f (exact %.0f)  wall deficit %+.1f%%  " *
                    "shock %.4f  L1 pre %.2e | filter mass %+.2e mom %+.2e " *
                    "energy %+.2e | steps %d\n", ν, wt, plat, 4.0^ν, 100deficit,
                    Rnum, epre, defect[1], defect[2], defect[3], solver.step)
        end
    end
end

function sedov_part(opt)
    println("\n=== Sedov through the spherical origin, both weightings ===")
    for wt in WEIGHTINGS
        num = Numerics(n_global=(SEDOV_N, 1, 1), art=ArtificialProperties(enabled=true),
                       cfl=0.3,
                       filter=StateFilter(compact_filter(opt.alphaf); interval=0,
                                          weighting=wt),
                       control=StepControl(validity=:permissive))
        solver, Q, defect = run_tallied(sedov_problem(), num, SEDOV_T)
        rs, ρ, u, p = case_line_profile(solver, Q)
        Rex = sedov_shock_radius(SEDOV_E, SEDOV_T, 3, 1.4)
        Rnum = front_position(rs, ρ, 2.0)
        printf("  %-7s R_s %.4f vs %.4f (%+.2f%%)  peak rho %.3f | filter mass " *
                "%+.2e mom %+.2e energy %+.2e | steps %d\n", wt, Rnum, Rex,
                100 * (Rnum / Rex - 1), maximum(ρ), defect[1], defect[2],
                defect[3], solver.step)
    end
end

function interface_part(opt)
    println("\n=== shock/interface on a clustered grid, both weightings ===")
    st = sine_cluster(0.0, 1.0, SI_X_IFACE, 0.5)
    for (label, stretch1) in (("uniform", nothing), ("clustered a=0.5", st))
        for wt in WEIGHTINGS
            r = shock_interface(N=121, tfin=0.15, delta=2.0, nmax=4000,
                                stretch1=stretch1, filt=compact_filter(opt.alphaf),
                                filter_weighting=wt)
            printf("  %-16s %-7s worst Y %+.4f / %.4f  width %d cells  steps %d  " *
                    "completed %s\n", label, wt, r.worst_min_Y, r.worst_max_Y,
                    r.width_cells, r.steps, r.completed)
        end
    end
end

# --- the composite line of two patches ------------------------------------------

# One pass of `scheme` over the periodic composite values `v`, the line held as
# two patches, nodes 1..nA and nA..N then 1, sharing nodes nA and 1. Each
# patch's line closes at both ends with `rows`, its ghost layers holding the
# other patch's nodes as the exchange fills them, and each node takes its
# patch's value, the mean of the two copies at a shared node as the averaging
# leaves it. Returns the composite values and the largest difference between
# the two copies of a shared node.
function composite_pass(scheme, rows, v)
    N = length(v)
    nA = N ÷ 2 + 1
    out = zeros(N)
    copies = zeros(N)
    first_copy = fill(NaN, N)
    spread = 0.0
    for (c0, n) in ((0, nA), (nA - 1, N - nA + 2))
        d = CL.Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1))
        plan = CL.plan_direction(d, scheme, 1, 1.0; lo_closures=rows, hi_closures=rows)
        f = CL.field(d)
        g = CL.field(d)
        pad = d.n_halo_d
        for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
            f[i, j, k] = v[mod(c0 + i - pad[1] - 1, N) + 1]
        end
        CL.apply_along!(g, plan, f, d)
        for i in 1:n
            c = mod(c0 + i - 1, N) + 1
            x = g[i+pad[1], pad[2]+1, pad[3]+1]
            if isnan(first_copy[c])
                first_copy[c] = x
            else
                spread = max(spread, abs(x - first_copy[c]))
            end
            out[c] += x
            copies[c] += 1
        end
    end
    return out ./ copies, spread
end

# The single periodic line's pass of `scheme` on `v`.
function single_pass(scheme, v)
    N = length(v)
    d = CL.Decomp((N, 4, 4), (true, true, true); dims=(1, 1, 1))
    plan = CL.plan_direction(d, scheme, 1, 1.0)
    f = CL.field(d)
    g = CL.field(d)
    pad = d.n_halo_d
    for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
        f[i, j, k] = v[mod(i - pad[1] - 1, N) + 1]
    end
    CL.apply_along!(g, plan, f, d)
    return [g[i+pad[1], pad[2]+1, pad[3]+1] for i in 1:N]
end

function shared_part(opt)
    N = opt.N
    nA = N ÷ 2 + 1
    schemes = (("compact_filter()", compact_filter()),
               ("compact_filter($(opt.alphaf))", compact_filter(opt.alphaf)),
               ("pyranda_filter()", pyranda_filter()),
               ("gaussian_filter()", gaussian_filter()))
    println("\n=== a periodic line of $N nodes as two patches sharing nodes 1 and $nA ===")
    println("d = Mᵀ V − V relative to V (V = h), at the shared node nA and beside it; " *
            "spread: the two copies of a shared node before the mean")
    for (label, scheme) in schemes
        println("\n--- $label ---")
        for (rlabel, rows) in (("identity row", CL.interface_closures(scheme)),
                               ("conservative rows", CL._conservative_interface_rows(scheme)))
            M = zeros(N, N)
            spread = 0.0
            for j in 1:N
                e = zeros(N)
                e[j] = 1
                col, s = composite_pass(scheme, rows, e)
                M[:, j] = col
                spread = max(spread, s)
            end
            rel = vec(sum(M; dims=1)) .- 1
            printf("  %-17s %d rows  constant %.1e  max |d|/V %.2e  Σ|d|/ΣV %.2e  " *
                   "spread %.1e\n", rlabel, length(rows), maximum(abs, M * ones(N) .- 1),
                   maximum(abs, rel), sum(abs, rel) / N, spread)
            println("                    nA-3..nA+3 ", fmt(rel[nA-3:nA+3]))
        end
        # A smooth wave through both shared nodes, off its zeros there: the
        # composite pass against the single line's, and the single line's own
        # correction.
        println("  wave sin(2π k i / n + 0.37), k = 3: max |composite − single| and the " *
                "correction max |single − v|")
        for (rlabel, rowsof) in (("identity row", CL.interface_closures),
                                 ("conservative rows", CL._conservative_interface_rows))
            errs = Float64[]
            ns = (N, 2N, 4N)
            for n in ns
                v = [sin(2π * 3 * (i - 1) / n + 0.37) for i in 1:n]
                single = single_pass(scheme, v)
                comp, _ = composite_pass(scheme, rowsof(scheme), v)
                push!(errs, maximum(abs, comp .- single))
                rlabel == "identity row" && n == N &&
                    printf("    correction at n = %d: %.2e\n", n, maximum(abs, single .- v))
            end
            p = log(errs[1] / errs[3]) / log(4)
            printf("    %-17s n = %s: %s  order %.2f\n", rlabel, join(ns, ", "), fmt(errs), p)
        end
    end
end

function main(args)
    opt = CL.script_args(args, (parts="all", N=64, alphaf=0.45))
    parts = opt.parts == "all" ? ["operator", "noh", "sedov", "interface", "shared"] :
            split(opt.parts, ',')
    for part in parts
        part == "operator" ? operator_part(opt) :
        part == "noh" ? noh_part(opt) :
        part == "sedov" ? sedov_part(opt) :
        part == "interface" ? interface_part(opt) :
        part == "shared" ? shared_part(opt) :
        error("unknown part '$part'; want operator, noh, sedov, interface or shared")
    end
end

main(ARGS)
