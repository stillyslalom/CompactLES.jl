# The artificial-property sensors and the state filter at an interface: the
# imposed shell of a refined patch (its ghost ring and boundary-plane nodes,
# overwritten from the coarse solution after every stage) and the shared plane
# of a same-level patch pair.
#
#   julia --project=. -t 16 bench/interfacesensor.jl smoke=true
#   julia --project=. -t 16 bench/interfacesensor.jl probe
#   mpiexec -n 4 julia --project=. -t 1 bench/interfacesensor.jl crossing
#
# `parts` is the first positional argument and takes `all` or a
# comma-separated subset of `probe`, `crossing`, `filter`, `undershoot`.
# `smoke=true` shrinks every grid, end time and step cap so the whole script
# runs in about a minute. Scratch tooling like the rest of bench/: it prints
# tables and asserts nothing, and the conclusions are written up in
# reference/CALIBRATION_APPENDIX.md.
#
# Parts:
#
#   probe       one application of the detector, the smoother and the state
#               filter on a fixed smooth field, at an imposed fine shell and
#               at a same-level interface, against the uniform periodic grid
#               of the same spacing. Prints |interface − uniform| over the
#               input amplitude at the first `nodes` nodes of each interface
#               face and the log2 orders between successive resolutions, the
#               table bench/sensorwall.jl prints at a wall. Seconds, one rank.
#   crossing    a Sod shock through a nest with the sensors and the filter
#               live, one configuration axis varied at a time from a base
#               row. Per row: the momentum ahead of the shock, the minimum
#               density and pressure seen over the composite on any step, the
#               steps whose `state_report` counted an inadmissible point, the
#               composite density error against the uniform fine run split
#               into shell window / fine interior / covered root / uncovered
#               root, and the artificial diffusivity number over each of
#               those regions before, during and after the crossing. Closes
#               with a two-species Sod whose contact carries the composition.
#   filter      the filter's own change at the shell: max |Δρ| per distance
#               from a coarse-fine face over the run's passes, under the
#               three interface filter row sets. A smooth entropy wave and
#               the Sod crossing, plus one two-dimensional row for the
#               transverse pass at the plane nodes.
#   undershoot  the root-edge mass-fraction undershoot of the two-species
#               layer: the minimum Y by distance from the level-1 box edge,
#               root nodes and each fine level's nodes separately, over
#               stepping mode, filter cadence and relaxation, nesting depth
#               and the sensor toggle. `undershoot_depths=2,3` and
#               `undershoot_variants=all` (or a comma-separated list of
#               label substrings) select the rows. Minutes per row.
#
# What the mechanism is, as found in the tree:
#
# * The shell is imposed without stepping by `CL._presync!(solver, states)`,
#   which is `sync_patches!` followed by `sync_levels!` (restriction then
#   `prolong_level_ghosts!`). A single-patch uniform run needs
#   `CL.exchange_state!` instead, since `_presync!` returns a lone state
#   untouched. The refined patch's boundary plane coincides with a coarse
#   node, so interpolation reproduces it exactly and only the ghost ring
#   carries the order-6 interpolation error.
# * One operator runs on one patch through `CL.PatchSolver(solver, patch)`,
#   whose property forwarding gives the patch's own plans, faces and spacing.
#   `probe` therefore needs one rank; `crossing`, `filter` and `undershoot`
#   run under any rank count.
# * The state filter at an interface end takes one identity closure row at
#   the edge node plus the interior C8 rows, which read one to three layers
#   of exchanged or imposed ghosts. Under `interface_rhs = :onesided` it
#   falls back to the closed-edge rows of `compact_filter`. In one dimension
#   the identity row leaves the plane node untouched; in two a transverse
#   pass filters it with the full stencil, and the shell is re-imposed
#   afterwards.
# * The flux divergence keeps one-sided rows at an interface whatever
#   `interface_rhs` says, and `:neutral3` maps to `:cascade3` there, so the
#   default derivative row at an interface is the cascade3 row.
# * The δ⁴ detector clamps its taps at every interface face (a zeroth-order
#   extension) and the smoother takes closed-edge rows there. A build
#   carrying `CompactLES.SENSOR_INTERFACE_GHOSTS` lets the detector read the
#   exchanged or imposed ghosts instead; this script compares the two through
#   that toggle and, on a build without it, prints the clamp rows alone and
#   says so in the row label.
# * The artificial coefficients are computed on a fine patch's imposed
#   boundary-plane nodes, and the coarse patch's covered region is never
#   masked, so both appear in the tables as ordinary nodes.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES

# `@printf` takes a literal format only; this wraps a format assembled from
# several literals so a long line can be split.
printfmt(fmt::String, args...) = Printf.format(stdout, Printf.Format(fmt), args...)

const args = CL.script_args(ARGS,
    (parts="all", smoke=false, ns="48,96,192", nodes=6,
     N=201, nmax=40000, waveN=96, wave_tfinal=0.5,
     layerN=96, layerny=24, layer_tfinal=50.26548245743669, samples=8,
     undershoot_depths="2,3", undershoot_variants="all");
    positional=(:parts,))

const RANK = MPI.Comm_rank(MPI.COMM_WORLD)
const NP = MPI.Comm_size(MPI.COMM_WORLD)

# The detector's interface behavior is a toggle in the parallel change that
# adds ghost-reading taps at interface faces; without it every face clamps.
const HAS_GHOST_TOGGLE = isdefined(CL, :SENSOR_INTERFACE_GHOSTS)
const GHOST_MODES = HAS_GHOST_TOGGLE ? (true, false) : (false,)
ghost_label(on) = HAS_GHOST_TOGGLE ? (on ? "ghost taps" : "clamped taps") :
                  "clamped taps (no toggle in this build)"

function with_ghosts(f, on::Bool)
    HAS_GHOST_TOGGLE || return f()
    old = CL.SENSOR_INTERFACE_GHOSTS[]
    CL.SENSOR_INTERFACE_GHOSTS[] = on
    try
        return f()
    finally
        CL.SENSOR_INTERFACE_GHOSTS[] = old
    end
end

const PER3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
const PER = (PeriodicBC(), PeriodicBC())
const WALL2 = (SlipWallBC(), SlipWallBC())

# The three interface filter row sets compared throughout: the extended-data
# identity row, and the two closed-edge sets `compact_filter` offers.
const FILTER_ROWS = (("extended", (interface_rhs=:extended,
                                   filt=compact_filter(0.45))),
                     ("onesided", (interface_rhs=:onesided,
                                   filt=compact_filter(0.45; closures=:onesided))),
                     ("cascade", (interface_rhs=:onesided,
                                  filt=compact_filter(0.45; closures=:cascade))))

# Rank-0 printing. The parameter is not named `args`, which is the parsed
# option table above.
say(xs...) = RANK == 0 && (println(xs...); flush(stdout))

# A configuration that loses positivity raises `SolverFailure` from `run!`;
# the row then reads the failure and the study goes on. No other exception is
# caught (bench/wallfilter.jl).
function attempt(f)
    try
        return f()
    catch err
        err isa SolverFailure || rethrow()
        return @sprintf("FAILED %s at step %d, t = %.4f", err.reason, err.step, err.t)
    end
end

failed(r) = r isa String

line(ps, a) = (pad = ps.decomp.n_halo_d[1];
               [a[i + pad, 1, 1] for i in 1:ps.decomp.n_local[1]])

order(a, b) = (a > 0 && b > 0) ? log2(a / b) : NaN

# Mixture density at one interior node, summed from the partial densities so
# the reading does not depend on current primitives.
function node_density(ps, Q, i, j=1, k=1)
    I = gidx(ps, i, j, k)
    ρ = zero(eltype(Q))
    for sp in 1:ps.equations.n_species
        ρ += Q[I, sp]
    end
    return ρ
end

# Distance of a node from the nearest interface face of its own patch, in that
# patch's own nodes; 1 is the boundary plane and 0 means the patch has no
# interface face. Read through the patch-global index, so a decomposed patch
# gives the same answer on every rank holding a piece of it.
function face_distance(ps, idx)
    p, dcp = ps.patch, ps.decomp
    best = 0
    for d in 1:3
        dcp.active[d] || continue
        g = dcp.offset[d] + idx[d]
        ng = dcp.n_global[d]
        if p.bcs[d][1] isa CL.InterfaceBC
            best = best == 0 ? g : min(best, g)
        end
        if p.bcs[d][2] isa CL.InterfaceBC
            best = best == 0 ? ng - g + 1 : min(best, ng - g + 1)
        end
    end
    return best
end

# --- part: probe --------------------------------------------------------------

# Two modes, the second near the grid scale, both periodic on [0, 1) and both
# phase-shifted off the symmetry points of the grid. Without the shift the
# same-level interface at x = 1/2 falls on a point about which both modes are
# odd, the filter's own change there vanishes, and every row of that table
# reads round-off while measuring nothing.
const PROBE_PHASE = 1 / 7
rho_probe(x) = 1 + 0.3 * sinpi(2(x + PROBE_PHASE)) +
               0.05 * sinpi(10(x + PROBE_PHASE))
probe_ic(x, y, z) = Prim(rho=rho_probe(x), p=1.0, u=(0.0, 0.0, 0.0))

# The middle third of the root, four or more nodes clear of the periodic edge
# at every resolution this part runs.
probe_region(N) = BlockRegion((N ÷ 3, 0, 0), (N ÷ 3 + 1, 1, 1))

function probe_solver(mode, N; kw...)
    extra = mode === :levels ? (refine=probe_region(N),) :
            mode === :patches ? (patch_grid=(2, 1, 1),) : (;)
    s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER3,
               filter_cfl=0.0, filter_interval=1; kw..., extra...)
    Q = allocate_state(s)
    initialize!(s, Q, probe_ic)
    # The shell (and the same-level ghost refill) without taking a step.
    Q isa Vector ? CL._presync!(s, Q) : CL.exchange_state!(Q, s.decomp)
    return s, Q
end

# Specific internal energy over the whole padded extent, so the operator reads
# the imposed or exchanged ghosts as the solver's own sensor does.
function energy_field(ps, Q)
    e = CL.field(ps.decomp)
    eq = ps.equations
    m1, m2, m3 = eq.i_mom
    for k in axes(e, 3), j in axes(e, 2), i in axes(e, 1)
        ρ = zero(eltype(e))
        for sp in 1:eq.n_species
            ρ += Q[i, j, k, sp]
        end
        if ρ > 0
            ke = (Q[i, j, k, m1]^2 + Q[i, j, k, m2]^2 + Q[i, j, k, m3]^2) / (2ρ)
            e[i, j, k] = (Q[i, j, k, eq.i_energy] - ke) / ρ
        else
            e[i, j, k] = zero(eltype(e))   # physical-edge halo of a closed face
        end
    end
    return e
end

function op_detector(ps, Q)
    e = energy_field(ps, Q)
    out = CL.field(ps.decomp)
    # The internal energy is recovered on the padded extent, so the solver's
    # own call declares its interface ghosts valid; the toggle then decides.
    if HAS_GHOST_TOGGLE
        CL.detect_sum!(out, e, ps, 0; ghosts=true)
    else
        CL.detect_sum!(out, e, ps, 0)
    end
    return line(ps, out), maximum(abs, line(ps, e))
end

function op_smoother(ps, Q)
    e = energy_field(ps, Q)
    scale = maximum(abs, line(ps, e))
    CL.smooth!(e, ps)
    return line(ps, e), scale
end

function op_filter(ps, Q)
    before = [node_density(ps, Q, i) for i in 1:ps.decomp.n_local[1]]
    filter_state!(ps, Q)
    return [node_density(ps, Q, i) for i in 1:ps.decomp.n_local[1]],
           maximum(abs, before)
end

# Node `i` of a patch sits at reference node `region.offset + i` whenever the
# reference carries this patch's own spacing: a patch of a two-patch grid
# against the uniform grid of `N` nodes, the level-1 patch against the uniform
# grid of `3N`. Both wrap, the reference being periodic.
ref_node(ps, i, nref) =
    mod(ps.patch.region.offset[1] + ps.decomp.offset[1] + i - 1, nref) + 1

"""One operator at one resolution: the two windows of scaled departures."""
function probe_case(N, mode, op, kw, nodes, ghosts)
    nref = mode === :levels ? 3N : N
    s, Q = probe_solver(mode, N; kw...)
    su, Qu = probe_solver(:uniform, nref; kw...)
    states = Q isa Vector ? Q : [Q]
    uni, scale = with_ghosts(() -> op(su, Qu), ghosts)
    # `:levels` reads both faces of the level-1 patch; `:patches` reads the one
    # interior interface from each side, patch 2's low face and patch 1's high.
    lo_patch, hi_patch = mode === :levels ? (2, 2) : (2, 1)
    vals = Dict{Int,Vector{Float64}}()
    solvers = Dict{Int,Any}()
    for pi in unique((lo_patch, hi_patch))
        ps = CL.PatchSolver(s, s.patches[pi])
        solvers[pi] = ps
        vals[pi] = first(with_ghosts(() -> op(ps, states[pi]), ghosts))
    end
    dep(pi, i) = abs(vals[pi][i] - uni[ref_node(solvers[pi], i, nref)]) / scale
    nhi = length(vals[hi_patch])
    return [dep(lo_patch, i) for i in 1:nodes],
           [dep(hi_patch, nhi + 1 - m) for m in 1:nodes]
end

function probe_report(name, ns, windows, nodes)
    say(name)
    RANK == 0 || return
    print("  node")
    for N in ns
        @printf("%13s", "N=$N")
    end
    println("      orders")
    for i in 1:nodes
        @printf("  %4d", i)
        vals = [w[i] for w in windows]
        for v in vals
            @printf("%13.3e", v)
        end
        print("   ")
        for k in 1:(length(vals)-1)
            @printf(" %6.2f", order(vals[k], vals[k+1]))
        end
        println()
    end
    flush(stdout)
end

function probe_part(ns, nodes)
    NP == 1 || error("parts=probe runs one operator on one patch; use one rank")
    say("\n=== part probe: one operator application at an interface face ===")
    say("field rho = 1 + 0.3 sin(2 pi (x + 1/7)) + 0.05 sin(10 pi (x + 1/7)), ",
        "p = 1, u = 0, periodic [0, 1)")
    say("printed: |interface - uniform| / input amplitude, then the log2 ",
        "orders; round-off is near 1e-16")
    cases = Any[]
    for g in GHOST_MODES
        push!(cases, ("detector :delta4, " * ghost_label(g), op_detector, (;), g))
    end
    for smoo in (:gaussian, :compact)
        push!(cases, ("smoother :$smoo", op_smoother,
                      (art=ArtParams(smoother=smoo),), first(GHOST_MODES)))
    end
    for (flab, fkw) in FILTER_ROWS
        push!(cases, ("state filter, $flab rows", op_filter, fkw,
                      first(GHOST_MODES)))
    end
    modes = ((:levels, "two-level nest, level-1 patch at h/3, " *
                       "reference = uniform 3N",
              "shell, low face", "shell, high face"),
             (:patches, "same-level pair, patch_grid = (2, 1, 1), " *
                        "reference = uniform N",
              "interface, patch 2 side", "interface, patch 1 side"))
    for (mode, title, lolab, hilab) in modes
        say("\n--- $title ---")
        for (name, op, kw, g) in cases
            results = [probe_case(N, mode, op, kw, nodes, g) for N in ns]
            say("")
            probe_report("$name, $lolab", ns, [r[1] for r in results], nodes)
            probe_report("$name, $hilab", ns, [r[2] for r in results], nodes)
        end
    end
end

# --- part: crossing -----------------------------------------------------------

sod_ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)

# The composition sheet is blended over a fixed physical width, two coarse
# cells at the default N, and not over a grid-dependent one: the same initial
# data must reach the root run and the uniform fine reference. A step in Y at
# the diaphragm is not a resolvable start at all, uniform grid included, since
# the species diffusivity it raises collapses the step within three of them.
const SPECIES_BLEND = 0.01
function sod2_ic(x, y, z)
    θ = 0.5 * (1 + tanh((x - 0.5) / SPECIES_BLEND))
    return Prim(Y=(1 - θ, θ), u=(0, 0, 0), p=x < 0.5 ? 1.0 : 0.1,
                rho=x < 0.5 ? 1.0 : 0.125)
end

two_gases() = IdealMixture([IdealSpecies{Float64}("species-a", 1.0, 1.4),
                            IdealSpecies{Float64}("species-b", 1.0, 1.4)])

deriv_of(row) = row.scheme === :C10 ? lele_d1_10(closures=row.closures) :
                lele_d1_6(closures=row.closures)

const CROSSING_CFL = 0.4
const SHELL_W = 4

# The level-1 box of the level-test gate scaled with N: x in [0.6, 0.8] at
# N = 201, and a level-2 box over the middle half of it.
function crossing_regions(N, depth)
    off1 = round(Int, 0.6 * (N - 1))
    ext1 = round(Int, 0.2 * (N - 1)) + 1
    r1 = BlockRegion((off1, 0, 0), (ext1, 1, 1))
    depth == 2 && return r1
    foff, fext = 3off1, 3ext1 - 2
    ext2 = fext ÷ 2
    ext2 >= 21 || error("N = $N leaves no room for a level-2 box")
    return [r1, BlockRegion((foff + (fext - ext2) ÷ 2, 0, 0), (ext2, 1, 1))]
end

function crossing_solver(row, N)
    extra = row.mode === :patches ? (patch_grid=row.patch_grid,) :
            (refine=crossing_regions(N, row.depth), subcycle=row.subcycle,
             tile=row.tile)
    eos = row.species == 2 ? two_gases() : IdealSpecies("gas"; gamma=1.4, R=1.0)
    s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=(WALL2, PER, PER),
               cfl=CROSSING_CFL, eos=eos, deriv=deriv_of(row),
               art=ArtParams(enabled=row.art), filter_cfl=row.filter_cfl,
               control=StepControl(validity=:permissive); extra...)
    Q = allocate_state(s)
    initialize!(s, Q, row.species == 2 ? sod2_ic : sod_ic)
    return s, Q
end

# Reference node of patch node `i` on the uniform grid of spacing h/3. Level 0
# takes every third node; level 2 coincides with one only where its own index
# is divisible by three, and 0 means "no coincident reference node".
function crossing_ref_node(ps, i)
    p = ps.patch
    g = p.region.offset[1] + ps.decomp.offset[1] + i - 1
    p.level == 0 && return 3g + 1
    p.level == 1 && return g + 1
    return rem(g, 3) == 0 ? g ÷ 3 + 1 : 0
end

# Shell window, fine interior, covered root, uncovered root. A same-level
# patch has no covered nodes, and its interface window is reported as `shell`.
function node_region(ps, i)
    ps.covered[gidx(ps, i, 1, 1)] != 0 && return 3
    d = face_distance(ps, (i, 1, 1))
    (d != 0 && d <= SHELL_W) && return 1
    return ps.patch.level == 0 ? 4 : 2
end

const REGION_NAMES = ("shell", "fine", "covered", "root")

function region_max(solver, states, f)
    acc = zeros(4)
    for (ps, Q) in CL.eachpatch(solver, states)
        for i in 1:ps.decomp.n_local[1]
            v = f(ps, Q, i)
            v === nothing && continue
            r = node_region(ps, i)
            acc[r] = max(acc[r], v)
        end
    end
    return [MPI.Allreduce(a, max, solver.comm) for a in acc]
end

density_error(ref) = (ps, Q, i) -> begin
    j = crossing_ref_node(ps, i)
    j == 0 ? nothing : abs(node_density(ps, Q, i) - ref[j])
end

# The artificial diffusivity number ((mu*+beta*)/rho + kappa*/(rho cp) +
# max_k D*_k) / (c h), read from the persistent coefficient arrays.
function diffusivity_number(ps, Q, i)
    I = gidx(ps, i, 1, 1)
    ν = (ps.mu_art[I] + ps.beta_art[I]) / ps.rho[I] +
        ps.kappa_art[I] / (ps.rho[I] * ps.cp_mix[I])
    isempty(ps.D_art) || (ν += maximum(D[I] for D in ps.D_art))
    return ν / (ps.c[I] * ps.h[1])
end

# The per-step sweep of the composite: the smallest density and pressure any
# step passed through, and how many steps the EOS called inadmissible.
mutable struct Monitor
    rho_min::Float64
    p_min::Float64
    inadmissible::Int
end
Monitor() = Monitor(Inf, Inf, 0)

function monitor!(mon, solver, states)
    rep = state_report(solver, states)
    rep.inadmissible > 0 && (mon.inadmissible += 1)
    mon.rho_min = min(mon.rho_min, rep.rho_min)
    pmin = Inf
    refresh_primitives!(solver, states)
    for (ps, Q) in CL.eachpatch(solver, states)
        for i in 1:ps.decomp.n_local[1]
            pmin = min(pmin, ps.p[gidx(ps, i, 1, 1)])
        end
    end
    mon.p_min = min(mon.p_min, MPI.Allreduce(pmin, min, solver.comm))
    return nothing
end

# Momentum ahead of the shock on the root, the quantity the level-test gate
# reads: the exact solution is still quiescent above x = 0.85 at t = 0.1.
function ahead_noise(solver, states)
    m1 = solver.equations.i_mom[1]
    worst = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        ps.patch.level == 0 || continue
        for i in 1:ps.decomp.n_local[1]
            xcoord(ps, 1, i) > 0.85 || continue
            worst = max(worst, abs(Q[gidx(ps, i, 1, 1), m1]))
        end
    end
    return MPI.Allreduce(worst, max, solver.comm)
end

# The largest mass-fraction excursion outside [0, 1], split over the shell
# window and everything else (the `species_excursion` pattern of
# bench/interfaceconservation.jl).
function shell_excursion(solver, states)
    shell, rest = 0.0, 0.0
    refresh_primitives!(solver, states)
    for (ps, Q) in CL.eachpatch(solver, states)
        for Y in ps.Y, i in 1:ps.decomp.n_local[1]
            y = Y[gidx(ps, i, 1, 1)]
            e = max(-y, y - 1)
            node_region(ps, i) == 1 ? (shell = max(shell, e)) :
                                      (rest = max(rest, e))
        end
    end
    return MPI.Allreduce(shell, max, solver.comm),
           MPI.Allreduce(rest, max, solver.comm)
end

function reference_lines(row, N, ts, nmax)
    eos = row.species == 2 ? two_gases() : IdealSpecies("gas"; gamma=1.4, R=1.0)
    s = Solver(n_global=(3N - 2, 1, 1), L_domain=(1.0, 1.0, 1.0),
               bcs=(WALL2, PER, PER), cfl=CROSSING_CFL, eos=eos,
               deriv=deriv_of(row), art=ArtParams(enabled=row.art),
               filter_cfl=row.filter_cfl,
               control=StepControl(validity=:permissive))
    Q = allocate_state(s)
    initialize!(s, Q, row.species == 2 ? sod2_ic : sod_ic)
    ws = Workspace(Q)
    out = Vector{Vector{Float64}}()
    for t in ts
        run!(s, Q, ws; tfinal=t, nmax=nmax)
        refresh_primitives!(s, Q)
        push!(out, last(line_sample(s, Q, :rho)))
    end
    return out
end

refkey(row) = (row.scheme, row.closures, row.filter_cfl, row.art, row.species)

function crossing_row(row, N, ts, nmax, refs)
    have_ref = !failed(refs)
    attempt() do
        s, Q = crossing_solver(row, N)
        states = Q isa Vector ? Q : [Q]
        ws = Workspace(Q)
        mon = Monitor()
        cb = Callback(EveryStep(1), (sv, st) -> (monitor!(mon, sv, st); nothing))
        errs = Vector{Vector{Float64}}()
        qs = Vector{Vector{Float64}}()
        noise = NaN
        for (k, t) in enumerate(ts)
            run!(s, Q, ws; tfinal=t, nmax=nmax, callback=cb)
            refresh_primitives!(s, states)
            push!(qs, region_max(s, states, diffusivity_number))
            k == 1 && continue
            have_ref && push!(errs, region_max(s, states, density_error(refs[k])))
            k == 2 && (noise = ahead_noise(s, states))
        end
        exc = row.species == 2 ? shell_excursion(s, states) : (0.0, 0.0)
        return (; steps=s.step, noise, mon, errs, qs, exc, t=s.t,
                npatch=npatches(s), have_ref)
    end
end

function print_crossing(row, r, ts)
    RANK == 0 || return
    if failed(r)
        printfmt("  %-22s %s\n", row.label, r)
        return
    end
    printfmt("  %-22s patches %d  steps %5d  noise(x>0.85, t=%.3g) %.3e  " *
             "rho_min %.5f  p_min %.5f  inadmissible steps %d\n",
             row.label, r.npatch, r.steps, ts[2], r.noise, r.mon.rho_min,
             r.mon.p_min, r.mon.inadmissible)
    r.have_ref ||
        println("      rho error: the uniform reference at this row's ",
                "numerics did not complete")
    for (k, e) in enumerate(r.errs)
        printfmt("      rho error t=%.3g   %s\n", ts[k + 1],
                 join((@sprintf("%s %.3e", REGION_NAMES[i], e[i]) for i in 1:4),
                      "  "))
    end
    for (k, q) in enumerate(r.qs)
        printfmt("      diffusivity number t=%.3g   %s\n", ts[k],
                 join((@sprintf("%s %.3e", REGION_NAMES[i], q[i]) for i in 1:4),
                      "  "))
    end
    if row.species == 2
        printfmt("      Y excursion outside [0,1]  shell %.3e  elsewhere %.3e\n",
                 r.exc[1], r.exc[2])
    end
    flush(stdout)
end

function crossing_rows()
    base = (; label="base (C6, subcycled)", mode=:levels, depth=2, subcycle=true,
            tile=0, scheme=:C6, closures=:neutral3, filter_cfl=0.35, art=true,
            ghosts=true, patch_grid=(1, 1, 1), species=1)
    rows = Any[base]
    HAS_GHOST_TOGGLE &&
        push!(rows, merge(base, (; label="sensor taps clamped", ghosts=false)))
    push!(rows, merge(base, (; label="C10", scheme=:C10)))
    push!(rows, merge(base, (; label="global dt", subcycle=false)))
    push!(rows, merge(base, (; label="tile 8", tile=8)))
    push!(rows, merge(base, (; label="three levels", depth=3)))
    push!(rows, merge(base, (; label="filter_cfl 0", filter_cfl=0.0)))
    for cl in (:cascade3, :cascade4, :brady_livescu)
        push!(rows, merge(base, (; label="deriv $cl", closures=cl)))
    end
    push!(rows, merge(base, (; label="art off", art=false)))
    push!(rows, merge(base, (; label="two patches", mode=:patches,
                             patch_grid=(2, 1, 1))))
    push!(rows, merge(base, (; label="three patches", mode=:patches,
                             patch_grid=(3, 1, 1))))
    species = Any[]
    for g in GHOST_MODES
        push!(species, merge(base, (; label="two species, " * ghost_label(g),
                                    species=2, ghosts=g)))
    end
    return rows, species
end

function crossing_part(N, ts, nmax)
    say("\n=== part crossing: Sod through a nest, sensors and filter live ===")
    say("root N = $N over [0, 1], slip walls, cfl = $CROSSING_CFL, ",
        "level-1 box $(crossing_regions(N, 2))")
    say("reference: the uniform run at the fine spacing, ", 3N - 2, " nodes, ",
        "with the row's own numerics")
    say("regions: shell = within $SHELL_W nodes of an interface face; fine = ",
        "the rest of a refined patch;")
    say("         covered = root nodes under a child level; root = the rest")
    say("sample times t = ", join(ts, ", "), " (the shock enters the box near ",
        "t = 0.057 and leaves near t = 0.17)")
    rows, species = crossing_rows()
    refs = Dict{Any,Any}()
    for row in vcat(rows, species)
        k = refkey(row)
        haskey(refs, k) && continue
        refs[k] = attempt(() -> reference_lines(row, N, ts, nmax))
        failed(refs[k]) &&
            say("  reference for $(row.label) did not complete: ", refs[k])
    end
    say("")
    run_rows(rows, N, ts, nmax, refs)
    say("\n--- two-species Sod, the contact carrying the composition ---")
    run_rows(species, N, ts, nmax, refs)
end

# A same-level grid needs one rank per patch off a single rank, where every
# patch is advanced in sequence instead.
function skip_reason(row)
    np = prod(row.patch_grid)
    (NP > 1 && np > NP) || return nothing
    return "needs $np ranks or one; this run has $NP"
end

function run_rows(rows, N, ts, nmax, refs)
    for row in rows
        why = skip_reason(row)
        if why !== nothing
            RANK == 0 && printfmt("  %-22s SKIPPED: %s\n", row.label, why)
            continue
        end
        r = with_ghosts(() -> crossing_row(row, N, ts, nmax, refs[refkey(row)]),
                        row.ghosts)
        print_crossing(row, r, ts)
    end
end

# --- part: filter -------------------------------------------------------------
#
# The filter runs from a callback with the solver's own pass off, so the
# change across each pass is the filter's alone; the interval is raised to one
# for the pass so `filter_weight` reads the solver's cadence, the pattern of
# bench/wallfilter.jl. A subcycled fine level filters inside `_advance_level!`
# and never through this callback, so every case here steps globally.

const FILTER_BINS = 6

mutable struct FilterTally
    shell::Vector{Float64}      # max |dQ| at distance 1..FILTER_BINS
    interior::Float64
    plane::Float64              # plane nodes of a two-dimensional shell
end
FilterTally() = FilterTally(zeros(FILTER_BINS), 0.0, 0.0)

function tally_filter!(tally, solver, states)
    before = [Array(view(parent(Q), :, :, :, 1)) for Q in states]
    solver.filter_interval = 1
    filter_state!(solver, states)
    solver.filter_interval = 0
    for (pi, (ps, Q)) in enumerate(CL.eachpatch(solver, states))
        ps.patch.level == 0 && continue
        n = ps.decomp.n_local
        gx = ps.decomp.offset[1]
        ngx = ps.decomp.n_global[1]
        for k in 1:n[3], j in 1:n[2], i in 1:n[1]
            I = gidx(ps, i, j, k)
            Δ = abs(Q[I, 1] - before[pi][I])
            d = face_distance(ps, (i, j, k))
            if d == 0 || d > FILTER_BINS
                tally.interior = max(tally.interior, Δ)
            else
                tally.shell[d] = max(tally.shell[d], Δ)
                # Plane nodes of the x faces: in one dimension the identity row
                # leaves them untouched, in two the transverse pass reaches them.
                (gx + i == 1 || gx + i == ngx) &&
                    (tally.plane = max(tally.plane, Δ))
            end
        end
    end
    return nothing
end

function reduce_tally!(tally, comm)
    tally.shell .= [MPI.Allreduce(v, max, comm) for v in tally.shell]
    tally.interior = MPI.Allreduce(tally.interior, max, comm)
    tally.plane = MPI.Allreduce(tally.plane, max, comm)
    return tally
end

function filter_run(build, tfinal, nmax)
    attempt() do
        s, Q = build()
        states = Q isa Vector ? Q : [Q]
        tally = FilterTally()
        cb = Callback(EveryStep(1),
                      (sv, st) -> (tally_filter!(tally, sv, st); nothing))
        run!(s, Q; tfinal=tfinal, nmax=nmax, callback=cb)
        reduce_tally!(tally, s.comm)
        return (; tally, steps=s.step)
    end
end

function print_filter(label, r)
    RANK == 0 || return
    if failed(r)
        printfmt("  %-12s %s\n", label, r)
        return
    end
    t = r.tally
    ratio(v) = t.interior > 0 ? v / t.interior : NaN
    printfmt("  %-12s steps %5d  interior %.3e  plane %.3e (%.3e of interior)\n",
             label, r.steps, t.interior, t.plane, ratio(t.plane))
    printfmt("               %s\n",
             join((@sprintf("d%d %.3e (%.2e)", d, t.shell[d], ratio(t.shell[d]))
                   for d in 1:FILTER_BINS), "  "))
    flush(stdout)
end

wave_ic(x, y, z) = Prim(rho=1.0 + 0.1 * sinpi(2x), u=(1.0, 0.0, 0.0), p=1.0)

# The two-dimensional row needs a transverse variation: a y-constant field
# passes the transverse filter unchanged and the plane column then reads
# round-off whatever the interface rows do.
wave2d_ic(x, y, z) = Prim(rho=1.0 + 0.1 * sinpi(2x) + 0.05 * sinpi(2(y + 1 / 9)),
                          u=(1.0, 0.5, 0.0), p=1.0)

function wave_build(fkw, N)
    () -> begin
        s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER3,
                   refine=probe_region(N), subcycle=false, filter_interval=0,
                   filter_cfl=0.35, control=StepControl(validity=:permissive);
                   fkw...)
        Q = allocate_state(s)
        initialize!(s, Q, wave_ic)
        s, Q
    end
end

function wave2d_build(fkw, N, ny)
    () -> begin
        ext = (N ÷ 3, ny ÷ 3, 1)
        s = Solver(n_global=(N, ny, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER3,
                   refine=BlockRegion((N ÷ 3, ny ÷ 3, 0), ext), subcycle=false,
                   filter_interval=0, filter_cfl=0.35,
                   control=StepControl(validity=:permissive); fkw...)
        Q = allocate_state(s)
        initialize!(s, Q, wave2d_ic)
        s, Q
    end
end

function sod_filter_build(fkw, N)
    () -> begin
        s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(WALL2, PER, PER), cfl=CROSSING_CFL,
                   refine=crossing_regions(N, 2), subcycle=false,
                   filter_interval=0, filter_cfl=0.35,
                   control=StepControl(validity=:permissive); fkw...)
        Q = allocate_state(s)
        initialize!(s, Q, sod_ic)
        s, Q
    end
end

function filter_part(waveN, wave_tfinal, sodN, sod_tfinal, nmax, ny)
    say("\n=== part filter: the filter's own change at an imposed shell ===")
    say("max |d(rho)| over the run's passes, by distance in fine nodes from a ",
        "coarse-fine face")
    say("(distance 1 is the boundary plane), over the fine interior; global ",
        "stepping throughout")
    say("\n--- entropy wave, root N = $waveN, t = $wave_tfinal ---")
    for (flab, fkw) in FILTER_ROWS
        print_filter(flab, filter_run(wave_build(fkw, waveN), wave_tfinal, nmax))
    end
    say("\n--- Sod crossing, root N = $sodN, t = $sod_tfinal ---")
    for (flab, fkw) in FILTER_ROWS
        print_filter(flab, filter_run(sod_filter_build(fkw, sodN), sod_tfinal, nmax))
    end
    say("\n--- entropy wave on ($waveN, $ny, 1) with a 2-D box: the plane ",
        "column is the transverse pass ---")
    for (flab, fkw) in FILTER_ROWS
        print_filter(flab, filter_run(wave2d_build(fkw, waveN, ny),
                                      wave_tfinal, nmax))
    end
end

# --- part: undershoot ---------------------------------------------------------
#
# `layer_ic` and `regions` mirror bench/interfaceconservation.jl, whose
# two-species layer this part reuses; they are copied rather than included so
# that script stays a standalone instrument.

function layer_ic()
    δ = 0.22
    return (x, y, z) -> begin
        y2 = 0.5 * (1 + tanh(sin(x / 4 - 0.12sin(y)) / δ))
        Prim(Y=(1 - y2, y2), rho=1.0, p=1.0, u=(1.0, 0.08sin(x / 4), 0.0))
    end
end

function layer_regions(N, ny, depth=2)
    ylo = max(4, ny ÷ 4)
    r1 = BlockRegion((N ÷ 2 - N ÷ 6, ylo, 0),
                     (N ÷ 3, min(ny - 4, ylo + ny ÷ 2) - ylo, 1))
    depth == 2 && return r1
    # A further four-node imposed shell surrounds level 1, so the level-2
    # region starts eight level-1 nodes from the nominal parent edge.
    n1x, n1y = 3r1.extent[1] - 2, 3r1.extent[2] - 2
    r2 = BlockRegion((3r1.offset[1] + 8, 3r1.offset[2] + 8, 0),
                     (n1x - 16, n1y - 16, 1))
    return [r1, r2]
end

const UNDER_BINS = 5      # 0, 1, 2, 3, 4+

# Distance of a root node from the level-1 box boundary, in root nodes, and
# whether it lies inside the box. A node outside takes the Chebyshev distance
# to the box; a node inside takes the distance to the nearest face, so 0 is the
# coarse node coincident with the fine boundary plane either way.
function box_distance(region, gi, gj)
    lo = (region.offset[1] + 1, region.offset[2] + 1)
    hi = (region.offset[1] + region.extent[1], region.offset[2] + region.extent[2])
    g = (gi, gj)
    out = ntuple(d -> max(lo[d] - g[d], g[d] - hi[d], 0), 2)
    any(>(0), out) && return maximum(out), false
    return minimum(min(g[d] - lo[d], hi[d] - g[d]) for d in 1:2), true
end

bin_of(d) = min(d, UNDER_BINS - 1) + 1

function undershoot_sample(solver, states)
    region = CL.refined_region(solver)
    # Inf marks a bin no node fell in; the box face is always counted inside,
    # so the outside row has no distance-0 entry.
    inside = fill(Inf, UNDER_BINS)
    outside = fill(Inf, UNDER_BINS)
    fine = fill(Inf, UNDER_BINS + 1)
    fine2 = fill(Inf, UNDER_BINS + 1)   # level 2 and below, by their own planes
    refresh_primitives!(solver, states)
    for (ps, Q) in CL.eachpatch(solver, states)
        n = ps.decomp.n_local
        off = ps.patch.region.offset .+ ps.decomp.offset
        for k in 1:n[3], j in 1:n[2], i in 1:n[1]
            I = gidx(ps, i, j, k)
            y = minimum(Y[I] for Y in ps.Y)
            if ps.patch.level == 0
                d, isin = box_distance(region, off[1] + i, off[2] + j)
                b = bin_of(d)
                isin ? (inside[b] = min(inside[b], y)) :
                       (outside[b] = min(outside[b], y))
            else
                # `face_distance` counts the boundary plane as 1, so bin b
                # holds the nodes b − 1 fine nodes in from an imposed plane.
                b = min(face_distance(ps, (i, j, k)), UNDER_BINS + 1)
                if ps.patch.level == 1
                    fine[b] = min(fine[b], y)
                else
                    fine2[b] = min(fine2[b], y)
                end
            end
        end
    end
    red(v) = [MPI.Allreduce(x, min, solver.comm) for x in v]
    return red(inside), red(outside), red(fine), red(fine2)
end

function undershoot_run(label, N, ny, tfinal, samples, nmax; subcycle, filt_int,
                        ghosts, depth=2, filter_cfl=0.35)
    attempt() do
        s = Solver(n_global=(N, ny, 1), L_domain=(8pi, 2pi, 1.0), bcs=PER3,
                   eos=two_gases(), cfl=0.45,
                   art=ArtParams(C_mu=0.0, C_beta=0.0, C_kappa=0.0, C_D=0.0),
                   control=StepControl(validity=:permissive),
                   filter_interval=filt_int, filter_cfl=filter_cfl,
                   refine=layer_regions(N, ny, depth),
                   subcycle=subcycle)
        Q = allocate_state(s)
        initialize!(s, Q, layer_ic())
        states = Q isa Vector ? Q : [Q]
        ws = Workspace(Q)
        out = Any[]
        for t in range(tfinal / samples, tfinal; length=samples)
            with_ghosts(() -> run!(s, Q, ws; tfinal=t, nmax=nmax), ghosts)
            push!(out, (s.t, s.step, undershoot_sample(s, states)...))
            s.step >= nmax && break
        end
        return out
    end
end

function print_undershoot(label, r)
    RANK == 0 || return
    if failed(r)
        printfmt("  %-34s %s\n", label, r)
        return
    end
    say("  " * label)
    cell(name, v) = isfinite(v) ? @sprintf("%s %+.3e", name, v) :
                    @sprintf("%s %10s", name, "n/a")
    bins(v, last) = join((cell(d == length(v) ? last : string(d - 1), v[d])
                          for d in eachindex(v)), "  ")
    for (t, step, inside, outside, fine, fine2) in r
        printfmt("    t=%9.4f step %6d  root in  %s\n", t, step,
                 bins(inside, "4+"))
        printfmt("    %-22s root out %s\n", "", bins(outside, "4+"))
        printfmt("    %-22s level 1  %s\n", "", bins(fine, "5+"))
        any(isfinite, fine2) &&
            printfmt("    %-22s level 2  %s\n", "", bins(fine2, "5+"))
    end
    flush(stdout)
end

function undershoot_part(N, ny, tfinal, samples, nmax)
    say("\n=== part undershoot: the root-edge mass-fraction undershoot ===")
    say("two-species layer on ($N, $ny, 1) over [0, 8pi) x [0, 2pi), only the ",
        "mass-fraction bound on,")
    say("permissive validity, level-1 box ", layer_regions(N, ny),
        ", level-2 box ", layer_regions(N, ny, 3)[2])
    say("minimum Y by distance in root nodes from the level-1 box edge (0 is ",
        "the coarse node")
    say("coincident with the fine boundary plane; RESTRICT_MARGIN = ",
        CL.RESTRICT_MARGIN, " root nodes inside the box keep")
    say("coarse-evolved values), and by distance in fine nodes from each ",
        "level's own imposed plane")
    depths = Set(parse.(Int, split(args.undershoot_depths, ',')))
    variants = Any[("two levels, global step", (subcycle=false, filt_int=1)),
                   ("two levels, global step, no filter",
                    (subcycle=false, filt_int=0)),
                   ("two levels, subcycled", (subcycle=true, filt_int=1)),
                   ("three levels, global step",
                    (subcycle=false, filt_int=1, depth=3)),
                   ("three levels, subcycled",
                    (subcycle=true, filt_int=1, depth=3)),
                   # The relaxed weight of a root pass under global stepping is
                   # the finest level's step over the root's own; the unrelaxed
                   # pass separates that weight from the stepping itself.
                   ("three levels, global step, unrelaxed filter",
                    (subcycle=false, filt_int=1, depth=3, filter_cfl=0.0))]
    wanted = args.undershoot_variants == "all" ? nothing :
             split(args.undershoot_variants, ',')
    for (vlab, vkw) in variants, g in GHOST_MODES
        get(vkw, :depth, 2) in depths || continue
        wanted === nothing || any(w -> occursin(w, vlab), wanted) || continue
        label = "$vlab, $(ghost_label(g))"
        r = undershoot_run(label, N, ny, tfinal, samples, nmax;
                           vkw..., ghosts=g)
        print_undershoot(label, r)
    end
end

# --- driver -------------------------------------------------------------------

function main()
    parts = Set(Symbol.(split(args.parts, ',')))
    known = (:all, :probe, :crossing, :filter, :undershoot)
    all(p -> p in known, parts) ||
        error("parts must be all or a subset of probe, crossing, filter, " *
              "undershoot")
    allparts = :all in parts
    ns = parse.(Int, split(args.ns, ','))
    N, nmax = args.N, args.nmax
    waveN, wave_tfinal = args.waveN, args.wave_tfinal
    layerN, layerny = args.layerN, args.layerny
    layer_tfinal, samples = args.layer_tfinal, args.samples
    ts = (0.03, 0.1, 0.2)
    sod_tfinal = 0.2
    ny = 24
    if args.smoke
        ns = [24, 48]
        N = 201
        ts = (0.005, 0.01, 0.02)
        nmax = min(nmax, 200)
        sod_tfinal = 0.02
        waveN, wave_tfinal = 48, 0.05
        layerN, layerny, layer_tfinal, samples = 48, 24, 0.4, 2
        ny = 18
    end
    say("=== interface sensors and filter: np=$NP, parts=$(args.parts), ",
        "smoke=$(args.smoke) ===")
    say("sensor interface taps: ",
        HAS_GHOST_TOGGLE ? "toggle present, both settings run" :
        "toggle absent in this build, clamped taps only")
    if allparts || :probe in parts
        probe_part(ns, args.nodes)
    end
    if allparts || :crossing in parts
        crossing_part(N, ts, nmax)
    end
    if allparts || :filter in parts
        filter_part(waveN, wave_tfinal, N, sod_tfinal, nmax, ny)
    end
    if allparts || :undershoot in parts
        undershoot_part(layerN, layerny, layer_tfinal, samples, nmax)
    end
end

mpi_main(main)
