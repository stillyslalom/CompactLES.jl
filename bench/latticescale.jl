# The lattice decision of a regrid check at a cluster's rank count, without a
# cluster: the functions a check runs on every rank or on rank 0 (the chain,
# the wanted tiles of each lattice, the pricing of each under the placement
# it would get, the stored-ownership placement, the inherited creation
# checks, the fold of the gathered timings, the nesting test of a deeper
# level), called serially on synthetic tile sets at P ranks, and the
# predicted step of the lattice and placement each rule takes.
#
# Geometry: a 3-D Richtmyer–Meshkov channel, 4 : 1 : 1, of `B`³ root nodes
# per rank, so the root grows with P as a weak-scaling run does. Four
# features cross it: an incident and a reflected shock (planes), a
# transmitted shock with a small ripple, and a mixing layer of thickness
# 3% of the channel length with a ripple of amplitude 2% and wavelength half
# the width (`scene=front`), or, early in a run, the unmixed interface and
# the incident shock alone (`scene=thin`), which leaves fewer tiles than
# ranks on a level at the larger counts. A cell is tagged when its buffered
# node range meets a feature's band; the old layout is the same features a
# half-tile behind, on the configured lattice. Level 2 refines the mixing
# layer alone, in level 1's node space, on its configured lattice, priced on
# the ranks level 1 spans.
#
# Part 1 times each function at every P (seconds, serial, one rank's share
# of the check); the cost of the decision is what a rank pays at every
# check. Part 2 prices four rules under one cost model: `configured` the
# lattice given in the `AMR` keyword, `count` the coarsest lattice of the
# chain with a tile per rank, `model, flat` the lattice the model picks with
# every message at one cost and the placement of a single node, and
# `model, nodes` the lattice the model picks with the placement and the
# message costs of nodes of `node` ranks. Each rule's lattice and placement
# is then priced under the same model, `m_x = ratio m`, so the columns
# compare like with like. The coefficients (`coeffs`, a/b/k/m/c) default to
# those the model fitted on the workstation on bench/rmicost.jl 3-D amr2 at
# eight ranks (reference/CALIBRATION_APPENDIX.md); `ratio`, the cost of a
# message leaving a node against one within it, is not measured anywhere
# yet and is swept. The root costs `a` per node of its largest block. Every
# number of part 2 is a prediction of the model, not a measurement.
#
#   julia --project=. -t 1 bench/latticescale.jl
#   julia --project=. -t 1 bench/latticescale.jl P=112,1120 scene=thin ratio=1,10
#
# Options: `P` rank counts, `B` root nodes per rank per dimension, `tile`
# the configured edge, `node` ranks per node, `ratio` the m_x / m values,
# `coeffs` the model's a/b/k/m/c,
# `scene` front or thin, `levels` 1 or 2 (amr2 or amr3), `parts` 1, 2 or
# 12, and `affinity` a CPU mask for the process (0x5555: one logical CPU per
# performance core of the 12900K workstation; 0 leaves it alone).

using CompactLES
using Printf

const CL = CompactLES

const OPTIONS = CL.script_args(ARGS, (P="112,1120,11200", B=48, tile=16, node=112,
                                   ratio="1,3,10", scene="front", levels=2,
                                   parts="12", affinity=0,
                                   coeffs="2.15e-6/3.2e-7/0.028/5.4e-3/0.044"))

if OPTIONS.affinity != 0 && Sys.iswindows()
    ccall((:SetProcessAffinityMask, "kernel32"), stdcall, Cint, (Ptr{Cvoid}, Csize_t),
          ccall((:GetCurrentProcess, "kernel32"), stdcall, Ptr{Cvoid}, ()), OPTIONS.affinity)
end

ints(s) = parse.(Int, split(s, ','))
floats(s) = parse.(Float64, split(s, ','))

# Seconds per node (a), per node and interface dimension (b), per tile (k)
# and per message (m), each per step of the level, and per tile of the
# busiest rank at a check (c).
const COEFFS = NamedTuple{(:a, :b, :k, :m, :c)}(parse.(Float64, split(OPTIONS.coeffs, '/')))

const INTERVAL = 4
const BUFFER = 4

struct Feature
    x::Float64
    amp::Float64
    half::Float64
end

root_nodes(P, B) = (ny = round(Int, B * cbrt(P / 4)); (4ny, ny, ny))

# The features of level 1 over a node space of `N` nodes, or of level 2,
# which refines the mixing layer alone.
function features(N, scene, deep)
    nx = N[1]
    layer = scene == "thin" ? Feature(0.35nx, 0.003nx, 3.0) :
            Feature(0.35nx, 0.02nx, 0.015nx)
    deep && return [layer]
    scene == "thin" && return [Feature(0.20nx, 0.0, 2.0), layer]
    return [Feature(0.20nx, 0.0, 2.0), layer, Feature(0.50nx, 0.0, 2.0),
            Feature(0.62nx, 0.005nx, 2.0)]
end

cosrange(lo, hi, λ) = extrema(cos(2π * y / λ) for y in lo:hi)

# The cell flags of lattice `a` over a node space of `N` nodes: the tag level
# where a buffered cell meets a feature's band, the hold level a little wider.
function lattice_flags(N, a, feats, λ, b, shift)
    K = ntuple(d -> (N[d] - 1) ÷ a[d] + 1, 3)
    flags = zeros(Int8, K)
    for k3 in 0:K[3]-1, k2 in 0:K[2]-1
        y = cosrange(max(k2 * a[2] + 1 - b, 1), min((k2 + 1) * a[2] + 1 + b, N[2]), λ)
        z = cosrange(max(k3 * a[3] + 1 - b, 1), min((k3 + 1) * a[3] + 1 + b, N[3]), λ)
        prods = (y[1] * z[1], y[1] * z[2], y[2] * z[1], y[2] * z[2])
        cmin, cmax = minimum(prods), maximum(prods)
        for f in feats, (w, level) in ((f.half + 2, CL.TAG_HOLD), (f.half, CL.TAG_MARK))
            lo = floor(Int, f.x + shift + f.amp * cmin - w) - b
            hi = ceil(Int, f.x + shift + f.amp * cmax + w) + b
            for k1 in intersect(CL._tile_span(lo, hi, a[1]), 0:K[1]-1)
                flags[k1 + 1, k2 + 1, k3 + 1] = max(flags[k1 + 1, k2 + 1, k3 + 1], level)
            end
        end
    end
    return flags, K
end

# The tiles of the cells at the tag level.
function marked_tiles(flags, K, a, lo, hi)
    out = BlockRegion[]
    for k3 in 0:K[3]-1, k2 in 0:K[2]-1, k1 in 0:K[1]-1
        flags[k1 + 1, k2 + 1, k3 + 1] == CL.TAG_MARK || continue
        t = CL._lattice_tile((k1, k2, k3), (true, true, true), a, lo, hi)
        t === nothing || push!(out, t)
    end
    return out
end

# A regrid specification to read the hysteresis settings from.
function spec_of()
    per = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    s = Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per,
               refine=BlockRegion((16, 0, 0), (17, 1, 1)), tile=8, regrid_interval=5)
    spec = getfield(s, :regrid)
    spec.checks = 10
    return s, spec
end

# One level's node space, chain and tile sets.
struct LevelSet
    N::NTuple{3,Int}
    chain::Vector{NTuple{3,Int}}
    flags::Vector{Array{Int8,3}}
    Ks::Vector{NTuple{3,Int}}
    wanted::Vector{Vector{BlockRegion}}
    old::Vector{BlockRegion}
    lo::NTuple{3,Int}
    hi::NTuple{3,Int}
end

function level_set(N, P, tile, scene, margin, deep=false)
    feats = features(N, scene, deep)
    λ = N[2] / 2
    lo = (1 + margin, 1 + margin, 1 + margin)
    hi = N .- margin
    chain = CL._edge_chain(tile, (true, true, true), N, CL._NO_BOUNDARY, CL._NO_BOUNDARY,
                           margin, 8, CL.MIN_TILE_EDGE, P, CL._lattice_budget(P))
    flags = Array{Int8,3}[]
    Ks = NTuple{3,Int}[]
    for a in chain
        f, K = lattice_flags(N, a, feats, λ, BUFFER, 0.0)
        push!(flags, f)
        push!(Ks, K)
    end
    wanted = [marked_tiles(f, K, a, lo, hi) for (f, K, a) in zip(flags, Ks, chain)]
    fo, Ko = lattice_flags(N, chain[1], feats, λ, BUFFER, -tile[1] / 2)
    old = marked_tiles(fo, Ko, chain[1], lo, hi)
    return LevelSet(N, chain, flags, Ks, wanted, old, lo, hi)
end

function costs_of(ratio, nodes)
    c = CL.LatticeCosts()
    c.a, c.b, c.k, c.m, c.c = COEFFS.a, COEFFS.b, COEFFS.k, COEFFS.m, COEFFS.c
    c.m_x = ratio * COEFFS.m
    c.nodes = nodes
    c.ready = true
    return c
end

const ACTIVE = (true, true, true)

# The predicted seconds per root step of a level of `regions` on `np` ranks,
# its tiles placed as the regrid would with the node map `place` and priced
# under `costs` (its own node map), `s` steps per root step.
function level_cost(costs, regions, np, s, place)
    isempty(regions) && return 0.0
    bnd = fill(CL._NO_BOUNDARY, length(regions))
    owners = CL._predicted_owners(regions, ACTIVE, np, place)
    return CL._layout_cost(costs, regions, bnd, bnd, owners, ACTIVE, (0, 0, 0), s, np) +
           costs.c * CL._most_held(owners) / INTERVAL
end

# The lattices each rule takes, level 1 first with level 2 on its configured
# lattice, then level 2 on the ranks level 1 spans, and the predicted step
# of each rule's lattices and placement under the true model.
const RULES = ("configured", "count", "model, flat", "model, nodes")

function decide(l1, l2, P, truth, flat, nodes)
    root = COEFFS.a * prod(cld.(l1.N, CL._amr_dims_or_nothing(l1.N, ACTIVE, P)))
    # Level 2's price depends on the ranks level 1 leaves it alone, which
    # most lattices share.
    memo = Dict{Tuple{UInt,Int,Int,Bool},Float64}()
    deep(costs, i2, n1, place) =
        l2 === nothing ? 0.0 :
        get!(memo, (objectid(costs), i2, CL._level_rank_count(n1, P), isempty(place))) do
            level_cost(costs, l2.wanted[i2], CL._level_rank_count(n1, P), 9.0, place)
        end
    out = []
    for rule in RULES
        costs, place = rule == "model, nodes" ? (truth, nodes) : (flat, Int[])
        pick(set, np, price) =
            rule == "configured" ? 1 :
            rule == "count" ?
                something(findfirst(w -> length(w) >= np, set.wanted), length(set.chain)) :
                argmin([price(i) for i in eachindex(set.chain)])
        i1 = pick(l1, P, i -> level_cost(costs, l1.wanted[i], P, 3.0, place) +
                              deep(costs, 1, length(l1.wanted[i]), place))
        n1 = length(l1.wanted[i1])
        r2 = CL._level_rank_count(n1, P)
        i2 = l2 === nothing ? 0 :
             pick(l2, r2, i -> level_cost(costs, l2.wanted[i], r2, 9.0, place))
        step = root + level_cost(truth, l1.wanted[i1], P, 3.0, place) +
               deep(truth, max(i2, 1), n1, place)
        push!(out, (rule, i1, i2, r2, step))
    end
    return out
end

function part1(P, l1, l2, spec, solver, nodes)
    costs = costs_of(3.0, nodes)
    times = Dict{String,Float64}()
    times["chain"] = @elapsed CL._edge_chain(l1.chain[1], ACTIVE, l1.N, CL._NO_BOUNDARY,
                                             CL._NO_BOUNDARY, spec.margin, 8,
                                             CL.MIN_TILE_EDGE, P, CL._lattice_budget(P))
    empty!(spec.created)
    foreach(r -> spec.created[r] = 0, l1.old)
    times["wanted"] = @elapsed for i in eachindex(l1.chain)
        CL._lattice_wanted(l1.flags[i], l1.Ks[i], l1.chain[i], ACTIVE, l1.lo, l1.hi, spec,
                           l1.old, i == 1, (0, 0, 0))
    end
    times["price"] = @elapsed for w in l1.wanted
        level_cost(costs, w, P, 3.0, nodes)
    end
    oldown = CL._predicted_owners(l1.old, ACTIVE, P, nodes)
    w = l1.wanted[1]
    times["place"] = @elapsed begin
        owners, np_new = CL._place_tiles(w, ACTIVE, P, l1.old, oldown)
        oldset = Set(l1.old)
        CL._balanced_owners(w, ACTIVE, P, owners, np_new, [r in oldset for r in w]; nodes)
    end
    fine = l1.wanted[end]
    times["inherit"] = @elapsed begin
        bins = CL._RegionIndex(l1.old, (0, 0, 0))
        for r in fine
            CL._inherited_check(r, view(l1.old, CL._candidates(bins, r)), spec.created,
                                spec.checks, ACTIVE, (0, 0, 0))
        end
    end
    # The fold of the gathered records: one per rank, its tiles' fields.
    nlev = length(getfield(solver, :levels))
    per = max(1, cld(length(w), P))
    records = [vcat(fill(1e-3, 2nlev), [0.0, 0.0, 0.0],
                    repeat([1.0, 97336.0, 3.0, 0.3, 0.01], per)) for _ in 1:P]
    sc = getfield(solver, :regrid)
    times["fold"] = @elapsed begin
        CL._fold_lattice_costs!(solver, sc, records, nlev)
        CL._fit_lattice_costs(sc.costs)
    end
    # The nesting test of level 2's tiles in level 1's.
    times["nest"] = NaN
    if l2 !== nothing
        parents = [CL._fine_region(r, ACTIVE) for r in l1.wanted[1]]
        times["nest"] = @elapsed begin
            bins = CL._RegionIndex(parents, (0, 0, 0))
            for t in l2.wanted[1]
                box = CL._buffered(t, ACTIVE, spec.margin)
                CL._covered_by(box, parents[CL._candidates(bins, box)])
            end
        end
    end
    bytes = sum(prod, l1.Ks)
    @printf("%6d  %5d  %9d  %8.1f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f\n",
            P, length(l1.chain), maximum(length, l1.wanted), bytes / 1e6, times["chain"],
            times["wanted"], times["price"], times["place"], times["inherit"],
            times["fold"], times["nest"])
end

function part2(P, l1, l2, ratio, nodes)
    truth = costs_of(ratio, nodes)
    flat = costs_of(1.0, Int[])
    t = @elapsed rows = decide(l1, l2, P, truth, flat, nodes)
    best = minimum(r[5] for r in rows)
    for (name, i1, i2, r2, step) in rows
        n1 = length(l1.wanted[i1])
        deep = l2 === nothing ? "" :
               @sprintf("  %-12s %7d  %6.2f", string(l2.chain[i2]), length(l2.wanted[i2]),
                        length(l2.wanted[i2]) / r2)
        @printf("%6d  %5.0f  %-13s %-12s %7d  %6.2f%s  %9.3f  %6.3f
", P, ratio, name,
                string(l1.chain[i1]), n1, n1 / P, deep, step, step / best)
    end
    @printf("%6d  %5.0f  (decision priced in %.2f s)
", P, ratio, t)
end

function main()
    opt = OPTIONS
    Ps = ints(opt.P)
    tile = (opt.tile, opt.tile, opt.tile)
    solver, spec = spec_of()
    # Compile every function on a small case first.
    for warm in (true, false)
        sets = Dict{Int,Any}()
        for P in (warm ? [8] : Ps)
            N = root_nodes(P, opt.B)
            l1 = level_set(N, P, tile, opt.scene, spec.margin)
            l2 = opt.levels >= 2 ? level_set(3 .* (N .- 1) .+ 1, P, tile, opt.scene,
                                              spec.margin, true) : nothing
            sets[P] = (l1, l2)
        end
        if occursin('1', opt.parts)
            warm || println("\npart 1: seconds of each step of the decision, serial\n",
                            "     P  chain  max tiles  flags MB    chain   wanted    price",
                            "    place  inherit     fold     nest")
            for P in sort!(collect(keys(sets)))
                l1, l2 = sets[P]
                nodes = [r ÷ opt.node for r in 0:(P - 1)]
                warm ? redirect_stdout(devnull) do
                           part1(P, l1, l2, spec, solver, nodes)
                       end : part1(P, l1, l2, spec, solver, nodes)
            end
        end
        if occursin('2', opt.parts)
            warm || println("\npart 2: predicted s per root step of each rule's level-1 ",
                            "lattice and placement, ", opt.levels, " level(s), scene ",
                            opt.scene, "\n     P  ratio  rule          level 1       ",
                            "  tiles   /rank  level 2        tiles   /rank     s/step",
                            "  vs best")
            for P in sort!(collect(keys(sets))), ratio in floats(opt.ratio)
                l1, l2 = sets[P]
                nodes = [r ÷ opt.node for r in 0:(P - 1)]
                warm ? redirect_stdout(devnull) do
                           part2(P, l1, l2, ratio, nodes)
                       end : part2(P, l1, l2, ratio, nodes)
            end
        end
    end
end

main()
