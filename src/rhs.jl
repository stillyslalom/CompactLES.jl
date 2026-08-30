# Solver container and conservative multicomponent Navier–Stokes RHS in
# orthogonal curvilinear coordinates, with collapsed (1-D/2-D) dimensions and
# a regularized cylindrical axis.
#
# Collapsed dimensions (n_global[d] == 1) carry no derivatives, filters, halos,
# or exchanges, but keep their velocity component and all metric source
# terms, as axisymmetric (r, z) flow with optional swirl
# requires. Coordinate singularities (cylindrical axis, spherical origin and
# poles, axisymmetric or fully resolved) are regularized
# by a half-offset grid, r_i = (i − ½)h, plus parity conditions: halos below
# the axis are mirror-filled with per-field signs (even scalars/u_z, odd
# u_r/u_θ), interior stencils run all the way to the first node, and the
# implicit LHS coupling to the ghost unknown is folded analytically onto the
# diagonal (per solution parity σg: derivatives flip field parity, filters
# preserve it). No node sits at r = 0 and no scale factor vanishes.

# The patch parameter `P` is unconstrained, not `P <: Patch{T}`: a
# refined solver stores its root and level-1 patches in one vector, and the
# two differ in their boundary-condition type, so `P` is their typejoin. The
# per-patch loops then cost one dynamic dispatch per patch per call, behind
# the PatchSolver function barrier so the bodies stay concrete, while the
# single-patch and same-level multi-patch cases keep a concrete `P`.
mutable struct Solver{T,Eq<:EquationSet,E<:EOS,M<:Metric,St,Src,P}
    equations::Eq
    eos::E
    transport::Transport{T}
    art::ArtParams{T}
    metric::M
    stretch::St
    sources::Src
    L_domain::NTuple{3,T}
    origin::NTuple{3,T}
    coord_shift::NTuple{3,T}                # half-cell offset (axis grids)
    h::NTuple{3,T}
    cfl::T
    filter_interval::Int
    filter_cfl::T                           # 0 = unrelaxed; see filter_weight
    control::StepControl                    # timestep floors, prediction, retry
    n_global::NTuple{3,Int}                 # whole-grid extents (all patches)
    # Per-patch state: everything from the decomposition and the operator plans
    # through the field arrays lives on `Patch` (patches.jl). With one patch,
    # `Base.getproperty` below forwards the patch-owned names to it, so
    # `solver.rho`, `solver.decomp` and the rest read as they always did.
    patches::Vector{P}                      # this rank's patches, globally ordered
    patch_regions::Vector{BlockRegion}      # every patch, id order, global nodes
    comm::MPI.Comm                          # reduction communicator spanning all
                                            # ranks (the sole patch's Cartesian
                                            # communicator when there is one patch)
    ghost_sends::Vector{GhostRecord{T}}     # interface exchange records; empty
    ghost_recvs::Vector{GhostRecord{T}}     # with one patch
    plane_pairs::Vector{PlaneRecord{T}}
    levels::Vector{Level{T}}                # the refinement hierarchy, root
                                            # first; one entry without refinement
                                            # (levels.jl)
    subcycle::Bool                          # Berger–Oliger subcycling; timestep.jl
    regrid::Union{Nothing,RegridSpec{T}}    # tagging + regridding; regrid.jl
    t::T
    tstage::T
    step::Int
    dt_prev::T                              # last accepted step, for growth capping
    rate_prev::T                            # last CFL rate, for the predictor
    # Wall-clock accounting, filled in by `run!`. Rank-local: a reduction here
    # would be a collective on every step, paid by every run whether or not
    # anything reads it. Callers wanting load imbalance reduce these
    # themselves at their own (much lower) reporting frequency; see
    # `ProgressLog`. Seconds, and Float64 regardless of T.
    wall_step::Float64                      # last completed step, excl. callbacks
    wall_total::Float64                     # cumulative over the run
    # What the positivity failsafe has seen and repaired. Global, unlike the
    # rank-local wall-clock fields above: the repair reduces its own
    # tally on every step it runs, and it runs only when
    # `StepControl.floor_ratio` is set, so a run that leaves it off pays nothing.
    floor_tally::FloorTally
end

# Patch-owned property names forward to the sole patch, which keeps every
# existing consumer of the pre-patch field layout (tests, benches, examples,
# and the compute path itself in the single-patch case) reading unchanged.
# On a multi-patch solver those names have no single answer, so the forward
# throws; the multi-patch drivers hand each routine a `PatchSolver` instead.
@inline function Base.getproperty(s::Solver, name::Symbol)
    if _is_patch_prop(name)
        ps = getfield(s, :patches)
        length(ps) == 1 || _patch_prop_error(name)
        return getfield(@inbounds(ps[1]), name)
    end
    return getfield(s, name)
end

@noinline _patch_prop_error(name::Symbol) =
    error("property `$name` is per-patch state and this solver holds several " *
          "patches; access it through solver.patches or a per-patch driver")

"""
Union of the two objects the compute path accepts: a (single-patch) `Solver`,
whose property forwarding exposes its sole patch, and a [`PatchSolver`](@ref)
binding one patch of a multi-patch solver. Every routine between the
step drivers and the arrays is written against this type.
"""
const SolverLike{T} = Union{Solver{T},PatchSolver{T}}

"Number of patches this rank's solver holds."
npatches(s::Solver) = length(getfield(s, :patches))

"Iterate `(patch_solver, state)` pairs over this rank's patches, in the global
patch order, with `states` aligned with `solver.patches`."
eachpatch(solver::Solver, states) =
    ((PatchSolver(solver, p), states[i]) for (i, p) in
     enumerate(getfield(solver, :patches)))

"""
    Solver(; n_global, L_domain, bcs, kwargs...)

Backend constructor: allocate the distributed solver, plan every compact
operator, and fill the geometry. [`setup`](@ref) is the normal entry point and
calls this after splitting a [`Problem`](@ref) from a [`Numerics`](@ref);
construct a `Solver` directly when there is no `Problem` to build from, as the
test and benchmark suites do. It allocates no conserved state, so pair it with
[`allocate_state`](@ref) and [`initialize!`](@ref).

# Required keywords

- `n_global`: global point count per direction. A count of one collapses that
  direction, as described under [`Numerics`](@ref).
- `L_domain`: three domain *extents*, one per direction. Note the difference from
  `Problem.domain`, which gives endpoints; here `origin` carries the low corner.
- `bcs`: three `(low, high)` pairs of [`BoundaryCondition`](@ref) objects.

# Optional keywords

`eos`, `transport`, `metric`, and `sources` take their defaults and meaning from
[`Problem`](@ref); `art`, `deriv`, `filt`, `cfl`, `control`, `filter_interval`,
`filter_cfl`, `dims`, `n_halo`, and `stretch` from [`Numerics`](@ref). The two with no
`Problem`/`Numerics` counterpart are:

- `origin`: low corner of the domain, one value per direction. Default
  `(0.0, 0.0, 0.0)`. A folded direction requires its origin at zero. A stretched
  direction ignores this entry: its computational coordinate runs over [0, 1] and
  the [`Stretch`](@ref) mapping carries both endpoints.
- `equations`: the [`EquationSet`](@ref) owning the conserved layout. The default
  builds [`NavierStokes1T`](@ref) from `eos`; a supplied set must agree with
  `eos` on the species count.

# Geometry selected by the boundary conditions

Collapsed dimensions and coordinate singularities are not keywords: setup
recognizes them from `bcs` and validates the rest of the configuration against
them.

- Collapsed dimensions: set `n_global[d] = 1` with `(PeriodicBC(), PeriodicBC())`
  for that dimension; e.g. axisymmetric cylindrical is `n_global = (Nr, 1, Nz)`.
- Cylindrical axis: `bcs[1] = (AxisBC(), <outer bc>)` with
  `metric = CylindricalMetric()`, `origin[1] = 0`, and dimension 1 unstretched.
  θ may be collapsed (axisymmetric) or resolved over 2π with an even point
  count. The grid is half-offset in r, so no node sits at r = 0.
- Spherical origin and poles: [`OriginBC`](@ref) at the low end of r and
  [`PoleBC`](@ref) at *both* ends of θ, with `metric = SphericalMetric()`. The
  origin additionally requires a θ range symmetric about π/2, and the poles the
  full range (0, π). Either may be combined with the other. φ may be collapsed
  or resolved over 2π with an even point count, as θ may be at the axis.
"""
function Solver(; n_global::NTuple{3,Int}, L_domain, bcs,
                eos::EOS=single_species(),
                equations=nothing,
                transport::Transport{T}=Transport(),
                art::ArtParams=ArtParams(),
                metric::Metric=CartesianMetric(),
                stretch::NTuple{3,Union{Nothing,Stretch}}=(nothing, nothing, nothing),
                sources=(),
                origin=(0.0, 0.0, 0.0),
                deriv::AbstractCompactScheme=lele_d1_6(),
                filt::AbstractCompactScheme=compact_filter(0.45),
                cfl::Real=0.5, filter_interval::Int=1, filter_cfl::Real=0.0,
                control::StepControl=StepControl(),
                dims=nothing, n_halo::Int=4,
                comm::MPI.Comm=MPI.COMM_WORLD,
                patch_grid::NTuple{3,Int}=(1, 1, 1),
                backend::AbstractBackend=CPUBackend(),
                interface_rhs::Symbol=:extended,
                refine::Union{Nothing,BlockRegion,Vector{BlockRegion}}=nothing,
                level_restriction::Symbol=:inject,
                subcycle::Bool=false,
                regrid_interval::Int=0,
                tag_threshold::Real=0.02,
                tag_buffer::Int=4,
                tile::Int=0) where {T}
    for d in 1:3
        isperiodic(bcs[d][1]) == isperiodic(bcs[d][2]) ||
            error("dimension $d mixes periodic and non-periodic conditions")
        n_global[d] > 1 || (isperiodic(bcs[d][1]) ||
            error("collapsed dimension $d must use (PeriodicBC(), PeriodicBC())"))
    end
    art.mu_sensor in (:strain, :velocity) ||
        error("art.mu_sensor must be :strain or :velocity, got :$(art.mu_sensor)")
    art.beta_sensor in (:strain, :gated_strain, :dilatation, :ungated_dilatation) ||
        error("art.beta_sensor must be :strain, :gated_strain, :dilatation or " *
              ":ungated_dilatation, got :$(art.beta_sensor)")
    art.reduction in (:sum, :max) ||
        error("art.reduction must be :sum or :max, got :$(art.reduction)")
    art.smoother in (:compact, :gaussian) ||
        error("art.smoother must be :compact or :gaussian, got :$(art.smoother)")
    art.detector in (:delta4, :d8) ||
        error("art.detector must be :delta4 or :d8, got :$(art.detector)")
    # Per-condition restrictions on geometry and EOS agreement (boundary.jl).
    for d in 1:3, side in 1:2
        validate_bc(bcs[d][side], metric, eos, d, side)
    end
    # ---- Coordinate-singularity folds -----------------------------------
    axis   = bcs[1][1] isa AxisBC
    orig1  = bcs[1][1] isa OriginBC
    pole_l = bcs[2][1] isa PoleBC
    pole_h = bcs[2][2] isa PoleBC
    pole_l == pole_h || error("PoleBC must be applied at both ends of θ")
    poles = pole_l
    # A domain endpoint supplied in the solver's storage type can only
    # represent π to that type's precision. Configuration checks should reject
    # a different angular range, not the Float32 representation of
    # the requested one.
    angle_tol = max(1e-10, 8 * Float64(eps(T)) * π)
    if axis
        metric isa CylindricalMetric || error("AxisBC requires CylindricalMetric")
        stretch[1] === nothing ||
            error("folded dimensions cannot be stretched")
        abs(Float64(origin[1])) < 1e-14 || error("AxisBC requires origin[1] = 0")
        n_global[2] == 1 || iseven(n_global[2]) ||
            error("resolved-θ AxisBC requires an even θ point count over 2π")
    end
    if orig1
        metric isa SphericalMetric || error("OriginBC requires SphericalMetric")
        stretch[1] === nothing || error("folded dimensions cannot be stretched")
        abs(Float64(origin[1])) < 1e-14 || error("OriginBC requires origin[1] = 0")
        n_global[3] == 1 || iseven(n_global[3]) ||
            error("resolved-φ OriginBC requires an even φ point count over 2π")
        θsum = 2 * Float64(origin[2]) + Float64(L_domain[2])
        n_global[2] == 1 || isapprox(θsum, π; atol=angle_tol) ||
            error("OriginBC requires a θ range symmetric about π/2")
    end
    if poles
        metric isa SphericalMetric || error("PoleBC requires SphericalMetric")
        stretch[2] === nothing || error("folded dimensions cannot be stretched")
        abs(Float64(origin[2])) < 1e-14 &&
            isapprox(Float64(L_domain[2]), π; atol=angle_tol) ||
            error("PoleBC requires the θ domain (0, π)")
        n_global[3] == 1 || iseven(n_global[3]) ||
            error("resolved-φ PoleBC requires an even φ point count over 2π")
    end
    periodic = ntuple(d -> n_global[d] > 1 ? isperiodic(bcs[d][1]) : true, 3)
    (axis || orig1) && (periodic = (false, periodic[2], periodic[3]))
    poles && (periodic = (periodic[1], false, periodic[3]))
    for d in 1:3
        stretch[d] === nothing || !periodic[d] ||
            error("dimension $d: stretched dimensions must be non-periodic")
    end
    Lt = ntuple(d -> T(L_domain[d]), 3)
    active_g = ntuple(d -> n_global[d] > 1, 3)
    # Grid spacing: computational ξ ∈ [0,1] for stretched dims; half-offset
    # r ∈ (0, R] for axis grids (h = R/(N − ½), r₁ = h/2); standard otherwise.
    # Half-offset grids on folded dimensions: fold at the low end only
    # (r: axis/origin) gives h = L/(N − ½); folds at both ends (θ poles)
    # give h = L/N; either way the first node sits at h/2.
    fold_lo_dim = ntuple(d -> (d == 1 && (axis || orig1)) || (d == 2 && poles), 3)
    fold_hi_dim = ntuple(d -> d == 2 && poles, 3)
    h = ntuple(3) do d
        active_g[d] || return one(T)
        stretch[d] === nothing || return one(T) / (n_global[d] - 1)
        fold_lo_dim[d] && fold_hi_dim[d] && return Lt[d] / n_global[d]
        fold_lo_dim[d] && return Lt[d] / (n_global[d] - T(0.5))
        periodic[d] ? Lt[d] / n_global[d] : Lt[d] / (n_global[d] - 1)
    end
    coord_shift = ntuple(d -> fold_lo_dim[d] ? h[d] / 2 : zero(T), 3)
    # --- Patch layout ----------------------------------------------------
    npatch = prod(patch_grid)
    regions = patch_slabs(n_global, periodic, patch_grid)
    if npatch > 1
        (axis || orig1 || poles) &&
            error("patch decomposition across a coordinate fold is not " *
                  "supported; a folded run takes a single patch")
        (deriv isa CompactScheme && filt isa CompactScheme) ||
            error("patch interfaces carry closure variants for tridiagonal " *
                  "schemes only; banded schemes (C10) take a single patch")
        art.detector === :delta4 ||
            error("patch interfaces support the :delta4 detector only; the " *
                  ":d8 detector's banded scheme takes a single patch")
        dims === nothing ||
            error("an explicit process grid cannot combine with patch_grid; " *
                  "each patch derives its own")
        interface_rhs in (:extended, :onesided) ||
            error("interface_rhs must be :extended or :onesided, " *
                  "got :$interface_rhs")
        ds = findfirst(>(1), patch_grid)
        stretch[ds] === nothing ||
            error("the patched dimension cannot be stretched")
        for r in regions
            r.extent[ds] >= max(9, n_halo + 2) ||
                error("patch extent $(r.extent[ds]) along dim $ds is below " *
                      "the scheme minimum; use fewer patches or more points")
        end
    end
    # --- Static refinement (levels.jl) -----------------------------------
    # `refines[ℓ]` is level ℓ's region in level ℓ−1's node space.
    refines = refine === nothing ? BlockRegion[] :
              refine isa BlockRegion ? [refine] : refine
    if isempty(refines)
        subcycle &&
            error("subcycle requires a refined region (the refine keyword)")
        regrid_interval == 0 ||
            error("regrid_interval requires a refined region (the refine " *
                  "keyword supplies the initial one)")
    end
    regrid_interval >= 0 || error("regrid_interval must be non-negative")
    tag_buffer >= 0 || error("tag_buffer must be non-negative")
    tag_threshold > 0 || error("tag_threshold must be positive")
    # A lattice cell of `tile` parent nodes is a patch of tile + 1 nodes,
    # 3·tile + 1 fine nodes; the four-node minimum gives tile ≥ 3.
    tile == 0 || tile >= 3 ||
        error("tile must be 0 (one patch per level) or at least 3 parent nodes")
    tile == 0 || !(backend isa DeviceBackend) ||
        error("tiled levels take the host backend; the same-level " *
              "interface records are host loops")
    regrid_interval == 0 || length(refines) <= 1 ||
        error("regridding is implemented for a two-level hierarchy; " *
              "$(length(refines)) refined levels were given")
    if !isempty(refines)
        MPI.Initialized() || MPI.Init(threadlevel=:funneled)
        MPI.Comm_size(comm) == 1 || level_restriction === :inject ||
            error("level_restriction = :filter restricts through a " *
                  "whole-patch line solve and is serial-only; use :inject " *
                  "under MPI")
        npatch == 1 ||
            error("refine cannot combine with a same-level patch_grid yet")
        metric isa CartesianMetric ||
            error("refinement requires CartesianMetric in this stage")
        all(isnothing, stretch) ||
            error("refinement requires an unstretched grid")
        (axis || orig1 || poles) &&
            error("refinement across a coordinate fold is forbidden " *
                  "(constraint 4 of reference/AMR_GPU.md)")
        (deriv isa CompactScheme && filt isa CompactScheme) ||
            error("the coarse-fine boundary carries closure variants for " *
                  "tridiagonal schemes only")
        art.detector === :delta4 ||
            error("refinement supports the :delta4 detector only")
        level_restriction in (:inject, :filter) ||
            error("level_restriction must be :inject or :filter, " *
                  "got :$level_restriction")
        # The level shell is read from the fine box at an offset of
        # 3·LEVEL_BUFFER nodes; a halo wider than that would index the box's
        # own zero halo silently (`_write_fine_shell!`, `_impose_shell!`).
        n_halo <= 3 * LEVEL_BUFFER ||
            error("refinement supports n_halo ≤ $(3 * LEVEL_BUFFER), got $n_halo")
        # Each region is nested by the margin inside the patches of the level
        # above it, in that level's node space (the root's is the grid).
        margin = max(n_halo, LEVEL_BUFFER)
        parent_regions = [BlockRegion((0, 0, 0), n_global)]
        for (ℓ, rg) in enumerate(refines)
            for d in 1:3
                if active_g[d]
                    rg.extent[d] >= 4 ||
                        error("level $ℓ region needs at least 4 parent nodes " *
                              "along dimension $d (9 fine points for the C8 " *
                              "filter)")
                else
                    rg.offset[d] == 0 && rg.extent[d] == 1 ||
                        error("level $ℓ region must span collapsed dimension " *
                              "$d with offset 0 and extent 1")
                end
            end
            _covered_by(_buffered(rg, active_g, margin), parent_regions) ||
                error("level $ℓ region must be nested at least $margin " *
                      "level-$(ℓ - 1) nodes inside the level-$(ℓ - 1) patches")
            parent_regions = [BlockRegion(
                ntuple(d -> active_g[d] ? 3 * rg.offset[d] : 0, 3),
                fine_extent(rg, active_g))]
        end
    end
    # --- Device residency -------------------------------------------------
    # A DeviceBackend supports a decomposed patch, refined or not: halos,
    # fold pairs, and the level transfer's gathers and writes stage through
    # the backend. The same-level interface records are still host loops, so
    # a patch_grid on device stays barred, as does the `:filter` restriction
    # (a whole-patch host line solve).
    if backend isa DeviceBackend
        npatch == 1 ||
            error("a DeviceBackend takes a single patch today; the " *
                  "interface-record staging has not been converted")
        refine === nothing || level_restriction === :inject ||
            error("level_restriction = :filter is host-only; use :inject " *
                  "on a DeviceBackend")
        eos isa Nasa9Mixture &&
            error("Nasa9Mixture has no device coefficient mirror yet; use " *
                  "IdealMixture or StiffenedGas on a DeviceBackend")
    end
    ds_split = npatch > 1 ? findfirst(>(1), patch_grid) : 0
    faces_all = [ntuple(3) do d
        d != ds_split && return (0, 0)
        P = patch_grid[d]
        lo = pid > 1 ? pid - 1 : (periodic[d] ? P : 0)
        hi = pid < P ? pid + 1 : (periodic[d] ? 1 : 0)
        (lo, hi)
    end for pid in 1:npatch]
    # The sensor smoother stands in for Cook's Gaussian test filter. `:compact`
    # reuses `filt`, which was the smoother before the option existed and keeps
    # every plan identical; `:gaussian` is the explicit nine-point stencil the
    # reference implementation uses, and carries no line solve.
    smoo = art.smoother === :gaussian ? gaussian_filter(T) : filt
    # The sensor detector. `:delta4` is the explicit undivided fourth
    # difference applied inside `delta4_sum!`, which needs no plan at all;
    # `:d8` is the pentadiagonal compact eighth derivative and needs one per
    # dimension and one pair per fold, matching the smoother.
    ring = compact_d8(T)
    equations = equations === nothing ? NavierStokes1T(eos) : equations
    equations isa EquationSet || error("equations must be an EquationSet")
    equations.n_species == nspecies(eos) ||
        error("equation set carries $(equations.n_species) species; " *
              "EOS has $(nspecies(eos))")
    n_species = equations.n_species
    n_cons = equations.n_cons
    if npatch > 1
        return _build_patched_solver(T, n_global, periodic, regions, faces_all,
                                     patch_grid, bcs, eos, equations, transport,
                                     art, metric, stretch, sources, origin, Lt,
                                     coord_shift, h, deriv, filt, smoo, cfl,
                                     filter_interval, filter_cfl, control,
                                     n_halo, comm, backend, interface_rhs, n_cons,
                                     n_species)
    end
    decomp = Decomp{T}(n_global, periodic; dims=dims, n_halo=n_halo, comm=comm)
    mkd(sch, d; kw...) =
        backend_plan(backend, plan_direction(decomp, sch, d, h[d]; kw...))
    f() = field(backend, decomp)

    # Fold specs. sigvel/sigflux derivations live in folds.jl and the README.
    function pairspec(pdim, revdim)
        # Degenerate mappings on collapsed dims are identities.
        shift_needed = pdim != 0 && decomp.active[pdim]
        rev_needed = revdim != 0 && decomp.active[revdim]
        (!shift_needed && !rev_needed) && return nothing   # self-paired
        shift_local = !shift_needed || decomp.dims[pdim] == 1
        rev_local = !rev_needed || decomp.dims[revdim] == 1
        if shift_needed
            shift_local || (iseven(decomp.dims[pdim]) &&
                            n_global[pdim] % decomp.dims[pdim] == 0) ||
                error("pairing dim $pdim must be on one rank or split into an " *
                      "even number of uniform blocks")
            shift_local && (iseven(decomp.n_local[pdim]) ||
                error("pairing dim $pdim local extent must be even"))
        end
        if rev_needed && !rev_local
            n_global[revdim] % decomp.dims[revdim] == 0 ||
                error("reversed dim $revdim must split into uniform blocks")
            iseven(decomp.dims[revdim]) ||
                error("reversed dim $revdim needs an even rank count")
        end
        crd = collect(Int, decomp.coords)
        shift_needed && !shift_local &&
            (crd[pdim] = mod(crd[pdim] + decomp.dims[pdim] ÷ 2, decomp.dims[pdim]))
        rev_needed && !rev_local &&
            (crd[revdim] = decomp.dims[revdim] - 1 - crd[revdim])
        partner = Int(MPI.Cart_rank(decomp.comm, Cint.(crd)))
        loc = shift_local && rev_local
        keep_e = if shift_needed && !shift_local
            decomp.coords[pdim] < decomp.dims[pdim] ÷ 2
        elseif rev_needed && !rev_local
            decomp.coords[revdim] < decomp.dims[revdim] ÷ 2
        else
            true
        end
        PairSpec(shift_needed ? pdim : 0, rev_needed ? revdim : 0,
                 shift_local, rev_local, loc, partner, keep_e,
                 loc ? empty_field(backend, T) : field(backend, decomp))
    end
    function foldspec(d, lo, hi, pdim, revdim, sigvel, sigflux)
        dp = (mkd(deriv, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(deriv, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        fp = (mkd(filt, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(filt, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        # `:compact` aliases the filter plans to avoid duplicating them.
        # This matches `smooth!` before it had an operator of its
        # own, so the default path keeps both its answer and its footprint.
        sp = art.smoother === :compact ? fp :
             (mkd(smoo, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(smoo, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        rp = art.detector === :delta4 ? (nothing, nothing) :
             (mkd(ring, d; lo_fold=(lo ? 1 : nothing), hi_fold=(hi ? 1 : nothing)),
              mkd(ring, d; lo_fold=(lo ? -1 : nothing), hi_fold=(hi ? -1 : nothing)))
        FoldSpec(d, lo, hi, pairspec(pdim, revdim), sigvel, sigflux, dp, fp, sp, rp)
    end
    folds = (nothing, nothing, nothing)
    if axis
        sv = (-1, -1, 1)
        folds = (foldspec(1, true, false, 2, 0, sv,
                          flux_parities(equations, sv, 1, -1)),
                 folds[2], folds[3])           # A₁ = r is odd
    elseif orig1
        sv = (-1, 1, -1)
        folds = (foldspec(1, true, false, 3, 2, sv,
                          flux_parities(equations, sv, 1, 1)),
                 folds[2], folds[3])           # A₁ = r² sinθ is even
    end
    if poles
        sv = (1, -1, -1)
        # Fold along θ: flux parities use σ(u_θ); A₂ = r sinθ is odd in θ.
        sf2 = flux_parities(equations, sv, 2, -1)
        folds = (folds[1], foldspec(2, true, true, 3, 0, sv, sf2), folds[3])
    end
    deriv_plans = ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                         mkd(deriv, d) : nothing, 3)
    filter_plans = ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                         mkd(filt, d) : nothing, 3)
    smooth_plans = art.smoother === :compact ? filter_plans :
                   ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                          mkd(smoo, d) : nothing, 3)
    # `nothing`, not a tuple of nothings: `detect_sum!` dispatches on this
    # field's type to decide which detector runs, so under `:delta4` the whole
    # d8 path (`ring_sum!`, `ring_along!`, and the `apply_along!` call taking a
    # possibly-absent plan) is not reachable from inference and costs the
    # default configuration nothing.
    # A tuple would not serve: a fully folded run carries no plans here even
    # under `:d8`, since those live on the FoldSpec.
    ring_plans = art.detector === :delta4 ? nothing :
                 ntuple(d -> decomp.active[d] && folds[d] === nothing ?
                        mkd(ring, d) : nothing, 3)
    orig = ntuple(d -> stretch[d] === nothing ? T(origin[d]) : zero(T), 3)
    bcs_t = ntuple(d -> (bcs[d][1], bcs[d][2]), 3)
    patch = Patch(1, 0, regions[1], comm, decomp, h,
                  ntuple(d -> (0, 0), 3), bcs_t, folds,
                  deriv_plans, deriv_plans, filter_plans, smooth_plans, ring_plans,
                  any(fold -> fold !== nothing, folds) ? f() : empty_field(backend, T),
                  any(fold -> fold !== nothing, folds) ? f() : empty_field(backend, T),
                  f(), f(), f(), f(), f(), f(), f(), f(),
                  [f() for _ in 1:n_species],
                  [f() for _ in 1:3, _ in 1:3],
                  (f(), f(), f()),
                  [f() for _ in 1:3, _ in 1:n_species],
                  f(), f(), f(),
                  [f() for _ in 1:n_species],
                  f(), f(), f(), f(), f(),
                  art.detector === :delta4 ? empty_field(backend, T) : f(),
                  f(), (f(), f(), f()), (f(), f(), f()), f(), f(), f(),
                  [f() for _ in 1:3, _ in 1:n_cons])
    if isempty(refines)
        patches = [patch]
        solver = Solver{T,typeof(equations),typeof(eos),typeof(metric),
                        typeof(stretch),typeof(sources),typeof(patch)}(
                      equations, eos, transport, art, metric, stretch, sources,
                      Lt, orig, coord_shift, h,
                      T(cfl), filter_interval, T(filter_cfl), control,
                      n_global, patches, regions, decomp.comm,
                      GhostRecord{T}[], GhostRecord{T}[], PlaneRecord{T}[],
                      [Level{T}(0, [1], LevelTransfer{T}[])], false, nothing,
                      zero(T), zero(T), 0, zero(T), zero(T), 0.0, 0.0, FloorTally())
        init_geometry!(solver)
        return solver
    end
    # --- Refined patches and their couplings (levels.jl) ------------------
    # Level ℓ covers `refines[ℓ]` of level ℓ − 1: one patch over the region
    # exactly with `tile = 0`, the lattice tiles meeting it otherwise.
    fines = Patch{T}[]
    levels = [Level{T}(0, [1], LevelTransfer{T}[])]
    parent_indices = [1]
    parent_regions = [BlockRegion((0, 0, 0), n_global)]
    parent_h = h
    decomp_of = Dict{Int,Decomp{T}}(1 => decomp)
    margin = max(n_halo, LEVEL_BUFFER)
    for (ℓ, rg) in enumerate(refines)
        if tile == 0
            tregions = [rg]
        else
            # Clip the lattice to the parent patches' bounding box less the
            # margin; a tile that then still leaves the union is refused.
            lo = ntuple(d -> minimum(r.offset[d] for r in parent_regions) +
                             1 + margin, 3)
            hi = ntuple(d -> maximum(r.offset[d] + r.extent[d]
                                     for r in parent_regions) - margin, 3)
            tregions = _level_tiles(rg, active_g, tile, lo, hi)
            isempty(tregions) &&
                error("level $ℓ region admits no tile of edge $tile inside " *
                      "the nesting margin")
        end
        faces = _tile_faces(tregions)
        indices = Int[]
        transfers = LevelTransfer{T}[]
        for (ti, tr) in enumerate(tregions)
            _covered_by(_buffered(tr, active_g, margin), parent_regions) ||
                error("level $ℓ tile $tr must be nested at least $margin " *
                      "level-$(ℓ - 1) nodes inside the level-$(ℓ - 1) patches")
            idx = length(fines) + 2
            fine = _build_fine_patch(T, tr, active_g, parent_h, n_halo, comm,
                                     deriv, filt, smoo, art.smoother,
                                     interface_rhs, backend, n_species, n_cons,
                                     idx, ℓ, faces[ti])
            pidx = _parents_of(tr, active_g, parent_indices, parent_regions)
            lt = build_level_transfer(T, tr, active_g, n_halo, pidx, idx,
                                      level_restriction, n_cons, subcycle,
                                      [decomp_of[i] for i in pidx], fine.decomp,
                                      faces[ti])
            push!(fines, fine)
            push!(indices, idx)
            push!(transfers, lt)
            decomp_of[idx] = fine.decomp
        end
        lvl = fines[end-length(tregions)+1:end]
        # Records address ranks of the communicator the exchange runs over,
        # the root's Cartesian one (`solver.comm`), whose numbering may be
        # reordered relative to `comm`.
        sends, recvs, planes = _level_records(T, decomp.comm, [p.region for p in lvl],
                                              indices, [p.decomp for p in lvl],
                                              n_cons)
        push!(levels, Level{T}(ℓ, indices, transfers, sends, recvs, planes))
        parent_indices = indices
        parent_regions = [p.region for p in lvl]
        parent_h = lvl[1].h
    end
    # The literal form types the vector by the patches' join (the root's
    # boundary-condition tuple differs from a refined patch's), as the
    # two-patch construction always has.
    patches = [patch, fines...]
    regrid = regrid_interval == 0 ? nothing :
             RegridSpec{T}(regrid_interval, T(tag_threshold), tag_buffer,
                           margin, n_halo, interface_rhs,
                           deriv, filt, smoo, backend, tile, 0)
    solver = Solver{T,typeof(equations),typeof(eos),typeof(metric),
                    typeof(stretch),typeof(sources),eltype(patches)}(
                  equations, eos, transport, art, metric, stretch, sources,
                  Lt, orig, coord_shift, h,
                  T(cfl), filter_interval, T(filter_cfl), control,
                  n_global, patches, regions, decomp.comm,
                  GhostRecord{T}[], GhostRecord{T}[], PlaneRecord{T}[],
                  levels, subcycle, regrid,
                  zero(T), zero(T), 0, zero(T), zero(T), 0.0, 0.0, FloorTally())
    for p in getfield(solver, :patches)
        init_geometry!(PatchSolver(solver, p))
    end
    return solver
end

# Refined patch construction over the region `refine` (in the parent level's
# node space; `h` is the parent level's spacing), shared between the `Solver`
# constructor and `regrid!`. It takes the schemes, not plans off an existing
# patch, because a regrid changes the extents and every plan must be
# rebuilt.
function _build_fine_patch(::Type{T}, refine::BlockRegion,
                           active_g::NTuple{3,Bool}, h::NTuple{3,T},
                           n_halo::Int, comm::MPI.Comm, deriv, filt, smoo,
                           smoother::Symbol,
                           interface_rhs::Symbol, backend::AbstractBackend,
                           n_species::Int, n_cons::Int, id::Int, level::Int,
                           faces::NTuple{3,NTuple{2,Int}}=ntuple(d -> (0, 0), 3)
                           ) where {T}
    hf = ntuple(d -> active_g[d] ? h[d] / 3 : h[d], 3)
    # Parent-level node g is refined-level node 3(g − 1) + 1.
    region_f = BlockRegion(ntuple(d -> active_g[d] ? 3 * refine.offset[d] : 0, 3),
                           fine_extent(refine, active_g))
    pper_f = ntuple(d -> !active_g[d], 3)
    np_f = (MPI.Initialized() || MPI.Init(threadlevel=:funneled);
            MPI.Comm_size(comm))
    decomp_f = Decomp{T}(region_f.extent, pper_f;
                         dims=_amr_dims(region_f.extent,
                                        ntuple(d -> region_f.extent[d] > 1, 3),
                                        np_f),
                         n_halo=n_halo, comm=comm)
    # Every face of a refined patch closes with the interface rows and reads
    # ghosts; the boundary condition only records where they come from.
    bcs_f = _fine_bcs(active_g, faces)
    mkf(sch, d; kw...) =
        backend_plan(backend, plan_direction(decomp_f, sch, d, hf[d]; kw...))
    ext_f = interface_rhs === :extended
    icd = ext_f ? interface_closures(deriv) : nothing
    icf = ext_f ? interface_closures(filt) : nothing
    dplans_f = ntuple(d -> decomp_f.active[d] ?
        mkf(deriv, d; lo_closures=icd, hi_closures=icd) : nothing, 3)
    vplans_f = ntuple(d -> !decomp_f.active[d] ? nothing :
        (ext_f ? mkf(deriv, d) : dplans_f[d]), 3)
    fplans_f = ntuple(d -> decomp_f.active[d] ?
        mkf(filt, d; lo_closures=icf, hi_closures=icf) : nothing, 3)
    # The sensor smoother's input is built per patch and its coarse-fine
    # ghosts are never filled, so its plans keep the standard closures even
    # under `smoother = :compact`, as the same-level patch path does below.
    # Aliasing `fplans_f` here would read four ghost layers of allocation
    # zeros at every coarse-fine face through the C8 interior rows the
    # interface closures leave in place.
    splans_f = ntuple(d -> decomp_f.active[d] ? mkf(smoo, d) : nothing, 3)
    g() = field(backend, decomp_f)
    empty3 = empty_field(backend, T)
    return Patch(id, level, region_f, comm, decomp_f, hf,
                 faces, bcs_f, (nothing, nothing, nothing),
                 dplans_f, vplans_f, fplans_f, splans_f, nothing,
                 empty3, empty3,
                 g(), g(), g(), g(), g(), g(), g(), g(),
                 [g() for _ in 1:n_species],
                 [g() for _ in 1:3, _ in 1:3],
                 (g(), g(), g()),
                 [g() for _ in 1:3, _ in 1:n_species],
                 g(), g(), g(),
                 [g() for _ in 1:n_species],
                 g(), g(), g(), g(), g(),
                 empty3,
                 g(), (g(), g(), g()), (g(), g(), g()), g(), g(), g(),
                 [g() for _ in 1:3, _ in 1:n_cons])
end

_fine_bcs(active_g::NTuple{3,Bool}, faces::NTuple{3,NTuple{2,Int}}) =
    ntuple(d -> !active_g[d] ? (PeriodicBC(), PeriodicBC()) :
                (faces[d][1] == 0 ? CoarseFineBC() : InterfaceBC(faces[d][1]),
                 faces[d][2] == 0 ? CoarseFineBC() : InterfaceBC(faces[d][2])), 3)

# A patch with new id, faces and boundary conditions sharing every array and
# plan of `p`: what a regrid hands a surviving tile whose neighbors changed.
function _repatch(p::Patch, id::Int, faces::NTuple{3,NTuple{2,Int}}, bcs)
    names = fieldnames(typeof(p))[1:end-1]     # field_tuples is derived
    args = map(names) do f
        f === :id ? id : f === :faces ? faces : f === :bcs ? bcs : getfield(p, f)
    end
    return Patch(args...)
end

# Multi-patch construction: the rank set is partitioned over the patch slabs,
# each patch builds its own decomposition, plans and arrays over its own
# communicator, and the interface exchange records are derived from one
# world Allgather. Folds, banded schemes, the :d8 detector, and an explicit
# process grid are rejected by the caller before this runs.
function _build_patched_solver(::Type{T}, n_global, periodic, regions, faces_all,
                               patch_grid, bcs, eos, equations, transport, art,
                               metric, stretch, sources, origin, Lt, coord_shift,
                               h, deriv, filt, smoo, cfl, filter_interval,
                               filter_cfl, control, n_halo, comm, backend,
                               interface_rhs, n_cons, n_species) where {T}
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    world = comm
    np = MPI.Comm_size(world)
    npatch = length(regions)
    if np == 1
        my_pids = collect(1:npatch)
        pcomm = world
    else
        counts = patch_rank_counts(regions, np)
        myrank = MPI.Comm_rank(world)
        color = searchsortedfirst(cumsum(counts), myrank + 1)
        pcomm = MPI.Comm_split(world, color, myrank)
        my_pids = [color]
    end
    ext = interface_rhs === :extended
    icd = ext ? interface_closures(deriv) : nothing
    icf = ext ? interface_closures(filt) : nothing
    nofold = (nothing, nothing, nothing)
    patches = map(my_pids) do pid
        region = regions[pid]
        faces = faces_all[pid]
        pper = ntuple(d -> patch_grid[d] > 1 ? false : periodic[d], 3)
        dcp = Decomp{T}(region.extent, pper; n_halo=n_halo, comm=pcomm)
        pbcs = ntuple(d -> (faces[d][1] == 0 ? bcs[d][1] : InterfaceBC(faces[d][1]),
                            faces[d][2] == 0 ? bcs[d][2] : InterfaceBC(faces[d][2])), 3)
        mk(sch, d; kw...) = plan_direction(dcp, sch, d, h[d]; kw...)
        locl(rows, d) = faces[d][1] == 0 ? nothing : rows
        hicl(rows, d) = faces[d][2] == 0 ? nothing : rows
        dplans = ntuple(d -> dcp.active[d] ?
            mk(deriv, d; lo_closures=locl(icd, d), hi_closures=hicl(icd, d)) :
            nothing, 3)
        # The flux divergence keeps the scheme's own one-sided closures at an
        # interface (ghost fluxes are unavailable; see patches.jl), so it takes
        # separate plans exactly where the gradient plans read ghosts.
        vplans = ntuple(d -> !dcp.active[d] ? nothing :
            (ext && (faces[d][1] != 0 || faces[d][2] != 0) ? mk(deriv, d) :
             dplans[d]), 3)
        fplans = ntuple(d -> dcp.active[d] ?
            mk(filt, d; lo_closures=locl(icf, d), hi_closures=hicl(icf, d)) :
            nothing, 3)
        # The sensor smoother's input is built per patch, so its interface
        # ghosts carry no data and its plans keep the standard closures even
        # under `smoother = :compact`.
        splans = ntuple(d -> dcp.active[d] ? mk(smoo, d) : nothing, 3)
        g() = field(backend, dcp)
        empty3 = zeros(T, 0, 0, 0)
        Patch(pid, 0, region, pcomm, dcp, h, faces, pbcs, nofold,
              dplans, vplans, fplans, splans, nothing,
              empty3, empty3,
              g(), g(), g(), g(), g(), g(), g(), g(),
              [g() for _ in 1:n_species],
              [g() for _ in 1:3, _ in 1:3],
              (g(), g(), g()),
              [g() for _ in 1:3, _ in 1:n_species],
              g(), g(), g(),
              [g() for _ in 1:n_species],
              g(), g(), g(), g(), g(),
              empty3,
              g(), (g(), g(), g()), (g(), g(), g()), g(), g(), g(),
              [g() for _ in 1:3, _ in 1:n_cons])
    end
    ghost_sends, ghost_recvs, plane_pairs = build_interface_records(
        T, world, regions, faces_all, my_pids, [p.decomp for p in patches], n_cons)
    orig = ntuple(d -> stretch[d] === nothing ? T(origin[d]) : zero(T), 3)
    solver = Solver{T,typeof(equations),typeof(eos),typeof(metric),
                    typeof(stretch),typeof(sources),eltype(patches)}(
                  equations, eos, transport, art, metric, stretch, sources,
                  Lt, orig, coord_shift, h,
                  T(cfl), filter_interval, T(filter_cfl), control,
                  n_global, patches, regions, world,
                  ghost_sends, ghost_recvs, plane_pairs,
                  [Level{T}(0, collect(eachindex(patches)), LevelTransfer{T}[])],
                  false, nothing,
                  zero(T), zero(T), 0, zero(T), zero(T), 0.0, 0.0, FloorTally())
    for p in getfield(solver, :patches)
        init_geometry!(PatchSolver(solver, p))
    end
    return solver
end

"""
    xcoord(solver, d, i)

Physical coordinate of rank-local, one-based interior index `i` in direction
`d`. The index does not include halo padding. This is equivalent to
`global_xcoord(solver, d, solver.region.offset[d] + solver.decomp.offset[d] + i)`,
the patch region offset placing a patch's block in the whole grid (zero for a
single-patch solver).
"""
xcoord(solver::SolverLike, d::Int, i::Int) =
    global_xcoord(solver, d,
                  solver.region.offset[d] + solver.decomp.offset[d] + i)

"""
    global_xcoord(solver, d, g)

Physical coordinate of global, one-based index `g` in direction `d`, including
any half-cell fold offset and [`Stretch`](@ref) mapping. `g` indexes the whole
grid, not a patch. Every rank returns the same value for the same `(d, g)`.
Use this form when assembling a global coordinate vector and [`xcoord`](@ref)
for a local interior index.
"""
function global_xcoord(solver::SolverLike, d::Int, g::Int)
    ξ = solver.origin[d] + solver.coord_shift[d] + (g - 1) * solver.h[d]
    st = solver.stretch[d]
    return st === nothing ? ξ : st.x(ξ)
end

"""
    gidx(solver, i, j, k) -> CartesianIndex

Convert rank-local, one-based interior indices to a `CartesianIndex` for the
halo-padded state and solver fields. Collapsed directions have zero padding.
The result is rank-local; it does not locate a global point owned by another
rank.
"""
function gidx(solver::SolverLike, i::Int, j::Int, k::Int)
    pad = solver.decomp.n_halo_d
    return CartesianIndex(i + pad[1], j + pad[2], k + pad[3])
end

"""
    interior_index(solver, I) -> (i, j, k)

Rank-local, one-based interior indices of the padded `CartesianIndex` `I`; the
inverse of [`gidx`](@ref). Use it wherever a padded index has to be handed to
something that takes interior ones. [`boundary_plane`](@ref) yields padded
indices, while [`xcoord`](@ref) expects an interior index:

```julia
for I in boundary_plane(solver, 1, 1)
    i, j, k = interior_index(solver, I)
    x1 = xcoord(solver, 1, i)
end
```

Indices of halo cells come back outside `1:n_local[d]`, and are meaningful only
as an offset from this rank's block.
"""
@inline function interior_index(solver::SolverLike, I::CartesianIndex{3})
    pad = solver.decomp.n_halo_d
    return (I[1] - pad[1], I[2] - pad[2], I[3] - pad[3])
end

# --- Reading the in-flight conserved state -----------------------------------
#
# `Q` is a 4-D array in the layout defined by the equation set, so a caller
# wanting the density writes `Q[I, 1] + Q[I, 2]` and has silently hardcoded a
# two-species run. The functions below are the layout-independent equivalents,
# for callback conditions, custom diagnostics, and any other code reading `Q`
# between steps, when the primitives are stale; see
# `refresh_primitives!`.
#
# All of them index the PADDED arrays, following the convention of `gidx` and
# `boundary_plane`, and all are rank-local: they report nothing about points
# this rank does not hold. A predicate built from them is reduced by
# `WhenState`, or must be reduced by the caller.

"""
    mixture_density(solver, Q, I) -> ρ

Mixture density at padded index `I`, the sum of the partial densities. Prefer
this method to explicit component indices; the expression then remains valid
when the species count changes.
"""
@inline function mixture_density(solver::SolverLike, Q, I)
    ρ = zero(eltype(Q))
    @inbounds for sp in 1:solver.equations.n_species
        ρ += Q[I, sp]
    end
    return ρ
end

"""
    velocity(solver, Q, I) -> (u, v, w)

Physical, coordinate-aligned velocity components at padded index `I`, recovered
from the momenta and the mixture density.
"""
@inline function velocity(solver::SolverLike, Q, I)
    ri = one(eltype(Q)) / mixture_density(solver, Q, I)
    m1, m2, m3 = solver.equations.i_mom
    @inbounds return (Q[I, m1] * ri, Q[I, m2] * ri, Q[I, m3] * ri)
end

"Total energy per unit volume at padded index `I`."
@inline total_energy(solver::SolverLike, Q, I) =
    @inbounds Q[I, solver.equations.i_energy]

"Mass fraction of species `sp` at padded index `I`."
@inline mass_fraction(solver::SolverLike, Q, I, sp::Int) =
    @inbounds Q[I, sp] / mixture_density(solver, Q, I)

"""
    boundary_plane(solver, d, side) -> CartesianIndices or nothing

Padded indices of this rank's plane on the global `side` (1 low, 2 high) of
dimension `d`, or `nothing` when this rank does not own that edge. Ownership
along `d` is the only test applied, so a periodic or collapsed dimension yields
a plane as well. The plane spans the full local extent of the other two
dimensions and is one point thick in `d`. For example, a boundary diagnostic may
be written as

```julia
plane = boundary_plane(solver, 1, 1)
plane === nothing && return -Inf
return maximum(I -> mixture_density(solver, Q, I), plane)
```

The result is rank-local. [`WhenState`](@ref) reduces a predicate constructed
from it; other uses require an explicit reduction. An unreduced value is valid
globally only in a serial calculation.

These are padded indices. Convert one with [`interior_index`](@ref) before
passing it to [`xcoord`](@ref) or anything else expecting an interior index.
"""
boundary_plane(solver::SolverLike, d::Int, side::Int) = wallplane(solver.decomp, d, side)

"""
    allocate_state(solver)

Zero-filled conserved storage for `solver`: a [`ConservedState`](@ref) matching
its sole patch, or, for a patch-decomposed solver, a `Vector` of them aligned
with `solver.patches`. The vector form is selected by the *global* patch
count: a rank of a partitioned multi-patch run holds exactly one local patch,
but its state must still be the vector the multi-patch drivers dispatch on,
or the run would silently skip the interface synchronization.
"""
allocate_state(solver::Solver) =
    (length(getfield(solver, :patch_regions)) == 1 && npatches(solver) == 1) ?
        _state_like(solver.rho, solver.equations.n_cons) :
        [_state_like(p.rho, solver.equations.n_cons)
         for p in getfield(solver, :patches)]
allocate_state(ps::PatchSolver) = _state_like(ps.rho, ps.equations.n_cons)

# Zero-filled conserved storage matching one patch's field storage, so a
# device-resident patch gets device state without the solver carrying its
# backend object around.
_state_like(rho::AbstractArray{T,3}, n_cons::Int) where {T} =
    ConservedState(fill!(similar(rho, size(rho)..., n_cons), zero(T)))

# --- Operator routing through folds ----------------------------------------

# A plans tuple is heterogeneous whenever a dimension is collapsed or folded
# (`nothing` in that slot), so indexing it with a runtime `d` yields a union
# that the call below it must split: measured ~330 B per operator application
# and 11.9 kB per RHS on a planar (32, 16, 1) run against 336 B in 3-D.
# Branching on `d` cuts that to 160 B per application and 3.8 kB per RHS; the
# remainder is the union itself, and removing it would take a sentinel plan of
# the concrete type in the empty slot, which needs a `LineSolver` and a
# communicator to construct for a dimension that is never swept.
@inline _plan_at(plans::Tuple, d::Int) = d == 1 ? plans[1] : d == 2 ? plans[2] : plans[3]

"""
    deriv_along!(out, f, solver, d, σf)

Compact derivative of `f` along active dimension `d`; `σf` is the field's
antipodal sign for the fold on `d` (ignored when there is none). The caller
ensures current rank-boundary halos of `f`. Only the interior of `out` is
written.

This is a distributed line solve along `d`, so it is collective over that
dimension's sub-communicator and every rank must call it, including ranks
holding no part of a fold. At a self-paired fold `f` is also written:
`fold_fill!` mirrors its halos beyond the folded end before the sweep. A paired
fold leaves `f` untouched, running the even/odd butterfly through
`solver.pairbuf` instead.
"""
function deriv_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _plan_at(solver.deriv_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:deriv))
    end
    return out
end

"""
    div_along!(out, f, solver, d, σf)

Compact derivative of `f` along `d` through the divergence plans. These are
`solver.deriv_plans` except at a patch-interface end under
`interface_rhs = :extended`, where the gradient plans read exchanged ghost
data that a flux array does not carry, so the divergence keeps the scheme's
one-sided closure rows (`solver.div_plans`). A folded dimension draws on the
fold's own derivative plans instead, which is the same operator because folds
and patch interfaces never share a dimension. The flux-divergence loop and the
discrete-GCL construction `gcl_cotr!` go through here so the two apply the
identical operator. Same collective, halo, and fold contract as
[`deriv_along!`](@ref).
"""
function div_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _plan_at(solver.div_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:deriv))
    end
    return out
end

"""
    deriv_scaled_along!(out, f, solver, d, σf)

Compact derivative of `f` along active dimension `d`, scaled pointwise by
`solver.inv_h[d]` inside the scatter of the line solve. Interior results are
bit-identical to [`deriv_along!`](@ref) followed by `_scale_grad!`; only halo
cells of `out` differ (the two-pass rescale also scaled them, but they hold no
data any consumer reads without a fresh exchange). A fold dimension or a
device plan takes the two-pass route unchanged. Same collective, halo, and
fold contract as `deriv_along!`.
"""
function deriv_scaled_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    plan = _plan_at(solver.deriv_plans, d)
    if fold === nothing && !(plan isa DevicePlan)
        apply_along_scaled!(out, plan, f, solver.decomp, solver.inv_h[d])
    else
        deriv_along!(out, f, solver, d, σf)
        _scale_grad!(out, solver, d)
    end
    return out
end

"""
    div_subtract_along!(dQ, c, f, solver, d, σf, inv_J)

Compact derivative of `f` along `d` through the divergence plans, subtracted
from conserved component `c` of `dQ` inside the scatter of the line solve:
`dQ[I, c] -= inv_J[I] * (D f)[I]`, or `dQ[I, c] -= (D f)[I]` when
`inv_J === nothing` (the unit-geometry case). Interior results are
bit-identical to [`div_along!`](@ref) into scratch followed by the
subtraction pass. A fold dimension or a device plan takes that two-pass route
through `solver.tmp_a`. Same collective, halo, and fold contract as
`div_along!`.
"""
function div_subtract_along!(dQ, c::Int, f, solver::SolverLike, d::Int,
                             σf::Int, inv_J)
    fold = solver.folds[d]
    plan = _plan_at(solver.div_plans, d)
    if fold === nothing && !(plan isa DevicePlan)
        apply_along_subtract!(dQ, c, plan, f, solver.decomp, inv_J)
    else
        div_along!(solver.tmp_a, f, solver, d, σf)
        nx, ny, nz = solver.decomp.n_local
        o1, o2, o3 = solver.decomp.n_halo_d
        if inv_J === nothing
            pointwise!(_subtract_div_point!, dQ, nx, ny, nz,
                       dQ, solver.tmp_a, c, o1, o2, o3)
        else
            pointwise!(_subtract_jac_div_point!, dQ, nx, ny, nz,
                       dQ, solver.tmp_a, inv_J, c, o1, o2, o3)
        end
    end
    return dQ
end

"""Compact filter of `f` along dimension `d` with antipodal sign `σf`. Collective,
with the same halo and fold contract as `deriv_along!`."""
function filt_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _plan_at(solver.filter_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:filter))
    end
    return out
end

"""
    smooth_along!(out, f, solver, d, σf)

Sensor smoother of `f` along dimension `d` with antipodal sign `σf`. This is
the Cook test filter, selected by `ArtParams.smoother`, and is a distinct
operator from `filt_along!`: the two coincide only under
`ArtParams(smoother = :compact)`, which aliases the filter plans and avoids
planning an operator of its own; the default `:gaussian` plans the explicit
nine-point stencil of [`gaussian_filter`](@ref). Only the artificial-property
sensors go through here, by way of `smooth!`.

Collective, as `deriv_along!` is.
"""
function smooth_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _plan_at(solver.smooth_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:smooth))
    end
    return out
end

"""
    ring_along!(out, f, solver, d, σf)

Compact eighth derivative of `f` along dimension `d` with antipodal sign `σf`,
the ringing detector selected by `ArtParams(detector = :d8)`. Only `ring_sum!`
calls this, and only under that setting: `solver.ring_plans` is `nothing`
otherwise, which keeps this function off the default configuration's inference
path. Indexing that field under `:delta4` would throw. See `detect_sum!`.

Collective, with the same halo and fold contract as `deriv_along!`.
"""
function ring_along!(out, f, solver::SolverLike, d::Int, σf::Int)
    fold = solver.folds[d]
    if fold === nothing
        apply_along!(out, _plan_at(solver.ring_plans, d), f, solver.decomp)
    else
        fold_apply!(out, f, solver, fold, σf, Val(:ring))
    end
    return out
end

# Scale a raw coordinate-derivative field by 1/h_d pointwise (full array).
@inline function _scale_grad_point!(g, ih, i, j, k)
    @inbounds g[i, j, k] *= ih[i, j, k]
    return nothing
end

function _scale_grad!(g, solver, d)
    ih = solver.inv_h[d]
    n1, n2, n3 = size(g)
    pointwise!(_scale_grad_point!, g, n1, n2, n3, g, ih)
    return g
end

# The flux-divergence accumulation bodies of compute_rhs!: zero one conserved
# component's interior, subtract a divergence, subtract a Jacobian-scaled
# divergence, and form the A_d·F_d product over the full padded array.
@inline function _zero_component_point!(dQ, c, o1, o2, o3, i, j, k)
    @inbounds dQ[i+o1, j+o2, k+o3, c] = 0
    return nothing
end

@inline function _subtract_div_point!(dQ, tmp, c, o1, o2, o3, i, j, k)
    @inbounds dQ[i+o1, j+o2, k+o3, c] -= tmp[i+o1, j+o2, k+o3]
    return nothing
end

@inline function _subtract_jac_div_point!(dQ, tmp, inv_J, c, o1, o2, o3, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        dQ[I, c] -= inv_J[I] * tmp[I]
    end
    return nothing
end

@inline function _area_flux_point!(tmp_b, Ad, F, i, j, k)
    @inbounds tmp_b[i, j, k] = Ad[i, j, k] * F[i, j, k]
    return nothing
end

# Antipodal signs of velocity and conserved components for the fold (if any)
# on dimension d; scalars, partial densities, and energy are always +1.
vel_parity(solver::SolverLike, d::Int, j::Int) =
    solver.folds[d] === nothing ? 1 : solver.folds[d].sigvel[j]
cons_parity(solver::SolverLike, d::Int, c::Int) =
    solver.folds[d] === nothing ? 1 :
    conserved_parity(solver.equations, solver.folds[d].sigvel, c)

assemble_fluxes!(solver::SolverLike, Q) = _assemble_fluxes!(solver, solver.eos, Q)

# No `::Type` argument here: a `Type` inside `pointwise!`'s Vararg defeats
# Julia's specialization heuristics and the body call turns into a per-point
# runtime dispatch, measured as assemble_fluxes! at 9× its cost. The element
# type comes off an array argument instead.
@inline function _fluxes_point!(Q, eos, rho, u, v, w, p, T_ion,
                                cp_mix, mu_art, beta_art, kappa_art, D_art, Y,
                                grad_u, gT, gY, flux, mu0, Pr, Sc, n_species,
                                m1, m2, m3, i_energy, act, o1, o2, o3, i, j, k)
    T = eltype(rho)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = rho[I]
        uv = (u[I], v[I], w[I])
        pI = p[I]
        Tp = T_ion[I]
        E = Q[I, i_energy]
        μ = mu0 + mu_art[I]
        β = beta_art[I]
        κ = mu0 * cp_mix[I] / Pr + kappa_art[I]
        D0 = mu0 / (Sc * ρ)                  # molecular part of each D_k
        divu = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
        # T(2)/T(3), not the literal 2/3: the Float64 literal promotes the
        # normal stresses under a narrower T, making τ a heterogeneous tuple
        # whose runtime indexing is a dynamic field access, an InvalidIRError
        # on device. The Float64 value is identical.
        two_thirds = T(2) / T(3)
        τ11 = μ * (2*grad_u[1,1][I] - two_thirds * divu) + β * divu
        τ22 = μ * (2*grad_u[2,2][I] - two_thirds * divu) + β * divu
        τ33 = μ * (2*grad_u[3,3][I] - two_thirds * divu) + β * divu
        τ12 = μ * (grad_u[1,2][I] + grad_u[2,1][I])
        τ13 = μ * (grad_u[1,3][I] + grad_u[3,1][I])
        τ23 = μ * (grad_u[2,3][I] + grad_u[3,2][I])
        τ = ((τ11, τ12, τ13), (τ12, τ22, τ23), (τ13, τ23, τ33))
        # A collapsed dimension's flux is neither exchanged nor differenced,
        # so assembling it was a third of this phase wasted on a planar run.
        for d in 1:3
            act[d] || continue
            ud = uv[d]
            τd = τ[d]
            # Per-species diffusion with a correction velocity:
            # J_k = −ρ D_k ∇Y_k + ρ Y_k V_c, V_c = Σ_j D_j ∇Y_j,
            # which enforces Σ_k J_k = 0 exactly since ΣY_k = 1.
            Vc = zero(T)
            for sp in 1:n_species
                Vc += (D0 + D_art[sp][I]) * gY[d, sp][I]
            end
            hdiff = zero(T)              # Σ_k h_k J_{k,d}
            for sp in 1:n_species
                Dk = D0 + D_art[sp][I]
                Jkd = ρ * (-Dk * gY[d, sp][I] + Y[sp][I] * Vc)
                flux[d, sp][I] = ρ * Y[sp][I] * ud + Jkd
                hdiff += species_enthalpy(eos, sp, Tp) * Jkd
            end
            flux[d, m1][I] = ρ * ud * uv[1] + (d == 1 ? pI : zero(T)) - τd[1]
            flux[d, m2][I] = ρ * ud * uv[2] + (d == 2 ? pI : zero(T)) - τd[2]
            flux[d, m3][I] = ρ * ud * uv[3] + (d == 3 ? pI : zero(T)) - τd[3]
            flux[d, i_energy][I] = (E + pI) * ud -
                           (uv[1]*τd[1] + uv[2]*τd[2] + uv[3]*τd[3]) -
                           κ * gT[d][I] + hdiff
        end
    end
    return nothing
end

function _assemble_fluxes!(solver::SolverLike{T}, eos, Q) where {T}
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    tr = solver.transport
    n_species = solver.equations.n_species
    i_energy = solver.equations.i_energy
    m1, m2, m3 = solver.equations.i_mom
    ft = solver.field_tuples
    pointwise!(_fluxes_point!, solver.rho, nx, ny, nz,
               Q, eos, solver.rho, solver.u, solver.v, solver.w, solver.p,
               solver.T_ion, solver.cp_mix, solver.mu_art, solver.beta_art,
               solver.kappa_art, ft.D_art, ft.Y, ft.grad_u,
               solver.grad_T_ion, ft.grad_Y, ft.flux,
               tr.mu0, tr.Pr, tr.Sc, n_species, m1, m2, m3, i_energy,
               decomp.active, o1, o2, o3)
    return solver
end

"""
    refresh_primitives!(solver, Q)

Update the primitive fields on `solver` (`rho`, `u`, `v`, `w`, `p`, `T_ion`,
`c`, `cp_mix`, and `Y`) from `Q` by exchanging rank-boundary halos and calling
`primitives!`. This operation is collective and idempotent.

During `compute_rhs!`, including source terms and boundary corrections, the
primitive fields correspond to the state being evaluated. In a `run!` callback,
they instead correspond to the input state of the fifth RK stage and predate the
final stage update and subsequent [`filter_state!`](@ref) pass.

A callback must therefore call this function before reading the primitive
fields. `save_vtk`, `dissipation_rate`, and the mixing diagnostics perform this
update internally. The conserved state `Q` is current in a callback;
`mixture_density` and related functions read it without depending on the
conserved layout.
"""
refresh_primitives!(solver::SolverLike, Q) =
    (exchange_state!(Q, solver.decomp); primitives!(solver, Q); solver)

"""
    compute_primitives_and_gradients!(solver, Q, primitives_current=false)

Refresh halos, primitives, and the physical-component velocity gradients from
`Q`. Sharing this sequence between the RHS and the diagnostics gives both the
same parity and curvature-correction routing, and gradients taken from the
current state.

`solver.grad_u[d, j]` is overwritten with the physical component (∇u)_{dj},
zeroed on a collapsed dimension, and the metric curvature terms are added on
top. Collective: every rank must call it, since each active dimension is a
distributed line solve.

Pass `primitives_current = true` when [`refresh_primitives!`](@ref) has
run on this exact `Q` and only the gradients are wanted; the caller is then
responsible for the claim that nothing has touched `Q` since.
"""
function compute_primitives_and_gradients!(solver::SolverLike, Q,
                                           primitives_current::Bool=false)
    decomp = solver.decomp
    primitives_current || refresh_primitives!(solver, Q)
    vel = (solver.u, solver.v, solver.w)
    for jj in 1:3, d in 1:3
        if decomp.active[d]
            # scaled by 1/h_d incl. stretching Jacobian
            deriv_scaled_along!(solver.grad_u[d, jj], vel[jj], solver, d,
                                vel_parity(solver, d, jj))
        else
            fill!(solver.grad_u[d, jj], 0)
        end
    end
    metric_correct_gradients!(solver, solver.metric)   # additive curvature terms
    return solver
end

"""
    compute_rhs!(solver, Q, dQ, primitives_current=false)

Evaluate dQ/dt into the interior of `dQ` from the conserved state `Q`
(boundary conditions should be enforced on `Q` beforehand). Collapsed dimensions
contribute no derivatives; the axis dimension routes through parity-folded
plans with mirror-filled halos. The interior of `dQ` is zeroed first, so it is
overwritten, not accumulated into.

This is collective: it exchanges halos, runs a distributed line solve per active
dimension per field, and calls `correct_rhs!` for every boundary condition,
which under NSCBC carries collectives of its own. Every rank must call it at the
same point in the step.

The primitives, the gradients, the artificial coefficients, `strain_mag`,
`flux`, and the scratch fields `tmp_a`, `tmp_b`, `sensor` and `sensor_sp` on
`solver` are all overwritten; so are `pairbuf` and `pairout` wherever a paired
fold exists, and `ring_buf` under `detector = :d8`. The docstring of
`compute_artificial!` records which of the sensor scratch fields are dead on
return and may therefore be borrowed by a later phase of the same call.

A trailing `primitives_current = true` skips the opening halo exchange and
primitives pass, and is valid only immediately after the caller performs both on
this same `Q`. See [`compute_primitives_and_gradients!`](@ref); [`step!`](@ref)
passes it for the first RK stage of a `prepared` step, where [`max_rate`](@ref)
has done the work. It is positional, not a keyword, allowing `bench/audit.jl`
to reach the body with `code_typed`, which returns only the
forwarding method of a function with keywords.
"""
function compute_rhs!(solver::SolverLike, Q, dQ, primitives_current::Bool=false)
    decomp = solver.decomp
    compute_primitives_and_gradients!(solver, Q, primitives_current)
    compute_artificial!(solver, Q)
    for d in 1:3
        decomp.active[d] || continue
        deriv_scaled_along!(solver.grad_T_ion[d], solver.T_ion, solver, d, 1)
        for sp in 1:solver.equations.n_species
            deriv_scaled_along!(solver.grad_Y[d, sp], solver.Y[sp], solver, d, 1)
        end
    end
    assemble_fluxes!(solver, Q)
    for d in 1:3
        exchange_dim_batch!(view(solver.flux, d, :), decomp, d)
    end
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    # On an unstretched Cartesian grid every scale factor is exactly 1, so
    # A_d ≡ 1 and inv_J ≡ 1: the A_d·F_d product below is a full-array copy
    # whose result equals its input, and the inv_J multiply is a no-op. Skipping
    # both removes three array streams per (component, dimension), 45 of them
    # for a 5-component 3-D RHS, in what the phase budget shows is the single
    # largest phase. Curved or stretched grids take the general path unchanged.
    unitgeom = solver.metric isa CartesianMetric && all(isnothing, solver.stretch)
    for c in 1:solver.equations.n_cons
        pointwise!(_zero_component_point!, dQ, nx, ny, nz, dQ, c, o1, o2, o3)
        for d in 1:3
            decomp.active[d] || continue
            Fdc = solver.flux[d, c]
            σ = solver.folds[d] === nothing ? 1 : solver.folds[d].sigflux[c]
            if unitgeom
                div_subtract_along!(dQ, c, Fdc, solver, d, σ, nothing)
            else
                # tmp_b = A_d F_d over the full array; A_d is odd in r for the
                # cylindrical axis (A₁ = r), flipping the flux parity.
                Ad = solver.area_d[d]
                n1f, n2f, n3f = size(solver.tmp_b)
                pointwise!(_area_flux_point!, solver.tmp_b, n1f, n2f, n3f,
                           solver.tmp_b, Ad, Fdc)
                div_subtract_along!(dQ, c, solver.tmp_b, solver, d, σ,
                                    solver.inv_J)
            end
        end
    end
    add_metric_sources!(solver, dQ, Q, solver.metric)
    for d in 1:3, side in 1:2
        decomp.active[d] || continue
        correct_rhs!(solver.bcs[d][side], solver, Q, dQ, d, side)
    end
    add_sources!(solver, dQ, Q, solver.tstage)
    return dQ
end


