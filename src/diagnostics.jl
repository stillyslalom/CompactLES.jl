# Volume-integrated and plane-averaged reductions, and the mixing diagnostics
# built on them.
#
# For variable-density mixing the answer is not the flow field: it is the mix
# width, the molecular mixing fraction, the composition PDF, and where the
# kinetic energy went. Those were previously the user's problem to write inside
# a `callback`, so every user rewrote the same three MPI reductions and
# got the metric weighting wrong in curvilinear coordinates.
#
# Everything here reduces over the `Decomp` communicators: the transverse
# sub-communicators for a plane average, the full Cartesian communicator for a
# volume integral. Every routine is collective (every rank must call it) and
# every routine returns the same value on every rank.
#
# Quadrature. The computational grid is uniform, so the integrals are midpoint
# sums with the metric Jacobian as the weight, times a half weight at a
# node-centered physical edge. Half-offset (folded) edges take a full weight,
# because those grids are cell-centered by construction and the first node
# sits half a cell in.
#
# That is exact for a constant on a Cartesian grid, which is the property the
# test asserts. It is not exact in curvilinear coordinates: at a node-centered
# edge the half cell is sampled at its boundary, not its midpoint, so the
# varying Jacobian is approximated to O(h²); e.g. a cylindrical domain closed at
# r = R integrates to (R²/2)(1 + h²/4R²). Second order is well below the O(h)
# statistical noise of the mixing quantities these feed, and the alternative is
# a metric-specific edge rule for a diagnostic. A folded edge has no such term,
# since every one of its cells is a full cell sampled at its midpoint.
#
# Collapsed dimensions contribute a factor of one: a 2-D run integrates per unit
# depth and an axisymmetric run per radian, the same convention the flux
# divergence uses.
#
# Composite forms. A multi-patch solver (a `patch_grid`, a refined level,
# or both) takes the forms whose field argument is a `Vector` aligned with
# `solver.patches`, one full padded array per held patch, or a state
# vector; these accumulate every patch's contribution on this rank and
# reduce once over `solver.comm`, so a rank holding no piece of a refined
# level enters the same reduction and returns the same number. Each
# coarse node's weight is multiplied by the fraction of its quadrature
# cell no child level covers (`Patch.covered`, `uncovered_fraction`), and
# a plane average by the in-plane fraction, so a covered coarse node is
# excluded exactly once and a child's face plane is counted once between
# its two sides. A profile on a composite grid is sampled at the root's
# stations, each plane average combining the uncovered root nodes and the
# nodes of every finer patch whose planes coincide with it, with each
# patch's own transverse cell measure. The single-array forms are the
# single-patch quadrature as it always was and apply no mask; on a
# refined run they are meaningful on a `PatchSolver` of one patch only.

"""
    quad_weight(solver, d, i) -> Float64

Quadrature weight of local interior index `i` along dimension `d`: 1 in the
interior and on any folded (half-offset) edge, ½ on a node-centered physical
edge, 1 on collapsed and periodic dimensions.
"""
@inline function quad_weight(solver::SolverLike, d::Int, i::Int)
    decomp = solver.decomp
    (decomp.active[d] && !decomp.periodic[d]) || return 1.0
    g = decomp.offset[d] + i                          # 1-based global index
    fold = solver.folds[d]
    g == 1 && return (fold !== nothing && fold.lo) ? 1.0 : 0.5
    g == decomp.n_global[d] && return (fold !== nothing && fold.hi) ? 1.0 : 0.5
    return 1.0
end

"Computational cell measure Πh_d over active dimensions (collapsed dims give 1)."
cell_measure(solver::SolverLike) = prod(ntuple(d -> solver.decomp.active[d] ?
                                           solver.h[d] : one(eltype(solver.h)), 3))

# Whether a solver's diagnostics take the composite (multi-patch, masked)
# forms: several patches on this rank, or a refined level somewhere in the
# run, which a rank holding only the root must still enter the reductions
# of. A `PatchSolver` is one patch by construction.
_composite(solver::Solver) = nlevels(solver) > 1 || npatches(solver) > 1
_composite(::PatchSolver) = false

"""
    volume_integral(solver, f) -> Float64

∫ f dV over the global domain, with `f` a full padded scalar array. Only the
interior is read, so halo values do not enter the result. Collective. See the
quadrature note at the top of this file for the edge treatment. The
single-patch quadrature: on a refined run use the `Vector` form below.
"""
function volume_integral(solver::SolverLike, f::AbstractArray{<:Real,3})
    return MPI.Allreduce(_local_volume_integral(solver, f, false), +, solver.comm)
end

"""
    volume_integral(solver, fs::Vector) -> Float64

Multi-patch form: `fs` holds one full padded array per local patch, aligned
with `solver.patches`. Patch contributions accumulate locally and reduce over
`solver.comm` once. A shared interface-plane node carries a half weight on each
side (each patch's interface end is a node-centered closed edge to its own
quadrature), so the plane is counted exactly once in the total, and a coarse
node a child level covers is weighted by the uncovered fraction of its cell
(`Patch.covered`), so the composite grid is integrated once. Every rank of
`solver.comm` must call it, a rank holding no piece of a refined level
included.
"""
function volume_integral(solver::Solver, fs::Vector{<:AbstractArray{<:Real,3}})
    acc = 0.0
    for (i, p) in enumerate(getfield(solver, :patches))
        acc += _local_volume_integral(PatchSolver(solver, p), fs[i], true)
    end
    return MPI.Allreduce(acc, +, solver.comm)
end

# One patch's interior contribution. `masked` applies the covered mask
# (the composite forms); the single-patch forms leave it off and are
# bit-identical to the quadrature before the masks existed.
function _local_volume_integral(solver::SolverLike, f::AbstractArray{<:Real,3},
                                masked::Bool)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    dV = cell_measure(solver)
    covered = solver.covered
    acc = 0.0
    @inbounds for k in 1:nz
        wk = quad_weight(solver, 3, k)
        for j in 1:ny
            wj = wk * quad_weight(solver, 2, j)
            for i in 1:nx
                I = CartesianIndex(i + o1, j + o2, k + o3)
                w = wj * quad_weight(solver, 1, i)
                if masked
                    m = covered[I]
                    m == 0 || (w *= uncovered_fraction(m))
                end
                # J = 1/inv_J is the metric Jacobian; inv_J carries any
                # stretching, so this is the physical cell volume.
                acc += w * f[I] / solver.inv_J[I]
            end
        end
    end
    return acc * dV
end

"""
    domain_volume(solver) -> Float64

∫ dV, i.e. `volume_integral` of unity. Collective. On a multi-patch or
refined solver this is the composite quadrature with the covered masks
applied, so it is the physical volume exactly once.
"""
function domain_volume(solver::SolverLike)
    if _composite(solver)
        acc = 0.0
        for p in getfield(solver, :patches)
            acc += _local_domain_volume(PatchSolver(solver, p), true)
        end
        return MPI.Allreduce(acc, +, solver.comm)
    end
    return MPI.Allreduce(_local_domain_volume(solver, false), +, solver.comm)
end

function _local_domain_volume(solver::SolverLike, masked::Bool)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    covered = solver.covered
    acc = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        w = quad_weight(solver, 1, i) * quad_weight(solver, 2, j) *
            quad_weight(solver, 3, k)
        if masked
            m = covered[I]
            m == 0 || (w *= uncovered_fraction(m))
        end
        acc += w / solver.inv_J[I]
    end
    return acc * cell_measure(solver)
end

"""
    volume_average(solver, f) -> Float64
    volume_average(solver, fs::Vector) -> Float64

∫ f dV / ∫ dV. Collective. The `Vector` form is the composite one.
"""
volume_average(solver::SolverLike, f) = volume_integral(solver, f) / domain_volume(solver)

"""
    plane_profile(solver, f, d) -> Vector{Float64}

Area-weighted average of `f` over the two dimensions transverse to `d`, as a
freshly allocated profile of length `n_global[d]` returned on every rank. `f` is
a full padded scalar array, of which only the interior is read. The weight is the
transverse area element `area_d[d]`, so this is the plain arithmetic mean on a
Cartesian grid and the correct area average on a curvilinear one. Collective.
The single-patch form; see the `Vector` form for a multi-patch or refined
solver.
"""
function plane_profile(solver::SolverLike, f::AbstractArray{<:Real,3}, d::Int)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    n_d = decomp.n_local[d]
    num = zeros(n_d)
    den = zeros(n_d)
    Ad = solver.area_d[d]
    @inbounds for k in 1:nz
        wk = quad_weight(solver, 3, k)
        for j in 1:ny
            wj = wk * quad_weight(solver, 2, j)
            for i in 1:nx
                w = wj * quad_weight(solver, 1, i)
                # Undo the weight along d itself: this averages over the plane,
                # it does not integrate along the profile direction.
                w /= quad_weight(solver, d, (d == 1 ? i : d == 2 ? j : k))
                I = CartesianIndex(i + o1, j + o2, k + o3)
                idx = d == 1 ? i : d == 2 ? j : k
                num[idx] += w * Ad[I] * f[I]
                den[idx] += w * Ad[I]
            end
        end
    end
    # Sum over the two transverse sub-communicators; the one along d is left
    # alone, then used to gather the stations into the global profile.
    for t in 1:3
        t == d && continue
        decomp.active[t] || continue
        # An undivided transverse dimension has a one-rank sub-communicator,
        # where the reduction is a copy of a length-n_d vector for no effect.
        # The same guard covers the gather along `d` below.
        decomp.sub_size[t] == 1 && continue
        num = MPI.Allreduce(num, +, decomp.sub[t])
        den = MPI.Allreduce(den, +, decomp.sub[t])
    end
    local_prof = num ./ den
    decomp.sub_size[d] == 1 && return local_prof
    counts = MPI.Allgather(Int32(n_d), decomp.sub[d])
    out = Vector{Float64}(undef, sum(counts))
    MPI.Allgatherv!(local_prof, MPI.VBuffer(out, counts), decomp.sub[d])
    return out
end

# One patch's contribution to the composite plane average along `d`, into
# the root-station accumulators `num`/`den` (length `n_global[d]`). A
# level-ℓ node m (this level's node space) lies on root station
# (m − 1) / 3^ℓ + 1 when 3^ℓ divides m − 1; the other planes of a finer
# patch are not sampled. The weight is the transverse quadrature weights,
# the in-plane uncovered fraction, the patch's transverse cell measure
# (patches differ in spacing, so it does not cancel as it does on one
# grid) and the area element.
function _plane_accumulate!(num::Vector{Float64}, den::Vector{Float64},
                            solver::PatchSolver, f::AbstractArray{<:Real,3},
                            d::Int)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    patch = solver.patch
    stride = 3^patch.level
    base = patch.region.offset[d] + decomp.offset[d]
    # A plane shared with a same-level neighbor along `d` is present in
    # both patches, so each side contributes half of it; a plane facing
    # the parent is the patch's alone, the parent's in-plane fraction
    # there being zero.
    shared_lo = patch.faces[d][1] != 0
    shared_hi = patch.faces[d][2] != 0
    n_patch = patch.region.extent[d]
    area = 1.0
    for t in 1:3
        t == d && continue
        decomp.active[t] && (area *= solver.h[t])
    end
    Ad = solver.area_d[d]
    covered = solver.covered
    @inbounds for k in 1:nz
        wk = d == 3 ? 1.0 : quad_weight(solver, 3, k)
        for j in 1:ny
            wj = wk * (d == 2 ? 1.0 : quad_weight(solver, 2, j))
            for i in 1:nx
                il = d == 1 ? i : d == 2 ? j : k
                m = base + il
                (m - 1) % stride == 0 || continue
                g = (m - 1) ÷ stride + 1
                w = wj * (d == 1 ? 1.0 : quad_weight(solver, 1, i)) * area
                node = decomp.offset[d] + il
                ((node == 1 && shared_lo) || (node == n_patch && shared_hi)) &&
                    (w *= 0.5)
                I = CartesianIndex(i + o1, j + o2, k + o3)
                c = covered[I]
                c == 0 || (w *= uncovered_plane_fraction(c, d))
                w *= Ad[I]
                num[g] += w * f[I]
                den[g] += w
            end
        end
    end
    return num
end

# The composite plane average: `field(ps, i)` hands back the array to
# average for patch `i` (bound as `ps`), evaluated and consumed one patch
# at a time, so a shared workspace scratch may serve every patch in turn.
# One Allreduce over `solver.comm` of both accumulators.
function _composite_profile(field::F, solver::Solver, d::Int) where {F}
    n = solver.n_global[d]
    num = zeros(n)
    den = zeros(n)
    for (i, p) in enumerate(getfield(solver, :patches))
        ps = PatchSolver(solver, p)
        _plane_accumulate!(num, den, ps, field(ps, i), d)
    end
    red = MPI.Allreduce([num; den], +, solver.comm)
    return red[1:n] ./ red[n+1:2n]
end

"""
    plane_profile(solver, fs::Vector, d) -> Vector{Float64}

The composite form: `fs` holds one full padded array per held patch,
aligned with `solver.patches`, and the profile is sampled at the root's
`n_global[d]` stations, each plane average combining the uncovered root
nodes with the nodes of every finer patch whose planes coincide with it.
Collective over `solver.comm`, which every rank must enter.
"""
plane_profile(solver::Solver, fs::Vector{<:AbstractArray{<:Real,3}}, d::Int) =
    _composite_profile((ps, i) -> fs[i], solver, d)

"""
    profile_coordinate(solver, d) -> Vector{Float64}

Physical coordinate of each station of a `plane_profile` along `d`, as a vector
of length `n_global[d]`. The companion length element is `profile_spacing`.
Collective, and returns the same vector on every rank. On a refined solver
the stations are the root's, so this is the root patch's coordinate vector.
"""
function profile_coordinate(solver::SolverLike, d::Int)
    _composite(solver) &&
        return profile_coordinate(PatchSolver(solver, getfield(solver, :patches)[1]), d)
    decomp = solver.decomp
    loc = Float64[xcoord(solver, d, i) for i in 1:decomp.n_local[d]]
    decomp.sub_size[d] == 1 && return loc
    counts = MPI.Allgather(Int32(length(loc)), decomp.sub[d])
    out = Vector{Float64}(undef, sum(counts))
    MPI.Allgatherv!(loc, MPI.VBuffer(out, counts), decomp.sub[d])
    return out
end

"""
    profile_spacing(solver, d) -> Vector{Float64}

Physical length element at each station along `d`, including the half weight at
a node-centered edge. `sum(profile_spacing(solver, d))` is therefore the domain
extent. On a curvilinear grid the scale factor varies across the transverse
plane. The returned spacing is therefore its area-weighted plane average, using
the same weights as `plane_profile`, so the two quantities compose into a line
integral.

On an active dimension this is collective, since the plane average is. For a
collapsed dimension the result is `[1.0]` instead, the factor-of-one convention
described at the top of this file, and no communication takes place; every rank
takes that branch together. On a refined solver the stations are the root's
and this is the root patch's spacing, unmasked: the length element along the
profile is geometry, not a quadrature over the plane.
"""
function profile_spacing(solver::SolverLike, d::Int)
    _composite(solver) &&
        return profile_spacing(PatchSolver(solver, getfield(solver, :patches)[1]), d)
    decomp = solver.decomp
    decomp.active[d] || return [1.0]
    # 1/inv_h[d] is the physical arclength per unit computational coordinate.
    invh = solver.inv_h[d]
    o1, o2, o3 = decomp.n_halo_d
    scale = similar(invh)
    @inbounds for k in axes(invh, 3), j in axes(invh, 2), i in axes(invh, 1)
        scale[i, j, k] = 1 / invh[i, j, k]
    end
    prof = plane_profile(solver, scale, d) .* solver.h[d]
    w = ones(length(prof))
    fold = solver.folds[d]
    if !decomp.periodic[d]
        (fold !== nothing && fold.lo) || (w[1] = 0.5)
        (fold !== nothing && fold.hi) || (w[end] = 0.5)
    end
    return prof .* w
end

# ---------------------------------------------------------------------------
# Mixing diagnostics.
#
# Conventions follow the variable-density turbulent-mixing literature (Youngs;
# Cook & Dimotakis). `dim` is the inhomogeneous direction, the one the
# interface moves along, and everything is an integral of plane averages along
# it. Mass fractions are used, not volume fractions: they are what the
# conserved layout carries directly and what an Atwood-number-independent
# statement of θ supports.

"""
    mix_width(solver, Q; dim=1, species=(1, 2)) -> Float64

Integral mix width W = ∫ 4⟨Y_a⟩⟨Y_b⟩ dx along `dim`, with ⟨·⟩ the plane average.
The factor of 4 normalizes a fully mixed layer of thickness L to W = L. This
bulk measure describes the extent of a Rayleigh–Taylor or Richtmyer–Meshkov
layer but does not distinguish stirring from molecular mixing: two unmixed
fluids interleaved at the grid scale give the same W as a molecularly mixed
layer. [`molecular_mixing`](@ref) provides that distinction. `species` names the
pair `(a, b)` of species indices. Collective.

Calls [`refresh_primitives!`](@ref) before evaluating the diagnostic, including
when invoked from a `run!` callback.
"""
function mix_width(solver::Solver, Q; dim::Int=1, species=(1, 2))
    a, b = species
    refresh_primitives!(solver, Q)
    Ya = plane_profile(solver, solver.Y[a], dim)
    Yb = plane_profile(solver, solver.Y[b], dim)
    dx = profile_spacing(solver, dim)
    return sum(4 .* Ya .* Yb .* dx)
end

"""
    mix_width(solver, states::Vector; dim=1, species=(1, 2)) -> Float64

The composite form for a multi-patch or refined solver, `states` aligned
with `solver.patches`: the plane averages are the composite ones at the
root's stations. Collective over `solver.comm`.
"""
function mix_width(solver::Solver, states::Vector{<:ConservedState};
                   dim::Int=1, species=(1, 2))
    a, b = species
    refresh_primitives!(solver, states)
    Ya = _composite_profile((ps, i) -> ps.Y[a], solver, dim)
    Yb = _composite_profile((ps, i) -> ps.Y[b], solver, dim)
    dx = profile_spacing(solver, dim)
    return sum(4 .* Ya .* Yb .* dx)
end

"""
    molecular_mixing(solver, Q; dim=1, species=(1, 2)) -> Float64

Youngs' molecular mixing fraction θ = ∫⟨Y_a Y_b⟩ dx / ∫⟨Y_a⟩⟨Y_b⟩ dx along
`dim`. A value of 0 denotes interleaved but unmixed fluids, and a value of 1
denotes uniform composition across the layer. The result is 0 when the
denominator is not positive, which is the case where the two species overlap
nowhere. Collective.

Calls [`refresh_primitives!`](@ref) before evaluating the diagnostic, including
when invoked from a `run!` callback, and overwrites `solver.tmp_a` as scratch.
"""
function molecular_mixing(solver::Solver, Q; dim::Int=1, species=(1, 2))
    a, b = species
    refresh_primitives!(solver, Q)
    prod_ab = solver.tmp_a
    @inbounds for idx in eachindex(prod_ab)
        prod_ab[idx] = solver.Y[a][idx] * solver.Y[b][idx]
    end
    num = plane_profile(solver, prod_ab, dim)
    Ya = plane_profile(solver, solver.Y[a], dim)
    Yb = plane_profile(solver, solver.Y[b], dim)
    dx = profile_spacing(solver, dim)
    den = sum(Ya .* Yb .* dx)
    return den > 0 ? sum(num .* dx) / den : 0.0
end

"""
    molecular_mixing(solver, states::Vector; dim=1, species=(1, 2)) -> Float64

The composite form, `states` aligned with `solver.patches`. The product
Y_a Y_b is formed in each patch's `tmp_a` and consumed before the next
patch's, since equal-extent tiles share that scratch. Collective over
`solver.comm`.
"""
function molecular_mixing(solver::Solver, states::Vector{<:ConservedState};
                          dim::Int=1, species=(1, 2))
    a, b = species
    refresh_primitives!(solver, states)
    function product(ps, i)
        prod_ab = ps.tmp_a
        Ya, Yb = ps.Y[a], ps.Y[b]
        @inbounds for idx in eachindex(prod_ab)
            prod_ab[idx] = Ya[idx] * Yb[idx]
        end
        return prod_ab
    end
    num = _composite_profile(product, solver, dim)
    Ya = _composite_profile((ps, i) -> ps.Y[a], solver, dim)
    Yb = _composite_profile((ps, i) -> ps.Y[b], solver, dim)
    dx = profile_spacing(solver, dim)
    den = sum(Ya .* Yb .* dx)
    return den > 0 ? sum(num .* dx) / den : 0.0
end

"""
    species_pdf(solver, sp; nbins=32) -> (centers, pdf)

Volume-weighted probability density of mass fraction Y_sp over the whole domain,
on `nbins` uniform bins of [0, 1]. Its normalization gives
`sum(pdf) * binwidth == 1`. Peaks at 0 and 1 indicate separated fluids, whereas
an interior distribution indicates mixed compositions. Collective.

This method reads `solver.Y` directly and does not accept `Q`; call
[`refresh_primitives!`](@ref) first when current primitive fields are not
otherwise guaranteed, including from a `run!` callback. On a multi-patch or
refined solver it reads every held patch's `Y` and applies the covered
masks, the composite quadrature.
"""
function species_pdf(solver::Solver, sp::Int; nbins::Int=32)
    bins = zeros(nbins)
    if _composite(solver)
        for p in getfield(solver, :patches)
            _pdf_accumulate!(bins, PatchSolver(solver, p), sp, true)
        end
    else
        _pdf_accumulate!(bins, solver, sp, false)
    end
    bins = MPI.Allreduce(bins, +, solver.comm)
    width = 1 / nbins
    total = sum(bins)
    total > 0 && (bins ./= total * width)
    return ([(b - 0.5) * width for b in 1:nbins], bins)
end

function _pdf_accumulate!(bins::Vector{Float64}, solver::SolverLike, sp::Int,
                          masked::Bool)
    nbins = length(bins)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    dV = cell_measure(solver)
    Y = solver.Y[sp]
    covered = solver.covered
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        w = quad_weight(solver, 1, i) * quad_weight(solver, 2, j) *
            quad_weight(solver, 3, k) * dV / solver.inv_J[I]
        if masked
            m = covered[I]
            m == 0 || (w *= uncovered_fraction(m))
        end
        b = clamp(1 + floor(Int, Y[I] * nbins), 1, nbins)
        bins[b] += w
    end
    return bins
end

"""
    tke_profile(solver, Q; dim=1) -> Vector{Float64}

Favre-averaged turbulent kinetic energy ⟨ρ|u − ũ|²⟩ / (2⟨ρ⟩) as a profile along
`dim`, where ũ is the Favre (density-weighted) plane mean. Subtracting the plane
mean removes the bulk translation of the interface and retains velocity
fluctuations associated with the instability. The profile has length
`n_global[dim]` and is the same on every rank. Collective.

Calls [`refresh_primitives!`](@ref) before evaluating the diagnostic, and
overwrites `solver.tmp_a` as scratch.
"""
function tke_profile(solver::Solver, Q; dim::Int=1)
    refresh_primitives!(solver, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    ρ = solver.rho
    vel = (solver.u, solver.v, solver.w)
    tmp = solver.tmp_a
    # Favre mean of each velocity component: ũ = ⟨ρu⟩/⟨ρ⟩.
    ρbar = plane_profile(solver, ρ, dim)
    favre = ntuple(3) do c
        @inbounds for idx in eachindex(tmp)
            tmp[idx] = ρ[idx] * vel[c][idx]
        end
        plane_profile(solver, tmp, dim) ./ ρbar
    end
    off = decomp.offset[dim]
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        g = off + (dim == 1 ? i : dim == 2 ? j : k)
        s = 0.0
        for c in 1:3
            du = vel[c][I] - favre[c][g]
            s += du * du
        end
        tmp[I] = ρ[I] * s
    end
    return plane_profile(solver, tmp, dim) ./ (2 .* ρbar)
end

"""
    tke_profile(solver, states::Vector; dim=1) -> Vector{Float64}

The composite form, `states` aligned with `solver.patches`: the Favre
means are the composite plane averages at the root's stations, and a finer
patch's node reads the mean of the root station its plane coincides with.
Each patch fills its `tmp_a` in turn. Collective over `solver.comm`.
"""
function tke_profile(solver::Solver, states::Vector{<:ConservedState}; dim::Int=1)
    refresh_primitives!(solver, states)
    ρbar = _composite_profile((ps, i) -> ps.rho, solver, dim)
    favre = ntuple(3) do c
        function momentum(ps, i)
            tmp = ps.tmp_a
            ρ = ps.rho
            vel = (ps.u, ps.v, ps.w)[c]
            @inbounds for idx in eachindex(tmp)
                tmp[idx] = ρ[idx] * vel[idx]
            end
            return tmp
        end
        _composite_profile(momentum, solver, dim) ./ ρbar
    end
    function fluctuation(ps, i)
        decomp = ps.decomp
        o1, o2, o3 = decomp.n_halo_d
        nx, ny, nz = decomp.n_local
        ρ = ps.rho
        vel = (ps.u, ps.v, ps.w)
        tmp = ps.tmp_a
        stride = 3^ps.patch.level
        base = ps.patch.region.offset[dim] + decomp.offset[dim]
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            m = base + (dim == 1 ? i : dim == 2 ? j : k)
            # Only coincident planes enter the average; the others take
            # any finite value.
            g = (m - 1) % stride == 0 ? (m - 1) ÷ stride + 1 : 1
            s = 0.0
            for c in 1:3
                du = vel[c][I] - favre[c][g]
                s += du * du
            end
            tmp[I] = ρ[I] * s
        end
        return tmp
    end
    return _composite_profile(fluctuation, solver, dim) ./ (2 .* ρbar)
end

"""
    turbulent_kinetic_energy(solver, Q; dim=1) -> Float64

Average of [`tke_profile`](@ref) along `dim` weighted by `profile_spacing`, so
one number for the whole layer. Collective, and it inherits `tke_profile`'s
refresh of the primitives and its use of `solver.tmp_a` as scratch. With a
state vector it is the composite form.
"""
function turbulent_kinetic_energy(solver::Solver, Q; dim::Int=1)
    k = tke_profile(solver, Q; dim=dim)
    dx = profile_spacing(solver, dim)
    return sum(k .* dx) / sum(dx)
end

"""
    dissipation_rate(solver, Q) -> Float64

Volume-averaged resolved dissipation ⟨τ_ij ∂u_i/∂x_j⟩ / ⟨ρ⟩, with τ built from
the total viscosity: molecular μ₀ plus the artificial μ\\* and β\\*. The
artificial terms represent the subgrid model in this scheme and can dominate
the energy sink at a shock; they are therefore included in the reported rate.

Calls `compute_primitives_and_gradients!` and `compute_artificial!`, so
the result is independent of the current RK stage. This requires one additional
gradient pass, so the diagnostic is intended for periodic, not per-step,
evaluation. Those two passes overwrite `solver.grad_u` and the artificial
coefficient, sensor and scratch fields, and the integrand is then accumulated
into `solver.tmp_b`; `compute_rhs!` rebuilds all of them at the next stage.
Collective.
"""
function dissipation_rate(solver::Solver, Q)
    compute_primitives_and_gradients!(solver, Q)
    compute_artificial!(solver, Q)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    grad_u = solver.grad_u
    mu0 = solver.transport.mu0
    diss = solver.tmp_b
    fill!(diss, 0)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        μ = mu0 + solver.mu_art[I]
        β = solver.beta_art[I]
        divu = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
        acc = 0.0
        for b in 1:3, a in 1:3
            τ = μ * (grad_u[a, b][I] + grad_u[b, a][I]) +
                (a == b ? (β - 2μ / 3) * divu : 0.0)
            acc += τ * grad_u[a, b][I]
        end
        diss[I] = acc
    end
    # One collective, not two: both integrals are sums over the same rank set,
    # and this is called once per diagnostic output on every rank.
    red = MPI.Allreduce([_local_volume_integral(solver, diss, false),
                         _local_volume_integral(solver, solver.rho, false)], +,
                        solver.comm)
    return red[1] / red[2]
end

"""
    dissipation_rate(solver, states::Vector) -> Float64

The composite form, `states` aligned with `solver.patches`: each held patch
takes its gradient and artificial passes in turn (its shared scratch is
consumed before the next patch's), the integrand is accumulated under the
covered masks, and both integrals reduce once over `solver.comm`. The
gradients read each patch's ghost and shell nodes, so the state must be as
`run!` leaves it after a step, shells imposed and shared planes exchanged
([`sync_patches!`](@ref), [`sync_levels!`](@ref)); a state that has only been
initialized needs those first.
"""
function dissipation_rate(solver::Solver, states::Vector{<:ConservedState})
    acc = zeros(2)
    for (ps, Q) in eachpatch(solver, states)
        compute_primitives_and_gradients!(ps, Q)
        compute_artificial!(ps, Q)
        decomp = ps.decomp
        o1, o2, o3 = decomp.n_halo_d
        nx, ny, nz = decomp.n_local
        grad_u = ps.grad_u
        mu0 = solver.transport.mu0
        diss = ps.tmp_b
        fill!(diss, 0)
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            I = CartesianIndex(i + o1, j + o2, k + o3)
            μ = mu0 + ps.mu_art[I]
            β = ps.beta_art[I]
            divu = grad_u[1, 1][I] + grad_u[2, 2][I] + grad_u[3, 3][I]
            s = 0.0
            for b in 1:3, a in 1:3
                τ = μ * (grad_u[a, b][I] + grad_u[b, a][I]) +
                    (a == b ? (β - 2μ / 3) * divu : 0.0)
                s += τ * grad_u[a, b][I]
            end
            diss[I] = s
        end
        acc[1] += _local_volume_integral(ps, diss, true)
        acc[2] += _local_volume_integral(ps, ps.rho, true)
    end
    red = MPI.Allreduce(acc, +, solver.comm)
    return red[1] / red[2]
end
