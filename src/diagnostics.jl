# Volume-integrated and plane-averaged reductions, and the mixing diagnostics
# built on them.
#
# For variable-density mixing the answer is not the flow field: it is the mix
# width, the molecular mixing fraction, the composition PDF, and where the
# kinetic energy went. Those were previously the user's problem to write inside
# a `callback`, which meant every user rewrote the same three MPI reductions and
# got the metric weighting wrong in curvilinear coordinates.
#
# Everything here reduces over the `Decomp` communicators: the transverse
# sub-communicators for a plane average, the full Cartesian communicator for a
# volume integral. Every routine is collective — every rank must call it — and
# every routine returns the same value on every rank.
#
# Quadrature. The computational grid is uniform, so the integrals are midpoint
# sums with the metric Jacobian as the weight, times a half weight at a
# node-centered physical edge. Half-offset (folded) edges take a full weight,
# because those grids are cell-centered by construction and the first node
# already sits half a cell in.
#
# That is exact for a constant on a Cartesian grid, which is the property the
# test asserts. It is NOT exact in curvilinear coordinates: at a node-centered
# edge the half cell is sampled at its boundary rather than its midpoint, so the
# varying Jacobian is caught to O(h²) — e.g. a cylindrical domain closed at
# r = R integrates to (R²/2)(1 + h²/4R²). Second order is well below the O(h)
# statistical noise of the mixing quantities these feed, and the alternative is
# a metric-specific edge rule for a diagnostic. A folded edge has no such term,
# since every one of its cells is a full cell sampled at its midpoint.
#
# Collapsed dimensions contribute a factor of one: a 2-D run integrates per unit
# depth and an axisymmetric run per radian, the same convention the flux
# divergence already uses.

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

"""
    volume_integral(solver, f) -> Float64

∫ f dV over the global domain, with `f` a full padded scalar array. Only the
interior is read, so halo values do not enter the result. Collective. See the
quadrature note at the top of this file for the edge treatment.
"""
function volume_integral(solver::SolverLike, f::AbstractArray{<:Real,3})
    return MPI.Allreduce(_local_volume_integral(solver, f), +, solver.comm)
end

"""
    volume_integral(solver, fs::Vector) -> Float64

Multi-patch form: `fs` holds one full padded array per local patch, aligned
with `solver.patches`. Patch contributions accumulate locally and reduce over
`solver.comm` once. A shared interface-plane node carries a half weight on each
side (each patch's interface end is a node-centered closed edge to its own
quadrature), so the plane is counted exactly once in the total.
"""
function volume_integral(solver::Solver, fs::Vector{<:AbstractArray{<:Real,3}})
    acc = 0.0
    for (i, p) in enumerate(getfield(solver, :patches))
        acc += _local_volume_integral(PatchSolver(solver, p), fs[i])
    end
    return MPI.Allreduce(acc, +, solver.comm)
end

function _local_volume_integral(solver::SolverLike, f::AbstractArray{<:Real,3})
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    dV = cell_measure(solver)
    acc = 0.0
    @inbounds for k in 1:nz
        wk = quad_weight(solver, 3, k)
        for j in 1:ny
            wj = wk * quad_weight(solver, 2, j)
            for i in 1:nx
                I = CartesianIndex(i + o1, j + o2, k + o3)
                # J = 1/inv_J is the metric Jacobian; inv_J already carries any
                # stretching, so this is the physical cell volume.
                acc += wj * quad_weight(solver, 1, i) * f[I] / solver.inv_J[I]
            end
        end
    end
    return acc * dV
end

"""
    domain_volume(solver) -> Float64

∫ dV, i.e. `volume_integral` of unity. Collective.
"""
function domain_volume(solver::SolverLike)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    acc = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        acc += quad_weight(solver, 1, i) * quad_weight(solver, 2, j) *
               quad_weight(solver, 3, k) / solver.inv_J[I]
    end
    return MPI.Allreduce(acc * cell_measure(solver), +, solver.comm)
end

"""
    volume_average(solver, f) -> Float64

∫ f dV / ∫ dV. Collective.
"""
volume_average(solver::SolverLike, f) = volume_integral(solver, f) / domain_volume(solver)

"""
    plane_profile(solver, f, d) -> Vector{Float64}

Area-weighted average of `f` over the two dimensions transverse to `d`, as a
freshly allocated profile of length `n_global[d]` returned on every rank. `f` is
a full padded scalar array, of which only the interior is read. The weight is the
transverse area element `area_d[d]`, so this is the plain arithmetic mean on a
Cartesian grid and the correct area average on a curvilinear one. Collective.
"""
function plane_profile(solver::Solver, f::AbstractArray{<:Real,3}, d::Int)
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

"""
    profile_coordinate(solver, d) -> Vector{Float64}

Physical coordinate of each station of a `plane_profile` along `d`, as a vector
of length `n_global[d]`. The companion length element is `profile_spacing`.
Collective, and returns the same vector on every rank.
"""
function profile_coordinate(solver::Solver, d::Int)
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
a node-centered edge, so that `sum(profile_spacing(solver, d))` is the domain
extent. On a curvilinear grid the scale factor varies across the transverse
plane. The returned spacing is therefore its area-weighted plane average, using
the same weights as `plane_profile`, so the two quantities compose into a line
integral.

On an active dimension this is collective, since the plane average is. For a
collapsed dimension the result is `[1.0]` instead, the factor-of-one convention
described at the top of this file, and no communication takes place; every rank
takes that branch together.
"""
function profile_spacing(solver::Solver, d::Int)
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
# Cook & Dimotakis). `dim` is the inhomogeneous direction — the one the
# interface moves along — and everything is an integral of plane averages along
# it. Mass fractions are used rather than volume fractions, which is what the
# conserved layout carries directly and what an Atwood-number-independent
# statement of θ wants.

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
    species_pdf(solver, sp; nbins=32) -> (centers, pdf)

Volume-weighted probability density of mass fraction Y_sp over the whole domain,
on `nbins` uniform bins of [0, 1]. It is normalized so that
`sum(pdf) * binwidth == 1`. Peaks at 0 and 1 indicate separated fluids, whereas
an interior distribution indicates mixed compositions. Collective.

This method reads `solver.Y` directly and does not accept `Q`; call
[`refresh_primitives!`](@ref) first when current primitive fields are not
otherwise guaranteed, including from a `run!` callback.
"""
function species_pdf(solver::Solver, sp::Int; nbins::Int=32)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    dV = cell_measure(solver)
    Y = solver.Y[sp]
    bins = zeros(nbins)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        w = quad_weight(solver, 1, i) * quad_weight(solver, 2, j) *
            quad_weight(solver, 3, k) * dV / solver.inv_J[I]
        b = clamp(1 + floor(Int, Y[I] * nbins), 1, nbins)
        bins[b] += w
    end
    bins = MPI.Allreduce(bins, +, decomp.comm)
    width = 1 / nbins
    total = sum(bins)
    total > 0 && (bins ./= total * width)
    return ([(b - 0.5) * width for b in 1:nbins], bins)
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
    turbulent_kinetic_energy(solver, Q; dim=1) -> Float64

Average of [`tke_profile`](@ref) along `dim` weighted by `profile_spacing`, so
one number for the whole layer. Collective, and it inherits `tke_profile`'s
refresh of the primitives and its use of `solver.tmp_a` as scratch.
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
gradient pass, so the diagnostic is intended for periodic rather than per-step
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
    red = MPI.Allreduce([_local_volume_integral(solver, diss),
                         _local_volume_integral(solver, solver.rho)], +,
                        solver.comm)
    return red[1] / red[2]
end
