# The binned form of the ablation drive (src/ablation.jl): the front and the layer
# normalization summed over angular bins about the capsule centre rather than along
# grid lines, so the drive runs on r–z (cylindrical) and 3-D Cartesian grids, on
# stretched grids and on refined levels. docs/src/explanation/ablation-drive.md,
# "Geometry".
#
# Three pointwise passes per evaluation, as in the per-line form: the front marker
# g = ρₐ(−∂ρₐ/∂R)⁺ (binned into Σ g dV, Σ g R dV → R_f per bin), the layer weight k and
# the front-area kernel a = e^(−x²)/(√π δ) (binned into Σ k dV, Σ a dV → N = A/K per
# bin), and the source with w = k N, R_f and N interpolated at the point's direction.
# On a refined solver the first two passes run over a whole level before its
# right-hand side (`prepare_source!`), the bins reduced over the level's ranks; on
# one patch they run inside `add_source!`, reduced over the patch's ranks.

_binned(d::AblationDrive, solver) =
    d.geometry === :binned || (d.geometry === :auto && !(solver.metric isa SphericalMetric))

const _ABLATION_FOLDS = Union{AxisBC,OriginBC,PoleBC,SymmetryPlaneBC}

# (nμ, nφ, polar resolved, azimuth resolved) of the bins on `solver`'s grid.
_bin_layout(d::AblationDrive, solver) = _bin_layout(d, solver.metric, solver.decomp.active)

function _bin_layout(d::AblationDrive, m, a)
    if m isa CartesianMetric
        all(a) || throw(ArgumentError("AblationDrive: a capsule on a CartesianMetric grid needs all three dimensions resolved"))
        polar, azim = true, true
    elseif m isa CylindricalMetric
        a[1] && a[3] ||
            throw(ArgumentError("AblationDrive: a capsule on a CylindricalMetric grid needs r and z resolved (an r–z plane)"))
        d.center[1] == 0 && d.center[2] == 0 ||
            throw(ArgumentError("AblationDrive: on a CylindricalMetric grid the capsule centre must lie on the axis, center = (0, 0, z)"))
        polar, azim = true, a[2]
    elseif m isa SphericalMetric
        all(iszero, d.center) ||
            throw(ArgumentError("AblationDrive: on a SphericalMetric grid the capsule centre is the origin"))
        a[1] || throw(ArgumentError("AblationDrive needs a resolved radius"))
        polar, azim = a[2], a[3]
    else
        throw(ArgumentError("AblationDrive: unsupported metric $(typeof(m))"))
    end
    nμ, nφ = d.bins === nothing ? (azim ? (32, 64) : (64, 1)) : d.bins
    return (polar ? nμ : 1, azim ? nφ : 1, polar, azim)
end

function _on_device(like, A::Array)
    like isa Array && return A
    B = similar(like, eltype(A), size(A))
    copyto!(B, A)
    return B
end

# The static per-point geometry of one patch: distance R from the centre, the
# components q_d = e_d · R̂ of the radial direction in the local orthonormal basis,
# its direction (μ, φ), the volume weight J (dV up to the constant h₁h₂h₃) and bin,
# and per dimension whether each block end is a fold or symmetry plane (the density
# is mirrored there; other block ends take one-sided differences).
function _build_bin_geometry(d::AblationDrive, solver)
    T = eltype(solver.rho)
    decomp = solver.decomp
    nx, ny, nz = decomp.n_local
    o1, o2, o3 = decomp.n_halo_d
    nμ, nφ, polar, azim = _bin_layout(d, solver)
    R = zeros(T, nx, ny, nz)
    q1, q2, q3 = zeros(T, nx, ny, nz), zeros(T, nx, ny, nz), zeros(T, nx, ny, nz)
    μa, φa, dV = zeros(T, nx, ny, nz), zeros(T, nx, ny, nz), zeros(T, nx, ny, nz)
    bin = zeros(Int32, nx, ny, nz)
    invJ = Array(solver.inv_J)
    c1, c2, c3 = d.center
    # a node on a plane this patch shares with a same-level neighbor (abutting tiles
    # and slabs hold that plane twice) counts half in each copy
    shared(dd, side) = decomp.active[dd] && solver.bcs[dd][side] isa InterfaceBC &&
                       solver.bcs[dd][side].neighbor != 0 &&
                       (side == 1 ? decomp.sub_rank[dd] == 0 : decomp.sub_rank[dd] == decomp.sub_size[dd] - 1)
    sh = ntuple(dd -> (shared(dd, 1), shared(dd, 2)), 3)
    nl = (nx, ny, nz)
    m = solver.metric
    for k in 1:nz, j in 1:ny, i in 1:nx
        x1, x2, x3 = xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k)
        if m isa CartesianMetric
            X = (x1 - c1, x2 - c2, x3 - c3)
            Rr = sqrt(X[1]^2 + X[2]^2 + X[3]^2)
            q = X ./ max(Rr, floatmin(Float64))
        elseif m isa CylindricalMetric
            X = (x1 * cos(x2), x1 * sin(x2), x3 - c3)
            Rr = sqrt(x1^2 + X[3]^2)
            q = (x1, 0.0, X[3]) ./ max(Rr, floatmin(Float64))
        else
            X = (x1 * sin(x2) * cos(x3), x1 * sin(x2) * sin(x3), x1 * cos(x2))
            Rr = x1
            q = (1.0, 0.0, 0.0)
        end
        μ = polar ? clamp(X[3] / max(Rr, floatmin(Float64)), -1.0, 1.0) : 0.0
        φ = azim ? mod(atan(X[2], X[1]), 2π) : 0.0
        R[i, j, k] = Rr
        q1[i, j, k], q2[i, j, k], q3[i, j, k] = q
        μa[i, j, k], φa[i, j, k] = μ, φ
        # dV/R²: each bin then sums ∫ … dr per steradian, the per-line form's convention
        w = 1.0
        for (dd, idx) in ((1, i), (2, j), (3, k))
            (idx == 1 && sh[dd][1]) && (w /= 2)
            (idx == nl[dd] && sh[dd][2]) && (w /= 2)
        end
        dV[i, j, k] = w / (invJ[i+o1, j+o2, k+o3] * max(Rr^2, floatmin(Float64)))
        iμ = clamp(floor(Int, (μ + 1) / 2 * nμ) + 1, 1, nμ)
        iφ = clamp(floor(Int, φ / 2π * nφ) + 1, 1, nφ)
        bin[i, j, k] = iμ + (iφ - 1) * nμ
    end
    fold(dd, side) = decomp.active[dd] && solver.bcs[dd][side] isa _ABLATION_FOLDS &&
                     (side == 1 ? decomp.sub_rank[dd] == 0 : decomp.sub_rank[dd] == decomp.sub_size[dd] - 1)
    rules = (ntuple(dd -> fold(dd, 1), 3), ntuple(dd -> fold(dd, 2), 3))
    dev(A) = _on_device(solver.rho, A)
    return (; R = dev(R), q1 = dev(q1), q2 = dev(q2), q3 = dev(q3), μ = dev(μa),
            φ = dev(φa), dV = dev(dV), bin, nμ, nφ, polar, azim, rules,
            v1 = dev(zeros(T, nx, ny, nz)), v2 = dev(zeros(T, nx, ny, nz)))
end

function _bin_geometry(d::AblationDrive, solver)
    p = _patch_of(solver)
    return lock(d.binned.lock) do
        get!(() -> _build_bin_geometry(d, solver), d.binned.geometry, p)
    end
end

# ∂f/∂ξ_d of the partial density at padded `I` (local index `idx` of `n`), central
# inside the block, mirrored at a fold end (`fold`), one-sided at any other block end.
@inline function _ablation_dξ(Q, s, I, d, idx, n, fold_lo, fold_hi, h)
    @inbounds begin
        e = CartesianIndex(d == 1 ? 1 : 0, d == 2 ? 1 : 0, d == 3 ? 1 : 0)
        f0 = Q[I, s]
        n == 1 && return zero(f0)
        idx == 1 && return (Q[I+e, s] - f0) / (fold_lo ? 2h : h)
        idx == n && return (f0 - Q[I-e, s]) / (fold_hi ? 2h : h)
        return (Q[I+e, s] - Q[I-e, s]) / (2h)
    end
end

@inline function _ablation_bin_marker_point!(v1, v2, Q, s, o1, o2, o3, n1, n2, n3, a1, a2,
                                             a3, lo, hi, h1, h2, h3, ih1, ih2, ih3, R,
                                             q1, q2, q3, dV, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = Q[I, s]
        dρ = zero(ρ)
        a1 && (dρ += q1[i, j, k] * ih1[I] * _ablation_dξ(Q, s, I, 1, i, n1, lo[1], hi[1], h1))
        a2 && (dρ += q2[i, j, k] * ih2[I] * _ablation_dξ(Q, s, I, 2, j, n2, lo[2], hi[2], h2))
        a3 && (dρ += q3[i, j, k] * ih3[I] * _ablation_dξ(Q, s, I, 3, k, n3, lo[3], hi[3], h3))
        g = max(ρ, zero(ρ)) * max(-dρ, zero(dρ)) * dV[i, j, k]
        v1[i, j, k] = g
        v2[i, j, k] = g * R[i, j, k]
    end
    return nothing
end

# Bilinear interpolation between bin centres (μ clamped at the ends, φ periodic) over
# the bins holding a positive value; zero where none of the four does.
@inline function _bin_interp(B, nμ, nφ, μ, φ)
    @inbounds begin
        tμ = (μ + 1) / 2 * nμ - oftype(μ, 0.5)
        i0 = floor(Int, tμ)
        fμ = tμ - i0
        ia, ib = clamp(i0 + 1, 1, nμ), clamp(i0 + 2, 1, nμ)
        tφ = φ / (2 * oftype(φ, π)) * nφ - oftype(φ, 0.5)
        j0 = floor(Int, tφ)
        fφ = tφ - j0
        ja, jb = mod(j0, nφ) + 1, mod(j0 + 1, nφ) + 1
        s = zero(μ)
        w = zero(μ)
        for (ii, wi) in ((ia, 1 - fμ), (ib, fμ)), (jj, wj) in ((ja, 1 - fφ), (jb, fφ))
            v = B[ii+(jj-1)*nμ]
            if v > 0
                s += wi * wj * v
                w += wi * wj
            end
        end
        return w > 0 ? s / w : zero(μ)
    end
end

@inline function _ablation_bin_layer_point!(v1, v2, Q, s, o1, o2, o3, R, μ, φ, dV, RfB,
                                            nμ, nφ, inv_δ, outside, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        Rf = _bin_interp(RfB, nμ, nφ, μ[i, j, k], φ[i, j, k])
        ρ = Q[I, s]
        if Rf > 0
            x = (R[i, j, k] - Rf) * inv_δ
            e = exp(-x * x)
            K = outside && x > 0 ? one(x) : e
            v1[i, j, k] = max(ρ, zero(ρ)) * K * dV[i, j, k]
            v2[i, j, k] = e * inv_δ / sqrt(oftype(x, π)) * dV[i, j, k]
        else
            v1[i, j, k] = zero(ρ)
            v2[i, j, k] = zero(ρ)
        end
    end
    return nothing
end

# Σ v over each bin, on the host.
function _bin_sums!(S1, S2, bin, v1, v2)
    a1 = v1 isa Array ? v1 : Array(v1)
    a2 = v2 isa Array ? v2 : Array(v2)
    @inbounds for q in eachindex(bin)
        b = bin[q]
        S1[b] += a1[q]
        S2[b] += a2[q]
    end
    return nothing
end

function _bin_reduce!(S1, S2, comm)
    comm === nothing && return nothing
    MPI.Comm_size(comm) > 1 || return nothing
    buf = vcat(S1, S2)
    MPI.Allreduce!(buf, +, comm)
    n = length(S1)
    S1 .= view(buf, 1:n)
    S2 .= view(buf, n+1:2n)
    return nothing
end

# The front R_f and the normalization N = A/K of every bin, from the patches `items`
# ((solver, Q, geometry) on this rank), reduced over `comm`.
function _ablation_bins(d::AblationDrive, items, nb::Int, ::Type{T}, comm, t) where {T}
    S1, S2 = zeros(T, nb), zeros(T, nb)
    for (ps, Q, G) in items
        decomp = ps.decomp
        o1, o2, o3 = decomp.n_halo_d
        nx, ny, nz = decomp.n_local
        a = decomp.active
        h = map(T, ps.h)
        ih = ps.inv_h
        pointwise!(_ablation_bin_marker_point!, G.v1, nx, ny, nz, G.v1, G.v2, Q, d.species,
                   o1, o2, o3, nx, ny, nz, a[1], a[2], a[3], G.rules[1], G.rules[2],
                   h[1], h[2], h[3], ih[1], ih[2], ih[3], G.R, G.q1, G.q2, G.q3, G.dV)
        _bin_sums!(S1, S2, G.bin, G.v1, G.v2)
    end
    _bin_reduce!(S1, S2, comm)
    Rf = [S1[b] > 0 ? S2[b] / S1[b] : zero(T) for b in 1:nb]
    fill!(S1, zero(T))
    fill!(S2, zero(T))
    for (ps, Q, G) in items
        decomp = ps.decomp
        o1, o2, o3 = decomp.n_halo_d
        nx, ny, nz = decomp.n_local
        RfD = _on_device(ps.rho, Rf)
        pointwise!(_ablation_bin_layer_point!, G.v1, nx, ny, nz, G.v1, G.v2, Q, d.species,
                   o1, o2, o3, G.R, G.μ, G.φ, G.dV, RfD, G.nμ, G.nφ, T(1 / _layer_width(d, ps)),
                   d.driver > 0)
        _bin_sums!(S1, S2, G.bin, G.v1, G.v2)
    end
    _bin_reduce!(S1, S2, comm)
    N = [S1[b] > 0 && S2[b] > 0 ? S2[b] / S1[b] : zero(T) for b in 1:nb]
    return (; t, Rf, N)
end

# The incident flux factor at a direction: 1 (polar angle unresolved), the Legendre
# sum (azimuth unresolved or an axisymmetric drive), or every m.
@inline _bin_factor(fmode, rmodes, hc, μ, φ) =
    fmode == 0 ? one(μ) : fmode == 1 ? _legendre_sum(rmodes, μ) : _harmonic_sum(hc, μ, φ)

@inline function _ablation_bin_driver_point!(dQ, Q, p, s, drv, n_species, ie, o1, o2, o3,
                                             P, mdot, fmode, rmodes, hc, aP, am, gm1τ, inv_τ,
                                             R, μ, φ, RfB, NB, nμ, nφ, inv_δ, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        μc, φc = μ[i, j, k], φ[i, j, k]
        Rf = _bin_interp(RfB, nμ, nφ, μc, φc)
        Nc = _bin_interp(NB, nμ, nφ, μc, φc)
        f = max(_bin_factor(fmode, rmodes, hc, μc, φc), zero(P))
        Pl, ml = P * f^aP, mdot * f^am
        x = (R[i, j, k] - Rf) * inv_δ
        ρa = Q[I, s]
        if Rf > 0 && Nc > 0 && ρa > 0
            # at most a cell's ablator per relaxation time: a bin holding a cell or two
            # of a coarse level has a noisy N, which must not empty a cell in one step
            sk = min(ml * ρa * (x > 0 ? one(x) : exp(-x * x)) * Nc, ρa * inv_τ)
            dQ[I, s] -= sk
            dQ[I, drv] += sk
        end
        ρ = zero(eltype(Q))
        for sp in 1:n_species
            ρ += Q[I, sp]
        end
        Yd = max(Q[I, drv] / ρ, zero(ρ))
        S = x > 0 || !(Rf > 0) ? one(x) : exp(-x * x)
        dQ[I, ie] += S * Yd * Yd * (Pl - p[I]) * gm1τ
    end
    return nothing
end

@inline function _ablation_bin_force_point!(dQ, Q, s, n_species, m1, m2, m3, ie, o1, o2, o3,
                                            P, mdot, fmode, rmodes, hc, aP, am, inv_τ, R, q1, q2,
                                            q3, μ, φ, RfB, NB, nμ, nφ, inv_δ, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        μc, φc = μ[i, j, k], φ[i, j, k]
        Rf = _bin_interp(RfB, nμ, nφ, μc, φc)
        Nc = _bin_interp(NB, nμ, nφ, μc, φc)
        ρa = Q[I, s]
        (Rf > 0 && Nc > 0 && ρa > 0) || return nothing
        f = max(_bin_factor(fmode, rmodes, hc, μc, φc), zero(P))
        Pl, ml = P * f^aP, mdot * f^am
        x = (R[i, j, k] - Rf) * inv_δ
        w = ρa * exp(-x * x) * Nc
        ρ = zero(eltype(Q))
        for sp in 1:n_species
            ρ += Q[I, sp]
        end
        ir = one(ρ) / ρ
        u1, u2, u3 = Q[I, m1] * ir, Q[I, m2] * ir, Q[I, m3] * ir
        fr = -Pl * w
        sk = min(ml * w, ρa * inv_τ)
        e1, e2, e3 = q1[i, j, k], q2[i, j, k], q3[i, j, k]
        dQ[I, s] -= sk
        dQ[I, m1] += fr * e1 - sk * u1
        dQ[I, m2] += fr * e2 - sk * u2
        dQ[I, m3] += fr * e3 - sk * u3
        dQ[I, ie] += fr * (e1 * u1 + e2 * u2 + e3 * u3) - sk * Q[I, ie] * ir
    end
    return nothing
end

_patch_comm(solver) = solver.decomp.comm

# The layer width on `solver`'s patch: `layer`, or `min_layer` of its largest length
# spacing where that is wider.
function _layer_width(d::AblationDrive, solver)
    d.min_layer > 0 || return d.layer
    a = solver.decomp.active
    m = solver.metric
    hmax = 0.0
    for dd in 1:3
        a[dd] || continue
        (m isa CartesianMetric || (m isa CylindricalMetric && dd != 2) ||
         (m isa SphericalMetric && dd == 1)) || continue
        hmax = max(hmax, Float64(solver.h[dd]))
    end
    return max(d.layer, d.min_layer * hmax)
end

function _add_binned_source!(d::AblationDrive{L}, solver, dQ, Q, t) where {L}
    eq = solver.equations
    s = d.species
    s <= eq.n_species || throw(ArgumentError("AblationDrive: species $s of $(eq.n_species)"))
    T = eltype(Q)
    G = _bin_geometry(d, solver)
    p = _patch_of(solver)
    prep = get(d.binned.prepared, getfield(p, :id), nothing)
    if prep === nothing || prep.t != t || eltype(prep.Rf) != T
        prep = _ablation_bins(d, ((solver, Q, G),), G.nμ * G.nφ, T, _patch_comm(solver), t)
        solver isa Solver && (d.binned.front[] = (; prep.Rf, G.nμ, G.nφ))
    end
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    τ = Float64(t) / d.units.time
    P = T(ablation_pressure(d, τ) * d.units.pressure)
    mdot = T(ablation_rate(d, τ) * d.units.mass_flux / d.units.time)
    fmode = !G.polar ? 0 : (G.azim && !isempty(d.harmonics)) ? 2 : L > 0 ? 1 : 0
    rmodes = map(T, drive_modes(d, τ))
    hc = map(T, drive_harmonics(d, τ))
    aP, am = T(d.pressure_exponent / 4), T(d.rate_exponent / 4)
    RfB, NB = _on_device(solver.rho, prep.Rf), _on_device(solver.rho, prep.N)
    if d.driver > 0
        d.driver <= eq.n_species ||
            throw(ArgumentError("AblationDrive: driver species $(d.driver) of $(eq.n_species)"))
        pointwise!(_ablation_bin_driver_point!, dQ, nx, ny, nz, dQ, Q, solver.p, s, d.driver,
                   eq.n_species, eq.i_energy, o1, o2, o3, P, mdot, fmode, rmodes, hc, aP, am,
                   T(1 / ((d.driver_gamma - 1) * d.relaxation)), T(1 / d.relaxation), G.R, G.μ, G.φ, RfB, NB,
                   G.nμ, G.nφ, T(1 / _layer_width(d, solver)))
        return dQ
    end
    m1, m2, m3 = eq.i_mom
    pointwise!(_ablation_bin_force_point!, dQ, nx, ny, nz, dQ, Q, s, eq.n_species, m1, m2,
               m3, eq.i_energy, o1, o2, o3, P, mdot, fmode, rmodes, hc, aP, am,
               T(1 / d.relaxation), G.R, G.q1,
               G.q2, G.q3, G.μ, G.φ, RfB, NB, G.nμ, G.nφ, T(1 / _layer_width(d, solver)))
    return dQ
end

# The level's bins before its right-hand side (see `prepare_source!`): every patch of
# `lev` on this rank, reduced over `comm` (every rank of the level enters).
function prepare_source!(d::AblationDrive, solver, lev, states, comm)
    _binned(d, solver) || return nothing
    patches = getfield(solver, :patches)
    T = isempty(states) ? Float64 : eltype(first(states))
    nμ, nφ, _, _ = _bin_layout(d, solver.metric, map(>(1), solver.n_global))
    items = [(ps, states[pi], _bin_geometry(d, ps))
             for (pi, ps) in ((pi, PatchSolver(solver, patches[pi])) for pi in lev.patches)]
    prep = _ablation_bins(d, items, nμ * nφ, T, comm, solver.tstage)
    lev.index == 0 && (d.binned.front[] = (; prep.Rf, nμ, nφ))
    lock(d.binned.lock) do
        for pi in lev.patches
            d.binned.prepared[getfield(patches[pi], :id)] = prep
        end
    end
    return nothing
end

function release_source!(d::AblationDrive, solver, lev)
    isempty(d.binned.prepared) && return nothing
    patches = getfield(solver, :patches)
    lock(d.binned.lock) do
        for pi in lev.patches
            delete!(d.binned.prepared, getfield(patches[pi], :id))
        end
    end
    return nothing
end

"""
    ablation_band(drive; inside, outside, initial, metric = CartesianMetric()) -> predicate

An `AMR(initial = ...)` predicate `(x, y, z, t) -> Bool` for the binned
[`AblationDrive`](@ref): the points from `inside` below to `outside` beyond the
ablation front `R_f(θ, φ)` that `drive` found on the root level at its last
evaluation, the coordinates `(x, y, z)` those of `metric` (`(r, θ, z)` for
`CylindricalMetric`) about the drive's `center`. Before the first evaluation (at
setup) the front is the sphere of radius `initial`. Under `AMR(regrid_interval = K,
tile = …)` the band follows the front inward as the shell converges; give `inside`
the shell's thickness and its travel over `K` steps, and `outside` the layer and the
beyond-front ablator (several `layer`s). `R_f` is the root's, so it lags the
fine level's front by the root's resolution.
"""
function ablation_band(d::AblationDrive; inside::Real, outside::Real, initial::Real,
                       metric = CartesianMetric())
    c1, c2, c3 = d.center
    return function (x, y, z, t)
        if metric isa CylindricalMetric
            X = (x * cos(y), x * sin(y), z - c3)
        elseif metric isa SphericalMetric
            X = (x * sin(y) * cos(z), x * sin(y) * sin(z), x * cos(y))
        else
            X = (x - c1, y - c2, z - c3)
        end
        R = sqrt(X[1]^2 + X[2]^2 + X[3]^2)
        f = d.binned.front[]
        Rf = if f === nothing
            Float64(initial)
        else
            μ = clamp(X[3] / max(R, floatmin(Float64)), -1.0, 1.0)
            φ = mod(atan(X[2], X[1]), 2π)
            v = _bin_interp(f.Rf, f.nμ, f.nφ, f.nμ == 1 ? 0.0 : μ, f.nφ == 1 ? 0.0 : φ)
            v > 0 ? v : Float64(initial)
        end
        return Rf - inside < R < Rf + outside
    end
end
