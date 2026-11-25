# Stage-0 x-ray ablation drive for capsule-only implosions (the Briareus coupling,
# docs/src/explanation/ablation-drive.md): a rocket closure applied in the thin layer
# where the ablator's density falls off at its outer surface.

"""
    MM_NS_MG

Unit factors of the problem system with lengths in mm, times in ns and masses in mg
(density g/cm³, velocity 1000 km/s, pressure 10⁴ Mbar, energy MJ): the code value of
1 Mbar (`pressure`), of 1 g cm⁻² s⁻¹ (`mass_flux`) and of 1 ns (`time`). The default
`units` of [`AblationDrive`](@ref), and the units of a Briareus `CapsuleDrive` (mm, ns).
"""
const MM_NS_MG = (pressure = 1e-4, mass_flux = 1e-8, time = 1.0)

# The binned form's state: per patch (by `Patch` identity) its static geometry, and,
# between a level's `prepare_source!` and its release, the level's bins by patch id.
struct _BinnedState
    geometry::WeakKeyDict{Any,Any}
    prepared::Dict{Int,Any}
    lock::ReentrantLock
    front::Base.RefValue{Any}         # the root level's last bins (`ablation_band`)
end
_BinnedState() = _BinnedState(WeakKeyDict{Any,Any}(), Dict{Int,Any}(), ReentrantLock(),
                              Ref{Any}(nothing))

"""
    AblationDrive(time, Tr; modes = nothing, harmonics = nothing, species = 1,
                  driver = 0, driver_gamma = 5/3, relaxation = 0.01, pressure = 4.0,
                  pressure_exponent = 3.3, time_exponent = -0.09, rate = 3e5,
                  rate_exponent = 3.0, layer = 0.005, t_on = 0.0, t_ref = 1.0,
                  t_min = 0.1, units = MM_NS_MG, center = (0, 0, 0),
                  geometry = :auto, bins = nothing, min_layer = 0)

Stage-0 x-ray drive of a capsule-only implosion from a radiation-temperature history
`Tr` (eV) on the times `time` (ns): an ablation pressure and a mass ablation rate,
applied as an explicit source (`Problem(sources = (drive,))`) in the layer where the
ablator's partial density (species `species`) falls off at its outer surface.

**Closure** (rocket model, hydro_coupling.md §1). With `T` in heV,

    P_abl(t) = pressure · T^pressure_exponent · (t̂ / t_ref)^time_exponent   [Mbar]
    ṁ(t)     = rate · T^rate_exponent                                       [g cm⁻² s⁻¹]

with `t̂ = max(t − t_on, t_min)`. The defaults are carbon's subsonic exponents (3.3,
−0.09; Trickey, Walsh & Pasley, HEDP 44, 100995 (2022), Table 1) and Lindl's mass
ablation rate 3×10⁵ T³ (Lindl et al., PoP 11, 339 (2004), Eq. 6-3). The prefactor
`pressure` is not tabulated anywhere; calibrate it with [`ablation_pressure_scale`](@ref)
(4.0 Mbar gives ≈ 15 Mbar at 150 eV after 1 ns and ≈ 140 Mbar at 300 eV after 7 ns,
HYBRID-E's first shock > 12 Mbar and peak ≈ 140–150 Mbar; Kritcher et al., PoP 28
(2021); Lindl et al., PoP 33, 050503 (2026)).

**Angle.** `modes` is an `L × nt` matrix of Legendre ratios `aₗ/a₀`, `l = 1…L`, of the
incident flux about the +z axis (P2 > 0 pole-hot). On a spherical `(r, θ)` grid the
local temperature is `T_eff⁴ = Tr⁴ (1 + Σₗ (aₗ/a₀) Pₗ(cos θ))`, so `P_abl ∝
(1 + Σ…)^(pressure_exponent/4)` and `ṁ ∝ (1 + Σ…)^(rate_exponent/4)`. A collapsed θ
dimension (1-D spherical) takes the angle average, `a₀` alone.

`harmonics` is the drive with every `m`: an `(L+1)² × nt` matrix of ratios `bₗₘ`
of real Schmidt semi-normalized harmonics (rows `l² + l + m + 1`, row 1 the
constant 1; Briareus's `CapsuleDrive.ylm`), `T_eff⁴ = Tr⁴ Σₗₘ bₗₘ Ŷₗₘ(θ, φ)`,
`Ŷₗ₀ = Pₗ(cos θ)`, `Ŷₗ,±ₘ = √(2(l−m)!/(l+m)!) Pₗᵐ(cos θ) (cos mφ, sin mφ)` without
the Condon–Shortley phase, `φ` from +x toward +y. Its `m = 0` rows are the Legendre
`modes` (`bₗ₀ = aₗ/a₀`), which it replaces. A grid without a resolved azimuth (r–θ,
r–z) takes the azimuthal average, `m = 0`; a 3-D grid takes every `m`.

**Geometry.** On a `SphericalMetric` (r, θ) grid (`geometry = :auto` or `:lines`)
the front is searched along each radial line, as below. On a `CylindricalMetric`
r–z grid (axis `AxisBC`, θ collapsed or resolved) and a 3-D `CartesianMetric` grid,
or with `geometry = :binned` on any of the three, the radial direction is measured
from the capsule `center` (on the axis for cylindrical, the origin for spherical),
and the sums run over angular bins instead of lines: `bins = (nμ, nφ)` equal
intervals of `μ = cos θ` and of `φ` (default 32 × 64 with a resolved azimuth,
64 × 1 without, `nμ = 1` without a resolved polar angle). A bin should span several
cells of every level at the front: a 3-D bin holding one cell of a 60 µm coarse level
made that level's normalization noisy enough to fail the run, and the conversion is
capped at `ρₐ/relaxation` per cell for the same reason. Per bin the front is
`R_f = Σ g R dV / Σ g dV`, `g = ρₐ (−∂ρₐ/∂R)⁺`, and the layer weight is
normalized by the front area the bin's cells hold, `A = Σ e^(−x²)/(√π δ) dV`
(`x = (R − R_f)/δ`), so `w = k A / Σ k dV` carries `P` and `ṁ` per unit area of the
front: the per-line `∫ w dr = 1`, up to `R²` across the layer. `R_f` and `A/Σ k dV`
are interpolated bilinearly between bin centres at each cell's direction, and the
drive factor is evaluated there. With refinement (nested or tiled levels, static
or regridded, subcycled or not) the bins of a level are summed over all its patches
and ranks before the level's right-hand side, at every stage
(`prepare_source!`), so a tile holding part of the layer gets the level's front and
normalization; each level, the coarse one under a fine band included, carries its
own consistent source. A refined band must hold the whole layer where it covers the
front (the drive is then the band's; the coarse level's tail outside it is the
coarse solution's). On a level whose spacing does not resolve `layer`, the binned
form widens the layer to `min_layer` of its cells (the largest length spacing; `0`
by default, so nothing changes): a coarse level under a refined band, whose cells
are overwritten by the band's, then carries a smooth source instead of a sub-cell one.

**Source.** Along each radial line, with `ρₐ` the ablator's partial density, the front
is the mean radius of the ablator's outer density gradient,

    R_f = ∫ r g dr / ∫ g dr,     g = ρₐ (−∂ρₐ/∂r)⁺,

and the drive acts on the ablator mass in a Gaussian layer of width `layer` about it,

    w(r) = ρₐ e^(−((r − R_f)/layer)²) / ∫ ρₐ e^(−((r − R_f)/layer)²) dr,   ∫ w dr = 1,

the sums exact on the grid. Two forms carry the pressure:

- **Driver gas** (`driver = k`, the recommended form): the ablated ablator mass
  (`ṁ w` per unit volume) becomes species `k` in place, with momentum and energy
  unchanged, and species `k`'s pressure relaxes to the local ablation pressure through
  the energy source `S Y_k² (P_abl − p)/((driver_gamma − 1) relaxation)`. The driver gas
  (the problem's outer gas, and the converted blowoff) is then a piston at `P_abl`
  on the shell, and the energy the source adds is the work the ablation plasma does.
  The driver gas sets the timestep through its sound speed. Two one-sided weights keep
  the rocket on the shell. The conversion weight is `ρₐ` itself beyond the front
  (`ρₐ e^(−x²)` inside it, `x = (r − R_f)/layer`), so ablator left outside the front
  is converted first; with the two-sided layer a partly converted layer was left
  behind by the accelerating shell (22 % of the ablator at 6 ns in the NIF run), and
  its own outer gradient then pulled `R_f` off the shell and stopped the ablation.
  The piston weight is `S = 1` beyond the front and `e^(−x²)` inside it, so driver gas
  converted inside the shell does not heat the shell toward `P_abl` behind the front.
- **Body force** (`driver = 0`): the inward force density `−P_abl w r̂` on the layer,
  with its work, and the ablated mass removed from the problem with its momentum and
  specific total energy. Cheaper, but on cold ideal-gas ablators the accelerated layer
  lags the drive and spikes (it failed at the NIF rise in the 1-D tests, see
  docs/src/explanation/ablation-drive.md).

Give `layer` (a length in problem units; 5 µm by default) several cells, and the driver
gas a density that keeps its temperature at `P_abl` moderate: at 1 mg/cc the
ideal-gas driver reaches keV temperatures against a 0.05 eV ablator, the contact goes
to negative internal energy in the foot and the run fails near 0.5 ns, while 10 mg/cc
runs through without a retry (the driver's mass stays ~1 % of the ablator's). Either way
the shell is a rocket driven at `P_abl` and losing `ṁ`. Nothing here is radiation
transport: no spectrum, no albedo, no preheat; shock timing and adiabat are not
predictions.

Requirements of the per-line form: `SphericalMetric`, an unstretched `(r, θ)` grid,
one patch (no refinement); its sums are reduced over the radial sub-communicator. The
binned form takes any decomposition (bins are reduced over the patch's, or the
level's, ranks), stretching and refinement; its bin sums are taken on the host (a
device run copies the per-point values over). Both need a problem in the `units`
system (by default mm, ns, mg, [`MM_NS_MG`](@ref)).
"""
struct AblationDrive{L}
    time::Vector{Float64}
    Tr::Vector{Float64}
    modes::Matrix{Float64}            # L × nt, l = 1…L
    species::Int
    driver::Int                       # 0: body force; else the driver-gas species
    driver_gamma::Float64
    relaxation::Float64
    pressure::Float64
    pressure_exponent::Float64
    time_exponent::Float64
    rate::Float64
    rate_exponent::Float64
    layer::Float64
    t_on::Float64
    t_ref::Float64
    t_min::Float64
    units::NamedTuple{(:pressure, :mass_flux, :time),NTuple{3,Float64}}
    scratch::Base.RefValue{Any}       # per-point weights and per-line sums
    harmonics::Matrix{Float64}        # (L+1)² × nt Schmidt ratios, or 0 × nt
    center::NTuple{3,Float64}
    geometry::Symbol                  # :auto, :lines or :binned
    bins::Union{Nothing,NTuple{2,Int}}
    min_layer::Float64
    binned::_BinnedState              # per-patch geometry, the levels' bins
end

function AblationDrive(time::AbstractVector, Tr::AbstractVector; modes = nothing,
                       harmonics = nothing, species::Integer = 1, driver::Integer = 0, driver_gamma::Real = 5 / 3,
                       relaxation::Real = 0.01, pressure::Real = 4.0,
                       pressure_exponent::Real = 3.3, time_exponent::Real = -0.09,
                       rate::Real = 3e5, rate_exponent::Real = 3.0, layer::Real = 0.005, t_on::Real = 0.0,
                       t_ref::Real = 1.0, t_min::Real = 0.1, units = MM_NS_MG,
                       center = (0.0, 0.0, 0.0), geometry::Symbol = :auto, bins = nothing,
                       min_layer::Real = 0)
    nt = length(time)
    nt >= 1 && length(Tr) == nt ||
        throw(ArgumentError("AblationDrive: time and Tr must be non-empty and of one length"))
    issorted(time) || throw(ArgumentError("AblationDrive: times must be sorted"))
    all(>=(0), Tr) || throw(ArgumentError("AblationDrive: Tr must be non-negative"))
    M = modes === nothing ? zeros(0, nt) : Matrix{Float64}(modes)
    size(M, 2) == nt ||
        throw(ArgumentError("AblationDrive: modes must have one column per time (l = 1…L rows)"))
    H = zeros(0, nt)
    if harmonics !== nothing
        H = Matrix{Float64}(harmonics)
        Lh = isqrt(size(H, 1)) - 1
        (size(H, 1) == (Lh + 1)^2 && size(H, 2) == nt) ||
            throw(ArgumentError("AblationDrive: harmonics must be (L+1)² × nt (rows l² + l + m + 1)"))
        Mh = Lh == 0 ? zeros(0, nt) : H[[l^2 + l + 1 for l in 1:Lh], :]
        modes === nothing || M == Mh ||
            throw(ArgumentError("AblationDrive: the m = 0 rows of harmonics must equal modes"))
        M = Mh
    end
    geometry in (:auto, :lines, :binned) ||
        throw(ArgumentError("AblationDrive: geometry must be :auto, :lines or :binned"))
    bins === nothing || (length(bins) == 2 && all(>=(1), bins)) ||
        throw(ArgumentError("AblationDrive: bins must be (nμ, nφ), each at least 1"))
    length(center) == 3 || throw(ArgumentError("AblationDrive: center must have three coordinates"))
    species >= 1 || throw(ArgumentError("AblationDrive: species must be a species index"))
    driver >= 0 && driver != species ||
        throw(ArgumentError("AblationDrive: driver must be 0 or a species other than the ablator"))
    relaxation > 0 && driver_gamma > 1 ||
        throw(ArgumentError("AblationDrive: relaxation must be positive and driver_gamma > 1"))
    layer > 0 || throw(ArgumentError("AblationDrive: layer must be positive"))
    t_ref > 0 && t_min > 0 || throw(ArgumentError("AblationDrive: t_ref and t_min must be positive"))
    u = (pressure = Float64(units.pressure), mass_flux = Float64(units.mass_flux),
         time = Float64(units.time))
    return AblationDrive{size(M, 1)}(collect(Float64, time), collect(Float64, Tr), M,
                                     Int(species), Int(driver), driver_gamma, relaxation,
                                     pressure, pressure_exponent,
                                     time_exponent, rate, rate_exponent, layer, t_on, t_ref,
                                     t_min, u, Ref{Any}(nothing), H,
                                     ntuple(i -> Float64(center[i]), 3), geometry,
                                     bins === nothing ? nothing : (Int(bins[1]), Int(bins[2])),
                                     Float64(min_layer), _BinnedState())
end

function Base.show(io::IO, d::AblationDrive{L}) where {L}
    print(io, "AblationDrive(", length(d.time), " times on [", first(d.time), ", ",
          last(d.time), "] ns, peak Tr ", round(maximum(d.Tr); digits=1), " eV, l ≤ ", L,
          isempty(d.harmonics) ? "" : " (all m)", ", P = ", d.pressure, " T^", d.pressure_exponent, " t^", d.time_exponent,
          " Mbar, ṁ = ", d.rate, " T^", d.rate_exponent, ")")
end

# Linear interpolation, held at the end values.
function _table_interp(ts, vs, t)
    t <= ts[1] && return vs[1]
    t >= ts[end] && return vs[end]
    k = searchsortedlast(ts, t)
    w = (t - ts[k]) / (ts[k+1] - ts[k])
    return (1 - w) * vs[k] + w * vs[k+1]
end

"Radiation temperature (eV) of the drive at time `t` (ns)."
drive_temperature(d::AblationDrive, t::Real) = _table_interp(d.time, d.Tr, Float64(t))

_time_factor(d::AblationDrive, t) =
    (max(Float64(t) - d.t_on, d.t_min) / d.t_ref)^d.time_exponent

"""
    ablation_pressure(drive, t) -> Mbar

The angle-averaged ablation pressure of `drive` at time `t` (ns).
"""
ablation_pressure(d::AblationDrive, t::Real) =
    d.pressure * (drive_temperature(d, t) / 100)^d.pressure_exponent * _time_factor(d, t)

"""
    ablation_rate(drive, t) -> g cm⁻² s⁻¹

The angle-averaged mass ablation rate of `drive` at time `t` (ns).
"""
ablation_rate(d::AblationDrive, t::Real) =
    d.rate * (drive_temperature(d, t) / 100)^d.rate_exponent

"""
    drive_harmonics(drive, t) -> NTuple{(L+1)²}

The Schmidt ratios `bₗₘ` of `drive` at time `t` (row order `l² + l + m + 1`, the
first the constant 1); the Legendre ratios at `m = 0` and zeros elsewhere for an
axisymmetric drive.
"""
function drive_harmonics(d::AblationDrive{L}, t::Real) where {L}
    τ = Float64(t)
    a = drive_modes(d, τ)
    h = isempty(d.harmonics)
    return ntuple(Val((L + 1)^2)) do q
        q == 1 && return 1.0
        h || return _table_interp(d.time, view(d.harmonics, q, :), τ)
        l = isqrt(q - 1)
        q == l^2 + l + 1 ? a[l] : 0.0
    end
end

"Legendre ratios `aₗ/a₀`, `l = 1…L`, of `drive` at time `t`, as an `NTuple{L}`."
drive_modes(d::AblationDrive{L}, t::Real) where {L} =
    ntuple(l -> _table_interp(d.time, view(d.modes, l, :), Float64(t)), Val(L))

"""
    drive_factor(drive, μ, t) -> 1 + Σₗ (aₗ/a₀) Pₗ(μ)

The incident flux at `μ = cos θ` relative to its angle average.
"""
drive_factor(d::AblationDrive, μ::Real, t::Real) = _legendre_sum(drive_modes(d, t), μ)

"""
    drive_factor(drive, μ, φ, t) -> Σₗₘ bₗₘ Ŷₗₘ(μ, φ)

The incident flux in the direction `(μ = cos θ, φ)` relative to its angle average,
with every `m` of `harmonics` (the Legendre sum for an axisymmetric drive).
"""
drive_factor(d::AblationDrive, μ::Real, φ::Real, t::Real) =
    _harmonic_sum(drive_harmonics(d, t), μ, φ)

# Σ c_q Ŷ_q(μ, φ) over real Schmidt semi-normalized harmonics (q = l² + l + m + 1, no
# Condon–Shortley phase), by the recursions S_m^m = √((2m−1)/2m) s S_{m−1}^{m−1}
# (S_1^1 = s) and S_l^m = ((2l−1) μ S_{l−1}^m − √((l−1)²−m²) S_{l−2}^m)/√(l²−m²).
@inline function _harmonic_sum(c::NTuple{N,T}, μ, φ) where {N,T}
    L = isqrt(N) - 1
    s = sqrt(max(1 - μ * μ, zero(μ)))
    tot = zero(μ)
    pmm = one(μ)
    @inbounds for m in 0:L
        m == 1 && (pmm = s)
        m > 1 && (pmm *= sqrt(T(2m - 1) / T(2m)) * s)
        cm, sm = cos(m * φ), sin(m * φ)
        p2, p1 = zero(μ), pmm
        for l in m:L
            if l > m
                p = (T(2l - 1) * μ * p1 - sqrt(T((l - 1)^2 - m^2)) * p2) / sqrt(T(l^2 - m^2))
                p2, p1 = p1, p
            end
            q = l^2 + l + 1
            tot += m == 0 ? c[q] * p1 : p1 * (c[q+m] * cm + c[q-m] * sm)
        end
    end
    return tot
end

@inline function _legendre_sum(r::NTuple{L,T}, μ) where {L,T}
    s = one(μ)
    p0, p1 = one(μ), μ
    @inbounds for l in 1:L
        s += r[l] * p1
        p0, p1 = p1, ((2l + 1) * μ * p1 - l * p0) / (l + 1)
    end
    return s
end

"""
    ablation_pressure_scale(time, Tr; peak = 145.0, pressure_exponent = 3.3,
                            time_exponent = -0.09, t_on = 0.0, t_ref = 1.0,
                            t_min = 0.1) -> Mbar

The prefactor `pressure` of [`AblationDrive`](@ref) whose largest pressure over the
drive `(time, Tr)` is `peak` Mbar (HYBRID-E: ≈ 140–150 Mbar at peak, hydro_coupling.md
§1.3). Check the foot against the first-shock requirement (> 12 Mbar at ≈ 150 eV)
with [`ablation_pressure`](@ref).
"""
function ablation_pressure_scale(time, Tr; peak::Real = 145.0, pressure_exponent::Real = 3.3,
                                 time_exponent::Real = -0.09, t_on::Real = 0.0,
                                 t_ref::Real = 1.0, t_min::Real = 0.1)
    d = AblationDrive(time, Tr; pressure = 1.0, pressure_exponent, time_exponent, t_on,
                      t_ref, t_min)
    return peak / maximum(ablation_pressure(d, τ) for τ in d.time)
end

# --- The source --------------------------------------------------------------

function _check_ablation_solver(solver)
    solver isa Solver ||
        throw(ArgumentError("AblationDrive supports a single-patch solver (no refinement or patches)"))
    solver.metric isa SphericalMetric ||
        throw(ArgumentError("AblationDrive needs SphericalMetric (r, θ, φ); got $(typeof(solver.metric))"))
    solver.stretch[1] === nothing && solver.stretch[2] === nothing ||
        throw(ArgumentError("AblationDrive needs an unstretched (r, θ) grid"))
    solver.decomp.active[1] || throw(ArgumentError("AblationDrive needs a resolved radius"))
    return nothing
end

# The front marker g = ρₐ (−∂ρₐ/∂r)⁺ by central differences on the padded state, and
# its first radial moment; zero on the first and last global radial points (the
# origin fold and the outer boundary).
@inline function _ablation_front_point!(g, gr, Q, s, o1, o2, o3, g1, nr, r0, h1, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        gi = g1 + i
        v = zero(eltype(g))
        if 1 < gi < nr
            ρ = Q[I, s]
            dρ = (Q[CartesianIndex(i + o1 + 1, j + o2, k + o3), s] -
                  Q[CartesianIndex(i + o1 - 1, j + o2, k + o3), s]) / (2h1)
            v = max(ρ, zero(ρ)) * max(-dρ, zero(dρ))
        end
        g[i, j, k] = v
        gr[i, j, k] = v * (r0 + (gi - 1) * h1)
    end
    return nothing
end

# The layer's mass weight k = ρₐ exp(−((r − R_f)/δ)²) about the line's front R_f, or,
# `outside`, one-sided: ρₐ exp(−x²) inside the front and ρₐ beyond it.
@inline function _ablation_layer_point!(g, Q, s, o1, o2, o3, g1, r0, h1, Rf, inv_δ,
                                        outside, i, j, k)
    @inbounds begin
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = Q[I, s]
        x = (r0 + (g1 + i - 1) * h1 - Rf[1, j, k]) * inv_δ
        K = outside && x > 0 ? one(x) : exp(-x * x)
        g[i, j, k] = Rf[1, j, k] > 0 ? max(ρ, zero(ρ)) * K : zero(ρ)
    end
    return nothing
end

@inline function _ablation_apply_point!(dQ, Q, g, G, s, n_species, m1, m2, m3, ie,
                                        o1, o2, o3, P, mdot, rmodes, angular, θ0, hθ,
                                        jθ, aP, am, i, j, k)
    @inbounds begin
        Gl = G[1, j, k]
        gi = g[i, j, k]
        (Gl > 0 && gi > 0) || return nothing
        w = gi / Gl
        if angular
            μ = cos(θ0 + (jθ + j - 1) * hθ)
            f = max(_legendre_sum(rmodes, μ), zero(μ))
            Pl = P * f^aP
            ml = mdot * f^am
        else
            Pl, ml = P, mdot
        end
        I = CartesianIndex(i + o1, j + o2, k + o3)
        ρ = zero(eltype(Q))
        for sp in 1:n_species
            ρ += Q[I, sp]
        end
        ir = one(ρ) / ρ
        u1, u2, u3 = Q[I, m1] * ir, Q[I, m2] * ir, Q[I, m3] * ir
        fr = -Pl * w                      # inward force per unit volume
        sk = ml * w                       # ablator mass removed per unit volume and time
        dQ[I, s] -= sk
        dQ[I, m1] += fr - sk * u1
        dQ[I, m2] -= sk * u2
        dQ[I, m3] -= sk * u3
        dQ[I, ie] += fr * u1 - sk * Q[I, ie] * ir
    end
    return nothing
end

@inline _drive_angle(P, mdot, rmodes, angular, θ0, hθ, jθ, aP, am, j) =
    if angular
        f = max(_legendre_sum(rmodes, cos(θ0 + (jθ + j - 1) * hθ)), zero(P))
        (P * f^aP, mdot * f^am)
    else
        (P, mdot)
    end

# Driver-gas form: the ablated ablator mass becomes driver gas in place (momentum and
# energy unchanged), and the driver gas's pressure relaxes to the local ablation
# pressure, an energy source Y_d² (P_abl − p)/((γ_d − 1) τ).
@inline function _ablation_driver_point!(dQ, Q, p, g, G, s, drv, n_species, ie, o1, o2,
                                         o3, P, mdot, rmodes, angular, θ0, hθ, jθ, aP, am,
                                         gm1τ, g1, r0, h1, Rf, inv_δ, i, j, k)
    @inbounds begin
        Pl, ml = _drive_angle(P, mdot, rmodes, angular, θ0, hθ, jθ, aP, am, j)
        I = CartesianIndex(i + o1, j + o2, k + o3)
        Gl = G[1, j, k]
        gi = g[i, j, k]
        if Gl > 0 && gi > 0
            sk = ml * gi / Gl
            dQ[I, s] -= sk
            dQ[I, drv] += sk
        end
        ρ = zero(eltype(Q))
        for sp in 1:n_species
            ρ += Q[I, sp]
        end
        Yd = max(Q[I, drv] / ρ, zero(ρ))
        # the piston acts at and beyond the front; driver gas inside the shell (where
        # the layer converts) relaxes with weight exp(−x²), so it does not heat the
        # shell toward P_abl behind the front
        x = (r0 + (g1 + i - 1) * h1 - Rf[1, j, k]) * inv_δ
        S = x > 0 || !(Rf[1, j, k] > 0) ? one(x) : exp(-x * x)
        dQ[I, ie] += S * Yd * Yd * (Pl - p[I]) * gm1τ
    end
    return nothing
end

# Per radial line, h Σᵢ a[i, j, k], summed over the line's ranks.
function _line_integral!(G, a, h, decomp)
    sum!(G, a)
    G .*= h
    if decomp.sub_size[1] > 1
        Gh = Array(G)
        MPI.Allreduce!(Gh, +, decomp.sub[1])
        copyto!(G, Gh)
    end
    return G
end

function add_source!(d::AblationDrive{L}, solver, dQ, Q, t) where {L}
    _binned(d, solver) && return _add_binned_source!(d, solver, dQ, Q, t)
    _check_ablation_solver(solver)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    eq = solver.equations
    s = d.species
    s <= eq.n_species || throw(ArgumentError("AblationDrive: species $s of $(eq.n_species)"))
    T = eltype(Q)
    sc = d.scratch[]
    # one scratch set per drive, rebuilt when the extent or element type changes (a
    # drive shared by two solvers of different size reallocates on every switch)
    if !(sc isa Tuple) || size(sc[1]) != (nx, ny, nz) || eltype(sc[1]) != T
        sc = (similar(solver.rho, T, nx, ny, nz), similar(solver.rho, T, nx, ny, nz),
              similar(solver.rho, T, 1, ny, nz), similar(solver.rho, T, 1, ny, nz))
        d.scratch[] = sc
    end
    g, gr, G, Rf = sc
    h1 = T(solver.h[1])
    r0 = T(solver.origin[1] + solver.coord_shift[1])
    g1 = solver.region.offset[1] + decomp.offset[1]
    # the front of each line: the g-weighted mean radius of the outer density gradient
    pointwise!(_ablation_front_point!, dQ, nx, ny, nz, g, gr, Q, s, o1, o2, o3, g1,
               solver.n_global[1], r0, h1)
    _line_integral!(G, g, h1, decomp)
    _line_integral!(Rf, gr, h1, decomp)
    Rf .= ifelse.(G .> 0, Rf ./ max.(G, floatmin(T)), zero(T))
    # the layer about it, weighted by the ablator mass
    # (driver-gas form: and all the ablator beyond the front, so that none is stranded)
    pointwise!(_ablation_layer_point!, dQ, nx, ny, nz, g, Q, s, o1, o2, o3, g1, r0, h1,
               Rf, T(1 / d.layer), d.driver > 0)
    _line_integral!(G, g, h1, decomp)
    τ = Float64(t) / d.units.time
    P = T(ablation_pressure(d, τ) * d.units.pressure)
    mdot = T(ablation_rate(d, τ) * d.units.mass_flux / d.units.time)
    angular = decomp.active[2] && L > 0
    rmodes = map(T, drive_modes(d, τ))
    θ0 = T(solver.origin[2] + solver.coord_shift[2])
    hθ = T(solver.h[2])
    jθ = solver.region.offset[2] + decomp.offset[2]
    aP, am = T(d.pressure_exponent / 4), T(d.rate_exponent / 4)
    if d.driver > 0
        d.driver <= eq.n_species ||
            throw(ArgumentError("AblationDrive: driver species $(d.driver) of $(eq.n_species)"))
        pointwise!(_ablation_driver_point!, dQ, nx, ny, nz, dQ, Q, solver.p, g, G, s,
                   d.driver, eq.n_species, eq.i_energy, o1, o2, o3, P, mdot, rmodes,
                   angular, θ0, hθ, jθ, aP, am,
                   T(1 / ((d.driver_gamma - 1) * d.relaxation)), g1, r0, h1, Rf,
                   T(1 / d.layer))
        return dQ
    end
    m1, m2, m3 = eq.i_mom
    pointwise!(_ablation_apply_point!, dQ, nx, ny, nz, dQ, Q, g, G, s, eq.n_species,
               m1, m2, m3, eq.i_energy, o1, o2, o3, P, mdot, rmodes, angular, θ0, hθ,
               jθ, aP, am)
    return dQ
end


"""
    radiation_drive(drive; kwargs...) -> AblationDrive

An [`AblationDrive`](@ref) from a hohlraum code's capsule drive. Methods come from
package extensions: with Briareus loaded (`using Briareus`), `radiation_drive(d::
Briareus.CapsuleDrive; lmax, pressure = :calibrate, peak, kwargs...)` takes the
capsule's `Tr(t)` and Legendre ratios `aₗ/a₀` (ext/CompactLESBriareusExt.jl).
"""
function radiation_drive end
