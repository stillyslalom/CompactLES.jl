# Pointwise thermodynamics of a `Prim`, and the gas-dynamic relations built on
# it: composition by name, normal-shock jumps, the shock-tube driver pressure
# and the reflected shock.
#
# These run once at setup, on scalars, so they are written for clarity and
# generality over the EOS rather than speed. Each EOS supplies four scalar
# methods below (`_point_temperature`, `_point_pressure`, `_point_energy`,
# `_point_cp`, and the sound speed) and every relation is written against those,
# so a calorically imperfect `Nasa9Mixture` gets the real Rankine–Hugoniot jump
# and the real isentrope rather than a constant-γ estimate. The constant-γ
# formulas appear only as starting guesses for the iterations.

# --- Scalar EOS layer ---------------------------------------------------------

_mixture_R(eos::Union{IdealMixture,Nasa9Mixture}, Y) =
    sum(Y[k] * eos.Rk[k] for k in eachindex(Y))

_point_temperature(eos::Union{IdealMixture,Nasa9Mixture}, ρ, p, Y) =
    p / (ρ * _mixture_R(eos, Y))
_point_pressure(eos::Union{IdealMixture,Nasa9Mixture}, ρ, T_ion, Y) =
    ρ * _mixture_R(eos, Y) * T_ion
_point_energy(eos::IdealMixture, ρ, T_ion, Y) =
    sum(Y[k] * eos.cvk[k] for k in eachindex(Y)) * T_ion
_point_energy(eos::Nasa9Mixture, ρ, T_ion, Y) =
    sum(Y[k] * species_energy(eos, k, T_ion) for k in eachindex(Y))
_point_cp(eos::IdealMixture, T_ion, Y) = sum(Y[k] * eos.cpk[k] for k in eachindex(Y))
_point_cp(eos::Nasa9Mixture, T_ion, Y) =
    sum(Y[k] * species_cp(eos, k, T_ion) for k in eachindex(Y))
function _point_sound_speed(eos::Union{IdealMixture,Nasa9Mixture}, ρ, p, Y)
    T_ion = _point_temperature(eos, ρ, p, Y)
    R = _mixture_R(eos, Y)
    cp = _point_cp(eos, T_ion, Y)
    return sqrt(cp / (cp - R) * R * T_ion)
end

# p + p∞ = ρ R T and ρe = ρ c_v T + p∞, as in `conserved_from_prim`.
_point_temperature(eos::StiffenedGas, ρ, p, Y) =
    (p + eos.p_inf) / (ρ * gas_constant(eos))
_point_pressure(eos::StiffenedGas, ρ, T_ion, Y) =
    ρ * gas_constant(eos) * T_ion - eos.p_inf
_point_energy(eos::StiffenedGas, ρ, T_ion, Y) = eos.cv * T_ion + eos.p_inf / ρ
_point_cp(eos::StiffenedGas, T_ion, Y) = eos.gamma * eos.cv
_point_sound_speed(eos::StiffenedGas, ρ, p, Y) = sqrt(eos.gamma * (p + eos.p_inf) / ρ)

_point_enthalpy(eos, ρ, p, Y) =
    _point_energy(eos, ρ, _point_temperature(eos, ρ, p, Y), Y) + p / ρ

_as_point_eos(eos::EOS) = eos
_as_point_eos(species::IdealSpecies) = IdealMixture(species)

function _check_species(eos, pr::Prim{N}) where {N}
    N == nspecies(eos) ||
        throw(ArgumentError("Prim carries $N mass fractions; the EOS has " *
                            "$(nspecies(eos)) species"))
    return nothing
end

# Density and pressure of a `Prim`, whichever two of (p, ρ, T) it was given.
function _density_pressure(eos, pr::Prim)
    _check_species(eos, pr)
    if isnan(pr.rho)
        # (p, T): invert p(ρ, T) for ρ. Every model here is linear in ρ at
        # fixed T apart from the stiffened offset, so two evaluations suffice.
        p1 = _point_pressure(eos, 1.0, pr.T_ion, pr.Y)
        p0 = _point_pressure(eos, 0.0, pr.T_ion, pr.Y)
        return (pr.p - p0) / (p1 - p0), pr.p
    elseif isnan(pr.p)
        return pr.rho, _point_pressure(eos, pr.rho, pr.T_ion, pr.Y)
    end
    return pr.rho, pr.p
end

"""
    thermodynamic_state(eos, pr::Prim) -> NamedTuple

Complete the thermodynamic state of a [`Prim`](@ref) through `eos`. The result
has fields `rho`, `p`, `T_ion`, `e` (specific internal energy), `h` (specific
enthalpy), `cp`, `gamma` (the frozen ratio `c² ρ / p` for an ideal gas, `cp/cv`
in general), `c` (frozen sound speed), `u`, `Y` and `X` (mole fractions).

Use it to size a run from its initial state (a sound speed for a crossing time,
a Mach number) and to check a hand-built state before handing it to a
[`Problem`](@ref). It evaluates the same EOS the solver integrates with, so the
numbers agree with what [`line_profile`](@ref) reports at the initial state.
"""
function thermodynamic_state(eos, pr::Prim)
    eos = _as_point_eos(eos)
    ρ, p = _density_pressure(eos, pr)
    Y = pr.Y
    T_ion = _point_temperature(eos, ρ, p, Y)
    e = _point_energy(eos, ρ, T_ion, Y)
    cp = _point_cp(eos, T_ion, Y)
    c = _point_sound_speed(eos, ρ, p, Y)
    cv = eos isa StiffenedGas ? eos.cv : cp - _mixture_R(eos, Y)
    return (rho=ρ, p=p, T_ion=T_ion, e=e, h=e + p / ρ, cp=cp, gamma=cp / cv, c=c,
            u=pr.u, Y=Y, X=mole_fractions(eos, Y))
end

# --- Composition by name ------------------------------------------------------

"""
    mass_fractions(eos, "He" => 0.95, "acetone" => 0.05; basis) -> NTuple

Mass fractions in the species order of `eos`, from fractions given by species
name. `basis = :mole` reads the numbers as mole (number) fractions and converts
them through the species molar masses; `basis = :mass` reads them as mass
fractions and only reorders them. The keyword is required, because the two
readings of the same numbers differ by the ratio of molar masses and neither is
a safe default.

Species not named are zero. The given fractions must sum to one. The names are
those of [`species_names`](@ref), for example `Nasa9Mixture(["He", "CO2"])`
has `"He"` and `"CO2"`.

```julia
eos = Nasa9Mixture(["He", "Ar", "SF6"])
Prim(Y = mass_fractions(eos, "He" => 0.95, "Ar" => 0.05; basis = :mole),
     p = 101_325.0, T_ion = 300.0)
```
"""
function mass_fractions(eos, pairs::Pair...; basis::Symbol)
    basis in (:mass, :mole) ||
        throw(ArgumentError("mass_fractions: basis must be :mass or :mole"))
    eos = _as_point_eos(eos)
    names = species_names(eos)
    n = length(names)
    f = zeros(n)
    for (name, value) in pairs
        k = findfirst(==(String(name)), names)
        k === nothing &&
            throw(ArgumentError("mass_fractions: no species \"$name\" in the EOS; " *
                                "its species are $(join(names, ", "))"))
        f[k] += Float64(value)
    end
    abs(sum(f) - 1) < 1e-10 ||
        throw(ArgumentError("mass_fractions: the fractions sum to $(sum(f)), not 1"))
    basis === :mass && return Tuple(f)
    # X_k ∝ Y_k / W_k and W_k ∝ 1/R_k, so Y_k ∝ X_k / R_k.
    R = _species_R(eos)
    w = f ./ R
    return Tuple(w ./ sum(w))
end

_species_R(eos::Union{IdealMixture,Nasa9Mixture}) = collect(Float64, eos.Rk)
_species_R(eos::StiffenedGas) = [Float64(gas_constant(eos))]

"""
    mole_fractions(eos, Y) -> NTuple

Mole (number) fractions from mass fractions `Y` in the species order of `eos`.
"""
function mole_fractions(eos, Y)
    R = _species_R(_as_point_eos(eos))
    w = ntuple(k -> Y[k] * R[k], length(Y))
    s = sum(w)
    return map(x -> x / s, w)
end

# --- Normal shocks ------------------------------------------------------------

"""
    shock_jump(eos, pre::Prim, Mach; dim = 1, direction = +1) -> NamedTuple

The state behind a normal shock of Mach number `Mach` travelling into `pre`,
from the Rankine–Hugoniot relations evaluated through `eos` (the real caloric
equation of a [`Nasa9Mixture`](@ref), not a constant-γ fit). The shock moves
along coordinate `dim`, towards increasing coordinate for `direction = +1` and
decreasing for `-1`; `Mach` is measured relative to the gas ahead of it, whose
velocity may be nonzero.

Returns `(post, shock_speed, velocity)`: `post` is the shocked state as a
[`Prim`](@ref) of pressure and temperature, with the composition of `pre`,
ready for [`Layers`](@ref), [`DirichletBC`](@ref) or [`NSCBCInflowBC`](@ref);
`shock_speed` the laboratory
shock velocity along `dim` (signed), and `velocity` the laboratory gas
velocity behind the shock along `dim` (signed).

```julia
eos = Nasa9Mixture(["N2", "SF6"])
air = Prim(Y = (1.0, 0.0), p = 101_325.0, T_ion = 300.0)
jump = shock_jump(eos, air, 1.5)
ic = Layers(air, Slab(1, hi = 0.1) => jump.post)
```
"""
function shock_jump(eos, pre::Prim, Mach::Real; dim::Int=1, direction::Int=1)
    Mach > 1 || throw(ArgumentError("shock_jump: Mach must exceed 1, got $Mach"))
    direction in (-1, 1) ||
        throw(ArgumentError("shock_jump: direction must be +1 or -1"))
    1 <= dim <= 3 || throw(ArgumentError("shock_jump: dim must be 1, 2 or 3"))
    eos = _as_point_eos(eos)
    s1 = thermodynamic_state(eos, pre)
    ρ1, p1, h1, Y = s1.rho, s1.p, s1.h, pre.Y
    W = Mach * s1.c                       # inflow speed in the shock frame
    # ε = ρ1/ρ2. Momentum and energy across the shock fix p2(ε) and h2(ε);
    # the EOS fixes h(ρ2, p2); the jump is the root of their difference other
    # than the trivial ε = 1.
    residual(ε) = begin
        p2 = p1 + ρ1 * W^2 * (1 - ε)
        h2 = h1 + W^2 * (1 - ε^2) / 2
        _point_enthalpy(eos, ρ1 / ε, p2, Y) - h2
    end
    γ = s1.gamma
    ε = ((γ - 1) * Mach^2 + 2) / ((γ + 1) * Mach^2)   # constant-γ start
    ε = _newton_root(residual, ε, "shock_jump")
    (0 < ε < 1) || error("shock_jump: no compressive solution found at Mach $Mach")
    p2 = p1 + ρ1 * W^2 * (1 - ε)
    ρ2 = ρ1 / ε
    Δu = direction * W * (1 - ε)
    u2 = ntuple(d -> d == dim ? pre.u[d] + Δu : pre.u[d], 3)
    # Pressure and temperature, the pair an inflow condition takes as its target.
    post = Prim(u=u2, p=p2, T_ion=_point_temperature(eos, ρ2, p2, Y), Y=pre.Y)
    return (post=post, shock_speed=pre.u[dim] + direction * W, velocity=u2[dim])
end

# A secant-safeguarded Newton iteration on a smooth scalar residual. The
# starting points here come from constant-γ closed forms, a few percent from the
# root, so no bracketing is needed; a failure to converge is reported rather
# than returned.
function _newton_root(f, x0, who; rtol=1e-13, maxiter=60)
    x = x0
    for _ in 1:maxiter
        fx = f(x)
        δ = 1e-7 * max(abs(x), 1e-12)
        dfdx = (f(x + δ) - f(x - δ)) / (2δ)
        dx = fx / dfdx
        isfinite(dx) || break
        x -= dx
        abs(dx) <= rtol * max(abs(x), 1e-300) && return x
    end
    error("$who: the iteration did not converge from $x0")
end

"""
    driver_pressure(eos, driver::Prim, driven::Prim, Mach) -> Prim

The driver state of a shock tube that sends a shock of Mach number `Mach` into
`driven`: `driver` with its pressure raised to the value the ideal (diaphragm
burst, no losses) shock-tube problem requires, at the temperature and
composition it was given with. The driver lies at lower coordinate and the
shock travels towards increasing coordinate along dimension 1.

The expansion is integrated along the real isentrope of `eos`, so a calorically
imperfect driver is handled without a constant-γ approximation. `driver` must
specify `T_ion`; its pressure is only a starting value.
"""
function driver_pressure(eos, driver::Prim, driven::Prim, Mach::Real)
    eos = _as_point_eos(eos)
    isnan(driver.T_ion) &&
        throw(ArgumentError("driver_pressure: the driver Prim must give T_ion"))
    _check_species(eos, driver)
    jump = shock_jump(eos, driven, Mach)
    p2, u2 = jump.post.p, jump.velocity
    # The contact is at rest in the driver's frame plus u2, so the unsteady
    # expansion must take driver gas from its own velocity to u2 while dropping
    # its pressure to p2: u3 = u4 + ∫_{p3}^{p4} dp / (ρ c).
    gain(p4) = _expansion(eos, Prim(p=p4, T_ion=driver.T_ion, Y=driver.Y),
                          p2).velocity + driver.u[1] - u2
    # Constant-γ starting guess from the classical shock-tube relation.
    s1 = thermodynamic_state(eos, driven)
    s4 = thermodynamic_state(eos, Prim(p=s1.p, T_ion=driver.T_ion, Y=driver.Y))
    γ1, γ4, a1, a4 = s1.gamma, s4.gamma, s1.c, s4.c
    base = 1 - (γ4 - 1) / (γ1 + 1) * (a1 / a4) * (Mach - 1 / Mach)
    guess = base > 0 ? p2 * (base)^(-2γ4 / (γ4 - 1)) : 10 * p2
    logp4 = _newton_root(q -> gain(exp(q)), log(guess), "driver_pressure")
    return Prim(u=driver.u, p=exp(logp4), T_ion=driver.T_ion, Y=driver.Y)
end

# ∫_{p_end}^{p_start} dp / (ρ c) along the isentrope through `start`, integrated
# in log p with the state carried as (ρ, p): at constant entropy dρ = dp / c².
# Returns that velocity gain and the density at `p_end`.
function _expansion(eos, start::Prim, p_end; steps=400)
    ρ, p = _density_pressure(eos, start)
    Y = start.Y
    q0, q1 = log(p), log(p_end)
    dq = (q1 - q0) / steps
    u = 0.0
    # RK4 on y = (ρ, u) with independent variable q = log p, dp = p dq.
    rhs(q, ρ) = begin
        pq = exp(q)
        c = _point_sound_speed(eos, ρ, pq, Y)
        (pq / c^2, -pq / (ρ * c))
    end
    q = q0
    for _ in 1:steps
        k1 = rhs(q, ρ)
        k2 = rhs(q + dq / 2, ρ + dq / 2 * k1[1])
        k3 = rhs(q + dq / 2, ρ + dq / 2 * k2[1])
        k4 = rhs(q + dq, ρ + dq * k3[1])
        ρ += dq / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
        u += dq / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
        q += dq
    end
    return (velocity=u, rho=ρ)
end

"""
    reflected_shock(eos, shocked::Prim; dim = 1, direction = +1, wall_velocity = 0.0)
        -> NamedTuple

The shock reflected from a wall at the high-coordinate end of dimension `dim`
(`direction = +1`; `-1` for the low end) by gas `shocked` arriving at it, such
that the gas behind the reflected shock moves with the wall. Returns
`(post, Mach, shock_speed)`, with `Mach` relative to `shocked`.
"""
function reflected_shock(eos, shocked::Prim; dim::Int=1, direction::Int=1,
                         wall_velocity::Real=0.0)
    eos = _as_point_eos(eos)
    s2 = thermodynamic_state(eos, shocked)
    (shocked.u[dim] - wall_velocity) * direction > 0 ||
        throw(ArgumentError("reflected_shock: the gas does not move towards the wall"))
    residual(M) = shock_jump(eos, shocked, M; dim=dim,
                             direction=-direction).velocity - wall_velocity
    # Constant-γ start: M_r/(M_r² − 1) = M_s/(M_s² − 1) · sqrt(...) is awkward to
    # invert, so start from the piston relation for the gas speed instead.
    Up = abs(shocked.u[dim] - wall_velocity) / s2.c
    γ = s2.gamma
    guess = (γ + 1) / 4 * Up + sqrt(((γ + 1) / 4 * Up)^2 + 1)
    M = _newton_root(residual, guess, "reflected_shock")
    jump = shock_jump(eos, shocked, M; dim=dim, direction=-direction)
    return (post=jump.post, Mach=M, shock_speed=jump.shock_speed)
end

"""
    shock_tube(eos, driver::Prim, driven::Prim, Mach) -> NamedTuple

The ideal shock-tube states for an incident shock of Mach number `Mach`,
travelling along dimension 1 from a driver at low coordinate into `driven`, and
its reflection from a closed end wall. Fields: `driver` (at the pressure
[`driver_pressure`](@ref) requires), `driven`, `shocked` (behind the incident
shock), `reflected` (behind the reflected shock, at rest), `Mach`,
`reflected_Mach`, `shock_speed`, `reflected_speed` and `velocity` (the gas
speed behind the incident shock).
"""
function shock_tube(eos, driver::Prim, driven::Prim, Mach::Real)
    incident = shock_jump(eos, driven, Mach)
    reflection = reflected_shock(eos, incident.post)
    return (driver=driver_pressure(eos, driver, driven, Mach), driven=driven,
            shocked=incident.post, reflected=reflection.post, Mach=Float64(Mach),
            reflected_Mach=reflection.Mach, shock_speed=incident.shock_speed,
            reflected_speed=reflection.shock_speed, velocity=incident.velocity)
end

"""
    NSCBCInflowBC(state::Prim; kwargs...)

A characteristic inflow whose targets are the velocity, temperature and
composition of `state`, which must give `T_ion`: the post-shock state of
[`shock_jump`](@ref) does. The keywords are those of the keyword constructor.
"""
function NSCBCInflowBC(state::Prim; kwargs...)
    isnan(state.T_ion) &&
        throw(ArgumentError("NSCBCInflowBC: the target state must give T_ion; " *
                            "build it from (p, T_ion), or read T_ion from " *
                            "thermodynamic_state(eos, state)"))
    return NSCBCInflowBC(; u=state.u, T_ion=state.T_ion, Y=collect(state.Y), kwargs...)
end

# --- The exact Riemann problem --------------------------------------------------

# The gas velocity behind the wave that takes side `state` to pressure `p`, and
# the state there: a shock when `p` exceeds the side's pressure, an isentropic
# expansion otherwise. `side = -1` is the left state (the wave travels left).
function _riemann_side(eos, state::Prim, s, p, dim, side)
    u = state.u[dim]
    if p > s.p
        # Mach number of the shock whose jump reaches p, from the constant-γ
        # inversion of the pressure ratio as a starting point.
        guess = sqrt(1 + (s.gamma + 1) / (2s.gamma) * (p / s.p - 1))
        M = _newton_root(m -> thermodynamic_state(eos,
                                  shock_jump(eos, state, m; dim=dim,
                                             direction=side).post).p / p - 1,
                         guess, "riemann_interface")
        jump = shock_jump(eos, state, M; dim=dim, direction=side)
        return (velocity=jump.velocity, state=jump.post, wave=:shock,
                speed=jump.shock_speed)
    end
    fan = _expansion(eos, Prim(p=s.p, rho=s.rho, Y=state.Y), p)
    ustar = u - side * fan.velocity
    post = Prim(u=ntuple(d -> d == dim ? ustar : state.u[d], 3), p=p,
                T_ion=_point_temperature(eos, fan.rho, p, state.Y), Y=state.Y)
    # The head of the fan travels at u ± c into the undisturbed gas.
    return (velocity=ustar, state=post, wave=:rarefaction, speed=u + side * s.c)
end

"""
    riemann_interface(eos, left::Prim, right::Prim; dim = 1) -> NamedTuple

The exact solution of the Riemann problem between `left` (at lower coordinate
along `dim`) and `right`, through `eos`: shocks from the Rankine–Hugoniot
relations and expansions along the real isentrope, so a calorically imperfect
gas needs no constant-γ approximation. The two states may be different gases.

Returns `(p_star, u_star, left, right, left_wave, right_wave, left_speed,
right_speed)`: the pressure and velocity of the contact, the states either side
of it as [`Prim`](@ref)s, the kind of each wave (`:shock` or `:rarefaction`),
and each wave's laboratory speed (the head of a rarefaction).

A shock striking an interface is the Riemann problem between the shocked gas
and the gas beyond the interface; its transmitted and reflected waves are the
right and left waves:

```julia
incident = shock_jump(eos, air, 1.5)
refraction = riemann_interface(eos, incident.post, sf6)
refraction.right_speed      # transmitted shock speed
refraction.u_star           # interface velocity after the impact
```
"""
function riemann_interface(eos, left::Prim, right::Prim; dim::Int=1)
    eos = _as_point_eos(eos)
    sL, sR = thermodynamic_state(eos, left), thermodynamic_state(eos, right)
    # Acoustic estimate of the contact pressure as the starting point.
    zL, zR = sL.rho * sL.c, sR.rho * sR.c
    p0 = (zL * sR.p + zR * sL.p + zL * zR * (left.u[dim] - right.u[dim])) / (zL + zR)
    p0 = max(p0, 1e-3 * min(sL.p, sR.p))
    mismatch(q) = _riemann_side(eos, left, sL, exp(q), dim, -1).velocity -
                  _riemann_side(eos, right, sR, exp(q), dim, 1).velocity
    p = exp(_newton_root(mismatch, log(p0), "riemann_interface"; rtol=1e-12))
    wl = _riemann_side(eos, left, sL, p, dim, -1)
    wr = _riemann_side(eos, right, sR, p, dim, 1)
    return (p_star=p, u_star=(wl.velocity + wr.velocity) / 2, left=wl.state,
            right=wr.state, left_wave=wl.wave, right_wave=wr.wave,
            left_speed=wl.speed, right_speed=wr.speed)
end

# --- Multimode perturbations ------------------------------------------------------

# SplitMix64: a fixed integer hash, so the phases of a seeded perturbation are
# the same on every rank, thread, platform and Julia version, which a
# generator from Random does not promise across versions.
function _splitmix(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

_unit_random(seed, a, b) = Float64(_splitmix(_splitmix(_splitmix(UInt64(seed)) +
                                                      reinterpret(UInt64, Int64(a))) +
                                            reinterpret(UInt64, Int64(b))) >> 11) *
                           2.0^-53

"""
    Multimode(; lengths, modes, rms, seed = 1, spectrum = k -> 1.0, mean = 0.0)

A multimode interface displacement `η(u, v) = mean + Σ a_m cos(k_m · (u, v) + φ_m)`,
for the bound of a [`Slab`](@ref), which calls it with the two coordinates other
than its own.

`lengths` holds the periods of the two coordinates, `(L_u,)` for a planar
perturbation or `(L_u, L_v)` for a surface. Every wave vector fitting those
periods whose mode number (`|n|`, with `k = 2π n / L` per direction) lies in
`modes`, a range such as `4:16`, is included once. The amplitude of each is
proportional to `sqrt(spectrum(|n|))`, so `spectrum` is the power per mode,
and the whole is scaled to the root-mean-square displacement `rms` over a
period. The phases are drawn from `seed` by a fixed hash, so the same arguments
give the same interface on every rank, every thread and every machine.

```julia
η = Multimode(lengths = (0.05,), modes = 1:8, rms = 5e-4,
              spectrum = n -> n^-2)          # a k⁻² amplitude-squared spectrum
Slab(1, lo = (y, z) -> 0.25 + η(y, z))
```
"""
struct Multimode
    mean::Float64
    k::Vector{NTuple{2,Float64}}
    amplitude::Vector{Float64}
    phase::Vector{Float64}
end

function Multimode(; lengths, modes, rms::Real, seed::Integer=1,
                   spectrum=n -> 1.0, mean::Real=0.0)
    L = Tuple(Float64.(lengths))
    1 <= length(L) <= 2 || throw(ArgumentError("Multimode: lengths has one or two periods"))
    nmax = ceil(Int, maximum(modes))
    ks = NTuple{2,Float64}[]
    power = Float64[]
    phase = Float64[]
    range2 = length(L) == 2 ? (-nmax:nmax) : (0:0)
    for n1 in 0:nmax, n2 in range2
        # One of each ± pair: cos(k·x + φ) and cos(−k·x − φ) are one mode.
        (n1 > 0 || n2 > 0) || continue
        n = hypot(n1, n2)
        minimum(modes) <= n <= maximum(modes) || continue
        push!(ks, (2π * n1 / L[1], length(L) == 2 ? 2π * n2 / L[2] : 0.0))
        push!(power, Float64(spectrum(n)))
        push!(phase, 2π * _unit_random(seed, n1, n2))
    end
    isempty(ks) && throw(ArgumentError("Multimode: no mode lies in $modes"))
    # A cosine of amplitude a has mean square a²/2 over its period.
    scale = rms / sqrt(sum(power) / 2)
    return Multimode(Float64(mean), ks, scale .* sqrt.(power), phase)
end

function (m::Multimode)(u, v=0.0)
    η = m.mean
    @inbounds for i in eachindex(m.k)
        η += m.amplitude[i] * cos(m.k[i][1] * u + m.k[i][2] * v + m.phase[i])
    end
    return η
end

# --- Synthetic turbulent inflow ---------------------------------------------------

# One wave vector of a `TurbulentInflow`: the velocity it contributes is
# `a cos θ + b sin θ` with `θ = k · x + omega t + phase`. `a` and `b` are the
# two polarizations normal to `k`, already scaled by the Cholesky factor.
struct _FourierMode
    k::NTuple{3,Float64}
    a::NTuple{3,Float64}
    b::NTuple{3,Float64}
    omega::Float64
    phase::Float64
end

"""
    TurbulentInflow(mean::Prim; length_scale, intensity = nothing,
                    reynolds_stress = nothing, n_modes = 192, seed = 1,
                    min_wavelength = length_scale / 2, spectrum = :von_karman,
                    convect = true)

A synthetic turbulent velocity fluctuation added to `mean`, callable as
`(x, y, z, t) -> Prim` for the target of a [`DirichletBC`](@ref) or an
[`NSCBCInflowBC`](@ref). Density, pressure, temperature and composition are
those of `mean`; an NSCBC target needs a `mean` that gives `T_ion`.

The fluctuation is a sum of `n_modes` random Fourier modes (Kraichnan 1970;
Smirnov, Shi and Celik 2001), `u' = A Σ_n (a_n cos θ_n + b_n sin θ_n)` with
`θ_n = k_n · (x - U t) + ω_n t + φ_n`, where `U` is the mean velocity, or zero
when `convect = false`. The wave numbers follow a von Kármán spectrum
truncated at `min_wavelength`, with its peak placed so that the longitudinal
integral length scale is `length_scale`. The wave vectors come in orthogonal
triads, each a random rotation of the coordinate axes, so the unscaled sum has
unit covariance for any `n_modes`, a multiple of 3, and zero divergence. The
frequencies `ω_n` are normal with standard deviation `u_rms |k_n|`.

`A` is the lower Cholesky factor of the Reynolds-stress tensor (Lund, Wu and
Squires 1998): `reynolds_stress`, a symmetric positive semidefinite 3×3
matrix, or `(intensity |U|)^2` times the identity. The long-time average of
`u'_i u'_j` over a face is then that tensor. An anisotropic tensor makes the
fluctuation divergent.

The modes are drawn from `seed` by a fixed integer hash at construction, so the
field is a function of `(x, y, z, t)` alone: it is the same on every rank, on
every process grid and after a restart, and has no state to checkpoint. A call
evaluates one `sincos` per mode and does not allocate. The coordinates are
taken as Cartesian positions.
"""
struct TurbulentInflow{N}
    mean::Prim{N}
    reynolds_stress::NTuple{3,NTuple{3,Float64}}
    length_scale::Float64
    min_wavelength::Float64
    n_modes::Int
    seed::Int
    spectrum::Symbol
    convect::Bool
    convection::NTuple{3,Float64}
    modes::Vector{_FourierMode}
end

# The modes are derived from the fields above, so a checkpoint's configuration
# record enters those and not the mode table.
_record_fields(::TurbulentInflow) = (:mean, :reynolds_stress, :length_scale,
                                     :min_wavelength, :n_modes, :seed, :spectrum,
                                     :convect)

# Lower Cholesky factor of a symmetric positive semidefinite 3×3 matrix. A zero
# pivot, a component without fluctuation, leaves its column zero.
function _cholesky3(R)
    L = zeros(3, 3)
    scale = max(maximum(abs, R), floatmin(Float64))
    for j in 1:3
        s = R[j, j] - sum(L[j, m]^2 for m in 1:j-1; init=0.0)
        s < -1e-12 * scale &&
            throw(ArgumentError("TurbulentInflow: reynolds_stress is not " *
                                "positive semidefinite"))
        L[j, j] = sqrt(max(s, 0.0))
        for i in j+1:3
            r = R[i, j] - sum(L[i, m] * L[j, m] for m in 1:j-1; init=0.0)
            if L[j, j] > 1e-12 * sqrt(scale)
                L[i, j] = r / L[j, j]
            elseif abs(r) > 1e-12 * scale
                throw(ArgumentError("TurbulentInflow: reynolds_stress is not " *
                                    "positive semidefinite"))
            end
        end
    end
    return L
end

# Unnormalized von Kármán energy spectrum with its peak near `ke`.
_von_karman(k, ke) = (k / ke)^4 / (1 + (k / ke)^2)^(17 / 6)

# Longitudinal integral length scale of a shell set carrying energies `q2`:
# L11 = (3π/4) ∫ E/k dk / ∫ E dk for isotropic turbulence.
_integral_length(k, q2) = 3π / 4 * sum(q2 ./ k) / sum(q2)

function TurbulentInflow(mean::Prim{N}; length_scale::Real, intensity=nothing,
                         reynolds_stress=nothing, n_modes::Integer=192,
                         seed::Integer=1, min_wavelength::Real=length_scale / 2,
                         spectrum::Symbol=:von_karman,
                         convect::Bool=true) where {N}
    L = Float64(length_scale)
    λmin = Float64(min_wavelength)
    L > 0 && isfinite(L) ||
        throw(ArgumentError("TurbulentInflow: length_scale must be positive"))
    0 < λmin < 2L ||
        throw(ArgumentError("TurbulentInflow: min_wavelength must lie in " *
                            "(0, 2 length_scale)"))
    n_modes > 0 && n_modes % 3 == 0 ||
        throw(ArgumentError("TurbulentInflow: n_modes must be a positive " *
                            "multiple of 3, got $n_modes"))
    spectrum === :von_karman ||
        throw(ArgumentError("TurbulentInflow: spectrum must be :von_karman"))
    (intensity === nothing) != (reynolds_stress === nothing) ||
        throw(ArgumentError("TurbulentInflow: give exactly one of intensity " *
                            "and reynolds_stress"))
    if intensity !== nothing
        speed = sqrt(sum(abs2, mean.u))
        intensity >= 0 && speed > 0 ||
            throw(ArgumentError("TurbulentInflow: intensity needs a nonnegative " *
                                "value and a nonzero mean velocity"))
        σ2 = (Float64(intensity) * speed)^2
        R = [i == j ? σ2 : 0.0 for i in 1:3, j in 1:3]
    else
        R = Float64.(collect(reynolds_stress))
        size(R) == (3, 3) && all(isfinite, R) ||
            throw(ArgumentError("TurbulentInflow: reynolds_stress must be a " *
                                "finite 3×3 matrix"))
        maximum(abs, R - R') <= 1e-12 * max(maximum(abs, R), floatmin(Float64)) ||
            throw(ArgumentError("TurbulentInflow: reynolds_stress must be symmetric"))
    end
    A = _cholesky3(R)
    u_rms = sqrt((R[1, 1] + R[2, 2] + R[3, 3]) / 3)

    # Shells evenly spaced in log k from well below the spectral peak to the
    # cutoff. Each carries E(k) dk, with dk ∝ k at even log spacing.
    shells = n_modes ÷ 3
    ke0 = 0.7468 / L        # the von Kármán peak for integral scale L
    klo, khi = ke0 / 8, 2π / λmin
    k = shells == 1 ? [sqrt(klo * khi)] :
        [klo * (khi / klo)^((m - 1) / (shells - 1)) for m in 1:shells]
    energy(ke) = [_von_karman(km, ke) * km for km in k]
    # The truncation shifts the integral scale; the peak is moved until the
    # discrete shells give `length_scale` exactly.
    lo, hi = ke0 / 16, 16ke0
    _integral_length(k, energy(lo)) > L > _integral_length(k, energy(hi)) ||
        throw(ArgumentError("TurbulentInflow: no von Kármán peak gives " *
                            "length_scale $L above min_wavelength $λmin; " *
                            "lower min_wavelength or raise n_modes"))
    for _ in 1:200
        mid = sqrt(lo * hi)
        _integral_length(k, energy(mid)) > L ? (lo = mid) : (hi = mid)
        hi / lo - 1 < 1e-14 && break
    end
    q2 = energy(sqrt(lo * hi))
    q2 ./= sum(q2)

    mix(v) = (A[1, 1] * v[1], A[2, 1] * v[1] + A[2, 2] * v[2],
              A[3, 1] * v[1] + A[3, 2] * v[2] + A[3, 3] * v[3])
    modes = Vector{_FourierMode}(undef, n_modes)
    for g in 1:shells
        r(slot) = _unit_random(seed, -g, slot)
        # A uniformly random rotation (Shoemake's quaternion); its columns
        # are the triad.
        u1, u2, u3 = r(1), r(2), r(3)
        qx, qy = sqrt(1 - u1) * sin(2π * u2), sqrt(1 - u1) * cos(2π * u2)
        qz, qw = sqrt(u1) * sin(2π * u3), sqrt(u1) * cos(2π * u3)
        Rot = ((1 - 2(qy^2 + qz^2), 2(qx * qy + qz * qw), 2(qx * qz - qy * qw)),
               (2(qx * qy - qz * qw), 1 - 2(qx^2 + qz^2), 2(qy * qz + qx * qw)),
               (2(qx * qz + qy * qw), 2(qy * qz - qx * qw), 1 - 2(qx^2 + qy^2)))
        q = sqrt(q2[g])
        for c in 1:3
            e, s1, s2 = Rot[c], Rot[mod1(c + 1, 3)], Rot[mod1(c + 2, 3)]
            # Two polarizations a quarter period apart at one wave vector: the
            # pair has covariance q²(I - e eᵀ)/2, and the triad q² I. The
            # sense of the pair is random.
            h = r(3 + c) < 0.5 ? -1.0 : 1.0
            phase = 2π * r(6 + c)
            ξ = sqrt(-2 * log(1 - r(9 + c))) * cos(2π * r(12 + c))
            modes[3(g-1)+c] = _FourierMode(k[g] .* e, mix(q .* s1),
                                           mix((h * q) .* s2),
                                           u_rms * k[g] * ξ, phase)
        end
    end
    Rt = ntuple(i -> ntuple(j -> R[i, j], 3), 3)
    convection = convect ? mean.u : (0.0, 0.0, 0.0)
    return TurbulentInflow{N}(mean, Rt, L, λmin, Int(n_modes), Int(seed),
                              spectrum, convect, convection, modes)
end

# The velocity fluctuation at a point.
@inline function _inflow_fluctuation(f::TurbulentInflow, x1, x2, x3, t)
    tt = Float64(t)
    X1 = Float64(x1) - f.convection[1] * tt
    X2 = Float64(x2) - f.convection[2] * tt
    X3 = Float64(x3) - f.convection[3] * tt
    u1 = u2 = u3 = 0.0
    @inbounds for m in f.modes
        s, c = sincos(m.k[1] * X1 + m.k[2] * X2 + m.k[3] * X3 +
                      m.omega * tt + m.phase)
        u1 += m.a[1] * c + m.b[1] * s
        u2 += m.a[2] * c + m.b[2] * s
        u3 += m.a[3] * c + m.b[3] * s
    end
    return (u1, u2, u3)
end

function (f::TurbulentInflow{N})(x1, x2, x3, t) where {N}
    du = _inflow_fluctuation(f, x1, x2, x3, t)
    m = f.mean
    return Prim{N}(m.Y, (m.u[1] + du[1], m.u[2] + du[2], m.u[3] + du[3]),
                   m.p, m.T_ion, m.rho)
end
