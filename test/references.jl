# Analytic reference solutions shared by the test suites.
#
# These are the *independent* half of validation: closed-form solutions of the
# Euler equations that the solver has never seen. `test/validation.jl` measures
# against them; `test/runtests.jl` uses the Riemann solver for Sod. Cases with
# no closed form (Shu–Osher, Woodward–Colella) are handled there instead by a
# stored high-resolution profile from this code, which is a regression guard
# rather than a validation — the distinction is kept explicit in both files.
#
# No `using` of its own: include this after CompactLES so it inherits Printf.

# --- Exact Riemann solver for the ideal-gas Euler equations (Toro, Ch. 4) ---

"Star-region (p*, u*) by Newton iteration on the pressure function."
function exact_riemann_star(ρL, uL, pL, ρR, uR, pR, γ)
    cL = sqrt(γ * pL / ρL); cR = sqrt(γ * pR / ρR)
    G1 = (γ - 1) / (2γ)
    f(p, ρk, pk, ck) = p > pk ?
        (p - pk) * sqrt((2 / ((γ + 1) * ρk)) / (p + (γ - 1) / (γ + 1) * pk)) :
        (2ck / (γ - 1)) * ((p / pk)^G1 - 1)
    fder(p, ρk, pk, ck) = p > pk ?
        sqrt((2 / ((γ + 1) * ρk)) / (p + (γ - 1) / (γ + 1) * pk)) *
            (1 - (p - pk) / (2 * (p + (γ - 1) / (γ + 1) * pk))) :
        (1 / (ρk * ck)) * (p / pk)^(-(γ + 1) / (2γ))
    p = max(0.5 * (pL + pR), 1e-8)
    for _ in 1:100
        F  = f(p, ρL, pL, cL) + f(p, ρR, pR, cR) + (uR - uL)
        Fd = fder(p, ρL, pL, cL) + fder(p, ρR, pR, cR)
        pnew = p - F / Fd
        abs(pnew - p) / (0.5 * (p + pnew)) < 1e-12 && (p = pnew; break)
        p = max(pnew, 1e-9)
    end
    ustar = 0.5 * (uL + uR) +
            0.5 * (f(p, ρR, pR, cR) - f(p, ρL, pL, cL))
    (p, ustar, cL, cR)
end

"Sample (ρ,u,p) of the exact solution at self-similar speed S = (x−x0)/t."
function exact_riemann_sample(S, ρL, uL, pL, ρR, uR, pR, γ, pstar, ustar, cL, cR)
    G1 = (γ - 1) / (2γ); G6 = (γ - 1) / (γ + 1); G7 = (γ - 1) / 2
    if S <= ustar                                   # left of contact
        if pstar > pL                               # left shock
            SL = uL - cL * sqrt((γ + 1) / (2γ) * pstar / pL + G1)
            S <= SL && return (ρL, uL, pL)
            return (ρL * ((pstar / pL + G6) / (G6 * pstar / pL + 1)), ustar, pstar)
        else                                        # left rarefaction
            SHL = uL - cL; STL = ustar - cL * (pstar / pL)^G1
            S <= SHL && return (ρL, uL, pL)
            S >= STL && return (ρL * (pstar / pL)^(1 / γ), ustar, pstar)
            u = (2 / (γ + 1)) * (cL + G7 * uL + S)
            c = (2 / (γ + 1)) * (cL + G7 * (uL - S))
            return (ρL * (c / cL)^(2 / (γ - 1)), u, pL * (c / cL)^(2γ / (γ - 1)))
        end
    else                                            # right of contact
        if pstar > pR                               # right shock
            SR = uR + cR * sqrt((γ + 1) / (2γ) * pstar / pR + G1)
            S >= SR && return (ρR, uR, pR)
            return (ρR * ((pstar / pR + G6) / (G6 * pstar / pR + 1)), ustar, pstar)
        else                                        # right rarefaction
            SHR = uR + cR; STR = ustar + cR * (pstar / pR)^G1
            S >= SHR && return (ρR, uR, pR)
            S <= STR && return (ρR * (pstar / pR)^(1 / γ), ustar, pstar)
            u = (2 / (γ + 1)) * (-cR + G7 * uR + S)
            c = (2 / (γ + 1)) * (cR - G7 * (uR - S))
            return (ρR * (c / cR)^(2 / (γ - 1)), u, pR * (c / cR)^(2γ / (γ - 1)))
        end
    end
end

"""
    riemann_profile(x, t, x0, left, right, γ) -> (ρ, u, p)

Convenience wrapper: exact solution of a Riemann problem with the diaphragm at
`x0`, sampled at `(x, t)`. `left`/`right` are `(ρ, u, p)` tuples.
"""
function riemann_profile(x, t, x0, left, right, γ)
    ρL, uL, pL = left
    ρR, uR, pR = right
    pstar, ustar, cL, cR = exact_riemann_star(ρL, uL, pL, ρR, uR, pR, γ)
    exact_riemann_sample((x - x0) / t, ρL, uL, pL, ρR, uR, pR, γ,
                         pstar, ustar, cL, cR)
end

# --- Noh's uniform-implosion problem (Noh, JCP 1987) ------------------------
#
# Cold gas (p → 0, ρ = ρ₀) converging at unit speed onto a symmetry point. The
# solution is exact for all time and in every geometry, which is why it is the
# standard probe of an artificial-viscosity model: the post-shock plateau,
# the shock speed, and the pre-shock compression are all fixed numbers, so
# wall heating (a spurious internal-energy excess at the symmetry point,
# showing as a density deficit there) has nowhere to hide.
#
# With ν = 1, 2, 3 (planar, cylindrical, spherical) and γ = 5/3:
#
#   shock speed        S = (γ − 1)/2 = 1/3
#   post-shock ρ       ρ₀ ((γ+1)/(γ−1))^ν = 4^ν  (4, 16, 64)
#   post-shock p       (γ − 1) ρ e with e = u²/2 = 1/2
#   pre-shock  ρ       ρ₀ (1 + t/r)^(ν−1)   — geometric convergence only
#   pre-shock  u, p    −1, 0

"""
    noh_exact(r, t, ν, γ; rho0=1.0) -> (ρ, u, p)

Exact Noh solution at radius `r` and time `t` in `ν` dimensions.
"""
function noh_exact(r, t, ν::Int, γ; rho0=1.0)
    S = (γ - 1) / 2
    if r <= S * t                                   # shocked, stagnant core
        ρ = rho0 * ((γ + 1) / (γ - 1))^ν
        return (ρ, 0.0, (γ - 1) * ρ * 0.5)
    else                                            # free-falling inflow
        return (rho0 * (1 + t / r)^(ν - 1), -1.0, 0.0)
    end
end

# --- Sedov–Taylor point blast ----------------------------------------------
#
# The self-similar strong-blast solution gives the shock trajectory
#
#   R_s(t) = ξ₀ (E t² / ρ₀)^(1/(ν+2)),
#
# with ξ₀ a pure number fixed by the energy integral over the similarity
# profile. The profile itself is parametric and is not reproduced here; what
# is checked in `validation.jl` is the trajectory (the origin
# fold and the geometric source terms actually determine) together with the
# strong-shock density jump (γ+1)/(γ−1).
#
# ξ₀ values below are the standard tabulated ones (Sedov 1959; Kamm & Timmes
# 2007, Table 1) and are γ-specific — the lookup errors rather than
# extrapolating, because a silently wrong ξ₀ would turn this test into a
# tautology.

const SEDOV_XI0 = Dict((3, 1.4) => 1.032840,      # spherical, γ = 7/5
                       (3, 5/3) => 1.151725,      # spherical, γ = 5/3
                       (2, 1.4) => 1.004965,      # cylindrical, γ = 7/5
                       (1, 1.4) => 1.121193)      # planar, γ = 7/5

"""
    sedov_shock_radius(E, t, ν, γ; rho0=1.0)

Self-similar blast-wave radius. `E` is the total (ν-dimensional) deposited
energy: for ν = 3 the physical energy integrated over the full sphere.
"""
function sedov_shock_radius(E, t, ν::Int, γ; rho0=1.0)
    key = (ν, γ)
    haskey(SEDOV_XI0, key) ||
        error("no tabulated Sedov ξ₀ for ν = $ν, γ = $γ")
    SEDOV_XI0[key] * (E * t^2 / rho0)^(1 / (ν + 2))
end

# --- Profile utilities ------------------------------------------------------

"""
    front_position(xs, f, level) -> x

Outermost crossing of `f` through `level`, linearly interpolated. `xs` must be
increasing; returns `NaN` when the profile never crosses. Used to locate shock
fronts without differentiating a captured (smeared) profile.
"""
function front_position(xs, f, level)
    for i in length(f):-1:2
        lo, hi = f[i-1], f[i]
        if (lo - level) * (hi - level) <= 0 && lo != hi
            return xs[i-1] + (xs[i] - xs[i-1]) * (level - lo) / (hi - lo)
        end
    end
    return NaN
end

"Mean |a − b| over the profile."
l1(a, b) = sum(abs, a .- b) / length(a)
