# The Phase 1 shock-capturing cases, as plain functions with no measurements
# and no assertions.
#
# `test/validation.jl` includes this and adds the references and regression
# guards; `bench/artcal.jl` includes it and sweeps the artificial-property
# constants over the same configurations. Keeping the setups here means the
# calibration study and the validation battery cannot drift apart — a swept
# constant is measured against exactly the run the guard is set from.
#
# Every case is one-dimensional along dimension 1 with the transverse
# dimensions collapsed, so each costs N points rather than N³, and every one
# returns the same `(x, rho, u, p)` tuple.
#
# Include after CompactLES and Printf; `references.jl` supplies noh_exact.

const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

# Step ceiling. A healthy run of the largest case here takes a few thousand
# steps; a configuration that loses positivity does not diverge, it drives the
# diffusive rate in compute_dt up until dt collapses and then grinds forever.
# Every case therefore takes an nmax, and a sweep over configurations that are
# expected to include bad ones should pass a much lower one — otherwise a
# single stalled point costs more wall time than the whole sweep.
const NMAX = 2_000_000

# --- profile extraction -----------------------------------------------------

"""
    case_line_profile(solver, Q) -> (x, rho, u, p)

Interior (x, ρ, u, p) along dimension 1 at j = k = 1, gathered across the
dimension-1 sub-communicator so the cases read the same under `mpiexec`.

Named apart from the exported `line_profile`, which these files bring into
scope through `using CompactLES` and which returns one named field, not the four
the cases compare against.
"""
function case_line_profile(solver, Q)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    nx = solver.decomp.n_local[1]
    loc = (Float64[xcoord(solver, 1, i) for i in 1:nx],
           [solver.rho[gidx(solver, i, 1, 1)] for i in 1:nx],
           [solver.u[gidx(solver, i, 1, 1)]   for i in 1:nx],
           [solver.p[gidx(solver, i, 1, 1)]   for i in 1:nx])
    comm = solver.decomp.sub[1]
    MPI.Comm_size(comm) == 1 && return loc
    counts = MPI.Allgather(Int32(nx), comm)
    map(loc) do v
        out = Vector{Float64}(undef, sum(counts))
        MPI.Allgatherv!(v, MPI.VBuffer(out, counts), comm)
        out
    end
end

"Linear interpolation of (xs, ys) onto `xq`; clamped outside the range."
function interp1(xs, ys, xq)
    xq <= xs[1] && return ys[1]
    xq >= xs[end] && return ys[end]
    i = searchsortedfirst(xs, xq)
    t = (xq - xs[i-1]) / (xs[i] - xs[i-1])
    return (1 - t) * ys[i-1] + t * ys[i]
end

"True when the run reached `tfin` rather than stalling or going non-finite."
completed(solver, tfin) = isfinite(solver.t) && solver.t >= tfin * (1 - 1e-9)

# --- shock tubes ------------------------------------------------------------

"""
    tube(left, right; N, L, x0, tfin, ...) -> (x, rho, u, p)

One-dimensional Riemann-problem run along x. Both ends hold their initial state
through a `DirichletBC`, which is exact here because no wave reaches either end
within `tfin` — a slip wall would be wrong the moment a state carries velocity,
since it zeroes the normal momentum on the wall plane and launches a spurious
wave inward.

The diaphragm is smeared over `delta`, two cells by default as in the Sod test
in runtests.jl. It is a keyword rather than always 2h because a resolution study
that scales the smearing with the grid is refining a *different continuous
problem* at every N, which quietly conflates discretization error with a changed
initial condition — the high-resolution reference profiles in test/refs are
therefore generated at the test resolution's delta, not their own.
"""
function tube(left, right; N, L=1.0, x0=0.5, tfin, γ=1.4,
              art=ArtParams(enabled=true), cfl=0.4, xlo=0.0, rhofun=nothing,
              nmax=NMAX, delta=nothing)
    h = L / (N - 1)
    δ = delta === nothing ? 2h : delta
    ρL, uL, pL = left
    ρR, uR, pR = right
    ρat(x) = rhofun === nothing ? ρR : rhofun(x)
    bcs = (DirichletBC((x, y, z, t) -> Prim(rho=ρL, u=(uL, 0.0, 0.0), p=pL)),
           DirichletBC((x, y, z, t) -> Prim(rho=ρat(x), u=(uR, 0.0, 0.0), p=pR)))
    prob = Problem(eos=IdealSpecies("gas"; gamma=γ, R=1.0),
                   transport=Transport(mu0=0.0),
                   domain=((xlo, xlo + L), (0.0, h), (0.0, h)),
                   bcs=(bcs, per3[2], per3[3]),
                   ic=(x, y, z) -> begin
                       θ = tanh_blend(x, x0, δ)
                       Prim(rho=(1 - θ) * ρL + θ * ρat(x),
                            u=((1 - θ) * uL + θ * uR, 0.0, 0.0),
                            p=(1 - θ) * pL + θ * pR)
                   end)
    solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=art, cfl=cfl,
                                     filter_interval=1))
    run!(solver, Q; tfinal=tfin, nmax=nmax)
    return case_line_profile(solver, Q)..., completed(solver, tfin)
end

# Lax: Sod's harder sibling. The left state carries momentum, so the star
# region sits far from both initial states and a wrong wave speed cannot hide
# behind a nearly-symmetric answer.
const LAX_L = (0.445, 0.698, 3.528)
const LAX_R = (0.5, 0.0, 0.571)
const LAX_T = 0.14
const LAX_N = 400

lax(; N=LAX_N, kw...) = tube(LAX_L, LAX_R; N=N, tfin=LAX_T, kw...)

# Shu–Osher: Mach 3 shock into a sinusoidal density field. The post-shock wave
# train is the diagnostic — a scheme that smears it loses amplitude, as does an
# over-eager artificial viscosity.
const SO_L = (3.857143, 2.629369, 10.333333333333333)
const SO_T = 1.8
const SO_N = 800

shu_osher(; N=SO_N, kw...) =
    tube(SO_L, (1.0, 0.0, 1.0); N=N, L=10.0, xlo=-5.0, x0=-4.0, tfin=SO_T,
         rhofun=x -> 1 + 0.2sin(5x), kw...)

# Woodward–Colella: two blast waves colliding between reflecting walls at a
# 10^5 pressure ratio. The test is whether the code survives it at all, and
# whether the collided contact ends up in the right place.
const WC_T = 0.038
const WC_N = 800

function woodward(; N=WC_N, art=ArtParams(enabled=true), cfl=0.3, nmax=NMAX,
                  delta=nothing, deriv=lele_d1_6(), filt=compact_filter(0.45))
    h = 1.0 / (N - 1)
    δ = delta === nothing ? 2h : delta
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   transport=Transport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                        p = 1000 * (1 - tanh_blend(x, 0.1, δ)) +
                            0.01 * (tanh_blend(x, 0.1, δ) - tanh_blend(x, 0.9, δ)) +
                            100 * tanh_blend(x, 0.9, δ)))
    solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=art, cfl=cfl,
                                     deriv=deriv, filt=filt, filter_interval=1))
    run!(solver, Q; tfinal=WC_T, nmax=nmax)
    return case_line_profile(solver, Q)..., completed(solver, WC_T)
end

# --- Sedov–Taylor -----------------------------------------------------------
#
# θ and φ are collapsed, so this costs N points; θ sits at π/2 so the collapsed
# scale factor r·sinθ is exactly r and the divergence is the true
# (1/r²)∂_r(r² F). The deposit is a Gaussian rather than a top hat: the origin
# fold will not take a discontinuity resolved over fewer than about three cells
# (see reference/CALIBRATION.md), and a smoothed source is standard practice
# for Sedov anyway.

const SEDOV_E = 0.851 * 0.8^5      # chosen for R_s(t = 1) ≈ 0.8
const SEDOV_T = 1.0
const SEDOV_S = 0.06               # deposit width, ≈ 13 cells at N = 256
const SEDOV_N = 256

function sedov(; N=SEDOV_N, R=1.2, σ=SEDOV_S, art=ArtParams(enabled=true), cfl=0.3,
               nmax=NMAX)
    γ = 1.4
    # E = ∫ p/(γ−1) dV over the full sphere for p = p_in exp(−r²/σ²), using
    # ∫₀^∞ r² e^{−r²/σ²} dr = σ³√π/4, so E = π^{3/2} p_in σ³ / (γ−1).
    pin = SEDOV_E * (γ - 1) / (π^1.5 * σ^3)
    prob = Problem(eos=IdealSpecies("gas"; gamma=γ, R=1.0),
                   transport=Transport(mu0=0.0),
                   metric=SphericalMetric(),
                   domain=((0.0, R), (π / 2, π / 2 + 1), (0.0, 1.0)),
                   bcs=((OriginBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(r, θ, φ) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                        p=1e-5 + pin * exp(-(r / σ)^2)))
    solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=art, cfl=cfl,
                                     filter_interval=1))
    run!(solver, Q; tfinal=SEDOV_T, nmax=nmax)
    return case_line_profile(solver, Q)..., completed(solver, SEDOV_T)
end

# --- Noh --------------------------------------------------------------------
#
# Cold gas at u = −1 stagnating on a symmetry point. Every feature of the answer
# is a fixed number, so the artificial viscosity has to produce the right
# plateau and avoid depositing spurious entropy at the point of symmetry.
#
# ν = 1 planar (slip wall), ν = 2 cylindrical axis fold, ν = 3 spherical origin
# fold. Only ν = 3 is warm-started, because the spherical origin will not take
# the singular t = 0 start; see reference/CALIBRATION.md.

const NOH_G = 5 / 3
const NOH_T = 0.6
const NOH_P0 = 1e-4
const NOH_CFL = 0.15
const NOH_N = (1 => 400, 2 => 256, 3 => 256)
const NOH_T0 = (1 => 0.0, 2 => 0.0, 3 => 0.3)

"""
    noh_case(ν; N, t0, art, cfl) -> (x, rho, u, p, completed)

Noh in ν dimensions, integrated from `t0` to `NOH_T`. `t0 > 0` initializes from
the exact solution, bypassing the uniform cold inflow. The outer boundary
carries the exact time-dependent inflow, which for ν > 1 is compressing as it
converges and is therefore not a constant state.
"""
function noh_case(ν::Int; N=Dict(NOH_N)[ν], t0=Dict(NOH_T0)[ν],
                  art=ArtParams(enabled=true), cfl=NOH_CFL, R=1.0, nmax=NMAX,
                  deriv=lele_d1_6(), filt=compact_filter(0.45))
    metric = ν == 1 ? CartesianMetric() :
             ν == 2 ? CylindricalMetric() : SphericalMetric()
    lobc = ν == 1 ? SlipWallBC() : ν == 2 ? AxisBC() : OriginBC()
    dom2 = ν == 1 ? (0.0, R / N) : ν == 2 ? (0.0, 1.0) : (π / 2, π / 2 + 1)
    dom3 = ν == 1 ? (0.0, R / N) : (0.0, 1.0)
    inflow = DirichletBC((x, y, z, t) -> begin
        # A run that has lost positivity drives solver.t to NaN, and the exact
        # solution would then hand Prim a NaN density — which throws, because
        # Prim requires exactly one of T_ion and rho to be NaN. A sweep needs
        # divergence reported as a result, not raised as an exception, so the
        # boundary state freezes at t0 once the clock stops being finite.
        ρ, uu, _ = noh_exact(x, isfinite(t) ? t + t0 : t0, ν, NOH_G)
        Prim(rho=ρ, u=(uu, 0.0, 0.0), p=NOH_P0)
    end)
    w = 4R / N
    ic = (x, y, z) -> begin
        t0 <= 0 && return Prim(rho=1.0, u=(-1.0, 0.0, 0.0), p=NOH_P0)
        ρin, _, pin = noh_exact(0.0, t0, ν, NOH_G)
        ρout, _, _ = noh_exact(x, t0, ν, NOH_G)
        θ = tanh_blend(x, (NOH_G - 1) / 2 * t0, w)
        Prim(rho=(1 - θ) * ρin + θ * ρout, u=(-θ, 0.0, 0.0),
             p=(1 - θ) * pin + θ * NOH_P0)
    end
    prob = Problem(eos=IdealSpecies("gas"; gamma=NOH_G, R=1.0),
                   transport=Transport(mu0=0.0), metric=metric,
                   domain=((0.0, R), dom2, dom3),
                   bcs=((lobc, inflow), per3[2], per3[3]), ic=ic)
    solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=art, cfl=cfl,
                                     deriv=deriv, filt=filt, filter_interval=1))
    run!(solver, Q; tfinal=NOH_T - t0, nmax=nmax)
    return case_line_profile(solver, Q)..., completed(solver, NOH_T - t0)
end

# --- species interface ------------------------------------------------------
#
# A sharp binary interface carried through a periodic domain at uniform ρ, p and
# u. Nothing else happens: no shock, no shear, no pressure gradient, so the
# interface growth measures the effect of artificial species diffusivity. This
# is the only case in the set that isolates C_D.

const MIX_N = 256
const MIX_T = 0.5
const MIX_U = 1.0

"""
    species_advection(; N, tfin, art, cfl) -> (x, Y1, rho, p, completed)

Uniform advection of a tanh species interface. Returns the FIRST mass fraction
in the slot the shock-tube cases use for velocity, since ρ, u and p are uniform
by construction and carry no information here.
"""
function species_advection(; N=MIX_N, tfin=MIX_T, art=ArtParams(enabled=true),
                           cfl=0.4, nmax=NMAX)
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 1.0, 1.4)])
    h = 1.0 / N
    prob = Problem(eos=eos, transport=Transport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)), bcs=per3,
                   ic=(x, y, z) -> begin
                       θ = tanh_blend(x, 0.25, 2h)
                       Prim(Y=(1 - θ, θ), u=(MIX_U, 0.0, 0.0), p=1.0, rho=1.0)
                   end)
    solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=art, cfl=cfl,
                                     filter_interval=1))
    run!(solver, Q; tfinal=tfin, nmax=nmax)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    nx = solver.decomp.n_local[1]
    xs = Float64[xcoord(solver, 1, i) for i in 1:nx]
    Y1 = [solver.Y[1][gidx(solver, i, 1, 1)] for i in 1:nx]
    ρ = [solver.rho[gidx(solver, i, 1, 1)] for i in 1:nx]
    p = [solver.p[gidx(solver, i, 1, 1)] for i in 1:nx]
    return xs, Y1, ρ, p, completed(solver, tfin)
end

# --- shock / species interface ----------------------------------------------
#
# A Mach 1.5 shock in air runs into a tanh interface with a heavy gas of SF6's
# density ratio and γ. Crossing the density and γ jump drives the mass
# fractions out of [0, 1] wherever the artificial species diffusivity does not
# hold them, so the excursion history over the run and the interface width at
# the end measure what C_D does at a shocked interface. species_advection
# cannot: its interface moves at uniform pressure and nothing excites it.

const SI_N = 400
const SI_T = 0.25
const SI_MACH = 1.5
const SI_X_SHOCK = 0.15
const SI_X_IFACE = 0.35
const SI_RHO_HEAVY = 5.04          # SF6 at the pre-shock p and T, air = 1
const SI_GAMMA_HEAVY = 1.09

"""
    shock_interface(; N, tfin, art, cfl, delta, nmax) -> NamedTuple

Mach `SI_MACH` shock in air (γ = 1.4, R = 1, ρ = 1 and p = 1 ahead of it) at
`SI_X_SHOCK` running into a tanh interface at `SI_X_IFACE` with a heavy gas
(γ = `SI_GAMMA_HEAVY`, R = 1 / `SI_RHO_HEAVY`, so ρ = `SI_RHO_HEAVY` at the
pre-shock p and T). The post-shock state follows from the Rankine–Hugoniot
relations, and both ends hold their initial state through a `DirichletBC`.
`delta` is the smearing width of the shock and the interface in cells, not a
length as in `tube`, so the sweep in bench/artcal.jl reads directly in cells.

Returns `(x, Y_air, rho, p, worst_min_Y, worst_max_Y, width_cells, steps,
completed)`. The two `worst` values are the extreme mass fractions over both
species and every step, read from `Q` in a callback because the excursions
are transient and the final profile need not show them; `width_cells` counts
the interior points with 0.05 < Y_air < 0.95 at the end.
"""
function shock_interface(; N=SI_N, tfin=SI_T, art=ArtParams(enabled=true),
                         cfl=0.4, delta=2.0, nmax=NMAX)
    γa = 1.4
    eos = IdealMixture([IdealSpecies{Float64}("air", 1.0, γa),
                        IdealSpecies{Float64}("sf6", 1 / SI_RHO_HEAVY,
                                              SI_GAMMA_HEAVY)])
    M = SI_MACH
    p2 = 1 + 2γa / (γa + 1) * (M^2 - 1)
    r2 = (γa + 1) * M^2 / ((γa - 1) * M^2 + 2)
    u2 = M * sqrt(γa) * (1 - 1 / r2)
    h = 1.0 / (N - 1)
    δ = delta * h
    bcs = (DirichletBC((x, y, z, t) -> Prim(Y=(1.0, 0.0), rho=r2,
                                            u=(u2, 0.0, 0.0), p=p2)),
           DirichletBC((x, y, z, t) -> Prim(Y=(0.0, 1.0), rho=SI_RHO_HEAVY,
                                            u=(0.0, 0.0, 0.0), p=1.0)))
    prob = Problem(eos=eos, transport=Transport(mu0=0.0),
                   domain=((0.0, 1.0), (0.0, h), (0.0, h)),
                   bcs=(bcs, per3[2], per3[3]),
                   ic=(x, y, z) -> begin
                       s = tanh_blend(x, SI_X_SHOCK, δ)   # 0 post-shock, 1 ahead
                       θ = tanh_blend(x, SI_X_IFACE, δ)   # 0 air, 1 heavy
                       ρ = (1 - s) * r2 + s * ((1 - θ) + θ * SI_RHO_HEAVY)
                       Prim(Y=(1 - θ, θ), rho=ρ, u=((1 - s) * u2, 0.0, 0.0),
                            p=(1 - s) * p2 + s)
                   end)
    solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=art, cfl=cfl,
                                     filter_interval=1,
                                     control=StepControl(retries=4)))
    nx = solver.decomp.n_local[1]
    worst_min = Ref(Inf)
    worst_max = Ref(-Inf)
    function excursion(s, Q)
        for i in 1:nx
            I = gidx(s, i, 1, 1)
            y = Q[I, 1] / (Q[I, 1] + Q[I, 2])
            worst_min[] = min(worst_min[], y, 1 - y)
            worst_max[] = max(worst_max[], y, 1 - y)
        end
    end
    run!(solver, Q; tfinal=tfin, nmax=nmax, callback=excursion)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    xs = Float64[xcoord(solver, 1, i) for i in 1:nx]
    Y1 = [solver.Y[1][gidx(solver, i, 1, 1)] for i in 1:nx]
    ρ = [solver.rho[gidx(solver, i, 1, 1)] for i in 1:nx]
    p = [solver.p[gidx(solver, i, 1, 1)] for i in 1:nx]
    width = count(y -> 0.05 < y < 0.95, Y1)
    comm = solver.decomp.sub[1]
    if MPI.Comm_size(comm) > 1
        worst_min[] = MPI.Allreduce(worst_min[], min, comm)
        worst_max[] = MPI.Allreduce(worst_max[], max, comm)
        width = MPI.Allreduce(width, +, comm)
    end
    return (x=xs, Y_air=Y1, rho=ρ, p=p, worst_min_Y=worst_min[],
            worst_max_Y=worst_max[], width_cells=width, steps=solver.step,
            completed=completed(solver, tfin))
end

# --- shared measurements ----------------------------------------------------
#
# validation.jl guards these and artcal.jl sweeps them, so both report the
# identical quantity.

"""
    noh_metrics(xs, ρ, ν) -> (plateau, deficit, shock, l1_preshock)

Post-shock plateau (sampled between the symmetry point and the shock, so it
misses both the wall-heating layer and the captured front), the density deficit
at the symmetry point — positive means too hot, hence too rarefied, i.e. wall
heating — the captured shock radius, and the pre-shock L1 error.
"""
function noh_metrics(xs, ρ, ν::Int)
    exact = 4.0^ν
    Rs = (NOH_G - 1) / 2 * NOH_T
    core = [i for i in eachindex(xs) if 0.3Rs <= xs[i] <= 0.7Rs]
    plat = sum(ρ[core]) / length(core)
    pre = [i for i in eachindex(xs) if xs[i] > Rs + 0.05]
    epre = l1(ρ[pre], [noh_exact(xs[i], NOH_T, ν, NOH_G)[1] for i in pre])
    return (plat, 1 - ρ[1] / exact, front_position(xs, ρ, 0.5exact), epre)
end

"Shu–Osher wave-train window: between the trailing entropy waves and the shock."
so_band(xs) = [i for i in eachindex(xs) if 0.5 <= xs[i] <= 2.2]

"""
    contact_width(xs, ρ, ρa, ρb) -> width

Thickness of a contact discontinuity, as the distance over which ρ crosses from
10% to 90% of the way between the two plateau values. The direct measure of
what C_D and C_beta do to an interface.
"""
function contact_width(xs, ρ, ρa, ρb)
    lo = ρa + 0.1 * (ρb - ρa)
    hi = ρa + 0.9 * (ρb - ρa)
    xlo = front_position(xs, ρ, lo)
    xhi = front_position(xs, ρ, hi)
    return abs(xhi - xlo)
end
