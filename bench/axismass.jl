# The mass and energy carried through the face at a cylindrical axis: the face
# value of the folded divergence at r = 0, measured from the solver's own
# right-hand side, and the total mass and energy of closed runs with radial
# flow through the axis region against the spherical origin and two Cartesian
# lines.
#
#   julia --project=. -t 1 bench/axismass.jl                         # everything
#   julia --project=. -t 1 bench/axismass.jl parts=face,vortex
#   julia --project=. -t 1 bench/axismass.jl parts=runs N=64,128 tfinal=0.5
#   julia --project=. -t 1 bench/axismass.jl parts=runs variants=on:on,off:off
#
# Parts: quadrature (`volume_integral` of smooth fields on the curvilinear and
# stretched lines, against their exact integrals), face (one right-hand side on
# the θ- and z-collapsed lines), vortex (the same on the resolved (r, θ) grid of
# the axis-crossing vortex, at half of each N radially and `ntheta`
# azimuthally) and runs (closed runs to `tfinal`).
#
# The face form. Under the node weights W of the positivity limiter
# (src/positivity.jl) the compact divergence of a closed line telescopes,
# Σ_i W_i (D g)_i = g_N − g_1. On a line folded at its low end the weights are
# h on the fold's half and the closed line's tail at the far end, and the
# running sum leaves the face value Ĝ_0 at the fold: Σ_i W_i (D g)_i = g_N − Ĝ_0.
# Ĝ_0 vanishes for an odd area-weighted flux (the mass and energy fluxes at the
# spherical origin, A = r²) and not for an even one (the same fluxes at the
# cylindrical axis, A = r), where the interior face relation on the mirrored
# data gives Ĝ_0 ≈ g(0) − h² g''(0)/24 = −h² ρ u_r'(0)/12 for g = r ρ u_r.
# With W J as the quadrature, the right-hand side's mass rate is therefore
# Σ W J dQ_ρ = Ĝ_0 − g_N, and the axis face is measured as that rate plus the
# far face g_N, read from the area-weighted flux the right-hand side used.
# The folded divergence of an even field annihilates a constant while the
# telescoped sum of a constant is not zero, so no node weights at all, positive
# or not, remove Ĝ_0: the face part prints max |D 1| as that check. With θ
# resolved, each line and its antipode form one diameter, the even part of the
# pair carries the face, and the faces of all lines sum to
# −(π h²/12) ∇·(ρu)(0) per unit length, the collapsed value times 2π; the
# vortex part measures that sum on the translated vortex, where
# ∇·(ρu)(0) = U ∂ρ/∂x(0), with the Dirichlet far face read as above.
#
# The runs. Closed lines of N nodes on (0, 1] (r-z axis or spherical origin
# with a slip wall at r = 1, a Cartesian line between a face-centred symmetry
# plane and a slip wall, and a Cartesian line between two slip walls), `pulse`
# (a pressure and density bump at rest on the low end) or `implode` (a uniform
# gas moving toward the low end with u = −U sin(πx), which compresses it onto
# the axis and rebounds). Each entry of `variants` is `filter:art`, the default
# state filter and the default artificial properties each on or off. The filter
# runs through a callback with the solver's own pass off, as in
# bench/filter_conservation.jl, so its share of each total is tallied
# separately. Every run prints the relative change of the total mass
# and energy over the run under three quadratures, with the filter's share of
# each: the diagnostics' (`volume_integral`: trapezoid weights, a full weight
# at a fold, each edge node's weight corrected for the metric's share of the
# edge term as in src/diagnostics.jl), W J, and W J less (h²/24) q_1 at a
# cylindrical axis. The fold side of W J is the midpoint rule on f = r q, whose
# Euler–Maclaurin error is (h²/24) f'(0) = (h²/24) q(0), and its time
# derivative at q = ρ is −(h²/24) ∇·(ρu)(0) = −h² ρ u_r'(0)/12, the axis face;
# the corrected total
# removes that term. The mass table also prints the time integral of the axis
# face from the state at each step (trapezoid rule in time), which under W J
# is the whole right-hand-side part of the mass change.
#
# Scratch tooling, like everything else in bench/: it prints tables and asserts
# nothing. The measurements are in reference/CALIBRATION_APPENDIX.md under
# "Fold order and geometry limits".

using CompactLES
const CL = CompactLES
using Printf
using Random: Xoshiro

const OPTS = CL.script_args(ARGS, (parts="quadrature,face,vortex,runs", N="64,128,256",
                                   geometries="rz,sphere,plane,walls",
                                   cases="pulse,implode",
                                   variants="on:on,off:off,on:off,off:on",
                                   tfinal=1.0, cfl=0.5, amplitude=0.5, ntheta=64,
                                   nmax=20_000))

const GAMMA = 1.4
printf(fmt::String, args...) = print(Printf.format(Printf.Format(fmt), args...))

# --- geometry ---------------------------------------------------------------

const PERIODIC = (PeriodicBC(), PeriodicBC())

function line_problem(geometry, ic)
    metric, lo, dom2 = geometry == "rz" ? (CylindricalMetric(), AxisBC(), (0.0, 1.0)) :
                       geometry == "sphere" ? (SphericalMetric(), OriginBC(),
                                               (π / 2, π / 2 + 1)) :
                       geometry == "plane" ? (CartesianMetric(), SymmetryPlaneBC(),
                                              (0.0, 1.0)) :
                       geometry == "walls" ? (CartesianMetric(), SlipWallBC(), (0.0, 1.0)) :
                       error("unknown geometry $geometry")
    return Problem(name=geometry, eos=IdealSpecies("gas"; R=1.0, gamma=GAMMA),
                   transport=ConstantTransport(mu0=0.0), metric=metric,
                   domain=((0.0, 1.0), dom2, (0.0, 1.0)),
                   bcs=((lo, SlipWallBC()), PERIODIC, PERIODIC), ic=ic)
end

# Node weights along dimension 1 under which the divergence telescopes: the
# positivity limiter's closed-line weights, or on a line folded at its low end
# h on the fold's half and the far-end tail of a closed line twice as long.
function line_weights(solver)
    decomp = solver.decomp
    N, h = decomp.n_global[1], Float64(solver.h[1])
    # The derivative scheme; a folded dimension holds its plans on the fold.
    scheme = getfield(solver, :schemes).deriv
    fold = solver.folds[1]
    if fold !== nothing && fold.lo
        tail = CL._limiter_weights(scheme, 2N, h, decomp.n_halo, Float64)[1][N+1:2N]
        return [i <= N ÷ 2 ? h : tail[i] for i in 1:N]
    end
    return CL._limiter_weights(scheme, N, h, decomp.n_halo, Float64)[1]
end

# Σ W_i w_j w_k J f over the interior, with w = h on a resolved periodic
# dimension and 1 on a collapsed one, which is the measure `volume_integral`
# gives those dimensions.
function weighted_total(solver, f, W)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    nx, ny, nz = decomp.n_local
    w(d) = decomp.active[d] ? Float64(solver.h[d]) : 1.0
    acc = 0.0
    for k in 1:nz, j in 1:ny, i in 1:nx
        I = CartesianIndex(i + o1, j + o2, k + o3)
        acc += W[i] * f[I] / solver.inv_J[I]
    end
    return acc * w(2) * w(3)
end

flux_sign(solver, c) = solver.folds[1] === nothing ? 1 : solver.folds[1].sigflux[c]

# The low face of a line: Ĝ_0 = g_N − Σ W (D g) on a folded line, and on a
# closed line the residual g_N − g_1 − Σ W (D g), both from the solver's
# divergence plan along dimension 1 (first line of the other dimensions).
function low_face(solver, g, W, σ, scratch)
    decomp = solver.decomp
    o1, o2, o3 = decomp.n_halo_d
    N = decomp.n_local[1]
    CL.div_along!(scratch, g, solver, 1, σ)
    s = sum(W[i] * scratch[i+o1, 1+o2, 1+o3] for i in 1:N)
    folded = solver.folds[1] !== nothing && solver.folds[1].lo
    first = folded ? 0.0 : g[1+o1, 1+o2, 1+o3]
    return g[N+o1, 1+o2, 1+o3] - first - s
end

# --- face: one right-hand side ----------------------------------------------

# The smooth converging profile of the design check: ρ(0) = 1.5, u_r'(0) = −1.
face_ic(x, y, z) = Prim(rho=1 + 0.5exp(-(x / 0.3)^2), u=(-x * exp(-(x / 0.4)^2), 0.0, 0.0),
                       p=1 + 0.2exp(-(x / 0.3)^2))

function face_part(Ns, geometries)
    println("\n=== the low face of the divergence, one right-hand side ===")
    println("profile rho = 1 + 0.5 exp(-(r/0.3)^2), u = -r exp(-(r/0.4)^2), " *
            "p = 1 + 0.2 exp(-(r/0.3)^2) on (0, 1]")
    println("prediction at the axis: mass -h^2 rho(0) u'(0)/12 = h^2/8, energy " *
            "-h^2 (E + p)(0) u'(0)/12 = 0.35 h^2 (per radian)")
    println("node-1 columns: the face over node 1's mass W_1 J_1 rho_1 and over its rate " *
            "W_1 J_1 drho_1/dt; corrected: the face less (h^2/24) dq_1/dt")
    println("geometry   N  weights  |D 1|    mass face  /predicted  /node-1 mass  " *
            "/node-1 rate   corrected   energy face  /predicted   corrected")
    for geometry in geometries, N in Ns
        solver, Q = setup(line_problem(geometry, face_ic),
                          Numerics(n_global=(N, 1, 1),
                                   art=ArtificialProperties(enabled=false)))
        decomp = solver.decomp
        o1, o2, o3 = decomp.n_halo_d
        h = Float64(solver.h[1])
        W = line_weights(solver)
        scratch = similar(solver.tmp_a)
        # The weights on random data of the odd parity (on a closed line, any
        # data), and the even folded divergence of a constant.
        g = zeros(size(scratch))
        g[o1+1:o1+N, :, :] .= randn(Xoshiro(N), N)
        folded = solver.folds[1] !== nothing
        odd = abs(low_face(solver, g, W, -1, scratch))
        fill!(g, 1.0)
        CL.div_along!(scratch, g, solver, 1, 1)
        dconst = folded ? maximum(abs, scratch[o1+1:o1+N, 1+o2, 1+o3]) : NaN
        # The right-hand side, and its faces.
        dQ = allocate_state(solver)
        fill!(parent(dQ), 0.0)
        CL.apply_bcs!(solver, Q)
        CL.compute_rhs!(solver, Q, dQ)
        iE = solver.equations.i_energy
        area = solver.area_d[1]
        first = CartesianIndex(1 + o1, 1 + o2, 1 + o3)
        last = CartesianIndex(N + o1, 1 + o2, 1 + o3)
        face(c) = weighted_total(solver, view(dQ, :, :, :, c), W) +
                  area[last] * solver.flux[1, c][last] -
                  (folded ? 0.0 : area[first] * solver.flux[1, c][first])
        mface, eface = face(1), face(iE)
        cell = W[1] / solver.inv_J[first]
        correction = geometry == "rz" ? h^2 / 24 : 0.0
        mpred, epred = geometry == "rz" ? (h^2 / 8, 0.35h^2) : (0.0, 0.0)
        ratio(a, b) = b == 0 ? NaN : a / b
        printf("%-7s %4d  %7.1e  %7.1e  %+11.3e  %8.4f   %10.4f   %10.4f  %+11.3e  " *
               "%+11.3e  %8.4f  %+11.3e\n", geometry, N, odd, dconst, mface,
               ratio(mface, mpred), mface / (cell * Q[first, 1]),
               mface / (cell * dQ[first, 1]), mface - correction * dQ[first, 1], eface,
               ratio(eface, epred), eface - correction * dQ[first, iE])
    end
end

# --- vortex: the resolved (r, θ) axis ---------------------------------------

const VBETA, VRC, VU, VX0 = 3.0, 0.15, 0.5, -0.4

function vortex(x, y, t)
    xi, eta = (x - VX0 - VU * t) / VRC, y / VRC
    f = exp(1 - xi^2 - eta^2)
    swirl = VBETA / (2pi) * sqrt(f)
    T = 1 - (GAMMA - 1) * VBETA^2 / (8GAMMA * pi^2) * f
    rho = T^(1 / (GAMMA - 1))
    return (; rho, p=rho * T, ux=VU - swirl * eta, uy=swirl * xi)
end

function vortex_prim(r, theta, t)
    v = vortex(r * cos(theta), r * sin(theta), t)
    s, c = sincos(theta)
    return Prim(rho=v.rho, p=v.p, u=(c * v.ux + s * v.uy, -s * v.ux + c * v.uy, 0.0))
end

# ∇·(ρu) at the origin by fourth-order central differences of the exact field.
function mass_divergence(t; δ=1e-3)
    fx(x) = (v = vortex(x, 0.0, t); v.rho * v.ux)
    fy(y) = (v = vortex(0.0, y, t); v.rho * v.uy)
    d(f) = (8(f(δ) - f(-δ)) - (f(2δ) - f(-2δ))) / (12δ)
    return d(fx) + d(fy)
end

function vortex_part(Ns, ntheta)
    println("\n=== the axis face on the resolved (r, θ) grid: the axis-crossing vortex ===")
    println("vortex of bench/axisvortex.jl at three centres, Dirichlet at r = 1; " *
            "prediction -(π h^2/12) div(ρu)(0) per unit length")
    println("  N  ntheta  centre   div(ρu)(0)   axis face    predicted    ratio")
    for N in Ns, t in (0.5, 0.7, 0.8)
        problem = Problem(name="vortex", eos=IdealSpecies("gas"; R=1.0, gamma=GAMMA),
                          transport=ConstantTransport(mu0=0.0), metric=CylindricalMetric(),
                          domain=((0.0, 1.0), (0.0, 2pi), (0.0, 1.0)),
                          bcs=((AxisBC(),
                                DirichletBC((r, θ, z, s) -> vortex_prim(r, θ, t))),
                               PERIODIC, PERIODIC),
                          ic=(r, θ, z) -> vortex_prim(r, θ, t))
        solver, Q = setup(problem, Numerics(n_global=(N, ntheta, 1),
                                            art=ArtificialProperties(enabled=false)))
        decomp = solver.decomp
        o1, o2, o3 = decomp.n_halo_d
        h, hθ = Float64(solver.h[1]), Float64(solver.h[2])
        W = line_weights(solver)
        dQ = allocate_state(solver)
        fill!(parent(dQ), 0.0)
        CL.apply_bcs!(solver, Q)
        CL.compute_rhs!(solver, Q, dQ)
        area = solver.area_d[1]
        far = sum(area[N+o1, j+o2, 1+o3] * solver.flux[1, 1][N+o1, j+o2, 1+o3]
                  for j in 1:ntheta) * hθ
        face = weighted_total(solver, view(dQ, :, :, :, 1), W) + far
        div0 = mass_divergence(t)
        pred = -π * h^2 / 12 * div0
        printf("%4d  %4d  %+6.2f  %+11.3e  %+11.3e  %+11.3e  %8.4f\n", N, ntheta,
               VX0 + VU * t, div0, face, pred, abs(pred) < 1e-12 ? NaN : face / pred)
    end
end

# --- runs: closed lines ------------------------------------------------------

pulse_ic(x, y, z) = (b = exp(-(x / 0.15)^2); Prim(rho=1 + 0.2b, u=(0.0, 0.0, 0.0),
                                                   p=1 + 0.5b))
implode_ic(U) = (x, y, z) -> Prim(rho=1.0, u=(-U * sin(π * x), 0.0, 0.0), p=1.0)

# The totals of mass and energy under three quadratures: the diagnostics'
# (`volume_integral`), W J, and W J with the midpoint rule's end correction at
# a cylindrical axis. The fold side of W J is the midpoint rule on f = J q,
# whose leading error is (h²/24) f'(0) by Euler–Maclaurin; f'(0) = q(0) for
# J = r and zero for J = r² or 1, so only the axis takes the correction, with
# q(0) read as q_1 (the difference is O(h⁴) in the total).
function totals(solver, Q, W, correction)
    o1, o2, o3 = solver.decomp.n_halo_d
    out = zeros(6)
    for (k, c) in enumerate((1, solver.equations.i_energy))
        q = view(Q, :, :, :, c)
        out[3k-2] = volume_integral(solver, q)
        out[3k-1] = weighted_total(solver, q, W)
        out[3k] = out[3k-1] - correction * Q[1+o1, 1+o2, 1+o3, c]
    end
    return out
end

mutable struct Tally
    filter::Vector{Float64}     # the filter's change of each total
    axis::Float64               # ∫ Ĝ_0 dt, mass
    rate::Float64               # Ĝ_0 at the last state
end

# The axis face of the mass from the state: g = A₁ ρu_r, the mass flux of a
# single inviscid species.
function state_face(solver, Q, W, g, scratch)
    g .= solver.area_d[1] .* view(Q, :, :, :, solver.equations.i_mom[1])
    return low_face(solver, g, W, flux_sign(solver, 1), scratch)
end

# Per step: the axis face at the end of the step (trapezoid rule in time), the
# filter pass and its change of the totals, and the face of the filtered state
# the next step starts from.
function tallying(tally, W, correction, filtering, g, scratch)
    return (solver, Q) -> begin
        rate = state_face(solver, Q, W, g, scratch)
        tally.axis += solver.dt_prev * (tally.rate + rate) / 2
        if filtering
            before = totals(solver, Q, W, correction)
            solver.filter_interval = 1
            CL.filter_state!(solver, Q)
            solver.filter_interval = 0
            tally.filter .+= totals(solver, Q, W, correction) .- before
            rate = state_face(solver, Q, W, g, scratch)
        end
        tally.rate = rate
        nothing
    end
end

function run_case(geometry, case, N, filtering, art, o)
    ic = case == "pulse" ? pulse_ic : case == "implode" ? implode_ic(o.amplitude) :
         error("unknown case $case")
    numerics = Numerics(n_global=(N, 1, 1), cfl=o.cfl,
                        filter=StateFilter(interval=0),
                        art=ArtificialProperties(enabled=art))
    solver, Q = setup(line_problem(geometry, ic), numerics)
    W = line_weights(solver)
    correction = geometry == "rz" ? Float64(solver.h[1])^2 / 24 : 0.0
    g, scratch = similar(solver.tmp_a), similar(solver.tmp_a)
    fill!(g, 0.0)
    start = totals(solver, Q, W, correction)
    tally = Tally(zeros(6), 0.0, state_face(solver, Q, W, g, scratch))
    status = "done"
    try
        run!(solver, Q; tfinal=o.tfinal, nmax=o.nmax,
             callback=Callback(EveryStep(1),
                               tallying(tally, W, correction, filtering, g, scratch)))
    catch err
        err isa SolverFailure || rethrow()
        status = "failed"
    end
    stop = totals(solver, Q, W, correction)
    change = (stop .- start) ./ start
    filter = tally.filter ./ start
    rhs = change .- filter
    axis = tally.axis / start[2]
    gap = abs(axis) < 1e-13 ? NaN : (rhs[2] - axis) / abs(axis)
    return (; geometry, case, N, filtering, art, status, t=solver.t, change, filter, rhs,
            axis, gap)
end

label(r) = @sprintf("%-7s %-7s %4d  %-3s %-3s %-6s %5.3f", r.geometry, r.case, r.N,
                    r.filtering ? "on" : "off", r.art ? "on" : "off", r.status, r.t)

function runs_part(Ns, geometries, o)
    variants = [Tuple(v == "on" for v in split(s, ':')) for s in split(o.variants, ',')]
    results = [run_case(geometry, case, N, filtering, art, o)
               for case in split(o.cases, ','), geometry in geometries,
                   (filtering, art) in variants, N in Ns]
    println("\n=== closed runs: relative change of the totals over the run ===")
    printf("tfinal %.2f, cfl %.2f, implode amplitude %.2f; each change relative to the " *
           "initial total; rhs = change - filter\n", o.tfinal, o.cfl, o.amplitude)
    for (k, quantity) in enumerate(("mass", "energy"))
        println("\n$quantity: volume_integral, W J, and W J with the axis " *
                "end correction" * (k == 1 ? "; ∫axis dt the axis face over the run" : ""))
        println("geometry case       N  flt art status    t     vol. int.    filter" *
                "       W J    filter       rhs  corrected      rhs" *
                (k == 1 ? "   ∫axis dt  rel. gap" : ""))
        for r in permutedims(results, (4, 3, 2, 1))[:]
            i = 3k - 2
            printf("%s  %+10.2e %+9.2e  %+10.2e %+9.2e %+9.2e  %+10.2e %+9.2e",
                   label(r), r.change[i], r.filter[i], r.change[i+1], r.filter[i+1],
                   r.rhs[i+1], r.change[i+2], r.rhs[i+2])
            k == 1 && printf("  %+10.2e %+9.1e", r.axis, r.gap)
            println()
        end
    end
end

# --- quadrature: manufactured integrals ---------------------------------------

const REST = (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1.0)

# (label, metric, domain, bcs, stretch, θ resolved, q(x1, x2), exact ∫ q dV)
const QUADRATURE_CASES = [
    ("axis, q = 2 + cos πr", CylindricalMetric(), ((0.0, 1.0), (0.0, 1.0)),
     (AxisBC(), SlipWallBC()), nothing, false, (r, t) -> 2 + cos(π * r), 1 - 2 / π^2),
    ("axis, q = exp(-r²)", CylindricalMetric(), ((0.0, 1.0), (0.0, 1.0)),
     (AxisBC(), SlipWallBC()), nothing, false, (r, t) -> exp(-r^2), (1 - exp(-1)) / 2),
    ("annulus, q = 2 + cos π(r - ½)", CylindricalMetric(), ((0.5, 1.5), (0.0, 1.0)),
     (SlipWallBC(), SlipWallBC()), nothing, false, (r, t) -> 2 + cos(π * (r - 0.5)),
     2 - 2 / π^2),
    ("origin, q = 2 + cos πr", SphericalMetric(), ((0.0, 1.0), (π / 2, π / 2 + 1)),
     (OriginBC(), SlipWallBC()), nothing, false, (r, t) -> 2 + cos(π * r),
     2 / 3 - 2 / π^2),
    ("poles, q = 2 + cos π(r - ½) cos 2θ", SphericalMetric(), ((0.5, 1.5), (0.0, π)),
     (SlipWallBC(), SlipWallBC()), nothing, true,
     (r, t) -> 2 + cos(π * (r - 0.5)) * cos(2t), 13 / 3 + 8 / (3π^2)),
    ("stretched, q = 2 + cos πx", CartesianMetric(), ((0.0, 1.0), (0.0, 1.0)),
     (SlipWallBC(), SlipWallBC()), sine_cluster(0.0, 1.0, 0.3, 0.4), false,
     (x, t) -> 2 + cos(π * x), 2.0),
    ("stretched, q = exp(x)", CartesianMetric(), ((0.0, 1.0), (0.0, 1.0)),
     (SlipWallBC(), SlipWallBC()), sine_cluster(0.0, 1.0, 0.3, 0.4), false,
     (x, t) -> exp(x), exp(1) - 1),
]

# `volume_integral` of a smooth q against its exact value, on N nodes along the
# first dimension (and N along a resolved θ between two poles).
function quadrature_part(Ns)
    println("\n=== volume_integral of smooth fields: error and order by refinement ===")
    println("case                                   N     error      order")
    for (label, metric, (dom1, dom2), bcs1, stretch, polar, q, exact) in QUADRATURE_CASES
        previous = nothing
        for N in Ns
            problem = Problem(name="quadrature", eos=IdealSpecies("gas"; R=1.0, gamma=GAMMA),
                              metric=metric, domain=(dom1, dom2, (0.0, 1.0)),
                              bcs=(bcs1, polar ? (PoleBC(), PoleBC()) : PERIODIC,
                                   PERIODIC), ic=REST)
            solver, _ = setup(problem, Numerics(n_global=(N, polar ? N : 1, 1),
                                                stretch=(stretch, nothing, nothing),
                                                art=ArtificialProperties(enabled=false)))
            decomp = solver.decomp
            o1, o2, o3 = decomp.n_halo_d
            f = zeros(size(solver.inv_J))
            for j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
                f[i+o1, j+o2, 1+o3] = q(CL.xcoord(solver, 1, i), CL.xcoord(solver, 2, j))
            end
            e = volume_integral(solver, f) - exact
            printf("%-36s %4d  %+.3e  %6.2f\n", label, N, e,
                   previous === nothing ? NaN : log(previous[1] / abs(e)) /
                                                log(N / previous[2]))
            previous = (abs(e), N)
        end
    end
end

function main(o)
    Ns = parse.(Int, split(o.N, ','))
    geometries = split(o.geometries, ',')
    parts = split(o.parts, ',')
    for (name, part) in (("quadrature", () -> quadrature_part(Ns)),
                         ("face", () -> face_part(Ns, geometries)),
                         ("vortex", () -> vortex_part([n ÷ 2 for n in Ns], o.ntheta)),
                         ("runs", () -> runs_part(Ns, geometries, o)))
        name in parts || continue
        printf("(%s: %.0f s)\n", name, @elapsed part())
    end
end

main(OPTS)
