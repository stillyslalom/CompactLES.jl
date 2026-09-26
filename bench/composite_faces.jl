# Composite faces (CompositeBC): the two flows that motivated them.
#
#   julia --project=. -t 4 bench/composite_faces.jl [jet slots]
#   julia --project=. -t 4 bench/composite_faces.jl jet N=49,97
#   julia --project=. -t 4 bench/composite_faces.jl slots tfinal=192 sample=16
#
# Parts:
#
#   jet    a jet entering a closed box through an orifice in its lower wall,
#          the rest of that face a no-slip wall. The orifice is a
#          characteristic inflow (NSCBCInflowBC) or a Dirichlet stream
#          (DirichletBC) whose velocity falls to zero at the orifice edge and
#          ramps up over `tau`. Reports the mass gained by the box against
#          the time integral of the mass flux through the orifice, per
#          resolution, with and without the artificial properties and the
#          filter.
#   slots  a heavy gas fed from the top of a vertical box and a light gas
#          from the bottom, at equal momentum flux, venting through a slot in
#          each side wall at the interface height (NSCBCOutflowBC in a slip
#          wall). Reports, every `sample` time units, the interface height
#          (where the plane-averaged heavy mass fraction crosses one half),
#          its change since the last sample, the relative L2 change of the
#          density since the last sample, and the largest |dρ/dt| of the
#          right-hand side with its location. The last does not decay: at
#          the slot edges it stays near 1 while the density there stops
#          changing, so the state change is the steadiness measure.
#
# Settings (`key=value`): N (comma-separated resolutions for jet), tfinal and
# sample (for slots), U (the heavy stream's speed; the light stream takes
# U·√5), ws (slot half-height), sigma (outflow relaxation), eta (inflow
# relaxation rates), mu (viscosity).
#
# Scratch tooling, like everything else in bench/: it prints tables and
# asserts nothing. `test/composite_face_tests.jl` holds the short versions.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES

const OPTS = CL.script_args(filter(a -> occursin('=', a), ARGS),
    (N="49,97", tfinal=192.0, sample=16.0, U=0.05, ws=0.3, sigma=2.0, eta=2.0,
     mu=1e-3))
const PARTS = let names = filter(a -> !occursin('=', a), ARGS)
    isempty(names) ? ["jet", "slots"] : names
end
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("composite_faces.jl runs serially")

function jet_case(n, inflow, art)
    U, w, xc, tau = 0.3, 0.1, 0.5, 0.1
    ramp(t) = t < tau ? sinpi(t / (2tau))^2 : 1.0
    profile(x) = abs(x - xc) < w ? U * cospi((x - xc) / (2w))^2 : 0.0
    orifice = inflow === :nscbc ?
        NSCBCInflowBC(u=(0.0, U, 0.0), T_ion=1.0, eta_u=2.0, eta_T=2.0, eta_t=2.0,
                      target=(x, y, z, t) -> Prim(u=(0.0, ramp(t) * profile(x), 0.0),
                                                  T_ion=1.0, rho=1.0)) :
        DirichletBC((x, y, z, t) -> Prim(u=(0.0, ramp(t) * profile(x), 0.0),
                                         rho=1.0, p=1.0))
    wall = NoSlipWallBC()
    face = CompositeBC((wall, orifice), (x, y, z) -> abs(x - xc) < w ? 2 : 1)
    s = Solver(n_global=(n, n, 1), L_domain=(1.0, 1.0, 1.0),
               bcs=((wall, wall), (face, wall), PeriodicBC()),
               transport=Transport(mu0=2e-3), art=ArtParams(enabled=art),
               filter_interval=art ? 1 : 0)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0))
    mass() = volume_integral(s, Array(view(Q, :, :, :, 1)))
    M0 = mass()
    o1, o2 = s.decomp.n_halo_d[1], s.decomp.n_halo_d[2]
    m2 = s.equations.i_mom[2]
    h = 1 / (n - 1)
    face_flux() = sum((i in (1, n) ? 0.5 : 1.0) * h * Q[i + o1, o2 + 1, 1, m2]
                      for i in 1:n)
    injected = Ref(0.0)
    last = Ref((0.0, face_flux()))
    tally = Callback(EveryStep(1), (solver, _) -> begin
        f = face_flux()
        injected[] += 0.5 * (solver.t - last[][1]) * (f + last[][2])
        last[] = (solver.t, f)
    end)
    run!(s, Q; tfinal=0.5, nmax=10_000, callback=tally)
    gain = mass() - M0
    return s.step, gain, injected[]
end

function jet()
    println("\njet through a walled orifice into a closed box, t = 0.5")
    @printf("  %-9s %5s %-10s %6s %12s %12s %11s\n", "orifice", "N", "art+filter",
            "steps", "mass gain", "injected", "rel defect")
    for inflow in (:nscbc, :dirichlet), n in parse.(Int, split(OPTS.N, ',')),
        art in (true, false)
        steps, gain, injected = jet_case(n, inflow, art)
        @printf("  %-9s %5d %-10s %6d %12.5e %12.5e %11.3e\n", inflow, n, art,
                steps, gain, injected, (gain - injected) / injected)
    end
end

function slots()
    U, ws = OPTS.U, OPTS.ws
    eos = IdealMixture([IdealSpecies{Float64}("heavy", 0.2, 1.1),
                        IdealSpecies{Float64}("light", 1.0, 1.4)])
    stream(v, Y) = NSCBCInflowBC(u=(0.0, v, 0.0), T_ion=1.0, Y=Y, eta_u=OPTS.eta,
                                 eta_T=OPTS.eta, eta_t=OPTS.eta, eta_Y=OPTS.eta)
    top = stream(-U, [1.0, 0.0])
    bottom = stream(U * sqrt(5.0), [0.0, 1.0])  # ρ_heavy/ρ_light = 5 at T = p = 1
    side = CompositeBC((SlipWallBC(), NSCBCOutflowBC(pinf=1.0, sigma=OPTS.sigma)),
                       (x, y, z) -> abs(y - 1) < ws ? 2 : 1)
    s = Solver(n_global=(33, 65, 1), L_domain=(1.0, 2.0, 1.0),
               bcs=((side, side), (bottom, top), PeriodicBC()), eos=eos,
               transport=Transport(mu0=OPTS.mu), art=ArtParams(enabled=true))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> begin
        θ = 0.5 * (1 + tanh((y - 1) / 0.05))
        Prim(Y=(θ, 1 - θ), T_ion=1.0, p=1.0)
    end)
    y = profile_coordinate(s, 2)
    inner = CL.interior(s.decomp)
    o = s.decomp.n_halo_d
    density() = Array(view(Q, :, :, :, 1)) .+ Array(view(Q, :, :, :, 2))
    function interface()
        Yh = Array(view(Q, :, :, :, 1)) ./ density()
        profile = plane_profile(s, Yh, 2)
        k = findfirst(>=(0.5), profile)
        (k === nothing || k == 1) && return NaN
        return y[k-1] + (0.5 - profile[k-1]) * (y[k] - y[k-1]) /
               (profile[k] - profile[k-1])
    end
    println("\ntwo-slot stagnation plane, 33 × 65, U = $U, slot half-height $ws")
    @printf("  %7s %10s %11s %13s %12s  %s\n", "t", "y_i", "Δy_i", "rel L2 Δρ",
            "max|dρ/dt|", "at (i, j)")
    y_prev, rho_prev = interface(), density()
    t = 0.0
    while t < OPTS.tfinal - 1e-12
        run!(s, Q; tfinal=min(t + OPTS.sample, OPTS.tfinal), nmax=10^6)
        t = s.t
        dQ = zero(Q)
        apply_bcs!(s, Q)
        compute_rhs!(s, Q, dQ)
        rate(I) = abs(dQ[I, 1] + dQ[I, 2])
        at = argmax(rate, collect(inner))
        rho = density()
        change = sqrt(sum(abs2, (rho .- rho_prev)[inner]) / sum(abs2, rho[inner]))
        yi = interface()
        @printf("  %7.1f %10.5f %11.3e %13.3e %12.3e  %s\n", t, yi, yi - y_prev,
                change, rate(at), Tuple(at) .- o)
        y_prev, rho_prev = yi, rho
    end
end

function main()
    "jet" in PARTS && jet()
    "slots" in PARTS && slots()
end
main()
