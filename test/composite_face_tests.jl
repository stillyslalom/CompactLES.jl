# Composite faces: one face divided among several conditions by a mask of face
# coordinates. Included by serial_suite.jl; runs standalone as
#   julia --project=. test/composite_face_tests.jl

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using Test

const CL = CompactLES

@testset "composite face: construction and setup checks" begin
    wall = SlipWallBC()
    @test_throws ArgumentError CompositeBC((), (x, y, z) -> 1)
    @test_throws ArgumentError CompositeBC((wall, PeriodicBC()), (x, y, z) -> 1)
    @test_throws ArgumentError CompositeBC((wall, AxisBC()), (x, y, z) -> 1)
    @test_throws ArgumentError CompositeBC((wall, SwitchableBC(wall, ExtrapolationBC())),
                                           (x, y, z) -> 1)
    inner = CompositeBC((wall,), (x, y, z) -> 1)
    @test_throws ArgumentError CompositeBC((wall, inner), (x, y, z) -> 1)
    @test !isperiodic(inner)
    # A composite face may itself be switched.
    @test SwitchableBC(inner, ExtrapolationBC()) isa SwitchableBC

    # The detector mirror holds only where every member is a wall.
    @test sensor_mirror(CompositeBC((SlipWallBC(), NoSlipWallBC()), (x, y, z) -> 1))
    @test !sensor_mirror(CompositeBC((SlipWallBC(), NSCBCOutflowBC(pinf=1.0)),
                                     (x, y, z) -> 1))

    # Members are validated at setup on the face they sit on.
    bad = CompositeBC((wall, NSCBCInflowBC(u=(0.1, 0, 0), T_ion=1.0, Y=[0.5, 0.5])),
                      (x, y, z) -> 1)
    @test_throws ErrorException Solver(n_global=(16, 16, 1), L_domain=(1.0, 1.0, 1.0),
                                       bcs=(bad, (wall, wall), PeriodicBC()))

    # A selector value outside the member range is an error at first use.
    out_of_range = CompositeBC((wall, wall), (x, y, z) -> 3)
    s = Solver(n_global=(16, 16, 1), L_domain=(1.0, 1.0, 1.0),
               bcs=((out_of_range, wall), (wall, wall), PeriodicBC()))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0))
    @test_throws ErrorException apply_bcs!(s, Q)
end

# A composite whose mask selects one member reproduces that member alone
# bitwise, although the other member runs on the whole plane (and, for the
# characteristic outflow, its distributed solves) before being discarded.
@testset "composite face: a uniform mask reproduces its member" begin
    eos = IdealMixture([IdealSpecies{Float64}("heavy", 0.2, 1.1),
                        IdealSpecies{Float64}("light", 1.0, 1.4)])
    ic(x, y, z) = begin
        θ = 0.5 * (1 + tanh((y - 1) / 0.1))
        Prim(Y=(θ, 1 - θ), T_ion=1.0, p=1.0 + 0.1 * sin(3y) * cos(2x),
             u=(0.1 * sin(3y), 0.05 * cos(2y + x), 0.0))
    end
    function run(side)
        top = NSCBCInflowBC(u=(0.0, -0.1, 0.0), T_ion=1.0, Y=[1.0, 0.0])
        bottom = NSCBCInflowBC(u=(0.0, 0.2, 0.0), T_ion=1.0, Y=[0.0, 1.0])
        s = Solver(n_global=(17, 25, 1), L_domain=(1.0, 2.0, 1.0),
                   bcs=((side, side), (bottom, top), PeriodicBC()), eos=eos,
                   transport=Transport(mu0=1e-3), art=ArtParams(enabled=false))
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        run!(s, Q; tfinal=1.0, nmax=10)
        return parent(Q)
    end
    outflow = NSCBCOutflowBC(pinf=1.0)
    @test run(CompositeBC((SlipWallBC(), outflow), (x, y, z) -> 2)) == run(outflow)
    @test run(CompositeBC((outflow, SlipWallBC()), (x, y, z) -> 2)) == run(SlipWallBC())
end

# A jet enters a closed box through an orifice in its lower wall. The orifice
# is a characteristic inflow whose target velocity falls to zero at the edge of
# the orifice, and the rest of the face is a no-slip wall. The mass in the box
# grows by the mass carried through the orifice.
#
# The budget does not close to round-off. The node-centred trapezoid is not the
# compact divergence's conservation norm, and the characteristic inflow
# corrects the right-hand side at the orifice nodes rather than the flux
# through them. Both are spatial discretization errors: the defect is
# unchanged to four digits when the step is halved or quartered, so the time
# integration of the face flux contributes nothing measurable.
@testset "composite face: jet through a walled orifice into a closed box" begin
    n, U, w, xc, tau = 49, 0.3, 0.1, 0.5, 0.1
    ramp(t) = t < tau ? sinpi(t / (2tau))^2 : 1.0
    profile(x) = abs(x - xc) < w ? U * cospi((x - xc) / (2w))^2 : 0.0
    orifice = NSCBCInflowBC(u=(0.0, U, 0.0), T_ion=1.0, eta_u=2.0, eta_T=2.0,
                            eta_t=2.0,
                            target=(x, y, z, t) -> Prim(u=(0.0, ramp(t) * profile(x), 0.0),
                                                        T_ion=1.0, rho=1.0))
    wall = NoSlipWallBC()
    face = CompositeBC((wall, orifice), (x, y, z) -> abs(x - xc) < w ? 2 : 1)
    s = Solver(n_global=(n, n, 1), L_domain=(1.0, 1.0, 1.0),
               bcs=((wall, wall), (face, wall), PeriodicBC()),
               transport=Transport(mu0=2e-3), art=ArtParams(enabled=true))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0))
    mass() = volume_integral(s, Array(view(Q, :, :, :, 1)))
    M0 = mass()
    o1, o2 = s.decomp.n_halo_d[1], s.decomp.n_halo_d[2]
    m2 = s.equations.i_mom[2]
    h = 1 / (n - 1)
    # Trapezoid in x of the normal mass flux ρv on the face, and its trapezoid
    # integral in time over the steps.
    face_flux() = sum((i in (1, n) ? 0.5 : 1.0) * h * Q[i + o1, o2 + 1, 1, m2]
                      for i in 1:n)
    injected = Ref(0.0)
    last = Ref((0.0, face_flux()))
    tally = Callback(EveryStep(1), (solver, _) -> begin
        f = face_flux()
        injected[] += 0.5 * (solver.t - last[][1]) * (f + last[][2])
        last[] = (solver.t, f)
    end)
    run!(s, Q; tfinal=0.5, nmax=400, callback=tally)
    @test s.t == 0.5
    gain = mass() - M0
    # Measured 3.45e-3 relative.
    @test injected[] > 3e-3
    @test abs(gain - injected[]) < 1e-2 * injected[]

    # The wall part of the face holds no normal momentum and passes no mass;
    # the orifice carries the jet.
    apply_bcs!(s, Q)
    compute_rhs!(s, Q, zero(Q))
    plane = CL.wallplane(s.decomp, 2, 1)
    wall_points = [I for I in plane if abs(xcoord(s, 1, I[1] - o1) - xc) >= w]
    jet_points = [I for I in plane if abs(xcoord(s, 1, I[1] - o1) - xc) < w / 2]
    @test all(I -> Q[I, m2] == 0 && s.flux[2, 1][I] == 0, wall_points)
    @test all(I -> Q[I, m2] > 0 && s.flux[2, 1][I] > 0, jet_points)
end

# Two opposed streams, a heavy gas from the top and a light gas from the
# bottom, meet at a stagnation plane and leave through a slot in each side
# wall at the interface height. The slots are characteristic outflows in a
# slip wall. This is a short run; `bench/composite_faces.jl` runs the case to
# a steady interface.
@testset "composite face: two-slot stagnation plane, short run" begin
    eos = IdealMixture([IdealSpecies{Float64}("heavy", 0.2, 1.1),
                        IdealSpecies{Float64}("light", 1.0, 1.4)])
    U, ws = 0.05, 0.3
    stream(v, Y) = NSCBCInflowBC(u=(0.0, v, 0.0), T_ion=1.0, Y=Y, eta_u=2.0,
                                 eta_T=2.0, eta_t=2.0, eta_Y=2.0)
    top = stream(-U, [1.0, 0.0])
    bottom = stream(U * sqrt(5.0), [0.0, 1.0])
    side = CompositeBC((SlipWallBC(), NSCBCOutflowBC(pinf=1.0, sigma=2.0)),
                       (x, y, z) -> abs(y - 1) < ws ? 2 : 1)
    s = Solver(n_global=(17, 33, 1), L_domain=(1.0, 2.0, 1.0),
               bcs=((side, side), (bottom, top), PeriodicBC()), eos=eos,
               transport=Transport(mu0=1e-3), art=ArtParams(enabled=true))
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> begin
        θ = 0.5 * (1 + tanh((y - 1) / 0.05))
        Prim(Y=(θ, 1 - θ), T_ion=1.0, p=1.0)
    end)
    run!(s, Q; tfinal=8.0, nmax=2000)
    @test s.t == 8.0
    apply_bcs!(s, Q)
    CL.refresh_primitives!(s, Q)
    o1, o2 = s.decomp.n_halo_d[1], s.decomp.n_halo_d[2]
    nx, ny = s.decomp.n_local[1], s.decomp.n_local[2]
    slot(j) = abs(xcoord(s, 2, j) - 1) < ws
    u_face(i, j) = s.u[CartesianIndex(o1 + i, o2 + j, 1)]
    # Gas leaves through both slots and not through the walls beside them.
    @test all(j -> !slot(j) || u_face(1, j) < 0, 1:ny)
    @test all(j -> !slot(j) || u_face(nx, j) > 0, 1:ny)
    @test all(j -> slot(j) || u_face(1, j) == 0 && u_face(nx, j) == 0, 1:ny)
    # The interface, where the plane-averaged heavy fraction crosses one half,
    # lies within the slots.
    Yh = Array(view(Q, :, :, :, 1)) ./ (Array(view(Q, :, :, :, 1)) .+
                                        Array(view(Q, :, :, :, 2)))
    profile = plane_profile(s, Yh, 2)
    y = profile_coordinate(s, 2)
    k = findfirst(>=(0.5), profile)
    y_interface = y[k-1] + (0.5 - profile[k-1]) * (y[k] - y[k-1]) /
                  (profile[k] - profile[k-1])
    @test abs(y_interface - 1) < ws
end
