using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: compute_rhs!, apply_bcs!, CPUBackend, padded_index
using Test

struct SpacingIC end
(::SpacingIC)(x, y, z, h) = Prim(u=(h, 0, 0), p=1.0, rho=1.0)

@testset "spacing-aware pointwise callbacks" begin
    walls = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())

    # The helper uses the current patch's computational spacing and turns it
    # into a physical length with the same local metric factors as the RHS.
    cart = Solver(n_global=(11, 1, 1), L_domain=(1.0, 1.0, 1.0),
                  bcs=(walls, per, per), art=ArtificialProperties(enabled=false))
    Ic = padded_index(cart, 6, 1, 1)
    @test CompactLES.point_spacing(cart, Ic) == 0.1

    st = sine_cluster(0.0, 1.0, 0.5, 0.4)
    stretched = Solver(n_global=(11, 1, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=(walls, per, per), stretch=(st, nothing, nothing),
                       art=ArtificialProperties(enabled=false))
    Is = padded_index(stretched, 6, 1, 1)
    @test CompactLES.point_spacing(stretched, Is) ≈
          stretched.h[1] / stretched.inv_h[1][Is]

    cyl = Solver(n_global=(11, 12, 1), L_domain=(1.0, 2π, 1.0),
                 metric=CylindricalMetric(), bcs=((AxisBC(), SlipWallBC()), per, per),
                 art=ArtificialProperties(enabled=false))
    Icy = padded_index(cyl, 1, 7, 1)
    expected = minimum(cyl.h[d] / cyl.inv_h[d][Icy] for d in 1:3
                       if cyl.decomp.active[d])
    @test CompactLES.point_spacing(cyl, Icy) ≈ expected
    @test cyl.h[2] / cyl.inv_h[2][Icy] < cyl.h[1] / cyl.inv_h[1][Icy]

    collapsed = Solver(n_global=(1, 1, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=(per, per, per), art=ArtificialProperties(enabled=false))
    @test CompactLES.point_spacing(collapsed, padded_index(collapsed, 1, 1, 1)) == 0

    amr = Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0),
                 bcs=(walls, per, per), refine=BlockRegion((16, 0, 0), (16, 1, 1)),
                 art=ArtificialProperties(enabled=false))
    root = CompactLES.PatchSolver(amr, amr.patches[1])
    fine = CompactLES.PatchSolver(amr, amr.patches[2])
    @test CompactLES.point_spacing(fine, padded_index(fine, 5, 1, 1)) ≈
          CompactLES.point_spacing(root, padded_index(root, 5, 1, 1)) / 3

    short = (x, y, z) -> Prim(u=(x + y + z, 0, 0), p=1.0, rho=1.0)
    long = (x, y, z, h) -> Prim(u=(x + y + z, 0, 0), p=1.0, rho=1.0)
    Qshort = allocate_state(cart)
    Qlong = allocate_state(cart)
    initialize!(cart, Qshort, short)
    initialize!(cart, Qlong, long)
    @test Qshort == Qlong

    Qh = allocate_state(cart)
    initialize!(cart, Qh, (x, y, z, h) -> Prim(u=(h, 0, 0), p=1.0, rho=1.0))
    @test Qh[padded_index(cart, 6, 1, 1), cart.equations.i_mom[1]] ≈ 0.1

    Qs = allocate_state(stretched)
    initialize!(stretched, Qs, SpacingIC())
    @test Qs[Is, stretched.equations.i_mom[1]] ≈ CompactLES.point_spacing(stretched, Is)

    Qc = allocate_state(cyl)
    initialize!(cyl, Qc, SpacingIC())
    @test Qc[Icy, cyl.equations.i_mom[1]] ≈ CompactLES.point_spacing(cyl, Icy)

    Qzero = allocate_state(collapsed)
    initialize!(collapsed, Qzero, SpacingIC())
    @test Qzero[padded_index(collapsed, 1, 1, 1), collapsed.equations.i_mom[1]] == 0

    @test_throws ErrorException initialize!(cart, allocate_state(cart),
                                             (x, y, z, h) -> error("user failure"))
    @test_throws MethodError initialize!(cart, allocate_state(cart),
                                          (x, y, z, h) -> sin("user failure"))

    dbc_short = DirichletBC((x, y, z, t) -> Prim(u=(x, 0, 0), p=1.0, rho=1.0))
    dbc_long = DirichletBC((x, y, z, t, h) -> Prim(u=(x, 0, 0), p=1.0, rho=1.0))
    drive(bc; backend=CPUBackend()) = Solver(n_global=(11, 1, 1),
                                               L_domain=(1.0, 1.0, 1.0),
                                               bcs=((bc, SlipWallBC()), per, per),
                                               backend=backend,
                                               art=ArtificialProperties(enabled=false))
    driven_short, driven_long = drive(dbc_short), drive(dbc_long)
    Qdshort, Qdlong = allocate_state(driven_short), allocate_state(driven_long)
    initialize!(driven_short, Qdshort, short)
    initialize!(driven_long, Qdlong, short)
    apply_bcs!(driven_short, Qdshort)
    apply_bcs!(driven_long, Qdlong)
    @test Qdshort == Qdlong

    dbc_h = DirichletBC((x, y, z, t, h) -> Prim(u=(h, 0, 0), p=1.0, rho=1.0))
    driven_h = drive(dbc_h)
    Qdh = allocate_state(driven_h)
    initialize!(driven_h, Qdh, short)
    apply_bcs!(driven_h, Qdh)
    Id = padded_index(driven_h, 1, 1, 1)
    @test Qdh[Id, driven_h.equations.i_mom[1]] ≈ CompactLES.point_spacing(driven_h, Id)

    device = drive(dbc_h; backend=DeviceBackend(CompactLES.KernelAbstractions.CPU()))
    Qdevice = allocate_state(device)
    initialize!(device, Qdevice, short)
    apply_bcs!(device, Qdevice)
    @test Array(parent(Qdevice)) == parent(Qdh)

    seen_h = Ref(0.0)
    target = (x, y, z, t, h) -> begin
        seen_h[] = h
        Prim(u=(0.05, 0.0, 0.0), rho=1.0, T_ion=1.0)
    end
    target_short = (x, y, z, t) -> Prim(u=(0.05, 0.0, 0.0), rho=1.0, T_ion=1.0)
    nscbc_solver(target) = begin
        inflow = NSCBCInflowBC(u=(0.05, 0.0, 0.0), T_ion=1.0, Y=[1.0], target=target)
        Solver(n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
               bcs=((inflow, NSCBCOutflowBC(pinf=1.0)), per, per),
               art=ArtificialProperties(enabled=false))
    end
    nscbc, nscbc_short = nscbc_solver(target), nscbc_solver(target_short)
    Qn, Qnshort = allocate_state(nscbc), allocate_state(nscbc_short)
    init_nscbc = (x, y, z) -> Prim(u=(0.05, 0, 0), p=1.0, rho=1.0)
    initialize!(nscbc, Qn, init_nscbc)
    initialize!(nscbc_short, Qnshort, init_nscbc)
    dQ, dQshort = zero(Qn), zero(Qnshort)
    compute_rhs!(nscbc, Qn, dQ)
    compute_rhs!(nscbc_short, Qnshort, dQshort)
    @test seen_h[] > 0
    @test dQ == dQshort
end
