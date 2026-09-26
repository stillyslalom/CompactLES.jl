using CompactLES, Test
using CompactLES: step!
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

@testset "symmetric boundary shorthand" begin
    wall, periodic = SlipWallBC(), PeriodicBC()
    bcs = (wall, periodic, periodic)
    domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    ic = (x, y, z) -> Prim(p=1.0, rho=1.0)
    prob = Problem(; domain, bcs, ic)
    full = ((wall, wall), (periodic, periodic), (periodic, periodic))
    @test prob.bcs == full
    @test Problem(; domain, bcs=((wall,), (periodic,), (periodic,)), ic).bcs == full
    num = Numerics(n_global=(max(24, 12MPI.Comm_size(MPI.COMM_WORLD)), 1, 1))
    s, Q = setup(prob, num)
    sf, Qf = setup(Problem(; domain, bcs=full, ic), num)
    @test parent(Q) == parent(Qf)
    step!(s, Q, Workspace(Q), 1e-4)
    step!(sf, Qf, Workspace(Qf), 1e-4)
    @test parent(Q) == parent(Qf)

    direct = Solver(n_global=num.n_global, L_domain=(1.0, 1.0, 1.0), bcs=bcs)
    @test direct.bcs == full
    mixed = ((wall, NSCBCOutflowBC(pinf=1.0)), periodic, periodic)
    @test Problem(; domain, bcs=mixed, ic).bcs[1] == mixed[1]

    shared = SwitchableBC(SlipWallBC(), NSCBCOutflowBC(pinf=1.0))
    switched_prob = Problem(; domain, bcs=(shared, bcs[2], bcs[3]), ic)
    @test switched_prob.bcs[1][1] === switched_prob.bcs[1][2]
    switch!(shared)
    @test all(switched, switched_prob.bcs[1])

    for bad in (((), bcs[2], bcs[3]), ((wall, wall, wall), bcs[2], bcs[3]),
                ((:wall,), bcs[2], bcs[3]), (bcs[1], bcs[2]))
        @test_throws ArgumentError Problem(; domain, bcs=bad, ic)
    end
end
