# The phase change, `setup(solver, Q; ...)`: a run that `run!` has ended
# continues under other boundary conditions or numerics in a solver built for
# them. The carry is the checkpoint image written to memory and read back, so
# the contract is that a phase change and a checkpoint written at the stop and
# loaded into the next phase's solver continue bit for bit alike, on one patch
# and on a tiled, regridded, subcycled hierarchy. The multi-rank form of the
# same comparison is in mpi_tests.jl.
#
# Serial. Included by runtests.jl, and runnable on its own:
#
#   julia --project=. test/phase_tests.jl

if !@isdefined(CL)
    using MPI
    MPI.Init(threadlevel=:funneled)
    using CompactLES
    using Test
    const CL = CompactLES
end

const ph_per = (PeriodicBC(), PeriodicBC())

ph_same(a, b, decomp) = (inner = CL.interior(decomp);
                         parent(a)[inner, :] == parent(b)[inner, :])

ph_tube(bcs) = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                       domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs,
                       ic=(x, y, z) -> Prim(rho=1 + 0.2exp(-100(x - 0.5)^2),
                                            u=(0.0, 0.0, 0.0),
                                            p=1 + 0.3exp(-100(x - 0.5)^2)))

@testset "phase change: one patch continues as a checkpoint does" begin
    # A pulse between two slip walls; after 20 steps the high wall opens to a
    # characteristic outflow and the filter runs every second step.
    walls = ((SlipWallBC(), SlipWallBC()), ph_per, ph_per)
    opened = ((SlipWallBC(), NSCBCOutflowBC(pinf=1.0)), ph_per, ph_per)
    num = Numerics(n_global=(64, 1, 1), art=ArtificialProperties(enabled=true))
    num2 = Numerics(num; filter=StateFilter(interval=2))
    s, q = setup(ph_tube(walls), num)
    run!(s, q; tfinal=1e9, nmax=20)
    dir = mktempdir()
    save_checkpoint(s, q, joinpath(dir, "stop"))
    p, pq = setup(s, q; bcs=opened, numerics=num2)
    r, rq = setup(ph_tube(opened), num2)
    load_checkpoint!(r, rq, joinpath(dir, "stop"); allow=(:boundaries, :numerics))
    rm(dir; recursive=true)
    @test p.bcs[1][2] isa NSCBCOutflowBC && p.filter_interval == 2
    @test (p.t, p.step, p.dt_prev, p.rate_prev, p.filter_rate_prev, p.cfl) ==
          (s.t, s.step, s.dt_prev, s.rate_prev, s.filter_rate_prev, s.cfl)
    @test p.wall_total == s.wall_total
    @test ph_same(pq, q, s.decomp) && CL.art_block(p) == CL.art_block(s)
    run!(p, pq; tfinal=1e9, nmax=60)
    run!(r, rq; tfinal=1e9, nmax=60)
    @test (p.t, p.step, p.dt_prev) == (r.t, r.step, r.dt_prev)
    @test ph_same(pq, rq, p.decomp) && CL.art_block(p) == CL.art_block(r)
    # The previous phase is left as it was, and continuing it under the walls
    # is a different run: the opened face lets the pulse out.
    @test s.step == 20 && s.bcs[1][2] isa SlipWallBC
    run!(s, q; tfinal=1e9, nmax=60)
    @test !ph_same(q, pq, s.decomp)
end

@testset "phase change: the CFL, and the changes refused" begin
    walls = ((SlipWallBC(), SlipWallBC()), ph_per, ph_per)
    num = Numerics(n_global=(64, 1, 1), art=ArtificialProperties(enabled=true))
    s, q = setup(ph_tube(walls), num)
    run!(s, q; tfinal=1e9, nmax=3)
    # A retry-lowered CFL continues unless the next phase gives another.
    s.cfl = 0.4
    @test setup(s, q)[1].cfl == 0.4
    @test setup(s, q; numerics=Numerics(num; cfl=0.3))[1].cfl == 0.3
    # The conditions a phase keeps: each names the keyword it came from.
    @test_throws "periodicity" setup(s, q; bcs=(ph_per, ph_per, ph_per))
    @test_throws "fold" setup(s, q; bcs=((SymmetryPlaneBC(), SlipWallBC()),
                                         ph_per, ph_per))
    @test_throws "n_global" setup(s, q; numerics=Numerics(num; n_global=(65, 1, 1)))
    @test_throws "art.enabled" setup(s, q;
        numerics=Numerics(num; art=ArtificialProperties(enabled=false)))
    @test_throws "refinement hierarchy" setup(s, q;
        numerics=Numerics(num; amr=AMR(initial=BlockRegion((24, 0, 0), (12, 1, 1)))))
    low = Solver(n_global=(64, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=walls)
    @test_throws "Solver(; ...)" setup(low, allocate_state(low))
    # The wrapper the phase change replaces warns that it is deprecated.
    @test_logs (:warn, r"SwitchableBC is deprecated") SwitchableBC(SlipWallBC(),
                                                                   SlipWallBC())
end

@testset "phase change: a tiled, regridded, subcycled hierarchy" begin
    # The tube of the unrefined-start regrid test in level_tests.jl: an
    # inflow fires a shock, whose tiles exist by step 50. There the high wall
    # opens to an outflow; the phase change and the checkpoint restart then
    # regrid, subcycle and rebalance alike to step 80.
    ramp(t) = clamp((t - 0.04) / 0.01, 0.0, 1.0)
    inflow = DirichletBC((x, y, z, t) -> (w = ramp(t);
        Prim(rho=1 + 0.8621w, u=(0.8216w, 0.0, 0.0), p=1 + 1.4583w)))
    rest(x, y, z) = Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1.0)
    tube(bcs) = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                        domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=bcs, ic=rest)
    closed = ((inflow, SlipWallBC()), ph_per, ph_per)
    opened = ((inflow, NSCBCOutflowBC(pinf=1.0)), ph_per, ph_per)
    num = Numerics(n_global=(121, 1, 1), cfl=0.3,
                   amr=AMR(initial=:sensor, tile=8, regrid_interval=5,
                           tag_buffer=2, subcycle=true))
    s, states = setup(tube(closed), num)
    run!(s, states; tfinal=1.0, nmax=50)
    regs = level_regions(s, 1)
    @test !isempty(regs)
    dir = mktempdir()
    save_checkpoint(s, states, joinpath(dir, "stop"))
    p, ps = setup(s, states; bcs=opened)
    r, rs = setup(tube(opened), num)
    load_checkpoint!(r, rs, joinpath(dir, "stop"); allow=(:boundaries,))
    rm(dir; recursive=true)
    @test level_regions(p, 1) == regs && length(ps) == length(states)
    @test getfield(p, :regrid).created == getfield(s, :regrid).created
    @test all(ph_same(ps[i], states[i], p.patches[i].decomp) for i in eachindex(ps))
    run!(p, ps; tfinal=1.0, nmax=80)
    run!(r, rs; tfinal=1.0, nmax=80)
    @test (p.t, p.step, p.dt_prev) == (r.t, r.step, r.dt_prev)
    @test level_regions(p, 1) == level_regions(r, 1)
    @test getfield(p, :regrid).created == getfield(r, :regrid).created
    @test length(ps) == length(rs) &&
          all(ph_same(ps[i], rs[i], p.patches[i].decomp) for i in eachindex(ps))
    @test all(all(isfinite, parent(Q)) for Q in ps)
end
