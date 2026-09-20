using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using Test

const CL = CompactLES

@testset "composite conserved-budget diagnostic" begin
    wall = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    bcs = (wall, per, per)
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 1.0, 1.4)])
    velocity = (0.4, -0.3, 0.2)
    ic(x, y, z) = Prim(Y=(0.3, 0.7), u=velocity, p=2.0 + x,
                       rho=1.0 + x)

    # Every conserved density is linear in x, so the masked node quadrature
    # has the same analytic answer at same-level interfaces and through a
    # three-level hierarchy.  Nonzero transverse momentum makes component
    # ordering errors visible.
    mass = 1.5
    pressure_energy = 2.5 / (1.4 - 1)
    kinetic_energy = 0.5 * sum(abs2, velocity) * mass
    expected = (species_masses=[0.3mass, 0.7mass], total_mass=mass,
                momentum=ntuple(c -> velocity[c] * mass, 3),
                total_energy=pressure_energy + kinetic_energy)

    layouts = ((patch_grid=(2, 1, 1), subcycle=false),
               (refine=[BlockRegion((36, 0, 0), (32, 1, 1)),
                        BlockRegion((120, 0, 0), (8, 1, 1))], subcycle=true))
    for layout in layouts
        solver = Solver(n_global=(192, 1, 1), L_domain=(1.0, 1.0, 1.0),
                        bcs=bcs, eos=eos, art=ArtParams(enabled=false),
                        filter_interval=0; layout...)
        states = allocate_state(solver)
        initialize!(solver, states, ic)
        budget = CL._conserved_budget(solver, states)
        @test budget.species_masses ≈ expected.species_masses atol=2e-13
        @test budget.total_mass ≈ expected.total_mass atol=2e-13
        @test all(isapprox.(budget.momentum, expected.momentum; atol=2e-13))
        @test budget.total_energy ≈ expected.total_energy atol=2e-13

        # Independent scalar quadratures guard the packed channel mapping.
        eq = solver.equations
        components(c) = [Array(view(Q, :, :, :, c)) for Q in states]
        @test budget.species_masses ≈
              [volume_integral(solver, components(sp)) for sp in 1:eq.n_species]
        momentum_check = ntuple(c -> volume_integral(solver, components(eq.i_mom[c])), 3)
        @test all(isapprox.(budget.momentum, momentum_check))
        @test budget.total_energy ≈
              volume_integral(solver, components(eq.i_energy))
        @test mix_width(solver, states) ≈ 0.84 atol=2e-13
        @test molecular_mixing(solver, states) ≈ 1.0 atol=2e-13
        @test length(profile_coordinate(solver, 1)) == 192
        @test sum(profile_spacing(solver, 1)) ≈ 1.0 atol=2e-13
        if haskey(layout, :refine) && MPI.Comm_size(solver.comm) > 2
            held = MPI.Allgather(npatches(solver), solver.comm)
            @test minimum(held) < maximum(held) # root-only ranks join the reduction
        end
    end

    periodic = Solver(n_global=(192, 1, 1), L_domain=(1.0, 1.0, 1.0),
                      bcs=(per, per, per), eos=eos,
                      art=ArtParams(enabled=false), patch_grid=(2, 1, 1))
    periodic_states = allocate_state(periodic)
    initialize!(periodic, periodic_states,
                (x, y, z) -> Prim(Y=(0.3, 0.7), u=velocity, p=2.0,
                                  rho=1.0 + 0.2sin(2π * x)))
    rho_fields = [Array(view(Q, :, :, :, 1)) + Array(view(Q, :, :, :, 2))
                  for Q in periodic_states]
    coords = profile_coordinate(periodic, 1)
    @test length(coords) == 192
    @test coords[1] ≈ 0.0 atol=1e-15
    @test coords[end] ≈ 1 - 1 / 192 atol=1e-15
    @test maximum(abs.(plane_profile(periodic, rho_fields, 1) .-
                       (1 .+ 0.2sin.(2π .* coords)))) < 2e-14
    @test all(isapprox.(profile_spacing(periodic, 1), 1 / 192; atol=1e-15))

    # Regridding changes both the patch list and the root cover mask.  A fresh
    # budget must follow the new layout while preserving this linear state.
    pred = (p, I) -> xcoord(p, 1, interior_index(p, I)[1]) > 0.72
    moving = Solver(n_global=(192, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=bcs, eos=eos, art=ArtParams(enabled=false),
                    filter_interval=0, subcycle=true, regrid_interval=1,
                    tag_buffer=0, tag_predicate=pred,
                    refine=BlockRegion((48, 0, 0), (32, 1, 1)))
    states = allocate_state(moving)
    initialize!(moving, states, ic)
    before = CL._conserved_budget(moving, states)
    old_region = only(level_regions(moving, 1))
    moving.step += 1
    CL._maybe_regrid!(moving, states, Workspace(states), nothing)
    @test only(level_regions(moving, 1)) != old_region
    after = CL._conserved_budget(moving, states)
    @test before.species_masses ≈ expected.species_masses atol=2e-13
    @test after.species_masses ≈ before.species_masses atol=2e-12
    @test after.total_mass ≈ before.total_mass atol=2e-12
    @test all(isapprox.(after.momentum, before.momentum; atol=2e-12))
    @test after.total_energy ≈ before.total_energy atol=2e-12

    single = Solver(n_global=(192, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=bcs, eos=eos, art=ArtParams(enabled=false))
    Q = allocate_state(single)
    initialize!(single, Q, ic)
    @test CL._conserved_budget(single, Q).total_mass ≈ mass atol=1e-13
    @test_throws ArgumentError CL._conserved_budget(single, ConservedState[])
end
