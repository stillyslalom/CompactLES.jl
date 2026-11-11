using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: npatches, ConservedState, interior_index, xcoord
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
                        bcs=bcs, eos=eos, art=ArtificialProperties(enabled=false),
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
                      art=ArtificialProperties(enabled=false), patch_grid=(2, 1, 1))
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
                    bcs=bcs, eos=eos, art=ArtificialProperties(enabled=false),
                    filter_interval=0, subcycle=true, regrid_interval=1,
                    tag_buffer=0, tag_predicate=pred,
                    refine=BlockRegion((48, 0, 0), (32, 1, 1)))
    states = allocate_state(moving)
    initialize!(moving, states, ic)
    # The cells alone: the junctions' correction of the quadrature is not
    # exact for a linear field where a level ends on the domain's face.
    before = CL._conserved_budget(moving, states; junctions=false)
    old_region = only(level_regions(moving, 1))
    moving.step += 1
    CL._maybe_regrid!(moving, states, Workspace(states), nothing)
    @test only(level_regions(moving, 1)) != old_region
    after = CL._conserved_budget(moving, states; junctions=false)
    @test before.species_masses ≈ expected.species_masses atol=2e-13
    # The new region reaches the wall at x = 1. The cells meet without
    # overlap at its coarse-fine face, and the two walls' half cells, at the
    # tile's spacing there and the root's at x = 0, leave −(H² − h²)/8 times
    # each density's slope, which for 1 + x is the mass's own.
    H = moving.patches[1].h[1]
    wall = -(H^2 - (H / 3)^2) / 8
    @test after.species_masses ≈ expected.species_masses .+ wall .* [0.3, 0.7] atol=2e-12
    @test after.total_mass ≈ mass + wall atol=2e-12
    @test all(isapprox.(after.momentum, expected.momentum .+ wall .* velocity; atol=2e-12))
    @test after.total_energy ≈
          expected.total_energy + wall * (1 / (1.4 - 1) + 0.5 * sum(abs2, velocity)) atol=2e-12

    single = Solver(n_global=(192, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=bcs, eos=eos, art=ArtificialProperties(enabled=false))
    Q = allocate_state(single)
    initialize!(single, Q, ic)
    @test CL._conserved_budget(single, Q).total_mass ≈ mass atol=1e-13
    @test_throws ArgumentError CL._conserved_budget(single, ConservedState[])
end

@testset "budget ledger: attribution, closure, and an unchanged run" begin
    wall = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    pulse(x, y, z) = Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                          p=1 + 0.1exp(-100(x - 0.3)^2))
    function closed_box()
        s = Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(wall, per, per), cfl=0.5)
        Q = allocate_state(s)
        initialize!(s, Q, pulse)
        return s, Q
    end
    s, Q = closed_box()
    CL._ledger_begin!(s, Q)
    run!(s, Q; tfinal=0.6)
    r = CL._ledger_end!(s, Q)
    @test !CL.BUDGET_LEDGER.on
    # The ledger reads the state and writes none of it.
    ref, Qref = closed_box()
    run!(ref, Qref; tfinal=0.6)
    @test parent(Q) == parent(Qref)
    # Every write is bracketed, so nothing lands between the brackets, and
    # the pieces telescope to the drift.
    @test !any(k -> k[1] === :unattributed, keys(r.pieces))
    @test maximum(abs, r.residual) < 1e-13
    rhs = r.pieces[(:rhs, 0)]
    @test maximum(abs, rhs .- r.pieces[(:rhs_integral, 0)]) < 1e-13
    # Budget channels: species 1, the three momenta, the energy. The walls'
    # pressure is the whole x-momentum source; what the enforcement of
    # u = 0 on the wall planes removes and the non-summation-by-parts
    # closure add are below a percent of it at this resolution.
    delivered = r.pieces[(:wall_flux, 0)][2]
    enforced = get(r.pieces, (:wall_enforce, 0), zeros(5))[2]
    @test abs(delivered) > 1e-3
    @test abs(rhs[2] + enforced - delivered) < 1e-2 * abs(delivered)
    @test r.pieces[(:wall_flux, 0)][1] == 0      # no mass crosses a slip wall

    # Two levels in a periodic box, global and subcycled, with a regrid that
    # moves the level: every mechanism's piece telescopes, the regrid lands
    # in its own row, and the right-hand side matches its stage integral on
    # each level.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 1.0, 1.4)])
    wave(x, y, z) = Prim(Y=(0.5 + 0.4sin(2π * x), 0.5 - 0.4sin(2π * x)),
                         u=(1.0, 0.0, 0.0), p=1.0, rho=1.0 + 0.2sin(2π * x))
    for subcycle in (false, true)
        s = Solver(n_global=(96, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(per, per, per), eos=eos, subcycle=subcycle,
                   refine=BlockRegion((36, 0, 0), (24, 1, 1)), regrid_interval=6,
                   tag_buffer=0,
                   tag_predicate=(p, I) -> 0.6 < xcoord(p, 1, interior_index(p, I)[1]) < 0.8)
        Q = allocate_state(s)
        initialize!(s, Q, wave)
        CL._ledger_begin!(s, Q)
        run!(s, Q; tfinal=0.2)
        r = CL._ledger_end!(s, Q)
        @test !any(k -> k[1] === :unattributed, keys(r.pieces))
        @test maximum(abs, r.residual) < 1e-12
        # The imposed shell writes only nodes the conserved quadrature does
        # not count.
        @test haskey(r.pieces, (:regrid, 1)) && !haskey(r.pieces, (:shell, 1))
        for level in 0:1
            @test maximum(abs, r.pieces[(:rhs, level)] .-
                               r.pieces[(:rhs_integral, level)]) < 1e-12
        end
    end
end

@testset "conservative filter rows at a shared node" begin
    # A periodic line of N nodes as two patches sharing nodes 1 and nA, each
    # closed at both ends by the filter's interface rows, its ghost layers
    # holding the other patch's nodes. The composite pass takes each node from
    # the patch holding it, the mean of the two copies at a shared node, so its
    # column sums are one exactly when the pass conserves Σ h q.
    function composite_pass(scheme, rows, v)
        N = length(v)
        nA = N ÷ 2 + 1
        out = zeros(N)
        copies = zeros(N)
        spread = 0.0
        first_copy = fill(NaN, N)
        for (c0, n) in ((0, nA), (nA - 1, N - nA + 2))
            d = CL.Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1))
            plan = CL.plan_direction(d, scheme, 1, 1.0; lo_closures=rows,
                                     hi_closures=rows)
            f = CL.field(d)
            g = CL.field(d)
            pad = d.n_halo_d
            for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
                f[i, j, k] = v[mod(c0 + i - pad[1] - 1, N) + 1]
            end
            CL.apply_along!(g, plan, f, d)
            for i in 1:n
                c = mod(c0 + i - 1, N) + 1
                x = g[i+pad[1], pad[2]+1, pad[3]+1]
                isnan(first_copy[c]) ? (first_copy[c] = x) :
                                       (spread = max(spread, abs(x - first_copy[c])))
                out[c] += x
                copies[c] += 1
            end
        end
        return out ./ copies, spread
    end
    function defect(scheme, rows; N=48)
        cols = zeros(N)
        spread = 0.0
        for j in 1:N
            e = zeros(N)
            e[j] = 1
            col, s = composite_pass(scheme, rows, e)
            cols[j] = sum(col) - 1
            spread = max(spread, s)
        end
        return maximum(abs, cols), spread
    end
    for scheme in (compact_filter(), compact_filter(0.45), pyranda_filter())
        rows = CL._conservative_interface_rows(scheme)
        @test length(rows) == 2
        # Row 1 reads the halo's four layers, and both rows pass a constant.
        @test rows[1].first == -3
        @test all(r -> abs(sum(r.rhs) - sum(r.lhs)) < 1e-14, rows)
        d_on, s_on = defect(scheme, rows)
        @test d_on < 1e-14 && s_on < 1e-14
        d_off, _ = defect(scheme, CL.interface_closures(scheme))
        @test d_off > 1e-3
        # The rows keep the filter's order: a polynomial of degree seven,
        # which the interior pass leaves unchanged, passes unchanged.
        p7(x) = 1 + x - 0.3x^2 + 0.2x^3 - 0.05x^4 + 0.01x^5 - 0.002x^6 + 3e-4x^7
        n = 32
        dl = CL.Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1))
        plan = CL.plan_direction(dl, scheme, 1, 1.0; lo_closures=rows, hi_closures=rows)
        f = CL.field(dl)
        g = CL.field(dl)
        pad = dl.n_halo_d
        for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
            f[i, j, k] = p7((i - pad[1] - 1) / n)
        end
        CL.apply_along!(g, plan, f, dl)
        @test maximum(abs(g[i+pad[1], pad[2]+1, pad[3]+1] - p7((i - 1) / n))
                      for i in 1:n) < 1e-12        # measured 1.6e-13
    end
    # The explicit Gaussian reads the ghosts through its interior stencil and
    # takes no rows: the composite pass is the single line's.
    @test isempty(CL._conservative_interface_rows(gaussian_filter()))
    @test defect(gaussian_filter(), CL._conservative_interface_rows(gaussian_filter()))[1] <
          1e-14

    # A tiled level of four tiles, a density step at two of the faces they
    # share, the state uniform within five fine nodes of the coarse-fine faces.
    # The rows close the coarse-fine ends too, so a pass changes the tiles'
    # mass (each tile's own quadrature, a shared node at h/2 on either side) by
    # the filter's face values at those ends, which vanish on uniform data. One
    # pass leaves it to round-off under the rows and moves it by the identity
    # row's defect without them.
    function filtered_change(conservative)
        saved = CL.CONSERVATIVE_INTERFACE_FILTER[]
        CL.CONSERVATIVE_INTERFACE_FILTER[] = conservative
        try
            per = (PeriodicBC(), PeriodicBC())
            s = Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=(per, per, per), filter_cfl=0.0, tile=8,
                       refine=BlockRegion((8, 0, 0), (32, 1, 1)))
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) ->
                Prim(rho=1.0 + 0.5 * (abs(x - 0.5) < 0.168), p=1.0, u=(0.3, 0.0, 0.0)))
            # The ghost layers from the neighbouring tiles and the parent's
            # shell, as `run!` holds them when it filters.
            CL.sync_patches!(s, Q)
            CL.prolong_level_ghosts!(s, Q)
            lev = s.levels[2]
            tiles_mass() = sum(CL._local_volume_integral(CL.PatchSolver(s, s.patches[i]),
                                                         Q[i][:, :, :, 1], false)
                               for i in lev.patches)
            before = tiles_mass()
            CL.filter_state!(s, Q)
            return length(lev.transfers), tiles_mass() / before - 1
        finally
            CL.CONSERVATIVE_INTERFACE_FILTER[] = saved
        end
    end
    tiles, change_on = filtered_change(true)
    @test tiles == 4
    @test abs(change_on) < 1e-14
    @test abs(filtered_change(false)[2]) > 1e-6
end
