# Companion to bench/interfaceconservation.jl: an acoustic pulse crosses
# both faces of a fixed refinement nest. Measure the left-running acoustic
# characteristic against a uniform run, excluding its nonlinear pulse wake.
# Then the δ⁴ sensor's reading of the ghost data at those same faces.
# Included by runtests.jl after the patch and level suites.

@isdefined(build_solver) || include(joinpath(@__DIR__, "build_solver.jl"))
using CompactLES: padded_index, xcoord

@testset "interface budgets: coarse-fine acoustic reflection" begin
    N = 192
    amp = 1e-3
    c0 = sqrt(1.4)
    bcs = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    pulse(x) = amp * exp(-40 * (x - pi / 2)^2)
    ic(x, y, z) = Prim(rho=(1 + pulse(x))^(1 / 1.4),
                       p=1 + pulse(x), u=(pulse(x) / c0, 0, 0))
    function reflected_characteristic(; refine=nothing, subcycle=false)
        s = build_solver(
            Problem(domain=((0.0, 2pi), (0.0, 1.0), (0.0, 1.0)), bcs=bcs, ic=nothing),
            Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=refine, regrid_interval=0, subcycle=subcycle)))
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        run!(s, Q; tfinal=pi / c0)
        states = Q.patches
        ps = CL.PatchSolver(s, s.patches[1])
        refresh_primitives!(ps, states[1])
        # This window is uncovered root grid, upstream of the first face.
        window = [padded_index(ps, i, 1, 1) for i in 1:ps.decomp.n_local[1]
                  if xcoord(ps, 1, i) < 2.1]
        (pressure=[ps.p[I] - 1 for I in window],
         leftgoing=[((ps.p[I] - 1) - c0 * ps.u[I]) / 2 for I in window])
    end
    reference = reflected_characteristic()
    r1 = BlockRegion((80, 0, 0), (33, 1, 1))
    r2 = BlockRegion((264, 0, 0), (49, 1, 1))
    for depth in (2, 3), subcycle in (false, true)
        reflected = reflected_characteristic(
            refine=depth == 2 ? r1 : [r1, r2], subcycle=subcycle)
        wake = maximum(abs.(reflected.pressure .- reference.pressure)) / amp
        leftgoing = maximum(abs.(reflected.leftgoing .- reference.leftgoing)) / amp
        @info "coarse-fine pulse reflection" depth subcycle wake leftgoing
        @test wake < 0.01
        @test leftgoing < 0.01
    end
end

# The unweighted internal-energy δ⁴ sensor of one patch, over its own grid
# line. `wpow = 0` drops the physical-spacing weight, so the two resolutions
# below are comparable without rescaling.
function _interface_sensor_line(ps, Q)
    refresh_primitives!(ps, Q)
    nxf, nyf, nzf = CL.padded_extent(ps.decomp)
    m1, m2, m3 = ps.equations.i_mom
    CL.pointwise!(CL._internal_energy_point!, ps.tmp_a, nxf, nyf, nzf,
                  ps.tmp_a, Q, ps.rho, m1, m2, m3, ps.equations.i_energy)
    exchange_halos!(ps.tmp_a, ps.decomp)
    CL.detect_sum!(ps.sensor, ps.tmp_a, ps, 0; ghosts=true)
    return [ps.sensor[padded_index(ps, i, 1, 1)] for i in 1:ps.decomp.n_local[1]]
end

@testset "interface ghosts: the δ⁴ sensor at a patch face" begin
    N = 96
    perx = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    # Smooth, resolved, and at rest, so the internal energy is 1/((γ−1)ρ) and
    # the sensor reads the discretization alone.
    ic(x, y, z) = Prim(rho=1 + 0.3 * sinpi(2x) + 0.05 * sinpi(10x), p=1.0,
                       u=(0.0, 0.0, 0.0))
    mk(n; patch_grid=(1, 1, 1), refine=nothing) = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=perx, ic=nothing),
        Numerics(n_global=(n, 1, 1), art=ArtificialProperties(enabled=false),
                 filter=StateFilter(interval=0), execution=Execution(patch_grid=patch_grid),
                 amr=AMR(initial=refine, regrid_interval=0)))
    # The references: one grid, no interface. The refined patch runs at h/3,
    # and the 3N grid carries its nodes at the same coordinates.
    function uniform(n)
        s = mk(n)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        _interface_sensor_line(CL.PatchSolver(s, s.patches[1]), Q)
    end
    reference = (uniform(N), uniform(3N))
    # The two nodes nearest each interface face, against the reference node at
    # the same coordinate. `_presync!` is what a step runs before its first
    # right-hand side: the same-level ghost exchange and the coarse-to-fine
    # imposition, without advancing anything.
    function face_error(s)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        CL._presync!(s, Q)
        e = 0.0
        for (li, p) in enumerate(s.patches)
            ps = CL.PatchSolver(s, p)
            line = _interface_sensor_line(ps, Q.patches[li])
            n = ps.decomp.n_local[1]
            nref = p.level == 0 ? N : 3N
            r = reference[p.level == 0 ? 1 : 2]
            for (side, nodes) in ((1, (1, 2)), (2, (n, n - 1)))
                ps.bcs[1][side] isa InterfaceBC || continue
                for i in nodes
                    m = mod(round(Int, xcoord(ps, 1, i) * nref), nref) + 1
                    e = max(e, abs(line[i] - r[m]))
                end
            end
        end
        return e
    end
    same_level = mk(N; patch_grid=(2, 1, 1))
    refined = mk(N; refine=BlockRegion((32, 0, 0), (32, 1, 1)))
    ghost_same, ghost_fine = face_error(same_level), face_error(refined)
    @info "interface ghosts: δ⁴ sensor against a uniform grid" ghost_same ghost_fine
    # A same-level ghost is a copy of the neighbor's interior node, so the two
    # sensors agree bitwise; measured 0.
    @test ghost_same < 1e-14
    # A coarse-fine ghost is the order-6 interpolant of the coarse solution,
    # and the undivided δ⁴ carries that error undiminished. Measured 2.9e-6
    # against a sensor of 1e-5 to 2.4e-5 on these nodes.
    @test ghost_fine < 1e-5
end
