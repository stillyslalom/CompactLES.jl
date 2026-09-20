# Companion to bench/interfaceconservation.jl: an acoustic pulse crosses
# both faces of a fixed refinement nest. Measure the left-running acoustic
# characteristic against a uniform run, excluding its nonlinear pulse wake.
# Included by runtests.jl after the patch and level suites.

@testset "interface budgets: coarse-fine acoustic reflection" begin
    N = 192
    amp = 1e-3
    c0 = sqrt(1.4)
    bcs = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    pulse(x) = amp * exp(-40 * (x - pi / 2)^2)
    ic(x, y, z) = Prim(rho=(1 + pulse(x))^(1 / 1.4),
                       p=1 + pulse(x), u=(pulse(x) / c0, 0, 0))
    function reflected_characteristic(; refine=nothing, subcycle=false)
        s = Solver(n_global=(N, 1, 1), L_domain=(2pi, 1.0, 1.0),
                   bcs=bcs, art=ArtParams(enabled=false), filter_interval=0,
                   refine=refine, subcycle=subcycle)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        run!(s, Q; tfinal=pi / c0)
        states = Q isa Vector ? Q : [Q]
        ps = CL.PatchSolver(s, s.patches[1])
        refresh_primitives!(ps, states[1])
        # This window is uncovered root grid, upstream of the first face.
        window = [gidx(ps, i, 1, 1) for i in 1:ps.decomp.n_local[1]
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
