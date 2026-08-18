# Focused CPU validation matrix for G4a of reference/AMR_GPU.md. The basic
# Float32 tests in runtests.jl compile the individual paths; these cases ask
# whether they retain the solver's physical invariants and reference behavior.

_f32_numerics(; art=false) =
    (transport=Transport{Float32}(),
     art=ArtParams{Float32}(enabled=art),
     deriv=lele_d1_6(Float32),
     filt=compact_filter(Float32(0.45), Float32))

@testset "Float32 freestream/GCL matrix" begin
    T = Float32
    per = (PeriodicBC(), PeriodicBC())
    typed = _f32_numerics()
    cases = [
        (; n_global=(24, 12, 12),
           L_domain=(one(T), one(T), one(T)),
           metric=CartesianMetric(), bcs=ntuple(_ -> per, 3), extra=(;)),
        (; n_global=(24, 12, 12),
           L_domain=(one(T), one(T), one(T)),
           metric=CartesianMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per, per),
           extra=(; stretch=(sine_cluster(zero(T), one(T), T(0.5), T(0.4)),
                              nothing, nothing))),
        (; n_global=(24, 12, 12),
           L_domain=(one(T), T(2π), T(0.5)),
           metric=CylindricalMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per, per),
           extra=(; origin=(T(0.2), zero(T), zero(T)))),
        (; n_global=(32, 1, 12),
           L_domain=(one(T), one(T), T(0.5)),
           metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per, per), extra=(;)),
        (; n_global=(24, 16, 1),
           L_domain=(one(T), T(2π), one(T)),
           metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per, per), extra=(;)),
        (; n_global=(24, 12, 12),
           L_domain=(one(T), T(π), T(2π)),
           metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per),
           extra=(;)),
    ]
    residuals = T[]
    for cs in cases
        kw = merge((; n_global=cs.n_global, L_domain=cs.L_domain,
                     metric=cs.metric, bcs=cs.bcs, eos=single_species(T),
                     filter_interval=0, typed...), cs.extra)
        solver = Solver(; kw...)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) ->
            Prim(u=(0, 0, 0), p=1, rho=1))
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        push!(residuals,
              maximum(abs(dQ[gidx(solver, i, j, k), c])
                      for c in 1:solver.equations.n_cons,
                          i in 1:solver.decomp.n_local[1],
                          j in 1:solver.decomp.n_local[2],
                          k in 1:solver.decomp.n_local[3]))
    end
    # The Cartesian cases are exact. Curvilinear metric/source cancellation
    # reaches the Float32 roundoff floor; spherical is the largest at 1.22e-4.
    @test residuals[1] == residuals[2] == zero(T)
    @test maximum(residuals) < T(2e-4)
end

function _f32_closed_derivative_error(N; spherical=false)
    T = Float32
    per = (PeriodicBC(), PeriodicBC())
    solver = if spherical
        Solver(; n_global=(N, 12, 12),
               L_domain=(one(T), T(π), T(2π)),
               metric=SphericalMetric(),
               bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per),
               eos=single_species(T), _f32_numerics()...)
    else
        Solver(; n_global=(N, 12, 12),
               L_domain=(one(T), one(T), one(T)),
               bcs=((SlipWallBC(), SlipWallBC()), per, per),
               eos=single_species(T), _f32_numerics()...)
    end
    f = CL.field(solver.decomp)
    df = similar(f)
    for k in 1:solver.decomp.n_local[3],
        j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        x = xcoord(solver, 1, i)
        f[gidx(solver, i, j, k)] =
            spherical ? exp(-T(4) * x^2) : exp(sin(T(3) * x))
    end
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1)
    CL._scale_grad!(df, solver, 1)
    err = zero(T)
    for k in 1:solver.decomp.n_local[3],
        j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        x = xcoord(solver, 1, i)
        exact = spherical ?
            -T(8) * x * exp(-T(4) * x^2) :
            T(3) * cos(T(3) * x) * exp(sin(T(3) * x))
        err = max(err, abs(df[gidx(solver, i, j, k)] - exact))
    end
    return err
end

@testset "Float32 closed and spherical smooth convergence" begin
    wall = [_f32_closed_derivative_error(N) for N in (24, 48, 96)]
    sphere = [_f32_closed_derivative_error(N; spherical=true)
              for N in (24, 48, 96)]
    wall_orders = log2.(wall[1:2] ./ wall[2:3])
    sphere_order = log2(sphere[1] / sphere[2])
    # The wall closure keeps its measured third-order behavior. Spherical
    # reaches ~1e-5 at N=96, where Float32 roundoff lowers the final apparent
    # order; require the pre-floor order and the achieved error separately.
    @test minimum(wall_orders) > Float32(2.7)
    @test sphere_order > Float32(2.7)
    @test sphere[end] < Float32(2e-5)
end

@testset "Float32 Sod profile and conservation" begin
    T = Float32
    N = 200
    hx = one(T) / T(N - 1)
    per = (PeriodicBC(), PeriodicBC())
    solver = Solver(; n_global=(N, 1, 1),
                    L_domain=(one(T), hx, hx),
                    bcs=((SlipWallBC(), SlipWallBC()), per, per),
                    eos=single_species(T; gamma=T(1.4), R=one(T)),
                    cfl=T(0.4), filter_interval=1,
                    _f32_numerics(art=true)...)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> begin
        θ = tanh_blend(x, T(0.5), T(2) * hx)
        Prim(rho=(one(T) - θ) + θ * T(0.125),
             u=(zero(T), zero(T), zero(T)),
             p=(one(T) - θ) + θ * T(0.1))
    end)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    mass0 = volume_integral(solver, solver.rho)
    run!(solver, Q; tfinal=T(0.2), nmax=100_000)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    mass1 = volume_integral(solver, solver.rho)

    pstar, ustar, cL, cR =
        exact_riemann_star(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4)
    eρ = eu = ep = 0.0
    for i in 1:N
        I = gidx(solver, i, 1, 1)
        x = Float64(xcoord(solver, 1, i))
        ρ, u, p = exact_riemann_sample(
            (x - 0.5) / 0.2, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4,
            pstar, ustar, cL, cR)
        eρ += abs(Float64(solver.rho[I]) - ρ)
        eu += abs(Float64(solver.u[I]) - u)
        ep += abs(Float64(solver.p[I]) - p)
    end
    eρ /= N
    eu /= N
    ep /= N
    @test eρ < 1e-2
    @test eu < 1.5e-2
    @test ep < 8e-3
    @test abs(mass1 - mass0) / mass0 < 1e-5
    @test all(isfinite, parent(Q))
end

function _tgv_short_history(::Type{T}) where {T<:AbstractFloat}
    N = 16
    per = (PeriodicBC(), PeriodicBC())
    solver = Solver(; n_global=(N, N, N),
                    L_domain=(T(2π), T(2π), T(2π)),
                    bcs=(per, per, per),
                    eos=single_species(T; gamma=T(1.4), R=one(T)),
                    transport=Transport{T}(mu0=one(T) / T(1600)),
                    art=ArtParams{T}(enabled=false),
                    deriv=lele_d1_6(T),
                    filt=compact_filter(T(0.45), T), cfl=T(0.6))
    Q = allocate_state(solver)
    p0 = T(100) / T(1.4)
    initialize!(solver, Q, (x, y, z) ->
        Prim(u=(sin(x) * cos(y) * cos(z),
                -cos(x) * sin(y) * cos(z), zero(T)),
             p=p0 + T(1) / T(16) *
                 (cos(T(2) * x) + cos(T(2) * y)) *
                 (cos(T(2) * z) + T(2)),
             rho=one(T)))

    function state_totals()
        mass = 0.0
        ke = 0.0
        m1, m2, m3 = solver.equations.i_mom
        for k in 1:N, j in 1:N, i in 1:N
            I = gidx(solver, i, j, k)
            ρ = Float64(Q[I, 1])
            mass += ρ
            ke += 0.5 * (Float64(Q[I, m1])^2 +
                         Float64(Q[I, m2])^2 +
                         Float64(Q[I, m3])^2) / ρ
        end
        return (mass / N^3, ke / N^3)
    end

    history = [state_totals()]
    for stop in (T(0.025), T(0.05), T(0.1))
        run!(solver, Q; tfinal=stop)
        push!(history, state_totals())
    end
    return history
end

@testset "Float32 short TGV history against Float64" begin
    h64 = _tgv_short_history(Float64)
    h32 = _tgv_short_history(Float32)
    ke64 = last.(h64)
    ke32 = last.(h32)
    @test all(diff(ke32) .< 0)
    @test maximum(abs.(ke32 .- ke64)) < 5e-7
    @test abs(first(h32[end]) - first(h32[1])) < 3e-6
end
