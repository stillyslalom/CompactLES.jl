# The serial test suite's body, run by test/runtests.jl inside the suite's one
# top-level testset (see the note there). Ordered so failures localize:
# solvers → operators → closures/folds → metric → full RHS. Multi-rank checks
# live in mpi_tests.jl.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: validate_bc, sensor_mirror, isperiodic, state_admissibility, EquationSet,
                  NavierStokes1T, rewind!, compute_rhs!, apply_bcs!, filter_state!,
                  max_rate, step!, npatches, ConservedState, interior_index, padded_index,
                  xcoord, global_xcoord, StateReport, CompactScheme, BandedCompactScheme,
                  pade_d1_4, compact_d8
import CompactLES: Decomp, exchange_halos!, interior, field, DirPlan, BandPlan,
                   DevicePlan, device_plan, apply_along!, filter_field!,
                   amr_transfer_schemes, amr_restriction_scheme,
                   amr_prolongation_scheme, amr_interpolation_weights,
                   TransferPlan, plan_transfer, restrict!, prolong!,
                   plan_direction, Patch, PatchSolver, InterfaceBC, CoarseFineBC,
                   LevelTransfer, Level, LevelComm, exchange_patch_ghosts!,
                   average_shared_planes!, prolong_level_ghosts!, restrict_level!,
                   scalar_field, container_extension, PLANCK_TIME,
                   THREAD_MIN_WORK, script_args, script_grid
using Test, LinearAlgebra, Random

const CL = CompactLES
Random.seed!(7)

mkslv(; kw...) = Solver(; bcs=per3, L_domain=(2π, 2π, 2π),
                        art=ArtificialProperties(enabled=false), kw...)

"Max interior error of a scalar field against an analytic function."
function ferr(solver, f, fn)
    e = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        e = max(e, abs(f[padded_index(solver, i, j, k)] -
                       fn(xcoord(solver, 1, i), xcoord(solver, 2, j), xcoord(solver, 3, k))))
    end
    e
end

fillf!(solver, f, fn) = (for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                        i in 1:solver.decomp.n_local[1]
    f[padded_index(solver, i, j, k)] = fn(xcoord(solver, 1, i), xcoord(solver, 2, j),
                                          xcoord(solver, 3, k))
end; f)

# Analytic references (exact Riemann solver, Noh, Sedov) live in one place so
# the serial suite and test/validation.jl measure against the same solution.
include("references.jl")

# The shock-capturing case setups, shared with test/validation.jl and
# bench/artcal.jl. This suite runs only the cheap ones; the battery in
# validation.jl runs them at their reference resolutions. `per3`, used
# throughout below, is defined there.
include("cases.jl")

@testset "Problem preserves endpoint conversion" begin
    prob = Problem(domain=((0, 1), (0, pi), (0, 2pi)), bcs=per3,
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0))
    @test prob.domain === ((0.0, 1.0), (0.0, Float64(pi)), (0.0, 2pi))
    @test prob.ic(0.0, 0.0, 0.0).p == 1.0
end

@testset "Numerics groups, shorthands and copies" begin
    num = Numerics(n_global=(16, 1, 1))
    @test num.filter.interval == 1 && num.filter.cfl == 0.35
    @test num.patch_interfaces.flux === :ghost && num.execution.dims === nothing
    @test Numerics(n_global=(16, 1, 1), filter=nothing).filter.interval == 0
    shorthand = Numerics(n_global=(16, 1, 1), filter=pyranda_filter(),
                         patch_interfaces=:closure)
    @test shorthand.filter.scheme isa typeof(pyranda_filter())
    @test shorthand.filter.interval == 1
    @test shorthand.patch_interfaces.flux === :closure
    copied = Numerics(num; cfl=0.3, filter=StateFilter(num.filter; cfl=0.0))
    @test copied.cfl == 0.3 && copied.filter.cfl == 0.0 && copied.n_global == num.n_global
    @test num.cfl == 0.5
    @test_throws ArgumentError Numerics(n_global=(16, 1, 1), nonexistent=1)
    @test_throws ArgumentError Numerics(n_global=(16, 1, 1), filter=1.0)
    @test_throws ArgumentError StateFilter(StateFilter(); nonexistent=1)

    # The flat keywords of earlier versions fold into their groups.
    flat = @test_logs (:warn, r"flat keywords are deprecated") Numerics(
        n_global=(16, 1, 1), filter_interval=0, dims=(1, 1, 1), interface_flux=:closure)
    @test flat.filter.interval == 0 && flat.execution.dims == (1, 1, 1)
    @test flat.patch_interfaces.flux === :closure

    # A copy keeps the element type of a parametric group and still validates.
    art = ArtificialProperties(ArtificialProperties{Float32}(); C_D=3.0)
    @test art.C_D === 3.0f0
    @test StepControl(StepControl(retries=4); validity=:permissive).retries == 4
    @test_throws ArgumentError StepControl(StepControl(); retries=-1)
    @test AMR(AMR(tile=12); subcycle=true).tile == 12

    # The groups reach the solver under its flat names.
    prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0 + 0.1sin(2π * x)))
    solver, _ = setup(prob, Numerics(n_global=(32, 1, 1),
                                     filter=StateFilter(compact_filter(0.49);
                                                        interval=2, cfl=0.0)))
    @test solver.filter_interval == 2 && solver.filter_cfl == 0
    @test solver.control.retries == 0

    # A spherical origin takes retries unless the deck gives its own control.
    sphere = Problem(metric=SphericalMetric(),
                     domain=((0.0, 1.0), (π / 2, π / 2 + 1), (0.0, 1.0)),
                     bcs=((OriginBC(), SlipWallBC()), per3[2], per3[3]),
                     ic=(r, θ, φ) -> Prim(p=1.0 + 0.1exp(-(r / 0.3)^2), rho=1.0))
    origin, _ = setup(sphere, Numerics(n_global=(32, 1, 1)))
    @test origin.control.retries == 4
    explicit, _ = setup(sphere, Numerics(n_global=(32, 1, 1), control=StepControl()))
    @test explicit.control.retries == 0

    # A preset is a keyword set; keywords after it override it, and merge
    # combines two, the later winning.
    @test num.filter.scheme.alpha == 0.47
    resolved = Numerics(Presets.resolved(); n_global=(16, 1, 1), cfl=0.4)
    @test resolved.filter.scheme.alpha == 0.49 && resolved.cfl == 0.4
    shocked = Numerics(Presets.refined_shock(); n_global=(16, 1, 1))
    @test shocked.filter.scheme.alpha == 0.45 && shocked.patch_interfaces.flux === :closure
    cold = Numerics(merge(Presets.resolved(), Presets.converging(cold_ambient=true));
                    n_global=(16, 1, 1), filter=nothing)
    @test cold.control.retries == 4 && cold.control.validity === :permissive
    @test cold.filter.interval == 0
    walls = Numerics(Presets.smooth_walls(); n_global=(16, 1, 1)).deriv
    @test repr(walls.closures) == repr(lele_d1_6(closures=:brady_livescu).closures)
    @test repr(walls.closures) != repr(lele_d1_6().closures)
    @test occursin("filter: off",
                   sprint(show, MIME("text/plain"), Numerics(num; filter=nothing)))
end

@testset "concise frontend displays" begin
    prob = Problem(name="display test",
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                   ic=(x, y, z) -> Prim(u=(x, y, z), p=1.0, rho=1.0))
    num = Numerics(n_global=(12, 12, 12), art=ArtificialProperties(enabled=false))
    solver, Q = setup(prob, num)

    @test Q isa ConservedState
    @test parent(Q) isa Array{Float64,4}
    @test size(Q) == (20, 20, 20, 5)
    Q[1, 1, 1, 1] = 2.0
    @test parent(Q)[1, 1, 1, 1] == 2.0
    @test parent(view(Q, :, :, :, 1)) === parent(Q)
    @test copy(Q) isa ConservedState
    @test zero(Q) isa ConservedState

    q_display = sprint(show, MIME("text/plain"), Q)
    pair_display = sprint(show, MIME("text/plain"), (solver, Q))
    @test q_display == "ConservedState{Float64}(20 × 20 × 20 × 5)"
    @test length(pair_display) < 200
    @test occursin("Solver(grid=12 × 12 × 12", pair_display)
    @test occursin(q_display, pair_display)
    @test run!(solver, Q; tfinal=1.0, nmax=0) === Q

    solver_display = sprint(show, MIME("text/plain"), solver)
    problem_display = sprint(show, MIME("text/plain"), prob)
    numerics_display = sprint(show, MIME("text/plain"), num)
    @test occursin("5 conserved variables", solver_display)
    @test occursin(prob.name, problem_display)
    @test !occursin("#", problem_display)       # do not dump the IC closure type
    @test occursin("artificial properties: disabled", numerics_display)
    @test all(length.((solver_display, problem_display, numerics_display)) .< 500)

    # EOS objects: the NASA-9 records must not dump their fit coefficients.
    nasa = Nasa9Mixture(["He", "CO2"])
    nasa_display = sprint(show, MIME("text/plain"), nasa)
    @test occursin("He", nasa_display) && occursin("CO2", nasa_display)
    @test occursin("T_guess = 300.0", nasa_display)
    @test !occursin("Nasa9Interval", nasa_display)
    @test length(nasa_display) < 200
    @test length(sprint(show, nasa)) < 80
    @test length(sprint(show, MIME("text/plain"), nasa.sp)) < 300
    @test length(sprint(show, MIME("text/plain"), nasa.sp[1].intervals)) < 300
    @test sprint(show, IdealSpecies("gas"; R=1.0, gamma=1.4)) ==
          "IdealSpecies(\"gas\"; R=1.0, gamma=1.4)"
    ideal_display = sprint(show, MIME("text/plain"), IdealMixture(["He", "CO2"]))
    @test occursin("gamma", ideal_display) && length(ideal_display) < 200
    @test sprint(show, StiffenedGas()) ==
          "StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0, name=\"liquid\")"
end

@testset "Float32 frontend and full step" begin
    T = Float32
    prob = Problem(name="Float32 smoke", eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)),
                   transport=ConstantTransport{T}(),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                   ic=(x, y, z) -> Prim(rho=1 + 0.01sin(2π * x), T_ion=1,
                                        u=(0.1, 0.0, 0.0)))
    num = Numerics(n_global=(12, 12, 12), deriv=lele_d1_6(T),
                   filter=compact_filter(T(0.45), T), art=ArtificialProperties{T}(),
                   cfl=0.2)
    solver, Q = setup(prob, num)

    @test prob.transport isa ConstantTransport{T}
    @test num.art isa ArtificialProperties{T}
    @test parent(Q) isa Array{T,4}
    @test solver.deriv_plans[1] isa DirPlan{T}
    @test CL.positive_floor(T) > zero(T)
    @test CL.positive_floor(Float64) == 1e-300

    dt = compute_dt(solver, Q)
    @test dt isa T
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    @test all(isfinite, parent(dQ))
    @test maximum(abs, parent(dQ)) > zero(T)

    # Compile every converted pointwise body through KernelAbstractions in
    # Float32 as well as through the ordinary threaded CPU route.
    dQ_ka = zero(Q)
    CL.FORCE_KA[] = true
    try
        compute_rhs!(solver, Q, dQ_ka)
    finally
        CL.FORCE_KA[] = false
    end
    @test parent(dQ_ka) == parent(dQ)

    run!(solver, Q; tfinal=T(1e-4), nmax=1)
    @test solver.step == 1
    @test solver.t isa T
    @test all(isfinite, parent(Q))

    # Retries keep a savepoint, whose clock is Float64 in either precision.
    run!(solver, Q; tfinal=T(1e-3), nmax=3, control=StepControl(retries=2))
    @test solver.step > 1 && solver.t isa T && all(isfinite, parent(Q))
end

@testset "Float32 operator accuracy to the roundoff floor" begin
    T = Float32
    errs = T[]
    for N in (12, 18, 24)
        solver = Solver(n_global=(N, 12, 12),
                        L_domain=(T(2π), T(2π), T(2π)), bcs=per3,
                        transport=ConstantTransport{T}(),
                        art=ArtificialProperties{T}(enabled=false),
                        deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T))
        f = CL.field(solver.decomp)
        df = similar(f)
        fillf!(solver, f, (x, y, z) -> sin(x))
        exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, 1)
        CL._scale_grad!(df, solver, 1)
        push!(errs, ferr(solver, df, (x, y, z) -> cos(x)))
    end
    # C6 improves rapidly until truncation error meets Float32 roundoff around
    # 1e-6; requiring formal sixth order beyond that would test the format,
    # not the scheme.
    @test errs[2] < errs[1] / 4
    @test maximum(errs[2:3]) < T(3e-6)
end

@testset "Float32 built-in EOS and closed boundary matrix" begin
    T = Float32
    typed_num(; enabled=false) =
        (transport=ConstantTransport{T}(), art=ArtificialProperties{T}(enabled=enabled),
         deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T))

    # A NASA-9 fit is tabulated in kelvin over a declared interval, so the
    # states here are physical: `nasa9_constant_cp` carries the CEA default
    # range of 200 K to 6000 K, and a nondimensional temperature would be an
    # argument outside it rather than a state the solver cannot carry.
    nasa = Nasa9Mixture([CL.nasa9_constant_cp(T, "gas", T(1), T(3.5))])
    tn = typed_num()
    ns = Solver(n_global=(12, 12, 12),
                L_domain=(T(2π), T(2π), T(2π)), bcs=per3, eos=nasa,
                filter_interval=0, transport=tn.transport, art=tn.art,
                deriv=tn.deriv, filt=tn.filt)
    NQ = allocate_state(ns)
    initialize!(ns, NQ, (x, y, z) -> Prim(u=(0.1, 0, 0), p=1, T_ion=300))
    CL.exchange_state!(NQ, ns.decomp)
    CL.primitives!(ns, NQ)
    I = padded_index(ns, 3, 4, 5)
    @test ns.p[I] ≈ T(1) rtol=T(2e-6)
    # The inversion's own criterion is 32 eps(Float32) relative, so the
    # recovered temperature is checked at that scale and not tighter.
    @test ns.T_ion[I] ≈ T(300) rtol=T(1e-5)
    @test state_valid(state_report(ns, NQ))
    run!(ns, NQ; tfinal=T(1e-4), nmax=1)
    @test all(isfinite, parent(NQ))

    sg = StiffenedGas{T}(gamma=T(1.4), p_inf=zero(T), cv=T(2.5),
                         name="gas")
    wall = (SlipWallBC(), SlipWallBC())
    ss = Solver(n_global=(24, 1, 1), L_domain=(one(T), one(T), one(T)),
                bcs=(wall, per3[2], per3[3]), eos=sg,
                filter_interval=0, transport=tn.transport, art=tn.art,
                deriv=tn.deriv, filt=tn.filt)
    SQ = allocate_state(ss)
    initialize!(ss, SQ, (x, y, z) ->
        Prim(u=(0.1sin(T(π) * x), 0, 0), p=1, rho=1))
    apply_bcs!(ss, SQ)
    m1 = ss.equations.i_mom[1]
    @test SQ[padded_index(ss, 1, 1, 1), m1] == zero(T)
    @test SQ[padded_index(ss, 24, 1, 1), m1] == zero(T)
    SdQ = zero(SQ)
    compute_rhs!(ss, SQ, SdQ)
    @test all(isfinite, parent(SdQ))
    run!(ss, SQ; tfinal=T(1e-4), nmax=1)
    @test all(isfinite, parent(SQ))
end

@testset "Float32 static two-level AMR step" begin
    T = Float32
    solver = Solver(n_global=(48, 1, 1),
                    L_domain=(T(2π), one(T), one(T)), bcs=per3,
                    transport=ConstantTransport{T}(),
                    art=ArtificialProperties{T}(enabled=false),
                    deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T),
                    filter_interval=0, cfl=T(0.2),
                    refine=BlockRegion((20, 0, 0), (8, 1, 1)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(0.5, 0, 0), p=1, rho=1 + 0.1sin(x)))
    @test all(Q -> eltype(Q) === T, states)
    @test compute_dt(solver, states) isa T
    run!(solver, states; tfinal=T(0.01), nmax=10)
    @test solver.t isa T
    @test all(Q -> all(isfinite, parent(Q)), states)
end

@testset "Float32 varying-cp mixture and open/viscous boundaries" begin
    T = Float32
    typed = (transport=ConstantTransport{T}(), art=ArtificialProperties{T}(enabled=false),
             deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T))

    # Both fits are tabulated over the CEA default range of 200 K to 6000 K and
    # the states below are physical, so the mixture is evaluated where its data
    # is defined. The first species has cp/R = 3.5 + 1e-3 T, which exercises the
    # iterative temperature recovery rather than its constant-cp limit.
    varying = Nasa9Species{T}(
        name="varying", R=one(T),
        a=(zero(T), zero(T), T(3.5), T(1e-3), zero(T), zero(T), zero(T)))
    inert = CL.nasa9_constant_cp(T, "inert", T(0.7), T(2.8))
    eos = Nasa9Mixture([varying, inert])
    @test CL.species_cp(eos, 1, T(600)) > CL.species_cp(eos, 1, T(300))
    ns = Solver(; n_global=(12, 12, 12),
                L_domain=(T(2π), T(2π), T(2π)), bcs=per3, eos=eos,
                filter_interval=0, typed...)
    NQ = allocate_state(ns)
    initialize!(ns, NQ, (x, y, z) ->
        Prim(u=(0.1, 0, 0), p=1, T_ion=300, Y=(0.3, 0.7)))
    CL.exchange_state!(NQ, ns.decomp)
    CL.primitives!(ns, NQ)
    I = padded_index(ns, 3, 4, 5)
    @test ns.p[I] ≈ T(1) rtol=T(2e-6)
    @test ns.T_ion[I] ≈ T(300) rtol=T(1e-5)
    @test ns.Y[1][I] ≈ T(0.3) rtol=T(2e-6)
    @test state_valid(state_report(ns, NQ))
    run!(ns, NQ; tfinal=T(1e-4), nmax=1)
    @test all(isfinite, parent(NQ))

    per = (PeriodicBC(), PeriodicBC())
    uin = (T(0.2), zero(T), zero(T))
    sb = Solver(; n_global=(32, 1, 1),
                L_domain=(one(T), one(T), one(T)),
                bcs=((NSCBCInflowBC(u=uin, T_ion=one(T), Y=T[1]),
                      NSCBCOutflowBC(pinf=one(T))), per, per),
                eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)), filter_interval=0, typed...)
    BQ = allocate_state(sb)
    initialize!(sb, BQ, (x, y, z) -> Prim(u=uin, p=1, T_ion=1))
    apply_bcs!(sb, BQ)
    BdQ = zero(BQ)
    compute_rhs!(sb, BQ, BdQ)
    matched = maximum(abs(BdQ[padded_index(sb, i, 1, 1), c])
                      for i in (1, 32), c in 1:sb.equations.n_cons)
    @test matched < T(5e-5)
    @test sb.bcs[1][1] isa NSCBCInflowBC{T}
    @test sb.bcs[1][2] isa NSCBCOutflowBC{T}
    run!(sb, BQ; tfinal=T(1e-4), nmax=1)
    @test all(isfinite, parent(BQ))

    Twall = T(1.2)
    sw = Solver(; n_global=(24, 1, 1),
                L_domain=(one(T), one(T), one(T)),
                bcs=((NoSlipWallBC(Twall=Twall),
                      NoSlipWallBC(Twall=Twall)), per, per),
                eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)), filter_interval=0, typed...)
    WQ = allocate_state(sw)
    initialize!(sw, WQ, (x, y, z) ->
        Prim(u=(0.3, -0.2, 0.1), p=1, T_ion=1))
    apply_bcs!(sw, WQ)
    CL.exchange_state!(WQ, sw.decomp)
    CL.primitives!(sw, WQ)
    for i in (1, 24)
        Iw = padded_index(sw, i, 1, 1)
        @test sw.u[Iw] == sw.v[Iw] == sw.w[Iw] == zero(T)
        @test sw.T_ion[Iw] == Twall
    end
    @test sw.bcs[1][1] isa NoSlipWallBC{T}
end

@testset "Float32 resolved fold and moving subcycled level" begin
    T = Float32
    per = (PeriodicBC(), PeriodicBC())
    typed = (transport=ConstantTransport{T}(), art=ArtificialProperties{T}(enabled=false),
             deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T))

    sf = Solver(; n_global=(32, 16, 1),
                L_domain=(one(T), T(2π), one(T)),
                metric=CylindricalMetric(),
                bcs=((AxisBC(), SlipWallBC()), per, per),
                filter_interval=0, typed...)
    f = CL.field(sf.decomp)
    df = similar(f)
    fillf!(sf, f, (r, θ, z) -> r * cos(θ) * exp(-T(4) * r^2))
    CL.exchange_halos!(f, sf.decomp)
    CL.deriv_along!(df, f, sf, 1, 1)
    CL._scale_grad!(df, sf, 1)
    @test ferr(sf, df, (r, θ, z) ->
        cos(θ) * (one(T) - T(8) * r^2) * exp(-T(4) * r^2)) < T(6e-5)

    wall = (SlipWallBC(), SlipWallBC())
    sr = Solver(n_global=(96, 1, 1),
                L_domain=(one(T), one(T), one(T)),
                bcs=(wall, per, per), cfl=T(0.1),
                subcycle=true, regrid_interval=1,
                refine=BlockRegion((34, 0, 0), (28, 1, 1)),
                transport=ConstantTransport{T}(), art=ArtificialProperties{T}(),
                deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T))
    states = allocate_state(sr)
    initialize!(sr, states, (x, y, z) ->
        x < T(0.5) ? Prim(u=(0, 0, 0), p=1, rho=1) :
                     Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    initial_offset = CL.refined_region(sr).offset[1]
    run!(sr, states; tfinal=one(T), nmax=2)
    @test CL.refined_region(sr).offset[1] != initial_offset
    @test sr.t isa T
    @test all(Q -> eltype(Q) === T && all(isfinite, parent(Q)), states)
    for (ps, Q) in CL.eachpatch(sr, states)
        @test minimum(Q[padded_index(ps, i, 1, 1), 1]
                      for i in 1:ps.decomp.n_local[1]) > T(0.05)
    end
end

include("float32_validation.jl")

@testset "banded LU vs dense" begin
    for q in (1, 2), n in (9, 17)
        A = zeros(n, n)
        for i in 1:n, jj in max(1, i-q):min(n, i+q)
            A[i, jj] = (i == jj ? 3.0 : 0.0) + randn()
        end
        Ab = zeros(2q + 1, n)
        for i in 1:n, ss in -q:q
            1 <= i + ss <= n && (Ab[q+1+ss, i] = A[i, i+ss])
        end
        F = CL.BandFactor(Ab, q)
        x = randn(n); b = A * x
        y = copy(b); CL.solve_col!(y, F)
        @test y ≈ x atol = 1e-9 rtol = 1e-9
    end
end

@testset "reduced interface band LU vs dense" begin
    # The reduced stage of P ranks from random spike corner blocks: every
    # rank's band solve against a dense solve of the same matrix, closed and
    # periodic, the second with spikes large enough that the band LU pivots.
    rng = MersenneTwister(12)
    for q in (1, 2), periodic in (false, true), P in (2, 3, 4, 7), scale in (0.4, 1.5)
        m2, L = 2q, CL.REDUCED_BLOCK + 22   # two blocks of lines
        allb = zeros(4q * q * P)
        for rk in 0:(P-1), block in 0:3, t in 1:q, r in 1:q
            # V toward the previous rank is absent at a closed line's low end,
            # W toward the next at its high end.
            absent = !periodic && ((rk == 0 && block <= 1) || (rk == P - 1 && block >= 2))
            allb[4q*q*rk + block*q*q + (t-1)*q + r] =
                absent ? 0.0 : scale * (2rand(rng) - 1)
        end
        R = zeros(m2 * P, m2 * P)
        CL._reduced_entries!((i, j, v) -> (R[i, j] += v), allb, q, P)
        gath = randn(rng, m2, L, P)
        z = reduce(vcat, [gath[:, :, rk+1] for rk in 0:(P-1)])
        exact = R \ z
        for p in 0:(P-1)
            red, band = CL._reduced_factor(allb, q, P, p, periodic, L)
            @test red === nothing
            @test CL._band_matrix(band) ≈ R rtol = 1e-14
            zbp, zbn = zeros(L, q), zeros(L, q)
            CL._band_reduced!(band, gath, zbp, zbn, q, L)
            cprev, cnext = m2 * mod(p - 1, P) + q, m2 * mod(p + 1, P)
            tol = 1e-13 * cond(R)
            @test maximum(abs, zbp' .- exact[cprev+1:cprev+q, :]) < tol
            @test maximum(abs, zbn' .- exact[cnext+1:cnext+q, :]) < tol
            # Bitwise independent of the number of lines per call.
            zbp2, zbn2 = zeros(L, q), zeros(L, q)
            CL._band_reduced!(band, gath, zbp2, zbn2, q, 5)
            @test zbp2[1:5, :] == zbp[1:5, :] && zbn2[1:5, :] == zbn[1:5, :]
        end
    end
end

@testset "periodic C6 derivative: spectral accuracy" begin
    solver = mkslv(n_global=(32, 32, 32))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> sin(3x) * cos(2y))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (x, y, z) -> 3cos(3x) * cos(2y)) < 1e-4  # C6 at k=3 on 32³
    CL.deriv_along!(df, f, solver, 2, 1); CL._scale_grad!(df, solver, 2)
    @test ferr(solver, df, (x, y, z) -> -2sin(3x) * sin(2y)) < 2e-5
    CL.deriv_along!(df, f, solver, 3, 1); CL._scale_grad!(df, solver, 3)
    @test ferr(solver, df, (x, y, z) -> 0.0) < 1e-10
end

@testset "pentadiagonal C10 derivative: accuracy vs C6" begin
    # Exercises the q=2 band path end to end: BandPlan assembly, the banded
    # LU + solve_col!, and the periodic self-coupling reduced-interface solve
    # (BandLineSolver) in all three dimensions — plus the closed-domain
    # closure rows. The only other C10 coverage is a finiteness smoke test.
    solver = mkslv(n_global=(32, 32, 32), deriv=lele_d1_10())
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> sin(3x) * cos(2y))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    e10 = ferr(solver, df, (x, y, z) -> 3cos(3x) * cos(2y))
    @test e10 < 1e-7     # C10 at k=3 on 32³ (measured ≈ 2.8e-8)
    CL.deriv_along!(df, f, solver, 2, 1); CL._scale_grad!(df, solver, 2)
    @test ferr(solver, df, (x, y, z) -> -2sin(3x) * sin(2y)) < 1e-8   # transposed path
    CL.deriv_along!(df, f, solver, 3, 1); CL._scale_grad!(df, solver, 3)
    @test ferr(solver, df, (x, y, z) -> 0.0) < 1e-12
    # C10 error on this field is ≈2000× below C6; the guard requires 100×.
    s6 = mkslv(n_global=(32, 32, 32), deriv=lele_d1_6())
    f6 = CL.field(s6.decomp); df6 = CL.field(s6.decomp)
    fillf!(s6, f6, (x, y, z) -> sin(3x) * cos(2y))
    CL.exchange_halos!(f6, s6.decomp)
    CL.deriv_along!(df6, f6, s6, 1, 1); CL._scale_grad!(df6, s6, 1)
    @test e10 < ferr(s6, df6, (x, y, z) -> 3cos(3x) * cos(2y)) / 100
    # Closed domain: deg-3 polynomial is exact through the C10 closure rows
    # (a distinct band path: closure substitution, V = W = 0, no reduced stage).
    sc = Solver(n_global=(32, 12, 12), L_domain=(1.0, 1.0, 1.0),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                deriv=lele_d1_10(), art=ArtificialProperties(enabled=false))
    fc = CL.field(sc.decomp); dfc = CL.field(sc.decomp)
    fillf!(sc, fc, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(fc, sc.decomp)
    CL.deriv_along!(dfc, fc, sc, 1, 1); CL._scale_grad!(dfc, sc, 1)
    @test ferr(sc, dfc, (x, y, z) -> 2 + 6x - 3x^2) < 1e-10
end

@testset "C10 interface closures: two rows per end, exact to degree 9" begin
    # The pentadiagonal scheme's patch-interface rows: the two edge rows
    # whose left-hand side would couple a ghost unknown become compact rows
    # Taylor-matched to order 10 on a 9-point right-hand side, row 1
    # one-sided on the left-hand side (nodes 1, 2, 3) and reading four ghost
    # layers, row 2 a centered tridiagonal row. A closed line whose ghosts
    # carry the analytic function must therefore differentiate a degree-9
    # polynomial to round-off through both ends, where the scheme's own C6
    # cascade would leave a third-order error.
    rows = CL.interface_closures(lele_d1_10())
    @test length(rows) == 2
    @test [r.first for r in rows] == [-3, -2]
    @test rows[1].lhs == [0, 0, 1, 8 / 5, 2 / 5]
    @test rows[2].lhs == [0, 2 / 5, 1, 2 / 5, 0]
    @test all(r -> abs(sum(r.rhs)) < 1e-15, rows)
    @test all(r -> length(r.rhs) == 9, rows)
    ring = CL.interface_closures(compact_d8())
    @test [(r.first, r.rhs) for r in ring] == [(1, [1.0]), (2, [1.0])]
    # The detector's own interface rows keep the undivided eighth difference:
    # degree-7 polynomials are annihilated, and the response at k = π is
    # the interior's 16, row 2 also matching its k → 0 ratio 1/232.
    d8rows = CL._ring_interface_rows(Float64)
    @test [r.first for r in d8rows] == [-3, -2]
    @test all(r -> all(n -> abs(sum(r.rhs .* (-4:4) .^ n)) < 1e-12, 0:7), d8rows)
    nyquist(r) = sum(r.rhs .* (-1) .^ (-4:4)) / (1 - 2r.lhs[2])
    @test all(r -> nyquist(r) ≈ 16, d8rows)
    @test d8rows[2].rhs[5] / 70 / (1 + 2d8rows[2].lhs[2]) ≈ 1 / 232
    n = 40
    h = 0.1
    d = Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1))
    plan = CL.plan_direction(d, lele_d1_10(), 1, h; lo_closures=rows, hi_closures=rows)
    p9(x) = 1 + x - 0.5x^2 + 0.3x^3 - 0.2x^4 + 0.1x^5 - 0.05x^6 + 0.02x^7 -
            0.01x^8 + 0.004x^9
    dp9(x) = 1 - x + 0.9x^2 - 0.8x^3 + 0.5x^4 - 0.3x^5 + 0.14x^6 - 0.08x^7 +
             0.036x^8
    f = CL.field(d); df = CL.field(d)
    pad = d.n_halo_d
    for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
        f[i, j, k] = p9((i - pad[1] - 1) * h)
    end
    CL.apply_along!(df, plan, f, d)
    e = maximum(abs(df[i+pad[1], pad[2]+1, pad[3]+1] - dp9((i - 1) * h)) for i in 1:n)
    @test e < 1e-10                     # measured 2.0e-12
    # The same line under the scheme's own closures is third order at the
    # edge rows, so the interface rows are what carry the exactness.
    own = CL.plan_direction(d, lele_d1_10(), 1, h)
    CL.apply_along!(df, own, f, d)
    eo = maximum(abs(df[i+pad[1], pad[2]+1, pad[3]+1] - dp9((i - 1) * h)) for i in 1:n)
    @test eo > 1e-4
    # A narrower halo than the rows read is refused with the width named.
    d3 = Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1), n_halo=3)
    @test_throws "n_halo ≥ 4" CL.plan_direction(d3, lele_d1_10(), 1, h;
                                                lo_closures=rows, hi_closures=rows)
end

@testset "pade_d1_4: fourth-order interior convergence" begin
    # The only coverage of the 4th-order tridiagonal scheme; a wrong
    # coefficient shows as a wrong slope, not a wrong level.
    errs = Float64[]
    for N in (16, 32, 64)
        solver = mkslv(n_global=(N, 12, 12), deriv=pade_d1_4())
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, (x, y, z) -> sin(x))
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
        push!(errs, ferr(solver, df, (x, y, z) -> cos(x)))
    end
    p = log(errs[1] / errs[3]) / log(4)          # observed order over 16 -> 64
    @test 3.5 < p < 4.5
    @test errs[3] < errs[2] < errs[1]
end

@testset "closures on the transposed path: y and z walls" begin
    # operators.jl applies boundary closure rows through a DIFFERENT code path
    # for y/z (the transposed line gather) than for x. Every other closed-domain
    # test in this suite walls off x only, so those rows were never executed.
    # Deg-3 polynomial exactness is the same assertion the x test makes.
    for d in (2, 3)
        ng = ntuple(k -> k == d ? 32 : 12, 3)
        bcs = ntuple(k -> k == d ? (SlipWallBC(), SlipWallBC()) : per3[k], 3)
        solver = Solver(n_global=ng, L_domain=(1.0, 1.0, 1.0), bcs=bcs,
                   art=ArtificialProperties(enabled=false))
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        poly = t -> 1 + 2t + 3t^2 - t^3
        dpoly = t -> 2 + 6t - 3t^2
        fillf!(solver, f, (x, y, z) -> poly(d == 2 ? y : z))
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, d, 1); CL._scale_grad!(df, solver, d)
        @test ferr(solver, df, (x, y, z) -> dpoly(d == 2 ? y : z)) < 1e-10
    end
end

@testset "C10 through a coordinate-singularity fold" begin
    # operators_banded.jl folds the ghost-unknown coupling onto the diagonal
    # for the pentadiagonal scheme (lo_fold/hi_fold). The C6 axis tests never
    # reach it and the C10 tests are all periodic or plain-walled.
    solver = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 0.5),
               metric=CylindricalMetric(), deriv=lele_d1_10(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtificialProperties(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * exp(-4r^2))            # odd across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, -1); CL._scale_grad!(df, solver, 1)
    # Both errors sit at the closed outer end, so they carry the default
    # closure's wall constant: measured 3.6e-6 and 2.1e-5 under `:neutral3`,
    # 2.4e-6 and 1.6e-5 under `:cascade3`.
    @test ferr(solver, df, (r, θ, z) -> (1 - 8r^2) * exp(-4r^2)) < 6e-6
    fillf!(solver, f, (r, θ, z) -> exp(-4r^2))                # even across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> -8r * exp(-4r^2)) < 3e-5
end

@testset "transposed y/z path ≡ x path on permuted data" begin
    solver = mkslv(n_global=(24, 24, 24))
    f = CL.field(solver.decomp); g = CL.field(solver.decomp)
    d1 = CL.field(solver.decomp); d2 = CL.field(solver.decomp)
    fn = (x, y, z) -> sin(2x + 0.3) * cos(y) + 0.1z^0   # z-independent
    fillf!(solver, f, fn)                                     # varies in x
    fillf!(solver, g, (x, y, z) -> fn(y, x, z))               # same profile along y
    CL.exchange_halos!(f, solver.decomp); CL.exchange_halos!(g, solver.decomp)
    CL.deriv_along!(d1, f, solver, 1, 1)
    CL.deriv_along!(d2, g, solver, 2, 1)
    e = maximum(abs(d1[padded_index(solver, i, j, k)] - d2[padded_index(solver, j, i, k)])
                for i in 1:24, j in 1:24, k in 1:24)
    @test e < 1e-11
end

@testset "staggered operators: adjoint pair, telescoping, symmetry, sweep paths" begin
    # Dense matrices of the three staggered operators on one line of N nodes,
    # columns from unit vectors. The midpoint rows and columns are the first
    # N - 1 on a closed line, where slot N carries no midpoint.
    function staggered_matrices(N, periodic, parity)
        decomp = Decomp((N, 1, 1), (periodic, false, false))
        h = periodic ? 1 / N : 1 / (N - 1)
        plans = (CL.plan_staggered(decomp, :to_mid, 1, h; parity=parity),
                 CL.plan_staggered(decomp, :to_node, 1, h; parity=-parity),
                 CL.plan_staggered(decomp, :interpolate, 1, h; parity=1))
        f = field(decomp); out = field(decomp); pad = decomp.n_halo
        mats = map(plans) do plan
            M = zeros(N, N)
            for j in 1:N
                f .= 0; f[j+pad, 1, 1] = 1
                CL.exchange_dim!(f, decomp, 1)
                out .= NaN
                apply_along!(out, plan, f, decomp)
                M[:, j] = out[pad+1:pad+N, 1, 1]
            end
            M
        end
        return mats..., h
    end
    for N in (16, 17)
        D, G, Ip, h = staggered_matrices(N, true, 1)
        # Periodic: G is exactly the negative transpose of D, so L = -Dᵀ K D.
        @test norm(G + D') < 1e-12 * norm(D)
        @test norm(D * ones(N)) < 1e-12 * norm(D)
        @test norm(Ip * ones(N) .- 1) < 1e-14
        κ = 1 .+ 0.5 .* sin.(2π .* (0:N-1) ./ N)
        L = G * Diagonal(Ip * κ) * D
        @test norm(L - L') < 1e-12 * norm(L)
        @test maximum(eigvals(Symmetric((L + L') / 2))) < 1e-10 * norm(L)
        @test abs(sum(L * sin.(1:N))) < 1e-11 * norm(L)
    end
    for N in (16, 17), parity in (1, -1)
        D, G, Ip, h = staggered_matrices(N, false, parity)
        M = N - 1
        # The slot of midpoint N is zero whatever the input.
        @test all(iszero, D[N, :]) && all(iszero, Ip[N, :])
        D = D[1:M, :]; G = G[:, 1:M]; Ip = Ip[1:M, :]
        # Under the wall mirror the pair stays adjoint in the trapezoidal node
        # weights: W_n G = -Dᵀ W_m with W_m = h, on the nodes the parity
        # leaves free (all of them for the even temperature of an adiabatic
        # wall; the interior for the odd one of an isothermal wall).
        Wn = fill(h, N); Wn[1] = Wn[N] = h / 2
        free = parity == 1 ? (1:N) : (2:N-1)
        @test norm((Diagonal(Wn) * G .+ h .* D')[free, :]) < 1e-12 * norm(D) * h
        @test norm(Ip * ones(N) .- 1) < 1e-14
        κ = 1 .+ 0.5 .* cos.(π .* (0:N-1) ./ (N - 1))
        WL = (Diagonal(Wn) * G * Diagonal(Ip * κ) * D)[free, free]
        @test norm(WL - WL') < 1e-12 * norm(WL)
        @test maximum(eigvals(Symmetric((WL + WL') / 2))) < 1e-10 * norm(WL)
        if parity == 1
            # The even mirror is the adiabatic wall: a constant carries no
            # flux, and Σ W_n L T telescopes to the zero wall flux. (The odd
            # mirror passes flux through the wall and conserves nothing.)
            @test norm(D * ones(N)) < 1e-12 * norm(D)
            @test abs(sum(WL * sin.(1:N))) < 1e-11 * norm(WL)
        end
    end
    # The transposed y and z sweeps against the x sweep on permuted data, on a
    # decomposition with every dimension active.
    for periodic in (true, false), op in (:to_mid, :to_node, :interpolate)
        decomp = Decomp((12, 12, 12), (periodic, periodic, periodic))
        h = 1 / 12
        f = field(decomp); pad = decomp.n_halo
        for k in 1:12, j in 1:12, i in 1:12
            f[i+pad, j+pad, k+pad] = sin(0.7i + 0.2) * cos(0.3j) + 0.1k
        end
        CL.exchange_halos!(f, decomp)
        outs = map(1:3) do d
            out = field(decomp)
            g = permutedims(f, d == 1 ? (1, 2, 3) : d == 2 ? (2, 1, 3) : (3, 2, 1))
            apply_along!(out, CL.plan_staggered(decomp, op, d, h), g, decomp)
            permutedims(out, d == 1 ? (1, 2, 3) : d == 2 ? (2, 1, 3) : (3, 2, 1))
        end
        inner = ntuple(_ -> pad+1:pad+12, 3)
        @test maximum(abs.(outs[2][inner...] .- outs[1][inner...])) < 1e-12
        @test maximum(abs.(outs[3][inner...] .- outs[1][inner...])) < 1e-12
    end
end

@testset "staggered operators: half-offset folds and curvilinear metric" begin
    # A line of N nodes folded half a cell beyond an end is half of a longer
    # line carrying the mirrored data: between two planes, a periodic line of
    # 2N nodes; with a plane at one end and a wall at the other, a closed
    # line of 2N nodes with walls of the same parity at both ends. Every
    # operator on the folded line, the plane midpoints included (slot 0 at a
    # low plane, slot N at a high one), must equal the doubled line's.
    N = 12; h = 1 / N
    smooth(x) = exp(sin(3x) + 0.3cos(7x)) + 0.2x
    for (lo, hi) in ((true, true), (true, false), (false, true)), σ in (1, -1),
        p in (1, -1), op in (:to_mid, :to_node, :interpolate)
        (lo && hi) && p == -1 && continue          # no wall to take p
        periodic = lo && hi
        df = Decomp((N, 1, 1), (false, false, false))
        dd = Decomp((2N, 1, 1), (periodic, false, false))
        mid_in = op === :to_node
        # Node j of the doubled line sits at (j - N - 1/2) h and its midpoint
        # j at (j - N) h, with a plane at x = 0 (and between two planes a
        # second one at x = N h, the periodic wrap). The folded line is its
        # upper half below a low plane (index i + N) and its lower half above
        # a high one (index i).
        shift = lo ? N : 0
        onplane(x) = abs(x) < h / 4 || (periodic && abs(abs(x) - N * h) < h / 4)
        value(x) = onplane(x) && mid_in && σ < 0 ? 0.0 :
                   x >= 0 ? smooth(x) : σ * smooth(-x)
        pf = CL.plan_staggered(df, op, 1, h; parity=p, lo_fold=lo ? σ : nothing,
                               hi_fold=hi ? σ : nothing)
        pd = CL.plan_staggered(dd, op, 1, h; parity=p)
        pad = df.n_halo
        ff = field(df); fd = field(dd)
        for j in 1:2N
            fd[j+pad, 1, 1] = value((j - N - (mid_in ? 0.0 : 0.5)) * h)
        end
        for i in (mid_in && lo ? 0 : 1):N
            ff[i+pad, 1, 1] = fd[i+shift+pad, 1, 1]
        end
        CL.exchange_dim!(fd, dd, 1)
        outf = fill!(field(df), NaN); outd = field(dd)
        apply_along!(outf, pf, ff, df)
        apply_along!(outd, pd, fd, dd)
        rows = mid_in ? (1:N) : ((lo ? 0 : 1):N)
        e = maximum(abs(outf[i+pad, 1, 1] - outd[i+shift+pad, 1, 1]) for i in rows)
        @test e < 1e-12 * maximum(abs, outd[pad+1:pad+2N, 1, 1])
    end

    # The adjoint identity between two planes, with weight h on every node
    # and h/2 on the plane midpoints (slots 0 and N), holds for both parities,
    # and W_n L is symmetric, negative semidefinite and, for the even
    # temperature, conservative.
    for σ in (1, -1)
        df = Decomp((N, 1, 1), (false, false, false))
        plans = (CL.plan_staggered(df, :to_mid, 1, h; lo_fold=σ, hi_fold=σ),
                 CL.plan_staggered(df, :to_node, 1, h; lo_fold=-σ, hi_fold=-σ),
                 CL.plan_staggered(df, :interpolate, 1, h; lo_fold=1, hi_fold=1))
        pad = df.n_halo
        mids = 0:N
        dense(plan, rows, cols) = begin
            M = zeros(length(rows), length(cols)); f = field(df); out = field(df)
            for (c, j) in enumerate(cols)
                f .= 0; f[j+pad, 1, 1] = 1
                out .= NaN
                apply_along!(out, plan, f, df)
                M[:, c] = [out[i+pad, 1, 1] for i in rows]
            end
            M
        end
        D = dense(plans[1], mids, 1:N)
        G = dense(plans[2], 1:N, mids)
        Ip = dense(plans[3], mids, 1:N)
        Wm = fill(h, N + 1); Wm[1] = Wm[end] = h / 2
        @test norm(h .* G .+ D' * Diagonal(Wm)) < 1e-12 * norm(D) * h
        κ = 1 .+ 0.5 .* cospi.(((1:N) .- 0.5) ./ N)
        WL = h .* G * Diagonal(Ip * κ) * D
        @test norm(WL - WL') < 1e-12 * norm(WL)
        @test maximum(eigvals(Symmetric((WL + WL') / 2))) < 1e-10 * norm(WL)
        σ == 1 && @test abs(sum(WL * sin.(1:N))) < 1e-11 * norm(WL)
    end

    # The metric: J W_n L with J W_n the node volumes is symmetric, negative
    # semidefinite and conservative, tested on random vectors over the block,
    # on every geometry the solver builds but the folds whose area vanishes
    # oddly (the axis, the poles). There the flux continues smoothly through
    # the singular set, which is not the mirror adjoint to D_s, and the
    # defect is bounded instead. A constant carries no flux anywhere. W_n is
    # h, and h/2 on a wall node.
    noart = ArtificialProperties(enabled=false)
    walls = (SlipWallBC(), SlipWallBC())
    geometries = (
        ("stretched line", 1, true, Solver(n_global=(17, 1, 1), L_domain=(1.0, 1.0, 1.0),
            bcs=(walls, per3[2], per3[3]), art=noart,
            stretch=(sine_cluster(0.0, 1.0, 0.3, 0.4), nothing, nothing))),
        ("cylindrical shell", 1, true, Solver(n_global=(17, 1, 1), L_domain=(1.0, 1.0, 1.0),
            metric=CylindricalMetric(), origin=(0.5, 0.0, 0.0),
            bcs=(walls, per3[2], per3[3]), art=noart)),
        ("cylindrical axis", 1, false, Solver(n_global=(16, 1, 1), L_domain=(1.0, 1.0, 1.0),
            metric=CylindricalMetric(), bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
            art=noart)),
        ("resolved-θ axis", 1, false, Solver(n_global=(12, 12, 1), L_domain=(1.0, 2π, 1.0),
            metric=CylindricalMetric(), bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
            art=noart)),
        ("cylindrical θ", 2, true, Solver(n_global=(12, 12, 1), L_domain=(1.0, 2π, 1.0),
            metric=CylindricalMetric(), origin=(0.5, 0.0, 0.0),
            bcs=(walls, per3[2], per3[3]), art=noart)),
        ("spherical origin", 1, true, Solver(n_global=(12, 12, 12), L_domain=(1.0, π, 2π),
            metric=SphericalMetric(), bcs=((OriginBC(), SlipWallBC()),
            (PoleBC(), PoleBC()), per3[3]), art=noart)),
        ("spherical poles", 2, false, Solver(n_global=(12, 12, 12), L_domain=(1.0, π, 2π),
            metric=SphericalMetric(), bcs=((OriginBC(), SlipWallBC()),
            (PoleBC(), PoleBC()), per3[3]), art=noart)),
    )
    for (label, d, symmetric, solver) in geometries
        decomp = solver.decomp
        op = CL.StaggeredDiffusion(solver, d)
        pad = decomp.n_halo_d
        inner = CartesianIndices(ntuple(k -> pad[k]+1:pad[k]+decomp.n_local[k], 3))
        n = decomp.n_local[d]
        closed = !decomp.periodic[d]
        wall_lo = closed && (solver.folds[d] === nothing || !solver.folds[d].lo)
        wall_hi = closed && (solver.folds[d] === nothing || !solver.folds[d].hi)
        vol = zeros(size(solver.inv_J))
        for I in inner
            i = I[d] - pad[d]
            w = solver.h[d] * ((i == 1 && wall_lo) || (i == n && wall_hi) ? 0.5 : 1.0)
            vol[I] = w / solver.inv_J[I]
        end
        κ = field(decomp)
        for I in inner
            x = Tuple(I) .* 0.37
            κ[I] = 1 + 0.3sin(x[1] + 2x[2]) * cos(x[3])
        end
        apply(u) = (out = field(decomp); t = copy(u);
                    CL.staggered_diffusion!(out, op, t, copy(κ), decomp); out)
        inprod(u, v) = sum(vol[I] * u[I] * v[I] for I in inner)
        one_f = field(decomp); one_f[inner] .= 1
        L1 = apply(one_f)
        @test maximum(abs, L1[inner]) < 1e-9
        for trial in 1:4
            u = field(decomp); v = field(decomp)
            u[inner] .= randn(size(inner)); v[inner] .= randn(size(inner))
            Lu, Lv = apply(u), apply(v)
            tol = symmetric ? 1e-11 : 1e-2
            @test abs(inprod(u, Lv) - inprod(v, Lu)) < tol * abs(inprod(u, Lu))
            @test inprod(u, Lu) < 0
            @test abs(inprod(one_f, Lu)) < tol * abs(inprod(u, Lu))
        end
    end
end

@testset "implicit diffusion stage" begin
    # Tridiagonal lines with their own coefficients against a dense solve,
    # closed and periodic, on one rank (the MPI suite splits them).
    for periodic in (false, true)
        n, L = 7, 3
        vl = CL.VariableLines{Float64}(n, L, MPI.COMM_SELF, 1, 0, periodic)
        vl.lower .= -rand(n, L); vl.upper .= -rand(n, L)
        periodic || (vl.lower[1, :] .= 0; vl.upper[n, :] .= 0)
        vl.diag .= 0.1 .+ rand(n, L) .- vl.lower .- vl.upper
        CL.factor_lines!(vl)
        B = randn(n, L); X = CL.solve_variable_lines!(copy(B), vl)
        for l in 1:L
            A = diagm(vl.diag[:, l])
            for i in 1:n
                A[i, mod1(i - 1, n)] += vl.lower[i, l]
                A[i, mod1(i + 1, n)] += vl.upper[i, l]
            end
            @test norm(A * X[:, l] - B[:, l]) < 1e-12 * norm(B[:, l])
        end
    end

    # ∇·(κ∇T) in the metric's coordinates, the inner derivative by a complex
    # step and the outer by a sixth-order central difference.
    function divergence(metric, active, Tfn, κfn, x)
        J(y) = prod(CL.scalefactors(metric, y...))
        total = 0.0
        for d in 1:3
            active[d] || continue
            e = ntuple(k -> k == d ? 1.0 : 0.0, 3)
            flux(s) = (y = x .+ s .* e;
                       J(y) / CL.scalefactors(metric, y...)[d]^2 * κfn(y...) *
                       imag(Tfn((y .+ 1e-30im .* e)...)) / 1e-30)
            δ = 1e-4
            total += sum(c * (flux(m * δ) - flux(-m * δ))
                         for (m, c) in enumerate((3 / 4, -3 / 20, 1 / 60))) / δ
        end
        return total / J(x)
    end
    noart = ArtificialProperties(enabled=false)
    walls = (SlipWallBC(), SlipWallBC())
    axis = ((AxisBC(), SlipWallBC()), per3[2], per3[3])
    gauss(r) = exp(-16r^2)
    # name, solver of scale N, wall parity, whether V L is symmetric, the
    # manufactured order and its tolerance, T and κ. The shell's curved walls
    # are first order at the wall node, which the solve carries into a
    # second-order solution; on the resolved axis T is negligible at the wall.
    cases = (
        ("walls", N -> Solver(n_global=(N + 1, N + 1, 1), L_domain=(1.0, 1.0, 1.0),
                              bcs=(walls, walls, per3[3]), art=noart), 1, true, 6.0,
         (x, y, z) -> cospi(x) * cospi(y) + 0.3cospi(2x),
         (x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
        ("isothermal walls", N -> Solver(n_global=(N + 1, N + 1, 1),
                                         L_domain=(1.0, 1.0, 1.0),
                                         bcs=(walls, walls, per3[3]), art=noart),
         -1, true, 6.0, (x, y, z) -> sinpi(x) * sinpi(y) * (1 + 0.3cospi(x)),
         (x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
        ("symmetry planes", N -> Solver(n_global=(N, N + 1, 1), L_domain=(1.0, 1.0, 1.0),
                                        bcs=((SymmetryPlaneBC(), SymmetryPlaneBC()), walls,
                                             per3[3]), art=noart), 1, true, 6.0,
         (x, y, z) -> cospi(x) * cospi(y) + 0.3cospi(2x),
         (x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
        ("stretched walls", N -> Solver(n_global=(N + 1, N + 1, 1), L_domain=(1.0, 1.0, 1.0),
                                        bcs=(walls, walls, per3[3]), art=noart,
                                        stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing,
                                                 nothing)), 1, true, 6.0,
         (x, y, z) -> cospi(x) * cospi(y) + 0.3cospi(2x),
         (x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
        ("cylindrical shell", N -> Solver(n_global=(N + 1, N, 1), L_domain=(1.0, 2π, 1.0),
                                          metric=CylindricalMetric(),
                                          origin=(0.5, 0.0, 0.0),
                                          bcs=(walls, per3[2], per3[3]), art=noart),
         1, true, 2.0, (r, θ, z) -> cospi(r - 0.5) * (1 + 0.3cos(θ)),
         (r, θ, z) -> 1 + 0.3r * sin(θ)),
        ("resolved axis", N -> Solver(n_global=(N, N, 1), L_domain=(1.5, 2π, 1.0),
                                      metric=CylindricalMetric(), bcs=axis, art=noart),
         1, false, 6.0, (r, θ, z) -> gauss(r) * (1 + r * cos(θ)),
         (r, θ, z) -> 1 + 0.5r^2 + 0.2r * sin(θ)),
    )
    γΔt = 0.01
    for (label, build, parity, symmetric, order, Tfn, κfn) in cases
        errs = Float64[]; hs = Float64[]
        for N in (16, 32)
            solver = build(N)
            decomp = solver.decomp
            stage = CL.DiffusionStage(solver; parity=parity)
            @test stage.symmetric == symmetric
            κ = field(decomp); exact = field(decomp); rhs = field(decomp)
            fillf!(solver, κ, κfn); fillf!(solver, exact, Tfn)
            fillf!(solver, rhs, (x...) -> Tfn(x...) - γΔt * divergence(
                solver.metric, decomp.active, Tfn, κfn, Float64.(x)))
            pad = CartesianIndex(decomp.n_halo_d)
            for I in CartesianIndices(decomp.n_local)
                stage.dirichlet[I] && (rhs[I+pad] = exact[I+pad] = 0)
            end
            T = copy(rhs)
            result = CL.solve_stage!(T, stage, rhs, κ, γΔt)
            @test result.converged
            inner = interior(decomp)
            push!(errs, maximum(abs, T[inner] .- exact[inner]))
            push!(hs, solver.h[1])
            # An odd temperature stays zero on the wall nodes.
            parity < 0 && @test all(T[I+pad] == 0 for I in CartesianIndices(decomp.n_local)
                                    if stage.dirichlet[I])
            N == 16 || continue
            # The multigrid cycle is a symmetric positive definite operator.
            u = field(decomp); v = field(decomp); Bu = field(decomp); Bv = field(decomp)
            u[inner] .= randn(size(inner)); v[inner] .= randn(size(inner))
            CL._precondition!(Bu, stage, u); CL._precondition!(Bv, stage, v)
            @test abs(dot(u[inner], Bv[inner]) - dot(v[inner], Bu[inner])) <
                  1e-12 * abs(dot(u[inner], Bu[inner]))
            @test dot(u[inner], Bu[inner]) > 0
            # A uniform right-hand side returns unchanged without iterating.
            parity < 0 && continue
            rhs .= 1.3; T .= 1.3
            uniform = CL.solve_stage!(T, stage, rhs, κ, 10.0)
            @test uniform.iterations == 0
            @test maximum(abs, T[inner] .- 1.3) < 1e-13
        end
        @test log(errs[1] / errs[2]) / log(hs[1] / hs[2]) > order - 0.3
    end

    # Iteration counts flat in the grid and in the step, from the explicit
    # limit to four orders beyond it, for conjugate gradients on the walls and
    # GMRES on the resolved axis.
    for (build, parity) in ((cases[1][2], 1), (cases[6][2], 1))
        counts = Int[]
        for N in (16, 32), γ in (1e-3, 1e-1, 1e1)
            solver = build(N)
            stage = CL.DiffusionStage(solver; parity=parity)
            κ = fillf!(solver, field(solver.decomp), (x, y, z) -> 1 + 0.5x * sin(3y))
            rhs = fillf!(solver, field(solver.decomp),
                         (x, y, z) -> exp(-3(x - 0.3)^2) * cos(y))
            result = CL.solve_stage!(copy(rhs), stage, rhs, κ, γ)
            @test result.converged
            push!(counts, result.iterations)
        end
        @test maximum(counts) <= 20
        @test all(abs.(counts[4:6] .- counts[1:3]) .<= 3)
    end

    # A spherical ball with its origin and poles, on the GMRES path: the
    # residual of the solve and a uniform right-hand side.
    solver = Solver(n_global=(12, 12, 12), L_domain=(1.5, π, 2π),
                    metric=SphericalMetric(),
                    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                    art=noart)
    decomp = solver.decomp
    stage = CL.DiffusionStage(solver)
    κ = fillf!(solver, field(decomp), (r, θ, φ) -> 1 + 0.5r^2 + 0.2r * sin(θ) * cos(φ))
    rhs = fillf!(solver, field(decomp), (r, θ, φ) -> gauss(r) * (1 + 0.5r * cos(θ)))
    T = copy(rhs)
    result = CL.solve_stage!(T, stage, rhs, κ, 0.1)
    out = CL.stage_operator!(field(decomp), stage, T, κ, 0.1)
    inner = interior(decomp)
    V = stage.volume[inner]
    @test result.converged
    @test sqrt(sum(V .* (out[inner] .- rhs[inner]) .^ 2) / sum(V .* rhs[inner] .^ 2)) < 1e-9
    rhs .= 2.0; T .= 2.0
    @test CL.solve_stage!(T, stage, rhs, κ, 10.0).iterations == 0
    @test maximum(abs, T[inner] .- 2) < 1e-13

    @test_throws ArgumentError CL.DiffusionStage(Solver(n_global=(16, 16, 1),
        L_domain=(1.0, 1.0, 1.0), bcs=per3, art=noart, patch_grid=(2, 1, 1)))
end

@testset "additive Runge–Kutta conduction" begin
    # The tableau: row sums, and every order condition of a two-part
    # additive pair to fourth order, the embedded weights to third. a^I is
    # exact; a^E is published to thirteen digits.
    big(A) = Rational{BigInt}.(A)
    AE, AI = big(CL.ARK436_EXPLICIT), big(CL.ARK436_IMPLICIT)
    b, bh, c = big(CL.ARK436_WEIGHTS), big(CL.ARK436_EMBEDDED), big(CL.ARK436_NODES)
    defect(x) = abs(Float64(x))
    @test all(i -> AE[i, i] == 0 && all(j -> AE[i, j] == 0 && AI[i, j] == 0, i+1:6), 1:6)
    @test all(i -> AI[i, i] == (i == 1 ? 0 : 1 // 4), 1:6)
    @test AI[6, :] == b                               # stiffly accurate
    one6 = ones(Rational{BigInt}, 6)
    @test maximum(defect, AE * one6 .- c) < 1e-24
    @test AI * one6 == c
    for (w, order) in ((b, 4), (bh, 3))
        conditions = Any[sum(w) - 1, w' * c - 1 // 2, w' * c .^ 2 - 1 // 3]
        for X in (AE, AI)
            push!(conditions, w' * (X * c) - 1 // 6)
        end
        if order == 4
            push!(conditions, w' * c .^ 3 - 1 // 4)
            for X in (AE, AI)
                push!(conditions, w' * (c .* (X * c)) - 1 // 8,
                      w' * (X * c .^ 2) - 1 // 12)
                for Y in (AE, AI)
                    push!(conditions, w' * (X * (Y * c)) - 1 // 24)
                end
            end
        end
        @test maximum(defect, conditions) < 1e-24
    end
    @test all(==(0), (b' * (AI * c) - 1 // 6, b' * (AI * (AI * c)) - 1 // 24,
                      bh' * (AI * c) - 1 // 6))

    # The conduction pulse on [-1, 1): c_v = 1, κ = α, and a gas constant small
    # enough that the flow the pulse drives stays below the time error
    # (bench/imexconduction.jl, which also measures the step rules).
    α, σ, A = 1.0, 0.2, 0.5
    t0 = σ^2 / (2α)
    noart = ArtificialProperties(enabled=false)
    pulse(x, t) = 1 + A * sqrt(t0 / (t + t0)) *
                  sum(exp(-(x + 2m)^2 / (4α * (t + t0))) for m in -4:4)
    function line(n; implicit=ImplicitConduction(step_rule=:none), gas=1e-10,
                  control=StepControl())
        h = 2 / n
        mu = 1e-3 * α * h^2
        Solver(n_global=(n, 1, 1), L_domain=(2.0, 1.0, 1.0), origin=(-1.0, 0.0, 0.0),
               bcs=per3, eos=IdealSpecies("gas"; R=gas, gamma=1 + gas),
               transport=ConstantTransport(mu0=mu, Pr=mu * (1 + gas) / α, Sc=1.0),
               art=noart, filter_interval=0, cfl=1e6, implicit=implicit,
               control=control)
    end
    function start(solver, T0)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                                 p=1e-10 * T0(x)))
        Q
    end
    function steps!(solver, Q, tf, n)
        for k in 1:n
            run!(solver, Q; tfinal=k == n ? tf : k * tf / n)
        end
        Q
    end
    temperature(solver, Q, i) =
        (refresh_primitives!(solver, Q); solver.T_ion[padded_index(solver, i, 1, 1)])

    n = 256
    limit = (2 / n)^2 / (2α)                          # forward-Euler diffusive limit
    tf = 4t0
    solver = line(n)
    @test solver.transport isa CL.WithoutConduction
    @test solver.implicit isa CL.ImexIntegrator
    Q = start(solver, x -> pulse(x, 0.0))
    nsteps = round(Int, tf / (100limit))             # R = 100
    steps!(solver, Q, tf, nsteps)
    refresh_primitives!(solver, Q)
    err = maximum(abs(solver.T_ion[padded_index(solver, i, 1, 1)] -
                      pulse(xcoord(solver, 1, i), tf)) for i in 1:n) / A
    @test err < 2e-5
    @test solver.implicit.accepted == nsteps && solver.implicit.rejected == 0
    @test solver.implicit.solves == 5nsteps          # one per implicit stage

    # Stiff stability: one step at R = 1e5 removes a grid-Nyquist mode.
    solver = line(64)
    h = 2 / 64
    Q = start(solver, x -> 1 + 0.01 * (-1)^round(Int, (x + 1) / h))
    nyquist(solver, Q) = abs(sum((-1)^(i - 1) * temperature(solver, Q, i)
                                 for i in 1:64) / 64)
    @test nyquist(solver, Q) ≈ 0.01
    run!(solver, Q; tfinal=1e5 * h^2 / (2α))
    @test nyquist(solver, Q) < 1e-6

    # A stage that does not converge within its Krylov budget ends the step
    # with SolverFailure, which takes StepControl's rollback: each retry
    # lowers the CFL and caps the step below the one that failed, and the
    # failure is raised once the retries are spent.
    tight = ImplicitConduction(step_rule=:none, max_iterations=2, rtol=1e-10)
    tf = 1e4 * h^2 / (2α)
    solver = line(64; implicit=tight)
    Q = start(solver, x -> pulse(x, 0.0))
    failure = try
        run!(solver, Q; tfinal=tf)
        nothing
    catch e
        e
    end
    @test failure isa SolverFailure && failure.reason === :implicit_solve
    @test failure.dt == tf
    solver = line(64; implicit=tight, control=StepControl(retries=2))
    Q = start(solver, x -> pulse(x, 0.0))
    failure = try
        run!(solver, Q; tfinal=tf)
        nothing
    catch e
        e
    end
    @test failure isa SolverFailure && failure.reason === :implicit_solve
    @test solver.cfl == 1e6 / 4 && failure.dt == tf / 4
    @test solver.implicit.dt_cap == tf / 8

    # The step rules: `:error` rejects and subdivides an attempt beyond its
    # tolerance, and the state lands at the requested endpoint either way.
    for rule in (ImplicitConduction(tolerance=1e-4),
                 ImplicitConduction(step_rule=:temperature, target_change=0.02))
        solver = line(128; implicit=rule)
        Q = start(solver, x -> pulse(x, 0.0))
        run!(solver, Q; tfinal=t0)
        @test solver.t == t0
        refresh_primitives!(solver, Q)
        @test maximum(abs(solver.T_ion[padded_index(solver, i, 1, 1)] -
                          pulse(xcoord(solver, 1, i), t0)) for i in 1:128) / A < 1e-2
    end

    # The default integrator is untouched, and the unsupported ends are
    # refused.
    plain = Solver(n_global=(16, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3, art=noart)
    @test plain.implicit === nothing && plain.transport isa ConstantTransport
    iso = NoSlipWallBC(Twall=1.0)
    @test_throws ArgumentError Solver(n_global=(16, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                      bcs=((iso, iso), per3[2], per3[3]), art=noart,
                                      implicit=ImplicitConduction())
    @test_throws ArgumentError Solver(n_global=(32, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                      bcs=per3, art=noart, patch_grid=(2, 1, 1),
                                      implicit=ImplicitConduction())
    @test_throws ArgumentError ImplicitConduction(step_rule=:splitting)
end

@testset "closed-domain closures: polynomial exactness (deg ≤ 3)" begin
    solver = Solver(n_global=(32, 12, 12), L_domain=(1.0, 1.0, 1.0),
               bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtificialProperties(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (x, y, z) -> 2 + 6x - 3x^2) < 1e-10
end

@testset "closure sets: exact through the closure order, not beyond" begin
    # Each closure set is exact for polynomials up to the order of its lowest
    # row and visibly inexact one degree higher, which pins both the
    # coefficients and the row count: a dropped row would expose an interior
    # stencil reading past the edge, and a wrong coefficient breaks exactness
    # at some degree ≤ 3. The Brady–Livescu rows are held to the same
    # tolerance despite their ~1e3 closed-line condition number.
    for (deriv, deg) in ((lele_d1_6(), 3),
                         (lele_d1_6(closures=:cascade3), 3),
                         (lele_d1_6(closures=:cascade4), 4),
                         (lele_d1_6(closures=:brady_livescu), 5),
                         (lele_d1_8(), 3),
                         (lele_d1_8(closures=:cascade3), 3),
                         (lele_d1_8(closures=:cascade4), 4),
                         (lele_d1_8(closures=:brady_livescu), 7),
                         (lele_d1_10(), 3),
                         (lele_d1_10(closures=:cascade3), 3))
        solver = Solver(n_global=(32, 16, 16), L_domain=(1.0, 1.0, 1.0),
                        bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                        deriv=deriv, art=ArtificialProperties(enabled=false))
        f = CL.field(solver.decomp); df = CL.field(solver.decomp)
        fillf!(solver, f, (x, y, z) -> sum(x^m for m in 0:deg))
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
        @test ferr(solver, df, (x, y, z) -> sum(m * x^(m - 1) for m in 1:deg)) < 1e-10
        fillf!(solver, f, (x, y, z) -> x^(deg + 1))
        CL.exchange_halos!(f, solver.decomp)
        CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
        @test ferr(solver, df, (x, y, z) -> (deg + 1) * x^deg) > 1e-7
    end
    @test_throws ErrorException lele_d1_6(closures=:unknown)
    # The C8 neutral set is the C6 neutral rows over the C6 interior row,
    # which is also the third cascade row.
    @test lele_d1_8().closures[1:2] == lele_d1_6().closures
    @test lele_d1_8().closures[3] == lele_d1_8(closures=:cascade3).closures[3]
    @test_throws ErrorException CL.neutral_closures(Float64, 4)
    # The C10 rows are the C8 rows with a zero outer band, whichever set is
    # selected, and the cascade set reproduces the rows the scheme carried
    # before it took a keyword.
    for j in 1:3
        @test lele_d1_10().closures[j].lhs == [0; lele_d1_8().closures[j].lhs...; 0]
        @test lele_d1_10().closures[j].rhs == lele_d1_8().closures[j].rhs
        @test lele_d1_10(closures=:cascade3).closures[j].lhs ==
            [0; lele_d1_8(closures=:cascade3).closures[j].lhs...; 0]
        @test lele_d1_10(closures=:cascade3).closures[j].rhs ==
            lele_d1_8(closures=:cascade3).closures[j].rhs
    end
    let rows = lele_d1_10(closures=:cascade3).closures
        @test rows[1].lhs == [0, 0, 1, 2, 0]
        @test rows[1].rhs == [-5//2, 2, 1//2]
        @test rows[2].lhs == [0, 1//4, 1, 1//4, 0]
        @test rows[2].rhs == [-3//4, 0, 3//4]
        @test rows[3].lhs == Float64[0, 1//3, 1, 1//3, 0]
        @test rows[3].rhs == Float64[-1//36, -7//9, 0, 7//9, 1//36]
        @test all(row -> row.first == 1, rows)
    end
    @test_throws ErrorException lele_d1_10(closures=:cascade4)
    @test_throws ErrorException lele_d1_10(closures=:brady_livescu)
    # An interface imposes no wall condition, so a neutral scheme falls back
    # to the cascade rows of its own width there, and a cascade scheme keeps
    # what it has.
    @test CL.interface_divergence_closures(lele_d1_8()) ==
        lele_d1_8(closures=:cascade3).closures
    @test length(CL.interface_divergence_closures(lele_d1_8())) == 3
    @test CL.interface_divergence_closures(lele_d1_8(closures=:cascade3)) ==
        lele_d1_8(closures=:cascade3).closures
    @test CL.interface_divergence_closures(lele_d1_6()) ==
        lele_d1_6(closures=:cascade3).closures
    @test CL.interface_divergence_closures(lele_d1_10()) ==
        lele_d1_10(closures=:cascade3).closures
    @test length(CL.interface_divergence_closures(lele_d1_10())) == 3
    @test CL.interface_divergence_closures(lele_d1_10(closures=:cascade3)) ==
        lele_d1_10(closures=:cascade3).closures
    @test CL.interface_divergence_closures(compact_d8()) == compact_d8().closures
    @test CL.interface_divergence_closures(pade_d1_4()) == pade_d1_4().closures
end

@testset "filter closures: one-sided rows, published row, wall exactness" begin
    af = 0.45
    # The derivation reproduces the interior stencil at the centered point
    # and Gaitonde–Visbal's tabulated row 2 (their (1 + 254αf)/256 and
    # (31 + 2αf)/32 leading entries), and follows the element type.
    base = compact_filter(af; closures=:cascade)
    r5 = CL.onesided_filter_row(af, 5, 4)
    @test maximum(abs.(r5 .- [reverse(base.coeffs); base.a0; base.coeffs])) < 1e-14
    os = compact_filter(af; closures=:onesided)
    @test abs(os.closures[2].rhs[1] - (1 + 254af) / 256) < 1e-14
    @test abs(os.closures[2].rhs[2] - (31 + 2af) / 32) < 1e-14
    @test os.closures[1].rhs == [1.0]
    os32 = compact_filter(0.45f0, Float32; closures=:onesided)
    @test eltype(os32.closures[2].rhs) === Float32
    @test_throws ErrorException compact_filter(af; closures=:unknown)
    # One closed-domain pass: the one-sided rows leave a degree-7 polynomial
    # unchanged; the cascade's F2 row alters anything above degree 1.
    for (filt, deg, exact) in ((os, 7, true), (base, 3, false))
        solver = Solver(n_global=(32, 12, 12), L_domain=(1.0, 1.0, 1.0),
                        bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                        filt=filt, art=ArtificialProperties(enabled=false))
        f = CL.field(solver.decomp); g = CL.field(solver.decomp)
        fillf!(solver, f, (x, y, z) -> sum(x^m for m in 0:deg))
        CL.exchange_halos!(f, solver.decomp)
        CL.filt_along!(g, f, solver, 1, 1)
        e = ferr(solver, g, (x, y, z) -> sum(x^m for m in 0:deg))
        @test exact ? e < 1e-12 : e > 1e-6
    end
end

@testset "filter: constants exact, Nyquist damped, parity of closures" begin
    solver = mkslv(n_global=(32, 12, 12))
    f = CL.field(solver.decomp)
    fillf!(solver, f, (x, y, z) -> 1.0)
    filter_field!(f, solver)
    @test ferr(solver, f, (x, y, z) -> 1.0) < 1e-12
    nx = solver.decomp.n_local[1]
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:nx
        f[padded_index(solver, i, j, k)] = 1.0 + 0.5 * (-1)^i        # constant + Nyquist
    end
    filter_field!(f, solver)
    dev = maximum(abs(f[padded_index(solver, i, 1, 1)] - 1.0) for i in 1:nx)
    @test dev < 0.35                                      # sawtooth strongly damped
end

@testset "filter relaxation: weight, exact blend, full strength at the reference" begin
    # The relaxed filter must be an exact linear interpolation between the state
    # and its filtered image, and must reduce to the unrelaxed pass at w = 1 —
    # bit-identically, since every guarded number in the suite was measured on
    # that path.
    ic = (x, y, z) -> Prim(rho=1.0 + 0.2sin(x) * cos(2y), p=1.0 + 0.1sin(3z),
                           u=(0.3sin(2x), 0.2cos(y), 0.1sin(z) * cos(x)))
    build(fc) = setup(Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                              transport=ConstantTransport(mu0=0.0),
                              domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)),
                              bcs=per3, ic=ic),
                      Numerics(n_global=(16, 16, 16), art=ArtificialProperties(enabled=false),
                               cfl=0.4, filter=StateFilter(; cfl=fc)))

    solver0, Q0 = build(0.0)
    @test CL.filter_weight(solver0, 1) == 1.0       # disabled: full strength

    # dt_prev = 0 before any step, where the unrelaxed meaning is the only one
    # available, so the weight must be 1 even with the relaxation configured.
    solverW, _ = build(0.4)
    @test CL.filter_weight(solverW, 1) == 1.0
    # The weight of a pass along d is dt · r_d · √n / filter_cfl on n active
    # dimensions, three here: r_d = 1/√3 makes dt · r_d · √3 the plain dt.
    r3 = 1 / sqrt(3.0)
    solverW.dt_prev = 0.2; solverW.filter_rate_prev = (r3, r3, r3)  # 0.2 / 0.4
    @test CL.filter_weight(solverW, 1) ≈ 0.5
    @test CL.filter_weight(solverW, 3) ≈ 0.5
    # Each direction reads its own rate: a slower direction filters less.
    solverW.filter_rate_prev = (r3, r3 / 2, r3 / 4)
    @test CL.filter_weight(solverW, 2) ≈ 0.25
    @test CL.filter_weight(solverW, 3) ≈ 0.125
    solverW.dt_prev = 2.0                            # dt · r_1 · √3 = 2, well over
    @test CL.filter_weight(solverW, 1) == 1.0        # capped, never over-filters

    # Interior only. `filter_state!` exchanges halos, so comparing whole padded
    # arrays measures halo initialization rather than the filter — a difference
    # of 2.83 against a filter effect three orders smaller.
    function interior(solver, Q)
        nx, ny, nz = solver.decomp.n_local
        [Q[padded_index(solver, i, j, k), c]
         for i in 1:nx, j in 1:ny, k in 1:nz, c in 1:solver.equations.n_cons]
    end
    dev(A, B) = maximum(abs, A .- B)

    ref = interior(solver0, Q0)                      # unfiltered state
    filter_state!(solver0, Q0)
    full = dev(interior(solver0, Q0), ref)           # fully filtered image
    @test full > 0

    solverB, QB = build(0.8)                         # w = dt · r_d · √3 / 0.8
    solverB.dt_prev = 0.4; solverB.filter_rate_prev = (r3, r3, r3)   # w = 0.5
    @test CL.filter_weight(solverB, 1) ≈ 0.5
    filter_state!(solverB, QB)
    # Filtering is applied dimension by dimension, so the composite is not
    # (1-w)Q + w·F(Q) exactly; it is that per direction. Check the property that
    # does hold globally and motivates the relaxation: the blended state
    # lies strictly between the two, and approaches each end as w does.
    @test 0 < dev(interior(solverB, QB), ref) < full

    solverS, QS = build(1000.0)                      # w = 4e-4, nearly no filter
    solverS.dt_prev = 0.4; solverS.filter_rate_prev = (r3, r3, r3)
    filter_state!(solverS, QS)
    @test dev(interior(solverS, QS), ref) < 0.01 * full

    # w == 1 must take the copy path, not the blend, so a run at or above the
    # reference CFL is the unrelaxed solver exactly.
    solverE, QE = build(0.4)
    solverE.dt_prev = 0.4                            # dt · r_d · √3 = 0.69 ≥ 0.4
    solverE.filter_rate_prev = (1.0, 1.0, 1.0)
    @test all(d -> CL.filter_weight(solverE, d) == 1.0, 1:3)
    filter_state!(solverE, QE)
    @test QE == Q0                                   # bit-identical, not approx
end

@testset "volume-weighted filter: identity, constants, and the measured defects" begin
    # `filter_weighting = :volume` is the form of the public Pyranda
    # implementation, F(J q) / F(J). bench/filter_conservation.jl measures it
    # against the unweighted operator on every geometry; the facts pinned here
    # are the ones the default rests on (reference/CALIBRATION_APPENDIX.md, the
    # filter on non-uniform volumes). One directional pass on a line is the
    # matrix M assembled from unit impulses, and Mᵀ V − V is its conservation
    # defect on the quadrature volumes V: the mass a pass creates on q is that
    # row vector applied to q.
    # 64 nodes, as the bench measures: on a shorter line the wall rows' leak
    # reaches the fold rows and the origin figures below are not round-off.
    # The figures were measured under the filter's cascade rows, so the lines
    # pin them; the one-sided rows, which are the default, move
    # the wall leak inward and are checked separately below.
    N = 64
    walls = ((SlipWallBC(), SlipWallBC()), per3[2], per3[3])
    line(wt; filt=compact_filter(0.45; closures=:cascade), kw...) =
        Solver(; n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), filt=filt,
               art=ArtificialProperties(enabled=false), filter_weighting=wt, kw...)
    function line_operator(s)
        Q = allocate_state(s)
        M = zeros(N, N)
        for n in 1:N
            fill!(parent(Q), 0.0)
            Q[padded_index(s, n, 1, 1), 1] = 1.0
            filter_state!(s, Q)
            for k in 1:N
                M[k, n] = Q[padded_index(s, k, 1, 1), 1]
            end
        end
        V = [CL.quad_weight(s, 1, n) / s.inv_J[padded_index(s, n, 1, 1)] for n in 1:N]
        return M, V
    end
    defect(s) = ((M, V) = line_operator(s); (M' * V .- V) ./ V)

    # A periodic line has unit column sums, so it conserves; a wall line does
    # not: the cascade closure rows carry a defect of a few percent of a
    # node's content that decays into the interior at the tridiagonal root,
    # 0.627 per row at α = 0.45, and is still above 1e-4 twelve rows in. On a
    # uniform volume the weighting is skipped and the operator is the same
    # matrix bit for bit.
    Mp, _ = line_operator(line(:none; bcs=per3))
    @test maximum(abs, sum(Mp; dims=1) .- 1) < 1e-13
    Mw, _ = line_operator(line(:none; bcs=walls))
    Mv, _ = line_operator(line(:volume; bcs=walls))
    @test Mw == Mv
    cs = vec(sum(Mw; dims=1)) .- 1
    @test 0.01 < abs(cs[2]) < 0.08                  # measured -4.0e-2
    @test 1e-4 < abs(cs[12]) < 1e-2                 # measured -3.8e-4
    # The one-sided rows leave row 2 nearly conservative and move the peak
    # of the leak to rows 3–7 (2.9e-2 at row 5) without reducing its total.
    Mo, _ = line_operator(line(:none; bcs=walls, filt=compact_filter(0.45)))
    co = vec(sum(Mo; dims=1)) .- 1
    @test abs(co[2]) < 0.01                         # measured 2.6e-3
    @test 0.01 < maximum(abs, co[3:7]) < 0.08
    @test 0.5 < sum(abs, co) / sum(abs, cs) < 2

    # On a clustered wall line the closure leak dominates either way; the
    # weighting moves the interior defect by less than a tenth of it.
    st = (sine_cluster(0.0, 1.0, 0.5, 0.5), nothing, nothing)
    dn = defect(line(:none; bcs=walls, stretch=st))
    dv = defect(line(:volume; bcs=walls, stretch=st))
    @test maximum(abs, dn[9:N-8]) < 5e-3            # measured 1.7e-3
    @test maximum(abs, dv[9:N-8]) < 5e-3            # measured 1.6e-3
    @test abs(maximum(abs, dn[9:N-8]) - maximum(abs, dv[9:N-8])) < 5e-4

    # At the cylindrical axis the odd-parity fold the weighted form needs
    # (J = r) does not have unit column sums: its first-row defect is 0.149
    # against 8.6e-3 unweighted. At the spherical origin (J = r² even) both
    # conserve to round-off.
    axis = ((AxisBC(), SlipWallBC()), per3[2], per3[3])
    dn = defect(line(:none; bcs=axis, metric=CylindricalMetric()))
    dv = defect(line(:volume; bcs=axis, metric=CylindricalMetric()))
    @test abs(dn[1]) < 2e-2
    @test abs(dv[1]) > 0.1
    @test abs(dv[1]) > 10 * abs(dn[1])
    orig = ((OriginBC(), SlipWallBC()), per3[2], per3[3])
    sph = (; metric=SphericalMetric(), origin=(0.0, π / 2 - 0.5, 0.0))
    dn = defect(line(:none; bcs=orig, sph...))
    dv = defect(line(:volume; bcs=orig, sph...))
    @test maximum(abs, dn[1:4]) < 1e-9                # measured 6.5e-11
    @test maximum(abs, dv[1:4]) < 1e-9                # measured 2e-14

    # Constants: the unweighted operator never reads the volume, and the
    # weighted one divides F(J) by itself, so both hold a uniform state on
    # every metric, folds and stretching included.
    function interior_change(s, Q, Q0)
        nx, ny, nz = s.decomp.n_local
        maximum(abs(Q[padded_index(s, i, j, k), c] - Q0[padded_index(s, i, j, k), c])
                for c in 1:s.equations.n_cons, i in 1:nx, j in 1:ny, k in 1:nz)
    end
    metrics = [
        (; n_global=(24, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=walls, stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing, nothing)),
        (; n_global=(32, 1, 12), L_domain=(1.0, 1.0, 0.5), metric=CylindricalMetric(),
           bcs=axis),
        (; n_global=(24, 16, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=axis),
        (; n_global=(24, 12, 12), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3])),
    ]
    for cs in metrics, wt in (:none, :volume)
        s = Solver(; cs..., art=ArtificialProperties(enabled=false), filter_weighting=wt)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(s, Q)
        Q0 = copy(Q)
        filter_state!(s, Q)
        @test interior_change(s, Q, Q0) < 1e-12
    end

    # Uniform Cartesian in three dimensions: the weighting is skipped, so a
    # pass on a non-uniform state is the unweighted pass bit for bit.
    ic = (x, y, z) -> Prim(u=(0.3sin(x)cos(y), -0.3cos(x)sin(y), 0.1sin(z)),
                            p=1 + 0.1cos(x)cos(z), rho=1 + 0.2sin(y))
    states = map((:none, :volume)) do wt
        s = Solver(n_global=(16, 16, 16), L_domain=(2π, 2π, 2π), bcs=per3,
                   art=ArtificialProperties(enabled=false), filter_weighting=wt)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        filter_state!(s, Q)
        Q
    end
    @test states[1] == states[2]
    @test_throws ErrorException Solver(n_global=(16, 1, 1), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, filter_weighting=:mass)
end

# --- AMR level-transfer operators (reference/AMR_GPU.md) -------------------

"Fill the interior of a padded field with fn(x), x = (i-1)h along `dim`."
function transfer_fill!(f, decomp, dim, fn, h)
    pad = decomp.n_halo_d
    for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        f[i+pad[1], j+pad[2], k+pad[3]] = fn(((i, j, k)[dim] - 1) * h)
    end
    return f
end

"Max interior difference of two padded fields on the same decomposition."
function transfer_dev(f, g, decomp)
    pad = decomp.n_halo_d
    e = 0.0
    for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        e = max(e, abs(f[i+pad[1], j+pad[2], k+pad[3]] - g[i+pad[1], j+pad[2], k+pad[3]]))
    end
    return e
end

@testset "AMR transfer pair: DC gain, closures, invertibility" begin
    # Conservation is by unit zero-wavenumber gain: the two sides of the
    # transfer relation agree on it to the last bit, row 1 of the closure set
    # is exact, and row 2 is within one ULP of the interior gain.
    @test CL.AMR_A + 2 * CL.AMR_B + 2 * CL.AMR_C == 1 + 2 * CL.AMR_ALPHA
    @test sum(CL.AMR_EDGE_ROW1) == 1.0
    @test abs(sum(CL.AMR_EDGE_ROW2) - (1 + 2 * CL.AMR_ALPHA)) <= eps()

    restriction, prolongation = amr_transfer_schemes()
    @test restriction isa CompactScheme{Float64}
    @test prolongation isa BandedCompactScheme{Float64}

    # prolong(restrict(f)) == f with no sampling, through the distributed
    # solvers and the one-sided closure rows (measured 1.6e-15 closed,
    # 2.7e-15 periodic at n = 96).
    for periodic in (false, true)
        d = Decomp((96, 4, 4), (periodic, true, true); dims=(1, 1, 1))
        rplan = CL.plan_direction(d, restriction, 1, 1)
        pplan = CL.plan_direction(d, prolongation, 1, 1)
        f = CL.field(d); fbar = CL.field(d); back = CL.field(d)
        transfer_fill!(f, d, 1, x -> sin(5x) + 0.4cos(11x) + 0.1sin(29x), 2π / 96)
        CL.exchange_dim!(f, d, 1)
        apply_along!(fbar, rplan, f, d)
        CL.exchange_dim!(fbar, d, 1)
        apply_along!(back, pplan, fbar, d)
        @test transfer_dev(f, back, d) < 1e-14
    end

    # Constants through the full transfer, closures included: restriction
    # reproduces a constant to the last bit (measured 0.0), prolongation to a
    # few ULPs (measured 13 eps).
    nc = 24; nf = 3nc - 2
    df = Decomp((nf, 4, 4), (false, true, true); dims=(1, 1, 1))
    dc = Decomp((nc, 4, 4), (false, true, true); dims=(1, 1, 1))
    tp = plan_transfer(df, dc, 1)
    fine = CL.field(df); coarse = CL.field(dc)
    unit_f = CL.field(df); unit_c = CL.field(dc)
    transfer_fill!(unit_f, df, 1, x -> 1.0, 1.0)
    transfer_fill!(unit_c, dc, 1, x -> 1.0, 1.0)
    restrict!(coarse, tp, unit_f)
    @test transfer_dev(coarse, unit_c, dc) <= eps()
    prolong!(fine, tp, unit_c)
    @test transfer_dev(fine, unit_f, df) <= 32 * eps()

    # Each Lagrange weight row is computed in exact rational arithmetic and
    # sums to one within rounding of the converted weights.
    for order in (4, 6, 8)
        W = CL.amr_interpolation_weights(Float64, order)
        for r in 1:order-1, sub in 1:2
            @test abs(sum(W[:, sub, r]) - 1) <= 4 * eps()
        end
    end

    # Setup-time validation.
    @test_throws ErrorException plan_transfer(df, df, 1)          # wrong extents
    @test_throws ErrorException plan_transfer(df, dc, 1; interp_order=5)
    d1 = Decomp((nf, 1, 1), (false, true, true); dims=(1, 1, 1))
    @test_throws ErrorException plan_transfer(d1, d1, 2)          # collapsed dim
end

@testset "AMR transfer: 3:1 sampling convention" begin
    # The convention (reference/AMR_GPU.md risk 1, resolved by measurement):
    # restriction filters the fine line and takes the coincident nodes;
    # prolongation interpolates the coarse field onto the intermediate fine
    # nodes and deconvolves. Restriction is then a LEFT INVERSE of
    # prolongation, so coarse → fine → coarse is exact for arbitrary data —
    # the property that keeps levels consistent without refluxing.
    for (periodic, dim) in ((true, 1), (false, 2))
        nc = 32
        nf = periodic ? 3nc : 3nc - 2
        shape(n) = ntuple(d -> d == dim ? n : 4, 3)
        pers = ntuple(d -> d == dim ? periodic : true, 3)
        df = Decomp(shape(nf), pers; dims=(1, 1, 1))
        dc = Decomp(shape(nc), pers; dims=(1, 1, 1))
        tp = plan_transfer(df, dc, dim)
        hc = periodic ? 2π / nc : 1.0 / (nc - 1)
        coarse = CL.field(dc); coarse2 = CL.field(dc); fine = CL.field(df)
        transfer_fill!(coarse, dc, dim, x -> sin(5x) + exp(sin(3x)) + x, hc)
        prolong!(fine, tp, coarse)
        restrict!(coarse2, tp, fine)
        @test transfer_dev(coarse, coarse2, dc) < 1e-14
    end

    # Fine → coarse → fine loses subsampled content and converges at the
    # interpolation order: measured 5.93 between n_coarse = 32 and 64 at the
    # default order 6 (3.97 at order 4, 7.97 at order 8; bench/amr_transfer.jl).
    errs = Float64[]
    for nc in (32, 64)
        nf = 3nc
        df = Decomp((nf, 4, 4), (true, true, true); dims=(1, 1, 1))
        dc = Decomp((nc, 4, 4), (true, true, true); dims=(1, 1, 1))
        tp = plan_transfer(df, dc, 1)
        fine = CL.field(df); fine2 = CL.field(df); coarse = CL.field(dc)
        transfer_fill!(fine, df, 1, x -> sin(2x) + cos(3x), 2π / nf)
        restrict!(coarse, tp, fine)
        prolong!(fine2, tp, coarse)
        push!(errs, transfer_dev(fine, fine2, df))
    end
    @test errs[2] < 2e-5
    @test log2(errs[1] / errs[2]) > 5.5
end

@testset "AMR transfer through a fold: symmetric closure variants" begin
    # Pyranda tabulates ±1 symmetric closure variants via ghost folding —
    # the same algebra `plan_direction`'s lo_fold performs. Pin the mapping:
    # a half line under a parity fold with mirror-filled halos must reproduce
    # the closed full line on parity-extended data, row for row.
    m = 12
    n = 2m
    dfull = Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1))
    dhalf = Decomp((m, 4, 4), (false, true, true); dims=(1, 1, 1))
    data = [exp(sin(0.4j)) + 0.05j for j in 1:m]
    for scheme in amr_transfer_schemes(), σ in (1, -1)
        plan_full = CL.plan_direction(dfull, scheme, 1, 1)
        plan_half = CL.plan_direction(dhalf, scheme, 1, 1; lo_fold=σ)
        pad = dfull.n_halo_d
        ffull = CL.field(dfull); ofull = CL.field(dfull)
        fhalf = CL.field(dhalf); ohalf = CL.field(dhalf)
        for k in 1:4, j in 1:4, i in 1:m
            ffull[pad[1]+m+i, pad[2]+j, pad[3]+k] = data[i]
            ffull[pad[1]+m+1-i, pad[2]+j, pad[3]+k] = σ * data[i]
            fhalf[pad[1]+i, pad[2]+j, pad[3]+k] = data[i]
        end
        for k in 1:4, j in 1:4, i in 1:pad[1]     # half-offset mirror halo
            fhalf[pad[1]+1-i, pad[2]+j, pad[3]+k] = σ * data[i]
        end
        apply_along!(ofull, plan_full, ffull, dfull)
        apply_along!(ohalf, plan_half, fhalf, dhalf)
        e = maximum(abs(ohalf[pad[1]+i, pad[2]+j, pad[3]+k] -
                        ofull[pad[1]+m+i, pad[2]+j, pad[3]+k])
                    for k in 1:4, j in 1:4, i in 1:m)
        @test e < 1e-13
    end
end

@testset "AMR transfer: sensor injection near a transfer end" begin
    # The deconvolution amplifies fine-Nyquist content by ≈ 20
    # (bench/amr_transfer.jl), and the risk named in the plan is that content
    # injected by the Cook sensor chain near a transfer boundary rides through
    # prolongation. Measured: the smoothed δ⁴ sensor of a 2h shock profile
    # round-trips with amplification ≤ 1.13 at any distance from the end, the
    # state field undershoots by ≤ 3% of ambient, and the round-trip error
    # decays ≈ 3× per point once outside the shock footprint (≤ 1.6e-4 by 12
    # cells). The bounds here carry factor 2–6 headroom.
    nc = 33
    nf = 3nc - 2
    solver = Solver(; n_global=(nf, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=((ExtrapolationBC(), ExtrapolationBC()), per3[2], per3[3]))
    df = solver.decomp
    dc = Decomp((nc, 1, 1), (false, true, true); dims=(1, 1, 1))
    tp = plan_transfer(df, dc, 1)
    hf = 1.0 / (nf - 1)
    pad = df.n_halo_d
    rho = CL.field(df); rho2 = CL.field(df); coarse = CL.field(dc)
    sensor2 = CL.field(df)
    for dist in (6, 12)
        xs = 1.0 - dist * hf
        transfer_fill!(rho, df, 1, x -> 1.0 + 0.5 * (1 + tanh((x - xs) / (2hf))), hf)
        restrict!(coarse, tp, rho)
        prolong!(rho2, tp, coarse)
        @test minimum(rho2[pad[1]+1:pad[1]+nf, pad[2]+1, pad[3]+1]) > 1.0 - 0.06
        CL.exchange_halos!(rho, df)
        CL.delta4_sum!(solver.sensor, rho, solver, 1)
        CL.smooth!(solver.sensor, solver)
        smax = maximum(abs, solver.sensor)
        restrict!(coarse, tp, solver.sensor)
        prolong!(sensor2, tp, coarse)
        s2max = maximum(abs, sensor2[pad[1]+1:pad[1]+nf, pad[2]+1, pad[3]+1])
        @test s2max <= 2 * smax
    end
    # Localization: with the shock at mid-domain, the round-trip error 12 or
    # more cells away is below 1e-3 of the O(6e-2) error at the shock.
    transfer_fill!(rho, df, 1, x -> 1.0 + 0.5 * (1 + tanh((x - 0.5) / (2hf))), hf)
    restrict!(coarse, tp, rho)
    prolong!(rho2, tp, coarse)
    ic = (nf + 1) ÷ 2
    efar = maximum(abs(rho[pad[1]+i, pad[2]+1, pad[3]+1] -
                       rho2[pad[1]+i, pad[2]+1, pad[3]+1])
                   for i in 1:nf if abs(i - ic) >= 12)
    @test efar < 1e-3
end

@testset "axisymmetric axis fold: manufactured smooth solution" begin
    # u_r = r·g(r) is an odd smooth function; d/dr through the fold must
    # match analytics at the first half-offset nodes — the sharpest probe of
    # the folded row and mirror fill.
    solver = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 0.5),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtificialProperties(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * exp(-4r^2))            # odd across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, -1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> (1 - 8r^2) * exp(-4r^2)) < 5e-6
    fillf!(solver, f, (r, θ, z) -> exp(-4r^2))                # even across the axis
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    # Even fold: third order with a larger constant. The maximum sits at the
    # outer slip wall: 2.14e-5 under the `:neutral3` rows, 1.4e-5 under
    # `:cascade3`.
    @test ferr(solver, df, (r, θ, z) -> -8r * exp(-4r^2)) < 3e-5
end

@testset "resolved-θ axis: antipodal pairing (local)" begin
    # f = r cosθ · e^{−4r²} = x·g is a scalar that is globally smooth through
    # the axis; its antipodal image is −f, so the odd combination carries all
    # of it. It is not a velocity component (a u_r of that form has no smooth
    # Cartesian preimage), so σ = +1.
    solver = Solver(n_global=(48, 16, 1), L_domain=(1.0, 2π, 1.0),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtificialProperties(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, z) -> r * cos(θ) * exp(-4r^2))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    # The maximum sits at the outer slip wall: 1.06e-5 under the `:neutral3`
    # rows, below 1e-5 under `:cascade3`.
    @test ferr(solver, df, (r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2)) < 1.5e-5
    # A scalar even case: f = e^{−4r²}·(1 + ½cos 2θ) maps to itself at θ+π.
    fillf!(solver, f, (r, θ, z) -> exp(-4r^2) * (1 + 0.5cos(2θ)))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, z) -> -8r * exp(-4r^2) * (1 + 0.5cos(2θ))) < 1e-4  # 3rd-order, larger const
end

@testset "azimuthal mode truncation: table, projection, rate cap" begin
    # N_θ = 32 over 24 radial nodes at κ = 1: mode_limit(r_i) =
    # max(2, ⌊π(i − ½)⌋) is 2, 4, 7, 10, 14 on the first five rings, and
    # ring 6 (17 ≥ N_θ/2) is the first to keep every mode. The radial and
    # azimuthal momenta keep one mode more, and the θ rate is capped at it.
    mk(κ; kw...) = Solver(; n_global=(24, 32, 1), L_domain=(1.0, 2π, 1.0),
                          metric=CylindricalMetric(),
                          bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                          art=ArtificialProperties(enabled=false),
                          polar_truncation=κ, kw...)
    solver = mk(1.0)
    tr = solver.truncation
    o1 = solver.decomp.n_halo_d[1]
    @test tr.mode_limit == [2, 4, 7, 10, 14]
    @test tr.rings == (o1 + 1):(o1 + 5)
    @test tr.vector == solver.equations.i_mom[1]:solver.equations.i_mom[2]
    @test tr.theta_cap ≈ [(m + 1) / (π * xcoord(solver, 1, i))
                          for (i, m) in enumerate(tr.mode_limit)] rtol = 1e-14
    @test isempty(mk(0.0).truncation.rings)
    @test mk(2.0).truncation.mode_limit == [2, 2, 3, 5, 7, 8, 10, 11, 13, 14]
    @test_throws ArgumentError mk(0.5)
    @test_throws ErrorException mk(1.0; n_global=(24, 1, 1), L_domain=(1.0, 1.0, 1.0))
    @test_throws ErrorException Solver(n_global=(24, 32, 1),
                                       L_domain=(1.0, π, 1.0), bcs=per3,
                                       polar_truncation=1.0)
    @test_throws ErrorException Solver(n_global=(24, 32, 1), L_domain=(1.0, π, 2π),
                                       metric=SphericalMetric(),
                                       bcs=((OriginBC(), SlipWallBC()),
                                            (PoleBC(), PoleBC()), per3[3]),
                                       polar_truncation=1.0)

    nθ = 32
    ring(Q, i, c) = [Q[padded_index(solver, i, j, 1), c] for j in 1:nθ]
    # Amplitude of mode m on ring i of component c.
    amp(Q, i, c, m) = (v = ring(Q, i, c); θ = [xcoord(solver, 2, j) for j in 1:nθ];
                       hypot(sum(v .* cos.(m .* θ)), sum(v .* sin.(m .* θ))) * 2 / nθ)
    maxdiff(A, B) = maximum(abs, parent(A) .- parent(B))
    im = solver.equations.i_mom
    U = 0.3
    stream(r, θ, z) = Prim(u=(U * cos(θ), -U * sin(θ), 0.0), p=1.0, rho=1.0)

    # A uniform freestream is m = 1 in the physical velocity components.
    Q = allocate_state(solver)
    initialize!(solver, Q, stream)
    Q0 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test maxdiff(Q, Q0) < 1e-14

    # A quadratic field is Cartesian mode 2 at O(r²): mode 2 of a density
    # and, through the rotation to u_r and u_θ, mode 3 of the physical
    # momenta. Ring 1 (limit 2) keeps both. The density varies in one field
    # and the velocity in the other, so that neither momentum carries their
    # product; the energy, quartic in r through the kinetic energy, is not
    # compared.
    r1 = xcoord(solver, 1, 1)
    for (a, b) in ((0.3, 0.0), (0.0, 1.0))
        function quadratic(r, θ, z)
            s, c = sincos(θ)
            x, y = r * c, r * s
            ux = U + b * (0.2x + 0.1y + 0.5(x^2 - y^2))
            uy = b * (0.3x - 0.2y + 0.4x * y)
            return Prim(u=(c * ux + s * uy, -s * ux + c * uy, 0.0), p=1.0,
                        rho=1.0 + a * x * y)
        end
        initialize!(solver, Q, quadratic)
        Q0 = deepcopy(Q)
        @test amp(Q0, 1, a > 0 ? 1 : im[1], a > 0 ? 2 : 3) > 0.1r1^2
        CL.truncate_modes!(solver, Q)
        @test all(c -> maximum(abs, ring(Q, 1, c) .- ring(Q0, 1, c)) < 1e-14,
                  (1, im...))
    end

    # Rigid rotation over a radially varying state is m = 0 on every ring.
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(0.0, 0.4r, 0.0), p=1 + r^2,
                                             rho=1 + 0.5r^2))
    Q0 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test maxdiff(Q, Q0) < 1e-14

    # m = 6 seeded on ring 2 (limit 4) is removed and m = 3 there is kept; the
    # same m = 6 on ring 10, outside the table, is untouched bit for bit.
    seeded(r, θ, z) = Prim(u=(U * cos(θ), -U * sin(θ), 0.0), p=1.0,
                           rho=1 + (abs(r - xcoord(solver, 1, 2)) < 1e-12 ?
                                    0.1cos(6θ) + 0.05sin(3θ) : 0.0) +
                               (abs(r - xcoord(solver, 1, 10)) < 1e-12 ?
                                    0.1cos(6θ) : 0.0))
    initialize!(solver, Q, seeded)
    Q0 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test amp(Q0, 2, 1, 6) ≈ 0.1
    @test amp(Q, 2, 1, 6) < 1e-14
    @test amp(Q, 2, 1, 3) ≈ 0.05 rtol = 1e-13
    @test ring(Q, 10, 1) == ring(Q0, 10, 1)
    @test ring(Q, 10, 5) == ring(Q0, 10, 5)
    # Idempotent: a second projection changes nothing beyond round-off.
    Q1 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test maxdiff(Q, Q1) < 1e-14

    # Ring sums of mass, energy and Cartesian momentum, over a state with
    # content in every mode on every ring.
    rng = MersenneTwister(11)
    for k in axes(parent(Q), 4), I in CartesianIndices(solver.rho)
        parent(Q)[I, k] = (k == 1 ? 1.0 : k == 5 ? 2.5 : 0.0) + 0.1 * randn(rng)
    end
    Q0 = deepcopy(Q)
    CL.truncate_modes!(solver, Q)
    @test maxdiff(Q, Q0) > 1e-2
    for i in 1:6
        θ = [xcoord(solver, 2, j) for j in 1:nθ]
        for S in (Q -> sum(ring(Q, i, 1)), Q -> sum(ring(Q, i, 5)),
                  Q -> sum(ring(Q, i, im[1]) .* cos.(θ) .- ring(Q, i, im[2]) .* sin.(θ)),
                  Q -> sum(ring(Q, i, im[1]) .* sin.(θ) .+ ring(Q, i, im[2]) .* cos.(θ)))
            @test S(Q) ≈ S(Q0) atol = 1e-13
        end
    end

    # The rate cap: the same state steps longer under the table, dt_report
    # agrees with compute_dt, and at κ = 0 the loop is the untruncated one.
    off = mk(0.0)
    Qoff = allocate_state(off)
    initialize!(off, Qoff, stream)
    initialize!(solver, Q, stream)
    @test compute_dt(solver, Q) > 4 * compute_dt(off, Qoff)
    @test dt_report(solver, Q).dt ≈ compute_dt(solver, Q) rtol = 1e-12
    @test dt_report(off, Qoff).dt ≈ compute_dt(off, Qoff) rtol = 1e-12

    # run! applies the projection after every step: on ring 1 (limit 2) a
    # seeded pressure in modes 3 and 4 leaves no mode above 2 in the scalars
    # and none above 3 in the radial and azimuthal momenta after one step,
    # while the momenta's mode 3, which the pressure gradient drives, stays.
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(U * cos(θ), -U * sin(θ), 0.0),
                                             p=1.0 + 0.01(cos(3θ) + sin(4θ)) *
                                                     exp(-20r^2),
                                             rho=1.0))
    run!(solver, Q; tfinal=1.0, nmax=1)
    scalars = setdiff(1:solver.equations.n_cons, tr.vector)
    @test all(c -> amp(Q, 1, c, 3) < 1e-14 && amp(Q, 1, c, 4) < 1e-14, scalars)
    @test all(c -> amp(Q, 1, c, 4) < 1e-14, tr.vector)
    @test all(c -> amp(Q, 1, c, 3) > 1e-6, tr.vector)
end

@testset "spherical poles + origin: derivative of a smooth 3-D Gaussian" begin
    # f = e^{−4r²} is smooth at origin and poles; ∂f/∂r and (1/r)∂f/∂θ = 0.
    solver = Solver(n_global=(40, 16, 12), L_domain=(1.0, π, 2π),
               metric=SphericalMetric(),
               bcs=((OriginBC(), SlipWallBC()),
                    (PoleBC(), PoleBC()), per3[3]),
               art=ArtificialProperties(enabled=false))
    f = CL.field(solver.decomp); df = CL.field(solver.decomp)
    fillf!(solver, f, (r, θ, φ) -> exp(-4r^2))
    CL.exchange_halos!(f, solver.decomp)
    CL.deriv_along!(df, f, solver, 1, 1); CL._scale_grad!(df, solver, 1)
    @test ferr(solver, df, (r, θ, φ) -> -8r * exp(-4r^2)) < 1e-4  # 3rd-order, larger const
    CL.deriv_along!(df, f, solver, 2, 1); CL._scale_grad!(df, solver, 2)
    @test ferr(solver, df, (r, θ, φ) -> 0.0) < 1e-8
end

@testset "symmetry plane: construction, spacing and the self-paired fold" begin
    # The plane sits half a cell outside the end it is applied to, so each
    # folded end takes half a cell off the line: one fold gives L/(N − ½) and
    # two give L/N, with node 1 at h/2 wherever the low end is folded.
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    wall = SlipWallBC()
    cart(bcs1) = Solver(n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                        bcs=(bcs1, per3[2], per3[3]),
                        art=ArtificialProperties(enabled=false))
    both = cart(sym)
    @test both.h[1] ≈ 1 / 16
    @test both.coord_shift[1] ≈ both.h[1] / 2
    @test xcoord(both, 1, 1) ≈ 1 / 32
    @test xcoord(both, 1, 16) ≈ 31 / 32
    lo = cart((SymmetryPlaneBC(), wall))
    @test lo.h[1] ≈ 1 / 15.5
    @test xcoord(lo, 1, 1) ≈ lo.h[1] / 2
    @test xcoord(lo, 1, 16) ≈ 1.0
    hi = cart((wall, SymmetryPlaneBC()))
    @test hi.h[1] ≈ 1 / 15.5
    @test hi.coord_shift[1] == 0.0
    @test xcoord(hi, 1, 1) ≈ 0.0
    @test xcoord(hi, 1, 16) ≈ 1 - hi.h[1] / 2
    @test (hi.folds[1].lo, hi.folds[1].hi) == (false, true)
    # Self-paired: each line continues into itself, so there is no pairing
    # dimension and no partner block to buffer.
    @test both.folds[1].pair === nothing
    @test both.folds[1].sigvel == (-1, 1, 1)
    @test isempty(both.pairbuf) && isempty(both.pairout)
    # Only the normal velocity is odd, so the mass, energy and tangential
    # momentum fluxes are odd and the normal momentum flux is even.
    eq = both.equations
    @test both.folds[1].sigflux[1] == -1
    @test both.folds[1].sigflux[eq.i_energy] == -1
    @test both.folds[1].sigflux[eq.i_mom[1]] == 1
    @test both.folds[1].sigflux[eq.i_mom[2]] == -1
    # The sensor operators' node-centred wall rows must not be planned here:
    # the fold takes its own half-offset mirror inside the same routines.
    @test sensor_mirror(SymmetryPlaneBC()) === false
    @test CL._sensor_wall_faces(((SymmetryPlaneBC(), SlipWallBC()),
                                 (PeriodicBC(), PeriodicBC()),
                                 (PeriodicBC(), PeriodicBC()))) ==
          ((false, true), (false, false), (false, false))
    @test isperiodic(SymmetryPlaneBC()) === false
    @test CL._is_fold_bc(SymmetryPlaneBC())
    # A plane on dimension 3 beside a cylindrical axis on dimension 1: the
    # pairing is per dimension, and the axis' resolved-θ fold still buffers.
    cyl = Solver(n_global=(16, 16, 12), L_domain=(1.0, 2π, 0.5),
                 metric=CylindricalMetric(), art=ArtificialProperties(enabled=false),
                 bcs=((AxisBC(), SlipWallBC()), per3[2], sym))
    @test cyl.h[3] ≈ 0.5 / 12
    @test cyl.coord_shift[3] ≈ cyl.h[3] / 2
    @test cyl.folds[3].pair === nothing
    @test cyl.folds[3].sigvel == (1, 1, -1)
    @test cyl.folds[1] !== nothing && cyl.folds[1].pair !== nothing
    @test !isempty(cyl.pairbuf)
end

@testset "symmetry plane: a uniform state is preserved" begin
    # No node sits on a plane and nothing is enforced there, so a uniform
    # state has to come out of the folded operators unchanged on its own: the
    # mirror halo is the interior value and every flux divergence cancels.
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.5, 1.3)])
    solver = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0), eos=eos,
                    bcs=(sym, sym, sym), art=ArtificialProperties(enabled=true))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(Y=(0.3, 0.7), rho=1.2, p=0.8, u=(0.0, 0.0, 0.0)))
    dQ = zero(Q)
    apply_bcs!(solver, Q)
    compute_rhs!(solver, Q, dQ)
    @test maximum(abs, dQ) < 1e-14
    Q0 = copy(Q)
    run!(solver, Q; tfinal=1e9, nmax=5)
    spread, drift = 0.0, 0.0
    for c in 1:solver.equations.n_cons
        v0 = Q0[padded_index(solver, 1, 1, 1), c]
        for k in 1:12, j in 1:12, i in 1:12
            v = Q[padded_index(solver, i, j, k), c]
            spread = max(spread, abs(v - Q[padded_index(solver, 1, 1, 1), c]))
            drift = max(drift, abs(v - v0))
        end
    end
    @test spread < 1e-12
    @test drift < 1e-12
end

@testset "symmetry plane: the mirror of the periodic run on the doubled line" begin
    # Data carrying the slip-wall parity about both ends (rho, p and the mass
    # fractions even, the normal velocity odd, the tangential velocity even)
    # make the run between symmetry planes on [0, 1] at N nodes the
    # restriction of the periodic run on [0, 2) at 2N nodes, node for node.
    # The two agree to round-off because the folded operator is the periodic
    # operator restricted by parity: no closure row enters either, so there
    # is no closure defect to separate them. The node-centred wall is the
    # contrast — its defect is what test/convergence.jl measures.
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    two = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.5, 1.3)])
    a, b, N, nsteps = 0.05, 0.05, 32, 30
    function mirror_error(; deriv=lele_d1_6(), art=ArtificialProperties(enabled=true),
                          mu=0.0, dim=1, eos=nothing, n=N)
        h = 1 / n
        nt = 12                                    # transverse extent, 2-D form
        tang = dim == 2
        ic = (x, y, z) -> begin
            s = dim == 1 ? x : y
            rho = 1 + a * cos(pi * s)
            un = b * sin(pi * s)
            ut = tang ? 0.07 * cos(pi * s) * sin(2pi * x) : 0.0
            u = dim == 1 ? (un, ut, 0.0) : (ut, un, 0.0)
            eos === nothing ? Prim(rho=rho, u=u, p=rho^1.4) :
                Prim(Y=(0.5 + 0.2cos(pi * s), 0.5 - 0.2cos(pi * s)),
                     rho=rho, u=u, p=rho^1.4)
        end
        common = (; deriv=deriv, art=art, filter_interval=1, filter_cfl=0.0,
                  cfl=0.4, transport=ConstantTransport(mu0=mu, Pr=0.7))
        eos === nothing || (common = merge(common, (; eos=eos)))
        function advance(ng, L, bcs, origin)
            s = Solver(; n_global=ng, L_domain=L, bcs=bcs, origin=origin,
                       common...)
            Q = allocate_state(s)
            initialize!(s, Q, ic)
            run!(s, Q; tfinal=1e9, nmax=nsteps)
            (s, Q)
        end
        sf, Qf = dim == 1 ?
            advance((n, 1, 1), (1.0, 1.0, 1.0), (sym, per3[2], per3[3]),
                    (0.0, 0.0, 0.0)) :
            advance((nt, n, 1), (1.0, 1.0, 1.0), (per3[1], sym, per3[3]),
                    (0.0, 0.0, 0.0))
        # The doubled periodic line samples the same nodes: its origin is
        # shifted half a cell so that node j sits at (j − ½)h, as the folded
        # grid's node i does.
        sp, Qp = dim == 1 ?
            advance((2n, 1, 1), (2.0, 1.0, 1.0), per3, (h / 2, 0.0, 0.0)) :
            advance((nt, 2n, 1), (1.0, 2.0, 1.0), per3, (0.0, h / 2, 0.0))
        err, scale = 0.0, 0.0
        ni, nj = dim == 1 ? (n, 1) : (nt, n)
        for c in 1:sf.equations.n_cons, j in 1:nj, i in 1:ni
            vf = Qf[padded_index(sf, i, j, 1), c]
            vp = Qp[padded_index(sp, i, j, 1), c]
            err = max(err, abs(vf - vp))
            scale = max(scale, abs(vp))
        end
        return err / scale
    end
    # Measured on the workstation: 2.4e-15 to 3.6e-15 over the twelve
    # one-dimensional combinations below, 3.1e-15 viscous, and 4.3e-15 to
    # 5.5e-15 for the two-dimensional plane on dimension 2.
    for deriv in (lele_d1_6(), lele_d1_8(), lele_d1_10()),
        detector in (:delta4, :d8), smoother in (:gaussian, :compact)
        art = ArtificialProperties(enabled=true, detector=detector, smoother=smoother)
        @test mirror_error(deriv=deriv, art=art) < 2e-14
    end
    # Two species, the mass fractions even about both planes.
    @test mirror_error(eos=two) < 2e-14
    # Physical viscosity: the tangential shear traction and the normal heat
    # flux at the plane come from the fold's parities and nothing else.
    @test mirror_error(mu=0.005) < 2e-14
    # The plane on dimension 2, with the transverse direction periodic and a
    # tangential velocity, inviscid and viscous.
    @test mirror_error(dim=2, n=24) < 2e-14
    @test mirror_error(dim=2, n=24, mu=0.005) < 2e-14
end

@testset "symmetry plane: setup rejections" begin
    sym = (SymmetryPlaneBC(), SymmetryPlaneBC())
    wall = (SlipWallBC(), SlipWallBC())
    off = ArtificialProperties(enabled=false)
    cart(; kw...) = Solver(n_global=(48, 12, 12), L_domain=(1.0, 1.0, 1.0),
                           bcs=(sym, per3[2], per3[3]), art=off; kw...)
    @test cart() isa Solver
    # A stretched dimension has no mirror: the fold reflects the computational
    # coordinate and the map does not commute with it.
    @test_throws ErrorException cart(
        stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing, nothing))
    # The scale factors of the plane's own dimension must not depend on its
    # coordinate, which excludes the cylindrical radius and every spherical
    # dimension (validate_bc). The same rule makes a shared dimension with a
    # coordinate fold unreachable, since every dimension carrying one fails it.
    @test_throws ErrorException Solver(n_global=(16, 16, 12),
        L_domain=(1.0, π, 2π), metric=SphericalMetric(), art=off,
        bcs=(sym, (PoleBC(), PoleBC()), per3[3]))
    @test_throws ErrorException Solver(n_global=(16, 16, 12),
        L_domain=(1.0, 2π, 0.5), metric=CylindricalMetric(), art=off,
        origin=(0.2, 0.0, 0.0), bcs=(sym, per3[2], per3[3]))
    @test_throws ErrorException Solver(n_global=(16, 16, 12),
        L_domain=(1.0, 2π, 0.5), metric=CylindricalMetric(), art=off,
        bcs=((AxisBC(), SymmetryPlaneBC()), per3[2], per3[3]))
    # A patched run takes SlipWallBC at that face instead. A refined run
    # keeps the plane, whether its level stays off it or reaches it (the
    # level tests cover the second).
    @test_throws ErrorException cart(patch_grid=(2, 1, 1))
    @test length(Solver(n_global=(48, 1, 1),
        L_domain=(2π, 1.0, 1.0), bcs=(sym, per3[2], per3[3]), art=off,
        filter_interval=0, refine=BlockRegion((20, 0, 0), (8, 1, 1))).patches) == 2
end

@testset "rigid rotation in cylindrical: zero strain" begin
    # u_θ = Ω r ⇒ S_ij = 0 identically; probes the curvature corrections.
    solver = Solver(n_global=(32, 16, 12), L_domain=(1.0, 2π, 0.5),
               metric=CylindricalMetric(), origin=(0.2, 0.0, 0.0),
               bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtificialProperties(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(0.0, 0.3r, 0.0), p=1.0, rho=1.0))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    smax = 0.0
    for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        I = padded_index(solver, i, j, k)
        for b in 1:3, a in 1:3
            smax = max(smax, abs(0.5 * (solver.grad_u[a, b][I] + solver.grad_u[b, a][I])))
        end
    end
    @test smax < 1e-8
end

@testset "freestream preservation (uniform state ⇒ dQ ≈ 0)" begin
    # Uniform ρ, p, u = 0 must give zero RHS in every metric, with stretch,
    # and with folds: divergence/source/geometry consistency in one number.
    cases = [
        (; n_global=(24, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=per3, kw=(;)),
        (; n_global=(24, 12, 12), L_domain=(1.0, 1.0, 1.0), metric=CartesianMetric(),
           bcs=per3, kw=(; stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4),
                                    nothing, nothing),
                         bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]))),
        (; n_global=(24, 12, 12), L_domain=(1.0, 2π, 0.5), metric=CylindricalMetric(),
           bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
           kw=(; origin=(0.2, 0.0, 0.0))),
        (; n_global=(32, 1, 12), L_domain=(1.0, 1.0, 0.5), metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), kw=(;)),
        (; n_global=(24, 16, 1), L_domain=(1.0, 2π, 1.0), metric=CylindricalMetric(),
           bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), kw=(;)),
        (; n_global=(24, 12, 12), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
           bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
           kw=(;)),
    ]
    for cs in cases
        kw = merge((; n_global=cs.n_global, L_domain=cs.L_domain, metric=cs.metric,
                     bcs=cs.bcs, art=ArtificialProperties(enabled=false)), cs.kw)
        solver = Solver(; kw...)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        m = maximum(abs(dQ[padded_index(solver, i, j, k), c])
                    for c in 1:solver.equations.n_cons, i in 1:solver.decomp.n_local[1],
                        j in 1:solver.decomp.n_local[2], k in 1:solver.decomp.n_local[3])
        @test m < 1e-8
    end
end

@testset "EOS: conserved ↔ primitive round trip (two species)" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7, Y=(0.35, 0.65))
    initialize!(solver, Q, (x, y, z) -> pr)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    I = padded_index(solver, 3, 4, 5)
    @test solver.p[I] ≈ 0.8 atol = 1e-12
    @test solver.T_ion[I] ≈ 1.7 atol = 1e-12
    @test solver.u[I] ≈ 0.3 atol = 1e-12
    @test solver.Y[2][I] ≈ 0.65 atol = 1e-12
end

@testset "EquationSet owns the conserved layout" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    @test solver.equations isa NavierStokes1T
    @test solver.equations.n_species == 2
    @test solver.equations.n_cons == 6
    @test solver.equations.i_mom == (3, 4, 5)
    @test solver.equations.i_energy == 6
    @test solver.equations.component_names ==
          ["rho_a", "rho_b", "rho_u1", "rho_u2", "rho_u3", "rho_E"]
    @test fieldtype(typeof(solver), :eos) === typeof(solver.eos)
    @test fieldtype(typeof(solver), :metric) === typeof(solver.metric)
    # `folds` is per-patch state; the concreteness contract moved with it.
    @test fieldtype(eltype(fieldtype(typeof(solver), :patches)), :folds) ===
          typeof(solver.folds)
end

@testset "tuple source terms add momentum and energy work" begin
    force = ConstantBodyForce((1.0, 2.0, -3.0))
    solver = mkslv(n_global=(12, 12, 12), sources=(force,))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(u=(0.5, 0.25, -0.1), p=1.0, rho=2.0))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    I = padded_index(solver, 3, 4, 5)
    @test dQ[I, solver.equations.i_mom[1]] ≈ 2.0 atol = 1e-10
    @test dQ[I, solver.equations.i_mom[2]] ≈ 4.0 atol = 1e-10
    @test dQ[I, solver.equations.i_mom[3]] ≈ -6.0 atol = 1e-10
    @test dQ[I, solver.equations.i_energy] ≈ 2.6 atol = 1e-10
    @test typeof(solver.sources) === Tuple{typeof(force)}
end

@testset "Workspace reproduces explicit stage arrays" begin
    solver1 = mkslv(n_global=(12, 12, 12))
    solver2 = mkslv(n_global=(12, 12, 12))
    Q1 = allocate_state(solver1)
    Q2 = allocate_state(solver2)
    ic = (x, y, z) -> Prim(u=(0.1sin(x), -0.1cos(y), 0.05sin(z)),
                            p=1 + 0.02cos(x), rho=1 + 0.03sin(y))
    initialize!(solver1, Q1, ic)
    initialize!(solver2, Q2, ic)
    dQ = zero(Q1)
    du = zero(Q1)
    workspace = Workspace(Q2)
    step!(solver1, Q1, dQ, du, 1e-4)
    step!(solver2, Q2, workspace, 1e-4)
    @test Q1 == Q2
    @test dQ == workspace.dQ
    @test du == workspace.du
end

@testset "conservation: periodic RHS integrates to zero" begin
    solver = mkslv(n_global=(16, 16, 16), transport=ConstantTransport(mu0=1e-3))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) ->
        Prim(u=(0.1sin(x)cos(y), -0.1cos(x)sin(y), 0.05sin(z)),
             p=1 + 0.05cos(x)cos(z), rho=1 + 0.1sin(y)))
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    for c in 1:solver.equations.n_cons
        tot = sum(dQ[padded_index(solver, i, j, k), c] for i in 1:16, j in 1:16, k in 1:16)
        @test abs(tot) < 1e-8 * 16^3
    end
end

@testset "NSCBC outflow: matched uniform stream ⇒ no correction" begin
    solver = Solver(n_global=(32, 12, 12), L_domain=(1.0, 0.4, 0.4),
               bcs=((DirichletBC((x, y, z, t) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0)),
                     NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
               art=ArtificialProperties(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    apply_bcs!(solver, Q)
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    nx = solver.decomp.n_local[1]
    m = maximum(abs(dQ[padded_index(solver, nx, j, k), c])
                for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12)
    @test m < 1e-8
end

@testset "NSCBC outflow: a radial face relaxes at the curvature rate" begin
    # Gas at rest at the uniform pressure p∞ + δ: every term of the right-hand
    # side vanishes but the face's relaxation, ΔL = K δ, which enters the mass
    # as −ΔL/(2c²). On the outer radial face of r ∈ [1/2, 3/2], K is the
    # larger of σc/L and n_c c/(2r), n_c = 1 on a cylinder and 2 on a sphere;
    # on a Cartesian face it is σc/L.
    δ = 1e-3
    c = sqrt(1.4 * (1 + δ))
    for (metric, origin, nc) in ((CartesianMetric(), (0.5, 0.0, 0.0), 0),
                                 (CylindricalMetric(), (0.5, 0.0, 0.0), 1),
                                 (SphericalMetric(), (0.5, π / 2 - 0.5, 0.0), 2)),
        sigma in (0.0, 0.25, 4.0)
        solver = Solver(n_global=(24, 1, 1), L_domain=(1.0, 1.0, 1.0), origin=origin,
                        metric=metric,
                        bcs=((SlipWallBC(), NSCBCOutflowBC(pinf=1.0, sigma=sigma)),
                             per3[2], per3[3]),
                        art=ArtificialProperties(enabled=false))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(rho=1.0, p=1.0 + δ))
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        nx = solver.decomp.n_local[1]
        K = max(sigma * c, nc * c / 3)
        face = dQ[padded_index(solver, nx, 1, 1), 1]
        @test isapprox(face, -K * δ / (2c^2); rtol=1e-8, atol=1e-14)
        @test maximum(abs(dQ[padded_index(solver, i, 1, 1), 1]) for i in 2:nx-1) < 1e-12
    end
end

# The Oscillating sphere tutorial's dipole on its radial grid with half its
# polar nodes, at the default relaxation: the plane form of the outflow
# reflects it by 1/(2kR), and with the curvature term the reflection is the
# second-order remainder 1/(2(kR)²) and the discretization's share.
include("sphere_dipole.jl")

@testset "NSCBC outflow: a spherical wave leaves a radial face" begin
    m = sphere_dipole_reflection(2.1; n=128)
    plane = 1 / (2m.kR)
    @info "spherical outflow reflection" m.reflection plane m.amplitude
    @test m.reflection < plane / 10
    @test abs(m.amplitude - 1) < 0.03
end

@testset "NoSlipWallBC: adiabatic zeroes velocity, isothermal sets T_ion" begin
    for Twall in (NaN, 2.5)
        solver = Solver(n_global=(24, 12, 12), L_domain=(1.0, 0.4, 0.4),
                   bcs=((NoSlipWallBC(Twall=Twall), NoSlipWallBC(Twall=Twall)),
                        per3[2], per3[3]),
                   art=ArtificialProperties(enabled=false))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, -0.2, 0.1), p=1.0, T_ion=1.0))
        apply_bcs!(solver, Q)
        CL.exchange_state!(Q, solver.decomp)
        CL.primitives!(solver, Q)
        nx = solver.decomp.n_local[1]
        uw = maximum(abs(solver.u[padded_index(solver, i, j, k)]) + abs(solver.v[padded_index(solver, i, j, k)]) +
                     abs(solver.w[padded_index(solver, i, j, k)])
                     for i in (1, nx), j in 1:12, k in 1:12)   # both walls
        @test uw < 1e-12
        if !isnan(Twall)                       # isothermal wall holds Twall
            e = maximum(abs(solver.T_ion[padded_index(solver, i, j, k)] - Twall)
                        for i in (1, nx), j in 1:12, k in 1:12)
            @test e < 1e-12
        end
    end
end

@testset "ExtrapolationBC: copies the adjacent interior plane" begin
    solver = Solver(n_global=(24, 12, 12), L_domain=(1.0, 0.4, 0.4),
               bcs=((ExtrapolationBC(), ExtrapolationBC()), per3[2], per3[3]),
               art=ArtificialProperties(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2 + x, 0, 0), p=1 + 0.1x, rho=1 + 0.3x))
    apply_bcs!(solver, Q)
    nx = solver.decomp.n_local[1]
    d = 0.0
    for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12
        d = max(d, abs(Q[padded_index(solver, 1, j, k), c]  - Q[padded_index(solver, 2, j, k), c]))
        d = max(d, abs(Q[padded_index(solver, nx, j, k), c] - Q[padded_index(solver, nx-1, j, k), c]))
    end
    @test d == 0.0
    # a uniform state must be untouched by the extrapolation
    Q2 = allocate_state(solver)
    initialize!(solver, Q2, (x, y, z) -> Prim(u=(0.2, 0, 0), p=1.0, rho=1.0))
    ref = copy(Q2)
    apply_bcs!(solver, Q2)
    @test Q2 == ref
end

@testset "NSCBC inflow: matched uniform stream ⇒ no correction" begin
    # Mirrors the outflow test. Nothing else in the suite compiles this
    # method, so without it the inflow path, the transverse terms included,
    # is never executed.
    uin = (0.3, 0.0, 0.0)
    solver = Solver(n_global=(32, 12, 12), L_domain=(1.0, 0.4, 0.4),
               bcs=((NSCBCInflowBC(u=uin, T_ion=1.0), NSCBCOutflowBC(pinf=1.0)),
                    per3[2], per3[3]),
               eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
               art=ArtificialProperties(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=uin, p=1.0, T_ion=1.0))
    apply_bcs!(solver, Q)
    dQ = zero(Q)
    compute_rhs!(solver, Q, dQ)
    m = maximum(abs(dQ[padded_index(solver, 1, j, k), c])
                for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12)
    @test m < 1e-8
    # and a mismatched stream must produce a non-trivial correction
    Q2 = allocate_state(solver)
    initialize!(solver, Q2, (x, y, z) -> Prim(u=(0.15, 0, 0), p=1.0, T_ion=1.0))
    apply_bcs!(solver, Q2)
    dQ2 = zero(Q2)
    compute_rhs!(solver, Q2, dQ2)
    m2 = maximum(abs(dQ2[padded_index(solver, 1, j, k), c])
                 for c in 1:solver.equations.n_cons, j in 1:12, k in 1:12)
    @test m2 > 1e-3
end

@testset "NSCBC inflow: transverse terms freeze the incoming characteristics" begin
    # With the relaxation rates at zero and beta_t = 1, every imposed
    # incoming amplitude is −𝒯, so at the plane the rates of the incoming
    # characteristic variables, p ± ρc u_n for the incoming acoustic wave,
    # c²ρ − p, u_t and Y_k, vanish up to the truncation gap between the
    # conservative divergence in dQ and the characteristic form; the outgoing
    # one stays of order one. The transverse pressure gradient's share of 𝒯_t
    # carries the Mach number instead (src/nscbc.jl), so the u_t rate measured
    # is the remainder after the −(1 − M) ∇_t p/ρ that share leaves, with ∇_t p
    # the solver's own derivative. Under beta_t = 0 the same rates are of
    # order one, and doubling the transverse resolution shrinks the gap at the
    # sixth order (measured ×70; ×16 is asserted). Both faces, two species
    # of one γ so that ρe = p/(γ − 1) gives ∂p/∂t from dQ.
    γ = 1.4
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, γ),
                        IdealSpecies{Float64}("b", 0.5, γ)])
    ic(sgn) = (x, y, z) -> begin
        θ = 0.5 + 0.2 * sin(2π * y) * cos(4π * z)
        Prim(u=(sgn * (0.3 + 0.05 * sin(2π * y) + 0.02 * cos(4π * z)),
                0.1 * cos(2π * y), 0.05 * sin(4π * z) * cos(2π * y)),
             p=1 + 0.05 * sin(2π * y + 1) + 0.02 * cos(4π * z),
             T_ion=1 + 0.1 * cos(2π * y) * sin(4π * z), Y=(θ, 1 - θ))
    end
    function rhs(side, beta_t, n; eta=0.0, target=nothing)
        sgn = side == 1 ? 1.0 : -1.0
        bc = NSCBCInflowBC(u=(0.3sgn, 0.0, 0.0), T_ion=1.0, Y=[0.6, 0.4],
                           eta_u=eta, eta_T=eta, eta_t=eta, eta_Y=eta,
                           beta_t=beta_t, target=target)
        out = NSCBCOutflowBC(pinf=1.0)
        solver = Solver(n_global=n, L_domain=(1.0, 1.0, 0.5),
                        bcs=(side == 1 ? (bc, out) : (out, bc), per3[2], per3[3]),
                        eos=eos, transport=ConstantTransport(mu0=0.0),
                        art=ArtificialProperties(enabled=false))
        Q = allocate_state(solver)
        initialize!(solver, Q, ic(sgn))
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        return solver, dQ
    end
    # Plane maxima of the incoming characteristic rates and of the outgoing.
    function rates(solver, dQ, side; pressure=true)
        dpy, dpz = solver.tmp_a, solver.tmp_b
        CompactLES.deriv_along!(dpy, solver.p, solver, 2, 1)
        CompactLES.deriv_along!(dpz, solver.p, solver, 3, 1)
        m = solver.equations.i_mom
        ie = solver.equations.i_energy
        nx, ny, nz = solver.decomp.n_local
        i = side == 1 ? 1 : nx
        sgn = side == 1 ? 1.0 : -1.0
        ac_in = ac_out = en = tv = ty = 0.0
        for k in 1:nz, j in 1:ny
            I = padded_index(solver, i, j, k)
            ρ = solver.rho[I]; c = solver.c[I]
            u = (solver.u[I], solver.v[I], solver.w[I])
            ρt = dQ[I, 1] + dQ[I, 2]
            ut = ntuple(a -> (dQ[I, m[a]] - u[a] * ρt) / ρ, 3)
            pt = (γ - 1) * (dQ[I, ie] - sum(abs2, u) / 2 * ρt - ρ * sum(u .* ut))
            ac_in = max(ac_in, abs(pt + sgn * ρ * c * ut[1]))
            ac_out = max(ac_out, abs(pt - sgn * ρ * c * ut[1]))
            en = max(en, abs(c^2 * ρt - pt))
            gy, gz = solver.inv_h[2][I] * dpy[I], solver.inv_h[3][I] * dpz[I]
            share = pressure ? (1 - abs(u[1]) / c) / ρ : 0.0
            tv = max(tv, abs(ut[2] + share * gy), abs(ut[3] + share * gz))
            ty = max(ty, abs((dQ[I, 1] - solver.Y[1][I] * ρt) / ρ))
        end
        return (ac_in, en, tv, ty), ac_out
    end
    for side in (1, 2)
        full, out = rates(rhs(side, 1.0, (24, 24, 16))..., side)
        lodi, _ = rates(rhs(side, 0.0, (24, 24, 16))..., side; pressure=false)
        fine, _ = rates(rhs(side, 1.0, (24, 48, 32))..., side)
        @test out > 1.0
        @test all(lodi .> (1.0, 0.05, 0.1, 0.05))
        @test all(full ./ lodi .< 1e-2)
        @test all(full ./ fine .> 16)
    end
    # A pointwise target returning the constants reproduces the constant
    # body bitwise, relaxation on; the weight changes the answer.
    _, d1 = rhs(1, 1.0, (24, 24, 16); eta=0.28)
    _, d2 = rhs(1, 1.0, (24, 24, 16); eta=0.28,
                target=(x, y, z, t) -> Prim(u=(0.3, 0.0, 0.0), T_ion=1.0,
                                            p=1.0, Y=(0.6, 0.4)))
    _, d3 = rhs(1, 0.0, (24, 24, 16); eta=0.28)
    @test parent(d1) == parent(d2)
    @test parent(d1) != parent(d3)
end

@testset "NSCBC inflow: the r-z face freezes the incoming wave beside the axis" begin
    # The top face of an r-z grid with a radial velocity proportional to r
    # beside the axis, relaxation off. The transverse divergence there is
    # ∂u_r/∂r + u_r/r, the second term from the collapsed θ; with it in 𝒯
    # the incoming acoustic rate p − ρc u_z and the u_r rate fall with the
    # resolution as on a Cartesian face (×19 and ×12 from 24 to 48 nodes).
    # Without it the acoustic rate stays at half its beta_t = 0 value. The u_r
    # rate is measured after the −(1 − M) ∂p/∂r/ρ that the pressure share of
    # 𝒯_t leaves, as on the Cartesian face above.
    γ = 1.4
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, γ),
                        IdealSpecies{Float64}("b", 0.5, γ)])
    ic(r, θ, z) = begin
        s = 0.5 + 0.2 * cos(2π * r^2) * sin(2π * z)
        Prim(u=(0.1 * r * (1 - r^2) * (1 + 0.5 * cos(2π * z)), 0.0,
                -(0.3 + 0.05 * cos(π * r^2) + 0.02 * sin(2π * z))),
             p=1 + 0.05 * cos(2π * r^2 + 1) + 0.02 * cos(2π * z),
             T_ion=1 + 0.1 * cos(π * r^2) * sin(2π * z), Y=(s, 1 - s))
    end
    function face_rates(beta_t, n; pressure=true)
        bc = NSCBCInflowBC(u=(0.0, 0.0, -0.3), T_ion=1.0, Y=[0.6, 0.4], eta_u=0.0,
                           eta_T=0.0, eta_t=0.0, eta_Y=0.0, beta_t=beta_t)
        solver = Solver(n_global=(n, 1, n), L_domain=(1.0, 2π, 1.0),
                        metric=CylindricalMetric(),
                        bcs=((AxisBC(), SlipWallBC()), per3[2],
                             (NSCBCOutflowBC(pinf=1.0), bc)),
                        eos=eos, transport=ConstantTransport(mu0=0.0),
                        art=ArtificialProperties(enabled=false))
        Q = allocate_state(solver)
        initialize!(solver, Q, ic)
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        m = solver.equations.i_mom
        ie = solver.equations.i_energy
        dpr = solver.tmp_a
        CompactLES.deriv_along!(dpr, solver.p, solver, 1, 1)
        acoustic = radial = 0.0
        for i in 1:n
            I = padded_index(solver, i, 1, n)
            ρ = solver.rho[I]; c = solver.c[I]
            u = (solver.u[I], solver.v[I], solver.w[I])
            ρt = dQ[I, 1] + dQ[I, 2]
            ut = ntuple(a -> (dQ[I, m[a]] - u[a] * ρt) / ρ, 3)
            pt = (γ - 1) * (dQ[I, ie] - sum(abs2, u) / 2 * ρt - ρ * sum(u .* ut))
            acoustic = max(acoustic, abs(pt - ρ * c * ut[3]))
            share = pressure ? (1 - abs(u[3]) / c) / ρ : 0.0
            radial = max(radial, abs(ut[1] + share * solver.inv_h[1][I] * dpr[I]))
        end
        return acoustic, radial
    end
    full = face_rates(1.0, 24)
    fine = face_rates(1.0, 48)
    lodi = face_rates(0.0, 24; pressure=false)
    @test all(lodi .> (0.1, 0.1))
    @test all(full ./ lodi .< 1e-2)
    @test all(full ./ fine .> 8)
end

@testset "NSCBC outflow: the r-z face freezes the incoming wave beside the axis" begin
    # The inflow test above with the stream reversed, leaving through the top
    # face under sigma = 0 and beta_t = 1, where the imposed amplitude is −𝒯
    # and the incoming acoustic rate p − ρc u_z vanishes up to the truncation
    # gap. With the collapsed θ's u_r/r in 𝒯 the rate falls with the
    # resolution (measured ×14 from 24 to 48 nodes); without it the rate stays
    # at half its beta_t = 0 value.
    γ = 1.4
    ic(r, θ, z) = Prim(u=(0.1 * r * (1 - r^2) * (1 + 0.5 * cos(2π * z)), 0.0,
                          0.3 + 0.05 * cos(π * r^2) + 0.02 * sin(2π * z)),
                       p=1 + 0.05 * cos(2π * r^2 + 1) + 0.02 * cos(2π * z),
                       T_ion=1 + 0.1 * cos(π * r^2) * sin(2π * z))
    function face_rate(beta_t, n)
        solver = Solver(n_global=(n, 1, n), L_domain=(1.0, 2π, 1.0),
                        metric=CylindricalMetric(),
                        bcs=((AxisBC(), SlipWallBC()), per3[2],
                             (NSCBCInflowBC(u=(0.0, 0.0, 0.3), T_ion=1.0),
                              NSCBCOutflowBC(pinf=1.0, sigma=0.0, beta_t=beta_t))),
                        eos=IdealSpecies("gas"; gamma=γ, R=1.0),
                        transport=ConstantTransport(mu0=0.0),
                        art=ArtificialProperties(enabled=false))
        Q = allocate_state(solver)
        initialize!(solver, Q, ic)
        apply_bcs!(solver, Q)
        dQ = zero(Q)
        compute_rhs!(solver, Q, dQ)
        m = solver.equations.i_mom
        ie = solver.equations.i_energy
        acoustic = 0.0
        for i in 1:n
            I = padded_index(solver, i, 1, n)
            ρ = solver.rho[I]; c = solver.c[I]
            u = (solver.u[I], solver.v[I], solver.w[I])
            ρt = dQ[I, 1]
            ut = ntuple(a -> (dQ[I, m[a]] - u[a] * ρt) / ρ, 3)
            pt = (γ - 1) * (dQ[I, ie] - sum(abs2, u) / 2 * ρt - ρ * sum(u .* ut))
            acoustic = max(acoustic, abs(pt - ρ * c * ut[3]))
        end
        return acoustic
    end
    full = face_rate(1.0, 24)
    lodi = face_rate(0.0, 24)
    @test lodi > 0.1
    @test full / lodi < 1e-2
    @test full / face_rate(1.0, 48) > 8
end

@testset "validate_bc: NSCBC restrictions are setup errors" begin
    # Both restrictions are setup errors rather than documented caveats: left
    # unchecked, an angular face would fail nothing, since the wave analysis
    # drops the curvature terms carried by grad_u[d,d], and the run would
    # complete with a wrong answer.
    two = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    wall = (SlipWallBC(), SlipWallBC())
    cyl(θbc) = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0),
                      bcs=(wall, θbc, per3[3]), metric=CylindricalMetric(),
                      origin=(1.0, 0.0, 0.0), art=ArtificialProperties(enabled=false))
    # r is a length under CylindricalMetric, θ is not.
    @test cyl(wall) isa Solver
    @test_throws ErrorException cyl((NSCBCOutflowBC(pinf=1.0), SlipWallBC()))
    @test_throws ErrorException cyl((SlipWallBC(),
                                     NSCBCInflowBC(u=(0.0, 0.1, 0.0), T_ion=1.0)))
    # Composition length, checked once at setup rather than on every RHS call.
    cart(Y) = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0),
                     bcs=((NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0, Y=Y),
                           NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
                     eos=two, art=ArtificialProperties(enabled=false))
    @test cart([0.4, 0.6]) isa Solver
    @test_throws ErrorException cart([1.0])
    @test_throws ErrorException cart([0.2, 0.3, 0.5])
end

@testset "multi-species artificial diffusivity responds to Y gradients" begin
    # artificial.jl computes per-species D* only when n_species > 1; that branch was
    # never executed, since the only multi-species test disables art.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = Solver(n_global=(64, 12, 12), L_domain=(1.0, 0.2, 0.2), bcs=per3, eos=eos,
               art=ArtificialProperties(enabled=true))
    Q = allocate_state(solver)
    dQ = zero(Q)
    # smooth composition: D* must stay tiny
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.5 + 0.1sin(2π * x), 0.5 - 0.1sin(2π * x)),
                                        p=1.0, rho=1.0))
    compute_rhs!(solver, Q, dQ)
    smooth_max = max(maximum(solver.D_art[1]), maximum(solver.D_art[2]))
    # sharp interface: D* must switch on
    initialize!(solver, Q, (x, y, z) -> begin
        θ = tanh_blend(x, 0.5, 0.01)
        Prim(Y=(1 - θ, θ), p=1.0, rho=1.0)
    end)
    compute_rhs!(solver, Q, dQ)
    sharp_max = max(maximum(solver.D_art[1]), maximum(solver.D_art[2]))
    @test sharp_max > 100 * max(smooth_max, 1e-14)
    @test all(isfinite, solver.D_art[1]) && all(isfinite, solver.D_art[2])
end

@testset "artificial-property weights are the local physical spacing" begin
    # Every sensor weight is h[d] / inv_h[d][I], the arclength of one
    # computational cell, and the mass-fraction bound is the geometric mean of
    # the active ones. Two grids carrying the same physical spacing must
    # therefore produce the same coefficients whatever computational spacing
    # they are written in. N − 1 is a power of two so that the coordinates,
    # the spacings and the Jacobian are all exact and nothing below turns on
    # a last-bit difference in the grid itself.
    Nx = 33
    eos2 = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                         IdealSpecies{Float64}("b", 0.2, 1.09)])
    # One closure type for both mappings below, so the two stretched runs share
    # a single `Solver` specialization instead of compiling one each.
    linmap(a) = Stretch(ξ -> a * ξ, ξ -> a)
    function art_coefficients(L, st)
        s = Solver(n_global=(Nx, 12, 12), L_domain=(L, 0.2, 0.2), eos=eos2,
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   art=ArtificialProperties(enabled=true), stretch=(st, nothing, nothing))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = tanh_blend(x, 0.5L, 0.03L)
            Prim(Y=(1 - θ, θ), rho=1 + θ, p=1.0,
                 u=(0.4sin(2π * x / L), 0.0, 0.0))
        end)
        # Drive one plane's mass fractions out of [0, 1] so the bound term,
        # the only consumer of the geometric mean, is exercised as well.
        for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2]
            I = padded_index(s, Nx ÷ 4, j, k)
            ρ = Q[I, 1] + Q[I, 2]
            Q[I, 1] -= 0.05ρ
            Q[I, 2] += 0.05ρ
        end
        apply_bcs!(s, Q)
        compute_rhs!(s, Q, zero(Q))
        (copy(s.mu_art), copy(s.beta_art), copy(s.kappa_art),
         copy(s.D_art[1]), copy(s.D_art[2]))
    end
    names = ("mu_art", "beta_art", "kappa_art", "D_art[1]", "D_art[2]")

    # An identity mapping leaves inv_h at exactly one, so the per-point weight
    # is the scalar it replaced and every coefficient reproduces bitwise.
    plain = art_coefficients(1.0, nothing)
    ident = art_coefficients(1.0, linmap(1.0))
    for (nm, a, b) in zip(names, ident, plain)
        a == b || println("identity stretch moved $nm")
        @test a == b
    end
    @test maximum(plain[1]) > 0 && maximum(plain[4]) > 0

    # A mapping of constant Jacobian 2 over (0, 2) is the uniform grid of the
    # same physical spacing written in half the computational spacing. The
    # coefficients agree to round-off; weighting by the computational spacing
    # instead makes them differ by 2^wpow, a factor of four on mu*/beta* and
    # two on kappa*/D*.
    uniform = art_coefficients(2.0, nothing)
    linear = art_coefficients(2.0, linmap(2.0))
    for (nm, a, b) in zip(names, linear, uniform)
        scale = maximum(abs, b)
        rel = maximum(abs, a - b) / scale
        rel < 1e-12 || println("linear stretch moved $nm by $rel relative")
        @test scale > 0
        @test rel < 1e-12
    end
end

@testset "stretched shocked interface: the species bound holds" begin
    # The only case combining Numerics.stretch with the multi-species sensors.
    # The cluster point sits just short of the interface, so the shock crosses
    # it where the physical spacing is about half the computational one and the
    # two weightings differ most. Cheap by construction: 121 points, and the
    # run stops once the transmitted shock has cleared the interface.
    st = sine_cluster(0.0, 1.0, SI_X_IFACE, 0.5)
    r = shock_interface(N=121, tfin=0.15, delta=2.0, nmax=4000, stretch1=st)
    @test r.completed
    @test all(isfinite, r.Y_air)
    # Measured -0.0076 / 1.0076 over 196 steps; the guard is roughly 4x that,
    # the same run on a uniform grid of the same point count reaching 0.0137.
    @test -0.03 < r.worst_min_Y && r.worst_max_Y < 1.03
    @test 0 < r.width_cells < 30
end

@testset "consistent species channels: invariance, shocked interface, slab, width" begin
    # `species_flux = :bulk` diffuses every conserved variable with one D_b, and
    # the default `:partial_density` diffuses the partial densities with it and
    # carries their mass flux into momentum and energy (reference/DESIGN.md,
    # "The species channel"). Four measured properties are guarded for both.
    fick = ArtificialProperties(enabled=true, species_flux=:fickian)
    for bulk in (ArtificialProperties(enabled=true, species_flux=:bulk),
                 ArtificialProperties(enabled=true, species_flux=:partial_density))
        # (a) A resting air/SF6 interface at uniform p, T and u = 0, with the
        # artificial properties and the filter on: every operator of either
        # channel preserves the uniform state to round-off while the partial
        # densities interdiffuse, so u, p and T stay at round-off and ρ moves.
        # Under the Fickian channel the enthalpy flux drives u to 1e-4 here.
        let N = 400, h = 1.0 / N, δ = 2h
            eos = IdealMixture([IdealSpecies{Float64}("air", 1.0, 1.4),
                                IdealSpecies{Float64}("sf6", 1 / 5.04, 1.09)])
            Rk, cvk = eos.Rk, eos.cvk
            two(x) = (θ = tanh_blend(x, 0.3, δ) - tanh_blend(x, 0.7, δ); (1 - θ, θ))
            prob = Problem(eos=eos, transport=ConstantTransport(mu0=0.0),
                           domain=((0.0, 1.0), (0.0, h), (0.0, h)), bcs=per3,
                           ic=(x, y, z) -> begin
                               Y = two(x)
                               Prim(Y=Y, rho=1 / (Y[1] * Rk[1] + Y[2] * Rk[2]),
                                    u=(0.0, 0.0, 0.0), p=1.0)
                           end)
            solver, Q = setup(prob, Numerics(n_global=(N, 1, 1), art=bulk, cfl=0.4,
                                             control=StepControl(validity=:permissive)))
            nx = solver.decomp.n_local[1]
            eqs = solver.equations
            rho0 = [Q[padded_index(solver, i, 1, 1), 1] + Q[padded_index(solver, i, 1, 1), 2] for i in 1:nx]
            umax = Ref(0.0); dpmax = Ref(0.0); dTmax = Ref(0.0); drmax = Ref(0.0)
            function drift(s, Q)
                for i in 1:nx
                    I = padded_index(s, i, 1, 1)
                    ρ = Q[I, 1] + Q[I, 2]
                    y = Q[I, 1] / ρ
                    u = Q[I, eqs.i_mom[1]] / ρ
                    e = Q[I, eqs.i_energy] / ρ - 0.5u^2
                    T = e / (y * cvk[1] + (1 - y) * cvk[2])
                    p = ρ * (y * Rk[1] + (1 - y) * Rk[2]) * T
                    umax[] = max(umax[], abs(u))
                    dpmax[] = max(dpmax[], abs(p - 1))
                    dTmax[] = max(dTmax[], abs(T - 1))
                    drmax[] = max(drmax[], abs(ρ - rho0[i]) / rho0[i])
                end
            end
            run!(solver, Q; tfinal=0.25, nmax=1000, callback=drift)
            # Measured 1.2e-14, 3.8e-14, 4.0e-14 and 2.0e-2 over 298 steps; the
            # Fickian channel measured 2.2e-4, 1.3e-4, 2.5e-5 and 1.8e-2 here
            # under `detector = :delta4, C_D = 0.1`.
            @test completed(solver, 0.25)
            @test umax[] < 1e-13
            @test dpmax[] < 1e-13
            @test dTmax[] < 1e-13
            @test drmax[] > 1e-3
        end

        # (b) The Mach 1.5 shocked interface at density ratio 100 and 2h: the
        # consistent channels carry it (measured worst Y −0.0091 in 673 steps,
        # volume-fraction width 4.8 cells and TV − 1 0.020 under `:bulk`;
        # −0.0092 in 696, 5.3 cells and 0.0020 under `:partial_density`);
        # the Fickian channel loses positivity at step 427 with a DomainError out
        # of the sound speed, so a bounded attempt must either raise or stop
        # short of completion.
        let r = shock_interface(art=bulk, delta=2.0, rho_heavy=100.0, nmax=1500)
            @test r.completed
            @test all(isfinite, r.rho) && minimum(r.rho) > 0
            @test r.worst_min_Y > -0.03
            @test r.worst_max_Y < 1.03
            # The width is read in volume fraction: at this density ratio the
            # mass-fraction count of `width_cells` reads the heavy-gas tail
            # on the air side, 0.84 < V < 0.9995, not the interface. The
            # total variation is the ringing the species detector removes.
            tv_excess = sum(abs, diff(r.Y_air)) - 1
            @test volume_width(r.x, r.Y_air, 100.0) <= 7
            @test tv_excess < 0.04
        end

        # (c) The Brill slab at density ratio 100 and 7 cells per interface, ten
        # periods: the pressure error at the end is round-off under both channels
        # (measured about 5e-11 in 4049 steps) and 5.5e-2 under the
        # Fickian one (4.0e-2 under `detector = :delta4, C_D = 0.1`), whose
        # enthalpy flux is the one operator that moves ρE
        # across a uniform-pressure interface of unequal gas constants.
        let r = brill_slab(art=bulk, nmax=6000)
            @test r.completed
            @test all(isfinite, r.rho) && minimum(r.rho) > 0
            @test r.p_error < 1e-9
            @test r.u_error < 1e-9
            @test r.worst_min_Y > -0.1
        end

        # (d) At equal molecular weights the mole fraction is the mass fraction,
        # the bulk flux of ρY_k at uniform ρ is the Fickian flux, and the two
        # channels' energy fluxes both vanish, so the advected interface's width
        # agrees to round-off accumulation (measured identical to eight digits,
        # the mass fractions within 5e-14).
        let (xs_f, Yf, _, _, okf) = species_advection(art=fick, nmax=2000),
            (xs_b, Yb, _, _, okb) = species_advection(art=bulk, nmax=2000)
            @test okf && okb
            wf = contact_width(xs_f, Yf, 0.0, 1.0)
            wb = contact_width(xs_b, Yb, 0.0, 1.0)
            @test abs(wb - wf) < 1e-6 * wf
        end
    end

    # The Fickian channel loses positivity on case (b).
    let ok = try
            shock_interface(art=fick, delta=2.0, rho_heavy=100.0, nmax=600).completed
        catch err
            err isa DomainError || err isa SolverFailure || rethrow()
            false
        end
        @test !ok
    end

    # The option is validated at setup.
    @test_throws ErrorException Solver(n_global=(32, 12, 12), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, art=ArtificialProperties(species_flux=:brill))
end

@testset "consistent species channels on patched, refined and tiled layouts" begin
    # A slab of density ratio 20 advected at uniform (u, p, T) through a
    # periodic domain, under both `species_flux = :bulk` and the default
    # `:partial_density`. Every operator either channel meets on these layouts
    # is linear on the conserved variables, the interface rows and the level
    # transfers included, so u, p and T must stay at round-off on every
    # layout while the partial densities interdiffuse (reference/DESIGN.md,
    # "The species channel"). The Fickian channel's enthalpy flux moves p to
    # 1e-4 on the same case. Measured drifts are 1e-14 on each layout below;
    # D_b is small on the smooth slab but must be nonzero on every patch,
    # which is what shows the channel is built and running there.
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 1 / 20, 1.4)])
    Rk, cvk = eos.Rk, eos.cvk
    function slab(channel; kw...)
        N = 96
        h = 1.0 / N
        s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3, eos=eos,
                   art=ArtificialProperties(enabled=true, species_flux=channel),
                   transport=ConstantTransport(mu0=0.0), filter_interval=1; kw...)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) -> begin
            V = (1 - tanh((abs(x - 0.5) - 0.25) / 3h)) / 2
            ρ = V * 20 + (1 - V)
            Prim(Y=(1 - V * 20 / ρ, V * 20 / ρ), rho=ρ, u=(1.0, 0.0, 0.0), p=1.0)
        end)
        drift = Ref(0.0)
        function watch(sol, Qs)
            for (ps, Q) in CL.eachpatch(sol, Qs isa Vector ? Qs : [Qs])
                eq = ps.equations
                for i in 1:ps.decomp.n_local[1]
                    I = padded_index(ps, i, 1, 1)
                    ρ = Q[I, 1] + Q[I, 2]
                    y = Q[I, 1] / ρ
                    u = Q[I, eq.i_mom[1]] / ρ
                    e = Q[I, eq.i_energy] / ρ - 0.5u^2
                    T = e / (y * cvk[1] + (1 - y) * cvk[2])
                    p = ρ * (y * Rk[1] + (1 - y) * Rk[2]) * T
                    drift[] = max(drift[], abs(u - 1), abs(p - 1), abs(T - 1))
                end
            end
        end
        run!(s, states; tfinal=1.0, nmax=40, callback=watch)
        sv = states isa Vector ? states : [states]
        Dmax = [maximum(ps.D_art[1]) for (ps, _) in CL.eachpatch(s, sv)]
        rho = Dict{Int,Float64}()          # root nodes, by global index
        for (ps, Q) in CL.eachpatch(s, sv)
            ps.patch.level == 0 || continue
            for i in 1:ps.decomp.n_local[1]
                I = padded_index(ps, i, 1, 1)
                rho[ps.patch.region.offset[1] + i] = Q[I, 1] + Q[I, 2]
            end
        end
        return (s=s, states=sv, drift=drift[], Dmax=Dmax, rho=rho)
    end
    for channel in (:bulk, :partial_density)
        single = slab(channel)
        @test single.s.step == 40
        @test single.drift < 1e-12
        @test single.Dmax[1] > 0
        # Two patches under both interface forms: the conserved gradients take
        # the interface plans `grad_Y` takes. The patched root differs from the
        # single patch by the interface rows' own error (measured 2e-5 with the
        # extended rows and 5e-4 with the one-sided ones on a wider slab), not
        # by the channel. The one-sided form takes the closure rows.
        for irhs in (:extended, :onesided)
            iflux = irhs === :onesided ? :closure : :ghost
            r = slab(channel; patch_grid=(2, 1, 1), interface_rhs=irhs,
                     interface_flux=iflux)
            @test npatches(r.s) == 2
            @test r.s.step == 40
            @test r.drift < 1e-12
            @test all(>(0), r.Dmax)
            d = maximum(abs(r.rho[k] - single.rho[k]) for k in keys(single.rho))
            @test d < 1e-2
        end
        # One refined level over the slab's left edge, as a box and as a
        # lattice of tiles; the fine patches carry their own D_b.
        for tile in (0, 8)
            r = slab(channel; refine=BlockRegion((16, 0, 0), (24, 1, 1)),
                     tile=tile)
            @test npatches(r.s) == (tile == 0 ? 2 : 4)
            @test r.s.step == 40
            @test all(Q -> all(isfinite, parent(Q)), r.states)
            @test r.drift < 1e-12
            @test all(>(0), r.Dmax)
        end
    end
end

@testset "compression-keyed beta sensors: gated_strain and dilatation" begin
    # The point of both non-default sensors is that bulk viscosity stops firing
    # on vortical structures and on expansions. Both halves are checked here,
    # along with the requirement that mu* — which keeps the strain sensor in
    # every case — is bit-identical across the three settings.
    tgv(sensor) = begin
        s = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3,
                   art=ArtificialProperties(enabled=true, beta_sensor=sensor))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=100.0,
            u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.0)))
        compute_rhs!(s, Q, zero(Q))
        s
    end
    ss, sg, sd = tgv(:strain), tgv(:gated_strain), tgv(:dilatation)
    # Taylor–Green is solenoidal, so neither compression-keyed sensor has
    # anything to fire on; the strain sensor responds to the vortical structure and
    # fires. :dilatation collapses to round-off because its sensor is built
    # from ∇·u. :gated_strain keeps the firing sensor and leans on the switch
    # alone, which removes 99.4% of the total but not the maximum: the strain
    # sensor peaks at the cusps of |S|, and the switch degenerates at exactly
    # those points, where the vorticity vanishes along with |S| itself. The
    # test is therefore on the total, with the surviving peak recorded here
    # rather than asserted away.
    bsum(s) = sum(s.beta_art[padded_index(s, i, j, k)] for i in 1:32, j in 1:32, k in 1:32)
    @test maximum(sd.beta_art) < 1e-12 * maximum(ss.beta_art)
    @test bsum(sg) < 1e-2 * bsum(ss)
    @test sd.mu_art == ss.mu_art
    @test sg.mu_art == ss.mu_art
    @test all(isfinite, sd.beta_art) && all(isfinite, sg.beta_art)

    # A velocity ramp: beta* must be exactly zero wherever the flow expands,
    # and must still switch on where it compresses.
    ramp(sensor) = begin
        s = Solver(n_global=(64, 12, 12), L_domain=(1.0, 0.2, 0.2), bcs=per3,
                   art=ArtificialProperties(enabled=true, beta_sensor=sensor))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0,
                                            u=(0.5tanh((x - 0.5) / 0.02), 0.0, 0.0)))
        compute_rhs!(s, Q, zero(Q))
        s
    end
    div1(s, I) = s.grad_u[1, 1][I] + s.grad_u[2, 2][I] + s.grad_u[3, 3][I]
    rstrain = ramp(:strain)
    for sensor in (:gated_strain, :dilatation)
        s = ramp(sensor)
        idx = [padded_index(s, i, 1, 1) for i in 1:64]
        expanding = [I for I in idx if div1(s, I) > 0]
        @test !isempty(expanding)
        @test all(I -> s.beta_art[I] == 0.0, expanding)
        @test maximum(s.beta_art) > 0.1 * maximum(rstrain.beta_art)
    end
    # gated_strain multiplies the strain sensor by a factor in [0, 1], so it can
    # only ever reduce beta*, never move it somewhere new.
    sg2 = ramp(:gated_strain)
    @test all(i -> sg2.beta_art[padded_index(sg2, i, 1, 1)] <=
                   rstrain.beta_art[padded_index(rstrain, i, 1, 1)] + 1e-300, 1:64)

    @test_throws ErrorException Solver(n_global=(16, 16, 16), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, art=ArtificialProperties(beta_sensor=:bogus))
end

@testset "compact_d8 ring detector: symbol, closures, and selectivity" begin
    # The d8 detector differs below the Nyquist,
    # so the operator is pinned against its analytic symbol rather than
    # against a convergence order. It is also the only symmetric banded
    # scheme in the package, hence the only exercise of BandPlan's filter-side
    # sign conventions (RHS added rather than subtracted, high-edge closure
    # rows mirrored rather than negated).
    N = 64
    art8 = ArtificialProperties(detector=:d8)
    s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), art=art8, bcs=per3)
    f = CL.field(s.decomp); out = CL.field(s.decomp)
    pad = s.decomp.n_halo_d[1]
    A(k) = 1 + 2 * (14 / 29) * cos(k) + 2 * (3 / 58) * cos(2k)
    # RHS is 60/(240·29) times the undivided eighth difference (2 − 2cos k)⁴.
    symbol(k) = (60 / (240 * 29)) * (2 - 2cos(k))^4 / A(k)
    for kf in (0.25, 0.5, 0.75, 1.0)
        k = kf * π
        for i in 1:N
            f[i+pad, 1, 1] = cos(k * i)
        end
        CL.exchange_halos!(f, s.decomp)
        # Each dimension's entry is a (wall sign +1, wall sign −1) pair; on a
        # periodic dimension the two slots hold one plan.
        CL.apply_along!(out, s.ring_plans[1][1], f, s.decomp)
        i0 = N ÷ 2
        # rtol is 1e-9, not machine epsilon: the detector is a high-pass, so a
        # well-resolved wave is the difference of coefficients of order 1
        # producing an answer of order 1e-4, and three digits go to
        # cancellation. That is a property of the operator, not slack here.
        @test out[i0+pad, 1, 1] / cos(k * i0) ≈ symbol(k) rtol = 1e-9
    end
    # Normalization: the response at two points per wavelength is 16, matching
    # undivided δ⁴ there. This separation lets the four
    # constants transfer between detectors.
    @test symbol(π) ≈ 16.0 rtol = 1e-12
    # Selectivity against undivided δ⁴ on the same normalization: measured
    # 569× at eight points per wavelength and 26× at four.
    d4(k) = (2 - 2cos(k))^2
    @test symbol(0.25π) < d4(0.25π) / 500
    @test symbol(0.5π) < d4(0.5π) / 25

    # Constants must be annihilated exactly through the four closure rows, not
    # by cancellation: every row's weights sum to zero by construction. A slip
    # wall takes the node-centred rows, an extrapolation face the scheme's own.
    for lo in (SlipWallBC(), ExtrapolationBC())
        sw = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), art=art8,
                    bcs=((lo, lo), per3[2], per3[3]))
        fw = CL.field(sw.decomp); ow = CL.field(sw.decomp)
        fill!(fw, 1.0)
        CL.exchange_halos!(fw, sw.decomp)
        CL.apply_along!(ow, sw.ring_plans[1][1], fw, sw.decomp)
        @test maximum(abs, ow[(pad+1):(pad+N), 1, 1]) < 1e-14
    end

    @test_throws ErrorException Solver(n_global=(16, 16, 16), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, art=ArtificialProperties(detector=:bogus))
end

@testset "sensor operators at a reflecting wall" begin
    # The `:gaussian` smoother and the `:d8` detector carry closure rows that
    # fold onto the half-offset mirror, half a cell out at a node-centred wall.
    # A reflecting face takes `wall_closures` instead, so a field exactly even
    # about both wall nodes, or exactly odd for a wall-normal velocity, must
    # reproduce the periodic run on the same spacing: the extended domain of
    # 2(N − 1) cells whose restriction is the wall run.
    N = 49
    wallbc = ((SlipWallBC(), SlipWallBC()), per3[2], per3[3])
    even_field(x) = cospi(x) + 0.5cospi(5x) + 0.1cospi(13x)
    odd_field(x) = sinpi(x) + 0.1sinpi(13x)
    both(art) = (Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), art=art,
                        bcs=wallbc),
                 Solver(n_global=(2(N - 1), 1, 1), L_domain=(2.0, 1.0, 1.0),
                        art=art, bcs=per3))
    function load(s, fn)
        f = CL.field(s.decomp)
        pad = s.decomp.n_halo_d[1]
        for i in 1:s.decomp.n_local[1]
            f[i+pad, 1, 1] = fn(CL.xcoord(s, 1, i))
        end
        CL.exchange_halos!(f, s.decomp)
        f
    end
    # The first and last six nodes of the wall run, and the same nodes of the
    # periodic one.
    win(s, a) = (pad = s.decomp.n_halo_d[1];
                 [a[i+pad, 1, 1] for i in [1:6; (N-5):N]])

    sw, sp = both(ArtificialProperties())
    fw, fp = load(sw, even_field), load(sp, even_field)
    CL.smooth!(fw, sw)
    CL.smooth!(fp, sp)
    @test maximum(abs, win(sw, fw) .- win(sp, fp)) <
          1e-14 * maximum(abs, win(sp, fp))

    dw, dp = both(ArtificialProperties(detector=:d8))
    for (fn, σw) in ((even_field, 1), (odd_field, -1))
        gw, gp = load(dw, fn), load(dp, fn)
        ow, op = CL.field(dw.decomp), CL.field(dp.decomp)
        CL.detect_sum!(ow, gw, dw, 1; wall_parity=(σw, 1, 1))
        CL.detect_sum!(op, gp, dp, 1)
        # The detector is a high-pass: on a resolved field it is the difference
        # of order-one coefficients producing an answer four orders smaller, so
        # 1e-8 relative is its round-off and not slack here.
        @test maximum(abs, win(dw, ow) .- win(dp, op)) <
              1e-8 * maximum(win(dp, op))
    end
    # An odd field vanishes on the wall node, and the odd row 1 returns that
    # exactly rather than to round-off.
    gw = load(dw, odd_field)
    ow = CL.field(dw.decomp)
    CL.detect_sum!(ow, gw, dw, 1; wall_parity=(-1, 1, 1))
    @test ow[1+dw.decomp.n_halo_d[1], 1, 1] == 0.0

    # The rows themselves: built from the interior weights, so the filter's
    # unit row sum and the eighth derivative's zero row sum are inherited.
    a, b = 3565 / 10368, 3091 / 12960
    c, d, e = 1997 / 25920, 149 / 12960, 107 / 103680
    grows = CL.wall_closures(gaussian_filter(), 1)
    @test length(grows) == 4
    @test all(row -> sum(row.rhs) ≈ 1.0, grows)
    @test grows[1].rhs ≈ [a, 2b, 2c, 2d, 2e]
    @test grows[2].rhs ≈ [b, a + c, b + d, c + e, d, e]
    @test grows[3].rhs ≈ [c, b + d, a + e, b, c, d, e]
    @test grows[4].rhs ≈ [d, c + e, b, a, b, c, d, e]
    drows = CL.wall_closures(compact_d8(), 1)
    @test all(row -> abs(sum(row.rhs)) < 1e-15, drows)
    # Row 1 folds each left-hand-side band onto the interior side, doubling it
    # for an even field; at σ = −1 the whole row collapses to g₁ = a₀ f₁.
    @test drows[1].lhs ≈ [0, 0, 1, 2 * 14 / 29, 2 * 3 / 58]
    @test drows[2].lhs ≈ [0, 14 / 29, 1 + 3 / 58, 14 / 29, 3 / 58]
    orows = CL.wall_closures(compact_d8(), -1)
    @test orows[1].lhs == [0, 0, 1, 0, 0]
    @test orows[1].rhs ≈ [35 / 58, 0, 0, 0, 0]
    # An antisymmetric scheme has no such fold.
    @test_throws ErrorException CL.wall_closures(lele_d1_6(), 1)

    # A fold's far end may be a wall. The outer end of a radial line takes the
    # same rows, so the fold holds one detector plan per ghost parity and per
    # wall sign.
    sa = Solver(n_global=(32, 1, 12), L_domain=(1.0, 1.0, 0.5),
                metric=CylindricalMetric(), art=ArtificialProperties(detector=:d8),
                bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]))
    rp = sa.folds[1].ring_plans
    @test length(rp) == 2 && all(p -> length(p) == 2, rp)
    @test rp[1][1] !== rp[1][2] && rp[2][1] !== rp[2][2]
    # An outer end that reflects nothing keeps one plan in both slots.
    sx = Solver(n_global=(32, 1, 12), L_domain=(1.0, 1.0, 0.5),
                metric=CylindricalMetric(), art=ArtificialProperties(detector=:d8),
                bcs=((AxisBC(), ExtrapolationBC()), per3[2], per3[3]))
    @test sx.folds[1].ring_plans[1][1] === sx.folds[1].ring_plans[1][2]
end

@testset "pyranda_filter: symbol, Nyquist zero, 9/10 integral, closures" begin
    # Pyranda's c8ff8 transcribed as a symmetric banded scheme, pinned against
    # its analytic symbol the way the d8 detector is; the same filter-side
    # sign conventions of BandPlan carry it.
    N = 64
    s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), filt=pyranda_filter(),
               bcs=per3)
    f = CL.field(s.decomp)
    pad = s.decomp.n_halo_d[1]
    β, α = 1.6688e-1, 6.6624e-1
    a, b, c, d, e = 9.9965e-1, 6.6652e-1, 1.6674e-1, 4.0e-5, -5.0e-6
    symbol(k) = (a + 2b * cos(k) + 2c * cos(2k) + 2d * cos(3k) + 2e * cos(4k)) /
                (1 + 2α * cos(k) + 2β * cos(2k))
    for kf in (0.25, 0.5, 0.75)
        k = kf * π
        for i in 1:N
            f[i+pad, 1, 1] = cos(k * i)
        end
        filter_field!(f, s)
        i0 = N ÷ 2
        @test f[i0+pad, 1, 1] / cos(k * i0) ≈ symbol(k) rtol = 1e-10
    end
    @test symbol(0.0) ≈ 1.0 rtol = 1e-14
    @test abs(symbol(π)) < 1e-12
    @test symbol(0.5π) > 0.99
    ks = range(0, π; length=20001)
    @test sum(symbol.(ks)) * step(ks) / π ≈ 0.9 atol = 2e-3
    # Grid-to-grid oscillation on a constant is removed to round-off.
    for i in 1:N
        f[i+pad, 1, 1] = 1.0 + 0.5 * (-1)^i
    end
    filter_field!(f, s)
    @test maximum(abs(f[i+pad, 1, 1] - 1.0) for i in 1:N) < 1e-12
    # Closed edges: the telescoping rows pass a constant exactly, and each
    # row's two sides sum equally by construction.
    for row in pyranda_filter().closures
        @test sum(row.lhs) ≈ sum(row.rhs) rtol = 1e-14
    end
    sw = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), filt=pyranda_filter(),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]))
    fw = CL.field(sw.decomp)
    fill!(fw, 1.0)
    filter_field!(fw, sw)
    @test maximum(abs(fw[i+pad, 1, 1] - 1.0) for i in 1:N) < 1e-13
end

@testset "detector = :d8 through compute_artificial!" begin
    # End to end on all three dimensions and both sensor weights, against the
    # δ⁴ default on identical data. A near-grid-scale ramp must still fire; a
    # resolved wave must fire far less than δ⁴ does, which is the reason the
    # detector exists.
    #
    # The selectivity assertion is on κ\*, not β\*, and that is the measurement
    # rather than a convenience. κ\*'s input is the internal energy, which is
    # smooth when the flow is; β\*'s is |S|, which has a cusp wherever the
    # strain passes through zero, and a cusp is a grid-scale feature no
    # detector can decline to see. On a resolved sine the two detectors
    # therefore differ by 2.6e6 on κ\* and by a factor of 1.8 on β\*. See
    # `reference/CALIBRATION_APPENDIX.md`.
    sensors(detector, fn) = begin
        s = Solver(n_global=(64, 12, 12), L_domain=(1.0, 0.2, 0.2), bcs=per3,
                   art=ArtificialProperties(enabled=true, detector=detector))
        Q = allocate_state(s)
        initialize!(s, Q, fn)
        compute_rhs!(s, Q, zero(Q))
        s
    end
    ramp = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.5tanh((x - 0.5) / 0.02), 0.0, 0.0))
    wave = (x, y, z) -> Prim(rho=1.0, p=1.0 + 0.1sinpi(2x), u=(0.0, 0.0, 0.0))
    r4, r8 = sensors(:delta4, ramp), sensors(:d8, ramp)
    w4, w8 = sensors(:delta4, wave), sensors(:d8, wave)
    for s in (r8, w8)
        @test all(isfinite, s.beta_art) && all(isfinite, s.mu_art)
        @test all(isfinite, s.kappa_art)
        @test all(>=(0.0), s.beta_art)
    end
    # A ramp two to three cells wide is close enough to the grid scale that
    # the shared normalization holds; an order of magnitude either way would
    # mean the scaling is wrong. Measured 0.23.
    @test 0.1 < maximum(r8.beta_art) / maximum(r4.beta_art) < 10
    # A pressure wave resolved over 64 points is where they must not agree.
    # Measured 3.9e-7.
    @test maximum(w8.kappa_art) < 1e-4 * maximum(w4.kappa_art)
    @test maximum(w4.kappa_art) > 0
end

@testset "detector = :d8 through a coordinate-singularity fold" begin
    # BandPlan's fold assembly with a SYMMETRIC scheme, plus the :ring role in
    # fold_apply!. The C10 fold test covers the antisymmetric case only.
    s = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 0.5),
               metric=CylindricalMetric(),
               bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
               art=ArtificialProperties(enabled=true, detector=:d8))
    f = CL.field(s.decomp); out = CL.field(s.decomp)
    # An even function of r: d8 across the axis fold must stay smooth and
    # small, and in particular must not blow up in the first cells.
    fillf!(s, f, (r, θ, z) -> exp(-4r^2))
    CL.exchange_halos!(f, s.decomp)
    CL.ring_along!(out, f, s, 1, 1)
    @test all(isfinite, out)
    # The window stops at the domain's midpoint. This field has a slope at the
    # outer slip wall, so it is not the reflection the wall closure rows
    # continue it as, and the detector reads that mismatch over the last cells;
    # the fold at the other end is what this case measures.
    @test maximum(abs, out[padded_index(s, i, 1, k)] for i in 1:32, k in 1:12) < 1e-3
    # And the full sensor path runs through the fold.
    Q = allocate_state(s)
    initialize!(s, Q, (r, θ, z) -> Prim(rho=1.0, p=1.0, u=(-1.0, 0.0, 0.0)))
    compute_rhs!(s, Q, zero(Q))
    @test all(isfinite, s.beta_art) && maximum(s.beta_art) > 0
end

@testset "sensor fields: mu* from the velocity, beta* from the dilatation" begin
    # Cook (2007) builds both sensors from |S|; Cook (2009) changes beta* to
    # ∇·u. Pyranda builds them from the velocity components and from ∇·u. The
    # difference between the two is the absolute value, and these tests pin its
    # two consequences.
    oned(art, fn; N=64) = begin
        s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3, art=art)
        Q = allocate_state(s)
        initialize!(s, Q, fn)
        compute_rhs!(s, Q, zero(Q))
        s
    end
    # 1. The two-point wave. A centered derivative annihilates it exactly, so
    # |S| and ∇·u are identically zero on the shortest wave the grid carries
    # and every sensor built from them returns zero there. The velocity sensor
    # returns the full undivided response, C_mu·ρ·16h.
    nyq = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(cospi(64x), 0.0, 0.0))
    @test maximum(oned(ArtificialProperties(mu_sensor=:strain), nyq).mu_art) == 0.0
    @test maximum(oned(ArtificialProperties(mu_sensor=:velocity), nyq).mu_art) ≈
          0.002 * 16 / 64 rtol = 1e-12
    @test maximum(oned(ArtificialProperties(beta_sensor=:ungated_dilatation),
                       nyq).beta_art) == 0.0

    # 2. The detector's selectivity survives the field change, which is the
    # reason for making it. On a wave resolved over eight points the two
    # detectors are built to differ by 569×, and they do so when applied to a
    # smooth field. Applied to |S| the same pair differs by 1.28×, the cusps
    # where the strain passes through zero being grid-scale structure at every
    # wavelength. Measured 569 and 1.28.
    wave = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(cospi(16x), 0.0, 0.0))
    peak(ms, det) =
        maximum(oned(ArtificialProperties(mu_sensor=ms, detector=det), wave).mu_art)
    @test peak(:velocity, :d8) < peak(:velocity, :delta4) / 100
    @test peak(:strain, :d8) > peak(:strain, :delta4) / 10

    # 3. Σ_d against MAX. They are the same operation in one dimension, and the
    # reduction is a per-direction one, so MAX can never exceed Σ_d anywhere.
    ramp1 = (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.5tanh((x - 0.5) / 0.02), 0.0, 0.0))
    @test oned(ArtificialProperties(reduction=:sum), ramp1).mu_art ==
          oned(ArtificialProperties(reduction=:max), ramp1).mu_art
    tgv3(red) = begin
        s = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π), bcs=per3,
                   art=ArtificialProperties(reduction=red))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=100.0,
            u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.0)))
        compute_rhs!(s, Q, zero(Q))
        s
    end
    tsum, tmax = tgv3(:sum), tgv3(:max)
    @test all(tmax.mu_art .<= tsum.mu_art)
    @test maximum(tmax.mu_art) < maximum(tsum.mu_art)

    # 4. Parity across a fold. u_r is odd there, so the detector requires the
    # sign. With it, u_r = r produces no sensor at all, that being the regular
    # behaviour of a radial velocity at the axis, while a uniform u_r produces
    # one on the first cell, the folded field having a genuine kink there.
    axial(art, fn) = begin
        s = Solver(n_global=(32, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   metric=CylindricalMetric(),
                   bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), art=art)
        Q = allocate_state(s)
        initialize!(s, Q, fn)
        compute_rhs!(s, Q, zero(Q))
        s
    end
    lin = (r, θ, z) -> Prim(rho=1.0, p=1.0, u=(r, 0.0, 0.0))
    uni = (r, θ, z) -> Prim(rho=1.0, p=1.0, u=(-1.0, 0.0, 0.0))
    # δ⁴ across the fold reads the half-offset mirror for an odd field, so this
    # is exact; the d8 closure reaches it through a line solve, hence round-off.
    # Its floor is the wider one, and not for the fold's sake: u_r = r is not
    # odd about the outer wall either, the slip condition leaving a kink on the
    # wall node, and the pentadiagonal inverse carries a decaying tail of that
    # mismatch back across the 31 cells to the axis. Measured 8.7e-14 against a
    # wall-node 5.5e-5 and the uniform case's 1.6e-4 at the axis below.
    sl = axial(ArtificialProperties(mu_sensor=:velocity), lin)
    @test sl.mu_art[padded_index(sl, 1, 1, 1)] == 0.0
    s8 = axial(ArtificialProperties(mu_sensor=:velocity, detector=:d8), lin)
    @test s8.mu_art[padded_index(s8, 1, 1, 1)] < 1e-12
    su = axial(ArtificialProperties(mu_sensor=:velocity), uni)
    @test su.mu_art[padded_index(su, 1, 1, 1)] > 0

    # 5. The ungated dilatation sensor is the form the reference uses. The
    # gated one is that sensor multiplied by the Ducros switch, so it is never
    # larger, and is exactly zero wherever the flow expands.
    ug = oned(ArtificialProperties(beta_sensor=:ungated_dilatation), ramp1)
    g = oned(ArtificialProperties(beta_sensor=:dilatation), ramp1)
    @test all(g.beta_art .<= ug.beta_art .+ 1e-300)
    div1(s, I) = s.grad_u[1, 1][I] + s.grad_u[2, 2][I] + s.grad_u[3, 3][I]
    expanding = [padded_index(ug, i, 1, 1) for i in 1:64 if div1(ug, padded_index(ug, i, 1, 1)) > 0]
    @test !isempty(expanding)
    @test all(I -> g.beta_art[I] == 0.0, expanding)
    @test any(I -> ug.beta_art[I] > 0.0, expanding)

    # 6. The channels are independent: rebuilding one leaves the other's
    # numbers bit-identical to the default configuration's.
    base = oned(ArtificialProperties(), ramp1)
    @test oned(ArtificialProperties(mu_sensor=:velocity), ramp1).beta_art == base.beta_art
    @test oned(ArtificialProperties(beta_sensor=:ungated_dilatation),
               ramp1).mu_art == base.mu_art
    @test oned(ArtificialProperties(mu_sensor=:velocity), ramp1).mu_art != base.mu_art

    @test_throws ErrorException Solver(n_global=(16, 16, 16), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, art=ArtificialProperties(mu_sensor=:bogus))
    @test_throws ErrorException Solver(n_global=(16, 16, 16), L_domain=(1.0, 1.0, 1.0),
                                       bcs=per3, art=ArtificialProperties(reduction=:bogus))
end

@testset "dt_report agrees with compute_dt and names the limiter" begin
    solver = mkslv(n_global=(16, 16, 16), transport=ConstantTransport(mu0=1e-3))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    r = dt_report(solver, Q)
    @test r.dt ≈ compute_dt(solver, Q) rtol = 1e-12
    @test r.kind in (:acoustic, :diffusive, :curvature)
    @test r.dim in 1:3
    @test all(1 .<= r.index .<= 16)
end

@testset "curvature_rate: collapsed angular dims restrict dt" begin
    # The spherical branch of curvature_rate had no coverage at all, and the
    # whole point of the term is that a COLLAPSED angular dimension carries a
    # stiff geometric source the advective CFL loop never sees. Swirl must
    # therefore shorten dt even though nothing varies in θ or φ.
    mk(metric, uang) = begin
        solver = Solver(n_global=(64, 1, 1), L_domain=(1.0, 1.0, 1.0), metric=metric,
                   origin=(0.5, π / 2, 0.0),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   art=ArtificialProperties(enabled=false))
        Q = allocate_state(solver)
        initialize!(solver, Q, (r, θ, φ) -> Prim(u=(0.0, uang, uang), p=1.0, rho=1.0))
        solver, Q
    end
    for metric in (CylindricalMetric(), SphericalMetric())
        s0, Q0 = mk(metric, 0.0)
        s1, Q1 = mk(metric, 2.0)
        @test compute_dt(s1, Q1) < compute_dt(s0, Q0)
        @test CL.curvature_rate(s1, metric, padded_index(s1, 5, 1, 1), (0.0, 2.0, 2.0)) > 0
        @test CL.curvature_rate(s0, metric, padded_index(s0, 5, 1, 1), (0.0, 0.0, 0.0)) == 0
    end
    # Cartesian has no curvature term at all
    sc = mkslv(n_global=(16, 16, 16))
    @test CL.curvature_rate(sc, CartesianMetric(), padded_index(sc, 2, 2, 2),
                            (1.0, 1.0, 1.0)) == 0
end

@testset "StepControl: floors, positivity, and the default no-op" begin
    # The floor logic is pure, so it is checked directly rather than by
    # constructing a run that fails in each of five ways.
    c = StepControl()
    ok(dt, ρ; seen=1.0, ctl=c) = CL.check_step(ctl, dt, ρ, seen, 1, 0.0, 0.5)
    @test ok(1e-3, 1.0) === nothing
    @test ok(NaN, 1.0).reason === :nonfinite
    @test ok(Inf, 1.0).reason === :nonfinite
    @test ok(1e-3, -1e-9).reason === :negative_density
    @test ok(1e-3, 0.0).reason === :negative_density
    @test ok(1e-50, 1.0).reason === :planck            # below the Planck failsafe
    @test ok(1e-40, 1.0).reason === :dt_collapse       # above it, but collapsed
    # The Planck floor is unconditional: it fires even with every user floor off.
    bare = StepControl(dt_min=0.0, dt_min_ratio=0.0)
    @test ok(1e-50, 1.0; ctl=bare).reason === :planck
    @test ok(1e-40, 1.0; ctl=bare) === nothing         # relative floor disabled
    # An explicit absolute floor.
    @test ok(1e-9, 1.0; ctl=StepControl(dt_min=1e-6)).reason === :dt_min
    # Ordering: a non-finite dt is reported as such, not as a floor breach.
    @test ok(NaN, -1.0).reason === :nonfinite
    # Message carries the state that localizes the failure.
    e = ok(1e-50, 1.0)
    msg = sprint(showerror, e)
    @test occursin("SolverFailure(:planck)", msg)
    @test occursin("StepControl(retries", msg)

    # With prediction off (the default) the chosen step is exactly compute_dt,
    # so adding all of this changed nothing for a healthy run.
    solver = mkslv(n_global=(16, 16, 16))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2sin(x), 0, 0), p=1.0, rho=1.0))
    rate, ρmin = max_rate(solver, Q)
    @test ρmin ≈ 1.0 atol = 1e-12
    @test CL.predicted_dt(solver, StepControl(), rate) == compute_dt(solver, Q)
    # Prediction only ever shortens the step, and only when the rate is rising.
    solver.rate_prev = 0.5 * rate
    @test CL.predicted_dt(solver, StepControl(predict=1.0), rate) < compute_dt(solver, Q)
    solver.rate_prev = 2.0 * rate              # falling rate: no extrapolation
    @test CL.predicted_dt(solver, StepControl(predict=1.0), rate) == compute_dt(solver, Q)
    # Growth capping is relative to the previous accepted step.
    solver.rate_prev = 0.0
    solver.dt_prev = 1e-9
    @test CL.predicted_dt(solver, StepControl(max_growth=1.5), rate) ≈ 1.5e-9 rtol = 1e-14
end

@testset "positivity failsafe: floors, scope, and what each conserves" begin
    @test_throws ArgumentError StepControl(floor_ratio=1.0)
    @test_throws ArgumentError StepControl(floor_ratio=-1e-9)
    @test_throws ArgumentError StepControl(floor_scope=:everything)
    @test StepControl().floor_ratio == 0.0                 # off by default
    @test StepControl().floor_scope === :representable

    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=2.0, u=(0.5, 0.0, 0.0), p=1.0,
                                             Y=(0.25, 0.75)))
    m1, m2, m3 = solver.equations.i_mom
    ie = solver.equations.i_energy
    # The reference state is uniform, so the derived floors are exactly the
    # ratio times the values it carries: e = p / ((γ_m − 1) ρ) with the
    # mass-averaged mixture constants.
    ρ0 = 2.0
    e0 = let Rm = 0.25 * 1.0 + 0.75 * 2.0, cvm = 0.25 * 1.0 / 0.4 + 0.75 * 2.0 / 0.6
        1.0 / (ρ0 * Rm) * cvm
    end
    rho_floor, e_floor = CL.positivity_floors(solver, Q, StepControl(floor_ratio=1e-6))
    @test rho_floor ≈ 1e-6 * ρ0 rtol = 1e-12
    @test e_floor ≈ 1e-6 * e0 rtol = 1e-12
    # Disabled by default, and disabled again when the state supplies no scale.
    @test CL.positivity_floors(solver, Q, StepControl()) == (0.0, 0.0)
    saved = Q[padded_index(solver, 2, 2, 2), 1]
    Q[padded_index(solver, 2, 2, 2), 1] = -3.0
    @test CL.positivity_floors(solver, Q, StepControl(floor_ratio=1e-6)) == (0.0, 0.0)
    Q[padded_index(solver, 2, 2, 2), 1] = saved

    floor!(Qx, scope) = CL.apply_positivity_floor!(solver, Qx, rho_floor, e_floor,
                                                   scope)
    # A healthy state is untouched under either scope, and reports nothing.
    for scope in (:representable, :internal_energy)
        Q1 = copy(Q)
        t = floor!(Q1, scope)
        @test (t.cells, t.low_energy) == (0, 0)
        @test Q1 == Q
    end

    # 1. A negative partial density is clipped and the mixture density is left
    #    exactly where it was, so the repair adds no mass at all.
    I = padded_index(solver, 3, 3, 3)
    Q2 = copy(Q)
    ρ_before = mixture_density(solver, Q2, I)
    Q2[I, 1] = -0.4
    Q2[I, 2] = ρ_before + 0.4
    t = floor!(Q2, :representable)
    @test t.cells == 1
    @test t.mass == 0.0
    @test Q2[I, 1] == 0.0
    @test mixture_density(solver, Q2, I) ≈ ρ_before rtol = 1e-15

    #    An undershoot inside the species band is one the validation accepts,
    #    so the clip leaves it and counts nothing; a zero band clips it.
    Qb = copy(Q)
    Qb[I, 1] = -0.02 * ρ_before
    Qb[I, 2] = 1.02 * ρ_before
    Qb0 = copy(Qb)
    t = floor!(Qb, :representable)
    @test t.cells == 0 && all(iszero, t.species)
    @test Qb == Qb0
    t = CL.apply_positivity_floor!(solver, Qb, rho_floor, e_floor, :representable;
                                   species_band=0.0)
    @test t.cells == 1
    @test Qb[I, 1] == 0.0

    #    With three species an excess of one can be shared between partners
    #    that each stay inside the band. The validation counts it on the upper
    #    side, and the clip returns the point to [0, 1].
    eos3 = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                         IdealSpecies{Float64}("b", 2.0, 1.6),
                         IdealSpecies{Float64}("c", 3.0, 1.3)])
    s3 = mkslv(n_global=(12, 12, 12), eos=eos3)
    Q3s = allocate_state(s3)
    initialize!(s3, Q3s, (x, y, z) -> Prim(rho=2.0, p=1.0, Y=(0.5, 0.25, 0.25)))
    floors3 = CL.positivity_floors(s3, Q3s, StepControl(floor_ratio=1e-6))
    J = padded_index(s3, 3, 3, 3)
    for (Y, counted) in (((1.08, -0.04, -0.04), true), ((1.03, -0.015, -0.015), false))
        Qx = copy(Q3s)
        foreach(sp -> Qx[J, sp] = 2.0 * Y[sp], 1:3)
        @test state_report(s3, Qx).negative_species == counted
        t = CL.apply_positivity_floor!(s3, Qx, floors3..., :representable)
        @test t.cells == counted
        if counted
            @test (Qx[J, 2], Qx[J, 3]) == (0.0, 0.0)
            @test Qx[J, 1] ≈ 2.0 rtol = 1e-15
            @test state_valid(state_report(s3, Qx))
        else
            @test Qx[J, 2] == 2.0 * Y[2]
        end
    end

    # 2. A mixture density below the floor cannot be repaired conservatively, so
    #    the added mass is reported.
    Q3 = copy(Q)
    Q3[I, 1] = 1e-30
    Q3[I, 2] = 1e-30
    t = floor!(Q3, :representable)
    @test t.cells == 1
    @test mixture_density(solver, Q3, I) ≈ rho_floor rtol = 1e-12
    @test t.mass > 0

    # 3. Internal energy below the floor: counted under both scopes, repaired
    #    only under :internal_energy. This is the case the Noh validation run
    #    carries for its whole duration while still reaching the exact plateau.
    Q4 = copy(Q)
    ke = 0.5 * (Q4[I, m1]^2 + Q4[I, m2]^2 + Q4[I, m3]^2) / ρ0
    Q4[I, ie] = ke * 0.9                       # e = −0.1 ke / ρ, E still positive
    Q5 = copy(Q4)
    t = floor!(Q4, :representable)
    @test (t.cells, t.low_energy) == (0, 1)
    @test Q4 == Q5                              # counted, and nothing else
    E_before = Q5[I, ie]
    p_before = sqrt(Q5[I, m1]^2 + Q5[I, m2]^2 + Q5[I, m3]^2)
    t = floor!(Q5, :internal_energy)
    @test (t.cells, t.low_energy) == (1, 1)
    @test Q5[I, ie] == E_before                 # total energy conserved exactly
    @test t.energy == 0.0
    ke5 = 0.5 * (Q5[I, m1]^2 + Q5[I, m2]^2 + Q5[I, m3]^2) / ρ0
    @test (Q5[I, ie] - ke5) / ρ0 ≈ e_floor rtol = 1e-10
    @test t.momentum > 0
    @test sqrt(Q5[I, m1]^2 + Q5[I, m2]^2 + Q5[I, m3]^2) < p_before

    # 4. A negative total energy density is unrepresentable in any frame, so
    #    both scopes repair it, and the branch that does conserves momentum.
    for scope in (:representable, :internal_energy)
        Q6 = copy(Q)
        Q6[I, ie] = -1.0
        p_before = (Q6[I, m1], Q6[I, m2], Q6[I, m3])
        t = floor!(Q6, scope)
        @test (t.cells, t.low_energy) == (1, 1)
        @test (Q6[I, m1], Q6[I, m2], Q6[I, m3]) == p_before
        @test t.momentum == 0.0
        @test t.energy > 0
        ke6 = 0.5 * sum(x -> x^2, p_before) / ρ0
        @test (Q6[I, ie] - ke6) / ρ0 ≈ e_floor rtol = 1e-10
    end
end

@testset "state validity: the report and what each policy mode does" begin
    @test StepControl().validity === :strict
    @test_throws ArgumentError StepControl(validity=:lenient)
    # :repair has nothing to repair with unless the failsafe floors exist.
    @test_throws ArgumentError StepControl(validity=:repair)
    @test StepControl(validity=:repair, floor_ratio=1e-6).validity === :repair

    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=2.0, u=(0.5, 0.0, 0.0), p=1.0,
                                             Y=(0.25, 0.75)))
    m1, m2, m3 = solver.equations.i_mom
    ie = solver.equations.i_energy
    I = padded_index(solver, 3, 3, 3)

    clean = state_report(solver, Q)
    @test state_valid(clean)
    @test clean.points == 12^3
    @test clean.rho_min ≈ 2.0 rtol = 1e-14
    @test clean.nonfinite == 0 && clean.negative_density == 0
    @test occursin("StateReport(1728 points;", sprint(show, clean))

    # Each rejected kind is counted on its own, and a point that cannot carry a
    # density or an energy is not put to the EOS.
    poison(f) = (Qx = copy(Q); f(Qx); state_report(solver, Qx))
    r = poison(Qx -> Qx[I, ie] = NaN)
    @test (r.nonfinite, r.negative_density, r.inadmissible) == (1, 0, 0)
    @test isfinite(r.rho_min)                   # the NaN point is skipped whole
    r = poison(Qx -> (Qx[I, 1] = -3.0; Qx[I, 2] = 0.0))
    @test (r.nonfinite, r.negative_density, r.negative_species) == (0, 1, 0)
    r = poison(Qx -> (Qx[I, 1] = -0.2; Qx[I, 2] = 2.2))
    @test (r.negative_density, r.negative_species) == (0, 1)
    @test state_valid(r) == false
    # A mass fraction of -0.01 is the excursion a species interface carries at
    # any resolution, inside the default band; a zero band rejects it.
    @test StepControl().species_band == 0.05
    @test_throws ArgumentError StepControl(species_band=1.0)
    @test_throws ArgumentError StepControl(species_band=-0.01)
    Qs = copy(Q); Qs[I, 1] = -0.02; Qs[I, 2] = 2.02
    @test state_valid(state_report(solver, Qs))
    @test state_report(solver, Qs; species_band=0.0).negative_species == 1
    @test CL.check_validity(StepControl(species_band=0.0),
                            state_report(solver, Qs; species_band=0.0), "s",
                            1, 0.0, 1e-3, 0.4) isa SolverFailure
    @test validate_state!(solver, Qs; warn=false).negative_species == 0
    @test_throws SolverFailure validate_state!(solver, Qs;
                                               control=StepControl(species_band=0.0))
    # Internal energy below zero is inadmissible for a calorically perfect gas,
    # where e = cv T, and is reported as such rather than by a hardcoded test.
    r = poison(Qx -> Qx[I, ie] = 0.5 * Qx[I, m1]^2 / 2.0 - 1.0)
    @test (r.inadmissible, r.negative_density) == (1, 0)
    @test r.e_min < 0

    # The verdict is the policy, and it is taken from the reduced counts.
    bad = poison(Qx -> Qx[I, 1] = -3.0)
    for mode in (:strict, :repair)
        control = mode === :strict ? StepControl() :
                  StepControl(validity=:repair, floor_ratio=1e-6)
        f = CL.check_validity(control, bad, "the probe state", 7, 0.5, 1e-3, 0.4)
        @test f isa SolverFailure
        @test f.reason === :invalid_state
        @test (f.step, f.t) == (7, 0.5)
        @test occursin("negative density", sprint(showerror, f))
        @test occursin("validity = :permissive", sprint(showerror, f))
    end
    @test CL.check_validity(StepControl(validity=:permissive), bad, "s",
                            1, 0.0, 1e-3, 0.4) === nothing
    @test CL.check_validity(StepControl(), clean, "s", 1, 0.0, 1e-3, 0.4) ===
          nothing

    # validate_state! applies it. Permissive accepts and reports; strict raises.
    Qbad = copy(Q)
    Qbad[I, 1] = -3.0
    @test_throws SolverFailure validate_state!(solver, Qbad)
    permissive = StepControl(validity=:permissive)
    @test (@test_logs (:warn,) match_mode=:any validate_state!(solver, Qbad;
              control=permissive)).negative_density == 1
    @test Qbad[I, 1] == -3.0                    # accepted, never touched
    @test state_valid(validate_state!(solver, Q))

    # Each scope keeps its own postcondition where it acts, which is what makes
    # the residual rejection below a contract rather than a defect.
    rho_floor, e_floor = CL.positivity_floors(solver, Q,
                                              StepControl(floor_ratio=1e-6))
    floor!(Qx, scope) = CL.apply_positivity_floor!(solver, Qx, rho_floor,
                                                   e_floor, scope)
    dens(Qx, J) = sum(Qx[J, sp] for sp in 1:2)
    kin(Qx, J) = (Qx[J, m1]^2 + Qx[J, m2]^2 + Qx[J, m3]^2) / (2 * dens(Qx, J))
    spec(Qx, J) = (Qx[J, ie] - kin(Qx, J)) / dens(Qx, J)

    # Raising a density toward the floor at fixed momentum and total energy
    # increases the internal energy DENSITY, so the substitution never deepens
    # the deficit it leaves behind.
    Qm = copy(Q)
    Qm[I, 1] = rho_floor * 0.3
    Qm[I, 2] = 0.0
    rhoe_before = Qm[I, ie] - kin(Qm, I)
    mom_before = (Qm[I, m1], Qm[I, m2], Qm[I, m3])
    E_before = Qm[I, ie]
    floor!(Qm, :representable)
    @test Qm[I, ie] - kin(Qm, I) > rhoe_before
    @test (Qm[I, m1], Qm[I, m2], Qm[I, m3]) == mom_before
    @test Qm[I, ie] == E_before

    # Where :representable does act on the energy, it lands exactly on the
    # floor and leaves the momentum alone.
    Qr = copy(Q)
    Qr[I, ie] = dens(Qr, I) * e_floor * 0.1     # E < rho * e_floor
    mom_before = (Qr[I, m1], Qr[I, m2], Qr[I, m3])
    floor!(Qr, :representable)
    @test spec(Qr, I) >= e_floor * (1 - 1e-12)
    @test (Qr[I, m1], Qr[I, m2], Qr[I, m3]) == mom_before

    # :internal_energy carries the stronger postcondition on a point
    # :representable declines: the internal energy reaches the floor, the total
    # energy is conserved exactly, and the momentum it removed is tallied.
    Qd = copy(Q)
    Qd[I, ie] = kin(Qd, I) + dens(Qd, I) * e_floor * 0.1
    E_before = Qd[I, ie]
    t_shallow = floor!(copy(Qd), :representable)
    @test t_shallow.low_energy == 1 && t_shallow.cells == 0   # counted, not fixed
    t_deep = floor!(Qd, :internal_energy)
    @test spec(Qd, I) >= e_floor * (1 - 1e-12)
    @test Qd[I, ie] == E_before
    @test t_deep.momentum > 0

    # Repair mode substitutes, reports what it substituted, and then rejects
    # what the substitution did not fix. The two scopes differ in what they
    # promise, and this pins the difference. A nonpositive density carries no
    # recoverable internal energy, so the floor substitutes one; the original
    # momentum at that minimal density leaves a large specific kinetic energy,
    # and the point's total energy still clears rho*e_floor, which is the
    # bound :representable acts on. It therefore counts the point and leaves
    # it, exactly as documented, and strict validation rejects what remains.
    # The repair does not deepen the deficit: raising rho at fixed momentum and
    # total energy increases E - |m|^2/(2rho) monotonically.
    repair = StepControl(validity=:repair, floor_ratio=1e-6)
    floors = CL.positivity_floors(solver, Q, repair)
    @test floors[1] > 0
    Qrep = copy(Q)
    Qrep[I, 1] = -3.0
    Qrep[I, 2] = 0.0
    tally_before = solver.floor_tally.cells
    @test_throws SolverFailure validate_state!(solver, Qrep; control=repair,
                                               floors=floors, warn=false)
    @test solver.floor_tally.cells == tally_before + 1
    @test mixture_density(solver, Qrep, I) ≈ floors[1] rtol = 1e-12
    # Every species change the repairs make is tallied: the clip moves mass
    # between species and the density floor adds `mass`.
    @test sum(solver.floor_tally.species) ≈ solver.floor_tally.mass rtol = 1e-12

    # The scope that converts kinetic energy back into internal energy does
    # fix it, and the state is then accepted.
    deep = StepControl(validity=:repair, floor_ratio=1e-6,
                       floor_scope=:internal_energy)
    Qdeep = copy(Q)
    Qdeep[I, 1] = -3.0
    Qdeep[I, 2] = 0.0
    rep = @test_logs (:warn,) match_mode=:any validate_state!(solver, Qdeep;
              control=deep, floors=floors)
    @test state_valid(rep)
    @test mixture_density(solver, Qdeep, I) ≈ floors[1] rtol = 1e-12
    @test solver.floor_tally.momentum > 0
end

@testset "state validity: a repair before the step renews the prepared state" begin
    # run! measures the rate, which refreshes the primitives the first stage
    # reuses, and then validates. A repair there rewrites points after that
    # refresh, so the step must start from the repaired state as if it had
    # been handed that state: the run below is compared with one whose
    # callback applies the same repair itself.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    control = StepControl(validity=:repair, floor_ratio=1e-6, validity_interval=1)
    ic(x, y, z) = Prim(rho=2.0 + 0.1sin(x), u=(0.5, 0.0, 0.0), p=1.0 + 0.1cos(y),
                       Y=(0.25, 0.75))
    fresh() = begin
        s = mkslv(n_global=(12, 12, 12), eos=eos)
        Qs = allocate_state(s)
        initialize!(s, Qs, ic)
        (s, Qs)
    end
    s0, Q0 = fresh()
    floors = CL.positivity_floors(s0, Q0, control)
    # After the first step, move a fifth of one point's mixture mass from the
    # first species onto the second, outside the species band.
    function poison!(s, Qs)
        s.step == 1 || return false
        I = padded_index(s, 3, 3, 3)
        rho = Qs[I, 1] + Qs[I, 2]
        Qs[I, 1] = -0.2rho
        Qs[I, 2] = 1.2rho
        return false
    end
    repaired!(s, Qs) = (poison!(s, Qs);
                        s.step == 1 && CL.apply_positivity_floor!(s, Qs, floors...,
                                                                  control.floor_scope);
                        false)
    sa, Qa = fresh()
    @test_logs (:warn, r"repaired 1 cell") match_mode=:any run!(sa, Qa;
        tfinal=1.0, nmax=2, callback=poison!, control=control)
    sb, Qb = fresh()
    run!(sb, Qb; tfinal=1.0, nmax=2, callback=repaired!, control=control)
    @test sa.floor_tally.cells == sb.floor_tally.cells + 1
    @test sa.dt_prev == sb.dt_prev
    @test maximum(abs, parent(Qa) .- parent(Qb)) == 0
end

@testset "state validity: initial and returned states are checked" begin
    # An initial condition outside the EOS domain is rejected by setup, where
    # the previous behaviour was to integrate it.
    build(p, control) = begin
        prob = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                       ic=(x, y, z) -> Prim(rho=1.0, p=p))
        setup(prob, Numerics(n_global=(12, 12, 12), control=control))
    end
    @test_throws SolverFailure build(0.0, StepControl())
    solver, Q = build(0.0, StepControl(validity=:permissive))
    @test state_report(solver, Q).inadmissible == 12^3
    solver, Q = build(1.0, StepControl())
    @test state_valid(state_report(solver, Q))

    # The state a run returns is validated on the way out, at each of the three
    # ways a run ends. The step checks read the state ENTERING a step, so
    # without this an nmax, tfinal or callback exit returned its last result
    # uninspected.
    sink!(s, Qs) = (Qs[padded_index(s, 2, 2, 2), 1] = -1.0; false)
    fresh(; control=StepControl()) = begin
        s = mkslv(n_global=(16, 12, 12), control=control)
        Qs = allocate_state(s)
        initialize!(s, Qs, (x, y, z) -> Prim(rho=1.0, p=1.0))
        (s, Qs)
    end
    s, Qs = fresh()
    @test_throws SolverFailure run!(s, Qs; tfinal=1.0, nmax=1, callback=sink!)
    # A callback that ends the run is the third exit, and is checked too.
    stop!(s, Qs) = (Qs[padded_index(s, 2, 2, 2), 1] = -1.0; true)
    s, Qs = fresh()
    @test_throws SolverFailure run!(s, Qs; tfinal=1.0, nmax=99, callback=stop!)
    # Permissive returns it and says so.
    s, Qs = fresh(control=StepControl(validity=:permissive))
    @test_logs (:warn,) match_mode=:any run!(s, Qs; tfinal=1.0, nmax=1,
                                             callback=sink!)
    @test s.step == 1
    @test mixture_density(s, Qs, padded_index(s, 2, 2, 2)) < 0
    @test state_valid(state_report(s, Qs)) == false

    # The rejection reaches the retry path rather than raising past it: with a
    # savepoint and retries available the run rolls back and lowers the CFL,
    # exactly as a step check does. The sink fires only on the first attempt,
    # so the replacement trajectory returns a valid state.
    s, Qs = fresh(control=StepControl(retries=2, savepoint_interval=1))
    fired = Ref(0)
    once!(sv, Qv) = (fired[] += 1; fired[] == 1 &&
                     (Qv[padded_index(sv, 2, 2, 2), 1] = -1.0); false)
    cfl0 = s.cfl
    @test_logs (:warn,) match_mode=:any run!(s, Qs; tfinal=1.0, nmax=2,
                                             callback=once!)
    @test s.cfl < cfl0                          # rolled back, not raised past
    @test state_valid(state_report(s, Qs))

    s, Qs = fresh()
    @test_throws SolverFailure run!(s, Qs; tfinal=1.0, nmax=1,
                                    callback=(sink!, state_guard(s, Qs)))

    # The guard reports under its own control; the solver's has to agree that
    # the state may be returned, since run! validates the result as well.
    s, Qs = fresh(control=StepControl(validity=:permissive))
    guard = CL.StateGuard(s, Qs; control=StepControl(validity=:permissive))
    @test_logs (:warn,) match_mode=:any run!(s, Qs; tfinal=1.0, nmax=1,
        callback=(sink!, Callback(EveryStep(), guard)))
    @test (guard.checks, guard.rejected) == (1, 1)
    @test mixture_density(s, Qs, padded_index(s, 2, 2, 2)) < 0   # accepted, not repaired

    s, Qs = fresh()
    repair = StepControl(validity=:repair, floor_ratio=1e-6)
    guard = CL.StateGuard(s, Qs; control=repair)
    run!(s, Qs; tfinal=1.0, nmax=1, callback=(sink!, Callback(EveryStep(), guard)))
    @test guard.rejected == 0                   # the repair left nothing to reject
    @test s.floor_tally.cells == 1
    @test mixture_density(s, Qs, padded_index(s, 2, 2, 2)) ≈ 1e-6 rtol = 1e-12
end

@testset "run!: failure is raised, and recoverable with retries" begin
    # Noh, three behaviours:
    #
    #   Planar Noh from cfl = 0.9 completes without a retry. Its first step
    #   is sized from the coefficients the initial data produces, because
    #   run! evaluates the right-hand side once before that step; sized on
    #   the acoustic rate alone the same run lost positivity above cfl 0.25,
    #   and the rollback that appeared to recover it rested on the failed
    #   trajectory's coefficients throttling the retry's first step.
    #
    #   Spherical Noh from the warm start at cfl = 0.9 fails abruptly in the
    #   origin's excursion near t = 0.39, representative of a guessed CFL,
    #   and must fail loudly rather than grind.
    #
    #   The same run from cfl = 1.8 with retries rolls back past the
    #   excursion with a halved CFL, twice, and recovers the correct plateau.
    #
    # All arms run the unrelaxed filter (filter_cfl = 0), the configuration
    # they were characterized under.
    γ = 5 / 3; p0 = 1e-4; tfin = 0.6
    build(ν, cfl, control; N, t0) = begin
        metric = ν == 1 ? CartesianMetric() : SphericalMetric()
        lobc = ν == 1 ? SlipWallBC() : OriginBC()
        dom2 = ν == 1 ? (0.0, 1 / N) : (π / 2, π / 2 + 1)
        dom3 = ν == 1 ? (0.0, 1 / N) : (0.0, 1.0)
        inflow = DirichletBC((x, y, z, t) -> begin
            ρ, u, _ = noh_exact(x, isfinite(t) ? t + t0 : t0, ν, γ)
            Prim(rho=ρ, u=(u, 0.0, 0.0), p=p0)
        end)
        w = 4 / N
        ic = (x, y, z) -> begin
            t0 <= 0 && return Prim(rho=1.0, u=(-1.0, 0.0, 0.0), p=p0)
            ρin, _, pin = noh_exact(0.0, t0, ν, γ)
            ρout, _, _ = noh_exact(x, t0, ν, γ)
            θ = tanh_blend(x, (γ - 1) / 2 * t0, w)
            Prim(rho=(1 - θ) * ρin + θ * ρout, u=(-θ, 0.0, 0.0),
                 p=(1 - θ) * pin + θ * p0)
        end
        prob = Problem(eos=IdealSpecies("gas"; gamma=γ, R=1.0),
                       transport=ConstantTransport(mu0=0.0),
                       metric=metric, domain=((0.0, 1.0), dom2, dom3),
                       bcs=((lobc, inflow), per3[2], per3[3]), ic=ic)
        setup(prob, Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=true),
                             cfl=cfl, control=control, filter=StateFilter(; cfl=0.0)))
    end
    # Post-shock plateau, sampled between the wall-heating layer and the shock
    # at x = (γ−1)t/2 = 0.2 — a window that straddles the shock would average
    # the answer with the undisturbed inflow and pass for the wrong reason.
    plateau(s, Q, N) = begin
        CL.exchange_state!(Q, s.decomp); CL.primitives!(s, Q)
        core = [i for i in 1:N if 0.06 <= xcoord(s, 1, i) <= 0.14]
        sum(s.rho[padded_index(s, i, 1, 1)] for i in core) / length(core)
    end

    # Permissive on the returned state: a Noh run ends with a handful of
    # negative-internal-energy cells ahead of the front at any CFL, so a
    # strict exit check would reject a correct trajectory.
    s1, Q1 = build(1, 0.9, StepControl(validity=:permissive); N=400, t0=0.0)
    run!(s1, Q1; tfinal=tfin, nmax=20_000)
    @test s1.t ≈ tfin rtol = 1e-9
    @test s1.cfl == 0.9                        # no retry was needed
    @test plateau(s1, Q1, 400) ≈ 4.0 rtol = 0.05  # exact Noh plateau for nu = 1

    s2, Q2 = build(3, 0.9, StepControl(); N=256, t0=0.3)
    err = nothing
    try
        run!(s2, Q2; tfinal=tfin - 0.3, nmax=20_000)
    catch e
        err = e
    end
    @test err isa SolverFailure
    @test err.reason in (:negative_density, :dt_collapse)
    @test s2.step < 20_000                     # it stopped early, it did not grind

    s3, Q3 = build(3, 1.8, StepControl(retries=5, savepoint_interval=20,
                                       validity=:permissive); N=256, t0=0.3)
    run!(s3, Q3; tfinal=tfin - 0.3, nmax=20_000)
    @test s3.t ≈ tfin - 0.3 rtol = 1e-9
    @test s3.cfl < 1.8                         # it backed off, and says by how much
    @test plateau(s3, Q3, 256) ≈ 64.0 rtol = 0.05  # exact Noh plateau for nu = 3
    # The backoff compounds: successive retries must keep halving, not keep
    # re-applying one factor to the same starting CFL.
    @test s3.cfl <= 1.8 * 0.25 + 1e-12
end

@testset "NASA-9 mixture reduces exactly to the ideal mixture" begin
    @test_throws ArgumentError IdealMixture(IdealSpecies{Float64}[])
    @test_throws ArgumentError Nasa9Mixture(Nasa9Species{Float64}[])
    # The strongest available check on the polynomial machinery: with only the
    # constant term a3 populated, cp is temperature-independent and every
    # quantity — including the Newton inversion of e(T) — must reproduce the
    # closed-form ideal-gas answer to round-off. A transcription error in the
    # enthalpy integral, the cv, or the sound speed shows up here. The
    # temperatures are nondimensional and lie below the record's 200 K edge, so
    # the polynomial is evaluated there by stating `:polynomial`.
    γ, R = 1.4, 1.0
    cp = γ * R / (γ - 1)
    ideal = IdealSpecies("gas"; gamma=γ, R=R)
    poly = Nasa9Mixture([nasa9_constant_cp("gas", R, cp)]; extrapolate=:polynomial)
    @test nspecies(poly) == 1
    for T in (0.3, 1.0, 7.5, 300.0)
        @test CL.species_cp(poly, 1, T) ≈ cp rtol = 1e-14
        @test CL.species_enthalpy(poly, 1, T) ≈ cp * T rtol = 1e-14
        @test CL.species_energy(poly, 1, T) ≈ (cp - R) * T rtol = 1e-14
    end
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7)
    qi = conserved_from_prim(ideal, pr)
    qp = conserved_from_prim(poly, pr)
    @test all(qi .≈ qp)
    # ... and through the full primitives path, which is where the Newton
    # inversion actually runs.
    for eos in (ideal, poly)
        solver = mkslv(n_global=(12, 12, 12), eos=eos)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=0.8, T_ion=1.7))
        CL.exchange_state!(Q, solver.decomp)
        CL.primitives!(solver, Q)
        I = padded_index(solver, 3, 4, 5)
        @test solver.p[I] ≈ 0.8 rtol = 1e-12
        @test solver.T_ion[I] ≈ 1.7 rtol = 1e-12
        @test solver.c[I] ≈ sqrt(γ * R * 1.7) rtol = 1e-12
        @test solver.cp_mix[I] ≈ cp rtol = 1e-12
    end
end

@testset "NASA-9: thermodynamic consistency of a varying cp" begin
    # A genuinely temperature-dependent coefficient set, checked against the
    # two identities that must hold whatever the coefficients are:
    # dh/dT = cp, and the Newton inversion is the inverse of e(T).
    R = 287.0
    sp = Nasa9Species{Float64}(name="fake", R=R,
                               a=(1.2e4, -50.0, 3.6, 6.0e-4, -1.0e-7, 1.0e-11,
                                  -4.0e-16), b1=-1.0e3)
    eos = Nasa9Mixture([sp, nasa9_constant_cp("inert", 200.0, 900.0)];
                       T_guess=500.0)
    for T in (250.0, 800.0, 2500.0)
        δ = 1e-4 * T
        dh = (CL.species_enthalpy(eos, 1, T + δ) -
              CL.species_enthalpy(eos, 1, T - δ)) / 2δ
        @test dh ≈ CL.species_cp(eos, 1, T) rtol = 1e-7
    end
    # cp really does vary — otherwise the identity above is vacuous — and cv
    # stays positive across the range, which the Newton solve relies on.
    @test CL.species_cp(eos, 1, 2500.0) / CL.species_cp(eos, 1, 300.0) > 1.2
    @test all(CL.species_cp(eos, 1, T) > R for T in 250.0:50.0:2500.0)
    # e(T) round trip through the Newton solve, at three compositions.
    for Y in ((1.0, 0.0), (0.5, 0.5), (0.2, 0.8))
        for T in (250.0, 800.0, 2500.0)
            e = sum(Y[k] * CL.species_energy(eos, k, T) for k in 1:2)
            @test CL.mixture_temperature(eos, e, k -> Y[k]) ≈ T rtol = 1e-12
        end
    end
    # And end to end: a state initialized from (p, T_ion) recovers both.
    solver = mkslv(n_global=(12, 12, 12), eos=eos)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.35, 0.65), u=(120.0, 0, 0),
                                             p=2.5e5, T_ion=1400.0))
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    I = padded_index(solver, 3, 4, 5)
    @test solver.p[I] ≈ 2.5e5 rtol = 1e-10
    @test solver.T_ion[I] ≈ 1400.0 rtol = 1e-10
    @test solver.Y[2][I] ≈ 0.65 rtol = 1e-12
    # A step must stay finite: the Newton solve runs inside every RK stage.
    run!(solver, Q; tfinal=1e9, nmax=3)
    @test all(isfinite, Q)
end

@testset "NASA-9: what the temperature recovery reports about itself" begin
    # Every check runs in both precisions, since the criterion is written from
    # eps of the coefficient type and a Float32 mechanism must behave the same
    # way rather than merely not crash.
    for T in (Float64, Float32)
        # A recovery inside the fitted range succeeds and says so.
        mix = Nasa9Mixture(T, read_nasa9(["He", "CO2"]))
        Y = (T(0.3), T(0.7))
        for Tref in T.((500.0, 2000.0, 4000.0, 8000.0))
            e = sum(Y[k] * CL.species_energy(mix, k, Tref) for k in 1:2)
            Trec, status = CL.mixture_temperature_status(mix, e, k -> Y[k])
            @test status == CL.TEMPERATURE_OK
            @test Trec ≈ Tref rtol = 256eps(T)
            @test CL.mixture_temperature(mix, e, k -> Y[k]) === Trec
        end
        # An interval join is approached from both sides. The fits are only
        # continuous at a join to the tolerance the reader enforces, so the
        # inversion is accurate there to that continuity and not to eps.
        for Tjoin in T.((1000.0, 6000.0)), δ in T.((-1e-3, 1e-3))
            Tref = Tjoin * (1 + δ)
            e = sum(Y[k] * CL.species_energy(mix, k, Tref) for k in 1:2)
            Trec, status = CL.mixture_temperature_status(mix, e, k -> Y[k])
            @test status == CL.TEMPERATURE_OK
            @test Trec ≈ Tref rtol = 1e-4
        end
        # He is fitted from 300 K up, so an ambient state below that is an
        # extrapolation. It recovers accurately and is reported as extrapolated.
        e_cold = sum(Y[k] * CL.species_energy(mix, k, T(250)) for k in 1:2)
        Tcold, status = CL.mixture_temperature_status(mix, e_cold, k -> Y[k])
        @test status & CL.TEMPERATURE_OUT_OF_RANGE != 0
        @test status & CL.TEMPERATURE_NOT_CONVERGED == 0
        @test Tcold ≈ 250 rtol = 1e-5
        # Temperature extremes: no root in the search range, diagnosed, finite,
        # and returned rather than raised, because this runs per point.
        for e_extreme in T.((-1e12, 1e30))
            Text, status = CL.mixture_temperature_status(mix, e_extreme, k -> Y[k])
            @test status & CL.TEMPERATURE_NOT_CONVERGED != 0
            @test isfinite(Text) && Text > 0
        end
        # An invalid composition leaves the mixture cv nonpositive, so there is
        # no bracket and no unique root to converge to.
        _, status = CL.mixture_temperature_status(mix, T(1e5), _ -> zero(T))
        @test status & CL.TEMPERATURE_NO_BRACKET != 0
        @test status & CL.TEMPERATURE_NOT_CONVERGED != 0
        # The failure reaches the state validation as the EOS's own verdict.
        @test state_admissibility(mix, T(1), T(1e5), _ -> zero(T), 2) &
              CL.STATE_UNRECOVERABLE != 0
        @test state_admissibility(mix, T(1), e_cold, k -> Y[k], 2) &
              CL.STATE_EXTRAPOLATED != 0
        # Extrapolating is not by itself a rejection: the polynomial and
        # tangent policies say the extension is acceptable, so the point is
        # reported and carried. `:missing` says the fit is undefined there and
        # marks the same point inadmissible, which is what strict validation
        # rejects.
        for policy in (:polynomial, :linear)
            m = Nasa9Mixture(["He", "CO2"]; extrapolate=policy)
            f = state_admissibility(m, T(1), e_cold, k -> Y[k], 2)
            @test f & CL.STATE_EXTRAPOLATED != 0
            @test f & CL.STATE_INADMISSIBLE == 0
        end
        gone = Nasa9Mixture(["He", "CO2"]; extrapolate=:missing)
        f = state_admissibility(gone, T(1), e_cold, k -> Y[k], 2)
        @test f & CL.STATE_EXTRAPOLATED != 0
        @test f & CL.STATE_INADMISSIBLE != 0
        # Inside the range the setting changes nothing.
        e_in = sum(Y[k] * CL.species_energy(gone, k, T(1000)) for k in 1:2)
        @test state_admissibility(gone, T(1), e_in, k -> Y[k], 2) == CL.STATE_OK
        @test state_admissibility(mix, T(1),
                                  sum(Y[k] * CL.species_energy(mix, k, T(1000))
                                      for k in 1:2), k -> Y[k], 2) == CL.STATE_OK
    end

    # The extrapolation policy is a choice, and it is inert inside the range.
    poly = Nasa9Mixture(["CO2"]; extrapolate=:polynomial)
    lin = Nasa9Mixture(["CO2"]; extrapolate=:linear)
    @test_throws ArgumentError Nasa9Mixture(["CO2"]; extrapolate=:constant)
    @test poly.extrapolate === :polynomial
    @test Nasa9Mixture(["CO2"]).extrapolate === :linear
    @test Nasa9Mixture(Float32, ["CO2"]).extrapolate === :linear
    @test Nasa9Mixture(read_nasa9(["CO2"])).extrapolate === :linear
    # `:missing` evaluates as the tangent extension does, so a step in progress
    # completes and the offending state can be read back; the difference is the
    # verdict it carries, not the number it returns.
    gone1 = Nasa9Mixture(["CO2"]; extrapolate=:missing)
    for Tq in (250.0, 1000.0, 6000.0, 20000.0)
        @test CL.species_cp(gone1, 1, Tq) === CL.species_cp(lin, 1, Tq)
    end
    for Tq in (250.0, 1000.0, 6000.0, 20000.0)
        @test CL.species_cp(poly, 1, Tq) === CL.species_cp(lin, 1, Tq)
        @test CL.species_enthalpy(poly, 1, Tq) === CL.species_enthalpy(lin, 1, Tq)
    end
    # Outside it, the degree-four fit is not a model of anything: run past
    # 20000 K its energy is not monotone in T, and the inversion finds another
    # temperature of the same energy, thousands of kelvin from the one the
    # energy was formed at, while the tangent extension recovers that one. Both
    # report that the answer came from outside the data.
    e_hot = CL.species_energy(poly, 1, 30000.0)
    Tpoly, spoly = CL.mixture_temperature_status(poly, e_hot, _ -> 1.0)
    Tlin, slin = CL.mixture_temperature_status(lin, CL.species_energy(lin, 1,
                                                                     30000.0),
                                               _ -> 1.0)
    @test spoly & CL.TEMPERATURE_OUT_OF_RANGE != 0
    @test slin & CL.TEMPERATURE_OUT_OF_RANGE != 0
    @test abs(Tpoly - 30000.0) > 1000
    @test Tlin ≈ 30000.0 rtol = 1e-12
    @test CL.species_cp(lin, 1, 40000.0) === CL.species_cp(lin, 1, 20000.0)
    # A hot state inside the range recovers under `:polynomial` as well. Its seed lands
    # past 20000 K, where the polynomial CO2 fit has a negative cv; an iterate
    # there would end the search with no bracket and a temperature twice the
    # root's. The iterate is held on the edge of the fitted range instead.
    for T in (Float64, Float32)
        co2 = Nasa9Mixture(T, ["CO2"]; extrapolate=:polynomial)
        for Tref in T.((13000, 15000, 17500, 19900))
            e = CL.species_energy(co2, 1, Tref)
            Trec, status = CL.mixture_temperature_status(co2, e, _ -> one(T))
            @test status == CL.TEMPERATURE_OK
            @test Trec ≈ Tref rtol = 1e-4
        end
    end

    # The inversion and its status run per point inside the primitives pass, so
    # neither the status nor the bracket may allocate. Measured on the pass
    # itself rather than on a loop over the routine, since that is the path and
    # a hand-written loop measures its own closure instead.
    # The transverse directions are collapsed so the pass runs serially at any
    # thread count: a threaded region allocates per region per thread, which
    # would measure the launcher rather than the inversion.
    mix = Nasa9Mixture(["He", "CO2"])
    solver = mkslv(n_global=(12, 1, 1), eos=mix)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.3, 0.7), p=1e5, T_ion=1000.0))
    CL.primitives!(solver, Q)
    @test (@allocated CL.primitives!(solver, Q)) == 0
    @test solver.T_ion[padded_index(solver, 3, 1, 1)] ≈ 1000.0 rtol = 1e-10
end

@testset "NASA-9: leaving the fitted range ends a run only under :missing" begin
    # The per-point verdicts above, reached through run!, which validates the
    # state entering the call under the solver's policy. He is fitted from
    # 300 K, so a 250 K state is an extrapolation at every point.
    ic = (x, y, z) -> Prim(Y=(0.3, 0.7), p=1e5, T_ion=250.0)
    build(policy; kw...) = begin
        s = mkslv(n_global=(12, 1, 1),
                  eos=Nasa9Mixture(["He", "CO2"]; extrapolate=policy); kw...)
        Qs = allocate_state(s)
        initialize!(s, Qs, ic)
        (s, Qs)
    end
    for policy in (:polynomial, :linear)
        s, Qs = build(policy)
        run!(s, Qs; tfinal=1.0, nmax=1)
        r = state_report(s, Qs)
        @test s.step == 1
        @test (r.extrapolated, r.inadmissible) == (12, 0)
        @test state_valid(r)
    end
    s, Qs = build(:missing)
    err = try run!(s, Qs; tfinal=1.0, nmax=1); nothing catch e; e end
    @test err isa SolverFailure && err.reason === :invalid_state
    @test s.step == 0
    s, Qs = build(:missing; control=StepControl(validity=:permissive))
    @test_logs (:warn,) match_mode=:any run!(s, Qs; tfinal=1.0, nmax=1)
    r = state_report(s, Qs)
    @test s.step == 1
    @test (r.extrapolated, r.inadmissible) == (12, 12)
    @test state_valid(r) == false
end

@testset "NASA CEA reader: intervals, molar mass, and energy reference" begin
    he, co2 = read_nasa9(["He", "CO2"])
    @test (he.name, co2.name) == ("He", "CO2")
    @test co2.R ≈ 8.31446261815324 / 44.0095e-3 rtol = 1e-14
    @test length(co2.intervals) == 3
    @test [(item.Tmin, item.Tmax) for item in co2.intervals] ==
          [(200.0, 1000.0), (1000.0, 6000.0), (6000.0, 20000.0)]

    sensible = Nasa9Mixture([co2])
    @test CL.species_enthalpy(sensible, 1, 298.15) ≈ 0.0 atol = 1e-6
    @test CL.species_cp(sensible, 1, 300.0) ≈ 845.7241586606974 rtol = 1e-13
    for i in 1:2
        T_join = co2.intervals[i].Tmax
        left, right = co2.intervals[i], co2.intervals[i + 1]
        @test CL._nasa9_cp_over_R(left, T_join) ≈
              CL._nasa9_cp_over_R(right, T_join) rtol = 5e-7
        @test CL._nasa9_h_over_R(left, T_join) ≈
              CL._nasa9_h_over_R(right, T_join) rtol = 5e-7
    end

    co2_formation = read_nasa9("CO2"; reference=:formation)
    formation = Nasa9Mixture([co2_formation])
    h298_molar = CL.species_enthalpy(formation, 1, 298.15) * 44.0095e-3
    @test h298_molar ≈ -393510.0 atol = 5.0
    @test CL.species_cp(formation, 1, 300.0) == CL.species_cp(sensible, 1, 300.0)
    for eos in (sensible, formation), T in (220.0, 300.0, 999.0, 1001.0,
                                                    1500.0, 5999.0, 6001.0, 10000.0)
        e = CL.species_energy(eos, 1, T)
        @test CL.mixture_temperature(eos, e, _ -> 1.0) ≈ T rtol = 2e-13
    end

    Y = (0.3, 0.7)
    for reference in (:sensible, :formation)
        mixture = Nasa9Mixture(read_nasa9(["He", "CO2"]; reference))
        for T in (300.0, 1400.0, 7000.0)
            e = sum(Y[k] * CL.species_energy(mixture, k, T) for k in 1:2)
            @test CL.mixture_temperature(mixture, e, k -> Y[k]) ≈ T rtol = 2e-13
        end
    end
    # The late Air record exercises the CEA file's product/reactant separators.
    @test read_nasa9("Air").name == "Air"
    @test CL.species_names(Nasa9Mixture(read_nasa9(["CO2", "He"]))) == ["CO2", "He"]
    @test_throws ArgumentError read_nasa9("CO2"; reference=:unknown)

    # Zero-interval records list a heat of formation and no fit, and are three
    # lines, not two. Miscounting them desynchronizes the scan against
    # the file without a local failure, so pin both the record that has to be
    # rejected and a real species that only parses if the skip is right.
    @test_throws "no temperature intervals" read_nasa9("n-Butanol")
    @test length(read_nasa9("Jet-A(g)").intervals) == 2
    # Assert the message because a desynchronized scan also throws ArgumentError
    # after misreading a coefficient line as a species header.
    @test_throws "NASA CEA species not found" read_nasa9("not-a-CEA-species")

    # The two CEA records whose interval joins are loosest; the tolerance is
    # sized to admit them and still reject a mistranscribed coefficient.
    @test all(nspecies(Nasa9Mixture(read_nasa9([name]))) == 1
              for name in ("ALOCL", "SnCL2"))
    good = Nasa9Interval{Float64}(200.0, 1000.0,
                                  (0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0), 0.0)
    bad = Nasa9Interval{Float64}(1000.0, 6000.0,
                                 (0.0, 0.0, 4.5, 0.0, 0.0, 0.0, 0.0), 0.0)
    @test_throws ArgumentError Nasa9Species{Float64}("broken", 287.0, [good, bad])
end

@testset "StiffenedGas: perfect-gas limit, and a real liquid" begin
    # p_inf = 0 must reproduce a perfect gas exactly, which pins the algebra;
    # then a water-like parameter set exercises the nonzero-p_inf branch.
    γ = 1.4; R = 1.0; cv = R / (γ - 1)
    sg = StiffenedGas(gamma=γ, p_inf=0.0, cv=cv, name="gas")
    @test nspecies(sg) == 1
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7)
    @test all(conserved_from_prim(IdealSpecies("gas"; gamma=γ, R=R), pr) .≈
              conserved_from_prim(sg, pr))
    solver = mkslv(n_global=(12, 12, 12), eos=sg)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> pr)
    CL.exchange_state!(Q, solver.decomp); CL.primitives!(solver, Q)
    I = padded_index(solver, 3, 4, 5)
    @test solver.p[I] ≈ 0.8 rtol = 1e-12
    @test solver.T_ion[I] ≈ 1.7 rtol = 1e-12
    @test solver.c[I] ≈ sqrt(γ * R * 1.7) rtol = 1e-12
    @test CL.eos_phi(sg, 1.0, 0.8, 1.7, γ * cv) ≈ cv / R rtol = 1e-13

    # Water: γ = 4.4, p∞ = 6e8 Pa. The point of the model is that the sound
    # speed is set by p∞, not by p, so it stays near 1500 m/s at 1 atm where a
    # perfect gas would give a few hundred.
    water = StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0, name="water")
    s2 = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0), bcs=per3,
                eos=water, art=ArtificialProperties(enabled=false))
    Q2 = allocate_state(s2)
    initialize!(s2, Q2, (x, y, z) -> Prim(u=(0, 0, 0), p=101325.0, rho=1000.0))
    CL.exchange_state!(Q2, s2.decomp); CL.primitives!(s2, Q2)
    J = padded_index(s2, 3, 4, 5)
    @test s2.p[J] ≈ 101325.0 rtol = 1e-9
    @test 1400 < s2.c[J] < 1700                      # c = sqrt(γ(p+p∞)/ρ)
    @test s2.c[J] ≈ sqrt(4.4 * (101325.0 + 6.0e8) / 1000.0) rtol = 1e-12
    # Uniform state ⇒ zero RHS, the same freestream statement made for every
    # other configuration in this suite.
    apply_bcs!(s2, Q2)
    dQ2 = zero(Q2)
    compute_rhs!(s2, Q2, dQ2)
    @test maximum(abs, dQ2) < 1e-8 * 101325.0
    # An acoustic pulse stays finite and does not leave the stiffened branch.
    initialize!(s2, Q2, (x, y, z) ->
        Prim(u=(0, 0, 0), p=101325.0 * (1 + 0.01sin(2π * x)), rho=1000.0))
    run!(s2, Q2; tfinal=1e9, nmax=5)
    CL.primitives!(s2, Q2)
    @test all(isfinite, Q2)
    @test minimum(s2.rho[padded_index(s2, i, j, k)] for i in 1:12, j in 1:12, k in 1:12) > 0
end

@testset "line_sample: one grid line, distinct from the plane average" begin
    # A density linear in all three coordinates separates the two extractions:
    # a sample along x at (j, k) carries 0.2 y_j + 0.3 z_k, the plane average
    # carries the transverse means instead, and every value is definitional.
    ic(x, y, z) = Prim(p=1.0, rho=1 + 0.1x + 0.2y + 0.3z)
    rho_fn(x, y, z) = 1 + 0.1x + 0.2y + 0.3z
    solver = Solver(n_global=(24, 16, 12), L_domain=(2.0, 1.0, 0.5), bcs=per3,
                    art=ArtificialProperties(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, ic)
    gx(d, g) = global_xcoord(solver, d, g)

    coord, value = line_sample(solver, Q, :rho; dim=1, index=(7, 4))
    @test coord == [gx(1, g) for g in 1:24]
    @test maximum(abs, value .- [rho_fn(gx(1, g), gx(2, 7), gx(3, 4)) for g in 1:24]) < 1e-14
    coord, value = line_sample(solver, Q, :rho; dim=2, index=(3, 11))
    @test coord == [gx(2, g) for g in 1:16]
    @test maximum(abs, value .- [rho_fn(gx(1, 3), gx(2, g), gx(3, 11)) for g in 1:16]) < 1e-14
    coord, value = line_sample(solver, Q, :rho; dim=3, index=(24, 16))
    @test coord == [gx(3, g) for g in 1:12]
    @test maximum(abs, value .- [rho_fn(gx(1, 24), gx(2, 16), gx(3, g)) for g in 1:12]) < 1e-14

    # The default line is (i, 1, 1); `at` snaps each transverse coordinate to
    # its nearest node, here (7, 4) again from positions a third of a cell off.
    _, default_line = line_sample(solver, Q, :rho)
    _, first_line = line_sample(solver, Q, :rho; index=(1, 1))
    @test default_line == first_line
    _, snapped = line_sample(solver, Q, :rho;
                             at=(gx(2, 7) + solver.h[2] / 3, gx(3, 4) - solver.h[3] / 3))
    _, indexed = line_sample(solver, Q, :rho; index=(7, 4))
    @test snapped == indexed

    # The plane average is a different quantity when a transverse dimension is
    # resolved, and the same one when both are collapsed.
    _, mean_line = line_profile(solver, Q, :rho; dim=1)
    @test maximum(abs, mean_line .- first_line) > 0.1
    solver1 = Solver(n_global=(32, 1, 1), L_domain=(2.0, 1.0, 1.0), bcs=per3,
                     art=ArtificialProperties(enabled=false))
    Q1 = allocate_state(solver1)
    initialize!(solver1, Q1, ic)
    _, sample1 = line_sample(solver1, Q1, :rho)
    _, mean1 = line_profile(solver1, Q1, :rho)
    @test maximum(abs, sample1 .- mean1) < 1e-14

    @test_throws ArgumentError line_sample(solver, Q, :rho; dim=4)
    @test_throws ArgumentError line_sample(solver, Q, :rho; index=(0, 1))
    @test_throws ArgumentError line_sample(solver, Q, :rho; dim=1, index=(1, 13))
    @test_throws ArgumentError line_sample(solver, Q, :rho; index=(1, 1), at=(0.0, 0.0))
end

@testset "diagnostics: quadrature, plane averages, mixing measures" begin
    # The quadrature is the load-bearing part: every mixing number is a ratio of
    # two of these integrals, so a wrong edge weight biases θ and W silently
    # rather than failing. Cartesian must be exact, and so must a constant over
    # a linear or quadratic Jacobian once the metric's share of each edge term
    # is removed (see the note in diagnostics.jl).
    solver = Solver(n_global=(24, 16, 12), L_domain=(2.0, 1.0, 0.5),
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                    art=ArtificialProperties(enabled=false))
    @test domain_volume(solver) ≈ 1.0 atol = 1e-12          # 2.0 × 1.0 × 0.5
    ones_f = CL.field(solver.decomp); fill!(ones_f, 1.0)
    @test volume_integral(solver, ones_f) ≈ 1.0 atol = 1e-12
    @test volume_average(solver, ones_f) ≈ 1.0 atol = 1e-12
    # ∫x dV over x ∈ [0,2] with unit transverse area = 1.0 (trapezoid is exact
    # for a linear integrand, as provided by the half edge weight).
    lin = CL.field(solver.decomp)
    fillf!(solver, lin, (x, y, z) -> x)
    @test volume_integral(solver, lin) ≈ 1.0 atol = 1e-12
    # A plane average of a function of x alone returns that function, and the
    # spacing profile sums to the extent.
    prof = plane_profile(solver, lin, 1)
    xs = profile_coordinate(solver, 1)
    @test length(prof) == 24
    @test maximum(abs, prof .- xs) < 1e-12
    @test sum(profile_spacing(solver, 1)) ≈ 2.0 atol = 1e-12

    # Cylindrical with the axis fold, spherical with the origin fold. The
    # midpoint rule at the axis and the trapezoid rule at the outer wall err by
    # h²/24 and h²/12 on ∫r dr; the axis node's weight, 11/12 of the
    # midpoint's, and the wall node's remove both, so the linear J = r and the
    # quadratic J = r² integrate exactly. The origin has no such term.
    rz(N) = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   metric=CylindricalMetric(),
                   bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                   art=ArtificialProperties(enabled=false))
    cyl = Solver(n_global=(64, 1, 12), L_domain=(1.0, 1.0, 1.0),
                 metric=CylindricalMetric(),
                 bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                 art=ArtificialProperties(enabled=false))
    ones_c = CL.field(cyl.decomp); fill!(ones_c, 1.0)
    @test volume_integral(cyl, ones_c) ≈ 0.5 atol = 1e-14  # ∫r dr dθ dz, θ collapsed
    @test CL._edge_factor(cyl, 1, 1, padded_index(cyl, 1, 1, 1)) ≈ 11 / 12
    sph = Solver(n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0),
                 metric=SphericalMetric(), origin=(0.0, π / 2 - 0.5, 0.0),
                 bcs=((OriginBC(), SlipWallBC()), per3[2], per3[3]),
                 art=ArtificialProperties(enabled=false))
    @test CL._edge_factor(sph, 1, 1, padded_index(sph, 1, 1, 1)) == 1.0
    @test domain_volume(sph) ≈ sin(xcoord(sph, 2, 1)) / 3 atol = 1e-14
    # A field even across the axis whose derivative vanishes at the wall
    # converges at fourth order: ∫ r (2 + cos πr) dr = 1 − 2/π².
    function rz_error(N)
        s = rz(N)
        f = fillf!(s, CL.field(s.decomp), (r, θ, z) -> 2 + cos(π * r))
        return abs(volume_integral(s, f) - (1 - 2 / π^2))
    end
    @test rz_error(32) / rz_error(64) > 12

    # Mixing measures against states whose answers are definitional.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 1.0, 1.4)])
    mk(ic) = begin
        s = Solver(n_global=(64, 12, 12), L_domain=(1.0, 0.2, 0.2), bcs=per3,
                   eos=eos, art=ArtificialProperties(enabled=false))
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        CL.exchange_state!(Q, s.decomp); CL.primitives!(s, Q)
        s, Q
    end
    # The pair W and θ exists precisely to separate stirring from mixing, and
    # these three states are the demonstration: the first two have IDENTICAL
    # mix width and opposite θ.
    #
    # (a) uniformly mixed everywhere: W is the full extent, θ = 1.
    s1, Q1 = mk((x, y, z) -> Prim(Y=(0.5, 0.5), p=1.0, rho=1.0))
    @test mix_width(s1, Q1) ≈ 1.0 atol = 1e-12
    @test molecular_mixing(s1, Q1) ≈ 1.0 atol = 1e-12
    # (b) stirred but not mixed: each plane is half pure a and half pure b, so
    # ⟨Y_a⟩⟨Y_b⟩ is unchanged from (a) but ⟨Y_a Y_b⟩ vanishes pointwise.
    s2, Q2 = mk((x, y, z) -> Prim(Y=(y < 0.1 ? 1.0 : 0.0, y < 0.1 ? 0.0 : 1.0),
                                  p=1.0, rho=1.0))
    @test mix_width(s2, Q2) ≈ 1.0 atol = 1e-12
    @test molecular_mixing(s2, Q2) < 1e-12
    # (c) an x-only interface: nothing varies within a plane, so θ is 1 by
    # construction and W collapses onto the interface.
    s3, Q3 = mk((x, y, z) -> begin
        θ = tanh_blend(x, 0.5, 1 / 64)
        Prim(Y=(1 - θ, θ), p=1.0, rho=1.0)
    end)
    @test mix_width(s3, Q3) < 0.05
    @test molecular_mixing(s3, Q3) ≈ 1.0 atol = 1e-12
    # The PDF of a segregated field piles up at 0 and 1 and integrates to 1.
    centers, pdf = species_pdf(s2, 1; nbins=20)
    @test sum(pdf) * (centers[2] - centers[1]) ≈ 1.0 atol = 1e-10
    @test pdf[1] + pdf[end] > 0.99 * sum(pdf)
    _, pdf1 = species_pdf(s1, 1; nbins=20)
    @test pdf1[11] > 0.99 * sum(pdf1)                # all mass in the Y=0.5 bin

    # TKE removes the plane mean, so a uniform stream carries none; a
    # transverse-varying velocity does.
    s4, Q4 = mk((x, y, z) -> Prim(Y=(1.0, 0.0), u=(0.7, 0, 0), p=1.0, rho=1.0))
    @test turbulent_kinetic_energy(s4, Q4) < 1e-20
    @test maximum(abs, tke_profile(s4, Q4)) < 1e-20
    s5, Q5 = mk((x, y, z) -> Prim(Y=(1.0, 0.0), u=(0.7, 0.3sin(2π * y / 0.2), 0),
                                  p=1.0, rho=1.0))
    @test turbulent_kinetic_energy(s5, Q5) > 1e-3
    # Dissipation is a sink: zero for a uniform state, positive under shear.
    @test abs(dissipation_rate(s4, Q4)) < 1e-20
    s6, Q6 = mk((x, y, z) -> Prim(Y=(1.0, 0.0), u=(0.0, 0.3sin(2π * x), 0),
                                  p=1.0, rho=1.0))
    s6.transport = ConstantTransport(mu0=1e-2)
    @test dissipation_rate(s6, Q6) > 0
end

@testset "save_vtk writes a readable .pvtr/.vtr pair" begin
    solver = mkslv(n_global=(12, 12, 12))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    save_vtk(solver, Q, "test_vtk")
    files = filter(startswith("test_vtk"), readdir())
    @test any(endswith(".pvtr"), files)
    @test any(endswith(".vtr"), files)
    @test all(f -> filesize(f) > 0, files)
    txt = read(first(filter(endswith(".pvtr"), files)), String)
    @test occursin("PRectilinearGrid", txt)
    foreach(rm, files)
end

# Walk the appended-data blocks of a .vtr/.vts piece and return the Float32
# point-data arrays by name. Each block is a UInt64 byte count followed by its
# payload, so this also verifies that the offsets and lengths are consistent. A
# writer that miscounted desynchronizes here rather than producing a plausible
# file that only ParaView rejects.
function read_vtk_pointdata(path)
    raw = read(path)
    marker = Vector{UInt8}("<AppendedData encoding=\"raw\">\n_")
    i0 = findfirst(marker, raw)[end]
    header = String(copy(raw[1:i0]))
    decls = [(m.captures[1], parse(Int, m.captures[2])) for m in eachmatch(
        r"<DataArray type=\"Float32\" Name=\"([^\"]+)\" NumberOfComponents=\"(\d+)\"",
        header)]
    blocks = Vector{Vector{UInt8}}()
    pos = i0 + 1
    stop = length(raw) - length("\n</AppendedData>\n</VTKFile>\n")
    while pos <= stop
        n = only(reinterpret(UInt64, raw[pos:pos+7]))
        push!(blocks, raw[pos+8:pos+7+n])
        pos += 8 + n
    end
    @assert length(blocks) >= length(decls)
    geom = blocks[1:end-length(decls)]           # coordinates or points
    out = Dict{String,Vector{Float32}}()
    for (m, (name, nc)) in enumerate(decls)
        arr = collect(reinterpret(Float32, blocks[end-length(decls)+m]))
        out[name] = arr
    end
    return out, geom, decls
end

@testset "save_vtk: field selection and derived fields" begin
    dir = mktempdir()
    solver = Solver(bcs=per3, n_global=(32, 32, 12), L_domain=(1.0, 1.0, 1.0),
                    art=ArtificialProperties(enabled=true))
    Q = allocate_state(solver)

    # (a) A uniform stream, for which every derived field is zero by
    # definition. This detects a sign error or a transposed index in the curl.
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
    derived = (:vorticity, :vorticity_magnitude, :qcriterion, :divergence,
               :schlieren, :mach)
    save_vtk(solver, Q, joinpath(dir, "uniform"); fields=derived)
    got, _, decls = read_vtk_pointdata(joinpath(dir, "uniform.r0000.vtr"))
    @test [n for (n, _) in decls] == ["vorticity", "vorticity_magnitude",
                                      "qcriterion", "divergence", "schlieren", "mach"]
    for name in ("vorticity", "vorticity_magnitude", "qcriterion", "divergence",
                 "schlieren")
        @test maximum(abs, got[name]) < 1e-5
    end
    @test all(≈(0.3 / sqrt(1.4)), got["mach"])        # |u|/c, γRT = 1.4 here

    # (b) A shear layer u₁ = sin(2πy), for which ω₃ = −∂u₁/∂x₂ = −2π cos(2πy)
    # and the divergence is zero. The interior scheme is C6, so the tolerance
    # is tight.
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(2π * y), 0, 0), p=1.0, rho=1.0))
    save_vtk(solver, Q, joinpath(dir, "shear"); fields=(:vorticity, :divergence))
    got, _, _ = read_vtk_pointdata(joinpath(dir, "shear.r0000.vtr"))
    ω = got["vorticity"]
    nx, ny = 32, 32
    werr = 0.0
    for j in 1:ny, i in 1:nx
        m = (j - 1) * nx + i                          # k = 1 plane, i fastest
        want = -2π * cos(2π * xcoord(solver, 2, j))
        werr = max(werr, abs(ω[3m] - want))           # third component
    end
    @test werr < 1e-4
    @test maximum(abs, got["divergence"]) < 1e-4

    # (c) The default set is unchanged from before field selection was added.
    save_vtk(solver, Q, joinpath(dir, "default"))
    _, _, decls = read_vtk_pointdata(joinpath(dir, "default.r0000.vtr"))
    @test [n for (n, _) in decls] == ["rho", "velocity", "p", "T_ion", "Y1"]
    @test [nc for (_, nc) in decls] == [1, 3, 1, 1, 1]

    # (d) The artificial coefficients are readable, which is the reason for
    # exposing them: they report the local action of the regularization.
    save_vtk(solver, Q, joinpath(dir, "art"); fields=(:sensor, :mu_art, :D_art))
    _, _, decls = read_vtk_pointdata(joinpath(dir, "art.r0000.vtr"))
    @test [n for (n, _) in decls] == ["sensor", "mu_art", "D_art1"]

    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        fields=(:not_a_field,))
    rm(dir; recursive=true)
end

@testset "save_vtk: strided subsampling" begin
    dir = mktempdir()
    N = (32, 16, 16)
    solver = Solver(bcs=per3, n_global=N, L_domain=(1.0, 1.0, 1.0),
                    art=ArtificialProperties(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                             rho=1 + x + 2y + 4z))

    # Coordinates and payload for a given stride, read back from the piece.
    strided(stem, stride) = begin
        save_vtk(solver, Q, stem; fields=(:rho,), stride=stride)
        got, geom, _ = read_vtk_pointdata(stem * ".r0000.vtr")
        coords = [collect(reinterpret(Float64, g)) for g in geom]
        header = String(copy(read(stem * ".pvtr")))
        whole = match(r"WholeExtent=\"([^\"]+)\"", header).captures[1]
        (coords, got["rho"], whole)
    end

    # Stride 1 reproduces the unstrided grid, confirming that the general path
    # introduces no index shift.
    coords, rho1, whole = strided(joinpath(dir, "s1"), 1)
    @test length.(coords) == [32, 16, 16]
    @test whole == "0 31 0 15 0 15"
    for d in 1:3
        @test coords[d] ≈ [xcoord(solver, d, i) for i in 1:N[d]] atol = 1e-15
    end

    # Stride 2 gives half the points per dimension and extents in the coarse
    # index space. The coordinates are the ODD global stations 1, 3, 5, ...,
    # not a uniformly re-spaced grid.
    coords, rho2, whole = strided(joinpath(dir, "s2"), 2)
    @test length.(coords) == [16, 8, 8]
    @test whole == "0 15 0 7 0 7"
    for d in 1:3
        @test coords[d] ≈ [xcoord(solver, d, i) for i in 1:2:N[d]] atol = 1e-15
    end
    @test length(rho2) == 16 * 8 * 8

    # The payload holds the same values, sampled rather than averaged or
    # shifted.
    want = Float32[]
    for k in 1:2:16, j in 1:2:16, i in 1:2:32
        push!(want, Float32(solver.rho[padded_index(solver, i, j, k)]))
    end
    @test rho2 == want

    # Per-dimension strides are independent.
    coords, _, whole = strided(joinpath(dir, "s421"), (4, 2, 1))
    @test length.(coords) == [8, 8, 16]
    @test whole == "0 7 0 7 0 15"

    # An odd extent keeps the ceiling: 1, 3, ..., 15 out of 15 is 8 stations.
    odd = Solver(bcs=per3, n_global=(15, 16, 16), L_domain=(1.0, 1.0, 1.0),
                 art=ArtificialProperties(enabled=false))
    Qo = allocate_state(odd)
    initialize!(odd, Qo, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
    save_vtk(odd, Qo, joinpath(dir, "odd"); fields=(:rho,), stride=2)
    @test occursin("WholeExtent=\"0 7 0 7 0 7\"",
                   String(copy(read(joinpath(dir, "odd.pvtr")))))

    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad"); stride=0)
    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        stride=(2, 0, 1))
    rm(dir; recursive=true)
end

@testset "save_vtk: slicing" begin
    dir = mktempdir()
    N = (24, 16, 12)
    solver = Solver(bcs=per3, n_global=N, L_domain=(1.0, 1.0, 1.0),
                    art=ArtificialProperties(enabled=false))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                             rho=1 + x + 100y + 10000z))
    refresh_primitives!(solver, Q)

    # A slice is one point thick in the sliced dimension and full in the others.
    for (d, g) in ((1, 7), (2, 5), (3, 9))
        stem = joinpath(dir, "s$(d)_$(g)")
        save_vtk(solver, Q, stem; fields=(:rho,), slice=(d, g))
        want = ntuple(t -> t == d ? 1 : N[t], 3)
        header = String(copy(read(stem * ".pvtr")))
        @test occursin("WholeExtent=\"0 $(want[1]-1) 0 $(want[2]-1) " *
                       "0 $(want[3]-1)\"", header)
        got, geom, _ = read_vtk_pointdata(stem * ".r0000.vtr")
        coords = [collect(reinterpret(Float64, b)) for b in geom]
        @test length.(coords) == [want...]
        # The retained station is the requested one, not the first.
        @test coords[d] ≈ [xcoord(solver, d, g)] atol = 1e-15
        @test length(got["rho"]) == prod(want)
        # And the values are the plane at g, not some other plane.
        idx = ntuple(t -> t == d ? (g:g) : (1:N[t]), 3)
        expect = Float32[solver.rho[padded_index(solver, i, j, k)]
                         for i in idx[1], j in idx[2], k in idx[3]][:]
        @test got["rho"] == expect
    end

    # A slice composes with a stride on the two dimensions still resolved.
    stem = joinpath(dir, "both")
    save_vtk(solver, Q, stem; fields=(:rho,), stride=2, slice=(3, 5))
    header = String(copy(read(stem * ".pvtr")))
    @test occursin("WholeExtent=\"0 11 0 7 0 0\"", header)
    _, geom, _ = read_vtk_pointdata(stem * ".r0000.vtr")
    @test length.([collect(reinterpret(Float64, b)) for b in geom]) == [12, 8, 1]

    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        slice=(4, 1))
    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        slice=(2, 0))
    @test_throws ArgumentError save_vtk(solver, Q, joinpath(dir, "bad");
                                        slice=(2, 17))
    rm(dir; recursive=true)
end

@testset "save_vtk: a resolved angle writes a curvilinear grid" begin
    dir = mktempdir()
    # A cylindrical annulus with θ resolved. Written as a rectilinear grid it
    # would render unwrapped, as a box in (r, θ) rather than as an annulus.
    bcs = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
           (PeriodicBC(), PeriodicBC()))
    nr, nθ, nz = 12, 24, 10
    solver = Solver(bcs=bcs, n_global=(nr, nθ, nz), L_domain=(1.0, 2π, 1.0),
                    origin=(0.5, 0.0, 0.0), metric=CylindricalMetric(),
                    art=ArtificialProperties(enabled=false))
    Q = allocate_state(solver)
    # Solid-body swirl: purely azimuthal, constant magnitude.
    initialize!(solver, Q, (r, θ, z) -> Prim(u=(0.0, 0.5, 0.0), p=1.0, rho=1.0))

    @test container_extension(solver) == ".pvts"
    save_vtk(solver, Q, joinpath(dir, "annulus"))
    @test isfile(joinpath(dir, "annulus.pvts"))
    @test isfile(joinpath(dir, "annulus.r0000.vts"))
    @test occursin("PStructuredGrid", read(joinpath(dir, "annulus.pvts"), String))

    got, geom, _ = read_vtk_pointdata(joinpath(dir, "annulus.r0000.vts"))
    pts = collect(reinterpret(Float64, only(geom)))
    @test length(pts) == 3 * nr * nθ * nz
    radii = [hypot(pts[3m-2], pts[3m-1]) for m in 1:(nr*nθ*nz)]
    @test minimum(radii) ≈ 0.5 rtol = 1e-12
    @test maximum(radii) ≈ 1.5 rtol = 1e-12

    # The velocity must be rotated into the frame in which the points are
    # written. Unrotated, (u_r, u_θ, u_z) = (0, 0.5, 0) would appear as a
    # uniform y-directed stream; rotated, it is tangent to every circle.
    vel = got["velocity"]
    radial = 0.0; magnitude = 0.0
    for m in 1:(nr*nθ*nz)
        x, y = pts[3m-2], pts[3m-1]
        ux, uy = Float64(vel[3m-2]), Float64(vel[3m-1])
        radial = max(radial, abs(ux * x + uy * y) / hypot(x, y))
        magnitude = max(magnitude, abs(hypot(ux, uy) - 0.5))
    end
    @test radial < 1e-6                    # no radial component: purely tangential
    @test magnitude < 1e-6                 # and the swirl speed is preserved

    # An axisymmetric (θ-collapsed) polar run remains rectilinear, since the
    # meridional half-plane is the computed domain and a rectangle represents
    # it correctly.
    axi = Solver(bcs=(bcs[1], (PeriodicBC(), PeriodicBC()), bcs[3]),
                 n_global=(nr, 1, nz), L_domain=(1.0, 2π, 1.0),
                 origin=(0.5, 0.0, 0.0), metric=CylindricalMetric(),
                 art=ArtificialProperties(enabled=false))
    @test container_extension(axi) == ".pvtr"
    rm(dir; recursive=true)
end

@testset "FieldWriter: numbered frames and a .pvd carrying physical time" begin
    dir = mktempdir()
    solver = Solver(bcs=((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
                         (PeriodicBC(), PeriodicBC())),
                    n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                    art=ArtificialProperties(enabled=false), cfl=0.4)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                             p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
    writer = FieldWriter(joinpath(dir, "frames", "field"))
    run!(solver, Q; tfinal=0.03,
         callback=Callback(EveryTime(0.01), writer))

    # One frame per instant, the initial state first, numbered from zero, with
    # the directory created.
    @test writer.index == 4
    @test writer.times ≈ [0.0, 0.01, 0.02, 0.03] rtol = 1e-14
    for m in 0:3
        @test isfile(CL.frame_prefix(writer, m) * ".pvtr")
        @test filesize(CL.frame_prefix(writer, m) * ".pvtr") > 0
    end

    # The collection allows the sequence to animate against physical time, not
    # frame index, so its timesteps are the assertion.
    pvd = read(joinpath(dir, "frames", "field.pvd"), String)
    @test occursin("type=\"Collection\"", pvd)
    @test count("<DataSet", pvd) == 4
    stamps = [parse(Float64, m.captures[1])
              for m in eachmatch(r"timestep=\"([^\"]+)\"", pvd)]
    @test stamps ≈ [0.0, 0.01, 0.02, 0.03] rtol = 1e-14
    # Pieces are named relative to the .pvd's own directory rather than by
    # absolute path, so the collection remains readable on another machine.
    @test occursin("file=\"field_0000.pvtr\"", pvd)
    @test !occursin(dir, pvd)

    @test writer.wall_io > 0            # I/O cost is not in solver.wall_step
    rm(dir; recursive=true)
end

@testset "conserved_from_prim / nspecies round trip" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    @test nspecies(eos) == 2
    @test nspecies(IdealMixture(IdealSpecies("gas"; gamma=1.4, R=1.0))) == 1
    pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T_ion=1.7, Y=(0.35, 0.65))
    q = conserved_from_prim(eos, pr)
    ρ = q[1] + q[2]
    @test q[1] / ρ ≈ 0.35 atol = 1e-12
    @test q[3] / ρ ≈ 0.3 atol = 1e-12          # ρu / ρ
    Rm = 0.35 * eos.Rk[1] + 0.65 * eos.Rk[2]
    @test ρ * Rm * 1.7 ≈ 0.8 atol = 1e-12      # p = ρ R_m T_ion

    # Any two of (p, rho, T_ion) determine the state; all three spellings of the
    # same point must give the same conserved vector.
    ρ0 = 0.8 / (Rm * 1.7)
    from_rho_T = conserved_from_prim(eos, Prim(u=(0.3, -0.1, 0.2), rho=ρ0,
                                              T_ion=1.7, Y=(0.35, 0.65)))
    from_p_rho = conserved_from_prim(eos, Prim(u=(0.3, -0.1, 0.2), p=0.8,
                                              rho=ρ0, Y=(0.35, 0.65)))
    @test all(isapprox.(from_rho_T, q; rtol=1e-14))
    @test all(isapprox.(from_p_rho, q; rtol=1e-14))
    # The other two EOS models take the same three spellings. Each one's thermal
    # relation supplies the pressure that (rho, T_ion) implies.
    liquid = StiffenedGas(gamma=4.4, p_inf=6.0e8, cv=1816.0)
    p_liquid = 1000.0 * CL.gas_constant(liquid) * 300.0 - liquid.p_inf
    ideal = IdealSpecies("gas"; gamma=1.4, R=287.0)
    for (E, p_of) in ((liquid, p_liquid), (ideal, 1000.0 * 287.0 * 300.0))
        @test all(isapprox.(conserved_from_prim(E, Prim(rho=1000.0, T_ion=300.0)),
                            conserved_from_prim(E, Prim(p=p_of, T_ion=300.0));
                            rtol=1e-12))
    end
    # Exactly two, no more and no fewer.
    @test_throws ArgumentError Prim(p=1.0)
    @test_throws ArgumentError Prim(rho=1.0)
    @test_throws ArgumentError Prim(p=1.0, rho=1.0, T_ion=1.0)
end

@testset "parameter ranges: early errors naming the parameter" begin
    # Each check raises an ArgumentError whose message names the parameter.
    function names(f, name)
        err = try f(); nothing catch e; e end
        return err isa ArgumentError && occursin(name, err.msg)
    end
    wall = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    mk(; kw...) = Solver(; n_global=(16, 12, 1), L_domain=(1.0, 1.0, 1.0),
                         bcs=(wall, per, per), kw...)
    for (kw, name) in [((; predict=-1.0), "predict"), ((; max_growth=0.5), "max_growth"),
                       ((; dt_min=NaN), "dt_min"), ((; dt_min_ratio=1.0), "dt_min_ratio"),
                       ((; retries=-1), "retries"), ((; cfl_backoff=1.0), "cfl_backoff"),
                       ((; cfl_backoff=0.0), "cfl_backoff")]
        @test names(() -> StepControl(; kw...), name)
    end
    @test names(() -> compact_filter(0.5), "alphaf")
    @test names(() -> compact_filter(-0.6), "alphaf")
    @test names(() -> StiffenedGas(gamma=1.0), "gamma")
    @test names(() -> StiffenedGas(p_inf=-1.0), "p_inf")
    @test names(() -> StiffenedGas(cv=0.0), "cv")
    @test names(() -> Prim(p=Inf, rho=1.0), "finite")
    @test names(() -> Prim(p=1.0, rho=-1.0), "rho")
    @test names(() -> Prim(p=1.0, T_ion=-1.0), "T_ion")
    @test names(() -> Prim(p=1.0, rho=1.0, u=(NaN, 0, 0)), "u")
    @test names(() -> Prim(p=1.0, rho=1.0, Y=(1.1, -0.1)), "nonnegative")
    @test names(() -> Prim(p=1.0, rho=1.0, Y=(0.5, 0.6)), "sum to 1")
    @test names(() -> AtTime([0.1, NaN]), "finite")
    @test names(() -> EveryTime(Inf), "interval")
    @test names(() -> mk(cfl=0.0), "cfl")
    @test names(() -> mk(filter_interval=-1), "filter_interval")
    @test names(() -> mk(filter_cfl=-0.1), "filter_cfl")
    @test names(() -> mk(L_domain=(1.0, Inf, 1.0)), "L_domain")
    @test names(() -> mk(n_global=(16, 0, 1)), "n_global")
    @test names(() -> mk(transport=ConstantTransport(mu0=NaN)), "mu0")
    @test names(() -> mk(transport=ConstantTransport(Pr=0.0)), "Pr")
    @test names(() -> mk(transport=ConstantTransport(Sc=-1.0)), "Sc")
    @test names(() -> mk(art=ArtificialProperties(C_beta=-1.0)), "C_beta")
    @test names(() -> mk(art=ArtificialProperties(Y_tolerance=NaN)), "Y_tolerance")
    # A grid below the scheme minimum is reported with the binding scheme and
    # the extent required before any plan is built: the C8 filter at a wall.
    err = try mk(n_global=(8, 12, 1)); nothing catch e; e end
    @test err isa ArgumentError && occursin("n_global[1]", err.msg) &&
          occursin("at least 9", err.msg) && occursin("filter", err.msg)
    @test names(() -> mk(n_global=(16, 6, 1)), "n_global[2]")
    prob = Problem(domain=((0.0, 1.0), (0.0, Inf), (0.0, 1.0)), bcs=(wall, per, per),
                   ic=(x, y, z) -> Prim(p=1.0, rho=1.0))
    @test names(() -> setup(prob, Numerics(n_global=(16, 12, 1))), "domain")
end

@testset "callbacks: triggers, dt landing, composition, termination" begin
    wall3 = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC()))
    mkrun() = begin
        solver = Solver(bcs=wall3, n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                        art=ArtificialProperties(enabled=false), cfl=0.4)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                                 p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
        solver, Q
    end

    # AtTime must be landed on exactly, not overshot: dt is clipped inside run!,
    # which is the whole reason the trigger cannot be written by a caller.
    solver, Q = mkrun()
    hits = Float64[]
    targets = [0.02, 0.05, 0.11]
    run!(solver, Q; tfinal=0.2,
         callback=Callback(AtTime(targets), (s, _) -> (push!(hits, s.t); nothing)))
    @test length(hits) == 3
    @test hits ≈ targets rtol = 1e-14
    @test solver.t ≈ 0.2 rtol = 1e-14

    # Unsorted input is ordered, and a scalar is accepted.
    @test AtTime([0.3, 0.1, 0.2]).times == [0.1, 0.2, 0.3]
    @test AtTime(0.5).times == [0.5]

    # A time already behind the solver must not drive dt to zero and stall.
    solver, Q = mkrun()
    late = Float64[]
    run!(solver, Q; tfinal=0.01, nmax=20,
         callback=Callback(AtTime(-1.0), (s, _) -> (push!(late, s.t); nothing)))
    @test length(late) == 1 && solver.step > 0

    # EveryStep, and a bare function still runs every step with its return
    # value ignored — returning true must not stop the run.
    solver, Q = mkrun()
    steps = Int[]
    every = 0
    run!(solver, Q; tfinal=1e9, nmax=10,
         callback=(Callback(EveryStep(3), (s, _) -> (push!(steps, s.step); nothing)),
                   (s, _) -> (every += 1; true)))
    @test steps == [3, 6, 9]
    @test every == 10 && solver.step == 10

    # WhenState fires once by default and repeatedly with once=false.
    for (once, want) in ((true, 1), (false, 3))
        solver, Q = mkrun()
        fires = 0
        run!(solver, Q; tfinal=1e9, nmax=5,
             callback=Callback(WhenState((s, _) -> s.step >= 3; once=once),
                               (_, _) -> (fires += 1; nothing)))
        @test fires == want
    end

    # An effect returning true ends the run after that step.
    solver, Q = mkrun()
    run!(solver, Q; tfinal=1e9, nmax=1000,
         callback=Callback(WhenState((s, _) -> s.step >= 4), (_, _) -> true))
    @test solver.step == 4
end

@testset "EveryTime: evenly spaced instants, restart anchoring, soft landing" begin
    wall3 = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC()))
    mkrun() = begin
        # cfl 0.244 keeps the step this test was written at (0.4 under the
        # summed acoustic rate): the sliver below needs dt below the 0.01 interval.
        solver = Solver(bcs=wall3, n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                        art=ArtificialProperties(enabled=false), cfl=0.244)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                                 p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
        solver, Q
    end

    # Every instant is landed on exactly, and none is skipped; the instant at
    # the initial time fires before the first step.
    solver, Q = mkrun()
    hits = Float64[]
    run!(solver, Q; tfinal=0.05,
         callback=Callback(EveryTime(0.01), (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ collect(0.0:0.01:0.05) rtol = 1e-14
    @test solver.step > 0 && count(iszero, hits) == 1

    # AtTime fires at a listed initial time too, and only once.
    solver, Q = mkrun()
    hits = Float64[]
    run!(solver, Q; tfinal=0.02,
         callback=Callback(AtTime([0.0, 0.01]), (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ [0.0, 0.01] rtol = 1e-14

    # Anchored to solver.t on first use, not to zero: a restarted run picks up at
    # the next instant rather than replaying the schedule.
    solver, Q = mkrun()
    solver.t = 0.055
    hits = Float64[]
    run!(solver, Q; tfinal=0.08,
         callback=Callback(EveryTime(0.01), (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ [0.06, 0.07, 0.08] rtol = 1e-14

    # `start` offsets the schedule, and instants before it are not visited.
    solver, Q = mkrun()
    hits = Float64[]
    run!(solver, Q; tfinal=0.05,
         callback=Callback(EveryTime(0.02; start=0.015),
                           (s, _) -> (push!(hits, s.t); nothing)))
    @test hits ≈ [0.015, 0.035] rtol = 1e-14

    @test_throws ArgumentError EveryTime(0.0)
    @test_throws ArgumentError StepControl(landing_steps=0)

    # The soft landing is the purpose of `landing_steps`. Under a hard clip the
    # same schedule leaves one very small step before every instant, so compare
    # the spread of accepted steps.
    spread(control) = begin
        s, q = mkrun()
        dts = Float64[]
        run!(s, q; tfinal=0.05, control=control,
             callback=(Callback(EveryTime(0.01), (_, _) -> nothing),
                       (x, _) -> push!(dts, x.dt_prev)))
        minimum(dts) / maximum(dts)
    end
    @test spread(StepControl(landing_steps=1)) < 0.35     # sliver: 2.2e-3 vs 7.8e-3
    @test spread(StepControl()) > 0.99                    # split evenly instead

    # An instant BEYOND tfinal must not be landed on. Instants are computed as
    # `start + n*interval`, so the third one here is `3 * 0.05`, which is the next
    # float above the 0.15 literal — the schedule ends one ULP past the endpoint.
    # The soft landing used to aim at it anyway, and since `dt` was already
    # clipped to `tfinal - t` the gap stayed a shade above `dt` and `ceil(gap/dt)`
    # stayed at 2, halving the step against a target the run never reaches. The
    # run still finished and every instant still fired, so the only symptom was
    # step count: 48 steps against 9 in the case this was found in.
    #
    # Measured against the same run with no trigger rather than a fixed number,
    # since the step count is a property of the CFL here and not of the landing.
    @test 3 * 0.05 > 0.15                          # the ULP the case rests on
    nsteps(cb) = begin
        s, q = mkrun()
        run!(s, q; tfinal=0.15, callback=cb)
        s.step
    end
    bare = nsteps(nothing)
    @test nsteps(Callback(EveryTime(0.05), (_, _) -> nothing)) <= bare + 4
end

@testset "rewind!: a rollback re-arms the instants it abandoned" begin
    # Unit behaviour first. AtTime rewinds to the first instant ahead of the
    # restored time, EveryTime re-anchors, and a step-counting trigger has no
    # schedule to move and takes the no-op default.
    trigger = AtTime([0.1, 0.2, 0.3])
    trigger.next = 3
    CL.rewind!(trigger, 0.15, 7)
    @test trigger.next == 2
    CL.rewind!(trigger, 0.35, 9)
    @test trigger.next == 4                              # all behind: none left
    CL.rewind!(trigger, 0.0, 0)
    @test trigger.next == 1

    et = EveryTime(0.01)
    solver = Solver(bcs=per3, n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                    art=ArtificialProperties(enabled=false))
    solver.t = 0.045
    @test CL.next_time(et, solver) ≈ 0.05
    CL.rewind!(et, 0.021, 3)
    solver.t = 0.021
    @test CL.next_time(et, solver) ≈ 0.03
    @test CL.rewind!(EveryStep(4), 0.1, 3) === nothing

    # Then the wiring, against a rollback placed where it can be reasoned about.
    # Physical failures occur in the startup transient, before a savepoint has
    # advanced past step 0, which demonstrates nothing here. Corrupting the
    # state once at a chosen time and letting the positivity check find it
    # places the rollback across instants that have already fired.
    walls = ((SlipWallBC(), SlipWallBC()), (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC()))
    s2 = Solver(bcs=walls, n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                art=ArtificialProperties(enabled=false), cfl=0.4)
    Q2 = allocate_state(s2)
    initialize!(s2, Q2, (x, y, z) -> Prim(u=(0.2, 0, 0),
                                          p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
    instants = collect(0.005:0.005:0.06)
    seen = Float64[]
    # A rollback re-arms a WhenState that fired after the savepoint, so the
    # one-shot fault is held in the condition, not in the trigger.
    spoiled = Ref(false)
    spoil = Callback(WhenState((x, _) -> x.t >= 0.037 && !spoiled[]),
                     (x, q) -> (spoiled[] = true; q[padded_index(x, 3, 3, 3), 1] = -1.0;
                                nothing))
    run!(s2, Q2; tfinal=0.06, nmax=5000,
         control=StepControl(retries=2, savepoint_interval=2),
         callback=(Callback(AtTime(instants), (x, _) -> (push!(seen, x.t); nothing)),
                   spoil))
    @test s2.cfl < 0.4                                    # it did roll back
    @test s2.t ≈ 0.06 rtol = 1e-12                        # and then finished
    # Re-arming has a positive signature: an instant crossed on the abandoned
    # trajectory is visited a second time on the replacement trajectory. Without
    # the rewind it is visited once and never revisited, and the `unique` check
    # below would fall short.
    @test length(seen) > length(instants)
    @test sort(unique(round.(seen, digits=10))) ≈ instants rtol = 1e-9
end

@testset "state queries and refresh_primitives! in a callback" begin
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    solver = Solver(bcs=per3, n_global=(16, 12, 12), L_domain=(1.0, 1.0, 1.0),
                    eos=eos, art=ArtificialProperties(enabled=false), cfl=0.4)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.3, 0.7), u=(0.2, -0.1, 0.05),
                                             p=1 + 0.1exp(-40(x - 0.5)^2), rho=1.0))
    refresh_primitives!(solver, Q)

    # The layout-free spellings agree with the primitives they stand in for, and
    # do not assume a species count.
    I = padded_index(solver, 5, 4, 3)
    @test mixture_density(solver, Q, I) ≈ solver.rho[I] rtol = 1e-14
    @test all(velocity(solver, Q, I) .≈ (solver.u[I], solver.v[I], solver.w[I]))
    @test mass_fraction(solver, Q, I, 2) ≈ solver.Y[2][I] rtol = 1e-14
    @test total_energy(solver, Q, I) == Q[I, solver.equations.i_energy]
    @test mixture_density(solver, Q, I) ≈ Q[I, 1] + Q[I, 2] rtol = 1e-14

    # In serial every rank holds every edge, so the `nothing` branch is
    # exercised only under MPI.
    @test boundary_plane(solver, 2, 1) == CL.wallplane(solver.decomp, 2, 1)

    # interior_index is the documented conversion between the two index bases:
    # boundary_plane yields padded indices, xcoord takes interior ones.
    @test interior_index(solver, I) == (5, 4, 3)
    @test padded_index(solver, interior_index(solver, I)...) == I
    for J in boundary_plane(solver, 1, 2)
        i, j, k = interior_index(solver, J)
        @test padded_index(solver, i, j, k) == J
        @test i == solver.decomp.n_local[1]              # the high-x face
        @test xcoord(solver, 1, i) ==
              global_xcoord(solver, 1, solver.decomp.offset[1] + i)
    end

    # The contract stated by refresh_primitives!: inside a callback the
    # primitives belong to the fifth RK stage's INPUT state, so they predate the
    # final stage update and the filter pass. The assertion is that refreshing
    # changes the answer, and that the refreshed value is the one Q holds.
    stale = Ref(0.0); fresh = Ref(0.0); fromQ = Ref(0.0)
    run!(solver, Q; tfinal=1e9, nmax=6, callback=(s, q) -> begin
        s.step == 6 || return
        stale[] = s.rho[I]
        fromQ[] = mixture_density(s, q, I)
        refresh_primitives!(s, q)
        fresh[] = s.rho[I]
    end)
    @test isapprox(fresh[], fromQ[]; rtol=1e-14)         # refreshed agrees with Q
    @test !isapprox(stale[], fromQ[]; rtol=1e-12)        # unrefreshed does not
end

@testset "phase change: opening the upstream face lets the reflected wave leave" begin
    # A shock/interface interaction in a translating frame, with the upstream
    # boundary changing from inflow to outflow, by a phase change, when the
    # interface-reflected wave reaches it. A finiteness check says nothing
    # about the question this case asks: whether the opened boundary lets the
    # wave out.
    #
    #        shocked driven | driven |    heavy
    #      ---------------->|        |
    #        (region 2)     ^shock   ^interface
    #      0               0.20     0.40              1
    #
    # The state initialized upstream of the shock is the SHOCKED driven gas —
    # region 2 of the normal-shock relations, not a driver — so the incident
    # shock is already formed and the run starts at the physics of interest.
    # Sustaining that state makes the upstream boundary an inflow
    # (u2 + U > 0 into the domain) for as long as the incident shock is being
    # driven. NSCBCInflowBC is the condition that holds it: it replaces every
    # incoming characteristic with a relaxation toward (u, T_ion, Y), the
    # appropriate condition while region 2 is uniform there.
    #
    # It becomes inappropriate when the interface-reflected shock arrives.
    # Behind that shock the state is no longer region 2, so an inflow condition
    # still relaxing toward region-2 targets is over-constraining the boundary,
    # and the mismatch radiates back inward onto the interface. The remedy is
    # to stop imposing and start absorbing: the second phase takes
    # NSCBCOutflowBC there.
    #
    # The frame translation U is chosen so the SHOCKED interface is nearly at
    # rest (it drifts ~0.04 once the shock has passed it), the standard frame for a
    # shock/interface calculation: it keeps the interface in the box, and it
    # gives both faces a mean flow so nothing is being tested at the degenerate
    # u = 0 point where NSCBC's Mach-dependent terms drop out.
    γ, RL, RH, MS, U = 1.4, 1.0, 0.25, 1.5, -0.5
    N, XS, XI, TEND = 160, 0.20, 0.40, 1.0
    Hx = 1 / (N - 1)

    # Region 1 (unshocked driven gas) and the normal-shock relations for the
    # region 2 that sits behind a Mach MS shock running into it.
    p1 = ρ1 = T1 = 1.0
    c1 = sqrt(γ)
    p2 = p1 * (2γ * MS^2 - (γ - 1)) / (γ + 1)
    ρ2 = ρ1 * ((γ + 1) * MS^2) / ((γ - 1) * MS^2 + 2)
    u2 = c1 * 2 / (γ + 1) * (MS^2 - 1) / MS
    T2 = p2 / (ρ2 * RL)
    eos2 = IdealMixture([IdealSpecies{Float64}("light", RL, γ),
                         IdealSpecies{Float64}("heavy", RH, γ)])

    # `xlo < 0` extends the domain upstream at the same spacing, which is how
    # the reference below is built. The heavy gas is at the driven gas's p and
    # T_ion, so RH sets the density ratio: Atwood 0.6 here.
    build(xbc; xlo=0.0) = begin
        n = round(Int, (1.0 - xlo) / Hx) + 1
        δ = 2Hx
        setup(Problem(eos=eos2, transport=ConstantTransport(mu0=0.0),
                      domain=((xlo, xlo + (n - 1) * Hx), (0.0, Hx), (0.0, Hx)),
                      bcs=(xbc, per3[2], per3[3]),
                      ic=(x, y, z) -> begin
                          s = tanh_blend(x, XS, δ)     # 0 shocked, 1 unshocked
                          θ = tanh_blend(x, XI, δ)     # 0 light,   1 heavy
                          Prim(Y=(1 - θ, θ), u=((1 - s) * u2 + U, 0.0, 0.0),
                               p=(1 - s) * p2 + s * p1,
                               T_ion=(1 - s) * T2 + s * T1)
                      end),
              Numerics(n_global=(n, 1, 1), art=ArtificialProperties(enabled=true),
                       cfl=0.4,
                       # A shocked binary interface ends beyond the
                       # mass-fraction band, five of 160 points here, as the
                       # shock/SF6 validation case does. The boundary condition
                       # is what this tests, so the state is reported rather
                       # than rejected.
                       control=StepControl(validity=:permissive)))
    end

    inflow() = NSCBCInflowBC(u=(u2 + U, 0.0, 0.0), T_ion=T2, Y=[1.0, 0.0])
    # sigma small on purpose. The pressure this boundary will face once the
    # reflected shock has passed is the post-reflected-shock pressure (~3.23),
    # which is the answer to the interaction and not something the test may
    # assume; relaxing hard toward `pinf` would impose a pressure known to be
    # wrong and drag the driven section with it. A weak relaxation is the
    # honest reading of "let the wave out and do not hold a pressure I do not
    # know" — measured: sigma 0.25 costs a factor 3.5 in the agreement below.
    outflow() = NSCBCOutflowBC(pinf=p2, sigma=0.05)
    # Nothing reaches the downstream end within TEND (the transmitted shock
    # gets to x ~ 0.8), so a Dirichlet on the unshocked heavy state is exact
    # there and keeps this test about the upstream boundary alone.
    downstream() = DirichletBC((x, y, z, t) -> Prim(Y=(0.0, 1.0),
                                                    u=(U, 0.0, 0.0), p=p1,
                                                    T_ion=T1))

    # "The reflected shock has reached the upstream plane." A rank-local verdict
    # on the plane this rank owns, left to WhenState to reduce, as the
    # WhenState docstring describes: a run that ends for a phase change must
    # never be ended by an unreduced rank-local test.
    arrived(solver, Q) = begin
        plane = CL.wallplane(solver.decomp, 1, 1)
        plane === nothing && return false
        any(I -> solver.p[I] > 1.05 * p2, plane)
    end

    xline(s, Q) = begin
        CL.exchange_state!(Q, s.decomp)
        CL.primitives!(s, Q)
        nx = s.decomp.n_local[1]
        ([xcoord(s, 1, i) for i in 1:nx],
         [s.p[padded_index(s, i, 1, 1)] for i in 1:nx],
         [s.u[padded_index(s, i, 1, 1)] for i in 1:nx],
         [s.Y[2][padded_index(s, i, 1, 1)] for i in 1:nx])
    end
    # Interface position, interpolated across the cell where Y_heavy crosses
    # 1/2 — a sub-cell measure, since a re-shock moves the interface by a
    # fraction of a cell over this run.
    ipos(x, Y) = begin
        i = findfirst(>=(0.5), Y)
        x[i-1] + (0.5 - Y[i-1]) / (Y[i] - Y[i-1]) * (x[i] - x[i-1])
    end

    # (1) Opened on arrival: the first phase ends on the step the wave
    #     arrives, and the second continues from it under the outflow.
    s_sw, Q_sw = build((inflow(), downstream()))
    run!(s_sw, Q_sw; tfinal=TEND, nmax=100_000,
         callback=Callback(WhenState(arrived), Returns(true)))
    fired_step = Ref(s_sw.step)
    Q_fired = Ref{Any}(copy(Q_sw))
    @test 0 < fired_step[] && s_sw.t < TEND    # the first phase ended early
    s_sw, Q_sw = setup(s_sw, Q_sw; bcs=((outflow(), downstream()), per3[2], per3[3]))
    run!(s_sw, Q_sw; tfinal=TEND, nmax=100_000)
    @test s_sw.t == TEND

    # (2) Inflow held throughout: the control, and the same run bit-for-bit up
    #     to the change. WhenState does not clip dt, so the step sequences agree
    #     and this is an equality, not an approximation.
    s_h, Q_h = build((inflow(), downstream()))
    run!(s_h, Q_h; tfinal=TEND, nmax=fired_step[])
    @test s_h.step == fired_step[]
    @test Q_h == Q_fired[]
    run!(s_h, Q_h; tfinal=TEND, nmax=100_000)

    # (3) The reference: the same problem with the upstream boundary one domain
    #     length further away, so the reflected shock never reaches it and no
    #     boundary treatment is relevant. This makes the comparison a
    #     measurement rather than a difference of two guesses — a non-reflecting
    #     condition is only ever tested against a domain big enough not to need
    #     one. Same spacing, so the grids coincide on x >= 0 to round-off.
    s_r, Q_r = build((inflow(), downstream()); xlo=-1.0)
    run!(s_r, Q_r; tfinal=TEND, nmax=100_000)

    xr, pr, ur, Yr = xline(s_r, Q_r)
    xh, ph, uh, Yh = xline(s_h, Q_h)
    xs, ps, us, Ys = xline(s_sw, Q_sw)
    keep = [i for i in eachindex(xr) if xr[i] >= -1e-9]
    @test maximum(abs, xr[keep] .- xh) < 1e-12

    win = [i for i in eachindex(xh) if 0.05 <= xh[i] <= 0.95]
    errp(p) = maximum(abs.(p[win] .- pr[keep][win]))
    erru(u) = maximum(abs.(u[win] .- ur[keep][win]))
    ref_i = ipos(xr, Yr)

    # The measurement. Held, the inflow condition keeps imposing region 2 after
    # the reflected shock has arrived and the error radiates back over the
    # interface; opened, the wave leaves. Measured against the reference:
    # pressure 0.104 held vs 0.011 opened, velocity 0.0279 vs 0.0031,
    # interface displacement 0.0024 vs 0.00016 (about a fortieth of a cell).
    # Guards are ratios plus loose absolute bounds: this is a nonlinear run
    # through the artificial-property sensor, so it is not bit-reproducible and
    # only the order of magnitude is being asserted.
    @test errp(ps) < 0.04
    @test errp(ph) > 0.08
    @test errp(ph) > 3 * errp(ps)
    @test erru(us) < 0.008
    @test erru(uh) > 0.02
    @test erru(uh) > 3 * erru(us)
    @test abs(ipos(xs, Ys) - ref_i) < 0.0015
    @test abs(ipos(xh, Yh) - ref_i) > 2.5 * abs(ipos(xs, Ys) - ref_i)
end

@testset "checkpoint round trip" begin
    solver = mkslv(n_global=(12, 12, 12))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    solver.t = 0.37; solver.step = 42
    save_checkpoint(solver, Q, "test_ckpt")
    Q2 = allocate_state(solver); solver.t = 0.0; solver.step = 0
    load_checkpoint!(solver, Q2, "test_ckpt")
    @test solver.t == 0.37 && solver.step == 42
    @test all(Q2[padded_index(solver, i, j, k), c] == Q[padded_index(solver, i, j, k), c]
              for c in 1:5, i in 1:12, j in 1:12, k in 1:12)
    foreach(rm, filter(startswith("test_ckpt"), readdir()))
end

@testset "checkpoint header rejects a solver it does not describe" begin
    # Everything the payload's interpretation depends on is in the header. Each
    # solver below reads the block cleanly if its own field is not checked: the
    # extents agree in every case and the state means something else. The
    # expected message is asserted to ensure each case exercises its intended
    # check, not an earlier one.
    mk(; kw...) = Solver(; n_global=(12, 12, 12), bcs=per3, L_domain=(2π, 2π, 2π),
                         art=ArtificialProperties(enabled=false), kw...)
    eos2 = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                         IdealSpecies{Float64}("b", 2.0, 1.6)])
    base = mk(eos=eos2)
    Qb = allocate_state(base)
    initialize!(base, Qb, (x, y, z) -> Prim(Y=(0.3, 0.7), p=1.0, rho=1.0))
    save_checkpoint(base, Qb, "test_ckpt_header")
    reject(solver, msg) =
        @test_throws msg load_checkpoint!(solver, allocate_state(solver),
                                          "test_ckpt_header")

    # A different species set of the same size: same conserved count, same grid,
    # same metric. Only the component names separate them.
    reject(mk(eos=IdealMixture([IdealSpecies{Float64}("c", 1.0, 1.4),
                                IdealSpecies{Float64}("d", 2.0, 1.6)])),
           "conserved component mismatch")
    # A different species count, which also moves the conserved count.
    reject(mk(eos=IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4)])),
           "conserved layout mismatch")
    # The same grid dimensions over a longer domain, which the extents cannot
    # see and the coordinates can.
    reject(mk(eos=eos2, L_domain=(4π, 2π, 2π)), "grid coordinate mismatch")
    # The same grid shifted, likewise invisible to every other field.
    reject(mk(eos=eos2, origin=(0.5, 0.0, 0.0)), "grid coordinate mismatch")
    # A different metric on the same grid dimensions: the block is the same
    # shape and every momentum component means something else.
    reject(mk(eos=eos2, bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
              origin=(0.5, 0.0, 0.0), metric=CylindricalMetric()),
           "metric mismatch")

    # A file in the original unversioned format. The layout is fixed, so the
    # fields added since would be read from the wrong offsets; the magic
    # identifies it instead.
    open("test_ckpt_header_old.r0000.ckpt", "w") do io
        write(io, UInt64(CL.CKPT_MAGIC_V1))
    end
    @test_throws "unversioned format" load_checkpoint!(base, Qb,
                                                       "test_ckpt_header_old")

    # A stretched dimension against the uniform grid it was written on. The two
    # agree on extent, origin, species and metric and differ only point by
    # point, which is why the coordinates are stored rather than the extent. A
    # stretched dimension must be non-periodic, so this pair needs a checkpoint
    # of its own, and it is paired with a reload onto the grid that wrote it so
    # that the check is shown to discriminate rather than to refuse everything.
    mkw(st) = Solver(n_global=(12, 12, 12), L_domain=(1.0, 1.0, 1.0),
                     bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                     stretch=(st, nothing, nothing), art=ArtificialProperties(enabled=false))
    uniform = mkw(nothing)
    Qu = allocate_state(uniform)
    initialize!(uniform, Qu, (x, y, z) -> Prim(p=1.0, rho=1.0))
    uniform.step = 7
    save_checkpoint(uniform, Qu, "test_ckpt_uniform")
    stretched = mkw(sine_cluster(0.0, 1.0, 0.5, 0.3))
    @test_throws "grid coordinate mismatch" load_checkpoint!(
        stretched, allocate_state(stretched), "test_ckpt_uniform")
    reload = mkw(nothing)
    load_checkpoint!(reload, allocate_state(reload), "test_ckpt_uniform")
    @test reload.step == 7

    foreach(rm, filter(startswith("test_ckpt"), readdir()))
end

@testset "smoke: three RK steps of every headline configuration" begin
    for build in (
        () -> setup(Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                            ic=(x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0)),
                    Numerics(n_global=(16, 16, 16))),
        () -> setup(Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                            ic=(x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0)),
                    Numerics(n_global=(16, 16, 16), deriv=lele_d1_10())),
        () -> setup(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                            metric=CylindricalMetric(),
                            bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                            ic=(r, θ, z) -> Prim(u=(0, 0, 0), p=1 + exp(-40(r - 0.4)^2), rho=1.0)),
                    Numerics(n_global=(48, 1, 1))),
    )
        solver, Q = build()
        run!(solver, Q; tfinal=1e9, nmax=3)
        bad = any(!isfinite(Q[padded_index(solver, i, j, k), c])
                  for c in 1:solver.equations.n_cons, i in 1:solver.decomp.n_local[1],
                      j in 1:solver.decomp.n_local[2], k in 1:solver.decomp.n_local[3])
        @test !bad
    end
end

@testset "Sod shock tube: profile matches exact Riemann solution" begin
    # Classic single-gas Sod (γ = 1.4) on [0,1], diaphragm at x = 0.5:
    #   left (ρ,u,p) = (1, 0, 1), right (ρ,u,p) = (0.125, 0, 0.1).
    # Integrate to t = 0.2 in 1-D (collapsed transverse dims) with artificial
    # fluid properties capturing the shock, and compare the ρ/u/p line profile
    # against the analytic Riemann solution sampled at each node. Errors are
    # L1 over the profile — shock/contact smearing over a few cells is expected,
    # so the tolerance is looser than the smooth-operator tests but still tight
    # enough that a wrong wave speed, plateau, or star state fails it.
    γ = 1.4
    ρL, uL, pL = 1.0, 0.0, 1.0
    ρR, uR, pR = 0.125, 0.0, 0.1
    x0 = 0.5; tfin = 0.2
    Nx = 400; Lx = 1.0; hx = Lx / (Nx - 1); δ = 2hx
    prob = Problem(name="Sod", eos=IdealSpecies("gas"; gamma=γ, R=1.0),
                   transport=ConstantTransport(mu0=0.0),
                   domain=((0.0, Lx), (0.0, hx), (0.0, hx)),
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   ic=(x, y, z) -> begin
                       θ = tanh_blend(x, x0, δ)
                       Prim(rho=(1 - θ) * ρL + θ * ρR,
                            u=((1 - θ) * uL + θ * uR, 0.0, 0.0),
                            p=(1 - θ) * pL + θ * pR)
                   end)
    solver, Q = setup(prob,
                      Numerics(n_global=(Nx, 1, 1), art=ArtificialProperties(enabled=true),
                                cfl=0.4))
    run!(solver, Q; tfinal=tfin, nmax=100_000)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)

    pstar, ustar, cL, cR = exact_riemann_star(ρL, uL, pL, ρR, uR, pR, γ)
    # Star state is the canonical Sod result (Toro): p* ≈ 0.30313, u* ≈ 0.92745.
    @test pstar ≈ 0.30313 atol = 1e-4
    @test ustar ≈ 0.92745 atol = 1e-4

    nx = solver.decomp.n_local[1]
    eρ = eu = ep = 0.0
    for i in 1:nx
        I = padded_index(solver, i, 1, 1); x = xcoord(solver, 1, i)
        r, u, p = exact_riemann_sample((x - x0) / tfin, ρL, uL, pL, ρR, uR, pR,
                                       γ, pstar, ustar, cL, cR)
        eρ += abs(solver.rho[I] - r); eu += abs(solver.u[I] - u); ep += abs(solver.p[I] - p)
    end
    eρ /= nx; eu /= nx; ep /= nx
    # Guards at roughly twice the measured error, as in test/validation.jl. The
    # run integrates a nonlinear sensor for hundreds of steps, so the fourth
    # significant figure moves under an arithmetic reassociation; a moved third
    # digit is real.
    @test eρ < 6.0e-3    # measured 2.901e-3
    @test eu < 9.5e-3    # measured 4.622e-3
    @test ep < 5.0e-3    # measured 2.458e-3
end

# Not solver code, but `bench/` and the cluster scripts are not imported by this
# suite, so if these are not tested here they are not tested anywhere — and the
# whole reason they exist is to turn a silent wrong-default run into an error.
@testset "script argument parsing" begin
    defaults = (N=32, tfinal=10.0, configs="off:1", progress=0, quiet=false)
    pos = (:N, :tfinal)

    @test script_args(String[], defaults; positional=pos) == defaults
    opt = script_args(["128", "2.5", "progress=200", "configs=on:1:0.008"],
                      defaults; positional=pos)
    @test (opt.N, opt.tfinal, opt.progress) == (128, 2.5, 200)
    @test opt.configs == "on:1:0.008"      # String defaults pass through verbatim
    # Order is free, and a positional may be supplied by name instead.
    @test script_args(["progress=5", "64"], defaults; positional=pos).N == 64
    @test script_args(["N=64"], defaults; positional=pos).N == 64
    @test script_args(["nmax=1"], (nmax=typemax(Int),)).nmax == 1

    for text in ("true", "yes", "on", "1")
        @test script_args(["quiet=$text"], defaults; positional=pos).quiet
    end
    for text in ("false", "no", "off", "0")
        @test !script_args(["quiet=$text"], defaults; positional=pos).quiet
    end

    # The point of the exercise: every one of these would be a silent default
    # in a parser that falls through on an unknown name.
    @test_throws ArgumentError script_args(["progerss=1"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["progress=lots"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["quiet=maybe"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["1", "2", "3"], defaults; positional=pos)
    @test_throws ArgumentError script_args(["64"], defaults)
    @test_throws ArgumentError script_args(String[], defaults; positional=(:nope,))

    @test script_grid("128") == (128, 128, 128)
    @test script_grid("256,256,512") == (256, 256, 512)
    @test_throws ArgumentError script_grid("128,256")
    @test_throws ArgumentError script_grid("128x128")
end

@testset "pointwise kernels: the KA path reproduces the threaded path" begin
    # Every pointwise phase is one shared per-point
    # body behind two launchers. The bodies are per-point independent, so the
    # KernelAbstractions CPU path must reproduce the @threaded path BITWISE —
    # not to round-off — over a full run touching primitives, fluxes, sensors,
    # species diffusion, the RK update, and the state filter. Cylindrical
    # geometry in the second case adds the metric-correction and metric-source
    # bodies.
    per3k = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    function tube_pair()
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        s = Solver(n_global=(48, 32, 1), L_domain=(1.0, 0.6, 1.0), eos=eos,
                   bcs=((SlipWallBC(), SlipWallBC()), per3k[2], per3k[3]))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.0))
        end)
        return s, Q
    end
    s1, Q1 = tube_pair()
    run!(s1, Q1; tfinal=0.02, nmax=50)
    local s2, Q2
    CL.FORCE_KA[] = true
    try
        s2, Q2 = tube_pair()
        run!(s2, Q2; tfinal=0.02, nmax=50)
    finally
        CL.FORCE_KA[] = false
    end
    @test s1.step == s2.step
    @test parent(Q1) == parent(Q2)

    # The tube above runs the default partial-density species channel. The
    # same tube under the bulk channel adds the conserved-gradient flux body,
    # and under the Fickian one the per-species bound and diffusivity bodies
    # and the correction-velocity branch of the flux body.
    function channel_tube_pair(channel)
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        s = Solver(n_global=(48, 32, 1), L_domain=(1.0, 0.6, 1.0), eos=eos,
                   bcs=((SlipWallBC(), SlipWallBC()), per3k[2], per3k[3]),
                   art=ArtificialProperties(enabled=true, species_flux=channel))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.0))
        end)
        return s, Q
    end
    for channel in (:bulk, :fickian)
        s1b, Q1b = channel_tube_pair(channel)
        run!(s1b, Q1b; tfinal=0.02, nmax=50)
        local s2b, Q2b
        CL.FORCE_KA[] = true
        try
            s2b, Q2b = channel_tube_pair(channel)
            run!(s2b, Q2b; tfinal=0.02, nmax=50)
        finally
            CL.FORCE_KA[] = false
        end
        @test s1b.step == s2b.step
        @test parent(Q1b) == parent(Q2b)
        @test maximum(s1b.D_art[1]) > 0
        channel === :bulk && @test s1b.D_art[1] == s1b.D_art[2]
    end
    @test maximum(s1.D_art[1]) > 0
    @test s1.D_art[1] == s1.D_art[2]

    function axis_pair()
        s = Solver(n_global=(32, 1, 24), L_domain=(1.0, 2π, 1.0),
                   bcs=((AxisBC(), SlipWallBC()), per3k[2], per3k[3]),
                   metric=CylindricalMetric())
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, z) ->
            Prim(u=(0.0, 0.2r, 0.05 * sin(2π * z)), p=1.0 + 0.02r^2, rho=1.0))
        return s, Q
    end
    s3, Q3 = axis_pair()
    run!(s3, Q3; tfinal=0.02, nmax=20)
    local s4, Q4
    CL.FORCE_KA[] = true
    try
        s4, Q4 = axis_pair()
        run!(s4, Q4; tfinal=0.02, nmax=20)
    finally
        CL.FORCE_KA[] = false
    end
    @test s3.step == s4.step
    @test parent(Q3) == parent(Q4)
end

@testset "device line solves: DevicePlan reproduces the host plans" begin
    # The compact line solves run as
    # KernelAbstractions kernels in a (lines × n) layout, one thread per line,
    # with the reduced interface stage on the host. The device arithmetic
    # mirrors the host path per line operation for operation (including the
    # divide-vs-multiply-by-inverse split between the banded x and y/z
    # conventions), so on the KA CPU backend the comparison is BITWISE, the
    # same equality gate the pointwise kernels carry. Closed edges, folds
    # and periodic wrap all route through the same kernels; the fold cases
    # fill the halos with random data, since path equality does not require
    # physically meaningful mirror values.
    cpu = CL.KernelAbstractions.CPU()
    function compare(scheme, dim; periodic=false, lo_fold=nothing)
        # 13 is the smallest extent the six-row T8 closure set accepts.
        d = CL.Decomp((16, 13, 14), (periodic, periodic, periodic); dims=(1, 1, 1))
        plan = CL.plan_direction(d, scheme, dim, 0.1; lo_fold=lo_fold)
        dplan = device_plan(plan, cpu)
        f = CL.field(d)
        rand!(f)
        CL.exchange_halos!(f, d)
        out_h = CL.field(d)
        out_d = CL.field(d)
        apply_along!(out_h, plan, f, d)
        apply_along!(out_d, dplan, f, d)
        return view(out_h, CL.interior(d)) == view(out_d, CL.interior(d))
    end
    schemes = (lele_d1_6(Float64), lele_d1_6(Float64; closures=:brady_livescu),
               lele_d1_8(Float64), lele_d1_8(Float64; closures=:brady_livescu),
               lele_d1_10(Float64), compact_filter(0.45),
               gaussian_filter(Float64), compact_d8(Float64))
    for scheme in schemes, periodic in (false, true), dim in 1:3
        @test compare(scheme, dim; periodic=periodic)
    end
    for dim in 1:3, σ in (1, -1)
        @test compare(lele_d1_6(Float64), dim; lo_fold=σ)
    end

    # The allocation half of the backend interface: DeviceBackend routes field and
    # allocate_state through KernelAbstractions.zeros with the same halo
    # padding as the CPU backend.
    d = CL.Decomp((16, 12, 14), (false, true, true); dims=(1, 1, 1))
    bk = DeviceBackend(cpu)
    f = CL.field(bk, d)
    @test size(f) == size(CL.field(d))
    @test all(iszero, f)
    Q = allocate_state(bk, d, 7)
    @test Q isa ConservedState
    @test size(Q) == (size(CL.field(d))..., 7)
end

include("wall_flux_tests.jl")
include("composite_face_tests.jl")
include("transport_tests.jl")
include("binary_diffusion_tests.jl")
include("neutral_diffusion_tests.jl")
include("transport_integration_tests.jl")
include("neutral_transport_integration_tests.jl")
include("neutral_transport_coefficients_tests.jl")
include("neutral_transport_domain_tests.jl")
include("ionmix_tests.jl")
include("sesame_tests.jl")
include("device_tests.jl")
include("patch_tests.jl")
include("level_tests.jl")
include("substep_rate_tests.jl")
include("conservation_tests.jl")
include("interface_reflection_tests.jl")
include("sharpening_tests.jl")
include("positivity_tests.jl")
include("seam_tests.jl")
include("io_tests.jl")
include("runloop_tests.jl")
include("phase_tests.jl")
include("docrefs_tests.jl")
include("reference_tests.jl")
include("api_surface_tests.jl")
include("pointwise_callbacks_tests.jl")
include("amr_frontend_tests.jl")
include("initial_states_tests.jl")
include("turbulent_inflow_tests.jl")
include("boundary_shorthand_tests.jl")
include("capability_tests.jl")

# The extension suites run only where their weak dependency loads. A skip is
# recorded as a broken test, so the summary tree shows it in its own column
# rather than as a pass, and `require=hdf5,makie` (a test argument:
# `Pkg.test(test_args=["require=hdf5"])`, or on the command line) turns the
# skip of a named suite into a failure, for a job that expects the suite to
# run. `timing` is read by runtests.jl and listed here so that it parses.
const SUITE_OPTS = script_args(ARGS, (require = "", timing = false))
const REQUIRED_SUITES = Symbol.(filter(!isempty, split(SUITE_OPTS.require, ',')))

# HDF5 is a weak dependency and is not loadable from the package environment
# alone; the test target carries it, so `Pkg.test` runs the suite.
hdf5_loaded = try
    @eval using HDF5
    true
catch
    false
end
if hdf5_loaded
    include("hdf5_tests.jl")
else
    println("HDF5 not loadable in this environment — extension tests SKIPPED. " *
            "Run test/hdf5_tests.jl from an environment carrying both, and " *
            "under mpiexec for the decomposition-independent restart.")
    @testset "HDF5 extension: skipped" begin
        @test hdf5_loaded skip=!(:hdf5 in REQUIRED_SUITES)
    end
end

# A Makie backend is likewise a weak dependency, absent from the package
# environment. The extraction API it exercises is exported by the core, so this
# skip only foregoes the extension's plotting methods and the collective-profile
# check under decomposition.
#
# Unlike HDF5, CairoMakie is not in Project.toml's test target: it
# is a heavy dependency to resolve and precompile for two testsets, so
# `Pkg.test` never reaches this branch. The Makie extension is verified from
# the docs environment, which carries CairoMakie already; CI's documentation
# job runs test/makie_tests.jl there.
makie_loaded = try
    @eval using CairoMakie
    true
catch
    false
end
if makie_loaded
    include("makie_tests.jl")
else
    println("Makie backend not loadable in this environment — extension tests " *
            "SKIPPED. CairoMakie is not in the test target, so Pkg.test always " *
            "skips these: run test/makie_tests.jl from the docs environment, " *
            "and under mpiexec for the decomposition-independent profile.")
    @testset "Makie extension: skipped" begin
        @test makie_loaded skip=!(:makie in REQUIRED_SUITES)
    end
end

