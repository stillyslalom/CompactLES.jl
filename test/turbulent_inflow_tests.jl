# Synthetic turbulent inflow: the statistics of the random Fourier-mode field
# against the prescribed stresses and integral scale, its determinism, the
# cost of a call, and short runs with it as an NSCBC and a Dirichlet target.
#
# Serial. Included by runtests.jl, and runnable on its own:
#
#   julia --project=. test/turbulent_inflow_tests.jl

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES: apply_bcs!, padded_index, xcoord
using Test

const TI = CompactLES
const TI_MEAN = Prim(u=(1.0, 0.0, 0.0), p=1.0, T_ion=1.0)
const TI_L = 0.2

ti_fluct(f, x, y, z, t) = TI._inflow_fluctuation(f, x, y, z, t)

# Inside a function, so that the returned `Prim` is not boxed at top level.
function ti_allocated(f, x, y, z, t)
    f(x, y, z, t)
    return @allocated f(x, y, z, t)
end

# Covariance of the fluctuation over a face x = 0 and a time window: 16 × 16
# face points 2.5 L apart, 200 instants L / U apart.
function ti_face_covariance(f)
    R = zeros(3, 3)
    m = zeros(3)
    n = 0
    for it in 1:200, j in 1:16, k in 1:16
        u = ti_fluct(f, 0.0, 2.5TI_L * j, 2.5TI_L * k, TI_L * it)
        for a in 1:3
            m[a] += u[a]
            for b in 1:3
                R[a, b] += u[a] * u[b]
            end
        end
        n += 1
    end
    return m ./ n, R ./ n
end

# The correlation of u'_1 at separation r along x expected over realizations
# of a mode set: each wave vector's direction is uniform on the sphere, which
# gives 3 (sin s - s cos s) / s³ at s = |k| r per mode.
function ti_expected_correlation(f, r)
    num = den = 0.0
    for m in f.modes
        w = sum(abs2, m.a) + sum(abs2, m.b)
        s = sqrt(sum(abs2, m.k)) * r
        num += w * (s < 1e-6 ? 1.0 : 3 * (sin(s) - s * cos(s)) / s^3)
        den += w
    end
    return num / den
end

# The correlation of u'_1 of one realization, in space along x and in time at
# a fixed point, from the mode table.
function ti_realized_correlation(f, r; time=false)
    num = den = 0.0
    for m in f.modes
        w = (m.a[1]^2 + m.b[1]^2) / 2
        freq = time ? m.omega - sum(m.k .* f.convection) : m.k[1]
        num += w * cos(freq * r)
        den += w
    end
    return num / den
end

@testset "turbulent inflow: construction" begin
    @test_throws ArgumentError TurbulentInflow(TI_MEAN; length_scale=TI_L)
    @test_throws ArgumentError TurbulentInflow(TI_MEAN; length_scale=TI_L,
                                               intensity=0.1, n_modes=100)
    @test_throws ArgumentError TurbulentInflow(Prim(p=1.0, T_ion=1.0);
                                               length_scale=TI_L, intensity=0.1)
    @test_throws ArgumentError TurbulentInflow(TI_MEAN; length_scale=TI_L,
                                               reynolds_stress=[1 0 0; 0 -1 0; 0 0 1])
    @test_throws ArgumentError TurbulentInflow(TI_MEAN; length_scale=TI_L,
                                               reynolds_stress=[1 0.5 0; 0 1 0; 0 0 1])
    @test_throws ArgumentError TurbulentInflow(TI_MEAN; length_scale=TI_L,
                                               intensity=0.1, min_wavelength=0.5)
    # A component without fluctuation, as in a planar run.
    planar = TurbulentInflow(TI_MEAN; length_scale=TI_L,
                             reynolds_stress=[0.01 -0.003 0; -0.003 0.01 0; 0 0 0])
    @test all(m -> m.a[3] == 0 && m.b[3] == 0, planar.modes)
    f = TurbulentInflow(TI_MEAN; length_scale=TI_L, intensity=0.1)
    @test length(f.modes) == 192
    # The integral scale of the shells is the prescribed one.
    w = [sum(abs2, m.a) + sum(abs2, m.b) for m in f.modes]
    kmag = [sqrt(sum(abs2, m.k)) for m in f.modes]
    @test sum(w .* 3π ./ (4 .* kmag)) / sum(w) ≈ TI_L rtol = 1e-10
    @test maximum(kmag) ≈ 2π / (TI_L / 2) rtol = 1e-12
end

@testset "turbulent inflow: determinism, inference, allocation" begin
    f = TurbulentInflow(TI_MEAN; length_scale=TI_L, intensity=0.1, seed=7)
    g = TurbulentInflow(TI_MEAN; length_scale=TI_L, intensity=0.1, seed=7)
    h = TurbulentInflow(TI_MEAN; length_scale=TI_L, intensity=0.1, seed=8)
    pts = [(0.1i, 0.37i, -0.2i, 0.05i) for i in 1:20]
    @test all(f(p...) === g(p...) for p in pts)
    @test all(f(p...).u != h(p...).u for p in pts)
    pr = f(0.1, 0.2, 0.3, 0.4)
    @test pr isa Prim{1}
    @test pr.p == TI_MEAN.p && pr.T_ion == TI_MEAN.T_ion && isnan(pr.rho)
    @test (@inferred f(0.1, 0.2, 0.3, 0.4)) isa Prim{1}
    @test (@inferred f(0.1f0, 0.2f0, 0.3f0, 0.4f0)) isa Prim{1}
    @test ti_allocated(f, 0.1, 0.2, 0.3, 0.4) == 0
    @test ti_allocated(f, 0.1f0, 0.2f0, 0.3f0, 0.4f0) == 0
    # Without convection the field at a point moves only by the frequencies.
    still = TurbulentInflow(TI_MEAN; length_scale=TI_L, intensity=0.1, seed=7,
                            convect=false)
    @test still.convection == (0.0, 0.0, 0.0)
    @test still(0.3, 0.2, 0.1, 0.0) === f(0.3, 0.2, 0.1, 0.0)
end

@testset "turbulent inflow: Reynolds stresses" begin
    # Sampling tolerance: the window holds about 200 L/U by 40 L by 40 L, so
    # the relative error of a second moment is of order a few percent.
    σ2 = 0.01
    f = TurbulentInflow(TI_MEAN; length_scale=TI_L, intensity=0.1)
    m, R = ti_face_covariance(f)
    @test maximum(abs, m) / sqrt(σ2) < 0.05
    @test maximum(abs, R ./ σ2 - [1 0 0; 0 1 0; 0 0 1]) < 0.06
    target = [0.02 -0.006 0.001; -0.006 0.01 0.0; 0.001 0.0 0.008]
    g = TurbulentInflow(TI_MEAN; length_scale=TI_L, reynolds_stress=target, seed=3)
    m, R = ti_face_covariance(g)
    scale = sqrt.([target[a, a] * target[b, b] for a in 1:3, b in 1:3])
    @test maximum(abs, (R - target) ./ scale) < 0.06
    # The long-time covariance of the mode table is the prescribed tensor.
    exact = sum([(m.a[a] * m.a[b] + m.b[a] * m.b[b]) / 2 for a in 1:3, b in 1:3]
                for m in g.modes)
    @test exact ≈ target rtol = 1e-12
end

@testset "turbulent inflow: two-point correlation and integral scale" begin
    rs = range(0, 3TI_L, length=13)
    # One realization: the sampled correlation along x is the mode table's.
    f = TurbulentInflow(TI_MEAN; length_scale=TI_L, intensity=0.1, seed=2)
    C = zeros(length(rs))
    for j in 1:40, k in 1:40
        y, z = 2.5TI_L * j, 2.5TI_L * k + 0.3j
        u0 = ti_fluct(f, 0.0, y, z, 0.0)[1]
        for (i, r) in enumerate(rs)
            C[i] += u0 * ti_fluct(f, r, y, z, 0.0)[1]
        end
    end
    C ./= C[1]
    @test maximum(abs, C .- ti_realized_correlation.(Ref(f), rs)) < 0.1
    # Over realizations, the correlation in x and the correlation in time at a
    # point, at separation U τ, follow the expected curve, whose integral is
    # `length_scale`. One realization scatters about it as 1/sqrt(n_modes).
    seeds = 1:24
    space = zeros(length(rs))
    time = zeros(length(rs))
    for s in seeds
        g = TurbulentInflow(TI_MEAN; length_scale=TI_L, intensity=0.1, seed=s)
        space .+= ti_realized_correlation.(Ref(g), rs)
        time .+= ti_realized_correlation.(Ref(g), rs; time=true)
    end
    expected = ti_expected_correlation.(Ref(f), rs)
    @test maximum(abs, space ./ length(seeds) .- expected) < 0.08
    @test maximum(abs, time ./ length(seeds) .- expected) < 0.08
end

@testset "turbulent inflow: divergence" begin
    # Isotropic: every mode is normal to its wave vector, so the field is
    # divergence free to the round-off of the difference quotient. The
    # Cholesky factor of an anisotropic tensor breaks that.
    function divergence_ratio(f)
        ε = 1e-5
        dmax = gmax = 0.0
        for i in 1:50
            x, y, z, t = 0.13i, 0.07i, 0.05i, 0.01i
            d1 = (ti_fluct(f, x + ε, y, z, t)[1] - ti_fluct(f, x - ε, y, z, t)[1]) / 2ε
            d2 = (ti_fluct(f, x, y + ε, z, t)[2] - ti_fluct(f, x, y - ε, z, t)[2]) / 2ε
            d3 = (ti_fluct(f, x, y, z + ε, t)[3] - ti_fluct(f, x, y, z - ε, t)[3]) / 2ε
            dmax = max(dmax, abs(d1 + d2 + d3))
            gmax = max(gmax, abs(d1), abs(d2), abs(d3))
        end
        return dmax / gmax
    end
    iso = TurbulentInflow(TI_MEAN; length_scale=TI_L, intensity=0.1)
    @test divergence_ratio(iso) < 1e-6
    aniso = TurbulentInflow(TI_MEAN; length_scale=TI_L,
                            reynolds_stress=[0.02 -0.006 0; -0.006 0.01 0; 0 0 0.008])
    @test divergence_ratio(aniso) > 1e-2
end

@testset "turbulent inflow: NSCBC and Dirichlet targets in a run" begin
    eos = IdealSpecies("gas"; gamma=1.4, R=1.0)
    mean = Prim(u=(0.3, 0.0, 0.0), p=1.0, T_ion=1.0)
    per = (PeriodicBC(), PeriodicBC())
    turb = TurbulentInflow(mean; length_scale=0.1, intensity=0.1)
    function runwith(inlet)
        s = Solver(n_global=(16, 12, 12), L_domain=(0.5, 0.4, 0.4),
                   bcs=((inlet, NSCBCOutflowBC(pinf=1.0)), per, per), eos=eos,
                   art=ArtificialProperties(enabled=false), cfl=0.4)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> mean)
        run!(s, Q; tfinal=1e9, nmax=4)
        return s, Q
    end
    steady, Qs = runwith(NSCBCInflowBC(mean))
    s, Q = runwith(NSCBCInflowBC(mean; target=turb, eta_u=5.0, eta_T=5.0))
    inner = TI.interior(s.decomp)
    @test all(isfinite, parent(Q)[inner, :])
    # The uniform stream stays uniform; the turbulent one has taken up a
    # transverse velocity at the inflow face.
    face = [s.v[padded_index(s, 1, j, k)] for j in 1:12, k in 1:12]
    @test maximum(abs, face) > 1e-3
    @test maximum(abs,
                  [steady.v[padded_index(steady, 1, j, k)] for j in 1:12, k in 1:12]) < 1e-12
    # A checkpoint records the construction arguments, not the mode table.
    rec = TI.configuration_record(s)
    @test any(endswith(".target.seed"), rec.paths)
    @test !any(occursin(".modes"), rec.paths)
    d, Qd = runwith(DirichletBC(turb))
    @test all(isfinite, parent(Qd)[inner, :])
    # The Dirichlet face holds the target at the time it is enforced.
    d.tstage = d.t
    apply_bcs!(d, Qd)
    im = d.equations.i_mom
    err = maximum(abs(Qd[padded_index(d, 1, j, k), im[2]] / Qd[padded_index(d, 1, j, k), 1] -
                      turb(xcoord(d, 1, 1), xcoord(d, 2, j), xcoord(d, 3, k), d.t).u[2])
                  for j in 1:12, k in 1:12)
    @test err < 1e-12
end
