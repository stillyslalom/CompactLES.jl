module TransportTests

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

using CompactLES
using CompactLES: compute_rhs!, max_rate, CPUBackend, padded_index, xcoord
using Test
import KernelAbstractions

const CL = CompactLES
const PER = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

struct VariableTransport <: AbstractTransport{Float64}
    k0::Float64
    thermal_slope::Float64
    D0::Float64
    species_slope::Float64
end

function CL.transport_coefficients(tr::VariableTransport, eos, T, rho, cp,
                                   Y::NTuple{N}) where {N}
    return (mu=0.0, kappa=tr.k0*(1 + tr.thermal_slope*T),
            D=ntuple(k -> tr.D0*(1 + tr.species_slope*Y[k]), N))
end

struct ManufacturedTransportSource
    mode::Symbol
    base::Float64
    amplitude::Float64
    thermal_slope::Float64
    k0::Float64
    species_slope::Float64
    D0::Float64
end

function CL.add_source!(src::ManufacturedTransportSource, solver, dQ, Q, t)
    decay = exp(-t)
    for i in 1:solver.decomp.n_local[1]
        x = xcoord(solver, 1, i)
        c, s = cospi(2x), sinpi(2x)
        value = src.base + src.amplitude*decay*c
        vx = -2pi*src.amplitude*decay*s
        vxx = -4pi^2*src.amplitude*decay*c
        vt = -src.amplitude*decay*c
        I = padded_index(solver, i, 1, 1)
        if src.mode === :thermal
            divflux = src.k0*((1 + src.thermal_slope*value)*vxx +
                              src.thermal_slope*vx^2)
            cv = solver.eos.cvk[1]
            dQ[I, solver.equations.i_mom[1]] += vx
            dQ[I, solver.equations.i_energy] += cv*vt - divflux
        else
            Deff = src.D0*(1 + 2src.species_slope*value*(1-value))
            dDeff = 2src.D0*src.species_slope*(1-2value)
            divflux = Deff*vxx + dDeff*vx^2
            forcing = vt - divflux
            dQ[I, 1] += forcing
            dQ[I, 2] -= forcing
        end
    end
    return dQ
end

fit(T, c; scale=1.0) = scale * exp(c[1]*log(T) + c[2]/T + c[3]/T^2 + c[4])

function wilke(X, P, R, phi_property=P)
    sum(eachindex(X)) do i
        denominator = sum(eachindex(X)) do j
            phi = (1 + sqrt(phi_property[i]/phi_property[j]) * (R[i]/R[j])^0.25)^2 /
                  sqrt(8 * (1 + R[j]/R[i]))
            X[j] * phi
        end
        X[i] * P[i] / denominator
    end
end

function reference_coefficients(T, rho, cp, Y, R; Dref=nothing,
                                Tref=300.0, pref=101325.0, exponent=1.75)
    # Literal 200--1000 K rows from data/trans.inp.  This calculation is kept
    # independent of the reader and the package's fit/mixture helpers.
    mu_fit = ((0.74553182, 43.555109, -3257.9340, 0.13556243),
              (0.62526577, -31.779652, -1640.7983, 1.7454992))
    k_fit = ((1.0059461, 279.51262, -29792.018, 1.1996252),
             (0.85439436, 105.73224, -12347.848, 0.47793128))
    pure_mu = ntuple(i -> fit(T, mu_fit[i]; scale=1e-7), 2)
    # The conductivity fits produce microwatts/(cm K), hence 1e-4 W/(m K).
    pure_k = ntuple(i -> fit(T, k_fit[i]; scale=1e-4), 2)
    yr = Y[1]*R[1] + Y[2]*R[2]
    X = (Y[1]*R[1]/yr, Y[2]*R[2]/yr)
    mu = wilke(X, pure_mu, R)
    kappa = wilke(X, pure_k, R, pure_mu)
    if Dref === nothing
        D = ntuple(_ -> kappa/(rho*cp), 2)
    else
        p = rho*T*yr
        D12 = Dref * (T/Tref)^exponent * (pref/p)
        D = (D12, D12)
    end
    return (; mu, kappa, D)
end

function reference_mass_diffusion(T, rho, Y, R, Dref;
                                  Tref=300.0, pref=101325.0, exponent=1.75)
    yr = sum(Y[k]*R[k] for k in eachindex(Y))
    X = ntuple(k -> Y[k]*R[k]/yr, length(Y))
    scale = (T/Tref)^exponent * pref/(rho*T*yr)
    return ntuple(length(Y)) do i
        sx = 0.0; sxw = 0.0; sxwd = 0.0
        for j in eachindex(Y)
            j == i && continue
            Dij = Dref[i, j]*scale
            sx += X[j]/Dij
            xw = X[j]/R[j]
            sxw += xw
            sxwd += xw/Dij
        end
        inv(sx + X[i]*sxwd/sxw)
    end
end

@testset "CEA transport coefficients" begin
    records = read_cea_transport()
    @test count(r -> isempty(r.names[2]), records) == 66
    @test count(r -> !isempty(r.names[2]), records) == 41
    h2_record = only(filter(r -> r.names == ("H2", ""), records))
    @test length(h2_record.viscosity) == 3
    @test (h2_record.viscosity[1].Tmin, h2_record.viscosity[1].Tmax,
           h2_record.viscosity[1].A, h2_record.viscosity[1].B) ==
          (200.0, 1000.0, 0.74553182, 43.555109)

    # Ar spans all three intervals.  Values on the middle and high rows catch
    # a reader or selector that silently reuses the first interval.
    ar = Nasa9Mixture(["Ar"])
    ar_tr = CeaTransport(ar)
    for (temperature, mu_row, k_row) in (
            (2000.0, (0.69357334, 70.953943, -28386.007, 1.4856447),
                     (0.69075463, 62.676058, -25667.413, 1.2664189)),
            (8000.0, (0.76608935, 678.67215, -849914.17, 0.77935167),
                     (0.76269502, 623.41752, -718995.52, 0.56927918)))
        cp_ar = CL.species_cp(ar, 1, temperature)
        coeff = transport_coefficients(ar_tr, ar, temperature, 1.0, cp_ar, (1.0,))
        @test coeff.mu ≈ fit(temperature, mu_row; scale=1e-7) rtol=2e-13
        @test coeff.kappa ≈ fit(temperature, k_row; scale=1e-4) rtol=2e-13
    end

    mktemp() do path, io
        write(io, "transport property coefficients\n")
        write(io, rpad("BAD", 34), "V1C0\n V  300.0\nend\n")
        close(io)
        @test_throws ArgumentError read_cea_transport(path)
    end
    mktemp() do path, io
        write(io, "transport property coefficients\n")
        write(io, rpad("BAD", 34), "V2C0\n")
        write(io, " V  300.0   1000.0   0.10000000E+01 0.00000000E+00 0.00000000E+00 0.00000000E+00\n")
        close(io)
        @test_throws ArgumentError read_cea_transport(path)
    end

    eos = Nasa9Mixture(["H2", "N2"])
    T, rho, Y = 500.0, 0.7, (0.3, 0.7)
    cp = sum(Y[k] * CL.species_cp(eos, k, T) for k in 1:2)
    R = Tuple(eos.Rk)
    tr = CeaTransport(eos)
    got = @inferred transport_coefficients(tr, eos, T, rho, cp, Y)
    ref = reference_coefficients(T, rho, cp, Y, R)
    @test got.mu ≈ ref.mu rtol=2e-13
    @test got.kappa ≈ ref.kappa rtol=2e-13
    @test all(isapprox(got.D[k], ref.D[k]; rtol=2e-13) for k in 1:2)

    # Pure limits select the database row itself.  A trace constituent must
    # approach that same limit rather than polluting it through a mass/mole
    # fraction mix-up.
    for (Ypure, sp) in (((1.0, 0.0), 1), ((0.0, 1.0), 2))
        cp_pure = CL.species_cp(eos, sp, T)
        got_pure = transport_coefficients(tr, eos, T, rho, cp_pure, Ypure)
        ref_pure = reference_coefficients(T, rho, cp_pure, Ypure, R)
        @test got_pure.mu ≈ ref_pure.mu rtol=2e-13
        @test got_pure.kappa ≈ ref_pure.kappa rtol=2e-13
    end
    trace = transport_coefficients(tr, eos, T, rho,
        (1-1e-12)*CL.species_cp(eos, 1, T) + 1e-12*CL.species_cp(eos, 2, T),
        (1-1e-12, 1e-12))
    pure = transport_coefficients(tr, eos, T, rho, CL.species_cp(eos, 1, T),
                                  (1.0, 0.0))
    @test trace.mu ≈ pure.mu rtol=2e-11
    @test trace.kappa ≈ pure.kappa rtol=2e-11

    eos_rev = Nasa9Mixture(["N2", "H2"])
    tr_rev = CeaTransport(eos_rev)
    rev = transport_coefficients(tr_rev, eos_rev, T, rho, cp, reverse(Y))
    @test rev.mu ≈ got.mu rtol=2e-13
    @test rev.kappa ≈ got.kappa rtol=2e-13
    @test all(isapprox(rev.D[3-k], got.D[k]; rtol=2e-13) for k in 1:2)
end

@testset "binary mixture-averaged and unity-Lewis diffusion" begin
    eos = Nasa9Mixture(["H2", "N2"])
    T, Y, p = 620.0, (0.4, 0.6), 2.3e5
    R = Tuple(eos.Rk)
    rho = p/(T*sum(Y[k]*R[k] for k in 1:2))
    cp = sum(Y[k] * CL.species_cp(eos, k, T) for k in 1:2)
    Dref = 7.25e-5
    binary = BinaryDiffusion([0.0 Dref; Dref 0.0])
    tr = CeaTransport(eos; diffusion=:mixture_averaged,
                      binary_diffusion=binary)
    got = @inferred transport_coefficients(tr, eos, T, rho, cp, Y)
    ref = reference_coefficients(T, rho, cp, Y, R; Dref)
    @test all(isapprox(got.D[k], ref.D[k]; rtol=3e-13) for k in 1:2)

    scale = (T/300)^1.75 * 101325/p
    D12 = Dref*scale
    exact_pure = transport_coefficients(tr, eos, T, p/(T*R[1]),
                                        CL.species_cp(eos, 1, T), (1.0, 0.0)).D
    @test exact_pure[1] == 0.0
    @test exact_pure[2] ≈ D12 rtol=3e-13
    near_Y = (1-1e-12, 1e-12)
    near_rho = p/(T*sum(near_Y[k]*R[k] for k in 1:2))
    near_cp = sum(near_Y[k]*CL.species_cp(eos, k, T) for k in 1:2)
    near = transport_coefficients(tr, eos, T, near_rho, near_cp, near_Y).D
    @test all(isfinite, near)
    @test all(isapprox(d, D12; rtol=3e-13) for d in near)

    eos_rev = Nasa9Mixture(["N2", "H2"])
    tr_rev = CeaTransport(eos_rev; diffusion=:mixture_averaged,
                          binary_diffusion=binary)
    rev = transport_coefficients(tr_rev, eos_rev, T, rho, cp, reverse(Y))
    @test all(isapprox(rev.D[3-k], got.D[k]; rtol=3e-13) for k in 1:2)

    # Unequal ternary coefficients distinguish the mass-gradient Cantera
    # formula from the common but different (1-Yk)/sum(Xj/Dkj) expression.
    eos3 = Nasa9Mixture(["H2", "N2", "O2"])
    Y3 = (0.2, 0.3, 0.5)
    R3 = Tuple(eos3.Rk)
    rho3 = p/(T*sum(Y3[k]*R3[k] for k in 1:3))
    cp3 = sum(Y3[k]*CL.species_cp(eos3, k, T) for k in 1:3)
    Dmat = [0.0 7.25e-5 8.1e-5; 7.25e-5 0.0 2.0e-5;
            8.1e-5 2.0e-5 0.0]
    tr3 = CeaTransport(eos3; diffusion=:mixture_averaged,
                       binary_diffusion=BinaryDiffusion(Dmat))
    got3 = transport_coefficients(tr3, eos3, T, rho3, cp3, Y3).D
    ref3 = reference_mass_diffusion(T, rho3, Y3, R3, Dmat)
    @test all(isapprox(got3[k], ref3[k]; rtol=3e-13) for k in 1:3)

    Dequal = 4.2e-5
    equal_matrix = [0.0 Dequal Dequal; Dequal 0.0 Dequal;
                    Dequal Dequal 0.0]
    equal_tr = CeaTransport(eos3; diffusion=:mixture_averaged,
                            binary_diffusion=BinaryDiffusion(equal_matrix))
    equal_D = transport_coefficients(equal_tr, eos3, T, rho3, cp3, Y3).D
    equal_scaled = Dequal*(T/300)^1.75*101325/p
    @test all(isapprox(d, equal_scaled; rtol=3e-13) for d in equal_D)
    @test_throws ArgumentError CeaTransport(eos; diffusion=:mixture_averaged)
    @test_throws ArgumentError BinaryDiffusion([0.0 1.0; 2.0 0.0])
end

@testset "CEA transport reaches solver construction and runtime paths" begin
    eos = Nasa9Mixture(["H2", "N2"])
    tr = CeaTransport(eos)
    ic(x, y, z) = Prim(Y=(0.35, 0.65), rho=0.8 + 0.02sinpi(2x),
                       T_ion=500 + 10cospi(2x))
    problem = Problem(eos=eos, transport=tr,
                      domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                      bcs=PER, ic=ic)
    solver, Q = setup(problem, Numerics(n_global=(16, 1, 1),
                                        art=ArtificialProperties(enabled=false),
                                        filter_interval=0))
    @test solver.transport isa CeaTransport
    @test isfinite(compute_dt(solver, Q))
    compute_rhs!(solver, Q, zero(Q))

    patched = Solver(n_global=(32, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
                     eos=eos, transport=tr, patch_grid=(2, 1, 1),
                     art=ArtificialProperties(enabled=false))
    @test patched.transport isa CeaTransport
    @test length(patched.patches) == 2

    refined = Solver(n_global=(32, 16, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
                     eos=eos, transport=tr,
                     refine=BlockRegion((8, 4, 0), (16, 8, 1)),
                     art=ArtificialProperties(enabled=false))
    @test refined.transport isa CeaTransport
    @test length(refined.patches) > 1
end

@testset "variable thermal and corrected species diffusion reach the RHS" begin
    n = 64
    tr = VariableTransport(2e-3, 0.35, 3e-3, 0.8)

    eos1 = IdealMixture(IdealSpecies("gas"; R=1.0, gamma=1.4))
    st = Solver(n_global=(n, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
                eos=eos1, transport=tr, art=ArtificialProperties(enabled=false),
                filter_interval=0)
    Qt = allocate_state(st)
    base, amp = 1.2, 0.08
    initialize!(st, Qt, (x, y, z) -> Prim(rho=1.0, T_ion=base + amp*cospi(2x)))
    dQt = zero(Qt)
    compute_rhs!(st, Qt, dQt)
    ie = st.equations.i_energy
    thermal_error = maximum(1:n) do i
        x = xcoord(st, 1, i)
        temp = base + amp*cospi(2x)
        Tx = -2pi*amp*sinpi(2x)
        Txx = -4pi^2*amp*cospi(2x)
        exact = tr.k0*((1 + tr.thermal_slope*temp)*Txx +
                       tr.thermal_slope*Tx^2)
        abs(dQt[padded_index(st, i, 1, 1), ie] - exact)
    end
    @test thermal_error < 2e-7

    # Equal thermodynamics isolate the corrected mass-gradient flux.  Since
    # D1 and D2 vary differently with Y, this also checks the correction
    # velocity rather than only a scalar variable-coefficient Laplacian.
    eos2 = IdealMixture((IdealSpecies("a"; R=1.0, gamma=1.4),
                         IdealSpecies("b"; R=1.0, gamma=1.4)))
    ss = Solver(n_global=(n, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
                eos=eos2, transport=tr, art=ArtificialProperties(enabled=false),
                filter_interval=0)
    Qs = allocate_state(ss)
    ybase, yamp = 0.45, 0.12
    initialize!(ss, Qs, (x, y, z) -> begin
        y1 = ybase + yamp*cospi(2x)
        Prim(Y=(y1, 1-y1), rho=1.0, p=1.0)
    end)
    dQs = zero(Qs)
    compute_rhs!(ss, Qs, dQs)
    species_error = maximum(1:n) do i
        x = xcoord(ss, 1, i)
        y1 = ybase + yamp*cospi(2x)
        yx = -2pi*yamp*sinpi(2x)
        yxx = -4pi^2*yamp*cospi(2x)
        Deff = tr.D0*(1 + 2tr.species_slope*y1*(1-y1))
        dDeff = 2tr.D0*tr.species_slope*(1-2y1)
        exact = Deff*yxx + dDeff*yx^2
        abs(dQs[padded_index(ss, i, 1, 1), 1] - exact)
    end
    @test species_error < 2e-7
end

@testset "variable-coefficient manufactured diffusion evolution" begin
    tr = VariableTransport(2e-3, 0.35, 3e-3, 0.8)
    tf = 0.01
    thermal_errors = Float64[]
    species_errors = Float64[]
    for n in (32, 64)
        eos1 = IdealMixture(IdealSpecies("gas"; R=1.0, gamma=1.4))
        tsrc = ManufacturedTransportSource(:thermal, 1.2, 0.08,
                                            tr.thermal_slope, tr.k0, 0, 0)
        st = Solver(n_global=(n, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
                    eos=eos1, transport=tr, sources=(tsrc,), cfl=0.15,
                    art=ArtificialProperties(enabled=false), filter_interval=0)
        Qt = allocate_state(st)
        initialize!(st, Qt, (x, y, z) -> Prim(rho=1.0,
                    T_ion=tsrc.base + tsrc.amplitude*cospi(2x)))
        run!(st, Qt; tfinal=tf, nmax=1000)
        CL.refresh_primitives!(st, Qt)
        push!(thermal_errors, maximum(1:n) do i
            exact = tsrc.base + tsrc.amplitude*exp(-tf)*cospi(2xcoord(st, 1, i))
            abs(st.T_ion[padded_index(st, i, 1, 1)] - exact)
        end)

        eos2 = IdealMixture((IdealSpecies("a"; R=1.0, gamma=1.4),
                             IdealSpecies("b"; R=1.0, gamma=1.4)))
        ssrc = ManufacturedTransportSource(:species, 0.45, 0.12, 0, 0,
                                            tr.species_slope, tr.D0)
        ss = Solver(n_global=(n, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
                    eos=eos2, transport=tr, sources=(ssrc,), cfl=0.15,
                    art=ArtificialProperties(enabled=false), filter_interval=0)
        Qs = allocate_state(ss)
        initialize!(ss, Qs, (x, y, z) -> begin
            y1 = ssrc.base + ssrc.amplitude*cospi(2x)
            Prim(Y=(y1, 1-y1), rho=1.0, p=1.0)
        end)
        run!(ss, Qs; tfinal=tf, nmax=1000)
        push!(species_errors, maximum(1:n) do i
            exact = ssrc.base + ssrc.amplitude*exp(-tf)*cospi(2xcoord(ss, 1, i))
            abs(Qs[padded_index(ss, i, 1, 1), 1] - exact)
        end)
    end
    println("manufactured variable transport: thermal errors = ", thermal_errors,
            ", species errors = ", species_errors)
    @test thermal_errors[2] < thermal_errors[1]/8
    @test species_errors[2] < species_errors[1]/8
    @test thermal_errors[2] < 2e-8
    @test species_errors[2] < 2e-8
end

@testset "absolute molecular diffusion timestep" begin
    eos = IdealMixture((IdealSpecies("a"; R=1.0, gamma=1.4),
                        IdealSpecies("b"; R=1.0, gamma=1.4)))
    tr = VariableTransport(0.0, 0.0, 2.5, 0.0)
    s = Solver(n_global=(24, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=PER,
               eos=eos, transport=tr, cfl=0.4,
               art=ArtificialProperties(enabled=false), filter_interval=0)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(Y=(0.3, 0.7), rho=1.0, T_ion=1.0))
    h = s.h[1]
    sound = sqrt(1.4)
    expected_rate = sound/h + 2tr.D0/h^2
    rate, _ = max_rate(s, Q)
    @test rate ≈ expected_rate rtol=2e-14
    @test compute_dt(s, Q) ≈ s.cfl/expected_rate rtol=2e-14
    @test 2tr.D0/h^2 > 20sound/h
end

@testset "typed transport and EOS validation" begin
    eos32 = Nasa9Mixture(Float32, ("H2", "N2"))
    tr32 = CeaTransport(eos32; Lewis=Float32(1.2))
    @test isbits(tr32)
    cp = sum(Float32(0.5)*CL.species_cp(eos32, k, Float32(500)) for k in 1:2)
    coeff = @inferred transport_coefficients(tr32, eos32, Float32(500), Float32(1), cp,
                                             (Float32(0.5), Float32(0.5)))
    @test coeff.mu isa Float32
    @test coeff.kappa isa Float32
    @test coeff.D isa NTuple{2,Float32}
    # Public queries accept integer temperature literals without attempting
    # to convert the SI fit scale factors to integers.
    integer_temperature = @inferred transport_coefficients(
        tr32, eos32, 500, Float32(1), cp, (Float32(0.5), Float32(0.5)))
    @test integer_temperature == coeff

    binary32 = BinaryDiffusion(Float32[0 7e-5; 7e-5 0];
        temperature_ref=Float32(300), pressure_ref=Float32(101325),
        temperature_exponent=Float32(1.75))
    tr_binary32 = CeaTransport(eos32; diffusion=:mixture_averaged,
                               binary_diffusion=binary32)
    binary_coeff = @inferred transport_coefficients(
        tr_binary32, eos32, Float32(500), Float32(1), cp,
        (Float32(0.5), Float32(0.5)))
    @test binary_coeff.D isa NTuple{2,Float32}

    ar_from_species = CeaTransport(IdealSpecies("Ar"))
    @test transport_coefficients(ar_from_species, IdealMixture(IdealSpecies("Ar")),
                                 500.0, 1.0, 2.5, (1.0,)).mu > 0

    tr = CeaTransport(Nasa9Mixture(["H2", "N2"]))
    @test_throws ArgumentError Solver(n_global=(12, 1, 1),
        L_domain=(1.0, 1.0, 1.0), bcs=PER,
        eos=Nasa9Mixture(["N2", "H2"]), transport=tr)
    @test_throws ArgumentError Solver(n_global=(12, 1, 1),
        L_domain=(1.0, 1.0, 1.0), bcs=PER,
        eos=Nasa9Mixture(["H2"]), transport=tr)
end

@testset "CEA timestep agrees on host and KernelAbstractions paths" begin
    eos = IdealMixture(["H2", "N2"])
    tr = CeaTransport(eos)
    function build(backend)
        s = Solver(n_global=(16, 12, 1), L_domain=(1.0, 0.75, 1.0), bcs=PER,
                   eos=eos, transport=tr, backend=backend,
                   art=ArtificialProperties(enabled=false), filter_interval=0)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(Y=(0.2 + 0.1sinpi(2x),
                                                    0.8 - 0.1sinpi(2x)),
                                             rho=0.9, T_ion=450 + 30cospi(2y/0.75)))
        return s, Q
    end
    sh, Qh = build(CPUBackend())
    sd, Qd = build(DeviceBackend(KernelAbstractions.CPU()))
    CL.FORCE_KA[] = true
    dt_device = try
        compute_dt(sd, Qd)
    finally
        CL.FORCE_KA[] = false
    end
    @test dt_device == compute_dt(sh, Qh)
end

end # module
