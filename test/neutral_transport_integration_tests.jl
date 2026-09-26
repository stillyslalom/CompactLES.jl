module NeutralTransportIntegrationTests

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES, Test
using CompactLES: compute_rhs!, padded_index, xcoord

const CL = CompactLES
const PER = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
const T0 = 300.0
const P0 = 101325.0
const L = 0.01
const AMP = 0.08
const YBAR = 0.45

# Synthetic equal thermodynamics isolate ordinary binary diffusion: composition changes
# neither rho, p, T nor species enthalpy, so a binary cosine is an exact scalar
# diffusion problem. This is a numerical closure test, not a physical H2-D2
# experiment. The labels still select the independently sourced H2-D2
# Marrero--Mason correlation and the corresponding CEA transport records.
function isotope_eos(order=("H2", "D2"))
    species = ntuple(i -> IdealSpecies(order[i]; R=1e-4, gamma=1.4), 2)
    return IdealMixture(species)
end

function loschmidt_case(n; order=("H2", "D2"), reverse_profile=false)
    eos = isotope_eos(order)
    binary = neutral_binary_diffusion(order; temperature_min=250.0,
                                      temperature_max=400.0)
    tr = CeaTransport(eos; diffusion=:mixture_averaged,
                      binary_diffusion=binary)
    s = Solver(n_global=(n, 1, 1), L_domain=(L, 1.0, 1.0), bcs=PER,
               eos=eos, transport=tr, cfl=0.35,
               art=ArtificialProperties(enabled=false), filter_interval=0)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> begin
        y1 = YBAR + AMP*cospi(2x/L)
        reverse_profile && (y1 = 1 - y1)
        Prim(Y=(y1, 1-y1), p=P0, T_ion=T0)
    end)
    return s, Q, binary
end

@testset "neutral H2-D2 Loschmidt diffusion" begin
    tf = 1e-3
    errors = Float64[]
    for n in (24, 48)
        s, Q, binary = loschmidt_case(n)
        D12 = binary_diffusivity(binary, T0, P0, 1, 2)
        rho0 = P0/(T0*1e-4)
        mass0 = sum(Q[padded_index(s, i, 1, 1), 1] for i in 1:n)
        total0 = [sum(Q[padded_index(s, i, 1, 1), k] for i in 1:n) for k in 1:2]

        run!(s, Q; tfinal=tf, nmax=10000)
        @test s.t == tf
        decay = exp(-D12*(2pi/L)^2*tf)
        push!(errors, maximum(1:n) do i
            exact = rho0*(YBAR + AMP*decay*cospi(2xcoord(s, 1, i)/L))
            abs(Q[padded_index(s, i, 1, 1), 1] - exact)/rho0
        end)

        # Periodicity plus the correction velocity conserves each species and
        # keeps the equal-thermodynamics pressure and temperature uniform.
        @test sum(Q[padded_index(s, i, 1, 1), 1] for i in 1:n) ≈ mass0 rtol=2e-14
        @test all(isapprox(sum(Q[padded_index(s, i, 1, 1), k] for i in 1:n), total0[k];
                           rtol=2e-14) for k in 1:2)
        CL.refresh_primitives!(s, Q)
        @test maximum(abs(s.p[padded_index(s, i, 1, 1)] - P0) for i in 1:n) < 2e-9
        @test maximum(abs(s.T_ion[padded_index(s, i, 1, 1)] - T0) for i in 1:n) < 2e-11
    end
    @info "neutral Loschmidt diffusion errors" errors
    @test log2(errors[1]/errors[2]) > 5.5
    @test errors[2] < 2e-7
end

@testset "neutral diffusion respects species identity and model limit" begin
    # Reordering both the EOS and the labelled fit must only reorder the
    # conserved species fields.
    s, Q, _ = loschmidt_case(32)
    sr, Qr, _ = loschmidt_case(32; order=("D2", "H2"), reverse_profile=true)
    run!(s, Q; tfinal=5e-4, nmax=10000)
    run!(sr, Qr; tfinal=5e-4, nmax=10000)
    @test maximum(abs(Q[padded_index(s, i, 1, 1), 1] -
                      Qr[padded_index(sr, i, 1, 1), 2]) for i in 1:32) < 2e-9
    @test maximum(abs(Q[padded_index(s, i, 1, 1), 2] -
                      Qr[padded_index(sr, i, 1, 1), 1]) for i in 1:32) < 2e-9

    # A degree-one polynomial with coefficient 1.75 is exactly the legacy
    # BinaryDiffusion power law.  Comparing complete RHS evaluations exercises
    # model dispatch inside the solver rather than only the public evaluator.
    eos = isotope_eos()
    Dref = 7.25e-5
    direct = BinaryDiffusion([0.0 Dref; Dref 0.0])
    polynomial = BinaryDiffusionPolynomial(("H2", "D2"),
        [0.0 Dref; Dref 0.0], reshape([0.0, 1.75, 1.75, 0.0], 2, 2, 1),
        [0.0 200.0; 200.0 0.0], [0.0 1000.0; 1000.0 0.0])
    function rhs_with(model)
        tr = CeaTransport(eos; diffusion=:mixture_averaged,
                          binary_diffusion=model)
        sol = Solver(n_global=(32, 1, 1), L_domain=(L, 1.0, 1.0), bcs=PER,
                     eos=eos, transport=tr, art=ArtificialProperties(enabled=false),
                     filter_interval=0)
        state = allocate_state(sol)
        initialize!(sol, state, (x, y, z) -> begin
            y1 = YBAR + AMP*cospi(2x/L)
            Prim(Y=(y1, 1-y1), p=P0, T_ion=T0)
        end)
        rhs = zero(state)
        compute_rhs!(sol, state, rhs)
        return parent(rhs)
    end
    @test rhs_with(polynomial) ≈ rhs_with(direct) rtol=2e-13 atol=2e-8
end

@testset "physical H2-D2 thermodynamics reaches the conservative RHS" begin
    function physical_rhs(order, reverse_profile)
        eos = Nasa9Mixture(collect(order))
        binary = neutral_binary_diffusion(order; temperature_min=250.0,
                                          temperature_max=400.0)
        tr = CeaTransport(eos; diffusion=:mixture_averaged,
                          binary_diffusion=binary)
        sol = Solver(n_global=(32, 1, 1), L_domain=(L, 1.0, 1.0), bcs=PER,
                     eos=eos, transport=tr, art=ArtificialProperties(enabled=false),
                     filter_interval=0)
        state = allocate_state(sol)
        initialize!(sol, state, (x, y, z) -> begin
            y1 = YBAR + AMP*cospi(2x/L)
            reverse_profile && (y1 = 1-y1)
            Prim(Y=(y1, 1-y1), p=P0, T_ion=T0)
        end)
        rhs = zero(state)
        compute_rhs!(sol, state, rhs)
        return sol, state, rhs
    end

    s, Q, dQ = physical_rhs(("H2", "D2"), false)
    sr, Qr, dQr = physical_rhs(("D2", "H2"), true)
    CL.refresh_primitives!(s, Q)
    CL.refresh_primitives!(sr, Qr)
    @test maximum(abs(s.Y[1][padded_index(s, i, 1, 1)] -
                      sr.Y[2][padded_index(sr, i, 1, 1)]) for i in 1:32) < 2e-15
    @test maximum(abs(s.Y[2][padded_index(s, i, 1, 1)] -
                      sr.Y[1][padded_index(sr, i, 1, 1)]) for i in 1:32) < 2e-15
    @test all(isfinite, parent(dQ))
    @test maximum(abs, view(parent(dQ), :, :, :, s.equations.i_energy)) > 0
    species_scale = maximum(abs(dQ[padded_index(s, i, 1, 1), k])
                            for i in 1:32, k in 1:2)
    @test maximum(abs(dQ[padded_index(s, i, 1, 1), 1] +
                      dQ[padded_index(s, i, 1, 1), 2]) for i in 1:32) < 2e-12*species_scale
    for c in (1, 2, s.equations.i_energy)
        values = [dQ[padded_index(s, i, 1, 1), c] for i in 1:32]
        @test abs(sum(values)) < 2e-12*sum(abs, values)
    end
    @test maximum(abs(dQ[padded_index(s, i, 1, 1), 1] -
                      dQr[padded_index(sr, i, 1, 1), 2]) for i in 1:32) < 2e-7
    @test maximum(abs(dQ[padded_index(s, i, 1, 1), 2] -
                      dQr[padded_index(sr, i, 1, 1), 1]) for i in 1:32) < 2e-7
    @test maximum(abs(dQ[padded_index(s, i, 1, 1), s.equations.i_energy] -
                      dQr[padded_index(sr, i, 1, 1), sr.equations.i_energy])
                  for i in 1:32) < 2e-7
end

@testset "neutral polynomial RHS agrees on threaded and KA paths" begin
    s, Q, _ = loschmidt_case(16)
    host = zero(Q)
    compute_rhs!(s, Q, host)
    ka = zero(Q)
    CL.FORCE_KA[] = true
    try
        compute_rhs!(s, Q, ka)
    finally
        CL.FORCE_KA[] = false
    end
    @test parent(ka) ≈ parent(host) rtol=2e-14 atol=2e-7
end

end # module
