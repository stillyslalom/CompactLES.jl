# Hardware acceptance for R5: all wall normals, both precisions, molecular /
# artificial diffusion and both species channels, against the CPU solver.
#
# julia --project=<env-with-AMDGPU> -t 1 bench/wallflux.jl backend=amdgpu
# julia --project=. -t 1 bench/wallflux.jl backend=cpu
#
# The CPU option checks DeviceBackend construction on KA CPU; it does not
# establish hardware coverage. Manufactured evolution/budgets live in
# test/wall_flux_tests.jl. This script asserts its acceptance criteria.
using CompactLES, MPI, KernelAbstractions
const CL = CompactLES
opt = CL.script_args(ARGS, (backend="cpu",))
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

ka_backend = if opt.backend == "amdgpu"
    @eval using AMDGPU
    AMDGPU.functional() || error("AMDGPU is not functional")
    println("device: ", AMDGPU.device())
    AMDGPU.ROCBackend()
elseif opt.backend == "cuda"
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional")
    println("device: ", CUDA.device())
    CUDA.CUDABackend()
elseif opt.backend == "cpu"
    KernelAbstractions.CPU()
else
    error("backend must be cpu, amdgpu, or cuda")
end

function wall_case(::Type{T}, d, iso, channel, backend) where T
    wall = NoSlipWallBC(Twall=iso ? T(2) : T(NaN))
    eos = IdealMixture((IdealSpecies(T, "a"; R=T(1), gamma=T(1.4)),
                        IdealSpecies(T, "b"; R=T(0.7), gamma=T(1.3))))
    s = Solver(n_global=(12, 12, 12), L_domain=(one(T), one(T), one(T)),
               bcs=ntuple(_ -> (wall, wall), 3), eos=eos, backend=backend,
               transport=Transport{T}(mu0=T(0.01)),
               art=ArtParams{T}(enabled=true, species_flux=channel),
               deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T),
               filter_interval=0)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> begin
        r = (x, y, z)[d]
        y1 = T(0.4) + T(0.05)*cospi(r)
        Prim(rho=one(T)+T(0.02)*r, Y=(y1, one(T)-y1),
             T_ion=T(2)+T(0.1)*r)
    end)
    apply_bcs!(s, Q)
    compute_rhs!(s, Q, zero(Q))
    # Force nonzero artificial heat/species transport independently of the
    # detector, then assemble again with the already computed gradients.
    fill!(s.kappa_art, T(0.007))
    foreach(a -> fill!(a, T(0.011)), s.D_art)
    CL.assemble_fluxes!(s, Q)
    for ax in 1:3, side in 1:2
        correct_flux!(wall, s, Q, ax, side)
    end
    cp, kap, grad = Array(s.cp_mix), Array(s.kappa_art), Array(s.grad_T_ion[d])
    for c in (1, 2, s.equations.i_energy)
        f = Array(s.flux[d, c])
        for side in 1:2, I in CL.wallplane(s.decomp, d, side)
            expected = iso && c == s.equations.i_energy ?
                -(s.transport.mu0*cp[I]/s.transport.Pr+kap[I])*grad[I] : zero(T)
            @assert abs(f[I]-expected) <= 16eps(T)*max(one(T), abs(expected))
            (c <= 2 || !iso) && @assert f[I] == zero(T)
        end
    end
    run!(s, Q; tfinal=T(1e-4), nmax=10)
    @assert s.t == T(1e-4)
    return Array(parent(Q))
end

mpi_main() do
    @assert MPI.Comm_size(MPI.COMM_WORLD) == 1 "run this hardware probe on one rank"
    for T in (Float64, Float32), d in 1:3, iso in (false, true),
        channel in (:fickian, :bulk)
        host = wall_case(T, d, iso, channel, CPUBackend())
        device = wall_case(T, d, iso, channel, DeviceBackend(ka_backend))
        err = maximum(abs.(device .- host))
        @assert err <= 64eps(T)*max(one(T), maximum(abs, host))
        println("$T dim=$d iso=$iso $channel: max state difference $err")
        flush(stdout)
    end
    println("wall flux hardware/backend acceptance complete")
end
