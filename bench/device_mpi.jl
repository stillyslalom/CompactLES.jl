# Distributed device runs (reference/AMR_GPU.md, communication). Every rank
# holds a block of the one device-resident patch — on the workstation all
# ranks share the RX 6800 XT — and the halo, fold-pair, and reduced-solve
# traffic crosses ranks through backend staging: broadcast pack into a
# contiguous device buffer, one contiguous device<->host copy per message,
# MPI over the host buffers. The runs are compared against the identically
# decomposed CPU solver, and the staged transfer volume and copy time are
# read from the tracking counters, halo and reduced-interface separately.
#
# Run from an environment carrying CompactLES AND the device package:
#
#   mpiexec -n 2 julia --project=<env-with-AMDGPU> bench/device_mpi.jl backend=amdgpu
#   (repeat at -n 4 and -n 8; grids here satisfy 9 points/rank up to 8 ranks)

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES

const opt = script_args(ARGS, (backend = "amdgpu", steps = 10))

const ka_backend = if opt.backend == "amdgpu"
    @eval using AMDGPU
    @eval AMDGPU.functional() || error("AMDGPU is not functional here")
    @eval ROCBackend()
elseif opt.backend == "cuda"
    @eval using CUDA
    @eval CUDA.functional() || error("CUDA is not functional here")
    @eval CUDABackend()
else
    error("backend must be amdgpu or cuda, got $(opt.backend)")
end

const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

function compare_run(build, label; nmax)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    s1, Q1 = build(CPUBackend())
    run!(s1, Q1; tfinal=1e6, nmax=nmax)
    CL.DEVICE_TRANSFER_BYTES[] = 0
    CL.DEVICE_TRANSFER_TIME[] = 0.0
    CL.REDUCED_TRANSFER_BYTES[] = 0
    CL.REDUCED_TRANSFER_TIME[] = 0.0
    CL.TRACK_DEVICE_TRANSFERS[] = true
    s2, Q2 = build(DeviceBackend(ka_backend))
    try
        run!(s2, Q2; tfinal=1e6, nmax=nmax)
    finally
        CL.TRACK_DEVICE_TRANSFERS[] = false
    end
    dmax = MPI.Allreduce(maximum(abs.(Array(parent(Q2)) .- parent(Q1))),
                         max, comm)
    halo_b = MPI.Allreduce(CL.DEVICE_TRANSFER_BYTES[], +, comm)
    halo_t = MPI.Allreduce(CL.DEVICE_TRANSFER_TIME[], max, comm)
    red_b = MPI.Allreduce(CL.REDUCED_TRANSFER_BYTES[], +, comm)
    red_t = MPI.Allreduce(CL.REDUCED_TRANSFER_TIME[], max, comm)
    wall_c = MPI.Allreduce(s1.wall_total, max, comm)
    wall_d = MPI.Allreduce(s2.wall_total, max, comm)
    if rank == 0
        @printf("%-30s steps %3d/%3d  max|dev-cpu| = %g%s\n", label,
                s1.step, s2.step, dmax, dmax == 0 ? "  (bitwise)" : "")
        @printf("  wall/step: cpu %.3f s  device %.3f s\n",
                wall_c / s1.step, wall_d / s2.step)
        @printf("  staged halo/pair: %.1f MiB total, %.3f s slowest rank ",
                halo_b / 2.0^20, halo_t)
        @printf("(%.1f%% of device wall)\n", 100halo_t / wall_d)
        @printf("  reduced interface: %.1f MiB total, %.3f s slowest rank ",
                red_b / 2.0^20, red_t)
        @printf("(%.1f%% of device wall)\n", 100red_t / wall_d)
    end
    return nothing
end

function main()
    np = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    rank == 0 && println("=== distributed device runs, np = $np, ",
                         "backend $(opt.backend)")
    dims1 = (np, 1, 1)
    dims2 = (1, np, 1)

    compare_run("multispecies tube, split x"; nmax=opt.steps) do backend
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        s = Solver(n_global=(72, 16, 12), L_domain=(1.0, 0.6, 0.3), eos=eos,
                   bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                   dims=dims1, backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.05 * cos(2π * z / 0.3)))
        end)
        return s, Q
    end

    compare_run("resolved-θ axis fold, split θ"; nmax=opt.steps) do backend
        s = Solver(n_global=(20, 72, 12), L_domain=(1.0, 2π, 0.5),
                   bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                   metric=CylindricalMetric(), dims=dims2, backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, z) ->
            Prim(u=(0.05r * cos(θ), 0.2r, 0.05 * sin(2π * z / 0.5)),
                 p=1.0 + 0.02r^2, rho=1.0))
        return s, Q
    end
    return nothing
end

mpi_main(main)
