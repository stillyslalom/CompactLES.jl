# Wall time of the shared-file HDF5 writes under each MPI transfer mode of the
# block datasets, `:collective` (the default) and `:independent`, on a parallel
# libhdf5. Three writes are timed: a checkpoint (`save_checkpoint_hdf5`, the
# Float64 state), a field frame (`save_hdf5` with density and velocity, Float32)
# and a slice across the first dimension, on which every rank but one issues an
# empty-selection write. Each is the maximum over ranks of the wall time of the
# collective call, after one untimed write per mode that carries the compilation.
#
# The timed quantity is the whole write, file creation and close included, so it
# reports what a run pays per dump and not the transfer alone. On a serialized
# libhdf5 (`hdf5_parallel() == false`) no transfer property applies and the two
# modes measure the same relay.
#
# Usage: positional grid edge and repeat count, then `key=value` options:
#
#   MPIEXEC=$(julia --project=. -e 'using MPI; MPI.mpiexec(c -> print(c))')
#   "$MPIEXEC" -n 4 julia --project=. -t 1 bench/hdf5xfer.jl 64 5
#   "$MPIEXEC" -n 8 julia --project=. -t 1 bench/hdf5xfer.jl 128 3 modes=collective
#
# Run from an environment carrying HDF5 beside the package, and precompile it
# serially before launching `mpiexec`.
#
# Options:
#   N        cubic global grid edge (default 64): a 5 N^3 Float64 checkpoint and
#            a 4 N^3 Float32 frame.
#   reps     timed writes per mode and kind (default 5); the median and the
#            minimum are reported.
#   modes    comma-separated list of independent, collective (default both).
#   dir      output directory (default a temporary one, removed at the end).
#            Point it at the filesystem the run would write to.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using HDF5
using Printf
using Statistics: median

const DEFAULTS = (N = 64, reps = 5, modes = "independent,collective", dir = "")
const Ext = Base.get_extension(CompactLES, :CompactLESHDF5Ext)

function build(N, comm)
    per = (PeriodicBC(), PeriodicBC())
    s = Solver(bcs=(per, per, per), n_global=(N, N, N), L_domain=(1.0, 1.0, 1.0),
               art=ArtificialProperties(enabled=false), comm=comm)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(sin(2π * x), cos(2π * y), 0.0),
                                        p=1.0, rho=1 + 0.1z))
    return s, Q
end

# Seconds of the collective write on its slowest rank.
function timed(f, comm)
    MPI.Barrier(comm)
    t0 = MPI.Wtime()
    f()
    return MPI.Allreduce(MPI.Wtime() - t0, MPI.MAX, comm)
end

function main(opt)
    comm = MPI.COMM_WORLD
    rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    root = rank == 0
    say(fmt, args...) = root && print(Printf.format(Printf.Format(fmt), args...))
    modes = Symbol.(strip.(split(opt.modes, ',')))
    dir = isempty(opt.dir) ? (root ? mktempdir() : "") : opt.dir
    dir = MPI.bcast(dir, comm; root=0)
    s, Q = build(opt.N, comm)
    gx = opt.N ÷ 2 + 1
    kinds = (("checkpoint", () -> save_checkpoint_hdf5(s, Q, joinpath(dir, "ckpt"))),
             ("frame", () -> save_hdf5(s, Q, joinpath(dir, "frame");
                                       fields=(:rho, :velocity))),
             ("slice", () -> save_hdf5(s, Q, joinpath(dir, "slice");
                                       fields=(:rho,), slice=(1, gx))))
    say("=== shared-file HDF5 write cost: %d^3, %d rank(s), process grid %s\n",
        opt.N, nranks, string(s.decomp.dims))
    say("    libhdf5 %s, parallel %s, %s\n", string(HDF5.API.h5_get_libversion()),
        string(CompactLES.hdf5_parallel()),
        first(split(MPI.Get_library_version(), '\n')))
    say("    checkpoint %.1f MiB, frame %.1f MiB, %d timed write(s) per cell, in %s\n",
        5 * opt.N^3 * 8 / 2^20, 4 * opt.N^3 * 4 / 2^20, opt.reps, dir)
    root && println()
    root && println("  mode          kind         median s     min s")
    for mode in modes
        Ext.BLOCK_TRANSFER[] = mode
        for (kind, f) in kinds
            timed(f, comm)                        # compile and prime
            ts = [timed(f, comm) for _ in 1:opt.reps]
            say("  %-13s %-11s %9.4f %9.4f\n", mode, kind, median(ts), minimum(ts))
            say("row,%s,%s,%d,%d,%.6e,%.6e\n", mode, kind, nranks, opt.N,
                median(ts), minimum(ts))
        end
    end
    MPI.Barrier(comm)
    isempty(opt.dir) && root && rm(dir; recursive=true)
    return nothing
end

const _opt = CompactLES.script_args(ARGS, DEFAULTS; positional = (:N, :reps))
mpi_main(() -> main(_opt))
