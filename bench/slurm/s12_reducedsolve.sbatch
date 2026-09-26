#!/bin/tcsh
#SBATCH --job-name=cles_s12_reducedsolve
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --ntasks=224
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=<bank>
##SBATCH --partition=pbatch

# The stage-by-stage cost of the distributed line solve on rzhound
# (bench/reducedsolve.jl, ROADMAP S12 stage 1): local sweep, Allgather and
# replicated reduced solve, per direction, for C6 (tridiagonal) and C10
# (pentadiagonal), at the ranks per direction where the operation count puts
# the crossover (P ~ 5-7 at N = 256, 8-10 at N = 1024).
#
# Legs 1-3 are uniform process grids, the shape the S12 cost model assumes,
# at 8, 64 and 216 ranks spread evenly over both nodes. Leg 4 is the 224-rank
# production shape (8,7,4) of the node-scaling runs. Leg 5 puts 56 ranks on
# one direction, far past the count crossover, so the dense solve is measured
# where it dominates. Each leg is a few minutes; most of it is setup.
#
# Submit from a login shell carrying the usual modules (mvapich2);
# sbatch exports the environment by default.

if ($?CLES == 0) then
    echo "CLES is unset"
    exit 1
endif

cd ~/dev/cles_cpu
if ($status != 0) exit 1

# The reduced solve is the only BLAS call; CompactLES sets one BLAS thread
# itself, and this makes the setting explicit in the job's environment.
setenv OPENBLAS_NUM_THREADS 1

# Precompile serially first, in this exact environment. 224 ranks racing for
# the depot's pidfile is the documented way to hang a job with no output, and
# selecting the system MPI binary switches off src/precompile.jl's workload.
julia --project=. -e 'using MPI, CompactLES; println(MPI.MPI_LIBRARY)'
if ($status != 0) then
    echo "serial precompile failed"
    exit 1
endif

# The CPU masks, the node distribution and the decomposition the timings are
# read against.
srun -N 2 --ntasks-per-node=112 -n 224 --cpu-bind=threads \
    julia --project=. -t 1 $CLES/probes/clusterprobe.jl 256

# --- Legs 1-3: uniform grids, P = 2, 4, 6 per direction -------------------
srun -N 2 --ntasks-per-node=4 -n 8 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/reducedsolve.jl \
    256 dims=2,2,2
srun -N 2 --ntasks-per-node=32 -n 64 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/reducedsolve.jl \
    256,512 dims=4,4,4
srun -N 2 --ntasks-per-node=108 -n 216 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/reducedsolve.jl \
    256,512 dims=6,6,6

# --- Leg 4: the production shape of the node-scaling runs -----------------
srun -N 2 --ntasks-per-node=112 -n 224 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/reducedsolve.jl \
    256,512 dims=8,7,4

# --- Leg 5: 56 ranks along x, where the dense solve dominates -------------
# 512 / 56 = 9 points per block, the C8 filter's minimum.
srun -N 2 --ntasks-per-node=112 -n 224 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/reducedsolve.jl \
    512 dims=56,4,1
