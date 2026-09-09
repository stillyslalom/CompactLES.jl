#!/bin/tcsh
#SBATCH --job-name=cles_n1_legs23
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --ntasks=224
#SBATCH --time=6:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=<bank>
##SBATCH --partition=pbatch

# ROADMAP N1, the cadence and relaxation legs at 128^3 on rzhound.
# Same allocation shape and launch flags as leg 1, which fitted
# alpha to 0.485-0.49 at cfl 0.6 and filter_interval 1.
#
# Submit from a login shell carrying the usual modules (mvapich2 and
# hdf5-parallel/1.14.0); sbatch exports the environment by default.

if ($?CLES == 0) then
    echo "CLES is unset"
    exit 1
endif
if ($?LFILES == 0) then
    echo "LFILES is unset"
    exit 1
endif

cd ~/dev/cles_cpu
if ($status != 0) exit 1

set out = "$LFILES/CompactLES"
mkdir -p $out/leg2 $out/leg3
lfs setstripe -c 8 $out/leg2

# Precompile serially first, in this exact environment. 224 ranks racing for
# the depot's pidfile is the documented way to hang a job with no output, and
# selecting the system MPI binary switches off src/precompile.jl's workload.
julia --project=. -e 'using CompactLES, HDF5; println(hdf5_parallel())'
if ($status != 0) then
    echo "serial precompile failed"
    exit 1
endif

# --- The missing point inside the fitted band, ~20 min --------------------
# The battery results do not separate alpha = 0.486 from 0.49, and the peak
# crossing places the total-dissipation match at the former, so the choice
# inside the band rests on the history misfit, not yet evaluated there. One
# point, at the settings of leg 1, so its misfits compare directly with them.
srun -N 2 --ntasks-per-node=112 -n 224 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/tgv_energy.jl \
    128 10.0 configs=on:1 smoother=gaussian alphaf=0.486 \
    cfl=0.6 filter_cfl=0.0 nmax=25000 snapshots=9.0 \
    snapshot_dir=$out/leg1 sample=200 progress=500

# --- Leg 2: cadence crossed with alpha ------------------------------------
# 6 points, ~20-25 min each. Crossed rather than pinned at one alpha because
# N1 covers the two jointly, and the shock-battery results may yet return
# alpha to 0.45. filter_interval = 4 diverged at 32^3; a SolverFailure is reported
# and the sweep continues, and nmax bounds a run that grinds instead.
srun -N 2 --ntasks-per-node=112 -n 224 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/tgv_energy.jl \
    128 10.0 configs=on:1,on:2,on:4 smoother=gaussian alphaf=0.45,0.49 \
    cfl=0.6 filter_cfl=0.0 nmax=25000 snapshots=9.0 \
    snapshot_dir=$out/leg2 sample=200 progress=500

# --- Leg 3: relaxation ----------------------------------------------------
# 4 points: ~20 min at cfl 0.6, ~40 at 0.3. The unrelaxed pair should differ
# across the CFL and the relaxed pair should not, which is filter_weight's
# invariance claim at the resolution the default will be set from. Run at
# alpha = 0.45, where the filter's share of the sink is larger and the CFL
# sensitivity is easier to resolve; the claim itself is alpha-independent.
#
# No snapshots here. Under filter_cfl = 0 a step shortened to land on one
# still pays a full filter pass and under a positive filter_cfl it does not,
# so requesting them would bias exactly the comparison this leg makes.
srun -N 2 --ntasks-per-node=112 -n 224 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/tgv_energy.jl \
    128 10.0 configs=on:1 smoother=gaussian alphaf=0.45 \
    cfl=0.3,0.6 filter_cfl=0.0,0.6 nmax=45000 \
    sample=400 progress=1000
