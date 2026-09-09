#!/bin/tcsh
#SBATCH --job-name=cles_n1_recal256
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=112
#SBATCH --ntasks=896
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=<bank>
##SBATCH --partition=pbatch

# ROADMAP N1, the 256^3 transfer check re-taken at a production timestep.
# See n1_recal128.sbatch for why: the earlier 256^3 rows ran at what is
# cfl 0.35 under the corrected acoustic rate. Three points at cfl 0.5, about
# 1.2 h each at 896 ranks (steps fall from ~24k to ~17k), one 1.21 GB
# snapshot each for bench/tgv_spectrum.jl.

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
mkdir -p $out/recal256
lfs setstripe -c 16 $out/recal256

julia --project=. -e 'using CompactLES, HDF5; println(hdf5_parallel())'
if ($status != 0) then
    echo "serial precompile failed"
    exit 1
endif

srun -N 8 --ntasks-per-node=112 -n 896 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/tgv_energy.jl \
    256 10.0 configs=on:1 smoother=gaussian alphaf=0.45,0.49,0.499 \
    cfl=0.5 filter_cfl=0.0 nmax=60000 snapshots=9.0 \
    snapshot_dir=$out/recal256 sample=400 progress=1000
