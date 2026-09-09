#!/bin/tcsh
#SBATCH --job-name=cles_n1_256
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=112
#SBATCH --ntasks=896
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=<bank>
##SBATCH --partition=pbatch

# ROADMAP N1, the 256^3 transfer check. k_max*eta reaches 1.5 here against
# 0.75 at 128^3, so the dissipation range is resolved and the question is
# whether the alpha minimum stays where 128^3 put it or moves, which decides
# whether the fitted value is a constant or a per-resolution tuning.
#
# Work scales as N^4: one configuration is ~16x a 128^3 one, about 1.4 h at
# 896 ranks against 20 min at 224. Three points, plus a 1.21 GB snapshot each.

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

set out = "$LFILES/CompactLES/confirm256"
mkdir -p $out
lfs setstripe -c 16 $out

julia --project=. -e 'using CompactLES, HDF5; println(hdf5_parallel())'
if ($status != 0) then
    echo "serial precompile failed"
    exit 1
endif

# The node count and rank count are both new, so record what was allocated
# before spending eight hours on it. Pass the target grid: the probe sets the
# decomposition from it.
srun -N 8 --ntasks-per-node=112 -n 896 --cpu-bind=threads \
    julia --project=. -t 1 $CLES/probes/clusterprobe.jl 256

srun -N 8 --ntasks-per-node=112 -n 896 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/tgv_energy.jl \
    256 10.0 configs=on:1 smoother=gaussian alphaf=0.45,0.49,0.499 \
    cfl=0.6 filter_cfl=0.0 nmax=60000 snapshots=9.0 \
    snapshot_dir=$out sample=800 progress=2000
