#!/bin/tcsh
#SBATCH --job-name=cles_n1_recal128
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --ntasks=224
#SBATCH --time=6:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=<bank>
##SBATCH --partition=pbatch

# ROADMAP N1, the 128^3 filter legs re-taken at a production timestep.
#
# Every earlier 128^3 leg ran at nominal cfl 0.6 under the summed acoustic
# rate, which on an isotropic 3-D grid is sqrt(3) too large: that step is
# cfl 0.35 under the corrected Euclidean rate now in max_rate, a fifth of
# the measured acoustic ceiling (~1.7). Under filter_cfl = 0 the filter
# dissipates per application, so those fits carried 1.7x the filtering per
# unit time that a run at the default cfl 0.5 receives. This card repeats the
# alpha sweep and the relaxation leg at cfl 0.5, the solver default, which is
# the reference CFL the filter_cfl recommendation is to be stated at.
# reference/CALIBRATION_APPENDIX.md, "The CFL rate is normalized
# differently", carries the measurement behind the change.
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
mkdir -p $out/recal128
lfs setstripe -c 8 $out/recal128

# Precompile serially first, in this exact environment; 224 ranks racing for
# the depot's pidfile is the documented way to hang a job with no output.
julia --project=. -e 'using CompactLES, HDF5; println(hdf5_parallel())'
if ($status != 0) then
    echo "serial precompile failed"
    exit 1
endif

# --- The alpha sweep at cfl 0.5, ~15 min per point ------------------------
# The same four strengths as leg 1 plus the fitted 0.49, so the new table
# reads against the old one point for point. Steps fall from ~12k to ~8.5k
# per run, and the filter's share of the sink falls with them.
srun -N 2 --ntasks-per-node=112 -n 224 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/tgv_energy.jl \
    128 10.0 configs=on:1 smoother=gaussian alphaf=0.40,0.45,0.49,0.499 \
    cfl=0.5 filter_cfl=0.0 nmax=25000 snapshots=9.0 \
    snapshot_dir=$out/recal128 sample=200 progress=500

# --- The relaxation leg at the new reference ------------------------------
# cfl 0.25 and 0.5 crossed with filter_cfl 0 and 0.5, at alpha 0.45. The
# relaxed pair should agree and the unrelaxed pair should not; the
# cfl 0.5 / filter_cfl 0.5 row reproduces the cfl 0.5 / filter_cfl 0 row
# bit for bit, the instrument's own check. No snapshots, for the reason in
# n1_legs23.sbatch: a landing step pays a full pass unrelaxed and a scaled
# one relaxed, which would bias exactly this comparison.
srun -N 2 --ntasks-per-node=112 -n 224 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/tgv_energy.jl \
    128 10.0 configs=on:1 smoother=gaussian alphaf=0.45 \
    cfl=0.25,0.5 filter_cfl=0.0,0.5 nmax=45000 \
    sample=400 progress=1000
