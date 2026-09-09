#!/bin/tcsh
#SBATCH --job-name=cles_closure128
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --ntasks=224
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=<bank>
##SBATCH --partition=pbatch

# Can Taylor-Green at 128^3 discriminate a subgrid closure at all?
#
# The alpha fit reproduces the reference, but weakening the filter moves the
# sink mostly to resolved molecular dissipation rather than to mu*: across
# alpha 0.45 -> 0.499 the molecular share rises 61.1 -> 75.9 while the Cook
# shear share rises only 2.4 -> 8.4. If mu* can be removed entirely without
# moving the histories, this case is measuring truncation and resolution
# rather than a closure, and C_mu cannot be fitted from it at any alpha.
#
# Three controls at each of two filter strengths:
#   on:1     the full configuration
#   on:1:0   C_mu = 0, so mu* is off and beta*/kappa* remain
#   off:1    every artificial property off
#
# alpha 0.45 is the production filter and 0.499 the near-off one, so the pair
# separates "mu* matters" from "mu* matters once the filter lets go". Leg 1
# ran on:1 at both values already; they are repeated here because the budget
# columns are not comparable across the instrument change below.
#
# The same six points were measured at 64^3 first and the ordering is monotone
# at both alphas: removing mu* improves the peak, the peak time, the KE misfit
# and the dissipation misfit, and removing every artificial property improves
# them further (CALIBRATION.md, "Removing mu* improves the fit at 64 cubed").
# The expectation is that 128^3 reproduces that ordering and strengthens it,
# since the case is closer to resolved. A 128^3 run on which C_mu = 0.002
# scores better would confine the 64^3 result to coarse grids.
#
# 128^3 is run because the alpha fit, the C_mu sweep and the recorded channel
# shares are all 128^3 campaigns, so overturning the "retain 0.002" reading on
# 64^3 evidence alone would not be sound. If queue time is short, the alpha
# 0.45 arm alone is sufficient for the shipped configuration at half the
# cost: drop 0.499 from the alphaf list below.
#
# The budget columns here are not comparable with the recorded 128^3 tables.
# Those carry a residual in the filter column, -dKE/dt - (mol + mu* + beta*);
# this run measures the filter pass and prints pressure work as its own
# channel, with `unattr` for what neither accounts for.
#
# 6 points at ~20-25 min each, so about 2.5 h at 224 ranks.
#
# Submit from a login shell carrying the usual modules (mvapich2 and
# hdf5-parallel/1.14.0); sbatch exports the environment by default.

if ($?CLES == 0) then
    echo "CLES is unset"
    exit 1
endif

cd ~/dev/cles_cpu
if ($status != 0) exit 1

# Precompile serially first, in this exact environment. 224 ranks racing for
# the depot's pidfile is the documented way to hang a job with no output, and
# selecting the system MPI binary switches off src/precompile.jl's workload.
# No HDF5 here: this job writes no snapshots.
julia --project=. -e 'using CompactLES'
if ($status != 0) then
    echo "serial precompile failed"
    exit 1
endif

# configs is the outer axis and alphaf the inner one, so the six blocks print
# in the order on:1 (0.45, 0.499), on:1:0 (0.45, 0.499), off:1 (0.45, 0.499).
# A configuration that loses positivity raises SolverFailure, which is
# reported and the sweep continues; nmax bounds one that grinds instead.
srun -N 2 --ntasks-per-node=112 -n 224 --cpu-bind=threads \
    julia --project=. -t 1 --handle-signals=no $CLES/bench/tgv_energy.jl \
    128 10.0 configs=on:1,on:1:0,off:1 smoother=gaussian \
    alphaf=0.45,0.499 cfl=0.6 filter_cfl=0.0 nmax=25000 \
    sample=200 progress=500
