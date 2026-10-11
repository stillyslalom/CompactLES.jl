#!/bin/tcsh
#SBATCH --job-name=cles_rmicost_lattice
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=<bank>
##SBATCH --partition=pbatch

# The lattice cost model across nodes on rzhound: bench/rmicost.jl 3-D amr2
# at nres = 192 (582 x 96 x 96 root nodes, 12k to 48k per rank over one to
# four nodes), three legs per allocation, each twice and alternated:
#
#   configured  every level on the tile = 12 lattice (the subdivision off)
#   flat        the model's lattice, every rank on one node for the
#               placement and the message prices (nodesize past the count)
#   nodes       the model's lattice over the nodes MPI reports
#
# Submit once per node count, the task count following the allocation:
#   sbatch --nodes=1 bench/slurm/rmicost_lattice.sbatch
#   sbatch --nodes=2 bench/slurm/rmicost_lattice.sbatch
#   sbatch --nodes=4 bench/slurm/rmicost_lattice.sbatch
#
# Compare the steady wall per microsecond and the median step between legs
# of one allocation; across allocations the nodes differ. The `nodes` leg's
# report gives the fitted m and m_x (the per-message seconds within a node
# and out of it) and the node count; trace=true prints every lattice choice
# with the predicted seconds of each lattice. nmax = 300 root steps after
# the 10 of the warm-up, about 3 to 6 minutes a leg; the report projects the
# full window. A run that loses positivity raises at the step floors rather
# than grinding.

if ($?CLES == 0) then
    echo "CLES is unset"
    exit 1
endif

cd $CLES
if ($status != 0) exit 1

setenv OPENBLAS_NUM_THREADS 1
set n = `expr $SLURM_NNODES \* 112`

# Precompile serially first, in this exact environment: ranks racing for the
# depot's pidfile hang the job with no output.
julia --project=. -e 'using CompactLES'
if ($status != 0) then
    echo "serial precompile failed"
    exit 1
endif

# The allocation as granted: masks, node distribution, MPI library.
srun -N $SLURM_NNODES --ntasks-per-node=112 -n $n --cpu-bind=threads \
    julia --project=. -t 1 probes/clusterprobe.jl 128

foreach rep (1 2)
    foreach leg (configured flat nodes)
        set extra = ""
        if ($leg == configured) set extra = "lattice=configured"
        if ($leg == flat) set extra = "nodesize=1000000 trace=true"
        if ($leg == nodes) set extra = "trace=true"
        echo "=== leg $leg rep $rep nodes $SLURM_NNODES"
        srun -N $SLURM_NNODES --ntasks-per-node=112 -n $n --cpu-bind=threads \
            julia --project=. -t 1 --handle-signals=no bench/rmicost.jl \
            config=amr2 dim=3 nres=192 nmax=300 progress=100 $extra
    end
end
