# Diagnostic: what does MPI.Dims_create actually accept here?
#
# MPI.jl's second argument is the dims ARRAY (a zero entry means "choose this
# one for me"), NOT the number of dimensions. Passing the count works by
# accident for some values and throws MPIError(779) "cannot partition nodes as
# requested" for others — which is easy to misread as an MS-MPI limitation.
# The zeros form is the correct call and works at every rank count.
#
#   mpiexec -n 4 julia --project=. checkmpi.jl

using MPI
MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
MPI.Comm_rank(MPI.COMM_WORLD) == 0 || (MPI.Finalize(); exit(0))
println("np=", np)
for nact in 1:3
    try
        println("  WRONG  Dims_create(np, $nact)           = ",
                MPI.Dims_create(np, nact))
    catch e
        println("  WRONG  Dims_create(np, $nact)           threw ", typeof(e))
    end
    println("  RIGHT  Dims_create(np, zeros(Cint, $nact)) = ",
            MPI.Dims_create(np, zeros(Cint, nact)))
end
MPI.Finalize()
