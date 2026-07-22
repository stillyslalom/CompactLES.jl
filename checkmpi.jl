using MPI
MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
println("np=", np)
try
    println("Dims_create(np,3)=", MPI.Dims_create(np, 3))
catch e
    println("3D failed: ", e)
end
