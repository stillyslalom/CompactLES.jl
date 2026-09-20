# Included by mpi_tests.jl after its MPI harness definitions.

function test_composite_budgets()
    section("composite conserved budgets and mixing profiles")
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 1.0, 1.4)])
    vel = (0.4, -0.3, 0.2)
    ic(x, y, z) = Prim(Y=(0.3, 0.7), u=vel, p=2.0,
                       rho=1.0 + 0.2sin(2π * x))
    function make(comm_here, dims_here; kw...)
        common = (; n_global=(192, 1, 1), L_domain=(1.0, 1.0, 1.0),
                  bcs=per3, eos=eos, art=ArtParams(enabled=false),
                  filter_interval=0, comm=comm_here)
        s = dims_here === nothing ? Solver(; common..., kw...) :
                                    Solver(; common..., dims=dims_here, kw...)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        return s, Q
    end

    ref, Qref = make(MPI.COMM_SELF, (1, 1, 1))
    bref = CL._conserved_budget(ref, Qref)
    s, states = make(comm, nothing; patch_grid=(2, 1, 1))
    b = CL._conserved_budget(s, states)
    packed(x) = [x.species_masses; x.total_mass; collect(x.momentum); x.total_energy]
    check("same-level conserved vector vs COMM_SELF",
          maximum(abs.(packed(b) .- packed(bref))), 2e-13)
    check("same-level mix width vs COMM_SELF",
          abs(mix_width(s, states) - mix_width(ref, Qref)), 2e-13)
    check("same-level molecular mixing vs COMM_SELF",
          abs(molecular_mixing(s, states) - molecular_mixing(ref, Qref)), 2e-13)
    check("same-level root coordinates vs COMM_SELF",
          maximum(abs.(profile_coordinate(s, 1) .- profile_coordinate(ref, 1))), 2e-15)
    check("same-level root spacing vs COMM_SELF",
          maximum(abs.(profile_spacing(s, 1) .- profile_spacing(ref, 1))), 2e-15)

    refinement = [BlockRegion((36, 0, 0), (32, 1, 1)),
                  BlockRegion((120, 0, 0), (8, 1, 1))]
    deepref, Qdeepref = make(MPI.COMM_SELF, (1, 1, 1);
                              refine=refinement, subcycle=true)
    deep, Qdeep = make(comm, splitdims(1); refine=refinement, subcycle=true)
    bdeep = CL._conserved_budget(deep, Qdeep)
    bdeepref = CL._conserved_budget(deepref, Qdeepref)
    check("three-level conserved vector vs COMM_SELF",
          maximum(abs.(packed(bdeep) .- packed(bdeepref))), 2e-12)
    held = npatches(deep)
    held_min = MPI.Allreduce(held, min, comm)
    held_max = MPI.Allreduce(held, max, comm)
    check("root-only ranks enter refined collective",
          np > 2 && held_min < held_max ? 0.0 : np <= 2 ? 0.0 : 1.0, 0.5)
end
