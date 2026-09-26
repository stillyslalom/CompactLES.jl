# Shock–bubble interaction after Haas & Sturtevant (1987).
#
# A Mach 1.22 shock in air strikes a cylinder of helium contaminated with air.
# The flow is symmetric about the tube centerline, so only the upper half is
# computed, with a symmetry plane at y = 0. The shock starts inside the domain,
# its post-shock state from the Rankine–Hugoniot relations, and enters through a
# characteristic inflow; a characteristic outflow closes the downstream end.
#
# Run: julia --project=. -t 8 examples/shock_bubble.jl nx=384 ny=96 tfinal=2e-4
#      mpiexec -n 4 julia --project=. -t 1 examples/shock_bubble.jl nx=1152 ny=288

using CompactLES
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

const opt = CompactLES.script_args(ARGS,
    (nx=768, ny=192, Mach=1.22, tfinal=6e-4, nmax=1_000_000, frames=60, every=100,
     output="shock_bubble");
    positional=(:nx, :ny))

const radius = 0.025              # bubble radius, m
const center = (0.075, 0.0, 0.0)  # on the symmetry plane
const x_shock = 0.04
const domain = ((0.0, 0.3), (0.0, 0.075), (0.0, 1.0))

mpi_main() do
    eos = Nasa9Mixture(["Air", "He"])
    p0, T0 = 101_325.0, 295.0
    air = Prim(Y=mass_fractions(eos, "Air" => 1.0; basis=:mass), p=p0, T_ion=T0)
    # Haas & Sturtevant's helium carried about 28% air by mass.
    bubble = Prim(Y=mass_fractions(eos, "He" => 0.72, "Air" => 0.28; basis=:mass),
                  p=p0, T_ion=T0)
    incident = shock_jump(eos, air, opt.Mach)

    problem = Problem(
        name="Haas–Sturtevant helium bubble",
        eos=eos,
        domain=domain,
        bcs=((NSCBCInflowBC(incident.post), NSCBCOutflowBC(pinf=p0)),
             (SymmetryPlaneBC(), SlipWallBC()),
             PeriodicBC()),
        # The bubble and the air share p0 and T0, and the transition keeps them
        # through the interface, so the initial state carries no acoustic
        # disturbance besides the shock.
        ic=Layers(air,
                  Slab(1, hi=x_shock) => incident.post,
                  Sphere(center, radius) => bubble),
    )
    numerics = Numerics(n_global=(opt.nx, opt.ny, 1), art=ArtParams(enabled=true),
                        control=StepControl(retries=4))
    solver, Q = setup(problem, numerics)

    # Helium mass, which leaves only through the outflow once the bubble reaches
    # it, and the peak helium mole fraction, which molecular and artificial
    # diffusion erode.
    history = NTuple{3,Float64}[]
    record(s, Q) = begin
        rho = field_array(s, Q, :rho)
        helium = volume_integral(s, rho .* field_array(s, Q, :Y; species=2))
        peak = MPI.Allreduce(maximum(field_array(s, Q, :X; species=2)), max, s.comm)
        push!(history, (s.t, helium, peak))
        false
    end
    run!(solver, Q; tfinal=opt.tfinal, nmax=opt.nmax,
         callback=(ProgressLog(every=opt.every, tfinal=opt.tfinal),
                   Callback(EveryTime(opt.tfinal / opt.frames),
                            FieldWriter(joinpath(opt.output, "field");
                                        fields=(:rho, :p, :velocity, :X,
                                                :vorticity_magnitude, :schlieren))),
                   Callback(EveryTime(opt.tfinal / 100), record)))

    if MPI.Comm_rank(solver.comm) == 0
        open(joinpath(opt.output, "helium.csv"), "w") do io
            println(io, "t,helium_mass,peak_helium_mole_fraction")
            foreach(r -> println(io, join(r, ",")), history)
        end
        t0, m0, _ = first(history)
        t1, m1, x1 = last(history)
        println("helium mass change $(round(100 * (m1 / m0 - 1), sigdigits=3))% ",
                "by t = $t1; peak helium mole fraction $(round(x1, digits=3))")
    end
end
