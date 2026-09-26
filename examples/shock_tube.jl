# Multimode Richtmyer–Meshkov instability with reshock in a shock tube.
#
# A Mach 1.5 shock in air, generated inside the domain from the Rankine–Hugoniot
# relations rather than from a driver section, strikes an air/SF6 interface
# carrying a band of modes with seeded random phases. The transmitted shock
# reflects from the end wall and reshocks the mixing layer. Inflow at the left
# end carries the post-shock state. Before running, the script prints the wave
# pattern of the shock's impact on the interface from the exact Riemann
# solution, which the simulated transmitted shock speed can be checked against.
#
# Run: julia --project=. -t 8 examples/shock_tube.jl nx=320 ny=64 tfinal=1e-3
#      mpiexec -n 4 julia --project=. -t 1 examples/shock_tube.jl nx=1280 ny=256
#      mpiexec -n 64 julia --project=. -t 1 examples/shock_tube.jl nx=640 ny=128 nz=128
# nz > 1 makes the perturbation a surface over (y, z). Add amr=true to follow
# the shocks and the mixing layer with refined tiles instead of a uniform grid.
# Output goes to `output` (default shock_tube/): VTK frames with a .pvd
# collection, starting from the initial state, checkpoints, and a mix-width
# time series.

using CompactLES
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

const opt = CompactLES.script_args(ARGS,
    (nx=640, ny=128, nz=1, Mach=1.5, tfinal=3e-3, nmax=1_000_000, frames=60,
     every=100, seed=1, amr=false, output="shock_tube");
    positional=(:nx, :ny))

const Lx = 0.5             # tube length, m
const Ly = 0.1             # tube width (and depth when nz > 1)
const x_shock = 0.20       # initial shock position
const x_interface = 0.25   # mean interface position
const modes = 4:12         # perturbation band, in modes per tube width
const rms = 5e-4           # root-mean-square interface displacement

mpi_main() do
    eos = Nasa9Mixture(["Air", "SF6"])
    air = Prim(Y=mass_fractions(eos, "Air" => 1.0; basis=:mole),
               p=101_325.0, T_ion=295.0)
    sf6 = Prim(Y=mass_fractions(eos, "SF6" => 1.0; basis=:mole),
               p=101_325.0, T_ion=295.0)
    incident = shock_jump(eos, air, opt.Mach)

    # Three regions: shocked air behind the shock, quiescent air, and SF6 beyond
    # the perturbed interface, whose phases are drawn from `seed`, so every
    # rank builds the same one. The error-function profile of the transitions
    # is the diffusion profile of the benchmark specifications; its scale is
    # three cells.
    eta = Multimode(lengths=opt.nz > 1 ? (Ly, Ly) : (Ly,), modes=modes, rms=rms,
                    seed=opt.seed, mean=x_interface)
    initial = Layers(air,
                     Slab(1, hi=x_shock) => incident.post,
                     Slab(1, lo=eta) => sf6;
                     profile=:erf)

    problem = Problem(
        name="air/SF6 Richtmyer–Meshkov with reshock",
        eos=eos,
        domain=((0.0, Lx), (0.0, Ly), (0.0, opt.nz > 1 ? Ly : 1.0)),
        bcs=((NSCBCInflowBC(incident.post), SlipWallBC()), PeriodicBC(), PeriodicBC()),
        ic=initial,
    )
    # The sensor follows the shocks and the composition gradient the mixing
    # layer. They separate after the impact, so tiles cover them rather than
    # one box spanning both; the regrid interval takes its default.
    amr = opt.amr ? AMR(initial=:sensor, subcycle=true, tag_gradient_threshold=0.02,
                        tile=16) : nothing
    numerics = Numerics(n_global=(opt.nx, opt.ny, opt.nz), art=ArtParams(enabled=true),
                        control=StepControl(retries=4), amr=amr)
    solver, Q = setup(problem, numerics)

    if MPI.Comm_rank(solver.comm) == 0
        s2 = thermodynamic_state(eos, incident.post)
        impact = riemann_interface(eos, incident.post, sf6)
        println("incident shock: $(round(incident.shock_speed, digits=1)) m/s, ",
                "post-shock p = $(round(s2.p / 1e3, digits=1)) kPa, ",
                "T = $(round(s2.T_ion, digits=1)) K, ",
                "u = $(round(incident.velocity, digits=1)) m/s")
        println("at impact: transmitted shock $(round(impact.right_speed, digits=1)) m/s, ",
                "reflected $(impact.left_wave), interface velocity ",
                "$(round(impact.u_star, digits=1)) m/s")
    end

    # Each callback runs on every rank; only the output is restricted to rank 0.
    series = NTuple{3,Float64}[]
    record_mixing(s, Q) =
        (push!(series, (s.t, mix_width(s, Q), molecular_mixing(s, Q))); false)
    checkpoint(s, Q) =
        (save_checkpoint(s, Q, joinpath(opt.output, "checkpoint")); false)
    fields = (:rho, :velocity, :p, :T_ion, :X, :schlieren)
    callbacks = (
        ProgressLog(every=opt.every, tfinal=opt.tfinal, label="mix width [m]",
                    quantity=(s, Q) -> mix_width(s, Q)),
        Callback(EveryTime(opt.tfinal / opt.frames),
                 FieldWriter(joinpath(opt.output, "field"); fields=fields)),
        Callback(EveryTime(opt.tfinal / 200), record_mixing),
        Callback(EveryTime(opt.tfinal / 4), checkpoint),
    )
    run!(solver, Q; tfinal=opt.tfinal, nmax=opt.nmax, callback=callbacks)

    if MPI.Comm_rank(solver.comm) == 0
        open(joinpath(opt.output, "mixing.csv"), "w") do io
            println(io, "t,mix_width,molecular_mixing")
            foreach(r -> println(io, join(r, ",")), series)
        end
        println(solver.t >= opt.tfinal * (1 - 1e-9) ?
                "reached tfinal = $(opt.tfinal)" :
                "stopped at t = $(solver.t) after nmax = $(opt.nmax)")
    end
end
