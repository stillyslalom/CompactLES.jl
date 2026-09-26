# A vortex ring driven into an air/SF6 interface, then a shock fired through it.
#
# The configuration is a vertical shock tube with a vortex-ring injector 15 cm
# above an air/SF6 interface: a 1/2" injector, a 5" square tube represented by
# the axisymmetric tube of the same cross-sectional area, and an M = 1.36 shock.
# The calculation is axisymmetric, in (r, θ, z) with θ collapsed.
#
# A jet pulse through the injector in the top face rolls up into a ring that
# travels down to the interface. As the ring reaches it, the same face fires the
# shock: a Dirichlet face holds the jet state and then the post-shock state
# through a `Ramp`, which spreads the change over the time the shock takes to
# cross three cells, so the shock enters resolved rather than as a jump on the
# boundary plane. Once the shock is in, a scheduled `SwitchableBC` hands the
# face to a characteristic inflow of the post-shock state, which lets the shock
# reflected from the interface leave rather than reflecting it back. The
# transmitted shock reflects from the end wall below and reshocks the layer.
#
# The experiment's jet runs at about 1.5 m/s (Mach 0.004). Its ring then takes
# about 0.2 s to cross the 15 cm, which is some 3e5 acoustic steps at the
# default resolution, so the default jet speed is scaled up to 60 m/s: the ring
# travels at 0.38 of the jet speed, measured at this resolution, and the shock is
# fired to arrive with it. Pass jet_speed=1.5 for the experiment's value;
# t_shock follows the ring's measured speed ratio unless given.
#
# Run: julia --project=. -t 16 examples/vortex_ring_shock.jl
#      mpiexec -n 8 julia --project=. -t 1 examples/vortex_ring_shock.jl nr=224 nz=768

using CompactLES
using CompactLES.Regions
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

const opt = CompactLES.script_args(ARGS,
    (nr=112, nz=384, Mach=1.36, jet_speed=60.0, stroke=3.0, t_shock=0.0,
     t_after=1.6e-3, nmax=10_000_000, frames=80, every=500, output="vortex_ring_shock");
    positional=(:nr, :nz))

const R = 0.127 / sqrt(π)      # tube radius of the 5" square's area, m
const D = 0.0127               # injector diameter
const gap = 0.15               # injector to interface
const z_interface = 0.10       # SF6 below, down to the end wall; air above
const H = z_interface + gap    # the injector sits in the top face
# The ring's core leaves the injector plane 0.4 t_pulse after the pulse starts
# and travels at 0.38 of the peak jet speed: measured for stroke = 3 at the
# default resolution, and dependent on the stroke ratio.
const ring_speed_ratio = 0.38
const ring_delay = 0.4

mpi_main() do
    eos = Nasa9Mixture(["Air", "SF6"])
    air = Prim(Y=(1.0, 0.0), p=101_325.0, T_ion=295.0)
    sf6 = Prim(Y=(0.0, 1.0), p=101_325.0, T_ion=295.0)

    # A sin² velocity pulse whose stroke (the length of the ejected slug) is
    # `stroke` injector diameters; its mean speed is half the peak.
    U = opt.jet_speed
    t_pulse = 2 * opt.stroke * D / U
    pulse(t) = t <= 0 || t >= t_pulse ? 0.0 : sin(π * t / t_pulse)^2
    jet(r, θ, z, t) = Prim(Y=(1.0, 0.0), p=101_325.0, T_ion=295.0,
                           u=(0.0, 0.0, -U * pulse(t) * (1 - tanh_blend(r, D / 2, 5e-4))))

    # The shock travels down (dimension 3, direction -1) into quiescent air and
    # is fired so as to reach the interface when the ring does.
    incident = shock_jump(eos, air, opt.Mach; dim=3, direction=-1)
    arrival = ring_delay * t_pulse + gap / (ring_speed_ratio * U)
    t_shock = opt.t_shock > 0 ? opt.t_shock : arrival - gap / abs(incident.shock_speed)
    W = abs(incident.shock_speed)
    fire = Ramp(eos, jet, incident.post; start=t_shock, duration=Cells(3), speed=W)
    # Ten cells after firing, the ramp is long complete and the face can open.
    t_open = t_shock + 10 * (H / opt.nz) / W
    top = SwitchableBC(DirichletBC(fire), NSCBCInflowBC(incident.post); at=t_open)
    tfinal = t_shock + gap / W + opt.t_after

    problem = Problem(
        name="vortex ring and shock through an air/SF6 interface",
        eos=eos,
        metric=CylindricalMetric(),
        domain=((0.0, R), (0.0, 2π), (0.0, H)),
        bcs=((AxisBC(), SlipWallBC()), PeriodicBC(), (SlipWallBC(), top)),
        ic=Layers(air, Slab(3, hi=z_interface) => sf6; width=Cells(2)),
    )
    numerics = Numerics(n_global=(opt.nr, 1, opt.nz), art=ArtificialProperties(enabled=true),
                        control=StepControl(retries=4))
    solver, Q = setup(problem, numerics)

    if MPI.Comm_rank(solver.comm) == 0
        s = thermodynamic_state(eos, incident.post)
        println("jet pulse $(round(t_pulse * 1e3, digits=2)) ms; shock fired at ",
                "$(round(t_shock * 1e3, digits=2)) ms, ",
                "$(round(abs(incident.shock_speed), digits=1)) m/s, post-shock p = ",
                "$(round(s.p / 1e3, digits=1)) kPa; tfinal $(round(tfinal * 1e3, digits=2)) ms")
    end
    run!(solver, Q; tfinal=tfinal, nmax=opt.nmax,
         callback=(ProgressLog(every=opt.every, tfinal=tfinal),
                   Callback(EveryTime(tfinal / opt.frames),
                            FieldWriter(joinpath(opt.output, "field");
                                        fields=(:rho, :p, :velocity, :X,
                                                :vorticity_magnitude, :schlieren)))))
    MPI.Comm_rank(solver.comm) == 0 &&
        println(solver.t >= tfinal * (1 - 1e-9) ? "reached tfinal" :
                "stopped at t = $(solver.t) after nmax = $(opt.nmax)")
end
