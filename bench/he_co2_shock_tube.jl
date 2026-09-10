# He-driven shock tube with a perturbed He/CO2 contact on a calorically perfect
# mixture: the deck behind a like-for-like comparison with Pyranda, which
# carries no NASA-9 thermodynamics. examples/shock_tube.jl is the same case on
# `Nasa9Mixture`; everything else (geometry, states, perturbation, blends,
# boundaries, artificial properties, filter) is kept identical here so that a
# difference between the two codes is not a difference between the two decks.
#
#   julia --project=. -t 16 bench/he_co2_shock_tube.jl                # 768 48 2.5e-3
#   julia --project=. -t 16 bench/he_co2_shock_tube.jl 192 12 3e-4    # smoke run
#   mpiexec -n 4 julia --project=. -t 1 bench/he_co2_shock_tube.jl
#
# Positional: nx ny tfinal. Keys: cfl; alphaf (compact-filter alpha); deriv
# (c6 or c10); filter (gv, the Gaitonde–Visbal filter at alphaf, or pyranda,
# Pyranda's c8ff8 through `pyranda_filter`); detector (delta4 or d8);
# mu_sensor (strain or velocity); beta_sensor (strain, gated_strain,
# dilatation or ungated_dilatation); reduction (sum or max); the five
# constants C_mu C_beta C_kappa C_D C_Y; every (diagnostic cadence in steps);
# snapshots (comma-separated instants in ms at which profiles are written);
# prefix (output file stem). The Pyranda-matched configuration is deriv=c10
# filter=pyranda detector=d8 mu_sensor=velocity beta_sensor=ungated_dilatation
# reduction=max, with the Pyranda constants mapped through the detector
# normalization (C:/Users/Alex/Dev/pyranda/cases/README.md).
#
# Output, all ASCII with a header line, rank 0 only:
#   <prefix>_history.dat     step, t, rho min/max, |u| max, min/max Y_He, Y_CO2
#   <prefix>_t<ms>ms.dat     at each snapshot: x, the y-averaged rho, p, u and
#                            Y_CO2, then the same four sampled on the y = 0 line
#                            (where the contact bulges toward +x) and on the
#                            y = Lyz/2 line (where it bulges toward -x)
#
# Landing on a snapshot instant shortens a step, which under `filter_cfl = 0`
# is one extra full filter pass per snapshot; five snapshots in ~4000 steps is
# below the level the calibration measured.
#
# The state validation runs permissively. The artificial mass-fraction bound
# holds the interface's undershoot at about 1e-2 (the calibrated behavior of
# `C_Y = 100`, see the shocked-interface rows of reference/CALIBRATION.md),
# which is above the 1e-4 dead band the strict policy rejects at the end of a
# run. The history file records the undershoot; it is one of the quantities
# the comparison is for.
#
# The wall the closing line prints spans `run!` and so includes compiling it
# for this deck's callback closure, 3.2 s at -t 8 on the development
# workstation, a quarter of a 412-step run and 40% of the same run on eight
# ranks. The steady-state ms/step printed beside it averages `solver.wall_step`
# over the steps after the twentieth, which is the figure to compare across
# thread and rank counts; on a hybrid desktop, compare it pinned to the
# performance cores (reference: the hybrid-desktop paragraph of the cluster
# notes).
#
# Scratch tooling, like everything else in bench/: it prints and writes, and
# asserts nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const DEFAULTS = (nx = 768, ny = 48, tfinal = 2.5e-3, cfl = 0.5, alphaf = 0.45,
                  deriv = "c6", filter = "gv", detector = "delta4", mu_sensor = "strain",
                  beta_sensor = "strain", reduction = "sum",
                  C_mu = 0.002, C_beta = 1.0, C_kappa = 0.01, C_D = 0.01, C_Y = 100.0,
                  every = 50, snapshots = "0.5,1.0,1.5,2.0,2.5",
                  prefix = "he_co2_tube")
const opt = CompactLES.script_args(ARGS, DEFAULTS; positional = (:nx, :ny, :tfinal))

# --- Geometry (metres) and initial state, as in examples/shock_tube.jl.
const Lx = 4.0                            # tube length
const Lyz = 0.2                           # square cross-section side
const x_diaphragm = 2.0                   # driver / driven helium split
const x_iface     = 3.0                   # He / CO2 contact (1 m from end wall)
const atm = 101325.0                      # 1 atmosphere [Pa]
const p_driver, p_driven = 10atm, 1atm    # burst pressure ratio 10:1
const T0 = 300.0                          # uniform initial temperature [K]
const A_pert = 0.10Lyz                    # sinusoid amplitude (10% of the span)

# Calorically perfect species. R is the universal gas constant over the molar
# mass; CO2's gamma is its value at 300 K.
const R_universal = 8.314462618           # J / (mol K)
const eos = IdealMixture([IdealSpecies("He"; R = 1e3 * R_universal / 4.0026, gamma = 5 / 3),
                          IdealSpecies("CO2"; R = 1e3 * R_universal / 44.0095, gamma = 1.289)])

function main()
    nx, ny, nz = opt.nx, opt.ny, 1
    hx = Lx / (nx - 1)
    δ = 3hx                               # diaphragm / interface diffuse width
    np = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    prob = Problem(
        name = "He-driven RM shock tube, ideal mixture",
        eos = eos,
        transport = Transport(mu0=0.0),   # Euler + artificial regularization
        domain = ((0.0, Lx), (0.0, Lyz), (0.0, Lyz)),
        bcs = ((SlipWallBC(), SlipWallBC()),
               (PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())),
        ic = (x, y, z) -> begin
            xi = x_iface + A_pert * cos(2π * y / Lyz)
            θ  = tanh_blend(x, xi, δ)      # 0 in helium, 1 in CO2
            p  = p_driven + (p_driver - p_driven) * (1 - tanh_blend(x, x_diaphragm, δ))
            Prim(Y = (1 - θ, θ), p = p, T_ion = T0)
        end)

    art = ArtParams(enabled=true, C_mu=opt.C_mu, C_beta=opt.C_beta,
                    C_kappa=opt.C_kappa, C_D=opt.C_D, C_Y=opt.C_Y,
                    mu_sensor=Symbol(opt.mu_sensor), beta_sensor=Symbol(opt.beta_sensor),
                    reduction=Symbol(opt.reduction), detector=Symbol(opt.detector))
    deriv = opt.deriv == "c10" ? lele_d1_10() :
            opt.deriv == "c6" ? lele_d1_6() :
            error("deriv must be c6 or c10, got $(opt.deriv)")
    filt = opt.filter == "pyranda" ? pyranda_filter() :
           opt.filter == "gv" ? compact_filter(opt.alphaf) :
           error("filter must be gv or pyranda, got $(opt.filter)")
    num = Numerics(n_global=(nx, ny, nz), art=art, deriv=deriv,
                   cfl=opt.cfl, filt=filt,
                   control=StepControl(retries=4, validity=:permissive),
                   filter_interval=1, dims=(np, 1, 1))
    solver, Q = setup(prob, num)

    history = rank == 0 ? open(opt.prefix * "_history.dat", "w") : nothing
    rank == 0 && println(history, "# step t rho_min rho_max umax Y_He_min Y_He_max " *
                                  "Y_CO2_min Y_CO2_max")

    steady_wall = 0.0
    steady_steps = 0
    function diag(solver, Q)
        if solver.step > 20
            steady_wall += solver.wall_step
            steady_steps += 1
        end
        solver.step % opt.every == 0 || return
        n1, n2, n3 = solver.decomp.n_local
        m1 = solver.equations.i_mom[1]
        ρmin, ρmax, umax = Inf, -Inf, 0.0
        y1min, y1max, y2min, y2max = Inf, -Inf, Inf, -Inf
        for k in 1:n3, j in 1:n2, i in 1:n1
            I = gidx(solver, i, j, k)
            ρ = Q[I, 1] + Q[I, 2]
            ρmin = min(ρmin, ρ); ρmax = max(ρmax, ρ)
            umax = max(umax, abs(Q[I, m1] / ρ))
            y1 = Q[I, 1] / ρ; y2 = Q[I, 2] / ρ
            y1min = min(y1min, y1); y1max = max(y1max, y1)
            y2min = min(y2min, y2); y2max = max(y2max, y2)
        end
        comm = solver.decomp.comm
        ρmin = MPI.Allreduce(ρmin, min, comm); ρmax = MPI.Allreduce(ρmax, max, comm)
        umax = MPI.Allreduce(umax, max, comm)
        y1min = MPI.Allreduce(y1min, min, comm); y1max = MPI.Allreduce(y1max, max, comm)
        y2min = MPI.Allreduce(y2min, min, comm); y2max = MPI.Allreduce(y2max, max, comm)
        if rank == 0
            @printf("step %5d  t = %6.3f ms  rho [%.4f, %.4f]  |u|max %6.1f  ",
                    solver.step, 1e3 * solver.t, ρmin, ρmax, umax)
            @printf("Y_He [%+.2e, %.6f]  Y_CO2 [%+.2e, %.6f]\n",
                    y1min, y1max, y2min, y2max)
            @printf(history, "%d %.9e %.9e %.9e %.9e %.9e %.9e %.9e %.9e\n",
                    solver.step, solver.t, ρmin, ρmax, umax, y1min, y1max, y2min, y2max)
            flush(history)
        end
    end

    # Profiles along x: the plane average and two samples through the contact's
    # extreme points. Every rank calls the extraction (collective).
    function snapshot(solver, Q)
        x, ρm = line_profile(solver, Q, :rho)
        _, pm = line_profile(solver, Q, :p)
        _, um = line_profile(solver, Q, :u)
        _, Ym = line_profile(solver, Q, :Y; species = 2)
        jmid = ny ÷ 2 + 1
        cols = Any[x, ρm, pm, um, Ym]
        for j in (1, jmid), name in (:rho, :p, :u, :Y)
            _, v = line_sample(solver, Q, name; index = (j, 1), species = 2)
            push!(cols, v)
        end
        rank == 0 || return
        fname = @sprintf("%s_t%.2fms.dat", opt.prefix, 1e3 * solver.t)
        open(fname, "w") do io
            @printf(io, "# t = %.9e s, step %d\n", solver.t, solver.step)
            println(io, "# x rho_mean p_mean u_mean YCO2_mean " *
                        "rho_y0 p_y0 u_y0 YCO2_y0 rho_ymid p_ymid u_ymid YCO2_ymid")
            for i in eachindex(x)
                for c in cols
                    @printf(io, "%.9e ", c[i])
                end
                println(io)
            end
        end
        println("wrote ", fname)
    end

    instants = [1e-3 * parse(Float64, strip(s)) for s in split(opt.snapshots, ',')
                if !isempty(strip(s))]
    filter!(t -> 0 < t <= opt.tfinal, instants)
    callbacks = isempty(instants) ? (diag,) : (diag, Callback(AtTime(instants), snapshot))

    t0 = time()
    run!(solver, Q; tfinal=opt.tfinal, nmax=1_000_000, callback=callbacks)
    if rank == 0
        close(history)
        @printf("done: %d steps to t = %.3f ms in %.1f s; steady %.2f ms/step over the last %d\n",
                solver.step, 1e3 * solver.t, time() - t0,
                1e3 * steady_wall / max(steady_steps, 1), steady_steps)
    end
end

mpi_main(main)
