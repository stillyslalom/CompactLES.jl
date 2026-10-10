# Memory footprint and per-step allocation of configurations shaped like the
# documentation pages and examples:
#
#   planar   2-D, two species, NSCBC inflow and outflow, symmetry planes
#            (examples/shock_bubble.jl)
#   rz       2-D r-z cylindrical, two species, axis and slip walls
#            (examples/vortex_ring_shock.jl)
#   tgv      3-D periodic Taylor-Green, one species, constant viscosity
#            (examples/taylor_green.jl)
#   tiles    2-D periodic, four species, one tiled refined level
#            (docs/literate/advected_bubbles.jl)
#
# For each, the persistent memory the solver holds is reported in bytes per
# interior grid point (summed over this rank's patches), broken down by owner.
# Every array is counted once, under the first owner that reaches it in the
# order of the table, by the memory block behind it, so a view, a stacked
# tile, a workspace shared by several patches or a plan aliased into two
# tuples is not counted twice. The state and the two Runge-Kutta registers
# `run!` holds are the first row.
#
# Then the bytes allocated and the GC time per step of `run!`, from two runs of
# `steps` and `2 steps` steps after `warm` warm-up steps (the difference
# removes the per-call setup), bare, with a `ProgressLog` writing to devnull,
# and with callbacks shaped like the page's own. Last, `@allocated` of each
# phase of an iteration of `run!`, measured on its second call. These are byte
# counts, not timings, so a serial run at `-t 1` gives the same answer as any
# other; the default sizes run in about a minute after the package has loaded.
#
#   julia --project=. -t 1 bench/footprint.jl
#   julia --project=. -t 1 bench/footprint.jl case=tgv steps=10 scale=2
#
# `case` is one of the names above or `all`; `scale` multiplies the grid's
# extent in each active dimension; `detail=true` lists the footprint by field.
# `steps` should be a multiple of every callback cadence (4 and 10), so the
# two measured runs fire them in proportion.
using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using CompactLES.Regions
using Printf
const CL = CompactLES

# --- The walker ----------------------------------------------------------------

# Objects the walk does not enter: type and code objects, MPI handles, IO, and
# functions (a closure in a boundary condition or the stored `Problem` holds
# no field-sized storage).
_skipped(x) = x isa Union{Module,Type,Symbol,String,Function,MPI.Comm,IO,Task,
                          Core.MethodInstance,Core.CodeInstance,Method}

# The memory block behind an array of bits, by which arrays sharing storage
# are recognized, and its size in bytes.
@static if isdefined(Base, :Memory)
    _block(x::Array) = getfield(getfield(x, :ref), :mem)
    _block(x::Memory) = x
else
    _block(x::Array) = x
end

function _walk!(seen::IdDict{Any,Nothing}, x)::Int
    _skipped(x) && return 0
    if x isa Array || (isdefined(Base, :Memory) && x isa Memory)
        if isbitstype(eltype(x))
            b = _block(x)
            haskey(seen, b) && return 0
            seen[b] = nothing
            return sizeof(b)
        end
        haskey(seen, x) && return 0
        seen[x] = nothing
        s = 0
        for i in eachindex(x)
            isassigned(x, i) && (s += _walk!(seen, @inbounds x[i]))
        end
        return s
    end
    isbitstype(typeof(x)) && return 0
    if ismutable(x)
        haskey(seen, x) && return 0
        seen[x] = nothing
    end
    s = 0
    for i in 1:nfields(x)
        isdefined(x, i) && (s += _walk!(seen, getfield(x, i)))
    end
    return s
end

const PATCH_OWNERS = (
    primitives=(:rho, :u, :v, :w, :p, :T_ion, :c, :cp_mix, :Y),
    artificial=(:mu_art, :beta_art, :kappa_art, :D_art),
    geometry=(:inv_J, :area_d, :inv_h, :inv_r, :cot_over_r, :cot_over_r_gcl),
    workspace=(:rhs_workspace, :field_tuples),
    plans=(:folds, :deriv_plans, :div_plans, :filter_plans, :smooth_plans,
           :ring_plans, :pairbuf, :pairout),
    halo=(:decomp,),
    boundary=(:bcs,),
    levels=(:covered, :overwritten, :level_scratch, :ghost_flux, :sensed_fields,
            :child_deep, :reflux_captures),
)
const SOLVER_OWNERS = (geometry=(:metric, :stretch),
                       levels=(:levels, :ghost_sends, :ghost_recvs, :plane_pairs,
                               :regrid))
const ROWS = (:state, :primitives, :artificial, :geometry, :workspace, :plans, :halo,
              :boundary, :levels, :other)
const LABELS = (state="state Q + RK dQ, du", primitives="primitives",
                artificial="artificial coefficients", geometry="metric / geometry",
                workspace="RHS workspace", plans="plans, folds, line scratch",
                halo="halo buffers (decomp)", boundary="boundary conditions",
                levels="levels / AMR / reflux", other="other (solver)")

function footprint(solver, Q, work)
    seen = IdDict{Any,Nothing}()
    bytes = Dict{Symbol,Int}(r => 0 for r in ROWS)
    fields = Dict{Symbol,Int}()
    bytes[:state] = _walk!(seen, Q) + _walk!(seen, work)
    patches = getfield(solver, :patches)
    for row in ROWS
        if haskey(PATCH_OWNERS, row)
            for p in patches, name in PATCH_OWNERS[row]
                b = _walk!(seen, getfield(p, name))
                bytes[row] += b
                fields[name] = get(fields, name, 0) + b
            end
        end
        if haskey(SOLVER_OWNERS, row)
            for name in SOLVER_OWNERS[row]
                b = _walk!(seen, getfield(solver, name))
                bytes[row] += b
                fields[name] = get(fields, name, 0) + b
            end
        end
    end
    bytes[:other] = _walk!(seen, solver)
    points = sum(p -> prod(p.decomp.n_local), patches)
    padded = sum(p -> prod(CL.padded_extent(p.decomp)), patches)
    return bytes, points, padded, fields
end

function report_footprint(name, solver, Q, work, detail)
    bytes, points, padded, fields = footprint(solver, Q, work)
    T = eltype(solver.cfl)
    @printf("\n%s: %d patch(es), %d interior points, padded/interior %.3f, %d-byte %s\n",
            name, CL.npatches(solver), points, padded / points, sizeof(T), T)
    total = sum(values(bytes))
    for row in ROWS
        b = bytes[row]
        b == 0 && continue
        @printf("  %-30s %10.1f B/pt  %6.1f fields  %5.1f%%\n", LABELS[row],
                b / points, b / points / sizeof(T), 100b / total)
    end
    @printf("  %-30s %10.1f B/pt  %6.1f fields  %8.2f MB\n", "total", total / points,
            total / points / sizeof(T), total / 2^20)
    @printf("  %-30s %10.1f B per padded point\n", "", total / padded)
    if detail
        println("  by field, in padded fields:")
        for (k, b) in sort(collect(fields); by=last, rev=true)
            b == 0 && break
            @printf("    %-24s %7.2f\n", k, b / padded / sizeof(T))
        end
    end
    return total / points
end

# --- Allocation per step -------------------------------------------------------

function run_steps(solver, Q, work, n; callback=nothing)
    stats = @timed run!(solver, Q, work; tfinal=1e30, nmax=solver.step + n,
                        callback=callback)
    return stats.bytes, stats.gctime
end

function per_step(solver, Q, work, n; callback=nothing)
    b1, g1 = run_steps(solver, Q, work, n; callback)
    b2, g2 = run_steps(solver, Q, work, 2n; callback)
    return (b2 - b1) / n, 1e3 * (g2 - g1) / n, b1 - (b2 - b1)
end

# Bytes allocated by the second of two calls of `f`.
second(f) = (f(); @allocated f())

_effects(cb::Callback) = (cb.effect!,)
_effects(cbs::Tuple) = mapreduce(_effects, (a, b) -> (a..., b...), cbs)

function phases(solver, Q, work, callback)
    dQ, du = work.dQ, work.du
    dt = solver.dt_prev
    out = Pair{String,Int}[]
    push!(out, "apply_bcs!" => second(() -> CL.apply_bcs!(solver, Q)))
    push!(out, "max_rate" => second(() -> CL.max_rate(solver, Q)))
    rate = solver.rate_prev
    push!(out, "predicted_dt + check_step" => second(() -> begin
        d = CL.predicted_dt(solver, solver.control, rate)
        CL.check_step(solver.control, d, 1.0, 0.0, solver.step, solver.t, solver.cfl)
    end))
    push!(out, "step! (prepared)" => second(() -> CL.step!(solver, Q, dQ, du, dt, true)))
    push!(out, "filter_state!" => second(() -> CL.filter_state!(solver, Q)))
    push!(out, "_presync! + _post_step!" => second(() -> begin
        CL._presync!(solver, Q, true)
        CL._post_step!(solver, Q)
    end))
    push!(out, "_maybe_regrid! (cadence check)" =>
          second(() -> CL._maybe_regrid!(solver, Q, work, nothing)))
    push!(out, "callback_next_time" =>
          second(() -> CL.callback_next_time(callback, solver)))
    push!(out, "run_callbacks!, no trigger due" =>
          second(() -> CL.run_callbacks!(callback, solver, Q)))
    for (k, effect!) in enumerate(_effects(callback))
        push!(out, "page callback $k, one firing" => second(() -> effect!(solver, Q)))
    end
    progress = ProgressLog(io=devnull).effect!
    push!(out, "ProgressLog, one firing" => second(() -> progress(solver, Q)))
    return out
end

function report_steps(name, solver, Q, work, page_callbacks, nsteps, warm)
    run_steps(solver, Q, work, warm)
    progress = ProgressLog(every=10, io=devnull)
    @printf("  run! per step (%d steps):\n", nsteps)
    rows = Any[]
    for (label, cb) in (("bare", nothing), ("ProgressLog(every = 10)", progress),
                        ("page callbacks", page_callbacks))
        # Warm-up of a length every trigger fires within, so the measured
        # runs compile nothing.
        run_steps(solver, Q, work, nsteps; callback=cb)
        bytes, gc_ms, fixed = per_step(solver, Q, work, nsteps; callback=cb)
        @printf("    %-26s %12.0f B/step  %8.3f ms GC/step  %10.0f B per call\n",
                label, bytes, gc_ms, fixed)
        push!(rows, (label, bytes))
    end
    @printf("  phases (bytes on the second call):\n")
    for (label, b) in phases(solver, Q, work, page_callbacks)
        @printf("    %-34s %12d\n", label, b)
    end
    return rows
end

# --- Cases -------------------------------------------------------------------

function planar(scale)
    eos = IdealMixture(["Air", "He"])
    p0, T0 = 101_325.0, 295.0
    air = Prim(Y=mass_fractions(eos, "Air" => 1.0; basis=:mass), p=p0, T_ion=T0)
    helium = Prim(Y=mass_fractions(eos, "He" => 0.72, "Air" => 0.28; basis=:mass),
                  p=p0, T_ion=T0)
    incident = shock_jump(eos, air, 1.22)
    H, R, xc, x_shock, Lx = 0.0445, 0.025, 0.06, 0.025, 0.26
    problem = Problem(eos=eos, domain=((0.0, Lx), (0.0, H), (0.0, 1.0)),
                      bcs=((NSCBCInflowBC(incident.post), NSCBCOutflowBC(pinf=p0)),
                           (SymmetryPlaneBC(), SymmetryPlaneBC()), PeriodicBC()),
                      ic=Layers(air, Slab(1, hi=x_shock) => incident.post,
                                Cylinder((xc, 0.0, 0.0), R) => helium))
    ny = 32scale
    nx = round(Int, Lx / (H / ny)) + 1
    solver, Q = setup(problem, Numerics(n_global=(nx, ny, 1)))
    sample = Callback(EveryStep(4), function (solver, Q)
        _, _, X = field_slice(solver, Q, :X; species=2)
        _, _, p = field_slice(solver, Q, :p)
        nothing
    end)
    return solver, Q, sample
end

function rz(scale)
    eos = IdealMixture(["Air", "SF6"])
    p0, T0 = 101_325.0, 295.0
    air = Prim(Y=(1.0, 0.0), p=p0, T_ion=T0)
    sf6 = Prim(Y=(0.0, 1.0), p=p0, T_ion=T0)
    incident = shock_jump(eos, air, 1.36; dim=3, direction=-1)
    R, H = 0.0356, 0.25
    problem = Problem(eos=eos, metric=CylindricalMetric(),
                      domain=((0.0, R), (0.0, 2π), (0.0, H)),
                      bcs=((AxisBC(), SlipWallBC()), PeriodicBC(),
                           (SlipWallBC(), DirichletBC(incident.post))),
                      ic=Layers(air, Slab(3, hi=0.12) => sf6; width=Cells(2)))
    nr, nz = 24scale, 96scale
    solver, Q = setup(problem, Numerics(n_global=(nr, 1, nz)))
    record = Callback(EveryStep(4), function (solver, Q)
        line_sample(solver, Q, :p; dim=3, index=(nr, 1))
        line_sample(solver, Q, :X; dim=3, index=(1, 1), species=2)
        nothing
    end)
    return solver, Q, record
end

function tgv(scale)
    p0 = 10.0^2 / 1.4
    problem = Problem(eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                      transport=ConstantTransport(mu0=1 / 1600),
                      domain=((0.0, 2pi), (0.0, 2pi), (0.0, 2pi)),
                      bcs=(PeriodicBC(), PeriodicBC(), PeriodicBC()),
                      ic=(x, y, z) -> Prim(u=(sin(x) * cos(y) * cos(z),
                                              -cos(x) * sin(y) * cos(z), 0.0),
                                           p=p0 + (cos(2x) + cos(2y)) *
                                                  (cos(2z) + 2) / 16,
                                           rho=1.0))
    N = 24scale
    solver, Q = setup(problem, Numerics(n_global=(N, N, N)))
    record = Callback(EveryStep(4), function (solver, Q)
        rho, u = field_array(solver, Q, :rho), field_array(solver, Q, :u)
        volume_integral(solver, @. rho * u^2 / 2)
        dissipation_rate(solver, Q)
        nothing
    end)
    return solver, Q, record
end

function tiles(scale)
    names = ["Air", "He", "SF6", "Kr"]
    eos = IdealMixture(names)
    p0, T0, L, U, R, w = 101_325.0, 300.0, 0.96, 300.0, 0.04, 0.01
    centers = ((0.30, 0.24), (0.92, 0.48), (0.92, 0.92))
    wrap(d) = d - L * round(d / L)
    bubble(b, x, y) = (1 - tanh((hypot(wrap(x - centers[b][1]),
                                       wrap(y - centers[b][2])) - R) / w)) / 2
    function initial_state(x, y, z)
        X = ntuple(b -> bubble(b, x, y), 3)
        return Prim(Y=mass_fractions(eos, "Air" => 1 - sum(X), "He" => X[1],
                                     "SF6" => X[2], "Kr" => X[3]; basis=:mole),
                    p=p0, T_ion=T0, u=(U, U, 0.0))
    end
    problem = Problem(eos=eos, domain=((0.0, L), (0.0, L), (0.0, 1.0)),
                      bcs=(PeriodicBC(), PeriodicBC(), PeriodicBC()), ic=initial_state)
    n = 96scale
    solver, states = setup(problem, Numerics(n_global=(n, n, 1), amr=AMR(tile=12)))
    record = Callback(EveryStep(10), function (solver, states)
        for snap in field_snapshot(solver, states; fields=(:p,))
            maximum(abs, snap[:p] .- p0)
        end
        nothing
    end)
    return solver, states, record
end

const CASES = (planar=planar, rz=rz, tgv=tgv, tiles=tiles)

function main()
    opt = CL.script_args(ARGS, (case="all", steps=20, warm=4, scale=1, detail=false))
    names = opt.case == "all" ? keys(CASES) : (Symbol(opt.case),)
    @printf("julia %s, %d thread(s)\n", VERSION, Threads.nthreads())
    for name in names
        solver, Q, callbacks = CASES[name](opt.scale)
        work = CL.Workspace(Q)
        report_footprint(name, solver, Q, work, opt.detail)
        report_steps(name, solver, Q, work, callbacks, opt.steps, opt.warm)
    end
end

# Run as a script; an `include` from another script takes the cases alone.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
