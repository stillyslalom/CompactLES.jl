# The 3-D air/SF6 Richtmyer–Meshkov run behind the README header
# (docs/figures/readme_header.jl draws the header from its frames).
#
# A Mach 1.21 shock in acetone-seeded air strikes a multimode SF6 interface in a
# narrow tube. Lab frame, as docs/literate/richtmyer_meshkov.jl: NSCBC inflow of
# shocked air at x = 0, its target switched by a phase change once the reflected
# shock has left, NSCBC outflow at x = Lx, periodic y and z. The defaults give
# 800 × 192 × 48 nodes at Δx = 0.3125 mm and about 12 000 steps to 2.5 ms after
# impact, which fits a 16 GB GPU. The run writes mid-plane frames every 0.05 ms,
# SF6 volumes every 0.25 ms, checkpoints and a progress log into `out`:
#
#   julia --project=<environment carrying CompactLES and AMDGPU> -t 8 \
#       docs/figures/readme_header_run.jl
#   ... restart=true      continue from <out>/ckpt/state
#   ... backend=cpu       host storage, for a small test (h=1.0 tend=0.3)

using CompactLES
using CompactLES: CPUBackend
using CompactLES.Regions
using Printf, Serialization, Dates
MPI.Initialized() || MPI.Init(threadlevel = :funneled)

const opt = CompactLES.script_args(ARGS, (backend = "amdgpu", h = 0.3125, Lx = 0.25,
                                          Ly = 0.06, Lz = 0.015, lmin = 7.5,
                                          lmax = 30.0, arms = 1.2, delta = 2.0,
                                          tend = 2.5, frame = 0.05, volume = 0.25,
                                          ckpt_minutes = 45.0, margin = 12.0,
                                          nmax = typemax(Int),
                                          out = joinpath(@__DIR__, ".readme_header_run"),
                                          restart = false))

function make_backend(spec)
    spec == "cpu" && return CPUBackend()
    @eval using AMDGPU
    @eval AMDGPU.functional() || error("AMDGPU is not functional here")
    return DeviceBackend(@eval ROCBackend())
end

function build(backend)
    eos = IdealMixture(["Air", "C3H6O,acetone", "SF6"])
    p0, T0 = 92_555.0, 296.0
    acetone = 0.244
    air = Prim(Y = mass_fractions(eos, "Air" => 1 - acetone, "C3H6O,acetone" => acetone;
                                  basis = :mole), p = p0, T_ion = T0)
    sf6 = Prim(Y = mass_fractions(eos, "SF6" => 1.0; basis = :mole), p = p0, T_ion = T0)
    incident = shock_jump(eos, air, 1.21)
    impact = riemann_interface(eos, incident.post, sf6)
    W = incident.shock_speed
    h = opt.h * 1e-3
    Lx, Ly, Lz = opt.Lx, opt.Ly, opt.Lz
    nx, ny, nz = round.(Int, (Lx, Ly, Lz) ./ h)
    x_shock, x_i = 0.02, 0.03
    t_impact = (x_i - x_shock) / W
    t_reflected = t_impact + x_i / abs(impact.left_speed)

    # Modes in a wavelength band, amplitude ∝ 1/k, deterministic phases.
    modes = Tuple{Float64,Float64,Float64,Float64}[]
    mmax, nmmax = round(Int, Ly / (opt.lmin * 1e-3)), round(Int, Lz / (opt.lmin * 1e-3))
    for m in 0:mmax, n in -nmmax:nmmax
        (m == 0 && n <= 0) && continue
        ky, kz = 2pi * m / Ly, 2pi * n / Lz
        λ = 2pi / hypot(ky, kz)
        opt.lmin * 1e-3 <= λ <= opt.lmax * 1e-3 || continue
        s = sin(12.9898m + 78.233n + 0.5) * 43758.5453
        push!(modes, (ky, kz, 1 / hypot(ky, kz), 2pi * (s - floor(s))))
    end
    rms = isempty(modes) ? 1.0 : sqrt(sum(abs2(m[3]) for m in modes) / 2)
    amps = [(ky, kz, a * opt.arms * 1e-3 / rms, φ) for (ky, kz, a, φ) in modes]
    eta(y, z) = sum((a * cos(ky * y + kz * z + φ) for (ky, kz, a, φ) in amps); init = 0.0)

    bcs(inflow) = ((NSCBCInflowBC(inflow), NSCBCOutflowBC(pinf = p0)),
                   PeriodicBC(), PeriodicBC())
    problem(inflow) = Problem(
        name = "air/SF6 multimode RMI",
        eos = eos,
        domain = ((0.0, Lx), (0.0, Ly), (0.0, Lz)),
        bcs = bcs(inflow),
        ic = Layers(air,
                    Layer(Slab(1, lo = (y, z) -> x_i + eta(y, z)), sf6;
                          width = opt.delta * 1e-3 / sqrt(pi)),
                    Slab(1, hi = x_shock) => incident.post; profile = :erf),
    )
    numerics = Numerics(n_global = (nx, ny, nz), execution = Execution(storage = backend))
    return (; eos, incident, impact, problem, bcs, numerics, nx, ny, nz, h, Lx, Ly, Lz,
            t_impact, t_reflected, nmodes = length(modes), x_i)
end

function production(backend)
    c = build(backend)
    out = abspath(opt.out)
    mkpath(joinpath(out, "frames")); mkpath(joinpath(out, "volumes"))
    mkpath(joinpath(out, "ckpt"))
    prefix = joinpath(out, "ckpt", "state")
    logio = open(joinpath(out, "progress.log"), "a")
    note(s) = (println(logio, Dates.format(now(), "HH:MM:SS"), "  ", s); flush(logio))
    ms(t) = 1e3 * (t - c.t_impact)

    t_end = c.t_impact + opt.tend * 1e-3
    meta = (; opt, c.nx, c.ny, c.nz, c.h, c.Lx, c.Ly, c.Lz, c.t_impact, c.t_reflected,
            c.nmodes, c.x_i, species = ["Air", "C3H6O,acetone", "SF6"])
    serialize(joinpath(out, "meta.jls"), meta)
    note(@sprintf("grid %d × %d × %d = %.2f M nodes, h = %.4f mm, %d modes",
                  c.nx, c.ny, c.nz, c.nx * c.ny * c.nz / 1e6, 1e3c.h, c.nmodes))
    note(@sprintf("impact %.4f ms, inflow switch %.4f ms, end %.4f ms",
                  1e3c.t_impact, 1e3c.t_reflected, 1e3t_end))

    # The phase is decided by the clock, which a restart reads from the sidecar.
    phase_of(t) = t >= c.t_reflected ? 2 : 1
    inflow(ph) = ph == 1 ? c.incident.post : c.impact.left
    if opt.restart
        side = deserialize(prefix * ".meta.jls")
        ph = phase_of(side.t)
        solver, Q = setup(c.problem(inflow(ph)), c.numerics)
        load_checkpoint!(solver, Q, prefix)
        note(@sprintf("restart at step %d, t - t_impact = %.4f ms, phase %d",
                      solver.step, ms(solver.t), ph))
    else
        ph = 1
        t_setup = @elapsed solver, Q = setup(c.problem(inflow(1)), c.numerics)
        note(@sprintf("setup %.1f s", t_setup))
    end

    zmid, ymid = cld(c.nz, 2), cld(c.ny, 2)
    Rk = c.eos.Rk
    xs = collect(range(0, c.Lx, length = c.nx))
    stop_reason = Ref("")
    function frame!(solver, Q)
        k = round(Int, (solver.t - c.t_impact) / (opt.frame * 1e-3))
        name = @sprintf("frame_%+05d", k)
        fxy = field_snapshot(solver, Q; fields = (:Y, :rho, :p), normal = 3, index = zmid)
        fxz = field_snapshot(solver, Q; fields = (:Y, :rho), normal = 2, index = ymid)
        Xxy, Xxz = molefrac(fxy[:Y], Rk), molefrac(fxz[:Y], Rk)
        f32(a) = Float32.(a)
        serialize(joinpath(out, "frames", name * ".jls"),
                  (t = solver.t, t_after = solver.t - c.t_impact, step = solver.step,
                   xy = (X = f32(Xxy), rho = f32(fxy[:rho]), p = f32(fxy[:p])),
                   xz = (X = f32(Xxz), rho = f32(fxz[:rho]))))
        # Leading edge of the light gas (air + acetone) in the x-y mid-plane; an
        # interface reaching the NSCBC outflow fails, so stop short of the face.
        sf6 = selectdim(Xxy, ndims(Xxy), 3)
        light = vec(maximum(1 .- sf6; dims = 2:ndims(sf6)))
        i = findlast(>(0.02), light)
        front = i === nothing ? 0.0 : xs[i]
        note(@sprintf("frame %s  step %d  t - t_impact = %.4f ms  light-gas front %.1f mm",
                      name, solver.step, ms(solver.t), 1e3front))
        if c.Lx - front < opt.margin * 1e-3
            stop_reason[] = @sprintf("light-gas front at %.1f mm, inside the %.0f mm margin",
                                     1e3front, opt.margin)
            return true
        end
        return nothing
    end
    function volume!(solver, Q)
        k = round(Int, (solver.t - c.t_impact) / (opt.volume * 1e-3))
        X = molefrac(field_snapshot(solver, Q; fields = (:Y,))[:Y], Rk)
        serialize(joinpath(out, "volumes", @sprintf("sf6_%+03d.jls", k)),
                  (t_after = solver.t - c.t_impact, step = solver.step,
                   X_sf6 = Float32.(selectdim(X, 4, 3)),
                   X_acetone = Float32.(selectdim(X, 4, 2))))
        note(@sprintf("volume %d written", k))
        nothing
    end
    last_ckpt = Ref(time())
    function checkpoint!(solver, Q)
        t0 = time()
        save_checkpoint(solver, Q, prefix)
        serialize(prefix * ".meta.jls", (t = solver.t, step = solver.step))
        last_ckpt[] = time()
        note(@sprintf("checkpoint at step %d (%.1f s)", solver.step, time() - t0))
    end
    stamp = Ref((time(), solver.step, solver.t))
    function progress!(solver, Q)
        if solver.step % 50 == 0
            w0, s0, t0 = stamp[]
            w = time()
            steps = max(1, solver.step - s0)
            rate = (w - w0) / steps
            remaining = (t_end - solver.t) / max(eps(), (solver.t - t0) / steps)
            note(@sprintf("step %6d  t - t_impact = %.4f ms  dt = %.3e  %.3f s/step",
                          solver.step, ms(solver.t), solver.dt_prev, rate) *
                 @sprintf("  ≈ %.1f h left", remaining * rate / 3600))
            stamp[] = (w, solver.step, solver.t)
        end
        time() - last_ckpt[] > 60 * opt.ckpt_minutes && checkpoint!(solver, Q)
        nothing
    end
    callbacks = (Callback(EveryTime(opt.frame * 1e-3; start = c.t_impact), frame!),
                 Callback(EveryTime(opt.volume * 1e-3; start = c.t_impact), volume!),
                 Callback(EveryStep(1), progress!))
    control = StepControl(retries = 3, dt_min = 1e-9)
    try
        if ph == 1
            run!(solver, Q; tfinal = min(c.t_reflected, t_end), nmax = opt.nmax,
                 callback = callbacks, control = control)
            if solver.t >= c.t_reflected && isempty(stop_reason[]) && solver.step < opt.nmax
                ts = @elapsed solver, Q = setup(solver, Q; bcs = c.bcs(inflow(2)))
                # The previous phase's device arrays stay allocated until a
                # collection. With both phases resident a 16 GB card spills into
                # host memory and each step takes 60% longer.
                GC.gc(true)
                note(@sprintf("inflow switched at step %d, t - t_impact = %.4f ms (%.1f s)",
                              solver.step, ms(solver.t), ts))
                ph = 2
            end
        end
        if ph == 2 && isempty(stop_reason[]) && solver.t < t_end && solver.step < opt.nmax
            run!(solver, Q; tfinal = t_end, nmax = opt.nmax, callback = callbacks,
                 control = control)
        end
        checkpoint!(solver, Q)
        note(@sprintf("finished: step %d, t - t_impact = %.4f ms", solver.step, ms(solver.t)) *
             (isempty(stop_reason[]) ? "" : ", stopped: " * stop_reason[]))
    catch err
        note("FAILED: " * sprint(showerror, err)[1:min(end, 600)])
        try
            note(string(state_report(solver, Q)))
        catch
        end
        throw(err)
    finally
        close(logio)
    end
end

# Mole fractions from mass fractions stacked along the last dimension, computed
# on the host because field_snapshot's :X indexes device storage per element.
function molefrac(Y, Rk)
    d = ndims(Y)
    YR = Y .* reshape(Rk, ntuple(i -> i == d ? length(Rk) : 1, d))
    return YR ./ max.(sum(YR; dims = d), eps())
end

production(make_backend(opt.backend))
