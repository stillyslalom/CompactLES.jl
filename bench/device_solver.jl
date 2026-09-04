# Whole-solver device acceptance battery (reference/AMR_GPU.md): a solver
# resident on an actual GPU, compared end-to-end against the CPU solver — not
# isolated bodies. Covers four case families: smooth periodic and
# closed-boundary runs, freestream/GCL, one shock validation (Sod), and a
# short Taylor–Green energy history, in Float64 and Float32. The long-form
# 64³ TGV history on device runs through bench/tgv_energy.jl backend=amdgpu.
#
# Device packages are not CompactLES dependencies; run from an environment
# carrying CompactLES and the device package (see bench/device_bringup.jl):
#
#   julia --project=<env-with-AMDGPU> -t 8 bench/device_solver.jl backend=amdgpu
#
# The pointwise and line-solve kernels reproduce the host bitwise, and
# every remaining phase of a step is either a pointwise body, a DevicePlan solve, an exact reduction
# (max/min), or an element copy — so the whole-step comparison here is
# expected bitwise as well, and the printout says whether that held.

using CompactLES
using Printf
const CL = CompactLES

opt = script_args(ARGS, (backend = "amdgpu", n = 64, steps = 20, sync = false))

# `sync=true` restores the synchronize-per-launch conservative mode, so
# the launch-policy delta is one flag apart on identical cases. Either mode
# must be bitwise; only the wall time may move.
CompactLES.DEVICE_SYNC[] = opt.sync

device_array, ka_backend = if opt.backend == "amdgpu"
    @eval using AMDGPU
    @eval AMDGPU.functional() || error("AMDGPU is not functional here")
    @eval println("device: ", AMDGPU.device())
    @eval (ROCArray, ROCBackend())
elseif opt.backend == "cuda"
    @eval using CUDA
    @eval CUDA.functional() || error("CUDA is not functional here")
    @eval println("device: ", CUDA.device())
    @eval (CuArray, CUDABackend())
else
    error("backend must be amdgpu or cuda, got $(opt.backend)")
end

const per = (PeriodicBC(), PeriodicBC())

function compare_run(build, label; nmax, tfinal)
    s1, Q1 = build(CPUBackend())
    t_cpu = @elapsed run!(s1, Q1; tfinal=tfinal, nmax=nmax)
    s2, Q2 = build(DeviceBackend(ka_backend))
    t_dev = @elapsed run!(s2, Q2; tfinal=tfinal, nmax=nmax)
    Qd = Array(parent(Q2))
    dmax = maximum(abs.(Qd .- parent(Q1)))
    @printf("%-34s steps %3d/%3d  max|dev-cpu| = %g%s\n", label,
            s1.step, s2.step, dmax, dmax == 0 ? "  (bitwise)" : "")
    return (s1.step == s2.step, dmax, t_cpu, t_dev)
end

function main(opt)
    T = Float64

    # --- smooth periodic (TGV-shaped IC, multispecies sensors active) -----
    compare_run("periodic smooth 32^3", nmax=10, tfinal=0.5) do backend
        s = Solver(n_global=(32, 32, 32), L_domain=(2π, 2π, 2π),
                   bcs=(per, per, per), transport=Transport(mu0=1 / 1600),
                   backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(
            u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
            p=71.43 + (cos(2x) + cos(2y)) * (cos(2z) + 2) / 16, rho=1.0))
        return s, Q
    end

    # --- closed boundaries, two species ----------------------------------
    compare_run("closed multispecies tube", nmax=10, tfinal=0.02) do backend
        eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                            IdealSpecies{Float64}("heavy", 0.2, 1.09)])
        s = Solver(n_global=(32, 24, 12), L_domain=(1.0, 0.6, 0.3), eos=eos,
                   bcs=((SlipWallBC(), SlipWallBC()), per, per),
                   backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((x - 0.5) / 0.05))
            Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ,
                 u=(0.1 * sin(2π * y / 0.6), 0.0, 0.05 * cos(2π * z / 0.3)))
        end)
        return s, Q
    end

    # --- NSCBC inflow/outflow with isothermal no-slip wall ----------------
    compare_run("NSCBC duct", nmax=10, tfinal=0.05) do backend
        s = Solver(n_global=(48, 16, 1), L_domain=(2.0, 1.0, 1.0),
                   bcs=((NSCBCInflowBC(u=(0.3, 0.0, 0.0), T_ion=1.0),
                         NSCBCOutflowBC(pinf=1.0)),
                        (NoSlipWallBC(Twall=1.0), SlipWallBC()), per),
                   backend=backend, cfl=0.4, filter_cfl=0.6)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(rho=1.0, p=1.0, u=(0.3, 0.0, 0.0)))
        return s, Q
    end

    # --- resolved-θ cylindrical axis fold ---------------------------------
    compare_run("cylindrical axis fold", nmax=8, tfinal=0.02) do backend
        s = Solver(n_global=(20, 16, 12), L_domain=(1.0, 2π, 0.5),
                   bcs=((AxisBC(), SlipWallBC()), per, per),
                   metric=CylindricalMetric(), backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, z) ->
            Prim(u=(0.05r * cos(θ), 0.2r, 0.05 * sin(2π * z / 0.5)),
                 p=1.0 + 0.02r^2, rho=1.0))
        return s, Q
    end

    # --- freestream/GCL: spherical origin + poles, one RHS ----------------
    function sph(backend)
        s = Solver(n_global=(12, 12, 12), L_domain=(1.0, Float64(π), 2π),
                   bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per),
                   metric=SphericalMetric(), backend=backend,
                   art=ArtParams(enabled=false), transport=Transport(mu0=0.0))
        Q = allocate_state(s)
        initialize!(s, Q, (r, θ, φ) -> Prim(rho=1.0, p=1.0, u=(0.0, 0.0, 0.0)))
        return s, Q
    end
    s1, F1 = sph(CPUBackend())
    dQ1 = zero(F1)
    compute_rhs!(s1, F1, dQ1)
    s2, F2 = sph(DeviceBackend(ka_backend))
    dQ2 = zero(F2)
    compute_rhs!(s2, F2, dQ2)
    r1, m1 = max_rate(s1, F1), max_rate(s2, F2)
    df = maximum(abs.(Array(parent(dQ2)) .- parent(dQ1)))
    @printf("%-34s |dQ|max cpu %g dev %g  max|dev-cpu| = %g%s\n",
            "spherical freestream/GCL", maximum(abs.(parent(dQ1))),
            maximum(abs.(Array(parent(dQ2)))), df,
            df == 0 ? "  (bitwise)" : "")
    @printf("%-34s cpu %s dev %s  %s\n", "max_rate", r1, m1,
            r1 == m1 ? "(equal)" : "(DIFFER)")

    # --- Sod shock validation (Dirichlet-held ends, art on) ---------------
    compare_run("Sod shock N=400", nmax=2000, tfinal=0.2) do backend
        N = 400
        h = 1.0 / (N - 1)
        δ = 2h
        bcs1 = (DirichletBC((x, y, z, t) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1.0)),
                DirichletBC((x, y, z, t) -> Prim(rho=0.125, u=(0.0, 0.0, 0.0), p=0.1)))
        s = Solver(n_global=(N, 1, 1), L_domain=(1.0, h, h),
                   bcs=(bcs1, per, per), transport=Transport(mu0=0.0),
                   cfl=0.4, backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = tanh_blend(x, 0.5, δ)
            Prim(rho=(1 - θ) * 1.0 + θ * 0.125, u=(0.0, 0.0, 0.0),
                 p=(1 - θ) * 1.0 + θ * 0.1)
        end)
        return s, Q
    end

    # --- Refined solver resident on device --------------------------------
    function refined_wave(backend; kw...)
        N = 96
        s = Solver(n_global=(N, 1, 1), L_domain=(2π, 1.0, 1.0),
                   bcs=(per, per, per), art=ArtParams(enabled=false),
                   filter_interval=0,
                   refine=BlockRegion((N ÷ 2 - N ÷ 12, 0, 0), (N ÷ 6, 1, 1)),
                   backend=backend; kw...)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) ->
            Prim(u=(0.5, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x)))
        run!(s, states; tfinal=0.3)
        return s, states
    end
    for (label, kw) in (("refined static wave", (;)),
                        ("refined subcycle+regrid",
                         (subcycle=true, regrid_interval=20, tag_buffer=8)))
        s1, q1 = refined_wave(CPUBackend(); kw...)
        s2, q2 = refined_wave(DeviceBackend(ka_backend); kw...)
        dmax = maximum(maximum(abs.(Array(parent(q2[i])) .- parent(q1[i])))
                       for i in 1:2)
        @printf("%-34s steps %3d/%3d  max|dev-cpu| = %g%s\n", label,
                s1.step, s2.step, dmax, dmax == 0 ? "  (bitwise)" : "")
    end

    # --- Patch layout and tiled level on device ---------------------------
    # The interface records stage through the backend, the transfer chain
    # runs on the fine patches' device scratch, the tag sweep evaluates on
    # the device, and a tiled level's tiles take stacked storage: the
    # right-hand side, the stage update and the filter launch once per level
    # and the compact solves batch every tile's lines behind one interface
    # fence (reference/AMR_GPU.md, launch policy). Each case prints the
    # device step against the CPU step and the stacks the device level
    # holds; the per-tile launch floor this batching amortizes is what the
    # 1-D tiled case measured before it (0.11 s against 1.1 ms per step).
    function states_diff(q1, q2)
        return maximum(maximum(abs.(Array(parent(q2[i])) .- parent(q1[i])))
                       for i in eachindex(q1))
    end
    function stack_note(s)
        levs = getfield(s, :levels)
        length(levs) > 1 || return ""
        return "  stacks " * join([string(length(st.members)) for st in levs[2].stacks], "+")
    end
    function two_slabs(backend)
        s = Solver(n_global=(96, 1, 1), L_domain=(2π, 1.0, 1.0),
                   bcs=(per, per, per), art=ArtParams(enabled=false),
                   transport=Transport(mu0=2e-2), patch_grid=(2, 1, 1),
                   backend=backend)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) ->
            Prim(u=(0.5 + 0.1 * sin(2x), 0, 0), p=1.0 + 0.05 * cos(x),
                 rho=1.0 + 0.2 * sin(x)))
        run!(s, states; tfinal=0.3)
        return s, states
    end
    function tiled_sod(backend)
        wall2 = (SlipWallBC(), SlipWallBC())
        s = Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
                   bcs=(wall2, per, per), cfl=0.2, subcycle=true,
                   regrid_interval=5, refine=BlockRegion((85, 0, 0), (31, 1, 1)),
                   tile=8, backend=backend)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) -> x < 0.5 ?
            Prim(u=(0, 0, 0), p=1.0, rho=1.0) : Prim(u=(0, 0, 0), p=0.1, rho=0.125))
        run!(s, states; tfinal=0.06, nmax=400)
        return s, states
    end
    # A 3-D tiled level: twelve 16^3 tiles in one stack along an active
    # dimension, the configuration whose per-tile launch floor the stacked
    # storage is meant to amortize. Static refinement, so the timing is the
    # step's alone.
    function tiled_box(backend)
        N = 30
        s = Solver(n_global=(N, N, N), L_domain=(2π, 2π, 2π), bcs=(per, per, per),
                   transport=Transport(mu0=1e-2), subcycle=true,
                   refine=BlockRegion((11, 11, 11), (7, 7, 13)), tile=6,
                   backend=backend)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) ->
            Prim(u=(0.3 + 0.1 * sin(x) * cos(y) * cos(z),
                    -0.1 * cos(x) * sin(y) * cos(z), 0.05 * sin(z)),
                 p=1.0 + 0.02 * cos(2x) * cos(z), rho=1.0 + 0.1 * sin(x + y + z)))
        run!(s, states; tfinal=1e6, nmax=6)
        return s, states
    end
    for (label, build) in (("two viscous slabs", two_slabs),
                           ("tiled subcycled regridding Sod", tiled_sod),
                           ("tiled 3-D box, 12 tiles", tiled_box))
        s1, q1 = build(CPUBackend())
        s2, q2 = build(DeviceBackend(ka_backend))
        dmax = states_diff(q1, q2)
        @printf("%-34s steps %3d/%3d  tiles %d/%d  max|dev-cpu| = %g%s%s\n", label,
                s1.step, s2.step, length(q1), length(q2), dmax,
                dmax == 0 ? "  (bitwise)" : "", stack_note(s2))
        # Warm timing: fresh solvers, the first pass having compiled.
        sc, _ = build(CPUBackend())
        sd, _ = build(DeviceBackend(ka_backend))
        @printf("  timing (warm): cpu %.4f s/step   device %.4f s/step   ratio %.2fx\n",
                sc.wall_total / sc.step, sd.wall_total / sd.step,
                (sc.wall_total / sc.step) / (sd.wall_total / sd.step))
    end

    # --- Float32 TGV short history + step timing at n^3 -------------------
    for Tp in (Float64, Float32)
        n = opt.n
        function tgv(backend)
            γ = Tp(1.4)
            c0 = Tp(10)
            p0 = c0^2 / γ
            s = Solver(n_global=(n, n, n),
                       L_domain=(Tp(2π), Tp(2π), Tp(2π)),
                       bcs=(per, per, per), eos=single_species(Tp),
                       transport=Transport{Tp}(mu0=Tp(1 / 1600)),
                       art=ArtParams{Tp}(enabled=false),
                       deriv=lele_d1_6(Tp), filt=compact_filter(Tp(0.45), Tp),
                       cfl=Tp(0.6), backend=backend)
            Q = allocate_state(s)
            initialize!(s, Q, (x, y, z) -> Prim(
                u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z),
                   zero(Tp)),
                p=p0 + one(Tp) / Tp(16) * (cos(Tp(2)x) + cos(Tp(2)y)) *
                    (cos(Tp(2)z) + Tp(2)), rho=one(Tp)))
            return s, Q
        end
        same, dmax, t_cpu, t_dev =
            compare_run(tgv, "TGV $(Tp) $(n)^3 x$(opt.steps) steps";
                        nmax=opt.steps, tfinal=1e6)
        # Warmed second measurement: reuse fresh solvers so compilation from
        # the first pass does not pollute the timing.
        sc, Qc = tgv(CPUBackend())
        run!(sc, Qc; tfinal=1e6, nmax=opt.steps)
        sd, Qd = tgv(DeviceBackend(ka_backend))
        run!(sd, Qd; tfinal=1e6, nmax=opt.steps)
        @printf("  timing (warm): cpu %.3f s/step (%d thr)   ",
                sc.wall_total / sc.step, Threads.nthreads())
        @printf("device %.3f s/step   ratio %.2fx\n",
                sd.wall_total / sd.step,
                (sc.wall_total / sc.step) / (sd.wall_total / sd.step))
    end
    return nothing
end

main(opt)
