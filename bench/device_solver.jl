# G3a acceptance battery (reference/AMR_GPU.md): a whole single-patch solver
# resident on an actual GPU, compared end-to-end against the CPU solver — not
# isolated bodies. Covers the gate's four cases: smooth periodic and
# closed-boundary runs, freestream/GCL, one shock validation (Sod), and a
# short Taylor–Green energy history, in Float64 and Float32. The long-form
# 64³ TGV history on device runs through bench/tgv_energy.jl backend=amdgpu.
#
# Device packages are not CompactLES dependencies; run from an environment
# carrying CompactLES AND the device package (see bench/device_bringup.jl):
#
#   julia --project=<env-with-AMDGPU> -t 8 bench/device_solver.jl backend=amdgpu
#
# The G1/G2 kernels reproduce the host bitwise, and every remaining phase of
# a step is either a pointwise body, a DevicePlan solve, an exact reduction
# (max/min), or an element copy — so the whole-step comparison here is
# expected bitwise as well, and the printout says whether that held.

using CompactLES
using Printf
const CL = CompactLES

opt = script_args(ARGS, (backend = "amdgpu", n = 64, steps = 20))

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
