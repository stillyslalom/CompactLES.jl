# NASA-9 thermodynamics on an actual GPU: the device mirror of a
# `Nasa9Mixture` against the host mixture, in Float64 and Float32.
#
# Three parts:
#
# - Recovery. The temperature inversion, its status, the admissibility
#   verdict and the species and mixture evaluations, launched as one pointwise
#   body over a set of states the inversion finds hard (interval joins
#   approached to a few ulp, the edges of the fitted range and beyond them
#   under each extrapolation policy, energies with no root in the search
#   bounds, compositions with no bracket) and over a pseudo-random sweep of
#   temperature and composition. Prints the largest relative difference per
#   quantity and the number of points whose status or verdict differs.
# - Full runs. A wall-bounded two-species case crossing the 1000 K join, a
#   refined periodic wave through the ghost interface flux, and a shock tube,
#   each on the CPU and on the device; prints the largest difference per
#   conserved component relative to that component's magnitude.
# - Local timing of the recovery and of a whole step against `IdealMixture` at
#   `n`³, as a check for pitfalls (register spills, launch failures, a
#   recovery far out of proportion to the rest of the step), not a
#   performance claim: consumer RDNA2 runs FP64 at 1/16 of its FP32 rate.
#
# The host and the device do not share the logarithm and the integer powers
# of the fits, so agreement to round-off rather than bit-for-bit equality is
# the expected result here. The KernelAbstractions CPU backend runs the same
# mirror bitwise against the host; test/device_tests.jl holds that check.
#
# Device packages are not CompactLES dependencies; run from an environment
# carrying CompactLES and the device package (see probes/device_bringup.jl):
#
#   julia --project=<env-with-AMDGPU> -t 8 bench/device_nasa9.jl backend=amdgpu
#   julia --project=<env-with-AMDGPU> -t 8 bench/device_nasa9.jl n=32 steps=5
#   julia --project=<env-with-AMDGPU> -t 8 bench/device_nasa9.jl species=N2,O2,CO2,H2O,Ar,He,H2,CH4

using CompactLES
using CompactLES: CPUBackend, recover_primitives!, state_admissibility
using Printf
const CL = CompactLES

const DEFAULTS = (backend = "amdgpu", n = 48, steps = 10, n_points = 100_000,
                  species = "He,CO2,N2", timing = true)

function device_setup(name)
    if name == "amdgpu"
        @eval using AMDGPU
        @eval AMDGPU.functional() || error("AMDGPU is not functional here")
        @eval println("device: ", AMDGPU.device())
        return @eval (ROCArray, ROCBackend())
    elseif name == "cuda"
        @eval using CUDA
        @eval CUDA.functional() || error("CUDA is not functional here")
        @eval println("device: ", CUDA.device())
        return @eval (CuArray, CUDABackend())
    end
    error("backend must be amdgpu or cuda, got $name")
end

# The body test/device_tests.jl launches, repeated here because the test
# suite and the bench scripts do not include each other.
function nasa9_probe_point!(out, eos, e, Y, Tq, n_species, i, j, k)
    @inbounds begin
        Yat = sp -> Y[sp, i]
        T_rec, status = CL.mixture_temperature_status(eos, e[i], Yat)
        out[1, i] = T_rec
        out[2, i] = status
        out[3, i] = state_admissibility(eos, one(T_rec), e[i], Yat, n_species)
        T = Tq[i]
        point = CL._species_point(eos, T)
        cp = zero(T); h = zero(T); ev = zero(T); hp = zero(T); ep = zero(T)
        Rm = zero(T)
        for sp in 1:n_species
            cp += Y[sp, i] * CL.species_cp(eos, sp, T)
            h += Y[sp, i] * CL.species_enthalpy(eos, sp, T)
            ev += Y[sp, i] * CL.species_energy(eos, sp, T)
            hp += Y[sp, i] * CL.species_enthalpy(eos, sp, point)
            ep += Y[sp, i] * CL.species_energy(eos, sp, point)
            Rm += Y[sp, i] * eos.Rk[sp]
        end
        out[4, i] = cp; out[5, i] = h; out[6, i] = ev; out[7, i] = hp
        out[8, i] = ep
        p = Rm * T
        out[9, i] = CL.eos_phi(eos, one(T), p, T, cp)
        out[10, i] = CL.eos_dphi_dY(eos, n_species, one(T), p, T, cp)
        out[11, i] = CL.artificial_conductivity_scale(eos, one(T), one(T), T, cp)
    end
    return nothing
end

const PROBE_ROWS = ("T_ion", "status", "verdict", "cp", "h", "e", "h(powers)",
                    "e(powers)", "phi", "dphi/dY", "kappa scale")

# A deterministic sequence in [0, 1), as bench/nasa9_inversion.jl uses.
_unit(i, k) = mod(0.6180339887498949 * i + 0.4142135623730951 * k * k, 1.0)

function probe_states(eos, ::Type{T}, n_points) where {T}
    N = nspecies(eos)
    comps = [ntuple(k -> T(k == j), N) for j in 1:N]
    push!(comps, ntuple(_ -> T(1) / N, N))
    push!(comps, ntuple(k -> k == 1 ? T(-0.3) : T(1.3) / (N - 1), N))
    push!(comps, ntuple(_ -> zero(T), N))
    temps = T[]
    for Tj in (1000, 6000), s in (-1e-3, -4eps(T), 0, 4eps(T), 1e-3)
        push!(temps, T(Tj) * (1 + T(s)))
    end
    append!(temps, T.((100, 200, 250, 300, 2500, 20000, 30000, 1e5)))
    energy(Y, Tr) = sum(Y[k] * CL.species_energy(eos, k, Tr) for k in 1:N)
    states = [(Y, Tr, energy(Y, Tr)) for Y in comps for Tr in temps]
    for Y in comps, e in T.((-1e12, 1e30))
        push!(states, (Y, T(500), e))
    end
    for i in 1:n_points
        w = ntuple(k -> 0.05 + _unit(i, k), N)
        Y = T.(w ./ sum(w))
        Tr = T(150 + 29850 * _unit(i, 0)^2)
        push!(states, (Y, Tr, energy(Y, Tr)))
    end
    return states
end

function recovery_agreement(device_array, ka_backend, names, n_points)
    println("\n--- recovery: device mirror against the host mixture ---")
    for T in (Float64, Float32), policy in (:polynomial, :linear, :missing)
        eos = Nasa9Mixture(T, names; extrapolate=policy)
        N = nspecies(eos)
        states = probe_states(eos, T, n_points)
        n = length(states)
        Ym = T[states[i][1][k] for k in 1:N, i in 1:n]
        Tq = T[s[2] for s in states]
        e = T[s[3] for s in states]
        host = zeros(T, 11, n)
        for i in 1:n
            nasa9_probe_point!(host, eos, e, Ym, Tq, N, i, 1, 1)
        end
        dout = device_array(zeros(T, 11, n))
        CL.pointwise_ka!(nasa9_probe_point!, ka_backend, n, 1, 1, dout, eos,
                         device_array(e), device_array(Ym), device_array(Tq), N)
        dev = Array(dout)
        bitwise = count(isequal(host[:, i], dev[:, i]) for i in 1:n)
        status_diff = count(host[2, i] != dev[2, i] for i in 1:n)
        verdict_diff = count(host[3, i] != dev[3, i] for i in 1:n)
        # The relative differences over the points whose recovery converged on
        # both sides; the others carry no accurate value to compare. An
        # enthalpy passes through zero in its gauge, so each difference is
        # taken relative to the larger of the value and a millionth of the
        # quantity's range over the set.
        good = [i for i in 1:n if host[2, i] in (0, 4) && dev[2, i] in (0, 4)]
        function rel(r)
            pts = [i for i in good if isfinite(host[r, i])]
            scale = maximum(i -> abs(host[r, i]), pts; init=zero(T)) / 10^6
            return maximum((abs(dev[r, i] - host[r, i]) /
                            max(abs(host[r, i]), scale, floatmin(T)) for i in pts);
                           init=zero(T))
        end
        @printf("%-7s %-11s %6d points  bitwise %6d  status differs %d  verdict differs %d\n",
                T, policy, n, bitwise, status_diff, verdict_diff)
        # Where the two statuses differ, whether it is only because one side
        # met the convergence criterion and the other did not, and how often
        # the host itself misses the criterion over the in-range sweep.
        missed(x) = (UInt8(x) & CL.TEMPERATURE_NOT_CONVERGED) != 0
        other = count(host[2, i] != dev[2, i] &&
                      xor(UInt8(host[2, i]), UInt8(dev[2, i])) !=
                      CL.TEMPERATURE_NOT_CONVERGED for i in 1:n)
        n_hard = n - n_points
        swept = (n_hard + 1):n
        @printf("        status differences other than convergence: %d   sweep points not converged: host %d, device %d\n",
                other,
                count(i -> missed(host[2, i]), swept),
                count(i -> missed(dev[2, i]), swept))
        @printf("        max rel diff: %s\n",
                join(("$(PROBE_ROWS[r]) $(@sprintf("%.1e", rel(r)))"
                      for r in (1, 4, 5, 6, 7, 9, 10)), ", "))
        @printf("        in eps(%s): T_ion %.1f  cp %.1f  h %.1f\n", T,
                rel(1) / eps(T), rel(4) / eps(T), rel(5) / eps(T))
    end
end

function state_diff(q1, q2)
    a = q1 isa AbstractVector ? q1 : [q1]
    b = q2 isa AbstractVector ? q2 : [q2]
    n_cons = size(parent(a[1]), 4)
    worst = 0.0
    for c in 1:n_cons
        scale = maximum(maximum(abs, view(parent(x), :, :, :, c)) for x in a)
        d = maximum(maximum(abs.(Array(view(parent(b[i]), :, :, :, c)) .-
                                 view(parent(a[i]), :, :, :, c))) for i in eachindex(a))
        worst = max(worst, d / max(scale, floatmin(Float64)))
    end
    return worst
end

function full_runs(ka_backend, names)
    println("\n--- full runs: device against CPU (max over components of max|dev-cpu|/max|Q|) ---")
    per = (PeriodicBC(), PeriodicBC())
    function wall_case(backend, T; n2=12, species=["N2", "CO2"])
        eos = Nasa9Mixture(T, species)
        N = length(species)
        s = Solver(n_global=(32, n2, 1), L_domain=(T(0.1), T(0.04), T(1)),
                   eos=eos, precision=T, backend=backend, cfl=0.4,
                   bcs=((NoSlipWallBC(Twall=800.0), SlipWallBC()), per, per),
                   transport=ConstantTransport(mu0=2e-5))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = (1 + tanh((x - T(0.05)) / T(0.008))) / 2
            Y = ntuple(k -> k == 1 ? 1 - θ : θ / (N - 1), N)
            Prim(Y=Y, p=1e5, T_ion=900 + 700 * sin(π * x / T(0.1)),
                 u=(0.0, 20 * sin(2π * y / T(0.04)), 0.0))
        end)
        return s, Q
    end
    function level_case(backend)
        N = 72
        s = Solver(n_global=(N, 1, 1), L_domain=(0.1, 1.0, 1.0),
                   eos=Nasa9Mixture(["N2", "CO2"]), bcs=(per, per, per),
                   transport=ConstantTransport(mu0=2e-5),
                   refine=BlockRegion((N ÷ 2 - N ÷ 12, 0, 0), (N ÷ 6, 1, 1)),
                   subcycle=true, interface_flux=:ghost, backend=backend)
        states = allocate_state(s)
        initialize!(s, states, (x, y, z) -> begin
            θ = (1 + sin(2π * x / 0.1)) / 2
            Prim(Y=(1 - θ, θ), p=1e5, T_ion=700 + 600θ, u=(30.0, 0, 0))
        end)
        return s, states
    end
    # A shock tube between slip walls: 2000 K driver gas into 300 K CO2, so
    # the shocked gas and the contact cross the 1000 K join while the
    # recovery runs at every point of the discontinuities.
    function shock_case(backend, T)
        eos = Nasa9Mixture(T, ["He", "CO2"])
        wall2 = (SlipWallBC(), SlipWallBC())
        s = Solver(n_global=(200, 1, 1), L_domain=(T(1), T(1), T(1)), eos=eos,
                   precision=T, bcs=(wall2, per, per), cfl=0.4, backend=backend)
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> begin
            θ = (1 + tanh((x - T(0.4)) / T(0.01))) / 2
            Prim(Y=(1 - θ, θ), p=1e6 * (1 - θ) + 1e5 * θ,
                 T_ion=2000 * (1 - θ) + 300 * θ, u=(0.0, 0.0, 0.0))
        end)
        return s, Q
    end
    # A Float32 case carries its Float64 twin, run on the CPU, so that the
    # device's difference prints beside the difference the precision makes.
    cases = [("wall F64, 1000 K join", b -> wall_case(b, Float64), 20, nothing),
             ("wall F32, 1000 K join", b -> wall_case(b, Float32; n2=1), 20,
              b -> wall_case(b, Float64; n2=1)),
             ("refined wave, ghost flux", level_case, 12, nothing),
             ("He/CO2 shock tube F64", b -> shock_case(b, Float64), 200, nothing),
             ("He/CO2 shock tube F32", b -> shock_case(b, Float32), 200,
              b -> shock_case(b, Float64))]
    length(names) > 3 &&
        push!(cases, ("wall F64, $(length(names)) species",
                      b -> wall_case(b, Float64; species=names), 10, nothing))
    for (label, build, nmax, twin) in cases
        s1, q1 = build(CPUBackend())
        run!(s1, q1; tfinal=1.0, nmax=nmax)
        s2, q2 = build(DeviceBackend(ka_backend))
        run!(s2, q2; tfinal=1.0, nmax=nmax)
        d = state_diff(q1, q2)
        @printf("%-30s steps %3d/%3d  t %.4e/%.4e  %.2e%s", label, s1.step,
                s2.step, s1.t, s2.t, d, d == 0 ? "  (bitwise)" : "")
        if twin === nothing
            println()
        else
            s0, q0 = twin(CPUBackend())
            run!(s0, q0; tfinal=1.0, nmax=nmax)
            @printf("   Float64 against Float32 on the CPU: %.2e\n",
                    state_diff(q0, q1))
        end
    end
end

function local_timing(ka_backend, names, n, steps)
    println("\n--- local timing at $(n)^3 (pitfall check, not a performance claim) ---")
    per = (PeriodicBC(), PeriodicBC())
    for T in (Float64, Float32)
        N = length(names)
        for (label, eos) in (("IdealMixture", IdealMixture(T, names)),
                             ("Nasa9Mixture", Nasa9Mixture(T, names)))
            function build(backend)
                s = Solver(n_global=(n, n, n), L_domain=(T(0.01), T(0.01), T(0.01)),
                           bcs=(per, per, per), eos=eos, precision=T,
                           transport=ConstantTransport(mu0=2e-5), backend=backend)
                Q = allocate_state(s)
                initialize!(s, Q, (x, y, z) -> begin
                    θ = (1 + sin(2π * x / T(0.01)) * cos(2π * y / T(0.01))) / 2
                    Y = ntuple(k -> k == 1 ? 1 - θ : θ / (N - 1), N)
                    Prim(Y=Y, p=1e5, T_ion=600 + 1200θ,
                         u=(10 * sin(2π * z / T(0.01)), 0.0, 0.0))
                end)
                return s, Q
            end
            for (where, backend) in (("cpu", CPUBackend()),
                                     ("device", DeviceBackend(ka_backend)))
                s, Q = build(backend)
                recover_primitives!(s, s.eos, Q)   # compile
                CL.KernelAbstractions.synchronize(CL.KernelAbstractions.get_backend(Q))
                t_rec = minimum(1:5) do _
                    @elapsed begin
                        recover_primitives!(s, s.eos, Q)
                        CL.KernelAbstractions.synchronize(
                            CL.KernelAbstractions.get_backend(Q))
                    end
                end
                run!(s, Q; tfinal=1.0, nmax=2)       # compile the step
                w0, n0 = s.wall_total, s.step
                run!(s, Q; tfinal=1.0, nmax=n0 + steps)
                t_step = (s.wall_total - w0) / (s.step - n0)
                @printf("%-8s %-13s %-6s recovery %8.3f ms   step %8.3f ms\n", T,
                        label, where, 1e3 * t_rec, 1e3 * t_step)
            end
        end
    end
end

function main(opt, device_array, ka_backend)
    names = String.(split(opt.species, ","))
    @printf("species %s, Julia threads %d\n", join(names, ", "), Threads.nthreads())
    recovery_agreement(device_array, ka_backend, names, opt.n_points)
    full_runs(ka_backend, names)
    opt.timing && local_timing(ka_backend, names, opt.n, opt.steps)
    return nothing
end

# The device package is loaded by the top-level call, so that `main`, called
# by the next top-level statement, runs in a world that sees its methods.
opt = CompactLES.script_args(ARGS, DEFAULTS)
device_array, ka_backend = device_setup(opt.backend)
main(opt, device_array, ka_backend)
