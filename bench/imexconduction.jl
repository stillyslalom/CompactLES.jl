# The additive Runge–Kutta integrator of src/imex.jl on the molecular
# conduction: a Gaussian conduction pulse against its analytic solution, the
# two step rules, stiff stability, and the acoustic CFL limit of the explicit
# half.
#
# The pulse runs on the periodic line [-1, 1) in a gas of gas constant
# `gas` and γ = 1 + gas, so c_v = 1, the sound speed is √gas and the pressure
# gradient a temperature pulse carries is `gas` times its amplitude: the
# momentum and the compression work it drives stay below the time
# integration's error, and the energy equation reduces to T_t = α T_xx, whose
# solution from T = 1 + A exp(-x²/(2σ²)) is the spreading Gaussian summed over
# its periodic images. κ = α comes from `ConstantTransport` with a viscosity
# of 1e-3 α h², whose explicit rate is negligible.
#
# part=pulse   fixed steps of R times the forward-Euler diffusive limit
#              h²/(2α), for each R in `ratios`, to t = periods · σ²/(2α): the
#              max error over A, the steps, the linear solves and Krylov
#              iterations per step, and the wall time; `explicit=true` adds
#              the default integrator at its own diffusive limit.
# part=rules   the same pulse from t = 0 under the `:error` rule at each of
#              `tolerances` and the `:temperature` rule at each of `targets`,
#              the step otherwise unlimited: the accepted and rejected steps,
#              the error, the Krylov iterations and the wall time.
# part=stiff   a smooth mode plus a grid-Nyquist mode of amplitude 1e-2 on N
#              nodes, `stiff_steps` steps at each R in `stiff`: the Nyquist
#              amplitude after one step and at the end, the max difference
#              from the smooth mode's exact decay exp(-α π² t), the drift of
#              the total energy, and the Krylov iterations per solve.
# part=cfl     the imaginary-axis stability limits of the default five-stage
#              integrator and of the pair's explicit half, from their
#              stability functions, and the largest `cfl` at which each keeps
#              an inviscid acoustic wave on 64 periodic nodes bounded over
#              400 steps, bisected to 1%, with no filter and no artificial
#              properties.
#
# Usage (about a minute at the defaults; part=pulse with explicit=true adds
# about as much again):
#   julia --project=. -t 1 bench/imexconduction.jl [part=all] [n=1024]
#       [ratios=10,100,1000] [tolerances=1e-2,1e-3,1e-4] [targets=0.2,0.05,0.01]
#       [stiff=1e3,1e4,1e5,1e6] [stiff_n=256] [stiff_steps=5] [explicit=false]
#       [alpha=1.0] [sigma=0.2] [amplitude=0.5] [periods=4.0] [gas=1e-10]
#       [rtol=1e-8]

using CompactLES, Printf, LinearAlgebra

const CL = CompactLES
const OPT = CL.script_args(ARGS, (part = "all", n = 1024, ratios = "10,100,1000",
                                  tolerances = "1e-2,1e-3,1e-4",
                                  targets = "0.2,0.05,0.01",
                                  stiff = "1e3,1e4,1e5,1e6", stiff_n = 256,
                                  stiff_steps = 5, explicit = false, alpha = 1.0,
                                  sigma = 0.2, amplitude = 0.5, periods = 4.0,
                                  gas = 1e-10, rtol = 1e-8))

numbers(s) = parse.(Float64, split(s, ","))

const PER = (PeriodicBC(), PeriodicBC())
const LENGTH = 2.0

spacing(n) = LENGTH / n
diffusive_limit(n) = spacing(n)^2 / (2 * OPT.alpha)

# The line solver: c_v = 1 and κ = α exactly, with κ = μ c_p / Pr.
function line_solver(n; implicit=nothing, cfl=1e6)
    gas = OPT.gas
    cp = 1 + gas
    mu = 1e-3 * OPT.alpha * spacing(n)^2
    Solver(n_global=(n, 1, 1), L_domain=(LENGTH, 1.0, 1.0), origin=(-1.0, 0.0, 0.0),
           bcs=(PER, PER, PER), eos=IdealSpecies("gas"; R=gas, gamma=1 + gas),
           transport=ConstantTransport(mu0=mu, Pr=mu * cp / OPT.alpha, Sc=1.0),
           art=ArtificialProperties(enabled=false), filter_interval=0, cfl=cfl,
           implicit=implicit)
end

function start!(solver, T0)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                                             p=OPT.gas * T0(x)))
    return Q
end

t_origin() = OPT.sigma^2 / (2 * OPT.alpha)
tfinal() = OPT.periods * t_origin()

function pulse(x, t)
    t0 = t_origin()
    s = 4 * OPT.alpha * (t + t0)
    acc = 0.0
    for m in -4:4
        acc += exp(-(x + m * LENGTH)^2 / s)
    end
    return 1 + OPT.amplitude * sqrt(t0 / (t + t0)) * acc
end

function pulse_error(solver, Q, t)
    refresh_primitives!(solver, Q)
    e = 0.0
    for i in 1:solver.decomp.n_local[1]
        I = padded_index(solver, i, 1, 1)
        e = max(e, abs(solver.T_ion[I] - pulse(xcoord(solver, 1, i), t)))
    end
    return e / OPT.amplitude
end

function fixed_steps!(solver, Q, tf, nsteps)
    dt = tf / nsteps
    for k in 1:nsteps
        run!(solver, Q; tfinal=k == nsteps ? tf : k * dt)
    end
    return Q
end

counts(solver) = (s = solver.implicit;
                  (s.accepted, s.rejected, s.solves, s.krylov))

function part_pulse()
    n = OPT.n
    tf = tfinal()
    println("\n=== conduction pulse, fixed steps (N = $n, t = $(round(tf, sigdigits=3)), " *
            "explicit limit h²/2α = $(round(diffusive_limit(n), sigdigits=3))) ===")
    @printf("%8s %8s %12s %10s %12s %9s\n", "R", "steps", "max err/A", "solves/st",
            "krylov/solve", "wall s")
    for R in numbers(OPT.ratios)
        nsteps = max(1, round(Int, tf / (R * diffusive_limit(n))))
        solver = line_solver(n; implicit=ImplicitConduction(step_rule=:none,
                                                             rtol=OPT.rtol))
        Q = start!(solver, x -> pulse(x, 0.0))
        wall = @elapsed fixed_steps!(solver, Q, tf, nsteps)
        acc, _, solves, krylov = counts(solver)
        @printf("%8g %8d %12.3e %10.2f %12.2f %9.2f\n", R, acc, pulse_error(solver, Q, tf),
                solves / acc, krylov / max(solves, 1), wall)
    end
    if OPT.explicit
        solver = line_solver(n; cfl=0.5)
        Q = start!(solver, x -> pulse(x, 0.0))
        wall = @elapsed run!(solver, Q; tfinal=tf)
        @printf("%8s %8d %12.3e %10s %12s %9.2f   (default integrator, cfl 0.5)\n",
                "-", solver.step, pulse_error(solver, Q, tf), "-", "-", wall)
    end
end

function part_rules()
    n = OPT.n
    tf = tfinal()
    println("\n=== step rules on the pulse (N = $n, from t = 0) ===")
    @printf("%-12s %9s %9s %9s %12s %10s %9s\n", "rule", "target", "accepted",
            "rejected", "max err/A", "krylov", "wall s")
    rows = [(:error, t) for t in numbers(OPT.tolerances)]
    append!(rows, [(:temperature, t) for t in numbers(OPT.targets)])
    for (rule, target) in rows
        settings = rule === :error ?
            ImplicitConduction(step_rule=:error, tolerance=target, rtol=OPT.rtol) :
            ImplicitConduction(step_rule=:temperature, target_change=target,
                               rtol=OPT.rtol)
        solver = line_solver(n; implicit=settings)
        Q = start!(solver, x -> pulse(x, 0.0))
        wall = @elapsed run!(solver, Q; tfinal=tf)
        acc, rej, _, krylov = counts(solver)
        @printf("%-12s %9g %9d %9d %12.3e %10d %9.2f\n", rule, target, acc, rej,
                pulse_error(solver, Q, tf), krylov, wall)
    end
end

function part_stiff()
    n = OPT.stiff_n
    h = spacing(n)
    println("\n=== stiff stability (N = $n, $(OPT.stiff_steps) steps per R) ===")
    @printf("%8s %12s %14s %14s %14s %12s\n", "R", "dt", "nyquist 1 step",
            "nyquist end", "smooth err", "energy drift")
    smooth(x) = 1 + 0.1 * cospi(x)
    for R in numbers(OPT.stiff)
        dt = R * diffusive_limit(n)
        solver = line_solver(n; implicit=ImplicitConduction(step_rule=:none,
                                                             rtol=OPT.rtol))
        Q = start!(solver, x -> smooth(x) +
                   0.01 * (-1)^round(Int, (x + 1) / h))
        energy0 = total(solver, Q)
        nyq = Float64[]
        for k in 1:OPT.stiff_steps
            run!(solver, Q; tfinal=k * dt)
            push!(nyq, nyquist(solver, Q))
        end
        t = OPT.stiff_steps * dt
        refresh_primitives!(solver, Q)
        err = 0.0
        for i in 1:n
            I = padded_index(solver, i, 1, 1)
            x = xcoord(solver, 1, i)
            exact = 1 + 0.1 * exp(-OPT.alpha * π^2 * t) * cospi(x)
            err = max(err, abs(solver.T_ion[I] - exact))
        end
        _, _, solves, krylov = counts(solver)
        @printf("%8g %12.3e %14.3e %14.3e %14.3e %12.3e   krylov/solve %.2f\n", R, dt,
                nyq[1], nyq[end], err, (total(solver, Q) - energy0) / energy0,
                krylov / max(solves, 1))
    end
end

# The amplitude of the grid-Nyquist mode of the temperature.
function nyquist(solver, Q)
    refresh_primitives!(solver, Q)
    n = solver.decomp.n_local[1]
    s = 0.0
    for i in 1:n
        s += (-1)^(i - 1) * solver.T_ion[padded_index(solver, i, 1, 1)]
    end
    return abs(s / n)
end

function total(solver, Q)
    s = 0.0
    for i in 1:solver.decomp.n_local[1]
        s += Q[padded_index(solver, i, 1, 1), solver.equations.i_energy]
    end
    return s
end

# |R(iy)| of one step of y' = λy, λh = iy, under the default integrator's
# low-storage recurrence and under the explicit half of the pair.
function amplification_rk45(z)
    y = one(z); du = zero(z)
    for s in 1:5
        du = CL.RKA[s] * du + z * y
        y += CL.RKB[s] * du
    end
    return abs(y)
end

function amplification_ark(z)
    A = Float64.(CL.ARK436_EXPLICIT); b = Float64.(CL.ARK436_WEIGHTS)
    k = zeros(ComplexF64, 6)
    for i in 1:6
        k[i] = z * (1 + sum(A[i, j] * k[j] for j in 1:i-1; init=0.0im))
    end
    return abs(1 + sum(b .* k))
end

function axis_limit(amplification)
    y = 0.0
    while amplification(complex(0.0, y + 1e-4)) <= 1 + 1e-12
        y += 1e-4
    end
    return y
end

function acoustic_stable(cfl, implicit)
    n = 64
    solver = Solver(n_global=(n, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=(PER, PER, PER),
                    transport=ConstantTransport(mu0=0.0),
                    art=ArtificialProperties(enabled=false), filter_interval=0,
                    cfl=cfl, implicit=implicit)
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(rho=1 + 0.01 * sinpi(2x),
                                             u=(0.0, 0.0, 0.0), p=1 + 0.014 * sinpi(2x)))
    ok = try
        run!(solver, Q; tfinal=1e9, nmax=400)
        m = maximum(abs, Q[padded_index(solver, i, 1, 1), 2] for i in 1:n)
        isfinite(m) && m < 1.0
    catch err
        err isa SolverFailure || rethrow()
        false
    end
    return ok
end

function cfl_limit(implicit)
    lo, hi = 0.5, 4.0
    while hi - lo > 0.01 * lo
        mid = (lo + hi) / 2
        acoustic_stable(mid, implicit) ? (lo = mid) : (hi = mid)
    end
    return lo
end

function part_cfl()
    println("\n=== acoustic CFL limit of the explicit half ===")
    rk, ark = axis_limit(amplification_rk45), axis_limit(amplification_ark)
    @printf("imaginary-axis limit: default %.3f, pair's explicit half %.3f (ratio %.3f)\n",
            rk, ark, ark / rk)
    c_rk = cfl_limit(nothing)
    c_ark = cfl_limit(ImplicitConduction(step_rule=:none))
    @printf("acoustic wave, 400 steps: largest stable cfl %.3f default, %.3f pair (%.3f)\n",
            c_rk, c_ark, c_ark / c_rk)
end

function main()
    part = OPT.part
    part in ("all", "pulse") && part_pulse()
    part in ("all", "rules") && part_rules()
    part in ("all", "stiff") && part_stiff()
    part in ("all", "cfl") && part_cfl()
end

main()
