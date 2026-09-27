# Explicit nonlinear conduction on the staggered operator of src/staggered.jl
# against the wide form D κ D of the explicit molecular fluxes: the stage-1
# gate of reference/IMPLICIT.md.
#
#   T_t = ∂x(κ0 T^(5/2) ∂x T) + S,   T = 1 + A e^(-t) f(x),
#
# with the source S manufacturing the solution, integrated by classical RK4 at
# a step small enough (dt = cfl · h² / κmax) that the spatial error dominates.
# Two lines of N nodes along x:
#   periodic  f = sin(2πx) on [0, 1); staggered L = G K D_s and the wide
#             form, both from the package plans (lele_d1_6 for the wide one)
#   wall      f = cos(πx) on [0, 1] with nodes on the walls, the adiabatic
#             wall mirror of the staggered plans; the wide form has no
#             flux-form wall treatment and is not run
# It prints the max-norm error at t_final and its order, the drift of the
# conserved total Σ W T (W = h, trapezoidal on a closed line), and, from a
# second run whose initial data carry a grid-Nyquist perturbation of
# amplitude `nyquist`, that mode's amplitude at t_final: the wide form's
# symbol vanishes there, so it carries the mode undamped, while the staggered
# form damps it.
#
# Usage (seconds; single-threaded):
#   julia --project=. -t 1 bench/staggeredconduction.jl [n=32,64,128]
#       [kappa0=0.02] [amplitude=0.3] [tfinal=0.1] [cfl=0.2] [nyquist=1e-3]

using CompactLES, Printf

const CL = CompactLES
const OPT = CL.script_args(ARGS, (n = "32,64,128", kappa0 = 0.02, amplitude = 0.3,
                                  tfinal = 0.1, cfl = 0.2, nyquist = 1e-3))

struct Line
    periodic::Bool
    decomp::CL.Decomp{Float64}
    h::Float64
    x::Vector{Float64}
    staggered::CL.StaggeredDiffusion
    wide::Any
    kappa::Array{Float64,3}
    tmp::Array{Float64,3}
end

function Line(N, periodic)
    decomp = CL.Decomp((N, 1, 1), (periodic, false, false))
    h = periodic ? 1 / N : 1 / (N - 1)
    wide = periodic ? CL.plan_direction(decomp, lele_d1_6(), 1, h) : nothing
    Line(periodic, decomp, h, collect(0:N-1) .* h,
         CL.StaggeredDiffusion(decomp, 1, h; parity=1), wide,
         CL.field(decomp), CL.field(decomp))
end

shape(line, x) = line.periodic ? sin(2π * x) : cos(π * x)
wavenumber(line) = line.periodic ? 2π : π
exact(line, x, t) = 1 + OPT.amplitude * exp(-t) * shape(line, x)

function source(line, x, t)
    k = wavenumber(line); A = OPT.amplitude * exp(-t)
    T = exact(line, x, t)
    Tx = line.periodic ? A * k * cos(k * x) : -A * k * sin(k * x)
    Txx = -A * k^2 * shape(line, x)
    κ = OPT.kappa0 * T^2.5; κx = OPT.kappa0 * 2.5 * T^1.5 * Tx
    return -A * shape(line, x) - (κx * Tx + κ * Txx)
end

# dT/dt into `rate` (interior), `form` = :staggered or :wide.
function rate!(rate, line, T, t, form)
    pad = line.decomp.n_halo
    N = length(line.x)
    for i in 1:N
        line.kappa[i+pad, 1, 1] = OPT.kappa0 * T[i+pad, 1, 1]^2.5
    end
    if form === :staggered
        CL.staggered_diffusion!(rate, line.staggered, T, line.kappa, line.decomp)
    else
        CL.exchange_dim!(T, line.decomp, 1)
        CL.apply_along!(line.tmp, line.wide, T, line.decomp)
        line.tmp .*= line.kappa
        CL.exchange_dim!(line.tmp, line.decomp, 1)
        CL.apply_along!(rate, line.wide, line.tmp, line.decomp)
    end
    for i in 1:N
        rate[i+pad, 1, 1] += source(line, line.x[i], t)
    end
    return rate
end

function integrate(line, form; nyquist=0.0)
    pad = line.decomp.n_halo; N = length(line.x)
    T = CL.field(line.decomp)
    for i in 1:N
        T[i+pad, 1, 1] = exact(line, line.x[i], 0.0) + nyquist * (-1)^i
    end
    κmax = OPT.kappa0 * (1 + OPT.amplitude)^2.5
    steps = ceil(Int, OPT.tfinal / (OPT.cfl * line.h^2 / κmax))
    dt = OPT.tfinal / steps
    k = [CL.field(line.decomp) for _ in 1:4]; stage = CL.field(line.decomp)
    W = fill(line.h, N)
    line.periodic || (W[1] = W[N] = line.h / 2)
    total(T) = sum(W[i] * T[i+pad, 1, 1] for i in 1:N)
    total0 = total(T)
    t = 0.0
    for _ in 1:steps
        rate!(k[1], line, T, t, form)
        for (s, c) in ((2, 0.5), (3, 0.5), (4, 1.0))
            stage .= T .+ (c * dt) .* k[s-1]
            rate!(k[s], line, stage, t + c * dt, form)
        end
        T .+= (dt / 6) .* (k[1] .+ 2 .* k[2] .+ 2 .* k[3] .+ k[4])
        t += dt
    end
    err = maximum(abs(T[i+pad, 1, 1] - exact(line, line.x[i], t)) for i in 1:N)
    mode = abs(sum((T[i+pad, 1, 1] - exact(line, line.x[i], t)) * (-1)^i for i in 1:N)) / N
    return err, abs(total(T) - total0), mode, steps
end

function main()
    Ns = parse.(Int, split(OPT.n, ','))
    @printf("κ0 = %g, A = %g, t = %g, dt = %g h²/κmax, Nyquist perturbation %g\n",
            OPT.kappa0, OPT.amplitude, OPT.tfinal, OPT.cfl, OPT.nyquist)
    @printf("%-9s %-10s %5s %6s  %10s %6s  %10s  %10s\n", "line", "form", "N", "steps",
            "error", "order", "Σ W T drift", "Nyquist")
    for (periodic, forms) in ((true, (:staggered, :wide)), (false, (:staggered,)))
        for form in forms
            previous = NaN
            for (m, N) in enumerate(Ns)
                line = Line(periodic ? N : N + 1, periodic)
                err, drift, _, steps = integrate(line, form)
                _, _, mode, _ = integrate(line, form; nyquist=OPT.nyquist)
                order = m == 1 ? NaN : log(previous / err) / log(Ns[m] / Ns[m-1])
                previous = err
                @printf("%-9s %-10s %5d %6d  %10.3e %6.2f  %10.2e  %10.3e\n",
                        periodic ? "periodic" : "wall", form, N, steps, err, order,
                        drift, mode)
            end
        end
    end
end

main()
