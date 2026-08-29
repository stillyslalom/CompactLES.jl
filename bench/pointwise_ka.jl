# Pointwise launcher acceptance measurement (reference/AMR_GPU.md): each
# pointwise
# phase timed through its two launch paths — the default @threaded loop and
# the KernelAbstractions CPU backend — on the same solver and state. The
# acceptance bar for replacing @threaded outright was KA-CPU within 10% per
# phase at 64³; a larger regression keeps the current static routing (Array
# storage on @threaded, device storage on KA), which is the default
# either way. Run with threads (`-t 8` matches the numbers recorded in the
# threading.jl docstring) and read ratios, not absolutes: run-to-run spread
# is the usual 10–20%.
#
#   julia --project=. -t 8 bench/pointwise_ka.jl [n=64] [reps=20]

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES

opt = script_args(ARGS, (n = 64, reps = 20))
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

best(f; reps=opt.reps) = (f(); f(); minimum(@elapsed(f()) for _ in 1:reps))

function compare(name, f)
    CL.FORCE_KA[] = false
    t_thr = best(f)
    CL.FORCE_KA[] = true
    t_ka = best(f)
    CL.FORCE_KA[] = false
    @printf("  %-22s %9.3f ms   %9.3f ms   %6.2fx\n",
            name, 1e3 * t_thr, 1e3 * t_ka, t_ka / t_thr)
    return (t_thr, t_ka)
end

function main(opt)
    n = opt.n
    # tgv-like: 3-D periodic, single species, art off
    s1 = Solver(n_global=(n, n, n), L_domain=(2π, 2π, 2π), bcs=per3,
                transport=Transport(mu0=1e-3), art=ArtParams(enabled=false))
    Q1 = allocate_state(s1)
    initialize!(s1, Q1, (x, y, z) -> Prim(u=(sin(x) * cos(y), -cos(x) * sin(y), 0.0),
                                          p=1 + 0.05cos(2z), rho=1.0))
    CL.exchange_state!(Q1, s1.decomp)
    CL.primitives!(s1, Q1)
    CL.compute_rhs!(s1, Q1, zero(Q1))
    dQ1 = zero(Q1); du1 = zero(Q1)
    n_cons = s1.equations.n_cons
    @printf("\n===== tgv %d^3, 1 species, art off (%d threads) =====\n",
            n, Threads.nthreads())
    println("  phase                    @threaded          KA-CPU     ratio")
    compare("primitives!", () -> CL.primitives!(s1, Q1))
    compare("assemble_fluxes!", () -> CL.assemble_fluxes!(s1, Q1))
    compare("rk update", () -> CL._rk_update!(s1.decomp, n_cons, Q1, dQ1, du1,
                                              CL.RKA[2], CL.RKB[2], 1e-9))
    compare("scale_grad x3", () -> for d in 1:3
        CL._scale_grad!(s1.grad_T_ion[d], s1, d)
    end)
    compare("compute_rhs! (whole)", () -> CL.compute_rhs!(s1, Q1, dQ1))

    # tube-like: 2-D, two species, art on — the sensor and species phases
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)])
    s2 = Solver(n_global=(512, 32, 1), L_domain=(1.0, 0.06, 1.0), eos=eos,
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                art=ArtParams(enabled=true))
    Q2 = allocate_state(s2)
    initialize!(s2, Q2, (x, y, z) -> begin
        θ = tanh_blend(x, 0.5, 0.02)
        Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ)
    end)
    CL.exchange_state!(Q2, s2.decomp)
    CL.primitives!(s2, Q2)
    CL.compute_rhs!(s2, Q2, zero(Q2))
    dQ2 = zero(Q2)
    @printf("\n===== tube 512x32, 2 species, art on =====\n")
    println("  phase                    @threaded          KA-CPU     ratio")
    compare("primitives!", () -> CL.primitives!(s2, Q2))
    compare("compute_artificial!", () -> CL.compute_artificial!(s2, Q2))
    compare("assemble_fluxes!", () -> CL.assemble_fluxes!(s2, Q2))
    compare("compute_rhs! (whole)", () -> CL.compute_rhs!(s2, Q2, dQ2))
    return nothing
end

main(opt)
