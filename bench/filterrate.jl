# Does the state filter remove energy per APPLICATION or per unit TIME?
#
#   julia --project=. -t auto bench/filterrate.jl
#   julia --project=. -t auto bench/filterrate.jl 64 filter_cfl=0.4
#
# --- What this measured, so it is not rediscovered ---------------------------
#
# At N = 32 to t = 0.5, kinetic energy removed by the filter alone:
#
#   cfl    steps   unrelaxed          relaxed (filter_cfl = 0.4)
#   0.4      73    4.092e-3  1.000    4.042e-3  1.000
#   0.2     145    8.107e-3  1.981    4.042e-3  1.000
#   0.1     289    1.608e-2  3.930    4.042e-3  1.000
#
# Unrelaxed, the loss tracks the STEP COUNT (73 : 145 : 289 = 1 : 1.99 : 3.96),
# not the elapsed time. That is the dt-inconsistency recorded as model debt 1 in
# reference/ROADMAP.md, measured directly rather than inferred: a calculation at
# half the CFL applies twice the subgrid dissipation over the same interval.
# `filter_cfl` makes it a rate, constant to six figures across a 4x CFL change.
#
# --- Choosing the case -------------------------------------------------------
#
# Two earlier attempts failed for useful reasons.
#
#   A broadband field loses 64% of its kinetic energy within tens of steps and
#   then cannot lose more. The scaling saturates out of sight, and the measured
#   spread across a 4x CFL change collapses to 1.2%.
#
#   A velocity sine in x at uniform pressure is an acoustic oscillation: it
#   trades kinetic for internal energy at the sound speed, hundreds of times
#   faster than the filter acts. TOTAL energy shows nothing either, because a
#   symmetric filter on a periodic grid conserves the discrete sum of every
#   conserved variable exactly. The filter moves energy between the two
#   reservoirs; it does not remove it.
#
# What works is a parallel shear layer. u_x = f(y) at uniform rho and p is an
# exact steady solution of the Euler equations and stays one discretely, since
# every x-derivative of the field vanishes. Kinetic energy is then constant in
# time and the filter is the only thing that can change it. Eight points per
# wavelength keeps the per-application loss small, so the total stays
# proportional to the number of applications rather than saturating.
#
# Scratch tooling, like everything else in bench/: it prints a table and asserts
# nothing.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf

const CL = CompactLES
const per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

const DEFAULTS = (N = 32, tfinal = 0.5, k = 4, amplitude = 0.1,
                  cfls = "0.4,0.2,0.1", filter_cfl = 0.4)
const opt = script_args(ARGS, DEFAULTS; positional = (:N, :tfinal))
const CFLS = [parse(Float64, strip(s)) for s in split(opt.cfls, ',')]

"Kinetic energy over the interior, reduced across the communicator."
function kinetic(solver, Q)
    CL.exchange_state!(Q, solver.decomp)
    CL.primitives!(solver, Q)
    nx, ny, nz = solver.decomp.n_local
    ke = 0.0
    for k in 1:nz, j in 1:ny, i in 1:nx
        I = gidx(solver, i, j, k)
        ke += solver.rho[I] * (solver.u[I]^2 + solver.v[I]^2 + solver.w[I]^2)
    end
    return MPI.Allreduce(0.5 * ke * prod(solver.h), +, solver.decomp.comm)
end

function run_one(cfl, filter_cfl)
    N = opt.N
    prob = Problem(eos = single_species(gamma = 1.4, R = 1.0),
                   transport = Transport(mu0 = 0.0),
                   domain = ((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs = per3,
                   ic = (x, y, z) -> Prim(rho = 1.0, p = 10.0,
                                          u = (opt.amplitude * sin(opt.k * y),
                                               0.0, 0.0)))
    solver, Q = setup(prob, Numerics(n_global = (N, N, N), cfl = cfl,
                                     art = ArtParams(enabled = false),
                                     filter_interval = 1,
                                     filter_cfl = filter_cfl))
    ke0 = kinetic(solver, Q)
    run!(solver, Q; tfinal = opt.tfinal)
    return (loss = 1 - kinetic(solver, Q) / ke0, steps = solver.step)
end

function main()
    @printf("N = %d, t_final = %.3g, u_x = %.3g sin(%d y), mu0 = 0, art off\n",
            opt.N, opt.tfinal, opt.amplitude, opt.k)
    println("Steady shear layer: the filter is the only sink of kinetic energy.\n")
    for fc in (0.0, opt.filter_cfl)
        println(fc == 0 ? "--- unrelaxed (filter_cfl = 0, default) ---" :
                          "--- relaxed (filter_cfl = $fc) ---")
        @printf("  %6s %7s %13s %12s\n", "cfl", "steps", "KE loss", "vs first")
        ref = NaN
        for cfl in CFLS
            r = run_one(cfl, fc)
            isnan(ref) && (ref = r.loss)
            @printf("  %6.3g %7d %13.5e %12.3f\n", cfl, r.steps, r.loss,
                    r.loss / ref)
        end
        println()
    end
    println("filterrate complete")
end

mpi_main(main)
