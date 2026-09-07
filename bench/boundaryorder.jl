# Diagnostic experiments for the wall / AMR accuracy audit.
# Run serially: julia --project=. -t 1 bench/boundaryorder.jl
# Flux-contract probe only: append wall_only=true.
# No production operators are changed. Slopes use actual grid spacing.
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES, Printf
const CL = CompactLES
const PER = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("run this diagnostic on one rank")

function printstudy(label, ns, rows; closed=false)
    println("\n", label)
    for k in eachindex(ns)
        p = k == 1 ? NaN : log(rows[k-1] / rows[k]) /
            log((ns[k] - Int(closed)) / (ns[k-1] - Int(closed)))
        @printf("N=%4d  error=%.9e  order=%.3f\n", ns[k], rows[k], p)
    end
    flush(stdout)
end

function wall_error(N, deriv, f, df)
    s = Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
               bcs=((SlipWallBC(), SlipWallBC()), PER[2], PER[3]),
               deriv=deriv, art=ArtParams(enabled=false), filter_interval=0)
    a = CL.field(s.decomp); da = similar(a)
    for i in 1:N
        a[gidx(s, i, 1, 1)] = f(xcoord(s, 1, i))
    end
    CL.deriv_along!(da, a, s, 1, 1)
    CL._scale_grad!(da, s, 1)
    maximum(abs(da[gidx(s, i, 1, 1)] - df(xcoord(s, 1, i))) for i in 1:N)
end

function wave_error(N; deriv=lele_d1_6(), cfl=0.5, subcycle=false, amr=true,
                    wavenumber=1, phase=0.0)
    region = BlockRegion((N ÷ 2 - N ÷ 12, 0, 0), (N ÷ 6, 1, 1))
    s = Solver(n_global=(N, 1, 1), L_domain=(2pi, 1.0, 1.0), bcs=PER,
               deriv=deriv, cfl=cfl, subcycle=subcycle,
               refine=amr ? region : nothing,
               art=ArtParams(enabled=false), filter_interval=0)
    q = allocate_state(s)
    initialize!(s, q, (x, y, z) -> Prim(u=(0.5, 0, 0), p=1.0,
                                      rho=1.0 + 0.2sin(wavenumber*x + phase)))
    run!(s, q; tfinal=0.5)
    err = 0.0
    # Match the existing gate: maximum over every patch, covered nodes included.
    patches = amr ? CL.eachpatch(s, q) : ((s, q),)
    for (ps, state) in patches, i in 1:ps.decomp.n_local[1]
        err = max(err, abs(state[gidx(ps, i, 1, 1), 1] -
                           (1.0 + 0.2sin(wavenumber*(xcoord(ps, 1, i) - 0.5s.t) + phase))))
    end
    err
end

function thermal_wall_flux()
    s = Solver(n_global=(33, 1, 1), L_domain=(1.0, 1.0, 1.0),
               bcs=((NoSlipWallBC(), NoSlipWallBC()), PER[2], PER[3]),
               transport=Transport(mu0=0.01), art=ArtParams(enabled=false),
               filter_interval=0)
    q = allocate_state(s)
    # Deliberately incompatible temperature gradient: this tests whether
    # the adiabatic condition imposes zero heat flux, not convergence.
    initialize!(s, q, (x, y, z) -> Prim(u=(0, 0, 0), rho=1.0, p=1.0 + 0.1x))
    CL.compute_rhs!(s, q, zero(q))
    [s.flux[1, s.equations.i_energy][gidx(s, i, 1, 1)] for i in (1, 33)]
end

function main()
    opts = CL.script_args(ARGS, (wall_only=false,))
    wall_flux = thermal_wall_flux()
    println("Adiabatic no-slip wall energy flux, incompatible linear T: ", wall_flux)
    all(iszero, wall_flux) || error("adiabatic wall leaks energy")
    opts.wall_only && return nothing
    for (label, deriv, degree) in (("C6 BL", lele_d1_6(closures=:brady_livescu), 6),
                                   ("C8 BL", lele_d1_8(closures=:brady_livescu), 8))
        ns = (17, 33, 65, 129)
        printstudy("$label derivative of x^$degree", ns,
                   [wall_error(n, deriv, x -> x^degree,
                               x -> degree*x^(degree-1)) for n in ns]; closed=true)
    end
    ns = (48, 96, 192)
    for (label, deriv) in (("C6 cascade3", lele_d1_6()),
                           ("C6 cascade4", lele_d1_6(closures=:cascade4)),
                           ("C6 BL", lele_d1_6(closures=:brady_livescu)),
                           ("C10 cascade3", lele_d1_10()))
        for cfl in (0.5, 0.125)
            printstudy("AMR $label CFL=$cfl", ns,
                       [wave_error(n; deriv=deriv, cfl=cfl) for n in ns])
        end
    end
    for subcycle in (false, true)
        printstudy("AMR C6 baseline subcycle=$subcycle", ns,
                   [wave_error(n; subcycle=subcycle) for n in ns])
    end
    for cfl in (0.5, 0.125)
        printstudy("Single periodic C6 CFL=$cfl", ns,
                   [wave_error(n; cfl=cfl, amr=false) for n in ns])
    end
    # A shifted, shorter wave avoids relying on one favorable phase and
    # keeps the high-order errors farther above Float64 roundoff.
    for (label, deriv) in (("C6 cascade3", lele_d1_6()),
                           ("C6 BL", lele_d1_6(closures=:brady_livescu)))
        for subcycle in (false, true), cfl in (0.5, 0.125)
            printstudy("AMR $label k=3 phase=0.37 subcycle=$subcycle CFL=$cfl", ns,
                       [wave_error(n; deriv=deriv, subcycle=subcycle, cfl=cfl,
                                   wavenumber=3, phase=0.37) for n in ns])
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
