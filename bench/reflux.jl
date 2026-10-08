# The conservative coarse-fine coupling (src/reflux.jl) against the uncorrected
# one (`CompactLES.REFLUX[] = false`), on three cases:
#
#   sod    the two-level Sod tube of test/level_tests.jl (root N = 201 between
#          slip walls, a box over [0.6, 0.8], CFL 0.4) to t = 0.2, so the shock
#          crosses both faces, global step and subcycled, with the parent's
#          derivative mask (`MASK_CHILD_DERIVATIVE`) off and on. Reported: the
#          relative change of the composite mass and energy in the conserved
#          quadrature (`_conserved_budget`).
#   wave   a 2-D entropy wave, ρ = 1 + 0.2 sin(2π(x − u t) + 0.37) cos(2π(y − v t))
#          at (u, v) = (0.5, 0.25), through a centred box on the periodic unit
#          square at N = 24, 48, 96, to t = 0.3, the artificial properties and
#          the filter off. Reported: the largest density error over the
#          level's nodes and its orders, uncorrected, gated (the default)
#          and with every line corrected (`REFLUX_GATED[] = false`).
#   noh    cylindrical Noh from the cold start (`noh_axis_level` of
#          test/cases.jl, root N = 256, CFL 0.15) with a static tile over the
#          first 43 root nodes, whose face the shock leaves near t = 0.5, to
#          t = 0.6, with the budget ledger: the mass the right-hand sides, the
#          filter passes and the correction move, against the inflow through
#          the outer boundary, and the composite mass against the exact one.
#
#   julia --project=. -t 4 bench/reflux.jl
#   julia --project=. -t 4 bench/reflux.jl cases=noh
#
# Serial only; the three take about five minutes after the package loads. The
# measurements are in reference/CALIBRATION_APPENDIX.md under this script's
# name.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES

const opt = CL.script_args(ARGS, (cases = "sod,wave,noh",))

function sod(; subcycle, mask, reflux)
    CL.MASK_CHILD_DERIVATIVE[] = mask
    CL.REFLUX[] = reflux
    wall2 = (SlipWallBC(), SlipWallBC())
    per = (PeriodicBC(), PeriodicBC())
    ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                            Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    s = Solver(n_global=(201, 1, 1), L_domain=(1.0, 1.0, 1.0),
               bcs=(wall2, per, per), cfl=0.4, subcycle=subcycle,
               refine=BlockRegion((120, 0, 0), (41, 1, 1)))
    q = allocate_state(s)
    initialize!(s, q, ic)
    b0 = CL._conserved_budget(s, q)
    run!(s, q; tfinal=0.2, nmax=40000)
    b1 = CL._conserved_budget(s, q)
    CL.MASK_CHILD_DERIVATIVE[] = false
    CL.REFLUX[] = true
    return ((b1.total_mass - b0.total_mass) / b0.total_mass,
            (b1.total_energy - b0.total_energy) / b0.total_energy)
end

function wave(N; reflux, gated=true)
    CL.REFLUX[] = reflux
    CL.REFLUX_GATED[] = gated
    u0 = (0.5, 0.25)
    f(x, y, t) = 1 + 0.2 * sin(2π * (x - u0[1] * t) + 0.37) * cos(2π * (y - u0[2] * t))
    per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    s = Solver(n_global=(N, N, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3, cfl=0.4,
               filter_interval=0, art=ArtificialProperties(enabled=false),
               refine=BlockRegion((N ÷ 4, N ÷ 4, 0), (N ÷ 2 + 1, N ÷ 2 + 1, 1)))
    q = allocate_state(s)
    initialize!(s, q, (x, y, z) -> Prim(u=(u0[1], u0[2], 0.0), p=1.0, rho=f(x, y, 0.0)))
    run!(s, q; tfinal=0.3)
    e = 0.0
    for (ps, Q) in CL.eachpatch(s, q)
        ps.patch.level == 1 || continue
        nl, pad = ps.decomp.n_local, ps.decomp.n_halo_d
        for j in 1:nl[2], i in 1:nl[1]
            x, y = CL.xcoord(ps, 1, i), CL.xcoord(ps, 2, j)
            e = max(e, abs(Q[i + pad[1], j + pad[2], 1 + pad[3], 1] - f(x, y, s.t)))
        end
    end
    CL.REFLUX[] = true
    CL.REFLUX_GATED[] = true
    return e
end

function main()
    cases = split(opt.cases, ',')
    if "sod" in cases
        println("sod: relative change of the composite mass and energy, t = 0.2")
        for subcycle in (false, true), mask in (false, true), reflux in (false, true)
            m, E = sod(; subcycle, mask, reflux)
            @printf("  %-9s mask %-3s %-12s mass %+.2e  energy %+.2e\n",
                    subcycle ? "subcycled" : "global", mask ? "on" : "off",
                    reflux ? "corrected" : "uncorrected", m, E)
        end
    end
    if "wave" in cases
        println("wave: largest density error on the level, t = 0.3")
        for (name, reflux, gated) in (("uncorrected", false, true), ("gated", true, true),
                                      ("every line", true, false))
            es = [wave(N; reflux, gated) for N in (24, 48, 96)]
            @printf("  %-12s %s  orders %.2f %.2f\n", name,
                    join([@sprintf("%.3e", e) for e in es], " "), log2(es[1] / es[2]),
                    log2(es[2] / es[3]))
        end
    end
    if "noh" in cases
        include(joinpath(@__DIR__, "..", "test", "references.jl"))
        include(joinpath(@__DIR__, "..", "test", "cases.jl"))
        println("noh: a tile over 43 root nodes, t = $(Base.invokelatest(() -> NOH_T))")
        for reflux in (false, true)
            CL.REFLUX[] = reflux
            Base.invokelatest() do
                prob = noh_problem(2; N=256, t0=0.0)
                amr = AMR(initial=BlockRegion((0, 0, 0), (43, 1, 1)), subcycle=false)
                solver, states = setup(prob, Numerics(n_global=(256, 1, 1),
                                       art=ArtificialProperties(enabled=true), cfl=NOH_CFL,
                                       amr=amr, control=StepControl(validity=:permissive)))
                CL._ledger_begin!(solver, states)
                run!(solver, states; tfinal=NOH_T, nmax=10^7)
                r = CL._ledger_end!(solver, states)
                piece(name) = sum((v[1] for (k, v) in r.pieces if k[1] === name); init=0.0)
                inflow = piece(:wall_flux)
                @printf("  %-12s right-hand sides %.6f  filter %+.2e  correction %+.2e\n",
                        reflux ? "corrected" : "uncorrected", piece(:rhs), piece(:filter),
                        piece(:reflux))
                @printf("  %-12s inflow %.6f  composite less inflow, of the final " *
                        "mass %+.2e\n", "", inflow,
                        (r.final[1] - r.initial[1] - inflow) / noh_cylinder_mass(NOH_T))
            end
        end
        CL.REFLUX[] = true
    end
end

main()
