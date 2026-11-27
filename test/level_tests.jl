# Two-level refinement tests: manufactured smooth solution spanning the
# coarse-fine boundary, a Sod shock through the refinement boundary in both
# directions with Cook sensors live, conservation drift, and the
# configuration guards; then the subcycled, tagged, and regridded variants.
# Serial; the decomposed forms live in mpi_tests.jl. This file defines the
# helpers the level tests share and includes them by topic: the coupling, the
# subcycled step, regridding, the parent's treatment of covered nodes, and
# levels at the domain's boundaries.
#
# Guards are measured values with headroom, per the convergence-suite
# convention. The interpolation/injection coupling and the reasons the
# invertible filter pair is not the default are documented at the top of
# src/amr/levels_types.jl.

# A dropped decomposition's communicators are freed at the drop, or were never
# created: at np = 1 every decomposition borrows `COMM_WORLD` or `COMM_SELF`
# and owns none, so this suite sees only the second case. The MPI suite checks
# the first on the regridded Sod case.
@isdefined(build_solver) || include(joinpath(@__DIR__, "build_solver.jl"))
using CompactLES: npatches, ConservedState, interior_index, padded_index, xcoord

released_communicators(decomp) =
    !decomp.owns_communicators || decomp.comm == CL.MPI.COMM_NULL

function _level_wave_error(N; mode=:inject, tfinal=0.5, subcycle=false,
                           levels=2, tile=0, deriv=lele_d1_6(), n_halo=4,
                           interface_flux=:ghost, cfl=0.5)
    per3l = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
    u0 = 0.5
    r1 = BlockRegion((N ÷ 2 - N ÷ 12, 0, 0), (N ÷ 6, 1, 1))
    # A third level over the middle half of the level-1 patch, in level-1
    # node space, so the nest scales with N.
    e1 = 3 * (N ÷ 6) - 2
    r2 = BlockRegion((3 * r1.offset[1] + e1 ÷ 4, 0, 0), (e1 ÷ 2, 1, 1))
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3l, ic=nothing),
        Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=false), deriv=deriv,
                 n_halo=n_halo, cfl=cfl, filter=StateFilter(interval=0),
                 patch_interfaces=PatchInterfaces(flux=interface_flux),
                 amr=AMR(initial=levels == 3 ? [r1, r2] : r1, regrid_interval=0,
                         level_restriction=mode, subcycle=subcycle, tile=tile)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x)))
    run!(solver, states; tfinal=tfinal)
    e = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        for i in 1:ps.decomp.n_local[1]
            I = padded_index(ps, i, 1, 1)
            e = max(e, abs(Q[I, 1] - (1.0 + 0.2 * sin(xcoord(ps, 1, i) -
                                                      u0 * solver.t))))
        end
    end
    return e
end

# The cfl of the subcycled entropy-wave rows. At the default cfl = 0.5 the
# subcycled coupling's temporal error exceeds the spatial error of the ghost
# fluxes from N = 96 on: 6.9e-12 at N = 192 against the global step's
# 3.3e-14. At fixed N = 192 it falls by 18 as cfl halves from 0.5 to 0.25,
# but at fixed cfl only by 3.6 as N doubles from 96 to 192, so the slope in
# N falls with N and measures the coupling rather than the interface. At
# cfl = 1/8 the error is the global step's to within 1% at N = 48 and 96
# and to within 15% at N = 192, where both sit near the round-off floor.
# The coupling's temporal order is the subcycled row of the temporal study
# in test/convergence.jl.
const SUBCYCLE_CFL = 0.125

# The composite mass in the quadrature the coarse-fine coupling conserves.
_two_level_mass(solver, states) = volume_integral(solver, states, :rho)

include("level_coupling_tests.jl")
include("level_subcycle_tests.jl")
include("level_regrid_tests.jl")
include("level_mask_tests.jl")
include("level_boundary_tests.jl")
