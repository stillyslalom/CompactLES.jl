# The MPI suite's phases of patch layouts. Included by test/mpi_tests.jl, which
# loads the package, defines `check`, `section` and the other helpers used here, and
# runs the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# 8. Two-patch layout under rank partitioning. The patch machinery splits the
#    rank set with MPI.Comm_split, each patch running its own decomposition and
#    plans; the interface ghost exchange, shared-plane averaging and the
#    world-hoisted reductions in run! are all np > 1-only paths. Within a patch
#    the spike solve reproduces the single-domain answer bit for bit at any
#    rank count, and every cross-patch transfer moves values unchanged, so the
#    max-norm error of the run — an associativity-free reduction — must equal
#    the serial two-patch value at every np. The reference below was
#    measured with test/patch_tests.jl's entropy-wave case at np = 1.
# ---------------------------------------------------------------------------
function test_two_patch_layout()
    section("two-patch layout: rank-partitioned run matches the serial answer")
    u0 = 0.5
    ic(x, y, z) = Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x))
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=false),
                 filter=StateFilter(interval=0), execution=Execution(patch_grid=(2, 1, 1))))
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    run!(solver, states; tfinal=0.5)
    err = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        for i in 1:ps.decomp.n_local[1]
            I = padded_index(ps, i, 1, 1)
            exact = 1.0 + 0.2 * sin(xcoord(ps, 1, i) - u0 * solver.t)
            err = max(err, abs(Q[I, 1] - exact))
        end
    end
    gerr = gmax(err)
    # Serial (np = 1) value of the identical run. At np = 2 each patch sits on
    # one rank, the arithmetic is identical to the serial patch-sequential
    # order, and the reproduction is bitwise. At np ≥ 4 the patch itself is
    # decomposed and its closed lines take the spike/reduced path where the
    # serial patch ran a plain Thomas sweep, so the answer agrees to round-off
    # accumulation instead, against an 8.2e-12 signal.
    ref = 8.2289730585216603e-12
    tol = np <= 2 ? 1e-20 : 1e-13
    check("two-patch entropy wave: max error matches serial", abs(gerr - ref), tol)
    check("two-patch run: step count matches serial", abs(solver.step - 28), 0.5)
    # The pentadiagonal C10 under the same partitioning, viscous so that the
    # gradients run through its two-row interface closures, each patch's
    # closed-end ranks carrying those rows beside the banded reduced stage. The max of rho over the domain is
    # order-independent; the reference is the serial value of this run.
    ic10(x, y, z) = Prim(u=(u0 + 0.1 * sin(2x), 0, 0), p=1.0 + 0.05 * cos(x),
                         rho=1.0 + 0.2 * sin(x))
    solver10 = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                transport=ConstantTransport(mu0=2e-2), ic=nothing),
        Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=false),
                 deriv=lele_d1_10(), filter=StateFilter(interval=0),
                 execution=Execution(patch_grid=(2, 1, 1))))
    states10 = allocate_state(solver10)
    initialize!(solver10, states10, ic10)
    run!(solver10, states10; tfinal=0.5)
    m = 0.0
    for (ps, Q) in CL.eachpatch(solver10, states10)
        for i in 1:ps.decomp.n_local[1]
            m = max(m, Q[padded_index(ps, i, 1, 1), 1])
        end
    end
    check("two-patch C10 viscous wave: max rho matches serial",
          abs(gmax(m) - 1.3014438000381001), tol)
    check("two-patch C10 run: step count matches serial", abs(solver10.step - 78), 0.5)
    # Slip walls at the outer ends and one-sided gradients, the flux
    # divergence taking the Brady–Livescu rows at the interface ends only
    # (`interface_divergence`): each patch's rank owning a closed end plans
    # it with the wall's or the interface's rows. Serial value of this run.
    wall2 = (SlipWallBC(), SlipWallBC())
    solverbl = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per3[2], per3[3]),
                transport=ConstantTransport(mu0=5e-3), ic=nothing),
        Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=false),
                 filter=StateFilter(interval=0),
                 patch_interfaces=PatchInterfaces(
                     flux=:closure, rhs=:onesided,
                     divergence=lele_d1_6(closures=:brady_livescu)),
                 execution=Execution(patch_grid=(2, 1, 1))))
    statesbl = allocate_state(solverbl)
    initialize!(solverbl, statesbl, (x, y, z) ->
        Prim(u=(0.05 * sin(π * x), 0, 0), p=(1 + 0.05 * cos(π * x))^1.4,
             rho=1 + 0.05 * cos(π * x)))
    run!(solverbl, statesbl; tfinal=0.4, nmax=40)
    m = 0.0
    for (ps, Q) in CL.eachpatch(solverbl, statesbl)
        for i in 1:ps.decomp.n_local[1]
            m = max(m, Q[padded_index(ps, i, 1, 1), 1])
        end
    end
    check("two-patch walls, source divergence rows: max rho matches serial",
          abs(gmax(m) - 1.0433664263818199), tol)
    check("two-patch walls, source divergence rows: step count matches serial",
          abs(solverbl.step - 40), 0.5)
    # The same pair with the whole flux but the wall corrections differenced
    # through the interface from ghost fluxes, the viscous part's ghosts from
    # the neighbouring patch's flux records after both patches evaluate, and
    # the wall corrections on the one-sided rows: two line solves per
    # component along the split dimension, every rank of each patch entering
    # each. Serial value of this run.
    solvergf = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per3[2], per3[3]),
                transport=ConstantTransport(mu0=5e-3), ic=nothing),
        Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=false),
                 filter=StateFilter(interval=0),
                 patch_interfaces=PatchInterfaces(flux=:ghost),
                 execution=Execution(patch_grid=(2, 1, 1))))
    statesgf = allocate_state(solvergf)
    initialize!(solvergf, statesgf, (x, y, z) ->
        Prim(u=(0.05 * sin(π * x), 0, 0), p=(1 + 0.05 * cos(π * x))^1.4,
             rho=1 + 0.05 * cos(π * x)))
    run!(solvergf, statesgf; tfinal=0.4, nmax=40)
    m = 0.0
    for (ps, Q) in CL.eachpatch(solvergf, statesgf)
        for i in 1:ps.decomp.n_local[1]
            m = max(m, Q[padded_index(ps, i, 1, 1), 1])
        end
    end
    check("two-patch walls, ghost-flux divergence: max rho matches serial",
          abs(gmax(m) - 1.0433664263816418), tol)
    check("two-patch walls, ghost-flux divergence: step count matches serial",
          abs(solvergf.step - 40), 0.5)
end

# ---------------------------------------------------------------------------
# The bulk species channel on a two-patch layout. `species_flux = :bulk` puts
# n_cons distributed line solves per direction and a second detector pass per
# species inside each patch's RHS, through the interface plans `grad_Y`
# takes. A slab of density ratio 20 advected at uniform (u, p, T) is invariant
# under every operator of the channel to round-off, the interface rows
# included, so the drift of u, p and T is the check that no rank skipped an
# exchange or a solve; the maximum density is compared against the serial
# value of the same run.
# ---------------------------------------------------------------------------
function test_bulk_patched()
    section("bulk species channel: two-patch layout matches the serial answer")
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 1 / 20, 1.4)])
    Rk, cvk = eos.Rk, eos.cvk
    N = 96
    h = 1.0 / N
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, eos=eos,
                transport=ConstantTransport(mu0=0.0), ic=nothing),
        Numerics(n_global=(N, 1, 1),
                 art=ArtificialProperties(enabled=true, species_flux=:bulk),
                 filter=StateFilter(compact_filter(0.45), interval=1),
                 execution=Execution(patch_grid=(2, 1, 1))))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> begin
        V = (1 - tanh((abs(x - 0.5) - 0.25) / 3h)) / 2
        ρ = V * 20 + (1 - V)
        Prim(Y=(1 - V * 20 / ρ, V * 20 / ρ), rho=ρ, u=(1.0, 0.0, 0.0), p=1.0)
    end)
    run!(solver, states; tfinal=1.0, nmax=40)
    drift = 0.0
    m = 0.0
    Dmax = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        eq = ps.equations
        for i in 1:ps.decomp.n_local[1]
            I = padded_index(ps, i, 1, 1)
            ρ = Q[I, 1] + Q[I, 2]
            y = Q[I, 1] / ρ
            u = Q[I, eq.i_mom[1]] / ρ
            e = Q[I, eq.i_energy] / ρ - 0.5u^2
            T = e / (y * cvk[1] + (1 - y) * cvk[2])
            p = ρ * (y * Rk[1] + (1 - y) * Rk[2]) * T
            drift = max(drift, abs(u - 1), abs(p - 1), abs(T - 1))
            m = max(m, ρ)
            Dmax = max(Dmax, ps.D_art[1][I])
        end
    end
    check("bulk two-patch slab: u, p, T drift at round-off", gmax(drift), 1e-12)
    check("bulk two-patch slab: D_b is nonzero", 1e-30 / max(gmax(Dmax), 1e-300), 1.0)
    check("bulk two-patch slab: step count matches serial", abs(solver.step - 40), 0.5)
    # Serial (np = 1) value of the identical run; at np = 2 each patch sits on
    # one rank and the arithmetic is the serial patch-sequential one, at
    # np ≥ 4 the patch itself is decomposed and agrees to round-off
    # accumulation (as in `test_two_patch_layout`).
    ref = BULK_PATCHED_MAX_RHO
    tol = np <= 2 ? 1e-14 : 1e-12
    check("bulk two-patch slab: max rho matches serial", abs(gmax(m) - ref), tol)
end

# Under the default `:species_d8` detector at `C_D = 1`; 19.999997515570136
# under `detector = :delta4, C_D = 0.1`.
const BULK_PATCHED_MAX_RHO = 19.999997531797185
