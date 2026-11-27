# The MPI suite's phases of the artificial properties, the species channel, the ring
# detector, state validity and the positivity floor. Included by test/mpi_tests.jl,
# which loads the package, defines `check`, `section` and the other helpers used
# here, and runs the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# 7c. Artificial properties under decomposition. Every other test here disables
#     them, so the sensor path had no multi-rank coverage at all — and
#     `beta_sensor = :dilatation` adds a second distributed filter solve inside
#     `compute_artificial!`, on the far side of no early return. Splitting the
#     same field along each axis in turn must give the same integrated
#     coefficients: a halo the new path forgot to exchange changes the sensor
#     only near a rank boundary. A whole-domain sum over three different
#     decompositions detects that error.
# ---------------------------------------------------------------------------
function test_artificial_decomposition()
    section("artificial properties: coefficients independent of the split axis")
    N = SPLITN
    totals = Dict{Symbol,Vector{Float64}}()
    # The strain sensor reproduces to round-off. The dilatation sensor does not,
    # and the reason is in the switch rather than in any halo: H(−Δ) is
    # discontinuous at Δ = 0, and with the literature ε = 1e-32 the ratio is
    # still ≈ 1 there, so a point whose dilatation is zero to round-off toggles
    # its whole β* between 0 and C_β·ρ·sensor depending on which way the last bit
    # fell. Measured at 2e-6 relative for :gated_strain and 2e-7 for
    # :dilatation, against 1e-14 for the strain sensor — :gated_strain is the
    # worse of the two because the sensor it gates is O(1) at the points that
    # toggle. Each tolerance leaves one to two orders on top of the measured
    # value and stays two or more below the ~1e-2 an unexchanged halo would
    # produce. See reference/CALIBRATION_APPENDIX.md.
    # `:ungated_dilatation` is the same sensor field with no switch on it, so it
    # goes back into the round-off class the strain sensor is in: the field
    # itself always reproduced, and only the discontinuity spoiled it.
    tol_of = Dict(:strain => 1e-10, :gated_strain => 1e-4, :dilatation => 1e-5,
                  :ungated_dilatation => 1e-10)
    for sensor in (:strain, :gated_strain, :dilatation, :ungated_dilatation)
        sums = Float64[]
        for ax in 1:3
            solver = build_solver(
                Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
                Numerics(n_global=(N, N, N),
                         art=ArtificialProperties(enabled=true, beta_sensor=sensor),
                         execution=Execution(dims=splitdims(ax))))
            Q = allocate_state(solver)
            # Compressible and vortical at once, so both the dilatation gate and
            # the vorticity discriminator are away from their trivial values.
            initialize!(solver, Q, (x, y, z) ->
                Prim(u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.3sin(2z)),
                     p=1 + 0.2cos(x), rho=1 + 0.3sin(y)))
            compute_rhs!(solver, Q, zero(Q))
            loc = 0.0
            for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                i in 1:solver.decomp.n_local[1]
                loc += solver.beta_art[padded_index(solver, i, j, k)]
            end
            push!(sums, gsum(loc))
        end
        totals[sensor] = sums
        spread = maximum(sums) - minimum(sums)
        check("$sensor: Σ β* spread over the three split axes",
              spread, tol_of[sensor] * max(maximum(sums), 1e-30))
    end
    # No two sensors may agree here, or the checks above would also pass with an
    # ignored `beta_sensor`. `check` tests val < tol,
    # so each relative difference is inverted into one: this passes when the
    # totals differ by more than 0.1%.
    for sensor in (:gated_strain, :dilatation, :ungated_dilatation)
        rel = abs(totals[:strain][1] - totals[sensor][1]) /
              max(totals[:strain][1], 1e-300)
        check("β* differs between strain and $sensor", 1e-3 / max(rel, 1e-300), 1.0)
    end
    # The μ* channel, whose two settings are the field it reads and how the
    # directions combine. Both are plain reductions with no switch, so both
    # reproduce to round-off; `:max` is the one setting in `ArtificialProperties` that a
    # one-dimensional test cannot distinguish from the default at all, which
    # leaves this its only coverage outside Taylor–Green.
    musums = Dict{Symbol,Vector{Float64}}()
    for (label, art) in ((:strain_sum, ArtificialProperties(enabled=true)),
                         (:velocity_sum,
                          ArtificialProperties(enabled=true, mu_sensor=:velocity)),
                         (:strain_max, ArtificialProperties(enabled=true, reduction=:max)))
        sums = Float64[]
        for ax in 1:3
            solver = build_solver(
                Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
                Numerics(n_global=(N, N, N), art=art,
                         execution=Execution(dims=splitdims(ax))))
            Q = allocate_state(solver)
            initialize!(solver, Q, (x, y, z) ->
                Prim(u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.3sin(2z)),
                     p=1 + 0.2cos(x), rho=1 + 0.3sin(y)))
            compute_rhs!(solver, Q, zero(Q))
            loc = 0.0
            for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                i in 1:solver.decomp.n_local[1]
                loc += solver.mu_art[padded_index(solver, i, j, k)]
            end
            push!(sums, gsum(loc))
        end
        musums[label] = sums
        check("$label: Σ μ* spread over the three split axes",
              maximum(sums) - minimum(sums), 1e-10 * max(maximum(sums), 1e-30))
    end
    for label in (:velocity_sum, :strain_max)
        rel = abs(musums[:strain_sum][1] - musums[label][1]) /
              max(musums[:strain_sum][1], 1e-300)
        check("μ* differs between strain_sum and $label", 1e-3 / max(rel, 1e-300), 1.0)
    end
end

# ---------------------------------------------------------------------------
# 7c'. The shared-diffusivity species channels under decomposition. `:bulk`
#     adds n_cons distributed line solves per direction (the conserved
#     gradients), a mole-fraction pass over the padded extent that relies on
#     the exchanged mass fractions, and a second detector pass per species,
#     all below no early return. One planar two-species step split along each
#     in-plane axis must integrate to the same state; a forgotten exchange or
#     a rank-local sensor moves the sums near a rank boundary. The channel has
#     no discontinuous switch, so it reproduces to round-off like the strain
#     sensor. The channel must also differ from the Fickian one, or the checks
#     would pass with the option ignored.
# ---------------------------------------------------------------------------
function test_bulk_decomposition()
    section("bulk species channel: one planar step independent of the split axis")
    N = SPLITN
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)])
    function planar_step(flux, ax)
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, eos=eos,
                    ic=nothing),
            Numerics(n_global=(N, N, 1),
                     art=ArtificialProperties(enabled=true, species_flux=flux),
                     execution=Execution(dims=splitdims(ax))))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> begin
            θ = 0.5 * (1 + tanh((sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.25) / 0.03))
            Prim(Y=(θ, 1 - θ), rho=(1 - θ) * 5 + θ, p=1.0,
                 u=(0.3 + 0.1sin(2π * y), 0.1cos(2π * x), 0.0))
        end)
        run!(solver, Q; tfinal=1.0, nmax=1)
        sums = zeros(solver.equations.n_cons + 1)
        for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
            i in 1:solver.decomp.n_local[1]
            I = padded_index(solver, i, j, k)
            for c in 1:solver.equations.n_cons
                sums[c] += Q[I, c]
            end
            sums[end] += solver.D_art[1][I]
        end
        return [gsum(v) for v in sums]
    end
    bulk = (planar_step(:bulk, 1), planar_step(:bulk, 2))
    scale = maximum(abs, bulk[1])
    check("bulk: Σ Q and Σ D_b spread over the two split axes",
          maximum(abs, bulk[1] - bulk[2]), 1e-10 * scale)
    check("bulk: D_b is nonzero", 1e-30 / max(bulk[1][end], 1e-300), 1.0)
    fick = planar_step(:fickian, 1)
    rel = abs(fick[end] - bulk[1][end]) / max(abs(fick[end]), 1e-300)
    check("bulk: Σ D differs from the Fickian channel", 1e-3 / max(rel, 1e-300), 1.0)
    # The default partial-density channel shares the diffusivity and the
    # conserved-gradient solves, restricted to the partial densities.
    pd = (planar_step(:partial_density, 1), planar_step(:partial_density, 2))
    check("partial density: Σ Q and Σ D_b spread over the two split axes",
          maximum(abs, pd[1] - pd[2]), 1e-10 * scale)
end

# ---------------------------------------------------------------------------
# 7d. The d8 ringing detector under decomposition. `detector = :d8` puts a
#     pentadiagonal distributed line solve inside `compute_artificial!`, which
#     is the first time the banded reduced-interface stage — and its
#     `Allgather` — runs anywhere except the derivative path. It sits below no
#     early return and must be reached by every rank, so a mistake here hangs
#     rather than fails. Unlike the compression-keyed sensors this one has no
#     discontinuous switch in it, so it must reproduce to round-off.
# ---------------------------------------------------------------------------
function test_ring_detector_decomposition()
    section("d8 detector: sensors independent of the split axis")
    N = SPLITN
    totals = Dict{Symbol,NTuple{2,Vector{Float64}}}()
    # The third case closes dimension 1 with slip walls, where the detector
    # substitutes the node-centred `wall_closures` rows. Those rows belong to
    # the rank owning the wall plane alone, so a split along dimension 1 is the
    # case that would disagree if the substitution reached every block.
    wallx = ((SlipWallBC(), SlipWallBC()), per3[2], per3[3])
    for (case, detector, bcx) in ((:delta4, :delta4, per3), (:d8, :d8, per3),
                                  (:d8_wall, :d8, wallx))
        bsums, ksums = Float64[], Float64[]
        for ax in 1:3
            solver = build_solver(
                Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=bcx, ic=nothing),
                Numerics(n_global=(N, N, N),
                         art=ArtificialProperties(enabled=true, detector=detector),
                         execution=Execution(dims=splitdims(ax))))
            Q = allocate_state(solver)
            initialize!(solver, Q, (x, y, z) ->
                Prim(u=(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0.3sin(2z)),
                     p=1 + 0.2cos(x), rho=1 + 0.3sin(y)))
            compute_rhs!(solver, Q, zero(Q))
            bloc, kloc = 0.0, 0.0
            for k in 1:solver.decomp.n_local[3], j in 1:solver.decomp.n_local[2],
                i in 1:solver.decomp.n_local[1]
                I = padded_index(solver, i, j, k)
                bloc += solver.beta_art[I]
                kloc += solver.kappa_art[I]
            end
            push!(bsums, gsum(bloc)); push!(ksums, gsum(kloc))
        end
        totals[case] = (bsums, ksums)
        for (label, sums) in (("β*", bsums), ("κ*", ksums))
            check("$case: Σ $label spread over the three split axes",
                  maximum(sums) - minimum(sums), 1e-10 * max(maximum(sums), 1e-30))
        end
    end
    # As in 7c, the checks above would also pass with an ignored `detector`, so
    # require the two to disagree. κ* is the
    # channel where they separate by orders rather than by a factor, its input
    # being smooth where β*'s is not.
    rel = abs(totals[:delta4][2][1] - totals[:d8][2][1]) /
          max(totals[:delta4][2][1], 1e-300)
    check("κ* differs between delta4 and d8", 1e-3 / max(rel, 1e-300), 1.0)
end

function test_state_validity()
    section("state validation: one reduced verdict on every rank")
    # A single point is damaged on one rank only. Both halves of the contract
    # then have to hold: the report is the same on every rank, and the strict
    # policy raises on every rank rather than on the one that saw the damage.
    # A rank-local verdict here would be a hang under any collective that
    # followed it, which is the failure mode the whole file is arranged against.
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    reports = Float64[]
    for ax in 1:3
        n_global = ntuple(d -> d == ax ? SPLITN : 16, 3)
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, eos=eos,
                    ic=nothing),
            Numerics(n_global=n_global, art=ArtificialProperties(enabled=false),
                     execution=Execution(dims=splitdims(ax))))
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(rho=1.5, u=(0.4, 0.0, 0.0),
                                                 p=1.0, Y=(0.5, 0.5)))
        clean = state_report(solver, Q)
        check("split axis $ax: a healthy state is valid",
              state_valid(clean) ? 0.0 : 1.0, 0.5)
        check("split axis $ax: every rank sees the whole domain",
              abs(clean.points - prod(n_global)), 0.5)
        # Rank 0 owns global index 1 along the split axis under every layout.
        # The mixture density is left where it was, so the point is rejected on
        # its composition alone and the density check cannot mask it.
        if rank == 0
            I = padded_index(solver, 1, 2, 2)
            ρ = mixture_density(solver, Q, I)
            Q[I, 1] = -0.5
            Q[I, 2] = ρ + 0.5
        end
        poisoned = state_report(solver, Q)
        push!(reports, poisoned.negative_species)
        check("split axis $ax: the count reaches every rank",
              abs(poisoned.negative_species - 1), 0.5)
        raised = 0.0
        try
            validate_state!(solver, Q; warn=false)
        catch err
            err isa SolverFailure && err.reason === :invalid_state || rethrow()
            raised = 1.0
        end
        check("split axis $ax: ranks that raise (expect all $np)",
              abs(gsum(raised) - np), 0.5)
        # Permissive accepts the same state without raising anywhere.
        accepted = validate_state!(solver, Q;
                                   control=StepControl(validity=:permissive),
                                   warn=false)
        check("split axis $ax: permissive returns the same count",
              abs(accepted.negative_species - 1), 0.5)
    end
    check("the count does not depend on the split",
          maximum(reports) - minimum(reports), 0.5)
end

function test_positivity_floor()
    section("positivity failsafe: same tally and same repair under any split")
    # Six cells are damaged by hand at known global indices. Regardless of which
    # rank owns each one, the reduced tally must be the same. Two carry a
    # negative partial density with the mixture density still healthy, two an
    # internal energy below the floor with the total energy still positive, and
    # two a negative total energy. Two species make the first pair exercise
    # the clip-and-rescale path rather than the density floor.
    #
    # The first entry of each triple is the index along whichever axis is split,
    # which is the only one carrying SPLITN points; the other two follow in
    # ascending dimension order. Writing them the other way round would put a
    # 66 on a 16-point dimension and silently damage nothing.
    damaged = ((7, 3, 3), (41, 9, 5), (13, 5, 11), (58, 2, 8), (22, 7, 2),
               (66, 12, 14))
    eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 2.0, 1.6)])
    counts = Float64[]
    masses = Float64[]
    energies = Float64[]
    for ax in 1:3
        n_global = ntuple(d -> d == ax ? SPLITN : 16, 3)
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, eos=eos,
                    ic=nothing),
            Numerics(n_global=n_global, art=ArtificialProperties(enabled=false),
                     execution=Execution(dims=splitdims(ax))))
        others = filter(d -> d != ax, 1:3)
        Q = allocate_state(solver)
        initialize!(solver, Q, (x, y, z) -> Prim(rho=1.5, u=(0.4, 0.0, 0.0), p=1.0,
                                                 Y=(0.5, 0.5)))
        rho_floor, e_floor =
            CL.positivity_floors(solver, Q, StepControl(floor_ratio=1e-6))
        ie = solver.equations.i_energy
        m1 = solver.equations.i_mom[1]
        off = solver.decomp.offset
        nloc = solver.decomp.n_local
        for (n, g) in enumerate(damaged)
            gi = ntuple(d -> d == ax ? g[1] : (d == others[1] ? g[2] : g[3]), 3)
            # Global index to this rank's local one; skip what it does not own.
            loc = ntuple(d -> gi[d] - off[d], 3)
            all(d -> 1 <= loc[d] <= nloc[d], 1:3) || continue
            I = padded_index(solver, loc...)
            if n <= 2
                ρ = mixture_density(solver, Q, I)
                Q[I, 1] = -0.5
                Q[I, 2] = ρ + 0.5                  # mixture density unchanged
            elseif n <= 4
                ke = 0.5 * Q[I, m1]^2 / 1.5
                Q[I, ie] = 0.95 * ke               # e < 0, E > 0
            else
                Q[I, ie] = -2.0                    # unrepresentable
            end
        end
        t = CL.apply_positivity_floor!(solver, Q, rho_floor, e_floor,
                                       :internal_energy)
        push!(counts, t.cells); push!(masses, t.mass); push!(energies, t.energy)
        # The clip moves half a unit of density per cell from species 2 to
        # species 1 and leaves the mixture mass alone.
        vol = prod(2π ./ n_global)
        check("split axis $ax: species mass the clip moves",
              abs(t.species[1] - vol) + abs(t.species[2] + vol), 1e-12 * vol)
        check("split axis $ax: repaired cells (expect 6)", abs(t.cells - 6), 0.5)
        check("split axis $ax: low-energy cells (expect 4)",
              abs(t.low_energy - 4), 0.5)
        # Every damaged cell is now representable on the rank that owns it.
        nbad = 0.0
        for k in 1:nloc[3], j in 1:nloc[2], i in 1:nloc[1]
            I = padded_index(solver, i, j, k)
            ρ = mixture_density(solver, Q, I)
            ρ >= rho_floor || (nbad += 1)
            ke = 0.5 * (Q[I, m1]^2 + Q[I, m1 + 1]^2 + Q[I, m1 + 2]^2) / ρ
            (Q[I, ie] - ke) / ρ >= e_floor * (1 - 1e-9) || (nbad += 1)
        end
        check("split axis $ax: cells left below a floor", gsum(nbad), 0.5)
    end
    # The tally is a reduced quantity, so it must not depend on who owns what.
    check("repaired-cell count independent of the split",
          maximum(counts) - minimum(counts), 0.5)
    check("mass added independent of the split",
          maximum(masses) - minimum(masses), 1e-12 * max(maximum(masses), 1e-30))
    check("energy added independent of the split",
          maximum(energies) - minimum(energies),
          1e-12 * max(maximum(energies), 1e-30))
end
