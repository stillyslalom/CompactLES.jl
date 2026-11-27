# The MPI suite's phases of mode truncation, the staggered operators, the implicit
# stage, the IMEX pair and the incompressible solver. Included by test/mpi_tests.jl,
# which loads the package, defines `check`, `section` and the other helpers used
# here, and runs the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# Azimuthal mode truncation with θ split. Each rank packs its θ block of the
# active rings and one Allgatherv over the θ sub-communicator completes each
# ring, so the projection sums in global θ order on every process grid and
# must equal the same rank's serial rebuild on COMM_SELF bit for bit. With
# 24 radial nodes, N_θ = 72 and κ = 1 the active rings are i ≤ 11, so the
# grid splitting r in two leaves the outer radial block without any: its
# ranks make no call while the inner block's ranks gather. The third grid
# splits z, so a ring slab carries several z planes. A short run is then
# compared to round-off, since the reductions in `max_rate` depend on the
# summation order.
# ---------------------------------------------------------------------------
function test_mode_truncation()
    section("azimuthal mode truncation: θ split across ranks")
    mk(comm_here, dims_here, nz) =
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(24, SPLITN, nz), art=ArtificialProperties(enabled=false),
                     polar_truncation=1.0,
                     execution=Execution(comm=comm_here, dims=dims_here)))
    function blockdiff(s, a, ref, b)
        e = 0.0
        for I in CL.interior(s.decomp), c in 1:s.equations.n_cons
            loc = Tuple(I) .- s.decomp.n_halo_d
            e = max(e, abs(a[I, c] - b[padded_index(ref, (loc .+ s.decomp.offset)...), c]))
        end
        return gmax(e)
    end
    # Content in every mode of every ring, from the global node index alone.
    function seed!(sv, Q)
        for I in CL.interior(sv.decomp), c in 1:sv.equations.n_cons
            g = Tuple(I) .- sv.decomp.n_halo_d .+ sv.decomp.offset
            Q[I, c] = (c == 1 ? 1.0 : c == 5 ? 2.5 : 0.0) +
                      0.1 * sin(1.3g[1] + 2.9g[2] + 0.7g[3] + 1.1c)
        end
        return Q
    end
    flow = (r, θ, z) -> Prim(rho=1 + 0.1 * r^2 * cos(2θ) * exp(-4r^2),
                             p=1 + 0.05 * exp(-8r^2) * sin(3θ),
                             u=(0.3 * cos(θ), -0.3 * sin(θ), 0.0))
    for (label, dims_here, nz) in (("θ split", splitdims(2), 1),
                                   ("r and θ split", (2, np ÷ 2, 1), 1),
                                   ("θ and z split", (1, np ÷ 2, 2), 18))
        s = mk(comm, dims_here, nz)
        ref = mk(MPI.COMM_SELF, (1, 1, 1), nz)
        label = "$label $(s.decomp.dims)"
        if dims_here[1] == 2
            check("ranks without an active ring, $label",
                  abs(gsum(isempty(s.truncation.rings)) - np ÷ 2), 0.5)
        end
        Q = seed!(s, allocate_state(s))
        Qref = seed!(ref, allocate_state(ref))
        Q0 = copy(Q)
        CL.truncate_modes!(s, Q)
        CL.truncate_modes!(ref, Qref)
        # The reciprocal of the largest change, so that the check fails when
        # the projection leaves the seeded rings in place.
        check("reciprocal change of the seeded rings, $label",
              1 / gmax(maximum(abs, parent(Q) .- parent(Q0))), 1e2)
        check("truncation matches serial bit for bit, $label",
              blockdiff(s, Q, ref, Qref), 1e-300)
        bytes = @allocated CL.truncate_modes!(s, Q)
        check("truncation allocates nothing, $label", gmax(bytes), 0.5)
        nz == 1 || continue
        states = map((s, ref)) do sv
            Qs = allocate_state(sv)
            initialize!(sv, Qs, flow)
            run!(sv, Qs; tfinal=1e9, nmax=3)
            Qs
        end
        check("three truncated steps match serial, $label",
              blockdiff(s, states[1], ref, states[2]), 1e-12)
        check("simulation time matches serial, $label",
              gmax(abs(s.t - ref.t) / ref.t), 1e-12)
    end
end

# ---------------------------------------------------------------------------
# Staggered operators of the implicit diffusion operator (src/core/staggered.jl),
# split along each dimension in turn, periodic, closed under both wall
# parities and folded at both ends under both plane parities: the midpoint
# flux κ D_s T and L = G K D_s T against the same operators on an undivided
# copy of the grid on each rank. The derivatives' reduced-interface solve,
# the midpoint halo exchange and the mirror on the edge ranks are all on the
# path, and the undivided solve reproduces the distributed one to round-off,
# so an interface bug is an O(1) difference. The metric cases then build the
# operator from a solver, decomposed and on COMM_SELF: a stretched line, and
# the paired folds of the resolved axis and the spherical origin and poles
# with the pairing or the reversed dimension split, where the butterfly runs
# off-rank and carries the plane midpoint; a uniform temperature must give
# zero on every one of them.
# ---------------------------------------------------------------------------
function test_staggered()
    section("staggered operators: decomposed against undivided")
    for ax in 1:3, (periodic, parity, folded) in ((true, 1, false), (false, 1, false),
                                                  (false, -1, false), (false, 1, true),
                                                  (false, -1, true))
        ng = ntuple(d -> d == ax ? SPLITN : 6, 3)
        pers = ntuple(d -> d == ax ? periodic : true, 3)
        h = periodic || folded ? 1 / SPLITN : 1 / (SPLITN - 1)
        σ = folded ? parity : nothing
        results = map((Decomp(ng, pers; dims=splitdims(ax)),
                       Decomp(ng, pers; comm=MPI.COMM_SELF))) do decomp
            op = CL.StaggeredDiffusion(decomp, ax, h; parity=parity, lo_fold=σ,
                                       hi_fold=σ)
            T = CL.field(decomp); κ = CL.field(decomp); out = CL.field(decomp)
            pad = decomp.n_halo_d; off = decomp.offset
            for I in CartesianIndices(decomp.n_local)
                g = Tuple(I) .+ off
                x = (g[ax] - (folded ? 0.5 : 1.0)) * h
                other = 1 + 0.1cos(0.9g[mod1(ax + 1, 3)]) * sin(0.4g[mod1(ax + 2, 3)])
                J = I + CartesianIndex(pad)
                T[J] = (cos(2π * x) + 0.4sin(5x)) * other
                κ[J] = 1 + 0.5cos(π * x) + 0.2x
            end
            CL.staggered_diffusion!(out, op, T, κ, decomp)
            (decomp, op.flux, out)
        end
        (decomp, flux, out), (_, flux_ref, out_ref) = results
        diff = (0.0, 0.0); scale = (0.0, 0.0)
        pad = decomp.n_halo_d; off = decomp.offset
        for I in CartesianIndices(decomp.n_local)
            J = I + CartesianIndex(pad)
            K = CartesianIndex(Tuple(I) .+ off) + CartesianIndex(pad)
            diff = max.(diff, (abs(flux[J] - flux_ref[K]), abs(out[J] - out_ref[K])))
            scale = max.(scale, (abs(flux_ref[K]), abs(out_ref[K])))
        end
        label = "dim $ax " * (periodic ? "periodic" :
                              folded ? "folded, parity $parity" : "closed, parity $parity")
        check("κ D_s T, $label", gmax(diff[1]) / gmax(scale[1]), 1e-12)
        check("G K D_s T, $label", gmax(diff[2]) / gmax(scale[2]), 1e-12)
    end

    noart = ArtificialProperties(enabled=false)
    stretched(c, dims) =
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(SPLITN, 12, 1),
                     stretch=(sine_cluster(0.0, 1.0, 0.3, 0.4), nothing, nothing),
                     art=noart, execution=Execution(comm=c, dims=dims)))
    cyl(c, dims) =
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(40, SPLITN, 1), art=noart,
                     execution=Execution(comm=c, dims=dims)))
    sph(ng) = (c, dims) ->
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, π), (0.0, 2π)), metric=SphericalMetric(),
                    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
                    ic=nothing),
            Numerics(n_global=ng, art=noart, execution=Execution(comm=c, dims=dims)))
    rdims = (2, np ÷ 2, 1)
    cases = (("stretched line, x split", stretched, splitdims(1), 1),
             ("cyl axis, θ split", cyl, splitdims(2), 1),
             ("cyl axis, r and θ split", cyl, rdims, 1),
             ("sph origin, θ split", sph((40, SPLITN, 12)), splitdims(2), 1),
             ("sph poles, θ split", sph((40, SPLITN, 12)), splitdims(2), 2),
             ("sph origin, φ split", sph((40, 16, SPLITN)), splitdims(3), 1),
             ("sph poles, φ split", sph((40, 16, SPLITN)), splitdims(3), 2))
    Tfn(x, y, z) = exp(-4x^2) * (1 + 0.5x * cos(y) + 0.3x * sin(y) * cos(z))
    κfn(x, y, z) = 1 + 0.5x^2 + 0.2x * sin(y) * cos(z)
    for (label, mk, dims, d) in cases
        s = mk(comm, dims)
        ref = mk(MPI.COMM_SELF, (1, 1, 1))
        out = map((s, ref)) do sv
            op = CL.StaggeredDiffusion(sv, d)
            T = CL.field(sv.decomp); κ = CL.field(sv.decomp)
            L = CL.field(sv.decomp); L1 = CL.field(sv.decomp)
            fillf!(sv, T, Tfn); fillf!(sv, κ, κfn)
            CL.staggered_diffusion!(L, op, T, κ, sv.decomp)
            fillf!(sv, T, (x, y, z) -> 1.0)
            CL.staggered_diffusion!(L1, op, T, κ, sv.decomp)
            (L, L1)
        end
        (L, L1), (L_ref, _) = out
        e = 0.0; scale = 0.0; e1 = 0.0
        for I in CL.interior(s.decomp)
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
            e = max(e, abs(L[I] - L_ref[J]))
            scale = max(scale, abs(L_ref[J]))
            e1 = max(e1, abs(L1[I]))
        end
        check("J⁻¹ G C K D_s T, $label, dim $d", gmax(e) / gmax(scale), 1e-11)
        check("uniform T gives zero, $label, dim $d", gmax(e1) / gmax(scale), 1e-12)
    end
    test_implicit_stage()
end

# The implicit diffusion stage (src/solver/implicit.jl) on a decomposed solver
# against the same stage on COMM_SELF: the solution of (I − γΔt L) T = r at
# a stiffness far beyond the explicit limit, its residual evaluated by the
# decomposed operator, and a uniform right-hand side, which must return at
# once and unchanged. The preconditioner's aggregates and line systems
# follow the blocks, so the two solves take different paths to the same
# solution; the conjugate-gradient cases split the walls' dimension (both
# parities) and the periodic one, and the GMRES cases the resolved axis
# along the pairing dimension (the butterfly off-rank) and along both.
function test_implicit_stage()
    section("implicit stage: decomposed against undivided")
    noart = ArtificialProperties(enabled=false)
    walls(c, dims) =
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(SPLITN, 16, 1), art=noart,
                     execution=Execution(comm=c, dims=dims)))
    periodic(c, dims) =
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(16, SPLITN, 1), art=noart,
                     execution=Execution(comm=c, dims=dims)))
    cyl(c, dims) =
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 2π), (0.0, 1.0)), metric=CylindricalMetric(),
                    bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(24, SPLITN, 1), art=noart,
                     execution=Execution(comm=c, dims=dims)))
    cases = (("walls, x split", walls, splitdims(1), 1),
             ("isothermal walls, x split", walls, splitdims(1), -1),
             ("periodic, y split", periodic, splitdims(2), 1),
             ("resolved axis, θ split", cyl, splitdims(2), 1),
             ("resolved axis, r and θ split", cyl, (2, np ÷ 2, 1), 1))
    rfn(x, y, z) = exp(-3(x - 0.3)^2) * (1 + 0.3cos(2y)) * sinpi(x)
    κfn(x, y, z) = 1 + 0.5x^2 + 0.2x * sin(y)
    γΔt = 0.1
    for (label, mk, dims, parity) in cases
        s = mk(comm, dims)
        ref = mk(MPI.COMM_SELF, (1, 1, 1))
        solved = map((s, ref)) do sv
            stage = CL.DiffusionStage(sv; parity=parity)
            r = CL.field(sv.decomp); κ = CL.field(sv.decomp)
            fillf!(sv, r, rfn); fillf!(sv, κ, κfn)
            T = copy(r)
            result = CL.solve_stage!(T, stage, r, κ, γΔt)
            (stage, r, κ, T, result)
        end
        (stage, r, κ, T, result), (_, _, _, T_ref, _) = solved
        # The residual through the decomposed operator, in the L2 norm the
        # solve stops in.
        out = CL.field(s.decomp)
        CL.stage_operator!(out, stage, T, κ, γΔt)
        num = 0.0; den = 0.0; e = 0.0; scale = 0.0
        for I in CL.interior(s.decomp)
            V = stage.volume[I]
            num += V * (out[I] - r[I])^2; den += V * r[I]^2
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
            e = max(e, abs(T[I] - T_ref[J])); scale = max(scale, abs(T_ref[J]))
        end
        method = stage.symmetric ? "CG" : "GMRES"
        check("$method converged, $label ($(result.iterations) it)",
              gmax(result.converged ? 0.0 : 1.0) +
              gmax(result.iterations) - MPI.Allreduce(result.iterations, min, comm), 0.5)
        check("distributed residual, $label", sqrt(gsum(num) / gsum(den)), 1e-9)
        check("against undivided, $label", gmax(e) / gmax(scale), 1e-8)
        parity < 0 && continue       # the odd wall takes T − T_wall, not a constant
        fill!(r, 1.5); T = copy(r)
        uniform = CL.solve_stage!(T, stage, r, κ, γΔt)
        e1 = maximum(abs(T[I] - r[I]) for I in CL.interior(s.decomp))
        check("uniform r returns unchanged, $label",
              gmax(e1) + gmax(uniform.iterations), 1e-12)
    end
end

# ---------------------------------------------------------------------------
# The additive pair with the conduction implicit (src/solver/imex_step.jl) on a
# decomposed patch, against the same run on COMM_SELF.
#
#   1. A temperature pulse in a periodic box with an acoustic flow, at about
#      twenty times the explicit diffusive limit, under the default `:error`
#      step rule: the stage residual and the step rule's measure are reduced
#      across ranks, and the decomposed run must reproduce the undivided one
#      to the stage tolerance. The preconditioner's aggregates follow the
#      blocks and the reductions are summed in a different order, so the
#      linear solves differ between the two runs by up to their tolerance and
#      bitwise agreement is not expected. At `rtol = 1e-12` the states agree
#      to about 1e-14 of each component's magnitude after twenty steps. The
#      tolerance is 1e-10, a hundred times the stage tolerance, against the
#      O(1) difference of an interface defect.
#   2. A one-node temperature spike on a cold line, placed inside rank 0's
#      block, at a step far beyond the explicit limit. The second stage is
#      the trapezoidal rule at z = γhλ, which reflects the stiff modes of the
#      spike, so its temperature is negative at the spike node on rank 0 only.
#      Every rank must take the same failure from the reduced validity flag,
#      and with retries every rank must take the same rollbacks and reach the
#      endpoint at the state of the undivided run. A rank-local decision
#      leaves the other ranks waiting in a reduction, which hangs the suite.
# ---------------------------------------------------------------------------
function test_imex_pair()
    section("implicit conduction: decomposed pair against undivided")
    noart = ArtificialProperties(enabled=false)
    function normalized_diff(s, Q, ref, Qref, components)
        worst = 0.0
        for c in components
            e = 0.0; scale = 0.0
            for I in CL.interior(s.decomp)
                loc = Tuple(I) .- s.decomp.n_halo_d
                J = padded_index(ref, (loc .+ s.decomp.offset)...)
                e = max(e, abs(Q[I, c] - Qref[J, c])); scale = max(scale, abs(Qref[J, c]))
            end
            # A component that stays zero (the spanwise momentum) enters unscaled.
            magnitude = gmax(scale)
            worst = max(worst, gmax(e) / (magnitude > 0 ? magnitude : 1.0))
        end
        worst
    end
    counters(s) = (s.step, s.implicit.accepted, s.implicit.rejected)
    # The largest difference over ranks of an integer or a time.
    spread(x) = gmax(x) + gmax(-x)

    # 1. The pulse: a gas constant of one, γ = 1.4, κ = 2.5 and ρ c_v ≤ 2.5, so
    #    α ≥ 1, and a viscous rate far below the acoustic one.
    box(c, dims) =
        build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3,
                    eos=IdealSpecies("gas"; R=1.0, gamma=1.4),
                    transport=ConstantTransport(mu0=1e-3, Pr=1e-3 * 3.5 / 2.5, Sc=1.0),
                    ic=nothing),
            Numerics(n_global=(24, 36, 1), art=noart, cfl=0.5,
                     implicit=ImplicitConduction(rtol=1e-12),
                     execution=Execution(comm=c, dims=dims)))
    pulse(x, y, z) = Prim(rho=1 / (1 + 0.3exp(-((x - 0.4)^2 + (y - 0.55)^2) / 0.02)),
                          u=(0.2sinpi(2y), 0.2cospi(2x), 0.0), p=1.0)
    nsteps = 20
    runs = map(((comm, (2, np ÷ 2, 1)), (MPI.COMM_SELF, (1, 1, 1)))) do (c, dims)
        sv = box(c, dims)
        Q = allocate_state(sv)
        initialize!(sv, Q, pulse)
        run!(sv, Q; tfinal=1e9, nmax=nsteps)
        (sv, Q)
    end
    (s, Q), (ref, Qref) = runs
    limit = 1 / (2 * (24^2 + 36^2))           # forward-Euler diffusive limit at α = 1
    check("pulse: steps far beyond the diffusive limit (dt / limit = " *
          "$(round(Int, ref.t / nsteps / limit)))", gmax(limit * nsteps / s.t), 0.2)
    check("pulse: steps, accepted and rejected attempts agree",
          sum(spread.(counters(s) .- counters(ref))), 0.5)
    check("pulse: time matches undivided", gmax(abs(s.t - ref.t) / ref.t), 1e-12)
    check("pulse: state matches undivided",
          normalized_diff(s, Q, ref, Qref, 1:s.equations.n_cons), 1e-10)

    # 2. The spike, on the serial suite's conduction line: c_v = 1, κ = α = 1,
    #    and a gas constant small enough that the flow stays negligible.
    n = SPLITN
    h = 2 / n
    gas = 1e-10
    mu = 1e-3 * h^2
    line(c, dims, control) =
        build_solver(
            Problem(domain=((-1.0, -1.0 + 2.0), (0.0, 0.0 + 1.0), (0.0, 0.0 + 1.0)),
                    bcs=per3, eos=IdealSpecies("gas"; R=gas, gamma=1 + gas),
                    transport=ConstantTransport(mu0=mu, Pr=mu * (1 + gas), Sc=1.0),
                    ic=nothing),
            Numerics(n_global=(n, 1, 1), art=noart, cfl=1e6, control=control,
                     implicit=ImplicitConduction(step_rule=:none, rtol=1e-12),
                     filter=StateFilter(interval=0), execution=Execution(comm=c, dims=dims)))
    spike_node = n ÷ (2np) + 1                  # the middle of rank 0's block
    x_spike = -1 + (spike_node - 1) * h
    spike(x, y, z) = Prim(rho=1.0, u=(0.0, 0.0, 0.0),
                          p=gas * (abs(x - x_spike) < h / 2 ? 1.2 : 0.2))
    tf = 32 * h^2 / 2                           # 32 forward-Euler diffusive limits
    s = line(comm, splitdims(1), StepControl())
    Q = allocate_state(s)
    initialize!(s, Q, spike)
    failure = try
        run!(s, Q; tfinal=tf)
        nothing
    catch e
        e isa SolverFailure || rethrow()
        e
    end
    failed = failure isa SolverFailure && failure.reason === :implicit_solve &&
             occursin("trial state", failure.detail)
    check("spike: every rank raises the trial-state failure",
          gsum(failed ? 0 : 1), 0.5)
    # The failed attempt leaves its trial state in Q: the ranks holding a
    # nonpositive internal energy are the ranks that found the state invalid.
    ie = s.equations.i_energy
    invalid = any(Q[I, ie] <= 0 for I in CL.interior(s.decomp))
    check("spike: the invalid trial state lies on one rank only",
          abs(gsum(invalid) - 1), 0.5)

    runs = map(((comm, splitdims(1)), (MPI.COMM_SELF, (1, 1, 1)))) do (c, dims)
        sv = line(c, dims, StepControl(retries=6))
        Qv = allocate_state(sv)
        initialize!(sv, Qv, spike)
        run!(sv, Qv; tfinal=tf)
        (sv, Qv)
    end
    (s, Q), (ref, Qref) = runs
    retried = round(Int, log2(1e6 / ref.cfl))
    check("spike: rollbacks agree with undivided ($retried retries)",
          spread(s.cfl) + gmax(abs(s.cfl - ref.cfl)) + (retried > 0 ? 0 : 1), 0.5)
    check("spike: every rank reaches the endpoint", gmax(abs(s.t - tf) / tf), 1e-14)
    check("spike: steps, accepted and rejected attempts agree",
          sum(spread.(counters(s) .- counters(ref))), 0.5)
    check("spike: state matches undivided", normalized_diff(s, Q, ref, Qref, (ie,)), 1e-10)
end

# ---------------------------------------------------------------------------
# The incompressible solver (src/solver/incompressible.jl) decomposed against the
# same run on COMM_SELF: a periodic plane split along x, a slip-walled box
# split along y, and a no-slip channel and a three-dimensional box with slip
# walls in z on the default process grid. The pressure solves stop at their
# tolerance after different iteration counts on the two grids, since the
# multigrid aggregates follow the blocks, so agreement is to that tolerance
# and not bitwise.
function test_incompressible()
    section("incompressible solver: decomposed against undivided")
    u2(x, y, z) = (sin(x) * cos(y) + 0.1sin(2y), -cos(x) * sin(y) + 0.1cos(3x), 0.0)
    u3(x, y, z) = (sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0)
    cases = (("periodic plane, x split", (SPLITN, 16, 1), (true, true, true),
              (2π, 2π, 1.0), :slip, splitdims(1), u2),
             ("slip box, y split", (17, SPLITN + 1, 1), (false, false, true),
              (π, π, 1.0), :slip, splitdims(2), u2),
             ("no-slip channel", (SPLITN, 17, 1), (true, false, true),
              (2π, π, 1.0), :noslip, nothing, u2),
             ("3-D, slip walls in z", (16, 16, 17), (true, true, false),
              (2π, 2π, π), :slip, nothing, u3))
    for (label, n, periodic, lengths, walls, dims, u) in cases
        s, ref = map(((comm, dims), (MPI.COMM_SELF, nothing))) do (c, g)
            sv = CL.Incompressible.IncompressibleSolver(n; lengths=lengths,
                                                        periodic=periodic, nu=0.01,
                                                        walls=walls, dims=g, comm=c)
            CL.Incompressible.set_velocity!(sv, u)
            for _ in 1:4
                CL.step!(sv, 0.02)
            end
            sv
        end
        shift = CartesianIndex(s.decomp.offset) - CartesianIndex(s.decomp.n_halo_d) +
                CartesianIndex(ref.decomp.n_halo_d)
        e = 0.0; scale = 0.0
        for i in s.dims, I in CL.interior(s.decomp)
            e = max(e, abs(s.U[i][I] - ref.U[i][I+shift]))
            scale = max(scale, abs(ref.U[i][I+shift]))
        end
        check("against undivided, $label", gmax(e) / gmax(scale), 1e-10)
        check("divergence, $label", CL.Incompressible.max_divergence(s), 1e-10)
    end
end
