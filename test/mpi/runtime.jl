# The MPI suite's phases of callbacks, the observation clock and the state queries.
# Included by test/mpi_tests.jl, which loads the package, defines `check`, `section`
# and the other helpers used here, and runs the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# 7b. Callback triggers must be globally consistent. A trigger whose effect
#     changes what the ranks do next, a stop ahead of a phase change to an
#     NSCBCOutflowBC face whose correction is collective, deadlocks when the
#     ranks disagree about whether it fired: some enter a collective the others
#     never reach, and the run hangs at zero CPU without failing. WhenState
#     reduces its condition to prevent this.
#
#     The condition below tests the worst case: true on one rank only, the
#     one owning the high-x boundary plane. Every rank must fire on the step
#     that rank first sees it, as a condition every rank evaluates for itself
#     does; the phase change phase ends a run on the same condition and
#     continues it under the collective face.
# ---------------------------------------------------------------------------
function test_callback_consistency()
    section("callback triggers agree across ranks")
    build() = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                            bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                            ic=(x, y, z) -> Prim(u=(0, 0, 0),
                                                 p=1 + 4exp(-200(x - 0.3)^2),
                                                 rho=1 + exp(-200(x - 0.3)^2))),
                    Numerics(n_global=(SPLITN, 16, 16),
                             execution=Execution(dims=splitdims(1))))
    first_fire(fired) = (s, _) -> (fired[] < 0 && (fired[] = s.step); nothing)

    # (a) rank-local condition: only the last rank along x can see it.
    s_local, Q_local = build()
    fired_local = Ref(-1)
    run!(s_local, Q_local; tfinal=1e9, nmax=12,
         callback=Callback(WhenState((s, _) -> rank == np - 1 && s.step >= 5),
                           first_fire(fired_local)))

    # (b) the same condition, evaluated identically on every rank.
    s_glob, Q_glob = build()
    fired_glob = Ref(-1)
    run!(s_glob, Q_glob; tfinal=1e9, nmax=12,
         callback=Callback(WhenState((s, _) -> s.step >= 5), first_fire(fired_glob)))

    check("rank-local WhenState: every rank fires on step 5",
          MPI.Allreduce(abs(fired_local[] - 5), max, comm), 0.5)
    check("rank-local vs global WhenState: the step fired on",
          MPI.Allreduce(abs(fired_local[] - fired_glob[]), max, comm), 0.5)
    tspread = MPI.Allreduce(s_local.t, max, comm) -
              MPI.Allreduce(s_local.t, min, comm)
    check("time spread after a triggered run", tspread,
          1e-14 * max(s_local.t, 1e-30))

    # An AtTime trigger clips dt, which must stay a global decision: every rank
    # has to land on the same instant or they integrate different equations.
    #
    # The two instants sit a little over two CFL steps apart (dt ≈ 1.3e-3 here),
    # which is inside `landing_steps`, so the soft landing subdivides the gap on
    # the approach rather than clipping straight to it. Both branches of the clip
    # are therefore exercised in the eight steps this takes. See the note on run
    # length below.
    s_t, Q_t = build()
    landed = Float64[]
    run!(s_t, Q_t; tfinal=0.008,
         callback=Callback(AtTime([0.003, 0.006]),
                           (s, _) -> (push!(landed, s.t); nothing)))
    hit_spread = 0.0
    for (m, want) in enumerate((0.003, 0.006))
        m <= length(landed) || (hit_spread = Inf; break)
        hit_spread = max(hit_spread, abs(landed[m] - want),
                         MPI.Allreduce(landed[m], max, comm) -
                         MPI.Allreduce(landed[m], min, comm))
    end
    check("AtTime landed exactly, identically on all ranks", hit_spread, 1e-14)

    # EveryTime shortens dt in the same way and additionally re-anchors from
    # solver.t. That anchoring must also be a global decision: a rank computing
    # a different next instant would shorten to a different dt, and the ranks
    # would diverge on the following step. Three ticks, so the re-anchoring is
    # exercised twice rather than once.
    #
    # Run length. Everything this function checks is agreement between ranks on a
    # per-step decision, so it is the *step count* that has to be enough to catch
    # a divergence, and every step costs the same. The four runs here take 12, 12,
    # 8 and 9 steps, against 12, 12, 44 and 40 before the lengths were cut. That
    # The shorter runs reduce runtime at high rank counts on a runner with fewer
    # cores than ranks,
    # each step is a fixed toll of small collectives and the phase cost is linear
    # in steps: at np = 8 on a four-core CI runner these runs cost ~7 s per step
    # against ~0.04 s at np = 2, so the phase was 13 minutes of a 20-minute job.
    # Lengthen a run here only for a check that needs the extra steps.
    #
    # The four runs also take the same number of steps at every rank count, which
    # makes step count the relevant measure: measured 12, 12, 44 and 40
    # at both np = 2 and np = 8 before the cut.
    #
    # tfinal here is one ULP BELOW the third instant, since `0.006 + 0.003` is
    # not the 0.009 literal. That combination used to cost 48 steps rather than 9:
    # the soft landing in `run!` aimed at an instant past the endpoint and halved
    # dt against it forever. Nothing in the checks below moved, so the step count
    # is the only evidence — leave the two literals as they are, and see the note
    # on the landing in `run!`.
    s_e, Q_e = build()
    ticks = Float64[]
    run!(s_e, Q_e; tfinal=0.009,
         callback=Callback(EveryTime(0.003), (s, _) -> (push!(ticks, s.t); nothing)))
    tick_spread = length(ticks) == 4 ? 0.0 : Inf
    for (m, want) in enumerate((0.0, 0.003, 0.006, 0.009))
        m <= length(ticks) || break
        tick_spread = max(tick_spread, abs(ticks[m] - want),
                          MPI.Allreduce(ticks[m], max, comm) -
                          MPI.Allreduce(ticks[m], min, comm))
    end
    check("EveryTime landed exactly, identically on all ranks", tick_spread, 1e-14)
end

# ---------------------------------------------------------------------------
# 7c. Observation and the clock, under a real decomposition.
#
#     Both are collective properties. The artificial coefficients an
#     observational call rebuilds feed the reduced diffusive rate, so a rank
#     that observes while the others do not would propose a different dt for
#     the whole run; and the endpoint conversion decides how many steps every
#     rank takes. Serial coverage is test/runloop_tests.jl; what is added here
#     is the reduction path and the agreement between ranks.
# ---------------------------------------------------------------------------
function test_observation_clock()
    section("observation and endpoints under decomposition")
    # A shock tube split along x. SPLITN / 8 = 9 keeps the C8 filter closure
    # inside every rank's block at the largest rank count run.
    h = 1.0 / (SPLITN - 1)
    tube_ic(x, y, z) = x < 0.5 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                                 Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    function build()
        s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, h), (0.0, h)),
                    bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(SPLITN, 1, 1), cfl=0.15, filter=StateFilter(interval=1),
                     execution=Execution(dims=splitdims(1))))
        Q = allocate_state(s)
        initialize!(s, Q, tube_ic)
        return s, Q
    end
    function history(observe!)
        s, Q = build()
        dts = Float64[]
        cb = Callback(EveryStep(1), function (sv, q)
            push!(dts, sv.dt_prev)
            observe!(sv, q)
            return false
        end)
        run!(s, Q; tfinal=1e9, nmax=6, callback=cb)
        return s, Q, dts
    end
    s0, Q0, dt0 = history((s, q) -> nothing)
    s1, Q1, dt1 = history((s, q) -> (field_array(s, q, :beta_art);
                                     dissipation_rate(s, q); nothing))
    check("dt sequence unchanged by observation", gmax(dt0 == dt1 ? 0.0 : 1.0), 0.5)
    err = 0.0
    for c in 1:s0.equations.n_cons, i in 1:s0.decomp.n_local[1]
        I = padded_index(s0, i, 1, 1)
        err = max(err, abs(Q0[I, c] - Q1[I, c]))
    end
    check("state unchanged by observation", gmax(err), 1e-300)
    check("clock unchanged by observation", gmax(abs(s0.t - s1.t)), 1e-300)

    # A Float32 clock and an endpoint it cannot represent: 0.03 rounds down to
    # 0.0299999993294477, and comparing against the unconverted value leaves a
    # remainder no step can add to the clock. The run then turns over its
    # remaining steps at no advance, identically on every rank, which is a
    # stalled run rather than a hang.
    T = Float32
    sf = build_solver(
        Problem(domain=((0.0, one(T)), (0.0, T(h)), (0.0, T(h))),
                bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                eos=IdealSpecies(T, "gas"; R=one(T), gamma=T(1.4)),
                transport=ConstantTransport{T}(), ic=nothing),
        Numerics(n_global=(SPLITN, 1, 1), art=ArtificialProperties{T}(enabled=false),
                 deriv=lele_d1_6(T), cfl=T(0.4), control=StepControl(validity=:permissive),
                 filter=StateFilter(compact_filter(T(0.45), T), interval=0),
                 execution=Execution(dims=splitdims(1))))
    Qf = allocate_state(sf)
    initialize!(sf, Qf, tube_ic)
    run!(sf, Qf; tfinal=0.03, nmax=60)
    check("Float32 endpoint reached inside the step bound",
          gmax(sf.step >= 60 ? 1.0 : 0.0), 0.5)
    check("Float32 endpoint identical on every rank",
          MPI.Allreduce(Float64(sf.t), max, comm) -
          MPI.Allreduce(Float64(sf.t), min, comm), 1e-300)
    check("Float32 endpoint is the clock's own nearest value",
          gmax(abs(Float64(sf.t) - Float64(T(0.03)))), 1e-300)
end

# ---------------------------------------------------------------------------
# 8. State queries under decomposition. `boundary_plane` is rank-local by
#    design, returning `nothing` on every rank that does not hold the face, and
#    that branch does not arise in serial where one rank holds every edge. A
#    callback condition reading a boundary is written in this form, so the
#    branch is more significant than its size suggests.
# ---------------------------------------------------------------------------
function test_state_queries()
    section("state queries across ranks")
    s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                         bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                         ic=(x, y, z) -> Prim(u=(0.2, 0, 0),
                                              p=1 + 0.5exp(-200(x - 0.3)^2),
                                              rho=1 + exp(-200(x - 0.3)^2))),
                 Numerics(n_global=(SPLITN, 16, 16), execution=Execution(dims=splitdims(1))))

    # Split along x, so exactly one rank holds the low-x face and exactly one
    # holds the high-x face, for any number of ranks.
    owners(side) = gsum(boundary_plane(s, 1, side) === nothing ? 0 : 1)
    check("exactly one rank owns the low-x face",  abs(owners(1) - 1), 0.5)
    check("exactly one rank owns the high-x face", abs(owners(2) - 1), 0.5)
    # Transverse dimensions are undivided here, so every rank owns those faces.
    check("every rank owns an undivided face",
          abs(gsum(boundary_plane(s, 2, 1) === nothing ? 0 : 1) - np), 0.5)

    # The `upstream_density` pattern from the tutorial: a rank-local maximum
    # over the face held by this rank, reduced by the caller. It must agree with
    # the same quantity taken from the refreshed primitives.
    refresh_primitives!(s, Q)
    face_max = let plane = boundary_plane(s, 1, 1)
        plane === nothing ? -Inf : maximum(I -> mixture_density(s, Q, I), plane)
    end
    face_rho = let plane = boundary_plane(s, 1, 1)
        plane === nothing ? -Inf : maximum(I -> s.rho[I], plane)
    end
    check("face maximum from Q matches the refreshed primitives",
          abs(gmax(face_max) - gmax(face_rho)), 1e-14)

    # refresh_primitives! is collective, so every rank must reach it, and
    # afterwards each rank's primitives agree with its own Q.
    err = 0.0
    for k in 1:s.decomp.n_local[3], j in 1:s.decomp.n_local[2], i in 1:s.decomp.n_local[1]
        I = padded_index(s, i, j, k)
        err = max(err, abs(mixture_density(s, Q, I) - s.rho[I]))
        u, v, w = velocity(s, Q, I)
        err = max(err, abs(u - s.u[I]), abs(v - s.v[I]), abs(w - s.w[I]))
    end
    check("primitives agree with Q on every rank after a refresh", gmax(err), 1e-14)
end
