# The MPI suite's phases of hierarchy checkpoints, phase changes, regridding,
# lattice subdivision, placed levels and seams. Included by test/mpi_tests.jl, which
# loads the package, defines `check`, `section` and the other helpers used here, and
# runs the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# 7b. The hierarchy checkpoint, decomposed: a tiled regridding run written
#     between checks and restored on the same rank count rebuilds the recorded
#     tile set on the stored owner ranges, so every rank reads back its own
#     blocks and the continuation is bitwise, regrids and all. The case is the
#     tiled Sod of the ownership phase (three tiles at 168..184 by step 41).
# ---------------------------------------------------------------------------
function test_hierarchy_checkpoint()
    section("hierarchy checkpoint: same rank count, bitwise continuation")
    wall2 = (SlipWallBC(), SlipWallBC())
    ic = (x, y, z) -> x < 0.45 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                                 Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    mk() = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per3[2], per3[3]),
                ic=nothing),
        Numerics(n_global=(400, 1, 1), cfl=0.2,
                 amr=AMR(initial=BlockRegion((176, 0, 0), (16, 1, 1)), subcycle=true,
                         regrid_interval=5, tag_buffer=2, tile=8, subdivide_lattice=false)))
    solver = mk()
    states = allocate_state(solver)
    initialize!(solver, states, ic)
    run!(solver, states; tfinal=0.03, nmax=21)
    regs = level_regions(solver, 1)
    owners = copy(solver.levels[2].owners)
    save_checkpoint(solver, states, "mpi_hier")
    MPI.Barrier(comm)
    run!(solver, states; tfinal=0.03, nmax=41)

    restarted = mk()
    rstates = allocate_state(restarted)
    load_checkpoint!(restarted, rstates, "mpi_hier")
    check("restart: recorded tile set rebuilt",
          level_regions(restarted, 1) == regs ? 0.0 : 1.0, 0.5)
    check("restart: stored ownership restored",
          restarted.levels[2].owners == owners ? 0.0 : 1.0, 0.5)
    check("restart: state vector aligned with the patches",
          abs(length(rstates.patches) - length(restarted.patches)), 0.5)
    run!(restarted, rstates; tfinal=0.03, nmax=41)
    check("restart: time reached matches the uninterrupted run",
          gmax(abs(restarted.t - solver.t)), 1e-300)
    check("restart: tile set tracks the uninterrupted run",
          level_regions(restarted, 1) == level_regions(solver, 1) ? 0.0 : 1.0, 0.5)
    spec, rspec = getfield(solver, :regrid), getfield(restarted, :regrid)
    check("restart: tag history tracks the uninterrupted run",
          rspec.created == spec.created && rspec.checks == spec.checks ? 0.0 : 1.0,
          0.5)
    d = 0.0
    for i in eachindex(states.patches)
        inner = CL.interior(solver.patches[i].decomp)
        d = max(d, maximum(abs.(parent(rstates.patches[i])[inner, :] .-
                                parent(states.patches[i])[inner, :])))
    end
    check("restart: every block continues bitwise", gmax(d), 1e-300)
    # A different initial region is rebuilt to the recorded one too.
    moved = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per3[2], per3[3]),
                ic=nothing),
        Numerics(n_global=(400, 1, 1), cfl=0.2,
                 amr=AMR(initial=BlockRegion((200, 0, 0), (24, 1, 1)), subcycle=true,
                         regrid_interval=5, tag_buffer=2, tile=8, subdivide_lattice=false)))
    mstates = allocate_state(moved)
    load_checkpoint!(moved, mstates, "mpi_hier")
    check("restart from another initial region: recorded tile set rebuilt",
          level_regions(moved, 1) == regs ? 0.0 : 1.0, 0.5)
    # The multiblock dump of the hierarchy: one piece per patch per rank,
    # every rank's pieces listed under the index rank 0 writes.
    save_vtk(restarted, rstates, "mpi_hier_dump"; fields=(:rho,))
    MPI.Barrier(comm)
    held = MPI.Allreduce(length(restarted.patches), +, comm)
    listed = rank == 0 ? count("<DataSet", read("mpi_hier_dump.vtm", String)) : 0
    check("multiblock dump lists every rank's pieces", abs(gsum(listed) - held), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_hier"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# 7a''. The phase change, decomposed. A run ended by a WhenState condition
#     that is true on one rank only (the owner of the high-x plane) continues
#     under an NSCBC outflow there, whose correction is collective: every
#     rank must end the first phase on the same step, or the second hangs.
#     The continuation must match the one from a condition every rank
#     evaluates for itself, and the checkpoint restart into the same next
#     phase, bitwise. A phase switching the artificial properties on, and
#     one switching them off, continue bitwise as the solver of that setting
#     started from the carried state. Nothing here depends on the rank count
#     beyond one rank owning the plane and one not, which two ranks give.
# ---------------------------------------------------------------------------
function test_phase_change()
    section("phase change: one decision, bitwise as a checkpoint restart")
    pulse = (x, y, z) -> Prim(u=(0, 0, 0), p=1 + 4exp(-200(x - 0.3)^2),
                              rho=1 + exp(-200(x - 0.3)^2))
    faces(high) = ((SlipWallBC(), high), per3[2], per3[3])
    tube(high) = Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                         bcs=faces(high), ic=pulse)
    num = Numerics(n_global=(SPLITN, 16, 16), execution=Execution(dims=splitdims(1)))
    opened = faces(NSCBCOutflowBC(pinf=1.0))
    first_phase(condition) = begin
        s, Q = setup(tube(SlipWallBC()), num)
        run!(s, Q; tfinal=1e9, nmax=12,
             callback=Callback(WhenState(condition), Returns(true)))
        s, Q
    end
    s_local, Q_local = first_phase((s, _) -> rank == np - 1 && s.step >= 5)
    s_glob, Q_glob = first_phase((s, _) -> s.step >= 5)
    check("rank-local stop: the step every rank ended on",
          gmax(abs(s_local.step - 5)), 0.5)
    save_checkpoint(s_glob, Q_glob, "mpi_phase")
    MPI.Barrier(comm)
    p_local, pq_local = setup(s_local, Q_local; bcs=opened)
    p_glob, pq_glob = setup(s_glob, Q_glob; bcs=opened)
    r, rq = setup(tube(NSCBCOutflowBC(pinf=1.0)), num)
    load_checkpoint!(r, rq, "mpi_phase"; allow=(:boundaries,))
    for (s, Q) in ((p_local, pq_local), (p_glob, pq_glob), (r, rq))
        run!(s, Q; tfinal=1e9, nmax=12)
    end
    inner = CL.interior(r.decomp)
    gap(a, b) = maximum(abs.(parent(a)[inner, :] .- parent(b)[inner, :]))
    check("rank-local vs global stop: state difference",
          gmax(gap(pq_local, pq_glob)), 1e-300)
    check("phase change vs checkpoint restart: state difference",
          gmax(gap(pq_glob, rq)), 1e-300)
    check("phase change vs checkpoint restart: clock",
          gmax(abs(p_glob.t - r.t)) + gmax(abs(p_glob.dt_prev - r.dt_prev)), 1e-300)

    # The artificial properties switched on after a run without them, and off
    # again. Switched on, the phase continues as a fresh art-on run of the
    # carried state, whose coefficients `run!` computes before its first step
    # (its step count left at 0 for that); switched off, as the art-off solver
    # given the carried state.
    num_off = Numerics(num; art=ArtificialProperties(enabled=false))
    function start_from!(into, qi, from, qf; fresh)
        parent(qi)[inner, :] .= parent(qf)[inner, :]
        into.t, into.cfl = from.t, from.cfl
        into.dt_prev, into.rate_prev = from.dt_prev, from.rate_prev
        into.filter_rate_prev = from.filter_rate_prev
        fresh || (into.step = from.step)
        CL.refresh_primitives!(into, qi)
        return into
    end
    s0, q0 = setup(tube(SlipWallBC()), num_off)
    run!(s0, q0; tfinal=1e9, nmax=6)
    a_on, aq_on = setup(s0, q0; numerics=num)
    check("art switched on: coefficients computed at the phase change",
          gmax(maximum(abs, CL.art_block(a_on))) > 0 ? 0.0 : 1.0, 0.5)
    ref_on, rq_on = setup(tube(SlipWallBC()), num)
    start_from!(ref_on, rq_on, s0, q0; fresh=true)
    run!(a_on, aq_on; tfinal=1e9, nmax=14)
    run!(ref_on, rq_on; tfinal=1e9, nmax=8)
    check("art switched on vs a fresh art-on run: state difference",
          gmax(gap(aq_on, rq_on)), 1e-300)
    check("art switched on vs a fresh art-on run: clock",
          gmax(abs(a_on.t - ref_on.t)) + gmax(abs(a_on.dt_prev - ref_on.dt_prev)),
          1e-300)
    a_off, aq_off = setup(a_on, aq_on; numerics=num_off)
    ref_off, rq_off = setup(tube(SlipWallBC()), num_off)
    start_from!(ref_off, rq_off, a_on, aq_on; fresh=false)
    run!(a_off, aq_off; tfinal=1e9, nmax=20)
    run!(ref_off, rq_off; tfinal=1e9, nmax=20)
    check("art switched off vs the art-off solver: state difference",
          gmax(gap(aq_off, rq_off)) + gmax(abs(a_off.t - ref_off.t)), 1e-300)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_phase"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# 7a'''. The phase change of a hierarchy, decomposed. The tiled, regridded,
#     subcycled Sod of the hierarchy checkpoint phase changes its high wall to
#     an NSCBC outflow: the phase change rebuilds the tiles on their stored
#     owner ranges and continues, regrids and all, as its own checkpoint
#     restart does. The owner ranges and the tiles a rank holds depend on the
#     rank count.
# ---------------------------------------------------------------------------
function test_hierarchy_phase_change()
    section("hierarchy phase change: tiles carried on their owners")
    wall2 = (SlipWallBC(), SlipWallBC())
    sod = (x, y, z) -> x < 0.45 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                                  Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    shock_tube(xbc) = Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                              bcs=(xbc, per3[2], per3[3]), ic=sod)
    amr_num = Numerics(n_global=(400, 1, 1), cfl=0.2,
                       amr=AMR(initial=BlockRegion((176, 0, 0), (16, 1, 1)), tile=8,
                               subdivide_lattice=false,
                               regrid_interval=5, tag_buffer=2, subcycle=true))
    open_end = (SlipWallBC(), NSCBCOutflowBC(pinf=0.1))
    s, states = setup(shock_tube(wall2), amr_num)
    run!(s, states; tfinal=0.03, nmax=21)
    regs, owners = level_regions(s, 1), copy(s.levels[2].owners)
    save_checkpoint(s, states, "mpi_phase_hier")
    MPI.Barrier(comm)
    p, ps = setup(s, states; bcs=(open_end, per3[2], per3[3]))
    check("hierarchy phase: tile set and ownership carried",
          level_regions(p, 1) == regs && p.levels[2].owners == owners ? 0.0 : 1.0, 0.5)
    r, rs = setup(shock_tube(open_end), amr_num)
    load_checkpoint!(r, rs, "mpi_phase_hier"; allow=(:boundaries,))
    run!(p, ps; tfinal=0.03, nmax=41)
    run!(r, rs; tfinal=0.03, nmax=41)
    check("hierarchy phase: tile set tracks the restart",
          level_regions(p, 1) == level_regions(r, 1) &&
          p.levels[2].owners == r.levels[2].owners ? 0.0 : 1.0, 0.5)
    check("hierarchy phase: clock matches the restart", gmax(abs(p.t - r.t)), 1e-300)
    d = length(ps.patches) == length(rs.patches) ? 0.0 : Inf
    for i in eachindex(ps.patches)
        d == Inf && break
        inner = CL.interior(p.patches[i].decomp)
        d = max(d, maximum(abs.(parent(ps.patches[i])[inner, :] .- parent(rs.patches[i])[inner, :])))
    end
    check("hierarchy phase: every block matches the restart", gmax(d), 1e-300)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_phase"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# 7b'. The unrefined start, decomposed: the tiled, regridded level of the
#     serial case holds no tile on any rank at setup, the shock fired from a
#     Dirichlet inflow creates tiles that some ranks hold none of, the tiles
#     follow the shock as serially, and runs checkpointed before and after
#     the creation continue bitwise on the same rank count. A check on a
#     quiescent state then removes every tile on every rank together.
# ---------------------------------------------------------------------------
function test_unrefined_start()
    section("unrefined start: tiles created by a boundary-driven shock")
    ramp(t) = clamp((t - 0.04) / 0.01, 0.0, 1.0)
    inflow = DirichletBC((x, y, z, t) -> (w = ramp(t);
        Prim(rho=1 + 0.8621w, u=(0.8216w, 0.0, 0.0), p=1 + 1.4583w)))
    rest = (x, y, z) -> Prim(rho=1.0, u=(0.0, 0.0, 0.0), p=1.0)
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=((inflow, SlipWallBC()), per3[2], per3[3]), ic=rest)
    num = Numerics(n_global=(121, 1, 1), cfl=0.3,
                   amr=AMR(initial=:sensor, tile=8, regrid_interval=5,
                           subdivide_lattice=false, tag_buffer=2, subcycle=true))
    empty_everywhere(s, states) =
        gmax(!(isempty(level_regions(s, 1)) && length(states.patches) == 1 &&
               !s.levels[2].level_comm.owned &&
               all(==(0), s.patches[1].amr.covered)))
    solver, states = setup(prob, num)
    check("unrefined start: no tile on any rank at setup",
          empty_everywhere(solver, states), 0.5)
    run!(solver, states; tfinal=1.0, nmax=15)
    check("unrefined start: no tile before the shock (step 15)",
          empty_everywhere(solver, states), 0.5)
    save_checkpoint(solver, states, "mpi_unref_before")
    run!(solver, states; tfinal=1.0, nmax=50)
    regs50 = level_regions(solver, 1)
    check("unrefined start: tiles created by step 50 as serial (4, 8)",
          [r.offset[1] for r in regs50] == [4, 8] ? 0.0 : 1.0, 0.5)
    check("unrefined start: state vector aligned with the patches",
          gmax(length(states.patches) != length(solver.patches)), 0.5)
    save_checkpoint(solver, states, "mpi_unref_after")
    run!(solver, states; tfinal=1.0, nmax=110)
    regs = level_regions(solver, 1)
    check("unrefined start: tiles follow the shock as serial (4, 8, 16)",
          [r.offset[1] for r in regs] == [4, 8, 16] ? 0.0 : 1.0, 0.5)
    ps = PatchSolver(solver, solver.patches[1])
    off = ps.decomp.offset[1]
    local_shock = something(findfirst(i -> states.patches[1][padded_index(ps, i, 1, 1), 1] < 1.43,
                                      1:ps.decomp.n_local[1]), typemax(Int) - off)
    shock = Int(MPI.Allreduce(local_shock + off, MPI.MIN, comm))
    check("unrefined start: the shock lies in a tile",
          any(r -> r.offset[1] < shock <= r.offset[1] + r.extent[1], regs) ? 0.0 : 1.0,
          0.5)
    idle = gsum(length(states.patches) == 1)
    np > 2 * length(regs) &&
        check("unrefined start: some rank holds no tile", idle > 0 ? 0.0 : 1.0, 0.5)
    MPI.Barrier(comm)
    for stem in ("before", "after")
        r, rs = setup(prob, num)
        load_checkpoint!(r, rs, "mpi_unref_" * stem)
        want = stem == "before" ? BlockRegion[] : regs50
        check("unrefined start: restart $stem rebuilds the recorded tiles",
              level_regions(r, 1) == want ? 0.0 : 1.0, 0.5)
        run!(r, rs; tfinal=1.0, nmax=110)
        check("unrefined start: restart $stem reaches the same time",
              gmax(abs(r.t - solver.t)), 1e-300)
        tracks = level_regions(r, 1) == regs &&
                 getfield(r, :regrid).created == getfield(solver, :regrid).created
        check("unrefined start: restart $stem tracks the tile set",
              tracks ? 0.0 : 1.0, 0.5)
        d = length(rs.patches) == length(states.patches) ? 0.0 : Inf
        if isfinite(d)
            for i in eachindex(states.patches)
                inner = CL.interior(solver.patches[i].decomp)
                d = max(d, maximum(abs.(parent(rs.patches[i])[inner, :] .-
                                        parent(states.patches[i])[inner, :])))
            end
        end
        check("unrefined start: restart $stem continues bitwise", gmax(d), 1e-300)
    end
    initialize!(solver, states, rest)
    workspace = Workspace(states)
    getfield(solver, :regrid).checks += 1
    CL.regrid!(solver, states, workspace, nothing)
    check("unrefined start: a quiescent check removes every tile",
          empty_everywhere(solver, states), 0.5)
    run!(solver, states; tfinal=1.0, nmax=solver.step + 10)
    check("unrefined start: the re-formed shock creates tiles again",
          isempty(level_regions(solver, 1)) ? 1.0 : 0.0, 0.5)
    check("unrefined start: finite composite state",
          gmax(!all(all(isfinite, parent(Q)) for Q in states.patches)), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_unref"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# 7b''. Three regridded levels, decomposed: the serial level-test case (a
#     shock driven into light gas and through a contact into heavy gas,
#     max_levels = 3 through the AMR frontend) on this run's ranks against
#     the same run on COMM_SELF: the same tile sets on both refined levels and
#     the composite density to round-off. A checkpoint taken while level 2
#     exists continues bitwise on the same rank count.
# ---------------------------------------------------------------------------
function test_deep_regrid()
    section("deep regrid: three regridded levels follow a shock")
    ic = (x, y, z) -> x < 0.25 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                      x < 0.5  ? Prim(u=(0, 0, 0), p=0.1, rho=0.125) :
                                 Prim(u=(0, 0, 0), p=0.1, rho=0.5)
    wall2 = (SlipWallBC(), SlipWallBC())
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=(wall2, per3[2], per3[3]), ic=ic)
    amr = AMR(initial=:sensor, tile=8, regrid_interval=4, subcycle=true,
              subdivide_lattice=false, max_levels=3)
    num(c) = Numerics(n_global=(81, 1, 1), cfl=0.2, execution=Execution(comm=c), amr=amr)
    solver, states = setup(prob, num(comm))
    ref, rstates = setup(prob, num(MPI.COMM_SELF))
    # A rank outside a level's parent holds none of its layout; rank 0
    # holds every level's.
    layout(s) = (level_regions(s, 1), level_regions(s, 2))
    as_serial(s, r) = MPI.bcast(layout(s) == layout(r), comm; root=0) ? 0.0 : 1.0
    check("deep regrid: the initial layout is the serial one",
          as_serial(solver, ref), 0.5)
    run!(solver, states; tfinal=0.1, nmax=10000)
    save_checkpoint(solver, states, "mpi_deep")
    run!(solver, states; tfinal=0.15, nmax=10000)
    run!(ref, rstates; tfinal=0.1, nmax=10000)
    run!(ref, rstates; tfinal=0.15, nmax=10000)
    check("deep regrid: the same steps as serial",
          gmax(solver.step != ref.step), 0.5)
    check("deep regrid: both refined levels regrid as serial",
          as_serial(solver, ref), 0.5)
    # A rank outside level 2's rank set holds none of its tiles, so the test
    # is whether any rank does.
    check("deep regrid: level 2 holds tiles",
          1.0 - gmax(!isempty(level_regions(solver, 2))), 0.5)
    check("deep regrid: state vector aligned with the patches",
          gmax(length(states.patches) != length(solver.patches)), 0.5)
    _, rho = line_sample(solver, states, :rho)
    _, rho_ref = line_sample(ref, rstates, :rho)
    check("deep regrid: composite density matches serial",
          gmax(maximum(abs, rho .- rho_ref)), 1e-9)
    MPI.Barrier(comm)
    r, rs = setup(prob, num(comm))
    load_checkpoint!(r, rs, "mpi_deep")
    run!(r, rs; tfinal=0.15, nmax=10000)
    check("deep regrid: restart rebuilds and tracks both levels",
          gmax(layout(r) != layout(solver) ||
               getfield(r, :regrid).created !=
               getfield(solver, :regrid).created), 0.5)
    d = length(rs.patches) == length(states.patches) && r.step == solver.step ? 0.0 : Inf
    if isfinite(d)
        for i in eachindex(states.patches)
            inner = CL.interior(solver.patches[i].decomp)
            d = max(d, maximum(abs.(parent(rs.patches[i])[inner, :] .-
                                    parent(states.patches[i])[inner, :])))
        end
    end
    check("deep regrid: restart continues bitwise", gmax(d), 1e-300)
    check("deep regrid: finite composite state",
          gmax(!all(all(isfinite, parent(Q)) for Q in states.patches)), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_deep"), readdir()))
    MPI.Barrier(comm)
end

# The deep regrid on small tiles, a phase of its own because its rank-subset
# and moved-tile checks read only at np = 8.
function test_deep_regrid_subsets()
    section("deep regrid under subsets: levels on rank subsets, a moved tile")
    wall2 = (SlipWallBC(), SlipWallBC())
    layout(s) = (level_regions(s, 1), level_regions(s, 2))
    # A single shock on small tiles (edge 4, buffer 1), where the levels'
    # rank subsets fall below the run at np = 8 and a level-2 survivor's
    # range leaves its parent's subset, so the tile moves. The layouts are
    # compared every three steps and the density at the end.
    sod = (x, y, z) -> x < 0.35 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                                  Prim(u=(0, 0, 0), p=0.1, rho=0.125)
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=(wall2, per3[2], per3[3]), ic=sod)
    amr = AMR(initial=:sensor, tile=4, tag_buffer=1, regrid_interval=3,
              subdivide_lattice=false, subcycle=true, max_levels=3)
    num(c) = Numerics(n_global=(121, 1, 1), cfl=0.1, filter=compact_filter(0.45),
                      execution=Execution(comm=c), amr=amr)
    solver, states = setup(prob, num(comm))
    ref, rstates = setup(prob, num(MPI.COMM_SELF))
    differ = 0
    subsets = 0
    moves = 0
    # At np = 8 a level-2 survivor moves at the checks after steps 6 and 30.
    for n in 3:3:33
        owners0 = [(lt.region, o) for (lt, o) in
                   zip(solver.levels[3].transfers, solver.levels[3].owners)]
        run!(solver, states; tfinal=1.0, nmax=n)
        run!(ref, rstates; tfinal=1.0, nmax=n)
        differ += rank == 0 && layout(solver) != layout(ref)
        subsets += solver.levels[3].level_comm.size < np
        owners1 = Dict(lt.region => o for (lt, o) in
                       zip(solver.levels[3].transfers, solver.levels[3].owners))
        moves += count(((r, o),) -> haskey(owners1, r) && owners1[r] != o, owners0)
    end
    _, rho = line_sample(solver, states, :rho)
    _, rho_ref = line_sample(ref, rstates, :rho)
    check("deep regrid under subsets: composite density matches serial",
          gmax(maximum(abs, rho .- rho_ref)), 1e-9)
    # Rank 0 holds every level's layout and owners (a rank outside a
    # level's parent holds neither), so its counts are the run's.
    counts = MPI.bcast((differ, subsets, moves), comm; root=0)
    check("deep regrid under subsets: both levels regrid as serial",
          counts[1], 0.5)
    if np == 8
        check("deep regrid under subsets: level 2 on a rank subset",
              counts[2] > 0 ? 0.0 : 1.0, 0.5)
        check("deep regrid under subsets: a level-2 survivor moved",
              counts[3] > 0 ? 0.0 : 1.0, 0.5)
    end
end

# Lattice subdivision: a level with fewer tiles than ranks takes the finer
# lattice the cost model prices cheapest (`_choose_lattice`), every tile
# whole on one rank, and its new tiles take the old tiles' solution
# (`_carry_tiles!`). The model's coefficients are fixed to the fine nodes of
# the most loaded rank (`FIXED_LATTICE_COSTS`), so the layout is the same at
# every run. A 2-D level of one 16-node tile, built with the subdivision
# off, is regridded with it on: at np = 2, 4 and 8 the level takes 2, 4 and
# 8 tiles, one per rank, and every node the old tile held as its own
# solution reads the old value bit for bit, all components. A three-level
# shock then runs with the subdivision on; the "lattice over nodes" phase
# runs it with the coefficients measured.
function subdivision_carry()
    inside(p, I) = begin
        i, j, _ = CL.interior_index(p, I)
        gi, gj = i + p.decomp.offset[1], j + p.decomp.offset[2]
        18 <= gi <= 32 && 18 <= gj <= 32
    end
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(48, 48, 1),
                 amr=AMR(initial=BlockRegion((16, 16, 0), (17, 17, 1)), regrid_interval=1,
                         tag_threshold=Inf, tag_buffer=0, tile=16, subdivide_lattice=false));
        tag_predicate=inside)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(0.3, 0.2, 0), p=1.0, rho=1.0 + 0.1 * sin(2π * x) * sin(2π * y)))
    run!(solver, states; tfinal=1.0, nmax=2)
    lev = solver.levels[2]
    check("lattice subdivision: one tile before", abs(length(lev.transfers) - 1), 0.5)
    old = lev.transfers[1]
    owner = first(lev.owners[1])
    of = CL._fine_region(old)
    own = CL._erode(of, old.imposed, (true, true, false))
    # The old tile's padded state, from its owner to every rank.
    dims = zeros(Int64, 7)
    A = nothing
    if rank == owner
        li = lev.patches[1]
        A = Array(parent(states.patches[li]))
        dims .= [size(A)..., solver.patches[li].decomp.n_halo_d...]
    end
    MPI.Bcast!(dims, comm; root=owner)
    rank == owner || (A = zeros(Float64, dims[1:4]...))
    MPI.Bcast!(A, comm; root=owner)
    pad_old = Tuple(dims[5:7])
    getfield(solver, :regrid).subdivide = true
    CL.regrid!(solver, states, CL.Workspace(states), nothing)
    lev = solver.levels[2]
    spec = getfield(solver, :regrid)
    expect = Dict(1 => (16, 16, 16), 2 => (8, 16, 16), 4 => (8, 8, 16),
                  8 => (4, 8, 16))
    check("lattice subdivision: a tile per rank", abs(length(lev.transfers) - np), 0.5)
    check("lattice subdivision: the coarsest lattice giving them",
          CL._level_edges(spec, 1) == get(expect, np, (0, 0, 0)) ? 0.0 : 1.0, 0.5)
    check("lattice subdivision: each tile whole on one rank",
          all(length(o) == 1 for o in lev.owners) ? 0.0 : 1.0, 0.5)
    miss = 0
    compared = 0
    for (li, ti) in zip(lev.patches, lev.tiles)
        nf = CL._fine_region(lev.transfers[ti])
        pad = solver.patches[li].decomp.n_halo_d
        Q = Array(parent(states.patches[li]))
        for j in 1:nf.extent[2], i in 1:nf.extent[1]
            g = (nf.offset[1] + i, nf.offset[2] + j)
            all(d -> own.offset[d] < g[d] <= own.offset[d] + own.extent[d], 1:2) ||
                continue
            for c in 1:size(Q, 4)
                compared += 1
                Q[i + pad[1], j + pad[2], 1 + pad[3], c] ==
                    A[g[1] - of.offset[1] + pad_old[1], g[2] - of.offset[2] + pad_old[2],
                      1 + pad_old[3], c] || (miss += 1)
            end
        end
    end
    # The planes the new tiles share are compared on both sides.
    check("lattice subdivision: the old tile's nodes all compared",
          max(0.0, prod(own.extent) * size(A, 4) - gsum(compared)), 0.5)
    check("lattice subdivision: carried bit for bit", gsum(miss), 0.5)
    run!(solver, states; tfinal=1.0, nmax=6)
    check("lattice subdivision: the lattice holds while the tiles do",
          abs(length(solver.levels[2].transfers) - np), 0.5)
    check("lattice subdivision: finite state after the run",
          gmax(!all(all(isfinite, parent(Q)) for Q in states.patches)), 0.5)
end

# Three regridded levels following the shock of the deep-regrid phase, with
# the subdivision on: every tile whole on one rank, each level on a lattice
# of its chain, and a restart that continues bit for bit on the lattices the
# checkpoint records.
function subdivision_deep()
    ic = (x, y, z) -> x < 0.25 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                      x < 0.5  ? Prim(u=(0, 0, 0), p=0.1, rho=0.125) :
                                 Prim(u=(0, 0, 0), p=0.1, rho=0.5)
    wall2 = (SlipWallBC(), SlipWallBC())
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=(wall2, per3[2], per3[3]), ic=ic)
    amr = AMR(initial=:sensor, tile=8, regrid_interval=4, subcycle=true,
              max_levels=3)
    num = Numerics(n_global=(81, 1, 1), cfl=0.2, amr=amr)
    solver, states = setup(prob, num)
    run!(solver, states; tfinal=0.1, nmax=10000)
    save_checkpoint(solver, states, "mpi_subdivision")
    run!(solver, states; tfinal=0.15, nmax=10000)
    spec = getfield(solver, :regrid)
    edges = [CL._level_edges(spec, l) for l in 1:2]
    rank == 0 && println("    lattices ", edges, ", tiles ",
                         [length(level_regions(solver, l)) for l in 1:2])
    whole = all(all(length(o) == 1 for o in lev.owners) for lev in solver.levels[2:end])
    check("lattice subdivision, three levels: each tile whole on one rank",
          gmax(!whole), 0.5)
    check("lattice subdivision, three levels: finite composite state",
          gmax(!all(all(isfinite, parent(Q)) for Q in states.patches)), 0.5)
    if np == 8
        check("lattice subdivision, three levels: a level subdivided at eight ranks",
              MPI.bcast(any(!=((8, 8, 8)), edges), comm; root=0) ? 0.0 : 1.0, 0.5)
    end
    MPI.Barrier(comm)
    r, rs = setup(prob, num)
    load_checkpoint!(r, rs, "mpi_subdivision")
    run!(r, rs; tfinal=0.15, nmax=10000)
    rspec = getfield(r, :regrid)
    check("lattice subdivision, three levels: the restart takes the lattices",
          gmax([CL._level_edges(rspec, l) for l in 1:2] != edges), 0.5)
    d = length(rs.patches) == length(states.patches) && r.step == solver.step ? 0.0 : Inf
    if isfinite(d)
        for i in eachindex(states.patches)
            inner = CL.interior(solver.patches[i].decomp)
            d = max(d, maximum(abs.(parent(rs.patches[i])[inner, :] .- parent(states.patches[i])[inner, :])))
        end
    end
    check("lattice subdivision, three levels: restart continues bitwise", gmax(d), 1e-300)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_subdivision"), readdir()))
    MPI.Barrier(comm)
end

# The three-level shock with the model's coefficients measured on the run:
# the model is fitted from the run's timers, every rank holds the same model
# and takes the same lattices, and every tile stays whole on one rank.
function subdivision_measured(label="lattice subdivision, measured")
    ic = (x, y, z) -> x < 0.25 ? Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
                      x < 0.5  ? Prim(u=(0, 0, 0), p=0.1, rho=0.125) :
                                 Prim(u=(0, 0, 0), p=0.1, rho=0.5)
    wall2 = (SlipWallBC(), SlipWallBC())
    prob = Problem(eos=IdealSpecies("gas"; gamma=1.4, R=1.0),
                   domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                   bcs=(wall2, per3[2], per3[3]), ic=ic)
    amr = AMR(initial=:sensor, tile=8, regrid_interval=4, subcycle=true,
              max_levels=3)
    num = Numerics(n_global=(81, 1, 1), cfl=0.2, amr=amr)
    solver, states = setup(prob, num)
    run!(solver, states; tfinal=0.1, nmax=10000)
    spec = getfield(solver, :regrid)
    costs = spec.costs
    model = [costs.a, costs.b, costs.k, costs.m, costs.c, costs.root, costs.m_x]
    edges = [CL._level_edges(spec, l) for l in 1:2]
    # Level 1's placement, which every rank holds (its parent is the root).
    owners = solver.levels[2].owners
    rank == 0 && println("    measured model (a, b, k, m, c, root, m_x) ", model,
                         ", lattices ", edges)
    # A fit is made once a tile is timed, and it always carries a cost per
    # node (`_fit_lattice_costs`); the other coefficients depend on the run's
    # timings and may come out zero.
    check("$label: the model fitted",
          costs.ready && costs.a > 0 ? 0.0 : 1.0, 0.5)
    check("$label: every rank holds the same model",
          gmax(model != MPI.bcast(model, comm; root=0)), 0.5)
    check("$label: every rank takes the same lattices",
          gmax(edges != MPI.bcast(edges, comm; root=0)), 0.5)
    check("$label: every rank places level 1 alike",
          gmax(owners != MPI.bcast(owners, comm; root=0)), 0.5)
    whole = all(all(length(o) == 1 for o in lev.owners) for lev in solver.levels[2:end])
    check("$label: each tile whole on one rank", gmax(!whole), 0.5)
    check("$label: finite composite state",
          gmax(!all(all(isfinite, parent(Q)) for Q in states.patches)), 0.5)
end

function test_lattice_subdivision()
    section("lattice subdivision: a tile per rank, the solution carried")
    try
        CL.FIXED_LATTICE_COSTS[] = (1.0, 0.0, 0.0, 0.0, 0.0)
        subdivision_carry()
        subdivision_deep()
    finally
        CL.FIXED_LATTICE_COSTS[] = nothing
    end
end

# Lattice and placement over nodes (`_node_map`): the ranks grouped into
# nodes of two at np >= 4 and of one rank below (`NODE_SIZE`), so the paths
# of a run across nodes run on one machine. The node map MPI reports agrees
# on every rank. With the coefficients fixed to the fine nodes of the most
# loaded rank and a message leaving the node priced at 2000 of them, about
# the nodes of the level's one 16-node tile, while one within it is free,
# the level subdivides only where the new tiles' shared faces stay within a
# node: at nodes of one rank it keeps the one tile, and at nodes of two it
# takes two tiles on the two ranks of the first node.
# Placing a level of more tiles than ranks keeps each node's tiles a run of
# the space-filling curve, alike on every rank. The measured three-level
# shock then fits the per-message coefficients within and across nodes,
# alike on every rank.
function test_lattice_nodes()
    section("lattice over nodes: node map, off-node prices, placement by node")
    group = np >= 4 ? 2 : 1
    shared = CL._node_map(comm)
    check("lattice over nodes: the shared-memory map alike on every rank",
          gmax(shared != MPI.bcast(shared, comm; root=0)), 0.5)
    check("lattice over nodes: the shared-memory map numbered from 0",
          shared[1] == 0 && all(diff(shared) .>= 0) && length(shared) == np ? 0.0 : 1.0,
          0.5)
    try
        CL.NODE_SIZE[] = group
        check("lattice over nodes: the grouped map",
              CL._node_map(comm) == [r ÷ group for r in 0:(np - 1)] ? 0.0 : 1.0, 0.5)
        CL.FIXED_LATTICE_COSTS[] = (1.0, 0.0, 0.0, 0.0, 0.0, 2000.0)
        inside(p, I) = begin
            i, j, _ = CL.interior_index(p, I)
            gi, gj = i + p.decomp.offset[1], j + p.decomp.offset[2]
            18 <= gi <= 32 && 18 <= gj <= 32
        end
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(48, 48, 1),
                     amr=AMR(initial=BlockRegion((16, 16, 0), (17, 17, 1)),
                             regrid_interval=1, tag_threshold=Inf, tag_buffer=0, tile=16));
            tag_predicate=inside)
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            Prim(u=(0.3, 0.2, 0), p=1.0, rho=1.0 + 0.1 * sin(2π * x) * sin(2π * y)))
        run!(solver, states; tfinal=1.0, nmax=3)
        lev = solver.levels[2]
        spec = getfield(solver, :regrid)
        want = group == 1 ? (16, 16, 16) : (8, 16, 16)
        check("lattice over nodes: subdivided within a node only",
              CL._level_edges(spec, 1) == want ? 0.0 : 1.0, 0.5)
        check("lattice over nodes: the tiles on the first node",
              all(o -> last(o) < group, lev.owners) ? 0.0 : 1.0, 0.5)
        check("lattice over nodes: finite state",
              gmax(!all(all(isfinite, parent(Q)) for Q in states.patches)), 0.5)
        # A level of four tiles per rank placed afresh: each node's tiles a
        # run of the curve, the ranks of every run on its node.
        act = (true, true, false)
        tiles = [BlockRegion((4k1, 4k2, 0), (5, 5, 1)) for k2 in 0:(2np - 1) for k1 in 0:1]
        nodes = spec.costs.nodes
        owners, _ = CL._balanced_owners(tiles, act, np, fill(0:0, length(tiles)), 1;
                                        nodes)
        on = [nodes[first(owners[t]) + 1] for t in CL._sfc_order(tiles)]
        check("lattice over nodes: each node's tiles a run of the curve",
              on == sort(on) ? 0.0 : 1.0, 0.5)
        check("lattice over nodes: every rank places them alike",
              gmax(owners != MPI.bcast(owners, comm; root=0)), 0.5)
        check("lattice over nodes: four tiles a rank",
              all(r -> count(o -> o == r:r, owners) == 4, 0:(np - 1)) ? 0.0 : 1.0, 0.5)
        CL.FIXED_LATTICE_COSTS[] = nothing
        subdivision_measured("lattice over nodes, measured")
    finally
        CL.NODE_SIZE[] = 0
        CL.FIXED_LATTICE_COSTS[] = nothing
    end
end

# ---------------------------------------------------------------------------
# 7b'''. Levels placed on a wall by tags (`level_boundaries`), decomposed: a
#     tagged window moving onto the high wall of a 1-D tube, followed by one
#     box and by lattice tiles whose last cell reaches the wall, against the
#     serial rebuild on COMM_SELF every three steps through the regrids: the
#     same regions, a wall face on the last one, and every block this rank
#     holds equal to the serial patch's.
# ---------------------------------------------------------------------------
function test_placed_levels()
    section("placed levels: tags place a level on a wall")
    walls = (SlipWallBC(), SlipWallBC())
    function blockdiff(p, Q, rp, Qref)
        e = 0.0
        for I in CL.interior(p.decomp), c in 1:size(Q, 4)
            loc = Tuple(I) .- p.decomp.n_halo_d
            J = CartesianIndex(loc .+ p.decomp.offset .+ rp.decomp.n_halo_d)
            e = max(e, abs(Float64(Q[I, c] - Qref[J, c])))
        end
        e
    end
    function leveldiff(s, qs, ref, qref)
        e = blockdiff(s.patches[1], qs.patches[1], ref.patches[1], qref.patches[1])
        lev, rlev = s.levels[2], ref.levels[2]
        for (k, ti) in zip(lev.patches, lev.tiles)
            rk = rlev.patches[findfirst(==(ti), rlev.tiles)]
            e = max(e, blockdiff(s.patches[k], qs.patches[k], ref.patches[rk], qref.patches[rk]))
        end
        e
    end
    # A window of half-width 0.08 moving right at 2, which reaches the wall
    # at x = 1 after a few checks, over a standing acoustic wave.
    window(p, I) = abs(CL.xcoord(p, 1, CL.interior_index(p, I)[1]) - (0.85 + 2 * p.t)) <
                   0.08
    wave = (x, y, z) -> (rho = 1 + 0.05 * cos(pi * x);
                         Prim(rho=rho, u=(0.05 * sin(pi * x), 0.0, 0.0), p=rho^1.4))
    function build(comm_here, tile)
        local s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=(walls, per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(97, 1, 1), art=ArtificialProperties(enabled=false), cfl=0.5,
                     filter=StateFilter(interval=1), execution=Execution(comm=comm_here),
                     amr=AMR(initial=BlockRegion((64, 0, 0), (17, 1, 1)), regrid_interval=3,
                             tag_threshold=Inf, tag_buffer=2, tile=tile,
                             level_boundaries=true, subdivide_lattice=false));
            tag_predicate=window)
        local q = allocate_state(s)
        initialize!(s, q, wave)
        return s, q
    end
    for (label, tile) in (("box", 0), ("tiles", 8))
        s, q = build(comm, tile)
        ref, qref = build(MPI.COMM_SELF, tile)
        differ = 0
        walled = false
        e = 0.0
        for n in 3:3:30
            run!(s, q; tfinal=1e9, nmax=n)
            run!(ref, qref; tfinal=1e9, nmax=n)
            differ += level_regions(s, 1) != level_regions(ref, 1)
            regs = level_regions(s, 1)
            walled |= !isempty(regs) && regs[end].offset[1] + regs[end].extent[1] == 97 &&
                      s.levels[2].transfers[end].boundary[1][2]
            differ == 0 && (e = max(e, leveldiff(s, q, ref, qref)))
        end
        check("placed levels, $label: the regions regrid as serial", gmax(differ), 0.5)
        check("placed levels, $label: the level reaches the wall",
              gmax(!walled), 0.5)
        check("placed levels, $label: every block matches serial", gmax(e), 1e-10)
    end
end

# ---------------------------------------------------------------------------
# A level across a periodic seam, decomposed: the box of a tile across the
# seam draws parent nodes from the ranks on both sides of it and its
# restriction writes to both, so one message between a pair of ranks may
# carry pieces of both images of the parent, in the order both sides derive
# from the block tables. A static box and tiles across the seam, and a box
# and tiles regridded across it under level_boundaries, each against the
# serial rebuild on COMM_SELF, block by block.
# ---------------------------------------------------------------------------
function test_seam_levels()
    section("seam levels: a level across a periodic seam")
    function blockdiff(p, Q, rp, Qref)
        e = 0.0
        for I in CL.interior(p.decomp), c in 1:size(Q, 4)
            loc = Tuple(I) .- p.decomp.n_halo_d
            J = CartesianIndex(loc .+ p.decomp.offset .+ rp.decomp.n_halo_d)
            e = max(e, abs(Float64(Q[I, c] - Qref[J, c])))
        end
        e
    end
    function leveldiff(s, qs, ref, qref)
        e = blockdiff(s.patches[1], qs.patches[1], ref.patches[1], qref.patches[1])
        lev, rlev = s.levels[2], ref.levels[2]
        for (k, ti) in zip(lev.patches, lev.tiles)
            rk = rlev.patches[findfirst(==(ti), rlev.tiles)]
            e = max(e, blockdiff(s.patches[k], qs.patches[k], ref.patches[rk], qref.patches[rk]))
        end
        e
    end
    # A window of half-width 0.08 moving right at 2 from x = 0.95, across the
    # seam, over the entropy wave.
    window(p, I) = abs(mod(CL.xcoord(p, 1, CL.interior_index(p, I)[1]) -
                           (0.95 + 2 * p.t) + 0.5, 1) - 0.5) < 0.08
    wave = (x, y, z) -> Prim(rho=1 + 0.2 * sin(2pi * x + 0.37), u=(0.5, 0.0, 0.0),
                             p=1.0)
    function build(comm_here, tile, regrid; rebalance=0, rebalance_persist=2)
        local s = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(96, 1, 1), art=ArtificialProperties(enabled=false), cfl=0.5,
                     filter=StateFilter(interval=1), execution=Execution(comm=comm_here),
                     amr=AMR(initial=BlockRegion((84, 0, 0), (25, 1, 1)),
                             regrid_interval=regrid ? 3 : 0, tag_threshold=Inf,
                             tag_buffer=2, tile=tile, rebalance=rebalance,
                             rebalance_persist=rebalance_persist, level_boundaries=regrid,
                             subdivide_lattice=false));
            tag_predicate=regrid ? window : nothing)
        local q = allocate_state(s)
        initialize!(s, q, wave)
        return s, q
    end
    crosses(r) = r.offset[1] + r.extent[1] > 96
    for (label, tile, regrid) in (("static box", 0, false), ("static tiles", 8, false),
                                  ("regridded box", 0, true), ("regridded tiles", 8, true))
        s, q = build(comm, tile, regrid)
        ref, qref = build(MPI.COMM_SELF, tile, regrid)
        differ = 0
        crossed = false
        e = 0.0
        for n in 3:3:30
            run!(s, q; tfinal=1e9, nmax=n)
            run!(ref, qref; tfinal=1e9, nmax=n)
            differ += level_regions(s, 1) != level_regions(ref, 1)
            crossed |= any(crosses, level_regions(s, 1))
            differ == 0 && (e = max(e, leveldiff(s, q, ref, qref)))
        end
        check("seam levels, $label: the regions match serial", gmax(differ), 0.5)
        check("seam levels, $label: a region crosses the seam", gmax(!crossed), 0.5)
        check("seam levels, $label: every block matches serial", gmax(e), 1e-10)
        check("seam levels, $label: composite volume exact",
              abs(domain_volume(s) - 1), 1e-13)
    end
    # Repartitioned at every check, the tiles across the seam migrate block to
    # block, a tile's blocks being in its own node space; the audit holds the
    # migrated state to the replicated carry.
    CL.MIGRATION_AUDIT[] = true
    CL.MIGRATION_AUDIT_RESULT[] = (tiles=0, mismatches=0)
    s, q = build(comm, 8, true; rebalance=1.0, rebalance_persist=1)
    run!(s, q; tfinal=1e9, nmax=30)
    CL.MIGRATION_AUDIT[] = false
    audit = CL.MIGRATION_AUDIT_RESULT[]
    check("seam levels, rebalanced tiles: migrated state equals the carry bitwise",
          gsum(audit.mismatches), 0.5)
    check("seam levels, rebalanced tiles: the state stays finite",
          gmax(any(Q -> !all(isfinite, parent(Q)), q.patches)), 0.5)
end
