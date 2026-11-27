# The MPI suite's phases of decomposed refinement: coupling, tiles, rank subsets,
# covered masks and composite samples. Included by test/mpi_tests.jl, which loads
# the package, defines `check`, `section` and the other helpers used here, and runs
# the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# Distributed level transfer (reference/AMR_GPU.md). Both levels
# decompose over the whole rank set; the box and the covered samples move
# point to point between the root's and the fine patch's blocks, the fine
# patch's ranks divide the chains by component, and each rank writes its
# owned slots. The references are the np = 1 values of the identical cases; a
# decomposed run differs only where closed lines take the spike solve in
# place of the serial sweep, so the error metrics agree to round-off
# (the two-patch testset's tier), and the regridder must track to the same
# region through the same step count.
# ---------------------------------------------------------------------------
function test_refined_decomposed()
    section("distributed two-level refinement")
    u0 = 0.5
    # Twenty steps of each run: the checks are agreement with serial to
    # round-off, which every step tests alike, and on an oversubscribed
    # runner the phase cost is linear in steps (see the callback phase).
    # Returns the composite error, the step count and the error over the
    # refined levels alone, which the coarse-fine faces set.
    function wave_error(; subcycle, levels=2, art=ArtificialProperties(enabled=false),
                        filter_interval=0, k=1, filt=compact_filter(), tile=0,
                        interface_flux=:ghost, interface_divergence=nothing)
        N = 192
        r1 = BlockRegion((N ÷ 2 - N ÷ 12, 0, 0), (N ÷ 6, 1, 1))
        e1 = 3 * (N ÷ 6) - 2
        r2 = BlockRegion((3 * r1.offset[1] + e1 ÷ 4, 0, 0), (e1 ÷ 2, 1, 1))
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(N, 1, 1), art=art,
                     filter=StateFilter(filt; interval=filter_interval),
                     patch_interfaces=PatchInterfaces(flux=interface_flux,
                                                      divergence=interface_divergence),
                     amr=AMR(initial=levels == 3 ? [r1, r2] : r1, regrid_interval=0,
                             subcycle=subcycle, tile=tile)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(k * x)))
        run!(solver, states; tfinal=0.5, nmax=20)
        e = e_fine = 0.0
        for (ps, Q) in CL.eachpatch(solver, states)
            for i in 1:ps.decomp.n_local[1]
                I = padded_index(ps, i, 1, 1)
                ei = abs(Q[I, 1] - (1.0 + 0.2 * sin(k * (xcoord(ps, 1, i) -
                                                          u0 * solver.t))))
                e = max(e, ei)
                ps.h[1] < 2π / N / 2 && (e_fine = max(e_fine, ei))
            end
        end
        return gmax(e), solver.step, gmax(e_fine)
    end
    # The static rows carry the wavenumber-8 wave: at wavenumber 1 the ghost
    # fluxes leave a 20-step error of a few ulps (5.6e-15), and agreement
    # with serial at that level tests nothing. At wavenumber 8 the refined
    # levels' error, 1.9e-9 to 6.4e-9, peaks at a coarse-fine face, and the
    # decomposed runs reproduce it to 9e-16 at np = 2, 4 and 8. The
    # subcycled rows keep wavenumber 1, where the subcycled coupling sets a
    # 3.5e-12 error that the decomposed runs reproduce to 1e-15.
    e_static, n_static, ef_static = wave_error(; subcycle=false, k=8)
    check("static two-level refined-level wave error matches serial",
          abs(ef_static - 6.398861107648202e-9), 1e-14)
    check("static two-level step count matches serial",
          abs(n_static - 20), 0.5)
    # Subcycled: the fine patch's ghost fluxes are evaluated on the Hermite
    # shell imposed at each substage, interpolated at the order-8 default of
    # the ghost path.
    e_sub, n_sub = wave_error(; subcycle=true)
    check("subcycled two-level wave error matches serial",
          abs(e_sub - 3.4861002973229915e-12), 5e-15)
    check("subcycled two-level step count matches serial",
          abs(n_sub - 20), 0.5)
    # Three levels: the level-2 patch's coupling gathers over the level-1
    # patch's own decomposition, and the recursive driver's substep sequence
    # is collective at every depth.
    e3, n3, ef3 = wave_error(; subcycle=false, levels=3, k=8)
    check("static three-level refined-level wave error matches serial",
          abs(ef3 - 1.933603832604547e-9), 1e-14)
    check("static three-level step count matches serial", abs(n3 - 20), 0.5)
    e3s, n3s = wave_error(subcycle=true, levels=3)
    check("subcycled three-level wave error matches serial",
          abs(e3s - 3.4862113196254541e-12), 5e-15)
    check("subcycled three-level step count matches serial",
          abs(n3s - 20), 0.5)
    # The fine level's divergence under the Brady–Livescu interface rows,
    # planned on every rank of the level's decomposition. The rows close the
    # divergence at an interface, which only the closure-row path does.
    closure = (interface_flux=:closure,)
    e_src, _ = wave_error(; subcycle=false, closure...,
                          interface_divergence=lele_d1_6(closures=:brady_livescu))
    check("static two-level wave error, source divergence rows, matches serial",
          abs(e_src - 3.26405569239796e-14), 5e-15)
    # The d8 detector and the pentadiagonal filter, one fine box and eight-node
    # tiles: the detector's interface rows read the imposed and exchanged
    # ghosts, and both banded plans solve over the level's decomposition.
    # These rows test the detector and the filter, not the interface flux,
    # and keep the closure rows, whose 3e-10 error lies well above the
    # decomposition's effect. The pentadiagonal filter's decomposed solve
    # moves the solution by 1e-13 to 1.4e-12 against serial under either
    # interface flux and at every wavenumber measured, where the default
    # filter moves it by 7e-15. That is round-off: the pentadiagonal filter's
    # periodic left-hand side has condition number 2.1e3 against 19, and one
    # decomposed solve departs from serial by the same fraction of cond·eps
    # for both (`bench/archive/reducedsolve.jl mode=accuracy`). At wavenumber 1 the
    # ghost fluxes' error is 1.9e-13, the same size. The default's row runs
    # at wavenumber 8, where
    # the error is 7.5e-9 and the decomposed runs reproduce it to 1.6e-13.
    d8pyr = (art=ArtificialProperties(enabled=true, detector=:d8), filt=pyranda_filter(),
             filter_interval=1)
    e_d8, _ = wave_error(; subcycle=false, closure..., d8pyr...)
    check("static two-level wave error, d8 detector and pentadiagonal filter, " *
          "matches serial", abs(e_d8 - 3.0154478913857474e-10), 1e-12)
    e_d8t, _ = wave_error(; subcycle=true, tile=8, closure..., d8pyr...)
    check("subcycled tiled wave error, d8 detector and pentadiagonal filter, " *
          "matches serial", abs(e_d8t - 3.079330124222679e-10), 1e-12)
    e_d8g, _ = wave_error(; subcycle=false, k=8, d8pyr...)
    check("static two-level wave error, d8 and pentadiagonal filter, ghost " *
          "fluxes, matches serial", abs(e_d8g - 7.4608481615001665e-9), 1e-12)

    # Tagging-driven regridding tracks a Sod shock to the same region. The
    # four Sod regrid cases in this file run the unrelaxed filter
    # (filter_cfl = 0) at αf = 0.45: their tile histories, regions and times
    # reached were measured serially under it, and the tags follow the filter
    # strength. Every regrid check now refreshes artificial
    # coefficients before the next root CFL estimate, so the recorded times
    # below are the replacement trajectory; the relaxed default, a weaker
    # filter at cfl = 0.2, walks the tags differently.
    wall2 = (SlipWallBC(), SlipWallBC())
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per3[2], per3[3]),
                ic=nothing),
        Numerics(n_global=(400, 1, 1), cfl=0.2,
                 filter=StateFilter(compact_filter(0.45), cfl=0.0),
                 amr=AMR(initial=BlockRegion((160, 0, 0), (60, 1, 1)), subcycle=true,
                         regrid_interval=20, tag_buffer=16, subdivide_lattice=false)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> x < 0.45 ?
        Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
        Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    # Sixty-one steps: three regrids, at steps 21, 41 and 61. Measured
    # serially, the region ends at 157/48 (offset/extent) from 160/60, moved
    # and resized, and both numbers are checked. The run used to go
    # to t = 0.03 (265 steps, offset 149), which is the same test many more
    # times over: every regrid is a replicated tag-and-rebuild over the same
    # collectives, and on an oversubscribed runner the phase cost is linear in
    # steps (see the callback phase). The full crossing costs 1001 steps.
    lt0 = getfield(solver, :levels)[2].transfers[1]
    decomp0 = lt0.fine_index == 0 ? nothing : solver.patches[lt0.fine_index].decomp
    run!(solver, states; tfinal=0.03, nmax=61)
    region = CL.refined_region(solver)
    fin = all(all(isfinite, parent(Q)) for Q in states.patches)
    check("regridded Sod: finite composite state", fin ? 0.0 : 1.0, 0.5)
    check("regridded Sod: region offset tracks as serial (157)",
          abs(gmax(region.offset[1]) - 157), 0.5)
    check("regridded Sod: region extent tracks as serial (48)",
          abs(gmax(region.extent[1]) - 48), 0.5)
    # The step count is nmax by construction; the time reached is not. It is
    # the sum of 61 global dt decisions through three regrids, so a rank-count
    # dependence anywhere in the coarse–fine CFL reduction or the regridded
    # level's spacing moves it. Serial value at step 61; the tolerance is the
    # Allreduce round-off tier.
    check("regridded Sod: time reached matches serial",
          abs(gmax(solver.t) - 0.009761548245685925), 1e-14)
    # The regrids dropped the setup-time fine patch. Decomposed over more
    # than one rank, or over a split communicator, its decomposition owns
    # Cartesian communicators, which must be freed at the drop and not left
    # to the garbage collector (the serial suite's decompositions borrow
    # COMM_WORLD and own none, so only this suite can see the free happen).
    freed = decomp0 === nothing || !decomp0.owns_communicators ||
            decomp0.comm == MPI.COMM_NULL
    check("regridded Sod: dropped fine decomposition freed", freed ? 0.0 : 1.0, 0.5)
end

# ---------------------------------------------------------------------------
# Tiled level, decomposed: each tile is held whole by one rank of the level
# (`_tile_owners`), the tiles dealt out along a Morton curve. Four 37-node
# tiles at np = 2 cut the curve into two runs of two tiles; at np = 4 and 8
# each tile takes one rank, and at np = 8 four ranks hold none, which the CI
# job's np = 8 run of this phase exercises.
# ---------------------------------------------------------------------------
function test_partitioned_coupling()
    section("partitioned coupling: each rank moves its own tiles' data")
    # A planar slab of tiles, more tiles than ranks at np <= 8, the shape a
    # mixing layer takes. The coupling between the level and the root runs
    # point to point: a rank holds the chains and boxes of its own tiles only,
    # receives box data for those tiles only, and sends from its own root
    # block only; every box node arrives exactly once and every covered node
    # is written exactly once over the run. The interface tags number the
    # messages between one pair of ranks, so they stay small however many
    # tiles the level holds.
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 2.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(48, 96, 1),
                 amr=AMR(initial=BlockRegion((18, 8, 0), (12, 80, 1)), regrid_interval=0,
                         subcycle=true, tile=6)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) ->
        Prim(u=(0.2, 0, 0), p=1.0, rho=1.0 + 0.3 * tanh((x - 0.5) / 0.05)))
    lev = solver.levels[2]
    cp = lev.coupling
    tr = lev.transfers
    check("partitioned coupling: the level holds more tiles than ranks",
          length(tr) > 8 ? 0.0 : 1.0, 0.5)
    bare(lt) = isempty(lt.pstage) && isempty(lt.pplans) && isempty(lt.box_gather) &&
               isempty(lt.box_Q0) && isempty(lt.shell.slabs)
    check("partitioned coupling: chains and boxes on the tile's own ranks only",
          gmax(count(lt -> lt.fine_index == 0 && !bare(lt), tr)), 0.5)
    check("partitioned coupling: box and samples move for held tiles only",
          gmax(count(q -> tr[q.tile].fine_index == 0,
                     vcat(cp.box_recvs, cp.restrict_sends))), 0.5)
    check("partitioned coupling: sent from held parent blocks only",
          gmax(count(q -> tr[q.tile].coarse_local[q.part] == 0,
                     vcat(cp.box_sends, cp.restrict_recvs))), 0.5)
    # Every node of a held tile's box arrives once, from the pieces' union.
    miss = 0
    for (t, lt) in enumerate(tr)
        lt.fine_index == 0 && continue
        got = sum(prod(length.(q.ranges)) for q in cp.box_recvs if q.tile == t)
        miss += abs(got - prod(length.(CL._box_nodes(lt))))
    end
    check("partitioned coupling: each held box arrives whole", gmax(miss), 0.5)
    # Every covered node is written once, over all ranks together.
    written = zeros(Int, length(tr))
    for q in cp.restrict_recvs
        written[q.tile] += prod(length.(q.ranges))
    end
    written = MPI.Allreduce(written, +, comm)
    window = [prod(length.(CL._restrict_window(lt))) for lt in tr]
    check("partitioned coupling: each covered node written once",
          maximum(abs.(written .- window)), 0.5)
    tags = Int[]
    for d in 1:3
        append!(tags, (r.tag for r in lev.ghost_recvs[d]))
        append!(tags, (r.tag for r in lev.ghost_sends[d] if r.partner != MPI.Comm_rank(
                           lev.level_comm.comm)))
        append!(tags, (r.tag - CL._PLANE_TAG_BASE + CL._GHOST_TAG_BASE
                       for r in lev.plane_pairs[d] if r.partner != MPI.Comm_rank(
                           lev.level_comm.comm)))
    end
    check("partitioned coupling: interface tags number the messages of a rank pair",
          gmax(isempty(tags) ? 0 : maximum(tags) - CL._GHOST_TAG_BASE), 64.0)
    # The same run advanced, two regrid-free subcycled steps, reproduces
    # the serial composite mass to round-off.
    run!(solver, states; tfinal=1.0, nmax=2)
    mass = volume_integral(solver, [view(parent(Q), :, :, :, 1) for Q in states.patches])
    check("partitioned coupling: composite mass after two steps as serial",
          abs(mass - 1.9874998034111817) / 2, 1e-14)

    # One tile over every rank, up to eight: past five ranks some hold no
    # share of the chains' five components, so they receive no box and must
    # post no receive for one, while the subcycled Hermite endpoints move
    # through the same exchange.
    wave = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
        Numerics(n_global=(192, 1, 1),
                 amr=AMR(initial=BlockRegion((80, 0, 0), (32, 1, 1)), regrid_interval=0,
                         subcycle=true)))
    Qw = allocate_state(wave)
    initialize!(wave, Qw, (x, y, z) ->
        Prim(u=(0.5, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x)))
    check("partitioned coupling: one tile over every rank up to eight",
          abs(length(wave.levels[2].owners[1]) - min(np, 8)), 0.5)
    run!(wave, Qw; tfinal=1.0, nmax=3)
    sq = volume_integral(wave, [view(parent(Q), :, :, :, 1) .^ 2 for Q in Qw.patches])
    check("partitioned coupling: a tile over more ranks than components as serial",
          abs(sq - 6.40884901332379) / 6.4, 1e-14)
end

function test_tiled_level()
    section("tiled level: decomposed tiles and corner consensus")
    # Four 37×37 tiles meeting at a corner, coupled by the level's own
    # records. Ten steps: a 2-D case at np = 8 runs slowly on the workstation
    # (7 s/step, one patch or four; see CLUSTER.md on the hybrid cores),
    # and ten cross every interface many times over.
    function tiled_error(; subcycle,interface_flux=:ghost, transport=ConstantTransport())
        u0v, v0 = 0.4, 0.3
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3,
                    transport=transport, ic=nothing),
            Numerics(n_global=(48, 48, 1), art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     patch_interfaces=PatchInterfaces(flux=interface_flux),
                     amr=AMR(initial=BlockRegion((12, 12, 0), (24, 24, 1)),
                             regrid_interval=0, subcycle=subcycle, tile=12)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            Prim(u=(u0v, v0, 0), p=1.0, rho=1.0 + 0.1 * sin(x) * sin(y)))
        run!(solver, states; tfinal=0.5, nmax=10)
        e = 0.0
        for (ps, Q) in CL.eachpatch(solver, states)
            n = ps.decomp.n_local
            for j in 1:n[2], i in 1:n[1]
                I = padded_index(ps, i, j, 1)
                exact = 1.0 + 0.1 * sin(xcoord(ps, 1, i) - u0v * solver.t) *
                              sin(xcoord(ps, 2, j) - v0 * solver.t)
                e = max(e, abs(Q[I, 1] - exact))
            end
        end
        return gmax(e), solver.step, solver
    end
    et, nt, s4 = tiled_error(subcycle=false)
    lev = s4.levels[2]
    blocks = MPI.Allreduce(length(lev.patches), +, comm)
    check("tiled 2-D level: four tiles", abs(length(lev.transfers) - 4), 0.5)
    check("tiled 2-D level: each tile whole on one rank",
          all(length(o) == 1 for o in lev.owners) ? 0.0 : 1.0, 0.5)
    check("tiled 2-D level: one block per tile over the ranks", abs(blocks - 4), 0.5)
    check("tiled 2-D level: a rank holds its group's tiles only",
          all(lev.owners[t] == lev.group.ranks for t in lev.tiles) ? 0.0 : 1.0, 0.5)
    check("tiled 2-D wave error matches serial",
          abs(et - 1.1764478280440471e-11), 1e-14)
    check("tiled 2-D step count matches serial", abs(nt - 10), 0.5)
    # Subcycled, the ghost fluxes cross the tile faces and the coarse-fine
    # faces in both dimensions, between tiles on different ranks.
    ets, nts, _ = tiled_error(subcycle=true)
    check("subcycled tiled 2-D wave error matches serial",
          abs(ets - 3.3744118610457008e-11), 1e-14)
    check("subcycled tiled 2-D step count matches serial", abs(nts - 10), 0.5)
    # The molecular flux through the same faces: the tile faces' flux records
    # cross ranks, and the coarse-fine gradient ring carries each rank's
    # components.
    etv, ntv, _ = tiled_error(subcycle=true, interface_flux=:ghost,
                              transport=ConstantTransport(mu0=2e-2))
    check("subcycled tiled 2-D viscous wave, molecular ghost fluxes, matches serial",
          abs(etv - 8.7293825468126585e-7), 1e-12)
    check("subcycled tiled 2-D viscous step count, molecular ghost fluxes",
          abs(ntv - 10), 0.5)

    # Multi-tile corner consensus and diagonal ghosts, decomposed: the
    # dimension-phased sync must give every copy of the corner node the
    # mean of four tiles and fill a corner ghost from the diagonal tile,
    # across ranks, where a tile's neighbor may sit on another rank.
    let
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(48, 48, 1),
                     amr=AMR(initial=BlockRegion((12, 12, 0), (24, 24, 1)),
                             regrid_interval=0, tile=12)))
        states = allocate_state(solver)
        lev = solver.levels[2]
        for (li, ti) in zip(lev.patches, lev.tiles)
            fill!(parent(states.patches[li]), Float64(ti))
        end
        CL._sync_level_records!(solver, states.patches, lev)
        # Tiles A (12,12), B (24,12), C (12,24), D (24,24) of 37 fine nodes,
        # tile ids 1..4 in lattice order, carrying 1..4.
        function owned(t, i, j)
            k = findfirst(==(t), lev.tiles)
            k === nothing && return nothing
            p = lev.patches[k]
            dp = solver.patches[p].decomp
            pad = dp.n_halo_d
            inside = all(((g, o, n),) -> o + 1 - pad[1] <= g <= o + n,
                         ((i, dp.offset[1], dp.n_local[1]),
                          (j, dp.offset[2], dp.n_local[2])))
            inside || return nothing
            return states.patches[p][i - dp.offset[1] + pad[1], j - dp.offset[2] + pad[2],
                             1 + pad[3], 1]
        end
        dev(t, i, j, want) = (v = owned(t, i, j); v === nothing ? 0.0 : abs(v - want))
        e_corner = max(dev(1, 37, 37, 2.5), dev(2, 1, 37, 2.5),
                       dev(3, 37, 1, 2.5), dev(4, 1, 1, 2.5))
        e_ghost = max(dev(4, 0, 0, 1.0), dev(1, 38, 38, 4.0),
                      dev(2, 0, 38, 3.0), dev(3, 38, 0, 2.0))
        check("tiled corner: four copies at the mean", gmax(e_corner), 1e-15)
        check("tiled corner: diagonal ghosts from the diagonal tile",
              gmax(e_ghost), 1e-15)
    end

    # A snapshot of the decomposed tiles: every tile's blocks gathered from
    # its rank range into one snapshot on rank 0, root first.
    let
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(48, 48, 1), art=ArtificialProperties(enabled=false),
                     amr=AMR(initial=BlockRegion((12, 12, 0), (24, 24, 1)),
                             regrid_interval=0, tile=12)))
        states = allocate_state(solver)
        wave(x, y) = 1.0 + 0.1 * sin(x) * sin(y)
        initialize!(solver, states, (x, y, z) -> Prim(p=1.0, rho=wave(x, y)))
        snaps = field_snapshot(solver, states; fields=(:rho,))
        e = 0.0
        if rank == 0
            e = [s.level for s in snaps] == [0, 1, 1, 1, 1] ? 0.0 : 1.0
            e += count(snaps[1].covered) == 23 * 23 ? 0.0 : 1.0
            for s in snaps
                e = max(e, maximum(abs, s[:rho] .- [wave(x, y) for x in s.coords[1],
                                                    y in s.coords[2], z in 1:1]))
            end
        end
        check("tiled snapshot assembles every tile on rank 0", gmax(e), 1e-14)
    end

    # More tiles than ranks: twelve 25-node tiles of a 1-D level, each
    # admitting two ranks, so np = 2 and 4 cut the curve into runs of six
    # and three tiles and np = 8 gives four ranks two tiles and four one.
    # The wave error is the serial one to round-off.
    let
        u0 = 0.5
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(192, 1, 1), art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=BlockRegion((40, 0, 0), (96, 1, 1)), regrid_interval=0,
                             subcycle=true, tile=8)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(x)))
        run!(solver, states; tfinal=0.5, nmax=20)
        e = 0.0
        for (ps, Q) in CL.eachpatch(solver, states)
            for i in 1:ps.decomp.n_local[1]
                I = padded_index(ps, i, 1, 1)
                e = max(e, abs(Q[I, 1] - (1.0 + 0.2 * sin(xcoord(ps, 1, i) -
                                                          u0 * solver.t))))
            end
        end
        lev = solver.levels[2]
        held = length(lev.patches)
        most = MPI.Allreduce(held, max, comm)
        fewest = MPI.Allreduce(held, min, comm)
        check("twelve tiles: one rank per tile",
              all(length(o) == 1 for o in lev.owners) ? 0.0 : 1.0, 0.5)
        check("twelve tiles: tiles per rank balanced",
              abs(most - cld(12, np)) + abs(fewest - fld(12, np)), 0.5)
        check("twelve tiles: wave error matches serial",
              abs(gmax(e) - 8.3025808450543082e-12), 5e-15)
        check("twelve tiles: step count matches serial", abs(solver.step - 20), 0.5)
    end

    # A tiled regrid under per-tile ownership: the Sod shock's tile set moves
    # every five steps, a tile entering ahead of a survivor on the curve
    # shifts the survivor's owner range, and the survivor is rebuilt there
    # with its solution carried across. The tracked tile set and the time
    # reached are the serial ones. Measured serially: the set ends at
    # 168..184 after 41 steps, with the default hysteresis and without it
    # alike. The tag history
    # (`RegridSpec.created`, `checks`) is derived from the reduced flags and
    # must agree across ranks.
    let
        wall2 = (SlipWallBC(), SlipWallBC())
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=(wall2, per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(400, 1, 1), cfl=0.2,
                     filter=StateFilter(compact_filter(0.45), cfl=0.0),
                     amr=AMR(initial=BlockRegion((176, 0, 0), (16, 1, 1)), subcycle=true,
                             regrid_interval=5, tag_buffer=2, tile=8,
                             subdivide_lattice=false)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) -> x < 0.45 ?
            Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
            Prim(u=(0, 0, 0), p=0.1, rho=0.125))
        run!(solver, states; tfinal=0.03, nmax=41)
        regs = level_regions(solver, 1)
        offs = [r.offset[1] for r in regs]
        fin = all(all(isfinite, parent(Q)) for Q in states.patches)
        check("tiled regrid under ownership: finite composite state",
              fin ? 0.0 : 1.0, 0.5)
        check("tiled regrid under ownership: tile count tracks as serial (3)",
              abs(gmax(length(regs)) - 3), 0.5)
        check("tiled regrid under ownership: first tile tracks as serial (168)",
              abs(gmax(first(offs)) - 168), 0.5)
        check("tiled regrid under ownership: last tile tracks as serial (184)",
              abs(gmax(last(offs)) - 184), 0.5)
        check("tiled regrid under ownership: time reached matches serial",
              abs(gmax(solver.t) - 0.006478053648320283), 1e-13)
        spec = getfield(solver, :regrid)
        record = sort([(r.offset[1], c) for (r, c) in spec.created[1]])
        flat = Int[spec.checks; length(record);
                   [v for pair in record for v in pair]]
        # Identical on every rank, length included, when the elementwise
        # maximum and minimum over the ranks coincide.
        len = MPI.Allreduce(length(flat), max, comm)
        padded = [flat; fill(-1, len - length(flat))]
        spread = MPI.Allreduce(padded, max, comm) .- MPI.Allreduce(padded, min, comm)
        check("tag history: checks and creation record agree across ranks",
              maximum(abs.(spread)), 0.5)
        check("tag history: eight checks over forty steps", abs(spec.checks - 8), 0.5)
        # The root's covered mask follows the regrids: it equals a fresh
        # fill from the current tile set on every rank, and the composite
        # volume stays exact across them.
        root = solver.patches[1]
        kept = copy(root.amr.covered)
        CL._fill_covered!(root, level_regions(solver, 1))
        check("covered mask: rewritten by the regrids on every rank",
              gsum(count(kept .!= root.amr.covered)), 0.5)
        check("covered mask: composite volume exact after the regrids",
              gmax(abs(domain_volume(solver) - 1)), 1e-13)
    end

    # Stored ownership and rebalancing on the same regrids. A callback at
    # every step compares each surviving tile's owner range with the one it
    # had (the regrid runs at the head of the next step, so consecutive
    # readings straddle it): with rebalancing off no survivor moves, and
    # with it on at a threshold of one, which any measured spread of busy
    # time exceeds, the level is repartitioned on the measured weights, a
    # survivor moves, and the tracked tile set and time are still the
    # serial ones to round-off. Fresh coefficients after every regrid check
    # set the CFL sequence used by these serial baselines. The hold band is
    # off here: the block tests
    # ownership, and a survivor moving at every rank count rests on the
    # tile ahead of the shock entering and leaving the set, which the band
    # damps.
    function regrid_track(; rebalance, persist)
        moved = Ref(0)
        prev = Dict{BlockRegion,UnitRange{Int}}()
        watch = Callback(EveryStep(1), (s, _) -> begin
            lev = getfield(s, :levels)[2]
            for (lt, o) in zip(lev.transfers, lev.owners)
                haskey(prev, lt.region) && prev[lt.region] != o && (moved[] += 1)
            end
            empty!(prev)
            for (lt, o) in zip(lev.transfers, lev.owners)
                prev[lt.region] = o
            end
            false
        end)
        wall2 = (SlipWallBC(), SlipWallBC())
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                    bcs=(wall2, per3[2], per3[3]), ic=nothing),
            Numerics(n_global=(400, 1, 1), cfl=0.2,
                     filter=StateFilter(compact_filter(0.45), cfl=0.0),
                     amr=AMR(initial=BlockRegion((176, 0, 0), (16, 1, 1)), subcycle=true,
                             regrid_interval=5, tag_buffer=2, untag_ratio=1, tile=8,
                             rebalance=rebalance, rebalance_persist=persist,
                             subdivide_lattice=false)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) -> x < 0.45 ?
            Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
            Prim(u=(0, 0, 0), p=0.1, rho=0.125))
        run!(solver, states; tfinal=0.03, nmax=41, callback=watch)
        offs = [r.offset[1] for r in level_regions(solver, 1)]
        return solver, offs, moved[]
    end
    let
        _, offs, moved = regrid_track(rebalance=0, persist=1)
        check("stored ownership: no survivor moved with rebalance off",
              gmax(moved), 0.5)
        check("stored ownership: tile set tracks as serial",
              abs(gmax(first(offs)) - 168) + abs(gmax(last(offs)) - 184), 0.5)
        # The same run with the migration audit on: every moved tile is also
        # carried through the replicated gather, and the migrated state
        # must equal that reference at every slot.
        CL.MIGRATION_AUDIT[] = true
        CL.MIGRATION_AUDIT_RESULT[] = (tiles=0, mismatches=0)
        solver, offs, moved = regrid_track(rebalance=1.0, persist=1)
        CL.MIGRATION_AUDIT[] = false
        audit = CL.MIGRATION_AUDIT_RESULT[]
        spec = getfield(solver, :regrid)
        # At np = 2 the level is two runs of the curve cut by cumulative
        # measured weight, and whether that cut moves a survivor depends on
        # the busy-time spread the run happened to measure (observed both
        # ways on the workstation); from np = 4 a fresh tile entering ahead
        # of the survivors shifts them under any weights.
        check("rebalance on: a measured spread repartitioned the level",
              np >= 4 && gmax(moved) == 0 ? 1.0 : 0.0, 0.5)
        check("migration: a moved tile was audited against the gather",
              np >= 4 && gsum(audit.tiles) == 0 ? 1.0 : 0.0, 0.5)
        check("migration: migrated state equals the gathered carry bitwise",
              gsum(audit.mismatches), 0.5)
        check("rebalance on: tile count tracks as serial (3)",
              abs(gmax(length(offs)) - 3), 0.5)
        check("rebalance on: first tile tracks as serial (168)",
              abs(gmax(first(offs)) - 168), 0.5)
        check("rebalance on: last tile tracks as serial (184)",
              abs(gmax(last(offs)) - 184), 0.5)
        check("rebalance on: time reached matches serial",
              abs(gmax(solver.t) - 0.006476645618274905), 1e-13)
        check("rebalance on: max/mean busy time measured",
              isfinite(spec.imbalance) && spec.imbalance >= 1 ? 0.0 : 1.0, 0.5)
        # Hysteresis, on synthetic per-rank busy times: rank r reports
        # 1 + 3r seconds for an interval, a spread above the threshold at
        # every rank count here. With `persist = 2` the first check holds,
        # the second consecutive one repartitions and hands back the
        # Allgathered vector, and a balanced interval resets the streak.
        spec.rebalance = 1.5
        spec.persist = 2
        spec.streak = 0
        spec.wall_mark = solver.wall_total
        spec.wait_mark = solver.wait_total
        spec.wall_regrid = 0.0
        solver.wall_total += 1.0 + 3rank
        r1 = CL._rebalance_due!(solver, spec)
        held = r1 === nothing && spec.streak == 1 && spec.imbalance > 1.5
        solver.wall_total += 1.0 + 3rank
        r2 = CL._rebalance_due!(solver, spec)
        solver.wall_total += 1.0
        r3 = CL._rebalance_due!(solver, spec)
        check("hysteresis: the first imbalanced check holds",
              held ? 0.0 : 1.0, 0.5)
        check("hysteresis: the second consecutive check repartitions",
              r2 === nothing ? 1.0 :
              maximum(abs.(r2 .- [1.0 + 3r for r in 0:np-1])), 1e-9)
        check("hysteresis: a balanced interval resets the streak",
              r3 === nothing && spec.streak == 0 ? 0.0 : 1.0, 0.5)
        check("hysteresis: the balanced interval measures max/mean = 1",
              abs(spec.imbalance - 1.0), 1e-12)
    end
end

# ---------------------------------------------------------------------------
# Per-level rank subsets. A refined level too small to give every rank nine
# fine points is owned by a prefix of the ranks instead of failing at setup,
# and its level-local collectives run on that subset's communicator.
# Three fine extents pin the sizing: 22 nodes admit two ranks and no more
# (22 ÷ 3 = 7), 10 admit one, and 70 admit seven. Agreement with the serial
# answer is round-off, not bitwise, because a subset is a different
# decomposition of the fine patch; that is the tier the two-patch and
# distributed-refinement phases already use.
# ---------------------------------------------------------------------------
function test_level_subset()
    section("per-level rank subsets")
    N = 192
    u0 = 0.5
    # A refined region of `ext` coarse nodes at the middle of the domain,
    # advanced twenty steps. With `landing`, an `EveryTime` trigger shortens
    # the steps approaching each scheduled instant; a rank outside the
    # level's subset holds no state from which that shortening follows, so
    # its agreement rests on the global dt reduction alone. `fires` counts
    # the firings this rank saw. `e_fine` is the error over the refined
    # level alone, which the coarse-fine faces set.
    function wave(ext; subcycle=false, landing=false, k=1,)
        solver = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 1.0), (0.0, 1.0)), bcs=per3, ic=nothing),
            Numerics(n_global=(N, 1, 1), art=ArtificialProperties(enabled=false),
                     filter=StateFilter(interval=0),
                     amr=AMR(initial=BlockRegion((N ÷ 2, 0, 0), (ext, 1, 1)),
                             regrid_interval=0, subcycle=subcycle)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) ->
            Prim(u=(u0, 0, 0), p=1.0, rho=1.0 + 0.2 * sin(k * x)))
        fires = Ref(0)
        cb = landing ?
             Callback(EveryTime(0.02), (_, _) -> (fires[] += 1; nothing)) : nothing
        run!(solver, states; tfinal=0.5, nmax=20, callback=cb)
        e = e_fine = 0.0
        for (ps, Q) in CL.eachpatch(solver, states)
            for i in 1:ps.decomp.n_local[1]
                I = padded_index(ps, i, 1, 1)
                ei = abs(Q[I, 1] - (1.0 + 0.2 * sin(k * (xcoord(ps, 1, i) -
                                                          u0 * solver.t))))
                e = max(e, ei)
                ps.h[1] < 2π / N / 2 && (e_fine = max(e_fine, ei))
            end
        end
        owners = MPI.Allreduce(Int(solver.levels[2].level_comm.owned), +, comm)
        return solver, gmax(e), owners, fires[], gmax(e_fine)
    end
    spread(x) = MPI.Allreduce(Float64(x), max, comm) -
                MPI.Allreduce(Float64(x), min, comm)

    # Twenty-two fine nodes: two ranks, so np = 2 owns the level whole and
    # np = 4 and 8 own it on a prefix. The static rows carry the
    # wavenumber-8 wave, as in the distributed-refinement phase: at
    # wavenumber 1 the ghost fluxes leave an error of a few ulps. At
    # wavenumber 8 the refined level's error, 6.6e-9 to 7.7e-9, peaks at a
    # coarse-fine face, and the decomposed runs reproduce it to 1.5e-15 at
    # np = 2, 4 and 8.
    s8, _, own8, _, e8 = wave(8; k=8)
    check("22-node level takes two ranks", abs(own8 - min(np, 2)), 0.5)
    check("subset-owned level: refined-level wave error matches serial",
          abs(e8 - 6.554074283293687e-9), 1e-14)
    check("subset-owned level: step count matches serial", abs(s8.step - 20), 0.5)
    s8s, e8s, _, _ = wave(8; subcycle=true)
    check("subcycled subset-owned level: wave error matches serial",
          abs(e8s - 1.2722045639179669e-12), 5e-15)
    check("subcycled subset-owned level: step count matches serial",
          abs(s8s.step - 20), 0.5)

    # Ten fine nodes cannot be split at all: without a one-rank subset this
    # configuration admits no process grid at any np > 1.
    s4, _, own4, _, e4 = wave(4; k=8)
    check("10-node level takes one rank", abs(own4 - 1), 0.5)
    check("one-rank level: refined-level wave error matches serial",
          abs(e4 - 7.686864256228887e-9), 1e-14)
    check("one-rank level: step count matches serial", abs(s4.step - 20), 0.5)
    # A rank outside the level's subset runs no part of it, yet agrees on
    # the step sequence and on every trigger: dt comes from one reduction
    # over the whole run, and `t` and `step` advance from it. The
    # scheduled trigger also shortens dt on its approach, so a rank that
    # disagreed anywhere would land on a different instant.
    sL, _, _, firesL, eL = wave(4; landing=true, k=8)
    check("landed one-rank level: refined-level wave error matches serial",
          abs(eL - 6.981902833658182e-9), 1e-14)
    check("one-rank level: time agrees on every rank",
          spread(sL.t), 1e-14 * max(sL.t, 1e-30))
    check("one-rank level: step agrees on every rank", spread(sL.step), 0.5)
    check("one-rank level: callback fired equally on every rank",
          spread(firesL), 0.5)
    check("one-rank level: callback fired as serially",
          abs(firesL - 3), 0.5)

    # Regridding under subset ownership: the region grows from 8 coarse nodes
    # (22 fine, two ranks) to 24 (70 fine, seven), so the subset is resized
    # and the communicator it replaces is freed rather than left to the
    # garbage collector, as a dropped `Decomp` is.
    wall2 = (SlipWallBC(), SlipWallBC())
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per3[2], per3[3]),
                ic=nothing),
        Numerics(n_global=(400, 1, 1), cfl=0.2,
                 filter=StateFilter(compact_filter(0.45), cfl=0.0),
                 amr=AMR(initial=BlockRegion((180, 0, 0), (8, 1, 1)), subcycle=true,
                         regrid_interval=20, tag_buffer=2, subdivide_lattice=false)))
    lc0 = solver.levels[2].level_comm
    owners0 = MPI.Allreduce(Int(lc0.owned), +, comm)
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> x < 0.45 ?
        Prim(u=(0, 0, 0), p=1.0, rho=1.0) :
        Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    # Forty-one steps: two regrids, the second of which resizes the subset.
    run!(solver, states; tfinal=0.03, nmax=41)
    region = CL.refined_region(solver)
    owners1 = MPI.Allreduce(Int(solver.levels[2].level_comm.owned), +, comm)
    fin = all(all(isfinite, parent(Q)) for Q in states.patches)
    check("regrid: setup subset is two ranks", abs(owners0 - min(np, 2)), 0.5)
    check("regrid: subset recomputed for the grown region",
          abs(owners1 - min(np, 7)), 0.5)
    check("regrid: the replaced level communicator was freed",
          (!lc0.scoped || lc0.comm == MPI.COMM_NULL) ? 0.0 : 1.0, 0.5)
    check("regrid under subsets: finite composite state", fin ? 0.0 : 1.0, 0.5)
    # Serial values under the default hold band, with the root's sensors
    # dropped where the level's restriction overwrites it: the times reached
    # here and in the regridded Sod case are those of that rule.
    check("regrid under subsets: region offset tracks as serial (171)",
          abs(gmax(region.offset[1]) - 171), 0.5)
    check("regrid under subsets: region extent tracks as serial (24)",
          abs(gmax(region.extent[1]) - 24), 0.5)
    check("regrid under subsets: time reached matches serial",
          abs(gmax(solver.t) - 0.006980662716575952), 1e-13)
end

# ---------------------------------------------------------------------------
# Covered masks under decomposition. The composite quadrature of a linear
# field on a wall-bounded box is exact, so the masked volume, integral and
# plane profile are known to round-off on every layout, and every rank
# returns the same number: the reduction runs over the root communicator,
# which a rank holding no piece of the refined level enters with its root
# block alone. The small region admits one rank, so at np > 1 some ranks
# are outside the child's subset. No stepping: setup and the reductions.
# ---------------------------------------------------------------------------
function test_covered_masks()
    section("covered masks: composite quadrature under decomposition")
    wall = (SlipWallBC(), SlipWallBC())
    N = 24
    lin(x, y, z) = x + 2y + 3z
    spread(x) = MPI.Allreduce(Float64(x), max, comm) -
                MPI.Allreduce(Float64(x), min, comm)
    small = BlockRegion((8, 8, 8), (4, 4, 4))
    wide = BlockRegion((6, 6, 6), (8, 8, 8))
    for (label, tile, refine) in (("small box", 0, small),
                                  ("tile nest", 4, wide),
                                  ("three levels", 0, [wide, BlockRegion((24, 24, 24), (8, 8, 8))]))
        solver = build_solver(
            Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall, wall, wall),
                    ic=nothing),
            Numerics(n_global=(N, N, N),
                     amr=AMR(initial=refine, regrid_interval=0, tile=tile)))
        states = allocate_state(solver)
        initialize!(solver, states, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
        fs = map(CL.eachpatch(solver, states)) do (ps, Q)
            a = similar(ps.rho)
            d = ps.decomp
            for k in 1:d.n_local[3], j in 1:d.n_local[2], i in 1:d.n_local[1]
                a[padded_index(ps, i, j, k)] = lin(xcoord(ps, 1, i), xcoord(ps, 2, j),
                                                   xcoord(ps, 3, k))
            end
            a
        end
        V = domain_volume(solver)
        I = volume_integral(solver, fs)
        prof = plane_profile(solver, fs, 1)
        xs = profile_coordinate(solver, 1)
        outside = MPI.Allreduce(Int(!solver.levels[2].level_comm.owned), +, comm)
        check("$label: composite volume exact", gmax(abs(V - 1)), 1e-13)
        check("$label: composite linear integral exact", gmax(abs(I - 3)), 1e-13)
        check("$label: composite profile exact",
              gmax(maximum(abs.(prof .- (xs .+ 2.5)))), 1e-12)
        check("$label: every rank returns the same numbers",
              spread(V) + spread(I) + spread(sum(prof)), 1e-15)
        label == "small box" &&
            check("small box: ranks outside the child's subset at np > 1",
                  abs(outside - (np - 1)), 0.5)
        composite_samples(label, solver, states)
    end
    # The same after regrids have moved the tiles of a one-dimensional run.
    per = (PeriodicBC(), PeriodicBC())
    solver = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall, per, per),
                ic=nothing),
        Numerics(n_global=(201, 1, 1), cfl=0.2,
                 amr=AMR(initial=BlockRegion((85, 0, 0), (31, 1, 1)), subcycle=true,
                         regrid_interval=5, tile=8, subdivide_lattice=false)))
    states = allocate_state(solver)
    initialize!(solver, states, (x, y, z) -> x < 0.5 ?
        Prim(u=(0, 0, 0), p=1.0, rho=1.0) : Prim(u=(0, 0, 0), p=0.1, rho=0.125))
    before = level_regions(solver, 1)
    run!(solver, states; tfinal=1.0, nmax=20)
    check("regridded: the tiles moved", Float64(level_regions(solver, 1) == before), 0.5)
    composite_samples("regridded", solver, states)
end

# The composite `line_sample` and `field_slice` of a hierarchy: a linear density
# comes back exactly on every rank, and a state scaled by 1 + level on each patch
# reads the finest level holding each root node, found from the level regions.
function composite_samples(label, solver, states)
    N = solver.n_global
    spread(x) = MPI.Allreduce(Float64(x), max, comm) -
                MPI.Allreduce(Float64(x), min, comm)
    lin(x, y, z) = Prim(u=(0, 0, 0), p=1.0, rho=1 + x + 2y + 3z)
    initialize!(solver, states, lin)
    at = ntuple(d -> min(10, N[d]), 3)
    xs = ntuple(d -> profile_coordinate(solver, d), 3)
    x, v = line_sample(solver, states, :rho; dim=1, index=(at[2], at[3]))
    check("$label: composite line sample exact",
          gmax(maximum(abs.(v .- (1 .+ x .+ 2xs[2][at[2]] .+ 3xs[3][at[3]])))), 1e-13)
    check("$label: every rank samples the same line", spread(sum(v)), 1e-15)
    slice = field_slice(solver, states, :rho; normal=3, index=at[3])
    err = 0.0
    if MPI.Comm_rank(comm) == 0
        x1, x2, plane = slice
        err = maximum(abs(plane[i, j] - (1 + x1[i] + 2x2[j] + 3xs[3][at[3]]))
                      for i in 1:N[1], j in 1:N[2])
    else
        err = Float64(slice !== nothing)
    end
    check("$label: composite slice exact on rank 0, nothing elsewhere", gmax(err), 1e-13)
    # Level ℓ holds root node g when its node (g − 1)·3^(ℓ−1) + 1 in the parent
    # level's node space lies in one of its regions.
    function finest(g)
        top = 0
        for l in 1:CL.nlevels(solver)-1
            m = (g .- 1) .* 3^(l - 1) .+ 1
            any(r -> all(r.offset[d] < m[d] <= r.offset[d] + r.extent[d] for d in 1:3),
                level_regions(solver, l)) && (top = l)
        end
        return top
    end
    for (ps, Q) in CL.eachpatch(solver, states)
        parent(Q) .*= 1 + ps.patch.level
    end
    _, marked = line_sample(solver, states, :rho; dim=1, index=(at[2], at[3]))
    expect = [1 + finest((g, at[2], at[3])) for g in 1:N[1]]
    check("$label: each root node reads the finest level holding it",
          gmax(maximum(abs.(marked ./ v .- expect))), 1e-14)
    threw = try
        line_sample(solver, states, :rho; index=(0, 1))
        false
    catch e
        e isa ArgumentError
    end
    check("$label: an index out of range throws on every rank",
          MPI.Allreduce(Int(!threw), +, comm), 0.5)
    return nothing
end
