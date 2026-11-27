# The MPI suite's phases of the field writers, shared planes, slices, line samples,
# per-rank checkpoints and the wall-time limit. Included by test/mpi_tests.jl, which
# loads the package, defines `check`, `section` and the other helpers used here, and
# runs the phases; run that file, not this one.

# ---------------------------------------------------------------------------
# VTK pieces read back. A parallel container's reader assembles the whole
# extent from the cells of its pieces, so neighboring pieces must share their
# boundary plane; pieces that abut point to point leave a one-cell slab between
# them in no piece, and ParaView then refuses the file. The checks below read
# every piece of a dump in the writer's appended raw layout, place it on the
# whole output grid, and test the cell cover and the shared planes.
# ---------------------------------------------------------------------------

const VTK_TYPES = Dict("Float64" => Float64, "Float32" => Float32, "UInt8" => UInt8)

"Extent and arrays of one piece: `(extent, Dict(name => (ncomponents, values)))`."
function read_vtk_piece(path)
    bytes = read(path)
    mark = findfirst(codeunits("<AppendedData encoding=\"raw\">\n_"), bytes)
    head = String(bytes[1:first(mark)-1])
    ext = parse.(Int, split(match(r"<Piece Extent=\"([^\"]+)\"", head).captures[1]))
    arrays = Dict{String,Tuple{Int,Vector{Float64}}}()
    pat = r"<DataArray type=\"(\w+)\"(?: Name=\"([^\"]*)\")?" *
          r"(?: NumberOfComponents=\"(\d+)\")? format=\"appended\" offset=\"(\d+)\""
    for m in eachmatch(pat, head)
        T = VTK_TYPES[m.captures[1]]
        nc = m.captures[3] === nothing ? 1 : parse(Int, m.captures[3])
        at = last(mark) + parse(Int, m.captures[4])
        nbytes = Int(reinterpret(UInt64, bytes[at+1:at+8])[1])
        arrays[something(m.captures[2], "Points")] =
            (nc, Float64.(reinterpret(T, bytes[at+9:at+8+nbytes])))
    end
    return ext, arrays
end

"""
The pieces `files` placed on the whole output grid `whole` (a lo/hi 6-vector):
`missing_points` counts the grid points no piece holds, `missing_cells` the
cells no piece's cell extent covers (a collapsed dimension counting one cell),
and `mismatch` is the largest difference between two pieces at a shared point.
`fields` maps each name to its assembled `(nc, n1, n2, n3)` array, and the
rectilinear coordinates `x`, `y`, `z` to vectors.
"""
function assemble_pieces(files, whole)
    n = ntuple(d -> whole[2d] - whole[2d-1] + 1, 3)
    cells = falses(ntuple(d -> max(n[d] - 1, 1), 3))
    fields = Dict{String,Array{Float64}}()
    mismatch = 0.0
    for f in files
        e, arrays = read_vtk_piece(f)
        lo = ntuple(d -> e[2d-1] - whole[2d-1], 3)
        m = ntuple(d -> e[2d] - e[2d-1] + 1, 3)
        cells[ntuple(d -> n[d] == 1 ? (1:1) : (lo[d]+1:lo[d]+m[d]-1), 3)...] .= true
        span = ntuple(d -> lo[d]+1:lo[d]+m[d], 3)
        for (name, (nc, v)) in arrays
            d = findfirst(==(name), ("x", "y", "z"))
            A = get!(() -> fill(NaN, d === nothing ? (nc, n...) : (n[d],)), fields, name)
            idx = d === nothing ? (1:nc, span...) : (span[d],)
            vals = reshape(v, length.(idx))
            old = A[idx...]
            known = .!isnan.(old)
            any(known) &&
                (mismatch = max(mismatch, maximum(abs.(old[known] .- vals[known]))))
            A[idx...] = vals
        end
    end
    missing_points = maximum(count(isnan, A) for A in values(fields))
    return (; missing_points, missing_cells=count(!, cells), mismatch, fields)
end

"`assemble_pieces` over the pieces a `.pvtr` or `.pvts` container lists."
function assemble_container(path)
    txt = read(path, String)
    whole = parse.(Int, split(match(r"WholeExtent=\"([^\"]+)\"", txt).captures[1]))
    files = [joinpath(dirname(path), m.captures[1])
             for m in eachmatch(r"Source=\"([^\"]+)\"", txt)]
    return assemble_pieces(files, whole)
end

"""
The `FieldData` record of a VTK file, read from its ASCII arrays: a string
array as the strings its zero-terminated codes spell, an `Int64` or `Float64`
array as numbers.
"""
function read_vtk_record(path)
    raw = read(path)
    stop = findfirst(codeunits("<AppendedData"), raw)
    head = String(raw[1:(stop === nothing ? length(raw) : first(stop) - 1)])
    rec = Dict{String,Vector}()
    block = match(r"<FieldData>(.*?)</FieldData>"s, head)
    block === nothing && return rec
    pat = r"<(?:Data)?Array type=\"(\w+)\" Name=\"([^\"]+)\" " *
          r"NumberOfTuples=\"\d+\" format=\"ascii\">([^<]*)<"
    for m in eachmatch(pat, block.captures[1])
        body = split(m.captures[3])
        if m.captures[1] == "String"
            # Split the codes at each terminating zero.
            codes = parse.(UInt8, body)
            ends = findall(==(0x00), codes)
            starts = [1; ends[1:end-1] .+ 1]
            rec[m.captures[2]] = [String(codes[a:b-1]) for (a, b) in zip(starts, ends)]
        else
            rec[m.captures[2]] = parse.(m.captures[1] == "Int64" ? Int64 : Float64, body)
        end
    end
    return rec
end

"""
Check on every rank that the container at `path` and every piece it lists
carry the same record, and that the record holds each entry of `expect`.
"""
function check_record(label, path, expect)
    MPI.Barrier(comm)
    bad = 0
    if rank == 0
        rec = read_vtk_record(path)
        files = [joinpath(dirname(path), m.captures[1])
                 for m in eachmatch(r"Source=\"([^\"]+)\"", read(path, String))]
        bad += count(f -> read_vtk_record(f) != rec, files)
        for (key, value) in expect
            got = get(rec, key, nothing)
            ok = got !== nothing && length(got) == length(value) &&
                 (eltype(value) <: AbstractString ? got == value : isapprox(got, value))
            bad += !ok
        end
    end
    check("$label: one record, the expected one", gsum(bad), 0.5)
end

"""
Check on every rank that the pieces rank 0 reads from `assemble()` cover the
whole output grid, cells and points, and agree bitwise on every shared plane.
`expect(fields)`, when given, returns the largest error of the assembled fields
against their analytic values, checked at `tol`.
"""
function check_tiling(label, assemble; expect=nothing, tol=1e-6)
    MPI.Barrier(comm)
    holes, gaps, mismatch, err = 0, 0, 0.0, 0.0
    if rank == 0
        a = assemble()
        holes, gaps, mismatch = a.missing_cells, a.missing_points, a.mismatch
        err = expect === nothing ? 0.0 : expect(a.fields)
    end
    check("$label: every cell lies in a piece", gsum(holes), 0.5)
    check("$label: every point lies in a piece", gsum(gaps), 0.5)
    check("$label: shared planes agree bitwise", gsum(mismatch), 1e-300)
    expect === nothing || check("$label: values at their coordinates", gsum(err), tol)
end

# ρ = 1 + x on a rectilinear dump, read against its own coordinate vector.
rho_of_x(f) = maximum(abs(f["rho"][1, i, j, k] - (1 + f["x"][i]))
                      for i in axes(f["rho"], 2), j in axes(f["rho"], 3),
                          k in axes(f["rho"], 4))

# ---------------------------------------------------------------------------
# 9. FieldWriter under decomposition. save_vtk is collective, since it
#    Allgathers the piece extents, allowing rank 0 to write the container, and
#    the serial suite does not exercise this path: one rank writes one piece
#    there, leaving the per-rank naming and the container's piece list untested.
#    The writer adds directory creation and a rank-0 collection file, both of
#    which can succeed in serial and deadlock at high rank counts.
# ---------------------------------------------------------------------------
function test_field_writer()
    section("FieldWriter under decomposition")
    solver, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                              bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                              ic=(x, y, z) -> Prim(u=(0.2, 0, 0),
                                                   p=1 + 0.1exp(-40(x - 0.5)^2),
                                                   rho=1.0)),
                      Numerics(n_global=(SPLITN, 16, 16),
                               execution=Execution(dims=splitdims(1))))
    writer = FieldWriter(joinpath("mpi_frames", "field"))
    # The piece and container counts below assume four frames, the initial
    # state and three instants; the run is only
    # as long as it takes to produce them, since
    # every check here is on the files rather than on the state in them. The
    # interval is a little over one CFL step, so the frames still land on
    # separate steps. See the run-length note in test_callback_consistency.
    run!(solver, Q; tfinal=0.006, callback=Callback(EveryTime(0.002), writer))

    # Every rank must have written the same number of frames at the same times.
    # A rank that skipped one would have desynchronized the internal Allgather.
    check("frame count agrees on every rank",
          MPI.Allreduce(writer.index, max, comm) -
          MPI.Allreduce(writer.index, min, comm), 0.5)
    tspread = 0.0
    # Four frames: the initial state and three instants.
    for m in 1:min(writer.index, 4)
        tspread = max(tspread, abs(writer.times[m] - 0.002(m - 1)),
                      MPI.Allreduce(writer.times[m], max, comm) -
                      MPI.Allreduce(writer.times[m], min, comm))
    end
    check("frame times exact and identical on every rank", tspread, 1e-14)

    # One piece per rank per frame, one container per frame, one collection.
    MPI.Barrier(comm)
    counted = rank == 0 ? length(filter(endswith(".vtr"), readdir("mpi_frames"))) : 0
    check("one .vtr piece per rank per frame", abs(gsum(counted) - 4 * np), 0.5)
    containers = rank == 0 ? length(filter(endswith(".pvtr"), readdir("mpi_frames"))) : 0
    check("one .pvtr container per frame", abs(gsum(containers) - 4), 0.5)
    pieces = rank == 0 ?
        count("<Piece", read(joinpath("mpi_frames", "field_0000.pvtr"), String)) : 0
    check("container lists every rank's piece", abs(gsum(pieces) - np), 0.5)
    sets = rank == 0 ?
        count("<DataSet", read(joinpath("mpi_frames", "field.pvd"), String)) : 0
    check("collection written once, by rank 0", abs(gsum(sets) - 4), 0.5)
    check_tiling("frame", () -> assemble_container(joinpath("mpi_frames",
                                                            "field_0000.pvtr")))
    walls = ["open", "open", "periodic", "periodic", "periodic", "periodic"]
    check_record("frame", joinpath("mpi_frames", "field_0002.pvtr"),
                 Dict("TIME" => [writer.times[3]], "boundary" => walls,
                      "n_global" => [SPLITN, 16, 16],
                      "domain" => [0.0, 1.0, 0.0, 0.25, 0.0, 0.25]))

    MPI.Barrier(comm)
    rank == 0 && rm("mpi_frames"; recursive=true)
    MPI.Barrier(comm)

    # Strided output decomposed. The retained points are defined on the GLOBAL
    # index, so the pieces sample one common lattice and tile the coarse grid
    # exactly. Striding each rank's block from its own first point would produce
    # files that appear correct but tile incorrectly, with a doubled or missing
    # plane at a rank boundary. Serial runs cannot detect this, having only one
    # piece. The shared plane lies a stride past a rank's last retained point,
    # which at stride 5 can be past the halo: on two ranks rank 0 retains its
    # last point, 36, and writes global point 41. The last rank of the
    # periodic x writes no plane across the seam.
    strided, Qs = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                                bcs=(per3[1], per3[2], per3[3]),
                                ic=(x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                                     rho=1 + x)),
                        Numerics(n_global=(SPLITN, 16, 16),
                                 execution=Execution(dims=splitdims(1))))
    for sd in (2, 3, 5)
        save_vtk(strided, Qs, "mpi_stride"; fields=(:rho,), stride=sd)
        MPI.Barrier(comm)
        nkeep = (length(1:sd:SPLITN), length(1:sd:16), length(1:sd:16))
        inside, whole_ok = 0, 0
        if rank == 0
            txt = read("mpi_stride.pvtr", String)
            whole_ok = occursin("WholeExtent=\"0 $(nkeep[1]-1) 0 $(nkeep[2]-1) " *
                                "0 $(nkeep[3]-1)\"", txt) ? 1 : 0
            inside = 1
            for m in eachmatch(r"<Piece Extent=\"([^\"]+)\"", txt)
                e = parse.(Int, split(m.captures[1]))
                for d in 1:3
                    (0 <= e[2d-1] <= e[2d] <= nkeep[d] - 1) || (inside = 0)
                end
            end
        end
        # Every coarse cell in a piece, the shared points equal, and ρ = 1 + x
        # at every coordinate: a plane taken from the wrong point fails the
        # last.
        check_tiling("stride $sd", () -> assemble_container("mpi_stride.pvtr");
                     expect=rho_of_x)
        check("stride $sd: every piece within the whole extent",
              abs(gsum(inside) - 1), 0.5)
        check("stride $sd: whole extent is the coarse grid",
              abs(gsum(whole_ok) - 1), 0.5)
        # The record of every piece names the full grid, the grid as written
        # and the stride, from which the period follows: n_global spacings of
        # the written spacing over the stride.
        check_record("stride $sd", "mpi_stride.pvtr",
                     Dict("n_global" => [SPLITN, 16, 16], "stride" => [sd, sd, sd],
                          "n_sampled" => collect(nkeep),
                          "boundary" => fill("periodic", 6)))
        period_err = 0.0
        if rank == 0
            rec = read_vtk_record("mpi_stride.pvtr")
            x = assemble_container("mpi_stride.pvtr").fields["x"]
            period_err = abs(rec["n_global"][1] * (x[2] - x[1]) / rec["stride"][1] -
                             (rec["domain"][2] - rec["domain"][1]))
        end
        check("stride $sd: the period from the record", gmax(period_err), 1e-12)
        MPI.Barrier(comm)
        rank == 0 && foreach(rm, filter(startswith("mpi_stride"), readdir()))
        MPI.Barrier(comm)
    end

    # A stride coarse enough to leave a rank with no points gives that rank no
    # piece to write, and must fail on EVERY rank. One rank raising while the
    # others proceed into the Allgather produces a hang rather than an error.
    too_coarse = 2 * (SPLITN ÷ np) + 1
    threw = try
        save_vtk(strided, Qs, "mpi_toocoarse"; fields=(:rho,), stride=too_coarse)
        0
    catch e
        e isa ArgumentError ? 1 : 0
    end
    check("an over-coarse stride throws on every rank", abs(gsum(threw) - np), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_toocoarse"), readdir()))
    MPI.Barrier(comm)

    # The curvilinear path decomposed, split along the resolved angle. It writes
    # a different piece type and container, so no check above covers it, and
    # splitting the angle is the case in which the piece extents must agree
    # around a periodic wrap.
    cyl, Qc = setup(Problem(domain=((0.5, 1.5), (0.0, 2π), (0.0, 1.0)),
                            metric=CylindricalMetric(),
                            bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                            ic=(r, θ, z) -> Prim(u=(0.0, 0.5, 0.0), p=1.0, rho=1.0)),
                    Numerics(n_global=(12, SPLITN, 12),
                             execution=Execution(dims=splitdims(2))))
    check("resolved angle selects the structured container",
          container_extension(cyl) == ".pvts" ? 0.0 : 1.0, 0.5)
    save_vtk(cyl, Qc, "mpi_annulus")
    MPI.Barrier(comm)
    mine = isfile("mpi_annulus.r" * lpad(rank, 4, '0') * ".vts") ? 1 : 0
    check("every rank wrote its .vts piece", abs(gsum(mine) - np), 0.5)
    vts = rank == 0 ? read("mpi_annulus.pvts", String) : ""
    kind = rank == 0 ? (occursin("PStructuredGrid", vts) ? 1 : 0) : 0
    check("container is a PStructuredGrid", abs(gsum(kind) - 1), 0.5)
    check("container lists every angular piece",
          abs(gsum(rank == 0 ? count("<Piece", vts) : 0) - np), 0.5)
    check_tiling("annulus", () -> assemble_container("mpi_annulus.pvts"))
    check_record("annulus", "mpi_annulus.pvts",
                 Dict("metric" => ["cylindrical"],
                      "boundary" => ["open", "open", "periodic", "periodic",
                                     "periodic", "periodic"],
                      "domain" => [0.5, 1.5, 0.0, 2π, 0.0, 1.0]))
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_annulus"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# 10b. Shared piece planes on a two-dimensional process grid. A rank's piece
#      ends on the first point of its high neighbor in each split dimension,
#      and the corner point of a 2 x 2 block of ranks reaches the rank below
#      and to the left through the neighbor that took it first, so a plane
#      exchange done in the wrong order or from unextended payloads leaves a
#      wrong or missing corner. The patch layout writes the same shared planes
#      between the rank pieces of one patch.
# ---------------------------------------------------------------------------
function test_shared_planes()
    section("shared piece planes")
    rdims = (2, np ÷ 2, 1)
    solver, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 1.5), (0.0, 0.5)),
                              bcs=per3,
                              ic=(x, y, z) -> Prim(u=(0.1sin(2π * y / 1.5),
                                                      0.1sin(2π * x), 0.0),
                                                   p=1.0, rho=1 + x + 2y)),
                      Numerics(n_global=(24, 36, 12),
                               execution=Execution(dims=rdims)))
    rho_of_xy(f) = maximum(abs(f["rho"][1, i, j, k] - (1 + f["x"][i] + 2f["y"][j]))
                           for i in axes(f["rho"], 2), j in axes(f["rho"], 3),
                               k in axes(f["rho"], 4))
    for st in (1, (2, 3, 1))
        save_vtk(solver, Q, "mpi_planes"; fields=(:rho, :velocity, :vorticity),
                 stride=st)
        check_tiling("$rdims ranks, stride $st",
                     () -> assemble_container("mpi_planes.pvtr");
                     expect=rho_of_xy)
        MPI.Barrier(comm)
        rank == 0 && foreach(rm, filter(startswith("mpi_planes"), readdir()))
        MPI.Barrier(comm)
    end

    # The root of a refined run is one patch over every rank; its pieces and
    # the blanking array they carry close the cells between ranks as above.
    wall2 = (SlipWallBC(), SlipWallBC())
    amr = build_solver(
        Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), bcs=(wall2, per3[2], per3[3]),
                ic=nothing),
        Numerics(n_global=(400, 1, 1),
                 amr=AMR(initial=BlockRegion((176, 0, 0), (16, 1, 1)), regrid_interval=0)))
    states = allocate_state(amr)
    initialize!(amr, states, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1 + x))
    save_vtk(amr, states, "mpi_planes_amr"; fields=(:rho,))
    for ℓ in 0:1
        listed() = begin
            txt = read("mpi_planes_amr.vtm", String)
            files = [m.captures[1] for m in eachmatch(r"file=\"([^\"]+)\"", txt)
                     if occursin(".L$ℓ.", m.captures[1])]
            # The level's grid is the union of its pieces' extents.
            exts = [first(read_vtk_piece(f)) for f in files]
            whole = [(isodd(c) ? minimum : maximum)(e[c] for e in exts) for c in 1:6]
            assemble_pieces(files, whole)
        end
        check_tiling("patch layout, level $ℓ", listed; expect=rho_of_x)
    end
    # The index and every piece carry the root's record, a piece with its
    # level and its patch's node count as written besides. That count is the
    # union of the extents of the level's pieces, whichever rank wrote each.
    MPI.Barrier(comm)
    bad = 0
    if rank == 0
        rec = read_vtk_record("mpi_planes_amr.vtm")
        bad += rec["n_global"] != [400, 1, 1] || haskey(rec, "level")
        bad += haskey(rec, "n_sampled")
        files = [m.captures[1]
                 for m in eachmatch(r"file=\"([^\"]+)\"", read("mpi_planes_amr.vtm", String))]
        for f in files
            piece = read_vtk_record(f)
            ℓ = parse(Int, match(r"\.L(\d+)\.", f).captures[1])
            exts = [first(read_vtk_piece(g)) for g in files if occursin(".L$ℓ.", g)]
            span = [maximum(e[2d] for e in exts) - minimum(e[2d-1] for e in exts) + 1
                    for d in 1:3]
            bad += get(piece, "level", nothing) != [ℓ]
            bad += get(piece, "n_sampled", nothing) != span
            bad += delete!(delete!(piece, "level"), "n_sampled") != rec
        end
    end
    check("patch layout: one record in the index and its pieces", gsum(bad), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_planes_amr"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# 11. Slicing under decomposition. A slice is the one case in which a rank
#     legitimately has nothing to write: only the ranks whose block spans the
#     requested plane hold any of it. Those ranks must write no piece file and
#     must not be listed in the container, and the pieces that do exist must
#     still tile the plane. Serial runs cannot show any of this.
# ---------------------------------------------------------------------------
function test_slicing()
    section("slicing under decomposition")
    solver, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                              bcs=(per3[1], per3[2], per3[3]),
                              ic=(x, y, z) -> Prim(u=(0.1, 0, 0), p=1.0,
                                                   rho=1 + x)),
                      Numerics(n_global=(SPLITN, 16, 16),
                               execution=Execution(dims=splitdims(1))))

    # (a) Slice across the split dimension: exactly one rank spans the plane.
    save_vtk(solver, Q, "mpi_slice_x"; fields=(:rho,), slice=(1, SPLITN ÷ 2 + 1))
    MPI.Barrier(comm)
    pieces = rank == 0 ? length(filter(f -> startswith(f, "mpi_slice_x") &&
                                            endswith(f, ".vtr"), readdir())) : 0
    check("split-dimension slice writes one piece", abs(gsum(pieces) - 1), 0.5)
    listed = rank == 0 ? count("<Piece", read("mpi_slice_x.pvtr", String)) : 0
    check("container lists only the rank holding the plane",
          abs(gsum(listed) - 1), 0.5)
    whole = rank == 0 ?
        (occursin("WholeExtent=\"0 0 0 15 0 15\"",
                  read("mpi_slice_x.pvtr", String)) ? 1 : 0) : 0
    check("sliced dimension is one point thick", abs(gsum(whole) - 1), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_slice_x"), readdir()))
    MPI.Barrier(comm)

    # (b) Slice across an UNDIVIDED dimension: every rank spans the plane, and
    #     their pieces must tile it.
    save_vtk(solver, Q, "mpi_slice_z"; fields=(:rho,), slice=(3, 8))
    MPI.Barrier(comm)
    pieces = rank == 0 ? length(filter(f -> startswith(f, "mpi_slice_z") &&
                                            endswith(f, ".vtr"), readdir())) : 0
    check("undivided-dimension slice writes every piece",
          abs(gsum(pieces) - np), 0.5)
    inside = 0
    if rank == 0
        txt = read("mpi_slice_z.pvtr", String)
        inside = 1
        for m in eachmatch(r"<Piece Extent=\"([^\"]+)\"", txt)
            e = parse.(Int, split(m.captures[1]))
            (e[5] == 0 && e[6] == 0) || (inside = 0)
        end
    end
    check_tiling("sliced pieces", () -> assemble_container("mpi_slice_z.pvtr");
                 expect=rho_of_x)
    check("every piece is flat in the sliced dimension",
          abs(gsum(inside) - 1), 0.5)
    check_record("sliced pieces", "mpi_slice_z.pvtr",
                 Dict("n_global" => [SPLITN, 16, 16], "stride" => [1, 1, 1],
                      "n_sampled" => [SPLITN, 16, 1],
                      "domain" => [0.0, 1.0, 0.0, 0.25, 0.0, 0.25]))
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_slice_z"), readdir()))
    MPI.Barrier(comm)

    # (c) The whole grid gathered to rank 0, which alone returns it: every
    #     rank's block lands at its offset, so the snapshot is the initial
    #     condition at its own coordinates.
    snap = field_snapshot(solver, Q; fields=(:rho, :velocity))
    check("snapshot returned on rank 0 only",
          abs(gsum(snap === nothing ? 0 : 1) - 1), 0.5)
    e = 0.0
    if rank == 0
        x = snap.coords[1]
        e = max(maximum(abs, snap[:rho] .- (1 .+ x)),
                maximum(abs, snap[:velocity][:, :, :, 1] .- 0.1),
                size(snap) == (SPLITN, 16, 16) ? 0.0 : 1.0,
                maximum(abs, x .- [global_xcoord(solver, 1, g) for g in 1:SPLITN]))
    end
    check("snapshot assembles the decomposed grid", gmax(e), 1e-14)

    # A window across the split dimension gathers its nodes from every rank
    # holding a part of it, and equals that block of the whole snapshot.
    lo, hi = SPLITN ÷ 4, SPLITN ÷ 2 + 3
    box = field_snapshot(solver, Q; fields=(:rho, :velocity),
                         window=(lo:hi, 3:10, nothing))
    e = 0.0
    if rank == 0
        e = max(maximum(abs, box[:rho] .- snap[:rho][lo:hi, 3:10, :]),
                maximum(abs, box[:velocity] .- snap[:velocity][lo:hi, 3:10, :, :]),
                box.offset == (lo - 1, 2, 0) ? 0.0 : 1.0,
                box.n_global == (SPLITN, 16, 16) ? 0.0 : 1.0)
    end
    check("windowed snapshot is the block of the whole one", gmax(e), 1e-300)

    # An out-of-range plane is rejected on every rank, not only the one that
    # would have held it.
    threw = try
        save_vtk(solver, Q, "mpi_slice_bad"; fields=(:rho,), slice=(1, SPLITN + 1))
        0
    catch e
        e isa ArgumentError ? 1 : 0
    end
    check("an out-of-range slice throws on every rank", abs(gsum(threw) - np), 0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_slice_bad"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# 11b. Line samples under decomposition. A line along the split dimension is
#      assembled from every rank; a line transverse to it lives on one rank
#      and the others contribute nothing. Both must come back identical on
#      every rank, and both must equal the initial condition at the nodes.
# ---------------------------------------------------------------------------
function test_line_sample()
    section("line samples under decomposition")
    rho_fn(x, y, z) = 1 + 0.1x + 0.2y + 0.3z
    solver, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 0.25), (0.0, 0.25)),
                              bcs=(per3[1], per3[2], per3[3]),
                              ic=(x, y, z) -> Prim(p=1.0, rho=rho_fn(x, y, z))),
                      Numerics(n_global=(SPLITN, 16, 16),
                               execution=Execution(dims=splitdims(1))))
    gx(d, g) = global_xcoord(solver, d, g)
    same_everywhere(v) = gmax(maximum(abs, v .- MPI.bcast(v, comm; root=0)))

    # (a) Along the split dimension at (j, k) = (5, 9): every rank owns a piece.
    coord, value = line_sample(solver, Q, :rho; dim=1, index=(5, 9))
    check("split-dimension line matches the initial condition",
          gmax(maximum(abs, value .- [rho_fn(gx(1, g), gx(2, 5), gx(3, 9))
                                      for g in 1:SPLITN])), 1e-14)
    check("split-dimension line is identical on every rank",
          same_everywhere(value), 0.0 + eps())
    check("split-dimension coordinate is identical on every rank",
          same_everywhere(coord), 0.0 + eps())

    # (b) Transverse to the split at (i, k) = (SPLITN ÷ 2, 3): one rank owns
    #     the whole line, the rest contribute zeros.
    coord, value = line_sample(solver, Q, :rho; dim=2, index=(SPLITN ÷ 2, 3))
    check("transverse line matches the initial condition",
          gmax(maximum(abs, value .- [rho_fn(gx(1, SPLITN ÷ 2), gx(2, g), gx(3, 3))
                                      for g in 1:16])), 1e-14)
    check("transverse line is identical on every rank",
          same_everywhere(value), 0.0 + eps())

    # (c) `at` snaps to the nearest node on every rank alike.
    _, snapped = line_sample(solver, Q, :rho; dim=2,
                             at=(gx(1, SPLITN ÷ 2) + solver.h[1] / 3, gx(3, 3)))
    check("coordinate lookup snaps to the indexed line",
          gmax(maximum(abs, snapped .- value)), 0.0 + eps())

    # An out-of-range index throws on every rank, not only the owner.
    threw = try
        line_sample(solver, Q, :rho; dim=1, index=(17, 1))
        0
    catch e
        e isa ArgumentError ? 1 : 0
    end
    check("an out-of-range line index throws on every rank",
          abs(gsum(threw) - np), 0.5)
end

# ---------------------------------------------------------------------------
# 10. Checkpoint round trip with a real decomposition — one file per rank, so
#    the per-rank offsets and extents must line up on write and read back.
# ---------------------------------------------------------------------------
function test_checkpoint()
    section("checkpoint round trip, decomposed")
    solver = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
        Numerics(n_global=(SPLITN, 16, 16), art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(1))))
    Q = allocate_state(solver)
    initialize!(solver, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
    solver.t = 1.25; solver.step = 17
    save_checkpoint(solver, Q, "mpi_ckpt")
    MPI.Barrier(comm)
    Q2 = allocate_state(solver); solver.t = 0.0; solver.step = 0
    load_checkpoint!(solver, Q2, "mpi_ckpt")
    check("restored t and step", abs(solver.t - 1.25) + abs(solver.step - 17), 1e-15)
    d = 0.0
    for c in 1:solver.equations.n_cons, k in 1:solver.decomp.n_local[3],
        j in 1:solver.decomp.n_local[2], i in 1:solver.decomp.n_local[1]
        d = max(d, abs(Q2[padded_index(solver, i, j, k), c] - Q[padded_index(solver, i, j, k), c]))
    end
    check("state bit-identical after round trip", gmax(d), 1e-300)
    # The header describes the state, not only its shape. Each rank writes the
    # global coordinates rather than its own slice of them, so a solver holding
    # the same extents over a longer domain is refused on every rank at once
    # rather than on whichever ranks happen to notice.
    stretched = build_solver(
        Problem(domain=((0.0, 4π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
        Numerics(n_global=(SPLITN, 16, 16), art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(1))))
    threw = try
        load_checkpoint!(stretched, allocate_state(stretched), "mpi_ckpt")
        0
    catch e
        occursin("grid coordinate mismatch", sprint(showerror, e)) ? 1 : 0
    end
    check("a different domain is refused on every rank", abs(gsum(threw) - np), 0.5)
    # The configuration record, written by every rank from the same data: the
    # species under the same name with another gamma is refused on every rank,
    # and a numerics change is refused on every rank unless allowed.
    hot = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                eos=IdealMixture([IdealSpecies("gas", 1.0, 1.3)]), ic=nothing),
        Numerics(n_global=(SPLITN, 16, 16), art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(1))))
    threw = try
        load_checkpoint!(hot, allocate_state(hot), "mpi_ckpt")
        0
    catch e
        occursin("eos.sp[1].gamma", sprint(showerror, e)) ? 1 : 0
    end
    check("a different gamma under the same name is refused on every rank",
          abs(gsum(threw) - np), 0.5)
    c8 = build_solver(
        Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
        Numerics(n_global=(SPLITN, 16, 16), deriv=lele_d1_8(),
                 art=ArtificialProperties(enabled=false),
                 execution=Execution(dims=splitdims(1))))
    threw = try
        load_checkpoint!(c8, allocate_state(c8), "mpi_ckpt")
        0
    catch e
        occursin("allow = (:numerics,)", sprint(showerror, e)) ? 1 : 0
    end
    check("a numerics change is refused on every rank", abs(gsum(threw) - np), 0.5)
    Q8 = allocate_state(c8)
    load_checkpoint!(c8, Q8, "mpi_ckpt"; allow=(:numerics,))
    check("an allowed numerics change restores the state", gmax(abs(c8.step - 17)), 0.5)
    # One rank does the cleanup: every rank wrote its own file, but all of them
    # racing to rm the whole glob makes them delete each other's entries
    # and throw ENOENT.
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith("mpi_ckpt"), readdir()))
    MPI.Barrier(comm)
end

# ---------------------------------------------------------------------------
# 10a. The end of the job. A signal reaches one rank only, the last, and the
#     BeforeWallLimit verdict must still be every rank's on the same step, or
#     the checkpoint its effect writes is collective on some ranks only. The
#     checkpoint it writes continues the run bit for bit. A commit interrupted
#     between the ranks' renames, which leaves ranks on both sets, is then
#     completed by the load: every rank reads the newer set.
# ---------------------------------------------------------------------------
function test_wall_limit()
    section("wall limit: rank-local signal, interrupted commit")
    build() = begin
        s = build_solver(
            Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3, ic=nothing),
            Numerics(n_global=(SPLITN, 16, 16), art=ArtificialProperties(enabled=false),
                     execution=Execution(dims=splitdims(1))))
        Q = allocate_state(s)
        initialize!(s, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y),
                                            rho=1 + 0.1sin(x)))
        s, Q
    end
    stem = "mpi_wall"
    # Set on the last rank after step 3, ahead of the trigger in the tuple.
    signal = (s, _) -> (rank == np - 1 && s.step == 3 && (CL.WALL_SIGNAL[] = 10); nothing)
    stop = Callback(BeforeWallLimit(deadline=time() + 1e6),
                    (s, Q) -> (save_checkpoint(s, Q, stem); true))
    s, Q = build()
    run!(s, Q; tfinal=1e9, nmax=12, callback=(signal, stop))
    CL.WALL_SIGNAL[] = 0
    check("a signal on one rank stops every rank on step 3",
          MPI.Allreduce(abs(s.step - 3), max, comm), 0.5)

    r, Qr = build()
    load_checkpoint!(r, Qr, stem)
    run!(s, Q; tfinal=1e9, nmax=6)
    run!(r, Qr; tfinal=1e9, nmax=6)
    inner = CL.interior(s.decomp)
    check("the checkpoint it wrote continues bit for bit",
          gmax(maximum(abs, parent(Qr)[inner, :] .- parent(Q)[inner, :])), 1e-300)

    # The newer set at step 6, committed by hand and interrupted after the
    # even ranks' renames: the odd ranks still hold step 3 beside a `.tmp`.
    path = CL._ckpt_name(stem, rank)
    open(io -> CL._write_checkpoint(io, s, Q, nothing), path * ".tmp", "w")
    MPI.Barrier(comm)
    rank == 0 && write(CL._ckpt_marker(stem), string(np))
    MPI.Barrier(comm)
    iseven(rank) && Base.Filesystem.rename(path * ".tmp", path)
    MPI.Barrier(comm)
    u, Qu = build()
    load_checkpoint!(u, Qu, stem)
    check("an interrupted commit is completed: every rank reads step 6",
          MPI.Allreduce(abs(u.step - 6), max, comm), 0.5)
    check("the completed set holds the newer state",
          gmax(maximum(abs, parent(Qu)[inner, :] .- parent(Q)[inner, :])), 1e-300)
    check("the marker is gone and every .tmp renamed",
          gsum(Int(isfile(path * ".tmp")) + Int(rank == 0 && isfile(CL._ckpt_marker(stem)))),
          0.5)
    MPI.Barrier(comm)
    rank == 0 && foreach(rm, filter(startswith(stem), readdir()))
    MPI.Barrier(comm)
end
