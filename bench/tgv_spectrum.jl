# Three-dimensional kinetic-energy spectra of Taylor-Green snapshots, computed
# offline from the HDF5 checkpoints `bench/tgv_energy.jl snapshots=...` writes.
#
# The solver computes no transform. The run dumps state and this script reads
# it back serially, so the spectra under N1 cost no distributed FFT and no new
# dependency inside the package. Both dependencies live in the calling
# project, which needs CompactLES only to have produced the files:
#
#   julia --project=<a project carrying HDF5 and FFTW> \
#       ~/.julia/dev/CompactLES/bench/tgv_spectrum.jl 'tgv_snapshots/*.h5'
#
# Usage: one or more paths or globs, then `key=value` options:
#
#   tgv_spectrum.jl <path|glob> [<path|glob> ...] [key=value ...]
#
#   out       directory for the per-snapshot `.dat` tables, empty (default) to
#             print only. Each file is two columns, k and E(k), with the
#             summary in comment lines above them, so a plotting script needs
#             no parsing beyond whitespace.
#   rows      spectrum rows to print per snapshot (default 24). The `.dat`
#             always carries every shell. Set 0 to print the summary alone,
#             which is the useful setting when comparing a whole sweep.
#   band      the high-wavenumber band whose share is reported, as a fraction
#             of the Nyquist wavenumber (default 0.5). See the note below.
#
# --- What this measures ------------------------------------------------------
#
# E(k) is shell-averaged over integer wavenumbers, which are the physical ones
# because the domain is exactly (2 pi)^3:
#
#     E(k) = (1/2) sum_{|kvec| rounds to k} |uhat|^2,   uhat = fft(u)/npts
#
# summed over the three velocity components. With that normalization Parseval
# makes `sum(E)` the volume-averaged kinetic energy of the velocity field, so it
# is directly comparable to the `kinetic_energy` history `bench/tgv_energy.jl`
# prints and to the reference's own energy column. The script prints that sum
# beside the density-weighted mean of |u|^2/2 the solver reports; the two differ
# only through the density fluctuation, which is O(Ma^2) at Ma 0.1, so a ratio
# far from one means the snapshot is not what it is assumed to be rather than
# that the flow is compressible.
#
# The velocity is momentum over density, not sqrt(rho) times velocity. That is
# the incompressible convention and the one the reference solution uses, so
# these spectra stay comparable to published Taylor-Green results.
#
# The periodic grid carries no duplicated endpoint, which is the convention
# `fft` assumes: a `Patch` spans `L/N` per cell on a periodic dimension and
# `L/(N-1)` only on a non-periodic one.
#
# --- Why the high-k share is the quantity to compare -------------------------
#
# The compact filter removes energy at the grid scale, so two filter settings
# reaching the same total dissipation can distribute it very differently in
# wavenumber, and the -dKE/dt history cannot tell them apart: it is one number
# per instant, and the sinks compete for a supply fixed at the large scales
# (reference/CALIBRATION.md, "The timestep moves the attribution, not the
# total"). The share of energy above half the Nyquist wavenumber separates them
# directly. A filter that is too weak leaves a pile-up there; one that is too
# strong empties the band and takes part of the inertial range with it.
#
# Snapshots compare across a sweep only when taken at the same instants, which
# `AtTime` guarantees by shortening a step to land on each one. Under
# `filter_cfl = 0` that shortened step still pays a full filter pass, so
# requesting snapshots perturbs the budget slightly at each one; the perturbation
# is common to every configuration sharing the snapshot times.
using Printf
using HDF5
using FFTW

const DEFAULTS = (out = "", rows = 24, band = 0.5)

"""
Parse `key=value` options against `DEFAULTS`, which is both the fallback and the
schema, and reject an unknown key. This repeats a little of `script_args`
because the script runs in a project that needs HDF5 and FFTW to postprocess an
existing dump; loading the solver for an argument parser is out of proportion.
"""
function parse_options(opts)
    parsed = Dict{Symbol,Any}(pairs(DEFAULTS))
    for a in opts
        i = findfirst('=', a)
        key = Symbol(a[1:prevind(a, i)])
        haskey(parsed, key) ||
            error("unknown option '$key', want one of: " *
                  join(keys(DEFAULTS), ", "))
        text = a[nextind(a, i):end]
        d = DEFAULTS[key]
        if d isa AbstractString
            parsed[key] = String(text)
        else
            v = tryparse(typeof(d), text)
            v === nothing && error("option '$key=$text' wants a $(typeof(d))")
            parsed[key] = v
        end
    end
    return NamedTuple{keys(DEFAULTS)}(map(k -> parsed[k], keys(DEFAULTS)))
end

"Expand paths and shell-style globs into the files they name, sorted and unique."
function expand_paths(specs)
    out = String[]
    for spec in specs
        if occursin('*', spec) || occursin('?', spec)
            dir = dirname(spec)
            isempty(dir) && (dir = ".")
            isdir(dir) || continue
            pat = Regex("^" * replace(basename(spec), "." => "\\.",
                                      "*" => ".*", "?" => ".") * "\$")
            for f in readdir(dir)
                occursin(pat, f) && push!(out, joinpath(dir, f))
            end
        else
            push!(out, spec)
        end
    end
    isempty(out) && error("no snapshot files matched")
    return sort!(unique!(out))
end

"""
The velocity field and the density-weighted kinetic energy of one checkpoint.

The component layout comes from the file's own `meta/component_names`, which
`NavierStokes1T` writes as one partial density per species followed by `rho_u1`,
`rho_u2`, `rho_u3` and `rho_E`. Reading the names rather than assuming the
offsets keeps this correct for a multi-species snapshot, whose density is the
sum of the partial ones. The names come back from a null-padded fixed-width
record, so they are stripped before matching.
"""
function read_snapshot(path)
    Q, names, t, n_global = h5open(path, "r") do f
        (read(f["state/Q"]),
         String.(rstrip.(read(f["meta/component_names"]), '\0')),
         read(f["meta/t"]), read(f["meta/n_global"]))
    end
    imom = ntuple(d -> findfirst(==("rho_u$d"), names), 3)
    any(isnothing, imom) && error("$path: no rho_u1..3 among $names")
    momentum = ("rho_u1", "rho_u2", "rho_u3", "rho_E")
    ispecies = findall(n -> startswith(n, "rho_") && !(n in momentum), names)
    isempty(ispecies) && error("$path: no partial densities among $names")
    rho = sum(Q[:, :, :, sp] for sp in ispecies)
    u = ntuple(d -> Q[:, :, :, imom[d]] ./ rho, 3)
    ke = 0.0
    @inbounds for I in eachindex(rho)
        ke += rho[I] * (u[1][I]^2 + u[2][I]^2 + u[3][I]^2) / 2
    end
    return (; u, t = Float64(t), n_global = Tuple(Int.(n_global)),
            ke = ke / length(rho))
end

"""
Shell-averaged kinetic-energy spectrum of a velocity field on a `(2 pi)^3`
periodic grid, as `(0:kmax, E)`.
"""
function spectrum(u)
    n = size(u[1])
    npts = prod(n)
    wavenumbers(m) = [i <= m ÷ 2 ? i - 1 : i - 1 - m for i in 1:m]
    kx, ky, kz = wavenumbers(n[1]), wavenumbers(n[2]), wavenumbers(n[3])
    # `ceil` and `abs`: the most negative wavenumber is one larger in magnitude
    # than the most positive, and a corner shell rounds up, so both are needed
    # for `E` to be long enough for every point.
    kmax = ceil(Int, sqrt(maximum(abs, kx)^2 + maximum(abs, ky)^2 +
                          maximum(abs, kz)^2))
    E = zeros(Float64, kmax + 1)
    for d in 1:3
        uh = fft(u[d]) ./ npts
        @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
            shell = round(Int, sqrt(kx[i]^2 + ky[j]^2 + kz[k]^2))
            E[shell+1] += abs2(uh[i, j, k]) / 2
        end
    end
    return 0:kmax, E
end

"Fraction of the spectrum's energy above `band` times the Nyquist wavenumber."
function high_k_share(ks, E, n_global, band)
    kcut = band * (minimum(n_global) ÷ 2)
    total = sum(E)
    hi = sum((E[i] for (i, k) in enumerate(ks) if k > kcut), init=0.0)
    return hi / max(total, eps()), kcut, total
end

function report(path, opt)
    snap = read_snapshot(path)
    ks, E = spectrum(snap.u)
    share, kcut, total = high_k_share(ks, E, snap.n_global, opt.band)
    nx, ny, nz = snap.n_global
    @printf("\n--- %s\n", path)
    @printf("t = %.4f, grid %d x %d x %d, shells 0..%d\n",
            snap.t, nx, ny, nz, last(ks))
    @printf("sum E(k) = %.6e; rho-weighted KE = %.6e; ratio %.6f\n",
            total, snap.ke, total / snap.ke)
    @printf("above k = %.1f (%.2f x Nyquist): %.4e, %.3f%% of the total\n",
            kcut, opt.band, share * total, 100 * share)
    if opt.rows > 0
        println("     k        E(k)        k^(5/3) E(k)")
        for (i, k) in enumerate(ks)
            i > opt.rows && break
            @printf("  %4d   %.6e   %.6e\n", k, E[i],
                    k == 0 ? 0.0 : k^(5 / 3) * E[i])
        end
        length(ks) > opt.rows &&
            println("  ... $(length(ks) - opt.rows) more shells")
    end
    if !isempty(opt.out)
        mkpath(opt.out)
        stem = replace(basename(path), r"\.h5$" => "")
        dest = joinpath(opt.out, stem * "_spectrum.dat")
        open(dest, "w") do io
            @printf(io, "# %s\n", path)
            @printf(io, "# t = %.6f  grid = %d %d %d\n", snap.t, nx, ny, nz)
            @printf(io, "# sum E = %.8e  rho-weighted KE = %.8e\n", total, snap.ke)
            @printf(io, "# share above k = %.2f: %.6e\n", kcut, share)
            println(io, "# k  E(k)")
            for (i, k) in enumerate(ks)
                @printf(io, "%d %.10e\n", k, E[i])
            end
        end
        println("wrote $dest")
    end
    return (; path, t=snap.t, total, share)
end

function main(args)
    paths = filter(a -> !occursin('=', a), args)
    opt = parse_options(filter(a -> occursin('=', a), args))
    isempty(paths) && error("give at least one snapshot path or glob")
    summaries = [report(f, opt) for f in expand_paths(paths)]
    length(summaries) > 1 || return nothing
    # One row per snapshot, so that a sweep can be compared on the high-k
    # share without reading several hundred spectrum rows.
    println("\n=== summary ===")
    w = max(maximum(s -> length(basename(s.path)), summaries), 8)
    @printf("%-*s %7s %13s %9s\n", w, "snapshot", "t", "sum E", "high-k %")
    for s in summaries
        @printf("%-*s %7.3f %13.6e %9.3f\n",
                w, basename(s.path), s.t, s.total, 100 * s.share)
    end
    return nothing
end

main(ARGS)
