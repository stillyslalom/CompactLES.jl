# Independent audit of the Song--Wang supporting-information extraction.
#
# This verifier deliberately does not use the PyMuPDF text dump consumed by
# songwang_extract.jl. It runs Xpdf-compatible `pdftotext -layout`, whose row layout
# differs from PyMuPDF's cell order, and recovers D12 by column position.  The
# copyrighted source tables remain in the user-supplied PDF and are never
# written to the repository.
#
# Run with:
#   julia --project=. -t 1 data/songwang_verify.jl je6b00076_si_001.pdf
using CompactLES
using LinearAlgebra
using SHA

const SOURCE_SHA256 = "e93ee0a69c8f1d976e321065efe9bd3042a6352630a5e589d4e049587271de9a"
const TEMPERATURES = (298.15, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0,
                      440.0, 460.0, 480.0, 500.0, 550.0, 600.0, 650.0, 700.0,
                      750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1200.0, 1400.0,
                      1600.0, 1800.0, 2000.0)

function layout_text(pdf)
    exe = Sys.which("pdftotext")
    exe === nothing && error("Xpdf-compatible pdftotext is required for the independent audit")
    out = tempname()
    try
        run(`$exe -layout -enc UTF-8 $pdf $out`)
        return read(out, String)
    finally
        isfile(out) && rm(out)
    end
end

function parse_binary_tables(text)
    tables = Dict{Tuple{String,String,Float64},Vector{Tuple{Float64,Float64}}}()
    for page in split(text, '\f')
        header = match(r"Table S(\d+) Low-density transport properties of the system\s+(\S+) \(1\) \+ (\S+) \(2\) at[^=]*=\s+([\d.]+)", page)
        header === nothing && continue
        lines = split(page, '\n')
        firstrow = findfirst(line -> occursin("D12 / cm2", line), lines)
        lastrow = findfirst(line -> startswith(strip(line), "aThe pressure"), lines)
        (firstrow === nothing || lastrow === nothing) &&
            error("could not locate rows in Table S$(header.captures[1])")
        values = Float64[]
        for line in lines[firstrow+1:lastrow-1]
            fields = split(strip(line))
            # A data row has four properties plus an optional printed T.  In
            # pdftotext layout the temperature glyph is vertically associated
            # with the following property row, so row order is the independent
            # key and D12 is always the penultimate numeric column.
            length(fields) in (4, 5) || continue
            all(field -> tryparse(Float64, field) !== nothing, fields) || continue
            push!(values, parse(Float64, fields[end-1]))
        end
        length(values) == length(TEMPERATURES) ||
            error("Table S$(header.captures[1]) has $(length(values)) D12 rows, expected $(length(TEMPERATURES))")
        key = (String(header.captures[2]), String(header.captures[3]),
               parse(Float64, header.captures[4]))
        haskey(tables, key) && error("duplicate table for $key")
        tables[key] = collect(zip(TEMPERATURES, values))
        occursin(r"combined expanded uncertainties[\s\S]*2\.0\s*%", page) ||
            error("Table S$(header.captures[1]) lacks the printed 2.0% expanded uncertainty")
    end
    return tables
end

function fit(rows)
    z = [log(T / 300.0) for (T, _) in rows]
    y = [log(D) for (_, D) in rows]
    V = [zk^m for zk in z, m in 0:4]
    c = V \ y
    residual = maximum(abs.(expm1.(V * c .- y)))
    return c, residual
end

function audit(pdf)
    digest = bytes2hex(sha256(read(pdf)))
    digest == SOURCE_SHA256 || error("unexpected source PDF SHA-256 $digest")
    tables = parse_binary_tables(layout_text(pdf))
    length(tables) == 78 || error("recovered $(length(tables)) binary tables, expected 78")

    pairs = sort(unique((a, b) for (a, b, _) in keys(tables)))
    length(pairs) == 26 || error("recovered $(length(pairs)) binary pairs, expected 26")
    for (a, b) in pairs
        pair = song_wang_pair(a, b)
        pair === nothing && error("no vendored fit for $a-$b")
        rows = tables[(a, b, 0.5)]
        c, residual = fit(rows)
        Dref = exp(c[1]) * 1e-4
        isapprox(pair.D_ref, Dref; rtol=6e-8) || error("D_ref mismatch for $a-$b")
        all(isapprox.(collect(pair.coefficients), c[2:end]; rtol=6e-7, atol=6e-10)) ||
            error("coefficient mismatch for $a-$b")
        residual <= 1.1pair.node_residual || error("node residual mismatch for $a-$b")
        for (T, D) in rows
            departure = abs(song_wang_diffusivity(pair, T, pair.pressure_ref) / (D * 1e-4) - 1)
            departure <= 1.1pair.node_residual ||
                error("source-node mismatch for $a-$b at $T K")
        end
        spread = maximum(abs(tables[(a, b, x)][k][2] / rows[k][2] - 1)
                         for x in (0.25, 0.75), k in eachindex(rows))
        spread <= 1.1pair.composition_spread || error("composition-spread mismatch for $a-$b")
    end
    Set((pair.species for pair in SONG_WANG_2016)) == Set(pairs) ||
        error("vendored and recovered pair identities differ")
    println("verified 78 tables, 26 pairs, and 702 equimolar source nodes")
    println("source SHA-256: $digest")
end

length(ARGS) == 1 || error("usage: songwang_verify.jl <je6b00076_si_001.pdf>")
audit(only(ARGS))
