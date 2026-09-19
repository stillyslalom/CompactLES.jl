# Reproducible extraction of the Song--Wang (2016) binary diffusion fits.
# Run with: julia --project=. -t 1 data/songwang_extract.jl <text dump> [degree=4]
#
# The supporting information of B. Song, K. Kang, Z. Zhang, X. Wang and
# Z. Liu, J. Chem. Eng. Data 61, 1910 (2016), doi:10.1021/acs.jced.6b00076
# (je6b00076_si_001.pdf, DOI 10.1021/acs.jced.6b00076.s001, ACS Figshare
# record 3208147/file 5037301) tabulates the low-density
# binary diffusion coefficient D12 in cm^2/s at 101.3 kPa on 27 temperatures
# from 298.15 to 2000 K for every pair of 3He, 4He, H2, HD, HT, D2, DT and T2
# at mole fractions 0.25, 0.50 and 0.75, with a 2.0% expanded uncertainty.
# The PDF has a text layer; dump it page by page with form feeds between
# pages:
#
#     python -c "import pymupdf; d = pymupdf.open('je6b00076_si_001.pdf');
#                open('si.txt', 'w', encoding='utf-8').write(
#                    '\f'.join(p.get_text() for p in d))"
#
# This script reads that dump, fits log(D12) at x1 = 0.50 with a polynomial
# in log(T / 300 K) of the given degree, and prints the SONG_WANG_2016 rows
# of src/neutral_diffusion_data.jl together with each pair's largest node
# residual and its largest departure between the three compositions. It
# vendors the fit and its diagnostics, not the tables. The independently
# implemented Xpdf/layout audit in songwang_verify.jl checks the publisher
# artifact's SHA-256, all 78 mixture tables, all 702 equimolar source nodes,
# pair identities, composition spread and the printed uncertainty.
using LinearAlgebra
using Printf

const TEMPERATURE_REF = 300.0
const CHECK_NODES = (298.15, 500.0, 1000.0, 2000.0)

function parse_tables(path)
    tables = Dict{Tuple{String,String,Float64},Vector{Tuple{Float64,Float64}}}()
    for page in split(read(path, String), '\f')
        number = match(r"Table S(\d+)\s+Low-density transport properties of the system", page)
        m = match(r"of the system (\S+) \(1\) \+ (\S+) \(2\) at[^=]*=\s*([\d.]+)", page)
        (number === nothing || m === nothing) && continue
        lines = [strip(l) for l in split(page, '\n') if !isempty(strip(l))]
        start = findfirst(l -> startswith(l, "α"), lines) + 1
        stop = findfirst(l -> startswith(l, "aThe"), lines) - 1
        cells = lines[start:stop]
        rows = Tuple{Float64,Float64}[]
        for k in 1:5:length(cells)-4
            all(c -> occursin(r"^[\d.]+$", c), cells[k:k+4]) || continue
            push!(rows, (parse(Float64, cells[k]), parse(Float64, cells[k+3])))
        end
        length(rows) == 27 || error("table S$(number.captures[1]) has $(length(rows)) rows")
        tables[(String(m.captures[1]), String(m.captures[2]),
                parse(Float64, m.captures[3]))] = rows
    end
    return tables
end

function fit(rows, degree)
    z = [log(T / TEMPERATURE_REF) for (T, _) in rows]
    y = [log(D) for (_, D) in rows]
    vandermonde = [zk^m for zk in z, m in 0:degree]
    c = vandermonde \ y
    residual = maximum(abs.(expm1.(vandermonde * c .- y)))
    return c, residual
end

function main(path, degree)
    tables = parse_tables(path)
    pairs = sort(unique((a, b) for (a, b, _) in keys(tables)))
    println("    # x1 = 0.50 fits in log(T / 300 K), 298.15 to 2000 K, 101.3 kPa")
    for (a, b) in pairs
        rows = tables[(a, b, 0.5)]
        c, residual = fit(rows, degree)
        spread = 0.0
        for x in (0.25, 0.75), (r5, rx) in zip(rows, tables[(a, b, x)])
            spread = max(spread, abs(rx[2] / r5[2] - 1))
        end
        D_ref = exp(c[1]) * 1e-4
        coefficients = join((@sprintf("%.8g", ck) for ck in c[2:end]), ", ")
        @printf("    _song_wang(\"%s\", \"%s\", %.8g,\n               (%s), %.1e, %.1e),\n",
                a, b, D_ref, coefficients, residual, spread)
    end
    println("check nodes (T, D in cm^2/s at x1 = 0.50):")
    for (a, b) in pairs
        rows = tables[(a, b, 0.5)]
        nodes = [(T, D) for (T, D) in rows if T in CHECK_NODES]
        @printf("    (\"%s\", \"%s\") => %s,\n", a, b, string(Tuple(nodes)))
    end
end

length(ARGS) >= 1 || error("usage: songwang_extract.jl <text dump> [degree]")
main(ARGS[1], length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4)
