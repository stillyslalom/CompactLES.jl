# SESAME equation-of-state tables in the LANL ASCII 2 library format, read into
# the (ln T, ln ρ) table machinery of ionmix.jl. Like an `IonmixTable`, this is
# a standalone table model: no `EOS` method reads it, so nothing here enters the
# runtime flux.
#
# Format source: G. Young, "Sesame ASCII 2 File Format", LA-UR-21-23834 (2021).
# A file opens with a line "Version 2.0" and holds one or more materials, each a
# run of records in ascending table number. A record is a description line
#
#   File_Number Material_ID Table_ID Number_of_Words Create_Date Update_Date Version
#
# (File_Number 0 on a material's first record, 1 after), then its data. Tables
# 100-199 and 10101-10199 hold Number_of_Words characters, 80 to a line; every
# other table holds Number_of_Words numbers in E notation, at most five to a
# line, separated by blanks. The content of the numbered tables is that of the
# SESAME library (LA-UR-92-3407), in its units: density g/cm³ (= Mg/m³),
# temperature K, pressure GPa, specific energy and free energy MJ/kg. The tables
# read here are
#
#   201   Z̄, Ā, reference density, bulk modulus (GPa), exchange coefficient
#   301   nr, nt, ρ (nr), T (nt), then p, e and the Helmholtz free energy A, each
#         nr × nt with the density varying fastest (A absent in some tables)
#   303   the same layout, the cold curve plus the nuclear (ion) contribution
#   304   the thermal electronic contribution
#   305   the nuclear contribution alone
#   306   the cold curve: the same layout with nt = 1, usually at T = 0
#
# The decomposition follows J. McHardy, "An Introduction to the Theory and Use
# of SESAME Equations of State", LA-14503 (2018): A = φ(ρ) + A_nuc(ρ, T) +
# A_elec(ρ, T), with 301 the sum, 303 = φ + A_nuc, 304 = A_elec, 305 = A_nuc and
# 306 = φ, so 301 = 303 + 304 = 305 + 304 + 306 where the tables share a grid.
# Every other table (the 400 phase-boundary, 500 opacity and 600 conductivity
# series, the 321 phase fractions) is skipped.

const _SESAME_DENSITY_SCALE = 1e3        # g/cm³ → kg/m³
const _SESAME_PRESSURE_SCALE = 1e9       # GPa → Pa
const _SESAME_ENERGY_SCALE = 1e6         # MJ/kg → J/kg
const _SESAME_GRID_TABLES = ((301, :total), (303, :ion), (304, :electron),
                             (305, :nuclear))
const _SESAME_INTERPOLATIONS = (:bilinear, :free_energy)

_sesame_text_table(id::Integer) = 100 <= id <= 199 || 10101 <= id <= 10199

"""
    SesameComponent{T}

One ρ–T table of a [`SesameTable`](@ref) (a 301, 303, 304 or 305 record) in SI
units: temperature in K, density in kg/m³, pressure in Pa, specific energy and
Helmholtz free energy in J/kg. The `(ntemp, ndens)` matrices `p`, `e` and
`free_energy` vary temperature fastest, as an [`IonmixTable`](@ref)'s do (the
file varies density fastest); `free_energy` is a 0×0 matrix when the record
carries only `p` and `e`.

Fields: `table_id`; `temperature`, `density` and their natural logarithms
`log_temperature`, `log_density`; `p`, `e`, `free_energy`; `free_energy_xy`,
the cross derivative `∂²A/∂ln T ∂ln ρ` estimated at the nodes (0×0 under
`:bilinear`); `extrapolate` (`:missing` or `:linear`, as for
[`table_value`](@ref)); `interpolation`; `dropped`, the numbers of temperature
and density nodes at zero removed on reading, since the interpolation is in the
logarithms; and `monotone`, per density node, whether `e` increases strictly
with temperature.

[`table_value`](@ref), [`table_state`](@ref) and
[`table_temperature_status`](@ref) evaluate the component in one of two forms,
named by `interpolation`:

- `:bilinear`: `p`, `e` and `free_energy` are interpolated independently,
  bilinearly in `(ln T, ln rho)`, as for an `IonmixTable`. The interpolant is
  continuous, exact at the nodes and second order in the spacing, the
  inversion is exact, and a query costs four table reads per field. Between
  nodes the thermodynamic consistency relation `rho² ∂e/∂rho = p - T ∂p/∂T`
  holds only to the error of the interpolant's derivatives, which is first
  order in the spacing.
- `:free_energy`: one bicubic Hermite interpolant of `A` in `(ln T, ln rho)`,
  whose node slopes are the tabulated `∂A/∂ln T = A - e` and
  `∂A/∂ln rho = p/rho`, and whose cross derivative is the mean of the
  second-order differences of those slopes. Energy, pressure and entropy are
  its derivatives, `e = A - T ∂A/∂T`, `p = rho² ∂A/∂rho`, `s = -∂A/∂T`, so the
  consistency relation and `cv = T ∂s/∂T` hold to round-off inside the axes;
  `p` and `e` are continuous, third order in the spacing and reproduce the
  table at the nodes to round-off, and `cv` and the sound speed jump across
  cell edges. Outside the axes `e`, `p` and `A` continue linearly in the
  logarithms from the edge, as the bilinear interpolant's do. This form is
  available only with the free energy on file and costs sixteen table reads
  per query;
  the inversion makes one pass over the temperature nodes and a safeguarded
  Newton iteration in the bracketing cell.

Build one through [`read_sesame`](@ref), or from SI arrays:

    SesameComponent(; temperature, density, p, e, free_energy=nothing,
                    table_id=301, extrapolate=:missing, interpolation=:bilinear)
"""
struct SesameComponent{T<:AbstractFloat} <: _LogGridTable{T}
    table_id::Int
    temperature::Vector{T}
    density::Vector{T}
    log_temperature::Vector{T}
    log_density::Vector{T}
    p::Matrix{T}
    e::Matrix{T}
    free_energy::Matrix{T}
    free_energy_xy::Matrix{T}
    extrapolate::Symbol
    interpolation::Symbol
    dropped::NTuple{2,Int}
    monotone::Vector{Bool}
end

function SesameComponent(; temperature::AbstractVector, density::AbstractVector, p, e,
                         free_energy=nothing, table_id::Integer=301,
                         extrapolate::Symbol=:missing,
                         interpolation::Symbol=:bilinear)
    T = float(promote_type(eltype(temperature), eltype(density), eltype(p), eltype(e)))
    A = free_energy === nothing ? zeros(T, 0, 0) : Matrix{T}(free_energy)
    return _sesame_component(T, Int(table_id), Vector{T}(temperature),
                             Vector{T}(density), Matrix{T}(p), Matrix{T}(e), A,
                             extrapolate, interpolation, (0, 0))
end

function _check_axis(owner, name, axis; minimum=2)
    length(axis) >= minimum ||
        throw(ArgumentError("$owner: the $name axis must have at least $minimum nodes"))
    all(x -> isfinite(x) && x > 0, axis) ||
        throw(ArgumentError("$owner: the $name axis must be finite and positive"))
    all(i -> axis[i] < axis[i+1], 1:length(axis)-1) ||
        throw(ArgumentError("$owner: the $name axis must increase strictly"))
end

function _sesame_component(::Type{T}, table_id, temperature, density, p, e, A,
                           extrapolate, interpolation, dropped) where {T}
    extrapolate in (:missing, :linear) ||
        throw(ArgumentError("SesameComponent: extrapolate must be :missing or " *
                            ":linear, got :$extrapolate"))
    interpolation in _SESAME_INTERPOLATIONS ||
        throw(ArgumentError("SesameComponent: interpolation must be :bilinear or " *
                            ":free_energy, got :$interpolation"))
    _check_axis("SesameComponent", "temperature", temperature)
    _check_axis("SesameComponent", "density", density)
    nt, nd = length(temperature), length(density)
    for (name, F) in (("p", p), ("e", e))
        size(F) == (nt, nd) ||
            throw(DimensionMismatch("SesameComponent: $name is $(size(F)), expected " *
                                    "($nt, $nd)"))
    end
    isempty(A) || size(A) == (nt, nd) ||
        throw(DimensionMismatch("SesameComponent: free_energy is $(size(A)), " *
                                "expected ($nt, $nd) or empty"))
    all(isfinite, p) && all(isfinite, e) && all(isfinite, A) ||
        throw(ArgumentError("SesameComponent: table entries must be finite"))
    x, y = log.(temperature), log.(density)
    if interpolation === :free_energy
        isempty(A) &&
            throw(ArgumentError("SesameComponent: interpolation = :free_energy " *
                                "requires the free energy"))
        # ∂²A/∂x∂y as the mean of the x-derivative of ∂A/∂y = p/ρ and the
        # y-derivative of ∂A/∂x = A - e, each by a second-order difference on
        # the non-uniform nodes.
        A_y = p ./ reshape(density, 1, nd)
        A_x = A .- e
        A_xy = similar(A)
        for j in 1:nd
            A_xy[:, j] .= _node_derivative(view(A_y, :, j), x)
        end
        for i in 1:nt
            A_xy[i, :] .= (view(A_xy, i, :) .+ _node_derivative(view(A_x, i, :), y)) ./ 2
        end
    else
        A_xy = zeros(T, 0, 0)
    end
    monotone = Bool[_table_increasing(e, j) for j in 1:nd]
    return SesameComponent{T}(table_id, temperature, density, x, y, p, e, A, A_xy,
                              extrapolate, interpolation, dropped, monotone)
end

# The derivative of nodal values f on the strictly increasing nodes x: the
# three-point second-order difference, one-sided at the ends, and the two-point
# difference on two nodes.
function _node_derivative(f::AbstractVector, x::AbstractVector)
    n = length(x)
    d = similar(f, n)
    if n == 2
        d .= (f[2] - f[1]) / (x[2] - x[1])
        return d
    end
    for k in 1:n
        # The three nodes a < b < c used for node k, and k's position among them.
        m = clamp(k, 2, n - 1)
        a, b, c = x[m-1], x[m], x[m+1]
        fa, fb, fc = f[m-1], f[m], f[m+1]
        t = x[k]
        # The derivative at t of the quadratic through the three points.
        d[k] = fa * (2t - b - c) / ((a - b) * (a - c)) +
               fb * (2t - a - c) / ((b - a) * (b - c)) +
               fc * (2t - a - b) / ((c - a) * (c - b))
    end
    return d
end

"""
    SesameColdCurve{T}

The cold curve of a [`SesameTable`](@ref) (its 306 record) in SI units: the
pressure `p`, specific energy `e` and free energy `free_energy` on the isotherm
`temperature` (0 K on file as a rule, where `e` and `free_energy` coincide), at
the densities `density`, with `log_density`, `extrapolate`, and `dropped`, the
number of density nodes at zero removed on reading. [`table_value`](@ref)
interpolates it linearly in `ln rho`.
"""
struct SesameColdCurve{T<:AbstractFloat}
    table_id::Int
    temperature::T
    density::Vector{T}
    log_density::Vector{T}
    p::Vector{T}
    e::Vector{T}
    free_energy::Vector{T}
    extrapolate::Symbol
    dropped::Int
end

function _sesame_cold(::Type{T}, table_id, temperature, density, p, e, A, extrapolate,
                      dropped) where {T}
    _check_axis("SesameColdCurve", "density", density)
    n = length(density)
    for (name, F) in (("p", p), ("e", e), ("free_energy", A))
        length(F) == n ||
            throw(DimensionMismatch("SesameColdCurve: $name has $(length(F)) " *
                                    "values for $n densities"))
        all(isfinite, F) ||
            throw(ArgumentError("SesameColdCurve: table entries must be finite"))
    end
    return SesameColdCurve{T}(table_id, T(temperature), density, log.(density), p, e,
                              A, extrapolate, dropped)
end

"""
    SesameTable{T}

One material of a SESAME library in SI units, read by [`read_sesame`](@ref).

Fields:

- `material_id` and `comments`, the text of the 101-199 records as
  `table_id => text` pairs.
- From the 201 record: `zbar`, the mean atomic number; `abar`, the mean atomic
  mass in g/mol (unscaled); `reference_density` in kg/m³; `bulk_modulus` in Pa;
  `exchange_coefficient`. All five are `NaN` when the material has no 201
  record.
- `total`, the 301 record, a [`SesameComponent`](@ref).
- `ion` (303: the cold curve plus the nuclear contribution), `electron` (304:
  the thermal electronic contribution) and `nuclear` (305: the nuclear
  contribution alone), each a `SesameComponent` or `nothing` when absent.
- `cold`, the 306 cold curve, a [`SesameColdCurve`](@ref) or `nothing`.
- `records`, the `(table_id, create_date, update_date, version)` of each record
  read, which [`write_sesame`](@ref) writes back.

The free energy of a SESAME table decomposes as `A = φ(rho) + A_nuc(rho, T) +
A_elec(rho, T)` into the cold curve, the nuclear (ion) motion and the thermal
electrons. The 301 record is the sum; 303 and 304 split it into ion-plus-cold
and electron parts, and 305 and 306 split the ion part further. Each part is a
separate table on its own grid, so a sum of parts matches the 301 record only
where the grids coincide.

A SESAME table is a one-temperature equilibrium model. Each part is computed at
`T_ion = T_ele`, the derivatives of every interpolant are equilibrium
derivatives, and the table contains no ionization state, no charge-state
populations and no frozen derivatives. The ion and electron parts, where 303 and
304 exist, are the only temperature partition in it, and evaluating them at
different temperatures departs from the model that produced them.

[`table_value`](@ref), [`table_state`](@ref) and
[`table_temperature_status`](@ref) take the table, which evaluates `total`,
or any of its components directly.
"""
struct SesameTable{T<:AbstractFloat}
    material_id::Int
    comments::Vector{Pair{Int,String}}
    zbar::T
    abar::T
    reference_density::T
    bulk_modulus::T
    exchange_coefficient::T
    total::SesameComponent{T}
    ion::Union{Nothing,SesameComponent{T}}
    electron::Union{Nothing,SesameComponent{T}}
    nuclear::Union{Nothing,SesameComponent{T}}
    cold::Union{Nothing,SesameColdCurve{T}}
    records::Vector{NTuple{4,Int}}
end

"""
    SesameTable(material_id, total; ion=nothing, electron=nothing, nuclear=nothing,
                cold=nothing, zbar=NaN, abar=NaN, reference_density=NaN,
                bulk_modulus=NaN, exchange_coefficient=NaN, comments=[])

Assemble a table from its components, in SI units, for [`write_sesame`](@ref).
Every record is dated 0 with version 1.
"""
function SesameTable(material_id::Integer, total::SesameComponent{T}; ion=nothing,
                     electron=nothing, nuclear=nothing, cold=nothing, zbar=NaN,
                     abar=NaN, reference_density=NaN, bulk_modulus=NaN,
                     exchange_coefficient=NaN,
                     comments=Pair{Int,String}[]) where {T}
    ids = [first.(comments); 201; 301]
    for (part, id) in ((ion, 303), (electron, 304), (nuclear, 305), (cold, 306))
        part === nothing || push!(ids, id)
    end
    records = [(id, 0, 0, 1) for id in sort!(ids)]
    return SesameTable{T}(Int(material_id), collect(Pair{Int,String}, comments),
                          T(zbar), T(abar), T(reference_density), T(bulk_modulus),
                          T(exchange_coefficient), total, ion, electron, nuclear, cold,
                          records)
end

function Base.show(io::IO, table::SesameTable{T}) where {T}
    c = table.total
    parts = [string(id) for (id, name) in _SESAME_GRID_TABLES[2:end]
             if getfield(table, name) !== nothing]
    table.cold === nothing || push!(parts, "306")
    print(io, "SesameTable{$T}(", table.material_id, ", ", length(c.temperature),
          " temperatures ", @sprintf("%.4g", c.temperature[1]), "–",
          @sprintf("%.4g", c.temperature[end]), " K, ", length(c.density),
          " densities ", @sprintf("%.4g", c.density[1]), "–",
          @sprintf("%.4g", c.density[end]), " kg/m³, parts [", join(parts, ", "),
          "], :", c.interpolation, ", extrapolate=:", c.extrapolate, ")")
end

function Base.show(io::IO, c::SesameComponent{T}) where {T}
    print(io, "SesameComponent{$T}(", c.table_id, ", ", length(c.temperature), "×",
          length(c.density), ", :", c.interpolation, ", extrapolate=:",
          c.extrapolate, ")")
end

# --- Reading and writing ------------------------------------------------------

# The records of an ASCII 2 file: (material, table, create, update, version, data)
# with data a String for a text table and a Vector{Float64} otherwise.
function _sesame_records(path::AbstractString)
    lines = [rstrip(line, '\r') for line in readlines(path)]
    k = findfirst(line -> !isempty(strip(line)), lines)
    k !== nothing && startswith(lowercase(strip(lines[k])), "version") ||
        throw(ArgumentError("SESAME: $path does not open with a version line; " *
                            "read_sesame reads the ASCII 2 format, not the " *
                            "fixed-width ASCII 1 format"))
    records = Tuple{Int,Int,Int,Int,Int,Union{String,Vector{Float64}}}[]
    n = length(lines)
    k += 1
    while k <= n
        isempty(strip(lines[k])) && (k += 1; continue)
        words = split(lines[k])
        length(words) >= 4 ||
            throw(ArgumentError("SESAME: cannot read the record description " *
                                "\"$(lines[k])\" on line $k"))
        fields = [tryparse(Int, w) for w in words[1:min(7, end)]]
        any(isnothing, fields) &&
            throw(ArgumentError("SESAME: cannot read the record description " *
                                "\"$(lines[k])\" on line $k"))
        material, id, count = fields[2], fields[3], fields[4]
        dates = ntuple(i -> 4 + i <= length(fields) ? fields[4+i] : 0, 3)
        k += 1
        if _sesame_text_table(id)
            nlines = cld(count, 80)
            k + nlines - 1 <= n ||
                throw(ArgumentError("SESAME: table $id of material $material ends " *
                                    "before its $count characters"))
            data = join(lines[k:k+nlines-1])
            k += nlines
        else
            data = Float64[]
            sizehint!(data, count)
            while length(data) < count
                k <= n || throw(ArgumentError("SESAME: table $id of material " *
                                              "$material ends after $(length(data)) " *
                                              "of $count numbers"))
                for word in split(lines[k])
                    value = tryparse(Float64, word)
                    value === nothing &&
                        throw(ArgumentError("SESAME: cannot read \"$word\" on line " *
                                            "$k as a number"))
                    push!(data, value)
                end
                length(data) <= count ||
                    throw(ArgumentError("SESAME: table $id of material $material " *
                                        "has more than $count numbers"))
                k += 1
            end
        end
        push!(records, (material, id, dates..., data))
    end
    return records
end

# Drop the leading nodes of an axis at or below zero; return the kept range.
function _positive_nodes(axis)
    first = findfirst(>(0), axis)
    first === nothing && return 1:0
    return first:length(axis)
end

# A 30x record as (T axis, ρ axis, the (nt, nr) arrays on file) in file units.
function _sesame_grid(id, material, data::Vector{Float64})
    length(data) >= 2 ||
        throw(ArgumentError("SESAME: table $id of material $material is empty"))
    nr, nt = Int(data[1]), Int(data[2])
    nr >= 1 && nt >= 1 ||
        throw(ArgumentError("SESAME: table $id of material $material has $nr " *
                            "densities and $nt temperatures"))
    points = nr * nt
    arrays = (length(data) - 2 - nr - nt) ÷ points
    length(data) == 2 + nr + nt + arrays * points && arrays in (2, 3) ||
        throw(ArgumentError("SESAME: table $id of material $material holds " *
                            "$(length(data)) numbers, which is not p and e, or p, e " *
                            "and A, on $nr densities and $nt temperatures"))
    density = data[3:2+nr]
    temperature = data[3+nr:2+nr+nt]
    # On file the density varies fastest; the tables here vary temperature first.
    block(k) = permutedims(reshape(data[3+nr+nt+(k-1)*points:2+nr+nt+k*points], nr, nt))
    return temperature, density, ntuple(block, arrays)
end

"""
    read_sesame([T=Float64,] path, material_id; interpolation=:bilinear,
                extrapolate=:missing) -> SesameTable{T}

Read material `material_id` from a SESAME library file in the LANL ASCII 2
format into a [`SesameTable`](@ref), converting every field to SI: densities
from g/cm³ to kg/m³, pressures from GPa to Pa, energies and free energies from
MJ/kg to J/kg; temperatures are in K on file and stay so.

- `interpolation` is `:bilinear` or `:free_energy` for every ρ–T component, as
  described under [`SesameComponent`](@ref); `:free_energy` is an error unless
  each of the material's 301 and 303-305 records holds the free energy.
- `extrapolate` sets the policy outside the axes, as under
  [`table_value`](@ref); the default `:missing` reports every such query as out
  of domain.

The material's 301 record is required; 201, 303, 304, 305 and 306 are read when
present, and every other record is skipped. Nodes at zero temperature or
density, which many 30x records include at the cold end, are dropped because
the interpolation is in the logarithms, and their number is recorded in each
component's `dropped`; the cold curve's zero-temperature isotherm is retained.
A missing
material is an error that lists the materials in the file. The fixed-width
ASCII 1 format and the binary and HDF5 forms of the library are not read. The
cost is one pass over the file.
"""
read_sesame(path::AbstractString, material_id::Integer; kwargs...) =
    read_sesame(Float64, path, material_id; kwargs...)

function read_sesame(::Type{T}, path::AbstractString, material_id::Integer;
                     interpolation::Symbol=:bilinear,
                     extrapolate::Symbol=:missing) where {T<:AbstractFloat}
    records = _sesame_records(path)
    mine = filter(r -> r[1] == material_id, records)
    if isempty(mine)
        materials = unique(r[1] for r in records)
        throw(ArgumentError("SESAME: $path holds no material $material_id; it holds " *
                            (isempty(materials) ? "none" : join(materials, ", "))))
    end
    tables = Dict(r[2] => r[6] for r in mine)
    length(tables) == length(mine) ||
        throw(ArgumentError("SESAME: material $material_id repeats a table"))
    haskey(tables, 301) ||
        throw(ArgumentError("SESAME: material $material_id has no 301 table"))
    comments = Pair{Int,String}[id => text for (id, text) in sort!(collect(tables);
                                                                 by=first)
                                if _sesame_text_table(id) && id != 100]
    properties = fill(T(NaN), 5)
    if haskey(tables, 201)
        words = tables[201]
        scales = (1.0, 1.0, _SESAME_DENSITY_SCALE, _SESAME_PRESSURE_SCALE, 1.0)
        for k in 1:min(5, length(words))
            properties[k] = T(words[k] * scales[k])
        end
    end
    function component(id)
        haskey(tables, id) || return nothing
        temperature, density, arrays = _sesame_grid(id, material_id, tables[id])
        kt, kd = _positive_nodes(temperature), _positive_nodes(density)
        dropped = (length(temperature) - length(kt), length(density) - length(kd))
        si(k, scale) = T.(arrays[k][kt, kd] .* scale)
        A = length(arrays) == 3 ? si(3, _SESAME_ENERGY_SCALE) : zeros(T, 0, 0)
        return _sesame_component(T, id, T.(temperature[kt]),
                                 T.(density[kd] .* _SESAME_DENSITY_SCALE),
                                 si(1, _SESAME_PRESSURE_SCALE),
                                 si(2, _SESAME_ENERGY_SCALE), A, extrapolate,
                                 interpolation, dropped)
    end
    parts = Tuple(component(id) for (id, _) in _SESAME_GRID_TABLES)
    cold = nothing
    if haskey(tables, 306)
        temperature, density, arrays = _sesame_grid(306, material_id, tables[306])
        length(temperature) == 1 ||
            throw(ArgumentError("SESAME: the 306 table of material $material_id has " *
                                "$(length(temperature)) temperatures, not one"))
        kd = _positive_nodes(density)
        length(arrays) == 3 ||
            throw(ArgumentError("SESAME: the 306 table of material $material_id " *
                                "has no free energy"))
        column(k, scale) = T.(vec(arrays[k])[kd] .* scale)
        cold = _sesame_cold(T, 306, temperature[1],
                            T.(density[kd] .* _SESAME_DENSITY_SCALE),
                            column(1, _SESAME_PRESSURE_SCALE),
                            column(2, _SESAME_ENERGY_SCALE),
                            column(3, _SESAME_ENERGY_SCALE), extrapolate,
                            length(density) - length(kd))
    end
    records_read = [(r[2], r[3], r[4], r[5]) for r in mine
                    if r[2] == 201 || r[2] in (301, 303, 304, 305, 306) ||
                       (_sesame_text_table(r[2]) && r[2] != 100)]
    return SesameTable{T}(Int(material_id), comments, properties..., parts..., cold,
                          records_read)
end

# A value in file units as a medium-length SESAME word: sign or blank, 16
# significant digits, a two-digit exponent. Of the value divided by its scale
# and that quotient's nearest neighbours, the first whose text the reader maps
# back to `value` is written, so that a table read from a file and written
# again reads back bit for bit.
function _sesame_word(value::AbstractFloat, scale)
    isfinite(value) || throw(ArgumentError("SESAME: cannot write the value $value"))
    q = Float64(value) / scale
    candidates = (q, nextfloat(q), prevfloat(q), nextfloat(q, 2), prevfloat(q, 2))
    text = @sprintf("% .15E", q)
    for c in candidates
        candidate = @sprintf("% .15E", c)
        if convert(typeof(value), parse(Float64, candidate) * scale) == value
            text = candidate
            break
        end
    end
    length(text) == 22 ||
        throw(ArgumentError("SESAME: $value is outside the two-digit exponent range"))
    return text
end
_sesame_word(value::Integer, scale) = _sesame_word(Float64(value), scale)

function _write_sesame_record(io, file_number, material, id, dates, words)
    create, update, version = dates
    println(io, file_number, " ", material, " ", id, " ", length(words), " ", create,
            " ", update, " ", version)
    for start in 1:5:length(words)
        println(io, join(words[start:min(start + 4, end)], " "))
    end
end

function _sesame_grid_words(c, scale_A)
    nt, nd = length(c.temperature), length(c.density)
    words = String[_sesame_word(nd, 1.0), _sesame_word(nt, 1.0)]
    append!(words, _sesame_word.(c.density, _SESAME_DENSITY_SCALE))
    append!(words, _sesame_word.(c.temperature, 1.0))
    # Back to the file's order, density fastest.
    append!(words, _sesame_word.(permutedims(c.p), _SESAME_PRESSURE_SCALE))
    append!(words, _sesame_word.(permutedims(c.e), _SESAME_ENERGY_SCALE))
    isempty(c.free_energy) ||
        append!(words, _sesame_word.(permutedims(c.free_energy), scale_A))
    return words
end

"""
    write_sesame(path, tables::SesameTable...)

Write one or more materials to a SESAME library file in the ASCII 2 format,
converting back to the file's units, with the records each table holds (its
comments, 201 when its properties are not `NaN`, 301, and 303-306 where
present) in ascending order. Numbers are written in the medium length, 16
significant digits; each is chosen so that reading the file reproduces the
table's values exactly where a 16-digit word does, which holds for every table
read from a file, so that writing a table read from a file and reading it again
returns the same table bit for bit. Nodes dropped on reading are not written.
The cost is one pass over the tables.
"""
function write_sesame(path::AbstractString, tables::SesameTable...)
    ids = [t.material_id for t in tables]
    allunique(ids) ||
        throw(ArgumentError("write_sesame: a material may appear once in a file"))
    open(path, "w") do io
        println(io, "Version 2.0")
        for table in tables
            dates = Dict(r[1] => (r[2], r[3], r[4]) for r in table.records)
            date(id) = get(dates, id, (0, 0, 1))
            file_number = Ref(0)
            function record(id, words)
                _write_sesame_record(io, file_number[], table.material_id, id, date(id),
                                     words)
                file_number[] = 1
            end
            for (id, text) in table.comments
                println(io, file_number[], " ", table.material_id, " ", id, " ",
                        length(text), " ", join(date(id), " "))
                for start in 1:80:length(text)
                    println(io, text[start:min(start + 79, end)])
                end
                file_number[] = 1
            end
            properties = (table.zbar, table.abar, table.reference_density,
                          table.bulk_modulus, table.exchange_coefficient)
            if !all(isnan, properties)
                scales = (1.0, 1.0, _SESAME_DENSITY_SCALE, _SESAME_PRESSURE_SCALE, 1.0)
                record(201, [_sesame_word(v, s) for (v, s) in zip(properties, scales)])
            end
            for (id, name) in _SESAME_GRID_TABLES
                c = getfield(table, name)
                c === nothing ||
                    record(id, _sesame_grid_words(c, _SESAME_ENERGY_SCALE))
            end
            cold = table.cold
            if cold !== nothing
                words = String[_sesame_word(length(cold.density), 1.0),
                               _sesame_word(1, 1.0)]
                append!(words, _sesame_word.(cold.density, _SESAME_DENSITY_SCALE))
                push!(words, _sesame_word(cold.temperature, 1.0))
                append!(words, _sesame_word.(cold.p, _SESAME_PRESSURE_SCALE))
                append!(words, _sesame_word.(cold.e, _SESAME_ENERGY_SCALE))
                append!(words, _sesame_word.(cold.free_energy, _SESAME_ENERGY_SCALE))
                record(306, words)
            end
        end
    end
    return path
end

# --- Interpolation --------------------------------------------------------------

# The cubic Hermite basis on [0, 1] at t, as (value at 0, value at 1, slope at
# 0, slope at 1) weights, with its first and second derivatives. Each weight is
# exactly 0 or 1 at t = 0 and t = 1, so the interpolant reproduces the node data.
@inline function _hermite_basis(t)
    t2 = t * t
    t3 = t2 * t
    value = (2t3 - 3t2 + 1, 3t2 - 2t3, t3 - 2t2 + t, t3 - t2)
    slope = (6t2 - 6t, 6t - 6t2, 3t2 - 4t + 1, 3t2 - 2t)
    curvature = (12t - 6, 6 - 12t, 6t - 4, 6t - 2)
    return value, slope, curvature
end

# The bicubic Hermite interpolant of the free energy in cell (i, j) at (s, r),
# and its derivatives in x = ln T and y = ln ρ: (A, A_x, A_y, A_xx, A_xy, A_yy).
@inline function _free_energy_patch(c::SesameComponent{T}, i, j, s, r, dx,
                                    dy) where {T}
    bs, ds, cs = _hermite_basis(s)
    br, dr, cr = _hermite_basis(r)
    F = Fs = Fr = Fss = Fsr = Frr = zero(T)
    @inbounds for b in 0:1, a in 0:1
        ii, jj = i + a, j + b
        A = c.free_energy[ii, jj]
        # The node data in cell units: A, ∂A/∂s, ∂A/∂r, ∂²A/∂s∂r.
        v = A
        vs = dx * (A - c.e[ii, jj])
        vr = dy * (c.p[ii, jj] / c.density[jj])
        vsr = dx * dy * c.free_energy_xy[ii, jj]
        term(ws, wr) = v * ws[a+1] * wr[b+1] + vs * ws[a+3] * wr[b+1] +
                       vr * ws[a+1] * wr[b+3] + vsr * ws[a+3] * wr[b+3]
        F += term(bs, br)
        Fs += term(ds, br)
        Fr += term(bs, dr)
        Fss += term(cs, br)
        Fsr += term(ds, dr)
        Frr += term(bs, cr)
    end
    return F, Fs / dx, Fr / dy, Fss / dx^2, Fsr / (dx * dy), Frr / dy^2
end

# The fields of the free-energy interpolant in cell (i, j) at (s, r), each as
# (value, ∂/∂x, ∂/∂y) in x = ln T and y = ln ρ: e = A - A_x, p = ρ A_y and A.
# Outside the axes (s or r beyond [0, 1]) the three fields continue linearly in
# x and y from the nearest point of the edge cell, with that point's
# derivatives: the linear extension the bilinear interpolant makes, where a
# continued cubic would soon give a negative heat capacity. Inside, the
# extension terms are exact zeros.
@inline function _free_energy_fields(c::SesameComponent{T}, i, j, s, r, dx, dy,
                                     rho) where {T}
    sc, rc = clamp(s, zero(T), one(T)), clamp(r, zero(T), one(T))
    A, A_x, A_y, A_xx, A_xy, A_yy = _free_energy_patch(c, i, j, sc, rc, dx, dy)
    @inbounds ρc = rc == r ? T(rho) : rc == 0 ? c.density[j] : c.density[j+1]
    Δx, Δy = (s - sc) * dx, (r - rc) * dy
    extend(f, f_x, f_y) = (f + f_x * Δx + f_y * Δy, f_x, f_y)
    e = extend(A - A_x, A_x - A_xx, A_y - A_xy)
    p = extend(ρc * A_y, ρc * A_xy, ρc * (A_y + A_yy))
    F = extend(A, A_x, A_y)
    return e, p, F
end

# The state from the free-energy interpolant at a location:
# (p, dp/dT, dp/dρ), (e, de/dT, de/dρ), (A, dA/dT, dA/dρ), s.
@inline function _free_energy_state(c::SesameComponent, loc::TableLocation)
    e, p, F = _free_energy_fields(c, loc.i, loc.j, loc.s, loc.r, loc.dx, loc.dy,
                                  loc.rho)
    Tq, ρq = loc.T_ion, loc.rho
    physical((f, f_x, f_y)) = (f, f_x / Tq, f_y / ρq)
    return physical(p), physical(e), physical(F), (e[1] - F[1]) / Tq
end

function _sesame_field(c::SesameComponent, field::Symbol)
    field === :p && return c.p
    field === :e && return c.e
    if field === :free_energy
        isempty(c.free_energy) &&
            throw(ArgumentError("table_value: table $(c.table_id) has no free energy"))
        return c.free_energy
    end
    throw(ArgumentError("table_value: unknown field :$field; a SESAME component " *
                        "has :p, :e and :free_energy"))
end

"""
    table_value(component::SesameComponent, field, T_ion, rho)
        -> (f, ∂f/∂T_ion, ∂f/∂rho, status)
    table_value(table::SesameTable, field, T_ion, rho)

Interpolate `field` (`:p`, `:e` or `:free_energy`) of a
[`SesameComponent`](@ref), or of a [`SesameTable`](@ref)'s `total`, under the
component's `interpolation`, with the domain policy and status flags of the
`IonmixTable` method. Under `:free_energy` the three fields and their
derivatives come from the one free-energy interpolant.
"""
function table_value(c::SesameComponent, field::Symbol, T_ion, rho)
    F = _sesame_field(c, field)
    c.interpolation === :bilinear && return _table_value(c, F, T_ion, rho)
    loc = _table_location(c, T_ion, rho)
    p, e, A, _ = _free_energy_state(c, loc)
    return ((field === :p ? p : field === :e ? e : A)..., loc.status)
end

table_value(table::SesameTable, field::Symbol, T_ion, rho) =
    table_value(table.total, field, T_ion, rho)

"""
    table_value(cold::SesameColdCurve, field, rho) -> (f, ∂f/∂rho, status)

Interpolate `field` (`:p`, `:e` or `:free_energy`) of a
[`SesameColdCurve`](@ref) linearly in `ln rho`, with the domain policy and
status flags of the other methods.
"""
function table_value(cold::SesameColdCurve{T}, field::Symbol, rho) where {T}
    f = field === :p ? cold.p : field === :e ? cold.e :
        field === :free_energy ? cold.free_energy :
        throw(ArgumentError("table_value: unknown field :$field; a SESAME cold " *
                            "curve has :p, :e and :free_energy"))
    ρq = T(rho)
    valid = isfinite(ρq) && ρq > 0
    y = valid ? log(ρq) : T(NaN)
    j, r, dy, outside = _axis_position(cold.log_density, y)
    policy = cold.extrapolate === :missing ? TABLE_OUT_OF_DOMAIN : TABLE_EXTRAPOLATED
    status = outside ? ((valid ? policy : TABLE_OUT_OF_DOMAIN) | TABLE_DENSITY_AXIS) :
             TABLE_OK
    @inbounds lo, hi = f[j], f[j+1]
    return ((1 - r) * lo + r * hi, (hi - lo) / (dy * ρq), status)
end

"""
    table_state(component::SesameComponent, T_ion, rho) -> NamedTuple
    table_state(table::SesameTable, T_ion, rho)

The one-temperature equilibrium state of a [`SesameComponent`](@ref), or of a
[`SesameTable`](@ref)'s `total`: the fields of the `IonmixTable` method but
`zbar`, with the free energy `free_energy` and the specific entropy
`entropy = (e - free_energy)/T_ion` (`NaN` without a free energy), and
`consistency = rho² de_drho - (p - T_ion dp_dT)`, the residual in Pa of the
thermodynamic consistency relation. That residual is round-off under
`:free_energy` inside the axes and first order in the spacing under
`:bilinear`.
"""
function table_state(c::SesameComponent{T}, T_ion, rho) where {T}
    loc = _table_location(c, T_ion, rho)
    if c.interpolation === :free_energy
        (p, dp_dT, dp_drho), (e, cv, de_drho), (A, _, _), s = _free_energy_state(c, loc)
    else
        p, dp_dT, dp_drho = _table_bilinear(loc, c.p)
        e, cv, de_drho = _table_bilinear(loc, c.e)
        A = isempty(c.free_energy) ? T(NaN) : _table_bilinear(loc, c.free_energy)[1]
        s = (e - A) / loc.T_ion
    end
    Tq, ρq = loc.T_ion, loc.rho
    c2 = dp_drho + Tq * dp_dT^2 / (ρq^2 * cv)
    consistency = ρq^2 * de_drho - (p - Tq * dp_dT)
    status = loc.status
    cv > 0 && c2 > 0 || (status |= TABLE_UNSTABLE)
    return (; p, e, free_energy=A, entropy=s, cv, dp_dT, dp_drho, de_drho, c2,
            consistency, status)
end

table_state(table::SesameTable, T_ion, rho) = table_state(table.total, T_ion, rho)

# --- Temperature inversion ------------------------------------------------------

"""
    table_temperature_status(component::SesameComponent, e, rho) -> (T_ion, status)
    table_temperature_status(table::SesameTable, e, rho, component=:total)

Invert the interpolated specific energy of a [`SesameComponent`](@ref) for the
temperature at density `rho`; on a [`SesameTable`](@ref), `component` is
`:total`, `:ion`, `:electron` or `:nuclear`. Under `:bilinear` the inversion
and its statuses are those of the `IonmixTable` method. Under `:free_energy`
the energy along the density column is a cubic in `ln T` within each cell: the
lowest cell whose end energies bracket `e` is found in one pass over the
temperature nodes, and the root in it by Newton iteration safeguarded by
bisection, so that a forward query at the returned temperature reproduces `e`
to round-off. An energy equal to the interpolant's at a node returns the node's
temperature exactly, and a column whose node energies do not increase strictly
is reported as `TABLE_NOT_MONOTONE`. An energy beyond the column is placed on
the linear extension of the edge and reported under the `extrapolate` policy,
provided the energy increases there; otherwise the result is `NaN`, out of
domain.
"""
function table_temperature_status(c::SesameComponent, e, rho)
    c.interpolation === :bilinear && return _table_temperature(c, c.e, c.monotone, e, rho)
    return _free_energy_temperature(c, e, rho)
end

function table_temperature_status(table::SesameTable, e, rho, component::Symbol=:total)
    component in (:total, :ion, :electron, :nuclear) ||
        throw(ArgumentError("table_temperature_status: component must be :total, " *
                            ":ion, :electron or :nuclear, got :$component"))
    c = getfield(table, component)
    c === nothing &&
        throw(ArgumentError("table_temperature_status: material " *
                            "$(table.material_id) has no :$component component"))
    return table_temperature_status(c, e, rho)
end

table_temperature(c::SesameComponent, e, rho) = table_temperature_status(c, e, rho)[1]
table_temperature(table::SesameTable, e, rho, component::Symbol=:total) =
    table_temperature_status(table, e, rho, component)[1]

# The energy of the free-energy interpolant in cell (i, j) at (s, r), density
# rho, and its derivative in s.
@inline function _free_energy_energy(c::SesameComponent, i, j, s, r, rho)
    x, y = c.log_temperature, c.log_density
    @inbounds dx, dy = x[i+1] - x[i], y[j+1] - y[j]
    (e, e_x, _), _, _ = _free_energy_fields(c, i, j, s, r, dx, dy, rho)
    return e, dx * e_x
end

function _free_energy_temperature(c::SesameComponent{T}, e, rho) where {T}
    eq, ρq = T(e), T(rho)
    nan = T(NaN)
    isfinite(ρq) && ρq > 0 || return (nan, TABLE_OUT_OF_DOMAIN | TABLE_DENSITY_AXIS)
    isfinite(eq) || return (nan, TABLE_OUT_OF_DOMAIN)
    policy = _table_policy(c)
    j, r, _, outside_ρ = _axis_position(c.log_density, log(ρq))
    status = outside_ρ ? (policy | TABLE_DENSITY_AXIS) : TABLE_OK
    x = c.log_temperature
    nt = length(x)
    # The column energy at node k, from the cell above it (below it at the top).
    node(k) = k < nt ? _free_energy_energy(c, k, j, zero(T), r, ρq)[1] :
                       _free_energy_energy(c, nt - 1, j, one(T), r, ρq)[1]
    root = 0
    increasing = true
    below = node(1)
    e_first = e_lo = e_hi = below
    for i in 1:nt-1
        above = node(i + 1)
        increasing &= above > below
        if root == 0 && below != above && min(below, above) <= eq <= max(below, above)
            root, e_lo, e_hi = i, below, above
        end
        below = above
    end
    increasing || (status |= TABLE_NOT_MONOTONE)
    # An energy equal to a node's returns the node's temperature exactly.
    root > 0 && eq == e_lo && return (c.temperature[root], status)
    root > 0 && eq == e_hi && return (c.temperature[root+1], status)
    f(s) = (v = _free_energy_energy(c, root, j, s, r, ρq); (v[1] - eq, v[2]))
    if root == 0
        # Beyond the column the energy is linear in ln T, continuing the edge
        # node with its slope, which must be positive.
        root = eq < e_first ? 1 : nt - 1
        edge = eq < e_first ? zero(T) : one(T)
        g, dg = f(edge)
        dg > 0 || return (nan, status | TABLE_OUT_OF_DOMAIN | TABLE_TEMPERATURE_AXIS)
        s = edge - g / dg
        status |= policy | TABLE_TEMPERATURE_AXIS
    else
        # A sign change on [0, 1]: Newton steps kept inside the shrinking bracket.
        lo, hi = zero(T), one(T)
        g_lo = e_lo - eq
        s = (lo + hi) / 2
        for _ in 1:100
            g, dg = f(s)
            g == 0 && break
            (g < 0) == (g_lo < 0) ? (lo = s; g_lo = g) : (hi = s)
            candidate = s - g / dg
            if lo < candidate < hi
                done = abs(candidate - s) <= 2eps(T)
                s = candidate
                done && break
            else
                s = (lo + hi) / 2
            end
            hi - lo <= 2eps(T) && break
        end
    end
    return (exp(x[root] + s * (x[root+1] - x[root])), status)
end
