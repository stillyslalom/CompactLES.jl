# Reader for the fixed-column NASA CEA thermodynamic database. The source table,
# its Apache license, and its NOTICE are bundled verbatim in `data/`.
#
# Every magic column number below comes from the CEA record format, which is
# Fortran and therefore positional, not delimited. A field is where it is, and
# whitespace inside one is significant:
#
#   name     A16 name (cols 1-16), then a free-text reference comment
#   header   (I2, 1X, A6, 1X, 5(A2,F6.2), I2, F13.5, F15.3)
#            → n_intervals 1-2, date 4-9, formula 11-50, phase 51-52,
#              molar mass 53-65, heat of formation 66-80
#   bounds   (2F11.3, I1, 8F5.1, F17.3) → Tmin 1-11, Tmax 12-22
#   coeffs   (5D16.8) then (2D16.8, 16X, 2D16.8) → a₁..a₅, a₆a₇, b₁ 49-64
#
# Exponents are written `D` instead of `E`, so the float parser substitutes
# before parsing.

const NASA9_THERMO_PATH = normpath(joinpath(@__DIR__, "..", "data", "thermo.inp"))
const UNIVERSAL_GAS_CONSTANT = 8.31446261815324 # J/(mol K)

function _nasa9_field(line, first, last, context)
    ncodeunits(line) >= last ||
        throw(ArgumentError("short NASA CEA record while reading $context"))
    return SubString(line, first, last)
end

function _nasa9_float(field, context)
    value = replace(strip(String(field)), 'D' => 'E', 'd' => 'e')
    isempty(value) && throw(ArgumentError("empty NASA CEA field for $context"))
    try
        return parse(Float64, value)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("invalid NASA CEA number '$value' for $context"))
    end
end

function _nasa9_int(field, context)
    value = strip(String(field))
    try
        return parse(Int, value)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("invalid NASA CEA integer '$value' for $context"))
    end
end

function _nasa9_reference(species::Nasa9Species{T}, reference, T_ref) where {T}
    reference === :formation && return species
    reference === :sensible ||
        throw(ArgumentError("reference must be :sensible or :formation"))
    T_ref > 0 || throw(ArgumentError("T_ref must be positive"))
    interval = _nasa9_interval(species, T_ref)
    shift = -_nasa9_h_over_R(interval, T_ref)
    intervals = [Nasa9Interval{T}(item.Tmin, item.Tmax, item.a, item.b1 + shift)
                 for item in species.intervals]
    return Nasa9Species{T}(species.name, species.R, intervals)
end

function _nasa9_record(lines, i, header, n_intervals, name, path)
    n_intervals > 0 || throw(ArgumentError(
        "NASA CEA species $name has no temperature intervals; it is a " *
        "reactant-only record carrying a heat of formation, not a fit"))
    phase = _nasa9_int(_nasa9_field(header, 51, 52, name), name)
    phase == 0 || throw(ArgumentError("NASA CEA species $name is not gaseous"))
    molar_mass = _nasa9_float(_nasa9_field(header, 53, 65, name), name)
    molar_mass > 0 || throw(ArgumentError("non-positive molar mass for $name"))
    R = 1000 * UNIVERSAL_GAS_CONSTANT / molar_mass

    intervals = Vector{Nasa9Interval{Float64}}(undef, n_intervals)
    for interval_index in 1:n_intervals
        first_line = i + 3 * interval_index - 1
        last_line = first_line + 2
        last_line <= length(lines) ||
            throw(ArgumentError("truncated NASA CEA intervals for $name in $path"))
        bounds, coeffs1, coeffs2 = lines[first_line:last_line]
        Tmin = _nasa9_float(_nasa9_field(bounds, 1, 11, name), name)
        Tmax = _nasa9_float(_nasa9_field(bounds, 12, 22, name), name)
        a = ntuple(7) do coefficient
            line = coefficient <= 5 ? coeffs1 : coeffs2
            field = coefficient <= 5 ? coefficient : coefficient - 5
            _nasa9_float(_nasa9_field(line, 16 * field - 15, 16 * field, name), name)
        end
        b1 = _nasa9_float(_nasa9_field(coeffs2, 49, 64, name), name)
        intervals[interval_index] = Nasa9Interval{Float64}(Tmin, Tmax, a, b1)
    end
    return Nasa9Species{Float64}(name, R, intervals)
end

# Lines consumed by one record: the name line, the header, and three lines per
# interval, except that a zero-interval record (the REACTANTS section lists
# some species by heat of formation alone) still carries one temperature line.
# Getting this wrong desynchronizes the scan against the file, with no local
# failure, so it is written down once here.
_nasa9_record_lines(n_intervals) = n_intervals == 0 ? 3 : 2 + 3 * n_intervals

"""
    read_nasa9(names; path=NASA9_THERMO_PATH, reference=:sensible,
               T_ref=298.15)

Read gas species from the fixed-column NASA CEA `thermo.inp` database. The
specific gas constant is derived from each record's molar mass using
`R = R_universal/M`, with the tabulated molar mass converted from g/mol to
kg/mol; it is never entered independently of the coefficients.

# Arguments and keywords

- `names`: a species name or vector of names as written in the database. A
  vector returns a `Vector{Nasa9Species{Float64}}` in the requested order,
  including repeated names, and an empty vector returns an empty vector; a
  single name returns one [`Nasa9Species`](@ref). Where the database repeats a
  name, across phases or across the product/reactant split, the first record
  wins.
- `path`: database file. It defaults to the bundled `data/thermo.inp`; override
  it to read another file in the same fixed-column CEA format.
- `reference`: enthalpy gauge. The default `:sensible` shifts every interval's
  `b1` by the same constant so `h(T_ref) = 0`, reducing cancellation in
  non-reacting flow. Use `:formation` to preserve the tabulated gauge.
- `T_ref`: positive reference temperature for the sensible-enthalpy shift.
  Default: `298.15` K. It does not alter coefficients when
  `reference=:formation`, but it must still be positive.

The reader accepts gaseous records with one or more polynomial intervals and
throws an `ArgumentError` for missing species, malformed records, non-gaseous
records, or invalid keyword values.
"""
function read_nasa9(names::AbstractVector{<:AbstractString};
                    path::AbstractString=NASA9_THERMO_PATH,
                    reference=:sensible, T_ref=298.15)
    reference in (:sensible, :formation) ||
        throw(ArgumentError("reference must be :sensible or :formation"))
    T_ref > 0 || throw(ArgumentError("T_ref must be positive"))
    isfile(path) || throw(ArgumentError("NASA CEA thermo database not found: $path"))
    requested = String.(names)
    isempty(requested) && return Nasa9Species{Float64}[]
    wanted = Set(requested)
    found = Dict{String,Nasa9Species{Float64}}()
    lines = readlines(path)
    thermo_line = findfirst(line -> strip(line) == "thermo", lines)
    thermo_line === nothing &&
        throw(ArgumentError("NASA CEA thermo marker not found in $path"))

    i = thermo_line + 2
    while i <= length(lines)
        line = lines[i]
        stripped = strip(line)
        lowercase(stripped) == "end" && break
        if isempty(stripped) || startswith(stripped, "!") ||
           stripped in ("END PRODUCTS", "END REACTANTS")
            i += 1
            continue
        end
        name = String(strip(_nasa9_field(line, 1, 16, "species name")))
        i + 1 <= length(lines) ||
            throw(ArgumentError("truncated NASA CEA header for $name in $path"))
        header = lines[i + 1]
        n_intervals = _nasa9_int(_nasa9_field(header, 1, 2, name), name)
        # First match wins: CEA repeats a name across phases and across the
        # product/reactant split, and the gaseous fit is the earlier record.
        if name in wanted && !haskey(found, name)
            species = _nasa9_record(lines, i, header, n_intervals, name, path)
            found[name] = _nasa9_reference(species, reference, T_ref)
            length(found) == length(wanted) && break
        end
        i += _nasa9_record_lines(n_intervals)
    end

    absent = [name for name in unique(requested) if !haskey(found, name)]
    isempty(absent) ||
        throw(ArgumentError("NASA CEA species not found: " * join(absent, ", ")))
    return [found[name] for name in requested]
end

read_nasa9(name::AbstractString; kwargs...) = only(read_nasa9([name]; kwargs...))
