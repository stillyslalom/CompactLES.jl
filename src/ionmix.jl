# Tabulated equation of state and group opacities in the IONMIX4 and IONMIX6
# formats that FLASH reads, with bilinear interpolation in (ln T, ln ρ) and the
# temperature inversion along a density column. This is a standalone table
# model: no `EOS` method reads it, so nothing here enters the runtime flux.
#
# Format sources. The FLASH4 user's guide, "The IONMIX EOS/Opacity Format"
# (section 23.5.6 of the development guide,
# https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node149.html),
# lists the fields, their units and the Fortran write statements. The reader
# and writers of opacplot2,
# https://github.com/flash-center/opacplot2/blob/master/opacplot2/opg_ionmix.py,
# add the header variant with a log-spaced grid. In Fortran terms the layout is
#
#   (2i10)                                ntemp, ndens
#   (' atomic #s of gases: ',5i10)        atomic numbers (ignored by FLASH)
#   (' relative fractions: ',1p5e10.2)    their fractions by ion number
#   (i12)                                 ngroups; or, on the log-spaced grid,
#   (4e12.6,i12)                          Δlog10 n, log10 n₁, Δlog10 T,
#                                         log10 T₁, ngroups
#   then (4e12.6) blocks, each starting on a new line:
#     T [eV] (ntemp), n_ion [cm⁻³] (ndens)          explicit grid only
#     Z̄, dZ̄/dT [1/eV]
#     P_ion, P_ele [J/cm³], dP_ion/dT, dP_ele/dT [J/cm³/eV]
#     e_ion, e_ele [J/g], de_ion/dT, de_ele/dT [J/g/eV]
#     de_ion/dn_ion, de_ele/dn_ion [J/g per cm⁻³]
#     s_ele [J/g/eV]                                  IONMIX6 only
#     group boundaries [eV] (ngroups + 1)
#     Rosseland, Planck absorption, Planck emission opacities [cm²/g]
#
# A (ntemp, ndens) block varies temperature fastest, and the opacity blocks vary
# temperature, then density, then group: the column-major order of Julia arrays
# of those shapes. An E12.6 field whose exponent has three digits drops the E,
# as Fortran does ("0.123456+100"). The guide marks the units of the two de/dn
# fields as uncertain in its own writer; FLASH reads and ignores them, and so
# does the interpolation here, whose derivatives are those of the interpolant.

# Kelvin per electronvolt, from the exact SI values of e and k_B.
const _IONMIX_EV_KELVIN = 1.602176634e-19 / 1.380649e-23
const _IONMIX_EV_JOULE = 1.602176634e-19

# File unit → SI factor of each (ntemp, ndens) field, in file order.
const _IONMIX_FIELDS = (
    (:zbar, 1.0),
    (:dzbar_dT, 1 / _IONMIX_EV_KELVIN),
    (:p_ion, 1e6),                          # J/cm³ → Pa
    (:p_ele, 1e6),
    (:dp_ion_dT, 1e6 / _IONMIX_EV_KELVIN),
    (:dp_ele_dT, 1e6 / _IONMIX_EV_KELVIN),
    (:e_ion, 1e3),                          # J/g → J/kg
    (:e_ele, 1e3),
    (:cv_ion, 1e3 / _IONMIX_EV_KELVIN),
    (:cv_ele, 1e3 / _IONMIX_EV_KELVIN),
    (:de_ion_dn, 1e-3),                     # (J/g)/cm⁻³ → (J/kg)/m⁻³
    (:de_ele_dn, 1e-3),
)
const _IONMIX_ENTROPY_SCALE = 1e3 / _IONMIX_EV_KELVIN
const _IONMIX_DENSITY_SCALE = 1e6            # cm⁻³ → m⁻³
const _IONMIX_OPACITY_SCALE = 0.1            # cm²/g → m²/kg
const _IONMIX_OPACITIES = (:rosseland, :planck_absorption, :planck_emission)

"""
    IonmixTable{T}

A single-material equation-of-state and group-opacity table in the IONMIX4 or
IONMIX6 layout, held in SI units: temperature in K, ion number density in m⁻³,
mass density in kg/m³, pressure in Pa, specific energy in J/kg, heat capacity
and electron entropy in J/(kg K), group boundaries in J and opacities in m²/kg.
Temperatures on file are in eV and densities are ion number densities; the mass
density axis is `ion_density * ion_mass`, with `ion_mass` the mean mass per ion
in kg, which the file does not carry.

Fields:

- `format`: `:ionmix4`, or `:ionmix6` when the electron entropy `s_ele` is
  present (otherwise a 0×0 matrix).
- `atomic_numbers`, `fractions`: the header's elements and their fractions by
  ion number, as printed (three significant digits).
- `ion_mass`, `temperature`, `ion_density`, `density`, and `log_temperature`,
  `log_density` (natural logarithms, the interpolation coordinates).
- `log_grid`: the header's `(Δlog10 n, log10 n₁, Δlog10 T, log10 T₁)` in file
  units when the file uses the log-spaced grid, otherwise `nothing`.
- `extrapolate`: `:missing` or `:linear`, see [`table_value`](@ref).
- `(ntemp, ndens)` matrices: `zbar`, `dzbar_dT`, `p_ion`, `p_ele`, `dp_ion_dT`,
  `dp_ele_dT`, `e_ion`, `e_ele`, `cv_ion`, `cv_ele`, `de_ion_dn`, `de_ele_dn`,
  `s_ele`.
- `group_bounds` and the `(ntemp, ndens, ngroups)` arrays `rosseland`,
  `planck_absorption`, `planck_emission`.
- `monotone`: an `(ndens, 3)` matrix recording whether the total, ion and
  electron energies increase strictly with temperature along each density node.
  Between two such nodes [`table_temperature_status`](@ref) searches by
  bisection.

The tabulated derivative fields are carried so that a table round-trips; the
interpolation does not read them.

An IONMIX table is a one-temperature equilibrium model. Every entry, including
the split into ion and electron parts, is computed at `T_ion = T_ele`, and the
derivatives of its interpolant are equilibrium derivatives, taken with the
ionization balance following the temperature and density. The table contains
no frozen (fixed-ionization) derivative and no charge-state populations beyond
the mean `Z̄`, and it does not define a two-temperature equation of state:
evaluating `e_ion` and `e_ele` at different temperatures departs from the model
that produced them.

Build one with [`read_ionmix`](@ref), or from SI arrays with the
keyword constructor, whose derivative, entropy and opacity fields default to
zeros or to no groups:

    IonmixTable(; temperature, ion_density, ion_mass, zbar, p_ion, p_ele,
                  e_ion, e_ele, extrapolate=:missing, ...)
"""
struct IonmixTable{T<:AbstractFloat}
    format::Symbol
    atomic_numbers::Vector{Int}
    fractions::Vector{T}
    ion_mass::T
    temperature::Vector{T}
    ion_density::Vector{T}
    density::Vector{T}
    log_temperature::Vector{T}
    log_density::Vector{T}
    log_grid::Union{Nothing,NTuple{4,Float64}}
    extrapolate::Symbol
    zbar::Matrix{T}
    dzbar_dT::Matrix{T}
    p_ion::Matrix{T}
    p_ele::Matrix{T}
    dp_ion_dT::Matrix{T}
    dp_ele_dT::Matrix{T}
    e_ion::Matrix{T}
    e_ele::Matrix{T}
    cv_ion::Matrix{T}
    cv_ele::Matrix{T}
    de_ion_dn::Matrix{T}
    de_ele_dn::Matrix{T}
    s_ele::Matrix{T}
    group_bounds::Vector{T}
    rosseland::Array{T,3}
    planck_absorption::Array{T,3}
    planck_emission::Array{T,3}
    monotone::Matrix{Bool}
end

function IonmixTable(; temperature::AbstractVector, ion_density::AbstractVector,
                     ion_mass::Real, zbar, p_ion, p_ele, e_ion, e_ele,
                     dzbar_dT=nothing, dp_ion_dT=nothing, dp_ele_dT=nothing,
                     cv_ion=nothing, cv_ele=nothing, de_ion_dn=nothing,
                     de_ele_dn=nothing, s_ele=nothing, group_bounds=[0.0],
                     rosseland=nothing, planck_absorption=nothing,
                     planck_emission=nothing, atomic_numbers=Int[],
                     fractions=Float64[], extrapolate::Symbol=:missing)
    T = float(promote_type(eltype(temperature), eltype(ion_density), typeof(ion_mass)))
    nt, nd = length(temperature), length(ion_density)
    ng = length(group_bounds) - 1
    plane(A) = A === nothing ? zeros(T, nt, nd) : Matrix{T}(A)
    groups(A) = A === nothing ? zeros(T, nt, nd, max(ng, 0)) : Array{T,3}(A)
    fields = (plane(zbar), plane(dzbar_dT), plane(p_ion), plane(p_ele),
              plane(dp_ion_dT), plane(dp_ele_dT), plane(e_ion), plane(e_ele),
              plane(cv_ion), plane(cv_ele), plane(de_ion_dn), plane(de_ele_dn))
    format = s_ele === nothing ? :ionmix4 : :ionmix6
    entropy = s_ele === nothing ? zeros(T, 0, 0) : Matrix{T}(s_ele)
    return _ionmix_table(T, format, collect(Int, atomic_numbers), T.(fractions),
                         T(ion_mass), Vector{T}(temperature), Vector{T}(ion_density),
                         nothing, extrapolate, fields, entropy,
                         Vector{T}(group_bounds), groups(rosseland),
                         groups(planck_absorption), groups(planck_emission))
end

function _ionmix_table(::Type{T}, format, atomic_numbers, fractions, ion_mass,
                       temperature, ion_density, log_grid, extrapolate, fields,
                       s_ele, group_bounds, rosseland, planck_absorption,
                       planck_emission) where {T}
    extrapolate in (:missing, :linear) ||
        throw(ArgumentError("IonmixTable: extrapolate must be :missing or " *
                            ":linear, got :$extrapolate"))
    isfinite(ion_mass) && ion_mass > 0 ||
        throw(ArgumentError("IonmixTable: ion_mass must be a finite mass per " *
                            "ion > 0 in kg"))
    nt, nd = length(temperature), length(ion_density)
    for (name, axis) in (("temperature", temperature), ("ion_density", ion_density))
        length(axis) >= 2 ||
            throw(ArgumentError("IonmixTable: the $name axis must have at least two nodes"))
        # The interpolation is in the logarithm of both coordinates, so a node at
        # zero, which some converters add at the cold end, has no position.
        all(x -> isfinite(x) && x > 0, axis) ||
            throw(ArgumentError("IonmixTable: the $name axis must be finite and " *
                                "positive"))
        all(i -> axis[i] < axis[i+1], 1:length(axis)-1) ||
            throw(ArgumentError("IonmixTable: the $name axis must increase strictly"))
    end
    for (A, (name, _)) in zip(fields, _IONMIX_FIELDS)
        size(A) == (nt, nd) ||
            throw(DimensionMismatch("IonmixTable: $name is $(size(A)), expected " *
                                    "($nt, $nd)"))
    end
    format === :ionmix6 && size(s_ele) != (nt, nd) &&
        throw(DimensionMismatch("IonmixTable: s_ele is $(size(s_ele)), expected " *
                                "($nt, $nd)"))
    ng = length(group_bounds) - 1
    ng >= 0 || throw(ArgumentError("IonmixTable: group_bounds is empty"))
    # A table without groups written by opacplot2 carries the two boundaries
    # (0, 1) and no opacities; the group count is then zero, not one.
    ngroups = size(rosseland, 3)
    ngroups == ng || (ngroups == 0 && ng == 1) ||
        throw(DimensionMismatch("IonmixTable: $(length(group_bounds)) group " *
                                "boundaries for $ngroups groups"))
    for (A, name) in zip((rosseland, planck_absorption, planck_emission),
                         _IONMIX_OPACITIES)
        size(A) == (nt, nd, ngroups) ||
            throw(DimensionMismatch("IonmixTable: $name is $(size(A)), expected " *
                                    "($nt, $nd, $ngroups)"))
    end
    for A in (fields..., s_ele, group_bounds, rosseland, planck_absorption,
              planck_emission)
        all(isfinite, A) ||
            throw(ArgumentError("IonmixTable: table entries must be finite"))
    end
    density = ion_density .* ion_mass
    e_total = fields[7] .+ fields[8]
    energies = (e_total, fields[7], fields[8])
    monotone = Bool[_table_increasing(E, j) for j in 1:nd, E in energies]
    return IonmixTable{T}(format, atomic_numbers, fractions, ion_mass, temperature,
                          ion_density, density, log.(temperature), log.(density),
                          log_grid, extrapolate, fields..., s_ele, group_bounds,
                          rosseland, planck_absorption, planck_emission, monotone)
end

_table_increasing(E::AbstractMatrix, j::Int) = all(i -> E[i, j] < E[i+1, j], 1:size(E, 1)-1)

function Base.show(io::IO, table::IonmixTable{T}) where {T}
    print(io, "IonmixTable{$T}(:", table.format, ", ", length(table.temperature),
          " temperatures ", @sprintf("%.4g", table.temperature[1]), "–",
          @sprintf("%.4g", table.temperature[end]), " K, ", length(table.density),
          " densities ", @sprintf("%.4g", table.density[1]), "–",
          @sprintf("%.4g", table.density[end]), " kg/m³, ",
          size(table.rosseland, 3), " groups, extrapolate=:", table.extrapolate, ")")
end

# --- Reading and writing ------------------------------------------------------

# One Fortran E-format field: optional leading zero, D or E exponent marker, and
# a three-digit exponent written without its marker.
function _fortran_float(field::AbstractString)
    text = replace(strip(field), 'D' => 'E', 'd' => 'E')
    if !occursin('E', text) && !occursin('e', text)
        k = findlast(c -> c == '+' || c == '-', text)
        k !== nothing && k > 1 && (text = string(text[1:k-1], 'E', text[k:end]))
    end
    value = tryparse(Float64, text)
    value === nothing &&
        throw(ArgumentError("IONMIX: cannot read \"$field\" as a number"))
    return value
end

# A value as Fortran's E12.6 writes it: 0.dddddd with a two-digit exponent, or
# three digits and no E, and "-." for a negative value.
function _fortran_e12(value::Real)
    v = Float64(value)
    isfinite(v) || throw(ArgumentError("IONMIX: cannot write the value $value"))
    v == 0 && return "0.000000E+00"
    s = @sprintf("%.5E", abs(v))                  # d.dddddE±xx
    k = findfirst('E', s)
    mantissa = string(s[1], s[3:k-1])
    exponent = parse(Int, s[k+1:end]) + 1
    lead = v < 0 ? "-." : "0."
    sign = exponent < 0 ? '-' : '+'
    abs(exponent) < 100 && return string(lead, mantissa, 'E', sign,
                                         lpad(abs(exponent), 2, '0'))
    abs(exponent) < 1000 && return string(lead, mantissa, sign, abs(exponent))
    throw(ArgumentError("IONMIX: $value is outside the E12.6 exponent range"))
end

# The data section as one stream of fixed-width 12-character fields, four to a
# line. Fortran reads each block from the start of a line and a short last line
# pads with blanks, so a blank field ends its line.
function _ionmix_fields(lines)
    values = Float64[]
    for line in lines
        text = rstrip(line)
        isascii(text) || throw(ArgumentError("IONMIX: non-ASCII data line"))
        for start in 1:12:length(text)
            field = text[start:min(start + 11, length(text))]
            isempty(strip(field)) && break
            push!(values, _fortran_float(field))
        end
    end
    return values
end

_after_label(line) = (k = findfirst(':', line); k === nothing ? line : line[k+1:end])

"""
    read_ionmix([T=Float64,] path; ion_mass, format=:auto, extrapolate=:missing)
        -> IonmixTable{T}

Read an IONMIX4 or IONMIX6 equation-of-state and opacity file (the `.cn4` files
FLASH reads) into an [`IonmixTable`](@ref), converting every field to SI.

- `ion_mass` is the mean mass per ion in kg, required because the file tabulates
  ion number density and not mass density. For a single element it is the
  atomic mass; for a mixture it is the fraction-weighted mean.
- `format` is `:auto`, `:ionmix4` or `:ionmix6`. The two differ only by the
  electron-entropy block, so `:auto` identifies the file from its value count,
  and an explicit format is checked against it.
- `extrapolate` sets the policy outside the table's axes, as described under
  [`table_value`](@ref). The default `:missing` reports every such
  query as out of domain.

Both header variants are read: the explicit grid FLASH documents, which lists
the temperatures and densities, and the log-spaced grid, which gives the first
value and the step of each axis in log10. The single-temperature IONMIX layout
with four fields per grid point is recognized and rejected. Axes must be
positive and increasing, since the interpolation is in their logarithms. The
cost is one pass over the file.
"""
read_ionmix(path::AbstractString; kwargs...) = read_ionmix(Float64, path; kwargs...)

function read_ionmix(::Type{T}, path::AbstractString; ion_mass::Real,
                     format::Symbol=:auto,
                     extrapolate::Symbol=:missing) where {T<:AbstractFloat}
    format in (:auto, :ionmix4, :ionmix6) ||
        throw(ArgumentError("read_ionmix: format must be :auto, :ionmix4 or " *
                            ":ionmix6, got :$format"))
    lines = readlines(path)
    length(lines) >= 4 || throw(ArgumentError("IONMIX: $path has no header"))
    counts = split(lines[1])
    length(counts) >= 2 ||
        throw(ArgumentError("IONMIX: the first line must give ntemp and ndens"))
    nt, nd = parse(Int, counts[1]), parse(Int, counts[2])
    atomic_numbers = [parse(Int, x) for x in split(_after_label(lines[2]))]
    fractions = [_fortran_float(x) for x in split(_after_label(lines[3]))]
    grid_line = rstrip(lines[4])
    log_grid = nothing
    if length(split(grid_line)) == 1
        ngroups = parse(Int, grid_line)
    else
        length(grid_line) >= 49 ||
            throw(ArgumentError("IONMIX: cannot read the grid line \"$grid_line\""))
        log_grid = ntuple(k -> _fortran_float(grid_line[12k-11:12k]), 4)
        ngroups = parse(Int, grid_line[49:end])
    end
    stream = _ionmix_fields(@view lines[5:end])
    n = nt * nd
    head = log_grid === nothing ? nt + nd : 0
    opacity_count(bounds) = bounds + 3 * ngroups * n
    per_point = nothing
    for K in (12, 13), bounds in (ngroups + 1, ngroups == 0 ? 2 : -1)
        bounds > 0 && length(stream) == head + K * n + opacity_count(bounds) &&
            (per_point = (K, bounds); break)
    end
    if per_point === nothing
        single = length(stream) == head + 4 * n + opacity_count(ngroups + 1)
        throw(ArgumentError(single ?
            "IONMIX: $path has four fields per grid point, the single-temperature " *
            "layout, which read_ionmix does not read" :
            "IONMIX: $path holds $(length(stream)) values, which matches neither " *
            "IONMIX4 nor IONMIX6 for $nt temperatures, $nd densities and " *
            "$ngroups groups"))
    end
    K, bounds = per_point
    detected = K == 13 ? :ionmix6 : :ionmix4
    format === :auto || format === detected ||
        throw(ArgumentError("IONMIX: $path is $detected, not $format"))
    position = Ref(0)
    function take(count)
        block = @view stream[position[]+1:position[]+count]
        position[] += count
        return block
    end
    if log_grid === nothing
        temperature_eV = collect(take(nt))
        density_cm3 = collect(take(nd))
    else
        dn, n1, dT, T1 = log_grid
        temperature_eV = [10.0^(T1 + dT * (k - 1)) for k in 1:nt]
        density_cm3 = [10.0^(n1 + dn * (k - 1)) for k in 1:nd]
    end
    fields = Tuple(reshape(T.(take(n) .* scale), nt, nd) for (_, scale) in _IONMIX_FIELDS)
    s_ele = K == 13 ? reshape(T.(take(n) .* _IONMIX_ENTROPY_SCALE), nt, nd) :
                      zeros(T, 0, 0)
    group_bounds = T.(take(bounds) .* _IONMIX_EV_JOULE)
    opacity() = reshape(T.(take(ngroups * n) .* _IONMIX_OPACITY_SCALE), nt, nd, ngroups)
    rosseland = opacity()
    planck_absorption = opacity()
    planck_emission = opacity()
    return _ionmix_table(T, detected, atomic_numbers, T.(fractions), T(ion_mass),
                         T.(temperature_eV .* _IONMIX_EV_KELVIN),
                         T.(density_cm3 .* _IONMIX_DENSITY_SCALE), log_grid,
                         extrapolate, fields, s_ele, group_bounds, rosseland,
                         planck_absorption, planck_emission)
end

"""
    write_ionmix(path, table::IonmixTable)

Write `table` in its own format (`table.format`) and header variant, converting
back to the file's units. Each value is written as Fortran's E12.6, so a value
keeps six significant digits; reading the file back and writing it again
reproduces the file exactly. The cost is one pass over the table.
"""
function write_ionmix(path::AbstractString, table::IonmixTable)
    nt, nd = length(table.temperature), length(table.ion_density)
    ngroups = size(table.rosseland, 3)
    open(path, "w") do io
        @printf(io, "%10d%10d\n", nt, nd)
        print(io, " atomic #s of gases: ")
        foreach(z -> @printf(io, "%10d", z), table.atomic_numbers)
        print(io, "\n relative fractions: ")
        foreach(f -> @printf(io, "%10.2E", f), table.fractions)
        print(io, "\n")
        if table.log_grid === nothing
            @printf(io, "%12d\n", ngroups)
        else
            foreach(x -> print(io, _fortran_e12(x)), table.log_grid)
            @printf(io, "%12d\n", ngroups)
        end
        function block(values, scale)
            count = 0
            for v in values
                print(io, _fortran_e12(v / scale))
                count += 1
                count % 4 == 0 && print(io, "\n")
            end
            count % 4 == 0 || print(io, "\n")
        end
        if table.log_grid === nothing
            block(table.temperature, _IONMIX_EV_KELVIN)
            block(table.ion_density, _IONMIX_DENSITY_SCALE)
        end
        for (name, scale) in _IONMIX_FIELDS
            block(getfield(table, name), scale)
        end
        table.format === :ionmix6 && block(table.s_ele, _IONMIX_ENTROPY_SCALE)
        block(table.group_bounds, _IONMIX_EV_JOULE)
        for name in _IONMIX_OPACITIES
            block(getfield(table, name), _IONMIX_OPACITY_SCALE)
        end
    end
    return path
end

# --- Interpolation --------------------------------------------------------------

"""
Status flags returned beside a table query by [`table_value`](@ref),
[`table_state`](@ref) and [`table_temperature_status`](@ref).
`TABLE_OK` is the zero value; the others are independent bits.

- `TABLE_EXTRAPOLATED`: the point lies outside an axis and the value is the
  linear extension of the edge cell, under `extrapolate = :linear`.
- `TABLE_OUT_OF_DOMAIN`: the same under `extrapolate = :missing`, which makes
  the point inadmissible. The value is still the linear extension, so that a
  calculation in progress can complete and the point can be inspected. A
  temperature or density that is not finite and positive, and a non-finite
  energy, are always out of domain, and their value is `NaN`.
- `TABLE_TEMPERATURE_AXIS`, `TABLE_DENSITY_AXIS`: which axis was left.
- `TABLE_NOT_MONOTONE`: the inversion found the energy not increasing strictly
  with temperature along the density column, so the temperature returned (the
  lowest root) may not be the only one.
- `TABLE_UNSTABLE`: the interpolated heat capacity or squared sound speed is
  not positive, as in the two-phase region of a table without a Maxwell
  construction.
"""
const TABLE_OK = 0x00
const TABLE_EXTRAPOLATED = 0x01
const TABLE_OUT_OF_DOMAIN = 0x02
const TABLE_TEMPERATURE_AXIS = 0x04
const TABLE_DENSITY_AXIS = 0x08
const TABLE_NOT_MONOTONE = 0x10
const TABLE_UNSTABLE = 0x20

@inline _table_policy(table::IonmixTable) =
    table.extrapolate === :missing ? TABLE_OUT_OF_DOMAIN : TABLE_EXTRAPOLATED

# The cell of `q` on a strictly increasing axis and its fractional position in
# it, below 0 or above 1 outside the axis, where the edge cell is extended. The
# fraction is exactly 0 or 1 on a node. The domain test allows a relative margin
# (the axes are logarithms) of the tolerance `mixture_temperature_status`
# converges to, so that a temperature recovered at the edge node is not
# reported as extrapolated.
@inline function _axis_position(axis::AbstractVector, q)
    n = length(axis)
    i = clamp(searchsortedlast(axis, q), 1, n - 1)
    @inbounds lo, hi = axis[i], axis[i+1]
    margin = _nasa9_rtol(typeof(lo))
    @inbounds outside = !(axis[1] - margin <= q <= axis[n] + margin)
    return i, (q - lo) / (hi - lo), hi - lo, outside
end

# The position of a (T_ion, ρ) query in the table: the cell (i, j), the
# fractions (s, r) along ln T and ln ρ, the cell widths, and the status.
struct TableLocation{T}
    i::Int
    j::Int
    s::T
    r::T
    dx::T
    dy::T
    T_ion::T
    rho::T
    status::UInt8
end

@inline function _table_location(table::IonmixTable{T}, T_ion, rho) where {T}
    Tq, ρq = T(T_ion), T(rho)
    status = TABLE_OK
    policy = _table_policy(table)
    # log throws on a negative argument; a NaN coordinate carries through to
    # every value instead, and the point is reported whatever the policy.
    valid_T = isfinite(Tq) && Tq > 0
    valid_ρ = isfinite(ρq) && ρq > 0
    x = valid_T ? log(Tq) : T(NaN)
    y = valid_ρ ? log(ρq) : T(NaN)
    i, s, dx, outside_T = _axis_position(table.log_temperature, x)
    j, r, dy, outside_ρ = _axis_position(table.log_density, y)
    outside_T && (status |= (valid_T ? policy : TABLE_OUT_OF_DOMAIN) |
                            TABLE_TEMPERATURE_AXIS)
    outside_ρ && (status |= (valid_ρ ? policy : TABLE_OUT_OF_DOMAIN) |
                            TABLE_DENSITY_AXIS)
    return TableLocation{T}(i, j, s, r, dx, dy, Tq, ρq, status)
end

# The sum of two tabulated fields, node by node, so that a total is interpolated
# with the same arithmetic in the forward query and in the inversion.
struct _SumField{M}
    a::M
    b::M
end
Base.@propagate_inbounds Base.getindex(f::_SumField, i::Int, j::Int) = f.a[i, j] + f.b[i, j]

# One group of an opacity array.
struct _GroupField{A}
    data::A
    group::Int
end
Base.@propagate_inbounds Base.getindex(f::_GroupField, i::Int, j::Int) =
    f.data[i, j, f.group]

# The bilinear interpolant in (ln T, ln ρ) and its exact partial derivatives.
# The weighted form (1 - s)a + s b, rather than a + s(b - a), returns the node
# value exactly at s = 0 and at s = 1.
@inline function _table_bilinear(loc::TableLocation, F)
    i, j, s, r = loc.i, loc.j, loc.s, loc.r
    @inbounds f00, f10, f01, f11 = F[i, j], F[i+1, j], F[i, j+1], F[i+1, j+1]
    lo = (1 - s) * f00 + s * f10
    hi = (1 - s) * f01 + s * f11
    value = (1 - r) * lo + r * hi
    df_dx = ((1 - r) * (f10 - f00) + r * (f11 - f01)) / loc.dx
    df_dy = (hi - lo) / loc.dy
    return value, df_dx / loc.T_ion, df_dy / loc.rho
end

const _TABLE_FIELDS = (:zbar, :dzbar_dT, :p_ion, :p_ele, :dp_ion_dT, :dp_ele_dT,
                       :e_ion, :e_ele, :cv_ion, :cv_ele, :de_ion_dn, :de_ele_dn,
                       :s_ele)

"""
    table_value(table, field, T_ion, rho) -> (f, ∂f/∂T_ion, ∂f/∂rho, status)

Interpolate one field of an [`IonmixTable`](@ref) at temperature `T_ion` (K)
and mass density `rho` (kg/m³). `field` is a `Symbol` naming a tabulated
`(ntemp, ndens)` field (`:zbar`, `:p_ion`, `:e_ele`, ...) or one of the totals
`:p = p_ion + p_ele` and `:e = e_ion + e_ele`; an array of that shape may be
passed in its place. The derivatives are at fixed `rho` and fixed `T_ion`, and
`status` is a combination of the [`TABLE_OK`](@ref) flags.

The interpolant is bilinear in `(ln T_ion, ln rho)`, linear in the field value,
and reproduces the table at its nodes exactly. It is continuous, and its first
derivatives, which are the exact derivatives of the interpolant and not
separately interpolated tabulated derivatives, jump across cell edges; on an
interior node they are those of the cell above. The field value is interpolated
linearly because energies may be negative and `Z̄` may vanish.

Interpolating `e` and `p` independently does not make them satisfy the
thermodynamic consistency relation `rho² ∂e/∂rho = p - T ∂p/∂T` between
nodes, which holds exactly only when both derive from one interpolated free
energy; the table carries none.

Outside the table's axes the edge cell is extended linearly in the logarithms,
never clamped, and the point is reported. `extrapolate = :missing` (the
default) reports it as `TABLE_OUT_OF_DOMAIN`; `:linear` reports it as
`TABLE_EXTRAPOLATED`, a statement that the extension is acceptable for the
calculation. The cost is two binary searches and four table reads.
"""
@inline function table_value(table::IonmixTable, F::AbstractMatrix, T_ion, rho)
    size(F) == size(table.zbar) ||
        throw(DimensionMismatch("table_value: field is $(size(F)), table is " *
                                "$(size(table.zbar))"))
    loc = _table_location(table, T_ion, rho)
    return (_table_bilinear(loc, F)..., loc.status)
end

function table_value(table::IonmixTable{T}, field::Symbol, T_ion, rho) where {T}
    field === :e && return _table_value(table, _SumField(table.e_ion, table.e_ele),
                                        T_ion, rho)
    field === :p && return _table_value(table, _SumField(table.p_ion, table.p_ele),
                                        T_ion, rho)
    field in _TABLE_FIELDS ||
        throw(ArgumentError("table_value: unknown field :$field"))
    return table_value(table, getfield(table, field)::Matrix{T}, T_ion, rho)
end

@inline function _table_value(table::IonmixTable, F, T_ion, rho)
    loc = _table_location(table, T_ion, rho)
    return (_table_bilinear(loc, F)..., loc.status)
end

"""
    table_opacity(table, kind, group, T_ion, rho) -> (κ, ∂κ/∂T_ion, ∂κ/∂rho, status)

Interpolate the group opacity (m²/kg) of `kind` `:rosseland`,
`:planck_absorption` or `:planck_emission` in energy group `group`, with the
interpolant, derivatives and domain policy of [`table_value`](@ref).
"""
function table_opacity(table::IonmixTable{T}, kind::Symbol, group::Integer, T_ion,
                       rho) where {T}
    kind in _IONMIX_OPACITIES ||
        throw(ArgumentError("table_opacity: kind must be :rosseland, " *
                            ":planck_absorption or :planck_emission, got :$kind"))
    data = getfield(table, kind)::Array{T,3}
    1 <= group <= size(data, 3) ||
        throw(BoundsError(data, (:, :, group)))
    return _table_value(table, _GroupField(data, Int(group)), T_ion, rho)
end

"""
    table_state(table, T_ion, rho) -> NamedTuple

The one-temperature equilibrium state of an [`IonmixTable`](@ref) at
`(T_ion, rho)`, all from the interpolant of [`table_value`](@ref):
the total pressure `p`, specific energy `e` and `zbar`; the heat capacity
`cv = ∂e/∂T` and `dp_dT` at fixed density; `dp_drho` and `de_drho` at fixed
temperature; the squared equilibrium sound speed
`c2 = dp_drho + T dp_dT² / (rho² cv)`; and `status`, which adds
`TABLE_UNSTABLE` to the location's flags when `cv` or `c2` is not positive.
Every derivative is an equilibrium one. The cost is three interpolations.
"""
function table_state(table::IonmixTable, T_ion, rho)
    loc = _table_location(table, T_ion, rho)
    p, dp_dT, dp_drho = _table_bilinear(loc, _SumField(table.p_ion, table.p_ele))
    e, cv, de_drho = _table_bilinear(loc, _SumField(table.e_ion, table.e_ele))
    zbar, _, _ = _table_bilinear(loc, table.zbar)
    Tq, ρq = loc.T_ion, loc.rho
    c2 = dp_drho + Tq * dp_dT^2 / (ρq^2 * cv)
    status = loc.status
    cv > 0 && c2 > 0 || (status |= TABLE_UNSTABLE)
    return (; p, e, zbar, cv, dp_dT, dp_drho, de_drho, c2, status)
end

# --- Temperature inversion ------------------------------------------------------

"""
    table_temperature_status(table, e, rho, component=:total) -> (T_ion, status)

Invert the interpolated specific energy of an [`IonmixTable`](@ref) for the
temperature at mass density `rho`: `component` `:total` inverts `e_ion + e_ele`,
`:ion` and `:electron` one part (at the table's one temperature). The inverse
is that of the interpolant [`table_value`](@ref) evaluates, so a forward
query at the returned temperature reproduces `e` to round-off.

At fixed `rho` the interpolant is linear in `ln T` within each cell, so the
iteration reduces to bracketing followed by one Newton step, which is exact:
the bracketing cell is found by bisection on the energies of the interpolated
density column, and the root is placed in it by linear interpolation in `ln T`.
An energy equal to a node value returns the node's temperature exactly. When
both neighbouring density nodes increase strictly with temperature (recorded
in `table.monotone` when the table is built) the column does too, and the
root is unique; otherwise, and whenever `rho` lies outside the density axis,
every cell of the column is visited, the lowest root is returned, and a column
that does not increase strictly is reported as `TABLE_NOT_MONOTONE`.

An energy outside the column's range is placed on the linear extension of the
edge cell and reported under the table's `extrapolate` policy with
`TABLE_TEMPERATURE_AXIS`, provided that cell's energy increases; otherwise the
result is `NaN`, out of domain. The cost is a density search and a bisection
over the temperature nodes, or one pass over them.
"""
function table_temperature_status(table::IonmixTable, e, rho,
                                  component::Symbol=:total)
    component === :total &&
        return _table_temperature(table, _SumField(table.e_ion, table.e_ele), 1, e, rho)
    component === :ion && return _table_temperature(table, table.e_ion, 2, e, rho)
    component === :electron && return _table_temperature(table, table.e_ele, 3, e, rho)
    throw(ArgumentError("table_temperature_status: component must be :total, :ion " *
                        "or :electron, got :$component"))
end

"""
    table_temperature(table, e, rho, component=:total) -> T_ion

The value-only form of [`table_temperature_status`](@ref).
"""
table_temperature(table::IonmixTable, e, rho, component::Symbol=:total) =
    table_temperature_status(table, e, rho, component)[1]

@inline function _table_temperature(table::IonmixTable{T}, E, k::Int, e, rho) where {T}
    eq, ρq = T(e), T(rho)
    nan = T(NaN)
    isfinite(ρq) && ρq > 0 || return (nan, TABLE_OUT_OF_DOMAIN | TABLE_DENSITY_AXIS)
    isfinite(eq) || return (nan, TABLE_OUT_OF_DOMAIN)
    policy = _table_policy(table)
    j, r, _, outside_ρ = _axis_position(table.log_density, log(ρq))
    status = outside_ρ ? (policy | TABLE_DENSITY_AXIS) : TABLE_OK
    x = table.log_temperature
    nt = length(x)
    column(i) = @inbounds (1 - r) * E[i, j] + r * E[i, j+1]
    e_first, e_last = column(1), column(nt)
    root = 0
    if 0 <= r <= 1 && table.monotone[j, k] && table.monotone[j+1, k]
        # Both density nodes increase, so their convex combination does too.
        if e_first <= eq <= e_last
            lo, hi = 1, nt      # column(lo) <= e <= column(hi)
            while hi - lo > 1
                mid = (lo + hi) >>> 1
                column(mid) <= eq ? (lo = mid) : (hi = mid)
            end
            root = lo
        end
    else
        increasing = true
        below = column(1)
        for i in 1:nt-1
            above = column(i + 1)
            increasing &= above > below
            if root == 0 && below != above && min(below, above) <= eq <= max(below, above)
                root = i
            end
            below = above
        end
        increasing || (status |= TABLE_NOT_MONOTONE)
    end
    if root == 0
        # Beyond the column's range: extend the edge cell on the side of e.
        root = eq < e_first ? 1 : nt - 1
        column(root + 1) > column(root) ||
            return (nan, status | TABLE_OUT_OF_DOMAIN | TABLE_TEMPERATURE_AXIS)
        status |= policy | TABLE_TEMPERATURE_AXIS
        a, b = column(root), column(root + 1)
        t = (eq - a) / (b - a)
    else
        a, b = column(root), column(root + 1)
        t = clamp((eq - a) / (b - a), zero(T), one(T))
    end
    t == 0 && return (table.temperature[root], status)
    t == 1 && return (table.temperature[root+1], status)
    @inbounds return (exp((1 - t) * x[root] + t * x[root+1]), status)
end
