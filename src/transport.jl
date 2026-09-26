const CEA_TRANSPORT_PATH = normpath(joinpath(@__DIR__, "..", "data", "trans.inp"))

struct CeaTransportInterval{T}
    Tmin::T
    Tmax::T
    A::T
    B::T
    C::T
    D::T
end

struct CeaTransportRecord{T}
    names::Tuple{String,String}
    viscosity::Vector{CeaTransportInterval{T}}
    conductivity::Vector{CeaTransportInterval{T}}
end

struct UniformDiffusivity{T}
    value::T
end
@inline Base.getindex(D::UniformDiffusivity, ::Integer) = D.value

function _cea_transport_number(field, context)
    s = replace(strip(String(field)), r"([Ee])\s+(\d+)" => s"\1+\2")
    try
        return parse(Float64, s)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("invalid CEA transport number '$s' for $context"))
    end
end

function _cea_transport_interval(line, context)
    ncodeunits(line) >= 80 || throw(ArgumentError("short CEA transport record for $context"))
    f(a, b) = _cea_transport_number(SubString(line, a, b), context)
    return CeaTransportInterval(f(3, 10), f(11, 20), f(21, 35), f(36, 50),
                                f(51, 65), f(66, 80))
end

"""
    read_cea_transport([path])

Read the pure-species and binary-interaction fits in a NASA CEA `trans.inp`
file. The returned records preserve both names for binary entries. Binary
entries are viscosity interaction fits; they are not binary diffusion data.
"""
function read_cea_transport(path::AbstractString=CEA_TRANSPORT_PATH)
    isfile(path) || throw(ArgumentError("NASA CEA transport database not found: $path"))
    lines = readlines(path)
    records = CeaTransportRecord{Float64}[]
    i = 2
    while i <= length(lines)
        header = lines[i]
        isempty(strip(header)) && (i += 1; continue)
        lowercase(strip(header)) == "end" && break
        length(header) >= 38 || throw(ArgumentError("invalid CEA transport header at line $i"))
        name1 = String(strip(String(SubString(header, 1, 16))))
        name2 = String(strip(String(SubString(header, 17, 34))))
        counts = match(r"V(\d)C(\d)", String(SubString(header, 35)))
        counts === nothing && throw(ArgumentError("invalid CEA transport header at line $i"))
        nv, nc = parse(Int, counts.captures[1]), parse(Int, counts.captures[2])
        i + nv + nc <= length(lines) || throw(ArgumentError("truncated CEA transport record for $name1"))
        visc = CeaTransportInterval{Float64}[]
        cond = CeaTransportInterval{Float64}[]
        for j in 1:nv
            line = lines[i + j]
            stripped = strip(line)
            !isempty(stripped) && first(stripped) == 'V' ||
                throw(ArgumentError("expected viscosity record for $name1 at line $(i + j)"))
            push!(visc, _cea_transport_interval(line, name1))
        end
        for j in 1:nc
            line = lines[i + nv + j]
            stripped = strip(line)
            !isempty(stripped) && first(stripped) == 'C' ||
                throw(ArgumentError("expected conductivity record for $name1 at line $(i + nv + j)"))
            push!(cond, _cea_transport_interval(line, name1))
        end
        push!(records, CeaTransportRecord((name1, name2), visc, cond))
        i += 1 + nv + nc
    end
    return records
end

"""
    BinaryDiffusion(D_ref; temperature_ref=300, pressure_ref=101325,
                    temperature_exponent=1.75)

Binary gas diffusion coefficients at a reference temperature and pressure.
Off-diagonal entries of `D_ref` are symmetric and positive, in m^2/s. Pressure
is in Pa and temperature is in K. Runtime
values scale as `(T/temperature_ref)^temperature_exponent * pressure_ref/p`.
"""
struct BinaryDiffusion{T,N}
    D_ref::NTuple{N,NTuple{N,T}}
    temperature_ref::T
    pressure_ref::T
    temperature_exponent::T
end

function BinaryDiffusion(D_ref::AbstractMatrix{<:Real}; temperature_ref=300.0,
                         pressure_ref=101325.0, temperature_exponent=1.75)
    size(D_ref, 1) == size(D_ref, 2) ||
        throw(ArgumentError("BinaryDiffusion reference matrix must be square"))
    N = size(D_ref, 1)
    N > 0 || throw(ArgumentError("BinaryDiffusion reference matrix must be nonempty"))
    T = promote_type(eltype(D_ref), typeof(float(temperature_ref)),
                     typeof(float(pressure_ref)), typeof(float(temperature_exponent)))
    isfinite(temperature_ref) && temperature_ref > 0 ||
        throw(ArgumentError("temperature_ref must be finite and positive"))
    isfinite(pressure_ref) && pressure_ref > 0 ||
        throw(ArgumentError("pressure_ref must be finite and positive"))
    isfinite(temperature_exponent) || throw(ArgumentError("temperature_exponent must be finite"))
    all(isfinite, D_ref) || throw(ArgumentError("binary diffusion matrix must be finite"))
    for i in 1:N, j in i+1:N
        isfinite(D_ref[i, j]) && D_ref[i, j] > 0 ||
            throw(ArgumentError("binary D_ref[$i,$j] must be finite and positive"))
        isapprox(D_ref[i, j], D_ref[j, i]; rtol=8eps(T), atol=zero(T)) ||
            throw(ArgumentError("binary diffusion matrix must be symmetric"))
    end
    values = ntuple(i -> ntuple(j -> i == j ? zero(T) : T(D_ref[i, j]), Val(N)), Val(N))
    return BinaryDiffusion{T,N}(values, T(temperature_ref), T(pressure_ref),
                                T(temperature_exponent))
end

@doc raw"""
    BinaryDiffusionPolynomial(species, D_ref, temperature_coefficients,
                              temperature_min, temperature_max;
                              temperature_ref=300, pressure_ref=101325)

Pair-specific, dilute-neutral-gas binary-diffusion fits. `species` gives the
exact species identity and order of the matrices.  For each off-diagonal pair,
the evaluator uses

```math
D_{ij}(T,p) = D_{ij,ref}\exp\left(\sum_{m=1}^{M} c_{ij,m}
                 [\log(T/T_{ref})]^m\right)\frac{p_{ref}}{p},
```

where `temperature_coefficients[i,j,m]` is ``c_{ij,m}``.  `D_ref` is in
m^2/s, temperature is in K, and pressure is in Pa.  The input ranges are
pair-specific and inclusive; `temperature_ref` must lie in every pair range.
Use [`binary_diffusivity`](@ref) for checked evaluation.

This model is for a supplied dilute neutral-gas correlation only. It supplies
no isotope data or plasma closure, and it does not model ambipolar, thermo-,
electro-, or pressure diffusion. It can supply the pair data for
[`CeaTransport`](@ref)'s mixture-averaged closure when its ordered species list
exactly matches the EOS.
""" BinaryDiffusionPolynomial
struct BinaryDiffusionPolynomial{T,N,M,Names}
    D_ref::NTuple{N,NTuple{N,T}}
    temperature_coefficients::NTuple{N,NTuple{N,NTuple{M,T}}}
    temperature_min::NTuple{N,NTuple{N,T}}
    temperature_max::NTuple{N,NTuple{N,T}}
    temperature_ref::T
    pressure_ref::T
end

function BinaryDiffusionPolynomial(species, D_ref::AbstractMatrix{<:Real},
                                   temperature_coefficients::AbstractArray{<:Real,3},
                                   temperature_min::AbstractMatrix{<:Real},
                                   temperature_max::AbstractMatrix{<:Real};
                                   temperature_ref=300.0, pressure_ref=101325.0)
    species isa AbstractString &&
        throw(ArgumentError("BinaryDiffusionPolynomial species must be an ordered tuple or vector of names, not one string"))
    (species isa Tuple || species isa AbstractVector) ||
        throw(ArgumentError("BinaryDiffusionPolynomial species must be an ordered tuple or vector of strings or symbols"))
    all(name -> name isa AbstractString || name isa Symbol, species) ||
        throw(ArgumentError("BinaryDiffusionPolynomial species names must be strings or symbols"))
    names = Tuple(String(name) for name in species)
    N = length(names)
    N > 0 || throw(ArgumentError("BinaryDiffusionPolynomial species must be nonempty"))
    all(name -> !isempty(strip(name)), names) ||
        throw(ArgumentError("BinaryDiffusionPolynomial species names must not be blank"))
    length(unique(names)) == N ||
        throw(ArgumentError("BinaryDiffusionPolynomial species names must be unique"))
    size(D_ref) == (N, N) ||
        throw(ArgumentError("BinaryDiffusionPolynomial D_ref must be N by N in species order"))
    size(temperature_min) == (N, N) && size(temperature_max) == (N, N) ||
        throw(ArgumentError("BinaryDiffusionPolynomial temperature ranges must be N by N in species order"))
    size(temperature_coefficients, 1) == N && size(temperature_coefficients, 2) == N ||
        throw(ArgumentError("BinaryDiffusionPolynomial coefficients must be N by N by M in species order"))
    M = size(temperature_coefficients, 3)
    M > 0 || throw(ArgumentError("BinaryDiffusionPolynomial requires at least one temperature coefficient"))
    T = promote_type(eltype(D_ref), eltype(temperature_coefficients), eltype(temperature_min),
                     eltype(temperature_max), typeof(float(temperature_ref)),
                     typeof(float(pressure_ref)))
    isfinite(temperature_ref) && temperature_ref > 0 ||
        throw(ArgumentError("temperature_ref must be finite and positive"))
    isfinite(pressure_ref) && pressure_ref > 0 ||
        throw(ArgumentError("pressure_ref must be finite and positive"))
    temperature_ref_T = T(temperature_ref)
    pressure_ref_T = T(pressure_ref)
    isfinite(temperature_ref_T) && temperature_ref_T > zero(T) ||
        throw(ArgumentError("temperature_ref is not representable as a finite positive coefficient"))
    isfinite(pressure_ref_T) && pressure_ref_T > zero(T) ||
        throw(ArgumentError("pressure_ref is not representable as a finite positive coefficient"))
    for i in 1:N, j in i+1:N
        Dij, Dji = D_ref[i, j], D_ref[j, i]
        isfinite(Dij) && Dij > 0 ||
            throw(ArgumentError("binary D_ref[$i,$j] must be finite and positive"))
        isfinite(Dji) && Dji > 0 && isapprox(Dij, Dji; rtol=8eps(T), atol=zero(T)) ||
            throw(ArgumentError("binary D_ref must be symmetric"))
        isfinite(T(Dij)) && T(Dij) > zero(T) && isfinite(T(Dji)) && T(Dji) > zero(T) ||
            throw(ArgumentError("binary D_ref is not representable as finite positive coefficients"))
        Tmin, Tmax = temperature_min[i, j], temperature_max[i, j]
        Tmin_j, Tmax_j = temperature_min[j, i], temperature_max[j, i]
        isfinite(Tmin) && isfinite(Tmax) && Tmin > 0 && Tmax >= Tmin ||
            throw(ArgumentError("binary temperature range [$i,$j] must be finite, positive, and ordered"))
        isfinite(Tmin_j) && isfinite(Tmax_j) && isapprox(Tmin, Tmin_j; rtol=8eps(T), atol=zero(T)) &&
            isapprox(Tmax, Tmax_j; rtol=8eps(T), atol=zero(T)) ||
            throw(ArgumentError("binary temperature ranges must be symmetric"))
        isfinite(T(Tmin)) && T(Tmin) > zero(T) && isfinite(T(Tmax)) && T(Tmax) >= T(Tmin) &&
            isfinite(T(Tmin_j)) && isfinite(T(Tmax_j)) ||
            throw(ArgumentError("binary temperature ranges are not representable as finite positive coefficients"))
        T(Tmin) <= temperature_ref_T <= T(Tmax) ||
            throw(ArgumentError("temperature_ref must lie in every binary pair temperature range"))
        for m in 1:M
            cij, cji = temperature_coefficients[i, j, m], temperature_coefficients[j, i, m]
            isfinite(cij) && isfinite(cji) && isapprox(cij, cji; rtol=8eps(T), atol=zero(T)) ||
                throw(ArgumentError("binary temperature coefficients must be finite and symmetric"))
            isfinite(T(cij)) && isfinite(T(cji)) ||
                throw(ArgumentError("binary temperature coefficients are not representable in coefficient type"))
        end
    end
    values = ntuple(i -> ntuple(j -> i == j ? zero(T) : T(D_ref[i, j]), Val(N)), Val(N))
    coefficients = ntuple(i -> ntuple(j -> ntuple(m ->
        i == j ? zero(T) : T(temperature_coefficients[i, j, m]), Val(M)), Val(N)), Val(N))
    Tmin = ntuple(i -> ntuple(j -> i == j ? zero(T) : T(temperature_min[i, j]), Val(N)), Val(N))
    Tmax = ntuple(i -> ntuple(j -> i == j ? zero(T) : T(temperature_max[i, j]), Val(N)), Val(N))
    name_tuple = ntuple(i -> Symbol(names[i]), Val(N))
    return BinaryDiffusionPolynomial{T,N,M,name_tuple}(values, coefficients, Tmin, Tmax,
                                                        temperature_ref_T, pressure_ref_T)
end

species_names(::BinaryDiffusionPolynomial{T,N,M,Names}) where {T,N,M,Names} =
    [String(name) for name in Names]

@inline _binary_diffusion_names(::BinaryDiffusionPolynomial{T,N,M,Names}) where {T,N,M,Names} = Names

"""
    binary_diffusivity(model, temperature, pressure, i, j)

Evaluate a [`BinaryDiffusionPolynomial`](@ref) pair in SI units. `i` and `j`
are distinct one-based species indices in the exact `species` order passed to
the constructor. Temperature and pressure must be finite and positive, and the
temperature must lie in that pair's stated inclusive validity range.
"""
function binary_diffusivity(model::BinaryDiffusionPolynomial{T,N}, temperature, pressure,
                            i::Integer, j::Integer) where {T,N}
    1 <= i <= N && 1 <= j <= N || throw(ArgumentError("binary diffusion species indices must lie in 1:$N"))
    i != j || throw(ArgumentError("binary diffusion requires distinct species indices"))
    isfinite(temperature) && temperature > 0 ||
        throw(ArgumentError("binary diffusion temperature must be finite and positive"))
    isfinite(pressure) && pressure > 0 ||
        throw(ArgumentError("binary diffusion pressure must be finite and positive"))
    temperature_T, pressure_T = T(temperature), T(pressure)
    isfinite(temperature_T) && temperature_T > zero(T) ||
        throw(ArgumentError("binary diffusion temperature is not representable as finite and positive"))
    isfinite(pressure_T) && pressure_T > zero(T) ||
        throw(ArgumentError("binary diffusion pressure is not representable as finite and positive"))
    model.temperature_min[i][j] <= temperature <= model.temperature_max[i][j] ||
        throw(ArgumentError("binary diffusion temperature lies outside the pair validity range"))
    D = _binary_diffusivity(model, temperature_T, pressure_T, i, j)
    isfinite(D) && D > zero(T) ||
        throw(ArgumentError("binary diffusion coefficient is not finite and positive at this state"))
    return D
end

@inline function _binary_diffusivity(model::BinaryDiffusionPolynomial{T,N,M}, temperature,
                                     pressure, i::Integer, j::Integer) where {T,N,M}
    z = log(temperature / model.temperature_ref)
    exponent = zero(T)
    @inbounds for m in 1:M
        exponent += model.temperature_coefficients[i][j][m] * z^m
    end
    log_D = log(model.D_ref[i][j]) + exponent + log(model.pressure_ref) - log(pressure)
    return exp(log_D)
end

struct CeaSpeciesTransport{T}
    viscosity::NTuple{3,CeaTransportInterval{T}}
    conductivity::NTuple{3,CeaTransportInterval{T}}
    n_viscosity::UInt8
    n_conductivity::UInt8
end

"""
    CeaTransport(eos; diffusion=:unity_lewis, Lewis=1,
                 binary_diffusion=nothing, path=CEA_TRANSPORT_PATH)

Temperature-dependent molecular transport using NASA CEA pure-species
viscosity and conductivity fits. `:unity_lewis` obtains every species
diffusivity from `kappa/(rho*cp*Lewis)`. `:mixture_averaged` requires a
[`BinaryDiffusion`](@ref) or species-labelled
[`BinaryDiffusionPolynomial`](@ref), since CEA's transport table contains no
binary diffusion coefficients. The polynomial model's species order must
exactly match the EOS, and its pair validity ranges bound scalar coefficient
queries. Outside a CEA pure-property fit's stated temperature range, the
nearest interval polynomial is extrapolated. Inputs to
[`transport_coefficients`](@ref) must have positive finite temperature,
density and heat capacity and finite, nonnegative mass fractions with a
positive sum. The hot pointwise path assumes that setup and state validation
enforce this contract.

Returned `mu`, `kappa`, and `D` use SI units Pa s, W/(m K), and m^2/s.
"""
struct CeaTransport{T,N,D,Names} <: AbstractTransport{T}
    species::NTuple{N,CeaSpeciesTransport{T}}
    Rk::NTuple{N,T}
    diffusion::D
    Lewis::T
end

function _cea_fixed_intervals(::Type{T}, values, name, property) where {T}
    isempty(values) && throw(ArgumentError("CEA transport data for $name has no $property fit"))
    length(values) <= 3 || throw(ArgumentError("CEA transport data for $name has more than three intervals"))
    converted = [CeaTransportInterval{T}(T(x.Tmin), T(x.Tmax), T(x.A), T(x.B), T(x.C), T(x.D))
                 for x in values]
    for (i, x) in pairs(converted)
        all(isfinite, (x.Tmin, x.Tmax, x.A, x.B, x.C, x.D)) ||
            throw(ArgumentError("non-finite CEA $property fit for $name"))
        x.Tmin > 0 && x.Tmax > x.Tmin ||
            throw(ArgumentError("invalid CEA $property temperature range for $name"))
        i == 1 || x.Tmin >= converted[i - 1].Tmax ||
            throw(ArgumentError("overlapping CEA $property intervals for $name"))
    end
    filler = converted[end]
    return ntuple(i -> i <= length(converted) ? converted[i] : filler, 3)
end

function _transport_scalar(::Type{T}, value, description) where {T}
    converted = try
        T(value)
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("$description is not representable in EOS scalar type $T"))
    end
    return converted
end

function _convert_binary_diffusion(model::BinaryDiffusion{S,N}, ::Type{T}) where {S,N,T}
    values = ntuple(i -> ntuple(j -> _transport_scalar(T, model.D_ref[i][j],
        "BinaryDiffusion coefficients"), Val(N)), Val(N))
    temperature_ref = _transport_scalar(T, model.temperature_ref, "BinaryDiffusion temperature_ref")
    pressure_ref = _transport_scalar(T, model.pressure_ref, "BinaryDiffusion pressure_ref")
    exponent = _transport_scalar(T, model.temperature_exponent, "BinaryDiffusion temperature_exponent")
    all(isfinite(values[i][j]) for i in 1:N for j in 1:N) &&
        all(i == j || values[i][j] > zero(T) for i in 1:N for j in 1:N) ||
        throw(ArgumentError("BinaryDiffusion coefficients are not representable in EOS scalar type $T"))
    isfinite(temperature_ref) && temperature_ref > zero(T) &&
        isfinite(pressure_ref) && pressure_ref > zero(T) && isfinite(exponent) ||
        throw(ArgumentError("BinaryDiffusion scaling values are not representable in EOS scalar type $T"))
    return BinaryDiffusion{T,N}(values, temperature_ref, pressure_ref, exponent)
end

function _convert_binary_diffusion(model::BinaryDiffusionPolynomial{S,N,M,Names},
                                   ::Type{T}) where {S,N,M,Names,T}
    values = ntuple(i -> ntuple(j -> _transport_scalar(T, model.D_ref[i][j],
        "BinaryDiffusionPolynomial coefficients"), Val(N)), Val(N))
    coefficients = ntuple(i -> ntuple(j -> ntuple(m -> _transport_scalar(T,
        model.temperature_coefficients[i][j][m], "BinaryDiffusionPolynomial temperature coefficients"),
        Val(M)), Val(N)), Val(N))
    Tmin = ntuple(i -> ntuple(j -> _transport_scalar(T, model.temperature_min[i][j],
        "BinaryDiffusionPolynomial temperature ranges"), Val(N)), Val(N))
    Tmax = ntuple(i -> ntuple(j -> _transport_scalar(T, model.temperature_max[i][j],
        "BinaryDiffusionPolynomial temperature ranges"), Val(N)), Val(N))
    temperature_ref = _transport_scalar(T, model.temperature_ref,
                                        "BinaryDiffusionPolynomial temperature_ref")
    pressure_ref = _transport_scalar(T, model.pressure_ref,
                                     "BinaryDiffusionPolynomial pressure_ref")
    all(isfinite(values[i][j]) for i in 1:N for j in 1:N) &&
        all(i == j || values[i][j] > zero(T) for i in 1:N for j in 1:N) &&
        all(isfinite(coefficients[i][j][m]) for i in 1:N for j in 1:N for m in 1:M) &&
        all(i == j || (isfinite(Tmin[i][j]) && Tmin[i][j] > zero(T) &&
                       isfinite(Tmax[i][j]) && Tmax[i][j] >= Tmin[i][j])
            for i in 1:N for j in 1:N) &&
        isfinite(temperature_ref) && temperature_ref > zero(T) &&
        isfinite(pressure_ref) && pressure_ref > zero(T) &&
        all(i == j || Tmin[i][j] <= temperature_ref <= Tmax[i][j]
            for i in 1:N for j in 1:N) ||
        throw(ArgumentError("BinaryDiffusionPolynomial data are not representable in EOS scalar type $T"))
    return BinaryDiffusionPolynomial{T,N,M,Names}(values, coefficients, Tmin, Tmax,
                                                    temperature_ref, pressure_ref)
end

function CeaTransport(eos; diffusion::Symbol=:unity_lewis, Lewis=1.0,
                      binary_diffusion=nothing, path::AbstractString=CEA_TRANSPORT_PATH)
    eos = _as_eos(eos)
    names = species_names(eos)
    N = length(names)
    hasproperty(eos, :Rk) || throw(ArgumentError("CeaTransport requires an EOS with per-species Rk"))
    T = eltype(eos.Rk)
    records = read_cea_transport(path)
    pure = Dict(record.names[1] => record for record in records if isempty(record.names[2]))
    absent = [name for name in names if !haskey(pure, name)]
    isempty(absent) || throw(ArgumentError("NASA CEA transport species not found: " * join(absent, ", ")))
    sp = ntuple(Val(N)) do k
        record = pure[names[k]]
        CeaSpeciesTransport{T}(_cea_fixed_intervals(T, record.viscosity, names[k], "viscosity"),
                               _cea_fixed_intervals(T, record.conductivity, names[k], "conductivity"),
                               UInt8(length(record.viscosity)), UInt8(length(record.conductivity)))
    end
    isfinite(Lewis) && Lewis > 0 ||
        throw(ArgumentError("Lewis number must be finite and positive"))
    LewisT = T(Lewis)
    isfinite(LewisT) && LewisT > zero(T) ||
        throw(ArgumentError("Lewis number is not representable in EOS scalar type $T"))
    model = if diffusion === :unity_lewis
        binary_diffusion === nothing || throw(ArgumentError("binary_diffusion is unused with diffusion=:unity_lewis"))
        nothing
    elseif diffusion === :mixture_averaged
        (binary_diffusion isa BinaryDiffusion || binary_diffusion isa BinaryDiffusionPolynomial) || throw(ArgumentError(
            "diffusion=:mixture_averaged requires BinaryDiffusion or BinaryDiffusionPolynomial data; CEA trans.inp contains no diffusion coefficients"))
        length(binary_diffusion.D_ref) == N || throw(ArgumentError("BinaryDiffusion species count does not match EOS"))
        binary_diffusion isa BinaryDiffusionPolynomial &&
            _binary_diffusion_names(binary_diffusion) != ntuple(k -> Symbol(names[k]), Val(N)) &&
            throw(ArgumentError("BinaryDiffusionPolynomial species order does not match EOS"))
        _convert_binary_diffusion(binary_diffusion, T)
    else
        throw(ArgumentError("diffusion must be :unity_lewis or :mixture_averaged"))
    end
    name_tuple = ntuple(k -> Symbol(names[k]), Val(N))
    gas_constants = ntuple(k -> T(eos.Rk[k]), Val(N))
    all(x -> isfinite(x) && x > zero(T), gas_constants) ||
        throw(ArgumentError("EOS gas constants are not finite, positive, and representable in scalar type $T"))
    return CeaTransport{T,N,typeof(model),name_tuple}(sp,
        gas_constants, model, LewisT)
end

@inline function _cea_fit(intervals, count, temperature)
    interval = intervals[1]
    @inbounds for i in 2:Int(count)
        temperature > intervals[i - 1].Tmax && (interval = intervals[i])
    end
    invT = inv(temperature)
    return exp(interval.A * log(temperature) + interval.B * invT +
               interval.C * invT * invT + interval.D)
end

@inline function _wilke_mix(X, property, phi_property, Rk, ::Val{N}) where {N}
    total = zero(property[1])
    @inbounds for i in 1:N
        denominator = zero(total)
        for j in 1:N
            phi = (one(total) + sqrt(phi_property[i] / phi_property[j]) *
                  (Rk[i] / Rk[j])^(one(total)/4))^2 /
                  sqrt(8 * (one(total) + Rk[j] / Rk[i]))
            denominator += X[j] * phi
        end
        denominator > zero(total) && (total += X[i] * property[i] / denominator)
    end
    return total
end

"""
    transport_coefficients(transport, eos, temperature, rho, cp, Y)

Return pointwise dynamic viscosity `mu`, thermal conductivity `kappa`, and an
`NTuple` of species mass diffusivities `D`. The mass fractions `Y` must be an
`NTuple` in the EOS species order. Implement this function for custom
[`AbstractTransport`](@ref) models.
"""
function transport_coefficients end

const TRANSPORT_OK = UInt8(0)
const TRANSPORT_INVALID_TEMPERATURE = UInt8(1)
const TRANSPORT_INVALID_PRESSURE = UInt8(2)
const TRANSPORT_TEMPERATURE_OUT_OF_RANGE = UInt8(4)
const TRANSPORT_INVALID_COEFFICIENT = UInt8(8)

"Whether `transport` has a finite pointwise validity domain to preflight."
transport_has_domain(::AbstractTransport) = false
transport_has_domain(::CeaTransport{T,N,<:BinaryDiffusionPolynomial}) where {T,N} = true

"""
    transport_domain_status(transport, eos, temperature, rho, Y) -> UInt8

Return pointwise transport-domain flags without throwing. Solver preflight uses
this hook before entering unchecked coefficient evaluation. The flags describe
an invalid temperature or pressure, a polynomial pair-range violation, or a
non-finite/nonpositive evaluated pair coefficient.
"""
transport_domain_status(::AbstractTransport, eos, temperature, rho, Y) = TRANSPORT_OK

@inline function _transport_pressure_status(transport::CeaTransport{T,N}, temperature,
                                            rho, Y::NTuple{N}) where {T,N}
    isfinite(temperature) && temperature > zero(temperature) || return TRANSPORT_INVALID_TEMPERATURE, zero(T), zero(T)
    temperature_T = T(temperature)
    isfinite(temperature_T) && temperature_T > zero(T) || return TRANSPORT_INVALID_TEMPERATURE, zero(T), zero(T)
    isfinite(rho) && rho > zero(rho) || return TRANSPORT_INVALID_PRESSURE, temperature_T, zero(T)
    rho_T = T(rho)
    isfinite(rho_T) && rho_T > zero(T) || return TRANSPORT_INVALID_PRESSURE, temperature_T, zero(T)
    sumYR = zero(T)
    @inbounds for k in 1:N
        y = Y[k]
        isfinite(y) && y >= zero(y) || return TRANSPORT_INVALID_PRESSURE, temperature_T, zero(T)
        y_T = T(y)
        isfinite(y_T) && y_T >= zero(T) || return TRANSPORT_INVALID_PRESSURE, temperature_T, zero(T)
        sumYR += y_T * transport.Rk[k]
    end
    pressure = rho_T * sumYR * temperature_T
    isfinite(pressure) && pressure > zero(T) || return TRANSPORT_INVALID_PRESSURE, temperature_T, pressure
    return TRANSPORT_OK, temperature_T, pressure
end

@inline function transport_domain_status(transport::CeaTransport{T,N}, eos, temperature,
                                         rho, Y::NTuple{N}) where {T,N}
    status, _, _ = _transport_pressure_status(transport, temperature, rho, Y)
    return status
end

@inline function transport_domain_status(transport::CeaTransport{T,N,<:BinaryDiffusionPolynomial},
                                         eos, temperature, rho, Y::NTuple{N}) where {T,N}
    status, temperature_T, pressure = _transport_pressure_status(transport, temperature, rho, Y)
    status == TRANSPORT_OK || return status
    model = transport.diffusion
    @inbounds for i in 1:N, j in i + 1:N
        # Keep scalar queries strict in their caller precision.  Converting a
        # Float64 just above a Float32 endpoint first could otherwise round it
        # back onto the inclusive endpoint.
        model.temperature_min[i][j] <= temperature <= model.temperature_max[i][j] ||
            return TRANSPORT_TEMPERATURE_OUT_OF_RANGE
        Dij = _binary_diffusivity(model, temperature_T, pressure, i, j)
        isfinite(Dij) && Dij > zero(T) || return TRANSPORT_INVALID_COEFFICIENT
    end
    return TRANSPORT_OK
end

@inline function _checked_transport_state(transport::CeaTransport{T,N}, eos, temperature,
                                          rho, cp, Y::NTuple{N}) where {T,N}
    status = transport_domain_status(transport, eos, temperature, rho, Y)
    status == TRANSPORT_OK || throw(ArgumentError(_transport_domain_message(status)))
    isfinite(cp) && cp > zero(cp) ||
        throw(ArgumentError("transport heat capacity must be finite and positive"))
    cp_T = T(cp)
    isfinite(cp_T) && cp_T > zero(T) ||
        throw(ArgumentError("transport heat capacity is not representable as finite and positive"))
    return nothing
end

@inline function _transport_domain_message(status)
    status == TRANSPORT_INVALID_TEMPERATURE && return "transport temperature must be finite and positive"
    status == TRANSPORT_INVALID_PRESSURE && return "transport pressure must be finite and positive"
    status == TRANSPORT_TEMPERATURE_OUT_OF_RANGE &&
        return "transport temperature lies outside the binary-diffusion validity range"
    status == TRANSPORT_INVALID_COEFFICIENT &&
        return "binary diffusion coefficient is not finite and positive at this state"
    return "transport state is invalid"
end

@inline function transport_coefficients(transport::ConstantTransport, eos, temperature, rho,
                                        cp, Y::NTuple{N}) where {N}
    mu = transport.mu0
    return (mu=mu, kappa=mu * cp / transport.Pr,
            D=ntuple(_ -> mu / (rho * transport.Sc), Val(N)))
end

# Public coefficient queries commonly use a literal temperature such as 300.
# Solver fields already carry floating-point temperatures and use the method
# below directly, preserving its arithmetic and specialization.
@inline function transport_coefficients(transport::CeaTransport{T,N}, eos,
                                        temperature::Integer, rho, cp,
                                        Y::NTuple{N}) where {T,N}
    _checked_transport_state(transport, eos, temperature, rho, cp, Y)
    return _transport_coefficients(transport, T(temperature), rho, cp, Y)
end

@inline function transport_coefficients(transport::CeaTransport{T,N}, eos, temperature,
                                        rho, cp, Y::NTuple{N}) where {T,N}
    _checked_transport_state(transport, eos, temperature, rho, cp, Y)
    return _transport_coefficients(transport, temperature, rho, cp, Y)
end

@inline function _transport_coefficients(transport::CeaTransport{T,N}, temperature,
                                         rho, cp, Y::NTuple{N}) where {T,N}
    sumYR = sum(ntuple(k -> Y[k] * transport.Rk[k], Val(N)))
    X = ntuple(k -> Y[k] * transport.Rk[k] / sumYR, Val(N))
    viscosity = ntuple(k -> oftype(temperature, 1e-7) *
                       _cea_fit(transport.species[k].viscosity,
                                transport.species[k].n_viscosity, temperature), Val(N))
    conductivity = ntuple(k -> oftype(temperature, 1e-4) *
                          _cea_fit(transport.species[k].conductivity,
                                   transport.species[k].n_conductivity, temperature), Val(N))
    mu = _wilke_mix(X, viscosity, viscosity, transport.Rk, Val(N))
    # The Wassiljewa/Mason-Saxena conductivity rule uses the same viscosity and
    # molecular-weight interaction factors as Wilke's viscosity rule (see
    # NASA/TM-20220008776, Appendix B, equations B3-B7).
    kappa = _wilke_mix(X, conductivity, viscosity, transport.Rk, Val(N))
    D = _transport_diffusivities(transport, temperature, rho, cp, Y, X, sumYR, kappa)
    return (mu=mu, kappa=kappa, D=D)
end

@inline _transport_diffusivities(transport::CeaTransport{T,N,Nothing}, temperature,
                                 rho, cp, Y, X, sumYR, kappa) where {T,N} =
    ntuple(_ -> kappa / (rho * cp * transport.Lewis), Val(N))

@inline function _transport_diffusivities(transport::CeaTransport{T,N,<:BinaryDiffusion},
                                          temperature, rho, cp, Y, X, sumYR, kappa) where {T,N}
    N == 1 && return (zero(temperature),)
    binary = transport.diffusion
    pressure = rho * sumYR * temperature
    scale = (temperature / binary.temperature_ref)^binary.temperature_exponent *
            binary.pressure_ref / pressure
    return ntuple(Val(N)) do i
        sum_X_over_D = zero(temperature)
        sum_XW = zero(temperature)
        sum_XW_over_D = zero(temperature)
        @inbounds for j in 1:N
            if j != i
                Dij = binary.D_ref[i][j] * scale
                sum_X_over_D += X[j] / Dij
                XW = X[j] / transport.Rk[j]
                sum_XW += XW
                sum_XW_over_D += XW / Dij
            end
        end
        denominator = sum_X_over_D
        sum_XW > zero(sum_XW) && (denominator += X[i] * sum_XW_over_D / sum_XW)
        denominator > zero(denominator) ? inv(denominator) : zero(temperature)
    end
end

@inline function _transport_diffusivities(transport::CeaTransport{T,N,<:BinaryDiffusionPolynomial},
                                          temperature, rho, cp, Y, X, sumYR, kappa) where {T,N}
    N == 1 && return (zero(temperature),)
    binary = transport.diffusion
    pressure = rho * sumYR * temperature
    return ntuple(Val(N)) do i
        sum_X_over_D = zero(temperature)
        sum_XW = zero(temperature)
        sum_XW_over_D = zero(temperature)
        @inbounds for j in 1:N
            if j != i
                Dij = _binary_diffusivity(binary, temperature, pressure, i, j)
                sum_X_over_D += X[j] / Dij
                XW = X[j] / transport.Rk[j]
                sum_XW += XW
                sum_XW_over_D += XW / Dij
            end
        end
        denominator = sum_X_over_D
        sum_XW > zero(sum_XW) && (denominator += X[i] * sum_XW_over_D / sum_XW)
        denominator > zero(denominator) ? inv(denominator) : zero(temperature)
    end
end

@inline transport_at(transport::ConstantTransport, eos, temperature, rho, cp, Y, I) =
    (mu=transport.mu0, kappa=transport.mu0 * cp[I] / transport.Pr,
     D=UniformDiffusivity(transport.mu0 / (rho[I] * transport.Sc)))

@inline function transport_at(transport::CeaTransport{T,N}, eos, temperature, rho, cp,
                              Y, I) where {T,N}
    fractions = ntuple(k -> @inbounds(Y[k][I]), Val(N))
    return _transport_coefficients(transport, temperature[I], rho[I], cp[I], fractions)
end

@inline function transport_at(transport::AbstractTransport, eos, temperature, rho, cp,
                              Y, I)
    # Correctness fallback for custom host models. Models used in device kernels
    # should specialize transport_at with a statically known species count, as
    # CeaTransport does, so tuple construction is inferred and allocation-free.
    fractions = ntuple(k -> @inbounds(Y[k][I]), length(Y))
    return transport_coefficients(transport, eos, temperature[I], rho[I], cp[I], fractions)
end

validate_transport(::AbstractTransport, eos) = nothing
function validate_transport(transport::ConstantTransport, eos)
    (; mu0, Pr, Sc) = transport
    isfinite(mu0) && mu0 >= 0 ||
        throw(ArgumentError("ConstantTransport: mu0 must be finite and >= 0 " *
                            "(0 is inviscid), got $mu0"))
    isfinite(Pr) && Pr > 0 ||
        throw(ArgumentError("ConstantTransport: Pr must be finite and positive, got $Pr"))
    isfinite(Sc) && Sc > 0 ||
        throw(ArgumentError("ConstantTransport: Sc must be finite and positive, got $Sc"))
    return nothing
end
function validate_transport(transport::CeaTransport{T,N,D,Names}, eos) where {T,N,D,Names}
    nspecies(eos) == N || throw(ArgumentError("CeaTransport species count does not match EOS"))
    ntuple(k -> Symbol(species_names(eos)[k]), Val(N)) == Names ||
        throw(ArgumentError("CeaTransport species order does not match EOS"))
    ntuple(k -> T(eos.Rk[k]), Val(N)) == transport.Rk ||
        throw(ArgumentError("CeaTransport molecular weights do not match EOS"))
    return nothing
end
