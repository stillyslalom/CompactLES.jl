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
[`BinaryDiffusion`](@ref), since CEA's transport table contains no binary
diffusion coefficients. Outside a fit's stated temperature range, the nearest
interval polynomial is extrapolated. Inputs to [`transport_coefficients`](@ref)
must have positive finite temperature, density and heat capacity and finite,
nonnegative mass fractions with a positive sum. The hot pointwise path assumes
that setup and state validation enforce this contract.

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
        binary_diffusion isa BinaryDiffusion || throw(ArgumentError(
            "diffusion=:mixture_averaged requires BinaryDiffusion data; CEA trans.inp contains no diffusion coefficients"))
        length(binary_diffusion.D_ref) == N || throw(ArgumentError("BinaryDiffusion species count does not match EOS"))
        b = binary_diffusion
        values = ntuple(i -> ntuple(j -> T(b.D_ref[i][j]), Val(N)), Val(N))
        temperature_ref = T(b.temperature_ref)
        pressure_ref = T(b.pressure_ref)
        exponent = T(b.temperature_exponent)
        all(isfinite(values[i][j]) for i in 1:N for j in 1:N) &&
            all(i == j || values[i][j] > zero(T) for i in 1:N for j in 1:N) ||
            throw(ArgumentError("BinaryDiffusion coefficients are not representable in EOS scalar type $T"))
        isfinite(temperature_ref) && temperature_ref > zero(T) &&
            isfinite(pressure_ref) && pressure_ref > zero(T) && isfinite(exponent) ||
            throw(ArgumentError("BinaryDiffusion scaling values are not representable in EOS scalar type $T"))
        BinaryDiffusion{T,N}(values, temperature_ref, pressure_ref, exponent)
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

@inline function transport_coefficients(transport::Transport, eos, temperature, rho, cp,
                                        Y::NTuple{N}) where {N}
    mu = transport.mu0
    return (mu=mu, kappa=mu * cp / transport.Pr,
            D=ntuple(_ -> mu / (rho * transport.Sc), Val(N)))
end

# Public coefficient queries commonly use a literal temperature such as 300.
# Solver fields already carry floating-point temperatures and use the method
# below directly, preserving its arithmetic and specialization.
@inline transport_coefficients(transport::CeaTransport{T,N}, eos, temperature::Integer,
                               rho, cp, Y::NTuple{N}) where {T,N} =
    transport_coefficients(transport, eos, T(temperature), rho, cp, Y)

@inline function transport_coefficients(transport::CeaTransport{T,N}, eos, temperature,
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

@inline transport_at(transport::Transport, eos, temperature, rho, cp, Y, I) =
    (mu=transport.mu0, kappa=transport.mu0 * cp[I] / transport.Pr,
     D=UniformDiffusivity(transport.mu0 / (rho[I] * transport.Sc)))

@inline function transport_at(transport::CeaTransport{T,N}, eos, temperature, rho, cp,
                              Y, I) where {T,N}
    fractions = ntuple(k -> @inbounds(Y[k][I]), Val(N))
    return transport_coefficients(transport, eos, temperature[I], rho[I], cp[I], fractions)
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
validate_transport(::Transport, eos) = nothing
function validate_transport(transport::CeaTransport{T,N,D,Names}, eos) where {T,N,D,Names}
    nspecies(eos) == N || throw(ArgumentError("CeaTransport species count does not match EOS"))
    ntuple(k -> Symbol(species_names(eos)[k]), Val(N)) == Names ||
        throw(ArgumentError("CeaTransport species order does not match EOS"))
    ntuple(k -> T(eos.Rk[k]), Val(N)) == transport.Rk ||
        throw(ArgumentError("CeaTransport molecular weights do not match EOS"))
    return nothing
end
