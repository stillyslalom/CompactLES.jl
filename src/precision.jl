# Floating-point precision of a solver configuration.
#
# The storage element type `T` of a `Solver` fixes the arithmetic of every
# kernel. A component that carries its own scalar type (the EOS, the transport
# model, the artificial-property coefficients, the compact schemes) and
# disagrees with `T` either fails to convert where the solver stores it
# (`ArtParams{T}`) or promotes the kernel arithmetic it enters: a Float64
# `IdealMixture` under Float32 storage turns `cvk[sp] * T_ion` into a Float64
# product at every point. `_resolve_precision` settles `T` once, at setup,
# and either converts every component to it or rejects the mixture.
#
# Boundary conditions and sources are not listed: each converts its scalars
# to the state's element type where it is applied (`NSCBCOutflowBC`,
# `NSCBCInflowBC`, `NoSlipWallBC`, `ConstantBodyForce`). `StepControl` is
# Float64 by design, and `Metric` and `Stretch` carry no scalar type.

# The floating-point type a configuration component carries, or `nothing` for
# a component without one.
_precision_of(x) = nothing
_precision_of(::ArtParams{T}) where {T} = T
_precision_of(::AbstractTransport{T}) where {T} = T
_precision_of(::IdealMixture{T}) where {T} = T
_precision_of(::StiffenedGas{T}) where {T} = T
_precision_of(::Nasa9Mixture{T}) where {T} = T
_precision_of(::CompactScheme{T}) where {T} = T
_precision_of(::BandedCompactScheme{T}) where {T} = T

# `x` rebuilt with every stored scalar converted to `T`; the identity for a
# component already at `T` or without a scalar type. Derived quantities (the
# mixture's `cvk`, a NASA-9 mixture's seed energies) are recomputed at `T` from
# the converted inputs, so the result is the component built at `T` from the
# same inputs. A type with a scalar type and no method here is rejected.
_to_precision(::Type{T}, x) where {T} =
    _precision_of(x) in (nothing, T) ? x : _no_precision_conversion(T, x)
_to_precision(::Type{T}, x::ArtParams{T}) where {T} = x
_to_precision(::Type{T}, x::ArtParams) where {T} =
    ArtParams{T}((getfield(x, f) for f in fieldnames(ArtParams))...)
_to_precision(::Type{T}, x::Transport{T}) where {T} = x
_to_precision(::Type{T}, x::Transport) where {T} = Transport{T}(x.mu0, x.Pr, x.Sc)
_to_precision(::Type{T}, x::CeaTransport{T}) where {T} = x
function _to_precision(::Type{T}, x::CeaTransport{S,N,D,Names}) where {T,S,N,D,Names}
    ivl(a) = CeaTransportInterval{T}(T(a.Tmin), T(a.Tmax), T(a.A), T(a.B), T(a.C),
                                     T(a.D))
    sp = ntuple(k -> CeaSpeciesTransport{T}(map(ivl, x.species[k].viscosity),
                                            map(ivl, x.species[k].conductivity),
                                            x.species[k].n_viscosity,
                                            x.species[k].n_conductivity), Val(N))
    model = x.diffusion === nothing ? nothing :
            _convert_binary_diffusion(x.diffusion, T)
    return CeaTransport{T,N,typeof(model),Names}(sp, map(T, x.Rk), model, T(x.Lewis))
end
_to_precision(::Type{T}, x::IdealMixture{T}) where {T} = x
_to_precision(::Type{T}, x::IdealMixture) where {T} = IdealMixture(T, x.sp)
_to_precision(::Type{T}, x::StiffenedGas{T}) where {T} = x
_to_precision(::Type{T}, x::StiffenedGas) where {T} =
    StiffenedGas{T}(T(x.gamma), T(x.p_inf), T(x.cv), x.name)
_to_precision(::Type{T}, x::Nasa9Mixture{T}) where {T} = x
_to_precision(::Type{T}, x::Nasa9Mixture) where {T} =
    _nasa9_mixture(T, x.sp; T_guess=x.T_guess, extrapolate=x.extrapolate)
# A scheme's coefficients are rounded from the stored values. The presets
# evaluate theirs in at least Float64 and round once to `T`, so a converted
# Float64 preset is the preset built at `T`: `compact_filter(0.45)` converts
# to `compact_filter(0.45, Float32)`, though not to
# `compact_filter(Float32(0.45), Float32)`, whose αf is a different number.
_to_precision(::Type{T}, x::CompactScheme{T}) where {T} = x
_to_precision(::Type{T}, x::CompactScheme) where {T} =
    CompactScheme{T}(x.name, T(x.alpha), T(x.a0), T.(x.coeffs), x.symmetric,
                     [ClosureRow{T}(map(T, r.lhs), T.(r.rhs), r.first)
                      for r in x.closures])
_to_precision(::Type{T}, x::BandedCompactScheme{T}) where {T} = x
_to_precision(::Type{T}, x::BandedCompactScheme) where {T} =
    BandedCompactScheme{T}(x.name, x.q, T.(x.lhs), T(x.a0), T.(x.coeffs),
                           x.symmetric,
                           [BandedClosureRow{T}(T.(r.lhs), T.(r.rhs), r.first)
                            for r in x.closures])

@noinline _no_precision_conversion(::Type{T}, x) where {T} =
    throw(ArgumentError("precision = $T: no conversion is defined for " *
                        "$(typeof(x)); construct it with scalar type $T"))

# The components of a `Solver` call that carry a scalar type, by keyword.
# `nothing` marks a keyword left at its default, which is built at the
# resolved precision and so never disagrees.
function _resolve_precision(precision, components::NamedTuple)
    precision === nothing || (precision isa Type && precision <: AbstractFloat) ||
        throw(ArgumentError("precision must be a floating-point type such as " *
                            "Float32 or Float64, got $precision"))
    given = [(name, _precision_of(x)) for (name, x) in pairs(components)
             if x !== nothing && _precision_of(x) !== nothing]
    precision === nothing || return precision
    types = unique(last.(given))
    length(types) <= 1 && return isempty(types) ? Float64 : only(types)
    listing = join(("$name is $S" for (name, S) in given), ", ")
    throw(ArgumentError(
        "the solver components carry different floating-point types " *
        "($listing). Pass precision = Float32 or precision = Float64 to " *
        "convert every component to one type, or construct each at the same " *
        "type"))
end
