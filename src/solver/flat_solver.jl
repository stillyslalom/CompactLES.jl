# A solver from one flat keyword list, without a state: the form the
# precompile workload (precompile.jl) and the test and benchmark scripts build
# their configurations in. It is not part of the package's interface. The
# keywords are sorted into a `Problem`, a `Numerics` and an `AMR`, and the
# solver is built from those as `setup` builds it (`_build_solver`).

# An upper endpoint `hi` with `hi - lo == L` exactly, so that the extent the
# constructor recovers from the endpoints is the one given: `lo + L`, or a
# neighbor of it where the sum rounds.
function _extent_endpoint(lo::Float64, L::Float64)
    hi = lo + L
    isfinite(hi) && L > 0 || return hi
    for candidate in (hi, nextfloat(hi), prevfloat(hi))
        candidate - lo == L && return candidate
    end
    throw(ArgumentError("flat_solver: no endpoint reproduces the extent $L from " *
                        "the origin $lo; give the two so that their sum is exact"))
end

"""
    flat_solver(; n_global, L_domain, bcs, kwargs...) -> Solver

Build a `Solver` from one keyword list and no initial condition, for the
precompile workload and for test and benchmark scripts; pair it with
[`allocate_state`](@ref) and [`initialize!`](@ref). [`setup`](@ref) is the
entry point of a run.

`L_domain` holds the three extents and `origin` the low corner, where
`Problem` takes endpoints. `eos`, `transport`, `metric` and `sources` are
those of [`Problem`](@ref); `art`, `deriv`, `cfl`, `control`,
`polar_truncation`, `implicit`, `positivity_limiter`, `n_halo` and `stretch`
those of [`Numerics`](@ref); `filt`, `filter_interval`, `filter_cfl` and
`filter_weighting` the fields of [`StateFilter`](@ref); the `interface_`
keywords those of [`PatchInterfaces`](@ref); `dims`, `comm`, `backend`,
`precision` and `patch_grid` those of [`Execution`](@ref); and the remaining
ones those of [`AMR`](@ref), with `refine` for its `initial` regions. Two
keywords have no counterpart there: `tag_predicate`, a `(patch, I) -> Bool`
closure over the parent patch, and `equations`, the `EquationSet` owning the
conserved layout.

Without `precision`, the element type is the one shared by the components
passed explicitly (`eos`, `transport`, `art`, `deriv`, `filt`,
`interface_divergence`), or `Float64` when none is passed; the components
left out are built at that type.
"""
function flat_solver(; n_global::NTuple{3,Int}, L_domain, bcs,
                     precision::Union{Nothing,Type}=nothing,
                     eos=nothing,
                     transport::Union{Nothing,AbstractTransport}=nothing,
                     art::Union{Nothing,ArtificialProperties}=nothing,
                     deriv::Union{Nothing,AbstractCompactScheme}=nothing,
                     filt::Union{Nothing,AbstractCompactScheme}=nothing,
                     interface_divergence::Union{Nothing,AbstractCompactScheme}=nothing,
                     equations=nothing,
                     metric::Metric=CartesianMetric(),
                     stretch::NTuple{3,Union{Nothing,Stretch}}=(nothing, nothing, nothing),
                     sources=(), origin=(0.0, 0.0, 0.0),
                     cfl::Real=0.5, filter_interval::Int=1, filter_cfl::Real=0.35,
                     filter_weighting::Symbol=:none, polar_truncation::Real=0,
                     control::StepControl=StepControl(),
                     dims=nothing, n_halo::Int=4, comm::MPI.Comm=MPI.COMM_WORLD,
                     patch_grid::NTuple{3,Int}=(1, 1, 1),
                     backend::AbstractBackend=CPUBackend(),
                     interface_rhs::Symbol=:extended, interface_flux::Symbol=:ghost,
                     interface_conservative::Bool=true,
                     interface_limited_ghost::Bool=false,
                     refine::Union{Nothing,BlockRegion,Vector{BlockRegion}}=nothing,
                     level_restriction::Symbol=:inject,
                     level_interpolation_order::Union{Nothing,Int}=nothing,
                     subcycle::Bool=false, regrid_interval::Int=0,
                     tag_threshold::Real=DEFAULT_TAG_THRESHOLD, tag_buffer::Int=4,
                     tag_sensor_threshold::Real=0, tag_gradient_threshold::Real=0,
                     tag_vorticity_threshold::Real=0, tag_predicate=nothing,
                     untag_ratio::Real=2, tile_lifetime::Int=1, tile=0,
                     rebalance::Real=0, rebalance_persist::Int=2,
                     max_levels::Union{Nothing,Int}=nothing,
                     level_boundaries::Bool=true, subdivide_lattice::Bool=true,
                     implicit::Union{Nothing,ImplicitConduction}=nothing,
                     positivity_limiter::Bool=false)
    eos === nothing || (eos = _as_eos(eos))
    T = _resolve_precision(precision, (; eos, transport, art, deriv, filt,
                                        interface_divergence))
    cv(x, default) = x === nothing ? default : _to_precision(T, x)
    domain = ntuple(3) do d
        lo = Float64(origin[d])
        (lo, _extent_endpoint(lo, Float64(L_domain[d])))
    end
    prob = Problem(; eos=cv(eos, _default_ideal_mixture(T)),
                   transport=cv(transport, ConstantTransport{T}()), metric, sources,
                   domain, bcs, ic=nothing)
    filter = StateFilter(cv(filt, compact_filter(0.47, T)), filter_interval,
                         filter_cfl, filter_weighting)
    interfaces = PatchInterfaces(interface_flux, interface_rhs,
                                 cv(interface_divergence, nothing),
                                 interface_conservative, interface_limited_ghost)
    num = Numerics(n_global, cv(deriv, lele_d1_6(T)), filter,
                   cv(art, ArtificialProperties{T}()), cfl, control, interfaces,
                   Execution(dims, comm, backend, T, patch_grid), nothing,
                   polar_truncation, stretch, implicit, positivity_limiter, n_halo)
    amr = AMR(; initial=refine, level_restriction, level_interpolation_order,
              subcycle, regrid_interval, tag_threshold, tag_buffer,
              tag_sensor_threshold, tag_gradient_threshold,
              tag_vorticity_threshold, untag_ratio, tile_lifetime, tile, rebalance,
              rebalance_persist, max_levels, level_boundaries, subdivide_lattice)
    return _build_solver(prob, num, LevelSetup(amr; tag_predicate); equations)
end
