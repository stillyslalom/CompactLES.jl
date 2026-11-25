# A solver from one flat keyword list, without a state: the form the test
# suites, the test-only precompile workload (test/precompile_workload.jl) and
# the benchmark scripts build their configurations in. The keywords are sorted
# into a `Problem`, a `Numerics` and an `AMR`, and the solver is built from
# those by the package's constructor, as `setup` builds it. `L_domain` reaches
# the constructor as given: endpoints formed from an origin and an extent do
# not always return the extent on subtraction, and the grid spacing must not
# depend on which form a test wrote.
#
# Included by whichever file needs it, the test image's module among them, so
# every name is qualified and nothing is imported.

"""
    flat_solver(; n_global, L_domain, bcs, kwargs...) -> Solver

Build a `Solver` from one keyword list and no initial condition; pair it with
`allocate_state` and `initialize!`. `setup` is the entry point of a run.

`L_domain` holds the three extents and `origin` the low corner, where
`Problem` takes endpoints. `eos`, `transport`, `metric` and `sources` are
those of `Problem`; `art`, `deriv`, `cfl`, `control`,
`polar_truncation`, `implicit`, `positivity_limiter`, `n_halo` and `stretch`
those of `Numerics`; `filt`, `filter_interval`, `filter_cfl` and
`filter_weighting` the fields of `StateFilter`; the `interface_`
keywords those of `PatchInterfaces`; `dims`, `comm`, `backend`,
`precision` and `patch_grid` those of `Execution`; and the remaining
ones those of `AMR`, with `refine` for its `initial` regions. Two
keywords have no counterpart there: `tag_predicate`, a `(patch, I) -> Bool`
closure over the parent patch, and `equations`, the `EquationSet` owning the
conserved layout.

Without `precision`, the element type is the one shared by the components
passed explicitly (`eos`, `transport`, `art`, `deriv`, `filt`,
`interface_divergence`), or `Float64` when none is passed; the components
left out are built at that type.
"""
function flat_solver(; n_global, L_domain, bcs, precision=nothing, eos=nothing,
                     transport=nothing, art=nothing, deriv=nothing, filt=nothing,
                     interface_divergence=nothing, equations=nothing,
                     metric=CompactLES.CartesianMetric(),
                     stretch=(nothing, nothing, nothing), sources=(),
                     origin=(0.0, 0.0, 0.0), cfl=0.5, filter_interval=1,
                     filter_cfl=0.35, filter_weighting=:none, polar_truncation=0,
                     control=CompactLES.StepControl(), dims=nothing, n_halo=4,
                     comm=CompactLES.MPI.COMM_WORLD, patch_grid=(1, 1, 1),
                     backend=CompactLES.CPUBackend(), interface_rhs=:extended,
                     interface_flux=:ghost, interface_conservative=true,
                     interface_limited_ghost=false, refine=nothing,
                     level_restriction=:inject, level_interpolation_order=nothing,
                     subcycle=false, regrid_interval=0,
                     tag_threshold=CompactLES.DEFAULT_TAG_THRESHOLD, tag_buffer=4,
                     tag_sensor_threshold=0, tag_gradient_threshold=0,
                     tag_vorticity_threshold=0, tag_predicate=nothing, untag_ratio=2,
                     tile_lifetime=1, tile=0, rebalance=0, rebalance_persist=2,
                     max_levels=nothing, level_boundaries=true,
                     subdivide_lattice=true, implicit=nothing,
                     positivity_limiter=false)
    C = CompactLES
    eos === nothing || (eos = C._as_eos(eos))
    T = C._resolve_precision(precision, (; eos, transport, art, deriv, filt,
                                         interface_divergence))
    cv(x, default) = x === nothing ? default : C._to_precision(T, x)
    domain = ntuple(d -> (Float64(origin[d]),
                          Float64(origin[d]) + Float64(L_domain[d])), 3)
    prob = C.Problem(; eos=cv(eos, C._default_ideal_mixture(T)),
                     transport=cv(transport, C.ConstantTransport{T}()), metric,
                     sources, domain, bcs, ic=nothing)
    filter = C.StateFilter(cv(filt, C.compact_filter(0.47, T)), filter_interval,
                           filter_cfl, filter_weighting)
    interfaces = C.PatchInterfaces(interface_flux, interface_rhs,
                                   cv(interface_divergence, nothing),
                                   interface_conservative, interface_limited_ghost)
    num = C.Numerics(n_global, cv(deriv, C.lele_d1_6(T)), filter,
                     cv(art, C.ArtificialProperties{T}()), cfl, control, interfaces,
                     C.Execution(dims, comm, backend, T, patch_grid), nothing,
                     polar_truncation, stretch, implicit, positivity_limiter, n_halo)
    amr = C.AMR(; initial=refine, level_restriction, level_interpolation_order,
                subcycle, regrid_interval, tag_threshold, tag_buffer,
                tag_sensor_threshold, tag_gradient_threshold,
                tag_vorticity_threshold, untag_ratio, tile_lifetime, tile, rebalance,
                rebalance_persist, max_levels, level_boundaries, subdivide_lattice)
    return C._Solver(T, prob, num, C.LevelSetup(amr; tag_predicate); equations,
                     extent=L_domain)
end
