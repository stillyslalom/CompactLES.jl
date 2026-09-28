"""
    Presets

Named sets of [`Numerics`](@ref) keywords for regimes in which the defaults
are not the best choice. A preset is a `NamedTuple` of keywords, passed first
and overridden by any keyword given after it:

```julia
Numerics(Presets.resolved(); n_global=(128, 128, 128), cfl=0.4)
```

The defaults are the general shock-capturing configuration and need no
preset. Each preset changes only the settings its regime calls for, and each
rests on a measurement, not a guess; a regime the measurements do not yet
separate from the defaults has no preset. Presets combine by `merge`, the
later one winning where both set a keyword:
`Numerics(merge(Presets.converging(), Presets.refined_shock()); n_global)`.
"""
module Presets

using ..CompactLES: Numerics, StateFilter, StepControl, compact_filter, lele_d1_6, state_guard

"""
    Presets.resolved()

A resolved or smooth flow: decaying turbulence, acoustics, a flow whose
features span many cells. The state filter weakens to
`compact_filter(0.49)`, which fits the Taylor–Green energy history at 128³
and 256³ where the default's dissipation is too strong, and still completes
the shock battery. A shock through a refinement boundary carries more noise
ahead of it under this filter; see [`Presets.refined_shock`](@ref).
"""
resolved() = (; filter=StateFilter(compact_filter(0.49)))

"""
    Presets.refined_shock()

A shock crossing patch or refinement-level interfaces. The state filter
strengthens to `compact_filter(0.45)`, under which a shock through a
coarse–fine interface carries forty times less noise ahead of it than under
the default, and the interfaces take the closure rows
(`patch_interfaces = :closure`), which reach the same shock minima as the
ghost fluxes at 11 to 50% less cost per step. Refinement itself is still
selected with `amr`.
"""
refined_shock() = (; filter=StateFilter(compact_filter(0.45)), patch_interfaces=:closure)

"""
    Presets.converging(; cold_ambient=false)

A shock converging on a cylindrical axis or a spherical origin. The step
control retries a failed step at a lowered CFL four times, which recovers
the positivity excursion at the spherical origin at the default CFL in about
half the steps of a fixed CFL of 0.15; a problem with an `OriginBC` already
takes this when `control` is left unset. `cold_ambient = true` adds
`validity = :permissive`: a shock converging into a cold or near-vacuum gas
carries a few cells of negative internal energy for the whole run and still
reaches the exact plateau, so the state is reported rather than rejected.
Bound the count with a [`state_guard`](@ref) callback.
"""
converging(; cold_ambient::Bool=false) =
    (; control=StepControl(retries=4, validity=cold_ambient ? :permissive : :strict))

"""
    Presets.smooth_walls()

A smooth flow limited by the accuracy at its walls. The derivative takes the
Brady–Livescu closure rows, `lele_d1_6(closures = :brady_livescu)`, which
raise the smooth wall order from about 3 to about 6 under the default filter
and still take a shocked start at a lower CFL ceiling than the default rows.
Use Float64: in Float32 these rows floor a wall derivative near 1e-3. Where
the flow is symmetric about the wall plane, `SymmetryPlaneBC` keeps the
interior order without closure rows and is the better choice.
"""
smooth_walls() = (; deriv=lele_d1_6(closures=:brady_livescu))

end # module Presets
