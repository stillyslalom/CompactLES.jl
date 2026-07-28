# A boundary that changes type

This case studies a two-dimensional Richtmyer--Meshkov interaction whose
upstream boundary must change from subsonic inflow to subsonic outflow. It is a
scientific case study rather than a default tutorial: reproducing the complete
comparison evolves two `160 x 80` shock calculations and takes about 100 seconds
on the development workstation after package setup.

The complete Literate source is
[`docs/case_studies/shock_tube_2d.jl`](https://github.com/stillyslalom/CompactLES.jl/blob/main/docs/case_studies/shock_tube_2d.jl).
It is intentionally excluded from the default documentation build.

## Physical reason for switching

The incident shock is initialized inside the domain. Holding its post-shock
state at the upstream face requires `NSCBCInflowBC`. When the shock crosses a
perturbed light/heavy interface, it emits a reflected wave travelling back
toward that face.

An inflow condition continues to relax incoming characteristics toward the
post-shock target. After the reflected wave arrives, that target no longer
describes the boundary state. Retaining it returns a spurious disturbance to
the interface. Switching to a weakly pressure-relaxed `NSCBCOutflowBC` allows
the reflected wave to leave.

## Event definition

The switch is triggered when density anywhere on the upstream plane exceeds
the post-shock density by six percent. The threshold lies above a smaller
startup acoustic transient and below the reflected-shock jump.

`boundary_plane` returns the face only on ranks that own it. The predicate is
therefore rank-local, while `WhenState` reduces its Boolean result over the
communicator:

```julia
arrived = WhenState((solver, Q) -> begin
    plane = boundary_plane(solver, 1, 1)
    plane === nothing && return false
    maximum(I -> mixture_density(solver, Q, I), plane) > threshold
end)

change = Callback(arrived, (solver, Q) -> switch!(upstream))
```

Global agreement is required because the outflow correction enters collective
compact derivatives that the inflow path does not. Rank disagreement would
deadlock rather than merely produce different boundary values.

## Control calculation

The case repeats the calculation with the original inflow retained. Both runs
are identical until the switch, so their subsequent density difference
isolates the disturbance returned by the inappropriate boundary. The error
propagates to the material interface rather than remaining confined to a few
boundary cells.

## Evidence and limits

The switch behavior is also compared in `test/runtests.jl` with a domain long
enough that the upstream face cannot influence the interaction during the
measurement interval. That provides stronger evidence than visual smoothness
at the truncated boundary.

This case does not establish universal non-reflection. NSCBC performance
depends on Mach number, wave angle and spectrum, relaxation strength, and
transverse terms. A new application should compare its quantity of interest
with a longer domain or an independent boundary treatment.

## Regeneration policy

The extended script should be rerun manually after changes to NSCBC,
`SwitchableBC`, callbacks, artificial properties, filtering, or time
integration. Published outputs should record the source commit, Julia version,
grid, CFL, artificial coefficients, filter settings, and runtime environment.
