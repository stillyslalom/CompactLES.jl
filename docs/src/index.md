# CompactLES.jl

CompactLES is a [Pyranda](https://github.com/LLNL/pyranda)-inspired 
compressible large-eddy/direct-simulation solver built around
high-order compact finite differences, compact filtering, low-storage RK time
integration, and a distributed MPI line solve.

![Taylor–Green vorticity, a multicomponent shock-interface interaction, and a
cylindrical converging shock](assets/readme_header.png)

The `64³` Taylor–Green run peaks at `ε = 0.01265`, `t = 9.13`, closely matching
the pseudo-spectral reference peak `ε = 0.01289`, `t = 8.86`.

This manual documents the established package interfaces:

- the `Problem`/`Numerics` frontend;
- compact derivative and filter schemes;
- the structured-grid decomposition and MPI execution model;
- time stepping, diagnostics, checkpointing, and VTK output; and
- the numerical regression suite.

The detailed physics model, boundary-condition catalogue, and
coordinate-singularity machinery remain under active development. Their
current design and implementation notes live in
[`DESIGN.md`](https://github.com/stillyslalom/CompactLES.jl/blob/main/reference/DESIGN.md),
and the source remains authoritative while those interfaces evolve.

## Where to begin

Use [Getting started](@ref) to construct and advance a problem. Read
[Compact operators](@ref) before changing spatial schemes, and
[Parallel decomposition](@ref) before choosing a process grid. The
[Validation](@ref) page records the checks that protect the established
numerics.

!!! warning "Research software"
    CompactLES is research code under active development. Validate a
    configuration against an analytic or experimental reference before using
    its results for scientific conclusions.
