# CompactLES.jl

CompactLES is a compressible large-eddy/direct-simulation solver built around
high-order compact finite differences, compact filtering, low-storage RK time
integration, and a distributed MPI line solve.

This manual deliberately documents the parts of the package that are already
well established:

- the `Problem`/`Numerics` frontend;
- compact derivative and filter schemes;
- the structured-grid decomposition and MPI execution model;
- time stepping, diagnostics, checkpointing, and VTK output; and
- the numerical regression suite.

The detailed physics model, boundary-condition catalogue, and
coordinate-singularity machinery remain under active development. Their
current design and implementation notes live in
[`DESIGN.md`](https://github.com/stillyslalom/CompactLES.jl/blob/main/DESIGN.md),
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
