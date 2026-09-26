# Public API index

This index contains the exported input-deck and runtime bindings, and those of
the `CompactLES.Regions` and `CompactLES.DiffusionData` submodules, which are
loaded with their own `using`. The reference pages also document two narrower
supported surfaces: advanced numerical operations and extension hooks, the
latter declared `public` and reached as `CompactLES.name`. Developer internals
may be rendered on those pages to support cross-references, but are neither
exported nor public. The `MPI` module is re-exported as well; it is
documented by MPI.jl and therefore does not appear in the index below.

```@meta
CurrentModule = CompactLES
```

```@docs
CompactLES.CompactLES
```

```@index
Modules = [CompactLES, CompactLES.Regions, CompactLES.DiffusionData]
Order = [:type, :function, :constant, :macro]
```
