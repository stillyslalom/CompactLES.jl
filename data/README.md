# Vendored NASA CEA data

Read by [`read_nasa9`](../src/nasa9_data.jl) to build a `Nasa9Mixture`. Only the
two tables are used by the solver; the license files are here because
redistribution requires them.

| File | What it is | Used by |
|---|---|---|
| `thermo.inp` | NASA-9 thermodynamic polynomial coefficients, piecewise in T | `read_nasa9` |
| `trans.inp` | Viscosity and thermal conductivity fits | nothing yet — see below |
| `LICENSE.txt` | Apache License 2.0, verbatim | — |
| `NOTICE.txt` | Upstream NOTICE, verbatim | — |

`NOTICE.txt` is reproduced unchanged, as Apache 2.0 §4(d) requires. It therefore
lists upstream paths that are **not** vendored here (`data/thermo.lib`,
`data/trans.lib`, `samples/*`, `source/bind/python/cea/samples/*`) — those belong
to the full CEA distribution, not to this checkout. The four files above are
everything that was taken.

Source: NASA TP-2002-211556, <https://ntrs.nasa.gov/citations/20020085330>.

## Coverage

`thermo.inp` holds 1276 gaseous multi-interval species, all of which parse; the
reader rejects condensed phases and the reactant-only records that carry a heat
of formation with no fit. `trans.inp` holds 66 pure species and 41 binary
interaction pairs, and supplies viscosity and conductivity but **no diffusion
coefficients**.

`trans.inp` is not connected to `Transport`, which still uses constant
properties. The rationale for deferring that integration is recorded in the
`Transport` entry under **Known limitations** in `CLAUDE.md`.
