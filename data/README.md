# Vendored data

Third-party data used by the package or by the calibration benchmarks. Each file
is verbatim under its upstream name.

## NASA CEA thermodynamic and transport databases

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

## Taylor-Green reference solution

| File | What it is | Used by |
|---|---|---|
| `spectral_Re1600_512.gdiag` | 512³ dealiased pseudo-spectral Taylor-Green at Re = 1600 | `bench/tgv_energy.jl` |

Four whitespace-separated columns under two comment lines: time, volume-averaged
kinetic energy, dissipation rate `-dE/dt`, and enstrophy, from `t = 0` to
`t = 19.99` at `dt = 0.01`. Energy is normalized as
`(1/(2π)³) ∫ ρ|u|²/2 dV`, so it starts at 0.125 and is directly comparable to
`kinetic_energy` in `bench/tgv_energy.jl`; the dissipation column satisfies
`ε = 2ν ζ` against the enstrophy column at `ν = 1/1600`.

Source: the reference solution distributed with the International Workshops on
High-Order CFD Methods for the Taylor-Green vortex case, retrieved from
<http://www.as.dlr.de/hiocfd/spectral_Re1600_512.gdiag>. It carries no license
statement; it is published as benchmark reference data for exactly this
comparison and is redistributed here unmodified so that a calibration result is
reproducible from the checkout alone.

The tabulated peak dissipation is 1.28575e-2 at `t = 8.97`. That agrees with the
1.289e-2 at `t = 8.86` digitized from Fig. 8 of van Rees et al., JCP 230 (2011),
<https://doi.org/10.1016/j.jcp.2010.11.031>, to 0.3% in value and 0.11 in time,
which is figure-reading error, so the table and that figure are the same
solution. The rounded `1.2e-2 at t = 9` quoted throughout the earlier
calibration work is 6.7% below the tabulated peak.

The reference is incompressible. This solver runs the case at Ma 0.1, so the two
kinetic energies differ by the pressure-dilatation exchange, which is O(Ma²) and
appears early while the flow is still smooth.
