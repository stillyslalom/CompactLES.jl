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
| `trans.inp` | Viscosity and thermal conductivity fits | `read_cea_transport`, `CeaTransport` |
| `LICENSE.txt` | Apache License 2.0, verbatim | — |
| `NOTICE.txt` | Upstream NOTICE, verbatim | — |

`NOTICE.txt` is reproduced unchanged, as Apache 2.0 §4(d) requires. It therefore
lists upstream paths that are **not** vendored here (`data/thermo.lib`,
`data/trans.lib`, `samples/*`, `source/bind/python/cea/samples/*`); those belong
to the full CEA distribution, not to this checkout. The four files above are
everything that was taken.

Source: NASA TP-2002-211556, <https://ntrs.nasa.gov/citations/20020085330>.

## Coverage

`thermo.inp` holds 1276 gaseous multi-interval species, all of which parse; the
reader rejects condensed phases and the reactant-only records that carry a heat
of formation with no fit. `trans.inp` holds 66 pure species and 41 binary
interaction pairs, and supplies viscosity and conductivity but **no diffusion
coefficients**.

`CeaTransport` connects the pure-species fits to temperature-dependent mixture
properties. Its default diffusion model is unity Lewis; mixture-averaged
diffusion requires separately supplied binary diffusivities. The original
`Transport` remains the constant-viscosity, single-Schmidt model.

The binary diffusion correlations of Marrero and Mason, J. Phys. Chem. Ref.
Data 1, 3--118 (1972), Tables 12 and 13, are not a vendored file: they are
transcribed as source constants in `src/neutral_diffusion_data.jl`, the
printed digits converted to SI on construction, with each pair's stated range
and reliability group, from the NIST reprint at <https://www.nist.gov/system/files/documents/srd/jpcrd1.pdf>.
The transcription was made twice independently and diffed. The paper is a
National Standard Reference Data System evaluation whose copyright the
issue assigns to AIP and ACS; the correlation parameters are the recommended
values themselves, reproduced with attribution and without the paper's text,
figures or data tables.

Two more sources sit beside it in the same file. Song et al., J. Chem. Eng.
Data 61, 1910 (2016), tabulate calculated hydrogen-isotopologue and helium
pairs in their supporting information (DOI `10.1021/acs.jced.6b00076.s001`,
ACS Figshare record 3208147/file 5037301). `songwang_extract.jl` fits each
pair from a PyMuPDF text dump. `songwang_verify.jl` independently parses
Xpdf `pdftotext -layout` output and, given the user-local publisher PDF, checks the
artifact hash, all 78 mixture tables and 702 equimolar nodes against the
vendored fits, pair identities, composition spread and the printed 2.0%
expanded uncertainty. The verified PDF has SHA-256
`e93ee0a69c8f1d976e321065efe9bd3042a6352630a5e589d4e049587271de9a`
(publisher MD5 `d73f257d12bc84d99ba9f2a50f298175`). Only fits and diagnostics
are vendored, not the source tables; the Figshare record labels the supporting
artifact CC BY-NC 4.0. Müller and Klemm, Z. Naturforsch. 25a,
243 (1970), Tab. 1, is nine
measured room-temperature values transcribed twice from the page image.

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
