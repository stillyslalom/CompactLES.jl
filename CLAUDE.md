# Working in CompactLES

Orientation for coding agents. **This file does not describe the solver** —
`README.md` covers usage and `DESIGN.md` covers the numerics and mechanics
(source map, distributed compact solve, folds, GCL, NSCBC). Read those first
for anything about *what* the code does. This file is about *how to work on
it*: the gate, the conventions, and the traps that have actually cost time.

## Environment

- Julia 1.11.4, MPI.jl 0.20.26 over Microsoft MPI 10.1.12498.52.
- `mpiexec` is **not on PATH**. It lives in the MPI.jl artifact:
  `C:/Users/Alex/.julia/artifacts/4c2053ce1fd1b18af0517bdac687a718f567cdae/bin/mpiexec.exe`
- Dev machine is an i9-12900K: 8 P-cores + 8 E-cores, 24 logical threads.
  Julia defaults to 24 here, so a `-t 24` run is spread across E-cores and is
  **not** a clean scaling measurement. Production targets are Linux clusters,
  which is why `ThreadPinning` is a dependency (not yet wired up — it can only
  be validated on the target).
- Working tree is CRLF (`core.autocrlf=true`). Multi-line string matching in a
  helper script will silently fail to match `\n`-joined text; use the Edit tool
  or match one line at a time.

## The gate

Run all of it before claiming a change is safe. It takes a few minutes.

```bash
MPIEXEC="C:/Users/Alex/.julia/artifacts/4c2053ce1fd1b18af0517bdac687a718f567cdae/bin/mpiexec.exe"

julia --project=. test/runtests.jl        # 28 testsets, must be 0 failures
julia --project=. test/convergence.jl     # measured orders, see below
for np in 2 4 8; do
  "$MPIEXEC" -n $np julia --project=. test/mpi_tests.jl   # 26/26 each
done
julia --project=. bench/jetcheck.jl       # inference, counts below
julia --project=. bench/audit.jl          # allocation + non-concrete SSA counts
```

`test/convergence.jl` prints measured orders with regression guards baked in:
C6 6.01, C10 10.04, C6 wall closures 3.17, cylindrical axis odd 3.71 / even
3.00, resolved-θ 3.71, spherical origin 2.99. **For any refactor that is not
meant to change numerics, these should come out bit-identical down to the
error magnitudes.** A moved digit means you hit something real.

`bench/jetcheck.jl` baseline, per entry point: primitives! 1, deriv_along! 9,
apply_bcs! 1, compute_rhs! 16, compute_dt 2, filter_state! 9, step! 17,
compute_rhs! (axis fold) 16. These overlap heavily — `step!` contains
`compute_rhs!` — so compare entry point to entry point, not the sum. Most of
what remains is `Metric` / `EOS` / `BoundaryCondition` field reads behind
function barriers: one dispatch per array pass, not per point.

Coverage is a separate, slower sequence — see the header of
`bench/coverage.jl`. Serial alone reaches 94.8% of executable lines; the full
set with MPI reaches 97.2%. Note that Julia's `.cov` output omits methods that
were never compiled, so *percentage* flatters you; watch the executable-line
count too.

## Naming

A rename pass (commits `cd8e519`, `aeb2fca`) made the canonical names explicit.
Follow them:

- `solver`, `decomp`, `n_global`, `n_local`, `n_halo`, `n_halo_d`, `offset`,
  `neighbors`, `send_buf`/`recv_buf`, `sub_rank`/`sub_size`
- `n_species`, `n_cons`, `i_mom`, `i_energy`, `Y`, `cp_mix`
- `mu_art`, `beta_art`, `kappa_art`, `D_art`, `C_mu`/`C_beta`/`C_kappa`/`C_D`
- `grad_u`, `grad_T_ion`, `grad_Y`, `strain_mag`, `sensor`, `sensor_sp`
- `inv_J`, `area_d`, `inv_h`, `inv_r`, `cot_over_r`, `coord_shift`, `flux`
- `deriv_plans`, `filter_plans`, `line_solver`, `plan` (a DirPlan) vs `plane`
  (a wall plane), `fold`, `pair`, `pad` (per-dimension halo pad)

**Temperature is `T_ion`**, not `T` — there is one temperature today, and the
name leaves `T_ele` / `T_rad` free for a future 2T or 3T model without a second
API break. Bare `T` is the element-type parameter and nothing else.

Deliberately *not* renamed, do not "fix" these:

- `s` as a band offset in `banded.jl` / `operators_banded.jl` — it is the
  matrix-index convention the surrounding comments define (`Ab[q+1+s, i]`).
- `Q`, `dQ`, `du`, `d` (dimension), `sp` (species), `I` (CartesianIndex),
  `σ`/`σf`/`σg` (parity signs), `ξ` (computational coordinate), `h`, `c`, `p`.

## Traps

**Word-boundary renames.** A `\b`-anchored rule protects an identifier after a
word character but not after punctuation or a digit. Three classes bit this
repo, and none of them fail at compile time:

- `printf "%s"` and regex `"\s"` become `%<newname>` / `\<newname>`.
- Numeric juxtaposition: `2H`, `2Hd`, `2G[1,1]`, `2dec.Hd[d]` have no boundary
  before the identifier, so the rule skips them and leaves an undefined name.
- Prose and units: `ns` is also nanoseconds, `off` is also English, `m/s` is a
  unit, `L_{s,k}` is NSCBC notation.

After any bulk rename, grep for the new name adjacent to `%`, `\`, and digits,
and **re-run every bench script and example** — the test suite does not import
them, so it stays green while they are broken.

**MPI collectives before early returns.** `deriv_along!` and friends are
distributed solves along a dimension: *every* rank must call them. A boundary
routine that returns early on ranks not owning the wall plane will deadlock.
Both `correct_rhs!` methods in `nscbc.jl` hoist their collectives above the
`plane === nothing` return for exactly this reason. Symptom is zero CPU on all
ranks, not a crash.

**Minimum points per rank per dimension.** `plan_direction` errors if the local
extent is too small: C6 needs 5, C10 needs 7, and the **C8 filter needs 9**.
The filter is the binding constraint, so a test grid that is fine for
derivatives can still fail once filtering is on, and transverse extents of
12 or 16 are common in the suite for this reason.

**Julia soft scope.** A top-level `for` loop in a script that reassigns a
variable bound outside it throws `UndefVarError`. Wrap script bodies in a
function (`bench/coverage.jl` does).

**Measurement noise.** Run-to-run spread on this machine is roughly 10–20% for
the same configuration. A 10× result is real; a few-percent result is not
resolved by a single run. Prefer `bench/phases.jl` (phase budget) over a flat
sampling profile — the flat profile is dominated by the compact line solves and
tells you little.

## Threading

`@threaded <work> for ... end` (src/threading.jl) runs `Threads.@threads` only
when `work >= THREAD_MIN_WORK` (32768, override with `CL_THREAD_MIN_WORK`),
otherwise runs the loop serially. Allocation is per-region-per-thread, not
per-point, so small cases used to pay task-spawn cost for nothing — a 1-D case
was 8.3× *slower* at `-t 24` than `-t 1` before the threshold.

**One backend, and it is tasks.** Polyester's `@batch` was measured and
rejected: it wins on single large leaf loops but loses ~8-9× on every phase
driving the compact operators, which run many small regions back to back.
Mixing backends by call depth is catastrophic (58× worse) because Polyester's
spin-waiting workers occupy the threads `Threads.@threads` then targets. The
full measurement is recorded in the `@threaded` docstring — read it before
reaching for `@batch` again.

## Conventions

- Keep source lines under ~95 characters. Four pre-existing lines exceed it
  (two `export` lines, one NSCBC comment, one tridiag comment); don't add more.
- Comments explain *why*, and several encode measurements or derivations
  (`threading.jl` on Polyester, `metric.jl` on the discrete GCL, `folds.jl` on
  the antipodal butterfly, `decomposition.jl` on the `Dims_create` signature).
  Update them when the code moves; do not delete the reasoning.
- `bench/` is scratch tooling, not a test suite: `audit.jl` (allocation and
  inference), `jetcheck.jl`/`jetwhere.jl` (dispatch sites and where they are),
  `phases.jl` (RHS phase budget), `profile.jl`, `scaling.jl`,
  `threadscale.jl`, `bcbench.jl`, `coverage.jl`.
- Run artifacts (`*.dat`, `*.vtr`, `*.pvtr`, `*.ckpt`, `*.cov`) are gitignored;
  `git add -A` after running examples is how one got committed once.

## Open items

- `compute_artificial!` is ~31% of the multicomponent RHS. The cost is the
  filter line-solves in the sensor smoothing, one sweep per species. Reducing
  it is a numerics decision (shared vs per-species sensor), not a code tweak.
  Note that at `n_species == 2` the per-species machinery is measurably a
  no-op: on a sharp binary interface `D*_1` and `D*_2` agree to 4.8e-16
  relative, and the correction velocity comes out 2.5e-18 against a species
  gradient of 84. It only earns its cost at three or more species.
- `ThreadPinning` is a dependency but unused; only validatable on Linux.
- A `bench/` runner taking medians over repeated *processes*, to get under the
  run-to-run noise above.
- `FoldSpec` parameterization would close ~6 of the remaining JET sites.
