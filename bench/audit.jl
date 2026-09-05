# Allocation and type-inference audit of the hot paths.
#
#   julia --project=. -t auto bench/audit.jl
#
# Two questions, answered per call site:
#   1. How many bytes does a steady-state call allocate? Anything that scales
#      with the grid is a bug; a small constant is threading/closure overhead.
#   2. Does inference produce concrete types? Reported as the number of
#      non-concrete slots @code_warntype would colour red.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf, InteractiveUtils

const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

# --- a representative solver: 3-D periodic, artificial properties live ------
function build(; N=48, n_species=1, art=true, deriv=lele_d1_6())
    eos = n_species == 1 ? IdealSpecies("gas"; R=1.0, gamma=1.4) :
          IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = Solver(n_global=(N, N, N), L_domain=(2π, 2π, 2π), bcs=per3, eos=eos,
               transport=Transport(mu0=1e-3), deriv=deriv,
               art=ArtParams(enabled=art))
    Q = allocate_state(solver)
    if n_species == 1
        initialize!(solver, Q, (x, y, z) -> Prim(u=(0.1sin(x) * cos(y), -0.1cos(x) * sin(y), 0.05sin(z)),
                                            p=1 + 0.05cos(x) * cos(z), rho=1 + 0.1sin(y)))
    else
        initialize!(solver, Q, (x, y, z) -> Prim(Y=(0.6, 0.4), u=(0.1sin(x), 0, 0),
                                            p=1 + 0.05cos(z), rho=1 + 0.1sin(y)))
    end
    solver, Q
end

# --- allocation probes ------------------------------------------------------
"Run f() twice to warm up, then report the steady-state allocation."
function alloc(name, f; scale=1)
    f(); f()
    b = @allocated f()
    b2 = @allocated f()
    b = min(b, b2)
    tag = b > 1_000_000 ? "  <-- SCALES WITH GRID?" : ""
    @printf("  %-42s %10d B  (%7.2f B/pt)%s\n", name, b, b / scale, tag)
    b
end

println("\n=== allocation per call (48^3, single species, art on) ===")
solver, Q = build()
dQ = zero(Q); du = zero(Q)
npt = prod(solver.decomp.n_local)
@printf("  grid points: %d\n", npt)
alloc("primitives!(solver, Q)", () -> CL.primitives!(solver, Q); scale=npt)
alloc("exchange_state!(Q, decomp)", () -> CL.exchange_state!(Q, solver.decomp); scale=npt)
alloc("deriv_along!(tmp_a, rho, solver, 1, 1)", () -> CL.deriv_along!(solver.tmp_a, solver.rho, solver, 1, 1); scale=npt)
alloc("deriv_along!(tmp_a, rho, solver, 2, 1)", () -> CL.deriv_along!(solver.tmp_a, solver.rho, solver, 2, 1); scale=npt)
alloc("compute_artificial!(solver, Q)", () -> CL.compute_artificial!(solver, Q); scale=npt)
alloc("compute_rhs!(solver, Q, dQ)", () -> compute_rhs!(solver, Q, dQ); scale=npt)
alloc("apply_bcs!(solver, Q)", () -> apply_bcs!(solver, Q); scale=npt)
alloc("compute_dt(solver, Q)", () -> compute_dt(solver, Q); scale=npt)
alloc("filter_state!(solver, Q)", () -> filter_state!(solver, Q); scale=npt)
alloc("step!(solver, Q, dQ, du, dt)", () -> step!(solver, Q, dQ, du, 1e-4); scale=npt)

println("\n=== allocation per call (48^3, two species, C10, art on) ===")
s2, Q2 = build(n_species=2, deriv=lele_d1_10())
dQ2 = zero(Q2); du2 = zero(Q2)
alloc("compute_rhs! (2 species, C10)", () -> compute_rhs!(s2, Q2, dQ2); scale=npt)
alloc("step!        (2 species, C10)", () -> step!(s2, Q2, dQ2, du2, 1e-4); scale=npt)

println("\n=== allocation per call (cylindrical axis fold, 1-D radial) ===")
sf = Solver(n_global=(128, 1, 1), L_domain=(1.0, 1.0, 1.0), metric=CylindricalMetric(),
            bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
            art=ArtParams(enabled=true))
Qf = allocate_state(sf)
initialize!(sf, Qf, (r, θ, z) -> Prim(u=(0, 0, 0), p=1 + exp(-40(r - 0.4)^2), rho=1.0))
dQf = zero(Qf); duf = zero(Qf)
alloc("compute_rhs! (axis fold)", () -> compute_rhs!(sf, Qf, dQf); scale=128)
alloc("step!        (axis fold)", () -> step!(sf, Qf, dQf, duf, 1e-4); scale=128)

# --- inference probes -------------------------------------------------------
"Count non-concrete slot/ssa types in the inferred code for f(args...)."
function badtypes(f, types)
    out = code_typed(f, types; optimize=false)
    isempty(out) && return (-1, String[])
    ci = first(out).first
    bad = String[]
    for (i, t) in enumerate(ci.slottypes)
        # slot 1 is the function itself; #self# and unused slots are noise
        isconcretetype(t) || t === Any || push!(bad, "slot$i::$t")
    end
    n = 0
    for t in ci.ssavaluetypes
        (t isa Type && !isconcretetype(t) && t !== Union{}) && (n += 1)
    end
    (n, bad)
end

println("\n=== inference: non-concrete SSA values (lower is better) ===")
# Spell out every optional trailing argument. A method with a default generates a
# short forwarding method at the shorter arity, and `code_typed` on that arity
# returns the forward: one SSA value, and no sight of the body. `compute_rhs!` and
# `step!` both carry a trailing `Bool` (see their docstrings), so probing them
# without it would report a count of 1 and hide every regression.
probes = [
    ("primitives!",        CL.primitives!,  Tuple{typeof(solver), typeof(Q)}),
    ("recover_primitives!", CL.recover_primitives!, Tuple{typeof(solver), typeof(solver.eos), typeof(Q)}),
    ("compute_rhs!",       compute_rhs!,    Tuple{typeof(solver), typeof(Q), typeof(dQ), Bool}),
    ("compute_dt",         compute_dt,      Tuple{typeof(solver), typeof(Q)}),
    ("step!",              step!,           Tuple{typeof(solver), typeof(Q), typeof(dQ), typeof(du), Float64, Bool}),
    ("deriv_along!",       CL.deriv_along!, Tuple{typeof(solver.tmp_a), typeof(solver.rho), typeof(solver), Int, Int}),
    ("filter_state!",      filter_state!,   Tuple{typeof(solver), typeof(Q)}),
    ("apply_bcs!",         apply_bcs!,      Tuple{typeof(solver), typeof(Q)}),
]
for (name, f, T) in probes
    n, bad = badtypes(f, T)
    @printf("  %-24s  %4d non-concrete SSA values\n", name, n)
end

println("\naudit complete")
