# Allocation and type-inference audit of the hot paths.
#
#   julia --project=. -t auto bench/audit.jl
#
# Three questions, answered per call site:
#   1. How many bytes does a steady-state call allocate? Anything that scales
#      with the grid is a bug; a small constant is threading/closure overhead.
#   2. Does inference produce concrete types? Reported as the number of
#      non-concrete slots @code_warntype would colour red.
#   3. Does a Float32 run compute in Float32? Reported per point body as the
#      number of Float64 values in its optimized code.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf, InteractiveUtils

const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

# --- a representative solver: 3-D periodic, artificial properties live ------
function build(; N=48, n_species=1, art=true, deriv=lele_d1_6(),
               species_flux=:fickian)
    eos = n_species == 1 ? IdealSpecies("gas"; R=1.0, gamma=1.4) :
          IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    solver = Solver(n_global=(N, N, N), L_domain=(2π, 2π, 2π), bcs=per3, eos=eos,
               transport=Transport(mu0=1e-3), deriv=deriv,
               art=ArtParams(enabled=art, species_flux=species_flux))
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

println("\n=== allocation per call (48^3, two species, bulk species channel) ===")
sb, Qb = build(n_species=2, species_flux=:bulk)
dQb = zero(Qb); dub = zero(Qb)
alloc("compute_rhs! (2 species, bulk)", () -> compute_rhs!(sb, Qb, dQb); scale=npt)
alloc("step!        (2 species, bulk)", () -> step!(sb, Qb, dQb, dub, 1e-4); scale=npt)

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
    ("compute_rhs! (bulk)", compute_rhs!,   Tuple{typeof(sb), typeof(Qb), typeof(dQb), Bool}),
]
for (name, f, T) in probes
    n, bad = badtypes(f, T)
    @printf("  %-24s  %4d non-concrete SSA values\n", name, n)
end

# --- precision probe --------------------------------------------------------
# Float64 values in the per-point bodies of Float32 runs. Each run below
# compiles the bodies it launches at Float32 argument types; a Float64 SSA
# value in a body's optimized code is a Float64 operand (a literal, or a
# component stored at Float64) promoting Float32 arithmetic. Code behind a
# call the optimizer does not inline is not seen.
function float64_in_point_bodies()
    per = (PeriodicBC(), PeriodicBC())
    iso = (NoSlipWallBC(Twall=1.0), NoSlipWallBC(Twall=1.0))
    mix = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                        IdealSpecies{Float64}("b", 0.2, 1.09)])
    wave(x, y, z) = Prim(u=(0.1sin(x), 0.05cos(y), 0.0), p=1 + 0.05cos(z),
                         rho=1 + 0.1sin(y))
    pair(x, y, z) = Prim(Y=(0.6 + 0.1sin(x), 0.4 - 0.1sin(x)),
                         u=(0.0, 0.05sin(y), 0.0), p=1.0, rho=1 + 0.1sin(y))
    runs = [
        ((n_global=(24, 24, 24), L_domain=(2π, 2π, 2π), bcs=per3,
          transport=Transport(mu0=1e-3), art=ArtParams(enabled=true)), wave),
        ((n_global=(24, 16, 1), L_domain=(1.0, 2π, 1.0), bcs=(iso, per, per),
          eos=mix, transport=Transport(mu0=1e-3),
          art=ArtParams(enabled=true, species_flux=:bulk),
          sources=(ConstantBodyForce((0.0, -1.0, 0.0)),)), pair),
        ((n_global=(48, 1, 1), L_domain=(1.0, 1.0, 1.0),
          bcs=((NSCBCInflowBC(u=(0.1, 0.0, 0.0), T_ion=1.0),
                NSCBCOutflowBC(pinf=1.0)), per, per),
          art=ArtParams(enabled=true)),
         (x, y, z) -> Prim(u=(0.1, 0.0, 0.0), p=1.0, rho=1.0)),
        ((n_global=(32, 1, 1), L_domain=(1.0, 1.0, 1.0), bcs=per3,
          eos=StiffenedGas(gamma=4.4, p_inf=1.0, cv=1.0), art=ArtParams(enabled=true)),
         (x, y, z) -> Prim(u=(0.1, 0.0, 0.0), p=1 + 0.1sin(2π * x), rho=1.0)),
        ((n_global=(32, 1, 1), L_domain=(1.0, 1.0, 1.0), metric=CylindricalMetric(),
          bcs=((AxisBC(), SlipWallBC()), per, per), art=ArtParams(enabled=true)),
         (r, θ, z) -> Prim(u=(0, 0, 0), p=1 + 0.1exp(-40(r - 0.4)^2), rho=1.0)),
    ]
    for (deck, ic) in runs
        s = Solver(; precision=Float32, cfl=0.3, deck...)
        Q = allocate_state(s)
        initialize!(s, Q, ic)
        run!(s, Q; tfinal=1.0, nmax=2)
    end
    counts = Pair{Symbol,Int}[]
    scanned = 0
    for n in names(CL; all=true)
        endswith(string(n), "_point!") || continue
        f = getfield(CL, n)
        f isa Function || continue
        k = 0
        for m in methods(f), mi in Base.specializations(m)
            mi === nothing && continue
            occursin("Float32", string(mi.specTypes)) || continue
            scanned += 1
            for (ci, _) in Base.code_typed_by_type(mi.specTypes; optimize=true)
                k += count(t -> Core.Compiler.widenconst(t) === Float64,
                           ci.ssavaluetypes)
            end
        end
        k > 0 && push!(counts, n => k)
    end
    return counts, scanned
end

println("\n=== precision: Float64 SSA values in Float32 point bodies ===")
promoted, scanned = float64_in_point_bodies()
@printf("  %d Float32 specializations scanned\n", scanned)
isempty(promoted) && println("  none carries a Float64 value")
for (n, k) in promoted
    @printf("  %-32s %4d\n", n, k)
end

println("\naudit complete")
