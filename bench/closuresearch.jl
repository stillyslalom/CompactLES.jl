module ClosureSearch

# Deterministic research instrument for fifth-order boundary closures of the
# Lele C6 tridiagonal first derivative.  This file is intentionally independent
# of the production presets: it may be included by qualification scripts.

using CompactLES
using LinearAlgebra
using Printf
using Random

const CL = CompactLES

export fifth_order_row, fifth_order_family, extended_fifth_order_family, candidate_parameters,
       candidate_scheme, unfiltered_scheme, de_scheme, derivative_matrix, acoustic_operator, diagnostics,
       growth_sweep, filtered_radius, feasibility_violation, differential_evolution,
       assembly_mismatch, search, main

const MOMENT_AFFINE = let
    V = Rational{BigInt}[Rational{BigInt}(k - 1)^p for p in 0:5, k in 1:6]
    Vinv = inv(V)
    ntuple(4) do j
        x = Rational{BigInt}(j - 1)
        rhs(s, q) = Rational{BigInt}[p == 0 ? 0 : p *
            (s * (x - 1)^(p - 1) + x^(p - 1) + q * (x + 1)^(p - 1))
            for p in 0:5]
        w0 = Vinv * rhs(0, 0)
        (w0, Vinv * rhs(1, 0) - w0, Vinv * rhs(0, 1) - w0)
    end
end

"""Construct a fifth-order row at node `j` from its LHS off-diagonals.

The RHS uses nodes 1:6. Its weights come from an exact rational affine basis
that enforces exactness on monomials of degree 0:5. `sub` must be zero at j=1.
"""
function fifth_order_row(::Type{T}, j::Integer, sub::Real, super::Real) where {T}
    1 <= j <= 4 || throw(ArgumentError("j must be in 1:4"))
    j == 1 && !iszero(sub) && throw(ArgumentError("row 1 subdiagonal must be zero"))
    lhs = (T(sub), one(T), T(super))
    w0, ws, wq = MOMENT_AFFINE[j]
    CL.ClosureRow{T}(lhs, T.(w0) .+ T(sub) .* T.(ws) .+ T(super) .* T.(wq))
end

"""Four fifth-order rows from row-1 `super`, then three `(sub,super)` pairs."""
function fifth_order_family(::Type{T}, parameters) where {T}
    length(parameters) == 7 || throw(ArgumentError("expected 7 parameters"))
    pairs = ((0, parameters[1]),
             (parameters[2], parameters[3]),
             (parameters[4], parameters[5]),
             (parameters[6], parameters[7]))
    [fifth_order_row(T, j, pairs[j]...) for j in 1:4]
end

# Brady--Livescu T6 expressed in this family's seven free LHS coordinates. It is
# the conservative, neutrally stable control and deterministic search seed.
const BL_PARAMETERS = Float64[
                        6.736832494852786,
    0.4885251620537965, 2.7185849538712983,
   -0.3891997445794,   -1.1117328224921332,
   -0.5719411698333021,-0.11930391824998438,
]

# Best reproducible result is updated only from a measured search run.  Keeping
# a separate binding makes the constructor's interface stable for qualifiers.
const CANDIDATE_PARAMETERS = Float64[
     6.44549783494442,
     0.5895659645871965,  3.0445191662818054,
    -0.29695415441823525,-1.3290740341334466,
    -0.6671659659185682, -0.3155538974259181,
]
const UNFILTERED_ONLY_PARAMETERS = Float64[
     6.151263016311877,
     0.5195089207724447,  2.750826276249132,
    -0.052963961495487745,-1.8799327335582026,
    -1.3133286079021156, -0.9951264983380909,
]
const EXTENDED_BL_PARAMETERS = [BL_PARAMETERS; zeros(4)]
const EXTENDED_COMBINED_PARAMETERS = [CANDIDATE_PARAMETERS; zeros(4)]
const EXTENDED_REJECTED_PARAMETERS = Float64[
    5.865108300073922, 0.7148278140675014, 2.560882104279839,
   -0.425957889629787, -0.7460618884054762, -0.49589740860778764,
    0.462617680701808, 0.0002402835560477558, -0.0046302932937616675,
   -0.0039561728150460515, -0.0005473929487322502,
]
const EXTENDED_DE_PARAMETERS = Float64[
    7.012328805745522, 0.6958257718110491, 3.127936381311237,
   -0.3168341650243765, -1.5198347055081192, -0.6026583301990742,
   -0.23600464182577716, 0.0003537099191629179, 0.0027985458209475937,
    0.002103477821479526, -0.0007994045741576889,
]
const EXTENDED_FEASIBLE_SMALL_PARAMETERS = Float64[
    7.156459972509367, 0.6323236119854221, 2.877900223287518,
   -0.21138702841158744, -1.7063879918846345, -0.6357547501751779,
   -0.32613617214582513, -0.0003047194454015689, 0.004429430394889424,
    0.0026707913832941135, -0.0011088799189850822,
]
const EXTENDED_DE_FINAL_PARAMETERS = Float64[
    6.956459972509367, 0.6794866541559612, 2.677900223287518,
   -0.22613805071814658, -1.9063879918846345, -0.6489148076249647,
   -0.515310296896472, 0.0005902020384870031, 0.005347848654867536,
    0.003081657613812831, -0.002108879918985082,
]
candidate_parameters() = copy(CANDIDATE_PARAMETERS)

function _scheme(::Type{T}, parameters, name) where {T}
    CL.CompactScheme{T}(name, T(1//3),
        zero(T), T[7//9, 1//36], false,
        fifth_order_family(T, parameters))
end
function _trial_scheme(parameters, name="trial")
    rows = length(parameters) == 7 ? fifth_order_family(Float64, parameters) :
           length(parameters) == 11 ? extended_fifth_order_family(Float64, parameters) :
           throw(ArgumentError("expected 7 or 11 parameters"))
    CL.CompactScheme{Float64}(name, 1/3, 0.0, [7/9, 1/36], false, rows)
end
candidate_scheme(::Type{T}=Float64) where {T} = _scheme(T, CANDIDATE_PARAMETERS,
    "C6 experimental fifth-order searched closure")
unfiltered_scheme(::Type{T}=Float64) where {T} = _scheme(T, UNFILTERED_ONLY_PARAMETERS,
    "C6 fifth-order unfiltered-only rejected closure")
function de_scheme(::Type{T}=Float64) where {T}
    rows = extended_fifth_order_family(T, EXTENDED_DE_FINAL_PARAMETERS)
    CL.CompactScheme{T}("C6 experimental filtered-only DE fifth-order closure",
                        T(1//3), zero(T), T[7//9, 1//36], false, rows)
end

"""Seven-point fifth-order rows with four additional moment-null coordinates.

The first seven parameters are the LHS coordinates of `fifth_order_family`;
parameters 8:11 multiply the sixth-forward-difference vector in rows 1:4.
Thus every member remains exact through degree five and the all-zero extension
of `BL_PARAMETERS` reconstructs the six-point BL seed exactly.
"""
function extended_fifth_order_family(::Type{T}, parameters) where {T}
    length(parameters) == 11 || throw(ArgumentError("expected 11 parameters"))
    base = fifth_order_family(T, @view parameters[1:7])
    null = T[1, -6, 15, -20, 15, -6, 1]
    [CL.ClosureRow{T}(base[j].lhs,
        [base[j].rhs; zero(T)] .+ T(parameters[7+j]) .* null) for j in 1:4]
end

"""Assemble the closed-line derivative exactly as `A ÷ B` (unit spacing)."""
function derivative_matrix(scheme, N::Integer)
    nr = length(scheme.closures)
    N >= 2nr + 1 || throw(ArgumentError("N must be at least $(2nr + 1)"))
    T = eltype(scheme.coeffs)
    A = Matrix{T}(I, N, N)
    B = zeros(T, N, N)
    for j in 1:nr
        row = scheme.closures[j]
        A[j, j] = row.lhs[2]
        j > 1 && (A[j, j - 1] = row.lhs[1])
        j < N && (A[j, j + 1] = row.lhs[3])
        B[j, row.first:row.first + length(row.rhs) - 1] .= row.rhs
        hi = N + 1 - j
        A[hi, hi] = row.lhs[2]
        hi < N && (A[hi, hi + 1] = row.lhs[1])
        hi > 1 && (A[hi, hi - 1] = row.lhs[3])
        B[hi, N + 1 - row.first:-1:N + 2 - row.first - length(row.rhs)] .= -row.rhs
    end
    for j in nr + 1:N - nr
        A[j, j - 1] = scheme.alpha
        A[j, j + 1] = scheme.alpha
        for m in eachindex(scheme.coeffs)
            B[j, j + m] = scheme.coeffs[m]
            B[j, j - m] = -scheme.coeffs[m]
        end
    end
    return A \ B, A
end

"""Injected slip-wall acoustic operator on `(p,u_interior)` on a unit domain.

Eliminating the two prescribed endpoint velocities avoids the defective zero
modes of a full matrix whose injected rows have merely been zeroed.
"""
function acoustic_operator(scheme, N::Integer)
    D, A = derivative_matrix(scheme, N)
    D .*= N - 1                 # h = 1/(N-1), rates are physical c/L units
    Zpp = zeros(eltype(D), N, N)
    Zuu = zeros(eltype(D), N - 2, N - 2)
    L = [Zpp -D[:, 2:N-1]; -D[2:N-1, :] Zuu]
    return L, A
end

function filter_matrix(N::Integer; alphaf=0.45)
    scheme = CL.compact_filter(alphaf)
    nr = length(scheme.closures); T = Float64
    A = Matrix{T}(I, N, N); B = zeros(T, N, N)
    for j in 1:nr
        row = scheme.closures[j]; hi = N + 1 - j
        A[j,j] = row.lhs[2]; A[hi,hi] = row.lhs[2]
        j > 1 && (A[j,j-1] = row.lhs[1]); j < N && (A[j,j+1] = row.lhs[3])
        hi < N && (A[hi,hi+1] = row.lhs[1]); hi > 1 && (A[hi,hi-1] = row.lhs[3])
        B[j, row.first:row.first+length(row.rhs)-1] .= row.rhs
        B[hi, N+1-row.first:-1:N+2-row.first-length(row.rhs)] .= row.rhs
    end
    for j in nr+1:N-nr
        A[j,j-1] = scheme.alpha; A[j,j+1] = scheme.alpha
        B[j,j] = scheme.a0
        for m in eachindex(scheme.coeffs)
            B[j,j-m] = scheme.coeffs[m]; B[j,j+m] = scheme.coeffs[m]
        end
    end
    A \ B
end

const FILTER_BLOCK_CACHE = Dict{Tuple{Int,Float64},Matrix{Float64}}()
const FILTER_MATRIX_CACHE = Dict{Tuple{Int,Float64},Matrix{Float64}}()

base_filter(N, alphaf) = get!(() -> filter_matrix(N; alphaf),
                              FILTER_MATRIX_CACHE, (N, Float64(alphaf)))

function filter_block(N, alphaf)
    get!(FILTER_BLOCK_CACHE, (N, Float64(alphaf))) do
        F = base_filter(N, alphaf)
        [F zeros(N,N-2); zeros(N-2,N) F[2:N-1,2:N-1]]
    end
end

function production_matrix(scheme, N::Integer)
    decomp = CL.Decomp{Float64}((N, 1, 1), (false, true, true))
    plan = CL.plan_direction(decomp, scheme, 1, 1.0)
    P = zeros(N, N)
    for k in 1:N
        f = zeros(N); f[k] = 1
        b = zeros(N); nr = length(plan.clo); nh = length(plan.chi)
        for j in 1:nr
            first = plan.clo_first[j]
            b[j] = sum(plan.clo[j][q] * f[first + q - 1] for q in eachindex(plan.clo[j]))
        end
        for j in 1:nh
            first = N + 2 - plan.chi_first[j]
            b[N + 1 - j] = sum(plan.chi[j][q] * f[first - q] for q in eachindex(plan.chi[j]))
        end
        for j in nr+1:N-nh
            b[j] = plan.a0 * f[j]
            for m in eachindex(plan.ci)
                b[j] += plan.ci[m] * (scheme.symmetric ? f[j+m] + f[j-m] : f[j+m] - f[j-m])
            end
        end
        work = reshape(b, N, 1)
        CL.solve_lines!(work, plan.line_solver)
        P[:, k] .= vec(work)
    end
    P
end

"""Maximum basis-vector mismatch against production plan fill and solve."""
function assembly_mismatch(parameters=CANDIDATE_PARAMETERS; N=51)
    deriv = _scheme(Float64, parameters, "assembly check")
    dmodel = first(derivative_matrix(deriv, N))
    fmodel = filter_matrix(N)
    (derivative=maximum(abs, dmodel - production_matrix(deriv, N)),
     filter=maximum(abs, fmodel - production_matrix(CL.compact_filter(0.45), N)))
end

"""Spectral radius of one RK step followed by the unrelaxed one-sided filter."""
function filtered_radius(parameters, N; cfl=0.5, relaxed=false)
    scheme = _trial_scheme(parameters)
    L, _ = acoustic_operator(scheme, N)
    n = size(L, 1); Q = Matrix{Float64}(I, n, n); U = zeros(n, n)
    dt = cfl / (N - 1)
    for stage in eachindex(CL.RKA)
        U .= CL.RKA[stage] .* U .+ dt .* (L * Q)
        Q .+= CL.RKB[stage] .* U
    end
    F = filter_block(N, 0.45)
    scalar = base_filter(N, 0.45)
    if relaxed
        weight = min(cfl / 0.35, 1.0)
        F = (1 - weight) .* Matrix{Float64}(I, size(F,1), size(F,2)) .+ weight .* F
        scalar = (1 - weight) .* Matrix{Float64}(I, N, N) .+ weight .* scalar
    end
    max(maximum(abs, eigvals(F * Q)), maximum(abs, eigvals(scalar)))
end

"""Worst normalized hard-gate violation; zero is feasible."""
function feasibility_violation(parameters; Ns=(17,24,31,51),
                               filtered_Ns=(17,24,31,51),
                               growth_tol=1e-8, radius_tol=1e-10)
    scheme = _trial_scheme(parameters)
    ug = maximum(maximum(real, eigvals(first(acoustic_operator(scheme, N)))) for N in Ns)
    fr = maximum(filtered_radius(parameters, N) - 1 for N in filtered_Ns)
    max(0.0, (ug - growth_tol) / growth_tol, (fr - radius_tol) / radius_tol)
end

"""Deterministic differential evolution for feasibility, without cond(A)."""
function differential_evolution(; start=EXTENDED_REJECTED_PARAMETERS,
        Ns=(17,24,31,51), filtered_Ns=(17,24,31,51), population=36,
        generations=20, seed=0x5c105e, lhs_radius=1.0, null_radius=0.01,
        mutation=0.7, crossover=0.9, verbose=true)
    rng = MersenneTwister(seed); n = length(start)
    radii = n == 11 ? [fill(lhs_radius, 7); fill(null_radius, 4)] : fill(lhs_radius, n)
    pop = [copy(start) for _ in 1:population]
    for i in 2:population
        pop[i] .+= radii .* (2 .* rand(rng, n) .- 1)
    end
    score(p) = feasibility_violation(p; Ns, filtered_Ns)
    scores = score.(pop)
    for generation in 1:generations
        for i in 1:population
            pool = [j for j in 1:population if j != i]
            shuffle!(rng, pool); a, b, c = pool[1:3]
            mutant = pop[a] .+ mutation .* (pop[b] .- pop[c])
            trial = copy(pop[i]); forced = rand(rng, 1:n)
            for j in 1:n
                (rand(rng) < crossover || j == forced) && (trial[j] = mutant[j])
                trial[j] = clamp(trial[j], start[j] - 2radii[j], start[j] + 2radii[j])
            end
            s = score(trial)
            if s < scores[i]
                pop[i], scores[i] = trial, s
            end
        end
        k = argmin(scores)
        verbose && @printf("generation %d best violation %.6e\n", generation, scores[k])
        iszero(scores[k]) && return copy(pop[k]), scores[k]
    end
    k = argmin(scores)
    copy(pop[k]), scores[k]
end

function diagnostics(parameters; Ns=(17, 31, 51, 79), transient=true)
    scheme = _trial_scheme(parameters)
    rows = NamedTuple[]
    for N in Ns
        L, A = acoustic_operator(scheme, N)
        E = eigen(L)
        growth = maximum(real, E.values)
        evcond = cond(E.vectors)
        # Resolvent samples reveal non-normal amplification more cheaply and
        # reproducibly than a full pseudospectral grid.
        omega = range(-2N, 2N; length=41)
        resolvent = transient ? maximum(opnorm(inv((im*w)*I - L), 2) for w in omega
                                      if minimum(abs.((im*w) .- E.values)) > 1e-10) : NaN
        push!(rows, (; N, growth, lhscond=cond(A), evcond, resolvent))
    end
    rows
end

function objective(parameters; mode=:unfiltered, Ns=(17, 31, 51), filtered_Ns=(51, 101))
    try
        scheme = _trial_scheme(parameters)
        growth = 0.0; lhs = 0.0
        for N in Ns
            L, A = acoustic_operator(scheme, N)
            growth = max(growth, maximum(real, eigvals(L)))
            lhs = max(lhs, cond(A))
        end
        filter_excess = mode === :filtered ?
            maximum(max(0.0, filtered_radius(parameters, N) - 1) for N in filtered_Ns) : 0.0
        return 1e5max(0.0, growth) + 1e5filter_excess + log10(lhs)
    catch
        return Inf
    end
end

"""Cheap held-out sweep: `(N, max_real_part, cond_lhs)` only."""
function growth_sweep(parameters, Ns)
    scheme = _trial_scheme(parameters)
    map(Ns) do N
        L, A = acoustic_operator(scheme, N)
        (N=N, growth=maximum(real, eigvals(L)), lhscond=cond(A))
    end
end

"""Bounded deterministic Gaussian local search, seeded by the BL control."""
function search(; mode=:unfiltered, passes=8, trials=160, Ns=(17, 31, 51, 79, 101),
                filtered_Ns=(51, 101), seed=0x5c105e, support=6, null_scale=0.01,
                start=nothing, verbose=true)
    rng = MersenneTwister(seed)
    support in (6, 7) || throw(ArgumentError("support must be 6 or 7"))
    best = copy(isnothing(start) ?
        (support == 6 ? BL_PARAMETERS : EXTENDED_BL_PARAMETERS) : start)
    direction_scale = support == 6 ? ones(7) : [ones(7); fill(null_scale, 4)]
    score = objective(best; mode, Ns, filtered_Ns)
    scale = 0.35
    for pass in 1:passes
        for _ in 1:trials
            trial = best .+ scale .* direction_scale .* randn(rng, length(best))
            any(abs.(trial) .> 10) && continue
            s = objective(trial; mode, Ns, filtered_Ns)
            if s < score
                best, score = trial, s
                verbose && @printf("pass %d score %.6g\n", pass, score)
            end
        end
        scale *= 0.55
    end
    return best, diagnostics(best; Ns)
end

function print_diagnostics(label, parameters; Ns=(17, 31, 51, 79, 101))
    println("\n", label)
    println(" N       max Re(lambda)     cond(A)       cond(V)       sampled resolvent")
    for d in diagnostics(parameters; Ns)
        @printf("%4d   %+14.6e   %10.3e   %10.3e   %10.3e\n",
                d.N, d.growth, d.lhscond, d.evcond, d.resolvent)
    end
end

parse_Ns(s) = Tuple(parse.(Int, split(s, ',')))

function report_candidate(label, parameters; sweep=false)
    print_diagnostics(label, parameters)
    for N in (51, 101)
        @printf("  filtered N=%d radius %.10f\n", N, filtered_radius(parameters, N))
    end
    sweep || return
    held = growth_sweep(parameters, [12:200; 257; 371; 415; 459; 601])
    worst = held[argmax(getproperty.(held, :growth))]
    @printf("  held-out max growth %+.6e at N=%d; max cond(A) %.6g\n",
            worst.growth, worst.N, maximum(getproperty.(held, :lhscond)))
end

function print_gate_values(parameters, Ns, filtered_ns)
    scheme = _trial_scheme(parameters)
    for N in Ns
        L, _ = acoustic_operator(scheme, N)
        @printf("  gate unfiltered N=%d growth %+.6e\n", N, maximum(real, eigvals(L)))
    end
    for N in filtered_ns
        @printf("  gate filtered N=%d excess %+.6e\n", N,
                filtered_radius(parameters, N) - 1)
    end
end

function main(args=ARGS)
    opts = CL.script_args(args, (mode="report", objective="unfiltered", passes=6,
        trials=160, seed=6033502, train_ns="17,31,51,79,101",
        filtered_ns="51,101", support=6, null_scale=0.01, start="bl", sweep=false,
        population=36, generations=20, lhs_radius=1.0, null_radius=0.01))
    mismatch = assembly_mismatch()
    @printf("assembly mismatch: derivative %.3e filter %.3e\n",
            mismatch.derivative, mismatch.filter)
    if opts.mode == "report"
        report_candidate("Brady--Livescu control", BL_PARAMETERS; sweep=false)
        report_candidate("unfiltered-only rejected candidate", UNFILTERED_ONLY_PARAMETERS;
                         sweep=opts.sweep)
        report_candidate("filtered-objective rejected candidate", CANDIDATE_PARAMETERS;
                         sweep=opts.sweep)
    elseif opts.mode == "search"
        objective_mode = Symbol(opts.objective)
        objective_mode in (:unfiltered, :filtered) ||
            error("objective must be unfiltered or filtered")
        train_ns = parse_Ns(opts.train_ns); filtered_ns = parse_Ns(opts.filtered_ns)
        start = opts.start == "bl" ? nothing :
                opts.start == "extended_rejected" ? EXTENDED_REJECTED_PARAMETERS :
                error("start must be bl or extended_rejected")
        best, _ = search(; mode=objective_mode, passes=opts.passes, trials=opts.trials,
                         Ns=train_ns, filtered_Ns=filtered_ns, seed=opts.seed,
                         support=opts.support, null_scale=opts.null_scale, start)
        report_candidate("computed search result", best; sweep=opts.sweep)
        println("parameters = ", repr(best))
    elseif opts.mode == "de"
        train_ns = parse_Ns(opts.train_ns); filtered_ns = parse_Ns(opts.filtered_ns)
        de_start = opts.start == "bl" ? EXTENDED_BL_PARAMETERS :
                   opts.start == "combined" ? EXTENDED_COMBINED_PARAMETERS :
                   opts.start == "de_best" ? EXTENDED_DE_PARAMETERS :
                   opts.start == "feasible_small" ? EXTENDED_FEASIBLE_SMALL_PARAMETERS :
                   opts.start == "extended_rejected" ? EXTENDED_REJECTED_PARAMETERS :
                   error("DE start must be bl, combined, de_best, feasible_small or extended_rejected")
        best, violation = differential_evolution(; start=de_start, Ns=train_ns,
            filtered_Ns=filtered_ns, population=opts.population,
            generations=opts.generations, seed=opts.seed,
            lhs_radius=opts.lhs_radius, null_radius=opts.null_radius)
        @printf("final feasibility violation %.6e\n", violation)
        println("parameters = ", repr(best))
        print_gate_values(best, train_ns, filtered_ns)
        report_candidate("differential-evolution result", best; sweep=opts.sweep)
    elseif opts.mode == "validate"
        p = EXTENDED_FEASIBLE_SMALL_PARAMETERS
        held = growth_sweep(p, [12:200; 257; 371; 415; 459; 601])
        worst = held[argmax(getproperty.(held, :growth))]
        @printf("broad unfiltered max growth %+.6e at N=%d\n", worst.growth, worst.N)
        print_gate_values(p, (), (12,17,24,31,51,79,101,171,257,415))
        println("parameters = ", repr(p))
    elseif opts.mode == "filtervalidate"
        p = EXTENDED_DE_FINAL_PARAMETERS
        Ns = [17:200; 257; 371; 415; 459; 601]
        for cfl in (0.5, 0.25, 0.125)
            radii = [filtered_radius(p, N; cfl, relaxed=true) for N in Ns]
            k = argmax(radii)
            firstfail = findfirst(>(1 + 1e-10), radii)
            @printf("relaxed cfl %.3f max radius %.12f at N=%d first fail %s\n",
                    cfl, radii[k], Ns[k], isnothing(firstfail) ? "none" : string(Ns[firstfail]))
            failures = Ns[findall(>(1 + 1e-10), radii)]
            println("  failing N: ", isempty(failures) ? "none" : join(failures, ','))
        end
        println("parameters = ", repr(p))
    else
        error("mode must be report, search, de, validate or filtervalidate")
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    ClosureSearch.main()
end
