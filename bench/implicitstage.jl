# The implicit diffusion stage of src/implicit.jl, (I − γΔt L) T = r with the
# staggered conduction operator, on the geometries the solver builds.
#
# part=iterations prints the Krylov iteration count of one stage solve to
# rtol = 1e-10 (the L2 norm of the residual over that of r) on each grid and
# each γΔt, with a variable coefficient, the method (CG where V L is
# symmetric, GMRES at the axis and poles), and the stiffness
# R = γΔt max κ / h_min², the ratio of the step to the forward-Euler limit up
# to a constant.
#
# part=manufactured solves the stage for a manufactured solution T* with
# r = T* − γΔt ∇·(κ∇T*) evaluated analytically (the inner derivative by a
# complex step, the outer by a sixth-order central difference) and prints the
# maximum and L2 errors of T against T* and their observed orders, with a
# constant and a variable κ.
#
# part=preconditioner separates the multigrid cycle from the operator pairing:
# the iterations of unrestarted GMRES on the weighted stage system to the
# same tolerance, right-preconditioned by the package's cycle and by the
# exact inverse of the second-order operator S (a sparse Cholesky
# factorization), at γΔt = 10 on two grids. A count that grows
# under the cycle but not under the exact inverse is the cycle's.
#
# Usage (about three minutes at the defaults):
#   julia --project=. -t 16 bench/implicitstage.jl [part=all] [scale=1] [only=]
# `scale` multiplies every grid (scale=2 doubles them), `gamma` lists the
# steps of the iteration table, and `only` keeps the geometries whose name
# contains it.

using CompactLES, LinearAlgebra, SparseArrays, Printf

const CL = CompactLES
const OPT = CL.script_args(ARGS, (part = "all", scale = 1, gamma = "1e-3,1e-1,1e1",
                                  only = ""))

const PER = (PeriodicBC(), PeriodicBC())
const WALLS = (SlipWallBC(), SlipWallBC())
const NOART = ArtificialProperties(enabled=false)

gauss(r) = exp(-16r^2)

# Geometries: a name, the solver on a grid of scale N, the parity of the walls,
# and a manufactured temperature and coefficient smooth through every fold,
# of the walls' parity, and (on the coordinate singularities) negligible at
# the curved outer wall, whose node is first order.
const GEOMETRIES = (
    (name="periodic 2-D", parity=1,
     build=N -> Solver(n_global=(N, N, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=(PER, PER, PER), art=NOART),
     T=(x, y, z) -> sin(2π * x) * cos(2π * y) + 0.3cos(4π * x),
     κ=(x, y, z) -> 1 + 0.5sin(2π * x) * cos(2π * y)),
    (name="walls 2-D", parity=1,
     build=N -> Solver(n_global=(N + 1, N + 1, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=(WALLS, WALLS, PER), art=NOART),
     T=(x, y, z) -> cospi(x) * cospi(y) + 0.3cospi(2x),
     κ=(x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
    (name="isothermal walls 2-D", parity=-1,
     build=N -> Solver(n_global=(N + 1, N + 1, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=(WALLS, WALLS, PER), art=NOART),
     T=(x, y, z) -> sinpi(x) * sinpi(y) + 0.3sinpi(2x) * sinpi(y),
     κ=(x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
    (name="symmetry planes 2-D", parity=1,
     build=N -> Solver(n_global=(N, N + 1, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=((SymmetryPlaneBC(), SymmetryPlaneBC()), WALLS, PER),
                       art=NOART),
     T=(x, y, z) -> cospi(x) * cospi(y) + 0.3cospi(2x),
     κ=(x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
    (name="stretched walls 2-D", parity=1,
     build=N -> Solver(n_global=(N + 1, N + 1, 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=(WALLS, WALLS, PER), art=NOART,
                       stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4), nothing, nothing)),
     T=(x, y, z) -> cospi(x) * cospi(y) + 0.3cospi(2x),
     κ=(x, y, z) -> 1 + 0.5cospi(x) * cospi(2y)),
    (name="cylindrical shell r-θ", parity=1,
     build=N -> Solver(n_global=(N + 1, N, 1), L_domain=(1.0, 2π, 1.0),
                       metric=CylindricalMetric(), origin=(0.5, 0.0, 0.0),
                       bcs=(WALLS, PER, PER), art=NOART),
     T=(r, θ, z) -> cospi(r - 0.5) * (1 + 0.3cos(θ)),
     κ=(r, θ, z) -> 1 + 0.3r * sin(θ)),
    (name="cylindrical axis r-z", parity=1,
     build=N -> Solver(n_global=(N, 1, N), L_domain=(1.5, 1.0, 1.0),
                       metric=CylindricalMetric(),
                       bcs=((AxisBC(), SlipWallBC()), PER, PER), art=NOART),
     T=(r, θ, z) -> gauss(r) * (1 + 0.3cos(2π * z)),
     κ=(r, θ, z) -> 1 + 0.5r^2 + 0.2sin(2π * z)),
    (name="resolved axis r-θ", parity=1,
     build=N -> Solver(n_global=(N, N, 1), L_domain=(1.5, 2π, 1.0),
                       metric=CylindricalMetric(),
                       bcs=((AxisBC(), SlipWallBC()), PER, PER), art=NOART),
     T=(r, θ, z) -> gauss(r) * (1 + r * cos(θ)),
     κ=(r, θ, z) -> 1 + 0.5r^2 + 0.2r * sin(θ)),
    (name="walls 3-D", parity=1,
     build=N -> Solver(n_global=(N + 1, N + 1, N + 1), L_domain=(1.0, 1.0, 1.0),
                       bcs=(WALLS, WALLS, WALLS), art=NOART),
     T=(x, y, z) -> cospi(x) * cospi(y) * cospi(z) + 0.3cospi(2x),
     κ=(x, y, z) -> 1 + 0.5cospi(x) * cospi(2y) * cospi(z)),
    (name="cylindrical axis r-θ-z", parity=1,
     build=N -> Solver(n_global=(N, N, N), L_domain=(1.5, 2π, 1.0),
                       metric=CylindricalMetric(),
                       bcs=((AxisBC(), SlipWallBC()), PER, PER), art=NOART),
     T=(r, θ, z) -> gauss(r) * (1 + r * cos(θ)) * (1 + 0.3cos(2π * z)),
     κ=(r, θ, z) -> 1 + 0.5r^2 + 0.2r * sin(θ)),
    (name="spherical origin, poles", parity=1,
     build=N -> Solver(n_global=(N, N, N), L_domain=(1.5, π, 2π),
                       metric=SphericalMetric(),
                       bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), PER),
                       art=NOART),
     T=(r, θ, φ) -> gauss(r) * (1 + 0.5r * cos(θ) + 0.3r * sin(θ) * cos(φ)),
     κ=(r, θ, φ) -> 1 + 0.5r^2 + 0.2r * sin(θ) * cos(φ)),
    (name="spherical shell, poles", parity=1,
     build=N -> Solver(n_global=(N + 1, N, N), L_domain=(1.0, π, 2π),
                       metric=SphericalMetric(), origin=(0.5, 0.0, 0.0),
                       bcs=(WALLS, (PoleBC(), PoleBC()), PER), art=NOART),
     T=(r, θ, φ) -> cospi(r - 0.5) * (1 + 0.3cos(θ) + 0.2sin(θ) * sin(φ)),
     κ=(r, θ, φ) -> 1 + 0.3cos(θ)^2),
)

const GRIDS = Dict("periodic 2-D" => (16, 32, 64), "walls 2-D" => (16, 32, 64),
                   "isothermal walls 2-D" => (16, 32, 64),
                   "symmetry planes 2-D" => (16, 32, 64),
                   "stretched walls 2-D" => (16, 32, 64),
                   "cylindrical shell r-θ" => (16, 32, 64),
                   "cylindrical axis r-z" => (16, 32, 64),
                   "resolved axis r-θ" => (16, 32, 64),
                   "walls 3-D" => (12, 24, 48),
                   "cylindrical axis r-θ-z" => (12, 24, 48),
                   "spherical origin, poles" => (12, 24, 48),
                   "spherical shell, poles" => (12, 24, 48))

function fill_fn!(solver, f, fn)
    d = solver.decomp
    for k in 1:d.n_local[3], j in 1:d.n_local[2], i in 1:d.n_local[1]
        x = (CL.xcoord(solver, 1, i), CL.xcoord(solver, 2, j), CL.xcoord(solver, 3, k))
        f[CL.padded_index(solver, i, j, k)] = fn(x...)
    end
    return f
end

# ∇·(κ∇T) along the active dimensions, in the metric's physical coordinates.
function exact_divergence(metric, active, Tfn, κfn, x::NTuple{3,Float64})
    J(y) = prod(CL.scalefactors(metric, y...))
    total = 0.0
    for d in 1:3
        active[d] || continue
        e = ntuple(k -> k == d ? 1.0 : 0.0, 3)
        function flux(s)
            y = x .+ s .* e
            hs = CL.scalefactors(metric, y...)
            dT = imag(Tfn((y .+ 1e-30im .* e)...)) / 1e-30
            return J(y) / hs[d]^2 * κfn(y...) * dT
        end
        δ = 1e-4
        c = (3 / 4, -3 / 20, 1 / 60)
        total += sum(c[m] * (flux(m * δ) - flux(-m * δ)) for m in 1:3) / δ
    end
    return total / J(x)
end

function stage_problem(g, N, γΔt, κfn)
    solver = g.build(N)
    stage = CL.DiffusionStage(solver; parity=g.parity)
    decomp = solver.decomp
    κ = fill_fn!(solver, CL.field(decomp), κfn)
    return solver, stage, κ
end

function iterations()
    gammas = parse.(Float64, split(OPT.gamma, ","))
    println("\n=== stage iterations to rtol 1e-10, variable κ ===")
    @printf("%-26s %6s  %5s", "geometry", "method", "N")
    foreach(γ -> @printf("  γΔt=%-7.0e (R)      ", γ), gammas)
    println()
    for g in GEOMETRIES
        occursin(OPT.only, g.name) || continue
        for N in GRIDS[g.name] .* OPT.scale
            solver, stage, κ = stage_problem(g, N, gammas[1], g.κ)
            decomp = solver.decomp
            rhs = fill_fn!(solver, CL.field(decomp), g.T)
            hmin = minimum(solver.h[d] for d in 1:3 if decomp.active[d])
            @printf("%-26s %6s  %5d", g.name, stage.symmetric ? "CG" : "GMRES", N)
            for γ in gammas
                T = copy(rhs)
                res = CL.solve_stage!(T, stage, rhs, κ, γ)
                R = γ * maximum(κ[CL.interior(decomp)]) / hmin^2
                @printf("  %4d%s (%8.1e)     ", res.iterations, res.converged ? " " : "*", R)
            end
            println()
        end
    end
end

function manufactured()
    γΔt = 0.01
    println("\n=== manufactured stage solution, γΔt = $γΔt: max | L2 error and order ===")
    for g in GEOMETRIES, (label, κfn) in (("κ = 1", (x, y, z) -> 1.0), ("variable κ", g.κ))
        occursin(OPT.only, g.name) || continue
        errs = Tuple{Float64,Float64,Float64}[]
        its = Int[]
        for N in GRIDS[g.name] .* OPT.scale
            solver, stage, κ = stage_problem(g, N, γΔt, κfn)
            decomp = solver.decomp
            exact = fill_fn!(solver, CL.field(decomp), g.T)
            active = decomp.active
            rhs = fill_fn!(solver, CL.field(decomp),
                           (x...) -> g.T(x...) -
                                     γΔt * exact_divergence(solver.metric, active, g.T,
                                                            κfn, Float64.(x)))
            if g.parity == -1
                pad = CartesianIndex(decomp.n_halo_d)
                for I in CartesianIndices(decomp.n_local)
                    stage.dirichlet[I] && (rhs[I+pad] = exact[I+pad] = 0)
                end
            end
            T = copy(rhs)
            res = CL.solve_stage!(T, stage, rhs, κ, γΔt)
            push!(its, res.iterations)
            inner = CL.interior(decomp)
            e = abs.(T[inner] .- exact[inner])
            V = stage.volume[inner]
            h = minimum(solver.h[d] for d in 1:3 if active[d])
            push!(errs, (h, maximum(e), sqrt(sum(V .* e .^ 2) / sum(V))))
        end
        @printf("%-26s %-11s", g.name, label)
        for (m, (h, emax, e2)) in enumerate(errs)
            @printf("  %.2e | %.2e", emax, e2)
            if m > 1
                hp = errs[m-1]
                @printf(" (%4.2f | %4.2f)", log(hp[2] / emax) / log(hp[1] / h),
                        log(hp[3] / e2) / log(hp[1] / h))
            end
        end
        @printf("   it %s\n", join(its, "/"))
    end
end

# S of the finest level as a sparse matrix over the interior nodes.
function sparse_operator(stage)
    lev = stage.levels[1]
    n = lev.n
    index = LinearIndices(n)
    rows = Int[]; cols = Int[]; vals = Float64[]
    add(i, j, v) = (push!(rows, i); push!(cols, j); push!(vals, v))
    pad = CartesianIndex(lev.pad)
    for I in CartesianIndices(n)
        g = index[I]
        add(g, g, lev.mass[I+pad])
        for k in lev.dims
            c = lev.cond[k][I+pad]
            c == 0 && continue
            J = I + CL._unit3(k)
            q = index[CartesianIndex(ntuple(m -> mod1(J[m], n[m]), 3))]
            add(g, g, c); add(q, q, c); add(g, q, -c); add(q, g, -c)
        end
    end
    return sparse(rows, cols, vals, prod(n), prod(n))
end

# Unrestarted right-preconditioned GMRES on V A T = V r in the inner product
# of the package's solve; returns the iteration count.
function gmres_count(stage, rhs, κ, precondition!; rtol=1e-10, maxiter=300)
    decomp = stage.decomp
    b = stage.volume .* rhs
    inner(u, v) = CL._global_dot(u, v, stage)
    x = copy(rhs); w = CL.field(decomp); z = CL.field(decomp)
    CL._apply_weighted!(w, stage, x, κ)
    r = b .- w
    β = sqrt(inner(r, r)); bnorm = sqrt(inner(b, b))
    V = [r ./ β]
    H = zeros(maxiter + 1, maxiter)
    for j in 1:maxiter
        precondition!(z, V[j])
        CL._apply_weighted!(w, stage, z, κ)
        for i in 1:j, pass in 1:2
            h = inner(V[i], w); H[i, j] += h; w .-= h .* V[i]
        end
        H[j+1, j] = sqrt(inner(w, w))
        push!(V, w ./ H[j+1, j])
        e1 = zeros(j + 1); e1[1] = β
        Hj = H[1:j+1, 1:j]
        norm(Hj * (Hj \ e1) - e1) <= rtol * bnorm && return j
    end
    return maxiter
end

function preconditioner()
    γΔt = 10.0
    println("\n=== GMRES iterations at γΔt = $γΔt: multigrid cycle | exact S⁻¹ ===")
    for g in GEOMETRIES
        occursin(OPT.only, g.name) || continue
        @printf("%-26s", g.name)
        # The three-dimensional grids stop at 32 cubed, where the factorization
        # still fits.
        for N in (GRIDS[g.name][1] == 12 ? (16, 32) : (32, 64)) .* OPT.scale
            solver, stage, κ = stage_problem(g, N, γΔt, g.κ)
            decomp = solver.decomp
            rhs = fill_fn!(solver, CL.field(decomp), g.T)
            CL._assemble_stage!(stage, κ, γΔt)
            factor = cholesky(sparse_operator(stage))
            inner = CL.interior(decomp)
            exact!(z, v) = (z[inner] .= reshape(factor \ vec(v[inner]), size(inner)); z)
            cycle!(z, v) = CL._precondition!(z, stage, v)
            @printf("   N=%-3d %4d | %4d", N, gmres_count(stage, rhs, κ, cycle!),
                    gmres_count(stage, rhs, κ, exact!))
        end
        println()
    end
end

function main()
    OPT.part in ("all", "iterations") && iterations()
    OPT.part in ("all", "manufactured") && manufactured()
    OPT.part in ("all", "preconditioner") && preconditioner()
end

main()
