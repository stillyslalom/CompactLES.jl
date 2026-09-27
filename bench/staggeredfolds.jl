# Symmetry and conservation of the staggered diffusion operator of
# src/staggered.jl at a fold whose area vanishes oddly: the cylindrical axis
# (A = r) and the spherical poles (A = r sinθ). There the flux C K D_s T is
# continued smoothly through the singular set, as the explicit divergence
# does, which is not the mirror adjoint to D_s. Along one grid line of N
# nodes it forms the dense matrix of L = J⁻¹ G C K D_s from unit vectors and
# prints, with V = J W the node volumes (W = h, h/2 on a wall node):
#   asym   ‖V L − (V L)ᵀ‖ / ‖V L‖
#   rows   the rows of V L − (V L)ᵀ above 1e-3 of the largest row norm (0
#          where the asymmetry is round-off)
#   cons   |Σ V L T| / Σ V |L T| on a smooth T with zero flux at both ends
#   Re λ   the largest real part of an eigenvalue of L
#   sym λ  the largest eigenvalue of the symmetric part of V L over ‖V L‖
# A symmetry plane on a Cartesian line, where the operator is the mirror
# adjoint, is printed alongside as the control.
#
# Usage (seconds; single-threaded):
#   julia --project=. -t 1 bench/staggeredfolds.jl [n=16,32,64,128]

using CompactLES, LinearAlgebra, Printf

const CL = CompactLES
const OPT = CL.script_args(ARGS, (n = "16,32,64,128",))

const PER = (PeriodicBC(), PeriodicBC())
const NOART = ArtificialProperties(enabled=false)

geometries(N) = (
    ("symmetry plane", 1, x -> cospi(x),
     Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0),
            bcs=((SymmetryPlaneBC(), SlipWallBC()), PER, PER), art=NOART)),
    ("cylindrical axis", 1, r -> cospi(r),
     Solver(n_global=(N, 1, 1), L_domain=(1.0, 1.0, 1.0), metric=CylindricalMetric(),
            bcs=((AxisBC(), SlipWallBC()), PER, PER), art=NOART)),
    ("spherical poles", 2, θ -> cos(2θ),
     Solver(n_global=(12, N, 1), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
            origin=(0.5, 0.0, 0.0),
            bcs=((SlipWallBC(), SlipWallBC()), (PoleBC(), PoleBC()), PER), art=NOART)),
)

function measure(label, d, Tfn, solver)
    decomp = solver.decomp
    pad = decomp.n_halo_d
    N = decomp.n_local[d]
    base = ntuple(k -> pad[k] + max(1, decomp.n_local[k] ÷ 2), 3)
    at(i) = CartesianIndex(ntuple(k -> k == d ? pad[d] + i : base[k], 3))
    op = CL.StaggeredDiffusion(solver, d)
    κ = CL.field(decomp); κ .= 1
    fold = solver.folds[d]
    wall_hi = !fold.hi
    x = [CL.xcoord(solver, d, i) for i in 1:N]
    V = [solver.h[d] * (i == N && wall_hi ? 0.5 : 1.0) / solver.inv_J[at(i)] for i in 1:N]
    M = zeros(N, N)
    for j in 1:N
        u = CL.field(decomp); u[at(j)] = 1
        out = CL.field(decomp)
        CL.staggered_diffusion!(out, op, u, copy(κ), decomp)
        M[:, j] = [out[at(i)] for i in 1:N]
    end
    VL = Diagonal(V) * M
    A = VL - VL'
    rows = [norm(A[i, :]) for i in 1:N]
    LT = M * Tfn.(x)
    @printf("%-17s %4d  %9.2e  %4d  %9.2e  %9.2e  %9.2e\n", label, N,
            norm(A) / norm(VL),
            norm(A) > 1e-12 * norm(VL) ? count(rows .> 1e-3 * maximum(rows)) : 0,
            abs(sum(V .* LT)) / sum(V .* abs.(LT)), maximum(real, eigvals(M)),
            maximum(eigvals(Symmetric((VL + VL') / 2))) / norm(VL))
end

function main()
    println("geometry             N       asym  rows       cons       Re λ      sym λ")
    for N in parse.(Int, split(OPT.n, ","))
        for g in geometries(N)
            measure(g...)
        end
    end
end

main()
