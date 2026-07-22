# Serial test suite (run with: julia --project=. test/runtests.jl
# or under one MPI rank). Ordered so failures localize: solvers → operators →
# closures/folds → metric → full RHS. Multi-rank checks live in mpi_tests.jl.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Test, LinearAlgebra, Random

const CL = CompactLES
Random.seed!(7)

per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)
mkslv(; kw...) = Solver(; bcs=per3, Ldom=(2π, 2π, 2π), art=ArtParams(enabled=false), kw...)

"Max interior error of a scalar field against an analytic function."
function ferr(s, f, fn)
e = 0.0
for k in 1:s.dec.nloc[3], j in 1:s.dec.nloc[2], i in 1:s.dec.nloc[1]
e = max(e, abs(f[gidx(s, i, j, k)] -
fn(xcoord(s, 1, i), xcoord(s, 2, j), xcoord(s, 3, k))))
end
e
end

fillf!(s, f, fn) = (for k in 1:s.dec.nloc[3], j in 1:s.dec.nloc[2], i in 1:s.dec.nloc[1]
f[gidx(s, i, j, k)] = fn(xcoord(s, 1, i), xcoord(s, 2, j), xcoord(s, 3, k))
end; f)

@testset "banded LU vs dense" begin
for q in (1, 2), n in (9, 17)
A = zeros(n, n)
for i in 1:n, jj in max(1, i-q):min(n, i+q)
A[i, jj] = (i == jj ? 3.0 : 0.0) + randn()
end
Ab = zeros(2q + 1, n)
for i in 1:n, ss in -q:q
1 <= i + ss <= n && (Ab[q+1+ss, i] = A[i, i+ss])
end
F = CL.BandFactor(Ab, q)
x = randn(n); b = A * x
y = copy(b); CL.solve_col!(y, F)
@test y ≈ x atol = 1e-9 rtol = 1e-9
end
end

@testset "periodic C6 derivative: spectral accuracy" begin
s = mkslv(nglob=(32, 32, 32))
f = CL.field(s.dec); df = CL.field(s.dec)
fillf!(s, f, (x, y, z) -> sin(3x) * cos(2y))
CL.exchange_halos!(f, s.dec)
CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
@test ferr(s, df, (x, y, z) -> 3cos(3x) * cos(2y)) < 1e-4  # C6 at k=3 on 32³
CL.deriv_along!(df, f, s, 2, 1); CL._scale_grad!(df, s, 2)
@test ferr(s, df, (x, y, z) -> -2sin(3x) * sin(2y)) < 2e-5
CL.deriv_along!(df, f, s, 3, 1); CL._scale_grad!(df, s, 3)
@test ferr(s, df, (x, y, z) -> 0.0) < 1e-10
end

@testset "transposed y/z path ≡ x path on permuted data" begin
s = mkslv(nglob=(24, 24, 24))
f = CL.field(s.dec); g = CL.field(s.dec)
d1 = CL.field(s.dec); d2 = CL.field(s.dec)
fn = (x, y, z) -> sin(2x + 0.3) * cos(y) + 0.1z^0 # z-independent
fillf!(s, f, fn) # varies in x
fillf!(s, g, (x, y, z) -> fn(y, x, z)) # same profile along y
CL.exchange_halos!(f, s.dec); CL.exchange_halos!(g, s.dec)
CL.deriv_along!(d1, f, s, 1, 1)
CL.deriv_along!(d2, g, s, 2, 1)
e = maximum(abs(d1[gidx(s, i, j, k)] - d2[gidx(s, j, i, k)])
for i in 1:24, j in 1:24, k in 1:24)
@test e < 1e-11
end

@testset "closed-domain closures: polynomial exactness (deg ≤ 3)" begin
s = Solver(nglob=(32, 12, 12), Ldom=(1.0, 1.0, 1.0),
bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
art=ArtParams(enabled=false))
f = CL.field(s.dec); df = CL.field(s.dec)
fillf!(s, f, (x, y, z) -> 1 + 2x + 3x^2 - x^3)
CL.exchange_halos!(f, s.dec)
CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
@test ferr(s, df, (x, y, z) -> 2 + 6x - 3x^2) < 1e-10
end

@testset "filter: constants exact, Nyquist damped, parity of closures" begin
s = mkslv(nglob=(32, 12, 12))
f = CL.field(s.dec)
fillf!(s, f, (x, y, z) -> 1.0)
filter_field!(f, s)
@test ferr(s, f, (x, y, z) -> 1.0) < 1e-12
nx = s.dec.nloc[1]
for k in 1:s.dec.nloc[3], j in 1:s.dec.nloc[2], i in 1:nx
f[gidx(s, i, j, k)] = 1.0 + 0.5 * (-1)^i # constant + Nyquist
end
filter_field!(f, s)
dev = maximum(abs(f[gidx(s, i, 1, 1)] - 1.0) for i in 1:nx)
@test dev < 0.35 # sawtooth strongly damped
end

@testset "axisymmetric axis fold: manufactured smooth solution" begin
# u_r = r·g(r) is an odd smooth function; d/dr through the fold must
# match analytics at the first half-offset nodes — the sharpest probe of
# the folded row and mirror fill.
s = Solver(nglob=(64, 1, 12), Ldom=(1.0, 1.0, 0.5),
metric=CylindricalMetric(),
bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
art=ArtParams(enabled=false))
f = CL.field(s.dec); df = CL.field(s.dec)
fillf!(s, f, (r, θ, z) -> r * exp(-4r^2)) # odd across the axis
CL.exchange_halos!(f, s.dec)
CL.deriv_along!(df, f, s, 1, -1); CL._scale_grad!(df, s, 1)
@test ferr(s, df, (r, θ, z) -> (1 - 8r^2) * exp(-4r^2)) < 5e-6
fillf!(s, f, (r, θ, z) -> exp(-4r^2)) # even across the axis
CL.exchange_halos!(f, s.dec)
CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
@test ferr(s, df, (r, θ, z) -> -8r * exp(-4r^2)) < 2e-5  # even fold: 3rd-order, larger const
end

@testset "resolved-θ axis: antipodal pairing (local)" begin
# f = r cosθ · e^{−4r²} = x·g is globally smooth through the axis and is
# ODD under (r,θ)→(−r,θ) with the pairing (θ+π picks up the cos sign).
s = Solver(nglob=(48, 16, 1), Ldom=(1.0, 2π, 1.0),
metric=CylindricalMetric(),
bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
art=ArtParams(enabled=false))
f = CL.field(s.dec); df = CL.field(s.dec)
fillf!(s, f, (r, θ, z) -> r * cos(θ) * exp(-4r^2))
CL.exchange_halos!(f, s.dec)
CL.deriv_along!(df, f, s, 1, -1); CL._scale_grad!(df, s, 1)
@test ferr(s, df, (r, θ, z) -> cos(θ) * (1 - 8r^2) * exp(-4r^2)) < 1e-5
# A scalar even case: f = e^{−4r²}·(1 + ½cos 2θ) maps to itself at θ+π.
fillf!(s, f, (r, θ, z) -> exp(-4r^2) * (1 + 0.5cos(2θ)))
CL.exchange_halos!(f, s.dec)
CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
@test ferr(s, df, (r, θ, z) -> -8r * exp(-4r^2) * (1 + 0.5cos(2θ))) < 1e-4  # 3rd-order, larger const
end

@testset "spherical poles + origin: derivative of a smooth 3-D Gaussian" begin
# f = e^{−4r²} is smooth at origin and poles; ∂f/∂r and (1/r)∂f/∂θ = 0.
s = Solver(nglob=(40, 16, 12), Ldom=(1.0, π, 2π),
metric=SphericalMetric(),
bcs=((OriginBC(), SlipWallBC()),
(PoleBC(), PoleBC()), per3[3]),
art=ArtParams(enabled=false))
f = CL.field(s.dec); df = CL.field(s.dec)
fillf!(s, f, (r, θ, φ) -> exp(-4r^2))
CL.exchange_halos!(f, s.dec)
CL.deriv_along!(df, f, s, 1, 1); CL._scale_grad!(df, s, 1)
@test ferr(s, df, (r, θ, φ) -> -8r * exp(-4r^2)) < 1e-4  # 3rd-order, larger const
CL.deriv_along!(df, f, s, 2, 1); CL._scale_grad!(df, s, 2)
@test ferr(s, df, (r, θ, φ) -> 0.0) < 1e-8
end

@testset "rigid rotation in cylindrical: zero strain" begin
# u_θ = Ω r ⇒ S_ij = 0 identically; probes the curvature corrections.
s = Solver(nglob=(32, 16, 12), Ldom=(1.0, 2π, 0.5),
metric=CylindricalMetric(), origin=(0.2, 0.0, 0.0),
bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
art=ArtParams(enabled=false))
Q = allocate_state(s)
initialize!(s, Q, (r, θ, z) -> Prim(u=(0.0, 0.3r, 0.0), p=1.0, rho=1.0))
dQ = zero(Q)
compute_rhs!(s, Q, dQ)
smax = 0.0
for k in 1:s.dec.nloc[3], j in 1:s.dec.nloc[2], i in 1:s.dec.nloc[1]
I = gidx(s, i, j, k)
for b in 1:3, a in 1:3
smax = max(smax, abs(0.5 * (s.G[a, b][I] + s.G[b, a][I])))
end
end
@test smax < 1e-8
end

@testset "freestream preservation (uniform state ⇒ dQ ≈ 0)" begin
# Uniform ρ, p, u = 0 must give zero RHS in every metric, with stretch,
# and with folds: divergence/source/geometry consistency in one number.
cases = [
(; nglob=(24, 12, 12), Ldom=(1.0, 1.0, 1.0), metric=CartesianMetric(),
bcs=per3, kw=(;)),
(; nglob=(24, 12, 12), Ldom=(1.0, 1.0, 1.0), metric=CartesianMetric(),
bcs=per3, kw=(; stretch=(sine_cluster(0.0, 1.0, 0.5, 0.4),
nothing, nothing),
bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]))),
(; nglob=(24, 12, 12), Ldom=(1.0, 2π, 0.5), metric=CylindricalMetric(),
bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
kw=(; origin=(0.2, 0.0, 0.0))),
(; nglob=(32, 1, 12), Ldom=(1.0, 1.0, 0.5), metric=CylindricalMetric(),
bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), kw=(;)),
(; nglob=(24, 16, 1), Ldom=(1.0, 2π, 1.0), metric=CylindricalMetric(),
bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]), kw=(;)),
(; nglob=(24, 12, 12), Ldom=(1.0, π, 2π), metric=SphericalMetric(),
bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]),
kw=(;)),
]
for cs in cases
kw = merge((; nglob=cs.nglob, Ldom=cs.Ldom, metric=cs.metric,
bcs=cs.bcs, art=ArtParams(enabled=false)), cs.kw)
s = Solver(; kw...)
Q = allocate_state(s)
initialize!(s, Q, (x, y, z) -> Prim(u=(0, 0, 0), p=1.0, rho=1.0))
apply_bcs!(s, Q)
dQ = zero(Q)
compute_rhs!(s, Q, dQ)
m = maximum(abs(dQ[gidx(s, i, j, k), c])
for c in 1:s.ncons, i in 1:s.dec.nloc[1],
j in 1:s.dec.nloc[2], k in 1:s.dec.nloc[3])
@test m < 1e-8
end
end

@testset "EOS: conserved ↔ primitive round trip (two species)" begin
eos = IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
IdealSpecies{Float64}("b", 0.2, 1.09)])
s = mkslv(nglob=(12, 12, 12), eos=eos)
Q = allocate_state(s)
pr = Prim(u=(0.3, -0.1, 0.2), p=0.8, T=1.7, Y=(0.35, 0.65))
initialize!(s, Q, (x, y, z) -> pr)
CL.exchange_state!(Q, s.dec)
CL.primitives!(s, Q)
I = gidx(s, 3, 4, 5)
@test s.p[I] ≈ 0.8 atol = 1e-12
@test s.Tt[I] ≈ 1.7 atol = 1e-12
@test s.u[I] ≈ 0.3 atol = 1e-12
@test s.Ys[2][I] ≈ 0.65 atol = 1e-12
end

@testset "conservation: periodic RHS integrates to zero" begin
s = mkslv(nglob=(16, 16, 16), transport=Transport(mu0=1e-3))
Q = allocate_state(s)
initialize!(s, Q, (x, y, z) ->
Prim(u=(0.1sin(x)cos(y), -0.1cos(x)sin(y), 0.05sin(z)),
p=1 + 0.05cos(x)cos(z), rho=1 + 0.1sin(y)))
dQ = zero(Q)
compute_rhs!(s, Q, dQ)
for c in 1:s.ncons
tot = sum(dQ[gidx(s, i, j, k), c] for i in 1:16, j in 1:16, k in 1:16)
@test abs(tot) < 1e-8 * 16^3
end
end

@testset "NSCBC outflow: matched uniform stream ⇒ no correction" begin
s = Solver(nglob=(32, 12, 12), Ldom=(1.0, 0.4, 0.4),
bcs=((DirichletBC((x, y, z, t) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0)),
NSCBCOutflowBC(pinf=1.0)), per3[2], per3[3]),
art=ArtParams(enabled=false))
Q = allocate_state(s)
initialize!(s, Q, (x, y, z) -> Prim(u=(0.3, 0, 0), p=1.0, rho=1.0))
apply_bcs!(s, Q)
dQ = zero(Q)
compute_rhs!(s, Q, dQ)
nx = s.dec.nloc[1]
m = maximum(abs(dQ[gidx(s, nx, j, k), c])
for c in 1:s.ncons, j in 1:12, k in 1:12)
@test m < 1e-8
end

@testset "checkpoint round trip" begin
s = mkslv(nglob=(12, 12, 12))
Q = allocate_state(s)
initialize!(s, Q, (x, y, z) -> Prim(u=(sin(x), 0, 0), p=1 + 0.1cos(y), rho=1.0))
s.t = 0.37; s.step = 42
save_checkpoint(s, Q, "test_ckpt")
Q2 = allocate_state(s); s.t = 0.0; s.step = 0
load_checkpoint!(s, Q2, "test_ckpt")
@test s.t == 0.37 && s.step == 42
@test all(Q2[gidx(s, i, j, k), c] == Q[gidx(s, i, j, k), c]
for c in 1:5, i in 1:12, j in 1:12, k in 1:12)
foreach(rm, filter(startswith("test_ckpt"), readdir()))
end

@testset "smoke: three RK steps of every headline configuration" begin
for build in (
() -> setup(Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
ic=(x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0)),
Numerics(nglob=(16, 16, 16))),
() -> setup(Problem(domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
ic=(x, y, z) -> Prim(u=(0.1sin(x), 0, 0), p=1.0, rho=1.0)),
Numerics(nglob=(16, 16, 16), deriv=lele_d1_10())),
() -> setup(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
metric=CylindricalMetric(),
bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
ic=(r, θ, z) -> Prim(u=(0, 0, 0), p=1 + exp(-40(r - 0.4)^2), rho=1.0)),
Numerics(nglob=(48, 1, 1))),
)
s, Q = build()
run!(s, Q; tfinal=1e9, nmax=3)
bad = any(!isfinite(Q[gidx(s, i, j, k), c])
for c in 1:s.ncons, i in 1:s.dec.nloc[1],
j in 1:s.dec.nloc[2], k in 1:s.dec.nloc[3])
@test !bad
end
end

println("serial tests complete")
