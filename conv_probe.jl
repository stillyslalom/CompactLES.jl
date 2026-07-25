using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

cases = [
 (; n_global=(24, 12, 12), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), kw=(;)),
 (; n_global=(48, 24, 24), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), kw=(;)),
 (; n_global=(96, 48, 48), L_domain=(1.0, π, 2π), metric=SphericalMetric(),
    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), kw=(;)),
]
for (idx, cs) in enumerate(cases)
    kw = merge((; n_global=cs.n_global, L_domain=cs.L_domain, metric=cs.metric, bcs=cs.bcs,
                art=ArtParams(enabled=false)), cs.kw)
    s = Solver(; kw...)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0,0,0), p=1.0, rho=1.0))
    apply_bcs!(s, Q)
    dQ = zero(Q)
    compute_rhs!(s, Q, dQ)
    mc = [maximum(abs(dQ[gidx(s,i,j,k),c]) for i in 1:s.decomp.n_local[1],
          j in 1:s.decomp.n_local[2], k in 1:s.decomp.n_local[3]) for c in 1:s.n_cons]
    println("case $idx: max=", maximum(mc))
end
