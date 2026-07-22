using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

cases = [
 (; nglob=(24, 12, 12), Ldom=(1.0, π, 2π), metric=SphericalMetric(),
    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), kw=(;)),
 (; nglob=(48, 24, 24), Ldom=(1.0, π, 2π), metric=SphericalMetric(),
    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), kw=(;)),
 (; nglob=(96, 48, 48), Ldom=(1.0, π, 2π), metric=SphericalMetric(),
    bcs=((OriginBC(), SlipWallBC()), (PoleBC(), PoleBC()), per3[3]), kw=(;)),
]
for (idx, cs) in enumerate(cases)
    kw = merge((; nglob=cs.nglob, Ldom=cs.Ldom, metric=cs.metric, bcs=cs.bcs,
                art=ArtParams(enabled=false)), cs.kw)
    s = Solver(; kw...)
    Q = allocate_state(s)
    initialize!(s, Q, (x, y, z) -> Prim(u=(0,0,0), p=1.0, rho=1.0))
    apply_bcs!(s, Q)
    dQ = zero(Q)
    compute_rhs!(s, Q, dQ)
    mc = [maximum(abs(dQ[gidx(s,i,j,k),c]) for i in 1:s.dec.nloc[1],
          j in 1:s.dec.nloc[2], k in 1:s.dec.nloc[3]) for c in 1:s.ncons]
    println("case $idx: max=", maximum(mc))
end
