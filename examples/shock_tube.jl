# Two-gas shock tube along x on the Problem/Numerics frontend: light driver
# gas (γ = 1.4) at high pressure, heavy SF6-like gas (γ = 1.09) at low
# pressure, artificial fluid properties handling shock and interface. Slip
# walls in x, periodic transverse. Uncomment the NSCBC line to make the
# downstream end non-reflecting instead of reflective.
#
# Run:  mpiexec -n 4 julia --project=. -t 2 examples/shock_tube.jl

using MPI
MPI.Init(threadlevel=:funneled)

using CompactLES
using Printf

Nx, Ny, Nz = 512, 16, 16
Lx = 1.0
hx = Lx / (Nx - 1)
δ  = 4hx                          # interface width
np = MPI.Comm_size(MPI.COMM_WORLD)

eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                    IdealSpecies{Float64}("heavy", 0.2, 1.09)])

xbc = (SlipWallBC(), SlipWallBC())
# xbc = (SlipWallBC(), NSCBCOutflowBC(pinf=0.1))   # non-reflecting downstream

prob = Problem(
    name = "two-gas shock tube",
    eos = eos,
    transport = Transport(mu0=0.0),
    domain = ((0.0, Lx), (0.0, Ny * hx), (0.0, Nz * hx)),
    bcs = (xbc, (PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())),
    ic = (x, y, z) -> begin
        θ = tanh_blend(x, 0.5, δ)       # 0 left (light), 1 right (heavy)
        Prim(Y = (1 - θ, θ),
             rho = (1 - θ) * 1.0 + θ * 0.625,
             p   = (1 - θ) * 1.0 + θ * 0.1)
    end)

num = Numerics(nglob=(Nx, Ny, Nz), art=ArtParams(enabled=true),
               cfl=0.5, filter_interval=1, dims=(np, 1, 1))
# Cluster x-resolution near the initial interface (2.3:1 spacing ratio):
# num = Numerics(nglob=(Nx, Ny, Nz), art=ArtParams(enabled=true), cfl=0.5,
#                filter_interval=1, dims=(np, 1, 1),
#                stretch=(sine_cluster(0.0, Lx, 0.5, 0.4), nothing, nothing))

s, Q = setup(prob, num)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

function diag(s, Q)
    s.step % 25 == 0 || return
    nx, ny, nz = s.dec.nloc
    m1 = s.mom[1]
    ρmin, ρmax, umax = Inf, -Inf, 0.0
    for k in 1:nz, j in 1:ny, i in 1:nx
        ρ = Q[gidx(s, i, j, k), 1] + Q[gidx(s, i, j, k), 2]
        ρmin = min(ρmin, ρ); ρmax = max(ρmax, ρ)
        umax = max(umax, abs(Q[gidx(s, i, j, k), m1] / ρ))
    end
    ρmin = MPI.Allreduce(ρmin, min, s.dec.comm)
    ρmax = MPI.Allreduce(ρmax, max, s.dec.comm)
    umax = MPI.Allreduce(umax, max, s.dec.comm)
    rank == 0 && @printf("step %5d  t = %7.4f  ρ ∈ [%.4f, %.4f]  |u|max = %.4f\n",
                         s.step, s.t, ρmin, ρmax, umax)
end
##
run!(s, Q; tfinal=0.25, nmax=100_000, callback=diag)
# save_checkpoint(s, Q, "shock_tube_final")

# open(@sprintf("prof_rank%03d.dat", rank), "w") do io
#     nx = s.dec.nloc[1]
#     j = 1; k = 1
#     for i in 1:nx
#         ρ = Q[gidx(s, i, j, k), 1] + Q[gidx(s, i, j, k), 2]
#         @printf(io, "%.8e  %.8e  %.8e\n", xcoord(s, 1, i), ρ,
#                 Q[gidx(s, i, j, k), 2] / ρ)
#     end
# end
# rank == 0 && println("done; concatenate prof_rank*.dat (sorted) for profiles.")
