# Physically-realistic shock tube in SI units, exercising the multicomponent
# EOS and the artificial fluid properties on a real shock/contact interaction.
#
# A 4 m closed tube along x, square cross-section, with three regions:
#
#     |<--- He driver --->|<-- He driven -->|<---- CO2 ---->|
#     0                   2                 3               4   (metres)
#   left wall         diaphragm     He/CO2 interface     end wall
#
# The high-pressure helium driver (10 atm) bursts at the diaphragm (x = 2 m)
# and drives a Mach ~1.5 shock down the driven helium. At x = 3 m — one metre
# from the reflecting end wall — sits a sinusoidally-perturbed contact surface
# between the light driven helium and heavy CO2 (Atwood number ~0.83). The
# incident shock accelerates that interface (Richtmyer–Meshkov), then reflects
# off the end wall and returns to re-shock it, so the perturbation grows and
# inverts. Both x-ends are slip walls (a genuinely closed tube); the transverse
# directions are periodic and the interface holds one full sine wavelength.
#
# Everything is initialized at a uniform 300 K; the EOS supplies each region's
# density from its pressure and composition, so the initial state is
# thermodynamically consistent by construction. Uncomment the NSCBC line to let
# the downstream end radiate outward (a driven tube) instead of reflecting.
#
# Run:  mpiexec -n 4 julia --project=. -t 2 examples/shock_tube.jl

using MPI
MPI.Init(threadlevel=:funneled)

using CompactLES
using Printf

# --- Gas properties (SI): R = R_universal / M, γ from the molecular structure.
const Ru   = 8.314462618                 # universal gas constant [J/(mol·K)]
const R_He = Ru / 4.002602e-3            # helium      ≈ 2077 J/(kg·K)
const R_CO2 = Ru / 44.0095e-3            # carbon diox ≈ 189  J/(kg·K)
const γ_He, γ_CO2 = 5 / 3, 1.289

# --- Geometry (metres) and initial thermodynamic state.
const Lx = 4.0                            # tube length
const Lyz = 0.2                           # square cross-section side
const x_diaphragm = 2.0                   # driver / driven helium split
const x_iface     = 3.0                   # He / CO2 contact (1 m from end wall)
const atm = 101325.0                      # 1 atmosphere [Pa]
const p_driver, p_driven = 10atm, 1atm    # burst pressure ratio 10:1
const T0 = 300.0                          # uniform initial temperature [K]

const Nx, Ny, Nz = 768, 48, 1             # z collapsed → a 2-D run at 2-D cost
const hx = Lx / (Nx - 1)
const δ  = 3hx                            # diaphragm / interface diffuse width
const A_pert = 0.10Lyz                    # sinusoid amplitude (10% of the span)
np = MPI.Comm_size(MPI.COMM_WORLD)

eos = IdealMixture([IdealSpecies{Float64}("He",  R_He,  γ_He),
                    IdealSpecies{Float64}("CO2", R_CO2, γ_CO2)])

xbc = (SlipWallBC(), SlipWallBC())                    # closed reflecting tube
# xbc = (SlipWallBC(), NSCBCOutflowBC(pinf=p_driven)) # non-reflecting downstream

prob = Problem(
    name = "He-driven RM shock tube",
    eos = eos,
    transport = Transport(mu0=0.0),       # Euler + artificial regularization
    domain = ((0.0, Lx), (0.0, Lyz), (0.0, Lyz)),
    bcs = (xbc, (PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())),
    ic = (x, y, z) -> begin
        # Sinusoidally perturbed contact: interface sits at x_iface + A cos(2πy/L).
        xi = x_iface + A_pert * cos(2π * y / Lyz)
        θ  = tanh_blend(x, xi, δ)          # 0 in helium, 1 in CO2
        # Diaphragm pressure step: high in the driver (x < x_diaphragm), low beyond.
        p  = p_driven + (p_driver - p_driven) * (1 - tanh_blend(x, x_diaphragm, δ))
        Prim(Y = (1 - θ, θ), p = p, T = T0)   # EOS sets ρ from (p, T, Y)
    end)

num = Numerics(nglob=(Nx, Ny, Nz), art=ArtParams(enabled=true),
               cfl=0.5, filter_interval=1, dims=(np, 1, 1))

s, Q = setup(prob, num)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

function diag(s, Q)
    s.step % 50 == 0 || return
    nx, ny, nz = s.dec.nloc
    m1 = s.mom[1]
    ρmin, ρmax, umax = Inf, -Inf, 0.0
    for k in 1:nz, j in 1:ny, i in 1:nx
        I = gidx(s, i, j, k)
        ρ = Q[I, 1] + Q[I, 2]
        ρmin = min(ρmin, ρ); ρmax = max(ρmax, ρ)
        umax = max(umax, abs(Q[I, m1] / ρ))
    end
    ρmin = MPI.Allreduce(ρmin, min, s.dec.comm)
    ρmax = MPI.Allreduce(ρmax, max, s.dec.comm)
    umax = MPI.Allreduce(umax, max, s.dec.comm)
    rank == 0 && @printf("step %5d  t = %6.3f ms  ρ ∈ [%.4f, %.4f] kg/m³  |u|max = %6.1f m/s\n",
                         s.step, 1e3 * s.t, ρmin, ρmax, umax)
end
##
# ~2.5 ms captures the incident shock crossing the interface, reflecting off the
# end wall, and returning to re-shock it.
run!(s, Q; tfinal=2.5e-3, nmax=1_000_000, callback=diag)

# save_vtk(s, Q, "shock_tube_final")   # ρ, u, p, T, Y_He, Y_CO2 for ParaView
# save_checkpoint(s, Q, "shock_tube_final")
