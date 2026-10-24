# Stage-0 capsule-only implosion driven in memory by a Briareus hohlraum drive
# (LLNL's two-step method: Briareus the integrated hohlraum step, CompactLES the
# capsule-only step). docs/src/explanation/ablation-drive.md describes the model.
#
# Needs an environment with both packages (Briareus is not registered):
#   julia --project=<env> -t 4 examples/briareus_capsule.jl nx=1400 tfinal=10.5
# where <env> has `Pkg.develop` of CompactLES and Briareus. Units: mm, ns, mg
# (density g/cm³, pressure 10⁴ Mbar, energy MJ).
#
# Stage-0 physics only: an ablation-pressure / mass-ablation-rate closure (carbon
# exponents, pressure prefactor calibrated to HYBRID-E's 145 Mbar peak, ṁ prefactor to
# its 5–6.6 % ablator remaining) on ideal-gas, fully ionized materials. No radiation
# transport, no table EOS, no conduction: shock timing, adiabat and stagnation are not
# predictions; v_imp, with both prefactors fixed, is the rocket's check. About 6 min on
# one thread at nx=1400 (the timestep is the driver gas's).

using CompactLES, Briareus, Printf
using CompactLES: drive_temperature, xcoord
MPI.Initialized() || MPI.Init(threadlevel=:funneled)

# rate: the ṁ prefactor (g cm⁻² s⁻¹ at 100 eV), calibrated to HYBRID-E's 5–6.6 %
# ablator remaining (docs/src/explanation/ablation-drive.md; Lindl's value is 3e5).
# driver_rho: the driver gas (g/cc); at 1e-3 the foot fails, see the AblationDrive docs.
const opt = CompactLES.script_args(ARGS, (nx=1400, tfinal=10.5, h=0.4, capsule_h=0.2,
                                          peak=145.0, rate=2.85e5, layer=0.005, fill=3e-4,
                                          driver_rho=1e-2, cfl=0.4))

# --- Step 1: the hohlraum (Briareus). N221204-class geometry and delivered pulse, the
# public mock HYBRID-E pointing on 48 quads, a static gold wall (examples/n221204.jl
# in Briareus has the calibrated dynamics and CBET).
t = hohlraum(; length=11.24, leh_radius=1.55, capsule_radius=1.135, h=opt.h,
             capsule_h=opt.capsule_h, wall=gold(albedo=WallLossAlbedo(), conversion=0.75),
             capsule=ablator())
beams = hybride_mock_pointing(nif_quads())
pt = [0.0, 0.4, 0.8, 1.3, 2.0, 2.5, 3.3, 3.9, 4.5, 5.0, 5.5, 8.0, 8.3, 8.6, 8.9, 9.1, 9.25, 9.35]
outer = [0, 2, 50, 50, 25, 26, 60, 65, 150, 270, 296, 294, 250, 110, 40, 0, 0, 0]
inner = [0, 2, 4, 4, 4.5, 5, 7, 9, 25, 90, 146, 146, 146, 146, 146, 146, 140, 0]
total = Float64.(outer .+ inner)
pulses = cone_fraction_pulses(beams, pt, total, inner ./ max.(total, eps()))
model = RadiationModel(t)
history = drive_history(model, DepositionOperator(t, beams), τ -> with_pulses(beams, pulses, τ),
                        [0.2:0.1:9.3; 9.35]; lmax=8)
cd = capsule_drive(history, model; provenance=(; pulse="N221204 delivered, by eye"))
println(cd)

# --- Step 2: the capsule (CompactLES), driven by the capsule drive in memory.
drive = radiation_drive(cd; lmax=0, peak=opt.peak, rate=opt.rate, layer=opt.layer, driver=3)
println(drive)
@printf("P_abl: %.1f Mbar where Tr first reaches 150 eV, %.1f Mbar at peak\n",
        ablation_pressure(drive, cd.time[findfirst(>=(150), cd.Tr)]),
        maximum(ablation_pressure(drive, τ) for τ in cd.time))

kmu = 9.648e-5                                  # k_B/m_u in mm² ns⁻² eV⁻¹
species = [IdealSpecies{Float64}("C", 7 / 12 * kmu, 5 / 3),     # HDC ablator, Z = 6
           IdealSpecies{Float64}("DT", 2 / 2.5 * kmu, 5 / 3),   # ice and gas
           IdealSpecies{Float64}("He", 3 / 4 * kmu, 5 / 3)]     # driver (hohlraum fill)
R_abl, R_ice, R_gas = 1.135, 1.050, 0.985       # N221204-class layers (mm), approximate
ic = (r, θ, φ, h) -> begin
    w = 1.5h
    b1, b2, b3 = tanh_blend(r, R_gas, w), tanh_blend(r, R_ice, w), tanh_blend(r, R_abl, w)
    pc = (b2 - b3) * 3.5 + 1e-8
    pd = (1 - b1) * opt.fill + (b1 - b2) * 0.255 + 1e-8
    ph = b3 * opt.driver_rho + 1e-8
    ρ = pc + pd + ph
    Prim(rho=ρ, p=1e-5, Y=(pc / ρ, pd / ρ, ph / ρ))      # 0.1 Mbar
end
problem = Problem(name="stage-0 capsule", eos=IdealMixture(species), metric=SphericalMetric(),
                  sources=(drive,), domain=((0.0, 1.4), (π / 2, π / 2 + 1), (0.0, 1.0)),
                  bcs=((OriginBC(), SlipWallBC()), PeriodicBC(), PeriodicBC()), ic=ic)
numerics = Numerics(n_global=(opt.nx, 1, 1), cfl=opt.cfl,
                    control=StepControl(retries=4, validity=:permissive, floor_ratio=1e-3),
                    execution=Execution(dims=(1, 1, 1)))
solver, Q = setup(problem, numerics)

# The shell: the ablator inside the front (r < R_f + 3 layer, R_f as the source finds
# it) is the ablator remaining; the rest of the unconverted ablator is a decompressed
# tail behind the front. KE counts ablator and DT inside the front.
function shell(solver, Q)
    refresh_primitives!(solver, Q)
    o, h, n = solver.decomp.n_halo_d[1], solver.h[1], solver.decomp.n_local[1]
    r = [xcoord(solver, 1, i) for i in 1:n]
    ρC = [solver.rho[o+i, 1, 1] * solver.Y[1][o+i, 1, 1] for i in 1:n]
    g = [1 < i < n ? max(ρC[i], 0) * max(ρC[i-1] - ρC[i+1], 0) : 0.0 for i in 1:n]
    Rf = sum(g .* r) / sum(g)
    mC = mS = mD = pD = KE = 0.0
    for i in 1:n
        I = o + i
        dV = 4π * r[i]^2 * h
        u = solver.u[I, 1, 1]
        c, d = ρC[i] * dV, solver.rho[I, 1, 1] * solver.Y[2][I, 1, 1] * dV
        mC += c
        mD += d
        pD += d * u
        r[i] < Rf + 3opt.layer && (mS += c; KE += (c + d) * u^2 / 2)
    end
    return (; mC, mS, v=pD / mD, KE)
end

s0 = shell(solver, Q)
best = Ref((v=0.0, t=0.0, mS=s0.mS, mC=s0.mC, KE=0.0))
# One run! call; a callback samples the shell every 20 ps and stops the run well past
# peak velocity. The retries (StepControl) recover the first shock's convergence on the
# origin near 7.3 ns, which collapses the timestep once at CFL 0.4.
sample = (solver, Q) -> begin
    s = shell(solver, Q)
    -s.v > best[].v && (best[] = (v=-s.v, t=solver.t, mS=s.mS, mC=s.mC, KE=s.KE))
    isapprox(4solver.t, round(4solver.t); atol=1e-6) &&
        @printf("t %5.2f ns  Tr %5.1f eV  P_abl %6.1f Mbar  v_DT %6.1f km/s  ablator %5.1f %% (unconverted %5.1f %%)  KE %5.2f kJ\n",
                solver.t, drive_temperature(drive, solver.t), ablation_pressure(drive, solver.t),
                -1e3s.v, 100s.mS / s0.mS, 100s.mC / s0.mC, 1e3s.KE)
    flush(stdout)
    return best[].v > 0.05 && -s.v < 0.8 * best[].v   # well past peak velocity (> 50 km/s)
end
run!(solver, Q; tfinal=opt.tfinal, nmax=10_000_000, callback=Callback(EveryTime(0.02), sample))
b = best[]
@printf("peak DT velocity %.0f km/s at %.2f ns, ablator remaining %.1f %% (unconverted %.1f %%), shell KE %.1f kJ (stage 0)\n",
        1e3b.v, b.t, 100b.mS / s0.mS, 100b.mC / s0.mC, 1e3b.KE)
