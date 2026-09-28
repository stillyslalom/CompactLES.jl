# The problems, equations of state and driver shared by the two halves of the
# docs/src/reference/capabilities.md check: test/capability_tests.jl (the
# rejections and a smoke subset of accepted rows, in the serial suite) and
# test/capability_matrix.jl (every accepted row, in the weekly workflow).

module Capabilities

using CompactLES
using CompactLES.Regions: Box
using CompactLES: EOS, NavierStokes1T

# An equation of state defined outside the package. It implements the methods
# the extension page lists by forwarding to an IdealMixture it holds, so its
# run is the ideal one; a method missing from that list fails here as a
# MethodError.
struct ForwardingGas{T} <: EOS
    inner::IdealMixture{T}
end
const _FG = ForwardingGas
CompactLES.nspecies(e::_FG) = CompactLES.nspecies(e.inner)
CompactLES.species_names(e::_FG) = CompactLES.species_names(e.inner)
CompactLES.recover_primitives!(s, e::_FG, Q) =
    CompactLES.recover_primitives!(s, e.inner, Q)
CompactLES.conserved_from_prim(eq::NavierStokes1T, e::_FG, pr::Prim) =
    CompactLES.conserved_from_prim(eq, e.inner, pr)
CompactLES.species_enthalpy(e::_FG, k::Int, T_ion) =
    CompactLES.species_enthalpy(e.inner, k, T_ion)
CompactLES.eos_phi(e::_FG, ρ, p, T_ion, cp) = CompactLES.eos_phi(e.inner, ρ, p, T_ion, cp)
CompactLES.eos_dphi_dY(e::_FG, k::Int, ρ, p, T_ion, cp) =
    CompactLES.eos_dphi_dY(e.inner, k, ρ, p, T_ion, cp)
CompactLES.artificial_conductivity_scale(e::_FG, ρ, c, T_ion, cp) =
    CompactLES.artificial_conductivity_scale(e.inner, ρ, c, T_ion, cp)
CompactLES.wall_internal_energy(e::_FG, Q, I, n::Int, Twall) =
    CompactLES.wall_internal_energy(e.inner, Q, I, n, Twall)
CompactLES.mole_fraction(e::_FG, k::Int, Y, I, n::Int) =
    CompactLES.mole_fraction(e.inner, k, Y, I, n)
CompactLES.state_admissibility(e::_FG, ρ, en, Yat::F, n::Int) where {F} =
    CompactLES.state_admissibility(e.inner, ρ, en, Yat, n)

const per = (PeriodicBC(), PeriodicBC())
const wall = (SlipWallBC(), SlipWallBC())
const unit = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
const cyl_domain = ((0.0, 1.0), (0.0, 2π), (0.0, 1.0))
const axis = ((AxisBC(), SlipWallBC()), per, per)

pulse(x) = 1.0 + 0.1 * exp(-40 * (x - 0.5)^2)
ic_gas(x, y, z) = Prim(p=pulse(x), rho=1.0)
ic_two(x, y, z) = (θ = 0.5 + 0.4 * sin(2π * x); Prim(Y=(θ, 1 - θ), p=1.0, rho=1.0))
ic_air(x, y, z) = Prim(p=1e5 * pulse(x), T_ion=300.0)
ic_radial(r, a, b) = Prim(p=1.0 + 0.1 * exp(-20r^2), rho=1.0)
ic_shell(r, a, b) = Prim(p=1.0 + 0.1 * exp(-20(r - 0.75)^2), rho=1.0)

const EOSES = [
    ("IdealMixture", IdealSpecies("gas"; R=1, gamma=1.4), ic_gas),
    ("IdealMixture, two species",
     IdealMixture([IdealSpecies{Float64}("a", 1.0, 1.4),
                   IdealSpecies{Float64}("b", 0.5, 1.2)]), ic_two),
    ("StiffenedGas", StiffenedGas(gamma=4.4, p_inf=1.0, cv=1.0), ic_gas),
    ("Nasa9Mixture", Nasa9Mixture(["N2"]), ic_air),
    ("user EOS", ForwardingGas(IdealMixture([IdealSpecies{Float64}("gas", 1.0, 1.4)])),
     ic_gas),
]

device() = DeviceBackend(CompactLES.KernelAbstractions.CPU())

# The backend and precision pairs the geometry and layout rows run at; the
# EOS rows run all four.
const PAIRS = ((CompactLES.CPUBackend(), Float64), (device(), Float32))

problem(; domain=unit, bcs=(per, per, per), ic=ic_gas,
        eos=IdealSpecies("gas"; R=1, gamma=1.4), metric=CartesianMetric(),
        transport=ConstantTransport()) =
    Problem(; domain, bcs, ic, eos, metric, transport)

# Build and advance three steps. A device backend runs its pointwise bodies
# and staged exchanges through the KernelAbstractions kernels.
function advances(prob, num)
    on_device = num.execution.backend isa DeviceBackend
    CompactLES.FORCE_KA[] = on_device
    CompactLES.FORCE_DEVICE_EXCHANGE[] = on_device
    try
        solver, Q = setup(prob, num)
        run!(solver, Q; tfinal=1.0, nmax=3)
        return solver.step == 3
    finally
        CompactLES.FORCE_KA[] = false
        CompactLES.FORCE_DEVICE_EXCHANGE[] = false
    end
end

end # module Capabilities
