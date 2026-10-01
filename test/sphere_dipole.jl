# The dipole radiated by an oscillating sphere, leaving through an
# NSCBCOutflowBC at the outer radius of a spherical grid resolved in r and the
# polar angle, and the reflection coefficient of that face. The configuration
# is the Oscillating sphere tutorial's (docs/literate/oscillating_sphere.jl).
#
# `test/serial_suite.jl` includes this and asserts the reflection on a coarse
# grid; `bench/sphereoutflow.jl` includes it and tabulates the reflection
# over outer radii, relaxation strengths and resolutions. Include after
# CompactLES.

"""
    sphere_dipole_reflection(R; sigma=0.25, n=128, ntheta=16, samples=32)

The sphere of radius a = 0.1 oscillates along the polar axis with velocity
U cos(ωt), U = 1e-3, at wavelength λ = 0.5 (k = ω = 4π, ρ = c = 1,
γ = 1.4); a `DirichletBC` holds the exact dipole at r = a and the run starts
from it. The outer face at `R` carries `NSCBCOutflowBC(pinf = 1/γ; sigma)`.
The radial spacing is that of `n` nodes over [a, 2.1], and the run ends at
t = R − a, before the wave the face reflects returns from the sphere. Over
the last period the pressure on the line of nodes nearest the axis is
sampled at `samples` times and accumulated into its complex amplitude P̂(r),
which is fitted by least squares to α h₁(kr) + β h₂(kr), the outgoing and
incoming spherical Hankel functions, over the radii the reflected wave has
crossed by the start of the sampling: from a + λ + 0.1 to two spacings
inside the face. Since |h₁| = |h₂| on the real axis, |β/α| is the face's
reflection coefficient.

Returns `(reflection = |β/α|, amplitude = |α/A|, steps, nr, kR)`, `A` the
exact amplitude.
"""
function sphere_dipole_reflection(R; sigma=0.25, n=128, ntheta=16, samples=32)
    γ, a, λ, U = 1.4, 0.1, 0.5, 1e-3
    k = 2π / λ
    h1(x) = -cis(x) * (x + im) / x^2
    dh1(x) = -im * cis(x) / x - 2h1(x) / x       # h₁′ = h₀ − 2h₁/x
    A = im * U / dh1(k * a)
    function exact(r, θ, t)
        e = cis(-k * t)
        p = real(A * h1(k * r) * cos(θ) * e)
        ur = real(A * dh1(k * r) * cos(θ) * e / im)
        uθ = -real(A * h1(k * r) * sin(θ) * e / (im * k * r))
        return Prim(rho = 1 + p, p = 1 / γ + p, u = (ur, uθ, 0.0))
    end
    h = (2.1 - a) / (n - 1)
    nr = round(Int, (R - a) / h) + 1
    problem = Problem(
        name = "sphere dipole",
        eos = IdealSpecies("gas"; R = 1.0, gamma = γ),
        metric = SphericalMetric(),
        domain = ((a, R), (0.0, π), (0.0, 1.0)),
        bcs = ((DirichletBC((r, θ, φ, t) -> exact(r, θ, t)),
                NSCBCOutflowBC(pinf = 1 / γ, sigma = sigma)),
               (PoleBC(), PoleBC()), PeriodicBC()),
        ic = (r, θ, φ) -> exact(r, θ, 0.0))
    solver, Q = setup(problem, Numerics(n_global = (nr, ntheta, 1)))
    tend = R - a
    times = [tend - λ + m * λ / samples for m in 1:samples]
    Phat = zeros(ComplexF64, nr)
    sample = Callback(AtTime(times), function (solver, Q)
        snap = field_snapshot(solver, Q; fields = (:p,))
        Phat .+= (2 / samples) .* (snap[:p][:, 1, 1] .- 1 / γ) .* cis(k * solver.t)
        nothing
    end)
    run!(solver, Q; tfinal = tend, nmax = 100_000, callback = sample)
    snap = field_snapshot(solver, Q; fields = (:p,))
    r, θ1 = snap.coords[1], snap.coords[2][1]
    window = findall(x -> a + λ + 0.1 <= x <= R - 2h, r)
    M = hcat(h1.(k .* r[window]), conj.(h1.(k .* r[window]))) .* cos(θ1)
    α, β = M \ Phat[window]
    return (reflection = abs(β / α), amplitude = abs(α / A), steps = solver.step,
            nr = nr, kR = k * R)
end
