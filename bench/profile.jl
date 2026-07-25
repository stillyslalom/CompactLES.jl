# Profile the standard workloads and report self-time by CompactLES source line.
#
#   julia --project=. -t 8 bench/profile.jl [workload]
#
# Workloads (the three shapes real runs take):
#   tgv    -- Taylor-Green, 3-D periodic, single species, art off. The
#             bandwidth-bound case: every dimension resolved and threaded.
#   tube   -- multicomponent shock tube, 2-D (z collapsed), art on. Exercises
#             species transport and the artificial-property sensors.
#   radial -- 1-D cylindrical converging shock on the axis fold. The case where
#             per-call overhead, not bandwidth, used to dominate.
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Profile, Printf
const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

function tgv(N=64)
    γ = 1.4; c0 = 10.0; p0 = c0^2 / γ
    s, Q = setup(Problem(eos=single_species(gamma=γ), transport=Transport(mu0=1 / 1600),
                         domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)), bcs=per3,
                         ic=(x, y, z) -> Prim(
                             u=(sin(x) * cos(y) * cos(z), -cos(x) * sin(y) * cos(z), 0.0),
                             p=p0 + (1 / 16) * (cos(2x) + cos(2y)) * (cos(2z) + 2),
                             rho=1.0)),
                 Numerics(nglob=(N, N, N), art=ArtParams(enabled=false), cfl=0.6))
    s, Q
end

function tube(N=512)
    eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                        IdealSpecies{Float64}("heavy", 0.2, 1.09)])
    hx = 1.0 / (N - 1)
    s, Q = setup(Problem(eos=eos, transport=Transport(mu0=0.0),
                         domain=((0.0, 1.0), (0.0, 32hx), (0.0, 32hx)),
                         bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
                         ic=(x, y, z) -> begin
                             θ = tanh_blend(x, 0.5, 4hx)
                             Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ,
                                  p=(1 - θ) + 0.1θ)
                         end),
                 Numerics(nglob=(N, 32, 1), art=ArtParams(enabled=true), cfl=0.5))
    s, Q
end

function radial(N=1024)
    s, Q = setup(Problem(domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                         metric=CylindricalMetric(),
                         bcs=((AxisBC(), SlipWallBC()), per3[2], per3[3]),
                         ic=(r, θ, z) -> Prim(u=(0, 0, 0),
                                              p=1 + 4exp(-200(r - 0.7)^2), rho=1.0)),
                 Numerics(nglob=(N, 1, 1), art=ArtParams(enabled=true), cfl=0.5))
    s, Q
end

const BUILDERS = Dict("tgv" => tgv, "tube" => tube, "radial" => radial)
which = isempty(ARGS) ? ["tgv", "tube", "radial"] : ARGS

for name in which
    s, Q = BUILDERS[name]()
    dQ = zero(Q); du = zero(Q)
    dt = compute_dt(s, Q)
    step!(s, Q, dQ, du, dt)                      # warm up / compile

    nsteps = name == "tgv" ? 3 : 20
    t = @elapsed for _ in 1:nsteps
        step!(s, Q, dQ, du, dt)
    end
    npt = prod(s.dec.nloc)
    @printf("\n===== %s: %d points, %d steps, %.1f ms/step (%.1f ns/pt/step) =====\n",
            name, npt, nsteps, 1e3t / nsteps, 1e9t / nsteps / npt)

    Profile.clear()
    Profile.init(n=10_000_000, delay=0.0005)
    @profile for _ in 1:nsteps
        step!(s, Q, dQ, du, dt)
    end

    buf = IOBuffer()
    Profile.print(IOContext(buf, :displaysize => (24, 200)); format=:flat,
                  combine=true, sortedby=:count, mincount=0)
    lines = split(String(take!(buf)), '\n')
    # keep only frames in our own source, print the heaviest
    ours = filter(l -> occursin("CompactLES", l) && occursin(r"\bsrc[\\/]", l), lines)
    total = 0
    for l in ours
        m = match(r"^\s*(\d+)", l)
        m === nothing || (total = max(total, parse(Int, m.captures[1])))
    end
    println("  samples  self%  location")
    shown = 0
    for l in ours
        m = match(r"^\s*(\d+)\s", l)
        m === nothing && continue
        c = parse(Int, m.captures[1])
        c < max(total ÷ 50, 2) && continue
        loc = replace(strip(l), r"^\d+\s+" => "")
        loc = replace(loc, r".*[\\/]src[\\/]" => "src/")
        @printf("  %7d  %5.1f  %s\n", c, 100c / total, first(loc, 120))
        (shown += 1) >= 18 && break
    end
end
println("\nprofile complete")
