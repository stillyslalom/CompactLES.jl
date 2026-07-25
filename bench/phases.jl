# Phase budget of compute_rhs!. A sampling profile of this solver is flat --
# it is bandwidth-bound across many full-array passes, not concentrated in one
# kernel -- so what matters is which PHASE owns the traffic, and how many array
# passes each phase costs. This replays compute_rhs! phase by phase and times
# each, plus counts the derivative solves it issues.
using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES
per3 = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

best(f; reps=20) = (f(); f(); minimum(@elapsed(f()) for _ in 1:reps))

function budget(name, solver, Q)
    decomp = solver.decomp
    dQ = zero(Q)
    vel = (solver.u, solver.v, solver.w)
    nx, ny, nz = decomp.n_local
    o1, o2, o3 = decomp.n_halo_d
    nact = count(decomp.active)

    CL.exchange_state!(Q, decomp); CL.primitives!(solver, Q)

    t = Dict{String,Float64}()
    t["exchange_state!"] = best(() -> CL.exchange_state!(Q, decomp))
    t["primitives!"]     = best(() -> CL.primitives!(solver, Q))
    t["velocity grads"]  = best(() -> for jj in 1:3, d in 1:3
        if decomp.active[d]
            CL.deriv_along!(solver.grad_u[d, jj], vel[jj], solver, d, CL.vel_parity(solver, d, jj))
            CL._scale_grad!(solver.grad_u[d, jj], solver, d)
        end
    end)
    t["metric grad corr"] = best(() -> CL.metric_correct_gradients!(solver, solver.metric))
    t["artificial"]       = best(() -> CL.compute_artificial!(solver, Q))
    t["scalar grads"]     = best(() -> for d in 1:3
        decomp.active[d] || continue
        CL.deriv_along!(solver.grad_T_ion[d], solver.T_ion, solver, d, 1); CL._scale_grad!(solver.grad_T_ion[d], solver, d)
        for sp in 1:solver.equations.n_species
            CL.deriv_along!(solver.grad_Y[d, sp], solver.Y[sp], solver, d, 1)
            CL._scale_grad!(solver.grad_Y[d, sp], solver, d)
        end
    end)
    t["assemble_fluxes!"] = best(() -> CL.assemble_fluxes!(solver, Q))
    t["flux halo exch"]   = best(() -> for d in 1:3
        CL.exchange_dim_batch!(view(solver.flux, d, :), decomp, d)
    end)
    # mirrors the branch compute_rhs! actually takes for this solver
    unitgeom = solver.metric isa CartesianMetric && all(isnothing, solver.stretch)
    t["flux divergence"]  = best(() -> for c in 1:solver.equations.n_cons, d in 1:3
        decomp.active[d] || continue
        Fdc = solver.flux[d, c]
        σ = solver.folds[d] === nothing ? 1 : solver.folds[d].sigflux[c]
        if unitgeom
            CL.deriv_along!(solver.tmp_a, Fdc, solver, d, σ)
            @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
                dQ[i+o1, j+o2, k+o3, c] -= solver.tmp_a[i+o1, j+o2, k+o3]
            end
        else
            Ad = solver.area_d[d]
            @inbounds for idx in eachindex(solver.tmp_b)
                solver.tmp_b[idx] = Ad[idx] * Fdc[idx]
            end
            CL.deriv_along!(solver.tmp_a, solver.tmp_b, solver, d, σ)
            @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
                I = CartesianIndex(i + o1, j + o2, k + o3)
                dQ[I, c] -= solver.inv_J[I] * solver.tmp_a[I]
            end
        end
    end)
    t["metric sources"]   = best(() -> CL.add_metric_sources!(solver, dQ, Q, solver.metric))
    t["apply_bcs!"]       = best(() -> apply_bcs!(solver, Q))

    whole = best(() -> compute_rhs!(solver, Q, dQ))
    npt = prod(decomp.n_local)
    # velocity + scalar + flux solves
    nsolves = 3 * nact + nact * (1 + solver.equations.n_species) +
              solver.equations.n_cons * nact
    @printf("\n===== %s =====\n", name)
    @printf("  %d points, %d species, %d active dims; compute_rhs! = %.3f ms (%.1f ns/pt)\n",
            npt, solver.equations.n_species, nact, 1e3whole, 1e9whole / npt)
    @printf("  %d compact line-solves per RHS\n", nsolves)
    tot = sum(values(t))
    println("  phase                    ms      % of sum")
    for (k, v) in sort(collect(t), by=x -> -x[2])
        @printf("    %-20s %8.3f   %5.1f%%\n", k, 1e3v, 100v / tot)
    end
    @printf("    %-20s %8.3f\n", "(sum of phases)", 1e3tot)
end

# tgv-like: 3-D periodic, single species, art off
s1 = Solver(n_global=(64, 64, 64), L_domain=(2π, 2π, 2π), bcs=per3,
            transport=Transport(mu0=1e-3), art=ArtParams(enabled=false))
Q1 = allocate_state(s1)
initialize!(s1, Q1, (x, y, z) -> Prim(u=(sin(x) * cos(y), -cos(x) * sin(y), 0.0),
                                      p=1 + 0.05cos(2z), rho=1.0))
budget("tgv 64^3, 1 species, art off", s1, Q1)

# tube-like: 2-D, two species, art on
eos = IdealMixture([IdealSpecies{Float64}("light", 1.0, 1.4),
                    IdealSpecies{Float64}("heavy", 0.2, 1.09)])
s2 = Solver(n_global=(512, 32, 1), L_domain=(1.0, 0.06, 1.0), eos=eos,
            bcs=((SlipWallBC(), SlipWallBC()), per3[2], per3[3]),
            art=ArtParams(enabled=true))
Q2 = allocate_state(s2)
initialize!(s2, Q2, (x, y, z) -> begin
    θ = tanh_blend(x, 0.5, 0.02)
    Prim(Y=(1 - θ, θ), rho=(1 - θ) + 0.625θ, p=(1 - θ) + 0.1θ)
end)
budget("tube 512x32, 2 species, art on", s2, Q2)
