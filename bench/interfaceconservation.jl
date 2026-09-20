# Interface-conservation instrument: composite conserved budgets of the
# patch and level coupling.
#
# A periodic two-species layer makes the conservation accounting unambiguous:
# there is no physical boundary flux to subtract. There is no physical
# transport, and of the artificial properties only the mass-fraction bound
# (`C_Y`) is on: a passive sheet a few cells wide rings past the 1e-4 dead
# band within one transit without it, on the uniform grid as much as at an
# interface. The bound holds an excursion only asymptotically, and the root
# and level-1 nodes along the edges of a nested box still undershoot by a
# few 1e-4 over two transits, so the runs are permissive and the largest
# mass-fraction excursion is reported next to the drift; density, pressure
# and velocity stay at their initial values throughout.
# `layer` is a long, weakly perturbed mixing layer; `moving` advects a
# sharper species interface with a translating shear and takes immediate
# composite snapshots on either side of repeated regrids. The script reports
# temporal drift from each configuration's *own* initialized composite
# quadrature.
# Differences from the uniform run are attribution comparisons, not a claim
# that either interface difference is an exact operator defect.
#
# Typical runs:
#   julia --project=. bench/interfaceconservation.jl smoke=1
#   mpiexec -n 8 julia --project=. -t 1 bench/interfaceconservation.jl
#   mpiexec -n 8 julia --project=. -t 1 bench/interfaceconservation.jl N=96 ny=24 \
#       tfinal=50.26548245743669 moving_tfinal=50.26548245743669 samples=16 check=true
# `maxlevels=2` runs the bounded two-level comparison independently of the
# deeper hierarchy's validity gate. `parts=regrid` isolates moving refinement.
#
# The predeclared application budgets are deliberately coarse enough to be
# useful on a long, interface-crossing calculation: 0.1% of initial mass,
# each species mass, and energy; 0.1% of M0*c0 for momentum.  M0*c0 gives a
# meaningful scale to the initially zero transverse momentum components.
# The immediate regrid budget consumes that same 0.1% allowance.  Mixing
# comparisons have an independently declared 1% molecular-fraction and 1%
# domain-width tolerance.  These numbers are stated
# before executing this instrument; do not retune them to a measured drift.

using MPI
MPI.Initialized() || MPI.Init(threadlevel=:funneled)
using CompactLES
using Printf
const CL = CompactLES

# `@printf` takes a literal format only; this wraps a format assembled from
# several literals so a long line can be split.
printfmt(fmt::String, args...) = Printf.format(stdout, Printf.Format(fmt), args...)

const periodic = ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3)

const args = CompactLES.script_args(ARGS, (N=192, ny=32, tfinal=8.0, nmax=typemax(Int),
                                           smoke=false, moving_tfinal=8.0,
                                           samples=8, parts="all", check=false,
                                           moving_width=0.18, filter_interval=1,
                                           maxlevels=3);
                                    positional=(:N, :tfinal))

# Application comparison budgets, fixed independently of the results.
const EVOLUTION_BUDGET = 1e-3
const REGRID_BUDGET = 1e-3
const INITIAL_SAMPLING_BUDGET = 1e-3
const MOLECULAR_COMPARISON_BUDGET = 1e-2
const WIDTH_COMPARISON_BUDGET = 0.01 * 8pi

_states(Q::Vector) = Q
_states(Q) = [Q]

"""One collective, composite conserved snapshot plus the established mixing measures."""
function snapshot(solver, Q)
    # `_conserved_budget` has one Allreduce for all conserved components.  It
    # is called by every world rank, including ranks which own only the root.
    b = CL._conserved_budget(solver, Q)
    S = _states(Q)
    mixq = Q isa Vector ? S : Q
    return (; conserved=b,
            width=mix_width(solver, mixq; dim=1),
            molecular=molecular_mixing(solver, mixq; dim=1),
            excursion=species_excursion(solver, S),
            time=solver.t, step=solver.step)
end

# The largest mass-fraction excursion outside [0, 1] over every interior node
# of every patch, reduced over the run: the quantity strict validity would
# reject past its 1e-4 dead band.
function species_excursion(solver, states)
    worst = 0.0
    for (ps, Q) in CL.eachpatch(solver, states)
        refresh_primitives!(ps, Q)
        n1, n2, n3 = ps.decomp.n_local
        o1, o2, o3 = ps.decomp.n_halo_d
        for Y in ps.Y, k in 1:n3, j in 1:n2, i in 1:n1
            y = Y[i + o1, j + o2, k + o3]
            worst = max(worst, -y, y - 1)
        end
    end
    return MPI.Allreduce(worst, max, solver.comm)
end

function drift(now, before; c0, scale=before)
    a, b, s = now.conserved, before.conserved, scale.conserved
    # The nonzero scale for a zero initial momentum is the acoustic momentum
    # scale M0*c0, rather than an arbitrary division by a round-off vector.
    mscale = max(abs(s.total_mass), eps(Float64))
    escale = max(abs(s.total_energy), eps(Float64))
    sscale = max.(abs.(s.species_masses), eps(Float64))
    pscale = mscale * c0
    return (; mass=(a.total_mass - b.total_mass) / mscale,
            species=(a.species_masses .- b.species_masses) ./ sscale,
            momentum=(a.momentum .- b.momentum) ./ pscale,
            energy=(a.total_energy - b.total_energy) / escale,
            width=now.width - before.width,
            molecular=now.molecular - before.molecular)
end

maxconserved(d) = maximum((abs(d.mass), abs(d.energy),
                            maximum(abs, d.species), maximum(abs, d.momentum)))

_zero_drift(nsp) = (; mass=0.0, species=zeros(nsp), momentum=(0.0, 0.0, 0.0),
                    energy=0.0, width=0.0, molecular=0.0)
_add_drift(a, b) = (; mass=a.mass + b.mass, species=a.species .+ b.species,
                     momentum=ntuple(i -> a.momentum[i] + b.momentum[i], 3),
                     energy=a.energy + b.energy, width=a.width + b.width,
                     molecular=a.molecular + b.molecular)
_sub_drift(a, b) = (; mass=a.mass - b.mass, species=a.species .- b.species,
                     momentum=ntuple(i -> a.momentum[i] - b.momentum[i], 3),
                     energy=a.energy - b.energy, width=a.width - b.width,
                     molecular=a.molecular - b.molecular)

function layer_ic(sharp::Bool=false; moving_width=args.moving_width)
    # `sin(x/4)` makes the two species sheets periodic.  Equal thermodynamic
    # species keep density uniform; the translating x velocity leaves the y/z
    # momenta exactly zero, which is why their scale is M0*c0 above.
    # Still sharper than the mixing layer, but resolved on the root smoke grid
    # without accepting negative species under a permissive validity policy.
    δ = sharp ? moving_width : 0.22
    return (x, y, z) -> begin
        y2 = 0.5 * (1 + tanh(sin(x / 4 - 0.12sin(y)) / δ))
        Prim(Y=(1 - y2, y2), rho=1.0, p=1.0,
             u=(1.0, 0.08sin(x / 4), 0.0))
    end
end

function regions(N, ny, depth)
    # Four root nodes remain outside the level-1 box for the interpolation
    # shell, including in the small `smoke=1` transverse grid.
    ylo = max(4, ny ÷ 4)
    r1 = BlockRegion((N ÷ 2 - N ÷ 6, ylo, 0),
                     (N ÷ 3, min(ny - 4, ylo + ny ÷ 2) - ylo, 1))
    depth == 2 && return r1
    n1x, n1y = 3r1.extent[1] - 2, 3r1.extent[2] - 2
    # A further four-node imposed shell surrounds level 1, so the level-2
    # region starts eight level-1 nodes from the nominal parent edge.
    r2 = BlockRegion((3r1.offset[1] + 8, 3r1.offset[2] + 8, 0),
                     (n1x - 16, n1y - 16, 1))
    return [r1, r2]
end

function build(mode, N, ny; subcycle=false, regrid=false)
    eos = IdealMixture([IdealSpecies{Float64}("species-a", 1.0, 1.4),
                        IdealSpecies{Float64}("species-b", 1.0, 1.4)])
    kw = if mode === :uniform
        (;)
    elseif mode === :samelevel
        (patch_grid=(2, 1, 1),)
    else
        (refine=regions(N, ny, mode === :depth2 ? 2 : 3), subcycle=subcycle,
         # Kept out of the ordinary stepping loop; `regrid_history!` performs
         # one collective regrid between its before/after snapshots.
         regrid_interval=regrid ? 1_000_000 : 0, tag_threshold=1e6,
         # Species-gradient tagging follows the translating composition sheet;
         # density is intentionally uniform in this thermodynamic control.
         tag_gradient_threshold=regrid ? 0.02 : 0, tag_buffer=3)
    end
    return Solver(n_global=(N, ny, 1), L_domain=(8pi, 2pi, 1.0), bcs=periodic,
                  eos=eos, cfl=0.45,
                  art=ArtParams(C_mu=0.0, C_beta=0.0, C_kappa=0.0, C_D=0.0),
                  control=StepControl(validity=:permissive),
                  filter_interval=args.filter_interval; kw...)
end

function evolve(mode, N, ny, tfinal, nmax, subcycle)
    solver = build(mode, N, ny; subcycle)
    Q = allocate_state(solver)
    initialize!(solver, Q, layer_ic())
    initial = snapshot(solver, Q)
    ws = Workspace(Q)
    history = Any[initial]
    # Separate run! calls land exactly on each requested physical observation
    # time, retaining max excursion rather than allowing final cancellation to
    # hide an earlier interface defect.
    for target in range(tfinal / args.samples, tfinal; length=args.samples)
        run!(solver, Q, ws; tfinal=target, nmax=nmax)
        sample = snapshot(solver, Q)
        push!(history, sample)
        if MPI.Comm_rank(solver.comm) == 0
            printfmt("sample %-9s subcycle=%-5s t=%.15g W=%.8e theta=%.8e " *
                    "excursion %.2e\n", mode, subcycle, sample.time,
                    sample.width, sample.molecular, sample.excursion)
            show_drift("  sampled drift", drift(sample, initial; c0=sqrt(1.4)))
        end
        solver.step >= nmax && break
    end
    final = snapshot(solver, Q)
    # c0 = sqrt(gamma*p/rho) at the uniform initial state.
    ds = [drift(s, initial; c0=sqrt(1.4)) for s in history]
    return (; initial, final, history, drift=drift(final, initial; c0=sqrt(1.4)),
            max_excursion=maximum(maxconserved, ds),
            species_excursion=maximum(s.excursion for s in history),
            complete=solver.t >= tfinal - 16eps(tfinal),
            wall=MPI.Allreduce(solver.wall_total, max, solver.comm),
            steps=solver.step)
end

function regrid_history!(N, ny, tfinal, nmax, subcycle)
    solver = build(:depth2, N, ny; subcycle, regrid=true)
    Q = allocate_state(solver)
    initialize!(solver, Q, layer_ic(true))
    ws = Workspace(Q)
    initial = snapshot(solver, Q)
    events = Any[]
    history = Any[initial]
    cumulative_jump = 0.0
    signed_jump = _zero_drift(length(initial.conserved.species_masses))
    max_evolution = 0.0
    for target in range(tfinal / args.samples, tfinal; length=args.samples)
        run!(solver, Q, ws; tfinal=target, nmax=nmax)
        before = snapshot(solver, Q)
        push!(history, before)             # retain both sides of every transfer
        residual = _sub_drift(drift(before, initial; c0=sqrt(1.4)), signed_jump)
        max_evolution = max(max_evolution, maxconserved(residual))
        # `regrid!` expects its cadence counter to have been advanced by run!'s
        # normal collective check.  Advance it explicitly because automatic
        # regrids are suppressed to isolate each immediate transfer jump.
        solver.regrid.checks += 1
        changed = CL.regrid!(solver, _states(Q), ws, nothing)
        after = snapshot(solver, Q)
        push!(history, after)
        if changed
            jump = drift(after, before; c0=sqrt(1.4), scale=initial)
            cumulative_jump += maxconserved(jump)
            signed_jump = _add_drift(signed_jump, jump)
            push!(events, (; before, after, jump))
        end
        if MPI.Comm_rank(solver.comm) == 0
            @printf("regrid sample subcycle=%-5s t=%.15g changed=%s cumulative=%.8e\n",
                    subcycle, after.time, changed, cumulative_jump)
            changed && show_drift("  immediate regrid jump", events[end].jump)
            show_drift("  evolution residual", residual)
            flush(stdout)
        end
        solver.step >= nmax && break
    end
    final = history[end]
    ds = [drift(s, initial; c0=sqrt(1.4)) for s in history]
    total = drift(final, initial; c0=sqrt(1.4))
    return (; initial, final, history, events, drift=total,
            evolution_residual=_sub_drift(total, signed_jump), signed_jump,
            max_excursion=maximum(maxconserved, ds), cumulative_jump, max_evolution,
            species_excursion=maximum(s.excursion for s in history),
            complete=solver.t >= tfinal - 16eps(tfinal))
end

function show_drift(label, d; budget=EVOLUTION_BUDGET)
    @printf("%-24s max conserved %.3e  [budget %.1e]  mass %+.3e energy %+.3e  ",
            label, maxconserved(d), budget, d.mass, d.energy)
    @printf("species (%+.3e, %+.3e) momentum (%+.3e, %+.3e, %+.3e)  ",
            d.species..., d.momentum...)
    @printf("dwidth %+.3e dtheta %+.3e\n", d.width, d.molecular)
    flush(stdout) # MPI.Abort does not flush redirected stdout after a failed gate.
end

function show_scale(label, s)
    b = s.conserved
    printfmt("  %-22s initial M %.8e species (% .8e, % .8e) " *
            "P (% .8e, % .8e, % .8e) E %.8e; Pscale %.8e\n",
            label, b.total_mass, b.species_masses..., b.momentum..., b.total_energy,
            b.total_mass * sqrt(1.4))
end

verdict(value, budget, complete) = !complete ? "INCOMPLETE" :
                                   value <= budget ? "PASS" : "FAIL"

function main()
    rank, np = MPI.Comm_rank(MPI.COMM_WORLD), MPI.Comm_size(MPI.COMM_WORLD)
    N, ny, tfinal, nmax, moving_tfinal = args.N, args.ny, args.tfinal, args.nmax,
                                         args.moving_tfinal
    if args.smoke
        N, ny, tfinal, nmax, moving_tfinal = min(N, 72), min(ny, 24), min(tfinal, 0.3),
                                                min(nmax, 120), min(moving_tfinal, 0.8)
    end
    rank == 0 && printfmt("=== interface conservation: N=(%d,%d,1), t=%.15g, " *
                          "moving_t=%.15g, np=%d, samples=%d, parts=%s, " *
                          "maxlevels=%d ===\n",
                          N, ny, tfinal, moving_tfinal, np, args.samples,
                          args.parts, args.maxlevels)
    rank == 0 && println("Predeclared evolution and immediate-regrid budgets = 1e-3.")
    rank == 0 && println("Regularization: art=bound only, " *
                         "filter_interval=$(args.filter_interval), cfl=0.45, " *
                         "validity=permissive, moving_width=$(args.moving_width), " *
                         "check=$(args.check).")
    rank == 0 && flush(stdout)
    parts = Set(Symbol.(split(args.parts, ',')))
    allparts = :all in parts
    all(p -> p in (:all, :mixing, :regrid), parts) ||
        error("parts must be all, mixing, regrid, or mixing,regrid")
    args.samples >= 1 || error("samples must be at least one")
    all(t -> isfinite(t) && t > 0, (tfinal, moving_tfinal)) ||
        error("tfinal and moving_tfinal must be finite and positive")
    isfinite(args.moving_width) && args.moving_width > 0 ||
        error("moving_width must be finite and positive")
    args.maxlevels in (2, 3) || error("maxlevels must be 2 or 3")
    N >= 48 || error("N must be at least 48 for the nested C8-filter shells")
    if allparts || :mixing in parts || :regrid in parts
        ny >= 18 || error("ny must be at least 18 for the three-level nest")
    end
    results = Dict{Tuple{Symbol,Bool},Any}()
    regrid_failure = false
    mixing_failure = false
    initial_sampling_failure = false
    if allparts || :mixing in parts
        for mode in (:uniform, :samelevel, :depth2, :depth3), subcycle in (false, true)
            mode === :depth3 && args.maxlevels == 2 && continue
            mode in (:uniform, :samelevel) && subcycle && continue
            r = evolve(mode, N, ny, tfinal, nmax, subcycle)
            results[(mode, subcycle)] = r
            if rank == 0
                show_drift("$(mode), subcycle=$(subcycle)", r.drift)
                show_scale("$(mode), subcycle=$(subcycle)", r.initial)
                @printf("  evolution history max %.3e: %s (%d steps, t=%.4f)\n",
                        r.max_excursion,
                        verdict(r.max_excursion, EVOLUTION_BUDGET, r.complete),
                        r.steps, r.final.time)
                @printf("  species excursion max %.2e\n", r.species_excursion)
            end
        end
    end
    # Attribution only: subtracting two independently initialized composite
    # quadratures removes neither their spatial error nor their physical drift.
    if rank == 0 && (allparts || :mixing in parts)
        u = results[(:uniform, false)]
        println("Attribution deltas (configuration drift minus uniform drift):")
        for key in sort!(collect(keys(results)); by=string)
            key == (:uniform, false) && continue
            rr, ur = results[key], results[(:uniform, false)]
            r, ud = rr.drift, ur.drift
            init = drift(rr.initial, ur.initial; c0=sqrt(1.4))
            initial_sampling_failure |= maxconserved(init) > INITIAL_SAMPLING_BUDGET
            if rr.complete && ur.complete
                dtheta, dwidth = rr.final.molecular - ur.final.molecular,
                                 rr.final.width - ur.final.width
                htheta = maximum(abs(a.molecular - b.molecular)
                                 for (a, b) in zip(rr.history, ur.history))
                hwidth = maximum(abs(a.width - b.width)
                                 for (a, b) in zip(rr.history, ur.history))
                mixok = htheta <= MOLECULAR_COMPARISON_BUDGET &&
                        hwidth <= WIDTH_COMPARISON_BUDGET
                mixing_failure |= !mixok
                printfmt("  %-22s initial max %.3e [budget %.1e], mass %+.3e energy %+.3e " *
                        "width %+.3e theta %+.3e; final mass %+.3e energy %+.3e; " *
                        "dwidth %+.3e dtheta %+.3e history (width %.3e, theta %.3e); " *
                        "mixing %s\n",
                        string(key), maxconserved(init), INITIAL_SAMPLING_BUDGET,
                        init.mass, init.energy, rr.initial.width - ur.initial.width,
                        rr.initial.molecular - ur.initial.molecular,
                        r.mass - ud.mass, r.energy - ud.energy, dwidth, dtheta, hwidth, htheta,
                        mixok ? "PASS" : "FAIL")
            else
                printfmt("  %-22s initial mass %+.3e energy %+.3e; comparison INCOMPLETE " *
                        "(unequal endpoint)\n",
                        string(key), init.mass, init.energy)
            end
        end
        println("Predeclared mixing-comparison budgets: " *
                "|dtheta| <= $(MOLECULAR_COMPARISON_BUDGET), " *
                "|dwidth| <= $(WIDTH_COMPARISON_BUDGET).")
    end
    if allparts || :regrid in parts
        for subcycle in (false, true)
            j = regrid_history!(N, ny, moving_tfinal, nmax, subcycle)
            regrid_failure |= !j.complete || isempty(j.events) ||
                              j.cumulative_jump > REGRID_BUDGET ||
                              j.max_excursion > EVOLUTION_BUDGET ||
                              j.max_evolution > EVOLUTION_BUDGET
            if rank == 0
                @printf("regrid history, subcycle=%-5s changed=%d  cumulative jump %.3e: %s\n",
                        subcycle, length(j.events), j.cumulative_jump,
                        length(j.events) == 0 ? "UNEXERCISED" :
                        verdict(j.cumulative_jump, REGRID_BUDGET, j.complete))
                show_scale("regrid, subcycle=$(subcycle)", j.initial)
                @printf("  evolution history max %.3e: %s\n", j.max_excursion,
                        verdict(j.max_excursion, EVOLUTION_BUDGET, j.complete))
                @printf("  transfer-subtracted history max %.3e: %s\n", j.max_evolution,
                        verdict(j.max_evolution, EVOLUTION_BUDGET, j.complete))
                @printf("  species excursion max %.2e\n", j.species_excursion)
                show_drift("  moving final evolution", j.drift)
                show_drift("  transfer-subtracted residual", j.evolution_residual)
            end
        end
    end
    if args.check && rank == 0
        failures = any(r -> !r.complete || r.max_excursion > EVOLUTION_BUDGET,
                       values(results))
        (failures || regrid_failure || mixing_failure || initial_sampling_failure) &&
            error("budget exceeded, mixing comparison failed, or regrid was unexercised; " *
                  "see the report")
    end
end

mpi_main(main)
