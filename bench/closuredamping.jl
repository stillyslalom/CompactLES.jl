# Fifth-order-compatible boundary damping, as a joint derivative/filter trial.
#
#   julia --project=. -t 1 bench/closuredamping.jl parts=spectrum
#   julia --project=. -t 1 bench/closuredamping.jl parts=smooth strength=0.3
#   julia --project=. -t 1 bench/closuredamping.jl parts=jacobian,uniform
#       schemes=candidate strength=0.1 components=acoustic
#
# A seven-point sixth-difference vector v defines I-sigma*v*v'/(v'v),
# applied at each wall after the ordinary filter. It preserves polynomials
# through degree five and is contractive in Euclidean norm for sigma in [0,1].
# Its O(h^6) per-step correction is an O(h^5) boundary RHS perturbation at
# fixed hyperbolic CFL. These properties alone do not prove stability of
# the complete non-normal update. No production filtering code is changed.
# components=all is rejected: its scalar entropy/tangential filter branch
# grows even when the acoustic block is neutral. components=acoustic filters
# pressure and normal velocity only, with a consistent energy adjustment;
# this prototype supports one calorically perfect gas on a serial 1-D line.
# Spectrum checks include the scalar branch as well as the acoustic block.

module ClosureDamping

using CompactLES, MPI, LinearAlgebra, Printf
const CL = CompactLES
include(joinpath(@__DIR__, "closuresearch.jl"))
include(joinpath(@__DIR__, "closurequalify.jl"))
const CQ = ClosureQualification
const DIFFERENCE = Float64[1,-6,15,-20,15,-6,1]
const DIFFERENCE_NORM = sum(abs2, DIFFERENCE)

function scheme(name)
    name == "bl" && return lele_d1_6(closures=:brady_livescu)
    name == "neutral3" && return lele_d1_6()
    name == "unfiltered" && return ClosureSearch.unfiltered_scheme()
    name == "candidate" && return ClosureSearch.candidate_scheme()
    name == "de" && return ClosureSearch.de_scheme()
    error("unknown scheme")
end

function damping_matrix(n, strength; blocks=1)
    n >= 2*(6+blocks) || error("wall damping supports must not overlap")
    result = Matrix{Float64}(I,n,n)
    for start in 1:blocks, side in (1,2)
        indices = side == 1 ? (start:start+6) : (n-start+1:-1:n-start-5)
        vector = zeros(n)
        vector[indices] .= DIFFERENCE
        result = (I - strength .* (vector*vector') ./ DIFFERENCE_NORM)*result
    end
    result
end

function step_matrix(deriv, n; filtered=true, cfl=0.5, filter_cfl=0.0)
    operator, _ = ClosureSearch.acoustic_operator(deriv,n)
    m = size(operator,1)
    result = Matrix{Float64}(I,m,m)
    accumulator = zeros(m,m)
    dt = cfl/(n-1)
    for stage in eachindex(CL.RKA)
        accumulator .= CL.RKA[stage].*accumulator .+ dt.*(operator*result)
        result .+= CL.RKB[stage].*accumulator
    end
    if filtered
        weight = filter_cfl == 0 ? 1.0 : min(cfl/filter_cfl,1.0)
        filter = ClosureSearch.filter_block(n,0.45)
        result = ((1-weight)*I + weight*filter)*result
    end
    result
end

function spectrum(opts)
    ns = parse.(Int,split(opts.ns,','))
    strengths = parse.(Float64,split(opts.strengths,','))
    for name in split(opts.schemes,','), filtered in (false,true)
        deriv = scheme(name)
        for n in ns
            step = step_matrix(deriv,n; filtered=filtered,
                cfl=opts.cfl,filter_cfl=opts.filter_cfl)
            weight = opts.filter_cfl == 0 ? 1.0 : min(opts.cfl/opts.filter_cfl,1.0)
            for strength in strengths
                F = damping_matrix(n,strength*weight; blocks=opts.blocks)
                Fblock = [F zeros(n,n-2); zeros(n-2,n) F[2:n-1,2:n-1]]
                acoustic = maximum(abs,eigvals(Fblock*step))
                scalar = filtered ?
                    (1-weight)*I + weight*ClosureSearch.filter_matrix(n) :
                    Matrix{Float64}(I,n,n)
                opts.components == "all" && (scalar = F*scalar)
                radius = max(acoustic,maximum(abs,eigvals(scalar)))
                @printf("%s N=%d filtered=%s damping=%.3f radius=%.12f\n",
                        name,n,filtered,strength,radius)
            end
            flush(stdout)
        end
    end
end

function sweep(opts)
    ns = unique([collect(opts.firstn:opts.lastn); parse.(Int,split(opts.ns,','))])
    for name in split(opts.schemes,',')
        deriv = scheme(name)
        worst = (radius=0.0,n=0)
        failures = 0
        for n in ns
            step = step_matrix(deriv,n; cfl=opts.cfl, filter_cfl=opts.filter_cfl)
            weight = opts.filter_cfl == 0 ? 1.0 : min(opts.cfl/opts.filter_cfl,1.0)
            F = damping_matrix(n,opts.strength*weight; blocks=opts.blocks)
            Fblock = [F zeros(n,n-2); zeros(n-2,n) F[2:n-1,2:n-1]]
            acoustic = maximum(abs,eigvals(Fblock*step))
            scalar = (1-weight)*I + weight*ClosureSearch.filter_matrix(n)
            opts.components == "all" && (scalar = F*scalar)
            radius = max(acoustic,maximum(abs,eigvals(scalar)))
            radius > worst.radius && (worst=(radius=radius,n=n))
            if radius > 1+1e-10
                failures += 1
                @printf("FAIL %s N=%d radius=%.12f\n",name,n,radius)
            end
            flush(stdout)
        end
        @printf("sweep %s tested=%d failures=%d maxradius=%.12f atN=%d\n",
                name,length(ns),failures,worst.radius,worst.n)
    end
end

function jacobian(opts)
    for name in split(opts.schemes,','), n in parse.(Int,split(opts.jns,','))
        deriv = scheme(name)
        for delta in (3e-6,1e-5,3e-5)
            matrix,dt = CQ.production_jacobian(deriv,n; filtered=true, delta=delta,
                poststep=(s,q) -> damp!(s,q,opts.strength;
                    blocks=opts.blocks,components=opts.components))
            radius = maximum(abs,eigvals(matrix))
            @printf("production %s N=%d delta=%.0e radius=%.12f rate=%+.6e\n",
                    name,n,delta,radius,log(radius)/dt)
            flush(stdout)
        end
    end
end

function uniform(opts)
    for name in split(opts.schemes,','), n in parse.(Int,split(opts.jns,','))
        solver,state,_ = CQ.uniform_solver(scheme(name),n; filtered=true)
        initial = copy(state)
        try
            run!(solver,state; tfinal=40.0,nmax=30000,
                callback=(s,q) -> damp!(s,q,opts.strength;
                    blocks=opts.blocks,components=opts.components))
            velocity = maximum(abs(state[gidx(solver,i,1,1),2] /
                                   state[gidx(solver,i,1,1),1]) for i in 1:n)
            drift = maximum(abs(state[gidx(solver,i,1,1),c] -
                initial[gidx(solver,i,1,1),c]) for i in 1:n for c in 1:5)
            @printf("uniform %s N=%d t=%.1f maxvelocity=%.6e drift=%.6e\n",
                    name,n,solver.t,velocity,drift)
        catch err
            err isa SolverFailure || rethrow()
            println("uniform ",name," FAILED ",err)
        end
        flush(stdout)
    end
end

function stress(opts)
    for name in split(opts.schemes,','), start in (0.0,0.1)
        prob = CQ.noh_problem(1; N=200,t0=start)
        solver,state = setup(prob,Numerics(n_global=(200,1,1),deriv=scheme(name),
            art=ArtParams(enabled=true),cfl=0.3,filter_interval=1,filter_cfl=0.35,
            control=StepControl(validity=:permissive)))
        try
            run!(solver,state; tfinal=CQ.NOH_T-start,nmax=30000,
                callback=(s,q) -> damp!(s,q,opts.strength;
                    blocks=opts.blocks,components=opts.components))
            println("Noh ",name," start=",start," completed=",
                CQ.completed(solver,CQ.NOH_T-start)," ",state_report(solver,state))
        catch err
            err isa SolverFailure || rethrow()
            println("Noh ",name," start=",start," FAILED ",err)
        end
        flush(stdout)
    end
end

function damp!(solver, state, strength; blocks=1,components="all")
    n = solver.n_global[1]
    solver.decomp.n_local[1] == n || error("serial 1-D trial only")
    n >= 2*(6+blocks) || error("wall damping supports must not overlap")
    sigma = strength * CL.filter_weight(solver,1)
    if components == "acoustic"
        acoustic_damp!(solver,state,sigma; blocks=blocks)
        apply_bcs!(solver,state)
        return nothing
    end
    for start in 1:blocks, side in (1,2), c in 1:solver.equations.n_cons
        nodes = side == 1 ? (start:start+6) : (n-start+1:-1:n-start-5)
        anchor = state[gidx(solver,first(nodes),1,1),c]
        projection = sum(DIFFERENCE[j]*(state[gidx(solver,i,1,1),c]-anchor)
                         for (j,i) in enumerate(nodes))
        for (j,i) in enumerate(nodes)
            state[gidx(solver,i,1,1),c] -= sigma*DIFFERENCE[j]*projection/DIFFERENCE_NORM
        end
    end
    apply_bcs!(solver,state)
    return nothing
end

function acoustic_damp!(solver,state,sigma; blocks=1)
    solver.eos isa IdealMixture && solver.equations.n_species == 1 ||
        error("acoustic damping prototype supports one calorically perfect gas")
    gamma = solver.eos.sp[1].gamma
    n = solver.n_global[1]
    ie = solver.equations.i_energy
    m1,m2,m3 = solver.equations.i_mom
    for start in 1:blocks, side in (1,2)
        nodes = side == 1 ? (start:start+6) : (n-start+1:-1:n-start-5)
        velocity = [state[gidx(solver,i,1,1),m1]/state[gidx(solver,i,1,1),1]
                    for i in nodes]
        pressure = map(nodes) do i
            index = gidx(solver,i,1,1)
            kinetic = sum(state[index,m]^2 for m in (m1,m2,m3))/(2state[index,1])
            (gamma-1)*(state[index,ie]-kinetic)
        end
        du = sum(DIFFERENCE .* (velocity .- first(velocity)))/DIFFERENCE_NORM
        dp = sum(DIFFERENCE .* (pressure .- first(pressure)))/DIFFERENCE_NORM
        for (j,i) in enumerate(nodes)
            index = gidx(solver,i,1,1)
            rho = state[index,1]
            oldmomentum = state[index,m1]
            newmomentum = rho*(velocity[j]-sigma*DIFFERENCE[j]*du)
            state[index,ie] += -sigma*DIFFERENCE[j]*dp/(gamma-1) +
                (newmomentum^2-oldmomentum^2)/(2rho)
            state[index,m1] = newmomentum
        end
    end
    return nothing
end

function smooth(opts)
    ns = parse.(Int,split(opts.smooth_ns,','))
    for name in split(opts.schemes,','), art in (false,true), cfl in (0.25,0.125)
        errors = Float64[]
        deriv = scheme(name)
        for n in ns
            settings = (deriv=deriv, cfl=cfl, filter_interval=1,
                        art=ArtParams(enabled=art))
            solver, state = CQ.wall_case(n; settings...)
            mirror, reference = CQ.mirror_case(n; settings...)
            try
                run!(solver,state; tfinal=0.4,nmax=30000,
                    callback=(s,q) -> damp!(s,q,opts.strength;
                        blocks=opts.blocks,components=opts.components))
                run!(mirror,reference; tfinal=0.4,nmax=30000)
                e = CQ.regional_errors(solver,state,CQ.NodeReference(mirror,reference))
                push!(errors,e.wall)
                @printf("smooth %s N=%d art=%s cfl=%.3f wall=%.6e interior=%.6e\n",
                        name,n,art,cfl,e.wall,e.interior)
            catch err
                err isa SolverFailure || rethrow()
                println("smooth ",name," FAILED ",err)
                break
            end
        end
        length(errors) == length(ns) && println("orders ",
            CQ.successive_orders(1.0 ./ (ns .- 1),errors))
        flush(stdout)
    end
end

function main(args=ARGS)
    MPI.Initialized() || MPI.Init(threadlevel=:funneled)
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error("serial trial only")
    BLAS.set_num_threads(1)
    opts = CL.script_args(args, (parts="spectrum", schemes="bl,unfiltered,candidate",
        ns="17,31,51,79,101,171", strengths="0,0.01,0.03,0.1,0.3,0.6,1",
        strength=0.3, blocks=1, smooth_ns="49,97,193", firstn=14,lastn=200,
        cfl=0.5,filter_cfl=0.35,jns="51,101",components="all"))
    opts.components in ("all","acoustic") || error("unknown damping components")
    0 <= opts.strength <= 1 || error("strength must lie in [0,1]")
    opts.blocks >= 1 || error("blocks must be positive")
    for part in split(opts.parts,',')
        part == "spectrum" ? spectrum(opts) :
        part == "sweep" ? sweep(opts) :
        part == "jacobian" ? jacobian(opts) :
        part == "uniform" ? uniform(opts) :
        part == "stress" ? stress(opts) :
        part == "smooth" ? smooth(opts) : error("unknown part")
    end
end

end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && ClosureDamping.main()
