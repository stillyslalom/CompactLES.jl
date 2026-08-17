# Concise displays for the objects users commonly leave as the last expression
# in the REPL. In particular, setup returns (solver, Q) and run! returns Q, so
# the two-argument methods must remain short when called from tuple display.

function _show_dimensions(io::IO, dims)
    for (i, n) in enumerate(dims)
        i > 1 && print(io, " × ")
        print(io, n)
    end
end

_type_name(x) = nameof(typeof(x))

function Base.show(io::IO, Q::ConservedState)
    print(io, "ConservedState{")
    show(io, eltype(Q))
    print(io, "}(")
    _show_dimensions(io, size(Q))
    print(io, ')')
end

Base.show(io::IO, ::MIME"text/plain", Q::ConservedState) = show(io, Q)

function Base.show(io::IO, solver::Solver)
    print(io, "Solver(grid=")
    _show_dimensions(io, solver.n_global)
    npatches(solver) == 1 || print(io, ", patches=", length(solver.patch_regions))
    print(io, ", t=", solver.t, ", step=", solver.step,
          ", cfl=", solver.cfl, ", metric=", _type_name(solver.metric), ')')
end

function Base.show(io::IO, ::MIME"text/plain", solver::Solver)
    println(io, "Solver")
    print(io, "  grid: ")
    _show_dimensions(io, solver.n_global)
    if npatches(solver) == 1
        print(io, " (local ")
        _show_dimensions(io, solver.decomp.n_local)
        println(io, ')')
        print(io, "  process grid: ")
        _show_dimensions(io, solver.decomp.dims)
        println(io, " (rank ", MPI.Comm_rank(solver.comm), ')')
    else
        println(io, " (", length(solver.patch_regions), " patches, ",
                npatches(solver), " on this rank)")
    end
    println(io, "  equations: ", _type_name(solver.equations), " (",
            solver.equations.n_species, " species, ",
            solver.equations.n_cons, " conserved variables)")
    println(io, "  metric: ", _type_name(solver.metric))
    println(io, "  time: ", solver.t, " (step ", solver.step, ')')
    print(io, "  CFL: ", solver.cfl)
end

function Base.show(io::IO, prob::Problem)
    print(io, "Problem(")
    show(io, prob.name)
    print(io, ", domain=")
    show(io, prob.domain)
    print(io, ", metric=", _type_name(prob.metric),
          ", eos=", _type_name(prob.eos), ')')
end

function Base.show(io::IO, ::MIME"text/plain", prob::Problem)
    print(io, "Problem ")
    show(io, prob.name)
    print(io, "\n  domain: ")
    for d in 1:3
        d > 1 && print(io, " × ")
        show(io, prob.domain[d])
    end
    println(io, "\n  metric: ", _type_name(prob.metric))
    println(io, "  EOS: ", _type_name(prob.eos), " (", nspecies(prob.eos),
            nspecies(prob.eos) == 1 ? " species)" : " species)")
    println(io, "  boundary conditions: ",
            join((string(_type_name(prob.bcs[d][1]), " / ",
                         _type_name(prob.bcs[d][2])) for d in 1:3), ", "))
    print(io, "  sources: ", isempty(prob.sources) ? "none" : length(prob.sources))
end

function Base.show(io::IO, num::Numerics)
    print(io, "Numerics(grid=")
    _show_dimensions(io, num.n_global)
    print(io, ", deriv=", _type_name(num.deriv), ", filter=", _type_name(num.filt),
          ", cfl=", num.cfl, ", halo=", num.n_halo, ')')
end

function Base.show(io::IO, ::MIME"text/plain", num::Numerics)
    println(io, "Numerics")
    print(io, "  grid: ")
    _show_dimensions(io, num.n_global)
    println(io)
    println(io, "  derivative: ", _type_name(num.deriv))
    println(io, "  filter: ", _type_name(num.filt), " (every ",
            num.filter_interval, num.filter_interval == 1 ? " step" : " steps",
            num.filter_cfl > 0 ? ", relaxed to cfl $(num.filter_cfl))" : ")")
    println(io, "  artificial properties: ", num.art.enabled ? "enabled" : "disabled")
    println(io, "  CFL: ", num.cfl)
    print(io, "  process grid: ")
    num.dims === nothing ? print(io, "automatic") : _show_dimensions(io, num.dims)
    print(io, "\n  halo width: ", num.n_halo)
end
