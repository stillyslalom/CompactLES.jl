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

# --- Equation-of-state objects ----------------------------------------------
# A NASA-9 record carries every polynomial coefficient of every fit interval,
# so the default display of a two-species mixture is a dozen lines of numbers
# no one reads at the REPL. These methods print the data that identifies the
# model instead: species names, gas constants, fit ranges, and the
# temperature guess. The two-argument forms stay short because a `Vector` of
# species or a `Problem` prints its elements through them.

_sig(x) = round(x; sigdigits=7)

function _species_table(io::IO, names, rows)
    w = maximum(length, names)
    for (k, (name, row)) in enumerate(zip(names, rows))
        k > 1 && println(io)
        print(io, "  ", rpad(name, w), "  ", row)
    end
end

Base.show(io::IO, sp::IdealSpecies) =
    print(io, "IdealSpecies(", repr(sp.name), "; R=", _sig(sp.R),
          ", gamma=", _sig(sp.gamma), ')')

Base.show(io::IO, eos::IdealMixture{T}) where {T} =
    print(io, "IdealMixture{", T, "}(", join((sp.name for sp in eos.sp), ", "), ')')

function Base.show(io::IO, ::MIME"text/plain", eos::IdealMixture{T}) where {T}
    n = length(eos.sp)
    println(io, "IdealMixture{", T, "} with ", n, n == 1 ? " species" : " species")
    _species_table(io, [sp.name for sp in eos.sp],
                   ["R = $(_sig(sp.R))  gamma = $(_sig(sp.gamma))" for sp in eos.sp])
end

_interval_range(sp::Nasa9Species) =
    string(_sig(sp.intervals[1].Tmin), "–", _sig(sp.intervals[end].Tmax), " K")

Base.show(io::IO, iv::Nasa9Interval{T}) where {T} =
    print(io, "Nasa9Interval{", T, "}(", _sig(iv.Tmin), "–", _sig(iv.Tmax), " K)")

function Base.show(io::IO, sp::Nasa9Species)
    n = length(sp.intervals)
    print(io, "Nasa9Species(", repr(sp.name), "; R=", _sig(sp.R), ", ",
          _interval_range(sp), ", ", n, n == 1 ? " interval)" : " intervals)")
end

Base.show(io::IO, eos::Nasa9Mixture{T}) where {T} =
    print(io, "Nasa9Mixture{", T, "}(", join((sp.name for sp in eos.sp), ", "),
          "; T_guess=", _sig(eos.T_guess), ')')

function Base.show(io::IO, ::MIME"text/plain", eos::Nasa9Mixture{T}) where {T}
    n = length(eos.sp)
    println(io, "Nasa9Mixture{", T, "} with ", n, n == 1 ? " species" : " species",
            " (T_guess = ", _sig(eos.T_guess), ")")
    _species_table(io, [sp.name for sp in eos.sp],
                   [string("R = ", _sig(sp.R), "  ", _interval_range(sp), " (",
                           length(sp.intervals),
                           length(sp.intervals) == 1 ? " interval)" : " intervals)")
                    for sp in eos.sp])
end

Base.show(io::IO, eos::StiffenedGas) =
    print(io, "StiffenedGas(gamma=", _sig(eos.gamma), ", p_inf=", _sig(eos.p_inf),
          ", cv=", _sig(eos.cv), ", name=", repr(eos.name), ')')
