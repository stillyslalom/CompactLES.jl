# Script reference guard. The test suite does not include the scripts under
# bench/, examples/, probes/ and docs/literate/, so an API change that removes
# or renames a binding or a keyword leaves them broken with nothing to report
# it. This file parses every script without running it, since the scripts load
# JET, CairoMakie, AMDGPU, HDF5 and other packages the package environment
# does not carry, and checks its references to CompactLES against the loaded
# package:
#   1. every qualified reference `CompactLES.name`, through an alias bound by
#      `const CL = CompactLES` or `import CompactLES as CL` and through the
#      submodules (`CompactLES.Regions.Sphere`), names a binding that exists;
#   2. every name a `using` or `import` statement takes from CompactLES or a
#      submodule exists;
#   3. every keyword in a call whose callee resolves statically to a function
#      or constructor of CompactLES is accepted by some method of it. A callee
#      resolves when it is qualified, or when it is an unqualified name the
#      script brings in from CompactLES and does not bind itself.
#
# A reference that cannot be resolved with certainty is left unchecked rather
# than reported, so that every failure is a real breakage. A name guarded by
# `isdefined` or `hasproperty` anywhere in a script is not checked, an alias
# bound more than once is not followed, unqualified callees are not checked in
# a module whose `include` paths do not resolve statically, and a callable
# that a script extends with methods of its own is not checked for keywords.
#
# A method that collects keywords as `kw...` accepts any keyword at its own
# signature. Such a callable is checked only when the destination of those
# keywords is known: a method the compiler generated to forward keywords to
# another method of the same callable, for which `kwarg_decl` gives `...`, or
# an entry of `FORWARDS` below. Any other callable with such a method, the
# HDF5 and Makie stubs among them, is not checked for keywords.
#
# A script that parses and resolves can still compute the wrong thing; this
# file does not replace a small representative run of a script after an API
# change.
#
# Standalone: julia --project=. test/scriptrefs_tests.jl
# `ScriptRefs.check(paths)` checks other files or directories.

if !isdefined(Main, :CompactLES)
    using CompactLES
end
using Test

module ScriptRefs

using CompactLES

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SCRIPT_DIRS = [joinpath(ROOT, d) for d in
                     ("bench", "examples", "probes", joinpath("docs", "literate"))]

# The destinations of a callable's `kw...` keywords, where a method of it
# collects them. Each entry names the callable and what accepts the collected
# keywords: `:fields`, the copy constructor `_with`, which takes the type's
# field names; `:self`, another method of the same callable, so that the
# keywords of its other methods are the accepted set; or the name of an inner
# function, whose explicit keywords are accepted (`_numerics` collects the
# rest as `unknown...` only to reject them).
const FORWARDS = Dict{Symbol,Vector{Symbol}}(
    :Numerics => [:_numerics],
    :Solver => [:_Solver],
    :StateFilter => [:fields],
    :PatchInterfaces => [:fields],
    :ArtificialProperties => [:fields],
    :StepControl => [:fields],
    :AMR => [:fields],
    # `Execution(base; storage, kw...)` copies `base` with `kw`, and rejects
    # the `backend` and `precision` fields; the keywords of the other method
    # are exactly the accepted ones.
    :Execution => [:self],
    :IdealMixture => [:self],
    :Nasa9Species => [:self],
    :Nasa9Mixture => [:self, :_nasa9_mixture],
    :NSCBCInflowBC => [:self],
    :Decomp => [:self],
    :read_nasa9 => [:self],
    :read_ionmix => [:self],
    :read_sesame => [:self],
    :run! => [:self],
)

"""
    Failure(file, line, name, message)

One broken reference: the script, the line of the statement holding it, the
missing name or keyword, and a sentence describing it.
"""
struct Failure
    file::String
    line::Int
    name::String
    message::String
end

Base.show(io::IO, f::Failure) = print(io, f.file, ":", f.line, ": ", f.message)

root_module(m::Module) = (p = parentmodule(m); p === m ? m : root_module(p))
in_package(m::Module) = root_module(m) === CompactLES
owner(v) = v isa Module ? v : v isa Function || v isa Type ? parentmodule(v) : nothing
function owned(v)
    m = v isa UnionAll || v isa DataType || v isa Function ? owner(v) : nothing
    return m !== nothing && in_package(m)
end

display_path(file) =
    startswith(file, ROOT) ? replace(relpath(file, ROOT), '\\' => '/') : file

# --- Scopes -----------------------------------------------------------------

# What a module of a script binds, collected over its body and the files it
# includes: aliases of CompactLES modules and how many assignments made each,
# names imported from CompactLES, the CompactLES modules it uses whole, every
# other name it binds and how often, the names it tests with `isdefined`, and
# the CompactLES callables it adds methods to. An import from CompactLES is
# not counted as a binding, so that a script and a file it includes may both
# `using CompactLES`.
mutable struct Scope
    units::Vector{Tuple{String,Vector{Any}}}
    aliases::Dict{Symbol,Module}
    alias_count::Dict{Symbol,Int}
    conflicts::Set{Symbol}
    imported::Dict{Symbol,Any}
    used::Vector{Module}
    bound::Dict{Symbol,Int}
    guarded::Set{Symbol}
    extended::Set{Symbol}
    complete::Bool
    nested::Vector{Tuple{String,Expr}}
end
Scope() = Scope(Tuple{String,Vector{Any}}[], Dict{Symbol,Module}(), Dict{Symbol,Int}(),
                Set{Symbol}(), Dict{Symbol,Any}(), Module[], Dict{Symbol,Int}(),
                Set{Symbol}(), Set{Symbol}(), true, Tuple{String,Expr}[])

function alias!(scope, name, m::Module; assigned::Bool)
    get(scope.aliases, name, m) === m || push!(scope.conflicts, name)
    scope.aliases[name] = m
    assigned && (scope.alias_count[name] = get(scope.alias_count, name, 0) + 1)
end

const PARSED = Dict{String,Any}()

function parse_file(file)
    get!(PARSED, file) do
        Meta.parseall(read(file, String); filename=file)
    end
end

statements(ex) = ex isa Expr && ex.head in (:toplevel, :block) ? ex.args : Any[ex]

parse_error(ex) = ex isa Expr && ex.head in (:error, :incomplete)

bind!(scope, s::Symbol) = (scope.bound[s] = get(scope.bound, s, 0) + 1; nothing)
bind!(scope, s) = nothing

# The names an assignment target binds: `a`, `(a, b)`, `(; a, b)`, `a::T`,
# `a...`. A field or index target (`x.f`, `x[i]`) binds nothing.
function bind_target!(scope, lhs)
    if lhs isa Symbol
        bind!(scope, lhs)
    elseif lhs isa Expr
        h = lhs.head
        if h in (:tuple, :parameters)
            foreach(a -> bind_target!(scope, a), lhs.args)
        elseif h in (:(::), :..., :kw) && !isempty(lhs.args)
            bind_target!(scope, lhs.args[1])
        elseif h === :(=)
            bind_target!(scope, lhs.args[1])
        end
    end
end

# A function signature, with any return type and `where` clause removed.
function signature_call(sig)
    while sig isa Expr && sig.head in (:where, :(::)) && length(sig.args) >= 1 &&
          !(sig.head === :(::) && length(sig.args) == 1)
        sig = sig.args[1]
    end
    return sig
end

is_definition_target(lhs) = signature_call(lhs) isa Expr &&
                            signature_call(lhs).head === :call

# The name a function or constructor definition binds, and its arguments.
function bind_signature!(scope, sig)
    call = signature_call(sig)
    if call isa Symbol
        bind!(scope, call)        # `function f end`
        return
    end
    call isa Expr || return
    if call.head === :tuple       # an anonymous `function (x) ... end`
        foreach(a -> bind_target!(scope, a), call.args)
        return
    end
    call.head === :call || return
    name = call.args[1]
    name isa Expr && name.head === :curly && (name = name.args[1])
    if name isa Symbol
        bind!(scope, name)
        haskey(scope.imported, name) && push!(scope.extended, name)
    elseif name isa Expr && name.head === :.
        # A method added to a CompactLES callable, `CL.f(x) = ...`.
        v = resolve(scope, name)
        v isa Resolved && owned(v.value) && push!(scope.extended, last_name(name))
    end
    foreach(a -> bind_target!(scope, a), call.args[2:end])
end

last_name(ex) = ex isa Expr && ex.head === :. && ex.args[2] isa QuoteNode ?
                ex.args[2].value : ex isa Symbol ? ex : nothing

# --- Resolution -------------------------------------------------------------

struct Resolved
    value::Any
end
struct Absent
    parent::Module
    name::Symbol
end

# The binding a reference names, when it names one of CompactLES: a
# `Resolved` value, a `Absent` name of a CompactLES module that does not
# bind it, or `nothing` when the reference is not to CompactLES or cannot be
# followed statically (a property of a value, an interpolated name).
function resolve(scope, ex)
    if ex isa Symbol
        haskey(scope.aliases, ex) && return Resolved(scope.aliases[ex])
        ex === :CompactLES && get(scope.bound, ex, 0) == 0 && return Resolved(CompactLES)
        return nothing
    end
    ex isa Expr && ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode ||
        return nothing
    name = ex.args[2].value
    name isa Symbol || return nothing
    base = resolve(scope, ex.args[1])
    base isa Resolved || return base isa Absent ? base : nothing
    m = base.value
    m isa Module && in_package(m) || return nothing
    name in scope.guarded && return nothing
    isdefined(m, name) || return Absent(m, name)
    return Resolved(getfield(m, name))
end

# The module path of a `using`/`import` operand, `CompactLES.Regions` as
# `[:CompactLES, :Regions]`; a relative path (`.Local`) gives `nothing`.
function import_path(ex)
    ex isa Expr && ex.head === :. || return nothing
    path = ex.args
    isempty(path) && return nothing
    path[1] === :. && return nothing
    all(a -> a isa Symbol, path) || return nothing
    return Symbol[path...]
end

# Follow an import path from CompactLES, returning the bound value, a
# `Absent`, or `nothing` for a path outside the package.
function follow(scope, path)
    first(path) === :CompactLES || haskey(scope.aliases, first(path)) || return nothing
    v = first(path) === :CompactLES ? CompactLES : scope.aliases[first(path)]
    for name in path[2:end]
        v isa Module && in_package(v) || return nothing
        name in scope.guarded && return nothing
        isdefined(v, name) || return Absent(v, name)
        v = getfield(v, name)
    end
    return Resolved(v)
end

# --- Collecting a module's bindings -------------------------------------------

# The path an `include` argument names, when it is built from string literals,
# `@__DIR__`, `@__FILE__`, `joinpath` and `normpath` alone.
function static_path(ex, file)
    ex isa String && return ex
    if ex isa Expr
        if ex.head === :string && all(a -> a isa String, ex.args)
            return join(ex.args)
        elseif ex.head === :macrocall
            ex.args[1] === Symbol("@__DIR__") && return dirname(file)
            ex.args[1] === Symbol("@__FILE__") && return file
        elseif ex.head === :call && ex.args[1] in (:joinpath, :normpath, :abspath)
            parts = [static_path(a, file) for a in ex.args[2:end]]
            any(isnothing, parts) && return nothing
            return normpath(joinpath(parts...))
        end
    end
    return nothing
end

function collect_imports!(scope, ex)
    is_using = ex.head === :using
    for arg in ex.args
        if arg isa Expr && arg.head === :(:)
            path = import_path(arg.args[1])
            for item in arg.args[2:end]
                as = nothing
                if item isa Expr && item.head === :as
                    item, as = item.args
                end
                name = item isa Expr && item.head === :. ? item.args[end] : nothing
                name isa Symbol || continue
                local_name = as === nothing ? name : as
                r = path === nothing ? nothing : follow(scope, [path; name])
                if r === nothing
                    bind!(scope, local_name)      # a name of another package
                elseif r isa Resolved
                    scope.imported[local_name] = r.value
                    r.value isa Module && in_package(r.value) &&
                        alias!(scope, local_name, r.value; assigned=false)
                end
            end
        else
            as = nothing
            if arg isa Expr && arg.head === :as
                arg, as = arg.args
            end
            path = import_path(arg)
            path === nothing && continue
            local_name = as === nothing ? path[end] : as
            r = follow(scope, path)
            if r === nothing
                bind!(scope, local_name)          # another package
                continue
            end
            r isa Resolved || continue
            v = r.value
            if v isa Module && in_package(v)
                alias!(scope, local_name, v; assigned=false)
                is_using && !(v in scope.used) && push!(scope.used, v)
            elseif !is_using
                scope.imported[local_name] = v
            end
        end
    end
end

# The first pass over a module: every binding, import and include, in source
# order. A nested module is set aside to be scanned as a scope of its own.
function collect!(scope, ex, file, seen)
    ex isa Expr || return
    h = ex.head
    if h === :module
        bind!(scope, ex.args[2])
        push!(scope.nested, (file, ex))
        return
    elseif h in (:using, :import)
        collect_imports!(scope, ex)
        return
    elseif h === :function || h === :macro
        sig = ex.args[1]
        if h === :macro && signature_call(sig) isa Expr
            name = signature_call(sig).args[1]
            name isa Symbol && bind!(scope, Symbol("@", name))
            foreach(a -> bind_target!(scope, a), signature_call(sig).args[2:end])
        else
            bind_signature!(scope, sig)
        end
    elseif h === :(=) && is_definition_target(ex.args[1])
        bind_signature!(scope, ex.args[1])
    elseif h === :(=)
        lhs, rhs = ex.args
        bind_target!(scope, lhs)
        if lhs isa Symbol
            r = resolve(scope, rhs)
            r isa Resolved && r.value isa Module && in_package(r.value) &&
                alias!(scope, lhs, r.value; assigned=true)
        end
    elseif h === :->
        bind_target!(scope, ex.args[1])
    elseif h in (:struct, :abstract, :primitive)
        name = ex.args[h === :struct ? 2 : 1]
        name isa Expr && name.head === :<: && (name = name.args[1])
        name isa Expr && name.head === :curly && (name = name.args[1])
        bind!(scope, name)
    elseif h in (:local, :global)
        foreach(a -> a isa Symbol && bind!(scope, a), ex.args)
    elseif h === :try && length(ex.args) >= 2 && ex.args[2] isa Symbol
        bind!(scope, ex.args[2])
    elseif string(h)[end] == '=' && h ∉ (:(==), :(===), :(<=), :(>=), :(!=), :kw) &&
           !isempty(ex.args)
        bind_target!(scope, ex.args[1])       # `x += 1`
    elseif h === :tuple
        # A named tuple's `a = 1` binds nothing.
        for a in ex.args
            a isa Expr && a.head === :(=) ? collect!(scope, a.args[2], file, seen) :
                                            collect!(scope, a, file, seen)
        end
        return
    elseif h === :call
        f = ex.args[1]
        if f === :include || f === :includet
            path = length(ex.args) >= 2 ? static_path(ex.args[end], file) : nothing
            if path === nothing
                scope.complete = false
            else
                path = isabspath(path) ? path : normpath(joinpath(dirname(file), path))
                if !isfile(path)
                    scope.complete = false
                elseif !(path in seen)
                    push!(seen, path)
                    parsed = parse_file(path)
                    push!(scope.units, (path, statements(parsed)))
                    foreach(s -> collect!(scope, s, path, seen), statements(parsed))
                end
            end
        elseif f in (:isdefined, :hasproperty) && length(ex.args) >= 3 &&
               ex.args[3] isa QuoteNode && ex.args[3].value isa Symbol
            push!(scope.guarded, ex.args[3].value)
        end
    elseif h === :macrocall && ex.args[1] === Symbol("@isdefined")
        for a in ex.args[2:end]
            a isa Symbol && push!(scope.guarded, a)
        end
    end
    for a in ex.args
        collect!(scope, a, file, seen)
    end
end

function scan_scope(file, body)
    scope = Scope()
    push!(scope.units, (file, body))
    seen = Set{String}([file])
    foreach(s -> collect!(scope, s, file, seen), body)
    # An alias rebound anywhere in the module, as a local, an argument or a loop
    # variable, may name something else where it is used.
    for (name, _) in collect(scope.aliases)
        (name in scope.conflicts ||
         get(scope.bound, name, 0) > get(scope.alias_count, name, 0)) &&
            delete!(scope.aliases, name)
    end
    return scope
end

# --- Keywords ---------------------------------------------------------------

function callable_methods(f)
    ms = if f isa UnionAll && (u = Base.unwrap_unionall(f); u isa DataType) &&
            !isabstracttype(u)
        # A parametric type's constructors include the `T{P}(...)` methods,
        # which `methods(T)` does not list.
        [x.method for x in Base._methods_by_ftype(Tuple{Type{<:f},Vararg{Any}}, -1,
                                                  Base.get_world_counter())]
    else
        collect(methods(f))
    end
    return filter(m -> in_package(m.module), ms)
end

const ACCEPTED = IdDict{Any,Union{Nothing,Set{Symbol}}}()

# The keywords some method of `f` accepts, or `nothing` when a method of it
# collects keywords whose destination is not known.
accepted_keywords(f) = get!(() -> _accepted_keywords(f), ACCEPTED, f)

function _accepted_keywords(f)
    ms = callable_methods(f)
    isempty(ms) && return nothing
    names = Set{Symbol}()
    open = false
    for m in ms
        for k in Base.kwarg_decl(m)
            s = String(k)
            if s == "..."
                continue        # forwards to another method of `f`
            elseif endswith(s, "...")
                open = true
            else
                push!(names, k)
            end
        end
    end
    open || return names
    targets = get(FORWARDS, nameof(f), nothing)
    targets === nothing && return nothing
    getfield(parentmodule(f), nameof(f)) === f || return nothing
    for t in targets
        if t === :fields
            union!(names, fieldnames(Base.unwrap_unionall(f)))
        elseif t !== :self
            g = getfield(CompactLES, t)
            for m in callable_methods(g), k in Base.kwarg_decl(m)
                endswith(String(k), "...") || push!(names, k)
            end
        end
    end
    return names
end

function keyword_names(call)
    out = Symbol[]
    for a in call.args[2:end]
        if a isa Expr && a.head === :kw && a.args[1] isa Symbol
            push!(out, a.args[1])
        elseif a isa Expr && a.head === :parameters
            for p in a.args
                if p isa Symbol
                    push!(out, p)
                elseif p isa Expr && p.head in (:kw, :(=)) && p.args[1] isa Symbol
                    push!(out, p.args[1])
                elseif p isa Expr && p.head === :. && p.args[2] isa QuoteNode
                    push!(out, p.args[2].value)
                end
            end
        end
    end
    return out
end

# --- Checking ---------------------------------------------------------------

mutable struct Checker
    scope::Scope
    file::String
    line::Int
    report::Bool
    failures::Vector{Failure}
    stats::Dict{Symbol,Int}
    skipped::Set{String}
end

function fail!(c::Checker, name, message)
    c.report && push!(c.failures, Failure(display_path(c.file), c.line, string(name),
                                          message))
end

describe(ex) = string(ex)

function check_reference!(c::Checker, ex)
    r = resolve(c.scope, ex)
    r === nothing && return false
    c.stats[:references] += 1
    if r isa Absent
        fail!(c, r.name, "`$(describe(ex))`: $(r.parent) has no binding `$(r.name)`")
    end
    return true
end

function check_import!(c::Checker, ex)
    for arg in ex.args
        if arg isa Expr && arg.head === :(:)
            path = import_path(arg.args[1])
            path === nothing && continue
            base = follow(c.scope, path)
            base === nothing && continue
            c.stats[:imports] += 1
            if base isa Absent
                fail!(c, base.name, "`$(join(path, '.'))`: $(base.parent) has no " *
                                    "binding `$(base.name)`")
                continue
            end
            for item in arg.args[2:end]
                item isa Expr && item.head === :as && (item = item.args[1])
                name = item isa Expr && item.head === :. ? item.args[end] : nothing
                name isa Symbol || continue
                c.stats[:imports] += 1
                r = follow(c.scope, [path; name])
                r isa Absent && fail!(c, name, "imports `$name` from " *
                                                  "$(r.parent), which has no such binding")
            end
        else
            arg isa Expr && arg.head === :as && (arg = arg.args[1])
            path = import_path(arg)
            path === nothing && continue
            r = follow(c.scope, path)
            r === nothing && continue
            c.stats[:imports] += 1
            r isa Absent && fail!(c, r.name, "`$(ex.head) $(join(path, '.'))`: " *
                                               "$(r.parent) has no binding `$(r.name)`")
        end
    end
end

# The CompactLES callable a call's callee names, or `nothing`.
function callee_value(c::Checker, f)
    f isa Expr && f.head === :curly && (f = f.args[1])
    scope = c.scope
    if f isa Symbol
        scope.complete || return nothing
        get(scope.bound, f, 0) == 0 || return nothing
        f in scope.guarded && return nothing
        haskey(scope.imported, f) && return scope.imported[f]
        for m in scope.used
            Base.isexported(m, f) && isdefined(m, f) && return getfield(m, f)
        end
        return nothing
    end
    r = resolve(scope, f)
    r isa Resolved || return nothing
    last_name(f) in scope.extended && return nothing
    return r.value
end

function check_call!(c::Checker, ex)
    keywords = keyword_names(ex)
    isempty(keywords) && return
    v = callee_value(c, ex.args[1])
    v === nothing && return
    owned(v) || return
    nameof(v) in c.scope.extended && return
    accepted = accepted_keywords(v)
    if accepted === nothing
        c.stats[:skipped] += 1
        push!(c.skipped, string(nameof(v)))
        return
    end
    c.stats[:calls] += 1
    for k in keywords
        k in accepted || fail!(c, k, "`$(nameof(v))` accepts no keyword `$k`")
    end
end

# The default values and type annotations of a signature, without treating the
# signature itself as a call.
function visit_signature!(c::Checker, sig)
    sig isa Expr || return
    if sig.head === :kw
        visit!(c, sig.args[2])
    elseif sig.head === :(::)
        length(sig.args) == 2 ? visit!(c, sig.args[2]) : visit!(c, sig.args[1])
    elseif sig.head in (:call, :where, :tuple, :parameters, :...)
        foreach(a -> visit_signature!(c, a), sig.head === :call ? sig.args[2:end] :
                                                                  sig.args)
    end
end

function visit!(c::Checker, ex)
    if ex isa LineNumberNode
        c.line = ex.line
        return
    end
    ex isa Expr || return
    h = ex.head
    if h === :module
        return
    elseif h in (:using, :import)
        check_import!(c, ex)
        return
    elseif h === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        check_reference!(c, ex) || visit!(c, ex.args[1])
        return
    elseif h in (:function, :macro) ||
           (h === :(=) && is_definition_target(ex.args[1])) || h === :->
        visit_signature!(c, ex.args[1])
        foreach(a -> visit!(c, a), ex.args[2:end])
        return
    elseif h === :kw
        visit!(c, ex.args[2])
        return
    elseif h === :call
        check_call!(c, ex)
    end
    foreach(a -> visit!(c, a), ex.args)
end

function check_scope!(failures, stats, skipped, scope, targets)
    for (file, body) in scope.units
        c = Checker(scope, file, 0, file in targets, failures, stats, skipped)
        foreach(s -> visit!(c, s), body)
    end
    for (file, mod) in scope.nested
        inner = scan_scope(file, statements(mod.args[3]))
        check_scope!(failures, stats, skipped, inner, targets)
    end
end

function script_files(paths)
    files = String[]
    for p in paths
        p = normpath(abspath(p))
        if isdir(p)
            for (root, _, names) in walkdir(p), n in names
                endswith(n, ".jl") && push!(files, joinpath(root, n))
            end
        else
            push!(files, p)
        end
    end
    return sort!(unique!(files))
end

"""
    check(paths = SCRIPT_DIRS) -> (failures, stats, skipped)

Check every `.jl` file in `paths`, files or directories searched recursively.
`failures` is a vector of `Failure`, `stats` counts the references, imports
and keyword calls checked and the keyword calls not checked, and `skipped`
names the callables whose keywords were not checked.
"""
function check(paths=SCRIPT_DIRS)
    files = script_files(paths)
    targets = Set(files)
    failures = Failure[]
    stats = Dict(:files => length(files), :references => 0, :imports => 0,
                 :calls => 0, :skipped => 0)
    skipped = Set{String}()
    for file in files
        parsed = parse_file(file)
        body = statements(parsed)
        bad = findfirst(parse_error, body)
        if bad !== nothing
            push!(failures, Failure(display_path(file), 0, "", "does not parse: " *
                                    string(body[bad])))
            continue
        end
        check_scope!(failures, stats, skipped, scan_scope(file, body), targets)
    end
    unique!(f -> (f.file, f.line, f.message), failures)
    return failures, stats, skipped
end

end # module ScriptRefs

# Breakages recorded rather than repaired, as (script, missing name or
# keyword). Each is a `@test_broken`, so that repairing the script without
# removing its entry fails the suite.
const SCRIPT_BREAKAGES = Tuple{String,String}[
]

@testset "script references resolve" begin
    failures, stats, _ = ScriptRefs.check()
    known(f) = (f.file, f.name) in SCRIPT_BREAKAGES
    for (file, name) in SCRIPT_BREAKAGES
        @test_broken !any(f -> f.file == file && f.name == name, failures)
    end
    unexpected = filter(!known, failures)
    for f in unexpected
        @warn "broken script reference: $f"
    end
    @test isempty(unexpected)
    @info "script references: $(stats[:files]) files, $(stats[:references]) " *
          "qualified references, $(stats[:imports]) imported names, " *
          "$(stats[:calls]) keyword calls checked, $(stats[:skipped]) not checked"
end
