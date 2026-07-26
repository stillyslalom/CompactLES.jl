# Argument parsing for the scripts in `bench/` and at the repo root.
#
# Not solver code. It lives here because three scripts had grown three copies of
# it, and because the alternative each of them started from — `get(ENV, ...)`
# scattered through the body — has a failure mode this does not: a mistyped
# environment variable runs the default for the length of the job and says
# nothing. `CL_TGV_PROGERSS` is a silent hour; `progerss=200` is a message.
#
# ARGS also puts the whole invocation in the batch script and in the job log,
# where an `export` three lines up does not. Two library-level knobs stay in the
# environment on purpose (`CL_THREAD_MIN_WORK`, `CL_ERROR_BACKTRACE`): those are
# read from inside the package, where there is no ARGS to consult.

"""
    script_args(args, defaults; positional = ())

Parse `args` — the leading bare values named by `positional`, then `key=value`
pairs in any order — into a `NamedTuple` carrying the same keys as `defaults`.

Each value is parsed to the type of its default, so `defaults` is both the
fallback and the schema. `String` defaults are taken verbatim, which is how a
script accepts its own compound syntax (`configs=on:1:0.002,off:1`) without this
function needing to know anything about it. `Bool` defaults accept
`true|false|yes|no|on|off|1|0`.

An unknown key, a surplus positional, or a value that will not parse throws
`ArgumentError`. Rejecting a typo is the whole point — see the note above.

```julia
opt = script_args(ARGS, (N = 32, tfinal = 10.0, progress = 0);
                  positional = (:N, :tfinal))
```
"""
function script_args(args, defaults::NamedTuple; positional::Tuple=())
    for name in positional
        haskey(defaults, name) ||
            throw(ArgumentError("positional :$name is not a key of defaults"))
    end
    parsed = Dict{Symbol,Any}(pairs(defaults))
    npos = 0
    for a in args
        i = findfirst('=', a)
        local key::Symbol, text::AbstractString
        if i === nothing
            npos += 1
            npos <= length(positional) ||
                throw(ArgumentError("unexpected positional argument '$a'; " *
                    (isempty(positional) ? "this script takes none" :
                     "this script takes " *
                     join(("<" * String(p) * ">" for p in positional), " "))))
            key, text = positional[npos], a
        else
            key = Symbol(a[1:prevind(a, i)])
            haskey(defaults, key) ||
                throw(ArgumentError("unknown option '$key', want one of: " *
                                    join(keys(defaults), ", ")))
            text = a[nextind(a, i):end]
        end
        parsed[key] = _arg_value(defaults[key], text, key)
    end
    return NamedTuple{keys(defaults)}(map(k -> parsed[k], keys(defaults)))
end

_arg_value(::AbstractString, text, key) = String(text)

# Ahead of the numeric method because Bool <: Integer, and "on"/"off" is what a
# flag reads like on a launch line.
function _arg_value(::Bool, text, key)
    lowercase(text) in ("1", "true", "yes", "on") && return true
    lowercase(text) in ("0", "false", "no", "off") && return false
    throw(ArgumentError("option '$key=$text' wants true|false"))
end

function _arg_value(default::Union{Integer,AbstractFloat}, text, key)
    value = tryparse(typeof(default), text)
    value === nothing &&
        throw(ArgumentError("option '$key=$text' wants a $(typeof(default))"))
    return value
end

"""
    script_grid(spec) -> NTuple{3,Int}

Parse a grid argument written either as `N` for a cubic grid or as `NX,NY,NZ`.
Shared by `clusterprobe.jl` and `clusterlaunch.jl`, which have to agree on it —
the launch planner prints a probe command containing the grid it just sized.
"""
function script_grid(spec::AbstractString)
    parts = tryparse.(Int, split(spec, ','))
    (any(isnothing, parts) || !(length(parts) in (1, 3))) &&
        throw(ArgumentError("bad grid '$spec', want N or NX,NY,NZ"))
    return length(parts) == 1 ? ntuple(_ -> parts[1]::Int, 3) :
           (parts[1]::Int, parts[2]::Int, parts[3]::Int)
end
