# The positivity limiter's records, ahead of `Solver`, which holds them. The
# construction is in positivity_setup.jl and the passes in positivity_step.jl.

# The weights and stencils a limiter's bodies read, in the fields' storage.
struct LimiterWeights{T,A<:AbstractArray{T,3},V<:AbstractVector{T},
                      B<:AbstractVector{Bool}}
    weights::NTuple{3,V}            # W_d at each padded position along d
    inv_weights::NTuple{3,V}        # 1 / W_d, which the face pass multiplies by
    free::Tuple{B,B,B,A}            # positions on a face whose condition
                                    # overwrites the state (DirichletBC, a
                                    # coarse-fine face), and the patch's
                                    # `overwritten` nodes
    areas::NTuple{3,V}              # Ā at the face after each padded position
    volume::B                       # per component (and a trailing false): a
                                    # filter pass along a radial line weights
                                    # it by J
    deriv_lhs::V                    # the derivative's interior face relation
    deriv_rhs::V
    filter_lhs::V                   # the filter's, and its explicit face stencil
    filter_psi::V
end

# The line planes of the running sums and the buffers of their exchange.
struct LimiterLineBuffers{T,P<:AbstractArray{T,3},C<:AbstractArray{Int,3}}
    anchor::Vector{P}               # per d: (n_cons, n_a, n_b) Φ at the low face
    anchor_hi::Vector{P}            # Φ at the high face, at a patch interface
                                    # closed by the interface rows
    totals::Vector{P}               # line totals of the local running sums
    aux::Vector{P}                  # anchor (stage) or measurement (filter)
    offset::Vector{P}               # this rank's running-sum offset
    wrap::Vector{P}                 # the value at the global low face
    cstar::Vector{P}                # the constant removed from a filter flux
    zero_plane::Vector{P}
    dropped::Vector{P}              # a masked pass's dropped residual, summed
                                    # over this rank's part of each line, and
    dropped_mid::Vector{P}          # over that part's nodes up to the line's
                                    # middle face
    line_counts::Vector{C}          # per d: (4, n_a, n_b) each line's tallies
                                    # of a θ pass, summed into `counts`
    # Host copies of `offset`, `wrap` and `cstar` per d, which `_line_offsets!`
    # forms and uploads on device storage; empty until a device path needs them.
    staged::Vector{NTuple{3,Array{T,3}}}
    send::Vector{Vector{T}}
    recv::Vector{Vector{T}}
end

"""
    LimiterScratch

The padded arrays a limiter's stage and filter passes work in. A patch with a
same-level interface, and a tile of a radial grid, holds its face values, θ,
rates and speeds itself; any other patch borrows them, and every patch the
filter's pre-pass state, from the flux rows and `tmp_b` of the
`RHSWorkspace` its rank's patches share by extent. `lease` is that
workspace's lease, which `borrow_stage_buffers` and
`borrow_filter_buffers` take and `release_buffers!` returns: a
right-hand side, or another limiter of the same extent, that starts while it
is held raises an error.
"""
struct LimiterScratch{A<:AbstractArray}
    faces::FieldVector{A,Vector{A}} # face values, in the workspace's flux[1, :],
                                    # and the pressure's in flux[3, 1]
    base::FieldVector{A,Vector{A}}  # the filter's pre-pass state, flux[2, :]
    rates::A                        # a radial cell's rate, flux[2, 1] at a stage
    speeds::A                       # a radial face's speed, flux[2, 2] at a stage
    theta::A                        # θ per face; the workspace's tmp_b on one patch
    lease::Base.RefValue{Symbol}
end

# What a limiter has counted; `positivity_counts` reduces.
struct LimiterCounters
    # Stage faces, stage faces limited, filter faces, filter faces limited,
    # unguaranteed cell sides, and stage faces limited at r = 0, rank-local;
    # `positivity_counts` reduces.
    counts::Vector{Int}
end

"""
    PositivityLimiter

The state of the positivity limiter on a solver built with
`positivity_limiter = true` (see `Numerics`): the face weights, the
per-direction node registers, the line planes of the running sums, and the
bounds of the current `run!`. Built by setup.
"""
mutable struct PositivityLimiter{T,A<:AbstractArray{T,3},V<:AbstractVector{T},
                                 B<:AbstractVector{Bool},P<:AbstractArray{T,3},
                                 C<:AbstractArray{Int,3}}
    weight_set::LimiterWeights{T,A,V,B}
    line_buffers::LimiterLineBuffers{T,P,C}
    scratch::LimiterScratch{A}
    counters::LimiterCounters
    registers::Matrix{A}            # r[d, c]; c = n_cons + 1 the radial pressure
    register_fields::Vector{FieldVector{A,Vector{A}}}   # r[d, :] per d
    radial::NTuple{3,Bool}          # d is a radial line folded at r = 0
    fold_lo::NTuple{3,Bool}         # the global low end of d is a fold
    n_comp::NTuple{3,Int}           # registers of d: n_cons, and the pressure
    volume_state::NTuple{5,Bool}    # `volume` for ρ, the momenta and E
    areal::NTuple{3,Bool}           # d takes area-weighted fluxes, D(A_d F), on
                                    # lines of constant A_d (z on the r-z plane)
    interface::NTuple{3,NTuple{2,Bool}} # the ends of d at a patch interface
    open::NTuple{3,NTuple{2,Bool}}  # the ends of d open under `:ghost`, at a
                                    # same-level or a coarse-fine interface
    conservative::NTuple{3,NTuple{2,Bool}} # the ends of d at a same-level
                                    # interface whose filter row conserves
    periodic::NTuple{3,Bool}
    measure::NTuple{3,Int}          # local face of the filter relation, 0 none
    anchor_face::NTuple{3,Int}      # local face of a periodic or open-end anchor
    eps_rho::T
    eps_e::T
    active::Bool
end

"""
    borrow_stage_buffers(lim) -> LimiterScratch
    borrow_filter_buffers(lim) -> LimiterScratch

Take the lease of the workspace scratch a stage pass or a filter pass of
`lim` works in, and return the scratch. The pass ends with
`release_buffers!(lim)`. Between the two no right-hand side of a patch
sharing the workspace may start and no other limiter may borrow from it;
either raises an error naming the holder.
"""
borrow_stage_buffers(lim::PositivityLimiter) = _take_lease!(lim, :limiter_stage)
borrow_filter_buffers(lim::PositivityLimiter) = _take_lease!(lim, :limiter_filter)

@inline function _take_lease!(lim::PositivityLimiter, holder::Symbol)
    scratch = getfield(lim, :scratch)
    scratch.lease[] === :free || _lease_error(scratch.lease[], holder)
    scratch.lease[] = holder
    return scratch
end

@inline function release_buffers!(lim::PositivityLimiter)
    getfield(lim, :scratch).lease[] = :free
    return nothing
end

@noinline _lease_error(held::Symbol, wanted) =
    error("the right-hand-side scratch of this extent is held by `$held` and " *
          "`$wanted` asked for it: a limiter pass was not released")

"""
    PatchLimiters

The positivity limiters of a solver, one per patch this rank holds, in the
order of `solver.patches`; a single-patch solver holds its one, built with
the solver. On a refined solver they are
built as `run!` starts and rebuilt for the patches a regrid replaces
(`_follow_patches!`), since a patch's limiter holds the weights of its own
lines; the counts of a dropped patch's limiter are kept in `retired`.
`least` is the fewest parent nodes along a dimension a regridded box spans,
the shortest fine line between two coarse-fine faces that takes the face
form and the filter's face relation. `separate` is set for the tiles of a
radial grid, whose limiters hold the scratch of the radial pass themselves.
`lines` holds, per node count and per end at an interface, whether a closed
line takes the face form (`_limiter_tile_short`).
"""
mutable struct PatchLimiters
    limiters::Vector{PositivityLimiter}
    patches::Vector{Patch}
    retired::Vector{Int}
    least::Int
    separate::Bool
    lines::Dict{Tuple{Int,Bool,Bool},Bool}
end

PatchLimiters(limiters::Vector, patches::Vector, least::Int=4,
              separate::Bool=false) =
    PatchLimiters(limiters, patches, zeros(Int, 6), least, separate,
                  Dict{Tuple{Int,Bool,Bool},Bool}())
Base.getindex(pl::PatchLimiters, i::Int) = pl.limiters[i]
