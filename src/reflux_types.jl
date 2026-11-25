# The record a `Patch` holds of each coarse-fine junction it takes part in.
# It is defined ahead of patches.jl, whose `Patch` stores it; the coupling
# that fills and reads it is in reflux.jl.

"""
    RefluxCapture

One side of a coarse-fine junction as one patch holds it: the parent's lines
through the face node beside a child tile, or the child's lines through its
first counted nodes (src/reflux.jl). `nodes` are the padded indices along `d`
whose increments enter the junction's face flux, with the coefficients of the
derivative (`dcoef`) and of the filter (`fcoef`); `dr` and `fr` are the
padded index `j` of the anchor face `j + ½` whose explicit stencil this rank
evaluates, 0 where another rank holds it. Every per-line array is indexed by
the line's position in `lines`, the padded ranges of the two transverse
dimensions in ascending order. `stage` collects one right-hand side, `du` and
`reg` follow it through the low-storage recurrence, and `reg` takes each
filter pass directly; `om0` is scratch for the child's correction of the
quadrature. `window` lists the parent's padded indices along `d` of the
gate's nodes this rank holds, `wslots` their places among the junction's
`GATE_WIDTH`, and `rho0` the density there at the step's start.
"""
struct RefluxCapture{T,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    level::Int                  # index in `solver.levels` of the parent level
    junction::Int               # the junction's index under that level
    child::Bool
    d::Int
    sgn::T
    nodes::NTuple{6,Int}
    dcoef::NTuple{6,T}
    dr::Int
    drs::NTuple{8,T}            # derivative face stencil, l = 1 - dM .. dM
    dM::Int
    drscale::T
    fcoef::NTuple{6,T}
    fr::Int
    frs::NTuple{8,T}            # filter face stencil, l = 1 - fM .. fM
    fM::Int
    frscale::T
    hd::T
    lines::NTuple{2,UnitRange{Int}}
    kap::V                      # per line: the measure of a derivative's flux
    kapf::V                     # per line: that of a filter pass, 0 if not taken
    bnode::Int                  # parent: padded index of the correction's node, or 0
    wb::V                       # parent: that node's composite weight
    band::NTuple{6,Int}         # child: the quadrature correction's nodes
    omega::NTuple{6,T}
    entry::Vector{Int}          # per line, its line of the junction
    window::Vector{Int}         # parent: the gate's nodes along d held here
    wslots::Vector{Int}         # parent: their places among the gate's slots
    rho0::Matrix{Float64}       # parent: the density there at the step's start
    snap::M
    stage::M
    du::M
    reg::M
    om0::M
end
