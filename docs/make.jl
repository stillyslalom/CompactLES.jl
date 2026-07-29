const T_MAKE_START = time()

using CompactLES
using Documenter
using Literate
using Printf

const T_LOADED = time()

# Documentation is the second of the two long CI jobs, and its total says
# nothing about which tutorial is expensive. Two things are measured here.
#
# Phases (load, Literate conversion, makedocs, deploy) come from plain wall
# clocks around each stage. Per-tutorial execution needs more care, because
# the tutorials do not run here: Documenter runs them later, while expanding
# the `@example` blocks Literate produced. `time_tutorial` injects a pair of
# hidden `@setup` blocks into the generated markdown, which run in the same
# module as that page's examples and write the elapsed time back into `Main`.
# The instrumentation therefore stays in this file rather than spreading into
# the tutorial sources, which have to remain readable and runnable on their own.
const PAGE_TIMES = Dict{String,Float64}()
const PHASES = Tuple{String,Float64}[]

macro phase(name, ex)
    quote
        local t0 = time()
        local val = $(esc(ex))
        push!(PHASES, ($(esc(name)), time() - t0))
        val
    end
end

"Wrap a Literate-generated page so it reports its own execution time."
function time_tutorial(name)
    start = "\n```@setup $name\nMain.PAGE_TIMES[\"$name\"] = -time()\n```\n"
    stop = "\n```@setup $name\nMain.PAGE_TIMES[\"$name\"] += time()\n```\n"
    return function (markdown)
        # Literate opens the page with an `@meta` block carrying the EditURL.
        # The timer goes after it, so that block stays where the reader and
        # Documenter both expect it.
        cut = findfirst("```\n", markdown)
        at = startswith(markdown, "```@meta") && cut !== nothing ? last(cut) : 0
        return markdown[1:at] * start * markdown[at+1:end] * stop
    end
end

# Default-CI tutorials are deliberately registered rather than discovered.
# This keeps an expensive case study from entering every documentation build
# merely because a script was added to the repository. Each registered script
# is runnable on its own and is executed again by Documenter after Literate
# converts it to markdown.
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const TUTORIAL_DIR = joinpath(@__DIR__, "src", "tutorials")
const TUTORIALS = [
    "acoustic_pulse.jl",
    "shock_tube_1d.jl",
    "multicomponent_state.jl",
    "radial_coordinates.jl",
    "cylindrical_3d.jl",
    "spherical_3d.jl",
]

mkpath(TUTORIAL_DIR)
for page in readdir(TUTORIAL_DIR; join=true)
    isfile(page) || continue
    endswith(page, ".md") || continue
    rm(page)
end

@phase "Literate conversion" for name in TUTORIALS
    script = joinpath(LITERATE_DIR, name)
    Literate.markdown(script, TUTORIAL_DIR; documenter=true,
                      postprocess=time_tutorial(first(splitext(name))))
end

DocMeta.setdocmeta!(
    CompactLES,
    :DocTestSetup,
    :(using CompactLES);
    recursive=true,
)

@phase "makedocs" makedocs(;
    modules=[CompactLES],
    authors="Alex Ames and contributors",
    sitename="CompactLES.jl",
    repo=Documenter.Remotes.GitHub("stillyslalom", "CompactLES.jl"),
    doctest=true,
    checkdocs=:exports,
    treat_markdown_warnings_as_error=true,
    pagesonly=true,
    format=Documenter.HTML(;
        canonical="https://stillyslalom.github.io/CompactLES.jl",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Your first simulation" => "tutorials/acoustic_pulse.md",
            "Regularizing a shock" => "tutorials/shock_tube_1d.md",
            "A multicomponent state" => "tutorials/multicomponent_state.md",
            "Radial coordinates" => "tutorials/radial_coordinates.md",
            "3D cylinder" => "tutorials/cylindrical_3d.md",
            "3D sphere" => "tutorials/spherical_3d.md",
        ],
        "How-to guides" => [
            "Define a problem" => "how-to/problem-setup.md",
            "Choose boundary conditions" => "how-to/boundary-conditions.md",
            "Control and diagnose a run" => "how-to/run-control.md",
            "Write output and restart" => "how-to/output-restart.md",
            "Run in parallel" => "how-to/parallel-runs.md",
        ],
        "Explanation" => [
            "Governing equations" => "explanation/governing-equations.md",
            "Spatial and temporal discretization" =>
                "explanation/discretization.md",
            "Filtering and artificial properties" =>
                "explanation/regularization.md",
            "Thermodynamics and species transport" =>
                "explanation/thermodynamics.md",
            "Curvilinear coordinates" => "explanation/geometry.md",
            "Characteristic open boundaries" =>
                "explanation/open-boundaries.md",
            "The parallel compact solve" =>
                "explanation/parallel-compact-solve.md",
            "Verification, validation, and calibration" =>
                "explanation/verification-validation.md",
        ],
        "Case studies" => [
            "A boundary that changes type" =>
                "case-studies/switchable-boundary.md",
        ],
        "Reference" => [
            "Problem setup" => "reference/frontend.md",
            "Physics models" => "reference/physics.md",
            "Geometry and boundaries" => "reference/geometry-boundaries.md",
            "Operators and decomposition" => "reference/operators.md",
            "Runtime and output" => "reference/runtime.md",
            "Diagnostics" => "reference/diagnostics.md",
            "Public index" => "reference/index.md",
        ],
    ],
)

@phase "deploydocs" deploydocs(;
    repo="github.com/stillyslalom/CompactLES.jl",
    devbranch="main",
)

# The tutorial times are a subset of `makedocs`, not additional to it: each page
# runs inside the expansion stage. What is left of `makedocs` after subtracting
# them is doctests, cross-reference resolution, and HTML rendering.
let pages = sum(values(PAGE_TIMES); init=0.0),
    total = time() - T_MAKE_START

    println("\n=== documentation build timing ===")
    @printf("  %-32s %7.2f s\n", "package load", T_LOADED - T_MAKE_START)
    for (name, seconds) in sort(PHASES; by=p -> -p[2])
        @printf("  %-32s %7.2f s\n", name, seconds)
    end
    println("  tutorials, inside makedocs:")
    for (name, seconds) in sort(collect(PAGE_TIMES); by=p -> -p[2])
        @printf("    %-30s %7.2f s\n", name, seconds)
    end
    @printf("  %-32s %7.2f s\n", "  tutorials, total", pages)
    @printf("  %-32s %7.2f s\n", "TOTAL", total)
    println()
end
