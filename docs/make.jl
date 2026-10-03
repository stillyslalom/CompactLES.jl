const T_MAKE_START = time()

using CompactLES
using Documenter
using Literate
using Printf
using TOML

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
    "coalescing_shock.jl",
    "shock_tube.jl",
    "acoustic_interface.jl",
    "sound_absorption.jl",
    "loschmidt_cell.jl",
    "richtmyer_meshkov.jl",
    "rayleigh_taylor.jl",
    "supernova_remnant.jl",
    "imploding_shock.jl",
    "advected_bubbles.jl",
    "axis_crossing_vortex.jl",
    "oscillating_sphere.jl",
]

# Examples are longer calculations whose figures are computed once and
# committed. Each script in `examples/` listed here is both the driver and the
# page: a full run writes its figures and a `provenance.toml` into
# `src/assets/examples/<name>/`, and this build converts the script to
# markdown without executing it (plain `julia` fences, no `@example`), with a
# note built from the provenance file after the title. The weekly validation
# workflow runs every listed example with `smoke=true`.
const EXAMPLE_SCRIPTS = joinpath(@__DIR__, "..", "examples")
const EXAMPLE_DIR = joinpath(@__DIR__, "src", "examples")
const EXAMPLES = [
    "shock_capturing.jl",
    "shock_bubble.jl",
    "shock_tube.jl",
    "taylor_green.jl",
    "vortex_ring_shock.jl",
]

"The provenance note of an example, from the record its full run committed."
function provenance_note(name)
    stamp = joinpath(@__DIR__, "src", "assets", "examples", name, "provenance.toml")
    isfile(stamp) || error("examples/$name.jl has no committed figures: $stamp is " *
                           "missing; run the example to produce them")
    r = TOML.parsefile(stamp)
    commit = r["commit"]
    url = "https://github.com/stillyslalom/CompactLES.jl/commit/$commit"
    link = "[`$(first(commit, 7))`]($url)"
    dirty = r["dirty"] ? " with uncommitted changes to the package" : ""
    threads = r["threads"] == 1 ? "1 thread" : "$(r["threads"]) threads"
    ranks = r["ranks"] == 1 ? "1 rank" : "$(r["ranks"]) ranks"
    minutes = r["wall_seconds"] / 60
    wall = minutes < 2 ? @sprintf("%.0f s", r["wall_seconds"]) : @sprintf("%.0f min", minutes)
    return """
           !!! note "Provenance"
               The figures on this page were computed on $(r["date"]) at commit
               $link$dirty, with Julia $(r["julia"]) on $ranks × $threads of
               a $(r["hardware"]), on $(r["grid"]), in $wall. They are
               reproduced by `$(r["command"])`.
           """
end

"Give a converted example its edit link and provenance note."
function example_page(name)
    return function (markdown)
        lines = split(markdown, '\n')
        title = findfirst(l -> startswith(l, "# "), lines)
        title === nothing && error("examples/$name.jl has no `# # Title` line")
        edit = "```@meta\nEditURL = \"../../../examples/$name.jl\"\n```\n"
        return edit * join(lines[1:title], '\n') * "\n\n" * provenance_note(name) *
               join(lines[title+1:end], '\n')
    end
end

for dir in (TUTORIAL_DIR, EXAMPLE_DIR)
    mkpath(dir)
    for page in readdir(dir; join=true)
        isfile(page) || continue
        endswith(page, ".md") || continue
        rm(page)
    end
end

@phase "Literate conversion" begin
    for name in TUTORIALS
        script = joinpath(LITERATE_DIR, name)
        Literate.markdown(script, TUTORIAL_DIR; documenter=true,
                          postprocess=time_tutorial(first(splitext(name))))
    end
    for name in EXAMPLES
        script = joinpath(EXAMPLE_SCRIPTS, name)
        Literate.markdown(script, EXAMPLE_DIR; documenter=false,
                          postprocess=example_page(first(splitext(name))))
    end
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
        # Documenter does not detect a favicon from the assets tree; the head
        # link is emitted only for assets registered here. A raw head tag would
        # need a per-page relative href, so the registered `.ico` (which
        # Documenter path-corrects for each page) is the reliable route.
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "Cheat sheet" => "reference/input-deck-cheat-sheet.md",
        "Tutorials" => [
            "Coalescing shock" => "tutorials/coalescing_shock.md",
            "Shock tube" => "tutorials/shock_tube.md",
            "Acoustic interface" => "tutorials/acoustic_interface.md",
            "Sound absorption" => "tutorials/sound_absorption.md",
            "Loschmidt cell" => "tutorials/loschmidt_cell.md",
            "Richtmyer–Meshkov instability" => "tutorials/richtmyer_meshkov.md",
            "Rayleigh–Taylor instability" => "tutorials/rayleigh_taylor.md",
            "Supernova remnant" => "tutorials/supernova_remnant.md",
            "Imploding shock" => "tutorials/imploding_shock.md",
            "Advected bubbles" => "tutorials/advected_bubbles.md",
            "Axis-crossing vortex" => "tutorials/axis_crossing_vortex.md",
            "Oscillating sphere" => "tutorials/oscillating_sphere.md",
        ],
        "Examples" => [
            "Shock-capturing tests" => "examples/shock_capturing.md",
            "Shock–bubble interaction" => "examples/shock_bubble.md",
            "Reshocked mixing layer" => "examples/shock_tube.md",
            "Taylor–Green vortex" => "examples/taylor_green.md",
            "Vortex ring and shock" => "examples/vortex_ring_shock.md",
        ],
        "How-to guides" => [
            "Define a problem" => "how-to/problem-setup.md",
            "Choose boundary conditions" => "how-to/boundary-conditions.md",
            "Choose numerics" => "how-to/numerics-choices.md",
            "Control a run" => "how-to/run-control.md",
            "Write output and restart" => "how-to/output-restart.md",
            "Run in parallel" => "how-to/parallel-runs.md",
        ],
        "Explanation" => [
            "Governing equations" => "explanation/governing-equations.md",
            "Discretization" =>
                "explanation/discretization.md",
            "Filtering and artificial properties" =>
                "explanation/regularization.md",
            "Thermodynamics and transport" =>
                "explanation/thermodynamics.md",
            "Curvilinear coordinates" => "explanation/geometry.md",
            "Open boundaries" =>
                "explanation/open-boundaries.md",
            "Parallel compact solve" =>
                "explanation/parallel-compact-solve.md",
            "Verification and validation" =>
                "explanation/verification-validation.md",
        ],
        "Reference" => [
            "Supported combinations" => "reference/capabilities.md",
            "Input and runtime API" => [
                "Problem setup" => "reference/frontend.md",
                "Physics models" => "reference/physics.md",
                "Geometry and boundaries" => "reference/geometry-boundaries.md",
                "Runtime and output" => "reference/runtime.md",
                "Field extraction and plotting" => "reference/extraction.md",
                "Diagnostics" => "reference/diagnostics.md",
            ],
            "Advanced and extension API" => [
                "Adaptive mesh refinement" => "reference/amr.md",
                "Operators and decomposition" => "reference/operators.md",
                "Extending CompactLES" => "reference/extensions.md",
            ],
            "Public API index" => "reference/index.md",
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
