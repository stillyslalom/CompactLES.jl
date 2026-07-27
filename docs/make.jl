using CompactLES
using Documenter
using Literate

# Tutorials are written as literate Julia scripts: runnable on their own with
# `julia --project=docs docs/literate/<name>.jl`, and rendered here into pages
# whose code Documenter executes at build time, so a tutorial that has gone
# stale fails the build instead of quietly lying. The generated markdown is a
# build artifact and is gitignored.
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const TUTORIAL_DIR = joinpath(@__DIR__, "src", "tutorials")

isdir(LITERATE_DIR) && for script in sort(readdir(LITERATE_DIR; join=true))
    endswith(script, ".jl") || continue
    Literate.markdown(script, TUTORIAL_DIR; documenter=true)
end

DocMeta.setdocmeta!(
    CompactLES,
    :DocTestSetup,
    :(using CompactLES);
    recursive=true,
)

makedocs(;
    modules=[CompactLES],
    authors="Alex Ames and contributors",
    sitename="CompactLES.jl",
    doctest=true,
    # The manual intentionally covers only the established API while physics
    # interfaces are still changing.
    checkdocs=:none,
    format=Documenter.HTML(;
        canonical="https://stillyslalom.github.io/CompactLES.jl",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Tutorials" => [
            "2-D shock tube with a switchable boundary" =>
                "tutorials/shock_tube_2d.md",
        ],
        "Numerics" => [
            "Compact operators" => "numerics/compact-operators.md",
            "Parallel decomposition" => "numerics/parallelism.md",
            "Validation" => "numerics/validation.md",
        ],
        "API reference" => [
            "Problem setup" => "reference/frontend.md",
            "Operators and decomposition" => "reference/operators.md",
            "Runtime and output" => "reference/runtime.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/stillyslalom/CompactLES.jl",
    devbranch="main",
)
