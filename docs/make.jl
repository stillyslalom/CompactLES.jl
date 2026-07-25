using CompactLES
using Documenter

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
