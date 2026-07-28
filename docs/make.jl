using CompactLES
using Documenter
using Literate

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

for name in TUTORIALS
    script = joinpath(LITERATE_DIR, name)
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

deploydocs(;
    repo="github.com/stillyslalom/CompactLES.jl",
    devbranch="main",
)
