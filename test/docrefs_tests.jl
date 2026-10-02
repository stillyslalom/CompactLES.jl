# Documenter cross-reference guard. The docs job is the slowest leg of CI
# and the last to report, and it has failed repeatedly on one cheap mistake:
# a docstring or page links `[`Name`](@ref)` to a name that no `@docs` block
# on any page renders, so Documenter cannot resolve the link and, under
# `treat_markdown_warnings_as_error`, fails the build. This file resolves
# every such link the way Documenter will, against the `@docs` blocks under
# `docs/src`, without Literate, Documenter, or the tutorials, so it runs in
# the serial gate in well under a second after the package loads.
#
# Three things are checked, each a Documenter failure mode that needs no
# build to detect:
#   1. every `@ref` in a rendered docstring or a page resolves to a `@docs`
#      entry or to a page heading, and an unqualified one in a docstring names
#      a binding of the docstring's own module, where Documenter looks it up.
#      Documenter looks a link up among the heading anchors of every page
#      first, a code link such as [`Sphere`](@ref) included, and fails the
#      build when its slug is the anchor of more than one heading; a second
#      `## Sphere` anywhere in the docs therefore breaks every such link.
#      A code link to a rendered docstring whose slug is the anchor of a
#      single heading builds, but is linked to that heading rather than to the
#      docstring, and a second heading of the same name turns it into the
#      failure above; it fails here too. Qualifying the link,
#      [`Sphere`](@ref CompactLES.Regions.Sphere), gives it a slug no heading
#      has;
#   2. every `@docs` entry names a documented binding of CompactLES or of one
#      of its submodules;
#   3. every exported or public name CompactLES or a submodule owns has a
#      docstring and is rendered by some `@docs` block. Documenter's
#      `checkdocs = :exports` misses an exported binding that has no docstring
#      at all and does not look at `public` names, so this is deliberately
#      stricter. `public` is Julia 1.11 syntax; on 1.10 only the exports are
#      checked.
#
# Standalone: julia --project=. test/docrefs_tests.jl
# The serial suite includes it as one testset.

if !isdefined(Main, :CompactLES)
    using CompactLES
end
using Test

module DocRefs

using CompactLES

const DOCS_SRC = normpath(joinpath(@__DIR__, "..", "docs", "src"))
const TUTORIALS = joinpath(DOCS_SRC, "tutorials")   # generated; not scanned
const EXAMPLE_PAGES = joinpath(DOCS_SRC, "examples")  # generated; not scanned
const LITERATE = normpath(joinpath(@__DIR__, "..", "docs", "literate"))
const EXAMPLES = normpath(joinpath(@__DIR__, "..", "examples"))

# The identifier a `@docs` entry or a code reference names: the last
# component of a possibly qualified name, with any call signature dropped.
# `CompactLES.run!(solver, Q)` and `run!` both give `run!`; `@threaded` keeps
# its sigil.
function base_name(code::AbstractString)
    m = match(r"^\s*((?:[A-Za-z_][\w!]*\.)*)(@?[A-Za-z_][\w!]*)", code)
    m === nothing && return nothing
    return String(m.captures[2])
end

# Pages Documenter renders: every markdown file under docs/src except the
# generated tutorials and examples, which only exist after Literate runs.
function pages()
    out = String[]
    for (root, _, files) in walkdir(DOCS_SRC)
        (startswith(root, TUTORIALS) || startswith(root, EXAMPLE_PAGES)) && continue
        for f in files
            endswith(f, ".md") && push!(out, joinpath(root, f))
        end
    end
    return sort(out)
end

# The markdown of a Literate tutorial script: the text of its comment lines,
# which is what the generated page under `tutorials/` carries. Literate's
# control comments (`#src`, `#md`, `#-` and the like) are not markdown.
function literate_markdown(script)
    lines = String[]
    for line in eachline(script)
        m = match(r"^#(?: (.*))?$", line)
        # A code line becomes a blank line, so that a line of this text is the
        # same line of the script in a report.
        push!(lines, m === nothing || m.captures[1] === nothing ? "" :
                     String(m.captures[1]))
    end
    return join(lines, "\n")
end

# Whether a script in examples/ is a Literate page: one opens with its
# `# # Title` line, where a plain driver opens with an ordinary comment.
is_example_page(script) =
    endswith(script, ".jl") && startswith(readline(script), "# # ")

# Every rendered page's markdown, keyed by a path to report: the pages under
# docs/src as they are, and each tutorial and example as the markdown of its
# script.
function page_texts()
    texts = Pair{String,String}[]
    for page in pages()
        push!(texts, page => read(page, String))
    end
    for f in sort(readdir(LITERATE; join=true))
        endswith(f, ".jl") && push!(texts, f => literate_markdown(f))
    end
    for f in sort(readdir(EXAMPLES; join=true))
        is_example_page(f) && push!(texts, f => literate_markdown(f))
    end
    return texts
end

# `@docs` entries per page, as (page, entry) pairs, one per non-blank line
# inside a ```@docs fence.
function docs_entries(page_paths)
    entries = Tuple{String,String}[]
    for page in page_paths
        inblock = false
        for line in eachline(page)
            s = strip(line)
            if !inblock
                startswith(s, "```@docs") && (inblock = true)
            elseif startswith(s, "```")
                inblock = false
            elseif !isempty(s)
                push!(entries, (page, String(s)))
            end
        end
    end
    return entries
end

# Documenter's `slugify` (utilities.jl in Documenter 1.x), copied verbatim
# so that a heading anchor here is the string Documenter stores. It keeps
# case and punctuation and drops symbols such as backticks and `$`.
function slugify(s::AbstractString)
    s = replace(s, r"\s+" => "-")
    s = replace(s, r"&" => "-and-")
    s = replace(s, r"[^\p{L}\p{P}\d\-]+" => "")
    return String(strip(replace(s, r"\-\-+" => "-"), '-'))
end

# The heading anchors of every page, as Documenter's `TrackHeaders` records
# them: the slug of each top-level heading, mapped to every place it occurs,
# as "page:line `heading`". Documenter tracks only the headings at the top
# level of a page, so a heading inside an admonition (indented) or inside a
# docstring makes no anchor. A slug occurring more than once is legal; it
# fails the build only when an `@ref` names it.
function headers(texts)
    anchors = Dict{String,Vector{String}}()
    for (page, text) in texts
        infence = false
        for (n, line) in enumerate(split(text, '\n'))
            s = rstrip(line)
            startswith(s, "```") && (infence = !infence; continue)
            infence && continue
            m = match(r"^#{1,6}\s+(.*?)\s*$", s)
            m === nothing && continue
            push!(get!(anchors, slugify(m.captures[1]), String[]),
                  "$(relpath(page)):$n `$(s)`")
        end
    end
    return anchors
end

# Every `@ref` link in `text`, as (link, slug, code). `slug` is the string
# Documenter's `xref` looks up among the heading anchors before it tries a
# docstring: the code of a `[`x`](@ref)` label as written, the slugified text
# of a plain label, an explicit `@ref x` target as written, and a quoted
# `@ref "Some heading"` target slugified. `code` is the name a docstring
# lookup takes, `nothing` for a link that can only reach a heading.
function refs(text::AbstractString)
    out = Tuple{String,String,Union{String,Nothing}}[]
    for m in eachmatch(r"\[([^\[\]]*)\]\(@ref(?:\s+([^)]*?))?\s*\)", text)
        label, target = m.captures
        link = replace(m.match, r"\s+" => " ")
        if target !== nothing
            quoted = match(r"\"(.+)\"", target)
            if quoted === nothing
                push!(out, (link, String(target), String(target)))
            else
                push!(out, (link, slugify(quoted.captures[1]), nothing))
            end
        else
            code = match(r"^`([^`]*)`$", strip(label))
            if code === nothing
                push!(out, (link, slugify(strip(label)), nothing))
            else
                push!(out, (link, String(code.captures[1]), String(code.captures[1])))
            end
        end
    end
    return out
end

# CompactLES and the submodules whose names it documents.
const MODULES = (CompactLES, CompactLES.Regions, CompactLES.DiffusionData,
                 CompactLES.Presets)

# Raw docstring text of every documented binding of CompactLES and its
# submodules, keyed by the binding's name, and the module each is documented
# in. A binding documented at several signatures contributes all of them.
function docstrings()
    texts = Dict{String,String}()
    owners = Dict{String,Module}()
    for mod in MODULES, (binding, multidoc) in Base.Docs.meta(mod)
        parts = String[]
        for (_, docstr) in multidoc.docs
            for t in docstr.text
                t isa AbstractString && push!(parts, String(t))
            end
        end
        texts[String(binding.var)] = join(parts, "\n")
        owners[String(binding.var)] = mod
    end
    return texts, owners
end

# Exported names, and from Julia 1.11 public ones, that CompactLES or a
# submodule owns. A re-exported binding from a dependency, such as the `MPI`
# module, has its documentation in that package and is invisible to
# Documenter's `checkdocs`, which reads only this package's own docstring
# tables; skip it here for the same reason.
function exported_names()
    out = String[]
    for mod in MODULES, n in names(mod)
        n === nameof(mod) && continue
        Base.which(mod, n) === mod || continue
        push!(out, String(n))
    end
    return unique!(out)
end

"""
    check() -> (unresolved, unknown_entries, undocumented_exports, captured)

Run the three checks and return the offenders, each as a vector of
human-readable lines; all four empty means the docs cross-references are
sound. `captured` lists the code links to a rendered docstring whose slug is
also the anchor of a single heading: Documenter builds them without error and
links them to the heading.
"""
function check()
    page_paths = pages()
    entries = docs_entries(page_paths)
    rendered = Set{String}()
    unknown_entries = String[]
    texts, owners = docstrings()
    for (page, entry) in entries
        name = base_name(entry)
        if name === nothing || !haskey(texts, name)
            push!(unknown_entries,
                  "$(relpath(page)): `$entry` is not a documented binding of CompactLES")
        else
            push!(rendered, name)
        end
    end
    texts_by_page = page_texts()
    anchors = headers(texts_by_page)
    unresolved = String[]
    captured = String[]
    # Documenter resolves an unqualified name in a docstring's `@ref` in the
    # module the docstring belongs to, and falls back to `Main` only for a
    # fully qualified name.
    function binds(owner, target)
        occursin('.', first(split(target, '('))) && return true
        n = base_name(target)
        return n !== nothing && isdefined(owner, Symbol(n))
    end
    # The fully qualified name of a documented binding, for a link that names
    # it. A module's docstring is held by the module itself.
    function qualified(n)
        mod = owners[n]
        return nameof(mod) === Symbol(n) ? string(mod) : "$mod.$n"
    end
    # Documenter tries the heading anchors before the docstrings, for a code
    # link as for a text link. A slug naming one heading resolves to it; a
    # slug naming several is an error, and no docstring of that name is
    # tried. `owner` is the module of the docstring holding the link, or
    # `nothing` on a page.
    function resolve!(where, link, slug, code, owner)
        places = get(anchors, slug, String[])
        if length(places) > 1
            push!(unresolved, "$where: $link is ambiguous: its slug `$slug` is " *
                              "the anchor of $(length(places)) headings, " *
                              join(places, ", "))
        elseif length(places) == 1
            n = code === nothing ? nothing : base_name(code)
            n !== nothing && n in rendered &&
                push!(captured, "$where: $link resolves to the heading " *
                                "$(only(places)), not to the docstring of `$n`; " *
                                "qualify the link, as in " *
                                "[`$n`](@ref $(qualified(n))), or rename the heading")
        elseif code === nothing
            push!(unresolved, "$where: $link names no heading")
        else
            n = base_name(code)
            n !== nothing && n in rendered ||
                push!(unresolved, "$where: $link names no rendered docstring")
            owner === nothing || binds(owner, code) ||
                push!(unresolved, "$where in $owner: $link is not bound there; " *
                                  "qualify it")
        end
    end
    for name in sort(collect(rendered))
        for (link, slug, code) in refs(texts[name])
            resolve!("docstring of `$name`", link, slug, code, owners[name])
        end
    end
    for (page, text) in texts_by_page
        for (link, slug, code) in refs(text)
            resolve!(relpath(page), link, slug, code, nothing)
        end
    end
    undocumented_exports = String[]
    for name in exported_names()
        if !haskey(texts, name)
            push!(undocumented_exports, "exported `$name` has no docstring")
            continue
        end
        name in rendered ||
            push!(undocumented_exports, "exported `$name` has a docstring no @docs block renders")
    end
    return unresolved, unknown_entries, undocumented_exports, captured
end

end # module DocRefs

@testset "docs cross-references resolve" begin
    unresolved, unknown_entries, undocumented_exports, captured = DocRefs.check()
    for line in captured
        @warn "@ref linked to a heading: $line"
    end
    for line in unresolved
        @warn "unresolved @ref: $line"
    end
    for line in unknown_entries
        @warn "unknown @docs entry: $line"
    end
    for line in undocumented_exports
        @warn "checkdocs=:exports would fail: $line"
    end
    @test isempty(captured)
    @test isempty(unresolved)
    @test isempty(unknown_entries)
    @test isempty(undocumented_exports)
end
