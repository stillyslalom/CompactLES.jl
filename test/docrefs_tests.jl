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
#      entry or to a page header;
#   2. every `@docs` entry names a documented binding of CompactLES;
#   3. every exported name has a docstring and is rendered by some `@docs`
#      block. Documenter's `checkdocs = :exports` misses an exported binding
#      that has no docstring at all, so this is deliberately stricter.
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
const LITERATE = normpath(joinpath(@__DIR__, "..", "docs", "literate"))

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
# generated tutorials, which only exist after Literate runs.
function pages()
    out = String[]
    for (root, _, files) in walkdir(DOCS_SRC)
        startswith(root, TUTORIALS) && continue
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
        m === nothing && continue
        push!(lines, m.captures[1] === nothing ? "" : String(m.captures[1]))
    end
    return join(lines, "\n")
end

# Every rendered page's markdown, keyed by a path to report: the pages under
# docs/src as they are, and each tutorial as the markdown of its script.
function page_texts()
    texts = Pair{String,String}[]
    for page in pages()
        push!(texts, page => read(page, String))
    end
    for f in sort(readdir(LITERATE; join=true))
        endswith(f, ".jl") && push!(texts, f => literate_markdown(f))
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

# Section headers of every page, as Documenter slugifies them for `@ref`:
# lowercase, with the punctuation it strips removed and runs of whitespace
# collapsed to one hyphen. Header refs compare on this form.
slug(text) = replace(lowercase(strip(text)), r"[^\w\s-]" => "",
                     r"\s+" => "-")

function headers(texts)
    slugs = Set{String}()
    for (_, text) in texts
        infence = false
        for line in split(text, '\n')
            s = rstrip(line)
            startswith(s, "```") && (infence = !infence; continue)
            infence && continue
            m = match(r"^#{1,6}\s+(.*?)\s*$", s)
            m === nothing || push!(slugs, slug(m.captures[1]))
        end
    end
    return slugs
end

# Every `@ref` link in `text`: (target, is_code). `[`x`](@ref)` and
# `[text](@ref x)` both give target `x`; a bare `[Some header](@ref)` is a
# header reference.
function refs(text::AbstractString)
    out = Tuple{String,Bool}[]
    for m in eachmatch(r"\[([^\[\]]*)\]\(@ref(?:\s+([^)\s]+))?\s*\)", text)
        label, target = m.captures
        if target !== nothing
            push!(out, (String(target), true))
        else
            code = match(r"^`([^`]*)`$", strip(label))
            if code === nothing
                push!(out, (String(label), false))
            else
                push!(out, (String(code.captures[1]), true))
            end
        end
    end
    return out
end

# Raw docstring text of every documented binding of CompactLES, keyed by the
# binding's name. A binding documented at several signatures contributes all
# of them.
function docstrings()
    texts = Dict{String,String}()
    for (binding, multidoc) in Base.Docs.meta(CompactLES)
        parts = String[]
        for (_, docstr) in multidoc.docs
            for t in docstr.text
                t isa AbstractString && push!(parts, String(t))
            end
        end
        texts[String(binding.var)] = join(parts, "\n")
    end
    return texts
end

function exported_names()
    out = String[]
    for n in names(CompactLES)
        n === :CompactLES && continue
        push!(out, String(n))
    end
    return out
end

"""
    check() -> (unresolved, unknown_entries, undocumented_exports)

Run the three checks and return the offenders, each as a vector of
human-readable lines; all three empty means the docs cross-references are
sound.
"""
function check()
    page_paths = pages()
    entries = docs_entries(page_paths)
    rendered = Set{String}()
    unknown_entries = String[]
    texts = docstrings()
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
    header_slugs = headers(texts_by_page)
    unresolved = String[]
    resolves(target, is_code) =
        is_code ? (n = base_name(target); n !== nothing && n in rendered) :
                  slug(target) in header_slugs
    for name in sort(collect(rendered))
        for (target, is_code) in refs(texts[name])
            resolves(target, is_code) ||
                push!(unresolved, "docstring of `$name`: @ref `$target`")
        end
    end
    for (page, text) in texts_by_page
        for (target, is_code) in refs(text)
            resolves(target, is_code) ||
                push!(unresolved, "$(relpath(page)): @ref `$target`")
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
    return unresolved, unknown_entries, undocumented_exports
end

end # module DocRefs

@testset "docs cross-references resolve" begin
    unresolved, unknown_entries, undocumented_exports = DocRefs.check()
    for line in unresolved
        @warn "unresolved @ref: $line"
    end
    for line in unknown_entries
        @warn "unknown @docs entry: $line"
    end
    for line in undocumented_exports
        @warn "checkdocs=:exports would fail: $line"
    end
    @test isempty(unresolved)
    @test isempty(unknown_entries)
    @test isempty(undocumented_exports)
end
