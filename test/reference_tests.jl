# Reference-layout guard. The files under reference/ hold one kind of
# information each: measurements in CALIBRATION_APPENDIX.md, the defaults and
# the symptom list in CALIBRATION.md, the mechanism in DESIGN.md, the open
# work in ROADMAP.md, the history in the git log. Between September 2026
# reviews the appendix grew fivefold and every closed item was recorded in
# five places, because nothing checked the layout. This file does, in the
# form Documenter's cross-reference guard takes in docrefs_tests.jl: it
# fails the serial suite on
#   1. an exponent-formatted number with a decimal mantissa (`4.8e-16`,
#      `2.85e5`) in any prose file other than the appendix, since a digit of
#      that shape is a measurement and a measurement is recorded once. A
#      one-digit mantissa (`1e-3`) is not matched, since in the prose files
#      that form is mostly an order of magnitude, a tolerance or a default;
#   2. CALIBRATION.md over the line cap its header states;
#   3. a closed ROADMAP entry (`- [x]`) longer than five lines: one sentence
#      and a commit hash is the record, the commit message is the account;
#   4. the word "supersede" in the appendix, the mark of a number annotated
#      instead of deleted;
#   5. a relative link in a reference file, CLAUDE.md, README.md or a bench/
#      markdown archive whose file or header anchor does not exist, which is
#      how a reorganized appendix strands the pointers into it;
#   6. a sentence of a reference file carrying more than two numerals outside
#      a table, a fenced block, a heading or an inline code span, citations
#      excepted: a figure in a sentence is a measurement recited, and the
#      prose states the finding in words while the figures sit in a table.
# The checks read the repository only, so the file runs without the package.
#
# Standalone: julia --project=. test/reference_tests.jl
# The serial suite includes it as one testset.

using Test

module ReferenceLayout

const ROOT = normpath(joinpath(@__DIR__, ".."))
const REFERENCE = joinpath(ROOT, "reference")
const APPENDIX = joinpath(REFERENCE, "CALIBRATION_APPENDIX.md")
const CALIBRATION = joinpath(REFERENCE, "CALIBRATION.md")
const ROADMAP = joinpath(REFERENCE, "ROADMAP.md")
const CLOSED_ENTRY_LINES = 5

rel(path) = replace(relpath(path, ROOT), '\\' => '/')

# The prose files the layout rules apply to: the reference files other than
# the appendix and the archived bug reports, the agent orientation, the
# README, and the hand-written documentation pages (the tutorials and examples
# are generated from docs/literate and examples/ and hold code).
function prose_files()
    out = String[]
    for f in readdir(REFERENCE; join=true)
        endswith(f, ".md") && f != APPENDIX && push!(out, f)
    end
    push!(out, joinpath(ROOT, "CLAUDE.md"))
    push!(out, joinpath(ROOT, "README.md"))
    docs = joinpath(ROOT, "docs", "src")
    for (dir, _, files) in walkdir(docs)
        any(g -> startswith(dir, joinpath(docs, g)), ("tutorials", "examples")) &&
            continue
        for f in files
            endswith(f, ".md") && push!(out, joinpath(dir, f))
        end
    end
    return sort(out)
end

# Files whose relative links are resolved: the reference files, CLAUDE.md,
# README.md and the markdown archives under bench/. A link into the appendix
# from any of them names a header that must exist.
function linked_files()
    out = String[]
    for top in (REFERENCE, joinpath(ROOT, "bench"))
        for (dir, _, files) in walkdir(top)
            for f in files
                endswith(f, ".md") && push!(out, joinpath(dir, f))
            end
        end
    end
    push!(out, joinpath(ROOT, "CLAUDE.md"))
    push!(out, joinpath(ROOT, "README.md"))
    return sort(out)
end

const EXPONENT = r"\d\.\d+e[-+]?\d+"

# Occurrences of that shape that are not measurements, each named with its
# file: a parameter default written as the source writes it.
const EXPONENT_EXEMPT = [
    ("docs/src/reference/input-deck-cheat-sheet.md", "p_inf=6.0e8"),
]

function exponent_numbers()
    out = String[]
    for f in prose_files()
        exempt = [s for (file, s) in EXPONENT_EXEMPT if file == rel(f)]
        for (i, line) in enumerate(eachline(f))
            for s in exempt
                line = replace(line, s => "")
            end
            for m in eachmatch(EXPONENT, line)
                push!(out, "$(rel(f)):$i: $(m.match)")
            end
        end
    end
    return out
end

# CALIBRATION.md states its own cap in its header ("capped at N lines"), so
# the number lives beside the rule it enforces.
function calibration_cap()
    lines = readlines(CALIBRATION)
    cap = nothing
    for line in lines[1:min(end, 20)]
        m = match(r"capped at (\d+) lines", line)
        m === nothing || (cap = parse(Int, m.captures[1]); break)
    end
    return cap, length(lines)
end

# Closed ROADMAP entries: a `- [x]` list item runs to the next list item at
# its indent or shallower, a blank line, or a header.
function long_closed_entries()
    lines = readlines(ROADMAP)
    out = String[]
    i = 1
    while i <= length(lines)
        m = match(r"^(\s*)- \[x\]", lines[i])
        if m === nothing
            i += 1
            continue
        end
        indent = length(m.captures[1])
        j = i + 1
        while j <= length(lines)
            s = lines[j]
            isempty(strip(s)) && break
            startswith(s, '#') && break
            item = match(r"^(\s*)- ", s)
            item !== nothing && length(item.captures[1]) <= indent && break
            j += 1
        end
        n = j - i
        n > CLOSED_ENTRY_LINES &&
            push!(out, "$(rel(ROADMAP)):$i: closed entry spans $n lines")
        i = j
    end
    return out
end

function supersede_lines()
    out = String[]
    for (i, line) in enumerate(eachline(APPENDIX))
        occursin(r"supersede"i, line) && push!(out, "$(rel(APPENDIX)):$i")
    end
    return out
end

# Header anchors as GitHub renders them: lowercase, punctuation other than
# hyphens and spaces removed, spaces to hyphens. A repeated header gets a
# numeric suffix, which a link should not rely on, so only the first form is
# accepted. Fenced code is not scanned for headers.
function anchor(text)
    s = lowercase(strip(text))
    s = replace(s, r"[^\p{L}\p{N}\s_-]" => "")
    return replace(s, r"\s" => "-")
end

function anchors(path)
    out = Set{String}()
    infence = false
    for line in eachline(path)
        s = rstrip(line)
        startswith(s, "```") && (infence = !infence; continue)
        infence && continue
        m = match(r"^#{1,6}\s+(.*?)\s*$", s)
        m === nothing || push!(out, anchor(m.captures[1]))
    end
    return out
end

# Relative links `[text](target)` in `path`, outside fenced code, as
# (line, target) pairs. URLs, mailto and bare anchors within the file are
# excluded here; an in-file anchor is checked against the file's own headers.
function relative_links(path)
    out = Tuple{Int,String}[]
    infence = false
    for (i, line) in enumerate(eachline(path))
        startswith(rstrip(line), "```") && (infence = !infence; continue)
        infence && continue
        for m in eachmatch(r"\]\(([^)\s]+)\)", line)
            target = m.captures[1]
            occursin(r"^[a-z]+:", target) && continue
            startswith(target, '@') && continue   # a Documenter `@ref` in prose
            push!(out, (i, String(target)))
        end
    end
    return out
end

function broken_links()
    out = String[]
    cache = Dict{String,Set{String}}()
    for f in linked_files()
        for (i, target) in relative_links(f)
            file, frag = occursin('#', target) ? split(target, '#'; limit=2) :
                                                (target, nothing)
            dest = isempty(file) ? f : normpath(joinpath(dirname(f), file))
            if !isfile(dest)
                push!(out, "$(rel(f)):$i: $target (no file $(rel(dest)))")
                continue
            end
            frag === nothing && continue
            endswith(dest, ".md") || continue
            heads = get!(() -> anchors(dest), cache, dest)
            String(frag) in heads ||
                push!(out, "$(rel(f)):$i: $target (no header #$frag in $(rel(dest)))")
        end
    end
    return out
end

# Reference prose states its findings in words and keeps its figures in
# tables: a figure in a sentence is a measurement recited, which a reader
# cannot compare across cases and which later prose copies. A sentence
# outside a table, a fenced block or a heading may therefore carry at most
# NUMERAL_LIMIT numerals. A numeral is a maximal number token (integer,
# decimal, exponent form, with an optional sign and percent sign) that no
# letter, digit or underscore precedes: "C10" and "N32" hold none, "N = 256"
# holds one, "64²" one, "1e-14" one. A number written as a word is not a
# numeral, and an inline code span counts as one word.
const NUMERAL_LIMIT = 2
const NUMERAL = r"(?<![A-Za-z_0-9])[−+-]?\d+(?:\.\d+)?(?:e[+−-]?\d+)?%?"

# Reference files whose prose predates the rule; the list empties as each is
# converted.
const NUMERAL_PENDING = ["CALIBRATION_APPENDIX.md"]
# A local file the repository ignores (`.gitignore`), present in a maintainer's
# checkout only: questions to a paper's authors, which quote its figures.
const NUMERAL_LOCAL = ["AUTHOR_QUESTIONS.md"]

# The numbers of a citation do not count. Each pattern replaces one form by a
# word: a link (its text and a target holding one level of parentheses), an
# arXiv identifier, the number of an equation, figure, table, theorem, section,
# page or appendix, a journal's abbreviated title with its volume and pages
# ("Phys. Rev. Lett. 107, 115004", "Phys. Fluids 23, 024102, 2011"), a
# journal acronym with its volume and year ("JCP 257, 2014"), a report number,
# and the year after an author's name.
const CITED_LABEL = raw"(?:[Ee]qs?\.|[Ee]quations?|[Ff]igs?\.|[Ff]igures?|[Tt]ables?|" *
                    raw"[Tt]heorems?|[Ll]emmas?|[Ss]ections?|§+|pp?\.|[Aa]ppendix|" *
                    raw"[Cc]hapters?)"
const CITED_NUMBER = raw"[A-Z]?\d+(?:\.\d+)*"
const CITATIONS = [
    r"\[[^\]]*\]\((?:[^()]|\([^()]*\))*\)" => "Link",
    r"arXiv:\S+" => "arXiv",
    Regex(CITED_LABEL * raw"\s*" * CITED_NUMBER *
          raw"(?:\s*(?:–|-|and|,)\s*" * CITED_NUMBER * ")*") => "Ref",
    Regex(raw"(?:[A-Z][A-Za-z]*\.\s+){2,}\d+(?:\s*\(\d{4}\))?" *
          raw"(?:,\s*\d+(?:[–-]\d+)?)?") => "Journal",
    Regex(raw"\b(?:[A-Z][a-z]*\.\s+)+[A-Z][a-z]+\s+\d+,\s*\d+" *
          raw"(?:,\s*(?:19|20)\d\d)?") => "Journal",
    r"\b[A-Z]{2,}\s+\d+,\s*(?:19|20)\d\d\b" => "Journal",
    r"\b[A-Z]{3,}\s\d+[-–]\d+\b" => "Report",
    r"((?:[A-Z][\w'’–-]+|al\.),?\s+)\(?(?:19|20)\d\d[a-z]?\)?" => s"\1Year",
]

const LIST_ITEM = r"^(?:[-*] |\d+\. )"

# The prose paragraphs of a file as (first line, joined text): lines outside
# fenced blocks that are not blank, table rows or headings, each list item
# opening a paragraph of its own.
function prose_paragraphs(path)
    out = Tuple{Int,String}[]
    buf = String[]
    start = 0
    fenced = false
    function flush!()
        isempty(buf) || push!(out, (start, join(buf, " ")))
        empty!(buf)
    end
    for (i, line) in enumerate(eachline(path))
        s = strip(line)
        if startswith(s, "```")
            fenced = !fenced
            flush!()
            continue
        end
        fenced && continue
        if isempty(s) || startswith(s, "|") || startswith(s, "#")
            flush!()
            continue
        end
        occursin(LIST_ITEM, s) && flush!()
        isempty(buf) && (start = i)
        push!(buf, s)
    end
    flush!()
    return out
end

# A sentence ends at ".", ";" or ":" followed by white space and an upper-case
# letter, a backtick or an opening parenthesis; a code span is replaced before
# the split, so a sentence opening with one still splits from the one before.
function numeral_sentences(path)
    out = String[]
    for (start, p) in prose_paragraphs(path)
        q = replace(p, r"`[^`]*`" => "Code")
        for c in CITATIONS
            q = replace(q, c)
        end
        q = replace(q, LIST_ITEM => "")
        for s in split(q, r"(?<=[.;:])\s+(?=[A-Z`(])")
            n = count(_ -> true, eachmatch(NUMERAL, s))
            n > NUMERAL_LIMIT &&
                push!(out, "$(rel(path)):$start: $n numerals: $(first(s, 70))")
        end
    end
    return out
end

function numeral_sentences()
    out = String[]
    for f in readdir(REFERENCE; join=true)
        name = basename(f)
        endswith(name, ".md") && !(name in NUMERAL_PENDING) &&
            !(name in NUMERAL_LOCAL) || continue
        append!(out, numeral_sentences(f))
    end
    return out
end

end # module ReferenceLayout

@testset "reference layout" begin
    RL = ReferenceLayout
    exponents = RL.exponent_numbers()
    for line in exponents
        @warn "measurement outside the appendix: $line"
    end
    @test isempty(exponents)

    cap, n = RL.calibration_cap()
    cap === nothing && @warn "CALIBRATION.md states no line cap in its header"
    @test cap !== nothing
    cap !== nothing && n > cap &&
        @warn "CALIBRATION.md is $n lines against its cap of $cap"
    @test cap === nothing || n <= cap

    long = RL.long_closed_entries()
    for line in long
        @warn "closed ROADMAP entry over $(RL.CLOSED_ENTRY_LINES) lines: $line"
    end
    @test isempty(long)

    superseded = RL.supersede_lines()
    for line in superseded
        @warn "annotated instead of deleted: $line"
    end
    @test isempty(superseded)

    broken = RL.broken_links()
    for line in broken
        @warn "broken reference link: $line"
    end
    @test isempty(broken)

    recited = RL.numeral_sentences()
    for line in recited
        @warn "figures recited in prose: $line"
    end
    @test isempty(recited)
end
