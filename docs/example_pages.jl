# The pages an example script in examples/ gives the documentation. A script
# marks the code that reduces its runs to tables and figures, and the prose that
# describes the script itself (its settings, the smoke run), by enclosing it
# between a `#region` line and an `#endregion` line, which editors fold. The
# example's page leaves those lines out; a second page, built from the whole
# script and kept out of the navigation, is linked from the first. A script
# without regions gives one page. docs/make.jl builds the pages, and
# test/docrefs_tests.jl reads the same text to resolve the links on them, so
# both include this file. It needs nothing beyond Base.

const REGION_OPEN = r"^\h*#region\b"
const REGION_CLOSE = r"^\h*#endregion\b"
const COMPLETE_TITLE = ": complete script"

"Whether an example script marks any region, and so gives two pages."
has_regions(source::AbstractString) =
    any(l -> occursin(REGION_OPEN, l), eachline(IOBuffer(source)))

"""
    example_source(source; complete, keep_lines = false) -> String

The script text Literate converts into one of an example's pages. The marker
lines are dropped from both. The example's page (`complete = false`) drops the
lines between them as well; the complete page keeps them and appends
`$(repr(COMPLETE_TITLE))` to the `# # Title` line, so that the two pages'
titles differ. With `keep_lines` every dropped line leaves an empty line, so
that a line of the result is the same line of the script in a report.
"""
function example_source(source::AbstractString; complete::Bool, keep_lines::Bool = false)
    out = IOBuffer()
    inside = false
    titled = false
    for (n, line) in enumerate(eachline(IOBuffer(source); keep = true))
        drop = false
        if occursin(REGION_OPEN, line)
            inside && error("line $n: a #region inside another")
            inside = drop = true
        elseif occursin(REGION_CLOSE, line)
            inside || error("line $n: an #endregion without its #region")
            inside = false
            drop = true
        elseif inside && !complete
            drop = true
        end
        if drop
            keep_lines && print(out, endswith(line, "\r\n") ? "\r\n" : "\n")
        elseif complete && !titled && startswith(line, "# # ")
            body = rstrip(line, ('\r', '\n'))
            print(out, body, COMPLETE_TITLE, line[nextind(line, lastindex(body)):end])
            titled = true
        else
            print(out, line)
        end
    end
    inside && error("a #region without its #endregion")
    return String(take!(out))
end
