# The test suites' precompiled configurations (test/TestPrecompile).
#
# Included by a suite, this loads the helper package where the checkout has
# enabled it, and does nothing otherwise. Run as a script it enables the
# package, with the blocks named, and builds its image:
#
#     julia --project=. test/test_image.jl [device=true] [float32=true] [mpi_suite=false]
#
# Run that once before a serial or multi-rank suite, in one process and with
# the flags the suite's processes take: the ranks of an MPI launch would
# otherwise wait on one precompile lock at their first `using`. The image
# follows CompactLES's, so a change under `src/` rebuilds it at the next load.
# Delete `test/TestPrecompile/LocalPreferences.toml` to turn it off, in which
# case a suite compiles its configurations itself.

const TEST_IMAGE_DIR = joinpath(@__DIR__, "TestPrecompile")
const TEST_IMAGE_PREFERENCES = joinpath(TEST_IMAGE_DIR, "LocalPreferences.toml")

function use_test_image()
    isfile(TEST_IMAGE_PREFERENCES) || return false
    TEST_IMAGE_DIR in LOAD_PATH || push!(LOAD_PATH, TEST_IMAGE_DIR)
    @eval Main using TestPrecompile
    return true
end

function enable_test_image(args)
    opt = Main.CompactLES.script_args(args, (device=true, float32=true, mpi_suite=false))
    write(TEST_IMAGE_PREFERENCES,
          "[TestPrecompile]\n" * join(("$k = $v\n" for (k, v) in pairs(opt))))
    return use_test_image()
end

if abspath(PROGRAM_FILE) == @__FILE__
    using CompactLES
    enable_test_image(ARGS)
else
    use_test_image()
end
