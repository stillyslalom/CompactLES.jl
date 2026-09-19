# Fit residuals of the vendored Marrero--Mason binary-diffusion correlations.
# Run with: julia --project=. -t 1 bench/neutraldiffusion.jl [degree=10]
#
# For every pair of MARRERO_MASON_1972, fit the log-polynomial over the
# pair's full stated range at the given degree and report the largest
# relative departure from the source equation, together with the source
# value at 300 K and 1 atm. The residuals are the numbers behind the gate
# in test/neutral_diffusion_tests.jl; the source's own uncertainty limits
# are MARRERO_MASON_UNCERTAINTY. Pairs whose range excludes 300 K report
# the value at the nearest bound instead. Each fit is anchored at its range's
# geometric midpoint so overlapping rows select their own correlation.
using CompactLES
using Printf

opt = CompactLES.script_args(ARGS, (degree=10,); positional=(:degree,))

function main(degree)
    println("pair            group  range [K]       D(300 K, 1 atm) [cm^2/s]   max residual")
    worst = 0.0
    for pair in MARRERO_MASON_1972
        model = neutral_binary_diffusion(pair.species; degree=degree,
                                         temperature_ref=sqrt(pair.temperature_min *
                                                              pair.temperature_max),
                                         temperature_min=pair.temperature_min,
                                         temperature_max=pair.temperature_max)
        residual = neutral_binary_diffusion_residual(model, 1, 2)
        worst = max(worst, residual)
        T = clamp(300.0, pair.temperature_min, pair.temperature_max)
        D = marrero_mason_diffusivity(pair, T, 101325.0) * 1e4
        @printf("%-15s %-6s %6.0f-%-8.0f %8.4f at %6.0f K        %.2e\n",
                join(pair.species, "-"), pair.group, pair.temperature_min,
                pair.temperature_max, D, T, residual)
    end
    @printf("largest residual at degree %d: %.2e over %d pairs\n", degree, worst,
            length(MARRERO_MASON_1972))
end

main(opt.degree)
