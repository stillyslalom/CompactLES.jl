# Conditioning audit for Miranda's disabled 3:1 AMR transfer pair.
#
# Coefficients and one-sided closure rows are transcribed from
# LLNL/pyranda@b4e0afc, pyranda/parcop/stencils.f90, cfamrcf/cfamrfc.
# The coarse-to-fine operator is B \ A (deconvolution); fine-to-coarse is
# A \ B (filtering), where A is the compact tridiagonal side and B is the
# five-diagonal Gaussian side. The source constructs the two operations by
# swapping these matrices, including their closure rows.

using LinearAlgebra
using Printf

const ALPHA_AMR = -0.0321826755129339
const A_AMR = 0.4451523642186118
const B_AMR = 0.2207614172195584
const C_AMR = 0.0244797251582018

const COMPACT_INTERIOR = (0.0, ALPHA_AMR, 1.0, ALPHA_AMR, 0.0)
const GAUSSIAN_INTERIOR = (C_AMR, B_AMR, A_AMR, B_AMR, C_AMR)
const COMPACT_LOWER = (
    (0.0, 0.0, 1.0, 0.0, 0.0),
    COMPACT_INTERIOR,
    COMPACT_INTERIOR,
)
const GAUSSIAN_LOWER = (
    (0.0, 0.0, 1.375, -0.75, 0.375),
    (0.0, 0.3186803178523657, 0.2982740132694008,
     0.3186803178523657, 0.0),
    GAUSSIAN_INTERIOR,
)

"Dense band matrix with the three one-sided rows used by the AMR pair."
function closure_matrix(n, interior, lower)
    n >= 7 || error("need at least seven points for the three-row closures")
    M = zeros(n, n)
    for i in 1:n
        weights = i <= 3 ? lower[i] :
                  i > n - 3 ? reverse(lower[n - i + 1]) : interior
        for offset in -2:2
            j = i + offset
            1 <= j <= n && (M[i, j] = weights[offset + 3])
        end
    end
    return M
end

"Periodic transfer symbol: coarse-to-fine gain at nondimensional wavenumber k."
function prolongation_symbol(k)
    compact = 1 + 2ALPHA_AMR * cos(k)
    gaussian = A_AMR + 2B_AMR * cos(k) + 2C_AMR * cos(2k)
    return compact / gaussian
end

function audit(n=96)
    compact = closure_matrix(n, COMPACT_INTERIOR, COMPACT_LOWER)
    gaussian = closure_matrix(n, GAUSSIAN_INTERIOR, GAUSSIAN_LOWER)
    prolong = gaussian \ compact
    restrict = compact \ gaussian

    kfine = range(0, pi; length=100_001)
    gain = prolongation_symbol.(kfine)
    coarse_band = kfine .<= pi / 3
    roundtrip = prolong * restrict
    edge_mode = svd(prolong).V[:, 1]
    edge_fraction = sum(abs2, edge_mode[[1:6; n-5:n]]) / sum(abs2, edge_mode)
    edge_noise = [isodd(i) && i <= 6 ? 1.0 :
                  iseven(i) && i <= 6 ? -1.0 : 0.0 for i in 1:n]
    mid = n ÷ 2
    interior_noise = [mid - 2 <= i <= mid + 3 ? (-1.0)^i : 0.0 for i in 1:n]
    edge_supported_gain = opnorm(prolong[:, 1:6])
    interior_supported_gain = opnorm(prolong[:, mid-2:mid+3])

    @printf("Miranda 3:1 compact AMR transfer conditioning (n = %d)\n", n)
    @printf("  DC gain                                  %.16f\n",
            prolongation_symbol(0.0))
    @printf("  gain at coarse Nyquist (k = pi/3)       %.8f\n",
            prolongation_symbol(pi / 3))
    @printf("  max gain in representable coarse band   %.8f\n",
            maximum(gain[coarse_band]))
    @printf("  gain at fine Nyquist (k = pi)           %.8f\n",
            prolongation_symbol(pi))
    @printf("  cond(compact side A)                     %.8f\n", cond(compact))
    @printf("  cond(Gaussian side B, closures)          %.8f\n", cond(gaussian))
    @printf("  cond(prolongation B\\A)                   %.8f\n", cond(prolong))
    @printf("  sigma_max(prolongation B\\A)             %.8f\n", opnorm(prolong))
    @printf("  boundary energy of max-gain mode         %.6f\n", edge_fraction)
    @printf("  max gain of noise supported in 6 edge pts %.8f\n", edge_supported_gain)
    @printf("  max gain of noise in 6 interior pts       %.8f\n", interior_supported_gain)
    @printf("  gain of edge-local alternating noise      %.8f\n",
            norm(prolong * edge_noise) / norm(edge_noise))
    @printf("  gain of interior alternating noise        %.8f\n",
            norm(prolong * interior_noise) / norm(interior_noise))
    @printf("  ||prolong*restrict - I||_inf              %.3e\n",
            opnorm(roundtrip - I, Inf))
end

audit()
