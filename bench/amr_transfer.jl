# Conditioning audit for Miranda's disabled 3:1 AMR transfer pair, and the
# measurement battery for the live implementation in src/transfer.jl.
#
# Part 1 (audit): coefficients and one-sided closure rows are transcribed from
# LLNL/pyranda@b4e0afc, pyranda/parcop/stencils.f90, cfamrcf/cfamrfc.
# The coarse-to-fine operator is B \ A (deconvolution); fine-to-coarse is
# A \ B (filtering), where A is the compact tridiagonal side and B is the
# five-diagonal Gaussian side. The source constructs the two operations by
# swapping these matrices, including their closure rows.
#
# Part 2 (implementation) drives the same pair through the CompactScheme /
# BandedCompactScheme types and the TransferPlan sampling convention: pair
# invertibility through the plans, coarse → fine → coarse exactness,
# fine → coarse → fine measured order against the interpolation order, the
# edge/interior error split at closed ends, and the sensor-injection
# amplification sweep that sizes the coarse-fine interface buffers
# (reference/AMR_GPU.md, transfer operators).
#
# Part 3 (closure localization) measures constraint 7 of reference/AMR_GPU.md
# per derivative scheme: how fast the response of the compact solve to an
# error in the first ghost layer decays into the patch, through the
# interface closure rows and the interior left-hand side, against the root
# of the LHS symbol that sets the asymptotic rate. C10's inverse decays more
# slowly than C6's, and this is the number the interface-adjacent buffers
# are sized from.

using MPI
MPI.Init(threadlevel=:funneled)
using CompactLES
using LinearAlgebra
using Printf

const CL = CompactLES

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

# --- Part 2: the live implementation ----------------------------------------

"Fill the interior of a padded field with fn(x) along `dim`, x = x0 + (i-1)h."
function fill_line!(f, decomp, dim, fn, h)
    pad = decomp.n_halo_d
    for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        f[i+pad[1], j+pad[2], k+pad[3]] = fn(((i, j, k)[dim] - 1) * h)
    end
    return f
end

"Max interior difference of two same-shape padded fields."
function max_diff(f, g, decomp)
    pad = decomp.n_halo_d
    e = 0.0
    for k in 1:decomp.n_local[3], j in 1:decomp.n_local[2], i in 1:decomp.n_local[1]
        e = max(e, abs(f[i+pad[1], j+pad[2], k+pad[3]] - g[i+pad[1], j+pad[2], k+pad[3]]))
    end
    return e
end

"Pair invertibility through the plans: prolong(restrict(f)) with no sampling."
function pair_roundtrip(n=96; periodic=false)
    d = Decomp((n, 4, 4), (periodic, true, true); dims=(1, 1, 1))
    restriction, prolongation = amr_transfer_schemes()
    restrict_plan = CL.plan_direction(d, restriction, 1, 1)
    prolong_plan = CL.plan_direction(d, prolongation, 1, 1)
    f = field(d); fbar = field(d); back = field(d)
    fill_line!(f, d, 1, x -> sin(5x) + 0.4cos(11x) + 0.1sin(29x), 2pi / n)
    CL.exchange_dim!(f, d, 1)
    apply_along!(fbar, restrict_plan, f, d)
    CL.exchange_dim!(fbar, d, 1)
    apply_along!(back, prolong_plan, fbar, d)
    return max_diff(f, back, d)
end

function convention_checks()
    println()
    println("sampling-convention checks (TransferPlan)")
    @printf("  pair round-trip, closed line, no sampling    %.3e\n",
            pair_roundtrip(96; periodic=false))
    @printf("  pair round-trip, periodic line, no sampling  %.3e\n",
            pair_roundtrip(96; periodic=true))
    for periodic in (true, false)
        nc = 32
        nf = periodic ? 3nc : 3nc - 2
        df = Decomp((nf, 4, 4), (periodic, true, true); dims=(1, 1, 1))
        dc = Decomp((nc, 4, 4), (periodic, true, true); dims=(1, 1, 1))
        tp = plan_transfer(df, dc, 1)
        hc = periodic ? 2pi / nc : 1.0 / (nc - 1)
        coarse = field(dc); coarse2 = field(dc); fine = field(df)
        fill_line!(coarse, dc, 1, x -> sin(5x) + exp(sin(3x)) + x, hc)
        prolong!(fine, tp, coarse)
        restrict!(coarse2, tp, fine)
        @printf("  coarse->fine->coarse, %-8s rough data    %.3e\n",
                periodic ? "periodic" : "closed", max_diff(coarse, coarse2, dc))
    end
end

"Fine → coarse → fine error against the interpolation order, periodic."
function roundtrip_orders()
    println()
    println("fine->coarse->fine round trip, periodic, f = sin(2x) + cos(3x)")
    println("  n_coarse   p=4          p=6          p=8")
    prev = Dict{Int,Float64}()
    for nc in (16, 32, 64, 128)
        nf = 3nc
        errs = Float64[]
        for p in (4, 6, 8)
            df = Decomp((nf, 4, 4), (true, true, true); dims=(1, 1, 1))
            dc = Decomp((nc, 4, 4), (true, true, true); dims=(1, 1, 1))
            tp = plan_transfer(df, dc, 1; interp_order=p)
            fine = field(df); fine2 = field(df); coarse = field(dc)
            fill_line!(fine, df, 1, x -> sin(2x) + cos(3x), 2pi / nf)
            restrict!(coarse, tp, fine)
            prolong!(fine2, tp, coarse)
            push!(errs, max_diff(fine, fine2, df))
        end
        rates = [haskey(prev, p) ? log2(prev[p] / errs[i]) : NaN
                 for (i, p) in enumerate((4, 6, 8))]
        @printf("  %6d   %.3e", nc, errs[1])
        isnan(rates[1]) || @printf(" (%.2f)", rates[1])
        @printf("  %.3e", errs[2])
        isnan(rates[2]) || @printf(" (%.2f)", rates[2])
        @printf("  %.3e", errs[3])
        isnan(rates[3]) || @printf(" (%.2f)", rates[3])
        println()
        for (i, p) in enumerate((4, 6, 8))
            prev[p] = errs[i]
        end
    end
end

"Closed-end error split: within 6 fine points of an end versus the interior."
function edge_split(; p=6)
    println()
    println("closed fine->coarse->fine, Gaussian bump, edge (6 pts) vs interior, p=$p")
    println("  n_coarse   edge         interior")
    for nc in (16, 32, 64, 128)
        nf = 3nc - 2
        df = Decomp((nf, 4, 4), (false, true, true); dims=(1, 1, 1))
        dc = Decomp((nc, 4, 4), (false, true, true); dims=(1, 1, 1))
        tp = plan_transfer(df, dc, 1; interp_order=p)
        fine = field(df); fine2 = field(df); coarse = field(dc)
        hf = 1.0 / (nf - 1)
        fill_line!(fine, df, 1, x -> exp(-20(x - 0.5)^2), hf)
        restrict!(coarse, tp, fine)
        prolong!(fine2, tp, coarse)
        pad = df.n_halo_d
        e_edge = e_int = 0.0
        for i in 1:nf
            e = abs(fine[i+pad[1], pad[2]+1, pad[3]+1] -
                    fine2[i+pad[1], pad[2]+1, pad[3]+1])
            if i <= 6 || i > nf - 6
                e_edge = max(e_edge, e)
            else
                e_int = max(e_int, e)
            end
        end
        @printf("  %6d   %.3e    %.3e\n", nc, e_edge, e_int)
    end
end

# Sensor injection: a captured-shock-like profile at distance `dist` fine
# cells from the high end of the line (a stand-in for a patch interface: the
# extended-data closure rows are the one-sided ones). The Cook δ⁴ chain
# produces the smoothed sensor; both the profile and its sensor make the
# fine → coarse → fine round trip, and the table reports the amplification of
# each and the undershoot the deconvolution leaves in the state field.
function sensor_injection()
    nc = 33
    nf = 3nc - 2                        # 97
    solver = Solver(; n_global=(nf, 1, 1), L_domain=(1.0, 1.0, 1.0),
                    bcs=((ExtrapolationBC(), ExtrapolationBC()),
                         (PeriodicBC(), PeriodicBC()), (PeriodicBC(), PeriodicBC())))
    df = solver.decomp
    dc = Decomp((nc, 1, 1), (false, true, true); dims=(1, 1, 1))
    tp = plan_transfer(df, dc, 1)
    hf = 1.0 / (nf - 1)
    rho = field(df); rho2 = field(df); coarse = field(dc)
    sensor2 = field(df)
    println()
    println("sensor injection at a transfer end (n_fine = $nf, tanh over 2h)")
    println("  dist   max|RT(rho)-rho|  undershoot   max s      max|RT(s)|  amp(s)")
    for dist in (3, 6, 9, 12, 18, 30)
        xs = 1.0 - dist * hf
        fill_line!(rho, df, 1, x -> 1.0 + 0.5 * (1 + tanh((x - xs) / (2hf))), hf)
        # state-field round trip
        restrict!(coarse, tp, rho)
        prolong!(rho2, tp, coarse)
        e_rho = max_diff(rho, rho2, df)
        pad = df.n_halo_d
        under = minimum(rho2[pad[1]+1:pad[1]+nf, pad[2]+1, pad[3]+1]) - 1.0
        # Cook δ⁴ sensor of the profile, smoothed, then the same round trip
        CL.exchange_halos!(rho, df)
        CL.delta4_sum!(solver.sensor, rho, solver, 1)
        CL.smooth!(solver.sensor, solver)
        smax = maximum(abs, solver.sensor)
        restrict!(coarse, tp, solver.sensor)
        prolong!(sensor2, tp, coarse)
        s2max = maximum(abs,
            sensor2[pad[1]+1:pad[1]+nf, pad[2]+1, pad[3]+1])
        @printf("  %4d   %.3e         %+.3e   %.3e  %.3e   %.2f\n",
                dist, e_rho, under, smax, s2max, s2max / smax)
    end
end

# Does an extra anti-aliasing pass (the c4ff3 question) improve the shock
# round trip? The transfer filter already smooths before subsampling; this
# adds one explicit Gaussian pass ahead of it and compares error and
# undershoot on the same mid-domain shock profile.
function prefilter_comparison()
    nc = 33
    nf = 3nc - 2
    df = Decomp((nf, 1, 1), (false, true, true); dims=(1, 1, 1))
    dc = Decomp((nc, 1, 1), (false, true, true); dims=(1, 1, 1))
    tp = plan_transfer(df, dc, 1)
    gauss_plan = CL.plan_direction(df, gaussian_filter(), 1, 1)
    hf = 1.0 / (nf - 1)
    rho = field(df); rho2 = field(df); rho_g = field(df); coarse = field(dc)
    fill_line!(rho, df, 1, x -> 1.0 + 0.5 * (1 + tanh((x - 0.5) / (2hf))), hf)
    pad = df.n_halo_d
    println()
    println("anti-aliasing prefilter ahead of restriction (mid-domain shock)")
    for prefilter in (false, true)
        src = rho
        if prefilter
            CL.exchange_dim!(rho, df, 1)
            apply_along!(rho_g, gauss_plan, rho, df)
            src = rho_g
        end
        restrict!(coarse, tp, src)
        prolong!(rho2, tp, coarse)
        e = max_diff(rho, rho2, df)
        under = minimum(rho2[pad[1]+1:pad[1]+nf, pad[2]+1, pad[3]+1]) - 1.0
        @printf("  %-11s max|RT-rho| %.3e   undershoot %+.3e\n",
                prefilter ? "gaussian" : "plain", e, under)
    end
end

# The α^|i−j| localization (constraint 7 of reference/AMR_GPU.md): how fast
# the round-trip
# error of a mid-domain shock decays with distance from it, in fine cells.
# This is the measured basis for interface-adjacent buffer widths.
function pollution_decay()
    nc = 65
    nf = 3nc - 2
    df = Decomp((nf, 1, 1), (false, true, true); dims=(1, 1, 1))
    dc = Decomp((nc, 1, 1), (false, true, true); dims=(1, 1, 1))
    tp = plan_transfer(df, dc, 1)
    hf = 1.0 / (nf - 1)
    rho = field(df); rho2 = field(df); coarse = field(dc)
    fill_line!(rho, df, 1, x -> 1.0 + 0.5 * (1 + tanh((x - 0.5) / (2hf))), hf)
    restrict!(coarse, tp, rho)
    prolong!(rho2, tp, coarse)
    pad = df.n_halo_d
    ic = (nf + 1) ÷ 2
    println()
    println("round-trip error decay away from a mid-domain shock (n_fine = $nf)")
    println("  cells from shock   max error")
    for band in ((0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 20), (21, 40))
        e = 0.0
        for i in 1:nf
            d = abs(i - ic)
            band[1] <= d <= band[2] || continue
            e = max(e, abs(rho[i+pad[1], pad[2]+1, pad[3]+1] -
                           rho2[i+pad[1], pad[2]+1, pad[3]+1]))
        end
        @printf("  %4d–%-4d          %.3e\n", band[1], band[2], e)
    end
end

# --- Part 3: closure localization per derivative scheme --------------------

# The asymptotic decay rate of a compact scheme's LHS inverse: the root of
# smallest modulus below one of the symbol 1 + Σ_s 2 lhs[s] cos(s k) read as a
# polynomial in z = e^{ik}, so that (A⁻¹)_{ij} ~ r^{|i−j|}.
function lhs_decay_rate(scheme)
    lhs = scheme isa CL.BandedCompactScheme ? scheme.lhs : [scheme.alpha]
    q = length(lhs)
    # Coefficients of z^{2q} · symbol(z): lhs[q], …, lhs[1], 1, lhs[1], …, lhs[q].
    c = vcat(reverse(lhs), 1.0, lhs)
    # Companion-matrix roots of the monic polynomial.
    c = c ./ c[end]
    m = length(c) - 1
    C = zeros(m, m)
    for i in 2:m
        C[i, i-1] = 1.0
    end
    C[:, m] .= -c[1:m]
    roots = eigvals(C)
    return maximum(abs(z) for z in roots if abs(z) < 1 - 1e-9)
end

# Response of the derivative to a unit error in the first ghost layer of a
# closed line under the interface closure rows: the RHS is nonzero only where
# a stencil reaches that ghost, and the solve spreads it at the LHS inverse's
# rate. The table gives |response| per point and the measured per-point ratio
# over points 8–20, where the stencil footprint has ended.
function closure_localization()
    println()
    println("closure localization: derivative response to a unit error in ghost layer 1")
    println("  scheme   rate(theory)   |r| at 1      4         8         12        16",
            "        20     ratio/pt")
    n = 64
    for (label, scheme) in (("C6", lele_d1_6()), ("C8", lele_d1_8()),
                            ("C10", lele_d1_10()))
        n_halo = 1 - minimum(r.first for r in CL.interface_closures(scheme))
        d = Decomp((n, 4, 4), (false, true, true); dims=(1, 1, 1), n_halo=n_halo)
        rows = CL.interface_closures(scheme)
        plan = CL.plan_direction(d, scheme, 1, 1.0; lo_closures=rows, hi_closures=rows)
        f = field(d); df = field(d)
        pad = d.n_halo_d
        f[pad[1], pad[2]+1, pad[3]+1] = 1.0          # ghost layer 1, index 0
        apply_along!(df, plan, f, d)
        r = [abs(df[i+pad[1], pad[2]+1, pad[3]+1]) for i in 1:n]
        ratio = (r[8] / r[20])^(1 / 12)
        @printf("  %-6s   %.4f         %.2e  %.2e  %.2e  %.2e  %.2e  %.2e   %.3f\n",
                label, lhs_decay_rate(scheme), r[1], r[4], r[8], r[12], r[16], r[20],
                1 / ratio)
    end
end

convention_checks()
roundtrip_orders()
edge_split()
sensor_injection()
prefilter_comparison()
pollution_decay()
closure_localization()
