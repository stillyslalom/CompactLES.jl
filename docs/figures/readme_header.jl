# Draw the README header from the frames of docs/figures/readme_header_run.jl.
#
# The header is a space-time strip of the SF6 mole fraction in the x-y mid-plane,
# air on top. Each pixel column is one y position, advancing across the periodic
# span `periods` times over the width of the strip, at a time rising as
# s^`power` (s the fraction of the width) from impact to the last frame; a column
# between two frames is interpolated linearly in time. Each frame is shifted in x
# to keep the mixing layer centred in a window of ±`half`. From the checkout root:
#
#   julia --project=docs docs/figures/readme_header.jl [run=<run dir>] [out=<png>]

using CairoMakie, Serialization, Statistics
using CompactLES: script_args

const opt = script_args(ARGS, (run = joinpath(@__DIR__, ".readme_header_run"),
                               out = normpath(joinpath(@__DIR__, "..", "src", "assets",
                                                       "readme_header.png")),
                               periods = 4, width = 3200, height = 800, half = 0.030,
                               power = 1.3))

function main(run, out; P, W, H, half, power)
    meta = deserialize(joinpath(run, "meta.jls"))
    fr = [deserialize(joinpath(run, "frames", f))
          for f in sort(readdir(joinpath(run, "frames")))]
    nx, ny, hx, Ly = meta.nx, meta.ny, meta.h, meta.Ly
    xs = range(0, meta.Lx, length = nx)
    sf6(g) = g.xy.X[:, :, 1, 3]
    TF = [g.t_after for g in fr]

    # Window centre: the 50% crossing of the plane mean plus a quadratic fit in
    # time of the layer midpoint (2%–98% of the plane mean) relative to it.
    xc = map(fr) do g
        m = vec(mean(sf6(g), dims = 2))
        i = findfirst(k -> (m[k] - 0.5) * (m[k+1] - 0.5) <= 0, 1:nx-1)
        xs[i] + (0.5 - m[i]) / (m[i+1] - m[i]) * hx
    end
    mids = map(fr) do g
        m = vec(mean(sf6(g), dims = 2))
        (xs[findfirst(>(0.02), m)] + xs[findlast(<(0.98), m)]) / 2
    end
    A = hcat(ones(length(TF)), TF, TF .^ 2)
    cwin = xc .+ A * (A \ (mids .- xc))

    dx = range(-half, half, length = H)
    function centered(F, c)
        block = zeros(H, ny)
        for (v, d) in enumerate(dx)
            f = (c + d) / hx + 1
            i = clamp(floor(Int, f), 1, nx - 1); s = clamp(f - i, 0, 1)
            @views block[v, :] .= (1 - s) .* F[i, :] .+ s .* F[i+1, :]
        end
        block
    end
    cache = Dict{Int,Matrix{Float64}}()
    col(k) = get!(() -> centered(sf6(fr[k]), cwin[k]), cache, k)
    B = zeros(H, W)
    for u in 1:W
        s = (u - 0.5) / W
        fk = clamp(s^power * TF[end] / (TF[2] - TF[1]) + 1, 1, length(fr))
        k = min(floor(Int, fk), length(fr) - 1); a = fk - k
        y = mod(s * P * Ly, Ly) / Ly * ny + 1
        j = floor(Int, y); b = y - j; j1, j2 = mod1(j, ny), mod1(j + 1, ny)
        F1, F2 = col(k), col(k + 1)
        @views B[:, u] .= (1 - a) .* ((1 - b) .* F1[:, j1] .+ b .* F1[:, j2]) .+
                          a .* ((1 - b) .* F2[:, j1] .+ b .* F2[:, j2])
    end

    # Navy air, maroon SF6, the interface lit white.
    cmap = cgrad(["#06122b", "#0b2c5c", "#2f8fd0", "#bfe6f7", "#fff8ea", "#f6b45c",
                  "#c2452a", "#6a1a0f", "#2a0b08"],
                 [0, 0.18, 0.36, 0.46, 0.5, 0.56, 0.68, 0.85, 1.0])
    fig = Figure(size = (W, H), figure_padding = 0)
    ax = Axis(fig[1, 1]); hidedecorations!(ax); hidespines!(ax)
    image!(ax, permutedims(reverse(B, dims = 1)), colormap = cmap, colorrange = (0, 1),
           interpolate = true)
    tightlimits!(ax)
    save(out, fig; px_per_unit = 1)
end

main(opt.run, opt.out; P = opt.periods, W = opt.width, H = opt.height, half = opt.half,
     power = opt.power)
