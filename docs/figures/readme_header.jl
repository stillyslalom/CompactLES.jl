# Generate the README capability figure from three real CompactLES runs.
#
# The plotting dependency deliberately lives in the user's base environment,
# not CompactLES's Project.toml:
#
#   julia --project=. -t auto docs/figures/readme_header.jl
#
# Set CL_HERO_REFRESH=1 to rerun all simulations, or CL_HERO_REFRESH_TG=1 to
# rerun only Taylor–Green, instead of using the local ignored cache.

using MPI
MPI.Init(threadlevel=:funneled)

using CompactLES
using GLMakie
using Serialization
using Statistics

const FIGURE_DIR = @__DIR__
const ASSET_DIR = normpath(joinpath(FIGURE_DIR, "..", "src", "assets"))
const CACHE_PATH = joinpath(FIGURE_DIR, ".readme_header_cache.jls")
const OUTPUT_PATH = joinpath(ASSET_DIR, "readme_header.png")

function taylor_green_data(; n=64, tfinal=12.0)
    println("Running Taylor–Green showcase: $(n)^3 to t=$tfinal")
    gamma = 1.4
    c0 = 10.0
    problem = Problem(
        name="Taylor–Green showcase",
        eos=single_species(gamma=gamma),
        transport=Transport(mu0=1 / 1600),
        domain=((0.0, 2π), (0.0, 2π), (0.0, 2π)),
        bcs=ntuple(_ -> (PeriodicBC(), PeriodicBC()), 3),
        ic=(x, y, z) -> Prim(
            u=(sin(x) * cos(y) * cos(z),
               -cos(x) * sin(y) * cos(z),
               0.0),
            p=c0^2 / gamma +
              (cos(2x) + cos(2y)) * (cos(2z) + 2) / 16,
            rho=1.0,
        ),
    )
    numerics = Numerics(
        n_global=(n, n, n),
        art=ArtParams(enabled=false),
        cfl=0.6,
        filter_interval=1,
        dims=(1, 1, 1),
    )
    solver, Q = setup(problem, numerics)

    function kinetic_energy()
        total = 0.0
        m1, m2, m3 = solver.equations.i_mom
        for k in 1:n, j in 1:n, i in 1:n
            I = gidx(solver, i, j, k)
            rho = Q[I, 1]
            total += (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2rho)
        end
        return total / n^3
    end

    times = Float64[0.0]
    energy = Float64[kinetic_energy()]
    function record_energy(solver, Q)
        solver.step % 10 == 0 || return
        push!(times, solver.t)
        push!(energy, kinetic_energy())
        solver.step % 250 == 0 &&
            println("  TG step $(solver.step), t=$(round(solver.t; digits=3))")
    end
    run!(solver, Q; tfinal, nmax=10_000, callback=record_energy)
    if times[end] < solver.t
        push!(times, solver.t)
        push!(energy, kinetic_energy())
    end

    dissipation = similar(energy)
    dissipation[1] = -(energy[2] - energy[1]) / (times[2] - times[1])
    for i in 2:length(energy)-1
        dissipation[i] = -(energy[i + 1] - energy[i - 1]) /
                         (times[i + 1] - times[i - 1])
    end
    dissipation[end] = -(energy[end] - energy[end - 1]) /
                        (times[end] - times[end - 1])
    # A short centered average suppresses finite-difference noise without
    # moving the broad TGV dissipation peak.
    dissipation = [mean(@view dissipation[max(1, i-2):min(end, i+2)])
                   for i in eachindex(dissipation)]
    peak_index = argmax(dissipation)
    println(
        "  TG dissipation peak ε=$(round(dissipation[peak_index]; digits=5)) " *
        "at t=$(round(times[peak_index]; digits=3))",
    )

    work = Workspace(Q)
    compute_rhs!(solver, Q, work.dQ)
    omega = Array{Float64}(undef, n, n, n)
    for k in 1:n, j in 1:n, i in 1:n
        I = gidx(solver, i, j, k)
        wx = solver.grad_u[2, 3][I] - solver.grad_u[3, 2][I]
        wy = solver.grad_u[3, 1][I] - solver.grad_u[1, 3][I]
        wz = solver.grad_u[1, 2][I] - solver.grad_u[2, 1][I]
        omega[i, j, k] = sqrt(wx^2 + wy^2 + wz^2)
    end
    coords = [xcoord(solver, 1, i) for i in 1:n]
    return (;
        coords,
        omega,
        time=solver.t,
        times,
        dissipation,
        peak_time=times[peak_index],
        peak_dissipation=dissipation[peak_index],
    )
end

function shock_tube_data(; nx=384, ny=48, tfinal=2.0e-3)
    println("Running multicomponent shock showcase: $nx × $ny to t=$(1e3tfinal) ms")
    Ru = 8.314462618
    R_He = Ru / 4.002602e-3
    R_CO2 = Ru / 44.0095e-3
    eos = IdealMixture([
        IdealSpecies{Float64}("He", R_He, 5 / 3),
        IdealSpecies{Float64}("CO₂", R_CO2, 1.289),
    ])
    Lx, Ly = 4.0, 0.2
    hx = Lx / (nx - 1)
    delta = 3hx
    problem = Problem(
        name="He-driven RM showcase",
        eos=eos,
        transport=Transport(mu0=0.0),
        domain=((0.0, Lx), (0.0, Ly), (0.0, 1.0)),
        bcs=((SlipWallBC(), SlipWallBC()),
             (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC())),
        ic=(x, y, z) -> begin
            interface = 3.0 + 0.1Ly * cos(2π * y / Ly)
            co2 = tanh_blend(x, interface, delta)
            pressure = 101325.0 * (1 + 9 * (1 - tanh_blend(x, 2.0, delta)))
            Prim(Y=(1 - co2, co2), p=pressure, T_ion=300.0)
        end,
    )
    numerics = Numerics(
        n_global=(nx, ny, 1),
        art=ArtParams(enabled=true),
        cfl=0.5,
        filter_interval=1,
        dims=(1, 1, 1),
    )
    solver, Q = setup(problem, numerics)
    run!(solver, Q; tfinal, nmax=5000)

    rho = Array{Float64}(undef, nx, ny)
    Yco2 = similar(rho)
    for j in 1:ny, i in 1:nx
        I = gidx(solver, i, j, 1)
        rho[i, j] = Q[I, 1] + Q[I, 2]
        Yco2[i, j] = Q[I, 2] / rho[i, j]
    end
    x = [xcoord(solver, 1, i) for i in 1:nx]
    y = [xcoord(solver, 2, j) for j in 1:ny]
    return (; x, y, rho, Yco2, time=solver.t)
end

function converging_shock_data(; nr=640, tfinal=0.30)
    println("Running cylindrical shock showcase: $nr radial points to t=$tfinal")
    problem = Problem(
        name="converging shock showcase",
        eos=single_species(gamma=1.4),
        metric=CylindricalMetric(),
        domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        bcs=((AxisBC(), SlipWallBC()),
             (PeriodicBC(), PeriodicBC()),
             (PeriodicBC(), PeriodicBC())),
        ic=(r, theta, z) -> begin
            drive = tanh_blend(r, 0.7, 0.01)
            Prim(rho=1 + 3drive, p=1 + 19drive)
        end,
    )
    numerics = Numerics(
        n_global=(nr, 1, 1),
        art=ArtParams(enabled=true),
        cfl=0.4,
        filter_interval=1,
        dims=(1, 1, 1),
    )
    solver, Q = setup(problem, numerics)
    run!(solver, Q; tfinal, nmax=5000)

    radius = [xcoord(solver, 1, i) for i in 1:nr]
    rho = Vector{Float64}(undef, nr)
    pressure = similar(rho)
    m1, m2, m3 = solver.equations.i_mom
    ie = solver.equations.i_energy
    for i in 1:nr
        I = gidx(solver, i, 1, 1)
        rho[i] = Q[I, 1]
        kinetic = (Q[I, m1]^2 + Q[I, m2]^2 + Q[I, m3]^2) / (2rho[i])
        pressure[i] = 0.4 * (Q[I, ie] - kinetic)
    end
    return (; radius, rho, pressure, time=solver.t)
end

function radial_disk(radius, values; n=420, outer_radius=1.0)
    axis = collect(range(-outer_radius, outer_radius; length=n))
    disk = fill(NaN, n, n)
    for j in eachindex(axis), i in eachindex(axis)
        r = hypot(axis[i], axis[j])
        r <= outer_radius || continue
        if r <= radius[1]
            disk[i, j] = values[1]
        elseif r >= radius[end]
            disk[i, j] = values[end]
        else
            lo = searchsortedlast(radius, r)
            weight = (r - radius[lo]) / (radius[lo + 1] - radius[lo])
            disk[i, j] = (1 - weight) * values[lo] + weight * values[lo + 1]
        end
    end
    return axis, disk
end

function upsample_periodic3(values; factor=2)
    nx, ny, nz = size(values)
    smooth = Array{Float64}(undef, factor * nx, factor * ny, factor * nz)
    for kk in axes(smooth, 3), jj in axes(smooth, 2), ii in axes(smooth, 1)
        x = (ii - 1) / factor + 1
        y = (jj - 1) / factor + 1
        z = (kk - 1) / factor + 1
        i0, j0, k0 = floor.(Int, (x, y, z))
        tx, ty, tz = x - i0, y - j0, z - k0
        i1, j1, k1 = mod1(i0 + 1, nx), mod1(j0 + 1, ny), mod1(k0 + 1, nz)
        c00 = (1 - tx) * values[i0, j0, k0] + tx * values[i1, j0, k0]
        c10 = (1 - tx) * values[i0, j1, k0] + tx * values[i1, j1, k0]
        c01 = (1 - tx) * values[i0, j0, k1] + tx * values[i1, j0, k1]
        c11 = (1 - tx) * values[i0, j1, k1] + tx * values[i1, j1, k1]
        c0 = (1 - ty) * c00 + ty * c10
        c1 = (1 - ty) * c01 + ty * c11
        smooth[ii, jj, kk] = (1 - tz) * c0 + tz * c1
    end
    return smooth
end

function upsample_shock(x, y, values; factor_x=3, factor_y=4)
    nx, ny = size(values)
    nx_smooth = factor_x * (nx - 1) + 1
    ny_smooth = factor_y * ny
    smooth = Array{Float64}(undef, nx_smooth, ny_smooth)
    for jj in 1:ny_smooth, ii in 1:nx_smooth
        xi = (ii - 1) / factor_x + 1
        yi = (jj - 1) / factor_y + 1
        i0 = min(floor(Int, xi), nx - 1)
        j0 = floor(Int, yi)
        tx, ty = xi - i0, yi - j0
        i1 = i0 + 1
        j1 = mod1(j0 + 1, ny)
        low = (1 - tx) * values[i0, j0] + tx * values[i1, j0]
        high = (1 - tx) * values[i0, j1] + tx * values[i1, j1]
        smooth[ii, jj] = (1 - ty) * low + ty * high
    end
    x_smooth = collect(range(first(x), last(x); length=nx_smooth))
    dy = (y[2] - y[1]) / factor_y
    y_smooth = [first(y) + (j - 1) * dy for j in 1:ny_smooth]
    return x_smooth, y_smooth, smooth
end

function load_or_generate_data()
    refresh = get(ENV, "CL_HERO_REFRESH", "0") == "1"
    refresh_tg = get(ENV, "CL_HERO_REFRESH_TG", "0") == "1"
    if isfile(CACHE_PATH) && !refresh
        println("Loading cached showcase fields from $CACHE_PATH")
        data = deserialize(CACHE_PATH)
        if refresh_tg
            data = (
                taylor_green=taylor_green_data(),
                shock_tube=data.shock_tube,
                converging_shock=data.converging_shock,
            )
            serialize(CACHE_PATH, data)
        end
        return data
    end
    data = (
        taylor_green=taylor_green_data(),
        shock_tube=shock_tube_data(),
        converging_shock=converging_shock_data(),
    )
    serialize(CACHE_PATH, data)
    return data
end

function render_hero(data)
    mkpath(ASSET_DIR)
    background = RGBf(0.025, 0.035, 0.055)
    panel = RGBf(0.045, 0.060, 0.085)
    foreground = RGBf(0.93, 0.95, 0.98)
    muted = RGBf(0.68, 0.74, 0.82)
    grid = RGBAf(0.75, 0.80, 0.88, 0.12)
    accent = RGBf(1.0, 0.60, 0.20)
    reference = RGBf(0.35, 0.85, 1.0)
    domain_edge = RGBAf(0.72, 0.78, 0.88, 0.55)

    # Lay the composition out at its actual README display size. The final
    # export uses 2× pixel density for smooth rasterization without shrinking
    # the logical font sizes when GitHub fits it to the content column.
    fig = Figure(size=(840, 480), backgroundcolor=background, figure_padding=12)
    Label(
        fig[0, 1:3],
        "High-order compact numerics · shocks · curvilinear geometry",
        color=foreground,
        fontsize=18,
        font=:bold,
        padding=(0, 0, 0, 4),
    )

    # Left: rotate the axial direction vertically and reverse it so the shock
    # travels down the page. DataAspect preserves the tube's physical scale,
    # so its 0.2 m cross-section remains narrow beside the 1.5 m axial crop.
    st = data.shock_tube
    shock_x, shock_y, shock_rho = upsample_shock(st.x, st.y, st.rho)
    _, _, shock_Yco2 = upsample_shock(st.x, st.y, st.Yco2)
    transverse = shock_y
    shock_layout = GridLayout()
    fig[1, 1] = shock_layout
    axshock = Axis(
        shock_layout[1, 1],
        title="He/CO₂ shock tube  ↓",
        subtitle="density  ·  t = $(round(1e3st.time; digits=2)) ms\nwhite contour: YCO₂ = 0.5",
        titlecolor=foreground,
        subtitlecolor=muted,
        titlefont=:bold,
        titlesize=15,
        subtitlesize=11,
        xlabel="transverse y: 0–0.2 m",
        ylabel="axial position  x  [m]",
        xlabelcolor=muted,
        ylabelcolor=muted,
        xticklabelcolor=muted,
        yticklabelcolor=muted,
        xticklabelsvisible=false,
        xticksvisible=false,
        xticklabelsize=10,
        yticklabelsize=10,
        xlabelsize=11,
        ylabelsize=11,
        xgridcolor=grid,
        ygridcolor=grid,
        yreversed=true,
        aspect=DataAspect(),
        backgroundcolor=panel,
    )
    rho_range = quantile(vec(shock_rho), [0.01, 0.99])
    density_plot = heatmap!(
        axshock, transverse, shock_x, permutedims(shock_rho);
        colormap=:viridis,
        colorrange=Tuple(rho_range),
        interpolate=true,
    )
    contour!(
        axshock, transverse, shock_x, permutedims(shock_Yco2);
        levels=[0.5],
        color=:white,
        linewidth=2.6,
    )
    xlims!(axshock, 0.0, 0.2)
    ylims!(axshock, 2.5, 4.0)
    axshock.yreversed[] = true
    Colorbar(
        shock_layout[1, 2], density_plot;
        label="ρ  [kg m⁻³]",
        labelcolor=muted,
        ticklabelcolor=muted,
        labelsize=11,
        ticklabelsize=10,
        width=8,
    )

    # Center: late-time TGV vorticity plus a compact quantitative benchmark.
    tg = data.taylor_green
    tg_layout = GridLayout()
    fig[1, 2] = tg_layout
    omega = upsample_periodic3(tg.omega; factor=2)
    omega_coords = collect(range(0, 2π; length=size(omega, 1) + 1))[1:end-1]
    ax3 = Axis3(
        tg_layout[1, 1],
        title="Taylor–Green vortex  ·  |ω|  ·  t = $(round(tg.time; digits=1))",
        titlecolor=foreground,
        titlefont=:bold,
        titlesize=16,
        aspect=(1, 1, 1),
        azimuth=1.22π,
        elevation=0.18π,
        perspectiveness=0.62,
        xspinecolor_1=domain_edge,
        xspinecolor_2=domain_edge,
        xspinecolor_3=domain_edge,
        xspinecolor_4=domain_edge,
        yspinecolor_1=domain_edge,
        yspinecolor_2=domain_edge,
        yspinecolor_3=domain_edge,
        yspinecolor_4=domain_edge,
        zspinecolor_1=domain_edge,
        zspinecolor_2=domain_edge,
        zspinecolor_3=domain_edge,
        zspinecolor_4=domain_edge,
        xspinewidth=1.25,
        yspinewidth=1.25,
        zspinewidth=1.25,
        backgroundcolor=panel,
    )
    ax3.zoom_mult[] = 0.84
    hidedecorations!(ax3)
    omega_hi = quantile(vec(omega), 0.995)
    middle = cld(length(omega_coords), 2)
    surface!(
        ax3,
        omega_coords,
        omega_coords,
        fill(omega_coords[middle], length(omega_coords), length(omega_coords));
        color=omega[:, :, middle],
        colormap=:inferno,
        colorrange=(0, omega_hi),
        shading=NoShading,
        alpha=0.72,
        transparency=true,
    )
    levels = quantile(vec(omega), [0.87, 0.96])
    contour!(
        ax3,
        (0.0, 2π),
        (0.0, 2π),
        (0.0, 2π),
        omega;
        levels,
        colormap=:plasma,
        alpha=0.34,
        transparency=true,
    )

    axeps = Axis(
        tg_layout[2, 1],
        title="ε = −dEₖ/dt",
        titlecolor=foreground,
        titlesize=12,
        xlabel="t",
        ylabel="ε",
        xlabelcolor=muted,
        ylabelcolor=muted,
        xticklabelcolor=muted,
        yticklabelcolor=muted,
        xgridcolor=grid,
        ygridcolor=grid,
        xticklabelsize=9,
        yticklabelsize=9,
        xlabelsize=10,
        ylabelsize=10,
        aspect=3.8,
        backgroundcolor=panel,
    )
    lines!(
        axeps, tg.times, tg.dissipation;
        color=accent,
        linewidth=2.5,
        label="CompactLES $(length(tg.coords))³",
    )
    scatter!(
        axeps, [tg.peak_time], [tg.peak_dissipation];
        color=accent,
        markersize=11,
    )
    # Peak digitized from Fig. 8 of van Rees et al., JCP 230 (2011),
    # doi:10.1016/j.jcp.2010.11.031 (Re=1600 pseudo-spectral DNS).
    reference_time, reference_peak = 8.86, 0.01289
    scatter!(
        axeps, [reference_time], [reference_peak];
        color=reference,
        marker=:star5,
        markersize=15,
        label="spectral DNS peak",
    )
    text!(
        axeps, tg.peak_time, tg.peak_dissipation;
        text="  $(round(tg.peak_dissipation; digits=4)) @ $(round(tg.peak_time; digits=2))",
        color=foreground,
        fontsize=9,
        align=(:left, :top),
    )
    axislegend(
        axeps;
        position=:lt,
        orientation=:horizontal,
        labelcolor=foreground,
        backgroundcolor=RGBAf(0.02, 0.03, 0.05, 0.72),
        framecolor=RGBAf(1, 1, 1, 0.18),
        labelsize=9,
        patchsize=(12, 7),
        padding=(4, 4, 3, 3),
    )
    xlims!(axeps, 0, 12)
    ylims!(axeps, 0, 1.2max(maximum(tg.dissipation), reference_peak))

    # Right: the cylindrical run remains explicitly identified as a radial
    # solution, rather than implying a resolved two-dimensional calculation.
    cs = data.converging_shock
    disk_layout = GridLayout()
    fig[1, 3] = disk_layout
    disk_axis, pressure_disk = radial_disk(cs.radius, log10.(max.(cs.pressure, eps())))
    axdisk = Axis(
        disk_layout[1, 1],
        title="Cylindrically converging shock",
        subtitle="axisymmetric radial solution  ·  t = $(round(cs.time; digits=2))",
        titlecolor=foreground,
        subtitlecolor=muted,
        titlefont=:bold,
        titlesize=15,
        subtitlesize=11,
        aspect=DataAspect(),
        backgroundcolor=panel,
    )
    pressure_range = extrema(filter(isfinite, vec(pressure_disk)))
    pressure_plot = heatmap!(
        axdisk, disk_axis, disk_axis, pressure_disk;
        colormap=:thermal,
        colorrange=pressure_range,
        interpolate=true,
    )
    angles = range(0, 2π; length=361)
    lines!(axdisk, cos.(angles), sin.(angles); color=RGBAf(1, 1, 1, 0.35), linewidth=1.5)
    hidedecorations!(axdisk)
    hidespines!(axdisk)
    Colorbar(
        disk_layout[2, 1], pressure_plot;
        label="log₁₀ p",
        labelcolor=muted,
        ticklabelcolor=muted,
        labelsize=11,
        ticklabelsize=10,
        vertical=false,
        height=11,
    )

    rowsize!(tg_layout, 1, Relative(0.72))
    rowsize!(tg_layout, 2, Relative(0.28))
    colsize!(shock_layout, 1, Aspect(1, 0.2 / 1.5))
    colgap!(shock_layout, 4)
    rowgap!(tg_layout, 6)
    rowgap!(disk_layout, 6)
    colsize!(fig.layout, 1, Relative(0.17))
    colsize!(fig.layout, 2, Relative(0.51))
    colsize!(fig.layout, 3, Relative(0.32))
    colgap!(fig.layout, 10)

    save(OUTPUT_PATH, fig; px_per_unit=2)
    println("Wrote $OUTPUT_PATH")
    return OUTPUT_PATH
end

render_hero(load_or_generate_data())
