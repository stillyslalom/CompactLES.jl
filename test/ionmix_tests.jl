# IONMIX table reading, writing, interpolation and temperature inversion, on
# synthetic tables built from an analytic ideal plasma: no real table is in the
# repository. Included by the serial suite and runnable directly with
# `julia --project=. test/ionmix_tests.jl`.

if !isdefined(Main, :CompactLES)
    using CompactLES
end
using Test

module IonmixTestModel

using ..CompactLES
const CL = CompactLES

const K_B = 1.380649e-23
const ION_MASS = 4.002602 * 1.66053906660e-27     # helium
const Z_MAX = 2.0
const T_IONIZE = 3.0e5                             # K
const IONIZATION = 40.0 * 1.602176634e-19          # J per electron

# A mean ionization rising smoothly with temperature and falling with density,
# and the ideal-plasma energies and pressures it implies.
zbar(T, ρ) = Z_MAX * T / (T + T_IONIZE * (ρ / 1.0)^0.2)
e_ion(T, ρ) = 1.5 * K_B * T / ION_MASS
e_ele(T, ρ) = (1.5 * K_B * T + IONIZATION) * zbar(T, ρ) / ION_MASS
p_ion(T, ρ) = ρ * K_B * T / ION_MASS
p_ele(T, ρ) = zbar(T, ρ) * ρ * K_B * T / ION_MASS
opacity(T, ρ, g) = 1.0e3 * g * (T / 1.0e4)^-2 * ρ^0.5

axis(lo, hi, n) = [exp(log(lo) + (log(hi) - log(lo)) * (k - 1) / (n - 1)) for k in 1:n]

function plasma_table(nt, nd; ngroups=0, entropy=false, extrapolate=:missing,
                      T_range=(1.0e3, 1.0e8), ρ_range=(1.0e-4, 1.0e2))
    Ts = axis(T_range..., nt)
    ρs = axis(ρ_range..., nd)
    grid(f) = [f(T, ρ) for T in Ts, ρ in ρs]
    groups = [opacity(T, ρ, g) for T in Ts, ρ in ρs, g in 1:ngroups]
    return CL.IonmixTable(; temperature=Ts, ion_density=ρs ./ ION_MASS,
        ion_mass=ION_MASS, zbar=grid(zbar), p_ion=grid(p_ion), p_ele=grid(p_ele),
        e_ion=grid(e_ion), e_ele=grid(e_ele), cv_ion=grid((T, ρ) -> 1.5 * K_B / ION_MASS),
        s_ele=entropy ? grid((T, ρ) -> 1.0e4 * log(T)) : nothing,
        group_bounds=[1.0e-17 * g for g in 0:ngroups],
        rosseland=groups, planck_absorption=2 .* groups, planck_emission=3 .* groups,
        atomic_numbers=[2], fractions=[1.0], extrapolate=extrapolate)
end

# A table whose fields are exactly bilinear in (ln T, ln ρ), so that the
# interpolant and its linear extension reproduce them everywhere.
linear_e(T, ρ) = 1.0e6 * (1 + 0.5 * log(T / 1.0e3)) + 1.0e3 * log(ρ)
function linear_table(; extrapolate=:missing)
    Ts = axis(1.0e3, 1.0e6, 7)
    ρs = axis(1.0e-2, 1.0e1, 4)
    grid(f) = [f(T, ρ) for T in Ts, ρ in ρs]
    return CL.IonmixTable(; temperature=Ts, ion_density=ρs ./ ION_MASS,
        ion_mass=ION_MASS, zbar=grid((T, ρ) -> 1.0), p_ion=grid(linear_e),
        p_ele=grid((T, ρ) -> 0.0), e_ion=grid(linear_e), e_ele=grid((T, ρ) -> 0.0),
        extrapolate=extrapolate)
end

end # module IonmixTestModel

@testset "IONMIX tables" begin
    CL = CompactLES
    M = IonmixTestModel
    dir = mktempdir()

    @testset "Fortran E12.6 fields" begin
        @test CL._fortran_e12(-1.5) == "-.150000E+01"
        @test CL._fortran_e12(0.0) == "0.000000E+00"
        @test CL._fortran_e12(1.23456e99) == "0.123456+100"
        @test CL._fortran_e12(9.9999996e-5) == "0.100000E-03"
        @test all(v -> length(CL._fortran_e12(v)) == 12,
                  (1.0, -1.0e-300, 3.0e307, 1.234567e-7))
        @test CL._fortran_float("0.123456+100") == 1.23456e99
        @test CL._fortran_float("-.150000E+01") == -1.5
        @test CL._fortran_float(" 0.1D+01") == 1.0
        @test_throws ArgumentError CL._fortran_e12(NaN)
        @test_throws ArgumentError CL._fortran_float("0.1E+0x")
    end

    @testset "round trip ($(entropy ? "IONMIX6" : "IONMIX4"))" for entropy in (false, true)
        table = M.plasma_table(6, 5; ngroups=2, entropy=entropy)
        path = CL.write_ionmix(joinpath(dir, "plasma.cn4"), table)
        first = CL.read_ionmix(path; ion_mass=M.ION_MASS)
        @test first.format === (entropy ? :ionmix6 : :ionmix4)
        @test first.atomic_numbers == [2] && first.fractions == [1.0]
        # Six significant digits on file.
        for name in (:temperature, :density, :zbar, :p_ion, :p_ele, :e_ion, :e_ele,
                     :cv_ion, :group_bounds, :rosseland, :planck_emission, :s_ele)
            @test isapprox(getfield(first, name), getfield(table, name); rtol=5e-6)
        end
        # Writing what was read reproduces the file, and reading it again every
        # value, bit for bit.
        again = CL.write_ionmix(joinpath(dir, "again.cn4"), first)
        @test read(again, String) == read(path, String)
        second = CL.read_ionmix(again; ion_mass=M.ION_MASS)
        for name in fieldnames(CL.IonmixTable)
            @test isequal(getfield(second, name), getfield(first, name))
        end
        # CRLF line endings and an explicit format read the same table.
        crlf = joinpath(dir, "crlf.cn4")
        write(crlf, replace(read(path, String), "\n" => "\r\n"))
        @test CL.read_ionmix(crlf; ion_mass=M.ION_MASS, format=first.format).e_ele ==
              first.e_ele
        wrong = entropy ? :ionmix4 : :ionmix6
        @test_throws ArgumentError CL.read_ionmix(path; ion_mass=M.ION_MASS, format=wrong)
        @test eltype(CL.read_ionmix(Float32, path; ion_mass=M.ION_MASS).e_ion) == Float32
    end

    @testset "log-spaced header and rejected layouts" begin
        nt, nd = 4, 3
        table = M.plasma_table(nt, nd; T_range=(1.0e4, 1.0e7), ρ_range=(1.0e-3, 1.0e-1))
        lines = readlines(CL.write_ionmix(joinpath(dir, "explicit.cn4"), table))
        # Replace the group count and the two axis blocks (one line each) by the
        # log-spaced header: Δlog10 n, log10 n₁, Δlog10 T, log10 T₁, ngroups.
        n1 = log10(1.0e-3 / M.ION_MASS * 1e-6)
        header = join(CL._fortran_e12.((1.0, n1, 1.0, log10(1.0e4 / CL._IONMIX_EV_KELVIN))))
        logfile = joinpath(dir, "log.cn4")
        write(logfile, join([lines[1:3]; header * "           0"; lines[7:end]], "\n") * "\n")
        logged = CL.read_ionmix(logfile; ion_mass=M.ION_MASS)
        @test logged.log_grid !== nothing
        @test isapprox(logged.temperature, table.temperature; rtol=1e-5)
        @test isapprox(logged.density, table.density; rtol=1e-4)
        @test logged.e_ele == CL.read_ionmix(joinpath(dir, "explicit.cn4");
                                             ion_mass=M.ION_MASS).e_ele
        CL.write_ionmix(joinpath(dir, "log2.cn4"), logged)
        @test read(joinpath(dir, "log2.cn4"), String) == read(logfile, String)
        # Four fields per point (the single-temperature layout) are recognized
        # and rejected: keep Z̄ and the next three blocks, drop the other eight.
        block = cld(nt * nd, 4)
        fields = 7                                   # first line of Z̄
        single = [lines[1:6]; lines[fields:fields+4block-1]; lines[fields+12block:end]]
        singlefile = joinpath(dir, "single.cn4")
        write(singlefile, join(single, "\n") * "\n")
        err = try
            CL.read_ionmix(singlefile; ion_mass=M.ION_MASS)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("single-temperature", err.msg)
        truncated = joinpath(dir, "truncated.cn4")
        write(truncated, join(lines[1:end-1], "\n") * "\n")
        @test_throws ArgumentError CL.read_ionmix(truncated; ion_mass=M.ION_MASS)
    end

    @testset "construction checks" begin
        good = (temperature=[1.0, 2.0], ion_density=[1.0, 2.0], ion_mass=1.0,
                zbar=zeros(2, 2), p_ion=zeros(2, 2), p_ele=zeros(2, 2),
                e_ion=zeros(2, 2), e_ele=zeros(2, 2))
        @test CL.IonmixTable(; good...) isa CL.IonmixTable{Float64}
        @test_throws ArgumentError CL.IonmixTable(; good..., temperature=[0.0, 1.0])
        @test_throws ArgumentError CL.IonmixTable(; good..., ion_density=[2.0, 1.0])
        @test_throws ArgumentError CL.IonmixTable(; good..., extrapolate=:clamp)
        @test_throws ArgumentError CL.IonmixTable(; good..., ion_mass=0.0)
        @test_throws ArgumentError CL.IonmixTable(; good..., e_ion=[NaN 0; 0 0])
        @test_throws DimensionMismatch CL.IonmixTable(; good..., zbar=zeros(3, 2))
    end

    @testset "nodes and interpolation order" begin
        table = M.plasma_table(21, 13; ngroups=1)
        for j in eachindex(table.density), i in eachindex(table.temperature)
            T, ρ = table.temperature[i], table.density[j]
            @test CL.table_value(table, :e_ion, T, ρ)[1] == table.e_ion[i, j]
            e, _, _, status = CL.table_value(table, :e, T, ρ)
            @test e == table.e_ion[i, j] + table.e_ele[i, j] && status == CL.TABLE_OK
            @test CL.table_value(table, table.zbar, T, ρ)[1] == table.zbar[i, j]
            @test CL.table_opacity(table, :rosseland, 1, T, ρ)[1] == table.rosseland[i, j, 1]
        end
        # Linear interpolation at a third of a cell has error t(1-t)h²f''/2, and
        # a third of a coarse cell is two thirds of a fine one, so halving the
        # spacing divides the leading error by exactly four. The third-order
        # term changes sign between the two positions, which puts the measured
        # orders at 2.06-2.14 on these grids.
        function error_at_thirds(nt, nd)
            fine = M.plasma_table(nt, nd)
            coarse = M.plasma_table(41, 25)
            x, y = coarse.log_temperature, coarse.log_density
            worst = zeros(3)
            for j in 1:length(y)-1, i in 1:length(x)-1
                T = exp((2x[i] + x[i+1]) / 3)
                ρ = exp((2y[j] + y[j+1]) / 3)
                exact = ((:e, (T, ρ) -> M.e_ion(T, ρ) + M.e_ele(T, ρ)),
                         (:p, (T, ρ) -> M.p_ion(T, ρ) + M.p_ele(T, ρ)), (:zbar, M.zbar))
                for (k, (name, f)) in enumerate(exact)
                    value = CL.table_value(fine, name, T, ρ)[1]
                    worst[k] = max(worst[k], abs(value / f(T, ρ) - 1))
                end
            end
            return worst
        end
        orders = log2.(error_at_thirds(41, 25) ./ error_at_thirds(81, 49))
        @test all(o -> 1.95 < o < 2.2, orders)
    end

    @testset "derivatives of the interpolant" begin
        table = M.plasma_table(21, 13)
        δ = 1e-6
        for (T, ρ) in ((2.3e4, 3.1e-3), (7.7e6, 17.0), (1.9e3, 2.2e-4))
            for name in (:e, :p, :zbar, :e_ele)
                f, f_T, f_ρ, status = CL.table_value(table, name, T, ρ)
                @test status == CL.TABLE_OK
                fd_T = (CL.table_value(table, name, T * (1 + δ), ρ)[1] -
                        CL.table_value(table, name, T * (1 - δ), ρ)[1]) / (2T * δ)
                fd_ρ = (CL.table_value(table, name, T, ρ * (1 + δ))[1] -
                        CL.table_value(table, name, T, ρ * (1 - δ))[1]) / (2ρ * δ)
                @test isapprox(f_T, fd_T; rtol=1e-6)
                @test isapprox(f_ρ, fd_ρ; rtol=1e-6, atol=1e-9 * abs(f) / ρ)
            end
            state = CL.table_state(table, T, ρ)
            e, cv, de_dρ, _ = CL.table_value(table, :e, T, ρ)
            p, dp_dT, dp_dρ, _ = CL.table_value(table, :p, T, ρ)
            @test (state.e, state.cv, state.de_drho) == (e, cv, de_dρ)
            @test (state.p, state.dp_dT, state.dp_drho) == (p, dp_dT, dp_dρ)
            @test state.c2 ≈ dp_dρ + T * dp_dT^2 / (ρ^2 * cv)
            @test state.c2 > 0 && state.status == CL.TABLE_OK
        end
    end

    @testset "temperature inversion" begin
        table = M.plasma_table(21, 13)
        for j in eachindex(table.density), i in eachindex(table.temperature)
            ρ = table.density[j]
            for (component, E) in ((:total, table.e_ion .+ table.e_ele),
                                   (:ion, table.e_ion), (:electron, table.e_ele))
                @test CL.table_temperature_status(table, E[i, j], ρ, component) ==
                      (table.temperature[i], CL.TABLE_OK)
            end
        end
        for (T, ρ) in ((2.3e4, 3.1e-3), (7.7e6, 17.0), (1.9e3, 2.2e-4), (9.9e7, 1.0e-4))
            e = CL.table_value(table, :e, T, ρ)[1]
            T_rec, status = CL.table_temperature_status(table, e, ρ)
            @test isapprox(T_rec, T; rtol=1e-12) && status == CL.TABLE_OK
            @test CL.table_temperature(table, CL.table_value(table, :e_ion, T, ρ)[1], ρ,
                                       :ion) ≈ T
        end
        @test_throws ArgumentError CL.table_temperature_status(table, 1.0, 1.0, :ele)
    end

    @testset "domain policy" begin
        missing_policy = M.linear_table()
        linear_policy = M.linear_table(extrapolate=:linear)
        Tlo, Thi = missing_policy.temperature[1], missing_policy.temperature[end]
        ρlo, ρhi = missing_policy.density[1], missing_policy.density[end]
        for (table, flag) in ((missing_policy, CL.TABLE_OUT_OF_DOMAIN),
                              (linear_policy, CL.TABLE_EXTRAPOLATED))
            # Outside the axes the edge cell is extended, never clamped, and the
            # point reported.
            for (T, ρ, axis) in ((Tlo / 3, 0.1, CL.TABLE_TEMPERATURE_AXIS),
                                 (Thi * 5, 0.1, CL.TABLE_TEMPERATURE_AXIS),
                                 (1.0e4, ρhi * 7, CL.TABLE_DENSITY_AXIS),
                                 (1.0e4, ρlo / 7, CL.TABLE_DENSITY_AXIS))
                e, _, _, status = CL.table_value(table, :e, T, ρ)
                @test e ≈ M.linear_e(T, ρ)
                @test status == flag | axis
                T_rec, status = CL.table_temperature_status(table, M.linear_e(T, ρ), ρ)
                @test T_rec ≈ T && status == flag | axis
            end
            @test CL.table_value(table, :e, Thi, ρhi)[4] == CL.TABLE_OK
            @test CL.table_value(table, :e, Tlo, ρlo)[4] == CL.TABLE_OK
            for (T, ρ) in ((0.0, 0.1), (-1.0, 0.1), (1.0e4, 0.0), (NaN, 0.1))
                value, _, _, status = CL.table_value(table, :e, T, ρ)
                @test isnan(value) && status & CL.TABLE_OUT_OF_DOMAIN != 0
                @test status & CL.TABLE_EXTRAPOLATED == 0
            end
            @test isnan(CL.table_temperature(table, NaN, 0.1))
            @test CL.table_temperature_status(table, 1.0e6, 0.0)[2] ==
                  CL.TABLE_OUT_OF_DOMAIN | CL.TABLE_DENSITY_AXIS
        end
    end

    @testset "non-monotone and unstable regions" begin
        table = M.linear_table()
        # A dip in the energy along the second density node, as a table with a
        # phase transition and no Maxwell construction might carry.
        e_ion = copy(table.e_ion)
        e_ion[4, 2] = e_ion[2, 2]
        dipped = CL.IonmixTable(; temperature=table.temperature,
            ion_density=table.ion_density, ion_mass=table.ion_mass, zbar=table.zbar,
            p_ion=table.p_ion, p_ele=table.p_ele, e_ion=e_ion, e_ele=table.e_ele)
        @test dipped.monotone[:, 1] == [true, false, true, true]
        @test !dipped.monotone[2, 2] && !any(dipped.monotone[:, 3])
        ρ = sqrt(dipped.density[1] * dipped.density[2])
        T_rec, status = CL.table_temperature_status(dipped, e_ion[3, 2], ρ)
        @test status & CL.TABLE_NOT_MONOTONE != 0
        @test CL.table_value(dipped, :e, T_rec, ρ)[1] ≈ e_ion[3, 2]
        # The same energy on a monotone column inverts without the flag.
        @test CL.table_temperature_status(dipped, e_ion[3, 4], dipped.density[4])[2] ==
              CL.TABLE_OK
        T_dip = sqrt(dipped.temperature[4] * dipped.temperature[5])
        @test CL.table_value(dipped, :e, T_dip, dipped.density[2])[2] > 0
        state = CL.table_state(dipped, sqrt(dipped.temperature[3] * dipped.temperature[4]),
                               dipped.density[2])
        @test state.cv < 0 && state.status & CL.TABLE_UNSTABLE != 0
    end

    @testset "opacity queries" begin
        table = M.plasma_table(6, 5; ngroups=2)
        @test_throws BoundsError CL.table_opacity(table, :rosseland, 3, 1.0e4, 1.0)
        @test_throws ArgumentError CL.table_opacity(table, :compton, 1, 1.0e4, 1.0)
        κ, κ_T, κ_ρ, status = CL.table_opacity(table, :planck_emission, 2, 3.0e4, 0.3)
        @test status == CL.TABLE_OK && κ > 0 && κ_T < 0 && κ_ρ > 0
    end
end
