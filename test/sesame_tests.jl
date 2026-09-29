# SESAME ASCII 2 reading, writing, interpolation and temperature inversion, on
# synthetic libraries built from an analytic free energy: no real SESAME table
# is in the repository. Included by the serial suite and runnable directly with
# `julia --project=. test/sesame_tests.jl`.

if !isdefined(Main, :CompactLES)
    using CompactLES
end
using Test

module SesameTestModel

using ..CompactLES
const CL = CompactLES

const K_B = 1.380649e-23
const R_GAS = K_B / (26.9815 * 1.66053906660e-27)   # aluminium, J/(kg K)
const Z_FREE = 3.0                                   # electrons per ion
const RHO_0 = 2700.0                                 # kg/m³
const C_COLD = 1.0e6                                 # J/kg
const T_REF = 300.0                                  # K

# A free energy A = φ(ρ) + A_nuc(ρ, T) + A_elec(ρ, T): a cold curve with its
# minimum at RHO_0, and ideal ions and a fixed number of ideal electrons.
# Each part gives its energy e = A - T ∂A/∂T and pressure p = ρ² ∂A/∂ρ.
cold_A(ρ) = C_COLD * ((ρ / RHO_0)^2 / 2 - ρ / RHO_0 + 0.5)
cold_p(ρ) = C_COLD * RHO_0 * (ρ / RHO_0)^2 * (ρ / RHO_0 - 1)
ideal_A(T, ρ, n) = 1.5n * R_GAS * T * (1 - log(T / T_REF)) + n * R_GAS * T * log(ρ / RHO_0)
ideal_e(T, ρ, n) = 1.5n * R_GAS * T
ideal_p(T, ρ, n) = n * ρ * R_GAS * T

const PARTS = (
    total=((T, ρ) -> cold_A(ρ) + ideal_A(T, ρ, 1) + ideal_A(T, ρ, Z_FREE),
           (T, ρ) -> cold_A(ρ) + ideal_e(T, ρ, 1) + ideal_e(T, ρ, Z_FREE),
           (T, ρ) -> cold_p(ρ) + ideal_p(T, ρ, 1) + ideal_p(T, ρ, Z_FREE)),
    ion=((T, ρ) -> cold_A(ρ) + ideal_A(T, ρ, 1), (T, ρ) -> cold_A(ρ) + ideal_e(T, ρ, 1),
         (T, ρ) -> cold_p(ρ) + ideal_p(T, ρ, 1)),
    electron=((T, ρ) -> ideal_A(T, ρ, Z_FREE), (T, ρ) -> ideal_e(T, ρ, Z_FREE),
              (T, ρ) -> ideal_p(T, ρ, Z_FREE)),
    nuclear=((T, ρ) -> ideal_A(T, ρ, 1), (T, ρ) -> ideal_e(T, ρ, 1),
             (T, ρ) -> ideal_p(T, ρ, 1)),
)

axis(lo, hi, n) = [exp(log(lo) + (log(hi) - log(lo)) * (k - 1) / (n - 1)) for k in 1:n]

function component(part, nt, nd; table_id=301, interpolation=:bilinear,
                   extrapolate=:missing, T_range=(3.0e2, 1.0e6),
                   ρ_range=(1.0e1, 1.0e5))
    A, e, p = PARTS[part]
    Ts, ρs = axis(T_range..., nt), axis(ρ_range..., nd)
    grid(f) = [f(T, ρ) for T in Ts, ρ in ρs]
    return CL.SesameComponent(; temperature=Ts, density=ρs, p=grid(p), e=grid(e),
                              free_energy=grid(A), table_id, interpolation, extrapolate)
end

function library_table(nt, nd; kwargs...)
    ρs = axis(1.0e1, 1.0e5, nd)
    cold = CL._sesame_cold(Float64, 306, 0.0, ρs, cold_p.(ρs), cold_A.(ρs), cold_A.(ρs),
                           :missing, 0)
    return CL.SesameTable(3720, component(:total, nt, nd; kwargs...);
        ion=component(:ion, nt, nd; table_id=303, kwargs...),
        electron=component(:electron, nt, nd; table_id=304, kwargs...),
        nuclear=component(:nuclear, nt, nd; table_id=305, kwargs...), cold,
        zbar=13.0, abar=26.9815, reference_density=RHO_0, bulk_modulus=C_COLD * RHO_0,
        exchange_coefficient=0.0,
        comments=[101 => "material. synthetic (z=13, a=26.9815)/ source. " *
                         "analytic free energy/ other. " * "x"^60 * "/"])
end

end # module SesameTestModel

@testset "SESAME tables" begin
    CL = CompactLES
    M = SesameTestModel
    dir = mktempdir()
    function same(a, b)
        a === nothing && return b === nothing
        return all(name -> isequal(getfield(a, name), getfield(b, name)),
                   fieldnames(typeof(a)))
    end
    parts = (:total, :ion, :electron, :nuclear)

    @testset "numbers on file" begin
        @test CL._sesame_word(1.5, 1.0) == " 1.500000000000000E+00"
        @test CL._sesame_word(-2.5e9, 1e9) == "-2.500000000000000E+00"
        @test CL._sesame_word(3, 1.0) == " 3.000000000000000E+00"
        @test_throws ArgumentError CL._sesame_word(1.0e120, 1.0)
        @test_throws ArgumentError CL._sesame_word(Inf, 1.0)
    end

    @testset "round trip" begin
        table = M.library_table(9, 7)
        decoy = CL.SesameTable(1111, M.component(:electron, 3, 3))
        path = CL.write_sesame(joinpath(dir, "library.ses"), decoy, table)
        lines = readlines(path)
        @test lines[1] == "Version 2.0"
        @test all(line -> length(line) <= 160, lines)
        first = CL.read_sesame(path, 3720)
        @test first.comments == table.comments && first.material_id == 3720
        @test (first.zbar, first.abar, first.reference_density) == (13.0, 26.9815, M.RHO_0)
        # Sixteen significant digits on file.
        for part in parts, name in (:temperature, :density, :p, :e, :free_energy)
            @test isapprox(getfield(getfield(first, part), name),
                           getfield(getfield(table, part), name); rtol=1e-15)
        end
        @test first.cold.p ≈ table.cold.p && first.cold.temperature == 0
        @test first.cold.dropped == 0 && first.total.dropped == (0, 0)
        @test CL.read_sesame(path, 1111).ion === nothing
        # Writing what was read and reading it again returns every value bit for
        # bit, and writing that reproduces the file.
        again = CL.write_sesame(joinpath(dir, "again.ses"), first)
        second = CL.read_sesame(again, 3720)
        for name in fieldnames(CL.SesameTable)
            a, b = getfield(first, name), getfield(second, name)
            @test (a isa Union{CL.SesameComponent,CL.SesameColdCurve} ? same(a, b) :
                   isequal(a, b))
        end
        third = CL.write_sesame(joinpath(dir, "third.ses"), second)
        @test read(third, String) == read(again, String)
        crlf = joinpath(dir, "crlf.ses")
        write(crlf, replace(read(path, String), "\n" => "\r\n"))
        @test same(CL.read_sesame(crlf, 3720).total, first.total)
        @test CL.read_sesame(crlf, 3720).comments == first.comments
        @test eltype(CL.read_sesame(Float32, path, 3720).total.e) == Float32
        @test CL.read_sesame(path, 3720; interpolation=:free_energy).ion.interpolation ===
              :free_energy
    end

    @testset "rejected files and tables" begin
        table = M.library_table(4, 3)
        path = CL.write_sesame(joinpath(dir, "one.ses"), table)
        err = try
            CL.read_sesame(path, 3719)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("holds 3720", err.msg)
        lines = readlines(path)
        write(joinpath(dir, "ascii1.ses"), join(lines[2:end], "\n") * "\n")
        err = try
            CL.read_sesame(joinpath(dir, "ascii1.ses"), 3720)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("ASCII 1", err.msg)
        write(joinpath(dir, "cut.ses"), join(lines[1:end-1], "\n") * "\n")
        @test_throws ArgumentError CL.read_sesame(joinpath(dir, "cut.ses"), 3720)
        write(joinpath(dir, "no301.ses"),
              "Version 2.0\n0 5 201 5 0 0 1\n1 2 3 4 5\n")
        err = try
            CL.read_sesame(joinpath(dir, "no301.ses"), 5)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("no 301", err.msg)
        # A 301 record whose word count fits neither two nor three arrays.
        write(joinpath(dir, "short301.ses"),
              "Version 2.0\n0 5 301 9 0 0 1\n2 2 1 2 1\n2 1 2 3\n")
        @test_throws ArgumentError CL.read_sesame(joinpath(dir, "short301.ses"), 5)
        @test_throws ArgumentError CL.write_sesame(joinpath(dir, "twice.ses"), table, table)
        c = table.total
        @test_throws ArgumentError CL.SesameComponent(; temperature=c.temperature,
            density=c.density, p=c.p, e=c.e, interpolation=:free_energy)
        @test_throws ArgumentError CL.SesameComponent(; temperature=c.temperature,
            density=c.density, p=c.p, e=c.e, interpolation=:cubic)
        @test_throws DimensionMismatch CL.SesameComponent(; temperature=c.temperature,
            density=c.density, p=c.p[1:2, :], e=c.e)
        @test_throws ArgumentError CL.table_value(CL.SesameComponent(;
            temperature=c.temperature, density=c.density, p=c.p, e=c.e),
            :free_energy, 1.0e3, 1.0e2)
        @test_throws ArgumentError CL.table_value(c, :zbar, 1.0e3, 1.0e2)
        @test_throws ArgumentError CL.table_temperature_status(
            CL.SesameTable(1, c), 1.0e6, 1.0e2, :ion)
    end

    @testset "nodes at zero dropped" begin
        # A 301 record whose axes open at ρ = 0 and T = 0, as many do.
        ρs, Ts = [0.0, 1.0, 2.0], [0.0, 100.0, 200.0]
        field(f) = vec([f(T, ρ) for ρ in ρs, T in Ts])    # density fastest
        words = [3.0; 3.0; ρs; Ts; field((T, ρ) -> ρ * T); field((T, ρ) -> T + ρ);
                 field((T, ρ) -> -T)]
        path = joinpath(dir, "zeros.ses")
        open(path, "w") do io
            println(io, "Version 2.0")
            CL._write_sesame_record(io, 0, 7, 301, (0, 0, 1),
                                    CL._sesame_word.(words, 1.0))
        end
        c = CL.read_sesame(path, 7).total
        @test c.dropped == (1, 1)
        @test c.temperature == [100.0, 200.0] && c.density == [1.0e3, 2.0e3]
        @test c.e == [101.0 102.0; 201.0 202.0] .* 1e6
        @test c.p == [100.0 200.0; 200.0 400.0] .* 1e9
    end

    @testset "nodes and interpolation order ($interpolation)" for interpolation in
            (:bilinear, :free_energy)
        table = M.library_table(13, 9; interpolation)
        for part in parts
            c = getfield(table, part)
            for j in eachindex(c.density), i in eachindex(c.temperature)
                T, ρ = c.temperature[i], c.density[j]
                p, _, _, status = CL.table_value(c, :p, T, ρ)
                e = CL.table_value(c, :e, T, ρ)[1]
                @test status == CL.TABLE_OK
                if interpolation === :bilinear
                    @test p == c.p[i, j] && e == c.e[i, j]
                else
                    scale = max(abs(c.free_energy[i, j]), abs(c.e[i, j]))
                    @test abs(e - c.e[i, j]) <= 1e-14 * scale
                    @test isapprox(p, c.p[i, j]; rtol=1e-14)
                end
            end
        end
        # The error at a third of each cell of a 17 × 13 grid against the
        # analytic model: at a third of a cell of the 65-node grid and at two
        # thirds of one of the 129-node grid, where the leading error terms of
        # both interpolants have the same magnitude.
        function error_at_thirds(nt, nd)
            c = M.component(:total, nt, nd; interpolation)
            coarse = M.component(:total, 17, 13)
            x, y = coarse.log_temperature, coarse.log_density
            A, e, p = M.PARTS.total
            worst = zeros(2)
            for j in 1:length(y)-1, i in 1:length(x)-1
                T = exp((2x[i] + x[i+1]) / 3)
                ρ = exp((2y[j] + y[j+1]) / 3)
                worst[1] = max(worst[1], abs(CL.table_value(c, :e, T, ρ)[1] / e(T, ρ) - 1))
                worst[2] = max(worst[2], abs(CL.table_value(c, :p, T, ρ)[1] / p(T, ρ) - 1))
            end
            return worst
        end
        orders = log2.(error_at_thirds(65, 49) ./ error_at_thirds(129, 97))
        low, high = interpolation === :bilinear ? (1.95, 2.3) : (2.8, 3.4)
        @test all(o -> low < o < high, orders)
    end

    @testset "derivatives and consistency ($interpolation)" for interpolation in
            (:bilinear, :free_energy)
        table = M.library_table(13, 9; interpolation)
        c = table.total
        δ = 1e-6
        for (T, ρ) in ((2.3e3, 3.1e2), (7.7e5, 1.7e4), (4.1e2, 2.2e1))
            for name in (:e, :p, :free_energy)
                f, f_T, f_ρ, status = CL.table_value(table, name, T, ρ)
                @test status == CL.TABLE_OK
                fd_T = (CL.table_value(c, name, T * (1 + δ), ρ)[1] -
                        CL.table_value(c, name, T * (1 - δ), ρ)[1]) / (2T * δ)
                fd_ρ = (CL.table_value(c, name, T, ρ * (1 + δ))[1] -
                        CL.table_value(c, name, T, ρ * (1 - δ))[1]) / (2ρ * δ)
                @test isapprox(f_T, fd_T; rtol=1e-6, atol=1e-8 * abs(f) / T)
                @test isapprox(f_ρ, fd_ρ; rtol=1e-6, atol=1e-8 * abs(f) / ρ)
            end
            state = CL.table_state(table, T, ρ)
            @test (state.e, state.cv, state.de_drho)[1:2] ==
                  CL.table_value(c, :e, T, ρ)[1:2]
            @test state.c2 > 0 && state.cv > 0 && state.status == CL.TABLE_OK
            @test state.entropy ≈ (state.e - state.free_energy) / T
        end
        # The consistency residual ρ² ∂e/∂ρ - (p - T ∂p/∂T), relative to the
        # largest of its three terms, at a third of cells of a 17 × 13 grid:
        # round-off from one free-energy interpolant, and first order in the
        # spacing from independent bilinear fields. Refining four times keeps
        # each point at a third of a cell.
        function residuals(n)
            c = M.component(:total, n, (3n + 1) ÷ 4; interpolation)
            x, y = M.component(:total, 17, 13).log_temperature,
                   M.component(:total, 17, 13).log_density
            return map(Iterators.product(1:16, 1:12)) do (i, j)
                T, ρ = exp((2x[i] + x[i+1]) / 3), exp((2y[j] + y[j+1]) / 3)
                state = CL.table_state(c, T, ρ)
                scale = max(abs(state.p), abs(T * state.dp_dT),
                            abs(ρ^2 * state.de_drho))
                abs(state.consistency) / scale
            end
        end
        coarse, fine = residuals(65), residuals(257)
        if interpolation === :free_energy
            @test maximum(fine) < 1e-14 && maximum(coarse) < 1e-14
        else
            # From ρ₀ upward the cold pressure, growing as ρ³, keeps these grids
            # short of the asymptotic order, which they approach from below.
            orders = log.(4, coarse ./ fine)[:, 1:7]
            @test all(o -> 0.9 < o < 1.15, orders)
            @test maximum(fine) > 1e-3
        end
    end

    @testset "temperature inversion ($interpolation)" for interpolation in
            (:bilinear, :free_energy)
        table = M.library_table(13, 9; interpolation)
        for part in parts
            c = getfield(table, part)
            for j in eachindex(c.density), i in eachindex(c.temperature)
                T, ρ = c.temperature[i], c.density[j]
                e = CL.table_value(c, :e, T, ρ)[1]
                @test CL.table_temperature_status(c, e, ρ) == (T, CL.TABLE_OK)
            end
        end
        for (T, ρ) in ((2.3e3, 3.1e2), (7.7e5, 1.7e4), (4.1e2, 2.2e1), (9.9e5, 1.0e1))
            e = CL.table_value(table, :e, T, ρ)[1]
            T_rec, status = CL.table_temperature_status(table, e, ρ)
            @test isapprox(T_rec, T; rtol=1e-12) && status == CL.TABLE_OK
            @test isapprox(CL.table_value(table, :e, T_rec, ρ)[1], e; rtol=1e-14)
            e_ion = CL.table_value(table.ion, :e, T, ρ)[1]
            @test CL.table_temperature(table, e_ion, ρ, :ion) ≈ T
        end
        @test_throws ArgumentError CL.table_temperature_status(table, 1.0, 1.0, :cold)
    end

    @testset "domain policy ($interpolation)" for interpolation in
            (:bilinear, :free_energy)
        for (extrapolate, flag) in ((:missing, CL.TABLE_OUT_OF_DOMAIN),
                                    (:linear, CL.TABLE_EXTRAPOLATED))
            c = M.component(:total, 9, 7; interpolation, extrapolate)
            Tlo, Thi = c.temperature[1], c.temperature[end]
            ρlo, ρhi = c.density[1], c.density[end]
            for (T, ρ, axis) in ((Tlo / 2, 1.0e3, CL.TABLE_TEMPERATURE_AXIS),
                                 (Thi * 2, 1.0e3, CL.TABLE_TEMPERATURE_AXIS),
                                 (1.0e4, ρhi * 2, CL.TABLE_DENSITY_AXIS),
                                 (1.0e4, ρlo / 2, CL.TABLE_DENSITY_AXIS))
                e, _, _, status = CL.table_value(c, :e, T, ρ)
                @test isfinite(e) && status == flag | axis
                T_rec, status = CL.table_temperature_status(c, e, ρ)
                @test isapprox(T_rec, T; rtol=1e-10) && status == flag | axis
            end
            @test CL.table_value(c, :p, Thi, ρhi)[4] == CL.TABLE_OK
            for (T, ρ) in ((0.0, 1.0e3), (-1.0, 1.0e3), (1.0e4, 0.0), (NaN, 1.0e3))
                value, _, _, status = CL.table_value(c, :e, T, ρ)
                @test isnan(value) && status & CL.TABLE_OUT_OF_DOMAIN != 0
            end
            @test isnan(CL.table_temperature(c, NaN, 1.0e3))
            @test CL.table_temperature_status(c, 1.0e6, 0.0)[2] ==
                  CL.TABLE_OUT_OF_DOMAIN | CL.TABLE_DENSITY_AXIS
        end
    end

    @testset "parts and cold curve" begin
        table = M.library_table(9, 7)
        # 301 = 303 + 304 and 303 = 305 + 306 on a shared grid, node by node.
        for name in (:p, :e, :free_energy)
            total = getfield(table.total, name)
            @test total ≈ getfield(table.ion, name) .+ getfield(table.electron, name)
            cold = getfield(table.cold, name)
            @test getfield(table.ion, name) ≈
                  getfield(table.nuclear, name) .+ reshape(cold, 1, :)
        end
        cold = table.cold
        for j in eachindex(cold.density)
            @test CL.table_value(cold, :p, cold.density[j]) ==
                  (cold.p[j], CL.table_value(cold, :p, cold.density[j])[2], CL.TABLE_OK)
        end
        ρ = sqrt(cold.density[3] * cold.density[4])
        p, dp, status = CL.table_value(cold, :p, ρ)
        @test status == CL.TABLE_OK &&
              min(cold.p[3], cold.p[4]) < p < max(cold.p[3], cold.p[4])
        @test dp ≈ (cold.p[4] - cold.p[3]) / (log(cold.density[4] / cold.density[3]) * ρ)
        @test CL.table_value(cold, :e, cold.density[end] * 2)[3] ==
              CL.TABLE_OUT_OF_DOMAIN | CL.TABLE_DENSITY_AXIS
        @test isnan(CL.table_value(cold, :e, -1.0)[1])
        @test_throws ArgumentError CL.table_value(cold, :cv, ρ)
    end
end
