module CompactLESBriareusExt

# The in-memory hand-over from Briareus (the integrated hohlraum step) to CompactLES
# (the capsule-only step) of LLNL's two-step method. Briareus's `CapsuleDrive` is plain
# data (time grid, Tr, incident flux, Legendre ratios aₗ/a₀ with P2 > 0 pole-hot, the
# radius it was evaluated on, provenance); this module only reads it into the core
# `AblationDrive`, which knows nothing of Briareus. See
# docs/src/explanation/ablation-drive.md.

using CompactLES
using CompactLES: AblationDrive, ablation_pressure_scale
using Briareus: CapsuleDrive

"""
    radiation_drive(d::Briareus.CapsuleDrive; lmax = size(d.modes, 1) - 1,
                    pressure = :calibrate, peak = 145.0, kwargs...) -> AblationDrive

A stage-0 [`AblationDrive`](@ref) driven by Briareus's capsule radiation temperature
`d.Tr` on `d.time`, with the Legendre ratios `aₗ/a₀` for `l = 1…lmax` (`lmax = 0`
drops the angular structure). `pressure = :calibrate` scales the ablation pressure so
its largest value over the drive is `peak` Mbar ([`ablation_pressure_scale`](@ref));
a number sets the prefactor (Mbar at 100 eV and `t_ref`) directly. Other keywords go to
`AblationDrive` (exponents, `rate`, `t_on`, `species`, `units`).
"""
function CompactLES.radiation_drive(d::CapsuleDrive; lmax::Integer = size(d.modes, 1) - 1,
                                    pressure = :calibrate, peak::Real = 145.0, kwargs...)
    0 <= lmax <= size(d.modes, 1) - 1 ||
        throw(ArgumentError("the drive carries modes through l = $(size(d.modes, 1) - 1)"))
    p = if pressure === :calibrate
        keep = (:pressure_exponent, :time_exponent, :t_on, :t_ref, :t_min)
        ablation_pressure_scale(d.time, d.Tr; peak,
                                (k => v for (k, v) in pairs(kwargs) if k in keep)...)
    else
        Float64(pressure)
    end
    return AblationDrive(d.time, d.Tr; modes = d.modes[2:lmax+1, :], pressure = p, kwargs...)
end

end # module
