"""
    CompactLES.EOSTables

Tabulated plasma equations of state read from files: the IONMIX4 and IONMIX6
tables FLASH reads ([`IonmixTable`](@ref)) and SESAME ASCII 2 libraries
([`SesameTable`](@ref)), with the interpolants, domain flags and temperature
inversion they share. No `EOS` method reads either table, so a solver does not
evaluate them, and CompactLES does not export the module's names; load them with
`using CompactLES.EOSTables` or qualify them. The status flags, documented
under [`TABLE_OK`](@ref), are reached qualified.
"""
module EOSTables

using Printf
using ..CompactLES: _nasa9_rtol

export IonmixTable, read_ionmix, write_ionmix
export SesameTable, SesameComponent, SesameColdCurve, read_sesame, write_sesame
export table_value, table_opacity, table_state, table_temperature_status,
       table_temperature

include("ionmix.jl")
include("sesame.jl")

end # module EOSTables
