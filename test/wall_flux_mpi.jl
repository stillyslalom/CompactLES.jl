# Included by mpi_tests.jl: the hooks must be harmless on nonowners, and the
# corrected flux must enter the distributed compact solve, not just edge rows.
function test_no_slip_wall_flux()
    section("no-slip wall flux: impermeability and distributed divergence")
    for T in (Float64, Float32), ax in 1:3, iso in (false, true)
        # Alternate channels across axes; the serial tests isolate nonzero
        # artificial coefficients independently of the detector response.
        channel = ax == 2 ? :bulk : :fickian
        wall = NoSlipWallBC(Twall=iso ? T(2) : T(NaN))
        bcs = ntuple(d -> d == ax ? (wall, wall) : per3[d], 3)
        ng = ntuple(d -> d == ax ? SPLITN : 1, 3)
        eos = IdealMixture([IdealSpecies{T}("a", T(1), T(1.4)),
                            IdealSpecies{T}("b", T(0.7), T(1.3))])
        function build_wall(comm_here, dims_here)
            sol = Solver(n_global=ng, L_domain=(one(T), one(T), one(T)),
                       bcs=bcs, eos=eos, comm=comm_here, dims=dims_here,
                       transport=Transport{T}(mu0=T(0.01)),
                       art=ArtParams{T}(enabled=true, species_flux=channel),
                       deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T),
                       filter_interval=0)
            state = allocate_state(sol)
            initialize!(sol, state, (x, y, z) -> begin
                r = (x, y, z)[ax]
                a = T(0.4) + T(0.1) * cos(T(2pi) * r)
                Prim(rho=one(T) + T(0.05) * sin(T(pi) * r),
                     T_ion=T(2) + T(0.1) * r, Y=(a, 1-a))
            end)
            apply_bcs!(sol, state)
            rhs = zero(state)
            compute_rhs!(sol, state, rhs)
            return sol, state, rhs
        end
        s, Q, dQ = build_wall(comm, splitdims(ax))
        # Each rank builds its own full reference on COMM_SELF: no rank can
        # skip the distributed calls while a peer is waiting in their solve.
        ref, Qref, dQref = build_wall(MPI.COMM_SELF, (1, 1, 1))
        err = 0.0
        for I in CL.interior(s.decomp), c in 1:s.equations.n_cons
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = gidx(ref, (loc .+ s.decomp.offset)...)
            err = max(err, abs(Float64(dQ[I, c] - dQref[J, c])))
        end
        tol = T === Float32 ? 5e-4 : 1e-8
        label = "$T dim=$ax iso=$iso"
        check("wall-corrected distributed RHS $label", gmax(err), tol)
        leakage = 0.0
        heaterr = 0.0
        for side in 1:2
            plane = CL.wallplane(s.decomp, ax, side)
            plane === nothing && continue
            for I in plane
                for sp in 1:s.equations.n_species
                    leakage = max(leakage, abs(Float64(s.flux[ax, sp][I])))
                end
                heat = iso ? -(s.transport.mu0 * s.cp_mix[I] / s.transport.Pr +
                               s.kappa_art[I]) * s.grad_T_ion[ax][I] : zero(T)
                delta_heat = s.flux[ax, s.equations.i_energy][I] - heat
                heaterr = max(heaterr, abs(Float64(delta_heat)))
            end
        end
        check("zero normal species flux $label", gmax(leakage), 1e-300)
        check("prescribed normal heat flux $label", gmax(heaterr), 1e-300)
        owns_neither = all(side -> CL.wallplane(s.decomp, ax, side) === nothing, 1:2)
        check("interior ranks participate $label",
              abs(gsum(Int(owns_neither)) - max(np-2, 0)), 0.5)
    end
end
