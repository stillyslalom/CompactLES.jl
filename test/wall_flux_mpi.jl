# Included by mpi_tests.jl: the hooks must be harmless on nonowners, and the
# corrected flux must enter the distributed compact solve, not just edge rows.
#
# The slip-wall case carries a tangential velocity in its initial data, so
# that the tangential momentum fluxes its contract zeros are nonzero to
# begin with.
using CompactLES: compute_rhs!, apply_bcs!, padded_index

function test_no_slip_wall_flux()
    section("no-slip wall flux: impermeability and distributed divergence")
    for T in (Float64, Float32), ax in 1:3, iso in (false, true)
        # Alternate channels across axes; the serial tests isolate nonzero
        # artificial coefficients independently of the detector response.
        channel = (:fickian, :bulk, :partial_density)[ax]
        wall = NoSlipWallBC(Twall=iso ? T(2) : T(NaN))
        bcs = ntuple(d -> d == ax ? (wall, wall) : per3[d], 3)
        ng = ntuple(d -> d == ax ? SPLITN : 1, 3)
        eos = IdealMixture([IdealSpecies{T}("a", T(1), T(1.4)),
                            IdealSpecies{T}("b", T(0.7), T(1.3))])
        function build_wall(comm_here, dims_here)
            sol = Solver(n_global=ng, L_domain=(one(T), one(T), one(T)),
                       bcs=bcs, eos=eos, comm=comm_here, dims=dims_here,
                       transport=ConstantTransport{T}(mu0=T(0.01)),
                       art=ArtificialProperties{T}(enabled=true, species_flux=channel),
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
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
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

function test_slip_wall_flux()
    section("slip wall flux: the symmetry plane and the distributed divergence")
    for T in (Float64, Float32), ax in 1:3
        channel = (:fickian, :bulk, :partial_density)[ax]
        wall = SlipWallBC()
        bcs = ntuple(d -> d == ax ? (wall, wall) : per3[d], 3)
        ng = ntuple(d -> d == ax ? SPLITN : 1, 3)
        eos = IdealMixture([IdealSpecies{T}("a", T(1), T(1.4)),
                            IdealSpecies{T}("b", T(0.7), T(1.3))])
        function build_slip(comm_here, dims_here)
            sol = Solver(n_global=ng, L_domain=(one(T), one(T), one(T)),
                       bcs=bcs, eos=eos, comm=comm_here, dims=dims_here,
                       transport=ConstantTransport{T}(mu0=T(0.01)),
                       art=ArtificialProperties{T}(enabled=true, species_flux=channel),
                       deriv=lele_d1_6(T), filt=compact_filter(T(0.45), T),
                       filter_interval=0)
            state = allocate_state(sol)
            initialize!(sol, state, (x, y, z) -> begin
                r = (x, y, z)[ax]
                a = T(0.4) + T(0.1) * cos(T(2pi) * r)
                # Odd normal velocity, even tangential velocity: the
                # symmetry the contract rests on.
                vel = ntuple(d -> d == ax ? T(0.1) * sin(T(pi) * r) :
                                            T(0.08) * cos(T(pi) * r), 3)
                Prim(rho=one(T) + T(0.05) * sin(T(pi) * r),
                     T_ion=T(2) + T(0.1) * r, Y=(a, 1-a), u=vel)
            end)
            apply_bcs!(sol, state)
            rhs = zero(state)
            compute_rhs!(sol, state, rhs)
            return sol, state, rhs
        end
        s, Q, dQ = build_slip(comm, splitdims(ax))
        ref, Qref, dQref = build_slip(MPI.COMM_SELF, (1, 1, 1))
        err = 0.0
        for I in CL.interior(s.decomp), c in 1:s.equations.n_cons
            loc = Tuple(I) .- s.decomp.n_halo_d
            J = padded_index(ref, (loc .+ s.decomp.offset)...)
            err = max(err, abs(Float64(dQ[I, c] - dQref[J, c])))
        end
        tol = T === Float32 ? 5e-4 : 1e-8
        label = "$T dim=$ax"
        check("slip-corrected distributed RHS $label", gmax(err), tol)
        leakage = 0.0
        traction = 0.0
        normal = 0.0
        for side in 1:2
            plane = CL.wallplane(s.decomp, ax, side)
            plane === nothing && continue
            for I in plane
                for sp in 1:s.equations.n_species
                    leakage = max(leakage, abs(Float64(s.flux[ax, sp][I])))
                end
                leakage = max(leakage,
                              abs(Float64(s.flux[ax, s.equations.i_energy][I])))
                for t in 1:3
                    t == ax && continue
                    traction = max(traction,
                                   abs(Float64(s.flux[ax, s.equations.i_mom[t]][I])))
                end
                normal = max(normal,
                             abs(Float64(s.flux[ax, s.equations.i_mom[ax]][I])))
            end
        end
        check("zero normal species and energy flux $label", gmax(leakage), 1e-300)
        check("zero tangential traction $label", gmax(traction), 1e-300)
        # The reciprocal is infinite, and the check fails, if the hook has
        # also zeroed the normal momentum flux the wall carries.
        check("normal traction retained $label", 1 / gmax(normal), 10.0)
        owns_neither = all(side -> CL.wallplane(s.decomp, ax, side) === nothing, 1:2)
        check("interior ranks participate $label",
              abs(gsum(Int(owns_neither)) - max(np-2, 0)), 0.5)
    end
end
