"""Physical-invariant tests for the column energy solver.

The full solver can't be pinned to GEOtop end-to-end (it needs the surface
turbulence machinery), so it is validated by properties that must hold for any
correct discretisation: sensible-energy conservation under insulation, a fixed
point at steady state, relaxation toward the boundary, and latent-heat
accounting during melt. The constitutive laws inside are the oracle-pinned ones.
"""


import pytest

from geotop_py import constants as C
from geotop_py import laws
from geotop_py.energy.column import (
    EnergyColumn,
    SoilLayer,
    SolverOptions,
    solve_column_energy,
)
from geotop_py.point.step import (
    _merge_energy_snow_layer,
    _needs_surface_melt_retry,
    _remove_last_snow_for_soil_solve,
    _sux_verdict,
)


def warm_soil_layer(th_sat=0.4, th=0.4):
    """A saturated soil layer above freezing -> no phase change room."""
    return SoilLayer(sat=th_sat, res=0.05, alpha=0.004, n=1.4, ss=1e-3,
                     kt=2.5, ct=2.3e6, th0=th, thi0=0.0, Tstar=0.0)


def warm_soil_column(temps, D=0.1, insulated=True):
    """Build an all-soil column at the given (positive) temperatures.

    liq is set consistent with saturation so thw == th_sat in k_thermal.
    """
    n = len(temps)
    Dlayer = [0.0] + [D] * n
    soil = [None] + [warm_soil_layer() for _ in range(n)]
    # volumetric liquid th_sat -> liq [kg/m2] = th * rho_w * D(m)
    liq = [0.0] + [s.sat * C.rho_w * D for s in soil[1:]]
    ice = [0.0] * (n + 1)
    T0 = [0.0] + list(temps)
    return EnergyColumn(
        Dlayer=Dlayer, ice=ice, liq=liq, T0=T0, nsng=0, soil=soil,
        alpha_snow=1.0, snow_conductivity=1,
        Tboundary=temps[-1], Zboundary=1e9 if insulated else 0.5,
        Fboundary=0.0,
    )


def sensible_energy(col, Temp):
    """sum C(T)*D*T over the column -- the conserved quantity with no phase
    change and no flux."""
    tot = 0.0
    for l in range(1, col.n + 1):
        Cc = laws.C_soil(col.soil[l].ct, col.soil[l].sat,
                         col.ice[l], col.liq[l], 0.0, 1.0, col.Dlayer[l])
        tot += Cc * col.Dlayer[l] * Temp[l]
    return tot


def no_flux(Tg):
    return 0.0, 0.0


def test_massless_skin_is_distinct_from_first_material_node():
    col = warm_soil_column([0.0], D=0.1, insulated=True)
    col.surface_index = 0

    # Positive downward atmospheric flux, linearised exactly.
    def atmospheric_flux(Tg):
        return 20.0 * (10.0 - Tg), -20.0

    res = solve_column_energy(col, Dt=3600.0, surface_flux=atmospheric_flux)
    assert res.converged
    assert res.Temp[0] > res.Temp[1]
    assert res.Temp[1] > 0.0


def test_energy_merge_conserves_snow_enthalpy_and_shifts_soil():
    soil = [None, warm_soil_layer()]
    col = EnergyColumn(
        Dlayer=[0.0, 0.02, 0.03, 0.1], ice=[0.0, 2.0, 4.0, 40.0],
        liq=[0.0, 0.1, 0.2, 0.0], T0=[-1.0, -2.0, -4.0, 5.0],
        nsng=2, soil=soil, alpha_snow=1e5, snow_conductivity=1,
        Tboundary=5.0, Zboundary=1e9, Fboundary=0.0, surface_index=0)
    from geotop_py.energy.column import SolverResult
    res = SolverResult(list(col.T0), [0.0] * 4, 1, 0.0, True)
    h0 = sum(laws.internal_energy(col.ice[l], col.liq[l], res.Temp[l])
             for l in (1, 2))
    merged = _merge_energy_snow_layer(col, res, 1)
    h1 = laws.internal_energy(merged.ice[1], merged.liq[1], merged.T0[1])
    assert h1 == pytest.approx(h0, abs=1e-6)
    assert merged.nsng == 1
    assert merged.Dlayer[2] == pytest.approx(0.1)
    assert merged.T0[2] == pytest.approx(5.0)


def test_last_snow_retry_updates_temporary_top_soil_phase_state():
    soil = [None, warm_soil_layer(th_sat=0.5, th=0.3)]
    col = EnergyColumn(
        Dlayer=[0.0, 0.01, 0.1], ice=[0.0, 2.0, 0.0],
        liq=[0.0, 0.0, 30.0], T0=[-0.1, -0.1, 2.0],
        nsng=1, soil=soil, alpha_snow=1e5, snow_conductivity=1,
        Tboundary=2.0, Zboundary=1e9, Fboundary=0.0, surface_index=0)
    from geotop_py.energy.column import SolverResult
    res = SolverResult(list(col.T0), [0.0] * 3, 1, 0.0, True)
    bare, _ = _remove_last_snow_for_soil_solve(col, res)

    assert bare.nsng == 0
    assert bare.soil[1].th0 == pytest.approx(
        bare.liq[1] / (C.rho_w * bare.Dlayer[1]))
    assert bare.soil[1].thi0 == pytest.approx(
        bare.ice[1] / (C.rho_w * bare.Dlayer[1]))
    # The original persistent soil state must not be mutated by the retry.
    assert col.soil[1].th0 == pytest.approx(0.3)
    assert col.soil[1].thi0 == pytest.approx(0.0)


def test_nonconverged_multilayer_skin_requests_surface_melt_retry():
    from geotop_py.energy.column import SolverResult
    failed = SolverResult([-0.01, -0.02], [0.0, 0.0], 900, 0.2, False)
    assert _needs_surface_melt_retry(failed, ns=2, ng=0, surface_index=0)
    # A single snow layer follows GEOtop's separate sux=-6/-5 branch instead --
    # and that is judged on the snow count alone, glacier below or not.
    assert not _needs_surface_melt_retry(failed, ns=1, ng=0, surface_index=0)
    assert not _needs_surface_melt_retry(failed, ns=1, ng=2, surface_index=0)
    # Once already solving on material node 1, sux=-1 must not recurse.
    assert not _needs_surface_melt_retry(failed, ns=2, ng=0, surface_index=1)


def test_nonconverged_last_snow_node_is_merged_away():
    """energy.balance.cc:1827-1841 -- a Newton that runs out of iterations is
    NOT accepted while a single snow node is left: GEOtop reads the failure as
    evidence that the node should not be there and returns the merge code that
    removes it (-6 over bare soil, -5 over a glacier), re-solving the shortened
    column.  Only with no snow at all does it keep the non-converged state."""
    from geotop_py.energy.column import SolverResult

    class _Col:
        surface_index = 0
        ice = [0.0, 1.0]
        liq = [0.0, 0.0]

    failed = SolverResult([-0.01, -0.02], [0.0, 0.0], 900, 0.2, False)
    # one snow node, bare soil -> fold into the soil
    assert _sux_verdict(failed, _Col(), ns=1, ng=0, sur=0)[0] == -6
    # one snow node over a glacier -> merge into the glacier
    assert _sux_verdict(failed, _Col(), ns=1, ng=2, sur=0)[0] == -5
    # no snow at all -> GEOtop only prints a warning (code 1)
    assert _sux_verdict(failed, _Col(), ns=0, ng=0, sur=0)[0] == 1
    # several snow nodes -> the material-node retry comes first
    assert _sux_verdict(failed, _Col(), ns=3, ng=0, sur=0)[0] == -1
    # a converged solve with plenty of ice left is accepted
    ok = SolverResult([-0.01, -0.02], [0.0, 0.0], 5, 1e-4, True)
    assert _sux_verdict(ok, _Col(), ns=1, ng=0, sur=0) == (0, None)


# --- steady state --------------------------------------------------------

def test_uniform_insulated_is_fixed_point():
    col = warm_soil_column([5.0, 5.0, 5.0, 5.0], insulated=True)
    res = solve_column_energy(col, Dt=3600.0, surface_flux=no_flux)
    assert res.converged
    for l in range(1, col.n + 1):
        assert res.Temp[l] == pytest.approx(5.0, abs=1e-6)
    assert all(abs(res.deltaw[l]) < 1e-9 for l in range(1, col.n + 1))


# --- sensible energy conservation ---------------------------------------

def test_insulated_conserves_sensible_energy():
    col = warm_soil_column([8.0, 5.0, 3.0, 2.0], insulated=True)
    E0 = sensible_energy(col, col.T0)
    res = solve_column_energy(col, Dt=1800.0, surface_flux=no_flux)
    assert res.converged
    E1 = sensible_energy(col, res.Temp)
    assert E1 == pytest.approx(E0, rel=1e-6)
    # profile moved toward the mean (diffusion), i.e. top cooled, bottom warmed
    assert res.Temp[1] < 8.0
    assert res.Temp[col.n] > 2.0


def test_diffusion_reduces_gradient():
    col = warm_soil_column([10.0, 6.0, 4.0, 1.0], insulated=True)
    spread0 = max(col.T0[1:]) - min(col.T0[1:])
    res = solve_column_energy(col, Dt=1800.0, surface_flux=no_flux)
    spread1 = max(res.Temp[1:]) - min(res.Temp[1:])
    assert spread1 < spread0


def test_relaxes_to_uniform_over_many_steps():
    col = warm_soil_column([9.0, 7.0, 5.0, 3.0], insulated=True)
    Temp = col.T0
    for _ in range(400):
        col.T0 = Temp
        res = solve_column_energy(col, Dt=3600.0, surface_flux=no_flux)
        assert res.converged
        Temp = res.Temp
    spread = max(Temp[1:]) - min(Temp[1:])
    assert spread < 1e-2
    # mean temperature preserved (insulated)
    assert sum(Temp[1:]) / col.n == pytest.approx(6.0, abs=1e-2)


# --- surface forcing -----------------------------------------------------

def test_surface_cooling_pulls_top_down():
    col = warm_soil_column([5.0, 5.0, 5.0, 5.0], insulated=True)

    def cooling(Tg):
        # linearised loss proportional to surface temperature: EB = -k*Tg
        k = 20.0
        return -k * Tg, -k
    res = solve_column_energy(col, Dt=1800.0, surface_flux=cooling)
    assert res.converged
    assert res.Temp[1] < res.Temp[col.n]        # top colder than bottom
    for l in range(1, col.n):
        assert res.Temp[l] <= res.Temp[l + 1] + 1e-9   # monotone profile


# --- latent heat during melt --------------------------------------------

def snow_column_at(T0, swe_per_layer=20.0, D=0.2, nlayers=3, alpha_snow=1.0):
    """Snow column initialised in freezing-curve equilibrium at T0.

    The liquid fraction is set to theta_snow(T0) so the pack starts consistent
    with the freezing curve; otherwise deltaw jumps on the first step (an
    unphysical transient rather than a solver defect).
    """
    n = nlayers
    th = laws.theta_snow(alpha_snow, 1.0, T0)
    liq_l = th * swe_per_layer
    ice_l = swe_per_layer - liq_l
    Dlayer = [0.0] + [D] * n
    ice = [0.0] + [ice_l] * n
    liq = [0.0] + [liq_l] * n
    T = [0.0] + [T0] * n
    return EnergyColumn(
        Dlayer=Dlayer, ice=ice, liq=liq, T0=T, nsng=n, soil=[None],
        alpha_snow=alpha_snow, snow_conductivity=3,
        Tboundary=T0, Zboundary=1e9, Fboundary=0.0,
    )


def test_snow_absorbs_energy_and_warms():
    col = snow_column_at(-2.0)

    def warming(Tg):
        return 50.0, 0.0        # constant positive surface flux

    res = solve_column_energy(col, Dt=600.0, surface_flux=warming,
                              options=SolverOptions(maxiter_energy=400))
    assert res.converged
    # a positive surface flux raises the surface layer's liquid fraction
    # (melting): deltaw at the top is non-negative
    assert res.deltaw[1] >= -1e-9
    # surface layer warmed toward 0
    assert res.Temp[1] >= -2.0 - 1e-9


def test_cold_snow_pure_conduction_converges():
    col = snow_column_at(-10.0)

    def small_flux(Tg):
        return 5.0, 0.0

    res = solve_column_energy(col, Dt=300.0, surface_flux=small_flux)
    assert res.converged
    assert res.residual <= 1e-5
