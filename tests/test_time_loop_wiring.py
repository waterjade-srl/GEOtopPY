"""``point_step.simulate_energy_balance`` -- the wiring between the generic
``time_loop.run`` subdivision and ``step_independent``.

No oracle pin (``step_independent`` itself has none -- see its module
docstring: the surface energy balance is not bit-matchable without
``EnergyFluxes``); verified instead by consistency against calling
``step_independent`` directly at a fixed Dt (the ``WaterBalance=0``,
always-converges case, where the subdivision loop must be a no-op) and by
the ``Column1D``/``SurfaceState`` copy machinery not aliasing state between
trials.
"""

import pytest

from geotop_py.energy.column import SoilLayer
from geotop_py.energy.surface import SurfaceState, SurfaceStatics
from geotop_py.meteo.step import Meteo
from geotop_py.point.state import Column1D
from geotop_py.point.step import NominalStep, simulate_energy_balance, step_independent
from geotop_py.snow.state import SnowColumn

ALPHA_SNOW = 1e5


def soil_layer():
    return SoilLayer(sat=0.4, res=0.05, alpha=0.004, n=1.4, ss=1e-3,
                     kt=2.5, ct=2.3e6, th0=0.3, thi0=0.0, Tstar=0.0)


def bare_column(nsoil=3, D=0.2, soilT=2.0):
    soil = [None] + [soil_layer() for _ in range(nsoil)]
    return Column1D(
        snow=SnowColumn(max=15, lnum=0, type=0),
        soil=soil, soil_D=[0.0] + [D] * nsoil, soil_T=[0.0] + [soilT] * nsoil,
        alpha_snow=ALPHA_SNOW, snow_conductivity=3, slope=0.0,
        Tboundary=soilT, Zboundary=1.0, Fboundary=0.0,
        max_weq_snow=5.0, maxSWE=1e10,
    )


def statics(nsoil=3, D=0.2):
    # soil_D_mm is the parameter table's own thickness row, in mm: the
    # Ye--Pielke molecular resistance reads it directly and indexes it by soil
    # layer, so it must be as long as the column has soil layers.
    return SurfaceStatics(lat=46.1, lon=11.1, ST=1.0, sky=1.0,
                          soil_D_mm=[D * 1000.0] * nsoil)


def meteo_at(JDb, JDe):
    return Meteo(Ta=5.0, RH=0.7, P=850.0, wind=2.0, tau_cloud=1.0)


def test_column1d_copy_is_independent():
    col = bare_column()
    dup = col.copy()
    dup.soil[1].th0 = 0.99
    dup.soil_T[1] = 999.0
    dup.snow.lnum = 3
    assert col.soil[1].th0 == 0.3
    assert col.soil_T[1] == 2.0
    assert col.snow.lnum == 0


def test_surface_state_copy_is_independent():
    st = SurfaceState()
    dup = st.copy()
    dup.snowage = 5.0
    dup.swrefl_surr = 3.0
    assert st.snowage == 0.0
    assert st.swrefl_surr == 0.0


def test_simulate_energy_balance_matches_direct_iteration_when_it_never_retries():
    """With a trial that always converges on the first try, the subdivision
    loop must be a no-op: one committed sub-step per nominal step, at exactly
    Dt_nominal, identical to calling step_independent directly."""
    Dt = 3600.0
    JD0 = 100.0
    n_steps = 4

    col_direct = bare_column()
    state_direct = SurfaceState()
    st = statics()
    direct_outputs = []
    t = 0.0
    for _ in range(n_steps):
        JDb = JD0 + t / 86400.0
        JDe = JD0 + (t + Dt) / 86400.0
        out = step_independent(col_direct, state_direct, st, Dt,
                               meteo_at(JDb, JDe), JDb, JDe)
        direct_outputs.append(out)
        t += Dt

    col_loop = bare_column()
    state_loop = SurfaceState()
    steps = [NominalStep(JD0=JD0 + i * Dt / 86400.0, Dt_nominal=Dt) for i in range(n_steps)]
    col_final, state_final, richards_final, loop_outputs = simulate_energy_balance(
        col_loop, state_loop, st, min_Dt=1.0, steps=steps, meteo_at=meteo_at)

    assert len(loop_outputs) == n_steps
    for direct, looped in zip(direct_outputs, loop_outputs):
        assert looped.Tg == pytest.approx(direct.Tg)
        assert looped.swe == pytest.approx(direct.swe)
        assert looped.converged == direct.converged

    assert col_final.soil_T[1] == pytest.approx(col_direct.soil_T[1])
    assert state_final.tsurf_prev == pytest.approx(state_direct.tsurf_prev)
    assert richards_final is None
