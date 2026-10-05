"""Verify evaporation and turbulence functions directly against GEOtop.

Cover dry, wet and high-temperature conditions, including surface-flux inputs.
"""

import math

import pytest

from geotop_py import constants as C
from geotop_py.energy.column import SoilLayer
from geotop_py.energy.turbulence import Businger, soil_evaporation_parameters

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


def _pa(dz_mm, **columns):
    """``pa[row][layer]`` matrix, both axes 1-based, ``io.soil.SoilParameters``
    layout. ``dz_mm`` sets ``jdz`` (layer thickness [mm]) -- required for the
    molecular-diffusion resistance; omitting it drives the resistance to zero
    and the oracle to NaN, which is a test-fixture bug, not a model one."""
    n = len(dz_mm)
    pa = [[0.0] * (n + 1) for _ in range(C.jss + 1)]
    for layer, v in enumerate(dz_mm, start=1):
        pa[C.jdz][layer] = v
    for name, values in columns.items():
        row = getattr(C, name)
        for layer, v in enumerate(values, start=1):
            pa[row][layer] = v
    return pa


def _strip(pa):
    return [row[1:] for row in pa[1:]]


def _layers(sat, res, alpha, n, fc):
    return [
        SoilLayer(sat=s, res=r, alpha=a, n=nn, ss=1.0e-6, kt=2.5, ct=2.3e6,
                  th0=0.0, thi0=0.0, Tstar=0.0, fc=f)
        for s, r, a, nn, f in zip(sat, res, alpha, n, fc)
    ]


def _compare_evap(layers, dz_mm, theta1, T1, psi, P, rv, Ta, Qa, Qgsat):
    pa = _pa(dz_mm, jsat=[l.sat for l in layers], jres=[l.res for l in layers],
             ja=[l.alpha for l in layers], jns=[l.n for l in layers],
             jss=[l.ss for l in layers], jfc=[l.fc for l in layers],
             jKn=[1.0e-4] * len(layers), jKl=[1.0e-4] * len(layers),
             jv=[0.5] * len(layers))
    theta = [0.0] + theta1
    T = [0.0] + T1
    mine = soil_evaporation_parameters(
        [None] + layers, [0.0] + list(dz_mm), T, theta,
        P, rv, Ta, Qa, Qgsat, psi_surface=psi)
    theirs = _cxx.find_actual_evaporation_parameters(
        theta, T, _strip(pa), psi, P, rv, Ta, Qa, Qgsat, nsnow=0)
    assert mine[0] == pytest.approx(theirs[0])
    assert mine[1] == pytest.approx(theirs[1])
    assert mine[2] == pytest.approx(theirs[2])


def test_ponding_matches_the_cxx():
    layers = _layers([0.4], [0.05], [0.004], [1.3], [0.3])
    _compare_evap(layers, [50.0], [0.35], [5.0], psi=15.0, P=900.0, rv=100.0,
                  Ta=5.0, Qa=0.003, Qgsat=0.006)


def test_saturated_matches_the_cxx():
    layers = _layers([0.4], [0.05], [0.004], [1.3], [0.3])
    _compare_evap(layers, [50.0], [0.40], [5.0], psi=-1.0, P=900.0, rv=100.0,
                  Ta=5.0, Qa=0.003, Qgsat=0.006)


def test_unsaturated_single_layer_matches_the_cxx():
    layers = _layers([0.4], [0.05], [0.004], [1.3], [0.3])
    _compare_evap(layers, [50.0], [0.30], [5.0], psi=-100.0, P=900.0, rv=100.0,
                  Ta=5.0, Qa=0.003, Qgsat=0.006)


def test_unsaturated_multilayer_below_field_capacity_matches_the_cxx():
    """All layers below jfc -- exercises the cosine-taper ``hs`` branch."""
    layers = _layers([0.42, 0.40, 0.38], [0.05, 0.05, 0.05],
                     [0.004, 0.004, 0.004], [1.3, 1.3, 1.3],
                     [0.28, 0.29, 0.30])
    _compare_evap(layers, [50.0, 80.0, 120.0], [0.20, 0.22, 0.25],
                  [5.0, 4.5, 4.0], psi=-500.0, P=900.0, rv=80.0, Ta=6.0,
                  Qa=0.0028, Qgsat=0.0065)


def test_unsaturated_multilayer_above_field_capacity_matches_the_cxx():
    """All layers above jfc -- exercises the ``hs = 1`` branch."""
    layers = _layers([0.42, 0.40, 0.38], [0.05, 0.05, 0.05],
                     [0.004, 0.004, 0.004], [1.3, 1.3, 1.3],
                     [0.28, 0.29, 0.30])
    _compare_evap(layers, [50.0, 80.0, 120.0], [0.35, 0.30, 0.28],
                  [5.0, 4.5, 4.0], psi=-50.0, P=900.0, rv=80.0, Ta=6.0,
                  Qa=0.0028, Qgsat=0.0065)


# ---------------------------------------------------------------- Businger

def _compare_businger(a, zmu, zmt, d0, z0, v, Ta, Tg, DQ, z0_z0t, maxiter):
    T = 0.5 * (Tg + Ta)
    DT = Tg - Ta
    mine = Businger(a, zmu, zmt, d0, z0, v, T, DT, DQ, z0_z0t, maxiter)
    theirs = _cxx.businger(a, zmu, zmt, d0, z0, v, T, DT, DQ, z0_z0t, maxiter)
    for m, t in zip(mine, theirs):
        if math.isinf(m) or math.isinf(t):
            assert m == t
        else:
            assert m == pytest.approx(t)


def test_businger_matches_the_cxx_near_neutral():
    _compare_businger(1, 10.0, 2.0, 0.0, 0.01, 2.0, Ta=10.0, Tg=10.5,
                      DQ=0.0002, z0_z0t=0.0, maxiter=5)


def test_businger_matches_the_cxx_stable():
    _compare_businger(1, 10.0, 2.0, 0.0, 0.005, 0.3, Ta=10.0, Tg=5.0,
                      DQ=-0.001, z0_z0t=0.0, maxiter=5)


def test_businger_matches_the_cxx_extreme_midday_instability():
    """Businger matches the cxx extreme midday instability."""
    _compare_businger(1, 10.0, 2.0, 0.0, 0.01, 1.0, Ta=25.0, Tg=80.0,
                      DQ=0.01, z0_z0t=0.0, maxiter=5)


def test_businger_matches_the_cxx_low_wind_extreme_instability():
    _compare_businger(1, 10.0, 2.0, 0.0, 0.001, 0.1, Ta=20.0, Tg=90.0,
                      DQ=0.02, z0_z0t=0.0, maxiter=3)


def test_businger_matches_the_cxx_bending_surface():
    _compare_businger(1, 10.0, 2.0, 0.5, 0.02, 2.0, Ta=15.0, Tg=15.0,
                      DQ=0.0, z0_z0t=0.5, maxiter=10)


def test_evaporation_parameters_follow_the_newton_trial_state():
    """Evaporation parameters follow the newton trial state."""
    from geotop_py.energy import rad
    from geotop_py.energy.surface import SurfaceDiag, _ground_fluxes

    layers = _layers([0.6000000000000001], [0.1], [0.001], [1.8],
                     [0.2541344964215667])
    dz_mm = [1000.0]
    P = 978.1119657230319
    Ta = 26.060007410831872
    Qa = 0.013632327650761823
    rv = 3271.012183679694
    psi = -4534.825045899992
    theta = [0.0, 0.2450181008600416]
    T_soil = [0.0, 17.907979307505236]
    Tg_trial = 106.88324817101854      # the Newton's hottest trial skin
    Tg_step = 17.663                   # what a per-step freeze would have used
    pa = _pa(dz_mm, jsat=[l.sat for l in layers], jres=[l.res for l in layers],
             ja=[l.alpha for l in layers], jns=[l.n for l in layers],
             jss=[l.ss for l in layers], jfc=[l.fc for l in layers],
             jKn=[1.0e-4] * len(layers), jKl=[1.0e-4] * len(layers),
             jv=[0.5] * len(layers))

    diag = SurfaceDiag(
        SWbeam=0.0, SWdiff=0.0, SWnet=0.0, SWup=0.0, LWin=350.0, cosinc=0.9,
        hsun=1.0, rh=rv, rv=rv, Lobukhov=-10.0, eps=0.96, Qa=Qa, P=P, Ta=Ta,
        rh_c=rv, evap_live=True, evap_soil=[None] + layers,
        evap_Dsoil=[0.0] + list(dz_mm),
        evap_Tsoil=list(T_soil), evap_theta=list(theta), evap_psi=psi,
        evap_nlayers=1)
    _ground_fluxes(diag, Tg_trial)

    Qg_trial = rad.spec_humidity(rad.sat_vap_pressure(Tg_trial, P), P)
    live = _cxx.find_actual_evaporation_parameters(
        theta, T_soil, _strip(pa), psi, P, rv, Ta, Qa, Qg_trial, nsnow=0)
    assert diag.evap_alpha == pytest.approx(live[0])
    assert diag.evap_beta == pytest.approx(live[1])
    assert diag.soil_evap == pytest.approx(live[2])

    Qg_step = rad.spec_humidity(rad.sat_vap_pressure(Tg_step, P), P)
    frozen = _cxx.find_actual_evaporation_parameters(
        theta, T_soil, _strip(pa), psi, P, rv, Ta, Qa, Qg_step, nsnow=0)
    assert abs(frozen[0] - live[0]) / live[0] > 0.1
    # and the per-layer vapour flux -- the sink update_soil_land charges to the
    # soil -- does not merely shift, it changes sign between the two.
    assert frozen[2][1] * live[2][1] < 0.0
