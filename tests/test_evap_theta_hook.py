"""Regression checks for evap theta hook."""

import pytest

from geotop_py import constants as C
from geotop_py.energy import surface
from geotop_py.energy.column import EnergyColumn, SoilLayer


def _layer(th0, res=0.05, sat=0.4):
    return SoilLayer(sat=sat, res=res, alpha=0.004, n=1.3, ss=1.0e-6, kt=2.5,
                     ct=2.3e6, th0=th0, thi0=0.0, Tstar=0.0, fc=0.3)


def _column(th0s, D=1.0):
    n = len(th0s)
    return EnergyColumn(
        Dlayer=[0.0] + [D] * n, ice=[0.0] * (n + 1), liq=[0.0] * (n + 1),
        T0=[0.0] + [10.0] * n, nsng=0,
        soil=[None] + [_layer(t) for t in th0s], alpha_snow=1.0,
        snow_conductivity=1, Tboundary=10.0, Zboundary=1.0, Fboundary=0.0,
        surface_index=0)


def _diag(col, soil_evap, n_evap=None, fc=0.0):
    d = surface.SurfaceDiag(
        SWbeam=0.0, SWdiff=0.0, SWnet=0.0, SWup=0.0, LWin=0.0, cosinc=0.0,
        hsun=0.0, rh=1.0, rv=1.0, Lobukhov=0.0, eps=0.96, fc=fc)
    d.evap_soil = col.soil
    d.evap_Dsoil = list(col.Dlayer)
    d.evap_nlayers = n_evap
    d.soil_evap = soil_evap
    surface.evap_state_reset(d, col)
    return d


def test_hook_tracks_the_trial_temperatures():
    col = _column([0.20, 0.25])
    d = _diag(col, soil_evap=[0.0, 0.0, 0.0])
    hook = surface.evap_state_hook(d, col, Dt=3600.0)
    hook([0.0, 41.0, 12.5], [0.0, 0.0, 0.0])
    assert d.evap_Tsoil[1:] == [41.0, 12.5]


def test_hook_adds_the_newtons_own_phase_change_increment():
    """deltaw is kg/m2 of melt; THETA gains deltaw/(rho_w*Dlayer)."""
    col = _column([0.20], D=0.5)
    d = _diag(col, soil_evap=[0.0, 0.0])
    hook = surface.evap_state_hook(d, col, Dt=3600.0)
    dw = 12.0
    hook([0.0, 1.0], [0.0, dw])
    assert d.evap_theta[1] == pytest.approx(0.20 + dw / (C.rho_w * 0.5))


def test_hook_subtracts_the_previous_trials_evaporation():
    col = _column([0.30], D=0.5)
    E = 4.0e-4                                   # kg m-2 s-1 from the last trial
    d = _diag(col, soil_evap=[0.0, E])
    hook = surface.evap_state_hook(d, col, Dt=3600.0)
    hook([0.0, 20.0], [0.0, 0.0])
    assert d.evap_theta[1] == pytest.approx(0.30 - 3600.0 * E / (C.rho_w * 0.5))


def test_hook_ignores_layers_beyond_the_evaporation_depth():
    """``soil_evap_layer_bare->nh`` gates the subtraction, not the column."""
    col = _column([0.30, 0.30], D=0.5)
    d = _diag(col, soil_evap=[0.0, 4.0e-4], n_evap=1)
    hook = surface.evap_state_hook(d, col, Dt=3600.0)
    hook([0.0, 20.0, 15.0], [0.0, 0.0, 0.0])
    assert d.evap_theta[1] < 0.30
    assert d.evap_theta[2] == pytest.approx(0.30)


def test_hook_floors_at_the_layers_own_residual_content():
    col = _column([0.055], D=0.5)
    d = _diag(col, soil_evap=[0.0, 1.0])          # absurdly large demand
    hook = surface.evap_state_hook(d, col, Dt=3600.0)
    hook([0.0, 20.0], [0.0, 0.0])
    assert d.evap_theta[1] == pytest.approx(0.05 + 1.0e-3)


def test_hook_skips_a_layer_already_at_the_top_layers_residual_guard():
    """The entry guard reads layer *1*'s residual content whatever the layer
    being updated is; only the floor applied afterwards uses the layer's own.
    A deep layer drier than the top layer's residual+1e-3 is left untouched."""
    col = _column([0.30, 0.0505], D=0.5)
    d = _diag(col, soil_evap=[0.0, 0.0, 1.0], n_evap=2)
    hook = surface.evap_state_hook(d, col, Dt=3600.0)
    hook([0.0, 20.0, 15.0], [0.0, 0.0, 0.0])
    assert d.evap_theta[2] == pytest.approx(0.0505)


def test_reset_rewinds_between_repeated_solves():
    col = _column([0.30], D=0.5)
    d = _diag(col, soil_evap=[0.0, 4.0e-4])
    hook = surface.evap_state_hook(d, col, Dt=3600.0)
    hook([0.0, 20.0], [0.0, 0.0])
    assert d.evap_theta[1] < 0.30
    surface.evap_state_reset(d, col)
    assert d.evap_theta[1] == pytest.approx(0.30)


def test_no_hook_when_snow_covers_the_soil():
    """find_actual_evaporation_parameters short-circuits with snow present, so
    there is nothing for the refresh to feed."""
    col = _column([0.30])
    col.nsng = 1
    d = _diag(col, soil_evap=[0.0, 0.0])
    assert d.evap_live is False
    assert surface.evap_state_hook(d, col, Dt=3600.0) is None
