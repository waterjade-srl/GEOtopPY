"""Regression checks for snow.mass_balance."""

import pytest

from geotop_py import constants as C
from geotop_py import laws
from geotop_py.snow.mass_balance import EBSnow, SnowWBParams, WBsnow, new_snow, snow_compactation
from geotop_py.snow.state import SnowColumn

try:
    from geotop_py import _cxx
    HAVE_ORACLE = True
except FileNotFoundError:
    HAVE_ORACLE = False

A = 1.0
PAR = SnowWBParams()


def eb_snow(ns, ice, liq, deltaw, Temp, D_m=0.2):
    """Build top-down energy results (index 1 = surface) for ns snow layers."""
    return EBSnow(
        ice=[0.0] + list(ice),
        liq=[0.0] + list(liq),
        deltaw=[0.0] + list(deltaw),
        Temp=[0.0] + list(Temp),
        Dlayer=[0.0] + [D_m] * ns,
    )


def empty_snow(max=10, lnum=3, type=2):
    col = SnowColumn(max=max, lnum=lnum, type=type)
    return col


def pack_swe(col):
    return sum(col.w_ice[l] + col.w_liq[l] for l in range(1, col.max + 1))


def eb_swe(E, ns):
    return sum(E.ice[m] + E.liq[m] for m in range(1, ns + 1))


# --- water budget --------------------------------------------------------

def test_wbsnow_conserves_water_pure_melt():
    ns = 3
    E = eb_snow(ns,
                ice=[60.0, 55.0, 50.0],
                liq=[1.0, 1.0, 1.0],
                deltaw=[8.0, 4.0, 2.0],   # melt: ice -> liquid
                Temp=[0.0, 0.0, -0.5])
    swe0 = eb_swe(E, ns)
    col = empty_snow(lnum=ns)

    Melt, RoS = WBsnow(Dt=3600.0, ns=ns, snow=col, par=PAR,
                       slope=0.0, Rain=0.0, Evap=0.0, E=E)

    assert Melt == pytest.approx(swe0 - pack_swe(col), abs=1e-6)
    assert Melt >= 0.0
    assert RoS == 0.0


def test_wbsnow_conserves_water_with_rain():
    ns = 3
    E = eb_snow(ns, ice=[60.0, 55.0, 50.0], liq=[1.0, 1.0, 1.0],
                deltaw=[3.0, 2.0, 1.0], Temp=[0.0, -0.3, -0.6])
    swe0 = eb_swe(E, ns)
    col = empty_snow(lnum=ns)
    Rain = 5.0

    Melt, RoS = WBsnow(Dt=3600.0, ns=ns, snow=col, par=PAR,
                       slope=0.0, Rain=Rain, Evap=0.0, E=E)

    # budget: Melt = SWE_before - SWE_after - Evap  (Evap = 0)
    assert Melt == pytest.approx(swe0 - pack_swe(col), abs=1e-6)


def test_wbsnow_conserves_water_with_sublimation():
    ns = 3
    E = eb_snow(ns, ice=[60.0, 55.0, 50.0], liq=[1.0, 1.0, 1.0],
                deltaw=[2.0, 1.0, 0.5], Temp=[-1.0, -1.5, -2.0])
    swe0 = eb_swe(E, ns)
    col = empty_snow(lnum=ns)
    Evap = 4.0                     # sublimation from the surface layer

    Melt, RoS = WBsnow(Dt=3600.0, ns=ns, snow=col, par=PAR,
                       slope=0.0, Rain=0.0, Evap=Evap, E=E)

    assert Melt == pytest.approx(swe0 - pack_swe(col) - Evap, abs=1e-6)


def test_wbsnow_sublimation_can_remove_a_layer():
    ns = 2
    # tiny surface layer, large sublimation demand -> surface layer disappears
    E = eb_snow(ns, ice=[1.0, 40.0], liq=[0.0, 0.5], deltaw=[0.0, 0.0],
                Temp=[-3.0, -4.0])
    swe0 = eb_swe(E, ns)
    col = empty_snow(lnum=ns)

    Melt, RoS = WBsnow(Dt=3600.0, ns=ns, snow=col, par=PAR,
                       slope=0.0, Rain=0.0, Evap=1.5, E=E)

    # top layer (l = ns) fully sublimated
    assert col.w_ice[ns] == 0.0
    assert Melt == pytest.approx(swe0 - pack_swe(col) - 1.5, abs=1e-6)


# --- compaction ----------------------------------------------------------

def compacting_column():
    col = SnowColumn.from_layers(
        [(300.0, 60.0, 0.0, -5.0),
         (250.0, 45.0, 0.0, -3.0),
         (200.0, 30.0, 0.0, -1.0)],
        max=10,
    )
    return col


def test_compaction_only_densifies():
    col = compacting_column()
    for l in (1, 2, 3):
        D0 = col.Dzl[l]
        wi0, wl0 = col.w_ice[l], col.w_liq[l]
        snow_compactation(col, l, Dt=3600.0, slope=0.0, par=PAR)
        assert col.Dzl[l] < D0                 # depth shrinks
        assert col.w_ice[l] == wi0             # mass untouched
        assert col.w_liq[l] == wl0
        # density rose
        assert col.w_ice[l] / (1e-3 * col.Dzl[l] * C.rho_i) > \
            wi0 / (1e-3 * D0 * C.rho_i)


def test_compaction_matches_hand_computation():
    col = SnowColumn.from_layers([(400.0, 40.0, 0.0, -5.0)], max=4)
    D0 = col.Dzl[1]
    theta_i = 40.0 / (0.001 * 400.0 * C.rho_i)
    c1 = PAR.drysnowdef_rate  # theta_i*rho_i below cutoff for this case
    if theta_i * C.rho_i > PAR.snow_density_cutoff:
        c1 = PAR.drysnowdef_rate * \
            (2.718281828459045 ** (-0.046 * (C.rho_i * theta_i - PAR.snow_density_cutoff)))
    CR1 = -c1 * 2.777e-6 * (2.718281828459045 ** (-0.04 * (0.0 - (-5.0))))
    load = 40.0
    eta = PAR.snow_viscosity * (2.718281828459045 **
                                (0.08 * (0.0 - (-5.0)) + 0.023 * (C.rho_i * theta_i)))
    CR2 = -load / eta
    expected = D0 * (2.718281828459045 ** ((CR1 + CR2) * 3600.0))

    snow_compactation(col, 1, Dt=3600.0, slope=0.0, par=PAR)
    assert col.Dzl[1] == pytest.approx(expected, rel=1e-12)


# --- fresh snow ----------------------------------------------------------

def test_new_snow_on_empty_pack_seeds_layer_1():
    col = SnowColumn(max=10, lnum=0, type=0)
    new_snow(A, col, P=5.0, Dz=50.0, T=-3.0)
    assert col.w_ice[1] == 5.0
    assert col.Dzl[1] == 50.0


def test_new_snow_adds_mass_and_conserves_energy():
    col = SnowColumn.from_layers([(200.0, 40.0, 2.0, -4.0)], max=10)
    ns = col.lnum
    swe0 = col.w_ice[ns] + col.w_liq[ns]
    D0 = col.Dzl[ns]
    P, Dz, T = 6.0, 60.0, -8.0
    h_before = (laws.internal_energy(col.w_ice[ns], col.w_liq[ns], col.T[ns])
                + C.c_ice * P * (min(T, -0.1) - C.Tfreezing))

    new_snow(A, col, P=P, Dz=Dz, T=T)

    assert col.w_ice[ns] + col.w_liq[ns] == pytest.approx(swe0 + P, rel=1e-9)
    assert col.Dzl[ns] == pytest.approx(D0 + Dz)
    h_after = laws.internal_energy(col.w_ice[ns], col.w_liq[ns], col.T[ns])
    assert h_after == pytest.approx(h_before, rel=1e-7)


@pytest.mark.skipif(not HAVE_ORACLE, reason="oracle not built")
def test_new_snow_matches_cxx_oracle():
    col_py = SnowColumn.from_layers([(200.0, 40.0, 2.0, -4.0)], max=10)
    col_cxx = SnowColumn.from_layers([(200.0, 40.0, 2.0, -4.0)], max=10)

    new_snow(A, col_py, P=6.0, Dz=60.0, T=-8.0)

    # replicate new_snow using the C++ enthalpy laws directly
    ns = col_cxx.lnum
    h = _cxx.internal_energy(col_cxx.w_ice[ns], col_cxx.w_liq[ns], col_cxx.T[ns])
    h += C.c_ice * 6.0 * (min(-8.0, -0.1) - C.Tfreezing)
    col_cxx.Dzl[ns] += 60.0
    col_cxx.w_ice[ns] += 6.0
    wi, wl, T = _cxx.from_internal_energy(A, h, col_cxx.w_ice[ns], col_cxx.w_liq[ns])
    col_cxx.w_ice[ns], col_cxx.w_liq[ns], col_cxx.T[ns] = wi, wl, T

    assert col_py.w_ice[ns] == pytest.approx(col_cxx.w_ice[ns], rel=1e-9)
    assert col_py.w_liq[ns] == pytest.approx(col_cxx.w_liq[ns], rel=1e-9)
    assert col_py.T[ns] == pytest.approx(col_cxx.T[ns], rel=1e-9)
