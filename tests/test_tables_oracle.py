"""Verify diagnostic depths and table values against the C++ oracle."""

import pytest

from geotop_py import constants as C
from geotop_py.water import tables

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


def _pa(dz_mm, **columns):
    """``pa[row][layer]`` matrix, both axes 1-based, ``io.soil.SoilParameters``
    layout, following ``tests/test_soilwater.py``'s ``_pa`` pattern."""
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


DZ = [1000.0] * 7
RES = [0.05] * 7
PA = _pa(DZ, jres=RES, jsat=[0.4] * 7, ja=[0.004] * 7, jns=[1.3] * 7,
        jss=[1.0e-6] * 7)


# --------------------------------------------------------------- nlayer

@pytest.mark.parametrize("D,d", [
    (499.0, 1), (499.0, -1), (500.0, 1), (500.0, -1), (501.0, 1), (501.0, -1),
    (1000.0, 1), (1000.0, -1), (1499.0, 1), (1499.0, -1), (1500.0, 1),
    (1500.0, -1), (3500.0, 1), (3500.0, -1), (7000.0, 1), (7000.0, -1),
])
def test_nlayer_matches_the_cxx(D, d):
    mine = tables.nlayer(D, [0.0] + DZ, 7, d)
    theirs = _cxx.nlayer(D, DZ, d)
    assert mine == theirs


# --------------------------------------------------------- active-layer depth

def _compare_activelayer(T1, th1, thi1):
    T = [0.0] + T1
    th = [0.0] + th1
    thi = [0.0] + thi1
    mine_up = tables.find_activelayerdepth_up(T, th, thi, RES, [0.0] + DZ)
    theirs_up = _cxx.find_activelayerdepth_up(T, th, thi, _strip(PA))
    assert mine_up == pytest.approx(theirs_up)

    mine_dw = tables.find_activelayerdepth_dw(T, th, thi, RES, [0.0] + DZ)
    theirs_dw = _cxx.find_activelayerdepth_dw(T, th, thi, _strip(PA))
    assert mine_dw == pytest.approx(theirs_dw)


def test_activelayerdepth_bottom_frozen_top_thawed_matches_the_cxx():
    """Frost crosses partway down -- exercises the interpolated crossing
    branch of both _up (scanning from the surface) and _dw (from the
    bottom, where it finds nothing since the surface is thawed)."""
    _compare_activelayer(
        T1=[2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0],
        th1=[0.30, 0.28, 0.20, 0.12, 0.10, 0.08, 0.06],
        thi1=[0.0, 0.0, 0.05, 0.20, 0.25, 0.28, 0.30])


def test_activelayerdepth_top_frozen_bottom_thawed_matches_the_cxx():
    """The mirror case -- exercises _dw's interpolated crossing."""
    _compare_activelayer(
        T1=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
        th1=[0.08, 0.10, 0.12, 0.20, 0.28, 0.30, 0.32],
        thi1=[0.30, 0.28, 0.20, 0.05, 0.0, 0.0, 0.0])


def test_activelayerdepth_fully_thawed_matches_the_cxx():
    """Bottom node already thawed: _up returns the whole depth
    immediately, without scanning; _dw finds nothing (0)."""
    _compare_activelayer(T1=[1.0] * 7, th1=[0.3] * 7, thi1=[0.0] * 7)


def test_activelayerdepth_fully_frozen_matches_the_cxx():
    _compare_activelayer(T1=[-1.0] * 7, th1=[0.10] * 7, thi1=[0.25] * 7)


# ---------------------------------------------------------- water-table depth

def _compare_watertable(Ptot1, Z=6000.0):
    Ptot = [0.0] + Ptot1
    mine_up = tables.find_watertabledepth_up(Z, Ptot, [0.0] + DZ)
    theirs_up = _cxx.find_watertabledepth_up(Z, Ptot, _strip(PA))
    assert mine_up == pytest.approx(theirs_up)

    mine_dw = tables.find_watertabledepth_dw(Z, Ptot, [0.0] + DZ)
    theirs_dw = _cxx.find_watertabledepth_dw(Z, Ptot, _strip(PA))
    assert mine_dw == pytest.approx(theirs_dw)


def test_watertabledepth_crossing_matches_the_cxx():
    """Dry at the surface, saturated deeper -- exercises the interpolated
    crossing branch of both _up and _dw."""
    _compare_watertable([-2000.0, -1500.0, -800.0, -200.0, 100.0, 300.0, 500.0])


def test_watertabledepth_fully_dry_matches_the_cxx():
    """No saturated node anywhere: both report the capped depth Z."""
    _compare_watertable([-5000.0] * 7)


def test_watertabledepth_fully_saturated_matches_the_cxx():
    _compare_watertable([500.0] * 7)
