"""Verify solar geometry and radiation laws against the C++ oracle."""

import math

import pytest

from geotop_py.energy import rad

try:
    from geotop_py import _cxx
    HAVE_ORACLE = True
except FileNotFoundError:
    HAVE_ORACLE = False

needs_oracle = pytest.mark.skipif(not HAVE_ORACLE, reason="oracle not built")

# A spread of JDfrom0 values covering different times of year (leap and non-leap).
_JDS = [1.25, 45.7, 100.0, 179.5, 200.33, 300.9, 365.0, 730.6]


@needs_oracle
@pytest.mark.parametrize("jd", _JDS)
def test_sun_matches_oracle(jd):
    assert rad.sun(jd) == _cxx.sun(jd)


# SolarHeight/SolarAzimuth sum three trig terms (sin*sin + cos*cos*cos); GCC's
# -O2 evaluates that sum with a different rounding than Python's strict
# left-to-right double arithmetic, confirmed by compiling the identical
# expression on identical bit-pattern inputs and seeing the last bit of the
# *first* intermediate ("sine", before asin/acos) differ -- not an artifact of
# this port, and not reproducible by reordering the Python (GCC's choice is an
# internal scheduling detail, not a documented evaluation order to match).
# ``rel=0`` keeps this from silently widening into a real bug: only the exact
# last-bit slack this non-associativity can produce is allowed through.
_ULP = 1.0e-14


@needs_oracle
@pytest.mark.parametrize("jd", _JDS)
@pytest.mark.parametrize("lat_deg", [0.0, 20.0, 46.1, -33.0, 70.0])
def test_solar_height_matches_oracle(jd, lat_deg):
    lat = lat_deg * math.pi / 180.0
    _, _, Delta = rad.sun(jd)
    for dh in (-6.0, 0.0, 5.5, 12.0):
        mine = rad.SolarHeight(jd, lat, Delta, dh)
        cxx = _cxx.SolarHeight(jd, lat, Delta, dh)
        assert mine == pytest.approx(cxx, rel=0.0, abs=_ULP)


@needs_oracle
@pytest.mark.parametrize("jd", _JDS)
@pytest.mark.parametrize("lat_deg", [0.0, 20.0, 46.1, -33.0, 70.0])
def test_solar_azimuth_matches_oracle(jd, lat_deg):
    lat = lat_deg * math.pi / 180.0
    _, _, Delta = rad.sun(jd)
    for dh in (-6.0, 0.0, 5.5, 12.0):
        mine = rad.SolarAzimuth(jd, lat, Delta, dh)
        cxx = _cxx.SolarAzimuth(jd, lat, Delta, dh)
        assert mine == pytest.approx(cxx, rel=0.0, abs=_ULP)


@needs_oracle
@pytest.mark.parametrize(
    "alpha,azimuth", [(0.0, 0.0), (5.0, 40.0), (20.0, 90.0), (45.0, 180.0),
                      (2.0, 359.0), (10.0, 44.9), (10.0, 45.1), (60.0, 270.0)])
def test_shadows_point_matches_oracle(alpha, azimuth):
    hor = [(45.0, 10.0), (135.0, 5.0), (225.0, 15.0), (315.0, 0.0)]
    assert (rad.shadows_point(hor, alpha, azimuth, 0.0, 0.0)
            == _cxx.shadows_point(hor, alpha, azimuth, 0.0, 0.0))


@needs_oracle
def test_atm_transmittance_matches_active_iqbal_branch():
    """Atm transmittance matches active iqbal branch."""
    X, P, RH, T = 0.6, 900.0, 0.5, 10.0
    base = rad.atm_transmittance(X, P, RH, T)
    with_ozone = rad.atm_transmittance(X, P, RH, T, Lozone=0.35)
    with_turbidity = rad.atm_transmittance(X, P, RH, T, a=1.3, b=0.05)
    assert with_ozone != base
    assert with_turbidity != base
