"""``meteodistr`` checked against the oracle, piece by piece.

``get_temperature``/``get_relative_humidity``/``get_precipitation``/``get_wind``
are orchestration built from primitives pinned elsewhere: the lapse-rate
conversion and ``part_snow`` in ``tests/test_psychro.py``, the station-count
dispatch (``interpolate_meteo``) and the wind topography correction
(``topo_mod_winds``) here. Composing already-pinned pieces is checked with
invariants instead of a second pin, since reproducing GEOtop's ``METEO``
struct just to call the wrapping C functions directly would add marshalling
without adding verification -- every number that flows through them is
already independently pinned at its source.
"""

import math

import pytest

from geotop_py.meteo import meteodistr

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


# -------------------------------------------------------- interpolate_meteo

def test_interpolate_meteo_zero_stations_matches_the_cxx():
    ok, grid = _cxx.interpolate_meteo(1, 0, 0, [], [], [], 0, 100.0, 0)
    mine = meteodistr.interpolate_meteo([])
    assert ok == 0
    assert mine is None


def test_interpolate_meteo_one_station_matches_the_cxx():
    ok, grid = _cxx.interpolate_meteo(1, 0, 0, [0.0], [0.0], [[12.3]], 0, 100.0, 0)
    mine = meteodistr.interpolate_meteo([12.3])
    assert ok == 1
    assert mine == grid == 12.3


def test_interpolate_meteo_one_station_absent_matches_the_cxx():
    ok, _ = _cxx.interpolate_meteo(1, 0, 0, [0.0], [0.0],
                                   [[_cxx.NUMBER_ABSENT]], 0, 100.0, 0)
    assert ok == 0
    assert meteodistr.interpolate_meteo([None]) is None


def test_interpolate_meteo_ignores_none_entries_among_several_stations():
    """Two configured stations, only one with a reading -- still the trivial
    (nstn==1) branch, not Barnes."""
    assert meteodistr.interpolate_meteo([None, 7.5]) == 7.5


def test_interpolate_meteo_raises_for_more_than_one_valid_reading():
    """Barnes objective interpolation is out of scope (unreachable with
    PointSim's single station) and not implemented."""
    with pytest.raises(NotImplementedError):
        meteodistr.interpolate_meteo([1.0, 2.0])


# ------------------------------------------------------------- topo_mod_winds

WIND_CASES = [
    (winddir, windspd, slope, az)
    for winddir in (0.0, 22.5, 22.5 + 1e-6, 44.9, 45.0, 45.1, 67.5, 90.0,
                    112.5, 135.0, 157.5, 180.0, 202.5, 225.0, 270.0,
                    315.0, 337.5, 359.9)
    for windspd in (0.0, 2.5, 10.0)
    for slope in (0.0, 5.0, 15.0, 30.0)
    for az in (0.0, 90.0, 180.0, 270.0)
]


@pytest.mark.parametrize("winddir,windspd,slope,az", WIND_CASES)
def test_topo_mod_winds_matches_the_cxx(winddir, windspd, slope, az):
    curvatures = (0.2, -0.1, 0.05, -0.3)
    slopewtD, curvewtD, slopewtI, curvewtI = 0.58, -0.42, 0.58, -0.42

    mine = meteodistr.topo_mod_winds(
        winddir, windspd, slopewtD, curvewtD, slopewtI, curvewtI,
        *curvatures, az, slope)
    theirs = _cxx.topo_mod_winds(
        winddir, windspd, slopewtD, curvewtD, slopewtI, curvewtI,
        *curvatures, az, slope, 100.0, -9999.0)
    assert mine == theirs


def test_topo_mod_winds_is_identity_on_a_flat_point():
    """Zero slope and curvature: no correction, whatever the wind direction."""
    for winddir in (0.0, 45.0, 133.0, 271.0):
        winddir_out, windspd_out = meteodistr.topo_mod_winds(
            winddir, 4.0, 0.58, -0.42, 0.58, -0.42, 0.0, 0.0, 0.0, 0.0, 90.0, 0.0)
        assert winddir_out == pytest.approx(winddir)
        assert windspd_out == pytest.approx(4.0)


# --------------------------------------------------------- wind_speed_dir_from_uv

@pytest.mark.parametrize("Wx,Wy,expected_dir,expected_speed", [
    (5.0, 0.0, 270.0, 5.0),     # pure eastward component
    (0.0, 5.0, 180.0, 5.0),     # pure northward component
    (-5.0, 0.0, 90.0, 5.0),     # pure westward component
    (0.0, -5.0, 0.0, 5.0),      # pure southward component
    (1.0, 1.0, 225.0, math.sqrt(2.0)),
])
def test_wind_speed_dir_from_uv_matches_hand_computed_angles(Wx, Wy, expected_dir, expected_speed):
    winddir, windspd = meteodistr.wind_speed_dir_from_uv(Wx, Wy)
    assert winddir == pytest.approx(expected_dir, abs=1e-9)
    assert windspd == pytest.approx(expected_speed, abs=1e-9)


def test_wind_speed_dir_from_uv_speed_matches_pythagoras():
    for Wx, Wy in [(3.0, 4.0), (-2.0, 7.0), (0.1, 0.1)]:
        _, windspd = meteodistr.wind_speed_dir_from_uv(Wx, Wy)
        assert windspd == pytest.approx(math.hypot(Wx, Wy))


# ------------------------------------------------------------- find_cloudfactor

@pytest.mark.parametrize("Tair,RH,Z", [
    (T, rh, Z)
    for T in (-10.0, 0.0, 10.0, 25.0)
    for rh in (10.0, 40.0, 70.0, 100.0)
    for Z in (0.0, 800.0, 2500.0)
])
def test_find_cloudfactor_matches_the_cxx(Tair, RH, Z):
    mine = meteodistr.find_cloudfactor(Tair, RH, Z, -6.5, -2.5)
    theirs = _cxx.find_cloudfactor(Tair, RH, Z, -6.5, -2.5)
    assert mine == pytest.approx(theirs, rel=0, abs=1e-12)


def test_find_cloudfactor_is_bounded():
    for _ in range(20):
        fc = meteodistr.find_cloudfactor(15.0, 60.0, 1200.0, -6.5, -2.5)
        assert 0.0 <= fc <= 1.0


# --------------------------------------------------------------- get_temperature

def test_get_temperature_returns_none_with_no_reading():
    assert meteodistr.get_temperature(None, 500.0, 1500.0, -6.5) is None


def test_get_temperature_matches_direct_lapse_correction():
    """With a single station, the sea-level round trip must be exactly
    equivalent to one direct lapse correction, since GEOtop's own topo_ref
    cancels algebraically -- this is the identity that justifies not needing
    a fake METEO struct to pin this function: the two-step Kelvin round trip
    and the one-step formula are different code paths that must still agree."""
    from geotop_py import psychro
    T_point = meteodistr.get_temperature(10.0, 500.0, 1500.0, -6.5)
    assert T_point == pytest.approx(psychro.temperature(1500.0, 500.0, 10.0, -6.5), abs=1e-9)


def test_get_temperature_is_identity_at_zero_lapse_rate():
    assert meteodistr.get_temperature(8.5, 200.0, 3000.0, 0.0) == pytest.approx(8.5)


# ---------------------------------------------------------- get_relative_humidity

def test_get_relative_humidity_returns_none_with_no_reading():
    assert meteodistr.get_relative_humidity(None, 5.0, 500.0, 500.0, -2.5, 10.0) is None


def test_get_relative_humidity_is_floored_at_rh_min():
    # a very dry, very warm point should hit the RH_min floor
    rh = meteodistr.get_relative_humidity(-40.0, 30.0, 0.0, 0.0, -2.5, 15.0)
    assert rh == pytest.approx(0.15)


def test_get_relative_humidity_matches_rhfrom_tdew_at_the_station_elevation():
    from geotop_py import psychro
    rh = meteodistr.get_relative_humidity(2.0, 8.0, 500.0, 500.0, -2.5, 10.0)
    assert rh == pytest.approx(max(psychro.RHfromTdew(8.0, 2.0, 500.0), 0.1))


# ------------------------------------------------------------- get_precipitation

def test_get_precipitation_returns_none_with_no_reading():
    assert meteodistr.get_precipitation(
        None, 500.0, 1500.0, 2.0, 0.0, 2.0, 0.0, 3.0, -1.0, 1.0, 1.0) is None


def test_get_precipitation_conserves_total_at_zero_lapse_and_station_elevation():
    P = meteodistr.get_precipitation(
        5.0, 500.0, 500.0, -5.0, 0.0, 2.0, 0.0, 3.0, -1.0, 1.0, 1.0)
    assert P == pytest.approx(5.0)  # all-snow, corr factors 1: exact pass-through


def test_get_precipitation_elevation_factor_matches_the_formula_sign():
    """f = (1-alfa)/(1+alfa) with alfa = 1e-3*lapse_rate*(Z_point-Z_station):
    a *negative* lapse_rate is what increases precipitation with elevation in
    this formula (alfa < 0 => f > 1), not a positive one -- worth pinning
    explicitly since "lapse rate" sign conventions are formula-specific and
    easy to get backwards."""
    lower = meteodistr.get_precipitation(
        5.0, 500.0, 500.0, -5.0, -0.3, 5.0, 0.0, 3.0, -1.0, 1.0, 1.0)
    higher = meteodistr.get_precipitation(
        5.0, 500.0, 1500.0, -5.0, -0.3, 5.0, 0.0, 3.0, -1.0, 1.0, 1.0)
    assert lower == pytest.approx(5.0)
    assert higher > lower
    assert higher == pytest.approx(5.0 * 1.3 / 0.7)


# ------------------------------------------------------------------- get_wind

def test_get_wind_returns_none_with_no_reading_at_all():
    assert meteodistr.get_wind(
        None, None, None, prev_winddir=180.0,
        slopewtD=0.58, curvewtD=-0.42, slopewtI=0.58, curvewtI=-0.42,
        curvature1=0.0, curvature2=0.0, curvature3=0.0, curvature4=0.0,
        slope_az=0.0, terrain_slope=0.0, windspd_min=0.2) is None


def test_get_wind_falls_back_to_speed_only_and_keeps_previous_direction():
    winddir, windspd = meteodistr.get_wind(
        None, None, 3.0, prev_winddir=222.0,
        slopewtD=0.58, curvewtD=-0.42, slopewtI=0.58, curvewtI=-0.42,
        curvature1=0.0, curvature2=0.0, curvature3=0.0, curvature4=0.0,
        slope_az=0.0, terrain_slope=0.0, windspd_min=0.2)
    assert winddir == 222.0
    assert windspd == 3.0


def test_get_wind_uses_uv_and_applies_the_topo_correction_when_both_present():
    winddir, windspd = meteodistr.get_wind(
        5.0, 0.0, None, prev_winddir=0.0,
        slopewtD=0.58, curvewtD=-0.42, slopewtI=0.58, curvewtI=-0.42,
        curvature1=0.0, curvature2=0.0, curvature3=0.0, curvature4=0.0,
        slope_az=0.0, terrain_slope=0.0, windspd_min=0.2)
    # flat point: topo_mod_winds is an identity, so this must equal the raw
    # u/v conversion exactly.
    expected_dir, expected_speed = meteodistr.wind_speed_dir_from_uv(5.0, 0.0)
    assert (winddir, windspd) == (expected_dir, expected_speed)


def test_get_wind_floors_speed_at_windspd_min():
    _, windspd = meteodistr.get_wind(
        None, None, 0.01, prev_winddir=0.0,
        slopewtD=0.58, curvewtD=-0.42, slopewtI=0.58, curvewtI=-0.42,
        curvature1=0.0, curvature2=0.0, curvature3=0.0, curvature4=0.0,
        slope_az=0.0, terrain_slope=0.0, windspd_min=0.5)
    assert windspd == 0.5
