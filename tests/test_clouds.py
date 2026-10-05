"""Verify the cloud-transmissivity preprocessing against the C++ oracle."""

import glob
import math
import os

import pytest

from geotop_py.io import meteo, parfile
from geotop_py.meteo import clouds
from tools.paths import REFERENCE_1D

try:
    from geotop_py import _cxx
    HAVE_ORACLE = True
except FileNotFoundError:
    HAVE_ORACLE = False

needs_oracle = pytest.mark.skipif(not HAVE_ORACLE, reason="oracle not built")

needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D), reason=f"reference cases not found under {REFERENCE_1D}")

# A synthetic four-point horizon: enough to exercise shadows_point's
# interpolation without needing a real horizon file.
HOR = [(45.0, 0.0), (135.0, 0.0), (225.0, 0.0), (315.0, 0.0)]


def _load_station(case="PureDrainage", rows=400):
    inpts = os.path.join(REFERENCE_1D, case, "geotop.inpts")
    pf = parfile.parse(inpts)
    stem = pf.string("MeteoFile")
    paths = sorted(glob.glob(os.path.join(REFERENCE_1D, case, stem + "[0-9]*.txt")))
    opt = meteo.MeteoOptions.from_parfile(pf, 1)
    return pf, meteo.load(paths[0], meteo.column_names(pf.strings), opt)[:rows]


@pytest.fixture(scope="module")
def station():
    if not os.path.isdir(REFERENCE_1D):
        pytest.skip(f"reference cases not found under {REFERENCE_1D}")
    pf, data = _load_station()
    lat = pf.number("Latitude", 0, 45.0)
    lon = pf.number("Longitude", 0, 0.0)
    ST = pf.number("StandardTimeSimulation", 0, 0.0)
    Z = pf.number("PointElevation", 0, 0.0)
    return dict(pf=pf, data=data, lat=lat, lon=lon, ST=ST, Z=Z)


@needs_oracle
@needs_reference
def test_find_sunset_matches_oracle(station):
    lat, lon, ST = station["lat"], station["lon"], station["ST"]
    data = station["data"]
    lat_r, lon_r = lat * math.pi / 180.0, lon * math.pi / 180.0
    n = 0
    while n < len(data) - 1:
        n0, n1 = clouds.find_sunset(n, data, HOR, lat, lon, ST, 0.0)
        n0_cxx, n1_cxx = _cxx.find_sunset(n, data, HOR, lat_r, lon_r, ST, 0.0)
        assert (n0, n1) == (n0_cxx, n1_cxx)
        n = n1 + 1


@needs_oracle
@needs_reference
def test_find_cloudiness_matches_oracle(station):
    lat, lon, ST, Z = station["lat"], station["lon"], station["ST"], station["Z"]
    data = station["data"]
    lat_r, lon_r = lat * math.pi / 180.0, lon * math.pi / 180.0
    n0, n1 = clouds.find_sunset(0, data, HOR, lat, lon, ST, 0.0)
    rows = sorted({n0, n0 + 1, n0 + 5, n0 + 20, n0 + 40, max(n0, n1 - 1), n1})
    for n in rows:
        if n >= len(data):
            continue
        # ``rotation`` is a dead parameter of the real find_cloudiness (never
        # read in its body -- confirmed against clouds.cc; unlike find_sunset,
        # cloud inference from a station's own shortwave has no orientation
        # term), so clouds.find_cloudiness has no such argument at all.
        mine = clouds.find_cloudiness(n, data, lat, lon, ST, Z, 1.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0)
        cxx = _cxx.find_cloudiness(n, data, lat_r, lon_r, ST, Z, 1.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0)
        # Not `==`: cloud_transmittance chains several trig sums through
        # atm_transmittance and adaptive Simpson quadrature, each a candidate
        # for the same last-bit non-associativity documented in test_rad.py's
        # SolarHeight/SolarAzimuth pins -- confirmed here by seeing this same
        # (n, row) pair compare bit-exact in isolation but drift by 2 ULP when
        # run alongside the rest of the suite. Not a logic bug: the values
        # agree to 1e-13, only the machine's last bit or two moves.
        assert mine == pytest.approx(cxx, rel=0.0, abs=1.0e-12)


@needs_oracle
@needs_reference
def test_fill_meteo_data_with_cloudiness_matches_oracle(station):
    """End to end: mutate independent copies of the same rows through the
    Python and C++ paths and compare the whole ``itauC`` column, not just a
    handful of sample points.

    ``None`` (my representation of ``NUMBER_NOVALUE``/undefined, see
    ``clouds.py``'s module docstring) vs. the oracle wrapper's own ``None``
    for the same sentinel (``_cxx.py`` converts back on the way out) compare
    equal directly; only defined values need a numeric comparison.
    """
    lat, lon, ST, Z = station["lat"], station["lon"], station["ST"], station["Z"]
    data = station["data"]
    mine = [list(row) for row in data]
    cxx_in = [list(row) for row in data]

    added_mine = clouds.fill_meteo_data_with_cloudiness(
        mine, HOR, lat, lon, ST, Z, 1.0, 0.0, 10, 0.0, 0.0, 0.0, 0.0, 0.0)
    added_cxx, tauC_cxx = _cxx.fill_meteo_data_with_cloudiness(
        cxx_in, HOR, lat, lon, ST, Z, 1.0, 0.0, 10, 0.0, 0.0, 0.0, 0.0, 0.0)

    assert added_mine == added_cxx

    itauC = clouds.IDX["itauC"]
    tauC_mine = [row[itauC] for row in mine]
    for i, (a, b) in enumerate(zip(tauC_mine, tauC_cxx)):
        a_missing = a is None or clouds._novalue(a)
        b_missing = b is None
        assert a_missing == b_missing, f"row {i}: mine={a!r} cxx={b!r}"
        if not a_missing:
            # See the ULP note in test_find_cloudiness_matches_oracle above.
            assert a == pytest.approx(b, rel=0.0, abs=1.0e-12), f"row {i}: mine={a!r} cxx={b!r}"
