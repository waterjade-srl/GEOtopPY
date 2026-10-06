"""Verify soil-profile interpolation and layer selection against GEOtop."""

import pytest

from geotop_py.output.profiles import interpolate_soil

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)

# Three 10/20/30mm layers, node values 1/2/3.
DZ3 = [0.0, 10.0, 20.0, 30.0]
Q3_LMIN1 = [0.0, 1.0, 2.0, 3.0]
Q3_LMIN0 = [0.5, 1.0, 2.0, 3.0]

DEPTHS = [-1.0, 0.0, 2.5, 5.0, 9.999, 10.0, 15.0, 20.0, 25.0, 35.0,
         44.9, 45.0, 45.1, 100.0]


@pytest.mark.parametrize("h", DEPTHS)
def test_lmin1_matches_the_cxx(h):
    py = interpolate_soil(1, h, 3, DZ3, Q3_LMIN1)
    cc = _cxx.interpolate_soil(1, h, 3, DZ3, Q3_LMIN1)
    assert py == pytest.approx(cc, abs=1.0e-9) if cc != -9999.0 else py == cc


@pytest.mark.parametrize("h", DEPTHS)
def test_lmin0_matches_the_cxx(h):
    py = interpolate_soil(0, h, 3, DZ3, Q3_LMIN0)
    cc = _cxx.interpolate_soil(0, h, 3, DZ3, Q3_LMIN0)
    assert py == pytest.approx(cc, abs=1.0e-9) if cc != -9999.0 else py == cc


def test_single_layer_lmin1():
    dz = [0.0, 100.0]
    q = [0.0, 42.0]
    for h in [0.0, 25.0, 50.0, 75.0, 99.0, 100.0, 150.0]:
        py = interpolate_soil(1, h, 1, dz, q)
        cc = _cxx.interpolate_soil(1, h, 1, dz, q)
        assert py == pytest.approx(cc, abs=1.0e-9) if cc != -9999.0 else py == cc


def test_uneven_layers_lmin1():
    """Layers of very different thicknesses, like a real soil column
    (thin near the surface, thick at depth)."""
    dz = [0.0, 5.0, 15.0, 50.0, 200.0]
    q = [0.0, -1.5, -2.0, -3.5, -6.0]
    for h in [0.0, 1.0, 2.5, 10.0, 40.0, 100.0, 200.0, 269.9, 270.0, 400.0]:
        py = interpolate_soil(1, h, 4, dz, q)
        cc = _cxx.interpolate_soil(1, h, 4, dz, q)
        assert py == pytest.approx(cc, abs=1.0e-9) if cc != -9999.0 else py == cc


def test_returns_number_novalue_beyond_the_reachable_range():
    """A depth past the last clamp region has no answer -- neither side
    extrapolates past it."""
    assert interpolate_soil(1, 1000.0, 3, DZ3, Q3_LMIN1) == -9999.0
