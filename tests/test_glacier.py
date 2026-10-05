"""Glacier component: WBglacier, initialisation and column flattening.

The glacier reuses the snow constitutive laws verbatim (in GEOtop every branch
keyed on ``l <= ns+ng`` selects the snow law), so there is nothing new to pin to
the C++ oracle here.  What *is* new -- and what these tests cover -- is the
water balance, which is deliberately NOT WBsnow, and the plumbing that gives the
glacier its own block of nodes between the snow and the soil.
"""


import pytest

from geotop_py.energy.column import SoilLayer
from geotop_py.point.state import Column1D, flatten
from geotop_py.snow.state import SnowColumn
from geotop_py.snow.wb import EBSnow, GlacierWBParams, WBglacier


def _soil(n=2):
    soil = [None] + [SoilLayer(sat=0.4, res=0.05, alpha=0.004, n=1.3, ss=1e-6,
                               kt=2.5, ct=2.3e6, th0=0.2, thi0=0.0)
                     for _ in range(n)]
    return soil, [0.0] + [0.5] * n, [0.0] + [1.0] * n


def _column(glac=None, snow=None):
    soil, D, T = _soil()
    return Column1D(snow=snow or SnowColumn(max=5), soil=soil, soil_D=D,
                    soil_T=T, glac=glac)


def _eb_from(col):
    """Energy results that leave the column exactly as it is (no melt)."""
    ecol = flatten(col)
    n = ecol.nsng
    return EBSnow(ice=[0.0] + [ecol.ice[i] for i in range(1, n + 1)],
                  liq=[0.0] + [ecol.liq[i] for i in range(1, n + 1)],
                  deltaw=[0.0] * (n + 1),
                  Temp=[0.0] + [ecol.T0[i] for i in range(1, n + 1)],
                  Dlayer=[0.0] + [ecol.Dlayer[i] for i in range(1, n + 1)])


# ---------------------------------------------------------------------------
# flattening: snow, then glacier, then soil
# ---------------------------------------------------------------------------
def test_flatten_places_glacier_between_snow_and_soil():
    snow = SnowColumn.from_layers([(100.0, 30.0, 0.0, -2.0)], max=5)
    glac = SnowColumn.from_layers([(500.0, 400.0, 0.0, -5.0),
                                   (300.0, 250.0, 0.0, -4.0)], max=5)
    col = _column(glac=glac, snow=snow)
    ecol = flatten(col)

    # 1 snow + 2 glacier + 2 soil, and nsng covers both ice stacks
    assert ecol.n == 5
    assert ecol.nsng == 3
    # node 1 is the snow; nodes 2-3 the glacier top-down (so layer 2 then 1)
    assert ecol.ice[1] == pytest.approx(30.0)
    assert ecol.ice[2] == pytest.approx(250.0)      # glacier layer 2 = top
    assert ecol.ice[3] == pytest.approx(400.0)      # glacier layer 1 = base
    assert ecol.T0[2] == pytest.approx(-4.0)
    assert ecol.T0[3] == pytest.approx(-5.0)
    # the soil parameters must still be found from the *soil* index
    assert ecol._soil_of(4)[0] == 1
    assert ecol._soil_of(5)[0] == 2


def test_glacier_absent_leaves_the_column_untouched():
    """``glac=None`` must reproduce the snow-only flattening exactly."""
    snow = SnowColumn.from_layers([(100.0, 30.0, 0.0, -2.0)], max=5)
    a = flatten(_column(snow=snow))
    b = flatten(_column(snow=snow, glac=SnowColumn(max=5)))   # module on, 0 layers
    assert a.n == b.n == 3
    assert a.nsng == b.nsng == 1
    assert a.ice == b.ice and a.Dlayer == b.Dlayer


# ---------------------------------------------------------------------------
# WBglacier
# ---------------------------------------------------------------------------
def test_wbglacier_conserves_mass():
    """Wbglacier conserves mass."""
    glac = SnowColumn.from_layers([(500.0, 400.0, 5.0, -1.0),
                                   (300.0, 250.0, 3.0, -0.5)], max=5)
    col = _column(glac=glac)
    eb = _eb_from(col)
    eb.deltaw = [0.0, 2.0, 4.0]          # top-down: melt in both layers
    before = glac.swe()
    Evap = 1.5
    melt = WBglacier(3600.0, 0, 2, glac, GlacierWBParams(), Evap, eb)
    assert glac.swe() == pytest.approx(before - melt - Evap, abs=1e-12)


def test_wbglacier_drains_instantly_unlike_snow():
    """Water above irreducible saturation leaves in full within one step.

    WBsnow releases only ``5*Se^3*Dt`` of it; WBglacier has no such limiter
    (snow.cc:845-850).  With a short step the two therefore disagree, and that
    disagreement is the point of having a separate function.
    """
    liq = 60.0                            # far above irreducible saturation
    glac = SnowColumn.from_layers([(500.0, 400.0, liq, 0.0)], max=5)
    col = _column(glac=glac)
    eb = _eb_from(col)
    melt = WBglacier(1.0, 0, 1, glac, GlacierWBParams(Sr=0.02), 0.0, eb)

    thi = glac.w_ice[1] / (1e-3 * glac.Dzl[1] * 917.0)
    retained = 0.02 * (1.0 - thi) * glac.Dzl[1] * 1e-3 * 1000.0
    assert glac.w_liq[1] == pytest.approx(retained)
    assert melt == pytest.approx(liq - retained)
    assert melt > 0.5 * liq               # a 1 s snow step would release ~nothing


def test_wbglacier_does_not_compact():
    """No Anderson settling: with no melt the thickness must not change."""
    glac = SnowColumn.from_layers([(500.0, 400.0, 0.0, -10.0)], max=5)
    col = _column(glac=glac)
    eb = _eb_from(col)
    WBglacier(86400.0, 0, 1, glac, GlacierWBParams(), 0.0, eb)
    assert glac.Dzl[1] == pytest.approx(500.0)


def test_wbglacier_thin_layer_holds_no_water():
    par = GlacierWBParams(max_weq_glac=10.0)     # threshold = 0.1*10 = 1 kg/m2
    glac = SnowColumn.from_layers([(2.0, 0.5, 0.3, 0.0)], max=5)
    col = _column(glac=glac)
    eb = _eb_from(col)
    melt = WBglacier(3600.0, 0, 1, glac, par, 0.0, eb)
    assert glac.w_liq[1] == 0.0
    assert melt == pytest.approx(0.3)


def test_wbglacier_indexes_below_the_snow():
    """Glacier layer l reads energy node ns+ng-l+1, not ns-l+1."""
    snow = SnowColumn.from_layers([(100.0, 30.0, 0.0, -2.0)], max=5)
    glac = SnowColumn.from_layers([(500.0, 400.0, 0.0, -6.0),
                                   (300.0, 250.0, 0.0, -3.0)], max=5)
    col = _column(glac=glac, snow=snow)
    eb = _eb_from(col)
    WBglacier(3600.0, 1, 2, glac, GlacierWBParams(), 0.0, eb)
    # temperatures come back onto the right layers, top layer = 2
    assert glac.T[2] == pytest.approx(-3.0)
    assert glac.T[1] == pytest.approx(-6.0)
