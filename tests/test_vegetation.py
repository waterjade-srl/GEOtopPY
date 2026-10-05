"""Canopy component: burying, roughness, interception, radiative partition.

The canopy has no C++ oracle behind it (``geotopLaws`` only exports the pure
constitutive laws), so what is pinned here are the invariants that a wrong port
would break: mass closure of the interception, energy closure of the two-stream
shortwave, the analytic longwave derivative, and the two GEOtop-specific quirks
that a "cleaned-up" rewrite would silently drop -- the fully general
``fsnow`` ramp, and ``Fmin(1e20, NaN) == 1e20``.
"""

import math

import pytest

from geotop_py.energy import vegetation as veg
from geotop_py.energy.column import SoilLayer


def _pasture(**kw):
    p = veg.VegParams(Hveg=200.0, z0thresveg=200.0, z0thresveg2=200.0, LSAI=2.0,
                      cf=1.0, decay0=2.5, expveg=1.0, root=300.0, rs=60.0,
                      R_vis=0.15, R_nir=0.4, T_vis=0.07, T_nir=0.32, Ch=0.3,
                      cd=1.0)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


# ---------------------------------------------------------------------------
# fc and roughness
# ---------------------------------------------------------------------------
def test_fsnow_is_a_ramp_not_a_step():
    """With distinct thresholds the burying must interpolate linearly."""
    vp = _pasture(z0thresveg=300.0, z0thresveg2=100.0)
    assert veg.snow_burying_fraction(50.0, vp) == 0.0
    assert veg.snow_burying_fraction(200.0, vp) == pytest.approx(0.5)
    assert veg.snow_burying_fraction(400.0, vp) == 1.0
    assert veg.canopy_fraction(200.0, vp) == pytest.approx(0.5)


def test_fsnow_collapses_to_a_step_when_thresholds_coincide():
    vp = _pasture()
    assert veg.canopy_fraction(199.9, vp) == 1.0
    assert veg.canopy_fraction(200.0, vp) == 1.0
    assert veg.canopy_fraction(200.1, vp) == 0.0


def test_glacier_and_low_lsai_suppress_the_canopy():
    vp = _pasture()
    assert veg.canopy_fraction(0.0, vp, ng=1) == 0.0
    assert veg.canopy_fraction(0.0, _pasture(LSAI=0.05)) == 0.0


def test_update_roughness_veg_and_its_three_aborts():
    z0, d0, h = veg.update_roughness_veg(1900.0, 400.0, 5.0, 2.0)
    assert h == pytest.approx(1.5)
    assert d0 == pytest.approx(0.667 * 1.5)
    assert z0 == pytest.approx(0.15)
    with pytest.raises(ValueError):
        veg.update_roughness_veg(6000.0, 0.0, 5.0, 2.0)   # above the anemometer
    with pytest.raises(ValueError):
        veg.update_roughness_veg(3000.0, 0.0, 5.0, 2.0)   # above the thermometer


# ---------------------------------------------------------------------------
# interception
# ---------------------------------------------------------------------------
def test_rain_interception_conserves_mass():
    P = 4.0
    store0 = 0.05
    mx, store, drip = veg.canopy_rain_interception(veg.RAIN_MAX_LOADING, 2.0,
                                                   P, store0)
    assert mx == pytest.approx(0.2)
    assert store <= mx
    assert store - store0 + drip == pytest.approx(P)


def test_snow_interception_conserves_mass_including_unloading():
    P = 3.0
    store0 = 1.0
    mx, store, drip = veg.canopy_snow_interception(veg.SNOW_MAX_LOADING, 4.0, P,
                                                   Tc=-2.0, v=3.0, Dt=3600.0,
                                                   storage=store0)
    assert mx == pytest.approx(20.0)
    assert store - store0 + drip == pytest.approx(P)


def test_snow_unloading_below_a_tenth_of_a_mm_is_suppressed():
    """The ``unload < 0.1`` guard: a nearly-empty canopy sheds nothing."""
    _mx, store, drip = veg.canopy_snow_interception(
        veg.SNOW_MAX_LOADING, 4.0, 0.0, Tc=-10.0, v=0.1, Dt=3600.0,
        storage=0.05)
    assert drip == 0.0
    assert store == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# radiation
# ---------------------------------------------------------------------------
def test_shortwave_vegetation_closes_the_energy_budget():
    """Absorbed by canopy + absorbed by ground + reflected upward = incoming."""
    Sd, Sb = 120.0, 300.0
    Sv, Sg, Sup = veg.shortwave_vegetation(
        Sd, Sb, 0.6, 0.0, veg.wsn_vis, veg.Bsnd_vis, veg.Bsnb_vis,
        0.15, 0.15, 0.3, 0.15, 0.07, 2.0)
    assert Sv + Sg + Sup == pytest.approx(Sd + Sb, rel=1e-9)


def test_shortwave_vegetation_snow_on_leaves_raises_the_reflection():
    args = (120.0, 300.0, 0.6, None, veg.wsn_vis, veg.Bsnd_vis, veg.Bsnb_vis,
            0.15, 0.15, 0.3, 0.15, 0.07, 2.0)
    bare = veg.shortwave_vegetation(*args[:3], 0.0, *args[4:])
    laden = veg.shortwave_vegetation(*args[:3], 1.0, *args[4:])
    assert laden[2] > bare[2]
    assert laden[0] < bare[0]
    assert sum(laden) == pytest.approx(420.0, rel=1e-9)


def test_longwave_vegetation_derivative_is_analytic():
    args = (300.0, 0.99, -2.0, 4.0)
    Lv, _Lg, dLv, _Lup = veg.longwave_vegetation(*args, 2.0)
    h = 1e-6
    up, _, _, _ = veg.longwave_vegetation(300.0, 0.99, -2.0, 4.0 + h, 2.0)
    dn, _, _, _ = veg.longwave_vegetation(300.0, 0.99, -2.0, 4.0 - h, 2.0)
    assert dLv == pytest.approx((up - dn) / (2 * h), rel=1e-6)


def test_longwave_vegetation_is_transparent_for_a_bare_canopy():
    """LSAI -> 0 gives ev -> 0: the canopy neither absorbs nor emits."""
    Lv, Lg, _dLv, Lup = veg.longwave_vegetation(300.0, 0.99, 5.0, -3.0, 0.0)
    assert Lv == pytest.approx(0.0)
    assert Lg == pytest.approx(0.99 * (300.0 - veg.rad.SB(5.0)))
    assert Lup == pytest.approx(0.01 * 300.0 + 0.99 * veg.rad.SB(5.0))


# ---------------------------------------------------------------------------
# roots and stomata
# ---------------------------------------------------------------------------
def test_root_fraction_sums_to_one_and_stops_at_the_root_depth():
    Dz = [0.0, 280.0, 500.0, 2000.0]
    frac = veg.root_fraction(4, 2000.0, 0.0, Dz)
    assert sum(frac) == pytest.approx(1.0)
    shallow = veg.root_fraction(4, 300.0, 0.0, Dz)
    assert shallow[1] == pytest.approx(280.0 / 300.0)
    assert shallow[3] == 0.0


def test_transpiration_shuts_down_on_frozen_or_scorched_leaves():
    soil = [None] + [SoilLayer(sat=0.4, res=0.0, alpha=0.004, n=1.1, ss=1e-6,
                               kt=2.5, ct=2.3e6, th0=0.3, fc=0.03, wp=0.005)
                     for _ in range(3)]
    vp = _pasture(root=2000.0)
    vp.n_transp = 3
    vp.root_frac = veg.root_fraction(4, 2000.0, 0.0, [0.0, 280.0, 500.0, 2000.0])
    theta = [0.0, 0.3, 0.3, 0.3]
    warm, _ = veg.canopy_evapotranspiration(30.0, 15.0, 0.005, 850.0, 400.0,
                                            theta, vp, soil)
    cold, _ = veg.canopy_evapotranspiration(30.0, -1.0, 0.005, 850.0, 400.0,
                                            theta, vp, soil)
    assert warm == pytest.approx(30.0 / (30.0 + 60.0 / (400.0 / 650.0 * 1.25
                                 * (1.0 - (veg.rad.sat_vap_pressure(15.0, 850.0)
                                           - 0.005 * 850.0 / (0.378 * 0.005 + 0.622)) / 40.0)
                                 * 15.0 * 35.0 / 625.0)), rel=1e-9)
    assert cold < 1e-6


# ---------------------------------------------------------------------------
# in-canopy turbulence
# ---------------------------------------------------------------------------
def test_buried_canopy_decouples_the_ground_instead_of_producing_nan():
    """``zm < z0soil`` makes GEOtop's ``r`` negative and ``pow`` return NaN;
    ``Fmin`` keeps its first argument, so ``ruc`` must come out as 1e20."""
    rb, ruc, decay = veg.veg_transmittance(
        1, 1.0, 0.02, 0.1, Hveg=2.5e-5, z0soil=1e-4, z0veg=2.5e-6,
        d0veg=1.67e-5, LSAI=2.0, decaycoeff0=2.5, Lo=1.5, Loc=1e50)
    assert ruc == 1e20
    assert math.isfinite(rb)
    assert decay == pytest.approx(2.5, rel=1e-6)


def test_denser_canopy_gives_a_smaller_leaf_boundary_resistance():
    common = dict(v=2.0, u_star=0.2, u_top=1.0, Hveg=1.5, z0soil=1e-3,
                  z0veg=0.15, d0veg=1.0, decaycoeff0=2.5, Lo=-50.0, Loc=-30.0)
    rb2, _, _ = veg.veg_transmittance(1, LSAI=2.0, **common)
    rb4, _, _ = veg.veg_transmittance(1, LSAI=4.0, **common)
    assert rb4 == pytest.approx(rb2 / 2.0, rel=1e-12)
