"""GEOtop-compatible inversion of station shortwave to TAU_CLD."""

import pytest

from geotop_py.energy import rad


def _daylight_step():
    # 29 June 2025, 11:00--12:00 in GEOtop's JD-from-year-0 convention.
    end = rad.convert_JDandYear_JDfrom0(179.5, 2025)
    return end - 1.0 / 24.0, end


def test_iswr_inversion_round_trip_with_obstructed_sky():
    JDb, JDe = _daylight_step()
    lat, lon, ST, z = 46.1, 11.1, 1.0, 1650.0
    RH, T, tau = 0.55, 12.0, 0.63
    sky, reflected = 0.78, 18.0
    P = rad.pressure_from_elevation(z)

    SWb, SWd, *_ = rad.shortwave_step(
        JDb, JDe, lat, lon, ST, RH, T, P, tau,
        sky=sky, SWrefl_surr=reflected, cast_shadow=False,
    )
    inferred = rad.tau_cloud_from_iswr(
        JDb, JDe, lat, lon, ST, z, RH, T, SWb + SWd,
        sky=sky, SWrefl_surr=reflected,
    )

    # GEOtop stops when the Erbs diffuse fraction changes by <= 0.005.
    assert inferred == pytest.approx(tau, abs=2.0e-3)


def test_direct_diffuse_pair_has_precedence_and_closes_exactly():
    JDb, JDe = _daylight_step()
    E0, Et, Delta = rad.sun((JDb + JDe) / 2.0)
    o = rad.make_others(46.1, 11.1, 1.0, Et, Delta, 0.55, 12.0,
                        rad.pressure_from_elevation(1650.0))
    # With an unobstructed station horizon, the measured component ratio is
    # exactly the Erbs kd used by the forward model.
    tau, sky, reflected = 0.72, 1.0, 0.0
    sin_alpha = rad._avg(rad._Sinalpha, o, JDb, JDe, 1.0e-6)
    SWb, SWd, _ = rad.shortwave_radiation(
        JDb, JDe, o, sin_alpha, E0, sky, reflected, tau,
    )

    inferred = rad.cloud_transmittance(
        JDb, JDe, o, E0, ISWR=1.0, sky=sky, SWrefl_surr=reflected,
        SWdirect=SWb, SWdiffuse=SWd,
    )
    assert inferred == pytest.approx(tau, rel=0.0, abs=2.0e-15)


def test_a_very_overcast_sky_is_floored_not_zero():
    """GEOtop floors every branch's result at min_tau_cloud=0.1
    (radiation.cc:691-694) -- a station reading consistent with near-total
    cloud cover must not return a tau below that floor."""
    JDb, JDe = _daylight_step()
    tau = rad.tau_cloud_from_iswr(
        JDb, JDe, 46.1, 11.1, 1.0, 1650.0, 0.55, 12.0, 5.0,  # almost no light
        sky=1.0, SWrefl_surr=0.0,
    )
    assert tau == pytest.approx(rad.MIN_TAU_CLOUD)


def test_night_is_missing_not_zero():
    end = rad.convert_JDandYear_JDfrom0(179.0, 2025) + 1.0 / 24.0
    assert rad.tau_cloud_from_iswr(
        end - 1.0 / 24.0, end, 46.1, 11.1, 1.0, 1650.0,
        0.55, 12.0, 0.0,
    ) is None


def test_the_sw_branch_clips_unconditionally():
    """the SW-alone (ISWR) branch clips tau to [0, 1] at *every* step of
    its own kd-refinement loop, unconditionally -- not an optional
    preprocessing step. A very bright/over-large ISWR must not be allowed to
    push the returned tau above 1.0, and this is the real GEOtop behaviour
    (radiation.cc:679-681), not a Python-only safety clamp."""
    JDb, JDe = _daylight_step()
    args = (JDb, JDe, 46.1, 11.1, 1.0, 1650.0, 0.55, 12.0, 2000.0)
    assert rad.tau_cloud_from_iswr(*args) == 1.0
