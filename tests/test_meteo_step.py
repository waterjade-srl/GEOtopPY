"""``meteo_step`` -- the ``Meteodistr`` composition, checked two ways.

No oracle wrapper is possible (no standalone C++ function corresponds to
"assemble one point's Meteo for one step" -- it is inline in ``meteo_distr``
plus ``energy.balance.cc``'s rain/snow split downstream, verified by reading
both in full for this module's docstring, not assumed). So: synthetic cases
that pin down the composition itself (lapse-rate selection, the two-stage
rain/snow split, previous-step fallbacks), then a real end-to-end check
against GEOtop's own reference output on ``PureDrainage`` (``PointSim=1``,
no canopy, no raster maps) for the columns
``output-tabs-SE27XX/point0001.txt`` already echoes back: ``Tair``, RH and
wind speed.
"""

import glob
import os

import pytest

from geotop_py import constants as C
from geotop_py import dates
from geotop_py.energy import rad
from geotop_py.io import gt_output, meteo, parfile, points
from geotop_py.meteo import meteodistr as md
from geotop_py.meteo.step import (
    LapseRates,
    MeteoDistrConfig,
    StationState,
    assemble_meteo,
    interpolate_station_row,
    lapse_rate_for_jd,
    resolve_tau_cloud,
)
from tools.paths import REFERENCE_1D

needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D), reason=f"reference cases not found under {REFERENCE_1D}")


def _point(Z=1000.0, slope=0.0, aspect=0.0):
    return points.PointProperties(
        ID=1, East=0.0, North=0.0, Z=Z, LC=1, soil_type=1, slope=slope,
        aspect=aspect, sky=1.0, curvature1=0.0, curvature2=0.0,
        curvature3=0.0, curvature4=0.0, BC_DepthFreeSurface=0.0,
        horizon_point=1, maxSWE=1e10, latitude=46.0, longitude=11.0, bed=1e10)


def _row(JD=100.0, T=5.0, Tdew=0.0, Ws=2.0, Wdir=180.0, Wx=None, Wy=None,
        Prec=0.0, tauC=1.0):
    # NUMBER_NOVALUE everywhere by default -- not 0.0, which resolve_tau_cloud
    # (and anything else checking _undefined) would read as a real zero
    # shortwave measurement rather than "not given".
    r = [parfile.NUMBER_NOVALUE] * len(meteo.IDX)
    r[meteo.IDX["iJDfrom0"]] = JD
    r[meteo.IDX["iT"]] = T
    r[meteo.IDX["iTdew"]] = Tdew
    r[meteo.IDX["iWs"]] = Ws
    r[meteo.IDX["iWdir"]] = Wdir
    r[meteo.IDX["iWsx"]] = parfile.NUMBER_NOVALUE if Wx is None else Wx
    r[meteo.IDX["iWsy"]] = parfile.NUMBER_NOVALUE if Wy is None else Wy
    r[meteo.IDX["iRh"]] = parfile.NUMBER_NOVALUE
    r[meteo.IDX["iPrecInt"]] = Prec
    r[meteo.IDX["itauC"]] = tauC
    return r


# ---------------------------------------------------------- lapse_rate_for_jd

def test_lapse_rate_for_jd_single_component_is_constant():
    for jd in (1.0, 100.0, 200.0, 364.0):
        assert lapse_rate_for_jd([6.5], jd) == 6.5


def test_lapse_rate_for_jd_twelve_components_span_the_year():
    comps = list(range(12))
    seen = {lapse_rate_for_jd(comps, jd) for jd in
            (1.0, 32.0, 60.0, 91.0, 121.0, 152.0, 182.0, 213.0, 244.0,
             274.0, 305.0, 335.0, 364.0)}
    assert seen == set(range(12))


def test_lapse_rate_for_jd_matches_hand_computed_bucket():
    # JD 100 of a non-leap year: bucket = floor(100*12/365) = 3
    JD0 = dates.JDandYear_to_JDfrom0(100.0, 2025)
    assert lapse_rate_for_jd(list(range(12)), JD0) == 3


# ---------------------------------------------------------------- from_parfile

@needs_reference
def test_meteo_distr_config_reads_the_correct_geotop_defaults():
    pf = parfile.parse(os.path.join(REFERENCE_1D, "Jungfraujoch", "geotop.inpts"))
    cfg = MeteoDistrConfig.from_parfile(pf)
    # geotop.inpts does not set these -- must be GEOtop's own defaults
    # (3.0/-1.0), not geotoprun.py's legacy 2.0/0.5.
    assert cfg.T_rain == 3.0
    assert cfg.T_snow == -1.0
    assert cfg.Vmin == 0.5


@needs_reference
def test_lapse_rates_from_parfile_reads_the_declared_scalar():
    pf = parfile.parse(os.path.join(REFERENCE_1D, "Jungfraujoch", "geotop.inpts"))
    lapse = LapseRates.from_parfile(pf)
    assert lapse.Ta == [5.5]
    # not declared in this file -> GEOtop's hardcoded default
    assert lapse.Tdew == [C.LapseRateTdew]
    assert lapse.Prec == [C.LapseRatePrec]


# --------------------------------------------------------------- assemble_meteo

def test_temperature_at_the_station_elevation_passes_through():
    point = _point(Z=1000.0)
    row = _row(T=5.0)
    out = assemble_meteo(row, Dt=3600.0, Z_station=1000.0, point=point,
                         lapse=LapseRates(Ta=[6.5], Tdew=[2.5], Prec=[0.0]),
                         cfg=MeteoDistrConfig(), state=StationState(),
                         JDbeg=100.0, JDend=100.0)
    assert out.Ta == pytest.approx(5.0, abs=1e-9)


def test_temperature_cools_going_up_by_the_lapse_rate():
    point = _point(Z=2000.0)   # 1000 m above the station
    row = _row(T=5.0)
    out = assemble_meteo(row, Dt=3600.0, Z_station=1000.0, point=point,
                         lapse=LapseRates(Ta=[6.5], Tdew=[2.5], Prec=[0.0]),
                         cfg=MeteoDistrConfig(), state=StationState(),
                         JDbeg=100.0, JDend=100.0)
    # 6.5 degC/km * 1 km = 6.5 degC cooler
    assert out.Ta == pytest.approx(5.0 - 6.5, abs=1e-6)


def test_missing_reading_falls_back_to_the_previous_step():
    point = _point(Z=1000.0)
    lapse = LapseRates(Ta=[6.5], Tdew=[2.5], Prec=[0.0])
    cfg = MeteoDistrConfig()
    state = StationState()
    row1 = _row(T=5.0)
    out1 = assemble_meteo(row1, 3600.0, 1000.0, point, lapse, cfg, state, 100.0, 100.0)

    row2 = _row(T=5.0)
    row2[meteo.IDX["iT"]] = parfile.NUMBER_NOVALUE   # no reading this step
    out2 = assemble_meteo(row2, 3600.0, 1000.0, point, lapse, cfg, state, 100.0, 100.0)
    assert out2.Ta == out1.Ta


def test_wind_direction_holds_when_only_speed_is_given():
    point = _point()
    lapse = LapseRates(Ta=[6.5], Tdew=[2.5], Prec=[0.0])
    cfg = MeteoDistrConfig()
    state = StationState(prev_winddir=222.0)
    row = _row(Ws=3.0)
    assemble_meteo(row, 3600.0, 1000.0, point, lapse, cfg, state, 100.0, 100.0)
    assert state.prev_winddir == 222.0    # untouched: no direction to update it


def test_the_rain_snow_split_uses_the_point_temperature_not_the_stations():
    """The station is warm (5 degC, all rain if partitioned there), but the
    point is 1500 m higher and well below freezing once lapse-corrected --
    the split must reflect the *point*'s temperature (energy.balance.cc:
    388-399), not the station's."""
    point = _point(Z=2500.0)
    lapse = LapseRates(Ta=[6.5], Tdew=[2.5], Prec=[0.0])
    cfg = MeteoDistrConfig(T_rain=3.0, T_snow=-1.0)
    row = _row(T=5.0, Prec=2.0)   # 2 mm/h at the station
    out = assemble_meteo(row, 3600.0, 1000.0, point, lapse, cfg, StationState(),
                         100.0, 100.0)
    assert out.Prain == 0.0
    assert out.Psnow > 0.0


def test_tau_cloud_passes_through_when_present():
    point = _point()
    lapse = LapseRates(Ta=[6.5], Tdew=[2.5], Prec=[0.0])
    row = _row(tauC=0.73)
    out = assemble_meteo(row, 3600.0, 1000.0, point, lapse, MeteoDistrConfig(),
                         StationState(), 100.0, 100.0)
    assert out.tau_cloud == pytest.approx(0.73)


def test_tau_cloud_falls_back_to_the_rh_temperature_estimate_when_nothing_else_is_given():
    """Tier 3 of resolve_tau_cloud (energy.balance.cc:447-448): no live SW
    this step (station-side SWb/SWd/SW all absent) and no stored column
    (itauC/iC both absent) -- the only thing left is GEOtop's own "not very
    reliable" RH/T estimate (find_cloudfactor), never a hardcoded
    clear-sky 1.0."""
    point = _point()
    lapse = LapseRates(Ta=[6.5], Tdew=[2.5], Prec=[0.0])
    row = _row()
    row[meteo.IDX["itauC"]] = parfile.NUMBER_NOVALUE
    out = assemble_meteo(row, 3600.0, 1000.0, point, lapse, MeteoDistrConfig(),
                         StationState(), 100.0, 100.0)
    expected_factor = md.find_cloudfactor(out.Ta, out.RH, point.Z, 6.5, 2.5)
    assert out.tau_cloud == pytest.approx(1.0 - 0.71 * expected_factor)


def test_live_shortwave_wins_over_a_stale_stored_column():
    """Live shortwave wins over a stale stored column."""
    lat, lon, ST, Z = 46.1, 11.1, 1.0, 1650.0
    point = _point(Z=Z)
    point.latitude, point.longitude = lat, lon
    lapse = LapseRates(Ta=[6.5], Tdew=[2.5], Prec=[0.0])
    cfg = MeteoDistrConfig(ST=ST)

    JDb = rad.convert_JDandYear_JDfrom0(179.5, 2025) - 1.0 / 24.0
    JDe = JDb + 1.0 / 24.0
    RH, T, iswr = 0.55, 12.0, 300.0
    row = _row(JD=(JDb + JDe) / 2.0, T=T, tauC=0.05)  # itauC deliberately wrong
    row[meteo.IDX["iRh"]] = RH * 100.0
    row[meteo.IDX["iSW"]] = iswr

    out = assemble_meteo(row, 3600.0, Z, point, lapse, cfg, StationState(), JDb, JDe)

    expected, expected_av = resolve_tau_cloud(JDb, JDe, row, point, 6.5, 2.5, Z,
                                              out.Ta, out.RH, cfg)
    assert out.tau_cloud == pytest.approx(expected)
    assert out.tau_cloud != pytest.approx(0.05)  # not the stale stored column
    # tau_cloud_av never gets the live tier-1 value -- it is exactly the
    # stale stored column here, and must differ from the live-derived
    # tau_cloud (the two values serve different radiation calculations).
    assert out.tau_cloud_av == pytest.approx(expected_av)
    assert out.tau_cloud_av == pytest.approx(0.05)
    assert out.tau_cloud_av != pytest.approx(out.tau_cloud)


# ------------------------------------------------------- real-case smoke test

def _load_case(name):
    sim_dir = os.path.join(REFERENCE_1D, name)
    pf = parfile.parse(os.path.join(sim_dir, "geotop.inpts"))
    stem = pf.string("MeteoFile")
    paths = sorted(glob.glob(os.path.join(sim_dir, stem + "[0-9]*.txt")))
    opt = meteo.MeteoOptions.from_parfile(pf, 1)
    table = meteo.load(paths[0], meteo.column_names(pf.strings), opt)
    return pf, table


@needs_reference
@pytest.mark.parametrize("case", ["PureDrainage", "PureDrainageRainySlope"])
def test_end_to_end_against_the_real_reference_output(case):
    """End to end against the real reference output."""
    pf, table = _load_case(case)
    sim_dir = os.path.join(REFERENCE_1D, case)
    Z_station = pf.number("MeteoStationElevation", 0, 0.0)
    chkpt = points.load(pf, sim_dir)
    point = points.properties(chkpt, 1)

    Dt = pf.number("TimeStepEnergyAndWater", 0, 3600.0)
    lapse = LapseRates.from_parfile(pf)
    cfg = MeteoDistrConfig.from_parfile(pf)
    state = StationState()

    ref = gt_output.read_point(os.path.join(sim_dir, "output-tabs-SE27XX", "point0001.txt"))
    ref_Ta = ref.col("Tair[C]")
    ref_RH = ref.col("Relative_Humidity[-]")
    ref_wind = ref.col("Wind_speed[m/s]")
    ref_Psnow = ref.col("Psnow_over_canopy[mm]")
    ref_Prain = ref.col("Prain_over_canopy[mm]")

    JD0 = dates.dateeur12_to_JDfrom0(pf.number("InitDateDDMMYYYYhhmm", 0))
    istart = 0
    checked = 0
    for i in range(len(ref.dates)):
        t = i * Dt
        JDb = JD0 + t / 86400.0
        JDe = JD0 + (t + Dt) / 86400.0
        row, istart = interpolate_station_row(table, JD0, JDb, JDe, istart, cfg)
        out = assemble_meteo(row, Dt, Z_station, point, lapse, cfg, state, JDb, JDe)
        if ref_Ta[i] == ref_Ta[i]:   # not NaN
            assert out.Ta == pytest.approx(ref_Ta[i], abs=1.0e-6)
            ref_rh_frac = ref_RH[i] / 100.0 if ref_RH[i] > 1.0 else ref_RH[i]
            assert out.RH == pytest.approx(ref_rh_frac, abs=1.0e-3)
            assert out.wind == pytest.approx(ref_wind[i], abs=1.0e-6)
            assert out.Psnow == pytest.approx(ref_Psnow[i], abs=1.0e-4)
            assert out.Prain == pytest.approx(ref_Prain[i], abs=1.0e-4)
            checked += 1
    assert checked > 400
