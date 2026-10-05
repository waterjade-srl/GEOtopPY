"""Per-point, per-step meteo assembly: GEOtop's ``meteo_distr``/``Meteodistr``
composed into one call that returns a :class:`geotop_py.meteo.step.Meteo`.

Every individual piece here (:mod:`geotop_py.meteo.meteodistr`'s ``get_*``
functions, :mod:`geotop_py.psychro`) is already ported and pinned
elsewhere. End-to-end checks additionally verify call order and arguments
across the complete meteorological chain.

Composition order (``Meteodistr``): temperature, relative
humidity, wind, precipitation, pressure. With a single station,
``interpolate_meteo`` degenerates to the station's own (already
time-interpolated) reading -- no Barnes OI, so ``iobsint``/``dn`` never
enter here.

Lapse rates only cover the keyword-table branch of ``meteo_distr``
(``LRflag==0``): none of the 13 reference cases declare a
``LapseRateFile``, so the table-file branch (``LRflag==1``) is out of scope
until one does.

Rain/snow partitioning happens **twice**, deliberately, not once:
``get_precipitation`` splits at the *station's* temperature only to weight
the rain/snow elevation-correction factors internally (its return value is
one combined, still-unpartitioned intensity); the real Psnow/Prain split
for the point happens here, downstream, using the point's own -- already
lapse-corrected -- temperature or dew point (``PointEnergyBalance``).
Passing the station-side split on to the caller as if it were final would
be a real, silent divergence, not a simplification.
"""

# GEOtop: src/geotop/meteodistr.cc:56-96 (Meteodistr)
# GEOtop: src/geotop/meteo.cc:66-85 (lapse rates in meteo_distr)
# GEOtop: src/geotop/energy.balance.cc:388-400 (rain/snow split at the point)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .. import constants as C
from .. import dates, psychro
from ..energy import rad
from ..io.meteo import IDX
from ..io.parfile import NUMBER_ABSENT, NUMBER_NOVALUE
from . import meteodistr as md


@dataclass
class Meteo:
    Ta: float
    RH: float
    P: float
    wind: float
    tau_cloud: float
    Psnow: float = 0.0
    Prain: float = 0.0
    # GEOtop: src/geotop/energy.balance.cc:601-602 (longwave_radiation call)
    # Longwave uses a *different* cloud value than shortwave does
    # (PointEnergyBalance passes tau_cloud_av, not tau_cloud, to
    # longwave_radiation) -- GEOtop's own find_actual_cloudiness keeps the
    # two separate: tau_cloud prefers this step's live shortwave inversion,
    # tau_cloud_av never does. They usually agree closely (both ultimately
    # trace back to the same station), but not always. Defaults to tau_cloud
    # when only one value is supplied.
    tau_cloud_av: Optional[float] = None
    # GEOtop: src/geotop/energy.balance.cc:606
    # A directly measured incoming-longwave reading (``HeaderLWin`` column),
    # when the meteo file provides one. GEOtop prefers it over the cloud-
    # derived estimate whenever it is defined for this step
    # (``flux()``) -- ``LWin_min``/``LWin_max``
    # (the epsa-bound output columns) are *not* overridden, only the main
    # ``LWin``.
    LWin_measured: Optional[float] = None
    wind_dir: float = 0.0        # Wind_direction[deg] output column

    def __post_init__(self):
        if self.tau_cloud_av is None:
            self.tau_cloud_av = self.tau_cloud


def _novalue(v: float) -> bool:
    return int(v) == int(NUMBER_NOVALUE)


def _absent(v: float) -> bool:
    return int(v) == int(NUMBER_ABSENT)


def _undefined(v: float) -> bool:
    return _novalue(v) or _absent(v)


def _opt(v: float) -> Optional[float]:
    return None if _undefined(v) else v


# GEOtop: src/geotop/constants.h:60-62 (LapseRateTair/Tdew/Prec)
# GEOtop: src/geotop/input.cc:546-567 (met->LRd)
@dataclass
class LapseRates:
    """Keyword-table lapse rates -- the ``LRflag==0`` branch. Each list is
    the raw declared components (``LapseRateTemp`` etc. can give 1..12,
    interpreted as day-of-year buckets, not calendar months, by
    :func:`lapse_rate_for_jd`); a wholly-absent keyword falls back to
    GEOtop's own hardcoded default (``GTConst::LapseRate*``), matching
    ``met->LRd``.
    """
    Ta: List[float]
    Tdew: List[float]
    Prec: List[float]

    @classmethod
    def from_parfile(cls, pf) -> "LapseRates":
        def comps(name: str, fallback: float) -> List[float]:
            n = pf.components(name)
            vals = [pf.number(name, j, NUMBER_NOVALUE) for j in range(n)]
            return [fallback if _novalue(v) else v for v in vals]
        return cls(Ta=comps("LapseRateTemp", C.LapseRateTair),
                  Tdew=comps("LapseRateDewTemp", C.LapseRateTdew),
                  Prec=comps("LapseRatePrec", C.LapseRatePrec))


# GEOtop: src/geotop/meteo.cc:72-80
def lapse_rate_for_jd(components: List[float], JDfrom0: float) -> float:
    """GEOtop's component selection for the keyword-table branch of
    ``meteo_distr``: picks the ``j``-th of however many components were
    declared, ``j`` from the day of year scaled to that many buckets -- 12
    components behave like months, but any other count is a different,
    still valid, split of the year, not an error.
    """
    JD, year = dates.JDfrom0_to_JDandYear(JDfrom0)
    n = len(components)
    j = int(math.floor(JD * n / (365.0 + dates.is_leap(year))))
    j = min(max(j, 0), n - 1)
    return components[j]


# GEOtop: src/geotop/parameters.cc:1347-1348 (T_rain, T_snow)
@dataclass
class MeteoDistrConfig:
    """Run-wide station and point parameters with GEOtop defaults.

    Rain/snow thresholds are 3.0/-1.0 degrees Celsius, respectively.
    """
    Vmin: float = 0.5
    RHmin: float = 10.0                       # percent, GTConst-style
    ST: float = 0.0                            # StandardTimeSimulation [h from UTC]
    dew: int = 0                               # DewTempOrNormTemp
    T_rain: float = 3.0                        # ThresTempRain
    T_snow: float = -1.0                       # ThresTempSnow
    snow_corr_factor: float = 1.0
    rain_corr_factor: float = 1.0
    maxfactorP: float = 4.4                    # MaxPrecDecreaseFactorWithElev
    minfactorP: float = 0.1                    # MinPrecIncreaseFactorWithElev
    slopewtD: float = 0.0
    curvewtD: float = 0.0
    slopewtI: float = 0.0
    curvewtI: float = 0.0
    # GEOtop: src/geotop/parameters.cc:2165-2173 (linear_interpolation_meteo)
    # Per-station in GEOtop (a vector); here a
    # single flag for the one-station case PointSim=1 already assumes
    # everywhere else in this pipeline (interpolate_meteo's own single-
    # station degeneracy). Defaults to **constant**, not linear -- the
    # keyword's own default is 0, easy to get backwards
    # since "LinearInterpolation" reads like it should default to true.
    linear_interpolation: bool = False
    # GEOtop: src/geotop/energy.balance.cc:299-304
    # ActualOrProjectedArea: 0 means the point's
    # own precipitation gets a *second* cos(slope) beyond the geometric
    # "vertical to slope-normal" one every point/pixel gets -- described in
    # the source only as "another cosine correction... due to area
    # projection", specific to PointSim=1. Default 0 (both corrections
    # apply) -- getting this backwards costs a cos(slope) factor (1.15x at
    # 30 deg, 2x at 60 deg) on every drop of rain and snow.
    flag1D: bool = False
    # Sky view factor of the meteo *station* itself (MeteoStationSkyViewFactor)
    # -- used only by the live cloud-transmittance re-derivation
    # (find_tau_cloud_station reads (*met->st->sky)(i), not the point's own
    # sky), so it is a separate field from SurfaceStatics.sky.
    station_sky: float = 1.0
    # Iqbal ozone/turbidity -- same keywords surface.py's SurfaceStatics
    # needs (Lozone/AngstromAlpha/AngstromBeta); carried here too because the
    # live tau_cloud re-derivation calls atm_transmittance independently of
    # step_independent's own shortwave_step call.
    Lozone: float = 0.3
    alpha_iqbal: float = 1.3
    beta_iqbal: float = 0.1
    # GEOtop: src/geotop/parameters.cc:2240-2243
    # GEOtop: src/geotop/input.cc:2003-2016 (grid initialisation)
    # GEOtop: src/geotop/meteodistr.cc:68-85
    # Seeds for StationState.prev_* (BaseAirTemperature/BaseRelativeHumidity/
    # BaseWindSpeed/BaseWindDirection, keywords 393-396). Meteodistr's
    # Tair_grid/RH_grid/windspd_grid/winddir_grid are written once at startup
    # from these and never reset before a call (Meteodistr's "used the value
    # of the previous time step" fires when a station reading is undefined and just
    # leaves the grid as the last call left it) -- so a station with no valid
    # reading for the *entire* run (a deliberately out-of-range meteo file,
    # e.g. the PureDrainageFaked case) reports these constants forever, not
    # zero/None.
    Tair_default: float = 5.0
    RH_default: float = 0.7
    V_default: float = 0.5              # own default is Vmin, resolved below
    Vdir_default: float = 0.0
    # GEOtop: src/geotop/radiation.cc:744-748
    # The *station's* own coordinates, not the point's: find_tau_cloud_station
    # inverts the station's measured shortwave into tau_cloud using solar
    # geometry AT THE STATION ((*met->st->lat)(i)/(*met->st->lon)(i))
    # -- a station far from the point (a real,
    # documented case, not just theoretical) needs its own position, or the
    # inversion is done at the wrong hour angle. GEOtop's own default when
    # MeteoStationLatitude/Longitude are absent is the point's own
    # Latitude/Longitude (keywords.py: par->latitude/par->longitude).
    station_latitude: float = 0.0
    station_longitude: float = 0.0

    @classmethod
    def from_parfile(cls, pf) -> "MeteoDistrConfig":
        slopewt = pf.number("SlopeWeight", 0, 0.0)
        curvewt = pf.number("CurvatureWeight", 0, 0.0)
        return cls(
            Vmin=pf.number("Vmin", 0, 0.5),
            ST=pf.number("StandardTimeSimulation", 0, 0.0),
            RHmin=pf.number("RHmin", 0, 10.0),
            dew=int(pf.number("DewTempOrNormTemp", 0, 0.0)),
            T_rain=pf.number("ThresTempRain", 0, 3.0),
            T_snow=pf.number("ThresTempSnow", 0, -1.0),
            snow_corr_factor=pf.number("SnowCorrFactor", 0, 1.0),
            rain_corr_factor=pf.number("RainCorrFactor", 0, 1.0),
            maxfactorP=pf.number("MaxPrecDecreaseFactorWithElev", 0, 4.4),
            minfactorP=pf.number("MinPrecIncreaseFactorWithElev", 0, 0.1),
            slopewtD=pf.number("SlopeWeightD", 0, slopewt),
            curvewtD=pf.number("CurvatureWeightD", 0, curvewt),
            slopewtI=pf.number("SlopeWeightI", 0, slopewt),
            curvewtI=pf.number("CurvatureWeightI", 0, curvewt),
            linear_interpolation=bool(pf.number("LinearInterpolation", 0, 0.0)),
            flag1D=bool(pf.number("ActualOrProjectedArea", 0, 0.0)),
            station_sky=pf.number("MeteoStationSkyViewFactor", 0, 1.0),
            Lozone=pf.number("Lozone", 0, 0.3),
            alpha_iqbal=pf.number("AngstromAlpha", 0, 1.3),
            beta_iqbal=pf.number("AngstromBeta", 0, 0.1),
            Tair_default=pf.number("BaseAirTemperature", 0, 5.0),
            RH_default=pf.number("BaseRelativeHumidity", 0, 70.0) / 100.0,
            V_default=pf.number("BaseWindSpeed", 0, pf.number("Vmin", 0, 0.5)),
            Vdir_default=pf.number("BaseWindDirection", 0, 0.0),
            station_latitude=pf.number(
                "MeteoStationLatitude", 0, pf.number("Latitude", 0, 0.0)),
            station_longitude=pf.number(
                "MeteoStationLongitude", 0, pf.number("Longitude", 0, 0.0)),
        )


# GEOtop: src/geotop/meteodistr.cc:68-85
# GEOtop: src/geotop/input.cc:2003-2016 (grid initialisation)
@dataclass
class StationState:
    """What GEOtop keeps between steps when a reading is missing (each
    ``get_*`` falls back to "the value of the previous time step")
    -- one instance per station.

    The "previous" value before any step has ever run is GEOtop's own
    one-time grid initialisation, not zero/unset --
    construct with ``MeteoDistrConfig``'s ``*_default`` fields as the
    ``prev_*`` seeds, or a station with no valid reading for its very first
    step reports 0.0/``None`` instead of ``BaseAirTemperature`` etc."""
    prev_winddir: float = 0.0
    prev_Ta: Optional[float] = None
    prev_RH: Optional[float] = None
    prev_wind_speed: Optional[float] = None

    def copy(self) -> "StationState":
        return StationState(prev_winddir=self.prev_winddir, prev_Ta=self.prev_Ta,
                            prev_RH=self.prev_RH, prev_wind_speed=self.prev_wind_speed)


# GEOtop: src/geotop/meteo.cc:52-64
def interpolate_station_row(table, JD0: float, JDb: float, JDe: float,
                            istart: int, cfg: MeteoDistrConfig):
    """One station's time-interpolated row over ``[JDb, JDe]``, dispatching
    to linear or constant per ``cfg.linear_interpolation``
    (``meteo_distr``'s per-station choice). Returns
    ``(row, istart)``; feed ``istart`` back into the next call."""
    from . import meteodata
    fn = meteodata.time_interp_linear if cfg.linear_interpolation else meteodata.time_interp_constant
    return fn(JD0, JDb, JDe, table, IDX["iJDfrom0"], 1, istart)


# GEOtop: src/geotop/radiation.cc:708-750 (find_tau_cloud_station)
def _find_tau_cloud_live(JDbeg: float, JDend: float, E0: float, Et: float,
                         Delta: float, lat_deg: float, lon_deg: float, ST: float,
                         Z_station: float, station_sky: float, RH_raw: float,
                         T_raw: float, Td_raw: float, SWd: float, SWb: float,
                         SW: float, Lozone: float, alpha: float, beta: float
                         ) -> Optional[float]:
    """``find_tau_cloud_station``: tau_cloud inferred
    fresh, this step, from the station's own current shortwave reading --
    not a stored column. RH/T fall back exactly as
    :func:`geotop_py.meteo.clouds.find_cloudiness` does (same source pattern,
    different caller): dew-point-derived, then a fixed 0.4, floored at 0.01;
    missing T falls back to 0 degC.
    """
    P = psychro.pressure(Z_station)
    if not _undefined(RH_raw):
        RH = RH_raw / 100.0
    elif not _undefined(T_raw) and not _undefined(Td_raw):
        RH = psychro.RHfromTdew(T_raw, Td_raw, Z_station)
    else:
        RH = 0.4
    RH = max(RH, 0.01)
    T = 0.0 if _undefined(T_raw) else T_raw

    others = rad.make_others(lat_deg, lon_deg, ST, Et, Delta, RH, T, P,
                             Lozone=Lozone, alpha=alpha, beta=beta)
    return rad.cloud_transmittance(
        JDbeg, JDend, others, E0,
        ISWR=None if _undefined(SW) else SW,
        sky=station_sky, SWrefl_surr=0.0,
        SWdiffuse=None if _undefined(SWd) else SWd,
        SWdirect=None if _undefined(SWb) else SWb)


# GEOtop: src/geotop/energy.balance.cc:150-152
# GEOtop: src/geotop/energy.balance.cc:446-452
# GEOtop: src/geotop/radiation.cc:1065-1155 (find_actual_cloudiness)
# GEOtop: src/geotop/energy.balance.cc:601-602 (longwave gets tau_cloud_av)
def resolve_tau_cloud(JDbeg: float, JDend: float, row: List[float], point,
                      lapse_Ta: float, lapse_Tdew: float, Z_station: float,
                      Ta_point: float, RH_point: float,
                      cfg: MeteoDistrConfig) -> Tuple[float, float]:
    """``find_actual_cloudiness`` and its fallback in ``PointEnergyBalance``:
    three-tier fallback, in order --

    1. Re-derive tau_cloud fresh from whatever shortwave the station
       measured *this step* (beam+diffuse preferred, else global) --
       :func:`_find_tau_cloud_live`.
    2. Else, a stored average: the cloud-*factor* column (``iC``, Kimball
       1928: ``1 - 0.71*factor``) if given, else the cloud-*transmissivity*
       column (``itauC``) directly.
    3. Else (neither measured this step nor any stored column), estimate
       from the *point's* own already lapse-corrected Ta/RH via
       :func:`geotop_py.meteo.meteodistr.find_cloudfactor` -- GEOtop's own comment
       calls this "not very reliable", the last resort, not the default.

    Getting only tier 3 (or worse, an unconditional clear-sky guess) would
    miss that tier 1 is the *common* case whenever the station has live
    shortwave -- confirmed against a real data gap in ``PureDrainage``.

    Returns ``(tau_cloud, tau_cloud_av)`` -- **not** the same value.
    ``tau_cloud`` (tier 1 preferred) is what shortwave uses; ``tau_cloud_av``
    (tier 2/3 only, tier 1 never included) is what longwave uses
    (``PointEnergyBalance`` passes ``tau_cloud_av``, not ``tau_cloud``, to
    ``longwave_radiation``). They usually agree closely -- both ultimately
    trace back to the same station -- but conflating them is a real,
    measured divergence, not a simplification: it produced a 6.5 degC
    ``Tsurface`` error on ``PureDrainage``'s very first step, where the two
    tiers disagreed.
    """
    E0, Et, Delta = rad.sun((JDbeg + JDend) / 2.0)
    SWb, SWd, SW = row[IDX["iSWb"]], row[IDX["iSWd"]], row[IDX["iSW"]]
    tau_cloud = None
    if not (_undefined(SWb) or _undefined(SWd)) or not _undefined(SW):
        tau_cloud = _find_tau_cloud_live(
            JDbeg, JDend, E0, Et, Delta,
            cfg.station_latitude, cfg.station_longitude,
            cfg.ST, Z_station, cfg.station_sky,
            row[IDX["iRh"]], row[IDX["iT"]], row[IDX["iTdew"]], SWd, SWb, SW,
            cfg.Lozone, cfg.alpha_iqbal, cfg.beta_iqbal)

    tau_cloud_av = None
    iC = row[IDX["iC"]]
    itauC = row[IDX["itauC"]]
    if not _undefined(iC):
        tau_cloud_av = min(max(1.0 - 0.71 * iC, 0.0), 1.0)
    elif not _undefined(itauC):
        tau_cloud_av = min(max(itauC, 0.0), 1.0)

    if tau_cloud_av is None:
        cloud_factor = md.find_cloudfactor(Ta_point, RH_point, point.Z,
                                           lapse_Ta, lapse_Tdew)
        tau_cloud_av = 1.0 - 0.71 * cloud_factor

    resolved = tau_cloud if tau_cloud is not None else tau_cloud_av
    return resolved, tau_cloud_av


# GEOtop: src/geotop/parameters.cc:2165-2173 (LinearInterpolation)
def assemble_meteo(row: List[float], Dt: float, Z_station: float, point,
                   lapse: LapseRates, cfg: MeteoDistrConfig,
                   state: StationState, JDbeg: float, JDend: float) -> Meteo:
    """One step's point-level :class:`Meteo` from a time-interpolated
    station row.

    ``row`` is one row of interpolated meteo values (``io.meteo.IDX``
    columns), already produced by
    :func:`geotop_py.meteo.meteodata.time_interp_linear`/``_constant`` over the
    step's ``[JDbeg, JDend]`` window (:mod:`geotop_py.meteo.meteodata` decides
    linear vs. constant per *station*, not per column --
    ``LinearInterpolation``). ``point`` is a
    :class:`geotop_py.io.points.PointProperties`. Mutates ``state`` in
    place.

    ``JDbeg``/``JDend`` are passed explicitly, not read back from ``row``:
    the interpolators do not promise to leave a usable midpoint in
    ``row[iJDfrom0]``, and both the lapse-rate bucket
    (:func:`lapse_rate_for_jd`) and the live cloud re-derivation
    (:func:`resolve_tau_cloud`) need the *trial's own* window, which can
    differ step to step once the time loop starts halving ``Dt``.
    """
    JD_mid = (JDbeg + JDend) / 2.0
    Z_point = point.Z

    lapse_Ta = lapse_rate_for_jd(lapse.Ta, JD_mid)
    lapse_Tdew = lapse_rate_for_jd(lapse.Tdew, JD_mid)
    lapse_Prec = lapse_rate_for_jd(lapse.Prec, JD_mid)

    row = list(row)

    T_station = row[IDX["iT"]]
    Ta = md.get_temperature(_opt(T_station), Z_station, Z_point, lapse_Ta)
    if Ta is None:
        Ta = state.prev_Ta
    state.prev_Ta = Ta
    # The station reading itself is left where the sea-level round trip put it
    # (down to the *station's* own elevation, not the point's), and every later
    # consumer of this row -- the live cloud inversion, the rain/snow split --
    # reads that value, not the pristine one. The round trip is not the
    # identity in floating point (8.85 C comes back as 8.85 + 2.3e-14), and
    # that last bit reaches tau_cloud, then SWbeam, then the whole column.
    # GEOtop: src/geotop/meteodistr.cc:114-121
    # GEOtop: src/geotop/meteodistr.cc:138-144
    if not _undefined(T_station):
        row[IDX["iT"]] = md.get_temperature(T_station, Z_station, Z_station,
                                            lapse_Ta)

    Td_station = row[IDX["iTdew"]]
    RH = md.get_relative_humidity(_opt(Td_station), Ta, Z_station, Z_point,
                                  lapse_Tdew, cfg.RHmin)
    if RH is None:
        RH = state.prev_RH
    state.prev_RH = RH
    # GEOtop: src/geotop/meteodistr.cc:175-181
    # GEOtop: src/geotop/meteodistr.cc:200-207
    # Same round trip on the dew point, with the dew-point lapse rate.
    if not _undefined(Td_station):
        row[IDX["iTdew"]] = md.get_temperature(Td_station, Z_station, Z_station,
                                               lapse_Tdew)

    Wx, Wy, Ws = row[IDX["iWsx"]], row[IDX["iWsy"]], row[IDX["iWs"]]
    wind = md.get_wind(
        _opt(Wx), _opt(Wy), _opt(Ws), prev_winddir=state.prev_winddir,
        slopewtD=cfg.slopewtD, curvewtD=cfg.curvewtD,
        slopewtI=cfg.slopewtI, curvewtI=cfg.curvewtI,
        curvature1=point.curvature1, curvature2=point.curvature2,
        curvature3=point.curvature3, curvature4=point.curvature4,
        slope_az=point.aspect, terrain_slope=point.slope,
        windspd_min=cfg.Vmin)
    if wind is None:
        wind_speed = (state.prev_wind_speed if state.prev_wind_speed is not None
                      else cfg.V_default)
        wind_dir = state.prev_winddir
    else:
        wind_dir, wind_speed = wind
    state.prev_winddir = wind_dir
    state.prev_wind_speed = wind_speed

    P_station = row[IDX["iPrecInt"]]
    # the round-tripped station temperatures, not the pristine ones: the
    # rain/snow split at the station runs after get_temperature and
    # get_relative_humidity have already rewritten them in place
    # GEOtop: src/geotop/meteodistr.cc:455-460
    T_for_station_partition = (row[IDX["iTdew"]] if cfg.dew == 1
                               else row[IDX["iT"]])
    Prec_mmh = md.get_precipitation(
        _opt(P_station), Z_station, Z_point, T_for_station_partition,
        lapse_Prec, cfg.maxfactorP, cfg.minfactorP, cfg.T_rain, cfg.T_snow,
        cfg.snow_corr_factor, cfg.rain_corr_factor)
    Prec_mm = Prec_mmh if Prec_mmh is not None else 0.0
    # "define prec as normal (not vertical)": always applied, projecting the
    # vertically-measured rain onto the slope-normal direction.  Both cosines
    # are applied to the *intensity*, before the [mm/h] -> [mm] conversion,
    # and the angle is converted with GEOtop's own truncated Pi -- neither the
    # order of the three products nor the constant is free.
    # GEOtop: src/geotop/energy.balance.cc:299-304
    # GEOtop: src/geotop/energy.balance.cc:382
    cos_slope = math.cos(point.slope * C.Pi / 180.0)
    Prec_mm *= cos_slope
    # A second, PointSim-specific area-projection correction, unless
    # ActualOrProjectedArea says the point's own area is already projected.
    if not cfg.flag1D:
        Prec_mm *= cos_slope
    Prec_mm *= (Dt / 3600.0)

    Tdew_point = psychro.Tdew(Ta, RH, Z_point)
    T_for_point_partition = Tdew_point if cfg.dew == 1 else Ta
    Prain, Psnow = psychro.part_snow(Prec_mm, T_for_point_partition,
                                     cfg.T_rain, cfg.T_snow)

    tau_cloud, tau_cloud_av = resolve_tau_cloud(
        JDbeg, JDend, row, point, lapse_Ta, lapse_Tdew, Z_station, Ta, RH, cfg)

    return Meteo(Ta=Ta, RH=RH, P=psychro.pressure(Z_point), wind=wind_speed,
                tau_cloud=tau_cloud, tau_cloud_av=tau_cloud_av,
                Psnow=Psnow, Prain=Prain,
                LWin_measured=_opt(row[IDX["iLWi"]]), wind_dir=wind_dir)
