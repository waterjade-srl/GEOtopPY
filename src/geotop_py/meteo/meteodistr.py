"""Spatial distribution of station readings onto the model domain.

GEOtop's full distribution engine spreads several stations' readings over a
grid by objective (Barnes) interpolation, then applies elevation lapse rates
and a topographic wind correction. This project targets ``PointSim=1``: one
station, one point. That collapses the general engine to its trivial branch
-- see :func:`interpolate_meteo` -- so what is ported here is the
lapse-rate and topography physics that still apply even at a single point,
not the general multi-station spatial interpolation (Barnes objective
analysis is out of scope: it is unreachable with one station and is not
implemented).

Every ``get_*`` function returns ``None`` when the station has no reading for
that variable this step, mirroring the source: nothing here decides what to
do about a gap (hold the previous value, treat precipitation as zero, ...) --
that policy belongs to the caller, since it differs by variable.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from .. import constants as C
from .. import psychro


# GEOtop: src/geotop/meteodistr.cc:556-605
def interpolate_meteo(readings: Sequence[Optional[float]]) -> Optional[float]:
    """The value GEOtop's spatial engine would place at the point.

    ``readings`` holds one entry per configured station (``None`` where that
    station has no reading this step). With zero valid readings there is
    nothing to place (``None``); with exactly one, GEOtop copies it straight
    to the grid -- the only case reachable with a single station. More than
    one valid reading would trigger Barnes objective interpolation, which is
    not implemented (not reachable with ``PointSim=1``'s single station).
    """
    valid = [v for v in readings if v is not None]
    if len(valid) == 0:
        return None
    if len(valid) == 1:
        return valid[0]
    raise NotImplementedError(
        "multi-station Barnes interpolation is out of scope for PointSim=1 "
        f"(got {len(valid)} stations with a reading)")


# GEOtop: src/geotop/meteodistr.cc:104-147
def get_temperature(T_station: Optional[float], Z_station: float, Z_point: float,
                     lapse_rate: float) -> Optional[float]:
    """Point-elevation air temperature [degC] from a station reading.

    Converts the station reading to sea level and back down to the point's
    own elevation with the same lapse rate -- not a single-step correction,
    because the two conversions go through an intermediate round trip in
    Kelvin that a direct formula would not reproduce bit-for-bit.
    """
    T = interpolate_meteo([T_station])
    if T is None:
        return None
    T_sealevel_K = psychro.temperature(0.0, Z_station, T, lapse_rate) + C.tk
    return psychro.temperature(Z_point, 0.0, T_sealevel_K, lapse_rate) - C.tk


# GEOtop: src/geotop/meteodistr.cc:155-208
def get_relative_humidity(Td_station: Optional[float], T_point: float,
                           Z_station: float, Z_point: float, lapse_rate: float,
                           RH_min: float) -> Optional[float]:
    """Point relative humidity (fraction) from a station dew-point reading.

    Takes the station's dew point through the same sea-level round trip as
    :func:`get_temperature`, then combines it with the already-converted
    point temperature ``T_point`` to get RH. Floored at ``RH_min`` (percent).
    """
    Td = interpolate_meteo([Td_station])
    if Td is None:
        return None
    Td_sealevel_K = psychro.temperature(0.0, Z_station, Td, lapse_rate) + C.tk
    Td_point = psychro.temperature(Z_point, 0.0, Td_sealevel_K, lapse_rate) - C.tk
    RH = psychro.RHfromTdew(T_point, Td_point, Z_point)
    return max(RH, RH_min / 100.0)


# GEOtop: src/geotop/meteodistr.cc:400-482
def get_precipitation(P_station: Optional[float], Z_station: float, Z_point: float,
                       T_for_partition: float, lapse_rate: float,
                       maxfactorP: float, minfactorP: float,
                       Train: float, Tsnow: float,
                       snow_corr_factor: float, rain_corr_factor: float
                       ) -> Optional[float]:
    """Point precipitation intensity from a station reading.

    Splits into rain/snow at the station (by ``T_for_partition`` -- the
    station's own dew point or air temperature depending on ``dew``, chosen
    by the caller) and corrects each phase separately, then applies
    Thornton-Running-White's non-linear elevation adjustment: precipitation
    increases going up to ``maxfactorP`` and decreases going down to
    ``minfactorP``, using the *station's own* elevation as the reference
    surface (there is only one station, so the "interpolated" reference
    elevation is just ``Z_station``).
    """
    P = interpolate_meteo([P_station])
    if P is None:
        return None
    rain, snow = psychro.part_snow(P, T_for_partition, Train, Tsnow)
    P_corr = rain_corr_factor * rain + snow_corr_factor * snow

    alfa = 1.0e-3 * lapse_rate * (Z_point - Z_station)
    alfa = max(alfa, -1.0 + 1.0e-6)
    f = (1.0 - alfa) / (1.0 + alfa)
    f = min(f, maxfactorP)
    f = max(f, minfactorP)
    return P_corr * f


# GEOtop: src/geotop/meteodistr.cc:355-368 (get_wind's u/v -> speed/dir step)
def wind_speed_dir_from_uv(Wx: float, Wy: float) -> Tuple[float, float]:
    """Wind direction [deg from N, meteorological convention] and speed [m/s]
    from the eastward/northward components.

    Built on ``atan2``, distinct from the ``atan``-based quadrant logic
    :func:`geotop_py.io.meteo.fill_wind_dir` uses to complete a station file --
    the two are separate GEOtop code paths over the same components and are
    not guaranteed to agree bit-for-bit off the axes.
    """
    rad2deg = 180.0 / C.Pi
    winddir = 270.0 - rad2deg * math.atan2(Wy, Wx)
    if winddir >= 360.0:
        winddir -= 360.0
    # ``** 0.5`` and not ``math.sqrt``: GEOtop writes ``pow(..., 0.5)`` here
    # (meteodata.cc's station-file variant writes ``sqrt`` instead, and that
    # one keeps math.sqrt), and glibc's pow disagrees with sqrt in the last
    # bit for about one argument in 1400.
    windspd = (Wx * Wx + Wy * Wy) ** 0.5
    return winddir, windspd


# GEOtop: src/geotop/meteodistr.cc:218-306
def topo_mod_winds(winddir: float, windspd: float,
                    slopewtD: float, curvewtD: float,
                    slopewtI: float, curvewtI: float,
                    curvature1: float, curvature2: float,
                    curvature3: float, curvature4: float,
                    slope_az: float, terrain_slope: float
                    ) -> Tuple[float, float]:
    """Wind speed/direction adjusted for slope and curvature (Liston & Sturm 1998).

    Speed is scaled up or down depending on whether the wind blows up or down
    the local slope (downwind/upwind weights differ); direction is deflected
    toward the aspect by up to 22.5 degrees (Ryan 1977). ``curvature1..4`` are
    the curvature in the four cardinal-ish directions (N/S, E/W, NE/SW, NW/SE);
    whichever one matches the wind's own direction (within +/-22.5 degrees of
    its axis) is the one used, negated -- a convex ridge (positive curvature)
    slows the wind down, a concave hollow speeds it up.
    """
    deg2rad = C.Pi / 180.0
    wind_slope = math.tan(deg2rad * terrain_slope * math.cos(deg2rad * (winddir - slope_az)))
    wind_slope = min(1.0, max(-1.0, wind_slope))

    if (winddir > 360.0 - 22.5 or winddir <= 22.5) or \
            (winddir > 180.0 - 22.5 and winddir <= 180.0 + 22.5):
        wind_curv = -curvature1
    elif (winddir > 90.0 - 22.5 and winddir <= 90.0 + 22.5) or \
            (winddir > 270.0 - 22.5 and winddir <= 270.0 + 22.5):
        wind_curv = -curvature2
    elif (winddir > 135.0 - 22.5 and winddir <= 135.0 + 22.5) or \
            (winddir > 315.0 - 22.5 and winddir <= 315.0 + 22.5):
        wind_curv = -curvature3
    elif (winddir > 45.0 - 22.5 and winddir <= 45.0 + 22.5) or \
            (winddir > 225.0 - 22.5 and winddir <= 225.0 + 22.5):
        wind_curv = -curvature4
    else:
        wind_curv = 0.0  # unreached: the four bands above tile the full circle

    windwt = 1.0
    windwt += (slopewtD if wind_slope < 0 else slopewtI) * wind_slope
    windwt += (curvewtD if wind_curv < 0 else curvewtI) * wind_curv
    windspd = windspd * windwt

    dirdiff = slope_az - winddir
    # ``deg2rad * (2.0 * dirdiff)``: the doubled angle is formed first,
    # then converted -- not (deg2rad*2)*dirdiff
    winddir = winddir - 22.5 * min(abs(wind_slope), 1.0) * math.sin(
        deg2rad * (2.0 * dirdiff))
    if winddir > 360.0:
        winddir -= 360.0
    elif winddir < 0.0:
        winddir += 360.0

    return winddir, windspd


# GEOtop: src/geotop/meteodistr.cc:316-397
def get_wind(Wx: Optional[float], Wy: Optional[float], Ws: Optional[float], *,
             prev_winddir: float,
             slopewtD: float, curvewtD: float, slopewtI: float, curvewtI: float,
             curvature1: float, curvature2: float, curvature3: float, curvature4: float,
             slope_az: float, terrain_slope: float,
             windspd_min: float) -> Optional[Tuple[float, float]]:
    """Point wind direction/speed for one step.

    If the station provides both wind components this step, they are
    converted to speed/direction and run through the topographic correction.
    Otherwise, if it at least provides a scalar speed, that speed is used
    unmodified and the direction holds at ``prev_winddir`` -- GEOtop does not
    apply the topographic correction without a direction to correct.
    Returns ``None`` if neither is available this step. Speed is floored at
    ``windspd_min`` either way.
    """
    u, v = interpolate_meteo([Wx]), interpolate_meteo([Wy])
    if u is not None and v is not None:
        winddir, windspd = wind_speed_dir_from_uv(u, v)
        winddir, windspd = topo_mod_winds(
            winddir, windspd, slopewtD, curvewtD, slopewtI, curvewtI,
            curvature1, curvature2, curvature3, curvature4,
            slope_az, terrain_slope)
    else:
        windspd = interpolate_meteo([Ws])
        if windspd is None:
            return None
        winddir = prev_winddir

    return winddir, max(windspd, windspd_min)


# GEOtop: src/geotop/meteodistr.cc:513-536
def find_cloudfactor(Tair: float, RH: float, Z: float,
                      T_lapse_rate: float, Td_lapse_rate: float) -> float:
    """Cloud fraction [0, 1] from surface air temperature and RH (Walcek 1994).

    Projects surface conditions up to the 700mb level (taken as 3000m in a
    standard atmosphere) using the same lapse rates as the temperature/RH
    distribution, then reads the cloud fraction off Walcek's empirical
    RH-at-700mb curve.
    """
    Z_ref = 3000.0
    press_ratio = 0.7

    f_max = 78.0 + 80.0 / 15.5
    one_minus_RHe = 0.196 + (0.76 - 80.0 / 2834.0) * (1.0 - press_ratio)
    f_1 = f_max * (press_ratio - 0.1) / 0.6 / 100.0

    Td = psychro.Tdew(Tair, max(0.1, RH), Z)
    Td_700 = psychro.temperature(Z_ref, Z, Td, Td_lapse_rate)
    Tair_700 = psychro.temperature(Z_ref, Z, Tair, T_lapse_rate)
    rh_700 = psychro.RHfromTdew(Tair_700, Td_700, Z)

    fcloud = f_1 * math.exp((rh_700 - 1.0) / one_minus_RHe)
    fcloud = min(1.0, fcloud)
    fcloud = max(0.0, fcloud)
    return fcloud
