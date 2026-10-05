"""Infer cloud transmissivity from a station's own shortwave record.

A station rarely measures cloudiness directly; what it measures is incoming
shortwave radiation, and cloud transmissivity is *inferred* from how far that
falls short of what a clear sky at the same sun position would deliver
(:func:`geotop_py.energy.rad.cloud_transmittance` does that inversion; this module
is the orchestration around it). At night, or when the sun is below the
horizon this station can see, no inversion is possible -- inferring
cloudiness from an absence of light would be circular -- so the day is swept
first to find sunrise/sunset (:func:`find_sunset`), the daylight span is
split into even sub-intervals, and each sub-interval gets one cloudiness
estimate representing every sample inside it. Night-time samples between the
previous day's evaluation and this day's sunrise are filled by linearly
interpolating between the two days' own transmissivity values.

Everything here reads a station's meteo table in :mod:`geotop_py.io.meteo`'s
row/column convention (each row a plain 0-based list indexed by
:data:`geotop_py.io.meteo.IDX`); nothing is mutated in place -- unlike the
source, which overwrites ``meteo[n][itauC]`` and also appends every
evaluation to a ``clouds.txt`` log file on disk. That file write is not
reproduced, for the same reason no other reader in this project writes model
state as a side effect: it has no effect on the returned transmissivity.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .. import constants as C
from .. import psychro
from ..energy import rad
from ..io.meteo import IDX
from ..io.parfile import NUMBER_ABSENT, NUMBER_NOVALUE
from ..io.table import NUMBER_ABSENT as TABLE_NUMBER_ABSENT

Row = Sequence[float]
Table = Sequence[Row]
Horizon = Sequence[Tuple[float, float]]


def _absent(v: float) -> bool:
    return int(v) == int(NUMBER_ABSENT) or int(v) == int(TABLE_NUMBER_ABSENT)


def _novalue(v: float) -> bool:
    return int(v) == int(NUMBER_NOVALUE)


def _undefined(v: float) -> bool:
    return _absent(v) or _novalue(v)


# GEOtop: src/geotop/clouds.cc:216-278 (the "plotting" file write is not reproduced)
def find_cloudiness(n: int, meteo: Table, lat_deg: float, lon_deg: float,
                     ST: float, Z: float, sky: float, SWrefl_surr: float,
                     Lozone: float = 0.0, alpha: float = 0.0, beta: float = 0.0,
                     albedo: float = 0.0) -> Optional[float]:
    """Cloud transmissivity inferred from row ``n``'s own shortwave reading.

    The averaging window is the half-intervals to the neighbouring samples
    (row 0 and the last row use their single neighbour on both sides). RH
    falls back to a dew-point-derived value, then to a fixed 0.4, and is
    floored at 0.01 either way; missing air temperature falls back to 0 degC
    -- both match the source's own defaults for an incomplete record, not an
    attempt at a better estimate.
    """
    n_last = len(meteo) - 1
    JD_n = meteo[n][IDX["iJDfrom0"]]
    if n == 0:
        JDbegin = JD_n - 0.5 * (meteo[1][IDX["iJDfrom0"]] - JD_n)
    else:
        JDbegin = 0.5 * (meteo[n - 1][IDX["iJDfrom0"]] + JD_n)
    if n == n_last:
        JDend = JD_n + 0.5 * (JD_n - meteo[n - 1][IDX["iJDfrom0"]])
    else:
        JDend = 0.5 * (JD_n + meteo[n + 1][IDX["iJDfrom0"]])

    E0, Et, Delta = rad.sun(JD_n)
    P = psychro.pressure(Z)

    RH = meteo[n][IDX["iRh"]]
    if not _undefined(RH):
        RH = RH / 100.0
    else:
        T_raw, Td_raw = meteo[n][IDX["iT"]], meteo[n][IDX["iTdew"]]
        if not _undefined(T_raw) and not _undefined(Td_raw):
            RH = psychro.RHfromTdew(T_raw, Td_raw, Z)
        else:
            RH = 0.4
    if RH < 0.01:
        RH = 0.01

    T = meteo[n][IDX["iT"]]
    if _undefined(T):
        T = 0.0

    others = rad.make_others(lat_deg, lon_deg, ST, Et, Delta, RH, T, P,
                             Lozone=Lozone, alpha=alpha, beta=beta, albedo=albedo)

    SWd = meteo[n][IDX["iSWd"]]
    SWb = meteo[n][IDX["iSWb"]]
    SW = meteo[n][IDX["iSW"]]
    return rad.cloud_transmittance(
        JDbegin, JDend, others, E0,
        ISWR=None if _undefined(SW) else SW,
        sky=sky, SWrefl_surr=SWrefl_surr,
        SWdiffuse=None if _undefined(SWd) else SWd,
        SWdirect=None if _undefined(SWb) else SWb)


# GEOtop: src/geotop/clouds.cc:287-309
def average_cloudiness(n0: int, n1: int, meteo: Table, lat_deg: float,
                        lon_deg: float, ST: float, Z: float, sky: float,
                        SWrefl_surr: float, Lozone: float = 0.0,
                        alpha: float = 0.0, beta: float = 0.0,
                        albedo: float = 0.0) -> Optional[float]:
    """Mean cloudiness over rows ``[n0, n1)``; ``None`` if any row in that
    range has no estimate (one bad sample invalidates the whole average,
    rather than being silently excluded from it) or the range is empty."""
    if n0 == n1:
        return None
    total = 0.0
    for n in range(n0, n1):
        tc = find_cloudiness(n, meteo, lat_deg, lon_deg, ST, Z, sky, SWrefl_surr,
                             Lozone, alpha, beta, albedo)
        if tc is None:
            return None
        total += tc / (n1 - n0)
    return total


# GEOtop: src/geotop/clouds.cc:317-364
def find_sunset(nist: int, meteo: Table, horizon: Horizon, lat_deg: float,
                 lon_deg: float, ST: float, rotation: float = 0.0
                 ) -> Tuple[int, int]:
    """Scan forward from row ``nist`` for the next sunrise/sunset pair.

    Returns ``(n0, n1)``: ``n0`` is the first row the station's own horizon
    stops shading (sunrise), ``n1`` the last row scanned before the sun goes
    back behind it (sunset), or the end of the table if it never does.
    ``n0`` falls back to ``n1`` if no sunrise is found in the scan (the whole
    remaining span stays shaded).
    """
    n = nist
    shad = 1  # start assuming shaded
    n0 = -1
    stop = False
    while True:
        shad0 = shad
        JD_n = meteo[n][IDX["iJDfrom0"]]
        E0, Et, Delta = rad.sun(JD_n)
        dh = (lon_deg * C.Pi / 180.0 - ST * C.Pi / 12.0 + Et) / rad.OMEGA
        alpha_deg = rad.SolarHeight(JD_n, lat_deg * C.Pi / 180.0, Delta, dh) * 180.0 / C.Pi
        direction = rad.SolarAzimuth(JD_n, lat_deg * C.Pi / 180.0, Delta, dh) \
            + rotation * C.Pi / 180.0
        # GEOtop: src/geotop/clouds.cc:346-349
        # GEOtop's own truncated Pi, not math.pi: the wrap tests and the
        # radians-to-degrees conversion both read it
        if direction < 0:
            direction += 2 * C.Pi
        if direction > 2 * C.Pi:
            direction -= 2 * C.Pi
        direction_deg = direction * 180.0 / C.Pi

        shad = rad.shadows_point(horizon, alpha_deg, direction_deg,
                                 C.Tol_h_mount, C.Tol_h_flat)

        if shad0 == 1 and shad == 0:
            n0 = n
        if shad0 == 0 and shad == 1:
            stop = True

        n += 1
        if stop or n >= len(meteo):
            break

    n1 = n - 1
    if n0 == -1:
        n0 = n1
    return n0, n1


# GEOtop: src/geotop/clouds.cc:75-179
def cloudiness(meteo: List[List[float]], horizon: Horizon, lat_deg: float,
                lon_deg: float, ST: float, Z: float, sky: float,
                SWrefl_surr: float, ndivday: int, rotation: float = 0.0,
                Lozone: float = 0.0, alpha: float = 0.0, beta: float = 0.0,
                albedo: float = 0.0) -> List[Optional[float]]:
    """Cloud transmissivity for every row of ``meteo``.

    Sweeps the record day by day (:func:`find_sunset`): each daylight span is
    split into ``ndivday`` equal sub-intervals, each evaluated once
    (:func:`average_cloudiness`) and applied to every row inside it; the
    night-time gap before the first sub-interval is filled by linear
    interpolation between the previous day's last daylight value and this
    day's first one. The final row is evaluated on its own, using the point's
    own shadow status rather than a sub-interval average.
    """
    n_lines = len(meteo)
    cloudtrans: List[Optional[float]] = [None] * n_lines

    n00 = 0
    tc0: Optional[float] = None
    while n00 < n_lines - 1:
        n0, n1 = find_sunset(n00, meteo, horizon, lat_deg, lon_deg, ST, rotation)

        ndiv = [n0]
        for k in range(1, ndivday):
            ndiv.append(n0 + k * (n1 - n0) // ndivday)
        ndiv.append(n1)

        tc = None
        for k in range(1, ndivday + 1):
            tc = average_cloudiness(ndiv[k - 1], ndiv[k], meteo, lat_deg, lon_deg,
                                    ST, Z, sky, SWrefl_surr, Lozone, alpha, beta, albedo)

            if k == 1:
                for n in range(n00, n0):
                    if tc0 is not None and tc is not None:
                        cloudtrans[n] = tc0 + (tc - tc0) * (n - n00) / (n0 - n00)
                    else:
                        cloudtrans[n] = None

            for n in range(ndiv[k - 1], ndiv[k]):
                cloudtrans[n] = tc

        tc0 = tc
        n00 = n1

    n = n00
    JD_n = meteo[n][IDX["iJDfrom0"]]
    E0, Et, Delta = rad.sun(JD_n)
    dh = (lon_deg * C.Pi / 180.0 - ST * C.Pi / 12.0 + Et) / rad.OMEGA
    height_sun = rad.SolarHeight(JD_n, lat_deg * C.Pi / 180.0, Delta, dh)
    dir_sun = rad.SolarAzimuth(JD_n, lat_deg * C.Pi / 180.0, Delta, dh) \
        + rotation * C.Pi / 180.0
    if dir_sun < 0:
        dir_sun += 2 * C.Pi
    if dir_sun > 2 * C.Pi:
        dir_sun -= 2 * C.Pi

    shaded = rad.shadows_point(horizon, height_sun * 180.0 / C.Pi, dir_sun * 180.0 / C.Pi,
                               C.Tol_h_mount, C.Tol_h_flat)
    if shaded == 0:
        cloudtrans[n] = find_cloudiness(n, meteo, lat_deg, lon_deg, ST, Z, sky,
                                        SWrefl_surr, Lozone, alpha, beta, albedo)
    else:
        cloudtrans[n] = None

    return cloudtrans


# GEOtop: src/geotop/clouds.cc:39-72
def fill_meteo_data_with_cloudiness(meteo: List[List[float]], horizon: Horizon,
                                     lat_deg: float, lon_deg: float, ST: float,
                                     Z: float, sky: float, SWrefl_surr: float,
                                     ndivday: int, rotation: float = 0.0,
                                     Lozone: float = 0.0, alpha: float = 0.0,
                                     beta: float = 0.0, albedo: float = 0.0
                                     ) -> bool:
    """Fill column ``itauC`` in place from shortwave, if any is present.

    Returns ``True`` if the column was computed (there was global or
    direct+diffuse shortwave to infer from), ``False`` if there was nothing
    to work from -- matching the source's own ``added_cloud`` flag, which the
    caller uses to decide whether the column needs writing back to disk (a
    step this port does not reproduce -- see the module docstring).
    """
    has_sw = (not _absent(meteo[0][IDX["iSW"]])
              or (not _absent(meteo[0][IDX["iSWb"]]) and not _absent(meteo[0][IDX["iSWd"]])))
    if not has_sw:
        return False

    cloudtrans = cloudiness(meteo, horizon, lat_deg, lon_deg, ST, Z, sky,
                            SWrefl_surr, ndivday, rotation, Lozone, alpha, beta, albedo)
    for n in range(len(meteo)):
        meteo[n][IDX["itauC"]] = NUMBER_NOVALUE if cloudtrans[n] is None else cloudtrans[n]
    return True
