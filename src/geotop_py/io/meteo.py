"""Load a station meteo file the way GEOtop does.

``read_txt_matrix`` maps the file's header columns onto GEOtop's ``nmet`` slots
via the ``Header*`` keywords, then ``get_all_input`` completes the table:
dates are converted and shifted to the simulation's standard time, and the
derived columns each side of the wind / humidity / precipitation pairs are
filled in from whichever side the file provides.

The fill steps share a shape worth stating once, because it is easy to get
backwards: each one *replaces* the source columns with ``NUMBER_ABSENT`` once it
has derived the target -- but only when the target's ``Header*`` keyword is
actually configured. So whether ``Wind_speed`` survives in the loaded table
depends on the keywords, not on the file.

Two sentinels, two meanings, and the code tests for the right one throughout:
``NUMBER_ABSENT`` is "the file has no such column", ``NUMBER_NOVALUE`` is "this
line has no value here".

Cloudiness (``fill_meteo_data_with_cloudiness``) is not applied here: it needs
the station horizon and the solar geometry, and belongs with the radiation path.
"""

# GEOtop: src/geotop/input.cc:384-502 (meteo file loading and completion)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .. import constants as C
from .. import dates, psychro
from .parfile import NUMBER_NOVALUE, STRING_NOVALUE
from .table import NUMBER_ABSENT, read_txt_matrix

# GEOtop: src/geotop/constants.h:100-120
#: Meteo column slots. The order is the storage order.
SLOTS = ("iDate12", "iJDfrom0", "iPrecInt", "iPrec", "iWs", "iWdir",
         "iWsx", "iWsy", "iRh", "iT", "iTdew", "iSW", "iSWb", "iSWd",
         "itauC", "iC", "iLWi", "iSWn", "iTs", "iTbottom")
IDX = {name: i for i, name in enumerate(SLOTS)}
NMET = len(SLOTS)

# GEOtop: src/geotop/parameters.cc:245
#: The ``Header*`` keyword that names each slot, in slot order. These are the
#: first NMET entries of GEOtop's string keyword table, which is how
#: ``assign_string_parameter`` slices them.
HEADER_KEYWORDS = (
    "HeaderDateDDMMYYYYhhmmMeteo", "HeaderJulianDayfrom0Meteo",
    "HeaderIPrec", "HeaderPrec", "HeaderWindVelocity", "HeaderWindDirection",
    "HeaderWindX", "HeaderWindY", "HeaderRH", "HeaderAirTemp", "HeaderDewTemp",
    "HeaderSWglobal", "HeaderSWdirect", "HeaderSWdiffuse",
    "HeaderCloudSWTransmissivity", "HeaderCloudFactor", "HeaderLWin",
    "HeaderSWnet", "HeaderSurfaceTemperature", "HeaderBottomTemperature",
)


class MeteoError(ValueError):
    """Raised where GEOtop would abort while loading a meteo file."""


def _absent(value: float) -> bool:
    return int(value) == int(NUMBER_ABSENT)


def _novalue(value: float) -> bool:
    return int(value) == int(NUMBER_NOVALUE)


# GEOtop: src/geotop/parameters.cc:1157 (ST)
# GEOtop: src/geotop/parameters.cc:1317 (RHmin)
# GEOtop: src/geotop/parameters.cc:1808-1827 (station elevation and standard time)
# GEOtop: src/geotop/parameters.cc:2150-2162 (the switches and prec_as_intensity)
# GEOtop: src/geotop/parameters.cc:2165-2172 (LinearInterpolation)
@dataclass
class MeteoOptions:
    """The parameters of GEOtop's meteo completion pipeline.

    Defaults are GEOtop's own.

    The four switch names read backwards from what they control, and the pairing
    is worth spelling out because getting it wrong swaps which side of each pair
    is derived:

    ===============================  =====================
    keyword                          field
    ===============================  =====================
    ``WindAsSpeedAndDirection``      ``wind_as_dir``
    ``WindAsWindXAndWindY``          ``wind_as_xy``
    ``DewTemperatureAsRH``           ``vap_as_RH``
    ``RHAsDewTemperature``           ``vap_as_Td``
    ===============================  =====================
    """
    ST: float = 0.0            # StandardTimeSimulation
    STstat: float = 0.0        # MeteoStationStandardTime
    Zstat: float = 0.0         # MeteoStationElevation
    wind_as_xy: int = 0
    wind_as_dir: int = 1
    vap_as_Td: int = 0
    vap_as_RH: int = 1
    RHmin: float = 10.0
    prec_as_intensity: int = 0
    linear_interpolation: int = 0

    # GEOtop: src/geotop/parameters.cc:1808-1827
    @classmethod
    def from_parfile(cls, pf, station: int = 1) -> "MeteoOptions":
        """Resolve the options from a parsed ``geotop.inpts``.

        ``station`` is 1-based, as GEOtop numbers its meteo stations. Both
        station properties fall back the way GEOtop's do:
        elevation to 0, standard time to the simulation's own.
        """
        ST = pf.number("StandardTimeSimulation", 0, 0.0)
        return cls(
            ST=ST,
            STstat=pf.number("MeteoStationStandardTime", station - 1, ST),
            Zstat=pf.number("MeteoStationElevation", station - 1, 0.0),
            wind_as_dir=int(pf.number("WindAsSpeedAndDirection", 0, 1.0)),
            wind_as_xy=int(pf.number("WindAsWindXAndWindY", 0, 0.0)),
            vap_as_RH=int(pf.number("DewTemperatureAsRH", 0, 1.0)),
            vap_as_Td=int(pf.number("RHAsDewTemperature", 0, 0.0)),
            RHmin=pf.number("RHmin", 0, 10.0),
            prec_as_intensity=int(pf.number("PrecAsIntensity", 0, 0.0)),
            linear_interpolation=int(pf.number("LinearInterpolation", station - 1, 0.0)),
        )


# GEOtop: src/geotop/meteodata.cc:583-621
def fixing_dates(data: List[List[float]], ST: float, STstat: float) -> int:
    """Fill ``JDfrom0`` from ``Date12`` and shift to the simulation's time zone.

    Returns 1 if the column was added, 0 if the file
    already had it. A file with neither column is fatal in GEOtop.
    """
    jd, d12 = IDX["iJDfrom0"], IDX["iDate12"]
    if _absent(data[0][jd]) and not _absent(data[0][d12]):
        for row in data:
            row[jd] = dates.dateeur12_to_JDfrom0(row[d12])
            row[jd] += (ST - STstat) / 24.0
        return 1
    if not _absent(data[0][jd]):
        return 0
    raise MeteoError("date and time not available")


# GEOtop: src/geotop/meteodata.cc:903-923
def check_times(data: List[List[float]]) -> None:
    """``check_times``: the time axis must be strictly increasing."""
    jd = IDX["iJDfrom0"]
    for i in range(1, len(data)):
        if data[i][jd] <= data[i - 1][jd]:
            raise MeteoError(
                f"time {data[i][jd]} is not after {data[i - 1][jd]} at line {i}")


# GEOtop: src/geotop/meteodata.cc:628-672
def fill_wind_xy(data: List[List[float]], header_wx: str, header_wy: str) -> int:
    """Derive the wind components from speed and direction."""
    ws, wd = IDX["iWs"], IDX["iWdir"]
    wx, wy = IDX["iWsx"], IDX["iWsy"]
    if not (not _absent(data[0][ws]) and not _absent(data[0][wd])
            and (_absent(data[0][wx]) or _absent(data[0][wy]))):
        return 0

    replace = header_wx != STRING_NOVALUE and header_wy != STRING_NOVALUE
    for row in data:
        if not _novalue(row[ws]) and not _novalue(row[wd]):
            row[wx] = -row[ws] * math.sin(row[wd] * C.Pi / 180.0)
            row[wy] = -row[ws] * math.cos(row[wd] * C.Pi / 180.0)
        else:
            row[wx] = NUMBER_NOVALUE
            row[wy] = NUMBER_NOVALUE
        if replace:
            row[ws] = NUMBER_ABSENT
            row[wd] = NUMBER_ABSENT
    return 1 if replace else 0


# GEOtop: src/geotop/meteodata.cc:679-754
def fill_wind_dir(data: List[List[float]], header_ws: str, header_wd: str) -> int:
    """Derive speed and direction from the components.

    The quadrant logic is GEOtop's own, built on ``atan(|Wx/Wy|)`` rather than
    ``atan2``, and the ``|Wy| < 1e-10`` guard is what keeps it defined on the
    axis. Translated as-is: the branch boundaries use ``<=`` and ``>=`` on both
    components, so the axes belong to the first matching branch.
    """
    ws, wd = IDX["iWs"], IDX["iWdir"]
    wx, wy = IDX["iWsx"], IDX["iWsy"]
    if not (not _absent(data[0][wx]) and not _absent(data[0][wy])
            and (_absent(data[0][ws]) or _absent(data[0][wd]))):
        return 0

    replace = header_ws != STRING_NOVALUE and header_wd != STRING_NOVALUE
    for row in data:
        if not _novalue(row[wx]) and not _novalue(row[wy]):
            row[ws] = math.sqrt(row[wx] * row[wx] + row[wy] * row[wy])
            if abs(row[wy]) < 1.0e-10:
                a = C.Pi / 2.0
            else:
                a = math.atan(abs(row[wx] / row[wy]))
            if row[wx] <= 0 and row[wy] <= 0:
                row[wd] = a * 180.0 / C.Pi
            elif row[wx] <= 0 and row[wy] >= 0:
                row[wd] = a * 180.0 / C.Pi + 90.0
            elif row[wx] >= 0 and row[wy] >= 0:
                row[wd] = a * 180.0 / C.Pi + 180.0
            else:
                row[wd] = a * 180.0 / C.Pi + 270.0
        else:
            row[ws] = NUMBER_NOVALUE
            row[wd] = NUMBER_NOVALUE
        if replace:
            row[wx] = NUMBER_ABSENT
            row[wy] = NUMBER_ABSENT
    return 1 if replace else 0


# GEOtop: src/geotop/meteodata.cc:761-801
def fill_Tdew(data: List[List[float]], header_tdew: str, Zstat: float,
              RHmin: float) -> int:
    """Derive dew point from RH and air temperature.

    ``RHmin`` is a floor in **percent**, applied before the conversion to a
    fraction -- a station reporting 0% would otherwise give a dew point of
    minus infinity.
    """
    rh, ta, td = IDX["iRh"], IDX["iT"], IDX["iTdew"]
    if not (not _absent(data[0][rh]) and not _absent(data[0][ta])
            and _absent(data[0][td])):
        return 0

    replace = header_tdew != STRING_NOVALUE
    for row in data:
        if not _novalue(row[rh]) and not _novalue(row[ta]):
            row[td] = psychro.Tdew(row[ta], max(RHmin, row[rh]) / 100.0, Zstat)
        else:
            row[td] = NUMBER_NOVALUE
        if replace:
            row[rh] = NUMBER_ABSENT
    return 1 if replace else 0


# GEOtop: src/geotop/meteodata.cc:808-847
def fill_RH(data: List[List[float]], header_rh: str, Zstat: float) -> int:
    """Derive RH from dew point and air temperature.

    The result is stored as a **percentage**, matching the file convention.
    """
    rh, ta, td = IDX["iRh"], IDX["iT"], IDX["iTdew"]
    if not (_absent(data[0][rh]) and not _absent(data[0][ta])
            and not _absent(data[0][td])):
        return 0

    replace = header_rh != STRING_NOVALUE
    for row in data:
        if not _novalue(row[td]) and not _novalue(row[ta]):
            row[rh] = 100.0 * psychro.RHfromTdew(row[ta], row[td], Zstat)
        else:
            row[rh] = NUMBER_NOVALUE
        if replace:
            row[td] = NUMBER_ABSENT
    return 1 if replace else 0


# GEOtop: src/geotop/meteodata.cc:854-896
def fill_Pint(data: List[List[float]], header_precint: str) -> int:
    """Turn precipitation volume into intensity.

    The first line has no preceding interval, so it gets ``NUMBER_NOVALUE``
    rather than a rate -- the series is one sample shorter than it looks.
    """
    prec, pint, jd = IDX["iPrec"], IDX["iPrecInt"], IDX["iJDfrom0"]
    if not (not _absent(data[0][prec]) and _absent(data[0][pint])):
        return 0

    data[0][pint] = NUMBER_NOVALUE
    replace = header_precint != STRING_NOVALUE
    for i in range(1, len(data)):
        if not _novalue(data[i][prec]):
            data[i][pint] = data[i][prec] / (data[i][jd] - data[i - 1][jd])
            data[i][pint] /= 24.0
        else:
            data[i][pint] = NUMBER_NOVALUE
        if replace:
            data[i][prec] = NUMBER_ABSENT
    return 1 if replace else 0


# GEOtop: src/geotop/input.cc:384-502
def load(path: str, col_names: Sequence[str],
         options: Optional[MeteoOptions] = None) -> List[List[float]]:
    """Read and complete one station file, as GEOtop's ``get_all_input`` does.

    ``col_names`` gives the header name for each slot, in slot order -- i.e. the
    values of :data:`HEADER_KEYWORDS` resolved against ``geotop.inpts``.
    """
    if len(col_names) != NMET:
        raise ValueError(f"expected {NMET} column names, got {len(col_names)}")
    opt = options or MeteoOptions()

    data = read_txt_matrix(path, col_names)
    if not data:
        raise MeteoError(f"{path}: no data lines")

    if _absent(data[0][IDX["iDate12"]]) and _absent(data[0][IDX["iJDfrom0"]]):
        raise MeteoError(f"{path}: date column missing")

    fixing_dates(data, opt.ST, opt.STstat)
    check_times(data)

    if opt.wind_as_xy == 1:
        fill_wind_xy(data, col_names[IDX["iWsx"]], col_names[IDX["iWsy"]])
    if opt.wind_as_dir == 1:
        fill_wind_dir(data, col_names[IDX["iWs"]], col_names[IDX["iWdir"]])
    if opt.vap_as_Td == 1:
        fill_Tdew(data, col_names[IDX["iTdew"]], opt.Zstat, opt.RHmin)
    if opt.vap_as_RH == 1:
        fill_RH(data, col_names[IDX["iRh"]], opt.Zstat)

    if opt.linear_interpolation == 1 and not _absent(data[0][IDX["iPrec"]]):
        raise MeteoError(
            "precipitation given as volume, but LinearInterpolation is set "
            "-- remove one or the other")
    if opt.prec_as_intensity == 1:
        fill_Pint(data, col_names[IDX["iPrecInt"]])

    # GEOtop: src/geotop/input.cc:483-502
    # A second pass over exactly these three (not fill_wind_dir, not fill_RH),
    # with the opposite guard. Each fill_* is a no-op once its target column is
    # already populated, so this only does something where the first pass left
    # it undone -- which, at the keywords' defaults (0 for all three), is
    # every one of them. Skip it and Wx/Wy silently stay absent whenever a
    # station reports speed+direction rather than components, even though
    # GEOtop derives them anyway and uses them downstream (Meteodistr's wind
    # topography correction indexes the meteo table by Wx/Wy, not by speed).
    if opt.wind_as_xy != 1:
        fill_wind_xy(data, col_names[IDX["iWsx"]], col_names[IDX["iWsy"]])
    if opt.prec_as_intensity != 1:
        fill_Pint(data, col_names[IDX["iPrecInt"]])
    if opt.vap_as_Td != 1:
        fill_Tdew(data, col_names[IDX["iTdew"]], opt.Zstat, opt.RHmin)

    return data


def column_names(strings: Dict[str, str]) -> List[str]:
    """Resolve the ``Header*`` keywords against a parsed ``geotop.inpts``."""
    return [strings[k] for k in HEADER_KEYWORDS]
