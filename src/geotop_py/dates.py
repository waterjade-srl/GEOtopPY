"""GEOtop's date arithmetic (``times.cc``).

GEOtop carries two encodings and converts between them constantly:

* **dateeur12** -- a single double holding ``DDMMYYYYhhmm``, e.g.
  ``180620141900.`` for 18 June 2014 19:00. This is what ``geotop.inpts`` and
  the ``Date12`` column of every table hold.
* **JDfrom0** -- days since 1 January of year 0, with ``JDfrom0 = 1`` for
  ``01/01/0000``. This is the model's own time axis.

Verbatim translation, including the decomposition by repeated division
(``convert_dateeur12_daymonthyearhourmin``) rather than string slicing: the
encoding is a float, and the arithmetic is what defines which minute a given
double means.
"""

from __future__ import annotations

import math
from typing import Tuple

_DAYS_IN_MONTH_LEAP = (0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_DAYS_IN_MONTH_NORMAL = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


class DateError(ValueError):
    """Raised where GEOtop would abort on an unreadable date."""


# GEOtop: src/geotop/times.cc:87-102 (is_leap)
def is_leap(y: int) -> int:
    """Returns 1 or 0, as the C++ does, not a bool."""
    i = 0
    if math.fmod(y, 4.0) == 0.0:
        i = 1
        if math.fmod(y, 100.0) == 0.0:
            if math.fmod(y, 400) != 0.0:
                i = 0
    return i


# GEOtop: src/geotop/times.cc:293-314 (convert_dateeur12_daymonthyearhourmin)
def dateeur12_to_daymonthyearhourmin(date: float) -> Tuple[int, int, int, int, int]:
    """Split a ``DDMMYYYYhhmm`` double into its fields.

    The range check is GEOtop's: it rejects years outside 1700..2900, which also
    catches a value that is really a JDfrom0 fed in by mistake.
    """
    day = math.floor(date / 1.0e10)
    month = math.floor(date / 1.0e8 - day * 1.0e2)
    year = math.floor(date / 1.0e4 - day * 1.0e6 - month * 1.0e4)
    hour = math.floor(date / 1.0e2 - day * 1.0e8 - month * 1.0e6 - year * 1.0e2)
    minute = math.floor(date / 1.0e0 - day * 1.0e10 - month * 1.0e8
                        - year * 1.0e4 - hour * 1.0e2)

    if (day < 1 or day > 31 or month < 1 or month > 12 or year < 1700
            or year > 2900 or hour < 0 or hour > 23 or minute < 0 or minute > 59):
        raise DateError(f"the date {date:12.0f} cannot be read")

    return int(day), int(month), int(year), int(hour), int(minute)


def daymonthyearhourmin_to_dateeur12(day: int, month: int, year: int,
                                     hour: int, minute: int) -> float:
    return (day * 1.0e10 + month * 1.0e8 + year * 1.0e4
            + hour * 1.0e2 + minute * 1.0e0)


# GEOtop: src/geotop/times.cc:326-355 (convert_dateeur12_JDandYear)
def dateeur12_to_JDandYear(date: float) -> Tuple[float, int]:
    """Returns (day of year from 0.0, year)."""
    day, month, year, hour, minute = dateeur12_to_daymonthyearhourmin(date)
    months = _DAYS_IN_MONTH_LEAP if is_leap(year) == 1 else _DAYS_IN_MONTH_NORMAL

    JD = 0.0
    for i in range(1, month):
        JD += float(months[i])
    JD += float(day - 1)
    JD += float(hour) / 24.0
    JD += float(minute) / (24.0 * 60.0)
    return JD, year


# GEOtop: src/geotop/times.cc:111-139 (convert_JDandYear_JDfrom0)
def JDandYear_to_JDfrom0(JD: float, year: int) -> float:
    """``01/01/0000`` is JDfrom0 = 1, not 0."""
    if year < 0:
        raise DateError("dates with year earlier than 0 are not supported")
    if year == 0:
        days = 1
    else:
        days = 1
        days += 366                                  # year 0 is a leap year
        days += ((year - 1) * 365 + math.floor((year - 1) / 4.0)
                 - math.floor((year - 1) / 100.0) + math.floor((year - 1) / 400.0))
    return JD + float(days)


# GEOtop: src/geotop/times.cc:141-178 (convert_JDfrom0_JDandYear)
def JDfrom0_to_JDandYear(JDfrom0: float) -> Tuple[float, int]:
    """The year is guessed from 365.25 then walked into place."""
    if JDfrom0 < 1:
        raise DateError("dates with year earlier than 0 are not supported")
    if JDfrom0 < 367:
        return JDfrom0 - 1.0, 0

    year = int(math.floor((JDfrom0 - 1.0) / 365.25))
    JD = JDfrom0 - JDandYear_to_JDfrom0(0.0, year)
    while JD < 0 or JD >= 365 + is_leap(year):
        if JD < 0:
            year -= 1
            JD += 365 + is_leap(year)
        else:
            JD -= 365 + is_leap(year)
            year += 1
    return JD, year


# GEOtop: src/geotop/times.cc:369-377 (convert_dateeur12_JDfrom0)
def dateeur12_to_JDfrom0(date: float) -> float:
    """The conversion the meteo pipeline runs on every line."""
    JD, year = dateeur12_to_JDandYear(date)
    return JDandYear_to_JDfrom0(JD, year)


# GEOtop: src/geotop/times.cc:357-365 (convert_JDandYear_dateeur12)
def JDandYear_to_dateeur12(JD: float, year: int) -> float:
    """Day of year and year to a ``DDMMYYYYhhmm`` double."""
    day, month, hour, minute = JDandYear_to_daymonthhourmin(JD, year)
    return daymonthyearhourmin_to_dateeur12(day, month, year, hour, minute)


# GEOtop: src/geotop/times.cc:379-387 (convert_JDfrom0_dateeur12)
def JDfrom0_to_dateeur12(JDfrom0: float) -> float:
    """JDfrom0 to a ``DDMMYYYYhhmm`` double."""
    JD, year = JDfrom0_to_JDandYear(JDfrom0)
    return JDandYear_to_dateeur12(JD, year)


# GEOtop: src/geotop/times.cc:224-257 (convert_JDandYear_daymonthhourmin)
def JDandYear_to_daymonthhourmin(JD: float, year: int) -> Tuple[int, int, int, int]:
    """Day of year to ``(day, month, hour, minute)``.

    The clock arithmetic is not the obvious "floor the hour, then the minutes":
    GEOtop rounds the *total* minutes, takes that modulo 60, and then recovers
    the hour with a second rounding. The two agree in the middle of a day and
    part company at the ends of one, so this is translated as written.
    """
    JDint = int(math.floor(JD)) + 1
    frac = JD - math.floor(JD)
    months = _DAYS_IN_MONTH_LEAP if is_leap(year) == 1 else _DAYS_IN_MONTH_NORMAL

    m = 1
    while JDint > months[m]:
        JDint -= months[m]
        m += 1
    d = JDint

    minute = int(math.floor(frac * (24.0 * 60.0) + 0.5)) % 60
    hour = int(math.floor(((1440.0 * frac - float(minute)) / 60.0) + 0.5))
    return d, m, hour, minute

