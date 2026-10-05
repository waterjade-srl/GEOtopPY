"""Aggregate a station's raw sample rows into one value per model timestep.

A station table (as produced by :func:`geotop_py.io.meteo.load`) is a list of
rows, each holding one column per meteo variable plus a Julian-day timestamp.
The model timestep, however, rarely lines up with the sampling times, so each
variable needs to be reduced to a single representative value over a window
``[tbeg, tend]``:

* :func:`time_interp_linear` treats consecutive samples as the endpoints of a
  piecewise-linear signal and averages the trapezoid under it -- the right
  choice for a quantity measured at an instant (temperature, wind speed).
* :func:`time_interp_constant` treats each sample as holding until the next
  one (a step function) and averages the rectangles -- the right choice for an
  accumulated quantity reported once per interval (precipitation).
* :func:`time_no_interp` skips aggregation and returns the single sample that
  brackets ``tbeg``.

All three return one value per column, with two sentinels: a column the
station never provided comes back as ``NUMBER_ABSENT``; a column that is
undefined somewhere inside the window comes back as ``NUMBER_NOVALUE`` --
these propagate rather than being silently skipped, so a gap in the data
shows up as a gap in the model forcing, not as a biased average.

:func:`find_line_data` is the shared search: it walks the row index forward
from a starting guess to the pair of rows that bracket a given time. Passing
back the ``istart`` each aggregator returns keeps the next call's search
starting close to where the previous one left off, instead of scanning from
row 0 every timestep.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .. import dates
from ..io.meteo import MeteoError
from ..io.parfile import NUMBER_NOVALUE
from ..io.table import NUMBER_ABSENT

Row = Sequence[float]
Table = Sequence[Row]


def _absent(v: float) -> bool:
    return int(v) == int(NUMBER_ABSENT)


def _novalue(v: float) -> bool:
    return int(v) == int(NUMBER_NOVALUE)


# GEOtop: src/geotop/meteodata.cc:420-439
def time_in_JDfrom0(flag: int, i: int, col: int, data: Table) -> float:
    """Row ``i``'s timestamp, as Julian days from year 0.

    ``flag=0``: the column holds a date as ``DDMMYYYYhhmm`` and is converted.
    ``flag=1``: the column already holds Julian days from year 0.
    """
    if flag == 0:
        return dates.dateeur12_to_JDfrom0(data[i][col])
    return data[i][col]


# GEOtop: src/geotop/meteodata.cc:371-418
def find_line_data(flag: int, t: float, ibeg: int, data: Table,
                    col_date: int) -> Tuple[int, int]:
    """Advance from row ``ibeg`` to the row that brackets time ``t``.

    Returns ``(line, status)``: ``status`` is 1 once ``data[line]`` and
    ``data[line + 1]`` bracket ``t`` (``t0 <= t <= t1``), 2 if the series has
    already passed ``t`` at ``line``, 3 if the series ends before reaching it.
    """
    i = ibeg
    n = len(data)
    while True:
        a = 0
        if i + 1 <= n - 1:
            t0 = time_in_JDfrom0(flag, i, col_date, data)
            t1 = time_in_JDfrom0(flag, i + 1, col_date, data)
            if t0 <= t <= t1:
                a = 1
            elif t0 > t:
                a = 2
        if i + 1 >= n - 1 and a == 0:
            a = 3
        if a != 0:
            return i, a
        i += 1


# GEOtop: src/geotop/meteodata.cc:292-337
def integrate_meas_linear_beh(flag: int, t: float, i: int, data: Table,
                               col: int, col_date: int) -> float:
    """Trapezoid area under the linear interpolant of ``data[i-1:i+1, col]``,
    between time ``t`` and row ``i``'s timestamp."""
    vi, vim1 = data[i][col], data[i - 1][col]
    if _novalue(vi) or _absent(vi) or _novalue(vim1) or _absent(vim1):
        return NUMBER_NOVALUE
    t0 = time_in_JDfrom0(flag, i - 1, col_date, data)
    t1 = time_in_JDfrom0(flag, i, col_date, data)
    if abs(t0 - t1) < 1.0e-5:
        raise MeteoError(
            f"two consecutive samples share the same time: {t0} at rows {i - 1}, {i}")
    value = ((t - t0) * vi + (t1 - t) * vim1) / (t1 - t0)
    return 0.5 * (value + vi) * (t1 - t)


# GEOtop: src/geotop/meteodata.cc:339-369
def integrate_meas_constant_beh(flag: int, t: float, i: int, data: Table,
                                 col: int, col_date: int) -> float:
    """Rectangle area under ``data[i, col]`` held constant from row ``i-1``'s
    timestamp to time ``t``."""
    vi = data[i][col]
    if _novalue(vi) or _absent(vi):
        return NUMBER_NOVALUE
    t0 = time_in_JDfrom0(flag, i - 1, col_date, data)
    return vi * (t - t0)


# GEOtop: src/geotop/meteodata.cc:42-131
def time_interp_linear(t0: float, tbeg: float, tend: float, data: Table,
                        col_date: int, flag: int, istart: int
                        ) -> Tuple[List[float], int]:
    """Column means of ``data`` over ``[tbeg, tend]``, by linear interpolation.

    Returns ``(out, istart)``; feed ``istart`` back into the next call.
    """
    ncols = len(data[0])
    i0, _ = find_line_data(flag, t0, istart, data, col_date)
    ibeg, abeg = find_line_data(flag, tbeg, i0, data, col_date)
    iend, aend = find_line_data(flag, tend, ibeg, data, col_date)

    out = [0.0] * ncols
    for c in range(ncols):
        if _absent(data[0][c]):
            out[c] = NUMBER_ABSENT
            continue
        if not (abeg == 1 and aend == 1):
            out[c] = NUMBER_NOVALUE
            continue

        acc, ok = 0.0, True

        add = integrate_meas_linear_beh(flag, tbeg, ibeg + 1, data, c, col_date)
        ok = not _novalue(add)
        if ok:
            acc += add

        if ok:
            add = integrate_meas_linear_beh(flag, tend, iend + 1, data, c, col_date)
            ok = not _novalue(add)
            if ok:
                acc -= add

        j = ibeg + 1
        while ok and j <= iend:
            t = time_in_JDfrom0(flag, j, col_date, data)
            add = integrate_meas_linear_beh(flag, t, j + 1, data, c, col_date)
            j += 1
            ok = not _novalue(add)
            if ok:
                acc += add

        out[c] = acc / (tend - tbeg) if ok else NUMBER_NOVALUE

    return out, i0


# GEOtop: src/geotop/meteodata.cc:136-241
def time_interp_constant(t0: float, tbeg: float, tend: float, data: Table,
                          col_date: int, flag: int, istart: int
                          ) -> Tuple[List[float], int]:
    """Column means of ``data`` over ``[tbeg, tend]``, treating each sample as
    a step held until the next one.

    Same signature and sentinel conventions as :func:`time_interp_linear`.
    """
    ncols = len(data[0])
    i0, _ = find_line_data(flag, t0, istart, data, col_date)
    ibeg, abeg = find_line_data(flag, tbeg, i0, data, col_date)
    iend, aend = find_line_data(flag, tend, ibeg, data, col_date)

    out = [0.0] * ncols
    for c in range(ncols):
        if _absent(data[0][c]):
            out[c] = NUMBER_ABSENT
            continue
        if not (abeg == 1 and aend == 1):
            out[c] = NUMBER_NOVALUE
            continue

        acc, ok = 0.0, True

        j = ibeg
        while ok and j < iend:
            t = time_in_JDfrom0(flag, j + 1, col_date, data)
            add = integrate_meas_constant_beh(flag, t, j + 1, data, c, col_date)
            j += 1
            ok = not _novalue(add)
            if ok:
                acc += add

        if ok:
            add = integrate_meas_constant_beh(flag, tbeg, ibeg + 1, data, c, col_date)
            ok = not _novalue(add)
            if ok:
                acc -= add

        if ok:
            add = integrate_meas_constant_beh(flag, tend, iend + 1, data, c, col_date)
            ok = not _novalue(add)
            if ok:
                acc += add

        out[c] = acc / (tend - tbeg) if ok else NUMBER_NOVALUE

    return out, i0


# GEOtop: src/geotop/meteodata.cc:243-290
def time_no_interp(flag: int, istart: int, data: Table, col_date: int,
                    tbeg: float) -> Tuple[List[float], int]:
    """The single row bracketing ``tbeg`` (within 1e-5 day), read as-is."""
    ncols = len(data[0])
    ibeg, abeg = find_line_data(flag, tbeg + 1.0e-5, istart, data, col_date)
    if abeg == 3:
        ibeg = len(data) - 1
        abeg = 1

    out = [0.0] * ncols
    for c in range(ncols):
        if _absent(data[0][c]):
            out[c] = NUMBER_ABSENT
        elif abeg == 1 and not _novalue(data[ibeg][c]):
            out[c] = data[ibeg][c]
        else:
            out[c] = NUMBER_NOVALUE
    return out, ibeg


# GEOtop: src/geotop/meteodata.cc:441-456
def find_station(metvar: int, var: Table) -> int:
    """First station row whose ``metvar`` column is not ``NUMBER_ABSENT``."""
    i = 0
    while _absent(var[i][metvar]) and i < len(var) - 1:
        i += 1
    return i
