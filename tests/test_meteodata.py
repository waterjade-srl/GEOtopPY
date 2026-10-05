"""``meteodata`` checked against GEOtop's own aggregation, on real station data.

Windows are drawn from an actual half-hourly station file (``Bro``), both
aligned to sample times and straddling them, because the boundary conditions
(a window edge landing exactly on a sample vs. between two) are where a
translation of ``find_line_data``'s three-way status is most likely to drift.
"""

import glob
import os

import pytest

from geotop_py.io import meteo, parfile, table
from geotop_py.meteo import meteodata
from tools.paths import REFERENCE_1D

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D), reason=f"reference cases not found under {REFERENCE_1D}")


def _load_station(case="Bro", rows=4000):
    """ load station."""
    inpts = os.path.join(REFERENCE_1D, case, "geotop.inpts")
    pf = parfile.parse(inpts)
    stem = pf.string("MeteoFile")
    paths = sorted(glob.glob(os.path.join(REFERENCE_1D, case, stem + "[0-9]*.txt")))
    opt = meteo.MeteoOptions.from_parfile(pf, 1)
    return meteo.load(paths[0], meteo.column_names(pf.strings), opt)[:rows]


@pytest.fixture(scope="module")
def station():
    if not os.path.isdir(REFERENCE_1D):
        pytest.skip(f"reference cases not found under {REFERENCE_1D}")
    return _load_station()


JD = meteo.IDX["iJDfrom0"]
FLAG = 1  # column already holds Julian days from 0


def _windows(data, start=1000, n=8, step=97):
    """A mix of one-sample and multi-sample windows, some off-grid."""
    out = []
    for k in range(start, start + n * step, step):
        t0 = data[k][JD]
        out.append((t0, t0, data[k + 1][JD]))                       # one sample
        out.append((t0, t0 + 0.3 * (data[k + 1][JD] - t0), data[k + 5][JD]))  # off-grid start
        out.append((t0, t0, data[k + 20][JD] - 1e-4))                # off-grid end, many samples
    return out


# ------------------------------------------------------------------- pin: search

@needs_reference
def test_find_line_data_matches_the_cxx(station):
    for i in range(0, len(station) - 1, 733):
        t = station[i][JD] + 0.5 * (station[i + 1][JD] - station[i][JD])
        line, status = meteodata.find_line_data(FLAG, t, 0, station, JD)
        cline, cstatus = _cxx.find_line_data(FLAG, t, 0, station, JD)
        assert (line, status) == (cline, cstatus)


@needs_reference
def test_time_in_JDfrom0_matches_the_cxx(station):
    for i in range(0, len(station), 1301):
        assert meteodata.time_in_JDfrom0(FLAG, i, JD, station) == \
            _cxx.time_in_JDfrom0(FLAG, i, JD, station)


# --------------------------------------------------------- pin: integrands

@needs_reference
def test_integrate_meas_linear_beh_matches_the_cxx(station):
    col = meteo.IDX["iT"]
    for i in range(1, 4000, 197):
        t = station[i - 1][JD] + 0.4 * (station[i][JD] - station[i - 1][JD])
        mine = meteodata.integrate_meas_linear_beh(FLAG, t, i, station, col, JD)
        theirs = _cxx.integrate_meas_linear_beh(FLAG, t, i, station, col, JD)
        assert mine == theirs


@needs_reference
def test_integrate_meas_constant_beh_matches_the_cxx(station):
    col = meteo.IDX["iPrec"]
    for i in range(1, 4000, 211):
        t = station[i - 1][JD] + 0.7 * (station[i][JD] - station[i - 1][JD])
        mine = meteodata.integrate_meas_constant_beh(FLAG, t, i, station, col, JD)
        theirs = _cxx.integrate_meas_constant_beh(FLAG, t, i, station, col, JD)
        assert mine == theirs


# ---------------------------------------------------------- pin: aggregators

@needs_reference
def test_time_interp_linear_matches_the_cxx(station):
    for t0, tbeg, tend in _windows(station):
        mine, mi = meteodata.time_interp_linear(t0, tbeg, tend, station, JD, FLAG, 0)
        theirs, ti = _cxx.time_interp_linear(t0, tbeg, tend, station, JD, FLAG, 0)
        assert mi == ti
        assert mine == theirs


@needs_reference
def test_time_interp_constant_matches_the_cxx(station):
    for t0, tbeg, tend in _windows(station):
        mine, mi = meteodata.time_interp_constant(t0, tbeg, tend, station, JD, FLAG, 0)
        theirs, ti = _cxx.time_interp_constant(t0, tbeg, tend, station, JD, FLAG, 0)
        assert mi == ti
        assert mine == theirs


@needs_reference
def test_time_no_interp_matches_the_cxx(station):
    for i in range(0, len(station) - 1, 611):
        tbeg = station[i][JD] + 0.2 * (station[i + 1][JD] - station[i][JD])
        mine, mi = meteodata.time_no_interp(FLAG, 0, station, JD, tbeg)
        theirs, ti = _cxx.time_no_interp(FLAG, 0, station, JD, tbeg)
        assert mi == ti
        assert mine == theirs


@needs_reference
def test_time_interp_linear_running_istart_matches_the_cxx(station):
    """The search cursor is meant to be threaded call to call, not reset."""
    mi = ti = 0
    for k in range(0, 3000, 137):
        t0 = station[k][JD]
        tbeg, tend = station[k][JD], station[k + 3][JD]
        mine, mi = meteodata.time_interp_linear(t0, tbeg, tend, station, JD, FLAG, mi)
        theirs, ti = _cxx.time_interp_linear(t0, tbeg, tend, station, JD, FLAG, ti)
        assert mi == ti
        assert mine == theirs


# -------------------------------------------------------------------- find_station

def test_find_station_matches_the_cxx():
    rows = [[table.NUMBER_ABSENT, 1.0],
            [table.NUMBER_ABSENT, 2.0],
            [5.0, 3.0],
            [6.0, 4.0]]
    assert meteodata.find_station(0, rows) == _cxx.find_station(0, rows)
    assert meteodata.find_station(0, rows) == 2


def test_find_station_falls_back_to_the_last_row_when_no_column_is_present():
    rows = [[table.NUMBER_ABSENT], [table.NUMBER_ABSENT], [table.NUMBER_ABSENT]]
    assert meteodata.find_station(0, rows) == 2


# ------------------------------------------------------------------------- invariants

def _synthetic(n=6, step=1.0, value=1.0):
    """n rows: col 0 = JDfrom0, col 1 = a constant series, col 2 = NUMBER_ABSENT."""
    return [[i * step, value, table.NUMBER_ABSENT] for i in range(n)]


def test_absent_column_propagates_through_time_interp_linear():
    data = _synthetic()
    out, _ = meteodata.time_interp_linear(0.0, 0.0, 3.0, data, 0, 1, 0)
    assert out[2] == table.NUMBER_ABSENT


def test_absent_column_propagates_through_time_interp_constant():
    data = _synthetic()
    out, _ = meteodata.time_interp_constant(0.0, 0.0, 3.0, data, 0, 1, 0)
    assert out[2] == table.NUMBER_ABSENT


def test_a_gap_makes_the_whole_window_novalue_not_a_biased_average():
    data = _synthetic(n=6)
    data[3][1] = parfile.NUMBER_NOVALUE
    out, _ = meteodata.time_interp_linear(0.0, 0.0, 5.0, data, 0, 1, 0)
    assert out[1] == parfile.NUMBER_NOVALUE


def test_a_constant_series_averages_to_itself():
    data = _synthetic(n=6, value=7.5)
    out, _ = meteodata.time_interp_linear(0.0, 0.0, 5.0, data, 0, 1, 0)
    assert out[1] == pytest.approx(7.5)
    out, _ = meteodata.time_interp_constant(0.0, 0.0, 5.0, data, 0, 1, 0)
    assert out[1] == pytest.approx(7.5)


def test_duplicate_timestamps_are_rejected():
    data = _synthetic(n=4)
    data[2][0] = data[1][0]
    with pytest.raises(meteodata.MeteoError):
        meteodata.integrate_meas_linear_beh(1, data[1][0], 2, data, 1, 0)
