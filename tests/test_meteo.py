"""The meteo loader, checked against GEOtop's own, line by line.

The oracle runs ``read_txt_matrix`` followed by the completion pipeline of
``input.cc:384-476``; this compares every value of every line of every station
file in the 13 reference cases against it, at bit equality.

Loading the forcing is where a port quietly goes wrong: a column mapped to the
wrong slot, a sentinel confused with a value, a unit left in percent. None of
that shows up as an exception -- it shows up as a plausible run that disagrees
with the reference by a few percent, months later.
"""

import glob
import os

import pytest

from geotop_py.io import meteo, parfile, table
from tools import oracle_replay
from tools.paths import REFERENCE_1D

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


def _station_files():
    """(case, inpts path, meteo file) for every station of every case."""
    out = []
    if not os.path.isdir(REFERENCE_1D):
        return out
    for case in sorted(os.listdir(REFERENCE_1D)):
        inpts = os.path.join(REFERENCE_1D, case, "geotop.inpts")
        if not os.path.exists(inpts):
            continue
        pf = parfile.parse(inpts)
        stem = pf.string("MeteoFile")
        if stem == parfile.STRING_NOVALUE:
            continue
        for path in sorted(glob.glob(os.path.join(REFERENCE_1D, case,
                                                  stem + "[0-9]*.txt"))):
            out.append((case, inpts, path))
    return out


STATIONS = _station_files()
needs_reference = pytest.mark.skipif(
    not STATIONS, reason=f"reference cases not found under {REFERENCE_1D}")


def _options(pf, station=1):
    return meteo.MeteoOptions.from_parfile(pf, station)


def test_option_keywords_are_real():
    """A differential test cannot catch an invented keyword name.

    Both sides would take the same default and agree on the wrong number, so
    the names are checked against the compiled model's table directly. Three of
    them were invented when this file was first written, and two of the four
    switches were wired to each other's field.
    """
    known = set(_cxx.KEYWORDS_NUM)
    for name in ("StandardTimeSimulation", "MeteoStationStandardTime",
                 "MeteoStationElevation", "WindAsSpeedAndDirection",
                 "WindAsWindXAndWindY", "DewTemperatureAsRH",
                 "RHAsDewTemperature", "RHmin", "PrecAsIntensity"):
        assert name in known, name


def test_an_unknown_keyword_is_rejected():
    """parfile refuses names outside the table instead of defaulting."""
    pf = parfile.parse(os.path.join(REFERENCE_1D, "PureDrainage", "geotop.inpts"))
    with pytest.raises(KeyError):
        pf.number("MinRelHumidity")          # plausible, and not a GEOtop keyword
    with pytest.raises(KeyError):
        pf.string("NotAKeyword")


def test_the_wind_and_humidity_switches_are_not_swapped():
    """Each keyword must drive the side of the pair that its name announces.

    ``WindAsSpeedAndDirection = 1`` means the run wants speed and direction, so
    it is ``fill_wind_dir`` that must run, not ``fill_wind_xy``.
    """
    pf = parfile.parse(os.path.join(REFERENCE_1D, "PureDrainage", "geotop.inpts"))
    base = meteo.MeteoOptions.from_parfile(pf)
    assert base.wind_as_dir == int(pf.number("WindAsSpeedAndDirection", 0, 1.0))
    assert base.wind_as_xy == int(pf.number("WindAsWindXAndWindY", 0, 0.0))
    assert base.vap_as_RH == int(pf.number("DewTemperatureAsRH", 0, 1.0))
    assert base.vap_as_Td == int(pf.number("RHAsDewTemperature", 0, 0.0))


@needs_reference
@pytest.mark.parametrize("case,inpts,path",
                         STATIONS, ids=[f"{c}/{os.path.basename(p)}"
                                        for c, _, p in STATIONS])
def test_loaded_table_matches_the_cxx(case, inpts, path):
    pf = parfile.parse(inpts)
    names = meteo.column_names(pf.strings)
    opt = _options(pf)

    mine = meteo.load(path, names, opt)
    kwargs = dict(
        ST=opt.ST, STstat=opt.STstat, Zstat=opt.Zstat,
        wind_as_xy=opt.wind_as_xy, wind_as_dir=opt.wind_as_dir,
        vap_as_Td=opt.vap_as_Td, vap_as_RH=opt.vap_as_RH, RHmin=opt.RHmin,
        prec_as_intensity=opt.prec_as_intensity)
    recorded = oracle_replay.recorded_digest(_cxx, "load_meteo", path, names, **kwargs)
    if recorded is not None:
        # replayed offline: too large to record, but the digest pins every byte
        assert oracle_replay.digest(mine) == recorded
        return
    theirs = _cxx.load_meteo(path, names, **kwargs)

    assert len(mine) == len(theirs)
    for i, (a, b) in enumerate(zip(mine, theirs)):
        assert a == b, f"line {i}: {a} != {b}"


@needs_reference
def test_the_corpus_is_not_trivial():
    """Guard against the comparison passing over empty or one-column files."""
    assert len(STATIONS) >= 13
    case, inpts, path = STATIONS[0]
    pf = parfile.parse(inpts)
    rows = meteo.load(path, meteo.column_names(pf.strings), _options(pf))
    assert len(rows) > 100
    present = sum(1 for v in rows[0] if int(v) != int(table.NUMBER_ABSENT))
    assert present >= 6


@needs_reference
def test_wind_xy_is_filled_even_when_the_switch_is_off():
    """input.cc runs an unconditional second pass over fill_wind_xy after the
    conditional first one, so Wx/Wy end up populated whenever a station
    reports speed+direction -- regardless of WindAsWindXAndWindY -- because
    Meteodistr's wind topography correction reads them, not the speed/dir
    columns. All 13 reference cases leave the switch at its default (0); this
    pins that the completion still derives Wx/Wy for them.
    """
    case, inpts, path = next((c, i, p) for c, i, p in STATIONS if c == "PureDrainage")
    pf = parfile.parse(inpts)
    opt = _options(pf)
    assert opt.wind_as_xy == 0
    rows = meteo.load(path, meteo.column_names(pf.strings), opt)
    wx, wy = meteo.IDX["iWsx"], meteo.IDX["iWsy"]
    assert int(rows[0][wx]) != int(table.NUMBER_ABSENT)
    assert int(rows[0][wy]) != int(table.NUMBER_ABSENT)


def test_pi_is_geotops_own_truncated_constant_not_the_library_one():
    """GTConst::Pi is a 15-digit literal (constants.h:74), not the full-precision
    double math.pi rounds to -- they differ in the last bit or two. Any angle
    conversion using math.pi where the C++ uses GTConst::Pi drifts there, and
    it will not show up until a pin test compares at exact equality (see the
    Wx/Wy mismatch this constant caused in fill_wind_xy)."""
    import math

    from geotop_py import constants
    assert constants.Pi == 3.14159265358979
    assert constants.Pi != math.pi


# ------------------------------------------------------------ slot mapping

def test_header_keywords_are_the_first_nmet_string_keywords():
    """assign_string_parameter slices them positionally (parameters.cc:245).

    If the two ever diverge, every meteo column silently shifts by one slot.
    """
    assert list(meteo.HEADER_KEYWORDS) == _cxx.KEYWORDS_CHAR[:_cxx.NMET]


def test_slot_indices_match_the_cxx():
    for name, index in meteo.IDX.items():
        assert _cxx.MET[name] == index, name
    assert meteo.NMET == _cxx.NMET


def test_the_two_sentinels_are_distinct_and_match_the_cxx():
    """"no such column" and "no value on this line" must not be conflated."""
    assert table.NUMBER_ABSENT != parfile.NUMBER_NOVALUE
    assert table.NUMBER_ABSENT == _cxx.NUMBER_ABSENT
    assert parfile.NUMBER_NOVALUE == _cxx.NUMBER_NOVALUE


# ------------------------------------------------------------------- dates

# The dates are pinned against the C++ rather than round-tripped against
# themselves: a round trip passes just as happily when both directions share a
# mistake, and one of these functions was reconstructed from its callers before
# being read, which a loose round trip did not catch.

DATES = [
    180620141900.0,     # 18/06/2014 19:00, a PureDrainage start
    170620141200.0,     # first line of its meteo file
    21020090000.0,      # 02/10/2009 00:00, Matsch
    311219992359.0,     # last minute of a year
    10120000000.0,      # first minute of a leap year
    290220120000.0,     # a leap day
    10119000000.0,      # GEOtop's own default init date
    10118000000.0,      # 01/01/1800, the low end GEOtop accepts
    311228991200.0,     # 31/12/2899, the high end
]
# Dates outside 1700..2900 are deliberately absent: GEOtop rejects them with
# t_error(), which calls exit() and takes the test process with it. See
# The oracle exposes the model's abort paths as well as its
# arithmetic, so a test must stay inside the domain the C++ accepts.


@pytest.mark.parametrize("date12", DATES)
def test_dateeur12_to_JDfrom0_matches_the_cxx(date12):
    from geotop_py import dates
    assert dates.dateeur12_to_JDfrom0(date12) == \
        _cxx.convert_dateeur12_JDfrom0(date12)


@pytest.mark.parametrize("date12", DATES)
def test_dateeur12_decomposition_matches_the_cxx(date12):
    from geotop_py import dates
    assert dates.dateeur12_to_daymonthyearhourmin(date12) == \
        _cxx.convert_dateeur12_daymonthyearhourmin(date12)


@pytest.mark.parametrize("date12", DATES)
def test_JDfrom0_back_to_dateeur12_matches_the_cxx(date12):
    from geotop_py import dates
    jd = _cxx.convert_dateeur12_JDfrom0(date12)
    assert dates.JDfrom0_to_dateeur12(jd) == _cxx.convert_JDfrom0_dateeur12(jd)


@pytest.mark.parametrize("date12", DATES)
def test_JDandYear_split_matches_the_cxx(date12):
    from geotop_py import dates
    assert dates.dateeur12_to_JDandYear(date12) == \
        _cxx.convert_dateeur12_JDandYear(date12)


@pytest.mark.parametrize("date12", DATES)
def test_clock_arithmetic_matches_the_cxx(date12):
    """The one that was reconstructed instead of read.

    GEOtop rounds total minutes then recovers the hour with a second rounding;
    flooring the hour first agrees in mid-day and diverges at the ends of one.
    """
    from geotop_py import dates
    jd = _cxx.convert_dateeur12_JDfrom0(date12)
    JD, year = _cxx.convert_JDfrom0_JDandYear(jd)
    assert dates.JDandYear_to_daymonthhourmin(JD, year) == \
        _cxx.convert_JDandYear_daymonthhourmin(JD, year)


@pytest.mark.parametrize("frac", [0.0, 1e-9, 0.25, 0.5, 0.999, 0.9999999])
def test_clock_arithmetic_at_the_edges_of_a_day(frac):
    """Sweep the fraction of a day, where the two roundings can disagree."""
    from geotop_py import dates
    JD, year = 100.0 + frac, 2014
    assert dates.JDandYear_to_daymonthhourmin(JD, year) == \
        _cxx.convert_JDandYear_daymonthhourmin(JD, year)


@pytest.mark.parametrize("year", [1899, 1900, 1996, 2000, 2012, 2014, 2100])
def test_is_leap_matches_the_cxx(year):
    from geotop_py import dates
    assert dates.is_leap(year) == _cxx.is_leap(year)
