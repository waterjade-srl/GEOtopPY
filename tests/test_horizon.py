"""``horizon`` checked against the real files of the 13 reference cases.

There is no oracle wrapper here: ``read_horizon`` is a small, self-contained
function (unlike the tabular readers, it needs no ``ALLDATA``/keyword-table
context beyond the two column names), so a direct C++ pin would be easy to
add -- but the interesting behaviour (three-way fallback, the ``ptID`` vs
station-index distinction) is exactly what the real files already exercise,
and adding a pin here would mean building yet another oracle wrapper for a
function this small. What matters is checked against real data instead.
"""

import os

import pytest

from geotop_py.io import horizon, parfile, points
from tools.paths import REFERENCE_1D

try:
    from geotop_py import _cxx
    HAVE_ORACLE = True
except FileNotFoundError:
    HAVE_ORACLE = False

needs_oracle = pytest.mark.skipif(not HAVE_ORACLE, reason="oracle not built")

needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D), reason=f"reference cases not found under {REFERENCE_1D}")


@needs_oracle
def test_header_keywords_are_real():
    known = set(_cxx.KEYWORDS_CHAR)
    for kw in horizon.HEADER_KEYWORDS:
        assert kw in known, kw


@needs_oracle
def test_header_keywords_sit_right_after_the_soil_block():
    """assign_string_parameter slices them positionally, immediately after
    the meteo and soil blocks (parameters.cc:249-251) -- if the three ever
    drift apart, every horizon file is read with the wrong column names."""
    from geotop_py.io import meteo, soil
    start = meteo.NMET + soil.NSOILPROP
    assert _cxx.KEYWORDS_CHAR[start:start + 2] == list(horizon.HEADER_KEYWORDS)


def test_no_keyword_gives_the_flat_default():
    hor = horizon.read_horizon(parfile.STRING_NOVALUE, 1, ["az", "el"])
    assert hor == horizon.DEFAULT_HORIZON


def test_a_missing_file_gives_the_flat_default(tmp_path):
    stem = str(tmp_path / "does_not_exist")
    hor = horizon.read_horizon(stem, 1, ["az", "el"])
    assert hor == horizon.DEFAULT_HORIZON


def test_default_horizon_is_flat_and_covers_the_full_circle():
    azimuths = [row[0] for row in horizon.DEFAULT_HORIZON]
    elevations = [row[1] for row in horizon.DEFAULT_HORIZON]
    assert azimuths == [45.0, 135.0, 225.0, 315.0]
    assert all(e == 0.0 for e in elevations)


def test_a_real_file_is_read_with_the_configured_header_names(tmp_path):
    path = tmp_path / "hor0001.txt"
    path.write_text("Angle,Height\n10.0,5.0\n200.0,15.0\n")
    hor = horizon.read_horizon(str(tmp_path / "hor"), 1, ["Angle", "Height"])
    assert hor == [(10.0, 5.0), (200.0, 15.0)]


def test_a_file_missing_the_configured_columns_raises(tmp_path):
    path = tmp_path / "hor0001.txt"
    path.write_text("wrong,columns\n10.0,5.0\n")
    with pytest.raises(horizon.HorizonError):
        horizon.read_horizon(str(tmp_path / "hor"), 1, ["Angle", "Height"])


# ------------------------------------------------------------- real reference data

@needs_reference
def test_matsch_stations_read_a_real_seventy_two_row_horizon():
    for case in ("Matsch_B2_Ref_007", "Matsch_P2_Ref_007"):
        directory = os.path.join(REFERENCE_1D, case)
        pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
        stem = pf.string("HorizonMeteoStationFile")
        names = horizon.column_names(pf.strings)
        hor = horizon.read_horizon(os.path.join(directory, stem), 1, names)
        assert len(hor) == 72
        assert hor != horizon.DEFAULT_HORIZON


@needs_reference
def test_jungfraujoch_point_horizon_is_indexed_by_point_id_not_row_number():
    """Jungfraujoch's two points have IDs 32/33 (not 1/2), and its horizon
    files are named hor_0032.txt/hor_0033.txt accordingly -- read_horizon
    must be called with the point's own ptID, not its row position."""
    directory = os.path.join(REFERENCE_1D, "Jungfraujoch")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    chkpt = points.load(pf, directory)
    names = horizon.column_names(pf.strings)
    stem = pf.string("HorizonPointFile")

    ids_seen = []
    for row in range(1, len(chkpt)):
        p = points.properties(chkpt, row)
        ids_seen.append(p.ID)
        hor = horizon.read_horizon(os.path.join(directory, stem), p.ID, names)
        assert len(hor) > 4   # a real file, not the flat fallback
    assert sorted(ids_seen) == [32, 33]


@needs_reference
def test_cases_without_a_horizon_keyword_get_the_flat_default():
    directory = os.path.join(REFERENCE_1D, "PureDrainage")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    assert pf.string("HorizonMeteoStationFile") == parfile.STRING_NOVALUE
    names = horizon.column_names(pf.strings)
    hor = horizon.read_horizon(parfile.STRING_NOVALUE, 1, names)
    assert hor == horizon.DEFAULT_HORIZON


@needs_reference
@pytest.mark.parametrize("case", [
    c for c in sorted(os.listdir(REFERENCE_1D)) if os.path.isdir(
        os.path.join(REFERENCE_1D, c))
] if os.path.isdir(REFERENCE_1D) else [])
def test_every_case_resolves_without_raising(case):
    """Guard against a case whose horizon file exists but is malformed in a
    way that would only show up when the whole corpus is swept."""
    directory = os.path.join(REFERENCE_1D, case)
    inpts = os.path.join(directory, "geotop.inpts")
    if not os.path.exists(inpts):
        pytest.skip("not a case directory")
    pf = parfile.parse(inpts)
    names = horizon.column_names(pf.strings)
    stem = pf.string("HorizonMeteoStationFile")
    full_stem = stem if stem == parfile.STRING_NOVALUE else os.path.join(directory, stem)
    hor = horizon.read_horizon(full_stem, 1, names)
    assert len(hor) >= 4
