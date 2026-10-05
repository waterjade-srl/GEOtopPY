"""The point list reader, checked against GEOtop's own.

``points_from_keywords`` and ``read_point_file`` are pinned directly at the
oracle for every one of the 13 reference cases. ``apply_defaults`` has no
oracle wrapper (it is the last of three passes, entirely local arithmetic), so
it is checked by invariant and by hand-picked cases instead, and
``fill_from_maps`` -- the pass between them -- is checked against the
topography GEOtop's own run log reports for the two cases that use it.
"""

import os
import pathlib
import tempfile

import pytest

from geotop_py.io import parfile, points
from tools.paths import REFERENCE_1D

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


def _cases():
    if not os.path.isdir(REFERENCE_1D):
        return []
    return [(c, os.path.join(REFERENCE_1D, c))
            for c in sorted(os.listdir(REFERENCE_1D))
            if os.path.exists(os.path.join(REFERENCE_1D, c, "geotop.inpts"))]


CASES = _cases()
needs_reference = pytest.mark.skipif(
    not CASES, reason=f"reference cases not found under {REFERENCE_1D}")

#: The two cases whose topography comes from raster maps, with the values
#: GEOtop's own log reports for their single point: elevation [m], land cover,
#: soil type, slope [deg], aspect [deg], sky view factor [-], and the four
#: curvatures [1/m].
RASTER_CASES = {
    "Bro": (326.0, 1, 1, 11.048352, 219.805571, 0.853552,
            0.01, 0.0075, 0.0075, 0.0125),
    "Calabria": (486.7422, 1, 1, 32.571644, 23.769, 0.872554,
                 0.000004, -0.003072, -0.00744, 0.003164),
}


# --------------------------------------------------------------- column indices

def test_point_columns_match_the_cxx():
    for name, index in points.COL.items():
        assert _cxx.POINT[name] == index, name
    assert points.PTTOT == _cxx.POINT["ptTOT"]


def test_header_keywords_are_real():
    known = set(_cxx.KEYWORDS_CHAR)
    for kw in points.HEADER_KEYWORDS:
        assert kw in known, kw


def test_point_keywords_are_real():
    known = set(_cxx.KEYWORDS_NUM)
    for kw in points.POINT_KEYWORDS:
        if kw == "NONE":
            continue
        assert kw in known, kw


# --------------------------------------------------------- oracle differential

@needs_reference
@pytest.mark.parametrize("case,directory", CASES, ids=[c for c, _ in CASES])
def test_keyword_matrix_matches_the_cxx(case, directory):
    """Keyword matrix matches the cxx."""
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    mine = points.points_from_keywords(pf)
    # The oracle builds the same matrix as a side effect of read_point_file
    # with no file (None): pass the keyword matrix through unchanged and
    # compare, since gt_points_from_keywords is not separately exposed.
    theirs = _cxx.read_point_file(parfile.STRING_NOVALUE,
                                  points.column_names(pf.strings), mine)
    assert mine == theirs


@needs_reference
@pytest.mark.parametrize("case,directory", CASES, ids=[c for c, _ in CASES])
def test_read_point_file_matches_the_cxx(case, directory):
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    chkpt = points.points_from_keywords(pf)
    names = points.column_names(pf.strings)
    stem = pf.string("PointFile")
    path = None if stem == parfile.STRING_NOVALUE else os.path.join(directory, stem)

    mine = points.read_point_file(path, names, chkpt)
    theirs = _cxx.read_point_file(
        parfile.STRING_NOVALUE if path is None else path, names, chkpt)
    assert mine == theirs


@needs_reference
def test_the_corpus_is_not_trivial():
    """Guard against the comparison passing over a trivially empty case:
    Jungfraujoch's two points come from listpoints.txt, not inline keywords,
    so it is the file pass -- not points_from_keywords alone -- that must
    produce more than one row."""
    assert len(CASES) >= 13
    directory = os.path.join(REFERENCE_1D, "Jungfraujoch")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    chkpt = points.points_from_keywords(pf)
    names = points.column_names(pf.strings)
    stem = pf.string("PointFile")
    path = None if stem == parfile.STRING_NOVALUE else os.path.join(directory, stem)
    chkpt = points.read_point_file(path, names, chkpt)
    assert len(chkpt) - 1 == 2


# ------------------------------------------------------------- map dependency

@needs_reference
@pytest.mark.parametrize("case,directory", CASES, ids=[c for c, _ in CASES])
def test_load_resolves_every_reference_case(case, directory):
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    chkpt = points.load(pf, directory)
    assert len(chkpt) - 1 >= 1


@needs_reference
@pytest.mark.parametrize("case,directory", CASES, ids=[c for c, _ in CASES])
def test_only_the_raster_cases_depend_on_a_map(case, directory):
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    chkpt = points.points_from_keywords(pf)
    stem = pf.string("PointFile")
    chkpt = points.read_point_file(
        None if stem == parfile.STRING_NOVALUE else os.path.join(directory, stem),
        points.column_names(pf.strings), chkpt)
    derived = points.map_derived_columns(chkpt, points.available_maps(pf, directory))
    assert bool(derived) == (case in RASTER_CASES)


@needs_reference
@pytest.mark.parametrize("case", sorted(RASTER_CASES))
def test_map_derived_topography_matches_the_reference_run(case):
    """The whole map pass, against what GEOtop resolved for the same point:
    elevation off the smoothed DEM, three columns off their own map, and the
    four curvatures computed over the whole grid."""
    directory = os.path.join(REFERENCE_1D, case)
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    p = points.properties(points.load(pf, directory), 1)
    got = (p.Z, p.LC, p.soil_type, p.slope, p.aspect, p.sky,
           p.curvature1, p.curvature2, p.curvature3, p.curvature4)
    for value, want in zip(got, RASTER_CASES[case]):
        assert value == pytest.approx(want, abs=5.0e-7)


@needs_reference
def test_a_slope_that_would_be_derived_from_the_dem_is_refused(tmp_path):
    """Deriving slope, aspect or sky view factor from the elevation model is
    the one map path not ported: it must raise, not silently default to flat.
    """
    directory = os.path.join(REFERENCE_1D, "Bro")
    work = tmp_path / "Bro"
    (work / "input_maps").mkdir(parents=True)
    os.symlink(os.path.join(directory, "listpoints.txt"), work / "listpoints.txt")
    for name in os.listdir(os.path.join(directory, "input_maps")):
        if name != "slope.asc":
            os.symlink(os.path.join(directory, "input_maps", name),
                       work / "input_maps" / name)

    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    with pytest.raises(NotImplementedError):
        points.load(pf, str(work))


# -------------------------------------------------------------- apply_defaults

def _synthetic_row(**overrides):
    row = {name: parfile.NUMBER_NOVALUE for name in points.COLUMNS}
    row["ptX"] = 100.0
    row["ptY"] = 200.0
    row.update(overrides)
    chkpt = [[0.0] * (points.PTTOT + 1),
             [0.0] + [row[name] for name in points.COLUMNS]]
    return chkpt


def test_apply_defaults_fills_flat_unshaded_unit_land_cover():
    chkpt = _synthetic_row()
    out = points.apply_defaults(chkpt, points.PointOptions())
    p = points.properties(out, 1)
    assert p.slope == 0.0 and p.aspect == 0.0
    assert p.sky == 1.0
    assert p.LC == 1
    assert p.curvature1 == p.curvature2 == p.curvature3 == p.curvature4 == 0.0
    assert p.Z == 0.0


def test_apply_defaults_leaves_coordinates_alone():
    """A point without coordinates stays without them -- there is nothing
    sensible to default them to, unlike every other column."""
    chkpt = [[0.0] * (points.PTTOT + 1),
             [0.0] + [parfile.NUMBER_NOVALUE] * points.PTTOT]
    out = points.apply_defaults(chkpt, points.PointOptions())
    assert out[1][points.COL["ptX"]] == parfile.NUMBER_NOVALUE
    assert out[1][points.COL["ptY"]] == parfile.NUMBER_NOVALUE


def test_apply_defaults_uses_the_run_wide_lat_lon_and_boundary_depth():
    opt = points.PointOptions(DepthFreeSurface=1500.0, latitude=44.5, longitude=11.0,
                              soil_type_land_default=2)
    chkpt = _synthetic_row()
    out = points.apply_defaults(chkpt, opt)
    p = points.properties(out, 1)
    assert p.latitude == 44.5 and p.longitude == 11.0
    assert p.BC_DepthFreeSurface == 1500.0
    assert p.soil_type == 2


def test_apply_defaults_horizon_falls_back_to_the_points_own_id():
    chkpt = _synthetic_row()
    out = points.apply_defaults(chkpt, points.PointOptions())
    p = points.properties(out, 1)
    assert p.horizon_point == p.ID == 1


def test_apply_defaults_does_not_override_an_explicit_value():
    chkpt = _synthetic_row(ptS=12.0, ptSKY=0.8, ptLC=3.0)
    out = points.apply_defaults(chkpt, points.PointOptions())
    p = points.properties(out, 1)
    assert p.slope == 12.0
    assert p.sky == 0.8
    assert p.LC == 3


# ---------------------------------------------------------------- properties()

def test_properties_rejects_non_positive_land_cover_or_soil_type():
    chkpt = _synthetic_row(ptLC=0.0)
    out = points.apply_defaults(chkpt, points.PointOptions())
    with pytest.raises(points.PointError):
        points.properties(out, 1)


def test_properties_bedrock_default_is_far_below_any_real_column():
    chkpt = _synthetic_row()
    out = points.apply_defaults(chkpt, points.PointOptions())
    p = points.properties(out, 1)
    assert p.bed == points.DEFAULT_BEDROCK_DEPTH
    assert p.bed > 1.0e6          # deeper than any plausible soil column


# -------------------------------------------------------------- file cascade

def _write_point_file(tmp_path, header, rows):
    path = tmp_path / "listpoints.txt"
    path.write_text(",".join(header) + "\n"
                    + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")
    return str(tmp_path / "listpoints")


def _xy_col_names():
    """A col_names list where only X/Y resolve to real header names -- every
    other slot is left at "none" so it can never accidentally match a column."""
    names = ["none"] * points.PTTOT
    names[points.COL["ptX"] - 1] = "xcoord"
    names[points.COL["ptY"] - 1] = "ycoord"
    return names


def test_file_cell_falls_back_to_the_keyword_row_of_the_same_index():
    """A file cell that is absent/novalue takes the keyword matrix's own row,
    not a fixed default -- and only apply_defaults gives a fixed default."""
    chkpt = [[0.0] * (points.PTTOT + 1),
             [0.0] + [parfile.NUMBER_NOVALUE] * points.PTTOT]
    chkpt[1][points.COL["ptS"]] = 7.0     # slope from the keyword pass

    with tempfile.TemporaryDirectory() as d:
        stem = _write_point_file(pathlib.Path(d), ["xcoord", "ycoord"], [[100.0, 200.0]])
        out = points.read_point_file(stem, _xy_col_names(), chkpt)
        assert out[1][points.COL["ptX"]] == 100.0
        assert out[1][points.COL["ptY"]] == 200.0
        assert out[1][points.COL["ptS"]] == 7.0    # fell back to the keyword row


def test_a_file_with_more_points_repeats_the_keyword_matrixs_last_row():
    chkpt = [[0.0] * (points.PTTOT + 1),
             [0.0] + [parfile.NUMBER_NOVALUE] * points.PTTOT]
    chkpt[1][points.COL["ptS"]] = 3.0

    with tempfile.TemporaryDirectory() as d:
        stem = _write_point_file(pathlib.Path(d), ["xcoord", "ycoord"],
                                 [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        out = points.read_point_file(stem, _xy_col_names(), chkpt)
        assert len(out) - 1 == 3
        for n in range(1, 4):
            assert out[n][points.COL["ptS"]] == 3.0


def test_no_file_at_all_keeps_the_keyword_matrix_unchanged():
    chkpt = [[0.0] * (points.PTTOT + 1), [0.0] + [5.0] * points.PTTOT]
    names = ["none"] * points.PTTOT
    out = points.read_point_file(None, names, chkpt)
    assert out == chkpt
    out2 = points.read_point_file(parfile.STRING_NOVALUE, names, chkpt)
    assert out2 == chkpt
