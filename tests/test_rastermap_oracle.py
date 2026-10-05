"""The raster reader and the terrain operators, pinned against the oracle.

Three things are checked cell by cell, on synthetic grids and on the reference
cases' own maps: the ESRI ASCII reader (which does not go through ``float()``
and does not read the cell stream line by line), the low-pass filter, and the
four directional curvatures. All three have edge and missing-cell behaviour
that a plausible rewrite gets subtly wrong, and none of it shows up on a flat
test grid -- so the ridge and hollow cases below are built deliberately
asymmetric in both axes and both diagonals.
"""

import math
import os

import pytest

from geotop_py.io import geomorphology, rastermap
from tools import oracle_replay
from tools.paths import REFERENCE_1D

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


NOVALUE = -9999.0

#: Every raster the reference cases read, as (case, stem).
REFERENCE_MAPS = [
    ("Bro", "input_maps/dem"),
    ("Bro", "input_maps/landcover"),
    ("Bro", "input_maps/soiltype"),
    ("Bro", "input_maps/slope"),
    ("Bro", "input_maps/aspect"),
    ("Bro", "input_maps/SkyViewFactorMapFile"),
    ("Calabria", "input_maps/dem_mybasin_2"),
    ("Calabria", "input_maps/landcover_mybasin_2"),
    ("Calabria", "input_maps/soiltype_mybasin_2"),
    ("Calabria", "input_maps/gradient_grad_mybasin_2"),
    ("Calabria", "input_maps/skyview_mybasin_2"),
    ("Calabria", "aspect_mybasin_2"),
]

needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D),
    reason=f"reference cases not found under {REFERENCE_1D}")


# ------------------------------------------------------------------- fixtures

def _grid(rows):
    """A 1-based grid from a 0-based list of rows."""
    nc = len(rows[0])
    out = [[0.0] * (nc + 1)]
    for row in rows:
        out.append([0.0] + [float(v) for v in row])
    return out


def _plain(grid):
    """A 1-based grid back as the 0-based list of rows the oracle takes."""
    return [row[1:] for row in grid[1:]]


FLAT = _grid([[100.0] * 5 for _ in range(5)])

#: A ridge running NE-SW, so that the two diagonal curvatures differ from each
#: other and both differ from the two axial ones.
RIDGE = _grid([[100.0 + 3.0 * r - 2.0 * abs(c - 2) for c in range(6)]
               for r in range(5)])

#: A hollow with a hole in it: the missing cells make the filter's count and
#: the curvature's both-neighbours test do something other than nothing.
HOLLOW = _grid([[100.0 + (r - 2) ** 2 + 0.5 * (c - 3) ** 2 for c in range(7)]
                for r in range(6)])
for _r, _c in ((2, 3), (2, 4), (5, 1), (6, 7)):
    HOLLOW[_r][_c] = NOVALUE

GRIDS = {"flat": FLAT, "ridge": RIDGE, "hollow": HOLLOW}


# ---------------------------------------------------------------- topofilter

@pytest.mark.parametrize("name", sorted(GRIDS))
@pytest.mark.parametrize("ntimes", [0, 1, 2, 5])
@pytest.mark.parametrize("n", [1, 2])
def test_multipass_topofilter_matches_the_cxx(name, ntimes, n):
    Z = GRIDS[name]
    mine = geomorphology.multipass_topofilter(ntimes, Z, NOVALUE, n)
    theirs = _cxx.multipass_topofilter(ntimes, _plain(Z), int(NOVALUE), n)
    assert _plain(mine) == theirs


def test_topofilter_does_not_leave_a_flat_grid_exactly_alone():
    """Topofilter does not leave a flat grid exactly alone."""
    out = geomorphology.multipass_topofilter(1, FLAT, NOVALUE, 1)
    assert all(v == pytest.approx(100.0, abs=1.0e-12)
               for row in _plain(out) for v in row)
    assert any(v != 100.0 for row in _plain(out) for v in row)


def test_topofilter_keeps_missing_cells_missing():
    out = geomorphology.multipass_topofilter(1, HOLLOW, NOVALUE, 1)
    for r in range(1, len(HOLLOW)):
        for c in range(1, len(HOLLOW[0])):
            assert (out[r][c] == NOVALUE) == (HOLLOW[r][c] == NOVALUE)


# ------------------------------------------------------------------ curvature

@pytest.mark.parametrize("name", sorted(GRIDS))
@pytest.mark.parametrize("dx,dy", [(20.0, 20.0), (10.0, 25.0)])
def test_curvature_matches_the_cxx(name, dx, dy):
    Z = GRIDS[name]
    mine = geomorphology.curvature(dx, dy, Z, NOVALUE)
    theirs = _cxx.curvature(dx, dy, _plain(Z), int(NOVALUE))
    for got, want in zip(mine, theirs):
        assert _plain(got) == want


def test_curvature_of_a_flat_grid_is_zero_everywhere():
    for grid in geomorphology.curvature(20.0, 20.0, FLAT, NOVALUE):
        assert all(v == 0.0 for row in _plain(grid) for v in row)


def test_curvature_of_a_plane_is_zero_everywhere():
    """A tilted plane has no curvature either -- a stencil that got the
    second difference wrong would show it here and not on the flat grid."""
    plane = _grid([[100.0 + 3.0 * r + 7.0 * c for c in range(6)]
                   for r in range(6)])
    for grid in geomorphology.curvature(20.0, 20.0, plane, NOVALUE):
        assert all(abs(v) < 1.0e-12 for row in _plain(grid) for v in row)


def test_curvature_edge_cells_are_zero_not_one_sided():
    """A cell missing one of its two opposite neighbours gets 0, and that is
    not the same as leaving it out: the interior of the same ridge is not 0."""
    c1, c2, _c3, _c4 = geomorphology.curvature(20.0, 20.0, RIDGE, NOVALUE)
    nr = len(RIDGE) - 1
    nc = len(RIDGE[0]) - 1
    for c in range(1, nc + 1):
        assert c1[1][c] == 0.0 and c1[nr][c] == 0.0
    for r in range(1, nr + 1):
        assert c2[r][1] == 0.0 and c2[r][nc] == 0.0
    assert c2[2][3] != 0.0


def test_curvature_marks_missing_cells_undefined():
    grids = geomorphology.curvature(20.0, 20.0, HOLLOW, NOVALUE)
    for grid in grids:
        for r in range(1, len(HOLLOW)):
            for c in range(1, len(HOLLOW[0])):
                if HOLLOW[r][c] == NOVALUE:
                    assert grid[r][c] == NOVALUE


# --------------------------------------------------------------- row and col

@pytest.mark.parametrize("N", [4784371.0, 4784371.5, 4789643.0, 4790000.0,
                               4797051.0, 4784370.0, 4797052.0])
def test_row_matches_the_cxx(N):
    assert geomorphology.row(N, 634, 20.0, 4784371.0, NOVALUE) \
        == _cxx.row(N, 634, 20.0, 4784371.0)


@pytest.mark.parametrize("E", [265786.5625, 265790.0, 272533.0, 278366.5625,
                               265786.0, 278367.0])
def test_col_matches_the_cxx(E):
    assert geomorphology.col(E, 629, 20.0, 265786.5625, NOVALUE) \
        == _cxx.col(E, 629, 20.0, 265786.5625)


def test_row_counts_from_the_north_and_col_from_the_west():
    """The two are not symmetric: row 1 is the *last* cell northward, so it
    counts down from nrows, while column 1 counts up from the west corner."""
    assert geomorphology.row(4784371.0 + 634 * 20.0, 634, 20.0, 4784371.0,
                             NOVALUE) == 1
    assert geomorphology.row(4784371.0 + 0.5 * 20.0, 634, 20.0, 4784371.0,
                             NOVALUE) == 634
    assert geomorphology.col(265786.5625 + 0.5 * 20.0, 629, 20.0, 265786.5625,
                             NOVALUE) == 1
    assert geomorphology.col(265786.5625 + 629 * 20.0, 629, 20.0, 265786.5625,
                             NOVALUE) == 629


# -------------------------------------------------------------- the reader

@needs_reference
@pytest.mark.parametrize("case,stem", REFERENCE_MAPS,
                         ids=[f"{c}-{os.path.basename(s)}"
                              for c, s in REFERENCE_MAPS])
def test_read_map_matches_the_cxx(case, stem):
    path = os.path.join(REFERENCE_1D, case, stem)
    mine = rastermap.read_map(path, NOVALUE)
    recorded = oracle_replay.recorded_digest(_cxx, "read_map", path, NOVALUE)
    if recorded is not None:
        # replayed offline: too large to record, but the digest pins every byte
        # of the C++ answer -- header (Dy, Dx, Y0, X0, nrows, ncols, sign of the
        # novalue, novalue) and the cells row by row
        header = (mine.dy, mine.dx, mine.Y0, mine.X0, float(mine.nrows),
                  float(mine.ncols), -1.0 if NOVALUE < 0 else 1.0, NOVALUE)
        cells = [mine.data[r][1:] for r in range(1, mine.nrows + 1)]
        assert oracle_replay.digest((header, cells)) == recorded
        return
    header, cells = _cxx.read_map(path, NOVALUE)

    assert (mine.dy, mine.dx, mine.Y0, mine.X0) == header[:4]
    assert (mine.nrows, mine.ncols) == (int(header[4]), int(header[5]))
    for r in range(1, mine.nrows + 1):
        assert mine.data[r][1:] == cells[r - 1]


@needs_reference
def test_the_reference_maps_are_not_all_novalue():
    """Guard against the cell comparison passing over grids that are empty:
    the maps must actually carry values, in both projections."""
    for case, stem in (("Bro", "input_maps/dem"),
                       ("Calabria", "input_maps/dem_mybasin_2")):
        M = rastermap.read_map(os.path.join(REFERENCE_1D, case, stem), NOVALUE)
        real = [M.data[r][c] for r in range(1, M.nrows + 1)
                for c in range(1, M.ncols + 1) if M.data[r][c] != NOVALUE]
        assert len(real) > 1000
        assert max(real) > min(real)


@needs_reference
def test_read_map_rejects_a_map_off_the_reference_grid():
    dem = rastermap.read_map(
        os.path.join(REFERENCE_1D, "Bro", "input_maps", "dem"), NOVALUE)
    with pytest.raises(rastermap.RasterError):
        rastermap.read_map(
            os.path.join(REFERENCE_1D, "Calabria", "input_maps",
                         "dem_mybasin_2"), NOVALUE, dem)


def test_the_reader_does_not_go_through_float(tmp_path):
    """Digits are accumulated against ``pow(10, k)``, which is not always the
    same double ``float()`` produces -- and the difference is a real one, not
    a rounding of the printed form."""
    path = tmp_path / "g.asc"
    path.write_text("ncols 2\nnrows 1\nxllcorner 0\nyllcorner 0\n"
                    "cellsize 1\nNODATA_value -9999\n0.29 8.31 \n")
    header, cells = rastermap.read_esriascii(str(path), NOVALUE)
    _oh, ocells = _cxx.read_map(str(path)[:-4], NOVALUE)
    assert cells == [ocells[0][0], ocells[0][1]]
    assert cells[0] != float("0.29") or cells[1] != float("8.31")


def test_a_row_may_end_in_a_blank(tmp_path):
    """Both reference cases' grids do, and the cell stream is read by count,
    not by line: a trailing blank must not shift the grid by a row."""
    path = tmp_path / "g.asc"
    path.write_text("ncols 3\nnrows 2\nxllcorner 0\nyllcorner 0\n"
                    "cellsize 10\nNODATA_value -9999\n"
                    "1 2 3 \n4 5 6 \n")
    M = rastermap.read_map(str(path)[:-4], NOVALUE)
    assert M.data[1][1:] == [1.0, 2.0, 3.0]
    assert M.data[2][1:] == [4.0, 5.0, 6.0]


def test_a_short_row_is_refused(tmp_path):
    """A short row is refused."""
    path = tmp_path / "g.asc"
    path.write_text("ncols 4\nnrows 2\nxllcorner 0\nyllcorner 0\n"
                    "cellsize 10\nNODATA_value -9999\n"
                    "1 2 \n3 4 \n5 6 \n7 8 \n")
    with pytest.raises(rastermap.RasterError):
        rastermap.read_map(str(path)[:-4], NOVALUE)


def test_nodata_cells_become_the_callers_novalue(tmp_path):
    path = tmp_path / "g.asc"
    path.write_text("ncols 2\nnrows 1\nxllcorner 0\nyllcorner 0\n"
                    "cellsize 1\nNODATA_value -1.5\n-1.5 2\n")
    M = rastermap.read_map(str(path)[:-4], NOVALUE)
    assert M.data[1][1] == NOVALUE
    assert M.data[1][2] == 2.0


def test_a_grid_with_no_nodata_line_reads(tmp_path):
    path = tmp_path / "g.asc"
    path.write_text("ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\n"
                    "cellsize 1\n1 2\n3 4\n")
    M = rastermap.read_map(str(path)[:-4], NOVALUE)
    assert M.data[1][1:] == [1.0, 2.0]
    assert M.data[2][1:] == [3.0, 4.0]


def test_a_centred_header_is_refused(tmp_path):
    """xllcenter shifts the whole grid by half a cell; GEOtop refuses it
    rather than reading it as a corner."""
    path = tmp_path / "g.asc"
    path.write_text("ncols 1\nnrows 1\nxllcenter 0\nyllcenter 0\n"
                    "cellsize 1\nNODATA_value -9999\n1\n")
    with pytest.raises(rastermap.RasterError):
        rastermap.read_esriascii(str(path), NOVALUE)


def test_a_missing_map_is_an_error_not_an_empty_grid(tmp_path):
    assert rastermap.existing_file(str(tmp_path / "absent")) == 0
    with pytest.raises(rastermap.RasterError):
        rastermap.read_map(str(tmp_path / "absent"), NOVALUE)


def test_uniform_like_keeps_the_references_missing_cells():
    dem = rastermap.RasterMap(data=HOLLOW, nrows=len(HOLLOW) - 1,
                              ncols=len(HOLLOW[0]) - 1, dx=10.0, dy=10.0,
                              X0=0.0, Y0=0.0, novalue=NOVALUE)
    LU = rastermap.uniform_like(1.0, dem)
    for r in range(1, dem.nrows + 1):
        for c in range(1, dem.ncols + 1):
            assert LU.data[r][c] == (NOVALUE if HOLLOW[r][c] == NOVALUE else 1.0)


# ------------------------------------------------- the whole point resolution

@needs_reference
@pytest.mark.parametrize("case,X,Y,expected", [
    # The topography GEOtop itself resolved for these two points, as its own
    # run log reports it.
    ("Bro", 272533.0, 4789643.0,
     (326.0, 11.048352, 219.805571, 0.853552, 0.01, 0.0075, 0.0075, 0.0125)),
    ("Calabria", 609882.5, 4345618.0,
     (486.7422, 32.571644, 23.769, 0.872554,
      0.000004, -0.003072, -0.00744, 0.003164)),
])
def test_the_reference_points_topography(case, X, Y, expected):
    """End to end over the real maps: read, smooth, curve, look the cell up."""
    directory = os.path.join(REFERENCE_1D, case)
    dem_stem = {"Bro": "input_maps/dem",
                "Calabria": "input_maps/dem_mybasin_2"}[case]
    slope_stem = {"Bro": "input_maps/slope",
                  "Calabria": "input_maps/gradient_grad_mybasin_2"}[case]
    aspect_stem = {"Bro": "input_maps/aspect",
                   "Calabria": "aspect_mybasin_2"}[case]
    sky_stem = {"Bro": "input_maps/SkyViewFactorMapFile",
                "Calabria": "input_maps/skyview_mybasin_2"}[case]

    Z = rastermap.read_map(os.path.join(directory, dem_stem), NOVALUE)
    r = geomorphology.row(Y, Z.nrows, Z.dy, Z.Y0, NOVALUE)
    c = geomorphology.col(X, Z.ncols, Z.dx, Z.X0, NOVALUE)

    got = [Z.data[r][c]]
    for stem in (slope_stem, aspect_stem, sky_stem):
        M = rastermap.read_map(os.path.join(directory, stem), NOVALUE, Z)
        got.append(M.data[r][c])
    for grid in geomorphology.curvature(Z.dy, Z.dx, Z.data, NOVALUE):
        got.append(grid[r][c])

    for value, want in zip(got, expected):
        assert value == pytest.approx(want, abs=5.0e-7)


def test_diagonal_spacing_is_the_geometric_mean():
    """On a rectangular cell the two diagonals use sqrt(dx*dy), not the true
    diagonal length -- reproduced, not corrected."""
    Z = _grid([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    dx, dy = 10.0, 40.0
    _c1, _c2, c3, _c4 = geomorphology.curvature(dx, dy, Z, NOVALUE)
    assert c3[2][2] == pytest.approx(2.0 / math.pow(math.sqrt(dx * dy), 2.0))
