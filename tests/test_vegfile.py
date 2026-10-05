"""Time-dependent vegetation parameters: the reader, and how a step applies it.

The reader (``io.table.read_txt_matrix_2``) is pinned against the oracle on the
real ``ARF_1D`` file, because reading by *position* has no header to catch a
mistake: a column read one slot over, or a timestamp collapsed differently by
the tokenizer, produces plausible numbers that silently drive the whole canopy.

The interpolation itself is not re-pinned here -- it is
``meteodata.time_interp_linear``, already checked against the oracle in
``tests/test_meteodata.py``; what is checked here is that a vegetation series
is fed to it the way GEOtop feeds it (date column 0, ``DDMMYYYYhhmm`` flag) and
that the result is mapped onto the right nine parameters.
"""

import os

import pytest

from geotop_py.energy import vegetation as veg
from geotop_py.io import table, vegfile
from geotop_py.io.parfile import NUMBER_NOVALUE
from geotop_py.meteo import meteodata
from tools import oracle_replay
from tools.paths import REFERENCE_1D

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    _cxx = None
    _cxx_reason = str(exc)

VEG_FILE = os.path.join(REFERENCE_1D, "ARF_1D", "veg", "VegParams0001.txt")
needs_reference = pytest.mark.skipif(
    not os.path.isfile(VEG_FILE), reason=f"reference case not found at {VEG_FILE}")
needs_oracle = pytest.mark.skipif(_cxx is None, reason="oracle not built")


@needs_reference
def test_load_reads_the_file_by_position():
    rows = vegfile.load(os.path.join(REFERENCE_1D, "ARF_1D", "veg", "VegParams"), 1)
    assert rows is not None
    assert len(rows) == 6209
    assert all(len(r) == vegfile.NCOLS for r in rows)
    # 01/01/1999 13:00,200,50,10,0.05,0.0246900879716674,2.5,1,500,30 -- the
    # timestamp's punctuation is junk to the tokenizer, so the whole field
    # collapses into the single DDMMYYYYhhmm number the interpolator expects.
    assert rows[0][vegfile.COL_DATE] == 10119991300.0
    assert rows[0][1:] == [200.0, 50.0, 10.0, 0.05, 0.0246900879716674,
                            2.5, 1.0, 500.0, 30.0]


@needs_reference
def test_load_returns_none_for_a_class_with_no_file():
    assert vegfile.load(os.path.join(REFERENCE_1D, "ARF_1D", "veg", "VegParams"), 7) is None


@needs_reference
@needs_oracle
def test_read_txt_matrix_2_matches_the_cxx():
    mine = table.read_txt_matrix_2(VEG_FILE, vegfile.NCOLS)
    recorded = oracle_replay.recorded_digest(_cxx, "read_txt_matrix_2", VEG_FILE,
                                             vegfile.NCOLS)
    if recorded is not None:
        # replayed offline: too large to record, but the digest pins every byte
        assert oracle_replay.digest(mine) == recorded
        return
    theirs = _cxx.read_txt_matrix_2(VEG_FILE, vegfile.NCOLS)
    assert len(mine) == len(theirs)
    assert mine == theirs


@needs_oracle
def test_read_txt_matrix_2_pads_and_truncates_like_the_cxx(tmp_path):
    """A line shorter than the requested slot count is padded with
    ``number_novalue``; a longer one is cut. Both branches, against the oracle."""
    path = tmp_path / "ragged.txt"
    path.write_text("a,b,c,d\n"
                    "1,2\n"
                    "3,4,5,6,7\n"
                    "8,9,10,11\n")
    mine = table.read_txt_matrix_2(str(path), 4)
    theirs = _cxx.read_txt_matrix_2(str(path), 4)
    assert mine == theirs
    assert mine[0] == [1.0, 2.0, NUMBER_NOVALUE, NUMBER_NOVALUE]
    assert mine[1] == [3.0, 4.0, 5.0, 6.0]


def _static() -> veg.VegParams:
    return veg.VegParams(Hveg=1000.0, z0thresveg=1000.0, z0thresveg2=1000.0,
                         LSAI=1.0, cf=0.0, decay0=2.5, expveg=1.0,
                         root=300.0, rs=60.0, n_transp=2,
                         root_frac=[0.0, 0.0, 0.0])


DZ = [0.0, 100.0, 400.0]


def test_apply_time_dependent_overwrites_the_nine_fields():
    live, static = _static(), _static()
    values = [10119991300.0, 200.0, 50.0, 10.0, 0.05, 0.02469, 2.5, 1.0, 500.0, 30.0]
    veg.apply_time_dependent(live, static, values, DZ)
    assert (live.Hveg, live.z0thresveg, live.z0thresveg2) == (200.0, 50.0, 10.0)
    assert (live.LSAI, live.cf, live.decay0) == (0.05, 0.02469, 2.5)
    assert (live.expveg, live.root, live.rs) == (1.0, 500.0, 30.0)
    # RootDepth came from the series, so the root fraction follows it.
    assert live.root_frac == veg.root_fraction(3, 500.0, 0.0, DZ)


def test_apply_time_dependent_falls_back_to_the_static_keywords():
    """Outside the series' own time span the interpolator returns novalue for
    every column at once, and the per-class keywords take over."""
    live, static = _static(), _static()
    live.LSAI = 99.0
    veg.apply_time_dependent(live, static, [NUMBER_NOVALUE] * vegfile.NCOLS, DZ)
    assert live.LSAI == static.LSAI
    assert live.root == static.root


def test_apply_time_dependent_leaves_a_stale_root_fraction_on_fallback():
    """The root fraction is recomputed only when the series supplies a root
    depth: a fallback step keeps the last one computed, not one consistent with
    the static root depth now in use. Reproduced deliberately."""
    live, static = _static(), _static()
    values = [10119991300.0] + [NUMBER_NOVALUE] * 7 + [500.0, NUMBER_NOVALUE]
    veg.apply_time_dependent(live, static, values, DZ)
    from_series = list(live.root_frac)
    assert from_series == veg.root_fraction(3, 500.0, 0.0, DZ)

    veg.apply_time_dependent(live, static, [NUMBER_NOVALUE] * vegfile.NCOLS, DZ)
    assert live.root == static.root == 300.0
    assert live.root_frac == from_series
    assert live.root_frac != veg.root_fraction(3, 300.0, 0.0, DZ)


@needs_reference
def test_series_interpolates_between_two_daily_samples():
    """The series is a piecewise-linear signal averaged over the step, not a
    step function holding the most recent row: a window between two samples
    with different values lands strictly between them."""
    rows = vegfile.load(os.path.join(REFERENCE_1D, "ARF_1D", "veg", "VegParams"), 1)
    # Two consecutive samples whose LSAI differs, and a step inside them.
    i = next(k for k in range(len(rows) - 1)
             if rows[k][4] != rows[k + 1][4])
    t0 = meteodata.time_in_JDfrom0(0, i, vegfile.COL_DATE, rows)
    t1 = meteodata.time_in_JDfrom0(0, i + 1, vegfile.COL_DATE, rows)
    mid = 0.5 * (t0 + t1)
    out, _ = meteodata.time_interp_linear(t0, mid, mid + 1.0 / 24.0, rows,
                                          vegfile.COL_DATE, 0, 0)
    lo, hi = sorted((rows[i][4], rows[i + 1][4]))
    assert lo < out[4] < hi
