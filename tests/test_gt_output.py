"""Reading GEOtop's tabular outputs back, repeated column names included."""

import csv
import os

import pytest

from geotop_py.io import gt_output
from tools.paths import REFERENCE_1D

needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D), reason=f"reference cases not found under {REFERENCE_1D}")


def test_repeated_names_get_pandas_suffixes():
    assert gt_output.unique_names(["a", "b", "a", "a"]) == ["a", "b", "a.1", "a.2"]
    # a suffix that is already a real name is skipped
    assert gt_output.unique_names(["a", "a.1", "a"]) == ["a", "a.1", "a.2"]


def test_a_repeated_column_keeps_its_own_values(tmp_path):
    path = tmp_path / "basin.txt"
    path.write_text("Date12[DDMMYYYYhhmm],x,P,P\n"
                    "01/01/2000 01:00,1,2,3\n"
                    "01/01/2000 02:00,4,5,-9999\n")
    t = gt_output.read_point(str(path))
    assert t.header == ["Date12[DDMMYYYYhhmm]", "x", "P", "P"]
    assert t.col("P") == [2.0, 5.0]
    assert t.col("P.1")[0] == 3.0
    assert t.col("P.1")[1] != t.col("P.1")[1]          # -9999 is NaN
    assert all(len(v) == len(t) for v in t.columns.values())


@needs_reference
def test_basin_file_reads_every_column():
    path = os.path.join(REFERENCE_1D, "Matsch_P2_Ref_007", "output-tabs-SE27XX", "basin.txt")
    t = gt_output.read_point(path)
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(t.columns) == len(rows[0]) - 1
    assert "Prain_above_canopy[mm].1" in t.columns
    for j, name in enumerate(t.columns, start=1):
        assert len(t.col(name)) == len(rows) - 1
        assert t.col(name)[0] == pytest.approx(float(rows[1][j]))
