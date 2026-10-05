"""The public form of the in-memory results returned by ``run_simulation``."""

import math
import os
from datetime import datetime, timedelta

import pytest

import geotop_py
from geotop_py.results import StepRecord, steps_table
from tools.paths import REFERENCE_1D

needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D), reason=f"reference cases not found under {REFERENCE_1D}")


def test_the_package_exports_the_run_and_its_records():
    assert geotop_py.run_simulation is geotop_py.pipeline.run_simulation
    assert geotop_py.StepRecord is StepRecord
    assert geotop_py.steps_table is steps_table


@needs_reference
def test_records_cover_the_simulated_period_in_order(reference_run):
    _, recs = reference_run("PureDrainage")
    steps = recs[1]
    assert all(isinstance(r, StepRecord) for r in steps)
    # 18/06/2014 19:00 to 07/07/2014 01:00, as in geotop.inpts
    start, end = datetime(2014, 6, 18, 19, 0), datetime(2014, 7, 7, 1, 0)
    assert sum(r.dt for r in steps) == pytest.approx((end - start).total_seconds())
    assert steps[-1].date == end
    t = start
    for r in steps:
        t += timedelta(seconds=r.dt)
        assert abs((r.date - t).total_seconds()) < 1e-3
        assert r.trials >= 1
        assert r.dt <= r.dt_run <= 3600.0


@needs_reference
def test_steps_table_flattens_every_record_with_the_same_columns(reference_run):
    _, recs = reference_run("PureDrainage")
    steps = recs[1]
    table = steps_table(steps)
    assert len(table) == len(steps)
    columns = list(table[0])
    assert all(list(row) == columns for row in table)
    for name in ("date", "dt", "trials", "swe", "depth", "converged",
                 "solver_iterations", "wb_converged", "wb_loss",
                 "diag_SWnet", "diag_LWin", "meteo_Ta", "meteo_Prain"):
        assert name in columns
    for row, r in zip(table, steps):
        assert row["date"] == r.date
        assert row["Tg"] == r.out.Tg
        assert row["diag_SWnet"] == r.out.diag.SWnet
        assert row["meteo_Ta"] == r.meteo.Ta
        assert all(isinstance(v, (int, float, bool, datetime)) for v in row.values())
    assert not any(isinstance(v, float) and math.isnan(v) for v in
                   (row["meteo_Ta"] for row in table))


def test_steps_table_of_no_records_is_empty():
    assert steps_table([]) == []
