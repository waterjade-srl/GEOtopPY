"""A halved nominal step, driven end to end through the pipeline.

None of the 13 reference cases ever halves Dt with the GEOtop solver, so the
reference suite cannot see how sub-steps are chained; a failure is forced here
instead, by failing the energy solve of one trial of a real case. Checked against
``geotop.cc``'s run loop: each sub-step starts from the state the previous one
committed, the station record is interpolated over each trial's own window,
and every point advances with the same Dt -- the first point that fails ends
the trial and halves Dt for the whole domain.
"""

import os
import shutil

import pytest

from geotop_py import pipeline
from geotop_py.point import step as point_step
from tools.paths import REFERENCE_1D

needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D), reason=f"reference cases not found under {REFERENCE_1D}")


class _Stop(Exception):
    """Ends the run once the steps under test have been seen."""


def _run(case, tmp_path, monkeypatch, force_fail, stop_after):
    """Run ``case`` recording every point solve; ``force_fail(n, point)``
    makes solve ``n`` fail (GEOtop's ``sux = 1``). Returns the list of
    recorded solves."""
    sim_dir = tmp_path / case
    shutil.copytree(os.path.join(REFERENCE_1D, case), sim_dir,
                    ignore=shutil.ignore_patterns("output-tabs", "output-prof", "*.log"))
    real = point_step.energy_phase
    real_verdict = point_step._sux_verdict
    points = {}
    calls = []
    forcing = [False]

    def verdict(*args, **kw):
        return (1, None) if forcing[0] else real_verdict(*args, **kw)

    def recorded(col, state, st, Dt, m, JDb, JDe):
        if len(calls) >= stop_after:
            raise _Stop
        point = points.setdefault(id(st), len(points) + 1)
        forcing[0] = force_fail(len(calls), point)
        calls.append(dict(point=point, Dt=Dt, JDb=JDb, JDe=JDe, Ta=m.Ta,
                          before=list(col.soil_T), col=col))
        try:
            return real(col, state, st, Dt, m, JDb, JDe)
        finally:
            forcing[0] = False

    monkeypatch.setattr(point_step, "_sux_verdict", verdict)
    monkeypatch.setattr(point_step, "energy_phase", recorded)
    with pytest.raises(_Stop):
        pipeline.run_simulation(str(sim_dir), suffix="_py", verbose=False)
    # A trial column is never written again once its trial ends, so reading
    # it now gives the state that trial left.
    for c in calls:
        c["after"] = list(c.pop("col").soil_T)
    return calls


@needs_reference
def test_the_second_half_starts_where_the_first_ended(tmp_path, monkeypatch):
    calls = _run("PureDrainage", tmp_path, monkeypatch,
                 force_fail=lambda n, point: n == 5, stop_after=8)
    full, first, second = calls[5], calls[6], calls[7]
    assert first["Dt"] == second["Dt"] == full["Dt"] / 2
    assert first["before"] == full["before"]      # the failed trial is discarded
    assert second["before"] == first["after"]     # ...the committed half is not
    assert second["before"] != full["before"]


@needs_reference
def test_each_trial_reads_the_meteo_of_its_own_window(tmp_path, monkeypatch):
    calls = _run("PureDrainage", tmp_path, monkeypatch,
                 force_fail=lambda n, point: n == 5, stop_after=8)
    full, first, second = calls[5], calls[6], calls[7]
    assert (first["JDb"], first["JDe"]) == (full["JDb"], second["JDb"])
    assert second["JDe"] == full["JDe"]
    # a time mean over the full window is the mean of the two halves' means
    assert (first["Ta"] + second["Ta"]) / 2 == pytest.approx(full["Ta"], abs=1e-9)
    assert first["Ta"] != second["Ta"]


@needs_reference
def test_one_failing_point_halves_dt_for_every_point(tmp_path, monkeypatch):
    # nominal step 3 is calls 6 (point 1) and 7 (point 2); point 2 fails
    calls = _run("Jungfraujoch", tmp_path, monkeypatch,
                 force_fail=lambda n, point: n == 7, stop_after=12)
    assert [c["point"] for c in calls[6:12]] == [1, 2, 1, 2, 1, 2]
    p1_full, p1_half = calls[6], calls[8]
    assert p1_half["Dt"] == p1_full["Dt"] / 2
    assert p1_half["before"] == p1_full["before"]  # point 1's full trial is discarded too


@needs_reference
def test_the_first_failing_point_ends_the_trial(tmp_path, monkeypatch):
    # point 1 fails at nominal step 3: point 2 is not solved in that trial
    calls = _run("Jungfraujoch", tmp_path, monkeypatch,
                 force_fail=lambda n, point: n == 6, stop_after=11)
    assert [c["point"] for c in calls[6:11]] == [1, 1, 2, 1, 2]
    assert calls[7]["Dt"] == calls[6]["Dt"] / 2
