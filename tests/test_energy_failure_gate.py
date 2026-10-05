"""What a trial keeps when a point's energy balance fails, driven through the
pipeline.

None of the 13 reference cases ever fails an energy solve, so this path is
forced: the solve verdict of one chosen (trial, point) is replaced by
GEOtop's ``sux = 1``, and everything downstream runs as it would. Checked
against ``energy.balance.cc`` and ``geotop.cc``: a failed point skips
everything after its solve, the points after it are not solved at all, the
water balance does not run in that trial (and, when it does, stops at the
first point whose Richards solve fails), and at ``min_Dt`` the failed trial
is committed anyway while the clock advances by the halved ``Dt``. The point
output of a step that did not complete repeats the values of the last
accepted one, weighted by that one's ``Dt``.
"""

import os
import shutil

import pytest

from geotop_py import constants as C
from geotop_py import pipeline
from geotop_py.energy.column import SoilLayer
from geotop_py.energy.surface import SurfaceState, SurfaceStatics
from geotop_py.io import gt_output
from geotop_py.meteo.step import Meteo
from geotop_py.point import step as point_step
from geotop_py.point.state import Column1D
from geotop_py.snow.state import SnowColumn
from tools.paths import REFERENCE_1D

needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D), reason=f"reference cases not found under {REFERENCE_1D}")

CASE = "PureDrainage"      # WaterBalance=1, bare soil, hourly steps and rows


class _Stop(Exception):
    """Ends the run once the trials under test have been seen."""


def _case(tmp_path, npoints=1):
    """A copy of ``CASE`` with ``npoints`` identical points."""
    sim_dir = tmp_path / CASE
    shutil.copytree(os.path.join(REFERENCE_1D, CASE), sim_dir,
                    ignore=shutil.ignore_patterns("output-tabs", "output-prof", "*.log"))
    if npoints > 1:
        lp = sim_dir / "listpoints.txt"
        header, row = lp.read_text().splitlines()[:2]
        rest = row.split(",", 1)[1]
        lp.write_text("\n".join([header] + [f"{i},{rest}" for i in range(1, npoints + 1)])
                      + "\n")
    return sim_dir


def _soil(col):
    return dict(T=list(col.soil_T), thi=[s.thi0 for s in col.soil[1:]],
                P=[s.P0 for s in col.soil[1:]], lnum=col.snow.lnum,
                swe=col.snow.active_swe())


class Recorder:
    """Wraps the three phases of a point step, numbering the trials.

    ``fail_energy(trial, point)`` / ``fail_water(trial, point)`` choose which
    solves fail; ``stop_after`` ends the run at the first call of that trial.
    """

    def __init__(self, monkeypatch, fail_energy=lambda n, p: False,
                 fail_water=lambda n, p: False, stop_after=None):
        self.calls = []
        self.trial = -1
        self._points = {}
        self._cols = {}
        self._forcing = False
        real_energy = point_step.energy_phase
        real_post = point_step.post_energy_phase
        real_water = point_step.water_phase
        real_verdict = point_step._sux_verdict

        def verdict(*args, **kw):
            if self._forcing:
                return 1, None
            return real_verdict(*args, **kw)

        def energy(col, state, st, Dt, m, JDb, JDe):
            point = self._points.setdefault(id(st), len(self._points) + 1)
            if point == 1:
                self.trial += 1
                if stop_after is not None and self.trial > stop_after:
                    raise _Stop
            self._cols[id(col)] = point
            self._forcing = fail_energy(self.trial, point)
            before = _soil(col)
            ep = real_energy(col, state, st, Dt, m, JDb, JDe)
            self._forcing = False
            self.calls.append(dict(kind="energy", trial=self.trial, point=point,
                                   Dt=Dt, JDb=JDb, Ta=m.Ta, before=before,
                                   col=col, ok=ep.converged))
            return ep

        def post(col, state, Dt, m, ep, richards=None):
            self.calls.append(dict(kind="post", trial=self.trial,
                                   point=self._cols[id(col)]))
            return real_post(col, state, Dt, m, ep, richards)

        def water(col, richards, Dt, out, water_input):
            point = self._cols[id(col)]
            real_water(col, richards, Dt, out, water_input)
            if fail_water(self.trial, point):
                out.wb_converged = False
            self.calls.append(dict(kind="water", trial=self.trial, point=point,
                                   col=col))

        monkeypatch.setattr(point_step, "_sux_verdict", verdict)
        monkeypatch.setattr(point_step, "energy_phase", energy)
        monkeypatch.setattr(point_step, "post_energy_phase", post)
        monkeypatch.setattr(point_step, "water_phase", water)

    def of(self, kind, trial):
        return [c for c in self.calls if c["kind"] == kind and c["trial"] == trial]


def _run(sim_dir, rec, min_Dt=60.0, stop=True):
    if stop:
        with pytest.raises(_Stop):
            pipeline.run_simulation(str(sim_dir), suffix="_py", verbose=False,
                                    min_Dt=min_Dt)
        return None
    return pipeline.run_simulation(str(sim_dir), suffix="_py", verbose=False,
                                   min_Dt=min_Dt)


@needs_reference
@pytest.mark.parametrize("failing", [1, 2, 3])
def test_a_failed_energy_balance_ends_the_trial_before_the_water_balance(
        tmp_path, monkeypatch, failing):
    rec = Recorder(monkeypatch, fail_energy=lambda n, p: n == 3 and p == failing,
                   stop_after=4)
    _run(_case(tmp_path, npoints=3), rec)
    assert [c["point"] for c in rec.of("energy", 3)] == list(range(1, failing + 1))
    assert [c["point"] for c in rec.of("post", 3)] == list(range(1, failing))
    assert rec.of("water", 3) == []
    # the retry at half Dt solves every point, water balance included
    assert [c["point"] for c in rec.of("water", 4)] == [1, 2, 3]
    assert rec.of("energy", 4)[0]["Dt"] == rec.of("energy", 3)[0]["Dt"] / 2


@needs_reference
@pytest.mark.parametrize("failing", [1, 2, 3])
def test_at_min_dt_the_failed_trial_is_committed_as_geotop_keeps_it(
        tmp_path, monkeypatch, failing):
    # Dt=3600 fails, halves to min_Dt=1800 and gives up: trial 3 is committed
    # and the clock advances by 1800 s, so trial 4 runs the second half.
    rec = Recorder(monkeypatch, fail_energy=lambda n, p: n == 3 and p == failing,
                   stop_after=4)
    _run(_case(tmp_path, npoints=3), rec, min_Dt=1800.0)
    failed = {c["point"]: c for c in rec.of("energy", 3)}
    after = {c["point"]: c for c in rec.of("energy", 4)}
    assert rec.of("water", 3) == []
    first = rec.of("energy", 3)[0]
    assert after[1]["Dt"] == 1800.0 == first["Dt"] / 2
    assert after[1]["JDb"] == pytest.approx(first["JDb"] + 1800.0 / 86400.0, abs=1e-12)
    committed_2 = {c["point"]: c["col"] for c in rec.of("energy", 2)}
    for p in (1, 2, 3):
        start = after[p]["before"]
        if p < failing:
            # an accepted energy step is committed with its soil update
            assert start == _soil(failed[p]["col"])
            assert start["T"] != failed[p]["before"]["T"]
        else:
            # the failed point keeps what it started from (no update_soil_land,
            # no WBsnow), and the points after it were never solved
            assert start == _soil(committed_2[p])


@needs_reference
def test_a_water_failure_is_retried_and_committed_at_min_dt(tmp_path, monkeypatch):
    rec = Recorder(monkeypatch, fail_water=lambda n, p: n == 3, stop_after=4)
    _run(_case(tmp_path), rec)
    # retried: the half-Dt trial starts from what trial 3 started from
    assert rec.of("energy", 4)[0]["Dt"] == 1800.0
    assert rec.of("energy", 4)[0]["before"] == rec.of("energy", 3)[0]["before"]

    rec = Recorder(monkeypatch, fail_water=lambda n, p: n == 3, stop_after=4)
    _run(_case(tmp_path / "min"), rec, min_Dt=1800.0)
    # given up: both the energy and the water trial state are committed
    assert rec.of("energy", 4)[0]["Dt"] == 1800.0
    assert rec.of("energy", 4)[0]["before"] == _soil(rec.of("water", 3)[0]["col"])
    assert rec.of("energy", 4)[0]["before"] != rec.of("energy", 3)[0]["before"]


@needs_reference
@pytest.mark.parametrize("failing", [1, 2, 3])
def test_the_first_failing_richards_solve_ends_the_water_balance(
        tmp_path, monkeypatch, failing):
    # GEOtop's point water balance returns at the first column whose Richards
    # solve fails: the columns after it are not solved in that trial.
    # GEOtop: src/geotop/water.balance.cc:175-181
    rec = Recorder(monkeypatch, fail_water=lambda n, p: n == 3 and p == failing,
                   stop_after=4)
    _run(_case(tmp_path, npoints=3), rec)
    assert [c["point"] for c in rec.of("post", 3)] == [1, 2, 3]
    assert [c["point"] for c in rec.of("water", 3)] == list(range(1, failing + 1))
    # the retry at half Dt solves every point's water balance again
    assert [c["point"] for c in rec.of("water", 4)] == [1, 2, 3]
    assert rec.of("energy", 4)[0]["Dt"] == rec.of("energy", 3)[0]["Dt"] / 2


@needs_reference
def test_a_failed_step_repeats_the_last_accepted_output(tmp_path, monkeypatch):
    # Trial 3 fails and is committed at min_Dt; trial 4 runs the second half
    # hour. GEOtop folds into the hour's row the output of trial 2 -- the last
    # accepted energy step -- weighted by *its* Dt, plus that of trial 4: the
    # weights sum to 1.5.
    rec = Recorder(monkeypatch, fail_energy=lambda n, p: n == 3)
    sim_dir = _case(tmp_path)
    _run(sim_dir, rec, min_Dt=1800.0, stop=False)
    Ta = {c["trial"]: c["Ta"] for c in rec.calls if c["kind"] == "energy"}
    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "point0001.txt"))
    tair = py.col("Tair[C]")
    assert [c["Dt"] for c in rec.calls if c["kind"] == "energy"][:6] == \
        [3600.0] * 4 + [1800.0, 3600.0]
    assert tair[2] == pytest.approx(Ta[2], abs=1e-5)
    assert tair[3] == pytest.approx(Ta[2] * 3600.0 / 3600.0 + Ta[4] * 1800.0 / 3600.0,
                                    abs=1e-5)
    assert tair[4] == pytest.approx(Ta[5], abs=1e-5)


# -- the -6 retry, on a single column ----------------------------------------

def _one_snow_layer_column():
    soil = [None] + [SoilLayer(sat=0.4, res=0.05, alpha=0.004, n=1.4, ss=1e-3,
                               kt=2.5, ct=2.3e6, th0=0.3, thi0=0.05, Tstar=0.0)
                     for _ in range(3)]
    snow = SnowColumn.from_layers([(20.0, 5.0, 0.5, -0.5)], max=5)
    return Column1D(
        snow=snow, soil=soil, soil_D=[0.0] + [0.2] * 3, soil_T=[0.0] + [1.0] * 3,
        alpha_snow=1e5, snow_conductivity=3, slope=0.0,
        Tboundary=1.0, Zboundary=1.0, Fboundary=0.0,
        max_weq_snow=5.0, maxSWE=1e10,
    )


def test_a_failed_minus6_retry_keeps_the_folded_ice_in_the_soil(monkeypatch):
    """GEOtop's -6 branch writes the melting snow layer's ice into the top soil
    layer before re-solving the bare column; when that solve fails the soil
    keeps it and the snow layer stays as it was."""
    real_solve = point_step.solve_column_energy
    solves = []

    def solve(ecol, *a, **kw):
        res = real_solve(ecol, *a, **kw)
        solves.append(ecol)
        if len(solves) == 2:
            res.converged = False
        return res

    monkeypatch.setattr(point_step, "solve_column_energy", solve)
    monkeypatch.setattr(point_step, "_sux_verdict",
                        lambda res, ecol, ns, ng, sur: (-6, 1))
    col = _one_snow_layer_column()
    snow_before = (col.snow.lnum, list(col.snow.w_ice), list(col.snow.Dzl))
    st = SurfaceStatics(lat=46.1, lon=11.1, ST=1.0, sky=1.0, soil_D_mm=[200.0] * 3)
    m = Meteo(Ta=5.0, RH=0.7, P=850.0, wind=2.0, tau_cloud=1.0)
    ep = point_step.energy_phase(col, SurfaceState(), st, 3600.0, m, 100.0,
                                 100.0 + 1.0 / 24.0)
    assert not ep.converged
    assert len(solves) == 2 and solves[1].nsng == 0
    bare = solves[1]
    folded = bare.ice[1] / (C.rho_w * bare.Dlayer[1])
    assert col.soil[1].thi0 == folded
    assert col.soil[1].thi0 > 0.05
    assert (col.snow.lnum, col.snow.w_ice, col.snow.Dzl) == snow_before
