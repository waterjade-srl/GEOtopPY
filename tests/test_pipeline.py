"""End-to-end regression checks on the official 1D reference cases.

Exercise meteorological forcing, coupled physics, output selection, reporting
intervals and profile interpolation. Individual assertions retain diagnostic
bounds; tools.dashboard applies the complete 1e-5 acceptance criterion.
"""

import os
from datetime import datetime

import pytest

from geotop_py import pipeline
from geotop_py.io import gt_output
from geotop_py.output import tabs as output_tabs
from tools.paths import REFERENCE_1D

needs_reference = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_1D), reason=f"reference cases not found under {REFERENCE_1D}")


@needs_reference
@pytest.mark.parametrize("case", ["PureDrainage", "PureDrainageRainySlope"])
def test_runs_end_to_end_with_exact_radiative_forcing(case, reference_run):
    sim_dir, recs = reference_run(case)

    assert set(recs.keys()) == {1}
    steps = recs[1]
    # Records are *internal* sub-steps, so this is >= the 438 output rows
    # asserted below: the time loop may halve Dt on a step it cannot converge
    # at the nominal one and commit it in pieces, exactly as GEOtop does.
    assert len(steps) >= 438
    assert all(r.out.converged for r in steps)
    assert all(r.out.wb_converged for r in steps)

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "point0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case, "output-tabs-SE27XX",
                                            "point0001.txt"))
    assert len(py.dates) == len(ref.dates)

    def diffs(col):
        pv, rv = py.col(col), ref.col(col)
        return [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]

    for col in ("Tair[C]", "snow_water_equivalent[mm]", "LWin[W/m2]"):
        d = diffs(col)
        assert d, f"no comparable samples for {col}"
        assert max(d) == pytest.approx(0.0, abs=1.0e-4), f"{col}: max diff {max(d)}"

    # Shortwave is exact on PureDrainage but not quite on
    # PureDrainageRainySlope, where one step's water balance stops just above
    # its tolerance and the time loop commits that step in two halves.  The
    # solar geometry is integrated over each half separately, so the
    # time-weighted mean of the two is not bit-identical to the single
    # full-step value -- a ~0.02 W/m2 artefact of *where* the run subdivides,
    # not of the radiation itself.
    for col in ("SWin[W/m2]", "SWnet[W/m2]"):
        assert max(diffs(col)) < 0.03, f"{col}: max diff {max(diffs(col))}"

    # The surface energy balance: bounds both cases satisfy since the
    # bare-soil evaporation parameters became a function of the Newton trial
    # rather than of the timestep.  PureDrainage itself is
    # bit-exact -- see the dedicated test below; these bounds exist for
    # PureDrainageRainySlope, which is not.
    assert max(diffs("Tsurface[C]")) < 0.05
    assert max(diffs("Soil_heat_flux[W/m2]")) < 0.2
    assert max(diffs("LE[W/m2]")) < 0.35
    assert max(diffs("H[W/m2]")) < 0.2


@needs_reference
def test_pure_drainage_surface_energy_balance_is_bit_exact(reference_run):
    """Pure drainage surface energy balance is bit exact."""
    case = "PureDrainage"
    sim_dir, _ = reference_run(case)

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "point0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                            "output-tabs-SE27XX", "point0001.txt"))
    for col in ("Tsurface[C]", "Surface_Energy_balance[W/m2]",
                "Soil_heat_flux[W/m2]", "H[W/m2]", "LE[W/m2]",
                "LEg_unveg[W/m2]", "Hg_unveg[W/m2]", "Evap_surface[mm]",
                "LWnet[W/m2]", "SWnet[W/m2]", "LObukhov[m]"):
        pv, rv = py.col(col), ref.col(col)
        d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
        assert d, f"no comparable samples for {col}"
        assert max(d) == pytest.approx(0.0, abs=1.0e-4), f"{col}: max diff {max(d)}"


@needs_reference
@pytest.mark.parametrize("case", ["PureDrainageRainy", "PureDrainageRainySlope"])
def test_heavy_rain_soil_water_profile_is_bit_exact(case, reference_run):
    """The two cases with a sustained 50 mm/h rain event: every depth node of
    ``soilpsi``/``thetaliq``/``soiltemp`` matches the reference exactly, over
    the whole 438-step series.

    Two independent defects used to show up here and nowhere else, both only
    while water was actually infiltrating:

    * a point column drains its surface node at the end of every water-balance
      step, down to the free-surface boundary depth -- 0 for these cases, so
      all of it. Without that, the pond survived into the next step,
      re-infiltrated, and eventually saturated the profile: the head it built
      up appeared as a near-uniform ~4000 mm offset on every node's ``psi``.
    * the hydraulic conductivities are frozen at the head the step starts
      from (``UpdateHydraulicConductivity`` defaults to 0). Re-evaluating them
      at each Newton iterate -- the physically tidier choice -- makes a wetting
      layer's conductivity climb inside the step, so the wetting front runs
      1.5 to 3.4 times too fast.

    Both are invisible on the dry cases: the first needs standing water, the
    second needs the conductivity to move appreciably within one step.
    """
    sim_dir, _ = reference_run(case)

    for fn in ("soilpsi0001.txt", "thetaliq0001.txt", "soiltemp0001.txt"):
        py = gt_output.read_point(str(sim_dir / "output-tabs_py" / fn))
        ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                                 "output-tabs-SE27XX", fn))
        assert len(py.dates) == len(ref.dates) == 439
        depth_cols = [c for c in py.columns if c not in
                      ("Date12[DDMMYYYYhhmm]", "JulianDayFromYear0[days]",
                       "TimeFromStart[days]", "Simulation_Period", "Run",
                       "IDpoint")]
        for c in depth_cols:
            pv, rv = py.col(c), ref.col(c)
            d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
            assert d, f"{fn}: no comparable samples for {c}"
            assert max(d) == 0.0, f"{fn} {c}: max diff {max(d)}"

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "point0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                            "output-tabs-SE27XX", "point0001.txt"))
    for col in ("Tsurface[C]", "Surface_Energy_balance[W/m2]",
                "Soil_heat_flux[W/m2]", "H[W/m2]", "LE[W/m2]",
                "Prain_over_canopy[mm]", "SWnet[W/m2]", "LWnet[W/m2]"):
        pv, rv = py.col(col), ref.col(col)
        d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
        assert d, f"no comparable samples for {col}"
        assert max(d) == pytest.approx(0.0, abs=1.0e-4), f"{col}: max diff {max(d)}"


@needs_reference
def test_soiltemp_profile_and_wind_direction_and_glacier_defaults(reference_run):
    """Soiltemp profile and wind direction and glacier defaults."""
    case = "PureDrainage"
    sim_dir, _ = reference_run(case)

    py_prof = gt_output.read_point(str(sim_dir / "output-tabs_py" / "soiltemp0001.txt"))
    ref_prof = gt_output.read_point(os.path.join(REFERENCE_1D, case, "output-tabs-SE27XX",
                                                 "soiltemp0001.txt"))
    assert len(py_prof.dates) == len(ref_prof.dates) == 439
    assert py_prof.dates[0] == datetime(2014, 6, 18, 19, 0)

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "point0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case, "output-tabs-SE27XX",
                                            "point0001.txt"))
    for col in ("Wind_direction[deg]", "glac_density[kg/m3]", "glac_temperature[C]"):
        pv, rv = py.col(col), ref.col(col)
        d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
        assert d, f"no comparable samples for {col}"
        assert max(d) == pytest.approx(0.0, abs=1.0e-4), f"{col}: max diff {max(d)}"


@needs_reference
def test_station_with_no_valid_reading_falls_back_to_base_defaults(reference_run):
    """Station with no valid reading falls back to base defaults."""
    case = "PureDrainageFaked"
    sim_dir, recs = reference_run(case)
    steps = recs[1]
    assert all(r.out.converged for r in steps)

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "point0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case, "output-tabs-SE27XX",
                                            "point0001.txt"))
    assert len(py.dates) == len(ref.dates)

    for col in ("Tair[C]", "Relative_Humidity[-]", "Wind_speed[m/s]"):
        pv, rv = py.col(col), ref.col(col)
        d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
        assert d, f"no comparable samples for {col}"
        assert max(d) == pytest.approx(0.0, abs=1.0e-4), f"{col}: max diff {max(d)}"


@needs_reference
def test_coldelaporte_matches_with_measured_lwin_and_station_coordinates(reference_run):
    """``ColdelaPorte`` exposed two real bugs at once:

    1. A measured incoming-longwave column (``HeaderLWin="LW"``). GEOtop
       prefers it outright over the cloud-derived estimate whenever it is
       defined (``energy.balance.cc:606``, ``flux()``) -- ``io/meteo.py``
       already parsed the column (``IDX["iLWi"]``) but nothing downstream
       read it, so ``LWin`` was always cloud-derived.
    2. The live tau_cloud inversion (:func:`geotop_py.meteo.step
       ._find_tau_cloud_live`) used the *point's* ``Latitude``/``Longitude``
       for solar geometry instead of the *station's*
       (``MeteoStationLatitude``/``MeteoStationLongitude``,
       ``radiation.cc:742-744``: ``(*met->st->lat)(i)``/``(*met->st->lon)
       (i)``, not the point's). ``ColdelaPorte``'s station longitude
       (46.627) is wildly different from the point's (5.77), so the
       inversion ran at the wrong hour angle -- a ~175 W/m2 ``SWin`` error
       at times, cascading into an 7 degC ``Tsurface`` error.
    """
    case = "ColdelaPorte"
    sim_dir, recs = reference_run(case)
    assert all(r.out.converged for r in recs[1])

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "surface0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case, "output-tabs-SE27XX",
                                            "surface0001.txt"))
    assert len(py.dates) == len(ref.dates)

    def diffs(col):
        pv, rv = py.col(col), ref.col(col)
        d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
        assert d, f"no comparable samples for {col}"
        return d

    for col in ("LWin[W/m2]", "SWin[W/m2]"):
        assert max(diffs(col)) == pytest.approx(0.0, abs=1.0e-4), col

    # Regression guard on the tiny remaining Tsurface residual.
    assert max(diffs("Tsurface[C]")) < 0.15


class _FakePar:
    """Minimal ``ParFile`` stand-in: one keyword, one component."""

    def __init__(self, value):
        self.value = value

    def number(self, key, i=0, default=None):
        assert key == "DtPlotPoint" and i == 0
        return self.value if self.value is not None else default


def test_dtplot_point_converts_hours_and_snaps_up_to_the_timestep():
    """The keyword is in hours; an interval at or below the nominal timestep
    reports once per timestep instead."""
    assert pipeline.dtplot_point(_FakePar(1.0), 900.0) == 3600.0
    assert pipeline.dtplot_point(_FakePar(24.0), 3600.0) == 86400.0
    # equal, and just-below (Calabria's 0.0833333333333 h = 299.99999999880 s
    # against its 300 s timestep): both snap to the timestep
    assert pipeline.dtplot_point(_FakePar(1.0), 3600.0) == 3600.0
    assert pipeline.dtplot_point(_FakePar(0.0833333333333), 300.0) == 300.0
    # absent keyword: no point output at all
    assert pipeline.dtplot_point(_FakePar(None), 3600.0) == 0.0


def test_accumulation_classes_partition_the_columns():
    classified = (output_tabs.BOOK_COLUMNS + output_tabs.WEIGHTED_COLUMNS
                  + output_tabs.SUMMED_COLUMNS + output_tabs.SNAPSHOT_COLUMNS)
    assert sorted(classified) == sorted(output_tabs.POINT_COLUMNS)
    assert len(set(classified)) == len(classified)


def test_point_accumulator_averages_weights_sums_and_snapshots():
    """Four 900 s steps into a 3600 s window: fluxes come out time-weighted,
    mm amounts come out summed, state comes out as the last step's value."""
    acc = output_tabs.PointAccumulator(7, 3600.0)
    for k, (Ta, prec, swe) in enumerate(
            [(0.0, 1.0, 10.0), (4.0, 2.0, 12.0), (8.0, 0.0, 11.0), (12.0, 0.5, 9.0)]):
        row = {c: 0.0 for c in output_tabs.POINT_COLUMNS}
        row["Tair[C]"] = Ta
        row["Prain_over_canopy[mm]"] = prec
        row["snow_water_equivalent[mm]"] = swe
        acc.add(row, 900.0)
        assert acc.nsub == k + 1

    out = acc.flush(datetime(2014, 6, 18, 20, 0), 735768.833333, 735768.791666)
    assert out["Tair[C]"] == pytest.approx(6.0)          # (0+4+8+12)/4
    assert out["Prain_over_canopy[mm]"] == pytest.approx(3.5)   # 1+2+0+0.5
    assert out["snow_water_equivalent[mm]"] == pytest.approx(9.0)  # last step
    assert out["Date12[DDMMYYYYhhmm]"] == "18/06/2014 20:00"
    assert out["IDpoint"] == 7.0
    assert out["Run"] == 1.0
    # flush resets: a second window starts from zero
    assert acc.nsub == 0
    assert acc.acc["Tair[C]"] == 0.0


@pytest.mark.slow
@needs_reference
def test_dtplot_point_window_longer_than_the_timestep(reference_run):
    """Dtplot point window longer than the timestep."""
    case = "CostantMeteo"
    sim_dir, recs = reference_run(case)
    assert len(recs[1]) == 35040          # 365 days x 96 steps of 900 s
    assert all(r.out.converged for r in recs[1])

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "surface0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case, "output-tabs-SE27XX",
                                            "surface0001.txt"))
    assert len(ref.dates) == 8760
    assert py.dates == ref.dates

    def diffs(col):
        pv, rv = py.col(col), ref.col(col)
        return [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]

    for col in ("Tair[C]", "SWin[W/m2]", "snow_water_equivalent[mm]",
                "snow_depth[mm]", "snow_melted[mm]", "Prain_over_canopy[mm]",
                "Psnow_over_canopy[mm]"):
        d = diffs(col)
        assert d, f"no comparable samples for {col}"
        assert max(d) == pytest.approx(0.0, abs=1.0e-4), f"{col}: max diff {max(d)}"

    # Check surface fluxes after accumulation over the reporting interval.
    assert max(diffs("Tsurface[C]")) < 0.2          # measured 0.156
    assert max(diffs("Soil_heat_flux[W/m2]")) < 2.0  # measured 1.64
    assert max(diffs("SWup[W/m2]")) < 3.5            # measured 2.91


@pytest.mark.slow
@needs_reference
def test_arf_1d_starts_from_a_nonzero_initial_snowpack(reference_run):
    """Arf 1d starts from a nonzero initial snowpack."""
    case = "ARF_1D"
    sim_dir, recs = reference_run(case)
    assert all(r.out.converged for r in recs[1])

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "point0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case, "output-tabs-SE27XX",
                                            "point0001.txt"))
    assert len(py.dates) == len(ref.dates) == 10

    def diffs(col):
        pv, rv = py.col(col), ref.col(col)
        return [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]

    # Regression guard: before the fix these were off by the whole initial
    # pack (SWE ~24mm, depth ~122mm) from the very first row.
    assert max(diffs("snow_water_equivalent[mm]")) < 0.1
    assert max(diffs("snow_depth[mm]")) < 1.0
    assert max(diffs("snow_density[kg/m3]")) < 2.0
    assert max(diffs("Tsurface[C]")) < 1.0
    assert max(diffs("Soil_heat_flux[W/m2]")) < 5.0


@pytest.mark.slow
@needs_reference
@pytest.mark.parametrize("case,tsurf_bound,gef_bound", [
    ("Matsch_B2_Ref_007", 0.15, 4.5),
    ("Matsch_P2_Ref_007", 0.35, 11.0),
])
def test_matsch_matches_with_the_dtplotpoint_accumulator(case, tsurf_bound, gef_bound, reference_run):
    """Matsch matches with the dtplotpoint accumulator."""
    sim_dir, recs = reference_run(case)
    assert all(r.out.converged for r in recs[1])

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "point0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case, "output-tabs-SE27XX",
                                            "point0001.txt"))
    assert len(py.dates) == len(ref.dates)

    def diffs(col):
        pv, rv = py.col(col), ref.col(col)
        d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
        assert d, f"no comparable samples for {col}"
        return d

    for col in ("Tair[C]", "LWin[W/m2]"):
        assert max(diffs(col)) == pytest.approx(0.0, abs=1.0e-3), col

    assert max(diffs("Tsurface[C]")) < tsurf_bound
    assert max(diffs("Soil_heat_flux[W/m2]")) < gef_bound


@needs_reference
def test_richards_profile_outputs_are_produced(reference_run):
    """Richards profile outputs are produced."""
    case = "PureDrainage"
    sim_dir, _ = reference_run(case)

    for fn, bound in (("thetaliq0001.txt", 0.01), ("soilpsi0001.txt", 250.0)):
        py = gt_output.read_point(str(sim_dir / "output-tabs_py" / fn))
        ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                                 "output-tabs-SE27XX", fn))
        assert len(py.dates) == len(ref.dates) == 439
        assert py.dates[0] == datetime(2014, 6, 18, 19, 0)
        depth_cols = [c for c in py.columns if c not in
                     ("Date12[DDMMYYYYhhmm]", "JulianDayFromYear0[days]",
                      "TimeFromStart[days]", "Simulation_Period", "Run", "IDpoint")]
        for c in depth_cols:
            pv, rv = py.col(c), ref.col(c)
            d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
            assert d, f"{fn}: no comparable samples for {c}"
            assert max(d) < bound, f"{fn} {c}: max diff {max(d)}"

    # thetaliq's deepest node is unaffected by the surface residual --
    # soilpsi's is not, see the psi_teta fallback note above.
    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "thetaliq0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                            "output-tabs-SE27XX", "thetaliq0001.txt"))
    deepest = list(py.columns)[-1]
    assert max(abs(a - b) for a, b in
              zip(py.col(deepest), ref.col(deepest))) == 0.0


@pytest.mark.slow
@needs_reference
def test_soil_ice_content_profile_is_produced(reference_run):
    """``thetaice0001.txt`` (``SoilIceContentProfileFile``) was never
    written -- ``LayerProfile`` had no ice-content field at all
    (``snapshot_profile`` only ever populated ``soil_th``/``soil_psi``).
    ``B2_BeG_017`` is the smallest of the four cases that set this keyword.

    Distinct from ``SoilAveragedIceContentProfileFile`` (``ARF_1D``'s own
    keyword for this quantity): that one is time-weighted over the nominal
    step, not a same-step snapshot, and has its own writer.
    """
    case = "B2_BeG_017"
    sim_dir, _ = reference_run(case)

    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / "thetaice0001.txt"))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                            "output-tabs-SE27XX", "thetaice0001.txt"))
    assert len(py.dates) == len(ref.dates)
    depth_cols = [c for c in py.columns if c not in
                 ("Date12[DDMMYYYYhhmm]", "JulianDayFromYear0[days]",
                  "TimeFromStart[days]", "Simulation_Period", "Run", "IDpoint")]
    for c in depth_cols:
        pv, rv = py.col(c), ref.col(c)
        d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
        assert d, f"no comparable samples for {c}"
        assert max(d) == pytest.approx(0.0, abs=1.0e-4), f"{c}: max diff {max(d)}"


@pytest.mark.slow
@needs_reference
def test_discharge_file_is_produced(reference_run):
    """Discharge file is produced."""
    import csv

    case = "B2_BeG_017"
    sim_dir, _ = reference_run(case)

    def _read(path):
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        return rows[0], rows[1:]

    py_header, py_rows = _read(sim_dir / "output-tabs_py" / "discharge.txt")
    ref_header, ref_rows = _read(os.path.join(REFERENCE_1D, case,
                                              "output-tabs-SE27XX", "discharge.txt"))
    assert py_header == ref_header
    assert len(py_rows) == len(ref_rows)
    for i, (pr, rr) in enumerate(zip(py_rows, ref_rows)):
        assert pr[0] == rr[0], f"row {i}: date {pr[0]!r} != {rr[0]!r}"
        for j in range(1, len(rr)):
            assert float(pr[j]) == pytest.approx(float(rr[j]), abs=1.0e-9), \
                f"row {i} col {j} ({ref_header[j]}): {pr[j]} != {rr[j]}"


@pytest.mark.slow
@needs_reference
def test_basin_file_is_produced(reference_run):
    """``basin.txt`` (``BasinOutputFile``, a run-wide file like
    ``discharge.txt`` but with its own ``DtPlotBasin`` cadence and a mix of
    ``Dt/Dtplot_basin``-weighted averages and raw sums in one row) was never
    written. ``B2_BeG_017`` has no other
    residual once this file exists -- the case is fully green end to end.

    Uses ``csv`` directly, not :func:`geotop_py.io.gt_output.read_point`,
    for the same reason as the ``discharge.txt`` test above: two of
    ``basin.txt``'s own header cells are the literal duplicate string
    ``"Prain_above_canopy[mm]"`` (parameters.cc:707-711, kept verbatim), so
    matching by name is not meaningful here either -- position is.
    """
    import csv

    case = "B2_BeG_017"
    sim_dir, _ = reference_run(case)

    def _read(path):
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        return rows[0], rows[1:]

    py_header, py_rows = _read(sim_dir / "output-tabs_py" / "basin.txt")
    ref_header, ref_rows = _read(os.path.join(REFERENCE_1D, case,
                                              "output-tabs-SE27XX", "basin.txt"))
    assert py_header == ref_header
    assert len(py_rows) == len(ref_rows)
    for i, (pr, rr) in enumerate(zip(py_rows, ref_rows)):
        assert pr[0] == rr[0], f"row {i}: date {pr[0]!r} != {rr[0]!r}"
        for j in range(1, len(rr)):
            assert float(pr[j]) == pytest.approx(float(rr[j]), abs=1.0e-9), \
                f"row {i} col {j} ({ref_header[j]}): {pr[j]} != {rr[j]}"


@pytest.mark.slow
@needs_reference
def test_arf_1d_reads_the_time_dependent_vegetation_file(reference_run):
    """Arf 1d reads the time dependent vegetation file."""
    case = "ARF_1D"
    sim_dir, _ = reference_run(case)

    for fn in ("point0001.txt", "soiltemp0001.txt", "thetaice0001.txt",
              "thetaliq0001.txt"):
        py = gt_output.read_point(str(sim_dir / "output-tabs_py" / fn))
        ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                                "output-tabs-SE27XX", fn))
        assert py.dates == ref.dates
        for col in py.columns:
            pv, rv = py.col(col), ref.col(col)
            for i in range(len(rv)):
                if rv[i] != rv[i]:            # NaN in the reference too
                    continue
                assert pv[i] == pytest.approx(rv[i], abs=1.0e-4), \
                    f"{fn} row {i} col {col!r}: {pv[i]} != {rv[i]}"


@pytest.mark.slow
@needs_reference
def test_jungfraujoch_interpolates_soilplotdepths(reference_run):
    """Jungfraujoch interpolates soilplotdepths."""
    case = "Jungfraujoch"
    sim_dir, _ = reference_run(case)

    for fn in ("soil0032.txt", "soil0033.txt"):
        py = gt_output.read_point(str(sim_dir / "output-tabs_py" / fn))
        ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                                "output-tabs-SE27XX", fn))
        assert py.columns == ref.columns
        assert "100.000000" in py.columns
        assert py.dates == ref.dates
        pv, rv = py.col("100.000000"), ref.col("100.000000")
        for i in range(len(rv)):
            if rv[i] != rv[i]:
                continue
            assert pv[i] == pytest.approx(rv[i], abs=1.0e-9), \
                f"{fn} row {i}: {pv[i]} != {rv[i]}"


@needs_reference
def test_only_the_profile_files_the_case_asks_for_are_written(reference_run):
    """``PureDrainage`` sets three soil profile keywords, all under
    ``output-tabs``, and no snow or soil-ice one. GEOtop writes an output file
    only when its keyword is set (``output.cc:3239``), so nothing else may
    appear -- in particular no ``output-prof`` directory of defaults."""
    sim_dir, _ = reference_run("PureDrainage")
    produced = sorted(p.relative_to(sim_dir).as_posix()
                      for p in sim_dir.glob("output*/*") if p.is_file()
                      and not p.name.startswith("."))
    # the run's suffix applies to every output directory: output-prof_py
    assert list(sim_dir.glob("output-prof*")) == []
    assert [p for p in produced if "thetaice" in p or p.split("/")[1].startswith("snow")] == []
    assert {"output-tabs_py/soiltemp0001.txt", "output-tabs_py/thetaliq0001.txt",
            "output-tabs_py/soilpsi0001.txt"} <= set(produced)


@pytest.mark.slow
@needs_reference
def test_output_files_are_named_by_the_point_s_real_id(reference_run):
    """``Jungfraujoch``'s two points have IDs 32/33 (``listpoints.txt``), not
    1/2 -- every output file name *and* every row's own ``IDpoint`` column
    used the internal 1..N loop index instead, invisible on every other
    reference case because their real IDs happen to start at 1. ``soil0032.txt``/``soil0033.txt`` is
    GEOtop's own naming (``SoilAveragedTempProfileFile``); this driver
    used to write ``soil0001.txt``/``soil0002.txt`` instead.
    """
    case = "Jungfraujoch"
    sim_dir, recs = reference_run(case)
    # the in-memory records are keyed by the same ID as the files
    assert set(recs) == {32, 33}

    out_dir = sim_dir / "output-tabs_py"
    assert (out_dir / "soil0032.txt").exists()
    assert (out_dir / "soil0033.txt").exists()
    assert not (out_dir / "soil0001.txt").exists()
    assert not (out_dir / "soil0002.txt").exists()

    with open(out_dir / "soil0032.txt", newline="") as fh:
        import csv
        rows = list(csv.reader(fh))
    idpoint_col = rows[0].index("IDpoint")
    assert rows[1][idpoint_col] == "32"


@pytest.mark.slow
@needs_reference
def test_soil_averaged_temp_profile_is_produced(reference_run):
    """Soil averaged temp profile is produced."""
    case = "B2_BeG_017"
    sim_dir, _ = reference_run(case)

    fn = "soilTz0001.txt"
    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / fn))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                            "output-tabs-SE27XX", fn))
    assert len(py.dates) == len(ref.dates)
    depth_cols = [c for c in py.columns if c not in
                 ("Date12[DDMMYYYYhhmm]", "JulianDayFromYear0[days]",
                  "TimeFromStart[days]", "Simulation_Period", "Run", "IDpoint")]
    for c in depth_cols:
        pv, rv = py.col(c), ref.col(c)
        d = [abs(pv[i] - rv[i]) for i in range(len(rv)) if rv[i] == rv[i]]
        assert d, f"{fn}: no comparable samples for {c}"
        assert max(d) == pytest.approx(0.0, abs=1.0e-4), f"{fn} {c}: max diff {max(d)}"


@pytest.mark.slow
@needs_reference
def test_total_water_pressure_profile_is_produced(reference_run):
    """``SoilTotWaterPressProfileFile`` is ``sl->Ptot``, not the matric
    potential ``SS->P`` and not the soil ice content. ``CostantMeteo`` names
    that file ``Wice0001.txt`` and is the only reference case that requests
    it; every other table in the case was already bit-exact while this one
    was missing entirely.
    """
    case = "CostantMeteo"
    sim_dir, _ = reference_run(case)

    fn = "Wice0001.txt"
    py = gt_output.read_point(str(sim_dir / "output-tabs_py" / fn))
    ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                            "output-tabs-SE27XX", fn))
    assert py.header == ref.header
    assert py.dates == ref.dates
    for column in ref.columns:
        pv, rv = py.col(column), ref.col(column)
        for i in range(len(rv)):
            if rv[i] != rv[i]:
                continue
            assert pv[i] == pytest.approx(rv[i], rel=1.0e-5, abs=1.0e-8), \
                f"{fn} {column} row {i}: {pv[i]} != {rv[i]}"


@pytest.mark.slow
@needs_reference
def test_calabria_reads_its_topography_from_raster_maps(reference_run):
    """``Calabria``'s single point gives only an ID and a coordinate pair:
    elevation, land cover, soil type, slope, aspect and sky view factor all
    come from raster maps, and the four curvatures are computed over the whole
    elevation model. Nothing in this case can be right
    if the grid is read a row out or a cell out, so it is the end-to-end
    check on the raster path -- the driver used to refuse the case outright.
    """
    case = "Calabria"
    sim_dir, _ = reference_run(case)

    for fn in ("point0001.txt", "soiltemp0001.txt", "soilpsi0001.txt",
              "thetaliq0001.txt"):
        py = gt_output.read_point(str(sim_dir / "output-tabs_py" / fn))
        ref = gt_output.read_point(os.path.join(REFERENCE_1D, case,
                                                "output-tabs-SE27XX", fn))
        assert py.header == ref.header, fn
        assert py.dates == ref.dates, fn
        for column in ref.columns:
            pv, rv = py.col(column), ref.col(column)
            for i in range(len(rv)):
                if rv[i] != rv[i]:
                    continue
                assert pv[i] == pytest.approx(rv[i], rel=1.0e-5, abs=1.0e-8), \
                    f"{fn} {column} row {i}: {pv[i]} != {rv[i]}"
