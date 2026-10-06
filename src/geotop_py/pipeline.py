"""GEOtop-compatible 1D simulation runner.

Read native parameters and forcing, initialize point columns, and advance the
coupled energy and water balances. Each trial copies persistent column,
surface, hydrological and station state. Only accepted substeps are committed;
failed trials are retried with a shorter timestep shared by all points.
Output accumulators follow GEOtop's reporting intervals and units.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from . import dates
from .energy import surface
from .energy import vegetation as veg
from .initialize import init_points
from .io import meteo as io_meteo
from .io import parfile, points
from .io import soil as io_soil
from .io import vegfile as io_vegfile
from .meteo import forcing, meteodata
from .output.recorder import OutputRecorder
from .point import state as point_state
from .point import step as point_step
from .point import time_loop
from .results import Results, StepRecord
from .water import coupling

ProgressCallback = Callable[[int, int, datetime], None]


def _date12_to_datetime(date12: float) -> datetime:
    s = f"{int(date12):012d}"
    day, month, year, hh, mm = int(s[0:2]), int(s[2:4]), int(s[4:8]), int(s[8:10]), int(s[10:12])
    return datetime(year, month, day, hh, mm)


@dataclass
class _PointTrial:
    """One point's share of a trial: its working state and step record."""
    col: point_state.Column1D
    state: surface.SurfaceState
    rich: Optional[coupling.RichardsCoupling]
    m: forcing.Meteo
    out: point_step.StepOut
    water_input: Optional[point_step.WaterInput] = None


@dataclass
class _Trial:
    """A whole-domain trial: the points it reached, in order, and the
    per-point values that outlive it, as they stood when it ended."""
    points: Dict[int, _PointTrial]
    last_energy: Dict[int, Optional[Tuple[forcing.Meteo, point_step.StepOut, float]]]
    last_water: Dict[int, Tuple[float, float]]
    water_done: bool = False   # the water balance ran for every point and converged


def _by_point_id(recs: Dict[int, List[StepRecord]],
                 props: Dict[int, points.PointProperties]) -> Results:
    """Re-key the records from the internal 1..N loop index to the point ID."""
    return {props[p].ID: r for p, r in recs.items()}


def run_simulation(sim_dir: str, suffix: str = "", verbose: bool = True,
                   progress: Optional[ProgressCallback] = None,
                   min_Dt: float = 60.0) -> Results:
    """Run the case described by ``sim_dir/geotop.inpts`` end to end and
    write GEOtop-format outputs.

    Returns ``{point ID: [StepRecord, ...]}``, one record per committed
    internal step, keyed by the GEOtop point ID that also names the output
    files (see :mod:`geotop_py.results`).
    """
    pf = parfile.parse(os.path.join(sim_dir, "geotop.inpts"))

    exp = {"EnergyBalance": 1, "PointSim": 1}
    bad = [f"{k}={pf.number(k, 0)!r}" for k, v in exp.items()
           if int(pf.number(k, 0, -999.0)) != v]
    if bad:
        raise ValueError("unsupported configuration: " + ", ".join(bad))
    water_balance = int(pf.number("WaterBalance", 0, 0.0)) == 1

    Dt = pf.number("TimeStepEnergyAndWater", 0, 3600.0)
    JD0 = dates.dateeur12_to_JDfrom0(pf.number("InitDateDDMMYYYYhhmm", 0))
    init_dt = _date12_to_datetime(pf.number("InitDateDDMMYYYYhhmm", 0))
    end_dt = _date12_to_datetime(pf.number("EndDateDDMMYYYYhhmm", 0))
    nsteps = int(round((end_dt - init_dt).total_seconds() / Dt))

    chkpt = points.load(pf, sim_dir)
    point_ids = list(range(1, len(chkpt)))
    props = {p: points.properties(chkpt, p) for p in point_ids}

    soilp = io_soil.load(pf, sim_dir)
    lapse = forcing.LapseRates.from_parfile(pf)
    mcfg = forcing.MeteoDistrConfig.from_parfile(pf)

    stem = pf.string("MeteoFile")
    meteo_paths = sorted(glob.glob(os.path.join(sim_dir, stem + "[0-9]*.txt")))
    mopt = io_meteo.MeteoOptions.from_parfile(pf, 1)
    table = io_meteo.load(meteo_paths[0], io_meteo.column_names(pf.strings), mopt)
    Z_station = pf.number("MeteoStationElevation", 0, 0.0)

    setup = init_points(pf, sim_dir, props, soilp, mcfg, water_balance)
    cols, states, stations = setup.cols, setup.states, setup.stations
    statics, richards = setup.statics, setup.richards
    veg_series, veg_static, veg_Dz = setup.veg_series, setup.veg_static, setup.veg_Dz

    output = OutputRecorder(pf, Dt, JD0, init_dt, props, cols)
    recs: Dict[int, List[StepRecord]] = {p: [] for p in point_ids}
    istart_by_point = {p: 0 for p in point_ids}
    # GEOtop: src/geotop/energy.balance.cc:80 (static long line_interp)
    # One search cursor for every vegetation series, shared across points and
    # land cover classes exactly as GEOtop's own single static is. Boxed so
    # the per-attempt closure can advance it.
    veg_line = [0]
    # What survives a trial whatever its fate, per point: the output values of
    # the last accepted energy step (meteo, record, Dt) and the net
    # precipitation and mass error of the last water balance -- see the
    # output loop below.
    last_energy: Dict[int, Optional[Tuple[forcing.Meteo, point_step.StepOut, float]]] = {
        p: None for p in point_ids}
    last_water: Dict[int, Tuple[float, float]] = {p: (0.0, 0.0) for p in point_ids}
    t_elapsed = 0.0
    for i in range(nsteps):
        JDb_nominal = JD0 + t_elapsed / 86400.0

        # One trial advances every point by the same Dt, as GEOtop's
        # EnergyBalance does: the first point whose energy balance fails ends
        # the trial, and the whole domain halves Dt. Points after it keep the
        # state the trial started from, and the water balance runs only when
        # every point's energy balance was accepted.
        # GEOtop: src/geotop/energy.balance.cc:155-178
        # GEOtop: src/geotop/geotop.cc:326-336
        def attempt(t, dt):
            jb = JD0 + (t_elapsed + t) / 86400.0
            je = JD0 + (t_elapsed + t + dt) / 86400.0
            # The meteo is distributed over the whole domain before any energy
            # balance runs, and its grids are not part of the per-trial state
            # copy: a retried trial starts from the discarded one's "previous
            # time step" values.
            # GEOtop: src/geotop/geotop.cc:299-319
            # GEOtop: src/geotop/meteodistr.cc:68-84
            meteo = {}
            for p in point_ids:
                # the station record is interpolated over the trial's own
                # window, not the nominal step's
                # GEOtop: src/geotop/meteo.cc:56-61
                row, istart_by_point[p] = forcing.interpolate_station_row(
                    table, JDb_nominal, jb, je, istart_by_point[p], mcfg)
                # Vegetation parameters are re-derived per *attempt*, from the
                # attempt's own [jb, je] window but anchored on the nominal
                # step's start -- the same three times the meteo series uses.
                # GEOtop: src/geotop/energy.balance.cc:105-114
                series = veg_series.get(int(props[p].LC))
                if series is not None:
                    values, veg_line[0] = meteodata.time_interp_linear(
                        JDb_nominal, jb, je, series, io_vegfile.COL_DATE, 0, veg_line[0])
                    veg.apply_time_dependent(statics[p].vegpar, veg_static[p],
                                             values, veg_Dz[p])
                meteo[p] = forcing.assemble_meteo(row, dt, Z_station, props[p], lapse,
                                             mcfg, stations[p], jb, je)
            points_trial: Dict[int, _PointTrial] = {}
            energy_ok = True
            for p in point_ids:
                m = meteo[p]
                trial_col = cols[p].copy()
                trial_state = states[p].copy()
                trial_rich = richards[p].copy() if richards is not None else None
                ep = point_step.energy_phase(trial_col, trial_state, statics[p],
                                             dt, m, jb, je)
                if not ep.converged:
                    points_trial[p] = _PointTrial(
                        trial_col, trial_state, trial_rich, m,
                        point_step.failed_step_out(trial_col, ep, trial_rich))
                    energy_ok = False
                    break
                out, water_input = point_step.post_energy_phase(
                    trial_col, trial_state, dt, m, ep, trial_rich)
                last_energy[p] = (m, out, dt)
                point_step.keep_terrain_terms(states[p], trial_state)
                points_trial[p] = _PointTrial(trial_col, trial_state, trial_rich,
                                              m, out, water_input=water_input)
            # The first point whose Richards solve fails ends the water
            # balance: later points are not solved, and the net precipitation
            # and mass error are assigned only once every point has converged.
            # GEOtop: src/geotop/water.balance.cc:175-193
            water_ok = True
            water_done = False
            if energy_ok and richards is not None:
                for tr in points_trial.values():
                    point_step.water_phase(tr.col, tr.rich, dt, tr.out,
                                           tr.water_input)
                    if not tr.out.wb_converged:
                        water_ok = False
                        break
                if water_ok:
                    water_done = True
                    for p, tr in points_trial.items():
                        last_water[p] = (tr.out.pnet, tr.out.wb_loss)
            for tr in points_trial.values():
                if tr.out.converged:
                    point_step.finish_step(tr.col, tr.out, tr.rich)
            return energy_ok and water_ok, _Trial(
                points_trial, dict(last_energy), dict(last_water),
                water_done=water_done)

        def commit(trial):
            for p, tr in trial.points.items():
                cols[p], states[p] = tr.col, tr.state
                if richards is not None:
                    richards[p] = tr.rich

        substeps = time_loop.run(Dt, min_Dt, attempt, commit)
        for p in point_ids:
            rich = richards[p] if richards is not None else None
            for s in substeps:
                tr = s.payload.points.get(p)
                t_commit = t_elapsed + s.t_begin + s.dt_advanced
                date = init_dt + timedelta(days=t_commit / 86400.0)
                JD_commit = JD0 + t_commit / 86400.0
                if tr is not None:
                    recs[p].append(StepRecord(date, JD_commit, s.dt_advanced,
                                              s.dt_run, s.trials, tr.m, tr.out))
                output.add_substep(p, s, tr, date, JD_commit, cols[p], rich)
            output.end_point_step(p, cols[p], rich)
        t_elapsed += Dt
        output.end_nominal_step(t_elapsed, recs, cols)
        if progress is not None:
            progress(i + 1, nsteps, init_dt + timedelta(seconds=t_elapsed))

    output.write(sim_dir, suffix, cols, setup.initial_profiles, recs, verbose)
    return _by_point_id(recs, props)
