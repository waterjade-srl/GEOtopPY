"""Output bookkeeping of a run: reporting cadences, accumulators, file writing.

:class:`OutputRecorder` receives every committed internal step of every point,
accumulates the point, discharge and basin rows and the soil-profile averages
at GEOtop's reporting intervals (``DtPlotPoint``, ``DtPlotDischarge``,
``DtPlotBasin``), and at the end writes each output file whose keyword the
case sets.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .. import constants as C
from .. import dates, laws
from ..constants import STRING_NOVALUE
from ..io import points
from ..point import state as point_state
from ..point import step as point_step
from ..water import coupling, depths
from .profiles import SNOW_PROFILES, write_snow_profile, write_soil_profile
from .tabs import (
    BASIN_BOOK_KEYWORDS,
    BASIN_NFIELD,
    POINT_BOOK_KEYWORDS,
    POINT_COLUMNS,
    BasinAccumulator,
    PointAccumulator,
    basin_values,
    basin_values_unsolved,
    build_row,
    stale_row,
    write_basin,
    write_discharge,
    write_point,
)


def dtplot_point(pf, Dt: float) -> float:
    """Reporting interval of the point tables [s]; ``0`` disables them.

    ``Dt`` is the nominal timestep [s]. An interval at or below ``Dt`` is
    snapped up to ``Dt``: the tables then report once per timestep, the finest
    cadence the model can produce.
    """
    # GEOtop: src/geotop/parameters.cc:1184-1206
    # GEOtop: src/geotop/times.cc:75-76
    # The keyword is in hours, and an interval <= the smallest declared
    # timestep sets plot_point_with_Dt_integration, which set_time_step then
    # resolves to the timestep itself at every step. Calabria's
    # 0.0833333333333 h (299.99999999880 s, just under its 300 s timestep) is
    # the case where the snap is load-bearing rather than cosmetic.
    dtp = pf.number("DtPlotPoint", 0, 0.0) * 3600.0
    if 1.0e-5 < dtp <= Dt:
        dtp = Dt
    return dtp


# GEOtop: src/geotop/parameters.cc:1159-1182
def dtplot_discharge(pf, Dt: float) -> float:
    """Reporting interval of ``discharge.txt`` [s]; ``0`` disables it.

    Same keyword-in-hours, same-or-below-``Dt`` snap as :func:`dtplot_point`
    (the discharge sibling of the ``DtPlotPoint`` block) --
    an independently configurable cadence, not necessarily equal to
    ``DtPlotPoint``.
    """
    dtp = pf.number("DtPlotDischarge", 0, 0.0) * 3600.0
    if 1.0e-5 < dtp <= Dt:
        dtp = Dt
    return dtp


# GEOtop: src/geotop/parameters.cc:1208-1230
def dtplot_basin(pf, Dt: float) -> float:
    """Reporting interval of ``basin.txt`` [s]; ``0`` disables it.

    Same keyword-in-hours, same-or-below-``Dt`` snap as :func:`dtplot_point`
    (the basin sibling of the ``DtPlotPoint`` block).
    """
    dtp = pf.number("DtPlotBasin", 0, 0.0) * 3600.0
    if 1.0e-5 < dtp <= Dt:
        dtp = Dt
    return dtp


def _target(sim_dir: str, keyword: str, point: int, suffix: str) -> str:
    d, base = os.path.split(keyword)
    out_dir = os.path.join(sim_dir, d + suffix)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}{point:04d}.txt")


def _target_global(sim_dir: str, keyword: str, suffix: str) -> str:
    """Like :func:`_target`, for a run-wide file with no point-number suffix
    at all (``discharge.txt``, ``basin.txt`` -- one file for the whole run,
    not one per point)."""
    d, base = os.path.split(keyword)
    out_dir = os.path.join(sim_dir, d + suffix)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}.txt")


def _soil_depth_cols(soil_D_m: List[float], nsoil: int) -> List[str]:
    cols, acc = [], 0.0
    for k in range(1, nsoil + 1):
        cols.append(f"{1000.0 * (acc + soil_D_m[k] / 2.0):f}")
        acc += soil_D_m[k]
    return cols


class _AvgStepOut:
    """The minimal ``(date, JD, m, out)`` shape ``write_soil_profile``
    needs for a flushed record -- it only ever reads ``out.prof`` -- so the
    Tzav-averaged records (which have no real :class:`StepOut` behind them,
    only an accumulated profile) can reuse that writer unmodified."""

    def __init__(self, prof):
        self.prof = prof


# GEOtop: src/geotop/output.cc:216-253
# GEOtop: src/geotop/input.cc:1034 (Ptot = SS->P at t=0)
def _thaw_and_watertable_depths(col, richards=None) -> Tuple[float, float, float, float]:
    """The four soil diagnostics for one column's current state (``depths.py``,
    ``tables.cc``): ``(lowest_thawed, highest_thawed, lowest_water_table,
    highest_water_table)`` [mm].

    In ``write_output`` the ``Dthaw`` used as the depth
    cap ``Z`` for BOTH water-table calls is ``find_activelayerdepth_up``'s
    own result, never ``_dw``'s -- a coupling flagged with "look here!" in
    the reference source itself, reproduced verbatim regardless.

    ``Ptot`` (the water-table scan's total-head-equivalent potential) is
    ``sl->Ptot``, which GEOtop sets by direct assignment from the matric
    potential ``SS->P`` (in ``get_all_input`` at t=0; every Richards step keeps
    it in step by re-deriving it from theta, but *starting from* a psi it
    just solved for -- never the other way around). Inverting ``th0`` through
    ``psi_teta`` cannot recover this for an already-saturated node: theta
    plateaus at ``sat`` for every psi >= the saturation threshold, so the
    inversion returns approximately 0 instead of the node's true (possibly
    large) positive head -- confirmed on PureDrainage's own t=0 state, where
    two nodes below the 5000mm initial water table sit at ``th0==sat``
    exactly and a theta-inversion collapses their head to ~1e-9, moving the
    reported water-table depth by a full layer. Preferring
    ``richards.state.P`` (mirroring ``snapshot_profile``'s own ``soil_psi``
    fallback) avoids the round-trip for every node Richards tracks.
    """
    nsoil = col.nsoil()
    T = [0.0] + [col.soil_T[k] for k in range(1, nsoil + 1)]
    th = [0.0] + [col.soil[k].th0 for k in range(1, nsoil + 1)]
    thi = [0.0] + [col.soil[k].thi0 for k in range(1, nsoil + 1)]
    res = [0.0] + [col.soil[k].res for k in range(1, nsoil + 1)]
    dz_mm = [0.0] + [1000.0 * col.soil_D[k] for k in range(1, nsoil + 1)]
    nl_richards = richards.col.nl if richards is not None else 0
    Ptot = [0.0]
    for k in range(1, nsoil + 1):
        if k <= nl_richards:
            Ptot.append(richards.state.P[k])
        else:
            s = col.soil[k]
            Ptot.append(laws.psi_teta(s.th0 + s.thi0, 0.0, s.sat, s.res,
                                      s.alpha, s.n, s.m, C.PsiMin, s.ss))

    lowest_thawed = depths.find_activelayerdepth_up(T, th, thi, res, dz_mm)
    highest_thawed = depths.find_activelayerdepth_dw(T, th, thi, res, dz_mm)
    Z = lowest_thawed
    lowest_wt = depths.find_watertabledepth_up(Z, Ptot, dz_mm)
    highest_wt = depths.find_watertabledepth_dw(Z, Ptot, dz_mm)
    return lowest_thawed, highest_thawed, lowest_wt, highest_wt


# GEOtop: src/geotop/parameters.cc:2113-2147
# Which of the six soil-profile book columns (SOIL_BOOK_FIELDS' canonical
# order) get written, and at what position. SoilAll=1 forces all six in
# canonical order; otherwise each of DateSoil/JulianDayFromYear0Soil/
# TimeFromStartSoil/PeriodSoil/RunSoil/IDPointSoil claims a 1-based output
# position (default -1: not written).
_SOIL_BOOK_KEYWORDS = ("DateSoil", "JulianDayFromYear0Soil", "TimeFromStartSoil",
                       "PeriodSoil", "RunSoil", "IDPointSoil")


def _soil_book_fields(pf) -> tuple:
    if int(pf.number("SoilAll", 0, 0.0)) == 1:
        return (0, 1, 2, 3, 4, 5)
    osl = [-1] * 6
    for field, keyword in enumerate(_SOIL_BOOK_KEYWORDS):
        pos = int(pf.number(keyword, 0, -1.0))
        if 1 <= pos <= 6:
            osl[pos - 1] = field
    # GEOtop: src/geotop/parameters.cc:2142-2146
    # nosl stops right after the last POSITIVE field index, not the last
    # assigned one: field 0 (Date) alone at the final position would
    # truncate to nothing. Reproduced verbatim;
    # none of the reference cases' configurations ever hit it (every one
    # that sets these keywords also assigns IDPointSoil, field index 5, to
    # its highest used position).
    nosl = 0
    for i in range(6):
        if osl[i] > 0:
            nosl = i + 1
    return tuple(osl[:nosl])


# GEOtop: src/geotop/parameters.cc:1912-1953 (keyword codes 295-321)
# GEOtop: src/geotop/parameters.cc:1948-1952 (the nobsn quirk)
# The basin sibling of _soil_book_fields, generalised to the full 27-field
# BASIN_FIELD_NAMES set (soil's book is only the leading identity columns;
# basin's position keywords cover every column, data included). Same
# BasinAll=1 shortcut, same nobsn "stop after the last *positive* field
# index" quirk -- untested by any of the three
# reference cases that need it, all of which set BasinAll=1 and never
# exercise the explicit-position branch.
def _basin_book_fields(pf) -> tuple:
    if int(pf.number("BasinAll", 0, 0.0)) == 1:
        return tuple(range(BASIN_NFIELD))
    obsn = [-1] * BASIN_NFIELD
    for field, keyword in enumerate(BASIN_BOOK_KEYWORDS):
        pos = int(pf.number(keyword, 0, -1.0))
        if 1 <= pos <= BASIN_NFIELD:
            obsn[pos - 1] = field
    nobsn = 0
    for i in range(BASIN_NFIELD):
        if obsn[i] > 0:
            nobsn = i + 1
    return tuple(obsn[:nobsn])


# GEOtop: src/geotop/parameters.cc:1869-1910 (keyword codes 215-294)
# The point sibling of _basin_book_fields/_soil_book_fields, over the full
# POINT_COLUMNS set. Only CostantMeteo
# among the 13 reference cases leaves PointAll unset.
def _point_book_fields(pf) -> tuple:
    n = len(POINT_COLUMNS)
    if int(pf.number("PointAll", 0, 0.0)) == 1:
        return tuple(POINT_COLUMNS)
    opnt = [-1] * n
    for field, keyword in enumerate(POINT_BOOK_KEYWORDS):
        pos = int(pf.number(keyword, 0, -1.0))
        if 1 <= pos <= n:
            opnt[pos - 1] = field
    nopnt = 0
    for i in range(n):
        if opnt[i] > 0:
            nopnt = i + 1
    return tuple(POINT_COLUMNS[f] if f >= 0 else None for f in opnt[:nopnt])



class OutputRecorder:
    """Accumulates every point's output rows during a run and writes them.

    Per nominal step the time loop calls :meth:`add_substep` once per point
    and committed sub-step, :meth:`end_point_step` once per point after its
    sub-steps, then :meth:`end_nominal_step`; :meth:`write` closes the run.
    """

    def __init__(self, pf, Dt: float, JD0: float, init_dt: datetime,
                 props: Dict[int, points.PointProperties],
                 cols: Dict[int, point_state.Column1D]):
        point_ids = list(props)
        self.pf, self.Dt, self.JD0, self.init_dt = pf, Dt, JD0, init_dt
        self.point_ids, self.props = point_ids, props
        self.Dtplot = dtplot_point(pf, Dt)
        self.state_pixel = self.Dtplot > 1.0e-5
        # One entry per emitted row: the accumulated point row, and the record of
        # the internal step the row closes on -- the profile files report that
        # step's state directly rather than an accumulation of the window.
        self.rows: Dict[int, List[Dict[str, float]]] = {p: [] for p in point_ids}
        self.plotted: Dict[int, List[Tuple]] = {p: [] for p in point_ids}
        # GEOtop identifies every output file/row by the point's own ID (e.g.
        # Jungfraujoch's listpoints.txt IDs are 32/33, not 1/2 -- point_ids
        # above is a purely internal 1..N loop index, coincidentally equal to
        # the real ID on every case except this one, which is why the mismatch
        # went unnoticed until now).
        self.accs = {p: PointAccumulator(props[p].ID, self.Dtplot) for p in point_ids}
        # GEOtop: src/geotop/output.cc:216-253
        # The four thaw/water-table diagnostics accumulate on the *nominal* step
        # (par->Dt), never the internal sub-step -- a
        # different cadence from every other WEIGHTED_COLUMNS entry, which is
        # why they cannot go through PointAccumulator itself and get their own
        # small one here.
        self.tables_acc = {p: [0.0, 0.0, 0.0, 0.0] for p in point_ids}
        # GEOtop: src/geotop/output.cc:231-239
        # The Tzav family (SoilAveragedTempProfileFile/SoilAveragedIceContent
        # ProfileFile/SoilAveragedLiqContentProfileFile) shares the exact same
        # nominal-step cadence and Dt/Dtplot weight as the four scalars above,
        # just per soil node instead of a single value.
        self.soilT_acc = {p: [0.0] * (cols[p].nsoil() + 1) for p in point_ids}
        self.soilth_acc = {p: [0.0] * (cols[p].nsoil() + 1) for p in point_ids}
        self.soilthi_acc = {p: [0.0] * (cols[p].nsoil() + 1) for p in point_ids}
        self.avg_plotted: Dict[int, List[Tuple]] = {p: [] for p in point_ids}
        # GEOtop: src/geotop/output.cc:140-209
        # discharge.txt: an independent cadence (DtPlotDischarge) and, for a
        # point simulation with no channel network, only two nonzero terms --
        # Vbottom/Vlat, raw volumes [m3] summed (not Dt-weighted) across every
        # committed internal sub-step, divided by Dtplot_discharge only at
        # write time to report a rate.
        self.Dtplot_discharge = dtplot_discharge(pf, Dt)
        self.state_discharge = self.Dtplot_discharge > 1.0e-5
        self.discharge_acc = {p: [0.0, 0.0] for p in point_ids}
        self.discharge_rows: Dict[int, List[Dict[str, float]]] = {p: [] for p in point_ids}
        self.t_discharge = 0.0
        # basin.txt: same independent-cadence pattern as discharge.txt (its own
        # DtPlotBasin, not DtPlotPoint), but a mix of Dt/Dtplot_basin-weighted
        # averages and raw sums in one row instead of two raw sums -- see
        # BasinAccumulator. Also a run-wide file like discharge.txt: only
        # point_ids[0]'s accumulator is ever written; none of the reference cases
        # that set BasinOutputFile simulate more than one point.
        self.Dtplot_basin = dtplot_basin(pf, Dt)
        self.state_basin = self.Dtplot_basin > 1.0e-5
        self.basin_accs = {p: BasinAccumulator(self.Dtplot_basin) for p in point_ids}
        self.basin_rows: Dict[int, List[Dict[int, object]]] = {p: [] for p in point_ids}
        self.t_basin = 0.0
        self.t_point = 0.0

    def add_substep(self, p: int, s, tr, date: datetime, JD_commit: float,
                    col: point_state.Column1D,
                    rich: Optional[coupling.RichardsCoupling]) -> None:
        """Accumulate one committed sub-step ``s`` of point ``p``; ``tr`` is the
        point's share of the trial, ``None`` when the trial ended before it."""
        JD0 = self.JD0
        accepted = tr is not None and tr.out.converged
        if self.state_pixel:
            if accepted:
                self.accs[p].add(
                    build_row(self.props[p].ID, date, JD_commit, JD0, s.dt_run,
                              tr.m, tr.out),
                    s.dt_run)
            else:
                # A point whose energy step did not complete repeats
                # the output values of its last accepted one --
                # from whichever trial, committed or not, and
                # weighted by that trial's Dt -- while the snow and
                # glacier columns are read from the state.
                # GEOtop: src/geotop/energy.balance.cc:975-1216
                # GEOtop: src/geotop/output.cc:305-352
                # GEOtop: src/geotop/output.cc:4927
                state_out = (tr.out if tr is not None else
                             point_step.failed_step_out(col, None, rich))
                last = s.payload.last_energy[p]
                if last is None:
                    self.accs[p].add(stale_row(None, state_out), s.dt_run)
                else:
                    lm, lout, ldt = last
                    self.accs[p].add(stale_row(
                        build_row(self.props[p].ID, date, JD_commit, JD0, ldt,
                                  lm, lout),
                        state_out), ldt)
        if self.state_discharge and tr is not None:
            dacc = self.discharge_acc[p]
            dacc[0] += tr.out.wb_vbottom
            dacc[1] += tr.out.wb_vlat
        if self.state_basin:
            if accepted:
                bv = basin_values(s.dt_run, self.Dtplot_basin, tr.m, tr.out)
            else:
                bv = basin_values_unsolved()
            if not s.payload.water_done:
                # Net precipitation and mass error are assigned by a
                # completed water balance alone: a trial that never
                # reaches it, or fails in it, reports those of the
                # last one that completed.
                # GEOtop: src/geotop/water.balance.cc:192-193
                bv[9], bv[25] = s.payload.last_water[p]
            # The mean-timestep field weighs the Dt the clock advances
            # by, which after a give-up at min_Dt is the halved one.
            # GEOtop: src/geotop/geotop.cc:407
            # GEOtop: src/geotop/geotop.cc:466
            bv[26] = s.dt_advanced * (s.dt_advanced / self.Dtplot_basin)
            self.basin_accs[p].add(bv, s.dt_run)

    def end_point_step(self, p: int, col: point_state.Column1D,
                       rich: Optional[coupling.RichardsCoupling]) -> None:
        """Accumulate the nominal-step diagnostics of point ``p`` from its
        state after the step."""
        if not self.state_pixel:
            return
        lo_th, hi_th, lo_wt, hi_wt = _thaw_and_watertable_depths(col, rich)
        w = self.Dt / self.Dtplot
        acc = self.tables_acc[p]
        acc[0] += lo_th * w
        acc[1] += hi_th * w
        acc[2] += lo_wt * w
        acc[3] += hi_wt * w
        Tacc, thacc, thiacc = self.soilT_acc[p], self.soilth_acc[p], self.soilthi_acc[p]
        for k in range(1, col.nsoil() + 1):
            Tacc[k] += col.soil_T[k] * w
            thacc[k] += col.soil[k].th0 * w
            thiacc[k] += col.soil[k].thi0 * w

    def end_nominal_step(self, t_elapsed: float, recs: Dict[int, list],
                         cols: Dict[int, point_state.Column1D]) -> None:
        """Advance the reporting clocks by one nominal step, ending at
        ``t_elapsed`` [s], and emit every row whose interval closes."""
        Dt, JD0, init_dt = self.Dt, self.JD0, self.init_dt
        # GEOtop: src/geotop/output.cc:216-219
        # GEOtop: src/geotop/output.cc:257
        # GEOtop: src/geotop/output.cc:773
        # GEOtop: src/geotop/output.cc:810
        # The reporting clock ticks once per *nominal* step, however many sub-steps that
        # step needed, and a row is emitted only when the clock lands on
        # DtPlotPoint exactly. An interval that is not a whole multiple of the
        # nominal timestep therefore emits nothing at all, here as there.
        if self.state_pixel:
            self.t_point += Dt
            if abs(self.t_point - self.Dtplot) < 1.0e-5:
                plot_dt = init_dt + timedelta(seconds=t_elapsed)
                plot_JD = JD0 + t_elapsed / 86400.0
                for p in self.point_ids:
                    row_out = self.accs[p].flush(plot_dt, plot_JD, JD0)
                    lo_th, hi_th, lo_wt, hi_wt = self.tables_acc[p]
                    row_out["lowest_thawed_soil_depth[mm]"] = lo_th
                    row_out["highest_thawed_soil_depth[mm]"] = hi_th
                    row_out["lowest_water_table_depth[mm]"] = lo_wt
                    row_out["highest_water_table_depth[mm]"] = hi_wt
                    self.rows[p].append(row_out)
                    self.plotted[p].append(
                        (plot_dt, plot_JD, recs[p][-1].meteo, recs[p][-1].out))
                    self.tables_acc[p] = [0.0, 0.0, 0.0, 0.0]
                    avg_prof = point_step.LayerProfile(
                        Dz=[], T=[], wice=[], wliq=[],
                        soil_T=self.soilT_acc[p][1:], soil_th=self.soilth_acc[p][1:],
                        soil_thi=self.soilthi_acc[p][1:], soil_psi=[], soil_ptot=[])
                    self.avg_plotted[p].append(
                        (plot_dt, plot_JD, None, _AvgStepOut(avg_prof)))
                    nsoil = cols[p].nsoil()
                    self.soilT_acc[p] = [0.0] * (nsoil + 1)
                    self.soilth_acc[p] = [0.0] * (nsoil + 1)
                    self.soilthi_acc[p] = [0.0] * (nsoil + 1)
                self.t_point = 0.0
        if self.state_discharge:
            self.t_discharge += Dt
            if abs(self.t_discharge - self.Dtplot_discharge) < 1.0e-5:
                date_str = (init_dt + timedelta(seconds=t_elapsed)).strftime(
                    "%d/%m/%Y %H:%M")
                JDfrom0 = JD0 + t_elapsed / 86400.0
                JD_yr, _year = dates.JDfrom0_to_JDandYear(JDfrom0)
                for p in self.point_ids:
                    Vbottom, Vlat = self.discharge_acc[p]
                    self.discharge_rows[p].append({
                        "DATE[day/month/year hour:min]": date_str,
                        "t[days]": t_elapsed / 86400.0,
                        "JDfrom0": JDfrom0,
                        "JD": JD_yr,
                        "Qtot[m3/s]": 0.0,
                        "Vsup/Dt[m3/s]": 0.0,
                        "Vsub/Dt[m3/s]": 0.0,
                        "Vchannel[m3]": 0.0,
                        "Qoutlandsup[m3/s]": 0.0,
                        "Qoutlandsub[m3/s]": Vlat / self.Dtplot_discharge,
                        "Qoutbottom[m3/s]": Vbottom / self.Dtplot_discharge,
                    })
                    self.discharge_acc[p] = [0.0, 0.0]
                self.t_discharge = 0.0
        if self.state_basin:
            self.t_basin += Dt
            if abs(self.t_basin - self.Dtplot_basin) < 1.0e-5:
                plot_dt = init_dt + timedelta(seconds=t_elapsed)
                JDfrom0 = JD0 + t_elapsed / 86400.0
                for p in self.point_ids:
                    self.basin_rows[p].append(self.basin_accs[p].flush(plot_dt, JDfrom0, JD0))
                self.t_basin = 0.0

    def write(self, sim_dir: str, suffix: str,
              cols: Dict[int, point_state.Column1D],
              initial_profiles: Dict[int, point_step.LayerProfile],
              recs: Dict[int, list], verbose: bool) -> None:
        """Write every output file whose keyword the case sets."""
        pf, JD0, init_dt, point_ids = self.pf, self.JD0, self.init_dt, self.point_ids
        rows, plotted, avg_plotted = self.rows, self.plotted, self.avg_plotted
        if self.state_discharge:
            # GEOtop: src/geotop/output.cc:2557
            # Gated on state_discharge and the keyword both, same as every other
            # output file here; discharge is a run-wide total (one file, not
            # per-point), but every point of a 1D case shares the same single
            # column, so a single-point run's own discharge is what gets written.
            q_kw = pf.string("DischargeFile", STRING_NOVALUE)
            if q_kw != STRING_NOVALUE:
                write_discharge(_target_global(sim_dir, q_kw, suffix),
                                self.discharge_rows[point_ids[0]])

        if self.state_basin:
            bas_kw = pf.string("BasinOutputFile", STRING_NOVALUE)
            if bas_kw != STRING_NOVALUE:
                obsn = _basin_book_fields(pf)
                write_basin(_target_global(sim_dir, bas_kw, suffix),
                           self.basin_rows[point_ids[0]], obsn)

        prof_keys = {"snowDepth": "SnowDepthLayersFile",
                     "snowTemp": "SnowTempProfileFile",
                     "snowThetaIce": "SnowIceContentProfileFile",
                     "snowThetaW": "SnowLiqContentProfileFile"}
        if not self.state_pixel:
            return
        for p in point_ids:
            col = cols[p]
            pid = self.props[p].ID
            # GEOtop: src/geotop/parameters.cc:225-230
            # Every output-file keyword defaults to string_novalue (absent), as HorizonPointFile does
            # too; "output-tabs/point" was never GEOtop's own default, it
            # only happened to match every case that sets the keyword
            # explicitly. Jungfraujoch and Bro both leave it unset and write no
            # point*.txt at all -- writing one anyway is spurious output that
            # a dashboard.py --run comparison then flags as "extra", not a
            # harmless default.
            pt_kw = pf.string("PointOutputFile", STRING_NOVALUE)
            if pt_kw != STRING_NOVALUE:
                write_point(_target(sim_dir, pt_kw, pid, suffix), rows[p],
                           columns=_point_book_fields(pf))

            # The profile files follow the same rule: written only when their
            # keyword is set, never to a default path.
            # GEOtop: src/geotop/output.cc:3239
            ncol = col.snow.max
            for stem_key in SNOW_PROFILES:
                kw = pf.string(prof_keys[stem_key], STRING_NOVALUE)
                if kw != STRING_NOVALUE:
                    write_snow_profile(_target(sim_dir, kw, pid, suffix), stem_key,
                                       plotted[p], JD0, pid, ncol,
                                       init_date=init_dt, initial=initial_profiles[p])
            depth_cols = _soil_depth_cols(col.soil_D, col.nsoil())
            book_fields = _soil_book_fields(pf)
            # GEOtop: src/geotop/output.cc:4475-4520 (write_soil_header)
            # GEOtop: src/geotop/output.cc:4605-4658 (interpolate_soil)
            # GEOtop: src/geotop/input.cc:1085-1086 (Pzplot from node 0)
            # SoilPlotDepths: interpolate every profile column onto these
            # user-given depths [mm] instead of printing one per internal node.
            # Applies uniformly to every soil
            # profile file *except* soilpsi -- Pzplot is the one profile with a
            # genuine surface node, needing interpolate_soil
            # with lmin=0 and a real P[0], which LayerProfile.soil_psi does not
            # carry (snapshot_profile only ever captures nodes 1..nsoil). None of
            # the 13 reference cases combines SoilPlotDepths with
            # SoilLiqWaterPressProfileFile, so soilpsi keeps its pre-existing,
            # unconverted per-node output rather than interpolating with the
            # wrong lmin.
            plot_depths = None
            plot_dz_mm = None
            if pf.has_number("SoilPlotDepths", 0):
                plot_depths = [pf.number("SoilPlotDepths", j)
                              for j in range(pf.components("SoilPlotDepths"))]
                depth_cols = [f"{v:f}" for v in plot_depths]
                plot_dz_mm = [0.0] + [1000.0 * col.soil_D[k]
                                      for k in range(1, col.nsoil() + 1)]
            soil_kw = pf.string("SoilTempProfileFile", STRING_NOVALUE)
            if soil_kw != STRING_NOVALUE:
                write_soil_profile(_target(sim_dir, soil_kw, pid, suffix), plotted[p], JD0,
                                   pid, depth_cols, init_date=init_dt,
                                   initial=initial_profiles[p], book_fields=book_fields,
                                   plot_depths=plot_depths, dz_mm=plot_dz_mm)
            th_kw = pf.string("SoilLiqContentProfileFile", STRING_NOVALUE)
            if th_kw != STRING_NOVALUE:
                write_soil_profile(_target(sim_dir, th_kw, pid, suffix), plotted[p], JD0,
                                   pid, depth_cols, init_date=init_dt,
                                   initial=initial_profiles[p], field="soil_th",
                                   book_fields=book_fields,
                                   plot_depths=plot_depths, dz_mm=plot_dz_mm)
            thi_kw = pf.string("SoilIceContentProfileFile", STRING_NOVALUE)
            if thi_kw != STRING_NOVALUE:
                write_soil_profile(_target(sim_dir, thi_kw, pid, suffix), plotted[p], JD0,
                                   pid, depth_cols, init_date=init_dt,
                                   initial=initial_profiles[p], field="soil_thi",
                                   book_fields=book_fields,
                                   plot_depths=plot_depths, dz_mm=plot_dz_mm)
            psi_depth_cols = _soil_depth_cols(col.soil_D, col.nsoil())
            psi_kw = pf.string("SoilLiqWaterPressProfileFile", STRING_NOVALUE)
            if psi_kw != STRING_NOVALUE:
                write_soil_profile(_target(sim_dir, psi_kw, pid, suffix), plotted[p], JD0,
                                   pid, psi_depth_cols, init_date=init_dt,
                                   initial=initial_profiles[p], field="soil_psi",
                                   book_fields=book_fields)
            ptot_kw = pf.string("SoilTotWaterPressProfileFile", STRING_NOVALUE)
            if ptot_kw != STRING_NOVALUE:
                write_soil_profile(_target(sim_dir, ptot_kw, pid, suffix), plotted[p],
                                   JD0, pid, depth_cols, init_date=init_dt,
                                   initial=initial_profiles[p], field="soil_ptot",
                                   book_fields=book_fields,
                                   plot_depths=plot_depths, dz_mm=plot_dz_mm)
            # GEOtop: src/geotop/output.cc:231-239
            # GEOtop: src/geotop/input.cc:1113-1117
            # Tzav family: DtPlotPoint-time-weighted-averaged over the window,
            # not a same-step snapshot -- avg_plotted[p]
            # carries the accumulated profile per window, not out.prof.
            # GEOtop's own t=0 initial condition for these is the raw initial
            # value (Tzavplot = SS->T at start-up, same
            # as Tzplot), so initial_profiles[p] -- already a snapshot at that
            # same instant -- is reused unchanged.
            Tavg_kw = pf.string("SoilAveragedTempProfileFile", STRING_NOVALUE)
            if Tavg_kw != STRING_NOVALUE:
                write_soil_profile(_target(sim_dir, Tavg_kw, pid, suffix), avg_plotted[p],
                                   JD0, pid, depth_cols, init_date=init_dt,
                                   initial=initial_profiles[p], field="soil_T",
                                   book_fields=book_fields,
                                   plot_depths=plot_depths, dz_mm=plot_dz_mm)
            thavg_kw = pf.string("SoilAveragedLiqContentProfileFile", STRING_NOVALUE)
            if thavg_kw != STRING_NOVALUE:
                write_soil_profile(_target(sim_dir, thavg_kw, pid, suffix), avg_plotted[p],
                                   JD0, pid, depth_cols, init_date=init_dt,
                                   initial=initial_profiles[p], field="soil_th",
                                   book_fields=book_fields,
                                   plot_depths=plot_depths, dz_mm=plot_dz_mm)
            thiavg_kw = pf.string("SoilAveragedIceContentProfileFile", STRING_NOVALUE)
            if thiavg_kw != STRING_NOVALUE:
                write_soil_profile(_target(sim_dir, thiavg_kw, pid, suffix), avg_plotted[p],
                                   JD0, pid, depth_cols, init_date=init_dt,
                                   initial=initial_profiles[p], field="soil_thi",
                                   book_fields=book_fields,
                                   plot_depths=plot_depths, dz_mm=plot_dz_mm)
            if verbose:
                nonconv = sum(1 for r in recs[p] if not r.out.converged)
                # not _target: it creates the directory, and with no
                # PointOutputFile there is none to name
                where = (os.path.join(sim_dir, os.path.dirname(pt_kw) + suffix)
                         if pt_kw != STRING_NOVALUE else "no point file")
                print(f"point {p}: {len(recs[p])} steps, {len(rows[p])} rows, "
                      f"{nonconv} non-conv -> {where}")
