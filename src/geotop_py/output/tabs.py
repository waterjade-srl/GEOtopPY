"""Write geotop_py's independent run in GEOtop's ``point*.txt`` format.

The point is diagnostic: emitting the *same* 79 columns GEOtop writes lets us
diff geotop_py against ``output_tabs/`` column by column and see exactly which term
diverges (SWnet? LWnet? H? LE? the ground heat flux?), instead of only the three
state variables SWE/depth/Tsurface.

Every column geotop_py actually models is filled from :class:`geotop_py.point.step.StepOut`
and its :class:`~geotop_py.energy.surface.SurfaceDiag` (via ``diag.breakdown(Tg)``), matching
the GEOtop output definitions (energy.balance.cc ~1000-1080, parameters.cc headers).
Columns for processes not in this scenario -- canopy, blowing snow, the soil
thaw depths and water table (Richards off) -- are emitted as GEOtop's own
no-data (``-9999``) or ``0``, so a diff on them is a no-op rather than noise.
The six glacier columns are filled when the glacier module is on and degrade to
the same no-data/zero when it is off.

A row does not necessarily cover one timestep: it covers a reporting window of
``DtPlotPoint`` seconds, over which :class:`PointAccumulator` combines however
many internal steps fall inside it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..constants import NUMBER_NOVALUE
from ..energy import rad, surface
from ..energy import turbulence as tb
from ..meteo.forcing import Meteo
from ..point.step import StepOut

# Exact header order of output_tabs/point*.txt (Date12 first).
POINT_COLUMNS: List[str] = [
    "Date12[DDMMYYYYhhmm]",
    "JulianDayFromYear0[days]", "TimeFromStart[days]", "Simulation_Period",
    "Run", "IDpoint",
    "Psnow_over_canopy[mm]", "Prain_over_canopy[mm]",
    "Psnow_under_canopy[mm]", "Prain_under_canopy[mm]", "Prain_rain_on_snow[mm]",
    "Wind_speed[m/s]", "Wind_direction[deg]", "Relative_Humidity[-]",
    "Pressure[mbar]", "Tair[C]", "Tdew[C]", "Tsurface[C]", "Tvegetation[C]",
    "Tcanopyair[C]", "Surface_Energy_balance[W/m2]", "Soil_heat_flux[W/m2]",
    "SWin[W/m2]", "SWbeam[W/m2]", "SWdiff[W/m2]",
    "LWin[W/m2]", "LWin_min[W/m2]", "LWin_max[W/m2]",
    "SWnet[W/m2]", "LWnet[W/m2]", "H[W/m2]", "LE[W/m2]",
    "Canopy_fraction[-]", "LSAI[m2/m2]", "z0veg[m]", "d0veg[m]",
    "Estored_canopy[W/m2]", "SWv[W/m2]", "LWv[W/m2]", "Hv[W/m2]", "LEv[W/m2]",
    "Hg_unveg[W/m2]", "LEg_unveg[W/m2]", "Hg_veg[W/m2]", "LEg_veg[W/m2]",
    "Evap_surface[mm]", "Trasp_canopy[mm]", "Water_on_canopy[mm]",
    "Snow_on_canopy[mm]", "Qvegetation[-]", "Qsurface[-]", "Qair[-]",
    "Qcanopyair[-]", "LObukhov[m]", "LObukhovcanopy[m]",
    "Wind_speed_top_canopy[m/s]", "Decay_of_K_in_canopy[-]",
    "SWup[W/m2]", "LWup[W/m2]", "Hup[W/m2]", "LEup[W/m2]",
    "snow_depth[mm]", "snow_water_equivalent[mm]", "snow_density[kg/m3]",
    "snow_temperature[C]", "snow_melted[mm]", "snow_subl[mm]",
    "snow_blown_away[mm]", "snow_subl_while_blown[mm]",
    "glac_depth[mm]", "glac_water_equivalent[mm]", "glac_density[kg/m3]",
    "glac_temperature[C]", "glac_melted[mm]", "glac_subl[mm]",
    "lowest_thawed_soil_depth[mm]", "highest_thawed_soil_depth[mm]",
    "lowest_water_table_depth[mm]", "highest_water_table_depth[mm]",
]


def build_row(idpoint: int, date, JD: float, JD0: float, Dt: float,
              m: Meteo, out: StepOut) -> Dict[str, float]:
    """One GEOtop-format record from a geotop_py step. ``date`` is a datetime."""
    d = out.diag
    b = d.reported if d.reported is not None else d.breakdown(out.Tg)

    row: Dict[str, float] = {c: 0.0 for c in POINT_COLUMNS}
    row["Date12[DDMMYYYYhhmm]"] = date.strftime("%d/%m/%Y %H:%M")

    row["JulianDayFromYear0[days]"] = JD
    row["TimeFromStart[days]"] = JD - JD0
    row["Simulation_Period"] = 1.0
    # GEOtop: src/geotop/output.cc:411-414
    # The column is i_run, the 1-based index of the run within the simulation
    # period; a single run reports 1, never 0. Only the multi-run keywords
    # (spin-up, recovery) ever move it.
    row["Run"] = 1.0
    row["IDpoint"] = float(idpoint)

    row["Psnow_over_canopy[mm]"] = m.Psnow
    row["Prain_over_canopy[mm]"] = m.Prain
    row["Psnow_under_canopy[mm]"] = out.Psnow_under
    row["Prain_under_canopy[mm]"] = out.Prain_under
    row["Prain_rain_on_snow[mm]"] = out.RainOnSnow
    row["Wind_speed[m/s]"] = m.wind
    row["Wind_direction[deg]"] = m.wind_dir
    row["Relative_Humidity[-]"] = m.RH
    row["Pressure[mbar]"] = m.P
    row["Tair[C]"] = m.Ta
    row["Tdew[C]"] = rad.dew_point(m.RH, m.Ta, m.P)
    row["Tsurface[C]"] = out.Tg
    # GEOtop: src/geotop/energy.balance.cc:2399 (EnergyFluxes)
    # EnergyFluxes sets Tv = Tg before anything else, so a point with no canopy
    # reports the ground temperature here, not no-data.
    row["Tvegetation[C]"] = out.Tv
    row["Tcanopyair[C]"] = b.Ts

    # surface energy balance
    row["Surface_Energy_balance[W/m2]"] = surface.reported_surface_EB(d, b, out.T1)
    row["Soil_heat_flux[W/m2]"] = out.GEF
    row["SWin[W/m2]"] = d.SWbeam + d.SWdiff
    row["SWbeam[W/m2]"] = d.SWbeam
    row["SWdiff[W/m2]"] = d.SWdiff
    row["LWin[W/m2]"] = d.LWin
    row["LWin_min[W/m2]"] = d.LWin_min
    row["LWin_max[W/m2]"] = d.LWin_max
    row["SWnet[W/m2]"] = d.SWnet
    row["LWnet[W/m2]"] = b.LWnet
    row["H[W/m2]"] = b.H
    row["LE[W/m2]"] = surface.reported_LE(b, out.T1)

    # GEOtop: src/geotop/energy.balance.cc:1131-1147
    # canopy.  Hg_unveg/LEg_unveg and Hg_veg/LEg_veg are the two halves of the
    # composite H/LE, each written unweighted and left at the EnergyFluxes
    # initialisation of 0 when its branch does not run -- so a point with fc=1
    # throughout reports Hg_unveg == 0 always, which is data, not a stale value.
    # Both latent columns use Levap(Tg) alone, never the sublimation-corrected
    # ``latent``.
    cs = out.canopy
    Levap = tb.Levap(out.Tg)
    row["Canopy_fraction[-]"] = out.fc
    row["LSAI[m2/m2]"] = out.LSAI
    row["z0veg[m]"] = out.z0veg
    row["d0veg[m]"] = out.d0veg
    row["Estored_canopy[W/m2]"] = (
        out.SWv + cs.LWv - cs.Hv - cs.LEv if cs is not None else 0.0)
    row["SWv[W/m2]"] = out.SWv
    row["LWv[W/m2]"] = cs.LWv if cs is not None else 0.0
    row["Hv[W/m2]"] = cs.Hv if cs is not None else 0.0
    row["LEv[W/m2]"] = cs.LEv if cs is not None else 0.0
    row["Hg_unveg[W/m2]"] = b.Hg0
    row["LEg_unveg[W/m2]"] = Levap * b.Eg0
    row["Hg_veg[W/m2]"] = b.Hg1
    row["LEg_veg[W/m2]"] = Levap * b.Eg1
    row["Trasp_canopy[mm]"] = (cs.Etrans * Dt) if cs is not None else 0.0
    for c in ("Water_on_canopy[mm]", "Snow_on_canopy[mm]",
              "Hup[W/m2]", "LEup[W/m2]"):
        row[c] = 0.0
    # The vapour flux is booked to exactly one compartment (energy.balance.cc:
    # 786-793): snow, else glacier, else soil.  Evap_surface is the soil one, so
    # it must be zero while either ice stack is present, not just snow.
    row["Evap_surface[mm]"] = out.evap_soil
    row["Qvegetation[-]"] = cs.Qv if cs is not None else NUMBER_NOVALUE
    row["Qsurface[-]"] = b.Qg
    row["Qair[-]"] = d.Qa
    row["Qcanopyair[-]"] = b.Qs
    row["LObukhov[m]"] = d.Lobukhov
    row["LObukhovcanopy[m]"] = cs.Locc if cs is not None else 0.0
    row["Decay_of_K_in_canopy[-]"] = cs.decay if cs is not None else 0.0
    row["Wind_speed_top_canopy[m/s]"] = cs.u_top if cs is not None else NUMBER_NOVALUE
    row["SWup[W/m2]"] = d.SWup
    row["LWup[W/m2]"] = b.LWup

    # snow and glacier state (end of step)
    row.update(state_columns(out))
    row["snow_melted[mm]"] = out.Melt
    row["snow_subl[mm]"] = out.Evap if out.had_snow else 0.0
    row["snow_blown_away[mm]"] = 0.0
    row["snow_subl_while_blown[mm]"] = 0.0

    row["glac_melted[mm]"] = out.Melt_glac
    row["glac_subl[mm]"] = out.Evap_glac if out.had_glac else 0.0
    return row


def state_columns(out: StepOut) -> Dict[str, float]:
    """The :data:`SNAPSHOT_COLUMNS` of a record: snow and glacier stacks at
    the end of the step, and no-data for the thaw and water-table depths
    (filled in at the end of the window)."""
    has_snow = out.depth > 0.0
    col: Dict[str, float] = {}
    col["snow_depth[mm]"] = out.depth
    col["snow_water_equivalent[mm]"] = out.swe
    col["snow_density[kg/m3]"] = (out.swe / out.depth * 1000.0) if has_snow else NUMBER_NOVALUE
    col["snow_temperature[C]"] = out.snow_T if not math.isnan(out.snow_T) else NUMBER_NOVALUE
    # GEOtop: src/geotop/output.cc:328-352
    # The whole glacier output block is gated by ``max_glac_layers>0``; with
    # the module off (every one of the 13 reference cases -- none sets
    # MaxGlacLayers) it never runs at all, so
    # every glacier column -- including the two "intensive" ones -- keeps
    # its zero-initialised default, not no-data. No-data only appears
    # *inside* an active module, for a step with the glacier module on but
    # temporarily no layers -- not reachable from any case this driver runs.
    has_glac = out.glac_depth > 0.0
    col["glac_depth[mm]"] = out.glac_depth
    col["glac_water_equivalent[mm]"] = out.gwe
    col["glac_density[kg/m3]"] = (out.gwe / out.glac_depth * 1000.0) if has_glac else 0.0
    col["glac_temperature[C]"] = out.glac_T if not math.isnan(out.glac_T) else 0.0
    # soil thaw / water table: accumulated apart, on the nominal step
    for c in ("lowest_thawed_soil_depth[mm]", "highest_thawed_soil_depth[mm]",
              "lowest_water_table_depth[mm]", "highest_water_table_depth[mm]"):
        col[c] = NUMBER_NOVALUE
    return col


def stale_row(last: Optional[Dict[str, float]], out: StepOut) -> Dict[str, float]:
    """The row of a step whose energy balance did not complete for the point:
    the fluxes and amounts of ``last``, the row of its last accepted step
    (all zero when there is none yet), with the state columns of ``out``.

    Fold it with the ``dt`` of the step ``last`` came from."""
    row = dict(last) if last is not None else {c: 0.0 for c in POINT_COLUMNS}
    row.update(state_columns(out))
    return row


# --- DtPlotPoint accumulation ---------------------------------------------
#
# One output row covers a reporting window of ``DtPlotPoint`` seconds, which
# may span several internal timesteps.  How the steps of a window combine is a
# property of the column, not of its physical dimension, so every column is
# listed explicitly in exactly one of the four tuples below (checked by
# :func:`_check_partition` at import time).

# Identity and timestamp: taken from the end of the window, not accumulated.
BOOK_COLUMNS = (
    "Date12[DDMMYYYYhhmm]", "JulianDayFromYear0[days]", "TimeFromStart[days]",
    "Simulation_Period", "Run", "IDpoint",
)

# Time-weighted mean: each internal step contributes ``value * dt /
# DtPlotPoint``, and over a complete window the weights sum to one.
# GEOtop: src/geotop/energy.balance.cc:1003-1148
# GEOtop: src/geotop/energy.balance.cc:1157-1200
WEIGHTED_COLUMNS = (
    "Wind_speed[m/s]", "Wind_direction[deg]", "Relative_Humidity[-]",
    "Pressure[mbar]", "Tair[C]", "Tdew[C]", "Tsurface[C]", "Tvegetation[C]",
    "Tcanopyair[C]", "Surface_Energy_balance[W/m2]", "Soil_heat_flux[W/m2]",
    "SWin[W/m2]", "SWbeam[W/m2]", "SWdiff[W/m2]",
    "LWin[W/m2]", "LWin_min[W/m2]", "LWin_max[W/m2]",
    "SWnet[W/m2]", "LWnet[W/m2]", "H[W/m2]", "LE[W/m2]",
    "Canopy_fraction[-]", "LSAI[m2/m2]", "z0veg[m]", "d0veg[m]",
    "Estored_canopy[W/m2]", "SWv[W/m2]", "LWv[W/m2]", "Hv[W/m2]", "LEv[W/m2]",
    "Hg_unveg[W/m2]", "LEg_unveg[W/m2]", "Hg_veg[W/m2]", "LEg_veg[W/m2]",
    "Qvegetation[-]", "Qsurface[-]", "Qair[-]", "Qcanopyair[-]",
    "LObukhov[m]", "LObukhovcanopy[m]", "Wind_speed_top_canopy[m/s]",
    "Decay_of_K_in_canopy[-]", "SWup[W/m2]", "LWup[W/m2]",
)

# Amounts [mm] already integrated over the internal step: added as they stand,
# with no time weighting.
# GEOtop: src/geotop/energy.balance.cc:983-1002
# GEOtop: src/geotop/energy.balance.cc:1149-1156
# GEOtop: src/geotop/energy.balance.cc:1201-1216
#
# The last six have no accumulation branch at all in GEOtop and stay at their
# zero initialisation for the whole run (canopy interception storage and the
# above-canopy H/LE are declared columns that nothing ever writes; the two
# blowing-snow terms are written only by the blowing-snow module, which no 1D
# case turns on).  Summing the zeros this writer emits reproduces that.
SUMMED_COLUMNS = (
    "Psnow_over_canopy[mm]", "Prain_over_canopy[mm]",
    "Psnow_under_canopy[mm]", "Prain_under_canopy[mm]",
    "Prain_rain_on_snow[mm]", "Evap_surface[mm]", "Trasp_canopy[mm]",
    "snow_melted[mm]", "snow_subl[mm]", "glac_melted[mm]", "glac_subl[mm]",
    "Water_on_canopy[mm]", "Snow_on_canopy[mm]", "Hup[W/m2]", "LEup[W/m2]",
    "snow_blown_away[mm]", "snow_subl_while_blown[mm]",
)

# State read once, at the end of the window: the snow and glacier stacks are
# re-summed over their layers there rather than averaged over the window, so
# the last internal step's value is the one that survives.
# GEOtop: src/geotop/output.cc:304-351
# GEOtop: src/geotop/output.cc:241-253 (the thaw-depth / water-table columns)
#
# The four thaw-depth / water-table columns do not belong here in GEOtop --
# they are time-weighted like the fluxes, but on the *nominal* step rather
# than the internal one.  This writer does not
# model them and emits no-data, which a snapshot carries through unchanged
# while a weighted sum would perturb in the last bits.
SNAPSHOT_COLUMNS = (
    "snow_depth[mm]", "snow_water_equivalent[mm]", "snow_density[kg/m3]",
    "snow_temperature[C]",
    "glac_depth[mm]", "glac_water_equivalent[mm]", "glac_density[kg/m3]",
    "glac_temperature[C]",
    "lowest_thawed_soil_depth[mm]", "highest_thawed_soil_depth[mm]",
    "lowest_water_table_depth[mm]", "highest_water_table_depth[mm]",
)


def _check_partition() -> None:
    """Every column is accumulated exactly one way, or writing it is a guess."""
    classified = (BOOK_COLUMNS + WEIGHTED_COLUMNS + SUMMED_COLUMNS
                  + SNAPSHOT_COLUMNS)
    missing = [c for c in POINT_COLUMNS if c not in classified]
    extra = [c for c in classified if c not in POINT_COLUMNS]
    dup = [c for c in classified if classified.count(c) > 1]
    if missing or extra or dup:
        raise AssertionError(
            f"accumulation classes do not partition POINT_COLUMNS: "
            f"missing={missing} extra={extra} duplicated={sorted(set(dup))}")


_check_partition()


class PointAccumulator:
    """Combine the internal steps of one ``DtPlotPoint`` window into one row.

    ``Dtplot`` is the window length in seconds.  Feed every committed internal
    step to :meth:`add` with the step's own length ``dt`` [s], then call
    :meth:`flush` once the window is full: it returns the row and resets the
    accumulator for the next window.
    """

    def __init__(self, idpoint: int, Dtplot: float) -> None:
        self.idpoint = idpoint
        self.Dtplot = Dtplot
        self.nsub = 0
        self.acc: Dict[str, float] = {c: 0.0 for c in POINT_COLUMNS}

    def add(self, row: Dict[str, float], dt: float) -> None:
        """Fold one internal step -- a :func:`build_row` row and its ``dt``
        [s] -- into the window being accumulated."""
        # GEOtop forms the whole per-step contribution first and only then
        # adds it, so the weighting divides the already-multiplied product and
        # the terms are summed in step order.  Keep both, they are what makes
        # the sum reproducible term by term.
        # GEOtop: src/geotop/energy.balance.cc:1023
        # GEOtop: src/geotop/output.cc:4927
        acc = self.acc
        for c in WEIGHTED_COLUMNS:
            acc[c] += row[c] * dt / self.Dtplot
        for c in SUMMED_COLUMNS:
            acc[c] += row[c]
        for c in SNAPSHOT_COLUMNS:
            acc[c] = row[c]
        self.nsub += 1

    def flush(self, date, JD: float, JD0: float) -> Dict[str, float]:
        """Close the window and return its row; ``date``/``JD`` are the window
        end.  Resets the accumulator."""
        row = dict(self.acc)
        row["Date12[DDMMYYYYhhmm]"] = date.strftime("%d/%m/%Y %H:%M")
        row["JulianDayFromYear0[days]"] = JD
        row["TimeFromStart[days]"] = JD - JD0
        row["Simulation_Period"] = 1.0
        row["Run"] = 1.0
        row["IDpoint"] = float(self.idpoint)
        self.acc = {c: 0.0 for c in POINT_COLUMNS}
        self.nsub = 0
        return row


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{NUMBER_NOVALUE:f}"
    return f"{v:f}"


# Position keywords, one per POINT_COLUMNS entry, for the same
# all-or-explicit-position mechanism as SoilAll/BasinAll (parameters.cc
# keyword codes 215-294): PointAll=1 forces every column in canonical
# order (12 of the 13 reference cases); explicit position keywords select
# a subset otherwise (only CostantMeteo).
POINT_BOOK_KEYWORDS: Tuple[str, ...] = (
    "DatePoint", "JulianDayFromYear0Point", "TimeFromStartPoint",
    "PeriodPoint", "RunPoint", "IDPointPoint",
    "PsnowPoint", "PrainPoint", "PsnowNetPoint", "PrainNetPoint",
    "PrainOnSnowPoint",
    "WindSpeedPoint", "WindDirPoint", "RHPoint", "AirPressPoint",
    "AirTempPoint", "TDewPoint", "TsurfPoint", "TvegPoint", "TCanopyAirPoint",
    "SurfaceEBPoint", "SoilHeatFluxPoint",
    "SWinPoint", "SWbeamPoint", "SWdiffPoint",
    "LWinPoint", "LWinMinPoint", "LWinMaxPoint",
    "SWNetPoint", "LWNetPoint", "HPoint", "LEPoint",
    "CanopyFractionPoint", "LSAIPoint", "z0vegPoint", "d0vegPoint",
    "EstoredCanopyPoint", "SWvPoint", "LWvPoint", "HvPoint", "LEvPoint",
    "HgUnvegPoint", "LEgUnvegPoint", "HgVegPoint", "LEgVegPoint",
    "EvapSurfacePoint", "TraspCanopyPoint",
    "WaterOnCanopyPoint", "SnowOnCanopyPoint",
    "QVegPoint", "QSurfPoint", "QAirPoint", "QCanopyAirPoint",
    "LObukhovPoint", "LObukhovCanopyPoint",
    "WindSpeedTopCanopyPoint", "DecayKCanopyPoint",
    "SWupPoint", "LWupPoint", "HupPoint", "LEupPoint",
    "SnowDepthPoint", "SWEPoint", "SnowDensityPoint", "SnowTempPoint",
    "SnowMeltedPoint", "SnowSublPoint", "SWEBlownPoint", "SWESublBlownPoint",
    "GlacDepthPoint", "GWEPoint", "GlacDensityPoint", "GlacTempPoint",
    "GlacMeltedPoint", "GlacSublPoint",
    "LowestThawedSoilDepthPoint", "HighestThawedSoilDepthPoint",
    "LowestWaterTableDepthPoint", "HighestWaterTableDepthPoint",
)


def write_point(path: str, rows: List[Dict[str, float]],
                columns: Optional[Sequence[Optional[str]]] = None) -> None:
    """Write ``rows`` (each keyed by :data:`POINT_COLUMNS`) as a GEOtop point
    table: comma-separated, ``%f`` numbers, one header line.

    ``columns`` overrides the default full :data:`POINT_COLUMNS` set/order --
    the explicit-position branch of the ``PointAll``/``*Point`` mechanism
    (see :func:`geotop_py.output.recorder._point_book_fields`); ``None`` entries are
    an unassigned output position, written as header ``"None"`` and value
    ``-9999`` (output.cc's own convention for the analogous basin/soil gap).
    """
    cols = columns if columns is not None else POINT_COLUMNS
    header = [c if c is not None else "None" for c in cols]
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(_fmt(r[c]) if c is not None else _fmt(NUMBER_NOVALUE)
                              for c in cols) + "\n")


DISCHARGE_COLUMNS: List[str] = [
    "DATE[day/month/year hour:min]", "t[days]", "JDfrom0", "JD",
    "Qtot[m3/s]", "Vsup/Dt[m3/s]", "Vsub/Dt[m3/s]", "Vchannel[m3]",
    "Qoutlandsup[m3/s]", "Qoutlandsub[m3/s]", "Qoutbottom[m3/s]",
]


def write_discharge(path: str, rows: List[Dict[str, float]]) -> None:
    """Write ``discharge.txt``: comma-separated, the last seven columns in
    ``%e`` (matching ``output.cc``'s own ``fprintf(...,"%e,%e,...")``,
    unlike every other table this driver writes, which is ``%f`` throughout).
    """
    with open(path, "w") as fh:
        fh.write(",".join(DISCHARGE_COLUMNS) + "\n")
        for r in rows:
            head = [r["DATE[day/month/year hour:min]"],
                   f"{r['t[days]']:f}", f"{r['JDfrom0']:f}", f"{r['JD']:f}"]
            tail = [f"{r[c]:e}" for c in DISCHARGE_COLUMNS[4:]]
            fh.write(",".join(head + tail) + "\n")


# GEOtop: src/geotop/parameters.cc:705-712
# basin.txt: the run-wide counterpart of point.txt, one row per DtPlotBasin
# window.  Keyed by GEOtop's own basin field index (0..26, ``ootot``) rather
# than header text: fields 7 and 8 share the literal header string
# "Prain_above_canopy[mm]" (parameters.cc assigns it to both oorainover and
# oosnowover -- a genuine label bug in GEOtop itself, kept verbatim), so
# header text cannot be used as a dict key.
BASIN_FIELD_NAMES: Tuple[str, ...] = (
    "Date12[DDMMYYYYhhmm]", "JulianDayFromYear0[days]", "TimeFromStart[days]",
    "Simulation_Period", "Run",
    "Prain_below_canopy[mm]", "Psnow_below_canopy[mm]",
    "Prain_above_canopy[mm]", "Prain_above_canopy[mm]",
    "Pnet[mm]", "Tair[C]", "Tsurface[C]", "Tvegetation[C]",
    "Evap_surface[mm]", "Transpiration_canopy[mm]",
    "LE[W/m2]", "H[W/m2]", "SW[W/m2]", "LW[W/m2]",
    "LEv[W/m2]", "Hv[W/m2]", "SWv[W/m2]", "LWv[W/m2]",
    "SWin[W/m2]", "LWin[W/m2]",
    "Mass_balance_error[mm]", "Mean_Time_Step[s]",
)
BASIN_NFIELD = len(BASIN_FIELD_NAMES)

# GEOtop: src/geotop/parameters.cc:1912-1953
# Position keywords, one per field index (0..26), for the same
# all-or-explicit-position mechanism as SoilAll/DateSoil etc, generalised to
# the full basin field set (keyword codes 295-321).
BASIN_BOOK_KEYWORDS: Tuple[str, ...] = (
    "DateBasin", "JulianDayFromYear0Basin", "TimeFromStartBasin",
    "PeriodBasin", "RunBasin",
    "PRainNetBasin", "PSnowNetBasin", "PRainBasin", "PSnowBasin", "PNetBasin",
    "AirTempBasin", "TSurfBasin", "TvegBasin",
    "EvapSurfaceBasin", "TraspCanopyBasin",
    "LEBasin", "HBasin", "SWNetBasin", "LWNetBasin",
    "LEvBasin", "HvBasin", "SWvBasin", "LWvBasin",
    "SWinBasin", "LWinBasin",
    "MassErrorBasin", "MeanTimeStep",
)

# GEOtop: src/geotop/energy.balance.cc:1247-1261
# GEOtop: src/geotop/energy.balance.cc:1270-1319
# Time-weighted mean over the window: each committed internal step
# contributes ``value * dt / DtPlotBasin``.
BASIN_WEIGHTED_FIELDS: Tuple[int, ...] = (10, 11, 12, 15, 16, 17, 18, 19, 20,
                                          21, 22, 23, 24)
# GEOtop: src/geotop/energy.balance.cc:1231-1246
# GEOtop: src/geotop/energy.balance.cc:1262-1269
# GEOtop: src/geotop/water.balance.cc:192-193
# GEOtop: src/geotop/geotop.cc:466 (the mean-timestep field)
# Added as they stand, no time weighting (water_balance overwrites rather than
# accumulates within one attempt, but the attempt-scratch/commit-sum split
# already makes that a plain sum across commits).
BASIN_SUMMED_FIELDS: Tuple[int, ...] = (5, 6, 7, 8, 9, 13, 14, 25, 26)


# GEOtop: src/geotop/energy.balance.cc:1231-1319
# GEOtop: src/geotop/energy.balance.cc:983-1216 (the opnt block)
# GEOtop: src/geotop/geotop.cc:466
def basin_values(Dt: float, Dtplot_basin: float, m: Meteo,
                 out: StepOut) -> Dict[int, float]:
    """One committed internal step's contribution to every accumulated
    ``basin.txt`` field (indices 5-26 of :data:`BASIN_FIELD_NAMES`) --
    the same quantities :func:`build_row` reads for ``point.txt``
    (the ``obsn`` block of PointEnergyBalance pulls from the same point-loop
    locals as the ``opnt`` block just above it). ``BASIN_WEIGHTED_FIELDS``
    entries are the raw per-step value here; :meth:`BasinAccumulator.add`
    applies the ``dt/Dtplot_basin`` weight. Field 26 (mean timestep) is the
    one field GEOtop weights before accumulating rather than after
    (``Dt * (Dt/Dtplot_basin)``), so it is folded in already and belongs to
    ``BASIN_SUMMED_FIELDS``, not the weighted set.
    """
    d = out.diag
    b = d.reported if d.reported is not None else d.breakdown(out.Tg)
    cs = out.canopy
    return {
        5: out.Prain_under, 6: out.Psnow_under,
        7: m.Prain, 8: m.Psnow,
        9: out.pnet,
        10: m.Ta, 11: out.Tg, 12: out.Tv,
        13: out.evap_soil,
        14: (cs.Etrans * Dt) if cs is not None else 0.0,
        15: surface.reported_LE(b, out.T1), 16: b.H,
        17: d.SWnet, 18: b.LWnet,
        19: (cs.LEv if cs is not None else 0.0),
        20: (cs.Hv if cs is not None else 0.0),
        21: out.SWv,
        22: (cs.LWv if cs is not None else 0.0),
        23: d.SWbeam + d.SWdiff, 24: d.LWin,
        25: out.wb_loss,
        26: Dt * (Dt / Dtplot_basin),
    }


def basin_values_unsolved() -> Dict[int, float]:
    """The contribution of a point whose energy step did not complete: zero
    for every accumulated field; the caller sets the water-balance fields
    (9, 25) and the mean timestep (26)."""
    return {i: 0.0 for i in BASIN_WEIGHTED_FIELDS + BASIN_SUMMED_FIELDS}


class BasinAccumulator:
    """Combine the internal steps of one ``DtPlotBasin`` window into one
    ``basin.txt`` row, keyed by basin field index (see :data:`BASIN_FIELD_NAMES`).

    Fields 0-4 (Date12/JulianDayFromYear0/TimeFromStart/Simulation_Period/Run)
    are never folded in by :meth:`add` -- GEOtop itself never assigns
    ``odb[ooperiod]``/``odb[oorun]`` anywhere (they stay at their zero
    initialisation for the whole run, a dead pair of columns kept only
    because the header exists), and Date12/JulianDayFromYear0/TimeFromStart
    are derived from the window's own end time, not accumulated.
    """

    def __init__(self, Dtplot_basin: float) -> None:
        self.Dtplot_basin = Dtplot_basin
        self.acc: Dict[int, float] = {i: 0.0 for i in range(BASIN_NFIELD)}

    def add(self, values: Dict[int, float], dt: float) -> None:
        acc = self.acc
        for i in BASIN_WEIGHTED_FIELDS:
            acc[i] += values[i] * dt / self.Dtplot_basin
        for i in BASIN_SUMMED_FIELDS:
            acc[i] += values[i]

    def flush(self, date, JDfrom0: float, JD0: float) -> Dict[int, float]:
        """Close the window and return its row; ``date`` is the window end
        (a ``datetime``), ``JDfrom0``/``JD0`` its Julian day and the run's
        start Julian day. Resets the accumulator."""
        row = dict(self.acc)
        row[0] = date
        row[1] = JDfrom0
        row[2] = JDfrom0 - JD0
        row[3] = 0.0
        row[4] = 0.0
        self.acc = {i: 0.0 for i in range(BASIN_NFIELD)}
        return row


# GEOtop: src/geotop/output.cc:3667-3686
# GEOtop: src/geotop/output.cc:3713-3732
# GEOtop: src/geotop/output.cc:850-886
def write_basin(path: str, rows: List[Dict[int, object]],
                obsn: Sequence[int]) -> None:
    """Write ``basin.txt``: comma-separated, ``%f`` throughout, columns in
    the order (and subset) given by ``obsn`` -- a sequence of basin field
    indices, one per output position, ``-1`` for a position with no field
    assigned (writes the header text ``"None"`` and value ``-9999``)."""
    header = [BASIN_FIELD_NAMES[i] if i >= 0 else "None" for i in obsn]
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            cells = []
            for i in obsn:
                if i < 0:
                    cells.append(_fmt(NUMBER_NOVALUE))
                elif i == 0:
                    cells.append(r[0].strftime("%d/%m/%Y %H:%M"))
                else:
                    cells.append(_fmt(r[i]))
            fh.write(",".join(cells) + "\n")
