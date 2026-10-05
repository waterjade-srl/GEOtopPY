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

from . import constants as C
from . import dates, laws
from .constants import jdz
from .energy import albedo, surface
from .energy import vegetation as veg
from .energy.column import SolverOptions
from .io import horizon as io_horizon
from .io import meteo as io_meteo
from .io import parfile, points
from .io import soil as io_soil
from .io import vegfile as io_vegfile
from .io.parfile import STRING_NOVALUE
from .meteo import meteodata
from .meteo import step as ms
from .output.prof import SNOW_PROFILES, write_snow_profile, write_soil_profile
from .output.tabs import (
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
from .point import state as point_state
from .point import step as point_step
from .point import time_loop
from .snow import strati
from .snow.state import SnowColumn
from .water import coupling, richards1d, tables
from .water import init as soil_init


def _class_pf(pf, key: str, lu: Optional[int], default: float) -> float:
    """Per-land-cover-class keyword value for class ``lu`` (1-based).

    ``getDoubleVectorValueWithDefault(..., pUsePrevElement=true)``: a class
    beyond the end of the declared list inherits the last one given.
    """
    n = pf.components(key)
    i = int(lu) if lu is not None else 1
    j = min(max(i, 1), n) - 1
    return pf.number(key, j, default)


def _albpar_pf(pf, lu: Optional[int]) -> albedo.AlbedoParams:
    return albedo.AlbedoParams(
        avo=pf.number("FreshSnowReflVis", 0, 0.9),
        airo=pf.number("FreshSnowReflNIR", 0, 0.65),
        aep=pf.number("AlbExtParSnow", 0, 10.0),
        aging_vis=pf.number("SnowAgingCoeffVis", 0, 0.2),
        aging_nir=pf.number("SnowAgingCoeffNIR", 0, 0.5),
        avis_dry=_class_pf(pf, "SoilAlbVisDry", lu, 0.16),
        avis_wet=_class_pf(pf, "SoilAlbVisWet", lu, 0.08),
        anir_dry=_class_pf(pf, "SoilAlbNIRDry", lu, 0.33),
        anir_wet=_class_pf(pf, "SoilAlbNIRWet", lu, 0.16),
    )


# GEOtop: src/geotop/input.cc:775
# root() is called with slope hardcoded to 0.0 in the real source too; not a
# simplification here.
def _vegpar_pf(pf, lu: int, soil_D_mm: List[float]) -> veg.VegParams:
    hveg1 = _class_pf(pf, "VegHeight", 1, 1000.0)
    thres1 = _class_pf(pf, "ThresSnowVegUp", 1, hveg1)
    vp = veg.VegParams(
        Hveg=_class_pf(pf, "VegHeight", lu, 1000.0),
        z0thresveg=_class_pf(pf, "ThresSnowVegUp", lu, hveg1),
        z0thresveg2=_class_pf(pf, "ThresSnowVegDown", lu, thres1),
        LSAI=_class_pf(pf, "LSAI", lu, 1.0),
        cf=_class_pf(pf, "CanopyFraction", lu, 0.0),
        decay0=_class_pf(pf, "DecayCoeffCanopy", lu, 2.5),
        expveg=_class_pf(pf, "VegSnowBurying", lu, 1.0),
        root=_class_pf(pf, "RootDepth", lu, 300.0),
        rs=_class_pf(pf, "MinStomatalRes", lu, 60.0),
        R_vis=_class_pf(pf, "VegReflectVis", lu, 0.2),
        R_nir=_class_pf(pf, "VegReflNIR", lu, 0.2),
        T_vis=_class_pf(pf, "VegTransVis", lu, 0.2),
        T_nir=_class_pf(pf, "VegTransNIR", lu, 0.2),
        Ch=_class_pf(pf, "LeafAngles", lu, 0.0),
        cd=_class_pf(pf, "CanDensSurface", lu, 2.0),
    )
    Dz = [0.0] + list(soil_D_mm)
    nsoil = len(soil_D_mm)
    vp.n_transp = veg.n_transp_layers(Dz, nsoil)
    vp.root_frac = veg.root_fraction(vp.n_transp + 1, vp.root, 0.0, Dz)
    return vp


def _statics_pf(pf, point: points.PointProperties,
                horizon: List[Tuple[float, float]], theta_sup: float,
                soil_D_mm: List[float], soil_res: float,
                soil_sat: float) -> surface.SurfaceStatics:
    par = _albpar_pf(pf, point.LC)
    # The bare-ground albedo interpolates between its dry and saturated values
    # over the top layer's *retention* range: the endpoints are the soil's own
    # residual and saturated water contents, not 0 and the current content.
    # GEOtop: src/geotop/energy.balance.cc:476-481 (pa(sy,jres,1), pa(sy,jsat,1))
    par.soil_res = soil_res
    par.soil_sat = soil_sat
    # GEOtop: src/geotop/input.cc:1462-1473
    # Layers reached by bare-soil evaporation: a
    # hardcoded 100mm depth, not the whole column -- summing over every
    # layer (SurfaceStatics.n_evap's own None default) drags in resistance
    # contributions from meters of soil evaporation never actually reaches.
    n_evap = veg.n_transp_layers([0.0] + list(soil_D_mm), len(soil_D_mm),
                                 depth=C.z_evap)
    # GEOtop: src/geotop/parameters.cc:1246-1248
    # ``ThresSnowSoilRough`` has no fixed default: when the keyword is absent it
    # takes the *soil roughness* of land class 1, the same chaining the
    # vegetation thresholds get above.  A fixed 10 mm
    # is only right for a site that also leaves SoilRoughness at its default;
    # Bro sets it to 135 mm, so the snow/soil roughness switch was flipping at
    # 10 mm of snow instead of 135 -- a factor-10 jump in the aerodynamic
    # resistance in the middle of a run.
    z0_soil_mm = _class_pf(pf, "SoilRoughness", 1, 10.0)
    return surface.SurfaceStatics(
        lat=point.latitude, lon=point.longitude,
        ST=pf.number("StandardTimeSimulation", 0, 0.0),
        slope=point.slope, aspect=point.aspect, sky=point.sky,
        horizon=horizon, albpar=par, theta_sup=theta_sup,
        tres_up_albedo=pf.number("MinPrecToRestoreFreshSnowAlbedo", 0, 10.0),
        lw_state=int(pf.number("LWinParameterization", 0, 9.0)),
        Lozone=pf.number("Lozone", 0, 0.3),
        alpha_iqbal=pf.number("AngstromAlpha", 0, 1.3),
        beta_iqbal=pf.number("AngstromBeta", 0, 0.1),
        k1=pf.number("KonzelmannA", 0, 0.484),
        k2=pf.number("KonzelmannB", 0, 8.0),
        zmu=pf.number("MeteoStationWindVelocitySensorHeight", 0, 10.0),
        zmt=pf.number("MeteoStationTemperatureSensorHeight", 0, 2.0),
        MO=int(pf.number("MoninObukhov", 0, 1.0)),
        maxiter=int(pf.number("BusingerMaxIter", 0, 5.0)),
        z0_snow=0.001 * pf.number("SnowRoughness", 0, 0.1),
        z0_soil=0.001 * _class_pf(pf, "SoilRoughness", point.LC, 10.0),
        z0_thres=_class_pf(pf, "ThresSnowSoilRough", point.LC, z0_soil_mm),
        eps_snow=pf.number("SnowEmissiv", 0, 0.98),
        eps_soil=_class_pf(pf, "SoilEmissiv", point.LC, 0.99),
        vegpar=_vegpar_pf(pf, point.LC, soil_D_mm),
        canopy_opt=veg.CanopyOptions(
            maxiter_canopy=int(pf.number("CanopyMaxIter", 0, 3.0)),
            maxiter_Ts=int(pf.number("TsMaxIter", 0, 2.0)),
            maxiter_Loc=int(pf.number("LocMaxIter", 0, 3.0)),
            maxiter_Businger=int(pf.number("BusingerMaxIter", 0, 5.0)),
            stabcorr_incanopy=int(pf.number("CanopyStabCorrection", 0, 1.0)),
            MO=int(pf.number("MoninObukhov", 0, 1.0))),
        n_evap=n_evap,
        soil_D_mm=list(soil_D_mm),
    )


def build_richards_column(pf, soilp: io_soil.SoilParameters,
                          point: points.PointProperties) -> richards1d.RichardsColumn:
    nsoil = soilp.nlayers
    n_spinup = int(pf.number("SpinUpLayerBottom", 0, 10000.0))
    nl = min(n_spinup, nsoil)
    dz = [0.0] + [soilp.pa[jdz][l] for l in range(1, nsoil + 1)]
    Z = richards1d.node_depths(dz, point.slope)
    return richards1d.RichardsColumn(
        dz=dz, Z=Z, pa=soilp.pa, nl=nl, slope_deg=point.slope, area=1.0,
        imp=pf.number("FrozenSoilHydrCondReduction", 0, 7.0),
        k_to_ksat=pf.number("MinRatioKactualToKSat", 0, 0.0),
        free_drainage_bottom=bool(pf.number("FreeDrainageAtBottom", 0, 0.0)),
        free_drainage_lateral=pf.number("FreeDrainageAtLateralBorder", 0, 1.0),
        # Per point, not the bare keyword: the point table's own column wins
        # where it is given, and only falls back to the keyword when absent.
        # GEOtop: src/geotop/input.cc:3442-3443
        # GEOtop: src/geotop/input.cc:3575
        bc_depth_free_surface=point.BC_DepthFreeSurface,
    )


def _z_boundary(pf, soilp) -> float:
    """Bottom boundary depth [m] below the last soil node, GEOtop semantics.

    The keyword is read in [mm], the whole soil column depth (soil type 1's
    own layer thicknesses) is subtracted from it, a negative result is fatal,
    and only then the conversion to [m] happens.  The subtraction is what
    makes a *given* depth meaningful (it is measured from the column bottom,
    not the surface); with the 1e20 default the subtraction is absorbed by
    double rounding while the mm->m conversion is not -- the two orders
    together are what reproduce the C++ bottom heat leak bit for bit."""
    # GEOtop: src/geotop/input.cc:1267-1282
    from .errors import GeotopAbort
    zb = pf.number("ZeroTempAmplitDepth", 0, 1.0e20)
    for l in range(1, soilp.nlayers + 1):
        zb -= soilp.pa[jdz][l]
    if zb < 0:
        raise GeotopAbort(
            "Z at which 0 annual temperature takes place is not lower than "
            "the soil column")
    return zb * 1.0e-3


# GEOtop: src/geotop/parameters.cc:1320-1324
def build_solver_options(pf) -> SolverOptions:
    """Read GEOtop's energy-solver stopping criteria.

    Preserve the original tolerances: a tighter tolerance changes the stopping
    point and therefore the numerical result even for the same equation.
    """
    return SolverOptions(
        tol_energy=pf.number("HeatEqTol", 0, 1.0e-4),
        maxiter_energy=int(pf.number("HeatEqMaxIter", 0, 500.0)),
        min_lambda_en=pf.number("MinLambdaEnergy", 0, 1.0e-5),
        max_times_min_lambda_en=int(pf.number("MaxTimesMinLambdaEnergy", 0, 0.0)),
        exit_lambda_min_en=bool(pf.number("ExitMinLambdaEnergy", 0, 0.0)),
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


ProgressCallback = Callable[[int, int, datetime], None]

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
    """The four soil diagnostics for one column's current state (``tables.py``,
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

    lowest_thawed = tables.find_activelayerdepth_up(T, th, thi, res, dz_mm)
    highest_thawed = tables.find_activelayerdepth_dw(T, th, thi, res, dz_mm)
    Z = lowest_thawed
    lowest_wt = tables.find_watertabledepth_up(Z, Ptot, dz_mm)
    highest_wt = tables.find_watertabledepth_dw(Z, Ptot, dz_mm)
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
    m: ms.Meteo
    out: point_step.StepOut
    water_input: Optional[point_step.WaterInput] = None


@dataclass
class _Trial:
    """A whole-domain trial: the points it reached, in order, and the
    per-point values that outlive it, as they stood when it ended."""
    points: Dict[int, _PointTrial]
    last_energy: Dict[int, Optional[Tuple[ms.Meteo, point_step.StepOut, float]]]
    last_water: Dict[int, Tuple[float, float]]
    water_done: bool = False   # the water balance ran for every point and converged


def run_simulation(sim_dir: str, suffix: str = "", verbose: bool = True,
                   progress: Optional[ProgressCallback] = None,
                   min_Dt: float = 60.0) -> dict:
    """Run the case described by ``sim_dir/geotop.inpts`` end to end and
    write GEOtop-format outputs. Returns ``{point: records}``."""
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
    lapse = ms.LapseRates.from_parfile(pf)
    mcfg = ms.MeteoDistrConfig.from_parfile(pf)

    stem = pf.string("MeteoFile")
    meteo_paths = sorted(glob.glob(os.path.join(sim_dir, stem + "[0-9]*.txt")))
    mopt = io_meteo.MeteoOptions.from_parfile(pf, 1)
    table = io_meteo.load(meteo_paths[0], io_meteo.column_names(pf.strings), mopt)
    Z_station = pf.number("MeteoStationElevation", 0, 0.0)

    cols: Dict[int, point_state.Column1D] = {}
    initial_profiles: Dict[int, point_step.LayerProfile] = {}
    states: Dict[int, surface.SurfaceState] = {}
    stations: Dict[int, ms.StationState] = {}
    statics: Dict[int, surface.SurfaceStatics] = {}
    richards: Optional[Dict[int, coupling.RichardsCoupling]] = {} if water_balance else None
    horizon_cache: Dict[int, List[Tuple[float, float]]] = {}
    # TimeDependentVegetationParameterFile: one optional series per land cover
    # class, read once here and interpolated onto every step below. The static
    # per-class keywords stay the fallback for classes with no file and for
    # steps outside a file's own time span, so they are kept alongside the
    # live parameters rather than replaced.
    veg_stem = pf.string("TimeDependentVegetationParameterFile", STRING_NOVALUE)
    veg_series: Dict[int, List[List[float]]] = {}
    veg_static: Dict[int, veg.VegParams] = {}
    veg_Dz: Dict[int, List[float]] = {}
    for p in point_ids:
        pt = props[p]
        hor_stem = pf.string("HorizonPointFile", STRING_NOVALUE)
        hor_stem_path = hor_stem if hor_stem == STRING_NOVALUE else os.path.join(sim_dir, hor_stem)
        hor = horizon_cache.setdefault(
            pt.horizon_point,
            io_horizon.read_horizon(hor_stem_path, pt.horizon_point,
                                    io_horizon.column_names(pf.strings)))
        init = soil_init.initial_soil_state(
            soilp.pa, soilp.nlayers, soilp.init_water_table_depth, pt.slope)
        soil_layers = point_state.soil_layers_from_pa(soilp.pa, soilp.nlayers, init)
        soil_D_mm = [soilp.pa[jdz][l] for l in range(1, soilp.nlayers + 1)]
        # 1e-3 * d, not d / 1000: GEOtop builds the energy column's layer
        # depths as 1.E-3*pa(jdz,l) and the two round differently for some d.
        # GEOtop: src/geotop/energy.balance.cc:642
        soil_D_m = [0.0] + [1.0e-3 * d for d in soil_D_mm]

        # GEOtop: src/geotop/parameters.cc:1376-1399
        # Snow layer allocation: the total layer
        # count and the merge/split reference positions are *computed* from
        # three keywords, not read directly -- a naive per-layer default
        # (Column1D.inf_snow_layers' own fallback) targets the wrong pairs
        # on a real run.
        max_weq_snow = pf.number("MaxWaterEqSnowLayerContent", 0, 5.0)
        n_middle = int(pf.number("MaxSnowLayersMiddle", 0, 2.0))
        swe_bottom = pf.number("SWEbottom", 0, 20.0)
        swe_top = pf.number("SWEtop", 0, 20.0)
        n_bottom = int(swe_bottom // max_weq_snow)
        n_top = int(swe_top // max_weq_snow)
        max_snow_layers = n_bottom + n_top + n_middle
        inf_snow_layers = [0] + [n_bottom + i for i in range(1, n_middle + 1)]

        # GEOtop: src/geotop/input.cc:1699-1797
        # Initial snowpack: most cases start bare
        # (InitSWE defaults to 0), but a few pin a real observed pack --
        # skipping this reads the column as empty until the first snowfall
        # commits, drifting the whole run from the reference from step one.
        Tsnow0 = pf.number("InitSnowTemp", 0, -3.0)
        init_layers = strati.initial_snow_layers(
            pf.number("InitSWE", 0, 0.0), pf.number("InitSnowDensity", 0, 200.0),
            Tsnow0, max_weq_snow, max_snow_layers, inf_snow_layers)
        init_snow = (SnowColumn.from_layers(init_layers, max=max_snow_layers)
                    if init_layers else SnowColumn(max=max_snow_layers, lnum=0, type=0))
        # GEOtop: src/geotop/input.cc:1801-1812
        # Then the layering rules, with Ta = -0.1 and the point's maxSWE, as for
        # every pack: they cap the SWE, drop a negligible pack and mark a thin
        # one as simplified (type 1), which from_layers leaves at type 2.
        strati.snow_layer_combination(
            init_snow, pf.number("AlphaSnow", 0, 1.0e5), -0.1, inf_snow_layers,
            max_weq_snow, pt.maxSWE)

        col = point_state.Column1D(
            snow=init_snow,
            soil=soil_layers, soil_D=soil_D_m, soil_T=[0.0] + init.T[1:],
            alpha_snow=pf.number("AlphaSnow", 0, 1.0e5),
            # k_thermal's snow branch selector (laws.py): 1 = Cosenza (the
            # same quadratic-parallel mixing law the soil uses), 2 = Sturm
            # (1997), 3 = Jordan (1991). A keyword does select among the
            # three, and its default is 1 -- Jordan is never reached unless
            # the case asks for it, and none of the 13 reference cases does.
            # GEOtop: src/geotop/parameters.cc:2255
            snow_conductivity=int(pf.number("SnowThermalConductivityPar", 0, 1.0)),
            slope=pt.slope,
            Tboundary=pf.number("ZeroTempAmplitTemp", 0, 20.0),
            # ZeroTempAmplitDepth is read in [mm], has the whole soil column
            # depth subtracted from it (the boundary depth is measured from
            # the column bottom), then converted to [m].  With the 1e20
            # default the subtraction vanishes in double rounding
            # (ULP(1e20) ~ 1.6e4 mm, far above any column depth) but the
            # mm->m conversion does not: the boundary sits at 1e17 m, and
            # passing 1e20 m here shrinks the bottom heat leak a thousand
            # times -- an invisible per-step difference that seeds a slowly
            # growing ULP-level state drift, amplified by every capped-melt
            # Newton into visible flux errors.
            # GEOtop: src/geotop/input.cc:1267-1282
            Zboundary=_z_boundary(pf, soilp),
            Fboundary=pf.number("BottomBoundaryHeatFlux", 0, 0.0),
            max_weq_snow=max_weq_snow, maxSWE=pt.maxSWE,
            inf_override=inf_snow_layers,
            # GEOtop: src/geotop/parameters.cc:1319
            # Column1D's own dataclass default (1) is NOT GEOtop's: the real
            # default is 0, a massless skin above material node 1
            # (HighestNodeCorrespondsToLayer) -- with
            # nsurface=1 the reported "surface" is the top *material* node,
            # which has real heat capacity and responds far too slowly to
            # the SW/LW forcing. Confirmed against
            # PureDrainage: Tsurface barely moves under nsurface=1 while the
            # reference tracks the forcing closely.
            nsurface=int(pf.number("HighestNodeCorrespondsToLayer", 0, 0.0)),
            solver_opt=build_solver_options(pf),
        )
        cols[p] = col
        states[p] = surface.SurfaceState(snowage=albedo.initial_snow_age(
            pf.number("InitSnowAge", 0, 0.0), bool(init_layers), Tsnow0))
        stations[p] = ms.StationState(
            prev_Ta=mcfg.Tair_default, prev_RH=mcfg.RH_default,
            prev_wind_speed=mcfg.V_default, prev_winddir=mcfg.Vdir_default)
        theta_sup = soil_layers[1].sat - soil_layers[1].thi0
        statics[p] = _statics_pf(pf, pt, hor, theta_sup, soil_D_mm,
                                 soil_res=soil_layers[1].res,
                                 soil_sat=soil_layers[1].sat)
        # A second, independent copy of the same static parameters: the one
        # inside statics[p] becomes live state once a series overwrites it.
        veg_static[p] = _vegpar_pf(pf, pt.LC, soil_D_mm)
        veg_Dz[p] = [0.0] + list(soil_D_mm)
        lu = int(pt.LC)
        if veg_stem != STRING_NOVALUE and lu not in veg_series:
            series = io_vegfile.load(os.path.join(sim_dir, veg_stem), lu)
            if series is not None:
                veg_series[lu] = series
        if richards is not None:
            rcol = build_richards_column(pf, soilp, pt)
            richards[p] = coupling.RichardsCoupling(
                col=rcol, params=richards1d.RichardsParams.from_parfile(pf),
                state=init.as_richards_state())
        initial_profiles[p] = point_step.snapshot_profile(
            col, richards[p] if richards is not None else None)

    Dtplot = dtplot_point(pf, Dt)
    state_pixel = Dtplot > 1.0e-5
    recs: Dict[int, List[Tuple]] = {p: [] for p in point_ids}
    # One entry per emitted row: the accumulated point row, and the record of
    # the internal step the row closes on -- the profile files report that
    # step's state directly rather than an accumulation of the window.
    rows: Dict[int, List[Dict[str, float]]] = {p: [] for p in point_ids}
    plotted: Dict[int, List[Tuple]] = {p: [] for p in point_ids}
    # GEOtop identifies every output file/row by the point's own ID (e.g.
    # Jungfraujoch's listpoints.txt IDs are 32/33, not 1/2 -- point_ids
    # above is a purely internal 1..N loop index, coincidentally equal to
    # the real ID on every case except this one, which is why the mismatch
    # went unnoticed until now).
    accs = {p: PointAccumulator(props[p].ID, Dtplot) for p in point_ids}
    # GEOtop: src/geotop/output.cc:216-253
    # The four thaw/water-table diagnostics accumulate on the *nominal* step
    # (par->Dt), never the internal sub-step -- a
    # different cadence from every other WEIGHTED_COLUMNS entry, which is
    # why they cannot go through PointAccumulator itself and get their own
    # small one here.
    tables_acc = {p: [0.0, 0.0, 0.0, 0.0] for p in point_ids}
    # GEOtop: src/geotop/output.cc:231-239
    # The Tzav family (SoilAveragedTempProfileFile/SoilAveragedIceContent
    # ProfileFile/SoilAveragedLiqContentProfileFile) shares the exact same
    # nominal-step cadence and Dt/Dtplot weight as the four scalars above,
    # just per soil node instead of a single value.
    soilT_acc = {p: [0.0] * (cols[p].nsoil() + 1) for p in point_ids}
    soilth_acc = {p: [0.0] * (cols[p].nsoil() + 1) for p in point_ids}
    soilthi_acc = {p: [0.0] * (cols[p].nsoil() + 1) for p in point_ids}
    avg_plotted: Dict[int, List[Tuple]] = {p: [] for p in point_ids}
    # GEOtop: src/geotop/output.cc:140-209
    # discharge.txt: an independent cadence (DtPlotDischarge) and, for a
    # point simulation with no channel network, only two nonzero terms --
    # Vbottom/Vlat, raw volumes [m3] summed (not Dt-weighted) across every
    # committed internal sub-step, divided by Dtplot_discharge only at
    # write time to report a rate.
    Dtplot_discharge = dtplot_discharge(pf, Dt)
    state_discharge = Dtplot_discharge > 1.0e-5
    discharge_acc = {p: [0.0, 0.0] for p in point_ids}
    discharge_rows: Dict[int, List[Dict[str, float]]] = {p: [] for p in point_ids}
    t_discharge = 0.0
    # basin.txt: same independent-cadence pattern as discharge.txt (its own
    # DtPlotBasin, not DtPlotPoint), but a mix of Dt/Dtplot_basin-weighted
    # averages and raw sums in one row instead of two raw sums -- see
    # BasinAccumulator. Also a run-wide file like discharge.txt: only
    # point_ids[0]'s accumulator is ever written; none of the reference cases
    # that set BasinOutputFile simulate more than one point.
    Dtplot_basin = dtplot_basin(pf, Dt)
    state_basin = Dtplot_basin > 1.0e-5
    basin_accs = {p: BasinAccumulator(Dtplot_basin) for p in point_ids}
    basin_rows: Dict[int, List[Dict[int, object]]] = {p: [] for p in point_ids}
    t_basin = 0.0
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
    last_energy: Dict[int, Optional[Tuple[ms.Meteo, point_step.StepOut, float]]] = {
        p: None for p in point_ids}
    last_water: Dict[int, Tuple[float, float]] = {p: (0.0, 0.0) for p in point_ids}
    t_elapsed = 0.0
    t_point = 0.0
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
                row, istart_by_point[p] = ms.interpolate_station_row(
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
                meteo[p] = ms.assemble_meteo(row, dt, Z_station, props[p], lapse,
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
            for s in substeps:
                tr = s.payload.points.get(p)
                accepted = tr is not None and tr.out.converged
                t_commit = t_elapsed + s.t_begin + s.dt_advanced
                date = init_dt + timedelta(days=t_commit / 86400.0)
                JD_commit = JD0 + t_commit / 86400.0
                if tr is not None:
                    recs[p].append((date, JD_commit, tr.m, tr.out))
                if state_pixel:
                    if accepted:
                        accs[p].add(
                            build_row(props[p].ID, date, JD_commit, JD0, s.dt_run,
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
                                     point_step.failed_step_out(
                                         cols[p], None,
                                         richards[p] if richards is not None
                                         else None))
                        last = s.payload.last_energy[p]
                        if last is None:
                            accs[p].add(stale_row(None, state_out), s.dt_run)
                        else:
                            lm, lout, ldt = last
                            accs[p].add(stale_row(
                                build_row(props[p].ID, date, JD_commit, JD0, ldt,
                                          lm, lout),
                                state_out), ldt)
                if state_discharge and tr is not None:
                    dacc = discharge_acc[p]
                    dacc[0] += tr.out.wb_vbottom
                    dacc[1] += tr.out.wb_vlat
                if state_basin:
                    if accepted:
                        bv = basin_values(s.dt_run, Dtplot_basin, tr.m, tr.out)
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
                    bv[26] = s.dt_advanced * (s.dt_advanced / Dtplot_basin)
                    basin_accs[p].add(bv, s.dt_run)
            if state_pixel:
                lo_th, hi_th, lo_wt, hi_wt = _thaw_and_watertable_depths(
                    cols[p], richards[p] if richards is not None else None)
                w = Dt / Dtplot
                acc = tables_acc[p]
                acc[0] += lo_th * w
                acc[1] += hi_th * w
                acc[2] += lo_wt * w
                acc[3] += hi_wt * w
                col = cols[p]
                Tacc, thacc, thiacc = soilT_acc[p], soilth_acc[p], soilthi_acc[p]
                for k in range(1, col.nsoil() + 1):
                    Tacc[k] += col.soil_T[k] * w
                    thacc[k] += col.soil[k].th0 * w
                    thiacc[k] += col.soil[k].thi0 * w
        t_elapsed += Dt
        # GEOtop: src/geotop/output.cc:216-219
        # GEOtop: src/geotop/output.cc:257
        # GEOtop: src/geotop/output.cc:773
        # GEOtop: src/geotop/output.cc:810
        # The reporting clock ticks once per *nominal* step, however many sub-steps that
        # step needed, and a row is emitted only when the clock lands on
        # DtPlotPoint exactly. An interval that is not a whole multiple of the
        # nominal timestep therefore emits nothing at all, here as there.
        if state_pixel:
            t_point += Dt
            if abs(t_point - Dtplot) < 1.0e-5:
                plot_dt = init_dt + timedelta(seconds=t_elapsed)
                plot_JD = JD0 + t_elapsed / 86400.0
                for p in point_ids:
                    row_out = accs[p].flush(plot_dt, plot_JD, JD0)
                    lo_th, hi_th, lo_wt, hi_wt = tables_acc[p]
                    row_out["lowest_thawed_soil_depth[mm]"] = lo_th
                    row_out["highest_thawed_soil_depth[mm]"] = hi_th
                    row_out["lowest_water_table_depth[mm]"] = lo_wt
                    row_out["highest_water_table_depth[mm]"] = hi_wt
                    rows[p].append(row_out)
                    plotted[p].append(
                        (plot_dt, plot_JD, recs[p][-1][2], recs[p][-1][3]))
                    tables_acc[p] = [0.0, 0.0, 0.0, 0.0]
                    avg_prof = point_step.LayerProfile(
                        Dz=[], T=[], wice=[], wliq=[],
                        soil_T=soilT_acc[p][1:], soil_th=soilth_acc[p][1:],
                        soil_thi=soilthi_acc[p][1:], soil_psi=[], soil_ptot=[])
                    avg_plotted[p].append(
                        (plot_dt, plot_JD, None, _AvgStepOut(avg_prof)))
                    nsoil = cols[p].nsoil()
                    soilT_acc[p] = [0.0] * (nsoil + 1)
                    soilth_acc[p] = [0.0] * (nsoil + 1)
                    soilthi_acc[p] = [0.0] * (nsoil + 1)
                t_point = 0.0
        if state_discharge:
            t_discharge += Dt
            if abs(t_discharge - Dtplot_discharge) < 1.0e-5:
                date_str = (init_dt + timedelta(seconds=t_elapsed)).strftime(
                    "%d/%m/%Y %H:%M")
                JDfrom0 = JD0 + t_elapsed / 86400.0
                JD_yr, _year = dates.JDfrom0_to_JDandYear(JDfrom0)
                for p in point_ids:
                    Vbottom, Vlat = discharge_acc[p]
                    discharge_rows[p].append({
                        "DATE[day/month/year hour:min]": date_str,
                        "t[days]": t_elapsed / 86400.0,
                        "JDfrom0": JDfrom0,
                        "JD": JD_yr,
                        "Qtot[m3/s]": 0.0,
                        "Vsup/Dt[m3/s]": 0.0,
                        "Vsub/Dt[m3/s]": 0.0,
                        "Vchannel[m3]": 0.0,
                        "Qoutlandsup[m3/s]": 0.0,
                        "Qoutlandsub[m3/s]": Vlat / Dtplot_discharge,
                        "Qoutbottom[m3/s]": Vbottom / Dtplot_discharge,
                    })
                    discharge_acc[p] = [0.0, 0.0]
                t_discharge = 0.0
        if state_basin:
            t_basin += Dt
            if abs(t_basin - Dtplot_basin) < 1.0e-5:
                plot_dt = init_dt + timedelta(seconds=t_elapsed)
                JDfrom0 = JD0 + t_elapsed / 86400.0
                for p in point_ids:
                    basin_rows[p].append(basin_accs[p].flush(plot_dt, JDfrom0, JD0))
                t_basin = 0.0
        if progress is not None:
            progress(i + 1, nsteps, init_dt + timedelta(seconds=t_elapsed))

    if state_discharge:
        # GEOtop: src/geotop/output.cc:2557
        # Gated on state_discharge and the keyword both, same as every other
        # output file here; discharge is a run-wide total (one file, not
        # per-point), but every point of a 1D case shares the same single
        # column, so a single-point run's own discharge is what gets written.
        q_kw = pf.string("DischargeFile", STRING_NOVALUE)
        if q_kw != STRING_NOVALUE:
            write_discharge(_target_global(sim_dir, q_kw, suffix),
                            discharge_rows[point_ids[0]])

    if state_basin:
        bas_kw = pf.string("BasinOutputFile", STRING_NOVALUE)
        if bas_kw != STRING_NOVALUE:
            obsn = _basin_book_fields(pf)
            write_basin(_target_global(sim_dir, bas_kw, suffix),
                       basin_rows[point_ids[0]], obsn)

    prof_keys = {"snowDepth": "SnowDepthLayersFile",
                 "snowTemp": "SnowTempProfileFile",
                 "snowThetaIce": "SnowIceContentProfileFile",
                 "snowThetaW": "SnowLiqContentProfileFile"}
    if not state_pixel:
        return recs
    for p in point_ids:
        col = cols[p]
        pid = props[p].ID
        # GEOtop: src/geotop/parameters.cc:225-230
        # Every output-file keyword defaults to string_novalue (absent), same as HorizonPointFile
        # above; "output-tabs/point" was never GEOtop's own default, it
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
            nonconv = sum(1 for r in recs[p] if not r[3].converged)
            # not _target: it creates the directory, and with no
            # PointOutputFile there is none to name
            where = (os.path.join(sim_dir, os.path.dirname(pt_kw) + suffix)
                     if pt_kw != STRING_NOVALUE else "no point file")
            print(f"point {p}: {len(recs[p])} steps, {len(rows[p])} rows, "
                  f"{nonconv} non-conv -> {where}")
    return recs
