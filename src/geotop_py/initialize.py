"""Initial state and static parameters of every point, from ``geotop.inpts``.

Translates the keywords into the model's own parameter objects (albedo,
vegetation, surface, Richards column, energy solver options) and builds each
point's starting column: soil layers at their initial water content, the
initial snowpack and its layering, surface and station state, and the
optional time-dependent vegetation series.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import constants as C
from .constants import STRING_NOVALUE, jdz
from .energy import albedo, surface
from .energy import vegetation as veg
from .energy.column import SolverOptions
from .io import horizon as io_horizon
from .io import points
from .io import soil as io_soil
from .io import vegfile as io_vegfile
from .meteo import forcing
from .point import state as point_state
from .point import step as point_step
from .snow import snow_layers
from .snow.state import SnowColumn
from .water import coupling, richards1d
from .water import initial_state as soil_init


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


@dataclass
class PointSetup:
    """Every point's starting state and static parameters, keyed by the
    internal 1..N point index."""
    cols: Dict[int, point_state.Column1D]
    states: Dict[int, surface.SurfaceState]
    stations: Dict[int, forcing.StationState]
    statics: Dict[int, surface.SurfaceStatics]
    richards: Optional[Dict[int, coupling.RichardsCoupling]]
    initial_profiles: Dict[int, point_step.LayerProfile]
    # TimeDependentVegetationParameterFile: one optional series per land cover
    # class, read once here and interpolated onto every step. The static
    # per-class keywords stay the fallback for classes with no file and for
    # steps outside a file's own time span, so they are kept alongside the
    # live parameters rather than replaced.
    veg_series: Dict[int, List[List[float]]]
    veg_static: Dict[int, veg.VegParams]
    veg_Dz: Dict[int, List[float]]


def init_points(pf, sim_dir: str, props: Dict[int, points.PointProperties],
                soilp: io_soil.SoilParameters,
                mcfg: forcing.MeteoDistrConfig,
                water_balance: bool) -> PointSetup:
    """Build the starting state of every point in ``props``."""
    cols: Dict[int, point_state.Column1D] = {}
    initial_profiles: Dict[int, point_step.LayerProfile] = {}
    states: Dict[int, surface.SurfaceState] = {}
    stations: Dict[int, forcing.StationState] = {}
    statics: Dict[int, surface.SurfaceStatics] = {}
    richards: Optional[Dict[int, coupling.RichardsCoupling]] = {} if water_balance else None
    horizon_cache: Dict[int, List[Tuple[float, float]]] = {}
    veg_stem = pf.string("TimeDependentVegetationParameterFile", STRING_NOVALUE)
    veg_series: Dict[int, List[List[float]]] = {}
    veg_static: Dict[int, veg.VegParams] = {}
    veg_Dz: Dict[int, List[float]] = {}
    for p in props:
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
        init_layers = snow_layers.initial_snow_layers(
            pf.number("InitSWE", 0, 0.0), pf.number("InitSnowDensity", 0, 200.0),
            Tsnow0, max_weq_snow, max_snow_layers, inf_snow_layers)
        init_snow = (SnowColumn.from_layers(init_layers, max=max_snow_layers)
                    if init_layers else SnowColumn(max=max_snow_layers, lnum=0, type=0))
        # GEOtop: src/geotop/input.cc:1801-1812
        # Then the layering rules, with Ta = -0.1 and the point's maxSWE, as for
        # every pack: they cap the SWE, drop a negligible pack and mark a thin
        # one as simplified (type 1), which from_layers leaves at type 2.
        snow_layers.snow_layer_combination(
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
        stations[p] = forcing.StationState(
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
    return PointSetup(cols, states, stations, statics, richards,
                      initial_profiles, veg_series, veg_static, veg_Dz)
