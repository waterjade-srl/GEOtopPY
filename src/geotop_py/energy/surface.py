"""Surface energy-balance assembly from meteorological forcing and column state.

Combines the four ported forcing components (R1 shortwave `rad`, R2 longwave
`rad`, R3 albedo `albedo`, turbulence `turbulence`) into the single Neumann top
flux GEOtop injects into the column solver (``SolvePointEnergyBalance``):

    EB(Tg)     = LWin - eps*sky*SB(Tg) - H(Tg) - lat(Tg)*E(Tg) + SWlayer[0]
    dEB_dT(Tg) = -eps*sky*dSB_dT(Tg) - dH_dT - lat(Tg)*dE_dT

The turbulent resistances are frozen once per step (as ``EnergyFluxes`` does),
then ``EB`` varies with ``Tg`` through the radiative and turbulent terms. The
shortwave absorbed *below* the surface is distributed over the snow layers by
``rad_snow_absorption`` and passed to the solver as ``layer_source``; the surface
share ``SWlayer[0]`` rides in ``EB``.

``surface_forcing`` also carries the two lagged/stateful couplings discovered
while validating the components: the snow-age evolution (albedo) and the
terrain-reflected shortwave ``SWrefl_surr`` (previous step's ``SWup``), which
re-enters the diffuse stream.
"""

# GEOtop: src/geotop/energy.balance.cc:1494-1499

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

from .. import constants as C
from .. import laws
from . import albedo, rad
from . import turbulence as tb
from . import vegetation as veg


# ---------------------------------------------------------------------------
# shortwave penetration into snow (radiation.cc)
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/radiation.cc:582-614
def rad_snow_absorption(R: float, ecol, ns: int) -> List[float]:
    """Distribute net shortwave ``R`` over the column nodes (GEOtop
    ``rad_snow_absorption``). Returns ``SWlayer[0..n+1]`` in column indexing:
    ``[0]`` = surface (folded into EB), ``[1..ns]`` = snow layers (1 = top),
    ``[ns+1]`` = residual reaching the first soil node.

    With <=1 snow layer all of ``R`` sits at the surface (``SWlayer[0]``)."""
    n = ecol.n
    SWlayer = [0.0] * (n + 2)
    if ns > 1:
        res = R
        z = 0.0
        for l in range(1, ns + 1):          # column layer 1 = top snow
            z += ecol.Dlayer[l]             # [m]
            rho = (ecol.ice[l] + ecol.liq[l]) / ecol.Dlayer[l]   # [kg/m3]
            k = rho / 3.0 + 50.0
            SWlayer[l] = res - R * math.exp(-k * z)
            res = R * math.exp(-k * z)
        SWlayer[ns + 1] = res
    else:
        SWlayer[0] = R
    return SWlayer


# ---------------------------------------------------------------------------
# per-point static configuration and running state
@dataclass
class SurfaceStatics:
    lat: float
    lon: float
    ST: float = 0.0
    slope: float = 0.0
    aspect: float = 0.0
    sky: float = 1.0
    horizon: Optional[list] = None
    albpar: albedo.AlbedoParams = field(default_factory=albedo.AlbedoParams)
    theta_sup: float = 0.3            # top-soil liquid content (ground albedo)
    tres_up_albedo: float = 10.0
    # longwave
    lw_state: int = 9
    k1: float = 0.484
    k2: float = 8.0
    # GEOtop: src/geotop/energy.balance.cc:119-121
    # GEOtop: src/geotop/parameters.cc:2229-2231
    # shortwave: Iqbal ozone/turbidity, always on (PointEnergyBalance
    # sets these from the keywords unconditionally); GEOtop's own defaults
    # (Lozone/AngstromAlpha/AngstromBeta), not 0 --
    # zero here silently strips ozone and aerosol extinction from every step.
    Lozone: float = 0.3
    alpha_iqbal: float = 1.3
    beta_iqbal: float = 0.1
    # turbulence
    zmu: float = 5.0
    zmt: float = 2.0
    MO: int = 2
    maxiter: int = 5
    z0_snow: float = 0.0001
    z0_soil: float = 0.01
    z0_thres: float = 10.0
    vmin: float = 1e-3
    # emissivity
    eps_snow: float = 0.98
    eps_soil: float = 0.99
    # canopy: ``None`` (or a class with LSAI below LSAIthres) leaves the point
    # bare and every canopy branch below inactive.
    vegpar: Optional[veg.VegParams] = None
    canopy_opt: veg.CanopyOptions = field(default_factory=veg.CanopyOptions)
    # number of soil layers reached by z_evap (GEOtop's soil_evap_layer size);
    # None keeps the historical behaviour of summing over the whole column.
    n_evap: Optional[int] = None
    # the parameter table's own layer thicknesses [mm] (pa(jdz,l)); the
    # molecular resistance of the bare-soil evaporation consumes these exact
    # values, which metre thicknesses scaled back up do not reproduce.
    soil_D_mm: List[float] = field(default_factory=list)


@dataclass
class SurfaceState:
    snowage: float = 0.0              # non-dimensional snow age
    swrefl_surr: float = 0.0          # previous step's reflected SW [W/m2]
    tsurf_prev: float = 0.0           # previous step's surface temp [C]
                                      # (terrain term of LWin_min/max)

    def copy(self) -> "SurfaceState":
        """Cheap copy for the time loop's attempt/commit pattern -- see
        ``Column1D.copy``."""
        return replace(self)


NODATA_TS = -9999.0

# GEOtop: src/geotop/energy.balance.cc:57-58
# GEOtop: src/geotop/energy.balance.cc:1835-1840 (the gate)
# Surface-flux refresh gate inside the Newton.
# Past the iteration cap the surface energy balance is no longer re-evaluated
# at each line-search trial: the residual is minimised against the fluxes
# frozen at their last evaluated state, and those -- not fluxes re-read at the
# final temperature -- are what the step then reports.
SURF_ENERGY_RECALC_MAX_ITER = 50
# A trial this cold arms the refresh permanently (and from iteration 11 on
# switches the ground aerodynamic resistance to its MO=2, no-stable-correction
# form).
TMIN_SURF_RECALC = -50.0
NEUTRAL_RESISTANCE_MIN_ITER = 10


@dataclass
class NewtonGate:
    """Per-solve state of the surface-flux refresh gate.

    One instance per ``SolvePointEnergyBalance`` attempt: ``flagTmin``/``cont``
    are locals of that function, reset by every retry."""
    cont: int = 0
    flag_tmin: int = 0

    def trial(self, Tg: float, cont: int) -> bool:
        """Count a cold trial; return True when the surface flux must be
        re-evaluated for this line-search trial."""
        self.cont = cont
        if Tg < TMIN_SURF_RECALC:
            self.flag_tmin += 1
        return (cont < SURF_ENERGY_RECALC_MAX_ITER
                or self.flag_tmin > 0)

    def neutral_resistance(self) -> bool:
        """True when the ground resistance must be recomputed with the stable
        correction dropped (flagTmin>0 past iteration 10)."""
        return (self.flag_tmin > 0
                and self.cont > NEUTRAL_RESISTANCE_MIN_ITER)


@dataclass
class AeroInputs:
    """Statics the ground aerodynamic resistance needs to be recomputable
    mid-solve (the MO=2 branch): measurement heights, the roughness selected
    from the step's snow depth, the wind and the Businger iteration cap."""
    zmu: float
    zmt: float
    z0: float
    v: float
    maxiter: int


@dataclass
class FluxBreakdown:
    """The individual surface fluxes at a given ``Tg`` (all W/m2 except E, Qg)."""
    H: float            # sensible heat [W/m2]
    E: float            # vapour mass flux [kg/(s m2)]
    LE: float           # latent heat [W/m2]
    LWnet: float        # net longwave used in EB = LWin - eps*sky*SB(Tg)
    LWup: float         # upward longwave above the canopy
    Qg: float           # surface specific humidity [-]
    # GEOtop: src/geotop/energy.balance.cc:776
    surfEB: float       # SWnet + LWnet - H - LE
    dH_dT: float = 0.0
    dE_dT: float = 0.0
    Hg0: float = 0.0    # bare-fraction sensible heat (Hg_unveg column)
    Eg0: float = 0.0
    Hg1: float = 0.0    # vegetated-fraction sensible heat (Hg_veg column)
    Eg1: float = 0.0
    Ts: float = NODATA_TS   # canopy-air temperature
    Qs: float = NODATA_TS   # canopy-air specific humidity


def _ground_fluxes(diag: "SurfaceDiag", Tg: float,
                   gate_live: bool = False) -> FluxBreakdown:
    """Every surface flux at trial temperature ``Tg`` [W/m2, E in kg/(s m2)],
    composed over the bare and vegetated fractions.

    Split by what does and does not follow ``Tg`` within one timestep: the
    aerodynamic resistances, the canopy temperature and the canopy humidity
    stay frozen at their once-per-step values, while every ground term follows
    the trial -- including the Ye--Pielke evaporation parameters, which are a
    function of the trial surface humidity and of the trial soil state, not
    properties of the step.
    """
    # GEOtop: src/geotop/energy.balance.cc:2571 (EnergyFluxes_no_rec_turbulence)
    # GEOtop: src/geotop/energy.balance.cc:2622 (the evaporation-parameter call)
    e = rad.sat_vap_pressure(Tg, diag.P)
    denom = 240.97 + Tg
    # ``denom * denom`` rather than ``denom ** 2``: a Newton line-search trial
    # Tg can be arbitrarily wild before being rejected, and Python's float
    # ``**`` raises OverflowError on a too-large result where C's
    # multiplication (and GEOtop's) silently saturates to inf.
    de_dT = e * (17.502 / denom - 17.502 * Tg / (denom * denom))
    Qg = rad.spec_humidity(e, diag.P)
    d = diag.P - 0.378 * e
    dQgdT = (0.622 / d + 0.235116 * e / (d * d)) * de_dT

    fc = diag.fc
    H = E = dH_dT = dE_dT = 0.0
    LW = LWup = 0.0
    Hg0 = Eg0 = Hg1 = Eg1 = 0.0
    Ts = Qs = NODATA_TS

    if fc < 1.0:
        # A trial cold enough to arm the refresh gate drops the stable-
        # stratification correction from the ground resistance past iteration
        # 10: the recompute writes rh/rv/Lobukhov back into the diag, and the
        # recomputed rv is what the Ye--Pielke call below must see.
        gate = diag.gate
        if gate_live and gate is not None and diag.aero is not None \
                and gate.neutral_resistance():
            a = diag.aero
            _rm, rh, rv, L = tb.aero_resistance(
                a.zmu, a.zmt, a.z0, 0.0, 0.0, a.v, diag.Ta, Tg, diag.Qa, Qg,
                diag.P, MO=2, maxiter=a.maxiter)
            diag.rh, diag.rv, diag.Lobukhov = rh, rv, L
            diag.rh_c = min(rh, 2.6e3)
        if diag.evap_live:
            (diag.evap_alpha, diag.evap_beta,
             diag.soil_evap) = tb.soil_evaporation_parameters(
                diag.evap_soil, diag.evap_Dsoil, diag.evap_Tsoil,
                diag.evap_theta, diag.P, diag.rv, diag.Ta, diag.Qa, Qg,
                diag.evap_psi, nlayers=diag.evap_nlayers)
        rho = tb.air_density(0.5 * (diag.Ta + Tg), diag.Qa, diag.P)
        cp = tb.air_cp(0.5 * (diag.Ta + Tg))
        Hg = cp * rho * (Tg - diag.Ta) / diag.rh_c
        dHg = cp * rho / diag.rh_c
        rv_eff = diag.rv / diag.evap_beta
        Eg = rho * (diag.evap_alpha * Qg - diag.Qa) / rv_eff
        # ``dQgdT * alpha`` is formed first: turbulent_fluxes receives the
        # already-scaled derivative as one argument, so the product is
        # associated that way and not as (rho*dQgdT)*alpha.
        dEg = rho * (dQgdT * diag.evap_alpha) / rv_eff
        H += (1.0 - fc) * Hg
        E += (1.0 - fc) * Eg
        dH_dT += (1.0 - fc) * dHg
        dE_dT += (1.0 - fc) * dEg
        # the emissivity multiplies the net flux first, and only then the bare
        # fraction: ``(1-fc)*( e*(LWin - sky*SB(Tg)) )``, not ((1-fc)*e)*(...)
        LW += (1.0 - fc) * (diag.eps * (diag.LWin - diag.sky * rad.SB(Tg)))
        LWup += (1.0 - fc) * ((1.0 - diag.eps) * diag.LWin
                              + diag.eps * rad.SB(Tg))
        Hg0, Eg0 = Hg, Eg

    cs = diag.canopy
    if fc > 0.0 and cs is not None:
        if diag.evap_live:
            # Same Ye--Pielke call as the bare fraction, but through the
            # under-canopy resistance and filling the vegetated evaporation
            # profile.
            # GEOtop: src/geotop/energy.balance.cc:2676
            alpha, beta, diag.soil_evap_veg = tb.soil_evaporation_parameters(
                diag.evap_soil, diag.evap_Dsoil, diag.evap_Tsoil,
                diag.evap_theta, diag.P, cs.ruc, diag.Ta, diag.Qa, Qg,
                diag.evap_psi, nlayers=diag.evap_nlayers)
        else:
            alpha, beta = cs.alpha, cs.beta
        Ts = ((diag.Ta / cs.rh + Tg / cs.ruc + cs.Tv / cs.rb)
              / (1.0 / cs.rh + 1.0 / cs.ruc + 1.0 / cs.rb))
        Qs = ((diag.Qa / cs.rv + Qg * alpha * beta / cs.ruc + cs.Qv / cs.rc)
              / (1.0 / cs.rv + beta / cs.ruc + 1.0 / cs.rc))
        Hg, dHg, Eg, dEg = tb.turbulent_fluxes(
            cs.ruc, cs.ruc / beta, diag.P, Ts, Tg, Qs, Qg * alpha,
            dQgdT * alpha)
        _LWv, LWg, _dLWv, LWup_v = veg.longwave_vegetation(
            diag.LWin, diag.eps, Tg, cs.Tv, diag.LSAI)
        H += fc * Hg
        dH_dT += fc * dHg
        E += fc * Eg
        dE_dT += fc * dEg
        LW += fc * LWg
        LWup += fc * LWup_v
        Hg1, Eg1 = Hg, Eg

    LE = tb.latent(Tg, tb.Levap(Tg)) * E
    return FluxBreakdown(H=H, E=E, LE=LE, LWnet=LW, LWup=LWup, Qg=Qg,
                         surfEB=diag.SWnet + LW - H - LE,
                         dH_dT=dH_dT, dE_dT=dE_dT, Hg0=Hg0, Eg0=Eg0,
                         Hg1=Hg1, Eg1=Eg1, Ts=Ts, Qs=Qs)


@dataclass
class SurfaceDiag:
    SWbeam: float
    SWdiff: float
    SWnet: float
    SWup: float
    LWin: float
    cosinc: float
    hsun: float
    rh: float
    rv: float
    Lobukhov: float
    eps: float
    rho: float = 0.0
    cp: float = 0.0
    Qa: float = 0.0
    P: float = 0.0
    Ta: float = 0.0
    sky: float = 1.0
    rh_c: float = 0.0            # capped rh actually used for H (min(rh,2600))
    LWin_min: float = 0.0        # LWin_min output column (epsa_min branch)
    LWin_max: float = 0.0        # LWin_max output column (epsa_max branch)
    evap_alpha: float = 1.0      # Ye--Pielke alpha (1 over snow)
    evap_beta: float = 1.0       # Ye--Pielke beta (1 over snow)
    soil_evap: List[float] = field(default_factory=list)
    # Ye--Pielke inputs kept live for the Newton: alpha and beta are a function
    # of the trial surface humidity, of the trial column temperatures and of the
    # trial soil water content, so they move with every line-search trial rather
    # than being properties of the timestep.  ``evap_Tsoil``/``evap_theta`` are
    # mutable and are rewritten by :func:`evap_state_hook` before each surface
    # evaluation; ``evap_live`` is off exactly when the Ye--Pielke branch does
    # not apply (snow or glacier on top, or a closed canopy).
    evap_live: bool = False
    evap_soil: Optional[List] = None
    # layer thicknesses in millimetres (the C++ reads soil(jdz,l) from the
    # parameter table inside the Ye--Pielke resistance)
    evap_Dsoil: Optional[List[float]] = None
    evap_Tsoil: Optional[List[float]] = None
    evap_theta: Optional[List[float]] = None
    evap_psi: float = -1.0
    evap_nlayers: Optional[int] = None
    soil_evap_veg: List[float] = field(default_factory=list)
    # canopy
    fc: float = 0.0
    LSAI: float = 0.0
    SWv: float = 0.0             # shortwave absorbed by the canopy [W/m2]
    canopy: Optional[veg.CanopyState] = None
    canopy_forcing: Optional[veg.CanopyForcing] = None
    canopy_in: Optional[veg.CanopyInputs] = None
    # Newton refresh gate + the statics the MO=2 resistance recompute needs.
    # ``gate`` is (re)created by step_independent for every solve attempt;
    # ``aero`` is fixed for the step.
    gate: Optional[NewtonGate] = None
    aero: Optional[AeroInputs] = None
    # The fluxes the step reports: the last surface-flux evaluation performed
    # inside the Newton (the C++ never re-reads H/E/LW at the final
    # temperature).  Identical to breakdown(Tg_final) whenever the refresh
    # gate never closed; stale by construction when it did.
    reported: Optional[FluxBreakdown] = None

    def breakdown(self, Tg: float) -> FluxBreakdown:
        """Every EB term at the (solved) surface temperature ``Tg``."""
        return _ground_fluxes(self, Tg)

    def evaporation(self, Tg: float) -> float:
        """Surface vapour flux E [kg/(s m2)] at temperature ``Tg`` (>0 =
        sublimation/loss); recovered after the solve for the snow mass sink."""
        return _ground_fluxes(self, Tg).E


# GEOtop: src/geotop/energy.balance.cc:1497-1498
def _flux_from_diag(diag: SurfaceDiag, sw0: float):
    """Build GEOtop's temperature-dependent EB with frozen resistances.

    ``dEB_dT`` keeps the bare-surface longwave derivative ``-eps*sky*dSB_dT``
    whatever ``fc`` is: ``SolvePointEnergyBalance`` never weights it by the canopy
    fraction, so under a full canopy the Jacobian is inconsistent with the
    residual on purpose."""
    sky_eps = diag.eps * diag.sky

    def flux(Tg: float):
        b = _ground_fluxes(diag, Tg, gate_live=True)
        diag.reported = b
        lat = tb.latent(Tg, tb.Levap(Tg))
        EB = b.LWnet - b.H - lat * b.E + sw0
        dEB = -sky_eps * rad.dSB_dT(Tg) - b.dH_dT - lat * b.dE_dT
        return EB, dEB

    return flux


def evap_state_reset(diag: SurfaceDiag, ecol) -> None:
    """Rewind the live Ye--Pielke state to the start of a column solve.

    The soil water content and the column temperatures the bare-soil
    evaporation parameters are read from are rebuilt from the persistent state
    at the top of every surface solve, so a repeated solve starts from the
    step's own state and not from what the previous attempt left behind."""
    # GEOtop: src/geotop/energy.balance.cc:1414-1450
    if diag.evap_soil is None:
        return
    diag.evap_live = ecol.nsng == 0
    if not diag.evap_live:
        return
    n = len(ecol.soil) - 1
    diag.evap_Tsoil = [0.0] + [ecol.T0[k] for k in range(1, n + 1)]
    diag.evap_theta = [0.0] + [ecol.soil[k].th0 for k in range(1, n + 1)]


def evap_state_hook(diag: SurfaceDiag, ecol, Dt: float,
                    transp_layer: Optional[Sequence[float]] = None,
                    n_transp: int = 0):
    """Build the per-Newton-trial refresh of the soil state behind alpha/beta.

    Returns a ``hook(Temp, deltaw)`` for :func:`geotop_py.energy.column.solve_column_energy`,
    or ``None`` when the Ye--Pielke branch does not apply.  The trial soil water
    content is the start-of-step content plus the Newton's own phase-change
    increment, less the water the previous trial's transpiration and bare/under-
    canopy evaporation would remove over ``Dt``, floored just above the residual
    content.  Both the water content and the trial temperatures feed the next
    evaluation of the surface fluxes, so evaporation throttles itself within the
    iteration instead of being a property of the step."""
    # GEOtop: src/geotop/energy.balance.cc:1841-1880
    if not diag.evap_live or diag.evap_soil is None:
        return None
    rw = C.rho_w
    soil = ecol.soil
    n = len(soil) - 1
    Dlayer = ecol.Dlayer
    fc = diag.fc
    n_evap = diag.evap_nlayers if diag.evap_nlayers is not None else n
    transp_layer = transp_layer if transp_layer is not None else ()
    # The guard reads the residual content of the *top* layer whatever the layer
    # being updated is, while the floor applied right after uses the layer's own
    # -- the two are not the same number on a layered profile.
    res_top = soil[1].res

    def hook(Temp, deltaw):
        for l in range(1, n + 1):
            diag.evap_Tsoil[l] = Temp[l]
        for l in range(1, n + 1):
            th = soil[l].th0 + deltaw[l] / (rw * Dlayer[l])
            floor = soil[l].res + 1.0e-3
            if th > res_top + 1.0e-3 and l <= n_transp:
                if l < len(transp_layer):
                    th -= max(Dt * fc * transp_layer[l] / (rw * Dlayer[l]), 0.0)
                if th < floor:
                    th = floor
            if th > res_top + 1.0e-3 and l <= n_evap:
                if l < len(diag.soil_evap):
                    th -= max(Dt * (1.0 - fc) * diag.soil_evap[l]
                              / (rw * Dlayer[l]), 0.0)
                if l < len(diag.soil_evap_veg):
                    th -= max(Dt * fc * diag.soil_evap_veg[l]
                              / (rw * Dlayer[l]), 0.0)
                if th < floor:
                    th = floor
            diag.evap_theta[l] = th

    return hook


def _resolve_canopy(st, diag: SurfaceDiag, ecol, Tg: float) -> None:
    """Re-run ``Tcanopy`` for a repeated ``SolvePointEnergyBalance``.

    GEOtop's ``do{...}while(sux<0)`` loop calls the whole of ``EnergyFluxes``
    again on every retry, and ``EnergyFluxes`` calls ``Tcanopy``.  The retry
    therefore sees the *new* skin temperature (and, after a merge or the
    ``sux=-6`` fold, a different column geometry), while ``Tv0`` is whatever the
    rejected solve left in ``V->Tv[j]`` and ``Wcrn``/``Wcsn`` are still the
    start-of-step values -- they are only committed on the successful return.
    Skipping this is what made a canopy point diverge the first time the last
    snow layer melted out: the canopy was still solved against a snow skin at
    -0.1 C while GEOtop had already re-solved it against bare soil.
    """
    cf = diag.canopy_forcing
    ci = diag.canopy_in
    if cf is None or ci is None:
        return
    cf.Tg = Tg
    cf.Qgsat = rad.spec_humidity(rad.sat_vap_pressure(Tg, diag.P), diag.P)
    cf.nsng = ecol.nsng
    cf.soil = ecol.soil
    # same placeholder structure as the diag's evap_Dsoil: the canopy's own
    # Ye--Pielke short-circuits over ice, so only the soil entries are read
    cf.Dsoil = ([0.0] * (ecol.nsng + 1)
                + [st.soil_D_mm[k] for k in range(0, ecol.n - ecol.nsng)])
    cf.Tsoil = [0.0] + [ecol.T0[k] for k in range(1, ecol.n + 1)]
    cf.theta = [0.0] + [ecol.soil[k].th0 for k in range(1, len(ecol.soil))]
    diag.canopy = veg.Tcanopy(diag.canopy.Tv, ci.Wcrn, ci.Wcsn, ci.Wcrnmax,
                              ci.Wcsnmax, diag.SWv, cf, ci.vp, st.canopy_opt)
    diag.Lobukhov = diag.canopy.Lobukhov


def _retry_ground_refresh(st, scol, diag: SurfaceDiag, snowD: float,
                          wind: float) -> None:
    """Re-freeze the ground resistances at a repeated solve's own start.

    Every repeated column solve re-evaluates the aerodynamic resistances at
    the surface temperature *that solve* starts from, not the one the step
    began with -- and a repeat changes it, either because the surface index
    moved onto the first material node or because a layer was merged away.
    Radiation, albedo, emissivity and the roughness stay fixed: they are
    chosen outside the solve and no repeat revisits them.

    Ye--Pielke is refreshed only where it applies at all (no ice above the
    soil); with snow or glacier present its parameters are the neutral pair
    and there is nothing to recompute.
    """
    # GEOtop: src/geotop/energy.balance.cc:1456-1461
    # GEOtop: src/geotop/energy.balance.cc:2450-2457
    if diag.fc >= 1.0:
        return
    Tg0 = scol.T0[scol.surface_index]
    Qg0 = rad.spec_humidity(rad.sat_vap_pressure(Tg0, diag.P), diag.P)
    z0 = st.z0_snow if snowD > st.z0_thres else st.z0_soil
    _rm, rh, rv, Lob = tb.aero_resistance(
        st.zmu, st.zmt, z0, 0.0, 0.0, max(wind, st.vmin), diag.Ta, Tg0,
        diag.Qa, Qg0, diag.P, MO=st.MO, maxiter=st.maxiter)
    diag.rh, diag.rv = rh, rv
    diag.rh_c = min(rh, 2.6e3)
    if diag.fc <= 0.0:
        diag.Lobukhov = Lob
    if scol.nsng == 0 and diag.evap_soil is not None:
        (diag.evap_alpha, diag.evap_beta,
         diag.soil_evap) = tb.soil_evaporation_parameters(
            diag.evap_soil, diag.evap_Dsoil,
            [0.0] + [scol.T0[k] for k in range(1, scol.n + 1)],
            diag.evap_theta, diag.P, rv, diag.Ta, diag.Qa, Qg0,
            diag.evap_psi, nlayers=diag.evap_nlayers)


def _bare_retry_flux(st, bare, diag: SurfaceDiag, sw0: float,
                     snowD: float, wind: float):
    """Rebuild only the flux pieces GEOtop changes during ``sux=-6``.

    Radiation, albedo, emissivity and the roughness selected from the original
    snow depth stay fixed outside PointEnergyBalance.  Aerodynamic resistance
    and Ye--Pielke soil evaporation are recomputed from the temporary bare
    column and its new starting skin temperature.
    """
    Tg0 = bare.T0[0]
    Qg0 = rad.spec_humidity(rad.sat_vap_pressure(Tg0, diag.P), diag.P)
    z0 = st.z0_snow if snowD > st.z0_thres else st.z0_soil
    soil = bare.soil
    Tsoil = [0.0] + [bare.T0[k] for k in range(1, bare.n + 1)]
    theta = [0.0] + [soil[k].th0 for k in range(1, len(soil))]
    # GEOtop: src/geotop/energy.balance.cc:1888 (sl->pa->matrix(sy))
    # the parameter table's own mm row, even though the trial column's node 1
    # is temporarily inflated by the folded snow: the C++ reads soil(jdz,l)
    # and never sees the augmentation
    Dsoil = ([0.0] * (bare.nsng + 1)
             + [st.soil_D_mm[k] for k in range(0, bare.n - bare.nsng)])
    # ``PointEnergyBalance`` folds the exhausted snow ice into ``SL->th`` /
    # ``SL->thi`` before the bare retry, but it does not update ``SL->P``.
    # ``SolvePointEnergyBalance`` therefore still hands EnergyFluxes the
    # surface pressure from the start of the timestep (``psi0 = SL->P(0,j)``),
    # rather than inverting the temporarily augmented water content.  Keeping
    # that pressure is important near a complete-melt threshold: recomputing
    # it perturbs Ye--Pielke evaporation and leaves a small but persistent
    # temperature error in the soil column.
    psi = diag.evap_psi
    # Under a closed canopy the bare fraction does not exist: the ground
    # resistances are the infinite placeholders, no Ye--Pielke pair is
    # evaluated for it, and the reported Obukhov length stays the one the
    # canopy solve left -- the bare aerodynamic call is never made at all.
    # GEOtop: src/geotop/energy.balance.cc:2440-2470
    if diag.fc < 1.0:
        _rm, rh, rv, Lob = tb.aero_resistance(
            st.zmu, st.zmt, z0, 0.0, 0.0, max(wind, st.vmin), diag.Ta, Tg0,
            diag.Qa, Qg0, diag.P, MO=st.MO, maxiter=st.maxiter)
        alpha, beta, soil_evap = tb.soil_evaporation_parameters(
            soil, Dsoil, Tsoil, theta, diag.P, rv, diag.Ta, diag.Qa, Qg0, psi,
            nlayers=st.n_evap)
        diag.rh, diag.rv = rh, rv
        diag.rh_c = min(rh, 2.6e3)
        diag.evap_alpha, diag.evap_beta = alpha, beta
        diag.soil_evap = soil_evap
        if diag.fc <= 0.0:
            # GEOtop: src/geotop/energy.balance.cc:2452 (bare aero_resistance)
            # GEOtop: src/geotop/energy.balance.cc:2510 (Tcanopy)
            # Both branches of EnergyFluxes write the same Obukhov length, and
            # the canopy one runs second (bare, then Tcanopy), so
            # under a partial canopy it is the canopy's value that survives --
            # and ``_resolve_canopy`` has already stored it.  Only a point with
            # no canopy at all reports the bare one.
            diag.Lobukhov = Lob
    else:
        diag.rh = diag.rv = 1.0e99
        diag.rh_c = min(diag.rh, 2.6e3)
    diag.evap_soil, diag.evap_Dsoil = soil, Dsoil
    diag.evap_psi = psi
    return _flux_from_diag(diag, sw0)


# GEOtop: src/geotop/energy.balance.cc:601-602
# GEOtop: src/geotop/energy.balance.cc:606
def surface_forcing(st: SurfaceStatics, state: SurfaceState, ecol, ns: int,
                    snowD: float, Ts_surf: float, Ta: float, RH: float, P: float,
                    wind: float, tau_cloud: float, Psnow: float,
                    JDb: float, JDe: float, Dt: float,
                    canopy_in: Optional[veg.CanopyInputs] = None,
                    tau_cloud_av: Optional[float] = None,
                    LWin_measured: Optional[float] = None):
    """Build ``(surface_flux, SWlayer, diag)`` for one step and advance the
    snow age. ``diag.SWup`` is the next step's ``SWrefl_surr``, stored by the
    caller once the energy solve is accepted.

    ``Ts_surf`` = current surface-node temperature (= ecol.T0 at the surface),
    used both for the snow-age rate and to freeze the turbulent resistances.

    ``tau_cloud_av`` feeds longwave only (``PointEnergyBalance`` passes
    ``tau_cloud_av``, not ``tau_cloud``, to ``longwave_radiation``) --
    GEOtop keeps the two separate (see :class:`Meteo`); ``None`` reuses
    ``tau_cloud`` for callers that only ever had one value.

    ``LWin_measured``, when not ``None``, replaces the cloud-derived
    ``LWin`` outright (``flux`` on ``iLWi``) -- ``LWin_min``/
    ``LWin_max`` keep the cloud-derived bounds regardless."""
    if tau_cloud_av is None:
        tau_cloud_av = tau_cloud
    # --- 1. snow age (stateful) then albedo bands ------------------------
    # Nested inside ``snowD > 0``: the snowfall that creates a new pack does
    # not age it until the following step.  The rate is driven by the top snow
    # layer's temperature (energy node 1, the pack top-down), not by the skin.
    # GEOtop: src/geotop/energy.balance.cc:455-459
    if snowD > 0.0:
        state.snowage = albedo.update_snow_age(
            state.snowage, max(0.0, Psnow), ecol.T0[1], Dt,
            st.tres_up_albedo)

    # --- 2. shortwave (R1) with last step's reflected SW -----------------
    swb, swd, cosinc, hsun, _, _ = rad.shortwave_step(
        JDb, JDe, st.lat, st.lon, st.ST, RH, Ta, P, tau_cloud,
        sky=st.sky, slope_deg=st.slope, aspect_deg=st.aspect,
        SWrefl_surr=state.swrefl_surr, horizon=st.horizon,
        Lozone=st.Lozone, alpha=st.alpha_iqbal, beta=st.beta_iqbal)

    # --- 3. albedo -> net / reflected shortwave (R3) ---------------------
    # GEOtop evaluates ground albedo from the current top-soil liquid content,
    # not from its initial value.  It remains relevant under shallow snow via
    # the AlbExtParSnow blend.
    # The first soil node sits below snow AND glacier: ecol.nsng == ns + ng.
    # Indexing with ns alone would read a glacier layer's liquid water as if it
    # were topsoil moisture -- near zero, so find_albedo would return the DRY
    # ground albedo where GEOtop returns the wet one.
    soil_top = ecol.nsng + 1
    theta_sup = ecol.liq[soil_top] / (C.rho_w * ecol.Dlayer[soil_top])
    alb = albedo.albedos(snowD, state.snowage, cosinc, theta_sup, st.albpar)
    sw_abs, sw_up = albedo.swnet(swb, swd, alb)
    fc = canopy_in.fc if canopy_in is not None else 0.0
    SWnet = (1.0 - fc) * sw_abs
    SWup = (1.0 - fc) * sw_up
    SWv = 0.0
    if fc > 0.0:
        # GEOtop: src/geotop/energy.balance.cc:543-569
        # The canopy two-stream runs per band on half the incoming flux, with
        # the ground albedo of that band as the lower boundary condition.
        # It returns the canopy, ground and
        # upward shares of the vegetated fraction; SWv drives Tcanopy,
        # and the ground keeps only the (1-fc) share computed above.
        vp = canopy_in.vp
        avis_b, avis_d, anir_b, anir_d = alb
        if hsun > 0:
            fsnowcan = (canopy_in.Wcsn / canopy_in.Wcsnmax) ** (2.0 / 3.0)
            SWv_vis, SWg_vis, SWup_vis = veg.shortwave_vegetation(
                0.5 * swd, 0.5 * swb, cosinc, fsnowcan, veg.wsn_vis,
                veg.Bsnd_vis, veg.Bsnb_vis, avis_d, avis_b, vp.Ch, vp.R_vis,
                vp.T_vis, vp.LSAI)
            SWv_nir, SWg_nir, SWup_nir = veg.shortwave_vegetation(
                0.5 * swd, 0.5 * swb, cosinc, fsnowcan, veg.wsn_nir,
                veg.Bsnd_nir, veg.Bsnb_nir, anir_d, anir_b, vp.Ch, vp.R_nir,
                vp.T_nir, vp.LSAI)
        else:
            SWv_vis = SWg_vis = SWup_vis = 0.0
            SWv_nir = SWg_nir = SWup_nir = 0.0
        SWnet += fc * (SWg_vis + SWg_nir)
        SWup += fc * (SWup_vis + SWup_nir)
        SWv = SWv_vis + SWv_nir

    # --- 4. incoming longwave (R2) ---------------------------------------
    #   lwin_step returns (LWin, epsa_max-branch, epsa_min-branch); the output
    #   columns are LWin_min<-epsa_min (col27) and LWin_max<-epsa_max (col28).
    LWin, lwin_epsamax, lwin_epsamin = rad.lwin_step(
        Ta, RH, P, tau_cloud_av, st.sky, state=st.lw_state, k1=st.k1, k2=st.k2)
    if LWin_measured is not None:
        LWin = LWin_measured

    # --- 5. shortwave penetration -> per-node source ---------------------
    SWlayer = rad_snow_absorption(SWnet, ecol, ns)

    # --- 6. freeze turbulence at the current surface temperature ---------
    eps = st.eps_snow if snowD > 10 else st.eps_soil
    Qa = rad.spec_humidity(RH * rad.sat_vap_pressure(Ta, P), P)
    Qg0 = rad.spec_humidity(rad.sat_vap_pressure(Ts_surf, P), P)
    z0 = st.z0_snow if snowD > st.z0_thres else st.z0_soil
    soil = ecol.soil
    Tsoil = [0.0] + [ecol.T0[k] for k in range(1, ecol.n + 1)]
    theta = [0.0] + [soil[k].th0 for k in range(1, len(soil))]
    # the parameter table's own mm row, indexed by soil layer whatever lies
    # above the soil: a repeated solve that has merged the last ice node away
    # reads it as it stands, with no ice left to shift the indices
    # GEOtop: src/geotop/turbulence.cc:600-601
    Dsoil_mm = [0.0] + [st.soil_D_mm[k] for k in range(0, ecol.n - ecol.nsng)]
    psi_surface = laws.psi_teta(
        soil[1].th0 + soil[1].thi0, 0.0, soil[1].sat, soil[1].res,
        soil[1].alpha, soil[1].n, soil[1].m, C.PsiMin, soil[1].ss)

    rh = rv = 1.0e99
    L = 1.0e5
    rh_c = min(rh, 2.6e3)
    evap_alpha = 1.0
    evap_beta = 1.0
    soil_evap = []
    if fc < 1.0:
        _rm, rh, rv, L = tb.aero_resistance(st.zmu, st.zmt, z0, 0.0, 0.0,
                                            max(wind, st.vmin), Ta, Ts_surf, Qa, Qg0, P,
                                            MO=st.MO, maxiter=st.maxiter)
        rh_c = min(rh, 2.6e3)
        # Ye--Pielke bare-soil evaporation applies only when NOTHING covers the
        # soil.  find_actual_evaporation_parameters (turbulence.cc) short-circuits
        # to alpha = beta = 1 whenever its ``nsnow`` argument -- called with ns+ng,
        # and commented "snow or ice" -- is positive.  Using ns alone would throttle
        # sublimation from a bare glacier with a soil resistance that is not there.
        if ecol.nsng == 0:
            evap_alpha, evap_beta, soil_evap = tb.soil_evaporation_parameters(
                soil, Dsoil_mm, Tsoil, theta, P, rv, Ta, Qa, Qg0, psi_surface,
                nlayers=st.n_evap)
    rho = tb.air_density(0.5 * (Ta + Ts_surf), Qa, P)
    cp = tb.air_cp(0.5 * (Ta + Ts_surf))

    canopy = None
    if fc > 0.0:
        cf = veg.CanopyForcing(
            Tg=Ts_surf, Ta=Ta, Qgsat=Qg0, Qa=Qa, zmu=st.zmu, zmt=st.zmt,
            z0v=canopy_in.z0veg, z0s=z0, d0v=canopy_in.d0veg, rz0v=1.0,
            hveg=canopy_in.hveg, v=max(wind, st.vmin), P=P,
            SWin=swb + swd, LW=LWin, eps=eps, Dt=Dt, nsng=ecol.nsng,
            theta=theta, soil=soil, Dsoil=Dsoil_mm, Tsoil=Tsoil, psi=psi_surface,
            n_evap=(st.n_evap if st.n_evap is not None else ecol.n - ecol.nsng))
        canopy = veg.Tcanopy(canopy_in.Tv0, canopy_in.Wcrn, canopy_in.Wcsn,
                             canopy_in.Wcrnmax, canopy_in.Wcsnmax, SWv, cf,
                             canopy_in.vp, st.canopy_opt)
        L = canopy.Lobukhov
    sw0 = SWlayer[0]

    # LWin_min/max output columns carry the surrounding-terrain term
    # (1-sky)*eps*SB(Tsurface_prev), added unconditionally (energy.balance.cc:
    # 1014-1016), unlike the EB's LWin. eps here = surface emissivity.
    terrain = (1.0 - st.sky) * eps * rad.SB(state.tsurf_prev)
    diag = SurfaceDiag(swb, swd, SWnet, SWup, LWin, cosinc, hsun, rh, rv, L, eps,
                       rho=rho, cp=cp, Qa=Qa, P=P, Ta=Ta, sky=st.sky, rh_c=rh_c,
                       LWin_min=lwin_epsamin + terrain, LWin_max=lwin_epsamax + terrain,
                       evap_alpha=evap_alpha, evap_beta=evap_beta,
                       soil_evap=soil_evap, fc=fc,
                       evap_live=(ecol.nsng == 0),
                       evap_soil=soil, evap_Dsoil=Dsoil_mm,
                       evap_Tsoil=list(Tsoil), evap_theta=list(theta),
                       evap_psi=psi_surface, evap_nlayers=st.n_evap,
                       LSAI=(canopy_in.vp.LSAI if canopy_in is not None else 0.0),
                       SWv=SWv, canopy=canopy,
                       canopy_forcing=(cf if fc > 0.0 else None),
                       canopy_in=canopy_in,
                       aero=AeroInputs(zmu=st.zmu, zmt=st.zmt, z0=z0,
                                       v=max(wind, st.vmin),
                                       maxiter=st.maxiter))
    return _flux_from_diag(diag, sw0), SWlayer, diag


def reported_LE(b: FluxBreakdown, T1: float) -> float:
    """Latent heat flux as the output columns report it [W/m2].

    The vapour flux is converted to an energy flux at the first *material*
    node, not at the skin temperature the Newton solved for.  With a massless
    skin (the default surface convention) the two nodes are at different
    temperatures during a fast transient, and the conversion factor differs
    with them; the per-fraction ``LEg_unveg``/``LEg_veg`` columns deliberately
    use the skin instead, so the reported total is not the sum of its reported
    parts."""
    # GEOtop: src/geotop/energy.balance.cc:775
    return tb.latent(T1, tb.Levap(T1)) * b.E


def reported_surface_EB(diag: SurfaceDiag, b: FluxBreakdown, T1: float) -> float:
    """Surface energy budget as the output columns report it [W/m2], i.e.
    built on :func:`reported_LE` rather than on the skin-temperature latent
    heat the Newton's own residual uses."""
    # GEOtop: src/geotop/energy.balance.cc:776
    return diag.SWnet + b.LWnet - b.H - reported_LE(b, T1)


# GEOtop: src/geotop/energy.balance.cc:778-796
def _ground_heat_flux(ecol, res, ns: int, diag: SurfaceDiag, Tg: float,
                      T1: float) -> float:
    """Soil_heat_flux column (``GEF``): with no snow it is the
    whole surface budget; with snow, the conductive flux across the snow/soil
    interface (soil branch of ``k_thermal``, GEOtop's ``Arithmetic_Mean``)."""
    if ns == 0:
        b = diag.reported if diag.reported is not None \
            else diag.breakdown(Tg)
        return reported_surface_EB(diag, b, T1)

    def am(D1, D2, K1, K2):                       # (D1*K2 + D2*K1)/(D1+D2)
        return (D1 * K2 + D2 * K1) / (D1 + D2)

    D0, D1 = ecol.Dlayer[ns], ecol.Dlayer[ns + 1]
    rw = C.rho_w
    th_liq = am(D0, D1, ecol.liq[ns] / (rw * D0), ecol.liq[ns + 1] / (rw * D1))
    th_ice = am(D0, D1, ecol.ice[ns] / (rw * D0), ecol.ice[ns + 1] / (rw * D1))
    th_sat = am(D0, D1, 1.0, ecol.soil[1].sat)
    k_sol = am(D0, D1, 0.0, ecol.soil[1].kt)
    k = laws.k_thermal(0, 1, th_liq, th_ice, th_sat, k_sol)
    return k * (res.Temp[ns] - res.Temp[ns + 1]) / (0.5 * D0 + 0.5 * D1)
