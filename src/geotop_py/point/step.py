"""One coupled point timestep, and the sub-stepping loop around it.

A point step runs in three phases, cut where a failure stops it:
:func:`energy_phase` (canopy interception, the surface flux from the meteo
(:mod:`geotop_py.energy.surface`), the implicit energy solve with its
accept/merge/retry verdict loop), :func:`post_energy_phase` (snow and glacier
mass balance, restratification, fresh snow, soil update), run only on an
accepted solve, and :func:`water_phase` (Richards), run only when every point
of the trial got that far. :func:`step_independent` chains them for one point;
:func:`simulate_energy_balance` wraps it in the halve-on-failure /
double-on-success ``Dt`` subdivision of :mod:`geotop_py.point.time_loop`.

:class:`StepOut` and :class:`LayerProfile` are what a step reports to the
output writers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional

from .. import constants as C
from .. import laws
from ..energy import vegetation as veg
from ..energy.column import solve_column_energy
from ..energy.surface import (
    NewtonGate,
    SurfaceDiag,
    SurfaceState,
    SurfaceStatics,
    _bare_retry_flux,
    _ground_heat_flux,
    _resolve_canopy,
    _retry_ground_refresh,
    evap_state_hook,
    evap_state_reset,
    surface_forcing,
)
from ..meteo.step import Meteo
from ..snow.strati import snow_layer_combination
from ..snow.wb import EBSnow, WBglacier, WBsnow, new_snow
from . import time_loop
from .state import carry_soil_state, flatten, fresh_snow_depth


# GEOtop: src/geotop/energy.balance.cc:2100 (the skin-temperature guard)
# GEOtop: src/geotop/energy.balance.cc:2077-2080 (the non-convergence guard)
def _needs_surface_melt_retry(res, ns: int, ng: int, surface_index: int) -> bool:
    """GEOtop ``sux=-1`` condition for retrying with material node 1.

    The two clauses count the stack differently, and the difference is not
    cosmetic:

    * the skin-temperature guard reads
      ``sur == 0 && ns+ng > 0 && egy->Temp[sur] > 0`` -- snow **plus** glacier,
      because counting only snow would let the skin of a bare glacier climb
      above 0 C and vent the surplus as longwave and sensible heat instead of
      melting ice, which is precisely the ablation season a glacier run is
      about;
    * the non-convergence guard reads ``ns > 1 && surfacemelting == 0``
      -- **snow only**.  A failed solve over a single snow node (whatever sits
      below it) is not retried on node 1: it goes straight to the merge that
      removes that node, ``-6`` or ``-5`` (see :func:`_sux_verdict`).
    """
    if surface_index != 0:
        return False
    return ((ns + ng > 0 and res.Temp[0] > 0.0)
            or (ns > 1 and not res.converged))


# GEOtop: src/geotop/energy.balance.cc:2074-2142 (the sux verdict of SolvePointEnergyBalance)
# GEOtop: src/geotop/energy.balance.cc:2075-2093 (non-convergence)
# GEOtop: src/geotop/energy.balance.cc:2100 (skin-temperature test)
# GEOtop: src/geotop/energy.balance.cc:2102-2142 (exhausted-ice scan)
def _sux_verdict(res, ecol, ns: int, ng: int, sur: int):
    """GEOtop's ``sux`` for one completed solve, in the order of
    ``SolvePointEnergyBalance``.

    **Non-convergence is judged first, and purely on the stack geometry**
    -- the state is not even looked at::

        if (res > tol) {
            if      (ns > 1 && surfacemelting == 0) return -1;
            else if (ns == 1 && ng == 0)            return -6;
            else if (ns == 1 && ng > 0)             return -5;
            else                                  { print; return 1; }
        }

    So a Newton that runs out of iterations over a *single* snow node is not
    accepted: GEOtop takes it as evidence that the node should not be there and
    goes straight to the merge that removes it (``-6`` fold into the soil over
    bare ground, ``-5`` into the glacier below), re-solving the shortened
    column.  Only with no snow at all -- or already on material node 1 -- does
    it keep the non-converged state (code ``1``, warning only).

    Only when the solve *did* converge do the skin-temperature test and the
    exhausted-ice scan run.
    """
    if _needs_surface_melt_retry(res, ns, ng, sur):
        return -1, None
    if not res.converged:
        if ns == 1 and ng == 0:
            return -6, ns
        if ns == 1 and ng > 0:
            return -5, ns
        return 1, None
    return _exhausted_ice_node(ecol, res, ns, ng)


# GEOtop: src/geotop/energy.balance.cc:2866-2913 (merge)
def _merge_energy_snow_layer(ecol, res, l: int, ldw: Optional[int] = None,
                             lup: int = 1):
    """Apply GEOtop ``merge`` to a fully melted ice energy node.

    Energy nodes are top-down.  The problematic node is merged with the
    lighter adjacent node (or the only available neighbour at a boundary),
    all following material nodes are shifted upward, and the active column is
    shortened by one.  The merged temperature/phase partition is recovered
    from conserved internal energy exactly as in energy.balance.cc.

    ``lup``/``ldw`` bound the search for the merge partner, mirroring the
    ``merge(..., l, lup, ldw, tot)`` signature.  They
    matter only when a glacier is present: the ``sux=-2`` branch passes
    ``(1, ns)`` so a snow node never merges into the glacier below, and the
    ``sux=-3`` branch passes ``(ns+1, ns+ng)`` for the mirror-image constraint.
    ``ldw`` defaults to the whole ice stack, which is the snow-only case.

    ``lup`` may be 0: the ``sux=-4`` branch passes ``(ns, ns+ng)`` and with no
    snow left that makes the partner node 0, the massless skin.  GEOtop merges
    into it all the same, and nothing ever reads ``ice[0]``/``liq[0]``/
    ``Dlayer[0]`` back -- the residual mass of the last glacier node is simply
    dropped.  Reproduced verbatim.
    """
    nsng = ecol.nsng
    if ldw is None:
        ldw = nsng
    if not (0 <= lup <= l <= ldw <= nsng):
        raise ValueError("a removable ice node must lie within [lup, ldw]")

    if l == lup:
        m, n = l + 1, l
    elif l < ldw and ecol.ice[l + 1] < ecol.ice[l - 1]:
        m, n = l + 1, l
    else:
        m, n = l, l - 1

    D = list(ecol.Dlayer)
    ice = list(ecol.ice)
    liq = list(ecol.liq)
    Temp = list(res.Temp)
    h = (laws.internal_energy(ice[m], liq[m], Temp[m])
         + laws.internal_energy(ice[n], liq[n], Temp[n]))
    D[n] += D[m]
    ice[n] += ice[m]
    liq[n] += liq[m]
    ice[n], liq[n], Temp[n] = laws.from_internal_energy(
        ecol.alpha_snow, h, ice[n], liq[n])

    for values in (D, ice, liq, Temp):
        del values[m]
    Temp[0] = Temp[1]
    return replace(ecol, Dlayer=D, ice=ice, liq=liq, T0=Temp,
                   nsng=nsng - 1, surface_index=ecol.surface_index)


def _exhausted_ice_node(ecol, res, ns: int, ng: int):
    """GEOtop's ``sux`` verdict on a fully melted ice node (energy.balance.cc:
    1877-1906).

    Scans the snow+glacier stack top-down and returns ``(code, l)`` for the
    FIRST node left with less than 1e-5 kg m-2 of ice, or ``(0, None)`` when
    the solve stands.  The code depends on the *global* stack geometry, not on
    where the node sits, because each one needs a different merge partner:

    ``-2`` snow node with more snow around, ``-3`` glacier node with more
    glacier around, ``-4`` the single remaining glacier node, ``-5`` the single
    remaining snow node over a glacier, ``-6`` the single remaining snow node
    over bare soil.  A node matching none of the five (impossible for a
    non-empty stack) is skipped, as the C++ ``else if`` chain does.
    """
    for l in range(1, ns + ng + 1):
        if ecol.ice[l] - res.deltaw[l] >= 1e-5:
            continue
        if ns > 1 and l <= ns:
            return -2, l
        if ng > 1 and l >= ns + 1:
            return -3, l
        if ng == 1 and l == ns + ng:
            return -4, l
        if ns == 1 and ng > 0 and l == ns:
            return -5, l
        if ns == 1 and ng == 0 and l == ns:
            return -6, l
    return 0, None


def _remove_last_snow_for_soil_solve(ecol, res):
    """Prepare GEOtop's ``sux=-6`` temporary snow-to-soil column."""
    ic, wa = ecol.ice[1], ecol.liq[1]
    rho = ic / ecol.Dlayer[1]
    soil_D1 = ecol.Dlayer[2]
    D, ice, liq, Temp = (list(ecol.Dlayer), list(ecol.ice),
                         list(ecol.liq), list(res.Temp))
    for values in (D, ice, liq, Temp):
        del values[1]
    D[1] *= 1.0 + ic / (ice[1] + liq[1])
    ice[1] += ic
    Temp[0] = Temp[1]
    # The C++ branch immediately writes the augmented masses back to the top
    # soil water/ice fractions before repeating SolvePointEnergyBalance.  The
    # freezing curve must therefore use this temporary state as its th0/thi0,
    # not the original soil fractions.
    soil = list(ecol.soil)
    th0 = liq[1] / (C.rho_w * D[1])
    thi0 = ice[1] / (C.rho_w * D[1])
    # SolvePointEnergyBalance rebuilds Tstar from the *current* total soil
    # water at the top of every call, retries included, so the repeated solve
    # sees a freezing point that already accounts for the folded snow ice.
    # GEOtop: src/geotop/energy.balance.cc:1436-1447
    s1 = soil[1]
    psi0 = laws.psi_teta(th0 + thi0, 0.0, s1.sat, s1.res, s1.alpha, s1.n,
                         s1.m, C.PsiMin, s1.ss)
    soil[1] = replace(s1, th0=th0, thi0=thi0,
                      Tstar=min(psi0 / (1000.0 * C.Lf / (C.GRAVITY * C.tk)),
                                0.0))
    bare = replace(ecol, Dlayer=D, ice=ice, liq=liq, T0=Temp,
                   nsng=0, surface_index=0, soil=soil)
    return bare, (ic, wa, rho, soil_D1)


def _restore_last_snow(ecol, bare, res, saved):
    """Undo the temporary ``sux=-6`` geometry and allocate soil thaw to snow."""
    ic, wa, rho, soil_D1 = saved
    D, ice, liq, Temp, dw = (list(bare.Dlayer), list(bare.ice), list(bare.liq),
                             list(res.Temp), list(res.deltaw))
    D[1] = soil_D1
    D.insert(1, ic / rho)
    ice.insert(1, ic)
    liq.insert(1, wa)
    Temp.insert(1, min(Temp[1], -0.1))
    dw.insert(1, 0.0)
    if dw[2] > ic:
        dw[1] = ic
        dw[2] -= ic
    elif dw[2] > 0.0:
        dw[1] = dw[2]
        dw[2] = 0.0
    ice[2] -= ic
    restored = replace(ecol, Dlayer=D, ice=ice, liq=liq, T0=list(ecol.T0),
                       nsng=1)
    res.Temp, res.deltaw = Temp, dw
    return restored, res


# ---------------------------------------------------------------------------
# fully-independent step: geotop_py computes its own surface flux from the meteo
# ---------------------------------------------------------------------------


@dataclass
class LayerProfile:
    """End-of-step per-layer snapshot for the GEOtop ``output_prof`` files.

    Snow layers are bottom-up (index 0 = base adjacent to the soil, last =
    surface), matching GEOtop's ``L1..Ln`` ordering. Soil nodes are top-first.
    """
    Dz: List[float]        # snow layer thickness [mm]
    T: List[float]         # snow layer temperature [C]
    wice: List[float]      # snow layer ice content [kg/m2]
    wliq: List[float]      # snow layer liquid content [kg/m2]
    soil_T: List[float]    # soil node temperature [C], top-first
    soil_th: List[float]   # soil node liquid content [-], top-first
    soil_thi: List[float]  # soil node ice content [-], top-first
    soil_psi: List[float]  # soil node matric potential [mm], top-first
    soil_ptot: List[float]  # total-water-equivalent potential [mm], top-first


@dataclass
class StepOut:
    """State and fluxes of one point after one internal step.

    The public fields, read through :class:`geotop_py.results.StepRecord`:

    - snow: ``swe`` [kg/m2 = mm] and ``depth`` [mm] of the active layers,
      ``snow_T`` [C] (thickness-weighted), ``HN`` fresh snow depth [mm],
      ``Melt`` water leaving the pack base [mm], ``RainOnSnow`` [mm];
    - glacier: ``gwe`` [mm], ``glac_depth`` [mm], ``glac_T`` [C],
      ``Melt_glac`` [mm], ``Evap_glac`` [mm];
    - surface: ``Tg`` surface temperature [C], ``T1`` first material node
      [C], ``GEF`` ground heat flux [W/m2], ``Evap`` surface evaporation or
      sublimation [mm], ``evap_soil`` the part charged to bare soil [mm];
    - canopy: ``fc`` [-], ``LSAI`` [-], ``Tv`` [C], ``SWv`` [W/m2];
    - energy solver: ``converged``, ``solver_iterations``,
      ``solver_residual``, ``solver_code`` (0 ok, nonzero failure);
    - water balance (Richards, ``WaterBalance=1`` only): ``wb_converged``,
      ``wb_iterations``, ``wb_loss`` mass-balance residual [mm], ``pnet``
      water reaching the soil surface [mm], ``wb_vbottom``/``wb_vlat``
      drained volumes [m3];
    - ``diag``: radiation and turbulence diagnostics
      (:class:`geotop_py.energy.surface.SurfaceDiag`).

    Other fields serve the model itself and may change.
    """
    swe: float
    depth: float
    Tg: float
    Melt: float
    Evap: float
    HN: float
    converged: bool
    diag: SurfaceDiag
    RainOnSnow: float = 0.0
    snow_T: float = float("nan")     # bulk mass-weighted snow temperature [C]
    GEF: float = 0.0                 # ground/soil heat flux [W/m2]
    T1: float = 0.0                  # first material node temperature [degC]
    had_snow: bool = False           # snowpack present at start of the step
    # GEOtop: src/geotop/energy.balance.cc:798-810
    # Evaporation charged to the bare soil.  GEOtop splits the surface flux
    # into Evap_snow / Evap_glac / Evap_soil on the *active* layer counts of
    # the solve and only Evap_soil reaches the Evap_surface column; the
    # end-of-step snow depth is not the same test.
    evap_soil: float = 0.0
    # glacier (all zero / NaN when the module is off)
    Melt_glac: float = 0.0           # glacier meltwater leaving the base [mm]
    Evap_glac: float = 0.0           # glacier sublimation [mm]
    gwe: float = 0.0                 # glacier water equivalent after the step
    glac_depth: float = 0.0          # glacier thickness after the step [mm]
    glac_T: float = float("nan")     # bulk mass-weighted glacier temperature
    had_glac: bool = False           # glacier present at start of the step
    solver_iterations: int = 0
    solver_residual: float = 0.0
    solver_code: int = 0
    # water balance (Richards): unset (True/0/0.0) whenever no `richards`
    # coupling was passed to step_independent -- matches GEOtop's own
    # `wt` staying 0 for a step that never calls water_balance at all.
    wb_converged: bool = True
    wb_iterations: int = 0
    wb_loss: float = 0.0
    wb_vbottom: float = 0.0     # [m3] volume drained through the bottom this step
    wb_vlat: float = 0.0        # [m3] volume drained laterally this step
    # GEOtop: src/geotop/water.balance.cc:192 (odb[oopnet])
    # GEOtop: src/geotop/geotop.cc:332 (water_balance gated on en == 0)
    # Water reaching the soil surface this step [mm]: Melt + under-canopy net
    # rain, the same quantity passed to the Richards solver. Stays 0 whenever
    # no `richards` coupling runs, matching GEOtop's own `odb[oopnet]` (set
    # only inside `water_balance`, never called when EnergyBalance itself did
    # not converge for the step).
    pnet: float = 0.0
    prof: Optional[LayerProfile] = None   # end-of-step per-layer state
    # canopy (all zero when the point carries no vegetation)
    fc: float = 0.0                  # active canopy fraction [-]
    LSAI: float = 0.0
    z0veg: float = 0.0               # canopy roughness length [m]
    d0veg: float = 0.0               # canopy displacement height [m]
    Psnow_under: float = 0.0
    Prain_under: float = 0.0
    SWv: float = 0.0                 # shortwave absorbed by the canopy [W/m2]
    Tv: float = 0.0                  # canopy temperature [degC] (= Tg if fc=0)
    canopy: Optional[veg.CanopyState] = None


def snapshot_profile(col, richards=None) -> LayerProfile:
    """Capture the final snow column + soil state for output_prof.

    ``soil_th`` always comes from ``col.soil[l].th0`` (meaningful whether or
    not Richards is coupled -- with ``WaterBalance=0`` there is no flow, but
    freezing and thawing still move water between the two phases).
    ``soil_psi`` comes from whichever state actually maintains it: the
    Richards solver's own converged potential for the layers it tracks
    (``1..richards.col.nl``, the spin-up-truncated range), the column's own
    ``P0`` when Richards is off entirely, and -- for the deeper layers below
    ``nl`` that no solver touches -- the hydrostatic inversion of
    ``th0``/``thi0`` through ``psi_teta`` that :mod:`geotop_py.water.init` uses for
    the t=0 profile.
    """
    snow = col.snow
    # GEOtop's write_snow_file loops over all allocated layers and prints any
    # slot whose Dzl is positive, even during the one-step interval in which
    # fresh snow has been deposited but lnum is still zero.
    highest = max((l for l in range(1, snow.max + 1) if snow.Dzl[l] > 0.0),
                  default=0)
    rng = range(1, highest + 1)             # 1 = base .. highest = surface
    nl_richards = richards.col.nl if richards is not None else 0
    soil_psi = []
    soil_ptot = []
    for l in range(1, col.nsoil() + 1):
        s = col.soil[l]
        if l <= nl_richards:
            soil_psi.append(richards.state.P[l])
            soil_ptot.append(richards.state.Ptot[l])
        elif richards is None:
            soil_psi.append(s.P0)
            soil_ptot.append(laws.psi_teta(
                s.th0 + s.thi0, 0.0, s.sat, s.res, s.alpha, s.n, s.m,
                C.PsiMin, s.ss))
        else:
            soil_psi.append(laws.psi_teta(s.th0 + s.thi0, 0.0, s.sat, s.res,
                                          s.alpha, s.n, s.m, C.PsiMin, s.ss))
            soil_ptot.append(laws.psi_teta(
                s.th0 + s.thi0, 0.0, s.sat, s.res, s.alpha, s.n, s.m,
                C.PsiMin, s.ss))
    return LayerProfile(
        Dz=[snow.Dzl[l] for l in rng],
        T=[snow.T[l] for l in rng],
        wice=[snow.w_ice[l] for l in rng],
        wliq=[snow.w_liq[l] for l in rng],
        soil_T=[col.soil_T[k] for k in range(1, col.nsoil() + 1)],
        soil_th=[col.soil[l].th0 for l in range(1, col.nsoil() + 1)],
        soil_thi=[col.soil[l].thi0 for l in range(1, col.nsoil() + 1)],
        soil_psi=soil_psi,
        soil_ptot=soil_ptot,
    )




# GEOtop: src/geotop/output.cc:306-318 (snow), :328-341 (glacier)
def _bulk_snow_T(snow) -> float:
    """Thickness-weighted mean snow/glacier temperature over the pack [C].

    The weight is layer *thickness*, not layer mass. The two agree only while
    the pack has a single density: a fresh, light layer over an old, dense one
    counts for as much of the reported mean as its depth, not as its water
    equivalent.
    """
    depth = 0.0
    acc = 0.0
    for l in range(1, snow.lnum + 1):
        depth += snow.Dzl[l]
        acc += snow.T[l] * snow.Dzl[l]
    return acc / depth if depth > 0.0 else float("nan")


@dataclass
class EnergyPhase:
    """The energy solve of one point step, as the rest of the step reads it.

    ``converged`` is the solve's final verdict: ``False`` means nothing past
    the solve may run for this point in this trial. ``ecol``/``res`` are the
    column and solution of the last solve (the accepted one on success);
    ``active_ns``/``active_ng`` count the ice nodes it ran on, after any merge.
    ``Prain``/``Psnow`` are the under-canopy amounts [mm].
    """
    converged: bool
    ns: int
    ng: int
    active_ns: int
    active_ng: int
    ecol: object
    res: object
    diag: SurfaceDiag
    vp: Optional[veg.VegParams]
    fc: float
    Prain: float
    Psnow: float
    z0veg: float
    d0veg: float
    SWv: float


@dataclass
class WaterInput:
    """What the soil water balance of one point needs from its energy step:
    the water reaching the soil surface ``Pnet`` [mm] and the per-layer
    evapotranspiration sink ``ET`` [mm]."""
    Pnet: float
    ET: List[float]


def energy_phase(col, state: SurfaceState, st: SurfaceStatics, Dt: float,
                 m: Meteo, JDb: float, JDe: float) -> EnergyPhase:
    """Canopy interception, surface forcing and the implicit energy solve with
    its accept/merge/retry verdict loop, for one point.

    Mutates ``col``/``state`` only with what survives a failed solve too: the
    canopy interception, the canopy temperature, the snow age and -- after a
    failed ``-6`` retry -- the top soil layer's folded ice fraction. Everything
    that depends on an accepted solve is left to :func:`post_energy_phase`.
    """
    snow = col.snow
    ns = snow.lnum
    ng = col.ng()
    ecol = flatten(col)
    Ts_surf = ecol.T0[ecol.surface_index]   # skin or top-layer temperature
    snowD = snow.active_depth()             # [mm] GEOtop DEPTH(): 1..lnum

    # GEOtop: src/geotop/energy.balance.cc:336-372 (canopy fraction)
    # GEOtop: src/geotop/energy.balance.cc:418-439 (interception)
    # GEOtop: src/geotop/energy.balance.cc:608-612 (roughness)
    # --- canopy: fraction, interception, roughness
    vp = st.vegpar
    fc = veg.canopy_fraction(snowD, vp, ng) if vp is not None else 0.0
    Prain = (1.0 - fc) * m.Prain
    Psnow = (1.0 - fc) * m.Psnow
    z0veg = d0veg = hveg = 0.0
    wcan_rain_max = wcan_snow_max = 0.0
    if fc > 0.0:
        wcan_rain_max, col.Wcrn, drip_rain = veg.canopy_rain_interception(
            veg.RAIN_MAX_LOADING, vp.LSAI, m.Prain, col.Wcrn)
        wcan_snow_max, col.Wcsn, drip_snow = veg.canopy_snow_interception(
            veg.SNOW_MAX_LOADING, vp.LSAI, m.Psnow, col.Tv, m.wind, Dt,
            col.Wcsn)
        Prain += fc * drip_rain
        Psnow += fc * drip_snow
        col.Wcsn = max(0.0, col.Wcsn)
        z0veg, d0veg, hveg = veg.update_roughness_veg(vp.Hveg, snowD, st.zmu,
                                                      st.zmt)

    canopy_in = veg.CanopyInputs(
        fc=fc, vp=vp, z0veg=z0veg, d0veg=d0veg, hveg=hveg, ng=ng,
        Wcrn=col.Wcrn, Wcsn=col.Wcsn, Wcrnmax=wcan_rain_max,
        Wcsnmax=wcan_snow_max, Tv0=col.Tv) if fc > 0.0 else None

    flux, SWlayer, diag = surface_forcing(
        st, state, ecol, ns, snowD, Ts_surf, m.Ta, m.RH, m.P, m.wind,
        m.tau_cloud, m.Psnow, JDb, JDe, Dt, canopy_in,
        tau_cloud_av=m.tau_cloud_av, LWin_measured=m.LWin_measured)
    SWv = diag.SWv
    if fc <= 0.0 and vp is not None:
        # GEOtop: src/geotop/energy.balance.cc:2399 (EnergyFluxes)
        # GEOtop: src/geotop/energy.balance.cc:1033-1036 (the Tvegetation column)
        # EnergyFluxes sets Tv = Tg whenever the canopy is off, and that value
        # is what the Tvegetation column reports.
        col.Tv = Ts_surf

    # GEOtop: src/geotop/energy.balance.cc:654-749
    # GEOtop: src/geotop/energy.balance.cc:658 (surface = 0)
    # GEOtop's ``do{...}while(sux<0)``: every rejected solve modifies the
    # column and the whole thing is repeated before WBsnow/WBglacier ever see a
    # result.  ``scol`` is the column actually handed to the solver; it differs
    # from the persistent ``ecol`` only for the ``sux=-1`` retry, which
    # re-solves on material node 1 (``surfacemelting=1``) without changing the
    # geometry, so that excess energy enters the snow phase change instead of
    # warming a massless skin.  The next pass through the loop resets
    # ``surface = 0``, exactly as the C++ does.
    active_ns, active_ng = ns, ng
    scol = ecol
    repeat = False
    while True:
        # A repeated solve rewinds the trial soil state (the C++ rebuilds
        # egy->THETA at the top of every SolvePointEnergyBalance call), then
        # refreshes it once per Newton line-search trial through the hook.
        evap_state_reset(diag, scol)
        if repeat:
            # Order matters: the reset above feeds the resistance refresh, as
            # it does in the C++ (THETA is rebuilt, then EnergyFluxes runs).
            _retry_ground_refresh(st, scol, diag, snowD, m.wind)
        cs = diag.canopy
        # A fresh gate per solve attempt: flagTmin/cont are locals of each
        # SolvePointEnergyBalance invocation, retries included.
        gate = NewtonGate()
        diag.gate = gate
        res = solve_column_energy(scol, Dt, flux, options=col.solver_opt,
                    layer_source=SWlayer, gate=gate,
                    iter_hook=evap_state_hook(
                        diag, scol, Dt,
                        transp_layer=(cs.transp_layer if cs is not None
                                      else None),
                        n_transp=(vp.n_transp if (cs is not None
                                                  and vp is not None) else 0)))
        if scol.surface_index == 1:
            # GEOtop: src/geotop/energy.balance.cc:2096-2097
            res.Temp[0] = res.Temp[1]
        code, exhausted = _sux_verdict(res, ecol, active_ns, active_ng,
                                       scol.surface_index)
        if code >= 0 or code == -6:
            break
        if code == -1:
            melt_T0 = list(res.Temp)
            melt_T0[0] = melt_T0[1]
            scol = replace(ecol, T0=melt_T0, surface_index=1)
            _resolve_canopy(st, diag, scol, melt_T0[1])
            repeat = True
            continue
        # -2..-5: a node of the snow+glacier stack was left with less than
        # 1e-5 kg m-2 of ice, or the Newton failed over the single remaining
        # snow node above a glacier.  Merge it in the energy arrays and repeat.
        if code == -2:
            ecol = _merge_energy_snow_layer(ecol, res, exhausted,
                                            lup=1, ldw=active_ns)
            active_ns -= 1
        elif code == -3:
            ecol = _merge_energy_snow_layer(ecol, res, exhausted,
                                            lup=active_ns + 1,
                                            ldw=active_ns + active_ng)
            active_ng -= 1
        elif code == -4:
            # l == ldw, so merge() always takes its ``m=l, n=l-1`` branch: the
            # last glacier node is absorbed by whatever is above it.
            ecol = _merge_energy_snow_layer(ecol, res, active_ns + active_ng,
                                            lup=active_ns,
                                            ldw=active_ns + active_ng)
            active_ng -= 1
        else:                                   # sux=-5
            # l == lup, so merge() always takes its ``m=l+1, n=l`` branch: the
            # last snow node is absorbed by the glacier top below it, never by
            # the node above.
            ecol = _merge_energy_snow_layer(ecol, res, active_ns,
                                            lup=active_ns, ldw=active_ns + 1)
            active_ns -= 1
        scol = ecol
        _resolve_canopy(st, diag, ecol, ecol.T0[ecol.surface_index])
        repeat = True

    # GEOtop: src/geotop/energy.balance.cc:2137-2141 (the sux=-6 guard)
    # GEOtop: src/geotop/energy.balance.cc:702-729 (the fold into the soil)
    # GEOtop: src/geotop/energy.balance.cc:758-762 (sux_minus6_condition)
    # Last-layer complete melt with NO glacier (GEOtop's sux=-6, guarded by
    # ``ns == 1 && ng == 0``): GEOtop temporarily folds its ice into the top
    # soil node, resolves the bare column, then restores a snow layer and gives
    # it the portion of soil thaw attributable to that ice.  The ``ng == 0``
    # guard is essential, not cosmetic: the branch assumes node 2 is the first
    # soil layer, which is false as soon as a glacier sits underneath (that
    # case is sux=-5, handled in the loop).
    if code == -6:
        original = ecol
        bare, saved = _remove_last_snow_for_soil_solve(ecol, res)
        _resolve_canopy(st, diag, bare, bare.T0[0])
        bare_flux = _bare_retry_flux(st, bare, diag, SWlayer[0], snowD, m.wind)
        evap_state_reset(diag, bare)
        bare_gate = NewtonGate()
        diag.gate = bare_gate
        bare_cs = diag.canopy
        bare_res = solve_column_energy(bare, Dt, bare_flux, options=col.solver_opt,
                         layer_source=SWlayer, gate=bare_gate,
                         iter_hook=evap_state_hook(
                             diag, bare, Dt,
                             transp_layer=(bare_cs.transp_layer
                                           if bare_cs is not None else None),
                             n_transp=(vp.n_transp if (bare_cs is not None
                                                       and vp is not None)
                                       else 0)))
        # With no snow or glacier left the verdict is convergence alone: no
        # skin to retry on, no ice node to merge.
        code = 0 if bare_res.converged else 1
        if code == 1:
            # The -6 branch writes the folded ice into the persistent top soil
            # layer *before* the retry, so a failed retry keeps it there -- next
            # to the snow layer it came from, which is left untouched.
            # GEOtop: src/geotop/energy.balance.cc:716-727
            col.soil[1].thi0 = bare.soil[1].thi0
            col.soil[1].th0 = bare.soil[1].th0
        ecol, res = _restore_last_snow(original, bare, bare_res, saved)

    converged = code == 0
    # The canopy temperature is written through a pointer at every flux
    # evaluation, failed solves included; the canopy water increments are
    # added only on the successful return of SolvePointEnergyBalance, so the
    # retries above never commit their store -- only an accepted solve does.
    # GEOtop: src/geotop/energy.balance.cc:2183-2184
    canopy = diag.canopy
    if canopy is not None:
        col.Tv = canopy.Tv
        if converged:
            col.Wcrn += canopy.dWcrn
            col.Wcsn += canopy.dWcsn

    return EnergyPhase(converged=converged, ns=ns, ng=ng, active_ns=active_ns,
                       active_ng=active_ng, ecol=ecol, res=res, diag=diag,
                       vp=vp, fc=fc, Prain=Prain, Psnow=Psnow, z0veg=z0veg,
                       d0veg=d0veg, SWv=SWv)


def post_energy_phase(col, state: SurfaceState, Dt: float, m: Meteo,
                      ep: EnergyPhase, richards=None):
    """Everything a point step does after an accepted energy solve and before
    the soil water balance: glacier and snow mass balance, restratification,
    fresh snow, canopy re-partition and the soil update from the phase change.

    Returns ``(out, water_input)``: the step's :class:`StepOut` without the
    water-balance fields and the profile, which :func:`water_phase` and
    :func:`finish_step` fill in, and the :class:`WaterInput` of the step
    (``None`` when no ``richards`` coupling is passed).
    """
    snow = col.snow
    ns, ng = ep.ns, ep.ng
    active_ns, active_ng = ep.active_ns, ep.active_ng
    ecol, res, diag, vp, fc = ep.ecol, ep.res, ep.diag, ep.vp, ep.fc
    Prain, Psnow = ep.Prain, ep.Psnow
    canopy = diag.canopy

    Tg_final = res.Temp[ecol.surface_index]
    # The first material node: what GEOtop converts the vapour flux at for the
    # reported LE, distinct from the skin the Newton solved for.
    T1_final = res.Temp[1]
    # The fluxes the step books (mass balance, outputs) are the ones the
    # Newton last evaluated, not a fresh evaluation at the final temperature:
    # past the surface-refresh cap the two differ, and GEOtop reports and
    # consumes the frozen ones.
    b = diag.reported if diag.reported is not None \
        else diag.breakdown(Tg_final)
    Evap = b.E * Dt  # kg/m2 == mm (sublimation demand)
    GEF = _ground_heat_flux(ecol, res, active_ns, diag, Tg_final, T1_final)

    # GEOtop: src/geotop/energy.balance.cc:798-810
    # The vapour flux goes to ONE compartment, never split: snow if any snow is
    # left, else the glacier, else the soil.
    Evap_snow = Evap if active_ns > 0 else 0.0
    Evap_glac = Evap if (active_ns == 0 and active_ng > 0) else 0.0

    # The energy arrays cover the whole snow+glacier stack, top-down; both water
    # balances read from them, so build the view once over ns+ng nodes.
    nsng = active_ns + active_ng
    eb = EBSnow(
        ice=[0.0] + [ecol.ice[i] for i in range(1, nsng + 1)],
        liq=[0.0] + [ecol.liq[i] for i in range(1, nsng + 1)],
        deltaw=[0.0] + [res.deltaw[i] for i in range(1, nsng + 1)],
        Temp=[0.0] + [res.Temp[i] for i in range(1, nsng + 1)],
        Dlayer=[0.0] + [ecol.Dlayer[i] for i in range(1, nsng + 1)],
    )

    # GEOtop: src/geotop/energy.balance.cc:816-829
    # GLACIER FIRST: WBglacier and its own restratification run before the snow
    # ones.  Its meltwater is added to Pnet alongside the snow's, but the two
    # never mix inside the pack -- the glacier's percolates straight out, it
    # does not enter the snow above.
    Melt_glac = 0.0
    if col.glac is not None:
        # Called even with active_ng == 0: a sux=-4 merge hands the last
        # glacier node to the stack above, and WBglacier is what then clears
        # the orphaned layers out of the glacier column.
        Melt_glac = WBglacier(Dt, active_ns, active_ng, col.glac,
                              col.glac_wb_par, Evap_glac, eb)
        snow_layer_combination(col.glac, col.alpha_snow, m.Ta,
                               col.inf_glac_layers, col.max_weq_glac, 1e10)

    # Unconditional, as in GEOtop: with active_ns == 0 the call is what clears
    # the snow column left orphaned by a sux=-5 merge into the glacier.
    Melt, RainOnSnow = WBsnow(Dt, active_ns, snow, col.wb_par, col.slope,
                              Prain, Evap_snow, eb)

    snow_layer_combination(snow, col.alpha_snow, m.Ta, col.inf_snow_layers,
                           col.max_weq_snow, col.maxSWE)
    HN = 0.0
    if Psnow > 0.0:
        Dz = fresh_snow_depth(col, Psnow, m.Ta, m.wind)
        new_snow(col.alpha_snow, snow, Psnow, Dz, m.Ta)
        HN = Dz

    # GEOtop: src/geotop/energy.balance.cc:850-894
    # Re-partition the canopy store when the new snow depth moves fc: a growing
    # canopy fraction dilutes what is already on the leaves, a shrinking one
    # dumps the difference to the ground -- the snow as fresh snow at a
    # hardcoded 300 kg/m3, the water to Pnet.
    shed_rain = 0.0
    if vp is not None and vp.LSAI >= veg.LSAIthres and active_ng == 0:
        fc0 = fc
        fc_new = vp.cf * (1.0 - veg.snow_burying_fraction(
            snow.active_depth(), vp)) ** vp.expveg
        if fc_new > fc0:
            col.Wcrn *= fc0 / fc_new
            col.Wcsn *= fc0 / fc_new
        elif fc_new < fc0:
            shed = col.Wcsn * (fc0 - fc_new)
            new_snow(col.alpha_snow, snow, shed, shed * 1000.0 / 300.0, col.Tv)
            # GEOtop: src/geotop/energy.balance.cc:886
            shed_rain = col.Wcrn * (fc0 - fc_new)
        if fc_new < 1.0e-6:
            col.Wcrn = 0.0
            col.Wcsn = 0.0

    # GEOtop: src/geotop/energy.balance.cc:2197-2237 (update_soil_land)
    # GEOtop: src/geotop/energy.balance.cc:2209-2218 (the ET accumulation)
    # NB: no soil-water depletion by evaporation.  With Richards disabled
    # GEOtop never removes the evaporated mass from ``sl->th``:
    # ``update_soil_land`` rewrites th/thi from ``liq+deltaw``/``ice-deltaw``
    # only, and routes ``soil_evap_layer_*`` into the purely diagnostic
    # accumulator ``ET``, which nothing but the water balance and the
    # cumulative output ever reads.  The total soil water is therefore
    # *exactly constant* for the whole run -- verified against GEOtop's
    # own SoilLiqContentProfileFile/SoilIceContentProfileFile on
    # tests/gt_sim1D (0.346749 at every step of the season).
    # With Richards on this is the baseline for every soil layer: it sets
    # col.soil_T, and gives the deeper, spin-up-truncated layers below
    # richards.col.nl -- inert to flow either way -- their usual energy-only
    # th0/thi0, while water_phase overwrites layers 1..nl on convergence.
    # `Tstar` itself is not persisted here at all, by either path, since
    # `flatten` recomputes it fresh from th0+thi0 at the top of the *next*
    # energy step.
    carry_soil_state(col, ecol, res, nsng)
    water_input = None
    if richards is not None:
        from ..water import coupling
        egy = coupling.EnergyColumnResult(
            ice=ecol.ice, liq=ecol.liq, deltaw=res.deltaw, Dlayer=ecol.Dlayer,
            Temp=res.Temp, nsng=nsng)
        # GEOtop: src/geotop/energy.balance.cc:2209-2218
        # The three per-layer sink arrays update_soil_land folds
        # into ET, already computed by surface_forcing (bare: diag.soil_evap,
        # the fc<1 branch) and Tcanopy (canopy.evap_layer/transp_layer, the
        # under-canopy soil evaporation and root-weighted transpiration) --
        # update_soil_land applies the (1-fc)/fc weighting itself, matching
        # the C++ term by term.
        th, thi, P, T, ET = coupling.update_soil_land(
            fc, Dt, egy, richards.col.pa, richards.col.nl, richards.state.P,
            T_transp=(canopy.transp_layer if canopy is not None else None),
            E_bare=(diag.soil_evap if diag.soil_evap else None),
            E_veg=(diag.soil_evap_veg if diag.soil_evap_veg
                   else (canopy.evap_layer
                         if canopy is not None else None)))
        richards.state.P, richards.state.thi, richards.state.T = P, thi, T
        # Water reaching the soil surface this step: what leaves the base of
        # the snowpack (`Melt` counts only the water beyond the rain that
        # entered it, hence `+ Prain`, the under-canopy net rain) and of the
        # glacier, then the rain a shrinking canopy sheds, summed in this
        # order.
        # GEOtop: src/geotop/energy.balance.cc:848
        # GEOtop: src/geotop/energy.balance.cc:886
        Pnet = Melt + Melt_glac + Prain
        Pnet += shed_rain
        water_input = WaterInput(Pnet=Pnet, ET=ET)

    # The terrain terms of the next step: GEOtop keeps them outside the
    # per-trial state copy and writes them only after an accepted solve.
    # GEOtop: src/geotop/energy.balance.cc:188-201
    state.swrefl_surr = diag.SWup
    state.tsurf_prev = Tg_final

    out = StepOut(swe=snow.active_swe(), depth=snow.active_depth(), Tg=Tg_final,
                  Melt=Melt, Evap=Evap, HN=HN, converged=True, diag=diag,
                  RainOnSnow=RainOnSnow, snow_T=_bulk_snow_T(snow), GEF=GEF,
                  T1=T1_final,
                  Melt_glac=Melt_glac, Evap_glac=Evap_glac,
                  gwe=col.gwe(),
                  glac_depth=col.glac.depth() if col.glac is not None else 0.0,
                  glac_T=_bulk_snow_T(col.glac) if col.glac is not None
                         else float("nan"),
                  had_glac=(ng > 0),
                  evap_soil=(Evap if (active_ns == 0 and active_ng == 0)
                             else 0.0),
                  had_snow=(ns > 0), solver_iterations=res.iterations,
                  solver_residual=res.residual, solver_code=res.code,
                  fc=fc, LSAI=(vp.LSAI if vp is not None else 0.0),
                  z0veg=ep.z0veg, d0veg=ep.d0veg,
                  Psnow_under=Psnow, Prain_under=Prain,
                  SWv=ep.SWv, Tv=col.Tv, canopy=canopy)
    return out, water_input


def water_phase(col, richards, Dt: float, out: StepOut,
                water_input: WaterInput) -> None:
    """One Richards trial for a point whose energy step was accepted, and its
    write-back into the column's soil layers.

    Mutates ``richards.state`` and, on convergence, ``col.soil[1..nl]``; fills
    the water-balance fields of ``out``.
    """
    from ..water import richards1d
    Pnet = water_input.Pnet
    # The reported net precipitation is the depth measured on the *slope*,
    # i.e. per unit real ground area, while `Pnet` is the depth on the
    # horizontal projection the meteo file supplies -- hence the 1/cos.
    # It is a reporting quantity only: the mass the residual actually
    # infiltrates is the unprojected `Pnet` passed to the solver below.
    # Negative net precipitation contributes nothing rather than draining.
    # GEOtop: src/geotop/water.balance.cc:665-669 (Total_Pnet)
    # GEOtop: src/geotop/water.balance.cc:192 (reported as odb[oopnet])
    # Its /total_pixel is 1 for a single point.
    if Pnet > 0.0:
        out.pnet = Pnet / math.cos(
            min(C.max_slope, richards.col.slope_deg) * C.Pi / 180.0)
    else:
        out.pnet = 0.0
    wb_result = richards1d.solve_richards_1d(
        Dt, richards.state, richards.col, richards.params, Pnet,
        ET=water_input.ET)
    richards.state = wb_result.state
    # Surface runoff, point-column flavour: any head still standing on the
    # surface node above the free-surface boundary depth leaves the column
    # at the end of the step. With that depth at 0 -- the default, and what
    # every reference case uses -- *all* ponded water is discarded: a point
    # column has no neighbour to route it to, so this is the only outflow
    # path the surface node has. Carried over instead, the pond
    # re-infiltrates the next step and saturates the whole profile, and the
    # head it accumulates appears as a uniform offset on every node's
    # matric potential.
    # GEOtop: src/geotop/water.balance.cc:182-184
    if wb_result.converged and richards.state.P[0] > 0.0:
        richards.state.P[0] = min(
            richards.state.P[0],
            max(0.0, -richards.col.bc_depth_free_surface)
            * math.cos(richards.col.slope_deg * C.Pi / 180.0))
    out.wb_converged = wb_result.converged
    out.wb_iterations = wb_result.iterations
    out.wb_loss = wb_result.loss
    out.wb_vbottom = wb_result.Vbottom
    out.wb_vlat = wb_result.Vlat
    # Overwrite layers 1..nl with Richards' own result: evolving th/thi
    # changes the next step's thermal capacity and conductivity.
    # Only on convergence:
    # `RichardsState.th` is a solve *output* (empty until the first one
    # ever succeeds), and a non-converged trial is discarded by the time
    # loop's caller anyway -- the `carry_soil_state` baseline of
    # post_energy_phase is what stands in for it meanwhile.
    if wb_result.converged:
        for l in range(1, richards.col.nl + 1):
            col.soil[l].th0 = wb_result.state.th[l]
            col.soil[l].thi0 = wb_result.state.thi[l]
            col.soil_T[l] = wb_result.state.T[l]
            # The potential too, not just the contents: with flow on it is
            # Richards that owns it, and the next energy step re-derives
            # its liquid content from exactly this number (see
            # ``point_state.flatten``).  Leaving the energy step's own ``P0``
            # here would hand that step a potential the water balance has
            # already superseded.
            col.soil[l].P0 = wb_result.state.P[l]


def finish_step(col, out: StepOut, richards=None) -> StepOut:
    """Attach the end-of-step profile to ``out`` once every phase has run."""
    out.prof = snapshot_profile(col, richards)
    return out


def failed_step_out(col, ep: Optional[EnergyPhase], richards=None) -> StepOut:
    """The record of a point whose energy step did not complete in a trial:
    ``ep`` is its failed solve, or ``None`` when the trial ended at an earlier
    point and this one was never solved.

    Only the state fields (snow, glacier, profile) are meaningful; the fluxes
    are those of the failed solve, or zero, and no output row is built from
    them.
    """
    snow = col.snow
    diag = ep.diag if ep is not None else None
    Tg = ep.res.Temp[ep.ecol.surface_index] if ep is not None else float("nan")
    return StepOut(swe=snow.active_swe(), depth=snow.active_depth(), Tg=Tg,
                   Melt=0.0, Evap=0.0, HN=0.0, converged=False, diag=diag,
                   snow_T=_bulk_snow_T(snow),
                   gwe=col.gwe(),
                   glac_depth=col.glac.depth() if col.glac is not None else 0.0,
                   glac_T=_bulk_snow_T(col.glac) if col.glac is not None
                          else float("nan"),
                   had_snow=(snow.lnum > 0), had_glac=(col.ng() > 0),
                   solver_iterations=(ep.res.iterations if ep is not None else 0),
                   solver_residual=(ep.res.residual if ep is not None else 0.0),
                   solver_code=(ep.res.code if ep is not None else 0),
                   Tv=col.Tv, prof=snapshot_profile(col, richards))


def step_independent(col, state: SurfaceState, st: SurfaceStatics, Dt: float,
                     m: Meteo, JDb: float, JDe: float,
                     richards=None) -> StepOut:
    """One point step on its own: :func:`energy_phase`, then -- only if the
    energy solve was accepted -- :func:`post_energy_phase` and, with a
    ``richards`` coupling (:class:`geotop_py.water.coupling.RichardsCoupling`),
    :func:`water_phase`.

    With ``richards=None`` the soil keeps its total water (``WaterBalance=0``,
    no flow). With it, this step's phase-change result seeds one Richards
    trial whose ``th``/``thi`` feed the next step's thermal properties.
    A failed energy solve returns :func:`failed_step_out` and leaves the water
    balance untouched."""
    ep = energy_phase(col, state, st, Dt, m, JDb, JDe)
    if not ep.converged:
        return failed_step_out(col, ep, richards)
    out, water_input = post_energy_phase(col, state, Dt, m, ep, richards)
    if richards is not None:
        water_phase(col, richards, Dt, out, water_input)
    return finish_step(col, out, richards)


@dataclass
class NominalStep:
    JD0: float          # start-of-run JD, for meteo_at's own interpolation
    Dt_nominal: float    # [s]


def simulate_energy_balance(col, state: SurfaceState, st: SurfaceStatics,
                            min_Dt: float, steps: List[NominalStep], meteo_at,
                            richards=None):
    """Run geotop.cc's own Dt-subdivision loop (:mod:`geotop_py.point.time_loop`)
    around :func:`step_independent`, for one point.

    ``meteo_at(JDb, JDe) -> Meteo`` stands in for GEOtop's ``meteo_distr``:
    every trial -- including a retried, halved one -- needs the meteo
    re-evaluated over its *own* ``[JDb, JDe]`` window, exactly as
    ``geotop.cc``'s inner loop recomputes ``JDb``/``JDe`` from the elapsed
    ``t`` before every attempt (not just once per nominal step). Wiring a
    real ``meteodistr.py`` station in here is the caller's job; this function
    only needs the callback shape.

    ``richards`` (a :class:`geotop_py.water.coupling.RichardsCoupling`, optional)
    adds the water balance to each trial: it runs only when the energy step
    was accepted, and the trial succeeds only when both do. ``None`` (the
    default) reproduces the ``WaterBalance=0`` branch of that loop.

    Returns ``(col, state, richards, outputs)`` -- the final persistent
    state (``richards`` echoed back, ``None`` if it was never passed) and one
    :class:`StepOut` per *committed* GEOtop sub-step, not one per nominal
    step: a nominal step that needed retries produces more than one, exactly
    as GEOtop's own output would (each committed sub-step is a real,
    separate row of the persistent state's evolution, including a
    give-up-at-min_Dt one).
    """
    outputs: List[StepOut] = []
    current = [col, state, richards]
    for nstep in steps:
        def attempt(t, Dt, nstep=nstep):
            JDb = nstep.JD0 + t / 86400.0
            JDe = nstep.JD0 + (t + Dt) / 86400.0
            m = meteo_at(JDb, JDe)
            c, s, r = current
            trial_col = c.copy()
            trial_state = s.copy()
            trial_richards = r.copy() if r is not None else None
            out = step_independent(trial_col, trial_state, st, Dt, m, JDb, JDe,
                                   richards=trial_richards)
            if out.converged:
                keep_terrain_terms(s, trial_state)
            ok = out.converged and out.wb_converged
            return ok, (trial_col, trial_state, trial_richards, out)

        def commit(payload):
            current[:] = payload[:3]
            outputs.append(payload[3])

        time_loop.run(nstep.Dt_nominal, min_Dt, attempt, commit)
    col, state, richards = current
    return col, state, richards, outputs


def keep_terrain_terms(persistent: SurfaceState, trial: SurfaceState) -> None:
    """Copy the terrain terms an accepted energy step wrote into ``trial``
    over to ``persistent`` straight away, whether or not the trial is
    committed: they are not part of the per-trial state, so a retried trial
    sees those of the discarded one."""
    # GEOtop: src/geotop/energy.balance.cc:188-201
    # GEOtop: src/geotop/geotop.cc:299-306
    persistent.swrefl_surr = trial.swrefl_surr
    persistent.tsurf_prev = trial.tsurf_prev
