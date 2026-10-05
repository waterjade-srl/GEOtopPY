"""Implicit vertical energy solver for one surface/snow/soil column.

This is the kernel of GEOtop's ``SolvePointEnergyBalance`` (energy.balance.cc):
one Backward-Euler timestep of vertical heat conduction with phase change,
solved by Newton-Raphson with a non-monotone line search on a tridiagonal
system. The column spans, top to bottom, the snow and glacier layers
(indices ``1..nsng``) and the soil layers below; the bottom boundary is a
Dirichlet sink, exactly as in GEOtop.

**Scope.** The surface energy balance itself -- turbulence, radiation, canopy --
is *not* recomputed here. It enters as a caller-supplied linearised flux
``surface_flux(Tg) -> (EB, dEB_dT)``. That is the real architectural seam: in
GEOtop ``EnergyFluxes`` fills EB and freezes the turbulent resistances for the
timestep, and the column solver consumes it. Keeping the surface as an injected
boundary flux is what lets this component be tested on its own. Surface-node
selection and sub-surface shortwave absorption are supplied through the column
state.

Fidelity: the constitutive laws (``k_thermal``, ``C_snow``/``C_soil``,
``theta_snow``/``dtheta_snow``, van Genuchten) are the oracle-pinned ones from
:mod:`geotop_py.laws`; the assembly, stencils and Thomas solve are verbatim
translations (see :mod:`geotop_py.numerics`). The ``KNe`` weighting is carried
through verbatim; it is the constant :data:`geotop_py.constants.KNe` = 0
(Backward Euler).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .. import constants as C
from .. import laws
from .. import numerics as num

# surface_flux(Tg) -> (EB [W/m2 or the flux units GEOtop uses], dEB_dT)
SurfaceFlux = Callable[[float], Tuple[float, float]]


def _arithmetic_mean(D1: float, D2: float, K1: float, K2: float) -> float:
    """Depth-weighted mean, pedo.funct.cc:Arithmetic_Mean (cross weighting)."""
    return (D1 * K2 + D2 * K1) / (D1 + D2)


@dataclass
class SoilLayer:
    """Van Genuchten + thermal parameters for one soil layer, plus the
    start-of-step liquid/ice content used by the freezing closure."""
    sat: float          # saturated water content theta_s (pa jsat)
    res: float          # residual water content theta_r (pa jres)
    alpha: float        # van Genuchten alpha (pa ja) [mm^-1]
    n: float            # van Genuchten n (pa jns)
    ss: float           # specific storativity (pa jss)
    kt: float           # solid thermal conductivity (pa jkt)
    ct: float           # solid volumetric heat capacity (pa jct)
    th0: float          # start-of-step volumetric liquid content
    thi0: float = 1e12  # start-of-step volumetric ice content (melt clamp)
    P0: Optional[float] = None  # start-of-step matric potential [mm]
    Tstar: float = 0.0  # freezing-point depression temperature [degC]
    fc: float = 0.0     # field capacity (pa jfc), bare-soil evaporation
    wp: float = 0.0     # wilting point (pa jwp), canopy transpiration stress

    def __post_init__(self) -> None:
        # ``P0`` is authoritative: the liquid content the energy step works from
        # is re-derived from it every step (see ``point_state.flatten``).  A caller
        # that gives only ``th0`` therefore has to get the matching potential,
        # not a silent 0.0 -- which would read as saturated and quietly refill
        # the layer.
        if self.P0 is None:
            self.P0 = laws.psi_teta(self.th0, self.thi0, self.sat, self.res,
                                    self.alpha, self.n, self.m, C.PsiMin,
                                    self.ss)

    @property
    def m(self) -> float:
        return 1.0 - 1.0 / self.n


@dataclass
class EnergyColumn:
    """Vertical column state at one (r, c). Material layers are 1-based.

    Layers ``1..nsng`` are snow/glacier, ``nsng+1..n`` soil. ``soil[k]`` holds
    the parameters of soil layer ``k`` (k = 1 for the topmost soil layer),
    matching GEOtop's ``pa[...][l-ns-ng]`` indexing.
    """
    Dlayer: List[float]     # [m]  (energy solver uses metres; strati uses mm)
    ice: List[float]        # [kg/m2]
    liq: List[float]        # [kg/m2]
    T0: List[float]         # previous-step temperature [degC]
    nsng: int               # number of snow + glacier layers
    soil: List[SoilLayer]   # 1-based (soil[0] unused)
    alpha_snow: float
    snow_conductivity: int  # k_thermal snow branch selector (1/2/3)
    Tboundary: float        # bottom Dirichlet temperature [degC]
    Zboundary: float        # depth below last node to the boundary [m]
    Fboundary: float        # bottom geothermal flux term
    surface_index: int = 1  # 0 = separate massless skin; 1 = top material node

    @property
    def n(self) -> int:
        return len(self.Dlayer) - 1

    def _soil_of(self, l: int) -> Tuple[int, SoilLayer]:
        m = l - self.nsng
        return m, self.soil[m]


@dataclass
class SolverOptions:
    # Defaults mirror geotop.inpts: MinLambdaEnergy 1e-5,
    # MaxTimesMinLambdaEnergy 0, ExitMinLambdaEnergy 0. With max_times = 0 the
    # line search accepts a step as soon as lambda reaches the floor, which is
    # what stops it from collapsing to underflow on a stiff phase-change step.
    tol_energy: float = 1e-5
    maxiter_energy: int = 200
    min_lambda_en: float = 1e-5
    max_times_min_lambda_en: int = 0
    exit_lambda_min_en: bool = False
    MM: int = 1                       # non-monotone memory (1 = monotone)


@dataclass
class SolverResult:
    Temp: List[float]
    deltaw: List[float]
    iterations: int
    residual: float
    converged: bool
    code: int = 0                     # 0 ok; <0 GEOtop-style non-convergence


# -------------------------------------------------------------------------
# assembly helpers
# -------------------------------------------------------------------------

def _calc_C(col: EnergyColumn, l: int, a: float, deltaw: Sequence[float]) -> float:
    if l <= col.nsng:
        return laws.C_snow(col.ice[l], col.liq[l], deltaw[l], a, col.Dlayer[l])
    _, s = col._soil_of(l)
    return laws.C_soil(s.ct, s.sat, col.ice[l], col.liq[l], deltaw[l], a, col.Dlayer[l])


def _cond_props(col: EnergyColumn, l: int):
    """(snow_flag, a_cond, sat, kt) for the conductivity call at layer l."""
    if l <= col.nsng:
        return 1, col.snow_conductivity, 1.0, 0.0
    _, s = col._soil_of(l)
    return 0, 1, s.sat, s.kt


def _assemble_residual(col: EnergyColumn, Temp, deltaw, Dt, EB, EB0, KNe, sur,
                       layer_source=None):
    """Build F and the interface conductances Kth1; return (F, Kth1, res, kbb1).

    One routine serves both the initial build and every in-loop rebuild, which
    are the same computation. ``EB`` is the top flux at the current iterate,
    ``EB0`` the one at the first iterate of the solve; ``KNe`` blends them.

    ``layer_source[l]`` (optional) is the shortwave absorbed *inside* node
    ``l`` [W/m2] (snow layers plus the first soil layer), subtracted from
    ``F[l]``. The surface share is already in ``EB``/``EB0``.
    """
    n = col.n
    nsng = col.nsng
    F = [0.0] * (n + 1)
    Kth1 = [0.0] * (n + 1)      # Kth1[l-1] = conductance between nodes l-1 and l

    thw = thi = sat = kt = 0.0
    for l in range(1, n + 1):
        thw = (col.liq[l] + deltaw[l]) / (C.rho_w * col.Dlayer[l])
        thi = (col.ice[l] - deltaw[l]) / (C.rho_i * col.Dlayer[l])
        snow, a_cond, sat, kt = _cond_props(col, l)

        if l > 1:
            thwn = (col.liq[l - 1] + deltaw[l - 1]) / (C.rho_w * col.Dlayer[l - 1])
            thin = (col.ice[l - 1] - deltaw[l - 1]) / (C.rho_i * col.Dlayer[l - 1])
            snown, a_condn, satn, ktn = _cond_props(col, l - 1)

            thwn = _arithmetic_mean(col.Dlayer[l], col.Dlayer[l - 1], thw, thwn)
            thin = _arithmetic_mean(col.Dlayer[l], col.Dlayer[l - 1], thi, thin)
            satn = _arithmetic_mean(col.Dlayer[l], col.Dlayer[l - 1], sat, satn)
            ktn = _arithmetic_mean(col.Dlayer[l], col.Dlayer[l - 1], kt, ktn)

            # snow/soil selector follows the *upper* layer l (as in GEOtop)
            snow_i, a_i = (1, col.snow_conductivity) if l <= nsng else (0, 1)
            Kth1[l - 1] = -2.0 * laws.k_thermal(snow_i, a_i, thwn, thin, satn, ktn) \
                / (col.Dlayer[l - 1] + col.Dlayer[l])
        elif l > sur:
            snow_i, a_i = (1, col.snow_conductivity) if l <= nsng else (0, 1)
            Kth1[l - 1] = -laws.k_thermal(snow_i, a_i, thw, thi, sat, kt) \
                / (col.Dlayer[l] / 2.0)

        C1 = _calc_C(col, l, 1.0, deltaw)
        if l <= nsng and col.ice[l] - deltaw[l] < 1e-7:
            C1 = C.Csnow_at_T_greater_than_0
        C0 = _calc_C(col, l, 0.0, deltaw)
        F[l] += (C.Lf * deltaw[l] + C1 * col.Dlayer[l] * Temp[l]
                 - C0 * col.Dlayer[l] * col.T0[l]) / Dt

        # shortwave penetrating below the surface (micro==1, l<=ns+1)
        if layer_source is not None and l <= nsng + 1:
            F[l] -= layer_source[l]

    # top boundary (Neumann)
    F[sur] -= (1.0 - KNe) * EB + KNe * EB0

    # bottom boundary (Dirichlet sink)
    snow_bb, a_bb = (1, col.snow_conductivity) if n <= nsng else (0, 1)
    kbb1 = laws.k_thermal(snow_bb, a_bb, thw, thi, sat, kt)
    denom = col.Dlayer[n] / 2.0 + col.Zboundary
    F[n] -= ((col.Tboundary - Temp[n]) * (1.0 - KNe) * kbb1
             + (col.Tboundary - col.T0[n]) * KNe * kbb1) / denom
    F[n] -= col.Fboundary

    # conduction (Kth0 == Kth1; the KNe term is dead when KNe == 0)
    num.update_F_energy(sur, n, F, 1.0 - KNe, Kth1, Temp)
    num.update_F_energy(sur, n, F, KNe, Kth1, col.T0)

    res = num.norm_2(F, sur, n)
    return F, Kth1, res, kbb1


def _assemble_jacobian(col, Temp, deltaw, Dt, dEB_dT, Kth1, kbb1, KNe, sur):
    """Build the tridiagonal Jacobian (diagonal dF, off-diagonal udF)."""
    n = col.n
    nsng = col.nsng
    dF = [0.0] * (n + 1)

    for l in range(1, n + 1):
        C1 = _calc_C(col, l, 1.0, deltaw)
        if l <= nsng and col.ice[l] - deltaw[l] < 1e-7:
            C1 = C.Csnow_at_T_greater_than_0
        if l <= nsng:                       # snow apparent capacity
            C1 += C.Lf * (col.ice[l] + col.liq[l]) \
                * laws.dtheta_snow(col.alpha_snow, 1.0, Temp[l]) / col.Dlayer[l]
        else:                               # soil apparent capacity
            _, s = col._soil_of(l)
            if Temp[l] <= s.Tstar:
                C1 += C.rho_w * C.Lf * (C.Lf / (C.GRAVITY * C.tk) * 1e3) \
                    * laws.dteta_dpsi(laws.Psif(Temp[l]), 0.0, s.sat, s.res,
                                      s.alpha, s.n, s.m, C.PsiMin, 0.0)
        dF[l] += C1 * col.Dlayer[l] / Dt

    dF[sur] -= (1.0 - KNe) * dEB_dT                       # Neumann top
    dF[n] += (1.0 - KNe) * kbb1 / (col.Dlayer[n] / 2.0 + col.Zboundary)

    num.update_diag_dF_energy(sur, n, dF, 1.0 - KNe, Kth1)

    udF = [0.0] * (n + 1)
    for l in range(sur, n):
        udF[l] = (1.0 - KNe) * Kth1[l]
    return dF, udF


def _update_deltaw(col: EnergyColumn, Temp, deltaw, sur):
    """Recompute the melt/freeze increment deltaw from the current Temp."""
    n = col.n
    nsng = col.nsng
    for l in range(sur, n + 1):
        if l <= 0:
            continue
        if l > nsng:                        # soil
            m, s = col._soil_of(l)
            th0 = s.th0
            th1 = laws.teta_psi(laws.Psif(min(s.Tstar, Temp[l])), 0.0, s.sat,
                                s.res, s.alpha, s.n, s.m, C.PsiMin, s.ss)
            if th1 > s.th0 + s.thi0:
                th1 = s.th0 + s.thi0
            deltaw[l] = (th1 - th0) * col.Dlayer[l] * C.rho_w
        else:                               # snow
            th0 = col.liq[l] / (col.ice[l] + col.liq[l])
            th1 = laws.theta_snow(col.alpha_snow, 1.0, Temp[l])
            deltaw[l] = (th1 - th0) * (col.ice[l] + col.liq[l])


# -------------------------------------------------------------------------
# the driver
# -------------------------------------------------------------------------

# GEOtop: src/geotop/energy.balance.cc:1837-1894 (THETA refresh, then the flux call)
def solve_column_energy(col: EnergyColumn, Dt: float, surface_flux: SurfaceFlux,
                        options: Optional[SolverOptions] = None,
                        layer_source=None, iter_hook=None, gate=None, trace=None) -> SolverResult:
    """March one timestep. Returns temperatures, phase-change increments and a
    convergence flag. Follows ``SolvePointEnergyBalance`` for either GEOtop
    surface convention (``sur`` = 0 or 1).

    ``layer_source`` (optional) is the per-node shortwave absorbed below the
    surface (``SWlayer[1..ns+1]``); its surface part ``SWlayer[0]`` must already
    be folded into what ``surface_flux`` returns.

    ``iter_hook`` (optional) is called ``iter_hook(Temp, deltaw)`` inside the
    line search, right after the trial state is formed and immediately before
    ``surface_flux`` is evaluated on it.  That is exactly where GEOtop refreshes
    ``egy->THETA`` before calling ``EnergyFluxes_no_rec_turbulence``, and it
    is the only seam through which the surface component can see the Newton's
    own phase-change increment.  The
    default ``None`` leaves the kernel bit-for-bit unchanged.

    ``gate`` (optional) decides per trial whether the surface flux is
    re-evaluated at all: ``gate.trial(trial_Tg, cont) -> bool``.  When it
    returns False the trial's flux stays at its last evaluated value (both the
    residual's EB and the Jacobian's dEB/dT), which is what the reference
    solver does past its surface-refresh iteration cap.  ``None`` (the
    default) re-evaluates at every trial, the historical behaviour here and
    the C++ behaviour below its cap."""
    opt = options or SolverOptions()
    KNe = C.KNe
    sur = col.surface_index
    if sur not in (0, 1):
        raise ValueError(f"surface_index must be 0 or 1, got {sur}")
    if sur == 0:
        # PointEnergyBalance initialises the diagnostic skin temperature from
        # the first material node at the beginning of every timestep.
        col.T0[0] = col.T0[1]
    n = col.n

    Temp = list(col.T0)                     # initial guess = previous state
    deltaw = [0.0] * (n + 1)

    EB, dEB_dT = surface_flux(Temp[sur])
    # GEOtop: src/geotop/energy.balance.cc:1505
    EB0 = EB
    F, Kth1, res, kbb1 = _assemble_residual(col, Temp, deltaw, Dt, EB, EB0, KNe,
                                            sur, layer_source)

    cont = 0
    iter_close = 0
    while iter_close != 1:
        cont += 1
        T1 = list(Temp)

        # dEB_dT is frozen for the line search (as EnergyFluxes freezes it):
        # it is whatever the last flux evaluation set -- at the previous
        # accepted trial, i.e. exactly at T1[sur] -- and past the surface
        # refresh cap it is older still, which is the point of the gate.
        dF, udF = _assemble_jacobian(col, Temp, deltaw, Dt, dEB_dT, Kth1, kbb1, KNe, sur)

        Newton_dir = [0.0] * (n + 1)
        if num.tridiag2(sur, n, udF, dF, udF, F, Newton_dir) == 1:
            return SolverResult(Temp, deltaw, cont, res, False, code=1)

        res0 = [res, 0.0, 0.0]
        lam = [0.0, 0.0, 0.0]
        res_av = res                        # MM == 1 -> monotone
        cont2 = 0
        cont_lambda_min = 0
        iter_close2 = 0
        while iter_close2 != 1:
            cont2 += 1
            if cont2 == 1:
                lam[0] = 1.0
            elif cont2 == 2:
                lam[1] = lam[0]
                res0[1] = res
                lam[0] = C.thmax
            else:
                lam[2] = lam[1]
                res0[2] = res0[1]
                lam[1] = lam[0]
                res0[1] = res
                lam[0] = num.minimize_merit_function(res0[0], lam[1], res0[1],
                                                     lam[2], res0[2])

            for l in range(sur, n + 1):
                Temp[l] = T1[l] + lam[0] * Newton_dir[l]
            _update_deltaw(col, Temp, deltaw, sur)

            # GEOtop: src/geotop/energy.balance.cc:1837-1894
            # The trial state is complete here, and GEOtop refreshes egy->THETA
            # from it before re-evaluating the surface fluxes on the same trial
            # temperature -- but only while
            # the refresh gate says so; past the cap both the hook and the
            # flux evaluation are skipped and EB/dEB_dT stay frozen.
            if gate is None or gate.trial(Temp[sur], cont):
                if iter_hook is not None:
                    iter_hook(Temp, deltaw)
                EB, dEB_dT = surface_flux(Temp[sur])
            F, Kth1, res, kbb1 = _assemble_residual(col, Temp, deltaw, Dt, EB, EB0,
                                                    KNe, sur, layer_source)
            if trace is not None:
                trace(cont, cont2, lam[0], res, Temp[sur])

            if res <= res_av * (1.0 - C.ni_en * lam[0]):
                iter_close2 = 1
            if lam[0] <= opt.min_lambda_en:
                cont_lambda_min += 1
            if cont_lambda_min > opt.max_times_min_lambda_en:
                if opt.exit_lambda_min_en:
                    return SolverResult(Temp, deltaw, cont, res, False, code=1)
                iter_close2 = 1
                cont_lambda_min = 0

        if res <= opt.tol_energy:
            iter_close = 1
        if cont >= opt.maxiter_energy:
            iter_close = 1

    converged = res <= opt.tol_energy
    return SolverResult(Temp, deltaw, cont, res, converged,
                        code=0 if converged else -1)
