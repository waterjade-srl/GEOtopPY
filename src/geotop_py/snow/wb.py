"""Snow mass balance: compaction, meltwater percolation, fresh-snow input.

Line-by-line translation of ``WBsnow``, ``snow_compactation`` and ``new_snow``
from ``geotop/snow.cc``. These run *after* the energy solver: they take the
per-layer results of :mod:`geotop_py.energy.column` (temperature, ice, liquid and the
phase-change increment ``deltaw``) and write the updated snowpack back into a
:class:`geotop_py.snow.state.SnowColumn`, moving liquid water downward and settling the
layers.

Two indexing conventions meet here, and getting them right is the whole point:

* the **energy** arrays (``EBSnow``) are top-down -- index 1 is the surface,
  thickness in metres -- exactly what :func:`geotop_py.energy.column.solve_column_energy`
  produces;
* the **snowpack** (``SnowColumn``) is bottom-up -- layer 1 is the ground,
  thickness in mm.

``WBsnow`` bridges them with ``m = ns - l + 1``, so this module is where the
column and stratigraphy components actually connect.

Verification is by conservation: :func:`WBsnow` closes the water budget
(snowpack SWE change = rain in - meltwater out - sublimation), compaction only
densifies (depth down, mass fixed), and :func:`new_snow` conserves internal
energy while adding exactly the deposited mass and depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .. import constants as C
from .. import laws
from .state import SnowColumn

SIMPL_SNOW = 1e-1  # snow.cc file-local


@dataclass
class SnowWBParams:
    """Snow water-balance parameters (defaults from geotop.inpts)."""
    snow_maxpor: float = 0.7            # MaxSnowPorosity
    snow_density_cutoff: float = 100.0  # SnowDensityCutoff [kg/m3]
    drysnowdef_rate: float = 1.0        # DrySnowDefRate
    wetsnowdef_rate: float = 1.5        # WetSnowDefRate
    snow_viscosity: float = 1e6         # SnowViscosity [kg s m^-2]
    Sr: float = 0.02                    # IrriducibleWatSatSnow
    max_weq_snow: float = 5.0           # MaxWaterEqSnowLayerContent [kg/m2]


@dataclass
class GlacierWBParams:
    """Glacier water-balance parameters (defaults from geotop.inpts).

    Deliberately *not* a subclass of :class:`SnowWBParams`: the glacier balance
    uses none of the compaction constants, because ``WBglacier`` has no
    Anderson-style settling at all (see :func:`WBglacier`).
    """
    Sr: float = 0.02                    # IrriducibleWatSatGlacier
    max_weq_glac: float = 5.0           # MaxWaterEqGlacLayerContent [kg/m2]
    # GEOtop: src/geotop/snow.cc:1249-1253 (limit on max porosity, in WBglacier)
    max_por: float = 0.95               # hard-coded in GEOtop, not a keyword


@dataclass
class EBSnow:
    """Energy-solver results for the snow layers, top-down (index 1 = surface).

    ``Dlayer`` is in metres, as the energy column stores it; ``ice``/``liq`` are
    the start-of-step masses and ``deltaw`` the melt/freeze increment [kg/m2].
    """
    ice: List[float]
    liq: List[float]
    deltaw: List[float]
    Temp: List[float]
    Dlayer: List[float]


def snow_compactation(snow: SnowColumn, l: int, Dt: float, slope: float,
                      par: SnowWBParams) -> None:
    """Destructive metamorphism + overburden settling of layer l (bottom-up).

    Reduces ``Dzl[l]`` (mm) at fixed mass; snow.cc:snow_compactation.
    """
    theta_i = snow.w_ice[l] / (0.001 * snow.Dzl[l] * C.rho_i)
    theta_w = snow.w_liq[l] / (0.001 * snow.Dzl[l] * C.rho_w)

    if theta_i < par.snow_maxpor:
        # destructive metamorphism
        if theta_i * C.rho_i <= par.snow_density_cutoff:
            c1 = par.drysnowdef_rate
        else:
            c1 = par.drysnowdef_rate * math.exp(
                -0.046 * (C.rho_i * theta_i - par.snow_density_cutoff))
        if theta_w > 0.001:
            c1 *= par.wetsnowdef_rate
        c2 = 2.777e-6      # [s^-1]
        c3 = 0.04          # [K^-1]
        CR1 = -c1 * c2 * math.exp(-c3 * (C.Tfreezing - snow.T[l]))

        # overburden: weight of this layer and everything above it
        eta0 = par.snow_viscosity
        c4 = 0.08          # [K^-1]
        c5 = 0.023         # [m^3 kg^-1]
        load = 0.0
        for m in range(l, snow.lnum + 1):
            load += snow.w_ice[m] + snow.w_liq[m]
        load *= abs(math.cos(slope * C.Pi / 180.0))
        eta = eta0 * math.exp(c4 * (C.Tfreezing - snow.T[l])
                              + c5 * (C.rho_i * theta_i))
        CR2 = -load / eta

        snow.Dzl[l] *= math.exp((CR1 + CR2) * Dt)


def WBsnow(Dt: float, ns: int, snow: SnowColumn, par: SnowWBParams,
           slope: float, Rain: float, Evap: float,
           E: EBSnow) -> Tuple[float, float]:
    """Apply melt/sublimation/rain and percolate liquid down the pack.

    Writes the updated snowpack into ``snow`` and returns ``(Melt, RainOnSnow)``
    where Melt is the net water leaving the base beyond the rain that entered.
    snow.cc:WBsnow.
    """
    Wdt = Rain
    Edt = Evap

    for l in range(snow.lnum, 0, -1):
        if l > ns:
            snow.w_ice[l] = 0.0
            snow.w_liq[l] = 0.0
            snow.Dzl[l] = 0.0
            continue

        m = ns - l + 1

        if Edt > E.ice[m] - E.deltaw[m]:
            # sublimation removes the whole layer; its liquid drains down
            Edt -= (E.ice[m] - E.deltaw[m])
            Wdt += (E.liq[m] + E.deltaw[m])
            snow.w_ice[l] = 0.0
            snow.w_liq[l] = 0.0
            snow.Dzl[l] = 0.0
            continue

        snow.T[l] = E.Temp[m]
        snow.w_ice[l] = max(0.0, E.ice[m] - E.deltaw[m] - Edt)
        snow.w_liq[l] = max(0.0, E.liq[m] + E.deltaw[m] + Wdt)
        snow.Dzl[l] = 1e3 * E.Dlayer[m]
        Edt = 0.0
        Wdt = 0.0

        if snow.w_ice[l] > SIMPL_SNOW * par.max_weq_snow:
            # (a) compaction
            snow_compactation(snow, l, Dt, slope, par)

            # (b) melting shrinks depth at constant density
            if snow.w_ice[l] / E.ice[m] < 1:
                snow.Dzl[l] *= (snow.w_ice[l] / E.ice[m])

            # (c) cap max porosity
            if snow.w_ice[l] / (1e-3 * snow.Dzl[l] * C.rho_w) > par.snow_maxpor:
                snow.Dzl[l] = 1e3 * snow.w_ice[l] / (C.rho_w * par.snow_maxpor)

            # (d) liquid water percolating below (gravitational drainage 5*Se^3*Dt)
            th = snow.w_liq[l] / (1e-3 * snow.Dzl[l] * C.rho_w)
            thi = snow.w_ice[l] / (1e-3 * snow.Dzl[l] * C.rho_i)
            Se = (th - par.Sr * (1.0 - thi)) / ((1.0 - thi) - par.Sr * (1.0 - thi))
            if Se < 0:
                Se = 0.0
            if Se > 1:
                Se = 1.0
            if th > par.Sr * (1.0 - thi):
                Wdt += min(5.0 * Se ** 3 * Dt,
                           (th - par.Sr * (1.0 - thi))) * snow.Dzl[l] * 1e-3 * C.rho_w
            snow.w_liq[l] -= Wdt
        else:
            # thin layer holds no water: it all drains down
            Wdt += snow.w_liq[l]
            snow.w_liq[l] = 0.0

    snow.lnum = ns

    Melt = Wdt - Rain
    RainOnSnow = Rain if Wdt < Rain else 0.0
    return Melt, RainOnSnow


# GEOtop: src/geotop/energy.balance.cc:798-810 (Evap goes to the glacier only without snow)
# GEOtop: src/geotop/energy.balance.cc:828-829 (WBsnow receives all of Prain)
def WBglacier(Dt: float, ns: int, ng: int, glac: SnowColumn,
              par: GlacierWBParams, Evap: float, E: EBSnow) -> float:
    """Apply melt/sublimation to the glacier column and return ``Melt`` [kg/m2].

    Verbatim port of ``snow.cc:WBglacier`` (777-866).  **It is not WBsnow with
    different parameters**, and the differences are the point:

    * no ``snow_compactation``: the only densification is "melting shrinks depth
      at constant density", plus a hard cap at 0.95 porosity;
    * no gravitational drainage ``5*Se^3*Dt``: water above irreducible
      saturation leaves the layer *instantaneously*, in full;
    * below ``simpl_snow * max_weq_glac`` the layer holds no water at all;
    * no rain input.  Rain never enters here: ``PointEnergyBalance`` hands all
      of it to ``WBsnow``, even when a bare glacier is
      what the rain actually lands on.  Reproduced deliberately.

    ``E`` carries the energy-solver results for the *whole* snow+glacier stack,
    top-down; the glacier occupies nodes ``ns+1 .. ns+ng``, so glacier layer
    ``l`` (bottom-up) maps to node ``m = ns + ng - l + 1``.  ``Evap`` reaches
    this function only when there is no snow above.
    """
    Melt = 0.0
    Edt = Evap

    for l in range(glac.lnum, 0, -1):
        if l > ng:
            glac.w_ice[l] = 0.0
            glac.w_liq[l] = 0.0
            glac.Dzl[l] = 0.0
            continue

        m = ns + ng - l + 1

        if Edt > E.ice[m] - E.deltaw[m]:
            # sublimation consumes the whole layer; its liquid becomes melt
            Edt -= (E.ice[m] - E.deltaw[m])
            Melt += (E.liq[m] + E.deltaw[m])
            glac.w_ice[l] = 0.0
            glac.w_liq[l] = 0.0
            glac.Dzl[l] = 0.0
            continue

        glac.T[l] = E.Temp[m]
        glac.w_ice[l] = max(0.0, E.ice[m] - E.deltaw[m] - Edt)
        glac.w_liq[l] = max(0.0, E.liq[m] + E.deltaw[m])
        glac.Dzl[l] = 1e3 * E.Dlayer[m]
        Edt = 0.0

        if glac.w_ice[l] > SIMPL_SNOW * par.max_weq_glac:
            # melting shrinks depth at constant density
            if glac.w_ice[l] / E.ice[m] < 1:
                glac.Dzl[l] *= (glac.w_ice[l] / E.ice[m])

            # cap max porosity (0.95, hard-coded in GEOtop)
            if glac.w_ice[l] / (1e-3 * glac.Dzl[l] * C.rho_w) > par.max_por:
                glac.Dzl[l] = 1e3 * glac.w_ice[l] / (C.rho_w * par.max_por)

            # liquid above irreducible saturation leaves at once (no 5*Se^3*Dt)
            th = glac.w_liq[l] / (1e-3 * glac.Dzl[l] * C.rho_w)
            thi = glac.w_ice[l] / (1e-3 * glac.Dzl[l] * C.rho_i)
            if th > par.Sr * (1.0 - thi):
                Melt += (th - par.Sr * (1.0 - thi)) * glac.Dzl[l] * 1e-3 * C.rho_w
                glac.w_liq[l] = par.Sr * (1.0 - thi) * glac.Dzl[l] * 1e-3 * C.rho_w
        else:
            # thin layer holds no water
            Melt += glac.w_liq[l]
            glac.w_liq[l] = 0.0

    glac.lnum = ng
    return Melt


def new_snow(a: float, snow: SnowColumn, P: float, Dz: float, T: float) -> None:
    """Deposit fresh snow: P [kg/m2] of ice, Dz [mm] of depth, at temperature T.

    For an empty pack it seeds layer 1; otherwise it merges into the top layer,
    conserving internal energy. snow.cc:new_snow.
    """
    if snow.type == 0:
        snow.Dzl[1] += Dz
        snow.w_ice[1] += P
        return

    ns = snow.lnum
    h = laws.internal_energy(snow.w_ice[ns], snow.w_liq[ns], snow.T[ns])
    h += (C.c_ice * P) * (min(T, -0.1) - C.Tfreezing)

    snow.Dzl[ns] += Dz
    snow.w_ice[ns] += P

    wi, wl, Tn = laws.from_internal_energy(a, h, snow.w_ice[ns], snow.w_liq[ns])
    snow.w_ice[ns], snow.w_liq[ns], snow.T[ns] = wi, wl, Tn
