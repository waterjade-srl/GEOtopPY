"""Couple energy-driven phase changes and evapotranspiration to Richards flow.

Convert the energy column's liquid and ice masses to soil water contents and
pressure, then solve the water balance with net precipitation and layerwise
water uptake. Units and indexing are explicit at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

from .. import constants as C
from .. import laws
from . import richards1d
from . import soilwater as sw

Vec = Sequence[float]


@dataclass
class EnergyColumnResult:
    """The per-node outputs of one energy-balance step that Richards needs.

    Indexed like :class:`geotop_py.energy.column.EnergyColumn`: node ``1..nsng`` are
    snow/glacier, ``nsng+1..n`` are soil (``n - nsng`` soil layers). Node 0 is
    unused, matching every other 1-based array in this codebase.
    """
    ice: List[float]        # [kg/m2]
    liq: List[float]        # [kg/m2]
    deltaw: List[float]     # [kg/m2], this step's net melt (+) / freeze (-)
    Dlayer: List[float]     # [m]
    Temp: List[float]       # [degC]
    nsng: int                # snow + glacier node count


# GEOtop: src/geotop/energy.balance.cc:2197-2235
def update_soil_land(fc: float, Dt: float, egy: EnergyColumnResult,
                      pa: Sequence[Sequence[float]], nl: int,
                      P: Vec, T_transp: Optional[Vec] = None,
                      E_bare: Optional[Vec] = None, E_veg: Optional[Vec] = None,
                      ET: Optional[Vec] = None):
    """Update soil ``th``/``thi``/``P``/``T`` from one energy step's results.

    ``P`` is the *current* matric potential [mm], read only to recover this
    layer's own over-saturation before it is overwritten -- it is not the
    fresh Richards solve's own ``P``, which has not run yet at this point in
    the coupled step.

    ``T_transp``/``E_bare``/``E_veg`` are the per-layer canopy-transpiration
    and bare-soil-evaporation rates [kg/(m2 s)] that GEOtop's vegetation and
    surface-evaporation modules produce; ``None`` (the default) treats them
    as zero everywhere, which is exact whenever that physics has nothing to
    contribute this step -- and currently always, since neither is ported
    yet. ``ET`` is the accumulator those rates add into [kg/m2 over the
    step]; ``None`` starts a fresh one at zero.

    Returns ``(th, thi, P_out, T_out, ET_out)``, each a fresh list -- the
    caller decides whether and how to fold them back into persistent state.
    """
    nsng = egy.nsng
    th = [0.0] * (nl + 1)
    thi = [0.0] * (nl + 1)
    P_out = list(P)
    T_out = [0.0] * (nl + 1)
    ET_out = list(ET) if ET is not None else [0.0] * (nl + 1)

    for l in range(1, nl + 1):
        i = l + nsng
        s, r, a, n = pa[C.jsat][l], pa[C.jres][l], pa[C.ja][l], pa[C.jns][l]
        m = 1.0 - 1.0 / n
        Ss = pa[C.jss][l]

        if T_transp is not None and l <= len(T_transp) - 1:
            ET_out[l] += fc * T_transp[l] * Dt
        if E_bare is not None and l <= len(E_bare) - 1:
            ET_out[l] += (1.0 - fc) * E_bare[l] * Dt
        if E_veg is not None and l <= len(E_veg) - 1:
            ET_out[l] += fc * E_veg[l] * Dt

        ice_vol = max(0.0, egy.ice[i]) / (C.rho_w * egy.Dlayer[i])
        psisat = sw.psi_saturation(ice_vol, s, r, a, n, m)
        th_oversat = max(P_out[l] - psisat, 0.0) * Ss

        th[l] = max(0.0, egy.liq[i] + egy.deltaw[i]) / (C.rho_w * egy.Dlayer[i])
        thi[l] = max(0.0, egy.ice[i] - egy.deltaw[i]) / (C.rho_w * egy.Dlayer[i])

        P_out[l] = laws.psi_teta(th[l] + th_oversat, thi[l], s, r, a, n, m,
                                 C.PsiMin, Ss)
        T_out[l] = egy.Temp[i]

    return th, thi, P_out, T_out, ET_out


@dataclass
class RichardsCoupling:
    """Bundles what a coupled EB+WB step needs from the Richards side.

    ``col``/``params`` are the run's static geometry, soil van Genuchten
    parameters and Newton tolerances -- built once, shared, never mutated
    during a run (same status as :class:`geotop_py.energy.surface.SurfaceStatics`).
    ``state`` is the persistent, evolving per-node ``P``/``thi``/``T``/``th``
    Richards owns (same status as :class:`geotop_py.energy.surface.SurfaceState`).

    Passing this to :func:`geotop_py.point.step.step_independent` switches its
    soil update from the ``WaterBalance=0`` path (``carry_soil_state``,
    conserves total soil water exactly, no flow) to the coupled one: this
    step's phase-change result feeds :func:`update_soil_land`, whose output
    seeds a :func:`geotop_py.water.richards1d.solve_richards_1d` trial, whose own
    ``th``/``thi`` are what get written back into the column's soil layers
    for the next energy step's thermal properties.
    """
    col: richards1d.RichardsColumn
    params: richards1d.RichardsParams
    state: richards1d.RichardsState

    def copy(self) -> "RichardsCoupling":
        """Cheap copy for the time loop's attempt/commit pattern -- see
        ``point_state.Column1D.copy``. ``col``/``params`` are static, shared."""
        return replace(self, state=self.state.copy())
