"""Initial soil state: the hydrostatic profile and the t=0 freezing correction.

Two independent things happen here, always in this order:

1. Build an initial matric potential ``P`` for every node, either from a
   known water table depth (a hydrostatic profile, one value per node
   derived from its own depth) or, when the soil parameters already give a
   pressure for every layer, from those pressures directly -- the two are
   alternative ways of specifying the same initial condition, never combined.
2. Derive ``th`` (liquid content) from ``P`` assuming no ice, then -- only
   for a layer at or below freezing -- split that into ice and liquid by
   equilibrium with the layer's own temperature, and recompute ``P`` from
   the corrected liquid content. A layer above freezing skips this entirely
   and keeps its ice-free ``th``/``P`` from step 1.

The result is a state directly usable as a :class:`geotop_py.water.richards1d.RichardsState`
(node 0 is the surface, matching Richards' own convention -- a hydrostatic
water table above the soil surface gives it a positive, ponded ``P[0]``, one
below the surface a negative one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from .. import constants as C
from .. import laws
from ..io.parfile import NUMBER_NOVALUE
from . import soilwater as sw
from .richards1d import RichardsState, node_depths

Vec = Sequence[float]


@dataclass
class SoilInitResult:
    P: List[float]       # [mm] matric potential, node 0 = surface
    th: List[float]      # [-] liquid content, node 0 unused
    thi: List[float]     # [-] ice content, node 0 unused
    T: List[float]        # [degC] temperature, node 0 unused
    Ptot: List[float]     # [mm] total (liquid+ice-equivalent) potential, node 0 unused

    def as_richards_state(self) -> RichardsState:
        return RichardsState(P=list(self.P), thi=list(self.thi), T=list(self.T),
                             th=list(self.th), Ptot=list(self.Ptot))


# GEOtop: src/geotop/input.cc:962-994
def hydrostatic_pressure(pa: Sequence[Sequence[float]], nl: int,
                          init_water_table_depth: float, slope_deg: float
                          ) -> List[float]:
    """Matric potential ``P`` [mm] from a water table depth [mm below the
    surface], node 0 (surface) through node ``nl``.

    ``P[0] = -init_water_table_depth * cos(slope)``: negative (unsaturated
    surface) when the table sits below ground, positive (ponded) if it were
    given negative (above ground) -- the formula does not special-case either.
    Every deeper node adds the hydrostatic increase with depth, using each
    layer's own thickness projected onto the vertical exactly as
    :func:`geotop_py.water.richards1d.node_depths` does.
    """
    dz = [0.0] + [pa[C.jdz][l] for l in range(1, nl + 1)]
    Z = node_depths(dz, slope_deg)
    P0 = -init_water_table_depth * _cos_deg(slope_deg)
    return [P0 - Z[l] for l in range(nl + 1)]


def _cos_deg(deg: float) -> float:
    return math.cos(deg * C.Pi / 180.0)


# GEOtop: src/geotop/input.cc:958-991 (the per-layer-pressure branch)
def layer_pressures(pa: Sequence[Sequence[float]], nl: int) -> List[float]:
    """Matric potential straight from each layer's own ``jpsi`` -- used
    instead of :func:`hydrostatic_pressure` when every layer already has an
    explicit initial pressure (see :mod:`geotop_py.io.soil`'s
    ``init_water_table_depth`` going novalue in that case). Node 0 (surface)
    has no ``jpsi`` of its own and is left at 0.0 -- the caller decides
    whether that is meaningful for this branch (GEOtop does not read it in
    this case either)."""
    return [0.0] + [pa[C.jpsi][l] for l in range(1, nl + 1)]


# GEOtop: src/geotop/input.cc:1019-1063
def apply_freezing_equilibrium(P: Vec, T: Vec, pa: Sequence[Sequence[float]],
                                nl: int) -> SoilInitResult:
    """Split each layer's ice-free liquid content into ice and liquid, for
    layers at or below freezing, and recompute ``P`` from the corrected state.

    A layer's over-saturation term (``max(P, 0) * Ss``, the excess head above
    full saturation) is carried through both the initial and the corrected
    ``theta_from_psi``/``psi_teta`` calls, so it is neither lost nor double
    counted by the freezing correction.
    """
    th = [0.0] * (nl + 1)
    thi = [0.0] * (nl + 1)
    Ptot = [0.0] * (nl + 1)
    P_out = list(P)

    for l in range(1, nl + 1):
        s, r, a, n = pa[C.jsat][l], pa[C.jres][l], pa[C.ja][l], pa[C.jns][l]
        m = 1.0 - 1.0 / n
        Ss = pa[C.jss][l]

        Ptot[l] = P[l]
        th_l = sw.teta_psi(P[l], 0.0, s, r, a, n, m, C.PsiMin, Ss)
        th_oversat = max(P[l], 0.0) * Ss
        th_l -= th_oversat

        if T[l] <= C.Tfreezing:
            thi_l = th_l - sw.teta_psi(laws.Psif(T[l]), 0.0, s, r, a, n, m, C.PsiMin, Ss)
            if thi_l < 0.0:
                thi_l = 0.0
            th_l -= thi_l
            P_out[l] = sw.psi_teta(th_l + th_oversat, thi_l, s, r, a, n, m, C.PsiMin, Ss)
            thi[l] = thi_l

        th[l] = th_l

    return SoilInitResult(P=P_out, th=th, thi=thi, T=list(T), Ptot=Ptot)


def initial_soil_state(pa: Sequence[Sequence[float]], nl: int,
                        init_water_table_depth: float, slope_deg: float
                        ) -> SoilInitResult:
    """The full initial state for one soil column: hydrostatic (or per-layer)
    pressure, then the freezing correction. ``init_water_table_depth`` being
    :data:`geotop_py.io.parfile.NUMBER_NOVALUE` selects the per-layer-pressure
    branch -- see :func:`geotop_py.io.soil.read_soil_parameters`, which sets
    that sentinel exactly when every layer already carries its own ``jpsi``.
    """
    T = [0.0] + [pa[C.jT][l] for l in range(1, nl + 1)]

    if int(init_water_table_depth) != int(NUMBER_NOVALUE):
        P = hydrostatic_pressure(pa, nl, init_water_table_depth, slope_deg)
    else:
        P = layer_pressures(pa, nl)

    return apply_freezing_equilibrium(P, T, pa, nl)
