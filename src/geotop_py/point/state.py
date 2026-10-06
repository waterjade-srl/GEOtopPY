"""Persistent state of one point column, and its bridge to the energy solver.

:class:`Column1D` holds everything a step carries forward: the snowpack, the
optional glacier stack, soil layers and temperatures, canopy storage. The
energy solver works on a different view of the same column -- one top-down
node array in metres -- so :func:`flatten` builds that view before each solve
and :func:`carry_soil_state` writes the soil part of the result back.

Index conventions: the snow and glacier stacks are bottom-up in mm; the energy
column is top-down in metres (snow layer ``ns - i + 1`` <-> energy node ``i``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

from .. import constants as C
from .. import laws
from ..energy.column import EnergyColumn, SoilLayer, SolverOptions
from ..snow.mass_balance import GlacierWBParams, SnowWBParams
from ..snow.state import SnowColumn
from ..water import initial_state as soil_init
from ..water import soilwater as sw


@dataclass
class Column1D:
    """Full 1-D column state: the snowpack sitting on a fixed soil substrate.

    With Richards disabled (the project scenario) the soil moisture is frozen in
    place; only the soil temperature evolves, providing the lower boundary for
    the snow. Soil mass is therefore irrelevant to the snow-water budget.
    """
    snow: SnowColumn
    soil: List[SoilLayer]           # 1-based (soil[0] unused), top soil first
    soil_D: List[float]             # soil layer thickness [m], 1-based
    soil_T: List[float]             # soil temperature [degC], 1-based (evolves)

    # AlphaSnow: shape of the snow freezing curve theta_w = 1/(1+(alpha*T)^2).
    # GEOtop's default is 1e5 -- large on purpose, so cold snow holds essentially
    # no liquid water. A small value (e.g. 1) makes the curve absurdly wet
    # (~7.5% liquid at -3.5 degC) and that "equilibrium liquid" drains out as
    # spurious melt while the pack is well below freezing.
    alpha_snow: float = 1e5
    snow_conductivity: int = 3
    slope: float = 0.0              # [deg]
    Tboundary: float = 5.0          # bottom Dirichlet temperature [degC]
    Zboundary: float = 1.0          # depth below last soil node [m]
    Fboundary: float = 0.0
    # 0: a massless skin node lies above layer 1 (GEOtop default);
    # 1: the highest material node is also the surface node.
    nsurface: int = 1

    # GEOtop: src/geotop/energy.balance.cc:1569-1593 (the snow branch for l <= ns+ng)
    # GEOtop: src/geotop/energy.balance.cc:822-823 (snow_layer_combination on the glacier)
    # Glacier: an optional second stack of layers between snow and soil.  It is
    # a SnowColumn because in GEOtop it *is* one -- same Statevar3D, same
    # constitutive laws (SolvePointEnergyBalance selects the snow branch for
    # every l <= ns+ng), same snow_layer_combination called with the glacier
    # parameters.  Only the water balance differs (mass_balance.WBglacier).  ``None``
    # means the glacier module is off, which is not the same as a glacier with
    # zero layers.
    glac: Optional[SnowColumn] = None
    max_weq_glac: float = 5.0       # [kg/m2] max GWE per layer
    inf_glac_override: Optional[List[int]] = None
    glac_wb_par: GlacierWBParams = field(default_factory=GlacierWBParams)

    # GEOtop: src/geotop/input.cc:3941-3948 (initialize_veg_state)
    # Canopy state (GEOtop ``StateVeg``): water and snow held on the leaves
    # [kg/m2] and the canopy temperature [degC].  All three start at zero,
    # Tv included -- it is not initialised from Ta.
    Wcrn: float = 0.0
    Wcsn: float = 0.0
    Tv: float = 0.0

    max_weq_snow: float = 5.0       # [kg/m2] max SWE per layer
    maxSWE: float = 1e10            # total-SWE cap (PointSim uses maxSWE map)

    # GEOtop's preferred merge/split positions (par->inf_snow_layers), built in
    # parameters.cc from SWEbottom/SWEtop/MaxSnowLayersMiddle:
    #   inf[i] = floor(SWEbottom/max_weq) + i,  i = 1..MaxSnowLayersMiddle
    # Length is MaxSnowLayersMiddle+1 (index 0 unused), *not* one-per-layer. When
    # left None the property falls back to the naive [0,1,..,max]; real runs must
    # pass the GEOtop-computed list, otherwise merging targets the wrong pairs.
    inf_override: Optional[List[int]] = None

    wb_par: SnowWBParams = field(default_factory=SnowWBParams)
    solver_opt: SolverOptions = field(default_factory=SolverOptions)

    @property
    def inf_snow_layers(self) -> List[int]:
        # 1-based reference positions for combination/splitting (index 0 unused)
        if self.inf_override is not None:
            return self.inf_override
        return [0] + list(range(1, self.snow.max + 1))

    # GEOtop: src/geotop/parameters.cc:1409-1434
    @property
    def inf_glac_layers(self) -> List[int]:
        """Merge/split reference positions for the glacier stack.

        Built like the snow ones (same formula with GWEbottom/GWEtop/
        MaxGlacLayersMiddle); the naive fallback is only good enough for
        synthetic tests.
        """
        if self.inf_glac_override is not None:
            return self.inf_glac_override
        n = self.glac.max if self.glac is not None else 0
        return [0] + list(range(1, n + 1))

    def ng(self) -> int:
        """Number of active glacier layers (0 when the module is off)."""
        return self.glac.lnum if self.glac is not None else 0

    def gwe(self) -> float:
        return self.glac.swe() if self.glac is not None else 0.0

    def nsoil(self) -> int:
        return len(self.soil) - 1

    def swe(self) -> float:
        return self.snow.swe()

    def copy(self) -> "Column1D":
        """Cheap copy for the attempt/commit pattern of the time loop
        (geotop.cc's ``copy_snowvar3D``/``copy_soil_state``/``copy_veg_state``
        before every Newton trial): every field a step can mutate gets its
        own storage, everything else (static config: ``wb_par``,
        ``solver_opt``, ``inf_override``, ...) is shared, since it is never
        written during a step.
        """
        return replace(
            self, snow=self.snow.copy(),
            soil=[None if s is None else replace(s) for s in self.soil],
            soil_T=list(self.soil_T),
            glac=None if self.glac is None else self.glac.copy(),
        )


def soil_layers_from_pa(pa: List[List[float]], nl: int,
                        init: soil_init.SoilInitResult) -> List[SoilLayer]:
    """Build the energy solver's per-layer soil state from the same ``pa``
    matrix (:mod:`geotop_py.io.soil`, oracle-pinned) and initial condition
    (:func:`geotop_py.water.initial_state.initial_soil_state`) the water-balance side
    already reads, so the energy and water sides start from one soil state.

    ``Tstar`` is left at its dataclass default (0.0): :func:`flatten`
    recomputes it fresh from ``th0 + thi0`` at the top of every energy step,
    as ``SolvePointEnergyBalance`` does, so the value seeded here is never read.
    """
    return [None] + [
        SoilLayer(sat=pa[C.jsat][l], res=pa[C.jres][l], alpha=pa[C.ja][l],
                 n=pa[C.jns][l], ss=pa[C.jss][l], kt=pa[C.jkt][l],
                 ct=pa[C.jct][l], th0=init.th[l], thi0=init.thi[l],
                 P0=init.P[l], fc=pa[C.jfc][l], wp=pa[C.jwp][l])
        for l in range(1, nl + 1)]


# GEOtop: src/geotop/energy.balance.cc:614-648
def flatten(col: Column1D) -> EnergyColumn:
    """Build the top-down energy column (snow over glacier over soil).

    Mirrors ``PointEnergyBalance``: nodes ``1..ns`` are snow (1 = surface),
    ``ns+1..ns+ng`` glacier, the rest soil.  Both ice stacks are stored
    bottom-up, so each maps to nodes by reversing within its own block.
    """
    snow = col.snow
    ns = snow.lnum
    ng = col.ng()
    nsoil = col.nsoil()
    n = ns + ng + nsoil

    Dlayer = [0.0] * (n + 1)
    ice = [0.0] * (n + 1)
    liq = [0.0] * (n + 1)
    T0 = [0.0] * (n + 1)
    esoil: List[Optional[SoilLayer]] = [None] * (nsoil + 1)

    # snow: energy node i (1=surface) <- snowpack layer ns - i + 1
    for i in range(1, ns + 1):
        l = ns - i + 1
        Dlayer[i] = 1e-3 * snow.Dzl[l]
        ice[i] = snow.w_ice[l]
        liq[i] = snow.w_liq[l]
        T0[i] = snow.T[l]

    # glacier: energy nodes ns+1..ns+ng <- glacier layer ns + ng - i + 1
    for i in range(ns + 1, ns + ng + 1):
        l = ns + ng - i + 1
        Dlayer[i] = 1e-3 * col.glac.Dzl[l]
        ice[i] = col.glac.w_ice[l]
        liq[i] = col.glac.w_liq[l]
        T0[i] = col.glac.T[l]

    # soil: energy nodes ns+ng+1..n <- soil layers 1..nsoil (top first)
    for k in range(1, nsoil + 1):
        i = ns + ng + k
        s = col.soil[k]
        # The liquid content the energy step works from is re-derived here from
        # the potential and the ice -- it is not the number the previous step
        # left behind.  Both are round trips of the same water, but through
        # different curves: inverting the potential and reconverting the mass
        # do not land on the same last bit, and by the third timestep the two
        # have parted (measured on Bro: node 6, 1 ULP at call 3).  Tstar below
        # and the mass filled in further down both read the re-derived value.
        # The number written back at the *end* of the step is untouched, so the
        # reported profile stays the mass-derived one, as in GEOtop.
        # GEOtop: src/geotop/energy.balance.cc:283-289
        s.th0 = min(laws.teta_psi(s.P0, s.thi0, s.sat, s.res, s.alpha, s.n,
                                  s.m, C.PsiMin, s.ss),
                    s.sat - s.thi0)
        # GEOtop: src/geotop/energy.balance.cc:1437-1448
        # GEOtop recomputes Tstar at the start of every energy solve from the
        # current total soil water.  Evaporation can change that water content
        # even when Richards is disabled.
        psi0 = laws.psi_teta(s.th0 + s.thi0, 0.0, s.sat, s.res, s.alpha,
                             s.n, s.m, C.PsiMin, s.ss)
        s.Tstar = min(psi0 / (1000.0 * C.Lf / (C.GRAVITY * C.tk)), 0.0)
        Dlayer[i] = col.soil_D[k]
        # The multiplication order matters at the last ULP: energy.balance.cc
        # builds the mass content as th * Dlayer * rho_w, not th * rho_w * D.
        # GEOtop: src/geotop/energy.balance.cc:642-643
        ice[i] = s.thi0 * col.soil_D[k] * C.rho_w
        liq[i] = s.th0 * col.soil_D[k] * C.rho_w
        T0[i] = col.soil_T[k]
        esoil[k] = s

    # PointEnergyBalance sets the massless skin from the uppermost material
    # node before albedo aging and surface-flux assembly.  Waiting until the
    # column solver would make update_snow_age see the zero-initialized skin
    # instead of the actual top-snow temperature.
    T0[0] = T0[1]

    return EnergyColumn(
        Dlayer=Dlayer, ice=ice, liq=liq, T0=T0, nsng=ns + ng, soil=esoil,
        alpha_snow=col.alpha_snow, snow_conductivity=col.snow_conductivity,
        Tboundary=col.Tboundary, Zboundary=col.Zboundary, Fboundary=col.Fboundary,
        surface_index=col.nsurface,
    )


# GEOtop: src/geotop/energy.balance.cc:813-814 (update_soil_land called with ns+ng)
# GEOtop: src/geotop/energy.balance.cc:2220-2233 (update_soil_land indexes l+n)
def carry_soil_state(col: Column1D, ecol: EnergyColumn, res, nsng: int) -> None:
    """Persist soil temperature and phase partition after the energy solve.

    ``nsng`` is the number of ice nodes above the soil -- snow *plus* glacier,
    GEOtop's ``ns+ng`` (``update_soil_land`` indexes ``l+n`` with the same
    quantity).

    ``WaterBalance=0`` disables Richards flow, but it does not disable the
    liquid/ice exchange solved by ``SolvePointEnergyBalance``.  GEOtop writes
    both fractions back after every step; failing to do so freezes the same
    water repeatedly and injects latent heat on every timestep.

    The matric potential is written back too, from the *new* liquid and ice
    fractions.  Total water is conserved with no flow, so inverting the total
    through the unfrozen retention curve instead would give a constant --
    it is the ice fraction, not the total, that drives the potential down as
    a layer freezes.
    """
    for k in range(1, col.nsoil() + 1):
        i = nsng + k
        D = ecol.Dlayer[i]
        dw = res.deltaw[i]
        s = col.soil[k]
        col.soil_T[k] = res.Temp[i]
        # GEOtop: src/geotop/energy.balance.cc:2220-2230 (update_soil_land)
        psisat = sw.psi_saturation(max(0.0, ecol.ice[i]) / (C.rho_w * D),
                                   s.sat, s.res, s.alpha, s.n, s.m)
        th_oversat = max(s.P0 - psisat, 0.0) * s.ss
        s.th0 = max(0.0, (ecol.liq[i] + dw) / (C.rho_w * D))
        s.thi0 = max(0.0, (ecol.ice[i] - dw) / (C.rho_w * D))
        s.P0 = laws.psi_teta(s.th0 + th_oversat, s.thi0, s.sat, s.res,
                             s.alpha, s.n, s.m, C.PsiMin, s.ss)


def fresh_snow_depth(col: Column1D, Psnow: float, Ta: float, wind: float) -> float:
    """Depth [mm] of Psnow [kg/m2] of new snow at the Jordan fresh-snow density."""
    return Psnow * C.rho_w / laws.rho_newlyfallensnow(wind, Ta)
