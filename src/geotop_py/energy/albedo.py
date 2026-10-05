"""Snow/ground albedo and net shortwave -- Module R3 of the surface forcing.

The albedo is *stateful*: ``snowage`` (non-dimensional) grows every step by a
temperature-driven grain-growth rate and is scaled back toward zero by fresh
snowfall. Fresh snow -> low ``snowage`` -> high albedo; aged snow -> albedo
decays toward the ``SnowAgingCoeff`` floor.

Two spectral bands (vis, nir), each with a diffuse and a beam albedo; the value
the surface energy balance uses is the mean ``0.5*vis + 0.5*nir`` per stream.
``AlbExtParSnow`` (AEP) blends toward the ground albedo under shallow snow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TK = 273.15


# ---------------------------------------------------------------------------
# ground and snow albedo
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/radiation.cc:1050-1057
def find_albedo(dry_alb: float, sat_alb: float, wat: float,
                res_wc: float, sat_wc: float) -> float:
    """Bare-ground albedo linearly interpolated by top-soil water content."""
    return dry_alb + (sat_alb - dry_alb) * (wat - res_wc) / (sat_wc - res_wc)


# GEOtop: src/geotop/snow.cc:913-925
def Fzen(cosinc: float) -> float:
    """Zenith-angle albedo increase for the direct beam."""
    if cosinc < 0.5:
        b = 2.0
        return (1.0 / b) * ((b + 1.0) / (1.0 + 2.0 * b * cosinc) - 1.0)
    return 0.0


def _zero(cosinc: float) -> float:
    return 0.0


# GEOtop: src/geotop/snow.cc:893-906
def snow_albedo(ground_alb: float, snowD: float, AEP: float,
                freshsnow_alb: float, C: float, tsnow: float,
                cosinc: float, F) -> float:
    """Snow albedo. ``tsnow`` = non-dim snow age, ``C`` = SnowAgingCoeff,
    ``F`` = ``Fzen`` (beam) or ``_zero`` (diffuse). Blends toward ``ground_alb``
    when the cover is shallower than ``AEP``; ``snowD`` and ``AEP`` in mm."""
    Fage = 1.0 - 1.0 / (1.0 + tsnow)
    A = freshsnow_alb * (1.0 - C * Fage)
    A += 0.4 * (1.0 - A) * F(cosinc)
    if snowD < AEP:
        w = (1.0 - snowD / AEP) * math.exp(-snowD * 0.5 / AEP)
        A = w * ground_alb + (1.0 - w) * A
    return A


# ---------------------------------------------------------------------------
# snow age
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/snow.cc:865-887
def update_snow_age(snowage: float, Psnow: float, Ts: float, Dt: float,
                    Prestore: float) -> float:
    """Return the advanced non-dimensional snow age (Tarboton & Luce 1996).

    Three grain-growth rates -- vapour diffusion, melt/refreeze, dirt -- add to
    the age each step, and fresh snow scales it back down: ``Prestore`` mm of
    snowfall in one step take it to zero. ``Psnow`` [mm over the step] is the
    snowfall *above* the canopy, ``Ts`` [C] the top snow layer's temperature,
    ``Dt`` [s]. Fresh snow -> low age -> high albedo.
    """
    r1 = math.exp(5000.0 * (1.0 / TK - 1.0 / (Ts + TK)))
    r2 = min(r1 ** 10, 1.0)
    r3 = 0.3
    return max(0.0, (snowage + (r1 + r2 + r3) * Dt * 1.0e-6)
               * (1.0 - Psnow / Prestore))


# GEOtop: src/geotop/snow.cc:967-978
def non_dimensionalize_snowage(snowage: float, Ta: float) -> float:
    """Scale a snow age by the grain-growth rate at ``Ta`` [C].

    Uses 273.16, not the 273.15 of :func:`update_snow_age`, as GEOtop does.
    """
    r1 = math.exp(5000.0 * (1.0 / 273.16 - 1.0 / (Ta + 273.16)))
    r2 = min(r1 ** 10, 1.0)
    r3 = 0.3
    return snowage * ((r1 + r2 + r3) * 1.0e-6)


# GEOtop: src/geotop/input.cc:1733
# GEOtop: src/geotop/input.cc:1799
def initial_snow_age(agesnow0: float, has_snow: bool, Tsnow0: float) -> float:
    """The non-dimensional snow age at start-up from ``InitSnowAge`` [days].

    With an initial pack the age is converted to seconds first; without one it
    is not, but it is still non-dimensionalized, as in GEOtop. ``Tsnow0`` is
    ``InitSnowTemp`` [C].
    """
    age = agesnow0 * 86400.0 if has_snow else agesnow0
    return non_dimensionalize_snowage(age, Tsnow0)


# ---------------------------------------------------------------------------
# albedos + net shortwave
# ---------------------------------------------------------------------------
@dataclass
class AlbedoParams:
    # GEOtop: src/geotop/parameters.cc:1351 (FreshSnowReflVis default)
    avo: float = 0.9               # FreshSnowReflVis
    airo: float = 0.65            # FreshSnowReflNIR
    aep: float = 50.0             # AlbExtParSnow
    aging_vis: float = 0.35       # SnowAgingCoeffVis
    aging_nir: float = 0.75       # SnowAgingCoeffNIR
    # ground (SoilAlb{Vis,NIR}{Dry,Wet}); Wet == saturated
    avis_dry: float = 0.16
    avis_wet: float = 0.08
    anir_dry: float = 0.33
    anir_wet: float = 0.16
    soil_res: float = 0.0         # top-soil residual/sat water content
    soil_sat: float = 0.4


# GEOtop: src/geotop/energy.balance.cc:454-533
def albedos(snowD: float, snowage: float, cosinc: float, theta_sup: float,
            par: AlbedoParams):
    """Return ``(avis_b, avis_d, anir_b, anir_d)`` for the step. ``snowD`` [mm],
    ``theta_sup`` = top-soil liquid content, which sets the bare-ground albedo
    when there is no snow.

    Under snow the ground albedo that the shallow-cover blend interpolates
    toward is **zero**, not the bare-ground value: the substrate the blend sees
    is a black lower boundary. Physically the blend is then a pure attenuation
    of the snow albedo with depth, and only a snow-free surface ever reports the
    soil's own reflectance.
    """
    # GEOtop: src/geotop/energy.balance.cc:238 (avis_ground, anir_ground = 0.)
    # GEOtop: src/geotop/energy.balance.cc:476-481 (find_albedo, no-snow branch)
    # GEOtop: src/geotop/energy.balance.cc:459-463 (snow branch)
    # GEOtop: src/geotop/energy.balance.cc:529-532 (beam update)
    # The zero is GEOtop's, and it is a quirk of the original rather than a
    # stated model choice: avis_ground/anir_ground are locals of
    # PointEnergyBalance initialised to 0. and assigned by find_albedo only
    # inside the no-snow branch, so the snow branch above it and the beam
    # update below it both read the 0.
    # Passing the real ground albedo instead raises the surface albedo under
    # every shallow cover -- visible from the first trace snowfall onward.
    if snowD > 0:
        avis_g = 0.0
        anir_g = 0.0
        avis_d = snow_albedo(avis_g, snowD, par.aep, par.avo, par.aging_vis, snowage, 0.0, _zero)
        anir_d = snow_albedo(anir_g, snowD, par.aep, par.airo, par.aging_nir, snowage, 0.0, _zero)
        avis_b = snow_albedo(avis_g, snowD, par.aep, par.avo, par.aging_vis, snowage, cosinc, Fzen)
        anir_b = snow_albedo(anir_g, snowD, par.aep, par.airo, par.aging_nir, snowage, cosinc, Fzen)
    else:
        avis_g = find_albedo(par.avis_dry, par.avis_wet, theta_sup, par.soil_res, par.soil_sat)
        anir_g = find_albedo(par.anir_dry, par.anir_wet, theta_sup, par.soil_res, par.soil_sat)
        avis_b = avis_d = avis_g
        anir_b = anir_d = anir_g
    return avis_b, avis_d, anir_b, anir_d


def swnet(SWbeam: float, SWdiff: float, alb):
    """Net (absorbed) and reflected shortwave from the four albedos
    ``(avis_b, avis_d, anir_b, anir_d)``."""
    avis_b, avis_d, anir_b, anir_d = alb
    # The two halves are grouped differently in the two sums, and not for
    # symmetry: the absorbed share subtracts each band from 1.0 in turn
    # (``1.0 - 0.5*avis_d - 0.5*anir_d``) while the reflected share adds them
    # first.  Pre-summing the albedo and writing ``1.0 - a_d`` rounds
    # differently, and that bit reaches the surface net shortwave, hence the
    # top soil node.
    # GEOtop: src/geotop/energy.balance.cc:534-538
    sw_abs = (SWdiff * (1.0 - 0.5 * avis_d - 0.5 * anir_d)
              + SWbeam * (1.0 - 0.5 * avis_b - 0.5 * anir_b))
    sw_up = (SWdiff * (0.5 * avis_d + 0.5 * anir_d)
             + SWbeam * (0.5 * avis_b + 0.5 * anir_b))
    return sw_abs, sw_up
