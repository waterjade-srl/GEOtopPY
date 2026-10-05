"""Van Genuchten soil hydraulics: the constitutive laws Richards' equation calls.

Matric potentials and piezometric heads are in **mm**, hydraulic conductivity
in **mm/s**, water contents are volumetric fractions [-], temperatures degC,
layer thicknesses mm.

The ``*_from_*`` functions read their parameters out of a per-soil-type
parameter matrix ``pa``, indexed ``pa[row][layer]`` with **1-based** rows and
layers (position 0 unused on both axes); the row indices are the ``j*``
constants in :mod:`geotop_py.constants`. They are thin adapters over the
plain-argument laws below them, which is where the physics is.

The three retention-curve primitives these adapters call -- ``teta_psi``,
``psi_teta``, ``dteta_dpsi`` -- live in :mod:`geotop_py.laws`, which the energy
branch already needed; they are re-exported here so this module carries the
whole soil-hydraulics surface.
"""

from __future__ import annotations

from typing import Sequence

from .. import constants as C
from ..laws import dteta_dpsi, psi_teta, teta_psi

__all__ = [
    "psi_teta", "teta_psi", "dteta_dpsi",
    "k_hydr_soil", "psi_saturation",
    "Harmonic_Mean", "Arithmetic_Mean", "Mean",
    "theta_from_psi", "psi_from_theta", "dtheta_dpsi_from_psi",
    "k_from_psi", "psisat_from",
]

#: A soil parameter matrix: ``pa[row][layer]``, both 1-based.
ParamMatrix = Sequence[Sequence[float]]


# GEOtop: src/geotop/pedo.funct.h:145-171
def k_hydr_soil(psi: float, ksat: float, imp: float, i: float, s: float,
                r: float, a: float, n: float, m: float, v: float, T: float,
                ratio: float) -> float:
    """Unsaturated hydraulic conductivity [mm/s] (Mualem-van Genuchten).

    ``psi`` [mm] is capped at the air-entry value, so a positive head gives the
    saturated conductivity rather than extrapolating past it. Three multipliers
    are then applied in order: a floor at ``ratio * ksat``, the temperature
    dependence of water viscosity (frozen below 0 degC, where it is held at its
    0 degC value), and an impedance factor ``10**(-imp*i/(s-r))`` for the ice
    fraction ``i`` blocking the pores.
    """
    psisat = ((1.0 - i / (s - r)) ** (-1.0 / m) - 1.0) ** (1.0 / n) * (-1.0 / a)
    TETA = 1.0 / (1.0 + (a * (-min(psisat, psi))) ** n) ** m

    q = 1.0 - (1.0 - TETA ** (1.0 / m)) ** m
    k = ksat * TETA ** v * (q * q)

    if k / ksat < ratio:
        k = ratio * ksat

    if T >= 0:
        k *= (0.000158685828 * T * T + 0.025263459766 * T + 0.731495819)
    else:
        k *= 0.731495819

    k *= 10.0 ** (-imp * i / (s - r))

    return k


# GEOtop: src/geotop/pedo.funct.h:174-190
def psi_saturation(i: float, s: float, r: float, a: float, n: float,
                   m: float) -> float:
    """Air-entry matric potential [mm] of a layer holding ice fraction ``i``.

    Ice takes pore space away, so the head at which the *remaining* pore space
    saturates rises with ``i``; once ice fills the pores to within 1e-6 the
    layer is treated as already saturated (0). Negative ``i`` is clamped to 0.
    """
    if i < 0:
        i = 0.0
    if 1.0 - i / (s - r) > 1.0e-6:
        psisat = ((1.0 - i / (s - r)) ** (-1.0 / m) - 1.0) ** (1.0 / n) * (-1.0 / a)
    else:
        psisat = 0.0

    return psisat


# GEOtop: src/geotop/pedo.funct.h:193-200
def Harmonic_Mean(D1: float, D2: float, K1: float, K2: float) -> float:
    """Series (harmonic) interface value of ``K`` between layers of depth
    ``D1``, ``D2``: ``(D1+D2)/(D1/K1 + D2/K2)``."""
    return (D1 + D2) / (D1 / K1 + D2 / K2)


# GEOtop: src/geotop/pedo.funct.h:203-206
def Arithmetic_Mean(D1: float, D2: float, K1: float, K2: float) -> float:
    """Parallel interface value of ``K`` between layers of depth ``D1``, ``D2``.

    The weighting is **crossed**: ``(D1*K2 + D2*K1)/(D1+D2)``, ``D1`` paired
    with ``K2``. This is the conductance of two layers side by side seen from
    the interface, not the depth-weighted average of ``K``; because of that
    the usual "harmonic <= arithmetic" ordering does not hold, and pairing each
    depth with its own conductivity -- the reading the name suggests -- gives a
    different number whenever the two layers differ in *both* depth and
    conductivity. Pinned in ``tests/test_soilwater.py``; do not "fix" it.
    """
    return (D1 * K2 + D2 * K1) / (D1 + D2)


# GEOtop: src/geotop/pedo.funct.h:210-224
def Mean(a: int, D1: float, D2: float, K1: float, K2: float) -> float:
    """Interface value of ``K``, harmonic for ``a == 0``, arithmetic for
    ``a == 1``, and 0 for anything else (no error, no default)."""
    if a == 0:
        return Harmonic_Mean(D1, D2, K1, K2)
    elif a == 1:
        return Arithmetic_Mean(D1, D2, K1, K2)
    else:
        return 0.0


# GEOtop: src/geotop/pedo.funct.h:239-251
def theta_from_psi(psi: float, ice: float, l: int, pa: ParamMatrix,
                   pmin: float) -> float:
    """Volumetric liquid water content [-] of layer ``l`` at head ``psi`` [mm]."""
    s = pa[C.jsat][l]
    res = pa[C.jres][l]
    a = pa[C.ja][l]
    n = pa[C.jns][l]
    m = 1.0 - 1.0 / n
    Ss = pa[C.jss][l]

    return teta_psi(psi, ice, s, res, a, n, m, pmin, Ss)


# GEOtop: src/geotop/pedo.funct.h:254-264
def psi_from_theta(th: float, ice: float, l: int, pa: ParamMatrix,
                   pmin: float) -> float:
    """Matric potential [mm] of layer ``l`` holding liquid content ``th`` [-]."""
    s = pa[C.jsat][l]
    res = pa[C.jres][l]
    a = pa[C.ja][l]
    n = pa[C.jns][l]
    m = 1.0 - 1.0 / n
    Ss = pa[C.jss][l]

    return psi_teta(th, ice, s, res, a, n, m, pmin, Ss)


# GEOtop: src/geotop/pedo.funct.h:268-279
def dtheta_dpsi_from_psi(psi: float, ice: float, l: int, pa: ParamMatrix,
                         pmin: float) -> float:
    """Specific moisture capacity ``dtheta/dpsi`` [mm^-1] of layer ``l``.

    This is the diagonal term of Richards' Jacobian; above the air-entry head
    it degenerates to the specific storativity ``Ss``.
    """
    s = pa[C.jsat][l]
    res = pa[C.jres][l]
    a = pa[C.ja][l]
    n = pa[C.jns][l]
    m = 1.0 - 1.0 / n
    Ss = pa[C.jss][l]

    return dteta_dpsi(psi, ice, s, res, a, n, m, pmin, Ss)


# GEOtop: src/geotop/pedo.funct.h:283-295
def k_from_psi(jK: int, psi: float, ice: float, T: float, l: int,
               pa: ParamMatrix, imp: float, ratio: float) -> float:
    """Hydraulic conductivity [mm/s] of layer ``l`` at head ``psi`` [mm].

    ``jK`` selects which saturated conductivity row to start from -- ``C.jKn``
    for the vertical direction, ``C.jKl`` for the lateral one -- so the same
    law serves both without a second copy of the van Genuchten parameters.
    """
    kmax = pa[jK][l]
    s = pa[C.jsat][l]
    res = pa[C.jres][l]
    a = pa[C.ja][l]
    n = pa[C.jns][l]
    m = 1.0 - 1.0 / n
    v = pa[C.jv][l]

    return k_hydr_soil(psi, kmax, imp, ice, s, res, a, n, m, v, T, ratio)


# GEOtop: src/geotop/pedo.funct.h:299-308
def psisat_from(ice: float, l: int, pa: ParamMatrix) -> float:
    """Air-entry matric potential [mm] of layer ``l`` holding ice ``ice`` [-]."""
    s = pa[C.jsat][l]
    res = pa[C.jres][l]
    a = pa[C.ja][l]
    n = pa[C.jns][l]
    m = 1.0 - 1.0 / n

    return psi_saturation(ice, s, res, a, n, m)
