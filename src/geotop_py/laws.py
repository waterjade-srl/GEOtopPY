"""Pure-Python reimplementation of GEOtop's constitutive laws.

Line-by-line translations of the functions in ``geotop/laws.cc``. They exist so
the physics can be read, tested and modified in Python; ``tests/test_laws.py``
pins each one against the C++ oracle in :mod:`geotop_py._cxx`, so any divergence is
caught immediately.

Every function here is pure. No state, no I/O.
"""

from __future__ import annotations

import math

from . import constants as C

# --- fresh snow density --------------------------------------------------

def rho_newlyfallensnow(u: float, Tatm: float) -> float:
    """Jordan et al. (1999). u = wind speed [m/s], Tatm [degC] -> [kg/m3]."""
    if Tatm > 260.15 - C.tk:
        T = Tatm
        if T >= 275.65 - C.tk:
            T = 275.65 - C.tk
        return 500.0 * (1.0 - 0.951 * math.exp(
            -1.4 * (278.15 - C.tk - T) ** -1.15 - 0.008 * u ** 1.7))
    return 500.0 * (1.0 - 0.904 * math.exp(-0.008 * u ** 1.7))


# --- state equation ------------------------------------------------------

def internal_energy(w_ice: float, w_liq: float, T: float) -> float:
    """Internal energy of a snow layer [J/m2], relative to ice at Tfreezing."""
    return (C.c_ice * w_ice + C.c_liq * w_liq) * (T - C.Tfreezing) + C.Lf * w_liq


def from_internal_energy(a: float, h: float, w_ice: float, w_liq: float):
    """Inverse of :func:`internal_energy`. Returns ``(w_ice, w_liq, T)``.

    ``w_ice + w_liq`` carries in the total SWE; the melt partition and the
    temperature come out by Newton iteration on the cubic that results from
    substituting the snow freezing curve.
    """
    SWE = w_ice + w_liq
    if SWE <= 0:
        return 0.0, 0.0, 0.0

    A = -h / (C.c_ice * SWE)
    B = C.c_liq / (C.c_ice * a * a)
    Cc = (C.Lf * SWE - h) / (C.c_ice * SWE * a * a)

    T = 0.0
    cont = 0
    while True:
        T0 = T
        # ``T0 * T0`` and not ``T0 ** 2``: GEOtop writes ``pow_2``, a plain
        # multiplication, while Python's ``**`` goes through libm's pow and
        # rounds differently for some arguments -- and this Newton stops on a
        # 1e-10 increment, so the last bit decides the iterate it stops at.
        T = T0 - (T0 ** 3 + A * (T0 * T0) + B * T0 + Cc) / (
            3.0 * (T0 * T0) + 2 * A * T0 + B)
        cont += 1
        if not (abs(T - T0) > 1.0e-10 and cont < 100):
            break

    w_liq = theta_snow(a, 1.0, T) * SWE
    w_ice = SWE - w_liq
    return w_ice, w_liq, T


# --- freezing curves -----------------------------------------------------

def _max_dtheta_snow(a: float, b: float) -> float:
    # ``** 0.5`` and not ``math.sqrt``: GEOtop writes ``pow(x, 0.5)``, and
    # glibc's pow is not correctly rounded for exponent 0.5 -- it disagrees
    # with sqrt on roughly one argument in 1400.  Python's ``**`` is the same
    # libm pow, so this form is the bit-identical one.
    return -(b / (3 * a * a)) ** 0.5


def theta_snow(a: float, b: float, T: float) -> float:
    """Snow freezing curve: liquid water fraction theta_w = 1/(b + (aT)^2)."""
    if T > 0:
        x = a * 0.0
        return 1.0 / (b + x * x)
    x = a * T
    return 1.0 / (b + x * x)


def dtheta_snow(a: float, b: float, T: float) -> float:
    if T > 0:
        mT = _max_dtheta_snow(a, b)
        x = a * mT
        y = b + x * x
        return -2.0 * a * a * mT / (y * y)
    x = a * T
    y = b + x * x
    return -2.0 * a * a * T / (y * y)


def Psif(T: float) -> float:
    """Soil freezing: matric potential [mm] at temperature T [degC] (T<0)."""
    if T < 0:
        return T * (1000.0 * C.Lf) / (C.GRAVITY * (C.Tfreezing + C.tk))
    return 0.0


# --- van Genuchten water retention ---------------------------------------

def psi_teta(w, i, s, r, a, n, m, pmin, Ss):
    TETAsat = 1.0 - i / (s - r)
    TETAmin = 1.0 / (1.0 + (a * (-pmin)) ** n) ** m

    if w > s - i:
        TETA = TETAsat
        sat = True
    else:
        TETA = (w - r) / (s - r)
        sat = False

    if TETA < TETAmin:
        TETA = TETAmin

    if TETA > 1.0 - 1.0e-6:
        psi = 0.0
    else:
        psi = ((TETA ** (-1.0 / m) - 1.0) ** (1.0 / n)) * (-1.0 / a)
    if sat:
        psi += (w - (s - i)) / Ss
    return psi


def teta_psi(psi, i, s, r, a, n, m, pmin, Ss):
    psisat = ((1.0 - i / (s - r)) ** (-1.0 / m) - 1.0) ** (1.0 / n) * (-1.0 / a)
    sat = psi > psisat

    if psi < pmin:
        psi = pmin

    if not sat:
        if psi > -1.0e-6:
            TETA = 1.0
        else:
            TETA = 1.0 / (1.0 + (a * (-psi)) ** n) ** m
        return r + TETA * (s - r)
    return s - i + Ss * (psi - psisat)


def dteta_dpsi(psi, i, s, r, a, n, m, pmin, Ss):
    """Derivative of teta wrt psi [mm^-1]."""
    psisat = ((1.0 - i / (s - r)) ** (-1.0 / m) - 1.0) ** (1.0 / n) * (-1.0 / a)
    if psi >= psisat:
        return Ss
    return (s - r) * (a * m * n) * ((-a * psi) ** (n - 1.0)) * \
        (1.0 + (-a * psi) ** n) ** (-m - 1.0)


# --- thermal conductivity ------------------------------------------------

def k_thermal(snow: int, a: int, th_liq: float, th_ice: float,
              th_sat: float, k_solid: float) -> float:
    """snow==0 -> soil (Cosenza et al. 2003 quadratic parallel).
    snow==1 -> a==1 Cosenza, a==2 Sturm (1997), a==3 Jordan (1991)."""
    if snow == 0 or (a != 2 and a != 3):
        # C++ pow(expr, 2.0): GCC -O3 rewrites pow(x, 2.0) as x*x (an exact
        # transformation), while Python's ** calls libm pow -- the two round
        # differently for ~0.1% of arguments.  Always square by multiplication.
        x = ((1.0 - th_sat) * math.sqrt(k_solid)
             + th_liq * math.sqrt(C.k_liq)
             + th_ice * math.sqrt(C.k_ice)
             + (th_sat - th_liq - th_ice) * math.sqrt(C.k_air))
        return x * x

    rr = th_liq * C.rho_w + th_ice * C.rho_i
    if a == 2:
        if rr < 156:
            return 0.023 + 0.234 * rr * 1.0e-3
        return 0.138 - 1.01 * rr * 1.0e-3 + 3.233 * rr * rr * 1.0e-6
    # a == 3
    return C.k_air + (7.75e-5 * rr + 1.105e-6 * rr * rr) * (C.k_ice - C.k_air)


# --- volumetric heat capacity --------------------------------------------

def C_snow(wi: float, wl: float, dw: float, a: float, D: float) -> float:
    """Apparent heat capacity of a snow layer of thickness D [mm]."""
    return (C.c_ice * (wi - a * dw) + C.c_liq * (wl + a * dw)) / D


def C_soil(ct: float, sat: float, wi: float, wl: float,
           dw: float, a: float, D: float) -> float:
    return ct * (1.0 - sat) + C_snow(wi, wl, dw, a, D)
