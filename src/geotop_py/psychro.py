"""Psychrometrics and the standard atmosphere.

Verbatim translation of GEOtop's own constitutive laws, not textbook formulas
re-derived independently -- see :func:`SpecHumidity_2` for a case where that
distinction changes the answer. Pressures are in **mbar**, temperatures in
**degC**, relative humidity as a fraction in [0, 1] (the meteo *files* carry
it as a percentage; the conversion happens where the file is read).
"""

from __future__ import annotations

import math

#: Mean atmospheric pressure at sea level [mbar].
Pa0 = 1013.25

#: Scale height of the barometric formula below [m].
SCALE_HEIGHT = 8500.0


# GEOtop: src/geotop/meteo.cc:116-120
def pressure(Z: float) -> float:
    """Atmospheric pressure at elevation ``Z`` [m] -> [mbar].

    A plain barometric formula with a fixed scale height, not a temperature
    dependent one.
    """
    return Pa0 * math.exp(-Z / SCALE_HEIGHT)


# GEOtop: src/geotop/meteo.cc:127-130
def temperature(Z: float, Z0: float, T0: float, gamma: float) -> float:
    """Air temperature at ``Z`` from ``T0`` at ``Z0``.

    ``gamma`` is a lapse rate in **degC/km**; the ``1e-3`` converts the
    ``Z - Z0`` elevation difference from metres to kilometres.
    """
    return T0 - (Z - Z0) * gamma * 1.0e-3


# GEOtop: src/geotop/meteo.cc:137-166
def part_snow(prec_total: float, T: float, t_rain: float, t_snow: float):
    """Split precipitation into ``(rain, snow)`` by a linear ramp in air
    temperature between ``t_snow`` (all solid) and ``t_rain`` (all liquid)."""
    if T <= t_snow:
        return 0.0, prec_total
    if T >= t_rain:
        return prec_total, 0.0
    snow = prec_total * (t_rain - T) / (t_rain - t_snow)
    rain = prec_total * (T - t_snow) / (t_rain - t_snow)
    return rain, snow


# GEOtop: src/geotop/meteo.cc:181-188
def SatVapPressure(T: float, P: float) -> float:
    """Saturation vapour pressure [mbar] (Buck)."""
    A = 6.1121 * (1.0007 + 3.46e-6 * P)
    b = 17.502
    c = 240.97
    return A * math.exp(b * T / (c + T))


# GEOtop: src/geotop/meteo.cc:191-199
def SatVapPressure_2(T: float, P: float):
    """Saturation vapour pressure and its temperature derivative ``de/dT``."""
    A = 6.1121 * (1.0007 + 3.46e-6 * P)
    b = 17.502
    c = 240.97
    e = A * math.exp(b * T / (c + T))
    de_dT = e * (b / (c + T) - b * T / ((c + T) * (c + T)))
    return e, de_dT


# GEOtop: src/geotop/meteo.cc:202-209
def TfromSatVapPressure(e: float, P: float) -> float:
    """Inverse of :func:`SatVapPressure`: temperature at which ``e`` saturates."""
    A = 6.1121 * (1.0007 + 3.46e-6 * P)
    b = 17.502
    c = 240.97
    return c * math.log(e / A) / (b - math.log(e / A))


# GEOtop: src/geotop/meteo.cc:211-214
def SpecHumidity(e: float, P: float) -> float:
    """Specific humidity [kg/kg] from vapour pressure ``e`` and total pressure ``P``."""
    return 0.622 * e / (P - 0.378 * e)


# GEOtop: src/geotop/meteo.cc:216-223
def SpecHumidity_2(RH: float, T: float, P: float):
    """Specific humidity and ``dQ/dT`` for a surface at temperature ``T``, RH ``RH``.

    ``dQ_dT`` is **not** the derivative of the ``Q`` returned alongside it:
    ``dQ_de`` is evaluated at the saturation pressure ``e`` and carries no RH
    factor, so the derivative returned is the *saturation* one and is
    independent of RH. GEOtop uses this for a saturated surface, where that is
    exactly what is needed. Pinned in ``tests/test_psychro.py``; do not "fix" it.
    """
    e, de_dT = SatVapPressure_2(T, P)
    Q = SpecHumidity(RH * e, P)
    d = P - 0.378 * e
    dQ_de = 0.622 / d + 0.235116 * e / (d * d)
    return Q, dQ_de * de_dT


# GEOtop: src/geotop/meteo.cc:225-228
def VapPressurefromSpecHumidity(Q: float, P: float) -> float:
    """Inverse of :func:`SpecHumidity`: vapour pressure from specific humidity."""
    return Q * P / (0.378 * Q + 0.622)


# GEOtop: src/geotop/meteo.cc:230-236
def Tdew(T: float, RH: float, Z: float) -> float:
    """Dew-point temperature [degC] at elevation ``Z``. ``RH`` is a fraction,
    clamped at 1 (a station occasionally reports slightly above 100%)."""
    P = pressure(Z)
    e = SatVapPressure(T, P)
    return TfromSatVapPressure(e * min(RH, 1.0), P)


# GEOtop: src/geotop/meteo.cc:238-248
def RHfromTdew(T: float, Td: float, Z: float) -> float:
    """Relative humidity as a fraction in [0, 1], clamped to that range."""
    P = pressure(Z)
    e = SatVapPressure(Td, P)
    es = SatVapPressure(T, P)
    RH = e / es
    if RH > 1:
        RH = 1.0
    if RH < 0:
        RH = 0.0
    return RH


# GEOtop: src/geotop/meteo.cc:250-255
def air_density(T: float, Q: float, P: float) -> float:
    """Moist-air density [kg/m3]."""
    return P * 100 / (287.04 * (T + 273.15)) * \
        (1 - (Q * P / (0.622 + 0.368 * Q)) / P * (1 - 0.622))


def air_cp(T: float) -> float:
    """Specific heat of air at constant pressure [J/(kg K)] (Garratt 1992)."""
    return 1005.00 + (T + 23.15) * (T + 23.15) / 3364.0
