"""Surface radiation forcing, ported verbatim from GEOtop ``radiation.cc``.

Compute surface radiation in Python from meteorological forcing, using
GEOtop's solar geometry, atmospheric transmissivity and quadrature formulas.

Module R1 -- **shortwave**: solar geometry (``sun``, ``SolarHeight``,
``SolarAzimuth``, ``Sinalpha``, ``Cosinc``), clear-sky atmospheric transmittance
(``atm_transmittance``, the active non-Iqbal branch), the Erbs diffuse/global
split (``diff2glob``) and the beam/diffuse assembly (``shortwave_radiation``).
Sub-hourly quantities are time-averaged over the step with the same adaptive
Simpson quadrature GEOtop uses (``adaptiveSimpsons2``, verbatim of
``tensors3D.cc``).

All angles in radians. ``others`` mirrors GEOtop's ``sun[0..11]`` layout so the
integrand helpers read identically to the C++:

    0 latitude   1 Delta      2 dh (hour offset)  3 RH   4 T[C]   5 P[mbar]
    6 slope      7 aspect     8 Lozone            9 alpha 10 beta 11 albedo

The active ``atm_transmittance`` only uses P, RH, T -- indices 8..11 are inert
here (they feed the disabled Iqbal branch), kept for a faithful signature.
With ``Surroundings=0`` (this run) ``SWrefl_surr = 0``, so the diffuse
``(1-sky)*SWrefl_surr`` term vanishes.
"""

from __future__ import annotations

import ctypes as _ctypes
import ctypes.util as _ctypes_util
import math
from functools import lru_cache as _lru_cache
from typing import Callable, List, Optional, Tuple

# GEOtop GTConst (constants.h)
PI = 3.14159265358979
OMEGA = 0.261799388        # Earth rotation velocity [rad/h]
ISC = 1367.0               # solar constant [W/m2]
MIN_TAU_CLOUD = 0.1        # floor on inferred cloud transmissivity (constants.h)
TK = 273.15
Pa0 = 1013.25               # mean atmospheric pressure at sea level [mbar]

# GCC -O3 fuses a sin(x) and a cos(x) on the same argument into one libm
# ``sincos(x)`` call, and glibc's sincos does not always return the same last
# bit as its own sin and cos do separately (measured: about 0.07% of arguments,
# 1 ULP, on each of the two results).  Every GEOtop function that takes both
# the sine and the cosine of one angle therefore reads slightly different
# values than a literal sin/cos translation would, and in the solar geometry
# that bit reaches the adaptive-Simpson averages, then the shortwave, then the
# whole column.  The six fused call sites are visible in the official binary:
# sun, SolarHeight, SolarAzimuth, Cosinc, TauatmCosinc (plus shadow_haiden,
# out of perimeter).
#
# The cache is what makes this affordable: a ctypes call costs ~3x a math.sin,
# but the angles fused here are latitude, declination, slope and aspect --
# fixed for a run or a step -- plus the solar height, which repeats across the
# Simpson refinement of one window.
_libm_name = _ctypes_util.find_library("m")
_libm = _ctypes.CDLL(_libm_name) if _libm_name is not None else None

if _libm is not None:
    _libm.sincos.restype = None
    _libm.sincos.argtypes = [
        _ctypes.c_double,
        _ctypes.POINTER(_ctypes.c_double),
        _ctypes.POINTER(_ctypes.c_double),
    ]


@_lru_cache(maxsize=8192)
def sincos(x: float) -> Tuple[float, float]:
    """Return ``(sin(x), cos(x))``, using libc ``sincos`` when available."""
    if _libm is None:
        return math.sin(x), math.cos(x)

    sn = _ctypes.c_double()
    cs = _ctypes.c_double()
    _libm.sincos(x, _ctypes.byref(sn), _ctypes.byref(cs))
    return sn.value, cs.value


# GEOtop: src/geotop/radiation.cc:560
# GEOtop: src/geotop/radiation.cc:572
# Thresholds above which x ** n exceeds the largest representable double.
# SB and dSB_dT must match GEOtop's pow(T+tk, n)
# without raising on extreme line-search trials; see SB below.
_MAX_DOUBLE = 1.7976931348623157e308
_SB_MAX = _MAX_DOUBLE ** 0.25
_DSB_MAX = _MAX_DOUBLE ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# time (times.cc)
# ---------------------------------------------------------------------------
def is_leap(year: int) -> int:
    return 1 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 0


def convert_JDandYear_JDfrom0(JD: float, year: int) -> float:
    if year == 0:
        days = 1
    else:
        days = 1 + 366
        days += (year - 1) * 365 + math.floor((year - 1) / 4.0) \
            - math.floor((year - 1) / 100.0) + math.floor((year - 1) / 400.0)
    return JD + float(days)


# GEOtop: src/geotop/times.cc:180-220
def convert_JDfrom0_JD(JDfrom0: float) -> float:
    """Fractional day-of-year from a from-year-0 Julian day."""
    if JDfrom0 < 367:
        return JDfrom0 - 1.0
    year = int(math.floor((JDfrom0 - 1.0) / 365.25))
    JD = JDfrom0 - convert_JDandYear_JDfrom0(0.0, year)
    while JD < 0 or JD >= 365 + is_leap(year):
        if JD < 0:
            year -= 1
            JD += 365 + is_leap(year)
        else:
            JD -= 365 + is_leap(year)
            year += 1
    return JD


# ---------------------------------------------------------------------------
# adaptive Simpson (tensors3D.cc)
# ---------------------------------------------------------------------------
def _simpson_aux(f: Callable[[float], float], a: float, b: float, eps: float,
                 S: float, fa: float, fb: float, fc: float, bottom: int) -> float:
    c = (a + b) / 2.0
    h = b - a
    d = (a + c) / 2.0
    e = (c + b) / 2.0
    fd = f(d)
    fe = f(e)
    Sleft = (h / 12.0) * (fa + 4 * fd + fc)
    Sright = (h / 12.0) * (fc + 4 * fe + fb)
    S2 = Sleft + Sright
    if bottom <= 0 or abs(S2 - S) <= 15 * eps:
        return S2 + (S2 - S) / 15.0
    return (_simpson_aux(f, a, c, eps / 2, Sleft, fa, fc, fd, bottom - 1)
            + _simpson_aux(f, c, b, eps / 2, Sright, fc, fb, fe, bottom - 1))


def adaptiveSimpsons2(f: Callable[[float], float], a: float, b: float,
                      eps: float, max_depth: int) -> float:
    c = (a + b) / 2.0
    h = b - a
    fa = f(a)
    fb = f(b)
    fc = f(c)
    S = (h / 6.0) * (fa + 4 * fc + fb)
    return _simpson_aux(f, a, b, eps, S, fa, fb, fc, max_depth)


# ---------------------------------------------------------------------------
# solar geometry (radiation.cc)
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/radiation.cc:43-59
def sun(JDfrom0: float) -> Tuple[float, float, float]:
    """Return ``(E0, Et, Delta)`` -- sun-earth distance factor, equation of
    time [rad], declination [rad]."""
    Gamma = 2.0 * PI * convert_JDfrom0_JD(JDfrom0) / 365.25
    sin1, cos1 = sincos(Gamma)
    sin2, cos2 = sincos(2 * Gamma)
    sin3, cos3 = sincos(3 * Gamma)
    E0 = (1.00011 + 0.034221 * cos1 + 0.00128 * sin1
          + 0.000719 * cos2 + 0.000077 * sin2)
    Et = (0.000075 + 0.001868 * cos1 - 0.032077 * sin1
          - 0.014615 * cos2 - 0.04089 * sin2)
    Delta = (0.006918 - 0.399912 * cos1 + 0.070257 * sin1
             - 0.006758 * cos2 + 0.000907 * sin2
             - 0.002697 * cos3 + 0.00148 * sin3)
    return E0, Et, Delta


def SolarHeight(JD: float, latitude: float, Delta: float, dh: float) -> float:
    h = (JD - math.floor(JD)) * 24.0 + dh
    if h >= 24:
        h -= 24.0
    if h < 0:
        h += 24.0
    sin_lat, cos_lat = sincos(latitude)
    sin_del, cos_del = sincos(Delta)
    sine = (sin_lat * sin_del
            + cos_lat * cos_del * math.cos(OMEGA * (12 - h)))
    sine = min(1.0, max(-1.0, sine))
    return max(math.asin(sine), 0.0)


def SolarAzimuth(JD: float, latitude: float, Delta: float, dh: float) -> float:
    h = (JD - math.floor(JD)) * 24.0 + dh
    if h >= 24:
        h -= 24.0
    if h < 0:
        h += 24.0
    sin_lat, cos_lat = sincos(latitude)
    sin_del, cos_del = sincos(Delta)
    sine = (sin_lat * sin_del
            + cos_lat * cos_del * math.cos(OMEGA * (12 - h)))
    sine = min(1.0, max(-1.0, sine))
    alpha = math.asin(sine)
    sin_a, cos_a = sincos(alpha)
    if h <= 12:
        if alpha == PI / 2.0:
            direction = PI / 2.0
        else:
            cosine = ((sin_a * sin_lat - sin_del) / (cos_a * cos_lat))
            cosine = min(1.0, max(-1.0, cosine))
            direction = PI - math.acos(cosine)
    else:
        if alpha == PI / 2.0:
            direction = 3 * PI / 2.0
        else:
            cosine = ((sin_a * sin_lat - sin_del) / (cos_a * cos_lat))
            cosine = min(1.0, max(-1.0, cosine))
            direction = PI + math.acos(cosine)
    return direction


def atm_transmittance(X: float, P: float, RH: float, T: float,
                      Lozone: float = 0.0, a: float = 0.0, b: float = 0.0,
                      rho_g: float = 0.0) -> float:
    """Global clear-sky atmospheric transmittance, direct plus reflected
    (Iqbal, "An Introduction to Solar Radiation", section 7.5).

    ``X`` is the sun's elevation above the horizon [rad], ``P`` pressure
    [mbar], ``RH`` relative humidity [0-1], ``T`` air temperature [degC],
    ``Lozone`` total ozone [cm], ``a``/``b`` the Angstrom turbidity
    coefficients, ``rho_g`` ground albedo (feeds back into the
    multiply-reflected term between the ground and the aerosol/Rayleigh
    layer above it).
    """
    w0 = 0.9
    Fc = 0.84

    mr = 1.0 / (math.sin(X) + 0.15 * (3.885 + X * 180.0 / PI) ** -1.253)
    ma = mr * P / Pa0
    w = 0.493 * RH * math.exp(26.23 - 5416.0 / (T + TK)) / (T + TK)  # [cm]

    U1 = w * mr
    U3 = Lozone * mr

    tau_r = math.exp(-0.0903 * ma ** 0.84 * (1.0 + ma - ma ** 1.01))
    tau_o = 1.0 - (0.1611 * U3 * (1.0 + 139.48 * U3) ** -0.3035
                   - 0.002715 * U3 / (1.0 + 0.044 * U3 + 0.0003 * U3 * U3))
    tau_g = math.exp(-0.0127 * ma ** 0.26)
    tau_w = 1.0 - 2.4959 * U1 / ((1.0 + 79.034 * U1) ** 0.6828 + 6.385 * U1)
    tau_a = 0.12445 * a - 0.0162 + (1.003 - 0.125 * a) * math.exp(-b * ma * (1.089 * a + 0.5123))
    tau_aa = 1.0 - (1.0 - w0) * (1.0 - ma + ma ** 1.06) * (1.0 - tau_a)

    rho_a = 0.0685 + (1.0 - Fc) * (1.0 - tau_a / tau_aa)

    tau_atm_n = 0.9751 * tau_r * tau_o * tau_g * tau_w * tau_a
    tau_atm_dr = (0.79 * tau_o * tau_g * tau_w * tau_aa * 0.5 * (1.0 - tau_r)
                  / (1.0 - ma + ma ** 1.02))
    tau_atm_da = (0.79 * tau_o * tau_g * tau_w * tau_aa * Fc * (1.0 - tau_a / tau_aa)
                  / (1.0 - ma + ma ** 1.02))
    tau_atm_dm = (tau_atm_n + tau_atm_dr + tau_atm_da) * rho_g * rho_a / (1.0 - rho_g * rho_a)

    return tau_atm_n + tau_atm_dr + tau_atm_da + tau_atm_dm


# GEOtop: src/geotop/radiation.cc:382-404
def diff2glob(a: float) -> float:
    """Diffuse-to-global ratio, Erbs et al. (1982)."""
    if a < 0.22:
        return 1.0 - 0.09 * a
    if a < 0.80:
        return (0.9511 - 0.1604 * a + 4.388 * (a * a)
                - 16.638 * a ** 3 + 12.336 * a ** 4)
    return 0.165


# GEOtop: src/geotop/meteo.cc:116-120
def pressure_from_elevation(elevation_m: float) -> float:
    """GEOtop standard pressure [mbar] at elevation [m] (``pressure``)."""
    return Pa0 * math.exp(-elevation_m / 8500.0)


# --- integrand helpers (others = sun[0..11]) --------------------------------
def _SolarHeight_o(JD: float, o: List[float]) -> float:
    return SolarHeight(JD, o[0], o[1], o[2])


def _Sinalpha(JD: float, o: List[float]) -> float:
    a = _SolarHeight_o(JD, o)
    return max(math.sin(a), 0.05) if a > 0 else 0.0


def _Cosinc(JD: float, o: List[float]) -> float:
    slope, aspect = o[6], o[7]
    a = _SolarHeight_o(JD, o)
    if a > 0:
        direction = SolarAzimuth(JD, o[0], o[1], o[2])
        sin_s, cos_s = sincos(slope)
        sin_a, cos_a = sincos(a)
        return max(0.0, cos_s * sin_a
                   + sin_s * cos_a * math.cos(-aspect + direction))
    return 0.0


def _Tauatm(JD: float, o: List[float]) -> float:
    height = _SolarHeight_o(JD, o)
    if height < math.asin(0.05):
        height = math.asin(0.05)
    return atm_transmittance(height, o[5], o[3], o[4], o[8], o[9], o[10], o[11])


def _TauatmSinalpha(JD: float, o: List[float]) -> float:
    RH, T, P = o[3], o[4], o[5]
    height = _SolarHeight_o(JD, o)
    if height > 0:
        return (atm_transmittance(max(height, math.asin(0.05)), P, RH, T,
                                  o[8], o[9], o[10], o[11])
                * max(math.sin(height), 0.05))
    return 0.0


def _TauatmCosinc(JD: float, o: List[float]) -> float:
    RH, T, P = o[3], o[4], o[5]
    slope, aspect = o[6], o[7]
    height = _SolarHeight_o(JD, o)
    if height > 0:
        direction = SolarAzimuth(JD, o[0], o[1], o[2])
        sin_s, cos_s = sincos(slope)
        sin_h, cos_h = sincos(height)
        return (atm_transmittance(max(height, math.asin(0.05)), P, RH, T,
                                  o[8], o[9], o[10], o[11])
                * max(0.0, cos_s * sin_h
                      + sin_s * cos_h * math.cos(-aspect + direction)))
    return 0.0


def _SolarAzimuth_o(JD: float, o: List[float]) -> float:
    return SolarAzimuth(JD, o[0], o[1], o[2])


def _avg(f, o, a, b, eps, depth=20):
    return adaptiveSimpsons2(lambda JD: f(JD, o), a, b, eps, depth) / (b - a)


# ---------------------------------------------------------------------------
# cloud transmissivity from measured shortwave (radiation.cc)
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/radiation.cc:620-701
def cloud_transmittance(JDbeg: float, JDend: float, others: List[float],
                        E0: float, ISWR: Optional[float] = None,
                        sky: float = 1.0, SWrefl_surr: float = 0.0,
                        SWdiffuse: Optional[float] = None,
                        SWdirect: Optional[float] = None) -> Optional[float]:
    """Infer cloud transmissivity from station shortwave measurements.

    ``ISWR`` is global incoming shortwave from a horizontal pyranometer,
    averaged over the same ``[JDbeg, JDend]`` interval used here.  If both
    ``SWdirect`` and ``SWdiffuse`` are supplied, GEOtop gives that pair
    precedence over ``ISWR``.

    ``others`` must describe the *station*: its latitude/longitude-derived
    solar geometry, RH [0--1], T [C], and pressure [mbar].  Use
    :func:`tau_cloud_from_iswr` when station elevation is available instead of
    pressure.

    ``None`` corresponds to GEOtop's ``gDoubleNoValue``: no stable inversion
    is possible at night/twilight (average atmospheric-transmittance-times-
    sine-of-elevation non-positive over the window) or when neither
    measurement pair is usable. A non-``None`` result is otherwise always in
    ``[min_tau_cloud, 1]``: the SW-alone branch clips its own iterative
    estimate to ``[0, 1]`` at every step (needed for the ``kd`` refinement to
    converge stably, not just for the returned value), and every branch is
    floored at ``min_tau_cloud`` (0.1) at the end -- both unconditional, not
    an optional preprocessing choice.
    """
    if JDend <= JDbeg:
        raise ValueError("JDend must be greater than JDbeg")
    if not 0.0 < sky <= 1.0:
        raise ValueError("sky must be in (0, 1]")

    tau_atm_sin_alpha = _avg(_TauatmSinalpha, others, JDbeg, JDend, 1.0e-6)
    if not tau_atm_sin_alpha > 0.0:
        return None

    tau = None
    have_components = SWdirect is not None and SWdiffuse is not None
    if have_components:
        if SWdirect + SWdiffuse > 0.0 and SWdiffuse > 0.0:
            kd = SWdiffuse / (max(0.0, SWdirect) + SWdiffuse)
            tau = ((SWdiffuse - (1.0 - sky) * SWrefl_surr)
                   / (ISC * E0 * tau_atm_sin_alpha * sky * kd))
    elif ISWR is not None:
        kd = 0.2
        tau_atm = _avg(_Tauatm, others, JDbeg, JDend, 1.0e-6)
        for _ in range(1000):
            kd0 = kd
            tau = ((ISWR - (1.0 - sky) * SWrefl_surr)
                   / (ISC * E0 * tau_atm_sin_alpha
                      * ((1.0 - kd) + sky * kd)))
            tau = min(1.0, max(0.0, tau))
            kd = diff2glob(tau * tau_atm)
            if abs(kd0 - kd) <= 0.005:
                break

    if tau is not None and tau < MIN_TAU_CLOUD:
        tau = MIN_TAU_CLOUD
    return tau


def tau_cloud_from_iswr(JDbeg: float, JDend: float, lat_deg: float,
                        lon_deg: float, ST: float, elevation_m: float,
                        RH: float, T: float, ISWR: float,
                        sky: float = 1.0, SWrefl_surr: float = 0.0,
                        Lozone: float = 0.3, alpha: float = 1.3,
                        beta: float = 0.1) -> Optional[float]:
    """Station-level component producing ``TAU_CLD`` ready to interpolate.

    Pressure is deliberately computed from station elevation as
    ``P0*exp(-z/8500)``, matching GEOtop even when measured pressure exists.
    The result describes cloud attenuation only: do not apply elevation,
    slope, aspect, or shadow corrections before spatially interpolating it.

    ``Lozone``/``alpha``/``beta`` default to GEOtop's own keyword defaults
    (``Lozone``/``AngstromAlpha``/``AngstromBeta``), not 0 -- see the note on
    :func:`shortwave_step`.
    """
    E0, Et, Delta = sun((JDbeg + JDend) / 2.0)
    P = pressure_from_elevation(elevation_m)
    o = make_others(lat_deg, lon_deg, ST, Et, Delta, RH, T, P,
                    Lozone=Lozone, alpha=alpha, beta=beta)
    return cloud_transmittance(
        JDbeg, JDend, o, E0, ISWR=ISWR, sky=sky, SWrefl_surr=SWrefl_surr,
    )


# ---------------------------------------------------------------------------
# horizon shadowing (radiation.cc)
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/radiation.cc:757-827
# GEOtop: src/geotop/energy.balance.cc:502-505 (the call site)
def shadows_point(hor: List[Tuple[float, float]], alpha: float, azimuth: float,
                  tol_mount: float = 0.0, tol_flat: float = 0.0) -> int:
    """1 if the point is shaded by its horizon, 0 if in sun. ``alpha`` and
    ``azimuth`` in degrees; ``hor`` a list of ``(azimuth_deg, horizon_height_deg)``
    rows. The call site in ``PointEnergyBalance`` passes ``tol_mount=tol_flat=0``.
    """
    n = len(hor)
    if azimuth >= hor[n - 1][0] or azimuth < hor[0][0]:
        iend, ibeg = 0, n - 1
    else:
        iend = ibeg = -1
        for i in range(1, n):
            if hor[i - 1][0] <= azimuth < hor[i][0]:
                iend, ibeg = i, i - 1
                break
    if iend > ibeg:
        w = (azimuth - hor[ibeg][0]) / (hor[iend][0] - hor[ibeg][0])
    elif azimuth > hor[ibeg][0]:
        w = (azimuth - hor[ibeg][0]) / (hor[iend][0] + 360.0 - hor[ibeg][0])
    else:
        w = (azimuth - (hor[ibeg][0] - 360.0)) / (hor[iend][0] - (hor[ibeg][0] - 360.0))
    horiz_H = w * hor[iend][1] + (1.0 - w) * hor[ibeg][1]
    if alpha < tol_flat:
        return 1
    if alpha < horiz_H + tol_mount:
        return 1
    return 0


# ---------------------------------------------------------------------------
# shortwave (radiation.cc)
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/radiation.cc:320-375
def shortwave_radiation(JDbeg: float, JDend: float, others: List[float],
                        sin_alpha: float, E0: float, sky: float,
                        SWrefl_surr: float, tau_cloud: float,
                        shadow: int = 0) -> Tuple[float, float, float]:
    """Return ``(SWbeam, SWdiff, cos_inc)`` averaged over ``[JDbeg, JDend]``.

    Mirrors ``shortwave_radiation`` with ``tau_cloud_distr`` disabled (point
    run: no distributed cloud map), so ``tau_cloud_to_use = tau_cloud``.
    """
    tau_atm = _avg(_Tauatm, others, JDbeg, JDend, 1.0e-6)
    kd = diff2glob(tau_cloud * tau_atm)
    tau_atm_sin_alpha = _avg(_TauatmSinalpha, others, JDbeg, JDend, 1.0e-6)

    SWd = ISC * E0 * tau_cloud * tau_atm_sin_alpha * sky * kd + (1.0 - sky) * SWrefl_surr

    if shadow == 1:
        cos_inc = 0.0
        SWb = 0.0
    else:
        cos_inc = _avg(_Cosinc, others, JDbeg, JDend, 1.0e-6)
        tau_atm_cos_inc = _avg(_TauatmCosinc, others, JDbeg, JDend, 1.0e-8)
        SWb = (1.0 - kd) * ISC * E0 * tau_cloud * tau_atm_cos_inc

    cos_inc_bd = kd * sin_alpha + (1.0 - kd) * cos_inc
    return SWb, SWd, cos_inc_bd


# ---------------------------------------------------------------------------
# convenience driver for a single time step
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/energy.balance.cc:117-122
# GEOtop: src/geotop/energy.balance.cc:491-493
# GEOtop: src/geotop/energy.balance.cc:508-514
def make_others(lat_deg: float, lon_deg: float, ST: float, Et: float, Delta: float,
                RH: float, T: float, P: float,
                slope_deg: float = 0.0, aspect_deg: float = 0.0,
                Lozone: float = 0.0, alpha: float = 0.0, beta: float = 0.0,
                albedo: float = 0.0) -> List[float]:
    """Build the ``sun[0..11]`` array for one step, as ``PointEnergyBalance`` does."""
    o = [0.0] * 12
    o[0] = lat_deg * PI / 180.0
    o[1] = Delta
    o[2] = (lon_deg * PI / 180.0 - ST * PI / 12.0 + Et) / OMEGA
    o[3] = RH
    o[4] = T
    o[5] = P
    o[6] = slope_deg * PI / 180.0
    o[7] = aspect_deg * PI / 180.0
    o[8] = Lozone
    o[9] = alpha
    o[10] = beta
    o[11] = albedo
    return o


# ---------------------------------------------------------------------------
# longwave (radiation.cc, meteo.cc) -- Module R2
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/radiation.cc:554-562
def SB(T: float) -> float:
    """Stefan-Boltzmann emissive power [W/m2] for T in Celsius.

    Use ``x ** 4`` to match GEOtop's ``pow(T+tk, 4.0)``. Repeated
    multiplication can differ in the last bit, shifting layer-merging decisions
    and subsequent trajectories. Clamp extreme trial temperatures to infinity,
    matching C++ overflow behavior so the line search can reject the trial.
    """
    x = T + TK
    return 5.67e-8 * (x ** 4 if x <= _SB_MAX else math.inf)


# GEOtop: src/geotop/radiation.cc:569-574
def dSB_dT(T: float) -> float:
    """Derivative of SB [W/m2/K].

    Use x ** 3 to match pow(T+tk, 3.0), with an overflow guard.
    """
    x = T + TK
    return 4.0 * 5.67e-8 * (x ** 3 if x <= _DSB_MAX else math.inf)


# GEOtop: src/geotop/meteo.cc:181-188
def sat_vap_pressure(T: float, P: float) -> float:
    """Saturation vapour pressure [mbar], T in C, P in mbar.

    A Newton line-search trial T can be wild before being rejected. C's
    ``exp`` saturates silently to ``HUGE_VAL`` past its overflow threshold;
    Python's ``math.exp`` raises ``OverflowError`` at that same threshold
    instead, so the exponent is clamped to it explicitly here to match."""
    A = 6.1121 * (1.0007 + 3.46e-6 * P)
    x = 17.502 * T / (240.97 + T)
    return A * (math.exp(x) if x < 709.0 else math.inf)


# GEOtop: src/geotop/meteo.cc:211-214
def spec_humidity(e: float, P: float) -> float:
    """Specific humidity from vapour pressure e and pressure P."""
    return 0.622 * e / (P - 0.378 * e)


def dew_point(RH: float, T: float, P: float) -> float:
    """Dew-point temperature [C] -- inverse of :func:`sat_vap_pressure` applied
    to the actual vapour pressure ``ea = RH*es(T,P)`` (Tdew[C] output column)."""
    A = 6.1121 * (1.0007 + 3.46e-6 * P)
    ea = RH * sat_vap_pressure(T, P)
    if ea <= 0.0:
        return float("nan")
    f = math.log(ea / A)
    return 240.97 * f / (17.502 - f)


# GEOtop: src/geotop/radiation.cc:477-548
def longwave_radiation(state: int, pvap: float, RH: float, T: float,
                       k1: float, k2: float, tau_cloud: float):
    """Atmospheric emissivity. Returns ``(eps, eps_max,
    eps_min)``; ``eps_min`` is the clear-sky emissivity per the chosen
    parameterization ``state`` (GEOtop ``LWinParameterization``; default 9 =
    Dilley 1998). ``pvap`` = vapour pressure [mbar], ``T`` in C.

    ``tau_cloud`` is the cloud transmissivity actually applied (GEOtop passes
    ``tau_cloud_distr = tau_cl_av_map`` when available; here it is
    ``clamp(TAU_CLD, 0, 1)``, the same value the shortwave uses).
    """
    tk = TK
    taucloud_overcast = 0.29  # Kimball (1928)
    # ``5.95 * 0.00001`` and not the folded ``5.95e-5``: the two are different
    # doubles (5.950000000000001e-05 vs 5.95e-05), and the last bit reaches
    # the atmospheric emissivity, then LWin, then the whole column.
    c_idso = 5.95 * 0.00001
    if state == 1:
        eps_min = 1.24 * (pvap / (T + tk)) ** (1.0 / 7.0)          # Brutsaert 1975
    elif state == 2:
        eps_min = 1.08 * (1.0 - math.exp(-pvap ** ((T + tk) / 2016.0)))  # Satterlund 1979
    elif state == 3:
        eps_min = 0.7 + c_idso * pvap * math.exp(1500 / (T + tk))        # Idso 1981
    elif state == 4:
        e = 0.7 + c_idso * pvap * math.exp(1500 / (T + tk))
        eps_min = -0.792 + 3.161 * e - 1.573 * e * e                    # Idso+Hodges
    elif state == 5:
        eps_min = 0.765                                                 # Koenig-Langlo 1994
    elif state == 6:
        eps_min = 0.601 + c_idso * pvap * math.exp(1500.0 / (T + tk))   # Andreas-Ackley 1982
    elif state == 7:
        eps_min = 0.23 + k1 * ((pvap * 100.0) / (T + tk)) ** (1.0 / k2)  # Konzelmann 1994
    elif state == 8:
        eps_min = 1.0 - (1.0 + 46.5 * pvap / (T + tk)) * math.exp(
            -(1.2 + 3.0 * 46.5 * pvap / (T + tk)) ** 0.5)               # Prata 1996
    elif state == 9:
        eps_min = (59.38 + 113.7 * ((T + tk) / 273.16) ** 6
                   + 96.96 * ((465.0 * pvap / (T + tk)) / 25.0) ** 0.5) \
            / (5.67e-8 * (T + tk) ** 4)                                 # Dilley 1998
    else:
        raise ValueError(f"unknown LWinParameterization state {state}")

    eps = eps_min * tau_cloud + 1.0 * (1.0 - tau_cloud)
    eps_max = eps_min * taucloud_overcast + 1.0 * (1.0 - tau_cloud)
    return eps, eps_max, eps_min


# GEOtop: src/geotop/energy.balance.cc:601-603
def lwin_step(T: float, RH: float, P: float, tau_cloud: float, sky: float,
              state: int = 9, k1: float = 0.484, k2: float = 8.0):
    """Incoming longwave at the surface [W/m2]:
    ``LWin = sky * eps_atm * SB(T)``. Returns ``(LWin, LWin_max, LWin_min)``.

    Valid with ``Surroundings=0`` (this run); with surroundings on GEOtop adds
    ``(1-sky)*eps*SB(Tgskin_surr)``, not modelled here.
    """
    ea = RH * sat_vap_pressure(T, P)
    eps, eps_max, eps_min = longwave_radiation(state, ea, RH, T, k1, k2, tau_cloud)
    sbt = SB(T)
    return sky * eps * sbt, sky * eps_max * sbt, sky * eps_min * sbt


# GEOtop: src/geotop/energy.balance.cc:502-505 (the shadows_point gate)
# GEOtop: src/geotop/energy.balance.cc:119-121
# GEOtop: src/geotop/parameters.cc:2229-2231
def shortwave_step(JDbeg: float, JDend: float, lat_deg: float, lon_deg: float,
                   ST: float, RH: float, T: float, P: float, tau_cloud: float,
                   sky: float = 1.0, slope_deg: float = 0.0, aspect_deg: float = 0.0,
                   SWrefl_surr: float = 0.0, shadow: int = 0,
                   horizon: "List[Tuple[float, float]] | None" = None,
                   cast_shadow: bool = True,
                   Lozone: float = 0.3, alpha: float = 1.3, beta: float = 0.1,
                   albedo: float = 0.0):
    """One-step wrapper: returns ``(SWbeam, SWdiff, cos_inc, hsun, sinhsun, E0)``.

    ``E0``/``Et``/``Delta`` come from ``sun`` at the step midpoint, exactly as
    ``energy_balance`` does; ``hsun``/``sinhsun`` are the step-averaged solar
    height and its sine. When ``horizon`` is given and ``cast_shadow`` (GEOtop
    default ``CalculateCastShadow=1``), the direct beam is killed for steps whose
    step-averaged sun (``hsun``, ``dsun``) sits below the horizon -- reproducing
    the ``shadows_point`` gate of ``PointEnergyBalance``.

    ``Lozone``/``alpha``/``beta`` feed ``atm_transmittance``'s Iqbal ozone and
    aerosol-turbidity terms; the defaults are GEOtop's own
    (``Lozone``/``AngstromAlpha``/``AngstromBeta``), not zero --
    ``PointEnergyBalance`` always sets them from those
    keywords, never from a disabled default. ``albedo`` (``sun[11]``) stays 0
    unless ``AlbedoSWin`` is on, which is a real, separate keyword-gated
    feedback, not implemented here yet.
    """
    E0, Et, Delta = sun((JDbeg + JDend) / 2.0)
    o = make_others(lat_deg, lon_deg, ST, Et, Delta, RH, T, P,
                    slope_deg, aspect_deg, Lozone, alpha, beta, albedo)
    hsun = _avg(_SolarHeight_o, o, JDbeg, JDend, 1.0e-6)
    sinhsun = _avg(_Sinalpha, o, JDbeg, JDend, 1.0e-6)
    if horizon is not None and cast_shadow:
        dsun = _avg(_SolarAzimuth_o, o, JDbeg, JDend, 1.0e-6)
        if dsun < 0:
            dsun += 2 * PI
        if dsun > 2 * PI:
            dsun -= 2 * PI
        shadow = shadows_point(horizon, hsun * 180.0 / PI, dsun * 180.0 / PI)
    SWb, SWd, cos_inc = shortwave_radiation(JDbeg, JDend, o, sinhsun, E0, sky,
                                            SWrefl_surr, tau_cloud, shadow)
    return SWb, SWd, cos_inc, hsun, sinhsun, E0
