"""Canopy component -- verbatim port of ``geotop/vegetation.cc``.

The canopy is a single big leaf sitting above the ground surface with its own
prognostic temperature ``Tv`` and its own water/snow storage ``Wcrn``/``Wcsn``.
It intercepts precipitation, splits the shortwave and longwave streams between
itself and the ground, and inserts a second aerodynamic layer (canopy air) into
the turbulent exchange.

The whole component runs **once per timestep**, at the same point in the step as
``geotop_py.energy.surface.surface_forcing``: :func:`Tcanopy` solves the scalar canopy
energy balance with its own damped Newton iteration, and everything it produces
(``Tv``, the resistances ``rh``/``rv``/``rb``/``rc``/``ruc``, ``Qv``) is then
**frozen** while the column solver iterates on the ground temperature.  That is
what GEOtop does too: inside the column Newton loop only
``EnergyFluxes_no_rec_turbulence`` runs, which recomputes the canopy-air
temperature/humidity and the ground fluxes below the canopy from those frozen
resistances.

Index conventions: vegetation heights and thresholds are in **mm** in the
parameters (``VegHeight``, ``ThresSnowVegUp/Down``, ``RootDepth``) and converted
to metres only inside :func:`update_roughness_veg`.
"""
# GEOtop: src/geotop/energy.balance.cc:2675-2700 (canopy branch, no_rec_turbulence)

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from ..io.parfile import NUMBER_NOVALUE
from . import rad
from . import turbulence as tb

# constants.h
c_can = 2700.0            # heat capacity of canopy [J/(kg K)]
c_ice = 2117.0
c_liq = 4188.0
Lf = 333700.0
LSAIthres = 0.1
wsn_vis, wsn_nir = 0.8, 0.4
Bsnd_vis = Bsnd_nir = 0.5
Bsnb_vis = Bsnb_nir = 0.5
z_transp = 10000.0        # soil depth contributing to transpiration [mm]

# GEOtop: src/geotop/energy.balance.cc:61-62
RAIN_MAX_LOADING = 0.1    # ratio_max_storage_RAIN_over_canopy_to_LSAI
SNOW_MAX_LOADING = 5.0    # ratio_max_storage_SNOW_over_canopy_to_LSAI


@dataclass
class VegParams:
    """Per-land-cover-class vegetation properties (``land->ty[lu][j*]``)."""
    Hveg: float = 0.0         # VegHeight [mm]
    z0thresveg: float = 0.0   # ThresSnowVegUp [mm]
    z0thresveg2: float = 0.0  # ThresSnowVegDown [mm]
    LSAI: float = 0.0         # LSAI [m2/m2]
    cf: float = 0.0           # CanopyFraction [-]
    decay0: float = 0.0       # DecayCoeffCanopy [-]
    expveg: float = 1.0       # VegSnowBurying [-]
    root: float = 0.0         # RootDepth [mm]
    rs: float = 0.0           # MinStomatalRes [s/m]
    R_vis: float = 0.0        # VegReflectVis
    R_nir: float = 0.0        # VegReflNIR
    T_vis: float = 0.0        # VegTransVis
    T_nir: float = 0.0        # VegTransNIR
    Ch: float = 0.0           # LeafAngles
    cd: float = 0.0           # CanDensSurface [kg/(m2 LSAI)]
    # root fraction per soil layer, 1-based, length = n_transp_layers + 1
    root_frac: List[float] = field(default_factory=lambda: [0.0])
    # number of soil layers contributing to transpiration (root_frac index max)
    n_transp: int = 0


# GEOtop: src/geotop/parameters.cc:1326-1330
@dataclass
class CanopyOptions:
    """The four iteration counts and the stability switch."""
    maxiter_canopy: int = 3       # CanopyMaxIter
    maxiter_Ts: int = 2           # TsMaxIter
    maxiter_Loc: int = 3          # LocMaxIter
    maxiter_Businger: int = 5     # BusingerMaxIter
    stabcorr_incanopy: int = 1    # CanopyStabCorrection
    MO: int = 2                   # MoninObukhov
    state_turb: int = 1


# GEOtop: src/geotop/meteo.cc:216-223
def spec_humidity_2(T: float, P: float) -> Tuple[float, float]:
    """``SpecHumidity_2`` with RH = 1: saturation specific humidity and its
    temperature derivative."""
    e = rad.sat_vap_pressure(T, P)
    denom = 240.97 + T
    de_dT = e * (17.502 / denom - 17.502 * T / (denom * denom))
    Q = rad.spec_humidity(e, P)
    d = P - 0.378 * e
    dQ_de = 0.622 / d + 0.235116 * e / (d * d)
    return Q, dQ_de * de_dT


# ---------------------------------------------------------------------------
# canopy fraction and roughness
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/energy.balance.cc:336-348 (fsnow)
def snow_burying_fraction(snowD: float, vp: VegParams) -> float:
    """``fsnow``: the linear ramp between ThresSnowVegDown and ThresSnowVegUp.

    The two thresholds coincide in every class of ``gt_sim1D_veg`` (the ramp
    collapses to a step at ``VegHeight``); the general form is ported all the
    same."""
    if snowD > vp.z0thresveg:
        return 1.0
    if snowD > vp.z0thresveg2:
        return ((snowD - vp.z0thresveg2)
                / (vp.z0thresveg - vp.z0thresveg2))
    return 0.0


# GEOtop: src/geotop/energy.balance.cc:350-364
def canopy_fraction(snowD: float, vp: VegParams, ng: int = 0) -> float:
    """Active canopy fraction ``fc``. A glacier suppresses the canopy entirely,
    whatever the land-cover class says."""
    if ng > 0:
        return 0.0
    if vp.LSAI < LSAIthres:
        return 0.0
    return vp.cf * (1.0 - snow_burying_fraction(snowD, vp)) ** vp.expveg


# GEOtop: src/geotop/vegetation.cc:605-652
def update_roughness_veg(hc: float, snowD: float, zmu: float,
                         zmt: float) -> Tuple[float, float, float]:
    """Return ``(z0veg, d0veg, hveg)`` in metres from the canopy height above
    the snow. ``hc`` and ``snowD`` in mm."""
    h = hc - snowD
    d0 = 0.667 * h * 1e-3
    z0 = 0.1 * h * 1e-3
    h *= 1e-3
    if zmu < h:
        raise ValueError("wind speed must be measured above the vegetation")
    if zmt < h:
        raise ValueError("temperature must be measured above the vegetation")
    if h - d0 < z0:
        raise ValueError("it must always be Hcanopy-d0 >= z0")
    return z0, d0, h


# ---------------------------------------------------------------------------
# interception
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/vegetation.cc:540-562
def canopy_rain_interception(rain_max_loading: float, LSAI: float,
                             Prain: float,
                             storage: float) -> Tuple[float, float, float]:
    """Return ``(max_storage, storage, drip)`` [mm]."""
    max_storage = rain_max_loading * LSAI
    load = 0.25 * Prain * (1.0 - math.exp(-0.5 * LSAI))
    storage += load
    drip = Prain - load
    if storage > max_storage:
        drip += storage - max_storage
        storage = max_storage
    return max_storage, storage, drip


# GEOtop: src/geotop/vegetation.cc:569-597
def canopy_snow_interception(snow_max_loading: float, LSAI: float, Psnow: float,
                             Tc: float, v: float, Dt: float,
                             storage: float) -> Tuple[float, float, float]:
    """Return ``(max_storage, storage, drip)`` [mm] (Niu & Yang 2004 loading,
    temperature/wind unloading)."""
    CT, CV = 1.87e5, 1.56e5
    max_storage = snow_max_loading * LSAI
    load = (max_storage - storage) * (1.0 - math.exp(-Psnow / max_storage))
    storage += load
    drip = Psnow - load
    if storage > max_storage:
        drip -= max_storage - storage
        storage = max_storage
    unload = min(storage, storage * (max(0.0, Tc + 3.0) / CT + v / CV) * Dt)
    if unload < 0.1:
        unload = 0.0
    drip += unload
    storage -= unload
    return max_storage, storage, drip


# ---------------------------------------------------------------------------
# radiative partition
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/vegetation.cc:428-509
def shortwave_vegetation(Sd: float, Sb: float, x: float, fwsn: float,
                         wsn: float, Bsnd: float, Bsnb: float, Agd: float,
                         Agb: float, C: float, R: float, T: float,
                         L: float) -> Tuple[float, float, float]:
    """Two-stream canopy shortwave (Dickinson): ``(Sv, Sg, Sup_above)``."""
    phi1 = 0.5 - 0.633 * C - 0.33 * C * C
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    G = phi1 + phi2 * x

    if C == 0:
        xm = 1.0
    else:
        xm = (1.0 - (phi1 / phi2) * math.log((phi1 + phi2) / phi1)) / phi2

    K = G / x

    w = (R + T) * (1.0 - fwsn) + wsn * fwsn
    gc = 0.5 * (1.0 + C)
    wBd = (0.5 * (R + T + gc * gc * (R - T))) * (1.0 - fwsn) \
        + wsn * Bsnd * fwsn
    wBb = (((1.0 + xm * K) / (xm * K)) * 0.5 * w
           * ((G / (x * phi2 + G))
              * (1.0 - (x * phi1 / (x * phi2 + G))
                 * math.log((x * phi1 + x * phi2 + G) / (x * phi1))))) \
        * (1.0 - fwsn) + wsn * Bsnb * fwsn

    b = 1.0 - w + wBd
    c = wBd
    d = xm * K * wBb
    f = w * xm * K * (1.0 - wBb / w)
    h = (b * b - c * c) ** 0.5 / xm
    xk = xm * K
    s = xk * xk + c * c - b * b
    u1 = b - c / Agd
    u2 = b - c * Agd
    u3 = f + c * Agd
    s1 = math.exp(-h * L)
    s2 = math.exp(-K * L)
    p1 = b + xm * h
    p2 = b - xm * h
    p3 = b + xm * K
    p4 = b - xm * K
    d1 = p1 * (u1 - xm * h) / s1 - p2 * (u1 + xm * h) * s1
    d2 = (u2 + xm * h) / s1 - (u2 - xm * h) * s1
    h1 = -d * p4 - c * f
    h2 = ((d - h1 * p3 / s) * (u1 - xm * h) / s1
          - p2 * (d - c - h1 * (u1 + xm * K) / s) * s2) / d1
    h3 = -((d - h1 * p3 / s) * (u1 + xm * h) * s1
           - p1 * (d - c - h1 * (u1 + xm * K) / s) * s2) / d1
    h4 = -f * p3 - c * d
    h5 = -(h4 * (u2 + xm * h) / (s * s1)
           + (u3 - h4 * (u2 - xm * K) / s) * s2) / d2
    h6 = (h4 * (u2 - xm * h) * s1 / s
          + (u3 - h4 * (u2 - xm * K) / s) * s2) / d2
    h7 = c * (u1 - xm * h) / (d1 * s1)
    h8 = -c * (u1 + xm * h) * s1 / d1
    h9 = (u2 + xm * h) / (d2 * s1)
    h10 = -s1 * (u2 - xm * h) / d2

    Iupb = h1 / s + h2 + h3
    Iupd = h7 + h8
    Idwb = h4 * math.exp(-K * L) / s + h5 * s1 + h6 / s1
    Idwd = h9 * s1 + h10 / s1

    Ib = (1.0 - Iupb) - (1.0 - Agd) * Idwb - (1.0 - Agb) * math.exp(-K * L)
    Id = (1.0 - Iupd) - (1.0 - Agd) * Idwd

    if x > 0:
        Sv = Sb * Ib + Sd * Id
        Sg = Sb * (1.0 - Agb) * math.exp(-K * L) + (Sb * Idwb + Sd * Idwd) * (1.0 - Agd)
        Sup = Sb * Iupb + Sd * Iupd
    else:
        Sv = Sd * Id
        Sg = Sd * Idwd * (1.0 - Agd)
        Sup = Sd * Iupd
    return Sv, Sg, Sup


# GEOtop: src/geotop/vegetation.cc:516-533
def longwave_vegetation(Lin: float, eg: float, Tg: float, Tv: float,
                        L: float) -> Tuple[float, float, float, float]:
    """Return ``(Lv, Lg, dLv_dTv, Lup_above)``."""
    ev = 1.0 - math.exp(-L)
    Lvdw = (1.0 - ev) * Lin + ev * rad.SB(Tv)
    Lv = (ev * (1.0 + (1.0 - eg) * (1.0 - ev)) * Lin + ev * eg * rad.SB(Tg)
          - (2.0 - ev * (1.0 - eg)) * ev * rad.SB(Tv))
    Lg = eg * Lvdw - eg * rad.SB(Tg)
    dLv_dTv = -(2.0 - ev * (1.0 - eg)) * ev * rad.dSB_dT(Tv)
    Lup = ((1.0 - ev) * (1.0 - eg) * (1.0 - ev) * Lin
           + (1.0 - ev) * eg * rad.SB(Tg)
           + (1.0 + (1.0 - ev) * (1.0 - eg)) * ev * rad.SB(Tv))
    return Lv, Lg, dLv_dTv, Lup


# ---------------------------------------------------------------------------
# roots and transpiration
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/input.cc:707-715
def n_transp_layers(Dz: Sequence[float], nsoil: int,
                    depth: float = z_transp) -> int:
    """Number of soil layers reached by ``depth`` [mm], GEOtop's allocation
    loop. ``Dz`` is 1-based in mm."""
    z = 0.0
    l = 0
    while True:
        l += 1
        z += Dz[l]
        if not (l < nsoil and z < depth):
            break
    return l


# GEOtop: src/geotop/vegetation.cc:659-693 (root)
def root_fraction(n: int, d: float, slope: float,
                  Dz: Sequence[float]) -> List[float]:
    """Per-layer transpiration weights. ``n`` is
    ``root_fraction.getCols()`` = number of layers + 1, so the loop runs over
    ``1..n-1``. ``d`` = root depth [mm], ``Dz`` 1-based in mm."""
    frac = [0.0] * n
    d_corr = d * math.cos(slope)
    z = 0.0
    for l in range(1, n):
        z += Dz[l]
        if d_corr > z:
            frac[l] = Dz[l] / d_corr
        elif d_corr > z - Dz[l]:
            frac[l] = (d_corr - (z - Dz[l])) / d_corr
        else:
            frac[l] = 0.0
    return frac


#: Vegetation properties that a time series may supply, in the order the
#: series carries them. Position 0 of a series row is its timestamp, so entry
#: ``k`` of this tuple is column ``k + 1``.
TIME_DEPENDENT_FIELDS = ("Hveg", "z0thresveg", "z0thresveg2", "LSAI", "cf",
                         "decay0", "expveg", "root", "rs")


# GEOtop: src/geotop/energy.balance.cc:322-334
def apply_time_dependent(vp: VegParams, static: VegParams,
                         values: Sequence[float], Dz: Sequence[float]) -> None:
    """Overwrite ``vp``'s nine time-varying properties in place for one step.

    ``values`` is one interpolated series row, indexed as the file is: slot 0
    is the timestamp and slots ``1..9`` follow :data:`TIME_DEPENDENT_FIELDS`.
    A slot holding ``NUMBER_NOVALUE`` -- what the interpolator returns for a
    step the series does not cover -- falls back to ``static``, the value the
    per-class keywords gave. ``Dz`` is the 1-based soil layer thickness [mm].

    Two asymmetries are deliberate. The root fraction is recomputed only when
    the series actually supplies a root depth, so a step that falls back to
    the static root depth keeps whatever root fraction was last computed
    rather than one consistent with the depth now in use. And ``vp`` is
    persistent state, not a per-step copy: a step that does not converge and
    is retried at a shorter interval leaves its overwrite behind.
    """
    for k, name in enumerate(TIME_DEPENDENT_FIELDS):
        value = values[k + 1]
        if int(value) != int(NUMBER_NOVALUE):
            setattr(vp, name, value)
            if name == "root":
                vp.root_frac = root_fraction(vp.n_transp + 1, vp.root, 0.0, Dz)
        else:
            setattr(vp, name, getattr(static, name))


# GEOtop: src/geotop/vegetation.cc:700-771
def canopy_evapotranspiration(rbv: float, Tv: float, Qa: float, Pa: float,
                              SWin: float, theta: Sequence[float],
                              vp: VegParams,
                              soil: Sequence) -> Tuple[float, List[float]]:
    """Return ``(ft, fl)``: the column transpiration fraction and the
    normalised per-layer weights."""
    fS = SWin / (SWin + 250.0) * 1.25
    ea = Qa * Pa / (0.378 * Qa + 0.622)
    ev = rad.sat_vap_pressure(Tv, Pa)
    fe = 1.0 - (ev - ea) / 40.0

    if Tv <= 0:
        fTemp = 1e-12
    elif Tv >= 50.0:
        fTemp = 1e-12
    else:
        fTemp = (Tv - 0.0) * (50.0 - Tv) / 625.0

    fl = [0.0] * (vp.n_transp + 1)
    f = 0.0
    for l in range(1, vp.n_transp + 1):
        s = soil[l]
        if theta[l] >= s.fc:
            v = 1.0
        elif theta[l] > s.wp:
            v = (theta[l] - s.wp) / (s.fc - s.wp)
        else:
            v = 0.0

        if fS * fe * fTemp * v < 6.0e-11:
            v = 1.0e12
        else:
            v = vp.rs / (fS * fe * fTemp * v)

        fl[l] = vp.root_frac[l] * (rbv / (rbv + v))
        f += fl[l]

    if f != 0:
        for l in range(1, vp.n_transp + 1):
            fl[l] /= f
    return f, fl


# ---------------------------------------------------------------------------
# in-canopy turbulence
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/vegetation.cc:778-848
def veg_transmittance(stabcorr_incanopy: int, v: float, u_star: float,
                      u_top: float, Hveg: float, z0soil: float, z0veg: float,
                      d0veg: float, LSAI: float, decaycoeff0: float,
                      Lo: float, Loc: float) -> Tuple[float, float, float]:
    """Return ``(rb, ruc, decay)``: leaf boundary-layer resistance, undercanopy
    ground resistance and the wind-decay coefficient."""
    Lc = 0.4
    zm = d0veg + z0veg

    if Lo < 0:
        phi_above = (1.0 - 16.0 * Hveg / Lo) ** -0.5
    else:
        phi_above = 1.0 + 5.0 * Hveg / Lo
    if Loc < 0:
        phi_below = (1.0 - 15.0 * zm / Loc) ** -0.25
    else:
        phi_below = 1.0 + 4.7 * zm / Loc

    if stabcorr_incanopy == 1:
        decay = min(1.0e5, decaycoeff0 * math.sqrt(phi_below))
    else:
        decay = decaycoeff0

    u_veg = max(0.001, u_top * math.exp(decay * (zm / Hveg - 1.0)))
    rb = min(1.0e20, 70.0 * math.sqrt(Lc / u_veg)) / LSAI

    r = ((Hveg * math.exp(decay) / (d0veg * decay))
         * (math.exp(-decay * z0soil / Hveg) - math.exp(-decay * zm / Hveg)))
    Ktop = tb.KA * u_star * (Hveg - d0veg) / phi_above
    # GEOtop: src/geotop/vegetation.cc:845-846 (std::min<double>(1.E20, ...))
    # ``r`` goes negative once the canopy is buried to within the ground
    # roughness (zm < z0soil), which happens for a step or two at the melt-out
    # threshold.  C's pow() then returns NaN and std::min keeps its FIRST
    # argument when the comparison is false, so ruc becomes 1e20 and the
    # ground decouples from the canopy air.  Python's ``**`` would
    # return a complex number instead, so the NaN has to be made explicit.
    rpow = math.nan if r < 0.0 else r ** 0.45
    ruc = min(1.0e20, r * d0veg / Ktop + 2.0 * rpow / (tb.KA * u_star))
    return rb, ruc, decay


# ---------------------------------------------------------------------------
# canopy energy balance
# ---------------------------------------------------------------------------
@dataclass
class CanopyState:
    """Everything ``Tcanopy`` leaves behind for the frozen-resistance flux."""
    Tv: float = 0.0
    Ts: float = 0.0
    Qs: float = 0.0
    Qv: float = 0.0
    rh: float = 1e99
    rv: float = 1e99
    rb: float = 1e99
    rc: float = 1e99
    ruc: float = 1e99
    rm: float = 1e20
    Lobukhov: float = 1e5
    Locc: float = 1e50
    u_top: float = 0.0
    decay: float = 0.0
    LWv: float = 0.0
    Hv: float = 0.0
    LEv: float = 0.0
    Etrans: float = 0.0
    dWcrn: float = 0.0
    dWcsn: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    transp_layer: List[float] = field(default_factory=list)
    evap_layer: List[float] = field(default_factory=list)


@dataclass
class CanopyInputs:
    """What the caller knows about the canopy before the fluxes are built."""
    fc: float
    vp: VegParams
    z0veg: float
    d0veg: float
    hveg: float
    ng: int
    Wcrn: float
    Wcsn: float
    Wcrnmax: float
    Wcsnmax: float
    Tv0: float


@dataclass
class CanopyForcing:
    """The inputs ``canopy_fluxes`` needs and that do not change within a step."""
    Tg: float
    Ta: float
    Qgsat: float
    Qa: float
    zmu: float
    zmt: float
    z0v: float
    z0s: float
    d0v: float
    rz0v: float
    hveg: float
    v: float
    P: float
    SWin: float
    LW: float
    eps: float
    Dt: float
    nsng: int
    theta: List[float]
    soil: List
    Dsoil: List[float]
    Tsoil: List[float]
    psi: float
    n_evap: int


# GEOtop: src/geotop/vegetation.cc:245-421
def _canopy_fluxes(Tv: float, Wcrn: float, Wcrnmax: float, Wcsn: float,
                   Wcsnmax: float, chgsgn: int, st: CanopyState,
                   f: CanopyForcing, vp: VegParams, opt: CanopyOptions):
    """``canopy_fluxes``. Mutates ``st`` in place and
    returns ``(h, dhdT, Esubl)``."""
    max_chgsgn = 10
    MO = opt.MO

    fwliq = (Wcrn / Wcrnmax) ** (2.0 / 3.0)
    fwice = (Wcsn / Wcsnmax) ** (2.0 / 3.0)
    fw = max(fwliq, fwice)
    fw = min(1.0, max(0.0, fw))

    st.LWv, LWg, dLWvdT, LWup = longwave_vegetation(
        f.LW, f.eps, f.Tg, Tv, vp.LSAI)

    st.Qv, dQvdT = spec_humidity_2(Tv, f.P)

    Loc = 1e50
    R = 1.0
    cont2 = 0
    while True:
        cont2 += 1
        Ts0 = st.Ts

        if chgsgn > max_chgsgn:
            MO = 4

        st.rm, st.rh, st.rv, st.Lobukhov = tb.aero_resistance(
            f.zmu, f.zmt, f.z0v, f.d0v, f.rz0v, f.v, f.Ta, Ts0, f.Qa, st.Qs,
            f.P, MO=MO, maxiter=opt.maxiter_Businger)

        u_star = math.sqrt(f.v / st.rm)
        st.u_top = (u_star / tb.KA) * tb.CZ(MO, f.hveg, f.z0v, f.d0v,
                                            st.Lobukhov, tb.Psim)

        cont = 0
        while True:
            cont += 1
            Loc0 = Loc

            st.rb, st.ruc, st.decay = veg_transmittance(
                opt.stabcorr_incanopy, f.v, u_star, st.u_top, f.hveg, f.z0s,
                f.z0v, f.d0v, vp.LSAI, vp.decay0, st.Lobukhov, Loc)

            st.alpha, st.beta, st.evap_layer = _evap_parameters(f, st.ruc)

            if st.Qv < st.Qs:                       # condensation
                R = 1.0
            else:
                ft, fl = canopy_evapotranspiration(
                    st.rb, Tv, f.Qa, f.P, f.SWin, f.theta, vp, f.soil)
                st.transp_layer = fl
                R = fw + (1.0 - fw) * ft
            st.rc = st.rb / R

            st.Ts = ((f.Ta / st.rh + f.Tg / st.ruc + Tv / st.rb)
                     / (1.0 / st.rh + 1.0 / st.ruc + 1.0 / st.rb))
            st.Qs = ((f.Qa / st.rv + st.alpha * f.Qgsat * st.beta / st.ruc
                      + st.Qv / st.rc)
                     / (1.0 / st.rv + st.beta / st.ruc + 1.0 / st.rc))

            Hg = (tb.air_cp(0.5 * (st.Ts + f.Tg))
                  * tb.air_density(0.5 * (st.Ts + f.Tg),
                                   0.5 * (st.Qs + st.alpha * f.Qgsat), f.P)
                  * (f.Tg - st.Ts) / st.ruc)

            Loc = -u_star ** 3.0 / (
                tb.KA * (tb.GRAVITY / (st.Ts + tb.TK))
                * (Hg / (tb.air_density(st.Ts, st.Qs, f.P) * tb.air_cp(st.Ts))))
            # ``Loc`` has a pole wherever ``Hg`` changes sign, and GEOtop
            # guards only the exact zero -- literally ``Hg == 0.0``, not a
            # tolerance -- so the value it reports on the steps where the
            # undercanopy stratification is neutral is an amplified difference,
            # not a physical length.  The comparison stays exact on purpose:
            # widening it into a tolerance would cap steps GEOtop leaves
            # uncapped, matching GEOtop's canopy stability calculation.
            if Hg == 0.0 or Hg != Hg:
                Loc = 1e50

            if not (abs(Loc0 - Loc) > 0.01 and cont <= opt.maxiter_Loc):
                break

        if not (cont2 < opt.maxiter_Ts and abs(st.Ts - Ts0) > 0.01):
            break

    st.Hv, dHdT, E, dEdT = tb.turbulent_fluxes(
        st.rb, st.rc, f.P, st.Ts, Tv, st.Qs, st.Qv, dQvdT)

    Lt = tb.Levap(Tv)
    if E > 0:                                       # evaporation/sublimation
        Esubl = E * fw / R
        dEsubldT = dEdT * fw / R
        if fwliq + fwice > 0:
            Lv = Lt + Lf * fwice / (fwliq + fwice)
        else:
            Lv = Lt
    else:                                           # condensation
        Esubl = E
        dEsubldT = dEdT
        Lv = Lt if Tv >= 0 else Lt + Lf

    st.Etrans = E - Esubl
    st.transp_layer = [x * st.Etrans for x in st.transp_layer]

    st.LEv = Lt * st.Etrans + Lv * Esubl
    h = st.LWv - st.Hv - st.LEv
    dhdT = dLWvdT - dHdT - Lt * (dEdT - dEsubldT) - Lv * dEsubldT
    st.Locc = Loc
    return h, dhdT, Esubl


# GEOtop: src/geotop/turbulence.cc:561-568 (snow/ice short-circuit)
def _evap_parameters(f: CanopyForcing, rv: float):
    """``find_actual_evaporation_parameters`` as ``canopy_fluxes`` calls it: the
    snow/ice short-circuit first, otherwise Ye--Pielke."""
    if f.nsng > 0:
        return 1.0, 1.0, [0.0] * (f.n_evap + 1)
    return tb.soil_evaporation_parameters(
        f.soil, f.Dsoil, f.Tsoil, f.theta, f.P, rv, f.Ta, f.Qa, f.Qgsat,
        f.psi, nlayers=f.n_evap)


# GEOtop: src/geotop/vegetation.cc:43-237
def Tcanopy(Tv0: float, Wcrn0: float, Wcsn0: float, Wcrnmax: float,
            Wcsnmax: float, SWv: float, f: CanopyForcing, vp: VegParams,
            opt: CanopyOptions) -> CanopyState:
    """Solve the canopy energy balance for ``Tv``.

    A damped Newton iteration on the single unknown ``Tv``: each trial value is
    reconciled with the phase change of the intercepted water before the fluxes
    are re-evaluated, and the step is halved (by thirds) until the residual
    stops growing.  ``A = 1`` in GEOtop, so the ``t0`` evaluation of the
    trapezoidal form is dead code and is not ported."""
    C0 = vp.cd * vp.LSAI * c_can + c_ice * Wcsn0 + c_liq * Wcrn0
    C = C0
    Wcrn, Wcsn = Wcrn0, Wcsn0

    fwliq = (Wcrn / Wcrnmax) ** (2.0 / 3.0)
    fwice = (Wcsn / Wcsnmax) ** (2.0 / 3.0)

    T00 = Tv0
    T11 = T11p = Tv0
    chgsgn = 0

    st = CanopyState(Tv=Tv0)
    st.Ts = 0.5 * f.Tg + 0.5 * f.Ta
    st.Qs = 0.5 * f.Qgsat + 0.5 * f.Qa
    h1, dhdT, subl_can = _canopy_fluxes(T11, Wcrn0, Wcrnmax, Wcsn0, Wcsnmax,
                                        chgsgn, st, f, vp, opt)

    cont = 0
    converged = False
    while not converged and cont < opt.maxiter_canopy:
        T10 = T11
        cont2 = 0
        nw = 1.0
        err0 = abs(C * (T11 - T00) / f.Dt - SWv - h1)
        DT = ((-C * (T10 - T00) / f.Dt + SWv + h1) / (C / f.Dt - dhdT))
        Lobukhov0 = st.Lobukhov

        while True:
            T11 = T10 + nw * DT

            if subl_can < 0 and T11 < 0:            # condensation as frost
                Wcsn = Wcsn0 - subl_can * f.Dt
                Wcrn = Wcrn0
            elif subl_can < 0 and T11 >= 0:         # condensation as dew
                Wcrn = Wcrn0 - subl_can * f.Dt
                Wcsn = Wcsn0
            else:                                   # part evap, part subl
                if fwliq + fwice > 0:
                    Wcsn = Wcsn0 - (fwice / (fwliq + fwice)) * subl_can * f.Dt
                    Wcrn = Wcrn0 - (fwliq / (fwliq + fwice)) * subl_can * f.Dt
                else:
                    Wcsn, Wcrn = Wcsn0, Wcrn0

            Wcrn = min(Wcrnmax, max(0.0, Wcrn))
            Wcsn = min(Wcsnmax, max(0.0, Wcsn))

            if T11 > 0 and Wcsn > 0:                # melting
                melt_can = min(Wcsn, c_ice * Wcsn * (T11 - 0.0) / Lf)
                T11p = T11 - Lf * melt_can / C
                Wcsn -= melt_can
                Wcrn += melt_can
            elif T11 < 0 and Wcrn > 0:              # freezing
                melt_can = -min(Wcrn, c_liq * Wcrn * (0.0 - T11) / Lf)
                T11p = T11 - Lf * melt_can / C
                Wcsn -= melt_can
                Wcrn += melt_can
            else:
                T11p = T11

            C = vp.cd * vp.LSAI * c_can + c_ice * Wcsn + c_liq * Wcrn
            C = (C + C0) / 2.0

            h1, dhdT, subl_can = _canopy_fluxes(
                T11p, Wcrn, Wcrnmax, Wcsn, Wcsnmax, chgsgn, st, f, vp, opt)

            err1 = abs(C * (T11 - T00) / f.Dt - SWv - h1)
            nw /= 3.0
            cont2 += 1
            if not (err1 > err0 and cont2 < 5):
                break

        if Lobukhov0 * st.Lobukhov < 0:
            chgsgn += 1
        cont += 1
        if abs(T11 - T10) < 0.01 and err1 < 0.1:
            converged = True

    st.Tv = T11p
    st.dWcrn = Wcrn - Wcrn0
    st.dWcsn = Wcsn - Wcsn0
    return st
