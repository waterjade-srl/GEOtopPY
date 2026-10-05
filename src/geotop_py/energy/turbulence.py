"""Surface-layer turbulence (Monin-Obukhov) -- sensible/latent heat fluxes.

Ported verbatim from GEOtop ``turbulence.cc`` (Endrizzi's ``Turbulence::``):
the Businger iterative Monin-Obukhov solver (``Businger`` + ``Star`` + ``CZ`` +
``cz`` + ``roughT``/``roughQ`` + the ``Psi`` stability functions), the flux
closure ``turbulent_fluxes``, air properties (``air_density``/``air_cp`` from
meteo.cc) and the latent-heat helpers ``Levap``/``latent``.

For a bare snow/soil surface (no canopy):

    aero_resistance(...) -> rh, rv, Lobukhov         # Businger
    H = cp*rho*(Tg-Ta)/min(rh, 2600)                 # turbulent_fluxes
    E = rho*(alpha*Qg - Qa)/(rv/beta)
    LE = latent(Tg, Levap(Tg)) * E                   # +Lf below 0C (sublimation)

Over snow the evaporation parameters are ``alpha = beta = 1``
(find_actual_evaporation_parameters, snow branch).
"""

from __future__ import annotations

import math

from .. import constants as C

KA = 0.41            # von Karman
GRAVITY = 9.81
TK = 273.15


# ---------------------------------------------------------------------------
# air properties (meteo.cc)
# ---------------------------------------------------------------------------
def air_density(T: float, Q: float, P: float) -> float:
    """Moist-air density [kg/m3]; T in C, Q spec. humidity, P in mbar."""
    return P * 100 / (287.04 * (T + 273.15)) * (
        1 - (Q * P / (0.622 + 0.368 * Q)) / P * (1 - 0.622))


def air_cp(T: float) -> float:
    """Air specific heat [J/(kg K)] (Garratt 1992)."""
    return 1005.00 + (T + 23.15) * (T + 23.15) / 3364.0


# GEOtop: src/geotop/turbulence.cc:512-524
def Levap(T: float) -> float:
    """Latent heat of vaporisation [J/kg]."""
    if T > 0.0:
        return 2501000.0 + (2406000.0 - 2501000.0) / 40.0 * T
    return 2501000.0


LF = 333700.0        # latent heat of fusion [J/kg] (GTConst::Lf)


def latent(Ts: float, Le: float) -> float:
    """Latent heat: +Lf (sublimation) below 0 C, else evaporation."""
    return Le + LF if Ts < 0 else Le


# ---------------------------------------------------------------------------
# stability functions
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/turbulence.cc:120-155 (Psim, Psih, Zero, PsiStab)
def Psim(z: float) -> float:
    x = (1.0 - 15.0 * z) ** 0.25
    return (2.0 * math.log((1.0 + x) / 2.0) + math.log((1.0 + x * x) / 2.0)
            - 2.0 * math.atan(x) + 0.5 * C.Pi)


def Psih(z: float) -> float:
    x = (1.0 - 15.0 * z) ** 0.25
    return 2.0 * math.log((1.0 + x * x) / 2.0)


def Zero(z: float) -> float:
    return 0.0


# GEOtop: src/geotop/turbulence.cc:144-155
def PsiStab(z: float) -> float:
    """Holtslag & De Bruin stable-side function."""
    return 10.71 + 0.7 * z + 0.75 * (z - 14.28) * math.exp(-0.35 * z)


# ---------------------------------------------------------------------------
# conductances
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/turbulence.cc:273-418 (cz, CZ, roughT, roughQ)
def cz(zmeas, z0, d0, L, unstab, stab):
    zeta = (zmeas - d0) / L
    if zeta < 0:
        return math.log((zmeas - d0) / z0) - unstab(zeta) + unstab(z0 / L)
    return math.log((zmeas - d0) / z0) + stab(zeta) - stab(z0 / L)


def CZ(state, zmeas, z0, d0, L, Psi):
    if state == 1:      # instability + stability
        return cz(zmeas, z0, d0, L, Psi, PsiStab)
    if state == 2:      # instability only
        return cz(zmeas, z0, d0, L, Psi, Zero)
    if state == 3:      # stability only
        return cz(zmeas, z0, d0, L, Zero, PsiStab)
    if state == 4:      # neither
        return cz(zmeas, z0, d0, L, Zero, Zero)
    raise ValueError(f"bad turbulence state {state}")


def roughT(M, N, R):
    if M <= 0.135:
        b0, b1, b2 = 1.250, 0.0, 0.0
    elif M < 2.5:
        b0, b1, b2 = 0.149, -0.550, 0.0
    else:
        b0, b1, b2 = 0.317, -0.565, -0.183
    lm = math.log(M)
    return R + N * math.exp(b0 + b1 * lm + b2 * (lm * lm))


def roughQ(M, N, R):
    if M <= 0.135:
        b0, b1, b2 = 1.610, 0.0, 0.0
    elif M < 2.5:
        b0, b1, b2 = 0.351, -0.628, 0.0
    else:
        b0, b1, b2 = 0.396, -0.512, -0.180
    lm = math.log(M)
    return R + N * math.exp(b0 + b1 * lm + b2 * (lm * lm))


# GEOtop: src/geotop/turbulence.cc:339-350
def Star(a, zmeas, z0, d0, L, u, delta, M, N, R, Psi, roughness):
    """Return (var, c, z0v)."""
    z0v = z0 * roughness(M, N, R)
    c = CZ(a, zmeas, z0v, d0, L, Psi)
    var = delta * KA / c
    return var, c, z0v


# ---------------------------------------------------------------------------
# Businger Monin-Obukhov iteration
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/turbulence.cc:425-504
def Businger(a, zmu, zmt, d0, z0, v, T, DT, DQ, z0_z0t, maxiter):
    """Return ``(rm, rh, rv, Lobukhov)``. ``T`` = 0.5*(Tg+Ta) [C], ``DT`` =
    Tg-Ta, ``DQ`` = Qg-Qa, ``z0_z0t`` = 0 for a rigid (snow/soil) surface."""
    L = 1.0e5 if DT < 0 else -1.0e5
    u_star = T_star = Q_star = 0.0
    cm = ch = cv = 0.0
    cont = 0
    tol = 1.0e99
    while True:
        if cont > 0:
            tol = 10 * T_star + 100 * u_star + 1000 * Q_star
        u_star, cm, z0v = Star(a, zmu, z0, d0, L, 0.0, v, 1.0, 0.0, 1.0, Psim, roughT)
        if z0_z0t == 0.0:   # rigid surface
            T_star, ch, z0t = Star(a, zmt, z0, d0, L, u_star, DT,
                                   u_star * z0 / 1.4e-5, 1.0, 0.0, Psih, roughT)
            Q_star, cv, z0q = Star(a, zmt, z0, d0, L, u_star, DQ,
                                   u_star * z0 / 1.4e-5, 1.0, 0.0, Psih, roughQ)
        else:               # bending surface
            T_star, ch, z0t = Star(a, zmt, z0, d0, L, u_star, DT,
                                   1.0, 0.0, 1.0 / z0_z0t, Psih, roughT)
            Q_star, cv, z0q = Star(a, zmt, z0, d0, L, u_star, DQ,
                                   1.0, 0.0, 1.0 / z0_z0t, Psih, roughQ)
        numerator = -u_star * u_star * (T + TK)
        denominator = KA * GRAVITY * (T_star - 0.61 * Q_star * (T + TK))
        if denominator == 0.0:
            # C++ floating-point division yields +/-inf at exact neutrality;
            # Python raises ZeroDivisionError.  Preserve the C++ limit, for
            # which the stability corrections tend to zero.
            L = math.copysign(math.inf, numerator * math.copysign(1.0, denominator))
        else:
            L = numerator / denominator
        cont += 1
        if not (abs(100 * T_star + 100 * u_star + 1000 * Q_star - tol) > 0.01
                and cont <= maxiter):
            break
    rm = cm * cm / (KA * KA * v)
    rh = ch * cm / (KA * KA * v)
    rv = cv * cm / (KA * KA * v)
    return rm, rh, rv, L


# GEOtop: src/geotop/turbulence.cc:36-80 (only the state_turb==1 branch)
def aero_resistance(zmu, zmt, z0, d0, z0_z0t, v, Ta, T, Qa, Q, P,
                    MO=2, maxiter=5):
    """Aerodynamic resistances via Businger (state_turb=1). Returns
    ``(rm, rh, rv, Lobukhov)``."""
    return Businger(MO, zmu, zmt, d0, z0, v, 0.5 * T + 0.5 * Ta,
                    T - Ta, Q - Qa, z0_z0t, maxiter)


# ---------------------------------------------------------------------------
# flux closure
# ---------------------------------------------------------------------------
# GEOtop: src/geotop/turbulence.cc:87-112
def turbulent_fluxes(rh, rv, P, Ta, T, Qa, Q, dQdT):
    """Return ``(H, dHdT, E, dEdT)``. ``rh`` is capped at
    2600 s/m (windless-exchange floor on conductance, Jordan et al. 1999)."""
    rho = air_density(0.5 * (Ta + T), Qa, P)
    cp = air_cp(0.5 * (Ta + T))
    rh_c = min(rh, 2.6e3)
    H = cp * rho * (T - Ta) / rh_c
    dHdT = cp * rho / rh_c
    E = rho * (Q - Qa) / rv
    dEdT = rho * dQdT / rv
    return H, dHdT, E, dEdT


# GEOtop: src/geotop/turbulence.cc:550-652 (find_actual_evaporation_parameters)
# GEOtop: src/geotop/input.cc:1462-1472 (soil_evap_layer_bare, sized by z_evap)
def soil_evaporation_parameters(soil, Dlayer_mm, Temp, theta, P, rv, Ta, Qa,
                                Qgsat, psi_surface=-1.0, nlayers=None):
    """Ye--Pielke bare-soil ``(alpha, beta, evap_layer)``.

    This is ``find_actual_evaporation_parameters``.
    Inputs are 1-based sequences; ``Dlayer_mm`` carries the layer thicknesses
    in millimetres, the unit the molecular resistance formula consumes and
    the unit of the parameter table's own ``jdz`` row -- converting metre
    thicknesses back to mm with ``*1000`` does not recover those exact
    values, and the last ULP of the resistance reaches alpha/beta, then the
    surface humidity, then the whole Newton direction.
    ``evap_layer`` is the water-vapour mass flux from each soil layer
    [kg m-2 s-1], positive for evaporation.

    ``nlayers`` is the length of GEOtop's ``soil_evap_layer_bare`` minus one --
    the layers reached by ``z_evap``, not the whole soil column.
    ``None`` keeps the historical geotop_py behaviour of summing over every layer.
    """
    n = len(soil) if nlayers is None else nlayers + 1
    evap = [0.0] * n
    rho = air_density(0.5 * (Ta + Temp[1]), Qa, P)

    if psi_surface > 10.0:                 # ponding
        evap[1] = rho * (Qgsat - Qa) / rv
        return 1.0, 1.0, evap
    if theta[1] >= soil[1].sat:            # saturation
        evap[1] = theta[1] * rho * (Qgsat - Qa) / rv
        return 1.0, theta[1], evap

    A = 0.0
    B = 0.0
    resistance = [0.0] * n
    sat_deficit_top = soil[1].sat - theta[1]
    for l in range(1, n):
        # C++ pow_2((T+tk)/tk): GCC turns pow(x, 2.0) into x*x; Python's **
        # goes through libm pow and rounds differently for some arguments.
        x = (Temp[l] + TK) / TK
        diffusivity = 21.7 * (x * x) * (1013.25 / P)
        # GEOtop: (1.E3/D) * soil(jdz,l), with jdz the parameter table's own
        # millimetre row -- which is why Dlayer_mm must carry those exact
        # values rather than metre thicknesses scaled back up.
        resistance[l] = (1.0e3 / diffusivity) * Dlayer_mm[l]
        if l > 1:
            resistance[l] += resistance[l - 1]

        # Saturation specific humidity, using the same Buck expression as
        # meteo.cc/rad.sat_vap_pressure without importing rad circularly.
        sat_vap = 6.1121 * (1.0007 + 3.46e-6 * P) * math.exp(
            17.502 * Temp[l] / (240.97 + Temp[l]))
        Qsat = 0.622 * sat_vap / (P - 0.378 * sat_vap)
        if theta[l] <= soil[l].fc:
            hs = 0.5 * (1.0 - math.cos(
                C.Pi * (theta[l] - soil[l].res) /
                (soil[l].fc - soil[l].res)))
        else:
            hs = 1.0

        deficit = soil[l].sat - theta[l]
        ratio = deficit / sat_deficit_top
        evap[l] = rho * deficit * hs * Qsat / resistance[l]
        A += ratio * (rv / resistance[l]) * hs * Qsat
        B += ratio * (rv / resistance[l])

    Qs = (Qa + A) / (1.0 + B)
    for l in range(1, n):
        evap[l] -= rho * (soil[l].sat - theta[l]) * Qs / resistance[l]

    F = soil[1].sat / (soil[1].sat - soil[1].res)
    evap[1] += rho * (theta[1] - soil[1].res) * F * (Qgsat - Qa) / rv
    beta = ((soil[1].sat - theta[1]) + (theta[1] - soil[1].res) * F
            - (soil[1].sat - theta[1]) / (1.0 + B))
    alpha = (((theta[1] - soil[1].res) * F
              + (soil[1].sat - theta[1]) * A / (Qgsat * (1.0 + B))) / beta)
    return alpha, beta, evap
