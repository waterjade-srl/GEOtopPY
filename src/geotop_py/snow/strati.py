"""Snow stratigraphy: layer merging, splitting and recombination.

Line-by-line translation of the layer-management routines in ``geotop/snow.cc``
(``snow_layer_combination``, ``snowlayer_merging``, ``merge_layers``,
``initialize_snow``), operating on a :class:`geotop_py.snow.state.SnowColumn` instead of
a ``Statevar3D`` at a fixed (r, c).

The heart of the module is :func:`snowlayer_merging`: it fuses two layers by
**summing their internal energy** and re-inverting the state equation, so the
remap is enthalpy-conservative by construction. That conservation is exactly
what the tests assert -- and, unlike a spot value, it holds for the whole
operation, not just the underlying laws.

The enthalpy laws are injectable (``ienergy`` / ``from_ienergy``) so a test can
drive the same merge through GEOtop's compiled C++ oracle and confirm the Python
composition agrees with it.

Not translated here: ``set_snowice_min`` (operates on the 1-D ``Statevar1D``
recovery path, a different structure) and ``snow_compactation`` (belongs to the
water-balance step, wb.py).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from .. import laws
from .state import SnowColumn, SnowError

# File-local constants from snow.cc
SIMPL_SNOW = 1e-1
NO_SNOW = 1e-6

IEnergy = Callable[[float, float, float], float]
FromIEnergy = Callable[[float, float, float, float], tuple]


def _defaults(ienergy, from_ienergy):
    return (ienergy or laws.internal_energy,
            from_ienergy or laws.from_internal_energy)


# -------------------------------------------------------------------------
# elementary operations
# -------------------------------------------------------------------------

def initialize_snow(col: SnowColumn, l: int) -> None:
    """Zero out layer l (GEOtop initialize_snow)."""
    col.w_ice[l] = 0.0
    col.w_liq[l] = 0.0
    col.Dzl[l] = 0.0
    col.T[l] = 0.0


def snowlayer_merging(col: SnowColumn, a: float, l1: int, l2: int, lres: int,
                      ienergy: Optional[IEnergy] = None,
                      from_ienergy: Optional[FromIEnergy] = None) -> None:
    """Fuse layers l1 and l2 into lres, conserving internal energy.

    Depth, ice mass and liquid mass add; the enthalpy of the merged layer is the
    sum of the two enthalpies, and the temperature/partition are recovered from
    it. Mirrors snow.cc:snowlayer_merging.
    """
    ienergy, from_ienergy = _defaults(ienergy, from_ienergy)

    h = (ienergy(col.w_ice[l1], col.w_liq[l1], col.T[l1])
         + ienergy(col.w_ice[l2], col.w_liq[l2], col.T[l2]))
    col.Dzl[lres] = col.Dzl[l1] + col.Dzl[l2]
    col.w_ice[lres] = col.w_ice[l1] + col.w_ice[l2]
    col.w_liq[lres] = col.w_liq[l1] + col.w_liq[l2]
    if col.Dzl[lres] < 0 or col.w_ice[lres] < 0 or col.w_liq[lres] < 0:
        raise SnowError(
            f"ERROR 1 in snow layer merging l1:{l1} l2:{l2} lres:{lres}")

    wi, wl, T = from_ienergy(a, h, col.w_ice[lres], col.w_liq[lres])
    col.w_ice[lres], col.w_liq[lres], col.T[lres] = wi, wl, T
    if col.T[lres] > 0:
        raise SnowError(
            f"ERROR 2 in snow layer merging l1:{l1} l2:{l2} lres:{lres}")


def merge_layers(col: SnowColumn, a: float, l1: int,
                 ienergy: Optional[IEnergy] = None,
                 from_ienergy: Optional[FromIEnergy] = None) -> None:
    """Merge layer l1 into a neighbour, shift the rest down, drop lnum by one.

    Mirrors snow.cc:merge_layers.
    """
    if l1 > col.lnum:
        raise SnowError("Error 1 in merge_layers")

    if l1 == col.lnum:
        snowlayer_merging(col, a, l1, l1 - 1, l1 - 1, ienergy, from_ienergy)
        initialize_snow(col, l1)
    else:
        snowlayer_merging(col, a, l1, l1 + 1, l1, ienergy, from_ienergy)
        for l in range(l1 + 1, col.lnum):
            col.w_ice[l] = col.w_ice[l + 1]
            col.w_liq[l] = col.w_liq[l + 1]
            col.T[l] = col.T[l + 1]
            col.Dzl[l] = col.Dzl[l + 1]
        initialize_snow(col, col.lnum)
    col.lnum -= 1


# -------------------------------------------------------------------------
# the recombination driver
# -------------------------------------------------------------------------

def snow_layer_combination(col: SnowColumn, a: float, Ta: float,
                           inf: Sequence[int],
                           SWEmax_layer: float, SWEmax_tot: float,
                           ienergy: Optional[IEnergy] = None,
                           from_ienergy: Optional[FromIEnergy] = None) -> None:
    """Enforce the layering rules on the column (snow.cc:snow_layer_combination).

    Caps total SWE, classifies the pack (type 0/1/2), removes too-thin layers,
    splits too-thick ones and, at the maximum layer count, merges the pair of
    adjacent layers with the least combined ice. ``inf`` is GEOtop's 1-based list
    of preferred merge/split positions (``inf[0]`` is unused).

    The final consistency check (total depth and SWE unchanged to 1e-3) raises
    :class:`SnowError` instead of aborting the process.
    """
    ienergy, from_ienergy = _defaults(ienergy, from_ienergy)

    def merging(l1, l2, lres):
        snowlayer_merging(col, a, l1, l2, lres, ienergy, from_ienergy)

    max = col.max

    # check on SWEmax [kg/m2]: trim the pack from the top so cumulative ice
    # never exceeds SWEmax_tot
    SWE = 0.0
    l = 1
    while l <= col.lnum:
        SWE += col.w_ice[l]
        ice = col.w_ice[l]
        col.w_ice[l] -= max_(0.0, SWE - SWEmax_tot)
        if col.w_ice[l] < 0:
            col.w_ice[l] = 0.0
        if ice > 0:
            col.Dzl[l] *= col.w_ice[l] / ice
            col.w_liq[l] *= col.w_ice[l] / ice
        l += 1

    # D = snow depth [mm], SWE over all allocated layers
    D = 0.0
    SWE = 0.0
    for l in range(1, max + 1):
        D += col.Dzl[l]
        SWE += col.w_ice[l] + col.w_liq[l]

    # PREPROCESSING -- classify the pack
    if SWE < NO_SNOW * SWEmax_layer:
        # 1. negligible: reset to no snow
        col.lnum = 0
        col.type = 0
        for l in range(1, max + 1):
            col.Dzl[l] = 0.0
            col.w_ice[l] = 0.0
            col.w_liq[l] = 0.0
            col.T[l] = 0.0
    elif col.lnum > 0 and SWE < SIMPL_SNOW * SWEmax_layer:
        # 2. thin existing pack -> simplified (single layer)
        col.type = 1
        if col.lnum > 1:
            for l in range(col.lnum, 1, -1):
                merging(l, l - 1, l - 1)
            for l in range(2, max + 1):
                col.T[l] = 0.0
                col.Dzl[l] = 0.0
                col.w_liq[l] = 0.0
                col.w_ice[l] = 0.0
            col.lnum = 1
    elif col.lnum > 0 and SWE >= SIMPL_SNOW * SWEmax_layer:
        # 3. ordinary
        col.type = 2
    elif col.lnum == 0 and SWE < SIMPL_SNOW * SWEmax_layer:
        # 4. fresh thin pack
        col.lnum = 1
        col.type = 1
        col.T[1] = min_(Ta, -0.1)
    elif col.lnum == 0 and SWE >= SIMPL_SNOW * SWEmax_layer:
        # 5. fresh deep pack
        col.lnum = 1
        col.type = 2
        col.T[1] = min_(Ta, -0.1)

    # SYMMETRICAL PARAMETERIZATION SCHEME (ordinary case only)
    if col.type != 2:
        return

    # remove layers < simpl_snow*SWEmax_layer of ice
    n = 0
    while True:
        occurring = 0
        if col.lnum > 1:
            for l in range(1, col.lnum + 1):
                if l <= col.lnum and col.w_ice[l] < SIMPL_SNOW * SWEmax_layer:
                    merge_layers(col, a, l, ienergy, from_ienergy)
                    occurring = 1
        n += 1
        if not (n <= max and occurring == 1):
            break

    # add a new layer if the top layer holds too much ice
    occurring = 1 if col.w_ice[col.lnum] > SWEmax_layer * (1.0 + SIMPL_SNOW) else 0
    if occurring == 1:
        if col.lnum == max:
            # at capacity: merge the cheapest adjacent pair among inf positions
            linf = 0
            Dmin = 9e99
            for num in range(1, len(inf)):
                if 0 < inf[num] < max:
                    if Dmin > col.w_ice[inf[num]] + col.w_ice[inf[num] + 1]:
                        Dmin = col.w_ice[inf[num]] + col.w_ice[inf[num] + 1]
                        linf = inf[num]
                elif 1 < inf[num] <= max:
                    if Dmin > col.w_ice[inf[num]] + col.w_ice[inf[num] - 1]:
                        Dmin = col.w_ice[inf[num]] + col.w_ice[inf[num] - 1]
                        linf = -inf[num]

            if linf > 0:
                merging(linf, linf + 1, linf)
            elif linf < 0:
                merging(-linf, -linf - 1, -linf - 1)
                linf = -linf - 1
            else:
                raise SnowError(
                    "Error in snow combination - Rules to combine layers "
                    "not applicable")

            for l in range(linf + 1, col.lnum):
                col.w_ice[l] = col.w_ice[l + 1]
                col.w_liq[l] = col.w_liq[l + 1]
                col.T[l] = col.T[l + 1]
                col.Dzl[l] = col.Dzl[l + 1]

            initialize_snow(col, col.lnum)
            col.lnum -= 1

        top = col.lnum
        col.w_liq[top + 1] = col.w_liq[top] * (col.w_ice[top] - SWEmax_layer) / col.w_ice[top]
        col.Dzl[top + 1] = col.Dzl[top] * (col.w_ice[top] - SWEmax_layer) / col.w_ice[top]
        col.w_ice[top + 1] = col.w_ice[top] - SWEmax_layer
        col.T[top + 1] = col.T[top]

        col.w_ice[top] -= col.w_ice[top + 1]
        col.w_liq[top] -= col.w_liq[top + 1]
        col.Dzl[top] -= col.Dzl[top + 1]

        col.lnum += 1

    # split layers holding more than 2*SWEmax_layer of ice, from the top down
    if col.lnum < max:
        while True:
            occurring = 0
            k = len(inf) - 1
            while True:
                l = inf[k]
                if col.w_ice[l] > SWEmax_layer * 2.0:
                    occurring = 1
                    for j in range(col.lnum, l, -1):
                        col.w_ice[j + 1] = col.w_ice[j]
                        col.Dzl[j + 1] = col.Dzl[j]
                        col.w_liq[j + 1] = col.w_liq[j]
                        col.T[j + 1] = col.T[j]

                    col.Dzl[l + 1] = col.Dzl[l] * SWEmax_layer / col.w_ice[l]
                    col.w_liq[l + 1] = col.w_liq[l] * SWEmax_layer / col.w_ice[l]
                    col.w_ice[l + 1] = SWEmax_layer
                    col.T[l + 1] = col.T[l]

                    col.Dzl[l] -= col.Dzl[l + 1]
                    col.w_liq[l] -= col.w_liq[l + 1]
                    col.w_ice[l] -= col.w_ice[l + 1]

                    col.lnum += 1
                k -= 1
                if not (occurring == 0 and k > 0):
                    break
            if not (col.lnum != max and occurring != 0):
                break

    # conservation check
    Dnew = 0.0
    SWEnew = 0.0
    for l in range(1, max + 1):
        Dnew += col.Dzl[l]
        SWEnew += col.w_ice[l] + col.w_liq[l]

    if abs(D - Dnew) > 0.001 or abs(SWE - SWEnew) > 0.001:
        raise SnowError(
            f"Error in snow combination: Dold:{D} Dnew:{Dnew} "
            f"SWEold:{SWE} SWEnew:{SWEnew}")


def initial_snow_layers(swe0: float, rho0: float, T0: float, max_weq_snow: float,
                        max_snow_layers: int, inf_snow_layers: Sequence[int]
                        ) -> List[Tuple[float, float, float, float]]:
    """Distribute an initial snow water equivalent into layers, bottom-up.

    Returns a list of ``(Dzl, w_ice, w_liq, T)`` tuples suitable for
    :meth:`geotop_py.snow.state.SnowColumn.from_layers` -- empty when ``swe0`` is
    (near) zero. Every layer starts liquid-free, at uniform density
    ``rho0`` and temperature ``T0``.

    Below the "packed" threshold (``swe0 <= max_weq_snow * max_snow_layers``),
    layers fill from the bottom at exactly ``max_weq_snow`` each; the
    remainder becomes its own (thinner) top layer unless it is under a tenth
    of ``max_weq_snow``, in which case it merges into the last full layer
    instead of appearing as a near-empty one. At or above the threshold,
    every layer is forced to exactly ``max_weq_snow`` except the
    ``inf_snow_layers`` positions (the two merge/split reference layers,
    ``Column1D.inf_snow_layers``), which evenly absorb the rest.
    """
    # GEOtop: src/geotop/input.cc:1699-1766
    if swe0 < 1.0e-5:
        return []
    if swe0 <= max_weq_snow * max_snow_layers:
        i = int(swe0 // max_weq_snow)
        if i > 0:
            w_ice = [max_weq_snow] * i
            remainder = swe0 - i * max_weq_snow
            if remainder > 0.1 * max_weq_snow:
                w_ice.append(remainder)
            else:
                w_ice[-1] += remainder
        else:
            w_ice = [swe0]
    else:
        markers = {abs(x) for x in inf_snow_layers}
        share = ((swe0 - max_weq_snow * (max_snow_layers - len(markers)))
                / len(markers))
        w_ice = [share if n in markers else max_weq_snow
                for n in range(1, max_snow_layers + 1)]
    D0 = swe0 * 1000.0 / rho0
    return [(D0 * wi / swe0, wi, 0.0, T0) for wi in w_ice]


# Local aliases to keep the translation visually close to the C (Fmax/Fmin).
def max_(x, y):
    return x if x > y else y


def min_(x, y):
    return x if x < y else y
