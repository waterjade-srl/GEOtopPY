"""Soil-column diagnostics: active-layer (thaw) depth and water-table depth.

Each function scans one soil column from a boundary (surface or bottom) for
the first node crossing a fixed threshold (0 degC for thaw, 0 mm of matric
potential for the water table), then linearly interpolates the exact
crossing depth within the straddling layer. All are pure and stateless:
every input is a 1-based sequence over the column's nodes (index 0 unused),
matching the layout ``column.py``/``richards1d.py`` already use.

``"_up"`` scans down from the surface (the shallowest boundary found first);
``"_dw"`` scans up from the bottom. GEOtop reports both because a column can
have more than one thawed/saturated zone (e.g. a thawed surface layer over
permafrost, or a perched water table above a deeper one).
"""

from __future__ import annotations

from typing import Sequence


def nlayer(D: float, dz: Sequence[float], max_layers: int, d: int) -> int:
    """Index of the layer whose node-centre depth first reaches ``D``.

    ``dz`` is 1-based (index 0 unused). ``d=1`` rounds to the nearer node
    when ``D`` falls (non-negligibly) past a node's centre but short of the
    next; ``d=-1`` always takes the far node instead.
    """
    z = 0.0
    layer = 1
    while True:
        if layer > 1:
            z += dz[layer - 1] / 2.0
        z += dz[layer] / 2.0
        layer += 1
        if not (z < D and layer <= max_layers):
            break

    if z >= D:
        if d == -1:
            return layer - 1
        if abs(z - D) < 1.0e-5 or layer - 1 == 1:
            return layer - 1
        return layer - 2
    return layer - 1


def find_activelayerdepth_up(T: Sequence[float], th: Sequence[float],
                             thi: Sequence[float], res: Sequence[float],
                             dz: Sequence[float]) -> float:
    """Thaw depth scanning down from the surface [mm].

    0 when the top node is already frozen. All arguments 1-based,
    length ``nmax + 1``.
    """
    nmax = len(dz) - 1
    thresh = 0.0
    n = nmax
    if T[n] >= thresh:
        return sum(dz[1:n + 1])

    out = 0
    while out == 0:
        n -= 1
        if n == 1:
            out = -1
        if T[n + 1] < thresh and T[n] >= thresh:
            out = 1
    if out != 1:
        return 0.0

    table = sum(dz[1:n + 1])
    table += (1.0 - thi[n + 1] / (thi[n + 1] + th[n + 1] - res[n + 1])) * dz[n + 1]
    return table


def find_activelayerdepth_dw(T: Sequence[float], th: Sequence[float],
                             thi: Sequence[float], res: Sequence[float],
                             dz: Sequence[float]) -> float:
    """Thaw depth scanning up from the bottom [mm].

    0 when the surface node is already thawed (or the column has a single
    layer). All arguments 1-based, length ``nmax + 1``.
    """
    nmax = len(dz) - 1
    thresh = 0.0
    n = 1
    if not (T[n] < thresh and nmax > 1):
        return 0.0

    out = 0
    while out == 0:
        n += 1
        if n == nmax:
            out = -1
        if T[n] >= thresh and T[n - 1] < thresh:
            out = 1
    table = sum(dz[1:n - 1])
    if out == 1:
        table += (thi[n - 1] / (thi[n - 1] + th[n - 1] - res[n - 1])) * dz[n - 1]
    else:
        table = 0.0
    return table


# GEOtop: src/geotop/water.balance.cc:502-503 (SOIL::Ptot)
# GEOtop: src/geotop/water.balance.cc:873-874 (SOIL::Ptot)
def find_watertabledepth_up(Z: float, Ptot: Sequence[float],
                            dz: Sequence[float]) -> float:
    """Water-table depth scanning down from the surface [mm], capped at ``Z``.

    ``Ptot`` is the total-head-equivalent matric potential
    (``psi_teta`` of ``th + thi``, i.e. GEOtop's own ``SOIL::Ptot``),
    1-based, length ``nmax + 1``. ``dz``
    likewise.
    """
    thresh = 0.0
    n = nlayer(Z, dz, len(dz) - 1, -1)
    table = 0.0
    if n > 1:
        if Ptot[n] < thresh:
            table = sum(dz[1:n + 1])
        else:
            out = 0
            while out == 0:
                n -= 1
                if n == 1:
                    out = -1
                if Ptot[n + 1] >= thresh and Ptot[n] < thresh:
                    out = 1
            if out == 1:
                table = sum(dz[1:n + 1])
                table += 0.5 * dz[n + 1]
                table -= (0.5 * (dz[n] + dz[n + 1]) * (Ptot[n + 1] - thresh)
                         / (Ptot[n + 1] - Ptot[n]))
    return min(table, Z) if table > Z else table


def find_watertabledepth_dw(Z: float, Ptot: Sequence[float],
                            dz: Sequence[float]) -> float:
    """Water-table depth scanning up from the bottom [mm], capped at ``Z``.

    Starts the scan at node 3 (or node 1 in a column with 3 or fewer
    layers) to avoid reporting a water table right at the surface from a
    rain event that only saturates the very top nodes. ``Ptot``/``dz``:
    see :func:`find_watertabledepth_up`.
    """
    thresh = 0.0
    nmax = len(dz) - 1
    n = 1 if nmax <= 3 else 3
    table = 0.0
    if Ptot[n] < thresh:
        out = 0
        while out == 0:
            n += 1
            if n >= nmax:
                out = -1
            if n <= nmax and Ptot[n] >= thresh and Ptot[n - 1] < thresh:
                out = 1
        table = sum(dz[1:n])
        if out == 1:
            table += (0.5 * dz[n]
                     - (Ptot[n] - thresh) * 0.5 * (dz[n - 1] + dz[n])
                     / (Ptot[n] - Ptot[n - 1]))
        else:
            table += dz[n]
    return min(table, Z) if table > Z else table
