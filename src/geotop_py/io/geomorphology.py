"""Terrain operators over a raster grid: smoothing, curvature, cell lookup.

A point simulation reaches these when its topography is not given directly but
has to come from a digital elevation model: the DEM is smoothed, the four
directional curvatures are taken over the whole grid, and the point's own cell
is then read out of each.

Grids are 1-based on both axes -- ``Z[r][c]`` for ``r`` in ``1..nr``, ``c`` in
``1..nc``, index 0 unused -- with row 1 the northernmost, matching how a raster
map is stored. A cell equal to ``novalue`` is missing and stays missing.

Units: elevations [m], cell sizes [m], curvatures [1/m], and the coordinates
:func:`row` and :func:`col` take are in the grid's own projection [m].
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Grid = List[List[float]]


def _zeros(nr: int, nc: int) -> Grid:
    return [[0.0] * (nc + 1) for _ in range(nr + 1)]


# GEOtop: src/libraries/geomorphology/geomorphology.cc:292
def topofilter(Zin: Sequence[Sequence[float]], novalue: float,
               n: int) -> Grid:
    """One smoothing pass: each cell becomes the mean of its ``(2n+1)`` square.

    Only cells that have a value take part, and the count is the number of
    those, so a cell near the edge of the grid or of the valid area averages
    over fewer neighbours rather than being pulled towards a missing one.
    """
    nr = len(Zin) - 1
    nc = len(Zin[0]) - 1
    Zout = _zeros(nr, nc)
    values = [0.0] * ((2 * n + 1) * (2 * n + 1) + 1)

    for r in range(1, nr + 1):
        for c in range(1, nc + 1):
            if int(Zin[r][c]) != int(novalue):
                cnt = 0
                for ir in range(-n, n + 1):
                    for ic in range(-n, n + 1):
                        if 1 <= r + ir <= nr and 1 <= c + ic <= nc:
                            if int(Zin[r + ir][c + ic]) != int(novalue):
                                cnt += 1
                                values[cnt] = Zin[r + ir][c + ic]
                # Each term is divided by the count before it is added, which
                # is not the same sum as dividing the total once.
                Zout[r][c] = 0.0
                for i in range(1, cnt + 1):
                    Zout[r][c] += values[i] / float(cnt)
            else:
                Zout[r][c] = float(novalue)
    return Zout


# GEOtop: src/libraries/geomorphology/geomorphology.cc:383
def multipass_topofilter(ntimes: int, Zin: Sequence[Sequence[float]],
                         novalue: float, n: int) -> Grid:
    """``ntimes`` smoothing passes over ``Zin``; ``ntimes`` of 0 copies it."""
    Zout = [list(row) for row in Zin]
    for _ in range(ntimes):
        Zout = topofilter(Zout, novalue, n)
    return Zout


# GEOtop: src/libraries/geomorphology/geomorphology.cc:192
def curvature(deltax: float, deltay: float, topo: Sequence[Sequence[float]],
              undef: float) -> Tuple[Grid, Grid, Grid, Grid]:
    """The four directional curvatures of ``topo``: N-S, W-E, NW-SE, NE-SW.

    Each is the second difference along that direction over the square of the
    spacing, so a convex cell (a ridge) is negative and a concave one (a
    hollow) positive. A cell whose two opposite neighbours are not both
    present -- on the edge of the grid, or next to a missing cell -- gets 0,
    not a one-sided estimate; a cell that is itself missing gets ``undef``.

    Diagonal spacing is the geometric mean of the two cell sizes, so the
    diagonal curvatures are comparable with the axial ones on a square grid.
    """
    nr = len(topo) - 1
    nc = len(topo[0]) - 1
    c1 = _zeros(nr, nc)
    c2 = _zeros(nr, nc)
    c3 = _zeros(nr, nc)
    c4 = _zeros(nr, nc)
    diagonal = math.sqrt(deltax * deltay)

    #: (output, row offsets, column offsets, spacing) for each direction.
    directions = ((c1, -1, 1, 0, 0, deltay),
                  (c2, 0, 0, 1, -1, deltax),
                  (c3, -1, 1, -1, 1, diagonal),
                  (c4, -1, 1, 1, -1, diagonal))

    for r in range(1, nr + 1):
        for c in range(1, nc + 1):
            if int(topo[r][c]) != int(undef):
                for out, dr1, dr2, dc1, dc2, delta in directions:
                    out[r][c] = 0.0
                    R1, R2 = r + dr1, r + dr2
                    C1, C2 = c + dc1, c + dc2
                    if 1 <= R1 <= nr and 1 <= R2 <= nr \
                            and 1 <= C1 <= nc and 1 <= C2 <= nc:
                        if int(topo[R1][C1]) != int(undef) \
                                and int(topo[R2][C2]) != int(undef):
                            out[r][c] += (topo[R1][C1] + topo[R2][C2]
                                          - 2. * topo[r][c]) / math.pow(delta, 2.)
            else:
                for out, _dr1, _dr2, _dc1, _dc2, _delta in directions:
                    out[r][c] = float(undef)
    return c1, c2, c3, c4


# GEOtop: src/libraries/geomorphology/geomorphology.cc:492
def row(N: float, nrows: int, dy: float, Y0: float, novalue: float) -> int:
    """The 1-based grid row holding northing ``N``, or ``novalue`` outside it.

    Row 1 is the northernmost; ``Y0`` is the northing of the grid's lower-left
    corner and ``dy`` the cell size along that axis.
    """
    if N < Y0 or N > Y0 + nrows * dy:
        return int(novalue)
    cnt = 0
    while True:
        cnt += 1
        if not Y0 + (nrows - cnt) * dy > N:
            return cnt


# GEOtop: src/libraries/geomorphology/geomorphology.cc:519
def col(E: float, ncols: int, dx: float, X0: float, novalue: float) -> int:
    """The 1-based grid column holding easting ``E``, or ``novalue`` outside it.

    Column 1 is the westernmost; ``X0`` is the easting of the grid's lower-left
    corner and ``dx`` the cell size along that axis.
    """
    if E < X0 or E > X0 + ncols * dx:
        return int(novalue)
    cnt = 0
    while True:
        cnt += 1
        if not X0 + cnt * dx < E:
            return cnt
