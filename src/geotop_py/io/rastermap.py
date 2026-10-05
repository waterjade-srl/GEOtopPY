"""Raster maps in ESRI ASCII grid format.

A grid is six header lines -- ``ncols``, ``nrows``, ``xllcorner``,
``yllcorner``, ``cellsize``, ``NODATA_value`` -- followed by the cells,
northernmost row first, west to east. :func:`read_map` returns the cells as a
1-based :class:`RasterMap` together with the corner and cell size the
coordinate lookups need; cells equal to the file's ``NODATA_value`` come back
as the caller's own novalue.

Numbers are not parsed by :func:`float`: they are accumulated digit by digit
against ``pow(10, k)``, the same way the model's own reader does, because a
coordinate or an elevation one unit in the last place away moves everything
downstream of it. Neither is the cell stream parsed line by line: a row is
whatever the next ``ncols`` numbers are, so a grid whose lines carry a
trailing blank -- both reference cases' do -- reads the same as one whose
lines end on their last digit.

Units: coordinates and cell size in the grid's own projection [m]; cell values
are whatever the map holds.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

#: Longest run of digits a single number may have.
MAX_FIGURES = 30

#: Extensions a raster map is looked for under, in the order they are tried.
#: The first is GRASS ASCII, which is recognised but not read here.
GRASS_EXT = ".grass"
ESRI_EXT = ".asc"


class RasterError(ValueError):
    """Raised where GEOtop would abort while reading a raster map."""


@dataclass
class RasterMap:
    """A grid and the georeference the coordinate lookups need.

    ``data`` is 1-based on both axes -- ``data[r][c]`` for ``r`` in
    ``1..nrows``, ``c`` in ``1..ncols``, index 0 unused -- with row 1 the
    northernmost. ``X0``/``Y0`` are the lower-left corner, ``dx``/``dy`` the
    cell size, ``novalue`` the value standing for a missing cell.
    """
    data: List[List[float]]
    nrows: int
    ncols: int
    dx: float
    dy: float
    X0: float
    Y0: float
    novalue: float

    def at(self, r: int, c: int) -> float:
        """The cell at 1-based row ``r``, column ``c``."""
        return self.data[r][c]


class _Reader:
    """A byte stream read one character at a time, ``-1`` past the end."""

    def __init__(self, data: bytes):
        self._data = data
        self._i = 0

    def getc(self) -> int:
        if self._i >= len(self._data):
            self._i += 1
            return -1
        c = self._data[self._i]
        self._i += 1
        return c


def _no_end(c: int, *forbidden: int) -> None:
    """Abort if the character just read is one the stream may not end on."""
    if c in forbidden:
        raise RasterError("file incomplete: end of file or end of line reached")


def _skip_to_newline(r: _Reader, c: int) -> None:
    while c != 10:
        c = r.getc()
        if c == -1:
            raise RasterError("file incomplete: end of file reached")


# GEOtop: src/libraries/ascii/import_ascii.cc:290 (read_esriascii)
def read_esriascii(path: str, novalue: float) -> Tuple[List[float], List[float]]:
    """Read an ESRI ASCII grid; returns ``(header, cells)``.

    ``header`` is ``[ncols, nrows, xllcorner, yllcorner, cellsize,
    NODATA_value]``; ``cells`` is flat and row-major, ``nrows * ncols`` long,
    with every cell equal to ``NODATA_value`` replaced by ``novalue``. A grid
    with no ``NODATA_value`` line reads too, and then no cell is replaced.
    """
    with open(path, "rb") as fh:
        stream = _Reader(fh.read())

    header = [0.0] * 6
    ch = [0] * (MAX_FIGURES + 1)

    # The five mandatory header lines. Only the first character of each is
    # checked, which is all that tells them apart: n, n, x, y, c.
    first = (110, 110, 120, 121, 99)
    for i in range(5):
        header[i] = 0.0
        cont = 0
        sgn = 0
        while True:
            ch[0] = stream.getc()
            _no_end(ch[0], -1, 10, 0)
            if not (ch[0] == 32 and cont == 0):
                cont += 1
            if cont == 1 and ch[0] != first[i]:
                raise RasterError(f"{path}: header line {i + 1} is not "
                                  "in ESRI ASCII format")
            if cont == 5 and i == 2 and ch[0] == 101:
                raise RasterError(f"{path}: xllcenter/yllcenter headers are "
                                  "not supported, only xllcorner/yllcorner")
            if not (ch[0] <= 44 or ch[0] == 47 or ch[0] >= 58):
                break

        cont = 0
        if ch[0] == 45:                                  # '-'
            sgn = 1
            ch[0] = stream.getc()
            if ch[0] <= 45 or ch[0] == 47 or ch[0] >= 58:
                raise RasterError(f"{path}: sign not followed by a number")
        if 48 <= ch[0] <= 57:
            while True:
                cont += 1
                if cont >= MAX_FIGURES:
                    raise RasterError(f"{path}: number longer than "
                                      f"{MAX_FIGURES} digits")
                ch[cont] = stream.getc()
                if not (48 <= ch[cont] <= 57):
                    break
            for j in range(cont):
                header[i] += (ch[j] - 48) * math.pow(10, cont - j - 1)
            ch[0] = ch[cont]
        if ch[0] == 46:                                  # '.'
            cont = 0
            while True:
                ch[0] = stream.getc()
                if not 48 <= ch[0] <= 57:
                    break
                cont += 1
                header[i] += (ch[0] - 48) * math.pow(10, -cont)
        _no_end(ch[0], -1, 0)
        _skip_to_newline(stream, ch[0])
        if sgn == 1:
            header[i] *= -1

    if header[0] <= 0 or header[1] <= 0:
        raise RasterError(f"{path}: nrows or ncols negative or null")

    nr = int(header[1])
    nc = int(header[0])
    dtm = [0.0] * (nr * nc)

    # An optional sixth header line, NODATA_value.
    while True:
        ch[0] = stream.getc()
        _no_end(ch[0], -1, 10, 0)
        if ch[0] != 32:
            break

    has_novalue = ch[0] in (78, 110)                     # 'N' or 'n'
    if has_novalue:
        header[5] = 0.0
        cont = 0
        sgn = 0
        while True:
            ch[0] = stream.getc()
            _no_end(ch[0], -1, 10, 0)
            if not (ch[0] <= 44 or ch[0] == 47 or ch[0] >= 58):
                break
        if ch[0] == 45:
            sgn = 1
            ch[0] = stream.getc()
            if ch[0] <= 45 or ch[0] == 47 or ch[0] >= 58:
                raise RasterError(f"{path}: sign not followed by a number")
        if 48 <= ch[0] <= 57:
            while True:
                cont += 1
                if cont >= MAX_FIGURES:
                    raise RasterError(f"{path}: number longer than "
                                      f"{MAX_FIGURES} digits")
                ch[cont] = stream.getc()
                if not (48 <= ch[cont] <= 57):
                    break
            for j in range(cont):
                header[5] += (ch[j] - 48) * math.pow(10, cont - j - 1)
            ch[0] = ch[cont]
        if ch[0] == 46:
            cont = 0
            while True:
                ch[0] = stream.getc()
                if not 48 <= ch[0] <= 57:
                    break
                cont += 1
                header[5] += (ch[0] - 48) * math.pow(10, -cont)
        _no_end(ch[0], -1, 0)
        _skip_to_newline(stream, ch[0])
        if sgn == 1:
            header[5] *= -1

    _read_cells(stream, ch, dtm, nr, nc, novalue,
                header[5] if has_novalue else None, path, not has_novalue)
    return header, dtm


def _read_cells(stream: _Reader, ch: List[int], dtm: List[float],
                nr: int, nc: int, novalue: float, nodata: Optional[float],
                path: str, consumed_first: bool) -> None:
    """Fill ``dtm`` from the cell stream.

    ``consumed_first`` says the caller already read the first character of the
    cell stream -- it did, when it looked for a ``NODATA_value`` line and did
    not find one.

    A line break is only allowed where a row ends, but it may be preceded by a
    blank: the column counter then stands one past the last column, and it is
    the *stream* that starts the next row, not the enclosing loop.
    """
    end = 0
    r = 1
    while r <= nr:
        c = 0
        while True:
            c += 1
            sgn = 0
            while True:
                if not consumed_first or c != 1 or r != 1:
                    ch[0] = stream.getc()
                consumed_first = False
                if ch[0] == 10:
                    if c == nc + 1 or r > nr:
                        r += 1
                        c = 1
                    elif c != 1:
                        raise RasterError(
                            f"{path}: fewer columns than declared in row {r}")
                if ch[0] == -1:
                    if (r == nr and c == nc + 1) or r > nr:
                        end = 1
                        c = nc
                        r = nr
                    else:
                        raise RasterError(
                            f"{path}: fewer rows than declared")
                if not ((0 <= ch[0] <= 41) or ch[0] == 44 or ch[0] == 47
                        or ch[0] >= 58):
                    break

            if end == 0:
                k = (r - 1) * nc + c - 1
                if ch[0] == 45:
                    sgn = 1
                if ch[0] == 43 or ch[0] == 45:
                    ch[0] = stream.getc()
                    if ch[0] <= 45 or ch[0] == 47 or ch[0] >= 58:
                        raise RasterError(
                            f"{path}: sign not followed by a number at "
                            f"row {r}, column {c}")
                if 48 <= ch[0] <= 57:
                    cont = 0
                    while True:
                        cont += 1
                        if cont >= len(ch):
                            ch.extend([0] * len(ch))
                        ch[cont] = stream.getc()
                        if not 48 <= ch[cont] <= 57:
                            break
                    for j in range(cont):
                        dtm[k] += (ch[j] - 48) * math.pow(10, cont - j - 1)
                    ch[0] = ch[cont]
                if ch[0] == 46:                          # '.'
                    cont = 0
                    while True:
                        ch[0] = stream.getc()
                        if not 48 <= ch[0] <= 57:
                            break
                        cont += 1
                        dtm[k] += (ch[0] - 48) * math.pow(10, -cont)
                elif ch[0] == 42:                        # '*', a missing cell
                    dtm[k] = novalue
                    ch[0] = stream.getc()
                elif ch[0] != 32 and ch[0] != 10:
                    ch[0] = stream.getc()
                if sgn == 1:
                    dtm[k] *= -1
                if nodata is not None and dtm[k] == nodata:
                    dtm[k] = novalue

            if ch[0] == 10 or ch[0] == -1:
                break

        if c < nc and r <= nr:
            raise RasterError(f"{path}: fewer columns than declared in row {r}")
        if c > nc and r <= nr:
            raise RasterError(f"{path}: more columns than declared in row {r}")
        if ch[0] == -1 and r < nr:
            raise RasterError(f"{path}: fewer rows than declared")
        r += 1


# GEOtop: src/libraries/ascii/rw_maps.cc:275 (existing_file)
def existing_file(stem: str) -> int:
    """Which raster format ``stem`` exists in: 0 none, 2 GRASS, 3 ESRI ASCII."""
    if os.path.exists(stem + GRASS_EXT):
        return 2
    if os.path.exists(stem + ESRI_EXT):
        return 3
    return 0


# GEOtop: src/libraries/ascii/rw_maps.cc:374 (read_map)
def read_map(stem: str, novalue: float,
             reference: Optional[RasterMap] = None) -> RasterMap:
    """Read the raster map at file stem ``stem`` (no extension).

    ``reference``, when given, is the grid every other map of the run must
    agree with: same cell size, same corner, same shape. A map that disagrees
    is an error, not something to be resampled.
    """
    kind = existing_file(stem)
    if kind == 2:
        raise RasterError(f"{stem}{GRASS_EXT}: GRASS ASCII maps are not read")
    if kind != 3:
        raise RasterError(f"{stem}{ESRI_EXT}: does not exist")

    header, cells = read_esriascii(stem + ESRI_EXT, novalue)
    nr = int(header[1])
    nc = int(header[0])
    dx = header[4]
    dy = header[4]
    X0 = header[2]
    Y0 = header[3]

    if reference is not None:
        for got, want, what in ((dx, reference.dx, "Dx"), (dy, reference.dy, "Dy"),
                                (X0, reference.X0, "X0"), (Y0, reference.Y0, "Y0"),
                                (nr, reference.nrows, "number of rows"),
                                (nc, reference.ncols, "number of columns")):
            if got != want:
                raise RasterError(
                    f"{stem}{ESRI_EXT}: {what} is {got}, but the reference "
                    f"map has {want}")

    data = [[0.0] * (nc + 1) for _ in range(nr + 1)]
    for r in range(1, nr + 1):
        row_ = data[r]
        base = (r - 1) * nc
        for c in range(1, nc + 1):
            row_[c] = cells[base + c - 1]

    return RasterMap(data=data, nrows=nr, ncols=nc, dx=dx, dy=dy,
                     X0=X0, Y0=Y0, novalue=novalue)


def uniform_like(value: float, reference: RasterMap) -> RasterMap:
    """A grid of ``value`` over the reference's cells, missing where it is."""
    data = [[0.0] * (reference.ncols + 1) for _ in range(reference.nrows + 1)]
    for r in range(1, reference.nrows + 1):
        for c in range(1, reference.ncols + 1):
            data[r][c] = (reference.novalue
                          if reference.data[r][c] == reference.novalue else value)
    return RasterMap(data=data, nrows=reference.nrows, ncols=reference.ncols,
                     dx=reference.dx, dy=reference.dy, X0=reference.X0,
                     Y0=reference.Y0, novalue=reference.novalue)
