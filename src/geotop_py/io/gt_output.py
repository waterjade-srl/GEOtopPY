"""Readers for GEOtop's plain-text point outputs (the reference to compare to).

Two kinds of file are produced by a ``PointSim`` run:

* ``output_tabs/point0001.txt`` -- one row per output step, ~79 comma-separated
  columns (meteo, surface energy balance, bulk snow/glacier/soil diagnostics).
* ``output_prof/{snowDepth,snowTemp,snowThetaIce,snowThetaW,soilTemp}0001.txt``
  -- per-layer profiles, columns ``L1..Lmax``; ``-9999`` marks a layer that is
  not present at that step (turned into NaN here).

Both share the leading bookkeeping columns
``Date12[DDMMYYYYhhmm],JulianDayFromYear0[days],TimeFromStart[days],
Simulation_Period,Run,IDpoint``. Dates are parsed to :class:`datetime`.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

_DATE_COL = "Date12[DDMMYYYYhhmm]"
_NODATA = -9999.0


@dataclass
class PointTable:
    """A GEOtop point*.txt table: named columns + parsed dates."""
    header: List[str]
    dates: List[datetime]
    columns: Dict[str, List[float]]         # header name -> floats (NaN if nodata)

    def __len__(self) -> int:
        return len(self.dates)

    def col(self, name: str) -> List[float]:
        return self.columns[name]


@dataclass
class Profile:
    """A per-layer profile file: for each step, the list of layer values.

    ``layers[t]`` is the length-``nlayers`` list at step ``t`` (top-of-file layer
    order, i.e. GEOtop's L1..Lmax); absent layers are NaN.
    """
    dates: List[datetime]
    nlayers: int
    layers: List[List[float]]


def _parse_date(tok: str) -> datetime:
    return datetime.strptime(tok.strip(), "%d/%m/%Y %H:%M")


def _to_float(tok: str) -> float:
    tok = tok.strip()
    if tok == "":
        return math.nan
    v = float(tok)
    return math.nan if v <= _NODATA + 1e-6 else v


def read_point(path: str) -> PointTable:
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        if header[0] != _DATE_COL:
            raise ValueError(f"{path}: unexpected first column {header[0]!r}")
        cols: Dict[str, List[float]] = {h: [] for h in header[1:]}
        dates: List[datetime] = []
        for row in reader:
            if not row or row[0].strip() == "":
                continue
            dates.append(_parse_date(row[0]))
            for name, tok in zip(header[1:], row[1:]):
                cols[name].append(_to_float(tok))
    return PointTable(header=header, dates=dates, columns=cols)


def read_profile(path: str) -> Profile:
    """Read an ``output_prof`` per-layer file. Public reader for analysis: the
    test suite compares profiles through ``tools.dashboard`` instead."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        # layer columns are the trailing L1..Ln (everything after IDpoint)
        first_layer = _first_layer_index(header)
        nlayers = len(header) - first_layer
        dates: List[datetime] = []
        layers: List[List[float]] = []
        for row in reader:
            if not row or row[0].strip() == "":
                continue
            dates.append(_parse_date(row[0]))
            layers.append([_to_float(tok) for tok in row[first_layer:]])
    return Profile(dates=dates, nlayers=nlayers, layers=layers)


def _first_layer_index(header: List[str]) -> int:
    for i, h in enumerate(header):
        if h.upper().startswith("L") and h[1:].isdigit():
            return i
    raise ValueError(f"no layer columns (L1..) found in header: {header}")
