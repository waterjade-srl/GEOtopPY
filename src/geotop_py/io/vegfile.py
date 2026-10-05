"""Per-land-cover-class time series of vegetation parameters.

``TimeDependentVegetationParameterFile`` names the stem of one file per land
cover class (``<stem>%04d.txt``, the class index). Each file is a plain
comma-separated table whose header is *not* consulted: the nine data columns
are read by position, in the canonical order of :data:`FIELDS`, right after a
leading timestamp column.

A class with no file of its own simply has none -- the static per-class
keywords stay in force for it -- so the return value is optional and a missing
file is not an error.

The values themselves are not a schedule of step changes: they are samples of a
piecewise-linear signal, averaged over each model timestep by
:func:`geotop_py.meteo.meteodata.time_interp_linear`, exactly like meteo readings.
Any timestep falling outside the file's own time span yields no value at all,
and the static keywords take over again for that step.
"""

from __future__ import annotations

import os
from typing import List, Optional

from .table import read_txt_matrix_2

#: Column order of the nine vegetation properties, after the leading date
#: column. Names are :class:`geotop_py.energy.vegetation.VegParams` fields.
FIELDS = ("Hveg", "z0thresveg", "z0thresveg2", "LSAI", "cf",
          "decay0", "expveg", "root", "rs")

#: Total column count of the file: the date column plus :data:`FIELDS`.
NCOLS = 1 + len(FIELDS)

#: Index of the timestamp column. Its values are ``DDMMYYYYhhmm`` integers --
#: the punctuation of ``01/01/1999 13:00`` is discarded by the tokenizer, so
#: the field arrives as the single number 10119991300.
COL_DATE = 0


# GEOtop: src/geotop/input.cc:741-763
def load(stem: str, lu: int) -> Optional[List[List[float]]]:
    """The raw table for land cover class ``lu`` (1-based), or ``None``.

    ``stem`` is the path stem as ``geotop.inpts`` gives it; the file read is
    ``<stem>%04d.txt % lu``. ``None`` means the class has no file, whether
    because none was named or because the named one does not exist.
    """
    path = f"{stem}{lu:04d}.txt"
    if not os.path.exists(path):
        return None
    return read_txt_matrix_2(path, NCOLS)
