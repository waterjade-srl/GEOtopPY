"""The horizon (azimuth -> obscuring elevation angle) of a point or station.

A horizon is a small table -- azimuth in degrees, the elevation angle above
which the sky is open in that direction -- that :func:`geotop_py.energy.rad.shadows_point`
consumes to decide whether the sun is blocked by terrain. Reading one is the
same three-way fallback used by GEOtop everywhere a per-point/per-station file
is optional: no keyword at all, a keyword naming a file that does not exist,
or a real file, and only the third case is actually read from disk.

**File-write side effect not reproduced.** When a file is *named* but missing,
GEOtop writes out a flat default horizon file next to where the missing one
would have been, so a later run finds it. That is a side effect on the
filesystem, and this project's readers never write model state as a side
effect (see the project conventions) -- the *returned* horizon is identical
either way (flat, unobscured), so nothing about the computation changes, only
that this port does not also leave a file behind.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from ..constants import STRING_NOVALUE, is_absent
from .table import read_txt_matrix

#: The flat, unobscured horizon GEOtop falls back to: four points 90 degrees
#: apart, all at zero elevation.
DEFAULT_HORIZON: List[Tuple[float, float]] = [(45.0 + 90.0 * k, 0.0) for k in range(4)]

#: The ``Header*`` keywords naming the two columns, in file-column order.
HEADER_KEYWORDS = ("HeaderHorizonAngle", "HeaderHorizonHeight")


class HorizonError(ValueError):
    """Raised where GEOtop would abort while reading a horizon file."""


# GEOtop: src/geotop/meteodata.cc:458-577
def read_horizon(stem: str, index: int, col_names: List[str]
                  ) -> List[Tuple[float, float]]:
    """The horizon table for point/station ``index``, or the flat default.

    ``stem`` is the file stem as ``geotop.inpts`` gives it (``HorizonPointFile``
    or ``HorizonMeteoStationFile``); the file read is ``<stem>%04d.txt % index``.
    ``stem`` being :data:`geotop_py.io.parfile.STRING_NOVALUE`, or naming a
    file that does not exist, both give the flat default -- GEOtop treats
    "no keyword" and "keyword names a missing file" identically once the
    (skipped, see the module docstring) file-write side effect is set aside.
    """
    if stem == STRING_NOVALUE:
        return list(DEFAULT_HORIZON)

    path = f"{stem}{index:04d}.txt"
    if not os.path.exists(path):
        return list(DEFAULT_HORIZON)

    rows = read_txt_matrix(path, col_names)
    if not rows or is_absent(rows[0][0]) or is_absent(rows[0][1]):
        raise HorizonError(f"{path}: missing {col_names[0]!r} and/or {col_names[1]!r}")

    return [(row[0], row[1]) for row in rows]


def column_names(strings) -> List[str]:
    """Resolve the horizon ``Header*`` keywords against a parsed ``geotop.inpts``."""
    return [strings[k] for k in HEADER_KEYWORDS]
