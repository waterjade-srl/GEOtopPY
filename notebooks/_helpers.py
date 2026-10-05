"""Small helpers shared by the get-started notebooks.

They are conveniences for the notebooks, not part of the ``geotop_py`` API:
copy a reference case to a scratch directory, edit keywords in its
``geotop.inpts``, read GEOtop tables into pandas and compare them with the C++
outputs. The runs themselves are in the notebooks.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / "tests" / "1D"
RUNS = Path(__file__).resolve().parent / "runs"
REFERENCE_DIR = "output-tabs-SE27XX"

# The comparator of the acceptance suite lives in tools/, which is not
# installed with the package.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geotop_py.io import gt_output  # noqa: E402
from geotop_py.io import keywords as _keywords  # noqa: E402

PathLike = Union[str, Path]


def reference_dir(name: str) -> Path:
    """The official GEOtop C++ outputs of a bundled case."""
    return CASES / name / REFERENCE_DIR


def prepare_case(name: str, dest: Optional[PathLike] = None) -> Path:
    """Fresh copy of ``tests/1D/<name>`` to run in; the bundled case stays intact.

    The copy leaves out the reference outputs and logs, and replaces whatever
    a previous run left at ``dest`` (default ``notebooks/runs/<name>``).
    """
    dest = Path(dest) if dest is not None else RUNS / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(CASES / name, dest, ignore=shutil.ignore_patterns(
        REFERENCE_DIR, "output-tabs", "output-prof", "geotop.log*", "_SUCCESSFUL_RUN"))
    return dest


_KNOWN = {k.lower(): k for k in (*_keywords.KEYWORDS_NUM, *_keywords.KEYWORDS_CHAR)}


def _format(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return ",".join(_format(v) for v in value)
    return repr(value)


def set_keywords(case_dir: PathLike, **values) -> List[str]:
    """Set keywords in ``case_dir/geotop.inpts``, e.g. ``SnowCorrFactor=1.6``.

    An existing assignment is rewritten in place; a keyword the file does not
    mention is appended. An unknown keyword name raises ``KeyError`` instead of
    being written where GEOtop would silently ignore it. Returns the lines
    written.
    """
    path = Path(case_dir) / "geotop.inpts"
    lines = path.read_text().splitlines()
    written = []
    for key, value in values.items():
        if key.lower() not in _KNOWN:
            raise KeyError(f"{key!r} is not a GEOtop keyword")
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=", re.IGNORECASE)
        new = f"{_KNOWN[key.lower()]} = {_format(value)}"
        hits = [i for i, line in enumerate(lines) if pattern.match(line)]
        if hits:
            # GEOtop reads the last occurrence: rewrite that one, drop the others.
            for i in hits[:-1]:
                lines[i] = "! " + lines[i]
            lines[hits[-1]] = new
        else:
            lines.append(new)
        written.append(new)
    path.write_text("\n".join(lines) + "\n")
    return written


def point_df(path: PathLike) -> pd.DataFrame:
    """A GEOtop table read by ``gt_output.read_point``, as a frame indexed by date.

    GEOtop's -9999 no-data value is NaN. A repeated header (``basin.txt`` has
    two ``Prain_above_canopy[mm]``) is keyed ``name.1``.
    """
    t = gt_output.read_point(str(path))
    df = pd.DataFrame(t.columns, index=pd.DatetimeIndex(t.dates, name="date"))
    return df


def profile_df(path: PathLike) -> pd.DataFrame:
    """A profile table: the columns after ``IDpoint`` of :func:`point_df`.

    Snow files have one column per layer (``L1`` at the base .. ``Ln``); soil
    files one per depth [mm], returned as floats.
    """
    df = point_df(path)
    df = df.iloc[:, list(df.columns).index("IDpoint") + 1:]
    try:
        df.columns = [float(c) for c in df.columns]
    except ValueError:
        pass
    return df


def compare_with_reference(name: str, out_dir: PathLike,
                           tol: float = 1e-5) -> pd.DataFrame:
    """Compare every reference table of case ``name`` with ``out_dir``.

    Same criterion as ``tools.dashboard``: a value passes when its absolute
    error is at most ``tol`` or its relative error is at most ``tol``. One row
    per file, with the worst failing column if any.
    """
    from tools import dashboard

    rows = []
    for ref in sorted(reference_dir(name).glob("*.txt")):
        d = dashboard.compare_file(str(ref), str(Path(out_dir) / ref.name), tol, tol,
                                   strict_rel=False, fine_atol=1e-8, fine_rtol=1e-5)
        worst = max(d.columns, key=lambda c: c.worst_abs, default=None)
        rows.append({"file": d.name, "status": d.status,
                     "columns": d.columns_compared, "failing": d.failing,
                     "worst column": worst.name if worst else "",
                     "worst abs error": worst.worst_abs if worst else 0.0,
                     "note": d.note})
    return pd.DataFrame(rows).set_index("file")
