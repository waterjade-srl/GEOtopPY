"""In-memory results of :func:`geotop_py.pipeline.run_simulation`.

The output files are GEOtop's; these records are the Python addition. A run
returns ``{point ID: [StepRecord, ...]}``: one record for every internal step
the time loop committed, in time order. A nominal step that converged at once
gives one record; one that had to be halved gives one record per committed
piece. The step committed at the minimum timestep without converging is
recorded as well, with ``out.converged`` false.

Point IDs are GEOtop's own (the ``ID`` column of the point file), the same
number that names the point's output files.

:func:`steps_table` flattens the records into one dict of scalars per step, a
shape ``pandas.DataFrame`` accepts as it is, without the package depending on
pandas.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime
from typing import Dict, Iterable, List, NamedTuple, Union

from .meteo.forcing import Meteo
from .point.step import StepOut

Scalar = Union[float, int, bool, datetime]

_SCALAR_TYPES = {"float", "int", "bool", "Optional[float]"}


class StepRecord(NamedTuple):
    """One committed internal step of one point.

    ``date``
        End of the step.
    ``JD``
        End of the step as GEOtop's Julian day from year 0 [days].
    ``dt``
        Time the step advanced the clock [s].
    ``dt_run``
        Timestep of the trial that produced the record [s]. It equals ``dt``
        except when a trial that did not converge is committed: GEOtop halves
        the step once more before giving up, so the clock may advance by half
        the trial's step.
    ``trials``
        Trials the time loop ran to commit this step, failed halvings
        included; 1 when the nominal step converged at once.
    ``meteo``
        Forcing of the step at the point (:class:`geotop_py.meteo.forcing.Meteo`).
    ``out``
        State and fluxes after the step (:class:`geotop_py.point.step.StepOut`).
    """
    date: datetime
    JD: float
    dt: float
    dt_run: float
    trials: int
    meteo: Meteo
    out: StepOut


Results = Dict[int, List[StepRecord]]


def _type_name(t) -> str:
    # A string under ``from __future__ import annotations``, a type otherwise.
    if isinstance(t, str):
        return t
    return t.__name__ if isinstance(t, type) else str(t).replace("typing.", "")


def _scalar_fields(cls) -> List[str]:
    return [f.name for f in dataclasses.fields(cls) if _type_name(f.type) in _SCALAR_TYPES]


def steps_table(records: Iterable[StepRecord]) -> List[Dict[str, Scalar]]:
    """One flat row per record, for ``pandas.DataFrame(steps_table(...))``.

    Columns: ``date``, ``JD``, ``dt``, ``dt_run`` and ``trials``; the scalar
    fields of :class:`StepOut` under their own names (``swe``, ``depth``,
    ``converged``, ``solver_iterations``, ``wb_converged``, ...); those of its
    surface diagnostics prefixed ``diag_`` (``diag_SWnet``, ``diag_LWin``,
    ...); and those of :class:`Meteo` prefixed ``meteo_`` (``meteo_Ta``,
    ``meteo_Psnow``, ...). An optional value that is absent is NaN.
    """
    rows: List[Dict[str, Scalar]] = []
    out_fields = diag_fields = meteo_fields = None
    for r in records:
        if out_fields is None:
            out_fields = _scalar_fields(type(r.out))
            diag_fields = _scalar_fields(type(r.out.diag))
            meteo_fields = _scalar_fields(type(r.meteo))
        row: Dict[str, Scalar] = {"date": r.date, "JD": r.JD, "dt": r.dt,
                                  "dt_run": r.dt_run, "trials": r.trials}
        for name in out_fields:
            row[name] = getattr(r.out, name)
        for name in diag_fields:
            row["diag_" + name] = getattr(r.out.diag, name)
        for name in meteo_fields:
            v = getattr(r.meteo, name)
            row["meteo_" + name] = math.nan if v is None else v
        rows.append(row)
    return rows
