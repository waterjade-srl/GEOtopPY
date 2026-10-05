"""GEOtopPY: a Python port of the GEOtop hydrological model, 1D configuration.

The Python code mirrors GEOtop's C++ physics one function at a time, so each
piece can be read, tested and reworked in isolation. Correctness is defined
against the C++ itself: the pure laws are pinned to the compiled oracle in
:mod:`geotop_py._cxx`, which is the official v3.0 model built as a shared
library.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from . import laws  # noqa: F401
from .pipeline import run_simulation
from .results import StepRecord, steps_table

__all__ = ["laws", "run_simulation", "StepRecord", "steps_table"]

try:
    __version__ = _package_version("GEOtopPY")
except PackageNotFoundError:      # running from a source tree, not installed
    __version__ = "0.1.0+source"
