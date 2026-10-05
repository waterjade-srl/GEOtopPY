"""Make the source tree importable, and stand in for the oracle when absent.

Prepending src keeps python -m pytest working from a bare checkout without
requiring an editable installation.
"""

import importlib
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# The vendored reference cases are data, not tests: thousands of input and
# expected-output files with nothing to collect. Walking them costs seconds on
# every run and can only produce surprises.
collect_ignore = ["tests/1D"]


def _install_oracle_replay() -> None:
    """Serve ``geotop_py._cxx`` from the recorded table when it cannot load.

    Must run before the test modules are imported: the modules that use the
    oracle import it at module scope and skip the whole file when that fails,
    so by collection time the decision has already been made.  With the real library present
    nothing happens here -- the recording is a fallback, never a shortcut.
    """
    try:
        importlib.import_module("geotop_py._cxx")
        return
    except FileNotFoundError:
        pass
    except ImportError:
        return

    import pytest

    from tools.oracle_replay import build_module, load

    table = load()
    if table is not None:
        sys.modules["geotop_py._cxx"] = build_module(table, on_missing=pytest.skip)


_install_oracle_replay()


def pytest_collection_modifyitems(config, items):
    """Mark what each test depends on, so a tier can be selected.

    Derived from what the test module imported rather than declared by hand:
    ``_cxx`` in a module's globals means every test in it consults the oracle,
    ``REFERENCE_1D`` means it reads the 1D cases.  A hand-written marker on
    every test would drift; this cannot.

        pytest -m "not oracle"      only what needs no C++ at all
        pytest -m oracle            only the pins against GEOtop itself
    """
    import pytest

    for item in items:
        module = getattr(item, "module", None)
        if module is None:
            continue
        if getattr(module, "_cxx", None) is not None:
            item.add_marker(pytest.mark.oracle)
        if getattr(module, "REFERENCE_1D", None) is not None:
            item.add_marker(pytest.mark.reference)
