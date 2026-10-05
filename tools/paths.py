"""Resolve repository data and the optional upstream GEOtop checkout.

Reference cases are bundled under tests/1D. GEOTOP_SRC selects the C++ checkout;
its default is the upstream directory at the repository root.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent



def _read_upstream() -> Dict[str, str]:
    """Parse the ``UPSTREAM`` pin file into ``{key: value}``.

    Missing or unreadable, it yields an empty mapping: the pin says which
    revision the port was verified against, which is worth reporting and
    worth warning about, but never worth refusing to run over.
    """
    out: Dict[str, str] = {}
    try:
        text = (PROJECT_ROOT / "UPSTREAM").read_text()
    except OSError:                                      # pragma: no cover
        return out
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


UPSTREAM = _read_upstream()

# The GEOtop C++ checkout: what the oracle is built from and what the
# `# GEOtop: <path>:<line>` citations resolve against.  It must sit at the
# revision recorded in UPSTREAM, or the line numbers refer to other code.
_src = os.environ.get("GEOTOP_SRC")
GEOTOP_SRC = Path(_src).expanduser() if _src else PROJECT_ROOT / "upstream"

# The 13 1D acceptance cases, each with its `output-tabs-SE27XX` expected
# output.  Preferring an in-repo `tests/1D` over the checkout lets the cases be
# vendored later without any caller changing: drop them in and they win.
_cases = os.environ.get("GEOTOP_TESTS_1D")
if _cases:
    REFERENCE_1D = Path(_cases).expanduser()
elif (PROJECT_ROOT / "tests" / "1D").is_dir():
    REFERENCE_1D = PROJECT_ROOT / "tests" / "1D"
else:
    REFERENCE_1D = GEOTOP_SRC / "tests" / "1D"


def checkout_commit(path: Path = None) -> Optional[str]:
    """The full commit hash a git checkout is at, or ``None`` if unknown."""
    try:
        out = subprocess.run(["git", "-C", str(path or GEOTOP_SRC), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):        # pragma: no cover
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def upstream_drift(path: Path = None) -> Optional[str]:
    """Warn if the checkout is not at the revision recorded in ``UPSTREAM``.

    A pin nobody reads is not a pin.  Every citation and every reference table
    is line-addressed against one revision, and a checkout that has moved keeps
    resolving -- the file still exists, the line still fits -- while pointing at
    other code.  Returns the warning text, or ``None`` when the two agree or
    when there is nothing to compare.
    """
    want = UPSTREAM.get("commit")
    got = checkout_commit(path)
    if not want or not got or want == got:
        return None
    return (f"checkout is at {got[:12]}, but UPSTREAM pins {want[:12]}"
            f" ({UPSTREAM.get('describe', '?')}); line-addressed citations and"
            " reference outputs may not match this revision")
