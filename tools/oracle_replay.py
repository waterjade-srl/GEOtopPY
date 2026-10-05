#!/usr/bin/env python3
"""Record the oracle's answers once, replay them where it cannot be built.

The oracle is GEOtop v3.0 compiled as a shared library: it needs the C++
checkout and a compiler, and without it most of the suite is not even
collected -- every module that pins against the oracle skips at import.  That is the cost of pinning against
the real model, and it is worth paying, but it should not be the cost of
checking out the repository.

So the answers are recorded.  Every distinct ``(function, arguments)`` the
suite asks the oracle is stored with its result, once, from a run that was
green; offline, a stand-in module serves those answers under the name
``geotop_py._cxx``.  A test then still compares the port against what the C++
computed -- the same numbers, read from a file instead of from a process.  The
recording is only as good as the revision it was made at, which is why it
carries the ``UPSTREAM`` commit and refuses to load against another one.

Two kinds of entry, because payload sizes differ by five orders of magnitude:

* **verbatim** -- the result as a Python literal, restored with
  ``ast.literal_eval``.  A literal keeps floats bit-exact and, unlike JSON,
  keeps a tuple a tuple, which several tests compare against.  Infinities are
  written ``1e999``, since ``inf`` is not a literal but overflow is: a neutral
  Obukhov length is genuinely infinite and comes back from ``Businger`` that
  way.
* **digest only** -- for the two dozen calls that return a whole meteo table or
  raster (155 MB between them, against 0.8 MB for the other 2407).  The stored
  hash is not a value a test can index, so the replayed call itself refuses to
  answer; a test that can put its own result in the shape the C++ returns asks
  :func:`recorded_digest` instead and compares :func:`digest` of that result,
  which proves the same bytes.

Usage::

    python3 -m tools.oracle_replay --record     # run the suite, write the table
    python3 -m tools.oracle_replay --verify     # re-record and compare, no write
    python3 -m tools.oracle_replay --report     # what is in it
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import datetime
import gzip
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tools.paths import GEOTOP_SRC, PROJECT_ROOT, UPSTREAM

#: ``GEOTOP_REPLAY_TABLE`` points the recorder and the replay at another file;
#: pointing it at a path with no table is how the no-recording case is tested.
_env_table = os.environ.get("GEOTOP_REPLAY_TABLE")
TABLE_PATH = (Path(_env_table).expanduser() if _env_table
              else PROJECT_ROOT / "tests" / "data" / "oracle_replay.json.gz")

#: Above this, a result is stored as a hash rather than a value.
VERBATIM_LIMIT = 65536

#: Above this, an argument list is *keyed* by its hash instead of its text.  A
#: third of the calls pass a whole parsed meteo table in and get a couple of
#: floats back; spelling those arguments out would put 100 MB of keys in the
#: table to address 100 KB of answers.
KEY_LIMIT = 512

#: pytest's per-test temporary directory: the run number varies, and so does
#: the pytest-xdist worker it runs on; the test name does not, so folding the
#: prefix away makes the key reproducible.
_PYTEST_TMP = re.compile(r"/tmp/pytest-of-[^/]+/pytest-\d+/(?:popen-gw\d+/)?")

_SEP = "\x00"


# --------------------------------------------------------------- key and value

def canonical(text: str) -> str:
    """Fold machine-specific paths out of an argument repr."""
    text = _PYTEST_TMP.sub("<pytest-tmp>/", text)
    text = text.replace(str(PROJECT_ROOT), "<repo>")
    text = text.replace(str(GEOTOP_SRC), "<geotop>")
    return text


def call_key(name: str, args: tuple, kwargs: dict) -> str:
    text = canonical(repr((args, sorted(kwargs.items()))))
    if len(text) > KEY_LIMIT:
        text = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    return name + _SEP + text


class _NotLiteral(Exception):
    pass


def as_literal(value: Any) -> str:
    """``value`` as Python source that ``ast.literal_eval`` restores exactly.

    Written out by hand rather than left to ``repr`` for one reason: ``repr``
    spells an infinity ``inf``, which is not a literal, and a text substitution
    on the finished repr would also rewrite the letters of any string that
    happened to contain them.  ``nan`` has no literal form at all and raises --
    a value no comparison can succeed against is not worth storing.
    """
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, float):
        if value != value:
            raise _NotLiteral("nan")
        if value == math.inf:
            return "1e999"
        if value == -math.inf:
            return "-1e999"
        return repr(value)
    if isinstance(value, (int, str, bytes)):
        return repr(value)
    if isinstance(value, tuple):
        return "(" + "".join(as_literal(v) + "," for v in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(as_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{as_literal(k)}: {as_literal(v)}"
                               for k, v in value.items()) + "}"
    raise _NotLiteral(type(value).__name__)


def encode(value: Any) -> Tuple[Optional[str], str, int]:
    """``(verbatim, digest, size)``; ``verbatim`` is ``None`` when unstorable.

    Unstorable means either too large to be worth carrying, or not expressible
    as a literal.  The digest is always computed, so a caller that only needs to
    know whether the bytes changed is served either way.
    """
    try:
        text = as_literal(value)
    except _NotLiteral:
        text = repr(value)
        return None, hashlib.sha256(text.encode()).hexdigest(), len(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if len(text) > VERBATIM_LIMIT:
        return None, digest, len(text)
    return text, digest, len(text)


def decode(verbatim: str) -> Any:
    return ast.literal_eval(verbatim)


def digest(value: Any) -> str:
    """The SHA-256 the table would record for ``value``."""
    return encode(value)[1]


class Missing(Exception):
    """No recorded answer for this call."""


# ------------------------------------------------------------------- recording

class Recorder:
    """Wraps every public callable of ``_cxx`` and remembers what it answered."""

    def __init__(self) -> None:
        self.calls: Dict[str, dict] = {}
        self.constants: Dict[str, str] = {}
        self.callables: list = []
        self.failed: list = []

    def install(self, module) -> None:
        for name in dir(module):
            if name.startswith("_"):
                continue
            value = getattr(module, name)
            if inspect.isfunction(value) or isinstance(value, ctypes._CFuncPtr):
                self.callables.append(name)
                setattr(module, name, self._wrap(name, value))
            elif isinstance(value, (int, float, str, bool, list, tuple, dict)):
                verbatim, _digest, _n = encode(value)
                if verbatim is not None:
                    self.constants[name] = verbatim

    def _wrap(self, name, fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            verbatim, digest, size = encode(result)
            entry = {"d": digest, "n": size}
            if verbatim is not None:
                entry["v"] = verbatim
            else:
                self.failed.append((name, size))
            self.calls[call_key(name, args, kwargs)] = entry
            return result
        return wrapper

    def dump(self, path=TABLE_PATH) -> dict:
        payload = {
            "upstream": UPSTREAM.get("commit", ""),
            "recorded": datetime.datetime.now().replace(microsecond=0).isoformat(),
            "verbatim_limit": VERBATIM_LIMIT,
            "key_limit": KEY_LIMIT,
            "constants": self.constants,
            "callables": sorted(self.callables),
            "calls": self.calls,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # mtime 0 so re-recording an unchanged table produces an identical file.
        with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as fh:
            fh.write(json.dumps(payload, sort_keys=True, indent=0).encode())
        return payload


# --------------------------------------------------------------------- replay

def load(path=TABLE_PATH) -> Optional[dict]:
    """The recorded table, or ``None`` if absent or recorded elsewhere.

    A table from another upstream revision is refused rather than used: its
    answers are what a different GEOtop computed, and silently comparing
    against those would turn the whole suite into a tautology about the past.
    """
    try:
        with gzip.open(path, "rb") as fh:
            table = json.loads(fh.read().decode())
    except (OSError, ValueError, EOFError):
        return None
    if not isinstance(table, dict) or "calls" not in table:
        return None
    want = UPSTREAM.get("commit")
    if want and table.get("upstream") and table["upstream"] != want:
        return None
    return table


def build_module(table: dict, on_missing=None,
                 name: str = "geotop_py._cxx") -> types.ModuleType:
    """A stand-in for ``geotop_py._cxx`` answering from the recorded table.

    ``on_missing`` is called with an explanation when a call has no recorded
    answer; pass ``pytest.skip`` to turn that into a skip rather than an error,
    which is what a caller without the oracle wants.  Default is to raise.
    """
    module = types.ModuleType(name)
    module.__doc__ = (
        "Recorded answers of the GEOtop v3.0 oracle, replayed offline.\n\n"
        f"Recorded {table.get('recorded', '?')} against upstream "
        f"{table.get('upstream', '?')[:12]}."
    )
    module.REPLAY = True
    module.RECORDED = table.get("recorded", "")
    calls = table["calls"]
    module._replay_calls = calls

    for const_name, verbatim in table["constants"].items():
        setattr(module, const_name, decode(verbatim))

    # Every callable the real module exposes, not only those the suite happened
    # to call: a name that is missing raises AttributeError, which no guard
    # turns into a skip.
    names = set(table.get("callables") or ())
    names.update(key.split(_SEP, 1)[0] for key in calls)
    for fn_name in names:
        setattr(module, fn_name, _replayer(fn_name, calls, on_missing))
    return module


def _replayer(fn_name, calls, on_missing):
    def fail(message):
        if on_missing is not None:
            on_missing(message)
        raise Missing(message)

    def replay(*args, **kwargs):
        entry = calls.get(call_key(fn_name, args, kwargs))
        if entry is None:
            fail(f"no recorded oracle answer for {fn_name}(); build the oracle "
                 f"(oracle/build.sh) or re-record with "
                 f"`python3 -m tools.oracle_replay --record`")
        elif "v" not in entry:
            fail(f"{fn_name}() answers {entry['n']} bytes, above the "
                 f"{VERBATIM_LIMIT}-byte verbatim limit, so only its digest is "
                 f"recorded; comparing element by element needs the real oracle")
        return decode(entry["v"])
    replay.__name__ = fn_name
    return replay


def recorded_digest(module, fn_name: str, *args, **kwargs) -> Optional[str]:
    """The digest ``module`` holds in place of the answer to this call.

    ``None`` when ``module`` is the real oracle, or a replay holding this answer
    verbatim or not at all: then the call itself is the way to ask.  The
    arguments must be exactly those of the call, since they are the key.
    """
    calls = getattr(module, "_replay_calls", None)
    if calls is None:
        return None
    entry = calls.get(call_key(fn_name, args, kwargs))
    if entry is None or "v" in entry:
        return None
    return entry["d"]


# ------------------------------------------------------------------------- CLI

def content(table: dict) -> dict:
    """The part of a table that must not change between recordings.

    Excludes the timestamp: a re-recording is expected to differ there and
    nowhere else, which is exactly what ``--verify`` asserts.
    """
    return {k: v for k, v in table.items() if k != "recorded"}


def _verify(argv) -> int:
    """Re-record against the live oracle and require the same answers.

    This is the loop nothing else closes. A recording is trusted by everything
    that runs offline, and drifts from the library it was taken from with no
    symptom -- an edited table, a stale entry a test now happens to hit. Here
    the oracle is rebuilt and asked again.
    """
    shipped = load()
    if shipped is None:
        print(f"no usable table at {TABLE_PATH} to verify", file=sys.stderr)
        return 1
    fresh_path = PROJECT_ROOT / ".trace" / "verify.json.gz"
    rc = _record(argv, table_path=fresh_path)
    if rc != 0:
        return rc
    fresh = load(fresh_path)
    if fresh is None:
        print("the re-recording produced no usable table", file=sys.stderr)
        return 1
    if content(shipped) == content(fresh):
        print(f"\nthe recorded table still matches the oracle: "
              f"{len(fresh['calls'])} calls")
        return 0

    old, new = shipped["calls"], fresh["calls"]
    changed = [k for k in old.keys() & new.keys() if old[k] != new[k]]
    print("\nthe recording no longer matches the oracle:", file=sys.stderr)
    print(f"  {len(new.keys() - old.keys())} calls not in the shipped table",
          file=sys.stderr)
    print(f"  {len(old.keys() - new.keys())} calls the suite no longer makes",
          file=sys.stderr)
    print(f"  {len(changed)} calls answered differently", file=sys.stderr)
    for key in sorted(changed)[:10]:
        print(f"    {key.split(_SEP, 1)[0]}", file=sys.stderr)
    return 1


def _record(argv, table_path=None) -> int:
    plugin = PROJECT_ROOT / "tools" / "_record_plugin.py"
    plugin.write_text(_PLUGIN_SOURCE)
    try:
        env = dict(os.environ, ORACLE_RECORD="1")
        env.pop("GEOTOP_ORACLE_SO", None)
        if table_path is not None:
            os.makedirs(os.path.dirname(table_path), exist_ok=True)
            env["GEOTOP_REPLAY_TABLE"] = str(table_path)
        rc = subprocess.call(
            [sys.executable, "-m", "pytest", "-q", "-p", "tools._record_plugin"]
            + list(argv), cwd=PROJECT_ROOT, env=env)
    finally:
        plugin.unlink(missing_ok=True)
    if rc != 0:
        print("the suite was not green; the table was not written",
              file=sys.stderr)
    return rc


_PLUGIN_SOURCE = '''\
"""Generated by tools/oracle_replay.py --record; deleted afterwards."""
import atexit
import sys

sys.path.insert(0, "src")
from geotop_py import _cxx                                   # noqa: E402

from tools.oracle_replay import Recorder                     # noqa: E402

_rec = Recorder()
_rec.install(_cxx)


@atexit.register
def _dump():
    payload = _rec.dump()
    print(f"\\nrecorded {len(payload['calls'])} calls, "
          f"{len(payload['constants'])} constants, "
          f"{len(_rec.failed)} digest-only")
'''


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--record", action="store_true",
                   help="run the suite against the real oracle and write the table")
    p.add_argument("--verify", action="store_true",
                   help="re-record against the real oracle and require the same "
                        "answers; leaves the recorded table untouched (writes only "
                        "the gitignored .trace/verify.json.gz, and a temporary "
                        "tools/_record_plugin.py)")
    p.add_argument("--report", action="store_true", help="summarise the table")
    args, rest = p.parse_known_args(argv)

    if args.record:
        return _record(rest)
    if args.verify:
        return _verify(rest)

    table = load()
    if table is None:
        print(f"no usable table at {TABLE_PATH}", file=sys.stderr)
        return 1
    calls = table["calls"]
    digest_only = [k for k, v in calls.items() if "v" not in v]
    by_fn: Dict[str, int] = {}
    for key in calls:
        fn = key.split(_SEP, 1)[0]
        by_fn[fn] = by_fn.get(fn, 0) + 1
    print(f"{TABLE_PATH.relative_to(PROJECT_ROOT)}  "
          f"{TABLE_PATH.stat().st_size / 1024:.0f} KiB")
    print(f"recorded {table['recorded']} against {table['upstream'][:12]}")
    print(f"{len(calls)} calls over {len(by_fn)} functions, "
          f"{len(table['constants'])} constants, "
          f"{len(digest_only)} digest-only")
    for fn, n in sorted(by_fn.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {fn:52s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
