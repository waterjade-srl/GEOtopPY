#!/usr/bin/env python3
"""Check source citations against the pinned GEOtop revision.

Comments of the form ``# GEOtop: src/...:line[-line]`` identify translated
code. With GEOTOP_SRC available, verify the cited lines and their recorded
hashes. Without it, check citation coverage against tools/upstream_index.json.
Short file-name references are reported separately and are not hash-verified.

    python -m tools.citations --check
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from tools.paths import GEOTOP_SRC, PROJECT_ROOT, UPSTREAM, upstream_drift

SRC_DIR = PROJECT_ROOT / "src" / "geotop_py"
INDEX_PATH = PROJECT_ROOT / "tools" / "upstream_index.json"

#: A compliant citation: "# GEOtop: <path>:<line>[-<line>]", optionally followed
#: by a trailing note in parentheses, e.g. "(the point_sim==1 branch)".
_COMPLIANT = re.compile(
    r'^\s*#\s*GEOtop:\s*(?P<path>\S+\.(?:cc|h)):(?P<lo>\d+)(?:-(?P<hi>\d+))?'
    r'\s*(?:\(.*\))?\s*$')

#: Any file.cc:NNN-shaped reference, compliant or not -- used to find the
#: legacy ones by subtracting out the compliant matches per file.
_ANY_REF = re.compile(r'[\w./]+\.(?:cc|h):\d+(?:-\d+)?')


@dataclass
class Citation:
    file: Path
    lineno: int
    ref_path: str
    lo: int
    hi: int


@dataclass
class LegacyRef:
    file: Path
    lineno: int
    text: str


def iter_python_files() -> Iterator[Path]:
    yield from sorted(SRC_DIR.rglob("*.py"))


def find_compliant_citations(path: Path) -> List[Citation]:
    out = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        m = _COMPLIANT.match(line)
        if m:
            lo = int(m.group("lo"))
            hi = int(m.group("hi")) if m.group("hi") else lo
            out.append(Citation(path, lineno, m.group("path"), lo, hi))
    return out


def find_legacy_refs(path: Path, compliant_lines: set) -> List[LegacyRef]:
    out = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if lineno in compliant_lines:
            continue
        for m in _ANY_REF.finditer(line):
            out.append(LegacyRef(path, lineno, m.group(0)))
    return out


def address(c: Citation) -> str:
    """The citation as it is keyed in the index: ``<path>:<lo>-<hi>``."""
    return f"{c.ref_path}:{c.lo}-{c.hi}"


def _where(c: Citation) -> str:
    return f"{c.file.relative_to(PROJECT_ROOT)}:{c.lineno}"


def cited_lines(c: Citation, geotop_src: Path) -> Optional[List[str]]:
    """The lines a citation points at, or ``None`` if it does not resolve."""
    target = geotop_src / c.ref_path
    if not target.is_file():
        return None
    lines = target.read_text(errors="replace").splitlines()
    if c.hi > len(lines) or c.lo < 1:
        return None
    return lines[c.lo - 1:c.hi]


def digest_lines(lines: List[str]) -> str:
    """Hash of the cited text, whitespace-insensitive at the edges.

    Trailing whitespace and the indentation of a block are exactly what a
    reformatting commit changes without changing what the code does, and a
    citation that survives a reindent should not be reported as moved.
    """
    body = "\n".join(line.strip() for line in lines)
    return hashlib.sha256(body.encode()).hexdigest()[:32]


def verify_citation(c: Citation, geotop_src: Path) -> Optional[str]:
    """Returns an error string, or ``None`` if the citation resolves."""
    target = geotop_src / c.ref_path
    if not target.is_file():
        return (f"{_where(c)}: {c.ref_path} does not exist under {geotop_src}")
    nlines = sum(1 for _ in target.open())
    if c.hi > nlines:
        return (f"{_where(c)}: {c.ref_path}:{c.lo}-{c.hi} but the file only "
                f"has {nlines} lines")
    return None


# ---------------------------------------------------------------- the index

def load_index(path: Path = INDEX_PATH) -> Optional[dict]:
    """The recorded index, or ``None`` if absent or from another revision."""
    try:
        index = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    want = UPSTREAM.get("commit")
    if want and index.get("upstream") and index["upstream"] != want:
        return None
    return index


def build_index(citations: List[Citation], geotop_src: Path) -> dict:
    entries: Dict[str, str] = {}
    unresolved: List[str] = []
    for c in citations:
        lines = cited_lines(c, geotop_src)
        if lines is None:
            unresolved.append(f"{_where(c)}: {address(c)}")
            continue
        entries[address(c)] = digest_lines(lines)
    return {
        "upstream": UPSTREAM.get("commit", ""),
        "describe": UPSTREAM.get("describe", ""),
        "generated": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "unresolved": sorted(unresolved),
        "addresses": entries,
    }


def check_against_index(citations: List[Citation], index: dict) -> List[str]:
    """Errors from checking the source's citations against the record alone."""
    known = index["addresses"]
    return [f"{_where(c)}: {address(c)} is not in the index; run "
            f"`python3 -m tools.citations --index` against a checkout at "
            f"{index.get('describe') or 'the pinned revision'}"
            for c in citations if address(c) not in known]


def check_content(citations: List[Citation], index: dict,
                  geotop_src: Path) -> List[str]:
    """Errors from re-hashing what each citation points at now."""
    known = index["addresses"]
    errors = []
    for c in citations:
        recorded = known.get(address(c))
        if recorded is None:
            continue                          # reported by check_against_index
        lines = cited_lines(c, geotop_src)
        if lines is None:
            continue                          # reported by verify_citation
        if digest_lines(lines) != recorded:
            errors.append(
                f"{_where(c)}: {address(c)} still resolves but its content "
                f"changed since the index was built -- the code it cites has "
                f"moved")
    return errors


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true",
                   help="verify every compliant citation; exit 1 on failure. "
                        "With a checkout, re-hashes what each one points at; "
                        "without one, checks them against the index")
    p.add_argument("--index", action="store_true",
                   help="rebuild tools/upstream_index.json from the checkout")
    p.add_argument("--list-legacy", action="store_true",
                   help="print every legacy (non-converted) reference")
    args = p.parse_args(argv)

    geotop_src = GEOTOP_SRC

    all_compliant: List[Citation] = []
    all_legacy: List[LegacyRef] = []
    for path in iter_python_files():
        compliant = find_compliant_citations(path)
        all_compliant.extend(compliant)
        compliant_lines = {c.lineno for c in compliant}
        all_legacy.extend(find_legacy_refs(path, compliant_lines))

    have_src = geotop_src.is_dir()
    index = load_index()
    print(f"GEOTOP_SRC: {geotop_src}" + ("" if have_src else "  (NOT FOUND)"))
    print("index: " + (f"{len(index['addresses'])} addresses, built "
                        f"{index['generated']}" if index else "absent or stale"))
    print(f"compliant citations: {len(all_compliant)}")
    print(f"legacy references (not yet converted): {len(all_legacy)}")

    if args.list_legacy:
        for ref in all_legacy:
            print(f"  {ref.file.relative_to(PROJECT_ROOT)}:{ref.lineno}: {ref.text}")

    if args.index:
        if not have_src:
            print(f"error: --index needs the checkout at {geotop_src}",
                  file=sys.stderr)
            return 2
        if (drift := upstream_drift(geotop_src)):
            print(f"error: {drift}", file=sys.stderr)
            print("       indexing from the wrong revision would record the "
                  "wrong content", file=sys.stderr)
            return 2
        built = build_index(all_compliant, geotop_src)
        INDEX_PATH.write_text(json.dumps(built, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {INDEX_PATH.relative_to(PROJECT_ROOT)}: "
              f"{len(built['addresses'])} addresses"
              + (f", {len(built['unresolved'])} unresolved"
                 if built["unresolved"] else ""))
        for line in built["unresolved"]:
            print(f"  unresolved: {line}", file=sys.stderr)
        return 1 if built["unresolved"] else 0

    if not args.check:
        return 0

    if (drift := upstream_drift(geotop_src)):
        print(f"warning: {drift}", file=sys.stderr)

    errors: List[str] = []
    if index is None:
        if not have_src:
            print("error: no index and no checkout -- nothing to verify "
                  "against", file=sys.stderr)
            return 1
        print("\nno index: falling back to checking that citations resolve")
        errors = [msg for c in all_compliant
                  if (msg := verify_citation(c, geotop_src))]
    else:
        errors = check_against_index(all_compliant, index)
        if have_src:
            errors += [msg for c in all_compliant
                       if (msg := verify_citation(c, geotop_src))]
            errors += check_content(all_compliant, index, geotop_src)

    if errors:
        print(f"\n{len(errors)} citation problem(s):", file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)
        return 1

    how = ("against the index and the checkout" if index and have_src
           else "against the index" if index else f"against {geotop_src}")
    print(f"\nall {len(all_compliant)} compliant citations verified {how}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
