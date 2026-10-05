#!/usr/bin/env python3
"""Status dashboard for the 13 reference 1D cases.

The goal of the port is defined by the 1D suite of the GEOtop checkout (located
by ``tools.paths.REFERENCE_1D``): each case ships a GEOtop reference output
(``output-tabs-SE27XX/``) that the C++ v3.0 build reproduces within ``numdiff -a
1e-5 -r 1e-5``. This tool answers, in one table, how far the Python model is
from that.

``numdiff`` is not assumed to be installed, so the comparison is done here. That
is a feature rather than a workaround: the comparator reports *which columns*
drift and by how much, which a textual diff cannot.

Two values are considered equal when EITHER holds::

    |a - b| <= atol            OR      |a - b| / max(|a|, |b|) <= rtol

matching numdiff's "absolute OR relative" acceptance. The relative form here is
the symmetric one; ``--strict-rel`` switches the denominator to ``min(|a|,|b|)``
if a discrepancy with the C++ suite ever needs chasing.

A case is also out when the model writes output the reference lacks: a file
in ``output-tabs`` that GEOtop did not produce, or anything under another
``output*`` directory. GEOtop writes an output file only when its keyword is
set, so such a file is a real discrepancy.

A second, finer-grained pass/fail count rides alongside the primary one: how
many columns satisfy ``numpy.isclose``'s own formula and defaults
(``|got - ref| <= atol + rtol*|ref|``, ``atol=1e-8``, ``rtol=1e-5``). This is
strictly tighter on the absolute side than the ``1e-5`` acceptance tolerance
above, so it tracks progress *within* an already-green column -- a column
whose worst row is 1e-6 off looks identical to an exact match under the
primary metric, but the fine one tells them apart.

Usage::

    python3 -m tools.dashboard                  # all cases, from <repo>/runs
    python3 -m tools.dashboard --inplace        # compare the C++ run instead
    python3 -m tools.dashboard --case PureDrainage --verbose
    python3 -m tools.dashboard --run            # run the model first
    python3 -m tools.dashboard --json

Exit code is 0 only when every case is green.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

from tools.paths import PROJECT_ROOT, REFERENCE_1D

DEFAULT_TESTS_DIR = str(REFERENCE_1D)
DEFAULT_WORK_DIR = str(PROJECT_ROOT / "runs")
REFERENCE_DIR = "output-tabs-SE27XX"
OUTPUT_DIR = "output-tabs"
# the interpreter running the dashboard, not whichever python3 is on PATH
DEFAULT_CMD = f"{shlex.quote(sys.executable)} -m geotop_py"

# Reference cases ordered by increasing complexity. Unlisted cases keep alphabetical
# order at the end, so a newly added reference case still shows up.
LADDER: Sequence[tuple[str, Sequence[str]]] = (
    ("L0", ("Jungfraujoch",)),
    ("L1", ("Bro",)),
    ("L2", ("PureDrainage", "PureDrainageFaked")),
    ("L3", ("PureDrainageRainy", "PureDrainageRainySlope")),
    ("L4", ("ARF_1D",)),
    ("L5", ("Calabria",)),
    ("L6", ("ColdelaPorte", "CostantMeteo")),
    ("L7", ("B2_BeG_017",)),
    ("L8", ("Matsch_B2_Ref_007", "Matsch_P2_Ref_007")),
)

# Leading columns that identify the record rather than carry a result. They are
# compared, but a mismatch there is reported as a shape problem: it means the
# time axis itself is wrong, and every other column is then meaningless.
INDEX_COLUMNS = 6


# --------------------------------------------------------------------------
# table parsing
# --------------------------------------------------------------------------

@dataclass
class Table:
    """One GEOtop output table: a header line plus numeric rows.

    The first field of every row is a ``DD/MM/YYYY hh:mm`` stamp; the rest are
    numbers. Column headers are *not* unique (``basin.txt`` repeats
    ``Prain_above_canopy[mm]``), so columns are matched by position and named
    only for reporting.
    """
    path: str
    header: list[str]
    dates: list[str]
    rows: list[list[float]]

    @property
    def ncols(self) -> int:
        return len(self.header)

    @property
    def nrows(self) -> int:
        return len(self.rows)


def _as_float(token: str) -> float:
    token = token.strip()
    if not token:
        return math.nan
    try:
        return float(token)
    except ValueError:
        return math.nan


def read_table(path: str) -> Table:
    with open(path, newline="") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: empty")
    header = [h.strip() for h in lines[0].split(",")]
    dates: list[str] = []
    rows: list[list[float]] = []
    for line in lines[1:]:
        fields = line.split(",")
        dates.append(fields[0].strip())
        rows.append([_as_float(f) for f in fields[1:]])
    return Table(path=path, header=header, dates=dates, rows=rows)


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

@dataclass
class ColumnDiff:
    index: int
    name: str
    bad_rows: int
    worst_abs: float
    worst_rel: float
    worst_row: int
    worst_date: str
    ref_value: float
    got_value: float


@dataclass
class FileDiff:
    name: str
    status: str                       # "ok" | "diff" | "shape" | "missing" | "extra"
    note: str = ""
    columns_compared: int = 0
    columns: list[ColumnDiff] = field(default_factory=list)
    # Columns with at least one row failing the fine numpy.isclose metric --
    # independent of `columns` above, since a column can pass the primary
    # (looser) tolerance and still fail this one.
    fine_failing: int = 0

    @property
    def failing(self) -> int:
        return len(self.columns)


def _equal(ref: float, got: float, atol: float, rtol: float,
           strict_rel: bool) -> tuple[bool, float, float]:
    """Return (equal, absolute error, relative error).

    NaN is treated as a value: NaN matches NaN, and NaN against a number is a
    mismatch with infinite error. GEOtop writes novalue as a large sentinel
    rather than NaN, so this mostly guards against our own output.
    """
    if math.isnan(ref) and math.isnan(got):
        return True, 0.0, 0.0
    if math.isnan(ref) or math.isnan(got):
        return False, math.inf, math.inf
    err = abs(ref - got)
    if err == 0.0:
        return True, 0.0, 0.0
    scale = min(abs(ref), abs(got)) if strict_rel else max(abs(ref), abs(got))
    rel = err / scale if scale > 0 else math.inf
    return (err <= atol or rel <= rtol), err, rel


def _isclose(ref: float, got: float, atol: float, rtol: float) -> bool:
    """``numpy.isclose``'s own formula: ``|got - ref| <= atol + rtol*|ref|``.

    NaN handling matches :func:`_equal` (NaN matches NaN) rather than
    numpy's own ``equal_nan=False`` default, so the fine metric differs from
    the primary one only in the tolerance -- not in how missing/no-data
    values are judged, which would make the two counts incomparable.
    """
    if math.isnan(ref) and math.isnan(got):
        return True
    if math.isnan(ref) or math.isnan(got):
        return False
    return abs(got - ref) <= atol + rtol * abs(ref)


def compare_file(ref_path: str, got_path: str, atol: float, rtol: float,
                 strict_rel: bool, fine_atol: float, fine_rtol: float) -> FileDiff:
    name = os.path.basename(ref_path)
    if not os.path.exists(got_path):
        return FileDiff(name=name, status="missing", note="not produced")
    try:
        ref = read_table(ref_path)
        got = read_table(got_path)
    except (OSError, ValueError) as exc:
        return FileDiff(name=name, status="shape", note=str(exc))

    if ref.nrows != got.nrows:
        return FileDiff(name=name, status="shape",
                        note=f"{got.nrows} rows, reference has {ref.nrows}")
    if ref.ncols != got.ncols:
        return FileDiff(name=name, status="shape",
                        note=f"{got.ncols} columns, reference has {ref.ncols}")
    mismatched_dates = sum(1 for a, b in zip(ref.dates, got.dates) if a != b)
    if mismatched_dates:
        return FileDiff(name=name, status="shape",
                        note=f"{mismatched_dates} timestamps differ")

    ncols = ref.ncols - 1   # header includes the date column, rows do not
    diffs: list[ColumnDiff] = []
    fine_failing = 0
    for j in range(ncols):
        bad = 0
        worst_abs = worst_rel = 0.0
        worst_row = -1
        fine_ok_col = True
        for i in range(ref.nrows):
            rv, gv = ref.rows[i][j], got.rows[i][j]
            ok, err, rel = _equal(rv, gv, atol, rtol, strict_rel)
            if not ok:
                bad += 1
                if err > worst_abs:
                    worst_abs, worst_rel, worst_row = err, rel, i
            if fine_ok_col and not _isclose(rv, gv, fine_atol, fine_rtol):
                fine_ok_col = False
        if bad:
            diffs.append(ColumnDiff(
                index=j + 1,
                name=ref.header[j + 1] if j + 1 < len(ref.header) else f"col{j + 1}",
                bad_rows=bad,
                worst_abs=worst_abs,
                worst_rel=worst_rel,
                worst_row=worst_row,
                worst_date=ref.dates[worst_row],
                ref_value=ref.rows[worst_row][j],
                got_value=got.rows[worst_row][j],
            ))
        if not fine_ok_col:
            fine_failing += 1

    return FileDiff(name=name, status="ok" if not diffs else "diff",
                    columns_compared=ncols, columns=diffs,
                    fine_failing=fine_failing)


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

@dataclass
class CaseResult:
    rung: str
    name: str
    status: str            # "ok" | "diff" | "shape" | "no-output" | "no-ref" | "error"
    note: str = ""
    files: list[FileDiff] = field(default_factory=list)

    @property
    def failing_columns(self) -> int:
        return sum(f.failing for f in self.files)

    @property
    def total_columns(self) -> int:
        return sum(f.columns_compared for f in self.files)

    @property
    def fine_failing_columns(self) -> int:
        return sum(f.fine_failing for f in self.files)


def discover_cases(tests_dir: str) -> list[tuple[str, str]]:
    """Return (rung, case) pairs in ladder order, then any extras found."""
    present = {d for d in os.listdir(tests_dir)
               if os.path.isdir(os.path.join(tests_dir, d, REFERENCE_DIR))}
    ordered: list[tuple[str, str]] = []
    for rung, names in LADDER:
        for name in names:
            if name in present:
                ordered.append((rung, name))
                present.discard(name)
    ordered.extend(("--", name) for name in sorted(present))
    return ordered


def find_extra_outputs(got_dir: str, ref_files: Sequence[str]) -> list[str]:
    """Output the model wrote that GEOtop did not: files in ``got_dir`` absent
    from the reference, and any file under a sibling ``output*`` directory
    other than ``got_dir`` itself and the reference directory. Paths are
    relative to the case directory; dot files (``.placeholder``) are ignored.

    GEOtop writes an output file only when its keyword is set, so a file the
    reference lacks is spurious output, not a harmless extra.
    """
    case_dir = os.path.dirname(os.path.abspath(got_dir))
    out_name = os.path.basename(os.path.abspath(got_dir))
    expected = set(ref_files)
    extra: list[str] = []
    for entry in sorted(os.listdir(case_dir)):
        path = os.path.join(case_dir, entry)
        if not os.path.isdir(path) or not entry.startswith("output") \
                or entry == REFERENCE_DIR:
            continue
        for root, _, names in os.walk(path):
            for fn in sorted(names):
                if fn.startswith("."):
                    continue
                if entry == out_name and root == path and fn in expected:
                    continue
                extra.append(os.path.relpath(os.path.join(root, fn), case_dir))
    return extra


def evaluate_case(rung: str, name: str, tests_dir: str, got_dir: str,
                  atol: float, rtol: float, strict_rel: bool,
                  fine_atol: float = 1e-8, fine_rtol: float = 1e-5) -> CaseResult:
    ref_dir = os.path.join(tests_dir, name, REFERENCE_DIR)
    if not os.path.isdir(ref_dir):
        return CaseResult(rung, name, "no-ref", "no reference outputs")
    ref_files = sorted(f for f in os.listdir(ref_dir) if f.endswith(".txt"))
    if not ref_files:
        return CaseResult(rung, name, "no-ref", "reference directory is empty")
    if not os.path.isdir(got_dir):
        return CaseResult(rung, name, "no-output", os.path.relpath(got_dir))

    files = [compare_file(os.path.join(ref_dir, f), os.path.join(got_dir, f),
                          atol, rtol, strict_rel, fine_atol, fine_rtol)
             for f in ref_files]

    if all(f.status == "missing" for f in files):
        return CaseResult(rung, name, "no-output", "no tables produced", files)
    if any(f.status == "shape" for f in files):
        note = next(f"{f.name}: {f.note}" for f in files if f.status == "shape")
        return CaseResult(rung, name, "shape", note, files)
    if any(f.status == "missing" for f in files):
        missing = [f.name for f in files if f.status == "missing"]
        return CaseResult(rung, name, "diff",
                          f"missing {', '.join(missing)}", files)
    extra = find_extra_outputs(got_dir, ref_files)
    if extra:
        files += [FileDiff(name=e, status="extra", note="not in the reference")
                  for e in extra]
        shown = ", ".join(extra[:3]) + (f" (+{len(extra) - 3} more)"
                                        if len(extra) > 3 else "")
        return CaseResult(rung, name, "diff", f"unexpected {shown}", files)
    if any(f.status == "diff" for f in files):
        return CaseResult(rung, name, "diff", "", files)
    return CaseResult(rung, name, "ok", "", files)


def run_case(name: str, tests_dir: str, work_dir: str, cmd: str,
             timeout: int) -> str | None:
    """Copy the case into the work dir and run the model there.

    The reference clone is never written to: GEOtop resolves the output paths in
    ``geotop.inpts`` relative to the working directory, so a run in place would
    overwrite the C++ outputs that the comparator may still need.

    Returns an error string, or None on success.
    """
    src = os.path.join(tests_dir, name)
    dst = os.path.join(work_dir, name)
    os.makedirs(work_dir, exist_ok=True)
    # A fresh copy every time: what a previous run left behind would otherwise
    # be compared -- and reported as unexpected output -- as this run's.
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns(REFERENCE_DIR, OUTPUT_DIR,
                                      "failing_output", "*.log"))
    os.makedirs(os.path.join(dst, OUTPUT_DIR), exist_ok=True)
    try:
        proc = subprocess.run(shlex.split(cmd) + [dst], capture_output=True,
                              text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return f"command not found: {shlex.split(cmd)[0]}"
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout}s"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return tail[-1][:120] if tail else f"exit {proc.returncode}"
    return None


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

_MARK = {"ok": "OK", "diff": "DIFF", "shape": "SHAPE",
         "no-output": "--", "no-ref": "no ref", "error": "ERROR"}


def render(results: Sequence[CaseResult], atol: float, rtol: float,
           source: str, fine_atol: float = 1e-8, fine_rtol: float = 1e-5) -> str:
    width = max((len(r.name) for r in results), default=10)
    fine_width = 11
    lines = [
        f"GEOtop 1D -- {len(results)} reference cases",
        f"source: {source}   tolerance: abs {atol:g} or rel {rtol:g}"
        f"   fine (np.isclose): atol {fine_atol:g} rtol {fine_rtol:g}",
        "",
        f"     {'case':<{width}}  {'status':<8}  {'np.isclose':<{fine_width}}  detail",
        f"     {'-' * width}  {'-' * 8}  {'-' * fine_width}  {'-' * 40}",
    ]
    for r in results:
        if r.status == "ok":
            detail = f"{len(r.files)} tables, {r.total_columns} columns"
        elif r.status == "diff":
            detail = f"{r.failing_columns}/{r.total_columns} columns out"
            if r.note:
                detail += f" ({r.note})"
        else:
            detail = r.note
        total = r.total_columns
        fine = (f"{total - r.fine_failing_columns}/{total}" if total else "--")
        lines.append(f"{r.rung:<4} {r.name:<{width}}  "
                     f"{_MARK.get(r.status, r.status):<8}  {fine:<{fine_width}}  {detail}")

    green = sum(1 for r in results if r.status == "ok")
    lines += ["", f"{green}/{len(results)} green"]
    if green < len(results):
        nxt = next((r for r in results if r.status != "ok"), None)
        if nxt is not None:
            lines.append(f"next: {nxt.rung} {nxt.name}")
    return "\n".join(lines)


def render_detail(result: CaseResult, limit: int) -> str:
    lines = [f"{result.rung} {result.name}: {result.status}"]
    if result.note:
        lines.append(f"  {result.note}")
    for f in result.files:
        fine_note = (f"  ({f.columns_compared - f.fine_failing}/{f.columns_compared} "
                    f"pass np.isclose)" if f.columns_compared else "")
        if f.status == "ok":
            lines.append(f"  {f.name}: ok ({f.columns_compared} columns){fine_note}")
            continue
        if f.status != "diff":
            lines.append(f"  {f.name}: {f.status} -- {f.note}")
            continue
        lines.append(f"  {f.name}: {f.failing}/{f.columns_compared} columns out{fine_note}")
        worst = sorted(f.columns, key=lambda c: -c.worst_rel)[:limit]
        for c in worst:
            lines.append(
                f"    [{c.index:>3}] {c.name:<34} "
                f"{c.bad_rows:>5} rows  "
                f"abs {c.worst_abs:.3e}  rel {c.worst_rel:.3e}")
            lines.append(
                f"          worst at row {c.worst_row + 1} ({c.worst_date}): "
                f"ref {c.ref_value:.8g}  got {c.got_value:.8g}")
        if f.failing > limit:
            lines.append(f"    ... {f.failing - limit} more columns")
    return "\n".join(lines)


def to_json(results: Sequence[CaseResult]) -> str:
    return json.dumps([
        {
            "rung": r.rung,
            "case": r.name,
            "status": r.status,
            "note": r.note,
            "failing_columns": r.failing_columns,
            "total_columns": r.total_columns,
            "fine_failing_columns": r.fine_failing_columns,
            "files": [
                {
                    "name": f.name,
                    "status": f.status,
                    "note": f.note,
                    "columns_compared": f.columns_compared,
                    "fine_failing": f.fine_failing,
                    "failing": [
                        {
                            "index": c.index,
                            "name": c.name,
                            "bad_rows": c.bad_rows,
                            "worst_abs": c.worst_abs,
                            "worst_rel": c.worst_rel,
                            "worst_row": c.worst_row + 1,
                            "worst_date": c.worst_date,
                            "reference": c.ref_value,
                            "got": c.got_value,
                        } for c in f.columns
                    ],
                } for f in r.files
            ],
        } for r in results
    ], indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Status of the Python port against the GEOtop 1D references.")
    p.add_argument("--tests-dir", default=DEFAULT_TESTS_DIR,
                   help=f"reference cases (default: {DEFAULT_TESTS_DIR})")
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR,
                   help=f"where model runs live (default: {DEFAULT_WORK_DIR})")
    p.add_argument("--inplace", action="store_true",
                   help="compare <case>/output-tabs in the reference tree "
                        "instead of the work dir (self-test against the C++ run)")
    p.add_argument("--case", action="append", metavar="NAME",
                   help="restrict to this case; repeatable")
    p.add_argument("--exclude", action="append", metavar="NAME", default=[],
                   help="skip this case; repeatable. Bro alone is 8.7 of the "
                        "11 minutes a full sweep takes, so `--exclude Bro` "
                        "buys most of the coverage for a quarter of the wait")
    p.add_argument("--run", action="store_true",
                   help="run the model into the work dir before comparing")
    p.add_argument("--cmd", default=os.environ.get("GEOTOP_PY_CMD", DEFAULT_CMD),
                   help=f"model command, given the case dir (default: {DEFAULT_CMD})")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-case run timeout in seconds (default: 1800)")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--strict-rel", action="store_true",
                   help="use min(|a|,|b|) as the relative-error denominator")
    p.add_argument("--fine-atol", type=float, default=1e-8,
                   help="absolute tolerance for the extra np.isclose column "
                        "(default: numpy's own, 1e-8)")
    p.add_argument("--fine-rtol", type=float, default=1e-5,
                   help="relative tolerance for the extra np.isclose column "
                        "(default: numpy's own, 1e-5)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="per-column breakdown of every non-green case")
    p.add_argument("--limit", type=int, default=8,
                   help="columns shown per table in verbose mode (default: 8)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if not os.path.isdir(args.tests_dir):
        print(f"reference cases not found: {args.tests_dir}", file=sys.stderr)
        return 2

    cases = discover_cases(args.tests_dir)
    known = {c[1] for c in cases}
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c[1] in wanted]
        if (unknown := wanted - known):
            print(f"unknown case(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
    if args.exclude:
        dropped = set(args.exclude)
        if (unknown := dropped - known):
            print(f"unknown case(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c[1] not in dropped]
    if not cases:
        print("no cases with reference outputs", file=sys.stderr)
        return 2

    results: list[CaseResult] = []
    for rung, name in cases:
        if args.inplace:
            got_dir = os.path.join(args.tests_dir, name, OUTPUT_DIR)
        else:
            got_dir = os.path.join(args.work_dir, name, OUTPUT_DIR)
            if args.run:
                err = run_case(name, args.tests_dir, args.work_dir,
                               args.cmd, args.timeout)
                if err:
                    results.append(CaseResult(rung, name, "error", err))
                    continue
        results.append(evaluate_case(rung, name, args.tests_dir, got_dir,
                                     args.atol, args.rtol, args.strict_rel,
                                     args.fine_atol, args.fine_rtol))

    if args.json:
        print(to_json(results))
    else:
        source = (f"{args.tests_dir}/<case>/{OUTPUT_DIR}" if args.inplace
                  else f"{args.work_dir}/<case>/{OUTPUT_DIR}")
        print(render(results, args.atol, args.rtol, source,
                     args.fine_atol, args.fine_rtol))
        if args.verbose:
            for r in results:
                if r.status != "ok":
                    print()
                    print(render_detail(r, args.limit))

    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
