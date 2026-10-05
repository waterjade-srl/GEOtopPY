"""Fixtures shared by the test modules."""

import dataclasses
import glob
import hashlib
import os
import shutil

import pytest

from geotop_py import pipeline
from geotop_py.io import meteo, rastermap
from tools.paths import REFERENCE_1D


def _content(*paths):
    """The files' bytes, digested: a copy of a case reads from the cache too,
    and a file rewritten in place does not."""
    h = hashlib.sha256()
    for p in paths:
        try:
            with open(p, "rb") as fh:
                h.update(os.path.basename(p).encode() + b"\0" + fh.read())
        except OSError:
            continue
    return h.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def _cached_readers():
    """Parse each meteo file and raster map once per session.

    Parsing is the bulk of many tests' time (a meteo file goes through the
    digit-by-digit number parser, which is kept on purpose) and several tests
    read the same files. The key is what the files contain, not where they are,
    plus the arguments: the copy of a case a pipeline test runs from hits the
    same entry, and a test that rewrites a file reads it again. Every call gets
    its own copy, so no test sees another's edits.
    """
    real_load, real_read_map = meteo.load, rastermap.read_map
    tables, maps = {}, {}

    def load(path, col_names, options=None):
        key = (_content(path), tuple(col_names), repr(options))
        if key not in tables:
            tables[key] = real_load(path, col_names, options)
        return [list(row) for row in tables[key]]

    def read_map(stem, novalue, reference=None):
        ref = None if reference is None else (
            reference.nrows, reference.ncols, reference.dx, reference.dy,
            reference.X0, reference.Y0, reference.novalue)
        files = [stem] + sorted(glob.glob(glob.escape(stem) + ".*"))
        key = (_content(*files), novalue, ref)
        if key not in maps:
            maps[key] = real_read_map(stem, novalue, reference)
        m = maps[key]
        return dataclasses.replace(m, data=[list(row) for row in m.data])

    meteo.load, rastermap.read_map = load, read_map
    yield
    meteo.load, rastermap.read_map = real_load, real_read_map


@pytest.fixture(scope="session")
def reference_run(tmp_path_factory):
    """``reference_run(case) -> (sim_dir, recs)``: ``case`` copied from the
    reference tree and simulated once per session, outputs suffixed ``_py``.

    For tests that only read what the run produced. A test that edits the case
    or patches the model must make its own run.
    """
    runs = {}

    def run(case):
        if case not in runs:
            sim_dir = tmp_path_factory.mktemp("reference") / case
            shutil.copytree(
                os.path.join(REFERENCE_1D, case), sim_dir,
                ignore=shutil.ignore_patterns("output-tabs", "output-prof", "*.log"))
            recs = pipeline.run_simulation(str(sim_dir), suffix="_py", verbose=False)
            runs[case] = (sim_dir, recs)
        return runs[case]

    return run
