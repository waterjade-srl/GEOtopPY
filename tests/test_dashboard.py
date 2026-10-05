"""The acceptance comparator's own rules, on hand-made tables."""

from tools import dashboard

HEADER = "Date12[DDMMYYYYhhmm],Tair[C]\n"
ROWS = "01/01/2000 01:00,1.0\n01/01/2000 02:00,2.0\n"


def _case(tmp_path, got_files, extra_dirs=()):
    """A reference case with one table, and a model output directory holding
    ``got_files``; ``extra_dirs`` are further ``(dir, file)`` outputs."""
    tests_dir = tmp_path / "tests"
    ref = tests_dir / "Case" / dashboard.REFERENCE_DIR
    ref.mkdir(parents=True)
    (ref / "point0001.txt").write_text(HEADER + ROWS)
    got = tmp_path / "runs" / "Case" / dashboard.OUTPUT_DIR
    got.mkdir(parents=True)
    for fn in got_files:
        (got / fn).write_text(HEADER + ROWS)
    for d, fn in extra_dirs:
        (got.parent / d).mkdir(exist_ok=True)
        (got.parent / d / fn).write_text(HEADER + ROWS)
    return dashboard.evaluate_case("L0", "Case", str(tests_dir), str(got),
                                   1e-5, 1e-5, False)


def test_matching_output_is_green(tmp_path):
    assert _case(tmp_path, ["point0001.txt", ".placeholder"]).status == "ok"


def test_a_file_the_reference_lacks_fails_the_case(tmp_path):
    r = _case(tmp_path, ["point0001.txt", "soilTemp0001.txt"])
    assert r.status == "diff"
    assert "output-tabs/soilTemp0001.txt" in r.note


def test_output_in_another_directory_fails_the_case(tmp_path):
    r = _case(tmp_path, ["point0001.txt"],
              extra_dirs=[("output-prof", "snowDepth0001.txt")])
    assert r.status == "diff"
    assert [f.name for f in r.files if f.status == "extra"] == \
        ["output-prof/snowDepth0001.txt"]


def test_run_case_starts_from_a_fresh_copy(tmp_path):
    tests_dir = tmp_path / "tests"
    (tests_dir / "Case").mkdir(parents=True)
    (tests_dir / "Case" / "geotop.inpts").write_text("")
    stale = tmp_path / "runs" / "Case" / "output-prof"
    stale.mkdir(parents=True)
    (stale / "old.txt").write_text("")
    err = dashboard.run_case("Case", str(tests_dir), str(tmp_path / "runs"),
                             "true", timeout=10)
    assert err is None
    assert not stale.exists()
    assert (tmp_path / "runs" / "Case" / "geotop.inpts").exists()
