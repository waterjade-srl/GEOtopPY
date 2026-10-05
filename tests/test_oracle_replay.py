"""The recorded-oracle machinery, pinned.

A recording that silently stops matching is worse than no recording: the tests
that read it keep passing while comparing against nothing. What is checked here
is therefore the mechanism, not the physics -- that the key a call is stored
under is the key it is looked up under, that a value survives the round trip
bit for bit, and that a table from the wrong GEOtop revision is refused rather
than used.
"""

import gzip
import json
import math
import struct

import pytest

from tools.oracle_replay import (
    KEY_LIMIT,
    Missing,
    build_module,
    call_key,
    canonical,
    decode,
    encode,
    load,
)
from tools.paths import PROJECT_ROOT, UPSTREAM


def _bits(x):
    return struct.pack("<d", x)


# ------------------------------------------------------------ the shipped table

@pytest.fixture(scope="module")
def table():
    t = load()
    if t is None:
        pytest.skip("no recorded oracle table")
    return t


def test_the_table_was_recorded_against_the_pinned_revision(table):
    assert table["upstream"] == UPSTREAM["commit"]


def test_the_table_is_not_a_truncated_recording(table):
    """A run that died early would leave a small but well-formed table."""
    assert len(table["calls"]) > 2000
    assert len(table["constants"]) >= 9
    assert len(table["callables"]) >= 70


def test_every_entry_carries_a_digest_and_a_size(table):
    for key, entry in table["calls"].items():
        assert entry["d"] and entry["n"] >= 0, key


def test_the_oversized_answers_are_digest_only(table):
    """The whole-table and whole-raster answers, kept as hashes by design."""
    digest_only = [k for k, v in table["calls"].items() if "v" not in v]
    assert digest_only
    names = {k.split("\x00", 1)[0] for k in digest_only}
    assert names <= {"load_meteo", "read_map", "read_txt_matrix_2"}


# ------------------------------------------------------------------ refusal

def test_a_table_from_another_revision_is_refused(tmp_path):
    path = tmp_path / "other.json.gz"
    with gzip.open(path, "wb") as fh:
        fh.write(json.dumps({"upstream": "0" * 40, "constants": {},
                             "callables": [], "calls": {}}).encode())
    assert load(path) is None


def test_a_missing_table_is_not_an_error(tmp_path):
    assert load(tmp_path / "absent.json.gz") is None


# ------------------------------------------------------------- value round trip

@pytest.mark.parametrize("value", [
    0.0, -0.0, 1.0, -9999.0, 0.1, 1e-300, 1e300,
    math.pi, math.nextafter(math.pi, 4.0),
])
def test_a_float_survives_the_round_trip_bit_for_bit(value):
    verbatim, _digest, _n = encode(value)
    assert verbatim is not None
    assert _bits(decode(verbatim)) == _bits(value)


def test_a_tuple_does_not_come_back_as_a_list():
    """Several tests compare against a tuple; JSON would flatten the difference."""
    verbatim, _d, _n = encode((1.0, 2.0))
    restored = decode(verbatim)
    assert isinstance(restored, tuple)
    assert restored == (1.0, 2.0)


def test_a_nested_structure_survives():
    value = {"a": [1.0, (2.0, "x")], "b": None, "c": True}
    verbatim, _d, _n = encode(value)
    assert decode(verbatim) == value


def test_a_value_that_is_not_a_literal_is_kept_as_a_digest_only():
    """``nan`` has no literal form, so it must not be stored as a value."""
    verbatim, digest, _n = encode(float("nan"))
    assert verbatim is None
    assert digest


def test_an_oversized_value_is_kept_as_a_digest_only():
    verbatim, digest, size = encode([float(i) for i in range(100_000)])
    assert verbatim is None
    assert digest and size > 65536


# --------------------------------------------------------------------- the key

def test_the_key_folds_the_checkout_path_away():
    """A key must not carry the absolute path of whoever recorded it."""
    assert str(PROJECT_ROOT) not in canonical(f"('{PROJECT_ROOT}/tests/1D/Bro',)")


def test_the_key_folds_pytests_run_number_away(tmp_path):
    a = canonical("/tmp/pytest-of-someone/pytest-3/test_thing0/g.asc")
    b = canonical("/tmp/pytest-of-someone/pytest-91/test_thing0/g.asc")
    assert a == b
    assert "pytest-3" not in a


def test_the_key_folds_the_xdist_worker_away():
    """The same test lands on a different worker from run to run."""
    serial = canonical("/tmp/pytest-of-someone/pytest-3/test_thing0/g.asc")
    worker = canonical("/tmp/pytest-of-someone/pytest-7/popen-gw5/test_thing0/g.asc")
    assert worker == serial


def test_a_long_argument_list_is_keyed_by_its_hash():
    long_arg = [float(i) for i in range(KEY_LIMIT)]
    key = call_key("f", (long_arg,), {})
    assert "sha256:" in key
    assert len(key) < 120


def test_two_different_long_argument_lists_get_different_keys():
    a = call_key("f", ([float(i) for i in range(KEY_LIMIT)],), {})
    b = call_key("f", ([float(i) + 1 for i in range(KEY_LIMIT)],), {})
    assert a != b


def test_keyword_arguments_are_order_independent():
    assert call_key("f", (), {"a": 1, "b": 2}) == call_key("f", (), {"b": 2, "a": 1})


def test_the_function_name_is_part_of_the_key():
    assert call_key("f", (1.0,), {}) != call_key("g", (1.0,), {})


# ------------------------------------------------------------- the replay module

def test_the_replay_module_answers_a_recorded_call():
    entry_key = call_key("f", (2.0,), {})
    verbatim, digest, n = encode(7.5)
    module = build_module({
        "upstream": UPSTREAM["commit"], "recorded": "t",
        "constants": {"NMET": "20"}, "callables": ["f"],
        "calls": {entry_key: {"v": verbatim, "d": digest, "n": n}},
    })
    assert module.NMET == 20
    assert module.f(2.0) == 7.5


def test_an_unrecorded_call_raises_rather_than_inventing_an_answer():
    module = build_module({"upstream": UPSTREAM["commit"], "recorded": "t",
                           "constants": {}, "callables": ["f"], "calls": {}})
    with pytest.raises(Missing):
        module.f(1.0)


def test_a_callable_the_suite_never_exercised_still_exists():
    """Otherwise the caller gets AttributeError, which no guard turns into a skip."""
    module = build_module({"upstream": UPSTREAM["commit"], "recorded": "t",
                           "constants": {}, "callables": ["never_called"],
                           "calls": {}})
    with pytest.raises(Missing):
        module.never_called()


def test_on_missing_is_called_instead_of_raising():
    seen = []
    module = build_module({"upstream": UPSTREAM["commit"], "recorded": "t",
                           "constants": {}, "callables": ["f"], "calls": {}},
                          on_missing=seen.append)
    with pytest.raises(Missing):
        module.f(1.0)
    assert seen and "no recorded oracle answer" in seen[0]


def test_a_digest_only_entry_does_not_pretend_to_have_a_value():
    entry_key = call_key("big", (), {})
    module = build_module({"upstream": UPSTREAM["commit"], "recorded": "t",
                           "constants": {}, "callables": ["big"],
                           "calls": {entry_key: {"d": "ab", "n": 10 ** 7}}})
    with pytest.raises(Missing, match="digest"):
        module.big()


# ------------------------------------------------- the upstream citation index
#
# Kept here rather than in a file of its own: it is the same idea as the oracle
# recording -- something the C++ checkout would answer, recorded so it can be
# answered without one -- and it fails the same way, by going quietly stale.

def _citations():
    from tools.citations import find_compliant_citations, iter_python_files
    return [c for p in iter_python_files() for c in find_compliant_citations(p)]


@pytest.fixture(scope="module")
def index():
    from tools.citations import load_index
    idx = load_index()
    if idx is None:
        pytest.skip("no upstream index for the pinned revision")
    return idx


def test_the_index_was_built_against_the_pinned_revision(index):
    assert index["upstream"] == UPSTREAM["commit"]


def test_the_index_left_nothing_unresolved(index):
    """An address that did not resolve when the index was built is a citation
    pointing outside the pinned revision, which no later check can catch."""
    assert index["unresolved"] == []


def test_every_citation_in_the_source_is_in_the_index(index):
    """The check that runs without a C++ checkout, so it runs in CI.

    It does not prove a citation is right -- only that it was verified against
    the pinned revision at some point and has not been edited since without
    being re-verified.
    """
    from tools.citations import check_against_index

    assert check_against_index(_citations(), index) == []


def test_a_citation_absent_from_the_index_is_reported(index):
    """Guard against the check silently accepting anything."""
    from tools.citations import Citation, check_against_index

    invented = Citation(PROJECT_ROOT / "src" / "geotop_py" / "x.py", 1,
                        "src/geotop/nowhere.cc", 1, 2)
    errors = check_against_index([invented], index)
    assert len(errors) == 1
    assert "not in the index" in errors[0]


def test_the_digest_ignores_indentation_but_not_the_code():
    """A reindent upstream must not read as a citation having moved; a changed
    token must."""
    from tools.citations import digest_lines

    assert digest_lines(["  a = 1", "  b = 2"]) == digest_lines(["a = 1", "\tb = 2"])
    assert digest_lines(["a = 1"]) != digest_lines(["a = 2"])
