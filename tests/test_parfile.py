"""Regression checks for parfile."""

import os

import pytest

from geotop_py.io import parfile
from tools.paths import REFERENCE_1D

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


def _cases():
    if not os.path.isdir(REFERENCE_1D):
        return []
    return sorted(
        name for name in os.listdir(REFERENCE_1D)
        if os.path.exists(os.path.join(REFERENCE_1D, name, "geotop.inpts"))
    )


CASES = _cases()
needs_reference = pytest.mark.skipif(
    not CASES, reason=f"reference cases not found under {REFERENCE_1D}")


# ------------------------------------------------------------- find_number

@pytest.mark.parametrize("text", [
    "0", "1", "-1", "3600", "0.5", "-150.27", "0.0595", "1.E-3", "1.e-3",
    "1E5", "-2.5e+3", "0.15", "0.6", "0.65", "010119000000.", "46.25",
    "0.08563509608926037",
])
def test_find_number_matches_the_cxx(text):
    got = parfile.find_number([ord(ch) for ch in text])
    assert got == _cxx.find_number(text), (text, got, _cxx.find_number(text))


def test_find_number_is_not_float():
    """The point of translating it: on real inputs the two differ by an ULP.

    If this ever stops being true the translation is still correct, but the
    reason for it has gone -- so the assertion is on the C++, not on us.
    """
    one_ulp_off = [t for t in ("0.15", "0.6", "0.65", "0.313", "-150.27")
                   if _cxx.find_number(t) != float(t)]
    assert one_ulp_off, "find_number now agrees with float() on every sample"
    for text in one_ulp_off:
        assert parfile.find_number([ord(c) for c in text]) == _cxx.find_number(text)


def test_find_number_handles_an_empty_token():
    assert parfile.find_number([]) == 0.0


# ------------------------------------------------------- whole-file parsing

@needs_reference
@pytest.mark.parametrize("case", CASES)
def test_numeric_table_matches_the_cxx(case):
    path = os.path.join(REFERENCE_1D, case, "geotop.inpts")
    mine = parfile.parse(path, _cxx.KEYWORDS_NUM, _cxx.KEYWORDS_CHAR)
    theirs, _ = _cxx.parse_inpts(path)

    assert set(mine.numeric) == set(theirs)
    for name, values in theirs.items():
        assert mine.numeric[name] == values, name


@needs_reference
@pytest.mark.parametrize("case", CASES)
def test_string_table_matches_the_cxx(case):
    path = os.path.join(REFERENCE_1D, case, "geotop.inpts")
    mine = parfile.parse(path, _cxx.KEYWORDS_NUM, _cxx.KEYWORDS_CHAR)
    _, theirs = _cxx.parse_inpts(path)

    assert set(mine.strings) == set(theirs)
    for name, value in theirs.items():
        assert mine.strings[name] == value, name


@needs_reference
def test_the_corpus_actually_exercises_the_parser():
    """Guard against the comparison passing because nothing was parsed."""
    path = os.path.join(REFERENCE_1D, "PureDrainage", "geotop.inpts")
    pf = parfile.parse(path, _cxx.KEYWORDS_NUM, _cxx.KEYWORDS_CHAR)
    declared = sum(1 for v in pf.numeric.values() if v != [parfile.NUMBER_NOVALUE])
    named = sum(1 for v in pf.strings.values() if v != parfile.STRING_NOVALUE)
    assert declared > 30 and named > 25


# ---------------------------------------------------- assignation semantics

@needs_reference
def test_number_falls_back_to_the_default_when_absent():
    path = os.path.join(REFERENCE_1D, "PureDrainage", "geotop.inpts")
    pf = parfile.parse(path, _cxx.KEYWORDS_NUM, _cxx.KEYWORDS_CHAR)

    assert pf.number("TimeStepEnergyAndWater", 0, 900.0) == 3600.0   # declared
    assert not pf.has_number("MaxSnowLayersMiddle")
    assert pf.number("MaxSnowLayersMiddle", 0, 5.0) == 5.0           # defaulted
    # a component past the declared ones also falls back
    assert pf.number("TimeStepEnergyAndWater", 7, 42.0) == 42.0


@needs_reference
def test_string_falls_back_to_novalue():
    path = os.path.join(REFERENCE_1D, "PureDrainage", "geotop.inpts")
    pf = parfile.parse(path, _cxx.KEYWORDS_NUM, _cxx.KEYWORDS_CHAR)
    assert pf.string("MeteoFile") == "meteo/meteo"
    assert not pf.has_string("DemFile")
    assert pf.string("DemFile") == parfile.STRING_NOVALUE
    assert pf.string("DemFile", "fallback.asc") == "fallback.asc"


@needs_reference
def test_keyword_matching_is_case_insensitive(tmp_path):
    src = os.path.join(REFERENCE_1D, "PureDrainage", "geotop.inpts")
    text = open(src, errors="replace").read()
    shouted = text.replace("TimeStepEnergyAndWater", "TIMESTEPENERGYANDWATER")
    target = tmp_path / "geotop.inpts"
    target.write_text(shouted)

    pf = parfile.parse(str(target), _cxx.KEYWORDS_NUM, _cxx.KEYWORDS_CHAR)
    assert pf.number("TimeStepEnergyAndWater") == 3600.0
