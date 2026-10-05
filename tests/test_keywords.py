"""The generated keyword tables, pinned against the compiled model.

``src/geotop_py/keywords.py`` is generated from the GEOtop sources so the port
can run without the oracle. That convenience is only safe while the generated
copy still matches what the C++ actually holds -- including the *order*, since
the index of a name is the ``cod`` that ``assign_numeric_parameters`` uses.
"""

import pytest

from geotop_py.io import keywords
from tools.paths import GEOTOP_SRC

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


def test_numeric_table_matches_the_cxx_exactly():
    assert keywords.KEYWORDS_NUM == _cxx.KEYWORDS_NUM


def test_string_table_matches_the_cxx_exactly():
    assert keywords.KEYWORDS_CHAR == _cxx.KEYWORDS_CHAR


@pytest.mark.skipif(not GEOTOP_SRC.is_dir(),
                    reason="regenerating the table needs the GEOtop sources")
def test_the_generated_file_is_not_stale():
    """Regenerating must be a no-op, or the committed copy has drifted.

    Needs the C++ sources: unlike the two comparisons above, which the recorded
    oracle constants can serve, this one re-runs the scrape.
    """
    from tools.gen_keywords import main
    assert main(["--check"]) == 0


def test_defaults_cover_the_common_parameters():
    """A smoke check that the scrape found real values, not an empty table."""
    assert keywords.DEFAULTS["ThresTempRain"] == 3.0
    assert keywords.DEFAULTS["ThresTempSnow"] == -1.0
    assert keywords.DEFAULTS["AlphaSnow"] == 1.0e5
    assert len(keywords.DEFAULTS) > 150


def test_ambiguous_defaults_are_listed_not_flattened():
    """Parameters GEOtop defaults per-component must not appear as one scalar.

    ``DtPlotPoint`` defaults its first component to 0 and every later one to the
    previous component's value; collapsing that to a single number would look
    authoritative and be wrong.
    """
    assert "DtPlotPoint" in keywords.DEFAULTS_AMBIGUOUS
    assert "DtPlotPoint" not in keywords.DEFAULTS
    assert keywords.DEFAULTS_AMBIGUOUS["DtPlotPoint"]


def test_every_ambiguous_name_is_a_real_keyword():
    known = set(keywords.KEYWORDS_NUM)
    assert set(keywords.DEFAULTS).issubset(known)
    assert set(keywords.DEFAULTS_AMBIGUOUS).issubset(known)
