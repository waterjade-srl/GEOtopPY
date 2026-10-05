"""Tests for geotop_py.snow.strati.

Two kinds of check:

* **Invariants** -- merging and recombination must conserve total depth, total
  SWE and total internal energy. These are physical-correctness properties,
  provable without any oracle, and they hold for the whole composed operation,
  not just the underlying laws.

* **Laws-level pinning** -- running the same merge through GEOtop's compiled C++
  enthalpy laws (the oracle) must give the same column as running it through the
  Python laws. This ties the composition back to GEOtop even though GEOtop's
  ``snow_layer_combination`` itself is not in the oracle library.
"""

import pytest

from geotop_py import laws
from geotop_py.snow import strati
from geotop_py.snow.state import SnowColumn

try:
    from geotop_py import _cxx
    HAVE_ORACLE = True
except FileNotFoundError:
    HAVE_ORACLE = False

A = 1.0  # alpha_snow; conservation holds for any positive value


def cold_column(max=10):
    """A three-layer cold pack (no liquid), bottom-up: layer 1 is the bottom."""
    return SnowColumn.from_layers(
        [(200.0, 60.0, 0.0, -8.0),
         (150.0, 45.0, 0.0, -5.0),
         (100.0, 30.0, 2.0, -1.0)],
        max=max,
    )


# --- merging invariants --------------------------------------------------

def test_merge_conserves_enthalpy_mass_depth():
    col = cold_column()

    # merge layer 3 into layer 2 (result in slot 2), as merge_layers would
    strati.snowlayer_merging(col, A, 3, 2, 2)

    # slot 3 still holds its old values until the caller shifts/clears it;
    # compare the merged slot 2 against the sum of the originals.
    assert col.Dzl[2] == pytest.approx(150.0 + 100.0)
    merged_h = laws.internal_energy(col.w_ice[2], col.w_liq[2], col.T[2])
    orig_h = (laws.internal_energy(45.0, 0.0, -5.0)
              + laws.internal_energy(30.0, 2.0, -1.0))
    assert merged_h == pytest.approx(orig_h, rel=1e-9)
    # SWE of the merged layer equals the sum of the two originals' SWE
    assert col.w_ice[2] + col.w_liq[2] == pytest.approx(45.0 + 30.0 + 2.0)


def test_merge_layers_full_conserves_totals():
    col = cold_column()
    h0 = col.total_internal_energy()
    swe0 = col.swe()
    d0 = col.depth()

    strati.merge_layers(col, A, 2)  # merge middle layer into a neighbour

    assert col.lnum == 2
    assert col.total_internal_energy() == pytest.approx(h0, rel=1e-9)
    assert col.swe() == pytest.approx(swe0, rel=1e-12)
    assert col.depth() == pytest.approx(d0, rel=1e-12)


@pytest.mark.skipif(not HAVE_ORACLE, reason="oracle not built")
def test_merge_matches_cxx_oracle():
    """The Python merge and the C++-laws merge must land on the same state."""
    col_py = cold_column()
    col_cxx = cold_column()

    strati.merge_layers(col_py, A, 2)
    strati.merge_layers(
        col_cxx, A, 2,
        ienergy=_cxx.internal_energy,
        from_ienergy=_cxx.from_internal_energy,
    )

    for l in range(1, col_py.max + 1):
        assert col_py.w_ice[l] == pytest.approx(col_cxx.w_ice[l], rel=1e-9, abs=1e-9)
        assert col_py.w_liq[l] == pytest.approx(col_cxx.w_liq[l], rel=1e-9, abs=1e-9)
        assert col_py.T[l] == pytest.approx(col_cxx.T[l], rel=1e-9, abs=1e-9)
        assert col_py.Dzl[l] == pytest.approx(col_cxx.Dzl[l], rel=1e-9, abs=1e-9)


# --- combination driver --------------------------------------------------

def inf_of(max):
    """GEOtop's 1-based list of reference positions; index 0 unused."""
    return [0] + list(range(1, max + 1))


def test_combination_conserves_depth_and_swe():
    col = cold_column(max=12)
    d0 = col.depth()
    swe0 = col.swe()

    # SWEmax_layer small enough that the thick layers must split
    strati.snow_layer_combination(
        col, A, Ta=-5.0, inf=inf_of(12),
        SWEmax_layer=20.0, SWEmax_tot=1e12,
    )

    assert col.type == 2
    assert col.lnum > 3            # layers were split
    assert col.depth() == pytest.approx(d0, abs=1e-3)
    assert col.swe() == pytest.approx(swe0, abs=1e-3)


def test_combination_splits_thick_layer():
    # single layer far above 2*SWEmax_layer must be split into several
    col = SnowColumn.from_layers([(400.0, 120.0, 0.0, -6.0)], max=12)
    swe0 = col.swe()
    strati.snow_layer_combination(
        col, A, Ta=-6.0, inf=inf_of(12),
        SWEmax_layer=20.0, SWEmax_tot=1e12,
    )
    assert col.lnum >= 5
    assert col.swe() == pytest.approx(swe0, abs=1e-3)
    # every active layer at or below one SWEmax band (plus the split remainder)
    for l in range(1, col.lnum + 1):
        assert col.w_ice[l] <= 20.0 * 2.0 + 1e-6


def test_combination_merges_thin_layers():
    # a stack of very thin layers should collapse toward fewer, thicker ones
    col = SnowColumn.from_layers(
        [(10.0, 1.0, 0.0, -3.0)] * 6, max=12)
    swe0 = col.swe()
    d0 = col.depth()
    strati.snow_layer_combination(
        col, A, Ta=-3.0, inf=inf_of(12),
        SWEmax_layer=20.0, SWEmax_tot=1e12,
    )
    assert col.lnum < 6
    assert col.depth() == pytest.approx(d0, abs=1e-3)
    assert col.swe() == pytest.approx(swe0, abs=1e-3)


def test_combination_resets_negligible_pack():
    col = SnowColumn.from_layers([(0.01, 1e-8, 0.0, -1.0)], max=12)
    strati.snow_layer_combination(
        col, A, Ta=-1.0, inf=inf_of(12),
        SWEmax_layer=20.0, SWEmax_tot=1e12,
    )
    assert col.lnum == 0
    assert col.type == 0
    assert col.swe() == pytest.approx(0.0, abs=1e-9)


def test_combination_caps_total_swe():
    col = cold_column(max=12)         # total ice 135
    strati.snow_layer_combination(
        col, A, Ta=-5.0, inf=inf_of(12),
        SWEmax_layer=20.0, SWEmax_tot=100.0,   # cap below current total
    )
    assert col.swe() <= 100.0 + 1e-6
