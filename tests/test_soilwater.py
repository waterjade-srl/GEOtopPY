"""Verify soil hydraulic laws directly against the pinned C++ oracle."""

import pytest

from geotop_py import constants as C
from geotop_py.water import soilwater

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


# ------------------------------------------------------------------ constants

def test_row_indices_match_the_compiled_model():
    """These are hand-written literals in geotop_py.constants (unlike
    io.soil.ROW, which derives its indices positionally from a name tuple and
    reads the count from the model) -- so unlike that module, nothing stops
    them from silently drifting except this pin."""
    names = ("jdz", "jpsi", "jT", "jKn", "jKl", "jres", "jwp", "jfc",
             "jsat", "ja", "jns", "jv", "jkt", "jct", "jss")
    for name in names:
        assert getattr(C, name) == _cxx.SOIL[name], name
    assert C.jss == _cxx.SOIL["nsoilprop"]


# -------------------------------------------------------------- scalar pins

VG = dict(i=0.0, s=0.5, r=0.05, a=0.004, n=1.3, m=1.0 - 1.0 / 1.3)
VG_K = dict(VG, v=0.5)


@pytest.mark.parametrize("psi", [-1.0e5, -1.0e4, -1000.0, -100.0, -10.0, -1.0, 0.0])
def test_k_hydr_soil_matches_the_cxx(psi):
    args = dict(ksat=1.0e-3, imp=7.0, T=5.0, ratio=1.0e-4, **VG_K)
    assert soilwater.k_hydr_soil(psi=psi, **args) == _cxx.k_hydr_soil(psi=psi, **args)


@pytest.mark.parametrize("T", [-10.0, -0.5, 0.0, 0.5, 10.0])
def test_k_hydr_soil_matches_the_cxx_across_the_freezing_branch(T):
    args = dict(psi=-500.0, ksat=1.0e-3, imp=7.0, ratio=1.0e-4, T=T, **VG_K)
    assert soilwater.k_hydr_soil(**args) == _cxx.k_hydr_soil(**args)


@pytest.mark.parametrize("i", [0.0, 0.05, 0.1, 0.2])
def test_k_hydr_soil_matches_the_cxx_with_ice(i):
    args = dict(psi=-200.0, ksat=1.0e-3, imp=7.0, T=-1.0, ratio=1.0e-4,
               s=VG["s"], r=VG["r"], a=VG["a"], n=VG["n"], m=VG["m"], v=VG_K["v"])
    assert soilwater.k_hydr_soil(i=i, **args) == _cxx.k_hydr_soil(i=i, **args)


def test_psi_saturation_matches_the_cxx():
    assert soilwater.psi_saturation(**VG) == _cxx.psi_saturation(**VG)


@pytest.mark.parametrize("i", [0.0, 0.1, 0.3, 0.45])
def test_psi_saturation_matches_the_cxx_with_ice(i):
    args = dict(VG)
    args["i"] = i
    assert soilwater.psi_saturation(**args) == _cxx.psi_saturation(**args)


@pytest.mark.parametrize("D1,D2,K1,K2", [
    (1.0, 3.0, 10.0, 1.0), (0.1, 2.0, 1e-3, 1e-1), (5.0, 5.0, 2.0, 2.0),
])
def test_harmonic_mean_matches_the_cxx(D1, D2, K1, K2):
    assert soilwater.Harmonic_Mean(D1, D2, K1, K2) == _cxx.Harmonic_Mean(D1, D2, K1, K2)


@pytest.mark.parametrize("D1,D2,K1,K2", [
    (1.0, 3.0, 10.0, 1.0), (0.1, 2.0, 1e-3, 1e-1), (5.0, 5.0, 2.0, 2.0),
])
def test_arithmetic_mean_matches_the_cxx(D1, D2, K1, K2):
    assert soilwater.Arithmetic_Mean(D1, D2, K1, K2) == _cxx.Arithmetic_Mean(D1, D2, K1, K2)


def test_arithmetic_mean_pins_the_crossed_weighting():
    """The same quirk pinned for _cxx.Arithmetic_Mean in test_oracle.py, now
    pinned for the Python translation too: (D1*K2+D2*K1)/(D1+D2), D1 with K2."""
    D1, D2, K1, K2 = 1.0, 3.0, 10.0, 1.0
    assert soilwater.Arithmetic_Mean(D1, D2, K1, K2) == pytest.approx(
        (D1 * K2 + D2 * K1) / (D1 + D2))
    assert soilwater.Arithmetic_Mean(D1, D2, K1, K2) != \
        soilwater.Arithmetic_Mean(D1, D2, K2, K1)


@pytest.mark.parametrize("a", [0, 1, -1, 2])
def test_mean_dispatch_matches_the_cxx(a):
    args = (0.5, 2.0, 1e-3, 1e-1)
    assert soilwater.Mean(a, *args) == _cxx.Mean(a, *args)


# ------------------------------------------------------------- pa-indexed pins

def _pa(nlayers=2, **columns):
    """A ``pa[row][layer]`` matrix, both axes 1-based (index 0 unused), in the
    same layout ``io.soil.SoilParameters.pa`` uses. ``columns[name]`` is a
    per-layer sequence, 0-indexed (layer 1 first)."""
    pa = [[0.0] * (nlayers + 1) for _ in range(C.jss + 1)]
    for name, values in columns.items():
        row = getattr(C, name)
        for layer, v in enumerate(values, start=1):
            pa[row][layer] = v
    return pa


def _strip(pa):
    """Convert a 1-based-both-axes pa matrix to the 0-based nested list
    _cxx._flatten_pa expects (its ``[0][0]`` is GEOtop's ``[1][1]``)."""
    return [row[1:] for row in pa[1:]]


TWO_LAYERS = dict(
    jsat=(0.5, 0.45), jres=(0.05, 0.08), ja=(0.004, 0.006),
    jns=(1.3, 1.5), jss=(1.0e-7, 1.0e-7),
    jKn=(1.0e-4, 5.0e-5), jKl=(1.0e-4, 5.0e-5), jv=(0.5, 0.5),
)


@pytest.mark.parametrize("layer,psi", [(1, -500.0), (1, -10.0), (2, -500.0), (2, 0.0)])
def test_theta_from_psi_matches_the_cxx(layer, psi):
    pa = _pa(**TWO_LAYERS)
    mine = soilwater.theta_from_psi(psi, 0.0, layer, pa, C.PsiMin)
    theirs = _cxx.theta_from_psi(psi, 0.0, layer, _strip(pa), C.PsiMin)
    assert mine == theirs


@pytest.mark.parametrize("layer", [1, 2])
def test_psi_from_theta_matches_the_cxx(layer):
    pa = _pa(**TWO_LAYERS)
    th = 0.3
    mine = soilwater.psi_from_theta(th, 0.0, layer, pa, C.PsiMin)
    theirs = _cxx.psi_from_theta(th, 0.0, layer, _strip(pa), C.PsiMin)
    assert mine == theirs


@pytest.mark.parametrize("layer,psi", [(1, -500.0), (2, -200.0)])
def test_dtheta_dpsi_from_psi_matches_the_cxx(layer, psi):
    pa = _pa(**TWO_LAYERS)
    mine = soilwater.dtheta_dpsi_from_psi(psi, 0.0, layer, pa, C.PsiMin)
    theirs = _cxx.dtheta_dpsi_from_psi(psi, 0.0, layer, _strip(pa), C.PsiMin)
    assert mine == theirs


@pytest.mark.parametrize("jK,layer,psi", [
    ("jKn", 1, -500.0), ("jKl", 1, -500.0), ("jKn", 2, -50.0),
])
def test_k_from_psi_matches_the_cxx(jK, layer, psi):
    pa = _pa(**TWO_LAYERS)
    row = getattr(C, jK)
    mine = soilwater.k_from_psi(row, psi, 0.0, 5.0, layer, pa, 7.0, 1.0e-4)
    theirs = _cxx.k_from_psi(row, psi, 0.0, 5.0, layer, _strip(pa), 7.0, 1.0e-4)
    assert mine == theirs


@pytest.mark.parametrize("layer,ice", [(1, 0.0), (1, 0.1), (2, 0.0)])
def test_psisat_from_matches_the_cxx(layer, ice):
    pa = _pa(**TWO_LAYERS)
    mine = soilwater.psisat_from(ice, layer, pa)
    theirs = _cxx.psisat_from(ice, layer, _strip(pa))
    assert mine == theirs


# -------------------------------------------------------------------- re-exports

def test_re_exported_retention_curve_is_the_pinned_laws_implementation():
    from geotop_py import laws
    assert soilwater.teta_psi is laws.teta_psi
    assert soilwater.psi_teta is laws.psi_teta
    assert soilwater.dteta_dpsi is laws.dteta_dpsi
