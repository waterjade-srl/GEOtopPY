"""Harness for the GEOtop v3.0 oracle boundary.

These tests do not compare Python against C++ -- that is what test_laws.py does.
They check the *boundary itself*: that the ctypes wrappers pass arguments in the
order and indexing the C++ expects, and that array-taking functions are wired to
the right elements. Two wrapper bugs were caught this way during wrapper verification
(``from_internal_energy`` is in/out, and ``tridiag2`` indexes both off-diagonals
by the lower index of the pair), and both would have been invisible in a test
that only compared two implementations of the same mistake.

The method is invariants and independent solves, never transcribed constants.
"""

import math

import pytest

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


VG = dict(i=0.0, s=0.5, r=0.05, a=0.004, n=1.3, m=1.0 - 1.0 / 1.3,
          pmin=-1.0e6, Ss=1.0e-7)


def pa_matrix(**overrides):
    """A one-layer soil parameter matrix, filled by GEOtop's own row indices."""
    pa = [[0.0] for _ in range(_cxx.SOIL["nsoilprop"])]
    for name, value in overrides.items():
        pa[_cxx.SOIL[name] - 1][0] = value
    return pa


# ------------------------------------------------------------------- indices

def test_soil_indices_are_contiguous_and_from_the_cxx():
    soil = _cxx.SOIL
    assert soil["jdz"] == 1
    ordered = ["jdz", "jpsi", "jT", "jKn", "jKl", "jres", "jwp", "jfc",
               "jsat", "ja", "jns", "jv", "jkt", "jct", "jss"]
    assert [soil[k] for k in ordered] == list(range(1, len(ordered) + 1))
    assert soil["nsoilprop"] == soil["jss"]


# ------------------------------------------------ van Genuchten, pedo.funct.h

@pytest.mark.parametrize("psi", [-1.0e5, -1.0e4, -1000.0, -100.0, -10.0, -1.0])
def test_teta_psi_and_psi_teta_are_inverses(psi):
    theta = _cxx.teta_psi(psi, **VG)
    assert _cxx.psi_teta(theta, **VG) == pytest.approx(psi, rel=1e-9)


@pytest.mark.parametrize("psi", [-1.0e4, -1000.0, -100.0, -10.0])
def test_dteta_dpsi_matches_a_finite_difference(psi):
    h = abs(psi) * 1e-6
    numeric = (_cxx.teta_psi(psi + h, **VG) - _cxx.teta_psi(psi - h, **VG)) / (2 * h)
    assert _cxx.dteta_dpsi(psi, **VG) == pytest.approx(numeric, rel=1e-5)


def test_teta_psi_is_bounded_by_residual_and_saturation():
    for psi in (-1.0e6, -1.0e3, -1.0, 0.0):
        theta = _cxx.teta_psi(psi, **VG)
        assert VG["r"] <= theta <= VG["s"] + 1e-12


def test_psi_saturation_is_where_ice_free_theta_reaches_saturation():
    psisat = _cxx.psi_saturation(VG["i"], VG["s"], VG["r"], VG["a"], VG["n"],
                                 VG["m"])
    assert _cxx.teta_psi(psisat, **VG) == pytest.approx(VG["s"], rel=1e-9)


def test_k_hydr_soil_is_monotone_and_capped_by_ksat():
    ksat = 1.0e-3
    args = dict(ksat=ksat, imp=7.0, i=0.0, s=0.5, r=0.05, a=0.004, n=1.3,
                m=1.0 - 1.0 / 1.3, v=0.5, T=5.0, ratio=1.0e-4)
    values = [_cxx.k_hydr_soil(psi=p, **args)
              for p in (-1.0e5, -1.0e4, -1000.0, -100.0, -1.0, 0.0)]
    assert values == sorted(values)
    assert all(0.0 <= v <= ksat * (1.0 + 1e-12) for v in values)


def test_ice_reduces_hydraulic_conductivity():
    args = dict(psi=-100.0, ksat=1.0e-3, imp=7.0, s=0.5, r=0.05, a=0.004,
                n=1.3, m=1.0 - 1.0 / 1.3, v=0.5, T=-1.0, ratio=1.0e-4)
    assert _cxx.k_hydr_soil(i=0.2, **args) < _cxx.k_hydr_soil(i=0.0, **args)


def test_Psif_is_zero_above_freezing_and_negative_below():
    assert _cxx.Psif(5.0) == 0.0
    assert _cxx.Psif(0.0) == 0.0
    assert _cxx.Psif(-1.0) < 0.0
    assert _cxx.Psif(-10.0) < _cxx.Psif(-1.0)


# ------------------------------------------------------------ interface means

def test_Arithmetic_Mean_weights_each_K_by_the_OTHER_layer_thickness():
    """``(D1*K2 + D2*K1) / (D1 + D2)`` -- the weighting is crossed.

    This is the interface conductivity of two layers in series, not the
    arithmetic mean of K weighted by D, so the usual "harmonic <= arithmetic"
    inequality does not hold and must not be assumed. Reimplementing it the
    intuitive way (D1 with K1) is a silent error: it only shows up when the two
    layers differ in both thickness and conductivity.
    """
    D1, D2, K1, K2 = 1.0, 3.0, 10.0, 1.0
    assert _cxx.Arithmetic_Mean(D1, D2, K1, K2) == pytest.approx(
        (D1 * K2 + D2 * K1) / (D1 + D2))
    # equal thicknesses hide the crossing; unequal ones expose it
    assert _cxx.Arithmetic_Mean(1.0, 1.0, K1, K2) == \
        _cxx.Arithmetic_Mean(1.0, 1.0, K2, K1)
    assert _cxx.Arithmetic_Mean(D1, D2, K1, K2) != \
        _cxx.Arithmetic_Mean(D1, D2, K2, K1)


def test_Harmonic_Mean_is_the_series_resistance_form():
    D1, D2, K1, K2 = 0.1, 2.0, 1e-3, 1e-1
    assert _cxx.Harmonic_Mean(D1, D2, K1, K2) == pytest.approx(
        (D1 + D2) / (D1 / K1 + D2 / K2))


def test_Mean_dispatches_on_its_flag():
    args = (0.5, 2.0, 1e-3, 1e-1)
    assert _cxx.Mean(0, *args) == _cxx.Harmonic_Mean(*args)
    assert _cxx.Mean(1, *args) == _cxx.Arithmetic_Mean(*args)


# --------------------------------------------- the pa-indexed pedo.funct wrappers

def test_theta_from_psi_agrees_with_the_scalar_closure():
    """theta_from_psi reads the van Genuchten parameters out of `pa`.

    If a row index were off, the result would silently use the wrong parameter,
    so this pins the whole indexing path against the scalar form.
    """
    pa = pa_matrix(jres=VG["r"], jsat=VG["s"], ja=VG["a"], jns=VG["n"],
                   jss=VG["Ss"])
    for psi in (-1.0e4, -1000.0, -10.0):
        assert _cxx.theta_from_psi(psi, 0.0, 1, pa, VG["pmin"]) == \
            pytest.approx(_cxx.teta_psi(psi, **VG), rel=1e-12)


def test_psi_from_theta_inverts_theta_from_psi():
    pa = pa_matrix(jres=VG["r"], jsat=VG["s"], ja=VG["a"], jns=VG["n"],
                   jss=VG["Ss"])
    for psi in (-1.0e4, -1000.0, -10.0):
        theta = _cxx.theta_from_psi(psi, 0.0, 1, pa, VG["pmin"])
        assert _cxx.psi_from_theta(theta, 0.0, 1, pa, VG["pmin"]) == \
            pytest.approx(psi, rel=1e-8)


def test_psisat_from_agrees_with_psi_saturation():
    pa = pa_matrix(jres=VG["r"], jsat=VG["s"], ja=VG["a"], jns=VG["n"])
    assert _cxx.psisat_from(0.0, 1, pa) == pytest.approx(
        _cxx.psi_saturation(0.0, VG["s"], VG["r"], VG["a"], VG["n"], VG["m"]),
        rel=1e-12)


# ------------------------------------------------------------- util_math.h

@pytest.mark.parametrize("n", [2, 3, 5, 12])
def test_tridiag2_matches_an_independent_dense_solve(n):
    """A(ld, d, ud) * e + b = 0, checked against a general LU.

    Both off-diagonals are indexed by the lower index of the pair, which is the
    convention the wrapper documents; a mis-indexed wrapper fails here.
    """
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(seed=n)
    d = list(3.0 + rng.random(n))            # diagonally dominant, so solvable
    ld = list(-rng.random(n))
    ud = list(-rng.random(n))
    b = list(rng.standard_normal(n))

    status, e = _cxx.tridiag2(1, n, ld, d, ud, b)
    assert status == 0

    A = numpy.zeros((n, n))
    for i in range(n):
        A[i, i] = d[i]
    for i in range(n - 1):
        A[i + 1, i] = ld[i]
        A[i, i + 1] = ud[i]
    expected = numpy.linalg.solve(A, -numpy.array(b))
    assert numpy.allclose(e, expected, rtol=1e-12, atol=1e-12)


def test_tridiag2_reports_a_singular_pivot():
    status, _ = _cxx.tridiag2(1, 3, [0.0, 0.0, 0.0], [0.0, 1.0, 1.0],
                              [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert status == 1


def test_norms_use_an_inclusive_upper_bound():
    """v3.0's norms run ``for l = nbeg; l <= nend``.

    Not a detail: SolvePointEnergyBalance takes its Newton residual as
    ``norm_2(Fenergy, sur, n)`` (energy.balance.cc:1650), so the bound decides
    whether the last node counts towards convergence.
    """
    v = [1.0, -7.0, 3.0, 100.0]
    assert _cxx.norm_inf(v, 1, 3) == 7.0            # 100.0 excluded by nend=3
    assert _cxx.norm_inf(v, 1, 4) == 100.0          # included when nend=4
    assert _cxx.norm_1(v, 1, 3) == 11.0
    assert _cxx.norm_2(v, 1, 3) == pytest.approx(math.sqrt(1 + 49 + 9))


def test_Cramer_rule_solves_a_2x2_system():
    # 2x + 3y = 8 ; x - y = -1  ->  x = 1, y = 2
    x, y = _cxx.Cramer_rule(2.0, 3.0, 8.0, 1.0, -1.0, -1.0)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(2.0)


def test_minimize_merit_function_returns_a_damping_factor():
    lam = _cxx.minimize_merit_function(1.0, 1.0, 0.8, 0.5, 0.9)
    assert 0.0 < lam <= 1.0


# ------------------------------------------------------------------- snow.cc

def test_enthalpy_round_trip_conserves_swe():
    w_ice, w_liq = 40.0, 5.0
    h = _cxx.internal_energy(w_ice, w_liq, -3.0)
    new_ice, new_liq, T = _cxx.from_internal_energy(1.0e5, h, w_ice, w_liq)
    assert new_ice + new_liq == pytest.approx(w_ice + w_liq, rel=1e-12)
    assert _cxx.internal_energy(new_ice, new_liq, T) == pytest.approx(h, rel=1e-9)


def test_from_internal_energy_needs_the_current_contents():
    """SWE is formed from the w_ice/w_liq the caller passes in.

    With zeros it takes the empty-layer branch, which is what made the first
    version of the wrapper look like it worked while returning nothing.
    """
    h = _cxx.internal_energy(40.0, 5.0, -3.0)
    assert _cxx.from_internal_energy(1.0e5, h, 0.0, 0.0) == (0.0, 0.0, 0.0)
    assert _cxx.from_internal_energy(1.0e5, h, 40.0, 5.0) != (0.0, 0.0, 0.0)


def test_snow_conductivity_laws_increase_with_density():
    for law in (_cxx.k_thermal_snow_Sturm, _cxx.k_thermal_snow_Yen):
        values = [law(rho) for rho in (100.0, 200.0, 300.0, 400.0)]
        assert values == sorted(values)


# ---------------------------------------------------------- energy.balance.cc

def test_calc_C_soil_branch_adds_the_solid_term():
    """calc_C takes the soil branch when l > nsng, adding ct*(1 - sat)."""
    ct, sat = 2.3e6, 0.5
    snow = _cxx.C_snow(40.0, 5.0, 0.1, 1.0e5, 0.2)
    soil = _cxx.C_soil(ct, sat, 40.0, 5.0, 0.1, 1.0e5, 0.2)
    assert soil - snow == pytest.approx(ct * (1.0 - sat), rel=1e-12)


def test_calc_C_indexes_the_node_it_is_asked_for():
    D = [0.2, 0.4, 0.6]
    wi = [10.0, 20.0, 30.0]
    wl = [1.0, 2.0, 3.0]
    dw = [0.0, 0.0, 0.0]
    pa = pa_matrix()
    for node in (1, 2, 3):
        assert _cxx.calc_C(node, 3, 0.0, wi, wl, dw, D, pa) == pytest.approx(
            _cxx.C_snow(wi[node - 1], wl[node - 1], 0.0, 0.0, D[node - 1]),
            rel=1e-12)


# ------------------------------------------------------------------ meteo.cc

def test_part_snow_partitions_the_total():
    for T in (-5.0, -1.0, 0.0, 1.0, 3.0, 10.0):
        rain, snow = _cxx.part_snow(5.0, T, t_rain=3.0, t_snow=-1.0)
        assert rain + snow == pytest.approx(5.0, rel=1e-12)
        assert rain >= 0.0 and snow >= 0.0
    assert _cxx.part_snow(5.0, -10.0, 3.0, -1.0)[0] == 0.0     # all snow
    assert _cxx.part_snow(5.0, 10.0, 3.0, -1.0)[1] == 0.0      # all rain


def test_SatVapPressure_2_derivative_matches_a_finite_difference():
    T, P, h = 10.0, 1000.0, 1e-4
    _, de_dT = _cxx.SatVapPressure_2(T, P)
    numeric = (_cxx.SatVapPressure(T + h, P) - _cxx.SatVapPressure(T - h, P)) / (2 * h)
    assert de_dT == pytest.approx(numeric, rel=1e-6)


def test_TfromSatVapPressure_inverts_SatVapPressure():
    P = 1000.0
    for T in (-10.0, 0.0, 15.0, 30.0):
        e = _cxx.SatVapPressure(T, P)
        assert _cxx.TfromSatVapPressure(e, P) == pytest.approx(T, rel=1e-6)


def test_Tdew_and_RHfromTdew_round_trip():
    T, Z = 12.0, 500.0
    for RH in (0.3, 0.6, 0.95):
        td = _cxx.Tdew(T, RH, Z)
        assert _cxx.RHfromTdew(T, td, Z) == pytest.approx(RH, rel=1e-6)


def test_SpecHumidity_2_returns_the_SATURATION_derivative():
    """``dQ_dT`` is not the derivative of the ``Q`` the same call returns.

    ``Q = SpecHumidity(RH*e, P)``, but ``dQ_de`` is evaluated at ``e`` and
    carries no RH factor (meteo.cc:216-222). So the derivative is the saturation
    one, independent of RH, and only agrees with a finite difference of Q when
    RH == 1. GEOtop uses it for a saturated surface, where that is what it
    wants -- but a Python reimplementation that "fixed" it would change the
    surface energy balance.
    """
    T, P, h = 10.0, 1000.0, 1e-4

    # independent of RH
    _, at_06 = _cxx.SpecHumidity_2(0.6, T, P)
    _, at_10 = _cxx.SpecHumidity_2(1.0, T, P)
    assert at_06 == at_10

    # equal to a finite difference of Q only at saturation
    up, _ = _cxx.SpecHumidity_2(1.0, T + h, P)
    dn, _ = _cxx.SpecHumidity_2(1.0, T - h, P)
    assert at_10 == pytest.approx((up - dn) / (2 * h), rel=1e-5)

    up6, _ = _cxx.SpecHumidity_2(0.6, T + h, P)
    dn6, _ = _cxx.SpecHumidity_2(0.6, T - h, P)
    assert at_06 != pytest.approx((up6 - dn6) / (2 * h), rel=1e-3)


def test_pressure_decreases_with_elevation():
    assert _cxx.pressure(2000.0) < _cxx.pressure(0.0)


# -------------------------------------------------------------- turbulence.cc

def test_stability_functions_vanish_in_neutral_conditions():
    assert _cxx.Psim(0.0) == pytest.approx(0.0, abs=1e-12)
    assert _cxx.Psih(0.0) == pytest.approx(0.0, abs=1e-12)


def test_Levap_decreases_with_temperature():
    assert _cxx.Levap(20.0) < _cxx.Levap(0.0)
