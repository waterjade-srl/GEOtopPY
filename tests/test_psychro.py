"""Regression checks for psychro."""

import random

import pytest

from geotop_py import psychro

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


random.seed(0)
ELEVATIONS = [-100.0, 0.0, 500.0, 1500.0, 3000.0, 4500.0]
TEMPS = [-30.0, -10.0, -1.0, 0.0, 5.0, 15.0, 30.0]
LAPSE_RATES = [-8.0, -6.5, -2.5, -0.5, 0.0, 5.5]


@pytest.mark.parametrize("Z", ELEVATIONS)
def test_pressure_matches_the_cxx(Z):
    assert psychro.pressure(Z) == _cxx.pressure(Z)


@pytest.mark.parametrize("Z,Z0,T0,gamma", [
    (Z, Z0, T0, g)
    for Z in ELEVATIONS for Z0 in (0.0, 500.0) for T0 in (0.0, 10.0)
    for g in LAPSE_RATES
])
def test_temperature_matches_the_cxx(Z, Z0, T0, gamma):
    assert psychro.temperature(Z, Z0, T0, gamma) == _cxx.temperature(Z, Z0, T0, gamma)


def test_temperature_is_not_a_no_op_at_realistic_lapse_rates():
    """Regression guard: the missing 1e-3 once made a 1 km rise at -6.5 degC/km
    cool the air by 6500 degC instead of 6.5."""
    t = psychro.temperature(1500.0, 500.0, 10.0, -6.5)
    assert t == pytest.approx(16.5)


@pytest.mark.parametrize("T,Train,Tsnow", [
    (T, 3.0, -1.0) for T in (-5.0, -1.0, -1e-9, 0.0, 1.0, 2.0, 3.0, 3.0 + 1e-9, 8.0)
])
def test_part_snow_matches_the_cxx(T, Train, Tsnow):
    rain, snow = psychro.part_snow(12.5, T, Train, Tsnow)
    crain, csnow = _cxx.part_snow(12.5, T, Train, Tsnow)
    assert (rain, snow) == (crain, csnow)


def test_part_snow_conserves_total_precipitation():
    for T in (-10.0, -1.0, 0.5, 1.5, 3.0, 10.0):
        rain, snow = psychro.part_snow(7.0, T, 3.0, -1.0)
        assert rain + snow == pytest.approx(7.0)


@pytest.mark.parametrize("T,P", [(T, P) for T in TEMPS for P in (900.0, 1013.25, 1050.0)])
def test_sat_vap_pressure_matches_the_cxx(T, P):
    assert psychro.SatVapPressure(T, P) == pytest.approx(_cxx.SatVapPressure(T, P), rel=0, abs=1e-12)


@pytest.mark.parametrize("T,P", [(T, P) for T in TEMPS for P in (900.0, 1013.25, 1050.0)])
def test_sat_vap_pressure_2_matches_the_cxx(T, P):
    e, de_dT = psychro.SatVapPressure_2(T, P)
    ce, cde_dT = _cxx.SatVapPressure_2(T, P)
    assert e == pytest.approx(ce, rel=0, abs=1e-12)
    assert de_dT == pytest.approx(cde_dT, rel=0, abs=1e-9)


@pytest.mark.parametrize("T,P", [(T, P) for T in TEMPS for P in (900.0, 1013.25, 1050.0)])
def test_sat_vap_pressure_round_trips_through_the_cxx_inverse(T, P):
    e = psychro.SatVapPressure(T, P)
    assert psychro.TfromSatVapPressure(e, P) == pytest.approx(
        _cxx.TfromSatVapPressure(e, P), rel=0, abs=1e-12)


@pytest.mark.parametrize("e,P", [(e, 1013.25) for e in (1.0, 5.0, 10.0, 20.0)])
def test_spec_humidity_matches_the_cxx(e, P):
    assert psychro.SpecHumidity(e, P) == _cxx.SpecHumidity(e, P)


@pytest.mark.parametrize("RH,T,P", [
    (rh, T, 1013.25) for rh in (0.1, 0.5, 0.9, 1.0) for T in TEMPS
])
def test_spec_humidity_2_matches_the_cxx(RH, T, P):
    Q, dQ_dT = psychro.SpecHumidity_2(RH, T, P)
    cQ, cdQ_dT = _cxx.SpecHumidity_2(RH, T, P)
    assert Q == pytest.approx(cQ, rel=0, abs=1e-12)
    assert dQ_dT == pytest.approx(cdQ_dT, rel=0, abs=1e-12)


@pytest.mark.parametrize("Q,P", [(q, 1013.25) for q in (0.001, 0.005, 0.01, 0.02)])
def test_vap_pressure_from_spec_humidity_matches_the_cxx(Q, P):
    assert psychro.VapPressurefromSpecHumidity(Q, P) == \
        _cxx.VapPressurefromSpecHumidity(Q, P)


@pytest.mark.parametrize("T,RH,Z", [
    (T, rh, Z) for T in TEMPS for rh in (0.2, 0.5, 0.8, 1.0) for Z in (0.0, 1500.0)
])
def test_tdew_matches_the_cxx(T, RH, Z):
    assert psychro.Tdew(T, RH, Z) == pytest.approx(_cxx.Tdew(T, RH, Z), rel=0, abs=1e-12)


@pytest.mark.parametrize("T,Td,Z", [
    (T, T - 5.0, Z) for T in TEMPS for Z in (0.0, 1500.0)
])
def test_rhfrom_tdew_matches_the_cxx(T, Td, Z):
    assert psychro.RHfromTdew(T, Td, Z) == pytest.approx(
        _cxx.RHfromTdew(T, Td, Z), rel=0, abs=1e-12)


@pytest.mark.parametrize("T,Q,P", [
    (T, q, 1013.25) for T in TEMPS for q in (0.001, 0.005, 0.01)
])
def test_air_density_matches_the_cxx(T, Q, P):
    assert psychro.air_density(T, Q, P) == pytest.approx(
        _cxx.air_density(T, Q, P), rel=0, abs=1e-12)


@pytest.mark.parametrize("T", TEMPS)
def test_air_cp_matches_the_cxx(T):
    assert psychro.air_cp(T) == pytest.approx(_cxx.air_cp(T), rel=0, abs=1e-12)


def test_tdew_and_rhfrom_tdew_are_consistent_round_trip():
    """Property check (not a substitute for the pins above): the two should
    invert each other to numerical precision at a fixed elevation."""
    for T in TEMPS:
        for rh in (0.3, 0.6, 0.9):
            Td = psychro.Tdew(T, rh, 500.0)
            rh2 = psychro.RHfromTdew(T, Td, 500.0)
            assert rh2 == pytest.approx(rh, rel=1e-9)
