"""Verify snow albedo and snow-age updates against the official C++ functions."""

import pytest

from geotop_py.energy import albedo

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


@pytest.mark.parametrize("cosinc", [0.0, 0.1, 0.3, 0.49, 0.5, 0.51, 0.8, 1.0])
def test_fzen_matches_the_cxx(cosinc):
    assert albedo.Fzen(cosinc) == pytest.approx(_cxx.Fzen(cosinc), abs=1.0e-12)


@pytest.mark.parametrize("wat", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4])
def test_find_albedo_matches_the_cxx(wat):
    py = albedo.find_albedo(0.16, 0.08, wat, 0.05, 0.4)
    cc = _cxx.find_albedo(0.16, 0.08, wat, 0.05, 0.4)
    assert py == pytest.approx(cc, abs=1.0e-12)


@pytest.mark.parametrize("snowD", [0.0, 1.0, 10.0, 25.0, 49.9, 50.0, 100.0])
@pytest.mark.parametrize("tsnow", [0.0, 1.0, 5.0, 50.0])
def test_snow_albedo_diffuse_matches_the_cxx(snowD, tsnow):
    py = albedo.snow_albedo(0.16, snowD, 50.0, 0.95, 0.35, tsnow, 0.0, albedo._zero)
    cc = _cxx.snow_albedo(0.16, snowD, 50.0, 0.95, 0.35, tsnow, 0.0, False)
    assert py == pytest.approx(cc, abs=1.0e-12)


@pytest.mark.parametrize("cosinc", [0.0, 0.2, 0.49, 0.5, 0.7, 1.0])
@pytest.mark.parametrize("snowD", [0.0, 10.0, 50.0, 100.0])
def test_snow_albedo_beam_matches_the_cxx(snowD, cosinc):
    py = albedo.snow_albedo(0.16, snowD, 50.0, 0.95, 0.35, 3.0, cosinc, albedo.Fzen)
    cc = _cxx.snow_albedo(0.16, snowD, 50.0, 0.95, 0.35, 3.0, cosinc, True)
    assert py == pytest.approx(cc, abs=1.0e-12)


@pytest.mark.parametrize("Psnow,Ts,Dt,Prestore,snowage", [
    (0.0, -5.0, 3600.0, 10.0, 0.0),
    (2.0, -3.0, 3600.0, 10.0, 5.0),
    (15.0, -1.0, 3600.0, 10.0, 5.0),   # Psnow > Prestore: age driven negative before the clamp
    (0.0, 0.0, 3600.0, 10.0, 100.0),   # Ts at freezing: r1 saturates
    (0.0, -20.0, 86400.0, 10.0, 0.0),  # a full day, cold
])
def test_update_snow_age_matches_the_cxx(Psnow, Ts, Dt, Prestore, snowage):
    py = albedo.update_snow_age(snowage, Psnow, Ts, Dt, Prestore)
    cc = _cxx.update_snow_age(Psnow, Ts, Dt, Prestore, snowage)
    assert py == pytest.approx(cc, abs=1.0e-9)
