"""Check snow-albedo laws and the initial snow age."""

import math

from geotop_py.energy import albedo


def _mean_surface_albedo(snowage, snowD=500.0):
    # Every parameter is passed explicitly, ``avo`` included: these are a
    # synthetic set chosen to make the arithmetic below round, not the
    # keyword defaults (the real ``FreshSnowReflVis`` default is 0.9 --
    # ``parameters.cc:1351``), and the test must not move when a default is
    # corrected.
    a = albedo.albedos(snowD, snowage, cosinc=1.0, theta_sup=0.3,
                        par=albedo.AlbedoParams(avo=0.95, airo=0.85,
                                                aging_vis=0.35,
                                                aging_nir=0.75, aep=50.0))
    avis_b, avis_d, anir_b, anir_d = a
    return 0.5 * (0.5 * avis_d + 0.5 * anir_d) + 0.5 * (0.5 * avis_b + 0.5 * anir_b)


def test_fresh_snow_albedo_high():
    # snowage 0 -> vis 0.95, nir 0.85 -> mean 0.90
    assert abs(_mean_surface_albedo(0.0) - 0.90) < 1e-9


def test_aging_lowers_albedo_monotonically():
    ages = [0.0, 0.5, 1.0, 3.0, 10.0]
    albs = [_mean_surface_albedo(a) for a in ages]
    assert all(albs[i] > albs[i + 1] for i in range(len(albs) - 1))


def test_shallow_snow_blends_to_ground():
    # a thin pack (< AEP) must sit below a thick pack of the same age
    assert _mean_surface_albedo(1.0, snowD=5.0) < _mean_surface_albedo(1.0, snowD=500.0)


def test_initial_snow_age_is_zero_by_default():
    assert albedo.initial_snow_age(0.0, True, -3.0) == 0.0
    assert albedo.initial_snow_age(0.0, False, -3.0) == 0.0


def test_initial_snow_age_follows_geotop():
    """InitSnowAge [days] goes to seconds only under an initial pack, and is
    non-dimensionalized at InitSnowTemp either way (input.cc:1733, 1799)."""
    days, T = 3.0, -5.0
    r1 = math.exp(5000.0 * (1.0 / 273.16 - 1.0 / (T + 273.16)))
    rate = (r1 + min(r1 ** 10, 1.0) + 0.3) * 1.0e-6
    assert albedo.initial_snow_age(days, True, T) == days * 86400.0 * rate
    assert albedo.initial_snow_age(days, False, T) == days * rate
