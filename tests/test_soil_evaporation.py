import pytest

from geotop_py.energy.column import SoilLayer
from geotop_py.energy.turbulence import air_density, soil_evaporation_parameters


def _soil(theta=0.30):
    layer = SoilLayer(sat=0.40, res=0.0, alpha=0.004, n=1.1, ss=1e-6,
                      kt=2.5, ct=2.3e6, th0=theta, thi0=0.0,
                      Tstar=-0.01, fc=0.03)
    return [None, layer]


def test_saturated_soil_uses_geotop_saturated_branch():
    soil = _soil(theta=0.40)
    P, Ta, Qa, Qgsat, rv = 900.0, 5.0, 0.003, 0.006, 100.0
    alpha, beta, evap = soil_evaporation_parameters(
        soil, [0.0, 200.0], [0.0, 5.0], [0.0, 0.40],
        P, rv, Ta, Qa, Qgsat, psi_surface=-10.0)
    rho = air_density(5.0, Qa, P)
    assert alpha == 1.0
    assert beta == 0.40
    assert evap[1] == pytest.approx(0.40 * rho * (Qgsat - Qa) / rv)


def test_unsaturated_soil_reduces_surface_vapour_availability():
    soil = _soil(theta=0.30)
    alpha, beta, evap = soil_evaporation_parameters(
        soil, [0.0, 200.0], [0.0, 5.0], [0.0, 0.30],
        900.0, 100.0, 5.0, 0.003, 0.006, psi_surface=-100.0)
    # GEOtop's algebra can put alpha microscopically above one; beta carries
    # the dominant surface-resistance reduction.
    assert 0.0 < alpha < 1.01
    assert 0.0 < beta <= 1.0
    assert evap[1] > 0.0
