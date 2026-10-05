"""Pin every pure law in geotop_py.laws against the C++ oracle in geotop_py._cxx.

The oracle is the compiled GEOtop v3.0 model itself; agreement here means the Python
column reproduces GEOtop bit-for-bit at the level of the constitutive laws. If
the oracle shared object is missing, the whole module is skipped with a hint to
run ``oracle/build.sh``.
"""


import pytest

from geotop_py import laws

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


# Agreement to ~1e-12 relative; these are the same double ops in a different
# language, so they should match to the last few ULPs, not just "closely".
def close(py, cxx, rel=1e-12, abs_=1e-12):
    assert py == pytest.approx(cxx, rel=rel, abs=abs_), (py, cxx)


TEMPS = [-40.0, -21.0, -20.0, -4.0, -1.0, -0.5, 0.0, 0.5, 5.0]
WINDS = [0.0, 1.0, 3.5, 10.0]


@pytest.mark.parametrize("T", TEMPS)
@pytest.mark.parametrize("u", WINDS)
def test_rho_newlyfallensnow(u, T):
    close(laws.rho_newlyfallensnow(u, T), _cxx.rho_newlyfallensnow(u, T))


@pytest.mark.parametrize("T", TEMPS)
def test_theta_snow(T):
    close(laws.theta_snow(1.0, 1.0, T), _cxx.theta_snow(1.0, 1.0, T))


@pytest.mark.parametrize("T", TEMPS)
def test_dtheta_snow(T):
    close(laws.dtheta_snow(1.0, 1.0, T), _cxx.dtheta_snow(1.0, 1.0, T))


@pytest.mark.parametrize("T", TEMPS)
def test_internal_energy(T):
    close(laws.internal_energy(10.0, 1.0, T), _cxx.internal_energy(10.0, 1.0, T))


@pytest.mark.parametrize("T", TEMPS)
def test_Psif(T):
    close(laws.Psif(T), _cxx.Psif(T))


@pytest.mark.parametrize("snow,a", [(0, 1), (1, 1), (1, 2), (1, 3)])
def test_k_thermal(snow, a):
    py = laws.k_thermal(snow, a, 0.1, 0.4, 1.0, 3.0)
    cxx = _cxx.k_thermal(snow, a, 0.1, 0.4, 1.0, 3.0)
    close(py, cxx)


def test_C_snow():
    close(laws.C_snow(10.0, 1.0, 0.2, 1.0, 300.0),
          _cxx.C_snow(10.0, 1.0, 0.2, 1.0, 300.0))


def test_C_soil():
    close(laws.C_soil(2.3e6, 0.4, 10.0, 1.0, 0.2, 1.0, 300.0),
          _cxx.C_soil(2.3e6, 0.4, 10.0, 1.0, 0.2, 1.0, 300.0))


# van Genuchten: sample a realistic silty-loam parameter set across psi.
VG = dict(i=0.0, s=0.45, r=0.05, a=0.004, n=1.6, m=1.0 - 1.0 / 1.6,
          pmin=-1.0e10, Ss=1.0e-3)


@pytest.mark.parametrize("psi", [-1.0e5, -1000.0, -100.0, -10.0, -1.0, 0.0])
def test_teta_psi(psi):
    close(laws.teta_psi(psi, **VG), _cxx.teta_psi(psi, **VG))


@pytest.mark.parametrize("psi", [-1.0e5, -1000.0, -100.0, -10.0, -1.0])
def test_dteta_dpsi(psi):
    close(laws.dteta_dpsi(psi, **VG), _cxx.dteta_dpsi(psi, **VG))


@pytest.mark.parametrize("w", [0.05, 0.15, 0.30, 0.45])
def test_psi_teta(w):
    close(laws.psi_teta(w, **VG), _cxx.psi_teta(w, **VG))


@pytest.mark.parametrize("T", [-10.0, -5.0, -1.0, -0.2])
def test_from_internal_energy_roundtrip(T):
    # Build an enthalpy from a known frozen state, then recover it two ways.
    SWE = 10.0
    h = laws.internal_energy(SWE, 0.0, T)
    py = laws.from_internal_energy(1.0, h, SWE, 0.0)
    cxx = _cxx.from_internal_energy(1.0, h, SWE, 0.0)
    for a, b in zip(py, cxx):
        close(a, b, rel=1e-9, abs_=1e-9)


def test_from_internal_energy_conserves():
    # The defining property: recovering (wi, wl, T) from h and feeding them back
    # through internal_energy must return h.
    SWE = 12.0
    h = laws.internal_energy(SWE, 0.0, -3.0)
    wi, wl, T = laws.from_internal_energy(1.0, h, SWE, 0.0)
    assert laws.internal_energy(wi, wl, T) == pytest.approx(h, rel=1e-7)
    assert wi + wl == pytest.approx(SWE, rel=1e-12)
