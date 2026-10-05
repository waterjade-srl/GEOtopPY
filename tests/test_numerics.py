"""Tests for geotop_py.numerics.

``tridiag2`` and ``norm_2`` are pinned directly against the oracle; ``tridiag2``
is additionally cross-checked against ``scipy.linalg.solve_banded`` and a dense
``numpy`` solve, both independent LU implementations that would not share a
mistake made only in the oracle wrapper's own array marshalling. The stencils
and the merit-function minimiser are pinned by direct hand computation.
"""

import random

import numpy as np
import pytest
from scipy.linalg import solve_banded

from geotop_py import constants as C
from geotop_py import numerics as num

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


def _dense_from_diags(nbeg, nend, ld, d, ud):
    """Build the dense tridiagonal A that tridiag2 represents (1-based diags)."""
    sz = nend - nbeg + 1
    A = np.zeros((sz, sz))
    for i, l in enumerate(range(nbeg, nend + 1)):
        A[i, i] = d[l]
        if l < nend:
            A[i, i + 1] = ud[l]
        if l > nbeg:
            A[i, i - 1] = ld[l - 1]
    return A


@pytest.mark.parametrize("seed", range(8))
def test_tridiag2_matches_scipy(seed):
    rng = random.Random(seed)
    nbeg, nend = 1, rng.randint(3, 9)
    # diagonally dominant -> well conditioned, symmetric off-diagonals
    d = [0.0] * (nend + 1)
    ld = [0.0] * (nend + 1)
    ud = [0.0] * (nend + 1)
    b = [0.0] * (nend + 1)
    for l in range(nbeg, nend + 1):
        d[l] = rng.uniform(5.0, 10.0)
        b[l] = rng.uniform(-3.0, 3.0)
    for l in range(nbeg, nend):
        off = rng.uniform(-1.5, 1.5)
        ud[l] = off
        ld[l] = off

    e = [0.0] * (nend + 1)
    assert num.tridiag2(nbeg, nend, ld, d, ud, b, e) == 0

    # tridiag2 solves A e + b = 0  ->  A e = -b
    A = _dense_from_diags(nbeg, nend, ld, d, ud)
    rhs = -np.array([b[l] for l in range(nbeg, nend + 1)])
    ref = np.linalg.solve(A, rhs)
    got = np.array([e[l] for l in range(nbeg, nend + 1)])
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-12)

    # directly against the oracle: numerics.tridiag2 keeps GEOtop's 1-based
    # indices (index 0 unused), _cxx.tridiag2 takes plain 0-based rows -- the
    # shift is the only translation needed since nbeg == 1 here.
    status, e_cxx = _cxx.tridiag2(nbeg, nend,
                                  ld[nbeg:nend], d[nbeg:nend + 1],
                                  ud[nbeg:nend], b[nbeg:nend + 1])
    assert status == 0
    assert e_cxx == [e[l] for l in range(nbeg, nend + 1)]

    # and against scipy's banded solver as a second, independent witness
    ab = np.zeros((3, nend - nbeg + 1))
    ab[0, 1:] = [ud[l] for l in range(nbeg, nend)]
    ab[1, :] = [d[l] for l in range(nbeg, nend + 1)]
    ab[2, :-1] = [ld[l] for l in range(nbeg, nend)]
    ref2 = solve_banded((1, 1), ab, rhs)
    assert np.allclose(got, ref2, rtol=1e-10, atol=1e-12)


def test_tridiag2_zero_pivot():
    e = [0.0] * 3
    assert num.tridiag2(1, 2, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1], e) == 1


def test_update_F_energy_matches_explicit_stencil():
    # F += w * (K * grad T) with GEOtop's boundary handling; check by hand
    K = [0.0, 2.0, 3.0]          # K[1] interface 1-2, K[2] interface 2-3
    T = [0.0, 10.0, 4.0, 1.0]
    F = [0.0] * 4
    num.update_F_energy(1, 3, F, 1.0, K, T)
    assert F[1] == pytest.approx(-K[1] * T[1] + K[1] * T[2])
    assert F[2] == pytest.approx(K[1] * T[1] - (K[2] + K[1]) * T[2] + K[2] * T[3])
    assert F[3] == pytest.approx(K[2] * T[2] - K[2] * T[3])


def test_update_diag_dF_energy_matches_explicit():
    K = [0.0, 2.0, 3.0]
    dF = [0.0] * 4
    num.update_diag_dF_energy(1, 3, dF, 1.0, K)
    assert dF[1] == pytest.approx(-K[1])
    assert dF[2] == pytest.approx(-(K[2] + K[1]))
    assert dF[3] == pytest.approx(-K[2])


def test_norm_2_includes_last_node():
    """fixed from an exclusive bound inherited from a fork's own copy of
    this function -- the official upper bound is inclusive,
    and excluding the last node changes when the Newton solver stops."""
    V = [0.0, 3.0, 4.0, 12.0]     # index 3 (nend) must be included
    assert num.norm_2(V, 1, 3) == pytest.approx(13.0)


@pytest.mark.parametrize("seed", range(5))
def test_norm_2_matches_the_cxx(seed):
    """Norm 2 matches the cxx."""
    rng = random.Random(seed)
    nend = rng.randint(2, 8)
    V = [0.0] + [rng.uniform(-50.0, 50.0) for _ in range(nend)]
    assert num.norm_2(V, 1, nend) == _cxx.norm_2(V[1:], 1, nend)


@pytest.mark.parametrize("seed", range(5))
def test_norm_inf_matches_the_cxx(seed):
    rng = random.Random(seed)
    nend = rng.randint(2, 8)
    V = [0.0] + [rng.uniform(-50.0, 50.0) for _ in range(nend)]
    assert num.norm_inf(V, 1, nend) == _cxx.norm_inf(V[1:], 1, nend)


def test_norm_inf_includes_last_node():
    V = [0.0, 1.0, 2.0, 99.0]
    assert num.norm_inf(V, 1, 3) == 99.0


@pytest.mark.parametrize("seed", range(5))
def test_norm_1_matches_the_cxx(seed):
    rng = random.Random(seed)
    nend = rng.randint(2, 8)
    V = [0.0] + [rng.uniform(-50.0, 50.0) for _ in range(nend)]
    assert num.norm_1(V, 1, nend) == _cxx.norm_1(V[1:], 1, nend)


def test_norm_1_includes_last_node():
    V = [0.0, -1.0, 2.0, -3.0]
    assert num.norm_1(V, 1, 3) == 6.0


def test_minimize_merit_convex_interior():
    # merit(x) = (x - 0.3)^2 sampled at the three probe points -> min at 0.3,
    # clamped into [thmin, thmax]*lambda1
    def merit(x):
        return (x - 0.3) ** 2
    lambda1, lambda2 = 1.0, 0.5
    lam = num.minimize_merit_function(merit(0.0), lambda1, merit(lambda1),
                                      lambda2, merit(lambda2))
    assert lam == pytest.approx(0.3, abs=1e-9)


def test_minimize_merit_clamped_low():
    def merit(x):
        return (x + 5.0) ** 2         # minimiser far left -> clamp to thmin
    lam = num.minimize_merit_function(merit(0.0), 1.0, merit(1.0), 0.5, merit(0.5))
    assert lam == pytest.approx(1.0 * C.thmin)
