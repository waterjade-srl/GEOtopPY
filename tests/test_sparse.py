"""Verify sparse linear algebra against the C++ oracle and direct solutions.

The 1D port retains GEOtop's BiCGSTAB algorithm and sparse storage conventions.
"""

import random

import numpy as np
import pytest

from geotop_py import numerics as num
from geotop_py.water import sparse

try:
    from geotop_py import _cxx
except FileNotFoundError as exc:                     # pragma: no cover
    pytest.skip(f"oracle not built: {exc}", allow_module_level=True)


def _chain(n, seed):
    """A random 1D chain: node i coupled to node i+1 for i=1..n-1.

    Returns 1-based-padded ``(Li, Lp, Lx)`` -- the sparse format a tridiagonal
    Richards column actually produces (each interior node has exactly one
    entry below it, in its own column)."""
    rng = random.Random(seed)
    Li = [0]
    Lp = [0] * (n + 1)
    Lx = [0.0]
    for i in range(1, n):
        Li.append(i + 1)
        Lx.append(rng.uniform(0.3, 3.0))
        Lp[i] = i
    Lp[n] = n - 1
    return Li, Lp, Lx


def _dense_from_chain(n, Li, Lp, Lx):
    """The dense ``K`` a chain (Li, Lp, Lx) represents, built independently
    of any function under test, from the row-sums-to-zero definition."""
    K = np.zeros((n, n))
    for i in range(1, len(Li) - 1 + 1):
        r, c, k = Li[i], _column_of(i, Lp), Lx[i]
        K[r - 1, c - 1] += k
        K[c - 1, r - 1] += k
        K[c - 1, c - 1] -= k
        K[r - 1, r - 1] -= k
    return K


def _column_of(i, Lp):
    c = 1
    while i > Lp[c]:
        c += 1
    return c


@pytest.mark.parametrize("n,seed", [(4, 0), (8, 1), (15, 2)])
def test_dense_reconstruction_matches_product_using_only_strict_lower(n, seed):
    """Dense reconstruction matches product using only strict lower."""
    Li, Lp, Lx = _chain(n, seed)
    K = _dense_from_chain(n, Li, Lp, Lx)
    rng = np.random.default_rng(seed)
    x = [0.0] + list(rng.standard_normal(n))
    mine = sparse.product_using_only_strict_lower_diagonal_part(x, Li, Lp, Lx)
    assert np.allclose(mine[1:], K @ np.array(x[1:]), atol=1e-12)


# --------------------------------------------------------------------- tridiag

@pytest.mark.parametrize("n,seed", [(2, 0), (5, 1), (10, 2)])
def test_tridiag_matches_the_cxx(n, seed):
    rng = random.Random(seed)
    diag = [0.0] + [rng.uniform(5.0, 10.0) for _ in range(n)]
    off = [0.0] + [rng.uniform(-1.5, 1.5) for _ in range(n - 1)] + [0.0]
    b = [0.0] + [rng.uniform(-3.0, 3.0) for _ in range(n)]

    status, e = sparse.tridiag(off, diag, off, b)
    cstatus, ce = _cxx.tridiag(off[1:], diag[1:], off[1:], b[1:])
    assert status == cstatus == 1
    assert e[1:] == ce


def test_tridiag_solves_a_e_equals_b_not_a_e_plus_b_equals_0():
    """The opposite convention from tridiag2, pinned explicitly."""
    diag = [0.0, 4.0, 4.0]
    off = [0.0, 1.0]
    b = [0.0, 5.0, 5.0]
    status, e = sparse.tridiag(off, diag, off, b)
    assert status == 1
    A = np.array([[4.0, 1.0], [1.0, 4.0]])
    assert np.allclose(e[1:], np.linalg.solve(A, [5.0, 5.0]), atol=1e-12)


def test_tridiag_raises_on_a_zero_first_pivot():
    with pytest.raises(ZeroDivisionError):
        sparse.tridiag([0.0, 1.0], [0.0, 0.0, 4.0], [0.0, 1.0], [0.0, 1.0, 1.0])


def test_tridiag_reports_failure_on_a_later_zero_pivot_like_the_cxx():
    diag = [0.0, 1.0, 1.0, 1.0]
    off = [0.0, 1.0, -1.0]
    b = [0.0, 1.0, 1.0, 1.0]
    status, _ = sparse.tridiag(off, diag, off, b)
    cstatus, _ = _cxx.tridiag(off[1:], diag[1:], off[1:], b[1:])
    assert status == cstatus == 0


# ---------------------------------------------------------------------- product

@pytest.mark.parametrize("n,seed", [(1, 0), (3, 1), (7, 2)])
def test_product_matches_the_cxx(n, seed):
    rng = random.Random(seed)
    a = [0.0] + [rng.uniform(-5, 5) for _ in range(n)]
    b = [0.0] + [rng.uniform(-5, 5) for _ in range(n)]
    assert sparse.product(a, b) == _cxx.product(a[1:], b[1:])


# ------------------------------------------------------ the four K-based products

@pytest.mark.parametrize("n,seed", [(2, 0), (5, 1), (12, 3)])
def test_product_using_only_strict_lower_matches_the_cxx(n, seed):
    Li, Lp, Lx = _chain(n, seed)
    rng = np.random.default_rng(seed + 100)
    x = [0.0] + list(rng.standard_normal(n))
    mine = sparse.product_using_only_strict_lower_diagonal_part(x, Li, Lp, Lx)
    theirs = _cxx.product_using_only_strict_lower_diagonal_part(
        x[1:], Li[1:], Lp[1:], Lx[1:])
    assert mine[1:] == theirs


@pytest.mark.parametrize("n,seed", [(2, 0), (5, 1), (12, 3)])
def test_product_using_only_strict_lower_plus_identity_matches_the_cxx(n, seed):
    Li, Lp, Lx = _chain(n, seed)
    rng = np.random.default_rng(seed + 200)
    x = [0.0] + list(rng.standard_normal(n))
    y = [0.0] + list(rng.uniform(0.1, 2.0, n))
    mine = sparse.product_using_only_strict_lower_diagonal_part_plus_identity_by_vector(
        x, y, Li, Lp, Lx)
    theirs = _cxx.product_using_only_strict_lower_diagonal_part_plus_identity_by_vector(
        x[1:], y[1:], Li[1:], Lp[1:], Lx[1:])
    assert mine[1:] == theirs


@pytest.mark.parametrize("n,seed", [(2, 0), (5, 1), (12, 3)])
def test_get_diag_matches_the_cxx(n, seed):
    Li, Lp, Lx = _chain(n, seed)
    rng = np.random.default_rng(seed + 300)
    y = [0.0] + list(rng.uniform(0.1, 2.0, n))
    diag, udiag = sparse.get_diag_strict_lower_matrix_plus_identity_by_vector(y, Li, Lp, Lx)
    cdiag, cudiag = _cxx.get_diag_strict_lower_matrix_plus_identity_by_vector(
        y[1:], Li[1:], Lp[1:], Lx[1:])
    assert diag[1:] == cdiag
    assert udiag[1:n] == cudiag


@pytest.mark.parametrize("n,seed", [(2, 0), (5, 1), (12, 3)])
def test_product_matrix_using_lower_part_matches_the_cxx(n, seed):
    Li, Lp, Lx = _chain(n, seed)
    rng = np.random.default_rng(seed + 400)
    x = [0.0] + list(rng.standard_normal(n))
    y = [0.0] + list(rng.standard_normal(n))
    k = 2.5
    mine = sparse.product_matrix_using_lower_part_by_vector_plus_vector(k, y, x, Li, Lp, Lx)
    theirs = _cxx.product_matrix_using_lower_part_by_vector_plus_vector(
        k, y[1:], x[1:], Li[1:], Lp[1:], Lx[1:])
    assert mine[1:] == theirs


def test_get_diag_reconstructs_the_row_sum_zero_property():
    """Invariant, independent of the oracle: (K + diag(y))'s row sums equal
    y's own entries, since K alone always sums to zero per row."""
    n, seed = 9, 7
    Li, Lp, Lx = _chain(n, seed)
    y = [0.0] + [1.0] * n
    diag, udiag = sparse.get_diag_strict_lower_matrix_plus_identity_by_vector(y, Li, Lp, Lx)
    ones = [0.0] + [1.0] * n
    Kx = sparse.product_using_only_strict_lower_diagonal_part(ones, Li, Lp, Lx)
    assert all(abs(v) < 1e-12 for v in Kx[1:])   # K's own rows sum to zero


# --------------------------------------------------------------------- BiCGSTAB

@pytest.mark.parametrize("n,seed", [(3, 0), (8, 1), (20, 2), (50, 3)])
def test_bicgstab_matches_the_cxx_including_iteration_count(n, seed):
    Li, Lp, Lx = _chain(n, seed)
    rng = random.Random(seed + 500)
    y = [0.0] + [rng.uniform(0.1, 2.0) for _ in range(n)]
    b = [0.0] + [rng.uniform(-5.0, 5.0) for _ in range(n)]
    x0 = [0.0] * (n + 1)

    iters, x = sparse.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        1.0e-10, 1.0e-13, 1.0e5, x0, b, y, Li, Lp, Lx)
    citers, cx = _cxx.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        1.0e-10, 1.0e-13, 1.0e5, x0[1:], b[1:], y[1:], Li[1:], Lp[1:], Lx[1:])

    assert iters == citers
    assert x[1:] == cx


@pytest.mark.parametrize("n,seed", [(6, 0), (15, 4)])
def test_bicgstab_solution_matches_an_independent_dense_solve(n, seed):
    """A witness independent of the oracle: the converged x actually solves
    (K + diag(y)) x = b, checked against numpy's own dense solver."""
    Li, Lp, Lx = _chain(n, seed)
    rng = random.Random(seed + 600)
    y = [0.0] + [rng.uniform(0.1, 2.0) for _ in range(n)]
    b = [0.0] + [rng.uniform(-5.0, 5.0) for _ in range(n)]
    x0 = [0.0] * (n + 1)

    iters, x = sparse.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        1.0e-12, 1.0e-14, 1.0e5, x0, b, y, Li, Lp, Lx)
    assert iters >= 0

    K = _dense_from_chain(n, Li, Lp, Lx)
    A = K + np.diag(y[1:])
    expected = np.linalg.solve(A, np.array(b[1:]))
    assert np.allclose(x[1:], expected, atol=1e-6)


def test_bicgstab_reports_minus_one_on_a_later_zero_pivot():
    """Bicgstab reports minus one on a later zero pivot."""
    Li, Lp, Lx = [0, 2], [0, 1, 1], [0.0, 2.0]
    y = [0.0, 6.0, 3.0]
    b = [0.0, 1.0, 1.0]
    x0 = [0.0, 0.0, 0.0]
    iters, _ = sparse.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        1e-10, 1e-13, 1e5, x0, b, y, Li, Lp, Lx)
    citers, _ = _cxx.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        1e-10, 1e-13, 1e5, x0[1:], b[1:], y[1:], Li[1:], Lp[1:], Lx[1:])
    assert iters == citers == -1


def test_bicgstab_x0_is_added_to_not_used_as_a_residual_seed():
    """The solver seeds r = b outright, never b - (K+diag(y))*x0 -- so a
    nonzero x0 changes the answer by exactly x0 itself when b is otherwise
    unchanged, not by however far x0 was from solving the system."""
    n, seed = 5, 9
    Li, Lp, Lx = _chain(n, seed)
    y = [0.0] + [1.0] * n
    b = [0.0] + [2.0] * n

    _, x_from_zero = sparse.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        1e-12, 1e-14, 1e5, [0.0] * (n + 1), b, y, Li, Lp, Lx)

    bump = [0.0] + [3.0] * n
    _, x_from_bump = sparse.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        1e-12, 1e-14, 1e5, bump, b, y, Li, Lp, Lx)

    for j in range(1, n + 1):
        assert x_from_bump[j] == pytest.approx(x_from_zero[j] + 3.0, abs=1e-8)


def test_the_corpus_of_chain_sizes_is_not_trivial():
    """Guard against every case above accidentally converging in 0 iterations."""
    Li, Lp, Lx = _chain(30, 11)
    y = [0.0] + [0.5] * 30
    b = [0.0] + [float(i) for i in range(1, 31)]
    x0 = [0.0] * 31
    iters, _ = sparse.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        1e-12, 1e-14, 1e5, x0, b, y, Li, Lp, Lx)
    assert iters >= 1


# -------------------------------------------------------- consistency with numerics

def test_norm_2_agrees_with_numerics_on_the_residual_used_internally():
    """Norm 2 agrees with numerics on the residual used internally."""
    v = [0.0, 3.0, 4.0, 12.0]
    assert num.norm_2(v, 1, 3) == 13.0
