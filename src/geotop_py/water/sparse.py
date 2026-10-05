"""Iterative solver for Richards' Jacobian, in GEOtop's own sparse format.

Richards' equation in 1D assembles into a symmetric M-matrix ``K`` coupling
each soil layer to its neighbours, plus a diagonal term ``y`` (the moisture
capacity). ``K`` is stored by its **strict lower triangle only**, as three
parallel 1-based arrays:

* ``Lx[i]``: the value of the i-th stored entry
* ``Li[i]``: that entry's row
* ``Lp[c]``: the running count of entries through column ``c`` -- entry ``i``
  belongs to column ``c`` exactly when ``Lp[c-1] < i <= Lp[c]``

The diagonal of ``K`` is never stored: every function below reconstructs it
from ``K``'s defining property, that its rows sum to zero. An entry
``k = Lx[i]`` at ``(row=r, col=c)`` contributes ``k*(x[r]-x[c])`` to row
``c``'s equation and ``k*(x[c]-x[r])`` to row ``r``'s -- a diffusive coupling,
symmetric, with no net source. In 1D, where each node only couples to its
immediate neighbours, this degenerates to a tridiagonal system, but the format
and the code here make no such assumption.

The system solved is ``(K + diag(y)) x = b``, by BiCGSTAB preconditioned with
the tridiagonal part of ``K + diag(y)``. A direct tridiagonal solve would be
exact and cheaper in 1D, but GEOtop always takes the iterative path, and
reproducing which iteration it stops the Newton loop on to machine tolerance
requires reproducing the same iteration sequence -- see the "BiCGSTAB in 1D"
note in the project docs. This is a straight, unoptimised translation for
that reason: no vectorisation, no early-exit shortcuts beyond GEOtop's own.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .. import numerics as num

Vec = Sequence[float]
IntVec = Sequence[int]


# GEOtop: src/libraries/math/util_math.h:32-77
def tridiag(diag_inf: Vec, diag: Vec, diag_sup: Vec, b: Vec) -> Tuple[int, List[float]]:
    """Solve the tridiagonal system ``A * e = b``.

    Both off-diagonal arrays are indexed by the *lower* index of the pair
    they couple (``diag_inf[j-1]`` and ``diag_sup[j-1]`` both belong to the
    coupling between rows ``j-1`` and ``j``), 1-based with index 0 unused
    padding on every array. Returns ``(status, e)``: status 1 on success, 0 on
    a zero pivot past the first (a zero *first* pivot is fatal in GEoTop; here
    it raises instead of aborting the process).

    This is the opposite return convention from :func:`geotop_py.numerics.tridiag2`
    (which returns 0 for success), and solves ``A*e = b`` rather than
    ``A*e + b = 0``. The two are not interchangeable.
    """
    n = len(diag) - 1
    e = [0.0] * (n + 1)
    gam = [0.0] * (n + 1)

    if diag[1] == 0.0:
        raise ZeroDivisionError("tridiag: zero first pivot")

    bet = diag[1]
    e[1] = b[1] / bet

    for j in range(2, n + 1):
        gam[j] = diag_sup[j - 1] / bet
        bet = diag[j] - diag_inf[j - 1] * gam[j]
        if bet == 0.0:
            return 0, e
        e[j] = (b[j] - diag_inf[j - 1] * e[j - 1]) / bet

    for j in range(n - 1, 0, -1):
        e[j] -= gam[j + 1] * e[j + 1]

    return 1, e


# GEOtop: src/libraries/math/util_math.h:247-258
def product(a: Vec, b: Vec) -> float:
    """Dot product of two equal-length, 1-based (index 0 unused) vectors."""
    n = len(a) - 1
    p = 0.0
    for i in range(1, n + 1):
        p += a[i] * b[i]
    return p


def _walk_columns(Li: IntVec, Lp: IntVec):
    """Yield ``(i, row, col)`` for each stored entry, tracking which column
    ``i`` currently belongs to via ``Lp``'s running counts -- the same walk
    every function in this module repeats over the same arrays."""
    n_entries = len(Li) - 1
    c = 1
    for i in range(1, n_entries + 1):
        yield i, Li[i], c
        if i < n_entries:
            while i >= Lp[c]:
                c += 1


# GEOtop: src/libraries/math/sparse_matrix.cc:1167-1200
def product_using_only_strict_lower_diagonal_part(
        x: Vec, Li: IntVec, Lp: IntVec, Lx: Vec) -> List[float]:
    """``K x``, for ``K``'s off-diagonal coupling alone (no diagonal term)."""
    n = len(x) - 1
    out = [0.0] * (n + 1)
    for i, r, c in _walk_columns(Li, Lp):
        if r <= c:
            raise ValueError(f"matrix is not strictly lower triangular at entry {i}")
        k = Lx[i]
        out[c] += k * (x[r] - x[c])
        out[r] += k * (x[c] - x[r])
    return out


# GEOtop: src/libraries/math/sparse_matrix.cc:1210-1249
def product_using_only_strict_lower_diagonal_part_plus_identity_by_vector(
        x: Vec, y: Vec, Li: IntVec, Lp: IntVec, Lx: Vec) -> List[float]:
    """``(K + diag(y)) x``."""
    n = len(x) - 1
    out = [y[i] * x[i] for i in range(n + 1)]
    for i, r, c in _walk_columns(Li, Lp):
        if r <= c:
            raise ValueError(f"matrix is not strictly lower triangular at entry {i}")
        k = Lx[i]
        out[c] += k * (x[r] - x[c])
        out[r] += k * (x[c] - x[r])
    return out


# GEOtop: src/libraries/math/sparse_matrix.cc:1256-1281
def get_diag_strict_lower_matrix_plus_identity_by_vector(
        y: Vec, Li: IntVec, Lp: IntVec, Lx: Vec) -> Tuple[List[float], List[float]]:
    """Diagonal and upper off-diagonal of the **tridiagonal part** of
    ``K + diag(y)`` -- what :func:`tridiag` needs as a preconditioner.

    A node coupled to more than its immediate neighbours still has every one
    of those couplings folded into its diagonal entry (rows always sum to
    zero); only couplings between *consecutive* indices (``row == col + 1``)
    show up in ``udiag``, so this is an exact tridiagonal factor only when
    ``K`` itself is tridiagonal -- true for a 1D Richards column, not assumed
    by the code.
    """
    n = len(y) - 1
    diag = list(y)
    udiag = [0.0] * n
    for i, r, c in _walk_columns(Li, Lp):
        k = Lx[i]
        diag[c] -= k
        diag[r] -= k
        if r == c + 1:
            udiag[c] = k
    return diag, udiag


# GEOtop: src/libraries/math/sparse_matrix.cc:1413-1429
def product_matrix_using_lower_part_by_vector_plus_vector(
        k: float, y: Vec, x: Vec, Li: IntVec, Lp: IntVec, Lx: Vec) -> List[float]:
    """``k * (y + K x)``, ``K``'s diagonal excluded (folded into ``y`` by the
    caller, if wanted -- unlike the ``_plus_identity_by_vector`` product above,
    this one adds no diagonal term of its own)."""
    out = product_using_only_strict_lower_diagonal_part(x, Li, Lp, Lx)
    n = len(x) - 1
    for i in range(1, n + 1):
        out[i] = k * (out[i] + y[i])
    return out


# GEOtop: src/libraries/math/sparse_matrix.cc:1294-1400
def BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        tol_rel: float, tol_min: float, tol_max: float,
        x0: Vec, b: Vec, y: Vec, Li: IntVec, Lp: IntVec, Lx: Vec
        ) -> Tuple[int, List[float]]:
    """Solve ``(K + diag(y)) x = b`` by BiCGSTAB, preconditioned by the
    tridiagonal part of ``K + diag(y)``.

    Starts from ``x0`` (added to, not overwritten: the residual is seeded as
    ``r = b`` outright, not ``b - (K+diag(y))*x0``, so a nonzero ``x0`` is
    only a good starting guess if it is also consistent with that seed).
    Iterates until the residual's 2-norm drops to
    ``max(tol_min, min(tol_max, tol_rel * norm(b)))`` or a fixed iteration cap
    (``max(100, n // 100)``) is reached, whichever comes first -- so a request
    for a residual below ``tol_min`` or above ``tol_max`` is silently
    reinterpreted at that clamp, never an error.

    Returns ``(iterations, x)``. ``iterations`` is the count of BiCGSTAB
    iterations actually run (which is what has to match GEOtop's own run for
    the Newton loop's stopping point to agree, not just the final ``x``), or
    -1 if the tridiagonal preconditioner solve hit a zero pivot.
    """
    n = len(x0) - 1
    x = list(x0)
    diag, udiag = get_diag_strict_lower_matrix_plus_identity_by_vector(y, Li, Lp, Lx)

    r = list(b)
    r0 = list(r)
    p = [0.0] * (n + 1)
    v = [0.0] * (n + 1)

    norm_r0 = num.norm_2(r0, 1, n)

    rho = 1.0
    alpha = 1.0
    omeg = 1.0

    maxiter = max(100, n // 100)

    i = 0
    threshold = max(tol_min, min(tol_max, tol_rel * norm_r0))
    while i <= maxiter and num.norm_2(r, 1, n) > threshold:
        rho1 = product(r0, r)
        beta = (rho1 / rho) * (alpha / omeg)
        rho = rho1

        for j in range(1, n + 1):
            p[j] = r[j] + beta * (p[j] - omeg * v[j])

        status, yy = tridiag(udiag, diag, udiag, p)
        if status == 0:
            return -1, x

        v = product_using_only_strict_lower_diagonal_part_plus_identity_by_vector(
            yy, y, Li, Lp, Lx)

        alpha = rho / product(r0, v)

        s = [0.0] * (n + 1)
        for j in range(1, n + 1):
            s[j] = r[j] - alpha * v[j]

        if num.norm_2(s, 1, n) > 1.0e-10:
            status, z = tridiag(udiag, diag, udiag, s)
            if status == 0:
                return -1, x

            t = product_using_only_strict_lower_diagonal_part_plus_identity_by_vector(
                z, y, Li, Lp, Lx)

            omeg = product(t, s) / product(t, t)

            for j in range(1, n + 1):
                x[j] += alpha * yy[j] + omeg * z[j]
                r[j] = s[j] - omeg * t[j]
        else:
            for j in range(1, n + 1):
                x[j] += alpha * yy[j]
                r[j] = s[j]

        i += 1

    return i, x
