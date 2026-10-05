"""Numerical primitives of GEOtop's implicit solver.

Verbatim translations of the reusable kernels the energy and water balance
solvers build on: a tridiagonal (Thomas) solve, the conduction stencil and its
Jacobian contribution, the Euclidean residual norm, and a three-point
quadratic line-search step. These operate on plain 1-based Python lists
(index 0 unused), matching the 1-based containers the originals use.

:func:`tridiag2` is checked two ways in the tests: directly against the C++
oracle (bit-exact, the real fidelity bar), and against
``scipy.linalg.solve_banded`` -- an independent LU implementation, useful
because it catches an error the oracle pin alone would not: one that is also
present, identically, in the oracle wrapper's own array marshalling.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from . import constants as C


# GEOtop: src/libraries/math/util_math.h:84-133
def tridiag2(nbeg: int, nend: int,
             ld: Sequence[float], d: Sequence[float], ud: Sequence[float],
             b: Sequence[float], e: List[float]) -> int:
    """Solve the tridiagonal system ``A(ld, d, ud) * e + b = 0`` in place on
    ``e`` (Thomas algorithm). ``ld``/``ud`` are indexed by the *lower* index
    of the pair they connect (``ld[j-1]``/``ud[j-1]`` are both the coupling
    between rows ``j-1`` and ``j``). Returns 0 on success, 1 on a zero pivot.
    """
    gam = [0.0] * (nend + 1)

    bet = d[nbeg]
    if bet == 0.0:
        return 1

    e[nbeg] = -b[nbeg] / bet

    # decomposition and forward substitution
    for j in range(nbeg + 1, nend + 1):
        gam[j] = ud[j - 1] / bet
        bet = d[j] - ld[j - 1] * gam[j]
        if bet == 0.0:
            return 1
        e[j] = (-b[j] - ld[j - 1] * e[j - 1]) / bet

    # backsubstitution
    for j in range(nend - 1, nbeg - 1, -1):
        e[j] -= gam[j + 1] * e[j + 1]

    return 0


# GEOtop: src/geotop/energy.balance.cc:2289-2318
def update_F_energy(nbeg: int, nend: int, F: List[float], w: float,
                    K: Sequence[float], T: Sequence[float]) -> None:
    """Accumulate one time-weighted conduction stencil into the residual ``F``.

    Three-point stencil in the column interior, one-sided at each end (a
    Neumann-style top flux built from ``K[nbeg]``, a Dirichlet-style bottom
    term built from ``K[nend-1]``). Called twice per assembly with weights
    ``w`` and ``1-w`` on the new and old temperature respectively -- ``w=1``
    is Backward Euler, GEOtop's only actual choice; the caller decides ``w``.
    """
    for l in range(nbeg, nend + 1):
        if l == nbeg:
            F[l] += w * (-K[l] * T[l] + K[l] * T[l + 1])
        elif l < nend:
            F[l] += w * (K[l - 1] * T[l - 1] - (K[l] + K[l - 1]) * T[l] + K[l] * T[l + 1])
        else:
            F[l] += w * (K[l - 1] * T[l - 1] - K[l - 1] * T[l])


# GEOtop: src/geotop/energy.balance.cc:2319-2337
def update_diag_dF_energy(nbeg: int, nend: int, dF: List[float], w: float,
                          K: Sequence[float]) -> None:
    """Subtract this stencil's contribution from the Jacobian diagonal.

    The off-diagonal Jacobian terms come from the ``K`` values directly
    (they don't need accumulating); this only ever touches the diagonal.
    """
    for l in range(nbeg, nend + 1):
        if l == nbeg:
            dF[l] -= w * K[l]
        elif l < nend:
            dF[l] -= w * (K[l] + K[l - 1])
        else:
            dF[l] -= w * K[l - 1]


# GEOtop: src/libraries/math/util_math.h:145-154
def norm_2(V: Sequence[float], nbeg: int, nend: int) -> float:
    """Euclidean norm over ``[nbeg, nend]`` -- upper bound INCLUSIVE.

    ``nend`` is the last node's own equation, assembled in the Jacobian and
    solved by :func:`tridiag2` like every other node; excluding it from the
    residual would measure convergence of a strict subset of the system
    actually being solved.

    ``V[l] * V[l]`` rather than ``V[l] ** 2``: a Newton line-search trial can
    carry a huge-but-finite residual before being rejected, and Python's
    float ``**`` raises OverflowError on a too-large result where C's
    multiplication (and GEOtop's) silently saturates to inf -- letting the
    line search reject the trial on the next iteration instead of crashing."""
    N = 0.0
    for l in range(nbeg, nend + 1):
        N += V[l] * V[l]
    return math.sqrt(N)


# GEOtop: src/libraries/math/util_math.h:132-144
def norm_inf(V: Sequence[float], nbeg: int, nend: int) -> float:
    """Maximum absolute value over ``[nbeg, nend]`` -- upper bound inclusive,
    same convention as :func:`norm_2`."""
    N = 0.0
    for l in range(nbeg, nend + 1):
        if abs(V[l]) > N:
            N = abs(V[l])
    return N


# GEOtop: src/libraries/math/util_math.h:167-179
def norm_1(V: Sequence[float], nbeg: int, nend: int) -> float:
    """Sum of absolute values over ``[nbeg, nend]`` -- upper bound inclusive,
    same convention as :func:`norm_2`."""
    N = 0.0
    for l in range(nbeg, nend + 1):
        N += abs(V[l])
    return N


# GEOtop: src/libraries/math/util_math.h:184-198
def _cramer(A, B, Cc, D, E, F):
    """Solve the 2x2 system ``Ax+By=Cc``, ``Dx+Ey=F`` by Cramer's rule."""
    x = (Cc * E - F * B) / (A * E - D * B)
    y = (A * F - Cc * D) / (A * E - D * B)
    return x, y


# GEOtop: src/libraries/math/util_math.h:204-232
def minimize_merit_function(res0: float, lambda1: float, res1: float,
                            lambda2: float, res2: float) -> float:
    """Backtracking step size for the Newton line search.

    Fits the quadratic through the three known merit-function samples
    ``(0, res0)``, ``(lambda1, res1)``, ``(lambda2, res2)`` and returns the
    step that minimises it, clamped to ``[thmin, thmax] * lambda1``. If the
    fit isn't convex (``a <= 0``, no interior minimum), picks whichever clamp
    bound gives the lower fitted value instead.
    """
    c = res0
    a, b = _cramer(lambda1 * lambda1, lambda1, res1 - res0,
                   lambda2 * lambda2, lambda2, res2 - res0)

    if a > 0:
        lam = -b / (2 * a)
        if lam < lambda1 * C.thmin:
            lam = lambda1 * C.thmin
        elif lam > lambda1 * C.thmax:
            lam = lambda1 * C.thmax
    else:
        lo = lambda1 * C.thmin
        hi = lambda1 * C.thmax
        if a * lo * lo + b * lo + c < a * hi * hi + b * hi + c:
            lam = lo
        else:
            lam = hi
    return lam
