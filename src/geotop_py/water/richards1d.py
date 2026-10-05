"""Richards' equation for one 1D soil column: assembly and Newton driver.

The column has ``nl`` soil layers under a surface node. State lives on
``nl + 1`` nodes, numbered ``0..nl``: node 0 is the surface (its head is a
ponding depth above the ground, not a soil matric potential), nodes ``1..nl``
are the soil layers. The unknown solved for is total head ``H = psi + Z``
(matric potential plus node elevation, both in mm); the flux between two
adjacent nodes is proportional to ``(H[i] - H[i+1])``, which is what makes the
system assemble into GEOtop's sparse chain format (see :mod:`geotop_py.water.sparse`)
in the first place -- gravity is folded into ``H`` rather than appearing as a
separate source term.

Three pieces are assembled by the Newton driver:

* :func:`find_matrix_K_1D` -- the off-diagonal flux conductivities ``Lx``
  between adjacent nodes (and the bottom boundary conductivity, if that
  boundary is a free-drainage one). Assembled once per step from the head the
  step starts at, and re-assembled per iteration only under ``update_K``;
* :func:`find_dfdH_1D` -- the diagonal moisture-capacity term (plus lateral
  drainage's own diagonal contribution);
* :func:`find_f_1D` -- the residual: the volume-balance mismatch this step,
  plus drainage and evapotranspiration source/sink terms.

:func:`solve_richards_1d` is the Newton driver built on top: assemble, solve
one linear step with :func:`geotop_py.water.sparse.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector`,
take a damped step chosen by a non-monotone line search
(:func:`geotop_py.numerics.minimize_merit_function`), repeat until the
residual's infinity norm is small enough or the iteration budget runs out.

Every quantity here that has a natural test against a synthetic small system
is pinned directly against the oracle (assembly, one Newton step). The full
driver's Newton *trajectory* -- how many outer iterations, how many line-search
backtracks, matching GEOtop node for node on a real soil column -- is not
pinned against a struct-level oracle call (``Richards1D`` takes GEOtop's full
``ALLDATA``/``SOIL_STATE``, which the oracle does not construct); it is
verified instead by invariants (conservation, the assembled system's own
residual definition, monotonicity of the constitutive pieces it is built
from), and by complete reference-case simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .. import constants as C
from .. import numerics as num
from . import soilwater as sw
from . import sparse

Vec = Sequence[float]


# GEOtop: src/geotop/input.cc:2426-2445 (the point_sim==1 branch)
def chain_topology(nl: int) -> Tuple[List[int], List[int]]:
    """The fixed sparse topology of an ``nl``-layer column: node 0 (surface)
    chained to node 1, node 1 to node 2, ..., node ``nl-1`` to node ``nl``.

    Returns ``(Li, Lp)`` in :mod:`geotop_py.water.sparse`'s 1-based, index-0-unused
    convention, with ``nl`` stored entries. This never changes during a run
    (only the entries' values, ``Lx``, do), so it is computed once and reused.
    """
    Li = [0] + [l + 1 for l in range(1, nl + 1)]
    Lp = [0] + list(range(1, nl + 1)) + [nl]
    return Li, Lp


# GEOtop: src/geotop/input.cc:3793-3868
def node_depths(dz: Vec, slope_deg: float) -> List[float]:
    """Node elevations ``Z`` [mm], node 0 (surface) at 0, decreasing (more
    negative) with depth. ``Z[l]`` is the *centre* of layer ``l``, projected
    onto the vertical through ``cos(slope)`` -- matching how ``dz`` (measured
    along the layer, not vertically) enters every flux calculation below."""
    nl = len(dz) - 1
    cosine = _cos_deg(slope_deg)
    Z = [0.0] * (nl + 1)
    z = 0.0
    for l in range(1, nl + 1):
        z -= 0.5 * dz[l] * cosine
        Z[l] = z
        z -= 0.5 * dz[l] * cosine
    return Z


def _cos_deg(deg: float) -> float:
    return math.cos(deg * C.Pi / 180.0)


@dataclass
class RichardsColumn:
    """Fixed-for-the-step geometry, soil and boundary configuration of one
    1D Richards column -- everything :func:`find_matrix_K_1D`,
    :func:`find_dfdH_1D` and :func:`find_f_1D` need besides the state
    ``H``/``P``/``thi``/``T`` itself.

    ``pa`` follows :mod:`geotop_py.water.soilwater`'s convention: ``pa[row][layer]``,
    both axes 1-based. ``nl`` may be smaller than ``len(dz) - 1`` (the spin-up
    truncation, ``min(Nl_spinup, Nl)``): only layers ``1..nl`` take part in the
    solve, deeper ones are inert for this step.
    """
    dz: List[float]                 # [mm] layer thickness, 1-based (index 0 unused)
    Z: List[float]                  # [mm] node depth, 0=surface, 1..nl=layer centres
    pa: List[List[float]]           # soil parameters, [row][layer], both 1-based
    nl: int                         # active layers (<= len(dz) - 1)
    slope_deg: float = 0.0
    area: float = 1.0               # [m2] horizontal cell area
    imp: float = 7.0
    k_to_ksat: float = 0.0
    free_drainage_bottom: bool = False
    free_drainage_lateral: float = 1.0
    bc_depth_free_surface: float = 0.0   # [mm]

    def __post_init__(self):
        self.Li, self.Lp = chain_topology(self.nl)


# GEOtop: src/geotop/water.balance.cc:1451-1568
def find_matrix_K_1D(H: Vec, thi: Vec, T: Vec, col: RichardsColumn
                      ) -> Tuple[List[float], float]:
    """Off-diagonal flux conductivities between adjacent nodes, and the
    bottom-boundary conductivity if that boundary drains freely.

    Between two nodes, the conductivity is evaluated **upwind**: at the
    receiving node's own head if flow is upward, at the source node's if
    downward (the top link uses the saturated conductivity for "downward",
    since node 0 carries no matric-potential-like ``thi``/``T`` of its own).
    For two soil layers it is then capped by the smaller of the two layers'
    own saturated conductivities -- flow cannot exceed what the more
    restrictive of the two neighbours can carry.

    Returns ``(Lx, Kbottom)``: ``Lx`` in :mod:`geotop_py.water.sparse`'s 1-based
    convention (``nl`` entries), ``Kbottom`` the bottom-boundary conductivity
    (0.0 unless the bottom is a free-drainage boundary evaluated this call).
    """
    nl = col.nl
    Lx = [0.0] * (nl + 1)
    Kbottom = 0.0

    for l in range(0, nl + 1):
        i = l + 1

        if l > 0 and l == nl and col.free_drainage_bottom:
            Kbottom = sw.k_from_psi(C.jKn, H[i] - col.Z[l], thi[l], T[l], l,
                                    col.pa, col.imp, col.k_to_ksat)

        if l < nl:
            I = i + 1
            if l == 0:
                if H[i] < H[I]:
                    kn = sw.k_from_psi(C.jKn, H[I] - col.Z[1], thi[1], T[1], 1,
                                       col.pa, col.imp, col.k_to_ksat)
                else:
                    psisat = sw.psisat_from(thi[1], 1, col.pa)
                    kn = sw.k_from_psi(C.jKn, psisat, thi[1], T[1], 1,
                                       col.pa, col.imp, col.k_to_ksat)
                dD = 0.5 * col.dz[1]
            else:
                if H[i] < H[I]:
                    kn = sw.k_from_psi(C.jKn, H[I] - col.Z[l + 1], thi[l + 1], T[l + 1],
                                       l + 1, col.pa, col.imp, col.k_to_ksat)
                else:
                    kn = sw.k_from_psi(C.jKn, H[i] - col.Z[l], thi[l], T[l], l,
                                       col.pa, col.imp, col.k_to_ksat)

                kmax = sw.k_from_psi(C.jKn, sw.psisat_from(thi[l], l, col.pa),
                                     thi[l], T[l], l, col.pa, col.imp, col.k_to_ksat)
                kmaxn = sw.k_from_psi(C.jKn, sw.psisat_from(thi[l + 1], l + 1, col.pa),
                                      thi[l + 1], T[l + 1], l + 1, col.pa, col.imp, col.k_to_ksat)
                kn = min(kn, min(kmax, kmaxn))
                dD = 0.5 * col.dz[l] + 0.5 * col.dz[l + 1]

            Lx[l + 1] = -col.area * kn / dD

    return Lx, Kbottom


def _lateral_drainage_active(l: int, H_i: float, col: RichardsColumn) -> bool:
    """Whether layer ``l`` (node ``i``) is above the free-surface boundary
    depth and saturated -- the shared gate every ``find_*`` function below
    applies before adding a lateral-drainage term."""
    if l == 0:
        return False
    cosine = _cos_deg(col.slope_deg)
    return (-col.Z[l] <= col.bc_depth_free_surface * cosine
            and H_i - col.Z[l] > 0.0)


# GEOtop: src/geotop/water.balance.cc:1669-1721
def find_dfdH_1D(H: Vec, thi: Vec, Klat: Vec, Dt: float, col: RichardsColumn
                  ) -> List[float]:
    """Diagonal moisture-capacity term of the Jacobian, plus lateral drainage.

    Node 0 (surface ponding) only has capacity while actually ponded
    (``H[0] > 0``): a dry surface contributes nothing, since there is no
    stored volume to have a capacity with respect to. ``Klat`` is the lateral
    conductivity :func:`find_matrix_K_1D`'s twin, per-layer lateral flow,
    computes -- passed in rather than recomputed here.
    """
    nl = col.nl
    df = [0.0] * (nl + 2)      # nl+1 nodes (0..nl), 1-based: index 0 unused
    max_slope_rad = min(C.max_slope, col.slope_deg) * C.Pi / 180.0

    for l in range(0, nl + 1):
        i = l + 1
        psi1 = H[i] - col.Z[l]

        if l == 0:
            if psi1 > 0:
                df[i] += col.area / math.cos(max_slope_rad) / Dt
        else:
            dz = col.dz[l]
            s, r, a, n = (col.pa[C.jsat][l], col.pa[C.jres][l],
                         col.pa[C.ja][l], col.pa[C.jns][l])
            m = 1.0 - 1.0 / n
            df[i] += sw.dteta_dpsi(psi1, thi[l], s, r, a, n, m, C.PsiMin,
                                   col.pa[C.jss][l]) * col.area * dz / Dt

        if _lateral_drainage_active(l, H[i], col):
            dz = col.dz[l]
            ds = math.sqrt(col.area)
            dn = ds
            dD = 0.5 * 1.0e3 * ds
            df[i] += (dn * dz * 1.0e-3) * Klat[l] * col.free_drainage_lateral / dD

    return df


# GEOtop: src/geotop/water.balance.cc:1893-1965
def find_f_1D(H: Vec, P0: Vec, thi: Vec, Klat: Vec, Kbottom: float,
              Pnet: float, ET: Vec, Dt: float, col: RichardsColumn) -> List[float]:
    """The residual: volume change this step, plus drainage and source terms.

    ``P0`` is the *previous* step's matric potential (node 0's own ponding
    depth included), used to compute the stored-volume change ``(V1-V0)/Dt``
    that is this residual's core. ``Pnet`` [mm over the step, already
    slope-projected by the caller] is the net precipitation reaching the
    surface (a sink on node 0's equation, since it adds water that must be
    balanced by a change in storage or an outflow); ``ET[l]`` [mm/s] is the
    per-layer evapotranspiration sink (a source on that layer's own equation
    in this residual's sign convention, since it removes water the volume
    term must otherwise account for).
    """
    nl = col.nl
    f = [0.0] * (nl + 2)      # nl+1 nodes (0..nl), 1-based: index 0 unused
    max_slope_rad = min(C.max_slope, col.slope_deg) * C.Pi / 180.0

    for l in range(0, nl + 1):
        i = l + 1
        psi0 = P0[l]
        psi1 = H[i] - col.Z[l]

        if l == 0:
            cosine = math.cos(max_slope_rad)
            V1 = col.area * max(0.0, psi1) / cosine
            V0 = col.area * max(0.0, psi0) / cosine
        else:
            dz = col.dz[l]
            V1 = col.area * dz * sw.theta_from_psi(psi1, thi[l], l, col.pa, C.PsiMin)
            V0 = col.area * dz * sw.theta_from_psi(psi0, thi[l], l, col.pa, C.PsiMin)

        f[i] = (V1 - V0) / Dt

        if l == nl:
            f[i] += col.area * Kbottom

        if _lateral_drainage_active(l, H[i], col):
            dz = col.dz[l]
            ds = math.sqrt(col.area)
            dn = ds
            dD = 0.5 * 1.0e3 * ds
            f[i] += (dn * dz * 1.0e-3) * Klat[l] * col.free_drainage_lateral * \
                (H[i] - col.Z[l]) / dD

        if l > 0:
            f[i] += col.area * ET[l] / Dt
        else:
            f[i] -= col.area * Pnet / Dt

    return f


@dataclass
class RichardsParams:
    """Newton-loop tolerances and switches (``geotop.inpts``, ``TolVWb`` etc.)."""
    TolVWb: float = 1.0e-6
    RelTolVWb: float = 1.0e-10           # GTConst::RelativeErrorRichards, not a keyword
    MaxiterTol: int = 100
    TolCG: float = 0.01
    min_lambda_wat: float = 1.0e-7
    max_times_min_lambda_wat: int = 0
    exit_lambda_min_wat: bool = True
    # Whether the hydraulic conductivities are re-evaluated at the current
    # Newton iterate. Off by default -- and off in every reference case -- so
    # the whole step is solved with the conductivities of the head the step
    # *started* from. That is a modelling choice, not an approximation to be
    # improved: switching it on makes the wetting front advance measurably
    # faster, because a wetting layer's conductivity rises within the step.
    update_K: bool = False

    @classmethod
    def from_parfile(cls, pf) -> "RichardsParams":
        return cls(
            TolVWb=pf.number("RichardTol", 0, 1.0e-6),
            MaxiterTol=int(pf.number("RichardMaxIter", 0, 100.0)),
            TolCG=pf.number("RichardInitForc", 0, 0.01),
            min_lambda_wat=pf.number("MinLambdaWater", 0, 1.0e-7),
            max_times_min_lambda_wat=int(pf.number("MaxTimesMinLambdaWater", 0, 0.0)),
            exit_lambda_min_wat=bool(pf.number("ExitMinLambdaWater", 0, 1.0)),
            update_K=bool(pf.number("UpdateHydraulicConductivity", 0, 0.0)),
        )


@dataclass
class RichardsState:
    """The persistent per-node state Richards owns and updates.

    ``P`` is matric potential [mm] (node 0: surface ponding depth), ``thi``
    ice content [-] (node 0 unused), ``T`` temperature [degC] (node 0 unused,
    needed by the hydraulic-conductivity freezing branch). ``th``/``Ptot`` are
    *outputs* of a solve, not inputs -- :func:`solve_richards_1d` fills them.
    """
    P: List[float]
    thi: List[float]
    T: List[float]
    th: List[float] = field(default_factory=list)
    Ptot: List[float] = field(default_factory=list)

    def copy(self) -> "RichardsState":
        return RichardsState(P=list(self.P), thi=list(self.thi), T=list(self.T),
                             th=list(self.th), Ptot=list(self.Ptot))


@dataclass
class RichardsResult:
    converged: bool
    iterations: int
    loss: float               # [mm], mass-balance residual over the step
    Vbottom: float            # [m3] volume drained through the bottom this step
    Vlat: float                # [m3] volume drained laterally this step
    state: RichardsState


# GEOtop: src/geotop/water.balance.cc:635-908
def solve_richards_1d(Dt: float, state: RichardsState, col: RichardsColumn,
                       params: RichardsParams, Pnet: float,
                       ET: Optional[Vec] = None,
                       max_iter_rec_K: int = 10) -> RichardsResult:
    """One implicit timestep of Richards' equation for the column ``col``.

    Newton-Raphson on total head ``H``, each step solved by BiCGSTAB with a
    Newton forcing term ``mu`` that tightens as the outer residual shrinks,
    each accepted step chosen by a non-monotone line search on step length
    ``lambda`` (see :func:`geotop_py.numerics.minimize_merit_function`).
    Returns a :class:`RichardsResult` with ``converged=False`` (and the state
    left unchanged) if the Newton loop exhausts its iteration budget, the
    line search stalls at ``min_lambda_wat`` too many times and
    ``exit_lambda_min_wat`` is set, or BiCGSTAB's own preconditioner hits a
    singular pivot.

    The conductivities (``Lx``, ``Klat``, ``Kbottom``) are assembled once from
    the starting head and then held fixed unless ``params.update_K`` says
    otherwise -- and even then only for the first ``max_iter_rec_K`` outer
    iterations. The Jacobian ignores ``dK/dH`` either way.
    """
    nl = col.nl
    N = nl + 1
    ET = list(ET) if ET is not None else [0.0] * (nl + 1)

    # H (like every vector here) is 1-based, index 0 unused.
    H1 = [0.0] * (N + 1)
    for i in range(1, N + 1):
        l = i - 1
        if l == 0 and Pnet > 0:
            cosine = math.cos(min(C.max_slope, col.slope_deg) * C.Pi / 180.0)
            H1[i] = max(0.0, state.P[l]) + Pnet / cosine + col.Z[l]
        else:
            H1[i] = state.P[l] + col.Z[l]

    Lx, Kbottom = find_matrix_K_1D(H1, state.thi, state.T, col)
    Klat = [0.0] * (nl + 1)
    for l in range(1, nl + 1):
        Klat[l] = sw.k_from_psi(C.jKl, H1[l + 1] - col.Z[l], state.thi[l], state.T[l],
                                l, col.pa, col.imp, col.k_to_ksat)

    f = find_f_1D(H1, state.P, state.thi, Klat, Kbottom, Pnet, ET, Dt, col)
    B = sparse.product_matrix_using_lower_part_by_vector_plus_vector(
        -1.0, f, H1, col.Li, col.Lp, Lx)

    res = num.norm_inf(B, 1, N)
    res00 = res
    epsilon = params.TolVWb + params.RelTolVWb * min(res00, math.sqrt(N))

    cont = 0
    cont_lambda_min = 0
    mu = params.TolCG
    res0 = [0.0, 0.0, 0.0]
    lambd = [0.0, 0.0, 0.0]
    res_prev: List[float] = []

    out = res <= min(epsilon, C.max_res_adm) or cont >= params.MaxiterTol

    while not out:
        cont += 1
        H0 = list(H1)
        dH = [0.0] * (N + 1)

        if cont == 1:
            mu = params.TolCG
        else:
            mu *= min(1.0, res / res0[0])
            if mu < 0.5 * epsilon / res:
                mu = 0.5 * epsilon / res

        df = find_dfdH_1D(H1, state.thi, Klat, Dt, col)

        iters, dH = sparse.BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
            mu, C.tol_min_GC, C.tol_max_GC, dH, B, df, col.Li, col.Lp, Lx)
        if iters == -1:
            return RichardsResult(False, cont, 0.0, 0.0, 0.0, state)

        m = min(cont, C.MM)
        res_prev = [res] + res_prev[:m - 1]
        res_av = max(res_prev[:m]) if res_prev else res
        cont2 = 0
        res0[0] = res

        out2 = False
        while not out2:
            cont2 += 1

            if cont2 == 1:
                lambd[0] = 1.0
            elif cont2 == 2:
                lambd[1] = lambd[0]
                res0[1] = res
                lambd[0] = C.thmax
            else:
                lambd[2] = lambd[1]
                res0[2] = res0[1]
                lambd[1] = lambd[0]
                res0[1] = res
                lambd[0] = num.minimize_merit_function(res0[0], lambd[1], res0[1],
                                                       lambd[2], res0[2])

            for i in range(1, N + 1):
                H1[i] = H0[i] + lambd[0] * dH[i]

            # GEOtop: src/geotop/water.balance.cc:805-806 (the guard)
            # GEOtop: src/geotop/parameters.cc:2194 (UpdateK, default 0)
            if params.update_K and cont <= max_iter_rec_K:
                Lx, Kbottom = find_matrix_K_1D(H1, state.thi, state.T, col)
                for l in range(1, nl + 1):
                    Klat[l] = sw.k_from_psi(C.jKl, H1[l + 1] - col.Z[l], state.thi[l],
                                            state.T[l], l, col.pa, col.imp, col.k_to_ksat)

            f = find_f_1D(H1, state.P, state.thi, Klat, Kbottom, Pnet, ET, Dt, col)
            B = sparse.product_matrix_using_lower_part_by_vector_plus_vector(
                -1.0, f, H1, col.Li, col.Lp, Lx)
            res = num.norm_inf(B, 1, N)

            out2 = res <= (1.0 - C.ni * lambd[0] * (1.0 - mu)) * res_av

            if lambd[0] <= params.min_lambda_wat:
                cont_lambda_min += 1
            if cont_lambda_min > params.max_times_min_lambda_wat:
                if params.exit_lambda_min_wat:
                    return RichardsResult(False, cont, 0.0, 0.0, 0.0, state)
                out2 = True
                cont_lambda_min = 0

        out = res <= min(epsilon, C.max_res_adm) or cont >= params.MaxiterTol

    if res > epsilon:
        return RichardsResult(False, cont, 0.0, 0.0, 0.0, state)

    loss = num.norm_1(B, 1, N) * Dt / col.area

    new_state = state.copy()
    new_state.th = [0.0] * (nl + 1)
    new_state.Ptot = [0.0] * (nl + 1)
    Vbottom = 0.0
    Vlat = 0.0

    for i in range(1, N + 1):
        l = i - 1
        new_state.P[l] = H1[i] - col.Z[l]

        if l > 0:
            th = sw.theta_from_psi(new_state.P[l], state.thi[l], l, col.pa, C.PsiMin)
            Ptot = sw.psi_from_theta(th + state.thi[l], 0.0, l, col.pa, C.PsiMin)
            th = min(th, col.pa[C.jsat][l] - state.thi[l])
            new_state.th[l] = th
            new_state.Ptot[l] = Ptot

        if l == nl:
            Vbottom += col.area * Kbottom * 1.0e-3 * Dt

        if _lateral_drainage_active(l, H1[i], col):
            dz = col.dz[l]
            ds = math.sqrt(col.area)
            dn = ds
            dD = 0.5 * 1.0e3 * ds
            Vlat += Dt * (dn * dz * 1.0e-3) * 1.0e-3 * Klat[l] * \
                col.free_drainage_lateral * (H1[i] - col.Z[l]) / dD

    return RichardsResult(True, cont, loss, Vbottom, Vlat, new_state)
