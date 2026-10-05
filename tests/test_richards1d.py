"""Check Richards assembly against the oracle and the solver by invariants.

The full solver is exercised through conservation, residual checks and the
end-to-end cases; its internal iteration trajectory is not oracle-pinned.
"""


import pytest

from geotop_py import constants as C
from geotop_py.water import richards1d as r1d
from geotop_py.water import soilwater as sw

try:
    from geotop_py import _cxx
    HAVE_ORACLE = True
except FileNotFoundError:
    HAVE_ORACLE = False

needs_oracle = pytest.mark.skipif(not HAVE_ORACLE, reason="oracle not built")


def _uniform_column(nl=4, dz_val=100.0, slope_deg=0.0, **overrides):
    """A homogeneous sandy-loam-ish column, ``nl`` layers of ``dz_val`` mm."""
    dz = [0.0] + [dz_val] * nl
    Z = r1d.node_depths(dz, slope_deg)
    pa = [[0.0] * (nl + 1) for _ in range(C.jss + 1)]
    props = dict(jsat=0.5, jres=0.05, ja=0.004, jns=1.3, jss=1.0e-7,
                jKn=1.0e-4, jKl=1.0e-4, jv=0.5)
    props.update(overrides)
    for l in range(1, nl + 1):
        pa[C.jdz][l] = dz[l]
        for name, v in props.items():
            pa[getattr(C, name)][l] = v
    col = r1d.RichardsColumn(dz=dz, Z=Z, pa=pa, nl=nl, area=1.0,
                             slope_deg=slope_deg, imp=7.0, k_to_ksat=1.0e-4)
    return col, pa


def _hydrostatic_state(col, nl, thi_val=0.0, T_val=5.0, water_table_layer=None):
    """Equilibrium profile: psi(z) = Zwt - z, water table at the bottom node
    (or at ``col.Z[water_table_layer]`` if given)."""
    Zwt = col.Z[water_table_layer if water_table_layer is not None else nl]
    P = [Zwt - col.Z[l] for l in range(nl + 1)]
    thi = [thi_val] * (nl + 1)
    T = [T_val] * (nl + 1)
    return r1d.RichardsState(P=P, thi=thi, T=T)


def _total_storage(state, result_state, pa, dz, nl):
    """Column water volume [mm], surface ponding plus every soil layer, at
    the state a solve started from and the state it produced."""
    def storage(st, th):
        surf = max(0.0, st.P[0])
        soil = sum(th[l] * dz[l] for l in range(1, nl + 1)) if th else \
            sum(sw.theta_from_psi(st.P[l], st.thi[l], l, pa, C.PsiMin) * dz[l]
                for l in range(1, nl + 1))
        return surf + soil

    V0 = storage(state, None)
    V1 = storage(result_state, result_state.th)
    return V0, V1


# ------------------------------------------------------------------ topology

def test_chain_topology_matches_the_fixed_point_sim_layout():
    """The exact construction ``input.cc``'s ``point_sim==1`` branch performs:
    Li(l)=l+1, Lp(l)=l for l=1..n, Lp(n+1)=n -- pinned by direct comparison
    with that formula, since there is no runtime oracle call for it (it is a
    one-time layout decision the model makes at startup, not a per-step law)."""
    for n in (1, 2, 5, 20):
        Li, Lp = r1d.chain_topology(n)
        assert Li == [0] + [l + 1 for l in range(1, n + 1)]
        assert Lp == [0] + list(range(1, n + 1)) + [n]
        assert len(Li) - 1 == n
        assert len(Lp) - 1 == n + 1


def test_chain_topology_is_what_sparse_expects():
    """Round-trip: the topology this module builds is directly usable by
    sparse.py's own product functions without raising "not strictly lower
    triangular"."""
    from geotop_py.water import sparse
    n = 6
    Li, Lp = r1d.chain_topology(n)
    Lx = [0.0] + [1.0] * n
    x = [0.0] + list(range(1, n + 2))
    # must not raise
    sparse.product_using_only_strict_lower_diagonal_part(x, Li, Lp, Lx)


# --------------------------------------------------------------- node depths

def test_node_depths_surface_is_zero():
    dz = [0.0, 100.0, 200.0, 50.0]
    Z = r1d.node_depths(dz, 0.0)
    assert Z[0] == 0.0


def test_node_depths_are_layer_midpoints_on_flat_ground():
    dz = [0.0, 100.0, 200.0]
    Z = r1d.node_depths(dz, 0.0)
    assert Z[1] == pytest.approx(-50.0)          # centre of [0, 100]
    assert Z[2] == pytest.approx(-100.0 - 100.0)  # centre of [100, 300]


def test_node_depths_shrink_with_slope():
    """The vertical projection cos(slope) makes every depth shallower on a
    slope than on flat ground, for the same along-layer thickness."""
    dz = [0.0, 100.0, 100.0]
    flat = r1d.node_depths(dz, 0.0)
    sloped = r1d.node_depths(dz, 30.0)
    for l in (1, 2):
        assert abs(sloped[l]) < abs(flat[l])


def test_node_depths_are_strictly_decreasing():
    dz = [0.0] + [50.0] * 6
    Z = r1d.node_depths(dz, 10.0)
    assert all(Z[l] > Z[l + 1] for l in range(0, 6))


# ------------------------------------------------------- find_matrix_K_1D

def test_find_matrix_K_1D_bottom_conductivity_is_zero_without_free_drainage():
    col, pa = _uniform_column(nl=3)
    col.free_drainage_bottom = False
    state = _hydrostatic_state(col, 3)
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(4)]
    Lx, Kbottom = r1d.find_matrix_K_1D(H, state.thi, state.T, col)
    assert Kbottom == 0.0


def test_find_matrix_K_1D_bottom_conductivity_is_positive_with_free_drainage():
    col, pa = _uniform_column(nl=3)
    col.free_drainage_bottom = True
    state = _hydrostatic_state(col, 3)
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(4)]
    Lx, Kbottom = r1d.find_matrix_K_1D(H, state.thi, state.T, col)
    assert Kbottom > 0.0


def test_find_matrix_K_1D_produces_exactly_nl_negative_couplings():
    """Every Lx entry is -area*k/dD with k, area, dD all positive -- so
    strictly negative -- and there are exactly nl of them (one per link in
    the chain), matching sparse.py's own topology."""
    col, pa = _uniform_column(nl=5)
    state = _hydrostatic_state(col, 5)
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(6)]
    Lx, _ = r1d.find_matrix_K_1D(H, state.thi, state.T, col)
    assert len(Lx) - 1 == 5
    assert all(v < 0.0 for v in Lx[1:])


def test_find_matrix_K_1D_conductivity_is_capped_by_the_drier_neighbour():
    """A very dry layer 2 next to a saturated layer 1 must not let flux exceed
    what the dry layer's own saturated conductivity would carry -- the
    kmax/kmaxn capping in the C++, pinned as a property rather than a number
    since it depends on the whole van Genuchten closure."""
    col, pa = _uniform_column(nl=2)
    state = _hydrostatic_state(col, 2)
    # force layer 2 much drier than layer 1 by giing it a tiny saturated Kn
    pa[C.jKn][2] = 1.0e-8
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(3)]
    Lx, _ = r1d.find_matrix_K_1D(H, state.thi, state.T, col)
    # the single coupling (layer1<->layer2) must be bottlenecked by layer 2
    assert abs(Lx[2]) <= col.area * pa[C.jKn][2] / (0.5 * col.dz[1] + 0.5 * col.dz[2]) * 1.0001


# ------------------------------------------------------------ find_dfdH_1D

def test_find_dfdH_1D_surface_capacity_is_zero_when_dry():
    col, pa = _uniform_column(nl=2)
    state = _hydrostatic_state(col, 2)
    assert state.P[0] < 0.0        # dry surface by construction (WT at bottom)
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(3)]
    Klat = [0.0] * (col.nl + 1)
    df = r1d.find_dfdH_1D(H, state.thi, Klat, 3600.0, col)
    assert df[1] == 0.0            # node 1 == surface (i=1, l=0)


def test_find_dfdH_1D_surface_capacity_is_positive_when_ponded():
    col, pa = _uniform_column(nl=2)
    state = _hydrostatic_state(col, 2)
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(3)]
    H[1] = col.Z[0] + 5.0   # pond 5mm at the surface
    Klat = [0.0] * (col.nl + 1)
    df = r1d.find_dfdH_1D(H, state.thi, Klat, 3600.0, col)
    assert df[1] > 0.0


def test_find_dfdH_1D_soil_capacity_matches_dteta_dpsi_directly():
    col, pa = _uniform_column(nl=1)
    state = _hydrostatic_state(col, 1)
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(2)]
    Klat = [0.0] * (col.nl + 1)
    Dt = 1800.0
    df = r1d.find_dfdH_1D(H, state.thi, Klat, Dt, col)
    psi1 = H[2] - col.Z[1]
    expected = sw.dteta_dpsi(psi1, 0.0, pa[C.jsat][1], pa[C.jres][1], pa[C.ja][1],
                             pa[C.jns][1], 1.0 - 1.0 / pa[C.jns][1], C.PsiMin,
                             pa[C.jss][1]) * col.area * col.dz[1] / Dt
    assert df[2] == pytest.approx(expected)


# --------------------------------------------------------------- find_f_1D

def test_find_f_1D_is_zero_at_a_true_hydrostatic_equilibrium():
    """No bottom/lateral drainage, no ET, no Pnet: the residual at the
    equilibrium profile itself must be exactly zero (the same volume both
    steps, nothing added or removed)."""
    col, pa = _uniform_column(nl=4)
    state = _hydrostatic_state(col, 4)
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(5)]
    Klat = [0.0] * (col.nl + 1)
    ET = [0.0] * (col.nl + 1)
    f = r1d.find_f_1D(H, state.P, state.thi, Klat, 0.0, 0.0, ET, 3600.0, col)
    assert all(v == 0.0 for v in f[1:])


def test_find_f_1D_surface_sink_is_minus_pnet_over_dt():
    col, pa = _uniform_column(nl=2)
    state = _hydrostatic_state(col, 2)
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(3)]
    Klat = [0.0] * (col.nl + 1)
    ET = [0.0] * (col.nl + 1)
    Dt = 3600.0
    Pnet = 7.0
    f = r1d.find_f_1D(H, state.P, state.thi, Klat, 0.0, Pnet, ET, Dt, col)
    f0 = r1d.find_f_1D(H, state.P, state.thi, Klat, 0.0, 0.0, ET, Dt, col)
    assert f[1] - f0[1] == pytest.approx(-col.area * Pnet / Dt)


def test_find_f_1D_bottom_drainage_adds_area_times_kbottom():
    col, pa = _uniform_column(nl=2)
    state = _hydrostatic_state(col, 2)
    H = [0.0] + [state.P[l] + col.Z[l] for l in range(3)]
    Klat = [0.0] * (col.nl + 1)
    ET = [0.0] * (col.nl + 1)
    Kbottom = 1.0e-5
    f_with = r1d.find_f_1D(H, state.P, state.thi, Klat, Kbottom, 0.0, ET, 3600.0, col)
    f_without = r1d.find_f_1D(H, state.P, state.thi, Klat, 0.0, 0.0, ET, 3600.0, col)
    assert f_with[col.nl + 1] - f_without[col.nl + 1] == pytest.approx(col.area * Kbottom)


# -------------------------------------------------------------- solve_richards_1d

def test_hydrostatic_equilibrium_is_a_fixed_point():
    """The single most important invariant: with no forcing, a column already
    at hydrostatic equilibrium must not move -- zero Newton iterations,
    exactly zero loss, state bit-identical to what it started from."""
    col, pa = _uniform_column(nl=4)
    state = _hydrostatic_state(col, 4)
    params = r1d.RichardsParams()
    result = r1d.solve_richards_1d(3600.0, state, col, params, Pnet=0.0)
    assert result.converged
    assert result.iterations == 0
    assert result.loss == 0.0
    assert result.state.P == state.P


@pytest.mark.parametrize("slope_deg", [0.0, 15.0, 45.0])
def test_hydrostatic_equilibrium_is_a_fixed_point_on_a_slope(slope_deg):
    col, pa = _uniform_column(nl=3, slope_deg=slope_deg)
    state = _hydrostatic_state(col, 3)
    params = r1d.RichardsParams()
    result = r1d.solve_richards_1d(3600.0, state, col, params, Pnet=0.0)
    assert result.converged
    assert result.iterations == 0


def test_infiltration_conserves_mass_to_newton_tolerance():
    """Rain in, minus the tiny leftover residual the Newton loop stopped at
    (``loss``), must equal the column's own total storage change -- a budget
    independent of anything richards1d.py computes internally, built only
    from the states it hands back (the conservation diagnostic)."""
    col, pa = _uniform_column(nl=4)
    state = _hydrostatic_state(col, 4)
    params = r1d.RichardsParams()
    Pnet = 5.0
    result = r1d.solve_richards_1d(3600.0, state, col, params, Pnet=Pnet)
    assert result.converged

    V0, V1 = _total_storage(state, result.state, pa, col.dz, col.nl)
    budget = Pnet - result.loss
    assert (V1 - V0) == pytest.approx(budget, abs=max(1e-3, 5.0 * result.loss))


def test_more_rain_infiltrates_more_water_not_less():
    col, pa = _uniform_column(nl=4)
    params = r1d.RichardsParams()

    state_a = _hydrostatic_state(col, 4)
    result_a = r1d.solve_richards_1d(3600.0, state_a, col, params, Pnet=2.0)
    state_b = _hydrostatic_state(col, 4)
    result_b = r1d.solve_richards_1d(3600.0, state_b, col, params, Pnet=8.0)

    V0, V1a = _total_storage(state_a, result_a.state, pa, col.dz, col.nl)
    _, V1b = _total_storage(state_b, result_b.state, pa, col.dz, col.nl)
    assert V1b - V0 > V1a - V0


def test_liquid_content_never_exceeds_saturation_minus_ice():
    """theta_from_psi's own saturation clamp (in find_f_1D) plus the explicit
    min() in the state-update step -- pinned as a property over a scenario
    designed to saturate the column (heavy rain, tight soil)."""
    col, pa = _uniform_column(nl=3, jKn=1.0e-7)   # tight soil: rain will pond/saturate
    state = _hydrostatic_state(col, 3)
    params = r1d.RichardsParams()
    result = r1d.solve_richards_1d(3600.0, state, col, params, Pnet=50.0)
    assert result.converged
    for l in range(1, col.nl + 1):
        assert result.state.th[l] <= pa[C.jsat][l] - result.state.thi[l] + 1e-12


def test_evapotranspiration_removes_water_like_negative_rain_at_that_layer():
    col, pa = _uniform_column(nl=3)
    params = r1d.RichardsParams()

    state_no_et = _hydrostatic_state(col, 3)
    result_no_et = r1d.solve_richards_1d(3600.0, state_no_et, col, params, Pnet=0.0)

    state_et = _hydrostatic_state(col, 3)
    ET = [0.0, 0.01, 0.0, 0.0]     # a small sink in layer 1, mm/s
    result_et = r1d.solve_richards_1d(3600.0, state_et, col, params, Pnet=0.0, ET=ET)

    V0, V1_no_et = _total_storage(state_no_et, result_no_et.state, pa, col.dz, col.nl)
    _, V1_et = _total_storage(state_et, result_et.state, pa, col.dz, col.nl)
    assert V1_et < V1_no_et


def test_free_drainage_bottom_removes_water_over_time():
    col, pa = _uniform_column(nl=3)
    col.free_drainage_bottom = True
    state = _hydrostatic_state(col, 3, water_table_layer=1)  # saturate lower layers
    params = r1d.RichardsParams()
    result = r1d.solve_richards_1d(3600.0, state, col, params, Pnet=0.0)
    assert result.converged
    assert result.Vbottom > 0.0


def test_richards_result_state_is_a_new_object_not_an_alias():
    """Richards result state is a new object not an alias."""
    col, pa = _uniform_column(nl=2)
    state = _hydrostatic_state(col, 2)
    original_P = list(state.P)
    params = r1d.RichardsParams()
    result = r1d.solve_richards_1d(3600.0, state, col, params, Pnet=3.0)
    assert state.P == original_P          # the input was not mutated
    assert result.state is not state


@needs_oracle
def test_richards_params_keywords_are_real():
    known = set(_cxx.KEYWORDS_NUM)
    for name in ("RichardTol", "RichardMaxIter", "RichardInitForc",
                "MinLambdaWater", "MaxTimesMinLambdaWater", "ExitMinLambdaWater"):
        assert name in known, name


def test_richards_params_defaults_match_parameters_cc():
    p = r1d.RichardsParams()
    assert p.TolVWb == 1.0e-6
    assert p.RelTolVWb == 1.0e-10
    assert p.MaxiterTol == 100
    assert p.TolCG == 0.01
    assert p.min_lambda_wat == 1.0e-7
    assert p.max_times_min_lambda_wat == 0
    assert p.exit_lambda_min_wat is True


def test_solve_converges_within_the_max_iter_budget_for_a_moderate_step():
    col, pa = _uniform_column(nl=6)
    state = _hydrostatic_state(col, 6)
    params = r1d.RichardsParams()
    result = r1d.solve_richards_1d(900.0, state, col, params, Pnet=3.0)
    assert result.converged
    assert 0 <= result.iterations <= params.MaxiterTol


def test_a_pathologically_tiny_iteration_budget_fails_to_converge():
    col, pa = _uniform_column(nl=6)
    state = _hydrostatic_state(col, 6)
    params = r1d.RichardsParams(MaxiterTol=0, TolVWb=1e-30)
    result = r1d.solve_richards_1d(900.0, state, col, params, Pnet=10.0)
    # MaxiterTol=0 with an unreachable tolerance: the very first check must
    # already fail (res > epsilon), so the loop cannot run at all.
    assert not result.converged
