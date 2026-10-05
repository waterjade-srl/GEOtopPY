"""Regression checks for init."""

import os

import pytest

from geotop_py import constants as C
from geotop_py import laws
from geotop_py.io import parfile, soil
from geotop_py.water import init
from geotop_py.water.richards1d import node_depths
from tools.paths import REFERENCE_1D


def _cases():
    if not os.path.isdir(REFERENCE_1D):
        return []
    return [c for c in sorted(os.listdir(REFERENCE_1D))
            if os.path.exists(os.path.join(REFERENCE_1D, c, "geotop.inpts"))]


CASES = _cases()
needs_reference = pytest.mark.skipif(
    not CASES, reason=f"reference cases not found under {REFERENCE_1D}")


def _pa(nl=4, **overrides):
    pa = [[0.0] * (nl + 1) for _ in range(C.jss + 1)]
    props = dict(jsat=0.5, jres=0.05, ja=0.004, jns=1.3, jss=1.0e-7, jT=5.0)
    props.update(overrides)
    for l in range(1, nl + 1):
        pa[C.jdz][l] = 100.0
        for name, v in props.items():
            v_l = v[l - 1] if isinstance(v, (list, tuple)) else v
            pa[getattr(C, name)][l] = v_l
    return pa


# ------------------------------------------------------------- hydrostatic_pressure

def test_hydrostatic_pressure_is_zero_at_the_water_table():
    """By construction the profile crosses psi=0 exactly at the water table's
    own depth -- checked with a table placed at a node's exact depth."""
    pa = _pa(nl=5)
    dz = [0.0] + [pa[C.jdz][l] for l in range(1, 6)]
    Z = node_depths(dz, 0.0)
    wtd = -Z[3]     # water table at node 3's own depth
    P = init.hydrostatic_pressure(pa, 5, wtd, 0.0)
    assert P[3] == pytest.approx(0.0, abs=1e-9)


def test_hydrostatic_pressure_increases_with_depth():
    """Below the water table (deeper nodes) psi is more positive; above it,
    more negative -- monotonic in depth throughout, since it's linear in Z."""
    pa = _pa(nl=6)
    P = init.hydrostatic_pressure(pa, 6, 300.0, 0.0)
    assert all(P[l] < P[l + 1] for l in range(0, 6))


def test_hydrostatic_pressure_matches_the_direct_formula():
    pa = _pa(nl=3)
    P = init.hydrostatic_pressure(pa, 3, 250.0, 0.0)
    dz = [0.0] + [pa[C.jdz][l] for l in range(1, 4)]
    Z = node_depths(dz, 0.0)
    expected = [-250.0 - Z[l] for l in range(4)]
    assert P == pytest.approx(expected)


def test_hydrostatic_pressure_shrinks_with_slope():
    """The same water table depth gives a shallower hydrostatic gradient on a
    slope, because dz is projected through cos(slope) just like node_depths."""
    pa = _pa(nl=3)
    flat = init.hydrostatic_pressure(pa, 3, 300.0, 0.0)
    sloped = init.hydrostatic_pressure(pa, 3, 300.0, 30.0)
    assert abs(sloped[3] - sloped[0]) < abs(flat[3] - flat[0])


# ------------------------------------------------------------------ layer_pressures

def test_layer_pressures_takes_jpsi_directly():
    pa = _pa(nl=3, jpsi=[-100.0, -50.0, 20.0])
    P = init.layer_pressures(pa, 3)
    assert P[1:] == [-100.0, -50.0, 20.0]


def test_layer_pressures_does_not_touch_node_0():
    pa = _pa(nl=2, jpsi=[-10.0, -10.0])
    P = init.layer_pressures(pa, 2)
    assert P[0] == 0.0


# --------------------------------------------------------- apply_freezing_equilibrium

def test_no_ice_above_freezing():
    pa = _pa(nl=3, jT=5.0)
    P = [0.0, -100.0, -200.0, -300.0]
    T = [0.0, 5.0, 5.0, 5.0]
    result = init.apply_freezing_equilibrium(P, T, pa, 3)
    assert result.thi[1:] == [0.0, 0.0, 0.0]
    assert result.P == pytest.approx(P)  # unchanged: no freezing correction applied


def test_ice_appears_below_freezing():
    pa = _pa(nl=1, jT=-5.0)
    P = [0.0, -50.0]
    T = [0.0, -5.0]
    result = init.apply_freezing_equilibrium(P, T, pa, 1)
    assert result.thi[1] > 0.0


def test_ice_content_never_negative():
    """The C++ clamps thi >= 0 explicitly (a layer whose ice-free theta is
    already below the freezing-equilibrium theta would otherwise go
    negative) -- exercised with a psi so dry the clamp must bind."""
    pa = _pa(nl=1, jT=-10.0, jres=0.05, jsat=0.5)
    P = [0.0, -1.0e6]      # extremely dry
    T = [0.0, -10.0]
    result = init.apply_freezing_equilibrium(P, T, pa, 1)
    assert result.thi[1] >= 0.0


def test_liquid_plus_ice_is_conserved_by_the_freezing_split():
    """Liquid plus ice is conserved by the freezing split."""
    pa = _pa(nl=1, jT=-3.0)
    P = [0.0, -80.0]
    T = [0.0, -3.0]
    th_ice_free = laws.teta_psi(P[1], 0.0, pa[C.jsat][1], pa[C.jres][1],
                                pa[C.ja][1], pa[C.jns][1],
                                1.0 - 1.0 / pa[C.jns][1], C.PsiMin, pa[C.jss][1])
    th_oversat = max(P[1], 0.0) * pa[C.jss][1]
    result = init.apply_freezing_equilibrium(P, T, pa, 1)
    assert result.th[1] + result.thi[1] == pytest.approx(th_ice_free - th_oversat)


def test_ptot_is_the_pre_freezing_potential():
    pa = _pa(nl=2, jT=[-2.0, 8.0])
    P = [0.0, -60.0, -40.0]
    T = [0.0, -2.0, 8.0]
    result = init.apply_freezing_equilibrium(P, T, pa, 2)
    assert result.Ptot[1:] == pytest.approx(P[1:])


def test_freezing_correction_at_exactly_zero_degrees_uses_psif_zero():
    """Psif(0) == 0 (laws.py: only T < 0 gives a nonzero freezing-point
    depression), so at T==Tfreezing the branch still runs (<=) but the
    freezing-equilibrium theta it subtracts is evaluated at psi=0 -- pinned
    as a boundary case since it is easy to get the </<= distinction wrong."""
    pa = _pa(nl=1, jT=0.0)
    P = [0.0, -50.0]
    T = [0.0, 0.0]
    result = init.apply_freezing_equilibrium(P, T, pa, 1)
    th_at_zero = laws.teta_psi(0.0, 0.0, pa[C.jsat][1], pa[C.jres][1],
                               pa[C.ja][1], pa[C.jns][1],
                               1.0 - 1.0 / pa[C.jns][1], C.PsiMin, pa[C.jss][1])
    th_ice_free = laws.teta_psi(P[1], 0.0, pa[C.jsat][1], pa[C.jres][1],
                                pa[C.ja][1], pa[C.jns][1],
                                1.0 - 1.0 / pa[C.jns][1], C.PsiMin, pa[C.jss][1])
    assert result.thi[1] == pytest.approx(max(0.0, th_ice_free - th_at_zero))


# ------------------------------------------------------------------ initial_soil_state

def test_initial_soil_state_selects_hydrostatic_when_water_table_known():
    pa = _pa(nl=3, jpsi=parfile.NUMBER_NOVALUE)
    result = init.initial_soil_state(pa, 3, 250.0, 0.0)
    expected = init.hydrostatic_pressure(pa, 3, 250.0, 0.0)
    assert result.Ptot[1:] == pytest.approx(expected[1:])


def test_initial_soil_state_selects_layer_pressures_when_water_table_novalue():
    pa = _pa(nl=3, jpsi=[-10.0, -20.0, -30.0])
    result = init.initial_soil_state(pa, 3, parfile.NUMBER_NOVALUE, 0.0)
    assert result.Ptot[1:] == pytest.approx([-10.0, -20.0, -30.0])


def test_as_richards_state_round_trips_into_richards1d():
    from geotop_py.water.richards1d import RichardsState
    pa = _pa(nl=3)
    result = init.initial_soil_state(pa, 3, 250.0, 0.0)
    state = result.as_richards_state()
    assert isinstance(state, RichardsState)
    assert state.P == result.P
    assert state.thi == result.thi


# ------------------------------------------------------ real reference cases

@needs_reference
@pytest.mark.parametrize("case", CASES)
def test_real_cases_produce_physically_plausible_initial_state(case):
    directory = os.path.join(REFERENCE_1D, case)
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    sp = soil.load(pf, directory)
    result = init.initial_soil_state(sp.pa, sp.nlayers, sp.init_water_table_depth, 0.0)

    for l in range(1, sp.nlayers + 1):
        sat = sp.pa[C.jsat][l]
        assert not math_isnan(result.th[l])
        assert not math_isnan(result.P[l])
        assert 0.0 <= result.thi[l] <= sat
        # liquid content is bounded by the pore space ice does not occupy,
        # not by (sat - res): res is teta_psi's floor as psi -> -inf, not a
        # cap on how much of [res, sat] a moderately dry layer can reach.
        assert -1.0e-6 <= result.th[l] <= sat - result.thi[l] + 1.0e-6


def math_isnan(x):
    return x != x


@needs_reference
def test_below_the_water_table_the_column_is_saturated():
    """PureDrainage's water table sits well inside the column (5000mm, 7
    layers of 1000mm): every layer below it must show liquid content at
    saturation (theta == jsat, the oversaturation term folded away by the
    subtraction in apply_freezing_equilibrium leaves exactly jsat when
    ice is zero and psi > 0)."""
    directory = os.path.join(REFERENCE_1D, "PureDrainage")
    pf = parfile.parse(os.path.join(directory, "geotop.inpts"))
    sp = soil.load(pf, directory)
    result = init.initial_soil_state(sp.pa, sp.nlayers, sp.init_water_table_depth, 0.0)
    for l in range(1, sp.nlayers + 1):
        if result.Ptot[l] > 0:
            assert result.th[l] == pytest.approx(sp.pa[C.jsat][l])
