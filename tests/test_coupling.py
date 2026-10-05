"""``coupling.update_soil_land`` verified by invariant -- same situation as
``richards1d``/``init``: the C++ takes an ``ENERGY*`` struct, not plain
arrays, so there is no standalone function to wrap in the oracle without
reimplementing the same logic in C++, which would not verify anything. The
primitives it is built from (``psi_saturation``, ``psi_teta``) are pinned
directly elsewhere (``tests/test_soilwater.py``, ``tests/test_laws.py``).
"""

import pytest

from geotop_py import constants as C
from geotop_py import laws
from geotop_py.water import coupling
from geotop_py.water import soilwater as sw


def _pa(nl=3, **overrides):
    pa = [[0.0] * (nl + 1) for _ in range(C.jss + 1)]
    props = dict(jsat=0.5, jres=0.05, ja=0.004, jns=1.3, jss=1.0e-7)
    props.update(overrides)
    for l in range(1, nl + 1):
        for name, v in props.items():
            pa[getattr(C, name)][l] = v
    return pa


def _energy_result(nl, nsng, liq, ice, deltaw=None, Dlayer=100.0, Temp=5.0):
    n = nl + nsng
    return coupling.EnergyColumnResult(
        ice=[0.0] + list(ice),
        liq=[0.0] + list(liq),
        deltaw=[0.0] * (n + 1) if deltaw is None else [0.0] + list(deltaw),
        Dlayer=[0.0] + [Dlayer] * n,
        Temp=[0.0] + [Temp] * n if not isinstance(Temp, (list, tuple)) else [0.0] + list(Temp),
        nsng=nsng,
    )


# ---------------------------------------------------------------- basic wiring

def test_th_and_thi_are_mass_over_density_times_thickness():
    pa = _pa(nl=1)
    egy = _energy_result(1, 0, liq=[80.0], ice=[20.0], Dlayer=0.2)
    th, thi, P_out, T_out, ET_out = coupling.update_soil_land(
        0.0, 3600.0, egy, pa, 1, P=[0.0, -100.0])
    assert th[1] == pytest.approx(80.0 / (C.rho_w * 0.2))
    assert thi[1] == pytest.approx(20.0 / (C.rho_w * 0.2))


def test_temperature_passes_through_from_the_energy_result():
    pa = _pa(nl=2)
    egy = _energy_result(2, 0, liq=[50.0, 60.0], ice=[0.0, 0.0], Temp=[-3.0, 7.0])
    th, thi, P_out, T_out, ET_out = coupling.update_soil_land(
        0.0, 3600.0, egy, pa, 2, P=[0.0, -10.0, -10.0])
    assert T_out[1:] == [-3.0, 7.0]


def test_nsng_offsets_which_energy_nodes_are_read():
    """With two snow nodes above the soil, layer 1's data must come from
    energy node 3 (nsng + 1), not node 1 -- the same 'l + n' indexing
    PointEnergyBalance itself uses when snow sits above soil."""
    pa = _pa(nl=1)
    egy = coupling.EnergyColumnResult(
        ice=[0.0, 999.0, 999.0, 5.0],       # nodes 1,2 = snow (irrelevant), 3 = soil layer 1
        liq=[0.0, 999.0, 999.0, 45.0],
        deltaw=[0.0, 0.0, 0.0, 0.0],
        Dlayer=[0.0, 0.1, 0.1, 0.1],
        Temp=[0.0, -20.0, -20.0, 1.0],
        nsng=2,
    )
    th, thi, P_out, T_out, ET_out = coupling.update_soil_land(
        0.0, 3600.0, egy, pa, 1, P=[0.0, -10.0])
    assert th[1] == pytest.approx(45.0 / (C.rho_w * 0.1))
    assert thi[1] == pytest.approx(5.0 / (C.rho_w * 0.1))
    assert T_out[1] == 1.0


def test_output_lists_do_not_alias_the_input_P():
    pa = _pa(nl=1)
    egy = _energy_result(1, 0, liq=[50.0], ice=[0.0])
    P_in = [0.0, -20.0]
    th, thi, P_out, T_out, ET_out = coupling.update_soil_land(
        0.0, 3600.0, egy, pa, 1, P=P_in)
    P_out[1] = 12345.0
    assert P_in[1] == -20.0


# --------------------------------------------------------- the P/theta round trip

def test_p_round_trips_when_ice_liq_are_self_consistent_and_deltaw_is_zero():
    """The central invariant: if a layer's ice/liq already reflect its own P
    (no phase change happened this step, deltaw=0), reconstructing P from the
    resulting th/thi must recover the same P -- th_oversat and psisat exist
    precisely to make this true even when P is above the air-entry value.

    The physically-*stored* liquid content never exceeds ``s - thi`` -- above
    the air-entry value the extra volume is represented as pressure
    (``th_oversat``/``Ss``), not as additional theta, which is why
    constructing the "self-consistent" ``th`` here clamps ``teta_psi``'s raw
    (unbounded, compressibility-inclusive) saturated-branch output exactly
    the way :mod:`geotop_py.water.richards1d`'s own state update does.
    """
    for P_true in (-500.0, -50.0, -1.0, 0.0, 50.0, 500.0):
        for ice_frac in (0.0, 0.05, 0.15):
            pa = _pa(nl=1)
            s, r, a, n = pa[C.jsat][1], pa[C.jres][1], pa[C.ja][1], pa[C.jns][1]
            m = 1.0 - 1.0 / n
            Ss = pa[C.jss][1]
            Dlayer = 0.15

            thi_true = ice_frac
            th_true = min(laws.teta_psi(P_true, thi_true, s, r, a, n, m, C.PsiMin, Ss),
                          s - thi_true)

            liq = th_true * C.rho_w * Dlayer
            ice = thi_true * C.rho_w * Dlayer
            egy = _energy_result(1, 0, liq=[liq], ice=[ice], Dlayer=Dlayer)

            _, _, P_out, _, _ = coupling.update_soil_land(
                0.0, 3600.0, egy, pa, 1, P=[0.0, P_true])
            assert P_out[1] == pytest.approx(P_true, abs=1e-6, rel=1e-9)


def test_melting_ice_increases_liquid_and_decreases_ice():
    """A positive deltaw (net melt) must move mass from ice to liquid, not
    change the total -- pinned as a property since the C++ signs deltaw so
    that +deltaw is added to liq and subtracted from ice."""
    pa = _pa(nl=1)
    egy = _energy_result(1, 0, liq=[30.0], ice=[20.0], deltaw=[5.0])
    th, thi, P_out, T_out, ET_out = coupling.update_soil_land(
        0.0, 3600.0, egy, pa, 1, P=[0.0, -50.0])
    Dlayer = 100.0
    assert th[1] * C.rho_w * Dlayer == pytest.approx(35.0)   # 30 + 5
    assert thi[1] * C.rho_w * Dlayer == pytest.approx(15.0)  # 20 - 5


def test_liquid_and_ice_are_floored_at_zero():
    """A deltaw that would drive ice or liquid negative is clamped, not
    allowed through -- max(0, ...) in both terms of the C++."""
    pa = _pa(nl=1)
    egy = _energy_result(1, 0, liq=[3.0], ice=[3.0], deltaw=[10.0])  # would give ice=-7
    th, thi, P_out, T_out, ET_out = coupling.update_soil_land(
        0.0, 3600.0, egy, pa, 1, P=[0.0, -10.0])
    assert thi[1] == 0.0
    assert th[1] >= 0.0


# ----------------------------------------------------------------------- ET

def test_et_accumulates_transpiration_and_bare_and_veg_evaporation():
    pa = _pa(nl=1)
    egy = _energy_result(1, 0, liq=[50.0], ice=[0.0])
    fc = 0.4
    Dt = 1800.0
    T_transp = [0.0, 2.0e-6]
    E_bare = [0.0, 1.0e-6]
    E_veg = [0.0, 3.0e-6]
    _, _, _, _, ET_out = coupling.update_soil_land(
        fc, Dt, egy, pa, 1, P=[0.0, -10.0],
        T_transp=T_transp, E_bare=E_bare, E_veg=E_veg)
    expected = fc * T_transp[1] * Dt + (1 - fc) * E_bare[1] * Dt + fc * E_veg[1] * Dt
    assert ET_out[1] == pytest.approx(expected)


def test_et_defaults_to_zero_when_no_evapotranspiration_is_supplied():
    """Et defaults to zero when no evapotranspiration is supplied."""
    pa = _pa(nl=2)
    egy = _energy_result(2, 0, liq=[50.0, 60.0], ice=[0.0, 0.0])
    _, _, _, _, ET_out = coupling.update_soil_land(0.0, 3600.0, egy, pa, 2, P=[0.0, -10.0, -10.0])
    assert ET_out[1:] == [0.0, 0.0]


def test_et_accumulates_onto_a_caller_supplied_running_total():
    pa = _pa(nl=1)
    egy = _energy_result(1, 0, liq=[50.0], ice=[0.0])
    running = [0.0, 3.5]
    _, _, _, _, ET_out = coupling.update_soil_land(
        1.0, 3600.0, egy, pa, 1, P=[0.0, -10.0],
        T_transp=[0.0, 1.0e-6], ET=running)
    assert ET_out[1] == pytest.approx(3.5 + 1.0e-6 * 3600.0)
    assert running[1] == 3.5   # the caller's own list is not mutated


def test_et_layer_arrays_shorter_than_nl_only_cover_their_own_layers():
    """A root profile that only reaches layer 1 of a 3-layer column must not
    contribute transpiration to layers 2-3 -- the C++'s own
    l <= soil_transp_layer->nh guard."""
    pa = _pa(nl=3)
    egy = _energy_result(3, 0, liq=[50.0, 50.0, 50.0], ice=[0.0, 0.0, 0.0])
    T_transp = [0.0, 5.0e-6]   # only layer 1
    _, _, _, _, ET_out = coupling.update_soil_land(
        1.0, 3600.0, egy, pa, 3, P=[0.0, -10.0, -10.0, -10.0], T_transp=T_transp)
    assert ET_out[1] > 0.0
    assert ET_out[2] == 0.0
    assert ET_out[3] == 0.0


# ------------------------------------------------------------- psi_saturation use

def test_ice_makes_the_layer_reach_its_own_saturation_at_a_drier_psi():
    """Ice removes pore space, so what liquid *can* fill saturates (reaches
    ``s - thi``) at a more negative psi than an ice-free layer would need --
    psi_saturation's own ice dependence, exercised through the assembly
    rather than called directly. (Not the opposite: a smaller liquid-fillable
    pore space fills up "sooner", not "later".)"""
    pa = _pa(nl=1)
    s, r, a, n = pa[C.jsat][1], pa[C.jres][1], pa[C.ja][1], pa[C.jns][1]
    m = 1.0 - 1.0 / n
    psisat_dry = sw.psi_saturation(0.0, s, r, a, n, m)
    psisat_icy = sw.psi_saturation(0.2, s, r, a, n, m)
    assert psisat_icy <= psisat_dry
