"""Regression checks for richards coupling."""

import pytest

from geotop_py import constants as C
from geotop_py.energy import vegetation as veg
from geotop_py.energy.column import SoilLayer
from geotop_py.energy.surface import SurfaceState, SurfaceStatics
from geotop_py.meteo.step import Meteo
from geotop_py.point import step as point_step
from geotop_py.point.state import Column1D
from geotop_py.point.step import NominalStep, simulate_energy_balance, step_independent
from geotop_py.snow.state import SnowColumn
from geotop_py.water import richards1d as r1d
from geotop_py.water import soilwater as sw
from geotop_py.water.coupling import RichardsCoupling

NL = 3
DZ = 200.0  # mm


def _pa_and_richards_col(sat=0.4, res=0.05, alpha=0.004, n=1.3, ss=1.0e-7,
                         bc_depth_free_surface=0.0):
    dz = [0.0] + [DZ] * NL
    Z = r1d.node_depths(dz, slope_deg=0.0)
    pa = [[0.0] * (NL + 1) for _ in range(C.jss + 1)]
    props = dict(jsat=sat, jres=res, ja=alpha, jns=n, jss=ss,
                jKn=1.0e-5, jKl=1.0e-5, jv=0.5, jkt=2.5, jct=2.3e6)
    for l in range(1, NL + 1):
        pa[C.jdz][l] = dz[l]
        for name, v in props.items():
            pa[getattr(C, name)][l] = v
    rcol = r1d.RichardsColumn(dz=dz, Z=Z, pa=pa, nl=NL, area=1.0,
                              slope_deg=0.0, imp=7.0, k_to_ksat=1.0e-4,
                              free_drainage_bottom=False,
                              bc_depth_free_surface=bc_depth_free_surface)
    return pa, rcol


def _coupled_fixture(psi0=-500.0, T0=5.0, ponding_allowed=False):
    """A dry-ish, above-freezing, snow-free bare-soil column, EB and WB sides
    seeded from the *same* hydrostatic (th, thi) so the two representations
    agree at t=0."""
    # A free-surface boundary at 0 -- the default, and what every reference
    # case sets -- discards all surface ponding as runoff at the end of the
    # step. ``ponding_allowed`` moves that boundary 10 m up instead, which
    # closes the column: with no bottom drainage and no lateral drainage,
    # rain then has nowhere to go but storage.
    pa, rcol = _pa_and_richards_col(
        bc_depth_free_surface=-1.0e4 if ponding_allowed else 0.0)
    thi0 = 0.0
    th0 = sw.theta_from_psi(psi0, thi0, 1, pa, C.PsiMin)
    soil = [None] + [
        SoilLayer(sat=pa[C.jsat][l], res=pa[C.jres][l], alpha=pa[C.ja][l],
                 n=pa[C.jns][l], ss=pa[C.jss][l], kt=pa[C.jkt][l],
                 ct=pa[C.jct][l], th0=th0, thi0=thi0, Tstar=0.0)
        for l in range(1, NL + 1)]
    col = Column1D(
        snow=SnowColumn(max=1, lnum=0, type=0),
        soil=soil, soil_D=[0.0] + [DZ / 1000.0] * NL,
        soil_T=[0.0] + [T0] * NL,
        alpha_snow=1e5, snow_conductivity=3, slope=0.0,
        Tboundary=T0, Zboundary=1.0, Fboundary=0.0,
        max_weq_snow=5.0, maxSWE=1e10,
    )
    rstate = r1d.RichardsState(P=[0.0] + [psi0] * NL, thi=[0.0] * (NL + 1),
                               T=[0.0] + [T0] * NL)
    richards = RichardsCoupling(col=rcol, params=r1d.RichardsParams(), state=rstate)
    # soil_D_mm is the parameter table's own thickness row, in mm, indexed
    # by soil layer: Ye--Pielke's molecular resistance reads it directly.
    st = SurfaceStatics(lat=46.1, lon=11.1, ST=1.0, sky=1.0,
                        soil_D_mm=[DZ] * NL)
    state = SurfaceState()
    return col, state, st, richards


def _storage_mm(col, richards):
    """Total column water [mm]: surface ponding (node 0 of Richards' own
    state) plus every soil layer -- matching test_richards1d.py's own
    ``_total_storage`` helper. Omitting the ponding term would make this
    look like water vanished whenever a step infiltrates less than the full
    Pnet in one Dt, which unsaturated infiltration into fairly dry soil
    (psi0=-500mm here) routinely does."""
    ponding = max(0.0, richards.state.P[0])
    return ponding + sum(col.soil[l].th0 * DZ for l in range(1, NL + 1))


def test_richards_coupling_is_opt_in_default_none_untouched():
    col, state, st, richards = _coupled_fixture()
    m = Meteo(Ta=5.0, RH=0.7, P=850.0, wind=2.0, tau_cloud=1.0)
    out_default = step_independent(col.copy(), state.copy(), st, 900.0, m, 100.0, 100.0 + 900.0 / 86400.0)
    assert out_default.wb_converged is True
    assert out_default.wb_iterations == 0
    assert out_default.wb_loss == 0.0


def test_a_rain_pulse_shows_up_as_stored_soil_water():
    col, state, st, richards = _coupled_fixture(ponding_allowed=True)
    Dt = 900.0
    V0 = _storage_mm(col, richards)
    Prain_rate = 2.0e-3   # kg/(m2 s) -> mm/s of liquid rain
    m = Meteo(Ta=5.0, RH=0.7, P=850.0, wind=2.0, tau_cloud=1.0, Prain=Prain_rate * Dt)
    out = step_independent(col, state, st, Dt, m, 100.0, 100.0 + Dt / 86400.0,
                           richards=richards)
    assert out.wb_converged
    V1 = _storage_mm(col, richards)
    # No snow, no canopy, no drainage (free_drainage_bottom=False): with bare-
    # soil evaporation now wired into ET (E_bare), storage gain is Pnet minus
    # whatever molecular-diffusion-limited evaporation the layer array
    # estimates -- update_soil_land adds that term unscaled, exactly as
    # energy.balance.cc:2213 does, so it need not match the *actual* surface
    # Evap exactly. Bounded, not exact: strictly positive (still net wetting
    # at this RH/Ta), and not more than the rain itself (evaporation only
    # removes water here, never adds).
    gain = V1 - V0
    assert 0.0 < gain < Prain_rate * Dt
    assert gain == pytest.approx(Prain_rate * Dt, abs=0.05)


def test_the_next_energy_step_sees_richards_own_moisture_not_the_frozen_one():
    """Without `richards=`, soil moisture is exactly conserved step to step
    (the WaterBalance=0 behaviour). With it, a wetting step must leave a
    measurably different th0 behind -- proving the feedback path actually
    runs, not just that it doesn't crash."""
    col_frozen, state_frozen, st, _ = _coupled_fixture()
    col_wet, state_wet, _, richards = _coupled_fixture()
    Dt = 900.0
    m = Meteo(Ta=5.0, RH=0.7, P=850.0, wind=2.0, tau_cloud=1.0, Prain=1.0)

    step_independent(col_frozen, state_frozen, st, Dt, m, 100.0, 100.0 + Dt / 86400.0)
    step_independent(col_wet, state_wet, st, Dt, m, 100.0, 100.0 + Dt / 86400.0,
                     richards=richards)

    assert col_wet.soil[1].th0 != pytest.approx(col_frozen.soil[1].th0)
    assert col_wet.soil[1].th0 > col_frozen.soil[1].th0


def test_simulate_energy_balance_carries_richards_across_nominal_steps():
    """The time-loop wiring (not just a single step_independent call): a
    multi-step run with `richards=` must accumulate infiltrated water across
    committed sub-steps, and hand back the same `richards` object it was
    given (mutated), not a detached copy."""
    col, state, st, richards = _coupled_fixture(ponding_allowed=True)
    Dt = 900.0
    V0 = _storage_mm(col, richards)

    def meteo_at(JDb, JDe):
        return Meteo(Ta=5.0, RH=0.7, P=850.0, wind=2.0, tau_cloud=1.0, Prain=1.0e-3 * Dt)

    steps = [NominalStep(JD0=100.0 + i * Dt / 86400.0, Dt_nominal=Dt) for i in range(3)]
    col2, state2, richards2, outputs = simulate_energy_balance(
        col, state, st, min_Dt=1.0, steps=steps, meteo_at=meteo_at, richards=richards)

    assert len(outputs) == 3
    assert all(o.wb_converged for o in outputs)
    V1 = _storage_mm(col2, richards2)
    total_rain = 3 * 1.0e-3 * Dt
    # See the note in test_a_rain_pulse_shows_up_as_stored_soil_water: bare
    # evaporation is real ET now, so the gain is bounded by, not equal to,
    # the rain input.
    assert 0.0 < V1 - V0 < total_rain
    assert V1 - V0 == pytest.approx(total_rain, abs=0.1)


def test_surface_ponding_leaves_the_column_as_runoff():
    """With the free-surface boundary at its default depth of 0, a point
    column keeps no standing water: whatever the step leaves ponded on the
    surface node is gone by the time the next step starts.

    The soil here is tight enough (Ksat 1e-5 mm/s) that a 1.8 mm pulse over
    900 s cannot infiltrate, so the surface node really does pond during the
    solve -- the same column with the boundary moved up keeps that water and
    stores it, which is what makes this a test of the boundary and not of the
    infiltration rate.
    """
    Dt = 900.0
    Prain = 2.0e-3 * Dt
    m = Meteo(Ta=5.0, RH=0.7, P=850.0, wind=2.0, tau_cloud=1.0, Prain=Prain)

    col, state, st, richards = _coupled_fixture()
    V0 = _storage_mm(col, richards)
    out = step_independent(col, state, st, Dt, m, 100.0, 100.0 + Dt / 86400.0,
                           richards=richards)
    assert out.wb_converged
    assert richards.state.P[0] <= 0.0
    drained = _storage_mm(col, richards) - V0

    colp, statep, stp, richardsp = _coupled_fixture(ponding_allowed=True)
    V0p = _storage_mm(colp, richardsp)
    outp = step_independent(colp, statep, stp, Dt, m, 100.0, 100.0 + Dt / 86400.0,
                            richards=richardsp)
    assert outp.wb_converged
    assert richardsp.state.P[0] > 0.0        # the same step really does pond
    kept = _storage_mm(colp, richardsp) - V0p

    assert kept > 10.0 * drained
    assert kept == pytest.approx(Prain, abs=0.05)


def test_bare_soil_evaporation_actually_drains_the_column():
    """No rain, dry air, warm surface: the only thing that can move water is
    evaporation. If `E_bare` were not really wired into `ET`
    (`update_soil_land`'s sink term), storage would stay flat instead --
    this is the sign that distinguishes a merely-non-crashing wiring from
    one that actually feeds the evaporation array through."""
    col, state, st, richards = _coupled_fixture(psi0=-300.0, T0=15.0)
    V0 = _storage_mm(col, richards)
    Dt = 3600.0
    m = Meteo(Ta=15.0, RH=0.2, P=850.0, wind=3.0, tau_cloud=1.0)  # dry, no rain

    for _ in range(6):
        out = step_independent(col, state, st, Dt, m, 100.0, 100.0 + Dt / 86400.0,
                               richards=richards)
        assert out.wb_converged

    V1 = _storage_mm(col, richards)
    assert V1 < V0


# -- what reaches the soil surface: Pnet ------------------------------------

def _phases(col, state, st, richards, m, Dt=3600.0, JDb=100.5):
    """Run the energy and post-energy phases of one step, returning the
    energy phase, the canopy rain store between the two, and their outputs."""
    ep = point_step.energy_phase(col, state, st, Dt, m, JDb, JDb + Dt / 86400.0)
    assert ep.converged
    Wcrn = col.Wcrn
    out, water_input = point_step.post_energy_phase(col, state, Dt, m, ep, richards)
    return ep, Wcrn, out, water_input


def test_the_rain_a_buried_canopy_sheds_reaches_the_soil():
    """Snowfall that buries the canopy takes its fraction ``fc`` to zero: the
    rain held on it, ``fc * Wcrn`` per unit ground area, leaves the canopy
    store and must reach the soil surface with the rest of the net
    precipitation."""
    col, state, st, richards = _coupled_fixture(T0=1.0)
    # a shallow pack the canopy still stands out of (fc > 0), which the
    # snowfall of the step then buries
    col.snow = SnowColumn.from_layers([(40.0, 10.0, 0.0, -1.0)], max=5)
    col.Wcrn = 0.3
    st.vegpar = veg.VegParams(Hveg=200.0, z0thresveg=50.0, z0thresveg2=10.0,
                              LSAI=2.0, cf=0.8, decay0=2.5, expveg=1.0,
                              root=300.0, rs=60.0, R_vis=0.1, R_nir=0.3,
                              T_vis=0.05, T_nir=0.3, Ch=0.0, cd=0.1,
                              root_frac=[0.0, 1.0], n_transp=1)
    m = Meteo(Ta=-2.0, RH=0.9, P=850.0, wind=2.0, tau_cloud=0.5,
              Psnow=20.0, Prain=1.0)
    ep, Wcrn, out, water_input = _phases(col, state, st, richards, m)
    fc0 = ep.fc
    assert fc0 > 0.0 and Wcrn > 0.0
    assert out.depth > st.vegpar.z0thresveg          # buried: fc drops to 0
    assert col.Wcrn == 0.0
    shed = fc0 * Wcrn
    assert water_input.Pnet == out.Melt + out.Melt_glac + out.Prain_under + shed


def test_glacier_meltwater_reaches_the_soil():
    """Water leaving the base of the glacier drains to the soil surface, next
    to the snowpack's own."""
    col, state, st, richards = _coupled_fixture(T0=0.0)
    col.snow = SnowColumn(max=5, lnum=0, type=0)
    col.glac = SnowColumn.from_layers([(2000.0, 1600.0, 5.0, -0.5)], max=5)
    m = Meteo(Ta=15.0, RH=0.5, P=850.0, wind=3.0, tau_cloud=1.0, Prain=2.0)
    ep, _, out, water_input = _phases(col, state, st, richards, m, JDb=180.5)
    assert out.Melt_glac > 0.0
    assert water_input.Pnet == out.Melt + out.Melt_glac + out.Prain_under


def test_a_glacier_that_melts_out_is_solved_again_as_bare_soil():
    """The last glacier node left with no ice is merged away (``sux=-4``) and
    the step is solved again on the bare column, whose soil evaporation then
    reads the soil layer thicknesses -- not the placeholders of the ice nodes
    the step started with."""
    col, state, st, richards = _coupled_fixture(T0=0.0)
    col.snow = SnowColumn(max=5, lnum=0, type=0)
    # at 0 C the freezing curve turns the whole layer to water
    col.glac = SnowColumn.from_layers([(2000.0, 1600.0, 5.0, 0.0)], max=5)
    m = Meteo(Ta=15.0, RH=0.5, P=850.0, wind=3.0, tau_cloud=1.0, Prain=2.0)
    ep = point_step.energy_phase(col, state, st, 3600.0, m, 180.5,
                                 180.5 + 1.0 / 24.0)
    assert (ep.active_ns, ep.active_ng) == (0, 0)
    assert ep.ecol.nsng == 0
    assert ep.diag.evap_Dsoil == [0.0] + st.soil_D_mm
    assert ep.converged
