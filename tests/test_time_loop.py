"""``time_loop.run`` checked against a hand-traced copy of ``geotop.cc``'s own
subdivision loop (main run loop, the ``do { ... } while (t < Dt)`` nest around
``EnergyBalance``/``water_balance``) -- there is no oracle wrapper for a
whole-program loop, so this is a from-source hand trace instead of a pin, the
same situation as ``richards1d``/``coupling``.
"""

import pytest

from geotop_py.point import time_loop as tl


def _ignore(payload):
    pass


def test_a_trial_that_always_converges_never_subdivides():
    calls = []

    def attempt(t, Dt):
        calls.append(Dt)
        return True, Dt

    substeps = tl.run(Dt_nominal=100.0, min_Dt=1.0, attempt=attempt, commit=_ignore)

    assert calls == [100.0]
    assert len(substeps) == 1
    s = substeps[0]
    assert (s.dt_run, s.dt_advanced, s.converged, s.trials) == (100.0, 100.0, True, 1)


def test_dt_grows_back_geometrically_after_a_halved_success():
    """Once a sub-step needed a smaller Dt, the next one starts from double
    that -- not straight back to the nominal value -- and is clipped to
    whatever nominal time remains. Driven by a fixed pass/fail script (not a
    Dt threshold) so the trace is unambiguous by hand:

        Dt=1000  fail -> 500 fail -> 250 ok        (two halvings: 1000->250)
        Dt=500   (250 doubled, 750 still fits) ok  (no clip, no failure)
        Dt=1000  (500 doubled) clipped to the 250 remaining -> ok
    """
    script = iter([False, False, True, True, True])

    def attempt(t, Dt):
        return next(script), Dt

    substeps = tl.run(Dt_nominal=1000.0, min_Dt=1.0, attempt=attempt, commit=_ignore)

    assert [(s.dt_run, s.dt_advanced, s.converged, s.trials) for s in substeps] == [
        (250.0, 250.0, True, 3),
        (500.0, 500.0, True, 1),
        (250.0, 250.0, True, 1),
    ]
    assert sum(s.dt_advanced for s in substeps) == 1000.0


def test_giving_up_at_min_dt_commits_the_last_trial_but_advances_by_less():
    """The documented quirk: when halving a failing Dt crosses below min_Dt,
    the loop gives up *without* retrying at that smaller value -- the
    committed payload is from the trial at the larger (pre-halving) Dt, but
    the clock advances by the smaller one. Hand-traced for Dt_nominal=8,
    min_Dt=3, a trial that never converges:

        Dt=8  fail -> halve to 4 (>3, retry)
        Dt=4  fail -> halve to 2 (<=3, give up)   dt_run=4  dt_advanced=2, t=2
        Dt=4  fail -> halve to 2 (<=3, give up)   dt_run=4  dt_advanced=2, t=4
        Dt=4  fail -> halve to 2 (<=3, give up)   dt_run=4  dt_advanced=2, t=6
        Dt=2 (clipped to the 2 remaining) fail, 2 is already <= min_Dt: give
             up after a single trial, dt_run == dt_advanced == 2, t=8
    """
    trial_log = []

    def attempt(t, Dt):
        trial_log.append(Dt)
        return False, Dt

    substeps = tl.run(Dt_nominal=8.0, min_Dt=3.0, attempt=attempt, commit=_ignore)

    assert [(s.dt_run, s.dt_advanced, s.converged, s.trials) for s in substeps] == [
        (4.0, 2.0, False, 2),
        (4.0, 2.0, False, 1),
        (4.0, 2.0, False, 1),
        (2.0, 2.0, False, 1),
    ]
    assert sum(s.dt_advanced for s in substeps) == 8.0
    assert trial_log == [8.0, 4.0, 4.0, 4.0, 2.0]


def test_a_trial_already_at_or_below_min_dt_gives_up_after_one_try():
    calls = []

    def attempt(t, Dt):
        calls.append(Dt)
        return False, Dt

    substeps = tl.run(Dt_nominal=2.0, min_Dt=2.0, attempt=attempt, commit=_ignore)

    assert calls == [2.0]
    assert len(substeps) == 1
    assert substeps[0].dt_run == substeps[0].dt_advanced == 2.0


def test_t_begin_reflects_elapsed_time_not_just_the_trial_dt():
    """A caller needs `t` (not just `Dt`) to reconstruct JDb/JDe the way
    geotop.cc does (``JDb = init_date + (time+t)/secinday``) -- two
    sub-steps at the same Dt must still see different `t_begin`."""
    seen = []

    def attempt(t, Dt):
        seen.append((t, Dt))
        return True, None

    tl.run(Dt_nominal=30.0, min_Dt=1.0, attempt=attempt, commit=_ignore)
    assert seen == [(0.0, 30.0)]

    script = iter([False, True, True])

    def attempt2(t, Dt):
        seen.append((t, Dt))
        return next(script), None

    seen.clear()
    tl.run(Dt_nominal=30.0, min_Dt=1.0, attempt=attempt2, commit=_ignore)
    # sub-step 1: t=0, Dt=30 fails, halves to 15 and succeeds (t_begin stays 0)
    # sub-step 2: t=15, Dt starts at 30 (doubled), clipped to the 15 remaining
    assert seen == [(0.0, 30.0), (0.0, 15.0), (15.0, 15.0)]


def test_payload_is_whatever_the_attempt_returns():
    def attempt(t, Dt):
        return True, {"Dt": Dt, "note": "ok"}

    substeps = tl.run(Dt_nominal=10.0, min_Dt=1.0, attempt=attempt, commit=_ignore)
    assert substeps[0].payload == {"Dt": 10.0, "note": "ok"}


@pytest.mark.parametrize("Dt_nominal,min_Dt", [(3600.0, 1.0), (1.0, 0.001), (10.0, 10.0)])
def test_the_clock_always_reaches_exactly_dt_nominal(Dt_nominal, min_Dt):
    """Regardless of how convergence behaves, the sum of every committed
    sub-step's advance must reconstruct the nominal step exactly -- GEOtop's
    own outer condition (``t < Dt``) guarantees this, and a Python
    reimplementation that drifted here would silently mis-time every output."""
    def flaky(t, Dt):
        # fails on the first two calls at any given Dt magnitude, then gives up
        return False, Dt

    substeps = tl.run(Dt_nominal=Dt_nominal, min_Dt=min_Dt, attempt=flaky, commit=_ignore)
    assert sum(s.dt_advanced for s in substeps) == pytest.approx(Dt_nominal)


def test_each_substep_starts_from_the_previous_commit():
    """Each substep starts from the previous commit."""
    state = [0.0]
    seen = []

    def attempt(t, Dt):
        seen.append((t, Dt, state[0]))
        if Dt == 100.0:
            return False, None                 # force one halving
        return True, state[0] + Dt             # integrate: state grows by Dt

    def commit(payload):
        state[0] = payload

    substeps = tl.run(Dt_nominal=100.0, min_Dt=1.0, attempt=attempt, commit=commit)
    assert [s.dt_run for s in substeps] == [50.0, 50.0]
    assert seen == [(0.0, 100.0, 0.0), (0.0, 50.0, 0.0), (50.0, 50.0, 50.0)]
    assert state[0] == 100.0


def test_a_given_up_trial_is_committed_too():
    committed = []
    tl.run(Dt_nominal=4.0, min_Dt=2.0, attempt=lambda t, Dt: (False, (t, Dt)),
           commit=committed.append)
    assert committed == [(0.0, 4.0), (2.0, 2.0)]
