"""Nominal-timestep subdivision, verbatim from ``geotop.cc``'s run loop.

Every nominal ``Dt`` is attempted whole; on a failed trial (energy and/or
water balance not converging) ``Dt`` is halved and retried, down to
``min_Dt``, at which point the failed trial's result is committed anyway (a
warning, not a fatal error) and the loop moves on. A successful sub-step lets
``Dt`` grow back geometrically (doubling, capped at the nominal value) rather
than jumping straight back to it -- so a run through a rough patch recovers
gradually, the same way it degraded.

The subtlety worth being explicit about: when a failing trial's ``Dt`` is
above ``min_Dt`` but *halving* it lands at or below ``min_Dt``, the loop
gives up immediately -- it does not retry once more at the smaller ``Dt``.
The committed trial is therefore the last one actually *run* (at the
pre-halving ``Dt``), while the loop's own clock (``t``) advances by the
already-halved value the next sub-step would have used. ``Substep.dt_run``
and ``Substep.dt_advanced`` record both, so this does not have to be
rediscovered by reading the trace.

Each sub-step starts from the state the previous one committed: ``commit``
is called on every committed trial before the next trial runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, Tuple, TypeVar

T = TypeVar("T")

# One coupled trial at a candidate Dt, starting at elapsed time `t` within
# the nominal step. The callback owns the physics and its gating: it runs the
# energy balance of every point up to the first that fails, and the water
# balance only when none did (`pipeline.run_simulation`,
# `point.step.simulate_energy_balance`). `t` is exposed
# because GEOtop recomputes JDb/JDe from it on every trial (geotop.cc: `JDb =
# init_date + (time+t)/secinday`, `JDe` likewise with `+Dt`) -- a callback
# that only saw `Dt` could not reconstruct the trial's actual time window
# once a nominal step has been subdivided more than once. Returns whether the
# whole trial converged and whatever payload the caller wants committed on
# success (or logged on give-up).
Attempt = Callable[[float, float], Tuple[bool, T]]

# Writes a committed trial's payload into the persistent state the next
# attempt copies from.
Commit = Callable[[T], None]


@dataclass
class Substep(Generic[T]):
    t_begin: float          # elapsed time at the start of this sub-step
    dt_run: float          # Dt of the trial that produced `payload`
    dt_advanced: float     # Dt actually added to the loop clock for this substep
    converged: bool        # whether that trial reported success
    trials: int             # number of Attempt calls this substep needed
    payload: T


def run(Dt_nominal: float, min_Dt: float, attempt: Attempt,
        commit: Commit) -> List[Substep]:
    """Advance one nominal step, in as many sub-steps as convergence forces.

    ``attempt(t, Dt)`` must run one full trial on its own copy of the
    persistent state, mutating nothing persistent, and return
    ``(converged, payload)``. ``commit(payload)`` must write a committed
    trial into that persistent state; it runs before the next attempt, so
    every sub-step starts where the previous one ended. The returned
    :class:`Substep` list has one entry per committed trial, in order.
    """
    t = 0.0
    Dt = Dt_nominal
    substeps: List[Substep] = []
    while True:
        if t + Dt > Dt_nominal:
            Dt = Dt_nominal - t
        t_begin = t
        trials = 0
        while True:
            dt_run = Dt
            converged, payload = attempt(t, Dt)
            trials += 1
            if converged:
                break
            if Dt > min_Dt:
                Dt *= 0.5
            if Dt > min_Dt:
                continue
            break  # give up: commit the failed trial, advance by the new (smaller) Dt
        substeps.append(Substep(t_begin=t_begin, dt_run=dt_run, dt_advanced=Dt,
                                converged=converged, trials=trials, payload=payload))
        # GEOtop: src/geotop/geotop.cc:449-455
        commit(payload)
        t += Dt
        if Dt < Dt_nominal:
            Dt *= 2.0
        if t >= Dt_nominal:
            break
    return substeps
