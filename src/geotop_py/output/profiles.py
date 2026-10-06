"""Write geotop_py's per-layer state in GEOtop's ``output_prof`` format.

The vertical-profile analogue of :mod:`geotop_py.output.tabs`. From each step's
:class:`~geotop_py.point.step.LayerProfile` snapshot it emits the five profile files
GEOtop writes, so they can be diffed layer by layer against ``output_prof/``:

* ``snowDepth``    -- layer thickness [mm]
* ``snowTemp``     -- layer temperature [C]
* ``snowThetaIce`` -- GEOtop profile value ``w_ice/Dz``
* ``snowThetaW``   -- GEOtop profile value ``w_liq/Dz``
* ``soilTemp``     -- soil-node temperature [C]
* ``thetaliq``     -- soil-node liquid content [-]
* ``soilpsi``      -- soil-node matric potential [mm]
* ``soilptot``     -- total-water-equivalent soil potential [mm]

Snow layers are written ``L1..LN`` with ``L1`` the base (adjacent to the soil)
and the surface at the highest present index -- GEOtop's ordering; columns past
the current layer count are ``-9999``. Soil nodes are written top-first, at the
node-centre depths in the reference header.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from ..constants import NUMBER_NOVALUE, is_novalue
from .tabs import _fmt

_BOOK = ["Date12[DDMMYYYYhhmm]", "JulianDayFromYear0[days]", "TimeFromStart[days]",
         "Simulation_Period", "Run", "IDpoint"]

# GEOtop: src/geotop/parameters.cc:2113-2147
# The six soil-profile book fields in GEoTop's own canonical order (matching
# hsl[]/osl[] indices 0-5). write_soil_profile's
# `book_fields` selects a subset and an order among these -- unlike
# write_snow_profile's book, which is unconditionally the full six (every
# reference case that ever writes a snow profile sets SnowAll=1; none pick a
# subset), the soil-profile book is genuinely configurable per DateSoil/
# JulianDayFromYear0Soil/TimeFromStartSoil/PeriodSoil/RunSoil/IDPointSoil
# (keywords 349-354) or forced to this full order by SoilAll (keyword 355).
SOIL_BOOK_FIELDS = _BOOK
DEFAULT_SOIL_BOOK = (0, 1, 2, 3, 4, 5)


def _theta_ice(p, l: int) -> float:
    # output.cc:write_snow_file(choice=1) prints var_to_print / dz directly.
    return p.wice[l] / p.Dz[l] if p.Dz[l] > 0 else NUMBER_NOVALUE


def _theta_w(p, l: int) -> float:
    return p.wliq[l] / p.Dz[l] if p.Dz[l] > 0 else NUMBER_NOVALUE


# profile stem -> per-snow-layer value function (l is 0-based into the prof lists)
SNOW_PROFILES: Dict[str, Callable] = {
    "snowDepth": lambda p, l: p.Dz[l],
    "snowTemp": lambda p, l: p.T[l],
    "snowThetaIce": _theta_ice,
    "snowThetaW": _theta_w,
}


def _book(date, JD: float, JD0: float, point: int, sim_period: int) -> List[str]:
    # GEOtop: src/geotop/output.cc:4326-4333
    # "Simulation_Period" is i_sim, "Run" is i_run.  i_sim is 0 for the write_soil_output call made
    # once at t=0 with the initial condition (before the time loop starts),
    # 1 for every write made from inside it; i_run is always 1 for the
    # single-run simulations this driver supports.
    return [date.strftime("%d/%m/%Y %H:%M"), f"{JD:f}", f"{JD - JD0:f}",
            str(sim_period), "1", str(point)]


def write_snow_profile(path: str, stem: str, records, JD0: float, point: int,
                       ncol: int, init_date=None, initial=None) -> None:
    """Write one snow profile (``snowDepth``/``snowTemp``/``snowThetaIce``/
    ``snowThetaW``) with ``ncol`` layer columns (``L1..Lncol``).

    ``init_date``/``initial`` (a :class:`geotop_py.point.step.LayerProfile`),
    when given, prepend the t=0 initial-condition row GEOtop always writes
    ahead of the first committed step -- omitting it leaves the file one row
    short of the reference.
    """
    fn = SNOW_PROFILES[stem]
    header = _BOOK + [f"L{i}" for i in range(1, ncol + 1)]

    def _row(date, JD, sim_period, prof):
        row = _book(date, JD, JD0, point, sim_period)
        nl = len(prof.Dz) if prof is not None else 0
        for i in range(ncol):
            row.append(_fmt(fn(prof, i)) if i < nl else _fmt(NUMBER_NOVALUE))
        return row

    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        if init_date is not None:
            fh.write(",".join(_row(init_date, JD0, 0, initial)) + "\n")
        for date, JD, _m, out in records:
            fh.write(",".join(_row(date, JD, 1, out.prof)) + "\n")


# GEOtop: src/geotop/output.cc:4605-4653
def interpolate_soil(lmin: int, h: float, max_l: int, dz: Sequence[float],
                     q: Sequence[float]) -> float:
    """One soil-node profile ``q``, linearly interpolated to depth ``h`` [mm].

    ``dz``/``q`` are 1-based sequences of length ``max_l + 1`` (index 0
    unused in ``dz`` always, and in ``q`` too unless ``lmin == 0``).
    ``lmin`` is the profile's own lower bound: ``0`` for a quantity with a
    genuine surface-node value (``psiz`` -- ``q[0]`` is that value), ``1``
    for every other profile this driver writes (``soilTz``/``thetaliq``/
    ``thetaice``/Tzav family), which have no node below the first material
    layer. Depths above the first tracked node or below the last one clamp
    to that node's own value rather than extrapolating; a depth beyond the
    reach of even that clamp returns ``NUMBER_NOVALUE``.

    Walks node-centre depths cumulatively (0 at the surface, then each
    node's own mid-thickness point) exactly as the reference source does,
    including its one extra trailing step past ``max_l`` for the
    below-last-node clamp -- reproduced verbatim, not simplified, because a
    plausible-looking rewrite would get the same boundary cases (index-0
    handling, the clamp regions) subtly wrong the way ``nlayer``'s own
    do-while cadence once did.
    """
    l = lmin
    z0 = 0.0
    qval = NUMBER_NOVALUE
    while True:
        if l == lmin:
            z = z0
            if l > 0:
                z += dz[l] / 2.0
        elif l <= max_l:
            z = z0 + dz[l] / 2.0
            if l > 1:
                z += dz[l - 1] / 2.0
        else:
            z = z0 + dz[max_l] / 2.0

        if h < z and h >= z0:
            if l == lmin:
                qval = q[lmin]
            elif l <= max_l:
                qval = (q[l - 1] * (z - h) + q[l] * (h - z0)) / (z - z0)
            else:
                qval = q[max_l]

        z0 = z
        l += 1

        if not (is_novalue(qval) and l <= max_l + 1):
            break

    return qval


def _soil_book_row(fields, date, JD: float, JD0: float, point: int,
                   sim_period: int) -> List[str]:
    # GEOtop: src/geotop/output.cc:4301-4343
    # One CSV field per entry of `fields` (each a book-field index 0-5, matching SOIL_BOOK_FIELDS'
    # order), -9999 for a position no field claims (osl[j] < 0).
    values = [date.strftime("%d/%m/%Y %H:%M"), f"{JD:f}", f"{JD - JD0:f}",
             str(sim_period), "1", str(point)]
    return [values[i] if i >= 0 else _fmt(NUMBER_NOVALUE) for i in fields]


def write_soil_profile(path: str, records, JD0: float, point: int,
                       depth_cols: List[str], init_date=None, initial=None,
                       field: str = "soil_T",
                       book_fields=DEFAULT_SOIL_BOOK,
                       plot_depths: Optional[List[float]] = None,
                       dz_mm: Optional[List[float]] = None) -> None:
    """Write a soil-node profile (node columns from the reference header,
    top-first). ``field`` selects the :class:`~geotop_py.point.step.LayerProfile`
    attribute -- ``soil_T`` (``soilTemp``, the default), ``soil_th``
    (``thetaliq``), ``soil_psi`` (``soilpsi``) or ``soil_ptot`` (the
    ``SoilTotWaterPressProfileFile`` profile). ``init_date``/``initial``: see
    :func:`write_snow_profile`.

    ``book_fields``: which of the six book columns (``SOIL_BOOK_FIELDS``,
    in that canonical order) to write and in what order -- GEoTop's own
    default (``SoilAll``\\ =1, or unset) is all six in canonical order, but
    ``DateSoil``/``JulianDayFromYear0Soil``/``TimeFromStartSoil``/
    ``PeriodSoil``/``RunSoil``/``IDPointSoil`` (keywords 349-354) can select
    a subset and reorder it -- see ``recorder._soil_book_fields``.

    ``plot_depths``/``dz_mm`` (``SoilPlotDepths``): when given, each column
    is :func:`interpolate_soil` at that target depth [mm] instead of the
    corresponding internal node's own value -- ``depth_cols`` then only
    supplies the header text (the raw keyword values, not the node depths
    :func:`geotop_py.output.recorder._soil_depth_cols` would compute). Every field
    this driver writes through here has no genuine surface node
    (``interpolate_soil``'s ``lmin=0``, only ``psiz``/``Pzplot`` in the real
    source), so this always interpolates with ``lmin=1``.
    """
    header = [SOIL_BOOK_FIELDS[i] for i in book_fields] + depth_cols

    def _row(date, JD, sim_period, prof):
        row = _soil_book_row(book_fields, date, JD, JD0, point, sim_period)
        st = getattr(prof, field) if prof is not None else []
        if plot_depths is not None:
            if st:
                q = [0.0] + list(st)
                nsoil = len(st)
                for h in plot_depths:
                    row.append(_fmt(interpolate_soil(1, h, nsoil, dz_mm, q)))
            else:
                row.extend(_fmt(NUMBER_NOVALUE) for _ in plot_depths)
        else:
            for k in range(len(depth_cols)):
                row.append(_fmt(st[k]) if k < len(st) else _fmt(NUMBER_NOVALUE))
        return row

    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        if init_date is not None:
            fh.write(",".join(_row(init_date, JD0, 0, initial)) + "\n")
        for date, JD, _m, out in records:
            fh.write(",".join(_row(date, JD, 1, out.prof)) + "\n")
