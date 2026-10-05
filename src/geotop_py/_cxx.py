"""ctypes bindings to the GEOtop v3.0 oracle (``libgeotop_oracle.so``).

This is the *oracle*: the real C++ functions of the official model, compiled
from the unmodified reference checkout and called directly. Tests compare the
Python reimplementation against these, so "correct" always means "what GEOtop
computes", never a number transcribed by hand.

Build it with ``oracle/build.sh`` (or point ``GEOTOP_ORACLE_SO`` at an existing
shared object).

Array-taking wrappers follow GEOtop's **1-based** indexing: pass the same
indices the C++ uses, and pass arrays whose element 0 is GEOtop's element 1.
The conversion is made explicit at this boundary rather than smeared downstream.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Sequence

_DBL = ctypes.c_double
_DBLP = ctypes.POINTER(ctypes.c_double)
_SHORT = ctypes.c_short
_LONG = ctypes.c_long


_BUILD_HINT = "Run oracle/build.sh, or set GEOTOP_ORACLE_SO to the path of a built one."


def _oracle_path() -> Path:
    """Where the oracle library should be, honouring ``GEOTOP_ORACLE_SO``."""
    env = os.environ.get("GEOTOP_ORACLE_SO")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    # src/geotop_py/_cxx.py -> <repo>/oracle/libgeotop_oracle.so
    return here.parents[2] / "oracle" / "libgeotop_oracle.so"


def _load_library() -> ctypes.CDLL:
    """Load the oracle, or raise ``FileNotFoundError`` if it is unavailable.

    Every reason for not having the oracle -- absent, unreadable, built for
    another architecture, a stale ``GEOTOP_ORACLE_SO`` -- surfaces as that one
    exception, so the callers that degrade to a skip need to guard against one
    type. ``FileNotFoundError`` subclasses ``OSError``, which is what ``CDLL``
    raises, so a caller may equally catch the broader type.
    """
    path = _oracle_path()
    try:
        return ctypes.CDLL(str(path))
    except OSError as exc:
        raise FileNotFoundError(f"cannot load the GEOtop oracle at {path}: {exc}. "
                                f"{_BUILD_HINT}") from exc


_lib = _load_library()


def _bind(name: str, argtypes, restype=_DBL):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def _arr(values: Sequence[float]) -> ctypes.Array:
    return (_DBL * len(values))(*values)


# --------------------------------------------------------------- pedo.funct.h
_D9 = [_DBL] * 9
_psi_teta = _bind("gt_psi_teta", _D9)
_teta_psi = _bind("gt_teta_psi", _D9)
_dteta_dpsi = _bind("gt_dteta_dpsi", _D9)
_k_hydr_soil = _bind("gt_k_hydr_soil", [_DBL] * 12)
_psi_saturation = _bind("gt_psi_saturation", [_DBL] * 6)


# Wrapped in Python defs rather than exposed as raw ctypes handles, so callers
# can name the van Genuchten parameters instead of counting positions.
def psi_teta(w, i, s, r, a, n, m, pmin, Ss):
    return _psi_teta(w, i, s, r, a, n, m, pmin, Ss)


def teta_psi(psi, i, s, r, a, n, m, pmin, Ss):
    return _teta_psi(psi, i, s, r, a, n, m, pmin, Ss)


def dteta_dpsi(psi, i, s, r, a, n, m, pmin, Ss):
    return _dteta_dpsi(psi, i, s, r, a, n, m, pmin, Ss)


def k_hydr_soil(psi, ksat, imp, i, s, r, a, n, m, v, T, ratio):
    return _k_hydr_soil(psi, ksat, imp, i, s, r, a, n, m, v, T, ratio)


def psi_saturation(i, s, r, a, n, m):
    return _psi_saturation(i, s, r, a, n, m)


Harmonic_Mean = _bind("gt_Harmonic_Mean", [_DBL] * 4)
Arithmetic_Mean = _bind("gt_Arithmetic_Mean", [_DBL] * 4)
Mean = _bind("gt_Mean", [_SHORT] + [_DBL] * 4)
Psif = _bind("gt_Psif", [_DBL])

_theta_from_psi = _bind("gt_theta_from_psi",
                        [_DBL, _DBL, _LONG, _DBLP, _LONG, _LONG, _DBL])
_psi_from_theta = _bind("gt_psi_from_theta",
                        [_DBL, _DBL, _LONG, _DBLP, _LONG, _LONG, _DBL])
_dtheta_dpsi_from_psi = _bind("gt_dtheta_dpsi_from_psi",
                              [_DBL, _DBL, _LONG, _DBLP, _LONG, _LONG, _DBL])
_k_from_psi = _bind("gt_k_from_psi",
                    [_LONG, _DBL, _DBL, _DBL, _LONG, _DBLP, _LONG, _LONG,
                     _DBL, _DBL])
_psisat_from = _bind("gt_psisat_from", [_DBL, _LONG, _DBLP, _LONG, _LONG])
_find_actual_evaporation_parameters = _bind(
    "gt_find_actual_evaporation_parameters",
    [_LONG, _DBLP, _DBLP, _DBLP, _LONG, _LONG, _DBL, _DBL, _DBL, _DBL, _DBL,
     _DBL, _LONG, _DBLP, _DBLP, _DBLP],
    None)

_businger = _bind(
    "gt_businger",
    [_SHORT, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _LONG,
     _DBLP, _DBLP, _DBLP, _DBLP],
    None)

_find_activelayerdepth_up = _bind(
    "gt_find_activelayerdepth_up",
    [_DBLP, _DBLP, _DBLP, _DBLP, _LONG, _LONG], _DBL)
_find_activelayerdepth_dw = _bind(
    "gt_find_activelayerdepth_dw",
    [_DBLP, _DBLP, _DBLP, _DBLP, _LONG, _LONG], _DBL)
_find_watertabledepth_up = _bind(
    "gt_find_watertabledepth_up",
    [_DBL, _DBLP, _DBLP, _LONG, _LONG], _DBL)
_find_watertabledepth_dw = _bind(
    "gt_find_watertabledepth_dw",
    [_DBL, _DBLP, _DBLP, _LONG, _LONG], _DBL)
_nlayer = _bind("gt_nlayer", [_DBL, _DBLP, _LONG, _SHORT], _LONG)
_interpolate_soil = _bind(
    "gt_interpolate_soil", [_LONG, _DBL, _LONG, _DBLP, _DBLP], _DBL)


def _flatten_pa(pa: Sequence[Sequence[float]]):
    """Flatten GEOtop's per-soil-type parameter matrix, 1-based on both axes.

    ``pa[i][j]`` in GEOtop is row ``i`` (a jdz/jsat/... index) and layer ``j``.
    Pass it here as a plain nested list whose ``[0][0]`` is GEOtop's ``[1][1]``.
    """
    nr = len(pa)
    nc = len(pa[0]) if nr else 0
    flat = [v for row in pa for v in row]
    return _arr(flat), nr, nc


def theta_from_psi(psi, ice, l, pa, pmin):
    buf, nr, nc = _flatten_pa(pa)
    return _theta_from_psi(psi, ice, l, buf, nr, nc, pmin)


def psi_from_theta(th, ice, l, pa, pmin):
    buf, nr, nc = _flatten_pa(pa)
    return _psi_from_theta(th, ice, l, buf, nr, nc, pmin)


def dtheta_dpsi_from_psi(psi, ice, l, pa, pmin):
    buf, nr, nc = _flatten_pa(pa)
    return _dtheta_dpsi_from_psi(psi, ice, l, buf, nr, nc, pmin)


def k_from_psi(jK, psi, ice, T, l, pa, imp, ratio):
    buf, nr, nc = _flatten_pa(pa)
    return _k_from_psi(jK, psi, ice, T, l, buf, nr, nc, imp, ratio)


def psisat_from(ice, l, pa):
    buf, nr, nc = _flatten_pa(pa)
    return _psisat_from(ice, l, buf, nr, nc)


def find_actual_evaporation_parameters(theta, T, pa, psi, P, rv, Ta, Qa,
                                       Qgsat, nsnow):
    """``theta``/``T`` are 1-based sequences (index 0 unused, matching every
    other array in this codebase); the *number of layers processed* is
    ``len(theta) - 1`` -- the oracle's own ``evap_layer`` is allocated at
    exactly that size before the call, matching how the real C++ reads
    ``n = evap_layer->nh`` internally. Returns ``(alpha, beta, evap_layer)``,
    ``evap_layer`` a 1-based list the same length as ``theta``.
    """
    n = len(theta) - 1
    buf, nr, nc = _flatten_pa(pa)
    theta_buf = _arr(theta[1:])
    T_buf = _arr(T[1:])
    alpha, beta = _DBL(), _DBL()
    evap_out = (_DBL * n)()
    _find_actual_evaporation_parameters(
        n, theta_buf, T_buf, buf, nr, nc, psi, P, rv, Ta, Qa, Qgsat, nsnow,
        ctypes.byref(alpha), ctypes.byref(beta), evap_out)
    return alpha.value, beta.value, [0.0] + list(evap_out)


def businger(a, zmu, zmt, d0, z0, v, T, DT, DQ, z0_z0t, maxiter):
    """Return ``(rm, rh, rv, Lobukhov)`` from GEOtop's Monin-Obukhov
    iteration (``turbulence.cc``, ``aero_resistance``'s ``state_turb==1``
    branch)."""
    rm, rh, rv, L = _DBL(), _DBL(), _DBL(), _DBL()
    _businger(a, zmu, zmt, d0, z0, v, T, DT, DQ, z0_z0t, maxiter,
             ctypes.byref(rm), ctypes.byref(rh), ctypes.byref(rv),
             ctypes.byref(L))
    return rm.value, rh.value, rv.value, L.value


def find_activelayerdepth_up(T, th, thi, pa):
    """``T``/``th``/``thi`` are 1-based sequences (index 0 unused), length
    ``nlayers + 1``. Returns the thaw depth [mm] scanning from the surface.
    """
    buf, nr, nc = _flatten_pa(pa)
    return _find_activelayerdepth_up(_arr(T), _arr(th), _arr(thi), buf, nr, nc)


def find_activelayerdepth_dw(T, th, thi, pa):
    """Thaw depth [mm] scanning from the bottom. See
    :func:`find_activelayerdepth_up` for the argument layout."""
    buf, nr, nc = _flatten_pa(pa)
    return _find_activelayerdepth_dw(_arr(T), _arr(th), _arr(thi), buf, nr, nc)


def find_watertabledepth_up(Z, Ptot, pa):
    """``Ptot`` is a 1-based sequence (index 0 unused), length
    ``nlayers + 1``. Returns the water-table depth [mm] scanning from the
    surface, capped at ``Z``."""
    buf, nr, nc = _flatten_pa(pa)
    return _find_watertabledepth_up(Z, _arr(Ptot), buf, nr, nc)


def find_watertabledepth_dw(Z, Ptot, pa):
    """Water-table depth [mm] scanning from the bottom, capped at ``Z``.
    See :func:`find_watertabledepth_up` for the argument layout."""
    buf, nr, nc = _flatten_pa(pa)
    return _find_watertabledepth_dw(Z, _arr(Ptot), buf, nr, nc)


def nlayer(D, dz, d):
    """``dz`` is a plain 0-based sequence (no unused index-0 slot, unlike
    every other array in this module) of per-layer thickness. Returns the
    1-based index of the layer whose node-centre depth first reaches ``D``.
    """
    n = len(dz)
    return _nlayer(D, _arr(dz), n, d)


def interpolate_soil(lmin, h, max_l, dz, q):
    """``dz``/``q`` are 1-based sequences (index 0 unused in ``dz`` always,
    in ``q`` too unless ``lmin == 0``), length ``max_l + 1``. Returns the
    profile ``q`` linearly interpolated to depth ``h`` [mm]."""
    return _interpolate_soil(lmin, h, max_l, _arr(dz), _arr(q))


# -------------------------------------------------------------------- keywords
_num_par_number = _bind("gt_num_par_number", [], _LONG)
_num_par_char = _bind("gt_num_par_char", [], _LONG)
_keyword_num = _bind("gt_keyword_num", [_LONG], ctypes.c_char_p)
_keyword_char = _bind("gt_keyword_char", [_LONG], ctypes.c_char_p)


def _read_keywords(count_fn, get_fn):
    return [get_fn(i).decode() for i in range(count_fn())]


#: The numeric keyword table (``keywords.h``), in GEOtop's own order. The index
#: of a name here is the ``cod`` that parameters.cc uses to reach it, so order
#: is meaning, not presentation.
KEYWORDS_NUM = _read_keywords(_num_par_number, _keyword_num)

#: The string keyword table, same convention.
KEYWORDS_CHAR = _read_keywords(_num_par_char, _keyword_char)


# ---------------------------------------------------------------- meteo files
_nmet = _bind("gt_nmet", [], _LONG)
_meteo_load = _bind("gt_meteo_load",
                    [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), _LONG,
                     _DBL, _DBL, _DBL, _SHORT, _SHORT, _SHORT, _SHORT, _DBL,
                     _SHORT], _LONG)
_meteo_get = _bind("gt_meteo_get", [_LONG, _DBLP, _LONG], _LONG)
_meteo_indices = _bind("gt_meteo_indices", [ctypes.POINTER(_LONG)], None)

#: GEOtop's sentinel for a column the meteo file did not provide.
NUMBER_ABSENT = _bind("gt_number_absent", [])()

NMET = _nmet()


def _read_meteo_indices():
    names = ("iDate12", "iJDfrom0", "iPrecInt", "iPrec", "iWs", "iWdir",
             "iWsx", "iWsy", "iRh", "iT", "iTdew", "iSW", "iSWb", "iSWd",
             "itauC", "iC", "iLWi", "iSWn", "iTs", "iTbottom", "nmet")
    buf = (_LONG * len(names))()
    _meteo_indices(buf)
    return dict(zip(names, list(buf)))


# GEOtop: src/geotop/constants.h:100-120
#: Meteo column slots, read from the compiled model.
MET = _read_meteo_indices()


# GEOtop: src/geotop/input.cc:384-476
def load_meteo(path, col_names, ST=0.0, STstat=0.0, Zstat=0.0,
               wind_as_xy=0, wind_as_dir=1, vap_as_Td=0, vap_as_RH=1,
               RHmin=10.0, prec_as_intensity=0):
    """Read and complete one station file exactly as the meteo loop of ``get_all_input`` does.

    ``col_names`` is the ``Header*`` mapping: ``NMET`` names, in slot order,
    which is the first ``NMET`` entries of :data:`KEYWORDS_CHAR`.

    Returns a list of rows, each ``NMET`` long, with ``NUMBER_ABSENT`` in the
    slots the file does not provide. Cloudiness is not derived here -- that step
    needs the station horizon and is exposed separately.
    """
    if len(col_names) != NMET:
        raise ValueError(f"expected {NMET} column names, got {len(col_names)}")
    arr = (ctypes.c_char_p * NMET)(*[c.encode() for c in col_names])
    nlines = _meteo_load(str(path).encode(), arr, NMET,
                         ST, STstat, Zstat,
                         wind_as_xy, wind_as_dir, vap_as_Td, vap_as_RH,
                         RHmin, prec_as_intensity)
    if nlines < 0:
        raise RuntimeError(f"could not load {path}")
    buf = (_DBL * NMET)()
    rows = []
    for i in range(nlines):
        n = _meteo_get(i, buf, NMET)
        rows.append(list(buf[:n]))
    return rows


# ------------------------------------------------------------ time interpolation
_time_in_JDfrom0 = _bind("gt_time_in_JDfrom0", [_SHORT, _LONG, _LONG, _DBLP, _LONG, _LONG])
_find_line_data = _bind("gt_find_line_data",
                        [_SHORT, _DBL, _LONG, _DBLP, _LONG, _LONG, _LONG,
                         ctypes.POINTER(_SHORT)], _LONG)
_integrate_meas_linear_beh = _bind("gt_integrate_meas_linear_beh",
                                   [_SHORT, _DBL, _LONG, _DBLP, _LONG, _LONG, _LONG, _LONG])
_integrate_meas_constant_beh = _bind("gt_integrate_meas_constant_beh",
                                     [_SHORT, _DBL, _LONG, _DBLP, _LONG, _LONG, _LONG, _LONG])
_time_interp_linear = _bind("gt_time_interp_linear",
                            [_DBL, _DBL, _DBL, _DBLP, _LONG, _LONG, _LONG, _SHORT,
                             _LONG, _DBLP, _LONG], _LONG)
_time_interp_constant = _bind("gt_time_interp_constant",
                              [_DBL, _DBL, _DBL, _DBLP, _LONG, _LONG, _LONG, _SHORT,
                               _LONG, _DBLP, _LONG], _LONG)
_time_no_interp = _bind("gt_time_no_interp",
                        [_SHORT, _LONG, _DBLP, _LONG, _LONG, _LONG, _DBL,
                         _DBLP, _LONG], _LONG)
_find_station = _bind("gt_find_station", [_LONG, _LONG, _DBLP, _LONG], _LONG)


def _flatten_matrix(data: Sequence[Sequence[float]]):
    nlines = len(data)
    ncols = len(data[0]) if nlines else 0
    flat = [v for row in data for v in row]
    return _arr(flat), nlines, ncols


def time_in_JDfrom0(flag, i, col, data):
    buf, nlines, ncols = _flatten_matrix(data)
    return _time_in_JDfrom0(flag, i, col, buf, nlines, ncols)


def find_line_data(flag, t, ibeg, data, col_date):
    """Returns ``(line, status)``: status is 1 bracketed, 2 past the series
    start, 3 past the series end."""
    buf, nlines, ncols = _flatten_matrix(data)
    a = _SHORT()
    line = _find_line_data(flag, t, ibeg, buf, nlines, ncols, col_date, ctypes.byref(a))
    return line, a.value


def integrate_meas_linear_beh(flag, t, i, data, col, col_date):
    buf, nlines, ncols = _flatten_matrix(data)
    return _integrate_meas_linear_beh(flag, t, i, buf, nlines, ncols, col, col_date)


def integrate_meas_constant_beh(flag, t, i, data, col, col_date):
    buf, nlines, ncols = _flatten_matrix(data)
    return _integrate_meas_constant_beh(flag, t, i, buf, nlines, ncols, col, col_date)


def time_interp_linear(t0, tbeg, tend, data, col_date, flag, istart):
    """Returns ``(out, istart)``: ``out`` has one mean per column, ``istart`` is
    the updated search cursor for the next call."""
    buf, nlines, ncols = _flatten_matrix(data)
    out = (_DBL * ncols)()
    new_istart = _time_interp_linear(t0, tbeg, tend, buf, nlines, ncols,
                                     col_date, flag, istart, out, ncols)
    return list(out), new_istart


def time_interp_constant(t0, tbeg, tend, data, col_date, flag, istart):
    buf, nlines, ncols = _flatten_matrix(data)
    out = (_DBL * ncols)()
    new_istart = _time_interp_constant(t0, tbeg, tend, buf, nlines, ncols,
                                       col_date, flag, istart, out, ncols)
    return list(out), new_istart


def time_no_interp(flag, istart, data, col_date, tbeg):
    buf, nlines, ncols = _flatten_matrix(data)
    out = (_DBL * ncols)()
    new_istart = _time_no_interp(flag, istart, buf, nlines, ncols, col_date, tbeg, out, ncols)
    return list(out), new_istart


def find_station(metvar, var):
    buf, nstat, ncols = _flatten_matrix(var)
    return _find_station(metvar, nstat, buf, ncols)


# ------------------------------------------------------- positional tables
_read_txt_matrix_2 = _bind("gt_read_txt_matrix_2", [ctypes.c_char_p, _LONG], _LONG)
_read_txt_matrix_2_get = _bind("gt_read_txt_matrix_2_get",
                               [_LONG, _DBLP, _LONG], _LONG)


def read_txt_matrix_2(path, ncols):
    """``path`` read into ``ncols`` slots per line, by position, header discarded."""
    n = _read_txt_matrix_2(str(path).encode(), ncols)
    if n < 0:
        raise OSError(f"oracle could not read {path}")
    rows = []
    for i in range(n):
        out = (_DBL * ncols)()
        got = _read_txt_matrix_2_get(i, out, ncols)
        rows.append(list(out)[:got])
    return rows


# ------------------------------------------------------------- inpts parsing
_inpts_open = _bind("gt_inpts_open", [ctypes.c_char_p], _LONG)
_inpts_num_components = _bind("gt_inpts_num_components", [_LONG], _LONG)
_inpts_num_values = _bind("gt_inpts_num_values", [_LONG, _DBLP, _LONG], _LONG)
_inpts_string = _bind("gt_inpts_string", [_LONG], ctypes.c_char_p)

_find_number = _bind("gt_find_number", [ctypes.c_char_p])


# GEOtop: src/libraries/ascii/tabs.cc:277-353
def find_number(text):
    """GEOtop's own decimal parser (``find_number``), not strtod."""
    return _find_number(text.encode())


#: GEOtop's sentinels for an absent parameter, read from the model.
NUMBER_NOVALUE = _bind("gt_number_novalue", [])()
STRING_NOVALUE = _bind("gt_string_novalue", [], ctypes.c_char_p)().decode()

_MAX_COMPONENTS = 4096


def parse_inpts(path):
    """Parse a ``geotop.inpts`` with GEOtop's own tokenizer.

    Returns ``(numeric, strings)``, both keyed by keyword name as it appears in
    ``KEYWORDS_NUM`` / ``KEYWORDS_CHAR``:

    * ``numeric[name]`` is the list of components read, or ``[NUMBER_NOVALUE]``
      when the file does not mention the keyword;
    * ``strings[name]`` is the value, or ``STRING_NOVALUE`` when absent.

    This is the raw table ``assign_numeric_parameters`` works from -- deliberately
    *before* any default is applied, so a Python parser can be checked against
    it with nothing interpreted in between. Keyword matching is case-insensitive,
    as in the C++.
    """
    rc = _inpts_open(str(path).encode())
    if rc != 0:
        raise FileNotFoundError(path)

    buf = (_DBL * _MAX_COMPONENTS)()
    numeric = {}
    for cod, name in enumerate(KEYWORDS_NUM):
        n = _inpts_num_values(cod, buf, _MAX_COMPONENTS)
        numeric[name] = list(buf[:n])
    strings = {name: _inpts_string(cod).decode()
               for cod, name in enumerate(KEYWORDS_CHAR)}
    return numeric, strings


# --------------------------------------------------------------- soil indices
_soil_indices = _bind("gt_soil_indices", [ctypes.POINTER(_LONG)], None)


def _read_soil_indices():
    names = ("jdz", "jpsi", "jT", "jKn", "jKl", "jres", "jwp", "jfc",
             "jsat", "ja", "jns", "jv", "jkt", "jct", "jss", "nsoilprop")
    buf = (_LONG * len(names))()
    _soil_indices(buf)
    return dict(zip(names, list(buf)))


#: GEOtop's soil-parameter row indices (``constants.h``), read from the compiled
#: model rather than transcribed, so they cannot drift.
SOIL = _read_soil_indices()

_point_indices = _bind("gt_point_indices", [ctypes.POINTER(_LONG)], None)


def _read_point_indices():
    names = ("ptID", "ptX", "ptY", "ptZ", "ptLC", "ptSY", "ptS", "ptA",
             "ptSKY", "ptCNS", "ptCWE", "ptCNwSe", "ptCNeSw", "ptDrDEPTH",
             "ptHOR", "ptMAXSWE", "ptLAT", "ptLON", "ptBED", "ptTOT")
    buf = (_LONG * len(names))()
    _point_indices(buf)
    return dict(zip(names, list(buf)))


#: GEOtop's point-matrix column indices (``constants.h``), same convention.
POINT = _read_point_indices()


# ------------------------------------------------------- soil and point files
_read_soil_parameters = _bind(
    "gt_read_soil_parameters",
    [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), _LONG,
     _DBLP, _DBLP, _LONG, _DBL, _LONG, _DBLP, _DBLP, _DBLP, _LONG], _LONG)
_read_point_file = _bind(
    "gt_read_point_file",
    [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), _LONG,
     _DBLP, _LONG, _DBLP, _LONG], _LONG)

_MAX_SOIL_LAYERS = 4096
_MAX_POINTS = 4096


def _flat(matrix):
    """A GEOtop-indexed ``m[row][col]`` (both 1-based) as a flat C array."""
    nrows = len(matrix) - 1
    ncols = len(matrix[1]) - 1
    buf = (_DBL * (nrows * ncols))()
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            buf[(i - 1) * ncols + (j - 1)] = matrix[i][j]
    return buf, nrows, ncols


def _unflat(buf, nrows, ncols):
    """Inverse of :func:`_flat`: index 0 of each axis is left unused."""
    out = [[0.0] * (ncols + 1) for _ in range(nrows + 1)]
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            out[i][j] = buf[(i - 1) * ncols + (j - 1)]
    return out


def read_soil_parameters(name, col_names, pa, pa_bed, init_water_table_depth,
                         bed=1):
    """``read_soil_parameters`` for one soil type, as GEOtop runs it.

    ``pa`` and ``pa_bed`` are the pre-file matrices, ``[row][layer]`` with both
    axes 1-based. ``name`` is the soil file stem (``<name>0001.txt``), or
    :data:`STRING_NOVALUE` for no file. Returns
    ``(pa, pa_bed, init_water_table_depth)`` with the same shape.
    """
    if len(col_names) != SOIL["nsoilprop"]:
        raise ValueError(f"expected {SOIL['nsoilprop']} column names")
    arr = (ctypes.c_char_p * len(col_names))(*[c.encode() for c in col_names])
    pa_in, nrows, nlayers_in = _flat(pa)
    bed_in, _, _ = _flat(pa_bed)
    out = (_DBL * (nrows * _MAX_SOIL_LAYERS))()
    bed_out = (_DBL * (nrows * _MAX_SOIL_LAYERS))()
    iwtd = _DBL(0.0)
    n = _read_soil_parameters(str(name).encode(), arr, len(col_names),
                              pa_in, bed_in, nlayers_in,
                              init_water_table_depth, bed,
                              out, bed_out, ctypes.byref(iwtd),
                              _MAX_SOIL_LAYERS)
    if n < 0:
        raise RuntimeError(f"gt_read_soil_parameters failed for {name}")
    return _unflat(out, nrows, n), _unflat(bed_out, nrows, n), iwtd.value


def read_point_file(name, col_names, chkpt):
    """``read_point_file`` as GEOtop runs it, for a ``point_sim = 1`` run.

    ``chkpt`` is the pre-file point matrix, ``[point][column]`` with both axes
    1-based; ``name`` is the point file stem (``<name>.txt``), or
    :data:`STRING_NOVALUE` for no file.
    """
    if len(col_names) != POINT["ptTOT"]:
        raise ValueError(f"expected {POINT['ptTOT']} column names")
    arr = (ctypes.c_char_p * len(col_names))(*[c.encode() for c in col_names])
    buf, npoints_in, ncols = _flat(chkpt)
    out = (_DBL * (_MAX_POINTS * ncols))()
    n = _read_point_file(str(name).encode(), arr, len(col_names),
                         buf, npoints_in, out, _MAX_POINTS)
    if n < 0:
        raise RuntimeError(f"gt_read_point_file failed for {name}")
    return _unflat(out, n, ncols)


# ---------------------------------------------------------------- util_math.h
_tridiag2 = _bind("gt_tridiag2",
                  [_LONG, _LONG, _DBLP, _DBLP, _DBLP, _DBLP, _DBLP], _SHORT)
_norm_inf = _bind("gt_norm_inf", [_DBLP, _LONG, _LONG])
_norm_2 = _bind("gt_norm_2", [_DBLP, _LONG, _LONG])
_norm_1 = _bind("gt_norm_1", [_DBLP, _LONG, _LONG])
_Cramer_rule = _bind("gt_Cramer_rule", [_DBL] * 6 + [_DBLP, _DBLP], None)
minimize_merit_function = _bind("gt_minimize_merit_function", [_DBL] * 5)
_tridiag = _bind("gt_tridiag", [_LONG, _DBLP, _DBLP, _DBLP, _DBLP, _DBLP], _SHORT)
_product = _bind("gt_product", [_DBLP, _DBLP, _LONG])


def tridiag2(nbeg, nend, ld, d, ud, b):
    """Solve ``A(ld, d, ud) * e + b = 0``; returns (status, e).

    Both off-diagonals are indexed by the **lower** index of the pair, matching
    ``util_math.h``. With 0-based Python lists standing in for GEOtop's 1-based
    vectors::

        d[i - 1]  = A[i][i]      i = 1..n
        ld[i - 1] = A[i+1][i]    i = 1..n-1
        ud[i - 1] = A[i][i+1]    i = 1..n-1

    The last element of ``ld`` and ``ud`` is never read; pass any value.
    """
    e = _arr([0.0] * len(d))
    status = _tridiag2(nbeg, nend, _arr(ld), _arr(d), _arr(ud), _arr(b), e)
    return status, list(e)


def norm_inf(v, nbeg, nend):
    return _norm_inf(_arr(v), nbeg, nend)


def norm_2(v, nbeg, nend):
    return _norm_2(_arr(v), nbeg, nend)


def norm_1(v, nbeg, nend):
    return _norm_1(_arr(v), nbeg, nend)


def Cramer_rule(A, B, C, D, E, F):
    x, y = _DBL(), _DBL()
    _Cramer_rule(A, B, C, D, E, F, ctypes.byref(x), ctypes.byref(y))
    return x.value, y.value


def tridiag(diag_inf, diag, diag_sup, b):
    """Solve ``A * e = b``; returns ``(status, e)``.

    The **opposite** convention from :func:`tridiag2`: solves ``A*e = b`` (not
    ``A*e + b = 0``), and ``status`` is 1 on success, 0 on a zero pivot --
    :func:`tridiag2` returns 0 for success, 1 for failure. Both off-diagonals
    are indexed by the lower index of the pair, same as :func:`tridiag2`.
    """
    nx = len(diag)
    e = _arr([0.0] * nx)
    status = _tridiag(nx, _arr(diag_inf), _arr(diag), _arr(diag_sup), _arr(b), e)
    return status, list(e)


def product(a, b):
    """Dot product of two equal-length vectors."""
    n = len(a)
    return _product(_arr(a), _arr(b), n)


# ---------------------------------------------------------- sparse_matrix.cc
# GEOtop's own sparse format for the symmetric M-matrix K of Richards' Newton
# step: only the strict lower triangle is stored, in parallel arrays Li/Lp/Lx.
#
#   Lx[i]  the value of the i-th stored entry (1-based)
#   Li[i]  its row
#   Lp[c]  the running count of entries through column c, so entry i belongs
#          to column c with Lp[c-1] < i <= Lp[c]
#
# The diagonal is never stored: every product below rebuilds it from K's
# defining property, that each row sums to zero -- entry k(r,c) contributes
# k*(x[r]-x[c]) to row c and k*(x[c]-x[r]) to row r.
_gt_product_lower = _bind(
    "gt_product_using_only_strict_lower_diagonal_part",
    [_DBLP, _DBLP, _LONG, ctypes.POINTER(_LONG), _LONG, ctypes.POINTER(_LONG),
     _LONG, _DBLP], None)
_gt_product_lower_plus_identity = _bind(
    "gt_product_using_only_strict_lower_diagonal_part_plus_identity_by_vector",
    [_DBLP, _DBLP, _DBLP, _LONG, ctypes.POINTER(_LONG), _LONG,
     ctypes.POINTER(_LONG), _LONG, _DBLP], None)
_gt_get_diag = _bind(
    "gt_get_diag_strict_lower_matrix_plus_identity_by_vector",
    [_DBLP, _DBLP, _DBLP, _LONG, ctypes.POINTER(_LONG), _LONG,
     ctypes.POINTER(_LONG), _LONG, _DBLP], None)
_gt_product_matrix_plus_vector = _bind(
    "gt_product_matrix_using_lower_part_by_vector_plus_vector",
    [_DBL, _DBLP, _DBLP, _DBLP, _LONG, ctypes.POINTER(_LONG), _LONG,
     ctypes.POINTER(_LONG), _LONG, _DBLP], None)
_gt_BiCGSTAB = _bind(
    "gt_BiCGSTAB_strict_lower_matrix_plus_identity_by_vector",
    [_DBL, _DBL, _DBL, _DBLP, _DBLP, _DBLP, _LONG, ctypes.POINTER(_LONG),
     _LONG, ctypes.POINTER(_LONG), _LONG, _DBLP], _LONG)


def _arr_long(values):
    return (_LONG * len(values))(*values)


def product_using_only_strict_lower_diagonal_part(x, Li, Lp, Lx):
    nx = len(x)
    out = _arr([0.0] * nx)
    _gt_product_lower(out, _arr(x), nx, _arr_long(Li), len(Li),
                      _arr_long(Lp), len(Lp), _arr(Lx))
    return list(out)


def product_using_only_strict_lower_diagonal_part_plus_identity_by_vector(x, y, Li, Lp, Lx):
    nx = len(x)
    out = _arr([0.0] * nx)
    _gt_product_lower_plus_identity(out, _arr(x), _arr(y), nx, _arr_long(Li),
                                    len(Li), _arr_long(Lp), len(Lp), _arr(Lx))
    return list(out)


def get_diag_strict_lower_matrix_plus_identity_by_vector(y, Li, Lp, Lx):
    """Returns ``(diag, udiag)`` of the tridiagonal part of ``K + diag(y)``."""
    nx = len(y)
    diag = _arr([0.0] * nx)
    udiag = _arr([0.0] * max(nx - 1, 1))
    _gt_get_diag(diag, udiag, _arr(y), nx, _arr_long(Li), len(Li),
                _arr_long(Lp), len(Lp), _arr(Lx))
    return list(diag), list(udiag)[:nx - 1]


def product_matrix_using_lower_part_by_vector_plus_vector(k, y, x, Li, Lp, Lx):
    """``k * (y + K x)``."""
    nx = len(x)
    out = _arr([0.0] * nx)
    _gt_product_matrix_plus_vector(k, out, _arr(y), _arr(x), nx, _arr_long(Li),
                                   len(Li), _arr_long(Lp), len(Lp), _arr(Lx))
    return list(out)


def BiCGSTAB_strict_lower_matrix_plus_identity_by_vector(
        tol_rel, tol_min, tol_max, x0, b, y, Li, Lp, Lx):
    """Solve ``(K + diag(y)) x = b`` starting from ``x0``.

    Returns ``(iterations, x)``; ``iterations`` is -1 if the tridiagonal
    preconditioner solve hit a zero pivot.
    """
    nx = len(x0)
    x = _arr(x0)
    iters = _gt_BiCGSTAB(tol_rel, tol_min, tol_max, x, _arr(b), _arr(y), nx,
                         _arr_long(Li), len(Li), _arr_long(Lp), len(Lp), _arr(Lx))
    return iters, list(x)


# -------------------------------------------------------------------- snow.cc
_rho_newlyfallensnow = _bind("gt_rho_newlyfallensnow", [_DBL] * 3)


# GEOtop: src/geotop/snow.cc:55-81
def rho_newlyfallensnow(u, Tatm, Tfreez=0.0):
    """Fresh-snow density (Jordan).

    v3.0 declares a third argument ``Tfreez`` and never reads it, so it
    defaults here and any value gives the same result.
    """
    return _rho_newlyfallensnow(u, Tatm, Tfreez)
internal_energy = _bind("gt_internal_energy", [_DBL] * 3)
theta_snow = _bind("gt_theta_snow", [_DBL] * 3)
dtheta_snow = _bind("gt_dtheta_snow", [_DBL] * 3)
k_thermal_snow_Sturm = _bind("gt_k_thermal_snow_Sturm", [_DBL])
k_thermal_snow_Yen = _bind("gt_k_thermal_snow_Yen", [_DBL])

Fzen = _bind("gt_Fzen", [_DBL])
find_albedo = _bind("gt_find_albedo", [_DBL] * 5)
_snow_albedo = _bind("gt_snow_albedo", [_DBL] * 7 + [ctypes.c_int])
_update_snow_age = _bind("gt_update_snow_age", [_DBL] * 4 + [_DBLP], None)


def snow_albedo(ground_alb, snowD, AEP, freshsnow_alb, C, tsnow, cosinc,
                zenith):
    """Snow albedo, blended towards ``ground_alb`` when the cover is shallow.

    ``zenith`` picks the zenith-angle correction GEOtop passes as a function
    pointer: 0 for ``Zero`` (diffuse), 1 for ``Fzen`` (direct beam).
    ``snowD`` and ``AEP`` are in mm.
    """
    return _snow_albedo(ground_alb, snowD, AEP, freshsnow_alb, C, tsnow,
                        cosinc, 1 if zenith else 0)


def update_snow_age(Psnow, Ts, Dt, Prestore, tsnow_nondim):
    """Return the advanced non-dimensional snow age.

    ``Psnow`` [mm over the step], ``Ts`` [C] the top snow layer's temperature,
    ``Dt`` [s], ``Prestore`` the snowfall depth that resets the age to zero.
    """
    t = _DBL(tsnow_nondim)
    _update_snow_age(Psnow, Ts, Dt, Prestore, ctypes.byref(t))
    return t.value


_from_internal_energy = _bind("gt_from_internal_energy",
                              [_DBL, _DBL, _DBLP, _DBLP, _DBLP], None)


def from_internal_energy(a, h, w_ice, w_liq):
    """Invert the enthalpy closure; returns the new (w_ice, w_liq, T).

    ``w_ice`` and ``w_liq`` are inputs as well as outputs: GEOtop forms
    ``SWE = w_ice + w_liq`` from them and returns zeros when that is 0, so pass
    the layer's current contents, not placeholders.
    """
    wi, wl, T = _DBL(w_ice), _DBL(w_liq), _DBL()
    _from_internal_energy(a, h, ctypes.byref(wi), ctypes.byref(wl),
                          ctypes.byref(T))
    return wi.value, wl.value, T.value


# ----------------------------------------------------------- energy.balance.cc
k_thermal = _bind("gt_k_thermal", [_SHORT, _SHORT] + [_DBL] * 4)

_calc_C = _bind("gt_calc_C",
                [_LONG, _LONG, _DBL, _DBLP, _DBLP, _DBLP, _DBLP, _LONG,
                 _DBLP, _LONG, _LONG])


def calc_C(l, nsng, a, wi, wl, dw, D, pa):
    """Heat capacity of column node ``l`` (1-based), snow if ``l <= nsng``."""
    buf, nr, nc = _flatten_pa(pa)
    n = len(wi)
    return _calc_C(l, nsng, a, _arr(wi), _arr(wl), _arr(dw), _arr(D), n,
                   buf, nr, nc)


def C_snow(wi, wl, dw, a, D):
    """Volumetric heat capacity of a snow node, via the snow branch of calc_C."""
    return calc_C(1, 1, a, [wi], [wl], [dw], [D], _dummy_pa())


def C_soil(ct, sat, wi, wl, dw, a, D):
    """Volumetric heat capacity of a soil node, via the soil branch of calc_C.

    ``calc_C`` takes the soil branch when ``l > nsng``, and reads ``jct`` and
    ``jsat`` at layer ``l - nsng``; with ``l = 1, nsng = 0`` that is layer 1.
    """
    pa = _dummy_pa()
    pa[SOIL["jct"] - 1][0] = ct
    pa[SOIL["jsat"] - 1][0] = sat
    return calc_C(1, 0, a, [wi], [wl], [dw], [D], pa)


def _dummy_pa():
    """A one-layer soil parameter matrix of zeros, sized to nsoilprop."""
    return [[0.0] for _ in range(SOIL["nsoilprop"])]


# -------------------------------------------------------------------- times.cc
is_leap = _bind("gt_is_leap", [_LONG], _SHORT)
convert_dateeur12_JDfrom0 = _bind("gt_convert_dateeur12_JDfrom0", [_DBL])
convert_JDfrom0_dateeur12 = _bind("gt_convert_JDfrom0_dateeur12", [_DBL])
convert_JDandYear_JDfrom0 = _bind("gt_convert_JDandYear_JDfrom0", [_DBL, _LONG])

_JDfrom0_JDandYear = _bind("gt_convert_JDfrom0_JDandYear",
                           [_DBL, _DBLP, ctypes.POINTER(_LONG)], None)
_dateeur12_JDandYear = _bind("gt_convert_dateeur12_JDandYear",
                             [_DBL, _DBLP, ctypes.POINTER(_LONG)], None)
_JDandYear_daymonthhourmin = _bind(
    "gt_convert_JDandYear_daymonthhourmin",
    [_DBL, _LONG] + [ctypes.POINTER(_LONG)] * 4, None)
_dateeur12_daymonthyearhourmin = _bind(
    "gt_convert_dateeur12_daymonthyearhourmin",
    [_DBL] + [ctypes.POINTER(_LONG)] * 5, None)


def convert_JDfrom0_JDandYear(JDfrom0):
    JD, year = _DBL(), _LONG()
    _JDfrom0_JDandYear(JDfrom0, ctypes.byref(JD), ctypes.byref(year))
    return JD.value, year.value


def convert_dateeur12_JDandYear(date):
    JD, year = _DBL(), _LONG()
    _dateeur12_JDandYear(date, ctypes.byref(JD), ctypes.byref(year))
    return JD.value, year.value


def convert_JDandYear_daymonthhourmin(JD, year):
    d, m, h, mi = _LONG(), _LONG(), _LONG(), _LONG()
    _JDandYear_daymonthhourmin(JD, year, ctypes.byref(d), ctypes.byref(m),
                               ctypes.byref(h), ctypes.byref(mi))
    return d.value, m.value, h.value, mi.value


def convert_dateeur12_daymonthyearhourmin(date):
    d, m, y, h, mi = (_LONG() for _ in range(5))
    _dateeur12_daymonthyearhourmin(date, ctypes.byref(d), ctypes.byref(m),
                                   ctypes.byref(y), ctypes.byref(h),
                                   ctypes.byref(mi))
    return d.value, m.value, y.value, h.value, mi.value


# -------------------------------------------------------------- turbulence.cc
Psim = _bind("gt_Psim", [_DBL])
Psih = _bind("gt_Psih", [_DBL])
PsiStab = _bind("gt_PsiStab", [_DBL])
roughT = _bind("gt_roughT", [_DBL] * 3)
roughQ = _bind("gt_roughQ", [_DBL] * 3)
Levap = _bind("gt_Levap", [_DBL])
latent = _bind("gt_latent", [_DBL] * 2)
cz = _bind("gt_cz", [_SHORT] + [_DBL] * 4)
CZ = _bind("gt_CZ", [_SHORT, _SHORT] + [_DBL] * 4)

# ------------------------------------------------------------------- meteo.cc
pressure = _bind("gt_pressure", [_DBL])
temperature = _bind("gt_temperature", [_DBL] * 4)
SatVapPressure = _bind("gt_SatVapPressure", [_DBL] * 2)
TfromSatVapPressure = _bind("gt_TfromSatVapPressure", [_DBL] * 2)
SpecHumidity = _bind("gt_SpecHumidity", [_DBL] * 2)
VapPressurefromSpecHumidity = _bind("gt_VapPressurefromSpecHumidity", [_DBL] * 2)
Tdew = _bind("gt_Tdew", [_DBL] * 3)
RHfromTdew = _bind("gt_RHfromTdew", [_DBL] * 3)
air_density = _bind("gt_air_density", [_DBL] * 3)
air_cp = _bind("gt_air_cp", [_DBL])

_part_snow = _bind("gt_part_snow", [_DBL, _DBLP, _DBLP, _DBL, _DBL, _DBL], None)
_SatVapPressure_2 = _bind("gt_SatVapPressure_2", [_DBLP, _DBLP, _DBL, _DBL], None)
_SpecHumidity_2 = _bind("gt_SpecHumidity_2",
                        [_DBLP, _DBLP, _DBL, _DBL, _DBL], None)


def part_snow(prec_total, T, t_rain, t_snow):
    """Split precipitation into (rain, snow) by air temperature."""
    rain, snow = _DBL(), _DBL()
    _part_snow(prec_total, ctypes.byref(rain), ctypes.byref(snow),
               T, t_rain, t_snow)
    return rain.value, snow.value


def SatVapPressure_2(T, P):
    """Saturation vapour pressure and its temperature derivative."""
    e, de_dT = _DBL(), _DBL()
    _SatVapPressure_2(ctypes.byref(e), ctypes.byref(de_dT), T, P)
    return e.value, de_dT.value


def SpecHumidity_2(RH, T, P):
    """Specific humidity and its temperature derivative."""
    Q, dQ_dT = _DBL(), _DBL()
    _SpecHumidity_2(ctypes.byref(Q), ctypes.byref(dQ_dT), RH, T, P)
    return Q.value, dQ_dT.value


# ---------------------------------------------------------------- meteodistr.cc
_interpolate_meteo = _bind("gt_interpolate_meteo",
                           [_SHORT, _DBL, _DBL, _DBLP, _DBLP, _LONG, _DBLP,
                            _LONG, _LONG, _DBL, _SHORT, _DBLP], _SHORT)
get_dn = _bind("gt_get_dn", [_LONG, _LONG, _DBL, _DBL, _LONG])
_topo_mod_winds = _bind(
    "gt_topo_mod_winds",
    [_DBLP, _DBLP, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL,
     _DBL, _DBL, _DBL], None)
find_cloudfactor = _bind("gt_find_cloudfactor", [_DBL] * 5)


def interpolate_meteo(flag, dX, dY, xst, yst, value, metcod, dn0, iobsint):
    """``interpolate_meteo`` onto the single point of a 1D run.

    ``value`` is one row per station (only column ``metcod`` matters).
    Returns ``(ok, grid_value)``; ``ok`` is the station count actually used
    (0 if the column is absent everywhere -- ``grid_value`` is then 0.0 and
    should be ignored, matching what the caller does with GEOtop's own
    ``ok == 0``).
    """
    nstn = len(value)
    ncols = len(value[0]) if nstn else 1
    xbuf = _arr(xst) if nstn else _arr([0.0])
    ybuf = _arr(yst) if nstn else _arr([0.0])
    vbuf = _arr([v for row in value for v in row]) if nstn else _arr([0.0])
    out = _DBL()
    ok = _interpolate_meteo(flag, dX, dY, xbuf, ybuf, nstn, vbuf, ncols,
                            metcod, dn0, iobsint, ctypes.byref(out))
    return ok, out.value


def topo_mod_winds(winddir, windspd, slopewtD, curvewtD, slopewtI, curvewtI,
                   curvature1, curvature2, curvature3, curvature4,
                   slope_az, terrain_slope, topo, undef):
    """Topographic wind-speed/direction correction at a single point.

    Returns ``(winddir, windspd)``.
    """
    wd, ws = _DBL(winddir), _DBL(windspd)
    _topo_mod_winds(ctypes.byref(wd), ctypes.byref(ws),
                    slopewtD, curvewtD, slopewtI, curvewtI,
                    curvature1, curvature2, curvature3, curvature4,
                    slope_az, terrain_slope, topo, undef)
    return wd.value, ws.value


# ----------------------------------------------------------------- radiation.cc
_gt_sun = _bind("gt_sun", [_DBL, _DBLP, _DBLP, _DBLP], None)
SolarHeight = _bind("gt_SolarHeight", [_DBL] * 4)
SolarAzimuth = _bind("gt_SolarAzimuth", [_DBL] * 4)
_gt_shadows_point = _bind("gt_shadows_point",
                          [_DBLP, _LONG, _DBL, _DBL, _DBL, _DBL], _SHORT)


def sun(JDfrom0):
    """Returns ``(E0, Et, Delta)``."""
    E0, Et, Delta = _DBL(), _DBL(), _DBL()
    _gt_sun(JDfrom0, ctypes.byref(E0), ctypes.byref(Et), ctypes.byref(Delta))
    return E0.value, Et.value, Delta.value


def shadows_point(hor, alpha, azimuth, tol_mount, tol_flat):
    """``hor`` is a list of ``(azimuth, elevation)`` pairs."""
    flat = [v for row in hor for v in row]
    return _gt_shadows_point(_arr(flat), len(hor), alpha, azimuth, tol_mount, tol_flat)


# --------------------------------------------------------------------- clouds.cc
_gt_find_cloudiness = _bind(
    "gt_find_cloudiness",
    [_LONG, _DBLP, _LONG] + [_DBL] * 11)
_gt_find_sunset = _bind(
    "gt_find_sunset",
    [_LONG, _DBLP, _LONG, _DBLP, _LONG, _DBL, _DBL, _DBL, _DBL, ctypes.POINTER(_LONG)],
    _LONG)
_gt_fill_meteo_data_with_cloudiness = _bind(
    "gt_fill_meteo_data_with_cloudiness",
    [_DBLP, _LONG, _DBLP, _LONG, _DBL, _DBL, _DBL, _DBL, _DBL, _DBL, _LONG,
     _DBL, _DBL, _DBL, _DBL, _DBL, _DBLP],
    _SHORT)


def _flatten_meteo(meteo):
    return _arr([v for row in meteo for v in row]), len(meteo)


def find_cloudiness(n, meteo, lat, lon, ST, Z, sky, SWrefl_surr, rotation,
                    Lozone, alpha, beta, albedo):
    """``lat``/``lon`` in **radians** -- unlike :func:`fill_meteo_data_with_cloudiness`
    below, this wraps ``find_cloudiness`` directly, which (like ``find_sunset``)
    expects the conversion already done: in the real model, only the
    ``fill_meteo_data_with_cloudiness`` entry point converts degrees to
    radians, once, before calling down into ``cloudiness()``/``find_sunset()``/
    ``find_cloudiness()``. Passing degrees here silently gives a wrong but
    plausible-looking result -- shadow tests still return 0/1, just for the
    wrong sun position -- which is exactly what makes this easy to get wrong
    without a direct pin against the real model to catch it.
    """
    buf, meteolines = _flatten_meteo(meteo)
    result = _gt_find_cloudiness(n, buf, meteolines, lat, lon, ST, Z, sky,
                                 SWrefl_surr, rotation, Lozone, alpha, beta, albedo)
    return None if int(result) == int(NUMBER_NOVALUE) else result


def find_sunset(nist, meteo, hor, lat, lon, ST, rotation):
    """Returns ``(n0, n1)``. ``lat``/``lon`` in **radians** -- see the note on
    :func:`find_cloudiness`."""
    mbuf, meteolines = _flatten_meteo(meteo)
    hbuf = _arr([v for row in hor for v in row])
    n0 = _LONG()
    n1 = _gt_find_sunset(nist, mbuf, meteolines, hbuf, len(hor), lat, lon, ST,
                        rotation, ctypes.byref(n0))
    return n0.value, n1


def fill_meteo_data_with_cloudiness(meteo, hor, lat, lon, ST, Z, sky,
                                    SWrefl_surr, ndivday, rotation,
                                    Lozone, alpha, beta, albedo):
    """``lat``/``lon`` in **degrees** (this is the one entry point that still
    matches its own caller's units -- it converts to radians once, internally,
    before calling ``cloudiness()``). Returns ``(added, tauC)``: ``added`` is
    whether the column was computed; ``tauC`` is one value per row (``None``
    where undefined)."""
    mbuf, meteolines = _flatten_meteo(meteo)
    hbuf = _arr([v for row in hor for v in row])
    tauC_out = (_DBL * meteolines)()
    added = _gt_fill_meteo_data_with_cloudiness(
        mbuf, meteolines, hbuf, len(hor), lat, lon, ST, Z, sky, SWrefl_surr,
        ndivday, rotation, Lozone, alpha, beta, albedo, tauC_out)
    tauC = [None if int(v) == int(NUMBER_NOVALUE) else v for v in tauC_out]
    return bool(added), tauC


# ------------------------------------------------ rw_maps.cc / geomorphology.cc
_gt_read_map = _bind("gt_read_map",
                     [ctypes.c_char_p, _DBL, _DBLP, _DBLP, _LONG], _LONG)
_gt_multipass_topofilter = _bind(
    "gt_multipass_topofilter",
    [_LONG, _LONG, _LONG, _DBLP, _DBLP, _LONG, _LONG], None)
_gt_curvature = _bind(
    "gt_curvature",
    [_DBL, _DBL, _LONG, _LONG, _DBLP, _DBLP, _DBLP, _DBLP, _DBLP, _LONG], None)
_gt_row = _bind("gt_row", [_DBL, _LONG, _DBL, _DBL, _LONG], _LONG)
_gt_col = _bind("gt_col", [_DBL, _LONG, _DBL, _DBL, _LONG], _LONG)
existing_file = _bind("gt_existing_file", [ctypes.c_char_p], _SHORT)


def read_map(path, no_value=NUMBER_NOVALUE, maxcells=4_000_000):
    """Returns ``(header, data)``.

    ``header`` is ``(Dy, Dx, Y0, X0, nrows, ncols, novalue_sign, novalue)``;
    ``data`` is a list of ``nrows`` rows of ``ncols`` values, 0-based here.
    ``path`` is the file stem, without the ``.asc`` extension, exactly as
    ``geotop.inpts`` gives it.
    """
    header = (_DBL * 8)()
    data = (_DBL * maxcells)()
    n = _gt_read_map(path.encode(), no_value, header, data, maxcells)
    if n < 0:
        raise ValueError(f"{path}: more than {maxcells} cells")
    nr, nc = int(header[4]), int(header[5])
    return (tuple(header),
            [[data[r * nc + c] for c in range(nc)] for r in range(nr)])


def multipass_topofilter(ntimes, Zin, novalue=int(NUMBER_NOVALUE), n=1):
    """``Zin`` is a list of rows, 0-based; the result has the same shape."""
    nr, nc = len(Zin), len(Zin[0])
    flat = _arr([v for row in Zin for v in row])
    out = (_DBL * (nr * nc))()
    _gt_multipass_topofilter(ntimes, nr, nc, flat, out, novalue, n)
    return [[out[r * nc + c] for c in range(nc)] for r in range(nr)]


def curvature(deltax, deltay, topo, undef=int(NUMBER_NOVALUE)):
    """Returns the four curvature grids ``(N-S, W-E, NW-SE, NE-SW)``, each with
    the shape of ``topo`` (a list of rows, 0-based)."""
    nr, nc = len(topo), len(topo[0])
    flat = _arr([v for row in topo for v in row])
    outs = [(_DBL * (nr * nc))() for _ in range(4)]
    _gt_curvature(deltax, deltay, nr, nc, flat, *outs, undef)
    return tuple([[o[r * nc + c] for c in range(nc)] for r in range(nr)]
                 for o in outs)


def row(N, nrows, Dy, Y0, novalue=int(NUMBER_NOVALUE)):
    return _gt_row(N, nrows, Dy, Y0, novalue)


def col(E, ncols, Dx, X0, novalue=int(NUMBER_NOVALUE)):
    return _gt_col(E, ncols, Dx, X0, novalue)
