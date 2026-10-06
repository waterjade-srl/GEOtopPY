"""Physical constants, mirrored from geotop/constants.h (namespace GTConst).

Keep these in exact sync with the C++ header -- the whole point of the oracle
tests is that the Python column reproduces GEOtop bit-for-bit, and a drifted
constant would break that silently.
"""

rho_w = 1000.0      # density of water [kg/m3]
rho_i = 917.0       # density of ice [kg/m3]
GRAVITY = 9.81      # gravity acceleration [m/s2]
tk = 273.15         # 0 degC in Kelvin
k_liq = 0.567       # thermal conductivity of water [W m^-1 K^-1]
k_ice = 2.290       # thermal conductivity of ice [W m^-1 K^-1]
k_air = 0.023       # thermal conductivity of air [W m^-1 K^-1]
c_liq = 4188.0      # heat capacity of water [J/(kg K)]
c_ice = 2117.0      # heat capacity of ice [J/(kg K)]
Tfreezing = 0.0     # freezing temperature [degC]
Lf = 333700.0       # latent heat of fusion [J/kg]  (see constants.h)
Pi = 3.14159265358979

# GEOtop: src/geotop/geotop.cc:157-159
#: GEOtop's sentinels: ``NUMBER_NOVALUE`` is "no value here" (also what the
#: output tables write for a missing value), ``NUMBER_ABSENT`` is "the file
#: has no such column", ``STRING_NOVALUE`` a string keyword left unset.
NUMBER_NOVALUE = -9999.0
NUMBER_ABSENT = -9998.0
STRING_NOVALUE = "none"


# GEOtop compares sentinels after a cast to long, so -9999.4 is novalue too.
def is_novalue(v: float) -> bool:
    return int(v) == int(NUMBER_NOVALUE)


def is_absent(v: float) -> bool:
    return int(v) == int(NUMBER_ABSENT)


def is_undefined(v: float) -> bool:
    return is_absent(v) or is_novalue(v)

# GEOtop: src/geotop/constants.h:60-62
# GEOtop: src/geotop/meteo.cc:84 (LRv falls back to LRd)
# Default lapse rates when the matching LapseRate* keyword component is
# absent/novalue; meteo_distr falls back to these after the keyword-table
# lookup misses.
LapseRateTair = 6.5    # [degC/km]
LapseRateTdew = 2.5    # [degC/km]
LapseRatePrec = 0.0    # [1/km]

# GEOtop: src/geotop/constants.h:47
# GEOtop: src/geotop/input.cc:1462-1470 (soil_evap_layer_bare size)
# Soil depth contributing to bare-soil evaporation -- a hardcoded constant,
# not a keyword. Determines n_evap (the number of soil layers
# find_actual_evaporation_parameters sums over):
# leaving this unbounded (summing the whole column) instead of just the
# near-surface layers it actually reaches changes the alpha/beta reduction
# factor substantially, not just a rounding difference.
z_evap = 100.0          # [mm]

# Euler parameter for the heat equation: 0 = Backward Euler (the only value
# GEOtop actually uses), 0.5 = Crank-Nicolson. Mirrored here for the column
# solver; see the KNe discrepancy note in the project docs.
KNe = 0.0

# Newton line-search backtracking bounds (GTConst::thmin/thmax).
thmin = 0.1
thmax = 0.5

# Armijo-like sufficient-decrease slope for the line search (energy.balance.h).
ni_en = 1.0e-4

# Lower clamp on matric potential [mm] (GTConst::PsiMin).
PsiMin = -1.0e10

# Snow layer whose ice is fully melted has its capacity pinned to this huge
# value so the solver holds its temperature at 0 degC (Csnow_at_T_greater_than_0).
Csnow_at_T_greater_than_0 = 1.0e20

# Steepest slope [deg] any area/volume correction is allowed to use, so that
# 1/cos(slope) stays finite (GTConst::max_slope).
max_slope = 89.999

# Row indices of the per-soil-type parameter matrix `pa`, 1-based, as used by
# every function that reads a van Genuchten or thermal property out of it.
# Pinned against the compiled model in tests/test_soilwater.py: they are an
# interface, not a convention, and a drift here would be silent.
jdz = 1          # layer thickness [mm]
jpsi = 2         # initial matric potential [mm]
jT = 3           # initial temperature [degC]
jKn = 4          # normal (vertical) hydraulic conductivity [mm/s]
jKl = 5          # lateral hydraulic conductivity [mm/s]
jres = 6         # residual water content [-]
jwp = 7          # wilting point water content [-]
jfc = 8          # field capacity water content [-]
jsat = 9         # porosity (saturated water content) [-]
ja = 10          # van Genuchten alpha [mm^-1]
jns = 11         # van Genuchten n [-]
jv = 12          # van Genuchten v (tortuosity exponent) [-]
jkt = 13         # solid thermal conductivity [W m^-1 K^-1]
jct = 14         # solid volumetric heat capacity [J m^-3 K^-1]
jss = 15         # specific storativity [mm^-1]
nsoilprop = 15   # number of soil properties, i.e. the last row index

# --- water balance solver (the #defines at the top of water.balance.cc) ---

# Bracket on the BiCGSTAB stopping tolerance: the Newton forcing term `mu` is
# clamped into [tol_min_GC, tol_max_GC] before the linear solve uses it.
tol_min_GC = 1.0e-13
tol_max_GC = 1.0e5

# Residual a Newton step is always accepted below, whatever the tolerances say.
max_res_adm = 1.0e-2

# Length of the non-monotone line search memory: 1 makes it monotone, which is
# what GEOtop actually ships.
MM = 1

# Armijo-like sufficient-decrease slope for Richards' line search.
ni = 1.0e-7

# Newton iterations after which the conductivity matrix stops being rebuilt.
maxITER_rec_K = 10

# Horizon-shadow tolerances [deg] used when inferring cloudiness from station
# radiation (looser than the strict 0/0 the surface energy balance itself
# uses to decide whether a point is shaded -- see geotop_py.meteo.clouds).
Tol_h_mount = 2.0
Tol_h_flat = 12.0
