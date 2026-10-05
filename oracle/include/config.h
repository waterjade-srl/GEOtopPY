#ifndef geotop_config_h_meson
#define geotop_config_h_meson

/* Mirrors the config.h that meson generates for the reference build of GEOtop
 * v3.0 (meson-build-release/meson/config/config.h). Kept here so the oracle
 * builds from a plain source checkout, without requiring meson to have run.
 *
 * MATH_OPTIM must stay undefined. When defined, math.optim.h replaces
 * std::pow(a,b) with std::exp(b*std::log(a)), which differs in the last bits.
 * The reference outputs in tests/1D were produced with it off, so an oracle
 * built with it on would pin the Python port to the wrong numbers.
 *
 * build.sh verifies this file against the generated one when a build tree is
 * present.
 */

#define COMPILER_HAS_DIAGNOSTIC_PRAGMA
// macros DISABLE_WARNINGS and ENABLE_WARNINGS
#include "warnings.h"

#define MUTE_GEOLOG
#define MUTE_GEOTIMER
#undef MATH_OPTIM
#undef WITH_OMP
#undef WITH_METEOIO
#endif
