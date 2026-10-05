#!/usr/bin/env bash
# Build the numerical oracle: GEOtop v3.0 itself, as a shared library callable
# from Python via ctypes.
#
# Everything except geotop.cc (which holds main()) is compiled from the
# *unmodified* reference checkout; oracle/globals.cc supplies the globals that
# geotop.cc would define, and oracle/oracle.cc adds the extern "C" surface.
# The reference tree is never written to.
#
#   GEOTOP_SRC   root of the GEOtop checkout (default: ./upstream at the repository root)
#   Output:      oracle/libgeotop_oracle.so
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="${GEOTOP_SRC:-$here/../upstream}"

if [ ! -d "$src/src/geotop" ]; then
    echo "GEOtop sources not found under $src" >&2
    echo "set GEOTOP_SRC to the checkout root" >&2
    exit 2
fi

# An oracle built from another revision answers a different model, and nothing
# downstream can tell: the symbols link, the calls return numbers. Warn loudly.
want="$(sed -n 's/^commit: *\([0-9a-f]*\).*/\1/p' "$here/../UPSTREAM" 2>/dev/null || true)"
got="$(git -C "$src" rev-parse HEAD 2>/dev/null || true)"
if [ -n "$want" ] && [ -n "$got" ] && [ "$want" != "$got" ]; then
    echo "warning: $src is at ${got:0:12}, but UPSTREAM pins ${want:0:12}" >&2
    echo "         the oracle will not match the revision this port was verified against" >&2
fi

# The reference build defines MATH_OPTIM off; with it on, math.optim.h swaps
# std::pow for exp/log and the last bits move. If a build tree is present,
# check that our config.h still matches what meson generated.
generated="$src/meson-build-release/meson/config/config.h"
if [ -f "$generated" ]; then
    if ! diff -q <(grep -E '^\s*#(define|undef)' "$generated" | tr -d ' ') \
                 <(grep -E '^\s*#(define|undef)' "$here/include/config.h" | tr -d ' ') \
                 >/dev/null; then
        echo "warning: oracle/include/config.h differs from the generated $generated" >&2
        echo "         the oracle may not match the reference build" >&2
        diff <(grep -E '^\s*#(define|undef)' "$generated") \
             <(grep -E '^\s*#(define|undef)' "$here/include/config.h") >&2 || true
    fi
fi

inc=(
    -I"$here/include"
    -I"$src/src/geotop"
    -I"$src/src/libraries/fluidturtle"
    -I"$src/src/libraries/math"
    -I"$src/src/libraries/ascii"
    -I"$src/src/libraries/geomorphology"
)

# Same flags as the reference build (meson buildtype=release), minus the
# warning noise: -O2 -DNDEBUG, C++11.
flags=(-std=c++11 -O2 -DNDEBUG -fPIC -w)

objdir="$here/.objs"
rm -rf "$objdir"
mkdir -p "$objdir"

mapfile -t sources < <(
    find "$src/src" -name '*.cc' ! -name 'geotop.cc' | sort
)
sources+=("$here/globals.cc" "$here/oracle.cc")

echo "compiling ${#sources[@]} translation units..."
for f in "${sources[@]}"; do
    o="$objdir/$(echo "${f#"$src/"}" | tr '/' '_' | sed 's/\.cc$/.o/')"
    g++ "${flags[@]}" "${inc[@]}" -c "$f" -o "$o"
done

out="$here/libgeotop_oracle.so"
g++ -shared -Wl,--no-undefined -o "$out" "$objdir"/*.o
rm -rf "$objdir"

echo "built $out"
