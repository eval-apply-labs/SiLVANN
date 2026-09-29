#!/usr/bin/env bash
# Build the engine into lib/: the evaluator, a Python module compiled with g++, and one library a silicon family —
# each built with its own toolchain where that toolchain is installed, and skipped where it is not.
#
#   SILVANN_PYTHON            the interpreter the engine is built for     default python3
#   SILVANN_SILICON_FAMILIES  which families to build                     default every one on the roster
#   SILVANN_HIP_ARCHES        the AMD cards to build for, e.g. gfx906     default every one the family lists
set -eu
cd "$(dirname "$0")"
PY="${SILVANN_PYTHON:-python3}"
SUFFIX=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')
PYBIND=$("$PY" -c 'import pybind11; print(pybind11.get_include())')
PYINC=$("$PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')
mkdir -p lib/silicon_families

echo "== the evaluator"
g++ -x c++ -O2 -std=c++17 -pthread -fPIC -shared -DSILVANN_MODULE_NAME=silvann_engine_cpu \
    -Iengine/src -I"$PYBIND" -I"$PYINC" engine/src/manifest.cu -o "lib/silvann_engine_cpu$SUFFIX"

echo "== the silicon families"
(cd engine && bash scripts/build_silicon_families.sh) || true    # a family whose toolchain is absent is skipped, not an error
cp engine/build/silicon_families/libsilvann_*.so lib/silicon_families/ 2>/dev/null || true
for f in engine/build/silicon_families/*.skipped; do
    [ -e "$f" ] && echo "   skipped $(basename "$f" .skipped): $(cat "$f")"
done

"$PY" - <<'PY'
import sys
sys.path.insert(0, "lib")
import silvann_engine_cpu as e
names = ["%s (%s)" % (f.get("silicon"), f.get("outcome")) if isinstance(f, dict) else str(f) for f in e.silicon_families()]
print("== built: the evaluator, and the families", ", ".join(names) or "(none)")
PY
