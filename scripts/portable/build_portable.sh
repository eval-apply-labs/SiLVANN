#!/usr/bin/env bash
# The portable app's Linux part (NN-194), as build_portable.ps1 makes Windows': SiLVANN/linux/ — SiLVANN.sh, a
# standalone Python with the runtime's packages in python/, the evaluator for it and the silicon families in lib/, and
# the release's tree — packed as silvann_portable_linux_part.tar.gz, which the release's job joins
# with Windows' part into silvann_portable_x64.zip (pack_portable.py): one folder, windows/ linux/ data/, that runs on
# both systems.
#
#   bash scripts/portable/build_portable.sh [--lib DIR] [--out DIR]
#
# Run from a release's checkout: the release's workflow runs it in a manylinux_2_28 container, so what it builds runs
# on glibc 2.28 and later (RHEL 8, Debian 10, Ubuntu 20.04 on). --lib names a folder built already, holding the evaluator
# for this Python (silvann_engine_cpu.cpython-312-x86_64-linux-gnu.so) and silicon_families/*.so; without it,
# scripts/build.sh builds them here for this Python, each family with its toolchain where this machine has it.
#
# The Python is python-build-standalone's (the build uv installs), stripped, its hash checked. Its symbolic links are
# copied as files: a drive formatted for both systems (exFAT) holds none.
set -euo pipefail
PBS_RELEASE=20261026
PBS_PYTHON=3.12.16
PBS_SHA256=ce4f6d0a6cfdb3bed6e44c2f9affd85943c76a42e25ad3a259002f31d6fb8a20
PBS_FILE="cpython-$PBS_PYTHON+$PBS_RELEASE-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_RELEASE/${PBS_FILE//+/%2B}"

LIB=""; OUT="portable"
while [ $# -gt 0 ]; do
    case "$1" in
        --lib) LIB=$(cd "$2" && pwd); shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        *) echo "usage: build_portable.sh [--lib DIR] [--out DIR]" >&2; exit 2 ;;
    esac
done
TREE=$(cd "$(dirname "$0")/../.." && pwd)              # the release's tree: scripts/portable/ is two below it
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ROOT="$OUT/SiLVANN"
PART="$ROOT/linux"                                      # Linux's own: Windows' beside it in windows/, data/ theirs together
rm -rf "$ROOT"; mkdir -p "$PART"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

echo "== ① the release's tree, without what only a repository has, into linux/"
# as build_portable.ps1 leaves out: the engine's source and the image's (neither runs here), and what a build writes
(cd "$TREE" && tar -cf - --exclude=./.git --exclude=./.github --exclude=./portable --exclude=./built --exclude=./lib \
    --exclude=./venv --exclude=./models --exclude=./user_sessions --exclude=./engine --exclude=./docker \
    --exclude='__pycache__' .) | (cd "$PART" && tar -xf -)

echo "== ② Python $PBS_PYTHON, standalone ($PBS_RELEASE), into linux/python/"
curl -fsSL -o "$WORK/$PBS_FILE" "$PBS_URL"
echo "$PBS_SHA256  $WORK/$PBS_FILE" | sha256sum -c --quiet - || { echo "the Python's hash is not $PBS_SHA256" >&2; exit 1; }
mkdir -p "$WORK/py" "$PART/python"
tar -xzf "$WORK/$PBS_FILE" -C "$WORK/py"
(cd "$WORK/py/python" && tar -chf - .) | (cd "$PART/python" && tar -xf -)     # -h: each link copied as its file
PY="$PART/python/bin/python3"
(cd "$WORK" && "$PY" -I -c "import sys; assert sys.version_info[:3] == (3, 12, ${PBS_PYTHON##*.}), sys.version")

echo "== ③ the runtime's packages, for that Python"
(cd "$WORK" && PYTHONNOUSERSITE=1 "$PY" -I -m pip install --quiet --no-cache-dir --only-binary=:all: \
    --no-warn-script-location -r "$TREE/scripts/requirements.txt")

echo "== ④ the evaluator for that Python, and the silicon families, into linux/lib/"
mkdir -p "$PART/lib/silicon_families"
if [ -z "$LIB" ]; then
    (cd "$TREE" && SILVANN_PYTHON="$PY" bash scripts/build.sh)
    LIB="$TREE/lib"
fi
SUFFIX=$(cd "$WORK" && "$PY" -I -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
[ -f "$LIB/silvann_engine_cpu$SUFFIX" ] || { echo "no evaluator for this Python (silvann_engine_cpu$SUFFIX) in $LIB" >&2; exit 1; }
cp "$LIB/silvann_engine_cpu$SUFFIX" "$PART/lib/"
cp "$LIB"/silicon_families/libsilvann_*.so "$PART/lib/silicon_families/"
(cd "$WORK" && PYTHONNOUSERSITE=1 "$PY" -I - "$PART/lib" <<'PY'
import os, sys
sys.path.insert(0, sys.argv[1])
import silvann_engine_cpu as e
names = ["%s (%s)" % (os.path.basename(f.get("path") or "?").replace("libsilvann_", "").split(".")[0], f.get("outcome"))
         for f in e.silicon_families()]
print("   the evaluator imports; the families:", ", ".join(names) or "(none)")
PY
)

echo "== ⑤ what the app never reads left out, SiLVANN.sh, and the archive"
# the terminal database, whose names differ only in case where exFAT and NTFS hold one of them (ncurses reads the
#   system's, and the app has no terminal), and libpython, which only a program embedding Python links (python3 is
#   linked whole) — then every package the app imports imported again, so one that needed them stops the build here
rm -rf "$PART/python/share/terminfo" "$PART/python/lib/"libpython3*.so*
(cd "$WORK" && PYTHONNOUSERSITE=1 "$PY" -I - "$PART/lib" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import silvann_engine_cpu, numpy, tokenizers, huggingface_hub, fastapi, uvicorn, pydantic, httpx, jwt, nicegui, PIL
import pypdf, safetensors, ssl, sqlite3, ctypes, curses, readline          # (tkinter: the system's X11, optional to the app)
print("   the app's packages import without them")
PY
)
cp "$TREE/scripts/portable/SiLVANN.sh" "$PART/SiLVANN.sh"
chmod +x "$PART/SiLVANN.sh" "$PART/python/bin/"*
find "$ROOT" -name __pycache__ -type d -prune -exec rm -rf {} +
[ -z "$(find "$ROOT" -type l)" ] || { echo "a symbolic link is left in the folder" >&2; exit 1; }
CASES=$(cd "$ROOT" && find . | awk '{print tolower($0)}' | sort | uniq -d)
[ -z "$CASES" ] || { echo "paths that differ only in case, which exFAT and NTFS cannot hold both of: $CASES" >&2; exit 1; }
(cd "$OUT" && tar -czf silvann_portable_linux_part.tar.gz SiLVANN)
echo "$OUT/silvann_portable_linux_part.tar.gz: $(du -h "$OUT/silvann_portable_linux_part.tar.gz" | cut -f1), the folder $(du -sh "$ROOT" | cut -f1)"
