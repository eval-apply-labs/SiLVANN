#!/usr/bin/env bash
# SiLVANN's installer: the release, its Python environment and its build, in a folder of its own under the current one,
# and a `silvann` command.
#
#   curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.sh | bash
#
# It installs nothing system-wide and asks for no password: a tool it needs and does not find is named, with the line
# that installs it, and the installer stops. Run it again to update; models/, user_sessions/ and the venv stay.
#
#   SILVANN_DIR      where it installs                         default ./silvann (SILVANN_DIR=. for the current folder)
#   SILVANN_SOURCE   the repository it fetches                 default https://github.com/eval-apply-labs/SiLVANN
#   SILVANN_REF      a branch of it                            default the repository's own
#   SILVANN_BIN      where the `silvann` command goes          default ~/.local/bin
#   SILVANN_NVCC     CUDA's compiler, when it is not found     default nvcc on the PATH, else /usr/local/cuda*/bin/nvcc
#
#   bash install.sh             fetch (or update), the environment, the build, the command
#   bash install.sh --build     the build alone, the cards and compilers looked for again (`silvann build`)
set -euo pipefail

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
stop() { printf '\033[31m⛔ %s\033[0m\n' "$*" >&2; exit 1; }

SOURCE=${SILVANN_SOURCE:-https://github.com/eval-apply-labs/SiLVANN}
BIN=${SILVANN_BIN:-$HOME/.local/bin}

# ── what the machine has ──────────────────────────────────────────────────────────────────────────────────────────────
check_machine() {
    if [ "$(uname -s)" != Linux ]; then
        stop "SiLVANN runs on Linux. On a Windows or Mac machine, boot Linux from a USB SSD (Ubuntu 22.04, the system it is
   tested on): docs/users.md, \"On Windows\". Windows' WSL2 may work but is untested."
    fi
    [ "$(uname -m)" = x86_64 ] || stop "SiLVANN is built for x86-64 processors; this one is $(uname -m)."
    local missing=() py
    py=$(command -v python3 || true)
    if [ -z "$py" ]; then missing+=(python3 python3-venv)
    elif ! "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
        stop "Python 3.10 or newer is needed; this one is $("$py" -V 2>&1)."
    elif ! "$py" -c 'import venv, ensurepip' 2>/dev/null; then missing+=(python3-venv)
    fi
    command -v g++ >/dev/null || missing+=(g++)
    command -v git >/dev/null || command -v tar >/dev/null || missing+=(git)
    command -v git >/dev/null || command -v curl >/dev/null || missing+=(curl)
    if [ ${#missing[@]} -gt 0 ]; then
        stop "missing: ${missing[*]}. On Ubuntu or Debian:  sudo apt update && sudo apt install -y ${missing[*]}
   then run this installer again."
    fi
}

# the cards, and the compiler each one's family is built with; what is chosen goes to build.sh as its variables
find_cards() {
    BUILD_ENV=()
    FAMILIES=x86_avx2
    if grep -qw avx2 /proc/cpuinfo; then
        note "the CPU: $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //') — AVX2$(grep -qw avx512bw /proc/cpuinfo && echo ', AVX-512')"
    else
        note "the CPU has no AVX2: the CPU's own family will not run here; the engine's portable host family will"
    fi
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        local cards caps nvcc
        # ⛳ each query ends `|| true`: under pipefail a pipeline that fails or matches nothing would end the installer
        #   here with no message — a driver that does not know the field, a card the user may not open
        cards=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' || true)
        caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | sort -u | paste -sd ',' || true)
        note "NVIDIA: $cards"
        nvcc=${SILVANN_NVCC:-$(command -v nvcc || ls -d /usr/local/cuda/bin/nvcc /usr/local/cuda-1[2-9]*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)}
        if [ -n "$nvcc" ] && [ -x "$nvcc" ]; then
            note "   built with $nvcc ($("$nvcc" --version | grep -o 'release [0-9.]*'))${caps:+, for compute capability $caps}"
            BUILD_ENV+=("SILVANN_NVCC=$nvcc"); FAMILIES="$FAMILIES nvidia_cuda12"
            [ -n "$caps" ] && BUILD_ENV+=("SILVANN_CUDA_ARCHES=$caps")
        else
            note "   ⚠ no nvcc: the card's family needs CUDA 12's compiler — the CUDA toolkit from developer.nvidia.com/cuda-downloads"
            note "     (the toolkit; the driver is installed already), then \`silvann build\`. Until then the CPU runs the models."
        fi
    fi
    if [ -e /dev/kfd ]; then
        if [ -x /opt/rocm/bin/hipcc ]; then
            local gfx
            # no gfx when rocminfo cannot open the card (a user outside the `render` group): the build is then for every
            #   gfx the family lists, and `|| true` keeps the empty match from ending the installer
            gfx=$( (/opt/rocm/bin/rocminfo 2>/dev/null || true) | grep -o 'gfx[0-9a-f]*' | sort -u | paste -sd ',' || true)
            note "AMD: ${gfx:-a card} — built with /opt/rocm/bin/hipcc"
            [ -n "$gfx" ] && BUILD_ENV+=("SILVANN_HIP_ARCHES=$gfx")
            FAMILIES="$FAMILIES amd_rocm6_wave64"
            [ -f /opt/rocm/include/CL/cl.h ] && FAMILIES="$FAMILIES khronos_opencl2"
        else
            note "AMD: a card, and no ROCm — its family needs ROCm 6's hipcc (/opt/rocm), then \`silvann build\`"
        fi
    fi
    note "memory: $(awk '/MemTotal/ {printf "%.0f GB", $2 / 1048576}' /proc/meminfo) of RAM"
}

# ── the release ───────────────────────────────────────────────────────────────────────────────────────────────────────
fetch() {
    if [ -d "$DIR/.git" ]; then
        say "updating $DIR"
        git -C "$DIR" pull --ff-only --quiet
        return
    fi
    # a folder of its own, or one this installer filled before: never files it did not write
    if [ ! -e "$DIR/build.sh" ] && [ -n "$(ls -A "$DIR" | grep -vxE 'models|venv|user_sessions' || true)" ]; then
        stop "$DIR already holds other files: install into a folder of its own (the default, ./silvann)"
    fi
    say "fetching SiLVANN into $DIR"
    local tmp src; tmp=$(mktemp -d)
    if command -v git >/dev/null; then
        git clone --quiet --depth 1 ${SILVANN_REF:+--branch "$SILVANN_REF"} "$SOURCE" "$tmp/silvann"
        src=$tmp/silvann
    else
        curl -fsSL "$SOURCE/archive/${SILVANN_REF:-HEAD}.tar.gz" | tar -xz -C "$tmp"
        src=$(ls -d "$tmp"/*/ | head -1)
    fi
    (cd "$src" && tar -cf - .) | (cd "$DIR" && tar -xf -)
    rm -rf "$tmp"
}

environment() {
    say "the Python environment ($DIR/venv)"
    [ -x "$DIR/venv/bin/python" ] || python3 -m venv "$DIR/venv"
    "$DIR/venv/bin/python" -m pip install --quiet --upgrade pip
    "$DIR/venv/bin/python" -m pip install --quiet -r "$DIR/requirements.txt"
}

build() {
    say "the cards and the compilers"
    find_cards
    say "building the engine and the families for this machine: $FAMILIES (a few minutes; everything in $DIR/build.log)"
    if ! (cd "$DIR" && env SILVANN_PYTHON="$DIR/venv/bin/python" SILVANN_SILICON_FAMILIES="$FAMILIES" "${BUILD_ENV[@]}" \
              ./build.sh > "$DIR/build.log" 2>&1); then
        tail -n 30 "$DIR/build.log" >&2
        stop "the build failed — the lines above, and all of it in $DIR/build.log"
    fi
    grep -E '^(✅|⛔|== built)' "$DIR/build.log" | sed 's/^/   /'
}

# ── the command ───────────────────────────────────────────────────────────────────────────────────────────────────────
command_file() {
    cat > "$DIR/silvann" <<EOF
#!/usr/bin/env bash
# \`silvann\`, written by install.sh for the release in this folder.
set -e
HERE="$DIR"
BIN="$BIN"
PY="\$HERE/venv/bin/python"
cmd=\${1:-help}; [ \$# -gt 0 ] && shift
case "\$cmd" in
    list|list-models)          exec "\$PY" "\$HERE/silvann_core.py" list-models "\$@" ;;
    download|download-model)   exec "\$PY" "\$HERE/silvann_core.py" download-model "\$@" ;;
    serve)
        [ \$# -gt 0 ] || { echo "silvann serve <model> [config] [--port 8765]   — a model in \$HERE/models"; exit 1; }
        model=\$1; shift
        [ -d "\$model" ] || model="\$HERE/models/\$model"
        cd "\$HERE" && exec "\$PY" start_silvann_server.py --model "\$model" "\$@" ;;
    ui)
        case " \$* " in *" --port "*) ;; *) set -- --port 8090 "\$@" ;; esac
        cd "\$HERE" && exec "\$PY" silvann_ui.py "\$@" ;;
    build)   exec env SILVANN_DIR="\$HERE" SILVANN_BIN="\$BIN" bash "\$HERE/install.sh" --build ;;
    update)  exec env SILVANN_DIR="\$HERE" SILVANN_BIN="\$BIN" bash "\$HERE/install.sh" ;;
    where)   echo "\$HERE" ;;
    *)
        cat <<HELP
silvann list                        the models this release can fetch, and which are here
silvann download <model>            a model into \$HERE/models
silvann serve <model> [config]      the server, on port 8765 (--port to change it)
silvann ui                          the chat in a browser, on port 8090, talking to the server
silvann build                       build again — after installing CUDA or ROCm, say
silvann update                      the newest release, then build — stop the server first
silvann where                       the folder it is installed in
HELP
        ;;
esac
EOF
    chmod +x "$DIR/silvann"
    mkdir -p "$BIN"
    ln -sf "$DIR/silvann" "$BIN/silvann"
}

main() {
    local only_build=0
    [ "${1:-}" = --build ] && only_build=1
    check_machine
    DIR=${SILVANN_DIR:-$PWD/silvann}
    mkdir -p "$DIR"; DIR=$(cd "$DIR" && pwd)
    if [ $only_build -eq 0 ] && [ -z "${SILVANN_FETCHED:-}" ]; then
        fetch
        # the rest by the installer just fetched, which an update makes newer than this one
        exec env SILVANN_FETCHED=1 SILVANN_DIR="$DIR" SILVANN_BIN="$BIN" bash "$DIR/install.sh" "$@"
    fi
    [ $only_build -eq 1 ] || environment
    build
    command_file
    say "done: SiLVANN is in $DIR, and \`silvann\` in $BIN"
    case ":$PATH:" in
        *":$BIN:"*) ;;
        *) note "$BIN is not on your PATH yet: open a new terminal, or  export PATH=\"$BIN:\$PATH\"" ;;
    esac
    [ $only_build -eq 1 ] || cat <<EOF

   silvann list                                       the models, and what each needs (README.md's table)
   silvann download Qwen3.6-35B-A3B_silvann_tq_D8E4   one of them
   silvann serve Qwen3.6-35B-A3B_silvann_tq_D8E4      the server
   silvann ui                                         the chat, at http://localhost:8090
EOF
}

main "$@"
