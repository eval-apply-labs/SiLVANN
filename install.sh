#!/usr/bin/env bash
# SiLVANN's installer: the release, its Python environment and its build, in a folder of its own under the current one,
# and a `silvann` command.
#
#   curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.sh | bash
#
# It installs nothing system-wide and asks for no password: a tool it needs and does not find is named, with the line
# that installs it, and the installer stops. Run it again to update; models/, user_sessions/, the venv and the folder
# `silvann models` named stay.
#
#   SILVANN_DIR      where it installs                         default ./silvann (SILVANN_DIR=. for the current folder)
#   SILVANN_SOURCE   the repository it fetches                 default https://github.com/eval-apply-labs/SiLVANN
#   SILVANN_REF      a branch of it                            default the repository's own
#   SILVANN_BIN      where the `silvann` command goes          default ~/.local/bin
#   SILVANN_NVCC     CUDA's compiler, when it is not found     default nvcc on the PATH, else /usr/local/cuda*/bin/nvcc
#   SILVANN_SILICON_FAMILIES  the families to build, e.g. "x86_avx2 nvidia_cuda12"   default the ones this machine's
#                    CPU and cards need — named for a build where the cards are not, as the Docker image's is
#   SILVANN_NONFREE  0 leaves requirements-nonfree.txt out (PyAV, for video files), as the Docker image does   default 1
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
        stop "SiLVANN runs on Linux. On Windows, run its Docker image (docker/README.md), or boot Linux from a USB SSD
   (Ubuntu 22.04, the system it is tested on): docs/users.md, \"On Windows\"."
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
    if [ -n "${SILVANN_SILICON_FAMILIES:-}" ]; then
        FAMILIES=$SILVANN_SILICON_FAMILIES
        note "the families SILVANN_SILICON_FAMILIES names: $FAMILIES"
    fi
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
    # what it reads more with and runs without: a failure is said, not fatal
    if [ "${SILVANN_NONFREE:-1}" != 0 ] && [ -f "$DIR/requirements-nonfree.txt" ]; then
        "$DIR/venv/bin/python" -m pip install --quiet -r "$DIR/requirements-nonfree.txt" \
            || note "requirements-nonfree.txt was not installed: SiLVANN runs without it (a video file other than an animated picture is refused)"
    fi
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
# the models and the conversations: the folder \`silvann models\` named, kept in data_folder (an update leaves it), else
# this one — SILVANN_DATA in the environment before either
DATA="\$HERE"
[ -s "\$HERE/data_folder" ] && DATA=\$(cat "\$HERE/data_folder")
export SILVANN_DATA="\${SILVANN_DATA:-\$DATA}"
usage() {
    cat <<HELP
silvann list                        the models this release can fetch, and which are here
silvann download <model>            a model into \$SILVANN_DATA/models
silvann token [--set | --forget]    the Hugging Face token the downloads use (faster than none)
silvann serve <model> [config]      the server, on port 8765 (--port to change it)
silvann ui                          the chat in a browser, on port 8090, talking to the server
silvann_cli                         the chat in this terminal (--notools, --nothink), talking to the server
silvann models [folder]             where the models and the conversations are — another folder, a drive's, from now on
silvann build                       build again — after installing CUDA or ROCm, say
silvann update                      the newest release, then build — stop the server first
silvann where                       the folder it is installed in
silvann help                        this list
HELP
}
cmd=\${1:-help}; [ \$# -gt 0 ] && shift
case "\$cmd" in
    list|list-models)          exec "\$PY" "\$HERE/silvann_core.py" list-models "\$@" ;;
    download|download-model)   exec "\$PY" "\$HERE/silvann_core.py" download-model "\$@" ;;
    token|hf-token)            exec "\$PY" "\$HERE/silvann_core.py" hf-token "\$@" ;;
    serve)
        [ \$# -gt 0 ] || { echo "silvann serve <model> [config] [--port 8765]   — a model in \$SILVANN_DATA/models"; exit 1; }
        model=\$1; shift
        [ -d "\$model" ] || model="\$SILVANN_DATA/models/\$model"
        cd "\$HERE" && exec "\$PY" start_silvann_server.py --model "\$model" "\$@" ;;
    models)
        if [ \$# -eq 0 ]; then
            echo "the models and the conversations are in \$SILVANN_DATA"
            [ "\$SILVANN_DATA" = "\$DATA" ] || echo "   (SILVANN_DATA in the environment; silvann models set \$DATA)"
            exit 0
        fi
        mkdir -p "\$1/models" "\$1/user_sessions" || { echo "silvann: \$1 cannot be written" >&2; exit 1; }
        to=\$(cd "\$1" && pwd)
        if [ "\$to" = "\$HERE" ]; then rm -f "\$HERE/data_folder"; else echo "\$to" > "\$HERE/data_folder"; fi
        echo "the models and the conversations are in \$to from now on"
        if [ "\$to" != "\$DATA" ]; then
            # what is the user's in the folder left behind, by name: a model's folder, a skill's, the conversations — not
            # the release's own files in models/ and skills/ when that folder is this one (an update needs them here)
            kept=\$(cd "\$DATA" && for f in models/* skills/* user_sessions workspace tools.json tools_library.json \\
                         .huggingface_token ui_last_start.json; do
                         case \$f in models/*|skills/*) [ -d "\$f" ] || continue ;;
                                     *) [ -n "\$(ls -A "\$f" 2>/dev/null)" ] || continue ;; esac
                         if [ "\$DATA" = "\$HERE" ] && [ "\${f%%/*}" = skills ]; then
                             [ -d "\$HERE/.git" ] || continue
                             ! git -C "\$HERE" ls-files --error-unmatch -- "\$f" >/dev/null 2>&1 || continue
                         fi
                         printf '%s ' "\$f"; done)
            [ -z "\$kept" ] || echo "   what was in \$DATA stays there — move it across to keep using it: \$kept"
            [ "\$DATA" != "\$HERE" ] || [ -d "\$HERE/.git" ] || \\
                echo "   and your own skills in \$HERE/skills, if you made any (the ones the release ships stay)"
        fi ;;
    ui)
        case " \$* " in *" --port "*) ;; *) set -- --port 8090 "\$@" ;; esac
        cd "\$HERE" && exec "\$PY" silvann_ui.py "\$@" ;;
    build)   exec env SILVANN_DIR="\$HERE" SILVANN_BIN="\$BIN" bash "\$HERE/install.sh" --build ;;
    update)  exec env SILVANN_DIR="\$HERE" SILVANN_BIN="\$BIN" bash "\$HERE/install.sh" ;;
    where)   echo "\$HERE" ;;
    help|-h|--help)   usage ;;
    *)       { echo "silvann: no command \"\$cmd\" — these are:"; usage; } >&2; exit 1 ;;
esac
EOF
    chmod +x "$DIR/silvann"
    # `silvann_cli`: the chat in a terminal, a command of its own
    cat > "$DIR/silvann_cli" <<EOF
#!/usr/bin/env bash
# \`silvann_cli\`, written by install.sh for the release in this folder: the chat in this terminal (silvann_cli.py).
exec "$DIR/venv/bin/python" "$DIR/silvann_cli.py" "\$@"
EOF
    chmod +x "$DIR/silvann_cli"
    mkdir -p "$BIN"
    ln -sf "$DIR/silvann" "$BIN/silvann"
    ln -sf "$DIR/silvann_cli" "$BIN/silvann_cli"
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

   silvann list                                       the models, their sizes, and which are here
   silvann download silvann_tq_D8E4_qwen3.6_35b_a3b   one of them
   silvann serve silvann_tq_D8E4_qwen3.6_35b_a3b      the server
   silvann ui                                         the chat, at http://localhost:8090
   silvann_cli                                        or the chat in this terminal
   silvann help                                       every command
EOF
}

main "$@"
