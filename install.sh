#!/usr/bin/env bash
# SiLVANN's installer: the release, its Python environment and its build, in a folder of its own under the current one,
# and a `silvann` command.
#
#   curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/install.sh | bash
#
# It installs nothing system-wide and asks for no password: a tool it needs and does not find is named, with the line
# that installs it, and the installer stops. Run it again to update; models/, user_sessions/, the venv and the folder
# `silvann data` named stay.
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
        stop "install.sh is Linux's. On Windows, install.ps1 (README.md), or boot Linux from a USB SSD
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
    # ⛳ Python's headers (Python.h), which the engine's module is compiled against: a separate package on Debian and
    #   Ubuntu, so the build stopped at `#include <Python.h>` without them (a tester's Python 3.12, 2026-10-21)
    if [ -n "$py" ] && ! "$py" -c 'import os, sysconfig, sys; sys.exit(0 if os.path.isfile(os.path.join(sysconfig.get_path("include"), "Python.h")) else 1)' 2>/dev/null; then
        missing+=(python3-dev)
    fi
    command -v g++ >/dev/null || missing+=(g++)
    command -v git >/dev/null || command -v tar >/dev/null || missing+=(git)
    command -v git >/dev/null || command -v curl >/dev/null || missing+=(curl)
    # ⛳ without git the release comes as GitHub's archive (`fetch`), a URL only GitHub has: any other source — a Gitea
    #   repository, one that asks for a login — is cloned, so it needs git (a tester's tarball was a login page, 2026-10-21)
    case "$SOURCE" in https://github.com/*) ;; *) command -v git >/dev/null || missing+=(git) ;; esac
    if [ ${#missing[@]} -gt 0 ]; then
        stop "missing: ${missing[*]}. On Ubuntu or Debian:  sudo apt update && sudo apt install -y ${missing[*]}
   then run this installer again."
    fi
}

# a Turing card's name (nvidia-smi's): whether it has no tensor cores — the list src/silicon_families/nvidia_tensor_cores.cuh
#   keeps for the engine's own choice
no_tensor_cores() {
    case "$1" in
        *"GTX 1630"*|*"GTX 1650"*|*"GTX 1660"*|*MX450*|*MX550*|*"Quadro T1000"*|*"Quadro T2000"*|*" T400"*|*" T500"*|*" T550"*|\
        *" T600"*|*" T1000"*|*" T1200"*) return 0 ;;
    esac
    return 1
}

# the cards, and the compiler each one's family is built with; what is chosen goes to build.sh as its variables
find_cards() {
    BUILD_ENV=()
    # ⭐ THE OPENCL FAMILIES ALWAYS (⚖ 2026-10-21 "why dont we build opencl2 and opencl 1 automatically so there is nothing
    #   no one needs to do?"): they need nothing installed to be built (the headers carried, the loader a stub), and at run
    #   time they take only the cards no other family does — an NVIDIA card older than CUDA's (OpenCL 1.2,
    #   nvidia_opencl1), an Intel GPU (intel_opencl2 from Gen9, intel_opencl1_igpu before), an AMD card without ROCm's HIP
    #   (khronos_opencl2) — so a machine whose cards CUDA or ROCm run is unchanged
    FAMILIES="x86_avx2 khronos_opencl2 intel_opencl2 nvidia_opencl1 intel_opencl1_igpu"
    local opencl_card=""
    if [ "$(uname -m)" = aarch64 ]; then
        # ⭐ AN ARM MACHINE'S OWN: arm_neon (Armv8.2 with half precision and the dot product) and arm_gb10 (the DGX Spark's
        #   GB10), one source, arm_base — the build skips x86_avx2 here, and each counts no device on a processor without
        #   what its name promises
        FAMILIES="arm_neon arm_gb10 khronos_opencl2 intel_opencl2 nvidia_opencl1 intel_opencl1_igpu"
        note "the CPU: $(grep -m1 -E 'CPU part|model name' /proc/cpuinfo | sed 's/.*: //') — Arm$(grep -qw asimddp /proc/cpuinfo && echo ', the dot product')"
    elif grep -qw avx2 /proc/cpuinfo; then
        note "the CPU: $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //') — AVX2$(grep -qw avx512bw /proc/cpuinfo && echo ', AVX-512')"
    else
        note "the CPU has no AVX2: the CPU's own family will not run here; the engine's portable host family will"
    fi
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        local cards caps nvcc
        # ⛳ each query ends `|| true`: under pipefail a pipeline that fails or matches nothing would end the installer
        #   here with no message — a driver that does not know the field, a card the user may not open
        cards=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' || true)
        note "NVIDIA: $cards"
        # ⛳ CUDA'S FAMILY TAKES COMPUTE 7.5 AND NEWER (its targets, sm_75 …) BUT A TURING CARD WITHOUT TENSOR CORES: an older
        #   card — Pascal's GTX 10x0, compute 6.1 — is not built for (CUDA 13 builds no sm_61, and 12.9 stops at its tensor
        #   cores), and a GTX 16xx — compute 7.5, no tensor cores (src/silicon_families/nvidia_tensor_cores.cuh) — runs CUDA's
        #   tensor-core kernels slower than OpenCL 1.2's: both run through OpenCL 1.2
        local c n old="" new="" said=""
        while IFS=, read -r n c; do
            c=$(echo "$c" | tr -d ' .'); n=$(echo "$n" | sed 's/^ *//; s/ *$//')
            [ -n "$c" ] || continue
            if [ "$c" = 75 ] && no_tensor_cores "$n"; then old="$old${old:+,}$c"; said="$said${said:+, }$n (7.5, no tensor cores)"
            elif [ "$c" -ge 75 ] 2>/dev/null; then new="$new${new:+,}$c"
            else old="$old${old:+,}$c"; said="$said${said:+, }$n (${c%?}.${c: -1})"; fi
        done < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true)
        if [ -n "$old" ]; then
            note "   $said: not CUDA's family (compute 7.5 and newer, with tensor cores) — runs through OpenCL 1.2"
            opencl_card=1
        fi
        caps=$(echo "$new" | tr ',' '\n' | sort -u | paste -sd ',' || true)
        new=$caps
        if [ -n "$new" ] || [ -z "$old" ]; then          # a card CUDA's family takes, or none whose compute was read
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
    fi
    if [ -e /dev/kfd ]; then
        if [ -x /opt/rocm/bin/hipcc ]; then
            local gfx
            # no gfx when rocminfo cannot open the card (a user outside the `render` group): the build is then for every
            #   gfx the family lists, and `|| true` keeps the empty match from ending the installer. An agent's own name
            #   alone (`Name: gfx950`): its ISA lines name a generic target too (`amdgcn-amd-amdhsa--gfx9-4-generic`),
            #   whose `gfx9` is no card's (`MEASURED` an MI355X, 2026-11-09)
            gfx=$( (/opt/rocm/bin/rocminfo 2>/dev/null || true) | awk '$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/ {print $2}' \
                | sort -u | paste -sd ',' || true)
            # ⭐ TWO AMD FAMILIES, BY THE ROCm HERE: HIP 7 and later build the MI100 on (amd_rocm10, their matrix cores),
            #   HIP 6 the MI50 (amd_rocm6_wave64) — each builds only the targets of its own (build_silicon_families.sh)
            local hipv fam
            hipv=$( (/opt/rocm/bin/hipcc --version 2>/dev/null || true) | grep -oE 'HIP version: [0-9]+' | grep -oE '[0-9]+$' || true)
            if [ "${hipv:-0}" -ge 7 ]; then fam=amd_rocm10; else fam=amd_rocm6_wave64; fi
            note "AMD: ${gfx:-a card} — built with /opt/rocm/bin/hipcc (HIP ${hipv:-?}): the family $fam"
            case "$fam,$gfx" in
                amd_rocm6_wave64,*gfx9[0-9][0-9a-f]*) echo "$gfx" | tr ',' '\n' | grep -qvx gfx906 \
                    && note "   ⚠ ROCm 6 builds the MI50 (gfx906) alone: a card from the MI100 on needs ROCm 7 or later (amd_rocm10)" ;;
                amd_rocm10,*gfx906*) note "   ⚠ the MI50 (gfx906) needs ROCm 6 (amd_rocm6_wave64): ROCm 7 does not build it" ;;
            esac
            # ⭐ A RADEON OF RDNA4 IS amd_rocm10's, IN WAVE64 — the targets its row lists
            #   (src/silicon_families/amd_rocm10/includes.cuh); any other Radeon, or one beside ROCm 6, runs OpenCL 2.0's
            local one
            for one in $(echo "$gfx" | tr ',' ' '); do
                case "$fam,$one" in
                    amd_rocm10,gfx1200|amd_rocm10,gfx1201) ;;
                    amd_rocm6_wave64,gfx1200|amd_rocm6_wave64,gfx1201)
                        note "   ⚠ $one, a Radeon: ROCm 7.2 or later builds it (amd_rocm10, in wave64); with ROCm 6 it runs OpenCL 2.0's" ;;
                    *,gfx1[0-9]*) note "   ⚠ $one, a Radeon no AMD family lists: it runs OpenCL 2.0's (khronos_opencl2)" ;;
                esac
            done
            [ -n "$gfx" ] && BUILD_ENV+=("SILVANN_HIP_ARCHES=$gfx")
            FAMILIES="$FAMILIES $fam"
        else
            note "AMD: a card, and no ROCm — its family needs ROCm's hipcc (/opt/rocm: 6 for the MI50, 7 or later from the MI100 on), then \`silvann build\`"
            opencl_card=1
        fi
    fi
    # an OpenCL family runs on the machine's own OpenCL loader, which a card's driver usually brings
    if [ -n "$opencl_card" ] && ! { ldconfig -p 2>/dev/null | grep -q 'libOpenCL\.so\.1'; }; then
        note "   ⚠ no OpenCL loader (libOpenCL.so.1): sudo apt install -y ocl-icd-libopencl1 — until then the CPU runs the models"
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
    if [ ! -e "$DIR/install.sh" ] && [ -n "$(ls -A "$DIR" | grep -vxE 'models|venv|user_sessions' || true)" ]; then
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
    "$DIR/venv/bin/python" -m pip install --quiet -r "$DIR/scripts/requirements.txt"
    # what it reads more with and runs without: a failure is said, not fatal
    if [ "${SILVANN_NONFREE:-1}" != 0 ] && [ -f "$DIR/scripts/requirements-nonfree.txt" ]; then
        "$DIR/venv/bin/python" -m pip install --quiet -r "$DIR/scripts/requirements-nonfree.txt" \
            || note "requirements-nonfree.txt was not installed: SiLVANN runs without it (a video file other than an animated picture is refused)"
    fi
}

build() {
    say "the cards and the compilers"
    find_cards
    say "building the engine and the families for this machine: $FAMILIES (a few minutes; everything in $DIR/build.log)"
    if ! (cd "$DIR" && env SILVANN_PYTHON="$DIR/venv/bin/python" SILVANN_SILICON_FAMILIES="$FAMILIES" ${BUILD_ENV[@]+"${BUILD_ENV[@]}"} \
              scripts/build.sh > "$DIR/build.log" 2>&1); then
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
# the data — the models, the conversations and the rest kept for the user: the folder \`silvann data\` named, kept in
# data_folder (an update leaves it), else this one — SILVANN_DATA in the environment before either
DATA="\$HERE"
[ -s "\$HERE/data_folder" ] && DATA=\$(cat "\$HERE/data_folder")
export SILVANN_DATA="\${SILVANN_DATA:-\$DATA}"
usage() {
    cat <<HELP
silvann list                        the models this release can fetch, and which are here
silvann download <model>            a model into \$SILVANN_DATA/models — when it is there, what changed in its pack
silvann check [model]               update or done, each model here — with one, the files its update fetches
silvann token [--set | --forget]    the Hugging Face token the downloads use (faster than none)
silvann users [add|passwd|remove|adopt] <name>   the users: each a sign-in, a folder and their own conversations
silvann serve <model> [config]      the server, on port 8765 (--port to change it)
silvann ui                          the chat in a browser, on port 8090, talking to the server
silvann_cli                         the chat in this terminal (--notools, --nothink), talking to the server
silvann registry                    the registry of a network's servers, on port 8770 (--port), for the subagents
silvann data [folder]               where the models, the conversations and the rest are kept — another folder, a drive's
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
    check|check-model)         [ \$# -eq 0 ] && exec "\$PY" "\$HERE/silvann_core.py" check-versions
                               exec "\$PY" "\$HERE/silvann_core.py" check-model "\$@" ;;
    token|hf-token)            exec "\$PY" "\$HERE/silvann_core.py" hf-token "\$@" ;;
    split|split-experts)       exec "\$PY" "\$HERE/silvann_core.py" split-experts "\$@" ;;
    stitch|stitch-experts)     exec "\$PY" "\$HERE/silvann_core.py" stitch-experts "\$@" ;;
    users|user)                exec "\$PY" "\$HERE/silvann_core.py" users "\$@" ;;
    serve)
        [ \$# -gt 0 ] || { echo "silvann serve <model> [config] [--port 8765]   — a model in \$SILVANN_DATA/models"; exit 1; }
        model=\$1; shift
        [ -d "\$model" ] || model="\$SILVANN_DATA/models/\$model"
        cd "\$HERE" && exec "\$PY" start_silvann_server.py --model "\$model" "\$@" ;;
    data)
        if [ \$# -eq 0 ]; then
            echo "the models, the conversations and the rest are kept in \$SILVANN_DATA"
            [ "\$SILVANN_DATA" = "\$DATA" ] || echo "   (SILVANN_DATA in the environment; silvann data set \$DATA)"
            exit 0
        fi
        mkdir -p "\$1/models" "\$1/user_sessions" || { echo "silvann: \$1 cannot be written" >&2; exit 1; }
        to=\$(cd "\$1" && pwd)
        if [ "\$to" = "\$HERE" ]; then rm -f "\$HERE/data_folder"; else echo "\$to" > "\$HERE/data_folder"; fi
        echo "the models, the conversations and the rest are kept in \$to from now on"
        if [ "\$to" != "\$DATA" ]; then
            # what is the user's in the folder left behind, by name: a model's folder, a skill's, the conversations — not
            # the release's own files in models/ and skills/ when that folder is this one (an update needs them here)
            kept=\$(cd "\$DATA" && for f in models/* skills/* user_sessions workspace tools tools.json tools_library.json \\
                         .huggingface_token ui_last_start.json; do
                         case \$f in (models/*|skills/*) [ -d "\$f" ] || continue ;;
                                     (*) [ -n "\$(ls -A "\$f" 2>/dev/null)" ] || continue ;; esac
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
    registry)  cd "\$HERE" && exec "\$PY" silvann_registry.py "\$@" ;;
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

# ── the models here: each one's program for this release, from its pack (a boot fetches it otherwise) ──────────────
programs() {
    local data="$DIR"
    [ -s "$DIR/data_folder" ] && data=$(cat "$DIR/data_folder")
    [ -d "${SILVANN_DATA:-$data}/models" ] || return 0
    say "The models here: each one's program for this release"
    SILVANN_DATA="${SILVANN_DATA:-$data}" "$DIR/venv/bin/python" "$DIR/silvann_core.py" update-programs \
        || note "not checked now — each model's boot checks its program again"
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
    [ $only_build -eq 1 ] || programs
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
