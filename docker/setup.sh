#!/usr/bin/env bash
# SiLVANN in a container, set up and started — the usual machine in one line:
#
#   curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/setup.sh | bash
#
# (or `wget -qO- … | bash` where there is no curl, as on Debian's netinstall)
#
# It finds Podman (or Docker) and the card — an AMD card, an NVIDIA card, or none — checks what the host needs and says
# what is missing, makes the folder SiLVANN keeps everything in (~/SiLVANN), asks for a Brave Search API key for the
# model's web search (Enter skips), pulls the image and starts it; then the chat is at http://localhost:8080.
#
#   … | bash -s -- update        the newest image, the container made again over the same folder
#   … | bash -s -- stop | start  stop it (giving it time to save the conversations), or start it again
#   … | bash -s -- --dry-run     say what it would run, and run nothing
#
#   --card amd|nvidia|cpu   which, where it guesses wrong       --data <folder>   another folder than ~/SiLVANN
#   --port <port>           a port of your own (by default 8080,  --name <name>     another container name than silvann
#                           or the first free one after it)
#   --engine podman|docker  which, where both are installed     --no-search       ask for no search key
#   --image <image>         an image of your own (docker/advanced.md, "Building an image") in place of the published one
#
# Everything else — SearXNG, names under localhost, a card a container, a model over several machines, building an
# image — is in docker/advanced.md. Nothing here needs root, but where the host is missing something that does (a
# group, NVIDIA's description of the card), the script names the command and stops.
set -euo pipefail

ACTION=setup; CARD=""; DATA="$HOME/SiLVANN"; PORT=""; NAME=silvann; ENGINE=""; DRY=0; SEARCH=1; OWN_IMAGE=""
IMAGE_BASE=ghcr.io/eval-apply-labs/silvann
while [ $# -gt 0 ]; do
    case "$1" in
        setup|update|stop|start) ACTION=$1 ;;
        --card)    CARD=${2:-}; shift ;;
        --data)    DATA=${2:-}; shift ;;
        --port)    PORT=${2:-}; shift ;;
        --name)    NAME=${2:-}; shift ;;
        --engine)  ENGINE=${2:-}; shift ;;
        --image)   OWN_IMAGE=${2:-}; shift ;;
        --dry-run) DRY=1 ;;
        --no-search) SEARCH=0 ;;
        -h|--help) cat <<'USAGE'
setup.sh [setup|update|stop|start] [--card amd|nvidia|cpu] [--data <folder>] [--port <port>] [--name <name>]
         [--engine podman|docker] [--image <an image of your own>] [--no-search] [--dry-run]
  setup    (the default) set SiLVANN up in a container and start it; the chat at http://localhost:8080
  update   the newest image, the container made again over the same folder
  stop     stop it, giving it time to save the conversations;  start  start it again
USAGE
            exit 0 ;;
        *) echo "setup.sh: unknown argument '$1' (--help lists them)" >&2; exit 2 ;;
    esac
    shift
done

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }
stop_here() { printf '\n\033[1;31m%s\033[0m\n' "$1" >&2; shift; for l in "$@"; do printf '  %s\n' "$l" >&2; done; exit 1; }
run() { if [ $DRY = 1 ]; then printf '  would run: %s\n' "$*" >&2; else "$@"; fi; }
tty_read() { # tty_read <prompt> <var> [silent]: a line from the terminal even when this script is piped into bash
    local answer=""
    if [ -r /dev/tty ] && { exec 3</dev/tty; } 2>/dev/null; then
        printf '%s' "$1" > /dev/tty
        if [ "${3:-}" = silent ]; then IFS= read -rs answer <&3 || true; echo > /dev/tty; else IFS= read -r answer <&3 || true; fi
        exec 3<&-
    fi
    printf -v "$2" '%s' "$answer"
}

[ "$(uname -s)" = Linux ] || stop_here "SiLVANN's container runs on Linux." \
    "On Windows, install SiLVANN instead: see the README's quick start (install.ps1)."

# ── the container engine ──────────────────────────────────────────────────────────────────────────────────────────
if [ -z "$ENGINE" ]; then
    if command -v podman >/dev/null; then ENGINE=podman
    elif command -v docker >/dev/null; then ENGINE=docker
    else stop_here "Neither Podman nor Docker is installed." \
        "On Ubuntu or Debian:  sudo apt install podman crun uidmap slirp4netns fuse-overlayfs" \
        "On Fedora, Podman is there already. Then run this again."
    fi
fi
command -v "$ENGINE" >/dev/null || stop_here "$ENGINE is not installed."
if [ "$ENGINE" = docker ] && ! docker info >/dev/null 2>&1; then
    stop_here "Docker is installed, but you cannot use it as you." \
        "Either install Podman (sudo apt install podman crun uidmap slirp4netns fuse-overlayfs) and run this again," \
        "or add yourself to the docker group (sudo usermod -aG docker \$USER), log out and in, and run this again."
fi

# ── the port: 8080, or the first free one after it ──────────────────────────────────────────────────────────────────
busy() { if command -v ss >/dev/null; then ss -Htln "( sport = :$1 )" 2>/dev/null | grep -q .; else (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; fi; }
# whether the chat answers on a port — through curl, or wget where there is no curl (Debian's netinstall), or a bare connect
answers() { if command -v curl >/dev/null; then curl -sf -o /dev/null "http://127.0.0.1:$1/"
            elif command -v wget >/dev/null; then wget -q -T 5 -t 1 -O /dev/null "http://127.0.0.1:$1/"
            else (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; fi; }
if command -v curl >/dev/null; then FETCH="curl -fsSL"; else FETCH="wget -qO-"; fi
published() { "$ENGINE" port "$NAME" 8080 2>/dev/null | head -1 | sed 's/.*://'; }
address() { echo "http://silvann.localhost:$1  (or http://localhost:$1)"; }

# ── stop, start ───────────────────────────────────────────────────────────────────────────────────────────────────
exists() { "$ENGINE" container inspect "$NAME" >/dev/null 2>&1; }
# the container started — one made with an NVIDIA card is handed it by the toolkit's description of the card (CDI),
#   which a driver update leaves stale: Podman's own word for that, and what to run
start_it() {
    local err
    if [ $DRY = 1 ]; then run "$ENGINE" start "$NAME"; return; fi
    err=$("$ENGINE" start "$NAME" 2>&1 >/dev/null) && return 0
    case "$err" in
        *CDI*) stop_here "$NAME would not start: $ENGINE cannot see the NVIDIA card it was made with." \
                   "1. NVIDIA's container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" \
                   "2. Run:  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   (again after every driver update)" \
                   "Then run this again — or make the container again on the CPU: this same line with 'bash -s -- update --card cpu'." ;;
        *) stop_here "$NAME would not start:" "$err" ;;
    esac
}
if [ "$ACTION" = stop ]; then
    exists || stop_here "There is no container called $NAME."
    say "Stopping $NAME (up to five minutes, while it saves the conversations)…"
    run "$ENGINE" stop -t 300 "$NAME" >/dev/null; say "Stopped."; exit 0
fi
if [ "$ACTION" = start ]; then
    exists || stop_here "There is no container called $NAME: run this script without 'start' to set it up."
    start_it; say "Started: the chat is at $(address "$(published)")"; exit 0
fi
if [ "$ACTION" = setup ] && exists; then
    say "SiLVANN is set up already (container $NAME)."
    if [ "$("$ENGINE" container inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]; then start_it; fi
    note "The chat: $(address "$(published)")"
    note "Newest version: run this again with 'update'.   Stop: with 'stop'."
    exit 0
fi

if [ -n "$PORT" ]; then
    { [ "$ACTION" = update ] && exists && [ "$(published)" = "$PORT" ]; } || ! busy "$PORT" \
        || stop_here "Port $PORT is taken by another program." "Leave --port out, and the first free one from 8080 is taken."
elif [ "$ACTION" = update ] && exists && [ -n "$(published)" ]; then
    PORT=$(published)                  # the port it had
else
    PORT=8080
    while busy "$PORT"; do PORT=$((PORT + 1)); [ $PORT -le 8099 ] || stop_here "Ports 8080 to 8099 are all taken: name one with --port."; done
    [ $PORT = 8080 ] || note "Port 8080 is taken by another program: SiLVANN's chat goes on $PORT."
fi

# ── the card ──────────────────────────────────────────────────────────────────────────────────────────────────────
if [ -z "$CARD" ]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then CARD=nvidia
    elif [ -e /dev/kfd ]; then CARD=amd
    else CARD=cpu
    fi
fi
case "$CARD" in amd|nvidia|cpu) ;; *) stop_here "--card is amd, nvidia or cpu, not '$CARD'." ;; esac
say "SiLVANN in a container: $ENGINE, $([ $CARD = cpu ] && echo 'no card — the CPU runs the models' || echo "an $(echo $CARD | sed 's/amd/AMD/;s/nvidia/NVIDIA/') card")"

FLAGS=()
if [ "$CARD" = amd ]; then
    # the image is built for the gfx906 generation (Instinct MI50, MI60): another card has no family in it
    other=""
    for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        v=$(awk '/gfx_target_version/ {print $2}' "$p" 2>/dev/null || true)
        [ -n "$v" ] && [ "$v" != 0 ] && [ "$v" != 90006 ] && other=$v
    done
    if [ -n "$other" ] && [ -z "$OWN_IMAGE" ]; then
        stop_here "This AMD card (gfx target $other) is not one the published image is built for (gfx906: MI50, MI60)." \
            "Build an image for it (docker/advanced.md, 'Building an image') and add --image <its name>," \
            "or run on the CPU: add --card cpu."
    fi
    missing=""
    for g in video render; do id -nG | tr ' ' '\n' | grep -qx "$g" || missing="$missing $g"; done
    if [ -n "$missing" ] && [ "$ENGINE" = podman ]; then
        stop_here "You are not in the group(s)$missing, which own the card." \
            "Run:  sudo usermod -aG video,render \$USER" "then log out and in again, and run this again."
    fi
    if [ "$ENGINE" = podman ]; then
        rt=$(podman info --format '{{.Host.OCIRuntime.Name}}' 2>/dev/null || echo "")
        if [ "$rt" != crun ]; then
            command -v crun >/dev/null || stop_here "Podman needs the crun runtime to hand the card in." "Run:  sudo apt install crun" "then run this again."
            if [ ! -e "$HOME/.config/containers/containers.conf" ]; then
                note "Podman set to use crun (~/.config/containers/containers.conf)"
                [ $DRY = 1 ] || { mkdir -p "$HOME/.config/containers"; printf '[engine]\nruntime = "crun"\n' > "$HOME/.config/containers/containers.conf"; }
            else
                stop_here "Podman runs '$rt'; handing the card in needs crun." \
                    "Add  runtime = \"crun\"  under [engine] in ~/.config/containers/containers.conf, then run this again."
            fi
        fi
        FLAGS+=(--group-add keep-groups)
    else
        for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [ -n "$gid" ] && FLAGS+=(--group-add "$gid"); done
    fi
    FLAGS+=(--device /dev/kfd --device /dev/dri)
    IMAGE=$IMAGE_BASE:0-amd
elif [ "$CARD" = nvidia ]; then
    if [ "$ENGINE" = podman ]; then
        pv=$(podman version --format '{{.Client.Version}}' 2>/dev/null || echo 0)
        if [ "$(printf '%s\n4.1\n' "$pv" | sort -V | head -1)" != 4.1 ]; then
            stop_here "Podman $pv cannot hand in an NVIDIA card: that needs Podman 4.1 or later (Ubuntu 24.04's, Debian 12's)." \
                "Or use Docker with NVIDIA's container toolkit:  run this again with --engine docker"
        fi
        if ! { command -v nvidia-ctk >/dev/null && nvidia-ctk cdi list 2>/dev/null | grep -q 'nvidia.com/gpu=all'; }; then
            stop_here "Podman cannot see the NVIDIA card yet: it needs NVIDIA's container toolkit and its description of the card." \
                "1. Install NVIDIA's container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" \
                "2. Run:  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml" "Then run this again."
        fi
        FLAGS+=(--device nvidia.com/gpu=all)
    else
        FLAGS+=(--gpus all)
    fi
    # a card older than CUDA's family takes (compute capability 7.5 on) — Pascal's GTX 10x0 — runs through OpenCL 1.2
    old=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk -F. '$1 * 10 + $2 < 75' | sort -u | paste -sd ',' || true)
    [ -n "$old" ] && note "compute capability $old: older than CUDA's family takes (7.5 and newer) — that card runs through OpenCL 1.2"
    IMAGE=$IMAGE_BASE:0-nvidia
else
    IMAGE=$IMAGE_BASE:0-amd           # either image runs on the CPU alone
fi

[ -z "$OWN_IMAGE" ] || IMAGE=$OWN_IMAGE

# ── what the model's commands may reach ───────────────────────────────────────────────────────────────────────────
# Podman's own seccomp profile lets the kernel's Landlock through from containers/common v0.49; an older one (Ubuntu
# 22.04's) refuses it, and the model's shell is then off: the profile is made again with Landlock's three calls added
if [ "$ENGINE" = podman ]; then
    FLAGS+=(--userns=keep-id)
    host_profile=/usr/share/containers/seccomp.json
    if [ -f "$host_profile" ] && ! grep -q landlock_create_ruleset "$host_profile"; then
        command -v python3 >/dev/null || stop_here "python3 is needed once, to write Podman's security profile." "Run:  sudo apt install python3" "then run this again."
        profile="$HOME/.config/containers/silvann-seccomp.json"
        note "Podman's security profile with Landlock let through (~/.config/containers/silvann-seccomp.json)"
        [ $DRY = 1 ] || { mkdir -p "$HOME/.config/containers"; python3 -c "import json, sys
p = json.load(open(sys.argv[1]))
p['syscalls'].append({'names': ['landlock_create_ruleset', 'landlock_add_rule', 'landlock_restrict_self'], 'action': 'SCMP_ACT_ALLOW'})
json.dump(p, open(sys.argv[2], 'w'))" "$host_profile" "$profile"; }
        FLAGS+=(--security-opt "seccomp=$profile")
    fi
else
    # rootless Docker runs the app as the container's root: the image's own files kept from it
    docker info -f '{{.SecurityOptions}}' 2>/dev/null | grep -q rootless && FLAGS+=(--read-only --tmpfs /tmp)
fi

# ── the folder ────────────────────────────────────────────────────────────────────────────────────────────────────
run mkdir -p "$DATA"
VOLUME="$DATA:/data"
if command -v getenforce >/dev/null && [ "$(getenforce 2>/dev/null)" = Enforcing ]; then VOLUME="$VOLUME:Z"; fi
note "Everything SiLVANN keeps: $DATA (the models, the conversations, the model's workspace)"

# ── web search: a Brave Search API key, kept in the tools file, where the model never reads it ───────────────────
TOOLS="$DATA/tools/default.json"
if [ $SEARCH = 1 ] && [ "$ACTION" = setup ]; then
    has_key=0
    [ -f "$TOOLS" ] && grep -q '"brave_key": *"' "$TOOLS" && has_key=1
    if [ $has_key = 0 ]; then
        echo
        note "The model can search the web with a Brave Search API key (free plan: https://brave.com/search/api/)."
        tty_read "  Paste the key, or press Enter to skip: " KEY silent
        if [ -n "$KEY" ]; then
            if command -v python3 >/dev/null; then
                [ $DRY = 1 ] || { mkdir -p "$DATA/tools"; ( umask 077; KEY="$KEY" python3 -c "import json, os, sys
f = sys.argv[1]
d = json.load(open(f)) if os.path.isfile(f) else {}
e = d.setdefault('search', {})
e.setdefault('enabled', 'auto')
e.setdefault('config', {})['brave_key'] = os.environ['KEY']
json.dump(d, open(f, 'w'), indent=2)" "$TOOLS" ); }
                note "Kept in $TOOLS (the server reads it; the model does not)."
            else
                note "python3 is missing, so the key was not written: put it in $TOOLS later (docs/tools.md)."
            fi
            KEY=""
        else
            note "No web search for now: add a key later in $TOOLS (search → brave_key)."
        fi
    fi
fi

# ── the image, and the container ──────────────────────────────────────────────────────────────────────────────────
echo
if [ -z "$OWN_IMAGE" ]; then
    say "Fetching $IMAGE (under 1 GB the first time)…"
    run "$ENGINE" pull "$IMAGE"
fi
if [ "$ACTION" = update ] && exists; then
    say "Replacing the old container (the folder stays)…"
    run "$ENGINE" stop -t 300 "$NAME" >/dev/null || true
    run "$ENGINE" rm "$NAME" >/dev/null
    # the port let go before the new one binds it (Podman 3.4 can still hold it a moment)
    [ $DRY = 1 ] || for i in $(seq 1 20); do busy "$PORT" || break; sleep 1; done
fi
say "Starting SiLVANN…"
run "$ENGINE" run -d --name "$NAME" "${FLAGS[@]}" -p "127.0.0.1:$PORT:8080" -v "$VOLUME" "$IMAGE" >/dev/null
if [ $DRY = 0 ]; then
    for i in $(seq 1 90); do answers "$PORT" && break; sleep 2; done
    answers "$PORT" || stop_here "SiLVANN did not answer on port $PORT." \
        "Its log says why:  $ENGINE logs $NAME"
fi
echo
[ $DRY = 1 ] && { say "(a dry run: nothing was changed)"; exit 0; }
say "SiLVANN is running: open $(address "$PORT")"
note "In Options, 'Download models' fetches a model and 'start' runs it."
note "Put files for the model in $DATA/workspace."
note "Stop: $ENGINE stop -t 300 $NAME     Start again: $ENGINE start $NAME"
note "Newest version: this same line with 'bash -s -- update' in place of 'bash'."
note "To read MP4 and other video files (PyAV, which the image leaves out), once:"
note "  $FETCH https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/install_nonfree.sh | bash"
