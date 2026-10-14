#!/usr/bin/env bash
# SiLVANN's Docker image as an app, on Linux: the same actions as docker/windows/silvann.ps1. On Linux install.sh is
# the way we recommend (README.md); the image is for a machine where a container suits better.
#
#   bash docker/silvann_docker.sh install         check docker (or podman) and the card, pull the image, make the volume
#   bash docker/silvann_docker.sh update          the newest image of the same line, the volume (models, conversations)
#                                                 kept; a running app restarted on it
#   bash docker/silvann_docker.sh start           the app on http://localhost:8080, this machine only
#   bash docker/silvann_docker.sh start_shared    the same on every network interface: the LAN reaches the UI
#   bash docker/silvann_docker.sh stop            stopped, the conversations saved first
#   bash docker/silvann_docker.sh delete          the container and the image; the volume (models, conversations) after a second yes
#   bash docker/silvann_docker.sh export <dir>    the image and the volume into <dir>, the models included
#   bash docker/silvann_docker.sh import <dir>    both back, on this or another machine — refused where SiLVANN is
#                                                 installed already: export it or delete it first
#   bash docker/silvann_docker.sh install_nonfree requirements-nonfree.txt (PyAV, for video files) from PyPI into the
#                                                 volume: a container started for it, then stopped; a running app
#                                                 restarted. Again after an update that changes the image's Python
#
#   SILVANN_IMAGE   the image        default ghcr.io/eval-apply-labs/silvann:main — the nightly, built from main.
#                   From v1.0 a major version has its tag (:1), which its minor and patch releases update in place
#   SILVANN_PORT    the UI's port    default 8080
#   SILVANN_ENGINE  docker | podman  default whichever is installed, docker first
#   HF_TOKEN        a Hugging Face token, handed to the container at start: its client downloads faster signed in
set -euo pipefail

IMAGE=${SILVANN_IMAGE:-ghcr.io/eval-apply-labs/silvann:main}
PORT=${SILVANN_PORT:-8080}
NAME=silvann
VOLUME=silvann-data
# the UI saves every conversation before it stops; a long one takes a while to write
STOP_SECONDS=300

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
stop() { printf '\033[31m⛔ %s\033[0m\n' "$*" >&2; exit 1; }
yes_to() { local a; read -r -p "$1 [y/N] " a; [[ $a =~ ^[Yy] ]]; }

ENGINE=${SILVANN_ENGINE:-$(command -v docker >/dev/null && echo docker || (command -v podman >/dev/null && echo podman) || true)}
[ -n "$ENGINE" ] || stop "neither docker nor podman is installed: Docker Engine (docs.docker.com/engine/install) or
   Podman (podman.io), then NVIDIA's container toolkit for the card — docs/users.md, \"In a container\""
E() { "$ENGINE" "$@"; }

# how a container is handed the NVIDIA card: docker's --gpus, podman's CDI device; nothing on a machine without one
gpu_flags() {
    command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || return 0
    if [ "$ENGINE" = podman ]; then echo "--device nvidia.com/gpu=all"; else echo "--gpus all"; fi
}

running() { [ "$(E inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || true)" = true ]; }

wait_ui() {
    local url=http://127.0.0.1:$PORT/ i
    for i in $(seq 1 120); do
        if curl -fsS -o /dev/null "$url" 2>/dev/null; then return 0; fi
        running || { E logs --tail 30 "$NAME" >&2; stop "the container stopped — its last lines are above"; }
        sleep 1
    done
    stop "the UI did not answer on $url in 2 minutes: \`$ENGINE logs $NAME\` says why"
}

open_browser() {
    say "SiLVANN: $1"
    command -v xdg-open >/dev/null && xdg-open "$1" >/dev/null 2>&1 || true
}

install() {
    say "the container engine: $ENGINE"
    E info >/dev/null 2>&1 || stop "$ENGINE is installed and not answering: start it (or its service), then run this again"
    local gpu; gpu=$(gpu_flags)
    if [ -n "$gpu" ]; then
        note "NVIDIA: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | paste -sd ';')"
    else
        note "no NVIDIA card (or no driver): the CPU runs the models, through x86_avx2"
    fi
    grep -qw avx2 /proc/cpuinfo || note "⚠ this CPU has no AVX2: without a card SiLVANN will not run here"
    say "the image: $IMAGE"
    pull
    running && note "the app running now keeps the image it started with: \`bash $0 update\` restarts it on this one"
    E volume inspect "$VOLUME" >/dev/null 2>&1 || E volume create "$VOLUME" >/dev/null
    note "the volume $VOLUME holds the models and the conversations: $(E volume inspect -f '{{.Mountpoint}}' "$VOLUME")"
    if [ -n "$gpu" ]; then
        say "a container sees the card:"
        # shellcheck disable=SC2086
        E run --rm $gpu --entrypoint nvidia-smi "$IMAGE" -L \
            || stop "a container does not see the card: NVIDIA's container toolkit is needed
   (docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run this again"
    fi
    say "done — \`bash $0 start\`, then choose and download a model in the UI's Options"
}

start() {   # the address the UI's port is published on
    local bind=$1 gpu
    if running; then open_browser "http://localhost:$PORT"; return; fi
    E rm -f "$NAME" >/dev/null 2>&1 || true
    gpu=$(gpu_flags)
    # shellcheck disable=SC2086
    E run -d --name "$NAME" $gpu ${HF_TOKEN:+-e HF_TOKEN} -p "$bind$PORT:8080" -v "$VOLUME:/data" \
        --stop-timeout "$STOP_SECONDS" "$IMAGE" >/dev/null
    wait_ui
    if [ -z "$bind" ]; then
        note "shared: every machine that reaches this one on port $PORT reaches the UI, and the UI has no login —"
        note "a conversation's password is its only lock. \`bash $0 start\` keeps it to this machine."
        local ip; ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        [ -n "$ip" ] && note "on the LAN: http://$ip:$PORT"
    fi
    open_browser "http://localhost:$PORT"
}

pull() {
    if ! E pull "$IMAGE"; then
        E image inspect "$IMAGE" >/dev/null 2>&1 || stop "the image $IMAGE could not be pulled"
        note "not pulled: the image built on this machine is used"
    fi
}

# a running app stopped and started again, on the interfaces it had
restart() {
    local bind
    bind=$(E inspect -f '{{range $p, $b := .HostConfig.PortBindings}}{{range $b}}{{.HostIp}}{{end}}{{end}}' "$NAME")
    stop_it
    case "$bind" in 127.0.0.1*) start "127.0.0.1:" ;; *) start "" ;; esac
}

update() {
    E info >/dev/null 2>&1 || stop "$ENGINE is installed and not answering: start it, then run this again"
    say "the newest $IMAGE — the volume $VOLUME, the models and the conversations, stays as it is"
    pull
    # the app restarted when it runs on another image than the one the tag names now
    if running && [ "$(E inspect -f '{{.Image}}' "$NAME")" != "$(E image inspect -f '{{.Id}}' "$IMAGE")" ]; then
        restart
    elif running; then
        note "the app runs on this image already"
    fi
    note "the scripts of this image are in it, at /opt/silvann/docker/"
    say "updated"
}

stop_it() {
    if running; then
        say "stopping SiLVANN — the conversations are saved first"
        E stop -t "$STOP_SECONDS" "$NAME" >/dev/null
    fi
    E rm -f "$NAME" >/dev/null 2>&1 || true
    say "stopped"
}

delete() {
    stop_it
    E rmi "$IMAGE" >/dev/null 2>&1 && say "the image removed" || true
    if E volume inspect "$VOLUME" >/dev/null 2>&1; then
        if yes_to "Delete the volume $VOLUME too — every model and conversation in it?"; then
            E volume rm "$VOLUME" >/dev/null && say "the volume removed"
        else
            note "the volume $VOLUME stays: \`bash $0 install\` uses it again"
        fi
    fi
}

export_to() {
    local dir=${1:-}; [ -n "$dir" ] || stop "export <dir> — a folder for the image and the volume"
    mkdir -p "$dir"; dir=$(cd "$dir" && pwd)
    E volume inspect "$VOLUME" >/dev/null 2>&1 || stop "there is no volume $VOLUME to export"
    stop_it
    say "the image -> $dir/silvann-image.tar"
    E save -o "$dir/silvann-image.tar" "$IMAGE"
    say "the volume (the models included) -> $dir/silvann-data.tar"
    E run --rm --user 0:0 -v "$VOLUME:/data:ro" -v "$dir:/out" --entrypoint tar "$IMAGE" -cf /out/silvann-data.tar -C /data .
    printf '%s\n' "$IMAGE" > "$dir/silvann-image.name"
    say "exported: $(du -sh "$dir" | cut -f1) in $dir — \`bash $0 import $dir\` on the other machine"
}

# what the image leaves out (requirements-nonfree.txt: its wheels carry LGPL and GPL binaries), fetched from PyPI by a
# container started for it on the volume, into /data/python/<the image's Python version> — which the image's
# environment reads when it starts (silvann-data.pth)
install_nonfree() {
    E info >/dev/null 2>&1 || stop "$ENGINE is installed and not answering: start it, then run this again"
    E image inspect "$IMAGE" >/dev/null 2>&1 || stop "there is no image $IMAGE here: \`bash $0 install\` first"
    say "requirements-nonfree.txt (PyAV, for video files) from PyPI into the volume $VOLUME"
    E run --rm -v "$VOLUME:/data" --entrypoint /opt/silvann/venv/bin/python "$IMAGE" -c \
        "import subprocess, sys; sys.exit(subprocess.call([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', '--target', '/data/python/%d.%d' % sys.version_info[:2], '-r', '/opt/silvann/requirements-nonfree.txt']))" \
        || stop "not installed: the lines above say why"
    if running; then
        say "the app restarted, to read it"
        restart
    fi
    say "installed — a video file (MP4 and the like) is read now"
}

import_from() {
    local dir=${1:-}; [ -n "$dir" ] || stop "import <dir> — the folder export wrote"
    dir=$(cd "$dir" && pwd)
    [ -f "$dir/silvann-image.tar" ] && [ -f "$dir/silvann-data.tar" ] || stop "$dir has no silvann-image.tar and silvann-data.tar"
    IMAGE=$(cat "$dir/silvann-image.name" 2>/dev/null || echo "$IMAGE")
    # an installation already here is never overwritten: its image, its container or its volume (its models and
    # conversations) — kept with export, or removed with delete, first; nothing is stopped or loaded before this
    local here=()
    E image inspect "$IMAGE" >/dev/null 2>&1 && here+=("the image $IMAGE")
    E container inspect "$NAME" >/dev/null 2>&1 && here+=("the container $NAME")
    E volume inspect "$VOLUME" >/dev/null 2>&1 && here+=("the volume $VOLUME, with its models and conversations")
    if [ ${#here[@]} -gt 0 ]; then
        stop "SiLVANN is installed here already: $(printf '%s; ' "${here[@]}" | sed 's/; $//').
   Keep it with \`bash $0 export <another folder>\` first,
   or remove it with \`bash $0 delete\` (the volume after a second yes) — then import"
    fi
    say "the image <- $dir/silvann-image.tar"
    E load -i "$dir/silvann-image.tar"
    E volume create "$VOLUME" >/dev/null
    say "the volume <- $dir/silvann-data.tar"
    E run --rm --user 0:0 -v "$VOLUME:/data" -v "$dir:/in:ro" --entrypoint tar "$IMAGE" -xpf /in/silvann-data.tar -C /data
    [ "$IMAGE" = "${SILVANN_IMAGE:-ghcr.io/eval-apply-labs/silvann:main}" ] \
        || note "the image is $IMAGE: start it with SILVANN_IMAGE=$IMAGE bash $0 start"
    say "imported — \`bash $0 start\`"
}

case "${1:-}" in
    install)       install ;;
    update)        update ;;
    start)         start "127.0.0.1:" ;;
    start_shared)  start "" ;;
    stop)          stop_it ;;
    delete)        delete ;;
    export)        export_to "${2:-}" ;;
    import)        import_from "${2:-}" ;;
    install_nonfree) install_nonfree ;;
    *)             sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2 ;;
esac
