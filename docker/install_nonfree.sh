#!/usr/bin/env bash
# What SiLVANN's image leaves out, added to the container setup.sh made — run once, after it:
#
#   curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/install_nonfree.sh | bash
#
# requirements-nonfree.txt: PyAV, through which SiLVANN reads MP4 and other video files (an animated GIF, WebP or PNG
# it reads without). PyAV's wheels carry FFmpeg's libraries under the LGPL and the GPL, which the image does not
# redistribute (ATTRIBUTION.md): this fetches them from PyPI into the container's folder, under python/, where they
# stay through an update and the model's own commands read them and never write them. Then SiLVANN restarts.
#
#   … | bash -s -- --name <name>          a container named otherwise than silvann
#   … | bash -s -- --engine podman|docker where both are installed
set -euo pipefail

NAME=silvann; ENGINE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --name)   NAME=${2:-}; shift ;;
        --engine) ENGINE=${2:-}; shift ;;
        -h|--help) echo "install_nonfree.sh [--name <container>] [--engine podman|docker]"; exit 0 ;;
        *) echo "install_nonfree.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift
done
stop_here() { printf '\n\033[1;31m%s\033[0m\n' "$1" >&2; shift; for l in "$@"; do printf '  %s\n' "$l" >&2; done; exit 1; }

if [ -z "$ENGINE" ]; then
    if command -v podman >/dev/null && podman container inspect "$NAME" >/dev/null 2>&1; then ENGINE=podman
    elif command -v docker >/dev/null && docker container inspect "$NAME" >/dev/null 2>&1; then ENGINE=docker
    else stop_here "There is no SiLVANN container called $NAME." "Set it up first: docker/setup.sh (the README's quick start)."
    fi
fi
"$ENGINE" container inspect "$NAME" >/dev/null 2>&1 || stop_here "There is no container called $NAME under $ENGINE."

DATA=$("$ENGINE" container inspect -f '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Source}}{{end}}{{end}}' "$NAME")
IMAGE=$("$ENGINE" container inspect -f '{{.Config.Image}}' "$NAME")
[ -n "$DATA" ] && [ -n "$IMAGE" ] || stop_here "The container $NAME has no folder at /data: is it one setup.sh made?"

FLAGS=()
if [ "$ENGINE" = podman ]; then
    FLAGS+=(--userns=keep-id)
else
    FLAGS+=(--user "$(id -u):$(id -g)")
fi
VOLUME="$DATA:/data"
if command -v getenforce >/dev/null && [ "$(getenforce 2>/dev/null)" = Enforcing ]; then VOLUME="$VOLUME:Z"; fi

printf '\033[1m%s\033[0m\n' "PyAV (MP4 and other video files) into $DATA/python, from PyPI — FFmpeg's libraries, LGPL and GPL…"
"$ENGINE" run --rm "${FLAGS[@]}" -v "$VOLUME" --entrypoint /opt/silvann/venv/bin/python "$IMAGE" -c "import subprocess, sys
target = '/data/python/%d.%d' % sys.version_info[:2]
sys.exit(subprocess.call([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', '--target', target,
                          '-r', '/opt/silvann/requirements-nonfree.txt']))" \
    || stop_here "pip could not install it (the lines above). SiLVANN runs without it: a video file other than an animated picture is refused."

printf '\033[1m%s\033[0m\n' "Restarting SiLVANN (up to five minutes, while it saves the conversations)…"
# stop, then start once its port is let go: Podman 3.4's `restart` can bind the port again before the stopped one has
#   given it up ("address already in use")
"$ENGINE" stop -t 300 "$NAME" >/dev/null
for i in $(seq 1 20); do "$ENGINE" start "$NAME" >/dev/null 2>&1 && break; sleep 1; done
[ "$("$ENGINE" container inspect -f '{{.State.Running}}' "$NAME")" = true ] \
    || stop_here "PyAV is installed, but $NAME did not start again." "Start it:  $ENGINE start $NAME"
printf '\033[1m%s\033[0m\n' "Done: SiLVANN reads video files from now on. Run this again after an update that changes the image's Python."
