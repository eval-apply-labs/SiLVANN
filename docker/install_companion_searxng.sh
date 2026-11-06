#!/usr/bin/env bash
# The search companion: SearXNG, a search engine that asks the others and answers in JSON, in a container beside
# SiLVANN's, for the model's (search …) — the alternative to a Brave Search API key. Run once, after setup.sh (or
# install_companions.sh for every one):
#
#   curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/install_companion_searxng.sh | bash
#
# Its settings — a secret key of its own, its JSON format turned on — go in ~/.config/silvann/searxng/, made once; it
# starts as searxng on the companions' network, silvann-companions, where SiLVANN finds it at http://searxng:8080; where
# SiLVANN runs without a container (install.sh), on this machine's loopback, http://127.0.0.1:8888. Run again: the
# newest SearXNG, the settings kept. Then SiLVANN restarted, or Rebuild the tools in its Options.
#
#   … | bash -s -- --name <name>          SiLVANN's container, named otherwise than silvann
#   … | bash -s -- --engine podman|docker where both are installed
#   … | bash -s -- remove                 the companion stopped and removed (its settings kept)
set -euo pipefail
NET=silvann-companions; IT=searxng; IMAGE=docker.io/searxng/searxng:latest
CONF="${XDG_CONFIG_HOME:-$HOME/.config}/silvann/searxng"
NAME=silvann; ENGINE=""; ACTION=install
while [ $# -gt 0 ]; do
    case "$1" in
        --name)   NAME=${2:-}; shift ;;
        --engine) ENGINE=${2:-}; shift ;;
        remove)   ACTION=remove ;;
        -h|--help) sed -n '2,18p' "$0" 2>/dev/null || true; exit 0 ;;
        *) echo "install_companion_searxng.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift
done
say()  { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }
stop_here() { printf '\n\033[1;31m%s\033[0m\n' "$1" >&2; shift; for l in "$@"; do printf '  %s\n' "$l" >&2; done; exit 1; }
# the companions' network, made where it is not — ⛳ at CNI's config version 0.4.0 under Podman 3, which writes 1.0.0 and
#   ships plugins that refuse it (Ubuntu 22.04's 3.4.4: "plugin firewall does not support config version 1.0.0")
make_net() {
    "$ENGINE" network inspect "$NET" >/dev/null 2>&1 && return 0
    "$ENGINE" network create "$NET" >/dev/null
    if [ "$ENGINE" = podman ] && [ "$(podman version -f '{{.Client.Version}}' 2>/dev/null | cut -d. -f1)" = 3 ]; then
        for f in "${XDG_CONFIG_HOME:-$HOME/.config}/cni/net.d/$NET.conflist" "/etc/cni/net.d/$NET.conflist"; do
            [ -w "$f" ] && sed -i 's/"cniVersion": "1.0.0"/"cniVersion": "0.4.0"/' "$f"
        done
    fi
}

if [ -z "$ENGINE" ]; then
    if command -v podman >/dev/null && podman container inspect "$NAME" >/dev/null 2>&1; then ENGINE=podman
    elif command -v docker >/dev/null && docker container inspect "$NAME" >/dev/null 2>&1; then ENGINE=docker
    elif command -v podman >/dev/null; then ENGINE=podman
    elif command -v docker >/dev/null; then ENGINE=docker
    else stop_here "Neither Podman nor Docker is installed." "The companions are containers: install Podman (or Docker) first."
    fi
fi
if [ "$ACTION" = remove ]; then
    "$ENGINE" rm -f "$IT" >/dev/null 2>&1 || true
    say "The search companion is removed (its settings stay in $CONF)."; exit 0
fi

if "$ENGINE" container inspect "$NAME" >/dev/null 2>&1; then
    make_net
    nets=$("$ENGINE" container inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$NAME")
    case " $nets " in
        *" $NET "*) ;;
        *) "$ENGINE" network connect "$NET" "$NAME" >/dev/null 2>&1 \
               || stop_here "SiLVANN's container ($NAME) is not on the companions' network, $NET." \
                            "Run setup.sh once with 'update' — it keeps the folder and puts SiLVANN on it — then this again." ;;
    esac
    WHERE=(--network "$NET"); ADDRESS=http://searxng:8080
else
    WHERE=(-p 127.0.0.1:8888:8080); ADDRESS=http://127.0.0.1:8888
    note "No SiLVANN container called $NAME: the companion goes on this machine's loopback, for SiLVANN installed directly."
fi

mkdir -p "$CONF"
if [ ! -f "$CONF/settings.yml" ]; then
    key=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
    printf 'use_default_settings: true\nserver:\n  secret_key: "%s"\n  limiter: false\n  image_proxy: false\nsearch:\n  formats: [html, json]\n' \
        "$key" > "$CONF/settings.yml"
    note "Its settings: $CONF/settings.yml"
fi
VOLUME="$CONF:/etc/searxng"
if command -v getenforce >/dev/null && [ "$(getenforce 2>/dev/null)" = Enforcing ]; then VOLUME="$VOLUME:Z"; fi

say "Fetching SearXNG…"
"$ENGINE" pull -q "$IMAGE" >/dev/null
"$ENGINE" rm -f "$IT" >/dev/null 2>&1 || true
"$ENGINE" run -d --name "$IT" "${WHERE[@]}" -v "$VOLUME" --restart unless-stopped "$IMAGE" >/dev/null
say "The search companion is running — SiLVANN finds it at $ADDRESS."
[ "$ENGINE" = podman ] && note "Its settings are SearXNG's own user's now: edit them with  podman unshare \$EDITOR $CONF/settings.yml"
note "Restart SiLVANN, or press Rebuild the tools in its Options, and the model's (search …) asks it."
