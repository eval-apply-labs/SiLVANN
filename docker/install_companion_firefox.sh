#!/usr/bin/env bash
# The browser companion: a Firefox, headless, in a container beside SiLVANN's — never in it — that the model's
# (browser …) drives over WebDriver BiDi. Its firewall lets it reach the public internet and nothing private: not
# SiLVANN's container, not this machine, not your network; whatever a page does stays in it, and each connection
# SiLVANN makes gets a Firefox with a fresh profile. Run once, after setup.sh (or install_companions.sh for every one):
#
#   curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/install_companion_firefox.sh | bash
#
# It builds the image from docker/companions/firefox/ — Debian's Firefox ESR, a minute or two the first time — and
# starts it as silvann-firefox on the companions' network, silvann-companions, where SiLVANN finds it by name; where
# SiLVANN runs without a container (install.sh), on this machine's loopback, 127.0.0.1:9222. Run again: the newest
# Firefox. Then SiLVANN restarted, or Rebuild the tools in its Options, and the model has (browser …).
#
#   … | bash -s -- --name <name>          SiLVANN's container, named otherwise than silvann
#   … | bash -s -- --engine podman|docker where both are installed
#   … | bash -s -- remove                 the companion stopped and removed
set -euo pipefail
RAW=https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker
NET=silvann-companions; IT=silvann-firefox
NAME=silvann; ENGINE=""; ACTION=install
while [ $# -gt 0 ]; do
    case "$1" in
        --name)   NAME=${2:-}; shift ;;
        --engine) ENGINE=${2:-}; shift ;;
        remove)   ACTION=remove ;;
        -h|--help) sed -n '2,20p' "$0" 2>/dev/null || true; exit 0 ;;
        *) echo "install_companion_firefox.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift
done
say()  { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }
stop_here() { printf '\n\033[1;31m%s\033[0m\n' "$1" >&2; shift; for l in "$@"; do printf '  %s\n' "$l" >&2; done; exit 1; }
fetch() { if command -v curl >/dev/null; then curl -fsSL "$1"; else wget -qO- "$1"; fi; }
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
    say "The browser companion is removed."; exit 0
fi

# ── where: SiLVANN's container's network, or this machine's loopback ─────────────────────────────────────────────
if "$ENGINE" container inspect "$NAME" >/dev/null 2>&1; then
    make_net
    nets=$("$ENGINE" container inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "$NAME")
    case " $nets " in
        *" $NET "*) ;;
        *) "$ENGINE" network connect "$NET" "$NAME" >/dev/null 2>&1 \
               || stop_here "SiLVANN's container ($NAME) is not on the companions' network, $NET." \
                            "Run setup.sh once with 'update' — it keeps the folder and puts SiLVANN on it — then this again." ;;
    esac
    WHERE=(--network "$NET"); ADDRESS=ws://$IT:9222
else
    WHERE=(-p 127.0.0.1:9222:9222); ADDRESS=ws://127.0.0.1:9222
    note "No SiLVANN container called $NAME: the companion goes on this machine's loopback, for SiLVANN installed directly."
fi

# ── the image ─────────────────────────────────────────────────────────────────────────────────────────────────────
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")
SRC="$HERE/companions/firefox"
if [ ! -f "$SRC/Containerfile" ]; then
    SRC=$(mktemp -d); trap 'rm -rf "$SRC"' EXIT
    for f in Containerfile entrypoint.sh run.sh firewall.nft user.js; do fetch "$RAW/companions/firefox/$f" > "$SRC/$f"; done
fi
say "Building the browser companion (Firefox ESR, a minute or two the first time)…"
"$ENGINE" build -q --pull -t "$IT" -f "$SRC/Containerfile" "$SRC" >/dev/null

# ── the container: its root read-only, its profile in memory, NET_ADMIN only for its firewall as it starts ─────────
"$ENGINE" rm -f "$IT" >/dev/null 2>&1 || true
"$ENGINE" run -d --name "$IT" "${WHERE[@]}" --read-only --tmpfs /tmp:rw,size=1g --tmpfs /run --cap-add NET_ADMIN \
    --memory 2g --restart unless-stopped "$IT" >/dev/null
for i in $(seq 1 60); do
    log=$("$ENGINE" logs "$IT" 2>&1 || true)
    case "$log" in *"BiDi listening"*) break ;; *"refusing to start"*) break ;; esac
    sleep 1
done
case "$log" in
    *"BiDi listening"*) ;;
    *"refusing to start"*) stop_here "The companion's firewall could not be set, so it did not start." \
                                     "Its log:  $ENGINE logs $IT" ;;
    *) stop_here "The companion did not start within a minute." "Its log:  $ENGINE logs $IT" ;;
esac
say "The browser companion is running — SiLVANN finds it at $ADDRESS."
note "Restart SiLVANN, or press Rebuild the tools in its Options, and the model has (browser …)."
note "It reaches the public internet and nothing private. Remove it:  … install_companion_firefox.sh | bash -s -- remove"
