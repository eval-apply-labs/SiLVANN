#!/usr/bin/env bash
# SiLVANN's companions — containers beside its own that the model's tools use — each set up by its own script,
# install_companion_<name>.sh, which says what it is. A companion is its script and its line below. Run once, after
# setup.sh:
#
#   curl -fsSL https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker/install_companions.sh | bash
#
# Its arguments go to every one: --name <SiLVANN's container>, --engine podman|docker, remove.
set -euo pipefail
RAW=https://raw.githubusercontent.com/eval-apply-labs/SiLVANN/HEAD/docker
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")
companion() {
    local name=$1; shift
    if [ -f "$HERE/install_companion_$name.sh" ]; then bash "$HERE/install_companion_$name.sh" "$@"
    elif command -v curl >/dev/null; then curl -fsSL "$RAW/install_companion_$name.sh" | bash -s -- "$@"
    else wget -qO- "$RAW/install_companion_$name.sh" | bash -s -- "$@"
    fi
}
companion firefox "$@"           # (browser …): a Firefox of the model's own, firewalled from everything private
companion searxng "$@"           # (search …): SearXNG, the search engine of your own
