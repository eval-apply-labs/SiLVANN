#!/bin/sh
# The firewall set while the container still has NET_ADMIN, then Firefox as `browser` with every capability dropped.
# Firefox's remote agent listens on the container's loopback; socat hands SiLVANN's connection on port 9222 to it
# (▶ run.sh: a Firefox a connection).
set -eu
RESOLVER=$(awk '/^nameserver/ && $2 ~ /^[0-9.]+$/ {print $2; exit}' /etc/resolv.conf)
[ -n "$RESOLVER" ] || RESOLVER=127.0.0.1
sed "s/RESOLVER/$RESOLVER/g" /etc/silvann/firewall.nft | nft -f - || { echo "the firewall could not be set: refusing to start" >&2; exit 1; }
mkdir -p /tmp/profile /tmp/downloads /tmp/home/.cache && cp /etc/silvann/user.js /tmp/profile/ \
    && chown -R browser:browser /tmp/profile /tmp/downloads /tmp/home
# ⛳ A HOME FIREFOX CAN WRITE: with the root read-only and its home there, its remote agent never starts
export HOME=/tmp/home XDG_CACHE_HOME=/tmp/home/.cache
echo "firewall set: out to the public internet and the resolver $RESOLVER only" >&2
exec setpriv --reuid=browser --regid=browser --init-groups --inh-caps=-all --bounding-set=-all --no-new-privs -- \
    /etc/silvann/run.sh
