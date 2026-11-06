#!/bin/sh
# As `browser`, every capability dropped (▶ entrypoint.sh): a Firefox for each connection SiLVANN makes. Firefox keeps a
# BiDi session after its client goes, and takes no other while it does — so each connection gets a Firefox of its own, its
# profile fresh, and when the connection closes, cleanly or not, that Firefox goes with it.
while true; do
    rm -rf /tmp/profile /tmp/downloads/* && mkdir -p /tmp/profile && cp /etc/silvann/user.js /tmp/profile/
    firefox-esr --headless --profile /tmp/profile --no-remote --remote-debugging-port 9223 --window-size 1280,800 &
    ff=$!
    i=0                                                         # its remote agent on 9223, hex 2407
    while ! grep -q ':2407 00000000:0000 0A' /proc/net/tcp && [ $i -lt 120 ]; do sleep 0.25; i=$((i + 1)); done
    socat TCP-LISTEN:9222,reuseaddr TCP:127.0.0.1:9223          # one connection, then back here
    kill "$ff" 2>/dev/null; wait "$ff" 2>/dev/null
done
