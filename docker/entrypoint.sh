#!/bin/sh
# The container's first process after tini. As root: what the volume holds becomes the app's user's — a model's folder
# pasted in through Windows' Explorer (the scripts' open_data) comes in as root's, and a model reading its experts from
# the disk writes beside them — then the app runs as that user. A container started as another user runs as it is, and
# so does a root that is a user outside (rootless Docker maps the container's root to the user who started it, and its
# uid 1000 to one of that user's subordinate ids): the folder stays the user's, unchanged.
# A GPU handed in through /dev/dri (an integrated GPU, `--device /dev/dri`) belongs to the host's render group, by number:
# the app's user joins a group of that number, so it may open the GPU as the host's users in that group do.
if [ "$(id -u)" = 0 ] && [ "$(awk '$1 == 0 { print $2 }' /proc/self/uid_map)" = 0 ]; then
    for node in /dev/dri/renderD*; do
        [ -e "$node" ] || continue
        gid=$(stat -c %g "$node")
        group=$(getent group "$gid" | cut -d: -f1)
        [ -n "$group" ] || { group=render_host; groupadd -g "$gid" "$group" 2>/dev/null || group=; }
        [ -n "$group" ] && usermod -aG "$group" silvann 2>/dev/null
    done
    find /data -xdev \( ! -user silvann -o ! -group silvann \) -exec chown -h silvann:silvann {} + \
        || echo "⚠ some of /data could not be given to the app's user: a model there may not start" >&2
    exec setpriv --reuid=silvann --regid=silvann --init-groups -- "$@"
fi
exec "$@"
