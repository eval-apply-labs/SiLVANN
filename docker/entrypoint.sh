#!/bin/sh
# The container's first process after tini. As root: what the volume holds becomes the app's user's — a model's folder
# pasted in through Windows' Explorer (the scripts' open_data) comes in as root's, and a model reading its experts from
# the disk writes beside them — then the app runs as that user. A container started as another user runs as it is.
if [ "$(id -u)" = 0 ]; then
    find /data -xdev \( ! -user silvann -o ! -group silvann \) -exec chown -h silvann:silvann {} + \
        || echo "⚠ some of /data could not be given to the app's user: a model there may not start" >&2
    exec setpriv --reuid=silvann --regid=silvann --init-groups -- "$@"
fi
exec "$@"
