#!/bin/sh
# update/linux.sh — SiLVANN's updater on Linux (NN-203), as update\windows.exe is on Windows, in the portable app's
# folder beside windows/, linux/ and data/. With SiLVANN closed it asks GitHub for the latest release, fetches it, asks
# whether the current version is to be removed, and puts the new one in linux/ and windows/
# (▶ scripts/portable/silvann_update.py, which this runs with the Linux part's Python).
#
#   update/linux.sh        from a terminal, or by a double-click in a file manager that runs scripts
UPDATE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
HERE=$(dirname -- "$UPDATE")
PY="$HERE/linux/python/bin/python3"
APP="$UPDATE/scripts/portable/silvann_update.py"

say() {   # the terminal's line always; a message box where there is a display and zenity or kdialog
    echo "SiLVANN: $1" >&2
    if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        if command -v zenity > /dev/null 2>&1; then zenity --error --no-wrap --title SiLVANN --text "$1" 2> /dev/null
        elif command -v kdialog > /dev/null 2>&1; then kdialog --title SiLVANN --error "$1" 2> /dev/null
        fi
    fi
}

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    say "This updater runs on Linux on 64-bit x86, and this machine is $(uname -s) on $(uname -m). On Windows open update\\windows.exe."
    exit 1
fi
if [ ! -f "$PY" ] || [ ! -f "$APP" ]; then
    say "This folder is not SiLVANN's whole: linux/python/bin/python3 or update/scripts/portable/silvann_update.py is missing. Unzip silvann_portable_x64.zip again, whole."
    exit 1
fi

# The updater asks its questions in a terminal. Started from a desktop without one, it opens one; with neither a
# terminal nor a desktop, a program drives it through its pipes.
WAIT=
if [ ! -t 0 ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    if [ -z "${SILVANN_UPDATE_TERMINAL:-}" ]; then
        export SILVANN_UPDATE_TERMINAL=1
        for T in x-terminal-emulator gnome-terminal konsole xfce4-terminal xterm; do
            if command -v "$T" > /dev/null 2>&1; then
                case "$T" in
                    gnome-terminal) exec "$T" -- "$UPDATE/linux.sh" ;;
                    *) exec "$T" -e "$UPDATE/linux.sh" ;;
                esac
            fi
        done
    fi
    say "SiLVANN's updater asks its questions in a terminal, and this desktop names none: run update/linux.sh from one."
    exit 1
fi
if [ -n "${SILVANN_UPDATE_TERMINAL:-}" ]; then
    WAIT=--wait                    # the terminal opened for this alone stays until its person has read it
fi

# A drive formatted for both systems (exFAT, NTFS) keeps no execute bit of its own: where the mount gives none and it
# cannot be set, or the drive is mounted noexec, Linux will not start a program from it.
if [ ! -x "$PY" ]; then
    chmod +x "$HERE/linux/python/bin/"* 2> /dev/null
fi
if ! "$PY" -I -c "" 2> /dev/null; then
    say "Linux will not start a program from this folder's drive ($HERE): it is mounted without the right to run programs (noexec), or gives its files none. Mount the drive with exec, or copy the SiLVANN folder to a disk of this machine."
    exit 1
fi

unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
export PYTHONNOUSERSITE=1
cd "$HERE" || exit 1
exec "$PY" -I "$APP" $WAIT
