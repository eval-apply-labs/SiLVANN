#!/bin/sh
# SiLVANN.sh — the portable app on Linux (NN-194), as SiLVANN.exe is on Windows, in the folder silvann_portable_x64.zip
# unzips to: one folder — a USB drive, say — carried between Windows and Linux, each system starting its own part on the
# same data.
#
#   windows/    SiLVANN.exe, Python's embeddable build, the evaluator and the families in lib\, the release's tree
#   linux/      SiLVANN.sh (this), a standalone Python in python/, the evaluator and the families in lib/, the tree
#   data/       the models, the conversations, the logs — both systems'
#
# This starts python/bin/python3 beside it on scripts/portable/silvann_portable.py: the UI in a window of its own, and
# everything it started ended when that window closes. Nothing is installed; its Python reads nothing of the machine's
# own (PYTHONPATH, PYTHONHOME and the user's site-packages left out).
#
#   linux/SiLVANN.sh           the app (a file manager that runs scripts asks first; or from a terminal)
#   linux/SiLVANN.sh --menu    SiLVANN in the desktop's applications menu, started from this folder where it is now
PART=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
HERE=$(dirname -- "$PART")
PY="$PART/python/bin/python3"
APP="$PART/scripts/portable/silvann_portable.py"

say() {   # the terminal's line always; a message box where there is a display and zenity or kdialog
    echo "SiLVANN: $1" >&2
    if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        if command -v zenity > /dev/null 2>&1; then zenity --error --no-wrap --title SiLVANN --text "$1" 2> /dev/null
        elif command -v kdialog > /dev/null 2>&1; then kdialog --title SiLVANN --error "$1" 2> /dev/null
        fi
    fi
}

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    say "This SiLVANN runs on Windows and on Linux on 64-bit x86, and this machine is $(uname -s) on $(uname -m)."
    exit 1
fi
if [ ! -f "$PY" ] || [ ! -f "$APP" ]; then
    say "This folder is not SiLVANN's whole: python/bin/python3 or scripts/portable/silvann_portable.py is missing beside SiLVANN.sh. Unzip silvann_portable_x64.zip again, whole."
    exit 1
fi

if [ "${1:-}" = "--menu" ]; then
    APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$APPS" || exit 1
    cat > "$APPS/silvann_portable.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SiLVANN
Comment=Language models on this machine, in a window of their own
Exec="$PART/SiLVANN.sh"
Path=$PART
Icon=$PART/scripts/portable/silvann.svg
Terminal=false
Categories=Utility;Development;
EOF
    echo "SiLVANN is in the applications menu, started from $PART: run --menu again if the folder moves."
    exit 0
fi

# A drive formatted for both systems (exFAT, NTFS) keeps no execute bit of its own: where the mount gives none and it
# cannot be set, or the drive is mounted noexec, Linux will not start a program from it.
if [ ! -x "$PY" ]; then
    chmod +x "$PART/python/bin/"* 2> /dev/null
fi
if ! "$PY" -I -c "" 2> /dev/null; then
    say "Linux will not start a program from this folder's drive ($HERE): it is mounted without the right to run programs (noexec), or gives its files none. Mount the drive with exec, or copy the SiLVANN folder to your own disk."
    exit 1
fi

unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
export PYTHONNOUSERSITE=1
cd "$PART" || exit 1
exec "$PY" "$APP" "$@"
