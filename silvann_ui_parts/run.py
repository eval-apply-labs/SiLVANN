"""The UI as a process: where it listens, the model it starts again, its restart, and `main`."""
import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from typing import Optional

from nicegui import app, ui

import silvann_hosts

from .downloads import DOWNLOADS
from .head import SILVANN_TILE
from .local import DATA_DIR, LAST_START, LISTEN, LOCAL_APP, MODELS_DIR, START_AGAIN, model_configs_dir, model_present
from .server_process import SERVER
from .sign_in import LOCAL_USERS


def _this_address() -> str:
    """This machine's address on its network — the one a connection outward would leave from (no packet sent) — or its
    name."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))                   # a documentation address: only the route is looked up
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def start_again_later(args: Optional[tuple]):
    """The model a switch stopped — (its folder, its config, its port) — started by the UI that starts next (▶ main)."""
    if args:
        START_AGAIN.write_text(json.dumps({"model_dir": args[0], "config": args[1], "port": args[2]}))


def restart_ui():
    """⭐ THE UI STARTED AGAIN AS IT WAS STARTED (NN-196): the switch between single and multi user changes where it
    listens (▶ listen_host), which a running UI cannot. The portable app's launcher starts it again itself — the UI exits
    the code it waits for (SILVANN_RESTART_EXIT); elsewhere this process becomes the new one (Windows has no exec that
    keeps the process: a new one, this one gone)."""
    DOWNLOADS.cancel()
    code = os.environ.get("SILVANN_RESTART_EXIT")
    if code:
        os._exit(int(code))
    argv = [sys.executable] + LISTEN["argv"]
    if os.name == "nt":
        subprocess.Popen(argv, cwd=os.getcwd())
        os._exit(0)
    os.execv(sys.executable, argv)


# ---------- CLI ----------


def _start_last():
    """The model a --local UI started last, started again as it was — the app opens where it was left. A model a switch
    between single and multi user stopped (▶ START_AGAIN) comes first, whatever UI this is."""
    try:
        again = json.loads(START_AGAIN.read_text())
        START_AGAIN.unlink()
        SERVER.start(again["model_dir"], again["config"], int(again["port"]))
        return
    except (OSError, ValueError, KeyError):
        pass
    if not LOCAL_APP["on"]:
        return
    try:
        last = json.loads(LAST_START.read_text())
    except (OSError, ValueError):
        return
    model, config = last.get("model") or "", last.get("config") or ""
    if model_present(model) and config and (model_configs_dir(model) / config).is_file():
        SERVER.start(str(MODELS_DIR / model), config, int(last.get("port", 8765)))


def _stop_all():
    """A --local UI stopping (a container's `docker stop`): the download stopped, the conversations saved, the server
    stopped — ServerProcess.stop asks the server to save them first."""
    DOWNLOADS.cancel()
    SERVER.stop()


def _quit_route(token):
    """The portable app's stop (▶ release/scripts/portable/silvann_portable.py): a POST carrying the token the app gave
    this UI stops it as `docker stop` does — _stop_all, then the process ends. Windows has no signal to ask a process
    with: a terminate there is TerminateProcess, which runs none of it. Without the token the route does not exist."""
    from fastapi import Header, HTTPException

    @app.post("/_silvann/quit")
    def quit_app(x_silvann_quit: str = Header(default="")):
        if not secrets.compare_digest(x_silvann_quit.encode(), token.encode()):
            raise HTTPException(403, "not this app's")
        app.shutdown()                                     # the shutdown handlers run _stop_all
        return {"stopping": True}


# ⭐ WHERE THE UI LISTENS: on every address only where the machine is multi user — ⚖ 2026-11-12 *"since we added users i
#   think we can now unblock the setup, maybe add a simpler flag for those who want it locked down"*. A single user's
#   machine is one person's at it, and stays this machine's: opening it to the network is its admin's choice, never a
#   default. `--private` keeps a multi user's here too, and `--host` names an address outright (the container's image,
#   the portable app, a single user who reaches it from another machine of theirs).
def listen_host(host: Optional[str], private: bool) -> tuple:
    """(the address the UI listens on, why): `host` where it is named, else 127.0.0.1 where `private` or the machine is
    single user, else 0.0.0.0."""
    if host:
        return host, "the address --host names"
    if private:
        return "127.0.0.1", "this machine alone (--private)"
    if LOCAL_USERS is None or not LOCAL_USERS.ready() or LOCAL_USERS.mode() != "multi":
        return "127.0.0.1", ("this machine alone: single user (Options, or `silvann users multi`, makes it multi user "
                             "and opens it to the network; --host names an address outright)")
    return "0.0.0.0", "every address: multi user"


def storage_secret() -> str:
    """⛔ WHAT A BROWSER'S SESSION IS SIGNED WITH IS THIS DATA FOLDER'S OWN, made once and kept beside the data, its owner's
    alone to read (⚖ 2026-11-19, the code review): a secret written in the source signs nothing — anyone can make a
    session under it. And the folder NiceGUI keeps each browser's state in, where a sign-in's token and a conversation's
    password are, is its owner's alone. A secret that cannot be kept is this process's alone: its browsers begin again
    when it starts again."""
    kept = DATA_DIR / ".ui_secret"
    try:
        secret = kept.read_text(encoding="utf-8").strip()
    except OSError:
        secret = ""
    if len(secret) < 32:
        secret = secrets.token_urlsafe(32)
        try:
            kept.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(kept, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(secret + "\n")
            os.chmod(kept, 0o600)
        except OSError as ex:
            print("[ui] the sessions' secret not kept in %s (%s): this process's alone" % (kept, ex), flush=True)
    try:
        app.storage.path.mkdir(parents=True, exist_ok=True)
        os.chmod(app.storage.path, 0o700)
    except OSError as ex:
        print("[ui] the browsers' folder %s stays as it is (%s)" % (app.storage.path, ex), flush=True)
    return secret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080, help="port for the NiceGUI app (default 8080)")
    p.add_argument("--host", default=None,
                   help="the address the UI listens on (default: every address where sign-in is on, this machine's "
                        "alone where it is not)")
    p.add_argument("--private", action="store_true",
                   help="listen on this machine alone (127.0.0.1), whether or not sign-in is on")
    p.add_argument("--endpoint", default=None,
                   help="default SiLVANN server endpoint (override on first launch)")
    p.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    # ⛳ off by default: a reload restarts this process, and a server it started goes with it — a boot of minutes lost to
    #   a file saved beside the UI
    p.add_argument("--reload", action="store_true", help="restart the UI when its files change (development)")
    p.add_argument("--local", action="store_true",
                   help="the UI as the app over this machine's models: the local panel first, "
                        "the model started last started again, and the server stopped with the UI")
    args = p.parse_args()
    restarted = os.environ.pop("SILVANN_RESTARTED", None) is not None     # by a switch (▶ restart_ui)

    if args.endpoint:
        os.environ["SILVANN_DEFAULT_ENDPOINT"] = args.endpoint
    if LOCAL_USERS is not None:
        for line in LOCAL_USERS.settle():                 # a machine from before its mode was written keeps it
            print("[ui] %s" % line, flush=True)
        # ⭐ THE APP OPENED, where no UI nor server of it was running: on a single user's machine whoever had it has it no
        #   more, and the next account to sign in does (▶ users.Users.open)
        LOCAL_USERS.open()
        if not LOCAL_USERS.ready():
            print("[ui] this machine has no account yet: its first page makes one (or `silvann users add <name>`)",
                  flush=True)
    app.on_startup(_start_last)                           # a --local app's last model, or the one a switch stopped
    if args.local:
        LOCAL_APP["on"] = True
        app.on_shutdown(_stop_all)
        if os.environ.get("SILVANN_QUIT_TOKEN"):
            _quit_route(os.environ["SILVANN_QUIT_TOKEN"])
        from .updates import settle                       # a portable app: its `_latest` link, and the version before
        settle()                                          #   this one removed where its admin said so (▶ updates.py)

    host, why = listen_host(args.host, args.private)
    LISTEN.update(host=host, private=args.private, named=args.host, port=args.port, argv=list(sys.argv))
    os.environ["SILVANN_RESTARTED"] = "1"                 # for the process a switch starts after this one, not this one
    if restarted:                                         # the UI before this one may hold the port a moment longer
        import socket
        for _ in range(60):
            with socket.socket() as s:
                if os.name != "nt":
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((host, args.port))
                    break
                except OSError:
                    time.sleep(0.25)
    # ⛔ WHERE IT ANSWERS WITH NO SIGN-IN OF ITS OWN THE UI ANSWERS UNDER THIS MACHINE'S ADDRESS AND NAMES ALONE (▶
    #   silvann_hosts.py) — a machine whose first account is still to be made, a UI that talks to a server elsewhere: a
    #   page of another site whose name answers this machine's address is refused, its scripts and its socket with it,
    #   and so cannot make the machine's first account its own
    app.add_middleware(silvann_hosts.OwnHosts, names=silvann_hosts.own_names(args.host),
                       applies=lambda: LOCAL_USERS is None or not LOCAL_USERS.ready())
    print("[ui] listening on %s:%d — %s" % (host, args.port, why), flush=True)
    ui.run(port=args.port, title="SiLVANN", favicon=SILVANN_TILE, show=not args.no_browser and not restarted,
           storage_secret=storage_secret(), reload=args.reload and not args.local, host=host)
