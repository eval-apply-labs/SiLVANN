"""AN ACCOUNT'S SPACE — what its folder holds, what of it is whose, and the most it may hold (NN-200).

⚖ 2026-11-20: *"the user account can show how much available space you have, what is occupying it"* · *"maybe the admin
can assign a per user allowed space, and the system can refuse to save things past that"*.

    usage(users, "ann")            {"limit", "used", "left", "workspace", "shared", "conversations": [...], "other"}
    room(users, "ann", more)       nothing where `more` bytes still fit; Full, said in words, where they do not

An account's space is its folder under the data folder, `users/<name>/`: its conversations — each one's transcript,
its state when it was put away, its own files — its workspace, where its model works and `attached/` is what every
conversation of the account reads, and its settings. A limit is an admin's to set (`users.Users.set_space`); with none
the account has what the disk has.

⛔ WHAT THE LIMIT HOLDS: what the server saves for the account by the user's act or the model's — a file sent to a
conversation, one uploaded to the shared folder, `(write …)`, a picture, a new conversation, a message. What a command
the model runs writes passes no door of the server's: it is counted the next time the folder is measured, and the saves
after it are refused until space is freed. What is already the account's is always put away — a conversation's state
is saved though the account is full — so nothing is lost to the limit.
"""
import json
import os
import re
import threading
import time

try:
    from .users import Refused
except ImportError:                   # loaded alone, as a check loads it: its refusals are then its own
    class Refused(Exception):
        """What is not done, said in words."""

ATTACHED = "attached"             # a conversation's own files, and in a workspace the account's shared ones
PICTURES = "pictures"             # what a conversation's model drew
FRESH = 15.0                      # seconds a folder's measured size is taken as it was: a walk is not made per save
UNITS = {"b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12, "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}

_lock = threading.Lock()
_measured = {}                    # an account's folder -> (when it was measured, its bytes then, bytes saved since)


class Full(Refused):
    """A save past an account's space."""


def parse(text):
    """Bytes from "5 GB", "500MB", "1.5gb", "1073741824" — None from "none", "no" or "0": no limit."""
    t = str(text).strip().lower().replace(" ", "")
    if t in ("none", "no", "off", "0", ""):
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]i?b|b)?", t)
    if not m:
        raise Refused("a space is a size — 5GB, 500MB — or none, not %r" % text)
    n = int(float(m.group(1)) * UNITS[m.group(2) or "b"])
    if n <= 0:
        raise Refused("a space is a size above zero, or none")
    return n


def said(n):
    """A count of bytes as a person reads it: 0 B, 12 kB, 3.4 MB, 1.25 GB."""
    n = int(n)
    for unit, name, digits in ((10**9, "GB", 2), (10**6, "MB", 1), (10**3, "kB", 0)):
        if n >= unit:
            return "%.*f %s" % (digits, n / unit, name)
    return "%d B" % n


def size(path):
    """The bytes of a file, or of everything under a folder — links counted as links, never followed."""
    try:
        st = os.lstat(path)
    except OSError:
        return 0
    if not os.path.isdir(path) or os.path.islink(path):
        return st.st_size
    total = 0
    for at, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(at, name)).st_size
            except OSError:
                pass
    return total


def conversations(users, name):
    """An account's conversations, each with what it holds: {"id", "model", "name", "archived", "transcript" — its
    history and settings —, "state" — what it was put away with, the larger part —, "files" — its attachments and the
    pictures made in it —, "bytes"}, the largest first."""
    top, out = os.path.join(users.folder(name), "user_sessions"), []
    try:
        models = sorted(os.listdir(top))
    except OSError:
        return out
    for model in models:
        if not os.path.isdir(os.path.join(top, model)):
            continue
        for cid in sorted(os.listdir(os.path.join(top, model))):
            d = os.path.join(top, model, cid)
            if not os.path.isdir(d):
                continue
            side = {}
            try:
                with open(os.path.join(d, "server_meta.json")) as f:
                    side = json.load(f)
            except (OSError, ValueError):
                pass
            state = size(os.path.join(d, "cartridge.npz"))
            files = size(os.path.join(d, ATTACHED)) + size(os.path.join(d, PICTURES))
            whole = size(d)
            out.append({"id": cid, "model": model, "name": side.get("name", cid), "archived": bool(side.get("archived")),
                        "transcript": max(0, whole - state - files), "state": state, "files": files, "bytes": whole})
    return sorted(out, key=lambda c: -c["bytes"])


def usage(users, name, workspace=None):
    """What an account's folder holds and the most it may: {"limit" — bytes or None —, "used", "left" — None with no
    limit —, "conversations" (▶ `conversations`), "workspace" — its model's working folder —, "shared" — the workspace's
    `attached/`, within it —, "other" — its settings and the rest}. `workspace`: the folder its model works in where
    that is not its own (an admin's `tools_folder`): then it is not of the account's space, and is said as 0."""
    folder = os.path.realpath(users.folder(name))
    own = os.path.realpath(users.workspace(name))
    work = os.path.realpath(workspace) if workspace else own
    inside = work == own or work.startswith(folder.rstrip(os.sep) + os.sep)
    used = size(folder)
    talks = conversations(users, name)
    w = size(work) if inside else 0
    limit = users.space(name)
    with _lock:
        _measured[folder] = (time.time(), used, 0)
    return {"limit": limit, "used": used, "left": None if limit is None else max(0, limit - used),
            "conversations": talks, "workspace": w, "shared": size(os.path.join(work, ATTACHED)),
            "shared_counted": inside, "other": max(0, used - w - sum(c["bytes"] for c in talks))}


def used(users, name):
    """An account's folder's bytes — measured at most every `FRESH` seconds, what was saved through `room` since added."""
    folder = os.path.realpath(users.folder(name))
    with _lock:
        got = _measured.get(folder)
    if got is None or time.time() - got[0] > FRESH:
        got = (time.time(), size(folder), 0)
        with _lock:
            _measured[folder] = got
    return got[1] + got[2]


def room(users, name, more=0, what="this"):
    """Nothing where the account has room for `more` bytes; `Full` where its space does not hold them. The bytes are
    counted as saved from here on, until the folder is measured again."""
    limit = users.space(name)
    if limit is None:
        return
    have = used(users, name)
    if have + more > limit:
        raise Full("%s's space is full: %s of %s used%s — %s is not saved. Delete files or conversations in Account, or "
                   "archive a conversation; an admin sets an account's space in Users"
                   % (name, said(have), said(limit), ", and it needs %s more" % said(more) if more else "", what))
    folder = os.path.realpath(users.folder(name))
    with _lock:
        got = _measured.get(folder)
        if got is not None:
            _measured[folder] = (got[0], got[1], got[2] + more)


def forget(users, name):
    """What was measured of an account's folder is measured again at the next ask — after a delete, an archive."""
    with _lock:
        _measured.pop(os.path.realpath(users.folder(name)), None)


def within(root, path):
    """`path`, a name or a relative path a client gave, as a path under `root` — Refused where it leaves it, by `..`, by
    its own beginning or by a link on the way."""
    root = os.path.realpath(root)
    rel = str(path or "").replace("\\", "/").strip("/")
    if any(part in ("..", "") for part in rel.split("/")) and rel:
        raise Refused("%r is not a path inside the folder" % path)
    full = os.path.realpath(os.path.join(root, rel)) if rel else root
    if full != root and not full.startswith(root.rstrip(os.sep) + os.sep):
        raise Refused("%r leaves the folder" % path)
    return full


def listing(root, path=""):
    """The names in `path` under `root`: [{"name", "path" — from the root —, "folder", "bytes", "modified"}], folders
    first; a folder's bytes are everything under it. A folder that is not there lists as empty."""
    here = within(root, path)
    out = []
    try:
        names = sorted(os.listdir(here))
    except OSError:
        return out
    base = os.path.realpath(root)
    for name in names:
        p = os.path.join(here, name)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        folder = os.path.isdir(p) and not os.path.islink(p)
        out.append({"name": name, "path": os.path.relpath(p, base).replace(os.sep, "/"), "folder": folder,
                    "bytes": size(p), "modified": int(st.st_mtime)})
    return sorted(out, key=lambda e: (not e["folder"], e["name"].lower()))


def file_name(name):
    """A name a file is kept under: its own, without a folder and with nothing a file system refuses."""
    base = os.path.basename(str(name or "").replace("\\", "/")).strip()
    return (re.sub(r"[^\w .()+,=@\[\]-]", "_", base).strip(" .") or "file")[:160]


def keep(folder, name, data):
    """`data` as a file named `name` in `folder`, and its path: the same bytes sent again are the file already there;
    other bytes under the same name are numbered (`report-2.pdf`)."""
    os.makedirs(folder, exist_ok=True)
    stem, suffix = os.path.splitext(file_name(name))
    n = 1
    while True:
        path = os.path.join(folder, "%s%s%s" % (stem, "" if n == 1 else "-%d" % n, suffix))
        if not os.path.lexists(path):
            part = path + ".part"
            with open(part, "wb") as f:
                f.write(data)
            os.replace(part, path)
            return path
        if os.path.isfile(path) and not os.path.islink(path) and os.path.getsize(path) == len(data):
            with open(path, "rb") as f:
                if f.read() == data:
                    return path
        n += 1


def archive(path):
    """A conversation's `cartridge.npz` without its state — its tokens and its pictures' rows kept, so it can be read
    again; the bytes given up. A file with no state is left as it is."""
    import numpy as np
    if not os.path.isfile(path):                   # a conversation that was never put away has no state to give up
        return 0
    before = os.path.getsize(path)
    with np.load(path) as z:
        header = json.loads(bytes(z["header"]).decode())
        if not header.get("has_state"):
            return 0
        arrays = {k: z[k] for k in z.files if not re.fullmatch(r"s\d+", k) and k != "header"}
    header["has_state"] = False
    arrays["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    part = str(path) + ".part"                 # whole or not at all, as a save is
    with open(part, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(part, path)
    return before - os.path.getsize(path)
