"""USERS — a folder, a password and a config each; a user sees only their own conversations (NN-150).

⚖ 2026-11-02 *"if we do the multi seat we should also set up a multi user ui, where users have a folder with at least a
password and they get in and only see their chat folder"* · *"there should be a per user config (with the tool overrides,
each users brave key maybe, etc)"*.

A machine with no user is the machine it was: no sign-in, every conversation in `user_sessions/<model>/`. A user made
(`silvann users add <name>`), or a provider to sign on with (`sso.json`, ▶ sso.py), turns sign-in on for every server on
that data folder. A user's folder is theirs whole:
```
  users/<name>/user.json                 their name, their password (scrypt) and the accounts they sign on with
  users/<name>/tools.json                their settings over the machine's tools file (▶ `UserToolbox`)
  users/<name>/user_sessions/<model>/    their conversations, as the machine's were
  users/<name>/workspace/                their REPL's workspace
  users/<name>/tools_library.json        the definitions their REPL kept; tools_allowed.json, the folders it allowed
```
A signed-in user's conversations are the ones in their folder, and no other: another's answers as one that is not there.
The conversations made before the first user are nobody's until `silvann users adopt <name>` moves them into a folder.

⭐ HIDDEN IS NOT ENOUGH: a user's model has a REPL, and the REPL reads files. So what the UI keeps apart, the REPL does too
(▶ `Fence`): of the data folder it reaches the user's workspace and the folders it runs from, and nothing else — every
other user's folder, the user's own password, settings and conversations, the rest of the machine's data — those read
and written by the server's Python alone; and where the machine leaves the model's commands unconfined they are
confined, the machine but what the fence closes (▶ `UserToolbox.config`'s `bash`).
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import threading
import time

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
IDLE = 12 * 3600                 # a sign-in not used for this long is over

# ⭐ WHAT A USER'S tools.json MAY SET — what they bring of their own: a search engine and its key (the search skill's),
#   the address a site is told to write to, a registry of servers and its key. Of every other built-in nothing: how the
#   machine confines its commands, shows its screen, runs its pictures and lends its cards stays the machine's. Of another
#   skill, what the machine left unset (null): nothing the machine chose is undone. And a user switches a tool off,
#   never on.
USER_SETTINGS = {"search": ("searxng", "brave_key"), "fetch": ("contact",), "agent": ("registry", "key")}
MACHINE_ONLY = ("add_tool",)     # off for every user: what it adds, every conversation on the machine would run


def hash_password(password):
    """A password as the server keeps one: scrypt, salted — the conversations' own passwords are kept the same way."""
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return {"alg": "scrypt", "salt": base64.b64encode(salt).decode(), "hash": base64.b64encode(h).decode(),
            "n": 16384, "r": 8, "p": 1, "dklen": 32}


def verify_password(password, stored):
    try:
        h = hashlib.scrypt(password.encode(), salt=base64.b64decode(stored["salt"]), n=stored.get("n", 16384),
                           r=stored.get("r", 8), p=stored.get("p", 1), dklen=stored.get("dklen", 32))
        return hmac.compare_digest(h, base64.b64decode(stored["hash"]))
    except Exception:
        return False


class Refused(Exception):
    """What a users' command cannot do, said."""


def _stamp(record):
    """What a sign-in keeps of its user: their password and the accounts they sign on with — either changed, by this
    process or the `silvann users` command, and the sign-in ends."""
    return hashlib.sha256(json.dumps([record.get("password"), record.get("sso", [])], sort_keys=True).encode()).hexdigest()


class Users:
    """The users of a data folder, and who is signed in to this process."""

    def __init__(self, data):
        self.data = os.path.abspath(str(data))
        self.root = os.path.join(self.data, "users")
        self.lock = threading.Lock()
        self.tokens = {}              # a sign-in's token -> [the user, when it was last used]

    # ── the folders ───────────────────────────────────────────────────────────
    def folder(self, name):
        return os.path.join(self.root, name)

    def sessions(self, name, model):
        return os.path.join(self.folder(name), "user_sessions", model)

    def workspace(self, name):
        return os.path.join(self.folder(name), "workspace")

    def tools_file(self, name):
        return os.path.join(self.folder(name), "tools.json")

    def names(self):
        """The users, by name: each a folder holding a `user.json`."""
        try:
            return sorted(d for d in os.listdir(self.root) if NAME.match(d)
                          and os.path.isfile(os.path.join(self.root, d, "user.json")))
        except FileNotFoundError:
            return []

    def on(self):
        """Whether sign-in is on: there is a user, or a provider to sign on with (▶ sso.py) — whose users are made at their
        first sign-in, so a server with none yet still asks."""
        return bool(self.names()) or os.path.isfile(os.path.join(self.data, "sso.json"))

    def _record(self, name):
        try:
            with open(os.path.join(self.folder(name), "user.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write(self, name, record):
        path = os.path.join(self.folder(name), "user.json")
        os.makedirs(self.folder(name), exist_ok=True)
        fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # only its owner reads a password
        with os.fdopen(fd, "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        os.replace(path + ".tmp", path)

    # ── the commands (`silvann users …`) ──────────────────────────────────────────────
    def add(self, name, password):
        if not NAME.match(name or ""):
            raise Refused("a user's name is 1 to 32 of a-z, 0-9, _ and -, beginning with a letter or a digit: not %r" % name)
        if not password:
            raise Refused("a user needs a password")
        if self._record(name) is not None:
            raise Refused("%s is a user already: `silvann users passwd %s` changes the password" % (name, name))
        self._write(name, {"name": name, "password": hash_password(password), "created": time.time()})
        for sub in ("user_sessions", "workspace"):
            os.makedirs(os.path.join(self.folder(name), sub), exist_ok=True)

    def passwd(self, name, password):
        record = self._record(name)
        if record is None:
            raise Refused("%s is no user here" % name)
        if not password:
            raise Refused("a user needs a password")
        record["password"] = hash_password(password)
        self._write(name, record)
        with self.lock:                                # a new password signs every session of theirs out
            self.tokens = {t: v for t, v in self.tokens.items() if v[0] != name}

    # ── signing on (▶ sso.py): an account of a provider's — its issuer and subject — is one user's ──────────────
    def linked(self, issuer, subject):
        """The user an account signs on as, or None."""
        for name in self.names():
            for a in (self._record(name) or {}).get("sso", []):
                if a.get("iss") == issuer and a.get("sub") == subject:
                    return name
        return None

    def link(self, name, issuer, subject, email=None):
        """`name` signs on with the account `issuer`/`subject` from now on."""
        with self.lock:
            record = self._record(name)
            if record is None:
                raise Refused("%s is no user here" % name)
            other = self.linked(issuer, subject)
            if other == name:
                return
            if other is not None:
                raise Refused("that account signs on as %s: `silvann users unlink %s %s %s` first" % (other, other, issuer,
                                                                                                    subject))
            record.setdefault("sso", []).append({k: v for k, v in (("iss", issuer), ("sub", subject), ("email", email))
                                                 if v})
            self._write(name, record)

    def unlink(self, name, issuer, subject):
        with self.lock:
            record = self._record(name)
            if record is None:
                raise Refused("%s is no user here" % name)
            kept = [a for a in record.get("sso", []) if not (a.get("iss") == issuer and a.get("sub") == subject)]
            if len(kept) == len(record.get("sso", [])):
                raise Refused("%s does not sign on with that account" % name)
            record["sso"] = kept
            self._write(name, record)
            self.tokens = {t: v for t, v in self.tokens.items() if v[0] != name}

    def add_signed_on(self, wanted, issuer, subject, email=None):
        """A user made at an account's first sign-on: named `wanted` (as a name may be: ▶ NAME), or `wanted-2`, `-3` … where
        that is someone's — never someone's folder. No password: a password's sign-in refuses them until one is set.
        Answers the name."""
        base = re.sub(r"[^a-z0-9_-]", "-", (wanted or "").lower()).lstrip("-_")[:32] or "user"
        with self.lock:
            other = self.linked(issuer, subject)
            if other is not None:
                return other
            name, n = base, 1
            while self._record(name) is not None or os.path.exists(self.folder(name)):
                n += 1
                name = "%s-%d" % (base[:32 - len(str(n)) - 1], n)
            self._write(name, {"name": name, "sso": [{k: v for k, v in (("iss", issuer), ("sub", subject),
                                                                         ("email", email)) if v}],
                               "created": time.time()})
            for sub in ("user_sessions", "workspace"):
                os.makedirs(os.path.join(self.folder(name), sub), exist_ok=True)
            return name

    def accounts(self, name):
        """The accounts `name` signs on with: (issuer, subject, email) each."""
        return [(a.get("iss"), a.get("sub"), a.get("email")) for a in (self._record(name) or {}).get("sso", [])]

    def conversations(self, name):
        """How many conversations a user has, every model's."""
        top = os.path.join(self.folder(name), "user_sessions")
        try:
            return sum(1 for m in os.listdir(top) if os.path.isdir(os.path.join(top, m))
                       for c in os.listdir(os.path.join(top, m)) if os.path.isdir(os.path.join(top, m, c)))
        except FileNotFoundError:
            return 0

    def remove(self, name, with_conversations=False):
        if self._record(name) is None:
            raise Refused("%s is no user here" % name)
        held = self.conversations(name)
        if held and not with_conversations:
            raise Refused("%s has %d conversation%s: `silvann users remove %s --with-conversations` removes them too"
                          % (name, held, "" if held == 1 else "s", name))
        shutil.rmtree(self.folder(name))
        with self.lock:
            self.tokens = {t: v for t, v in self.tokens.items() if v[0] != name}

    def adopt(self, name, machine_sessions):
        """The conversations made before there were users — `machine_sessions`, the data folder's `user_sessions/` —
        moved into `name`'s folder, every model's. Answers what moved, (model, id) each."""
        if self._record(name) is None:
            raise Refused("%s is no user here" % name)
        moved = []
        try:
            models = sorted(os.listdir(machine_sessions))
        except FileNotFoundError:
            return moved
        for model in models:
            src = os.path.join(machine_sessions, model)
            if not os.path.isdir(src):
                continue
            dst = self.sessions(name, model)
            os.makedirs(dst, exist_ok=True)
            for cid in sorted(os.listdir(src)):
                if os.path.isdir(os.path.join(src, cid)) and not os.path.exists(os.path.join(dst, cid)):
                    shutil.move(os.path.join(src, cid), os.path.join(dst, cid))
                    moved.append((model, cid))
        return moved

    # ── signing in ───────────────────────────────────────────────────────────
    def sign_in(self, name, password):
        """A token for `name`, or None — the same None for a name that is no user's as for a wrong password."""
        record = self._record(name) if NAME.match(name or "") else None
        if record is None or not verify_password(password or "", record.get("password") or {}):
            return None
        token = secrets.token_urlsafe(32)
        with self.lock:
            self.tokens[token] = [name, time.time(), _stamp(record)]
        return token

    def issue(self, name):
        """A token for `name` given by the server itself — a subagent's conversation on this server, made in its caller's
        folder (▶ the server's `_tool_agent`): no password is asked of the server's own Python."""
        record = self._record(name)
        if record is None:
            return None
        token = secrets.token_urlsafe(32)
        with self.lock:
            self.tokens[token] = [name, time.time(), _stamp(record)]
        return token

    def signed(self, token):
        """Who `token` signs in, or None: a token never given, given before this process started, unused for `IDLE`,
        or its user's since removed, given a new password, or given or taken an account to sign on with."""
        if not token:
            return None
        with self.lock:
            got = self.tokens.get(token)
            if got is None:
                return None
            record = self._record(got[0])
            if time.time() - got[1] > IDLE or record is None or _stamp(record) != got[2]:
                self.tokens.pop(token, None)
                return None
            got[1] = time.time()
            return got[0]

    def sign_out(self, token):
        with self.lock:
            self.tokens.pop(token, None)


def _closed(root, keep):
    """Every path under `root` that is neither kept — in `keep`, or under a path of it — nor on the way to one: what to
    seal so that, of `root`, only `keep` is reached. A folder on the way to a kept one is opened, and its other entries
    sealed in their turn."""
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        p = os.path.join(root, name)
        real = os.path.realpath(p)
        if any(real == k or real.startswith(k.rstrip(os.sep) + os.sep) for k in keep):
            continue                                       # kept, and everything under it
        if any(k.startswith(real.rstrip(os.sep) + os.sep) for k in keep):
            out += _closed(p, keep)                        # on the way to a kept one
            continue
        out.append(real)
    return out


class Fence:
    """⭐ WHAT A USER'S REPL REACHES — A CHROOT'S VIEW WITHOUT ROOT (⚖ 2026-11-06 *"maybe the model's repl is simply blind
    to the whole folder with the exception of the current user?"* · *"chroooted in the user's folder space … so it can
    access only the tools that the system allows them to access"*): it writes in the user's workspace alone; it reads
    there, in the system's programs (▶ jail's SYSTEM, READABLE), the folders of the tools the user has switched on, and
    the runtime and the venv it runs from; the rest is closed — of the data folder everything else (every other user's
    folder, the user's own password, settings and conversations, the Hugging Face token, a file made later), and of the
    machine everything else (its other disks, /home, /tmp). The commands' Landlock rules are the same view (▶
    UserToolbox, tools.confinement). Looked at afresh at each command: a tool switched off since is closed at once.
    It stands where a classifier keeps its `sealed` list (▶ tools.Classifier's `closed_to`)."""

    def __init__(self, users, me, machine_sealed=(), keep=(), tools=None):
        self.users, self.me = users, me
        self.machine = [os.path.realpath(p) for p in machine_sealed]
        self.keep = [os.path.realpath(k) for k in keep]       # the runtime and the venv: read and run
        self.tools = tools or (lambda: [])                     # the folders of the user's tools switched on, asked now

    def home(self):
        return os.path.realpath(self.users.workspace(self.me))

    def readable(self):
        from . import jail
        return self.keep + [os.path.realpath(t) for t in self.tools()] + list(jail.SYSTEM) + list(jail.READABLE)

    def paths(self):
        """What is closed of the data folder, as paths that are there now — a command line naming one is refused."""
        return self.machine + _closed(self.users.data, [self.home()] + self.readable())

    def covers(self, target, touches="read"):
        """⛔ Whether `target` is closed to a read or a write — a path not made yet as well as one that is."""
        real = os.path.realpath(str(target))
        under = lambda p, a: p == a or p.startswith(a.rstrip(os.sep) + os.sep)       # noqa: E731
        if any(under(real, a) for a in self.machine):
            return True
        if under(real, self.home()):
            return False
        if touches == "write":
            return True                                    # a user's REPL writes in their workspace alone
        readable = self.readable()
        if under(real, os.path.realpath(self.users.data)):  # the data folder's own: what is kept of it, and no more —
            return not any(under(real, r) for r in readable if under(r, os.path.realpath(self.users.data)))
        return not any(under(real, r) for r in readable)   #   though it lie under a system folder, /opt say

    def __iter__(self):
        return iter(self.paths())

    def __len__(self):
        return len(self.paths())

    def __add__(self, other):
        return self.paths() + list(other)


class UserToolbox:
    """⭐ THE MACHINE'S TOOLS AS ONE USER HAS THEM — the machine's `Toolbox` (▶ tools_config), its skills and its startup
    check, with the user's `tools.json` over it: a tool the user switched off is off for them, and a setting the user may
    give (▶ `USER_SETTINGS`) is theirs. `add_tool` is off; the commands are confined to the workspace (▶ `Fence`).
    Everything else is the machine's, asked of it."""

    follows_links = False     # a link at the workspace's top is not followed: one the model made would be its way out

    def __init__(self, machine, path, data=None):
        self.machine, self.path = machine, str(path)
        self.closed = [os.path.abspath(str(data))] if data else []   # the data folder, granted of no system folder
        self._read_at, self._entries = None, {}

    def __getattr__(self, name):                       # the skills, the file, refresh, added, writable_folders, …
        return getattr(self.machine, name)

    def entries(self):
        """The user's tools.json, its entries (no `_` keys) — read again when the file changes."""
        try:
            at = os.stat(self.path).st_mtime_ns
        except OSError:
            self._read_at, self._entries = None, {}
            return {}
        if at != self._read_at:
            try:
                with open(self.path, encoding="utf-8") as f:
                    got = json.load(f)
                self._entries = {k: v for k, v in got.items() if not str(k).startswith("_") and isinstance(v, dict)} \
                    if isinstance(got, dict) else {}
            except (OSError, ValueError):
                self._entries = {}
            self._read_at = at
        return self._entries

    @property
    def generation(self):
        """The machine's, and the user's file: a change in either binds a session's tools again."""
        return (self.machine.generation, self._read_at if self.entries() is not None else None)

    def _may_set(self, name, key):
        from .tools_config import BUILT_IN
        if key in USER_SETTINGS.get(name, ()):               # what a user brings of their own, built-in or skill
            return True
        if name in BUILT_IN:
            return False
        return self.machine.config(name).get(key) is None    # another skill's: what the machine left unset

    def switched_off(self, name):
        """Whether the user switched the tool off."""
        return str(self.entries().get(name, {}).get("enabled", "")).lower() == "no"

    def enabled(self, name):
        if name in MACHINE_ONLY or self.switched_off(name):
            return "no"
        return self.machine.enabled(name)

    def config(self, name, defaults=None):
        got = self.machine.config(name, defaults)
        mine = self.entries().get(name, {}).get("config")
        if isinstance(mine, dict):
            got.update({k: v for k, v in mine.items() if not str(k).startswith("_") and self._may_set(name, k)})
        # ⛔ A USER'S COMMANDS ARE CONFINED TO THEIR WORKSPACE, whatever the machine's own are: they read and run the
        #   system's programs and the user's tools, and nothing else of the machine (▶ `Fence`)
        if name == "bash":
            got["confined"] = "workspace"
        return got

    def tool_folders(self):
        """The folders of the tools the user has switched on: what their commands may read and run, and never write."""
        return [os.path.realpath(sk.folder) for sk in self.machine.skills if self.enabled(sk.name) != "no"]

    def writable_folders(self):
        """For a user, the folders of their tools switched on — read and run by their commands, never written (the
        confinement grants them so; ▶ tools.confinement)."""
        return self.tool_folders()

    def why_not(self, tool, command):
        if tool in MACHINE_ONLY:
            return "switched off for a user: what it adds, every conversation on this machine would run"
        if self.switched_off(tool):
            return "switched off in your settings (your tools.json)"
        return self.machine.why_not(tool, command)

    def listing(self):
        out = self.machine.listing()
        for name, e in out.items():
            e["enabled"] = self.enabled(name)
            e["config"] = self.config(name)
        return out

    def kept(self):
        """What the user's file asks that the machine keeps — a setting not theirs to give, a tool switched on — a line
        each, for the UI to say."""
        said = []
        for name, e in sorted(self.entries().items()):
            switch = str(e.get("enabled", "")).lower()
            if switch in ("yes", "auto") and name in MACHINE_ONLY:
                said.append("%s stays off: what it adds, every conversation on this machine would run" % name)
            elif switch in ("yes", "auto") and self.machine.enabled(name) == "no":
                said.append("%s stays off: the machine switched it off" % name)
            for k in sorted(e.get("config") or {}):
                if not str(k).startswith("_") and not self._may_set(name, k):
                    said.append("%s's %s stays the machine's" % (name, k))
        return said
