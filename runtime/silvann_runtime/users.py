"""USERS — a folder, a password and a config each; a user sees only their own conversations (NN-150).

⚖ 2026-11-02 *"if we do the multi seat we should also set up a multi user ui, where users have a folder with at least a
password and they get in and only see their chat folder"* · *"there should be a per user config (with the tool overrides,
each users brave key maybe, etc)"*.

⭐ SINGLE USER OR MULTI USER (NN-196, ⚖ 2026-11-18 *"i would still maintain a user folder as if it was multi user …
simply renaming a folder is all it takes to change an user back into what it was"*). A machine with no user is single
user: no sign-in, this machine alone, and its one user's folder `users/__local/` (▶ `settle`, `home`) — a name no user
can take, as a user's begins with a letter or a digit. Its model's REPL reaches the machine as the tools file lets it,
unconfined. The switch to multi user (▶ `multi`: the UI's Options, or `silvann users add` on a machine with no user)
renames that folder the admin's; the switch back (▶ `single`, the admin the last user) renames it `__local` again. A
user, or a provider to sign on with (`sso.json`, ▶ sso.py), turns sign-in on for every server on that data folder:
everyone signs in, on the machine itself too, and only an admin (`"admin": true` in their user.json) touches the
machine's settings. A user's folder is theirs whole:
```
  users/<name>/user.json                 their name, their password (scrypt), whether they are an admin, and the
                                           accounts they sign on with
  users/<name>/tools.json                their settings over the machine's tools file (▶ `UserToolbox`)
  users/<name>/skills_added/             the tools their model added (▶ tools' `add_tool`), theirs alone
  users/<name>/user_sessions/<model>/    their conversations, as the machine's were
  users/<name>/workspace/                their REPL's workspace
  users/<name>/tools_library.json        the definitions their REPL kept; tools_allowed.json, the folders it allowed
```
A signed-in user's conversations are the ones in their folder, and no other: another's answers as one that is not there.
On a machine whose users came before `__local` (NN-150), the conversations made before the first user are at the data
folder's top, nobody's until `silvann users adopt <name>` moves them into a folder.

A sign-in is every process's on the data folder (▶ `sign_in`, `signed`) — the server's, and the UI's, which checks one
itself: so the UI asks for it with no model running, and a model stopped and started signs no one out. The tokens'
hashes are kept in `users/.sign_ins.json`, which no user's REPL reaches (▶ `Fence`).

⭐ HIDDEN IS NOT ENOUGH: a user's model has a REPL, and the REPL reads files. So what the UI keeps apart, the REPL does too
(▶ `Fence`): of the data folder it reaches the user's workspace and the folders it runs from, and nothing else — every
other user's folder, the user's own password, settings and conversations, the rest of the machine's data — those read
and written by the server's Python alone; and where the machine leaves the model's commands unconfined they are
confined, the machine but what the fence closes (▶ `UserToolbox.config`'s `bash`).
"""
import base64
import contextlib
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
LOCAL = "__local"                # the single user's folder (▶ the module's note)
IDLE = 12 * 3600                 # a sign-in not used for this long is over
TOUCH = 300                      # a sign-in's last use written down at most this often
SIGN_INS = ".sign_ins.json"      # in users/: the sign-ins, every process's
SERVING = ".serving"             # in users/: held by each server running on the data folder (▶ `serve`)
SLOTS = 4096                     # on Windows a server holds one byte of these (▶ `serve`)
WINDOWS = os.name == "nt"

# ⭐ WHAT A USER'S tools.json MAY SET — what they bring of their own: a search engine and its key (the search skill's),
#   the address a site is told to write to, a registry of servers and its key. Of every other built-in nothing: how the
#   machine confines its commands, shows its screen, runs its pictures and lends its cards stays the machine's. Of another
#   skill, what the machine left unset (null): nothing the machine chose is undone. And a user switches a tool off,
#   never on.
USER_SETTINGS = {"search": ("searxng", "brave_key"), "fetch": ("contact",), "agent": ("registry", "key")}


@contextlib.contextmanager
def _locked(path):
    """`path` locked against every other process that locks it so — the sign-ins' writers, the layout's changes."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if WINDOWS:
            import msvcrt
            for _ in range(30):                        # LK_LOCK itself tries for ten seconds before it raises
                try:
                    os.lseek(fd, 0, 0)
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if WINDOWS:
            try:
                import msvcrt
                os.lseek(fd, 0, 0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        os.close(fd)                                   # flock's lock goes with its descriptor


def _write_private(path, value):
    """`value` as JSON at `path`, which only its owner reads, put in place whole."""
    fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
    os.replace(path + ".tmp", path)


def _hashed(token):
    return hashlib.sha256(token.encode()).hexdigest()


_HELD = []                       # this process's hold on `users/.serving` (▶ `Users.serve`), kept open while it runs


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
        self.store_lock = threading.Lock()
        self._store, self._store_at = {}, None       # the sign-ins file as last read, and what it was then (▶ `_signs`)

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
        """Whether sign-in is on — multi user: there is a user, or a provider to sign on with (▶ sso.py) — whose users are
        made at their first sign-in, so a server with none yet still asks."""
        return bool(self.names()) or os.path.isfile(os.path.join(self.data, "sso.json"))

    # ── the single user (▶ the module's note) ───────────────────────────────────────────────────────────────
    def local(self):
        return os.path.join(self.root, LOCAL)

    def home(self):
        """The machine's own folder: the single user's, users/__local/ — or, multi user, the data folder's top, where a
        machine's conversations from before its users (NN-150) wait to be adopted. Its `user_sessions/`, `workspace/`,
        `tools_library.json` and `tools_allowed.json` are the machine's conversations, workspace and REPL files."""
        return self.data if self.on() else self.local()

    def machine(self):
        """The machine's own conversations, workspace and REPL files, in its folder (▶ `home`): {"sessions",
        "workspace", "library", "allowed"} — on a machine from before `__local`, the data folder's top as it named them."""
        h = self.home()
        return {"sessions": os.path.join(h, "user_sessions"), "workspace": os.path.join(h, "workspace"),
                "library": os.path.join(h, "tools_library.json"),
                "allowed": os.path.join(h, "tools.json" if h == self.data else "tools_allowed.json")}

    def settle(self):
        """⭐ users/__local/ MADE, a single user's machine — what the data folder's top held before it (the releases
        before NN-196: user_sessions/, workspace/, tools_library.json, the folders allowed in tools.json) moved into it,
        each where `__local` has none of its own yet; a conversation at the top beside `__local`'s, moved in too. Run as
        each process starts; nothing on a multi user machine. Answers what moved, a line each."""
        if self.on():
            return []
        moved = []
        os.makedirs(self.root, exist_ok=True)
        with _locked(os.path.join(self.root, ".layout.lock")):
            os.makedirs(self.local(), exist_ok=True)
            top, mine = os.path.join(self.data, "user_sessions"), os.path.join(self.local(), "user_sessions")
            if os.path.isdir(top):
                for model in sorted(os.listdir(top)):
                    src = os.path.join(top, model)
                    if not os.path.isdir(src):
                        continue
                    if not os.path.exists(os.path.join(mine, model)):
                        os.makedirs(mine, exist_ok=True)
                        os.replace(src, os.path.join(mine, model))
                        moved.append("user_sessions/%s/" % model)
                        continue
                    for cid in sorted(os.listdir(src)):
                        if os.path.isdir(os.path.join(src, cid)) and not os.path.exists(os.path.join(mine, model, cid)):
                            os.replace(os.path.join(src, cid), os.path.join(mine, model, cid))
                            moved.append("user_sessions/%s/%s/" % (model, cid))
                for model in os.listdir(top):                # what is left: emptied folders
                    with contextlib.suppress(OSError):
                        os.rmdir(os.path.join(top, model))
                with contextlib.suppress(OSError):
                    os.rmdir(top)
            for was, now in (("workspace", "workspace"), ("tools_library.json", "tools_library.json"),
                             ("tools.json", "tools_allowed.json")):
                src, dst = os.path.join(self.data, was), os.path.join(self.local(), now)
                if os.path.lexists(src) and not os.path.lexists(dst):
                    os.replace(src, dst)
                    moved.append("%s → users/%s/%s" % (was, LOCAL, now))
            for sub in ("user_sessions", "workspace"):
                os.makedirs(os.path.join(self.local(), sub), exist_ok=True)
        return moved

    # ── a server on the data folder (▶ `multi`, `single`, which wait for none to run) ─────────────────────────────
    def serve(self):
        """This process a server on the data folder until it ends: a hold on `users/.serving` the system lets go of when
        the process ends, however it ends — a shared lock on Linux, one byte of SLOTS on Windows (whose locks are not
        shared)."""
        os.makedirs(self.root, exist_ok=True)
        fd = os.open(os.path.join(self.root, SERVING), os.O_RDWR | os.O_CREAT, 0o600)
        if WINDOWS:
            import msvcrt
            for i in range(SLOTS):
                try:
                    os.lseek(fd, (os.getpid() + i) % SLOTS, 0)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    continue
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_SH)
        _HELD.append(fd)

    def serving(self):
        """Whether a server runs on the data folder now (▶ `serve`)."""
        path = os.path.join(self.root, SERVING)
        if not os.path.exists(path):
            return False
        fd = os.open(path, os.O_RDWR)
        try:
            if WINDOWS:
                import msvcrt
                try:
                    os.lseek(fd, 0, 0)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, SLOTS)
                except OSError:
                    return True
                os.lseek(fd, 0, 0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, SLOTS)
                return False
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    # ── the switch (⚖ 2026-11-18 *"to enable that it makes you create the admin account before rebooting and asking
    #    for the credentials and starting as multi user"*) ───────────────────────────────────────────────────────────
    def multi(self, name, password):
        """⭐ SINGLE USER TO MULTI USER: users/__local/ renamed `name`'s, their password and `"admin": true` written — the
        single user's conversations, workspace and definitions the admin's. Refused where `name` is someone's, or while a
        server runs on the data folder (its conversations open in the folder renamed)."""
        if not NAME.match(name or ""):
            raise Refused("a user's name is 1 to 32 of a-z, 0-9, _ and -, beginning with a letter or a digit: not %r" % name)
        if not password:
            raise Refused("the admin needs a password")
        if self.on():
            raise Refused("this machine is multi user already: an admin adds users in the UI's Users, or `silvann users "
                          "add`")
        if os.path.lexists(self.folder(name)):
            raise Refused("%s is taken: users/%s/ is there already — choose another name" % (name, name))
        if self.serving():
            raise Refused("a server is running on this data folder: stop it first — its conversations are open in the "
                          "folder renamed")
        self.settle()
        with self.lock, _locked(os.path.join(self.root, ".layout.lock")):
            try:
                os.rename(self.local(), self.folder(name))
            except OSError as ex:
                raise Refused("users/%s/ could not be renamed %s: %s" % (LOCAL, name, ex))
            self._write(name, {"name": name, "password": hash_password(password), "admin": True, "created": time.time()})
        for sub in ("user_sessions", "workspace"):
            os.makedirs(os.path.join(self.folder(name), sub), exist_ok=True)

    def single(self):
        """⭐ MULTI USER TO SINGLE USER: the last user, an admin, renamed users/__local/ — their conversations the single
        user's — their user.json gone with its password, and every sign-in over. Refused while another user, a provider
        to sign on with (sso.json) or a running server is there. Answers the name the folder had."""
        names = self.names()
        if os.path.isfile(os.path.join(self.data, "sso.json")):
            raise Refused("sso.json signs people on to this machine: remove it first")
        if len(names) != 1:
            raise Refused("%d users: an admin alone takes the machine back to single user — remove the others first"
                          % len(names) if names else "this machine is single user already")
        name = names[0]
        if not self.admin(name):
            raise Refused("%s is not an admin" % name)
        if self.serving():
            raise Refused("a server is running on this data folder: stop it first")
        with self.lock, _locked(os.path.join(self.root, ".layout.lock")):
            if os.path.isdir(self.local()):
                try:
                    os.rmdir(self.local())                 # an empty one, left behind; one that holds anything stays
                except OSError:
                    raise Refused("users/%s/ is there already, and not empty: move it aside first" % LOCAL)
            try:
                os.rename(self.folder(name), self.local())
            except OSError as ex:
                raise Refused("users/%s/ could not be renamed %s: %s" % (name, LOCAL, ex))
            with contextlib.suppress(OSError):
                os.remove(os.path.join(self.local(), "user.json"))
        self._change_signs(lambda signs: signs.clear())
        return name

    def has_password(self, name):
        """Whether `name` signs in with a password (a user made at a sign-on has none until one is given)."""
        return bool((self._record(name) or {}).get("password"))

    def password_ok(self, name, password):
        """Whether `password` is `name`'s — checked, and no sign-in made of it."""
        record = self._record(name) if NAME.match(name or "") else None
        return record is not None and verify_password(password or "", record.get("password") or {})

    # ── admins ───────────────────────────────────────────────────────────────────────────────────────────────
    def admin(self, name):
        """Whether `name` is an admin: the machine's settings, its users and its models theirs to change."""
        return bool(name) and (self._record(name) or {}).get("admin") is True

    def admins(self):
        return [n for n in self.names() if self.admin(n)]

    def set_admin(self, name, on):
        with self.lock:
            record = self._record(name)
            if record is None:
                raise Refused("%s is no user here" % name)
            if not on and record.get("admin") is True and self.admins() == [name]:
                raise Refused("%s is the last admin: make another first" % name)
            if on:
                record["admin"] = True
            else:
                record.pop("admin", None)
            self._write(name, record)

    def _record(self, name):
        try:
            with open(os.path.join(self.folder(name), "user.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write(self, name, record):
        os.makedirs(self.folder(name), exist_ok=True)
        _write_private(os.path.join(self.folder(name), "user.json"), record)   # only its owner reads a password

    # ── the commands (`silvann users …`) ──────────────────────────────────────────────
    def add(self, name, password, admin=False):
        if not NAME.match(name or ""):
            raise Refused("a user's name is 1 to 32 of a-z, 0-9, _ and -, beginning with a letter or a digit: not %r" % name)
        if not password:
            raise Refused("a user needs a password")
        if self._record(name) is not None:
            raise Refused("%s is a user already: `silvann users passwd %s` changes the password" % (name, name))
        record = {"name": name, "password": hash_password(password), "created": time.time()}
        if admin:
            record["admin"] = True
        self._write(name, record)
        for sub in ("user_sessions", "workspace"):
            os.makedirs(os.path.join(self.folder(name), sub), exist_ok=True)

    def passwd(self, name, password):
        record = self._record(name)
        if record is None:
            raise Refused("%s is no user here" % name)
        if not password:
            raise Refused("a user needs a password")
        record["password"] = hash_password(password)
        self._write(name, record)                      # a new password signs every session of theirs out (▶ `_stamp`)
        self._forget(name)

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
        self._forget(name)

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
        if self.admin(name) and self.admins() == [name]:
            raise Refused("%s is the last admin: make another first, or take the machine back to single user" % name)
        held = self.conversations(name)
        if held and not with_conversations:
            raise Refused("%s has %d conversation%s: `silvann users remove %s --with-conversations` removes them too"
                          % (name, held, "" if held == 1 else "s", name))
        shutil.rmtree(self.folder(name))
        self._forget(name)

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

    # ── signing in: every process's (▶ the module's note) ───────────────────────────────────────────────────────
    def _signs_path(self):
        return os.path.join(self.root, SIGN_INS)

    def _signs(self, again=False):
        """The sign-ins — {a token's hash: [its user, when it was last used, its user's stamp then]} — read again where
        the file changed since (another process signed someone in or out) or `again`."""
        path = self._signs_path()
        try:
            st = os.stat(path)
            at = (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError:
            self._store, self._store_at = {}, None
            return self._store
        if again or at != self._store_at:
            try:
                with open(path, encoding="utf-8") as f:
                    got = json.load(f)
                self._store = {k: v for k, v in got.items() if isinstance(v, list) and len(v) == 3} \
                    if isinstance(got, dict) else {}
            except (OSError, ValueError):
                self._store = {}
            self._store_at = at
        return self._store

    def _change_signs(self, change):
        """`change(signs)` made under the lock every process takes, the sign-ins unused for IDLE dropped, and written."""
        os.makedirs(self.root, exist_ok=True)
        path = self._signs_path()
        with self.store_lock, _locked(path + ".lock"):
            signs = dict(self._signs(again=True))
            change(signs)
            now = time.time()
            signs = {k: v for k, v in signs.items() if now - v[1] <= IDLE}
            _write_private(path, signs)
            self._store_at = None                       # read again at the next look: the file as written
            self._signs()

    def _forget(self, name):
        """`name`'s sign-ins over: their password or accounts changed, or they were removed."""
        self._change_signs(lambda signs: [signs.pop(k) for k in [k for k, v in signs.items() if v[0] == name]])

    def issue(self, name):
        """A token for `name` — a sign-in's, or one the server gives itself: a subagent's conversation on this server,
        made in its caller's folder (▶ the server's `_tool_agent`), for which no password is asked of its own Python."""
        record = self._record(name)
        if record is None:
            return None
        token = secrets.token_urlsafe(32)
        h, entry = _hashed(token), [name, time.time(), _stamp(record)]
        self._change_signs(lambda signs: signs.__setitem__(h, entry))
        return token

    def sign_in(self, name, password):
        """A token for `name`, or None — the same None for a name that is no user's as for a wrong password."""
        record = self._record(name) if NAME.match(name or "") else None
        if record is None or not verify_password(password or "", record.get("password") or {}):
            return None
        return self.issue(name)

    def signed(self, token):
        """Who `token` signs in, or None: a token never given, unused for `IDLE`, signed out, or its user's since removed,
        given a new password, or given or taken an account to sign on with — whichever process gave it or ended it."""
        if not token:
            return None
        h = _hashed(token)
        got = self._signs().get(h) or self._signs(again=True).get(h)
        if got is None:
            return None
        record = self._record(got[0])
        now = time.time()
        if now - got[1] > IDLE or record is None or _stamp(record) != got[2]:
            self._change_signs(lambda signs: signs.pop(h, None))
            return None
        if now - got[1] > TOUCH:                        # its last use written down, now and then
            def touch(signs):
                if h in signs:
                    signs[h][1] = now
            self._change_signs(touch)
        return got[0]

    def sign_out(self, token):
        if token:
            h = _hashed(token)
            self._change_signs(lambda signs: signs.pop(h, None))


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


class OwnTools:
    """⭐ A USER'S OWN TOOLS (⚖ 2026-11-18 *"the tools would be the read only link to the base tools plus the local
    user's added script"*): the ones their model added (▶ tools' `add_tool`), in users/<name>/skills_added/ — theirs
    alone, beside the machine's, which they read and run and never write. Each is on with its own settings, which the
    user's tools.json may change, and runs as their commands do: confined, outside the server (▶ tools' `_added_call`)."""

    def __init__(self, folder):
        self.added_folder = str(folder)
        self.skills = []
        self.generation = 0

    def refresh(self):
        from . import skills as _skills
        from . import tools_config
        found = _skills.read_all([self.added_folder]) if os.path.isdir(self.added_folder) else []
        if tools_config._signature(found) != tools_config._signature(self.skills):
            self.generation += 1
        self.skills = found

    def added(self, skill):
        root = os.path.realpath(self.added_folder)
        return os.path.realpath(skill.folder).startswith(root.rstrip(os.sep) + os.sep)

    def get(self, name):
        return next((s for s in self.skills if s.name == name), None)


class UserToolbox:
    """⭐ THE MACHINE'S TOOLS AS ONE USER HAS THEM — the machine's `Toolbox` (▶ tools_config), its skills and its startup
    check, with the user's `tools.json` over it: a tool the user switched off is off for them, and a setting the user may
    give (▶ `USER_SETTINGS`) is theirs; and beside it the tools the user's model added (▶ `OwnTools`), which `add_tool`
    puts in their folder rather than the machine's. The commands are confined to the workspace (▶ `Fence`). Everything
    else is the machine's, asked of it."""

    follows_links = False     # a link at the workspace's top is not followed: one the model made would be its way out

    def __init__(self, machine, path, data=None, own=None):
        self.machine, self.path = machine, str(path)
        self.own = own if own is not None else OwnTools(os.path.join(os.path.dirname(self.path), "skills_added"))
        self.own.refresh()
        self.closed = [os.path.abspath(str(data))] if data else []   # the data folder, granted of no system folder
        self._read_at, self._entries = None, {}

    def __getattr__(self, name):                       # the skill folders, the file, the startup check's, …
        return getattr(self.machine, name)

    # ── the user's own beside the machine's ─────────────────────────────────────────────────────────────────────
    @property
    def added_folder(self):
        return self.own.added_folder

    def _mine(self, name):
        """The user's own tool of that name — one the machine has none of — or None."""
        return None if name in self.machine.tools else self.own.get(name)

    @property
    def skills(self):
        names = {s.name for s in self.machine.skills}
        return list(self.machine.skills) + [s for s in self.own.skills if s.name not in names]

    @property
    def tools(self):
        got = dict(self.machine.tools)
        for s in self.own.skills:
            got.setdefault(s.name, {"enabled": s.enabled, "config": dict(s.config),
                                    "about": "the %s skill: %s" % (s.name, s.description or "(no description)")})
        return got

    def added(self, skill):
        """Whether a model added this skill — the user's, or the single user's before the machine was multi user."""
        return self.own.added(skill) or self.machine.added(skill)

    def removable(self, skill):
        """A user takes away or replaces only what their own model added — never the machine's, the single user's ones
        among them."""
        return self.own.added(skill)

    def refresh(self, keep=False):
        changed = self.machine.refresh(keep=keep)
        self.own.refresh()
        return changed

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
        """The machine's, their own tools', and the user's file: a change in any binds a session's tools again."""
        return (self.machine.generation, self.own.generation, self._read_at if self.entries() is not None else None)

    def _may_set(self, name, key):
        from .tools_config import BUILT_IN
        if self._mine(name) is not None:                     # their own tool: every setting theirs
            return True
        if key in USER_SETTINGS.get(name, ()):               # what a user brings of their own, built-in or skill
            return True
        if name in BUILT_IN:
            return False
        return self.machine.config(name).get(key) is None    # another skill's: what the machine left unset

    def switched_off(self, name):
        """Whether the user switched the tool off."""
        return str(self.entries().get(name, {}).get("enabled", "")).lower() == "no"

    def enabled(self, name):
        if self.switched_off(name):
            return "no"
        mine = self._mine(name)
        if mine is not None:
            return mine.enabled if mine.enabled in ("yes", "auto", "no") else "yes"
        return self.machine.enabled(name)

    def config(self, name, defaults=None):
        mine_skill = self._mine(name)
        if mine_skill is not None:
            got = dict(defaults or {})
            got.update(mine_skill.config)
        else:
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
        """The folders of the tools the user has switched on — the machine's and their own: what their commands may read
        and run, and never write."""
        return [os.path.realpath(sk.folder) for sk in self.skills if self.enabled(sk.name) != "no"]

    def writable_folders(self):
        """For a user, the folders of their tools switched on — read and run by their commands, never written (the
        confinement grants them so; ▶ tools.confinement)."""
        return self.tool_folders()

    def why_not(self, tool, command):
        if self.switched_off(tool):
            return "switched off in your settings (your tools.json)"
        if self._mine(tool) is not None:
            return None
        return self.machine.why_not(tool, command)

    def listing(self):
        out = self.machine.listing()
        for s in self.own.skills:
            out.setdefault(s.name, {"about": "the %s skill: %s — yours" % (s.name, s.description or "(no description)"),
                                    "unavailable": None})
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
            if switch in ("yes", "auto") and self._mine(name) is None and self.machine.enabled(name) == "no":
                said.append("%s stays off: the machine switched it off" % name)
            for k in sorted(e.get("config") or {}):
                if not str(k).startswith("_") and not self._may_set(name, k):
                    said.append("%s's %s stays the machine's" % (name, k))
        return said
