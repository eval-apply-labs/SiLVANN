"""USERS — an account each: a folder, a password and a config; an account sees only its own conversations (NN-150).

⚖ 2026-11-02 *"if we do the multi seat we should also set up a multi user ui, where users have a folder with at least a
password and they get in and only see their chat folder"* · *"there should be a per user config (with the tool overrides,
each users brave key maybe, etc)"*.

⭐ AN ACCOUNT FROM THE FIRST RUN, AND NOTHING ANSWERED WITHOUT A SIGN-IN (NN-197, ⚖ 2026-11-19 *"maybe we need to create
the account at install time, and single user mode makes it so that a second user login is refused outright if an user is
in the system"* · *"now __local disappears, as it is a normal account with a password... for single_user we need to
check who is the active_account and we refuse other users when we do that. maybe until the app re-opens, then you can
log in as a different user"* · *"also on sign-out"*). A machine's first account is made where SiLVANN is installed — the
installer's question, the UI's first page, or `silvann users add <name>` (▶ `first`) — and is its admin
(`"admin": true` in their user.json): only an admin touches the machine's settings, its accounts and its models. A
provider to sign on with (`sso.json`, ▶ sso.py) makes an account at a first sign-on.

⭐ SINGLE USER OR MULTI USER is the machine's mode, kept in `users/.mode` (▶ `mode`, `multi`, `single`):
  single   ONE ACCOUNT AT A TIME HAS THE MACHINE: the first to sign in is the active one, and another account's sign-in
           is refused until the active one signs out, its last sign-in ends (a terminal's chat closed: ▶ `end`) or
           SiLVANN is opened again (▶ `issue`, `sign_out`, `open`). An
           admin's model reaches the machine as the tools file lets it (▶ `reaches_machine`); any other account's works
           in its own folder (▶ `Fence`). The UI listens on this machine alone.
  multi    the accounts are signed in side by side, each one's model in its own folder, the admins' too; the UI listens
           on the network.
An account's folder is theirs whole:
```
  users/<name>/user.json                 their name, their password (scrypt), whether they are an admin, and the
                                           accounts they sign on with
  users/<name>/tools.json                their settings over the machine's tools file (▶ `UserToolbox`)
  users/<name>/skills_added/             the tools their model added (▶ tools' `add_tool`), theirs alone
  users/<name>/user_sessions/<model>/    their conversations
  users/<name>/workspace/                their REPL's workspace
  users/<name>/tools_library.json        the definitions their REPL kept; tools_allowed.json, the folders it allowed
```
A signed-in account's conversations are the ones in its folder, and no other: another's answers as one that is not there.
A machine of the releases before accounts kept its single user's in `users/__local/` (NN-196), and before that at the
data folder's top: the first account takes them (▶ `first`). On a machine whose users came before `__local` (NN-150),
the conversations made before the first user are at the data folder's top, nobody's until `silvann users adopt <name>`
moves them into a folder.

A sign-in is every process's on the data folder (▶ `sign_in`, `signed`) — the server's, and the UI's, which checks one
itself: so the UI asks for it with no model running, and a model stopped and started signs no one out. The tokens'
hashes are kept in `users/.sign_ins.json`, which no user's REPL reaches (▶ `Fence`).

⭐ A SIGN-IN IS OVER TEN MINUTES AFTER ITS LAST USE, COUNTED FROM THE END OF A RESPONSE (⚖ 2026-11-20 *"make it 10
mins instead of 12h, and only start after the response has happened so an agentic workflow can continue on its
own"*). A use is something its account does — a request of theirs, a page of the UI they open or touch — and never a
page asking after the server by itself (▶ `signed`'s `use`). While a response to it is in the making the sign-in is
held, however long the model works, and its ten minutes begin as the response ends (▶ `hold`, `release`).

⭐ A WRONG PASSWORD COSTS A MINUTE (⚖ 2026-11-20 *"add a fixed 60 second wrong password counter, and log it in the
system log maybe, with the ip address and the failed user login"*): the name it was given for is not looked at again
from that address for `WAIT` seconds — a right password too is refused meanwhile, saying how long is left — and the
system's log is told the name and the address (▶ `_checked`, `_log`).

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
import sys
import threading
import time

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
BEFORE = "__local"               # the single user's folder of the releases before accounts: the first account takes it
MODE = ".mode"                   # in users/: "single" or "multi" (▶ the module's note)
IDLE = 600                       # a sign-in is over this long after its last use, or the end of a response to it
TOUCH = 10                       # a sign-in's last use written down at most this often
WAIT = 60                        # after a wrong password, the name it was given for waits this long at that address
SIGN_INS = ".sign_ins.json"      # in users/: the sign-ins, every process's
WAITS = ".sign_in_waits.json"    # in users/: when each wrong password was, every process's (▶ `Users._checked`)
SERVING = ".serving"             # in users/: held by each server running on the data folder (▶ `serve`)
OPEN = ".open"                   # in users/: held by each UI and server running on the data folder (▶ `open`)
SLOTS = 4096                     # on Windows a process holds one byte of these (▶ `_hold`)
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


_HELD = []                       # this process's holds (▶ `Users._hold`), kept open while it runs


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


_NO_ONE = hash_password(secrets.token_urlsafe(16))      # a password nobody has: what a sign-in as no user is checked against


class Refused(Exception):
    """What a users' command cannot do, said."""


class Wait(Refused):
    """A password not looked at: a wrong one was given for this name from this address less than `WAIT` ago."""

    def __init__(self, seconds):
        self.seconds = max(1, int(seconds + 0.999))
        Refused.__init__(self, "a wrong password was given for this name less than %d seconds ago: the next try in %d "
                               "seconds" % (WAIT, self.seconds))


_LOG = []                        # the system's log, opened once a process


def _log(line):
    """`line` in the system's log — syslog's auth facility, a warning, as `silvann`: where sshd's refusals go on Linux
    and macOS — and on this process's own output, which is the log a container or Windows keeps, and the one the UI
    shows of a server it started."""
    try:
        import syslog
        if not _LOG:
            syslog.openlog("silvann", syslog.LOG_PID, syslog.LOG_AUTH)
            _LOG.append(True)
        syslog.syslog(syslog.LOG_WARNING, line)
    except Exception:                                  # noqa: BLE001 — no syslog here (Windows): the line below is the log
        pass
    print("[users] " + line, file=sys.stderr, flush=True)


class InUse(Refused):
    """A sign-in refused on a single user's machine: another account has it."""

    def __init__(self, name):
        self.name = name
        Refused.__init__(self, "%s is signed in, and this machine is single user: one account at a time. They sign out, "
                               "or SiLVANN is closed and opened again" % name)


def _stamp(record):
    """What a sign-in keeps of its user: their password and the accounts they sign on with — either changed, by this
    process or the `silvann users` command, and the sign-in ends."""
    return hashlib.sha256(json.dumps([record.get("password"), record.get("sso", [])], sort_keys=True).encode()).hexdigest()


class Users:
    """The accounts of a data folder, the machine's mode, and who is signed in."""

    def __init__(self, data):
        self.data = os.path.abspath(str(data))
        self.root = os.path.join(self.data, "users")
        self.lock = threading.Lock()
        self.store_lock = threading.Lock()
        self._store, self._store_at = {}, None       # the sign-ins file as last read, and what it was then (▶ `_signs`)
        self.try_lock = threading.Lock()             # one password looked at at a time (▶ `_checked`)
        self.held_lock = threading.Lock()
        self.held, self.keeper = {}, None            # a sign-in's hash -> the responses in the making for it (▶ `hold`)

    # ── the folders ───────────────────────────────────────────────────────────
    def folder(self, name):
        return os.path.join(self.root, name)

    def sessions(self, name, model):
        return os.path.join(self.folder(name), "user_sessions", model)

    def workspace(self, name):
        return os.path.join(self.folder(name), "workspace")

    def tools_file(self, name):
        return os.path.join(self.folder(name), "tools.json")

    def files(self, name):
        """What an account's REPL keeps in its folder: {"workspace", "library" — the definitions kept —, "allowed" — the
        folders allowed}."""
        f = self.folder(name)
        return {"workspace": os.path.join(f, "workspace"), "library": os.path.join(f, "tools_library.json"),
                "allowed": os.path.join(f, "tools_allowed.json")}

    def unowned(self):
        """Where the conversations made before a machine's users (NN-150) wait, nobody's, for `adopt`: the data
        folder's `user_sessions/`."""
        return os.path.join(self.data, "user_sessions")

    def names(self):
        """The accounts, by name: each a folder holding a `user.json`."""
        try:
            return sorted(d for d in os.listdir(self.root) if NAME.match(d)
                          and os.path.isfile(os.path.join(self.root, d, "user.json")))
        except FileNotFoundError:
            return []

    def ready(self):
        """Whether anyone can sign in: the machine has an account, or a provider to sign on with (▶ sso.py), whose
        accounts are made at their first sign-on. Where it has neither, its first account is still to be made (▶
        `first`) and nothing else of it answers."""
        return bool(self.names()) or os.path.isfile(os.path.join(self.data, "sso.json"))

    # ── the mode (▶ the module's note) ──────────────────────────────────────────────────────────────────────
    def mode(self):
        """"single" or "multi": what users/.mode says — and, on a machine whose mode is not written yet (the releases
        before NN-197), multi where it has a user or a provider to sign on with, as that machine was."""
        try:
            with open(os.path.join(self.root, MODE), encoding="utf-8") as f:
                said = f.read().strip()
            if said in ("single", "multi"):
                return said
        except OSError:
            pass
        return "multi" if self.ready() else "single"

    def _write_mode(self, mode):
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, MODE)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write(mode + "\n")
        os.replace(path + ".tmp", path)

    def settle(self):
        """⭐ A MACHINE KEEPS THE MODE IT HAD: users/.mode written where it is not there yet — multi user where the
        machine has a user or a provider to sign on with (a machine of the releases before NN-197 with users was multi
        user), single user otherwise. Run as each process starts. Answers what it wrote, a line, or nothing."""
        os.makedirs(self.root, exist_ok=True)
        with _locked(os.path.join(self.root, ".layout.lock")):
            if os.path.isfile(os.path.join(self.root, MODE)):
                return []
            mode = self.mode()
            self._write_mode(mode)
        return ["users/%s: %s user" % (MODE, mode)]

    def reaches_machine(self, name):
        """Whether `name`'s model reaches the machine as the tools file lets it — its commands the machine's, its tools
        and their settings the machine's own — rather than working in the account's folder alone (▶ `Fence`): an admin
        on a single user's machine, the person it is installed for at it, one account at a time."""
        return self.mode() == "single" and self.admin(name)

    # ── the first account (▶ the module's note) ─────────────────────────────────────────────────────────────
    def _gather(self):
        """What a single user's machine kept at the data folder's top before users/__local/ (the releases before
        NN-196: user_sessions/, workspace/, tools_library.json, the folders allowed in tools.json) moved into
        users/__local/, each where it has none of its own yet. Answers what moved, a line each."""
        moved = []
        before = self.folder(BEFORE)
        os.makedirs(before, exist_ok=True)
        top, mine = self.unowned(), os.path.join(before, "user_sessions")
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
            src, dst = os.path.join(self.data, was), os.path.join(before, now)
            if os.path.lexists(src) and not os.path.lexists(dst):
                os.replace(src, dst)
                moved.append(was)
        return moved

    def first(self, name, password):
        """⭐ THE MACHINE'S FIRST ACCOUNT, ITS ADMIN — made where it has none. What a single user's machine of the
        releases before accounts kept is this account's from now on: users/__local/ renamed its folder, with what the
        data folder's top held before that — its conversations, workspace and definitions (within a major an update
        keeps them). The machine stays in the mode it had: single user, unless it was multi user already (a provider to
        sign on with and no user yet), whose conversations at the top then stay for `adopt`. Answers what the account
        took of the machine's, a line each."""
        if not NAME.match(name or ""):
            raise Refused("an account's name is 1 to 32 of a-z, 0-9, _ and -, beginning with a letter or a digit: not %r"
                          % name)
        if not password:
            raise Refused("the account needs a password")
        os.makedirs(self.root, exist_ok=True)
        took = []
        with self.lock, _locked(os.path.join(self.root, ".layout.lock")):
            if self.names():
                raise Refused("this machine has its first account: an admin adds another in the UI's Users, or with "
                              "`silvann users add`")
            if os.path.lexists(self.folder(name)):
                raise Refused("%s is taken: users/%s/ is there already — choose another name" % (name, name))
            if not os.path.isfile(os.path.join(self.root, MODE)):
                self._write_mode(self.mode())              # before the account is there: an account makes no machine multi user
            if self.mode() == "single":
                had = os.path.isdir(self.folder(BEFORE))
                took = (["users/%s/" % BEFORE] if had else []) + self._gather()
                try:
                    os.rename(self.folder(BEFORE), self.folder(name))
                except OSError as ex:
                    raise Refused("users/%s/ could not be renamed %s: %s" % (BEFORE, name, ex))
            self._write(name, {"name": name, "password": hash_password(password), "admin": True, "created": time.time()})
            for sub in ("user_sessions", "workspace"):
                os.makedirs(os.path.join(self.folder(name), sub), exist_ok=True)
        return took

    # ── a part of the app on the data folder: a server (▶ `multi`, `single`, which wait for none to run), or any ────
    @staticmethod
    def _someone(path):
        """Whether a process holds `path` now (▶ `_hold`)."""
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

    @staticmethod
    def _hold(path):
        """`path` held by this process until it ends, however it ends — a shared lock on Linux, one byte of SLOTS on
        Windows (whose locks are not shared) — which the system lets go of then."""
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
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

    def serve(self):
        """This process a server on the data folder until it ends (▶ `_hold`, of `users/.serving`)."""
        os.makedirs(self.root, exist_ok=True)
        self._hold(os.path.join(self.root, SERVING))

    def serving(self):
        """Whether a server runs on the data folder now (▶ `serve`)."""
        return self._someone(os.path.join(self.root, SERVING))

    def open(self):
        """⭐ THIS PROCESS A PART OF THE APP UNTIL IT ENDS — a UI or a server on the data folder (▶ `_hold`, of
        `users/.open`). Answers whether it opened the app: no other part of it was running. Where it did, on a single
        user's machine, every sign-in is over — whoever had the machine has it no more, and the next account to sign in
        does (⚖ 2026-11-19 *"until the app re-opens, then you can log in as a different user"*). A model stopped and
        started by a UI that stays, or a UI started again beside its server, opens nothing."""
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, OPEN)
        with _locked(os.path.join(self.root, ".layout.lock")):   # two parts starting together: one opens, one joins
            opened = not self._someone(path)
            self._hold(path)
        if opened and self.mode() == "single":
            self._change_signs(lambda signs: signs.clear())
        return opened

    # ── the switch (⚖ 2026-11-18 *"the admin consolle has the single user vs multi user option in the config
    #    panel"*): the mode written, read by each UI and server as it starts ──────────────────────────────────────────
    def multi(self):
        """⭐ SINGLE USER TO MULTI USER: the accounts signed in side by side from now on, each one's model in its own
        folder. Refused where the machine has no admin to keep it, or while a server runs on the data folder — its
        conversations' tools are set up for the mode it started in."""
        if self.mode() == "multi":
            raise Refused("this machine is multi user already")
        if not self.admins():
            raise Refused("this machine has no admin yet: `silvann users add <name>` makes its first account, an admin")
        if self.serving():
            raise Refused("a server is running on this data folder: stop it first — its conversations' tools are set up "
                          "for the mode it started in")
        with self.lock, _locked(os.path.join(self.root, ".layout.lock")):
            self._write_mode("multi")

    def single(self):
        """⭐ MULTI USER TO SINGLE USER: one account at a time from now on, and every sign-in over — the next to sign in
        has the machine. The accounts stay, each with its folder. Refused while a server runs on the data folder."""
        if self.mode() == "single":
            raise Refused("this machine is single user already")
        if self.serving():
            raise Refused("a server is running on this data folder: stop it first")
        with self.lock, _locked(os.path.join(self.root, ".layout.lock")):
            self._write_mode("single")
        self._change_signs(lambda signs: signs.clear())

    def has_password(self, name):
        """Whether `name` signs in with a password (a user made at a sign-on has none until one is given)."""
        return bool((self._record(name) or {}).get("password"))

    def password_ok(self, name, password, at=None):
        """Whether `password` is `name`'s — checked as a sign-in's is (▶ `_checked`, which may raise `Wait`), and no
        sign-in made of it."""
        return self._checked(name, password, at) is not None

    def _waits(self, change):
        """`change(waits, now)` made on the wrong passwords' times — {"<address> <name>": when} — under the lock every
        process takes, the ones `WAIT` old dropped, and written where it changed them."""
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, WAITS)
        with self.store_lock, _locked(path + ".lock"):
            try:
                with open(path, encoding="utf-8") as f:
                    read = json.load(f)
            except (OSError, ValueError):
                read = {}
            now = time.time()
            waits = {k: t for k, t in read.items() if isinstance(t, (int, float)) and 0 <= now - t < WAIT} \
                if isinstance(read, dict) else {}
            try:
                return change(waits, now)
            finally:
                if waits != read:
                    _write_private(path, waits)

    def _checked(self, name, password, at=None):
        """⭐ A PASSWORD CHECKED, ONE TRY A MINUTE: `name`'s record where `password` is theirs, else None — the same
        None for a name that is no account's — and `Wait` where a wrong one was given for this name from the address
        `at` (a browser's, a caller's; None from this machine's own commands) less than `WAIT` ago. The wait is fixed:
        a try refused for it is not looked at and does not lengthen it. One password is looked at at a time on a data
        folder, whichever process is asked: tries made at once are taken one after the other, so they are as many
        guesses as tries let through and no more — and a flood of them costs one processor. ⛔ What it bounds is the
        guesses at a name from an address — sixty an hour; a guesser at many addresses has as many times that, and the
        system's log names each."""
        known = NAME.match(name or "") is not None
        key = "%s %s" % (at or "", name if known else "")     # every name that could be no account's: one name

        def waiting(waits, now):
            left = WAIT - (now - waits[key]) if key in waits else 0
            if left > 0:
                raise Wait(left)

        os.makedirs(self.root, exist_ok=True)
        with self.try_lock, _locked(os.path.join(self.root, WAITS + ".one")):
            self._waits(waiting)
            record = self._record(name) if known else None
            stored = (record or {}).get("password") or {}
            if stored.get("salt"):
                right = verify_password(password or "", stored)
            else:
                # ⛳ A NAME THAT IS NO USER'S COSTS WHAT A WRONG PASSWORD COSTS: without it the refusal came back at
                #   once, and the time an answer took said which names are users'
                verify_password(password or "", _NO_ONE)
                right = False
            if right:
                return record
            self._waits(lambda waits, now: waits.__setitem__(key, now))
        _log("wrong password for %s from %s" % (name if record is not None else "%r, no account here," % str(name)[:40],
                                                at or "an address not known"))
        return None

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

    def space(self, name):
        """The most an account's folder may hold, in bytes — None where no admin set one (▶ space.py)."""
        got = (self._record(name) or {}).get("space")
        return int(got) if isinstance(got, (int, float)) and not isinstance(got, bool) and got > 0 else None

    def set_space(self, name, size):
        """An account's space: `size` bytes, or None for no limit — an admin's to set (▶ space.py, which holds to it)."""
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            raise Refused("an account's space is a number of bytes above zero, or none")
        with self.lock:
            record = self._record(name)
            if record is None:
                raise Refused("%s is no user here" % name)
            if size is None:
                record.pop("space", None)
            else:
                record["space"] = int(size)
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
        """Another account — the machine's first is an admin whatever is asked (▶ `first`)."""
        if not self.names():
            return self.first(name, password)
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
            raise Refused("%s is the last admin: make another first" % name)
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

    def _used(self, hashes, within=0.0):
        """The sign-ins of `hashes` used now — written where one's last use is more than `within` seconds old. One
        that is over stays over."""
        now = time.time()
        signs = self._signs()
        if all(h not in signs or now - signs[h][1] <= within for h in hashes):
            return

        def touch(signs):
            for h in hashes:
                if h in signs:
                    signs[h][1] = now
        self._change_signs(touch)

    def hold(self, token):
        """⭐ A RESPONSE IN THE MAKING FOR `token`'s SIGN-IN: held until `release`, however long the model works — an
        answer of an hour, a tool's long command — and whichever process asks after it meanwhile: its last use is
        written again every `TOUCH` seconds while this process lives. A process that ends leaves it to its ten
        minutes."""
        if not token:
            return
        h = _hashed(token)
        with self.held_lock:
            self.held[h] = self.held.get(h, 0) + 1
            if self.keeper is None or not self.keeper.is_alive():
                self.keeper = threading.Thread(target=self._keep, name="silvann-sign-ins", daemon=True)
                self.keeper.start()

    def release(self, token):
        """The response is done: the sign-in's ten minutes begin here (▶ `hold`)."""
        if not token:
            return
        h = _hashed(token)
        with self.held_lock:
            left = self.held.get(h, 0) - 1
            if left > 0:
                self.held[h] = left
            else:
                self.held.pop(h, None)
        self._used([h], within=1.0)

    def _keep(self):
        while True:
            time.sleep(TOUCH)
            with self.held_lock:
                hashes = list(self.held)
                if not hashes:
                    self.keeper = None
                    return
            try:
                self._used(hashes, within=TOUCH / 2)
            except Exception:                           # noqa: BLE001 — the file not written this once: again in TOUCH
                pass

    def _forget(self, name):
        """`name`'s sign-ins over: their password or accounts changed, or they were removed."""
        self._change_signs(lambda signs: [signs.pop(k) for k in [k for k, v in signs.items() if v[0] == name]])

    def _live(self, signs):
        """The accounts `signs` signs in now: a sign-in used within IDLE whose account is as it was when it was made."""
        now, names = time.time(), set()
        for name, used, stamp in signs.values():
            record = self._record(name) if name not in names and now - used <= IDLE else None
            if record is not None and _stamp(record) == stamp:
                names.add(name)
        return names

    def active(self):
        """On a single user's machine, the account that has it — the one signed in — or None."""
        if self.mode() != "single":
            return None
        return next(iter(sorted(self._live(self._signs(again=True)))), None)

    def issue(self, name):
        """A token for `name` — a sign-in's, or one the server gives itself: a subagent's conversation on this server,
        made in its caller's folder (▶ the server's `_tool_agent`), for which no password is asked of its own Python.
        ⛔ ON A SINGLE USER'S MACHINE ONE ACCOUNT AT A TIME: while another's sign-in lives this one is refused (`InUse`,
        which names it) — checked and written under the lock every process takes, so two sign-ins at once make one."""
        record = self._record(name)
        if record is None:
            return None
        token = secrets.token_urlsafe(32)
        h, entry = _hashed(token), [name, time.time(), _stamp(record)]
        single = self.mode() == "single"

        def put(signs):
            others = sorted(self._live(signs) - {name}) if single else []
            if others:
                raise InUse(others[0])
            signs[h] = entry
        self._change_signs(put)
        return token

    def sign_in(self, name, password, at=None):
        """A token for `name`, or None — the same None for a name that is no user's as for a wrong password, either
        said to the system's log with the address `at` it came from; `Wait` for a minute after one (▶ `_checked`). A
        password that is theirs, on a single user's machine another account has: `InUse` (▶ `issue`)."""
        return None if self._checked(name, password, at) is None else self.issue(name)

    def signed(self, token, use=True):
        """Who `token` signs in, or None: a token never given, unused for `IDLE`, signed out, or its user's since removed,
        given a new password, or given or taken an account to sign on with — whichever process gave it or ended it.
        Asking is a use of it, from which its ten minutes count again — but with `use` false: a page asking by itself
        whether its sign-in lives, which keeps none alive."""
        if not token:
            return None
        h = _hashed(token)
        got = self._signs().get(h) or self._signs(again=True).get(h)
        if got is None:
            return None
        record = self._record(got[0])
        if time.time() - got[1] > IDLE or record is None or _stamp(record) != got[2]:
            self._change_signs(lambda signs: signs.pop(h, None))
            return None
        if use:
            self._used([h], within=TOUCH)               # its last use written down, now and then
        return got[0]

    def end(self, token):
        """`token`'s sign-in over, and no other: a terminal's chat closed, a subagent's conversation done. On a single
        user's machine the account has it for as long as another sign-in of its own lives."""
        if token:
            h = _hashed(token)
            self._change_signs(lambda signs: signs.pop(h, None))

    def sign_out(self, token):
        """The person signs out. On a multi user machine that is `token`'s sign-in over (▶ `end`). ⛔ On a single
        user's the account leaves the machine whole — its every sign-in over, another browser's, a terminal's and the
        ones the server gave itself for it — and the machine is the next account's to sign in to (⚖ 2026-11-19 *"also on
        sign-out"*)."""
        if not token:
            return
        h = _hashed(token)
        single = self.mode() == "single"

        def out(signs):
            got = signs.pop(h, None)
            if got is not None and single:
                for k in [k for k, v in signs.items() if v[0] == got[0]]:
                    signs.pop(k)
        self._change_signs(out)


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

    def __init__(self, users, me, machine_sealed=(), keep=(), tools=None, own=()):
        self.users, self.me = users, me
        self.machine = [os.path.realpath(p) for p in machine_sealed]
        self.keep = [os.path.realpath(k) for k in keep]       # the runtime and the venv: read and run
        self.tools = tools or (lambda: [])                     # the folders of the user's tools switched on, asked now
        self.own = [os.path.realpath(p) for p in own]          # a conversation's own files: read, by its REPL alone

    def opened(self, *folders):
        """⭐ THIS FENCE WITH A DOOR — `folders` read as well: a conversation's own attachments and pictures, for that
        conversation's REPL and no other's (⚖ 2026-11-20: *"uploads to the per chat folder and says "uploaded <filename>
        from the chat" to the model, so the model can read it himself ... those remain private"*). Read, never written;
        the rest of the conversation — its state, its transcript, its password — stays closed, and every other
        conversation whole."""
        return Fence(self.users, self.me, self.machine, self.keep, self.tools, own=folders)

    def home(self):
        return os.path.realpath(self.users.workspace(self.me))

    def readable(self):
        from . import jail
        return (self.keep + self.own + [os.path.realpath(t) for t in self.tools()] + list(jail.SYSTEM)
                + list(jail.READABLE))

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


class Conversations:
    """⭐ WHAT A MODEL THAT REACHES THE MACHINE STILL DOES NOT READ: THE CONVERSATIONS (NN-200, ⚖ 2026-11-20: *"those
    remain private"* · *"i send them to a chat, and once the chat is deleted so are those files"*). An admin's model on
    a single user's machine reaches the machine as the tools file lets it (▶ `Users.reaches_machine`) — and a file sent
    to one of its conversations would then be every conversation's to read. So every account's conversations are
    closed to it: their transcripts, their states, the files sent to them. Its own conversation's files are its door
    (▶ `opened`), read and never written. It stands where a classifier keeps its `sealed` list, beside what the
    machine seals — the tools file's folder, the server's own process.
    ⛔ What holds it: the commands that run inside the server — (read …), (grep …), (ls …) — and a command line that
    names a closed path, everywhere; a shell command that reaches one by a path the classifier does not see, where the
    machine's commands are confined (a container, or the tools file's `bash` setting `confined`) — on a direct
    install a command is the server's user, as the docs say of everything there."""

    def __init__(self, users, machine_sealed=(), own=()):
        self.users = users
        self.machine = [os.path.realpath(p) for p in machine_sealed]
        self.own = [os.path.realpath(p) for p in own]

    def opened(self, *folders):
        """This seal with a door: `folders`, a conversation's own files, read by its REPL alone."""
        return Conversations(self.users, self.machine, own=folders)

    def roots(self):
        """Where conversations are kept: each account's `user_sessions/`, and the ones made before there were users."""
        return [os.path.realpath(os.path.join(self.users.folder(n), "user_sessions")) for n in self.users.names()] + [
            os.path.realpath(self.users.unowned())]

    def paths(self):
        """What is closed, as paths that are there now — a command line naming one is refused, and a confined command
        does not reach one: the machine's own, and of the conversations everything but this one's files."""
        out = list(self.machine)
        for root in self.roots():
            if any(o.startswith(root.rstrip(os.sep) + os.sep) for o in self.own):
                out += _closed(root, self.own)
            elif os.path.exists(root):
                out.append(root)
        return out

    def covers(self, target, touches="read"):
        """⛔ Whether `target` is closed to a read or a write — a path not made yet as well as one that is."""
        real = os.path.realpath(str(target))
        under = lambda p, a: p == a or p.startswith(a.rstrip(os.sep) + os.sep)       # noqa: E731
        if any(under(real, a) for a in self.machine):
            return True
        if any(under(real, o) for o in self.own):
            return False                                   # its own files: read (a write is the classifier's to refuse)
        return any(under(real, r) for r in self.roots())

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
        """Whether a model added this skill — the user's own, or the machine's: an admin's model on a single user's
        machine adds to the machine's (▶ `Users.reaches_machine`)."""
        return self.own.added(skill) or self.machine.added(skill)

    def removable(self, skill):
        """A user takes away or replaces only what their own model added — never the machine's."""
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
