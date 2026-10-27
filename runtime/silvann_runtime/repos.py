"""TOOLS FROM A REPOSITORY (NN-129) — the silvann agentic skills repository, and others laid out as it is.

⚖ 2026-10-26: *"the simplest way can be a new github project on eval apply labs that holds some python files separated
by packages, and each package contains the description of the package and the description of each verb inside the
package … an add_repo command so the agent can read and treat as mostly trustable other sources, and it has to have a
list_repo and list_package command so it can search between the available software list"* · *"add_repo and list_repo
can be based off git, so you get credential handling managed for free"*.

A repository is a git repository of packages, each a folder of one-file tools (the format `(add_tool …)` takes: `DOC`,
`CONFIG`, a command a function) and a `PACKAGE.md` saying what the package is; its `index.json` — written by its own
`make_index.py`, which reads the tools and never runs them — lists every package, every tool's file and its SHA-256,
its verbs and its settings. The server keeps a shallow clone of each in a cache of its own — the tools file's folder's
`repos/`, which the model's commands never reach — fetched again when it is older than REFRESH_SECONDS; the REPL reads
the index to list (▶ tools.py: `list_repo`, `list_package`) and a tool's file to copy it into the workspace
(`get_tool`), its hash checked against the index. Adding it is `(add_tool …)`'s.

A source is an `https://` git address — git's own credentials answer for a private one, and git never prompts — or a
folder on this machine. git is run with every protocol but https refused (some of git's transports run commands), and a
file whose real path leaves its repository, a link among them, is not read.
"""
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.parse

INDEX = "index.json"
GIT_SECONDS = 120
REFRESH_SECONDS = 600                  # a clone older than this is fetched again before it is read
MOST = 4 << 20                         # the most of an index or a tool read
_GIT = ["git", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always", "-c", "core.symlinks=true"]
_ENV = {"GIT_TERMINAL_PROMPT": "0", "GIT_PROTOCOL_FROM_USER": "0", "GIT_ASKPASS": "", "SSH_ASKPASS": ""}


class Unread(Exception):
    """A repository, or a file of it, that could not be read — and why."""


def _slug(source):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", re.sub(r"^https://", "", source.strip().rstrip("/")))[:120]


def _git(args, cwd=None):
    try:
        r = subprocess.run(_GIT + args, cwd=cwd, capture_output=True, text=True, timeout=GIT_SECONDS,
                           env=dict(os.environ, **_ENV), stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise Unread("git is not installed on this machine")
    except subprocess.TimeoutExpired:
        raise Unread("git took longer than %d s" % GIT_SECONDS)
    if r.returncode != 0:
        raise Unread("git: %s" % ((r.stderr or r.stdout).strip().splitlines() or ["exit %d" % r.returncode])[-1])


def checkout(source, cache):
    """The folder the repository's files are read from: a folder named as the source, or the cache's clone of an
    `https://` git address — cloned the first time, fetched again when older than REFRESH_SECONDS."""
    s = str(source).strip()
    if s.startswith("file://"):
        s = urllib.parse.urlsplit(s).path
    if os.path.isabs(s):
        if not os.path.isdir(s):
            raise Unread("%s is not a folder" % s)
        return s
    if not s.startswith("https://"):
        raise Unread("%r is no repository's address: an https:// git address, or a folder" % s)
    dest = os.path.join(cache, _slug(s))
    stamp = os.path.join(dest, ".git", "silvann_fetched")
    if not os.path.isdir(os.path.join(dest, ".git")):
        os.makedirs(cache, exist_ok=True)
        _git(["clone", "--depth", "1", "--quiet", "--", s, dest])
    elif not os.path.exists(stamp) or time.time() - os.path.getmtime(stamp) > REFRESH_SECONDS:
        _git(["fetch", "--depth", "1", "--quiet", "origin"], cwd=dest)
        _git(["reset", "--hard", "--quiet", "FETCH_HEAD"], cwd=dest)
    with open(stamp, "w") as f:
        f.write(s + "\n")
    return dest


def read(source, path, cache):
    """A file of the repository, as bytes: at most MOST of it, its real path inside the repository."""
    base = os.path.realpath(checkout(source, cache))
    full = os.path.realpath(os.path.join(base, path))
    if not full.startswith(base + os.sep) or os.sep + ".git" + os.sep in full[len(base):]:
        raise Unread("%s is not a file of the repository" % path)
    try:
        with open(full, "rb") as f:
            return f.read(MOST)
    except OSError as ex:
        raise Unread("%s: %s" % (path, ex.strerror))


def index(source, cache):
    """The repository's index — {"name", "description", "packages": {name: {"description", "about", "tools": {…}}}} —
    or Unread, why."""
    try:
        got = json.loads(read(source, INDEX, cache).decode("utf-8"))
    except ValueError as ex:
        raise Unread("its index.json is not JSON: %s" % ex)
    if not isinstance(got, dict) or not isinstance(got.get("packages"), dict):
        raise Unread("its index.json has no packages")
    return got


def tool_file(source, tool, cache):
    """A tool's file, as the index gives it — its SHA-256 checked: a file that is not what the index says is refused."""
    data = read(source, str(tool.get("file", "")), cache)
    if hashlib.sha256(data).hexdigest() != tool.get("sha256"):
        raise Unread("%s is not the file the repository's index lists (its SHA-256 differs)" % tool.get("file"))
    return data
