"""WHERE THE MODEL'S SHELL REACHES (NN-128) — Landlock, the Linux kernel's own (5.13 or later), in two shapes.

⚖ 2026-10-26: *"can we make python's bash be chrooted into a specific working folder? … if you want to give the agent
access to a folder you can then symlink it manually into its chrooted setup so he can have access and that is consent
first permission, done from an actual console"* · *"if we are in a container i think it is fine as is, we just need to
unauthorize the config folder"* · *"consider direct installs as developers, so they should know better. we only secure
the docker as that is the intended usage of the tool"*.

- **The machine but a slice** (`spec_except`, a container's): every folder and file but what is closed — the tools
  file's folder and the server's own process (/proc, but for a few of its files that describe the machine). The folders
  on the way down to a closed one cannot be listed, and nothing can be made directly in them; a link that leads into a
  closed one, or above it, is not followed.
- **The workspace alone** (`spec`, a developer's choice on a machine of their own): reads, writes and runs inside the
  workspace and what each link at its top points to — `ln -s /data/photos <workspace>/photos`, made from a console, is
  the consent — besides the system's programs and libraries (`SYSTEM`), a few of /proc's files and /sys (`READABLE`),
  and the harmless devices (`DEVICES`). HOME and TMPDIR are the workspace's.

A chroot would need root; Landlock needs nothing but the kernel, and a container's seccomp profile that lets its three
calls through (Podman 3.4's does not: docker/README.md). The command is launched as `python -I -S jail.py SPEC ARGV…`:
this file makes the rules, restricts its own process — for good, and every process it starts — and puts the command in
its place with exec, so the server's own process is never restricted and never forks into a half-made rule set. The
kernel's Landlock ABI sets what the rules hold: from 1 (Linux 5.13) reads, writes, runs and makes; moving a file between
folders is refused under 1 (`mv` copies instead); a file's truncation by path is held from 3 (Linux 6.2) — under 1 and
2 a confined command can still empty a file it cannot read; a file's metadata (`stat`) is never held.
"""
import ctypes
import json
import os
import stat
import sys

# the kernel's calls and their numbers, the same on every architecture
_CREATE_RULESET, _ADD_RULE, _RESTRICT_SELF = 444, 445, 446
_VERSION = 1                      # LANDLOCK_CREATE_RULESET_VERSION
_PATH_BENEATH = 1                 # LANDLOCK_RULE_PATH_BENEATH
_NO_NEW_PRIVS = 38                # PR_SET_NO_NEW_PRIVS
EXECUTE, WRITE, READ, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_HANDLED = {1: (1 << 13) - 1, 2: (1 << 14) - 1, 3: (1 << 15) - 1, 4: (1 << 15) - 1}   # an ABI's rights; 5 on: all 16
_ON_A_FILE = EXECUTE | WRITE | READ | TRUNCATE | IOCTL_DEV                  # what a rule on a file (not a folder) holds
EVERYTHING = (1 << 16) - 1
RUN = EXECUTE | READ | READ_DIR

# ⭐ WHAT A CONFINED COMMAND REACHES BESIDES ITS WORKSPACE: the system's programs, libraries and settings (read and
#   run), what describes the machine (read), and the devices a program writes to harmlessly — /etc/resolv.conf's
#   target among them, where systemd keeps it, so names resolve
SYSTEM = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/etc", "/opt", "/snap",
          "/run/systemd/resolve", "/run/resolvconf")
READABLE = ("/sys", "/proc/cpuinfo", "/proc/meminfo", "/proc/stat", "/proc/loadavg", "/proc/uptime", "/proc/version",
            "/proc/filesystems", "/proc/sys/kernel/random")
DEVICES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom")

_abi = None


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.restype = ctypes.c_int
    return libc


def available():
    """(Landlock's ABI version here, None), or (None, why it is not here) — asked of the kernel once a process."""
    global _abi
    if _abi is None:
        if not sys.platform.startswith("linux"):
            _abi = (None, "Landlock is Linux's, and this machine runs %s" % ("Windows" if os.name == "nt" else sys.platform))
        else:
            libc = _libc()
            got = libc.syscall(ctypes.c_long(_CREATE_RULESET), None, ctypes.c_long(0), ctypes.c_long(_VERSION))
            err = ctypes.get_errno()
            _abi = (int(got), None) if got > 0 else (None, {
                38: "this kernel has no Landlock (it came with Linux 5.13)",
                95: "Landlock is off in this kernel (the lsm= of its boot leaves it out)"}.get(
                    err, "Landlock answered %s" % os.strerror(err)))
    return _abi


def links(workspace):
    """{name: what it points to} — each link at the workspace's top whose target is there: what its owner gave it."""
    out = {}
    try:
        entries = list(os.scandir(workspace))
    except OSError:
        return out
    for e in entries:
        if e.is_symlink():
            target = os.path.realpath(e.path)
            if os.path.exists(target):
                out[e.name] = target
    return out


def in_container():
    """Whether this process runs in a container: Podman's mark, Docker's, or the environment's `container`."""
    return os.path.exists("/run/.containerenv") or os.path.exists("/.dockerenv") or bool(os.environ.get("container"))


def spec_except(closed, abi, read_only=()):
    """The rules for a command that reaches the whole machine but `closed` (and /proc, but READABLE's), and only reads
    and runs what is in `read_only` (the skill folders: ▶ tools' `add_tool`): every entry beside each folder on the way
    down to one of those, all of it — made at each launch, so what was made since is in. A link found on the way is
    followed only where it leads neither into nor above one."""
    read_only = sorted({os.path.realpath(r) for r in read_only})
    closed = sorted({os.path.realpath(c) for c in closed} | {"/proc"} | set(read_only))
    under = lambda p, c: p == c or p.startswith(c.rstrip(os.sep) + os.sep)        # noqa: E731
    rules = []

    def walk(folder):
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return
        for name in names:
            p = os.path.join(folder, name)
            if any(under(p, c) for c in closed):
                continue
            if any(under(c, p) for c in closed):                      # on the way down to a closed one
                walk(p)
                continue
            real = os.path.realpath(p)
            if any(under(real, c) or under(c, real) for c in closed):  # a link into a closed one, or above it
                continue
            rules.append((real, EVERYTHING))
    walk(os.sep)
    rules += [(p, READ | READ_DIR) for p in READABLE if p.startswith("/proc")] + [(p, RUN) for p in read_only]
    return {"abi": abi, "rules": rules, "closed": [c for c in closed if c not in read_only], "read_only": read_only}


def spec(workspace, abi, read_only=(), run_also=()):
    """The rules for a command confined to `workspace`: [(path, rights)], and the ABI they are made under — besides,
    what it reads and runs: the skill folders (`read_only`) and what a tool the model added needs to run (`run_also`)."""
    rules = [(os.path.realpath(workspace), EVERYTHING)]
    rules += [(target, EVERYTHING) for target in links(workspace).values()]
    rules += [(p, RUN) for p in SYSTEM] + [(p, READ | READ_DIR) for p in READABLE]
    rules += [(p, READ | WRITE | IOCTL_DEV) for p in DEVICES]
    rules += [(os.path.realpath(p), RUN) for p in list(read_only) + list(run_also)]
    return {"abi": abi, "rules": rules, "read_only": [os.path.realpath(p) for p in read_only]}


def reaches(spec_, path):
    """Whether a confined command reaches `path`: inside the workspace or what a link there points to."""
    path = os.path.realpath(path)
    return any(path == p or path.startswith(p.rstrip(os.sep) + os.sep) for p, rights in spec_["rules"]
               if rights == EVERYTHING)


def launch(argv, spec_):
    """The command line that runs `argv` confined by `spec_`: this file run by this Python, isolated and without its
    site, then `argv` in its place."""
    return [sys.executable, "-I", "-S", os.path.abspath(__file__), json.dumps(spec_)] + list(argv)


def confine(spec_):
    """This process, and every one it starts, restricted to `spec_`'s rules — for good. A path not on this machine is
    left out. Raises OSError where the kernel refuses."""
    libc = _libc()
    abi = int(spec_["abi"])
    handled = _HANDLED.get(abi, EVERYTHING)
    attr = _Ruleset(handled)
    ruleset = libc.syscall(ctypes.c_long(_CREATE_RULESET), ctypes.byref(attr), ctypes.c_long(ctypes.sizeof(attr)),
                           ctypes.c_long(0))
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset: %s" % os.strerror(ctypes.get_errno()))
    try:
        for path, rights in spec_["rules"]:
            try:
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue
            try:
                folder = stat.S_ISDIR(os.fstat(fd).st_mode)
                rule = _PathBeneath(rights & handled & (EVERYTHING if folder else _ON_A_FILE), fd)
                if rule.allowed_access and libc.syscall(ctypes.c_long(_ADD_RULE), ctypes.c_long(ruleset),
                                                        ctypes.c_long(_PATH_BENEATH), ctypes.byref(rule),
                                                        ctypes.c_long(0)) < 0:
                    raise OSError(ctypes.get_errno(), "landlock_add_rule %s: %s" % (path, os.strerror(ctypes.get_errno())))
            finally:
                os.close(fd)
        if libc.prctl(_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS): %s" % os.strerror(ctypes.get_errno()))
        if libc.syscall(ctypes.c_long(_RESTRICT_SELF), ctypes.c_long(ruleset), ctypes.c_long(0)) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self: %s" % os.strerror(ctypes.get_errno()))
    finally:
        os.close(ruleset)


def main():
    try:
        confine(json.loads(sys.argv[1]))
    except (OSError, ValueError, IndexError) as ex:
        sys.stderr.write("the command was not run: it could not be confined to the workspace (%s)\n" % ex)
        sys.exit(126)
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
