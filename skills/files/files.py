"""The files skill: names under a folder (▶ SKILL.md)."""
import fnmatch
import os

MOST = 500


def find(folder, pattern="*", context=None):
    """The paths under `folder` whose names match `pattern`, relative to it — a folder's with `/` — at most MOST."""
    cwd = context["cwd"] if context else os.getcwd()
    root = os.path.realpath(os.path.join(cwd, os.path.expanduser(str(folder))))
    if not os.path.isdir(root):
        raise ValueError("%s is not a folder" % folder)
    if context:
        context["check"]("read", root)            # asked as ls asks: unasked where reads go, asked elsewhere
    found, more = [], 0
    for at, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        rel = os.path.relpath(at, root)
        for name, tail in [(d, "/") for d in dirs] + [(f, "") for f in sorted(files)]:
            if fnmatch.fnmatch(name, str(pattern)):
                if len(found) < MOST:
                    found.append(os.path.normpath(os.path.join(rel, name)) + tail)
                else:
                    more += 1
    if more:
        found.append("[… %d more: a narrower folder or pattern finds them]" % more)
    return found
