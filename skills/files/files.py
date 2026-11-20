"""The files skill: names under a folder, and the lines of a file that hold a pattern (▶ SKILL.md)."""
import fnmatch
import os
import re

MOST = 500
LINES = 50                        # the lines `grep` answers at once
WIDTH = 300                       # the most of a line it shows


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


def grep(pattern, file, start=1, context=None):
    """The lines of `file` from line `start` on that hold `pattern`, each "number: line" — at most LINES, then how to go
    on. A regular expression, capitals and small letters alike; one that is not a regular expression is looked for as
    it is written."""
    cwd = context["cwd"] if context else os.getcwd()
    path = os.path.realpath(os.path.join(cwd, os.path.expanduser(str(file))))
    if not os.path.isfile(path):
        raise ValueError("%s is not a file" % file)
    if isinstance(start, bool) or not isinstance(start, int) or start < 1:
        raise ValueError("(grep \"pattern\" \"file\" from): from is a line's number, 1 the first")
    if not str(pattern):
        raise ValueError("(grep \"pattern\" \"file\"): the pattern is empty")
    if context:
        context["check"]("read", path)            # asked as read asks: unasked where reads go, asked elsewhere
    try:
        holds = re.compile(str(pattern), re.IGNORECASE).search
    except re.error:
        holds = re.compile(re.escape(str(pattern)), re.IGNORECASE).search
    found, more, next_at, n = [], 0, None, 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            if n < start or not holds(line):
                continue
            if len(found) < LINES:
                line = line.rstrip("\r\n")
                found.append("%d: %s" % (n, line if len(line) <= WIDTH else line[:WIDTH] + " […]"))
            else:
                more += 1
                next_at = next_at or n
    if more:
        found.append("[… %d more lines hold it: (grep \"%s\" \"%s\" %d), or a narrower pattern]"
                     % (more, str(pattern).replace("\\", "\\\\").replace('"', '\\"'), file, next_at))
    if not found:
        return ["[no line of %s holds it%s — %d lines looked at]" % (file, " from line %d on" % start if start > 1 else "", n)]
    return found
