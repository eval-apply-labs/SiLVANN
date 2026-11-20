#!/usr/bin/env python3
"""A signed Windows file's unit key: what its compiler read of this tree, after the preprocessor, hashed (NN-195).

⚖ 2026-11-16 *"yes, build it on a review branch"*, after rel1.1.0 signed all eleven of its files: sign_plan.py keyed a
file by every engine file its compile includes, and sys's verb table — a `#define` no family expands — is in every
family's closure, so a new verb re-signed them all; their preprocessed units and generated kernels were the same bytes
as 1.0.3's (measurements/2026-11-16_sign_keys_node03.md). A file's bytes would not do instead: the Windows runner's
toolset moves under the build (its Rich header, its import hints) where the code does not.

The key is the preprocessed text the real build's compiler produces with the real build's flags — clang-cl's `/E`,
nvcc's own `.ii` files kept with `--keep` — of which only the lines this tree wrote count: a line marker (`# 12 "f"`,
`#line 12 "f"`) says which file the lines after it come from, and only a file inside `root` (the checkout: the engine,
the generated kernels, the carried OpenCL headers) is kept; the toolset's, the SDK's, Python's and pybind11's headers
are left out, as the include closure left them out. The markers themselves and blank lines are dropped and each line's
ends trimmed, so a comment or a moved line changes nothing; a macro expanded differently, a branch taken differently or
a literal changed changes the key.

    python3 scripts/unit_key.py ROOT OUT.unit PREPROCESSED… [--line WORD…]    their key, to OUT.unit, with its compile line
"""
import hashlib
import json
import ntpath
import os
import posixpath
import re
import sys

MARKER = re.compile(r'^\s*#\s*(?:line\s+)?(\d+)\s+"((?:[^"\\]|\\.)*)"')


def _inside(path, root):
    """Whether a marker's file is this tree's: under `root` — a Windows path read by Windows' rules (case, either slash),
    whichever system reads it. A name in angle brackets (`<built-in>`) is the compiler's; a relative one the compile's own."""
    p = path.replace("\\\\", "\\")                                     # MSVC's markers double each backslash
    if p.startswith("<"):
        return False
    windows = re.match(r"^[A-Za-z]:[\\/]", p) or re.match(r"^[A-Za-z]:[\\/]", root)
    rules = ntpath if windows else posixpath
    if not rules.isabs(p):
        return True
    p, r = rules.normcase(rules.normpath(p)), rules.normcase(rules.normpath(root))
    return p == r or p.startswith(r.rstrip(rules.sep) + rules.sep)


def ours(text, root):
    """The lines of a preprocessed unit that this tree wrote, each trimmed, none blank, no marker among them."""
    keep, out = True, []
    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            keep = _inside(m.group(2), root)
            continue
        line = line.strip()
        if keep and line:
            out.append(line)
    return out


def key(texts, root):
    """The hash of the lines this tree wrote across `texts`, in their order, and how many there were."""
    h, n = hashlib.sha256(), 0
    for text in texts:
        for line in ours(text, root):
            h.update(line.encode("utf-8") + b"\n")
            n += 1
        h.update(b"\0")                                               # where one unit ends and the next begins
    return h.hexdigest(), n


def write(out, texts, root, note="", line=None):
    """`out` (a `.unit` beside its binary): the key, the lines it hashed, what made it, and the compile line — its
    compiler's name and switches, which sign_plan.py keys the file by beside the key (a switch can change the code and
    not the preprocessed text: `/O2`, `-O3`)."""
    digest, n = key(texts, root)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"unit": digest, "lines": n, "how": note, "line": list(line or [])}, f, indent=1)
    return digest, n


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    root, out, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    files, line = (rest[:rest.index("--line")], rest[rest.index("--line") + 1:]) if "--line" in rest else (rest, [])
    texts = [open(p, encoding="utf-8", errors="replace").read() for p in files]
    digest, n = write(out, texts, root, "preprocessed: " + ", ".join(os.path.basename(p) for p in files), line)
    print("%s: %s over %d lines" % (out, digest[:16], n))
