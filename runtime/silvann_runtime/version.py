"""THIS RELEASE'S VERSION — the one `release/README.md` declares ("Developer preview, vX.Y.Z"), and so which of a pack's
programs a boot reads: `lisp/v<X.Y.Z>/` of the highest release at or below this one (▶ `model_folder.program_dir`).
On main, the nightly, it is the release being prepared; `scripts/assemble_release.py` refuses a release whose README
names another.
"""
import re

RELEASE = "0.29.0"

_VERSION = re.compile(r"v?(\d+)\.(\d+)\.(\d+)$")


def parse(text):
    """`"v0.24.0"` or `"0.24.0"` as (0, 24, 0); None when it is not one."""
    m = _VERSION.match(text or "")
    return tuple(int(g) for g in m.groups()) if m else None
