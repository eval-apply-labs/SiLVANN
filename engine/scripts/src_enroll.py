"""Enrol new packages: a folder that is not on the package list yet is appended to it, with the next id.
Nothing on the list is ever renumbered or removed.

    python3 scripts/src_enroll.py            append what is new, and say so
    python3 scripts/src_enroll.py --check    change nothing; fail if a folder is not enrolled

⚖ *"if the id needs to be declared and cannot be specified then we shouldnt proceed … if it can be incremental
instead it is worth doing"*. An id is permanent — it is the top byte of every kind the package registers, and
kinds are written into saved state — so it cannot come from anything recomputed per build, like folder order. It comes from the list, which
is committed and which this tool only ever APPENDS to. Nobody writes an id, and nobody names a package
outside its own folder.

    a package   a folder of src/packages/ holding a manifest.cuh — row X(<n>, <folder>, "<folder>") in
                PACKAGE_LIST (packages/manifest__header.cuh)

⛔ NOT THE SILICON FAMILIES. ⚖ *"in the roster it has to be hand tended because we need the ordering to
   establish recency"*: a family's number is also its PREFERENCE — a card goes to the highest that works —
   so where a family goes on the roster is a decision, and it is made by hand.

⛔ A LISTED FOLDER THAT IS GONE STOPS THIS, by name: its id must never be reused, and a list with a hole is a
   decision — retiring a package is done by hand.
⛳ TWO BRANCHES THAT EACH ADD ONE both take the next number, and git shows that as a conflict on the same line
   of the list — never as two things sharing a number silently.
⛳ THE ORDER IS THE ID: packages are included in id order, so one added later comes after those it builds
   on."""
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if '--root' in sys.argv:                       # a copy of the tree, for trying it without touching this one
    ROOT = sys.argv[sys.argv.index('--root') + 1]
PACKAGES = os.path.join(ROOT, 'packages')
PACKAGE_LIST_FILE = os.path.join(PACKAGES, 'manifest__header.cuh')
IDENT = re.compile(r'^[a-z][a-z0-9_]*$')


def folders(root, marker):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and os.path.isfile(os.path.join(root, d, marker)))


def refuse(msg):
    print('⛔ ' + msg)
    sys.exit(1)


def packages(check):
    text = open(PACKAGE_LIST_FILE).read()
    m = re.search(r'#define PACKAGE_LIST\(X\) \\\n((?:[ \t]*X\([^\n]*\)[ \t]*\\?\n)+)', text)
    if not m:
        refuse('no PACKAGE_LIST block in %s' % PACKAGE_LIST_FILE)
    rows = [(int(i), n, p) for i, n, p in re.findall(r'X\((\d+),\s*(\w+),\s*"([^"]+)"\)', m.group(1))]
    listed = {p for _, _, p in rows}
    names = {n for _, n, _ in rows}
    present = folders(PACKAGES, 'manifest.cuh')
    gone = sorted(listed - set(present))
    if gone:
        refuse('package folder(s) %s are on the list and gone. A number is never reused: retire a package '
               'by hand.' % ', '.join(gone))
    # ⛳ A FOLDER WHOSE NAME IS ALREADY A ROW'S is that package's other version — a row whose path points at
    #   `nn_v2` leaves `nn/` on disk — and is not a new package.
    new = [f for f in present if f not in listed and f not in names]
    for f in new:
        if not IDENT.match(f):
            refuse('package folder "%s" is not a name the language can paste' % f)
    if not new:
        return []
    if check:
        return new
    nxt = max(i for i, _, _ in rows) + 1
    rows += [(nxt + k, f, f) for k, f in enumerate(new)]
    block = ''.join('    X(%d, %s, "%s")%s\n' % (i, n, p, ' \\' if k < len(rows) - 1 else '')
                    for k, (i, n, p) in enumerate(rows))
    open(PACKAGE_LIST_FILE, 'w').write(text[:m.start(1)] + block + text[m.end(1):])
    return new


check = '--check' in sys.argv[1:]
new_packages = packages(check)
if check:
    if new_packages:
        refuse('not enrolled: %s. Run scripts/src_enroll.py (every build does).'
               % ', '.join('package ' + p for p in new_packages))
    print('✅ ENROLLED — every package folder is on the list.')
else:
    for p in new_packages:
        print('⭐ ENROLLED package %s' % p)
