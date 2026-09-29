"""Which of the packages' requests a silicon family does not answer — asked by the build before it compiles
the family, so a family with a gap is skipped and said so rather than failing as a compile error.

    python3 scripts/silicon_family_unanswered.py <family> [src]

Prints one request per line and exits 1 when there are any; prints nothing and exits 0 when the family
answers every package. What a family owes is read by the same function the claim gate's
`silicon_seam_conformance` uses, so the build and the gate cannot disagree about it."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'claim_rules'))
import src_claim_gate                       # noqa: E402  (the tree, read as the gate reads it)
import silicon_seam_conformance as seam     # noqa: E402  (what a family owes)

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)
family = sys.argv[1]
files = src_claim_gate.load(sys.argv[2]) if len(sys.argv) > 2 else src_claim_gate.load()
if not any(p.startswith('silicon_families/%s/' % family) for p in files):
    print('no folder silicon_families/%s/' % family)
    sys.exit(2)
gaps = seam.unanswered(files, family)
for g in gaps:
    print(g)
sys.exit(1 if gaps else 0)
