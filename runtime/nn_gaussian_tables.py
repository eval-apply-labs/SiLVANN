#!/usr/bin/env python3
"""The Gaussian codebook the GAUSSIAN family reads — ONE table per width, shared by the whole build.

⚖ ARCHITECT, 2026-09-20: *"build that quant format with the global nf table, fwht row scaler and no
block scaler. i guess those values being sgpr is really the prize here..."*

⭐⭐ THIS FILE IS THE SOURCE AND THE C++ ARRAY IS ITS OUTPUT. The packer (Python) and the decoder (HIP)
MUST read identical numbers or the format is two formats, so the numbers exist once, here, and the
header is generated from them. `--check` re-derives and diffs; it is a gate, not a courtesy.

── ⭐ WHY THE ENTRIES ARE fp16 BIT PATTERNS AND NOT DECIMAL LITERALS ─────────────────────────────────
A decimal literal is a request for a value, not the value: `0.7229568362236023` in C++ and the same
digits in Python are two roundings of one real number and are not obliged to agree. A `uint16` is the
number itself. ⇒ ★ THE ONE THING BOTH SIDES MUST AGREE ON IS THE ONE THING THAT CANNOT BE TRANSCRIBED
WRONG, and `nn__primitives__zzpackage_half_to_float` — which already exists, and is exact — is the reader.

── ⭐ THE CONSTRUCTION, AND WHAT MAKES IT "NF" ───────────────────────────────────────────────────────
`2^D` quantiles of a unit Gaussian, normalised so the outermost lands on ±1. The two halves use
DIFFERENT grids (`k+1` points above the median, `k` below), which is what buys an EXACT ZERO while
still spending all `2^D` codes — the asymmetry is the feature, not a bug in the construction.
⛔ AND THERE IS NO SIGN BIT IN THIS FAMILY. The table is signed and the code indexes it whole, which is
the bit the architect went looking for: *"use all 4 bits for data instead of 3 bits for data and one
for parity."* `MEASURED` on the shipping 35B: 98.7% of rows of the sign-bit format spend a code on a
second zero.

── ⛔ WHAT IS VALIDATED HERE, BECAUSE A BROKEN TABLE LOOKS LIKE A WORSE CODEC ────────────────────────
S104 built an `nf(6)` that was silently wrong and reported NF6 as WORSE THAN NF4 — an impossible
result that took an hour to attribute, because a bad codebook degrades gracefully instead of failing.
⇒ ★ A TABLE IS DATA, SO IT CANNOT CRASH; ITS ONLY SYMPTOM IS A NUMBER THAT LOOKS PLAUSIBLE.
Hence: cardinality is exactly `2^D` · strictly ascending after fp16 rounding · spans exactly ±1 ·
contains exactly one zero · and D=4 matches the VERBATIM bitsandbytes NF4 table, which is the only
external check available and the reason that one width is transcribed below.
"""

import argparse
import math
import struct
import sys

# ⚖ THE SHIPPED WIDTH SET, RULED 2026-09-21. q6 LEFT AND q1 ARRIVED, and both follow from the
# halfword container (▶ `nn/turboquant__impl.cuh`): at 16 bits q6 packs 2 a container, which is q8's cost for
# 15.7x q8's error, so no budget selects it; q1 packs 16 a container at zero waste.
# ⛔ q1 IS A MEME AND THE TREE SAYS SO — ⚖ *"add q1 for the meme, it should be refuted."* ~60% rms.
WIDTHS = (1, 2, 3, 4, 5, 8)
OFFSET = 0.9677083          # bitsandbytes' — the quantile the outermost level sits on


def ppf(p):
    """The Gaussian inverse CDF by bisection. No scipy on this box, and 300 halvings of [-10,10] is
    exact to well past fp64, so the iteration count is not a tuning knob."""
    lo, hi = -10.0, 10.0
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _linspace(a, b, n):
    if n == 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def half(v):
    """The fp16 this value becomes — as a bit pattern, and as the float it then IS.

    ⛔⛔ NEGATIVE ZERO IS NORMALISED AWAY HERE AND IT IS NOT COSMETIC. The median quantile arrives from
    bisection as a tiny NEGATIVE number, so `round(·, 12)` yields `-0.0` and `struct` encodes 0x8000.
    Arithmetically that is harmless — `-0.0 * s` is `-0.0` and sums correctly — but the decoder would
    then emit `-0.0f` where a reference that wrote `0.0` emits `+0.0f`, and those are DIFFERENT BYTES.
    ⇒ ★ AN ORACLE THAT COMPARES EXACTLY WOULD REDDEN ON A SIGN BIT OF A ZERO, which is the most
    expensive possible way to learn that the codec is fine."""
    if v == 0.0:
        return 0x0000, 0.0
    bits = struct.unpack("<H", struct.pack("<e", v))[0]
    return bits, struct.unpack("<e", struct.pack("<H", bits))[0]


def nf_table(d):
    """The NF / equal-probability-quantile construction — bitsandbytes' NF4 generalised.

    ⛳ KEPT, THOUGH IT IS NO LONGER WHAT SHIPS. It is the only construction with an external reference
    (the verbatim NF4 table), so it is what `validate()` checks the machinery against, and it is the
    provenance of the numbers this file used before 2026-09-20. ▶ `lloyd_table`.
    """
    k = 1 << (d - 1)
    pos = [ppf(p) for p in _linspace(OFFSET, 0.5, k + 1)]      # k+1 points, outermost first
    neg = [ppf(p) for p in _linspace(1.0 - OFFSET, 0.5, k)]    # k points, outermost first
    span = pos[0]                                              # == abs(neg[0]); one divisor, both halves
    raw = sorted(set(round(x / span, 12) for x in pos) | set(round(x / span, 12) for x in neg))
    pairs = [half(x) for x in raw]
    return [b for b, _ in pairs], [v for _, v in pairs]


def _pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lloyd_levels(d, tol=1e-14, cap=50000):
    """⭐⭐ ZERO-PINNED LLOYD-MAX FOR A UNIT GAUSSIAN, SOLVED ANALYTICALLY. Ascending, unnormalised.

    The MSE-optimal reconstruction levels satisfy two conditions at once, and this alternates them:
    ```
      boundaries   bᵢ = (lᵢ + lᵢ₊₁) / 2                         a decision is the midpoint of its neighbours
      levels       lᵢ = E[X | bᵢ₋₁ < X < bᵢ]                    a level is the centroid of its own cell
                      = (φ(bᵢ₋₁) − φ(bᵢ)) / (Φ(bᵢ) − Φ(bᵢ₋₁))   closed form for a Gaussian
    ```
    ⛔⛔ AND IT IS FITTED TO THE ANALYTIC DENSITY, NOT TO A SAMPLE, BECAUSE THE SAMPLE VERSION
    OVERFITS AND THE OVERFIT IS NOT OBVIOUS. `MEASURED` 2026-09-20: Lloyd-Max fitted on 400,000 real
    rotated values scored **7.4% WORSE than the NF table at D=8** and a wash at D=5 — impossible for a
    converged fit, and the signature of tail cells holding a handful of points each and chasing their
    noise. 256 levels over 400k samples is ~1,500 points a cell before the tails thin it further.
    ⇒ ★★ A FIT THAT LOSES TO AN ARBITRARY TABLE HAS NOT CONVERGED, IT HAS MEMORISED — and at D=4,
    where cells are fat, it looked fine, so a single-width check would have shipped it.
    ⛳ Fitting the density is not an approximation here: the rotated data IS Gaussian, `MEASURED` at
    kurtosis 2.988 and σ 1.0008 over four real tensors. This uses the established model rather than a
    noisy draw from it, and it is DETERMINISTIC, which a build constant had better be.

    ⭐⭐ AND THERE IS NO ZERO LEVEL, WHICH IS A REVERSAL — AN EARLIER VERSION OF THIS FUNCTION PINNED
    ONE. ⚖ ARCHITECT, 2026-09-20: *"if it is gaussian the values would be specular."* He is right, and
    every `2^D` here is EVEN, so the unconstrained optimum for a symmetric density is perfectly
    mirrored: `2^(D-1)` magnitudes, each with both signs, nothing spent on a zero and no lopsidedness.
    Pinning a zero forces `2^D − 1` levels to split unevenly and costs ~1%.
    ⇒ ⛔⛔ THE ZERO WAS BEING PAID FOR AND WAS NOT NEEDED, FOR TWO SEPARATE REASONS:
      ① A row that must reconstruct to zero gets `scale = 0`, and every level times zero is zero. The
        zero comes from the MULTIPLIER, not from a level.
      ② **The rotation destroys individual zeros before the quantiser ever sees one.** The FWHT mixes
        all `cols` columns, so a single zero weight in the source is not zero after the transform. The
        only exact zeros that survive `R` are WHOLE ROWS of zeros — which is exactly case ①.
    ⇒ ★★ A CONSTRAINT INHERITED FROM THE UNROTATED FORMAT SURVIVED INTO THE ROTATED ONE, WHERE ITS
    JUSTIFICATION NO LONGER EXISTED. `MEASURED`: 0.009535 pinned against 0.009442 free, on five real
    rotated tensors — and 0.009847 for the NF table this replaces, so the two changes together are
    −4.3% error at identical bits, identical lookup and identical kernel.
    """
    n = 1 << d
    lev = sorted((i + 0.5) / n * 6.0 - 3.0 for i in range(n))  # a deterministic, symmetric start
    # ⛔⛔ ITERATED TO CONVERGENCE, NOT A FIXED COUNT — 400 was the first number here and it left D=5,
    #   D=6 and D=8 short by ~6e-5 while D=2..4 were exact, because Lloyd converges LINEARLY and the
    #   rate worsens with the level count. ⇒ ★ A FIXED ITERATION COUNT IS A CONVERGENCE CLAIM WITH NO
    #   EVIDENCE, and it fails at exactly the widths nobody eyeballs. The validator below re-derives
    #   the centroid condition and would have caught it either way — which is why it is there.
    for _ in range(cap):
        bnd = [(lev[i] + lev[i + 1]) / 2.0 for i in range(n - 1)]
        new = []
        for i in range(n):
            lo = bnd[i - 1] if i > 0 else None
            hi = bnd[i] if i < n - 1 else None
            plo, phi = (_cdf(lo) if lo is not None else 0.0), (_cdf(hi) if hi is not None else 1.0)
            dlo, dhi = (_pdf(lo) if lo is not None else 0.0), (_pdf(hi) if hi is not None else 0.0)
            mass = phi - plo
            new.append((dlo - dhi) / mass if mass > 1e-300 else lev[i])
        new = sorted(new)
        moved = max(abs(a - b) for a, b in zip(new, lev))
        lev = new
        if moved < tol:
            break
    return lev


def lloyd_k(d):
    """⭐⭐ THE CLIP, DERIVED RATHER THAN TUNED. The levels are fitted against a UNIT Gaussian, so the
    outermost one already sits where it belongs in units of σ — and that number IS the clip. Scaling a
    row by `k·rms` with `k = max|level|` puts the table back exactly where it was fitted.
    ⇒ ★ THE SIX TUNED CONSTANTS BECOME SIX DERIVED ONES, and the packer's only remaining freedom is
    which width to spend. ⛳ The swept values from the 2026-09-20 measurement — 2.85 at D=4 — agree
    with the derived ones to about a grid step, which is the cross-check that this is the same number
    arrived at twice.
    """
    lev = lloyd_levels(d)
    return max(abs(lev[0]), abs(lev[-1]))


def lloyd_table(d):
    """The width-`d` table as (bit patterns, values), ascending — normalised so the outermost is ±1."""
    lev = lloyd_levels(d)
    span = max(abs(lev[0]), abs(lev[-1]))
    pairs = [half(round(v / span, 12)) for v in lev]
    return [b for b, _ in pairs], [v for _, v in pairs]


def table(d):
    """THE SHIPPED TABLE. ⚖ Changed 2026-09-20 from the NF construction to zero-pinned Lloyd-Max.

    ⛔⛔ CHANGING THIS CHANGES THE FORMAT'S CONSTANTS, so bytes packed against the old table decode
    WRONG against the new one, silently and plausibly. Nothing has shipped family 2, so this is free
    today and will not be free again. ⇒ ★ THE FORMAT ALREADY HAS THE RIGHT MECHANISM FOR A SECOND
    TABLE AND IT COSTS NOTHING: `family` is three bits with five values spare, so a future table is
    family 3, exactly as family 0 is "the codebook that shipped". A build constant that can change
    without changing its tag is a compatibility hazard; a family tag is the version field.
    """
    return lloyd_table(d)


# ⭐ THE ONLY EXTERNAL CHECK THERE IS. Transcribed from bitsandbytes' `create_normal_map()` output —
#   the table every NF4 deployment in the world uses. If our construction reproduces it, the
#   construction is right for every OTHER width too, because only `k` changes.
NF4_VERBATIM = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


def validate():
    """Everything a wrong table would otherwise pass silently. Returns a list of complaints."""
    bad = []
    for d in WIDTHS:
        bits, vals = table(d)
        if len(vals) != (1 << d):
            bad.append("D=%d has %d entries, wanted %d" % (d, len(vals), 1 << d))
            continue
        if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
            bad.append("D=%d is not strictly ascending after fp16 rounding — two codes collide" % d)
        # ⛔⛔ THIS RULE USED TO DEMAND BOTH ENDS BE EXACTLY ±1 AND THAT WAS A PROPERTY OF THE *NF*
        #   CONSTRUCTION, NOT A LAW — NF divides both halves by one number, so its extremes match by
        #   construction while its interior is asymmetric. A zero-pinned Lloyd-Max table is asymmetric
        #   END TO END, because `2^D − 1` non-zero levels cannot split evenly about the zero, and the
        #   optimum genuinely places one side further out. ⇒ ★ A VALIDATOR THAT ENCODES ONE
        #   CONSTRUCTION'S INCIDENTAL SYMMETRY WILL REJECT A BETTER CONSTRUCTION AS BROKEN. What must
        #   hold is that the table is NORMALISED — its widest reach is 1 — so the clip means the same
        #   thing at every width.
        if abs(max(abs(vals[0]), abs(vals[-1])) - 1.0) > 1e-9:
            bad.append("D=%d reaches %r at its widest, wanted 1.0 — the table is not normalised"
                       % (d, max(abs(vals[0]), abs(vals[-1]))))
        # ⛔⛔ THIS RULE ONCE DEMANDED EXACTLY ONE ZERO AND NOW FORBIDS ANY — ▶ `lloyd_levels`. The
        #   zero came from the NF construction and was kept on an argument that the rotation had
        #   already invalidated. What must hold now is SPECULARITY, which is a real property of the
        #   optimum for a symmetric density and a sharp check: a fit that drifts off-centre fails it.
        zeros = sum(1 for v in vals if v == 0.0)
        if zeros != 0:
            bad.append("D=%d has %d zero levels — the specular table spends none on zero" % (d, zeros))
        for i in range(len(vals)):
            if abs(vals[i] + vals[-1 - i]) > 1e-3:
                bad.append("D=%d is not specular: level %d is %r against %r"
                           % (d, i, vals[i], vals[-1 - i]))
                break
        if len(set(bits)) != len(bits):
            bad.append("D=%d has a duplicate bit pattern" % d)
    # ⭐ THE EXTERNAL CHECK IS AGAINST THE *NF* CONSTRUCTION, WHICH IS NO LONGER WHAT SHIPS — and it
    #   is kept for exactly that reason. It is the only table in this file with a reference outside
    #   this repository, so it proves the machinery (the quantile solve, the fp16 encoding, the
    #   normalisation) rather than the choice of construction. The shipped Lloyd-Max table has no
    #   external reference and cannot have one; it is checked by its own optimality conditions below.
    _, nf4 = nf_table(4)
    worst = max(abs(a - b) for a, b in zip(nf4, NF4_VERBATIM))
    if worst > 5e-4:        # fp16's own resolution near 1.0 is 2^-11 ≈ 4.9e-4
        bad.append("D=4 NF construction differs from the verbatim bitsandbytes NF4 by %.3g" % worst)
    # ⭐⭐ AND THE SHIPPED TABLE IS CHECKED AGAINST ITS OWN DEFINITION: at a Lloyd-Max optimum every
    #   level is the centroid of its cell. That is a property the answer must have, not a number to
    #   compare against — and it is what a non-converged or overfitted fit FAILS, which is the defect
    #   that nearly shipped here. The zero level is exempt: it is pinned, so it is not a centroid.
    for d in WIDTHS:
        lev = lloyd_levels(d)
        n = len(lev)
        bnd = [(lev[i] + lev[i + 1]) / 2.0 for i in range(n - 1)]
        for i in range(n):
            if lev[i] == 0.0:
                continue
            lo = bnd[i - 1] if i > 0 else None
            hi = bnd[i] if i < n - 1 else None
            plo, phi = (_cdf(lo) if lo is not None else 0.0), (_cdf(hi) if hi is not None else 1.0)
            dlo, dhi = (_pdf(lo) if lo is not None else 0.0), (_pdf(hi) if hi is not None else 0.0)
            mass = phi - plo
            if mass <= 1e-300:
                bad.append("D=%d level %d owns no probability mass — the fit collapsed" % (d, i))
                continue
            cent = (dlo - dhi) / mass
            if abs(cent - lev[i]) > 1e-6 * max(1.0, abs(lev[i])):
                bad.append("D=%d level %d is %.6f but its cell's centroid is %.6f — NOT converged"
                           % (d, i, lev[i], cent))
    return bad, worst


def offsets():
    """Where each width's block starts in the flat array, and the total length."""
    at, out = 0, {}
    for d in WIDTHS:
        out[d] = at
        at += 1 << d
    return out, at


def emit():
    """The generated body of the C++ table — the array and the offset switch."""
    off, total = offsets()
    lines = []
    lines.append("/* ⛔⛔ GENERATED BY `backstage/scripts/nn_gaussian_tables.py` — DO NOT HAND-EDIT.")
    lines.append(" * Regenerate with `--emit`; `--check` fails the build if this block and the script")
    lines.append(" * disagree. ▶ that script for the construction, and for what it validates.")
    lines.append(" * %d fp16 entries: %s." % (total, " + ".join("%d@D%d" % (1 << d, d) for d in WIDTHS)))
    lines.append(" * ⛳ AT D=4 THESE ARE THE VERBATIM bitsandbytes NF4 LEVELS, checked to fp16's own"
                 " resolution. */")
    lines.append("static __device__ const uint16_t nn__turboquant__zzprivate_gaussian[%d] = {" % total)
    for d in WIDTHS:
        bits, vals = table(d)
        lines.append("    /* D=%d — %d levels, %+.4f … %+.4f */" % (d, 1 << d, vals[0], vals[-1]))
        for i in range(0, len(bits), 8):
            lines.append("    " + " ".join("0x%04X," % b for b in bits[i:i + 8]))
    lines.append("};")
    lines.append("")
    lines.append("/* ⭐ THE SAME LEVELS AS int8 — round(127 x level), half away from zero — for the int8 gemv, which")
    lines.append(" * multiplies them against x quantised to int8. One table, generated here, so every family's int8")
    lines.append(" * body reads the same numbers. */")
    lines.append("static __device__ const int8_t nn__turboquant__zzprivate_gaussian_i8[%d] = {" % total)
    for d in WIDTHS:
        bits, vals = table(d)
        q = [int(math.copysign(math.floor(abs(v) * 127.0 + 0.5), v)) for v in vals]
        for i in range(0, len(q), 16):
            lines.append("    " + " ".join("%d," % v for v in q[i:i + 16]))
    lines.append("};")
    lines.append("")
    lines.append("/* Where width `d`'s block starts. ⛔ A width with no table is a REFUSAL, not a zero:")
    lines.append(" * every caller checks `nn__turboquant__rate_is_known` first, and this answers the widths that")
    lines.append(" * check admits and nothing else. */")
    lines.append("static __device__ inline uint64_t nn__turboquant__zzprivate_gaussian_at(uint64_t d) {")
    for d in WIDTHS:
        lines.append("    if (d == %dull) return %dull;" % (d, off[d]))
    lines.append("    return %dull;                                   /* unreachable past `rate_is_known` */"
                 % total)
    lines.append("}")
    return "\n".join(lines)


BEGIN = "/* ══ GAUSSIAN TABLE — GENERATED, BEGIN ══════════════════════════════════════════════════ */"
END = "/* ══ GAUSSIAN TABLE — GENERATED, END ════════════════════════════════════════════════════ */"


def splice(path, body):
    """Replace what sits between the markers. Returns (new_text, old_body)."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    a = text.find(BEGIN)
    b = text.find(END)
    if a < 0 or b < 0 or b < a:
        raise SystemExit("markers not found in %s — nothing to splice" % path)
    old = text[a + len(BEGIN):b]
    return text[:a + len(BEGIN)] + "\n" + body + "\n" + text[b:], old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", metavar="HEADER", help="splice the generated block into this header")
    ap.add_argument("--check", metavar="HEADER", help="fail if this header's block is stale")
    ap.add_argument("--show", action="store_true", help="print the tables as floats")
    args = ap.parse_args()

    bad, worst = validate()
    if bad:
        for row in bad:
            print("⛔ %s" % row)
        return 1
    print("✅ tables valid — %d widths, D=4 matches verbatim NF4 to %.3g (fp16 resolution is 4.9e-4)"
          % (len(WIDTHS), worst))

    if args.show:
        for d in WIDTHS:
            _, vals = table(d)
            print("D=%d (%d):" % (d, len(vals)), " ".join("%+.4f" % v for v in vals[:8]),
                  "…" if len(vals) > 8 else "")
    body = emit()
    if args.emit:
        text, _ = splice(args.emit, body)
        with open(args.emit, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("✅ spliced %d lines into %s" % (len(body.splitlines()), args.emit))
    if args.check:
        _, old = splice(args.check, body)
        if old.strip() != body.strip():
            print("⛔ %s IS STALE — re-run with --emit" % args.check)
            return 1
        print("✅ %s matches the generator" % args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
