#!/usr/bin/env python3
"""The reference packer for the GAUSSIAN family — real weights in, the five VBR planes out.

⚖ ARCHITECT, 2026-09-20: *"build that quant format with the global nf table, fwht row scaler and no
block scaler."*

⛔⛔ THIS IS A REFERENCE PACKER, NOT THE SHIPPING ONE. `backstage/python/compressor.py` is what packs
a release; this file exists so the FORMAT has a producer that can be read against the decoder in
`src/packages/nn/vbr__impl.cuh`, and so the oracle has something to compare. Folding the family into
the shipping compressor is a separate change with a separate review.

⛔⛔ AND NOTHING HERE IS TESTED ON SYNTHETIC DATA. ⚖ ARCHITECT: *"never test compression on synthetic
data, use the 35b model directly as data source."* Every number this file's callers quote comes from
`/mnt/data/bigdisk/qwen_36_35_a3b`, read as raw bf16.

── ⭐⭐ THE THREE PIECES, AND ONLY THE MIDDLE ONE IS NEW ──────────────────────────────────────────────
```
  1  ROTATE   w' = H·(S⊙w)      the FWHT with a random sign flip, on the CONTRACTION axis
  2  SCALE    s  = k · rms(w')  one fp16 a row — and `k` is DERIVED from the table, not searched
  3  ASSIGN   code = argmin |w'/s − table[·]|     nearest level, whole code, no sign bit
```
⛳ Step 1 is what makes step 3 legal: a fixed table is only correct if the row is Gaussian, and a row
of weights is not — the FWHT makes it one by the central limit theorem.

── ⭐⭐⭐ WHY `s` IS `k · rms` AND NOT `max|row|` — `MEASURED`, AND IT REFUTED BOTH OF THE OBVIOUS GUESSES
Nine real 35B tensors, D=4, rotated, energy `Σδ²/Σw²`:
```
  scale = max|row|         0.01130    the obvious choice, and it wastes the top of the table
  scale = 1.849 · rms      0.02357    ⛔ TWICE AS BAD — see below, this one was MY guess and it is wrong
  scale = k · rms, k swept 0.00987    ⭐ 12.7% better than max, and it costs NOTHING: same one fp16
```
⛔ THE FAILED GUESS IS WORTH KEEPING BECAUSE IT IS SEDUCTIVE. The table's outermost level sits on the
96.77% quantile of a unit Gaussian, which is 1.849σ — so it *looks* as though scaling the row to put
its 96.77th percentile at 1.0 is what the table was designed for. It is not: that argument is about
where the LEVELS go, and the scale also decides where the CLIP goes. Clipping at 1.849σ saturates 6.4%
of the row, and those are the largest values, which are exactly the ones L2 is made of.
⇒ ★★ A TABLE'S DESIGN QUANTILE AND ITS OPTIMAL CLIP ARE DIFFERENT NUMBERS, and reasoning from the
first to the second produced a configuration twice as bad as the naive one.

⭐ AND `k` IS A CONSTANT BECAUSE THE ROTATION MADE IT ONE. After the FWHT every row is Gaussian, so
the best clip measured in units of the row's own σ is a property of the TABLE and not of the tensor —
which is why the packer does no search at all. It is now DERIVED from the levels rather than swept;
▶ `clip_for_width`.

── ⭐⭐ THE LEVELS ARE SPECULAR AND THERE IS NO ZERO — ⚖ RULED 2026-09-20 ────────────────────────────
⚖ ARCHITECT: *"if it is gaussian the values would be specular"*, and: *"yes land it."* The table is
`2^(D-1)` magnitudes each with both signs, MSE-optimal (Lloyd-Max) rather than equal-probability
quantiles. `MEASURED` on five real rotated tensors, D=4, identical bits and identical lookup:
```
  NF, one zero, lopsided 7/8    0.009847     what this file used to produce
  Lloyd-Max, zero pinned        0.009535
  Lloyd-Max, SPECULAR, no zero  0.009442     −4.3%, and the zero was never needed
```
⛳ THE ZERO COMES FROM THE MULTIPLIER: a row that must be zero gets `scale = 0`. And the rotation
destroys individual zeros before the quantiser sees one — only WHOLE zero rows survive `R`, which is
exactly the scale=0 case. ▶ `nn_gaussian_tables.lloyd_levels` for both arguments.
"""

import math
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from nn_gaussian_tables import table, lloyd_k, WIDTHS  # noqa: E402  (one source for the levels)

FP16_MIN_NORMAL = 6.103515625e-05     # 2^-14 — below this a stored fp16 scale is subnormal or zero

# ⭐⭐⭐ THE CLIP IS DERIVED, NOT TUNED — AND IT USED TO BE SIX HAND-SWEPT NUMBERS. The levels are
#   fitted against a UNIT Gaussian, so the outermost one already sits where it belongs in units of σ,
#   and THAT NUMBER IS THE CLIP: scaling a row by `k·rms` with `k = max|level|` puts the table back
#   exactly where it was fitted. ⇒ ★ SIX TUNED CONSTANTS BECAME ZERO, and the packer's only remaining
#   freedom is which width to spend.
#   ⛳ AND IT CROSS-CHECKS. Swept empirically on five real rotated tensors the D=4 optimum is 2.70;
#   derived from the fit it is 2.733 — 1.2% apart, on an optimum that is flat. Two routes, one number.
#   ⛔ The earlier hand-swept table {2:1.35, 3:2.20, 4:2.85, 5:3.25, 6:3.55, 8:4.15} was swept against
#   the NF levels, which are no longer the levels. A tuned constant outlives the thing it was tuned
#   for, silently — which is the argument for deriving it rather than re-sweeping it.
_K_CACHE = {}


def clip_for_width(d):
    """`k` such that `scale = k · rms(row)` lands the table where it was fitted."""
    if d not in _K_CACHE:
        _K_CACHE[d] = lloyd_k(d)
    return _K_CACHE[d]


class _KForWidth(dict):
    """Reads like the old constant, answers from the fit. Kept dict-shaped so callers need no edit.

    ⛔ `.get(d)` ON THIS RETURNS `None`, because `dict.get` bypasses `__missing__`. Index it, or call
    `clip_for_width`. ▶ `pack_row`, which was written with `.get` and refused every width."""
    def __missing__(self, d):
        v = clip_for_width(d)
        self[d] = v
        return v


K_FOR_WIDTH = _KForWidth()


def splitmix64(x):
    """One step of splitmix64 — stateless, indexable, and trivially portable to C.

    ⭐⭐ THE INDEXABILITY IS THE POINT, NOT THE STATISTICS. A sequential PRNG makes `S[i]` depend on
    `S[i-1]`, so a device that wants the sign vector must generate it serially or store it. This form
    takes the INDEX and answers, so every lane computes its own sign from its own `i` with no state
    and no ordering. ⇒ ★ A RANDOM VECTOR A PARALLEL MACHINE CAN INDEX IS A DIFFERENT OBJECT FROM ONE
    IT CAN ONLY REPLAY.
    ⛔ ASSUMED: that splitmix64's low bit is well-balanced enough to serve as a random sign. What would
    falsify it: a measurable bias between this vector and `numpy.default_rng`. What depends on it: the
    rotation's flattening, and therefore the whole family's error. `--calibrate` prints both.
    """
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def sign_vector(n, seed=0):
    """The ±1 vector the rotation flips by, as a list of floats.

    ⚖⚖ WHERE THIS VECTOR LIVES IN A SHIPPED FILE IS A RULING NOBODY HAS MADE, and this function does
    not make it. It offers a SEED, which is the cheapest of the three candidates (a seed costs no
    bytes; a plane costs `cols` bits a tensor; a build constant costs nothing but cannot vary). The
    packer and the runtime must agree on the vector, not on how it was obtained — so a later ruling
    for "a stored plane" replaces this call and changes nothing else.
    """
    return [1.0 if (splitmix64(seed + i) >> 63) == 0 else -1.0 for i in range(n)]


def fwht_inplace(v):
    """The normalised Fast Walsh-Hadamard Transform, in place. `len(v)` must be a power of two.

    ⭐ H IS ITS OWN INVERSE at this normalisation, and so is the sign flip — which is why the codec
    needs no second transform anywhere. ▶ `rotate`.
    """
    n = len(v)
    if n & (n - 1):
        raise ValueError("FWHT needs a power-of-two length, got %d" % n)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = v[j], v[j + h]
                v[j], v[j + h] = a + b, a - b
        h *= 2
    inv = 1.0 / math.sqrt(n)
    for i in range(n):
        v[i] *= inv
    return v


def rotate(v, signs):
    """`R(v) = H·(S⊙v)` — the transform BOTH sides apply, and they apply the SAME one.

    ⭐⭐ WHY IT WORKS, AND IT IS ONE LINE:
    ```
      RᵀR = (HS)ᵀ(HS) = SᵀHᵀHS = S·I·S = S² = I      ⇒  R IS ORTHOGONAL
      w'·x' = (Rw)ᵀ(Rx) = wᵀRᵀRx = wᵀx               ⇒  the product is unchanged
    ```
    ⇒ ★ THE WEIGHT ROWS AND THE ACTIVATION GET THE IDENTICAL TRANSFORM, and there is NO transform on
    the output — the rotation lives entirely on the contraction axis. ⛳ That is what makes it cheap:
    ONE rotation of length `cols` per matvec, shared by every one of the `rows` dot products.

    ⛔⛔ `R` IS NOT ITS OWN INVERSE, though `H` and `S` each are: `R∘R = H∘S∘H∘S` and the two do not
    commute. An earlier version of this docstring argued the conclusion above FROM involution, reached
    the right answer, and was wrong; the device suite caught it at n=4. `unrotate` below is the real
    inverse — the same two operations in the other order. ⇒ ★★ A TRUE CONCLUSION BY A FALSE ROUTE
    LOOKS EXACTLY LIKE A TRUE ONE UNTIL SOMETHING ELSE LEANS ON THE ROUTE.
    """
    return fwht_inplace([a * b for a, b in zip(v, signs)])


def unrotate(v, signs):
    """`R⁻¹(v) = S⊙(Hv)`. Only the packer needs this — to check itself."""
    return [a * b for a, b in zip(fwht_inplace(list(v)), signs)]


def _half(x):
    """The fp16 this becomes, and the float it then is. Negative zero is normalised away — ▶ the
    tables module, which explains why that is not cosmetic."""
    if x == 0.0:
        return 0x0000, 0.0
    b = struct.unpack("<H", struct.pack("<e", x))[0]
    return b, struct.unpack("<e", struct.pack("<H", b))[0]


def pack_row(row, d, k=None):
    """One rotated row -> (codes, scale_bits, reconstruction). `row` is already rotated.

    ⛔ A ROW WHOSE RMS UNDERFLOWS fp16 IS NOT AN ERROR AND IS NOT A CRASH — it is a row of (almost)
    zeros, and the honest encoding of it is the table's exact zero level in every column. Silently
    dividing by it would produce inf/nan codes that `argmin` resolves to *something*, and that
    something decodes to a large number where the truth is zero. ⇒ ★ THE DEGENERATE CASE IS THE ONE
    WHERE A WRONG ANSWER LOOKS MOST LIKE A RIGHT ONE.
    """
    tbl = table(d)[1]
    n = len(row)
    rms = math.sqrt(sum(x * x for x in row) / n) if n else 0.0
    if k is None:
        # ⛔ `clip_for_width`, NOT `K_FOR_WIDTH.get(d)` — `dict.get` does NOT consult `__missing__`,
        #   only `__getitem__` does, so the lazy mapping answered `None` for every width and the
        #   packer refused D=2 with "run --calibrate". A lazy container that looks like a constant is
        #   only lazy through the one access path Python routes that way.
        k = clip_for_width(d)
    s_bits, s = _half(k * rms)
    if s < FP16_MIN_NORMAL:
        # ⭐⭐ A DEGENERATE ROW GETS `scale = 0`, AND THAT IS WHY THE TABLE NEEDS NO ZERO LEVEL. Every
        #   level times zero is zero, whatever code each column happens to carry, so the row
        #   reconstructs EXACTLY — which is the honest encoding of values at 1e-38.
        #   ⛔ AN EARLIER VERSION STORED THE SMALLEST NORMAL fp16 AND POINTED EVERY CODE AT A ZERO
        #   LEVEL. That worked, and it made the zero level look load-bearing when it was not: the
        #   zero was already available from the multiplier. ▶ `nn_gaussian_tables.lloyd_levels`.
        return [0] * n, 0x0000, [0.0] * n
    codes, recon = [], []
    for x in row:
        y = x / s
        best, bi = None, 0
        for i, t in enumerate(tbl):
            e = abs(y - t)
            if best is None or e < best:
                best, bi = e, i
        codes.append(bi)
        recon.append(tbl[bi] * s)
    return codes, s_bits, recon


# ⛔⛔⛔ RETIRED 2026-09-21 — ⚖ ARCHITECT: *"retire those two functions."*
#
# This file used to carry a whole-matrix packer and a per-row width chooser. Both are gone, and the
# reason is not that they were superseded — it is that each had become a producer of bytes nothing
# can read, which is the most dangerous state a reference implementation can be in.
#
#   THE WIDTH CHOOSER  bisected a per-row width against a RELATIVE error target. `σ/rms = f(D)`
#                      depends on D alone, so a relative target is met at the SAME D by every row:
#                      it was a fixed-D packer with extra steps, and its only varying output came
#                      from its degenerate-row branch. ⚖ *"the ladder can disappear."*
#   THE MATRIX PACKER  emitted five planes into 32-bit containers. The format is two planes in
#                      halfwords. ⇒ ★ A REFERENCE THAT STILL RUNS AND NO LONGER DESCRIBES THE FORMAT
#                      IS WORSE THAN ONE THAT IS DELETED: it answers, plausibly, and a reader has no
#                      signal that the answer is from a previous world.
#
# ⛳ WHAT SURVIVES HERE IS THE PART THAT IS STILL INDEPENDENT AND STILL TRUE — `rotate`, `pack_row`,
#   `sign_vector`, `clip_for_width` — none of which knows anything about containers or planes. That
#   is exactly the half `python/nn_compressor.py --selftest` checks itself against, CODE FOR CODE.
#   ⇒ ★ THE VALUABLE PART OF A SECOND IMPLEMENTATION IS THE PART THAT SHARES NO ASSUMPTIONS, and the
#   layout was never it.
#
# ▶ `python/nn_compressor.py` is the producer. `backstage/test/src_device_selftest.py` carries a
#   third implementation of the layout, written from the format, for what this file no longer does.


def bits_per_weight(d, cols):
    """What the format actually costs. ⛳ One fp16 a row, and nothing else — no block scales, no
    per-row table. The row-width amortisation that decides CODEBOOK-vs-UNIFORM barely moves here,
    because 16 bits over 2048 columns is 0.008 b/w."""
    return d + 16.0 / cols


if __name__ == "__main__":
    print(__doc__)
    print("clip constants currently set:", K_FOR_WIDTH or "(none — calibrate against the 35B)")
    for d in WIDTHS:
        print("  D=%d  %d levels  %.4f bits/weight at 2048 cols" % (d, 1 << d, bits_per_weight(d, 2048)))
