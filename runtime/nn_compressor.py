#!/usr/bin/env python3
"""nn_compressor.py — THE SHIPPING PACKER FOR THE GAUSSIAN FAMILY. One D per tensor, chosen by name.

⚖ ARCHITECT: *"compressor.py was the old method, use it as a sort of reference to see how
i did it but make a new one for this strategy, like we can specify different quants for different
matrices type. make the D per tensor something chosen at pack time, like default = 8 and then check
the name of the expert type and see if it is in a key value map for a specific quant compression
level."*

── ⭐ WHAT THIS CODEC IS CALLED IN THE LITERATURE, so nobody re-derives the vocabulary ───────────────
`R = H·S` is the **randomized Hadamard transform**; what it does to the rows is **incoherence
processing** (QuIP's term); the shared fixed table is a **NormalFloat** codebook and reproduces
bitsandbytes' **NF4** verbatim at D=4; and rotation-then-precomputed-scalar-Lloyd-Max-codebook with
no calibration is **TurboQuant's** method, with the dense random rotation replaced by the RHT exactly
as **QuIP#** does. Applying the same rotation to activations and the KV cache is **QuaRot**. Past a
scalar codebook lies vector quantisation (QuIP#'s E8 lattices), which is what §⑭ means when it says
the level placement is finished. ▶ `src/docs/design/pending/quant_families.md` §⓪.

⛔ THIS IS NOT `python/compressor.py`. That one packs FAMILY 0 (a per-row codebook) and is what
`src_old` reads — `silvann_core/memory.py` and `silvann_model/composer.py` decode its output and every
deployed bundle is in that format. It is untouched, and it stays the reference for how the planes are
laid out. This file produces FAMILY 2, which only `src/packages/nn/cpu/turboquant__impl.cuh` reads.

── ⭐⭐ WHAT IS PER TENSOR AND WHAT IS PER ROW, AND BOTH HALVES ARE `MEASURED` ───────────
```
  the WIDTH  D   PER TENSOR   a per-row ladder is worth <= 0.058 b/w on this model and usually 0.00
  the SCALE      PER ROW      dropping it costs +1% .. +3150% error, and the penalty GROWS with D
```
⭐ THE WIDTH: the MSE-optimal per-row allocation is reverse water-filling — `rms_r^2 * 4^-D_r = const`,
i.e. +1 bit per doubling of a row's rms — and its whole prize is the coding gain
`0.5*log2(AM(rms_r^2)/GM(rms_r^2))`. `MEASURED` over 16 real 35B tensors that is **0.000 .. 0.058
b/w**, because the rows inside a tensor are too alike (AM/GM 1.001..1.084, p1..p99 rms spread
1.08x..3.28x). ⛳ AND THE ROTATION IS NOT WHY: `R` is orthogonal along the contraction axis, so
`||Rw_r|| == ||w_r||` exactly and the row-to-row spread is untouched by it. **This is a property of
Qwen3.6-35B, not of the codec** — a model with lumpier row norms would make a ladder pay, so re-run
the census before assuming it transfers.
⭐ THE SCALE: it is NOT a bit-allocation device, it is what makes the fixed table LEGAL for that row.
The table is a COMPANDED quantiser fitted to a unit Gaussian; a wrong scale puts the row in the wrong
part of the compander, so the clip and the level spacing are both wrong at once — which is why one
tensor-wide scale degrades *worse* as D rises (`+1.3%` at D=4 on `L3 down`, `+3150%` at D=8 on
`L3 o_proj`). ⇒ ★★ AND IT IS WHAT BUYS THE ONE PROPERTY THE WHOLE FAMILY RESTS ON: with a per-row
scale the pooled energy is the SAME CONSTANT for every tensor in the model (D=4 -> 0.00942..0.00952
across experts, q_proj and o_proj alike), so `sigma` is a function of D and nothing else. That is what
makes "pick a D for this tensor" a meaningful act.
▶ `src/docs/design/pending/quant_families.md` · `measurements/2026-09-21_scale_and_width_granularity.md`

── ⛔ WHAT THIS FILE DOES NOT DO, NAMED SO NOTHING READS AS FINISHED ─────────────────────────────────
· IT DOES NOT CHOOSE THE LEVELS FOR YOU. `POLICY` below is a DEFAULT, and every entry in it is
  `ASSUMED` — no end-task run has scored any of them. What would falsify one: a perplexity or
  output-agreement run at that level against bf16. What depends on it: the whole pack's quality.
· IT DOES NOT ROTATE THE ACTIVATION. A family-2 row decodes correctly and still computes a WRONG
  product until the runtime applies the same `R` to the activation — `nn__hadamard__rotate` is verb
  15 and the wiring is not done.
· THE SIGN VECTOR IS GENERATED FROM A SEED HERE AND ALSO EMITTED AS A PLANE. ⚖ The plane is what was
  ruled (512 bytes for the whole model); the seed is kept so a reader can regenerate it. If the two
  ever disagree the PLANE is the file's truth.
"""

import json
import math
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import nn_gaussian_pack as REF            # noqa: E402  the reference producer — the oracle for this one
import nn_gaussian_tables as TBL          # noqa: E402

FAMILY_GAUSSIAN = 2
FP16_MIN_NORMAL = 6.103515625e-05
SIGN_MASTER_BITS = 4096               # ⚖ the ruled plane: 512 bytes for the whole model

# ══ ⭐⭐⭐ THE CONTAINER IS A HALFWORD, AND IT IS A KERNEL RULING, NOT A FORMAT ONE ═══════════════════
#
# ⚖ ARCHITECT: *"loading and doing bitmask and bitshift to n bytes to get 8 values is not
# worth the float hosting 3 or 5 values already and i pay a 9% packing fee to have it in a manageable
# and accessible state."* ⛳ `MEASURED` by him on a previous build: the densely-packed alternative
# *"was very slow"*.
#
# ⛔⛔ AND THE ARGUMENT THAT LOST IS WORTH KEEPING, BECAUSE IT WAS ARITHMETICALLY CORRECT. 8 codes is
#   exactly D bytes at every width, so a group of 8 wastes NOTHING — 3.000 b/w at D=3 against 3.200
#   here, 5.000 against 5.333. Every one of those numbers is right. They are also beside the point:
#   an unaligned group costs a dependent shift chain on the LOAD path, and the load path is what the
#   gemv is made of. ⇒ ★★ A FORMAT ARGUMENT THAT NEVER TOUCHES THE KERNEL IS A CLAIM ABOUT THE WRONG
#   QUANTITY, and being exactly right about bits is what makes it convincing.
#
# THE FEE, EXACTLY: one bit in sixteen, and only at the two widths that do not divide 16.
#   D  values/u16   b/w     waste
#   2      8        2.000    0%
#   3      5        3.200    6.25%     <- the fee
#   4      4        4.000    0%
#   5      3        5.333    6.25%     <- the fee
#   8      2        8.000    0%
#  16      1       16.000    0%        ⭐ the raw-fp16 escape stops being a special case: it is just
#                                         "one value a container", with no branch in `row_bytes`.
#
# ⛔⛔ AND q6 DIES OF IT — ⚖ *"i think i should go up to q5 and then move to q8."* Two values in a
#   halfword is what q6 and q8 BOTH get, so q6 costs exactly q8's bytes and carries 15.7x its error
#   (0.000642 against 0.000041, `MEASURED` on `L3 gate_up e0`). **Dominated — there is no budget at
#   which it wins.** ⛳ NOTE THE DIRECTION OF THE ARGUMENT: at a 32-bit word q6 is 6.4 b/w and is NOT
#   dominated. The chain is kernel measurement -> halfword container -> q6 dies, so anyone who
#   reopens the container has to reopen this too.
# ⛳ q1 IS IN THE SET AND IT IS NOT SERIOUS — ⚖ *"add q1 for the meme, it should be refuted but id
#   like to meme it out and have a good laugh at the output."* It is kept because it costs nothing to
#   support and because it is the cleanest statement of what this codec IS: at D=1 the table is
#   {-1,+1} and `k` comes out of the Lloyd solve as 0.7979 = sqrt(2/pi) = E|X| for a unit Gaussian,
#   which is the textbook one-bit quantiser. A weight becomes THE SIGN OF ITS ROTATED SELF times one
#   fp16 a row — every magnitude inside the row is gone.
#   ⛔ AND IT IS THE WORST CASE FOR THE WHOLE APPROACH, not a neutral one: `quant_families.md` §⑭ⓐ
#   measures the rotation as worth -53.7% at D=1 rising to -91.8% at D=8, so q1 is exactly where the
#   Hadamard helps LEAST. Expect ~60% rms on the weights. **Nothing should ship at this width.**
# ⛳ THE ALLOWED VALUES, AND NOTHING MORE — ⚖ *"the ladder stays only as the possible allowed
#   values."* A width is CHOSEN by `POLICY`, never searched for. ▶ the error table beside it.
#
# ⭐⭐⭐ AND `16` IS ONE OF THEM — ⚖ ARCHITECT: *"turn that escape hatch into the
#   reality, we have abandoned the rest of the old vbr code so let's promote the d==16 as the real
#   user."* It was written into the format as a *"raw-fp16 escape"* that the packer never emitted, so
#   the device could read a width nothing produced. It is now the width every unpacked tensor takes.
# ⛳ WHAT MAKES IT A WIDTH AND NOT A SPECIAL CASE: at `d == 16` the code IS the fp16 bit pattern, so
#   `values_per_container` is 1, `row_bytes` is `cols * 2`, and a row is a contiguous fp16 array. Every
#   other width's geometry formula gives the right answer for it without a branch.
# ⇒ ★★ IT SUBSUMES EVERY REASON A TENSOR USED TO TRAVEL AS A SEPARATE `dense` RECORD, and that is why
#   the promotion deletes a storage kind rather than adding a width. There were three such reasons —
#   the policy said RAW, the tensor had no contraction axis, or its columns were not a power of two —
#   and ALL THREE were about the ROTATION. At `d == 16` nothing is rotated, scaled or looked up in a
#   table, so none of the three can apply.
CONTAINER_BITS = 16
# ⛳ THE QUANTISED WIDTHS ARE THE TABLE'S OWN SET, IMPORTED AND NOT RETYPED. `nn_gaussian_tables` builds
#   one Lloyd codebook per width and is therefore the only thing that knows which widths HAVE one; the
#   reference packer already imports the same tuple from it. ⇒ ★ A WIDTH EXISTS BECAUSE A TABLE EXISTS,
#   so the set lives with the tables — a second tuple here would be free to disagree with the codebook
#   the decoder was built from, and this file has spent today watching hardcoded sets fail three ways.
QUANT_WIDTHS = tuple(TBL.WIDTHS)
# ⭐ AND THE FORMAT'S SET IS THAT PLUS THE LOSSLESS ONE, DERIVED. `d == CONTAINER_BITS` has no table by
#   construction — key-as-value needs none — so it cannot come from `TBL` and must be added here. That
#   asymmetry is the honest one: it is the width that is not a codebook.
WIDTHS = QUANT_WIDTHS + (CONTAINER_BITS,)
RETIRED_WIDTHS = {6: "dominated by q8 in a halfword container — same bytes, 15.7x the error"}


def values_per_container(d):
    """How many codes sit in one 16-bit container. The format's whole word shape is this number."""
    return CONTAINER_BITS // d

# ⛔ `lloyd_levels` HAS NO CACHE AND `REF.pack_row` CALLS `table(d)` PER ROW, so a whole-tensor pack
#   re-solves Lloyd-Max once per row — 256 levels at D=8. Deterministic, so memoising changes no
#   value; without it the reference packer is unusable as an oracle at tensor scale.
_TBL, _table_orig = {}, TBL.table


def table(d):
    if d not in _TBL:
        _TBL[d] = _table_orig(d)
    return _TBL[d]


TBL.table = table
REF.table = table


# ══ ⭐⭐ THE POLICY — A TENSOR'S NAME PICKS ITS WIDTH ═════════════════════════════════════════════════
#
# ⚖ *"check the name of the expert type and see if it is in a key value map for a specific quant
# compression level."* First matching pattern wins, so ORDER IS MEANING; `DEFAULT_WIDTH` catches
# anything unnamed.
#
# ⛔⛔ EVERY WIDTH BELOW IS `ASSUMED`, NOT `MEASURED`. They are placed by role and by row width, not by
#   any score: nothing has run the model. The one thing that IS measured is what each width costs in
#   error (`quant_families.md` §⑭) — and an error in weight space is not a quality in output space.
#   ⇒ ★ THE MAP IS THE KNOB THIS FILE EXISTS TO EXPOSE. Do not read these numbers as recommendations.
#
# `RAW` means "do not QUANTISE this tensor": a 1-D tensor has no contraction axis to rotate, and the
# norms and biases are a rounding error of the model's bytes. ⛳ `MEASURED`: the 35B's 1-D tensors total
# under 0.01% of its parameters, so packing them buys nothing and risks the one thing a norm cannot
# survive, which is a scale error.
#
# ⭐⭐ AND IT IS NOW THE WIDTH `16`, NOT A SENTINEL — ⚖ *"promote the d==16 as the real user."* The two
#   were always the same instruction spelled twice: "store this at full fp16 precision" was a `storage`
#   kind on one side of the format and a width on the other, and the bytes they produce are IDENTICAL.
#   ⇒ ★ A SENTINEL THAT MEANS THE SAME AS A VALUE IN THE SAME FIELD'S RANGE IS A SECOND SPELLING, and
#   the cost of two spellings is that every consumer needs a branch to discover they agree. Making it
#   the value deletes the branch, the `dense` storage kind, and the reader arm that went with them.
#   ⛳ The NAME stays, because what a POLICY row means is still "do not quantise this one" — a policy
#   should read as a decision about a tensor, not as a container width that happens to be lossless.
RAW = 16
DEFAULT_WIDTH = 8                     # ⚖ *"like default = 8"*

# ══ ⭐⭐⭐ WHAT EACH LEVEL COSTS AND WHAT IT COSTS YOU — `MEASURED` ═══════════════════════
#
# ⚖ ARCHITECT: *"the ladder can disappear, you can have the commented error rate per quant and then
# have the dictionary of expert type to quant level, plus a default. the ladder stays only as the
# possible allowed values."* ⇒ **`WIDTHS` IS A SET OF LEGAL VALUES, NOT A SEARCH SPACE.** Nothing in
# this file chooses a width by measuring; `POLICY` names one and `DEFAULT_WIDTH` catches the rest.
#
# Real 35B tensors, raw bf16 as ground truth, energy = `Σδ²/Σw²`, rms err = its square root = the
# typical relative size of the error on one weight. b/w and the ratio are at 2048 columns and include
# the one fp16 scale a row.
# ```
#    D  | per u16 |    b/w   ratio |    energy   rms err | note
#   ----+---------+----------------+---------------------+---------------------------------------
#    1  |   16    |  1.0078  15.88x|  0.363073   60.26%  | ⛔ THE MEME. nothing ships here.
#    2  |    8    |  2.0078   7.97x|  0.117366   34.26%  |
#    3  |    5    |  3.2109   4.98x|  0.034482   18.57%  | the halfword fee lives here
#    4  |    4    |  4.0078   3.99x|  0.009458    9.73%  | ⭐ the level the architect named default
#    5  |    3    |  5.3438   2.99x|  0.002494    4.99%  | the halfword fee lives here too
#    8  |    2    |  8.0078   2.00x|  0.000041    0.64%  | near-lossless, and what `DEFAULT_WIDTH` is
# ```
# ⭐⭐ THE COLUMN THAT MATTERS IS `energy`, AND WHAT MATTERS ABOUT IT IS THAT IT DOES NOT MOVE. Across
#   `down` (512 wide), `gate_up` (2048), `q_proj` and `o_proj`, D=4 lands in 0.00942..0.00952 — the
#   per-row scale makes `σ` a function of D and of nothing else. ⇒ ★★ THAT IS WHAT MAKES A
#   PER-TENSOR WIDTH A MEANINGFUL CHOICE AT ALL: the table above is a property of the FORMAT, so it
#   can be read once and applied to a tensor nobody measured.
#
# ⛔⛔ AND EVERY NUMBER ABOVE IS A WEIGHT-SPACE PROXY. `9.73% rms on the weights` is not `9.73% worse
#   output`, and nothing in this repository can currently turn one into the other — no end-task run
#   has scored any level. ⇒ ★ THE TABLE PRICES THE BITS HONESTLY AND PRICES THE QUALITY NOT AT ALL.
#
# ⛳ WHAT USED TO BE HERE: a per-ROW ladder that picked a width by bisecting on a relative error
#   target. It is gone, and its epitaph is worth one line because it looked like it was working:
#   a relative target is met at the SAME D by every row (`σ/rms = f(D)`, D alone), so it selected one
#   width per tensor by construction and its only varying output came from a degenerate-row branch.
#   ⇒ ★★ A PER-ROW RULE THAT ALWAYS PICKS THE SAME VALUE IS NOT A LADDER, AND ITS UNIFORM HISTOGRAM
#   IS THE PROOF. Measured properly the best a CORRECT ladder could win is 0.058 b/w
#   (reverse water-filling over 16 real tensors; usually 0.000), against the 0.141 b/w its per-row
#   bookkeeping planes cost at 512 columns. **It was a net loss.**

# ⚖⚖ THE ROUTED EXPERTS PACK AT q4 AND EVERYTHING ELSE STAYS AT `DEFAULT_WIDTH` — ARCHITECT:
#   *"yes q4 for experts"*. It was q8 everywhere until then (*"that's fine leave it at q8 for now"*).
#   ⛳ WHY IT WAS DECIDED ON TRANSFER AND FIT, NOT QUALITY: `MEASURED`, a q4 expert
#   computes in 21.5 us against 22.6 at q8, so compute does not care; at q8 the 35B's experts are ~32 GB
#   and no longer fit two cards beside the rest, and a disk miss runs past a layer's budget.
#   ▶ `measurements/2026-09-24_expert_load_times.md`. ⛔ NOT MEASURED: what q4 does to the model's
#   OUTPUT — the energy table above is error in weight space. `test/src_quant_endtask.py` is how.
#   ⛳ THE SHARED EXPERT STAYS AT q8: it runs every token and is resident, so its bytes never stream.
#   ⛳ AND THE DECISION IS ESSENTIALLY ONE ROW: `MEASURED` from the shipping 35B, the MoE experts are
#   **91.8% of 35.95B parameters**; everything else together is 8.2%, so putting every NON-expert
#   tensor at q8 rather than q4 costs about **1.3 GB on a 36 GB model**. Whatever is ruled later, the
#   cautious choice is nearly free everywhere except `.experts.`.
#     experts q4, rest q8 -> ~18.5 GB · experts q5, rest q8 -> ~24.1 GB · uniform q8 -> ~36.0 GB
#   ⛳ `test/src_quant_endtask.py CANDIDATES` scores a POLICY rather than a level, for when it is.
POLICY = [
    # pattern (regex, matched against the tensor's name)     width
    (r"\.experts\.down_proj$",                               4),
    (r"\.experts\.gate_up_proj$",                             4),
    (r"\.shared_expert\.(down|gate|up)_proj\.weight$",        DEFAULT_WIDTH),
    (r"\.self_attn\.[qkvo]_proj\.weight$",                    DEFAULT_WIDTH),
    (r"\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$", DEFAULT_WIDTH),
    (r"\.mlp\.gate\.weight$",                                 RAW),   # the router: 256x2048, and a
                                                                      # routing decision is an argmax,
                                                                      # so its error is not averaged
                                                                      # over anything. ASSUMED.
    (r"(^|\.)(embed_tokens|lm_head)\.weight$",                DEFAULT_WIDTH),
    (r"norm\.weight$",                                        RAW),
    (r"\.(bias|dt_bias|A_log)$",                              RAW),
    (r"conv1d\.weight$",                                      RAW),   # [8192,1,4] — no axis to rotate
    # GLM 5.3 Flash: the router's per-expert bias (288 values between 7.4 and 7.8, whose DIFFERENCES pick the
    # experts) and the hyper-connections' small tensors (at most 24 x 16384) — kept exact, for nothing
    (r"e_score_correction_bias$",                             RAW),
    (r"\.hc_(attn|ffn)_(base|fn|scale)$",                     RAW),
]


def clip_for_width_t(d):
    """`k` such that `scale = k * rms(row)` lands the table where it was fitted — the reference's own
    constant, re-exported so a consumer never derives a second one. ⛳ There is exactly ONE producer
    of this number (`nn_gaussian_pack.clip_for_width`, from the Lloyd fit) and everything else asks
    it. A second derivation would be a constant obliged to agree with a first."""
    return REF.clip_for_width(d)


def width_for(name, policy=None, default=DEFAULT_WIDTH):
    """The width this tensor is packed at, or `RAW`. First match wins."""
    for pattern, w in (policy if policy is not None else POLICY):
        if re.search(pattern, name):
            return w
    return default


# ══ THE TRANSFORM — the same one the reference packer applies, vectorised over rows ═════════════════
def sign_master(seed=0, n=SIGN_MASTER_BITS):
    """The one +-1 vector the whole model shares, as ruled. A tensor takes its first `cols` entries.

    ⛳ IT IS SLICED, NOT RE-DRAWN PER TENSOR, because the plane that ships is 512 bytes for the model.
    A tensor of `cols` columns uses `master[:cols]`; nothing needs the tail to be independent, only
    that the packer and the runtime agree on the same bits."""
    return np.array(REF.sign_vector(n, seed), dtype=np.float64)


def sign_plane(signs):
    """The +-1 vector as a bit plane, LSB-first: bit i is 1 when `signs[i] < 0`."""
    bits = (np.asarray(signs) < 0).astype(np.uint8)
    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little").tobytes()


def fwht(a):
    """`H` at the reference's normalisation, over axis 1. `a.shape[1]` must be a power of two.

    ⛳ THE BUTTERFLY ORDER IS THE REFERENCE'S: blocks of `2h`, first half against second half. The
    `--selftest` checks this against `REF.rotate` rather than asserting it."""
    R, n = a.shape
    if n & (n - 1):
        raise ValueError("FWHT needs a power-of-two length, got %d" % n)
    a = np.asarray(a, dtype=np.float64).copy()
    h = 1
    while h < n:
        a = a.reshape(R, n // (2 * h), 2, h)
        x = a[:, :, 0, :].copy()
        y = a[:, :, 1, :].copy()
        a[:, :, 0, :] = x + y
        a[:, :, 1, :] = x - y
        a = a.reshape(R, n)
        h *= 2
    return a / math.sqrt(n)


def rotate(W, signs):
    """`R(W) = H(S .* W)` applied to every row. ⛔ `R` is orthogonal but NOT an involution."""
    return fwht(np.asarray(W, dtype=np.float64) * signs[None, :])


# ══ ⭐⭐⭐ THE BLOCK STRUCTURE — DERIVED FROM `cols` ALONE, WHICH IS THE WHOLE POINT ══════════════════
#
# ⚖ ARCHITECT: *"uniform count with the list as fallback."*
#
# ⭐⭐ WHY UNIFORM IS WORTH REACHING FOR, AND IT IS NOT ELEGANCE. `MEASURED`: the rotation span does not
# change the error over a 64x range (ratios 0.94..1.02, nothing monotone — the CLT has converged by span
# 32, kurtosis 3.02). So the span is free to be chosen on OTHER grounds, and the ground that matters is
# that a UNIFORM structure is ONE INTEGER while a heterogeneous one is a LIST.
# ⇒ ★★ THREE PLACES MUST AGREE ABOUT IT — the packer that rotates the weights, the reader that
# un-rotates them, and the runtime that rotates the ACTIVATION to match. A single count cannot disagree
# with itself; a list can, and a disagreement decodes CORRECTLY and computes a WRONG PRODUCT, which is
# the failure mode with no symptom. Same shape as the halfword-container ruling: the structural argument
# beats the marginal one.
#
# ⛳ AND IT IS DERIVED, SO THE FORMAT GAINS NO FIELD. Given `cols`, every consumer computes the same
# structure — the record does not carry it and therefore cannot carry it wrongly.
#
# ⭐⭐ THE RULE IS THE 2-ADIC PART, CAPPED BY THE SIGN PLANE, AND IT LANDS ON 512 FOR BOTH REAL MODELS:
# ```
#   7B    3584 = 512x7    18944 = 512x37    152064 = 512x297     gcd 512, every cofactor ODD
#   35B   2048 = 2048x1     512 = 512x1     248320 = 512x485     so 512 is MAXIMAL, not chosen
#   also  896 = 128x7 (0.5B)   1152 = 128x9 (vision)   128 = head_dim
# ```
# ⛔ `4096` DIVIDES NONE OF THE 7B's DIMENSIONS, which is why the arithmetic needed `4096*4 + 2048 + 512`
# and why 512 is the block a uniform structure can actually use.
# ⚖⚖ THE LADDER, RULED: *"id say we do blocks of 512 and at the end it is blocks of 256 then
#   eventually 128 and the rest is done by padding which adds a zero line every n values so its damage is
#   dissolved in the 512 pools."*
#
# ⭐⭐⭐ SO 512 IS *THE* BLOCK, AND BIGGER ONES ARE NOT USED EVEN WHERE THEY EXIST. A 2048-wide axis takes
#   FOUR 512s, not one 2048. That is deliberate and it is what "uniform" buys: one block width across the
#   whole model, so the activation rotation is one loop with one constant and the sign plane is only ever
#   read 512 entries deep. The error is flat across spans, so nothing is given up for it.
#
# ⛳ `MEASURED` OVER FOURTEEN REAL DIMENSIONS — the ladder leaves a remainder on exactly ONE:
# ```
#   3584 = 7x512        18944 = 37x512       152064 = 297x512       2048 = 4x512
#   248320 = 485x512    1536 = 3x512         3072 = 6x512           5120 = 10x512
#   1152 = 2x512+128    8960 = 17x512+256    896 = 512+256+128      128 = 1x128
#   4304 = 8x512+128 + 80 LEFT  <- the vision tower, and the only shape that needs padding
# ```
#   ⇒ ★ A LADDER OF THREE WIDTHS COVERS EVERY LANGUAGE-MODEL AXIS EXACTLY. An earlier rule here took the
#   2-adic part of `cols`, which is also uniform but picks a DIFFERENT width per axis (2048, 1024, 256,
#   and 16 for the vision tower) — correct, and uniform only within one tensor rather than across the
#   model. The ladder is uniform across it.
# ⚖⚖ AND IT STAYS AT THREE TIERS — RULED: *"if k is up until 192 then 128 becomes the minimum
#   chunk, so we only need to support 3 tiers."* That follows from the padding measurement and it is the
#   whole argument:
# ```
#   placeable remainders mod 512 with three tiers   {0, 128, 256, 384}      gaps of 128
#   so the WORST padding to reach the next one      127, at n = 513         (checked 512..300000)
#   and padding is measured safe to                 k = 192                 margin 65
# ```
# ⇒ ★★ THE LADDER'S FLOOR AND THE PADDING BUDGET ARE THE SAME DECISION SEEN FROM TWO SIDES. A floor of
#   128 leaves at most 127 to pad; a budget of 192 covers 127. Neither number was chosen for the other and
#   they meet with room to spare, which is why no fourth tier is needed.
# ⛔ A 64 TIER WAS ADDED AND REMOVED THE SAME HOUR. It is not wrong — `MEASURED`, a 64 chunk among 512s
#   costs 1.002x and even 32s cost 1.007x — it is UNNECESSARY, and a tier nothing needs is a case every
#   reader of the decomposition has to carry. ⇒ ★ "IT WOULD WORK" IS NOT A REASON TO ADD A TIER.
# ⛳ `MEASURED` ACROSS ELEVEN REAL DIMENSIONS: exactly ONE needs padding at all — the vision tower's 4304,
#   which takes 48 zeros (1.10% of the padded length). Every language-model axis places exactly.
ROTATION_BLOCK_LADDER = (512, 256, 128)


def blocks_for(cols):
    """The rotation's block widths for a contraction axis of `cols`, and the zeros to PREPEND.
    Returns `(blocks, pad)` with `sum(blocks) == cols + pad`.

    ⚖⚖ **THE RULE IS ONE LINE — ARCHITECT:** *"adding up to 128 values at the beginning means
    we at most add 64 in every case except when the reminder is between 1 and 64, in that case we add
    128-reminder and we dont ever need to manage the 64 case."*
    ⇒ **`pad = (-cols) % 128`** — round the length up to the next multiple of 128, and the 512/256/128
    ladder then places it exactly, because every multiple of 128 has a remainder mod 512 in
    `{0, 128, 256, 384}`, which is precisely the placeable set.
    ⛳ `MEASURED` EQUIVALENT to the longer "reach the next placeable remainder mod 512" formulation over
    every length 1..400000: **zero disagreements.** ⇒ ★ THE SIMPLER STATEMENT WAS NOT AN APPROXIMATION OF
    THE RULE, IT WAS THE RULE — the mod-512 reasoning was the same fact seen through the ladder instead
    of through the floor.

    ⭐ AND THE COST DISTRIBUTION IS WHY NO 64 TIER IS NEEDED: `pad <= 64` for **50.00%** of lengths (those
    with `cols % 128 >= 64`), and `65..127` for the rest — the band `cols % 128` in `[1,63]`, exactly as
    stated. A 64 tier would only shave that second band, and `MEASURED`, it shaves NONE of the real
    shapes: the one that needs padding is 4304, whose residue is 80 and which therefore already sits in
    the cheap half. ▶ `quant_families.md`'s padding section for the full pricing.

    ⛔⛔ **THE ZEROS GO AT THE FRONT AND THAT IS A CORRECTNESS CALL.** Pooled at the tail they would make
    the LAST block almost all zeros — an rms far below its neighbours' — while the row carries ONE scale
    for every block, which is the case a per-row scale is the wrong compander for. At the front they
    dilute against 512 and move that block's rms by `sqrt(1 - pad/512)`: 0.87 at the worst case, inside
    what one scale serves. ⛳ And padding does not degrade the data at all — `MEASURED`, the error on the
    REAL positions falls as exactly `(512-k)/512`, because the white noise is spread over the whole block
    and the discarded positions carry their share away. The price is bits, not quality.

    ⛳ BELOW 128 A POWER-OF-TWO WIDTH IS ONE TRANSFORM AND IS NOT PADDED. The fixtures rotate 4-, 8- and
    16-wide vectors, and rounding those up to 128 would be padding a 4-wide vector with 124 zeros —
    arithmetically fine and obviously not what anybody means. ⇒ ★ A RULE THAT IS RIGHT FOR THE SIZES IT
    WAS DERIVED FROM STILL NEEDS ITS FLOOR STATED."""
    if cols <= 0:
        raise ValueError("cols must be positive, got %r" % (cols,))
    if cols < ROTATION_BLOCK_LADDER[-1]:
        if cols & (cols - 1):
            return (), cols                  # not a power of two and below the floor: nothing places it
        return (cols,), 0
    pad = (-cols) % ROTATION_BLOCK_LADDER[-1]
    out, r = [], cols + pad
    n512 = r // 512
    out += [512] * n512
    r -= 512 * n512
    for b in ROTATION_BLOCK_LADDER[1:]:
        if r >= b:
            out.append(b)
            r -= b
    assert r == 0, "the ladder must place a multiple of %d exactly" % ROTATION_BLOCK_LADDER[-1]
    return tuple(out), pad


def blocks_or_refuse(cols):
    """`blocks_for` for a caller that cannot pad — the EXACT placements only.

    ⛳ TWO DOORS RATHER THAN A FLAG, because the two callers want different things and always did: a pack
    that cannot prepend zeros needs to know this length is not for it, while a survey wants the padding
    figure. ⛔ AND IT IS STILL THE DOOR THE PACKER USES TODAY: prepending zeros changes what a record
    holds, so the activation must be padded identically or the product is wrong — the same coupling the
    512 promotion had, and it needs the same treatment (a manifest marker, and the device side first).
    ⚠ RETIREMENT CONDITION: when the activation path pads, this door's callers move to `blocks_for` and
    the `lossless_because = "ladder-leaves-a-remainder"` arm in `nn_pack_model.pack_record` goes with it."""
    blocks, pad = blocks_for(cols)
    if pad or not blocks:
        raise ValueError(
            "cols=%d needs %d zero(s) prepended to reach a multiple of %d, and this caller cannot pad — "
            "the padding is specified and the activation side is not built"
            % (cols, pad or cols, ROTATION_BLOCK_LADDER[-1]))
    return blocks


def blocked_rotate(W, signs, blocks=None):
    """`R` applied per block. `blocks=None` derives them from the row width, which is what every caller
    should do — passing them is for a probe that wants to vary the span.

    ⛳ EACH BLOCK TAKES THE SIGN VECTOR'S PREFIX `signs[:b]`, not a slice further along it. The ruled
    plane is 4096 signs for the WHOLE MODEL, so a structure whose blocks sum past that has nowhere else
    to read from — and nothing needs the patterns to differ between blocks, because the flip's job is to
    break alignment with the Hadamard basis WITHIN one."""
    W = np.asarray(W, dtype=np.float64)
    cols = W.shape[1]
    if blocks is None:
        # ⛳ THE REFUSING DOOR, because a rotation cannot proceed on a remainder: the values the ladder
        #   could not place have no transform to belong to, and silently dropping or tail-padding them
        #   is exactly the choice the padding ruling exists to make deliberately.
        blocks = blocks_or_refuse(cols)
    if sum(blocks) != cols:
        raise ValueError("blocks %r sum to %d, not cols=%d" % (blocks, sum(blocks), cols))
    out = np.empty_like(W)
    a = 0
    for b in blocks:
        out[:, a:a + b] = rotate(W[:, a:a + b], np.asarray(signs)[:b])
        a += b
    return out


def blocked_unrotate(R, signs, blocks=None):
    """`Rᵀ` per block — the inverse of `blocked_rotate`.

    ⛔ `Rᵀ = S ⊙ (H u)`, THE SAME TWO OPERATIONS IN THE OTHER ORDER, and not `R` again: `H` and `S` are
    each involutions and their composition is not. Getting this backwards returns something with the
    right norm and the wrong values, which no energy check would catch."""
    R = np.asarray(R, dtype=np.float64)
    cols = R.shape[1]
    if blocks is None:
        blocks = blocks_or_refuse(cols)
    out = np.empty_like(R)
    a = 0
    for b in blocks:
        out[:, a:a + b] = fwht(R[:, a:a + b]) * np.asarray(signs)[:b][None, :]
        a += b
    return out


def _fp16(x):
    """The fp16 this becomes, as the float it then is. Round-to-nearest-even, same as `struct` does."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float64)


# ══ ⭐ THE PACK — one tensor, one width, a per-row fp16 scale ════════════════════════════════════════
def pack_tensor(W, d, signs=None, seed=0, blocks=None):
    """`[rows][cols]` -> (planes, recon, stats). `d` is ONE width for the whole tensor.

    ⛔ A ROW WHOSE SCALE UNDERFLOWS fp16 TAKES `scale = 0` AND RECONSTRUCTS TO EXACT ZEROS. That is
    the honest encoding of a row at 1e-38, and it is why the table needs no zero level — every level
    times zero is zero, whatever code a column carries. Dividing by the underflowed scale instead
    produces inf/nan that `argmin` resolves to *something*, and that something decodes LARGE where the
    truth is zero. ⇒ ★ THE DEGENERATE CASE IS THE ONE WHERE A WRONG ANSWER LOOKS MOST LIKE A RIGHT ONE.
    ⛳ `MEASURED`: this path protects ~1.2% of the 35B's expert rows, not the ~50% an
    earlier note claimed — the dead rows are layer 0 (48%) and layer 1 (4.3%), and ~0 elsewhere."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("pack_tensor wants a 2-D tensor, got shape %r" % (W.shape,))
    if d in RETIRED_WIDTHS:
        raise ValueError("D=%d is retired: %s" % (d, RETIRED_WIDTHS[d]))
    if d not in WIDTHS:
        raise ValueError("D=%r is not a width this format defines (%r)" % (d, WIDTHS))
    rows, cols = W.shape

    # ⭐⭐⭐ `d == 16` IS THE LOSSLESS WIDTH AND IT TAKES NONE OF THE MACHINERY BELOW — ⚖ *"promote the
    #   d==16 as the real user."* The code IS the weight's own fp16 bit pattern, so there is no table to
    #   index, no scale to divide by and NOTHING TO ROTATE. It returns here, before the rotation, and
    #   that early return is the whole of the width's implementation.
    # ⛔⛔ AND THE ROTATION MUST NOT BE APPLIED, WHICH IS THE ONE WAY TO GET THIS WRONG. The decoder's
    #   `d == 16` arm reads the halfword straight back as a value (`vbr__impl.cuh:204`) and does not
    #   un-rotate, so a packer that rotated here would produce a record that decodes to `R(w)` while
    #   every consumer believes it holds `w`. ⇒ ★ THE TWO SIDES AGREE BY BOTH DOING NOTHING; the danger
    #   is that one of them does something reasonable.
    # ⛳ SO THE POWER-OF-TWO REFUSAL IS ASKED *AFTER* THIS, not before: it is a fact about the rotation,
    #   and this width has no rotation. That ordering is what lets a 3584-column tensor, a 1-D norm and
    #   a policy-RAW row all take the same path.
    if d == CONTAINER_BITS:
        codes = np.asarray(W, dtype=np.float32).astype("<f2").view("<u2").astype(np.uint32)
        recon = codes.astype("<u2").view("<f2").astype(np.float64)
        # ⛳ ONE fp16 OF ZERO, SO THE FIELD SET IS THE SAME AT EVERY WIDTH. The plane is never read at
        #   this width — the device returns the base without bounds-checking it — but a record whose
        #   fields differ by width would make every consumer branch just to locate them, and the
        #   operand still has to mint. Two bytes a tensor is the price of one shape.
        planes = dict(vbr_data=np.asarray(codes, dtype="<u2").tobytes(),
                      vbr_lut=np.zeros(1, dtype="<f2").tobytes())
        meta = dict(family=FAMILY_GAUSSIAN, d=d, rows=rows, cols=cols, row_bytes=cols * 2)
        den = float((W ** 2).sum())
        stats = dict(meta, dead=0, bits_per_weight=bits_per_weight(d, cols),
                     stored_bits_per_weight=8.0 * sum(len(v) for v in planes.values())
                                            / (rows * cols),
                     energy=float(((recon - W) ** 2).sum() / den) if den > 0 else 0.0)
        return planes, recon, stats

    # ⭐⭐⭐ A NON-POWER-OF-TWO AXIS IS ROTATED IN BLOCKS, DERIVED FROM `cols` — it no longer refuses.
    #   ⚖ *"3584 is 2048+1024+512, and 18944 is 4096*4 + 2048 + 512, so it does fit"*, and ⚖ *"uniform
    #   count with the list as fallback"*. `MEASURED`: blocking costs NOTHING in error at the spans this
    #   produces — the real 7B's three shapes pack at q4 0.00931..0.00951 against the 35B's whole-row
    #   0.00941..0.00955. ▶ `measurements/2026-09-21_span_does_not_change_the_error.md`.
    #   ⛳ `signs` is taken at FULL LENGTH here rather than sliced to `cols`, because each block reads the
    #   prefix it needs; slicing to `cols` would starve a structure whose blocks sum past the plane.
    if signs is None:
        signs = sign_master(seed)
    signs = np.asarray(signs)
    if signs.shape[0] < min(cols, SIGN_MASTER_BITS):
        raise ValueError("the sign vector is %d long, too short for cols=%d"
                         % (signs.shape[0], cols))

    # ⛳ `blocks=None` ASKS THE TRANSITION DOOR; passing them is how a test exercises the ladder without
    #   changing what a real pack does. ⇒ ★ A DORMANT CAPABILITY STILL NEEDS A LIVE CHECK, or it is
    #   discovered broken on the day it is switched on.
    rot = blocked_rotate(W, signs, blocks if blocks is not None else blocks_or_refuse(cols))
    levels = np.array(table(d)[1], dtype=np.float64)
    mid = (levels[:-1] + levels[1:]) * 0.5

    rms = np.sqrt((rot ** 2).mean(axis=1))
    scale = _fp16(REF.clip_for_width(d) * rms)
    dead = scale < FP16_MIN_NORMAL
    scale = np.where(dead, 0.0, scale)

    safe = np.where(scale > 0.0, scale, 1.0)[:, None]
    codes = np.searchsorted(mid, rot / safe, side="left").astype(np.uint32)
    codes[dead, :] = 0
    recon = np.where(scale[:, None] > 0.0, levels[codes] * scale[:, None], 0.0)

    # ── ⭐⭐ TWO PLANES AND THREE SCALARS, WHERE THE FORMAT USED FIVE PLANES ──────────────────────
    # With ONE width for the tensor, the other three planes are arithmetic and were only ever there
    # to let a row differ from its neighbour:
    #     vbr_offsets[r]     == r * row_bytes          a stride
    #     vbr_lut_offsets[r] == r                      one fp16 a row, in order
    #     bitrates[r]        == (family << 5) | d      one constant
    # ⭐ `MEASURED`: they cost 9 bytes a row = 0.1406 b/w at 512 columns, 0.0352 at 2048 — which is
    #   MORE than the per-row width ladder they exist to support could ever have earned (<= 0.058 b/w,
    #   and usually 0.00). ⇒ ★★ THE KNOB COST MORE IN BOOKKEEPING THAN IT EARNED IN CODES, and the
    #   gap is widest exactly on the narrow rows where the ladder was weakest.
    per_unit = values_per_container(d)
    nunits = (cols + per_unit - 1) // per_unit
    pad = nunits * per_unit - cols
    c = codes if pad == 0 else np.concatenate(
        [codes, np.zeros((rows, pad), dtype=np.uint32)], axis=1)
    shifts = (d * np.arange(per_unit, dtype=np.uint32))
    units = np.bitwise_or.reduce(
        (c.reshape(rows, nunits, per_unit) & np.uint32((1 << d) - 1)) << shifts[None, None, :],
        axis=2).astype("<u2")

    row_bytes = nunits * 2
    scale_bits = np.asarray(scale, dtype=np.float32).astype("<f2").view("<u2")
    planes = dict(vbr_data=units.tobytes(), vbr_lut=scale_bits.tobytes())
    meta = dict(family=FAMILY_GAUSSIAN, d=d, rows=rows, cols=cols, row_bytes=row_bytes)
    den = float((rot ** 2).sum())
    stats = dict(meta, dead=int(dead.sum()),
                 bits_per_weight=bits_per_weight(d, cols),
                 stored_bits_per_weight=8.0 * sum(len(v) for v in planes.values()) / (rows * cols),
                 energy=float(((recon - rot) ** 2).sum() / den) if den > 0 else 0.0)
    return planes, recon, stats


def bits_per_weight(d, cols):
    """What the format costs: the codes in their containers, plus ONE fp16 a row and nothing else.

    ⛳ The codes cost `16 / values_per_container(d)`, NOT `d` — at D=3 and D=5 that is the 6.25%
    halfword fee. No block scales, no per-row table (family 2's levels are a build constant), and
 no offset, rate or lut-offset plane either.

    ⛳ AND `d == 16` PAYS NO PER-ROW fp16, because key-as-value carries its own magnitude and there is no
    scale to store. So it is exactly 16.0 b/w — the same as the bf16 source, which is the point of it:
    the lossless width costs what the model costs and not a bit more."""
    if d == CONTAINER_BITS:
        return float(CONTAINER_BITS)
    return CONTAINER_BITS / float(values_per_container(d)) + 16.0 / cols


# ══ ⛔ THE SELFTEST — this packer against the reference producer, exactly ════════════════════════════
def ladder_selftest(hi=40000):
    """The block arithmetic, as PROPERTIES over every length — and it needs no model, so it runs even
    where the raw weights are absent.

    ⛳ PROPERTIES AND NOT EXAMPLES, because the rule is arithmetic and an example checks one length. What
    must hold for every `cols`: the blocks sum to `cols + pad`; the pad is under the floor; every block is
    a ladder width; and a length already on the floor's multiple is not padded at all.
    ⇒ ★ THE ONE THAT WOULD HAVE CAUGHT THE FIRST DRAFT is the last: an earlier form rounded EVERY length
    up, which padded a 4-wide fixture with 124 zeros — arithmetically consistent and obviously wrong."""
    floor = ROTATION_BLOCK_LADDER[-1]
    bad = 0
    for n in range(1, hi):
        blocks, pad = blocks_for(n)
        if not blocks:                      # below the floor and not a power of two — nothing places it
            continue
        if sum(blocks) != n + pad:
            print("  ⛔ n=%d: blocks sum to %d, not %d+%d" % (n, sum(blocks), n, pad)); bad += 1
        if pad >= floor:
            print("  ⛔ n=%d: pad %d is not under the floor %d" % (n, pad, floor)); bad += 1
        if n >= floor and pad != (-n) % floor:
            print("  ⛔ n=%d: pad %d is not (-n) %% %d" % (n, pad, floor)); bad += 1
        if n >= floor and n % floor == 0 and pad != 0:
            print("  ⛔ n=%d is a multiple of %d and was padded %d" % (n, floor, pad)); bad += 1
        for b in blocks:
            if b not in ROTATION_BLOCK_LADDER and not (n < floor and b == n):
                print("  ⛔ n=%d: %d is not a ladder width" % (n, b)); bad += 1
    # ⛳ AND THE FIXTURES BY NAME, because "a power of two below the floor is not padded" is the clause a
    #   later simplification is most likely to drop, and the suites that would break run on a GPU.
    for n in (4, 8, 16, 32, 64):
        blocks, pad = blocks_for(n)
        if blocks != (n,) or pad:
            print("  ⛔ the %d-wide fixture became %r pad %d" % (n, blocks, pad)); bad += 1
    print("  the block ladder over 1..%d: %s" % (hi, "EVERY PROPERTY HOLDS" if bad == 0
                                                 else "%d violation(s)" % bad))
    return bad == 0


def selftest(raw=None, rows=6):
    """`pack_tensor` must agree with `REF.pack_row` CODE FOR CODE, not merely in aggregate error.

    ⇒ ★ AN AGGREGATE AGREEMENT IS A CLAIM ABOUT THE METRIC; a code-for-code one is a claim about the
    format. Only the second catches a butterfly order, a tie-break or a rounding mode."""
    ok_ladder = ladder_selftest()
    raw = raw or os.environ.get("SILVANN_RAW_MODEL", "/mnt/data/bigdisk/qwen_36_35_a3b")
    name = "model.language_model.layers.3.mlp.experts.down_proj"
    index = os.path.join(raw, "model.safetensors.index.json")
    if not os.path.isfile(index):
        print("SKIP — no raw model at %s (set SILVANN_RAW_MODEL); the ladder above still ran" % raw)
        return ok_ladder
    # ⛔⛔ TWO SUBJECTS, AND THE SECOND ONE EXISTS BECAUSE THE FIRST PASSES FOR THE WRONG REASON.
    #   `experts.down_proj` is 512 COLUMNS WIDE, which is exactly one block of the 512/256/128 ladder —
    #   so on it a blocked rotation and a whole-row rotation are the SAME COMPUTATION, and this selftest
    #   agreed with the reference before the ladder existed and after it, without ever exercising it.
    #   ⇒ ★★ A CHECK THAT PASSES BEFORE AND AFTER A CHANGE IS NOT EVIDENCE ABOUT THE CHANGE. The second
    #   subject is 2048 wide — FOUR blocks — so the reference has to be composed per block to match, and
    #   a whole-row reference disagrees with it. That disagreement is the ladder, and it is now checked.
    subjects = [(name, 0), ("model.language_model.layers.3.self_attn.q_proj.weight", None)]
    bad = 0
    for _name, _exp in subjects:
        W = read_tensor(raw, _name, expert=_exp)
        W = W.reshape(-1, W.shape[-1])[:rows]
        bad += _agree_on(W, _name.split("layers.3.")[-1], blocks_or_refuse(W.shape[1]))
    return _selftest_report(bad, rows)


def _agree_on(W, label, blocks):
    """`pack_tensor`'s codes against `REF.pack_row`'s, per block, for one tensor. Returns the bad count.

    ⛳ THE REFERENCE IS FED ONE BLOCK AT A TIME rather than taught the ladder. `REF.pack_row` is a
    whole-vector routine and is CORRECT for one block; composing it per block is what makes it an
    independent second opinion about the blocked format instead of a copy of this file's loop.
    ⇒ ★ THE THING THAT MUST NOT BE SHARED IS THE STRUCTURE, NOT THE ARITHMETIC."""
    bad = 0
    cols = W.shape[1]
    master = sign_master()
    # ⛳ THE QUANTISED WIDTHS ONLY, AND THAT IS NOT A NARROWING OF THE CLAIM. This row asserts that this
    #   packer agrees with the reference CODE FOR CODE, and at `d == CONTAINER_BITS` there is no
    #   reference to agree with: `REF.pack_row` needs a table, key-as-value has none, and the "code" is
    #   the input's own fp16 bit pattern. ⇒ ★ THE LOSSLESS WIDTH IS CHECKED BY A DIFFERENT INSTRUMENT —
    #   exact equality against `fp16(src)`, which `nn_unpack_model.py --bundle` runs per record and which
    #   is stronger than agreement with a second implementation.
    for d in QUANT_WIDTHS:
        planes, recon, _ = pack_tensor(W, d, blocks=blocks)
        for r in range(W.shape[0]):
            # ⛔ THE ROTATION IS PER BLOCK AND THE QUANTISATION IS PER ROW, which is the one asymmetry a
            #   reader of this loop has to hold: the blocks are rotated independently and then the WHOLE
            #   row shares one scale. Rotating per block and scaling per block would be a different
            #   format — ⚖ measured and refused, it buys 0..4% for 10x worse bits.
            rr = []
            a = 0
            for b in blocks:
                rr.extend(REF.rotate([float(v) for v in W[r, a:a + b]], list(master[:b])))
                a += b
            c_ref, s_ref, rec_ref = REF.pack_row(rr, d)
            got = _codes_of(planes, r, d, cols)
            if list(got) != list(c_ref):
                n = int((np.asarray(got) != np.asarray(c_ref)).sum())
                print("  ⛔ %s D=%d row %d — %d of %d codes differ" % (label, d, r, n, len(c_ref)))
                bad += 1
            if max(abs(x - y) for x, y in zip(recon[r], rec_ref)) > 1e-12:
                print("  ⛔ %s D=%d row %d — reconstruction differs" % (label, d, r))
                bad += 1
    return bad


def _selftest_report(bad, rows):
    # ⛔ AND IT COUNTS WHAT IT CHECKED, NOT WHAT THE FORMAT DEFINES. This said `len(WIDTHS)` and reported
    #   "7 widths" the moment `d == 16` joined the set, while the loop above deliberately covers only the
    #   six with a codebook. ⇒ ★ A PASS LINE THAT OVERSTATES ITS OWN SCOPE IS WORSE THAN A QUIETER ONE:
    #   the number is the only thing a reader has to tell "checked everything" from "checked most of it",
    #   and it read as the former while being the latter. It is derived from the loop's own set.
    print("  pack_tensor vs nn_gaussian_pack.pack_row: %s (%d rows x %d quantised widths; "
          "D=%d is lossless and is checked by exact equality instead)"
          % ("AGREES EXACTLY" if bad == 0 else "%d DISAGREEMENTS" % bad,
             rows, len(QUANT_WIDTHS), CONTAINER_BITS))
    return bad == 0


def _codes_of(planes, r, d, cols):
    """Unpack row `r`'s codes back out of the container plane — the decoder's half, so the selftest's
    round trip goes through the bytes that ship rather than through the array that made them."""
    per_unit = values_per_container(d)
    nunits = (cols + per_unit - 1) // per_unit
    units = np.frombuffer(planes["vbr_data"], dtype="<u2")[r * nunits:(r + 1) * nunits]
    out = np.empty(nunits * per_unit, dtype=np.uint32)
    for s in range(per_unit):
        out[s::per_unit] = (units.astype(np.uint32) >> np.uint32(d * s)) & np.uint32((1 << d) - 1)
    return out[:cols]


# ══ reading the bundle ══════════════════════════════════════════════════════════════════════════════
_HDR = {}


_IDX = {}
PER_EXPERT = "per_expert"    # a view's weight_map value: this stacked expert tensor is assembled from one tensor an expert


def _index(raw):
    """A checkpoint's `model.safetensors.index.json`, read once — the 397B heretic's holds 93,411 names."""
    if raw not in _IDX:
        _IDX[raw] = json.load(open(os.path.join(raw, "model.safetensors.index.json")))
    return _IDX[raw]


def read_tensor(raw, name, expert=None):
    """One tensor (or one expert's slice of a 3-D expert stack) as fp32, from its bytes in the dtype the file says.

    ⛳ A 3-D expert tensor is read SLICE BY SLICE, not whole: `gate_up_proj` is `[256,1024,2048]` =
    1 GiB, and a packer that reads it entire to take one expert pays that per expert.
    ⛔⛔ THE DTYPE IS READ FROM THE HEADER, NOT ASSUMED. It was always bf16, which every Qwen checkpoint is; GLM 5.3
    Flash keeps 291 small tensors in F32 (`hc_*`, `e_score_correction_bias`, `A_log`, `dt_bias`), and read as bf16
    they came out as values like 4e37 and NaN — from half their bytes. `MEASURED`.
    ⭐ A VIEW (▶ `nn_expert_view.py`) may name a stacked expert tensor `per_expert`: a checkpoint that keeps its experts
    one tensor each (the 397B heretic's `experts.<e>.{gate,up,down}_proj.weight`) is read as the stacked layout its
    base uses — `down_proj` the experts' downs stacked, `gate_up_proj` each expert's gate rows then its up rows."""
    idx = _index(raw)
    if idx["weight_map"].get(name) == PER_EXPERT:
        return _read_per_expert(raw, idx, name, expert)
    return _read_from(raw, idx["weight_map"][name], stored_name(idx, name), expert)


def stored_name(idx, name):
    """The name a tensor has in its file. ⭐ A VIEW MAY RENAME (▶ `nn_rename_view.py`): its index names the tensor as the
    runtime does and keeps, in `stored_names`, what the checkpoint called it — Mistral 3's `language_model.model.layers.N…`
    read as `model.language_model.layers.N…`."""
    return idx.get("stored_names", {}).get(name, name)


def _read_per_expert(raw, idx, name, expert):
    pm = idx["per_expert_map"]
    stem, kind = name.rsplit(".", 1)                         # `…mlp.experts`, `down_proj` | `gate_up_proj`
    parts = {"down_proj": ("down_proj",), "gate_up_proj": ("gate_proj", "up_proj")}[kind]
    key = lambda e, part: "%s.%d.%s.weight" % (stem, e, part)
    one = lambda e: np.concatenate([_read_from(raw, pm[key(e, part)], key(e, part)) for part in parts], axis=0)
    if expert is not None:
        return one(expert)
    n = 0
    while key(n, parts[0]) in pm:
        n += 1
    return np.stack([one(e) for e in range(n)])


def _read_from(raw, f, name, expert=None):
    if f not in _HDR:
        with open(os.path.join(raw, f), "rb") as fh:
            hl = int.from_bytes(fh.read(8), "little")
            _HDR[f] = (hl, json.loads(fh.read(hl)))
    hl, head = _HDR[f]
    t = head[name]
    shape, start, dtype = t["shape"], t["data_offsets"][0], t["dtype"]
    size = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
    if size is None:
        raise ValueError("%s is %s — a dtype this reader does not know" % (name, dtype))

    def widen(raw_bytes, count):
        if dtype == "BF16":
            return (np.frombuffer(raw_bytes, dtype="<u2", count=count).astype(np.uint32) << 16).view(np.float32)
        return np.frombuffer(raw_bytes, dtype="<f2" if dtype == "F16" else "<f4", count=count).astype(np.float32)

    if expert is not None and len(shape) == 3:
        per = shape[1] * shape[2]
        with open(os.path.join(raw, f), "rb") as fh:
            fh.seek(8 + hl + start + expert * per * size)
            return widen(fh.read(per * size), per).reshape(shape[1], shape[2])
    n = 1
    for s_ in shape:
        n *= s_
    with open(os.path.join(raw, f), "rb") as fh:
        fh.seek(8 + hl + start)
        return widen(fh.read(n * size), n).reshape(shape)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
    print("THE POLICY — first match wins, default D=%d\n" % DEFAULT_WIDTH)
    for pattern, w in POLICY:
        print("  %-52s %s" % (pattern, w if w == RAW else "D=%d  (%.4f b/w at 2048 cols)"
                              % (w, bits_per_weight(w, 2048))))
    print("\nTHE WIDTHS THE FORMAT DEFINES")
    for d in WIDTHS:
        print("  D=%d  %4d levels  %.4f b/w at 512 cols  %.4f at 2048"
              % (d, 1 << d, bits_per_weight(d, 512), bits_per_weight(d, 2048)))
