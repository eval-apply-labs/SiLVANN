#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_TURBOQUANT_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_TURBOQUANT_CUH

/* This file needs nothing: every name in it is its own. */
#define NN__TURBOQUANT__FAULT_BOUNDS 0x4E564247ull   /* "NVBG" — a plane is too small for what the row needs  */
#define NN__TURBOQUANT__FAULT_RATE   0x4E565254ull   /* "NVRT" — a bit width this codec has no word shape for */
#define NN__TURBOQUANT__FAULT_FAMILY 0x4E56464Dull   /* "NVFM" — a row claims a family this build cannot read */

/* ══ ⭐⭐ THE COMPRESSION FAMILY — WHAT A ROW'S CODES MEAN, CHOSEN PER ROW ════════════════════════════
 *
 * ⚖ RULED: *"ideally experts should declare their compression type in their id card so we could mix
 * and match and/or have multiple strategies without costing us performance … maybe even per row."*
 *
 * ⭐⭐ AND PER ROW WAS ALREADY HALF TRUE, WHICH IS WHY THIS COSTS NO BYTES. `bitrates[r]` is a `uint8`
 * carrying a width; a width reaches 16 and so needs five bits; the top three were never used.
 * `MEASURED` across 245,760 rows of the shipping 35B: the plane holds only 4 and 5, and NOT ONE ROW
 * has a bit above 4 set. ⇒ ⭐ **FAMILY 0 IS THE CODEBOOK, SO EVERY FILE EVER PACKED IS ALREADY TAGGED**
 * — backward compatibility by choosing the encoding rather than by writing a migration.
 * ```
 *     bitrates[r] = (family << 5) | width        width 0..31   ·   family 0..7
 * ```
 * ⛳ AND THE SCALE NEEDS NO PLANE EITHER: a UNIFORM row's codebook is ONE entry, so it rides
 * `vbr_lut` at this row's existing `vbr_lut_offsets[r]`. Five planes before, five planes after.
 *
 * ── ⭐⭐ THE TWO FAMILIES ARE OPPOSITE STRATEGIES, AND THE MEASUREMENT SAYS SO ────────────────────────
 * ⚖ *"those are two opposite strategies, one smears and does a linear quant addition and the other
 * keeps the row as sharp and has a lookup table, better for switchboard style rows where the values
 * are more sparse but also more quantized."*
 * `MEASURED` on a real expert (layer 0, `down_proj`, 2048×512) against the raw bf16:
 * ```
 *                        L1  Σ|δ|/Σ|w|      L2  Σδ²/Σw²
 *     CODEBOOK, shipped        0.064            0.0053     wins L1 — the many small values
 *     UNIFORM q5               0.093            0.0039     wins L2 — the few large ones
 * ```
 * ⇒ ★★ **THE TWO METRICS DISAGREE, AND THAT DISAGREEMENT *IS* THE DISCRIMINATOR.** A codebook packs
 * its levels near zero, so it represents a sparse "switchboard" row well and L1 rewards it; uniform
 * levels bound the error on the large values, which is what L2 — and therefore a dot product's output
 * error, `E[(Σδx)²] = σ²Σδ²` — rewards. ⇒ ⛳ **SO THE PACKER CAN CHOOSE PER ROW BY MEASURING BOTH**,
 * and neither family is the right global answer.
 *
 * ⛔⛔ WHAT IS *NOT* HERE, AND MUST NOT BE ASSUMED: THE ROTATION. The uniform family's whole premise is
 * that the row was Hadamard-rotated at pack time and the ACTIVATION is rotated to match at run time.
 * This file decodes; nothing here rotates anything — the rotation is its own verb,
 * `nn__hadamard__rotate`. ⇒ ★ A ROW TAGGED UNIFORM IS DECODED CORRECTLY AND WILL STILL PRODUCE A WRONG PRODUCT UNTIL THE ACTIVATION IS
 * ROTATED BY THE SAME TRANSFORM — which is a separate verb, and ⚖ where its sign vector lives is a
 * ruling nobody has made. ══════════════════════════════════════════════════════════════════════════ */

/* ══ ⭐⭐⭐ FAMILY 2 — THE GAUSSIAN TABLE, SHARED BY THE WHOLE BUILD ═══════════════════════════════════
 *
 * ⚖ ARCHITECT: *"build that quant format with the global nf table, fwht row scaler and no
 * block scaler. i guess those values being sgpr is really the prize here..."*
 *
 * ── THE THREE FAMILIES ARE ONE QUESTION ASKED THREE WAYS: WHERE DO A ROW'S LEVELS COME FROM? ────────
 * ```
 *   0 CODEBOOK   fitted per row, stored per row     2^(D-1) fp16 a row   sign in bit 0
 *   1 UNIFORM    evenly spaced, one scale a row     1 fp16 a row         sign in bit 0
 *   2 GAUSSIAN   a FIXED table, one scale a row     1 fp16 a row         ⭐ NO SIGN BIT
 * ```
 * ⇒ ⭐ FAMILY 2 IS THE ONLY ONE WHOSE LEVELS DO NOT LIVE IN THE FILE. The row carries a scale and
 * nothing else; the shape of the distribution is a build constant. ⛳ Which is the architect's point
 * about SGPRs: a table that is the same for every row of every tensor is loaded once and is UNIFORM
 * ACROSS THE WAVE, where a per-row codebook is a VRAM read on the critical path of every row.
 * ⛔ THAT IS `REASONED`, NOT `MEASURED` — ▶ `nn_port.md` §⑧ for why this tree cannot measure it:
 * every door runs its body as one block of one thread (`gpu/doors.cuh`), and the one `threadIdx` in `packages/nn` is the guard that sends every other lane home. **No throughput claim here
 * has been observed, and none may be made until the gemv is parallelised.**
 *
 * ── ⭐⭐ NO SIGN BIT, AND THAT IS THE BIT THE ARCHITECT WENT LOOKING FOR ──────────────────────────────
 * ⚖ *"use all 4 bits for data instead of 3 bits for data and one for parity because we can adjust the
 * vector so that the turboquant transform lands into a balanced result without wasting one out of four
 * bits for energy balance."*
 * Families 0 and 1 spend bit 0 on a sign and index `2^(D-1)` MAGNITUDES. Family 2's table is SIGNED
 * and the code indexes it whole — `2^D` distinct levels from the same D bits.
 * ⛳ AND THE RECOVERED BIT IS NOT THE ONLY GAIN: a magnitude table has `+0` and `−0` and they are the
 * same number, so one code is dead. `MEASURED` on the shipping 35B: **98.7% of rows carry a codebook
 * whose entry 0 is exactly 0.0**, i.e. 98.7% of rows spend a code saying nothing.
 * ⛔ THAT IS ABOUT FAMILIES 0 AND 1. Family 2 has no sign bit AND no zero level, so there is no
 * redundant `−0` here to reclaim — that code was already spent on a real level, which is where this
 * family's advantage comes from. An "escape code" in family 2 therefore costs a REAL level: `MEASURED`
 * at D=4, 15 specular levels plus an outlier escape scores 0.010298 against 0.009442 for 16 levels —
 * **8% worse**. ⛳ The same rescue applied as a POST-DOT correction instead of a code is worth about
 * +0.015 bits/weight, which is real and is not built: it needs two more per-row fields and a fixup in
 * every gemv, for 0.4% of a 4-bit budget. ⚖ Priced and left to the architect.
 *
 * ── ⭐ WHAT THE LEVELS ARE, AND WHY A FIXED TABLE IS ALLOWED TO WORK ─────────────────────────────────
 * The MSE-optimal (Lloyd-Max) reconstruction levels for a unit Gaussian, normalised to ±1. A fixed
 * table is only correct if every row really is Gaussian, and a row of weights is not; **the FWHT is
 * what makes it one**, by the central limit theorem, and the rotation is therefore this family's
 * PRECONDITION rather than its decoration.
 *
 * ── ⭐⭐ THE TABLE IS SPECULAR AND SPENDS NO LEVEL ON ZERO — ⚖ RULED ───────────────────────
 * ⚖ ARCHITECT: *"if it is gaussian the values would be specular"*, then *"yes land it."* Every `2^D`
 * here is EVEN, so the optimum for a symmetric density is perfectly mirrored: `2^(D-1)` magnitudes,
 * each with both signs, nothing lopsided and nothing spent on a zero.
 * `MEASURED`, five real rotated 35B tensors at D=4, identical bits and identical lookup:
 * ```
 *   NF quantiles, one zero, lopsided 7/8    0.009847     what this family shipped first
 *   Lloyd-Max with a zero pinned            0.009535
 *   Lloyd-Max, SPECULAR, no zero            0.009442     ⭐ −4.3%, and the zero was never needed
 * ```
 * ⛔⛔ AND THE ZERO LEVEL WAS PAID FOR ON AN ARGUMENT THE ROTATION HAD ALREADY VOIDED. It was kept so
 * a row of zeros could reconstruct as zero. Two reasons that is unnecessary:
 * ```
 *   ① a row that must be zero gets `scale = 0`, and every level times zero is zero — the zero comes
 *     from the MULTIPLIER, not from a level
 *   ② the rotation DESTROYS individual zeros before the quantiser sees one: `H` mixes all `cols`
 *     columns, so a single zero weight is not zero afterwards. Only WHOLE zero rows survive `R` —
 *     which is exactly case ①
 * ```
 * ⇒ ★★ A ZERO LEVEL IS WORTH ITS CODE ONLY WHERE EXACT ZEROS REACH THE QUANTISER, AND HERE NONE DO —
 * the rotation sees to that. It cost 1% until somebody asked what it was for.
 *
 * ⛔ CHANGING THESE CONSTANTS CHANGES THE FORMAT. Bytes packed against the old levels decode wrong
 * against these, silently and plausibly. Nothing had shipped family 2, so this was free once and will
 * not be free again: a future table is a new FAMILY (three bits, five values spare), exactly as
 * family 0 is "the codebook that shipped".
 * `MEASURED`, nine real 35B tensors, energy `Σδ²/Σw²` at D=4 with a per-row scale:
 * ```
 *   rotation helps 8 of 9 (−12.4% … −65.9%); it hurts only `experts.down`, at +4.1%
 *   this table beats evenly-spaced levels 9 of 9, by 23.7% … 49.7%
 *   ⭐ and every rotated tensor lands at 0.0102…0.0121 — a ±4% band, where UNROTATED they span 3.4×
 * ```
 * ⇒ ★★ THE ROTATION'S SECOND GIFT IS THE ONE THAT DOES NOT SHOW UP IN AN AVERAGE: it makes the error
 * the SAME everywhere, so a model-wide error budget becomes a real object instead of a per-tensor
 * negotiation. That is worth more operationally than the mean improvement.
 *
 * ⛔⛔ AND THE ROTATION IS **NOT** RECORDED IN THE FAMILY TAG — THE TWO ARE ORTHOGONAL AND MUST STAY SO.
 * This family decodes a code to a number; it neither knows nor needs to know whether the bytes were
 * rotated before packing. Rotation is a property of a TENSOR (the whole contraction axis moves at
 * once), the family is a property of a ROW, and the one tensor that wants the rotation OFF —
 * `experts.down` — still wants this family. ⇒ ★ FOLDING THEM INTO ONE TAG WOULD MAKE A PER-ROW FIELD
 * ANSWER A PER-TENSOR QUESTION, and the runtime would be reading row 0 to decide what to do with the
 * activation. ⚖ Where the rotation flag and its sign vector live is a ruling nobody has made; ▶ the
 * closing note of this file.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⭐ THE CONTAINER'S OWN WIDTH, NAMED ONCE. A code is packed into a `uint16`, so 16 is both the
 * container and the one width at which a code IS the weight — which is why it is the width nothing
 * rotates and nothing looks up. ⛳ IT IS A `#define` RATHER THAN A LITERAL because three separate
 * questions answer with it (the container, the un-quantised width, the no-rotation case) and a reader
 * meeting a bare `16` cannot tell which is meant. */
#define NN__TURBOQUANT__CONTAINER_WIDTH  16ull

#define NN__TURBOQUANT__WIDTH_BITS   5u
#define NN__TURBOQUANT__WIDTH_MASK   0x1Fu
#define NN__TURBOQUANT__FAMILY_CODEBOOK  0u   /* ± an entry of the row's own table — the shipped format */
#define NN__TURBOQUANT__FAMILY_UNIFORM   1u   /* ± magnitude × one scale — linear levels, no table       */
#define NN__TURBOQUANT__FAMILY_GAUSSIAN  2u   /* a SIGNED build-constant table × one scale — no sign bit */
#define NN__TURBOQUANT__FAMILIES         3u

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_TURBOQUANT_CUH */
