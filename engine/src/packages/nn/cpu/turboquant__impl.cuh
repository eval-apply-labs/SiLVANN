#ifndef SILVANN__PACKAGES_NN_CPU_TURBOQUANT__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_TURBOQUANT__IMPL_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "turboquant__header.cuh"               /* the contract these definitions answer */
#include "primitives__header.cuh"        /* the resolver and the bound every primitive shares */
#include "buffer__header.cuh"            /* what a buffer's node holds, and its kind */
#include "../../sys/cpu/list__header.cuh"       /* a form is a list, and the verb reads its arguments */
#include "../../sys/cpu/opcodes/opcodes.cuh"            /* how a verb answers, and how it refuses */
/* ══ nn — VBR, DECODED ═══════════════════════════════════════════════════════════════════════════════
 * ▶ `turboquant__header.cuh` for the format, and for what is deliberately left folded.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ ⭐⭐⭐ THE CONTAINER IS A HALFWORD — ⚖ RULED, AND IT IS A KERNEL RULING ════════════════
 *
 * ⚖ ARCHITECT: *"loading and doing bitmask and bitshift to n bytes to get 8 values is not worth the
 * float hosting 3 or 5 values already and i pay a 9% packing fee to have it in a manageable and
 * accessible state."* He had built the dense alternative before: *"it was very slow."*
 *
 * ⛔⛔ THE ARGUMENT THAT LOST WAS ARITHMETICALLY CORRECT, AND THAT IS WHY IT IS RECORDED. Eight codes
 * is exactly D BYTES at every width, so a group of eight wastes NOTHING — 3.000 b/w at D=3 against
 * 3.200 here, 5.000 against 5.333. Every one of those numbers is right and every one is beside the
 * point: an unaligned group costs a dependent shift chain on the LOAD path, and the load path is what
 * a gemv is made of. ⇒ ★★ A FORMAT ARGUMENT THAT NEVER TOUCHES THE KERNEL IS A CLAIM ABOUT THE WRONG
 * QUANTITY — and being exactly right about bits is what makes such an argument convincing.
 *
 * ⭐ THE FEE IS ONE BIT IN SIXTEEN, AND ONLY AT THE TWO WIDTHS THAT DO NOT DIVIDE 16:
 * ```
 *     D    1    2    3      4    5      8    16
 *   /u16  16    8    5      4    3      2     1
 *    b/w 1.000 2.000 3.200 4.000 5.333 8.000 16.000
 * ```
 * ⛳ AND `D == 16` STOPS BEING A SPECIAL CASE IN THIS FUNCTION: it is "one value a container", which
 * the same arithmetic already answers. Only its INTERPRETATION still branches — the code IS the value
 * rather than an index — and that branch lives in `zzprivate_weight` where it belongs.
 *
 * ⛔⛔ q6 IS GONE, AND THE DIRECTION OF THE ARGUMENT MATTERS: two values in a halfword is what q6 and
 * q8 BOTH get, so q6 costs exactly q8's bytes and carries **15.7x its error** (`MEASURED` on
 * `L3 gate_up e0`: 0.000642 against 0.000041). Dominated — no budget selects it. ⚖ *"i think i should
 * go up to q5 and then move to q8."* ⇒ ★ AT A 32-BIT WORD q6 IS 6.4 b/w AND IS **NOT** DOMINATED, so
 * anyone who reopens the container has to reopen this too.
 * ⛳ q1 IS IN THE SET AND IS NOT SERIOUS — ⚖ *"add q1 for the meme, it should be refuted."* It packs
 * perfectly (16 a container, zero waste) and its table is {-1,+1} with the scale coming out of the
 * Lloyd solve as sqrt(2/pi) = E|X|, i.e. the textbook one-bit quantiser. ~60% rms. Nothing ships here. */


static __device__ inline bool nn__turboquant__rate_is_known(uint64_t d) {
    return d == 1ull || d == 2ull || d == 3ull || d == 4ull || d == 5ull
        || d == 8ull || d == 16ull;
}





static __device__ inline uint64_t nn__turboquant__width(uint64_t rate) {
    return rate & (uint64_t)NN__TURBOQUANT__WIDTH_MASK;
}
static __device__ inline uint64_t nn__turboquant__family(uint64_t rate) {
    return rate >> NN__TURBOQUANT__WIDTH_BITS;
}

/* ⭐⭐ THE FORMAT HAS ONE FAMILY, AND IT STILL CARRIES THE FAMILY FIELD — ⚖ RULED:
 * *"the old vbr is dead, lets move to the new format fully."*
 *
 * ⛔⛔ AND THE VERSION FIELD IS NOT IN THESE BYTES — IT MOVED, AND THE FIRST VERSION OF THIS PARAGRAPH
 * SAID IT HAD NOT. It read *"every row's rate byte is `(FAMILY_GAUSSIAN << 5) | d`"*, which was true
 * of the five-plane format and false of this one the moment `bitrates` was collapsed — asserted in
 * the same commit that deleted it. ⇒ ★★ THE ARGUMENT FOR A FIELD CAN SURVIVE THE FIELD, and no gate
 * here catches it: `src_rot_scan` finds dead SYMBOL NAMES, and this was a true-sounding sentence
 * about live code. **A CHECKER THAT FINDS STALE POINTERS DOES NOT FIND STALE CLAIMS.**
 * ⇒ ★ A SINGLE-FAMILY FORMAT KEEPS ITS VERSION FIELD OR IT HAS NONE: the next table is family 3, and
 * without a tag it would be a silent reinterpretation of bytes already on disk. The tag now lives in
 * the bundle's per-tensor SCALARS, which is the right shape — a per-row field holding a value
 * identical in every row of every tensor was always the wrong one, and it cost 0.0156 b/w to be
 * wrong. ▶ `docs/design/pack_container.md` §②.
 * ⛳ THE `family`/`width` PACKING IS KEPT IN THIS FILE because the bit layout is still what a rate
 * byte WOULD mean, and family 3 will want it. Nothing reads one today.
 *
 * A row's parameter block is ONE fp16 — its scale — for every width except `D == 16`, which is
 * key-as-value and has none. That is why locating a row needs no table of block sizes: the answer is
 * one number, and `zzprivate_row` computes it.
 *
 * ⛔ AND THE READING COST OF THE CHANGE IS REAL: a bundle packed for the previous decoder is not
 * readable here, and `python/compressor.py` is what produces those. Repacking with
 * `python/nn_compressor.py` is the only route in. */
/* ── ⭐ fp16 TO fp32, EXACTLY, IN INTEGERS ───────────────────────────────────────────────────────────
 * Every fp16 is representable in fp32, so this rounds nothing and approximates nothing — it is a
 * rearrangement of bits. ▶ the header on why that keeps it out of the seam.
 * ⛳ THE SUBNORMAL ARM IS NOT DEAD WEIGHT: a codebook entry is a real magnitude and a well-compressed
 * row's smallest bin can land under 2^-14. Getting it wrong would make exactly the smallest weights
 * wrong, which is the direction an error hides best. */
/* ⛳ THE fp16 CONVERSIONS THIS FILE USES ARE THE PACKAGE'S, IN `primitives__impl.cuh`, because every
 * verb in `nn` reads halves and not only the codec. A conversion defined here would make `swiglu`
 * depend on the codec for arithmetic. ▶ that file, which also carries why neither crosses the seam. */



/* ── ⭐⭐ ONE WEIGHT, AND IT IS FACTORED BECAUSE TWO CONSUMERS READ IT ────────────────────────────────
 * The decoder materialises a dense matrix; the GEMV never does. They are different verbs with different
 * costs and they MUST agree about what byte means what — so the five rules live here once.
 * ⇒ ★ TWO IMPLEMENTATIONS OF A BIT LAYOUT IS THE DEFECT THE LAYOUT'S OWN ORACLE CANNOT CATCH, because
 * each would be checked against the packer separately and both could pass while disagreeing about a
 * case neither fixture reaches. */




/* ⛳ THE FAULT A BAD WIDTH RAISES, PICKED ONCE SO THE TWO VERBS CANNOT DISAGREE ABOUT IT. */
static __device__ inline uint64_t nn__turboquant__zzpackage_fault_for(uint64_t d) {
    return nn__turboquant__rate_is_known(d) ? NN__TURBOQUANT__FAULT_BOUNDS : NN__TURBOQUANT__FAULT_RATE;
}

#endif /* SILVANN__PACKAGES_NN_CPU_TURBOQUANT__IMPL_CUH */
