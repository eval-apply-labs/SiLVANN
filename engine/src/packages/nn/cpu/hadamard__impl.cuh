#ifndef SILVANN__PACKAGES_NN_CPU_HADAMARD__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_HADAMARD__IMPL_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "hadamard__header.cuh"         /* the contract these definitions answer */
#include "primitives__header.cuh"       /* the resolver and the bound every primitive shares */
#include "buffer__header.cuh"           /* what a buffer's node holds, and its kind */
#include "../../sys/cpu/list__header.cuh"      /* a form is a list, and the verb reads its arguments */
#include "../../sys/cpu/opcodes/opcodes.cuh"           /* how a verb answers, and how it refuses */
#include "../../sys/cpu/silicon/silicon__header.cuh"   /* the one square root this package is allowed to take */
/* ══ nn — THE ROTATION, APPLIED ══════════════════════════════════════════════════════════════════════
 * ▶ `hadamard__header.cuh` for what it is, why it is one verb, and the ruling it is waiting on.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


/* ⛔ A LENGTH THE BUTTERFLY CAN WALK. Written as a halving loop and not as a bit trick, because this
 * package uses no bit intrinsics anywhere — the tree compiles twice and `__popc` has no counterpart on
 * the other side of the seam. ⛳ `0` is refused: the transform of nothing is not the empty answer, it is
 * a caller who has lost track of a length. */
/* ⛳ `NN__HADAMARD__WHOLE_BELOW` ITSELF IS IN `hadamard__header.cuh` — `kernels.cuh` reads it too,
 * and a constant two files read is a header's, not an implementation's. */

static __device__ inline bool nn__hadamard__zzprivate_is_pow2(uint64_t n) {
    if (n == 0ull) return false;
    while ((n & 1ull) == 0ull) n >>= 1;
    return n == 1ull;
}

/* ══ ⭐⭐⭐ THE BLOCK LADDER — 512s, THEN ONE 256, THEN ONE 128 ════════════════════════════════════════
 *
 * ⚖ RULED: *"id say we do blocks of 512 and at the end it is blocks of 256 then eventually
 * 128 and the rest is done by padding"*, and ⚖ *"yea promote"*.
 *
 * ⛔⛔ THIS EXISTS BECAUSE THE PRODUCT ONLY SURVIVES IF BOTH SIDES ROTATE THE SAME WAY. The codec relies
 * on `w'·x' = (Rw)·(Rx) = w·x`, which holds only when the activation's `R` is the weights' `R` — so the
 * packer's structure and this verb's structure are ONE FACT, and it is derived from the width on both
 * sides rather than carried in the record. ⇒ ★ A DERIVED STRUCTURE IS THE ONE KIND THAT CANNOT DRIFT;
 * a field could be written wrong, and a wrong one decodes CORRECTLY and computes a WRONG PRODUCT.
 * ▶ `python/nn_compressor.py`'s `blocks_for`, which is the same ladder in the same order.
 *
 * ⭐⭐ AND THE CLAUSE THAT MAKES THE RULE TOTAL IS THE SMALL ONE: **below 512 the whole width is one
 * transform.** Without it the ladder places nothing of a 4-wide vector, and 4-wide vectors are what every
 * fixture rotates — so this verb would have had to grow an argument, and an argument is a second place
 * the two sides can disagree. ⇒ ★ NAMING THE SMALL WIDTHS IS WHAT KEPT THE SIGNATURE ALONE.
 *
 * ⛳ WHY IT IS ALSO CHEAPER, WHICH WAS NOT THE REASON: a transform of width `s` is `log2(s)` passes, so a
 * vector of `n` costs `n·log2(s)` — smaller blocks do LESS work. `MEASURED` 22,528 -> 18,432 element-ops
 * at n=2048 and 49,152 -> 36,864 at n=4096. ⚠ It is ~0.25% of the matvec it feeds, so this is free and

/* Can the ladder place every element of `n`? Asked BEFORE any work, so a refusal never leaves a
 * half-rotated buffer behind. ⛳ It walks the same three widths the worker does rather than computing a
 * remainder cleverly, because the two must agree and the cheapest way to guarantee that is one shape. */
static __device__ inline bool nn__hadamard__zzpackage_ladder_places(uint64_t n) {
    if (n < NN__HADAMARD__WHOLE_BELOW) return nn__hadamard__zzprivate_is_pow2(n);
    uint64_t rest = n % NN__HADAMARD__WHOLE_BELOW;
    if (rest >= 256ull) rest -= 256ull;
    if (rest >= 128ull) rest -= 128ull;
    return rest == 0ull;
}

#endif /* SILVANN__PACKAGES_NN_CPU_HADAMARD__IMPL_CUH */
