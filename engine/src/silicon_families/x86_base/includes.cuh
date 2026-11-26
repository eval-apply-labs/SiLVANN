#ifndef SILVANN__SILICON_FAMILIES_X86_BASE_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_X86_BASE_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE CARD THIS FAMILY TAKES: THE CPU ITSELF, WHERE x86_avx2 CANNOT RUN ═══════════════════════════════════════════
 * ⚖ 2026-11-25 *"if we dont have an igpu nor avx2 then it degrades to base avx"*: an x86-64 processor without AVX2, FMA
 * and F16C — Intel's Pentiums and Celerons to the 10th generation, which have no AVX at all, and the processors before
 * Haswell — named `x86_base` by the system's card list (▶ cards_present.cuh) and taken by this family alone: x86_avx2's
 * primitives with nn's generic bodies, compiled for x86-64-v2 (SSE4.2, POPCNT). */
#define X86_BASE__TARGETS(X)  X(x86_base)

static inline bool x86_base__includes(const char* card) {
#define X86_BASE__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    X86_BASE__TARGETS(X86_BASE__ZZPRIVATE_IS)
#undef X86_BASE__ZZPRIVATE_IS
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_X86_BASE_INCLUDES_CUH */
