#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE CARD THIS FAMILY TAKES: THE CPU ITSELF ══════════════════════════════════════════════════════
 * The machine's processors, named `x86_avx2` by `cards_present.cuh` when they have AVX2, FMA and F16C —
 * what this family's bodies are compiled for. One card for the whole machine: its sockets and cores are
 * the family's own pool (`primitives/pool.cuh`), not cards of their own. */
#define X86_AVX2__TARGETS(X)  X(x86_avx2)

static inline bool x86_avx2__includes(const char* card) {
#define X86_AVX2__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    X86_AVX2__TARGETS(X86_AVX2__ZZPRIVATE_IS)
#undef X86_AVX2__ZZPRIVATE_IS
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_INCLUDES_CUH */
