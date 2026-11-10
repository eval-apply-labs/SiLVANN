#ifndef SILVANN__SILICON_FAMILIES_ARM_NEON_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_ARM_NEON_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE CARD THIS FAMILY TAKES: THE PROCESSORS ═══════════════════════════════════════════════════════════
 * Any Arm processor of Armv8.2 with half precision and the dot product (`asimdhp`, `asimddp`) — phones since
 * 2018, Graviton, Ampere — named `arm_neon` by `cards_present.cuh`; and `arm_gb10`'s processors, where that family's
 * binary will not open. One card for the whole machine, as `x86_avx2`'s: its nodes and cores are the family's own pool. */
#define ARM_NEON__TARGETS(X)  X(arm_neon)

static inline bool arm_neon__includes(const char* card) {
    return silicon_families__card_is(card, "arm_neon") || silicon_families__card_is(card, "arm_gb10");
}

#endif /* SILVANN__SILICON_FAMILIES_ARM_NEON_INCLUDES_CUH */
