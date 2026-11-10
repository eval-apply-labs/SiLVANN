#ifndef SILVANN__SILICON_FAMILIES_ARM_GB10_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_ARM_GB10_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE CARD THIS FAMILY TAKES: THE PROCESSORS ═══════════════════════════════════════════════════════════
 * The DGX Spark's GB10 — ten Cortex-X925 (part 0xd85) and ten Cortex-A725 (0xd87), Armv9.2 with SVE2, i8mm
 * and bf16 — named `arm_gb10` by `cards_present.cuh`. One card for the whole machine, as `x86_avx2`'s: its nodes and cores are the family's own pool. */
#define ARM_GB10__TARGETS(X)  X(arm_gb10)

static inline bool arm_gb10__includes(const char* card) {
    return silicon_families__card_is(card, "arm_gb10");
}

#endif /* SILVANN__SILICON_FAMILIES_ARM_GB10_INCLUDES_CUH */
