#ifndef SILVANN__SILICON_FAMILIES_INTEL_OPENCL2_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_INTEL_OPENCL2_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */
#include "../opencl_versions.cuh"   /* which cards' OpenCL is 1.2 */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ 2026-11-09 *"opencl should have opencl12 and opencl2 as their base, and then have subclasses … the opencl2 can either
 * be the base implementation or the cool intel one"*: Intel's GPUs whose OpenCL is 2.0 — the integrated ones from Gen9
 * (HD / UHD 5xx 6xx 7xx, Iris Xe) and the discrete Arc and Max — named by their OpenCL device name (`Intel(R) …`), built
 * from `khronos_opencl2_base`. Its number is above `khronos_opencl2`'s, so Intel's cards are offered here first: the
 * place for what Intel's OpenCL has and the base does not use. Today it is the base, its kernels the same. Intel's
 * before Gen9 are `intel_opencl1_igpu`'s.
 *
 * ⛳ AN OPENCL BINARY IS NOT BUILT FOR ITS CARDS: the targets row is what the family has been RUN on — none yet, so
 *   every card it takes is a developer preview until nn's verbs pass on one through it. */
#define INTEL_OPENCL2__TARGETS(X)

static inline bool intel_opencl2__includes(const char* card) {
    return silicon_families__name_starts(card, "Intel(R)") && !silicon_families__opencl1_only(card);
}

#endif /* SILVANN__SILICON_FAMILIES_INTEL_OPENCL2_INCLUDES_CUH */
