#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"opencl will fit as the last chance before refusing"*. A card `amd_rocm6_wave64` also takes, reached
 * through ROCm's OpenCL instead of HIP: it is offered here only when every family above has passed on it
 * — a box whose HIP will not open, for one.
 *
 * ⛳ AN OPENCL BINARY IS NOT BUILT FOR ITS CARDS: the kernels travel as OpenCL C and the runtime compiles
 *   them for whatever card it finds. So this list is what the family has been RUN on, not what it could
 *   compile for — a card is added once the verb suite has passed on it through this family. */
#define KHRONOS_OPENCL2__TARGETS(X)  X(gfx906)

static inline bool khronos_opencl2__includes(const char* card) {
#define KHRONOS_OPENCL2__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    KHRONOS_OPENCL2__TARGETS(KHRONOS_OPENCL2__ZZPRIVATE_IS)
#undef KHRONOS_OPENCL2__ZZPRIVATE_IS
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_INCLUDES_CUH */
