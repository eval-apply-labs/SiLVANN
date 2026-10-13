#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"opencl 1.2 will go in the backlog alongside vulkan"* — for a card whose OpenCL is 1.2: no generic
 * address space, no shared virtual memory, no work-group functions. Older Intel iGPUs are what it is for;
 * NVIDIA's OpenCL is one too (its OpenCL C is 1.2), and is the card it was first run on.
 *
 * ⛳ AN OPENCL BINARY IS NOT BUILT FOR ITS CARDS: the kernels travel as OpenCL C and the runtime compiles
 *   them for whatever card it finds. So this list is what the family has been RUN on, not what it could
 *   compile for — a card is added once the verb suite has passed on it through this family.
 * ⛳ A CARD IS NAMED AS ITS VENDOR'S FAMILY NAMES IT: an NVIDIA card by its compute capability (`sm_120`),
 *   which its OpenCL answers through `cl_nv_device_attribute_query`; any other by its OpenCL device name.
 *   `nvidia_cuda12` takes these cards first — the roster's preference — so this family has one only when
 *   it is asked for by name, or when CUDA will not open. */
#define KHRONOS_OPENCL1_IGPU__TARGETS(X)  X(sm_120)

static inline bool khronos_opencl1_igpu__includes(const char* card) {
#define KHRONOS_OPENCL1_IGPU__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    KHRONOS_OPENCL1_IGPU__TARGETS(KHRONOS_OPENCL1_IGPU__ZZPRIVATE_IS)
#undef KHRONOS_OPENCL1_IGPU__ZZPRIVATE_IS
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_INCLUDES_CUH */
