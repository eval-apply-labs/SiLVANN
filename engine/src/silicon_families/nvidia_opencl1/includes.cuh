#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_OPENCL1_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_OPENCL1_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */
#include "../opencl_versions.cuh"   /* which cards' OpenCL is 1.2 */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"opencl 1.2 will go in the backlog alongside vulkan"* · 2026-11-09 *"old_nvidia does its own thing and does not
 * stomp against the intel old igpu"*: an NVIDIA card through NVIDIA's OpenCL, whose OpenCL C is 1.2 whatever the card —
 * built from `khronos_opencl1_base`. `nvidia_cuda12` takes NVIDIA's cards first, the roster's preference, so a card is
 * this family's when CUDA does not build for it (Kepler to Volta here: CUDA 13 builds none, and the CUDA family is sm_75
 * on), when it is a Turing card without tensor cores (`sm_75:tc-`, a GTX 16xx — ▶ `../nvidia_tensor_cores.cuh`), when
 * CUDA will not open, or when it is asked for by name.
 *
 * ⛳ AN OPENCL BINARY IS NOT BUILT FOR ITS CARDS: the kernels travel as OpenCL C and the runtime compiles
 *   them for whatever card it finds. So the targets row is what the family has been RUN on, not what it could
 *   compile for — a card is added once the verb suite has passed on it through this family; every other NVIDIA
 *   card it takes is a developer preview.
 * ⛳ A CARD IS NAMED BY ITS COMPUTE CAPABILITY (`sm_61`), which NVIDIA's OpenCL answers through
 *   `cl_nv_device_attribute_query`.
 *   `sm_120` the RTX 5060 Ti, the test card; `sm_61` a GTX 1080 Ti (Pascal): nn's verbs 343/0 and the 35B against HF 6/0
 *   (measurements/2026-11-09_ocl1_pascal_read_only.md). */
#define NVIDIA_OPENCL1__TARGETS(X)  X(sm_120) X(sm_61)

static inline bool nvidia_opencl1__includes(const char* card) {
#define NVIDIA_OPENCL1__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    NVIDIA_OPENCL1__TARGETS(NVIDIA_OPENCL1__ZZPRIVATE_IS)
#undef NVIDIA_OPENCL1__ZZPRIVATE_IS
    return silicon_families__opencl1_nvidia(card);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_OPENCL1_INCLUDES_CUH */
