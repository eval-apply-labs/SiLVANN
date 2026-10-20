#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */
#include "../opencl_versions.cuh"   /* which cards' OpenCL is 1.2 */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"opencl will fit as the last chance before refusing"*. A card `amd_rocm6_wave64` also takes, reached
 * through ROCm's OpenCL instead of HIP: it is offered here only when every family above has passed on it
 * — a box whose HIP will not open, for one.
 *
 * ⛳ AN OPENCL BINARY IS NOT BUILT FOR ITS CARDS: the kernels travel as OpenCL C and the runtime compiles
 *   them for whatever card it finds. So this list is what the family has been RUN on, not what it could
 *   compile for — a card is added once the verb suite has passed on it through this family.
 * ⭐ AND IT TAKES ANY GPU NO FAMILY ABOVE IT TAKES. ⚖ 2026-10-18, the architect: *"it is exactly where igpus are going
 *   to be the most popular so it is worth it to have opencl2 at least, so you can also support amd"*, and of which
 *   cards: *"any GPU, last chance"*. Every card but NVIDIA's (`sm_*`: NVIDIA's own family takes them, and NVIDIA's
 *   OpenCL has no fine-grained SVM) and the processors' (`x86_avx2`, `host`) is this family's, and runs here when its OpenCL is
 *   2.0 with fine-grained SVM (`device_count`) — an AMD or Intel GPU on Windows, named by OpenCL's loader. A card off
 *   the list is a developer preview. One without fine-grained SVM is reported, not run: the 1.2 family is of the same
 *   vendor, so it is not asked once this one is open — so a card whose OpenCL is 1.2 (NN-96's best guess,
 *   `silicon_families__opencl1_only`: every NVIDIA card, Intel before Gen9, AMD's TeraScale and GCN 1) is never this
 *   family's, and goes to `khronos_opencl1_igpu`. */
#define KHRONOS_OPENCL2__TARGETS(X)  X(gfx906)

static inline bool khronos_opencl2__includes(const char* card) {
#define KHRONOS_OPENCL2__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    KHRONOS_OPENCL2__TARGETS(KHRONOS_OPENCL2__ZZPRIVATE_IS)
#undef KHRONOS_OPENCL2__ZZPRIVATE_IS
    if (card == 0 || card[0] == 0 || silicon_families__card_is(card, "x86_avx2") || silicon_families__card_is(card, "host"))
        return false;
    return !silicon_families__opencl1_only(card);
}

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_INCLUDES_CUH */
