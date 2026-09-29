#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"the method is family.includes(card_name) and it returns true or false."* NVIDIA on CUDA 12: Turing
 * (`sm_75`) through Blackwell (`sm_120`), each named by the compute capability the binary carries code for.
 *
 * ⭐ THE LIST IS ALSO WHAT THE FAMILY IS BUILT FOR. `scripts/build_silicon_families.sh` reads its targets
 *   from the row below, so the family cannot say it takes a card its binary has no code for.
 * ⛳ A CARD NOT NAMED HERE IS NOT TAKEN — ⚖ *"we drop the optimistic setup"*. The binary also carries the
 *   last target as PTX, which a newer driver could compile for a newer card; that is a way a new card might
 *   run, not a statement that it does, so it is not a reason to take one. `REASONED`, not measured: this
 *   box has no NVIDIA card. Adding one is a row, checked against NVIDIA's CUDA release notes. */
#define NVIDIA_CUDA12__TARGETS(X)  X(sm_75) X(sm_80) X(sm_86) X(sm_89) X(sm_90) X(sm_120)

static inline bool nvidia_cuda12__includes(const char* card) {
#define NVIDIA_CUDA12__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    NVIDIA_CUDA12__TARGETS(NVIDIA_CUDA12__ZZPRIVATE_IS)
#undef NVIDIA_CUDA12__ZZPRIVATE_IS
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_INCLUDES_CUH */
