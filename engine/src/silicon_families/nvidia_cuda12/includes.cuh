#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */
#include "../nvidia_tensor_cores.cuh"   /* a Turing card without tensor cores */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"the method is family.includes(card_name) and it returns true or false."* NVIDIA on CUDA 12: Turing
 * (`sm_75`) through Blackwell — the data centre's B200 (`sm_100`) and B300 (`sm_103`), the GeForce RTX 50s (`sm_120`),
 * the DGX Spark's GB10 (`sm_121`, its memory the Grace CPU's) — each named by the compute capability the binary carries
 * code for. CUDA 12.9 builds every one (the release's
 * toolchain: release/docker/Dockerfile, the Windows job); an older nvcc refuses `sm_103` or `sm_100` (CUDA 12.9 and
 * 12.8 added them), and `SILVANN_CUDA_ARCHES` narrows the build.
 *
 * ⭐ THE LIST IS ALSO WHAT THE FAMILY IS BUILT FOR. `scripts/build_silicon_families.sh` reads its targets
 *   from the row below, so the family cannot say it takes a card its binary has no code for.
 * ⛳ A CARD NOT NAMED HERE IS NOT TAKEN — ⚖ *"we drop the optimistic setup"*. The binary also carries the
 *   last target as PTX, which a newer driver could compile for a newer card; that is a way a new card might
 *   run, not a statement that it does, so it is not a reason to take one. `REASONED`, not measured: this
 *   box has no NVIDIA card. Adding one is a row, checked against NVIDIA's CUDA release notes.
 * ⭐ BUT NOT A TURING CARD WITHOUT TENSOR CORES (`sm_75:tc-`, a GTX 16xx — ▶ `../nvidia_tensor_cores.cuh`): this family's
 *   kernels are the tensor cores', and `nvidia_opencl1` runs such a card faster. */
#define NVIDIA_CUDA12__TARGETS(X)  X(sm_75) X(sm_80) X(sm_86) X(sm_89) X(sm_90) X(sm_100) X(sm_103) X(sm_120) X(sm_121)

static inline bool nvidia_cuda12__includes(const char* card) {
    if (silicon_families__card_flag(card, "tc-")) return false;
#define NVIDIA_CUDA12__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    NVIDIA_CUDA12__TARGETS(NVIDIA_CUDA12__ZZPRIVATE_IS)
#undef NVIDIA_CUDA12__ZZPRIVATE_IS
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_INCLUDES_CUH */
