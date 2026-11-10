#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"the method is family.includes(card_name) and it returns true or false."* AMD's MI50 and MI60 (`gfx906`) on ROCm 6,
 * the ROCm that still builds them, 64 lanes to a wavefront — the legacy family (⚖ 2026-11-08 *"the legacy amd on rocm6 and
 * the modern one on rocm7"*; 10-29 *"rocm stays only on gfx906"*). The MI100 on (`gfx908`, `gfx90a`, `gfx942`, `gfx950`)
 * are `amd_rocm10`'s, ROCm 7 and later's, their matrix cores its own; both are built from `amd_rocm_base`. So are the
 * Radeons of RDNA4, which ROCm 7.2 builds in wave64 (▶ amd_rocm10/includes.cuh): ROCm 6.1's HIP refuses 64 lanes
 * on a consumer card (`gfx1100`, "HIP is not supported on the specified GPU ARCH with wavefront size 64" — `MEASURED`
 * node03, hipcc 6.1.3, 2026-10-29), so this family takes none.
 *
 * ⭐ THE LIST IS ALSO WHAT THE FAMILY IS BUILT FOR. `scripts/build_silicon_families.sh` reads its targets
 *   from the row below, so the family cannot say it takes a card its binary has no code for.
 *   ⚠ `SILVANN_HIP_ARCHES` still narrows a development build to fewer targets; a binary built that way
 *   takes cards it cannot run, and it is not a binary to ship.
 * ⛳ A CARD NOT NAMED HERE IS NOT TAKEN, even a newer one of the same line: AMD code is built for exact
 *   targets, so a card with none of them in the binary would fail at its first launch rather than here.
 *   Adding one is a row, checked against AMD's ROCm compatibility matrix. */
#define AMD_ROCM6_WAVE64__TARGETS(X)  X(gfx906)

static inline bool amd_rocm6_wave64__includes(const char* card) {
#define AMD_ROCM6_WAVE64__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    AMD_ROCM6_WAVE64__TARGETS(AMD_ROCM6_WAVE64__ZZPRIVATE_IS)
#undef AMD_ROCM6_WAVE64__ZZPRIVATE_IS
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_INCLUDES_CUH */
