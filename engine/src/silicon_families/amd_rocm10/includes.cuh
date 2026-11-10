#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM10_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM10_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ 2026-10-29 *"the rocm6 is a quirk needed by the mi50 and should not hold back the current cards. so the amd path needs
 * the latest rocm and adding the tensor core optimizations of the latest cards"* · *"shared source, call it amd_rocm10"* ·
 * 11-08 *"the legacy amd on rocm6 and the modern one on rocm7"*. AMD's data-centre line from the MI100 on, built with ROCm 7
 * and later (HIP 7 — TheRock's and ROCm Core SDK 10's), 64 lanes to a wavefront: the MI100 (`gfx908`), the MI200s
 * (`gfx90a`), the MI300s (`gfx942`) and the MI350s (`gfx950`) — the cards with matrix cores, which the prompt's GEMM tiles
 * use (▶ amd_rocm_base/overrides/expert.cuh). The MI50 (`gfx906`) is `amd_rocm6_wave64`'s; both are built from
 * `amd_rocm_base`. ROCm Core SDK 10's matrix lists gfx908 to gfx950 and not gfx906 (`MEASURED` node03, compile only,
 * 2026-10-29: today's source builds for gfx90a, gfx942 and gfx950); gfx950 runs here (`MEASURED` a Daytona MI355X,
 * TheRock's ROCm 7.14, measurements/2026-11-08_amd_mfma_tiles_mi355x.md).
 *
 * ⭐ AND THE RADEONS OF RDNA4, IN WAVE64 (⚖ 2026-11-10 *"maybe we can use it to build the home user amd setup … recycling the
 *   amdmi355x in wave64 mode"*): a consumer card runs 32 lanes to a wavefront by default and 64 when built with
 *   `-mwavefrontsize64`, which `scripts/build_silicon_families.sh` gives this family — the MI cards run 64 alone, and their
 *   instructions are the same with it. ROCm 7.2's RDNA4 Radeons on Linux: `gfx1200` (the RX 9060s), `gfx1201` (the RX
 *   9070s, the R9700). Their matrix instructions are WMMA, not the MI cards' MFMA, so the prompt's GEMM takes the plain
 *   tiles there; their dot instructions are the family's (▶ amd_rocm_base/nn.cuh). `MEASURED` an RX 9070 XT (gfx1201) on
 *   ROCm 7.2 under WSL2, measurements/2026-11-10_amd_radeon_wave64.md; gfx1200 is built and has not run.
 * ⛔ NOT RDNA3 (`gfx1100`, `gfx1101`): built in wave64 there, a shuffle reaches only its own half of the wave's 64 lanes, and
 *   the family's answers are wrong — `MEASURED` an RX 7900 XTX, ROCm 7.2: nn's verbs and the door bench far off nn's
 *   host body, the 4B's answer nonsense; the shuffles made whole by `v_permlane64` still left them wrong (the same record).
 *   RDNA3 runs OpenCL 2.0's (NN-172).
 *
 * ⭐ THE LIST IS ALSO WHAT THE FAMILY IS BUILT FOR. `scripts/build_silicon_families.sh` reads its targets from the row
 *   below. ⛳ A CARD NOT NAMED HERE IS NOT TAKEN: AMD code is built for exact targets — an RDNA3 or RDNA2 card goes to
 *   OpenCL 2.0. */
#define AMD_ROCM10__TARGETS(X)  X(gfx908) X(gfx90a) X(gfx942) X(gfx950) X(gfx1200) X(gfx1201)

static inline bool amd_rocm10__includes(const char* card) {
#define AMD_ROCM10__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
    AMD_ROCM10__TARGETS(AMD_ROCM10__ZZPRIVATE_IS)
#undef AMD_ROCM10__ZZPRIVATE_IS
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM10_INCLUDES_CUH */
