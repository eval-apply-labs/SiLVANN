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
 * use (▶ amd_rocm_base/overrides/expert.cuh). Both AMD families are built from `amd_rocm_base`. ROCm Core SDK 10's
 * matrix lists gfx908 to gfx950 (`MEASURED` node03, compile only, 2026-10-29: today's source builds for gfx90a, gfx942
 * and gfx950); gfx950 runs here (`MEASURED` a Daytona MI355X, TheRock's ROCm 7.14,
 * measurements/2026-11-08_amd_mfma_tiles_mi355x.md).
 *
 * ⭐ AND THE MI50, BESIDE ROCm 6's FAMILY (⚖ 2026-11-23 *"yes i'd add 906 to rocm7 and maintain the rocm6 version"*):
 *   `gfx906` — the MI50 and MI60, the Radeon VII — is a target of this family and of `amd_rocm6_wave64` both. AMD's
 *   matrix for ROCm 7 does not list the card; ROCm 7.2.4's compiler builds for it and its runtime drives it all the
 *   same, with no override of the card's name. `MEASURED` node03, two MI50, 2026-11-23
 *   (measurements/2026-11-23_nn206_gfx906_rocm7_node03.md): this family's build against ROCm 6.1.3's of
 *   `amd_rocm6_wave64` — the door bench 467 checks with either and 279 of its 280 outputs the same bytes, a model's
 *   logits the same to the bit, the same tokens at the rates there. A card that two families include goes to the
 *   higher one whose runtime is installed (▶ ../choose.cuh): this one where the machine has ROCm 7, ROCm 6's where it
 *   has ROCm 6 — so a machine whose kernel ROCm 6.1.3's runtime does not take has a family for the card.
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
#define AMD_ROCM10__TARGETS(X)  X(gfx906) X(gfx908) X(gfx90a) X(gfx942) X(gfx950) X(gfx1200) X(gfx1201)

/* ⛳ A WINDOWS BUILD HAS A ROW OF ITS OWN, WITHOUT `gfx906` (⚖ 2026-11-23 *"go with the linux-only list for now, and the
 *   next release will need to work on adding that card on windows through hip, even if it is untested"*). Every Windows
 *   library compiles the function below through the roster, and a line of it that changes is a signature paid for each
 *   of them (`release/scripts/sign_plan.py`, the unit key): the row is the one those libraries are signed with, to the
 *   token. ⭐ AND IT IS WHAT THE FAMILY IS BUILT FOR ON WINDOWS: `release/scripts/build.py` reads its targets from it —
 *   one list there too. AMD's requirements for ROCm on Windows name the Radeons and no MI card (the HIP SDK 7.2.0's
 *   and ROCm 10.2's pages, read 2026-11-23): `gfx1200` and `gfx1201` are the cards a Windows PC can hold, and the
 *   other four are built and never met. `ASSUMED`, no card of them has run it on Windows: what a Linux build of the
 *   same source does on the same cards, a Windows build does. A card says. */
#define AMD_ROCM10__TARGETS_ON_WINDOWS(X)  X(gfx908) X(gfx90a) X(gfx942) X(gfx950) X(gfx1200) X(gfx1201)

static inline bool amd_rocm10__includes(const char* card) {
#define AMD_ROCM10__ZZPRIVATE_IS(T)  if (silicon_families__card_is(card, #T)) return true;
#if defined(_WIN32)
    AMD_ROCM10__TARGETS_ON_WINDOWS(AMD_ROCM10__ZZPRIVATE_IS)
#else
    AMD_ROCM10__TARGETS(AMD_ROCM10__ZZPRIVATE_IS)
#endif
#undef AMD_ROCM10__ZZPRIVATE_IS
    return false;
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM10_INCLUDES_CUH */
