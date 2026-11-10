#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_UNIT_CUH

/* ══ THE WHOLE OF amd_rocm6_wave64, AS ONE TRANSLATION UNIT ═══════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with `hipcc` into `libsilvann_amd_rocm6_wave64.so`: this family's
 * answers, every package's card side written against them, and the one entry point. ⛳ It is a header and
 * not a `.cu` because the tree holds exactly one `.cu`, the program's; the build writes a one-line `.cu`
 * that includes this. */
#include <hip/hip_runtime.h>
#include "includes.cuh"                           /* this family's cards — the MI50's gfx906 */
/* ⭐ BUILT FROM `amd_rocm_base` (▶ its manifest): the source's answers, this family's cards and name */
#define AMD_ROCM_BASE__TAKES(card)  amd_rocm6_wave64__includes(card)
#define AMD_ROCM_BASE__NAME         "amd_rocm6_wave64"
#include "../amd_rocm_base/manifest.cuh"          /* the answers to every package the AMD families support */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#include "../amd_rocm_base/overrides/nn.cuh"      /* the doors run faster than the generic body */
#define SYS__SILICON_FAMILY__THIS amd_rocm_base   /* the doors' names: the source's */
#define SYS__SILICON_FAMILY__IS   amd_rocm6_wave64  /* the family it is: its number and its name */
/* the hook is handed every package's door table, by package id; the one this family overrides is nn's */
#define SYS__SILICON_FAMILY__OVERRIDE_DOORS(TABLES)  amd_rocm_base__override_doors((nn__doors*)(uintptr_t)(TABLES)[nn_pkg_id])
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_UNIT_CUH */
