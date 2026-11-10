#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM10_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM10_UNIT_CUH

/* ══ THE WHOLE OF amd_rocm10, AS ONE TRANSLATION UNIT ═════════════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with ROCm 7's `hipcc` into `libsilvann_amd_rocm10.so`: the AMD families'
 * answers (`amd_rocm_base`), every package's card side written against them, and the one entry point — the cards from the
 * MI100 on, their matrix cores where the base's kernels have a path for them. ⛳ A header and not a `.cu`, as every
 * family's: the build writes the one-line `.cu` that includes it. */
#include <hip/hip_runtime.h>
#include "includes.cuh"                           /* this family's cards — gfx908 and on */
/* ⭐ BUILT FROM `amd_rocm_base` (▶ its manifest): the source's answers, this family's cards and name */
#define AMD_ROCM_BASE__TAKES(card)  amd_rocm10__includes(card)
#define AMD_ROCM_BASE__NAME         "amd_rocm10"
#include "../amd_rocm_base/manifest.cuh"          /* the answers to every package the AMD families support */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#include "../amd_rocm_base/overrides/nn.cuh"      /* the doors run faster than the generic body */
#define SYS__SILICON_FAMILY__THIS amd_rocm_base   /* the doors' names: the source's */
#define SYS__SILICON_FAMILY__IS   amd_rocm10      /* the family it is: its number and its name */
/* the hook is handed every package's door table, by package id; the one this family overrides is nn's */
#define SYS__SILICON_FAMILY__OVERRIDE_DOORS(TABLES)  amd_rocm_base__override_doors((nn__doors*)(uintptr_t)(TABLES)[nn_pkg_id])
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM10_UNIT_CUH */
