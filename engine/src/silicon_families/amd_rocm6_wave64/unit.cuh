#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_UNIT_CUH

/* ══ THE WHOLE OF amd_rocm6_wave64, AS ONE TRANSLATION UNIT ═══════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with `hipcc` into `libsilvann_amd_rocm6_wave64.so`: this family's
 * answers, every package's card side written against them, and the one entry point. ⛳ It is a header and
 * not a `.cu` because the tree holds exactly one `.cu`, the program's; the build writes a one-line `.cu`
 * that includes this. */
#include <hip/hip_runtime.h>
#include "manifest.cuh"                           /* this family's answers to every package it supports */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#include "overrides/nn.cuh"                       /* the doors this family runs faster than the generic body */
#define SYS__SILICON_FAMILY__THIS amd_rocm6_wave64
/* the hook is handed every package's door table, by package id; the one this family overrides is nn's */
#define SYS__SILICON_FAMILY__OVERRIDE_DOORS(TABLES)  amd_rocm6_wave64__override_doors((nn__doors*)(uintptr_t)(TABLES)[nn_pkg_id])
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_UNIT_CUH */
