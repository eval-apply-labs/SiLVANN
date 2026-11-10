#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM_BASE_MANIFEST_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM_BASE_MANIFEST_CUH

/* ══ amd_rocm_base — THE AMD FAMILIES' ANSWERS TO EVERY PACKAGE THEY SUPPORT ══════════════════════════════════════
 * ⚖ *"only through a manifest so i dont have to keep track of them"*. One `<package>.cuh` per package this
 * family answers, and nothing else: whoever compiles the family includes this file and never names what
 * is in it, so a package the family comes to support is added here and nowhere else.
 * ⭐ ONE SOURCE, TWO FAMILIES (⚖ 2026-10-29 *"amd_rocm_base package that hosts the shared code, and is then
 *   "inherited" by the specific rocm6 and rocm10 engines"*; 11-08 *"the legacy amd on rocm6 and the modern one on
 *   rocm7"*): not a family of the roster — `amd_rocm6_wave64` (the MI50, ROCm 6) and `amd_rocm10` (gfx908 and on, ROCm 7
 *   and later) are each built from it by their own `unit.cuh`, which names the family, its cards and its name. */
#include "sys.cuh"   /* sys's doors */
#include "nn.cuh"    /* nn's requests */

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM_BASE_MANIFEST_CUH */
