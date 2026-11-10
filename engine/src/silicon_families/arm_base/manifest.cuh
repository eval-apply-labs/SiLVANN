#ifndef SILVANN__SILICON_FAMILIES_ARM_BASE_MANIFEST_CUH
#define SILVANN__SILICON_FAMILIES_ARM_BASE_MANIFEST_CUH

/* ══ arm_base — THE ARM PROCESSORS' ANSWERS TO EVERY PACKAGE THEY SUPPORT ══════════════════════════════════════════
 * ⚖ *"only through a manifest so i dont have to keep track of them"* · *"a folder for primitives and one for overrides so
 * it is clear what is expected to run and what are performance upgrades"*: `primitives/` is what a family must answer —
 * the seam: the memory, the pool a launch's blocks run on, the arithmetic — and `overrides/` a door's faster body.
 * ⭐ ONE SOURCE, SEVERAL FAMILIES (⚖ 2026-11-09 *"make the arm_base silicon platform and then add the overrides for that
 *   specific dgx sparx platform"* · *"it might be good for porting it to mobile phones as well"*): not a family of the
 *   roster — `arm_neon` (any Armv8.2 processor with half precision and the dot product: phones since 2018, Graviton,
 *   Ampere) and `arm_gb10` (the DGX Spark's GB10: ten Cortex-X925 and ten Cortex-A725, Armv9.2 with SVE2 and i8mm) are
 *   each built from it by their own `unit.cuh`, which names the family, what its processors must have
 *   (`ARM_BASE__CAPABLE`) and the flags it is compiled with (`scripts/build_silicon_families.sh`). The primitives are
 *   `x86_avx2`'s, the processor's own half conversion and its `yield` in place of F16C's and `pause`. */
#include "primitives/sys.cuh"   /* sys's doors */
#include "primitives/nn.cuh"    /* nn's requests */

#endif /* SILVANN__SILICON_FAMILIES_ARM_BASE_MANIFEST_CUH */
