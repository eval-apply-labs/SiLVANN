#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "kernels.cuh"      /* what the overrides share */
#include "turboquant.cuh"   /* the gemvs: exact, int8, 16-bit, over several matrices, and wide */
#include "hadamard.cuh"     /* the rotations */
#include "attention.cuh"    /* the scores, the mix and two residuals' merge */
#include "deltanet.cuh"     /* the step, a position and a prompt's rows */
#include "expert.cuh"       /* the expert-major GEMM */
#include "primitives.cuh"   /* rmsnorm */
#include "result.cuh"       /* argmax */
#include "vector.cuh"       /* the top k, a router's and a sampler's */
#include "doors.cuh"        /* into nn's table */

/* ⛳ `kernels.cuh` FIRST, since every file uses it, and `doors.cuh` LAST, since it names every door. The groups between
 *   use nothing of each other, and their order is still not free: `MEASURED` (the gfx906 code object, every kernel's
 *   instructions compared), with `primitives.cuh` ahead of `turboquant.cuh` two gemv kernels (`exact_4_h1`, `groups_u1`)
 *   come out rescheduled — the same metadata, other instructions. `REASONED`: the compiler inlines in the order
 *   functions are defined. */

/* ══ ⭐⭐ OVERRIDES — THE DOORS THIS FAMILY RUNS FASTER THAN THE GENERIC BODY ══════════════════════════
 * ⚖ *"a for the family bodies"*, in *"a folder for primitives and one for overrides so it is clear what is
 * expected to run and what are performance upgrades"*. Each function here has its door's signature and
 * takes its place in nn's table (`amd_rocm6_wave64__override_doors`, called from `../entry.cuh` once the
 * table is built). ⛳ EVERY ONE FALLS BACK TO nn's GENERIC DOOR for what it does not cover — another width,
 * a row that is not aligned, an `x` wider than the lanes hold — so an override can only ever be faster,
 * never different in what it accepts, and deleting this file loses nothing but speed. */

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_NN_CUH */
