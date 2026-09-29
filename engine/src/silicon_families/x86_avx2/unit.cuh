#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_UNIT_CUH

/* ══ THE WHOLE OF x86_avx2, AS ONE TRANSLATION UNIT ═══════════════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with `g++ -mavx2 -mfma -mf16c` into
 * `libsilvann_x86_avx2.so`. ⛳ A BLOCK IS A THREAD OF THIS PROCESS, so no kernel is compiled: nn's doors
 * pack their arguments into a struct and hand the launch a block function
 * (`SYS__SILICON_FAMILY__RUNS_ON_HOST_THREADS`, in nn's `doors.cuh`), which the pool runs once for every block. */
#include "../../packages/sys/cpu/silicon/environment.cuh"
#define SYS__SILICON_FAMILY__RUNS_ON_HOST_THREADS 1
#include "manifest.cuh"                           /* this family's primitives */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#include "overrides/nn.cuh"                       /* the doors this family runs faster than the generic body */
#define SYS__SILICON_FAMILY__THIS x86_avx2
/* the hook is handed every package's door table, by package id; the one this family overrides is nn's */
#define SYS__SILICON_FAMILY__OVERRIDE_DOORS(TABLES)  x86_avx2__override_doors((nn__doors*)(uintptr_t)(TABLES)[nn_pkg_id])
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_UNIT_CUH */
