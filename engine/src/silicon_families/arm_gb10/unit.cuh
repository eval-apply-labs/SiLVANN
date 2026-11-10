#ifndef SILVANN__SILICON_FAMILIES_ARM_GB10_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_ARM_GB10_UNIT_CUH

/* ══ THE WHOLE OF arm_gb10, AS ONE TRANSLATION UNIT ═══════════════════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with `g++ -march=armv9.2-a+sve2+i8mm+bf16` into `libsilvann_arm_gb10.so`:
 * the Arm families' answers (`arm_base`), every package's card side written against them, and the one entry point.
 * ⛳ A BLOCK IS A THREAD OF THIS PROCESS, so no kernel is compiled: nn's doors pack their arguments into a struct and hand
 *   the launch a block function (`SYS__SILICON_FAMILY__RUNS_ON_HOST_THREADS`), which the pool runs once for every block. */
#include "../../packages/sys/cpu/silicon/environment.cuh"
#define SYS__SILICON_FAMILY__RUNS_ON_HOST_THREADS 1
#include <sys/auxv.h>                             /* the processor's features, as the kernel reports them */
#include <asm/hwcap.h>
#include "includes.cuh"                           /* this family's card — the processors */
/* ⭐ BUILT FROM `arm_base` (▶ its manifest): what this family's flags assume — Armv8.2's half precision and dot product,
 *   and Armv9's SVE2, i8mm and bf16 — the kernel says the processors have */
#define ARM_BASE__CAPABLE()  ((getauxval(AT_HWCAP) & (HWCAP_FPHP | HWCAP_ASIMDHP | HWCAP_ASIMDDP)) ==               \
                                  (HWCAP_FPHP | HWCAP_ASIMDHP | HWCAP_ASIMDDP)                                      \
                              && (getauxval(AT_HWCAP2) & (HWCAP2_SVE2 | HWCAP2_I8MM | HWCAP2_BF16)) ==             \
                                  (HWCAP2_SVE2 | HWCAP2_I8MM | HWCAP2_BF16))
#include "../arm_base/manifest.cuh"               /* the answers to every package the Arm families support */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#include "overrides/nn.cuh"                       /* the doors run faster than the generic body: arm_base's, then its own */
#define SYS__SILICON_FAMILY__THIS arm_base        /* the doors' names: the source's */
#define SYS__SILICON_FAMILY__IS   arm_gb10        /* the family it is: its number and its name */
/* the hook is handed every package's door table, by package id; the one this family overrides is nn's */
#define SYS__SILICON_FAMILY__OVERRIDE_DOORS(TABLES)  arm_gb10__override_doors((nn__doors*)(uintptr_t)(TABLES)[nn_pkg_id])
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_ARM_GB10_UNIT_CUH */
