#ifndef SILVANN__SILICON_FAMILIES_INTEL_OPENCL2_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_INTEL_OPENCL2_UNIT_CUH

/* ══ THE WHOLE OF intel_opencl2, AS ONE TRANSLATION UNIT ══════════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with the host's C++ compiler into
 * `libsilvann_intel_opencl2.so`, linked against the OpenCL loader: the OpenCL 2.0 families' answers
 * (`khronos_opencl2_base`), every package's host side of its card work, and the one entry point. The kernels are not in it as code — they
 * are OpenCL C, generated beside it and carried as a string (`nn.cuh`).
 * ⛳ `SYS__SILICON_FAMILY__KERNELS_BY_NAME` IS WHAT MAKES THAT SO: nn's doors then compile no kernel and
 *   launch by name. `environment.cuh` comes first, as in every host translation unit, because it is what
 *   defines `__device__` away. */
#include "../../packages/sys/cpu/silicon/environment.cuh"
#define SYS__SILICON_FAMILY__KERNELS_BY_NAME 1
#include "includes.cuh"                           /* this family's cards — Intel's from Gen9 */
/* ⭐ BUILT FROM `khronos_opencl2_base` (▶ its manifest): the source's answers, this family's cards and name */
#define KHRONOS_OPENCL2_BASE__TAKES(card)  intel_opencl2__includes(card)
#define KHRONOS_OPENCL2_BASE__NAME         "intel_opencl2"
/* ⭐ ALLOCATIONS PAST 4 GB, Intel's way (its compute runtime's "allocations greater than 4GB"): the allocation asked for with
 *   `CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL` (bit 23) where it is larger than the device's largest, and the kernels built
 *   for buffers past 4 GB — a model's experts in RAM are one allocation of their worker's, many gigabytes (NN-142).
 *   `ASSUMED` from Intel's documentation, never run on an Intel GPU: what would falsify it is that allocation refused, or
 *   the program not built, on a Gen9 GPU — the first laptop to boot the experts on one says. */
#define KHRONOS_OPENCL2_BASE__LARGE_FLAG     (1u << 23)
#define KHRONOS_OPENCL2_BASE__BUILD_OPTIONS  " -cl-intel-greater-than-4GB-buffer-required"
#include "../khronos_opencl2_base/manifest.cuh"   /* the answers to every package the OpenCL 2.0 families support */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#include "../khronos_opencl2_base/overrides/nn.cuh"   /* the doors run faster than the generic body */
#define SYS__SILICON_FAMILY__THIS khronos_opencl2_base   /* the doors' names: the source's */
#define SYS__SILICON_FAMILY__IS   intel_opencl2          /* the family it is: its number and its name */
/* the hook is handed every package's door table, by package id; the one this family overrides is nn's */
#define SYS__SILICON_FAMILY__OVERRIDE_DOORS(TABLES)  khronos_opencl2_base__override_doors((nn__doors*)(uintptr_t)(TABLES)[nn_pkg_id])
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_INTEL_OPENCL2_UNIT_CUH */
