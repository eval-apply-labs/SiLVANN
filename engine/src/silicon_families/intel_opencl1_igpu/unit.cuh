#ifndef SILVANN__SILICON_FAMILIES_INTEL_OPENCL1_IGPU_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_INTEL_OPENCL1_IGPU_UNIT_CUH

/* ══ THE WHOLE OF intel_opencl1_igpu, AS ONE TRANSLATION UNIT ═════════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with the host's C++ compiler into `libsilvann_intel_opencl1_igpu.so`,
 * linked against the OpenCL loader: the OpenCL 1.2 families' answers (`khronos_opencl1_base`), every package's host side
 * of its card work, and the one entry point. The kernels are not in it as code — they are OpenCL C 1.2, generated beside
 * it (`scripts/opencl1_kernel_source.py`) and carried as a string.
 * ⛳ `SYS__SILICON_FAMILY__KERNELS_BY_NAME` IS WHAT MAKES THAT SO: nn's doors then compile no kernel and
 *   launch by name. `environment.cuh` comes first, as in every host translation unit, because it is what
 *   defines `__device__` away. Its overrides launch kernels of the same program by name too. */
#include "../../packages/sys/cpu/silicon/environment.cuh"
#define SYS__SILICON_FAMILY__KERNELS_BY_NAME 1
#include "includes.cuh"                           /* this family's cards — Intel's */
/* ⭐ BUILT FROM `khronos_opencl1_base` (▶ its manifest): the source's answers, this family's cards and name; no card of
 *   its takes the read-only arenas, NVIDIA's (▶ the base's nn.cuh) */
#define KHRONOS_OPENCL1_BASE__TAKES(card)  intel_opencl1_igpu__includes(card)
#define KHRONOS_OPENCL1_BASE__NAME         "intel_opencl1_igpu"
#define KHRONOS_OPENCL1_BASE__READ_ONLY_NV_BELOW  0u   /* none */
/* ⭐ EVERY FUNCTION OF THE PROGRAM INLINED: Intel's compiler leaves kernels out of the whole program where it keeps nn's row
 *   product a function of its own (▶ `scripts/opencl1_kernel_source.py`'s `INLINE`); and THE PRODUCTS' LEVELS HELD in the
 *   block's shared memory, which Intel's card reads a lookup from many times faster than from its memory (`LEVELS_HELD`) */
/* ⭐ AND BUFFERS PAST 4 GiB, Intel's way, as `intel_opencl2`'s: the buffer asked for with `CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL`
 *   (bit 23) where it is larger than the card's largest, and the kernels built for such buffers. `MEASURED` on a Rocket Lake's
 *   UHD P750 (measurements/2026-11-25_nn214_intel_opencl_phoenixnap.md): without them, Inkling 2 Mini's experts on the card
 *   refused at the boot — "would not allocate the pool" (its largest allocation 4 GiB). */
#define KHRONOS_OPENCL1_BASE__LARGE_FLAG           (1u << 23)
/* ⭐ AND THE PRODUCTS READ NARROW, a lane eight bytes of a row, a run eight lanes (▶ the base's kernels.cuh) */
#define KHRONOS_OPENCL1_BASE__GEMV_NARROW          1
#define KHRONOS_OPENCL1_BASE__BUILD_OPTIONS        " -D KHRONOS_OPENCL1_BASE__INLINE=__attribute__((always_inline))" \
                                                   " -D KHRONOS_OPENCL1_BASE__LEVELS_HELD"
/* …the option for buffers past 4 GiB asked of the compiler first, and the program built without it where the compiler does
 *   not take it — an older driver's — its card then holding no buffer past its largest (▶ the base's nn.cuh) */
#define KHRONOS_OPENCL1_BASE__BUILD_OPTIONS_LARGE  " -cl-intel-greater-than-4GB-buffer-required"
#include "../khronos_opencl1_base/manifest.cuh"   /* the answers to every package the OpenCL 1.2 families support */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#include "../khronos_opencl1_base/overrides/nn.cuh"   /* the doors run faster than the generic body */
#define SYS__SILICON_FAMILY__THIS khronos_opencl1_base   /* the doors' names: the source's */
#define SYS__SILICON_FAMILY__IS   intel_opencl1_igpu     /* the family it is: its number and its name */
/* the hook is handed every package's door table, by package id; the one this family overrides is nn's */
#define SYS__SILICON_FAMILY__OVERRIDE_DOORS(TABLES)  khronos_opencl1_base__override_doors((nn__doors*)(uintptr_t)(TABLES)[nn_pkg_id])
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_INTEL_OPENCL1_IGPU_UNIT_CUH */
