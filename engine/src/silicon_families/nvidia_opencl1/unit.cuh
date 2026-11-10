#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_OPENCL1_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_OPENCL1_UNIT_CUH

/* ══ THE WHOLE OF nvidia_opencl1, AS ONE TRANSLATION UNIT ═════════════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with the host's C++ compiler into `libsilvann_nvidia_opencl1.so`,
 * linked against the OpenCL loader: the OpenCL 1.2 families' answers (`khronos_opencl1_base`), every package's host side
 * of its card work, and the one entry point. The kernels are not in it as code — they are OpenCL C 1.2, generated beside
 * it (`scripts/opencl1_kernel_source.py`) and carried as a string.
 * ⛳ `SYS__SILICON_FAMILY__KERNELS_BY_NAME` IS WHAT MAKES THAT SO: nn's doors then compile no kernel and
 *   launch by name. `environment.cuh` comes first, as in every host translation unit, because it is what
 *   defines `__device__` away. Its overrides launch kernels of the same program by name too. */
#include "../../packages/sys/cpu/silicon/environment.cuh"
#define SYS__SILICON_FAMILY__KERNELS_BY_NAME 1
#include "includes.cuh"                           /* this family's cards — NVIDIA's */
/* ⭐ BUILT FROM `khronos_opencl1_base` (▶ its manifest): the source's answers, this family's cards and name, and the
 *   read-only arenas on a card before Ampere — Pascal's lanes cache a read of the card's memory only through the
 *   read-only path (▶ the base's nn.cuh); on a Turing card it takes the 4B's decode 1.20× (a GTX 1660, `MEASURED`
 *   measurements/2026-11-09_nn169_opencl_split.md: 21.5 → 25.8 tokens/s, every door's bytes the same) */
#define KHRONOS_OPENCL1_BASE__TAKES(card)  nvidia_opencl1__includes(card)
#define KHRONOS_OPENCL1_BASE__NAME         "nvidia_opencl1"
#define KHRONOS_OPENCL1_BASE__READ_ONLY_NV_BELOW  8u   /* the read-only arenas below compute capability 8 */
#include "../khronos_opencl1_base/manifest.cuh"   /* the answers to every package the OpenCL 1.2 families support */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#include "../khronos_opencl1_base/overrides/nn.cuh"   /* the doors run faster than the generic body */
#define SYS__SILICON_FAMILY__THIS khronos_opencl1_base   /* the doors' names: the source's */
#define SYS__SILICON_FAMILY__IS   nvidia_opencl1         /* the family it is: its number and its name */
/* the hook is handed every package's door table, by package id; the one this family overrides is nn's */
#define SYS__SILICON_FAMILY__OVERRIDE_DOORS(TABLES)  khronos_opencl1_base__override_doors((nn__doors*)(uintptr_t)(TABLES)[nn_pkg_id])
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_OPENCL1_UNIT_CUH */
