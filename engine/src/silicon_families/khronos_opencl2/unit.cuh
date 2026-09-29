#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_UNIT_CUH

/* ══ THE WHOLE OF khronos_opencl2, AS ONE TRANSLATION UNIT ════════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with the host's C++ compiler into
 * `libsilvann_khronos_opencl2.so`, linked against the OpenCL loader: this family's answers, every
 * package's host side of its card work, and the one entry point. The kernels are not in it as code — they
 * are OpenCL C, generated beside it and carried as a string (`nn.cuh`).
 * ⛳ `SYS__SILICON_FAMILY__KERNELS_BY_NAME` IS WHAT MAKES THAT SO: nn's doors then compile no kernel and
 *   launch by name. `environment.cuh` comes first, as in every host translation unit, because it is what
 *   defines `__device__` away. */
#include "../../packages/sys/cpu/silicon/environment.cuh"
#define SYS__SILICON_FAMILY__KERNELS_BY_NAME 1
#include "manifest.cuh"                           /* this family's answers to every package it supports */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#define SYS__SILICON_FAMILY__THIS khronos_opencl2
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_UNIT_CUH */
