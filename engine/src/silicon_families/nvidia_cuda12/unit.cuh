#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_UNIT_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_UNIT_CUH

/* ══ THE WHOLE OF nvidia_cuda12, AS ONE TRANSLATION UNIT ══════════════════════════════════════════════
 * `scripts/build_silicon_families.sh` compiles this with `nvcc` into `libsilvann_nvidia_cuda12.so`: this family's
 * answers, every package's card side written against them, and the one entry point. ⛳ It is a header and
 * not a `.cu` because the tree holds exactly one `.cu`, the program's; the build writes a one-line `.cu`
 * that includes this. */
#include <cuda_runtime.h>
#include "manifest.cuh"                           /* this family's answers to every package it supports */
#include "../../packages/manifest__gpu.cuh"       /* every package's card side, written against them */
#define SYS__SILICON_FAMILY__THIS nvidia_cuda12
#include "../entry.cuh"

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_UNIT_CUH */
