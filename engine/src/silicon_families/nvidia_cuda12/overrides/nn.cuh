#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "kernels.cuh"      /* what the overrides share */
#include "turboquant.cuh"   /* the gemvs, on the tensor cores */
#include "hadamard.cuh"     /* the rotations */
#include "deltanet.cuh"     /* a prompt's rows */
#include "attention.cuh"    /* a prompt's causal attention: the scores on the tensor cores */
#include "result.cuh"       /* argmax */
#include "expert.cuh"       /* the expert-major GEMM, on the tensor cores */
#include "conv.cuh"         /* a convolution's rows */
#include "vector.cuh"       /* a sampler's top k */
#include "doors.cuh"        /* into nn's table */

/* ══ ⭐⭐ OVERRIDES — THE DOORS THIS FAMILY RUNS FASTER THAN THE GENERIC BODY ══════════════════════════
 * Each function here has its door's signature and takes its place in nn's table (`nvidia_cuda12__override_doors`, called
 * from `../entry.cuh` once the table is built), in a file a verb group, as nn files its fallbacks. ⛳ EVERY ONE FALLS
 * BACK TO nn's GENERIC DOOR for what it does not cover — another width, codes that are not aligned, columns that are not
 * whole slices — so an override can only ever be faster, never different in what it accepts, and deleting this folder
 * loses nothing but speed. */

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_NN_CUH */
