#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_NN_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stddef.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include "../../packages/nn/contracts/defaults.cuh"   /* NN__ARITHMETIC__FAST */
#include "../../packages/nn/contracts/abi/gpu.cuh"    /* what is answered here, and NN__SILICON__LANES_MAX */

#ifndef NN__ARITHMETIC__FAST
#define NN__ARITHMETIC__FAST 0
#endif

/* ══ nn's REQUESTS, ANSWERED — nvidia_cuda12 ══════════════════════════════════════════════════════════
 * What nn asks a family for (`nn/contracts/abi/gpu.cuh`), and only that: the call that starts a kernel,
 * and the arithmetic its bodies use. nn's kernels and doors are nn's own (`nn/gpu/doors.cuh`) and are
 * written against these, so this file is the whole of what CUDA adds to them. */

/* Start `kernel` on `blocks` blocks of `threads` threads, with `args` holding a pointer to each of its
 * arguments, in order — CUDA's own launch by pointer. It returns once the launch is queued; the work is
 * waited for with sys's `compute_completed`, and a launch that failed is found there too. */
static inline void nn__silicon__launch(const void* kernel, const char* name, uint32_t blocks, uint32_t threads,
                                       void** args, const size_t* sizes, uint32_t count) {
    (void)name; (void)sizes; (void)count;   /* a launch by pointer needs only the address */
    (void)cudaLaunchKernel(kernel, dim3(blocks), dim3(threads), args, 0, 0);
}

/* The arithmetic, from CUDA's device library: the precise functions unless nn's `NN__ARITHMETIC__FAST`
 * asks for the card's fast ones, for the reasons nn gives. */
static __device__ inline float nn__silicon__expf(float x) {
#if NN__ARITHMETIC__FAST
    return __expf(x);
#else
    return expf(x);
#endif
}
static __device__ inline float nn__silicon__sqrtf(float x) { return sqrtf(x); }
static __device__ inline float nn__silicon__logf(float x) {
#if NN__ARITHMETIC__FAST
    return __logf(x);
#else
    return logf(x);
#endif
}
/* The angles are always the precise ones, whatever the dial says — ▶ nn for why. */
static __device__ inline float nn__silicon__cosf(float x) { return cosf(x); }
static __device__ inline float nn__silicon__sinf(float x) { return sinf(x); }

/* Which lane of its block a body runs on, and how many the block has. */
static __device__ inline uint32_t nn__silicon__lane(void)  { return (uint32_t)threadIdx.x; }
static __device__ inline uint32_t nn__silicon__lanes(void) { return (uint32_t)blockDim.x; }
static __device__ inline uint32_t nn__silicon__block(void)  { return (uint32_t)blockIdx.x; }
static __device__ inline uint32_t nn__silicon__blocks(void) { return (uint32_t)gridDim.x; }

/* The block's sums or largests, handed to every lane: each lane's parts go into block-shared memory, eight at a time,
 * each is folded pairwise in a fixed order (so a run is repeatable, and a value's answer is the same folded alone or
 * with others), and every lane reads the results.
 * ⛳ THE LAST WAIT IS WHAT LETS THE NEXT CALL REUSE THE SAME MEMORY: without it a fast lane could write
 * its next part before a slow one had read this answer. */
#define NVIDIA_CUDA12__ZZPRIVATE_FOLDS 8u
static __device__ inline void nvidia_cuda12__zzprivate_fold(float* part, uint32_t n, bool largest) {
    __shared__ float parts[NVIDIA_CUDA12__ZZPRIVATE_FOLDS][NN__SILICON__LANES_MAX];
    const uint32_t me = (uint32_t)threadIdx.x, lanes = (uint32_t)blockDim.x;
    for (uint32_t k0 = 0u; k0 < n; k0 += NVIDIA_CUDA12__ZZPRIVATE_FOLDS) {
        const uint32_t m = n - k0 < NVIDIA_CUDA12__ZZPRIVATE_FOLDS ? n - k0 : NVIDIA_CUDA12__ZZPRIVATE_FOLDS;
        for (uint32_t k = 0u; k < m; ++k) parts[k][me] = part[k0 + k];
        __syncthreads();
        for (uint32_t w = 1u; w < lanes; w <<= 1) {
            if ((me % (2u * w)) == 0u && me + w < lanes)
                for (uint32_t k = 0u; k < m; ++k)
                    parts[k][me] = largest ? fmaxf(parts[k][me], parts[k][me + w]) : parts[k][me] + parts[k][me + w];
            __syncthreads();
        }
        for (uint32_t k = 0u; k < m; ++k) part[k0 + k] = parts[k][0];
        __syncthreads();
    }
}
static __device__ inline float nn__silicon__lanes_sum(float part) { nvidia_cuda12__zzprivate_fold(&part, 1u, false); return part; }
static __device__ inline float nn__silicon__lanes_max(float part) { nvidia_cuda12__zzprivate_fold(&part, 1u, true); return part; }
/* Several sums, a tile of eight waiting for the block as often as one does. */
static __device__ inline void nn__silicon__lanes_sums(float* part, uint32_t n) { nvidia_cuda12__zzprivate_fold(part, n, false); }
/* The GEMM's tile: one row of the matrix a block, as the gemv, each decoded code multiplied into eight rows
 * of `x`. */
static __device__ inline nn__gemm__tile nn__silicon__gemm_tile(void) { nn__gemm__tile t = {1u, 8u}; return t; }

/* A half as a float: the card's own conversion, one instruction. */
static __device__ inline float nn__silicon__half_to_float(uint16_t h) { return __half2float(__ushort_as_half(h)); }
/* Two halves by two halves onto a float, in the order a sum over the columns has. */
static __device__ inline float nn__silicon__dot2(uint32_t a, uint32_t b, float c) {
    return (c + nn__silicon__half_to_float((uint16_t)(a & 0xffffu)) * nn__silicon__half_to_float((uint16_t)(b & 0xffffu)))
         + nn__silicon__half_to_float((uint16_t)(a >> 16)) * nn__silicon__half_to_float((uint16_t)(b >> 16));
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_NN_CUH */
