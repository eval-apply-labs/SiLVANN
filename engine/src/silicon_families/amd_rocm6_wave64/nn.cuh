#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_NN_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "../../packages/nn/contracts/defaults.cuh"   /* NN__ARITHMETIC__FAST */
#include "../../packages/nn/contracts/abi/gpu.cuh"    /* what is answered here, and NN__SILICON__LANES_MAX */

#ifndef NN__ARITHMETIC__FAST
#define NN__ARITHMETIC__FAST 0
#endif

/* ══ nn's REQUESTS, ANSWERED — amd_rocm6_wave64 ═══════════════════════════════════════════════════════
 * What nn asks a family for (`nn/contracts/abi/gpu.cuh`), and only that: the call that starts a kernel,
 * and the arithmetic its bodies use. nn's kernels and doors are nn's own (`nn/gpu/doors.cuh`) and are
 * written against these, so this file is the whole of what HIP adds to them. */

/* Start `kernel` on `blocks` blocks of `threads` threads, with `args` holding a pointer to each of its
 * arguments, in order — HIP's own launch by pointer. It returns once the launch is queued; the work is
 * waited for with sys's `compute_completed`. A launch over no blocks has nothing to do and is not made (HIP
 * refuses an empty grid); a launch HIP refuses is named on stderr, since the wait after it does not report it. */
static inline void nn__silicon__launch(const void* kernel, const char* name, uint32_t blocks, uint32_t threads,
                                       void** args, const size_t* sizes, uint32_t count) {
    (void)sizes; (void)count;   /* a launch by pointer needs only the address */
    if (blocks == 0u) return;
    const hipError_t e = hipLaunchKernel(kernel, dim3(blocks), dim3(threads), args, 0, 0);
    if (e != hipSuccess) fprintf(stderr, "amd_rocm6_wave64: %s did not launch (%u blocks of %u): %s\n", name, blocks, threads, hipGetErrorString(e));
}

/* The arithmetic, from HIP's device library: the precise functions unless nn's `NN__ARITHMETIC__FAST`
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

/* The block's sums or largests, handed to every lane: each part is folded pairwise in a fixed order (so a run is
 * repeatable, and a value's answer is the same folded alone or with others), and every lane reads the results. In a
 * block of whole waves, a power of two of them, the folds inside a wave are its lanes' exchanges — lane `me` and lane
 * `me + w` added as the block-shared tree adds them, the same two floats in the same order on the lanes the tree keeps —
 * and only the waves' sums go through block-shared memory, eight parts at a time, with one wait; any other block folds
 * every level through block-shared memory, a wait a level.
 * ⛳ THE LAST WAIT IS WHAT LETS THE NEXT CALL REUSE THE SAME MEMORY: without it a fast lane could write
 * its next part before a slow one had read this answer. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_FOLDS 8u
#define AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD_WAVE 64u
#define AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD_WAVES 16u     /* the waves of the largest block this family launches, 1024 lanes */
static __device__ inline void amd_rocm6_wave64__zzprivate_fold(float* part, uint32_t n, bool largest) {
    __shared__ float parts[AMD_ROCM6_WAVE64__ZZPRIVATE_FOLDS][NN__SILICON__LANES_MAX];
    const uint32_t me = (uint32_t)threadIdx.x, lanes = (uint32_t)blockDim.x;
    const uint32_t W = AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD_WAVE, waves = lanes / W;
    if (lanes % W == 0u && (waves & (waves - 1u)) == 0u && waves <= AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD_WAVES) {
        for (uint32_t k0 = 0u; k0 < n; k0 += AMD_ROCM6_WAVE64__ZZPRIVATE_FOLDS) {
            const uint32_t m = n - k0 < AMD_ROCM6_WAVE64__ZZPRIVATE_FOLDS ? n - k0 : AMD_ROCM6_WAVE64__ZZPRIVATE_FOLDS;
            for (uint32_t k = 0u; k < m; ++k) {
                float x = part[k0 + k];
                for (uint32_t w = 1u; w < W; w <<= 1) {
                    const float y = __shfl_xor(x, (int)w);
                    x = largest ? fmaxf(x, y) : x + y;
                }
                if (me % W == 0u) parts[k][me / W] = x;
            }
            __syncthreads();
            for (uint32_t k = 0u; k < m; ++k) {
                float v[AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD_WAVES];
                for (uint32_t i = 0u; i < waves; ++i) v[i] = parts[k][i];
                for (uint32_t w = 1u; w < waves; w <<= 1)
                    for (uint32_t i = 0u; i + w < waves; i += 2u * w) v[i] = largest ? fmaxf(v[i], v[i + w]) : v[i] + v[i + w];
                part[k0 + k] = v[0];
            }
            __syncthreads();
        }
        return;
    }
    for (uint32_t k0 = 0u; k0 < n; k0 += AMD_ROCM6_WAVE64__ZZPRIVATE_FOLDS) {
        const uint32_t m = n - k0 < AMD_ROCM6_WAVE64__ZZPRIVATE_FOLDS ? n - k0 : AMD_ROCM6_WAVE64__ZZPRIVATE_FOLDS;
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
static __device__ inline float nn__silicon__lanes_sum(float part) { amd_rocm6_wave64__zzprivate_fold(&part, 1u, false); return part; }
static __device__ inline float nn__silicon__lanes_max(float part) { amd_rocm6_wave64__zzprivate_fold(&part, 1u, true); return part; }
/* Several sums, a tile of eight waiting for the block as often as one does. */
static __device__ inline void nn__silicon__lanes_sums(float* part, uint32_t n) { amd_rocm6_wave64__zzprivate_fold(part, n, false); }
/* The GEMM's tile: four rows of the matrix a block, each decoded code multiplied into eight rows of `x`, so each
 * value of `x` a lane reads serves four rows. `MEASURED` on the 27B, a layer over 64 rows: 172 ms at one row a
 * block, 98 ms at four. */
static __device__ inline nn__gemm__tile nn__silicon__gemm_tile(void) { nn__gemm__tile t = {4u, 8u}; return t; }

/* A half as a float: the card's own conversion, one instruction. */
static __device__ inline float nn__silicon__half_to_float(uint16_t h) { return (float)__builtin_bit_cast(_Float16, h); }
/* Two halves by two halves onto a float: the card's dot instruction (`v_dot2_f32_f16`) where the target has it —
 * gfx906 and gfx908, as the overrides' own dot says — and the two products widened and added on any other. */
typedef _Float16 amd_rocm6_wave64__half2 __attribute__((ext_vector_type(2)));   /* two halves, one register */
static __device__ inline float nn__silicon__dot2(uint32_t a, uint32_t b, float c) {
#if defined(__gfx906__) || defined(__gfx908__)
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(amd_rocm6_wave64__half2, a),
                                  __builtin_bit_cast(amd_rocm6_wave64__half2, b), c, false);
#else
    c += nn__silicon__half_to_float((uint16_t)(a & 0xFFFFu)) * nn__silicon__half_to_float((uint16_t)(b & 0xFFFFu));
    c += nn__silicon__half_to_float((uint16_t)(a >> 16)) * nn__silicon__half_to_float((uint16_t)(b >> 16));
    return c;
#endif
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_NN_CUH */
