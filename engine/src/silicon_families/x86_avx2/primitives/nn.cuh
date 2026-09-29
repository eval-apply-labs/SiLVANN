#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_NN_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include "../../../packages/nn/contracts/defaults.cuh"   /* NN__ARITHMETIC__FAST */
#include "../../../packages/nn/contracts/abi/gpu.cuh"    /* what is answered here */
#include "pool.cuh"                                      /* where a launch's blocks run */

/* ══ nn's REQUESTS, ANSWERED — x86_avx2 ═════════════════════════════════════════════════════════════════
 * ⭐ A BLOCK IS ONE THREAD RUNNING THE BODY ON ONE LANE: `lanes()` is 1, so a body's lane loop is the whole
 * row and its combine step is the identity, exactly as on the host — and `blocks()` is the launch's own
 * count, spread over the pool, so a body that strides rows by block (every matvec) runs them in parallel.
 * The arithmetic is libm's, and a half is widened by F16C. */

/* One launch: every block of the door's struct-taking block function, on the pool, returning when done.
 * `args[0]` is the door's struct (`packages/nn/gpu/doors.cuh`, the host-threads form). */
static inline void nn__silicon__launch(const void* kernel, const char* name, uint32_t blocks, uint32_t threads,
                                       void** args, const size_t* sizes, uint32_t count) {
    (void)name; (void)threads; (void)sizes; (void)count;
    x86_avx2__zzpackage_run(blocks, (void (*)(void*))kernel, args[0]);
}

static __device__ inline float nn__silicon__expf(float x)  { return expf(x); }
static __device__ inline float nn__silicon__sqrtf(float x) { return sqrtf(x); }
static __device__ inline float nn__silicon__logf(float x)  { return logf(x); }
/* The angles are always the precise ones — ▶ nn for why. */
static __device__ inline float nn__silicon__cosf(float x) { return cosf(x); }
static __device__ inline float nn__silicon__sinf(float x) { return sinf(x); }

static __device__ inline uint32_t nn__silicon__lane(void)   { return 0u; }
static __device__ inline uint32_t nn__silicon__lanes(void)  { return 1u; }
static __device__ inline uint32_t nn__silicon__block(void)  { return x86_avx2__zzpackage_block; }
static __device__ inline uint32_t nn__silicon__blocks(void) { return x86_avx2__zzpackage_blocks; }
/* One lane holds the whole of its block's work, so its part is already the block's answer. */
static __device__ inline float nn__silicon__lanes_sum(float part) { return part; }
static __device__ inline void nn__silicon__lanes_sums(float* parts, uint32_t n) { (void)parts; (void)n; }
static __device__ inline float nn__silicon__lanes_max(float part) { return part; }
/* The GEMM's tile: four rows of the matrix by four of `x`, sixteen running sums — what AVX2's sixteen
 * registers hold. */
static __device__ inline nn__gemm__tile nn__silicon__gemm_tile(void) { nn__gemm__tile t = {4u, 4u}; return t; }

/* A half as a float: F16C's own conversion. */
static __device__ inline float nn__silicon__half_to_float(uint16_t h) { return _cvtsh_ss(h); }
/* Two halves by two halves onto a float, in the order a sum over the columns has. */
static __device__ inline float nn__silicon__dot2(uint32_t a, uint32_t b, float c) {
    return (c + nn__silicon__half_to_float((uint16_t)(a & 0xffffu)) * nn__silicon__half_to_float((uint16_t)(b & 0xffffu)))
         + nn__silicon__half_to_float((uint16_t)(a >> 16)) * nn__silicon__half_to_float((uint16_t)(b >> 16));
}

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_NN_CUH */
