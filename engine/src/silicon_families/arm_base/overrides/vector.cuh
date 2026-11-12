#ifndef SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_VECTOR_CUH
#define SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_VECTOR_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <arm_neon.h>
#include <stdint.h>
#include <string.h>                 /* memset: a zero as the memory's */
#include "../primitives/pool.cuh"   /* the threads a long vector is spread over */

/* ══ ⭐ A ZERO AND AN ADD AS THE MEMORY'S, NOT AS A LANE AN ELEMENT — `vector_zero`, `vector_add` ═════════════════════
 * nn's bodies give item `i` to block `i mod blocks`, so on this family every cache line is written by every thread.
 * `MEASURED` (the 2B on the Razer Edge, the launch profile, measurements/2026-11-11_nn175_arm_doors_razer_edge.md): a zero
 * of ~130,000 halves 432 µs, an add 35 µs. As `x86_avx2`'s: a vector the caller covers in a few microseconds is the
 * caller's own, and a longer one is cut into runs of whole cache lines, a run a pool block. The add's arithmetic is nn's —
 * both halves widened, summed in a float, rounded once by the processor's conversion (to the nearest even, as nn's), its
 * overflow in the fault word — so its bytes are nn's body's. */
#define ARM_BASE__ZZPRIVATE_CALLER_HALVES (64u * 1024u)    /* 128 KB: under it the caller's, over it the pool's */
#define ARM_BASE__ZZPRIVATE_RUN_HALVES    (16u * 1024u)    /* a pool block's run, 32 KB */
typedef struct arm_base__pointwise_call {
    uint16_t* z; const uint16_t* x; const uint16_t* y; uint64_t n; unsigned int* over;
} arm_base__pointwise_call;
static inline void arm_base__zzprivate_add_run(const arm_base__pointwise_call* c, uint64_t first, uint64_t end) {
    const float32x4_t inf = vdupq_n_f32(INFINITY);
    uint32x4_t hit = vdupq_n_u32(0u);
    uint64_t i = first;
    for (; i + 8u <= end; i += 8u) {
        const float16x8_t xh = vreinterpretq_f16_u16(vld1q_u16(c->x + i)), yh = vreinterpretq_f16_u16(vld1q_u16(c->y + i));
        const float32x4_t s0 = vaddq_f32(vcvt_f32_f16(vget_low_f16(xh)), vcvt_f32_f16(vget_low_f16(yh)));
        const float32x4_t s1 = vaddq_f32(vcvt_high_f32_f16(xh), vcvt_high_f32_f16(yh));
        const float16x8_t h = vcvt_high_f16_f32(vcvt_f16_f32(s0), s1);
        vst1q_u16(c->z + i, vreinterpretq_u16_f16(h));
        hit = vorrq_u32(hit, vandq_u32(vcltq_f32(vabsq_f32(s0), inf), vceqq_f32(vabsq_f32(vcvt_f32_f16(vget_low_f16(h))), inf)));
        hit = vorrq_u32(hit, vandq_u32(vcltq_f32(vabsq_f32(s1), inf), vceqq_f32(vabsq_f32(vcvt_high_f32_f16(h)), inf)));
    }
    for (; i < end; ++i)
        c->z[i] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(c->x[i]) + nn__silicon__half_to_float(c->y[i]), c->over);
    if (vmaxvq_u32(hit) != 0u) *c->over = NN__KERNELS__OVERFLOWED;
}
static void arm_base__zzprivate_add(void* a) {
    const arm_base__pointwise_call* c = (const arm_base__pointwise_call*)a;
    const uint64_t first = (uint64_t)nn__silicon__block() * ARM_BASE__ZZPRIVATE_RUN_HALVES;
    arm_base__zzprivate_add_run(c, first, first + ARM_BASE__ZZPRIVATE_RUN_HALVES < c->n ? first + ARM_BASE__ZZPRIVATE_RUN_HALVES : c->n);
}
static void arm_base__zzprivate_zero(void* a) {
    const arm_base__pointwise_call* c = (const arm_base__pointwise_call*)a;
    const uint64_t first = (uint64_t)nn__silicon__block() * ARM_BASE__ZZPRIVATE_RUN_HALVES;
    const uint64_t end = first + ARM_BASE__ZZPRIVATE_RUN_HALVES < c->n ? first + ARM_BASE__ZZPRIVATE_RUN_HALVES : c->n;
    memset(c->z + first, 0, (size_t)(2u * (end - first)));
}
static void arm_base__override_vector_zero(uint16_t* v, uint64_t n) {
    const arm_base__pointwise_call c = { v, 0, 0, n, 0 };
    const uint64_t runs = n <= ARM_BASE__ZZPRIVATE_CALLER_HALVES ? 1u : (n + ARM_BASE__ZZPRIVATE_RUN_HALVES - 1u) / ARM_BASE__ZZPRIVATE_RUN_HALVES;
    if (n == 0u || runs > 0xFFFFFFFFu) { nn__vector__zzabi_launch_zero(v, n); return; }
    if (runs == 1u) { memset(v, 0, (size_t)(2u * n)); return; }
    arm_base__zzpackage_run((uint32_t)runs, arm_base__zzprivate_zero, (void*)(uintptr_t)&c);
}
static void arm_base__override_vector_add(uint16_t* z, const uint16_t* x, const uint16_t* y, uint64_t n, unsigned int* over) {
    const arm_base__pointwise_call c = { z, x, y, n, over };
    const uint64_t runs = (n + ARM_BASE__ZZPRIVATE_RUN_HALVES - 1u) / ARM_BASE__ZZPRIVATE_RUN_HALVES;
    if (n == 0u || runs > 0xFFFFFFFFu) { nn__vector__zzabi_launch_add(z, x, y, n, over); return; }
    if (n <= ARM_BASE__ZZPRIVATE_CALLER_HALVES) { arm_base__zzprivate_add_run(&c, 0u, n); return; }
    arm_base__zzpackage_run((uint32_t)runs, arm_base__zzprivate_add, (void*)(uintptr_t)&c);
}

#endif /* SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_VECTOR_CUH */
