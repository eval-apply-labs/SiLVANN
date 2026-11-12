#ifndef SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_PRIMITIVES_CUH
#define SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_PRIMITIVES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include "../primitives/pool.cuh"   /* the threads the rows are spread over */

/* ══ ⭐ THE NORMS ON NEON — `rmsnorm`, `rmsnorm_rows`, and their rows of floats `rmsnorm_fp32`, `rmsnorm_fp32_rows` ════
 * nn's definition (`kernels.cuh`): a row's squares summed, `scale = 1 / √(sum / n + eps)`, each output `(x · scale) · w`
 * rounded to a half. On this family's one lane nn's body is that loop element by element, its halves widened one at a
 * time, and a lone row a launch of one block. `MEASURED` (the 2B on the Razer Edge, the launch profile,
 * measurements/2026-11-11_nn175_arm_doors_razer_edge.md): 22 µs a row of 2,048. Here the squares are summed in nn's order,
 * a multiply-add each onto one running sum, and the outputs made eight at a time: each output nn's arithmetic, two
 * products rounded as nn's and the half by the processor's own conversion, which rounds to the nearest even as nn's does.
 * A half's square is exact in a float, so a row of halves is nn's bytes however a compiler builds `part += v · v`
 * (`MEASURED` the door bench on the Razer Edge, clang 21: every row worst 0 ulp). A float's square rounds, and clang
 * builds nn's loop with each square rounded before its add (the products four a vector, the sum in order — `MEASURED`
 * llvm-objdump of main's arm_neon): of 655,360 outputs over rows of floats one 1 ulp from nn's host body, within the
 * bench's 2. One body a row: the one-row door is a block of one on the caller's thread, and the many-row doors give the
 * pool's blocks a row each, so every row of `rmsnorm_rows` is the bytes `rmsnorm` gives it alone. */
#define ARM_BASE__ZZPRIVATE_NORM_CALLER 16384u   /* values under which the rows are the caller's alone */

/* Eight outputs `(x · scale) · w` as halves, the fault word raised where a finite value became an infinite half. */
static inline void arm_base__zzprivate_norm_eight(uint16_t* o, float32x4_t x0, float32x4_t x1, const uint16_t* w, float scale,
                                                  unsigned int* over) {
    const float16x8_t wh = vreinterpretq_f16_u16(vld1q_u16(w));
    const float32x4_t r0 = vmulq_f32(vmulq_n_f32(x0, scale), vcvt_f32_f16(vget_low_f16(wh)));
    const float32x4_t r1 = vmulq_f32(vmulq_n_f32(x1, scale), vcvt_high_f32_f16(wh));
    const float16x8_t h = vcvt_high_f16_f32(vcvt_f16_f32(r0), r1);
    vst1q_u16(o, vreinterpretq_u16_f16(h));
    const float32x4_t inf = vdupq_n_f32(INFINITY);
    const uint32x4_t hit = vorrq_u32(vandq_u32(vcltq_f32(vabsq_f32(r0), inf), vceqq_f32(vabsq_f32(vcvt_f32_f16(vget_low_f16(h))), inf)),
                                     vandq_u32(vcltq_f32(vabsq_f32(r1), inf), vceqq_f32(vabsq_f32(vcvt_high_f32_f16(h)), inf)));
    if (vmaxvq_u32(hit) != 0u) *over = NN__KERNELS__OVERFLOWED;
}
/* One row of halves. */
static inline void arm_base__zzprivate_norm_row(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over) {
    float part = 0.0f;
    for (uint64_t i = 0u; i < n; ++i) { const float v = nn__silicon__half_to_float(x[i]); part = fmaf(v, v, part); }
    const float scale = 1.0f / sqrtf(part / (float)n + eps);
    uint64_t i = 0u;
    for (; i + 8u <= n; i += 8u) {
        const float16x8_t xh = vreinterpretq_f16_u16(vld1q_u16(x + i));
        arm_base__zzprivate_norm_eight(o + i, vcvt_f32_f16(vget_low_f16(xh)), vcvt_high_f32_f16(xh), w + i, scale, over);
    }
    for (; i < n; ++i)
        o[i] = nn__kernels__zzabi_to_half((nn__silicon__half_to_float(x[i]) * scale) * nn__silicon__half_to_float(w[i]), over);
}
/* One row of floats. */
static inline void arm_base__zzprivate_norm_row_fp32(uint16_t* o, const float* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over) {
    float part = 0.0f;
    for (uint64_t i = 0u; i < n; ++i) part = fmaf(x[i], x[i], part);
    const float scale = 1.0f / sqrtf(part / (float)n + eps);
    uint64_t i = 0u;
    for (; i + 8u <= n; i += 8u) arm_base__zzprivate_norm_eight(o + i, vld1q_f32(x + i), vld1q_f32(x + i + 4u), w + i, scale, over);
    for (; i < n; ++i) o[i] = nn__kernels__zzabi_to_half((x[i] * scale) * nn__silicon__half_to_float(w[i]), over);
}

/* Row `block` of a call: halves `x_stride` apart into `o_stride` apart, or floats `n` apart with the gain's row `r mod
 * period` (`period` 0: one gain). A block of one takes every row. */
typedef struct arm_base__norm_call {
    uint16_t* o; const uint16_t* x; const float* xf; const uint16_t* w; uint64_t n, rows, x_stride, o_stride, period; float eps;
    unsigned int* over;
} arm_base__norm_call;
static void arm_base__zzprivate_norm_rows(void* a) {
    const arm_base__norm_call* c = (const arm_base__norm_call*)a;
    const uint64_t blocks = nn__silicon__blocks();
    for (uint64_t r = nn__silicon__block(); r < c->rows; r += blocks) {
        if (c->xf) arm_base__zzprivate_norm_row_fp32(c->o + r * c->n, c->xf + r * c->n, c->w + (c->period != 0u ? r % c->period : 0u) * c->n,
                                                     c->n, c->eps, c->over);
        else arm_base__zzprivate_norm_row(c->o + r * c->o_stride, c->x + r * c->x_stride, c->w, c->n, c->eps, c->over);
    }
}
/* A few rows on the caller's thread alone, more a row a pool block. */
static inline void arm_base__zzprivate_norm_run(const arm_base__norm_call* c) {
    const uint32_t blocks = c->rows * c->n < ARM_BASE__ZZPRIVATE_NORM_CALLER ? 1u : (uint32_t)c->rows;
    arm_base__zzpackage_run(blocks, arm_base__zzprivate_norm_rows, (void*)(uintptr_t)c);
}
static void arm_base__override_rmsnorm(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over) {
    if (n == 0u) { nn__rmsnorm__zzabi_launch(o, x, w, n, eps, over); return; }
    const arm_base__norm_call c = { o, x, 0, w, n, 1u, n, n, 0u, eps, over };
    arm_base__zzpackage_run(1u, arm_base__zzprivate_norm_rows, (void*)(uintptr_t)&c);
}
static void arm_base__override_rmsnorm_rows(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, uint64_t rows, uint64_t x_stride,
                                            uint64_t o_stride, float eps, unsigned int* over) {
    if (n == 0u || rows == 0u || rows > 0xFFFFFFFFu) {
        nn__rmsnorm__zzabi_launch_rows(o, x, w, n, rows, x_stride, o_stride, eps, over); return;
    }
    const arm_base__norm_call c = { o, x, 0, w, n, rows, x_stride, o_stride, 0u, eps, over };
    arm_base__zzprivate_norm_run(&c);
}
static void arm_base__override_rmsnorm_fp32(uint16_t* o, const float* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over) {
    if (n == 0u) { nn__rmsnorm__zzabi_launch_fp32(o, x, w, n, eps, over); return; }
    const arm_base__norm_call c = { o, 0, x, w, n, 1u, n, n, 0u, eps, over };
    arm_base__zzpackage_run(1u, arm_base__zzprivate_norm_rows, (void*)(uintptr_t)&c);
}
static void arm_base__override_rmsnorm_fp32_rows(uint16_t* o, const float* x, const uint16_t* w, uint64_t n, uint64_t rows, uint64_t period,
                                                 float eps, unsigned int* over) {
    if (n == 0u || rows == 0u || rows > 0xFFFFFFFFu) {
        nn__rmsnorm__zzabi_launch_fp32_rows(o, x, w, n, rows, period, eps, over); return;
    }
    const arm_base__norm_call c = { o, 0, x, w, n, rows, n, n, period, eps, over };
    arm_base__zzprivate_norm_run(&c);
}

#endif /* SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_PRIMITIVES_CUH */
