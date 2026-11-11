#ifndef SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "../primitives/pool.cuh"   /* the threads the rows are spread over */

/* ══ ⭐⭐ OVERRIDES — THE DOORS THE ARM FAMILIES RUN FASTER THAN THE GENERIC BODY ══════════════════════════════════
 * ⚖ *"a folder for primitives and one for overrides so it is clear what is expected to run and what are performance
 * upgrades"*. Each function here has its door's signature and takes its place in nn's table (`arm_base__override_doors`);
 * every family built from this source has them, and a family's own (`arm_gb10/overrides/`) are put in after. ⛳ EVERY ONE
 * FALLS BACK TO nn's GENERIC DOOR for what it does not cover — another width, a row not a whole number of 32 columns — so
 * an override can only be faster, never different in what it accepts.
 * ⭐ `x86_avx2`'s exact route, in NEON: `x` widened to floats once a call; at 4 bits a byte's two codes split, each code's
 *   level found by `tbl` — its half's two bytes, from two tables of sixteen — the halves widened and multiplied on; at 8
 *   bits the levels a table of 256 floats; at 16 a code is the weight. The products are nn's; the order they are added in
 *   is not, so an answer is within the bench's bound of nn's, not its bytes. */

/* ── ⚗ EXPERIMENTAL — `SILVANN_ARM_INT8=1`: THE EXACT 4-BIT GEMVS IN nn's INT8 ARITHMETIC ─────────────────────────────
 * One matrix at 4 and 8 bits (the head), the grouped and the grouped onto the residual at 4, handed to this family's int8 doors (▶ "THE GEMVS
 * IN INTEGERS" below): `x` in blocks of 32 as int8, the levels as int8, a block's products through `sdot` — nn's int8 verb,
 * ~1% of the rows' rms from the exact answer. For a phone, whose cores spend the exact route's time widening and
 * multiplying floats rather than waiting on memory. Unset, every door below is as written. */
static int arm_base__zzprivate_int8_lever = -1;
static inline bool arm_base__zzprivate_int8_everywhere(void) {
    if (arm_base__zzprivate_int8_lever < 0) {
        const char* v = getenv("SILVANN_ARM_INT8");
        arm_base__zzprivate_int8_lever = v != 0 && v[0] == '1' ? 1 : 0;
    }
    return arm_base__zzprivate_int8_lever == 1;
}
static void arm_base__override_turboquant_gemv_int8(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                    unsigned int* over);
static void arm_base__override_gemv_groups_int8(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                unsigned int* over);
static void arm_base__override_gemv_groups_sum_int8(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                    const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over);
/* every matrix of the group 4 bits wide: the int8 doors' width */
static inline bool arm_base__zzprivate_groups_four(const nn__turboquant__groups* g) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX) return false;
    for (uint64_t i = 0u; i < g->count; ++i) if (g->d[i] != 4u) return false;
    return true;
}

/* ── the rows of one call, spread over the pool: each block a contiguous run of rows ───────────────────────────── */
typedef struct arm_base__gemv_call {
    uint16_t* out; const uint8_t* weights; const uint8_t* luts; uint64_t row_bytes, rows, cols, d;
    const float* xf; unsigned int* over;
} arm_base__gemv_call;

static inline float arm_base__zzprivate_row_scale(const uint8_t* luts, uint64_t r) {
    return nn__silicon__half_to_float((uint16_t)((uint32_t)luts[r * 2u] | ((uint32_t)luts[r * 2u + 1u] << 8)));
}
/* The block's rows: an even share of them, by the pool's block number. */
static inline void arm_base__zzprivate_rows(uint64_t rows, uint64_t* r0, uint64_t* r1) {
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    *r0 = rows * b / n; *r1 = rows * (b + 1u) / n;
}

/* The 4-bit levels as two tables of sixteen bytes — each level's half, its low byte and its high. */
typedef struct arm_base__tables4 { uint8x16_t lo, hi; } arm_base__tables4;
static inline arm_base__tables4 arm_base__zzprivate_tables4(void) {
    const uint16_t* lev = nn__turboquant__zzabi_levels(4u);
    uint8_t lo[16], hi[16];
    for (int i = 0; i < 16; ++i) { lo[i] = (uint8_t)lev[i]; hi[i] = (uint8_t)(lev[i] >> 8); }
    arm_base__tables4 t = { vld1q_u8(lo), vld1q_u8(hi) };
    return t;
}

/* ⭐ ONE ORDER FOR EVERY DOT: of each 32 columns, columns 4e .. 4e + 3 go to running sum `e` (eight of four lanes), and the
 *   eight are added up as below — the gemvs' dots (4, 8 and 16 bits) and the expert GEMM's tiles alike, so a row's dot with
 *   `x` is the same bits however it is reached. */
static inline float arm_base__zzprivate_sum8(float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3, float32x4_t a4,
                                             float32x4_t a5, float32x4_t a6, float32x4_t a7) {
    return vaddvq_f32(vaddq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)), vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7))));
}
/* A row of 4-bit codes against `x` as floats: 32 columns a step, sixteen bytes of codes, eight running sums. */
static inline float arm_base__zzprivate_dot4(const uint8_t* row, const float* xf, uint64_t cols, arm_base__tables4 t) {
    const uint8x16_t fifteen = vdupq_n_u8(15u);
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (uint64_t k = 0; k < cols; k += 32u) {
        const uint8x16_t v = vld1q_u8(row + k / 2u);
        const uint8x16_t lo = vandq_u8(v, fifteen), hi = vshrq_n_u8(v, 4);
        const uint8x16_t i0 = vzip1q_u8(lo, hi), i1 = vzip2q_u8(lo, hi);            /* columns 0..15, 16..31 */
        const uint8x16_t bl0 = vqtbl1q_u8(t.lo, i0), bh0 = vqtbl1q_u8(t.hi, i0);
        const uint8x16_t bl1 = vqtbl1q_u8(t.lo, i1), bh1 = vqtbl1q_u8(t.hi, i1);
        const float16x8_t h0 = vreinterpretq_f16_u8(vzip1q_u8(bl0, bh0)), h1 = vreinterpretq_f16_u8(vzip2q_u8(bl0, bh0));
        const float16x8_t h2 = vreinterpretq_f16_u8(vzip1q_u8(bl1, bh1)), h3 = vreinterpretq_f16_u8(vzip2q_u8(bl1, bh1));
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(xf + k));
        a1 = vfmaq_f32(a1, vcvt_high_f32_f16(h0),          vld1q_f32(xf + k + 4u));
        a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(xf + k + 8u));
        a3 = vfmaq_f32(a3, vcvt_high_f32_f16(h1),          vld1q_f32(xf + k + 12u));
        a4 = vfmaq_f32(a4, vcvt_f32_f16(vget_low_f16(h2)), vld1q_f32(xf + k + 16u));
        a5 = vfmaq_f32(a5, vcvt_high_f32_f16(h2),          vld1q_f32(xf + k + 20u));
        a6 = vfmaq_f32(a6, vcvt_f32_f16(vget_low_f16(h3)), vld1q_f32(xf + k + 24u));
        a7 = vfmaq_f32(a7, vcvt_high_f32_f16(h3),          vld1q_f32(xf + k + 28u));
    }
    return arm_base__zzprivate_sum8(a0, a1, a2, a3, a4, a5, a6, a7);
}

/* The 8-bit levels as floats, made once. */
static float arm_base__zzprivate_levels8[256];
static sys__os__once arm_base__zzprivate_levels8_once = SYS__OS__ONCE_INIT;
static void arm_base__zzprivate_levels8_make(void) {
    const uint16_t* lev = nn__turboquant__zzabi_levels(8u);
    for (int i = 0; i < 256; ++i) arm_base__zzprivate_levels8[i] = nn__silicon__half_to_float(lev[i]);
}
/* A row of 8-bit codes: a level a code from the table, 32 columns a step, eight running sums. */
static inline float arm_base__zzprivate_dot8(const uint8_t* row, const float* xf, uint64_t cols) {
    const float* lev = arm_base__zzprivate_levels8;
    float32x4_t a[8];
    for (int e = 0; e < 8; ++e) a[e] = vdupq_n_f32(0.0f);
    for (uint64_t k = 0; k < cols; k += 32u)
        for (int e = 0; e < 8; ++e) {
            const uint8_t* c = row + k + 4u * (uint64_t)e;
            const float wv[4] = { lev[c[0]], lev[c[1]], lev[c[2]], lev[c[3]] };
            a[e] = vfmaq_f32(a[e], vld1q_f32(wv), vld1q_f32(xf + k + 4u * (uint64_t)e));
        }
    return arm_base__zzprivate_sum8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
}
/* A row of halves, the weights themselves (the routers), no scale: 32 columns a step, eight running sums. */
static inline float arm_base__zzprivate_dot16(const uint16_t* row, const float* xf, uint64_t cols) {
    float32x4_t a[8];
    for (int e = 0; e < 8; ++e) a[e] = vdupq_n_f32(0.0f);
    for (uint64_t k = 0; k < cols; k += 32u)
        for (int h = 0; h < 4; ++h) {
            const float16x8_t v = vreinterpretq_f16_u16(vld1q_u16(row + k + 8u * (uint64_t)h));
            a[2 * h]     = vfmaq_f32(a[2 * h],     vcvt_f32_f16(vget_low_f16(v)), vld1q_f32(xf + k + 8u * (uint64_t)h));
            a[2 * h + 1] = vfmaq_f32(a[2 * h + 1], vcvt_high_f32_f16(v),          vld1q_f32(xf + k + 8u * (uint64_t)h + 4u));
        }
    return arm_base__zzprivate_sum8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
}

static void arm_base__zzprivate_exact(void* a) {
    const arm_base__gemv_call* c = (const arm_base__gemv_call*)a;
    const arm_base__tables4 t = arm_base__zzprivate_tables4();
    uint64_t r0, r1; arm_base__zzprivate_rows(c->rows, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        const uint8_t* row = c->weights + r * c->row_bytes;
        const float sum = c->d == 4u ? arm_base__zzprivate_dot4(row, c->xf, c->cols, t) * arm_base__zzprivate_row_scale(c->luts, r)
                        : c->d == 8u ? arm_base__zzprivate_dot8(row, c->xf, c->cols) * arm_base__zzprivate_row_scale(c->luts, r)
                                     : arm_base__zzprivate_dot16((const uint16_t*)(const void*)row, c->xf, c->cols);
        c->out[r] = nn__kernels__zzabi_to_half(sum, c->over);
    }
}

/* ── the call's own scratch: x widened, grown as needed, one a calling thread ─────────────────────────────────────
 * ⛔ AND GIVEN BACK WHEN ITS THREAD ENDS, as `x86_avx2`'s: the engine runs a launch's blocks on threads it starts for
 *   that launch, so a scratch kept past one is lost; a key's destructor frees it as the thread ends. */
static __thread float*   arm_base__zzprivate_xf = 0;
static __thread uint64_t arm_base__zzprivate_x_room = 0;
static sys__os__key  arm_base__zzprivate_scratch_key;
static sys__os__once arm_base__zzprivate_scratch_once = SYS__OS__ONCE_INIT;
static __thread int8_t*  arm_base__zzprivate_xq = 0;     /* the int8 GEMM's quantised rows and their scales (▶ below) */
static __thread float*   arm_base__zzprivate_xs = 0;
static __thread uint64_t arm_base__zzprivate_q_room = 0;
static __thread float*   arm_base__zzprivate_wf = 0;     /* the expert GEMM's decoded rows, a pool thread's (▶ below) */
static __thread uint64_t arm_base__zzprivate_w_room = 0;
static void arm_base__zzprivate_scratch_free(void* held) {
    (void)held;
    sys__os__memory_aligned_free(arm_base__zzprivate_xf);
    sys__os__memory_aligned_free(arm_base__zzprivate_xq); sys__os__memory_aligned_free(arm_base__zzprivate_xs);
    sys__os__memory_aligned_free(arm_base__zzprivate_wf);
    arm_base__zzprivate_xf = 0; arm_base__zzprivate_x_room = 0;
    arm_base__zzprivate_xq = 0; arm_base__zzprivate_xs = 0; arm_base__zzprivate_q_room = 0;
    arm_base__zzprivate_wf = 0; arm_base__zzprivate_w_room = 0;
}
static void arm_base__zzprivate_scratch_key_make(void) {
    (void)sys__os__key_make(&arm_base__zzprivate_scratch_key, arm_base__zzprivate_scratch_free);
}
static inline bool arm_base__zzprivate_x_room_for(uint64_t cols) {
    if (cols <= arm_base__zzprivate_x_room) return true;
    sys__os__once_run(&arm_base__zzprivate_scratch_once, arm_base__zzprivate_scratch_key_make);
    sys__os__key_set(arm_base__zzprivate_scratch_key, (void*)1);
    sys__os__memory_aligned_free(arm_base__zzprivate_xf);
    arm_base__zzprivate_xf = (float*)sys__os__memory_aligned(64, ((cols * 4u + 63u) / 64u) * 64u);
    arm_base__zzprivate_x_room = arm_base__zzprivate_xf ? cols : 0u;
    return arm_base__zzprivate_x_room != 0u;
}
/* `n` halves of `x` widened into `to`. */
static inline void arm_base__zzprivate_widen(const uint16_t* x, float* to, uint64_t n) {
    uint64_t k = 0;
    for (; k + 8u <= n; k += 8u) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(x + k));
        vst1q_f32(to + k, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(to + k + 4u, vcvt_high_f32_f16(h));
    }
    for (; k < n; ++k) to[k] = nn__silicon__half_to_float(x[k]);
}
/* `n` halves of `x` widened into the scratch. */
static inline void arm_base__zzprivate_widen_x(const uint16_t* x, uint64_t n) { arm_base__zzprivate_widen(x, arm_base__zzprivate_xf, n); }
/* How many blocks to cut a call's rows into: a few a thread, so a slow one does not hold the rest. ⛔ NEVER NONE: an empty
 * pool (a busy machine's) is the caller running the blocks itself, as `x86_avx2`'s. */
static inline uint32_t arm_base__zzprivate_blocks_for(uint64_t rows) {
    const uint32_t threads = arm_base__zzpackage_threads();
    const uint64_t want = 4ull * (threads > 0u ? threads : 1u);
    return (uint32_t)(rows < want ? rows : want);
}

/* ⭐ THE EXACT GEMV: 4, 8 and 16 bits with whole 32-column runs; anything else is the generic door. */
static void arm_base__override_turboquant_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                               uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                               unsigned int* over) {
    if ((d == 4u || d == 8u) && arm_base__zzprivate_int8_everywhere()) {
        arm_base__override_turboquant_gemv_int8(out, weights, w_room, luts, l_room, x, d, rows, cols, over); return;
    }
    const uint64_t row_bytes = nn__turboquant__row_bytes(d, cols);
    const bool fits = (d == 4u || d == 8u || d == 16u) && cols % 32u == 0u && row_bytes != 0u && rows <= w_room / row_bytes
                   && (d == 16u || rows <= l_room / 2u) && arm_base__zzprivate_x_room_for(cols);
    if (!fits) { nn__turboquant__zzabi_launch_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over); return; }
    sys__os__once_run(&arm_base__zzprivate_levels8_once, arm_base__zzprivate_levels8_make);
    (void)arm_base__zzpackage_start();
    arm_base__zzprivate_widen_x(x, cols);
    arm_base__gemv_call c = { out, weights, luts, row_bytes, rows, cols, d, arm_base__zzprivate_xf, over };
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(rows), arm_base__zzprivate_exact, &c);
}

/* ── THE GROUPED GEMVS: several matrices' rows against one `x` (the gate and up), and every matrix's row `r` weighted
 *   onto the residual (the down) — `x86_avx2`'s two bodies, the rows through the dots above. */
typedef struct arm_base__groups_call {
    uint16_t* out; nn__turboquant__groups g; const float* xf; const uint16_t* w; const uint16_t* residual;
    uint64_t rows, cols; unsigned int* over;
} arm_base__groups_call;

static inline float arm_base__zzprivate_group_row(const nn__turboquant__groups* g, uint64_t i, uint64_t r, const float* xf,
                                                  uint64_t cols, arm_base__tables4 t) {
    const uint8_t* row = (const uint8_t*)(uintptr_t)g->codes[i] + r * (cols * g->d[i] / 8u);
    const float dot = g->d[i] == 4u ? arm_base__zzprivate_dot4(row, xf, cols, t) : arm_base__zzprivate_dot8(row, xf, cols);
    return dot * arm_base__zzprivate_row_scale((const uint8_t*)(uintptr_t)g->luts[i], r);
}
/* the gate_ups: every group's rows, one `x`, the flat row range of this block */
static void arm_base__zzprivate_groups(void* a) {
    const arm_base__groups_call* c = (const arm_base__groups_call*)a;
    const arm_base__tables4 t = arm_base__zzprivate_tables4();
    uint64_t t0, t1; arm_base__zzprivate_rows(c->rows, &t0, &t1);
    uint64_t i = 0u, base = 0u;
    for (uint64_t k = t0; k < t1; ++k) {
        while (k >= base + c->g.rows[i]) { base += c->g.rows[i]; ++i; }
        const uint64_t r = k - base;
        c->out[c->g.out_at[i] + r] = nn__kernels__zzabi_to_half(arm_base__zzprivate_group_row(&c->g, i, r, c->xf, c->cols, t), c->over);
    }
}
/* the downs: every group's row `r` weighted onto the residual */
static void arm_base__zzprivate_groups_sum(void* a) {
    const arm_base__groups_call* c = (const arm_base__groups_call*)a;
    const arm_base__tables4 t = arm_base__zzprivate_tables4();
    uint64_t r0, r1; arm_base__zzprivate_rows(c->rows, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < c->g.count; ++i)
            sum += arm_base__zzprivate_group_row(&c->g, i, r, c->xf + i * c->cols, c->cols, t) * nn__silicon__half_to_float(c->w[c->g.out_at[i]]);
        c->out[r] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(c->residual[r]) + sum, c->over);
    }
}
static inline bool arm_base__zzprivate_groups_fit(const nn__turboquant__groups* g, uint64_t cols) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX || cols == 0u || cols % 32u != 0u) return false;
    for (uint64_t i = 0u; i < g->count; ++i) if ((g->d[i] != 4u && g->d[i] != 8u) || g->rows[i] == 0u) return false;
    return true;
}
static void arm_base__override_gemv_groups(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, unsigned int* over) {
    if (arm_base__zzprivate_groups_four(&g) && arm_base__zzprivate_int8_everywhere()) {
        arm_base__override_gemv_groups_int8(out, g, x, cols, over); return;
    }
    if (!arm_base__zzprivate_groups_fit(&g, cols) || !arm_base__zzprivate_x_room_for(cols)) {
        nn__turboquant__zzabi_launch_gemv_groups(out, g, x, cols, over); return;
    }
    sys__os__once_run(&arm_base__zzprivate_levels8_once, arm_base__zzprivate_levels8_make);
    (void)arm_base__zzpackage_start();
    arm_base__zzprivate_widen_x(x, cols);
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    arm_base__groups_call c = { out, g, arm_base__zzprivate_xf, 0, 0, total, cols, over };
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(total), arm_base__zzprivate_groups, &c);
}
/* The groups over `n` inputs: each input through the one-input door above, as `x86_avx2`'s — each answer that door's bytes. */
static void arm_base__override_gemv_groups_n(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                             uint64_t n, uint64_t out_stride, unsigned int* over) {
    for (uint64_t k = 0u; k < n; ++k) {
        uint16_t* o = out;
        const nn__turboquant__groups gk = nn__turboquant__groups_input(&g, &o, k, out_stride);
        arm_base__override_gemv_groups(o, gk, x + k * cols, cols, over);
    }
}
static void arm_base__override_gemv_groups_sum(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                               const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (arm_base__zzprivate_groups_four(&g) && arm_base__zzprivate_int8_everywhere()) {
        arm_base__override_gemv_groups_sum_int8(out, g, x, w, residual, rows, cols, over); return;
    }
    if (!arm_base__zzprivate_groups_fit(&g, cols) || !arm_base__zzprivate_x_room_for(g.count * cols)) {
        nn__turboquant__zzabi_launch_gemv_groups_sum(out, g, x, w, residual, rows, cols, over); return;
    }
    sys__os__once_run(&arm_base__zzprivate_levels8_once, arm_base__zzprivate_levels8_make);
    (void)arm_base__zzpackage_start();
    arm_base__zzprivate_widen_x(x, g.count * cols);
    arm_base__groups_call c = { out, g, arm_base__zzprivate_xf, w, residual, rows, cols, over };
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(rows), arm_base__zzprivate_groups_sum, &c);
}

/* ── ⭐⭐ THE EXPERT-MAJOR GEMM — `expert_rows`, `expert_rows_sum`, `expert_groups`: a prompt's routed experts ──────────
 * nn's definitions (`kernels.cuh`), `x86_avx2`'s body in NEON. A pool block takes a run of a matrix's rows and decodes them
 * three at a time into floats — the levels the gemvs find — in its thread's scratch; each pair's `x` was widened to floats
 * once for the call. The three rows meet four pairs at a time in eight passes: pass `e` takes, of each 32 columns, the four
 * a gemv's sum `e` takes (▶ `arm_base__zzprivate_sum8`), so a (row, pair)'s eight sums, and their adding up, are the
 * gemv's and its answer the gemv's bits — twelve sums in flight, a load of a row serving four products and a load of a
 * pair's `x` three. One block owns a row for every pair, so the sums need no lock. The pairs go in blocks whose `x` fits
 * half the L2, each through every row of the run before the next. 4, 8 and 16 bits (16 the routers': a code is the
 * weight, no scale), whole runs of 32 columns up to 32,768; anything else is the generic door. */
#define ARM_BASE__ZZPRIVATE_GEMM_COLS 32768u
#define ARM_BASE__ZZPRIVATE_TILE_ROWS 3u

typedef struct arm_base__rows_call {
    uint16_t* out; float* acc; const uint8_t* weights; const uint8_t* luts; const float* xf; const uint32_t* rows;
    const uint16_t* w; uint64_t count, d, out_rows, cols; unsigned int* over;
    nn__expert__groups g; uint64_t total;   /* several matrices: `g`, and all their rows */
    uint64_t xf_at[NN__EXPERT__GROUPS_MAX]; /* where each group's widened pairs begin in `xf` */
} arm_base__rows_call;

/* The L2 a core reports, 512 KiB where it reports none — read once. */
static uint64_t arm_base__zzprivate_l2 = 512u * 1024u;
static sys__os__once arm_base__zzprivate_l2_once = SYS__OS__ONCE_INIT;
static void arm_base__zzprivate_l2_read(void) {
    const long l2 = sys__os__cpu_l2_bytes();
    if (l2 > 0) arm_base__zzprivate_l2 = (uint64_t)l2;
}
/* How many pairs a block of them holds: their `x` half the L2, four at least, in fours. */
static inline uint64_t arm_base__zzprivate_pair_block(uint64_t cols) {
    const uint64_t fit = arm_base__zzprivate_l2 / 2u / (cols * 4u);
    return fit < 4u ? 4u : fit - fit % 4u;
}
/* A pool thread's room for `n` decoded floats, grown as needed; none when it cannot be had. */
static inline float* arm_base__zzprivate_rows_room(uint64_t n) {
    if (n <= arm_base__zzprivate_w_room) return arm_base__zzprivate_wf;
    sys__os__once_run(&arm_base__zzprivate_scratch_once, arm_base__zzprivate_scratch_key_make);
    sys__os__key_set(arm_base__zzprivate_scratch_key, (void*)1);
    sys__os__memory_aligned_free(arm_base__zzprivate_wf);
    arm_base__zzprivate_wf = (float*)sys__os__memory_aligned(64, ((n * 4u + 63u) / 64u) * 64u);
    arm_base__zzprivate_w_room = arm_base__zzprivate_wf ? n : 0u;
    return arm_base__zzprivate_wf;
}
/* A matrix row decoded into `wf`, `cols` floats — each level as the gemv's dot multiplies it. */
static inline void arm_base__zzprivate_decode_row(uint64_t d, const uint8_t* row, uint64_t cols, float* wf, arm_base__tables4 t) {
    if (d == 16u) {
        const uint16_t* h = (const uint16_t*)(const void*)row;
        for (uint64_t k = 0; k < cols; k += 8u) {
            const float16x8_t v = vreinterpretq_f16_u16(vld1q_u16(h + k));
            vst1q_f32(wf + k, vcvt_f32_f16(vget_low_f16(v)));
            vst1q_f32(wf + k + 4u, vcvt_high_f32_f16(v));
        }
    } else if (d == 4u) {
        const uint8x16_t fifteen = vdupq_n_u8(15u);
        for (uint64_t k = 0; k < cols; k += 32u) {
            const uint8x16_t v = vld1q_u8(row + k / 2u);
            const uint8x16_t lo = vandq_u8(v, fifteen), hi = vshrq_n_u8(v, 4);
            const uint8x16_t i0 = vzip1q_u8(lo, hi), i1 = vzip2q_u8(lo, hi);
            const uint8x16_t bl0 = vqtbl1q_u8(t.lo, i0), bh0 = vqtbl1q_u8(t.hi, i0);
            const uint8x16_t bl1 = vqtbl1q_u8(t.lo, i1), bh1 = vqtbl1q_u8(t.hi, i1);
            const float16x8_t h0 = vreinterpretq_f16_u8(vzip1q_u8(bl0, bh0)), h1 = vreinterpretq_f16_u8(vzip2q_u8(bl0, bh0));
            const float16x8_t h2 = vreinterpretq_f16_u8(vzip1q_u8(bl1, bh1)), h3 = vreinterpretq_f16_u8(vzip2q_u8(bl1, bh1));
            vst1q_f32(wf + k,       vcvt_f32_f16(vget_low_f16(h0))); vst1q_f32(wf + k + 4u,  vcvt_high_f32_f16(h0));
            vst1q_f32(wf + k + 8u,  vcvt_f32_f16(vget_low_f16(h1))); vst1q_f32(wf + k + 12u, vcvt_high_f32_f16(h1));
            vst1q_f32(wf + k + 16u, vcvt_f32_f16(vget_low_f16(h2))); vst1q_f32(wf + k + 20u, vcvt_high_f32_f16(h2));
            vst1q_f32(wf + k + 24u, vcvt_f32_f16(vget_low_f16(h3))); vst1q_f32(wf + k + 28u, vcvt_high_f32_f16(h3));
        }
    } else {
        const float* lev = arm_base__zzprivate_levels8;
        for (uint64_t k = 0; k < cols; ++k) wf[k] = lev[row[k]];
    }
}
/* `NR` decoded rows against `N` pairs' `x`, in the eight passes: `v[a][q]` row `a`'s dot with pair `q`. Called with
 * literal `NR` and `N` and always inlined, so each call's sums stay in registers. */
#define ARM_BASE__ZZPRIVATE_INLINED __attribute__((always_inline))
static inline ARM_BASE__ZZPRIVATE_INLINED void arm_base__zzprivate_tile(const float* const* w, uint64_t NR, const float* const* xq,
                                                                           uint64_t N, uint64_t cols, float v[3][4]) {
    float32x4_t keep[8][3][4];
    for (uint64_t e = 0; e < 8u; ++e) {
        float32x4_t s[3][4];
        for (uint64_t a = 0; a < NR; ++a) for (uint64_t q = 0; q < N; ++q) s[a][q] = vdupq_n_f32(0.0f);
        for (uint64_t k = 4u * e; k < cols; k += 32u) {
            float32x4_t wv[3];
            for (uint64_t a = 0; a < NR; ++a) wv[a] = vld1q_f32(w[a] + k);
            for (uint64_t q = 0; q < N; ++q) {
                const float32x4_t x = vld1q_f32(xq[q] + k);
                for (uint64_t a = 0; a < NR; ++a) s[a][q] = vfmaq_f32(s[a][q], wv[a], x);
            }
        }
        for (uint64_t a = 0; a < NR; ++a) for (uint64_t q = 0; q < N; ++q) keep[e][a][q] = s[a][q];
    }
    for (uint64_t a = 0; a < NR; ++a)
        for (uint64_t q = 0; q < N; ++q)
            v[a][q] = arm_base__zzprivate_sum8(keep[0][a][q], keep[1][a][q], keep[2][a][q], keep[3][a][q],
                                               keep[4][a][q], keep[5][a][q], keep[6][a][q], keep[7][a][q]);
}
static inline void arm_base__zzprivate_tile_any(const float* const* w, uint64_t nr, const float* const* xq, uint64_t n, uint64_t cols,
                                                float v[3][4]) {
#define ARM_BASE__ZZPRIVATE_TILE_N(NR)                                                                                       \
    switch (n) {                                                                                                             \
    case 4u: arm_base__zzprivate_tile(w, NR, xq, 4u, cols, v); break;                                                        \
    case 3u: arm_base__zzprivate_tile(w, NR, xq, 3u, cols, v); break;                                                        \
    case 2u: arm_base__zzprivate_tile(w, NR, xq, 2u, cols, v); break;                                                        \
    default: arm_base__zzprivate_tile(w, NR, xq, 1u, cols, v); break;                                                        \
    }
    if (nr == 3u)      { ARM_BASE__ZZPRIVATE_TILE_N(3u) }
    else if (nr == 2u) { ARM_BASE__ZZPRIVATE_TILE_N(2u) }
    else               { ARM_BASE__ZZPRIVATE_TILE_N(1u) }
#undef ARM_BASE__ZZPRIVATE_TILE_N
}
static inline float arm_base__zzprivate_gemm_scale(const uint8_t* luts, uint64_t d, uint64_t r) {
    return d == 16u ? 1.0f : arm_base__zzprivate_row_scale(luts, r);
}
/* A pair's product with a row, weighted into the running sum or rounded into `out` — nn's last step. */
static inline void arm_base__zzprivate_put(float v, const uint32_t* rows, const uint16_t* w, uint64_t j, uint64_t out_rows, uint64_t r,
                                           uint16_t* out, float* acc, unsigned int* over) {
    const uint64_t at = (uint64_t)rows[2u * j + 1u] * out_rows + r;
    if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
    else acc[at] += v * nn__silicon__half_to_float(w[rows[2u * j]]);
}
/* Rows `r0 .. r1` of one matrix against its `count` pairs (pair `j`'s `x` at `xf + j·cols`): the pairs a block at a time,
 * the rows three at a time — or, with no room for their floats, each (row, pair) the gemv's dot from the codes. */
static inline void arm_base__zzprivate_matrix_rows(uint64_t d, const uint8_t* codes, const uint8_t* luts, const float* xf,
                                                   const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t cols,
                                                   uint64_t out_rows, uint64_t r0, uint64_t r1, uint16_t* out, float* acc,
                                                   unsigned int* over, arm_base__tables4 t) {
    const uint64_t row_bytes = cols * d / 8u, block = arm_base__zzprivate_pair_block(cols);
    float* room = count != 0u ? arm_base__zzprivate_rows_room(ARM_BASE__ZZPRIVATE_TILE_ROWS * cols) : 0;
    if (!room) {
        for (uint64_t r = r0; r < r1; ++r) {
            const uint8_t* row = codes + r * row_bytes;
            const float scale = arm_base__zzprivate_gemm_scale(luts, d, r);
            for (uint64_t j = 0; j < count; ++j) {
                const float* x = xf + j * cols;
                const float v = d == 4u ? arm_base__zzprivate_dot4(row, x, cols, t) : d == 8u ? arm_base__zzprivate_dot8(row, x, cols)
                              : arm_base__zzprivate_dot16((const uint16_t*)(const void*)row, x, cols);
                arm_base__zzprivate_put(v * scale, rows, w, j, out_rows, r, out, acc, over);
            }
        }
        return;
    }
    for (uint64_t j0 = 0; j0 < count; j0 += block) {
        const uint64_t m = count - j0 < block ? count - j0 : block;
        for (uint64_t r = r0; r < r1; r += ARM_BASE__ZZPRIVATE_TILE_ROWS) {
            const uint64_t nr = r1 - r < ARM_BASE__ZZPRIVATE_TILE_ROWS ? r1 - r : ARM_BASE__ZZPRIVATE_TILE_ROWS;
            const float* wf[3] = { room, room, room };
            float scale[3] = { 0.0f, 0.0f, 0.0f };
            for (uint64_t a = 0; a < nr; ++a) {
                arm_base__zzprivate_decode_row(d, codes + (r + a) * row_bytes, cols, room + a * cols, t);
                wf[a] = room + a * cols;
                scale[a] = arm_base__zzprivate_gemm_scale(luts, d, r + a);
            }
            for (uint64_t j = j0; j < j0 + m; j += 4u) {
                const uint64_t n = j0 + m - j < 4u ? j0 + m - j : 4u;
                const float* xq[4] = { xf + j * cols, xf + (j + (n > 1u ? 1u : 0u)) * cols, xf + (j + (n > 2u ? 2u : 0u)) * cols,
                                       xf + (j + (n > 3u ? 3u : 0u)) * cols };
                float v[3][4];
                arm_base__zzprivate_tile_any(wf, nr, xq, n, cols, v);
                for (uint64_t a = 0; a < nr; ++a)
                    for (uint64_t q = 0; q < n; ++q) arm_base__zzprivate_put(v[a][q] * scale[a], rows, w, j + q, out_rows, r + a, out, acc, over);
            }
        }
    }
}
static void arm_base__zzprivate_expert_rows(void* a) {
    const arm_base__rows_call* c = (const arm_base__rows_call*)a;
    const arm_base__tables4 t = arm_base__zzprivate_tables4();
    uint64_t r0, r1; arm_base__zzprivate_rows(c->out_rows, &r0, &r1);
    if (r0 < r1)
        arm_base__zzprivate_matrix_rows(c->d, c->weights, c->luts, c->xf, c->rows, c->w, c->count, c->cols, c->out_rows, r0, r1,
                                        c->out, c->acc, c->over, t);
}
/* The block's share of every group's rows, laid end to end: each group's part of it at once. */
static void arm_base__zzprivate_expert_groups(void* a) {
    const arm_base__rows_call* c = (const arm_base__rows_call*)a;
    const arm_base__tables4 t = arm_base__zzprivate_tables4();
    uint64_t t0, t1; arm_base__zzprivate_rows(c->total, &t0, &t1);
    uint64_t i = 0u, base = 0u;
    while (t0 < t1) {
        while (t0 >= base + c->g.out_rows[i]) { base += c->g.out_rows[i]; ++i; }
        const uint64_t end = base + c->g.out_rows[i] < t1 ? base + c->g.out_rows[i] : t1;
        arm_base__zzprivate_matrix_rows(c->g.d[i], (const uint8_t*)(uintptr_t)c->g.codes[i], (const uint8_t*)(uintptr_t)c->g.luts[i],
                                        c->xf + c->xf_at[i], c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i], c->cols,
                                        c->g.out_rows[i], t0 - base, end - base, c->out + c->g.out_at[i], 0, c->over, t);
        t0 = end;
    }
}
/* Pairs' `x` rows widened to floats, pair `j` at `xf + j·cols`. */
static inline void arm_base__zzprivate_widen_pairs(float* xf, const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t cols) {
    for (uint64_t j = 0; j < count; ++j) arm_base__zzprivate_widen(x + (uint64_t)rows[2u * j] * cols, xf + j * cols, cols);
}
static inline bool arm_base__zzprivate_gemm_fit(uint64_t d, uint64_t cols) {
    return (d == 4u || d == 8u || d == 16u) && cols != 0u && cols % 32u == 0u && cols <= ARM_BASE__ZZPRIVATE_GEMM_COLS;
}
static inline void arm_base__zzprivate_gemm_ready(void) {
    sys__os__once_run(&arm_base__zzprivate_levels8_once, arm_base__zzprivate_levels8_make);
    sys__os__once_run(&arm_base__zzprivate_l2_once, arm_base__zzprivate_l2_read);
    (void)arm_base__zzpackage_start();
}
static void arm_base__override_expert_rows(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                           const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d, uint64_t out_rows,
                                           uint64_t cols, unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (!arm_base__zzprivate_gemm_fit(d, cols) || count == 0u || out_rows > w_room / row_bytes || (d != 16u && out_rows > l_room / 2u)
     || !arm_base__zzprivate_x_room_for(count * cols)) {
        nn__expert__zzabi_launch_rows(out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over); return;
    }
    arm_base__zzprivate_gemm_ready();
    arm_base__zzprivate_widen_pairs(arm_base__zzprivate_xf, x, rows, count, cols);
    arm_base__rows_call c = {};
    c.out = out; c.weights = weights; c.luts = luts; c.xf = arm_base__zzprivate_xf; c.rows = rows;
    c.count = count; c.d = d; c.out_rows = out_rows; c.cols = cols; c.over = over;
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(out_rows), arm_base__zzprivate_expert_rows, &c);
}
static void arm_base__override_expert_rows_sum(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                               const uint16_t* x, const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t d,
                                               uint64_t out_rows, uint64_t cols) {
    const uint64_t row_bytes = cols * d / 8u;
    if (!arm_base__zzprivate_gemm_fit(d, cols) || count == 0u || out_rows > w_room / row_bytes || (d != 16u && out_rows > l_room / 2u)
     || !arm_base__zzprivate_x_room_for(count * cols)) {
        nn__expert__zzabi_launch_rows_sum(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols); return;
    }
    arm_base__zzprivate_gemm_ready();
    arm_base__zzprivate_widen_pairs(arm_base__zzprivate_xf, x, rows, count, cols);
    arm_base__rows_call c = {};
    c.acc = acc; c.weights = weights; c.luts = luts; c.xf = arm_base__zzprivate_xf; c.rows = rows; c.w = w;
    c.count = count; c.d = d; c.out_rows = out_rows; c.cols = cols;
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(out_rows), arm_base__zzprivate_expert_rows, &c);
}
static void arm_base__override_expert_groups(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows, uint64_t cols,
                                             unsigned int* over) {
    bool fit = g.count != 0u && g.count <= NN__EXPERT__GROUPS_MAX;
    uint64_t total = 0u, pairs = 0u;
    for (uint64_t i = 0u; fit && i < g.count; ++i) {
        fit = arm_base__zzprivate_gemm_fit(g.d[i], cols) && g.out_rows[i] != 0u;
        total += g.out_rows[i]; pairs += g.pairs[i];
    }
    if (!fit || !arm_base__zzprivate_x_room_for(pairs * cols)) { nn__expert__zzabi_launch_groups(out, g, x, rows, cols, over); return; }
    arm_base__zzprivate_gemm_ready();
    arm_base__rows_call c = {};
    uint64_t placed = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) {
        c.xf_at[i] = placed * cols;
        arm_base__zzprivate_widen_pairs(arm_base__zzprivate_xf + placed * cols, x, rows + 2u * g.pairs_at[i], g.pairs[i], cols);
        placed += g.pairs[i];
    }
    c.out = out; c.xf = arm_base__zzprivate_xf; c.rows = rows; c.cols = cols; c.over = over; c.g = g; c.total = total;
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(total), arm_base__zzprivate_expert_groups, &c);
}

/* ── ⭐ THE EXPERT-MAJOR GEMM IN INTEGERS — `expert_groups_int8`, `expert_rows_sum_int8`, a prompt's routed experts ─────
 * nn's definition (`kernels.cuh`): each row of `x` in blocks of 256 quantised to int8 by its largest magnitude, the 4-bit
 * codes as their int8 levels, a block's products summed in integers and scaled once, the row's scale and 1/127 last. nn's
 * body quantises a block again for every row and pair; here the rows the pairs name are quantised once a call, a matrix
 * row's codes made levels once (`tbl`), and a block's products summed by Armv8.2's `sdot` — its lanes' sums the block's
 * exact integer sum, nn's — then the float steps in nn's order: a block's (one fused multiply-add), then the row's scale
 * and 1/127. A family with a faster way to the same integer sums hands its own rows to the same bodies
 * (`arm_base__int8_rows_fn`): the sums being exact, its answers are these bytes (▶ `arm_gb10/overrides/`). */
#define ARM_BASE__INT8_COLS 32768u   /* a row's levels are held on the stack: 32 KiB at most */

/* One matrix of a call, as a family's rows see it: its codes, its scales, its pairs, where its answers go. */
typedef struct arm_base__int8_matrix {
    const uint8_t* codes; const uint8_t* luts; const uint32_t* rows; const uint16_t* w; uint64_t count, out_rows, cols;
    const int8_t* xq; const float* xs; uint64_t x0;      /* the call's quantised rows of `x`, from row x0 */
    uint16_t* out; float* acc; unsigned int* over;       /* the answers: halves into `out`, or added into `acc` */
} arm_base__int8_matrix;
/* Rows `r0 .. r1` of one matrix against all its pairs — a family's way of making them. */
typedef void (*arm_base__int8_rows_fn)(const arm_base__int8_matrix* m, uint64_t r0, uint64_t r1);

typedef struct arm_base__int8_call {
    const uint16_t* x; int8_t* xq; float* xs; uint64_t cols, x0, nx;                  /* rows x0 .. x0 + nx, quantised */
    uint16_t* out; float* acc; const uint8_t* weights; const uint8_t* luts; const uint32_t* rows; const uint16_t* w;
    uint64_t count, out_rows;                                                          /* one matrix */
    nn__expert__groups g; uint64_t total;                                              /* several */
    unsigned int* over; arm_base__int8_rows_fn rows_fn;
} arm_base__int8_call;

/* The scale of matrix row `r` — its half in `luts`, as a float. */
static inline float arm_base__int8_row_scale(const uint8_t* luts, uint64_t r) {
    return nn__silicon__half_to_float((uint16_t)((uint32_t)luts[r * 2u] | ((uint32_t)luts[r * 2u + 1u] << 8)));
}
/* The 4-bit levels as int8, for `tbl`. */
static inline int8x16_t arm_base__int8_levels(void) { return vld1q_s8(nn__turboquant__zzabi_levels_i8(4u)); }
/* Thirty-two columns of a row's codes as their int8 levels — columns 0..15 into `a`, 16..31 into `b`. */
static inline void arm_base__int8_levels32(const uint8_t* codes, int8x16_t t8, int8x16_t* a, int8x16_t* b) {
    const uint8x16_t v = vld1q_u8(codes);
    const uint8x16_t lo = vandq_u8(v, vdupq_n_u8(15u)), hi = vshrq_n_u8(v, 4);
    *a = vqtbl1q_s8(t8, vzip1q_u8(lo, hi));
    *b = vqtbl1q_s8(t8, vzip2q_u8(lo, hi));
}
/* A block's float step: its exact integer sum `dot` at `x`'s block scale `s`, added to the running `part`. */
static inline float arm_base__int8_step(float part, int32_t dot, float s) { return fmaf((float)dot, s, part); }
/* Pair `j`'s answer for matrix row `r`, from its sum over the blocks: nn's last two steps, then nn's store. */
static inline void arm_base__int8_finish(const arm_base__int8_matrix* m, uint64_t j, uint64_t r, float part, float scale) {
    const float v = part * scale / 127.0f;
    const uint64_t at = (uint64_t)m->rows[2u * j + 1u] * m->out_rows + r;
    if (m->out) m->out[at] = nn__kernels__zzabi_to_half(v, m->over);
    else m->acc[at] += v * nn__silicon__half_to_float(m->w[m->rows[2u * j]]);
}
/* Rows `r0 .. r1` through `sdot`: a row's levels made once, then each pair's blocks, two running sums a block. */
static void arm_base__int8_rows(const arm_base__int8_matrix* m, uint64_t r0, uint64_t r1) {
    const int8x16_t t8 = arm_base__int8_levels();
    const uint64_t cols = m->cols, per = cols / NN__EXPERT__INT8_BLOCK;
    int8_t w8[cols] __attribute__((aligned(16)));
    for (uint64_t r = r0; r < r1; ++r) {
        const uint8_t* row = m->codes + r * (cols / 2u);
        for (uint64_t k = 0; k < cols; k += 32u) {
            int8x16_t a, b; arm_base__int8_levels32(row + k / 2u, t8, &a, &b);
            vst1q_s8(w8 + k, a); vst1q_s8(w8 + k + 16u, b);
        }
        const float scale = arm_base__int8_row_scale(m->luts, r);
        for (uint64_t j = 0; j < m->count; ++j) {
            const uint64_t xrow = (uint64_t)m->rows[2u * j] - m->x0;
            const int8_t* xr = m->xq + xrow * cols;
            const float* sr = m->xs + xrow * per;
            float part = 0.0f;
            for (uint64_t B = 0; B < per; ++B) {
                int32x4_t s0 = vdupq_n_s32(0), s1 = s0;
                for (uint64_t k = B * NN__EXPERT__INT8_BLOCK; k < (B + 1u) * NN__EXPERT__INT8_BLOCK; k += 32u) {
                    s0 = vdotq_s32(s0, vld1q_s8(w8 + k), vld1q_s8(xr + k));
                    s1 = vdotq_s32(s1, vld1q_s8(w8 + k + 16u), vld1q_s8(xr + k + 16u));
                }
                part = arm_base__int8_step(part, vaddvq_s32(vaddq_s32(s0, s1)), sr[B]);
            }
            arm_base__int8_finish(m, j, r, part, scale);
        }
    }
}

/* Block `x[0..256)` quantised into `q` as nn's definition says — the same operations in the same order; its scale answered. */
static inline float arm_base__zzprivate_quantise256(const uint16_t* x, int8_t* q) {
    float mx = 0.0f;
    for (uint32_t i = 0u; i < NN__EXPERT__INT8_BLOCK; ++i) {
        const float v = nn__silicon__half_to_float(x[i]);
        const float a = v < 0.0f ? -v : v;
        if (a > mx) mx = a;
    }
    const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
    for (uint32_t i = 0u; i < NN__EXPERT__INT8_BLOCK; ++i) {
        const float t = nn__silicon__half_to_float(x[i]) * inv;
        q[i] = (int8_t)(t >= 0.0f ? (int32_t)(t + 0.5f) : -(int32_t)(-t + 0.5f));
    }
    return mx / 127.0f;
}
static void arm_base__zzprivate_quantise_rows(void* a) {
    const arm_base__int8_call* c = (const arm_base__int8_call*)a;
    const uint64_t per = c->cols / NN__EXPERT__INT8_BLOCK;
    uint64_t r0, r1; arm_base__zzprivate_rows(c->nx, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r)
        for (uint64_t k = 0; k < per; ++k)
            c->xs[r * per + k] = arm_base__zzprivate_quantise256(c->x + (c->x0 + r) * c->cols + k * NN__EXPERT__INT8_BLOCK,
                                                                 c->xq + r * c->cols + k * NN__EXPERT__INT8_BLOCK);
}
/* The call's one matrix, or group `i`'s, as its rows see it. */
static inline arm_base__int8_matrix arm_base__zzprivate_matrix(const arm_base__int8_call* c) {
    arm_base__int8_matrix m = { c->weights, c->luts, c->rows, c->w, c->count, c->out_rows, c->cols,
                                c->xq, c->xs, c->x0, c->out, c->acc, c->over };
    return m;
}
static inline arm_base__int8_matrix arm_base__zzprivate_group(const arm_base__int8_call* c, uint64_t i) {
    arm_base__int8_matrix m = { (const uint8_t*)(uintptr_t)c->g.codes[i], (const uint8_t*)(uintptr_t)c->g.luts[i],
                                c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i], c->g.out_rows[i], c->cols,
                                c->xq, c->xs, c->x0, c->out + c->g.out_at[i], 0, c->over };
    return m;
}
static void arm_base__zzprivate_expert_rows_i8(void* a) {
    const arm_base__int8_call* c = (const arm_base__int8_call*)a;
    const arm_base__int8_matrix m = arm_base__zzprivate_matrix(c);
    uint64_t r0, r1; arm_base__zzprivate_rows(c->out_rows, &r0, &r1);
    if (r0 < r1) c->rows_fn(&m, r0, r1);
}
/* The block's share of every group's rows, laid end to end: each group's part of it handed to the rows at once. */
static void arm_base__zzprivate_expert_groups_i8(void* a) {
    const arm_base__int8_call* c = (const arm_base__int8_call*)a;
    uint64_t t0, t1; arm_base__zzprivate_rows(c->total, &t0, &t1);
    uint64_t i = 0u, base = 0u;
    while (t0 < t1) {
        while (t0 >= base + c->g.out_rows[i]) { base += c->g.out_rows[i]; ++i; }
        const uint64_t end = base + c->g.out_rows[i] < t1 ? base + c->g.out_rows[i] : t1;
        const arm_base__int8_matrix m = arm_base__zzprivate_group(c, i);
        c->rows_fn(&m, t0 - base, end - base);
        t0 = end;
    }
}
/* The call's quantised rows and their scales, a calling thread's, grown as needed (freed with the widened `x`). */
static inline bool arm_base__zzprivate_q_room_for(uint64_t values) {
    if (values <= arm_base__zzprivate_q_room) return true;
    sys__os__once_run(&arm_base__zzprivate_scratch_once, arm_base__zzprivate_scratch_key_make);
    sys__os__key_set(arm_base__zzprivate_scratch_key, (void*)1);
    sys__os__memory_aligned_free(arm_base__zzprivate_xq); sys__os__memory_aligned_free(arm_base__zzprivate_xs);
    arm_base__zzprivate_xq = (int8_t*)sys__os__memory_aligned(64, ((values + 63u) / 64u) * 64u);
    arm_base__zzprivate_xs = (float*)sys__os__memory_aligned(64, ((values / 32u * 4u + 64u) / 64u) * 64u);   /* a scale a 32 */
    arm_base__zzprivate_q_room = arm_base__zzprivate_xq && arm_base__zzprivate_xs ? values : 0u;
    return arm_base__zzprivate_q_room != 0u;
}
/* The rows of `x` the pairs name — the run from the lowest to the highest — quantised once, on the pool. False when
 * there is no room for them. */
static inline bool arm_base__zzprivate_quantise_pairs(arm_base__int8_call* c, const uint16_t* x, const uint32_t* rows,
                                                      uint64_t pairs, uint64_t cols) {
    uint64_t lo = ~0ull, hi = 0u;
    for (uint64_t j = 0; j < pairs; ++j) {
        if ((uint64_t)rows[2u * j] < lo) lo = rows[2u * j];
        if ((uint64_t)rows[2u * j] + 1u > hi) hi = (uint64_t)rows[2u * j] + 1u;
    }
    if (pairs == 0u || !arm_base__zzprivate_q_room_for((hi - lo) * cols)) return false;
    c->x = x; c->xq = arm_base__zzprivate_xq; c->xs = arm_base__zzprivate_xs; c->cols = cols; c->x0 = lo; c->nx = hi - lo;
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(c->nx), arm_base__zzprivate_quantise_rows, c);
    return true;
}
static inline bool arm_base__zzprivate_int8_fit(uint64_t d, uint64_t cols) {
    return d == 4u && cols != 0u && cols % NN__EXPERT__INT8_BLOCK == 0u && cols <= ARM_BASE__INT8_COLS;
}
/* `expert_rows_sum_int8`, its rows made by `rows_fn`. */
static inline void arm_base__int8_rows_sum(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                           const uint16_t* x, const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t d,
                                           uint64_t out_rows, uint64_t cols, arm_base__int8_rows_fn rows_fn) {
    arm_base__int8_call c = {};
    if (!arm_base__zzprivate_int8_fit(d, cols) || count == 0u || out_rows > w_room / (cols / 2u) || out_rows > l_room / 2u) {
        nn__expert__zzabi_launch_rows_sum_int8(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols); return;
    }
    (void)arm_base__zzpackage_start();
    if (!arm_base__zzprivate_quantise_pairs(&c, x, rows, count, cols)) {
        nn__expert__zzabi_launch_rows_sum_int8(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols); return;
    }
    c.acc = acc; c.weights = weights; c.luts = luts; c.rows = rows; c.w = w; c.count = count; c.out_rows = out_rows;
    c.rows_fn = rows_fn;
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(out_rows), arm_base__zzprivate_expert_rows_i8, &c);
}
/* `expert_groups_int8`, its rows made by `rows_fn`. */
static inline void arm_base__int8_groups(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows, uint64_t cols,
                                         unsigned int* over, arm_base__int8_rows_fn rows_fn) {
    bool fit = g.count != 0u && g.count <= NN__EXPERT__GROUPS_MAX;
    uint64_t total = 0u, last = 0u;
    for (uint64_t i = 0u; fit && i < g.count; ++i) {
        fit = arm_base__zzprivate_int8_fit(g.d[i], cols) && g.out_rows[i] != 0u;
        total += g.out_rows[i];
        if (g.pairs_at[i] + g.pairs[i] > last) last = g.pairs_at[i] + g.pairs[i];
    }
    arm_base__int8_call c = {};
    if (fit) (void)arm_base__zzpackage_start();
    if (!fit || !arm_base__zzprivate_quantise_pairs(&c, x, rows, last, cols)) {
        nn__expert__zzabi_launch_groups_int8(out, g, x, rows, cols, over); return;
    }
    c.out = out; c.rows = rows; c.g = g; c.total = total; c.over = over; c.rows_fn = rows_fn;
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(total), arm_base__zzprivate_expert_groups_i8, &c);
}
static void arm_base__override_expert_rows_sum_int8(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, const uint32_t* rows, const uint16_t* w,
                                                    uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols) {
    arm_base__int8_rows_sum(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols, arm_base__int8_rows);
}
static void arm_base__override_expert_groups_int8(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows,
                                                  uint64_t cols, unsigned int* over) {
    arm_base__int8_groups(out, g, x, rows, cols, over, arm_base__int8_rows);
}

/* ── THE GEMVS IN INTEGERS — `turboquant_gemv_int8`, `_groups_int8`, `_groups_sum_int8`, a token's with `experts_int8` ──
 * nn's int8 gemv (`kernels.cuh`): `x` quantised in blocks of 32 by nn's own (`nn__turboquant__zzabi_quantise`), once a call —
 * every group's own `x` for the weighted sum — the 4-bit codes as their int8 levels (`tbl`), a block's 32 products through
 * two `sdot`s, its exact integer sum, and nn's float steps as nn writes them, a block after another, the row's scale and
 * 1/127 last: nn's bytes. 4 bits, and the single gemv at 8 (a head): its 256 levels through four 64-entry tables, `tbl`
 * then three `tbx`; anything else is the generic door. */
typedef struct arm_base__gemv_i8_call {
    uint16_t* out; const uint8_t* weights; const uint8_t* luts; nn__turboquant__groups g; const int8_t* xq; const float* xs;
    const uint16_t* w; const uint16_t* residual; uint64_t rows, cols; unsigned int* over;
} arm_base__gemv_i8_call;
/* A row of 4-bit codes against the quantised `x`: nn's sum over its blocks. */
static inline float arm_base__zzprivate_dot_i8_4(const uint8_t* row, const int8_t* xq, const float* xs, uint64_t cols, int8x16_t t8) {
    float part = 0.0f;
    for (uint64_t k = 0; k < cols; k += 32u) {
        int8x16_t a, b;
        arm_base__int8_levels32(row + k / 2u, t8, &a, &b);
        const int32x4_t s = vdotq_s32(vdotq_s32(vdupq_n_s32(0), a, vld1q_s8(xq + k)), b, vld1q_s8(xq + k + 16u));
        part += (float)vaddvq_s32(s) * xs[k / 32u];
    }
    return part;
}
/* The 8-bit levels as int8: four tables of 64 levels, for `tbl` and `tbx`, and whether they are their mirror — a code
 * past 127 the negative of code 255 - c's, as nn's generated levels are; the table is checked, not assumed. */
typedef struct arm_base__int8_tables8 { int8x16x4_t t[4]; bool mirrored; } arm_base__int8_tables8;
static inline arm_base__int8_tables8 arm_base__int8_levels8(void) {
    const int8_t* lev = nn__turboquant__zzabi_levels_i8(8u);
    arm_base__int8_tables8 t;
    for (int q = 0; q < 4; ++q) t.t[q] = vld1q_s8_x4(lev + 64 * q);
    t.mirrored = true;
    for (int c = 0; c < 256; ++c) if (lev[c] != -lev[255 - c]) t.mirrored = false;
    return t;
}
/* Sixteen 8-bit codes as their int8 levels from the mirror: a code past 127 read as 255 - c (its bits flipped) from the
 * first two tables, and its level negated — two lookups, not four. */
static inline int8x16_t arm_base__int8_levels16_8m(uint8x16_t c, const arm_base__int8_tables8* t) {
    const int8x16_t s = vshrq_n_s8(vreinterpretq_s8_u8(c), 7);                       /* all ones past 127 */
    const uint8x16_t i = veorq_u8(c, vreinterpretq_u8_s8(s));
    const int8x16_t w = vqtbx4q_s8(vqtbl4q_s8(t->t[0], i), t->t[1], vsubq_u8(i, vdupq_n_u8(64u)));
    return vsubq_s8(veorq_s8(w, s), s);                                                /* negated where s */
}
/* Sixteen 8-bit codes as their int8 levels: a code below 64 from the first table, and each later table, its index 64 less,
 * fills the lanes whose code falls in it (`tbx` keeps a lane whose index is past its table). */
static inline int8x16_t arm_base__int8_levels16_8(uint8x16_t c, const arm_base__int8_tables8* t) {
    const uint8x16_t step = vdupq_n_u8(64u);
    int8x16_t w = vqtbl4q_s8(t->t[0], c);
    c = vsubq_u8(c, step); w = vqtbx4q_s8(w, t->t[1], c);
    c = vsubq_u8(c, step); w = vqtbx4q_s8(w, t->t[2], c);
    c = vsubq_u8(c, step); w = vqtbx4q_s8(w, t->t[3], c);
    return w;
}
/* A row of 8-bit codes, a code a byte, against the quantised `x`: nn's sum over its blocks — the levels from the mirror
 * when the table is one, else from all four tables. */
static inline float arm_base__zzprivate_dot_i8_8m(const uint8_t* row, const int8_t* xq, const float* xs, uint64_t cols,
                                                  const arm_base__int8_tables8* t) {
    float part = 0.0f;
    for (uint64_t k = 0; k < cols; k += 32u) {
        const int8x16_t a = arm_base__int8_levels16_8m(vld1q_u8(row + k), t);
        const int8x16_t b = arm_base__int8_levels16_8m(vld1q_u8(row + k + 16u), t);
        const int32x4_t s = vdotq_s32(vdotq_s32(vdupq_n_s32(0), a, vld1q_s8(xq + k)), b, vld1q_s8(xq + k + 16u));
        part += (float)vaddvq_s32(s) * xs[k / 32u];
    }
    return part;
}
static inline float arm_base__zzprivate_dot_i8_8(const uint8_t* row, const int8_t* xq, const float* xs, uint64_t cols,
                                                 const arm_base__int8_tables8* t) {
    if (t->mirrored) return arm_base__zzprivate_dot_i8_8m(row, xq, xs, cols, t);
    float part = 0.0f;
    for (uint64_t k = 0; k < cols; k += 32u) {
        const int8x16_t a = arm_base__int8_levels16_8(vld1q_u8(row + k), t);
        const int8x16_t b = arm_base__int8_levels16_8(vld1q_u8(row + k + 16u), t);
        const int32x4_t s = vdotq_s32(vdotq_s32(vdupq_n_s32(0), a, vld1q_s8(xq + k)), b, vld1q_s8(xq + k + 16u));
        part += (float)vaddvq_s32(s) * xs[k / 32u];
    }
    return part;
}
/* `n` halves of `x` quantised into the call's scratch a block of 32 at a time, as nn's int8 gemv says. */
static inline bool arm_base__zzprivate_quantise_x32(const uint16_t* x, uint64_t n) {
    if (!arm_base__zzprivate_q_room_for(n)) return false;
    for (uint64_t b = 0; b < n / 32u; ++b)
        arm_base__zzprivate_xs[b] = nn__turboquant__zzabi_quantise(x, b * 32u, (uint32_t*)(void*)(arm_base__zzprivate_xq + b * 32u));
    return true;
}
static void arm_base__zzprivate_rows_i8_4(void* a) {
    const arm_base__gemv_i8_call* c = (const arm_base__gemv_i8_call*)a;
    const int8x16_t t8 = arm_base__int8_levels();
    uint64_t r0, r1; arm_base__zzprivate_rows(c->rows, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        const float part = arm_base__zzprivate_dot_i8_4(c->weights + r * (c->cols / 2u), c->xq, c->xs, c->cols, t8);
        c->out[r] = nn__kernels__zzabi_to_half(part * arm_base__zzprivate_row_scale(c->luts, r) / 127.0f, c->over);
    }
}
static void arm_base__zzprivate_rows_i8_8(void* a) {
    const arm_base__gemv_i8_call* c = (const arm_base__gemv_i8_call*)a;
    const arm_base__int8_tables8 t = arm_base__int8_levels8();
    uint64_t r0, r1; arm_base__zzprivate_rows(c->rows, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        const float part = arm_base__zzprivate_dot_i8_8(c->weights + r * c->cols, c->xq, c->xs, c->cols, &t);
        c->out[r] = nn__kernels__zzabi_to_half(part * arm_base__zzprivate_row_scale(c->luts, r) / 127.0f, c->over);
    }
}
static void arm_base__zzprivate_groups_i8(void* a) {
    const arm_base__gemv_i8_call* c = (const arm_base__gemv_i8_call*)a;
    const int8x16_t t8 = arm_base__int8_levels();
    uint64_t t0, t1; arm_base__zzprivate_rows(c->rows, &t0, &t1);
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.rows[i]) { base += c->g.rows[i]; ++i; }
        const uint64_t r = t - base;
        const float dot = arm_base__zzprivate_dot_i8_4((const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u), c->xq, c->xs, c->cols, t8);
        c->out[c->g.out_at[i] + r] = nn__kernels__zzabi_to_half(
            dot * arm_base__zzprivate_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r) / 127.0f, c->over);
    }
}
static void arm_base__zzprivate_groups_sum_i8(void* a) {
    const arm_base__gemv_i8_call* c = (const arm_base__gemv_i8_call*)a;
    const int8x16_t t8 = arm_base__int8_levels();
    uint64_t r0, r1; arm_base__zzprivate_rows(c->rows, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        float part = 0.0f;
        for (uint64_t i = 0u; i < c->g.count; ++i) {
            const float mine = arm_base__zzprivate_dot_i8_4((const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u),
                                                            c->xq + i * c->cols, c->xs + i * (c->cols / 32u), c->cols, t8);
            const float scale = arm_base__zzprivate_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r);
            part += mine * scale / 127.0f * nn__silicon__half_to_float(c->w[c->g.out_at[i]]);
        }
        c->out[r] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(c->residual[r]) + part, c->over);
    }
}
static inline bool arm_base__zzprivate_groups_i8_fit(const nn__turboquant__groups* g, uint64_t cols) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX || cols == 0u || cols % 32u != 0u) return false;
    for (uint64_t i = 0u; i < g->count; ++i) if (g->d[i] != 4u || g->rows[i] == 0u) return false;
    return true;
}
static void arm_base__override_turboquant_gemv_int8(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                    unsigned int* over) {
    const uint64_t row_bytes = nn__turboquant__row_bytes(d, cols);
    const bool fits = (d == 4u || d == 8u) && cols % 32u == 0u && row_bytes != 0u && rows <= w_room / row_bytes
                   && rows <= l_room / 2u;
    if (fits) (void)arm_base__zzpackage_start();
    if (!fits || !arm_base__zzprivate_quantise_x32(x, cols)) {
        nn__turboquant__zzabi_launch_gemv_int8(out, weights, w_room, luts, l_room, x, d, rows, cols, over); return;
    }
    arm_base__gemv_i8_call c = {};
    c.out = out; c.weights = weights; c.luts = luts; c.xq = arm_base__zzprivate_xq; c.xs = arm_base__zzprivate_xs;
    c.rows = rows; c.cols = cols; c.over = over;
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(rows), d == 8u ? arm_base__zzprivate_rows_i8_8 : arm_base__zzprivate_rows_i8_4,
                            &c);
}
static void arm_base__override_gemv_groups_int8(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                unsigned int* over) {
    const bool fits = arm_base__zzprivate_groups_i8_fit(&g, cols);
    if (fits) (void)arm_base__zzpackage_start();
    if (!fits || !arm_base__zzprivate_quantise_x32(x, cols)) { nn__turboquant__zzabi_launch_gemv_groups_int8(out, g, x, cols, over); return; }
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    arm_base__gemv_i8_call c = {};
    c.out = out; c.g = g; c.xq = arm_base__zzprivate_xq; c.xs = arm_base__zzprivate_xs; c.rows = total; c.cols = cols; c.over = over;
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(total), arm_base__zzprivate_groups_i8, &c);
}
static void arm_base__override_gemv_groups_sum_int8(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                    const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    const bool fits = arm_base__zzprivate_groups_i8_fit(&g, cols);
    if (fits) (void)arm_base__zzpackage_start();
    if (!fits || !arm_base__zzprivate_quantise_x32(x, g.count * cols)) {
        nn__turboquant__zzabi_launch_gemv_groups_sum_int8(out, g, x, w, residual, rows, cols, over); return;
    }
    arm_base__gemv_i8_call c = {};
    c.out = out; c.g = g; c.xq = arm_base__zzprivate_xq; c.xs = arm_base__zzprivate_xs; c.w = w; c.residual = residual;
    c.rows = rows; c.cols = cols; c.over = over;
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(rows), arm_base__zzprivate_groups_sum_i8, &c);
}

/* ── THE ROTATIONS — `hadamard_rotate`, `hadamard_ladder_rows`, `hadamard_blocks`: a butterfly a block, in floats ──────
 * nn's definitions (`kernels.cuh`); its generic body sums each output directly over its block, `b` terms an output — on
 * the GB10 37% of the 4B's launch time on the Arm cores alone and 44% of the 35B's experts there (`MEASURED`, the launch
 * profile, measurements/2026-11-09_arm_base_gb10.md ④). Here `x86_avx2`'s bodies in NEON: a pool block takes a block, its
 * halves (by the signs) widened to floats, the butterfly's stages `x + y`, `x − y` in floats — the first two a value at a
 * time, the rest four lanes at a time, the same arithmetic — and each output rounded to a half once; the order of the
 * float sums is the butterfly's, not the generic body's, so within the bench's bounds of it. */
static inline void arm_base__zzprivate_butterfly(float* v, uint32_t B) {
    uint32_t h = 1u;
    for (; h < 4u && h < B; h <<= 1)
        for (uint32_t i = 0u; i < B; i += 2u * h)
            for (uint32_t j = i; j < i + h; ++j) { const float x = v[j], y = v[j + h]; v[j] = x + y; v[j + h] = x - y; }
    for (; h < B; h <<= 1)
        for (uint32_t i = 0u; i < B; i += 2u * h)
            for (uint32_t j = i; j < i + h; j += 4u) {
                const float32x4_t x = vld1q_f32(v + j), y = vld1q_f32(v + j + h);
                vst1q_f32(v + j, vaddq_f32(x, y)); vst1q_f32(v + j + h, vsubq_f32(x, y));
            }
}
typedef struct arm_base__rotate_call {
    uint16_t* out; const uint16_t* in; const uint16_t* sign; unsigned int* over;
    uint64_t whole, n;             /* the whole 512-blocks' halves, and every half */
} arm_base__rotate_call;
/* Pool block `b`'s place and size: the whole 512-blocks first, then nn's tail — a 256-block where 256 or more are left,
 * then a 128-block (▶ nn's `nn__hadamard__zzabi_body_rotate`). */
static inline void arm_base__zzprivate_rotate_block(const arm_base__rotate_call* c, uint64_t b, uint64_t* at, uint32_t* size) {
    const uint64_t whole_blocks = c->whole / 512u, rest = c->n - c->whole;
    if (b < whole_blocks) { *at = b * 512u; *size = 512u; return; }
    if (b == whole_blocks && rest >= 256u) { *at = c->whole; *size = 256u; return; }
    *at = c->whole + (rest >= 256u ? 256u : 0u); *size = 128u;
}
/* Block `b` of row `row` (rows of `n`, one after another): its butterfly in floats, rounded once. */
static inline void arm_base__zzprivate_rotate_one(const arm_base__rotate_call* c, uint64_t b, uint64_t row) {
    uint64_t at = 0u; uint32_t B = 512u;
    arm_base__zzprivate_rotate_block(c, b, &at, &B);
    at += row * c->n;
    float v[512] __attribute__((aligned(16)));
    for (uint32_t i = 0u; i < B; ++i) v[i] = nn__silicon__half_to_float(c->in[at + i]) * nn__silicon__half_to_float(c->sign[i]);
    arm_base__zzprivate_butterfly(v, B);
    const float inv = 1.0f / sqrtf((float)B);
    for (uint32_t i = 0u; i < B; ++i) c->out[at + i] = nn__kernels__zzabi_to_half(v[i] * inv, c->over);
}
static void arm_base__zzprivate_rotate(void* a) {
    arm_base__zzprivate_rotate_one((const arm_base__rotate_call*)a, (uint64_t)nn__silicon__block(), 0u);
}
/* The pool blocks one row of `n` takes: its whole 512-blocks, then nn's 256 and 128 tail. */
static inline uint64_t arm_base__zzprivate_rotate_blocks(uint64_t n) {
    const uint64_t rest = n % 512u;
    return n / 512u + (rest >= 256u ? 1u : 0u) + (rest % 256u == 128u ? 1u : 0u);
}
/* Whole 512-blocks and nn's 256 and 128 tail, `out` apart from `in`; anything else is the generic door. */
static void arm_base__override_hadamard_rotate(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n, unsigned int* over) {
    const bool apart = out + n <= in || in + n <= out;
    if (n < 512u || n % 128u != 0u || !apart) { nn__hadamard__zzabi_launch_rotate(out, in, sign, n, over); return; }
    (void)arm_base__zzpackage_start();
    arm_base__rotate_call c = { out, in, sign, over, n / 512u * 512u, n };
    arm_base__zzpackage_run((uint32_t)arm_base__zzprivate_rotate_blocks(n), arm_base__zzprivate_rotate, &c);
}
/* Rows of `cols`, each in nn's ladder, every row's blocks one launch — a row the bytes `hadamard_rotate` gives it alone.
 * Rows of 128 or wider and whole 128s, `out` apart from `in`; anything else is the generic door. */
typedef struct arm_base__ladder_call {
    arm_base__rotate_call row;     /* one row's: `n` its width */
    uint64_t per_row;              /* the pool blocks a row takes */
} arm_base__ladder_call;
static void arm_base__zzprivate_ladder(void* a) {
    const arm_base__ladder_call* c = (const arm_base__ladder_call*)a;
    const uint64_t b = (uint64_t)nn__silicon__block();
    arm_base__zzprivate_rotate_one(&c->row, b % c->per_row, b / c->per_row);
}
static void arm_base__override_hadamard_ladder_rows(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t cols,
                                                   uint64_t rows, unsigned int* over) {
    const uint64_t n = cols * rows;
    const bool apart = out + n <= in || in + n <= out;
    const uint64_t per_row = cols >= 128u && cols % 128u == 0u ? arm_base__zzprivate_rotate_blocks(cols) : 0u;
    if (per_row == 0u || rows == 0u || !apart || per_row * rows > 0xFFFFFFFFull) {
        nn__hadamard__zzabi_launch_ladder_rows(out, in, sign, cols, rows, over);
        return;
    }
    (void)arm_base__zzpackage_start();
    arm_base__ladder_call c = { { out, in, sign, over, cols / 512u * 512u, cols }, per_row };
    arm_base__zzpackage_run((uint32_t)(per_row * rows), arm_base__zzprivate_ladder, &c);
}
/* Blocks of `block` (a power of two, at most 512), each `H·(S ⊙ in) / √block`, or with `inverse` `S ⊙ H·in / √block` — a
 * pool block whole Hadamard blocks, read whole before written, so in place too. */
typedef struct arm_base__hblocks_call {
    uint16_t* out; const uint16_t* in; const uint16_t* sign; uint64_t n, block, inverse; unsigned int* over;
} arm_base__hblocks_call;
static void arm_base__zzprivate_hblocks(void* a) {
    const arm_base__hblocks_call* c = (const arm_base__hblocks_call*)a;
    uint64_t q0, q1; arm_base__zzprivate_rows(c->n / c->block, &q0, &q1);
    const uint32_t B = (uint32_t)c->block;
    const float norm = 1.0f / sqrtf((float)B);
    float v[512] __attribute__((aligned(16)));
    for (uint64_t q = q0; q < q1; ++q) {
        const uint64_t at = q * B;
        for (uint32_t e = 0u; e < B; ++e)
            v[e] = nn__silicon__half_to_float(c->in[at + e]) * (c->inverse ? 1.0f : nn__silicon__half_to_float(c->sign[e]));
        arm_base__zzprivate_butterfly(v, B);
        for (uint32_t e = 0u; e < B; ++e)
            c->out[at + e] = nn__kernels__zzabi_to_half(v[e] * norm * (c->inverse ? nn__silicon__half_to_float(c->sign[e]) : 1.0f), c->over);
    }
}
static void arm_base__override_hadamard_blocks(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n, uint64_t block,
                                               uint64_t inverse, unsigned int* over) {
    const bool apart = out + n <= in || in + n <= out || out == in;
    if (block == 0u || block > 512u || (block & (block - 1u)) != 0u || n == 0u || n % block != 0u || !apart) {
        nn__hadamard__zzabi_launch_blocks(out, in, sign, n, block, inverse, over);
        return;
    }
    (void)arm_base__zzpackage_start();
    arm_base__hblocks_call c = { out, in, sign, n, block, inverse, over };
    arm_base__zzpackage_run(arm_base__zzprivate_blocks_for(n / block), arm_base__zzprivate_hblocks, &c);
}

/* ── THE HYPER-CONNECTIONS' LOGITS IN DOUBLE — `hyper_logits_fp32`, DeepSeek's mixing a block ─────────────────────────
 * nn's definition (`kernels.cuh`): each row's logit `dot(fn_row, streams) / sqrt(mean(streams²) + eps)` over the streams'
 * `mult · hidden` floats. nn's body sums in floats, ~5e-5 of the largest off the sums in double at 20,480 (`MEASURED` on
 * the GB10, the door bench: its one failure through the generic body); `x86_avx2`'s sums in double. Here as there: a pool
 * block a row, its two sums — the squares and the dot — doubles over two lanes and four sums each (a half's float and a
 * stream's float multiply exactly in one), folded in one order, the logit rounded to a float once. */
typedef struct arm_base__hyper_logits_call {
    float* lg; const float* streams; const uint16_t* fn; uint64_t n; double eps;
} arm_base__hyper_logits_call;
static void arm_base__zzprivate_hyper_logits_fp32(void* a) {
    const arm_base__hyper_logits_call* c = (const arm_base__hyper_logits_call*)a;
    const uint64_t r = nn__silicon__block(), n = c->n;
    const float* s = c->streams;
    const uint16_t* f = c->fn + r * n;
    float64x2_t q[4], d[4];
    for (int k = 0; k < 4; ++k) { q[k] = vdupq_n_f64(0.0); d[k] = vdupq_n_f64(0.0); }
    uint64_t i = 0u;
    for (; i + 8u <= n; i += 8u) {
        const float32x4_t s0 = vld1q_f32(s + i), s1 = vld1q_f32(s + i + 4u);
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(f + i));
        const float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h)), f1 = vcvt_high_f32_f16(h);
        const float64x2_t sv[4] = { vcvt_f64_f32(vget_low_f32(s0)), vcvt_high_f64_f32(s0), vcvt_f64_f32(vget_low_f32(s1)), vcvt_high_f64_f32(s1) };
        const float64x2_t fv[4] = { vcvt_f64_f32(vget_low_f32(f0)), vcvt_high_f64_f32(f0), vcvt_f64_f32(vget_low_f32(f1)), vcvt_high_f64_f32(f1) };
        for (int k = 0; k < 4; ++k) { q[k] = vfmaq_f64(q[k], sv[k], sv[k]); d[k] = vfmaq_f64(d[k], fv[k], sv[k]); }
    }
    const float64x2_t qs = vaddq_f64(vaddq_f64(q[0], q[1]), vaddq_f64(q[2], q[3]));
    const float64x2_t ds = vaddq_f64(vaddq_f64(d[0], d[1]), vaddq_f64(d[2], d[3]));
    double sq = vgetq_lane_f64(qs, 0) + vgetq_lane_f64(qs, 1), dot = vgetq_lane_f64(ds, 0) + vgetq_lane_f64(ds, 1);
    for (; i < n; ++i) {
        const double v = s[i];
        sq += v * v;
        dot += (double)nn__silicon__half_to_float(f[i]) * v;
    }
    c->lg[r] = (float)(dot / sqrt(sq / (double)n + c->eps));
}
static void arm_base__override_hyper_logits_fp32(float* lg, const float* streams, const uint16_t* fn, uint64_t hidden, uint64_t mult,
                                                 float norm_eps) {
    const uint64_t rows = NN__HYPER__MIX(mult);
    if (hidden == 0u || mult == 0u || mult > NN__HYPER__MULT_MAX) {
        nn__hyper__zzabi_launch_logits_fp32(lg, streams, fn, hidden, mult, norm_eps); return;
    }
    (void)arm_base__zzpackage_start();
    arm_base__hyper_logits_call c = { lg, streams, fn, mult * hidden, (double)norm_eps };
    arm_base__zzpackage_run((uint32_t)rows, arm_base__zzprivate_hyper_logits_fp32, &c);
}

/* Every door above into nn's table — in place of its generic body. */
static inline void arm_base__override_doors(nn__doors* doors) {
    doors->turboquant_gemv            = arm_base__override_turboquant_gemv;
    doors->turboquant_gemv_groups     = arm_base__override_gemv_groups;
    doors->turboquant_gemv_groups_n   = arm_base__override_gemv_groups_n;
    doors->turboquant_gemv_groups_sum = arm_base__override_gemv_groups_sum;
    doors->expert_rows                = arm_base__override_expert_rows;
    doors->expert_rows_sum            = arm_base__override_expert_rows_sum;
    doors->expert_groups              = arm_base__override_expert_groups;
    doors->hyper_logits_fp32          = arm_base__override_hyper_logits_fp32;
    doors->hadamard_rotate            = arm_base__override_hadamard_rotate;
    doors->hadamard_ladder_rows       = arm_base__override_hadamard_ladder_rows;
    doors->hadamard_blocks            = arm_base__override_hadamard_blocks;
    doors->turboquant_gemv_int8           = arm_base__override_turboquant_gemv_int8;
    doors->turboquant_gemv_groups_int8     = arm_base__override_gemv_groups_int8;
    doors->turboquant_gemv_groups_sum_int8 = arm_base__override_gemv_groups_sum_int8;
    doors->expert_groups_int8         = arm_base__override_expert_groups_int8;
    doors->expert_rows_sum_int8       = arm_base__override_expert_rows_sum_int8;
}

#endif /* SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_NN_CUH */
