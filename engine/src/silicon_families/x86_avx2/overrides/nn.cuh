#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_OVERRIDES_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include "../primitives/pool.cuh"   /* the threads the rows are spread over */

/* ══ ⭐⭐ OVERRIDES — THE DOORS THIS FAMILY RUNS FASTER THAN THE GENERIC BODY ══════════════════════════
 * ⚖ *"a for the family bodies"*, in *"a folder for primitives and one for overrides so it is clear what is
 * expected to run and what are performance upgrades"*. Each function here has its door's signature and
 * takes its place in nn's table (`x86_avx2__override_doors`, called from `../entry.cuh` once the table is
 * built). ⛳ EVERY ONE FALLS BACK TO nn's GENERIC DOOR for what it does not cover — another width, a row
 * that is not aligned — so an override can only ever be faster, never different in what it accepts, and
 * deleting this file loses nothing but speed.
 * The kernels are the ones measured in `backstage/scripts/tq_cpu_probe.c`
 * (`measurements/2026-09-27_avx2_expert_gemv.md`): at 4 bits the levels through `pshufb` and F16C, at 8
 * bits gathered; and the int8 route's levels through one `pshufb` and `pmaddubsw`. */

/* ── the rows of one call, spread over the pool: each block a contiguous run of rows ───────────────── */
typedef struct x86_avx2__gemv_call {
    uint16_t* out; const uint8_t* weights; const uint8_t* luts; uint64_t row_bytes, rows, cols, d;
    const float* xf;                 /* x widened to floats, once a call (the exact route) */
    const int8_t* xq; const float* xs;   /* x quantised in blocks of 32, once a call (the int8 route) */
    unsigned int* over;
} x86_avx2__gemv_call;

static inline float x86_avx2__zzprivate_hsum(__m256 a) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(r);
}
static inline float x86_avx2__zzprivate_row_scale(const uint8_t* luts, uint64_t r) {
    return _cvtsh_ss((uint16_t)((uint32_t)luts[r * 2u] | ((uint32_t)luts[r * 2u + 1u] << 8)));
}
static inline void x86_avx2__zzprivate_write(const x86_avx2__gemv_call* c, uint64_t r, float sum) {
    c->out[r] = nn__kernels__zzabi_to_half(sum, c->over);
}
/* The block's rows: an even share of them, by the pool's block number. */
static inline void x86_avx2__zzprivate_rows(const x86_avx2__gemv_call* c, uint64_t* r0, uint64_t* r1) {
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    *r0 = c->rows * b / n; *r1 = c->rows * (b + 1u) / n;
}

/* ── the exact route ──────────────────────────────────────────────────────────────────────────────── */
static void x86_avx2__zzprivate_exact4(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    const uint16_t* lev = nn__turboquant__zzabi_levels(4u);
    uint8_t lo_b[16], hi_b[16];
    for (int i = 0; i < 16; ++i) { lo_b[i] = (uint8_t)lev[i]; hi_b[i] = (uint8_t)(lev[i] >> 8); }
    const __m256i tlo = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)lo_b));
    const __m256i thi = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)hi_b));
    const __m128i fifteen = _mm_set1_epi8(15);
    uint64_t r0, r1; x86_avx2__zzprivate_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        const uint8_t* row = c->weights + r * c->row_bytes;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (uint64_t k = 0; k < c->cols; k += 32u) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
            const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
            const __m256i idx = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
            const __m256i bl = _mm256_shuffle_epi8(tlo, idx), bh = _mm256_shuffle_epi8(thi, idx);
            const __m256i h0 = _mm256_unpacklo_epi8(bl, bh), h1 = _mm256_unpackhi_epi8(bl, bh);
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h0)),      _mm256_loadu_ps(c->xf + k),       a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h1)),      _mm256_loadu_ps(c->xf + k + 8u),  a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h0, 1)), _mm256_loadu_ps(c->xf + k + 16u), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h1, 1)), _mm256_loadu_ps(c->xf + k + 24u), a3);
        }
        const float sum = x86_avx2__zzprivate_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        x86_avx2__zzprivate_write(c, r, sum * x86_avx2__zzprivate_row_scale(c->luts, r));
    }
}
static float x86_avx2__zzprivate_levels8[256];
static void x86_avx2__zzprivate_exact8(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    uint64_t r0, r1; x86_avx2__zzprivate_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        const uint8_t* row = c->weights + r * c->row_bytes;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        for (uint64_t k = 0; k < c->cols; k += 16u) {
            const __m256i i0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(row + k)));
            const __m256i i1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(row + k + 8u)));
            a0 = _mm256_fmadd_ps(_mm256_i32gather_ps(x86_avx2__zzprivate_levels8, i0, 4), _mm256_loadu_ps(c->xf + k), a0);
            a1 = _mm256_fmadd_ps(_mm256_i32gather_ps(x86_avx2__zzprivate_levels8, i1, 4), _mm256_loadu_ps(c->xf + k + 8u), a1);
        }
        x86_avx2__zzprivate_write(c, r, x86_avx2__zzprivate_hsum(_mm256_add_ps(a0, a1)) * x86_avx2__zzprivate_row_scale(c->luts, r));
    }
}

/* ── the int8 route: nn's definition (`kernels.cuh`), x quantised once a call ─────────────────────── */
static void x86_avx2__zzprivate_int8_4(void* a) {
    const x86_avx2__gemv_call* c = (const x86_avx2__gemv_call*)a;
    const int8_t* lev = nn__turboquant__zzabi_levels_i8(4u);
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)lev));
    const __m128i fifteen = _mm_set1_epi8(15);
    const __m256i ones = _mm256_set1_epi16(1);
    uint64_t r0, r1; x86_avx2__zzprivate_rows(c, &r0, &r1);
    for (uint64_t r = r0; r < r1; ++r) {
        const uint8_t* row = c->weights + r * c->row_bytes;
        __m256 acc = _mm256_setzero_ps();
        for (uint64_t k = 0; k < c->cols; k += 32u) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
            const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
            const __m256i w = _mm256_shuffle_epi8(t8, _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)));
            const __m256i xq = _mm256_loadu_si256((const __m256i*)(c->xq + k));
            const __m256i p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(xq, w));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones)), _mm256_set1_ps(c->xs[k / 32u]), acc);
        }
        x86_avx2__zzprivate_write(c, r, x86_avx2__zzprivate_hsum(acc) * x86_avx2__zzprivate_row_scale(c->luts, r) / 127.0f);
    }
}

/* ── the call's own scratch: x widened or quantised, grown as needed, one a calling thread ─────────── */
static __thread float*  x86_avx2__zzprivate_xf = 0;
static __thread int8_t* x86_avx2__zzprivate_xq = 0;
static __thread float*  x86_avx2__zzprivate_xs = 0;
static __thread uint64_t x86_avx2__zzprivate_x_room = 0;
static inline bool x86_avx2__zzprivate_x_room_for(uint64_t cols) {
    if (cols <= x86_avx2__zzprivate_x_room) return true;
    free(x86_avx2__zzprivate_xf); free(x86_avx2__zzprivate_xq); free(x86_avx2__zzprivate_xs);
    x86_avx2__zzprivate_xf = (float*)aligned_alloc(64, ((cols * 4u + 63u) / 64u) * 64u);
    x86_avx2__zzprivate_xq = (int8_t*)aligned_alloc(64, ((cols + 63u) / 64u) * 64u);
    x86_avx2__zzprivate_xs = (float*)aligned_alloc(64, ((cols / 32u * 4u + 64u) / 64u) * 64u);
    x86_avx2__zzprivate_x_room = (x86_avx2__zzprivate_xf && x86_avx2__zzprivate_xq && x86_avx2__zzprivate_xs) ? cols : 0u;
    return x86_avx2__zzprivate_x_room != 0u;
}
/* How many blocks to cut a call's rows into: a few a thread, so a slow one does not hold the rest. */
static inline uint32_t x86_avx2__zzprivate_blocks_for(uint64_t rows) {
    const uint64_t want = 4ull * x86_avx2__zzpackage_threads();
    return (uint32_t)(rows < want ? rows : want);
}

/* ⭐ THE EXACT GEMV: 4 and 8 bits with whole 32-column runs and a row a multiple of 16 bytes; anything else
 *   is the generic door. */
static void x86_avx2__override_turboquant_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                               uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                               unsigned int* over) {
    const uint64_t row_bytes = nn__turboquant__row_bytes(d, cols);
    const bool fits = (d == 4u || d == 8u) && cols % 32u == 0u && row_bytes % 16u == 0u
                   && row_bytes != 0u && rows <= w_room / row_bytes && rows <= l_room / 2u
                   && x86_avx2__zzprivate_x_room_for(cols);
    if (!fits) { nn__turboquant__zzabi_launch_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over); return; }
    x86_avx2__zzpackage_start();
    for (uint64_t k = 0; k < cols; ++k) x86_avx2__zzprivate_xf[k] = _cvtsh_ss(x[k]);
    x86_avx2__gemv_call c = { out, weights, luts, row_bytes, rows, cols, d, x86_avx2__zzprivate_xf, 0, 0, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(rows), d == 4u ? x86_avx2__zzprivate_exact4 : x86_avx2__zzprivate_exact8, &c);
}

/* ⭐ THE int8 GEMV: 4 bits (the levels through one `pshufb`); anything else is the generic door. x is
 *   quantised here exactly as nn's definition says — blocks of 32, max|x|/127, half away from zero. */
static void x86_avx2__override_turboquant_gemv_int8(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                    unsigned int* over) {
    const uint64_t row_bytes = nn__turboquant__row_bytes(d, cols);
    const bool fits = d == 4u && cols % 32u == 0u && row_bytes % 16u == 0u
                   && row_bytes != 0u && rows <= w_room / row_bytes && rows <= l_room / 2u
                   && x86_avx2__zzprivate_x_room_for(cols);
    if (!fits) { nn__turboquant__zzabi_launch_gemv_int8(out, weights, w_room, luts, l_room, x, d, rows, cols, over); return; }
    x86_avx2__zzpackage_start();
    for (uint64_t b = 0; b < cols / 32u; ++b) {
        uint32_t q[8];
        x86_avx2__zzprivate_xs[b] = nn__turboquant__zzabi_quantise(x, b * 32u, q);
        memcpy(x86_avx2__zzprivate_xq + b * 32u, q, 32u);
    }
    x86_avx2__gemv_call c = { out, weights, luts, row_bytes, rows, cols, d, 0, x86_avx2__zzprivate_xq, x86_avx2__zzprivate_xs, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(rows), x86_avx2__zzprivate_int8_4, &c);
}

/* ══ ⭐⭐ THE LAYER'S OTHER DOORS — the grouped gemvs, the rotation, the DeltaNet step ═════════════════════════
 * ⚖ *"lets also do the avx2 bodies for the CPU family"*. `MEASURED` before them: a 35B token on this
 * family 333 ms, against ~22 ms for the bytes it reads — these four were nn's generic bodies. */

/* One row's product over `cols`, at 4 and at 8 bits, with `xf` the activation widened to floats. */
static inline float x86_avx2__zzprivate_dot4(const uint8_t* row, const float* xf, uint64_t cols, __m256i tlo, __m256i thi) {
    const __m128i fifteen = _mm_set1_epi8(15);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (uint64_t k = 0; k < cols; k += 32u) {
        const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
        const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
        const __m256i idx = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
        const __m256i bl = _mm256_shuffle_epi8(tlo, idx), bh = _mm256_shuffle_epi8(thi, idx);
        const __m256i h0 = _mm256_unpacklo_epi8(bl, bh), h1 = _mm256_unpackhi_epi8(bl, bh);
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h0)),      _mm256_loadu_ps(xf + k),       a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_castsi256_si128(h1)),      _mm256_loadu_ps(xf + k + 8u),  a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h0, 1)), _mm256_loadu_ps(xf + k + 16u), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm256_extracti128_si256(h1, 1)), _mm256_loadu_ps(xf + k + 24u), a3);
    }
    return x86_avx2__zzprivate_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}
static inline float x86_avx2__zzprivate_dot8(const uint8_t* row, const float* xf, uint64_t cols) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (uint64_t k = 0; k < cols; k += 16u) {
        const __m256i i0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(row + k)));
        const __m256i i1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(row + k + 8u)));
        a0 = _mm256_fmadd_ps(_mm256_i32gather_ps(x86_avx2__zzprivate_levels8, i0, 4), _mm256_loadu_ps(xf + k), a0);
        a1 = _mm256_fmadd_ps(_mm256_i32gather_ps(x86_avx2__zzprivate_levels8, i1, 4), _mm256_loadu_ps(xf + k + 8u), a1);
    }
    return x86_avx2__zzprivate_hsum(_mm256_add_ps(a0, a1));
}
static inline void x86_avx2__zzprivate_tables4(__m256i* tlo, __m256i* thi) {
    const uint16_t* lev = nn__turboquant__zzabi_levels(4u);
    uint8_t lo_b[16], hi_b[16];
    for (int i = 0; i < 16; ++i) { lo_b[i] = (uint8_t)lev[i]; hi_b[i] = (uint8_t)(lev[i] >> 8); }
    *tlo = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)lo_b));
    *thi = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)hi_b));
}

typedef struct x86_avx2__groups_call {
    uint16_t* out; nn__turboquant__groups g; const float* xf; const uint16_t* w; const uint16_t* residual;
    uint64_t rows, cols; unsigned int* over;
} x86_avx2__groups_call;

static inline float x86_avx2__zzprivate_group_row(const nn__turboquant__groups* g, uint64_t i, uint64_t r, const float* xf,
                                                  uint64_t cols, __m256i tlo, __m256i thi) {
    const uint8_t* row = (const uint8_t*)(uintptr_t)g->codes[i] + r * (cols * g->d[i] / 8u);
    const float dot = g->d[i] == 4u ? x86_avx2__zzprivate_dot4(row, xf, cols, tlo, thi) : x86_avx2__zzprivate_dot8(row, xf, cols);
    return dot * x86_avx2__zzprivate_row_scale((const uint8_t*)(uintptr_t)g->luts[i], r);
}

/* the gate_ups: every group's rows, one `x`, the flat row range of this block */
static void x86_avx2__zzprivate_groups(void* a) {
    const x86_avx2__groups_call* c = (const x86_avx2__groups_call*)a;
    __m256i tlo, thi; x86_avx2__zzprivate_tables4(&tlo, &thi);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->rows * b / n, t1 = c->rows * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.rows[i]) { base += c->g.rows[i]; ++i; }
        const uint64_t r = t - base;
        c->out[c->g.out_at[i] + r] = nn__kernels__zzabi_to_half(x86_avx2__zzprivate_group_row(&c->g, i, r, c->xf, c->cols, tlo, thi), c->over);
    }
}
/* the downs: every group's row `r` weighted onto the residual */
static void x86_avx2__zzprivate_groups_sum(void* a) {
    const x86_avx2__groups_call* c = (const x86_avx2__groups_call*)a;
    __m256i tlo, thi; x86_avx2__zzprivate_tables4(&tlo, &thi);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t r0 = c->rows * b / n, r1 = c->rows * (b + 1u) / n;
    for (uint64_t r = r0; r < r1; ++r) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < c->g.count; ++i)
            sum += x86_avx2__zzprivate_group_row(&c->g, i, r, c->xf + i * c->cols, c->cols, tlo, thi) * _cvtsh_ss(c->w[c->g.out_at[i]]);
        c->out[r] = nn__kernels__zzabi_to_half(_cvtsh_ss(c->residual[r]) + sum, c->over);
    }
}
static inline bool x86_avx2__zzprivate_groups_fit(const nn__turboquant__groups* g, uint64_t cols) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX || cols == 0u || cols % 32u != 0u) return false;
    for (uint64_t i = 0u; i < g->count; ++i) if ((g->d[i] != 4u && g->d[i] != 8u) || g->rows[i] == 0u) return false;
    return true;
}
static void x86_avx2__override_gemv_groups(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, unsigned int* over) {
    if (!x86_avx2__zzprivate_groups_fit(&g, cols) || !x86_avx2__zzprivate_x_room_for(cols)) {
        nn__turboquant__zzabi_launch_gemv_groups(out, g, x, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    for (uint64_t k = 0; k < cols; ++k) x86_avx2__zzprivate_xf[k] = _cvtsh_ss(x[k]);
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    x86_avx2__groups_call c = { out, g, x86_avx2__zzprivate_xf, 0, 0, total, cols, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(total), x86_avx2__zzprivate_groups, &c);
}
static void x86_avx2__override_gemv_groups_sum(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                               const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (!x86_avx2__zzprivate_groups_fit(&g, cols) || !x86_avx2__zzprivate_x_room_for(g.count * cols)) {
        nn__turboquant__zzabi_launch_gemv_groups_sum(out, g, x, w, residual, rows, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    for (uint64_t k = 0; k < g.count * cols; ++k) x86_avx2__zzprivate_xf[k] = _cvtsh_ss(x[k]);
    x86_avx2__groups_call c = { out, g, x86_avx2__zzprivate_xf, w, residual, rows, cols, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(rows), x86_avx2__zzprivate_groups_sum, &c);
}

/* ── THE GROUPED GEMVS IN INTEGERS — nn's int8 definition, as `turboquant_gemv_int8`'s override has it: `x` quantised once
 *   a call (every group's own `x` for the weighted sum), the 4-bit levels through one `pshufb`, int8 × int8 in `maddubs`. */
static inline float x86_avx2__zzprivate_dot_i8_4(const uint8_t* row, const int8_t* xq, const float* xs, uint64_t cols, __m256i t8) {
    const __m128i fifteen = _mm_set1_epi8(15);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (uint64_t k = 0; k < cols; k += 32u) {
        const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
        const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
        const __m256i w = _mm256_shuffle_epi8(t8, _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)));
        const __m256i q = _mm256_loadu_si256((const __m256i*)(xq + k));
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(q, w));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones)), _mm256_set1_ps(xs[k / 32u]), acc);
    }
    return x86_avx2__zzprivate_hsum(acc);
}
typedef struct x86_avx2__groups_i8_call {
    uint16_t* out; nn__turboquant__groups g; const int8_t* xq; const float* xs; const uint16_t* w; const uint16_t* residual;
    uint64_t rows, cols; unsigned int* over;
} x86_avx2__groups_i8_call;
static void x86_avx2__zzprivate_groups_i8(void* a) {
    const x86_avx2__groups_i8_call* c = (const x86_avx2__groups_i8_call*)a;
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)nn__turboquant__zzabi_levels_i8(4u)));
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->rows * b / n, t1 = c->rows * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.rows[i]) { base += c->g.rows[i]; ++i; }
        const uint64_t r = t - base;
        const uint8_t* row = (const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u);
        const float dot = x86_avx2__zzprivate_dot_i8_4(row, c->xq, c->xs, c->cols, t8);
        c->out[c->g.out_at[i] + r] = nn__kernels__zzabi_to_half(
            dot * x86_avx2__zzprivate_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r) / 127.0f, c->over);
    }
}
static void x86_avx2__zzprivate_groups_sum_i8(void* a) {
    const x86_avx2__groups_i8_call* c = (const x86_avx2__groups_i8_call*)a;
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)nn__turboquant__zzabi_levels_i8(4u)));
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t r0 = c->rows * b / n, r1 = c->rows * (b + 1u) / n;
    for (uint64_t r = r0; r < r1; ++r) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < c->g.count; ++i) {
            const uint8_t* row = (const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u);
            const float dot = x86_avx2__zzprivate_dot_i8_4(row, c->xq + i * c->cols, c->xs + i * (c->cols / 32u), c->cols, t8);
            sum += dot * x86_avx2__zzprivate_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r) / 127.0f * _cvtsh_ss(c->w[c->g.out_at[i]]);
        }
        c->out[r] = nn__kernels__zzabi_to_half(_cvtsh_ss(c->residual[r]) + sum, c->over);
    }
}
static inline bool x86_avx2__zzprivate_groups_i8_fit(const nn__turboquant__groups* g, uint64_t cols) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX || cols == 0u || cols % 32u != 0u) return false;
    for (uint64_t i = 0u; i < g->count; ++i) if (g->d[i] != 4u || g->rows[i] == 0u) return false;
    return true;
}
static inline void x86_avx2__zzprivate_quantise_x(const uint16_t* x, uint64_t cols) {
    for (uint64_t b = 0; b < cols / 32u; ++b) {
        uint32_t q[8];
        x86_avx2__zzprivate_xs[b] = nn__turboquant__zzabi_quantise(x, b * 32u, q);
        memcpy(x86_avx2__zzprivate_xq + b * 32u, q, 32u);
    }
}
static void x86_avx2__override_gemv_groups_int8(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                unsigned int* over) {
    if (!x86_avx2__zzprivate_groups_i8_fit(&g, cols) || !x86_avx2__zzprivate_x_room_for(cols)) {
        nn__turboquant__zzabi_launch_gemv_groups_int8(out, g, x, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_quantise_x(x, cols);
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    x86_avx2__groups_i8_call c = { out, g, x86_avx2__zzprivate_xq, x86_avx2__zzprivate_xs, 0, 0, total, cols, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(total), x86_avx2__zzprivate_groups_i8, &c);
}
static void x86_avx2__override_gemv_groups_sum_int8(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                    const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (!x86_avx2__zzprivate_groups_i8_fit(&g, cols) || !x86_avx2__zzprivate_x_room_for(g.count * cols)) {
        nn__turboquant__zzabi_launch_gemv_groups_sum_int8(out, g, x, w, residual, rows, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_quantise_x(x, g.count * cols);
    x86_avx2__groups_i8_call c = { out, g, x86_avx2__zzprivate_xq, x86_avx2__zzprivate_xs, w, residual, rows, cols, over };
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(rows), x86_avx2__zzprivate_groups_sum_i8, &c);
}

/* ── THE ROTATION: an fp32 butterfly a 512-block, a block a pool block, rounded to a half once ───────────── */
typedef struct x86_avx2__rotate_call { uint16_t* out; const uint16_t* in; const uint16_t* sign; unsigned int* over; } x86_avx2__rotate_call;
static void x86_avx2__zzprivate_rotate(void* a) {
    const x86_avx2__rotate_call* c = (const x86_avx2__rotate_call*)a;
    const uint64_t at = (uint64_t)nn__silicon__block() * 512u;
    float v[512] __attribute__((aligned(32)));
    for (uint32_t i = 0u; i < 512u; ++i) v[i] = _cvtsh_ss(c->in[at + i]) * _cvtsh_ss(c->sign[i]);
    for (uint32_t h = 1u; h < 8u; h <<= 1)
        for (uint32_t i = 0u; i < 512u; i += 2u * h)
            for (uint32_t j = i; j < i + h; ++j) { const float x = v[j], y = v[j + h]; v[j] = x + y; v[j + h] = x - y; }
    for (uint32_t h = 8u; h < 512u; h <<= 1)
        for (uint32_t i = 0u; i < 512u; i += 2u * h)
            for (uint32_t j = i; j < i + h; j += 8u) {
                const __m256 x = _mm256_load_ps(v + j), y = _mm256_load_ps(v + j + h);
                _mm256_store_ps(v + j, _mm256_add_ps(x, y)); _mm256_store_ps(v + j + h, _mm256_sub_ps(x, y));
            }
    const float inv = 1.0f / sqrtf(512.0f);
    for (uint32_t i = 0u; i < 512u; ++i) c->out[at + i] = nn__kernels__zzabi_to_half(v[i] * inv, c->over);
}
static void x86_avx2__override_hadamard_rotate(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n, unsigned int* over) {
    const bool apart = out + n <= in || in + n <= out;
    if (n == 0u || n % 512u != 0u || !apart) { nn__hadamard__zzabi_launch_rotate(out, in, sign, n, over); return; }
    x86_avx2__rotate_call c = { out, in, sign, over };
    x86_avx2__zzpackage_run((uint32_t)(n / 512u), x86_avx2__zzprivate_rotate, &c);
}

/* ── THE DELTANET STEP: a head a pool block, eight of its state's columns at a time ──────────────────────── */
typedef struct x86_avx2__step_call {
    float* S; const uint16_t* conved; const uint16_t* z; const uint16_t* beta; const uint16_t* g; const uint16_t* w;
    uint16_t* out; uint64_t k_heads, v_heads, d; unsigned int* over;
} x86_avx2__step_call;
static void x86_avx2__zzprivate_step(void* a) {
    const x86_avx2__step_call* c = (const x86_avx2__step_call*)a;
    const uint64_t h = nn__silicon__block(), d = c->d, rep = c->v_heads / c->k_heads;
    const uint16_t* q = c->conved + (h / rep) * d;
    const uint16_t* k = c->conved + c->k_heads * d + (h / rep) * d;
    const uint16_t* v = c->conved + 2ull * c->k_heads * d + h * d;
    float* Sh = c->S + h * d * d;
    uint16_t* oh = c->out + h * d;
    float qf[256], kf[256];
    float qq = 0.0f, kk = 0.0f;
    for (uint64_t i = 0u; i < d; ++i) { qf[i] = _cvtsh_ss(q[i]); kf[i] = _cvtsh_ss(k[i]); qq += qf[i] * qf[i]; kk += kf[i] * kf[i]; }
    const float inv_q = 1.0f / sqrtf(qq + 1e-6f) / sqrtf((float)d), inv_k = 1.0f / sqrtf(kk + 1e-6f);
    const __m256 decay = _mm256_set1_ps(expf(_cvtsh_ss(c->g[h]))), b = _mm256_set1_ps(_cvtsh_ss(c->beta[h]));
    float oo = 0.0f;
    for (uint64_t j = 0u; j < d; j += 8u) {
        __m256 kv = _mm256_setzero_ps();
        for (uint64_t i = 0u; i < d; ++i) kv = _mm256_fmadd_ps(_mm256_loadu_ps(Sh + i * d + j), _mm256_set1_ps(kf[i]), kv);
        float vj[8];
        for (int t = 0; t < 8; ++t) vj[t] = _cvtsh_ss(v[j + t]);
        const __m256 delta = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(vj), _mm256_mul_ps(decay, _mm256_mul_ps(kv, _mm256_set1_ps(inv_k)))), b);
        __m256 o = _mm256_setzero_ps();
        for (uint64_t i = 0u; i < d; ++i) {
            float* row = Sh + i * d + j;
            const __m256 s = _mm256_fmadd_ps(decay, _mm256_loadu_ps(row), _mm256_mul_ps(_mm256_set1_ps(kf[i] * inv_k), delta));
            _mm256_storeu_ps(row, s);
            o = _mm256_fmadd_ps(s, _mm256_set1_ps(qf[i]), o);
        }
        float of[8];
        _mm256_storeu_ps(of, _mm256_mul_ps(o, _mm256_set1_ps(inv_q)));
        for (int t = 0; t < 8; ++t) {
            oh[j + t] = nn__kernels__zzabi_to_half(of[t], c->over);
            const float r = _cvtsh_ss(oh[j + t]);
            oo += r * r;
        }
    }
    const float scale = 1.0f / sqrtf(oo / (float)d + 1e-6f);
    for (uint64_t j = 0u; j < d; ++j) {
        const float zr = _cvtsh_ss(c->z[h * d + j]);
        const float zc = zr < -88.0f ? -88.0f : (zr > 88.0f ? 88.0f : zr);
        oh[j] = nn__kernels__zzabi_to_half(_cvtsh_ss(oh[j]) * scale * _cvtsh_ss(c->w[j]) * (zc / (1.0f + expf(-zc))), c->over);
    }
}
static void x86_avx2__override_deltanet_step(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g,
                                             const uint16_t* w, uint16_t* out, uint64_t k_heads, uint64_t v_heads, uint64_t head_dim,
                                             unsigned int* over) {
    if (head_dim == 0u || head_dim % 8u != 0u || head_dim > 256u || k_heads == 0u || v_heads % k_heads != 0u) {
        nn__deltanet__zzabi_launch_step(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, over); return;
    }
    x86_avx2__step_call c = { S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, over };
    x86_avx2__zzpackage_run((uint32_t)v_heads, x86_avx2__zzprivate_step, &c);
}

/* ── ⭐⭐ THE EXPERT-MAJOR GEMM — a row of the matrix decoded once to floats, then every pair's dot with it ───────
 * nn's definitions (`kernels.cuh`, the expert-major GEMM). A pool block takes a run of the matrix's rows; for
 * each it decodes the row into floats once — the levels as the gemv finds them — and then takes the row's dot
 * product with every pair's `x`, which was widened to floats once for the call. So a row's codes are read and
 * decoded once however many pairs use it, and one block owns a row for every pair, so the fp32 sum needs no
 * lock. Covers 4 and 8 bits over whole blocks of 32 columns up to 8192; anything else is the generic door. */
#define X86_AVX2__ZZPRIVATE_GEMM_COLS 8192u

typedef struct x86_avx2__rows_call {
    uint16_t* out; float* acc; const uint8_t* weights; const uint8_t* luts; const float* xf; const uint32_t* rows;
    const uint16_t* w; uint64_t count, d, out_rows, cols; unsigned int* over;
    nn__expert__groups g; uint64_t total;   /* several matrices: `g`, and all their rows */
    uint64_t xf_at[NN__EXPERT__GROUPS_MAX]; /* where each group's widened pairs begin in `xf` */
} x86_avx2__rows_call;

/* Row `r` of a matrix decoded into `wf`, `cols` floats. */
static inline void x86_avx2__zzprivate_decode_row(uint64_t d, const uint8_t* row, uint64_t cols, float* wf, __m256i tlo, __m256i thi) {
    if (d == 4u) {
        const __m128i fifteen = _mm_set1_epi8(15);
        for (uint64_t k = 0; k < cols; k += 32u) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
            const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
            const __m256i idx = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
            const __m256i bl = _mm256_shuffle_epi8(tlo, idx), bh = _mm256_shuffle_epi8(thi, idx);
            const __m256i h0 = _mm256_unpacklo_epi8(bl, bh), h1 = _mm256_unpackhi_epi8(bl, bh);
            _mm256_store_ps(wf + k,       _mm256_cvtph_ps(_mm256_castsi256_si128(h0)));
            _mm256_store_ps(wf + k + 8u,  _mm256_cvtph_ps(_mm256_castsi256_si128(h1)));
            _mm256_store_ps(wf + k + 16u, _mm256_cvtph_ps(_mm256_extracti128_si256(h0, 1)));
            _mm256_store_ps(wf + k + 24u, _mm256_cvtph_ps(_mm256_extracti128_si256(h1, 1)));
        }
    } else {
        for (uint64_t k = 0; k < cols; k += 8u)
            _mm256_store_ps(wf + k, _mm256_i32gather_ps(x86_avx2__zzprivate_levels8,
                                                        _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(row + k))), 4));
    }
}
static inline float x86_avx2__zzprivate_dotf(const float* a, const float* b, uint64_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (uint64_t k = 0; k < n; k += 32u) {
        a0 = _mm256_fmadd_ps(_mm256_load_ps(a + k),       _mm256_loadu_ps(b + k),       a0);
        a1 = _mm256_fmadd_ps(_mm256_load_ps(a + k + 8u),  _mm256_loadu_ps(b + k + 8u),  a1);
        a2 = _mm256_fmadd_ps(_mm256_load_ps(a + k + 16u), _mm256_loadu_ps(b + k + 16u), a2);
        a3 = _mm256_fmadd_ps(_mm256_load_ps(a + k + 24u), _mm256_loadu_ps(b + k + 24u), a3);
    }
    return x86_avx2__zzprivate_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
}
/* One matrix row against its pairs, their `x` widened at `xf`, pair `j`'s at `xf + j·cols`. */
static inline void x86_avx2__zzprivate_row_pairs(uint64_t d, const uint8_t* row, float scale, const float* xf, const uint32_t* rows,
                                                 const uint16_t* w, uint64_t count, uint64_t cols, uint64_t out_rows, uint64_t r,
                                                 uint16_t* out, float* acc, unsigned int* over, __m256i tlo, __m256i thi) {
    float wf[X86_AVX2__ZZPRIVATE_GEMM_COLS] __attribute__((aligned(32)));
    x86_avx2__zzprivate_decode_row(d, row, cols, wf, tlo, thi);
    /* ⭐ FOUR PAIRS AT A TIME against the decoded row, two sums each: a load of the row serves four products, and
     *   eight sums in flight keep both multiply-add units busy — ▶ the commit that measured it. The rest one by one. */
    float v4[4];
    uint64_t j = 0;
    for (; j + 4u <= count; j += 4u) {
        const float* x0 = xf + j * cols;
        const float* x1 = x0 + cols;
        const float* x2 = x1 + cols;
        const float* x3 = x2 + cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        for (uint64_t k = 0; k < cols; k += 16u) {
            const __m256 w0 = _mm256_load_ps(wf + k), w1 = _mm256_load_ps(wf + k + 8u);
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x0 + k), a0); b0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x0 + k + 8u), b0);
            a1 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x1 + k), a1); b1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x1 + k + 8u), b1);
            a2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x2 + k), a2); b2 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x2 + k + 8u), b2);
            a3 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x3 + k), a3); b3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x3 + k + 8u), b3);
        }
        v4[0] = x86_avx2__zzprivate_hsum(_mm256_add_ps(a0, b0));
        v4[1] = x86_avx2__zzprivate_hsum(_mm256_add_ps(a1, b1));
        v4[2] = x86_avx2__zzprivate_hsum(_mm256_add_ps(a2, b2));
        v4[3] = x86_avx2__zzprivate_hsum(_mm256_add_ps(a3, b3));
        for (uint64_t q = 0; q < 4u; ++q) {
            const float v = v4[q] * scale;
            const uint64_t at = (uint64_t)rows[2u * (j + q) + 1u] * out_rows + r;
            if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
            else acc[at] += v * _cvtsh_ss(w[rows[2u * (j + q)]]);
        }
    }
    for (; j < count; ++j) {
        const float v = x86_avx2__zzprivate_dotf(wf, xf + j * cols, cols) * scale;
        const uint64_t at = (uint64_t)rows[2u * j + 1u] * out_rows + r;
        if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
        else acc[at] += v * _cvtsh_ss(w[rows[2u * j]]);
    }
}
static void x86_avx2__zzprivate_expert_rows(void* a) {
    const x86_avx2__rows_call* c = (const x86_avx2__rows_call*)a;
    __m256i tlo, thi; x86_avx2__zzprivate_tables4(&tlo, &thi);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t r0 = c->out_rows * b / n, r1 = c->out_rows * (b + 1u) / n, row_bytes = c->cols * c->d / 8u;
    for (uint64_t r = r0; r < r1; ++r)
        x86_avx2__zzprivate_row_pairs(c->d, c->weights + r * row_bytes, x86_avx2__zzprivate_row_scale(c->luts, r), c->xf, c->rows, c->w,
                                      c->count, c->cols, c->out_rows, r, c->out, c->acc, c->over, tlo, thi);
}
static void x86_avx2__zzprivate_expert_groups(void* a) {
    const x86_avx2__rows_call* c = (const x86_avx2__rows_call*)a;
    __m256i tlo, thi; x86_avx2__zzprivate_tables4(&tlo, &thi);
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->total * b / n, t1 = c->total * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.out_rows[i]) { base += c->g.out_rows[i]; ++i; }
        const uint64_t r = t - base;
        x86_avx2__zzprivate_row_pairs(c->g.d[i], (const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols * c->g.d[i] / 8u),
                                      x86_avx2__zzprivate_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r), c->xf + c->xf_at[i],
                                      c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i], c->cols, c->g.out_rows[i], r,
                                      c->out + c->g.out_at[i], 0, c->over, tlo, thi);
    }
}
/* Pairs' `x` rows widened to floats, pair `j` at `xf + j·cols`. */
static inline void x86_avx2__zzprivate_widen_pairs(float* xf, const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t cols) {
    for (uint64_t j = 0; j < count; ++j) {
        const uint16_t* xr = x + (uint64_t)rows[2u * j] * cols;
        for (uint64_t k = 0; k < cols; k += 8u)
            _mm256_storeu_ps(xf + j * cols + k, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(xr + k))));
    }
}
static inline bool x86_avx2__zzprivate_gemm_fit(uint64_t d, uint64_t cols) {
    return (d == 4u || d == 8u) && cols != 0u && cols % 32u == 0u && cols <= X86_AVX2__ZZPRIVATE_GEMM_COLS;
}
static void x86_avx2__override_expert_rows(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                           const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d, uint64_t out_rows,
                                           uint64_t cols, unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (!x86_avx2__zzprivate_gemm_fit(d, cols) || count == 0u || out_rows > w_room / row_bytes || out_rows > l_room / 2u
     || !x86_avx2__zzprivate_x_room_for(count * cols)) {
        nn__expert__zzabi_launch_rows(out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_widen_pairs(x86_avx2__zzprivate_xf, x, rows, count, cols);
    x86_avx2__rows_call c = {};
    c.out = out; c.weights = weights; c.luts = luts; c.xf = x86_avx2__zzprivate_xf; c.rows = rows;
    c.count = count; c.d = d; c.out_rows = out_rows; c.cols = cols; c.over = over;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(out_rows), x86_avx2__zzprivate_expert_rows, &c);
}
static void x86_avx2__override_expert_rows_sum(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                               const uint16_t* x, const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t d,
                                               uint64_t out_rows, uint64_t cols) {
    const uint64_t row_bytes = cols * d / 8u;
    if (!x86_avx2__zzprivate_gemm_fit(d, cols) || count == 0u || out_rows > w_room / row_bytes || out_rows > l_room / 2u
     || !x86_avx2__zzprivate_x_room_for(count * cols)) {
        nn__expert__zzabi_launch_rows_sum(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols); return;
    }
    x86_avx2__zzpackage_start();
    x86_avx2__zzprivate_widen_pairs(x86_avx2__zzprivate_xf, x, rows, count, cols);
    x86_avx2__rows_call c = {};
    c.acc = acc; c.weights = weights; c.luts = luts; c.xf = x86_avx2__zzprivate_xf; c.rows = rows; c.w = w;
    c.count = count; c.d = d; c.out_rows = out_rows; c.cols = cols;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(out_rows), x86_avx2__zzprivate_expert_rows, &c);
}
static void x86_avx2__override_expert_groups(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows, uint64_t cols,
                                             unsigned int* over) {
    bool fit = g.count != 0u && g.count <= NN__EXPERT__GROUPS_MAX;
    uint64_t total = 0u, pairs = 0u;
    for (uint64_t i = 0u; fit && i < g.count; ++i) {
        fit = x86_avx2__zzprivate_gemm_fit(g.d[i], cols) && g.out_rows[i] != 0u;
        total += g.out_rows[i]; pairs += g.pairs[i];
    }
    if (!fit || !x86_avx2__zzprivate_x_room_for(pairs * cols)) { nn__expert__zzabi_launch_groups(out, g, x, rows, cols, over); return; }
    x86_avx2__zzpackage_start();
    x86_avx2__rows_call c = {};
    uint64_t placed = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) {
        c.xf_at[i] = placed * cols;
        x86_avx2__zzprivate_widen_pairs(x86_avx2__zzprivate_xf + placed * cols, x, rows + 2u * g.pairs_at[i], g.pairs[i], cols);
        placed += g.pairs[i];
    }
    c.out = out; c.xf = x86_avx2__zzprivate_xf; c.rows = rows; c.cols = cols; c.over = over; c.g = g; c.total = total;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(total), x86_avx2__zzprivate_expert_groups, &c);
}


/* ══ ⭐⭐ THE EXPERT-MAJOR GEMM IN INTEGERS — nn's `expert_groups_int8` / `expert_rows_sum_int8` definition: each row of
 *   `x` quantised once a call, a block of `NN__EXPERT__INT8_BLOCK` at a time; a matrix row's 4-bit codes to their int8
 *   levels once (one `pshufb`), then four pairs at a time, `abs(w)·sign(x, w)` in `maddubs`, integer sums a block, one
 *   float step a block. `MEASURED` (the probe beside NN-35's measurements, 28 threads): 300-545 GMAC/s against the
 *   fp32 GEMM's 100-132. */
typedef struct x86_avx2__int8_call {
    const uint16_t* x; int8_t* xq; float* xs; uint64_t cols, x0, nx;                  /* rows x0 .. x0 + nx, quantised */
    uint16_t* out; float* acc; const uint8_t* weights; const uint8_t* luts; const uint32_t* rows; const uint16_t* w;
    uint64_t count, out_rows;                                                          /* one matrix */
    nn__expert__groups g; uint64_t total;                                              /* several */
    unsigned int* over;
} x86_avx2__int8_call;

/* Block `x[0..256)` quantised into `q` as nn's definition says; its scale answered. */
static inline float x86_avx2__zzprivate_quantise256(const uint16_t* x, int8_t* q) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 mx = _mm256_setzero_ps();
    for (uint32_t k = 0; k < NN__EXPERT__INT8_BLOCK; k += 8u)
        mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + k))), absmask));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    const float m = _mm_cvtss_f32(m4), inv = m > 0.0f ? 127.0f / m : 0.0f;
    const __m256 vinv = _mm256_set1_ps(inv), half = _mm256_set1_ps(0.5f);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (uint32_t k = 0; k < NN__EXPERT__INT8_BLOCK; k += 32u) {
        __m256i v[4];
        for (uint32_t i = 0; i < 4u; ++i) {
            const __m256 t = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + k + 8u * i))), vinv);
            /* half away from zero: the magnitude plus a half, truncated, the sign put back */
            v[i] = _mm256_sign_epi32(_mm256_cvttps_epi32(_mm256_add_ps(_mm256_and_ps(t, absmask), half)), _mm256_castps_si256(t));
        }
        const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(v[0], v[1]), _mm256_packs_epi32(v[2], v[3]));
        _mm256_storeu_si256((__m256i*)(q + k), _mm256_permutevar8x32_epi32(p, order));
    }
    return m / 127.0f;
}
static void x86_avx2__zzprivate_quantise_rows(void* a) {
    const x86_avx2__int8_call* c = (const x86_avx2__int8_call*)a;
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks(), per = c->cols / NN__EXPERT__INT8_BLOCK;
    for (uint64_t r = c->nx * b / n; r < c->nx * (b + 1u) / n; ++r)
        for (uint64_t k = 0; k < per; ++k)
            c->xs[r * per + k] = x86_avx2__zzprivate_quantise256(c->x + (c->x0 + r) * c->cols + k * NN__EXPERT__INT8_BLOCK,
                                                                   c->xq + r * c->cols + k * NN__EXPERT__INT8_BLOCK);
}
/* One matrix row against its pairs: the codes to int8 levels once, then the pairs four at a time. */
static inline void x86_avx2__zzprivate_row_pairs_i8(const uint8_t* row, float scale, const int8_t* xq, const float* xs, uint64_t x0,
                                                    const uint32_t* rows, const uint16_t* w, uint64_t count, uint64_t cols,
                                                    uint64_t out_rows, uint64_t r, uint16_t* out, float* acc, unsigned int* over,
                                                    __m256i t8) {
    int8_t w8[X86_AVX2__ZZPRIVATE_GEMM_COLS] __attribute__((aligned(32)));
    const __m128i fifteen = _mm_set1_epi8(15);
    for (uint64_t k = 0; k < cols; k += 32u) {
        const __m128i v = _mm_loadu_si128((const __m128i*)(row + k / 2u));
        const __m128i lo = _mm_and_si128(v, fifteen), hi = _mm_and_si128(_mm_srli_epi16(v, 4), fifteen);
        _mm256_store_si256((__m256i*)(w8 + k), _mm256_shuffle_epi8(t8, _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi))));
    }
    const __m256i ones = _mm256_set1_epi16(1);
    const uint64_t per = cols / NN__EXPERT__INT8_BLOCK;
    for (uint64_t j = 0; j < count; j += 4u) {
        const uint64_t n = count - j < 4u ? count - j : 4u;
        const int8_t* xr[4];
        const float* sr[4];
        for (uint64_t q = 0; q < n; ++q) {
            const uint64_t xrow = (uint64_t)rows[2u * (j + q)] - x0;
            xr[q] = xq + xrow * cols; sr[q] = xs + xrow * per;
        }
        __m256 f[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (uint64_t B = 0; B < per; ++B) {
            __m256i s4[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
            for (uint64_t k = B * NN__EXPERT__INT8_BLOCK; k < (B + 1u) * NN__EXPERT__INT8_BLOCK; k += 32u) {
                const __m256i wv = _mm256_load_si256((const __m256i*)(w8 + k)), aw = _mm256_abs_epi8(wv);
                for (uint64_t q = 0; q < n; ++q)
                    s4[q] = _mm256_add_epi32(s4[q], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(
                                                 _mm256_loadu_si256((const __m256i*)(xr[q] + k)), wv)), ones));
            }
            for (uint64_t q = 0; q < n; ++q) f[q] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s4[q]), _mm256_set1_ps(sr[q][B]), f[q]);
        }
        for (uint64_t q = 0; q < n; ++q) {
            const float v = x86_avx2__zzprivate_hsum(f[q]) * scale / 127.0f;
            const uint64_t at = (uint64_t)rows[2u * (j + q) + 1u] * out_rows + r;
            if (out) out[at] = nn__kernels__zzabi_to_half(v, over);
            else acc[at] += v * _cvtsh_ss(w[rows[2u * (j + q)]]);
        }
    }
}
static void x86_avx2__zzprivate_expert_rows_i8(void* a) {
    const x86_avx2__int8_call* c = (const x86_avx2__int8_call*)a;
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)nn__turboquant__zzabi_levels_i8(4u)));
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks(), row_bytes = c->cols / 2u;
    for (uint64_t r = c->out_rows * b / n; r < c->out_rows * (b + 1u) / n; ++r)
        x86_avx2__zzprivate_row_pairs_i8(c->weights + r * row_bytes, x86_avx2__zzprivate_row_scale(c->luts, r), c->xq, c->xs, c->x0, c->rows,
                                         c->w, c->count, c->cols, c->out_rows, r, c->out, c->acc, c->over, t8);
}
static void x86_avx2__zzprivate_expert_groups_i8(void* a) {
    const x86_avx2__int8_call* c = (const x86_avx2__int8_call*)a;
    const __m256i t8 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)nn__turboquant__zzabi_levels_i8(4u)));
    const uint64_t b = nn__silicon__block(), n = nn__silicon__blocks();
    const uint64_t t0 = c->total * b / n, t1 = c->total * (b + 1u) / n;
    uint64_t i = 0u, base = 0u;
    for (uint64_t t = t0; t < t1; ++t) {
        while (t >= base + c->g.out_rows[i]) { base += c->g.out_rows[i]; ++i; }
        const uint64_t r = t - base;
        x86_avx2__zzprivate_row_pairs_i8((const uint8_t*)(uintptr_t)c->g.codes[i] + r * (c->cols / 2u),
                                         x86_avx2__zzprivate_row_scale((const uint8_t*)(uintptr_t)c->g.luts[i], r), c->xq, c->xs, c->x0,
                                         c->rows + 2u * c->g.pairs_at[i], 0, c->g.pairs[i], c->cols, c->g.out_rows[i], r,
                                         c->out + c->g.out_at[i], 0, c->over, t8);
    }
}
/* The rows of `x` the pairs name — the run from the lowest to the highest — quantised once, on the pool. False when
 * there is no room for them. */
static inline bool x86_avx2__zzprivate_quantise_pairs(x86_avx2__int8_call* c, const uint16_t* x, const uint32_t* rows,
                                                      uint64_t pairs, uint64_t cols) {
    uint64_t lo = ~0ull, hi = 0u;
    for (uint64_t j = 0; j < pairs; ++j) {
        if ((uint64_t)rows[2u * j] < lo) lo = rows[2u * j];
        if ((uint64_t)rows[2u * j] + 1u > hi) hi = (uint64_t)rows[2u * j] + 1u;
    }
    if (pairs == 0u || !x86_avx2__zzprivate_x_room_for((hi - lo) * cols)) return false;
    c->x = x; c->xq = x86_avx2__zzprivate_xq; c->xs = x86_avx2__zzprivate_xs; c->cols = cols; c->x0 = lo; c->nx = hi - lo;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(c->nx), x86_avx2__zzprivate_quantise_rows, c);
    return true;
}
static inline bool x86_avx2__zzprivate_int8_fit(uint64_t d, uint64_t cols) {
    return d == 4u && cols != 0u && cols % NN__EXPERT__INT8_BLOCK == 0u && cols <= X86_AVX2__ZZPRIVATE_GEMM_COLS;
}
static void x86_avx2__override_expert_rows_sum_int8(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, const uint32_t* rows, const uint16_t* w,
                                                    uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols) {
    x86_avx2__int8_call c = {};
    if (!x86_avx2__zzprivate_int8_fit(d, cols) || count == 0u || out_rows > w_room / (cols / 2u) || out_rows > l_room / 2u) {
        nn__expert__zzabi_launch_rows_sum_int8(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols); return;
    }
    x86_avx2__zzpackage_start();
    if (!x86_avx2__zzprivate_quantise_pairs(&c, x, rows, count, cols)) {
        nn__expert__zzabi_launch_rows_sum_int8(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols); return;
    }
    c.acc = acc; c.weights = weights; c.luts = luts; c.rows = rows; c.w = w; c.count = count; c.out_rows = out_rows;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(out_rows), x86_avx2__zzprivate_expert_rows_i8, &c);
}
static void x86_avx2__override_expert_groups_int8(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows,
                                                  uint64_t cols, unsigned int* over) {
    bool fit = g.count != 0u && g.count <= NN__EXPERT__GROUPS_MAX;
    uint64_t total = 0u, last = 0u;
    for (uint64_t i = 0u; fit && i < g.count; ++i) {
        fit = x86_avx2__zzprivate_int8_fit(g.d[i], cols) && g.out_rows[i] != 0u;
        total += g.out_rows[i];
        if (g.pairs_at[i] + g.pairs[i] > last) last = g.pairs_at[i] + g.pairs[i];
    }
    x86_avx2__int8_call c = {};
    if (fit) x86_avx2__zzpackage_start();
    if (!fit || !x86_avx2__zzprivate_quantise_pairs(&c, x, rows, last, cols)) {
        nn__expert__zzabi_launch_groups_int8(out, g, x, rows, cols, over); return;
    }
    c.out = out; c.rows = rows; c.g = g; c.total = total; c.over = over;
    x86_avx2__zzpackage_run(x86_avx2__zzprivate_blocks_for(total), x86_avx2__zzprivate_expert_groups_i8, &c);
}

/* Into nn's table, once, when the family is first asked for it. */
static inline void x86_avx2__override_doors(nn__doors* doors) {
    const uint16_t* lev = nn__turboquant__zzabi_levels(8u);
    for (int i = 0; i < 256; ++i) x86_avx2__zzprivate_levels8[i] = _cvtsh_ss(lev[i]);
    doors->turboquant_gemv      = x86_avx2__override_turboquant_gemv;
    doors->turboquant_gemv_int8 = x86_avx2__override_turboquant_gemv_int8;
    doors->turboquant_gemv_groups     = x86_avx2__override_gemv_groups;
    doors->turboquant_gemv_groups_sum = x86_avx2__override_gemv_groups_sum;
    doors->turboquant_gemv_groups_int8     = x86_avx2__override_gemv_groups_int8;
    doors->turboquant_gemv_groups_sum_int8 = x86_avx2__override_gemv_groups_sum_int8;
    doors->hadamard_rotate      = x86_avx2__override_hadamard_rotate;
    doors->deltanet_step        = x86_avx2__override_deltanet_step;
    doors->expert_rows          = x86_avx2__override_expert_rows;
    doors->expert_rows_sum      = x86_avx2__override_expert_rows_sum;
    doors->expert_groups        = x86_avx2__override_expert_groups;
    doors->expert_groups_int8   = x86_avx2__override_expert_groups_int8;
    doors->expert_rows_sum_int8 = x86_avx2__override_expert_rows_sum_int8;
}

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_OVERRIDES_NN_CUH */
