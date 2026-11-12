#ifndef SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_ATTENTION_CUH
#define SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_ATTENTION_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include "../primitives/pool.cuh"   /* the threads the heads and positions are spread over */

/* ══ ⭐⭐ ATTENTION ON NEON — nn's `attention_*` doors over a cache: a decoded position's and a prompt's ═══════════════════
 * nn's bodies give a query head a block, and on this family's one lane that block walks each value element down the whole
 * cache — a value row read once an element. `MEASURED` (the 2B on the Razer Edge, the launch profile,
 * measurements/2026-11-11_nn175_arm_doors_razer_edge.md): `residual_mix` 320 µs a call after ~200 positions, `causal_mix`
 * 15 ms a prompt layer. Here, as `x86_avx2`'s:
 *   the scores   a pool block a (kv head, run of positions): each key row widened once — or decoded, a TurboQuant tier's —
 *                and its dot taken with every query head of the group, sixteen columns a step in four running sums;
 *   the softmax  a block a query head, nn's: its largest, `e^(score − largest)` in place, their sum position by position;
 *   a mix        a block a (row, kv head, 16 elements), the group's heads four at a time against each value row: each
 *                element's sum adds the positions one after another from the first, a multiply-add each, as nn's body does
 *                where its compiler fuses `sum += p · v` — so a mix is nn's bytes (`MEASURED` the door bench on the Razer
 *                Edge, clang 21: every mix's outputs worst 0 ulp from nn's host body, measurements/2026-11-11_nn175_arm_doors_razer_edge.md).
 * The decode's mix (`attention_residual_mix`, `_tq`), the prompt's (`attention_causal_mix`) and the one-call answer
 * (`attention_mix`) are one body, so a position decoded and the same position read as a prompt row mix alike.
 * The scores take heads whole runs of 32 up to 512 wide and a group's queries at most 4,096 floats, a mix heads whole runs
 * of 16, both the 4- and 8-bit tiers; anything else is the generic door. */
#define ARM_BASE__ZZPRIVATE_ATTEND_SPAN  16u     /* positions a block of the decode's scores takes */
#define ARM_BASE__ZZPRIVATE_CAUSAL_SPAN  16u     /* positions a block of the prompt's scores takes: their keys on the stack */
#define ARM_BASE__ZZPRIVATE_ATTEND_WIDTH 512u    /* the widest head */
#define ARM_BASE__ZZPRIVATE_ATTEND_GROUP 4096u   /* a kv head's query heads, as floats on a block's stack */

/* `n` halves as floats (whole runs of 16). */
static inline void arm_base__zzprivate_attend_widen(float* out, const uint16_t* h, uint64_t n) {
    for (uint64_t e = 0; e < n; e += 16u) {
        const float16x8_t a = vreinterpretq_f16_u16(vld1q_u16(h + e)), b = vreinterpretq_f16_u16(vld1q_u16(h + e + 8u));
        vst1q_f32(out + e,      vcvt_f32_f16(vget_low_f16(a)));
        vst1q_f32(out + e + 4u, vcvt_high_f32_f16(a));
        vst1q_f32(out + e + 8u, vcvt_f32_f16(vget_low_f16(b)));
        vst1q_f32(out + e + 12u, vcvt_high_f32_f16(b));
    }
}
/* A score's dot: sixteen columns a step, columns 4s .. 4s + 3 of each into running sum `s`, the four added as
 * (s0 + s1) + (s2 + s3) and then across — one order for every score. */
static inline float arm_base__zzprivate_attend_dot(const float* a, const float* b, uint64_t n) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
    for (uint64_t e = 0; e < n; e += 16u) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + e),       vld1q_f32(b + e));
        s1 = vfmaq_f32(s1, vld1q_f32(a + e + 4u),  vld1q_f32(b + e + 4u));
        s2 = vfmaq_f32(s2, vld1q_f32(a + e + 8u),  vld1q_f32(b + e + 8u));
        s3 = vfmaq_f32(s3, vld1q_f32(a + e + 12u), vld1q_f32(b + e + 12u));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
}

/* ── A TURBOQUANT TIER'S ROW AS FLOATS — `scale · level`, as nn's element (`nn__turboquant__zzprivate_element`) ──────
 * 4 bits: a byte's two codes split, each code's half found by `tbl` — its low and high bytes from two tables of sixteen —
 * and widened; 8 bits: the halves gathered from nn's table a code at a time. Elements `e0 .. e0 + n`, whole runs of 16. */
typedef struct arm_base__attend_levels { uint8x16_t lo, hi; const uint16_t* lev8; } arm_base__attend_levels;
static inline arm_base__attend_levels arm_base__zzprivate_attend_levels(uint64_t d) {
    arm_base__attend_levels t;
    uint8_t lo[16], hi[16];
    const uint16_t* lev4 = nn__turboquant__zzabi_levels(4u);
    for (int i = 0; i < 16; ++i) { lo[i] = (uint8_t)lev4[i]; hi[i] = (uint8_t)(lev4[i] >> 8); }
    t.lo = vld1q_u8(lo); t.hi = vld1q_u8(hi);
    t.lev8 = d == 8u ? nn__turboquant__zzabi_levels(8u) : 0;
    return t;
}
static inline void arm_base__zzprivate_attend_row_tq(float* out, const uint8_t* codes, uint16_t scale_h, uint64_t d, uint64_t e0,
                                                    uint64_t n, const arm_base__attend_levels* t) {
    const float scale = nn__silicon__half_to_float(scale_h);
    for (uint64_t e = e0; e < e0 + n; e += 16u) {
        float16x8_t h0, h1;
        if (d == 4u) {
            const uint8x8_t v = vld1_u8(codes + e / 2u);
            const uint8x8x2_t z = vzip_u8(vand_u8(v, vdup_n_u8(15u)), vshr_n_u8(v, 4));
            const uint8x16_t at = vcombine_u8(z.val[0], z.val[1]);
            const uint8x16_t bl = vqtbl1q_u8(t->lo, at), bh = vqtbl1q_u8(t->hi, at);
            h0 = vreinterpretq_f16_u8(vzip1q_u8(bl, bh)); h1 = vreinterpretq_f16_u8(vzip2q_u8(bl, bh));
        } else {
            uint16_t g[16];
            for (int i = 0; i < 16; ++i) g[i] = t->lev8[codes[e + (uint64_t)i]];
            h0 = vreinterpretq_f16_u16(vld1q_u16(g)); h1 = vreinterpretq_f16_u16(vld1q_u16(g + 8));
        }
        float* o = out + (e - e0);
        vst1q_f32(o,       vmulq_n_f32(vcvt_f32_f16(vget_low_f16(h0)), scale));
        vst1q_f32(o + 4u,  vmulq_n_f32(vcvt_high_f32_f16(h0), scale));
        vst1q_f32(o + 8u,  vmulq_n_f32(vcvt_f32_f16(vget_low_f16(h1)), scale));
        vst1q_f32(o + 12u, vmulq_n_f32(vcvt_high_f32_f16(h1), scale));
    }
}

/* ── THE DECODE'S WEIGHTS — `attention_weights`, `attention_weights_tq`: the scores, then each head's softmax half ─── */
typedef struct arm_base__attend_call {
    float* p; float* res; const uint16_t* q; const uint16_t* k; const uint8_t* codes; const uint16_t* scales;
    uint64_t q_heads, kv_heads, head_dim, length, d;
} arm_base__attend_call;

/* The scores, a block a (kv head, run of positions): `p[h·length + t]`, scaled — the group's queries widened once a block. */
static void arm_base__zzprivate_attend_scores(void* a) {
    const arm_base__attend_call* c = (const arm_base__attend_call*)a;
    const uint64_t runs = (c->length + ARM_BASE__ZZPRIVATE_ATTEND_SPAN - 1u) / ARM_BASE__ZZPRIVATE_ATTEND_SPAN;
    const uint64_t b = nn__silicon__block(), g = b / runs, t0 = (b % runs) * ARM_BASE__ZZPRIVATE_ATTEND_SPAN;
    const uint64_t t1 = t0 + ARM_BASE__ZZPRIVATE_ATTEND_SPAN < c->length ? t0 + ARM_BASE__ZZPRIVATE_ATTEND_SPAN : c->length;
    const uint64_t group = c->q_heads / c->kv_heads, hd = c->head_dim;
    const float scale = 1.0f / sqrtf((float)(uint32_t)hd);
    float qf[ARM_BASE__ZZPRIVATE_ATTEND_GROUP], kf[ARM_BASE__ZZPRIVATE_ATTEND_WIDTH];
    arm_base__zzprivate_attend_widen(qf, c->q + g * group * hd, group * hd);
    const arm_base__attend_levels t = arm_base__zzprivate_attend_levels(c->d);
    const uint64_t row_bytes = c->codes ? nn__turboquant__row_bytes(c->d, hd) : 0u;
    for (uint64_t pos = t0; pos < t1; ++pos) {
        const uint64_t row = pos * c->kv_heads + g;
        if (c->codes) arm_base__zzprivate_attend_row_tq(kf, c->codes + row * row_bytes, c->scales[row], c->d, 0u, hd, &t);
        else arm_base__zzprivate_attend_widen(kf, c->k + row * hd, hd);
        for (uint64_t hg = 0; hg < group; ++hg)
            c->p[(g * group + hg) * c->length + pos] = arm_base__zzprivate_attend_dot(kf, qf + hg * hd, hd) * scale;
    }
}
/* The softmax's first half, a block a query head: its largest, `e^(score − largest)` in place and their sum, as nn's. */
static void arm_base__zzprivate_attend_softmax(void* a) {
    const arm_base__attend_call* c = (const arm_base__attend_call*)a;
    const uint64_t h = nn__silicon__block();
    float* ph = c->p + h * c->length;
    float top = -NN__KERNELS__INFINITY;
    for (uint64_t t = 0; t < c->length; ++t) if (ph[t] > top) top = ph[t];
    float sum = 0.0f;
    for (uint64_t t = 0; t < c->length; ++t) { ph[t] = expf(ph[t] - top); sum += ph[t]; }
    c->res[h * NN__ATTENTION__RESIDUAL_FLOATS(c->head_dim)] = top;
    c->res[h * NN__ATTENTION__RESIDUAL_FLOATS(c->head_dim) + 1u] = sum;
}
static inline bool arm_base__zzprivate_attend_fit(uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim) {
    return kv_heads != 0u && q_heads % kv_heads == 0u && head_dim != 0u && head_dim % 32u == 0u
        && head_dim <= ARM_BASE__ZZPRIVATE_ATTEND_WIDTH && (q_heads / kv_heads) * head_dim <= ARM_BASE__ZZPRIVATE_ATTEND_GROUP;
}
static inline void arm_base__zzprivate_attend_weights(arm_base__attend_call* c) {
    (void)arm_base__zzpackage_start();
    const uint64_t runs = (c->length + ARM_BASE__ZZPRIVATE_ATTEND_SPAN - 1u) / ARM_BASE__ZZPRIVATE_ATTEND_SPAN;
    arm_base__zzpackage_run((uint32_t)(c->kv_heads * runs), arm_base__zzprivate_attend_scores, c);
    arm_base__zzpackage_run((uint32_t)c->q_heads, arm_base__zzprivate_attend_softmax, c);
}
static void arm_base__override_attention_weights(float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                 uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!arm_base__zzprivate_attend_fit(q_heads, kv_heads, head_dim) || length == 0u) {
        nn__attention__zzabi_launch_weights(p, res, q, k, q_heads, kv_heads, head_dim, length); return;
    }
    arm_base__attend_call c = { p, res, q, k, 0, 0, q_heads, kv_heads, head_dim, length, 0u };
    arm_base__zzprivate_attend_weights(&c);
}
static void arm_base__override_attention_weights_tq(float* p, float* res, const uint16_t* q, const uint8_t* codes, const uint16_t* scales,
                                                    uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d) {
    if (!arm_base__zzprivate_attend_fit(q_heads, kv_heads, head_dim) || length == 0u || (d != 4u && d != 8u)) {
        nn__attention__zzabi_launch_weights_tq(p, res, q, codes, scales, q_heads, kv_heads, head_dim, length, d); return;
    }
    arm_base__attend_call c = { p, res, q, 0, codes, scales, q_heads, kv_heads, head_dim, length, d };
    arm_base__zzprivate_attend_weights(&c);
}

/* ── EVERY MIX — a block a (row, kv head, 16 elements), the group's heads four at a time ───────────────────────────────
 * Row `r` of `rows` mixes the first `first + r + 1` positions with its heads' weights at `p + (r·q_heads + h)·span`; its
 * answer goes to `res` (a residual's `acc`, floats) or to `o` (halves, the fault word raised on an overflow). The decode's
 * mix is row 0 of one, `first` its length less one. */
typedef struct arm_base__mix_call {
    const float* p; const uint16_t* v; const uint8_t* codes; const uint16_t* scales; float* res; uint16_t* o; unsigned int* over;
    uint64_t q_heads, kv_heads, head_dim, first, rows, span, d;
} arm_base__mix_call;
static void arm_base__zzprivate_mix(void* a) {
    const arm_base__mix_call* c = (const arm_base__mix_call*)a;
    const uint64_t hd = c->head_dim, chunks = hd / 16u, b = nn__silicon__block(), e0 = (b % chunks) * 16u;
    const uint64_t g = (b / chunks) % c->kv_heads, r = b / chunks / c->kv_heads, len = c->first + r + 1u;
    const uint64_t group = c->q_heads / c->kv_heads;
    const arm_base__attend_levels t = arm_base__zzprivate_attend_levels(c->d);
    const uint64_t row_bytes = c->codes ? nn__turboquant__row_bytes(c->d, hd) : 0u;
    float vf[16];
    for (uint64_t h0 = 0; h0 < group; h0 += 4u) {
        /* four heads' weights; past the group's end a head repeats the last, and its sums are not kept */
        const float* ph[4];
        for (uint64_t j = 0; j < 4u; ++j) ph[j] = c->p + (r * c->q_heads + g * group + (h0 + j < group ? h0 + j : group - 1u)) * c->span;
        float32x4_t s[4][4];
        for (int j = 0; j < 4; ++j) for (int e = 0; e < 4; ++e) s[j][e] = vdupq_n_f32(0.0f);
        for (uint64_t pos = 0; pos < len; ++pos) {
            const uint64_t row = pos * c->kv_heads + g;
            float32x4_t v[4];
            if (c->codes) {
                arm_base__zzprivate_attend_row_tq(vf, c->codes + row * row_bytes, c->scales[row], c->d, e0, 16u, &t);
                for (int e = 0; e < 4; ++e) v[e] = vld1q_f32(vf + 4 * e);
            } else {
                const uint16_t* vr = c->v + row * hd + e0;
                const float16x8_t x0 = vreinterpretq_f16_u16(vld1q_u16(vr)), x1 = vreinterpretq_f16_u16(vld1q_u16(vr + 8u));
                v[0] = vcvt_f32_f16(vget_low_f16(x0)); v[1] = vcvt_high_f32_f16(x0);
                v[2] = vcvt_f32_f16(vget_low_f16(x1)); v[3] = vcvt_high_f32_f16(x1);
            }
            const float w[4] = { ph[0][pos], ph[1][pos], ph[2][pos], ph[3][pos] };
            const float32x4_t wv = vld1q_f32(w);
            for (int e = 0; e < 4; ++e) {
                s[0][e] = vfmaq_laneq_f32(s[0][e], v[e], wv, 0);
                s[1][e] = vfmaq_laneq_f32(s[1][e], v[e], wv, 1);
                s[2][e] = vfmaq_laneq_f32(s[2][e], v[e], wv, 2);
                s[3][e] = vfmaq_laneq_f32(s[3][e], v[e], wv, 3);
            }
        }
        for (uint64_t j = 0; j < 4u && h0 + j < group; ++j) {
            const uint64_t i = r * c->q_heads + g * group + h0 + j;
            if (c->res) {
                float* acc = c->res + i * NN__ATTENTION__RESIDUAL_FLOATS(hd) + 2u + e0;
                for (int e = 0; e < 4; ++e) vst1q_f32(acc + 4 * e, s[j][e]);
            } else {
                float sums[16];
                for (int e = 0; e < 4; ++e) vst1q_f32(sums + 4 * e, s[j][e]);
                uint16_t* oh = c->o + i * hd + e0;
                for (int e = 0; e < 16; ++e) oh[e] = nn__kernels__zzabi_to_half(sums[e], c->over);
            }
        }
    }
}
static inline bool arm_base__zzprivate_mix_fit(uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim) {
    return kv_heads != 0u && q_heads % kv_heads == 0u && head_dim != 0u && head_dim % 16u == 0u;
}
static inline void arm_base__zzprivate_mix_run(const arm_base__mix_call* c) {
    (void)arm_base__zzpackage_start();
    arm_base__zzpackage_run((uint32_t)(c->rows * c->kv_heads * (c->head_dim / 16u)), arm_base__zzprivate_mix, (void*)(uintptr_t)c);
}
static void arm_base__override_attention_residual_mix(float* res, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads,
                                                      uint64_t head_dim, uint64_t length) {
    if (!arm_base__zzprivate_mix_fit(q_heads, kv_heads, head_dim) || length == 0u) {
        nn__attention__zzabi_launch_residual_mix(res, p, v, q_heads, kv_heads, head_dim, length); return;
    }
    const arm_base__mix_call c = { p, v, 0, 0, res, 0, 0, q_heads, kv_heads, head_dim, length - 1u, 1u, length, 0u };
    arm_base__zzprivate_mix_run(&c);
}
static void arm_base__override_attention_residual_mix_tq(float* res, const float* p, const uint8_t* codes, const uint16_t* scales,
                                                         uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                         uint64_t d) {
    if (!arm_base__zzprivate_mix_fit(q_heads, kv_heads, head_dim) || length == 0u || (d != 4u && d != 8u)) {
        nn__attention__zzabi_launch_residual_mix_tq(res, p, codes, scales, q_heads, kv_heads, head_dim, length, d); return;
    }
    const arm_base__mix_call c = { p, 0, codes, scales, res, 0, 0, q_heads, kv_heads, head_dim, length - 1u, 1u, length, d };
    arm_base__zzprivate_mix_run(&c);
}
static void arm_base__override_attention_causal_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads,
                                                    uint64_t head_dim, uint64_t first, uint64_t rows, unsigned int* over) {
    if (!arm_base__zzprivate_mix_fit(q_heads, kv_heads, head_dim) || rows == 0u) {
        nn__attention__zzabi_launch_causal_mix(o, p, v, q_heads, kv_heads, head_dim, first, rows, over); return;
    }
    const arm_base__mix_call c = { p, v, 0, 0, 0, o, over, q_heads, kv_heads, head_dim, first, rows, first + rows, 0u };
    arm_base__zzprivate_mix_run(&c);
}
static void arm_base__override_attention_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads,
                                             uint64_t head_dim, uint64_t length, unsigned int* over) {
    if (!arm_base__zzprivate_mix_fit(q_heads, kv_heads, head_dim) || length == 0u) {
        nn__attention__zzabi_launch_mix(o, p, v, q_heads, kv_heads, head_dim, length, over); return;
    }
    const arm_base__mix_call c = { p, v, 0, 0, 0, o, over, q_heads, kv_heads, head_dim, length - 1u, 1u, length, 0u };
    arm_base__zzprivate_mix_run(&c);
}

/* ── THE PROMPT'S SCORES — `attention_causal_scores_grouped`: `rows` queries at positions `first ..`, each over the cache
 * up to its own position, row `r` head `h`'s at `p + (r·q_heads + h)·(first + rows)` and nothing past its length written.
 * A block a (kv head, run of 16 positions): the run's keys widened once onto the stack, then each query of the group in
 * every row that reaches the run widened once and its dots taken with the keys it reaches. */
typedef struct arm_base__causal_call {
    float* p; const uint16_t* q; const uint16_t* k; uint64_t q_heads, kv_heads, head_dim, first, rows;
} arm_base__causal_call;
static void arm_base__zzprivate_causal_scores(void* a) {
    const arm_base__causal_call* c = (const arm_base__causal_call*)a;
    const uint64_t span = c->first + c->rows, runs = (span + ARM_BASE__ZZPRIVATE_CAUSAL_SPAN - 1u) / ARM_BASE__ZZPRIVATE_CAUSAL_SPAN;
    const uint64_t b = nn__silicon__block(), g = b / runs, t0 = (b % runs) * ARM_BASE__ZZPRIVATE_CAUSAL_SPAN;
    const uint64_t t1 = t0 + ARM_BASE__ZZPRIVATE_CAUSAL_SPAN < span ? t0 + ARM_BASE__ZZPRIVATE_CAUSAL_SPAN : span;
    const uint64_t group = c->q_heads / c->kv_heads, hd = c->head_dim;
    const float scale = 1.0f / sqrtf((float)hd);
    float kf[ARM_BASE__ZZPRIVATE_CAUSAL_SPAN * ARM_BASE__ZZPRIVATE_ATTEND_WIDTH], qf[ARM_BASE__ZZPRIVATE_ATTEND_WIDTH];
    for (uint64_t pos = t0; pos < t1; ++pos) arm_base__zzprivate_attend_widen(kf + (pos - t0) * hd, c->k + (pos * c->kv_heads + g) * hd, hd);
    for (uint64_t r = t0 > c->first ? t0 - c->first : 0u; r < c->rows; ++r) {      /* the rows that reach position t0 */
        const uint64_t end = c->first + r + 1u < t1 ? c->first + r + 1u : t1;
        for (uint64_t hg = 0; hg < group; ++hg) {
            const uint64_t i = r * c->q_heads + g * group + hg;
            arm_base__zzprivate_attend_widen(qf, c->q + i * hd, hd);
            for (uint64_t pos = t0; pos < end; ++pos) c->p[i * span + pos] = arm_base__zzprivate_attend_dot(kf + (pos - t0) * hd, qf, hd) * scale;
        }
    }
}
static void arm_base__override_attention_causal_scores_grouped(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                               uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows) {
    if (!arm_base__zzprivate_attend_fit(q_heads, kv_heads, head_dim) || rows == 0u) {
        nn__attention__zzabi_launch_causal_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, first, rows); return;
    }
    (void)arm_base__zzpackage_start();
    arm_base__causal_call c = { p, q, k, q_heads, kv_heads, head_dim, first, rows };
    const uint64_t runs = (first + rows + ARM_BASE__ZZPRIVATE_CAUSAL_SPAN - 1u) / ARM_BASE__ZZPRIVATE_CAUSAL_SPAN;
    arm_base__zzpackage_run((uint32_t)(kv_heads * runs), arm_base__zzprivate_causal_scores, &c);
}

#endif /* SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_ATTENTION_CUH */
