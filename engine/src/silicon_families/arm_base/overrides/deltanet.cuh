#ifndef SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_DELTANET_CUH
#define SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_DELTANET_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include "../primitives/pool.cuh"   /* the threads the heads and channels are spread over */

/* ══ ⭐⭐ THE DELTANET ON NEON — nn's step, its rows, its gate named, and its conv ════════════════════════════════════════
 * nn's definitions (`kernels.cuh`). nn's step gives a value head a block and on this family's one lane walks its state a
 * column at a time — each column a stride of the state's rows, read twice. `MEASURED` (the 2B on the Razer Edge, the launch
 * profile, measurements/2026-11-11_nn175_arm_doors_razer_edge.md): `deltanet_step` 369 µs a decoded position, `steps` 46 ms
 * a layer of a 163-position prompt, `conv_steps` 8 ms. Here, as `x86_avx2`'s: a pool block a value head, its state 32
 * columns at a time — the column's sum against `k` down the rows, then its decay and update and the readout's sum down them
 * again, while the 32 columns are still in the core's cache — so the state is read from memory once a step. Each column's
 * arithmetic is nn's in nn's order, a multiply-add where nn's body's compiler makes one; the readout's norm adds its
 * squares in nn's order — so the outputs and the state are nn's bytes (`MEASURED` the door bench on the Razer Edge,
 * clang 21: every output worst 0 ulp, every float of the state bit-identical, the conv's too). */
typedef struct arm_base__step_call {
    float* S; const uint16_t* conved; const uint16_t* z; const uint16_t* beta; const uint16_t* g; const uint16_t* w;
    uint16_t* out; uint64_t k_heads, v_heads, d, rows; unsigned int gate; unsigned int* over;
} arm_base__step_call;
#define ARM_BASE__ZZPRIVATE_STEP_WIDTH 256u      /* the widest head */

/* Halves into floats, 16 a step, rounded once each as nn's verbs round them, and the fault word raised where a finite
 * value became an infinite half — as nn's conversion flags it (`nn__primitives__zzpackage_float_to_half`). */
static inline void arm_base__zzprivate_step_halves(uint16_t* o, const float32x4_t* f, unsigned int* over) {
    const float32x4_t inf = vdupq_n_f32(INFINITY);
    uint32x4_t hit = vdupq_n_u32(0u);
    for (int i = 0; i < 4; i += 2) {
        const float16x8_t h = vcvt_high_f16_f32(vcvt_f16_f32(f[i]), f[i + 1]);
        vst1q_u16(o + 4 * i, vreinterpretq_u16_f16(h));
        hit = vorrq_u32(hit, vandq_u32(vcltq_f32(vabsq_f32(f[i]), inf), vceqq_f32(vabsq_f32(vcvt_f32_f16(vget_low_f16(h))), inf)));
        hit = vorrq_u32(hit, vandq_u32(vcltq_f32(vabsq_f32(f[i + 1]), inf), vceqq_f32(vabsq_f32(vcvt_high_f32_f16(h)), inf)));
    }
    if (vmaxvq_u32(hit) != 0u) *over = NN__KERNELS__OVERFLOWED;
}

/* One value head's step: its state `Sh`, its `q`, `k`, `v` and `zh`, its write strength and decay, its readout gated by
 * silu(z) or σ(z) as `gate` says (`NN__DELTANET__GATE_*`). ⛳ Shared by the one-position door, the many-row one and the
 * one with its gate named, so a position decoded and the same position read as a prompt row are the same bytes. */
static inline void arm_base__zzprivate_step_head(float* Sh, const uint16_t* q, const uint16_t* k, const uint16_t* v, const uint16_t* zh,
                                                 uint16_t beta_h, uint16_t g_h, const uint16_t* w, uint16_t* oh, uint64_t d,
                                                 unsigned int gate, unsigned int* over) {
    float qf[ARM_BASE__ZZPRIVATE_STEP_WIDTH], kf[ARM_BASE__ZZPRIVATE_STEP_WIDTH];
    float qq = 0.0f, kk = 0.0f;
    for (uint64_t i = 0u; i < d; ++i) {
        qf[i] = nn__silicon__half_to_float(q[i]); kf[i] = nn__silicon__half_to_float(k[i]);
        qq = fmaf(qf[i], qf[i], qq); kk = fmaf(kf[i], kf[i], kk);
    }
    const float inv_q = 1.0f / sqrtf(qq + 1e-6f) / sqrtf((float)d), inv_k = 1.0f / sqrtf(kk + 1e-6f);
    const float decay = expf(nn__silicon__half_to_float(g_h)), b = nn__silicon__half_to_float(beta_h);
    float oo = 0.0f;
    for (uint64_t j = 0u; j < d; j += 32u) {
        /* the columns' sums against k: nn's `kv += S·k` down the rows */
        float32x4_t kv[8];
        for (int c = 0; c < 8; ++c) kv[c] = vdupq_n_f32(0.0f);
        for (uint64_t i = 0u; i < d; ++i) {
            const float* row = Sh + i * d + j;
            for (int c = 0; c < 8; ++c) kv[c] = vfmaq_n_f32(kv[c], vld1q_f32(row + 4 * c), kf[i]);
        }
        /* δ = (v − e^g·kv·inv_k)·β, the subtraction a multiply-add */
        float32x4_t delta[8], o[8];
        for (int c = 0; c < 8; ++c) {
            const float16x4_t vh = vreinterpret_f16_u16(vld1_u16(v + j + 4u * (uint64_t)c));
            delta[c] = vmulq_n_f32(vfmsq_f32(vcvt_f32_f16(vh), vmulq_n_f32(kv[c], decay), vdupq_n_f32(inv_k)), b);
            o[c] = vdupq_n_f32(0.0f);
        }
        /* S ← e^g·S + (k·inv_k)·δ, the decay's product a multiply-add onto the update's; the readout `o += S·q` */
        for (uint64_t i = 0u; i < d; ++i) {
            float* row = Sh + i * d + j;
            const float kn = kf[i] * inv_k;
            for (int c = 0; c < 8; ++c) {
                const float32x4_t s = vfmaq_n_f32(vmulq_n_f32(delta[c], kn), vld1q_f32(row + 4 * c), decay);
                vst1q_f32(row + 4 * c, s);
                o[c] = vfmaq_n_f32(o[c], s, qf[i]);
            }
        }
        for (int c = 0; c < 8; ++c) o[c] = vmulq_n_f32(o[c], inv_q);
        arm_base__zzprivate_step_halves(oh + j, o, over);
        arm_base__zzprivate_step_halves(oh + j + 16u, o + 4, over);
        for (uint64_t e = 0u; e < 32u; ++e) {
            const float r = nn__silicon__half_to_float(oh[j + e]);
            oo = fmaf(r, r, oo);
        }
    }
    const float scale = 1.0f / sqrtf(oo / (float)d + 1e-6f);
    for (uint64_t j = 0u; j < d; ++j) {
        const float zr = nn__silicon__half_to_float(zh[j]);
        const float zc = zr < -88.0f ? -88.0f : (zr > 88.0f ? 88.0f : zr);
        const float gz = gate == NN__DELTANET__GATE_SIGMOID ? 1.0f / (1.0f + expf(-zc)) : zc / (1.0f + expf(-zc));
        oh[j] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(oh[j]) * scale * nn__silicon__half_to_float(w[j]) * gz, over);
    }
}
/* Head `block` through each of the call's rows in order, its state carried from one to the next in place: row `t` reads
 * `conved`, `z`, `beta` and `g` at its own row and writes `out`'s. One row is the one-position step. */
static void arm_base__zzprivate_steps(void* a) {
    const arm_base__step_call* c = (const arm_base__step_call*)a;
    const uint64_t h = nn__silicon__block(), d = c->d, rep = c->v_heads / c->k_heads;
    const uint64_t ch = (2u * c->k_heads + c->v_heads) * d, vd = c->v_heads * d;
    for (uint64_t t = 0u; t < c->rows; ++t) {
        const uint16_t* r = c->conved + t * ch;
        arm_base__zzprivate_step_head(c->S + h * d * d, r + (h / rep) * d, r + c->k_heads * d + (h / rep) * d,
                                      r + 2u * c->k_heads * d + h * d, c->z + t * vd + h * d, c->beta[t * c->v_heads + h],
                                      c->g[t * c->v_heads + h], c->w, c->out + t * vd + h * d, d, c->gate, c->over);
    }
}
static inline bool arm_base__zzprivate_step_fit(uint64_t k_heads, uint64_t v_heads, uint64_t head_dim) {
    return head_dim != 0u && head_dim % 32u == 0u && head_dim <= ARM_BASE__ZZPRIVATE_STEP_WIDTH && k_heads != 0u && v_heads % k_heads == 0u;
}
static inline void arm_base__zzprivate_steps_run(arm_base__step_call* c) {
    (void)arm_base__zzpackage_start();
    arm_base__zzpackage_run((uint32_t)c->v_heads, arm_base__zzprivate_steps, c);
}
static void arm_base__override_deltanet_step(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g,
                                             const uint16_t* w, uint16_t* out, uint64_t k_heads, uint64_t v_heads, uint64_t head_dim,
                                             unsigned int* over) {
    if (!arm_base__zzprivate_step_fit(k_heads, v_heads, head_dim)) {
        nn__deltanet__zzabi_launch_step(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, over); return;
    }
    arm_base__step_call c = { S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, 1u, NN__DELTANET__GATE_SILU, over };
    arm_base__zzprivate_steps_run(&c);
}
/* ⭐ The prompt's rows: the step's own arithmetic a row at a time, so the two doors agree byte for byte. */
static void arm_base__override_deltanet_steps(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g,
                                              const uint16_t* w, uint16_t* out, uint64_t k_heads, uint64_t v_heads, uint64_t head_dim,
                                              uint64_t rows, unsigned int* over) {
    if (!arm_base__zzprivate_step_fit(k_heads, v_heads, head_dim)) {
        nn__deltanet__zzabi_launch_steps(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, over); return;
    }
    arm_base__step_call c = { S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, NN__DELTANET__GATE_SILU, over };
    arm_base__zzprivate_steps_run(&c);
}
/* ⭐ The same rows with the readout's gate named (Flash-Next's σ(z)): the one body, so silu(z) here is the plain door's
 * bytes. */
static void arm_base__override_deltanet_steps_gate(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta,
                                                   const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                   uint64_t v_heads, uint64_t head_dim, uint64_t rows, unsigned int gate,
                                                   unsigned int* over) {
    if (!arm_base__zzprivate_step_fit(k_heads, v_heads, head_dim)
        || (gate != NN__DELTANET__GATE_SILU && gate != NN__DELTANET__GATE_SIGMOID)) {
        nn__deltanet__zzabi_launch_steps_gate(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, gate, over); return;
    }
    arm_base__step_call c = { S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, gate, over };
    arm_base__zzprivate_steps_run(&c);
}

/* ── THE DELTANET'S CONV: a pool block a run of 64 channels, the positions in turn ──────────────────────────────────────
 * nn's definition: each channel's window of the last `NN__DELTANET__CONV_WINDOW` inputs and its taps, the output
 * `silu(Σ taps · window + tap · x)` as a half, the window slid by the input. nn's body strides the channels across the
 * blocks, a channel at a time on one lane. Here a block takes 64 channels: their taps and windows once as floats, then each
 * position's 64 inputs together, four channels a vector, each channel's sum in nn's order (each tap a multiply-add onto
 * it), libm's `expf` for the silu as nn's. ⛳ The one-position door is the same body at one row, so the two forms agree to
 * the bit. */
#define ARM_BASE__ZZPRIVATE_CONV_RUN 64u
typedef struct arm_base__conv_call {
    uint16_t* o; uint16_t* s; const uint16_t* w; const uint16_t* x; uint64_t ch, rows; unsigned int* over;
} arm_base__conv_call;
static void arm_base__zzprivate_conv(void* a) {
    const arm_base__conv_call* c = (const arm_base__conv_call*)a;
    const uint64_t c0 = (uint64_t)nn__silicon__block() * ARM_BASE__ZZPRIVATE_CONV_RUN;
    const uint64_t n = c->ch - c0 < ARM_BASE__ZZPRIVATE_CONV_RUN ? c->ch - c0 : ARM_BASE__ZZPRIVATE_CONV_RUN;
    float wt[NN__DELTANET__CONV_TAPS][ARM_BASE__ZZPRIVATE_CONV_RUN], st[NN__DELTANET__CONV_WINDOW][ARM_BASE__ZZPRIVATE_CONV_RUN];
    for (uint64_t j = 0; j < n; ++j) {
        for (unsigned t = 0u; t < NN__DELTANET__CONV_TAPS; ++t) wt[t][j] = nn__silicon__half_to_float(c->w[(c0 + j) * NN__DELTANET__CONV_TAPS + t]);
        for (unsigned t = 0u; t < NN__DELTANET__CONV_WINDOW; ++t) st[t][j] = nn__silicon__half_to_float(c->s[(c0 + j) * NN__DELTANET__CONV_WINDOW + t]);
    }
    float acc[ARM_BASE__ZZPRIVATE_CONV_RUN], xf[ARM_BASE__ZZPRIVATE_CONV_RUN];
    for (uint64_t r = 0; r < c->rows; ++r) {
        const uint16_t* xr = c->x + r * c->ch + c0;
        uint16_t* orow = c->o + r * c->ch + c0;
        uint64_t j = 0;
        for (; j + 4u <= n; j += 4u) {
            const float32x4_t xv = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(xr + j)));
            float32x4_t s = vfmaq_f32(vdupq_n_f32(0.0f), vld1q_f32(wt[0] + j), vld1q_f32(st[0] + j));
            for (unsigned t = 1u; t < NN__DELTANET__CONV_WINDOW; ++t) s = vfmaq_f32(s, vld1q_f32(wt[t] + j), vld1q_f32(st[t] + j));
            s = vfmaq_f32(s, vld1q_f32(wt[NN__DELTANET__CONV_WINDOW] + j), xv);
            vst1q_f32(acc + j, s); vst1q_f32(xf + j, xv);
        }
        for (; j < n; ++j) {
            xf[j] = nn__silicon__half_to_float(xr[j]);
            float s = 0.0f;
            for (unsigned t = 0u; t < NN__DELTANET__CONV_WINDOW; ++t) s = fmaf(wt[t][j], st[t][j], s);
            acc[j] = fmaf(wt[NN__DELTANET__CONV_WINDOW][j], xf[j], s);
        }
        for (j = 0; j < n; ++j) {
            orow[j] = nn__kernels__zzabi_to_half(acc[j] / (1.0f + expf(-acc[j])), c->over);
            for (unsigned t = 0u; t + 1u < NN__DELTANET__CONV_WINDOW; ++t) st[t][j] = st[t + 1u][j];
            st[NN__DELTANET__CONV_WINDOW - 1u][j] = xf[j];
        }
    }
    /* the window back as halves: every value in it was a half, so each is its own half again */
    for (uint64_t j = 0; j < n; ++j)
        for (unsigned t = 0u; t < NN__DELTANET__CONV_WINDOW; ++t) {
            union { __fp16 f; uint16_t u; } h;
            h.f = (__fp16)st[t][j];
            c->s[(c0 + j) * NN__DELTANET__CONV_WINDOW + t] = h.u;
        }
}
static void arm_base__override_deltanet_conv_steps(uint16_t* o, uint16_t* s, const uint16_t* w, const uint16_t* x, uint64_t ch,
                                                   uint64_t rows, unsigned int* over) {
    if (ch == 0u || rows == 0u) { nn__deltanet__zzabi_launch_conv_steps(o, s, w, x, ch, rows, over); return; }
    (void)arm_base__zzpackage_start();
    arm_base__conv_call c = { o, s, w, x, ch, rows, over };
    arm_base__zzpackage_run((uint32_t)((ch + ARM_BASE__ZZPRIVATE_CONV_RUN - 1u) / ARM_BASE__ZZPRIVATE_CONV_RUN), arm_base__zzprivate_conv, &c);
}
static void arm_base__override_deltanet_conv_step(uint16_t* o, uint16_t* s, const uint16_t* w, const uint16_t* x, uint64_t ch,
                                                  unsigned int* over) {
    arm_base__override_deltanet_conv_steps(o, s, w, x, ch, 1u, over);
}

#endif /* SILVANN__SILICON_FAMILIES_ARM_BASE_OVERRIDES_DELTANET_CUH */
