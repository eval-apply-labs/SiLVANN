#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_ATTENTION_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_ATTENTION_CUH
/* ══ attention's DOORS, RUN FASTER — amd_rocm6_wave64 ═════════════════════════════════════════════════════════════════
 * nn's attention doors (`nn/cpu/opcodes/attention__abi.cuh`), where this family covers them: the scores and the mix
 * over heads of 128 or 256, as one call or as the two residual passes, and two residuals' merge; the grouped scores, the
 * mix as halves and the causal mix over heads of 512 too. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ ATTENTION'S TWO RESIDUAL PASSES, SPREAD OVER THE BLOCK — nn's `attention_weights` gives a lane a position and sums
 *   its whole dot product alone, reading a key row with lanes two kilobytes apart; `attention_residual_mix` gives a lane
 *   an element of the head and walks every position in turn. `MEASURED` on the 27B at 1000 positions: 0.41 and 0.21 ms
 *   a layer, 20 ms of a 76 ms token, for 4 MB of keys and values a layer. Here, still a block a query head:
 *   · the scores: sixteen lanes a position, each holding `D / 16` of the head's query, so a wave reads four key rows
 *     whole a step, four steps in flight; the sixteen parts summed by shuffles. The softmax's largest score and sum
 *     are the family's fold, as nn's are, and each score is exponentiated by the lane that wrote it.
 *   · the mix: the block's four waves take every fourth position, each lane `D / 64` elements of the head, so a wave
 *     reads a value row whole; the four waves' sums added in shared memory once.
 *   The products and their fp32 sums are nn's; only the order they are added in differs. Heads of 128 or 256.
 * ⭐ AND THE ONE-CALL PAIR IS THE SAME TWO PASSES: `attention_scores` is the weights divided by their sum, by the lane
 *   that wrote each (`NORM`), and `attention_mix` the mix written as halves rather than into a residual (`HALVES`) —
 *   the 27B's fp16 cache ran nn's bodies, a block a head with a lane a position, and decoded 30 ms a token slower than
 *   the tiered cache over the same positions. */
template <uint32_t D, bool NORM>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_weights(float* p, float* res, const uint16_t* q, const uint16_t* k,
                                                                             uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    const uint32_t PER = D / 16u;                                         /* a lane's halves of a head: 8 or 16 */
    const uint32_t t = (uint32_t)nn__silicon__lane(), sub = t % 16u, slot = t / 16u;
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    const float scale = 1.0f / nn__silicon__sqrtf((float)D);
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const uint16_t* kh = k + (h / group) * D + sub * PER;
        float* ph = p + h * length;
        uint32_t qv[PER / 2u];
        const uint32_t* qw = (const uint32_t*)(q + h * D + sub * PER);
#pragma unroll
        for (uint32_t i = 0u; i < PER / 2u; ++i) qv[i] = qw[i];
        float top = -NN__KERNELS__INFINITY;
        for (uint64_t base = slot; base < length; base += 64u) {           /* four positions a lane group in flight */
            uint32_t kv[4][PER / 2u];
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                const uint64_t pos = base + 16u * u < length ? base + 16u * u : base;
                const uint32_t* kw = (const uint32_t*)(kh + pos * row);
#pragma unroll
                for (uint32_t i = 0u; i < PER / 2u; ++i) kv[u][i] = kw[i];
            }
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                float dot = 0.0f;
#pragma unroll
                for (uint32_t i = 0u; i < PER / 2u; ++i) dot = nn__silicon__dot2(qv[i], kv[u][i], dot);
                for (uint32_t off = 8u; off > 0u; off >>= 1) dot += __shfl_xor(dot, (int)off);
                const uint64_t pos = base + 16u * u;
                if (pos < length) {
                    const float score = dot * scale;
                    if (sub == 0u) ph[pos] = score;
                    top = score > top ? score : top;
                }
            }
        }
        top = nn__silicon__lanes_max(top);
        float part = 0.0f;                     /* each score exponentiated by the lane that wrote it */
        if (sub == 0u)
            for (uint64_t pos = slot; pos < length; pos += 16u) {
                const float e = nn__silicon__expf(ph[pos] - top);
                ph[pos] = e;
                part += e;
            }
        const float sum = nn__silicon__lanes_sum(part);
        if (NORM) {
            const float inv = 1.0f / sum;
            if (sub == 0u)
                for (uint64_t pos = slot; pos < length; pos += 16u) ph[pos] *= inv;
        } else if (t == 0u) {
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(D)] = top;
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 1ull] = sum;
        }
    }
}
/* One query head's mix over its first `length` positions, `vh` its kv head's first value: the block's four waves take
 * every fourth position, eight in flight, each lane `D / 64` elements of the head; the four waves' sums added in shared
 * memory once, into `res_h`'s accumulator or `o_h`'s halves. */
template <uint32_t D, bool HALVES>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_mix_head(float* res_h, uint16_t* o_h, const float* ph, const uint16_t* vh,
                                                                    uint64_t row, uint64_t length, unsigned int* over) {
    const uint32_t PER = D / 64u;                                         /* a lane's elements of a head: 2, 4 or 8 */
    __shared__ float waves[AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS][D];
    const uint32_t t = (uint32_t)nn__silicon__lane();
    const uint32_t lane = t % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, wave = t / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint16_t* vl = vh + lane * PER;
    float acc[PER];
#pragma unroll
    for (uint32_t e = 0u; e < PER; ++e) acc[e] = 0.0f;
    for (uint64_t base = wave; base < length; base += 8u * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {   /* eight positions in flight */
        uint32_t vv[8][PER / 2u];
        float wv[8];
#pragma unroll
        for (uint32_t u = 0u; u < 8u; ++u) {
            const uint64_t at = base + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * u;
            const bool real = at < length;
            const uint64_t pos = real ? at : base;
            const uint32_t* vw = (const uint32_t*)(vl + pos * row);
#pragma unroll
            for (uint32_t i = 0u; i < PER / 2u; ++i) vv[u][i] = vw[i];
            wv[u] = real ? ph[pos] : 0.0f;
        }
#pragma unroll
        for (uint32_t u = 0u; u < 8u; ++u)
#pragma unroll
            for (uint32_t i = 0u; i < PER / 2u; ++i) {
                acc[2u * i]      += wv[u] * nn__silicon__half_to_float((uint16_t)(vv[u][i] & 0xFFFFu));
                acc[2u * i + 1u] += wv[u] * nn__silicon__half_to_float((uint16_t)(vv[u][i] >> 16));
            }
    }
#pragma unroll
    for (uint32_t e = 0u; e < PER; ++e) waves[wave][lane * PER + e] = acc[e];
    __syncthreads();
    for (uint32_t e = t; e < D; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
        float sum = 0.0f;
#pragma unroll
        for (uint32_t w = 0u; w < AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS; ++w) sum += waves[w][e];
        if (HALVES) o_h[e] = nn__kernels__zzabi_to_half(sum, over);
        else res_h[2ull + e] = sum;
    }
    __syncthreads();
}
template <uint32_t D, bool HALVES>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_mix(float* res, uint16_t* o, const float* p, const uint16_t* v,
                                                                         uint64_t q_heads, uint64_t kv_heads, uint64_t length,
                                                                         unsigned int* over) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks())
        amd_rocm6_wave64__zzprivate_attention_mix_head<D, HALVES>(HALVES ? 0 : res + h * NN__ATTENTION__RESIDUAL_FLOATS(D),
                                                       HALVES ? o + h * D : 0, p + h * length, v + (h / group) * D, row,
                                                       length, over);
}
/* ⭐ THE CAUSAL MIX, A ROW AT A TIME AS THE ONE-QUERY MIX — `attention_causal_mix`, row `r` position `first + r` over the
 *   cache's first `first + r + 1` positions, its weights `p + (r·q_heads + h)·(first + rows)`: a block a (row, query head),
 *   each the same head's mix as `attention_mix` above, so a prompt's rows and its positions a token at a time add in one
 *   order — nn's causal mix sums a lane's element over every position in turn, which parted from this family's one-query
 *   mix in 5,119 of the 27B's chunk's 1,572,864 halves (`MEASURED` node03 2026-10-12, measurements/2026-10-12_nn66_node03.md
 *   ④, NN-72); as this, 0 halves apart and the 27B's causal mix 2.1x faster (`MEASURED` node03 2026-10-23,
 *   measurements/2026-10-23_nn72_causal_mix_node03.md). Heads of 128, 256 or 512. */
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_causal_mix(uint16_t* o, const float* p, const uint16_t* v,
                                                                                uint64_t q_heads, uint64_t kv_heads, uint64_t first,
                                                                                uint64_t rows, unsigned int* over) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D, span = first + rows;
    for (uint64_t i = nn__silicon__block(); i < rows * q_heads; i += nn__silicon__blocks()) {
        const uint64_t r = i / q_heads, h = i - r * q_heads;
        amd_rocm6_wave64__zzprivate_attention_mix_head<D, true>(0, o + i * D, p + i * span, v + (h / group) * D, row, first + r + 1ull, over);
    }
}
/* ⭐ A KV HEAD'S QUERY HEADS' SCORES AND NOTHING AFTER — `attention_scores_grouped`, the softmax left to the caller. nn's
 *   body gives a lane a query head and eight positions and walks the whole head alone: at 64 heads on one kv head of 512
 *   over 640 positions that is 20 blocks and 578 us a call (`MEASURED` with rocprof, DeepSeek's chunk on node03,
 *   measurements/2026-10-21_deepseek_attention_kernels_node03.md). Here, as `attention_weights` above, a block a query
 *   head and sixteen lanes a position, each holding `D / 16` of the query, four positions a lane group in flight; the
 *   score `dot · 1/√D` into `p`. The products are nn's; the sixteen parts are summed by shuffles. */
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_raw(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                         uint64_t kv_heads, uint64_t length) {
    const uint32_t PER = D / 16u;                                         /* a lane's halves of a head: 8, 16 or 32 */
    const uint32_t t = (uint32_t)nn__silicon__lane(), sub = t % 16u, slot = t / 16u;
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    const float scale = 1.0f / nn__silicon__sqrtf((float)D);
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const uint16_t* kh = k + (h / group) * D + sub * PER;
        float* ph = p + h * length;
        uint32_t qv[PER / 2u];
        const uint32_t* qw = (const uint32_t*)(q + h * D + sub * PER);
#pragma unroll
        for (uint32_t i = 0u; i < PER / 2u; ++i) qv[i] = qw[i];
        for (uint64_t base = slot; base < length; base += 64u) {           /* four positions a lane group in flight */
            uint32_t kv[4][PER / 2u];
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                const uint64_t pos = base + 16u * u < length ? base + 16u * u : base;
                const uint32_t* kw = (const uint32_t*)(kh + pos * row);
#pragma unroll
                for (uint32_t i = 0u; i < PER / 2u; ++i) kv[u][i] = kw[i];
            }
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                float dot = 0.0f;
#pragma unroll
                for (uint32_t i = 0u; i < PER / 2u; ++i) dot = nn__silicon__dot2(qv[i], kv[u][i], dot);
                for (uint32_t off = 8u; off > 0u; off >>= 1) dot += __shfl_xor(dot, (int)off);
                const uint64_t pos = base + 16u * u;
                if (pos < length && sub == 0u) ph[pos] = dot * scale;
            }
        }
    }
}
/* The entry points, written out — a head of 128 and of 256 for each, and of 512 for the grouped scores, the mix as
 * halves and the causal mix — so every kernel name has a definition a reader (and the C-subset gate) can find. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_raw_128(float* p, const uint16_t* q, const uint16_t* k,
    uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_raw<128u>(p, q, k, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_raw_256(float* p, const uint16_t* q, const uint16_t* k,
    uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_raw<256u>(p, q, k, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_raw_512(float* p, const uint16_t* q, const uint16_t* k,
    uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_raw<512u>(p, q, k, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_causal_mix_128(uint16_t* o, const float* p, const uint16_t* v,
    uint64_t q_heads, uint64_t kv_heads, uint64_t first, uint64_t rows, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_causal_mix<128u>(o, p, v, q_heads, kv_heads, first, rows, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_causal_mix_256(uint16_t* o, const float* p, const uint16_t* v,
    uint64_t q_heads, uint64_t kv_heads, uint64_t first, uint64_t rows, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_causal_mix<256u>(o, p, v, q_heads, kv_heads, first, rows, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_causal_mix_512(uint16_t* o, const float* p, const uint16_t* v,
    uint64_t q_heads, uint64_t kv_heads, uint64_t first, uint64_t rows, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_causal_mix<512u>(o, p, v, q_heads, kv_heads, first, rows, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_halves_512(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads,
    uint64_t kv_heads, uint64_t length, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_mix<512u, true>(0, o, p, v, q_heads, kv_heads, length, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_128(float* p, float* res, const uint16_t* q, const uint16_t* k,
    uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_weights<128u, false>(p, res, q, k, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_scores_128(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
    uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_weights<128u, true>(p, 0, q, k, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_128(float* res, const float* p, const uint16_t* v, uint64_t q_heads,
    uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_mix<128u, false>(res, 0, p, v, q_heads, kv_heads, length, 0);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_halves_128(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads,
    uint64_t kv_heads, uint64_t length, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_mix<128u, true>(0, o, p, v, q_heads, kv_heads, length, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_256(float* p, float* res, const uint16_t* q, const uint16_t* k,
    uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_weights<256u, false>(p, res, q, k, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_scores_256(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
    uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_weights<256u, true>(p, 0, q, k, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_256(float* res, const float* p, const uint16_t* v, uint64_t q_heads,
    uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_mix<256u, false>(res, 0, p, v, q_heads, kv_heads, length, 0);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_halves_256(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads,
    uint64_t kv_heads, uint64_t length, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_mix<256u, true>(0, o, p, v, q_heads, kv_heads, length, over);
}
/* Heads of 128 or 256, a whole group of query heads a key head, `q` and `k` / `v` on 4 bytes; anything else is nn's. */
static inline bool amd_rocm6_wave64__zzprivate_attention_fit(const void* a, const void* b, uint64_t q_heads, uint64_t kv_heads,
                                                             uint64_t head_dim) {
    return (head_dim == 128u || head_dim == 256u) && kv_heads != 0u && q_heads % kv_heads == 0u && q_heads != 0u
        && ((uint64_t)(uintptr_t)a) % 4u == 0u && ((uint64_t)(uintptr_t)b) % 4u == 0u;
}
/* ...and the grouped scores and the mix as halves, which take heads of 512 as well — a latent attention's 64 query heads
 * on one kv head of 512 (GLM's, DeepSeek's). */
static inline bool amd_rocm6_wave64__zzprivate_attention_fit_wide(const void* a, const void* b, uint64_t q_heads, uint64_t kv_heads,
                                                                  uint64_t head_dim) {
    return (head_dim == 128u || head_dim == 256u || head_dim == 512u) && kv_heads != 0u && q_heads % kv_heads == 0u && q_heads != 0u
        && ((uint64_t)(uintptr_t)a) % 4u == 0u && ((uint64_t)(uintptr_t)b) % 4u == 0u;
}
static void amd_rocm6_wave64__override_attention_weights(float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                         uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit(q, k, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_weights(p, res, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_weights_128
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_weights_256;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &res, &q, &k, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof p, sizeof res, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_weights", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 7u);
}
static void amd_rocm6_wave64__override_attention_residual_mix(float* res, const float* p, const uint16_t* v, uint64_t q_heads,
                                                              uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit(v, v, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_residual_mix(res, p, v, q_heads, kv_heads, head_dim, length);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_128
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_256;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &res, &p, &v, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_mix", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 6u);
}

static void amd_rocm6_wave64__override_attention_scores(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                        uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit(q, k, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_scores(p, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_scores_128
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_scores_256;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &q, &k, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_scores", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 6u);
}
static void amd_rocm6_wave64__override_attention_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads,
                                                     uint64_t kv_heads, uint64_t head_dim, uint64_t length, unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit_wide(v, v, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_mix(o, p, v, q_heads, kv_heads, head_dim, length, over);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_halves_128
                       : head_dim == 256u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_halves_256
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_halves_512;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &p, &v, &q_heads, &kv_heads, &length, &over };
    const size_t sizes[] = { sizeof o, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof length, sizeof over };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_mix", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 7u);
}
static void amd_rocm6_wave64__override_attention_causal_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads,
                                                            uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows,
                                                            unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit_wide(v, v, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_causal_mix(o, p, v, q_heads, kv_heads, head_dim, first, rows, over);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_causal_mix_128
                       : head_dim == 256u ? (const void*)amd_rocm6_wave64__zzprivate_attention_causal_mix_256
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_causal_mix_512;
    const uint64_t n = rows * q_heads;
    const uint32_t blocks = n < NN__KERNELS__BLOCKS_MAX ? (uint32_t)n : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &p, &v, &q_heads, &kv_heads, &first, &rows, &over };
    const size_t sizes[] = { sizeof o, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof first, sizeof rows, sizeof over };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_causal_mix", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 8u);
}
static void amd_rocm6_wave64__override_attention_scores_grouped(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit_wide(q, k, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_raw_128
                       : head_dim == 256u ? (const void*)amd_rocm6_wave64__zzprivate_attention_raw_256
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_raw_512;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &q, &k, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_raw", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 6u);
}

/* ⭐ TWO RESIDUALS INTO ONE, A WAVE A HEAD — what the tiered cache does once a tier is in: nn's `attention_merge` gives a
 *   lane a whole head and walks its `d + 2` floats alone, one lane of a block busy for each of 24 heads. Here a wave takes
 *   a head and each lane every 64th element of it. Every lane reads the head's two `m` and `l` before any writes, and the
 *   first lane writes the merged pair after its elements, so `into` may be `a` or `b` as nn allows; a head is one wave's
 *   alone. Each value is nn's own expression, so the output is nn's generic kernel's to the bit (`MEASURED`
 *   on gfx906; gfx90a unverified). `MEASURED` (rocprof over the bench): 24 heads of 256, 68 -> 4.3 us. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_merge(float* into, const float* a, const float* b,
                                                                                            uint64_t q_heads, uint64_t head_dim) {
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t w = NN__ATTENTION__RESIDUAL_FLOATS(head_dim);
    for (uint64_t h = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; h < q_heads;
         h += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        const float ma = a[h * w], la = a[h * w + 1ull], mb = b[h * w], lb = b[h * w + 1ull];
        const float m = ma > mb ? ma : mb;
        const float fa = nn__silicon__expf(ma - m), fb = nn__silicon__expf(mb - m);
        for (uint64_t e = lane; e < head_dim; e += AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
            into[h * w + 2ull + e] = a[h * w + 2ull + e] * fa + b[h * w + 2ull + e] * fb;
        if (lane == 0u) {
            into[h * w] = m;
            into[h * w + 1ull] = la * fa + lb * fb;
        }
    }
}
static void amd_rocm6_wave64__override_attention_merge(float* into, const float* a, const float* b, uint64_t q_heads, uint64_t head_dim) {
    if (q_heads == 0u || head_dim == 0u) {
        nn__attention__zzabi_launch_merge(into, a, b, q_heads, head_dim);
        return;
    }
    const uint64_t want = (q_heads + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &into, &a, &b, &q_heads, &head_dim };
    const size_t sizes[] = { sizeof into, sizeof a, sizeof b, sizeof q_heads, sizeof head_dim };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_merge, "amd_rocm6_wave64__zzprivate_merge", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 5u);
}

/* ⭐ THE TWO RESIDUAL PASSES OVER TURBOQUANT ROWS — the tiered cache's warm keys and values (8 bits) and its cold ones
 *   (4 bits). nn's `attention_weights_tq` gives a lane a position and decodes its whole key row alone, an element at a
 *   time through the level table; `attention_residual_mix_tq` gives a lane an element of the head and walks every
 *   position in turn. `MEASURED` (node03, the 35B at 6,033 positions, 1,681 of them warm, test/src_runtime_bench.py):
 *   decode 26 tokens/s with the tiered cache, 68 with every position whole, and 68 again with the hot ring widened past
 *   the prompt — the warm tier's passes on nn's bodies were 23 ms of a 38 ms token. Here the shapes of the fp16 passes
 *   above, the codes' levels read from the width's table copied into shared memory once a block:
 *   · the weights: sixteen lanes a position, each holding `D / 16` of the query as floats and reading its `D / 16`
 *     codes of a key row as whole words, so a wave reads four key rows whole a step; the row's scale multiplies the
 *     sixteen lanes' sum once.
 *   · the mix: the block's four waves take every fourth position, each lane `D / 64` elements of the head, so a wave
 *     reads a value row whole; a position's weight is multiplied by its row's scale once, and the four waves' sums are
 *     added in shared memory once.
 *   nn multiplies each level by the scale before the query or the weight; here the scale is taken out of the sum, so the
 *   products are added in another order and rounded once less — within the bench's tolerance of nn's host body. Heads
 *   of 128 or 256, codes of 4 or 8 bits. */
template <uint32_t D, uint32_t BITS>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_weights_tq(float* p, float* res, const uint16_t* q,
                                                                                const uint8_t* codes, const uint16_t* scales,
                                                                                uint64_t q_heads, uint64_t kv_heads,
                                                                                uint64_t length) {
    constexpr uint32_t PER = D / 16u;                                     /* a lane's elements of a head: 8 or 16 */
    constexpr uint32_t WORDS = PER * BITS / 32u;                          /* its codes of a row, as words: 1 to 4 */
    constexpr uint32_t CODES = 32u / BITS, LEVELS = 1u << BITS, ROW = D * BITS / 8u;
    __shared__ float levels[LEVELS];
    const uint32_t t = (uint32_t)nn__silicon__lane(), sub = t % 16u, slot = t / 16u;
    const uint16_t* table = nn__turboquant__zzabi_levels(BITS);
    for (uint32_t i = t; i < LEVELS; i += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        levels[i] = nn__silicon__half_to_float(table[i]);
    __syncthreads();                                                      /* the table, before any lane reads it */
    const uint64_t group = q_heads / kv_heads;
    const float inv = 1.0f / nn__silicon__sqrtf((float)D);
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const uint64_t g = h / group;
        float qf[PER];
#pragma unroll
        for (uint32_t i = 0u; i < PER; ++i) qf[i] = nn__silicon__half_to_float(q[h * D + sub * PER + i]);
        float* ph = p + h * length;
        float top = -NN__KERNELS__INFINITY;
        for (uint64_t base = slot; base < length; base += 64u) {           /* four positions a lane group in flight */
            uint32_t kw[4][WORDS];
            float sc[4];
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                const uint64_t pos = base + 16u * u < length ? base + 16u * u : base;
                const uint64_t row = pos * kv_heads + g;
                const uint32_t* kr = (const uint32_t*)(codes + row * ROW + sub * (PER * BITS / 8u));
#pragma unroll
                for (uint32_t i = 0u; i < WORDS; ++i) kw[u][i] = kr[i];
                sc[u] = nn__silicon__half_to_float(scales[row]);
            }
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                float dot = 0.0f;
#pragma unroll
                for (uint32_t i = 0u; i < WORDS; ++i)
#pragma unroll
                    for (uint32_t c = 0u; c < CODES; ++c)
                        dot += qf[i * CODES + c] * levels[(kw[u][i] >> (BITS * c)) & (LEVELS - 1u)];
                dot += amd_rocm6_wave64__zzpackage_xor(dot, 8u);
                dot += amd_rocm6_wave64__zzpackage_xor(dot, 4u);
                dot += amd_rocm6_wave64__zzpackage_xor(dot, 2u);
                dot += amd_rocm6_wave64__zzpackage_xor(dot, 1u);
                const uint64_t pos = base + 16u * u;
                if (pos < length) {
                    const float score = dot * sc[u] * inv;
                    if (sub == 0u) ph[pos] = score;
                    top = score > top ? score : top;
                }
            }
        }
        top = nn__silicon__lanes_max(top);
        float part = 0.0f;                     /* each score exponentiated by the lane that wrote it */
        if (sub == 0u)
            for (uint64_t pos = slot; pos < length; pos += 16u) {
                const float e = nn__silicon__expf(ph[pos] - top);
                ph[pos] = e;
                part += e;
            }
        const float sum = nn__silicon__lanes_sum(part);
        if (t == 0u) {
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(D)] = top;
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 1ull] = sum;
        }
    }
}
template <uint32_t D, uint32_t BITS>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_mix_tq(float* res, const float* p, const uint8_t* codes,
                                                                            const uint16_t* scales, uint64_t q_heads,
                                                                            uint64_t kv_heads, uint64_t length) {
    constexpr uint32_t PER = D / 64u;                                     /* a lane's elements of a head: 2 or 4 */
    constexpr uint32_t BYTES = PER * BITS / 8u;                           /* its codes of a row: 1, 2 or 4 bytes */
    constexpr uint32_t LEVELS = 1u << BITS, ROW = D * BITS / 8u;
    __shared__ float levels[LEVELS];
    __shared__ float waves[AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS][D];
    const uint32_t t = (uint32_t)nn__silicon__lane();
    const uint32_t lane = t % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, wave = t / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint16_t* table = nn__turboquant__zzabi_levels(BITS);
    for (uint32_t i = t; i < LEVELS; i += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        levels[i] = nn__silicon__half_to_float(table[i]);
    __syncthreads();                                                      /* the table, before any lane reads it */
    const uint64_t group = q_heads / kv_heads;
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const float* ph = p + h * length;
        const uint64_t g = h / group;
        float acc[PER];
#pragma unroll
        for (uint32_t e = 0u; e < PER; ++e) acc[e] = 0.0f;
        for (uint64_t base = wave; base < length; base += 8u * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {   /* eight positions in flight */
            uint32_t cw[8];
            float wv[8];
#pragma unroll
            for (uint32_t u = 0u; u < 8u; ++u) {
                const uint64_t at = base + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * u;
                const bool real = at < length;
                const uint64_t pos = real ? at : base;
                const uint64_t row = pos * kv_heads + g;
                const uint8_t* vr = codes + row * ROW + lane * BYTES;
                cw[u] = BYTES == 4u ? *(const uint32_t*)vr : BYTES == 2u ? (uint32_t)*(const uint16_t*)vr : (uint32_t)*vr;
                wv[u] = real ? ph[pos] * nn__silicon__half_to_float(scales[row]) : 0.0f;
            }
#pragma unroll
            for (uint32_t u = 0u; u < 8u; ++u)
#pragma unroll
                for (uint32_t e = 0u; e < PER; ++e)
                    acc[e] += wv[u] * levels[(cw[u] >> (BITS * e)) & (LEVELS - 1u)];
        }
#pragma unroll
        for (uint32_t e = 0u; e < PER; ++e) waves[wave][lane * PER + e] = acc[e];
        __syncthreads();
        for (uint32_t e = t; e < D; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
            float sum = 0.0f;
#pragma unroll
            for (uint32_t w = 0u; w < AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS; ++w) sum += waves[w][e];
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 2ull + e] = sum;
        }
        __syncthreads();
    }
}
/* The eight entry points, written out — heads of 128 and of 256, codes of 4 and of 8 bits, for each pass. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_tq_128_4(float* p, float* res, const uint16_t* q,
    const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_weights_tq<128u, 4u>(p, res, q, codes, scales, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_tq_128_8(float* p, float* res, const uint16_t* q,
    const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_weights_tq<128u, 8u>(p, res, q, codes, scales, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_tq_256_4(float* p, float* res, const uint16_t* q,
    const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_weights_tq<256u, 4u>(p, res, q, codes, scales, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_tq_256_8(float* p, float* res, const uint16_t* q,
    const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_weights_tq<256u, 8u>(p, res, q, codes, scales, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_tq_128_4(float* res, const float* p, const uint8_t* codes,
    const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_mix_tq<128u, 4u>(res, p, codes, scales, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_tq_128_8(float* res, const float* p, const uint8_t* codes,
    const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_mix_tq<128u, 8u>(res, p, codes, scales, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_tq_256_4(float* res, const float* p, const uint8_t* codes,
    const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_mix_tq<256u, 4u>(res, p, codes, scales, q_heads, kv_heads, length);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_tq_256_8(float* res, const float* p, const uint8_t* codes,
    const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    amd_rocm6_wave64__zzprivate_attention_mix_tq<256u, 8u>(res, p, codes, scales, q_heads, kv_heads, length);
}
/* Heads of 128 or 256, codes of 4 or 8 bits, a whole group of query heads a key head, the codes on 4 bytes; anything
 * else is nn's. */
static inline bool amd_rocm6_wave64__zzprivate_attention_tq_fit(const uint8_t* codes, uint64_t q_heads, uint64_t kv_heads,
                                                                uint64_t head_dim, uint64_t d) {
    return (head_dim == 128u || head_dim == 256u) && (d == 4u || d == 8u) && kv_heads != 0u && q_heads != 0u
        && q_heads % kv_heads == 0u && ((uint64_t)(uintptr_t)codes) % 4u == 0u;
}
static void amd_rocm6_wave64__override_attention_weights_tq(float* p, float* res, const uint16_t* q, const uint8_t* codes,
                                                            const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads,
                                                            uint64_t head_dim, uint64_t length, uint64_t d) {
    if (!amd_rocm6_wave64__zzprivate_attention_tq_fit(codes, q_heads, kv_heads, head_dim, d)) {
        nn__attention__zzabi_launch_weights_tq(p, res, q, codes, scales, q_heads, kv_heads, head_dim, length, d);
        return;
    }
    const void* kernel = head_dim == 128u
        ? (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_128_4
                   : (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_128_8)
        : (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_256_4
                   : (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_256_8);
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &res, &q, &codes, &scales, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof p, sizeof res, sizeof q, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads,
                             sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_weights_tq", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 8u);
}
static void amd_rocm6_wave64__override_attention_residual_mix_tq(float* res, const float* p, const uint8_t* codes,
                                                                 const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads,
                                                                 uint64_t head_dim, uint64_t length, uint64_t d) {
    if (!amd_rocm6_wave64__zzprivate_attention_tq_fit(codes, q_heads, kv_heads, head_dim, d)) {
        nn__attention__zzabi_launch_residual_mix_tq(res, p, codes, scales, q_heads, kv_heads, head_dim, length, d);
        return;
    }
    const void* kernel = head_dim == 128u
        ? (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_tq_128_4
                   : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_tq_128_8)
        : (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_tq_256_4
                   : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_tq_256_8);
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &res, &p, &codes, &scales, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads, sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_mix_tq", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 7u);
}

/* ⭐ ATTENTION OVER A SET — a picture's patches, every row to every row. nn's `attention_set` gives a block a (row, head)
 *   and its lanes the keys, each lane carrying the head's sums in an array it indexes at run time — on gfx906 an array
 *   in scratch memory — and every block reads every key and value row of its head. `MEASURED` (node03, v0.7.0-rc3, the
 *   27B's tower on an MI50, test/src_runtime_vision_time.py): 1,024 / 4,096 / 16,384 patches in 5.6 / 72 / 1,084 s.
 *   Here a block takes 128 rows of one head, two lanes a row, each lane half the head: its half of the query as half
 *   pairs and of the sums as floats, in registers (the head's width a template argument, so every index is a constant;
 *   a whole head a lane held 144 registers at a width of 72, one wave a SIMD). The block's keys and values come
 *   through shared memory 64 rows at a time: the four waves stage a tile, wait for it, read it, and wait again before
 *   the next (STAGING). A lane scores 16 keys at a time — its half's two-half dots, the pair's two halves added across
 *   the two lanes — folds their largest into its running one and rescales its sums once, and mixes the values two keys
 *   at a time: their weights rounded to halves and paired, the tile's values laid out as pairs of the two keys' same
 *   element, so each element's sum is one more two-half dot. A tile of 64 keys runs without a key's bound in sight.
 *   nn sums its products in another order and keeps its weights as floats — within the bench's tolerance of nn's host
 *   body, as the CUDA family's set attention, which rounds its weights to halves too. Heads of whole products of 8
 *   columns, 128 at most; anything else is nn's. `MEASURED` (the same test, with the 16-bit tiles of expert.cuh and the
 *   ladder of hadamard.cuh): 1,024 / 4,096 / 16,384 patches in 0.27 / 1.15 / 8.3 s, this kernel about 5.7 s of the
 *   last. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_SET_TILE  64u      /* keys a stage */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_SET_SUB   8u       /* keys a lane scores at once */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_SET_ROWS  128u     /* rows a block, two lanes each */
/* 2 to the `x`: the card's one instruction (`v_exp_f32`). The scores are scaled by log2(e) as well, so the softmax's
 * exponentials are base 2 and every one is of a number at most 0 — where the instruction is exact to a unit or two. */
static __device__ inline float amd_rocm6_wave64__zzprivate_exp2(float x) { return __builtin_amdgcn_exp2f(x); }
/* A staged tile's keys `0 .. n` for one lane's half of the head: `KW` words of the key a key (its half's pairs, padded
 * to whole loads), `HD` elements of the values a pair of keys. `N` the tile's keys when it is whole, else 0 and `n`. */
template <uint32_t D, uint32_t N>
static __device__ __forceinline__ void amd_rocm6_wave64__zzprivate_set_tile(float* acc, float& m, float& l, const uint32_t* q,
                                                                           const uint32_t* keys, const uint32_t* values,
                                                                           uint32_t n, float scale) {
    constexpr uint32_t HP = D / 4u, KW = (HP + 3u) & ~3u, HD = D / 2u, SUB = AMD_ROCM6_WAVE64__ZZPRIVATE_SET_SUB;
    const uint32_t count = N != 0u ? N : n;
#pragma unroll 1
    for (uint32_t s0 = 0u; s0 < count; s0 += SUB) {                     /* a sub-tile's registers at a time */
        float sc[SUB];
        float top = m;
#pragma unroll
        for (uint32_t u = 0u; u < SUB; ++u) {
            float dot = 0.0f;
            const uint32_t* kr = keys + (s0 + u) * 2u * KW;
#pragma unroll
            for (uint32_t w = 0u; w < HP; ++w) dot = nn__silicon__dot2(q[w], kr[w], dot);
            dot += amd_rocm6_wave64__zzpackage_xor(dot, 1u);           /* the other half's, from the pair's other lane */
            sc[u] = N != 0u || s0 + u < count ? dot * scale : -NN__KERNELS__INFINITY;
            top = sc[u] > top ? sc[u] : top;
        }
        const float f = amd_rocm6_wave64__zzprivate_exp2(m - top);      /* the sums so far, at the new largest */
        l *= f;
#pragma unroll
        for (uint32_t d = 0u; d < HD; ++d) acc[d] *= f;
        m = top;
#pragma unroll
        for (uint32_t u = 0u; u < SUB; u += 2u) {
            const float p0 = N != 0u || s0 + u < count ? amd_rocm6_wave64__zzprivate_exp2(sc[u] - m) : 0.0f;
            const float p1 = N != 0u || s0 + u + 1u < count ? amd_rocm6_wave64__zzprivate_exp2(sc[u + 1u] - m) : 0.0f;
            const _Float16 h0 = (_Float16)p0, h1 = (_Float16)p1;
            l += (float)h0 + (float)h1;                                 /* the weights the values are mixed with */
            const uint32_t pw = (uint32_t)__builtin_bit_cast(uint16_t, h0) | ((uint32_t)__builtin_bit_cast(uint16_t, h1) << 16);
            const uint32_t* vr = values + ((s0 + u) / 2u) * D;
#pragma unroll
            for (uint32_t d = 0u; d < HD; ++d) acc[d] = nn__silicon__dot2(pw, vr[d], acc[d]);
        }
    }
}
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_set(uint16_t* o, const uint16_t* qkv, uint64_t rows,
                                                                        uint64_t heads, unsigned int* over) {
    constexpr uint32_t HP = D / 4u, KW = (HP + 3u) & ~3u, HD = D / 2u;  /* a lane's half: pairs, padded words, elements */
    constexpr uint32_t TILE = AMD_ROCM6_WAVE64__ZZPRIVATE_SET_TILE, ROWS = AMD_ROCM6_WAVE64__ZZPRIVATE_SET_ROWS;
    constexpr uint32_t LANES = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    __shared__ uint32_t keys[TILE * 2u * KW];                             /* key j's half h, pair w at (2j + h)·KW + w */
    __shared__ uint32_t values[(TILE / 2u) * D];                          /* keys 2i and 2i+1's element d at i·D + d */
    const uint64_t H = heads * D, stride = 3ull * H, groups = (rows + ROWS - 1u) / ROWS;
    const uint32_t t = (uint32_t)nn__silicon__lane(), h = t & 1u;
    const float scale = 1.4426950408889634f / nn__silicon__sqrtf((float)D);   /* log2(e) / sqrt(D): base-2 scores */
    for (uint64_t item = nn__silicon__block(); item < heads * groups; item += nn__silicon__blocks()) {
        const uint64_t e = item / groups, r = (item - e * groups) * ROWS + t / 2u;
        const bool mine = r < rows;
        uint32_t q[HP];
        const uint32_t* qr = (const uint32_t*)(qkv + (mine ? r : 0ull) * stride + e * D + h * HD);
#pragma unroll
        for (uint32_t w = 0u; w < HP; ++w) q[w] = mine ? qr[w] : 0u;
        float acc[HD];
#pragma unroll
        for (uint32_t d = 0u; d < HD; ++d) acc[d] = 0.0f;
        float m = -NN__KERNELS__INFINITY, l = 0.0f;
        for (uint64_t base = 0ull; base < rows; base += TILE) {
            const uint32_t n = rows - base < TILE ? (uint32_t)(rows - base) : TILE;
            for (uint32_t i = t; i < TILE * 2u * KW; i += LANES) {        /* the tile's keys, each half on whole loads */
                const uint32_t j = i / (2u * KW), rest = i - j * 2u * KW, hh = rest / KW, w = rest - hh * KW;
                keys[i] = j < n && w < HP ? ((const uint32_t*)(qkv + (base + j) * stride + H + e * D + hh * HD))[w] : 0u;
            }
            for (uint32_t i = t; i < (TILE / 2u) * (D / 2u); i += LANES) { /* its values, two keys' same element a word */
                const uint32_t jp = i / (D / 2u), w = i - jp * (D / 2u), j = 2u * jp;
                const uint32_t a = j < n ? ((const uint32_t*)(qkv + (base + j) * stride + 2ull * H + e * D))[w] : 0u;
                const uint32_t b = j + 1u < n ? ((const uint32_t*)(qkv + (base + j + 1u) * stride + 2ull * H + e * D))[w] : 0u;
                values[jp * D + 2u * w] = (a & 0xFFFFu) | (b << 16);
                values[jp * D + 2u * w + 1u] = (a >> 16) | (b & 0xFFFF0000u);
            }
            __syncthreads();                                              /* the tile, before any lane reads it */
            if (n == TILE)
                amd_rocm6_wave64__zzprivate_set_tile<D, TILE>(acc, m, l, q, keys + h * KW, values + h * HD, n, scale);
            else
                amd_rocm6_wave64__zzprivate_set_tile<D, 0u>(acc, m, l, q, keys + h * KW, values + h * HD, n, scale);
            __syncthreads();                                              /* every lane done with the tile */
        }
        if (mine) {
            uint16_t* out = o + r * H + e * D + h * HD;
#pragma unroll
            for (uint32_t d = 0u; d < HD; ++d) out[d] = amd_rocm6_wave64__zzpackage_to_half(acc[d] / l, over);
        }
    }
}
/* The block's bound, and two waves a SIMD: the registers the compiler may take, 128 — the most two waves leave each
 * (left to itself it took 177, one wave a SIMD). `MEASURED` (node03, the 27B's tower at 16,384 patches, 16 keys a step
 * then): two waves 12.4 s, three 16.4 s — 84 registers, a lane's sums partly in scratch. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_SET_BOUNDS __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(2)))
template <uint32_t D>
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_SET_BOUNDS void amd_rocm6_wave64__zzprivate_attention_set_head(uint16_t* o, const uint16_t* qkv,
                                                                                                             uint64_t rows, uint64_t heads,
                                                                                                             unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_set<D>(o, qkv, rows, heads, over);
}
/* What it takes: a head of whole products of 8 columns, 128 at most, `qkv` on four bytes (its pairs' loads). */
static void amd_rocm6_wave64__override_attention_set(uint16_t* o, const uint16_t* qkv, uint64_t rows, uint64_t heads,
                                                     uint64_t head_dim, unsigned int* over) {
    static const void* const kernels[16] = {           /* a head's width in 8s, less one */
        (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<8u>,   (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<16u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<24u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<32u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<40u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<48u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<56u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<64u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<72u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<80u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<88u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<96u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<104u>, (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<112u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<120u>, (const void*)amd_rocm6_wave64__zzprivate_attention_set_head<128u> };
    if (rows == 0u || heads == 0u || head_dim == 0u || head_dim % 8u != 0u || head_dim > 128u
     || ((uint64_t)(uintptr_t)qkv) % 4u != 0u || ((uint64_t)(uintptr_t)o) % 2u != 0u) {
        nn__attention__zzabi_launch_set(o, qkv, rows, heads, head_dim, over);
        return;
    }
    const uint64_t lanes = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t items = heads * ((rows + AMD_ROCM6_WAVE64__ZZPRIVATE_SET_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_SET_ROWS);
    const uint32_t blocks = items < NN__KERNELS__BLOCKS_MAX ? (uint32_t)items : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &qkv, &rows, &heads, &over };
    const size_t sizes[] = { sizeof o, sizeof qkv, sizeof rows, sizeof heads, sizeof over };
    nn__silicon__launch(kernels[head_dim / 8u - 1u], "amd_rocm6_wave64__zzprivate_attention_set", blocks, (uint32_t)lanes,
                        args, sizes, 5u);
}


/* ⭐ ATTENTION WITH A KEY LIMIT A ROW — a diffusion transformer's sequence (nn's `attention_limits`): the set's kernel
 *   above — a block 128 rows of one head, two lanes a row, the keys and values through shared memory 64 rows at a time
 *   (STAGING), the same tile — with q, k and v apart, each on its own stride, and row r's keys `0 .. limits[r]`. A block
 *   stages the tiles up to the largest limit among its rows — every lane reads them all, so every lane loops alike and
 *   reaches every wait — and a lane scores a tile's keys up to its own row's limit, the whole tile without a bound in
 *   sight where its row reads past it. A row's arithmetic is the same whichever rows share its block: the tiles start at
 *   the same keys, so a later step's rows alone answer what they answer among every row. A row whose limit is 0 answers
 *   zeros, as nn's. Heads of whole products of 8 columns, 128 at most; anything else is nn's. `MEASURED` (node03, a block
 *   of Qwen-Image 2.1's transformer, test/src_qwen_image_time.py, measurements/2026-10-15_qwen_image_block_time_node03.md):
 *   at 4,123 rows 0.41 s, where nn's took ≈ 11.5 s; at 2,331 rows 0.21 s against 3.76 s. */
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_limits(uint16_t* o, const uint16_t* q, const uint16_t* k,
                                                                           const uint16_t* v, const uint32_t* limits,
                                                                           uint64_t rows, uint64_t heads, uint64_t q_stride,
                                                                           uint64_t kv_stride, unsigned int* over) {
    constexpr uint32_t HP = D / 4u, KW = (HP + 3u) & ~3u, HD = D / 2u;  /* a lane's half: pairs, padded words, elements */
    constexpr uint32_t TILE = AMD_ROCM6_WAVE64__ZZPRIVATE_SET_TILE, ROWS = AMD_ROCM6_WAVE64__ZZPRIVATE_SET_ROWS;
    constexpr uint32_t LANES = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    __shared__ uint32_t keys[TILE * 2u * KW];                             /* key j's half h, pair w at (2j + h)·KW + w */
    __shared__ uint32_t values[(TILE / 2u) * D];                          /* keys 2i and 2i+1's element d at i·D + d */
    const uint64_t H = heads * D, groups = (rows + ROWS - 1u) / ROWS;
    const uint32_t t = (uint32_t)nn__silicon__lane(), h = t & 1u;
    const float scale = 1.4426950408889634f / nn__silicon__sqrtf((float)D);   /* log2(e) / sqrt(D): base-2 scores */
    for (uint64_t item = nn__silicon__block(); item < heads * groups; item += nn__silicon__blocks()) {
        const uint64_t e = item / groups, g0 = (item - e * groups) * ROWS, r = g0 + t / 2u;
        const bool mine = r < rows;
        const uint32_t limit = mine ? limits[r] : 0u;
        const uint64_t here = rows - g0 < ROWS ? rows - g0 : ROWS;
        uint32_t most = 0u;                                               /* the block's largest limit, the same in every lane */
        for (uint64_t i = 0u; i < here; ++i) most = limits[g0 + i] > most ? limits[g0 + i] : most;
        uint32_t qw[HP];
        const uint32_t* qr = (const uint32_t*)(q + (mine ? r : 0ull) * q_stride + e * D + h * HD);
#pragma unroll
        for (uint32_t w = 0u; w < HP; ++w) qw[w] = mine ? qr[w] : 0u;
        float acc[HD];
#pragma unroll
        for (uint32_t d = 0u; d < HD; ++d) acc[d] = 0.0f;
        float m = -NN__KERNELS__INFINITY, l = 0.0f;
        for (uint64_t base = 0ull; base < most; base += TILE) {
            const uint32_t n = most - base < TILE ? (uint32_t)(most - base) : TILE;
            for (uint32_t i = t; i < TILE * 2u * KW; i += LANES) {        /* the tile's keys, each half on whole loads */
                const uint32_t j = i / (2u * KW), rest = i - j * 2u * KW, hh = rest / KW, w = rest - hh * KW;
                keys[i] = j < n && w < HP ? ((const uint32_t*)(k + (base + j) * kv_stride + e * D + hh * HD))[w] : 0u;
            }
            for (uint32_t i = t; i < (TILE / 2u) * (D / 2u); i += LANES) { /* its values, two keys' same element a word */
                const uint32_t jp = i / (D / 2u), w = i - jp * (D / 2u), j = 2u * jp;
                const uint32_t a = j < n ? ((const uint32_t*)(v + (base + j) * kv_stride + e * D))[w] : 0u;
                const uint32_t b = j + 1u < n ? ((const uint32_t*)(v + (base + j + 1u) * kv_stride + e * D))[w] : 0u;
                values[jp * D + 2u * w] = (a & 0xFFFFu) | (b << 16);
                values[jp * D + 2u * w + 1u] = (a >> 16) | (b & 0xFFFF0000u);
            }
            __syncthreads();                                              /* the tile, before any lane reads it */
            const uint32_t c = limit <= base ? 0u : (limit - base < n ? (uint32_t)(limit - base) : n);
            if (c == TILE)
                amd_rocm6_wave64__zzprivate_set_tile<D, TILE>(acc, m, l, qw, keys + h * KW, values + h * HD, c, scale);
            else if (c > 0u)
                amd_rocm6_wave64__zzprivate_set_tile<D, 0u>(acc, m, l, qw, keys + h * KW, values + h * HD, c, scale);
            __syncthreads();                                              /* every lane done with the tile */
        }
        if (mine) {
            uint16_t* out = o + r * H + e * D + h * HD;
#pragma unroll
            for (uint32_t d = 0u; d < HD; ++d) out[d] = l > 0.0f ? amd_rocm6_wave64__zzpackage_to_half(acc[d] / l, over) : 0u;
        }
    }
}
template <uint32_t D>
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_SET_BOUNDS void amd_rocm6_wave64__zzprivate_attention_limits_head(
        uint16_t* o, const uint16_t* q, const uint16_t* k, const uint16_t* v, const uint32_t* limits, uint64_t rows,
        uint64_t heads, uint64_t q_stride, uint64_t kv_stride, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_limits<D>(o, q, k, v, limits, rows, heads, q_stride, kv_stride, over);
}
/* What it takes: a head of whole products of 8 columns, 128 at most; q, k and v on four bytes and their strides even
 * (their pairs' loads). */
static void amd_rocm6_wave64__override_attention_limits(uint16_t* o, const uint16_t* q, const uint16_t* k, const uint16_t* v,
                                                        const uint32_t* limits, uint64_t rows, uint64_t heads,
                                                        uint64_t head_dim, uint64_t q_stride, uint64_t kv_stride,
                                                        unsigned int* over) {
    static const void* const kernels[16] = {           /* a head's width in 8s, less one */
        (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<8u>,   (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<16u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<24u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<32u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<40u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<48u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<56u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<64u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<72u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<80u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<88u>,  (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<96u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<104u>, (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<112u>,
        (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<120u>, (const void*)amd_rocm6_wave64__zzprivate_attention_limits_head<128u> };
    if (rows == 0u || heads == 0u || head_dim == 0u || head_dim % 8u != 0u || head_dim > 128u
     || ((uint64_t)(uintptr_t)q) % 4u != 0u || ((uint64_t)(uintptr_t)k) % 4u != 0u || ((uint64_t)(uintptr_t)v) % 4u != 0u
     || q_stride % 2u != 0u || kv_stride % 2u != 0u || ((uint64_t)(uintptr_t)o) % 2u != 0u) {
        nn__attention__zzabi_launch_limits(o, q, k, v, limits, rows, heads, head_dim, q_stride, kv_stride, over);
        return;
    }
    const uint64_t lanes = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t items = heads * ((rows + AMD_ROCM6_WAVE64__ZZPRIVATE_SET_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_SET_ROWS);
    const uint32_t blocks = items < NN__KERNELS__BLOCKS_MAX ? (uint32_t)items : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &q, &k, &v, &limits, &rows, &heads, &q_stride, &kv_stride, &over };
    const size_t sizes[] = { sizeof o, sizeof q, sizeof k, sizeof v, sizeof limits, sizeof rows, sizeof heads, sizeof q_stride,
                             sizeof kv_stride, sizeof over };
    nn__silicon__launch(kernels[head_dim / 8u - 1u], "amd_rocm6_wave64__zzprivate_attention_limits", blocks, (uint32_t)lanes,
                        args, sizes, 10u);
}
#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_ATTENTION_CUH */
