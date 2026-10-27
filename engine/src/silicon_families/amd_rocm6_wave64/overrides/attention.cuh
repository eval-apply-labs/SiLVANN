#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_ATTENTION_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_ATTENTION_CUH
/* ══ attention's DOORS, RUN FASTER — amd_rocm6_wave64 ═════════════════════════════════════════════════════════════════
 * nn's attention doors (`nn/cpu/opcodes/attention__abi.cuh`), where this family covers them: the scores and the mix
 * over heads of 128 or 256, as one call or as the two residual passes, and two residuals' merge; the grouped scores, the
 * mix as halves and the causal mix over heads of 512 too, and those two over rows of keys of their own; MLA's absorb and
 * expand, a row or many. */

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
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_raw_head(float* ph, const uint16_t* qh, const uint16_t* kg, uint64_t row,
                                                                              uint64_t length) {
    const uint32_t PER = D / 16u;                                         /* a lane's halves of a head: 8, 16 or 32 */
    const uint32_t t = (uint32_t)nn__silicon__lane(), sub = t % 16u, slot = t / 16u;
    const float scale = 1.0f / nn__silicon__sqrtf((float)D);
    const uint16_t* kh = kg + sub * PER;
    uint32_t qv[PER / 2u];
    const uint32_t* qw = (const uint32_t*)(qh + sub * PER);
#pragma unroll
    for (uint32_t i = 0u; i < PER / 2u; ++i) qv[i] = qw[i];
    for (uint64_t base = slot; base < length; base += 64u) {               /* four positions a lane group in flight */
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
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_raw(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                         uint64_t kv_heads, uint64_t length) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks())
        amd_rocm6_wave64__zzprivate_attention_raw_head<D>(p + h * length, q + h * D, k + (h / group) * D, row, length);
}
/* ⭐ THE GROUPED SCORES AND THE MIX AS HALVES FOR `rows` QUERIES, EACH OVER KEYS AND VALUES OF ITS OWN (NN-115 ③) — nn's
 *   `scores_grouped_rows` and `mix_rows`: a block a (row, query head), each the one-row kernel's head above, so a
 *   prompt's rows and a position at a time are the same bits; one launch a block of rows, where a launch a row gave the
 *   card a block a head — 64 at GLM's, about one a compute unit. */
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_raw_rows(float* p, uint64_t p_row, const uint16_t* q, uint64_t q_row,
                                                                              const uint16_t* k, uint64_t k_row, const uint32_t* lens,
                                                                              uint64_t q_heads, uint64_t kv_heads, uint64_t rows) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    for (uint64_t b = nn__silicon__block(); b < rows * q_heads; b += nn__silicon__blocks()) {
        const uint64_t u = b / q_heads, h = b - u * q_heads, length = (uint64_t)lens[u];
        amd_rocm6_wave64__zzprivate_attention_raw_head<D>(p + u * p_row + h * length, q + u * q_row + h * D, k + u * k_row + (h / group) * D,
                                                          row, length);
    }
}
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_mix_rows(uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row,
                                                                              const uint16_t* v, uint64_t v_row, const uint32_t* lens,
                                                                              uint64_t q_heads, uint64_t kv_heads, uint64_t rows,
                                                                              unsigned int* over) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    for (uint64_t b = nn__silicon__block(); b < rows * q_heads; b += nn__silicon__blocks()) {
        const uint64_t u = b / q_heads, h = b - u * q_heads, length = (uint64_t)lens[u];
        amd_rocm6_wave64__zzprivate_attention_mix_head<D, true>(0, o + u * o_row + h * D, p + u * p_row + h * length,
                                                                v + u * v_row + (h / group) * D, row, length, over);
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
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_raw_rows_128(float* p, uint64_t p_row,
    const uint16_t* q, uint64_t q_row, const uint16_t* k, uint64_t k_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads,
    uint64_t rows) {
    amd_rocm6_wave64__zzprivate_attention_raw_rows<128u>(p, p_row, q, q_row, k, k_row, lens, q_heads, kv_heads, rows);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_raw_rows_256(float* p, uint64_t p_row,
    const uint16_t* q, uint64_t q_row, const uint16_t* k, uint64_t k_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads,
    uint64_t rows) {
    amd_rocm6_wave64__zzprivate_attention_raw_rows<256u>(p, p_row, q, q_row, k, k_row, lens, q_heads, kv_heads, rows);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_raw_rows_512(float* p, uint64_t p_row,
    const uint16_t* q, uint64_t q_row, const uint16_t* k, uint64_t k_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads,
    uint64_t rows) {
    amd_rocm6_wave64__zzprivate_attention_raw_rows<512u>(p, p_row, q, q_row, k, k_row, lens, q_heads, kv_heads, rows);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_rows_128(uint16_t* o, uint64_t o_row,
    const float* p, uint64_t p_row, const uint16_t* v, uint64_t v_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads,
    uint64_t rows, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_mix_rows<128u>(o, o_row, p, p_row, v, v_row, lens, q_heads, kv_heads, rows, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_rows_256(uint16_t* o, uint64_t o_row,
    const float* p, uint64_t p_row, const uint16_t* v, uint64_t v_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads,
    uint64_t rows, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_mix_rows<256u>(o, o_row, p, p_row, v, v_row, lens, q_heads, kv_heads, rows, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_rows_512(uint16_t* o, uint64_t o_row,
    const float* p, uint64_t p_row, const uint16_t* v, uint64_t v_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads,
    uint64_t rows, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_attention_mix_rows<512u>(o, o_row, p, p_row, v, v_row, lens, q_heads, kv_heads, rows, over);
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

/* ⭐ A LATENT'S MIX OVER ROWS IN TILES (NN-115 ③) — `attention_mix_rows` for a kv head of 512 read by 32 or more query
 *   heads (GLM's latent, DeepSeek's): a row's answer the GEMM it is, the heads' weights against its value rows. Each
 *   element is the one-row mix's above to the bit — four chains, a wave's positions `w`, `w + 4` … in order, added as that
 *   kernel adds its four waves' — and the values are read once for every head, where a block a head read them 64 times.
 *   A block 64 heads by 64 elements, the positions 32 at a time staged in shared memory, a lane 4 heads by 4 elements 16
 *   apart. (A query alone keeps the one-row kernel: one row's 8 tiles leave the card idle.) */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_MX_KT 32u
static __device__ inline void amd_rocm6_wave64__zzprivate_mix_tile(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads,
                                                                   uint64_t kv_heads, uint64_t length, uint64_t g, uint64_t mi, uint64_t ei,
                                                                   float (*ps)[AMD_ROCM6_WAVE64__ZZPRIVATE_MX_KT + 1u], float (*vs)[65],
                                                                   unsigned int* over) {
    const uint32_t D = 512u, KT = AMD_ROCM6_WAVE64__ZZPRIVATE_MX_KT, W = 64u, J = 4u, R = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t t = (uint32_t)nn__silicon__lane(), n = (uint32_t)nn__silicon__lanes();
    const uint32_t tr = (t / 16u) * 4u, tp = t % 16u;
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D, m0 = mi * 64u, e0 = ei * W;
    float acc[4][4][AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS];      /* a chain a wave of the one-row kernel: positions ≡ w mod 4 */
#pragma unroll
    for (uint32_t a = 0u; a < 4u; ++a)
#pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j)
#pragma unroll
            for (uint32_t w = 0u; w < R; ++w) acc[a][j][w] = 0.0f;
    for (uint64_t pos0 = 0u; pos0 < length; pos0 += KT) {
        for (uint32_t i = t; i < 64u * KT; i += n) {               /* the heads' weights: 64 heads by KT positions */
            const uint32_t r = i / KT, c = i % KT;
            const uint64_t m = m0 + r, pos = pos0 + c;
            ps[r][c] = (m < group && pos < length) ? p[(g * group + m) * length + pos] : 0.0f;
        }
        for (uint32_t i = t; i < KT * W; i += n) {                 /* the values: KT positions by 64 elements */
            const uint32_t r = i / W, c = i % W;
            const uint64_t pos = pos0 + r;
            vs[r][c] = pos < length ? nn__silicon__half_to_float(v[pos * row + g * D + e0 + c]) : 0.0f;
        }
        __syncthreads();
        const uint64_t span = length - pos0 < KT ? length - pos0 : KT;
        for (uint32_t c0 = 0u; c0 < (uint32_t)span; c0 += R) {   /* KT a multiple of 4: position pos0 + c is chain c % 4's */
#pragma unroll
            for (uint32_t w = 0u; w < R; ++w) {
                const uint32_t c = c0 + w;
                if (c >= (uint32_t)span) break;
                float pv[4], vv[4];
#pragma unroll
                for (uint32_t a = 0u; a < 4u; ++a) pv[a] = ps[tr + a][c];
#pragma unroll
                for (uint32_t j = 0u; j < J; ++j) vv[j] = vs[c][tp + 16u * j];
#pragma unroll
                for (uint32_t a = 0u; a < 4u; ++a)
#pragma unroll
                    for (uint32_t j = 0u; j < J; ++j) acc[a][j][w] += pv[a] * vv[j];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t a = 0u; a < 4u; ++a) {
        const uint64_t m = m0 + tr + a;
        if (m >= group) continue;
#pragma unroll
        for (uint32_t j = 0u; j < J; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (uint32_t w = 0u; w < R; ++w) sum += acc[a][j][w];
            o[(g * group + m) * D + e0 + tp + 16u * j] = nn__kernels__zzabi_to_half(sum, over);
        }
    }
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_mix_tile_rows(
        uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row, const uint16_t* v, uint64_t v_row, const uint32_t* lens,
        uint64_t q_heads, uint64_t kv_heads, uint64_t rows, unsigned int* over) {
    __shared__ float ps[64][AMD_ROCM6_WAVE64__ZZPRIVATE_MX_KT + 1u];
    __shared__ float vs[AMD_ROCM6_WAVE64__ZZPRIVATE_MX_KT][65];
    const uint64_t mt = (q_heads / kv_heads + 63u) / 64u, per = kv_heads * mt * 8u;
    for (uint64_t b = nn__silicon__block(); b < rows * per; b += nn__silicon__blocks()) {
        const uint64_t u = b / per, rest = b - u * per, g = rest / (mt * 8u), tile = rest % (mt * 8u);
        amd_rocm6_wave64__zzprivate_mix_tile(o + u * o_row, p + u * p_row, v + u * v_row, q_heads, kv_heads, (uint64_t)lens[u], g,
                                             tile / 8u, tile % 8u, ps, vs, over);
    }
}
/* Whether the mix's tile takes the call: a kv head of 512 read by at least 32 query heads, the values on 4 bytes. */
static inline bool amd_rocm6_wave64__zzprivate_mix_tile_fit(const uint16_t* v, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim) {
    return head_dim == 512u && kv_heads != 0u && q_heads % kv_heads == 0u && q_heads / kv_heads >= 32u && ((uint64_t)(uintptr_t)v) % 4u == 0u;
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
/* ⭐ A PROMPT'S GROUPED CAUSAL SCORES IN TILES (NN-114) — nn's grouped causal scores give a lane 4 query rows by 8 keys
 *   straight from memory; here a kv head's work is the GEMM it is: its rows each query row's heads of the group, its
 *   columns the key positions, summed over the head's width — 64 rows by 64 positions a block, the width 64 columns at a
 *   time staged in shared memory as the expert GEMM's tiles are (▶ expert.cuh), a lane 4 rows by 4 positions 16 apart. A
 *   tile no row of which reaches its first position is not run; inside one, a position past its row's is not written. Each
 *   dot is nn's: one chain of `dot2` over the width's pairs in order, then the scale — so the scores are nn's generic
 *   kernel's to the bit (the bench's case). (The 27B's 2,000-position prompt: 0.43 s of its 10.2 s in nn's.) */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT 64u
/* One 64 × 64 tile of the scores — kv head `g`, query rows `mi`, positions `ni` — staged through `qs` and `ks`. */
static __device__ inline void amd_rocm6_wave64__zzprivate_scores_tile(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                      uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows,
                                                                      uint64_t g, uint64_t mi, uint64_t ni,
                                                                      uint32_t (*qs)[AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT / 2u + 1u],
                                                                      uint32_t (*ks)[AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT / 2u + 1u]) {
    const uint32_t t = (uint32_t)nn__silicon__lane(), n = (uint32_t)nn__silicon__lanes();
    const uint32_t tr = (t / 16u) * 4u, tp = t % 16u;
    const uint64_t group = q_heads / kv_heads, span_all = first + rows, krow = kv_heads * head_dim;
    const uint64_t M = rows * group;
    const float scale = 1.0f / nn__silicon__sqrtf((float)head_dim);
    {
        const uint64_t m0 = mi * 64u, n0 = ni * 64u;
        float sum[4][4];
        for (uint32_t a = 0u; a < 4u; ++a)
            for (uint32_t c = 0u; c < 4u; ++c) sum[a][c] = 0.0f;
        for (uint64_t k0 = 0u; k0 < head_dim; k0 += AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT) {
            /* a slice: 64 query rows and 64 keys, eight halves an item, two items a lane */
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t i = t + n * h, r = i / 8u, c = (i % 8u) * 4u;
                uint4 qv = make_uint4(0u, 0u, 0u, 0u), kv = make_uint4(0u, 0u, 0u, 0u);
                const uint64_t m = m0 + r, pos = n0 + r;
                if (m < M)
                    qv = *(const uint4*)(const void*)(q + ((m / group) * q_heads + g * group + m % group) * head_dim + k0 + 2u * c);
                if (pos < span_all)
                    kv = *(const uint4*)(const void*)(k + pos * krow + g * head_dim + k0 + 2u * c);
                qs[r][c] = qv.x; qs[r][c + 1u] = qv.y; qs[r][c + 2u] = qv.z; qs[r][c + 3u] = qv.w;
                ks[r][c] = kv.x; ks[r][c + 1u] = kv.y; ks[r][c + 2u] = kv.z; ks[r][c + 3u] = kv.w;
            }
            __syncthreads();
            for (uint32_t c = 0u; c < AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT / 2u; ++c) {
                uint32_t qv[4], kv[4];
                for (uint32_t a = 0u; a < 4u; ++a) qv[a] = qs[tr + a][c];
                for (uint32_t j = 0u; j < 4u; ++j) kv[j] = ks[tp + 16u * j][c];
                for (uint32_t a = 0u; a < 4u; ++a)
                    for (uint32_t j = 0u; j < 4u; ++j) sum[a][j] = nn__silicon__dot2(qv[a], kv[j], sum[a][j]);
            }
            __syncthreads();
        }
        for (uint32_t a = 0u; a < 4u; ++a) {
            const uint64_t m = m0 + tr + a;
            if (m >= M) continue;
            const uint64_t r = m / group, h = g * group + m % group;
            for (uint32_t j = 0u; j < 4u; ++j) {
                const uint64_t pos = n0 + tp + 16u * j;
                if (pos <= first + r) p[(r * q_heads + h) * span_all + pos] = sum[a][j] * scale;
            }
        }
    }
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_causal_scores_tile(
        float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t first,
        uint64_t rows) {
    __shared__ uint32_t qs[64][AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT / 2u + 1u];
    __shared__ uint32_t ks[64][AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT / 2u + 1u];
    const uint64_t group = q_heads / kv_heads, M = rows * group, mt = (M + 63u) / 64u, nt = (first + rows + 63u) / 64u;
    for (uint64_t b = nn__silicon__block(); b < kv_heads * mt * nt; b += nn__silicon__blocks()) {
        const uint64_t g = b / (mt * nt), mi = (b / nt) % mt, ni = b % nt;
        const uint64_t last = (mi * 64u + 63u < M ? mi * 64u + 63u : M - 1u) / group;     /* the tile's last query row */
        if (ni * 64u > first + last) continue;                                           /* past every row's reach */
        amd_rocm6_wave64__zzprivate_scores_tile(p, q, k, q_heads, kv_heads, head_dim, first, rows, g, mi, ni, qs, ks);
    }
}
/* ⭐ THE GROUPED SCORES OF `rows` QUERIES, EACH OVER `lens[u]` KEYS OF ITS OWN, IN THE SAME TILES (NN-115 ③): row `u` a
 *   query of the tile above at `first = lens[u] − 1` — every query head of a kv head against every key, a block a 64 × 64
 *   tile — so each score is nn's generic kernel's, and a row's keys are read once for all 64 heads of GLM's latent. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_scores_tile_rows(
        float* p, uint64_t p_row, const uint16_t* q, uint64_t q_row, const uint16_t* k, uint64_t k_row, const uint32_t* lens,
        uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t most, uint64_t rows) {
    __shared__ uint32_t qs[64][AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT / 2u + 1u];
    __shared__ uint32_t ks[64][AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT / 2u + 1u];
    const uint64_t mt = (q_heads / kv_heads + 63u) / 64u, nt = (most + 63u) / 64u, per = kv_heads * mt * nt;
    for (uint64_t b = nn__silicon__block(); b < rows * per; b += nn__silicon__blocks()) {
        const uint64_t u = b / per, rest = b - u * per, length = (uint64_t)lens[u];
        const uint64_t g = rest / (mt * nt), mi = (rest / nt) % mt, ni = rest % nt;
        if (length == 0u || ni * 64u >= length) continue;
        amd_rocm6_wave64__zzprivate_scores_tile(p + u * p_row, q + u * q_row, k + u * k_row, q_heads, kv_heads, head_dim, length - 1u, 1u,
                                               g, mi, ni, qs, ks);
    }
}
/* Through the tile where every query and key row starts on sixteen bytes and the width is whole slices of 64; else nn's. */
static void amd_rocm6_wave64__override_attention_causal_scores_grouped(float* p, const uint16_t* q, const uint16_t* k,
                                                                       uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                                                       uint64_t first, uint64_t rows) {
    if (rows == 0u || kv_heads == 0u || q_heads % kv_heads != 0u || head_dim % AMD_ROCM6_WAVE64__ZZPRIVATE_CS_KT != 0u
        || ((uint64_t)(uintptr_t)q) % 16u != 0u || ((uint64_t)(uintptr_t)k) % 16u != 0u) {
        nn__attention__zzabi_launch_causal_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, first, rows);
        return;
    }
    const uint64_t tiles = kv_heads * ((rows * (q_heads / kv_heads) + 63u) / 64u) * ((first + rows + 63u) / 64u);
    const uint32_t blocks = tiles < NN__KERNELS__BLOCKS_MAX ? (uint32_t)tiles : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &q, &k, &q_heads, &kv_heads, &head_dim, &first, &rows };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof first, sizeof rows };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_causal_scores_tile, "amd_rocm6_wave64__zzprivate_causal_scores_tile",
                        blocks, AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 8u);
}
/* Whether the tile takes a query's grouped scores: a latent's heads of 512, at least 32 query heads a kv head, every query
 * and key row on sixteen bytes — GLM's and DeepSeek's latent attention, whose 64 heads then read each key once. A query
 * alone takes it from 256 keys: `MEASURED` (test/src_gemv_27b_bench.cpp, BENCH_ATTENTION_ONLY, one MI50) at 2,048 keys 57
 * us against the kernel above's 197, at 640 53 against 72, at 37 52 against 22. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_SCORES_TILE_FROM 256u
static inline bool amd_rocm6_wave64__zzprivate_scores_tile_fit(const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads,
                                                               uint64_t head_dim) {
    return head_dim == 512u && kv_heads != 0u && q_heads % kv_heads == 0u && q_heads / kv_heads >= 32u
        && ((uint64_t)(uintptr_t)q) % 16u == 0u && ((uint64_t)(uintptr_t)k) % 16u == 0u;
}
static void amd_rocm6_wave64__override_attention_scores_grouped(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (length >= AMD_ROCM6_WAVE64__ZZPRIVATE_SCORES_TILE_FROM && amd_rocm6_wave64__zzprivate_scores_tile_fit(q, k, q_heads, kv_heads, head_dim)) {
        amd_rocm6_wave64__override_attention_causal_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, length - 1u, 1u);
        return;
    }
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
/* The rows forms where the one-row kernels fit, every row's queries, keys and values on 4 bytes as the first's; else nn's. */
static void amd_rocm6_wave64__override_attention_scores_grouped_rows(float* p, uint64_t p_row, const uint16_t* q, uint64_t q_row,
                                                                     const uint16_t* k, uint64_t k_row, const uint32_t* lens,
                                                                     uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t most,
                                                                     uint64_t rows) {
    if (rows != 0u && most != 0u && q_row % 8u == 0u && k_row % 8u == 0u
        && amd_rocm6_wave64__zzprivate_scores_tile_fit(q, k, q_heads, kv_heads, head_dim)) {
        const uint64_t tiles = rows * kv_heads * ((q_heads / kv_heads + 63u) / 64u) * ((most + 63u) / 64u);
        const uint32_t blocks = tiles < NN__KERNELS__BLOCKS_MAX ? (uint32_t)tiles : NN__KERNELS__BLOCKS_MAX;
        void* args[] = { &p, &p_row, &q, &q_row, &k, &k_row, &lens, &q_heads, &kv_heads, &head_dim, &most, &rows };
        const size_t sizes[] = { sizeof p, sizeof p_row, sizeof q, sizeof q_row, sizeof k, sizeof k_row, sizeof lens, sizeof q_heads,
                                 sizeof kv_heads, sizeof head_dim, sizeof most, sizeof rows };
        nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_scores_tile_rows, "amd_rocm6_wave64__zzprivate_scores_tile_rows",
                            blocks, AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 12u);
        return;
    }
    if (rows == 0u || q_row % 2u != 0u || k_row % 2u != 0u
        || !amd_rocm6_wave64__zzprivate_attention_fit_wide(q, k, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_scores_grouped_rows(p, p_row, q, q_row, k, k_row, lens, q_heads, kv_heads, head_dim, most, rows);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_raw_rows_128
                       : head_dim == 256u ? (const void*)amd_rocm6_wave64__zzprivate_attention_raw_rows_256
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_raw_rows_512;
    const uint64_t n = rows * q_heads;
    const uint32_t blocks = n < NN__KERNELS__BLOCKS_MAX ? (uint32_t)n : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &p_row, &q, &q_row, &k, &k_row, &lens, &q_heads, &kv_heads, &rows };
    const size_t sizes[] = { sizeof p, sizeof p_row, sizeof q, sizeof q_row, sizeof k, sizeof k_row, sizeof lens, sizeof q_heads,
                             sizeof kv_heads, sizeof rows };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_raw_rows", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 10u);
}
static void amd_rocm6_wave64__override_attention_mix_rows(uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row, const uint16_t* v,
                                                          uint64_t v_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads,
                                                          uint64_t head_dim, uint64_t rows, unsigned int* over) {
    if (rows != 0u && v_row % 2u == 0u && amd_rocm6_wave64__zzprivate_mix_tile_fit(v, q_heads, kv_heads, head_dim)) {
        const uint64_t tiles = rows * kv_heads * ((q_heads / kv_heads + 63u) / 64u) * 8u;
        const uint32_t blocks = tiles < NN__KERNELS__BLOCKS_MAX ? (uint32_t)tiles : NN__KERNELS__BLOCKS_MAX;
        void* args[] = { &o, &o_row, &p, &p_row, &v, &v_row, &lens, &q_heads, &kv_heads, &rows, &over };
        const size_t sizes[] = { sizeof o, sizeof o_row, sizeof p, sizeof p_row, sizeof v, sizeof v_row, sizeof lens, sizeof q_heads,
                                 sizeof kv_heads, sizeof rows, sizeof over };
        nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_mix_tile_rows, "amd_rocm6_wave64__zzprivate_mix_tile_rows", blocks,
                            AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 11u);
        return;
    }
    if (rows == 0u || v_row % 2u != 0u || !amd_rocm6_wave64__zzprivate_attention_fit_wide(v, v, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_mix_rows(o, o_row, p, p_row, v, v_row, lens, q_heads, kv_heads, head_dim, rows, over);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_rows_128
                       : head_dim == 256u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_rows_256
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_rows_512;
    const uint64_t n = rows * q_heads;
    const uint32_t blocks = n < NN__KERNELS__BLOCKS_MAX ? (uint32_t)n : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &o_row, &p, &p_row, &v, &v_row, &lens, &q_heads, &kv_heads, &rows, &over };
    const size_t sizes[] = { sizeof o, sizeof o_row, sizeof p, sizeof p_row, sizeof v, sizeof v_row, sizeof lens, sizeof q_heads,
                             sizeof kv_heads, sizeof rows, sizeof over };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_mix_rows", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 11u);
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
/* ⭐ THE SAME TWO PASSES OVER A LONG TIER, ITS POSITIONS IN SLICES (NN-118) — a block a query head walks every position
 *   of it, so a tier of 28,000 positions on the 27B's 24 heads keeps 24 of the MI50's 60 units busy: `MEASURED` (node03,
 *   rocprof over the 27B's answer after 32,098 positions) 1.42 ms for the weights and 2.84 ms for the mix of a layer's
 *   7 MB a head, ~10 GB/s. Here a head's positions are cut into `slices` of `length / slices` (the last takes the rest),
 *   a block a head and a slice — the heads of a slice next to each other, so a key head's group reads its rows while
 *   they are in the cache — as many as the card holds at once:
 *   · the weights in three launches: a slice's scores, as the one-block pass computes each, and their largest, kept in
 *     the residual's sums (a head's two numbers a slice there before the mix writes them); then each score exponentiated
 *     against the head's largest — the same bits the one-block pass leaves — and the slice's total kept beside; then the
 *     head's largest and its total, the slices' in their order.
 *   · the mix in two: a slice's sums of its weighted values, as the one-block pass adds a head's, left in the first
 *     `D` weights of the slice once the block has read them all (a slice is `D` positions or more); then each element's
 *     slices added in their order. The weights are the door's after the mix (▶ nn's `attention_residual_mix_tq`).
 *   A tier shorter than two slices of `D` takes the one-block passes. The sums are added in another order than theirs —
 *   within the bench's tolerance of nn's host body — and in the same order every run. */
template <uint32_t D, uint32_t BITS>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_weights_tq_slice(float* p, float* res, const uint16_t* q,
                                                                                      const uint8_t* codes, const uint16_t* scales,
                                                                                      uint64_t q_heads, uint64_t kv_heads,
                                                                                      uint64_t length, uint64_t slices) {
    constexpr uint32_t PER = D / 16u;
    constexpr uint32_t WORDS = PER * BITS / 32u;
    constexpr uint32_t CODES = 32u / BITS, LEVELS = 1u << BITS, ROW = D * BITS / 8u;
    __shared__ float levels[LEVELS];
    const uint32_t t = (uint32_t)nn__silicon__lane(), sub = t % 16u, slot = t / 16u;
    const uint16_t* table = nn__turboquant__zzabi_levels(BITS);
    for (uint32_t i = t; i < LEVELS; i += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        levels[i] = nn__silicon__half_to_float(table[i]);
    __syncthreads();                                                      /* the table, before any lane reads it */
    const uint64_t group = q_heads / kv_heads, per = length / slices;
    const float inv = 1.0f / nn__silicon__sqrtf((float)D);
    for (uint64_t b = nn__silicon__block(); b < q_heads * slices; b += nn__silicon__blocks()) {
        const uint64_t h = b % q_heads, s = b / q_heads, g = h / group;
        const uint64_t lo = s * per, hi = s + 1u == slices ? length : lo + per;
        float qf[PER];
#pragma unroll
        for (uint32_t i = 0u; i < PER; ++i) qf[i] = nn__silicon__half_to_float(q[h * D + sub * PER + i]);
        float* ph = p + h * length;
        float top = -NN__KERNELS__INFINITY;
        for (uint64_t base = lo + slot; base < hi; base += 64u) {
            uint32_t kw[4][WORDS];
            float sc[4];
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                const uint64_t pos = base + 16u * u < hi ? base + 16u * u : base;
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
                if (pos < hi) {
                    const float score = dot * sc[u] * inv;
                    if (sub == 0u) ph[pos] = score;
                    top = score > top ? score : top;
                }
            }
        }
        top = nn__silicon__lanes_max(top);
        if (t == 0u) res[h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 2ull + s] = top;
    }
}
/* The second launch: a slice's scores against its head's largest, and the slice's total beside its largest. */
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_exp_tq_slice(float* p, float* res, uint64_t q_heads,
                                                                                  uint64_t length, uint64_t slices) {
    const uint32_t t = (uint32_t)nn__silicon__lane();
    const uint64_t per = length / slices;
    for (uint64_t b = nn__silicon__block(); b < q_heads * slices; b += nn__silicon__blocks()) {
        const uint64_t h = b % q_heads, s = b / q_heads;
        const uint64_t lo = s * per, hi = s + 1u == slices ? length : lo + per;
        const float* kept = res + h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 2ull;
        float top = kept[0];
        for (uint64_t i = 1u; i < slices; ++i) top = kept[i] > top ? kept[i] : top;
        float* ph = p + h * length;
        float part = 0.0f;
        for (uint64_t pos = lo + t; pos < hi; pos += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
            const float e = nn__silicon__expf(ph[pos] - top);
            ph[pos] = e;
            part += e;
        }
        const float sum = nn__silicon__lanes_sum(part);
        if (t == 0u) res[h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 2ull + slices + s] = sum;
    }
}
/* The third: each head's largest and its total, the slices' added in their order — a lane a head. */
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_total_tq_slice(float* res, uint64_t q_heads, uint64_t slices) {
    const uint64_t h = (uint64_t)nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE
                     + nn__silicon__lane();
    if (h >= q_heads) return;
    float* r = res + h * NN__ATTENTION__RESIDUAL_FLOATS(D);
    float top = r[2], sum = 0.0f;
    for (uint64_t s = 1u; s < slices; ++s) top = r[2ull + s] > top ? r[2ull + s] : top;
    for (uint64_t s = 0u; s < slices; ++s) sum += r[2ull + slices + s];
    r[0] = top;
    r[1] = sum;
}
template <uint32_t D, uint32_t BITS>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_mix_tq_slice(float* p, const uint8_t* codes,
                                                                                  const uint16_t* scales, uint64_t q_heads,
                                                                                  uint64_t kv_heads, uint64_t length,
                                                                                  uint64_t slices) {
    constexpr uint32_t PER = D / 64u;
    constexpr uint32_t BYTES = PER * BITS / 8u;
    constexpr uint32_t LEVELS = 1u << BITS, ROW = D * BITS / 8u;
    __shared__ float levels[LEVELS];
    __shared__ float waves[AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS][D];
    const uint32_t t = (uint32_t)nn__silicon__lane();
    const uint32_t lane = t % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, wave = t / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint16_t* table = nn__turboquant__zzabi_levels(BITS);
    for (uint32_t i = t; i < LEVELS; i += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        levels[i] = nn__silicon__half_to_float(table[i]);
    __syncthreads();                                                      /* the table, before any lane reads it */
    const uint64_t group = q_heads / kv_heads, per = length / slices;
    for (uint64_t b = nn__silicon__block(); b < q_heads * slices; b += nn__silicon__blocks()) {
        const uint64_t h = b % q_heads, s = b / q_heads, g = h / group;
        const uint64_t lo = s * per, hi = s + 1u == slices ? length : lo + per;
        float* ph = p + h * length;
        float acc[PER];
#pragma unroll
        for (uint32_t e = 0u; e < PER; ++e) acc[e] = 0.0f;
        for (uint64_t base = lo + wave; base < hi; base += 8u * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
            uint32_t cw[8];
            float wv[8];
#pragma unroll
            for (uint32_t u = 0u; u < 8u; ++u) {
                const uint64_t at = base + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * u;
                const bool real = at < hi;
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
        __syncthreads();                                                  /* every weight of the slice read, its sums in */
        for (uint32_t e = t; e < D; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
            float sum = 0.0f;
#pragma unroll
            for (uint32_t w = 0u; w < AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS; ++w) sum += waves[w][e];
            ph[lo + e] = sum;
        }
        __syncthreads();
    }
}
/* The mix's second launch: an element a lane, its slices' sums added in their order into the head's residual. */
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_add_tq_slice(float* res, const float* p, uint64_t q_heads,
                                                                                  uint64_t length, uint64_t slices) {
    const uint32_t t = (uint32_t)nn__silicon__lane();
    const uint64_t per = length / slices;
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const float* ph = p + h * length;
        for (uint32_t e = t; e < D; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
            float sum = 0.0f;
            for (uint64_t s = 0u; s < slices; ++s) sum += ph[s * per + e];
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 2ull + e] = sum;
        }
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
/* The sliced passes' entry points, written out: the scores and the mix for each width and bits, then the exponentials,
 * the totals and the mix's sums for each width. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_128_4(float* p, float* res,
    const uint16_t* q, const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length,
    uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_weights_tq_slice<128u, 4u>(p, res, q, codes, scales, q_heads, kv_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_128_8(float* p, float* res,
    const uint16_t* q, const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length,
    uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_weights_tq_slice<128u, 8u>(p, res, q, codes, scales, q_heads, kv_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_256_4(float* p, float* res,
    const uint16_t* q, const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length,
    uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_weights_tq_slice<256u, 4u>(p, res, q, codes, scales, q_heads, kv_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_256_8(float* p, float* res,
    const uint16_t* q, const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length,
    uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_weights_tq_slice<256u, 8u>(p, res, q, codes, scales, q_heads, kv_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_tq_slice_128_4(float* p,
    const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length, uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_mix_tq_slice<128u, 4u>(p, codes, scales, q_heads, kv_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_tq_slice_128_8(float* p,
    const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length, uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_mix_tq_slice<128u, 8u>(p, codes, scales, q_heads, kv_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_tq_slice_256_4(float* p,
    const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length, uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_mix_tq_slice<256u, 4u>(p, codes, scales, q_heads, kv_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_tq_slice_256_8(float* p,
    const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t length, uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_mix_tq_slice<256u, 8u>(p, codes, scales, q_heads, kv_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_exp_tq_slice_128(float* p, float* res,
    uint64_t q_heads, uint64_t length, uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_exp_tq_slice<128u>(p, res, q_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_total_tq_slice_128(float* res, uint64_t q_heads,
    uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_total_tq_slice<128u>(res, q_heads, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_add_tq_slice_128(float* res, const float* p,
    uint64_t q_heads, uint64_t length, uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_add_tq_slice<128u>(res, p, q_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_exp_tq_slice_256(float* p, float* res,
    uint64_t q_heads, uint64_t length, uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_exp_tq_slice<256u>(p, res, q_heads, length, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_total_tq_slice_256(float* res, uint64_t q_heads,
    uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_total_tq_slice<256u>(res, q_heads, slices);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_add_tq_slice_256(float* res, const float* p,
    uint64_t q_heads, uint64_t length, uint64_t slices) {
    amd_rocm6_wave64__zzprivate_attention_add_tq_slice<256u>(res, p, q_heads, length, slices);
}
/* A tier's slices a head: as many blocks as the card holds of the sliced scores at once, at most a slice `D` positions
 * long — the mix leaves a slice's `D` sums in it — and at most `D / 2` of them, whose two numbers each the weights keep in
 * the residual's sums; fewer than two, and the one-block passes run. */
static inline uint64_t amd_rocm6_wave64__zzprivate_tq_slices(const void* kernel, uint64_t q_heads, uint64_t head_dim,
                                                             uint64_t length) {
    const uint64_t held = amd_rocm6_wave64__zzpackage_resident(kernel, AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS
                                                                       * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, 0u);
    uint64_t slices = (held + q_heads - 1u) / q_heads;
    if (slices > length / head_dim) slices = length / head_dim;
    if (slices > head_dim / 2u) slices = head_dim / 2u;
    return slices < 2u ? 1u : slices;
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
    const uint32_t lanes = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const void* sliced = head_dim == 128u
        ? (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_128_4
                   : (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_128_8)
        : (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_256_4
                   : (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_256_8);
    uint64_t slices = amd_rocm6_wave64__zzprivate_tq_slices(sliced, q_heads, head_dim, length);
    if (slices > 1u) {
        const uint64_t want = q_heads * slices;
        const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
        void* args[] = { &p, &res, &q, &codes, &scales, &q_heads, &kv_heads, &length, &slices };
        const size_t sizes[] = { sizeof p, sizeof res, sizeof q, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads,
                                 sizeof length, sizeof slices };
        nn__silicon__launch(sliced, "amd_rocm6_wave64__zzprivate_attention_weights_tq_slice", blocks, lanes, args, sizes, 9u);
        void* eargs[] = { &p, &res, &q_heads, &length, &slices };
        const size_t esizes[] = { sizeof p, sizeof res, sizeof q_heads, sizeof length, sizeof slices };
        nn__silicon__launch(head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_exp_tq_slice_128
                                             : (const void*)amd_rocm6_wave64__zzprivate_attention_exp_tq_slice_256,
                            "amd_rocm6_wave64__zzprivate_attention_exp_tq_slice", blocks, lanes, eargs, esizes, 5u);
        void* targs[] = { &res, &q_heads, &slices };
        const size_t tsizes[] = { sizeof res, sizeof q_heads, sizeof slices };
        nn__silicon__launch(head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_total_tq_slice_128
                                             : (const void*)amd_rocm6_wave64__zzprivate_attention_total_tq_slice_256,
                            "amd_rocm6_wave64__zzprivate_attention_total_tq_slice", (uint32_t)((q_heads + lanes - 1u) / lanes),
                            lanes, targs, tsizes, 3u);
        return;
    }
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &res, &q, &codes, &scales, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof p, sizeof res, sizeof q, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads,
                             sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_weights_tq", blocks, lanes, args, sizes, 8u);
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
    const uint32_t lanes = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    /* the weights' slices — the same count from the same shapes, so the slices the weights kept their numbers by */
    const void* weighed = head_dim == 128u
        ? (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_128_4
                   : (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_128_8)
        : (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_256_4
                   : (const void*)amd_rocm6_wave64__zzprivate_attention_weights_tq_slice_256_8);
    uint64_t slices = amd_rocm6_wave64__zzprivate_tq_slices(weighed, q_heads, head_dim, length);
    if (slices > 1u) {
        float* scratch = (float*)p;                   /* the weights, the door's after it reads them (▶ nn's contract) */
        const void* sliced = head_dim == 128u
            ? (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_tq_slice_128_4
                       : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_tq_slice_128_8)
            : (d == 4u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_tq_slice_256_4
                       : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_tq_slice_256_8);
        const uint64_t want = q_heads * slices;
        const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
        void* args[] = { &scratch, &codes, &scales, &q_heads, &kv_heads, &length, &slices };
        const size_t sizes[] = { sizeof scratch, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads, sizeof length,
                                 sizeof slices };
        nn__silicon__launch(sliced, "amd_rocm6_wave64__zzprivate_attention_mix_tq_slice", blocks, lanes, args, sizes, 7u);
        const uint32_t heads = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
        void* aargs[] = { &res, &p, &q_heads, &length, &slices };
        const size_t asizes[] = { sizeof res, sizeof p, sizeof q_heads, sizeof length, sizeof slices };
        nn__silicon__launch(head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_add_tq_slice_128
                                             : (const void*)amd_rocm6_wave64__zzprivate_attention_add_tq_slice_256,
                            "amd_rocm6_wave64__zzprivate_attention_add_tq_slice", heads, lanes, aargs, asizes, 5u);
        return;
    }
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &res, &p, &codes, &scales, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads, sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_mix_tq", blocks, lanes, args, sizes, 7u);
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
/* ⭐ MLA's ABSORB, A WAVE 64 OF A HEAD'S COLUMNS OVER 8 ROWS — nn's `absorb` and `absorb_rows`: each column's sum the
 *   same products added in the same order as nn's card body (`Σᵢ W[i][c] · q[i]`, `i` rising, a lane a column), so the
 *   two agree to the bit. nn's gives a block a head and walks a column's 256 products one load at a time, 64 blocks on
 *   the card; here a wave takes 64 columns of a head for 8 rows, so every weight it loads serves 8 rows, and the rows'
 *   queries, the same for every lane, are read by the scalar unit. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_ABSORB_ROWS  8u
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_absorb(
    uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads, uint64_t nope, uint64_t latent, uint64_t stride,
    float scale, uint64_t rows, unsigned int* over) {
    const uint32_t R = AMD_ROCM6_WAVE64__ZZPRIVATE_ABSORB_ROWS, WAVE = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint32_t l = (uint32_t)nn__silicon__lane() % WAVE, wave = (uint32_t)nn__silicon__lane() / WAVE;
    const uint64_t waves = nn__silicon__lanes() / WAVE, groups = latent / WAVE, hg_all = heads * groups;
    const uint64_t items = hg_all * ((rows + R - 1u) / R);
    for (uint64_t x = (uint64_t)nn__silicon__block() * waves + wave; x < items; x += (uint64_t)nn__silicon__blocks() * waves) {
        const uint64_t hg = x % hg_all, t0 = (x / hg_all) * R, h = hg / groups, c = (hg % groups) * WAVE + l;
        const uint16_t* w = W + h * stride * latent + c;
        const uint16_t* qr[AMD_ROCM6_WAVE64__ZZPRIVATE_ABSORB_ROWS];
        float sum[AMD_ROCM6_WAVE64__ZZPRIVATE_ABSORB_ROWS];
#pragma unroll
        for (uint32_t r = 0u; r < R; ++r) {
            const uint64_t t = t0 + r < rows ? t0 + r : rows - 1u;   /* a block's last rows read the last row, unwritten */
            qr[r] = q + t * heads * nope + h * nope;
            sum[r] = 0.0f;
        }
        for (uint64_t i0 = 0u; i0 < nope; i0 += 8u) {
            float wf[8];
#pragma unroll
            for (uint32_t u = 0u; u < 8u; ++u) wf[u] = nn__silicon__half_to_float(w[(i0 + u) * latent]);
#pragma unroll
            for (uint32_t r = 0u; r < R; ++r)
#pragma unroll
                for (uint32_t u = 0u; u < 8u; ++u) sum[r] += wf[u] * amd_rocm6_wave64__zzpackage_scalar_half(qr[r], (uint32_t)(i0 + u));
        }
#pragma unroll
        for (uint32_t r = 0u; r < R; ++r)
            if (t0 + r < rows) out[(t0 + r) * heads * latent + h * latent + c] = amd_rocm6_wave64__zzpackage_to_half(sum[r] * scale, over);
    }
}
/* Columns in whole waves, a query in whole groups of 8 on 4 bytes for the scalar reads, and `out` apart from the
 * queries; anything else is nn's. */
static bool amd_rocm6_wave64__zzprivate_absorb_fits(const uint16_t* out, const uint16_t* q, uint64_t heads, uint64_t nope,
                                                    uint64_t latent, uint64_t rows) {
    const uint64_t q0 = (uint64_t)(uintptr_t)q, q1 = q0 + rows * heads * nope * 2u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + rows * heads * latent * 2u;
    return heads != 0u && rows != 0u && nope != 0u && latent != 0u && latent % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE == 0u
        && nope % 8u == 0u && q0 % 4u == 0u && (o1 <= q0 || q1 <= o0);
}
static void amd_rocm6_wave64__zzprivate_absorb_launch(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads, uint64_t nope,
                                                      uint64_t latent, uint64_t stride, float scale, uint64_t rows, unsigned int* over) {
    const uint64_t lanes = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t items = heads * (latent / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
                         * ((rows + AMD_ROCM6_WAVE64__ZZPRIVATE_ABSORB_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ABSORB_ROWS);
    const uint64_t want = (items + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &out, &q, &W, &heads, &nope, &latent, &stride, &scale, &rows, &over };
    const size_t sizes[] = { sizeof out, sizeof q, sizeof W, sizeof heads, sizeof nope, sizeof latent, sizeof stride, sizeof scale,
                             sizeof rows, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_attention_absorb, "amd_rocm6_wave64__zzprivate_attention_absorb", blocks,
                        (uint32_t)lanes, args, sizes, 10u);
}
static void amd_rocm6_wave64__override_attention_absorb(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads, uint64_t nope,
                                                        uint64_t latent, uint64_t stride, float scale, unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_absorb_fits(out, q, heads, nope, latent, 1u)) {
        nn__attention__zzabi_launch_absorb(out, q, W, heads, nope, latent, stride, scale, over);
        return;
    }
    amd_rocm6_wave64__zzprivate_absorb_launch(out, q, W, heads, nope, latent, stride, scale, 1u, over);
}
static void amd_rocm6_wave64__override_attention_absorb_rows(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads,
                                                             uint64_t nope, uint64_t latent, uint64_t stride, float scale, uint64_t rows,
                                                             unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_absorb_fits(out, q, heads, nope, latent, rows)) {
        nn__attention__zzabi_launch_absorb_rows(out, q, W, heads, nope, latent, stride, scale, rows, over);
        return;
    }
    amd_rocm6_wave64__zzprivate_absorb_launch(out, q, W, heads, nope, latent, stride, scale, rows, over);
}

/* ⭐ MLA's EXPAND AT A LATENT OF 512, A WAVE 4 OF A HEAD'S OUTPUTS OVER 32 ROWS — nn's `expand` and `expand_rows`: nn's
 *   door gives an output a block of 256 lanes, lane `L` adding its columns `L` and `L + 256`, and the family's fold adds
 *   the 256 parts — each wave's by exchanges, then the four waves' sums in pairs. Here one wave holds the four waves'
 *   parts, lane `l` those of lanes `l`, `l + 64`, `l + 128` and `l + 192`, makes each the same way, exchanges each as
 *   its wave would and adds the four sums in the fold's pairs, so every output is nn's to the bit. nn's reads an
 *   output's weight row again for every row and folds through block-shared memory; here a wave's four weight rows stay
 *   in its registers for its 32 rows, and nothing waits for the block. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_OUTS  4u
#define AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_ROWS  32u
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_expand512(
    uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads, uint64_t width, uint64_t stride, uint64_t offset,
    uint64_t rows, unsigned int* over) {
    const uint32_t J = AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_OUTS, R = AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_ROWS;
    const uint32_t WAVE = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, LAT = 512u;
    const uint32_t l = (uint32_t)nn__silicon__lane() % WAVE, wave = (uint32_t)nn__silicon__lane() / WAVE;
    const uint64_t waves = nn__silicon__lanes() / WAVE, per = width / J, outs = heads * per;
    const uint64_t items = outs * ((rows + R - 1u) / R);
    for (uint64_t x = (uint64_t)nn__silicon__block() * waves + wave; x < items; x += (uint64_t)nn__silicon__blocks() * waves) {
        const uint64_t hj = x % outs, t0 = (x / outs) * R, h = hj / per, j0 = (hj % per) * J;
        const uint64_t tn = rows - t0 < R ? rows - t0 : R;
        float wv[AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_OUTS][4][2];
#pragma unroll
        for (uint32_t jj = 0u; jj < J; ++jj) {
            const uint16_t* row = W + (h * stride + offset + j0 + jj) * LAT;
#pragma unroll
            for (uint32_t v = 0u; v < 4u; ++v)
#pragma unroll
                for (uint32_t k = 0u; k < 2u; ++k) wv[jj][v][k] = nn__silicon__half_to_float(row[v * WAVE + l + k * 256u]);
        }
        for (uint64_t r = 0u; r < tn; ++r) {
            const uint16_t* xr = o + (t0 + r) * heads * LAT + h * LAT;
            float ov[4][2];
#pragma unroll
            for (uint32_t v = 0u; v < 4u; ++v)
#pragma unroll
                for (uint32_t k = 0u; k < 2u; ++k) ov[v][k] = nn__silicon__half_to_float(xr[v * WAVE + l + k * 256u]);
#pragma unroll
            for (uint32_t jj = 0u; jj < J; ++jj) {
                float s[4];
#pragma unroll
                for (uint32_t v = 0u; v < 4u; ++v) {
                    float part = 0.0f;
#pragma unroll
                    for (uint32_t k = 0u; k < 2u; ++k) part += wv[jj][v][k] * ov[v][k];
                    s[v] = part;
                }
#pragma unroll
                for (uint32_t m = 1u; m < WAVE; m <<= 1)
#pragma unroll
                    for (uint32_t v = 0u; v < 4u; ++v) s[v] = s[v] + amd_rocm6_wave64__zzpackage_xor(s[v], m);
                const float sum = (s[0] + s[1]) + (s[2] + s[3]);
                if (l == jj) out[(t0 + r) * heads * width + h * width + j0 + jj] = amd_rocm6_wave64__zzpackage_to_half(sum, over);
            }
        }
    }
}
/* A latent of 512 and outputs in whole fours; anything else is nn's. */
static void amd_rocm6_wave64__zzprivate_expand_launch(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads, uint64_t width,
                                                      uint64_t stride, uint64_t offset, uint64_t rows, unsigned int* over) {
    const uint64_t lanes = AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t items = heads * (width / AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_OUTS)
                         * ((rows + AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_ROWS);
    const uint64_t want = (items + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &out, &o, &W, &heads, &width, &stride, &offset, &rows, &over };
    const size_t sizes[] = { sizeof out, sizeof o, sizeof W, sizeof heads, sizeof width, sizeof stride, sizeof offset, sizeof rows,
                             sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_attention_expand512, "amd_rocm6_wave64__zzprivate_attention_expand512",
                        blocks, (uint32_t)lanes, args, sizes, 9u);
}
static void amd_rocm6_wave64__override_attention_expand(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads, uint64_t width,
                                                        uint64_t latent, uint64_t stride, uint64_t offset, unsigned int* over) {
    if (latent != 512u || heads == 0u || width == 0u || width % AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_OUTS != 0u) {
        nn__attention__zzabi_launch_expand(out, o, W, heads, width, latent, stride, offset, over);
        return;
    }
    amd_rocm6_wave64__zzprivate_expand_launch(out, o, W, heads, width, stride, offset, 1u, over);
}
static void amd_rocm6_wave64__override_attention_expand_rows(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads,
                                                             uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset,
                                                             uint64_t rows, unsigned int* over) {
    if (latent != 512u || heads == 0u || width == 0u || rows == 0u || width % AMD_ROCM6_WAVE64__ZZPRIVATE_EXPAND_OUTS != 0u) {
        nn__attention__zzabi_launch_expand_rows(out, o, W, heads, width, latent, stride, offset, rows, over);
        return;
    }
    amd_rocm6_wave64__zzprivate_expand_launch(out, o, W, heads, width, stride, offset, rows, over);
}
#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_ATTENTION_CUH */
