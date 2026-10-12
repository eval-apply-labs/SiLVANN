#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_ATTENTION_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_ATTENTION_CUH
/* ══ attention's DOORS, RUN FASTER — amd_rocm6_wave64 ═════════════════════════════════════════════════════════════════
 * nn's attention doors (`nn/cpu/opcodes/attention__abi.cuh`), where this family covers them: the scores and the mix
 * over heads of 128 or 256, as one call or as the two residual passes, and two residuals' merge. */

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
template <uint32_t D, bool HALVES>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_mix(float* res, uint16_t* o, const float* p, const uint16_t* v,
                                                                         uint64_t q_heads, uint64_t kv_heads, uint64_t length,
                                                                         unsigned int* over) {
    const uint32_t PER = D / 64u;                                         /* a lane's elements of a head: 2 or 4 */
    __shared__ float waves[AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS][D];
    const uint32_t t = (uint32_t)nn__silicon__lane();
    const uint32_t lane = t % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, wave = t / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const float* ph = p + h * length;
        const uint16_t* vh = v + (h / group) * D + lane * PER;
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
                const uint32_t* vw = (const uint32_t*)(vh + pos * row);
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
            if (HALVES) o[h * D + e] = nn__kernels__zzabi_to_half(sum, over);
            else res[h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 2ull + e] = sum;
        }
        __syncthreads();
    }
}
/* The eight entry points, written out — a head of 128 and of 256 for each — so every kernel name has a definition a reader
 * (and the C-subset gate) can find. */
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
    if (!amd_rocm6_wave64__zzprivate_attention_fit(v, v, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_mix(o, p, v, q_heads, kv_heads, head_dim, length, over);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_halves_128
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_halves_256;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &p, &v, &q_heads, &kv_heads, &length, &over };
    const size_t sizes[] = { sizeof o, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof length, sizeof over };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_mix", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 7u);
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

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_ATTENTION_CUH */
