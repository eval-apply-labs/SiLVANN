#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_INDEX_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_INDEX_CUH
/* ══ index's DOOR, RUN FASTER — amd_rocm6_wave64 ══════════════════════════════════════════════════════════════════════
 * nn's indexer doors (`nn/cpu/opcodes/index__abi.cuh`), where this family covers them: the pools' scores and the top `k` of
 * them, a row or many. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ THE POOLS' SCORES, A HALF-WAVE A POOL — nn's `index_scores` and `index_scores_rows` (DeepSeek's indexer, pooled as
 *   GLM 5.3 Flash pools it): nn's body gives a pool a block of 64 lanes, lane `h` a head's 128-long chain of products
 *   and the 32 lanes past the heads zeros, then the fold. Here each half of a wave takes a pool, lane `h` of it the same
 *   chain over the same products in the same order, its head's query held in registers for 16 pools of one row; the
 *   half's parts are exchanged 1, 2 … 16 apart, as the fold's first steps add them, and its last adds the other half's
 *   zeros — so every score is nn's to the bit, and a row's pools a few dozen waves rather than a block each. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_INDEX_POOLS  8u       /* pool pairs a wave walks with one query */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_index_scores(float* scores, uint64_t scores_row,
                                                                                                   const uint16_t* q, const uint16_t* w,
                                                                                                   const uint16_t* pooled, uint64_t heads,
                                                                                                   uint64_t len0, uint64_t kpool,
                                                                                                   uint64_t rows, float scale) {
    const uint32_t D = 128u, HALF = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u, G = AMD_ROCM6_WAVE64__ZZPRIVATE_INDEX_POOLS;
    const uint32_t lane = (uint32_t)nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint32_t wave = (uint32_t)nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint32_t half = lane / HALF, h = lane % HALF;
    const uint64_t waves = nn__silicon__lanes() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t groups = ((len0 + rows - 1u) / kpool + 2u * G - 1u) / (2u * G), items = rows * groups;
    const float per_head = 1.0f / nn__silicon__sqrtf((float)heads);
    for (uint64_t x = (uint64_t)nn__silicon__block() * waves + wave; x < items; x += (uint64_t)nn__silicon__blocks() * waves) {
        const uint64_t u = x / groups, base = (x - u * groups) * 2u * G, pools = (len0 + u) / kpool;
        const bool head = h < heads;
        uint32_t qv[64];
        const uint32_t* qw = (const uint32_t*)(const void*)(q + u * heads * D + (head ? h : 0u) * D);
#pragma unroll
        for (uint32_t i = 0u; i < 64u; ++i) qv[i] = qw[i];
        const float wh = head ? nn__silicon__half_to_float(w[u * heads + h]) : 0.0f;
        for (uint32_t i = 0u; i < G; ++i) {
            const uint64_t p = base + 2u * i + half;
            const bool real = p < pools;
            const uint32_t* kw = (const uint32_t*)(const void*)(pooled + (real ? p : 0u) * D);
            float part = 0.0f;
            if (head) {
                float dot = 0.0f;
#pragma unroll
                for (uint32_t c = 0u; c < 64u; ++c) {
                    const uint32_t k2 = kw[c];
                    dot += nn__silicon__half_to_float((uint16_t)(qv[c] & 0xFFFFu)) * nn__silicon__half_to_float((uint16_t)(k2 & 0xFFFFu));
                    dot += nn__silicon__half_to_float((uint16_t)(qv[c] >> 16)) * nn__silicon__half_to_float((uint16_t)(k2 >> 16));
                }
                dot *= scale;
                part += (dot > 0.0f ? dot : 0.0f) * wh * per_head;
            }
#pragma unroll
            for (uint32_t m = 1u; m < HALF; m <<= 1) part = part + amd_rocm6_wave64__zzpackage_xor(part, m);
            part = part + 0.0f;                                              /* the fold's last step: the idle lanes' zeros */
            if (h == 0u && real) scores[u * scores_row + p] = part;
        }
    }
}
/* Heads of 128, at most 32 of them, queries and pools on 4 bytes; anything else is nn's. */
static bool amd_rocm6_wave64__zzprivate_index_scores_fit(const uint16_t* q, const uint16_t* pooled, uint64_t heads, uint64_t dim) {
    return dim == 128u && heads != 0u && heads <= AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u && ((uint64_t)(uintptr_t)q) % 4u == 0u
        && ((uint64_t)(uintptr_t)pooled) % 4u == 0u;
}
static void amd_rocm6_wave64__zzprivate_index_scores_launch(float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w,
                                                            const uint16_t* pooled, uint64_t heads, uint64_t len0, uint64_t kpool,
                                                            uint64_t rows, float scale) {
    const uint64_t G2 = 2u * AMD_ROCM6_WAVE64__ZZPRIVATE_INDEX_POOLS;
    const uint64_t items = rows * (((len0 + rows - 1u) / kpool + G2 - 1u) / G2);
    const uint64_t want = (items + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t blocks = want == 0u ? 1u : want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &scores, &scores_row, &q, &w, &pooled, &heads, &len0, &kpool, &rows, &scale };
    const size_t sizes[] = { sizeof scores, sizeof scores_row, sizeof q, sizeof w, sizeof pooled, sizeof heads, sizeof len0, sizeof kpool,
                             sizeof rows, sizeof scale };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_index_scores, "amd_rocm6_wave64__zzprivate_index_scores", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 10u);
}
static void amd_rocm6_wave64__override_index_scores(float* scores, const uint16_t* q, const uint16_t* w, const uint16_t* pooled,
                                                    uint64_t heads, uint64_t dim, uint64_t pools, float scale) {
    if (pools == 0u || !amd_rocm6_wave64__zzprivate_index_scores_fit(q, pooled, heads, dim)) {
        nn__index__zzabi_launch_scores(scores, q, w, pooled, heads, dim, pools, scale);
        return;
    }
    amd_rocm6_wave64__zzprivate_index_scores_launch(scores, 0u, q, w, pooled, heads, pools, 1u, 1u, scale);
}
static void amd_rocm6_wave64__override_index_scores_rows(float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w,
                                                         const uint16_t* pooled, uint64_t heads, uint64_t dim, uint64_t len0,
                                                         uint64_t kpool, uint64_t rows, float scale) {
    if (rows == 0u || kpool == 0u || (len0 + rows - 1u) / kpool == 0u
        || !amd_rocm6_wave64__zzprivate_index_scores_fit(q, pooled, heads, dim)) {
        nn__index__zzabi_launch_scores_rows(scores, scores_row, q, w, pooled, heads, dim, len0, kpool, rows, scale);
        return;
    }
    amd_rocm6_wave64__zzprivate_index_scores_launch(scores, scores_row, q, w, pooled, heads, len0, kpool, rows, scale);
}

/* A score as a key that orders as the score does: larger floats, larger keys — nn's. */
static __device__ inline uint32_t amd_rocm6_wave64__zzprivate_index_key(float f) {
    union { float f; uint32_t u; } cast;
    cast.f = f;
    return (cast.u & 0x80000000u) != 0u ? ~cast.u : (cast.u | 0x80000000u);
}

/* ⭐ THE TOP `k` OF `pools` SCORES, AS POSITIONS — nn's `index_select`: the `k`-th largest key found a bit at a time across
 *   the block, as nn's body finds it, then the pools above it and those equal to it by lowest index, in index order. nn's
 *   body walks every pool twice on one lane for that last part: `MEASURED` with rocprof on DeepSeek's chunk (node03), 257
 *   us a call over ~700 latents. Here the first wave walks them 64 at a time: a pool's place among the chosen is the count
 *   of chosen pools below it in the wave (a ballot) plus those of the waves' steps before, and an equal key is chosen while
 *   the equal keys below it leave room — so `index` and `count` are nn's, to the bit. */
static __device__ inline void amd_rocm6_wave64__zzprivate_index_select_one(uint32_t* index, uint32_t* count, const float* scores,
                                                                           uint64_t pools, uint64_t k, uint64_t kpool, uint64_t tail_first,
                                                                           uint64_t tail);
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_index_select(uint32_t* index, uint32_t* count, const float* scores,
                                                                                                   uint64_t pools, uint64_t k, uint64_t kpool,
                                                                                                   uint64_t tail_first, uint64_t tail) {
    if (nn__silicon__block() != 0u) return;
    amd_rocm6_wave64__zzprivate_index_select_one(index, count, scores, pools, k, kpool, tail_first, tail);
}
/* ⭐ THE SAME FOR `rows` ROWS AT ONCE (NN-115 ③) — nn's `select_rows`: a block a row, each the kernel above's to the bit. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_index_select_rows(uint32_t* index, uint64_t index_row,
                                                                                                        uint32_t* counts, const float* scores,
                                                                                                        uint64_t scores_row, uint64_t k,
                                                                                                        uint64_t kpool, uint64_t len0,
                                                                                                        uint64_t rows) {
    for (uint64_t u = nn__silicon__block(); u < rows; u += nn__silicon__blocks()) {
        const uint64_t len = len0 + u, pools = len / kpool;
        amd_rocm6_wave64__zzprivate_index_select_one(index + u * index_row, counts + u, scores + u * scores_row, pools, k, kpool,
                                                     pools * kpool, len - pools * kpool);
    }
}
/* One block's choice of one row's. */
static __device__ inline void amd_rocm6_wave64__zzprivate_index_select_one(uint32_t* index, uint32_t* count, const float* scores,
                                                                           uint64_t pools, uint64_t k, uint64_t kpool, uint64_t tail_first,
                                                                           uint64_t tail) {
    const uint64_t t = nn__silicon__lane(), lanes = nn__silicon__lanes();
    uint32_t cut = 0u;
    if (pools > k) {
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t want = cut | (1u << (uint32_t)bit);
            float have = 0.0f;
            for (uint64_t p = t; p < pools; p += lanes) have += amd_rocm6_wave64__zzprivate_index_key(scores[p]) >= want ? 1.0f : 0.0f;
            if (nn__silicon__lanes_sum(have) >= (float)k) cut = want;
        }
    }
    if (t >= AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) return;
    const uint32_t lane = (uint32_t)t;
    const uint64_t below = (1ull << lane) - 1ull;
    uint64_t above = 0u;
    if (pools > k) {
        for (uint64_t p = lane; p < pools; p += AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
            above += amd_rocm6_wave64__zzprivate_index_key(scores[p]) > cut ? 1u : 0u;
        for (uint32_t off = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u; off > 0u; off >>= 1)
            above += (uint64_t)__shfl_xor((long long)above, (int)off);
    }
    const uint64_t ties = pools > k ? k - above : 0u;
    uint64_t n = 0u, equal = 0u;                          /* the same in every lane: pools chosen, equal keys passed */
    for (uint64_t base = 0u; base < pools; base += AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
        const uint64_t p = base + lane;
        const bool real = p < pools;
        const uint32_t key = real ? amd_rocm6_wave64__zzprivate_index_key(scores[p]) : 0u;
        const bool same = real && pools > k && key == cut;
        const uint64_t same_mask = __ballot(same);
        const bool take = real && (pools <= k || key > cut || (same && equal + (uint64_t)__popcll(same_mask & below) < ties));
        const uint64_t take_mask = __ballot(take);
        if (take) {
            const uint64_t at = (n + (uint64_t)__popcll(take_mask & below)) * kpool;
            for (uint64_t i = 0u; i < kpool; ++i) index[at + i] = (uint32_t)(p * kpool + i);
        }
        n += (uint64_t)__popcll(take_mask);
        equal += (uint64_t)__popcll(same_mask);
    }
    for (uint64_t i = lane; i < tail; i += AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) index[n * kpool + i] = (uint32_t)(tail_first + i);
    if (lane == 0u) count[0] = (uint32_t)(n * kpool + tail);
}

static void amd_rocm6_wave64__override_index_select(uint32_t* index, uint32_t* count, const float* scores, uint64_t pools, uint64_t k,
                                                    uint64_t kpool, uint64_t tail_first, uint64_t tail) {
    void* args[] = { &index, &count, &scores, &pools, &k, &kpool, &tail_first, &tail };
    const size_t sizes[] = { sizeof index, sizeof count, sizeof scores, sizeof pools, sizeof k, sizeof kpool, sizeof tail_first, sizeof tail };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_index_select, "amd_rocm6_wave64__zzprivate_index_select", 1u,
                        NN__SILICON__LANES_MAX, args, sizes, 8u);
}
static void amd_rocm6_wave64__override_index_select_rows(uint32_t* index, uint64_t index_row, uint32_t* counts, const float* scores,
                                                         uint64_t scores_row, uint64_t k, uint64_t kpool, uint64_t len0, uint64_t rows) {
    if (rows == 0u || kpool == 0u) {
        nn__index__zzabi_launch_select_rows(index, index_row, counts, scores, scores_row, k, kpool, len0, rows);
        return;
    }
    const uint32_t blocks = rows < NN__KERNELS__BLOCKS_MAX ? (uint32_t)rows : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &index, &index_row, &counts, &scores, &scores_row, &k, &kpool, &len0, &rows };
    const size_t sizes[] = { sizeof index, sizeof index_row, sizeof counts, sizeof scores, sizeof scores_row, sizeof k, sizeof kpool,
                             sizeof len0, sizeof rows };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_index_select_rows, "amd_rocm6_wave64__zzprivate_index_select_rows",
                        blocks, NN__SILICON__LANES_MAX, args, sizes, 9u);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_INDEX_CUH */
