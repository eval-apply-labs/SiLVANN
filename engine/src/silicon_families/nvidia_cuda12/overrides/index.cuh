#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_INDEX_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_INDEX_CUH
/* ══ index's DOORS, RUN FASTER — nvidia_cuda12 ════════════════════════════════════════════════════════════════════════
 * nn's indexer doors (`nn/cpu/opcodes/index__abi.cuh`), where this family covers them: the pools' scores and the top `k` of
 * them, a row or many. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ THE POOLS' SCORES, A WARP A POOL — nn's `index_scores` and `index_scores_rows` (DeepSeek's indexer, pooled as GLM 5.3
 *   Flash pools it): nn's body gives a pool a block of 64 lanes, lane `h` a head's 128-long chain of products and the 32
 *   lanes past the heads zeros, then the fold. Here a warp takes a pool, lane `h` the same chain over the same products in
 *   the same order, its head's query held in registers for 16 pools of one row; the warp's parts are exchanged 1, 2 … 16
 *   apart, as the fold's first steps add them, and its last adds the other 32 lanes' zeros — so every score is nn's to the
 *   bit, and a row's pools a few dozen warps rather than a block each. */
#define NVIDIA_CUDA12__ZZPRIVATE_INDEX_POOLS 16u      /* pools a warp walks with one query */
#define NVIDIA_CUDA12__ZZPRIVATE_INDEX_WARPS 8u
#define NVIDIA_CUDA12__ZZPRIVATE_INDEX_BOUNDS __launch_bounds__(256)
static __global__ NVIDIA_CUDA12__ZZPRIVATE_INDEX_BOUNDS void nvidia_cuda12__zzprivate_index_scores(float* scores, uint64_t scores_row, const uint16_t* q,
                                                                                    const uint16_t* w, const uint16_t* pooled, uint64_t heads,
                                                                                    uint64_t len0, uint64_t kpool, uint64_t rows, float scale) {
    const uint32_t D = 128u, G = NVIDIA_CUDA12__ZZPRIVATE_INDEX_POOLS, WARP = NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint32_t h = (uint32_t)nn__silicon__lane() % WARP, warp = (uint32_t)nn__silicon__lane() / WARP;
    const uint64_t warps = nn__silicon__lanes() / WARP;
    const uint64_t groups = ((len0 + rows - 1u) / kpool + G - 1u) / G, items = rows * groups;
    const float per_head = 1.0f / nn__silicon__sqrtf((float)heads);
    for (uint64_t x = (uint64_t)nn__silicon__block() * warps + warp; x < items; x += (uint64_t)nn__silicon__blocks() * warps) {
        const uint64_t u = x / groups, base = (x - u * groups) * G, pools = (len0 + u) / kpool;
        const bool head = h < heads;
        uint32_t qv[64];
        const uint32_t* qw = (const uint32_t*)(const void*)(q + u * heads * D + (head ? h : 0u) * D);
#pragma unroll
        for (uint32_t i = 0u; i < 64u; ++i) qv[i] = qw[i];
        const float wh = head ? nn__silicon__half_to_float(w[u * heads + h]) : 0.0f;
        for (uint32_t i = 0u; i < G; ++i) {
            const uint64_t p = base + i;
            if (p >= pools) break;                                           /* the same for every lane of the warp */
            const uint32_t* kw = (const uint32_t*)(const void*)(pooled + p * D);
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
            for (uint32_t m = 1u; m < WARP; m <<= 1) part = part + __shfl_xor_sync(0xFFFFFFFFu, part, (int)m);
            part = part + 0.0f;                                              /* the fold's last step: the idle lanes' zeros */
            if (h == 0u) scores[u * scores_row + p] = part;
        }
    }
}
/* Heads of 128, at most 32 of them, queries and pools on 4 bytes; anything else is nn's. */
static bool nvidia_cuda12__zzprivate_index_scores_fit(const uint16_t* q, const uint16_t* pooled, uint64_t heads, uint64_t dim) {
    return dim == 128u && heads != 0u && heads <= NVIDIA_CUDA12__ZZPACKAGE_WARP && ((uint64_t)(uintptr_t)q) % 4u == 0u
        && ((uint64_t)(uintptr_t)pooled) % 4u == 0u;
}
static void nvidia_cuda12__zzprivate_index_scores_launch(float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w,
                                                         const uint16_t* pooled, uint64_t heads, uint64_t len0, uint64_t kpool,
                                                         uint64_t rows, float scale) {
    const uint64_t G = NVIDIA_CUDA12__ZZPRIVATE_INDEX_POOLS, W = NVIDIA_CUDA12__ZZPRIVATE_INDEX_WARPS;
    const uint64_t items = rows * (((len0 + rows - 1u) / kpool + G - 1u) / G);
    const uint64_t want = (items + W - 1u) / W;
    const uint32_t blocks = want == 0u ? 1u : want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &scores, &scores_row, &q, &w, &pooled, &heads, &len0, &kpool, &rows, &scale };
    const size_t sizes[] = { sizeof scores, sizeof scores_row, sizeof q, sizeof w, sizeof pooled, sizeof heads, sizeof len0, sizeof kpool,
                             sizeof rows, sizeof scale };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_index_scores, "nvidia_cuda12__zzprivate_index_scores", blocks,
                        (uint32_t)(W * NVIDIA_CUDA12__ZZPACKAGE_WARP), args, sizes, 10u);
}
static void nvidia_cuda12__override_index_scores(float* scores, const uint16_t* q, const uint16_t* w, const uint16_t* pooled,
                                                 uint64_t heads, uint64_t dim, uint64_t pools, float scale) {
    if (pools == 0u || !nvidia_cuda12__zzprivate_index_scores_fit(q, pooled, heads, dim)) {
        nn__index__zzabi_launch_scores(scores, q, w, pooled, heads, dim, pools, scale);
        return;
    }
    nvidia_cuda12__zzprivate_index_scores_launch(scores, 0u, q, w, pooled, heads, pools, 1u, 1u, scale);
}
static void nvidia_cuda12__override_index_scores_rows(float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w,
                                                      const uint16_t* pooled, uint64_t heads, uint64_t dim, uint64_t len0,
                                                      uint64_t kpool, uint64_t rows, float scale) {
    if (rows == 0u || kpool == 0u || (len0 + rows - 1u) / kpool == 0u || !nvidia_cuda12__zzprivate_index_scores_fit(q, pooled, heads, dim)) {
        nn__index__zzabi_launch_scores_rows(scores, scores_row, q, w, pooled, heads, dim, len0, kpool, rows, scale);
        return;
    }
    nvidia_cuda12__zzprivate_index_scores_launch(scores, scores_row, q, w, pooled, heads, len0, kpool, rows, scale);
}

/* A score as a key that orders as the score does: larger floats, larger keys — nn's. */
static __device__ __forceinline__ uint32_t nvidia_cuda12__zzprivate_index_key(float f) {
    union { float f; uint32_t u; } cast;
    cast.f = f;
    return (cast.u & 0x80000000u) != 0u ? ~cast.u : (cast.u | 0x80000000u);
}

/* ⭐ THE TOP `k` OF `pools` SCORES, AS POSITIONS — nn's `index_select` and `select_rows`: the `k`-th largest key found a bit
 *   at a time across the block, as nn's body finds it, then the pools above it and those equal to it by lowest index, in
 *   index order. nn's body walks every pool twice on one lane for that last part; here the first warp walks them 32 at a
 *   time: a pool's place among the chosen is the count of chosen pools below it in the warp (a ballot) plus those of the
 *   warp's steps before, and an equal key is chosen while the equal keys below it leave room — so `index` and `count` are
 *   nn's, to the bit. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_index_select_one(uint32_t* index, uint32_t* count, const float* scores,
                                                                                 uint64_t pools, uint64_t k, uint64_t kpool,
                                                                                 uint64_t tail_first, uint64_t tail) {
    const uint32_t WARP = NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint64_t t = nn__silicon__lane(), lanes = nn__silicon__lanes();
    uint32_t cut = 0u;
    if (pools > k) {
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t want = cut | (1u << (uint32_t)bit);
            float have = 0.0f;
            for (uint64_t p = t; p < pools; p += lanes) have += nvidia_cuda12__zzprivate_index_key(scores[p]) >= want ? 1.0f : 0.0f;
            if (nn__silicon__lanes_sum(have) >= (float)k) cut = want;
        }
    }
    if (t >= WARP) return;
    const uint32_t lane = (uint32_t)t, below = (1u << lane) - 1u;
    uint64_t above = 0u;
    if (pools > k) {
        for (uint64_t p = lane; p < pools; p += WARP) above += nvidia_cuda12__zzprivate_index_key(scores[p]) > cut ? 1u : 0u;
        for (uint32_t off = WARP / 2u; off > 0u; off >>= 1) above += __shfl_xor_sync(0xFFFFFFFFu, above, (int)off);
    }
    const uint64_t ties = pools > k ? k - above : 0u;
    uint64_t n = 0u, equal = 0u;                          /* the same in every lane: pools chosen, equal keys passed */
    for (uint64_t base = 0u; base < pools; base += WARP) {
        const uint64_t p = base + lane;
        const bool real = p < pools;
        const uint32_t key = real ? nvidia_cuda12__zzprivate_index_key(scores[p]) : 0u;
        const bool same = real && pools > k && key == cut;
        const uint32_t same_mask = __ballot_sync(0xFFFFFFFFu, same);
        const bool take = real && (pools <= k || key > cut || (same && equal + (uint64_t)__popc(same_mask & below) < ties));
        const uint32_t take_mask = __ballot_sync(0xFFFFFFFFu, take);
        if (take) {
            const uint64_t at = (n + (uint64_t)__popc(take_mask & below)) * kpool;
            for (uint64_t i = 0u; i < kpool; ++i) index[at + i] = (uint32_t)(p * kpool + i);
        }
        n += (uint64_t)__popc(take_mask);
        equal += (uint64_t)__popc(same_mask);
    }
    for (uint64_t i = lane; i < tail; i += WARP) index[n * kpool + i] = (uint32_t)(tail_first + i);
    if (lane == 0u) count[0] = (uint32_t)(n * kpool + tail);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_INDEX_BOUNDS void nvidia_cuda12__zzprivate_index_select(uint32_t* index, uint32_t* count, const float* scores,
                                                                                    uint64_t pools, uint64_t k, uint64_t kpool,
                                                                                    uint64_t tail_first, uint64_t tail) {
    if (nn__silicon__block() != 0u) return;
    nvidia_cuda12__zzprivate_index_select_one(index, count, scores, pools, k, kpool, tail_first, tail);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_INDEX_BOUNDS void nvidia_cuda12__zzprivate_index_select_rows(uint32_t* index, uint64_t index_row,
                                                                                         uint32_t* counts, const float* scores,
                                                                                         uint64_t scores_row, uint64_t k, uint64_t kpool,
                                                                                         uint64_t len0, uint64_t rows) {
    for (uint64_t u = nn__silicon__block(); u < rows; u += nn__silicon__blocks()) {
        const uint64_t len = len0 + u, pools = len / kpool;
        nvidia_cuda12__zzprivate_index_select_one(index + u * index_row, counts + u, scores + u * scores_row, pools, k, kpool,
                                                  pools * kpool, len - pools * kpool);
    }
}
static void nvidia_cuda12__override_index_select(uint32_t* index, uint32_t* count, const float* scores, uint64_t pools, uint64_t k,
                                                 uint64_t kpool, uint64_t tail_first, uint64_t tail) {
    void* args[] = { &index, &count, &scores, &pools, &k, &kpool, &tail_first, &tail };
    const size_t sizes[] = { sizeof index, sizeof count, sizeof scores, sizeof pools, sizeof k, sizeof kpool, sizeof tail_first, sizeof tail };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_index_select, "nvidia_cuda12__zzprivate_index_select", 1u,
                        NN__SILICON__LANES_MAX, args, sizes, 8u);
}
static void nvidia_cuda12__override_index_select_rows(uint32_t* index, uint64_t index_row, uint32_t* counts, const float* scores,
                                                      uint64_t scores_row, uint64_t k, uint64_t kpool, uint64_t len0, uint64_t rows) {
    if (rows == 0u || kpool == 0u) {
        nn__index__zzabi_launch_select_rows(index, index_row, counts, scores, scores_row, k, kpool, len0, rows);
        return;
    }
    const uint32_t blocks = rows < NN__KERNELS__BLOCKS_MAX ? (uint32_t)rows : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &index, &index_row, &counts, &scores, &scores_row, &k, &kpool, &len0, &rows };
    const size_t sizes[] = { sizeof index, sizeof index_row, sizeof counts, sizeof scores, sizeof scores_row, sizeof k, sizeof kpool,
                             sizeof len0, sizeof rows };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_index_select_rows, "nvidia_cuda12__zzprivate_index_select_rows", blocks,
                        NN__SILICON__LANES_MAX, args, sizes, 9u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_INDEX_CUH */
