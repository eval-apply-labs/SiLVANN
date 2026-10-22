#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_INDEX_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_INDEX_CUH
/* ══ index's DOOR, RUN FASTER — amd_rocm6_wave64 ══════════════════════════════════════════════════════════════════════
 * nn's indexer doors (`nn/cpu/opcodes/index__abi.cuh`), where this family covers them: the top `k` of the scores. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

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
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_index_select(uint32_t* index, uint32_t* count, const float* scores,
                                                                                                   uint64_t pools, uint64_t k, uint64_t kpool,
                                                                                                   uint64_t tail_first, uint64_t tail) {
    if (nn__silicon__block() != 0u) return;
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

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_INDEX_CUH */
