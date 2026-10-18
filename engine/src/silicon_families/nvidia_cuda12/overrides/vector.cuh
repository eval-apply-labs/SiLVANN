#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_VECTOR_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_VECTOR_CUH
/* ══ vector's DOOR, RUN FASTER — nvidia_cuda12 ════════════════════════════════════════════════════════════════════════
 * nn's vector door (`nn/cpu/opcodes/vector__abi.cuh`), where this family covers it: a sampler's top k over a vocabulary. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ THE `k` LARGEST OF A LONG VECTOR — a vocabulary's logits, for a sampler — amd_rocm6_wave64's, on this family's lanes.
 *   nn's `k` rounds each read every value: 20 rounds over 248,320 logits are 20 passes over half a megabyte, on one block.
 *   Here two passes do it, each read as `argmax` reads, sixteen bytes a lane with `DEPTH` reads in flight. Every value is
 *   a WORD that orders as nn's rounds pick — its high half the value's order (the two zeros one), its low half the index
 *   taken from all ones, so of equal values the first is the larger word; a NaN is 0 — and the words are distinct.
 *   ① each lane's first largest, as `argmax` finds it — a read's largest is three packed half maxima (`__hmax2`, which
 *     passes over a NaN) and one more; the `k`-th largest of those 256 words is a BAR the `k`-th pick reaches — the `k`
 *     lanes' words at or above it are `k` distinct values' words;
 *   ② every word at or above the bar, into shared memory. Ties do not crowd it: of equal values only the indices up to
 *     the bar's reach it.
 *   Each held word's place is then the count of held words above it, and the first `k` places are nn's `k` rounds'
 *   picks, in its order. Each weight is nn's expression over a sum added in the rounds' order, as nn's is, so the
 *   indices and the weights are nn's to the bit (a zero comes back +0, which the weights read only as a difference).
 *   ⛳ MORE THAN `HELD` WORDS AT THE BAR — fewer than `k` lanes holding any valid value, and many valid values among
 *   them — and the block runs nn's own rounds instead, so the answer never depends on the input, only the time.
 *   ⛳ ONE BLOCK, WAITING FOR ITSELF: the lanes' words, the bar, the held words, the picks' values. No other block. */
#define NVIDIA_CUDA12__ZZPRIVATE_TOP_K_SHORT  1024u    /* up to this many values, nn's rounds: a router's few */
#define NVIDIA_CUDA12__ZZPRIVATE_TOP_K_HELD   1024u    /* words the block holds at or above the bar */
#define NVIDIA_CUDA12__ZZPRIVATE_TOP_K_DEPTH  16u      /* reads of sixteen bytes a lane has in flight */
#define NVIDIA_CUDA12__ZZPRIVATE_TOP_K_BOUNDS __launch_bounds__(256)
/* A half's word: its order above, the index taken from all ones below; 0 for a NaN, which no round takes. */
static __device__ inline uint64_t nvidia_cuda12__zzprivate_top_k_word(uint32_t h, uint32_t i) {
    const uint32_t z = (h & 0x7FFFu) == 0u ? 0u : h;
    const uint32_t order = (z & 0x8000u) != 0u ? (~z & 0xFFFFu) : (z | 0x8000u);
    return (h & 0x7FFFu) > 0x7C00u ? 0ull : ((uint64_t)order << 32) | (uint64_t)(0xFFFFFFFFu - i);
}
/* A word's value. */
static __device__ inline float nvidia_cuda12__zzprivate_top_k_value(uint64_t word) {
    const uint32_t order = (uint32_t)(word >> 32);
    return nn__silicon__half_to_float((uint16_t)((order & 0x8000u) != 0u ? (order & 0x7FFFu) : (~order & 0xFFFFu)));
}
static __device__ inline __half2 nvidia_cuda12__zzprivate_top_k_h2(uint32_t w) {
    __half2 h;
    *(uint32_t*)(void*)&h = w;
    return h;
}
/* The largest of a read's eight halves — a max passes over a NaN and leaves one only when all eight are. */
static __device__ inline float nvidia_cuda12__zzprivate_top_k_most(uint4 q) {
    const __half2 m = __hmax2(__hmax2(nvidia_cuda12__zzprivate_top_k_h2(q.x), nvidia_cuda12__zzprivate_top_k_h2(q.y)),
                              __hmax2(nvidia_cuda12__zzprivate_top_k_h2(q.z), nvidia_cuda12__zzprivate_top_k_h2(q.w)));
    return fmaxf(__low2float(m), __high2float(m));
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_TOP_K_BOUNDS void nvidia_cuda12__zzprivate_top_k_long(uint64_t* values, uint64_t stride,
                                                                                                 uint16_t* weights, const uint16_t* x,
                                                                                                 uint64_t n, uint64_t k) {
    const uint32_t DEPTH = NVIDIA_CUDA12__ZZPRIVATE_TOP_K_DEPTH, HELD = NVIDIA_CUDA12__ZZPRIVATE_TOP_K_HELD;
    __shared__ uint64_t lane_word[NN__SILICON__LANES_MAX];
    __shared__ uint64_t held[NVIDIA_CUDA12__ZZPRIVATE_TOP_K_HELD];
    __shared__ float    picked[NN__VECTOR__TOP_K_MAX];
    __shared__ uint64_t bar;
    __shared__ uint32_t count;
    const uint32_t t = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    const uint32_t groups = (uint32_t)(n / 8u);                          /* n is under 2^24 here */
    const uint4* xg = (const uint4*)(const void*)x;
    /* ① the lane's first largest: its first group holding its largest, as argmax keeps it, then that group's first half */
    uint32_t best_g = groups;
    float best_v = -NN__KERNELS__INFINITY;
    for (uint32_t g0 = t; g0 < groups; g0 += DEPTH * lanes) {
        uint4 q[NVIDIA_CUDA12__ZZPRIVATE_TOP_K_DEPTH];
#pragma unroll
        for (uint32_t d = 0u; d < DEPTH; ++d) { const uint32_t g = g0 + d * lanes; q[d] = xg[g < groups ? g : g0]; }
#pragma unroll
        for (uint32_t d = 0u; d < DEPTH; ++d) {
            const uint32_t g = g0 + d * lanes;
            const float most = nvidia_cuda12__zzprivate_top_k_most(q[d]);
            const bool take = g < groups && ((most > best_v) | ((best_g == groups) & (most == most)));
            best_v = take ? most : best_v;
            best_g = take ? g : best_g;
        }
    }
    uint64_t mine = 0ull;
    if (best_g != groups) {
        const uint4 q = xg[best_g];
        const uint32_t w[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            const uint64_t word = nvidia_cuda12__zzprivate_top_k_word((w[e / 2u] >> (16u * (e % 2u))) & 0xFFFFu, 8u * best_g + e);
            mine = word > mine ? word : mine;
        }
    }
    for (uint32_t i = 8u * groups + t; i < (uint32_t)n; i += lanes) {   /* the last few, past the whole groups */
        const uint64_t word = nvidia_cuda12__zzprivate_top_k_word(x[i], i);
        mine = word > mine ? word : mine;
    }
    lane_word[t] = mine;
    if (t == 0u) { count = 0u; bar = 0ull; }
    __syncthreads();
    /* the bar: the `k`-th largest of the lanes' words — distinct, so exactly one lane has `k - 1` above it — or 0, every
     * valid value, where fewer than `k` lanes hold one */
    uint32_t above = 0u;
    for (uint32_t j = 0u; j < lanes; ++j) above += lane_word[j] > mine ? 1u : 0u;
    if (mine != 0ull && above == (uint32_t)k - 1u) bar = mine;
    __syncthreads();
    const uint64_t floor_word = bar;
    const float floor_v = floor_word != 0ull ? nvidia_cuda12__zzprivate_top_k_value(floor_word) : -NN__KERNELS__INFINITY;
    /* ② every word at or above the bar: a group whose largest is below the bar's value is passed at once; the few others
     *   are read again, from the cache, a half at a time */
    for (uint32_t g0 = t; g0 < groups; g0 += DEPTH * lanes) {
        uint4 q[NVIDIA_CUDA12__ZZPRIVATE_TOP_K_DEPTH];
#pragma unroll
        for (uint32_t d = 0u; d < DEPTH; ++d) { const uint32_t g = g0 + d * lanes; q[d] = xg[g < groups ? g : g0]; }
        uint32_t near = 0u;
#pragma unroll
        for (uint32_t d = 0u; d < DEPTH; ++d)
            near |= (g0 + d * lanes < groups && nvidia_cuda12__zzprivate_top_k_most(q[d]) >= floor_v ? 1u : 0u) << d;
        while (near != 0u) {
            const uint32_t g = g0 + (uint32_t)(__ffs((int)near) - 1) * lanes;
            near &= near - 1u;
            const uint4 r = xg[g];
            const uint32_t w[4] = { r.x, r.y, r.z, r.w };
#pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e) {
                const uint64_t word = nvidia_cuda12__zzprivate_top_k_word((w[e / 2u] >> (16u * (e % 2u))) & 0xFFFFu, 8u * g + e);
                if (word == 0ull || word < floor_word) continue;
                const uint32_t at = atomicAdd(&count, 1u);
                if (at < HELD) held[at] = word;
            }
        }
    }
    for (uint32_t i = 8u * groups + t; i < (uint32_t)n; i += lanes) {
        const uint64_t word = nvidia_cuda12__zzprivate_top_k_word(x[i], i);
        if (word == 0ull || word < floor_word) continue;
        const uint32_t at = atomicAdd(&count, 1u);
        if (at < HELD) held[at] = word;
    }
    __syncthreads();
    const uint32_t c = count;
    if (c > HELD) {                                                      /* the same on every lane: nn's rounds */
        nn__vector__zzabi_body_top_k(values, stride, weights, x, n, k);
        return;
    }
    /* each held word's place among them; the first `k` places are the picks */
    for (uint32_t j = t; j < c; j += lanes) {
        const uint64_t word = held[j];
        uint32_t place = 0u;
        for (uint32_t q = 0u; q < c; ++q) place += held[q] > word ? 1u : 0u;
        if (place < (uint32_t)k) {
            values[place * stride] = (uint64_t)(0xFFFFFFFFu - (uint32_t)word);
            picked[place] = nvidia_cuda12__zzprivate_top_k_value(word);
        }
    }
    for (uint32_t r = c + t; r < (uint32_t)k; r += lanes) {              /* fewer valid values than picks: nn's none */
        values[r * stride] = n;
        picked[r] = -NN__KERNELS__INFINITY;
    }
    __syncthreads();
    if (t < (uint32_t)k) {
        float sum = 0.0f;
        for (uint32_t r = 0u; r < (uint32_t)k; ++r) sum += nn__silicon__expf(picked[r] - picked[0]);
        unsigned int unreported = 0u;                                    /* nn's top_k raises no overflow either */
        weights[t] = nn__kernels__zzabi_to_half(nn__silicon__expf(picked[t] - picked[0]) / sum, &unreported);
    }
}
/* More than `TOP_K_SHORT` values with `x` on sixteen bytes; one to `NN__VECTOR__TOP_K_MAX` picks, each its own word
 * (`stride` not 0); anything else is nn's. */
static void nvidia_cuda12__override_vector_top_k(uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x, uint64_t n,
                                                 uint64_t k) {
    if (n <= NVIDIA_CUDA12__ZZPRIVATE_TOP_K_SHORT || n > NN__KERNELS__INDEX_EXACT || k == 0u || k > NN__VECTOR__TOP_K_MAX
     || stride == 0u || ((uint64_t)(uintptr_t)x) % 16u != 0u) {
        nn__vector__zzabi_launch_top_k(values, stride, weights, x, n, k);
        return;
    }
    void* args[] = { &values, &stride, &weights, &x, &n, &k };
    const size_t sizes[] = { sizeof values, sizeof stride, sizeof weights, sizeof x, sizeof n, sizeof k };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_top_k_long, "nvidia_cuda12__zzprivate_top_k_long", 1u,
                        NN__SILICON__LANES_MAX, args, sizes, 6u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_VECTOR_CUH */
