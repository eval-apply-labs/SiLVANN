#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_VECTOR_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_VECTOR_CUH
/* ══ vector's DOOR, RUN FASTER — amd_rocm6_wave64 ═════════════════════════════════════════════════════════════════════
 * nn's vector door (`nn/cpu/opcodes/vector__abi.cuh`), where this family covers it: the top k, a router's and a sampler's,
 * and a sampler's repetition penalty. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ THE `k` LARGEST IN ONE WAVE — nn's `top_k` is `k` rounds of a block's two folds, each fold eight waits of 256 lanes,
 *   for 256 router logits. Here one wave holds them all, `HELD` a lane in registers, each as one word that orders the way
 *   nn's rounds pick: its high half the value's order — a zero's two signs made one, as they compare equal — and its low
 *   half 0xFFFF less its index, so of equal values the first is the larger word; a NaN is 0, which no round takes. A round
 *   is then one integer max over the lane's words and six shuffles, and it picks what nn's round picks: the first index of
 *   the largest value left. Each pick's weight is nn's expression, on the lane numbered by its round, over a sum added in
 *   the rounds' order, as nn's is. ⛳ A pick's value is read back from its word, so a zero comes back +0 where nn's fold may
 *   keep -0; the weights read the values only as differences `v - v0`, where a zero's sign changes nothing, so the indices
 *   and the weights are nn's to the bit. `MEASURED` (rocprof over the bench): 256 logits top 8, 29 -> 6.2 us; 512 top 10, 42 -> 7.5 us. Up to 1,024
 *   values, `HELD` a template count so a router of 256 holds four a lane; anything else is nn's. No barrier: one wave. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_MOST 1024u     /* values a wave holds, sixteen a lane */
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_top_k_wave(uint64_t* values, uint64_t stride, uint16_t* weights,
                                                                      const uint16_t* x, uint64_t n, uint64_t k) {
    const uint32_t W = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, lane = (uint32_t)nn__silicon__lane();
    uint32_t key[HELD];
#pragma unroll
    for (uint32_t j = 0u; j < HELD; ++j) {
        const uint32_t i = lane + j * W;
        const uint32_t h = i < n ? (uint32_t)x[i] : 0x7E00u;              /* past `n`, as a NaN */
        const uint32_t z = (h & 0x7FFFu) == 0u ? 0u : h;                  /* -0 as +0 */
        const uint32_t order = (z & 0x8000u) != 0u ? (~z & 0xFFFFu) : (z | 0x8000u);
        key[j] = (h & 0x7FFFu) > 0x7C00u ? 0u : (order << 16) | (0xFFFFu - i);
    }
    float mine_v = -NN__KERNELS__INFINITY, v0 = -NN__KERNELS__INFINITY;  /* lane `r` keeps round `r`'s pick */
    uint64_t mine_i = n;
    for (uint32_t r = 0u; r < (uint32_t)k; ++r) {
        uint32_t best = 0u;
#pragma unroll
        for (uint32_t j = 0u; j < HELD; ++j) best = key[j] > best ? key[j] : best;
        for (uint32_t off = 1u; off < W; off <<= 1) {
            const uint32_t other = (uint32_t)__shfl_xor((int)best, (int)off);
            best = other > best ? other : best;
        }
#pragma unroll
        for (uint32_t j = 0u; j < HELD; ++j) key[j] = key[j] == best ? 0u : key[j];    /* taken: a word names one item */
        const uint32_t order = best >> 16;
        const uint32_t h = (order & 0x8000u) != 0u ? (order & 0x7FFFu) : (~order & 0xFFFFu);
        const float v = best != 0u ? nn__silicon__half_to_float((uint16_t)h) : -NN__KERNELS__INFINITY;
        if (r == 0u) v0 = v;
        if (lane == r) { mine_v = v; mine_i = best != 0u ? (uint64_t)(0xFFFFu - (best & 0xFFFFu)) : n; }
    }
    const float e = nn__silicon__expf(mine_v - v0);
    float sum = 0.0f;
    for (uint32_t r = 0u; r < (uint32_t)k; ++r) sum += __shfl(e, (int)r);
    if (lane < k) {
        values[lane * stride] = mine_i;
        unsigned int unreported = 0u;                                    /* nn's top_k raises no overflow either */
        weights[lane] = nn__kernels__zzabi_to_half(e / sum, &unreported);
    }
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_top_k_4(uint64_t* values, uint64_t stride, uint16_t* weights,
                                                                                              const uint16_t* x, uint64_t n, uint64_t k) {
    amd_rocm6_wave64__zzprivate_top_k_wave<4u>(values, stride, weights, x, n, k);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_top_k_8(uint64_t* values, uint64_t stride, uint16_t* weights,
                                                                                              const uint16_t* x, uint64_t n, uint64_t k) {
    amd_rocm6_wave64__zzprivate_top_k_wave<8u>(values, stride, weights, x, n, k);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_top_k_16(uint64_t* values, uint64_t stride, uint16_t* weights,
                                                                                               const uint16_t* x, uint64_t n, uint64_t k) {
    amd_rocm6_wave64__zzprivate_top_k_wave<16u>(values, stride, weights, x, n, k);
}
/* ⭐ THE `k` LARGEST OF A LONG VECTOR — a vocabulary's logits, for a sampler. nn's `k` rounds each read every value: 20
 *   rounds over 248,320 logits are 20 passes over half a megabyte, on one block. Here two passes do it, each read as
 *   `argmax` reads, sixteen bytes a lane with `DEPTH` reads in flight. Every value is a WORD that orders as nn's rounds
 *   pick — its high half the value's order (the two zeros one), its low half the index taken from all ones, so of equal
 *   values the first is the larger word; a NaN is 0 — and the words are distinct.
 *   ① each lane's first largest, as `argmax` finds it; the `k`-th largest of those 256 words is a BAR the `k`-th pick
 *     reaches — the `k` lanes' words at or above it are `k` distinct values' words;
 *   ② every word at or above the bar, into shared memory. Ties do not crowd it: of equal values only the indices up to
 *     the bar's reach it.
 *   Each held word's place is then the count of held words above it, and the first `k` places are nn's `k` rounds'
 *   picks, in its order. Each weight is nn's expression over a sum added in the rounds' order, as nn's is, so the
 *   indices and the weights are nn's to the bit (a zero comes back +0, which the weights read only as a difference, as
 *   the wave's above).
 *   ⛳ MORE THAN `HELD` WORDS AT THE BAR — fewer than `k` lanes holding any valid value, and many valid values among
 *   them — and the block runs nn's own rounds instead, so the answer never depends on the input, only the time.
 *   ⛳ ONE BLOCK, WAITING FOR ITSELF: the lanes' words, the bar, the held words, the picks' values. No other block. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_HELD   1024u   /* words the block holds at or above the bar */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_DEPTH  16u     /* reads of sixteen bytes a lane has in flight */
/* A half's word: its order above, the index taken from all ones below; 0 for a NaN, which no round takes. */
static __device__ inline uint64_t amd_rocm6_wave64__zzprivate_top_k_word(uint32_t h, uint32_t i) {
    const uint32_t z = (h & 0x7FFFu) == 0u ? 0u : h;
    const uint32_t order = (z & 0x8000u) != 0u ? (~z & 0xFFFFu) : (z | 0x8000u);
    return (h & 0x7FFFu) > 0x7C00u ? 0ull : ((uint64_t)order << 32) | (uint64_t)(0xFFFFFFFFu - i);
}
/* A word's value. */
static __device__ inline float amd_rocm6_wave64__zzprivate_top_k_value(uint64_t word) {
    const uint32_t order = (uint32_t)(word >> 32);
    return nn__silicon__half_to_float((uint16_t)((order & 0x8000u) != 0u ? (order & 0x7FFFu) : (~order & 0xFFFFu)));
}
/* The largest of a read's eight halves — a max passes over a NaN and leaves one only when all eight are. */
static __device__ inline float amd_rocm6_wave64__zzprivate_top_k_most(uint4 q) {
    const amd_rocm6_wave64__half2 m = __builtin_elementwise_max(
        __builtin_elementwise_max(__builtin_bit_cast(amd_rocm6_wave64__half2, q.x), __builtin_bit_cast(amd_rocm6_wave64__half2, q.y)),
        __builtin_elementwise_max(__builtin_bit_cast(amd_rocm6_wave64__half2, q.z), __builtin_bit_cast(amd_rocm6_wave64__half2, q.w)));
    return fmaxf((float)m.x, (float)m.y);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_top_k_long(uint64_t* values, uint64_t stride,
                                                                                                uint16_t* weights, const uint16_t* x,
                                                                                                uint64_t n, uint64_t k) {
    const uint32_t DEPTH = AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_DEPTH, HELD = AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_HELD;
    __shared__ uint64_t lane_word[NN__SILICON__LANES_MAX];
    __shared__ uint64_t held[AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_HELD];
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
        uint4 q[AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_DEPTH];
#pragma unroll
        for (uint32_t d = 0u; d < DEPTH; ++d) { const uint32_t g = g0 + d * lanes; q[d] = xg[g < groups ? g : g0]; }
#pragma unroll
        for (uint32_t d = 0u; d < DEPTH; ++d) {
            const uint32_t g = g0 + d * lanes;
            const float most = amd_rocm6_wave64__zzprivate_top_k_most(q[d]);
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
            const uint64_t word = amd_rocm6_wave64__zzprivate_top_k_word((w[e / 2u] >> (16u * (e % 2u))) & 0xFFFFu, 8u * best_g + e);
            mine = word > mine ? word : mine;
        }
    }
    for (uint32_t i = 8u * groups + t; i < (uint32_t)n; i += lanes) {   /* the last few, past the whole groups */
        const uint64_t word = amd_rocm6_wave64__zzprivate_top_k_word(x[i], i);
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
    const float floor_v = floor_word != 0ull ? amd_rocm6_wave64__zzprivate_top_k_value(floor_word) : -NN__KERNELS__INFINITY;
    /* ② every word at or above the bar: a group whose largest is below the bar's value is passed at once; the few others
     *   are read again, from the cache, a half at a time */
    for (uint32_t g0 = t; g0 < groups; g0 += DEPTH * lanes) {
        uint4 q[AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_DEPTH];
#pragma unroll
        for (uint32_t d = 0u; d < DEPTH; ++d) { const uint32_t g = g0 + d * lanes; q[d] = xg[g < groups ? g : g0]; }
        uint32_t near = 0u;
#pragma unroll
        for (uint32_t d = 0u; d < DEPTH; ++d)
            near |= (g0 + d * lanes < groups && amd_rocm6_wave64__zzprivate_top_k_most(q[d]) >= floor_v ? 1u : 0u) << d;
        while (near != 0u) {
            const uint32_t g = g0 + (uint32_t)__builtin_ctz(near) * lanes;
            near &= near - 1u;
            const uint4 r = xg[g];
            const uint32_t w[4] = { r.x, r.y, r.z, r.w };
#pragma unroll
            for (uint32_t e = 0u; e < 8u; ++e) {
                const uint64_t word = amd_rocm6_wave64__zzprivate_top_k_word((w[e / 2u] >> (16u * (e % 2u))) & 0xFFFFu, 8u * g + e);
                if (word == 0ull || word < floor_word) continue;
                const uint32_t at = atomicAdd(&count, 1u);
                if (at < HELD) held[at] = word;
            }
        }
    }
    for (uint32_t i = 8u * groups + t; i < (uint32_t)n; i += lanes) {
        const uint64_t word = amd_rocm6_wave64__zzprivate_top_k_word(x[i], i);
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
            picked[place] = amd_rocm6_wave64__zzprivate_top_k_value(word);
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

/* ⭐ THE REPETITION PENALTY WITH ITS IDS IN SHARED MEMORY — nn's body asks each id whether an earlier lane holds it by
 *   reading the earlier ids again, and where they are the program's nodes, registered with the card, each of those reads
 *   crosses the bus. Here the block reads each id once, into shared memory, waits, and asks there. The same ids are
 *   penalised, once each, as nn's are. Up to `PENALIZE_HELD` ids, one block; more are nn's. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_PENALIZE_HELD 2048u
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_penalize(uint16_t* x, uint64_t n, const uint64_t* ids,
                                                                                              uint64_t stride, uint64_t count, float penalty,
                                                                                              unsigned int* over) {
    __shared__ uint64_t held[AMD_ROCM6_WAVE64__ZZPRIVATE_PENALIZE_HELD];
    const uint32_t t = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    for (uint32_t j = t; j < (uint32_t)count; j += lanes) held[j] = ids[j * stride];
    __syncthreads();
    for (uint32_t j = t; j < (uint32_t)count; j += lanes) {
        const uint64_t id = held[j];
        bool again = id >= n;
        for (uint32_t q = 0u; q < j; ++q) again |= held[q] == id;        /* no early exit: the reads are independent */
        if (again) continue;
        const float v = nn__silicon__half_to_float(x[id]);
        x[id] = nn__kernels__zzabi_to_half(v > 0.0f ? v / penalty : v * penalty, over);
    }
}
static void amd_rocm6_wave64__override_vector_penalize(uint16_t* x, uint64_t n, const uint64_t* ids, uint64_t stride, uint64_t count,
                                                       float penalty, unsigned int* over) {
    if (count == 0u || count > AMD_ROCM6_WAVE64__ZZPRIVATE_PENALIZE_HELD) {
        nn__vector__zzabi_launch_penalize(x, n, ids, stride, count, penalty, over);
        return;
    }
    void* args[] = { &x, &n, &ids, &stride, &count, &penalty, &over };
    const size_t sizes[] = { sizeof x, sizeof n, sizeof ids, sizeof stride, sizeof count, sizeof penalty, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_penalize, "amd_rocm6_wave64__zzprivate_penalize", 1u,
                        NN__SILICON__LANES_MAX, args, sizes, 7u);
}
/* Up to 1,024 values in one wave, more in one block with `x` on sixteen bytes; one to `NN__VECTOR__TOP_K_MAX` picks, each
 * its own word (`stride` not 0); anything else is nn's. */
static void amd_rocm6_wave64__override_vector_top_k(uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x, uint64_t n,
                                                    uint64_t k) {
    if (n == 0u || n > NN__KERNELS__INDEX_EXACT || k == 0u || k > NN__VECTOR__TOP_K_MAX || stride == 0u
     || (n > AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_MOST && ((uint64_t)(uintptr_t)x) % 16u != 0u)) {
        nn__vector__zzabi_launch_top_k(values, stride, weights, x, n, k);
        return;
    }
    if (n > AMD_ROCM6_WAVE64__ZZPRIVATE_TOP_K_MOST) {
        void* args[] = { &values, &stride, &weights, &x, &n, &k };
        const size_t sizes[] = { sizeof values, sizeof stride, sizeof weights, sizeof x, sizeof n, sizeof k };
        nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_top_k_long, "amd_rocm6_wave64__zzprivate_top_k_long", 1u,
                            NN__SILICON__LANES_MAX, args, sizes, 6u);
        return;
    }
    const void* kernel = n <= 4u * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE ? (const void*)amd_rocm6_wave64__zzprivate_top_k_4
                       : n <= 8u * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE ? (const void*)amd_rocm6_wave64__zzprivate_top_k_8
                       :                                               (const void*)amd_rocm6_wave64__zzprivate_top_k_16;
    void* args[] = { &values, &stride, &weights, &x, &n, &k };
    const size_t sizes[] = { sizeof values, sizeof stride, sizeof weights, sizeof x, sizeof n, sizeof k };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_top_k", 1u, AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 6u);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_VECTOR_CUH */
