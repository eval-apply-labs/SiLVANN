#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_RESULT_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_RESULT_CUH
/* ══ result's DOOR, RUN FASTER — amd_rocm6_wave64 ═════════════════════════════════════════════════════════════════════
 * nn's result door (`nn/cpu/opcodes/result__abi.cuh`), where this family covers it: argmax. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ══ ⭐ THE SMALL ONES — argmax, the router's top k and two residuals' merge ═════════════════════════════
 * Each is a launch of a few microseconds' work that nn's body takes far longer over, because one lane walks what many
 * could: `MEASURED` (rocprof over a decode of the 27B and the 35B) argmax ~450-510 us a call, top_k ~30 us, the merge
 * ~80-90 us at 2,035 positions. What each answers is nn's, to the bit. */

/* ⭐ argmax OVER EVERY LANE OF ONE BLOCK, SIXTEEN BYTES A READ. nn's body gives each of 256 lanes every 256th half, one
 *   two-byte read and one step of its rule at a time — ~970 of each a lane over a vocabulary of 248,320. Here each lane
 *   reads eight halves at once with `DEPTH` reads in flight and keeps, of its groups of eight, the first holding its
 *   largest — a group's largest is three packed half maxima and one more, which pass over a NaN — then reads that one
 *   group again for its first half equal to it. That is nn's rule over the lane's halves (its first valid one, then only a
 *   larger), and the block's answer is found exactly as nn's is: the largest of the lanes' largests, then the smallest
 *   index holding it, through the family's fold. So the answer is nn's — the first index of the largest, a NaN never
 *   chosen, 0 when every item is one. `MEASURED` (rocprof over the bench, 248,320 halves): 262 -> 18 us. Covers `n` up to
 *   nn's exact-index bound with `x` on 16 bytes; anything else is the generic door.
 * ⛳ ONE BLOCK, AS nn's. With one wave a SIMD every instruction is a turn, so the select is written without a branch:
 *   the select costs ~17 instructions a group where nn's rule written as a branch costs ~34 (the ISA). More blocks would need a combine across
 *   them, a mechanism this file has none of, for what is left: `REASONED`, under 15 us a token. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_ARGMAX_DEPTH 16u     /* reads of sixteen bytes a lane has in flight */
static __device__ inline void amd_rocm6_wave64__zzprivate_argmax_take(float v, uint32_t i, uint32_t none, float* best_v, uint32_t* best_i) {
    if (*best_i == none ? v == v : v > *best_v) { *best_v = v; *best_i = i; }
}
/* The eight halves of a read, as floats, `e` the half's place in it. */
static __device__ inline float amd_rocm6_wave64__zzprivate_half8_at(uint4 q, uint32_t e) {
    const uint32_t w = e < 2u ? q.x : e < 4u ? q.y : e < 6u ? q.z : q.w;
    return nn__silicon__half_to_float((uint16_t)(w >> (16u * (e % 2u))));
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_argmax(uint64_t* out, uint64_t stamp, const uint16_t* x,
                                                                                             uint64_t n) {
    const uint32_t DEPTH = AMD_ROCM6_WAVE64__ZZPRIVATE_ARGMAX_DEPTH;
    const uint32_t t = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    const uint32_t none = (uint32_t)n, groups = (uint32_t)(n / 8u);      /* n is under 2^24 here */
    const uint4* xg = (const uint4*)(const void*)x;
    uint32_t best_g = groups;                                            /* the lane's first group holding its largest */
    float best_v = -NN__KERNELS__INFINITY;
    for (uint32_t g0 = t; g0 < groups; g0 += DEPTH * lanes) {
        uint4 q[AMD_ROCM6_WAVE64__ZZPRIVATE_ARGMAX_DEPTH];
#pragma unroll
        for (uint32_t k = 0u; k < DEPTH; ++k) {                          /* every read issued before the first is used */
            const uint32_t g = g0 + k * lanes;
            q[k] = xg[g < groups ? g : g0];
        }
#pragma unroll
        for (uint32_t k = 0u; k < DEPTH; ++k) {
            const uint32_t g = g0 + k * lanes;
            /* the group's largest, two halves at a time: a max passes over a NaN and leaves one only when both are */
            const amd_rocm6_wave64__half2 m = __builtin_elementwise_max(
                __builtin_elementwise_max(__builtin_bit_cast(amd_rocm6_wave64__half2, q[k].x), __builtin_bit_cast(amd_rocm6_wave64__half2, q[k].y)),
                __builtin_elementwise_max(__builtin_bit_cast(amd_rocm6_wave64__half2, q[k].z), __builtin_bit_cast(amd_rocm6_wave64__half2, q[k].w)));
            const float most = fmaxf((float)m.x, (float)m.y);
            /* nn's rule, without a branch: a larger, or the lane's first valid group */
            const bool take = g < groups && ((most > best_v) | ((best_g == groups) & (most == most)));
            best_v = take ? most : best_v;
            best_g = take ? g : best_g;
        }
    }
    uint32_t best_i = none;
    if (best_g != groups) {                                              /* the group's first half equal to its largest */
        const uint4 q = xg[best_g];
        for (uint32_t e = 8u; e-- > 0u; )
            if (amd_rocm6_wave64__zzprivate_half8_at(q, e) == best_v) best_i = 8u * best_g + e;
    }
    for (uint32_t i = 8u * groups + t; i < none; i += lanes)             /* the last few, past the whole groups */
        amd_rocm6_wave64__zzprivate_argmax_take(nn__silicon__half_to_float(x[i]), i, none, &best_v, &best_i);
    const float top = nn__silicon__lanes_max(best_v);
    const float first = nn__silicon__lanes_max(best_i != none && best_v == top ? -(float)best_i : -NN__KERNELS__INFINITY);
    if (t == 0u) *out = stamp | (uint64_t)(uint32_t)(first == -NN__KERNELS__INFINITY ? 0ull : (uint64_t)(-first));
}
static void amd_rocm6_wave64__override_argmax(uint64_t* out, uint64_t stamp, const uint16_t* x, uint64_t n) {
    if (n == 0u || n > NN__KERNELS__INDEX_EXACT || ((uint64_t)(uintptr_t)x) % 16u != 0u) {
        nn__argmax__zzabi_launch(out, stamp, x, n);
        return;
    }
    void* args[] = { &out, &stamp, &x, &n };
    const size_t sizes[] = { sizeof out, sizeof stamp, sizeof x, sizeof n };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_argmax, "amd_rocm6_wave64__zzprivate_argmax", 1u,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 4u);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_RESULT_CUH */
