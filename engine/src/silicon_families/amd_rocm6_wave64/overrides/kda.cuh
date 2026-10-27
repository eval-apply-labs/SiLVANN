#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_KDA_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_KDA_CUH
/* ══ KDA's DOORS, RUN FASTER — amd_rocm6_wave64 ═══════════════════════════════════════════════════════════════════════
 * nn's KDA doors, where this family covers them: the steps over a prompt's rows with their decays made before, at a
 * head of 128. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* Float `i` of `at`, read by the scalar unit, on `zzpackage_scalar_half`'s premise. */
static __device__ inline float amd_rocm6_wave64__zzprivate_scalar_float(const float* at, uint32_t i) {
    return ((const __attribute__((address_space(4))) float*)(const void*)at)[i];
}

/* ⭐ KDA's STEPS OVER A PROMPT'S ROWS AT A HEAD OF 128 — nn's `kda_steps_decayed`, a block a head and a lane a column of its
 *   state, the same operations in the same order as nn's card body, so the two agree to the bit. What is this family's
 *   own, as in deltanet's steps:
 *   · the column stays in the lane's registers for every row of the call, read once and written once, where nn's body
 *     reads and writes it through memory an element at a time, each read waited for before the next
 *   · `q`, `k` and the row's decays are the same for every lane of the block, so they are read by the scalar unit
 *   · the norms of `q` and `k` are summed inside each wave, in the fold's order, so only the output's norm waits for the
 *     block
 *   · the block is the two waves that own a column; nn's door gives a head 256 lanes, the upper 128 adding zeros to the
 *     fold, which changes nothing in the order the rest are added in */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_KDA_LANES   128u     /* a lane a column of a head of 128 */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_KDA_BOUNDS  __launch_bounds__(AMD_ROCM6_WAVE64__ZZPRIVATE_KDA_LANES)
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_KDA_BOUNDS void amd_rocm6_wave64__zzprivate_kda_steps_decayed128(
    float* S, const uint16_t* conved, const float* eg, const uint16_t* b, const uint16_t* gate, const uint16_t* w, uint16_t* out,
    uint64_t heads, float eps, uint64_t rows, unsigned int* over) {
    const uint32_t D = 128u;
    const uint32_t j = (uint32_t)nn__silicon__lane();
    const uint64_t width = heads * D;
    const uint16_t* qs = conved;
    const uint16_t* ks = conved + rows * width;
    const uint16_t* vs = conved + 2ull * rows * width;
    for (uint64_t h = nn__silicon__block(); h < heads; h += nn__silicon__blocks()) {
        float* Sh = S + h * D * D;
        const uint32_t at = j * (uint32_t)sizeof(float);
        float col[128];
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) col[i] = *(const float*)((const char*)(Sh + i * D) + at);
        for (uint64_t t = 0ull; t < rows; ++t) {
            const uint64_t r = t * width + h * D;
            const uint16_t* q = qs + r;
            const uint16_t* k = ks + r;
            const float* e = eg + r;
            const float beta = 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[t * heads + h])));
            const float inv_q = 1.0f / nn__silicon__sqrtf(amd_rocm6_wave64__zzpackage_squares128(q) + 1e-6f) / nn__silicon__sqrtf((float)D);
            const float inv_k = 1.0f / nn__silicon__sqrtf(amd_rocm6_wave64__zzpackage_squares128(k) + 1e-6f);
            float kv = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) {
                col[i] = col[i] * amd_rocm6_wave64__zzprivate_scalar_float(e, i);
                kv += col[i] * amd_rocm6_wave64__zzpackage_scalar_half(k, i) * inv_k;
            }
            const float delta = (nn__silicon__half_to_float(vs[r + j]) - kv) * beta;
            float o = 0.0f;
#pragma unroll
            for (uint32_t c0 = 0u; c0 < D; c0 += 32u) {
                const uint16_t* k2 = amd_rocm6_wave64__zzpackage_afresh(k);
                const uint16_t* q2 = amd_rocm6_wave64__zzpackage_afresh(q);
#pragma unroll
                for (uint32_t i = c0; i < c0 + 32u; ++i) {
                    const float s = col[i] + amd_rocm6_wave64__zzpackage_scalar_half(k2, i) * inv_k * delta;
                    col[i] = s;
                    o += s * amd_rocm6_wave64__zzpackage_scalar_half(q2, i);
                }
            }
            const uint16_t first = nn__kernels__zzabi_to_half(o * inv_q, over);   /* rounded to a half, as nn's readout is */
            const float oj = nn__silicon__half_to_float(first);
            const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oj * oj) / (float)D + eps);
            const float gz = nn__silicon__half_to_float(gate[r + j]);
            out[r + j] = nn__kernels__zzabi_to_half(oj * scale * nn__silicon__half_to_float(w[j]) / (1.0f + nn__silicon__expf(-gz)), over);
        }
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) *(float*)((char*)(Sh + i * D) + at) = col[i];
    }
}
/* A head of 128, `conved` on 4 bytes for the scalar reads, and `out` and the state apart from what the steps read;
 * anything else is nn's. */
static void amd_rocm6_wave64__override_kda_steps_decayed(float* S, const uint16_t* conved, const float* eg, const uint16_t* b,
                                                         const uint16_t* gate, const uint16_t* w, uint16_t* out, uint64_t heads,
                                                         uint64_t head_dim, float eps, uint64_t rows, unsigned int* over) {
    const uint64_t width = heads * head_dim;
    const uint64_t c0 = (uint64_t)(uintptr_t)conved, c1 = c0 + 3u * rows * width * 2u;
    const uint64_t e0 = (uint64_t)(uintptr_t)eg, e1 = e0 + rows * width * 4u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + rows * width * 2u;
    const uint64_t s0 = (uint64_t)(uintptr_t)S, s1 = s0 + heads * head_dim * head_dim * 4u;
    const bool apart = (o1 <= c0 || c1 <= o0) && (s1 <= c0 || c1 <= s0) && (o1 <= e0 || e1 <= o0) && (s1 <= e0 || e1 <= s0);
    if (head_dim != 128u || heads == 0u || c0 % 4u != 0u || e0 % 4u != 0u || !apart) {
        nn__kda__zzabi_launch_steps_decayed(S, conved, eg, b, gate, w, out, heads, head_dim, eps, rows, over);
        return;
    }
    const uint32_t blocks = heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &S, &conved, &eg, &b, &gate, &w, &out, &heads, &eps, &rows, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof eg, sizeof b, sizeof gate, sizeof w, sizeof out, sizeof heads,
                             sizeof eps, sizeof rows, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_kda_steps_decayed128, "amd_rocm6_wave64__zzprivate_kda_steps_decayed128",
                        blocks, AMD_ROCM6_WAVE64__ZZPRIVATE_KDA_LANES, args, sizes, 11u);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_KDA_CUH */
