#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_DELTANET_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_DELTANET_CUH
/* ══ deltanet's DOORS, RUN FASTER — amd_rocm6_wave64 ══════════════════════════════════════════════════════════════════
 * nn's deltanet doors (`nn/cpu/opcodes/deltanet__abi.cuh`), where this family covers them: the step at a head of
 * 128, over a prompt's rows and one position at a time. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ ONE DELTANET STEP AT A HEAD OF 128 — nn's `deltanet_step`, a block a head and a lane a column of its state, the same
 *   operations in the same order as nn's card body, so a position decoded here and the prompt's rows above agree to the
 *   bit. What is this family's own:
 *   · the column is read into the lane's registers in one go, every load in flight together, where nn's body reads
 *     and writes it through memory an element at a time, each read waited for before the next
 *   · `q` and `k` are the same for every lane of the block, so the two passes read them by the scalar unit, into
 *     registers the lanes share, rather than every lane into its own
 *   · the norms of `q` and `k` are summed inside each wave, in the fold's order, so only the output's norm waits for
 *     the block
 *   · the block is the two waves that own a column; nn's door gives a head 256 lanes, the upper 128 adding zeros to the
 *     fold, which changes nothing in the order the rest are added in
 *   `MEASURED` (test/src_gemv_27b_bench.cpp, one MI50, rocprof): a step at the 27B's 48 heads 102 -> 14.7 us warm and
 *   back to back, 94 -> 15.5 with the L2 cold as a model meets it; the 35B's 32 97 -> 12.9 (cold 90 -> 13.6); the 122B's
 *   64 113 -> 18.4-18.7 (cold 105 -> 18.9). Output and state byte-identical to nn's card body, and to the rows kernel
 *   below, at one row and over many positions with `conved` rewritten between them. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_STEP_LANES  128u     /* a lane a column of a head of 128 */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_STEP_BOUNDS __launch_bounds__(AMD_ROCM6_WAVE64__ZZPRIVATE_STEP_LANES)
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_STEP_BOUNDS void amd_rocm6_wave64__zzprivate_deltanet_step128(
    float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g, const uint16_t* w, uint16_t* out,
    uint64_t k_heads, uint64_t v_heads, unsigned int* over) {
    const uint32_t D = 128u;
    const uint32_t j = (uint32_t)nn__silicon__lane();
    const uint64_t rep = v_heads / k_heads;
    for (uint64_t h = nn__silicon__block(); h < v_heads; h += nn__silicon__blocks()) {
        float* Sh = S + h * D * D;
        const uint16_t* q = conved + (h / rep) * D;
        const uint16_t* k = conved + k_heads * D + (h / rep) * D;
        const uint16_t* v = conved + 2ull * k_heads * D + h * D;
        uint16_t* oh = out + h * D;
        /* a row's address is the block's and a lane's place in it a 32-bit offset, so the loads and the stores below take one
         * register a lane for their addresses rather than two a row */
        const uint32_t at = j * (uint32_t)sizeof(float);
        float col[128];
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) col[i] = *(const float*)((const char*)(Sh + i * D) + at);
        const float b = nn__silicon__half_to_float(beta[h]), g_h = nn__silicon__half_to_float(g[h]);
        const float inv_q = 1.0f / nn__silicon__sqrtf(amd_rocm6_wave64__zzpackage_squares128(q) + 1e-6f) / nn__silicon__sqrtf((float)D);
        const float inv_k = 1.0f / nn__silicon__sqrtf(amd_rocm6_wave64__zzpackage_squares128(k) + 1e-6f);
        const float decay = nn__silicon__expf(g_h);
        float kv = 0.0f;
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) kv += col[i] * amd_rocm6_wave64__zzpackage_scalar_half(k, i);
        const float delta = (nn__silicon__half_to_float(v[j]) - decay * kv * inv_k) * b;
        const uint16_t* k2 = amd_rocm6_wave64__zzpackage_afresh(k);
        float o = 0.0f;
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) {
            const float sn = decay * col[i] + amd_rocm6_wave64__zzpackage_scalar_half(k2, i) * inv_k * delta;
            *(float*)((char*)(Sh + i * D) + at) = sn;
            o += sn * amd_rocm6_wave64__zzpackage_scalar_half(q, i);
        }
        const uint16_t first = nn__kernels__zzabi_to_half(o * inv_q, over);   /* rounded to a half, as nn's readout is */
        const float oj = nn__silicon__half_to_float(first);
        const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oj * oj) / (float)D + 1e-6f);
        const float zr = nn__silicon__half_to_float(z[h * D + j]);
        const float zc = (zr < -88.0f) ? -88.0f : ((zr > 88.0f) ? 88.0f : zr);
        oh[j] = nn__kernels__zzabi_to_half(oj * scale * nn__silicon__half_to_float(w[j]) * (zc / (1.0f + nn__silicon__expf(-zc))), over);
    }
}
/* A head of 128, whole key groups, `conved` on 4 bytes for the scalar reads, and `out` and the state apart from what the
 * step reads; anything else is nn's. */
static void amd_rocm6_wave64__override_deltanet_step(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta,
                                                     const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                     uint64_t v_heads, uint64_t head_dim, unsigned int* over) {
    const uint64_t c0 = (uint64_t)(uintptr_t)conved, c1 = c0 + (2u * k_heads + v_heads) * head_dim * 2u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + v_heads * head_dim * 2u;
    const uint64_t s0 = (uint64_t)(uintptr_t)S, s1 = s0 + v_heads * head_dim * head_dim * 4u;
    const bool apart = (o1 <= c0 || c1 <= o0) && (s1 <= c0 || c1 <= s0);
    if (head_dim != 128u || k_heads == 0u || v_heads % k_heads != 0u || ((uint64_t)(uintptr_t)conved) % 4u != 0u || !apart) {
        nn__deltanet__zzabi_launch_step(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, over);
        return;
    }
    const uint32_t blocks = v_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)v_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &S, &conved, &z, &beta, &g, &w, &out, &k_heads, &v_heads, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof z, sizeof beta, sizeof g, sizeof w, sizeof out, sizeof k_heads,
                             sizeof v_heads, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_deltanet_step128, "amd_rocm6_wave64__zzprivate_deltanet_step128", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_STEP_LANES, args, sizes, 10u);
}

/* ⭐ THE DELTANET OVER A PROMPT'S ROWS AT A HEAD OF 128 — nn's `deltanet_steps`: the step above a row at a time, its
 *   operations in its order, so a prompt read as rows and the same prompt a position at a time agree to the bit. The
 *   column stays in the lane's registers for every row of the call, read once and written once; `q` and `k` are read by
 *   the scalar unit, as the step reads them — held in every lane beside the column, they spilled the column to scratch
 *   (315 registers, 1,212 bytes a lane: 15.7 us a row at any number of heads). `MEASURED` gtasanandrea 2026-10-16
 *   (BENCH_DELTANET_ONLY=1 BENCH_BURST=20 scripts/src_gemv_27b_bench.sh): a 256-row call at the 27B's 48 heads 4.03 ->
 *   2.10 ms, at 12 1.92; every output the same bytes. It still spills 44 registers in the state's update. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_STEP_BOUNDS void amd_rocm6_wave64__zzprivate_deltanet_steps128(
    float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g, const uint16_t* w, uint16_t* out,
    uint64_t k_heads, uint64_t v_heads, uint64_t rows, unsigned int* over) {
    const uint32_t D = 128u;
    const uint32_t j = (uint32_t)nn__silicon__lane();
    const uint64_t rep = v_heads / k_heads, ch = (2ull * k_heads + v_heads) * D, vd = v_heads * D;
    for (uint64_t h = nn__silicon__block(); h < v_heads; h += nn__silicon__blocks()) {
        float* Sh = S + h * D * D;
        const uint32_t at = j * (uint32_t)sizeof(float);
        float col[128];
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) col[i] = *(const float*)((const char*)(Sh + i * D) + at);
        for (uint64_t t = 0ull; t < rows; ++t) {
            const uint16_t* c = conved + t * ch;
            const uint16_t* q = c + (h / rep) * D;
            const uint16_t* k = c + k_heads * D + (h / rep) * D;
            const uint16_t* v = c + 2ull * k_heads * D + h * D;
            const float b = nn__silicon__half_to_float(beta[t * v_heads + h]), g_h = nn__silicon__half_to_float(g[t * v_heads + h]);
            const float inv_q = 1.0f / nn__silicon__sqrtf(amd_rocm6_wave64__zzpackage_squares128(q) + 1e-6f) / nn__silicon__sqrtf((float)D);
            const float inv_k = 1.0f / nn__silicon__sqrtf(amd_rocm6_wave64__zzpackage_squares128(k) + 1e-6f);
            const float decay = nn__silicon__expf(g_h);
            float kv = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) kv += col[i] * amd_rocm6_wave64__zzpackage_scalar_half(k, i);
            const float delta = (nn__silicon__half_to_float(v[j]) - decay * kv * inv_k) * b;
            float o = 0.0f;
#pragma unroll
            for (uint32_t c0 = 0u; c0 < D; c0 += 32u) {
                const uint16_t* k2 = amd_rocm6_wave64__zzpackage_afresh(k);
                const uint16_t* q2 = amd_rocm6_wave64__zzpackage_afresh(q);
#pragma unroll
                for (uint32_t i = c0; i < c0 + 32u; ++i) {
                    const float sn = decay * col[i] + amd_rocm6_wave64__zzpackage_scalar_half(k2, i) * inv_k * delta;
                    col[i] = sn;
                    o += sn * amd_rocm6_wave64__zzpackage_scalar_half(q2, i);
                }
            }
            const uint16_t first = nn__kernels__zzabi_to_half(o * inv_q, over);
            const float oj = nn__silicon__half_to_float(first);
            const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oj * oj) / (float)D + 1e-6f);
            const float zr = nn__silicon__half_to_float(z[t * vd + h * D + j]);
            const float zc = (zr < -88.0f) ? -88.0f : ((zr > 88.0f) ? 88.0f : zr);
            out[t * vd + h * D + j] =
                nn__kernels__zzabi_to_half(oj * scale * nn__silicon__half_to_float(w[j]) * (zc / (1.0f + nn__silicon__expf(-zc))), over);
        }
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) *(float*)((char*)(Sh + i * D) + at) = col[i];
    }
}
/* A head of 128, whole key groups, `conved` on 4 bytes for the scalar reads, and `out` and the state apart from what the
 * rows read; anything else is nn's. */
static void amd_rocm6_wave64__override_deltanet_steps(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta,
                                                      const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                      uint64_t v_heads, uint64_t head_dim, uint64_t rows, unsigned int* over) {
    const uint64_t c0 = (uint64_t)(uintptr_t)conved, c1 = c0 + rows * (2u * k_heads + v_heads) * head_dim * 2u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + rows * v_heads * head_dim * 2u;
    const uint64_t s0 = (uint64_t)(uintptr_t)S, s1 = s0 + v_heads * head_dim * head_dim * 4u;
    const bool apart = (o1 <= c0 || c1 <= o0) && (s1 <= c0 || c1 <= s0);
    if (head_dim != 128u || k_heads == 0u || v_heads % k_heads != 0u || ((uint64_t)(uintptr_t)conved) % 4u != 0u || !apart) {
        nn__deltanet__zzabi_launch_steps(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, over);
        return;
    }
    const uint32_t blocks = v_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)v_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &S, &conved, &z, &beta, &g, &w, &out, &k_heads, &v_heads, &rows, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof z, sizeof beta, sizeof g, sizeof w, sizeof out, sizeof k_heads,
                             sizeof v_heads, sizeof rows, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_deltanet_steps128, "amd_rocm6_wave64__zzprivate_deltanet_steps128", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_STEP_LANES, args, sizes, 11u);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_DELTANET_CUH */
