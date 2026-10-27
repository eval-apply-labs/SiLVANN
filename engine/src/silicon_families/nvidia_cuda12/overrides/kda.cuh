#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_KDA_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_KDA_CUH
/* ══ KDA's DOORS, RUN FASTER — nvidia_cuda12 ══════════════════════════════════════════════════════════════════════════
 * nn's KDA doors (`nn/cpu/opcodes/kda__abi.cuh`), where this family covers them: the steps over a prompt's rows with their
 * decays made before, at a head of 128. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ KDA's STEPS OVER A PROMPT'S ROWS AT A HEAD OF 128 — nn's `kda_steps_decayed`, a block a head and a lane a column of its
 *   state, the same operations in the same order as nn's card body, so the two agree to the bit. What is this family's
 *   own, as in deltanet's steps (▶ deltanet.cuh):
 *   · the column stays in the lane's registers for every row of the call, read once and written once, where nn's body
 *     reads and writes it through memory an element at a time
 *   · the block is the 128 lanes that own a column; nn's door gives a head 256, the upper 128 adding zeros to the fold,
 *     which changes nothing in the order the rest are added in */
#define NVIDIA_CUDA12__ZZPRIVATE_KDA_D 128u
#define NVIDIA_CUDA12__ZZPRIVATE_KDA_BOUNDS __launch_bounds__(128)
static __global__ NVIDIA_CUDA12__ZZPRIVATE_KDA_BOUNDS void nvidia_cuda12__zzprivate_kda_steps_decayed128(
    float* S, const uint16_t* conved, const float* eg, const uint16_t* b, const uint16_t* gate, const uint16_t* w, uint16_t* out,
    uint64_t heads, float eps, uint64_t rows, unsigned int* over) {
    const uint32_t D = NVIDIA_CUDA12__ZZPRIVATE_KDA_D;
    const uint64_t j = nn__silicon__lane(), width = heads * D;
    const uint16_t* qs = conved;
    const uint16_t* ks = conved + rows * width;
    const uint16_t* vs = conved + 2ull * rows * width;
    for (uint64_t h = nn__silicon__block(); h < heads; h += nn__silicon__blocks()) {
        float* Sh = S + h * D * D;
        float col[NVIDIA_CUDA12__ZZPRIVATE_KDA_D];
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) col[i] = Sh[i * D + j];
        for (uint64_t t = 0ull; t < rows; ++t) {
            const uint64_t r = t * width + h * D;
            const uint16_t* q = qs + r;
            const uint16_t* k = ks + r;
            const float* e = eg + r;
            const float beta = 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[t * heads + h])));
            const float qi = nn__silicon__half_to_float(q[j]), ki = nn__silicon__half_to_float(k[j]);
            float qq = 0.0f, kk = 0.0f;
            qq += qi * qi;
            kk += ki * ki;
            const float inv_q = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(qq) + 1e-6f) / nn__silicon__sqrtf((float)D);
            const float inv_k = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(kk) + 1e-6f);
            float kv = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) {
                col[i] *= e[i];
                kv += col[i] * nn__silicon__half_to_float(k[i]) * inv_k;
            }
            const float delta = (nn__silicon__half_to_float(vs[r + j]) - kv) * beta;
            float o = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) {
                const float s = col[i] + nn__silicon__half_to_float(k[i]) * inv_k * delta;
                col[i] = s;
                o += s * nn__silicon__half_to_float(q[i]);
            }
            const float oj = nn__silicon__half_to_float(nn__kernels__zzabi_to_half(o * inv_q, over));   /* rounded, as nn's readout is */
            float oo = 0.0f;
            oo += oj * oj;
            const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oo) / (float)D + eps);
            const float gz = nn__silicon__half_to_float(gate[r + j]);
            out[r + j] = nn__kernels__zzabi_to_half(oj * scale * nn__silicon__half_to_float(w[j]) / (1.0f + nn__silicon__expf(-gz)), over);
        }
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) Sh[i * D + j] = col[i];
    }
}
/* A head of 128, and `out` and the state apart from what the steps read; anything else is nn's. */
static void nvidia_cuda12__override_kda_steps_decayed(float* S, const uint16_t* conved, const float* eg, const uint16_t* b,
                                                      const uint16_t* gate, const uint16_t* w, uint16_t* out, uint64_t heads,
                                                      uint64_t head_dim, float eps, uint64_t rows, unsigned int* over) {
    const uint64_t width = heads * head_dim;
    const uint64_t c0 = (uint64_t)(uintptr_t)conved, c1 = c0 + 3u * rows * width * 2u;
    const uint64_t e0 = (uint64_t)(uintptr_t)eg, e1 = e0 + rows * width * 4u;
    const uint64_t o0 = (uint64_t)(uintptr_t)out, o1 = o0 + rows * width * 2u;
    const uint64_t s0 = (uint64_t)(uintptr_t)S, s1 = s0 + heads * head_dim * head_dim * 4u;
    const bool apart = (o1 <= c0 || c1 <= o0) && (s1 <= c0 || c1 <= s0) && (o1 <= e0 || e1 <= o0) && (s1 <= e0 || e1 <= s0);
    if (head_dim != NVIDIA_CUDA12__ZZPRIVATE_KDA_D || heads == 0u || !apart) {
        nn__kda__zzabi_launch_steps_decayed(S, conved, eg, b, gate, w, out, heads, head_dim, eps, rows, over);
        return;
    }
    const uint32_t blocks = heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &S, &conved, &eg, &b, &gate, &w, &out, &heads, &eps, &rows, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof eg, sizeof b, sizeof gate, sizeof w, sizeof out, sizeof heads,
                             sizeof eps, sizeof rows, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_kda_steps_decayed128, "nvidia_cuda12__zzprivate_kda_steps_decayed128", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_KDA_D, args, sizes, 11u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_KDA_CUH */
