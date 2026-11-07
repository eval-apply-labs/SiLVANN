#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_DELTANET_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_DELTANET_CUH
/* ══ deltanet's DOORS, RUN FASTER — nvidia_cuda12 ═════════════════════════════════════════════════════════════════════
 * nn's deltanet doors (`nn/cpu/opcodes/deltanet__abi.cuh`), where this family covers them: the step over a prompt's rows
 * at a head of 128, with Qwen 3.5's output gate or the one a call names, and the gates over rows. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ THE DELTANET OVER A PROMPT'S ROWS AT A HEAD OF 128 — nn's `deltanet_steps`, the one-position step a row at a time, a
 *   block a head and a lane a column of its state, the same operations in the same order as nn's body, so a prompt read as
 *   rows and the same prompt a position at a time (nn's `deltanet_step`) agree to the bit. What is this family's own:
 *   · the column lives in the lane's registers for every row of the call, read once and written once, where nn's body
 *     reads and writes it through memory twice a row
 *   · the block is the 128 lanes that own a column; nn's door gives a head 256, the upper 128 adding zeros to the fold,
 *     which changes nothing in the order the rest are added in — and a block a third the registers, so every head of a
 *     layer is on the card at once */
#define NVIDIA_CUDA12__ZZPRIVATE_DN_D 128u
#define NVIDIA_CUDA12__ZZPRIVATE_DN_BOUNDS __launch_bounds__(128)
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_deltanet_steps128_body(float* S, const uint16_t* conved, const uint16_t* z,
                                                                                         const uint16_t* beta, const uint16_t* g,
                                                                                         const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                                                         uint64_t v_heads, uint64_t rows, unsigned int gate,
                                                                                         unsigned int* over) {
    const uint32_t D = NVIDIA_CUDA12__ZZPRIVATE_DN_D;
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), j = lane;
    const uint64_t rep = v_heads / k_heads, ch = (2ull * k_heads + v_heads) * D, vd = v_heads * D;
    for (uint64_t h = nn__silicon__block(); h < v_heads; h += nn__silicon__blocks()) {
        float* Sh = S + h * D * D;
        float col[NVIDIA_CUDA12__ZZPRIVATE_DN_D];
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) col[i] = Sh[i * D + j];
        for (uint64_t t = 0ull; t < rows; ++t) {
            const uint16_t* c = conved + t * ch;
            const uint16_t* q = c + (h / rep) * D;
            const uint16_t* k = c + k_heads * D + (h / rep) * D;
            const uint16_t* v = c + 2ull * k_heads * D + h * D;
            const uint16_t* zh = z + t * vd + h * D;
            uint16_t* oh = out + t * vd + h * D;
            const float b = nn__silicon__half_to_float(beta[t * v_heads + h]), g_h = nn__silicon__half_to_float(g[t * v_heads + h]);
            float qq = 0.0f, kk = 0.0f;
            for (uint64_t i = lane; i < D; i += lanes) {
                const float qi = nn__silicon__half_to_float(q[i]), ki = nn__silicon__half_to_float(k[i]);
                qq += qi * qi;
                kk += ki * ki;
            }
            const float inv_q = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(qq) + 1e-6f) / nn__silicon__sqrtf((float)D);
            const float inv_k = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(kk) + 1e-6f);
            const float decay = nn__silicon__expf(g_h);
            float oo = 0.0f;
            float kv = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) kv += col[i] * nn__silicon__half_to_float(k[i]);
            const float delta = (nn__silicon__half_to_float(v[j]) - decay * kv * inv_k) * b;
            float o = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) {
                const float s = decay * col[i] + nn__silicon__half_to_float(k[i]) * inv_k * delta;
                col[i] = s;
                o += s * nn__silicon__half_to_float(q[i]);
            }
            oh[j] = nn__kernels__zzabi_to_half(o * inv_q, over);
            const float oj = nn__silicon__half_to_float(oh[j]);
            oo += oj * oj;
            const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oo) / (float)D + 1e-6f);
            const float zr = nn__silicon__half_to_float(zh[j]);
            const float zc = (zr < -88.0f) ? -88.0f : ((zr > 88.0f) ? 88.0f : zr);
            const float gz = gate == NN__DELTANET__GATE_SIGMOID ? 1.0f / (1.0f + nn__silicon__expf(-zc))
                                                                : zc / (1.0f + nn__silicon__expf(-zc));
            oh[j] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(oh[j]) * scale * nn__silicon__half_to_float(w[j]) * gz, over);
        }
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) Sh[i * D + j] = col[i];
    }
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_DN_BOUNDS void nvidia_cuda12__zzprivate_deltanet_steps128(float* S, const uint16_t* conved, const uint16_t* z,
                                                                                         const uint16_t* beta, const uint16_t* g,
                                                                                         const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                                                         uint64_t v_heads, uint64_t rows, unsigned int* over) {
    nvidia_cuda12__zzprivate_deltanet_steps128_body(S, conved, z, beta, g, w, out, k_heads, v_heads, rows, NN__DELTANET__GATE_SILU, over);
}
/* ⭐ …AND WITH THE OUTPUT GATE NAMED — nn's `deltanet_steps_gate`: σ(z), Flash-Next's, where silu(z) is Qwen 3.5's; the rest
 *   of the step the kernel above's, so a gated prompt's rows and its positions agree to the bit as Qwen's do. Every
 *   Flash-Next DeltaNet layer ran nn's body without it (`MEASURED` on amd_rocm6_wave64, 2026-11-01: 36 a token, 109 µs each;
 *   this family's own cost unmeasured). */
static __global__ NVIDIA_CUDA12__ZZPRIVATE_DN_BOUNDS void nvidia_cuda12__zzprivate_deltanet_steps128_gate(float* S, const uint16_t* conved,
                                                                                         const uint16_t* z, const uint16_t* beta,
                                                                                         const uint16_t* g, const uint16_t* w, uint16_t* out,
                                                                                         uint64_t k_heads, uint64_t v_heads, uint64_t rows,
                                                                                         unsigned int gate, unsigned int* over) {
    nvidia_cuda12__zzprivate_deltanet_steps128_body(S, conved, z, beta, g, w, out, k_heads, v_heads, rows, gate, over);
}
/* A head of 128 and whole key groups; anything else is nn's. */
static void nvidia_cuda12__override_deltanet_steps(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta,
                                                   const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                   uint64_t v_heads, uint64_t head_dim, uint64_t rows, unsigned int* over) {
    if (head_dim != NVIDIA_CUDA12__ZZPRIVATE_DN_D || k_heads == 0u || v_heads % k_heads != 0u) {
        nn__deltanet__zzabi_launch_steps(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, over);
        return;
    }
    const uint32_t blocks = v_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)v_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &S, &conved, &z, &beta, &g, &w, &out, &k_heads, &v_heads, &rows, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof z, sizeof beta, sizeof g, sizeof w, sizeof out, sizeof k_heads,
                             sizeof v_heads, sizeof rows, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_deltanet_steps128, "nvidia_cuda12__zzprivate_deltanet_steps128", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_DN_D, args, sizes, 11u);
}

/* ⭐ AND ONE POSITION — nn's `deltanet_step` — as the rows above at one row: the same body, so a position decoded and a
 *   prompt's rows agree to the bit by construction. A head of 128 and whole key groups; anything else is nn's. */
static void nvidia_cuda12__override_deltanet_step(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta,
                                                  const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                  uint64_t v_heads, uint64_t head_dim, unsigned int* over) {
    if (head_dim != NVIDIA_CUDA12__ZZPRIVATE_DN_D || k_heads == 0u || v_heads % k_heads != 0u) {
        nn__deltanet__zzabi_launch_step(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, over);
        return;
    }
    nvidia_cuda12__override_deltanet_steps(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, 1u, over);
}

/* A head of 128, whole key groups and a gate nn names; anything else is nn's. */
static void nvidia_cuda12__override_deltanet_steps_gate(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta,
                                                        const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                        uint64_t v_heads, uint64_t head_dim, uint64_t rows, unsigned int gate,
                                                        unsigned int* over) {
    if (head_dim != NVIDIA_CUDA12__ZZPRIVATE_DN_D || k_heads == 0u || v_heads % k_heads != 0u
     || (gate != NN__DELTANET__GATE_SILU && gate != NN__DELTANET__GATE_SIGMOID)) {
        nn__deltanet__zzabi_launch_steps_gate(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, gate, over);
        return;
    }
    const uint32_t blocks = v_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)v_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &S, &conved, &z, &beta, &g, &w, &out, &k_heads, &v_heads, &rows, &gate, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof z, sizeof beta, sizeof g, sizeof w, sizeof out, sizeof k_heads,
                             sizeof v_heads, sizeof rows, sizeof gate, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_deltanet_steps128_gate, "nvidia_cuda12__zzprivate_deltanet_steps128_gate",
                        blocks, NVIDIA_CUDA12__ZZPRIVATE_DN_D, args, sizes, 12u);
}

/* ⭐ THE GATES OVER ROWS — nn's `deltanet_gates_rows`, β = σ(b) and g = −exp(A_log)·softplus(a + dt_bias), a lane an element
 *   as nn's body, nn's arithmetic in its order, its six roundings through the family's `to_half` (nn's conversion's bits
 *   and fault word, ▶ kernels.cuh) where nn's body takes its branches. */
#define NVIDIA_CUDA12__ZZPRIVATE_GATES_BOUNDS __launch_bounds__(256)
static __global__ NVIDIA_CUDA12__ZZPRIVATE_GATES_BOUNDS void nvidia_cuda12__zzprivate_deltanet_gates_rows(uint16_t* beta, uint16_t* g, const uint16_t* a,
                                                                                         const uint16_t* b, const uint16_t* a_log,
                                                                                         const uint16_t* dt_bias, uint64_t heads,
                                                                                         uint64_t rows, unsigned int* over) {
    const uint64_t n = heads * rows, stride = (uint64_t)nn__silicon__blocks() * nn__silicon__lanes();
    for (uint64_t i = (uint64_t)nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane(); i < n; i += stride) {
        const uint64_t h = i % heads;
        beta[i] = nvidia_cuda12__zzpackage_to_half(1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[i]))), over);
        const float sum = nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(
                              nn__silicon__half_to_float(a[i]) + nn__silicon__half_to_float(dt_bias[h]), over));
        const float mag = (sum < 0.0f) ? -sum : sum, top = (sum > 0.0f) ? sum : 0.0f;
        const float sp = nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(
                             top + nn__silicon__logf(1.0f + nn__silicon__expf(-mag)), over));
        const float ea = nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(
                             nn__silicon__expf(nn__silicon__half_to_float(a_log[h])), over));
        const float prod = nn__silicon__half_to_float(nvidia_cuda12__zzpackage_to_half(sp * ea, over));
        g[i] = nvidia_cuda12__zzpackage_to_half(prod * -1.0f, over);
    }
}
/* nn's launch's lanes and the same arguments: every call is this family's. */
static void nvidia_cuda12__override_deltanet_gates_rows(uint16_t* beta, uint16_t* g, const uint16_t* a, const uint16_t* b,
                                                        const uint16_t* a_log, const uint16_t* dt_bias, uint64_t heads, uint64_t rows,
                                                        unsigned int* over) {
    const uint64_t want = (heads * rows + 255u) / 256u;
    const uint32_t blocks = want == 0u ? 1u : want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &beta, &g, &a, &b, &a_log, &dt_bias, &heads, &rows, &over };
    const size_t sizes[] = { sizeof beta, sizeof g, sizeof a, sizeof b, sizeof a_log, sizeof dt_bias, sizeof heads, sizeof rows, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_deltanet_gates_rows, "nvidia_cuda12__zzprivate_deltanet_gates_rows",
                        blocks, 256u, args, sizes, 9u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_DELTANET_CUH */
