#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_DELTANET_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_DELTANET_CUH
/* ══ deltanet's DOORS, RUN FASTER — amd_rocm6_wave64 ══════════════════════════════════════════════════════════════════
 * nn's deltanet doors (`nn/cpu/opcodes/deltanet__abi.cuh`), where this family covers them: the step at a head of
 * 128, over a prompt's rows and one position at a time. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ THE DELTANET OVER A PROMPT'S ROWS AT A HEAD OF 128 — nn's `deltanet_steps`, the one-position step a row at a time,
 *   a block a head and a lane a column of its state, exactly as nn's: the same operations in the same order, so a
 *   prompt read as rows and the same prompt a position at a time (this family's `deltanet_step`, below) agree to the bit. What
 *   is this family's own is where the column lives: in the lane's registers for every row of the call, read once and
 *   written once, where nn's body reads and writes it through memory twice a row. `MEASURED` on the 27B (rocprof):
 *   12.9 ms a 256-row call, 23% of a prompt. */
static __device__ inline void amd_rocm6_wave64__zzprivate_deltanet_rows128(float* Sh, const uint16_t* conved, const uint16_t* z,
                                                                           const uint16_t* beta, const uint16_t* g, const uint16_t* w,
                                                                           uint16_t* out, uint64_t k_heads, uint64_t v_heads, uint64_t h,
                                                                           uint64_t rows, unsigned int* over) {
    const uint32_t D = 128u;
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint64_t rep = v_heads / k_heads, ch = (2ull * k_heads + v_heads) * D, vd = v_heads * D;
    const bool owner = lane < D;                      /* a column's lane; the others take part in the folds alone */
    const uint64_t j = owner ? lane : 0u;
    float col[128];
#pragma unroll
    for (uint32_t i = 0u; i < D; ++i) col[i] = owner ? Sh[i * D + j] : 0.0f;
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
        float oo = 0.0f, oj = 0.0f;
        if (owner) {
            float kv = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) kv += col[i] * nn__silicon__half_to_float(k[i]);
            const float delta = (nn__silicon__half_to_float(v[j]) - decay * kv * inv_k) * b;
            float o = 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < D; ++i) {
                const float sn = decay * col[i] + nn__silicon__half_to_float(k[i]) * inv_k * delta;
                col[i] = sn;
                o += sn * nn__silicon__half_to_float(q[i]);
            }
            oh[j] = nn__kernels__zzabi_to_half(o * inv_q, over);
            oj = nn__silicon__half_to_float(oh[j]);
            oo += oj * oj;
        }
        const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oo) / (float)D + 1e-6f);
        if (owner) {
            const float zr = nn__silicon__half_to_float(zh[j]);
            const float zc = (zr < -88.0f) ? -88.0f : ((zr > 88.0f) ? 88.0f : zr);
            oh[j] = nn__kernels__zzabi_to_half(oj * scale * nn__silicon__half_to_float(w[j]) * (zc / (1.0f + nn__silicon__expf(-zc))), over);
        }
    }
    if (owner) {
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) Sh[i * D + j] = col[i];
    }
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_deltanet_steps128(
    float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta, const uint16_t* g, const uint16_t* w, uint16_t* out,
    uint64_t k_heads, uint64_t v_heads, uint64_t rows, unsigned int* over) {
    for (uint64_t h = nn__silicon__block(); h < v_heads; h += nn__silicon__blocks())
        amd_rocm6_wave64__zzprivate_deltanet_rows128(S + h * 128u * 128u, conved, z, beta, g, w, out, k_heads, v_heads, h, rows, over);
}
/* A head of 128 and whole key groups; anything else is nn's. */
static void amd_rocm6_wave64__override_deltanet_steps(float* S, const uint16_t* conved, const uint16_t* z, const uint16_t* beta,
                                                      const uint16_t* g, const uint16_t* w, uint16_t* out, uint64_t k_heads,
                                                      uint64_t v_heads, uint64_t head_dim, uint64_t rows, unsigned int* over) {
    if (head_dim != 128u || k_heads == 0u || v_heads % k_heads != 0u) {
        nn__deltanet__zzabi_launch_steps(S, conved, z, beta, g, w, out, k_heads, v_heads, head_dim, rows, over);
        return;
    }
    const uint32_t blocks = v_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)v_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &S, &conved, &z, &beta, &g, &w, &out, &k_heads, &v_heads, &rows, &over };
    const size_t sizes[] = { sizeof S, sizeof conved, sizeof z, sizeof beta, sizeof g, sizeof w, sizeof out, sizeof k_heads,
                             sizeof v_heads, sizeof rows, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_deltanet_steps128, "amd_rocm6_wave64__zzprivate_deltanet_steps128", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 11u);
}

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
 *   above, at one row and over many positions with `conved` rewritten between them. That kernel at one row was 114-125 us: it holds `q` and `k` in every lane beside the column, and
 *   spills. */
/* Half `i` of `at`, read through the constant address space, which the compiler reads with the scalar unit when the
 * address is the same for the whole wave. `REASONED`, on the premise that the scalar cache is invalidated at every
 * dispatch (the LLVM AMDGPU memory model for gfx6-9) and does not see this launch's own stores: nothing this launch
 * writes is read here — `conved` was written by the launch before, and the door refuses an `out` or a state that
 * overlaps it. `MEASURED`: positions decoded with `conved` rewritten by a kernel between steps are byte-identical to the
 * rows kernel over the same positions (test/src_gemv_27b_bench.cpp's positions case). */
static __device__ inline float amd_rocm6_wave64__zzprivate_scalar_half(const uint16_t* at, uint32_t i) {
    const uint32_t word = ((const __attribute__((address_space(4))) uint32_t*)(const void*)at)[i / 2u];
    return nn__silicon__half_to_float((uint16_t)(i % 2u != 0u ? word >> 16 : word & 0xFFFFu));
}
/* `at` again, at an offset of zero the compiler cannot see through: reads through it are its own, so the compiler does
 * not keep an earlier pass's values of the same memory alive until a later one needs them — 128 registers a lane for
 * `k`, beside the column's 128, and the kernel spilled. */
static __device__ inline const uint16_t* amd_rocm6_wave64__zzprivate_afresh(const uint16_t* at) {
    uint32_t none = 0u;
    __asm__ volatile("" : "+s"(none));
    return at + none;
}
/* The sum of the squares of 128 halves, in the order the family's fold adds them when a lane holds one each: pairs,
 *   then pairs of pairs. A wave's lane holds the squares of halves `lane` and `lane + 64`; exchanging partners one lane,
 *   then two, ... then 32 apart builds the two 64-wide trees exactly as the fold does — a lane's partial and its partner's
 *   added in either order are the same float — and the fold's last step adds the two. No wait for the block. Each square
 *   is rounded on its own, as each lane's is before the fold, hence the pragma: a fused multiply-add of a square onto a
 *   partial would be a different sum. */
static __device__ inline float amd_rocm6_wave64__zzprivate_squares128(const uint16_t* x) {
#pragma clang fp contract(off)
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const float a = nn__silicon__half_to_float(x[lane]), b = nn__silicon__half_to_float(x[lane + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE]);
    float lo = a * a, hi = b * b;
    for (uint32_t m = 1u; m < AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE; m <<= 1) {
        lo = lo + __shfl_xor(lo, (int)m);
        hi = hi + __shfl_xor(hi, (int)m);
    }
    return lo + hi;
}
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
        const float inv_q = 1.0f / nn__silicon__sqrtf(amd_rocm6_wave64__zzprivate_squares128(q) + 1e-6f) / nn__silicon__sqrtf((float)D);
        const float inv_k = 1.0f / nn__silicon__sqrtf(amd_rocm6_wave64__zzprivate_squares128(k) + 1e-6f);
        const float decay = nn__silicon__expf(g_h);
        float kv = 0.0f;
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) kv += col[i] * amd_rocm6_wave64__zzprivate_scalar_half(k, i);
        const float delta = (nn__silicon__half_to_float(v[j]) - decay * kv * inv_k) * b;
        const uint16_t* k2 = amd_rocm6_wave64__zzprivate_afresh(k);
        float o = 0.0f;
#pragma unroll
        for (uint32_t i = 0u; i < D; ++i) {
            const float sn = decay * col[i] + amd_rocm6_wave64__zzprivate_scalar_half(k2, i) * inv_k * delta;
            *(float*)((char*)(Sh + i * D) + at) = sn;
            o += sn * amd_rocm6_wave64__zzprivate_scalar_half(q, i);
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

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_DELTANET_CUH */
