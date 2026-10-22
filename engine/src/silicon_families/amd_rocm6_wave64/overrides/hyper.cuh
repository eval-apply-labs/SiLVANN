#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_HYPER_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_HYPER_CUH
/* ══ hyper's DOOR, RUN FASTER — amd_rocm6_wave64 ══════════════════════════════════════════════════════════════════════
 * nn's hyper-connection doors (`nn/cpu/opcodes/hyper__abi.cuh`), where this family covers them: a site's `pre` over
 * streams of floats, at four streams. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ A SITE'S `pre`, ITS MIX AND ITS COLLAPSE, AT A KNOWN NUMBER OF STREAMS — nn's body sizes its arrays by the streams it
 *   is handed, and one lane works the mix's Sinkhorn iterations out of memory while the rest collapse: `MEASURED` with
 *   rocprof on DeepSeek's chunk (node03), 83 us a call where the collapse alone is 4. Here the count is the kernel's own,
 *   and the first wave works the mix a lane an entry of `comb` (`MULT · MULT` of them): a row's or a column's sum is read
 *   across the lanes by shuffles and added in nn's order by every lane of it, so each entry is nn's expression in nn's
 *   order (▶ nn's `nn__hyper__zzabi_body_pre_fp32` and its `weights`). Every lane of the wave runs the shuffles; the
 *   lanes past the entries work a copy and write nothing. */
template <uint32_t MULT>
static __device__ inline void amd_rocm6_wave64__zzprivate_hyper_weights(float* mix, const float* lg, const float* pre, const uint16_t* base,
                                                                         float s1, float s2, uint64_t iters, float eps) {
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE), l = lane % (MULT * MULT);
    const uint32_t i = l / MULT, j = l % MULT, k = 2u * MULT + l;
    float c = lg[k] * s2 + nn__silicon__half_to_float(base[k]);   /* each row a softmax over its columns, plus eps */
    float top = -NN__KERNELS__INFINITY, sum = 0.0f;
#pragma unroll
    for (uint32_t jj = 0u; jj < MULT; ++jj) {
        const float x = __shfl(c, (int)(i * MULT + jj));
        if (x > top) top = x;
    }
    c = nn__silicon__expf(c - top);
#pragma unroll
    for (uint32_t jj = 0u; jj < MULT; ++jj) sum += __shfl(c, (int)(i * MULT + jj));
    c = c / sum + eps;
    for (uint64_t it = 0u; it < iters; ++it) {              /* the columns, then — past the first — rows and columns */
        if (it != 0u) {
            float row = 0.0f;
#pragma unroll
            for (uint32_t jj = 0u; jj < MULT; ++jj) row += __shfl(c, (int)(i * MULT + jj));
            c /= row + eps;
        }
        float col = 0.0f;
#pragma unroll
        for (uint32_t ii = 0u; ii < MULT; ++ii) col += __shfl(c, (int)(ii * MULT + j));
        c /= col + eps;
    }
    if (lane < MULT * MULT) mix[k] = c;
#pragma unroll
    for (uint32_t ii = 0u; ii < MULT; ++ii)
        if (lane == ii) {
            mix[ii] = pre[ii];
            mix[MULT + ii] = 2.0f / (1.0f + nn__silicon__expf(-(lg[MULT + ii] * s1 + nn__silicon__half_to_float(base[MULT + ii]))));
        }
}
template <uint32_t MULT>
static __device__ inline void amd_rocm6_wave64__zzprivate_hyper_pre_fp32(float* out, float* mix, const float* streams, const uint16_t* base,
                                                                          const uint16_t* scale, uint64_t hidden, uint64_t iters, float eps) {
    const float* lg = mix + (2u + MULT) * MULT;
    const float s0 = nn__silicon__half_to_float(scale[0]), s1 = nn__silicon__half_to_float(scale[1]);
    const float s2 = nn__silicon__half_to_float(scale[2]);
    float pre[MULT];
#pragma unroll
    for (uint32_t i = 0u; i < MULT; ++i)
        pre[i] = 1.0f / (1.0f + nn__silicon__expf(-(lg[i] * s0 + nn__silicon__half_to_float(base[i])))) + eps;
    const uint64_t first = (uint64_t)nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane();
    if (first < AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) amd_rocm6_wave64__zzprivate_hyper_weights<MULT>(mix, lg, pre, base, s1, s2, iters, eps);
    for (uint64_t e = first; e < hidden; e += (uint64_t)nn__silicon__blocks() * nn__silicon__lanes()) {
        float sum = 0.0f;
#pragma unroll
        for (uint32_t i = 0u; i < MULT; ++i) sum += pre[i] * streams[i * hidden + e];
        out[e] = sum;
    }
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_hyper_pre_fp32_4(float* out, float* mix, const float* streams,
    const uint16_t* base, const uint16_t* scale, uint64_t hidden, uint64_t iters, float eps) {
    amd_rocm6_wave64__zzprivate_hyper_pre_fp32<4u>(out, mix, streams, base, scale, hidden, iters, eps);
}

/* Four streams — sixteen entries of `comb`, within a wave; anything else is nn's. */
static void amd_rocm6_wave64__override_hyper_pre_fp32(float* out, float* mix, const float* streams, const uint16_t* base,
                                                      const uint16_t* scale, uint64_t hidden, uint64_t mult, uint64_t iters, float eps) {
    if (mult != 4u || hidden == 0u) {
        nn__hyper__zzabi_launch_pre_fp32(out, mix, streams, base, scale, hidden, mult, iters, eps);
        return;
    }
    const uint64_t want = (hidden + NN__SILICON__LANES_MAX - 1u) / NN__SILICON__LANES_MAX;
    const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &out, &mix, &streams, &base, &scale, &hidden, &iters, &eps };
    const size_t sizes[] = { sizeof out, sizeof mix, sizeof streams, sizeof base, sizeof scale, sizeof hidden, sizeof iters, sizeof eps };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_hyper_pre_fp32_4, "amd_rocm6_wave64__zzprivate_hyper_pre_fp32", blocks,
                        NN__SILICON__LANES_MAX, args, sizes, 8u);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_HYPER_CUH */
