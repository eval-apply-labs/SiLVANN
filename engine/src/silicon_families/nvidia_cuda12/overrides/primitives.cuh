#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_PRIMITIVES_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_PRIMITIVES_CUH
/* ══ primitives' DOOR, RUN FASTER — nvidia_cuda12 ═════════════════════════════════════════════════════════════════════
 * nn's primitives doors (`nn/cpu/opcodes/primitives__abi.cuh`), where this family covers them: the norm of a row of floats. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ⭐ THE NORM OF A ROW OF FLOATS, INTO HALVES — nn's `rmsnorm_fp32`, a hyper-connection's collapse (DeepSeek's streams are
 *   floats): nn's block, lanes and fold, a lane's squares added element by element as nn's lane adds them, each element
 *   `(x · scale) · w` rounded once, its overflow in the fault word — so nn's bits. What is this family's own: a lane's
 *   elements read RUN at a time, all in flight together, where nn's body waits for each before the next, twice. */
#define NVIDIA_CUDA12__ZZPRIVATE_NORM_LANES 256u       /* nn's door's */
#define NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN   16u
#define NVIDIA_CUDA12__ZZPRIVATE_NORM_BOUNDS __launch_bounds__(NVIDIA_CUDA12__ZZPRIVATE_NORM_LANES)
static __global__ NVIDIA_CUDA12__ZZPRIVATE_NORM_BOUNDS void nvidia_cuda12__zzprivate_norm_fp32(uint16_t* o, const float* x, const uint16_t* w,
                                                                                               uint64_t n, float eps, unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint32_t RUN = NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN;
    float part = 0.0f;
    uint64_t i = lane;
    for (; i + (RUN - 1u) * lanes < n; i += RUN * lanes) {
        float v[NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN];
#pragma unroll
        for (uint32_t u = 0u; u < RUN; ++u) v[u] = x[i + u * lanes];
#pragma unroll
        for (uint32_t u = 0u; u < RUN; ++u) part += v[u] * v[u];
    }
    for (; i < n; i += lanes) part += x[i] * x[i];
    const float mean  = nn__silicon__lanes_sum(part) / (float)n;
    const float scale = 1.0f / nn__silicon__sqrtf(mean + eps);
    i = lane;
    for (; i + (RUN - 1u) * lanes < n; i += RUN * lanes) {
        float v[NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN], g[NVIDIA_CUDA12__ZZPRIVATE_NORM_RUN];
#pragma unroll
        for (uint32_t u = 0u; u < RUN; ++u) {
            v[u] = x[i + u * lanes];
            g[u] = nn__silicon__half_to_float(w[i + u * lanes]);
        }
#pragma unroll
        for (uint32_t u = 0u; u < RUN; ++u) o[i + u * lanes] = nn__kernels__zzabi_to_half((v[u] * scale) * g[u], over);
    }
    for (; i < n; i += lanes) o[i] = nn__kernels__zzabi_to_half((x[i] * scale) * nn__silicon__half_to_float(w[i]), over);
}
/* nn's door's one block of 256 lanes, so a lane's stride and the fold are nn's; a row written over its own floats is nn's
 * door's (a lane reads its elements again after the fold, and writes halves over another lane's floats). */
static void nvidia_cuda12__override_rmsnorm_fp32(uint16_t* o, const float* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over) {
    const bool apart = (const char*)(o + n) <= (const char*)x || (const char*)(x + n) <= (const char*)o;
    if (n == 0u || !apart) {
        nn__rmsnorm__zzabi_launch_fp32(o, x, w, n, eps, over);
        return;
    }
    void* args[] = { &o, &x, &w, &n, &eps, &over };
    const size_t sizes[] = { sizeof o, sizeof x, sizeof w, sizeof n, sizeof eps, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_norm_fp32, "nvidia_cuda12__zzprivate_norm_fp32", 1u,
                        NVIDIA_CUDA12__ZZPRIVATE_NORM_LANES, args, sizes, 6u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_PRIMITIVES_CUH */
