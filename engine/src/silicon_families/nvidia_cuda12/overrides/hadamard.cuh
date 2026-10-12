#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_HADAMARD_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_HADAMARD_CUH
/* ══ hadamard's DOORS, RUN FASTER — nvidia_cuda12 ═════════════════════════════════════════════════════════════════════
 * nn's hadamard doors (`nn/cpu/opcodes/hadamard__abi.cuh`), where this family covers them: the rotation over blocks of
 * 512, and over blocks of 128, 256 or 512 at a head's width. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ── THE ROTATION — a warp a block, the butterfly through shuffles ─────────────────────────────────────────────────────
 * nn's definition (`nn/gpu/kernels/kernels.cuh`): `out = H·(S ⊙ in) / √b` over blocks of `b`, the sign vector starting
 * afresh at each block (`inverse`: the input unsigned and the output signed instead); the generic body sums each output
 * directly, a lane an output, `b` terms each. Here a WARP takes a block and lane `l` holds its elements `l, l + 32, …` —
 * `E = b / 32` of them, so every read and write of the warp is one run of 32 halves. The butterfly's stages run in order
 * of their distance, as `amd_rocm6_wave64`'s do: 1 to 16 pair lanes and swap across them by shuffles, 32 and on pair a
 * lane's own registers. O(b log b) work against the direct sum's O(b²), in fp32 and rounded to a half once, as the
 * generic body is; its sums are added in another order, so its last bits can differ. A warp reads its whole block before
 * it writes any of it, so `out` may be `in`. */
#define NVIDIA_CUDA12__ZZPRIVATE_ROT_E_MAX  16u      /* elements a lane holds at 512 */
#define NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS  8u       /* blocks of the rotation a launch's block takes, a warp each */
#define NVIDIA_CUDA12__ZZPRIVATE_ROT_BOUNDS __launch_bounds__(256)

static __device__ __forceinline__ void nvidia_cuda12__zzprivate_rotation(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                          uint64_t blocks, uint64_t inverse, unsigned int* over,
                                                                          const uint32_t E) {
    const uint32_t lane = (uint32_t)nn__silicon__lane() % NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint64_t block = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS
                         + (uint32_t)nn__silicon__lane() / NVIDIA_CUDA12__ZZPACKAGE_WARP;
    if (block >= blocks) return;                     /* the whole warp, so its shuffles below have every lane */
    const uint32_t B = E * NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint64_t at = block * B;
    float v[NVIDIA_CUDA12__ZZPRIVATE_ROT_E_MAX];
    uint16_t sh[NVIDIA_CUDA12__ZZPRIVATE_ROT_E_MAX];
#pragma unroll
    for (uint32_t e = 0u; e < E; ++e) {
        const uint32_t c = lane + NVIDIA_CUDA12__ZZPACKAGE_WARP * e;
        sh[e] = sign[c];
        v[e] = nn__silicon__half_to_float(in[at + c]);
        if (inverse == 0u) v[e] *= nn__silicon__half_to_float(sh[e]);
    }
#pragma unroll
    for (uint32_t m = 1u; m < NVIDIA_CUDA12__ZZPACKAGE_WARP; m <<= 1) {            /* 1 to 16 apart: across lanes */
        const bool low = (lane & m) == 0u;
#pragma unroll
        for (uint32_t e = 0u; e < E; ++e) {
            const float other = __shfl_xor_sync(0xFFFFFFFFu, v[e], (int)m);
            v[e] = low ? v[e] + other : other - v[e];
        }
    }
#pragma unroll
    for (uint32_t h = 1u; h < E; h <<= 1) {                                         /* 32 and on: inside the lane */
#pragma unroll
        for (uint32_t e = 0u; e < E; ++e) {
            if ((e & h) == 0u) {
                const float x = v[e], y = v[e + h];
                v[e] = x + y;
                v[e + h] = x - y;
            }
        }
    }
    const float norm = 1.0f / nn__silicon__sqrtf((float)B);
#pragma unroll
    for (uint32_t e = 0u; e < E; ++e) {
        float r = v[e] * norm;
        if (inverse != 0u) r *= nn__silicon__half_to_float(sh[e]);
        out[at + lane + NVIDIA_CUDA12__ZZPACKAGE_WARP * e] = nn__kernels__zzabi_to_half(r, over);
    }
}

static __global__ NVIDIA_CUDA12__ZZPRIVATE_ROT_BOUNDS void nvidia_cuda12__zzprivate_rotation_128(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                                    uint64_t blocks, uint64_t inverse, unsigned int* over) {
    nvidia_cuda12__zzprivate_rotation(out, in, sign, blocks, inverse, over, 4u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ROT_BOUNDS void nvidia_cuda12__zzprivate_rotation_256(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                                    uint64_t blocks, uint64_t inverse, unsigned int* over) {
    nvidia_cuda12__zzprivate_rotation(out, in, sign, blocks, inverse, over, 8u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ROT_BOUNDS void nvidia_cuda12__zzprivate_rotation_512(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                                    uint64_t blocks, uint64_t inverse, unsigned int* over) {
    nvidia_cuda12__zzprivate_rotation(out, in, sign, blocks, inverse, over, 16u);
}

/* `n / block` blocks of `block` (128, 256 or 512) through the kernel for that width. */
static inline void nvidia_cuda12__zzprivate_rotate(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n, uint64_t block,
                                                   uint64_t inverse, unsigned int* over) {
    uint64_t blocks = n / block;
    const void* kernel = block == 128u ? (const void*)nvidia_cuda12__zzprivate_rotation_128
                       : block == 256u ? (const void*)nvidia_cuda12__zzprivate_rotation_256
                       :                 (const void*)nvidia_cuda12__zzprivate_rotation_512;
    const uint64_t want = (blocks + NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS;
    const uint32_t launch = want > 0x7FFFFFFFull ? 0x7FFFFFFFu : (uint32_t)want;
    void* args[] = { &out, &in, &sign, &blocks, &inverse, &over };
    const size_t sizes[] = { sizeof out, sizeof in, sizeof sign, sizeof blocks, sizeof inverse, sizeof over };
    nn__silicon__launch(kernel, "nvidia_cuda12__zzprivate_rotation", launch,
                        NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 6u);
}

/* ⭐ THE ROTATION of whole 512-blocks, `out` apart from `in`; anything else is the generic door, whose in-place butterfly
 *   defines that case. */
static void nvidia_cuda12__override_hadamard_rotate(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n,
                                                    unsigned int* over) {
    const bool apart = out + n <= in || in + n <= out;
    if (n == 0u || n % 512u != 0u || !apart || n / 512u > 0x7FFFFFFFull * NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS) {
        nn__hadamard__zzabi_launch_rotate(out, in, sign, n, over);
        return;
    }
    nvidia_cuda12__zzprivate_rotate(out, in, sign, n, 512u, 0u, over);
}

/* ⭐ `hadamard_blocks` over whole blocks of 128, 256 or 512, `out` either `in` or apart from it; anything else is the
 *   generic door. */
static void nvidia_cuda12__override_hadamard_blocks(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n,
                                                    uint64_t block, uint64_t inverse, unsigned int* over) {
    const bool apart = out == in || out + n <= in || in + n <= out;
    if (n == 0u || (block != 128u && block != 256u && block != 512u) || n % block != 0u || !apart
     || n / block > 0x7FFFFFFFull * NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS) {
        nn__hadamard__zzabi_launch_blocks(out, in, sign, n, block, inverse, over);
        return;
    }
    nvidia_cuda12__zzprivate_rotate(out, in, sign, n, block, inverse, over);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_HADAMARD_CUH */
