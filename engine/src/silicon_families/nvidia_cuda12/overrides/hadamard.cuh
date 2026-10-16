#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_HADAMARD_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_HADAMARD_CUH
/* ══ hadamard's DOORS, RUN FASTER — nvidia_cuda12 ═════════════════════════════════════════════════════════════════════
 * nn's hadamard doors (`nn/cpu/opcodes/hadamard__abi.cuh`), where this family covers them: the rotation over blocks of
 * 512, over blocks of 128, 256 or 512 at a head's width, and row by row in the pack's ladder. */

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

/* One block of `E · 32` at `at`, by the whole warp: its shuffles need every lane. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_rotation_at(uint16_t* out, const uint16_t* in, uint64_t at,
                                                                             const uint16_t* sign, uint64_t inverse,
                                                                             unsigned int* over, const uint32_t E) {
    const uint32_t lane = (uint32_t)nn__silicon__lane() % NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint32_t B = E * NVIDIA_CUDA12__ZZPACKAGE_WARP;
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
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_rotation(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                          uint64_t blocks, uint64_t inverse, unsigned int* over,
                                                                          const uint32_t E) {
    const uint64_t block = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS
                         + (uint32_t)nn__silicon__lane() / NVIDIA_CUDA12__ZZPACKAGE_WARP;
    if (block >= blocks) return;                     /* the whole warp, so its shuffles have every lane */
    nvidia_cuda12__zzprivate_rotation_at(out, in, block * E * NVIDIA_CUDA12__ZZPACKAGE_WARP, sign, inverse, over, E);
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

/* ⭐ THE PACK'S LADDER, ROW BY ROW — `hadamard_ladder_rows`: each row of `cols` rotated in blocks of 512, then a 256 when
 *   at least that much is left, then a 128 (a row of 1152: 512 · 512 · 128), each block's signs from the vector's start,
 *   as the packer rotated a matrix that wide. nn's generic body sums each output's whole block directly, a lane an output
 *   — `MEASURED` (nsys, the precision, Qwen-Image 2.1's block 0 over a 1024² picture's 4,123 rows): 13.6 ms a call and
 *   65% of the block. Here a warp takes a block of a row, the butterfly above. Rows whose width is whole 128s, `out`
 *   either `in` or apart from it; anything else is nn's. */
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ROT_BOUNDS void nvidia_cuda12__zzprivate_ladder(uint16_t* out, const uint16_t* in,
                                                                                            const uint16_t* sign, uint64_t cols,
                                                                                            uint64_t rows, unsigned int* over) {
    const uint64_t whole = cols / 512u, rest = cols - whole * 512u;
    const uint64_t units = whole + (rest >= 256u ? 1u : 0u) + (rest % 256u != 0u ? 1u : 0u);
    for (uint64_t u = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS
                    + (uint32_t)nn__silicon__lane() / NVIDIA_CUDA12__ZZPACKAGE_WARP;
         u < rows * units; u += (uint64_t)nn__silicon__blocks() * NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS) {
        const uint64_t row = u / units, j = u - row * units, at = row * cols;
        if (j < whole)
            nvidia_cuda12__zzprivate_rotation_at(out, in, at + j * 512u, sign, 0u, over, 16u);
        else if (rest >= 256u && j == whole)
            nvidia_cuda12__zzprivate_rotation_at(out, in, at + whole * 512u, sign, 0u, over, 8u);
        else
            nvidia_cuda12__zzprivate_rotation_at(out, in, at + whole * 512u + (rest >= 256u ? 256u : 0u), sign, 0u, over, 4u);
    }
}
static void nvidia_cuda12__override_hadamard_ladder_rows(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t cols,
                                                         uint64_t rows, unsigned int* over) {
    const uint64_t n = cols * rows;
    const bool apart = out == in || out + n <= in || in + n <= out;
    if (cols == 0u || rows == 0u || cols % 128u != 0u || !apart) {
        nn__hadamard__zzabi_launch_ladder_rows(out, in, sign, cols, rows, over);
        return;
    }
    const uint64_t units = rows * (cols / 512u + (cols % 512u >= 256u ? 1u : 0u) + (cols % 256u != 0u ? 1u : 0u));
    const uint64_t want = (units + NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS;
    const uint32_t launch = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &out, &in, &sign, &cols, &rows, &over };
    const size_t sizes[] = { sizeof out, sizeof in, sizeof sign, sizeof cols, sizeof rows, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_ladder, "nvidia_cuda12__zzprivate_ladder", launch,
                        NVIDIA_CUDA12__ZZPRIVATE_ROT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 6u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_HADAMARD_CUH */
