#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_HADAMARD_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_HADAMARD_CUH
/* ══ hadamard's DOORS, RUN FASTER — amd_rocm6_wave64 ══════════════════════════════════════════════════════════════════
 * nn's hadamard doors (`nn/cpu/opcodes/hadamard__abi.cuh`), where this family covers them: the rotation over blocks
 * of 512, and over blocks of 128, 256 or 512 at a head's width. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ── THE ROTATION — a wave a 512-block, the butterfly through shuffles ────────────────────────────────
 * nn's definition (`nn/gpu/kernels/kernels.cuh`): `out = H·(S ⊙ in) / √b` over blocks of 512, the sign vector
 * starting afresh at each block; the generic body sums each output directly, one lane an output. Here a WAVE
 * takes a block and each lane holds 8 consecutive elements: the stages that pair elements under 8 apart
 * are done inside a lane's registers, and the six that pair them 8..256 apart swap across lanes with a
 * shuffle — no barrier, since a wave's shuffle waits on nothing outside the wave. O(b log b) work against the
 * direct sum's O(b²), in fp32 and rounded to a half once, as the generic body is. Covers `n` a multiple of 512
 * with `out` apart from `in`; anything else is the generic door, whose in-place butterfly defines that case.
 * `MEASURED` (rocprof over test/src_gemv_27b_bench.cpp, gfx906): 6.2 us a call at every width a model rotates, against an
 * empty launch's 2.1; 3.5-4.3 with the conversion above and the shuffles inside a row on DPP, the same bits. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_BLOCK 512u
#define AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH  8u       /* elements a lane holds: 512 / 64 */

static __device__ inline void amd_rocm6_wave64__zzprivate_rotate_block(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                        uint64_t blocks, unsigned int* over) {
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t block = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS
                         + nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    if (block >= blocks) return;
    const uint64_t at = block * AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_BLOCK;
    const uint32_t c0 = lane * AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH;
    float v[AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH];
#pragma unroll
    for (uint32_t e = 0u; e < AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH; ++e)
        v[e] = nn__silicon__half_to_float(in[at + c0 + e]) * nn__silicon__half_to_float(sign[c0 + e]);
#pragma unroll
    for (uint32_t h = 1u; h < AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH; h <<= 1) {          /* inside the lane */
#pragma unroll
        for (uint32_t e = 0u; e < AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH; ++e) {
            if ((e & h) == 0u) {
                const float x = v[e], y = v[e + h];
                v[e] = x + y;
                v[e + h] = x - y;
            }
        }
    }
#pragma unroll
    for (uint32_t m = 1u; m < AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE; m <<= 1) {              /* across lanes */
        const bool low = (lane & m) == 0u;
#pragma unroll
        for (uint32_t e = 0u; e < AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH; ++e) {
            const float other = amd_rocm6_wave64__zzpackage_xor(v[e], m);
            v[e] = low ? v[e] + other : other - v[e];
        }
    }
    const float inv = 1.0f / nn__silicon__sqrtf((float)AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_BLOCK);
    /* every half made before any is written, so the eight leave as one write */
    uint16_t h[AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH];
#pragma unroll
    for (uint32_t e = 0u; e < AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH; ++e) h[e] = amd_rocm6_wave64__zzpackage_to_half(v[e] * inv, over);
#pragma unroll
    for (uint32_t e = 0u; e < AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH; ++e) out[at + c0 + e] = h[e];
}

static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_rotate(uint16_t* out, const uint16_t* in,
                                                                                             const uint16_t* sign, uint64_t blocks,
                                                                                             unsigned int* over) {
    amd_rocm6_wave64__zzprivate_rotate_block(out, in, sign, blocks, over);
}

/* ⭐ THE ROTATION of whole 512-blocks, `out` apart from `in`; anything else is the generic door. */
static void amd_rocm6_wave64__override_hadamard_rotate(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n,
                                                       unsigned int* over) {
    const bool apart = out + n <= in || in + n <= out;
    if (n == 0u || n % AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_BLOCK != 0u || !apart) {
        nn__hadamard__zzabi_launch_rotate(out, in, sign, n, over);
        return;
    }
    uint64_t blocks = n / AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_BLOCK;
    const uint64_t want = (blocks + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t launch = want > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)want;
    void* args[] = { &out, &in, &sign, &blocks, &over };
    const size_t sizes[] = { sizeof out, sizeof in, sizeof sign, sizeof blocks, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_rotate, "amd_rocm6_wave64__zzprivate_rotate", launch,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 5u);
}

/* ⭐ THE ROTATION OF ANY BLOCK OF 128, 256 OR 512 — `hadamard_blocks`, the rotation's own shape at a head's width. nn's
 *   generic body gives a lane an output and sums its block directly, 256 terms at a head of 256: `MEASURED` 55 us a call
 *   under rocprof, and a 27B prompt row makes two a layer for its keys and values. Here a WAVE takes a block, each lane
 *   `E = block / 64` consecutive elements, and the butterfly is the 512-block's above: the stages under `E` apart in a
 *   lane's registers, the rest across lanes by shuffles. `inverse` leaves the input unsigned and signs the output, as
 *   nn's body does. A wave reads its whole block before it writes any of it, so `out` may be `in`. */
template <uint32_t E>
static __device__ inline void amd_rocm6_wave64__zzprivate_hadamard_block(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                          uint64_t blocks, uint64_t inverse, unsigned int* over) {
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t block = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS
                         + nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    if (block >= blocks) return;
    const uint32_t B = E * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t at = block * B;
    const uint32_t c0 = lane * E;
    /* the lane's halves and signs read once, before either use, so each is one read: read where they are used, under
     * `inverse`, they are 3·E reads of two bytes, and a block of 512 6.3 us against 3.7 this way (`MEASURED`, back to back) */
    uint16_t xh[E], sh[E];
#pragma unroll
    for (uint32_t e = 0u; e < E; ++e) { xh[e] = in[at + c0 + e]; sh[e] = sign[c0 + e]; }
    float v[E];
#pragma unroll
    for (uint32_t e = 0u; e < E; ++e) {
        v[e] = nn__silicon__half_to_float(xh[e]);
        if (inverse == 0u) v[e] *= nn__silicon__half_to_float(sh[e]);
    }
#pragma unroll
    for (uint32_t h = 1u; h < E; h <<= 1) {                                         /* inside the lane */
#pragma unroll
        for (uint32_t e = 0u; e < E; ++e) {
            if ((e & h) == 0u) {
                const float x = v[e], y = v[e + h];
                v[e] = x + y;
                v[e + h] = x - y;
            }
        }
    }
#pragma unroll
    for (uint32_t m = 1u; m < AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE; m <<= 1) {         /* across lanes */
        const bool low = (lane & m) == 0u;
#pragma unroll
        for (uint32_t e = 0u; e < E; ++e) {
            const float other = amd_rocm6_wave64__zzpackage_xor(v[e], m);
            v[e] = low ? v[e] + other : other - v[e];
        }
    }
    const float norm = 1.0f / nn__silicon__sqrtf((float)B);
    uint16_t o[E];
#pragma unroll
    for (uint32_t e = 0u; e < E; ++e) {
        float r = v[e] * norm;
        if (inverse != 0u) r *= nn__silicon__half_to_float(sh[e]);
        o[e] = amd_rocm6_wave64__zzpackage_to_half(r, over);
    }
#pragma unroll
    for (uint32_t e = 0u; e < E; ++e) out[at + c0 + e] = o[e];
}

static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_hadamard_e2(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                                                  uint64_t blocks, uint64_t inverse, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_hadamard_block<2u>(out, in, sign, blocks, inverse, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_hadamard_e4(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                                                  uint64_t blocks, uint64_t inverse, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_hadamard_block<4u>(out, in, sign, blocks, inverse, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_hadamard_e8(uint16_t* out, const uint16_t* in, const uint16_t* sign,
                                                                                                  uint64_t blocks, uint64_t inverse, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_hadamard_block<8u>(out, in, sign, blocks, inverse, over);
}

/* ⭐ `hadamard_blocks` over whole blocks of 128, 256 or 512, `out` either `in` or apart from it; anything else is the
 *   generic door. */
static void amd_rocm6_wave64__override_hadamard_blocks(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n,
                                                       uint64_t block, uint64_t inverse, unsigned int* over) {
    const bool apart = out == in || out + n <= in || in + n <= out;
    if (n == 0u || (block != 128u && block != 256u && block != 512u) || n % block != 0u || !apart) {
        nn__hadamard__zzabi_launch_blocks(out, in, sign, n, block, inverse, over);
        return;
    }
    uint64_t blocks = n / block;
    const void* kernel = block == 128u ? (const void*)amd_rocm6_wave64__zzprivate_hadamard_e2
                       : block == 256u ? (const void*)amd_rocm6_wave64__zzprivate_hadamard_e4
                       :                 (const void*)amd_rocm6_wave64__zzprivate_hadamard_e8;
    const uint64_t want = (blocks + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t launch = want > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)want;
    void* args[] = { &out, &in, &sign, &blocks, &inverse, &over };
    const size_t sizes[] = { sizeof out, sizeof in, sizeof sign, sizeof blocks, sizeof inverse, sizeof over };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_hadamard", launch,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 6u);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_HADAMARD_CUH */
