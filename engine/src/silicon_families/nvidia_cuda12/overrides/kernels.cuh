#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_KERNELS_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_KERNELS_CUH
/* ══ WHAT THE OVERRIDES SHARE — nvidia_cuda12 ═════════════════════════════════════════════════════════════════════════
 * What more than one file beside this is written with: the warp's width and the tensor cores' one product, as nn's
 * bodies share theirs in `nn/gpu/kernels/kernels.cuh`. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>

#define NVIDIA_CUDA12__ZZPACKAGE_WARP 32u

/* ⭐ A FLOAT AS THE HALF AN OUTPUT HOLDS — nn's `zzabi_to_half`, the same bits and the same fault word, without its
 *   branches where the answer is a normal half: a float from 2^-14 up to 65520 is a normal half, rounded to nearest even
 *   at the 13th bit of its mantissa, in integers on the float's bits; anything else, and so every subnormal, zero,
 *   infinity, NaN and overflow, is nn's own conversion. amd_rocm6_wave64's `zzpackage_to_half`, the same integers —
 *   `MEASURED` there over all 2^32 floats, 0 halves and 0 fault words apart from nn's; `REASONED` the same here, since only
 *   integer arithmetic on the float's bits decides it. ⛔ NOT `__float2half_rn`, though it rounds the same way: a multiply
 *   before a conversion may then be folded into one rounding where nn's has two, and integers cannot be folded. */
static __device__ __forceinline__ uint16_t nvidia_cuda12__zzpackage_to_half(float f, unsigned int* over) {
    const uint32_t u = __float_as_uint(f), a = u & 0x7FFFFFFFu;
    if (a - 0x38800000u < 0x477FF000u - 0x38800000u)    /* 2^-14 <= |f| < 65520 */
        return (uint16_t)(((u >> 16) & 0x8000u) | (((a + 0x0FFFu + ((a >> 13) & 1u)) >> 13) - (112u << 10)));
    return nn__kernels__zzabi_to_half(f, over);
}

/* The sum of `v` over the warp's 32 lanes, handed to every lane: partners one lane apart, then two, … then sixteen — the
 * pairs of the block's fold over those 32 lanes, so the same float the fold makes of them. */
static __device__ __forceinline__ float nvidia_cuda12__zzpackage_warp_sum(float v) {
#pragma unroll
    for (uint32_t m = 1u; m < NVIDIA_CUDA12__ZZPACKAGE_WARP; m <<= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, (int)m);
    return v;
}

/* ⭐ THE TENSOR CORES' PRODUCT: `c += A · B`, A sixteen rows by sixteen columns of halves, B sixteen columns by eight of
 *   halves, `c` sixteen by eight floats — `mma.sync.aligned.m16n8k16`, the warp's 32 lanes each holding a part of every
 *   operand. Lane `4g + t` holds, of A, rows `g` and `g + 8` at columns `2t, 2t + 1` (`a0`, `a1`) and `2t + 8, 2t + 9`
 *   (`a2`, `a3`), two halves a register, the lower column low; of B, columns `g` at the same rows (`b0`, `b1`); of `c`,
 *   rows `g` (`c[0]`, `c[1]`) and `g + 8` (`c[2]`, `c[3]`) at columns `2t` and `2t + 1`. The products of two halves are
 *   exact in a float; the order the tensor core adds them in is its own.
 * ⛳ TURING (`sm_75`) HAS NO k16 SHAPE: there it is the same product as two k8 steps, the first eight columns and then the
 *   last eight. `REASONED` from the PTX ISA's fragment layouts; the family is tested on `sm_120` only. */
static __device__ __forceinline__ void nvidia_cuda12__zzpackage_mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                                     uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a2), "r"(a3), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

/* ⭐ THE 4-BIT LEVELS IN REGISTERS (NN-168) — nn's sixteen levels (`nn__turboquant__zzabi_levels(4)`) as two tables of
 *   bytes, four registers each: `tl` their low bytes, `th` their high bytes, level `4w + b` at byte `b` of word `w`. A word's
 *   eight 4-bit codes become four A registers by byte permutes alone — no table in shared memory, no load to wait for.
 *   `__byte_perm` reads a selector nibble's low three bits, so a code `k` picks byte `k & 7` from levels 0-7 and from
 *   levels 8-15 alike, and a third permute takes the one its top bit names (llama.cpp's 16-entry table, a byte table a
 *   half). Pair `i` is the levels of codes `2i` and `2i + 1`, the lower code's in the low half: the register the
 *   pair table gave for byte `i`. The levels are nn's, bit for bit. `MEASURED` the door bench on a B200 (sm_100,
 *   measurements/2026-11-08_nn168_levels_in_registers_b200.md): where every lane looked a byte up in a 256-entry table
 *   in shared memory, the 27B's MLP gate and up at 4 bits moved 1.15 TB/s; with the codes used raw (no lookup, wrong
 *   answers) 2.15; the table copied once a bank (no conflict) 0.96-1.20; looked up in registers 1.41. */
static __device__ __forceinline__ void nvidia_cuda12__zzpackage_levels4(uint32_t tl[4], uint32_t th[4]) {
    const uint16_t* lev4 = nn__turboquant__zzabi_levels(4u);
#pragma unroll
    for (uint32_t w = 0u; w < 4u; ++w) {
        tl[w] = th[w] = 0u;
#pragma unroll
        for (uint32_t b = 0u; b < 4u; ++b) {
            tl[w] |= (uint32_t)(lev4[4u * w + b] & 0xFFu) << (8u * b);
            th[w] |= (uint32_t)(lev4[4u * w + b] >> 8) << (8u * b);
        }
    }
}
static __device__ __forceinline__ void nvidia_cuda12__zzpackage_levels4_pairs(uint32_t q, const uint32_t tl[4], const uint32_t th[4],
                                                                              uint32_t h[4]) {
    const uint32_t upper = 0x32103210u | ((q & 0x88888888u) >> 1);  /* a code's byte from the upper eight where its top bit is set */
    uint32_t lo[2], hi[2];
#pragma unroll
    for (uint32_t i = 0u; i < 2u; ++i) {
        const uint32_t sh = 16u * i;
        lo[i] = __byte_perm(__byte_perm(tl[0], tl[1], q >> sh), __byte_perm(tl[2], tl[3], q >> sh), upper >> sh);
        hi[i] = __byte_perm(__byte_perm(th[0], th[1], q >> sh), __byte_perm(th[2], th[3], q >> sh), upper >> sh);
    }
    h[0] = __byte_perm(lo[0], hi[0], 0x5140u); h[1] = __byte_perm(lo[0], hi[0], 0x7362u);
    h[2] = __byte_perm(lo[1], hi[1], 0x5140u); h[3] = __byte_perm(lo[1], hi[1], 0x7362u);
}

/* ⭐ SIXTEEN BYTES INTO SHARED MEMORY WITHOUT A REGISTER — `cp.async`, Ampere's (`sm_80`) and on: the copy is issued and
 *   the lane goes on; `copies_issued` closes the lane's copies so far into a group, and `copies_landed` waits until at
 *   most `pending` groups (0 or 1) are still on their way. `valid` false writes sixteen zero bytes and reads nothing.
 *   Both addresses on sixteen bytes. On Turing there is no such copy: the lane loads and stores, and the waits are empty. */
static __device__ __forceinline__ void nvidia_cuda12__zzpackage_copy16(void* shared, const void* global, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t at = (uint32_t)__cvta_generic_to_shared(shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(at), "l"(global), "r"(valid ? 16u : 0u) : "memory");
#else
    *(uint4*)shared = valid ? *(const uint4*)global : make_uint4(0u, 0u, 0u, 0u);
#endif
}
static __device__ __forceinline__ void nvidia_cuda12__zzpackage_copies_issued(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
static __device__ __forceinline__ void nvidia_cuda12__zzpackage_copies_landed(uint32_t pending) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pending == 0u) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else               asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#else
    (void)pending;
#endif
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_KERNELS_CUH */
