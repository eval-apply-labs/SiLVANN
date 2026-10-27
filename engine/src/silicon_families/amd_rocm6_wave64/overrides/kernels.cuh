#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_KERNELS_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_KERNELS_CUH
/* ══ WHAT THE OVERRIDES SHARE — amd_rocm6_wave64 ══════════════════════════════════════════════════════════════════════
 * What more than one file beside this is written with: the wave's constants, the 4-bit levels as `v_perm_b32` finds
 * them, two half pairs' dot, the blocks the card holds at once, the half an output holds, the value of the lane
 * `m` away, a half read by the scalar unit and the sum of 128 squares in the fold's order — as nn's bodies share theirs
 * in `nn/gpu/kernels/kernels.cuh`. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>

#define AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE      64u
#define AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS      4u        /* rows a block, one a wave */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS    __launch_bounds__(256)

/* The levels of four codes — 16 bits of a row, low nibble first — packed as four int8. */
static __device__ inline uint32_t amd_rocm6_wave64__zzpackage_levels4(const uint32_t* lev, uint32_t codes) {
    const uint32_t spread = (codes & 0xFu) | ((codes & 0xF0u) << 4) | ((codes & 0xF00u) << 8) | ((codes & 0xF000u) << 12);
    const uint32_t pick = spread & 0x07070707u, top = ((spread >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t low = __builtin_amdgcn_perm(lev[1], lev[0], pick), high = __builtin_amdgcn_perm(lev[3], lev[2], pick);
    return (low & ~top) | (high & top);
}

/* The eight positive 4-bit levels' low and high bytes, as levels8 reads them. */
static __device__ inline void amd_rocm6_wave64__zzpackage_tables(uint32_t* low, uint32_t* high) {
    const uint16_t* table = nn__turboquant__zzabi_levels(4u) + 8u;
#pragma unroll
    for (uint32_t w = 0u; w < 2u; ++w) {
        low[w] = 0u; high[w] = 0u;
#pragma unroll
        for (uint32_t b = 0u; b < 4u; ++b) {
            low[w]  |= ((uint32_t)table[4u * w + b] & 0xFFu) << (8u * b);
            high[w] |= ((uint32_t)table[4u * w + b] >> 8) << (8u * b);
        }
    }
}

static __device__ inline float amd_rocm6_wave64__zzpackage_dot2(uint32_t a, uint32_t b, float c) {
    return nn__silicon__dot2(a, b, c);                         /* the family's — ▶ ../nn.cuh */
}

/* ⭐ THE BLOCKS THE CARD HOLDS AT ONCE for `kernel` — its occupancy times the device's compute units. A row kernel
 *   loads its lanes' `x` once a wave before its first row, so a grid larger than the card holds pays that load again
 *   for every wave that follows, for a row or two each. `MEASURED` (test/src_gemv_27b_bench.cpp, MI50): the 27B's
 *   out-projection 153 us at 960 blocks, 72 us at 120. Kept per thread: two TP ranks are two threads, each on its
 *   own device. */
static inline uint32_t amd_rocm6_wave64__zzpackage_resident(const void* kernel, uint32_t threads, size_t room) {
    struct seen { const void* kernel; int device; size_t room; uint32_t blocks; };
    static thread_local seen memo[32];
    static thread_local uint32_t used = 0u;
    int device = 0;
    (void)hipGetDevice(&device);
    for (uint32_t i = 0u; i < used; ++i)
        if (memo[i].kernel == kernel && memo[i].device == device && memo[i].room == room) return memo[i].blocks;
    int per = 0, units = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, (int)threads, room) != hipSuccess || per < 1) per = 1;
    if (hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess || units < 1) units = 1;
    const uint32_t blocks = (uint32_t)per * (uint32_t)units;
    if (used < 32u) memo[used++] = seen{kernel, device, room, blocks};
    return blocks;
}

/* ⭐ A FLOAT AS THE HALF AN OUTPUT HOLDS — nn's `zzabi_to_half`, the same bits and the same fault word, without its
 *   branches where the answer is a normal half. nn's conversion takes a case a branch, and on gfx906 it is most of a
 *   rotation's instructions (`MEASURED`, the ISA: 8 conversions a lane, ~150 lines each, against 48 shuffles). A float
 *   from 2^-14 up to 65520 is a normal half, rounded to nearest even at the 13th bit of its mantissa — nine integer
 *   instructions and a branch (the ISA); anything else, and so every subnormal, zero, infinity, NaN and overflow, is nn's own conversion.
 *   ⛔ NOT `v_cvt_f16_f32`, though it rounds the same way: the compiler folds the multiply before a conversion into
 *   `v_fma_mixlo_f16`, one rounding where nn's has two, and `MEASURED` 1 to 13 halves in 10^4-10^5 came out a last bit
 *   apart. Integers on the float's bits cannot be folded. `MEASURED` on gfx906 over all 2^32 floats: 0 halves and 0
 *   fault words differ from nn's, 503,308,288 inputs taking this path (the same sweep with its answer's low bit flipped
 *   finds all 503,308,288, so it can see a difference). */
static __device__ inline uint16_t amd_rocm6_wave64__zzpackage_to_half(float f, unsigned int* over) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f), a = u & 0x7FFFFFFFu;
    if (a - 0x38800000u < 0x477FF000u - 0x38800000u)    /* 2^-14 <= |f| < 65520 */
        return (uint16_t)(((u >> 16) & 0x8000u) | (((a + 0x0FFFu + ((a >> 13) & 1u)) >> 13) - (112u << 10)));
    return nn__kernels__zzabi_to_half(f, over);
}

/* ⭐ THE VALUE OF THE LANE `m` AWAY — `lane ^ m` — for `m` a power of two under 64, known when the caller is unrolled. A
 *   shuffle through `ds_bpermute` waits on the LDS crossbar, and a butterfly waits on it every stage; inside a row of 16
 *   lanes the move is a DPP modifier on a VALU instruction instead: 1 and 2 a quad's permutation, 4 a half-row's mirror then
 *   a quad's reversal, 8 a row's rotation by 8. Across rows, 16 is `ds_swizzle` (no address to compute) and 32 the
 *   shuffle. Only data moves, so a sum taken through it is the same bits as through `__shfl_xor`. */
static __device__ inline float amd_rocm6_wave64__zzpackage_xor(float v, uint32_t m) {
    const int b = __builtin_bit_cast(int, v);
    int r;
    if (m == 1u)       r = __builtin_amdgcn_mov_dpp(b, 0xB1, 0xF, 0xF, false);        /* quad_perm [1,0,3,2] */
    else if (m == 2u)  r = __builtin_amdgcn_mov_dpp(b, 0x4E, 0xF, 0xF, false);        /* quad_perm [2,3,0,1] */
    else if (m == 4u)  r = __builtin_amdgcn_mov_dpp(__builtin_amdgcn_mov_dpp(b, 0x141, 0xF, 0xF, false), 0x1B, 0xF, 0xF, false);
    else if (m == 8u)  r = __builtin_amdgcn_mov_dpp(b, 0x128, 0xF, 0xF, false);       /* row_ror:8 */
    else if (m == 16u) r = __builtin_amdgcn_ds_swizzle(b, 0x401F);                    /* xor 16 inside 32 lanes */
    else               return __shfl_xor(v, (int)m);
    return __builtin_bit_cast(float, r);
}

/* Half `i` of `at`, read through the constant address space, which the compiler reads with the scalar unit when the
 * address is the same for the whole wave. `REASONED`, on the premise that the scalar cache is invalidated at every
 * dispatch (the LLVM AMDGPU memory model for gfx6-9) and does not see this launch's own stores: a caller reads through
 * it only what the launch before wrote, and its door refuses an output that overlaps it. `MEASURED`: deltanet's
 * positions decoded with `conved` rewritten by a kernel between steps are byte-identical to its rows kernel over the same
 * positions (test/src_gemv_27b_bench.cpp's positions case). */
static __device__ inline float amd_rocm6_wave64__zzpackage_scalar_half(const uint16_t* at, uint32_t i) {
    const uint32_t word = ((const __attribute__((address_space(4))) uint32_t*)(const void*)at)[i / 2u];
    return nn__silicon__half_to_float((uint16_t)(i % 2u != 0u ? word >> 16 : word & 0xFFFFu));
}
/* `at` again, at an offset of zero the compiler cannot see through: reads through it are its own, so the compiler does
 * not keep an earlier pass's values of the same memory alive until a later one needs them — 128 registers a lane for
 * `k`, beside the column's 128, and the kernel spilled. */
static __device__ inline const uint16_t* amd_rocm6_wave64__zzpackage_afresh(const uint16_t* at) {
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
static __device__ inline float amd_rocm6_wave64__zzpackage_squares128(const uint16_t* x) {
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

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_KERNELS_CUH */
