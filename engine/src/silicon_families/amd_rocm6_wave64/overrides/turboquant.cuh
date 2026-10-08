#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_TURBOQUANT_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_TURBOQUANT_CUH
/* ══ turboquant's DOORS, RUN FASTER — amd_rocm6_wave64 ════════════════════════════════════════════════════════════════
 * nn's turboquant doors (`nn/cpu/opcodes/turboquant__abi.cuh`), where this family covers them: the gemv at 4 and 8
 * bits, exact and in int8, and at 16 bits; the gemv over several matrices, and their weighted sum onto a residual;
 * and the gemv over an `x` wider than the lanes hold. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ── THE int8 GEMV AT 4 BITS — the card's four-way int8 dot product ─────────────────────────────────────
 * nn's definition (`nn/gpu/kernels/kernels.cuh`, the int8 gemv): x in blocks of 32 columns, quantised by
 * nn's own `zzabi_quantise`; the levels as int8; each block's products summed exactly in int32, then its
 * float scale; the row's scale / 127 once the row is summed. What is this family's own:
 *   · one WAVE a row, four rows a block, a lane a block of 32 columns — `lane`, `lane + 64`, … — and each
 *     lane quantises its own blocks of `x` once, into registers, so nothing is shared between lanes but the
 *     row's sum, and there is no barrier: the wave's sum is a shuffle
 *   · the sixteen levels in four registers, and eight codes' levels in two `v_perm_b32` a half: a nibble's
 *     low three bits pick a byte from one pair of registers or the other, and its top bit picks the pair
 *   · four products at a time: `v_dot4_i32_i8`, which gfx906 and gfx908 have; any other target sums the
 *     four itself, to the same integer
 * The integer sums are the generic body's exactly; the float sum over blocks is taken in another order. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_HELD      4u        /* blocks of x a lane holds: up to 64 · 4 · 32 = 8192 columns */

static __device__ inline int32_t amd_rocm6_wave64__zzprivate_dot4(uint32_t a, uint32_t b, int32_t c) {
#if defined(__gfx906__) || defined(__gfx908__)
    return __builtin_amdgcn_sdot4((int)a, (int)b, c, false);
#else
    for (uint32_t i = 0u; i < 4u; ++i)
        c += (int32_t)(int8_t)(uint8_t)(a >> (8u * i)) * (int32_t)(int8_t)(uint8_t)(b >> (8u * i));
    return c;
#endif
}

/* ⭐ EIGHT CODES — a word of a row, low nibble first — as four registers of two fp16 levels, in the order
 *   (c0,c2) (c4,c6) (c1,c3) (c5,c7). The even codes are the word's low nibbles and the odd its high ones, so one mask
 *   puts each set four codes a byte. nn's sixteen levels are symmetric — `level[15 - c]` is `level[c]` with its sign
 *   turned — so only the eight positive ones are held, `low` and `high` their low and high bytes in two registers
 *   each: one exclusive-or over the whole word turns every code into its level's place among the eight and leaves,
 *   in the nibble's top bit, whether the level is negative, which `v_and_or_b32` puts into the high byte. Where a
 *   code's top bit chose between two lookups with a byte mask, it was 27 instructions for eight codes; this is 18.
 *   The caller holds `x` in the same order. */
static __device__ inline void amd_rocm6_wave64__zzprivate_levels8(const uint32_t* low, const uint32_t* high, uint32_t word,
                                                                  uint32_t* pairs) {
    const uint32_t tops = word & 0x88888888u;                     /* 8 in each nibble whose level is positive */
    /* each nibble c becomes: its place k among the positive levels (c - 8, or 7 - c), and 8 where c < 8 */
    const uint32_t w = ~(word ^ (tops - (tops >> 3)));
#pragma unroll
    for (uint32_t s = 0u; s < 2u; ++s) {
        const uint32_t n = s == 0u ? w : w >> 4;
        const uint32_t pick = n & 0x07070707u;
        const uint32_t sign = (s == 0u ? w << 4 : w) & 0x80808080u;
        const uint32_t lo = __builtin_amdgcn_perm(low[1], low[0], pick);
        const uint32_t hi = __builtin_amdgcn_perm(high[1], high[0], pick) | sign;
        pairs[2u * s + 0u] = __builtin_amdgcn_perm(hi, lo, 0x05010400u);
        pairs[2u * s + 1u] = __builtin_amdgcn_perm(hi, lo, 0x07030602u);
    }
}
/* `x`'s eight halves at `at` in levels8's order: (x0,x2) (x4,x6) (x1,x3) (x5,x7). */
static __device__ inline void amd_rocm6_wave64__zzprivate_x8(const uint16_t* at, uint32_t* out) {
    out[0] = (uint32_t)at[0] | ((uint32_t)at[2] << 16);
    out[1] = (uint32_t)at[4] | ((uint32_t)at[6] << 16);
    out[2] = (uint32_t)at[1] | ((uint32_t)at[3] << 16);
    out[3] = (uint32_t)at[5] | ((uint32_t)at[7] << 16);
}

static __device__ inline void amd_rocm6_wave64__zzprivate_int8_rows4(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                      const uint16_t* x, uint64_t rows, uint64_t cols,
                                                                      unsigned int* over) {
    const int8_t* table = nn__turboquant__zzabi_levels_i8(4u);
    uint32_t lev[4];
    for (uint32_t w = 0u; w < 4u; ++w)
        lev[w] = (uint32_t)(uint8_t)table[4u * w] | ((uint32_t)(uint8_t)table[4u * w + 1u] << 8)
               | ((uint32_t)(uint8_t)table[4u * w + 2u] << 16) | ((uint32_t)(uint8_t)table[4u * w + 3u] << 24);
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t blocks = cols / 32u, row_bytes = cols / 2u;
    uint32_t q[AMD_ROCM6_WAVE64__ZZPRIVATE_HELD][8];
    float s[AMD_ROCM6_WAVE64__ZZPRIVATE_HELD];
#pragma unroll
    for (uint32_t k = 0u; k < AMD_ROCM6_WAVE64__ZZPRIVATE_HELD; ++k) {
        const uint64_t b = lane + (uint64_t)k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
        s[k] = b < blocks ? nn__turboquant__zzabi_quantise(x, b * 32u, q[k]) : 0.0f;
    }
    for (uint64_t r = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; r < rows;
         r += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        const uint8_t* row = weights + r * row_bytes;
        float part = 0.0f;
#pragma unroll
        for (uint32_t k = 0u; k < AMD_ROCM6_WAVE64__ZZPRIVATE_HELD; ++k) {
            const uint64_t b = lane + (uint64_t)k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
            if (b < blocks) {
                const uint64_t* codes = (const uint64_t*)(row + b * 16u);    /* 32 columns: 16 bytes */
                int32_t acc = 0;
#pragma unroll
                for (uint32_t e = 0u; e < 2u; ++e) {
                    const uint64_t c = codes[e];
                    acc = amd_rocm6_wave64__zzprivate_dot4(amd_rocm6_wave64__zzpackage_levels4(lev, (uint32_t)c), q[k][4u * e + 0u], acc);
                    acc = amd_rocm6_wave64__zzprivate_dot4(amd_rocm6_wave64__zzpackage_levels4(lev, (uint32_t)(c >> 16)), q[k][4u * e + 1u], acc);
                    acc = amd_rocm6_wave64__zzprivate_dot4(amd_rocm6_wave64__zzpackage_levels4(lev, (uint32_t)(c >> 32)), q[k][4u * e + 2u], acc);
                    acc = amd_rocm6_wave64__zzprivate_dot4(amd_rocm6_wave64__zzpackage_levels4(lev, (uint32_t)(c >> 48)), q[k][4u * e + 3u], acc);
                }
                part += (float)acc * s[k];
            }
        }
        for (uint32_t off = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u; off > 0u; off >>= 1) part += __shfl_xor(part, (int)off);
        if (lane == 0u) {
            const float scale = nn__silicon__half_to_float(
                                    (uint16_t)((uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8)));
            out[r] = nn__kernels__zzabi_to_half(part * scale / 127.0f, over);
        }
    }
}

static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_int8_4(uint16_t* out, const uint8_t* weights,
                                                                                             const uint8_t* luts, const uint16_t* x,
                                                                                             uint64_t rows, uint64_t cols,
                                                                                             unsigned int* over) {
    amd_rocm6_wave64__zzprivate_int8_rows4(out, weights, luts, x, rows, cols, over);
}

/* ── THE EXACT GEMV AT 4 BITS — the same row, in fp32 ──────────────────────────────────────────────────
 * nn's definition: each code's fp16 level times its fp16 `x`, summed in fp32, times the row's scale. The
 * same shape as the int8 route above — a wave a row, a lane's blocks of `x` held in registers (as halves,
 * two a register), the levels found by `v_perm_b32` — and two things of its own:
 *   · a level is two bytes, so the sixteen are held as two byte tables of four registers each, the low
 *     bytes and the high; one nibble lookup into each gives four levels' two halves, and one more
 *     `v_perm_b32` interleaves them into two registers of two fp16 levels
 *   · two products at a time: `v_dot2_f32_f16`, which gfx906 and gfx908 have — a product of two halves is
 *     exact in fp32, so only the order of the sum differs from the generic body; any other target widens
 *     the halves and multiplies them itself */

/* ⭐ `x` IN SHARED MEMORY, ONCE A BLOCK: a lane held its blocks of `x` in registers — 48 of a 5120-wide row's 124 —
 *   which left room for two waves a SIMD, too few to cover a row's loads. The block's four waves write `x` once,
 *   in levels8's order, wait for each other once, and each lane then reads eight columns' halves in one 16-byte
 *   read. The one barrier is before the first row, so nothing waits on it once the rows begin. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_X_SHARED  8192u     /* columns `x` may have: 16 KB of halves */
/* The blocks of 32 columns a lane holds for a row of `cols`, which picks the kernel and sizes its room. */
static __device__ __host__ inline uint64_t amd_rocm6_wave64__zzprivate_held(uint64_t cols) {
    return (cols / 32u + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
}
/* The room a block's kernel is given, sized to the call: `x`'s halves as `stage_x` lays them out, then nn's 256 8-bit
 * levels where a kernel reads them. A row of 5120 is then 12.5 KB, where a room of the widest row allowed only three
 * blocks a CU. */
static inline size_t amd_rocm6_wave64__zzprivate_room_bytes(uint64_t cols) {
    return (size_t)amd_rocm6_wave64__zzprivate_held(cols) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE * 64u + 512u;
}
/* ⭐ `x` WRITTEN WORD-MAJOR: eight columns `q` of block `b` at `xs[q * HELD * 64 + b]`, so the lanes of a wave, which hold
 *   consecutive blocks, read consecutive 16 bytes. Laid out block-major, a lane's four reads are 64 bytes from its
 *   neighbour's and a wave's 16-byte read meets only 8 of the 32 banks: `MEASURED` (rocprof, the 27B's shapes on one
 *   MI50) shared memory stalled on its banks 36-46% of the time in every 4-bit row kernel, against 0.3-2.3% word-major,
 *   and the 8-bit head took 2571 us against 2050. Each word's run is as long as the kernel's blocks, `HELD * 64`,
 *   whatever the row's width, so every read is one register and a constant offset; the blocks past the row's end hold
 *   zeros. */
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_stage_x(uint4* xs, const uint16_t* x, uint64_t cols) {
    const uint32_t S = HELD * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t blocks = cols / 32u;
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < 4u * S; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
        const uint32_t q = e / S, b = e % S;
        uint32_t v[4] = { 0u, 0u, 0u, 0u };
        if (b < blocks) amd_rocm6_wave64__zzprivate_x8(x + 32u * (uint64_t)b + 8u * q, v);
        xs[e] = make_uint4(v[0], v[1], v[2], v[3]);
    }
}

/* ⭐ A ROW'S CODES ARE SAID TO BE IN THE CARD'S MEMORY — the type carries it: an address made from an integer, as a matrix
 *   taken from the groups' table is, is otherwise read with flat instructions, and a flat read is counted with shared
 *   memory's, so every wait for `x` waited for the codes too. */
typedef const __attribute__((address_space(1))) uint8_t amd_rocm6_wave64__global_byte;
typedef const __attribute__((address_space(1))) uint64_t amd_rocm6_wave64__global_word;
static __device__ inline amd_rocm6_wave64__global_byte* amd_rocm6_wave64__zzprivate_global(uint64_t at) {
    return (amd_rocm6_wave64__global_byte*)(uintptr_t)at;
}

/* ⭐ A LANE'S PART OF ONE ROW AT 4 BITS — its blocks of the row's codes and of `x`, each block's sixteen products summed
 *   on their own and then onto the row's. `last` is where the lane's last block starts, and `real` whether that block is
 *   in the row: a block past the row's end reads the row's first codes and is weighed by zeros, so every load is
 *   unconditional and a row's loads are issued together. Every block but the last is in the row, since a kernel holds
 *   the fewest blocks its width needs. The word loop is unrolled, so a word is a register and not a selection. */
template <uint32_t HELD>
static __device__ inline float amd_rocm6_wave64__zzprivate_part4(amd_rocm6_wave64__global_byte* row, uint32_t last, bool real, const uint4* xs,
                                                                 uint32_t lane, const uint32_t* low, const uint32_t* high) {
    const uint32_t S = HELD * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    uint64_t cur[HELD][2];
#pragma unroll
    for (uint32_t k = 0u; k < HELD; ++k) {
        amd_rocm6_wave64__global_word* c = (amd_rocm6_wave64__global_word*)(row + (k + 1u < HELD ? (lane + k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) * 16u : last));
        cur[k][0] = c[0]; cur[k][1] = c[1];
    }
    float part = 0.0f;
#pragma unroll
    for (uint32_t k = 0u; k < HELD; ++k) {
        float sub = 0.0f;
#pragma unroll
        for (uint32_t q = 0u; q < 4u; ++q) {                             /* eight columns at a time */
            uint32_t pairs[4];
            amd_rocm6_wave64__zzprivate_levels8(low, high, (uint32_t)(cur[k][q / 2u] >> (32u * (q % 2u))), pairs);
            const uint4 xv = xs[q * S + k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE + lane];
            sub = amd_rocm6_wave64__zzpackage_dot2(pairs[0], xv.x, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(pairs[1], xv.y, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(pairs[2], xv.z, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(pairs[3], xv.w, sub);
        }
        part += (k + 1u < HELD || real) ? sub : 0.0f;
    }
    return part;
}
/* One block's sixteen products at 4 bits, `x` from shared memory at block `b` of every word's run. */
template <uint32_t HELD>
static __device__ inline float amd_rocm6_wave64__zzprivate_block4(const uint64_t* cur, const uint4* xs, uint32_t b,
                                                                  const uint32_t* low, const uint32_t* high) {
    const uint32_t S = HELD * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    float sub = 0.0f;
#pragma unroll
    for (uint32_t q = 0u; q < 4u; ++q) {
        uint32_t pairs[4];
        amd_rocm6_wave64__zzprivate_levels8(low, high, (uint32_t)(cur[q / 2u] >> (32u * (q % 2u))), pairs);
        const uint4 xv = xs[q * S + b];
        sub = amd_rocm6_wave64__zzpackage_dot2(pairs[0], xv.x, sub);
        sub = amd_rocm6_wave64__zzpackage_dot2(pairs[1], xv.y, sub);
        sub = amd_rocm6_wave64__zzpackage_dot2(pairs[2], xv.z, sub);
        sub = amd_rocm6_wave64__zzpackage_dot2(pairs[3], xv.w, sub);
    }
    return sub;
}
/* ⭐ TWO ROWS AT ONCE, AND THE LAST BLOCK SHARED — both rows' codes loaded before either is decoded, so a wave has twice the
 *   bytes in flight. Where a row's last blocks fill no more than half the wave (`split`: the 27B's 5120 columns are 160
 *   blocks, 32 of them in the third run), the lower half of the wave takes the first row's last block and the upper half
 *   the second's, and one exchange 32 lanes apart hands each lower lane the second row's, where otherwise half the wave
 *   decodes a block it then weighs by zero, a sixth of all the decoding at that width. `last` is the lane's last block —
 *   counted from its half when split — and `real` whether it is in the row. Each lane computes each block exactly as
 *   part4 does and adds it in part4's order, so each row's sum is part4's to the bit. */
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_part4x2(amd_rocm6_wave64__global_byte* row0, amd_rocm6_wave64__global_byte* row1, bool split, uint32_t last,
                                                                  bool real, const uint4* xs, uint32_t lane, const uint32_t* low,
                                                                  const uint32_t* high, float* part) {
    const uint32_t L = HELD - 1u;
    const uint32_t off = real ? last * 16u : 0u;              /* a block past the row's end reads the row's first codes */
    uint64_t cur[2][HELD][2];
#pragma unroll
    for (uint32_t k = 0u; k < L; ++k)
#pragma unroll
        for (uint32_t j = 0u; j < 2u; ++j) {
            amd_rocm6_wave64__global_word* c = (amd_rocm6_wave64__global_word*)((j == 0u ? row0 : row1) + (lane + k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) * 16u);
            cur[j][k][0] = c[0]; cur[j][k][1] = c[1];
        }
    {
        amd_rocm6_wave64__global_word* c = (amd_rocm6_wave64__global_word*)((split && (lane & 32u) != 0u ? row1 : row0) + off);
        cur[0][L][0] = c[0]; cur[0][L][1] = c[1];
    }
    if (!split) {
        amd_rocm6_wave64__global_word* c = (amd_rocm6_wave64__global_word*)(row1 + off);
        cur[1][L][0] = c[0]; cur[1][L][1] = c[1];
    }
    part[0] = 0.0f; part[1] = 0.0f;
#pragma unroll
    for (uint32_t k = 0u; k < L; ++k)
#pragma unroll
        for (uint32_t j = 0u; j < 2u; ++j)
            part[j] += amd_rocm6_wave64__zzprivate_block4<HELD>(cur[j][k], xs, lane + k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, low, high);
    if (split) {
        const float own = amd_rocm6_wave64__zzprivate_block4<HELD>(cur[0][L], xs, last, low, high);
        const float other = __shfl_xor(own, 32);
        const bool lower = real && (lane & 32u) == 0u;
        part[0] += lower ? own : 0.0f;
        part[1] += lower ? other : 0.0f;
    } else {
#pragma unroll
        for (uint32_t j = 0u; j < 2u; ++j)
            part[j] += real ? amd_rocm6_wave64__zzprivate_block4<HELD>(cur[j][L], xs, last, low, high) : 0.0f;
    }
}
/* ⭐ R ROWS AT ONCE WHERE A LANE HOLDS ONE BLOCK — a row 2048 wide is a kilobyte a wave at 4 bits, too little in flight
 *   for a wave to cover its own wait, so four rows' codes (two at 8 bits) are loaded before any is decoded, and each read
 *   of `x` serves all of them. Each row's sum is part4's or part8's, in their order; a lane's block past the row's end
 *   reads the row's first codes and is weighed by zero. */
template <uint32_t D, uint32_t R>
static __device__ inline void amd_rocm6_wave64__zzprivate_part1xr(amd_rocm6_wave64__global_byte* const* rows, uint32_t off, bool real,
                                                                  const uint4* xs, uint32_t lane, const uint32_t* low,
                                                                  const uint32_t* high, const uint16_t* table, float* part) {
    uint64_t c[R][D / 2u];
#pragma unroll
    for (uint32_t j = 0u; j < R; ++j) {
        amd_rocm6_wave64__global_word* p = (amd_rocm6_wave64__global_word*)(rows[j] + off);
#pragma unroll
        for (uint32_t h = 0u; h < D / 2u; ++h) c[j][h] = p[h];
    }
    float sub[R];
#pragma unroll
    for (uint32_t j = 0u; j < R; ++j) sub[j] = 0.0f;
#pragma unroll
    for (uint32_t q = 0u; q < 4u; ++q) {
        const uint4 xv = xs[q * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE + lane];
#pragma unroll
        for (uint32_t j = 0u; j < R; ++j) {
            if (D == 4u) {
                uint32_t pairs[4];
                amd_rocm6_wave64__zzprivate_levels8(low, high, (uint32_t)(c[j][q / 2u] >> (32u * (q % 2u))), pairs);
                sub[j] = amd_rocm6_wave64__zzpackage_dot2(pairs[0], xv.x, sub[j]);
                sub[j] = amd_rocm6_wave64__zzpackage_dot2(pairs[1], xv.y, sub[j]);
                sub[j] = amd_rocm6_wave64__zzpackage_dot2(pairs[2], xv.z, sub[j]);
                sub[j] = amd_rocm6_wave64__zzpackage_dot2(pairs[3], xv.w, sub[j]);
            } else {
                const uint64_t cw = c[j][q % (D / 2u)];
                const uint32_t l0 = table[cw & 0xFFu], l1 = table[(cw >> 8) & 0xFFu], l2 = table[(cw >> 16) & 0xFFu];
                const uint32_t l3 = table[(cw >> 24) & 0xFFu], l4 = table[(cw >> 32) & 0xFFu], l5 = table[(cw >> 40) & 0xFFu];
                const uint32_t l6 = table[(cw >> 48) & 0xFFu], l7 = table[(cw >> 56) & 0xFFu];
                sub[j] = amd_rocm6_wave64__zzpackage_dot2(l0 | (l2 << 16), xv.x, sub[j]);
                sub[j] = amd_rocm6_wave64__zzpackage_dot2(l4 | (l6 << 16), xv.y, sub[j]);
                sub[j] = amd_rocm6_wave64__zzpackage_dot2(l1 | (l3 << 16), xv.z, sub[j]);
                sub[j] = amd_rocm6_wave64__zzpackage_dot2(l5 | (l7 << 16), xv.w, sub[j]);
            }
        }
    }
#pragma unroll
    for (uint32_t j = 0u; j < R; ++j) {
        part[j] = 0.0f;
        part[j] += real ? sub[j] : 0.0f;
    }
}
/* ...and at 8 bits: a block is 32 bytes, each code's level from nn's table in shared memory, two levels a half pair. */
template <uint32_t HELD>
static __device__ inline float amd_rocm6_wave64__zzprivate_part8(amd_rocm6_wave64__global_byte* row, uint32_t last, bool real, const uint4* xs,
                                                                 uint32_t lane, const uint16_t* table) {
    const uint32_t S = HELD * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    float part = 0.0f;
#pragma unroll
    for (uint32_t k = 0u; k < HELD; ++k) {
        amd_rocm6_wave64__global_word* codes = (amd_rocm6_wave64__global_word*)(row + (k + 1u < HELD ? (lane + k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) * 32u : last));
        uint64_t c[4];
#pragma unroll
        for (uint32_t q = 0u; q < 4u; ++q) c[q] = codes[q];
        float sub = 0.0f;
#pragma unroll
        for (uint32_t q = 0u; q < 4u; ++q) {                             /* eight columns at a time */
            const uint64_t cw = c[q];
            const uint32_t l0 = table[cw & 0xFFu], l1 = table[(cw >> 8) & 0xFFu], l2 = table[(cw >> 16) & 0xFFu];
            const uint32_t l3 = table[(cw >> 24) & 0xFFu], l4 = table[(cw >> 32) & 0xFFu], l5 = table[(cw >> 40) & 0xFFu];
            const uint32_t l6 = table[(cw >> 48) & 0xFFu], l7 = table[(cw >> 56) & 0xFFu];
            const uint4 xv = xs[q * S + k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE + lane];
            sub = amd_rocm6_wave64__zzpackage_dot2(l0 | (l2 << 16), xv.x, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(l4 | (l6 << 16), xv.y, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(l1 | (l3 << 16), xv.z, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(l5 | (l7 << 16), xv.w, sub);
        }
        part += (k + 1u < HELD || real) ? sub : 0.0f;
    }
    return part;
}

/* ⭐ FOUR ROWS FOLDED IN ONE PASS — a wave's parts of four rows summed across its lanes by the butterfly every row kernel
 *   here uses (lanes 32 apart, then 16, 8, 4, 2, 1), with the first two steps shared: 32 apart the lower half keeps rows
 *   0 and 2 and hands over 1 and 3, 16 apart each quarter keeps one row. Each addition is between the same two values as
 *   in the one-row fold, at most in the other order, so every row's sum is that fold's to the bit: 7 exchanges where
 *   four folds took 24, and the four results are rounded to halves by four lanes at once. Row 0 ends in lane 0, row 1
 *   in lane 32, row 2 in lane 16, row 3 in lane 48. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD      4u        /* rows a wave sums before it folds them */
/* The value of the lane `m` away (`lane ^ m`, `m` under 32): `ds_swizzle`, whose pattern is in the instruction, where a
 * shuffle needs a register a distance holding every lane's address, live through the whole kernel. */
template <uint32_t M>
static __device__ inline float amd_rocm6_wave64__zzprivate_xor(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), (int)(0x1Fu | (M << 10))));
}
static __device__ inline float amd_rocm6_wave64__zzprivate_fold4(float p0, float p1, float p2, float p3, uint32_t lane) {
    const bool upper = (lane & 32u) != 0u, second = (lane & 16u) != 0u;
    const float a = (upper ? p1 : p0) + __shfl_xor(upper ? p0 : p1, 32);
    const float b = (upper ? p3 : p2) + __shfl_xor(upper ? p2 : p3, 32);
    float v = (second ? b : a) + amd_rocm6_wave64__zzprivate_xor<16u>(second ? a : b);
    v += amd_rocm6_wave64__zzprivate_xor<8u>(v);
    v += amd_rocm6_wave64__zzprivate_xor<4u>(v);
    v += amd_rocm6_wave64__zzprivate_xor<2u>(v);
    v += amd_rocm6_wave64__zzprivate_xor<1u>(v);
    return v;
}
/* Which of the four rows a lane's folded sum is, for the four lanes that hold one. */
static __device__ inline uint32_t amd_rocm6_wave64__zzprivate_fold_row(uint32_t lane) { return ((lane >> 5) & 1u) | ((lane >> 3) & 2u); }

/* ⭐ THE ROWS A WAVE TAKES, AS SCALARS: the wave's number is the same in every lane of it, and said so — read from the first
 *   lane — the rows, their addresses and the loop around them are the compiler's scalar registers; worked out from the
 *   lane's own number they are 64-bit products in every lane, three a row. A wave takes rows `first`, `first + step`, …
 *   four at a time, and folds each four once. */
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_exact_rows4(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                       const uint16_t* x, uint64_t rows, uint64_t cols,
                                                                       unsigned int* over) {
    const uint32_t FOLD = AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD;
    extern __shared__ uint4 room[];
    uint4* xs = room;
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzpackage_tables(low, high);
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)__builtin_amdgcn_readfirstlane((int)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE));
    const uint64_t blocks = cols / 32u, row_bytes = cols / 2u;
    amd_rocm6_wave64__zzprivate_stage_x<HELD>(xs, x, cols);
    __syncthreads();
    /* the lane's last block: its own, or — where the last blocks fill no more than half the wave — its half's */
    const uint32_t used = (uint32_t)(blocks - (uint64_t)(HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const bool split = used <= AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u;
    const uint32_t pos = split ? lane % (AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u) : lane;
    const bool real = pos < used;
    const uint32_t last = pos + (HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t step = (uint64_t)nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    for (uint64_t r0 = (uint64_t)nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; r0 < rows; r0 += FOLD * step) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        if (HELD == 1u && r0 + 3u * step < rows) {                       /* a lane's one block: four whole rows at once */
            amd_rocm6_wave64__global_byte* rp[4];
#pragma unroll
            for (uint32_t a = 0u; a < 4u; ++a) rp[a] = (amd_rocm6_wave64__global_byte*)(weights + (r0 + a * step) * row_bytes);
            const bool own = lane < blocks;
            float four[4];
            amd_rocm6_wave64__zzprivate_part1xr<4u, 4u>(rp, own ? lane * 16u : 0u, own, xs, lane, low, high, 0, four);
            p3 = four[0]; p2 = four[1]; p1 = four[2]; p0 = four[3];
        } else {
#pragma unroll 1
            for (uint32_t a = 0u; a < FOLD; a += 2u) {                   /* p3 ends as the first row's, p0 the last's */
                const uint64_t ra = r0 + a * step, rb = ra + step;
                float two[2] = { 0.0f, 0.0f };
                if (rb < rows)
                    amd_rocm6_wave64__zzprivate_part4x2<HELD>((amd_rocm6_wave64__global_byte*)(weights + ra * row_bytes),
                                                              (amd_rocm6_wave64__global_byte*)(weights + rb * row_bytes), split, last, real,
                                                              xs, lane, low, high, two);
                else if (ra < rows) {                                    /* a wave's last row alone, its blocks the lane's own */
                    const bool own = lane + (HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE < blocks;
                    two[0] = amd_rocm6_wave64__zzprivate_part4<HELD>((amd_rocm6_wave64__global_byte*)(weights + ra * row_bytes),
                                                                     own ? (lane + (HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) * 16u : 0u,
                                                                     own, xs, lane, low, high);
                }
                p3 = p1; p2 = p0; p1 = two[0]; p0 = two[1];
            }
        }
        /* the row this lane writes, its scale read before the fold so the fold covers the wait */
        const uint64_t r = r0 + amd_rocm6_wave64__zzprivate_fold_row(lane) * step;
        const bool mine = (lane & 15u) == 0u && r < rows;
        uint32_t scale_bits = 0u;
        if (mine) scale_bits = (uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8);
        const float sum = amd_rocm6_wave64__zzprivate_fold4(p3, p2, p1, p0, lane);
        if (mine) out[r] = nn__kernels__zzabi_to_half(sum * nn__silicon__half_to_float((uint16_t)scale_bits), over);
    }
}

/* A kernel for each count of `x` blocks a lane holds, so a row needs only the registers its width uses. */

static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_exact_4_h1(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                                                 const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_exact_rows4<1u>(out, weights, luts, x, rows, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_exact_4_h2(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                                                 const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_exact_rows4<2u>(out, weights, luts, x, rows, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_exact_4_h3(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                                                 const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_exact_rows4<3u>(out, weights, luts, x, rows, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_exact_4_h4(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                                                 const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_exact_rows4<4u>(out, weights, luts, x, rows, cols, over);
}

/* ── THE EXACT GEMV AT 8 BITS — the 4-bit kernel's shape, a byte a code ──────────────────────────────────
 * A wave a row, a lane's blocks of `x` held as halves. A code is a byte, so a block of 32 columns is 32 bytes,
 * four 8-byte loads; its level is looked up in nn's 256-entry fp16 table, which stays in the L1 — a table
 * shared memory would hold needs a barrier to fill, and a card barrier belongs only in a fold. Two levels
 * make a half pair and `v_dot2_f32_f16` takes it, as at 4 bits. */
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_exact_rows8(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                       const uint16_t* x, uint64_t rows, uint64_t cols,
                                                                       unsigned int* over) {
    const uint32_t FOLD = AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD;
    extern __shared__ uint4 room[];
    uint4* xs = room;
    uint16_t* table = (uint16_t*)(room + 4u * HELD * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);   /* nn's 8-bit levels, read here
                                                                                             rather than gathered a code at a time */
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < 256u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        table[e] = nn__turboquant__zzabi_levels(8u)[e];
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)__builtin_amdgcn_readfirstlane((int)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE));
    const uint64_t blocks = cols / 32u, row_bytes = cols;
    amd_rocm6_wave64__zzprivate_stage_x<HELD>(xs, x, cols);
    __syncthreads();
    const bool real = lane + (HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE < blocks;
    const uint32_t last = real ? (lane + (HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) * 32u : 0u;
    const uint64_t step = (uint64_t)nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    for (uint64_t r0 = (uint64_t)nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; r0 < rows; r0 += FOLD * step) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll 1
        for (uint32_t a = 0u; a < FOLD; ++a) {
            const uint64_t r = r0 + a * step;
            const float part = r < rows ? amd_rocm6_wave64__zzprivate_part8<HELD>((amd_rocm6_wave64__global_byte*)(weights + r * row_bytes),
                                                                                  last, real, xs, lane, table)
                                        : 0.0f;
            p3 = p2; p2 = p1; p1 = p0; p0 = part;
        }
        const uint64_t r = r0 + amd_rocm6_wave64__zzprivate_fold_row(lane) * step;
        const bool mine = (lane & 15u) == 0u && r < rows;
        uint32_t scale_bits = 0u;
        if (mine) scale_bits = (uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8);
        const float sum = amd_rocm6_wave64__zzprivate_fold4(p3, p2, p1, p0, lane);
        if (mine) out[r] = nn__kernels__zzabi_to_half(sum * nn__silicon__half_to_float((uint16_t)scale_bits), over);
    }
}


static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_exact_8_h1(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                                                 const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_exact_rows8<1u>(out, weights, luts, x, rows, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_exact_8_h2(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                                                 const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_exact_rows8<2u>(out, weights, luts, x, rows, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_exact_8_h3(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                                                 const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_exact_rows8<3u>(out, weights, luts, x, rows, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_exact_8_h4(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                                                 const uint16_t* x, uint64_t rows, uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_exact_rows8<4u>(out, weights, luts, x, rows, cols, over);
}

/* Whether a call is one the kernels here cover: 4 or 8 bits, whole blocks of 32 columns, rows 8-byte aligned,
 * an `x` the lanes can hold, and the rows and scales it names actually present. */
static inline bool amd_rocm6_wave64__zzprivate_fits(const uint8_t* weights, uint64_t w_room, uint64_t l_room, uint64_t d,
                                                    uint64_t rows, uint64_t cols) {
    return (d == 4u || d == 8u) && cols % 32u == 0u && cols != 0u
        && cols <= 32u * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE * AMD_ROCM6_WAVE64__ZZPRIVATE_HELD
        && ((uint64_t)(uintptr_t)weights) % 8u == 0u && rows <= w_room / (cols * d / 8u) && rows <= l_room / 2u;
}

static inline uint32_t amd_rocm6_wave64__zzprivate_blocks(const void* kernel, uint64_t rows, size_t room) {
    const uint64_t want = (rows + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t most = amd_rocm6_wave64__zzpackage_resident(kernel, AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, room);
    return want == 0u ? 1u : want < most ? (uint32_t)want : most;
}
/* A row kernel's launch with its room. */
static inline void amd_rocm6_wave64__zzprivate_launch_room(const void* kernel, uint32_t blocks, void** args, size_t room) {
    (void)hipLaunchKernel(kernel, dim3(blocks), dim3(AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE), args, room, 0);
}

/* One launch of a row kernel: four rows a block, up to the most blocks a wide door starts. */
static inline void amd_rocm6_wave64__zzprivate_launch4(const void* kernel, const char* name, uint16_t* out, const uint8_t* weights,
                                                       const uint8_t* luts, const uint16_t* x, uint64_t rows, uint64_t cols,
                                                       unsigned int* over) {
    /* the int8 kernel holds its `x` in registers and is given no room */
    const size_t room = kernel == (const void*)amd_rocm6_wave64__zzprivate_int8_4 ? 0u : amd_rocm6_wave64__zzprivate_room_bytes(cols);
    (void)name;
    const uint32_t blocks = amd_rocm6_wave64__zzprivate_blocks(kernel, rows, room);
    void* args[] = { &out, &weights, &luts, &x, &rows, &cols, &over };
    amd_rocm6_wave64__zzprivate_launch_room(kernel, blocks, args, room);
}

/* An `x` wider than the lanes hold — below, with the groups' sum it is built like. */
static bool amd_rocm6_wave64__zzprivate_wide_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                  uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                  unsigned int* over);

/* A call whose rows are one or two blocks a lane — below, with the groups it is built for. */
static bool amd_rocm6_wave64__zzprivate_units(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, unsigned int* over,
                                              const void* per_row);

/* ── THE GEMV AT 16 BITS — a matrix of plain halves, as the MoE router is packed ──────────────────────────
 * nn's generic body gives a block of 256 lanes a row: lane `l` sums columns 4l..4l+3, then 4(l+256)..4(l+256)+3, and on,
 * each product onto its running sum in that order, and the block's fold adds the 256 parts pairwise — lanes 1 apart,
 * then 2, then 4, up to 128. `MEASURED` (rocprof, a 35B decode): 16.5 us a call, 40 a token, for 1 MB of codes. Here a
 * WAVE takes a row and each of its lanes plays four of those 256 — `l`, `l + 64`, `l + 128`, `l + 192` — over the same
 * columns in the same order; each quarter's parts are added by shuffles 1, 2, 4 … 32 apart, which is the fold's own
 * order, and the four quarters then as the fold adds them, `(q0 + q1) + (q2 + q3)`. The same sums in the same order,
 * so the answer is nn's generic kernel's to the bit (`MEASURED` on gfx906, every dumped output against the generic kernel's; on
 * gfx90a, where the compiler pairs products into packed multiplies in both, unverified). `MEASURED`
 * (rocprof over the bench): 256 x 2048, 15.7 -> 7.0 us. No barrier: a row is a wave's alone. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_halves(uint16_t* out, const uint8_t* weights,
                                                                                             const uint16_t* x, uint64_t rows,
                                                                                             uint64_t cols, unsigned int* over) {
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t PLAYED = 4u * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;      /* the generic body's lanes a row */
    const uint64_t runs = cols / 4u;                                     /* a run is four halves, eight bytes */
    const uint2* xr = (const uint2*)(const void*)x;
    for (uint64_t r = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; r < rows;
         r += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        const uint2* row = (const uint2*)(const void*)(weights + r * cols * 2u);
        float q[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        /* two of each quarter's runs a step, every read issued before the first product: a quarter's runs `j0 + 64 s` and
         * `j0 + 256 + 64 s`, taken in that order */
        for (uint64_t j0 = lane; j0 < runs; j0 += 2u * PLAYED) {
            uint2 c[8], xv[8];
#pragma unroll
            for (uint32_t u = 0u; u < 8u; ++u) {
                const uint64_t j = j0 + (uint64_t)(u / 4u) * PLAYED + (uint64_t)(u % 4u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
                c[u] = row[j < runs ? j : j0];
                xv[u] = xr[j < runs ? j : j0];
            }
#pragma unroll
            for (uint32_t u = 0u; u < 8u; ++u) {
                const uint64_t j = j0 + (uint64_t)(u / 4u) * PLAYED + (uint64_t)(u % 4u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
                if (j < runs) {
                    q[u % 4u] += nn__silicon__half_to_float((uint16_t)(c[u].x & 0xFFFFu)) * nn__silicon__half_to_float((uint16_t)(xv[u].x & 0xFFFFu));
                    q[u % 4u] += nn__silicon__half_to_float((uint16_t)(c[u].x >> 16)) * nn__silicon__half_to_float((uint16_t)(xv[u].x >> 16));
                    q[u % 4u] += nn__silicon__half_to_float((uint16_t)(c[u].y & 0xFFFFu)) * nn__silicon__half_to_float((uint16_t)(xv[u].y & 0xFFFFu));
                    q[u % 4u] += nn__silicon__half_to_float((uint16_t)(c[u].y >> 16)) * nn__silicon__half_to_float((uint16_t)(xv[u].y >> 16));
                }
            }
        }
#pragma unroll
        for (uint32_t s = 0u; s < 4u; ++s)
            for (uint32_t off = 1u; off < AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE; off <<= 1) q[s] += __shfl_xor(q[s], (int)off);
        if (lane == 0u) out[r] = nn__kernels__zzabi_to_half(((q[0] + q[1]) + (q[2] + q[3])) * 1.0f, over);
    }
}
/* Whether the halves' kernel takes the call: 16 bits, rows of whole runs, codes and `x` on 8 bytes, the rows present. */
static inline bool amd_rocm6_wave64__zzprivate_halves_fit(const uint8_t* weights, uint64_t w_room, const uint16_t* x, uint64_t d,
                                                          uint64_t rows, uint64_t cols) {
    return d == 16u && cols != 0u && cols % 4u == 0u && rows != 0u && ((uint64_t)(uintptr_t)weights) % 8u == 0u
        && ((uint64_t)(uintptr_t)x) % 8u == 0u && rows <= w_room / (cols * 2u);
}

/* ⭐ THE EXACT GEMV at 4 and 8 bits, and at 16; anything else is the generic door. */
static void amd_rocm6_wave64__override_turboquant_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room,
                                                       const uint8_t* luts, uint64_t l_room, const uint16_t* x,
                                                       uint64_t d, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (amd_rocm6_wave64__zzprivate_halves_fit(weights, w_room, x, d, rows, cols)) {
        const uint32_t blocks = amd_rocm6_wave64__zzprivate_blocks((const void*)amd_rocm6_wave64__zzprivate_halves, rows, 0u);
        void* args[] = { &out, &weights, &x, &rows, &cols, &over };
        amd_rocm6_wave64__zzprivate_launch_room((const void*)amd_rocm6_wave64__zzprivate_halves, blocks, args, 0u);
        return;
    }
    if (!amd_rocm6_wave64__zzprivate_fits(weights, w_room, l_room, d, rows, cols)) {
        if (!amd_rocm6_wave64__zzprivate_wide_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over))
            nn__turboquant__zzabi_launch_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over);
        return;
    }
    /* a row of one or two blocks a lane, and rows enough: the units kernel, the matrix one group of it */
    if ((cols / 32u + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE <= 2u) {
        nn__turboquant__groups g = {};
        g.codes[0] = (uint64_t)(uintptr_t)weights; g.luts[0] = (uint64_t)(uintptr_t)luts; g.rows[0] = rows; g.d[0] = d; g.out_at[0] = 0u;
        g.count = 1u;
        const bool one = cols <= 32u * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
        const void* per_row = d == 4u ? (one ? (const void*)amd_rocm6_wave64__zzprivate_exact_4_h1 : (const void*)amd_rocm6_wave64__zzprivate_exact_4_h2)
                                      : (one ? (const void*)amd_rocm6_wave64__zzprivate_exact_8_h1 : (const void*)amd_rocm6_wave64__zzprivate_exact_8_h2);
        if (amd_rocm6_wave64__zzprivate_units(out, g, x, cols, over, per_row)) return;
    }
    if (d == 4u) {
        const uint64_t held = (cols / 32u + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
        const void* kernel = held == 1u ? (const void*)amd_rocm6_wave64__zzprivate_exact_4_h1
                           : held == 2u ? (const void*)amd_rocm6_wave64__zzprivate_exact_4_h2
                           : held == 3u ? (const void*)amd_rocm6_wave64__zzprivate_exact_4_h3
                           :              (const void*)amd_rocm6_wave64__zzprivate_exact_4_h4;
        amd_rocm6_wave64__zzprivate_launch4(kernel, "amd_rocm6_wave64__zzprivate_exact_4", out, weights, luts, x, rows, cols, over);
    }
    else {
        const uint64_t held = (cols / 32u + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
        const void* kernel = held == 1u ? (const void*)amd_rocm6_wave64__zzprivate_exact_8_h1
                           : held == 2u ? (const void*)amd_rocm6_wave64__zzprivate_exact_8_h2
                           : held == 3u ? (const void*)amd_rocm6_wave64__zzprivate_exact_8_h3
                           :              (const void*)amd_rocm6_wave64__zzprivate_exact_8_h4;
        amd_rocm6_wave64__zzprivate_launch4(kernel, "amd_rocm6_wave64__zzprivate_exact_8", out, weights, luts, x, rows, cols, over);
    }
}

/* ⭐ THE int8 GEMV at 4 bits; anything else is the generic door. */
static void amd_rocm6_wave64__override_turboquant_gemv_int8(uint16_t* out, const uint8_t* weights, uint64_t w_room,
                                                            const uint8_t* luts, uint64_t l_room, const uint16_t* x,
                                                            uint64_t d, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (d != 4u || !amd_rocm6_wave64__zzprivate_fits(weights, w_room, l_room, d, rows, cols)) {
        nn__turboquant__zzabi_launch_gemv_int8(out, weights, w_room, luts, l_room, x, d, rows, cols, over);
        return;
    }
    amd_rocm6_wave64__zzprivate_launch4((const void*)amd_rocm6_wave64__zzprivate_int8_4, "amd_rocm6_wave64__zzprivate_int8_4",
                                        out, weights, luts, x, rows, cols, over);
}

/* ── THE GROUPED GEMVS — the row kernels' shape over several matrices ──────────────────────────────────
 * nn's definitions: `turboquant_gemv_groups` is a gemv a group over one `x`; `turboquant_gemv_groups_sum` is
 * `out[r] = residual[r] + Σ_i w[out_at[i]] · (W_i[r] · x_i)`. Here, as the single gemvs above: a wave a row, the levels
 * found as they are there — at 4 bits in registers by `v_perm_b32`, at 8 bits from nn's table in the L1 —
 * and a wave's sum a shuffle. What is new is only which matrix a row belongs to: a wave looks its row up in
 * the table, and the width is that group's. Covers groups at 4 and 8 bits over whole blocks of 32 columns
 * with an `x` the lanes can hold; anything else is the generic door. */

/* A lane's part of one row's product over 32 columns, block `b` of the row, `x` as 16 half pairs. */
static __device__ inline float amd_rocm6_wave64__zzprivate_block_dot(uint32_t d, const uint8_t* row, uint64_t b,
                                                                      const uint32_t* low, const uint32_t* high,
                                                                      const uint16_t* table8, const uint32_t* xw, float part) {
    if (d == 4u) {
        const uint64_t* codes = (const uint64_t*)(row + b * 16u);
#pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            const uint32_t four = (uint32_t)(codes[e / 4u] >> (16u * (e % 4u)));
            const uint32_t lo = amd_rocm6_wave64__zzpackage_levels4(low, four);
            const uint32_t hi = amd_rocm6_wave64__zzpackage_levels4(high, four);
            part = amd_rocm6_wave64__zzpackage_dot2(__builtin_amdgcn_perm(hi, lo, 0x05010400u), xw[2u * e + 0u], part);
            part = amd_rocm6_wave64__zzpackage_dot2(__builtin_amdgcn_perm(hi, lo, 0x07030602u), xw[2u * e + 1u], part);
        }
    } else {
        const uint64_t* codes = (const uint64_t*)(row + b * 32u);
#pragma unroll
        for (uint32_t e = 0u; e < 4u; ++e) {
            const uint64_t c = codes[e];
#pragma unroll
            for (uint32_t p = 0u; p < 4u; ++p) {
                const uint32_t pair = (uint32_t)table8[(c >> (16u * p)) & 0xFFu]
                                    | ((uint32_t)table8[(c >> (16u * p + 8u)) & 0xFFu] << 16);
                part = amd_rocm6_wave64__zzpackage_dot2(pair, xw[4u * e + p], part);
            }
        }
    }
    return part;
}


static __device__ inline float amd_rocm6_wave64__zzprivate_row_scale(const nn__turboquant__groups* g, uint64_t i, uint64_t r) {
    const uint8_t* lut = (const uint8_t*)(uintptr_t)g->luts[i];
    return nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2ull] | ((uint32_t)lut[r * 2ull + 1ull] << 8)));
}

/* Where a wave's row is among the groups: a wave's rows only rise, so each is found from the last — the matrix it is in
 * (`at`), that matrix's first row and the fields a row needs, read again only when a row is past the matrix. Found from
 * the first matrix for every row, the matrix is a chain of scalar reads ahead of every row's codes. */
typedef struct amd_rocm6_wave64__row_cursor { uint64_t at, first, rows, d, codes, luts, out_at; } amd_rocm6_wave64__row_cursor;
static __device__ inline void amd_rocm6_wave64__zzprivate_cursor_at(amd_rocm6_wave64__row_cursor* c, const nn__turboquant__groups* g,
                                                                    uint64_t i, uint64_t first) {
    c->at = i; c->first = first; c->rows = g->rows[i]; c->d = g->d[i]; c->codes = g->codes[i]; c->luts = g->luts[i]; c->out_at = g->out_at[i];
}
static __device__ inline void amd_rocm6_wave64__zzprivate_seek(amd_rocm6_wave64__row_cursor* c, const nn__turboquant__groups* g,
                                                               uint64_t t) {
    while (t - c->first >= c->rows) amd_rocm6_wave64__zzprivate_cursor_at(c, g, c->at + 1u, c->first + c->rows);
}

/* ⭐ THE GROUPS' ROWS, AS THE EXACT GEMV'S: `x` in shared memory word-major, the rows a wave takes its scalars, two rows
 *   at a time where both are 4-bit (part4x2, the last block shared where it fills half the wave), folded four at a time —
 *   and a lane's blocks a template count, so a row needs only the registers its width uses. A row's matrix is found from
 *   the wave's own row number, so the lookup and the matrix's codes, scales and width are scalar reads; each row keeps
 *   where its answer goes and its scale until the fold. An 8-bit group's codes are looked up in nn's table and paired in
 *   the same order, a row at a time. */
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_groups_rows(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                       uint64_t cols, unsigned int* over) {
    const uint32_t FOLD = AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD;
    extern __shared__ uint4 room[];
    uint4* xs = room;
    uint16_t* table8 = (uint16_t*)(room + 4u * HELD * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);   /* nn's 8-bit levels, read here
                                                                                              rather than gathered a code at a time */
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzpackage_tables(low, high);
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < 256u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        table8[e] = nn__turboquant__zzabi_levels(8u)[e];
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)__builtin_amdgcn_readfirstlane((int)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE));
    const uint64_t blocks = cols / 32u;
    amd_rocm6_wave64__zzprivate_stage_x<HELD>(xs, x, cols);
    __syncthreads();
    /* a row alone: the lane's own last block; two rows: the same, or its half's where the last blocks fill half the wave */
    const bool own = lane + (HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE < blocks;
    const uint32_t own_last = own ? lane + (HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE : 0u;
    const uint32_t used = (uint32_t)(blocks - (uint64_t)(HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const bool split = used <= AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u;
    const uint32_t pos = split ? lane % (AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u) : lane;
    const bool real = pos < used;
    const uint32_t last = pos + (HELD - 1u) * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    uint64_t total = 0ull;
    for (uint64_t i = 0ull; i < g.count; ++i) total += g.rows[i];
    const uint64_t step = (uint64_t)nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    amd_rocm6_wave64__row_cursor cur;
    amd_rocm6_wave64__zzprivate_cursor_at(&cur, &g, 0u, 0u);
    if (HELD > 1u)
    for (uint64_t t0 = (uint64_t)nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; t0 < total; t0 += FOLD * step) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        uint64_t o0 = 0ull, o1 = 0ull, o2 = 0ull, o3 = 0ull;                  /* each row's place in `out`, */
        amd_rocm6_wave64__global_byte *s0 = 0, *s1 = 0, *s2 = 0, *s3 = 0;    /* and its scale: none past the last row */
#pragma unroll 1
        for (uint32_t a = 0u; a < FOLD; a += 2u) {
            float two[2] = { 0.0f, 0.0f };
            uint64_t o[2] = { 0ull, 0ull };
            amd_rocm6_wave64__global_byte* s[2] = { 0, 0 };
            amd_rocm6_wave64__global_byte* row[2] = { 0, 0 };
            uint32_t d[2] = { 0u, 0u };
#pragma unroll
            for (uint32_t j = 0u; j < 2u; ++j) {
                const uint64_t t = t0 + (a + j) * step;
                if (t < total) {
                    amd_rocm6_wave64__zzprivate_seek(&cur, &g, t);
                    const uint64_t r = t - cur.first;
                    d[j] = (uint32_t)cur.d;
                    row[j] = amd_rocm6_wave64__zzprivate_global(cur.codes) + r * (cols * d[j] / 8u);
                    o[j] = cur.out_at + r;
                    s[j] = amd_rocm6_wave64__zzprivate_global(cur.luts) + r * 2ull;
                }
            }
            if (d[0] == 4u && d[1] == 4u)
                amd_rocm6_wave64__zzprivate_part4x2<HELD>(row[0], row[1], split, last, real, xs, lane, low, high, two);
            else {
#pragma unroll 1
                for (uint32_t j = 0u; j < 2u; ++j) {                     /* one body for both, the row chosen as a scalar */
                    amd_rocm6_wave64__global_byte* rj = j == 0u ? row[0] : row[1];
                    const uint32_t dj = j == 0u ? d[0] : d[1];
                    float v = 0.0f;
                    if (dj == 4u) v = amd_rocm6_wave64__zzprivate_part4<HELD>(rj, own_last * 16u, own, xs, lane, low, high);
                    else if (dj == 8u) v = amd_rocm6_wave64__zzprivate_part8<HELD>(rj, own_last * 32u, own, xs, lane, table8);
                    if (j == 0u) two[0] = v; else two[1] = v;
                }
            }
            p3 = p1; p2 = p0; p1 = two[0]; p0 = two[1];
            o3 = o1; o2 = o0; o1 = o[0]; o0 = o[1];
            s3 = s1; s2 = s0; s1 = s[0]; s0 = s[1];
        }
        const uint32_t f = amd_rocm6_wave64__zzprivate_fold_row(lane);
        const uint64_t at = f == 0u ? o3 : f == 1u ? o2 : f == 2u ? o1 : o0;
        amd_rocm6_wave64__global_byte* s = f == 0u ? s3 : f == 1u ? s2 : f == 2u ? s1 : s0;
        const bool mine = (lane & 15u) == 0u && s != 0;
        uint32_t scale_bits = 0u;
        if (mine) scale_bits = (uint32_t)s[0] | ((uint32_t)s[1] << 8);
        const float sum = amd_rocm6_wave64__zzprivate_fold4(p3, p2, p1, p0, lane);
        if (mine) out[at] = nn__kernels__zzabi_to_half(sum * nn__silicon__half_to_float((uint16_t)scale_bits), over);
    }
    /* ...and where a lane holds one block, a row is a kilobyte a wave at 4 bits: the four rows' matrices looked up at once,
     *   four 4-bit rows decoded together (two at 8 bits) */
    else
    for (uint64_t t0 = (uint64_t)nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; t0 < total; t0 += FOLD * step) {
        /* the four rows' matrices, their places in `out` and their scales — none past the last row */
        amd_rocm6_wave64__global_byte* row[4] = { 0, 0, 0, 0 };
        amd_rocm6_wave64__global_byte* sc[4] = { 0, 0, 0, 0 };
        uint64_t o[4] = { 0ull, 0ull, 0ull, 0ull };
        uint32_t d[4] = { 0u, 0u, 0u, 0u };
#pragma unroll
        for (uint32_t a = 0u; a < 4u; ++a) {
            const uint64_t t = t0 + a * step;
            if (t < total) {
                amd_rocm6_wave64__zzprivate_seek(&cur, &g, t);
                const uint64_t r = t - cur.first;
                d[a] = (uint32_t)cur.d;
                row[a] = amd_rocm6_wave64__zzprivate_global(cur.codes) + r * (cols * d[a] / 8u);
                o[a] = cur.out_at + r;
                sc[a] = amd_rocm6_wave64__zzprivate_global(cur.luts) + r * 2ull;
            }
        }
        float p[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (HELD == 1u && d[0] == 4u && d[1] == 4u && d[2] == 4u && d[3] == 4u)
            amd_rocm6_wave64__zzprivate_part1xr<4u, 4u>(row, own ? lane * 16u : 0u, own, xs, lane, low, high, 0, p);
        else {
#pragma unroll 1
            for (uint32_t a = 0u; a < FOLD; a += 2u) {                   /* one body for both pairs, the rows chosen as scalars */
                amd_rocm6_wave64__global_byte* pr[2] = { a == 0u ? row[0] : row[2], a == 0u ? row[1] : row[3] };
                const uint32_t d0 = a == 0u ? d[0] : d[2], d1 = a == 0u ? d[1] : d[3];
                float two[2] = { 0.0f, 0.0f };
                if (d0 == 4u && d1 == 4u)
                    amd_rocm6_wave64__zzprivate_part4x2<HELD>(pr[0], pr[1], split, last, real, xs, lane, low, high, two);
                else if (HELD == 1u && d0 == 8u && d1 == 8u)
                    amd_rocm6_wave64__zzprivate_part1xr<8u, 2u>(pr, own ? lane * 32u : 0u, own, xs, lane, 0, 0, table8, two);
                else {
#pragma unroll 1
                    for (uint32_t j = 0u; j < 2u; ++j) {                 /* one body for both, the row chosen as a scalar */
                        amd_rocm6_wave64__global_byte* rj = j == 0u ? pr[0] : pr[1];
                        const uint32_t dj = j == 0u ? d0 : d1;
                        float v = 0.0f;
                        if (dj == 4u) v = amd_rocm6_wave64__zzprivate_part4<HELD>(rj, own_last * 16u, own, xs, lane, low, high);
                        else if (dj == 8u) v = amd_rocm6_wave64__zzprivate_part8<HELD>(rj, own_last * 32u, own, xs, lane, table8);
                        if (j == 0u) two[0] = v; else two[1] = v;
                    }
                }
                if (a == 0u) { p[0] = two[0]; p[1] = two[1]; } else { p[2] = two[0]; p[3] = two[1]; }
            }
        }
        const uint32_t f = amd_rocm6_wave64__zzprivate_fold_row(lane);
        const uint64_t at = f == 0u ? o[0] : f == 1u ? o[1] : f == 2u ? o[2] : o[3];
        amd_rocm6_wave64__global_byte* s = f == 0u ? sc[0] : f == 1u ? sc[1] : f == 2u ? sc[2] : sc[3];
        const bool mine = (lane & 15u) == 0u && s != 0;
        uint32_t scale_bits = 0u;
        if (mine) scale_bits = (uint32_t)s[0] | ((uint32_t)s[1] << 8);
        const float sum = amd_rocm6_wave64__zzprivate_fold4(p[0], p[1], p[2], p[3], lane);
        if (mine) out[at] = nn__kernels__zzabi_to_half(sum * nn__silicon__half_to_float((uint16_t)scale_bits), over);
    }
}

/* `RW` sums across the wave at once, each a butterfly — partners `32, 16, …, 1` lanes apart, a lane's own part first. The
 *   first steps also trade halves of the set: a lane keeps the sums of half its rows and hands the other half to its
 *   partner, until it carries one, so `RW` rows cost `RW - 1 + log2(64 / RW)` exchanges, not `6 · RW`. A row's sum is its own
 *   butterfly's, the same pairs added in the same order, and it lands in every lane of row `a`'s slice, `64 / RW · a` on. */
template <uint32_t RW>
static __device__ inline float amd_rocm6_wave64__zzprivate_sums(float* part, uint32_t lane) {
    static_assert(RW == 1u || RW == 2u || RW == 4u, "sums' exchanges are written for 1, 2 or 4 rows");
    if (RW == 4u) {
        const bool up = (lane & 32u) != 0u;
        const float give0 = up ? part[0] : part[2], give1 = up ? part[1] : part[3];
        const float keep0 = up ? part[2] : part[0], keep1 = up ? part[3] : part[1];
        part[0] = keep0 + __shfl_xor(give0, 32);
        part[1] = keep1 + __shfl_xor(give1, 32);
    }
    if (RW >= 2u) {
        const uint32_t m = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / RW;
        const bool up = (lane & m) != 0u;
        const float give = up ? part[0] : part[1], keep = up ? part[1] : part[0];
        part[0] = keep + __shfl_xor(give, (int)m);
    }
    for (uint32_t off = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / RW / 2u; off > 0u; off >>= 1) part[0] += __shfl_xor(part[0], (int)off);
    return part[0];
}

/* ⭐ THE GROUPS' ROWS, SEVERAL A WAVE AT ONCE, WHERE A ROW IS ONE OR TWO BLOCKS A LANE — the 35B's projections at its
 *   hidden 2048 (the experts' gate and up, the DeltaNet's and attention's inputs, the head) and the 122B's at 3072. A row
 *   there is one 16-byte load a lane at 4 bits, and the kernel above asks for the next only once this one is summed across
 *   the wave and written. Here a wave takes a UNIT — four rows of a 4-bit group, two of an 8-bit one, the same 64 bytes a
 *   lane — asks for all of its codes at once, reads `x` from shared memory once for all its rows, and sums them across the
 *   wave together. `MEASURED` (test/src_gemv_27b_bench.cpp, MI50, back to back): the 35B's gate and up 38.7 -> 25.2 us,
 *   its DeltaNet inputs 74.0 -> 45.6, its attention q and k 55.1 -> 34.3, its head 1032 -> 880; the 122B's gate and up
 *   96.7 -> 62.6.
 *   ⛳ WHICH UNIT A WAVE TAKES IS ITS OWN, NOT ITS LANES': the wave's index is read as one value, so its group, the group's
 *   width and where its codes are come from the launch's arguments as scalar loads — by a lane's own index each would be a
 *   load through memory a lane at a time, waited on before the codes could be asked for.
 *   Each row's sum is the kernel above's — the same blocks to the same lanes, the same order, the same butterfly — so the
 *   answer is the same, bit for bit (`REASONED` so, and `MEASURED` identical in the bench's dumps on gfx906). */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_UNIT_ROWS 4u       /* a unit's rows at 4 bits; at 8 bits half as many */
static __host__ __device__ inline uint64_t amd_rocm6_wave64__zzprivate_unit_rows(uint64_t d) {
    return d == 4u ? AMD_ROCM6_WAVE64__ZZPRIVATE_UNIT_ROWS : AMD_ROCM6_WAVE64__ZZPRIVATE_UNIT_ROWS / 2u;
}
/* One unit: `RU` rows of a `D`-bit group from `r0`, row `r` written at `out[r]` by the first lane of its slice of the wave.
 * ⛳ EVERY LANE SCALES ITS SLICE'S SUM, though one writes it: read with the codes the scale costs no wait; read by the
 *   writing lane alone it would be read after the sum, a second wait a unit. */
template <uint32_t HELD, uint32_t RU, uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_groups_unit(uint16_t* out, const uint8_t* codes, const uint8_t* lut, uint64_t rows,
                                                                      uint64_t r0, uint64_t cols, const uint4* xs, const uint16_t* table8,
                                                                      const uint32_t* low, const uint32_t* high, const uint32_t* at,
                                                                      const bool* real, uint32_t lane, unsigned int* over) {
    const uint32_t slice = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / RU, mine = lane / slice;   /* the row whose sum this lane writes */
    const uint64_t row_bytes = cols * D / 8u, mrow = r0 + mine < rows ? r0 + mine : rows - 1u;
    uint64_t cur[RU][HELD][D / 2u];                                    /* a lane's block of a row: 16 bytes at 4 bits, 32 at 8 */
#pragma unroll
    for (uint32_t a = 0u; a < RU; ++a) {
        const uint8_t* row = codes + (r0 + a < rows ? r0 + a : rows - 1u) * row_bytes;
#pragma unroll
        for (uint32_t k = 0u; k < HELD; ++k) {
            const uint64_t* c = (const uint64_t*)(row + (uint64_t)at[k] * 4u * D);
#pragma unroll
            for (uint32_t w = 0u; w < D / 2u; ++w) cur[a][k][w] = c[w];
        }
    }
    const uint32_t lo = lut[mrow * 2ull], hi = lut[mrow * 2ull + 1ull];
    __builtin_amdgcn_sched_barrier(0);                                 /* the scale's loads stay with the codes' */
    const float scale = nn__silicon__half_to_float((uint16_t)(lo | (hi << 8)));
    float part[RU];
#pragma unroll
    for (uint32_t a = 0u; a < RU; ++a) part[a] = 0.0f;
#pragma unroll
    for (uint32_t k = 0u; k < HELD; ++k) {
        float sub[RU];
#pragma unroll
        for (uint32_t a = 0u; a < RU; ++a) sub[a] = 0.0f;
#pragma unroll
        for (uint32_t q = 0u; q < 4u; ++q) {                             /* eight columns at a time */
            const uint4 xv = xs[4u * at[k] + q];
#pragma unroll
            for (uint32_t a = 0u; a < RU; ++a) {
                uint32_t pairs[4];
                if (D == 4u) {
                    amd_rocm6_wave64__zzprivate_levels8(low, high, (uint32_t)(cur[a][k][q / 2u] >> (32u * (q % 2u))), pairs);
                } else {                                                 /* a word of eight codes, a byte each */
                    const uint64_t c = cur[a][k][q % (D / 2u)];
                    const uint32_t l0 = table8[c & 0xFFu], l1 = table8[(c >> 8) & 0xFFu], l2 = table8[(c >> 16) & 0xFFu];
                    const uint32_t l3 = table8[(c >> 24) & 0xFFu], l4 = table8[(c >> 32) & 0xFFu], l5 = table8[(c >> 40) & 0xFFu];
                    const uint32_t l6 = table8[(c >> 48) & 0xFFu], l7 = table8[(c >> 56) & 0xFFu];
                    pairs[0] = l0 | (l2 << 16); pairs[1] = l4 | (l6 << 16); pairs[2] = l1 | (l3 << 16); pairs[3] = l5 | (l7 << 16);
                }
                sub[a] = amd_rocm6_wave64__zzpackage_dot2(pairs[0], xv.x, sub[a]);
                sub[a] = amd_rocm6_wave64__zzpackage_dot2(pairs[1], xv.y, sub[a]);
                sub[a] = amd_rocm6_wave64__zzpackage_dot2(pairs[2], xv.z, sub[a]);
                sub[a] = amd_rocm6_wave64__zzpackage_dot2(pairs[3], xv.w, sub[a]);
            }
        }
#pragma unroll
        for (uint32_t a = 0u; a < RU; ++a) part[a] += real[k] ? sub[a] : 0.0f;
    }
    const float v = amd_rocm6_wave64__zzprivate_sums<RU>(part, lane) * scale;
    if (lane % slice == 0u && r0 + mine < rows) out[r0 + mine] = nn__kernels__zzabi_to_half(v, over);
}
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_groups_units(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                       uint64_t cols, unsigned int* over) {
    extern __shared__ uint4 room[];
    uint4* xs = room;
    uint16_t* table8 = (uint16_t*)(room + cols / 8u);  /* nn's 8-bit levels, read here rather than gathered a code at a time */
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzpackage_tables(low, high);
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < 256u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        table8[e] = nn__turboquant__zzabi_levels(8u)[e];
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)__builtin_amdgcn_readfirstlane((int)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE));
    const uint64_t blocks = cols / 32u;
    for (uint64_t e = nn__silicon__lane(); e < cols / 8u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
        uint32_t v[4];
        amd_rocm6_wave64__zzprivate_x8(x + 8u * e, v);
        xs[e] = make_uint4(v[0], v[1], v[2], v[3]);
    }
    __syncthreads();
    uint32_t at[HELD];
    bool real[HELD];
#pragma unroll
    for (uint32_t k = 0u; k < HELD; ++k) {
        const uint64_t b = lane + (uint64_t)k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
        real[k] = b < blocks;
        at[k] = real[k] ? (uint32_t)b : 0u;
    }
    /* the units of each group in turn, the last of a group's holding what is left of it; a wave's units rise, so the group
     * it is in only ever moves forward */
    uint64_t i = 0u, first = 0u, units = (g.rows[0] + amd_rocm6_wave64__zzprivate_unit_rows(g.d[0]) - 1u) / amd_rocm6_wave64__zzprivate_unit_rows(g.d[0]);
    for (uint64_t u = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; ; u += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        while (u >= first + units && i < g.count) {
            first += units;
            ++i;
            units = i < g.count ? (g.rows[i] + amd_rocm6_wave64__zzprivate_unit_rows(g.d[i]) - 1u) / amd_rocm6_wave64__zzprivate_unit_rows(g.d[i]) : 0u;
        }
        if (i >= g.count) break;
        uint16_t* o = out + g.out_at[i];
        const uint8_t* codes = (const uint8_t*)(uintptr_t)g.codes[i];
        const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
        if (g.d[i] == 4u)
            amd_rocm6_wave64__zzprivate_groups_unit<HELD, AMD_ROCM6_WAVE64__ZZPRIVATE_UNIT_ROWS, 4u>(
                o, codes, lut, g.rows[i], (u - first) * AMD_ROCM6_WAVE64__ZZPRIVATE_UNIT_ROWS, cols, xs, table8, low, high, at, real, lane, over);
        else
            amd_rocm6_wave64__zzprivate_groups_unit<HELD, AMD_ROCM6_WAVE64__ZZPRIVATE_UNIT_ROWS / 2u, 8u>(
                o, codes, lut, g.rows[i], (u - first) * (AMD_ROCM6_WAVE64__ZZPRIVATE_UNIT_ROWS / 2u), cols, xs, table8, low, high, at, real, lane, over);
    }
}

/* ⭐ THE GROUPS' WEIGHTED SUM, FOUR ROWS A WAVE: each (group, block) pair's `x` — 32 halves, read from memory, since a
 *   row may be wider than shared memory would hold for this many waves — is read once and put in levels8's order
 *   with four `v_perm_b32`, then applied to four rows' codes, where each row read it again: the 27B's down reads
 *   34 KB of `x` a row, more than the L1 keeps. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS  4u       /* rows a wave sums at once */
static __device__ inline void amd_rocm6_wave64__zzprivate_x8_order(const uint4 v, uint32_t* out) {
    out[0] = __builtin_amdgcn_perm(v.y, v.x, 0x05040100u);   /* (x0,x2) */
    out[1] = __builtin_amdgcn_perm(v.w, v.z, 0x05040100u);   /* (x4,x6) */
    out[2] = __builtin_amdgcn_perm(v.y, v.x, 0x07060302u);   /* (x1,x3) */
    out[3] = __builtin_amdgcn_perm(v.w, v.z, 0x07060302u);   /* (x5,x7) */
}
/* Four rows' sums over one block of 32 columns: `x`'s block in levels8's order (`xp`), row `rr[a]`'s codes `row_bytes` apart.
 *   At 4 bits every row's 16 bytes are loaded before the first is decoded, so the four loads are in flight together. */
static __device__ inline void amd_rocm6_wave64__zzprivate_block_rows(uint32_t d, const uint8_t* codes, uint64_t row_bytes, const uint64_t* rr,
                                                                     uint64_t b, const uint32_t* xp,
                                                                     const uint32_t* low, const uint32_t* high, const uint16_t* table8,
                                                                     float* sub) {
    const uint32_t RW = AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS;
    if (d == 4u) {
        uint64_t c[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS][2];
#pragma unroll
        for (uint32_t a = 0u; a < RW; ++a) {
            const uint64_t* cp = (const uint64_t*)(codes + rr[a] * row_bytes + b * 16u);
            c[a][0] = cp[0]; c[a][1] = cp[1];
        }
#pragma unroll
        for (uint32_t a = 0u; a < RW; ++a) {
            float s = 0.0f;
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) {
                uint32_t pairs[4];
                amd_rocm6_wave64__zzprivate_levels8(low, high, (uint32_t)(c[a][q / 2u] >> (32u * (q % 2u))), pairs);
#pragma unroll
                for (uint32_t p = 0u; p < 4u; ++p) s = amd_rocm6_wave64__zzpackage_dot2(pairs[p], xp[4u * q + p], s);
            }
            sub[a] = s;
        }
    } else {
#pragma unroll 1
        for (uint32_t a = 0u; a < RW; ++a) {
            const uint64_t* cp = (const uint64_t*)(codes + rr[a] * row_bytes + b * 32u);
            float s = 0.0f;
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) {
                const uint64_t cw = cp[q];
                const uint32_t l0 = table8[cw & 0xFFu], l1 = table8[(cw >> 8) & 0xFFu], l2 = table8[(cw >> 16) & 0xFFu];
                const uint32_t l3 = table8[(cw >> 24) & 0xFFu], l4 = table8[(cw >> 32) & 0xFFu], l5 = table8[(cw >> 40) & 0xFFu];
                const uint32_t l6 = table8[(cw >> 48) & 0xFFu], l7 = table8[(cw >> 56) & 0xFFu];
                s = amd_rocm6_wave64__zzpackage_dot2(l0 | (l2 << 16), xp[4u * q + 0u], s);
                s = amd_rocm6_wave64__zzpackage_dot2(l4 | (l6 << 16), xp[4u * q + 1u], s);
                s = amd_rocm6_wave64__zzpackage_dot2(l1 | (l3 << 16), xp[4u * q + 2u], s);
                s = amd_rocm6_wave64__zzpackage_dot2(l5 | (l7 << 16), xp[4u * q + 3u], s);
            }
            sub[a] = s;
        }
    }
}
/* ⛳ EACH GROUP'S CODES, SCALES, WIDTH AND WEIGHT ARE STAGED WITH THE LEVELS, before the one barrier: a lane's group is its
 *   own, and read from the launch's arguments a lane at a time each would be a load through memory, waited on before the
 *   row's codes could be asked for. And the four rows' sums go across the wave together (`sums`), the same butterflies.
 *   `MEASURED` (test/src_gemv_27b_bench.cpp, MI50, back to back): the 27B's down 100.9 -> 91.0 us, every output the same. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROOM  1024u    /* the levels' 512 bytes, then the groups' table */
static __device__ inline void amd_rocm6_wave64__zzprivate_groups_sum_rows(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                           const uint16_t* w, const uint16_t* residual,
                                                                           uint64_t rows, uint64_t cols, unsigned int* over) {
    const uint32_t RW = AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS;
    extern __shared__ uint4 room[];
    uint16_t* table8 = (uint16_t*)room;   /* as groups' — the one barrier is before the first row */
    uint64_t* group_codes = (uint64_t*)(room + 32u);
    uint64_t* group_luts = group_codes + NN__TURBOQUANT__GROUPS_MAX;
    float* group_weight = (float*)(group_luts + NN__TURBOQUANT__GROUPS_MAX);
    uint32_t* group_d = (uint32_t*)(group_weight + NN__TURBOQUANT__GROUPS_MAX);
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < 256u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        table8[e] = nn__turboquant__zzabi_levels(8u)[e];
    if (nn__silicon__lane() < g.count) {
        const uint32_t i = (uint32_t)nn__silicon__lane();
        group_codes[i] = g.codes[i];
        group_luts[i] = g.luts[i];
        group_weight[i] = nn__silicon__half_to_float(w[g.out_at[i]]);
        group_d[i] = (uint32_t)g.d[i];
    }
    __syncthreads();
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzpackage_tables(low, high);
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t blocks = (uint32_t)(cols / 32u), spread = (uint32_t)g.count * blocks;
    for (uint64_t r0 = (nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave) * RW; r0 < rows;
         r0 += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * RW) {
        float part[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS];
        uint64_t rr[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS];
#pragma unroll
        for (uint32_t a = 0u; a < RW; ++a) { part[a] = 0.0f; rr[a] = r0 + a < rows ? r0 + a : r0; }
#pragma unroll 1
        for (uint32_t j = lane; j < spread; j += AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {    /* (group, block) pairs */
            const uint32_t i = j / blocks, b = j - i * blocks;
            const uint32_t d = group_d[i];
            const uint64_t row_bytes = cols * d / 8u;
            const uint8_t* codes = (const uint8_t*)(uintptr_t)group_codes[i];
            const uint8_t* lut = (const uint8_t*)(uintptr_t)group_luts[i];
            const float weight = group_weight[i];
            const uint32_t* xw = (const uint32_t*)(x + (uint64_t)i * cols + (uint64_t)b * 32u);   /* `x` read in words: 4-byte aligned is enough */
            uint32_t xp[16];
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q)
                amd_rocm6_wave64__zzprivate_x8_order(make_uint4(xw[4u * q], xw[4u * q + 1u], xw[4u * q + 2u], xw[4u * q + 3u]), &xp[4u * q]);
            float sub[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS];
            amd_rocm6_wave64__zzprivate_block_rows(d, codes, row_bytes, rr, b, xp, low, high, table8, sub);
#pragma unroll
            for (uint32_t a = 0u; a < RW; ++a)
                part[a] += sub[a] * (nn__silicon__half_to_float((uint16_t)((uint32_t)lut[rr[a] * 2ull] | ((uint32_t)lut[rr[a] * 2ull + 1ull] << 8))) * weight);
        }
        const float v = amd_rocm6_wave64__zzprivate_sums<AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS>(part, lane);
        const uint32_t a = lane / (AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / RW);
        if (lane % (AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / RW) == 0u && r0 + a < rows)
            out[r0 + a] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(residual[r0 + a]) + v, over);
    }
}

/* ⭐ THE GROUPS' WEIGHTED SUM WHERE A ROW IS A FEW BLOCKS — the 35B's routed downs, nine groups of 512 columns, sixteen
 *   blocks a row. The kernel above deals a row's 144 (group, block) pairs to the wave's lanes, three rounds a row, the shared
 *   expert's 8 bits alone in the last with a quarter of the lanes, and a round's codes are asked for only once the round
 *   before is summed: `MEASURED` 27 us for 5 MB (test/src_gemv_27b_bench.cpp, MI50, back to back) whether a wave held one
 *   row, two or four, and 23 us here; the 122B's, 32 blocks a row, 44 and 39.
 *   Here the wave is cut in SLICES of `S` lanes, one row each, a lane a block of it — the same block of every group — and
 *   the groups go by in turn for the whole wave, so a group's width is the wave's to branch on, every lane is busy in every
 *   group, and `GB` groups' codes are asked for at once. Every group's `x` is staged in shared memory once a block, in
 *   levels8's order, with the levels and the groups' table before the one barrier. A row's parts are the same products as
 *   nn's, summed a block at a time across the groups and then across its slice: the order is this kernel's own. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_GB 3u          /* groups whose codes a lane asks for at once */
template <uint32_t S>
static __device__ inline void amd_rocm6_wave64__zzprivate_groups_sum_slices(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                             const uint16_t* w, const uint16_t* residual,
                                                                             uint64_t rows, uint64_t cols, unsigned int* over) {
    const uint32_t R = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / S, GB = AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_GB;
    extern __shared__ uint4 room[];
    const uint32_t blocks = (uint32_t)(cols / 32u), count = (uint32_t)g.count;
    uint4* xs = room;                                                      /* every group's `x`, four quads a block */
    uint16_t* table8 = (uint16_t*)(room + 4u * count * blocks);
    uint64_t* group_codes = (uint64_t*)(table8 + 256u);
    uint64_t* group_luts = group_codes + NN__TURBOQUANT__GROUPS_MAX;
    float* group_weight = (float*)(group_luts + NN__TURBOQUANT__GROUPS_MAX);
    uint32_t* group_d = (uint32_t*)(group_weight + NN__TURBOQUANT__GROUPS_MAX);
    const uint32_t t = (uint32_t)nn__silicon__lane();
    for (uint32_t e = t; e < 256u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        table8[e] = nn__turboquant__zzabi_levels(8u)[e];
    for (uint32_t e = t; e < 4u * count * blocks; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
        uint32_t v[4];
        amd_rocm6_wave64__zzprivate_x8(x + 8u * (uint64_t)e, v);
        xs[e] = make_uint4(v[0], v[1], v[2], v[3]);
    }
    if (t < count) {
        group_codes[t] = g.codes[t];
        group_luts[t] = g.luts[t];
        group_weight[t] = nn__silicon__half_to_float(w[g.out_at[t]]);
        group_d[t] = (uint32_t)g.d[t];
    }
    __syncthreads();
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzpackage_tables(low, high);
    const uint32_t lane = t % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, wave = t / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint32_t mine = lane / S, b = lane % S;                          /* the lane's row of the wave's, and its block */
    for (uint64_t r0 = (nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave) * R; r0 < rows;
         r0 += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * R) {
        const uint64_t r = r0 + mine < rows ? r0 + mine : rows - 1u;
        float part = 0.0f;
        for (uint32_t i0 = 0u; i0 < count; i0 += GB) {
            uint64_t c[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_GB][4];
            uint32_t scale_bits[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_GB];
            uint32_t d[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_GB];
#pragma unroll
            for (uint32_t j = 0u; j < GB; ++j) {
                const uint32_t i = i0 + j < count ? i0 + j : i0;
                d[j] = (uint32_t)__builtin_amdgcn_readfirstlane((int)group_d[i]);
                const uint8_t* lut = (const uint8_t*)(uintptr_t)group_luts[i];
                const uint64_t* cp = (const uint64_t*)((const uint8_t*)(uintptr_t)group_codes[i] + r * (cols * d[j] / 8u) + (uint64_t)b * 4u * d[j]);
                c[j][0] = cp[0]; c[j][1] = cp[1];
                c[j][2] = d[j] == 8u ? cp[2] : 0u;
                c[j][3] = d[j] == 8u ? cp[3] : 0u;
                scale_bits[j] = (uint32_t)lut[r * 2ull] | ((uint32_t)lut[r * 2ull + 1ull] << 8);
            }
#pragma unroll
            for (uint32_t j = 0u; j < GB; ++j) {
                const uint32_t i = i0 + j;
                if (i >= count) continue;
                const uint4* xb = xs + 4u * ((uint64_t)i * blocks + b);
                float sub = 0.0f;
                if (d[j] == 4u) {
#pragma unroll
                    for (uint32_t q = 0u; q < 4u; ++q) {
                        uint32_t pairs[4];
                        amd_rocm6_wave64__zzprivate_levels8(low, high, (uint32_t)(c[j][q / 2u] >> (32u * (q % 2u))), pairs);
                        const uint4 xv = xb[q];
                        sub = amd_rocm6_wave64__zzpackage_dot2(pairs[0], xv.x, sub);
                        sub = amd_rocm6_wave64__zzpackage_dot2(pairs[1], xv.y, sub);
                        sub = amd_rocm6_wave64__zzpackage_dot2(pairs[2], xv.z, sub);
                        sub = amd_rocm6_wave64__zzpackage_dot2(pairs[3], xv.w, sub);
                    }
                } else {
#pragma unroll
                    for (uint32_t q = 0u; q < 4u; ++q) {
                        const uint64_t cw = c[j][q];
                        const uint32_t l0 = table8[cw & 0xFFu], l1 = table8[(cw >> 8) & 0xFFu], l2 = table8[(cw >> 16) & 0xFFu];
                        const uint32_t l3 = table8[(cw >> 24) & 0xFFu], l4 = table8[(cw >> 32) & 0xFFu], l5 = table8[(cw >> 40) & 0xFFu];
                        const uint32_t l6 = table8[(cw >> 48) & 0xFFu], l7 = table8[(cw >> 56) & 0xFFu];
                        const uint4 xv = xb[q];
                        sub = amd_rocm6_wave64__zzpackage_dot2(l0 | (l2 << 16), xv.x, sub);
                        sub = amd_rocm6_wave64__zzpackage_dot2(l4 | (l6 << 16), xv.y, sub);
                        sub = amd_rocm6_wave64__zzpackage_dot2(l1 | (l3 << 16), xv.z, sub);
                        sub = amd_rocm6_wave64__zzpackage_dot2(l5 | (l7 << 16), xv.w, sub);
                    }
                }
                part += sub * (nn__silicon__half_to_float((uint16_t)scale_bits[j]) * group_weight[i]);
            }
        }
        for (uint32_t off = S / 2u; off > 0u; off >>= 1) part += __shfl_xor(part, (int)off);
        if (b == 0u && r0 + mine < rows)
            out[r0 + mine] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(residual[r0 + mine]) + part, over);
    }
}


/* ⭐ ONE MATRIX'S WEIGHTED SUM — `groups_sum` with a single group, as a dense model's MLP down calls it — as a row kernel:
 *   the 4-row kernel above reads each block of `x` from memory once for four rows, through a table of matrices looked up
 *   a lane at a time (`MEASURED`, rocprof over the 27B's down: 1121 VALU instructions a row where its decoding is 748,
 *   three waves a SIMD at 76 registers). Here `x` is staged in shared memory once a block — 36 KB staged at the 27B's 17,408
 *   columns, so a block is sixteen waves and one fills a CU — and a row is a wave's, its matrix, scale and weight
 *   scalars. A lane's blocks are the 4-row kernel's — `lane`, `lane + 64`, … — each weighed by the row's scale times the
 *   group's weight and added in the same order, and the wave's sum is the same butterfly, so each row is that kernel's
 *   to the bit. The next block's codes are loaded while one is decoded.
 *   `x` is chunk-major: block `64n + l`'s word `q` at `xs[256n + 64q + l]`, so a wave's reads are consecutive and every
 *   offset but the chunk's is in the instruction. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_WAVES 16u     /* waves a block of the one-matrix sum */
/* Chunk `n`'s block of a row for this lane, D bits a code: its codes into `c`, the row's first block where it is past the
 * row's end. */
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_dense_load(uint64_t* c, amd_rocm6_wave64__global_byte* row, uint32_t n,
                                                                     uint32_t chunks, bool real, uint32_t lane) {
    const bool in = n + 1u < chunks || (n + 1u == chunks && real);      /* a chunk past the last reads the first too */
    amd_rocm6_wave64__global_word* p = (amd_rocm6_wave64__global_word*)(row + (in ? (n * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE + lane) * 4u * D : 0u));
#pragma unroll
    for (uint32_t h = 0u; h < D / 2u; ++h) c[h] = p[h];
}
/* ...and its sixteen products onto `part`, weighed by `sw` — the scale times the weight — in one rounding, as the 4-row
 * kernel's `part += sub * (scale * weight)` is compiled (`v_fmac_f32`); a block past the row's end is not added at all. */
template <uint32_t D>
static __device__ inline float amd_rocm6_wave64__zzprivate_dense_block(const uint64_t* c, float part, float sw, bool in, const uint4* xb,
                                                                       const uint32_t* low, const uint32_t* high, const uint16_t* table8) {
    const uint32_t W = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    float sub = 0.0f;
    if (D == 4u) {
#pragma unroll
        for (uint32_t q = 0u; q < 4u; ++q) {
            uint32_t pairs[4];
            amd_rocm6_wave64__zzprivate_levels8(low, high, (uint32_t)(c[q / 2u] >> (32u * (q % 2u))), pairs);
            const uint4 xv = xb[q * W];
            sub = amd_rocm6_wave64__zzpackage_dot2(pairs[0], xv.x, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(pairs[1], xv.y, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(pairs[2], xv.z, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(pairs[3], xv.w, sub);
        }
    } else {
#pragma unroll
        for (uint32_t q = 0u; q < 4u; ++q) {
            const uint64_t cw = c[q % (D / 2u)];
            const uint32_t l0 = table8[cw & 0xFFu], l1 = table8[(cw >> 8) & 0xFFu], l2 = table8[(cw >> 16) & 0xFFu];
            const uint32_t l3 = table8[(cw >> 24) & 0xFFu], l4 = table8[(cw >> 32) & 0xFFu], l5 = table8[(cw >> 40) & 0xFFu];
            const uint32_t l6 = table8[(cw >> 48) & 0xFFu], l7 = table8[(cw >> 56) & 0xFFu];
            const uint4 xv = xb[q * W];
            sub = amd_rocm6_wave64__zzpackage_dot2(l0 | (l2 << 16), xv.x, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(l4 | (l6 << 16), xv.y, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(l1 | (l3 << 16), xv.z, sub);
            sub = amd_rocm6_wave64__zzpackage_dot2(l5 | (l7 << 16), xv.w, sub);
        }
    }
    const float next = __builtin_fmaf(sub, sw, part);
    return in ? next : part;
}
/* A lane's part of a row: its blocks two at a time from two sets of registers taking turns, each the next block's load
 * issued before this one is decoded, and an odd last block after. A register a load is filling is never copied, a load
 * and its decode are on the same path, and the compiler is told not to move the load down to its use
 * (`sched_barrier`, an order for the compiler's scheduler, nothing at run time): a copy waits for the load, a load used
 * only under a branch is moved to it, and the scheduler moves it to just before its decode — `MEASURED` in the ISA,
 * each time a wait for the load a few instructions after it. */
template <uint32_t D>
static __device__ inline float amd_rocm6_wave64__zzprivate_dense_part(amd_rocm6_wave64__global_byte* row, float sw, uint32_t chunks,
                                                                      bool real, const uint4* xs, uint32_t lane, const uint32_t* low,
                                                                      const uint32_t* high, const uint16_t* table8) {
    const uint32_t W = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    uint64_t c0[D / 2u], c1[D / 2u];
    amd_rocm6_wave64__zzprivate_dense_load<D>(c0, row, 0u, chunks, real, lane);
    float part = 0.0f;
    uint32_t n = 0u;
#pragma unroll 1
    for (; n + 1u < chunks; n += 2u) {
        amd_rocm6_wave64__zzprivate_dense_load<D>(c1, row, n + 1u, chunks, real, lane);
        __builtin_amdgcn_sched_barrier(0);
        part = amd_rocm6_wave64__zzprivate_dense_block<D>(c0, part, sw, true, xs + n * 4u * W + lane, low, high, table8);
        amd_rocm6_wave64__zzprivate_dense_load<D>(c0, row, n + 2u, chunks, real, lane);
        __builtin_amdgcn_sched_barrier(0);
        part = amd_rocm6_wave64__zzprivate_dense_block<D>(c1, part, sw, n + 2u < chunks || real, xs + (n + 1u) * 4u * W + lane, low,
                                                         high, table8);
    }
    if (n < chunks) part = amd_rocm6_wave64__zzprivate_dense_block<D>(c0, part, sw, real, xs + n * 4u * W + lane, low, high, table8);
    return part;
}
template <uint32_t D>
static __device__ inline void amd_rocm6_wave64__zzprivate_dense_sum_rows(uint16_t* out, const uint8_t* codes, const uint8_t* luts,
                                                                          const uint16_t* w, uint64_t at, const uint16_t* x,
                                                                          const uint16_t* residual, uint64_t rows, uint64_t cols,
                                                                          unsigned int* over) {
    const uint32_t FOLD = AMD_ROCM6_WAVE64__ZZPRIVATE_FOLD, W = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint32_t lanes = AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_WAVES * W;
    extern __shared__ uint4 room[];
    uint16_t* table8 = (uint16_t*)room;                                  /* nn's 8-bit levels, 512 bytes, then `x` */
    uint4* xs = room + 32u;
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < 256u; e += lanes) table8[e] = nn__turboquant__zzabi_levels(8u)[e];
    const uint32_t blocks = (uint32_t)(cols / 32u), chunks = (blocks + W - 1u) / W;
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < chunks * 4u * W; e += lanes) {
        const uint32_t j = (e / (4u * W)) * W + e % W, q = (e / W) % 4u;
        uint32_t v[4] = { 0u, 0u, 0u, 0u };
        if (j < blocks) amd_rocm6_wave64__zzprivate_x8(x + 32u * (uint64_t)j + 8u * q, v);
        xs[e] = make_uint4(v[0], v[1], v[2], v[3]);
    }
    __syncthreads();
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzpackage_tables(low, high);
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % W);
    const uint32_t wave = (uint32_t)__builtin_amdgcn_readfirstlane((int)(nn__silicon__lane() / W));
    const bool real = lane < blocks - (chunks - 1u) * W;                 /* the lane's block in the last chunk is in the row */
    const float weight = nn__silicon__half_to_float(w[at]);
    const uint64_t row_bytes = cols * D / 8u, step = (uint64_t)nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_WAVES;
    for (uint64_t r0 = (uint64_t)nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_WAVES + wave; r0 < rows; r0 += FOLD * step) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll 1
        for (uint32_t a = 0u; a < FOLD; ++a) {                           /* p3 ends as the first row's, p0 the last's */
            const uint64_t r = r0 + a * step;
            float part = 0.0f;
            if (r < rows) {
                const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8)));
                part = amd_rocm6_wave64__zzprivate_dense_part<D>((amd_rocm6_wave64__global_byte*)(codes + r * row_bytes), scale * weight,
                                                                 chunks, real, xs, lane, low, high, table8);
            }
            p3 = p2; p2 = p1; p1 = p0; p0 = part;
        }
        const uint64_t r = r0 + amd_rocm6_wave64__zzprivate_fold_row(lane) * step;
        const bool mine = (lane & 15u) == 0u && r < rows;
        float base = 0.0f;
        if (mine) base = nn__silicon__half_to_float(residual[r]);
        const float sum = amd_rocm6_wave64__zzprivate_fold4(p3, p2, p1, p0, lane);
        if (mine) out[r] = nn__kernels__zzabi_to_half(base + sum, over);
    }
}
#define AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_BOUNDS __launch_bounds__(1024)   /* AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_WAVES waves */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_BOUNDS void amd_rocm6_wave64__zzprivate_dense_sum_4(uint16_t* out, const uint8_t* codes, const uint8_t* luts,
                                                                                                       const uint16_t* w, uint64_t at, const uint16_t* x,
                                                                                                       const uint16_t* residual, uint64_t rows, uint64_t cols,
                                                                                                       unsigned int* over) {
    amd_rocm6_wave64__zzprivate_dense_sum_rows<4u>(out, codes, luts, w, at, x, residual, rows, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_BOUNDS void amd_rocm6_wave64__zzprivate_dense_sum_8(uint16_t* out, const uint8_t* codes, const uint8_t* luts,
                                                                                                       const uint16_t* w, uint64_t at, const uint16_t* x,
                                                                                                       const uint16_t* residual, uint64_t rows, uint64_t cols,
                                                                                                       unsigned int* over) {
    amd_rocm6_wave64__zzprivate_dense_sum_rows<8u>(out, codes, luts, w, at, x, residual, rows, cols, over);
}

static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_h1(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                                                uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_rows<1u>(out, g, x, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_h2(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                                                uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_rows<2u>(out, g, x, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_h3(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                                                uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_rows<3u>(out, g, x, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_h4(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                                                uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_rows<4u>(out, g, x, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_u1(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                                                uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_units<1u>(out, g, x, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_u2(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                                                uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_units<2u>(out, g, x, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_sum(uint16_t* out, nn__turboquant__groups g,
                                                                                                const uint16_t* x, const uint16_t* w,
                                                                                                const uint16_t* residual, uint64_t rows,
                                                                                                uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_sum_rows(out, g, x, w, residual, rows, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_sum_s16(uint16_t* out, nn__turboquant__groups g,
                                                                                                    const uint16_t* x, const uint16_t* w,
                                                                                                    const uint16_t* residual, uint64_t rows,
                                                                                                    uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_sum_slices<16u>(out, g, x, w, residual, rows, cols, over);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_sum_s32(uint16_t* out, nn__turboquant__groups g,
                                                                                                    const uint16_t* x, const uint16_t* w,
                                                                                                    const uint16_t* residual, uint64_t rows,
                                                                                                    uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_sum_slices<32u>(out, g, x, w, residual, rows, cols, over);
}

/* Whether every group is one the kernels here take: 4 or 8 bits, whole blocks, 8-byte aligned codes. */
static inline bool amd_rocm6_wave64__zzprivate_groups_codes_fit(const nn__turboquant__groups* g, uint64_t cols) {
    if (g->count == 0u || g->count > NN__TURBOQUANT__GROUPS_MAX || cols == 0u || cols % 32u != 0u) return false;
    for (uint64_t i = 0u; i < g->count; ++i)
        if ((g->d[i] != 4u && g->d[i] != 8u) || g->codes[i] % 8u != 0u || g->rows[i] == 0u) return false;
    return true;
}
/* ...and, for `groups`, an `x` its lanes can hold. `groups_sum` reads `x` a block at a time instead, so a row of
 * any width fits it — the 27B's down projection is 17,408 columns. */
static inline bool amd_rocm6_wave64__zzprivate_groups_fit(const nn__turboquant__groups* g, uint64_t cols) {
    return amd_rocm6_wave64__zzprivate_groups_codes_fit(g, cols)
        && cols <= 32u * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE * AMD_ROCM6_WAVE64__ZZPRIVATE_HELD;
}

/* A call whose rows are one or two blocks a lane, through the units kernel — when it has rows enough: for every wave the
 * row-a-wave kernel `per_row` would keep on the card, at least four rows where a row is one block a lane and eight where it
 * is two. Fewer, and a wave's unit leaves waves the card has room for unstarted: `MEASURED` (test/src_gemv_27b_bench.cpp,
 * MI50, back to back) the 35B's out-projection, 2048 rows, 23.6 us a row a wave and 27.7 in units, its attention v, 512
 * rows, 7.4 and 10.4, an 8-bit 4096 x 2048 22.0 and 22.4; its attention q and k, 8704 rows, 54.7 and 35.1. At two blocks a
 * lane four rows a wave was not enough: Ministral 3 14B D8's o_proj (5120 x 4096, 8 bits) 42.3 -> 43.1 us, 4-bit 6144 x
 * 4096 29.0 -> 31.0 and 8192 x 4096 33.3 -> 35.2, where the 122B's q, gate and k (16,896 rows) 158.9 -> 97.2 and GLM's
 * gate and up 196 -> 124 — every measured win there has 16,896 rows or more. False when it does not take the call. */
static bool amd_rocm6_wave64__zzprivate_units(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, unsigned int* over,
                                              const void* per_row) {
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    const uint64_t held = (cols / 32u + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t per_wave = held == 2u ? 8u : 4u;
    if (total < per_wave * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS
                   * amd_rocm6_wave64__zzpackage_resident(per_row, AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE,
                                                          amd_rocm6_wave64__zzprivate_room_bytes(cols)))
        return false;
    const void* kernel = held == 1u ? (const void*)amd_rocm6_wave64__zzprivate_groups_u1 : (const void*)amd_rocm6_wave64__zzprivate_groups_u2;
    uint64_t units = 0u;
    for (uint64_t i = 0u; i < g.count; ++i)
        units += (g.rows[i] + amd_rocm6_wave64__zzprivate_unit_rows(g.d[i]) - 1u) / amd_rocm6_wave64__zzprivate_unit_rows(g.d[i]);
    const size_t room = amd_rocm6_wave64__zzprivate_room_bytes(cols);
    const uint32_t blocks = amd_rocm6_wave64__zzprivate_blocks(kernel, units, room);
    void* args[] = { &out, &g, &x, &cols, &over };
    amd_rocm6_wave64__zzprivate_launch_room(kernel, blocks, args, room);
    return true;
}

static void amd_rocm6_wave64__override_gemv_groups(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                   unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_groups_fit(&g, cols)) { nn__turboquant__zzabi_launch_gemv_groups(out, g, x, cols, over); return; }
    const uint64_t held = (cols / 32u + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const void* kernel = held == 1u ? (const void*)amd_rocm6_wave64__zzprivate_groups_h1
                       : held == 2u ? (const void*)amd_rocm6_wave64__zzprivate_groups_h2
                       : held == 3u ? (const void*)amd_rocm6_wave64__zzprivate_groups_h3
                       :              (const void*)amd_rocm6_wave64__zzprivate_groups_h4;
    if (held <= 2u && amd_rocm6_wave64__zzprivate_units(out, g, x, cols, over, kernel)) return;
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    const size_t room = amd_rocm6_wave64__zzprivate_room_bytes(cols);
    const uint32_t blocks = amd_rocm6_wave64__zzprivate_blocks(kernel, total, room);
    void* args[] = { &out, &g, &x, &cols, &over };
    amd_rocm6_wave64__zzprivate_launch_room(kernel, blocks, args, room);
}

/* ⛳ `x` NEEDS ONLY 4-BYTE ALIGNMENT here: the four-row kernel reads `x` a word at a time and the one-matrix kernel a half
 *   at a time; only the slices, which stage it 16 bytes at a time, ask for 16. `MEASURED` (test/src_gemv_27b_bench.cpp, one
 *   MI50): a 5120 x 17408 down whose `x` is 4 bytes off 16 runs in 85 us here against nn's generic body's 835. */
static void amd_rocm6_wave64__override_gemv_groups_sum(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                       const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_groups_codes_fit(&g, cols) || ((uint64_t)(uintptr_t)x) % 4u != 0u) {
        nn__turboquant__zzabi_launch_gemv_groups_sum(out, g, x, w, residual, rows, cols, over);
        return;
    }
    /* one matrix whose `x` shared memory holds: the one-matrix kernel, a block a CU's sixteen waves */
    const uint64_t chunks = (cols / 32u + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const size_t dense_room = 512u + (size_t)chunks * 4u * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE * 16u;
    if (g.count == 1u && dense_room <= 65536u) {
        const void* kernel = g.d[0] == 4u ? (const void*)amd_rocm6_wave64__zzprivate_dense_sum_4
                                          : (const void*)amd_rocm6_wave64__zzprivate_dense_sum_8;
        const uint32_t lanes = AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_WAVES * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
        const uint64_t want = (rows + AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_WAVES - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_DENSE_WAVES;
        const uint32_t most = amd_rocm6_wave64__zzpackage_resident(kernel, lanes, dense_room);
        const uint32_t blocks = want == 0u ? 1u : want < most ? (uint32_t)want : most;
        const uint8_t* codes = (const uint8_t*)(uintptr_t)g.codes[0];
        const uint8_t* luts = (const uint8_t*)(uintptr_t)g.luts[0];
        uint64_t at = g.out_at[0];
        void* args[] = { &out, &codes, &luts, &w, &at, &x, &residual, &rows, &cols, &over };
        (void)hipLaunchKernel(kernel, dim3(blocks), dim3(lanes), args, dense_room, 0);
        return;
    }
    /* a row of 16 or 32 blocks, every group's `x` within 32 KB of shared memory and on 16 bytes (the slices stage it 16
     * bytes at a time): the slices; else four rows a wave */
    const uint64_t blocks_a_row = cols / 32u;
    const void* kernel = (const void*)amd_rocm6_wave64__zzprivate_groups_sum;
    uint32_t rw = AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS;
    size_t room = AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROOM;
    if ((blocks_a_row == 16u || blocks_a_row == 32u) && g.count * cols * 2u <= 32768u && ((uint64_t)(uintptr_t)x) % 16u == 0u) {
        kernel = blocks_a_row == 16u ? (const void*)amd_rocm6_wave64__zzprivate_groups_sum_s16 : (const void*)amd_rocm6_wave64__zzprivate_groups_sum_s32;
        rw = (uint32_t)(AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / blocks_a_row);
        room = (size_t)g.count * cols * 2u + AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROOM;
    }
    const uint32_t blocks = amd_rocm6_wave64__zzprivate_blocks(kernel, (rows + rw - 1u) / rw, room);
    void* args[] = { &out, &g, &x, &w, &residual, &rows, &cols, &over };
    amd_rocm6_wave64__zzprivate_launch_room(kernel, blocks, args, room);
}

/* ⭐ THE EXACT GEMV FOR AN `x` WIDER THAN THE LANES HOLD — GLM's attention output is 16,384 columns, where the row
 *   kernels above stop at 8,192 and nn's generic body read the 27B's down at 47 GB/s. Built as the groups' sum: four
 *   rows a wave, `x` read from memory a block at a time and used for all four, and no barrier. Each row's sum is
 *   taken whole and then scaled, as nn's body does; only the order of the sum differs. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_wide(uint16_t* out, const uint8_t* weights,
                                                                                           const uint8_t* luts, const uint16_t* x,
                                                                                           uint64_t rows, uint64_t cols, uint64_t d,
                                                                                           unsigned int* over) {
    const uint32_t RW = AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS;
    /* nn's 8-bit levels read where they are, kept by the L1: staging them in shared memory would need a barrier */
    const uint16_t* table8 = nn__turboquant__zzabi_levels(8u);
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzpackage_tables(low, high);
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t blocks = cols / 32u, row_bytes = cols * d / 8u;
    for (uint64_t r0 = (nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave) * RW; r0 < rows;
         r0 += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * RW) {
        float part[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS];
        uint64_t rr[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS];
#pragma unroll
        for (uint32_t a = 0u; a < RW; ++a) { part[a] = 0.0f; rr[a] = r0 + a < rows ? r0 + a : r0; }
        /* each wave starts a wave's width further along its rows than the one before, and wraps: where rows are a power of
         * two apart every wave reading the same columns reads the same memory channels */
        const uint64_t turn = (r0 / RW * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) % blocks;
#pragma unroll 1
        for (uint64_t j = lane; j < blocks; j += AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
            const uint64_t b = j + turn < blocks ? j + turn : j + turn - blocks;
            const uint4* xw = (const uint4*)(x + b * 32u);
            uint32_t xp[16];
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) amd_rocm6_wave64__zzprivate_x8_order(xw[q], &xp[4u * q]);
            float sub[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS];
            amd_rocm6_wave64__zzprivate_block_rows((uint32_t)d, weights, row_bytes, rr, b, xp, low, high, table8, sub);
#pragma unroll
            for (uint32_t a = 0u; a < RW; ++a) part[a] += sub[a];
        }
#pragma unroll
        for (uint32_t a = 0u; a < RW; ++a) {
            float v = part[a];
            for (uint32_t off = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u; off > 0u; off >>= 1) v += __shfl_xor(v, (int)off);
            const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)luts[rr[a] * 2ull] | ((uint32_t)luts[rr[a] * 2ull + 1ull] << 8)));
            if (lane == 0u && r0 + a < rows) out[r0 + a] = nn__kernels__zzabi_to_half(v * scale, over);
        }
    }
}
/* Whether the wide kernel takes the call, and its launch if so: 4 or 8 bits, whole blocks, rows 8-byte aligned, `x`
 * 16-byte aligned, and the rows and scales it names present. */
static bool amd_rocm6_wave64__zzprivate_wide_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                  uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                  unsigned int* over) {
    if ((d != 4u && d != 8u) || cols == 0u || cols % 32u != 0u || ((uint64_t)(uintptr_t)weights) % 8u != 0u
        || ((uint64_t)(uintptr_t)x) % 16u != 0u || rows > w_room / (cols * d / 8u) || rows > l_room / 2u)
        return false;
    const uint32_t blocks = amd_rocm6_wave64__zzprivate_blocks((const void*)amd_rocm6_wave64__zzprivate_wide,
                                                               (rows + AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS, 0u);
    void* args[] = { &out, &weights, &luts, &x, &rows, &cols, &d, &over };
    amd_rocm6_wave64__zzprivate_launch_room((const void*)amd_rocm6_wave64__zzprivate_wide, blocks, args, 0u);
    return true;
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_TURBOQUANT_CUH */
