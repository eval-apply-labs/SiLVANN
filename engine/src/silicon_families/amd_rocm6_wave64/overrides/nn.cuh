#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_NN_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_NN_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>

/* ══ ⭐⭐ OVERRIDES — THE DOORS THIS FAMILY RUNS FASTER THAN THE GENERIC BODY ══════════════════════════
 * ⚖ *"a for the family bodies"*, in *"a folder for primitives and one for overrides so it is clear what is
 * expected to run and what are performance upgrades"*. Each function here has its door's signature and
 * takes its place in nn's table (`amd_rocm6_wave64__override_doors`, called from `../entry.cuh` once the
 * table is built). ⛳ EVERY ONE FALLS BACK TO nn's GENERIC DOOR for what it does not cover — another width,
 * a row that is not aligned, an `x` wider than the lanes hold — so an override can only ever be faster,
 * never different in what it accepts, and deleting this file loses nothing but speed. */

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
#define AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE      64u
#define AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS      4u        /* rows a block, one a wave */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_HELD      4u        /* blocks of x a lane holds: up to 64 · 4 · 32 = 8192 columns */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS    __launch_bounds__(256)

static __device__ inline int32_t amd_rocm6_wave64__zzprivate_dot4(uint32_t a, uint32_t b, int32_t c) {
#if defined(__gfx906__) || defined(__gfx908__)
    return __builtin_amdgcn_sdot4((int)a, (int)b, c, false);
#else
    for (uint32_t i = 0u; i < 4u; ++i)
        c += (int32_t)(int8_t)(uint8_t)(a >> (8u * i)) * (int32_t)(int8_t)(uint8_t)(b >> (8u * i));
    return c;
#endif
}

/* The levels of four codes — 16 bits of a row, low nibble first — packed as four int8. */
static __device__ inline uint32_t amd_rocm6_wave64__zzprivate_levels4(const uint32_t* lev, uint32_t codes) {
    const uint32_t spread = (codes & 0xFu) | ((codes & 0xF0u) << 4) | ((codes & 0xF00u) << 8) | ((codes & 0xF000u) << 12);
    const uint32_t pick = spread & 0x07070707u, top = ((spread >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t low = __builtin_amdgcn_perm(lev[1], lev[0], pick), high = __builtin_amdgcn_perm(lev[3], lev[2], pick);
    return (low & ~top) | (high & top);
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
/* The eight positive 4-bit levels' low and high bytes, as levels8 reads them. */
static __device__ inline void amd_rocm6_wave64__zzprivate_tables(uint32_t* low, uint32_t* high) {
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
                    acc = amd_rocm6_wave64__zzprivate_dot4(amd_rocm6_wave64__zzprivate_levels4(lev, (uint32_t)c), q[k][4u * e + 0u], acc);
                    acc = amd_rocm6_wave64__zzprivate_dot4(amd_rocm6_wave64__zzprivate_levels4(lev, (uint32_t)(c >> 16)), q[k][4u * e + 1u], acc);
                    acc = amd_rocm6_wave64__zzprivate_dot4(amd_rocm6_wave64__zzprivate_levels4(lev, (uint32_t)(c >> 32)), q[k][4u * e + 2u], acc);
                    acc = amd_rocm6_wave64__zzprivate_dot4(amd_rocm6_wave64__zzprivate_levels4(lev, (uint32_t)(c >> 48)), q[k][4u * e + 3u], acc);
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
static __device__ inline float amd_rocm6_wave64__zzprivate_dot2(uint32_t a, uint32_t b, float c) {
    return nn__silicon__dot2(a, b, c);                         /* the family's — ▶ ../nn.cuh */
}

/* ⭐ `x` IN SHARED MEMORY, ONCE A BLOCK: a lane held its blocks of `x` in registers — 48 of a 5120-wide row's 124 —
 *   which left room for two waves a SIMD, too few to cover a row's loads. The block's four waves write `x` once,
 *   in levels8's order, wait for each other once, and each lane then reads eight columns' halves in one 16-byte
 *   read. The one barrier is before the first row, so nothing waits on it once the rows begin. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_X_SHARED  8192u     /* columns `x` may have: 16 KB of halves */
/* The room a block's kernel is given, sized to the call: `x`'s halves, then nn's 256 8-bit levels where a kernel
 * reads them. A row of 5120 is then 10.5 KB, where a room of the widest row allowed only three blocks a CU. */
static inline size_t amd_rocm6_wave64__zzprivate_room_bytes(uint64_t cols) { return (size_t)cols * 2u + 512u; }
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_exact_rows4(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                       const uint16_t* x, uint64_t rows, uint64_t cols,
                                                                       unsigned int* over) {
    extern __shared__ uint4 room[];
    uint4* xs = room;
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzprivate_tables(low, high);
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t blocks = cols / 32u, row_bytes = cols / 2u;
    for (uint64_t e = nn__silicon__lane(); e < cols / 8u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
        uint32_t v[4];
        amd_rocm6_wave64__zzprivate_x8(x + 8u * e, v);
        xs[e] = make_uint4(v[0], v[1], v[2], v[3]);
    }
    __syncthreads();
    /* past the row's end a lane's block reads the row's first codes and is weighed by zeros — a block of `x`
     * that is not there — so every load is unconditional and a row's loads are issued together */
    uint32_t at[HELD];
    bool real[HELD];
#pragma unroll
    for (uint32_t k = 0u; k < HELD; ++k) {
        const uint64_t b = lane + (uint64_t)k * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
        real[k] = b < blocks;
        at[k] = real[k] ? (uint32_t)b : 0u;
    }
    for (uint64_t r = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; r < rows;
         r += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        uint64_t cur[HELD][2];
#pragma unroll
        for (uint32_t k = 0u; k < HELD; ++k) {
            const uint64_t* c = (const uint64_t*)(weights + r * row_bytes + (uint64_t)at[k] * 16u);
            cur[k][0] = c[0]; cur[k][1] = c[1];
        }
        /* the row's scale is read with its codes, not after its sum, where it was a second wait a row */
        const uint32_t scale_bits = (uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8);
        float part = 0.0f;
#pragma unroll
        for (uint32_t k = 0u; k < HELD; ++k) {
            float sub = 0.0f;
#pragma unroll 1
            for (uint32_t q = 0u; q < 4u; ++q) {                         /* eight columns at a time, one word live */
                uint32_t pairs[4];
                const uint64_t two = q < 2u ? cur[k][0] : cur[k][1];
                amd_rocm6_wave64__zzprivate_levels8(low, high, (uint32_t)(two >> (32u * (q % 2u))), pairs);
                const uint4 xv = xs[4u * at[k] + q];
                sub = amd_rocm6_wave64__zzprivate_dot2(pairs[0], xv.x, sub);
                sub = amd_rocm6_wave64__zzprivate_dot2(pairs[1], xv.y, sub);
                sub = amd_rocm6_wave64__zzprivate_dot2(pairs[2], xv.z, sub);
                sub = amd_rocm6_wave64__zzprivate_dot2(pairs[3], xv.w, sub);
            }
            part += real[k] ? sub : 0.0f;
        }
        for (uint32_t off = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u; off > 0u; off >>= 1) part += __shfl_xor(part, (int)off);
        if (lane == 0u) out[r] = nn__kernels__zzabi_to_half(part * nn__silicon__half_to_float((uint16_t)scale_bits), over);
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
    extern __shared__ uint4 room[];
    uint4* xs = room;
    uint16_t* table = (uint16_t*)(room + cols / 8u);   /* nn's 8-bit levels, read here rather than gathered a code at a time */
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < 256u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        table[e] = nn__turboquant__zzabi_levels(8u)[e];
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t blocks = cols / 32u, row_bytes = cols;
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
    for (uint64_t r = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; r < rows;
         r += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        const uint8_t* row = weights + r * row_bytes;
        const uint32_t scale_bits = (uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8);
        float part = 0.0f;
#pragma unroll
        for (uint32_t k = 0u; k < HELD; ++k) {
            const uint64_t* codes = (const uint64_t*)(row + (uint64_t)at[k] * 32u);    /* 32 columns: 32 bytes */
            uint64_t c[4];
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) c[q] = codes[q];
            float sub = 0.0f;
#pragma unroll 1
            for (uint32_t q = 0u; q < 4u; ++q) {                         /* eight columns at a time */
                const uint64_t cw = q == 0u ? c[0] : q == 1u ? c[1] : q == 2u ? c[2] : c[3];
                const uint32_t l0 = table[cw & 0xFFu], l1 = table[(cw >> 8) & 0xFFu], l2 = table[(cw >> 16) & 0xFFu];
                const uint32_t l3 = table[(cw >> 24) & 0xFFu], l4 = table[(cw >> 32) & 0xFFu], l5 = table[(cw >> 40) & 0xFFu];
                const uint32_t l6 = table[(cw >> 48) & 0xFFu], l7 = table[(cw >> 56) & 0xFFu];
                const uint4 xv = xs[4u * at[k] + q];
                sub = amd_rocm6_wave64__zzprivate_dot2(l0 | (l2 << 16), xv.x, sub);
                sub = amd_rocm6_wave64__zzprivate_dot2(l4 | (l6 << 16), xv.y, sub);
                sub = amd_rocm6_wave64__zzprivate_dot2(l1 | (l3 << 16), xv.z, sub);
                sub = amd_rocm6_wave64__zzprivate_dot2(l5 | (l7 << 16), xv.w, sub);
            }
            part += real[k] ? sub : 0.0f;
        }
        for (uint32_t off = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u; off > 0u; off >>= 1) part += __shfl_xor(part, (int)off);
        if (lane == 0u) out[r] = nn__kernels__zzabi_to_half(part * nn__silicon__half_to_float((uint16_t)scale_bits), over);
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

/* ⭐ THE BLOCKS THE CARD HOLDS AT ONCE for `kernel` — its occupancy times the device's compute units. A row kernel
 *   loads its lanes' `x` once a wave before its first row, so a grid larger than the card holds pays that load again
 *   for every wave that follows, for a row or two each. `MEASURED` (test/src_gemv_27b_bench.cpp, MI50): the 27B's
 *   out-projection 153 us at 960 blocks, 72 us at 120. Kept per thread: two TP ranks are two threads, each on its
 *   own device. */
static inline uint32_t amd_rocm6_wave64__zzprivate_resident(const void* kernel, uint32_t threads, size_t room) {
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
static inline uint32_t amd_rocm6_wave64__zzprivate_blocks(const void* kernel, uint64_t rows, size_t room) {
    const uint64_t want = (rows + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t most = amd_rocm6_wave64__zzprivate_resident(kernel, AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, room);
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

/* ⭐ THE EXACT GEMV at 4 and 8 bits; anything else is the generic door. */
static void amd_rocm6_wave64__override_turboquant_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room,
                                                       const uint8_t* luts, uint64_t l_room, const uint16_t* x,
                                                       uint64_t d, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_fits(weights, w_room, l_room, d, rows, cols)) {
        if (!amd_rocm6_wave64__zzprivate_wide_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over))
            nn__turboquant__zzabi_launch_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over);
        return;
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

/* ── THE ROTATION — a wave a 512-block, the butterfly through shuffles ────────────────────────────────
 * nn's definition (`nn/gpu/kernels/kernels.cuh`): `out = H·(S ⊙ in) / √b` over blocks of 512, the sign vector
 * starting afresh at each block; the generic body sums each output directly, one lane an output. Here a WAVE
 * takes a block and each lane holds 8 consecutive elements: the stages that pair elements under 8 apart
 * are done inside a lane's registers, and the six that pair them 8..256 apart swap across lanes with a
 * shuffle — no barrier, since a wave's shuffle waits on nothing outside the wave. O(b log b) work against the
 * direct sum's O(b²), in fp32 and rounded to a half once, as the generic body is. Covers `n` a multiple of 512
 * with `out` apart from `in`; anything else is the generic door, whose in-place butterfly defines that case. */
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
            const float other = __shfl_xor(v[e], (int)m);
            v[e] = low ? v[e] + other : other - v[e];
        }
    }
    const float inv = 1.0f / nn__silicon__sqrtf((float)AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_BLOCK);
#pragma unroll
    for (uint32_t e = 0u; e < AMD_ROCM6_WAVE64__ZZPRIVATE_ROT_EACH; ++e)
        out[at + c0 + e] = nn__kernels__zzabi_to_half(v[e] * inv, over);
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
    float v[E];
#pragma unroll
    for (uint32_t e = 0u; e < E; ++e) {
        v[e] = nn__silicon__half_to_float(in[at + c0 + e]);
        if (inverse == 0u) v[e] *= nn__silicon__half_to_float(sign[c0 + e]);
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
            const float other = __shfl_xor(v[e], (int)m);
            v[e] = low ? v[e] + other : other - v[e];
        }
    }
    const float norm = 1.0f / nn__silicon__sqrtf((float)B);
#pragma unroll
    for (uint32_t e = 0u; e < E; ++e) {
        float r = v[e] * norm;
        if (inverse != 0u) r *= nn__silicon__half_to_float(sign[c0 + e]);
        out[at + c0 + e] = nn__kernels__zzabi_to_half(r, over);
    }
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

/* ⭐ ATTENTION'S TWO RESIDUAL PASSES, SPREAD OVER THE BLOCK — nn's `attention_weights` gives a lane a position and sums
 *   its whole dot product alone, reading a key row with lanes two kilobytes apart; `attention_residual_mix` gives a lane
 *   an element of the head and walks every position in turn. `MEASURED` on the 27B at 1000 positions: 0.41 and 0.21 ms
 *   a layer, 20 ms of a 76 ms token, for 4 MB of keys and values a layer. Here, still a block a query head:
 *   · the scores: sixteen lanes a position, each holding `D / 16` of the head's query, so a wave reads four key rows
 *     whole a step, four steps in flight; the sixteen parts summed by shuffles. The softmax's largest score and sum
 *     are the family's fold, as nn's are, and each score is exponentiated by the lane that wrote it.
 *   · the mix: the block's four waves take every fourth position, each lane `D / 64` elements of the head, so a wave
 *     reads a value row whole; the four waves' sums added in shared memory once.
 *   The products and their fp32 sums are nn's; only the order they are added in differs. Heads of 128 or 256.
 * ⭐ AND THE ONE-CALL PAIR IS THE SAME TWO PASSES: `attention_scores` is the weights divided by their sum, by the lane
 *   that wrote each (`NORM`), and `attention_mix` the mix written as halves rather than into a residual (`HALVES`) —
 *   the 27B's fp16 cache ran nn's bodies, a block a head with a lane a position, and decoded 30 ms a token slower than
 *   the tiered cache over the same positions. */
template <uint32_t D, bool NORM>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_weights(float* p, float* res, const uint16_t* q, const uint16_t* k,
                                                                             uint64_t q_heads, uint64_t kv_heads, uint64_t length) {
    const uint32_t PER = D / 16u;                                         /* a lane's halves of a head: 8 or 16 */
    const uint32_t t = (uint32_t)nn__silicon__lane(), sub = t % 16u, slot = t / 16u;
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    const float scale = 1.0f / nn__silicon__sqrtf((float)D);
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const uint16_t* kh = k + (h / group) * D + sub * PER;
        float* ph = p + h * length;
        uint32_t qv[PER / 2u];
        const uint32_t* qw = (const uint32_t*)(q + h * D + sub * PER);
#pragma unroll
        for (uint32_t i = 0u; i < PER / 2u; ++i) qv[i] = qw[i];
        float top = -NN__KERNELS__INFINITY;
        for (uint64_t base = slot; base < length; base += 64u) {           /* four positions a lane group in flight */
            uint32_t kv[4][PER / 2u];
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                const uint64_t pos = base + 16u * u < length ? base + 16u * u : base;
                const uint32_t* kw = (const uint32_t*)(kh + pos * row);
#pragma unroll
                for (uint32_t i = 0u; i < PER / 2u; ++i) kv[u][i] = kw[i];
            }
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                float dot = 0.0f;
#pragma unroll
                for (uint32_t i = 0u; i < PER / 2u; ++i) dot = nn__silicon__dot2(qv[i], kv[u][i], dot);
                for (uint32_t off = 8u; off > 0u; off >>= 1) dot += __shfl_xor(dot, (int)off);
                const uint64_t pos = base + 16u * u;
                if (pos < length) {
                    const float score = dot * scale;
                    if (sub == 0u) ph[pos] = score;
                    top = score > top ? score : top;
                }
            }
        }
        top = nn__silicon__lanes_max(top);
        float part = 0.0f;                     /* each score exponentiated by the lane that wrote it */
        if (sub == 0u)
            for (uint64_t pos = slot; pos < length; pos += 16u) {
                const float e = nn__silicon__expf(ph[pos] - top);
                ph[pos] = e;
                part += e;
            }
        const float sum = nn__silicon__lanes_sum(part);
        if (NORM) {
            const float inv = 1.0f / sum;
            if (sub == 0u)
                for (uint64_t pos = slot; pos < length; pos += 16u) ph[pos] *= inv;
        } else if (t == 0u) {
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(D)] = top;
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 1ull] = sum;
        }
    }
}
template <uint32_t D, bool HALVES>
static __device__ inline void amd_rocm6_wave64__zzprivate_attention_mix(float* res, uint16_t* o, const float* p, const uint16_t* v,
                                                                         uint64_t q_heads, uint64_t kv_heads, uint64_t length,
                                                                         unsigned int* over) {
    const uint32_t PER = D / 64u;                                         /* a lane's elements of a head: 2 or 4 */
    __shared__ float waves[AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS][D];
    const uint32_t t = (uint32_t)nn__silicon__lane();
    const uint32_t lane = t % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, wave = t / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const uint64_t group = q_heads / kv_heads, row = kv_heads * D;
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        const float* ph = p + h * length;
        const uint16_t* vh = v + (h / group) * D + lane * PER;
        float acc[PER];
#pragma unroll
        for (uint32_t e = 0u; e < PER; ++e) acc[e] = 0.0f;
        for (uint64_t base = wave; base < length; base += 8u * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {   /* eight positions in flight */
            uint32_t vv[8][PER / 2u];
            float wv[8];
#pragma unroll
            for (uint32_t u = 0u; u < 8u; ++u) {
                const uint64_t at = base + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * u;
                const bool real = at < length;
                const uint64_t pos = real ? at : base;
                const uint32_t* vw = (const uint32_t*)(vh + pos * row);
#pragma unroll
                for (uint32_t i = 0u; i < PER / 2u; ++i) vv[u][i] = vw[i];
                wv[u] = real ? ph[pos] : 0.0f;
            }
#pragma unroll
            for (uint32_t u = 0u; u < 8u; ++u)
#pragma unroll
                for (uint32_t i = 0u; i < PER / 2u; ++i) {
                    acc[2u * i]      += wv[u] * nn__silicon__half_to_float((uint16_t)(vv[u][i] & 0xFFFFu));
                    acc[2u * i + 1u] += wv[u] * nn__silicon__half_to_float((uint16_t)(vv[u][i] >> 16));
                }
        }
#pragma unroll
        for (uint32_t e = 0u; e < PER; ++e) waves[wave][lane * PER + e] = acc[e];
        __syncthreads();
        for (uint32_t e = t; e < D; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
            float sum = 0.0f;
#pragma unroll
            for (uint32_t w = 0u; w < AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS; ++w) sum += waves[w][e];
            if (HALVES) o[h * D + e] = nn__kernels__zzabi_to_half(sum, over);
            else res[h * NN__ATTENTION__RESIDUAL_FLOATS(D) + 2ull + e] = sum;
        }
        __syncthreads();
    }
}
#define AMD_ROCM6_WAVE64__ZZPRIVATE_ATTENTION_KERNELS(D)                                                                          \
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_weights_##D(                          \
    float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {               \
    amd_rocm6_wave64__zzprivate_attention_weights<D##u, false>(p, res, q, k, q_heads, kv_heads, length);                               \
}                                                                                                                                      \
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_scores_##D(                           \
    float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {                           \
    amd_rocm6_wave64__zzprivate_attention_weights<D##u, true>(p, 0, q, k, q_heads, kv_heads, length);                                  \
}                                                                                                                                      \
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_##D(                              \
    float* res, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads, uint64_t length) {                            \
    amd_rocm6_wave64__zzprivate_attention_mix<D##u, false>(res, 0, p, v, q_heads, kv_heads, length, 0);                                \
}                                                                                                                                      \
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_attention_mix_halves_##D(                       \
    uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads, uint64_t length, unsigned int* over) {       \
    amd_rocm6_wave64__zzprivate_attention_mix<D##u, true>(0, o, p, v, q_heads, kv_heads, length, over);                                \
}
AMD_ROCM6_WAVE64__ZZPRIVATE_ATTENTION_KERNELS(128)
AMD_ROCM6_WAVE64__ZZPRIVATE_ATTENTION_KERNELS(256)
/* Heads of 128 or 256, a whole group of query heads a key head, `q` and `k` / `v` on 4 bytes; anything else is nn's. */
static inline bool amd_rocm6_wave64__zzprivate_attention_fit(const void* a, const void* b, uint64_t q_heads, uint64_t kv_heads,
                                                             uint64_t head_dim) {
    return (head_dim == 128u || head_dim == 256u) && kv_heads != 0u && q_heads % kv_heads == 0u && q_heads != 0u
        && ((uint64_t)(uintptr_t)a) % 4u == 0u && ((uint64_t)(uintptr_t)b) % 4u == 0u;
}
static void amd_rocm6_wave64__override_attention_weights(float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                         uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit(q, k, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_weights(p, res, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_weights_128
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_weights_256;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &res, &q, &k, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof p, sizeof res, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_weights", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 7u);
}
static void amd_rocm6_wave64__override_attention_residual_mix(float* res, const float* p, const uint16_t* v, uint64_t q_heads,
                                                              uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit(v, v, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_residual_mix(res, p, v, q_heads, kv_heads, head_dim, length);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_128
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_256;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &res, &p, &v, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_mix", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 6u);
}

static void amd_rocm6_wave64__override_attention_scores(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                        uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit(q, k, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_scores(p, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_scores_128
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_scores_256;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &q, &k, &q_heads, &kv_heads, &length };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof length };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_scores", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 6u);
}
static void amd_rocm6_wave64__override_attention_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads,
                                                     uint64_t kv_heads, uint64_t head_dim, uint64_t length, unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_attention_fit(v, v, q_heads, kv_heads, head_dim)) {
        nn__attention__zzabi_launch_mix(o, p, v, q_heads, kv_heads, head_dim, length, over);
        return;
    }
    const void* kernel = head_dim == 128u ? (const void*)amd_rocm6_wave64__zzprivate_attention_mix_halves_128
                                          : (const void*)amd_rocm6_wave64__zzprivate_attention_mix_halves_256;
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &p, &v, &q_heads, &kv_heads, &length, &over };
    const size_t sizes[] = { sizeof o, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof length, sizeof over };
    nn__silicon__launch(kernel, "amd_rocm6_wave64__zzprivate_attention_mix", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 7u);
}

/* ⭐ THE DELTANET OVER A PROMPT'S ROWS AT A HEAD OF 128 — nn's `deltanet_steps`, the one-position step a row at a time,
 *   a block a head and a lane a column of its state, exactly as nn's: the same operations in the same order, so a
 *   prompt read as rows and the same prompt a position at a time (nn's `deltanet_step`) still agree to the bit. What
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
            const uint32_t lo = amd_rocm6_wave64__zzprivate_levels4(low, four);
            const uint32_t hi = amd_rocm6_wave64__zzprivate_levels4(high, four);
            part = amd_rocm6_wave64__zzprivate_dot2(__builtin_amdgcn_perm(hi, lo, 0x05010400u), xw[2u * e + 0u], part);
            part = amd_rocm6_wave64__zzprivate_dot2(__builtin_amdgcn_perm(hi, lo, 0x07030602u), xw[2u * e + 1u], part);
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
                part = amd_rocm6_wave64__zzprivate_dot2(pair, xw[4u * e + p], part);
            }
        }
    }
    return part;
}


static __device__ inline float amd_rocm6_wave64__zzprivate_row_scale(const nn__turboquant__groups* g, uint64_t i, uint64_t r) {
    const uint8_t* lut = (const uint8_t*)(uintptr_t)g->luts[i];
    return nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2ull] | ((uint32_t)lut[r * 2ull + 1ull] << 8)));
}

/* ⭐ THE GROUPS' ROWS, AS THE EXACT GEMV'S: `x` in shared memory in levels8's order, a row's loads issued together,
 *   its scale read with its codes, one word decoded at a time — and a lane's blocks a template count, so a row
 *   needs only the registers its width uses. An 8-bit group's codes are looked up in nn's table and paired in the
 *   same order. */
template <uint32_t HELD>
static __device__ inline void amd_rocm6_wave64__zzprivate_groups_rows(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                       uint64_t cols, unsigned int* over) {
    extern __shared__ uint4 room[];
    uint4* xs = room;
    uint16_t* table8 = (uint16_t*)(room + cols / 8u);  /* nn's 8-bit levels, read here rather than gathered a code at a time */
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzprivate_tables(low, high);
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < 256u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        table8[e] = nn__turboquant__zzabi_levels(8u)[e];
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
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
    uint64_t total = 0ull;
    for (uint64_t i = 0ull; i < g.count; ++i) total += g.rows[i];
    for (uint64_t t = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; t < total;
         t += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        uint64_t i = 0ull, r = t;
        while (r >= g.rows[i]) { r -= g.rows[i]; ++i; }
        const uint32_t d = (uint32_t)g.d[i];
        const uint8_t* row = (const uint8_t*)(uintptr_t)g.codes[i] + r * (cols * d / 8u);
        const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
        const uint32_t scale_bits = (uint32_t)lut[r * 2ull] | ((uint32_t)lut[r * 2ull + 1ull] << 8);
        float part = 0.0f;
        if (d == 4u) {
            uint64_t cur[HELD][2];
#pragma unroll
            for (uint32_t k = 0u; k < HELD; ++k) {
                const uint64_t* c = (const uint64_t*)(row + (uint64_t)at[k] * 16u);
                cur[k][0] = c[0]; cur[k][1] = c[1];
            }
#pragma unroll
            for (uint32_t k = 0u; k < HELD; ++k) {
                float sub = 0.0f;
#pragma unroll 1
                for (uint32_t q = 0u; q < 4u; ++q) {
                    uint32_t pairs[4];
                    const uint64_t two = q < 2u ? cur[k][0] : cur[k][1];
                    amd_rocm6_wave64__zzprivate_levels8(low, high, (uint32_t)(two >> (32u * (q % 2u))), pairs);
                    const uint4 xv = xs[4u * at[k] + q];
                    sub = amd_rocm6_wave64__zzprivate_dot2(pairs[0], xv.x, sub);
                    sub = amd_rocm6_wave64__zzprivate_dot2(pairs[1], xv.y, sub);
                    sub = amd_rocm6_wave64__zzprivate_dot2(pairs[2], xv.z, sub);
                    sub = amd_rocm6_wave64__zzprivate_dot2(pairs[3], xv.w, sub);
                }
                part += real[k] ? sub : 0.0f;
            }
        } else {
#pragma unroll
            for (uint32_t k = 0u; k < HELD; ++k) {
                float sub = 0.0f;
#pragma unroll 1
                for (uint32_t q = 0u; q < 4u; ++q) {                     /* a word of eight codes, a byte each */
                    const uint64_t c = *(const uint64_t*)(row + (uint64_t)at[k] * 32u + 8u * q);
                    const uint32_t l0 = table8[c & 0xFFu], l1 = table8[(c >> 8) & 0xFFu], l2 = table8[(c >> 16) & 0xFFu];
                    const uint32_t l3 = table8[(c >> 24) & 0xFFu], l4 = table8[(c >> 32) & 0xFFu], l5 = table8[(c >> 40) & 0xFFu];
                    const uint32_t l6 = table8[(c >> 48) & 0xFFu], l7 = table8[(c >> 56) & 0xFFu];
                    const uint4 xv = xs[4u * at[k] + q];
                    sub = amd_rocm6_wave64__zzprivate_dot2(l0 | (l2 << 16), xv.x, sub);
                    sub = amd_rocm6_wave64__zzprivate_dot2(l4 | (l6 << 16), xv.y, sub);
                    sub = amd_rocm6_wave64__zzprivate_dot2(l1 | (l3 << 16), xv.z, sub);
                    sub = amd_rocm6_wave64__zzprivate_dot2(l5 | (l7 << 16), xv.w, sub);
                }
                part += real[k] ? sub : 0.0f;
            }
        }
        for (uint32_t off = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u; off > 0u; off >>= 1) part += __shfl_xor(part, (int)off);
        if (lane == 0u) out[g.out_at[i] + r] = nn__kernels__zzabi_to_half(part * nn__silicon__half_to_float((uint16_t)scale_bits), over);
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
                for (uint32_t p = 0u; p < 4u; ++p) s = amd_rocm6_wave64__zzprivate_dot2(pairs[p], xp[4u * q + p], s);
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
                s = amd_rocm6_wave64__zzprivate_dot2(l0 | (l2 << 16), xp[4u * q + 0u], s);
                s = amd_rocm6_wave64__zzprivate_dot2(l4 | (l6 << 16), xp[4u * q + 1u], s);
                s = amd_rocm6_wave64__zzprivate_dot2(l1 | (l3 << 16), xp[4u * q + 2u], s);
                s = amd_rocm6_wave64__zzprivate_dot2(l5 | (l7 << 16), xp[4u * q + 3u], s);
            }
            sub[a] = s;
        }
    }
}
static __device__ inline void amd_rocm6_wave64__zzprivate_groups_sum_rows(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                           const uint16_t* w, const uint16_t* residual,
                                                                           uint64_t rows, uint64_t cols, unsigned int* over) {
    const uint32_t RW = AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS;
    extern __shared__ uint4 room[];
    uint16_t* table8 = (uint16_t*)room;   /* as groups' — the one barrier is before the first row */
    for (uint32_t e = (uint32_t)nn__silicon__lane(); e < 256u; e += AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE)
        table8[e] = nn__turboquant__zzabi_levels(8u)[e];
    __syncthreads();
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzprivate_tables(low, high);
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t blocks = cols / 32u, spread = g.count * blocks;
    for (uint64_t r0 = (nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave) * RW; r0 < rows;
         r0 += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * RW) {
        float part[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS];
        uint64_t rr[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS];
#pragma unroll
        for (uint32_t a = 0u; a < RW; ++a) { part[a] = 0.0f; rr[a] = r0 + a < rows ? r0 + a : r0; }
#pragma unroll 1
        for (uint64_t j = lane; j < spread; j += AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {    /* (group, block) pairs */
            const uint64_t i = j / blocks, b = j - i * blocks;
            const uint32_t d = (uint32_t)g.d[i];
            const uint64_t row_bytes = cols * d / 8u;
            const uint8_t* codes = (const uint8_t*)(uintptr_t)g.codes[i];
            const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
            const float weight = nn__silicon__half_to_float(w[g.out_at[i]]);
            const uint4* xw = (const uint4*)(x + i * cols + b * 32u);
            uint32_t xp[16];
#pragma unroll
            for (uint32_t q = 0u; q < 4u; ++q) amd_rocm6_wave64__zzprivate_x8_order(xw[q], &xp[4u * q]);
            float sub[AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS];
            amd_rocm6_wave64__zzprivate_block_rows(d, codes, row_bytes, rr, b, xp, low, high, table8, sub);
#pragma unroll
            for (uint32_t a = 0u; a < RW; ++a)
                part[a] += sub[a] * (nn__silicon__half_to_float((uint16_t)((uint32_t)lut[rr[a] * 2ull] | ((uint32_t)lut[rr[a] * 2ull + 1ull] << 8))) * weight);
        }
#pragma unroll
        for (uint32_t a = 0u; a < RW; ++a) {
            float v = part[a];
            for (uint32_t off = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / 2u; off > 0u; off >>= 1) v += __shfl_xor(v, (int)off);
            if (lane == 0u && r0 + a < rows)
                out[r0 + a] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(residual[r0 + a]) + v, over);
        }
    }
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
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_groups_sum(uint16_t* out, nn__turboquant__groups g,
                                                                                                const uint16_t* x, const uint16_t* w,
                                                                                                const uint16_t* residual, uint64_t rows,
                                                                                                uint64_t cols, unsigned int* over) {
    amd_rocm6_wave64__zzprivate_groups_sum_rows(out, g, x, w, residual, rows, cols, over);
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

static void amd_rocm6_wave64__override_gemv_groups(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                   unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_groups_fit(&g, cols)) { nn__turboquant__zzabi_launch_gemv_groups(out, g, x, cols, over); return; }
    uint64_t total = 0u;
    for (uint64_t i = 0u; i < g.count; ++i) total += g.rows[i];
    const uint64_t held = (cols / 32u + AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE;
    const void* kernel = held == 1u ? (const void*)amd_rocm6_wave64__zzprivate_groups_h1
                       : held == 2u ? (const void*)amd_rocm6_wave64__zzprivate_groups_h2
                       : held == 3u ? (const void*)amd_rocm6_wave64__zzprivate_groups_h3
                       :              (const void*)amd_rocm6_wave64__zzprivate_groups_h4;
    const size_t room = amd_rocm6_wave64__zzprivate_room_bytes(cols);
    const uint32_t blocks = amd_rocm6_wave64__zzprivate_blocks(kernel, total, room);
    void* args[] = { &out, &g, &x, &cols, &over };
    amd_rocm6_wave64__zzprivate_launch_room(kernel, blocks, args, room);
}

static void amd_rocm6_wave64__override_gemv_groups_sum(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                       const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_groups_codes_fit(&g, cols) || ((uint64_t)(uintptr_t)x) % 16u != 0u) {
        nn__turboquant__zzabi_launch_gemv_groups_sum(out, g, x, w, residual, rows, cols, over);
        return;
    }
    const size_t room = 512u;
    const uint32_t blocks = amd_rocm6_wave64__zzprivate_blocks((const void*)amd_rocm6_wave64__zzprivate_groups_sum,
                                                               (rows + AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_SUM_ROWS, room);
    void* args[] = { &out, &g, &x, &w, &residual, &rows, &cols, &over };
    amd_rocm6_wave64__zzprivate_launch_room((const void*)amd_rocm6_wave64__zzprivate_groups_sum, blocks, args, room);
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
    amd_rocm6_wave64__zzprivate_tables(low, high);
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

/* ── ⭐⭐ THE EXPERT-MAJOR GEMM — a wave a row of the matrix, its codes decoded once for every pair ─────────────
 * nn's definitions (`kernels.cuh`, the expert-major GEMM): a matrix applied to a list of pairs (a row of `x`, a
 * row of `out`), plain, weighted into an fp32 sum, or several matrices in one launch. ⚖ *"wave level tiles"*:
 *   · one WAVE owns one row of the matrix for EVERY pair, so no two waves write one element — the fp32 sum
 *     needs no atomics, and nothing is shared between waves, so there is no barrier: a wave's sums are shuffles
 *   · the row's codes are decoded ONCE into registers — two fp16 levels a register, as the gemv's — and every
 *     pair's `x` is then multiplied into them by `v_dot2_f32_f16`; the decode is what a batch saves
 *   · a lane holds blocks of 32 columns: at the hidden width one a lane, at 4096 two. A row narrower than the
 *     wave — the MoE's downs, 512 columns, 16 blocks — splits the wave into groups of lanes, each group a
 *     different pair, so no lane idles
 *   · eight pairs a round, each lane's eight parts then summed across its group by shuffles
 * Covers 4 and 8 bits over whole blocks of 32 columns, a row of 1..32 blocks or a multiple of 64 up to 128,
 * codes on 8 bytes and `x` on 4; anything else is the generic door. The products are the generic body's; the
 * order they are summed in is not. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_XR    8u        /* pairs a round */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_HB    2u        /* blocks of a row a lane holds */

/* A block of 32 codes as sixteen registers of two fp16 levels, in column order. */
static __device__ inline void amd_rocm6_wave64__zzprivate_decode_block(uint32_t d, const uint8_t* row, uint64_t b, const uint32_t* low,
                                                                        const uint32_t* high, const uint16_t* table8, uint32_t* lv) {
    if (d == 4u) {
        const uint64_t* codes = (const uint64_t*)(row + b * 16u);
#pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            const uint32_t four = (uint32_t)(codes[e / 4u] >> (16u * (e % 4u)));
            const uint32_t lo = amd_rocm6_wave64__zzprivate_levels4(low, four);
            const uint32_t hi = amd_rocm6_wave64__zzprivate_levels4(high, four);
            lv[2u * e]      = __builtin_amdgcn_perm(hi, lo, 0x05010400u);
            lv[2u * e + 1u] = __builtin_amdgcn_perm(hi, lo, 0x07030602u);
        }
    } else {
        const uint64_t* codes = (const uint64_t*)(row + b * 32u);
#pragma unroll
        for (uint32_t e = 0u; e < 4u; ++e) {
            const uint64_t c = codes[e];
#pragma unroll
            for (uint32_t p = 0u; p < 4u; ++p)
                lv[4u * e + p] = (uint32_t)table8[(c >> (16u * p)) & 0xFFu] | ((uint32_t)table8[(c >> (16u * p + 8u)) & 0xFFu] << 16);
        }
    }
}

/* How the wave is cut for a row of `blocks` blocks: `span` lanes a pair (a power of two up to 64), each holding
 * `held` blocks; false for a width the kernel does not take. */
static __device__ __host__ inline bool amd_rocm6_wave64__zzprivate_span(uint64_t blocks, uint32_t* span, uint32_t* held) {
    if (blocks >= AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE) {
        if (blocks % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE != 0u || blocks > AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE * AMD_ROCM6_WAVE64__ZZPRIVATE_HB) return false;
        *span = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE; *held = (uint32_t)(blocks / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
        return true;
    }
    if (blocks == 0u || (blocks & (blocks - 1u)) != 0u) return false;
    *span = (uint32_t)blocks; *held = 1u;
    return true;
}

/* One row `r` of one matrix against its `count` pairs, by one wave: into `out` (a half) or weighted into `acc`. */
static __device__ inline void amd_rocm6_wave64__zzprivate_row_pairs(uint32_t d, const uint8_t* row, float scale, const uint16_t* x,
                                                                     const uint32_t* rows, const uint16_t* w, uint64_t count,
                                                                     uint64_t cols, uint64_t out_rows, uint64_t r, uint16_t* out,
                                                                     float* acc, const uint32_t* low, const uint32_t* high,
                                                                     const uint16_t* table8, unsigned int* over) {
    uint32_t span = 0u, held = 0u;
    (void)amd_rocm6_wave64__zzprivate_span(cols / 32u, &span, &held);
    const uint32_t lane = (uint32_t)(nn__silicon__lane() % AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint32_t mine = lane % span, group = lane / span, groups = AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE / span;
    uint32_t lv[AMD_ROCM6_WAVE64__ZZPRIVATE_HB][16];
#pragma unroll
    for (uint32_t h = 0u; h < AMD_ROCM6_WAVE64__ZZPRIVATE_HB; ++h)
        if (h < held) amd_rocm6_wave64__zzprivate_decode_block(d, row, mine + (uint64_t)h * span, low, high, table8, lv[h]);
    for (uint64_t j0 = (uint64_t)group * AMD_ROCM6_WAVE64__ZZPRIVATE_XR; j0 < count; j0 += (uint64_t)groups * AMD_ROCM6_WAVE64__ZZPRIVATE_XR) {
        float part[AMD_ROCM6_WAVE64__ZZPRIVATE_XR];
#pragma unroll
        for (uint32_t q = 0u; q < AMD_ROCM6_WAVE64__ZZPRIVATE_XR; ++q) {
            part[q] = 0.0f;
            if (j0 + q < count) {
                const uint16_t* xr = x + (uint64_t)rows[2u * (j0 + q)] * cols;
#pragma unroll
                for (uint32_t h = 0u; h < AMD_ROCM6_WAVE64__ZZPRIVATE_HB; ++h)
                    if (h < held) {
                        const uint32_t* xw = (const uint32_t*)(xr + (mine + (uint64_t)h * span) * 32u);
#pragma unroll
                        for (uint32_t e = 0u; e < 16u; ++e) part[q] = amd_rocm6_wave64__zzprivate_dot2(lv[h][e], xw[e], part[q]);
                    }
            }
        }
#pragma unroll
        for (uint32_t q = 0u; q < AMD_ROCM6_WAVE64__ZZPRIVATE_XR; ++q)
            for (uint32_t off = span / 2u; off > 0u; off >>= 1) part[q] += __shfl_xor(part[q], (int)off);
        if (mine == 0u)
#pragma unroll
            for (uint32_t q = 0u; q < AMD_ROCM6_WAVE64__ZZPRIVATE_XR; ++q) {
                if (j0 + q >= count) continue;
                const uint64_t at = (uint64_t)rows[2u * (j0 + q) + 1u] * out_rows + r;
                if (out) out[at] = nn__kernels__zzabi_to_half(part[q] * scale, over);
                else acc[at] += part[q] * scale * nn__silicon__half_to_float(w[rows[2u * (j0 + q)]]);
            }
    }
}

typedef struct amd_rocm6_wave64__rows_call {
    uint16_t* out; float* acc; const uint8_t* weights; const uint8_t* luts; const uint16_t* x; const uint32_t* rows;
    const uint16_t* w; uint64_t count, d, out_rows, cols; unsigned int* over;
} amd_rocm6_wave64__rows_call;

static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_expert_rows(amd_rocm6_wave64__rows_call c) {
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzprivate_tables(low, high);
    const uint16_t* table8 = nn__turboquant__zzabi_levels(8u);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    const uint64_t row_bytes = c.cols * c.d / 8u;
    for (uint64_t r = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; r < c.out_rows;
         r += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)c.luts[r * 2ull] | ((uint32_t)c.luts[r * 2ull + 1ull] << 8)));
        amd_rocm6_wave64__zzprivate_row_pairs((uint32_t)c.d, c.weights + r * row_bytes, scale, c.x, c.rows, c.w, c.count, c.cols,
                                              c.out_rows, r, c.out, c.acc, low, high, table8, c.over);
    }
}

static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_expert_groups(uint16_t* out, nn__expert__groups g,
                                                                                                   const uint16_t* x, const uint32_t* rows,
                                                                                                   uint64_t cols, unsigned int* over) {
    uint32_t low[2], high[2];
    amd_rocm6_wave64__zzprivate_tables(low, high);
    const uint16_t* table8 = nn__turboquant__zzabi_levels(8u);
    const uint32_t wave = (uint32_t)(nn__silicon__lane() / AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE);
    uint64_t total = 0ull;
    for (uint64_t i = 0ull; i < g.count; ++i) total += g.out_rows[i];
    for (uint64_t t = nn__silicon__block() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS + wave; t < total;
         t += nn__silicon__blocks() * AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS) {
        uint64_t i = 0ull, r = t;
        while (r >= g.out_rows[i]) { r -= g.out_rows[i]; ++i; }
        const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
        const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2ull] | ((uint32_t)lut[r * 2ull + 1ull] << 8)));
        amd_rocm6_wave64__zzprivate_row_pairs((uint32_t)g.d[i], (const uint8_t*)(uintptr_t)g.codes[i] + r * (cols * g.d[i] / 8u), scale,
                                              x, rows + 2ull * g.pairs_at[i], 0, g.pairs[i], cols, g.out_rows[i], r, out + g.out_at[i], 0,
                                              low, high, table8, over);
    }
}

/* ══ ⭐⭐ THE EXPERT-MAJOR GEMM IN TILES — shared memory, so each weight and each row of `x` is read once a tile ═════
 * The kernel above gives a weight row to a wave and reads every pair's `x` again for every row. This one gives a
 * block a TILE — `TR` matrix rows by `TP` pairs — and walks the columns 128 at a time: the tile's slice of `x`
 * (halves) and of the matrix (its codes decoded to halves, once) are staged in shared memory, and each lane sums
 * `RT` rows by `PT` pairs from there, two products at a time (`nn__silicon__dot2`). Two shapes: 64 by 64 for a
 * matrix many rows read (a dense projection over a chunk), 64 by 16 for the few rows each routed expert gets.
 * A launch is up to twelve matrices, each its own tiles (`first`), as `expert_groups` hands them over.
 * ⛳ THE BARRIERS are the tile's own: the block waits once its slice is staged, and once more before the next slice
 *   overwrites it — the lanes of one launch waiting for each other, as the family's fold does. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_KT 64u
typedef struct amd_rocm6_wave64__tile_job {
    uint64_t codes[NN__EXPERT__GROUPS_MAX], luts[NN__EXPERT__GROUPS_MAX], out_rows[NN__EXPERT__GROUPS_MAX];
    uint64_t d[NN__EXPERT__GROUPS_MAX], out_at[NN__EXPERT__GROUPS_MAX], pairs_at[NN__EXPERT__GROUPS_MAX];
    uint64_t pairs[NN__EXPERT__GROUPS_MAX], first[NN__EXPERT__GROUPS_MAX + 1];
    uint64_t count, cols;
    uint16_t* out; float* acc; const uint16_t* x; const uint32_t* rows; const uint16_t* w; unsigned int* over;
} amd_rocm6_wave64__tile_job;

/* One tile body, the shape its arguments — inlined into each kernel below, so the shape is a constant there and
 * the loops and the sums' registers come out as if written for it. */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_RT_MAX 4u
#define AMD_ROCM6_WAVE64__ZZPRIVATE_PT_MAX 4u
static __device__ __forceinline__ void amd_rocm6_wave64__zzprivate_tile(const amd_rocm6_wave64__tile_job* jp, const uint32_t TR,
                                                                          const uint32_t TP, const uint32_t RT, const uint32_t PT) {
    const amd_rocm6_wave64__tile_job& j = *jp;
    __shared__ uint32_t ws[64][AMD_ROCM6_WAVE64__ZZPRIVATE_KT / 2u + 1u];
    __shared__ uint32_t xs[64][AMD_ROCM6_WAVE64__ZZPRIVATE_KT / 2u + 1u];
    const uint32_t HK = AMD_ROCM6_WAVE64__ZZPRIVATE_KT / 2u;
    const uint32_t t = (uint32_t)nn__silicon__lane(), n = (uint32_t)nn__silicon__lanes();
    /* nn's levels at 4 and 8 bits, read from here as a slice is staged rather than gathered a code at a time */
    __shared__ uint16_t lev4[16], lev8[256];
    for (uint32_t e = t; e < 256u; e += n) { lev8[e] = nn__turboquant__zzabi_levels(8u)[e]; if (e < 16u) lev4[e] = nn__turboquant__zzabi_levels(4u)[e]; }
    __syncthreads();
    const uint32_t tr = (t / (TP / PT)) * RT, tp = (t % (TP / PT)) * PT;
    for (uint64_t b = nn__silicon__block(); b < j.first[j.count]; b += nn__silicon__blocks()) {
        uint64_t i = 0u;
        while (b >= j.first[i + 1u]) ++i;
        const uint64_t tiles_p = (j.pairs[i] + TP - 1u) / TP, local = b - j.first[i];
        const uint64_t r0 = (local / tiles_p) * TR, p0 = (local % tiles_p) * TP;
        const uint64_t d = j.d[i], per = 16u / d, mask = (1u << d) - 1u, row_bytes = j.cols * d / 8u;
        const uint8_t* codes = (const uint8_t*)(uintptr_t)j.codes[i];
        const uint16_t* lev = d == 4u ? lev4 : lev8;
        const uint32_t* pr = j.rows + 2u * j.pairs_at[i];
        float sum[AMD_ROCM6_WAVE64__ZZPRIVATE_RT_MAX][AMD_ROCM6_WAVE64__ZZPRIVATE_PT_MAX];
        for (uint32_t a = 0u; a < RT; ++a)
            for (uint32_t q = 0u; q < PT; ++q) sum[a][q] = 0.0f;
        for (uint64_t k0 = 0u; k0 < j.cols; k0 += AMD_ROCM6_WAVE64__ZZPRIVATE_KT) {
            for (uint32_t e = t; e < TP * HK; e += n) {
                const uint32_t p = e / HK, c = e % HK;
                xs[p][c] = p0 + p < j.pairs[i]
                    ? ((const uint32_t*)(const void*)(j.x + (uint64_t)pr[2u * (p0 + p)] * j.cols + k0))[c] : 0u;
            }
            for (uint32_t e = t; e < TR * HK; e += n) {
                const uint32_t r = e / HK, c = e % HK;
                uint32_t v = 0u;
                if (r0 + r < j.out_rows[i]) {
                    const uint64_t col = k0 + 2u * c;
                    const uint32_t unit = ((const uint16_t*)(const void*)(codes + (r0 + r) * row_bytes))[col / per];
                    const uint32_t s = (uint32_t)((col % per) * d);
                    v = (uint32_t)lev[(unit >> s) & mask] | ((uint32_t)lev[(unit >> (s + (uint32_t)d)) & mask] << 16);
                }
                ws[r][c] = v;
            }
            __syncthreads();
            /* a slice's sums on their own, then onto the row's: a chain of 64 additions, not of the row's whole width */
            float part[AMD_ROCM6_WAVE64__ZZPRIVATE_RT_MAX][AMD_ROCM6_WAVE64__ZZPRIVATE_PT_MAX];
            for (uint32_t a = 0u; a < RT; ++a)
                for (uint32_t q = 0u; q < PT; ++q) part[a][q] = 0.0f;
            for (uint32_t c = 0u; c < HK; ++c) {
                uint32_t wv[AMD_ROCM6_WAVE64__ZZPRIVATE_RT_MAX], xv[AMD_ROCM6_WAVE64__ZZPRIVATE_PT_MAX];
                for (uint32_t a = 0u; a < RT; ++a) wv[a] = ws[tr + a][c];
                for (uint32_t q = 0u; q < PT; ++q) xv[q] = xs[tp + q][c];
                for (uint32_t a = 0u; a < RT; ++a)
                    for (uint32_t q = 0u; q < PT; ++q) part[a][q] = nn__silicon__dot2(wv[a], xv[q], part[a][q]);
            }
            for (uint32_t a = 0u; a < RT; ++a)
                for (uint32_t q = 0u; q < PT; ++q) sum[a][q] += part[a][q];
            __syncthreads();
        }
        const uint8_t* lut = (const uint8_t*)(uintptr_t)j.luts[i];
        for (uint32_t a = 0u; a < RT; ++a) {
            const uint64_t r = r0 + tr + a;
            if (r >= j.out_rows[i]) continue;
            const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
            for (uint32_t q = 0u; q < PT; ++q) {
                const uint64_t p = p0 + tp + q;
                if (p >= j.pairs[i]) continue;
                const uint64_t at = (uint64_t)pr[2u * p + 1u] * j.out_rows[i] + r;
                const float v = sum[a][q] * scale;
                if (j.out) (j.out + j.out_at[i])[at] = nn__kernels__zzabi_to_half(v, j.over);
                else j.acc[at] += v * nn__silicon__half_to_float(j.w[pr[2u * p]]);
            }
        }
    }
}
/* The two shapes: 64 matrix rows by 64 pairs for a matrix many rows read, 64 by 16 for a routed expert's few. */
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_tile_wide(amd_rocm6_wave64__tile_job j) {
    amd_rocm6_wave64__zzprivate_tile(&j, 64u, 64u, 4u, 4u);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_tile_few(amd_rocm6_wave64__tile_job j) {
    amd_rocm6_wave64__zzprivate_tile(&j, 64u, 16u, 4u, 1u);
}

/* A tile job for `count` matrices, each checked as the generic body would (4 or 8 bits, whole slices of 128 columns,
 * the rows the room holds), launched on the shape its pairs want; false when one does not fit, for the caller's
 * fallback. */
static inline bool amd_rocm6_wave64__zzprivate_tiles(amd_rocm6_wave64__tile_job* j) {
    if (j->count == 0u || j->count > NN__EXPERT__GROUPS_MAX || j->cols == 0u || j->cols % AMD_ROCM6_WAVE64__ZZPRIVATE_KT != 0u
     || ((uint64_t)(uintptr_t)j->x) % 4u != 0u)
        return false;
    uint64_t most = 0u;
    for (uint64_t i = 0u; i < j->count; ++i) {
        if ((j->d[i] != 4u && j->d[i] != 8u) || j->out_rows[i] == 0u || ((uint64_t)(uintptr_t)j->codes[i]) % 2u != 0u) return false;
        if (j->pairs[i] > most) most = j->pairs[i];
    }
    const bool few = most <= 16u;
    const uint64_t TP = few ? 16u : 64u;
    j->first[0] = 0u;
    for (uint64_t i = 0u; i < j->count; ++i)
        j->first[i + 1u] = j->first[i] + ((j->out_rows[i] + 63u) / 64u) * ((j->pairs[i] + TP - 1u) / TP);
    if (j->first[j->count] == 0u) return true;
    const uint32_t blocks = j->first[j->count] < NN__KERNELS__BLOCKS_MAX ? (uint32_t)j->first[j->count] : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { j };
    const size_t sizes[] = { sizeof *j };
    if (few)
        nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_tile_few, "amd_rocm6_wave64__zzprivate_tile_few", blocks,
                            AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 1u);
    else
        nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_tile_wide, "amd_rocm6_wave64__zzprivate_tile_wide", blocks,
                            AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 1u);
    return true;
}
/* One matrix's call as a tile job. */
static inline bool amd_rocm6_wave64__zzprivate_tiles_one(uint16_t* out, float* acc, const uint8_t* weights, uint64_t w_room,
                                                        const uint8_t* luts, uint64_t l_room, const uint16_t* x, const uint32_t* rows,
                                                        const uint16_t* w, uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols,
                                                        unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (count == 0u || row_bytes == 0u || out_rows > w_room / row_bytes || out_rows > l_room / 2u) return false;
    amd_rocm6_wave64__tile_job j = {};
    j.codes[0] = (uint64_t)(uintptr_t)weights; j.luts[0] = (uint64_t)(uintptr_t)luts; j.out_rows[0] = out_rows; j.d[0] = d;
    j.out_at[0] = 0u; j.pairs_at[0] = 0u; j.pairs[0] = count; j.count = 1u; j.cols = cols;
    j.out = out; j.acc = acc; j.x = x; j.rows = rows; j.w = w; j.over = over;
    return amd_rocm6_wave64__zzprivate_tiles(&j);
}

static inline bool amd_rocm6_wave64__zzprivate_rows_fit(uint64_t d, uint64_t cols, const void* codes, const void* x) {
    uint32_t span = 0u, held = 0u;
    return (d == 4u || d == 8u) && cols % 32u == 0u && amd_rocm6_wave64__zzprivate_span(cols / 32u, &span, &held)
        && ((uint64_t)(uintptr_t)codes) % 8u == 0u && ((uint64_t)(uintptr_t)x) % 4u == 0u;
}

static inline void amd_rocm6_wave64__zzprivate_launch_rows(amd_rocm6_wave64__rows_call c) {
    const uint64_t want = (c.out_rows + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t blocks = want == 0u ? 1u : want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &c };
    const size_t sizes[] = { sizeof c };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_expert_rows, "amd_rocm6_wave64__zzprivate_expert_rows", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 1u);
}

static void amd_rocm6_wave64__override_expert_rows(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                                   const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d, uint64_t out_rows,
                                                   uint64_t cols, unsigned int* over) {
    if (amd_rocm6_wave64__zzprivate_tiles_one(out, 0, weights, w_room, luts, l_room, x, rows, 0, count, d, out_rows, cols, over)) return;
    const uint64_t row_bytes = cols * d / 8u;             /* at 4 and 8 bits over whole blocks, which is all this takes */
    if (!amd_rocm6_wave64__zzprivate_rows_fit(d, cols, weights, x) || row_bytes == 0u || out_rows > w_room / row_bytes
     || out_rows > l_room / 2u || count == 0u) {
        nn__expert__zzabi_launch_rows(out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over);
        return;
    }
    const amd_rocm6_wave64__rows_call c = { out, 0, weights, luts, x, rows, 0, count, d, out_rows, cols, over };
    amd_rocm6_wave64__zzprivate_launch_rows(c);
}

static void amd_rocm6_wave64__override_expert_rows_sum(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                       uint64_t l_room, const uint16_t* x, const uint32_t* rows, const uint16_t* w,
                                                       uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols) {
    if (amd_rocm6_wave64__zzprivate_tiles_one(0, acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols, 0)) return;
    const uint64_t row_bytes = cols * d / 8u;             /* at 4 and 8 bits over whole blocks, which is all this takes */
    if (!amd_rocm6_wave64__zzprivate_rows_fit(d, cols, weights, x) || row_bytes == 0u || out_rows > w_room / row_bytes
     || out_rows > l_room / 2u || count == 0u) {
        nn__expert__zzabi_launch_rows_sum(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols);
        return;
    }
    const amd_rocm6_wave64__rows_call c = { 0, acc, weights, luts, x, rows, w, count, d, out_rows, cols, 0 };
    amd_rocm6_wave64__zzprivate_launch_rows(c);
}

static void amd_rocm6_wave64__override_expert_groups(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows,
                                                     uint64_t cols, unsigned int* over) {
    {
        amd_rocm6_wave64__tile_job j = {};
        for (uint64_t i = 0u; i < g.count && i < NN__EXPERT__GROUPS_MAX; ++i) {
            j.codes[i] = g.codes[i]; j.luts[i] = g.luts[i]; j.out_rows[i] = g.out_rows[i]; j.d[i] = g.d[i];
            j.out_at[i] = g.out_at[i]; j.pairs_at[i] = g.pairs_at[i]; j.pairs[i] = g.pairs[i];
        }
        j.count = g.count; j.cols = cols; j.out = out; j.acc = 0; j.x = x; j.rows = rows; j.w = 0; j.over = over;
        if (amd_rocm6_wave64__zzprivate_tiles(&j)) return;
    }
    bool fit = g.count != 0u && g.count <= NN__EXPERT__GROUPS_MAX;
    uint64_t total = 0u;
    for (uint64_t i = 0u; fit && i < g.count; ++i) {
        fit = amd_rocm6_wave64__zzprivate_rows_fit(g.d[i], cols, (const void*)(uintptr_t)g.codes[i], x) && g.out_rows[i] != 0u;
        total += g.out_rows[i];
    }
    if (!fit) { nn__expert__zzabi_launch_groups(out, g, x, rows, cols, over); return; }
    const uint64_t want = (total + AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS - 1u) / AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS;
    const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &out, &g, &x, &rows, &cols, &over };
    const size_t sizes[] = { sizeof out, sizeof g, sizeof x, sizeof rows, sizeof cols, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_expert_groups, "amd_rocm6_wave64__zzprivate_expert_groups", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 6u);
}

/* ══ ⭐ rmsnorm, A ROW A BLOCK, EIGHT HALVES A READ ═══════════════════════════════════════════════════════════
 * nn's body takes a row's squares one half at a time, `lane`, `lane + lanes`, …: at the 27B's 5120 that is twenty dependent
 * reads a lane, and a call cost ~24 us on the card (`MEASURED`, rocprof over a 27B decode) where its bytes are 30 KB. Here
 * each lane reads eight halves at once (16 bytes), so the row is three reads deep; the sum goes through the family's own
 * fold. The arithmetic is nn's — the mean of the squares in fp32, `1 / sqrt(mean + eps)`, each element `(x · scale) · w`
 * rounded once — with the squares grouped eight to a lane, so the sum's last bits are this kernel's own. ⛳ BOTH DOORS TAKE
 * IT, the one-row and the rows, so a prompt's rows and its positions norm alike. Rows whose width is not whole groups of
 * eight, or whose memory is not 16-byte aligned, take nn's generic door. */
static __device__ inline void amd_rocm6_wave64__zzprivate_rmsnorm_row(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n,
                                                                       float eps, unsigned int* over) {
    const uint32_t lane = (uint32_t)threadIdx.x, lanes = (uint32_t)blockDim.x;
    const uint64_t groups = n / 8u;
    float part = 0.0f;
    for (uint64_t g = lane; g < groups; g += lanes) {
        const uint4 q = ((const uint4*)(const void*)x)[g];
        const uint32_t words[4] = { q.x, q.y, q.z, q.w };
        for (uint32_t k = 0u; k < 4u; ++k) {
            const float a = nn__silicon__half_to_float((uint16_t)(words[k] & 0xFFFFu)), b = nn__silicon__half_to_float((uint16_t)(words[k] >> 16));
            part += a * a;
            part += b * b;
        }
    }
    const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(part) / (float)n + eps);
    for (uint64_t g = lane; g < groups; g += lanes) {
        const uint4 q = ((const uint4*)(const void*)x)[g], r = ((const uint4*)(const void*)w)[g];
        const uint32_t xs[4] = { q.x, q.y, q.z, q.w }, ws[4] = { r.x, r.y, r.z, r.w };
        uint32_t out[4];
        for (uint32_t k = 0u; k < 4u; ++k) {
            const uint32_t lo = nn__kernels__zzabi_to_half((nn__silicon__half_to_float((uint16_t)(xs[k] & 0xFFFFu)) * scale)
                                                           * nn__silicon__half_to_float((uint16_t)(ws[k] & 0xFFFFu)), over);
            const uint32_t hi = nn__kernels__zzabi_to_half((nn__silicon__half_to_float((uint16_t)(xs[k] >> 16)) * scale)
                                                           * nn__silicon__half_to_float((uint16_t)(ws[k] >> 16)), over);
            out[k] = lo | (hi << 16);
        }
        ((uint4*)(void*)o)[g] = make_uint4(out[0], out[1], out[2], out[3]);
    }
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_rmsnorm_rows(uint16_t* o, const uint16_t* x,
        const uint16_t* w, uint64_t n, uint64_t rows, uint64_t x_stride, uint64_t o_stride, float eps, unsigned int* over) {
    for (uint64_t r = blockIdx.x; r < rows; r += gridDim.x)
        amd_rocm6_wave64__zzprivate_rmsnorm_row(o + r * o_stride, x + r * x_stride, w, n, eps, over);
}
static inline bool amd_rocm6_wave64__zzprivate_norm_fit(const void* o, const void* x, const void* w, uint64_t n, uint64_t x_stride,
                                                         uint64_t o_stride) {
    return n != 0u && n % 8u == 0u && x_stride % 8u == 0u && o_stride % 8u == 0u && ((uint64_t)(uintptr_t)o) % 16u == 0u
        && ((uint64_t)(uintptr_t)x) % 16u == 0u && ((uint64_t)(uintptr_t)w) % 16u == 0u;
}
static void amd_rocm6_wave64__override_rmsnorm_rows(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, uint64_t rows,
                                                    uint64_t x_stride, uint64_t o_stride, float eps, unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_norm_fit(o, x, w, n, x_stride, o_stride) || rows == 0u) {
        nn__rmsnorm__zzabi_launch_rows(o, x, w, n, rows, x_stride, o_stride, eps, over);
        return;
    }
    const uint32_t blocks = rows < NN__KERNELS__BLOCKS_MAX ? (uint32_t)rows : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &o, &x, &w, &n, &rows, &x_stride, &o_stride, &eps, &over };
    const size_t sizes[] = { sizeof o, sizeof x, sizeof w, sizeof n, sizeof rows, sizeof x_stride, sizeof o_stride, sizeof eps, sizeof over };
    nn__silicon__launch((const void*)amd_rocm6_wave64__zzprivate_rmsnorm_rows, "amd_rocm6_wave64__zzprivate_rmsnorm_rows", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 9u);
}
static void amd_rocm6_wave64__override_rmsnorm(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, float eps,
                                               unsigned int* over) {
    if (!amd_rocm6_wave64__zzprivate_norm_fit(o, x, w, n, n, n)) { nn__rmsnorm__zzabi_launch(o, x, w, n, eps, over); return; }
    amd_rocm6_wave64__override_rmsnorm_rows(o, x, w, n, 1u, n, n, eps, over);
}

/* Into nn's table, once, when the family is first asked for it. */
static inline void amd_rocm6_wave64__override_doors(nn__doors* doors) {
    doors->turboquant_gemv      = amd_rocm6_wave64__override_turboquant_gemv;
    doors->turboquant_gemv_int8 = amd_rocm6_wave64__override_turboquant_gemv_int8;
    doors->hadamard_blocks              = amd_rocm6_wave64__override_hadamard_blocks;
    doors->attention_weights            = amd_rocm6_wave64__override_attention_weights;
    doors->attention_residual_mix       = amd_rocm6_wave64__override_attention_residual_mix;
    doors->attention_scores             = amd_rocm6_wave64__override_attention_scores;
    doors->attention_mix                = amd_rocm6_wave64__override_attention_mix;
    doors->deltanet_steps               = amd_rocm6_wave64__override_deltanet_steps;
    doors->hadamard_rotate      = amd_rocm6_wave64__override_hadamard_rotate;
    doors->turboquant_gemv_groups     = amd_rocm6_wave64__override_gemv_groups;
    doors->turboquant_gemv_groups_sum = amd_rocm6_wave64__override_gemv_groups_sum;
    doors->expert_rows          = amd_rocm6_wave64__override_expert_rows;
    doors->expert_rows_sum      = amd_rocm6_wave64__override_expert_rows_sum;
    doors->expert_groups        = amd_rocm6_wave64__override_expert_groups;
    doors->rmsnorm              = amd_rocm6_wave64__override_rmsnorm;
    doors->rmsnorm_rows         = amd_rocm6_wave64__override_rmsnorm_rows;
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_NN_CUH */
