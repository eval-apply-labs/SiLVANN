#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_EXPERT_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_EXPERT_CUH
/* ══ expert's DOORS, RUN FASTER — amd_rocm6_wave64 ════════════════════════════════════════════════════════════════════
 * nn's expert doors (`nn/cpu/opcodes/expert__abi.cuh`), where this family covers them: the expert-major GEMM, a
 * wave a row of the matrix or a block a tile, into halves, weighted into a sum, or over several matrices. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ── ⭐⭐ THE EXPERT-MAJOR GEMM — a wave a row of the matrix, its codes decoded once for every pair ─────────────
 * nn's definitions (`nn/gpu/kernels/kernels.cuh`, the expert-major GEMM): a matrix applied to a list of pairs (a row of `x`, a
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

/* nn's sixteen 4-bit levels' low and high bytes, four registers a plane, as `levels4` reads them: the codes 0-7 in the
 * first two, 8-15 in the last two. The GEMVs' `tables` keeps only the eight positive levels and decodes the sign
 * itself; `decode_block` below picks a level straight by its code, so it needs all sixteen. `MEASURED`
 * (test/src_abi_verbs_on_device.cpp, its one-block row): with the eight-level table this path read past its two
 * registers — 14,656 of 16,384 outputs NaN from the rows door, all of them from the groups door — at 4 bits and 32
 * columns, the one width the tiles leave to it. */
static __device__ inline void amd_rocm6_wave64__zzprivate_tables16(uint32_t* low, uint32_t* high) {
    const uint16_t* table = nn__turboquant__zzabi_levels(4u);
#pragma unroll
    for (uint32_t w = 0u; w < 4u; ++w) {
        low[w] = 0u; high[w] = 0u;
#pragma unroll
        for (uint32_t b = 0u; b < 4u; ++b) {
            low[w]  |= ((uint32_t)table[4u * w + b] & 0xFFu) << (8u * b);
            high[w] |= ((uint32_t)table[4u * w + b] >> 8) << (8u * b);
        }
    }
}

/* A block of 32 codes as sixteen registers of two fp16 levels, in column order. */
static __device__ inline void amd_rocm6_wave64__zzprivate_decode_block(uint32_t d, const uint8_t* row, uint64_t b, const uint32_t* low,
                                                                        const uint32_t* high, const uint16_t* table8, uint32_t* lv) {
    if (d == 4u) {
        const uint64_t* codes = (const uint64_t*)(row + b * 16u);
#pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            const uint32_t four = (uint32_t)(codes[e / 4u] >> (16u * (e % 4u)));
            const uint32_t lo = amd_rocm6_wave64__zzpackage_levels4(low, four);
            const uint32_t hi = amd_rocm6_wave64__zzpackage_levels4(high, four);
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
                        for (uint32_t e = 0u; e < 16u; ++e) part[q] = amd_rocm6_wave64__zzpackage_dot2(lv[h][e], xw[e], part[q]);
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
    uint32_t low[4], high[4];
    amd_rocm6_wave64__zzprivate_tables16(low, high);
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
    uint32_t low[4], high[4];
    amd_rocm6_wave64__zzprivate_tables16(low, high);
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
    uint64_t lined;                      /* 1: `x` on 16 bytes and every matrix's codes on 8 — the staging's wide loads (▶ the tile) */
    uint16_t* out; float* acc; const uint16_t* x; const uint32_t* rows; const uint16_t* w; unsigned int* over;
} amd_rocm6_wave64__tile_job;

/* One tile body, the shape its arguments — inlined into each kernel below, so the shape is a constant there and
 * the loops and the sums' registers come out as if written for it. `KS` the block's lanes split into that many groups
 * over a slice's columns: 1, every lane every column, a slice's sums on their own before the row's; 2, each half of
 * the block half the columns, summed straight into the row's, the halves added at the tile's end — twice the sums a
 * lane (`LONG_RT` rows) in the registers a slice's own sums took, a third fewer reads of shared memory a product.
 * ⛳ KS 1'S SOURCE IS THE SHORT ROWS' AND THE FEW PAIRS' AS IT WAS, LINE FOR LINE: its kernels compile to the same
 *   code (NN-114: with KS 2's unrolled loops shared, the 35B's prompt was 0.5% slower, the products the same) */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_RT_MAX 4u
#define AMD_ROCM6_WAVE64__ZZPRIVATE_PT_MAX 4u
#define AMD_ROCM6_WAVE64__ZZPRIVATE_LONG_RT 8u
template <bool HALVES, uint32_t KS>
static __device__ __forceinline__ void amd_rocm6_wave64__zzprivate_tile(const amd_rocm6_wave64__tile_job* jp, const uint32_t TR,
                                                                          const uint32_t TP, const uint32_t RT, const uint32_t PT) {
    const amd_rocm6_wave64__tile_job& j = *jp;
    __shared__ uint32_t ws[64][AMD_ROCM6_WAVE64__ZZPRIVATE_KT / 2u + (KS == 2u ? 2u : 1u)];
    __shared__ uint32_t xs[64][AMD_ROCM6_WAVE64__ZZPRIVATE_KT / 2u + (KS == 2u ? 2u : 1u)];
    const uint32_t HK = AMD_ROCM6_WAVE64__ZZPRIVATE_KT / 2u;
    const uint32_t t = (uint32_t)nn__silicon__lane(), n = (uint32_t)nn__silicon__lanes();
    /* nn's levels at 4 and 8 bits, read from here as a slice is staged rather than gathered a code at a time */
    __shared__ uint16_t lev4[16], lev8[256];
    for (uint32_t e = t; e < 256u; e += n) { lev8[e] = nn__turboquant__zzabi_levels(8u)[e]; if (e < 16u) lev4[e] = nn__turboquant__zzabi_levels(4u)[e]; }
    __syncthreads();
    /* a lane's sums: `RT` rows from `tr`, and `PT` pairs from `tp` a sixteenth of the tile apart — so the sixteen lanes
     *   reading a column read sixteen rows of `x` one after another, sixteen banks of shared memory (NN-114: four
     *   apart, they met in eight); with KS 2 `g` the lane's half of the block, `tl` its place in it, its rows `rs` apart */
    const uint32_t g = KS == 2u ? t / (n / 2u) : 0u, tl = KS == 2u ? t % (n / 2u) : t;
    const uint32_t tr = KS == 2u ? tl / (TP / PT) : (tl / (TP / PT)) * RT, rs = TR / RT, tp = tl % (TP / PT);
    for (uint64_t b = nn__silicon__block(); b < j.first[j.count]; b += nn__silicon__blocks()) {
        uint64_t i = 0u;
        while (b >= j.first[i + 1u]) ++i;
        const uint64_t tiles_p = (j.pairs[i] + TP - 1u) / TP, local = b - j.first[i];
        const uint64_t r0 = (local / tiles_p) * TR, p0 = (local % tiles_p) * TP;
        const uint64_t d = j.d[i], per = 16u / d, mask = (1u << d) - 1u, row_bytes = j.cols * d / 8u;
        const uint8_t* codes = (const uint8_t*)(uintptr_t)j.codes[i];
        const uint16_t* lev = d == 4u ? lev4 : lev8;
        const uint32_t* pr = j.rows + 2u * j.pairs_at[i];
        float sum[KS == 2u ? AMD_ROCM6_WAVE64__ZZPRIVATE_LONG_RT : AMD_ROCM6_WAVE64__ZZPRIVATE_RT_MAX][AMD_ROCM6_WAVE64__ZZPRIVATE_PT_MAX];
        /* ⛳ EVERY LOOP OVER KS 2'S SUMS UNROLLED: an 8 × 4 lane's left rolled indexed its sums, which then lived in memory */
        if (KS == 2u) {
#pragma unroll
            for (uint32_t a = 0u; a < RT; ++a)
#pragma unroll
                for (uint32_t q = 0u; q < PT; ++q) sum[a][q] = 0.0f;
        } else {
            for (uint32_t a = 0u; a < RT; ++a)
                for (uint32_t q = 0u; q < PT; ++q) sum[a][q] = 0.0f;
        }
        /* ⭐ THE STAGING'S OWN WORK ONCE A TILE (NN-114): a lane stages the same two rows of the matrix and the same two
         *   pairs' rows at every slice, four columns' halves of each — a row of the slice's 32 a lane's group of eight —
         *   so where each row starts, which row of `x` a pair reads and whether it is in the tile are worked out here,
         *   and a slice is one load of codes (4 bytes at 4 bits, 8 at 8) and one of `x` (16 bytes) an item. The halves
         *   staged are the ones the element-wise staging below stages, so the sums are its sums to the bit. */
        const bool lined = !HALVES && j.lined != 0u;
        /* each item's row as a byte offset from its matrix's codes and from `x` — under 4 GB each, as the fit check holds
         *   them (▶ `…_tiles`) — the bases the block's own: a register an item rather than a pointer's two */
        uint32_t wofs[2] = {0u, 0u}, xofs[2] = {0u, 0u};
        bool wok[2] = {false, false}, xok[2] = {false, false};
        if (lined)
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint32_t g = t + n * h, r = g / 8u;
                wok[h] = g < TR * 8u && r0 + r < j.out_rows[i];
                xok[h] = g < TP * 8u && p0 + r < j.pairs[i];
                if (wok[h]) wofs[h] = (uint32_t)((r0 + r) * row_bytes);
                if (xok[h]) xofs[h] = (uint32_t)((uint64_t)pr[2u * (p0 + r)] * j.cols * 2u);
            }
        for (uint64_t k0 = 0u; k0 < j.cols; k0 += AMD_ROCM6_WAVE64__ZZPRIVATE_KT) {
            if (lined) {
                for (uint32_t h = 0u; h < 2u; ++h) {
                    const uint32_t g = t + n * h, r = g / 8u, c = (g % 8u) * 4u;
                    if (g < TP * 8u) {
                        uint4 v = make_uint4(0u, 0u, 0u, 0u);
                        if (xok[h]) v = *(const uint4*)(const void*)((const uint8_t*)j.x + xofs[h] + 2u * (k0 + 2u * c));
                        xs[r][c] = v.x; xs[r][c + 1u] = v.y; xs[r][c + 2u] = v.z; xs[r][c + 3u] = v.w;
                    }
                    if (g < TR * 8u) {
                        uint32_t v[4] = {0u, 0u, 0u, 0u};
                        if (wok[h] && d == 4u) {             /* a byte two codes: the lower column's in its low nibble */
                            const uint32_t u = *(const uint32_t*)(const void*)(codes + wofs[h] + (k0 + 2u * c) / 2u);
                            for (uint32_t e = 0u; e < 4u; ++e) {
                                const uint32_t b = (u >> (8u * e)) & 0xFFu;
                                v[e] = (uint32_t)lev4[b & 0xFu] | ((uint32_t)lev4[b >> 4] << 16);
                            }
                        } else if (wok[h]) {                 /* a byte a code */
                            const uint2 u = *(const uint2*)(const void*)(codes + wofs[h] + k0 + 2u * c);
                            for (uint32_t e = 0u; e < 4u; ++e) {
                                const uint32_t two = ((e < 2u ? u.x : u.y) >> (16u * (e % 2u))) & 0xFFFFu;
                                v[e] = (uint32_t)lev8[two & 0xFFu] | ((uint32_t)lev8[two >> 8] << 16);
                            }
                        }
                        ws[r][c] = v[0]; ws[r][c + 1u] = v[1]; ws[r][c + 2u] = v[2]; ws[r][c + 3u] = v[3];
                    }
                }
                __syncthreads();
            }
            for (uint32_t e = t; !lined && e < TP * HK; e += n) {
                const uint32_t p = e / HK, c = e % HK;
                xs[p][c] = p0 + p < j.pairs[i] && (!HALVES || k0 + 2u * c < j.cols)
                    ? ((const uint32_t*)(const void*)(j.x + (uint64_t)pr[2u * (p0 + p)] * j.cols + k0))[c] : 0u;
            }
            for (uint32_t e = t; !lined && e < TR * HK; e += n) {
                const uint32_t r = e / HK, c = e % HK;
                uint32_t v = 0u;
                if (HALVES) {                                             /* the halves themselves, the tail as zeros */
                    const uint64_t col = k0 + 2u * c;
                    if (r0 + r < j.out_rows[i] && col < j.cols)
                        v = ((const uint32_t*)(const void*)(codes + (r0 + r) * row_bytes))[col / 2u];
                } else if (r0 + r < j.out_rows[i]) {
                    const uint64_t col = k0 + 2u * c;
                    const uint32_t unit = ((const uint16_t*)(const void*)(codes + (r0 + r) * row_bytes))[col / per];
                    const uint32_t s = (uint32_t)((col % per) * d);
                    v = (uint32_t)lev[(unit >> s) & mask] | ((uint32_t)lev[(unit >> (s + (uint32_t)d)) & mask] << 16);
                }
                ws[r][c] = v;
            }
            if (!lined) __syncthreads();
            if (KS == 1u) {
                /* a slice's sums on their own, then onto the row's: a chain of 64 additions, not of the row's whole width */
                float part[AMD_ROCM6_WAVE64__ZZPRIVATE_RT_MAX][AMD_ROCM6_WAVE64__ZZPRIVATE_PT_MAX];
                for (uint32_t a = 0u; a < RT; ++a)
                    for (uint32_t q = 0u; q < PT; ++q) part[a][q] = 0.0f;
                /* two columns a step: the compiler's four held the shared reads' registers past three waves (▶ the bound) */
#pragma unroll 2
                for (uint32_t c = 0u; c < HK; ++c) {
                    uint32_t wv[AMD_ROCM6_WAVE64__ZZPRIVATE_RT_MAX], xv[AMD_ROCM6_WAVE64__ZZPRIVATE_PT_MAX];
                    for (uint32_t a = 0u; a < RT; ++a) wv[a] = ws[tr + a][c];
                    for (uint32_t q = 0u; q < PT; ++q) xv[q] = xs[tp + q * (TP / PT)][c];
                    for (uint32_t a = 0u; a < RT; ++a)
                        for (uint32_t q = 0u; q < PT; ++q) part[a][q] = nn__silicon__dot2(wv[a], xv[q], part[a][q]);
                }
                for (uint32_t a = 0u; a < RT; ++a)
                    for (uint32_t q = 0u; q < PT; ++q) sum[a][q] += part[a][q];
            } else {
                /* this group's columns of the slice, straight onto the row's sums, two columns a read (rows of 34
                 *   words: a lane's eight rows 8 apart and its pairs 16 apart meet no bank twice) — a step at a time,
                 *   which keeps the shape within three waves' registers with its spills outside the loop (NN-114) */
#pragma unroll 1
                for (uint32_t c = g * (HK / KS); c < (g + 1u) * (HK / KS); c += 2u) {
                    uint2 wv[AMD_ROCM6_WAVE64__ZZPRIVATE_LONG_RT], xv[AMD_ROCM6_WAVE64__ZZPRIVATE_PT_MAX];
#pragma unroll
                    for (uint32_t a = 0u; a < RT; ++a) wv[a] = *(const uint2*)(const void*)&ws[tr + a * rs][c];
#pragma unroll
                    for (uint32_t q = 0u; q < PT; ++q) xv[q] = *(const uint2*)(const void*)&xs[tp + q * (TP / PT)][c];
#pragma unroll
                    for (uint32_t a = 0u; a < RT; ++a)
#pragma unroll
                        for (uint32_t q = 0u; q < PT; ++q) sum[a][q] = nn__silicon__dot2(wv[a].x, xv[q].x, sum[a][q]);
#pragma unroll
                    for (uint32_t a = 0u; a < RT; ++a)
#pragma unroll
                        for (uint32_t q = 0u; q < PT; ++q) sum[a][q] = nn__silicon__dot2(wv[a].y, xv[q].y, sum[a][q]);
                }
            }
            __syncthreads();
        }
        /* the two halves' sums added, each half finishing half the rows: it leaves the rows it does not finish in the
         *   shared memory the slices used, lane by lane, and adds the other half's — a sum of two, the same whichever is
         *   first */
        if (KS == 2u) {
            float* mine = (float*)(void*)(g == 0u ? &ws[0][0] : &xs[0][0]);
            const float* theirs = (const float*)(const void*)(g == 0u ? &xs[0][0] : &ws[0][0]);
            const uint32_t L = n / 2u;
#pragma unroll
            for (uint32_t a = 0u; a < RT; ++a)
#pragma unroll
                for (uint32_t q = 0u; q < PT; ++q)
                    if ((a < RT / 2u) != (g == 0u)) mine[((a % (RT / 2u)) * PT + q) * L + tl] = sum[a][q];
            __syncthreads();
#pragma unroll
            for (uint32_t a = 0u; a < RT; ++a)
#pragma unroll
                for (uint32_t q = 0u; q < PT; ++q)
                    if ((a < RT / 2u) == (g == 0u)) sum[a][q] += theirs[((a % (RT / 2u)) * PT + q) * L + tl];
            __syncthreads();
        }
        const uint8_t* lut = (const uint8_t*)(uintptr_t)j.luts[i];
        if (KS == 2u) {
#pragma unroll
            for (uint32_t a = 0u; a < RT; ++a) {
                if ((a < RT / 2u) != (g == 0u)) continue;
                const uint64_t r = r0 + tr + a * rs;
                if (r >= j.out_rows[i]) continue;
                const float scale = HALVES ? 1.0f
                    : nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
#pragma unroll
                for (uint32_t q = 0u; q < PT; ++q) {
                    const uint64_t p = p0 + tp + q * (TP / PT);
                    if (p >= j.pairs[i]) continue;
                    const uint64_t at = (uint64_t)pr[2u * p + 1u] * j.out_rows[i] + r;
                    const float v = sum[a][q] * scale;
                    if (j.out) (j.out + j.out_at[i])[at] = nn__kernels__zzabi_to_half(v, j.over);
                    else j.acc[at] += v * nn__silicon__half_to_float(j.w[pr[2u * p]]);
                }
            }
            continue;
        }
        for (uint32_t a = 0u; a < RT; ++a) {
            const uint64_t r = r0 + tr + a;
            if (r >= j.out_rows[i]) continue;
            const float scale = HALVES ? 1.0f                        /* a 16-bit row carries no scale */
                : nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
            for (uint32_t q = 0u; q < PT; ++q) {
                const uint64_t p = p0 + tp + q * (TP / PT);
                if (p >= j.pairs[i]) continue;
                const uint64_t at = (uint64_t)pr[2u * p + 1u] * j.out_rows[i] + r;
                const float v = sum[a][q] * scale;
                if (j.out) (j.out + j.out_at[i])[at] = nn__kernels__zzabi_to_half(v, j.over);
                else j.acc[at] += v * nn__silicon__half_to_float(j.w[pr[2u * p]]);
            }
        }
    }
}
/* The two shapes: 64 matrix rows by 64 pairs for a matrix many rows read, 64 by 16 for a routed expert's few — and the
 * first again for matrices of 16 bits (nn's width of halves, no scale, any even number of columns: a vision tower's
 * 4,304-wide MLP), a kernel of its own so the codes' are compiled as they were. The first sums 4 rows by 4 pairs a
 * lane, every lane every column; over rows of `LONG_COLS` columns or more, 8 rows by 4 pairs a lane, the block's halves
 * each half a slice's columns — its sums a third fewer reads of shared memory a product, its tile's end a sum of the
 * halves and its spills, which a short row's few slices do not repay (NN-114, the AMD bench over 512 to 17,408 columns:
 * 9-15% slower at 512, 3% at 1,024, 2% faster at 2,048, 7-11% from 3,072 on; from 2,048 the 35B's prompt was no faster
 * than from 4,096). */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_LONG_COLS 4096u
/* ⛳ THREE WAVES A SIMD, as a block's 17 KB of shared memory allows (three blocks a CU): the bound holds the long rows'
 *   shape to 84 registers — 24 spilled, around a tile's start and its end, none in the products' loop (NN-114: two
 *   waves without spills were 13-15% slower) */
#define AMD_ROCM6_WAVE64__ZZPRIVATE_TILE_BOUNDS __launch_bounds__(256, 3)
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_TILE_BOUNDS void amd_rocm6_wave64__zzprivate_tile_wide(amd_rocm6_wave64__tile_job j) {
    amd_rocm6_wave64__zzprivate_tile<false, 1u>(&j, 64u, 64u, 4u, 4u);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_TILE_BOUNDS void amd_rocm6_wave64__zzprivate_tile_wide_long(amd_rocm6_wave64__tile_job j) {
    amd_rocm6_wave64__zzprivate_tile<false, 2u>(&j, 64u, 64u, 8u, 4u);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_tile_few(amd_rocm6_wave64__tile_job j) {
    amd_rocm6_wave64__zzprivate_tile<false, 1u>(&j, 64u, 16u, 4u, 1u);
}
static __global__ AMD_ROCM6_WAVE64__ZZPRIVATE_BOUNDS void amd_rocm6_wave64__zzprivate_tile_wide_halves(amd_rocm6_wave64__tile_job j) {
    amd_rocm6_wave64__zzprivate_tile<true, 1u>(&j, 64u, 64u, 4u, 4u);
}

/* A tile job for `count` matrices, each checked as the generic body would — every one 4 or 8 bits over whole slices of
 * 64 columns, or every one 16 bits over any even number of columns — the rows the room holds, launched on the shape
 * its pairs want; false when one does not fit, for the caller's fallback. */
static inline bool amd_rocm6_wave64__zzprivate_tiles(amd_rocm6_wave64__tile_job* j) {
    if (j->count == 0u || j->count > NN__EXPERT__GROUPS_MAX || j->cols == 0u || ((uint64_t)(uintptr_t)j->x) % 4u != 0u)
        return false;
    const bool halves = j->d[0] == 16u;
    if (halves ? j->cols % 2u != 0u : j->cols % AMD_ROCM6_WAVE64__ZZPRIVATE_KT != 0u) return false;
    uint64_t most = 0u;
    for (uint64_t i = 0u; i < j->count; ++i) {
        if (halves ? j->d[i] != 16u || ((uint64_t)(uintptr_t)j->codes[i]) % 4u != 0u
                   : (j->d[i] != 4u && j->d[i] != 8u) || ((uint64_t)(uintptr_t)j->codes[i]) % 2u != 0u) return false;
        if (j->out_rows[i] == 0u) return false;
        if (j->pairs[i] > most) most = j->pairs[i];
    }
    /* the staging's wide loads (▶ the tile): `x` on 16 bytes, every matrix's codes on 8 — the codes' rows and slices fall
     *   on those widths from there, the columns being whole slices of 64 */
    /*   — and their rows' offsets in 32 bits: every matrix under 4 GB, and `x` 65,536 rows of `cols` at most (ASSUMED:
     *   nn's chunks are 4,096 rows at most; a longer one would need its rows' count here) */
    j->lined = 0u;
    if (!halves && ((uint64_t)(uintptr_t)j->x) % 16u == 0u && j->cols * 2u * 65536u <= (1ull << 32)) {
        j->lined = 1u;
        for (uint64_t i = 0u; i < j->count; ++i)
            if (((uint64_t)(uintptr_t)j->codes[i]) % 8u != 0u || j->out_rows[i] * (j->cols * j->d[i] / 8u) >= (1ull << 32)) j->lined = 0u;
    }
    const bool few = most <= 16u;
    /* a token's few rows of 16 bits stay where they were: through a tile the 35B's answer was 1.1% slower (`MEASURED`
     * on the MI50, four rounds, test/src_runtime_bench.py), where a prompt's many rows were 7% faster */
    if (halves && few) return false;
    const uint64_t TP = few ? 16u : 64u;
    j->first[0] = 0u;
    for (uint64_t i = 0u; i < j->count; ++i)
        j->first[i + 1u] = j->first[i] + ((j->out_rows[i] + 63u) / 64u) * ((j->pairs[i] + TP - 1u) / TP);
    if (j->first[j->count] == 0u) return true;
    const uint32_t blocks = j->first[j->count] < NN__KERNELS__BLOCKS_MAX ? (uint32_t)j->first[j->count] : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { j };
    const size_t sizes[] = { sizeof *j };
    const void* kernel = few ? (const void*)amd_rocm6_wave64__zzprivate_tile_few
                             : (halves ? (const void*)amd_rocm6_wave64__zzprivate_tile_wide_halves
                                       : (j->cols >= AMD_ROCM6_WAVE64__ZZPRIVATE_LONG_COLS
                                              ? (const void*)amd_rocm6_wave64__zzprivate_tile_wide_long
                                              : (const void*)amd_rocm6_wave64__zzprivate_tile_wide));
    nn__silicon__launch(kernel, few ? "amd_rocm6_wave64__zzprivate_tile_few" : "amd_rocm6_wave64__zzprivate_tile_wide", blocks,
                        AMD_ROCM6_WAVE64__ZZPRIVATE_ROWS * AMD_ROCM6_WAVE64__ZZPRIVATE_WAVE, args, sizes, 1u);
    return true;
}

/* One matrix's call as a tile job. */
static inline bool amd_rocm6_wave64__zzprivate_tiles_one(uint16_t* out, float* acc, const uint8_t* weights, uint64_t w_room,
                                                        const uint8_t* luts, uint64_t l_room, const uint16_t* x, const uint32_t* rows,
                                                        const uint16_t* w, uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols,
                                                        unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (count == 0u || row_bytes == 0u || out_rows > w_room / row_bytes || (d != 16u && out_rows > l_room / 2u)) return false;
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

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_EXPERT_CUH */
