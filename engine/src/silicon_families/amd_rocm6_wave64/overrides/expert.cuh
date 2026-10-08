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

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_OVERRIDES_EXPERT_CUH */
