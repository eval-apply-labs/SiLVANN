#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_EXPERT_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_EXPERT_CUH
/* ══ expert's DOORS, ON THE TENSOR CORES — nvidia_cuda12 ══════════════════════════════════════════════════════════════
 * nn's expert doors (`nn/cpu/opcodes/expert__abi.cuh`), where this family covers them: the expert-major GEMM through
 * `mma.sync`, into halves, weighted into a sum, or over several matrices. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ══ ⭐⭐ THE EXPERT-MAJOR GEMM IN TILES, ON THE TENSOR CORES ═══════════════════════════════════════════════════════════
 * nn's definitions (`nn/gpu/kernels/kernels.cuh`, the expert-major GEMM): a matrix applied to a list of pairs (a row of
 * `x`, a row of `out`), plain, weighted into an fp32 sum, or several matrices in one launch. A block takes a TILE — eight
 * warps of 16 matrix rows each, by `8 · NT` pairs — and walks the columns 64 at a time:
 *   · the tile's slices of `x` are staged in shared memory, two at once: the next is copied in (`cp.async`) while this
 *     one is multiplied, and every warp reads its pairs' columns from there
 *   · each lane loads its own codes of its warp's rows straight into registers, a slice ahead of the one being multiplied
 *     (two for the narrow tiles), and decodes them there into A operands (nn's fp16 levels): a code never touches shared memory
 *     and is decoded once a tile, then multiplied into all of the tile's pairs by `mma.sync` (▶ `kernels.cuh`), in fp32
 *   · ⭐ THE COLUMNS ARE TAKEN IN ANOTHER ORDER, THE SAME ONE FOR BOTH OPERANDS. The instruction gives lane `t` of a
 *     group of four the columns `2t, 2t + 1, 2t + 8, 2t + 9` of a step of 16; over a slice's four steps, lane `t` here
 *     takes the slice's columns `16t .. 16t + 15` instead, four a step, so its codes of a row are one 8-byte load at 4
 *     bits (two at 8) and its `x` of a pair, two steps at a time, one 16-byte read. A dot product is the same sum over
 *     its columns in any order; what can differ is where the fp32 sum rounds.
 * The tile is 128 matrix rows by 64 pairs for a matrix many rows read (a dense projection over a chunk), by 32 or by 16
 * for the few rows each routed expert gets. Covers 4 and 8 bits over whole slices of 64 columns, codes on 8 bytes and `x`
 * on 4, and 16 bits over any even number of columns (the halves' tile below); anything else is the generic door. A row of `out` is written by the one lane holding its sum: nn's contract has a
 * call's pairs name distinct out rows (the MoE's down: a pick's row is its own), so the weighted sum needs no guard.
 * ⛳ THE BARRIERS are the tile's own: the block waits once the levels are staged, once a slice has landed, and once more
 *   before the slice after next overwrites it — the lanes of one launch waiting for each other, as the family's fold does. */
#define NVIDIA_CUDA12__ZZPRIVATE_KT      64u                               /* columns a slice */
#define NVIDIA_CUDA12__ZZPRIVATE_WARPS   8u
#define NVIDIA_CUDA12__ZZPRIVATE_ROWS    (16u * NVIDIA_CUDA12__ZZPRIVATE_WARPS)   /* matrix rows a tile */
#define NVIDIA_CUDA12__ZZPRIVATE_XS      (NVIDIA_CUDA12__ZZPRIVATE_KT + 8u)  /* a staged row of `x`, in halves: 144 bytes */
#define NVIDIA_CUDA12__ZZPRIVATE_NT_MAX  8u
#define NVIDIA_CUDA12__ZZPRIVATE_BOUNDS  __launch_bounds__(256, 2)

typedef struct nvidia_cuda12__tile_job {
    uint64_t codes[NN__EXPERT__GROUPS_MAX], luts[NN__EXPERT__GROUPS_MAX], out_rows[NN__EXPERT__GROUPS_MAX];
    uint64_t d[NN__EXPERT__GROUPS_MAX], out_at[NN__EXPERT__GROUPS_MAX], pairs_at[NN__EXPERT__GROUPS_MAX];
    uint64_t pairs[NN__EXPERT__GROUPS_MAX], first[NN__EXPERT__GROUPS_MAX + 1];
    uint64_t count, cols, x16;
    uint16_t* out; float* acc; const uint16_t* x; const uint32_t* rows; const uint16_t* w; unsigned int* over;
} nvidia_cuda12__tile_job;

/* One lane's codes of rows `g` and `g + 8` of its warp's sixteen, at the slice's columns `16t .. 16t + 15`: `D / 2` words a
 * row — two at 4 bits, four at 8. A row past the matrix (`row` null) reads nothing; its sums are never written. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_codes(const uint32_t D, const uint8_t* lo, const uint8_t* hi, uint64_t k0,
                                                                       uint32_t t, uint32_t raw[2][4]) {
    const uint8_t* row[2] = { lo, hi };
#pragma unroll
    for (uint32_t r = 0u; r < 2u; ++r) {
        raw[r][0] = raw[r][1] = raw[r][2] = raw[r][3] = 0u;
        if (!row[r]) continue;
        const uint2* p = (const uint2*)(const void*)(row[r] + k0 * D / 8u + 2u * D * t);
#pragma unroll
        for (uint32_t h = 0u; h < D / 4u; ++h) {
            const uint2 v = p[h];
            raw[r][2u * h] = v.x; raw[r][2u * h + 1u] = v.y;
        }
    }
}

/* Those codes as the A operands of step `s`: four registers, each two fp16 levels in column order — rows `g` and `g + 8`
 * at the step's first two columns, then at its last two. At 4 bits a byte is two codes, and `pair4` holds every byte's
 * two levels as one register; at 8 bits a code is a byte, and `lev8` its level. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_operand(const uint32_t D, const uint32_t raw[2][4], uint32_t s,
                                                                         const uint32_t* pair4, const uint16_t* lev8, uint32_t a[4]) {
#pragma unroll
    for (uint32_t r = 0u; r < 2u; ++r) {
        if (D == 4u) {
            const uint32_t word = raw[r][s / 2u], at = 16u * (s % 2u);
            a[r]      = pair4[(word >> at) & 0xFFu];
            a[2u + r] = pair4[(word >> (at + 8u)) & 0xFFu];
        } else {
            const uint32_t word = raw[r][s];
            a[r]      = (uint32_t)lev8[word & 0xFFu] | ((uint32_t)lev8[(word >> 8) & 0xFFu] << 16);
            a[2u + r] = (uint32_t)lev8[(word >> 16) & 0xFFu] | ((uint32_t)lev8[word >> 24] << 16);
        }
    }
}

/* A slice of the tile's `x` into `xs`: pair `p`'s columns `k0 .. k0 + 63`, sixteen bytes a copy; a pair past the
 * matrix's is zeros. Copied without a register where `x` lies on sixteen bytes, loaded and stored where it lies on four. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_stage(const nvidia_cuda12__tile_job* j, uint16_t (*xs)[NVIDIA_CUDA12__ZZPRIVATE_XS],
                                                                       const uint32_t* pr, uint64_t p0, uint64_t pairs, uint64_t k0,
                                                                       uint32_t TP, uint32_t lane, uint32_t lanes) {
    for (uint32_t e = lane; e < TP * 8u; e += lanes) {
        const uint32_t p = e / 8u, c = e % 8u;
        const bool valid = p0 + p < pairs;
        const uint16_t* src = valid ? j->x + (uint64_t)pr[2u * (p0 + p)] * j->cols + k0 + 8u * c : j->x;
        if (j->x16) nvidia_cuda12__zzpackage_copy16(&xs[p][8u * c], src, valid);
        else {
            const uint32_t* s4 = (const uint32_t*)(const void*)src;
            *(uint4*)(void*)&xs[p][8u * c] = valid ? make_uint4(s4[0], s4[1], s4[2], s4[3]) : make_uint4(0u, 0u, 0u, 0u);
        }
    }
    nvidia_cuda12__zzpackage_copies_issued();
}

/* One tile body, its pairs' width `8 · NT`, its codes' width `D` and how many slices ahead its codes are loaded the
 * arguments — inlined into each kernel below, so all three are constants there and the loops and the registers come out as
 * if written for them. A launch's matrices all have its kernel's `D`. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_tile(const nvidia_cuda12__tile_job* j, const uint32_t NT, const uint32_t D,
                                                                      const uint32_t AHEAD) {
    __shared__ uint32_t pair4[256];
    __shared__ uint16_t lev8[256];
    __shared__ __align__(16) uint16_t xs[2][NVIDIA_CUDA12__ZZPRIVATE_NT_MAX * 8u][NVIDIA_CUDA12__ZZPRIVATE_XS];
    const uint32_t lane = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    for (uint32_t e = lane; e < 256u; e += lanes) {
        const uint16_t* lev4 = nn__turboquant__zzabi_levels(4u);
        pair4[e] = (uint32_t)lev4[e & 15u] | ((uint32_t)lev4[e >> 4] << 16);
        lev8[e] = nn__turboquant__zzabi_levels(8u)[e];
    }
    __syncthreads();
    const uint32_t warp = lane / NVIDIA_CUDA12__ZZPACKAGE_WARP, l = lane % NVIDIA_CUDA12__ZZPACKAGE_WARP, g = l / 4u, t = l % 4u;
    const uint32_t TP = 8u * NT, KT = NVIDIA_CUDA12__ZZPRIVATE_KT;
    for (uint64_t b = nn__silicon__block(); b < j->first[j->count]; b += nn__silicon__blocks()) {
        uint64_t i = 0u;
        while (b >= j->first[i + 1u]) ++i;
        const uint64_t tiles_p = (j->pairs[i] + TP - 1u) / TP, local = b - j->first[i];
        const uint64_t r0 = (local / tiles_p) * NVIDIA_CUDA12__ZZPRIVATE_ROWS + 16u * warp, p0 = (local % tiles_p) * TP;
        const uint64_t out_rows = j->out_rows[i], pairs = j->pairs[i], cols = j->cols, row_bytes = cols * D / 8u;
        const uint8_t* codes = (const uint8_t*)(uintptr_t)j->codes[i];
        const uint32_t* pr = j->rows + 2u * j->pairs_at[i];
        const uint8_t* lo = r0 + g < out_rows ? codes + (r0 + g) * row_bytes : 0;
        const uint8_t* hi = r0 + g + 8u < out_rows ? codes + (r0 + g + 8u) * row_bytes : 0;
        float acc[NVIDIA_CUDA12__ZZPRIVATE_NT_MAX][4];
#pragma unroll
        for (uint32_t n = 0u; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        nvidia_cuda12__zzprivate_stage(j, xs[0], pr, p0, pairs, 0u, TP, lane, lanes);
        /* the codes of the slice `AHEAD - 1` past the next, loaded while the next is multiplied */
        uint32_t raw[2][4], far[2][4];
        nvidia_cuda12__zzprivate_codes(D, lo, hi, 0u, t, raw);
        if (AHEAD > 1u && KT < cols) nvidia_cuda12__zzprivate_codes(D, lo, hi, KT, t, far);
        uint32_t buf = 0u;
        for (uint64_t k0 = 0u; k0 < cols; k0 += KT, buf ^= 1u) {
            const bool more = k0 + KT < cols;
            if (more) nvidia_cuda12__zzprivate_stage(j, xs[buf ^ 1u], pr, p0, pairs, k0 + KT, TP, lane, lanes);
            nvidia_cuda12__zzpackage_copies_landed(more ? 1u : 0u);
            __syncthreads();
            uint32_t now[2][4];
#pragma unroll
            for (uint32_t r = 0u; r < 2u; ++r)
#pragma unroll
                for (uint32_t q = 0u; q < 4u; ++q) now[r][q] = raw[r][q];
            if (AHEAD > 1u) {
#pragma unroll
                for (uint32_t r = 0u; r < 2u; ++r)
#pragma unroll
                    for (uint32_t q = 0u; q < 4u; ++q) raw[r][q] = far[r][q];
                if (k0 + 2u * KT < cols) nvidia_cuda12__zzprivate_codes(D, lo, hi, k0 + 2u * KT, t, far);
            } else if (more) nvidia_cuda12__zzprivate_codes(D, lo, hi, k0 + KT, t, raw);
#pragma unroll
            for (uint32_t s2 = 0u; s2 < 2u; ++s2) {
                uint32_t a0[4], a1[4];
                nvidia_cuda12__zzprivate_operand(D, now, 2u * s2, pair4, lev8, a0);
                nvidia_cuda12__zzprivate_operand(D, now, 2u * s2 + 1u, pair4, lev8, a1);
#pragma unroll
                for (uint32_t n = 0u; n < NT; ++n) {
                    const uint4 q = *(const uint4*)(const void*)&xs[buf][8u * n + g][16u * t + 8u * s2];
                    nvidia_cuda12__zzpackage_mma(acc[n], a0[0], a0[1], a0[2], a0[3], q.x, q.y);
                    nvidia_cuda12__zzpackage_mma(acc[n], a1[0], a1[1], a1[2], a1[3], q.z, q.w);
                }
            }
            __syncthreads();
        }
        const uint8_t* lut = (const uint8_t*)(uintptr_t)j->luts[i];
#pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint64_t r = r0 + g + 8u * h;
            if (r >= out_rows) continue;
            const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
#pragma unroll
            for (uint32_t n = 0u; n < NT; ++n)
#pragma unroll
                for (uint32_t e = 0u; e < 2u; ++e) {
                    const uint64_t p = p0 + 8u * n + 2u * t + e;
                    if (p >= pairs) continue;
                    const uint64_t at = (uint64_t)pr[2u * p + 1u] * out_rows + r;
                    const float v = acc[n][2u * h + e] * scale;
                    if (j->out) (j->out + j->out_at[i])[at] = nn__kernels__zzabi_to_half(v, j->over);
                    else j->acc[at] += v * nn__silicon__half_to_float(j->w[pr[2u * p]]);
                }
        }
    }
}
/* The three widths — 64 pairs for a matrix many rows read, 32 and 16 for a routed expert's few — at each of the two codes'
 * widths. The narrow tiles read little `x` a slice and wait on their codes, so theirs are loaded two slices ahead. */
static __global__ NVIDIA_CUDA12__ZZPRIVATE_BOUNDS void nvidia_cuda12__zzprivate_tile_64_4(nvidia_cuda12__tile_job j) {
    nvidia_cuda12__zzprivate_tile(&j, 8u, 4u, 1u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_BOUNDS void nvidia_cuda12__zzprivate_tile_32_4(nvidia_cuda12__tile_job j) {
    nvidia_cuda12__zzprivate_tile(&j, 4u, 4u, 2u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_BOUNDS void nvidia_cuda12__zzprivate_tile_16_4(nvidia_cuda12__tile_job j) {
    nvidia_cuda12__zzprivate_tile(&j, 2u, 4u, 2u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_BOUNDS void nvidia_cuda12__zzprivate_tile_64_8(nvidia_cuda12__tile_job j) {
    nvidia_cuda12__zzprivate_tile(&j, 8u, 8u, 1u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_BOUNDS void nvidia_cuda12__zzprivate_tile_32_8(nvidia_cuda12__tile_job j) {
    nvidia_cuda12__zzprivate_tile(&j, 4u, 8u, 2u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_BOUNDS void nvidia_cuda12__zzprivate_tile_16_8(nvidia_cuda12__tile_job j) {
    nvidia_cuda12__zzprivate_tile(&j, 2u, 8u, 2u);
}

/* ⭐ THE SAME TILE FOR A MATRIX OF 16 BITS — nn's width of halves, no levels and no scale: a VAE's convolutions over a
 * picture's pixels, a text encoder's exact layers over a prompt, a vision tower's MLP. A lane's 16 columns of a row are
 * eight words, the A operands as they stand, loaded a slice ahead; the columns are any even number, a slice past them
 * zeros in both operands (a 3 x 3 convolution of 96 channels is 864 columns, 13½ slices). `x` is staged a word at a time,
 * its row on four bytes. A kernel of its own, so the codes' tiles are compiled as they were. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_halves_row(const uint16_t* row, uint64_t k0, uint64_t cols, uint32_t t,
                                                                            uint32_t raw[8]) {
#pragma unroll
    for (uint32_t q = 0u; q < 8u; ++q) {
        const uint64_t c = k0 + 16u * t + 2u * q;
        raw[q] = row && c < cols ? *(const uint32_t*)(const void*)(row + c) : 0u;
    }
}
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_tile_halves(const nvidia_cuda12__tile_job* j, const uint32_t NT) {
    __shared__ __align__(16) uint16_t xs[NVIDIA_CUDA12__ZZPRIVATE_NT_MAX * 8u][NVIDIA_CUDA12__ZZPRIVATE_XS];
    const uint32_t lane = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    const uint32_t warp = lane / NVIDIA_CUDA12__ZZPACKAGE_WARP, l = lane % NVIDIA_CUDA12__ZZPACKAGE_WARP, g = l / 4u, t = l % 4u;
    const uint32_t TP = 8u * NT, KT = NVIDIA_CUDA12__ZZPRIVATE_KT;
    for (uint64_t b = nn__silicon__block(); b < j->first[j->count]; b += nn__silicon__blocks()) {
        uint64_t i = 0u;
        while (b >= j->first[i + 1u]) ++i;
        const uint64_t tiles_p = (j->pairs[i] + TP - 1u) / TP, local = b - j->first[i];
        const uint64_t r0 = (local / tiles_p) * NVIDIA_CUDA12__ZZPRIVATE_ROWS + 16u * warp, p0 = (local % tiles_p) * TP;
        const uint64_t out_rows = j->out_rows[i], pairs = j->pairs[i], cols = j->cols;
        const uint16_t* w = (const uint16_t*)(uintptr_t)j->codes[i];
        const uint32_t* pr = j->rows + 2u * j->pairs_at[i];
        const uint16_t* lo = r0 + g < out_rows ? w + (r0 + g) * cols : 0;
        const uint16_t* hi = r0 + g + 8u < out_rows ? w + (r0 + g + 8u) * cols : 0;
        float acc[NVIDIA_CUDA12__ZZPRIVATE_NT_MAX][4];
#pragma unroll
        for (uint32_t n = 0u; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        uint32_t raw[2][8];
        nvidia_cuda12__zzprivate_halves_row(lo, 0u, cols, t, raw[0]);
        nvidia_cuda12__zzprivate_halves_row(hi, 0u, cols, t, raw[1]);
        for (uint64_t k0 = 0u; k0 < cols; k0 += KT) {
            /* the tile's slice of `x`: pair `p`'s columns `k0 .. k0 + 63` a word at a time, past the pairs or the columns
             * zeros; the first wait is for the slice, the second keeps it until every warp has read it */
            for (uint32_t e = lane; e < TP * (KT / 2u); e += lanes) {
                const uint32_t p = e / (KT / 2u), c = e % (KT / 2u);
                const uint64_t col = k0 + 2u * c;
                ((uint32_t*)(void*)xs[p])[c] = p0 + p < pairs && col < cols
                    ? *(const uint32_t*)(const void*)(j->x + (uint64_t)pr[2u * (p0 + p)] * cols + col) : 0u;
            }
            __syncthreads();
            uint32_t now[2][8];
#pragma unroll
            for (uint32_t r = 0u; r < 2u; ++r)
#pragma unroll
                for (uint32_t q = 0u; q < 8u; ++q) now[r][q] = raw[r][q];
            if (k0 + KT < cols) {
                nvidia_cuda12__zzprivate_halves_row(lo, k0 + KT, cols, t, raw[0]);
                nvidia_cuda12__zzprivate_halves_row(hi, k0 + KT, cols, t, raw[1]);
            }
#pragma unroll
            for (uint32_t s2 = 0u; s2 < 2u; ++s2) {
                /* steps 2·s2 and 2·s2 + 1: the lane's columns `16t + 8·s2 ..` four a step, rows g and g + 8 */
                const uint32_t a0[4] = { now[0][4u * s2], now[1][4u * s2], now[0][4u * s2 + 1u], now[1][4u * s2 + 1u] };
                const uint32_t a1[4] = { now[0][4u * s2 + 2u], now[1][4u * s2 + 2u], now[0][4u * s2 + 3u], now[1][4u * s2 + 3u] };
#pragma unroll
                for (uint32_t n = 0u; n < NT; ++n) {
                    const uint4 q = *(const uint4*)(const void*)&xs[8u * n + g][16u * t + 8u * s2];
                    nvidia_cuda12__zzpackage_mma(acc[n], a0[0], a0[1], a0[2], a0[3], q.x, q.y);
                    nvidia_cuda12__zzpackage_mma(acc[n], a1[0], a1[1], a1[2], a1[3], q.z, q.w);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint64_t r = r0 + g + 8u * h;
            if (r >= out_rows) continue;
#pragma unroll
            for (uint32_t n = 0u; n < NT; ++n)
#pragma unroll
                for (uint32_t e = 0u; e < 2u; ++e) {
                    const uint64_t p = p0 + 8u * n + 2u * t + e;
                    if (p >= pairs) continue;
                    const uint64_t at = (uint64_t)pr[2u * p + 1u] * out_rows + r;
                    const float v = acc[n][2u * h + e];
                    if (j->out) (j->out + j->out_at[i])[at] = nn__kernels__zzabi_to_half(v, j->over);
                    else j->acc[at] += v * nn__silicon__half_to_float(j->w[pr[2u * p]]);
                }
        }
    }
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_BOUNDS void nvidia_cuda12__zzprivate_tile_64_16(nvidia_cuda12__tile_job j) {
    nvidia_cuda12__zzprivate_tile_halves(&j, 8u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_BOUNDS void nvidia_cuda12__zzprivate_tile_32_16(nvidia_cuda12__tile_job j) {
    nvidia_cuda12__zzprivate_tile_halves(&j, 4u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_BOUNDS void nvidia_cuda12__zzprivate_tile_16_16(nvidia_cuda12__tile_job j) {
    nvidia_cuda12__zzprivate_tile_halves(&j, 2u);
}

/* A tile job for `count` matrices, each checked as the generic body would — every one 4 or 8 bits over whole slices of 64
 * columns, codes on 8 bytes, or every one 16 bits over an even number of columns, on 4 bytes — false, and nothing launched,
 * when one does not fit, for the caller's fallback. The matrices of each width are one launch, on the tile its widest
 * pairs want. A token's few rows of 16 bits (16 pairs or fewer) stay nn's, as on AMD's family, where a tile made the 35B's
 * answer slower. */
static inline bool nvidia_cuda12__zzprivate_tiles(const nvidia_cuda12__tile_job* all) {
    if (all->count == 0u || all->count > NN__EXPERT__GROUPS_MAX || all->cols == 0u || ((uint64_t)(uintptr_t)all->x) % 4u != 0u)
        return false;
    const bool halves = all->d[0] == 16u;
    if (halves ? all->cols % 2u != 0u : all->cols % NVIDIA_CUDA12__ZZPRIVATE_KT != 0u) return false;
    uint64_t widest = 0u;
    for (uint64_t i = 0u; i < all->count; ++i) {
        if (halves ? all->d[i] != 16u || ((uint64_t)(uintptr_t)all->codes[i]) % 4u != 0u
                   : (all->d[i] != 4u && all->d[i] != 8u) || ((uint64_t)(uintptr_t)all->codes[i]) % 8u != 0u) return false;
        if (all->out_rows[i] == 0u) return false;
        if (all->pairs[i] > widest) widest = all->pairs[i];
    }
    if (halves && widest <= 16u) return false;
    for (uint64_t D = 4u; D <= 16u; D *= 2u) {
        nvidia_cuda12__tile_job j = *all;
        j.count = 0u;
        uint64_t most = 0u;
        for (uint64_t i = 0u; i < all->count; ++i) {
            if (all->d[i] != D) continue;
            const uint64_t k = j.count++;
            j.codes[k] = all->codes[i]; j.luts[k] = all->luts[i]; j.out_rows[k] = all->out_rows[i]; j.d[k] = D;
            j.out_at[k] = all->out_at[i]; j.pairs_at[k] = all->pairs_at[i]; j.pairs[k] = all->pairs[i];
            if (all->pairs[i] > most) most = all->pairs[i];
        }
        const uint64_t TP = most <= 16u ? 16u : most <= 32u ? 32u : 64u, TR = NVIDIA_CUDA12__ZZPRIVATE_ROWS;
        j.x16 = ((uint64_t)(uintptr_t)j.x) % 16u == 0u ? 1u : 0u;
        j.first[0] = 0u;
        for (uint64_t i = 0u; i < j.count; ++i)
            j.first[i + 1u] = j.first[i] + ((j.out_rows[i] + TR - 1u) / TR) * ((j.pairs[i] + TP - 1u) / TP);
        if (j.first[j.count] == 0u) continue;
        const uint32_t blocks = j.first[j.count] < NN__KERNELS__BLOCKS_MAX ? (uint32_t)j.first[j.count] : NN__KERNELS__BLOCKS_MAX;
        void* args[] = { &j };
        const size_t sizes[] = { sizeof j };
        const void* kernel = D == 4u ? (TP == 16u ? (const void*)nvidia_cuda12__zzprivate_tile_16_4
                                      : TP == 32u ? (const void*)nvidia_cuda12__zzprivate_tile_32_4 : (const void*)nvidia_cuda12__zzprivate_tile_64_4)
                           : D == 8u ? (TP == 16u ? (const void*)nvidia_cuda12__zzprivate_tile_16_8
                                      : TP == 32u ? (const void*)nvidia_cuda12__zzprivate_tile_32_8 : (const void*)nvidia_cuda12__zzprivate_tile_64_8)
                                     : (TP == 16u ? (const void*)nvidia_cuda12__zzprivate_tile_16_16
                                      : TP == 32u ? (const void*)nvidia_cuda12__zzprivate_tile_32_16 : (const void*)nvidia_cuda12__zzprivate_tile_64_16);
        nn__silicon__launch(kernel, "nvidia_cuda12__zzprivate_tile", blocks, NVIDIA_CUDA12__ZZPRIVATE_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP,
                            args, sizes, 1u);
    }
    return true;
}
/* One matrix's call as a tile job: the rows the room holds, as the generic body bounds them. */
static inline bool nvidia_cuda12__zzprivate_tiles_one(uint16_t* out, float* acc, const uint8_t* weights, uint64_t w_room,
                                                     const uint8_t* luts, uint64_t l_room, const uint16_t* x, const uint32_t* rows,
                                                     const uint16_t* w, uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols,
                                                     unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (count == 0u || row_bytes == 0u || out_rows > w_room / row_bytes || (d != 16u && out_rows > l_room / 2u)) return false;
    nvidia_cuda12__tile_job j = {};
    j.codes[0] = (uint64_t)(uintptr_t)weights; j.luts[0] = (uint64_t)(uintptr_t)luts; j.out_rows[0] = out_rows; j.d[0] = d;
    j.out_at[0] = 0u; j.pairs_at[0] = 0u; j.pairs[0] = count; j.count = 1u; j.cols = cols;
    j.out = out; j.acc = acc; j.x = x; j.rows = rows; j.w = w; j.over = over;
    return nvidia_cuda12__zzprivate_tiles(&j);
}

static void nvidia_cuda12__override_expert_rows(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room,
                                                const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t d, uint64_t out_rows,
                                                uint64_t cols, unsigned int* over) {
    if (nvidia_cuda12__zzprivate_tiles_one(out, 0, weights, w_room, luts, l_room, x, rows, 0, count, d, out_rows, cols, over)) return;
    nn__expert__zzabi_launch_rows(out, weights, w_room, luts, l_room, x, rows, count, d, out_rows, cols, over);
}

static void nvidia_cuda12__override_expert_rows_sum(float* acc, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, const uint32_t* rows, const uint16_t* w,
                                                    uint64_t count, uint64_t d, uint64_t out_rows, uint64_t cols) {
    if (nvidia_cuda12__zzprivate_tiles_one(0, acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols, 0)) return;
    nn__expert__zzabi_launch_rows_sum(acc, weights, w_room, luts, l_room, x, rows, w, count, d, out_rows, cols);
}

static void nvidia_cuda12__override_expert_groups(uint16_t* out, nn__expert__groups g, const uint16_t* x, const uint32_t* rows,
                                                  uint64_t cols, unsigned int* over) {
    nvidia_cuda12__tile_job j = {};
    for (uint64_t i = 0u; i < g.count && i < NN__EXPERT__GROUPS_MAX; ++i) {
        j.codes[i] = g.codes[i]; j.luts[i] = g.luts[i]; j.out_rows[i] = g.out_rows[i]; j.d[i] = g.d[i];
        j.out_at[i] = g.out_at[i]; j.pairs_at[i] = g.pairs_at[i]; j.pairs[i] = g.pairs[i];
    }
    j.count = g.count; j.cols = cols; j.out = out; j.acc = 0; j.x = x; j.rows = rows; j.w = 0; j.over = over;
    if (nvidia_cuda12__zzprivate_tiles(&j)) return;
    nn__expert__zzabi_launch_groups(out, g, x, rows, cols, over);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_EXPERT_CUH */
