#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_TURBOQUANT_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_TURBOQUANT_CUH
/* ══ turboquant's DOORS, ON THE TENSOR CORES — nvidia_cuda12 ══════════════════════════════════════════════════════════
 * nn's turboquant doors (`nn/cpu/opcodes/turboquant__abi.cuh`), where this family covers them: the gemv — one matrix,
 * several over one `x`, or several, each over its own `x`, weighted into a sum onto a residual. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ══ ⭐⭐ THE GEMV ON THE TENSOR CORES ══════════════════════════════════════════════════════════════════════════════════
 * nn's definitions (`nn/gpu/kernels/kernels.cuh`): `out[r] = half(scale_r · Σ_c level(code[r][c]) · x[c])`, over each
 * matrix of a group; and the sum's `out[r] = half(residual[r] + Σ_i w[out_at[i]] · scale_i,r · (W_i[r] · x_i))`, `x_i`
 * at `x + i · cols`. A gemv reads every code once and multiplies it once, so it is the codes' bytes that set its speed,
 * and what a lane spends on each code decides whether it keeps up with them. Here the multiply is `mma.sync`'s:
 *   · a WARP takes 16 rows of a matrix; each lane loads its two rows' codes straight into registers — 32 bytes a row a
 *     slice, four lanes a row's 128 bytes — and decodes them into A operands as `expert.cuh` does: a register, two
 *     levels, from one lookup in nn's level table staged in shared memory. `x` is the B operand, the same in all eight of
 *     its columns, read by every warp of the card from the cache it shares
 *   · so a code costs half a lookup and its shifts; the products and their sum are the tensor cores' — where a lane's own
 *     multiply-adds would also convert every level and every `x` to a float
 *   · the columns in `expert.cuh`'s order, widened: a slice is 256 columns at 4 bits and 128 at 8, and lane `t` of four
 *     takes its quarter of them in one run, four columns a step, both operands alike
 *   · a tile's codes loaded two slices ahead of its products; the sum in fp32, rounded once, as nn's
 * Covers 4 and 8 bits over whole slices, codes and `x` on sixteen bytes; anything else is nn's door. The products are
 * nn's and the order they are summed in is not.
 * ⛳ THE ONE BARRIER: the block waits once, after staging nn's levels and before its first row — the category of
 *   amd_rocm6_wave64's row kernels (`groups_rows`, `scripts/claim_rules/no_rendezvous.py`). */
#define NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS 8u
#define NVIDIA_CUDA12__ZZPRIVATE_GEMV_BOUNDS __launch_bounds__(256)

typedef struct nvidia_cuda12__gemv_job {
    uint64_t codes[NN__TURBOQUANT__GROUPS_MAX], luts[NN__TURBOQUANT__GROUPS_MAX], rows[NN__TURBOQUANT__GROUPS_MAX];
    uint64_t d[NN__TURBOQUANT__GROUPS_MAX], out_at[NN__TURBOQUANT__GROUPS_MAX], first[NN__TURBOQUANT__GROUPS_MAX + 1];
    uint64_t count, cols, sum;                      /* `sum`: each matrix over its own `x`, weighted onto `residual` */
    uint16_t* out; const uint16_t* x; const uint16_t* w; const uint16_t* residual; unsigned int* over;
} nvidia_cuda12__gemv_job;

/* One lane's 32 bytes of a row's codes at byte `at` — two 16-byte loads; zeros for a row past the matrix (`row` null). */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_codes32(const uint8_t* row, uint64_t at, uint32_t c[2][8], uint32_t r) {
    if (!row) { for (uint32_t i = 0u; i < 8u; ++i) c[r][i] = 0u; return; }
    const uint4 a = ((const uint4*)(const void*)(row + at))[0], b = ((const uint4*)(const void*)(row + at))[1];
    c[r][0] = a.x; c[r][1] = a.y; c[r][2] = a.z; c[r][3] = a.w; c[r][4] = b.x; c[r][5] = b.y; c[r][6] = b.z; c[r][7] = b.w;
}

/* A row's scale, its two bytes. */
static __device__ __forceinline__ float nvidia_cuda12__zzprivate_scale(const uint8_t* lut, uint64_t r) {
    return nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
}

/* Where a tile's walk is: matrix `m`, its slice from column `k`, and the rows' codes there. A slice is `64 · U` columns,
 * U = 4 at 4 bits and 2 at 8, and lane `t` of four takes its columns `16U · t ..` — 32 bytes of each row's codes. */
typedef struct nvidia_cuda12__gemv_walk { uint64_t m, k; } nvidia_cuda12__gemv_walk;
static __device__ __forceinline__ uint64_t nvidia_cuda12__zzprivate_slice(uint64_t d) { return d == 4u ? 256u : 128u; }
/* The codes of rows `g` and `g + 8` of tile `r0` at `w`, or zeros once the walk is past its last matrix `end`. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_fetch(const nvidia_cuda12__gemv_job* j, nvidia_cuda12__gemv_walk w,
                                                                       uint64_t end, uint64_t r0, uint32_t g, uint32_t t, uint32_t c[2][8]) {
    const uint64_t m = w.m < end ? w.m : end - 1u, D = j->d[m], row_bytes = j->cols * D / 8u;
    const uint8_t* codes = (const uint8_t*)(uintptr_t)j->codes[m];
    const uint64_t at = w.k * D / 8u + 32u * t;
    nvidia_cuda12__zzprivate_codes32(w.m < end && r0 + g < j->rows[m] ? codes + (r0 + g) * row_bytes : 0, at, c, 0u);
    nvidia_cuda12__zzprivate_codes32(w.m < end && r0 + g + 8u < j->rows[m] ? codes + (r0 + g + 8u) * row_bytes : 0, at, c, 1u);
}
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_advance(const nvidia_cuda12__gemv_job* j, nvidia_cuda12__gemv_walk* w) {
    w->k += nvidia_cuda12__zzprivate_slice(j->d[w->m]);
    if (w->k >= j->cols) { w->k = 0u; w->m += 1u; }
}
/* One slice's products into `acc`, at width `D`: each step's A operands decoded from the codes, `x`'s 16U columns the B. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_slice_mma(const uint32_t D, const uint32_t c[2][8], const uint16_t* x,
                                                                           uint32_t t, const uint32_t* pair4, const uint16_t* lev8,
                                                                           float acc[4]) {
    const uint32_t U = D == 4u ? 4u : 2u, W = 8u * U;
    uint32_t xw[32];
    const uint4* xs = (const uint4*)(const void*)(x + 16u * U * t);
#pragma unroll
    for (uint32_t q = 0u; q < W / 4u; ++q) {
        const uint4 v = xs[q];
        xw[4u * q] = v.x; xw[4u * q + 1u] = v.y; xw[4u * q + 2u] = v.z; xw[4u * q + 3u] = v.w;
    }
#pragma unroll
    for (uint32_t s = 0u; s < 4u * U; ++s) {
        uint32_t a[4];
#pragma unroll
        for (uint32_t r = 0u; r < 2u; ++r) {
            if (D == 4u) {
                const uint32_t word = c[r][s / 2u], sh = 16u * (s % 2u);
                a[r]      = pair4[(word >> sh) & 0xFFu];
                a[2u + r] = pair4[(word >> (sh + 8u)) & 0xFFu];
            } else {
                const uint32_t word = c[r][s];
                a[r]      = (uint32_t)lev8[word & 0xFFu] | ((uint32_t)lev8[(word >> 8) & 0xFFu] << 16);
                a[2u + r] = (uint32_t)lev8[(word >> 16) & 0xFFu] | ((uint32_t)lev8[word >> 24] << 16);
            }
        }
        nvidia_cuda12__zzpackage_mma(acc, a[0], a[1], a[2], a[3], xw[2u * s], xw[2u * s + 1u]);
    }
}

/* The body: nn's levels staged once, then each warp its tiles of 16 rows — a matrix's, into its own outputs, or every
 * matrix's at those rows, weighted into one sum onto the residual. A tile walks its matrices' slices in one sequence and
 * loads each slice's codes two slices ahead of its products, across a matrix's end as within one: a downs' sum of short
 * matrices keeps as many bytes in flight as one long matrix. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_groups_rows(const nvidia_cuda12__gemv_job* j) {
    __shared__ uint32_t pair4[256];
    __shared__ uint16_t lev8[256];
    const uint32_t lane = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    for (uint32_t e = lane; e < 256u; e += lanes) {
        const uint16_t* lev4 = nn__turboquant__zzabi_levels(4u);
        pair4[e] = (uint32_t)lev4[e & 15u] | ((uint32_t)lev4[e >> 4] << 16);
        lev8[e] = nn__turboquant__zzabi_levels(8u)[e];
    }
    __syncthreads();
    const uint32_t l = lane % NVIDIA_CUDA12__ZZPACKAGE_WARP, g = l / 4u, t = l % 4u, warps = lanes / NVIDIA_CUDA12__ZZPACKAGE_WARP;
    for (uint64_t it = (uint64_t)nn__silicon__block() * warps + lane / NVIDIA_CUDA12__ZZPACKAGE_WARP;
         it < j->first[j->count]; it += (uint64_t)nn__silicon__blocks() * warps) {
        uint64_t i = 0u;
        while (it >= j->first[i + 1u]) ++i;
        const uint64_t r0 = (it - j->first[i]) * 16u, begin = j->sum ? 0u : i, end = j->sum ? j->count : i + 1u;
        float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f }, part[2] = { 0.0f, 0.0f };
        uint32_t c0[2][8], c1[2][8], c2[2][8];
        nvidia_cuda12__gemv_walk now = { begin, 0u }, ahead = now;
        nvidia_cuda12__zzprivate_fetch(j, ahead, end, r0, g, t, c0);
        nvidia_cuda12__zzprivate_advance(j, &ahead);
        nvidia_cuda12__zzprivate_fetch(j, ahead, end, r0, g, t, c1);
        if (ahead.m < end) nvidia_cuda12__zzprivate_advance(j, &ahead);
        while (now.m < end) {
            nvidia_cuda12__zzprivate_fetch(j, ahead, end, r0, g, t, c2);
            if (ahead.m < end) nvidia_cuda12__zzprivate_advance(j, &ahead);
            const uint64_t m = now.m;
            const uint16_t* xm = (j->sum ? j->x + m * j->cols : j->x) + now.k;
            if (j->d[m] == 4u) nvidia_cuda12__zzprivate_slice_mma(4u, c0, xm, t, pair4, lev8, acc);
            else               nvidia_cuda12__zzprivate_slice_mma(8u, c0, xm, t, pair4, lev8, acc);
            nvidia_cuda12__zzprivate_advance(j, &now);
            if (now.m != m) {                                  /* the matrix is done: its rows' answers */
                const uint8_t* lut = (const uint8_t*)(uintptr_t)j->luts[m];
#pragma unroll
                for (uint32_t h = 0u; h < 2u; ++h) {
                    const uint64_t r = r0 + g + 8u * h;
                    if (r >= j->rows[m] || t != 0u) continue;
                    const float dot = acc[2u * h], scale = nvidia_cuda12__zzprivate_scale(lut, r);
                    if (j->sum) part[h] += dot * scale * nn__silicon__half_to_float(j->w[j->out_at[m]]);
                    else        j->out[j->out_at[m] + r] = nn__kernels__zzabi_to_half(dot * scale, j->over);
                }
                acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
            }
#pragma unroll
            for (uint32_t r = 0u; r < 2u; ++r)
#pragma unroll
                for (uint32_t q = 0u; q < 8u; ++q) { c0[r][q] = c1[r][q]; c1[r][q] = c2[r][q]; }
        }
        if (j->sum && t == 0u)
#pragma unroll
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint64_t r = r0 + g + 8u * h;
                if (r < j->rows[0]) j->out[r] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(j->residual[r]) + part[h], j->over);
            }
    }
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_GEMV_BOUNDS void nvidia_cuda12__zzprivate_gemv(nvidia_cuda12__gemv_job j) {
    nvidia_cuda12__zzprivate_groups_rows(&j);
}

/* ⛳ WARPS A BLOCK FROM THE CARD: eight unless the tiles are too few to spread. A matrix of few long rows — the 27B's MLP
 *   down, 5,120 rows of 17,408 columns — is 320 tiles, 40 blocks of eight on a card of 36 SMs: four SMs ran two blocks
 *   and the launch lasted as long as they did (`MEASURED`: 279 GB/s against the gate + up's 373, 272 blocks). So a
 *   block's warps halve while the blocks would be fewer than four an SM, down to one; each warp still takes its 16 rows
 *   the same way, so the answer is the same bytes. The SM count is the bound card's, asked once a card. */
static inline uint32_t nvidia_cuda12__zzprivate_gemv_warps(uint64_t tiles) {
    static int sms[64];
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) return NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS;
    if (sms[d] == 0 && cudaDeviceGetAttribute(&sms[d], cudaDevAttrMultiProcessorCount, d) != cudaSuccess) sms[d] = 0;
    uint32_t warps = NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS;
    while (warps > 1u && (tiles + warps - 1u) / warps < 4u * (uint64_t)(sms[d] > 0 ? sms[d] : 0)) warps /= 2u;
    return warps;
}

/* A job each matrix checked as nn's body would — 4 or 8 bits, whole slices, codes and `x` on sixteen bytes, the rows the
 * room holds — and launched; false, and nothing launched, when one does not fit. In a sum every matrix has the sum's
 * rows, so its tiles are those rows' and each tile walks every matrix. */
static inline bool nvidia_cuda12__zzprivate_gemvs(nvidia_cuda12__gemv_job* j) {
    if (j->count == 0u || j->count > NN__TURBOQUANT__GROUPS_MAX || j->cols == 0u || ((uint64_t)(uintptr_t)j->x) % 16u != 0u)
        return false;
    for (uint64_t i = 0u; i < j->count; ++i) {
        const uint64_t D = j->d[i];
        if ((D != 4u && D != 8u) || j->cols % (D == 4u ? 256u : 128u) != 0u || ((uint64_t)(uintptr_t)j->codes[i]) % 16u != 0u) return false;
        if (j->sum && j->rows[i] != j->rows[0]) return false;
    }
    j->first[0] = 0u;
    if (j->sum) {
        j->first[1] = (j->rows[0] + 15u) / 16u;
        for (uint64_t i = 1u; i < j->count; ++i) j->first[i + 1u] = j->first[1];   /* every tile is the first matrix's */
    } else {
        for (uint64_t i = 0u; i < j->count; ++i) j->first[i + 1u] = j->first[i] + (j->rows[i] + 15u) / 16u;
    }
    const uint64_t tiles = j->first[j->count];
    if (tiles == 0u) return true;
    const uint32_t warps = nvidia_cuda12__zzprivate_gemv_warps(tiles);
    const uint64_t want = (tiles + warps - 1u) / warps;
    const uint32_t blocks = want < 0x7FFFFFFFull ? (uint32_t)want : 0x7FFFFFFFu;
    void* args[] = { j };
    const size_t sizes[] = { sizeof *j };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_gemv, "nvidia_cuda12__zzprivate_gemv", blocks,
                        warps * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 1u);
    return true;
}

static void nvidia_cuda12__override_turboquant_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                    unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (row_bytes != 0u && rows <= w_room / row_bytes && rows <= l_room / 2u) {
        nvidia_cuda12__gemv_job j = {};
        j.codes[0] = (uint64_t)(uintptr_t)weights; j.luts[0] = (uint64_t)(uintptr_t)luts; j.rows[0] = rows; j.d[0] = d; j.out_at[0] = 0u;
        j.count = 1u; j.cols = cols; j.sum = 0u; j.out = out; j.x = x; j.w = 0; j.residual = 0; j.over = over;
        if (nvidia_cuda12__zzprivate_gemvs(&j)) return;
    }
    nn__turboquant__zzabi_launch_gemv(out, weights, w_room, luts, l_room, x, d, rows, cols, over);
}

static void nvidia_cuda12__override_turboquant_gemv_groups(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                           unsigned int* over) {
    nvidia_cuda12__gemv_job j = {};
    for (uint64_t i = 0u; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) {
        j.codes[i] = g.codes[i]; j.luts[i] = g.luts[i]; j.rows[i] = g.rows[i]; j.d[i] = g.d[i]; j.out_at[i] = g.out_at[i];
    }
    j.count = g.count; j.cols = cols; j.sum = 0u; j.out = out; j.x = x; j.w = 0; j.residual = 0; j.over = over;
    if (nvidia_cuda12__zzprivate_gemvs(&j)) return;
    nn__turboquant__zzabi_launch_gemv_groups(out, g, x, cols, over);
}

static void nvidia_cuda12__override_turboquant_gemv_groups_sum(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                               const uint16_t* residual, uint64_t rows, uint64_t cols, unsigned int* over) {
    nvidia_cuda12__gemv_job j = {};
    for (uint64_t i = 0u; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) {
        j.codes[i] = g.codes[i]; j.luts[i] = g.luts[i]; j.rows[i] = rows; j.d[i] = g.d[i]; j.out_at[i] = g.out_at[i];
    }
    j.count = g.count; j.cols = cols; j.sum = 1u; j.out = out; j.x = x; j.w = w; j.residual = residual; j.over = over;
    if (nvidia_cuda12__zzprivate_gemvs(&j)) return;
    nn__turboquant__zzabi_launch_gemv_groups_sum(out, g, x, w, residual, rows, cols, over);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_TURBOQUANT_CUH */
