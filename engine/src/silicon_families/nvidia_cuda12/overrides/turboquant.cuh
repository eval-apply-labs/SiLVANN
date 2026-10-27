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
 *   · a matrix of few tiles has its columns split between `split` warps of a block, each a run of whole slices, their
 *     sums added in shared memory in the warps' order — so a short matrix keeps the card's warps busy
 * Covers 4 and 8 bits over whole slices, codes and `x` on sixteen bytes; anything else is nn's door. The products are
 * nn's and the order they are summed in is not; it is a function of the shapes and the card, the same every call.
 * ⛳ THE BARRIERS: the block waits once after staging nn's levels, and where the columns are split, once before a tile's
 *   parts are added and once after — the category of amd_rocm6_wave64's row kernels (`groups_rows`,
 *   `scripts/claim_rules/no_rendezvous.py`); each is a block-wide pass every warp reaches. */
#define NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS 8u
#define NVIDIA_CUDA12__ZZPRIVATE_GEMV_BOUNDS __launch_bounds__(256)

typedef struct nvidia_cuda12__gemv_job {
    uint64_t codes[NN__TURBOQUANT__GROUPS_MAX], luts[NN__TURBOQUANT__GROUPS_MAX], rows[NN__TURBOQUANT__GROUPS_MAX];
    uint64_t d[NN__TURBOQUANT__GROUPS_MAX], out_at[NN__TURBOQUANT__GROUPS_MAX], first[NN__TURBOQUANT__GROUPS_MAX + 1];
    uint64_t count, cols, sum;                      /* `sum`: each matrix over its own `x`, weighted onto `residual` */
    uint64_t split;                                 /* warps a tile's columns are split between; 1 in a sum */
    uint16_t* out; const uint16_t* x; const uint16_t* w; const uint16_t* residual; unsigned int* over;
} nvidia_cuda12__gemv_job;

/* One lane's 32 bytes of a row's codes at byte `at` — two 16-byte loads; zeros for a row past the matrix (`row` null). */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_codes32(const uint8_t* row, uint64_t at, uint32_t c[2][8], uint32_t r) {
    if (!row) { for (uint32_t i = 0u; i < 8u; ++i) c[r][i] = 0u; return; }
    const uint4 a = __ldcs((const uint4*)(const void*)(row + at)), b = __ldcs((const uint4*)(const void*)(row + at) + 1);
    c[r][0] = a.x; c[r][1] = a.y; c[r][2] = a.z; c[r][3] = a.w; c[r][4] = b.x; c[r][5] = b.y; c[r][6] = b.z; c[r][7] = b.w;
}

/* A row's scale, its two bytes. */
static __device__ __forceinline__ float nvidia_cuda12__zzprivate_scale(const uint8_t* lut, uint64_t r) {
    return nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
}

/* Where a tile's walk is: matrix `m`, its slice from column `k`, and the rows' codes there. A slice is `64 · U` columns,
 * U = 4 at 4 bits and 2 at 8, and lane `t` of four takes its columns `16U · t ..` — 32 bytes of each row's codes. A walk
 * runs until it reaches `stop`. */
typedef struct nvidia_cuda12__gemv_walk { uint64_t m, k; } nvidia_cuda12__gemv_walk;
static __device__ __forceinline__ uint64_t nvidia_cuda12__zzprivate_slice(uint64_t d) { return d == 4u ? 256u : 128u; }
static __device__ __forceinline__ bool nvidia_cuda12__zzprivate_before(nvidia_cuda12__gemv_walk w, nvidia_cuda12__gemv_walk stop) {
    return w.m < stop.m || (w.m == stop.m && w.k < stop.k);
}
/* The codes of rows `g` and `g + 8` of tile `r0` at `w`, or zeros once the walk has reached `stop`. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_fetch(const nvidia_cuda12__gemv_job* j, nvidia_cuda12__gemv_walk w,
                                                                       nvidia_cuda12__gemv_walk stop, uint64_t r0, uint32_t g, uint32_t t,
                                                                       uint32_t c[2][8]) {
    const bool on = nvidia_cuda12__zzprivate_before(w, stop);
    const uint64_t m = w.m < j->count ? w.m : j->count - 1u, D = j->d[m], row_bytes = j->cols * D / 8u;
    const uint8_t* codes = (const uint8_t*)(uintptr_t)j->codes[m];
    const uint64_t at = w.k * D / 8u + 32u * t;
    nvidia_cuda12__zzprivate_codes32(on && r0 + g < j->rows[m] ? codes + (r0 + g) * row_bytes : 0, at, c, 0u);
    nvidia_cuda12__zzprivate_codes32(on && r0 + g + 8u < j->rows[m] ? codes + (r0 + g + 8u) * row_bytes : 0, at, c, 1u);
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

/* The body: each warp its tiles of 16 rows — a matrix's, into its own outputs, or every matrix's at those rows, weighted
 * into one sum onto the residual — or its part of a tile's columns, `split` warps a tile. A tile walks its slices in one
 * sequence and loads each slice's codes two slices ahead of its products, across a matrix's end as within one: a downs'
 * sum of short matrices keeps as many bytes in flight as one long matrix. A block's first codes are asked for before nn's
 * levels are staged, so the staging waits under their load. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_groups_rows(const nvidia_cuda12__gemv_job* j) {
    __shared__ uint32_t pair4[256];
    __shared__ uint16_t lev8[256];
    __shared__ float parts[NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS][16];
    const uint32_t lane = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    const uint32_t l = lane % NVIDIA_CUDA12__ZZPACKAGE_WARP, g = l / 4u, t = l % 4u, warps = lanes / NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint32_t warp = lane / NVIDIA_CUDA12__ZZPACKAGE_WARP, S = (uint32_t)j->split, per = warps / S, p = warp % S;
    const uint64_t tiles = j->first[j->count];
    bool staged = false;
    for (uint64_t base = (uint64_t)nn__silicon__block() * per; base < tiles; base += (uint64_t)nn__silicon__blocks() * per) {
        const uint64_t it = base + warp / S;
        const bool live = it < tiles;
        uint64_t i = 0u;
        while (live && it >= j->first[i + 1u]) ++i;
        const uint64_t r0 = live ? (it - j->first[i]) * 16u : 0u;
        nvidia_cuda12__gemv_walk now = { 0u, 0u }, stop = { 0u, 0u };
        if (live && j->sum) stop.m = j->count;
        else if (live) {                                   /* part `p` of the matrix's slices */
            /* in shifts: a slice is 2^7 or 2^8 columns and the split a power of two */
            const uint32_t ls = j->d[i] == 4u ? 8u : 7u, lS = (uint32_t)__ffs((int)S) - 1u;
            const uint32_t n = (uint32_t)(j->cols >> ls);
            now.m = i; now.k = (uint64_t)((n * p) >> lS) << ls;
            const uint64_t k1 = (uint64_t)((n * (p + 1u)) >> lS) << ls;
            if (k1 < j->cols) { stop.m = i; stop.k = k1; } else stop.m = i + 1u;
        }
        float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f }, part[2] = { 0.0f, 0.0f };
        uint32_t c0[2][8], c1[2][8], c2[2][8];
        nvidia_cuda12__gemv_walk ahead = now;
        nvidia_cuda12__zzprivate_fetch(j, ahead, stop, r0, g, t, c0);
        if (nvidia_cuda12__zzprivate_before(ahead, stop)) nvidia_cuda12__zzprivate_advance(j, &ahead);
        nvidia_cuda12__zzprivate_fetch(j, ahead, stop, r0, g, t, c1);
        if (nvidia_cuda12__zzprivate_before(ahead, stop)) nvidia_cuda12__zzprivate_advance(j, &ahead);
        if (!staged) {
            for (uint32_t e = lane; e < 256u; e += lanes) {
                const uint16_t* lev4 = nn__turboquant__zzabi_levels(4u);
                pair4[e] = (uint32_t)lev4[e & 15u] | ((uint32_t)lev4[e >> 4] << 16);
                lev8[e] = nn__turboquant__zzabi_levels(8u)[e];
            }
            __syncthreads();
            staged = true;
        }
        while (nvidia_cuda12__zzprivate_before(now, stop)) {
            nvidia_cuda12__zzprivate_fetch(j, ahead, stop, r0, g, t, c2);
            if (nvidia_cuda12__zzprivate_before(ahead, stop)) nvidia_cuda12__zzprivate_advance(j, &ahead);
            const uint64_t m = now.m;
            const uint16_t* xm = (j->sum ? j->x + m * j->cols : j->x) + now.k;
            if (j->d[m] == 4u) nvidia_cuda12__zzprivate_slice_mma(4u, c0, xm, t, pair4, lev8, acc);
            else               nvidia_cuda12__zzprivate_slice_mma(8u, c0, xm, t, pair4, lev8, acc);
            nvidia_cuda12__zzprivate_advance(j, &now);
            if (j->sum && now.m != m) {                        /* a matrix of the sum is done: its share */
                const uint8_t* lut = (const uint8_t*)(uintptr_t)j->luts[m];
#pragma unroll
                for (uint32_t h = 0u; h < 2u; ++h) {
                    const uint64_t r = r0 + g + 8u * h;
                    if (r < j->rows[m] && t == 0u) part[h] += acc[2u * h] * nvidia_cuda12__zzprivate_scale(lut, r) * nn__silicon__half_to_float(j->w[j->out_at[m]]);
                }
                acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
            }
#pragma unroll
            for (uint32_t r = 0u; r < 2u; ++r)
#pragma unroll
                for (uint32_t q = 0u; q < 8u; ++q) { c0[r][q] = c1[r][q]; c1[r][q] = c2[r][q]; }
        }
        if (j->sum) {
            if (live && t == 0u)
#pragma unroll
                for (uint32_t h = 0u; h < 2u; ++h) {
                    const uint64_t r = r0 + g + 8u * h;
                    if (r < j->rows[0]) j->out[r] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(j->residual[r]) + part[h], j->over);
                }
            continue;
        }
        if (S > 1u) {                                      /* the tile's parts, added in the warps' order */
            if (t == 0u) { parts[warp][g] = acc[0]; parts[warp][g + 8u] = acc[2]; }
            __syncthreads();
            if (p == 0u && t == 0u)
                for (uint32_t h = 0u; h < 2u; ++h) {
                    float dot = parts[warp][g + 8u * h];
                    for (uint32_t q = 1u; q < S; ++q) dot += parts[warp + q][g + 8u * h];
                    acc[2u * h] = dot;
                }
            __syncthreads();
        }
        if (live && p == 0u && t == 0u) {                  /* the tile's rows' answers */
            const uint8_t* lut = (const uint8_t*)(uintptr_t)j->luts[i];
#pragma unroll
            for (uint32_t h = 0u; h < 2u; ++h) {
                const uint64_t r = r0 + g + 8u * h;
                if (r < j->rows[i]) j->out[j->out_at[i] + r] = nn__kernels__zzabi_to_half(acc[2u * h] * nvidia_cuda12__zzprivate_scale(lut, r), j->over);
            }
        }
    }
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_GEMV_BOUNDS void nvidia_cuda12__zzprivate_gemv(nvidia_cuda12__gemv_job j) {
    nvidia_cuda12__zzprivate_groups_rows(&j);
}

/* The bound card's SM count and the kernel's warps resident on one, asked once a card; 0 where they cannot be asked. */
static inline uint64_t nvidia_cuda12__zzprivate_gemv_sms(uint64_t* resident) {
    static int sms[64], blocks[64];
    int d = 0;
    *resident = 0u;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) return 0u;
    if (sms[d] == 0 && cudaDeviceGetAttribute(&sms[d], cudaDevAttrMultiProcessorCount, d) != cudaSuccess) sms[d] = 0;
    if (blocks[d] == 0 && cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks[d], nvidia_cuda12__zzprivate_gemv,
                              NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, 0) != cudaSuccess) blocks[d] = 0;
    *resident = blocks[d] > 0 ? (uint64_t)blocks[d] * NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS : 0u;
    return sms[d] > 0 ? (uint64_t)sms[d] : 0u;
}

/* A job each matrix checked as nn's body would — 4 or 8 bits, whole slices, codes and `x` on sixteen bytes, the rows the
 * room holds — and launched; false, and nothing launched, when one does not fit. In a sum every matrix has the sum's
 * rows, so its tiles are those rows' and each tile walks every matrix.
 * ⛳ THE SPLIT AND THE WARPS A BLOCK FROM THE CARD. A matrix of 3,072 rows is 192 tiles, 5 warps an SM of the 5060 Ti's
 *   36: each walks every column alone, and the card idles between their loads (`MEASURED`, Nsight Compute, 2026-10-25:
 *   0.3 warps ready a scheduler; the 122B's shared expert down at 4 bits 34% of the DRAM peak) — and a matrix of 1,284
 *   tiles is 2.2 waves of the warps resident at once, its last wave a quarter full. So outside a sum a tile's columns
 *   are split between up to eight warps, each part at least a slice, while doubling the parts fills their waves more
 *   than a tenth better — a wave the warps resident on every SM at once, from the runtime's occupancy at the kernel's
 *   registers, so the split follows the card it runs on — and within one wave in two at most: there more parts only
 *   shorten each warp's walk (`MEASURED`, Nsight Compute: the 122B's shared expert at 128 and 192 tiles fastest in
 *   two, four 4-30% slower). Then a block's warps halve while the blocks
 *   would be fewer than four an SM, down to the split: a matrix of few long rows — the 27B's MLP down, 5,120 rows of
 *   17,408 columns — is 320 tiles, 40 blocks of eight on a card of 36 SMs, and four SMs ran two blocks and the launch
 *   lasted as long as they did (`MEASURED`: 279 GB/s against the gate + up's 373, 272 blocks). A tile's split and so
 *   its answer's bytes follow the shapes and the card, never the call. */
static inline bool nvidia_cuda12__zzprivate_gemvs(nvidia_cuda12__gemv_job* j) {
    if (j->count == 0u || j->count > NN__TURBOQUANT__GROUPS_MAX || j->cols == 0u || ((uint64_t)(uintptr_t)j->x) % 16u != 0u)
        return false;
    uint64_t fewest = ~0ull;                               /* the fewest slices a row of any matrix */
    for (uint64_t i = 0u; i < j->count; ++i) {
        const uint64_t D = j->d[i];
        if ((D != 4u && D != 8u) || j->cols % (D == 4u ? 256u : 128u) != 0u || ((uint64_t)(uintptr_t)j->codes[i]) % 16u != 0u) return false;
        if (j->sum && j->rows[i] != j->rows[0]) return false;
        const uint64_t n = j->cols / (D == 4u ? 256u : 128u);
        if (n < fewest) fewest = n;
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
    uint64_t resident = 0u;
    const uint64_t sms = nvidia_cuda12__zzprivate_gemv_sms(&resident);
    uint32_t split = 1u;
    const uint64_t wave = resident * sms;
    while (!j->sum && wave != 0u && split < NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS && 2u * split <= fewest) {
        /* the waves' fill, parts over the room their waves have: `now / (waves(now) · wave)`, against the doubled */
        const uint64_t now = tiles * split, next = 2u * now;
        const uint64_t waves = (now + wave - 1u) / wave, waves2 = (next + wave - 1u) / wave;
        if (10u * next * waves <= 11u * now * waves2) break;           /* less than a tenth better */
        if (split >= 2u && next <= wave) break;                        /* within one wave, two parts at most */
        split *= 2u;
    }
    uint32_t warps = NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS;
    while (warps > split && (tiles * split + warps - 1u) / warps < 4u * sms) warps /= 2u;
    j->split = split;
    const uint64_t per = warps / split, want = (tiles + per - 1u) / per;
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
