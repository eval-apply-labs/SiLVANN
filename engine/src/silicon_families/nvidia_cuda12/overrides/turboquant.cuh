#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_TURBOQUANT_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_TURBOQUANT_CUH
/* ══ turboquant's DOORS, ON THE TENSOR CORES — nvidia_cuda12 ══════════════════════════════════════════════════════════
 * nn's turboquant doors (`nn/cpu/opcodes/turboquant__abi.cuh`), where this family covers them: the gemv — one matrix,
 * several over one `x` or over several inputs, or several, each over its own `x`, weighted into a sum onto a residual —
 * the int8 gemv at 4 bits, and a router's matrix of halves. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ══ ⭐⭐ THE GEMV ON THE TENSOR CORES ══════════════════════════════════════════════════════════════════════════════════
 * nn's definitions (`nn/gpu/kernels/kernels.cuh`): `out[r] = half(scale_r · Σ_c level(code[r][c]) · x[c])`, over each
 * matrix of a group; and the sum's `out[r] = half(residual[r] + Σ_i w[out_at[i]] · scale_i,r · (W_i[r] · x_i))`, `x_i`
 * at `x + i · cols`. A gemv reads every code once and multiplies it once, so it is the codes' bytes that set its speed,
 * and what a lane spends on each code decides whether it keeps up with them. Here the multiply is `mma.sync`'s:
 *   · a WARP takes 16 rows of a matrix; each lane loads its two rows' codes straight into registers — 32 bytes a row a
 *     slice, four lanes a row's 128 bytes — and decodes them into A operands as `expert.cuh` does: at 4 bits a register,
 *     two levels, by byte permutes over nn's levels held in registers (▶ kernels.cuh's `levels4_pairs`), at 8 bits from
 *     nn's level table staged in shared memory. `x` is the B operand, the same in all eight of its columns, read by every
 *     warp of the card from the cache it shares
 *   · so eight 4-bit codes cost sixteen byte permutes; the products and their sum are the tensor cores' — where a lane's
 *     own multiply-adds would also convert every level and every `x` to a float
 *   · the columns in `expert.cuh`'s order, widened: a slice is 256 columns at 4 bits and 128 at 8, and lane `t` of four
 *     takes its quarter of them in one run, four columns a step, both operands alike
 *   · a tile's codes loaded two slices ahead of its products; the sum in fp32, rounded once, as nn's
 *   · a matrix of few tiles has its columns split between `split` warps of a block, each a run of whole slices, their
 *     sums added in shared memory in the warps' order — so a short matrix keeps the card's warps busy; a sum's tiles the
 *     same, on a kernel of their own (▶ `groups_sum_rows`)
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
    /* A group whose matrices take splits of their own (▶ `gemvs`): `classes` runs of matrices, one split each, class `c`
     * the tiles `class_tile[c]` .. `class_tile[c + 1]` and the blocks `class_block[c]` .. `class_block[c + 1]`; 0 where
     * the whole job takes `split`. */
    uint64_t classes, class_split[4], class_tile[5], class_block[5];
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
/* ⭐ THE 4-BIT LEVELS BY THEIR MIRROR (NN-168) — where nn's sixteen 4-bit levels are eight magnitudes and their
 *   negatives, level(15 - k) level(k) with the sign bit flipped, a code `k`'s level is magnitude `j` — k ^ 7 where its
 *   top bit is clear, k & 7 where set — from a table of eight: four codes' low bytes one byte permute, their high bytes
 *   another, the sign their top bit from a sign-replicating permute, then the halves interleaved as `levels4_pairs`
 *   does — ten permutes for eight codes where the sixteen-entry lookup takes sixteen (`MEASURED` an A100: the 4-bit
 *   GEMVs 4-9% faster cold, the 27B's decode 39.4 → 41.3 tokens/s, measurements/2026-11-10_nn168_a100.md, each output
 *   the sixteen-entry lookup's to the bit). Today's table is mirrored
 *   (`scripts/nn_gaussian_tables.py`: an even count of quantiles of a symmetric density), but nothing promises it, so
 *   the kernel checks it once and decodes as `levels4_pairs` where it is not. */
static __device__ __forceinline__ bool nvidia_cuda12__zzprivate_levels4_mirror(uint32_t ml[2], uint32_t mh[2]) {
    const uint16_t* lev4 = nn__turboquant__zzabi_levels(4u);
    bool mirrored = true;
#pragma unroll
    for (uint32_t w = 0u; w < 2u; ++w) {
        ml[w] = mh[w] = 0u;
#pragma unroll
        for (uint32_t b = 0u; b < 4u; ++b) {
            const uint32_t k = 8u + 4u * w + b;                /* magnitude 4w + b is level 8 + 4w + b */
            ml[w] |= (uint32_t)(lev4[k] & 0xFFu) << (8u * b);
            mh[w] |= (uint32_t)(lev4[k] >> 8) << (8u * b);
            mirrored = mirrored && (uint16_t)(lev4[k] ^ 0x8000u) == lev4[15u - k];
        }
    }
    return mirrored;
}
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_levels4_mirror_pairs(uint32_t q, const uint32_t ml[2], const uint32_t mh[2],
                                                                                     uint32_t h[4]) {
    const uint32_t clear = ~q & 0x88888888u;                   /* a code's top bit, where it is clear */
    const uint32_t j = q ^ (clear - (clear >> 3));             /* k ^ 7 there; the top bit the permute ignores */
    const uint32_t x = q << 4;                                 /* codes 0, 2, 4 and 6's top bits at their bytes' top */
    uint32_t lo[2], hi[2];
#pragma unroll
    for (uint32_t i = 0u; i < 2u; ++i) {
        const uint32_t s = j >> (16u * i);
        uint32_t set;                                          /* 0xFF where a code's top bit is set */
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(set) : "r"(x), "r"(q), "r"(i == 0u ? 0xD9C8u : 0xFBEAu));
        lo[i] = __byte_perm(ml[0], ml[1], s);
        hi[i] = __byte_perm(mh[0], mh[1], s) ^ (~set & 0x80808080u);
    }
    h[0] = __byte_perm(lo[0], hi[0], 0x5140u); h[1] = __byte_perm(lo[0], hi[0], 0x7362u);
    h[2] = __byte_perm(lo[1], hi[1], 0x5140u); h[3] = __byte_perm(lo[1], hi[1], 0x7362u);
}
/* One 4-bit slice's products into `acc` by the mirror: as `slice_mma` at 4 bits, the A operands from `levels4_mirror_pairs`. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_slice_mma_mirror(const uint32_t c[2][8], const uint16_t* x, uint32_t t,
                                                                                  const uint32_t ml[2], const uint32_t mh[2], float acc[4]) {
    uint32_t xw[32];
    const uint4* xs = (const uint4*)(const void*)(x + 64u * t);
#pragma unroll
    for (uint32_t q = 0u; q < 8u; ++q) {
        const uint4 v = xs[q];
        xw[4u * q] = v.x; xw[4u * q + 1u] = v.y; xw[4u * q + 2u] = v.z; xw[4u * q + 3u] = v.w;
    }
#pragma unroll
    for (uint32_t s = 0u; s < 16u; s += 2u) {
        uint32_t h0[4], h1[4];
        nvidia_cuda12__zzprivate_levels4_mirror_pairs(c[0][s / 2u], ml, mh, h0);
        nvidia_cuda12__zzprivate_levels4_mirror_pairs(c[1][s / 2u], ml, mh, h1);
        nvidia_cuda12__zzpackage_mma(acc, h0[0], h1[0], h0[1], h1[1], xw[2u * s], xw[2u * s + 1u]);
        nvidia_cuda12__zzpackage_mma(acc, h0[2], h1[2], h0[3], h1[3], xw[2u * s + 2u], xw[2u * s + 3u]);
    }
}

/* One slice's products into `acc`, at width `D`: each step's A operands decoded from the codes, `x`'s 16U columns the B. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_slice_mma(const uint32_t D, const uint32_t c[2][8], const uint16_t* x,
                                                                           uint32_t t, const uint32_t tl[4], const uint32_t th[4],
                                                                           const uint16_t* lev8, float acc[4]) {
    const uint32_t U = D == 4u ? 4u : 2u, W = 8u * U;
    uint32_t xw[32];
    const uint4* xs = (const uint4*)(const void*)(x + 16u * U * t);
#pragma unroll
    for (uint32_t q = 0u; q < W / 4u; ++q) {
        const uint4 v = xs[q];
        xw[4u * q] = v.x; xw[4u * q + 1u] = v.y; xw[4u * q + 2u] = v.z; xw[4u * q + 3u] = v.w;
    }
    if (D == 4u) {                       /* a word's eight codes, two steps: bytes 0 and 1, then 2 and 3 */
#pragma unroll
        for (uint32_t s = 0u; s < 4u * U; s += 2u) {
            uint32_t h0[4], h1[4];
            nvidia_cuda12__zzpackage_levels4_pairs(c[0][s / 2u], tl, th, h0);
            nvidia_cuda12__zzpackage_levels4_pairs(c[1][s / 2u], tl, th, h1);
            nvidia_cuda12__zzpackage_mma(acc, h0[0], h1[0], h0[1], h1[1], xw[2u * s], xw[2u * s + 1u]);
            nvidia_cuda12__zzpackage_mma(acc, h0[2], h1[2], h0[3], h1[3], xw[2u * s + 2u], xw[2u * s + 3u]);
        }
        return;
    }
#pragma unroll
    for (uint32_t s = 0u; s < 4u * U; ++s) {
        uint32_t a[4];
#pragma unroll
        for (uint32_t r = 0u; r < 2u; ++r) {
            const uint32_t word = c[r][s];
            a[r]      = (uint32_t)lev8[word & 0xFFu] | ((uint32_t)lev8[(word >> 8) & 0xFFu] << 16);
            a[2u + r] = (uint32_t)lev8[(word >> 16) & 0xFFu] | ((uint32_t)lev8[word >> 24] << 16);
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
    __shared__ uint16_t lev8[256];
    __shared__ float parts[NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS][16];
    const uint32_t lane = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    const uint32_t l = lane % NVIDIA_CUDA12__ZZPACKAGE_WARP, g = l / 4u, t = l % 4u, warps = lanes / NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint32_t warp = lane / NVIDIA_CUDA12__ZZPACKAGE_WARP;
    /* the block's tiles: the job's, or its class's — a split a class, a block's warps all one class's (▶ `gemvs`) */
    uint32_t S = (uint32_t)j->split;
    uint64_t lo = 0u, tiles = j->first[j->count], b0 = 0u;
    if (j->classes != 0u) {
        uint64_t c = 0u;
        while (c + 1u < j->classes && (uint64_t)nn__silicon__block() >= j->class_block[c + 1u]) ++c;
        S = (uint32_t)j->class_split[c]; lo = j->class_tile[c]; tiles = j->class_tile[c + 1u]; b0 = j->class_block[c];
    }
    const uint32_t per = warps / S, p = warp % S;
    bool staged = false;
    uint32_t tl[4], th[4];                                 /* the 4-bit levels, in registers (▶ kernels.cuh) */
    nvidia_cuda12__zzpackage_levels4(tl, th);
    uint32_t ml[2], mh[2];                                 /* ...and by their mirror, where they are mirrored */
    const bool mirrored = nvidia_cuda12__zzprivate_levels4_mirror(ml, mh);
    for (uint64_t base = lo + ((uint64_t)nn__silicon__block() - b0) * per; base < tiles;
         base += (uint64_t)nn__silicon__blocks() * per) {
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
            for (uint32_t e = lane; e < 256u; e += lanes) lev8[e] = nn__turboquant__zzabi_levels(8u)[e];
            __syncthreads();
            staged = true;
        }
        while (nvidia_cuda12__zzprivate_before(now, stop)) {
            nvidia_cuda12__zzprivate_fetch(j, ahead, stop, r0, g, t, c2);
            if (nvidia_cuda12__zzprivate_before(ahead, stop)) nvidia_cuda12__zzprivate_advance(j, &ahead);
            const uint64_t m = now.m;
            const uint16_t* xm = (j->sum ? j->x + m * j->cols : j->x) + now.k;
            if (j->d[m] == 4u && mirrored) nvidia_cuda12__zzprivate_slice_mma_mirror(c0, xm, t, ml, mh, acc);
            else if (j->d[m] == 4u) nvidia_cuda12__zzprivate_slice_mma(4u, c0, xm, t, tl, th, lev8, acc);
            else               nvidia_cuda12__zzprivate_slice_mma(8u, c0, xm, t, tl, th, lev8, acc);
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

/* ⭐ A SUM'S TILE SPLIT BETWEEN WARPS (NN-168) — `groups_rows` gives a sum's tile one warp, which walks every matrix's
 *   columns alone: the 27B's MLP down onto the residual, one matrix of 17,408 columns, is 320 walks of 68 slices each.
 *   Here `split` warps take a tile, part `p` its run of each matrix's slices — the same run in each, counted in the sum's
 *   slice: 256 columns where a matrix is 4-bit, else 128 — loaded two slices ahead across a matrix's end as within one;
 *   each matrix's share weighted as the part leaves it, the parts added in shared memory in the warps' order, the
 *   residual added last. A sum `gemvs` does not split runs on `groups_rows`, its answer that kernel's to the bit; and
 *   `groups_rows` is untouched by this one. `MEASURED` the door bench on an A40 (sm_86,
 *   measurements/2026-11-10_nn168_a40.md): the 27B's MLP down onto the residual from 439-445 GB/s to 457-462, the rate
 *   the same matrix reaches on its own. ⛳ THE BARRIERS as `groups_rows`': once after staging nn's levels, once before
 *   a tile's parts are added and once after. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_advance_part(const nvidia_cuda12__gemv_job* j, nvidia_cuda12__gemv_walk* w,
                                                                              uint64_t k0, uint64_t k1) {
    w->k += nvidia_cuda12__zzprivate_slice(j->d[w->m]);
    if (w->k >= k1) { w->k = k0; w->m += 1u; }
}
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_groups_sum_rows(const nvidia_cuda12__gemv_job* j) {
    __shared__ uint16_t lev8[256];
    __shared__ float parts[NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS][16];
    const uint32_t lane = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    const uint32_t l = lane % NVIDIA_CUDA12__ZZPACKAGE_WARP, g = l / 4u, t = l % 4u, warps = lanes / NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint32_t warp = lane / NVIDIA_CUDA12__ZZPACKAGE_WARP, S = (uint32_t)j->split, per = warps / S, p = warp % S;
    /* part `p`'s columns of every matrix, in shifts: the sum's slice 2^8 or 2^7 columns, the split a power of two */
    uint32_t ls = 7u;
    for (uint64_t m = 0u; m < j->count; ++m) if (j->d[m] == 4u) ls = 8u;
    const uint32_t lS = (uint32_t)__ffs((int)S) - 1u;
    const uint64_t n = j->cols >> ls, k0 = ((n * p) >> lS) << ls, k1 = ((n * (p + 1u)) >> lS) << ls;
    const uint64_t tiles = j->first[1], rows = j->rows[0];
    bool staged = false;
    uint32_t tl[4], th[4];                                 /* the 4-bit levels, in registers (▶ kernels.cuh) */
    nvidia_cuda12__zzpackage_levels4(tl, th);
    uint32_t ml[2], mh[2];                                 /* ...and by their mirror, where they are mirrored */
    const bool mirrored = nvidia_cuda12__zzprivate_levels4_mirror(ml, mh);
    for (uint64_t base = (uint64_t)nn__silicon__block() * per; base < tiles; base += (uint64_t)nn__silicon__blocks() * per) {
        const uint64_t it = base + warp / S;
        const bool live = it < tiles;
        const uint64_t r0 = live ? it * 16u : 0u;
        const nvidia_cuda12__gemv_walk stop = { live ? j->count : 0u, 0u };
        nvidia_cuda12__gemv_walk now = { 0u, k0 }, ahead = now;
        float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f }, part[2] = { 0.0f, 0.0f };
        uint32_t c0[2][8], c1[2][8], c2[2][8];
        nvidia_cuda12__zzprivate_fetch(j, ahead, stop, r0, g, t, c0);
        if (nvidia_cuda12__zzprivate_before(ahead, stop)) nvidia_cuda12__zzprivate_advance_part(j, &ahead, k0, k1);
        nvidia_cuda12__zzprivate_fetch(j, ahead, stop, r0, g, t, c1);
        if (nvidia_cuda12__zzprivate_before(ahead, stop)) nvidia_cuda12__zzprivate_advance_part(j, &ahead, k0, k1);
        if (!staged) {
            for (uint32_t e = lane; e < 256u; e += lanes) lev8[e] = nn__turboquant__zzabi_levels(8u)[e];
            __syncthreads();
            staged = true;
        }
        while (nvidia_cuda12__zzprivate_before(now, stop)) {
            nvidia_cuda12__zzprivate_fetch(j, ahead, stop, r0, g, t, c2);
            if (nvidia_cuda12__zzprivate_before(ahead, stop)) nvidia_cuda12__zzprivate_advance_part(j, &ahead, k0, k1);
            const uint64_t m = now.m;
            const uint16_t* xm = j->x + m * j->cols + now.k;
            if (j->d[m] == 4u && mirrored) nvidia_cuda12__zzprivate_slice_mma_mirror(c0, xm, t, ml, mh, acc);
            else if (j->d[m] == 4u) nvidia_cuda12__zzprivate_slice_mma(4u, c0, xm, t, tl, th, lev8, acc);
            else               nvidia_cuda12__zzprivate_slice_mma(8u, c0, xm, t, tl, th, lev8, acc);
            nvidia_cuda12__zzprivate_advance_part(j, &now, k0, k1);
            if (now.m != m) {                              /* the part is done with a matrix: its share */
                const uint8_t* lut = (const uint8_t*)(uintptr_t)j->luts[m];
#pragma unroll
                for (uint32_t h = 0u; h < 2u; ++h) {
                    const uint64_t r = r0 + g + 8u * h;
                    if (r < rows && t == 0u) part[h] += acc[2u * h] * nvidia_cuda12__zzprivate_scale(lut, r) * nn__silicon__half_to_float(j->w[j->out_at[m]]);
                }
                acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
            }
#pragma unroll
            for (uint32_t r = 0u; r < 2u; ++r)
#pragma unroll
                for (uint32_t q = 0u; q < 8u; ++q) { c0[r][q] = c1[r][q]; c1[r][q] = c2[r][q]; }
        }
        if (t == 0u) { parts[warp][g] = part[0]; parts[warp][g + 8u] = part[1]; }
        __syncthreads();
        if (live && p == 0u && t == 0u)                    /* the tile's parts, in the warps' order, onto the residual */
#pragma unroll
            for (uint32_t h = 0u; h < 2u; ++h) {
                float sum = parts[warp][g + 8u * h];
                for (uint32_t q = 1u; q < S; ++q) sum += parts[warp + q][g + 8u * h];
                const uint64_t r = r0 + g + 8u * h;
                if (r < rows) j->out[r] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(j->residual[r]) + sum, j->over);
            }
        __syncthreads();
    }
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_GEMV_BOUNDS void nvidia_cuda12__zzprivate_gemv_sum(nvidia_cuda12__gemv_job j) {
    nvidia_cuda12__zzprivate_groups_sum_rows(&j);
}

/* The bound card's SM count and a kernel's warps resident on one — the gemv's, or with `sum` the split sum's — asked once
 * a card; 0 where they cannot be asked. */
static inline uint64_t nvidia_cuda12__zzprivate_gemv_sms(uint64_t* resident, bool sum) {
    static int sms[64], blocks[2][64];
    int d = 0;
    *resident = 0u;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) return 0u;
    int* b = &blocks[sum ? 1 : 0][d];
    const void* kernel = sum ? (const void*)nvidia_cuda12__zzprivate_gemv_sum : (const void*)nvidia_cuda12__zzprivate_gemv;
    if (sms[d] == 0 && cudaDeviceGetAttribute(&sms[d], cudaDevAttrMultiProcessorCount, d) != cudaSuccess) sms[d] = 0;
    if (*b == 0 && cudaOccupancyMaxActiveBlocksPerMultiprocessor(b, kernel, NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, 0)
                   != cudaSuccess) *b = 0;
    *resident = *b > 0 ? (uint64_t)*b * NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS : 0u;
    return sms[d] > 0 ? (uint64_t)sms[d] : 0u;
}

/* A job each matrix checked as nn's body would — 4 or 8 bits, whole slices, codes and `x` on sixteen bytes, the rows the
 * room holds — and launched; false, and nothing launched, when one does not fit. In a sum every matrix has the sum's
 * rows, so its tiles are those rows' and each tile walks every matrix.
 * ⛳ THE SPLIT AND THE WARPS A BLOCK FROM THE CARD. A matrix of 3,072 rows is 192 tiles, 5 warps an SM of the 5060 Ti's
 *   36: each walks every column alone, and the card idles between their loads (`MEASURED`, Nsight Compute, 2026-10-25:
 *   0.3 warps ready a scheduler; the 122B's shared expert down at 4 bits 34% of the DRAM peak) — and a matrix of 1,284
 *   tiles is 2.2 waves of the warps resident at once, its last wave a quarter full. So a tile's columns — a sum's on its
 *   own kernel, `groups_sum_rows` — are split between up to eight warps, each part at least a slice, while doubling the parts fills their waves more
 *   than a tenth better — a wave the warps resident on every SM at once, from the runtime's occupancy at the kernel's
 *   registers, so the split follows the card it runs on — and past two parts only while the card has fewer than six
 *   and a half of the matrix's warps an SM: beyond, more parts only shorten each warp's walk. Both sides of that line
 *   are measured: on the 5060 Ti's 36 SMs the 122B's shared expert at 128 and 192 tiles (7 and 11 warps an SM in two
 *   parts) fastest in two, four 4-30% slower (`MEASURED`, Nsight Compute); on an A100's 108 the 27B's MLP down, 320
 *   tiles (6 warps an SM in two), 384 GB/s in two and 491 in four, GLM's attention output, 256 tiles, 306 and 395
 *   (`MEASURED` measurements/2026-11-10_nn168_a100.md, cold). Then a block's warps halve while the blocks
 *   would be fewer than four an SM, down to the split: a matrix of few long rows — the 27B's MLP down, 5,120 rows of
 *   17,408 columns — is 320 tiles, 40 blocks of eight on a card of 36 SMs, and four SMs ran two blocks and the launch
 *   lasted as long as they did (`MEASURED`: 279 GB/s against the gate + up's 373, 272 blocks). A tile's split and so
 *   its answer's bytes follow the shapes and the card, never the call. */
/* The split `tiles` tiles take, their rows `slices` slices long, on a card of `sms` SMs holding `wave` of the kernel's
 * warps at once (▶ above). */
static inline uint32_t nvidia_cuda12__zzprivate_split(uint64_t tiles, uint64_t slices, uint64_t sms, uint64_t wave) {
    uint32_t split = 1u;
    while (wave != 0u && split < NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS && 2u * split <= slices) {
        /* the waves' fill, parts over the room their waves have: `now / (waves(now) · wave)`, against the doubled */
        const uint64_t now = tiles * split, next = 2u * now;
        const uint64_t waves = (now + wave - 1u) / wave, waves2 = (next + wave - 1u) / wave;
        if (10u * next * waves <= 11u * now * waves2) break;           /* less than a tenth better */
        if (split >= 2u && 2u * now >= 13u * sms) break;               /* six and a half warps an SM: enough */
        split *= 2u;
    }
    return split;
}

/* A job whose tiles are counted (`first`), launched with `split`: a block's warps halved while its blocks would be
 * fewer than four an SM, down to the split. */
static inline void nvidia_cuda12__zzprivate_gemvs_launch(nvidia_cuda12__gemv_job* j, uint32_t split, uint64_t sms) {
    const uint64_t tiles = j->first[j->count];
    uint32_t warps = NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS;
    while (warps > split && (tiles * split + warps - 1u) / warps < 4u * sms) warps /= 2u;
    j->split = split;
    const uint64_t per = warps / split, want = (tiles + per - 1u) / per;
    const uint32_t blocks = want < 0x7FFFFFFFull ? (uint32_t)want : 0x7FFFFFFFu;
    void* args[] = { j };
    const size_t sizes[] = { sizeof *j };
    if (j->sum && split > 1u)
        nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_gemv_sum, "nvidia_cuda12__zzprivate_gemv_sum", blocks,
                            warps * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 1u);
    else
        nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_gemv, "nvidia_cuda12__zzprivate_gemv", blocks,
                            warps * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 1u);
}

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
    const uint64_t sms = nvidia_cuda12__zzprivate_gemv_sms(&resident, j->sum != 0u);
    const uint64_t wave = resident * sms;
    if (j->sum || j->count == 1u) {
        nvidia_cuda12__zzprivate_gemvs_launch(j, nvidia_cuda12__zzprivate_split(tiles, fewest, sms, wave), sms);
        return true;
    }
    /* ⭐ A GROUP OF MATRICES, EACH INTO ITS OWN OUTPUTS, ANSWERS EACH AS THE ONE-MATRIX DOOR DOES, TO THE BIT (NN-168): a
     *   matrix's parts and the order they are added in follow its split, so each takes the split its own tiles and
     *   slices choose — a split chosen from the group's tiles together would part a matrix's bytes from its own door's.
     *   Still one launch: the matrices that choose alike are a class, in the group's order, and each block takes one
     *   class's tiles at that class's split (▶ `groups_rows`). ⚖ 2026-11-11 *"ok for the 168"*. */
    uint32_t split[NN__TURBOQUANT__GROUPS_MAX];
    bool alike = true;
    for (uint64_t i = 0u; i < j->count; ++i) {
        split[i] = nvidia_cuda12__zzprivate_split(j->first[i + 1u] - j->first[i], j->cols / (j->d[i] == 4u ? 256u : 128u),
                                                  sms, wave);
        alike = alike && split[i] == split[0];
    }
    if (alike) {
        nvidia_cuda12__zzprivate_gemvs_launch(j, split[0], sms);
        return true;
    }
    nvidia_cuda12__gemv_job k = *j;
    bool sent[NN__TURBOQUANT__GROUPS_MAX] = {};
    uint32_t most = 1u;
    uint64_t slots = 0u;
    k.count = 0u; k.first[0] = 0u; k.classes = 0u; k.class_tile[0] = 0u;
    for (uint64_t i = 0u; i < j->count; ++i) {
        if (sent[i]) continue;
        const uint64_t c = k.classes++;                    /* the splits are powers of two to eight: four classes at most */
        k.class_split[c] = split[i];
        for (uint64_t m = i; m < j->count; ++m) {
            if (sent[m] || split[m] != split[i]) continue;
            const uint64_t n = k.count++;
            k.codes[n] = j->codes[m]; k.luts[n] = j->luts[m]; k.rows[n] = j->rows[m]; k.d[n] = j->d[m];
            k.out_at[n] = j->out_at[m];
            k.first[n + 1u] = k.first[n] + (j->rows[m] + 15u) / 16u;
            sent[m] = true;
        }
        k.class_tile[c + 1u] = k.first[k.count];
        slots += (k.class_tile[c + 1u] - k.class_tile[c]) * split[i];
        if (split[i] > most) most = split[i];
    }
    /* a block's warps as one split would have them, from the slots every class's tiles take: each class's split divides
     *   them, so a tile's parts are one block's */
    uint32_t warps = NVIDIA_CUDA12__ZZPRIVATE_GEMV_WARPS;
    while (warps > most && (slots + warps - 1u) / warps < 4u * sms) warps /= 2u;
    k.class_block[0] = 0u;
    for (uint64_t c = 0u; c < k.classes; ++c) {
        const uint64_t per = warps / k.class_split[c];
        k.class_block[c + 1u] = k.class_block[c] + (k.class_tile[c + 1u] - k.class_tile[c] + per - 1u) / per;
    }
    k.split = most;
    const uint64_t want = k.class_block[k.classes];
    const uint32_t blocks = want < 0x7FFFFFFFull ? (uint32_t)want : 0x7FFFFFFFu;
    void* args[] = { &k };
    const size_t sizes[] = { sizeof k };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_gemv, "nvidia_cuda12__zzprivate_gemv", blocks,
                        warps * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 1u);
    return true;
}

/* ⭐ THE GEMV OF HALVES — `d` 16, a router's matrix (the 35B's 256 × 2048 a layer at every position), no scales:
 *   `out[r] = half(Σ_c W[r][c] · x[c])`. nn's body reads a half of a row at a time. Here a warp takes a row, each lane
 *   eight halves a read (sixteen bytes) and four reads in flight before the first product, into four partial sums, one a
 *   read; the warp's butterfly adds each partial across the lanes, and the four are added in pairs, as
 *   amd_rocm6_wave64's halves kernel does. The products are nn's; the order they are summed in is this kernel's own, the
 *   same every call. Rows of whole groups of eight, the matrix and `x` on sixteen bytes (▶ the door). */
#define NVIDIA_CUDA12__ZZPRIVATE_ROUTER_BOUNDS __launch_bounds__(256)
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ROUTER_BOUNDS void nvidia_cuda12__zzprivate_router_gemv(uint16_t* out, const uint8_t* weights,
                                                                                              const uint16_t* x, uint64_t rows,
                                                                                              uint64_t cols, unsigned int* over) {
    const uint32_t W = NVIDIA_CUDA12__ZZPACKAGE_WARP, lane = (uint32_t)nn__silicon__lane() % W;
    const uint64_t warps = nn__silicon__lanes() / W, runs = cols / 8u;
    const uint4* xr = (const uint4*)(const void*)x;
    for (uint64_t r = (uint64_t)nn__silicon__block() * warps + nn__silicon__lane() / W; r < rows;
         r += (uint64_t)nn__silicon__blocks() * warps) {
        const uint4* row = (const uint4*)(const void*)(weights + r * cols * 2u);
        float q[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint64_t j0 = lane; j0 < runs; j0 += 4u * W) {
            uint4 c[4], xv[4];
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                const uint64_t j = j0 + (uint64_t)u * W;
                c[u] = row[j < runs ? j : j0];
                xv[u] = xr[j < runs ? j : j0];
            }
#pragma unroll
            for (uint32_t u = 0u; u < 4u; ++u) {
                if (j0 + (uint64_t)u * W >= runs) continue;
                const uint32_t cw[4] = { c[u].x, c[u].y, c[u].z, c[u].w }, xw[4] = { xv[u].x, xv[u].y, xv[u].z, xv[u].w };
#pragma unroll
                for (uint32_t e = 0u; e < 4u; ++e) {
                    q[u] += nn__silicon__half_to_float((uint16_t)(cw[e] & 0xFFFFu)) * nn__silicon__half_to_float((uint16_t)(xw[e] & 0xFFFFu));
                    q[u] += nn__silicon__half_to_float((uint16_t)(cw[e] >> 16)) * nn__silicon__half_to_float((uint16_t)(xw[e] >> 16));
                }
            }
        }
#pragma unroll
        for (uint32_t s = 0u; s < 4u; ++s)
#pragma unroll
            for (uint32_t off = 1u; off < W; off <<= 1) q[s] += __shfl_xor_sync(0xFFFFFFFFu, q[s], (int)off);
        if (lane == 0u) out[r] = nn__kernels__zzabi_to_half((q[0] + q[1]) + (q[2] + q[3]), over);
    }
}

static void nvidia_cuda12__override_turboquant_gemv(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                    uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                    unsigned int* over) {
    if (d == 16u && rows != 0u && cols != 0u && cols % 8u == 0u && rows <= w_room / (cols * 2u)
     && ((uint64_t)(uintptr_t)weights) % 16u == 0u && ((uint64_t)(uintptr_t)x) % 16u == 0u) {
        const uint64_t want = (rows + 7u) / 8u;
        const uint32_t blocks = want < NN__KERNELS__BLOCKS_MAX ? (uint32_t)want : NN__KERNELS__BLOCKS_MAX;
        void* args[] = { &out, &weights, &x, &rows, &cols, &over };
        const size_t sizes[] = { sizeof out, sizeof weights, sizeof x, sizeof rows, sizeof cols, sizeof over };
        nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_router_gemv, "nvidia_cuda12__zzprivate_router_gemv", blocks, 256u, args, sizes, 6u);
        return;
    }
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

/* ⭐ THE GROUPS OVER `n` INPUTS — `x`'s rows `cols` apart, their answers `out_stride` apart, or with `out_stride` 0 each
 *   group's answers one input after another (the rows verbs' layout) — a draft's check: several positions through every
 *   matrix of a step at once. Here an input at a time through the one-input door above, so each answer is that door's to
 *   the bit, on the tensor cores where nn's body is its generic gemv. amd_rocm6_wave64 reads the codes once for two
 *   inputs; the tensor cores' product has eight columns, and the one-input door fills all eight with the same `x`, so the
 *   same tile could take eight inputs at once — not written yet (`REASONED`: each column of the product is summed alone,
 *   so an input's answer would stay the one-input door's). */
static void nvidia_cuda12__override_turboquant_gemv_groups_n(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                             uint64_t n, uint64_t out_stride, unsigned int* over) {
    for (uint64_t k = 0u; k < n; ++k) {
        uint16_t* o = out;
        const nn__turboquant__groups gk = nn__turboquant__groups_input(&g, &o, k, out_stride);
        nvidia_cuda12__override_turboquant_gemv_groups(o, gk, x + k * cols, cols, over);
    }
}

/* ══ ⭐ THE int8 GEMV AT 4 BITS — a warp a row ═════════════════════════════════════════════════════════════════════════
 * nn's definition (`nn/gpu/kernels/kernels.cuh`, `rows_i8`): `x` quantised in blocks of 32 columns (nn's
 * `zzabi_quantise`), each block's int8 products with the int8 levels summed in int32 — exact — then the block's float scale,
 * and the row's scale / 127 once the parts are summed. nn's body gives a row a block of 256 lanes, a lane a block of 32
 * columns at a time, each code decoded alone. Here, as amd_rocm6_wave64's `int8_rows4`: a warp a row, eight rows a block,
 * a lane's blocks `l, l + 32, …` of `x` quantised once a kernel and held in registers, and four codes a step decoded into
 * four int8 levels by two `__byte_perm`s over the sixteen levels held in four registers — AMD's `v_perm_b32`, whose CUDA
 * form takes its selectors as nibbles, so a row's 16 bits of codes are the selector as they stand — and multiplied by
 * `__dp4a`. The products are exact; the parts are added in the lane's blocks' order and then by the warp's butterfly, so
 * the sum's last bits are this kernel's own, as AMD's are. Whole blocks of 32 columns up to 32 · 32 · `INT8_HELD`, codes
 * on eight bytes, the rows and scales the rooms hold; anything else is nn's. */
#define NVIDIA_CUDA12__ZZPRIVATE_INT8_HELD  8u         /* blocks of x a lane holds: up to 32 · 8 · 32 = 8192 columns */
#define NVIDIA_CUDA12__ZZPRIVATE_INT8_WARPS 8u
/* The int8 levels of four codes — 16 bits of a row, low nibble first — packed as four bytes. */
static __device__ __forceinline__ uint32_t nvidia_cuda12__zzprivate_levels4_i8(const uint32_t* lev, uint32_t codes) {
    const uint32_t sel = codes & 0x7777u, t = (codes >> 3) & 0x1111u;
    const uint32_t top = ((t & 0x1u) | ((t & 0x10u) << 4) | ((t & 0x100u) << 8) | ((t & 0x1000u) << 12)) * 0xFFu;
    const uint32_t low = __byte_perm(lev[0], lev[1], sel), high = __byte_perm(lev[2], lev[3], sel);
    return (low & ~top) | (high & top);
}
#define NVIDIA_CUDA12__ZZPRIVATE_INT8_BOUNDS __launch_bounds__(256)
static __global__ NVIDIA_CUDA12__ZZPRIVATE_INT8_BOUNDS void nvidia_cuda12__zzprivate_int8_4(uint16_t* out, const uint8_t* weights, const uint8_t* luts,
                                                                               const uint16_t* x, uint64_t rows, uint64_t cols,
                                                                               unsigned int* over) {
    const int8_t* table = nn__turboquant__zzabi_levels_i8(4u);
    uint32_t lev[4];
#pragma unroll
    for (uint32_t w = 0u; w < 4u; ++w)
        lev[w] = (uint32_t)(uint8_t)table[4u * w] | ((uint32_t)(uint8_t)table[4u * w + 1u] << 8)
               | ((uint32_t)(uint8_t)table[4u * w + 2u] << 16) | ((uint32_t)(uint8_t)table[4u * w + 3u] << 24);
    const uint32_t lane = (uint32_t)nn__silicon__lane() % NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint32_t warp = (uint32_t)nn__silicon__lane() / NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint64_t blocks = cols / 32u, row_bytes = cols / 2u;
    uint32_t q[NVIDIA_CUDA12__ZZPRIVATE_INT8_HELD][8];
    float s[NVIDIA_CUDA12__ZZPRIVATE_INT8_HELD];
#pragma unroll
    for (uint32_t k = 0u; k < NVIDIA_CUDA12__ZZPRIVATE_INT8_HELD; ++k) {
        const uint64_t b = lane + (uint64_t)k * NVIDIA_CUDA12__ZZPACKAGE_WARP;
        s[k] = b < blocks ? nn__turboquant__zzabi_quantise(x, b * 32u, q[k]) : 0.0f;
    }
    for (uint64_t r = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_INT8_WARPS + warp; r < rows;
         r += (uint64_t)nn__silicon__blocks() * NVIDIA_CUDA12__ZZPRIVATE_INT8_WARPS) {
        const uint8_t* row = weights + r * row_bytes;
        float part = 0.0f;
#pragma unroll
        for (uint32_t k = 0u; k < NVIDIA_CUDA12__ZZPRIVATE_INT8_HELD; ++k) {
            const uint64_t b = lane + (uint64_t)k * NVIDIA_CUDA12__ZZPACKAGE_WARP;
            if (b < blocks) {
                const uint64_t* codes = (const uint64_t*)(const void*)(row + b * 16u);    /* 32 columns: 16 bytes */
                int acc = 0;
#pragma unroll
                for (uint32_t e = 0u; e < 2u; ++e) {
                    const uint64_t c = codes[e];
#pragma unroll
                    for (uint32_t f = 0u; f < 4u; ++f)
                        acc = __dp4a((int)nvidia_cuda12__zzprivate_levels4_i8(lev, (uint32_t)(c >> (16u * f)) & 0xFFFFu),
                                     (int)q[k][4u * e + f], acc);
                }
                part += (float)acc * s[k];
            }
        }
        part = nvidia_cuda12__zzpackage_warp_sum(part);
        if (lane == 0u) {
            const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)luts[r * 2ull] | ((uint32_t)luts[r * 2ull + 1ull] << 8)));
            out[r] = nvidia_cuda12__zzpackage_to_half(part * scale / 127.0f, over);
        }
    }
}
/* The blocks the card holds at once, asked once a card: a block quantises its lanes' `x` once, so a grid larger than the
 * card holds would quantise it again for every block that waits. 0 where they cannot be asked. */
static inline uint64_t nvidia_cuda12__zzprivate_int8_resident(void) {
    static int sms[64], blocks[64];
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) return 0u;
    if (sms[d] == 0 && cudaDeviceGetAttribute(&sms[d], cudaDevAttrMultiProcessorCount, d) != cudaSuccess) sms[d] = 0;
    if (blocks[d] == 0 && cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks[d], nvidia_cuda12__zzprivate_int8_4,
                              NVIDIA_CUDA12__ZZPRIVATE_INT8_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, 0) != cudaSuccess) blocks[d] = 0;
    return sms[d] > 0 && blocks[d] > 0 ? (uint64_t)sms[d] * (uint64_t)blocks[d] : 0u;
}
static void nvidia_cuda12__override_turboquant_gemv_int8(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                         uint64_t l_room, const uint16_t* x, uint64_t d, uint64_t rows, uint64_t cols,
                                                         unsigned int* over) {
    const uint64_t row_bytes = cols * d / 8u;
    if (d != 4u || cols == 0u || cols % 32u != 0u || cols > 32u * NVIDIA_CUDA12__ZZPACKAGE_WARP * NVIDIA_CUDA12__ZZPRIVATE_INT8_HELD
     || ((uint64_t)(uintptr_t)weights) % 8u != 0u || rows > w_room / row_bytes || rows > l_room / 2u) {
        nn__turboquant__zzabi_launch_gemv_int8(out, weights, w_room, luts, l_room, x, d, rows, cols, over);
        return;
    }
    if (rows == 0u) return;
    const uint64_t want = (rows + NVIDIA_CUDA12__ZZPRIVATE_INT8_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_INT8_WARPS;
    const uint64_t most = nvidia_cuda12__zzprivate_int8_resident();
    const uint64_t cap = most != 0u && most < NN__KERNELS__BLOCKS_MAX ? most : NN__KERNELS__BLOCKS_MAX;
    const uint32_t blocks = (uint32_t)(want < cap ? want : cap);
    void* args[] = { &out, &weights, &luts, &x, &rows, &cols, &over };
    const size_t sizes[] = { sizeof out, sizeof weights, sizeof luts, sizeof x, sizeof rows, sizeof cols, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_int8_4, "nvidia_cuda12__zzprivate_int8_4", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_INT8_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 7u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_TURBOQUANT_CUH */
