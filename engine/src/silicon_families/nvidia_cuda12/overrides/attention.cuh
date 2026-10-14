#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_ATTENTION_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_ATTENTION_CUH
/* ══ attention's DOORS, RUN FASTER — nvidia_cuda12 ═══════════════════════════════════════════════════════════
 * nn's attention doors (`nn/cpu/opcodes/attention__abi.cuh`), where this family covers them: a prompt's rows over the
 * window, causally — the scores through `mma.sync`, the mix in nn's own order. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "kernels.cuh"      /* what the overrides share */

/* ── ⭐ A PROMPT'S CAUSAL ATTENTION ──────────────────────────────────────────────────────────────────────────────────
 * nn's definitions (`nn/gpu/kernels/kernels.cuh`): row `r`'s query head `h` at position `first + r` scores the window's
 * keys up to itself, `p[(r · q_heads + h) · (first + rows) + t] = q · k[t] / √d`; `attention_causal_softmax` makes each
 * row its softmax in place; the mix is `o[r][h] = Σ_t p[…][t] · v[t]`, rounded to a half. A kv head's queries are its
 * group's heads of every row, `j = r · group + hg`, and a WARP takes a tile of them:
 *   · THE SCORES on the tensor cores: 16 queries by 64 positions through `mma.sync`, the operands loaded from where they
 *     lie straight into registers, the columns in the order `expert.cuh` takes them (lane `t` of four the columns
 *     `16t .. 16t + 15` of a slice of 64, both operands alike); in fp32 as nn's, written up to each row's own length, a
 *     tile past every row's length skipped. The products are nn's and their sum's order is not — as nn's grouped scores
 *     already sum in pairs where its one-query scores do not.
 *   · THE MIX in nn's own order — ⛔ NOT on the tensor cores: it is the many-row form of `attention_mix`, and the two agree
 *     to the bit (test/src_abi_verbs_on_device.cpp). Each output is its weights times its values added from the first
 *     position on, as nn's lane adds them; what is this family's own is that a lane holds 8 queries' sums of one column,
 *     so a value is read once for all 8 — a slice of 32 positions' values loaded while the slice before is added — and the
 *     weights reach the lanes by shuffles from one coalesced read of 32. `MEASURED` (test/src_expert_gemm_sweep.cpp `a`,
 *     the 27B's 256 rows at 1779): 16 queries a lane held 180 registers and ran 3.47 ms, 8 ran 2.82, nn's body 5.55. */
#define NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS 8u
#define NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS __launch_bounds__(256)

/* The flat query `j` of kv head `g` as its row in `q` (and `o`, and `p` over the span). */
static __device__ __forceinline__ uint64_t nvidia_cuda12__zzprivate_query(uint64_t j, uint64_t g, uint64_t group, uint64_t q_heads) {
    return (j / group) * q_heads + g * group + j % group;
}

/* Sixteen halves at `at`, two to a register — two 16-byte loads; zeros where `at` is null. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_sixteen(const uint16_t* at, uint32_t w[8]) {
    if (!at) { for (uint32_t i = 0u; i < 8u; ++i) w[i] = 0u; return; }
    const uint4 a = ((const uint4*)(const void*)at)[0], b = ((const uint4*)(const void*)at)[1];
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
}

/* Where query `j` of kv head `g` writes and reads its weights: in the causal doors' layout a row of `span` for every query,
 * `p[(r · q_heads + h) · span]`; in the many-row residuals' (`per_row`), each row's block of `q_heads · span` laid out as
 * the one-row door lays it, a head after another at the row's own length `len`. */
static __device__ __forceinline__ uint64_t nvidia_cuda12__zzprivate_weights_at(uint64_t j, uint64_t g, uint64_t group, uint64_t q_heads,
                                                                               uint64_t span, uint64_t len, bool per_row) {
    return per_row ? (j / group) * q_heads * span + (g * group + j % group) * len
                   : ((j / group) * q_heads + g * group + j % group) * span;
}

/* ⭐ THE SCORES: a warp 16 queries by 64 positions, `d / 64` slices of the head's columns. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_scores(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                       uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows,
                                                                       bool per_row) {
    const uint32_t lane = (uint32_t)nn__silicon__lane() % NVIDIA_CUDA12__ZZPACKAGE_WARP, g8 = lane / 4u, t4 = lane % 4u;
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim, span = first + rows, J = rows * group;
    const uint64_t q_tiles = (J + 15u) / 16u, p_tiles = (span + 63u) / 64u, items = kv_heads * q_tiles * p_tiles;
    const float scale = 1.0f / nn__silicon__sqrtf((float)head_dim);
    for (uint64_t it = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS + nn__silicon__lane() / NVIDIA_CUDA12__ZZPACKAGE_WARP;
         it < items; it += (uint64_t)nn__silicon__blocks() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS) {
        const uint64_t pt = it % p_tiles, rest = it / p_tiles, qt = rest % q_tiles, g = rest / q_tiles;
        const uint64_t j0 = qt * 16u, p0 = pt * 64u;
        const uint64_t j_last = j0 + 15u < J ? j0 + 15u : J - 1u;
        if (p0 >= first + j_last / group + 1u) continue;          /* every position past every row's length */
        const uint64_t ja = j0 + g8, jb = j0 + g8 + 8u;
        const uint16_t* qa = ja < J ? q + nvidia_cuda12__zzprivate_query(ja, g, group, q_heads) * head_dim : 0;
        const uint16_t* qb = jb < J ? q + nvidia_cuda12__zzprivate_query(jb, g, group, q_heads) * head_dim : 0;
        float acc[8][4];
#pragma unroll
        for (uint32_t n = 0u; n < 8u; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        for (uint64_t e0 = 0u; e0 < head_dim; e0 += 64u) {
            uint32_t a[2][8];
            nvidia_cuda12__zzprivate_sixteen(qa ? qa + e0 + 16u * t4 : 0, a[0]);
            nvidia_cuda12__zzprivate_sixteen(qb ? qb + e0 + 16u * t4 : 0, a[1]);
#pragma unroll
            for (uint32_t n = 0u; n < 8u; ++n) {
                const uint64_t pos = p0 + 8u * n + g8;
                uint32_t b[8];
                nvidia_cuda12__zzprivate_sixteen(pos < span ? k + pos * row + g * head_dim + e0 + 16u * t4 : 0, b);
#pragma unroll
                for (uint32_t s = 0u; s < 4u; ++s)
                    nvidia_cuda12__zzpackage_mma(acc[n], a[0][2u * s], a[1][2u * s], a[0][2u * s + 1u], a[1][2u * s + 1u], b[2u * s], b[2u * s + 1u]);
            }
        }
#pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint64_t j = j0 + g8 + 8u * h;
            if (j >= J) continue;
            const uint64_t len = first + j / group + 1u;
            float* pr = p + nvidia_cuda12__zzprivate_weights_at(j, g, group, q_heads, span, len, per_row);
#pragma unroll
            for (uint32_t n = 0u; n < 8u; ++n)
#pragma unroll
                for (uint32_t e = 0u; e < 2u; ++e) {
                    const uint64_t pos = p0 + 8u * n + 2u * t4 + e;
                    if (pos < len) pr[pos] = acc[n][2u * h + e] * scale;
                }
        }
    }
}

static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_causal_scores(float* p, const uint16_t* q, const uint16_t* k,
                                                                                                  uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                                                                                  uint64_t first, uint64_t rows) {
    nvidia_cuda12__zzprivate_scores(p, q, k, q_heads, kv_heads, head_dim, first, rows, false);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_causal_scores_rows(float* p, const uint16_t* q, const uint16_t* k,
                                                                                                       uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                                                                                       uint64_t first, uint64_t rows) {
    nvidia_cuda12__zzprivate_scores(p, q, k, q_heads, kv_heads, head_dim, first, rows, true);
}

/* ⭐ THE MIX: a warp `Q` queries by 32 of a head's columns, a lane a column. Its `Q` sums run over the positions in order,
 * each the multiply-add nn's lane makes, a position past a query's length adding nothing to that query's sum. `Q` is 8 over
 * a prompt's rows, a value read once for eight, and 1 for a decoded position's one row — its kv head's few queries would
 * leave most of eight empty, and a warp each is four to eight times the warps. Each sum is the same either way. */
#define NVIDIA_CUDA12__ZZPRIVATE_MIX_Q 8u
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_mix(uint16_t* o, float* res, const float* p, const uint16_t* v,
                                                                     uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t first,
                                                                     uint64_t rows, unsigned int* over, const uint32_t Q, bool per_row) {
    const uint32_t lane = (uint32_t)nn__silicon__lane() % NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim, span = first + rows, J = rows * group;
    const uint64_t q_tiles = (J + Q - 1u) / Q;
    const uint64_t e_tiles = head_dim / NVIDIA_CUDA12__ZZPACKAGE_WARP, items = kv_heads * q_tiles * e_tiles;
    for (uint64_t it = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS + nn__silicon__lane() / NVIDIA_CUDA12__ZZPACKAGE_WARP;
         it < items; it += (uint64_t)nn__silicon__blocks() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS) {
        const uint64_t et = it % e_tiles, rest = it / e_tiles, qt = rest % q_tiles, g = rest / q_tiles;
        const uint64_t j0 = qt * Q, e = et * NVIDIA_CUDA12__ZZPACKAGE_WARP + lane;
        const uint64_t j_last = j0 + Q - 1u < J ? j0 + Q - 1u : J - 1u;
        const uint64_t most = first + j_last / group + 1u;
        const uint16_t* vg = v + g * head_dim + e;
        float sum[NVIDIA_CUDA12__ZZPRIVATE_MIX_Q];
        uint32_t len[NVIDIA_CUDA12__ZZPRIVATE_MIX_Q];             /* each query's length; 0 past the last query */
#pragma unroll
        for (uint32_t q = 0u; q < Q; ++q) {
            sum[q] = 0.0f;
            len[q] = j0 + q < J ? (uint32_t)(first + (j0 + q) / group + 1u) : 0u;
        }
        /* a slice is 32 positions: each query's weight at position `t0 + lane`, and this lane's column of the slice's
         * values — the next slice's loaded while this one's are added, so the sums never wait on memory */
        float w[NVIDIA_CUDA12__ZZPRIVATE_MIX_Q], wn[NVIDIA_CUDA12__ZZPRIVATE_MIX_Q];
        uint16_t vh[NVIDIA_CUDA12__ZZPACKAGE_WARP], vm[NVIDIA_CUDA12__ZZPACKAGE_WARP], vn[NVIDIA_CUDA12__ZZPACKAGE_WARP];
#pragma unroll
        for (uint32_t q = 0u; q < Q; ++q)
            w[q] = lane < len[q] ? p[nvidia_cuda12__zzprivate_weights_at(j0 + q, g, group, q_heads, span, len[q], per_row) + lane] : 0.0f;
#pragma unroll
        for (uint32_t i = 0u; i < NVIDIA_CUDA12__ZZPACKAGE_WARP; ++i) {
            vh[i] = i < most ? vg[(uint64_t)i * row] : (uint16_t)0u;
            vm[i] = NVIDIA_CUDA12__ZZPACKAGE_WARP + i < most ? vg[(uint64_t)(NVIDIA_CUDA12__ZZPACKAGE_WARP + i) * row] : (uint16_t)0u;
        }
        for (uint64_t t0 = 0u; t0 < most; t0 += NVIDIA_CUDA12__ZZPACKAGE_WARP) {
            const uint64_t t1 = t0 + NVIDIA_CUDA12__ZZPACKAGE_WARP, t2 = t1 + NVIDIA_CUDA12__ZZPACKAGE_WARP;
#pragma unroll
            for (uint32_t q = 0u; q < Q; ++q)
                wn[q] = t1 + lane < len[q] ? p[nvidia_cuda12__zzprivate_weights_at(j0 + q, g, group, q_heads, span, len[q], per_row) + t1 + lane] : 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < NVIDIA_CUDA12__ZZPACKAGE_WARP; ++i) vn[i] = t2 + i < most ? vg[(t2 + i) * row] : (uint16_t)0u;
#pragma unroll
            for (uint32_t i = 0u; i < NVIDIA_CUDA12__ZZPACKAGE_WARP; ++i) {
                const uint32_t t = (uint32_t)(t0 + i);
                const float vf = nn__silicon__half_to_float(vh[i]);
#pragma unroll
                for (uint32_t q = 0u; q < Q; ++q) {
                    const float pt = __shfl_sync(0xFFFFFFFFu, w[q], (int)i);
                    if (t < len[q]) sum[q] += pt * vf;
                }
            }
#pragma unroll
            for (uint32_t q = 0u; q < Q; ++q) w[q] = wn[q];
#pragma unroll
            for (uint32_t i = 0u; i < NVIDIA_CUDA12__ZZPACKAGE_WARP; ++i) { vh[i] = vm[i]; vm[i] = vn[i]; }
        }
#pragma unroll
        for (uint32_t q = 0u; q < Q; ++q) {
            const uint64_t j = j0 + q;
            if (j >= J) continue;
            const uint64_t at = nvidia_cuda12__zzprivate_query(j, g, group, q_heads);
            if (res) res[at * NN__ATTENTION__RESIDUAL_FLOATS(head_dim) + 2u + e] = sum[q];
            else     o[at * head_dim + e] = nn__kernels__zzabi_to_half(sum[q], over);
        }
    }
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_causal_mix(uint16_t* o, const float* p, const uint16_t* v,
                                                                                              uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                                                                              uint64_t first, uint64_t rows, unsigned int* over) {
    nvidia_cuda12__zzprivate_mix(o, 0, p, v, q_heads, kv_heads, head_dim, first, rows, over, NVIDIA_CUDA12__ZZPRIVATE_MIX_Q, false);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_causal_mix_one(uint16_t* o, const float* p, const uint16_t* v,
                                                                                                  uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                                                                                  uint64_t first, uint64_t rows, unsigned int* over) {
    nvidia_cuda12__zzprivate_mix(o, 0, p, v, q_heads, kv_heads, head_dim, first, rows, over, 1u, false);
}
/* ⭐ AND A RESIDUAL'S SECOND PASS — `attention_residual_mix`'s `acc`, the weights times the values in fp32 and nothing rounded,
 *   one query a warp, as the one-row mix above. */
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_residual_mix(float* res, const float* p, const uint16_t* v,
                                                                                                uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                                                                                uint64_t length) {
    nvidia_cuda12__zzprivate_mix(0, res, p, v, q_heads, kv_heads, head_dim, length - 1u, 1u, 0, 1u, false);
}
/* …and over many rows, the many-row residuals' layout: `Q` queries a warp, each sum the one-row kernel's. */
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_residual_mix_rows(float* res, const float* p, const uint16_t* v,
                                                                                                     uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim,
                                                                                                     uint64_t first, uint64_t rows) {
    nvidia_cuda12__zzprivate_mix(0, res, p, v, q_heads, kv_heads, head_dim, first, rows, 0, NVIDIA_CUDA12__ZZPRIVATE_MIX_Q, true);
}
/* ⭐ AND ITS FIRST PASS'S SECOND HALF — `attention_weights` after the scores: each head's largest into its residual's `m`, the
 *   weights `e^(score - m)` in place, their sum its `l` — nn's arithmetic, a block a head through the family's fold. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_residual_head(float* ph, float* rh, uint64_t length) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    float top = -NN__KERNELS__INFINITY;
    for (uint64_t t = lane; t < length; t += lanes) if (ph[t] > top) top = ph[t];
    top = nn__silicon__lanes_max(top);
    float part = 0.0f;
    for (uint64_t t = lane; t < length; t += lanes) {
        ph[t] = nn__silicon__expf(ph[t] - top);
        part += ph[t];
    }
    const float sum = nn__silicon__lanes_sum(part);
    if (lane == 0u) {
        rh[0] = top;
        rh[1] = sum;
    }
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_residual_weights(float* p, float* res, uint64_t q_heads,
                                                                                       uint64_t head_dim, uint64_t length) {
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks())
        nvidia_cuda12__zzprivate_residual_head(p + h * length, res + h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim), length);
}
/* …over many rows, row `r`'s heads at its own length `length + r · grow` (1 over the window, 0 over a tier) and its block
 *   of weights where the many-row residuals' layout puts it. */
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_residual_weights_rows(float* p, float* res, uint64_t q_heads,
                                                                                            uint64_t head_dim, uint64_t length, uint64_t rows,
                                                                                            uint64_t grow) {
    const uint64_t span = length + (rows - 1u) * grow;
    for (uint64_t f = nn__silicon__block(); f < rows * q_heads; f += nn__silicon__blocks()) {
        const uint64_t r = f / q_heads, h = f % q_heads, len = length + r * grow;
        nvidia_cuda12__zzprivate_residual_head(p + r * q_heads * span + h * len, res + f * NN__ATTENTION__RESIDUAL_FLOATS(head_dim), len);
    }
}

/* What both take: whole slices of 64 of a head's columns, whole groups, and `q` and `k` on sixteen bytes (the scores'
 * loads) or `o` and `v` on two (the mix's). */
static inline bool nvidia_cuda12__zzprivate_causal_fits(const void* q, const void* kv, uint64_t q_heads, uint64_t kv_heads,
                                                       uint64_t head_dim, uint64_t rows, uint64_t align) {
    return rows != 0u && kv_heads != 0u && q_heads % kv_heads == 0u && head_dim != 0u && head_dim % 64u == 0u
        && ((uint64_t)(uintptr_t)q) % align == 0u && ((uint64_t)(uintptr_t)kv) % align == 0u;
}

static void nvidia_cuda12__override_attention_causal_scores_grouped(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                    uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows) {
    if (!nvidia_cuda12__zzprivate_causal_fits(q, k, q_heads, kv_heads, head_dim, rows, 16u)) {
        nn__attention__zzabi_launch_causal_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, first, rows);
        return;
    }
    const uint64_t items = kv_heads * ((rows * (q_heads / kv_heads) + 15u) / 16u) * ((first + rows + 63u) / 64u);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    void* args[] = { &p, &q, &k, &q_heads, &kv_heads, &head_dim, &first, &rows };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof first, sizeof rows };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_causal_scores, "nvidia_cuda12__zzprivate_causal_scores", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 8u);
}

static void nvidia_cuda12__override_attention_causal_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads,
                                                         uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows,
                                                         unsigned int* over) {
    if (!nvidia_cuda12__zzprivate_causal_fits(o, v, q_heads, kv_heads, head_dim, rows, 2u) || ((uint64_t)(uintptr_t)p) % 4u != 0u) {
        nn__attention__zzabi_launch_causal_mix(o, p, v, q_heads, kv_heads, head_dim, first, rows, over);
        return;
    }
    const uint64_t Q = rows == 1u ? 1u : NVIDIA_CUDA12__ZZPRIVATE_MIX_Q;
    const uint64_t items = kv_heads * ((rows * (q_heads / kv_heads) + Q - 1u) / Q) * (head_dim / NVIDIA_CUDA12__ZZPACKAGE_WARP);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    void* args[] = { &o, &p, &v, &q_heads, &kv_heads, &head_dim, &first, &rows, &over };
    const size_t sizes[] = { sizeof o, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof first, sizeof rows,
                             sizeof over };
    nn__silicon__launch(Q == 1u ? (const void*)nvidia_cuda12__zzprivate_causal_mix_one : (const void*)nvidia_cuda12__zzprivate_causal_mix,
                        "nvidia_cuda12__zzprivate_causal_mix", blocks, NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP,
                        args, sizes, 9u);
}

/* ⭐ ONE QUERY — decode's `attention_scores` (its softmax in it) and `attention_mix` — AND THE CAUSAL ROWS' UNGROUPED SCORES,
 *   THROUGH THE SAME KERNELS. A query at position `length - 1` is the causal doors' one row from `first = length - 1`, in
 *   the same layout (`p[h · length + t]`, `q` and `o` a head a row), so a position decoded and a prompt's rows run the very
 *   kernels: the scores above, nn's own causal softmax — a lane's positions, the block's fold, the one-query softmax's
 *   arithmetic — and the mix in nn's order. nn's ungrouped causal scores go the same way, as its one-query scores' many-row
 *   form, to the bit. */
static void nvidia_cuda12__override_attention_causal_scores(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                            uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows) {
    if (!nvidia_cuda12__zzprivate_causal_fits(q, k, q_heads, kv_heads, head_dim, rows, 16u)) {
        nn__attention__zzabi_launch_causal_scores(p, q, k, q_heads, kv_heads, head_dim, first, rows);
        return;
    }
    nvidia_cuda12__override_attention_causal_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, first, rows);
    nn__attention__zzabi_launch_causal_softmax(p, q_heads, first, rows);
}
static void nvidia_cuda12__override_attention_scores(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads,
                                                     uint64_t head_dim, uint64_t length) {
    if (length == 0u || !nvidia_cuda12__zzprivate_causal_fits(q, k, q_heads, kv_heads, head_dim, 1u, 16u)) {
        nn__attention__zzabi_launch_scores(p, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    nvidia_cuda12__override_attention_causal_scores(p, q, k, q_heads, kv_heads, head_dim, length - 1u, 1u);
}
static void nvidia_cuda12__override_attention_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads,
                                                  uint64_t head_dim, uint64_t length, unsigned int* over) {
    if (length == 0u) { nn__attention__zzabi_launch_mix(o, p, v, q_heads, kv_heads, head_dim, length, over); return; }
    nvidia_cuda12__override_attention_causal_mix(o, p, v, q_heads, kv_heads, head_dim, length - 1u, 1u, over);
}

/* ⭐ A RESIDUAL OVER A RANGE — decode's tiered window: `attention_weights` as the scores above at one row and nn's own second
 *   half, `attention_residual_mix` as the one-row mix into fp32. */
static void nvidia_cuda12__override_attention_weights(float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                      uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (length == 0u || !nvidia_cuda12__zzprivate_causal_fits(q, k, q_heads, kv_heads, head_dim, 1u, 16u)) {
        nn__attention__zzabi_launch_weights(p, res, q, k, q_heads, kv_heads, head_dim, length);
        return;
    }
    nvidia_cuda12__override_attention_causal_scores_grouped(p, q, k, q_heads, kv_heads, head_dim, length - 1u, 1u);
    const uint32_t blocks = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* args[] = { &p, &res, &q_heads, &head_dim, &length };
    const size_t sizes[] = { sizeof p, sizeof res, sizeof q_heads, sizeof head_dim, sizeof length };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_residual_weights, "nvidia_cuda12__zzprivate_residual_weights", blocks, 256u,
                        args, sizes, 5u);
}
static void nvidia_cuda12__override_attention_residual_mix(float* res, const float* p, const uint16_t* v, uint64_t q_heads,
                                                           uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    if (length == 0u || !nvidia_cuda12__zzprivate_causal_fits(res, v, q_heads, kv_heads, head_dim, 1u, 2u)
     || ((uint64_t)(uintptr_t)p) % 4u != 0u) {
        nn__attention__zzabi_launch_residual_mix(res, p, v, q_heads, kv_heads, head_dim, length);
        return;
    }
    const uint64_t items = kv_heads * (q_heads / kv_heads) * (head_dim / NVIDIA_CUDA12__ZZPACKAGE_WARP);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    void* args[] = { &res, &p, &v, &q_heads, &kv_heads, &head_dim, &length };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof length };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_residual_mix, "nvidia_cuda12__zzprivate_residual_mix", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 7u);
}

/* ⭐ …AND OVER A PROMPT'S ROWS PAST THE TIERED CACHE'S WINDOW — `attention_weights_rows` as the scores over every row at once,
 *   causally, into the many-row residuals' layout, then each row's heads' second half; `attention_residual_mix_rows` as the mix,
 *   8 queries a warp. A row's weights and residual are the one-row doors' (test/src_abi_verbs_on_device.cpp, to the bit); what
 *   does not fit them is nn's, a row at a time through the one-row doors above. */
static void nvidia_cuda12__override_attention_weights_rows(float* p, float* res, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                           uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows) {
    if (!nvidia_cuda12__zzprivate_causal_fits(q, k, q_heads, kv_heads, head_dim, rows, 16u)) {
        nn__attention__zzabi_launch_weights_rows(p, res, q, k, q_heads, kv_heads, head_dim, first, rows);
        return;
    }
    const uint64_t items = kv_heads * ((rows * (q_heads / kv_heads) + 15u) / 16u) * ((first + rows + 63u) / 64u);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    void* args[] = { &p, &q, &k, &q_heads, &kv_heads, &head_dim, &first, &rows };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof k, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof first, sizeof rows };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_causal_scores_rows, "nvidia_cuda12__zzprivate_causal_scores_rows", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 8u);
    uint64_t length = first + 1u, grow = 1u;
    const uint32_t heads = rows * q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)(rows * q_heads) : NN__KERNELS__BLOCKS_MAX;
    void* wargs[] = { &p, &res, &q_heads, &head_dim, &length, &rows, &grow };
    const size_t wsizes[] = { sizeof p, sizeof res, sizeof q_heads, sizeof head_dim, sizeof length, sizeof rows, sizeof grow };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_residual_weights_rows, "nvidia_cuda12__zzprivate_residual_weights_rows", heads,
                        256u, wargs, wsizes, 7u);
}
static void nvidia_cuda12__override_attention_residual_mix_rows(float* res, const float* p, const uint16_t* v, uint64_t q_heads,
                                                                uint64_t kv_heads, uint64_t head_dim, uint64_t first, uint64_t rows) {
    if (!nvidia_cuda12__zzprivate_causal_fits(res, v, q_heads, kv_heads, head_dim, rows, 2u) || ((uint64_t)(uintptr_t)p) % 4u != 0u) {
        nn__attention__zzabi_launch_residual_mix_rows(res, p, v, q_heads, kv_heads, head_dim, first, rows);
        return;
    }
    const uint64_t Q = NVIDIA_CUDA12__ZZPRIVATE_MIX_Q;
    const uint64_t items = kv_heads * ((rows * (q_heads / kv_heads) + Q - 1u) / Q) * (head_dim / NVIDIA_CUDA12__ZZPACKAGE_WARP);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    void* args[] = { &res, &p, &v, &q_heads, &kv_heads, &head_dim, &first, &rows };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof v, sizeof q_heads, sizeof kv_heads, sizeof head_dim, sizeof first, sizeof rows };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_residual_mix_rows, "nvidia_cuda12__zzprivate_residual_mix_rows", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 8u);
}

/* ══ ⭐ THE TIERED CACHE'S TURBOQUANT ROWS — the warm tier at 8 bits, the cold at 4 ══════════════════════════════════════
 * nn's definitions: a key or value row is `head_dim` codes at the tier's width (row `t · kv_heads + g`) and one fp16 scale,
 * an element `scale · level(code)`. `attention_weights_tq` is `attention_weights` over such keys; `attention_residual_mix_tq`
 * the residual's `acc` over such values. Here:
 *   · the weights' scores on the tensor cores, as the window's: 16 queries by 64 positions, each key decoded from its codes
 *     into B operands as `turboquant.cuh` decodes a matrix's — nn's levels, two a register — its scale and `1/√d` then on
 *     its score; then nn's own max, exp and sum
 *   · the mix in nn's order, its element `scale · level` as nn's: a lane an element, one query a warp for a decoded position
 *     and eight over a prompt's rows, the 32 positions' codes and scales of a slice loaded while the slice before is added
 * Both take a prompt's rows too, every row the whole tier (`attention_weights_tq_rows`, `attention_residual_mix_tq_rows`).
 * Each stages nn's levels in shared memory once, the block waiting once before its first position — the category the rule
 * names for these two (`attention_weights_tq`, `attention_mix_tq`). */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_attention_weights_tq(float* p, const uint16_t* q, const uint8_t* codes,
                                                                                      const uint16_t* scales, uint64_t q_heads,
                                                                                      uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                                                      uint64_t d, uint64_t rows) {
    __shared__ uint32_t pair4[256];
    __shared__ uint16_t lev8[256];
    const uint32_t lid = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    for (uint32_t e = lid; e < 256u; e += lanes) {
        const uint16_t* lev4 = nn__turboquant__zzabi_levels(4u);
        pair4[e] = (uint32_t)lev4[e & 15u] | ((uint32_t)lev4[e >> 4] << 16);
        lev8[e] = nn__turboquant__zzabi_levels(8u)[e];
    }
    __syncthreads();
    const uint32_t lane = lid % NVIDIA_CUDA12__ZZPACKAGE_WARP, g8 = lane / 4u, t4 = lane % 4u;
    const uint64_t group = q_heads / kv_heads, row_bytes = head_dim * d / 8u, J = rows * group;
    const uint64_t q_tiles = (J + 15u) / 16u, p_tiles = (length + 63u) / 64u, items = kv_heads * q_tiles * p_tiles;
    const float inv = 1.0f / nn__silicon__sqrtf((float)(uint32_t)head_dim);
    for (uint64_t it = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS + lid / NVIDIA_CUDA12__ZZPACKAGE_WARP;
         it < items; it += (uint64_t)nn__silicon__blocks() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS) {
        const uint64_t pt = it % p_tiles, rest = it / p_tiles, qt = rest % q_tiles, g = rest / q_tiles;
        const uint64_t j0 = qt * 16u, p0 = pt * 64u, ja = j0 + g8, jb = j0 + g8 + 8u;
        const uint16_t* qa = ja < J ? q + nvidia_cuda12__zzprivate_query(ja, g, group, q_heads) * head_dim : 0;
        const uint16_t* qb = jb < J ? q + nvidia_cuda12__zzprivate_query(jb, g, group, q_heads) * head_dim : 0;
        float acc[8][4];
#pragma unroll
        for (uint32_t n = 0u; n < 8u; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        for (uint64_t e0 = 0u; e0 < head_dim; e0 += 64u) {
            uint32_t a[2][8];
            nvidia_cuda12__zzprivate_sixteen(qa ? qa + e0 + 16u * t4 : 0, a[0]);
            nvidia_cuda12__zzprivate_sixteen(qb ? qb + e0 + 16u * t4 : 0, a[1]);
#pragma unroll
            for (uint32_t n = 0u; n < 8u; ++n) {
                const uint64_t pos = p0 + 8u * n + g8;
                /* this lane's 16 codes of the key at `pos`: 8 bytes at 4 bits, 16 at 8 */
                uint32_t c[4] = { 0u, 0u, 0u, 0u };
                if (pos < length) {
                    const uint8_t* kr = codes + (pos * kv_heads + g) * row_bytes + (e0 + 16u * t4) * d / 8u;
                    const uint2 v0 = *(const uint2*)(const void*)kr;
                    c[0] = v0.x; c[1] = v0.y;
                    if (d == 8u) { const uint2 v1 = *(const uint2*)(const void*)(kr + 8); c[2] = v1.x; c[3] = v1.y; }
                }
#pragma unroll
                for (uint32_t s = 0u; s < 4u; ++s) {
                    uint32_t b0, b1;
                    if (d == 4u) {
                        const uint32_t word = c[s / 2u], sh = 16u * (s % 2u);
                        b0 = pair4[(word >> sh) & 0xFFu];
                        b1 = pair4[(word >> (sh + 8u)) & 0xFFu];
                    } else {
                        const uint32_t word = c[s];
                        b0 = (uint32_t)lev8[word & 0xFFu] | ((uint32_t)lev8[(word >> 8) & 0xFFu] << 16);
                        b1 = (uint32_t)lev8[(word >> 16) & 0xFFu] | ((uint32_t)lev8[word >> 24] << 16);
                    }
                    nvidia_cuda12__zzpackage_mma(acc[n], a[0][2u * s], a[1][2u * s], a[0][2u * s + 1u], a[1][2u * s + 1u], b0, b1);
                }
            }
        }
#pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint64_t j = j0 + g8 + 8u * h;
            if (j >= J) continue;
            float* ph = p + nvidia_cuda12__zzprivate_query(j, g, group, q_heads) * length;
#pragma unroll
            for (uint32_t n = 0u; n < 8u; ++n)
#pragma unroll
                for (uint32_t e = 0u; e < 2u; ++e) {
                    const uint64_t pos = p0 + 8u * n + 2u * t4 + e;
                    if (pos < length)
                        ph[pos] = acc[n][2u * h + e] * nn__silicon__half_to_float(scales[pos * kv_heads + g]) * inv;
                }
        }
    }
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_weights_tq(float* p, const uint16_t* q, const uint8_t* codes,
                                                                                              const uint16_t* scales, uint64_t q_heads,
                                                                                              uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                                                              uint64_t d, uint64_t rows) {
    nvidia_cuda12__zzprivate_attention_weights_tq(p, q, codes, scales, q_heads, kv_heads, head_dim, length, d, rows);
}

/* The mix: `Q` queries of one kv head a warp — over a prompt's rows 8, the rows' and the group's, so a lane decodes each element
 * of the tier once for all of them; for a decoded position's one row 1, a warp a query. Each query's sum is the same either way,
 * its products added from the first position on. */
static __device__ __forceinline__ void nvidia_cuda12__zzprivate_attention_mix_tq(float* res, const float* p, const uint8_t* codes,
                                                                                  const uint16_t* scales, uint64_t q_heads,
                                                                                  uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                                                  uint64_t d, uint64_t rows, const uint32_t Q) {
    __shared__ uint16_t lev[256];
    const uint32_t lid = (uint32_t)nn__silicon__lane(), lanes = (uint32_t)nn__silicon__lanes();
    for (uint32_t e = lid; e < (d == 4u ? 16u : 256u); e += lanes) lev[e] = nn__turboquant__zzabi_levels(d)[e];
    __syncthreads();
    const uint32_t lane = lid % NVIDIA_CUDA12__ZZPACKAGE_WARP, W = NVIDIA_CUDA12__ZZPACKAGE_WARP;
    const uint64_t group = q_heads / kv_heads, row_bytes = head_dim * d / 8u, J = rows * group, q_tiles = (J + Q - 1u) / Q;
    const uint64_t e_tiles = head_dim / W, items = kv_heads * q_tiles * e_tiles;
    const uint32_t mask = (1u << d) - 1u;
    for (uint64_t it = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS + lid / W; it < items;
         it += (uint64_t)nn__silicon__blocks() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS) {
        const uint64_t et = it % e_tiles, rest = it / e_tiles, qt = rest % q_tiles, g = rest / q_tiles;
        const uint64_t j0 = qt * Q, e = et * W + lane;
        const uint64_t at = e * d / 8u;
        const uint32_t sh = d == 4u ? 4u * (uint32_t)(e % 2u) : 0u;
        float sum[NVIDIA_CUDA12__ZZPRIVATE_MIX_Q], w[NVIDIA_CUDA12__ZZPRIVATE_MIX_Q], wn[NVIDIA_CUDA12__ZZPRIVATE_MIX_Q];
        uint64_t row[NVIDIA_CUDA12__ZZPRIVATE_MIX_Q];                 /* each query's weights, from `p`; past the last, none */
#pragma unroll
        for (uint32_t q = 0u; q < Q; ++q) {
            const bool live = j0 + q < J;
            row[q] = live ? nvidia_cuda12__zzprivate_query(j0 + q, g, group, q_heads) * length : 0u;
            sum[q] = 0.0f;
            w[q] = live && lane < length ? p[row[q] + lane] : 0.0f;
        }
        uint8_t cb[NVIDIA_CUDA12__ZZPACKAGE_WARP], cn[NVIDIA_CUDA12__ZZPACKAGE_WARP];
        uint16_t sc[NVIDIA_CUDA12__ZZPACKAGE_WARP], sn[NVIDIA_CUDA12__ZZPACKAGE_WARP];
#pragma unroll
        for (uint32_t i = 0u; i < W; ++i) {
            cb[i] = i < length ? codes[(i * kv_heads + g) * row_bytes + at] : (uint8_t)0u;
            sc[i] = i < length ? scales[i * kv_heads + g] : (uint16_t)0u;
        }
        for (uint64_t t0 = 0u; t0 < length; t0 += W) {
            const uint64_t t1 = t0 + W;
#pragma unroll
            for (uint32_t q = 0u; q < Q; ++q) wn[q] = j0 + q < J && t1 + lane < length ? p[row[q] + t1 + lane] : 0.0f;
#pragma unroll
            for (uint32_t i = 0u; i < W; ++i) {
                cn[i] = t1 + i < length ? codes[((t1 + i) * kv_heads + g) * row_bytes + at] : (uint8_t)0u;
                sn[i] = t1 + i < length ? scales[(t1 + i) * kv_heads + g] : (uint16_t)0u;
            }
#pragma unroll
            for (uint32_t i = 0u; i < W; ++i) {
                const float element = nn__silicon__half_to_float(sc[i]) * nn__silicon__half_to_float(lev[(cb[i] >> sh) & mask]);
#pragma unroll
                for (uint32_t q = 0u; q < Q; ++q) {
                    const float pt = __shfl_sync(0xFFFFFFFFu, w[q], (int)i);
                    if (t0 + i < length) sum[q] += pt * element;
                }
            }
#pragma unroll
            for (uint32_t q = 0u; q < Q; ++q) w[q] = wn[q];
#pragma unroll
            for (uint32_t i = 0u; i < W; ++i) { cb[i] = cn[i]; sc[i] = sn[i]; }
        }
#pragma unroll
        for (uint32_t q = 0u; q < Q; ++q) {
            const uint64_t j = j0 + q;
            if (j < J) res[nvidia_cuda12__zzprivate_query(j, g, group, q_heads) * NN__ATTENTION__RESIDUAL_FLOATS(head_dim) + 2u + e] = sum[q];
        }
    }
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_mix_tq(float* res, const float* p, const uint8_t* codes,
                                                                                          const uint16_t* scales, uint64_t q_heads,
                                                                                          uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                                                          uint64_t d) {
    nvidia_cuda12__zzprivate_attention_mix_tq(res, p, codes, scales, q_heads, kv_heads, head_dim, length, d, 1u, 1u);
}
static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_mix_tq_rows(float* res, const float* p, const uint8_t* codes,
                                                                                               const uint16_t* scales, uint64_t q_heads,
                                                                                               uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                                                               uint64_t d, uint64_t rows) {
    nvidia_cuda12__zzprivate_attention_mix_tq(res, p, codes, scales, q_heads, kv_heads, head_dim, length, d, rows,
                                              NVIDIA_CUDA12__ZZPRIVATE_MIX_Q);
}

/* 4 or 8 bits, whole slices of 64 of a head's columns, whole groups, `q` on 16 bytes and the codes on 8; anything else is nn's. */
static inline bool nvidia_cuda12__zzprivate_tq_fits(const void* q, const void* codes, uint64_t q_heads, uint64_t kv_heads,
                                                    uint64_t head_dim, uint64_t length, uint64_t d) {
    return length != 0u && (d == 4u || d == 8u) && kv_heads != 0u && q_heads % kv_heads == 0u && head_dim != 0u && head_dim % 64u == 0u
        && ((uint64_t)(uintptr_t)q) % 16u == 0u && ((uint64_t)(uintptr_t)codes) % 8u == 0u;
}
static void nvidia_cuda12__override_attention_weights_tq(float* p, float* res, const uint16_t* q, const uint8_t* codes, const uint16_t* scales,
                                                         uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d) {
    if (!nvidia_cuda12__zzprivate_tq_fits(q, codes, q_heads, kv_heads, head_dim, length, d)) {
        nn__attention__zzabi_launch_weights_tq(p, res, q, codes, scales, q_heads, kv_heads, head_dim, length, d);
        return;
    }
    const uint64_t items = kv_heads * ((q_heads / kv_heads + 15u) / 16u) * ((length + 63u) / 64u);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    uint64_t one = 1u;
    void* args[] = { &p, &q, &codes, &scales, &q_heads, &kv_heads, &head_dim, &length, &d, &one };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads, sizeof head_dim,
                             sizeof length, sizeof d, sizeof one };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_weights_tq, "nvidia_cuda12__zzprivate_weights_tq", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 10u);
    const uint32_t heads = q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)q_heads : NN__KERNELS__BLOCKS_MAX;
    void* wargs[] = { &p, &res, &q_heads, &head_dim, &length };
    const size_t wsizes[] = { sizeof p, sizeof res, sizeof q_heads, sizeof head_dim, sizeof length };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_residual_weights, "nvidia_cuda12__zzprivate_residual_weights", heads, 256u,
                        wargs, wsizes, 5u);
}
static void nvidia_cuda12__override_attention_residual_mix_tq(float* res, const float* p, const uint8_t* codes, const uint16_t* scales,
                                                              uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                              uint64_t d) {
    if (!nvidia_cuda12__zzprivate_tq_fits(res, codes, q_heads, kv_heads, head_dim, length, d) || head_dim % NVIDIA_CUDA12__ZZPACKAGE_WARP != 0u) {
        nn__attention__zzabi_launch_residual_mix_tq(res, p, codes, scales, q_heads, kv_heads, head_dim, length, d);
        return;
    }
    const uint64_t items = q_heads * (head_dim / NVIDIA_CUDA12__ZZPACKAGE_WARP);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    void* args[] = { &res, &p, &codes, &scales, &q_heads, &kv_heads, &head_dim, &length, &d };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads, sizeof head_dim,
                             sizeof length, sizeof d };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_mix_tq, "nvidia_cuda12__zzprivate_mix_tq", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 9u);
}

/* ⭐ …AND OVER A PROMPT'S ROWS — every row the whole tier: the scores over all the rows' queries at once, each row's heads' second
 *   half, and the mix 8 queries a warp; a row's weights and residual are the one-row doors' above. */
static void nvidia_cuda12__override_attention_weights_tq_rows(float* p, float* res, const uint16_t* q, const uint8_t* codes,
                                                              const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads,
                                                              uint64_t head_dim, uint64_t length, uint64_t d, uint64_t rows) {
    if (rows == 0u || !nvidia_cuda12__zzprivate_tq_fits(q, codes, q_heads, kv_heads, head_dim, length, d)) {
        nn__attention__zzabi_launch_weights_tq_rows(p, res, q, codes, scales, q_heads, kv_heads, head_dim, length, d, rows);
        return;
    }
    const uint64_t items = kv_heads * ((rows * (q_heads / kv_heads) + 15u) / 16u) * ((length + 63u) / 64u);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    void* args[] = { &p, &q, &codes, &scales, &q_heads, &kv_heads, &head_dim, &length, &d, &rows };
    const size_t sizes[] = { sizeof p, sizeof q, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads, sizeof head_dim,
                             sizeof length, sizeof d, sizeof rows };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_weights_tq, "nvidia_cuda12__zzprivate_weights_tq", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 10u);
    uint64_t grow = 0u;
    const uint32_t heads = rows * q_heads < NN__KERNELS__BLOCKS_MAX ? (uint32_t)(rows * q_heads) : NN__KERNELS__BLOCKS_MAX;
    void* wargs[] = { &p, &res, &q_heads, &head_dim, &length, &rows, &grow };
    const size_t wsizes[] = { sizeof p, sizeof res, sizeof q_heads, sizeof head_dim, sizeof length, sizeof rows, sizeof grow };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_residual_weights_rows, "nvidia_cuda12__zzprivate_residual_weights_rows", heads,
                        256u, wargs, wsizes, 7u);
}
static void nvidia_cuda12__override_attention_residual_mix_tq_rows(float* res, const float* p, const uint8_t* codes, const uint16_t* scales,
                                                                   uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                                   uint64_t d, uint64_t rows) {
    if (rows == 0u || !nvidia_cuda12__zzprivate_tq_fits(res, codes, q_heads, kv_heads, head_dim, length, d)
     || head_dim % NVIDIA_CUDA12__ZZPACKAGE_WARP != 0u) {
        nn__attention__zzabi_launch_residual_mix_tq_rows(res, p, codes, scales, q_heads, kv_heads, head_dim, length, d, rows);
        return;
    }
    const uint64_t Q = NVIDIA_CUDA12__ZZPRIVATE_MIX_Q;
    const uint64_t items = kv_heads * ((rows * (q_heads / kv_heads) + Q - 1u) / Q) * (head_dim / NVIDIA_CUDA12__ZZPACKAGE_WARP);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    void* args[] = { &res, &p, &codes, &scales, &q_heads, &kv_heads, &head_dim, &length, &d, &rows };
    const size_t sizes[] = { sizeof res, sizeof p, sizeof codes, sizeof scales, sizeof q_heads, sizeof kv_heads, sizeof head_dim,
                             sizeof length, sizeof d, sizeof rows };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_mix_tq_rows, "nvidia_cuda12__zzprivate_mix_tq_rows", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 10u);
}

/* ── ⭐ ATTENTION OVER A SET ───────────────────────────────────────────────────────────────────────────────────────────
 * nn's `attention_set` (`nn/gpu/kernels/kernels.cuh`): every row's head `e` attends every row's, unmasked —
 * `o[r][e] = Σ_t softmax_t(q_r · k_t / √d) · v_t` over a vision tower's patches, q, k and v side by side in a row of `qkv`.
 * nn's body keeps a softmax a lane over its share of the keys and folds the lanes; here a WARP takes 16 queries of a head
 * and walks every key 16 at a time on the tensor cores, each row's largest score and sum carried along: the scores through
 * `mma.sync` over the head's columns 16 at a time (zeros past its width), then the 16 keys' weights, rounded to halves,
 * times their values, 8 columns a product. The operands are read from where they lie, as the causal scores' are, and the
 * warps of a block take neighbouring queries of one head, so they read the same keys. The weights' sum is in floats and
 * their products with the values in halves — a row's output within the bench's tolerance of nn's, not its bits. */
#define NVIDIA_CUDA12__ZZPRIVATE_SET_STEPS 8u      /* the head's columns in slices of 16: 128 at most */
#define NVIDIA_CUDA12__ZZPRIVATE_SET_TILES 16u     /* …and in products of 8 */

/* Two halves at `at` as one register, or zeros. */
static __device__ __forceinline__ uint32_t nvidia_cuda12__zzprivate_pair(const uint16_t* at, bool valid) {
    return valid ? *(const uint32_t*)(const void*)at : 0u;
}
/* Two floats rounded to halves, the first low — a product's operand. */
static __device__ __forceinline__ uint32_t nvidia_cuda12__zzprivate_halves(float lo, float hi) {
    return (uint32_t)__half_as_ushort(__float2half_rn(lo)) | ((uint32_t)__half_as_ushort(__float2half_rn(hi)) << 16);
}

static __global__ NVIDIA_CUDA12__ZZPRIVATE_ATT_BOUNDS void nvidia_cuda12__zzprivate_set(uint16_t* o, const uint16_t* qkv, uint64_t rows,
                                                                                        uint64_t heads, uint64_t head_dim, unsigned int* over) {
    const uint32_t lane = (uint32_t)nn__silicon__lane() % NVIDIA_CUDA12__ZZPACKAGE_WARP, g = lane / 4u, t = lane % 4u;
    const uint64_t H = heads * head_dim, stride = 3u * H, q_tiles = (rows + 15u) / 16u, items = heads * q_tiles;
    const uint32_t steps = (uint32_t)((head_dim + 15u) / 16u), tiles = (uint32_t)(head_dim / 8u);
    const float scale = 1.4426950408889634f / nn__silicon__sqrtf((float)(uint32_t)head_dim);    /* log2(e) / √d */
    for (uint64_t it = (uint64_t)nn__silicon__block() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS + nn__silicon__lane() / NVIDIA_CUDA12__ZZPACKAGE_WARP;
         it < items; it += (uint64_t)nn__silicon__blocks() * NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS) {
        const uint64_t e = it / q_tiles, ra = (it % q_tiles) * 16u + g, rb = ra + 8u;
        const uint16_t* qh = qkv + e * head_dim;              /* row r's query head at `qh + r · stride`, its key and value */
        const uint16_t* kh = qh + H;                          /* heads H and 2H further */
        const uint16_t* vh = qh + 2u * H;
        uint32_t a[NVIDIA_CUDA12__ZZPRIVATE_SET_STEPS][4];
#pragma unroll
        for (uint32_t s = 0u; s < NVIDIA_CUDA12__ZZPRIVATE_SET_STEPS; ++s) {
            const uint64_t c0 = 16u * s + 2u * t, c1 = c0 + 8u;
            a[s][0] = nvidia_cuda12__zzprivate_pair(qh + ra * stride + c0, ra < rows && c0 < head_dim);
            a[s][1] = nvidia_cuda12__zzprivate_pair(qh + rb * stride + c0, rb < rows && c0 < head_dim);
            a[s][2] = nvidia_cuda12__zzprivate_pair(qh + ra * stride + c1, ra < rows && c1 < head_dim);
            a[s][3] = nvidia_cuda12__zzprivate_pair(qh + rb * stride + c1, rb < rows && c1 < head_dim);
        }
        float acc[NVIDIA_CUDA12__ZZPRIVATE_SET_TILES][4];
#pragma unroll
        for (uint32_t j = 0u; j < NVIDIA_CUDA12__ZZPRIVATE_SET_TILES; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
        float top[2] = { -NN__KERNELS__INFINITY, -NN__KERNELS__INFINITY }, sum[2] = { 0.0f, 0.0f };   /* rows g, g + 8 */
        for (uint64_t k0 = 0u; k0 < rows; k0 += 16u) {
            /* ① the scores of 16 keys, two products of 8: lane `4g + t` holds keys `2t, 2t + 1` of each, rows g and g + 8 */
            float sc[2][4];
#pragma unroll
            for (uint32_t n = 0u; n < 2u; ++n) {
                sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.0f;
                const uint64_t key = k0 + 8u * n + g;
                const uint16_t* kr = kh + key * stride;
#pragma unroll
                for (uint32_t s = 0u; s < NVIDIA_CUDA12__ZZPRIVATE_SET_STEPS; ++s) {
                    if (s < steps) {
                        const uint64_t c0 = 16u * s + 2u * t, c1 = c0 + 8u;
                        nvidia_cuda12__zzpackage_mma(sc[n], a[s][0], a[s][1], a[s][2], a[s][3],
                                                     nvidia_cuda12__zzprivate_pair(kr + c0, key < rows && c0 < head_dim),
                                                     nvidia_cuda12__zzprivate_pair(kr + c1, key < rows && c1 < head_dim));
                    }
                }
            }
            /* ② each row's largest so far, over the four lanes that hold it; what came before scaled down to it */
            float best[2] = { top[0], top[1] };
#pragma unroll
            for (uint32_t n = 0u; n < 2u; ++n)
#pragma unroll
                for (uint32_t c = 0u; c < 4u; ++c) {
                    const uint64_t key = k0 + 8u * n + 2u * t + (c & 1u);
                    sc[n][c] = key < rows ? sc[n][c] * scale : -NN__KERNELS__INFINITY;
                    best[c >> 1] = fmaxf(best[c >> 1], sc[n][c]);
                }
#pragma unroll
            for (uint32_t h = 0u; h < 2u; ++h) {
                best[h] = fmaxf(best[h], __shfl_xor_sync(0xFFFFFFFFu, best[h], 1));
                best[h] = fmaxf(best[h], __shfl_xor_sync(0xFFFFFFFFu, best[h], 2));
            }
            const float keep0 = exp2f(top[0] - best[0]), keep1 = exp2f(top[1] - best[1]);
            top[0] = best[0];
            top[1] = best[1];
            sum[0] *= keep0;
            sum[1] *= keep1;
#pragma unroll
            for (uint32_t j = 0u; j < NVIDIA_CUDA12__ZZPRIVATE_SET_TILES; ++j) {
                acc[j][0] *= keep0; acc[j][1] *= keep0; acc[j][2] *= keep1; acc[j][3] *= keep1;
            }
            float w[2][4];
#pragma unroll
            for (uint32_t n = 0u; n < 2u; ++n)
#pragma unroll
                for (uint32_t c = 0u; c < 4u; ++c) {
                    w[n][c] = exp2f(sc[n][c] - top[c >> 1]);
                    sum[c >> 1] += w[n][c];
                }
            /* ③ the weights as a product's A — the scores' layout is A's, keys `2t` and `2t + 8` — times the values' 8
             *   columns a product: lane `4g + t` reads column `8j + g` of keys `2t, 2t + 1` and `2t + 8, 2t + 9` */
            const uint32_t p0 = nvidia_cuda12__zzprivate_halves(w[0][0], w[0][1]), p1 = nvidia_cuda12__zzprivate_halves(w[0][2], w[0][3]);
            const uint32_t p2 = nvidia_cuda12__zzprivate_halves(w[1][0], w[1][1]), p3 = nvidia_cuda12__zzprivate_halves(w[1][2], w[1][3]);
            const uint64_t ka = k0 + 2u * t, kb = ka + 8u;
#pragma unroll
            for (uint32_t j = 0u; j < NVIDIA_CUDA12__ZZPRIVATE_SET_TILES; ++j) {
                if (j < tiles) {
                    const uint16_t* vc = vh + 8u * j + g;
                    const uint32_t b0 = (uint32_t)(ka < rows ? vc[ka * stride] : 0u) | ((uint32_t)(ka + 1u < rows ? vc[(ka + 1u) * stride] : 0u) << 16);
                    const uint32_t b1 = (uint32_t)(kb < rows ? vc[kb * stride] : 0u) | ((uint32_t)(kb + 1u < rows ? vc[(kb + 1u) * stride] : 0u) << 16);
                    nvidia_cuda12__zzpackage_mma(acc[j], p0, p1, p2, p3, b0, b1);
                }
            }
        }
#pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            sum[h] += __shfl_xor_sync(0xFFFFFFFFu, sum[h], 1);
            sum[h] += __shfl_xor_sync(0xFFFFFFFFu, sum[h], 2);
        }
#pragma unroll
        for (uint32_t j = 0u; j < NVIDIA_CUDA12__ZZPRIVATE_SET_TILES; ++j) {
            if (j < tiles) {
                const uint64_t d = e * head_dim + 8u * j + 2u * t;
                if (ra < rows) {
                    o[ra * H + d] = nn__kernels__zzabi_to_half(acc[j][0] / sum[0], over);
                    o[ra * H + d + 1u] = nn__kernels__zzabi_to_half(acc[j][1] / sum[0], over);
                }
                if (rb < rows) {
                    o[rb * H + d] = nn__kernels__zzabi_to_half(acc[j][2] / sum[1], over);
                    o[rb * H + d + 1u] = nn__kernels__zzabi_to_half(acc[j][3] / sum[1], over);
                }
            }
        }
    }
}

/* What it takes: a head of whole products of 8 columns, 128 at most, and `qkv` on four bytes (its pairs' loads). */
static void nvidia_cuda12__override_attention_set(uint16_t* o, const uint16_t* qkv, uint64_t rows, uint64_t heads, uint64_t head_dim,
                                                  unsigned int* over) {
    if (rows == 0u || heads == 0u || head_dim == 0u || head_dim % 8u != 0u || head_dim > 16u * NVIDIA_CUDA12__ZZPRIVATE_SET_STEPS
     || ((uint64_t)(uintptr_t)qkv) % 4u != 0u || ((uint64_t)(uintptr_t)o) % 2u != 0u) {
        nn__attention__zzabi_launch_set(o, qkv, rows, heads, head_dim, over);
        return;
    }
    const uint64_t items = heads * ((rows + 15u) / 16u);
    const uint64_t want = (items + NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS - 1u) / NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS;
    const uint32_t blocks = want < 65535u ? (uint32_t)want : 65535u;
    void* args[] = { &o, &qkv, &rows, &heads, &head_dim, &over };
    const size_t sizes[] = { sizeof o, sizeof qkv, sizeof rows, sizeof heads, sizeof head_dim, sizeof over };
    nn__silicon__launch((const void*)nvidia_cuda12__zzprivate_set, "nvidia_cuda12__zzprivate_set", blocks,
                        NVIDIA_CUDA12__ZZPRIVATE_ATT_WARPS * NVIDIA_CUDA12__ZZPACKAGE_WARP, args, sizes, 6u);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_OVERRIDES_ATTENTION_CUH */
