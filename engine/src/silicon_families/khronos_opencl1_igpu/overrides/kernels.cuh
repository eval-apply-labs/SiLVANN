#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_OVERRIDES_KERNELS_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_OVERRIDES_KERNELS_CUH
/* ══ THE OVERRIDES' KERNELS — khronos_opencl1_igpu ════════════════════════════════════════════════════════════════════
 * The card side of the doors this family runs faster than nn's generic body (▶ doors.cuh beside this). They are written in
 * nn's card dialect, against nn's helpers and the family's lane functions only — no shuffles, which OpenCL C 1.2 does not
 * have, and no block-shared memory but the folds' — and `scripts/opencl1_kernel_source.py` reads this file after nn's card
 * side, works out every pointer's space as it does nn's, and writes a `__kernel` for each row of
 * `KHRONOS_OPENCL1_IGPU__OVERRIDE_ROWS`, called `<row>__kernel`, which the doors launch by name. Nothing here is compiled
 * into the family's binary.
 * ⭐ THE GEMVS AND THE ROTATION (NN-96) ARE NOT nn's BYTES: their products are nn's, added in an order of the family's own —
 *   a run of 16 lanes a row, folded together (`lanes_sum_each`); the butterfly (`lanes_hadamard`) — within the bench's
 *   bound of nn's host body, as `nvidia_cuda12`'s and `amd_rocm6_wave64`'s are. The two-input gemv runs the one-input
 *   kernel an input at a time, so its answers are the one-input door's bytes.
 * ⭐ EVERY OTHER OUTPUT IS nn's TO THE BIT: the same products added in the same order as nn's card body, and every sum
 *   across a block nn's fold of it over the same lanes (or over the lanes nn's door gives with the rest zeros, which the fold
 *   adds first, at its widest span). What is the family's own is what a lane keeps between rows and how many outputs a block
 *   makes:
 *   · KDA's steps — a lane's column of the state in its private memory for every row, read and written once a call
 *   · MLA's absorb — a weight loaded once for `R` rows on an integrated GPU, a row a work item on a discrete card
 *   · MLA's expand — a block's eight weight rows loaded once for `R` rows
 *   · the indexer's pool scores — a block `G` pools of a row
 *   · the mix over rows of their own — a value loaded once for `Q` heads
 * ⛳ TWO SHAPES, AN INTEGRATED GPU'S AND A DISCRETE CARD'S: an integrated GPU shares the host's memory and its bandwidth,
 *   so each load serves twice as many outputs there (`G`, `Q` 16 against 8; the expand's rows 32 against 16; the
 *   absorb's 16 rows against one) — fewer
 *   loads for fewer, wider work items. The doors choose by the device (▶ doors.cuh). `REASONED` from that premise; tested
 *   on the RTX 5060 Ti through NVIDIA's OpenCL, each shape forced, never on an integrated GPU. */

/* ⭐ KDA's STEPS OVER A PROMPT'S ROWS AT A HEAD OF 128 — nn's `kda_steps_decayed`, a block a head and a lane a column. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_kda_steps(float* S, const uint16_t* conved, const float* eg, const uint16_t* b,
                                                                        const uint16_t* gate, const uint16_t* w, uint16_t* out, uint64_t heads,
                                                                        float eps, uint64_t rows, unsigned int* over) {
    const uint64_t D = 128u, j = nn__silicon__lane(), width = heads * D;
    const uint16_t* qs = conved;
    const uint16_t* ks = conved + rows * width;
    const uint16_t* vs = conved + 2u * rows * width;
    for (uint64_t h = nn__silicon__block(); h < heads; h += nn__silicon__blocks()) {
        float* Sh = S + h * D * D;
        float col[128];
        for (uint64_t i = 0u; i < D; ++i) col[i] = Sh[i * D + j];
        for (uint64_t t = 0u; t < rows; ++t) {
            const uint64_t r = t * width + h * D;
            const uint16_t* q = qs + r;
            const uint16_t* k = ks + r;
            const float* e = eg + r;
            const float beta = 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[t * heads + h])));
            const float qi = nn__silicon__half_to_float(q[j]), ki = nn__silicon__half_to_float(k[j]);
            float qq = 0.0f, kk = 0.0f;
            qq += qi * qi;
            kk += ki * ki;
            const float inv_q = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(qq) + 1e-6f) / nn__silicon__sqrtf((float)D);
            const float inv_k = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(kk) + 1e-6f);
            float kv = 0.0f;
            for (uint64_t i = 0u; i < D; ++i) {
                col[i] *= e[i];
                kv += col[i] * nn__silicon__half_to_float(k[i]) * inv_k;
            }
            const float delta = (nn__silicon__half_to_float(vs[r + j]) - kv) * beta;
            float o = 0.0f;
            for (uint64_t i = 0u; i < D; ++i) {
                const float s = col[i] + nn__silicon__half_to_float(k[i]) * inv_k * delta;
                col[i] = s;
                o += s * nn__silicon__half_to_float(q[i]);
            }
            const float oj = nn__silicon__half_to_float(nn__kernels__zzabi_to_half(o * inv_q, over));   /* rounded, as nn's readout is */
            float oo = 0.0f;
            oo += oj * oj;
            const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oo) / (float)D + eps);
            const float gz = nn__silicon__half_to_float(gate[r + j]);
            out[r + j] = nn__kernels__zzabi_to_half(oj * scale * nn__silicon__half_to_float(w[j]) / (1.0f + nn__silicon__expf(-gz)), over);
        }
        for (uint64_t i = 0u; i < D; ++i) Sh[i * D + j] = col[i];
    }
}

/* ⭐ MLA's ABSORB — nn's `absorb` and `absorb_rows`: an item a head's `lanes` columns over `R` rows, a lane a column, its sums
 *   `Σᵢ W[i][c] · q[i]`, `i` rising, as nn's lane adds them. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_absorb(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads,
                                                                     uint64_t nope, uint64_t latent, uint64_t stride, float scale,
                                                                     uint64_t rows, unsigned int* over, const uint64_t R) {
    const uint64_t n = nn__silicon__lanes(), cgroups = (latent + n - 1u) / n, hc_all = heads * cgroups;
    const uint64_t items = hc_all * ((rows + R - 1u) / R);
    for (uint64_t x = nn__silicon__block(); x < items; x += nn__silicon__blocks()) {
        const uint64_t hc = x % hc_all, t0 = (x / hc_all) * R, h = hc / cgroups, c = (hc % cgroups) * n + nn__silicon__lane();
        if (c >= latent) continue;
        const uint16_t* w = W + h * stride * latent + c;
        float sum[16];
        for (uint64_t r = 0u; r < R; ++r) sum[r] = 0.0f;
        for (uint64_t i = 0u; i < nope; ++i) {
            const float wf = nn__silicon__half_to_float(w[i * latent]);
            for (uint64_t r = 0u; r < R; ++r)
                if (t0 + r < rows) sum[r] += wf * nn__silicon__half_to_float(q[(t0 + r) * heads * nope + h * nope + i]);
        }
        for (uint64_t r = 0u; r < R; ++r)
            if (t0 + r < rows) out[(t0 + r) * heads * latent + h * latent + c] = nn__kernels__zzabi_to_half(sum[r] * scale, over);
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_absorb16(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads,
                                                                       uint64_t nope, uint64_t latent, uint64_t stride, float scale,
                                                                       uint64_t rows, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_absorb(out, q, W, heads, nope, latent, stride, scale, rows, over, 16u);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_absorb1(uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads,
                                                                      uint64_t nope, uint64_t latent, uint64_t stride, float scale,
                                                                      uint64_t rows, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_absorb(out, q, W, heads, nope, latent, stride, scale, rows, over, 1u);
}

/* ⭐ MLA's EXPAND OVER A LATENT OF AT MOST 512 — nn's `expand` and `expand_rows`: a block of 256 lanes eight of a head's outputs
 *   over `R` rows, lane `L` each output's part over its columns `L` and `L + 256` as nn's lane makes it, and the eight parts
 *   folded over the block as nn folds each — its weights loaded once for the `R` rows. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_expand(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads,
                                                                     uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset,
                                                                     uint64_t rows, unsigned int* over, const uint64_t R) {
    const uint64_t J = 8u, L = nn__silicon__lane(), n = nn__silicon__lanes();
    const uint64_t per = width / J, outs = heads * per, items = outs * ((rows + R - 1u) / R);
    for (uint64_t x = nn__silicon__block(); x < items; x += nn__silicon__blocks()) {
        const uint64_t hj = x % outs, t0 = (x / outs) * R, h = hj / per, j0 = (hj % per) * J;
        float wv[8][2];
        for (uint64_t jj = 0u; jj < J; ++jj) {
            const uint16_t* row = W + (h * stride + offset + j0 + jj) * latent;
            for (uint64_t c = 0u; c < 2u; ++c) wv[jj][c] = L + n * c < latent ? nn__silicon__half_to_float(row[L + n * c]) : 0.0f;
        }
        for (uint64_t r = 0u; r < R && t0 + r < rows; ++r) {
            const uint16_t* xr = o + (t0 + r) * heads * latent + h * latent;
            float xv[2];
            for (uint64_t c = 0u; c < 2u; ++c) xv[c] = L + n * c < latent ? nn__silicon__half_to_float(xr[L + n * c]) : 0.0f;
            float parts[8];
            for (uint64_t jj = 0u; jj < J; ++jj) {
                float part = 0.0f;
                for (uint64_t c = 0u; c < 2u; ++c)
                    if (L + n * c < latent) part += wv[jj][c] * xv[c];
                parts[jj] = part;
            }
            nn__silicon__lanes_sums(parts, (uint32_t)J);
            if (L == 0u)
                for (uint64_t jj = 0u; jj < J; ++jj)
                    out[(t0 + r) * heads * width + h * width + j0 + jj] = nn__kernels__zzabi_to_half(parts[jj], over);
        }
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_expand16(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads,
                                                                       uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset,
                                                                       uint64_t rows, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_expand(out, o, W, heads, width, latent, stride, offset, rows, over, 16u);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_expand32(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads,
                                                                       uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset,
                                                                       uint64_t rows, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_expand(out, o, W, heads, width, latent, stride, offset, rows, over, 32u);
}

/* ⭐ THE INDEXER's POOL SCORES — nn's `index_scores` and `index_scores_rows`: a block of 64 lanes `G` pools of a row, lane `h` a
 *   head's chain of products over each as nn's lane makes it, and each pool's 64 parts folded as nn folds them. Row `u` is
 *   position `len0 + u − 1`, scoring its `(len0 + u) / kpool` finished pools; the one-row door is one row at `kpool` 1. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_index_scores(float* scores, uint64_t scores_row, const uint16_t* q,
                                                                           const uint16_t* w, const uint16_t* pooled, uint64_t heads,
                                                                           uint64_t dim, uint64_t len0, uint64_t kpool, uint64_t rows,
                                                                           float scale, const uint64_t G) {
    const uint64_t h = nn__silicon__lane(), most = (len0 + rows - 1u) / kpool, groups = (most + G - 1u) / G;
    const float per_head = 1.0f / nn__silicon__sqrtf((float)heads);
    for (uint64_t x = nn__silicon__block(); x < rows * groups; x += nn__silicon__blocks()) {
        const uint64_t u = x / groups, base = (x - u * groups) * G, pools = (len0 + u) / kpool;
        if (base >= pools) continue;                                     /* the same for every lane of the block */
        const uint16_t* qh = q + u * heads * dim + h * dim;
        const float wh = h < heads ? nn__silicon__half_to_float(w[u * heads + h]) : 0.0f;
        float parts[16];
        for (uint64_t i = 0u; i < G; ++i) {
            const uint64_t p = base + i;
            float part = 0.0f;
            if (h < heads && p < pools) {
                float dot = 0.0f;
                for (uint64_t c = 0u; c < dim; ++c)
                    dot += nn__silicon__half_to_float(qh[c]) * nn__silicon__half_to_float(pooled[p * dim + c]);
                dot *= scale;
                part += (dot > 0.0f ? dot : 0.0f) * wh * per_head;
            }
            parts[i] = part;
        }
        nn__silicon__lanes_sums(parts, (uint32_t)G);
        if (h == 0u)
            for (uint64_t i = 0u; i < G; ++i)
                if (base + i < pools) scores[u * scores_row + base + i] = parts[i];
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_index_scores8(float* scores, uint64_t scores_row, const uint16_t* q,
                                                                            const uint16_t* w, const uint16_t* pooled, uint64_t heads,
                                                                            uint64_t dim, uint64_t len0, uint64_t kpool, uint64_t rows,
                                                                            float scale) {
    khronos_opencl1_igpu__zzprivate_index_scores(scores, scores_row, q, w, pooled, heads, dim, len0, kpool, rows, scale, 8u);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_index_scores16(float* scores, uint64_t scores_row, const uint16_t* q,
                                                                             const uint16_t* w, const uint16_t* pooled, uint64_t heads,
                                                                             uint64_t dim, uint64_t len0, uint64_t kpool, uint64_t rows,
                                                                             float scale) {
    khronos_opencl1_igpu__zzprivate_index_scores(scores, scores_row, q, w, pooled, heads, dim, len0, kpool, rows, scale, 16u);
}

/* ⭐ THE MIX FOR `rows` QUERIES, EACH OVER VALUES OF ITS OWN — nn's `mix_rows`: an item `Q` of a row's query heads of one kv
 *   head by the block's `lanes` columns, a lane a column, each sum over the row's positions in order as nn's lane adds it. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_mix_rows(uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row,
                                                                       const uint16_t* v, uint64_t v_row, const uint32_t* lens, uint64_t q_heads,
                                                                       uint64_t kv_heads, uint64_t head_dim, uint64_t rows, unsigned int* over,
                                                                       const uint64_t Q) {
    const uint64_t n = nn__silicon__lanes(), group = q_heads / kv_heads, row = kv_heads * head_dim;
    const uint64_t q_tiles = (group + Q - 1u) / Q, e_tiles = (head_dim + n - 1u) / n, per = kv_heads * q_tiles * e_tiles;
    for (uint64_t b = nn__silicon__block(); b < rows * per; b += nn__silicon__blocks()) {
        const uint64_t u = b / per, rest = b - u * per, et = rest % e_tiles, qt = (rest / e_tiles) % q_tiles, g = rest / (e_tiles * q_tiles);
        const uint64_t length = (uint64_t)lens[u], e = et * n + nn__silicon__lane(), h0 = g * group + qt * Q;
        const uint64_t qn = group - qt * Q < Q ? group - qt * Q : Q;
        if (e >= head_dim) continue;
        const float* pu = p + u * p_row;
        const uint16_t* vg = v + u * v_row + g * head_dim + e;
        float sum[16];
        for (uint64_t a = 0u; a < Q; ++a) sum[a] = 0.0f;
        for (uint64_t t = 0u; t < length; ++t) {
            const float vf = nn__silicon__half_to_float(vg[t * row]);
            for (uint64_t a = 0u; a < Q; ++a)
                if (a < qn) sum[a] += pu[(h0 + a) * length + t] * vf;
        }
        for (uint64_t a = 0u; a < qn; ++a) o[u * o_row + (h0 + a) * head_dim + e] = nn__kernels__zzabi_to_half(sum[a], over);
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_mix_rows8(uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row,
                                                                        const uint16_t* v, uint64_t v_row, const uint32_t* lens, uint64_t q_heads,
                                                                        uint64_t kv_heads, uint64_t head_dim, uint64_t rows, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_mix_rows(o, o_row, p, p_row, v, v_row, lens, q_heads, kv_heads, head_dim, rows, over, 8u);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_mix_rows16(uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row,
                                                                         const uint16_t* v, uint64_t v_row, const uint32_t* lens, uint64_t q_heads,
                                                                         uint64_t kv_heads, uint64_t head_dim, uint64_t rows, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_mix_rows(o, o_row, p, p_row, v, v_row, lens, q_heads, kv_heads, head_dim, rows, over, 16u);
}

/* The kernels, a row each — the body a `__kernel` runs and its parameters, as nn's door rows give them: read by
 * `scripts/opencl1_kernel_source.py`, which names each kernel `<body>__kernel`. */
#define KHRONOS_OPENCL1_IGPU__GEMV_RUN 16u
/* A run's part of one row of one matrix: its lanes' columns, sixteen bytes a step (▶ the gemv below). */
static __device__ inline float khronos_opencl1_igpu__zzprivate_gemv_part(const uint8_t* row, const uint16_t* x, uint64_t d, uint64_t row_bytes,
                                                                        uint64_t j, const float* levels) {
    const uint64_t W = KHRONOS_OPENCL1_IGPU__GEMV_RUN;
    float part = 0.0f;
    for (uint64_t at = 16u * j; at < row_bytes; at += 16u * W) {
        const uint32_t* cw = (const uint32_t*)(row + at);
        const uint32_t* xw = (const uint32_t*)(x + at * 8u / d);
        if (d == 8u) {
            for (uint64_t q = 0u; q < 4u; ++q) {
                const uint32_t c = cw[q], a = xw[2u * q], b = xw[2u * q + 1u];
                part += levels[c & 0xFFu] * nn__silicon__half_to_float((uint16_t)(a & 0xFFFFu));
                part += levels[(c >> 8) & 0xFFu] * nn__silicon__half_to_float((uint16_t)(a >> 16));
                part += levels[(c >> 16) & 0xFFu] * nn__silicon__half_to_float((uint16_t)(b & 0xFFFFu));
                part += levels[c >> 24] * nn__silicon__half_to_float((uint16_t)(b >> 16));
            }
        } else {
            for (uint64_t q = 0u; q < 4u; ++q) {
                const uint32_t c = cw[q];
                for (uint64_t e = 0u; e < 4u; ++e) {
                    const uint32_t a = xw[4u * q + e];
                    part += levels[256u + ((c >> (8u * e)) & 0xFu)] * nn__silicon__half_to_float((uint16_t)(a & 0xFFFFu));
                    part += levels[256u + ((c >> (8u * e + 4u)) & 0xFu)] * nn__silicon__half_to_float((uint16_t)(a >> 16));
                }
            }
        }
    }
    return part;
}
/* ⭐ THE GEMV, A RUN OF LANES A ROW — nn's `turboquant_gemv_groups` (and the gemv, one group), at 4 and 8 bits. nn's body
 *   makes a block one row and folds its 256 lanes for it, eight barriers an output; here a block is sixteen runs of 16
 *   lanes, a run a row: a lane reads sixteen bytes of the row's codes a step, the run's reads one stretch of the row, `x`'s
 *   halves two to a word, and the run's sixteen parts are folded together — all the block's rows at once
 *   (`lanes_sum_each`). The levels come as floats from the card's memory (`levels`: the 8 bits' 256, then the 4 bits' 16),
 *   not from the program's constants, which the card serves one lane at a time when the lanes ask for different ones.
 *   The products are nn's; the order they are added in is not, so an answer is within the bench's bound of nn's, not its
 *   bytes. Rows a multiple of sixteen bytes, codes and `x` on sixteen (▶ the door's fit). */
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemv(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                                   const float* levels, unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), W = KHRONOS_OPENCL1_IGPU__GEMV_RUN, runs = nn__silicon__lanes() / W, j = lane % W;
    uint64_t units = 0u;
    for (uint64_t i = 0u; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) units += (g.rows[i] + runs - 1u) / runs;
    for (uint64_t u = nn__silicon__block(); u < units; u += nn__silicon__blocks()) {
        uint64_t i = 0u, first = 0u;
        while (u >= first + (g.rows[i] + runs - 1u) / runs) { first += (g.rows[i] + runs - 1u) / runs; ++i; }
        const uint64_t d = g.d[i], r = (u - first) * runs + lane / W, row_bytes = cols * d / 8u;
        const float part = r < g.rows[i] ? khronos_opencl1_igpu__zzprivate_gemv_part((const uint8_t*)(uintptr_t)g.codes[i] + r * row_bytes, x, d,
                                                                                    row_bytes, j, levels)
                                         : 0.0f;
        const float sum = khronos_opencl1_igpu__zzpackage_lanes_sum_each(part, (uint32_t)W);
        if (j == 0u && r < g.rows[i]) {
            const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
            const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
            out[g.out_at[i] + r] = nn__kernels__zzabi_to_half(sum * scale, over);
        }
    }
}

/* ⭐ AND THE WEIGHTED SUM ONTO THE RESIDUAL — nn's `turboquant_gemv_groups_sum`: a run a row, every matrix's part of it at
 *   that row (`x_i` at `x + i·cols`), scaled and weighted in the run's lanes, folded once. `residual` may be `out`: a row's
 *   one writer reads its element first. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemv_sum(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                                       const uint16_t* residual, uint64_t rows, uint64_t cols,
                                                                       const float* levels, unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), W = KHRONOS_OPENCL1_IGPU__GEMV_RUN, runs = nn__silicon__lanes() / W, j = lane % W;
    const uint64_t units = (rows + runs - 1u) / runs;
    for (uint64_t u = nn__silicon__block(); u < units; u += nn__silicon__blocks()) {
        const uint64_t r = u * runs + lane / W;
        float part = 0.0f;
        if (r < rows)
            for (uint64_t i = 0u; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) {
                const uint64_t d = g.d[i], row_bytes = cols * d / 8u;
                const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
                const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
                part += khronos_opencl1_igpu__zzprivate_gemv_part((const uint8_t*)(uintptr_t)g.codes[i] + r * row_bytes, x + i * cols, d, row_bytes,
                                                                  j, levels)
                      * scale * nn__silicon__half_to_float(w[g.out_at[i]]);
            }
        const float sum = khronos_opencl1_igpu__zzpackage_lanes_sum_each(part, (uint32_t)W);
        if (j == 0u && r < rows) out[r] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(residual[r]) + sum, over);
    }
}

/* ⭐ THE HADAMARD ROTATION AS THE BUTTERFLY — nn's `hadamard_rotate`, `out = H·(s ⊙ in) / √b` over each block of the ladder
 *   (the 512s, then a 256, then a 128), a block of 256 lanes a block of the ladder: the butterfly's stages through the fold's
 *   scratch (`lanes_hadamard`), a 512's first stage in the lane (its halves' sum and difference, each half then a 256's).
 *   nn's body applies each output's row of the matrix directly — `b` terms an output, a lane an output — which leaves the
 *   card waiting on the lanes' chains; the butterfly is `log b` stages. The terms are nn's, added in the butterfly's order,
 *   so an answer is within the bench's bound of nn's, not its bytes. `out` apart from `in`, 256 lanes (▶ the door's fit). */
static __device__ inline void khronos_opencl1_igpu__zzprivate_rotate(uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n,
                                                                     unsigned int* over) {
    const uint64_t B = NN__HADAMARD__WHOLE_BELOW, whole = n < B ? 0u : (n / B) * B, rest = n - whole;
    const uint64_t blocks = whole / B + (whole != 0u && rest >= 256u ? 1u : 0u)
                          + (whole != 0u && rest - (rest >= 256u ? 256u : 0u) >= 128u ? 1u : 0u) + (whole == 0u ? 1u : 0u);
    const uint32_t l = (uint32_t)nn__silicon__lane();
    for (uint64_t k = nn__silicon__block(); k < blocks; k += nn__silicon__blocks()) {
        uint64_t a = k * B, b = B;
        if (whole == 0u) { a = 0u; b = n; }
        else if (a >= whole) {
            if (rest >= 256u && a == whole) b = 256u;
            else { a = whole + (rest >= 256u ? 256u : 0u); b = 128u; }
        }
        const float norm = 1.0f / nn__silicon__sqrtf((float)b);
        if (b == B) {
            const float lo = l < 256u ? nn__silicon__half_to_float(in[a + l]) * nn__silicon__half_to_float(sign[l]) : 0.0f;
            const float hi = l < 256u ? nn__silicon__half_to_float(in[a + 256u + l]) * nn__silicon__half_to_float(sign[256u + l]) : 0.0f;
            const float y0 = khronos_opencl1_igpu__zzpackage_lanes_hadamard(lo + hi, 256u), y1 = khronos_opencl1_igpu__zzpackage_lanes_hadamard(lo - hi, 256u);
            if (l < 256u) {
                out[a + l] = nn__kernels__zzabi_to_half(y0 * norm, over);
                out[a + 256u + l] = nn__kernels__zzabi_to_half(y1 * norm, over);
            }
        } else {
            const float v = l < b ? nn__silicon__half_to_float(in[a + l]) * nn__silicon__half_to_float(sign[l]) : 0.0f;
            const float y = khronos_opencl1_igpu__zzpackage_lanes_hadamard(v, (uint32_t)b);
            if (l < b) out[a + l] = nn__kernels__zzabi_to_half(y * norm, over);
        }
    }
}

#define KHRONOS_OPENCL1_IGPU__OVERRIDE_ROWS(X) \
    X(khronos_opencl1_igpu__zzprivate_gemv, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, const float* levels, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_gemv_sum, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w, const uint16_t* residual, uint64_t rows, uint64_t cols, const float* levels, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_rotate, (uint16_t* out, const uint16_t* in, const uint16_t* sign, uint64_t n, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_kda_steps, (float* S, const uint16_t* conved, const float* eg, const uint16_t* b, const uint16_t* gate, const uint16_t* w, uint16_t* out, uint64_t heads, float eps, uint64_t rows, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_absorb1, (uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads, uint64_t nope, uint64_t latent, uint64_t stride, float scale, uint64_t rows, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_absorb16, (uint16_t* out, const uint16_t* q, const uint16_t* W, uint64_t heads, uint64_t nope, uint64_t latent, uint64_t stride, float scale, uint64_t rows, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_expand16, (uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads, uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset, uint64_t rows, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_expand32, (uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads, uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset, uint64_t rows, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_index_scores8, (float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w, const uint16_t* pooled, uint64_t heads, uint64_t dim, uint64_t len0, uint64_t kpool, uint64_t rows, float scale)) \
    X(khronos_opencl1_igpu__zzprivate_index_scores16, (float* scores, uint64_t scores_row, const uint16_t* q, const uint16_t* w, const uint16_t* pooled, uint64_t heads, uint64_t dim, uint64_t len0, uint64_t kpool, uint64_t rows, float scale)) \
    X(khronos_opencl1_igpu__zzprivate_mix_rows8, (uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row, const uint16_t* v, uint64_t v_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t rows, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_mix_rows16, (uint16_t* o, uint64_t o_row, const float* p, uint64_t p_row, const uint16_t* v, uint64_t v_row, const uint32_t* lens, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t rows, unsigned int* over))

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL1_IGPU_OVERRIDES_KERNELS_CUH */
