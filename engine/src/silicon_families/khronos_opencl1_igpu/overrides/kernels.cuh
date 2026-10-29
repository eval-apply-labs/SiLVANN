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
 *   · MLA's expand — a run of 32 lanes an output, its weights loaded once for `R` rows, nn's fold's first steps in a lane
 *   · the norm of a row of halves — a lane's sixteen reads in flight
 *   · the indexer's pool scores — a block `G` pools of a row
 *   · the mix over rows of their own — a value loaded once for `Q` heads
 *   · KDA's step a row — a lane's column in its private memory between the step's two passes
 *   · a site's map's logits — a lane's sixteen reads in flight; its `pre` — the mix in the first lane's registers
 *   · a one-query attention's grouped scores — a tile of one, two or four positions an item, not eight
 *   · DeepSeek's map's logits and norm over floats — a lane's sixteen reads in flight; its router — each score made once
 *   · a sampler's top `k` — a bar first, then nn's rounds over the values that reach it alone
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

/* ⭐ MLA's EXPAND OVER A LATENT OF AT MOST 512 — nn's `expand` and `expand_rows`: an output nn's fold over 256 lanes of lane
 *   `L`'s part over its columns `L` and `L + 256`. Here a run of 32 lanes makes an output, eight a block, its weights loaded
 *   once for `R` rows: lane `m` of a run makes the parts of nn's lanes `m + 32·q` (`q` below eight) and adds them as the fold's
 *   first three steps add them (lane `i` and lane `i + 128`, then `+ 64`, then `+ 32`), and the run's fold (`lanes_sum_each`)
 *   is the fold's last five steps over the same lanes — the same additions in the same order, so nn's bits, with five of the
 *   block's waits a row for eight outputs where folding each over the block is eight. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_expand(uint16_t* out, const uint16_t* o, const uint16_t* W, uint64_t heads,
                                                                     uint64_t width, uint64_t latent, uint64_t stride, uint64_t offset,
                                                                     uint64_t rows, unsigned int* over, const uint64_t R) {
    const uint64_t J = 8u, RUN = 32u, l = nn__silicon__lane(), m = l % RUN, jr = l / RUN;
    const uint64_t per = width / J, outs = heads * per, items = outs * ((rows + R - 1u) / R);
    for (uint64_t x = nn__silicon__block(); x < items; x += nn__silicon__blocks()) {
        const uint64_t hj = x % outs, t0 = (x / outs) * R, h = hj / per, j = (hj % per) * J + jr;
        const uint16_t* row = W + (h * stride + offset + j) * latent;
        float wv[16];                                       /* nn's lane `m + 32·q`'s two columns at `2q`, `2q + 1` */
        for (uint64_t q = 0u; q < 8u; ++q)
            for (uint64_t c = 0u; c < 2u; ++c) {
                const uint64_t col = m + RUN * q + 256u * c;
                wv[2u * q + c] = col < latent ? nn__silicon__half_to_float(row[col]) : 0.0f;
            }
        for (uint64_t r = 0u; r < R && t0 + r < rows; ++r) {
            const uint16_t* xr = o + (t0 + r) * heads * latent + h * latent;
            float p[8];
            for (uint64_t q = 0u; q < 8u; ++q) {
                float part = 0.0f;
                for (uint64_t c = 0u; c < 2u; ++c) {
                    const uint64_t col = m + RUN * q + 256u * c;
                    if (col < latent) part += wv[2u * q + c] * nn__silicon__half_to_float(xr[col]);
                }
                p[q] = part;
            }
            const float a0 = p[0] + p[4], a1 = p[1] + p[5], a2 = p[2] + p[6], a3 = p[3] + p[7];
            const float sum = khronos_opencl1_igpu__zzpackage_lanes_sum_each((a0 + a2) + (a1 + a3), (uint32_t)RUN);
            if (m == 0u) out[(t0 + r) * heads * width + h * width + j] = nn__kernels__zzabi_to_half(sum, over);
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

/* ⭐ THE SAME, `R` ROWS A RUN — a lane reads `x`'s halves once for its `R` rows' codes, where a run a row reads them again
 *   for every row: `x` is four bytes for each byte of 4-bit codes, all through the cache. Each row's part is the one-row
 *   kernel's — the same products added in the same order by the same lane, folded by the same run — so the same bytes;
 *   the door chooses `R` by the matrices' rows (▶ doors.cuh). */
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemv_rows(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                        uint64_t cols, const float* levels, unsigned int* over,
                                                                        const uint64_t R) {
    const uint64_t lane = nn__silicon__lane(), W = KHRONOS_OPENCL1_IGPU__GEMV_RUN, runs = nn__silicon__lanes() / W, j = lane % W;
    const uint64_t span = runs * R;
    uint64_t units = 0u;
    for (uint64_t i = 0u; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) units += (g.rows[i] + span - 1u) / span;
    for (uint64_t u = nn__silicon__block(); u < units; u += nn__silicon__blocks()) {
        uint64_t i = 0u, first = 0u;
        while (u >= first + (g.rows[i] + span - 1u) / span) { first += (g.rows[i] + span - 1u) / span; ++i; }
        const uint64_t d = g.d[i], r0 = (u - first) * span + (lane / W) * R, row_bytes = cols * d / 8u, rows = g.rows[i];
        const uint8_t* codes = (const uint8_t*)(uintptr_t)g.codes[i] + r0 * row_bytes;
        float part[4];
        for (uint64_t k = 0u; k < R; ++k) part[k] = 0.0f;
        for (uint64_t at = 16u * j; at < row_bytes; at += 16u * W) {
            const uint32_t* xw = (const uint32_t*)(x + at * 8u / d);
            uint32_t cw[4][4];
            for (uint64_t k = 0u; k < R; ++k)
                for (uint64_t q = 0u; q < 4u; ++q) cw[k][q] = r0 + k < rows ? ((const uint32_t*)(codes + k * row_bytes + at))[q] : 0u;
            if (d == 8u) {
                for (uint64_t q = 0u; q < 4u; ++q) {
                    const uint32_t a = xw[2u * q], b = xw[2u * q + 1u];
                    const float x0 = nn__silicon__half_to_float((uint16_t)(a & 0xFFFFu)), x1 = nn__silicon__half_to_float((uint16_t)(a >> 16));
                    const float x2 = nn__silicon__half_to_float((uint16_t)(b & 0xFFFFu)), x3 = nn__silicon__half_to_float((uint16_t)(b >> 16));
                    for (uint64_t k = 0u; k < R; ++k) {
                        const uint32_t c = cw[k][q];
                        part[k] += levels[c & 0xFFu] * x0;
                        part[k] += levels[(c >> 8) & 0xFFu] * x1;
                        part[k] += levels[(c >> 16) & 0xFFu] * x2;
                        part[k] += levels[c >> 24] * x3;
                    }
                }
            } else {
                for (uint64_t q = 0u; q < 4u; ++q)
                    for (uint64_t e = 0u; e < 4u; ++e) {
                        const uint32_t a = xw[4u * q + e];
                        const float x0 = nn__silicon__half_to_float((uint16_t)(a & 0xFFFFu)), x1 = nn__silicon__half_to_float((uint16_t)(a >> 16));
                        for (uint64_t k = 0u; k < R; ++k) {
                            const uint32_t c = cw[k][q];
                            part[k] += levels[256u + ((c >> (8u * e)) & 0xFu)] * x0;
                            part[k] += levels[256u + ((c >> (8u * e + 4u)) & 0xFu)] * x1;
                        }
                    }
            }
        }
        for (uint64_t k = 0u; k < R; ++k) {
            const uint64_t r = r0 + k;
            const float sum = khronos_opencl1_igpu__zzpackage_lanes_sum_each(r < rows ? part[k] : 0.0f, (uint32_t)W);
            if (j == 0u && r < rows) {
                const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
                const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
                out[g.out_at[i] + r] = nn__kernels__zzabi_to_half(sum * scale, over);
            }
        }
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemv2(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                                    const float* levels, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_gemv_rows(out, g, x, cols, levels, over, 2u);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemv4(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols,
                                                                    const float* levels, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_gemv_rows(out, g, x, cols, levels, over, 4u);
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

/* ⭐ AND THE WEIGHTED SUM, `R` ROWS A RUN — as the gemv's rows above: a lane reads each matrix's `x_i` once for its `R` rows'
 *   codes. Each row's part is the one-row kernel's — every matrix's part at that row made in the same order, scaled and
 *   weighted and added in the matrices' order, folded by the same run — so the same bytes; `residual` may be `out`, each
 *   row's one writer reading its element first. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemv_sum_rows(uint16_t* out, nn__turboquant__groups g, const uint16_t* x,
                                                                            const uint16_t* w, const uint16_t* residual, uint64_t rows,
                                                                            uint64_t cols, const float* levels, unsigned int* over,
                                                                            const uint64_t R) {
    const uint64_t lane = nn__silicon__lane(), W = KHRONOS_OPENCL1_IGPU__GEMV_RUN, runs = nn__silicon__lanes() / W, j = lane % W;
    const uint64_t span = runs * R, units = (rows + span - 1u) / span;
    for (uint64_t u = nn__silicon__block(); u < units; u += nn__silicon__blocks()) {
        const uint64_t r0 = u * span + (lane / W) * R;
        float part[4];
        for (uint64_t k = 0u; k < R; ++k) part[k] = 0.0f;
        for (uint64_t i = 0u; i < g.count && i < NN__TURBOQUANT__GROUPS_MAX; ++i) {
            const uint64_t d = g.d[i], row_bytes = cols * d / 8u;
            const uint8_t* codes = (const uint8_t*)(uintptr_t)g.codes[i] + r0 * row_bytes;
            const uint8_t* lut = (const uint8_t*)(uintptr_t)g.luts[i];
            const float wi = nn__silicon__half_to_float(w[g.out_at[i]]);
            float p[4];
            for (uint64_t k = 0u; k < R; ++k) p[k] = 0.0f;
            for (uint64_t at = 16u * j; at < row_bytes; at += 16u * W) {
                const uint32_t* xw = (const uint32_t*)(x + i * cols + at * 8u / d);
                uint32_t cw[4][4];
                for (uint64_t k = 0u; k < R; ++k)
                    for (uint64_t q = 0u; q < 4u; ++q) cw[k][q] = r0 + k < rows ? ((const uint32_t*)(codes + k * row_bytes + at))[q] : 0u;
                if (d == 8u) {
                    for (uint64_t q = 0u; q < 4u; ++q) {
                        const uint32_t a = xw[2u * q], b = xw[2u * q + 1u];
                        const float x0 = nn__silicon__half_to_float((uint16_t)(a & 0xFFFFu)), x1 = nn__silicon__half_to_float((uint16_t)(a >> 16));
                        const float x2 = nn__silicon__half_to_float((uint16_t)(b & 0xFFFFu)), x3 = nn__silicon__half_to_float((uint16_t)(b >> 16));
                        for (uint64_t k = 0u; k < R; ++k) {
                            const uint32_t c = cw[k][q];
                            p[k] += levels[c & 0xFFu] * x0;
                            p[k] += levels[(c >> 8) & 0xFFu] * x1;
                            p[k] += levels[(c >> 16) & 0xFFu] * x2;
                            p[k] += levels[c >> 24] * x3;
                        }
                    }
                } else {
                    for (uint64_t q = 0u; q < 4u; ++q)
                        for (uint64_t e = 0u; e < 4u; ++e) {
                            const uint32_t a = xw[4u * q + e];
                            const float x0 = nn__silicon__half_to_float((uint16_t)(a & 0xFFFFu)), x1 = nn__silicon__half_to_float((uint16_t)(a >> 16));
                            for (uint64_t k = 0u; k < R; ++k) {
                                const uint32_t c = cw[k][q];
                                p[k] += levels[256u + ((c >> (8u * e)) & 0xFu)] * x0;
                                p[k] += levels[256u + ((c >> (8u * e + 4u)) & 0xFu)] * x1;
                            }
                        }
                }
            }
            for (uint64_t k = 0u; k < R; ++k) {
                const uint64_t r = r0 + k;
                if (r < rows) {
                    const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)lut[r * 2u] | ((uint32_t)lut[r * 2u + 1u] << 8)));
                    part[k] += p[k] * scale * wi;
                }
            }
        }
        for (uint64_t k = 0u; k < R; ++k) {
            const uint64_t r = r0 + k;
            const float sum = khronos_opencl1_igpu__zzpackage_lanes_sum_each(r < rows ? part[k] : 0.0f, (uint32_t)W);
            if (j == 0u && r < rows) out[r] = nn__kernels__zzabi_to_half(nn__silicon__half_to_float(residual[r]) + sum, over);
        }
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemv_sum2(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                                        const uint16_t* residual, uint64_t rows, uint64_t cols,
                                                                        const float* levels, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_gemv_sum_rows(out, g, x, w, residual, rows, cols, levels, over, 2u);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemv_sum4(uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w,
                                                                        const uint16_t* residual, uint64_t rows, uint64_t cols,
                                                                        const float* levels, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_gemv_sum_rows(out, g, x, w, residual, rows, cols, levels, over, 4u);
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

/* ⭐ KDA's STEP, ONE ROW AT A HEAD OF 128 — nn's `kda_step`, a block a head and a lane a column, as `kda_steps` above: the
 *   lane's column of the state in its private memory between the step's two passes, each state element read once and
 *   written once, where nn's body reads and writes it in memory in both. Each row's decay is nn's expression, made by every
 *   lane as nn's lane makes it. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_kda_step(float* S, const uint16_t* conved, const uint16_t* f, const uint16_t* b,
                                                                       const uint16_t* dt, const uint16_t* a_log, const uint16_t* gate,
                                                                       const uint16_t* w, uint16_t* out, uint64_t heads, float lower, float eps,
                                                                       unsigned int* over) {
    const uint64_t D = 128u, j = nn__silicon__lane(), width = heads * D;
    for (uint64_t h = nn__silicon__block(); h < heads; h += nn__silicon__blocks()) {
        float* Sh = S + h * D * D;
        const uint16_t* q = conved + h * D;
        const uint16_t* k = conved + width + h * D;
        const uint16_t* v = conved + 2u * width + h * D;
        const uint16_t* fh = f + h * D;
        const uint16_t* dth = dt + h * D;
        const float a = nn__silicon__expf(nn__silicon__half_to_float(a_log[h]));
        const float beta = 1.0f / (1.0f + nn__silicon__expf(-nn__silicon__half_to_float(b[h])));
        const float qi = nn__silicon__half_to_float(q[j]), ki = nn__silicon__half_to_float(k[j]);
        float qq = 0.0f, kk = 0.0f;
        qq += qi * qi;
        kk += ki * ki;
        const float inv_q = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(qq) + 1e-6f) / nn__silicon__sqrtf((float)D);
        const float inv_k = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(kk) + 1e-6f);
        float col[128];
        for (uint64_t i = 0u; i < D; ++i) col[i] = Sh[i * D + j];   /* the column's reads all in flight */
        float kv = 0.0f;
        for (uint64_t i = 0u; i < D; ++i) {
            const float gi = lower / (1.0f + nn__silicon__expf(-a * (nn__silicon__half_to_float(fh[i]) + nn__silicon__half_to_float(dth[i]))));
            col[i] *= nn__silicon__expf(gi);
            kv += col[i] * nn__silicon__half_to_float(k[i]) * inv_k;
        }
        const float delta = (nn__silicon__half_to_float(v[j]) - kv) * beta;
        float o = 0.0f;
        for (uint64_t i = 0u; i < D; ++i) {
            const float s = col[i] + nn__silicon__half_to_float(k[i]) * inv_k * delta;
            Sh[i * D + j] = s;
            o += s * nn__silicon__half_to_float(q[i]);
        }
        const float oj = nn__silicon__half_to_float(nn__kernels__zzabi_to_half(o * inv_q, over));   /* rounded, as nn's readout is */
        float oo = 0.0f;
        oo += oj * oj;
        const float scale = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(oo) / (float)D + eps);
        const float gz = nn__silicon__half_to_float(gate[h * D + j]);
        out[h * D + j] = nn__kernels__zzabi_to_half(oj * scale * nn__silicon__half_to_float(w[j]) / (1.0f + nn__silicon__expf(-gz)), over);
    }
}

/* ⭐ A SITE'S MAP'S LOGITS — nn's `hyper_logits`, a block a logit and a lane a stride of the streams, the same sums in the
 *   same order (a lane's squares and products element by element, then the block's fold), so the same bits. What is this
 *   family's own: a lane's elements read sixteen at a time, all in flight together, where nn's body waits for each before
 *   the next. */
#define KHRONOS_OPENCL1_IGPU__HYPER_RUN 16u
static __device__ inline void khronos_opencl1_igpu__zzprivate_hyper_logits(float* lg, const uint16_t* streams, const uint16_t* fn,
                                                                           uint64_t hidden, uint64_t mult, float norm_eps) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), n = mult * hidden, rows = NN__HYPER__MIX(mult);
    const uint64_t RUN = KHRONOS_OPENCL1_IGPU__HYPER_RUN;
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        const uint16_t* row = fn + r * n;
        float sq = 0.0f, dot = 0.0f;
        uint64_t c = lane;
        for (; c + (RUN - 1u) * lanes < n; c += RUN * lanes) {
            float v[16], g[16];
            for (uint64_t u = 0u; u < RUN; ++u) {
                v[u] = nn__silicon__half_to_float(streams[c + u * lanes]);
                g[u] = nn__silicon__half_to_float(row[c + u * lanes]);
            }
            for (uint64_t u = 0u; u < RUN; ++u) {
                sq += v[u] * v[u];
                dot += g[u] * v[u];
            }
        }
        for (; c < n; c += lanes) {
            const float v = nn__silicon__half_to_float(streams[c]);
            sq += v * v;
            dot += nn__silicon__half_to_float(row[c]) * v;
        }
        const float inv = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(sq) / (float)n + norm_eps);
        const float got = nn__silicon__lanes_sum(dot) * inv;
        if (lane == 0u) lg[r] = got;
    }
}

/* The same over streams of floats — DeepSeek's (▶ nn's THE STREAMS AS FLOATS). */
static __device__ inline void khronos_opencl1_igpu__zzprivate_hyper_logits_fp32(float* lg, const float* streams, const uint16_t* fn,
                                                                                uint64_t hidden, uint64_t mult, float norm_eps) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), n = mult * hidden, rows = NN__HYPER__MIX(mult);
    const uint64_t RUN = KHRONOS_OPENCL1_IGPU__HYPER_RUN;
    for (uint64_t r = nn__silicon__block(); r < rows; r += nn__silicon__blocks()) {
        const uint16_t* row = fn + r * n;
        float sq = 0.0f, dot = 0.0f;
        uint64_t c = lane;
        for (; c + (RUN - 1u) * lanes < n; c += RUN * lanes) {
            float v[16], g[16];
            for (uint64_t u = 0u; u < RUN; ++u) {
                v[u] = streams[c + u * lanes];
                g[u] = nn__silicon__half_to_float(row[c + u * lanes]);
            }
            for (uint64_t u = 0u; u < RUN; ++u) {
                sq += v[u] * v[u];
                dot += g[u] * v[u];
            }
        }
        for (; c < n; c += lanes) {
            const float v = streams[c];
            sq += v * v;
            dot += nn__silicon__half_to_float(row[c]) * v;
        }
        const float inv = 1.0f / nn__silicon__sqrtf(nn__silicon__lanes_sum(sq) / (float)n + norm_eps);
        const float got = nn__silicon__lanes_sum(dot) * inv;
        if (lane == 0u) lg[r] = got;
    }
}

/* ⭐ THE NORM OF A ROW OF FLOATS, INTO HALVES — nn's `rmsnorm_fp32` (DeepSeek's collapse): nn's block, lanes and fold, a
 *   lane's squares added element by element as nn's lane adds them, each element `(x · scale) · w` rounded once, its
 *   overflow in the fault word — so nn's bits; a lane's elements read sixteen at a time, all in flight together, where
 *   nn's body waits for each before the next, twice. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_norm_fp32(uint16_t* o, const float* x, const uint16_t* w, uint64_t n, float eps,
                                                                        unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), RUN = KHRONOS_OPENCL1_IGPU__HYPER_RUN;
    float part = 0.0f;
    uint64_t i = lane;
    for (; i + (RUN - 1u) * lanes < n; i += RUN * lanes) {
        float v[16];
        for (uint64_t u = 0u; u < RUN; ++u) v[u] = x[i + u * lanes];
        for (uint64_t u = 0u; u < RUN; ++u) part += v[u] * v[u];
    }
    for (; i < n; i += lanes) part += x[i] * x[i];
    const float mean  = nn__silicon__lanes_sum(part) / (float)n;
    const float scale = 1.0f / nn__silicon__sqrtf(mean + eps);
    i = lane;
    for (; i + (RUN - 1u) * lanes < n; i += RUN * lanes) {
        float v[16], g[16];
        for (uint64_t u = 0u; u < RUN; ++u) {
            v[u] = x[i + u * lanes];
            g[u] = nn__silicon__half_to_float(w[i + u * lanes]);
        }
        for (uint64_t u = 0u; u < RUN; ++u) o[i + u * lanes] = nn__kernels__zzabi_to_half((v[u] * scale) * g[u], over);
    }
    for (; i < n; i += lanes) o[i] = nn__kernels__zzabi_to_half((x[i] * scale) * nn__silicon__half_to_float(w[i]), over);
}

/* ⭐ THE NORM OF A ROW OF HALVES — nn's `rmsnorm`, as the floats' above: nn's block, lanes and fold, a lane's squares added
 *   element by element as nn's lane adds them, each element `(x · scale) · w` rounded once — nn's bits — a lane's elements
 *   read sixteen at a time, all in flight together, before the fold and after. A lane reads and writes its own elements
 *   alone, so a row normed in place is as nn's. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_norm(uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, float eps,
                                                                   unsigned int* over) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes(), RUN = KHRONOS_OPENCL1_IGPU__HYPER_RUN;
    float part = 0.0f;
    uint64_t i = lane;
    for (; i + (RUN - 1u) * lanes < n; i += RUN * lanes) {
        float v[16];
        for (uint64_t u = 0u; u < RUN; ++u) v[u] = nn__silicon__half_to_float(x[i + u * lanes]);
        for (uint64_t u = 0u; u < RUN; ++u) part += v[u] * v[u];
    }
    for (; i < n; i += lanes) {
        const float v = nn__silicon__half_to_float(x[i]);
        part += v * v;
    }
    const float mean  = nn__silicon__lanes_sum(part) / (float)n;
    const float scale = 1.0f / nn__silicon__sqrtf(mean + eps);
    i = lane;
    for (; i + (RUN - 1u) * lanes < n; i += RUN * lanes) {
        float v[16], g[16];
        for (uint64_t u = 0u; u < RUN; ++u) {
            v[u] = nn__silicon__half_to_float(x[i + u * lanes]);
            g[u] = nn__silicon__half_to_float(w[i + u * lanes]);
        }
        for (uint64_t u = 0u; u < RUN; ++u) o[i + u * lanes] = nn__kernels__zzabi_to_half((v[u] * scale) * g[u], over);
    }
    for (; i < n; i += lanes)
        o[i] = nn__kernels__zzabi_to_half((nn__silicon__half_to_float(x[i]) * scale) * nn__silicon__half_to_float(w[i]), over);
}

/* ⭐ THE SCORED ROUTER — nn's `top_k_scored`: nn's `k` rounds of the block's two folds, each round's largest score + bias
 *   and the first index holding it, so nn's picks and nn's weights. What is this family's own: a lane makes each of its
 *   scores once — an exponential, a logarithm and a root for √softplus — and keeps them, where nn's body makes every
 *   score again each round; a pick leaves its lane's list. Up to four scores a lane (▶ the door). */
static __device__ inline float khronos_opencl1_igpu__zzprivate_score(float x, uint64_t score) {   /* nn's `zzprivate_score` */
    if (score == NN__VECTOR__SCORE_SIGMOID) return 1.0f / (1.0f + nn__silicon__expf(-x));
    if (score == NN__VECTOR__SCORE_SQRT_SOFTPLUS) return nn__silicon__sqrtf(x > 20.0f ? x : nn__silicon__logf(1.0f + nn__silicon__expf(x)));
    return x;
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_top_k_scored(uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x,
                                                                           const uint16_t* bias, uint64_t n, uint64_t k, float scale, uint64_t score) {
    if (nn__silicon__block() != 0u) return;
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    float v[4];
    bool live[4];
    for (uint64_t j = 0u; j < 4u; ++j) {
        const uint64_t i = lane + j * lanes;
        v[j] = i < n ? khronos_opencl1_igpu__zzprivate_score(nn__silicon__half_to_float(x[i]), score) + nn__silicon__half_to_float(bias[i]) : 0.0f;
        live[j] = i < n && v[j] == v[j];
    }
    uint64_t chosen[16];
    for (uint64_t r = 0u; r < k; ++r) {
        uint64_t best_i = n;
        float best_v = -NN__KERNELS__INFINITY;
        for (uint64_t j = 0u; j < 4u; ++j)
            if (live[j] && (best_i == n || v[j] > best_v)) { best_v = v[j]; best_i = lane + j * lanes; }
        const float top = nn__silicon__lanes_max(best_v);
        const float first = nn__silicon__lanes_max(best_i != n && best_v == top ? -(float)(uint32_t)best_i : -NN__KERNELS__INFINITY);
        chosen[r] = first == -NN__KERNELS__INFINITY ? n : (uint64_t)(-first);
        for (uint64_t j = 0u; j < 4u; ++j)
            if (lane + j * lanes == chosen[r]) live[j] = false;
    }
    if (lane != 0u) return;
    float s[16], sum = 0.0f;
    for (uint64_t r = 0u; r < k; ++r) {
        s[r] = chosen[r] < n ? khronos_opencl1_igpu__zzprivate_score(nn__silicon__half_to_float(x[chosen[r]]), score) : 0.0f;
        sum += s[r];
    }
    for (uint64_t r = 0u; r < k; ++r) {
        values[r * stride] = chosen[r];
        weights[r] = nn__kernels__zzabi_to_half(s[r] / (sum + 1e-20f) * scale, 0);   /* nn's router raises no overflow either */
    }
}

/* ⭐ A SITE'S `pre` AT FOUR STREAMS — nn's `hyper_pre` and `hyper_pre_fp32`: nn's body sizes its arrays by the streams it is
 *   handed, and its first lane works the mix's Sinkhorn iterations on them out of memory while the rest collapse. Here the
 *   count is the kernel's own, four, so the first lane works them in its registers — nn's expressions in nn's order
 *   (▶ nn's `nn__hyper__zzprivate_weights`); the collapse is nn's loop. No shuffles in OpenCL C 1.2, so the mix is one
 *   lane's, as nn's is. */
#define KHRONOS_OPENCL1_IGPU__HYPER_MULT 4u
static __device__ inline void khronos_opencl1_igpu__zzprivate_hyper_weights4(float* mix, const float* lg, const float* pre, const uint16_t* base,
                                                                             float s1, float s2, uint64_t iters, float eps) {
    const uint64_t M = KHRONOS_OPENCL1_IGPU__HYPER_MULT;
    float post[4], comb[16];
    for (uint64_t i = 0u; i < M; ++i)
        post[i] = 2.0f / (1.0f + nn__silicon__expf(-(lg[M + i] * s1 + nn__silicon__half_to_float(base[M + i]))));
    for (uint64_t i = 0u; i < M; ++i) {                     /* each row a softmax over its columns, plus eps */
        float top = -NN__KERNELS__INFINITY, sum = 0.0f;
        for (uint64_t j = 0u; j < M; ++j) {
            const uint64_t k = 2u * M + i * M + j;
            comb[i * M + j] = lg[k] * s2 + nn__silicon__half_to_float(base[k]);
            if (comb[i * M + j] > top) top = comb[i * M + j];
        }
        for (uint64_t j = 0u; j < M; ++j) { comb[i * M + j] = nn__silicon__expf(comb[i * M + j] - top); sum += comb[i * M + j]; }
        for (uint64_t j = 0u; j < M; ++j) comb[i * M + j] = comb[i * M + j] / sum + eps;
    }
    for (uint64_t it = 0u; it < iters; ++it) {              /* the columns, then — past the first — rows and columns */
        if (it != 0u)
            for (uint64_t i = 0u; i < M; ++i) {
                float sum = 0.0f;
                for (uint64_t j = 0u; j < M; ++j) sum += comb[i * M + j];
                for (uint64_t j = 0u; j < M; ++j) comb[i * M + j] /= sum + eps;
            }
        for (uint64_t j = 0u; j < M; ++j) {
            float sum = 0.0f;
            for (uint64_t i = 0u; i < M; ++i) sum += comb[i * M + j];
            for (uint64_t i = 0u; i < M; ++i) comb[i * M + j] /= sum + eps;
        }
    }
    for (uint64_t i = 0u; i < M; ++i) { mix[i] = pre[i]; mix[M + i] = post[i]; }
    for (uint64_t k = 0u; k < M * M; ++k) mix[2u * M + k] = comb[k];
}
/* GLM's, the streams halves: the collapse nn's sum in nn's order, rounded to a half as nn rounds it, its overflow in the
 *   same fault word. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_hyper_pre4(uint16_t* out, float* mix, const uint16_t* streams, const uint16_t* base,
                                                                         const uint16_t* scale, uint64_t hidden, uint64_t iters, float eps,
                                                                         unsigned int* over) {
    const uint64_t M = KHRONOS_OPENCL1_IGPU__HYPER_MULT;
    const float* lg = mix + (2u + M) * M;
    const float s0 = nn__silicon__half_to_float(scale[0]), s1 = nn__silicon__half_to_float(scale[1]);
    const float s2 = nn__silicon__half_to_float(scale[2]);
    float pre[4];
    for (uint64_t i = 0u; i < M; ++i)
        pre[i] = 1.0f / (1.0f + nn__silicon__expf(-(lg[i] * s0 + nn__silicon__half_to_float(base[i])))) + eps;
    const uint64_t first = nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane();
    if (first == 0u) khronos_opencl1_igpu__zzprivate_hyper_weights4(mix, lg, pre, base, s1, s2, iters, eps);
    for (uint64_t e = first; e < hidden; e += nn__silicon__blocks() * nn__silicon__lanes()) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < M; ++i) sum += pre[i] * nn__silicon__half_to_float(streams[i * hidden + e]);
        out[e] = nn__kernels__zzabi_to_half(sum, over);
    }
}
/* DeepSeek's, the streams floats. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_hyper_pre4_fp32(float* out, float* mix, const float* streams, const uint16_t* base,
                                                                              const uint16_t* scale, uint64_t hidden, uint64_t iters, float eps) {
    const uint64_t M = KHRONOS_OPENCL1_IGPU__HYPER_MULT;
    const float* lg = mix + (2u + M) * M;
    const float s0 = nn__silicon__half_to_float(scale[0]), s1 = nn__silicon__half_to_float(scale[1]);
    const float s2 = nn__silicon__half_to_float(scale[2]);
    float pre[4];
    for (uint64_t i = 0u; i < M; ++i)
        pre[i] = 1.0f / (1.0f + nn__silicon__expf(-(lg[i] * s0 + nn__silicon__half_to_float(base[i])))) + eps;
    const uint64_t first = nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane();
    if (first == 0u) khronos_opencl1_igpu__zzprivate_hyper_weights4(mix, lg, pre, base, s1, s2, iters, eps);
    for (uint64_t e = first; e < hidden; e += nn__silicon__blocks() * nn__silicon__lanes()) {
        float sum = 0.0f;
        for (uint64_t i = 0u; i < M; ++i) sum += pre[i] * streams[i * hidden + e];
        out[e] = sum;
    }
}

/* ⭐ A ONE-QUERY ATTENTION'S GROUPED SCORES — nn's `attention_scores_grouped`, an item a (tile of `T` positions, query head),
 *   the heads innermost, as nn's items are with tiles of eight: each dot nn's, the same pairs added in the same order
 *   (`dot2`), so the same bits whatever the tile. At a decode's short context nn's tiles are few — sixty-four heads'
 *   one or two each for GLM's latent, each a long chain on one lane — so here a tile is one, two or four positions, as
 *   many items as fill the card (▶ the door), each reading its head's query once for its tile. Halves two to a word, as
 *   nn's `dots` reads them (▶ the door's fit). */
static __device__ inline void khronos_opencl1_igpu__zzprivate_scores_tile(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                          uint64_t kv_heads, uint64_t head_dim, uint64_t length, const uint64_t T) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim, tiles = (length + T - 1u) / T;
    const float scale = 1.0f / nn__silicon__sqrtf((float)head_dim);
    for (uint64_t it = nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane(); it < q_heads * tiles;
         it += nn__silicon__blocks() * nn__silicon__lanes()) {
        const uint64_t h = it % q_heads, t0 = (it / q_heads) * T, span = length - t0 < T ? length - t0 : T;
        const uint32_t* q2 = (const uint32_t*)(q + h * head_dim);
        const uint32_t* k2 = (const uint32_t*)(k + (h / group) * head_dim);
        float dot[4];
        for (uint64_t j = 0u; j < T; ++j) dot[j] = 0.0f;
        for (uint64_t e = 0u; e < head_dim / 2u; ++e) {
            const uint32_t qe = q2[e];
            for (uint64_t j = 0u; j < T; ++j)
                if (j < span) dot[j] = nn__silicon__dot2(qe, k2[((t0 + j) * row) / 2u + e], dot[j]);
        }
        for (uint64_t j = 0u; j < T; ++j)
            if (j < span) p[h * length + t0 + j] = dot[j] * scale;
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_scores1(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                      uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    khronos_opencl1_igpu__zzprivate_scores_tile(p, q, k, q_heads, kv_heads, head_dim, length, 1u);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_scores2(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                      uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    khronos_opencl1_igpu__zzprivate_scores_tile(p, q, k, q_heads, kv_heads, head_dim, length, 2u);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_scores4(float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads,
                                                                      uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    khronos_opencl1_igpu__zzprivate_scores_tile(p, q, k, q_heads, kv_heads, head_dim, length, 4u);
}

/* ⭐⭐ THE EXPERT-MAJOR GEMM — nn's `expert_rows` (and so `expert_groups`, a launch a group), every sum nn's to the bit,
 *   without nn's fold a (row, pair). nn gives a matrix row a block of 256 lanes and folds each of its sums over the block,
 *   four pairs a fold of nine waits, and at a 5,120-wide row a lane's share is five containers, so the block waits more
 *   than it multiplies: `MEASURED` Ministral D4's prompt, 82 of its 85 s in these two doors
 *   (measurements/2026-10-29_ministral_opencl1_embedding_precision.md). Here a run of 32 lanes makes `GEMM_W` rows' sums
 *   over `GEMM_X` pairs: lane `m` makes the parts of nn's lanes `m + 32·q` (q 0 … 7), each over the same containers in the
 *   same order, two codes at a time in `dot2`'s expression, and adds them as the fold's first three steps do (`i` and `i + 128`,
 *   then `+ 64`, then `+ 32`); the run's fold (`lanes_sum_each`) is the fold's last five. A block's eight runs take
 *   neighbouring tiles of the same rows, so they read the rows' codes together; a container is one 16-bit read, and the
 *   code's width a constant of its own kernel (`gemm4`, `gemm8`). `MEASURED` on the RTX 5060 Ti: wider tiles (2 × 8, or
 *   four of nn's lanes in one pass) spill and lose; one fold for a tile's eight sums gains nothing.  */
#define KHRONOS_OPENCL1_IGPU__GEMM_W 2u
#define KHRONOS_OPENCL1_IGPU__GEMM_X 4u
#define KHRONOS_OPENCL1_IGPU__GEMM_WX 8u    /* GEMM_W × GEMM_X */
/* nn's lane `vl`'s parts, `o[a · GEMM_X + b]` for row `a` and pair `b` of the tile — its containers `vl`, `vl + 256` …,
 * `D` bits a code, `16 / D` codes a container (a constant: the shifts fold). Every row is a real one: a tile's rows and
 * pairs past the matrix read its first, and their sums are not written. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemm_lane(float* o, const uint16_t* w0, const uint16_t* w1, const uint32_t* x0,
                                                                        const uint32_t* x1, const uint32_t* x2, const uint32_t* x3,
                                                                        const float* levels, uint32_t vl, uint32_t units, const uint32_t D) {
    const uint32_t per = 16u / D, mask = (1u << D) - 1u;
    for (uint32_t i = 0u; i < KHRONOS_OPENCL1_IGPU__GEMM_WX; ++i) o[i] = 0.0f;
    for (uint32_t u = vl; u < units; u += 256u) {
        const uint32_t u0 = w0[u], u1 = w1[u], c = u * (per / 2u);
        for (uint32_t s = 0u; s < per; s += 2u) {
            /* x's two columns one word, each half widened once, then `dot2`'s own expression: (c + w·x) + w'·x' */
            const float wl0 = levels[(u0 >> (D * s)) & mask], wh0 = levels[(u0 >> (D * (s + 1u))) & mask];
            const float wl1 = levels[(u1 >> (D * s)) & mask], wh1 = levels[(u1 >> (D * (s + 1u))) & mask];
            const uint32_t p0 = x0[c + s / 2u], p1 = x1[c + s / 2u], p2 = x2[c + s / 2u], p3 = x3[c + s / 2u];
            const float l0 = nn__silicon__half_to_float((uint16_t)(p0 & 0xFFFFu)), h0 = nn__silicon__half_to_float((uint16_t)(p0 >> 16));
            const float l1 = nn__silicon__half_to_float((uint16_t)(p1 & 0xFFFFu)), h1 = nn__silicon__half_to_float((uint16_t)(p1 >> 16));
            const float l2 = nn__silicon__half_to_float((uint16_t)(p2 & 0xFFFFu)), h2 = nn__silicon__half_to_float((uint16_t)(p2 >> 16));
            const float l3 = nn__silicon__half_to_float((uint16_t)(p3 & 0xFFFFu)), h3 = nn__silicon__half_to_float((uint16_t)(p3 >> 16));
            o[0] = (o[0] + wl0 * l0) + wh0 * h0; o[1] = (o[1] + wl0 * l1) + wh0 * h1;
            o[2] = (o[2] + wl0 * l2) + wh0 * h2; o[3] = (o[3] + wl0 * l3) + wh0 * h3;
            o[4] = (o[4] + wl1 * l0) + wh1 * h0; o[5] = (o[5] + wl1 * l1) + wh1 * h1;
            o[6] = (o[6] + wl1 * l2) + wh1 * h2; o[7] = (o[7] + wl1 * l3) + wh1 * h3;
        }
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemm_tile(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                                        uint64_t l_room, const uint16_t* x, const uint32_t* rows, uint64_t count,
                                                                        uint64_t out_rows, uint64_t cols, const float* table, unsigned int* over,
                                                                        const uint32_t D) {
    const uint64_t row_bytes = nn__turboquant__row_bytes(D, cols), units = cols / (16u / D), rows_held = w_room / row_bytes;
    const uint64_t held = rows_held < l_room / 2u ? rows_held : l_room / 2u;
    const float* levels = table + (D == 8u ? 0u : 256u);       /* nn's levels as floats: the 8 bits' 256, then the 4 bits' 16 */
    const uint64_t lane = nn__silicon__lane(), run = lane / 32u;
    const uint32_t m = (uint32_t)(lane % 32u), U = (uint32_t)units;
    const uint64_t row_tiles = (out_rows + KHRONOS_OPENCL1_IGPU__GEMM_W - 1u) / KHRONOS_OPENCL1_IGPU__GEMM_W;
    const uint64_t pair_tiles = (count + KHRONOS_OPENCL1_IGPU__GEMM_X - 1u) / KHRONOS_OPENCL1_IGPU__GEMM_X, items = row_tiles * pair_tiles;
    /* every lane of a block walks the same items, so every run reaches every fold; an item past the last is made, not written */
    for (uint64_t first = nn__silicon__block() * 8u; first < items; first += nn__silicon__blocks() * 8u) {
        const uint64_t it = first + run, r0 = (it / pair_tiles) * KHRONOS_OPENCL1_IGPU__GEMM_W, j0 = (it % pair_tiles) * KHRONOS_OPENCL1_IGPU__GEMM_X;
        const bool real = it < items;
        const bool v0 = real && r0 < held && r0 < out_rows, v1 = real && r0 + 1u < held && r0 + 1u < out_rows;
        const uint16_t* w0 = (const uint16_t*)(const void*)(weights + (v0 ? r0 : 0u) * row_bytes);
        const uint16_t* w1 = (const uint16_t*)(const void*)(weights + (v1 ? r0 + 1u : 0u) * row_bytes);
        const uint32_t* x0 = (const uint32_t*)(const void*)(x + (uint64_t)rows[real && j0 < count ? 2u * j0 : 0u] * cols);
        const uint32_t* x1 = (const uint32_t*)(const void*)(x + (uint64_t)rows[real && j0 + 1u < count ? 2u * (j0 + 1u) : 0u] * cols);
        const uint32_t* x2 = (const uint32_t*)(const void*)(x + (uint64_t)rows[real && j0 + 2u < count ? 2u * (j0 + 2u) : 0u] * cols);
        const uint32_t* x3 = (const uint32_t*)(const void*)(x + (uint64_t)rows[real && j0 + 3u < count ? 2u * (j0 + 3u) : 0u] * cols);
        /* nn's fold's first steps, its lanes m + 32q: ((q0 + q4) + (q2 + q6)) + ((q1 + q5) + (q3 + q7)) */
        float lo[KHRONOS_OPENCL1_IGPU__GEMM_WX], hi[KHRONOS_OPENCL1_IGPU__GEMM_WX], t[KHRONOS_OPENCL1_IGPU__GEMM_WX], t2[KHRONOS_OPENCL1_IGPU__GEMM_WX];
        khronos_opencl1_igpu__zzprivate_gemm_lane(lo, w0, w1, x0, x1, x2, x3, levels, m, U, D);
        khronos_opencl1_igpu__zzprivate_gemm_lane(t, w0, w1, x0, x1, x2, x3, levels, m + 128u, U, D);
        for (uint32_t i = 0u; i < KHRONOS_OPENCL1_IGPU__GEMM_WX; ++i) lo[i] = lo[i] + t[i];
        khronos_opencl1_igpu__zzprivate_gemm_lane(t2, w0, w1, x0, x1, x2, x3, levels, m + 64u, U, D);
        khronos_opencl1_igpu__zzprivate_gemm_lane(t, w0, w1, x0, x1, x2, x3, levels, m + 192u, U, D);
        for (uint32_t i = 0u; i < KHRONOS_OPENCL1_IGPU__GEMM_WX; ++i) lo[i] = lo[i] + (t2[i] + t[i]);
        khronos_opencl1_igpu__zzprivate_gemm_lane(hi, w0, w1, x0, x1, x2, x3, levels, m + 32u, U, D);
        khronos_opencl1_igpu__zzprivate_gemm_lane(t, w0, w1, x0, x1, x2, x3, levels, m + 160u, U, D);
        for (uint32_t i = 0u; i < KHRONOS_OPENCL1_IGPU__GEMM_WX; ++i) hi[i] = hi[i] + t[i];
        khronos_opencl1_igpu__zzprivate_gemm_lane(t2, w0, w1, x0, x1, x2, x3, levels, m + 96u, U, D);
        khronos_opencl1_igpu__zzprivate_gemm_lane(t, w0, w1, x0, x1, x2, x3, levels, m + 224u, U, D);
        for (uint32_t i = 0u; i < KHRONOS_OPENCL1_IGPU__GEMM_WX; ++i) hi[i] = hi[i] + (t2[i] + t[i]);
        for (uint32_t i = 0u; i < KHRONOS_OPENCL1_IGPU__GEMM_WX; ++i) {
            const float sum = khronos_opencl1_igpu__zzpackage_lanes_sum_each(lo[i] + hi[i], 32u);
            const uint32_t a = i / KHRONOS_OPENCL1_IGPU__GEMM_X, b = i % KHRONOS_OPENCL1_IGPU__GEMM_X;
            if (m != 0u || !(a == 0u ? v0 : v1) || !real || j0 + b >= count) continue;
            const uint64_t r = r0 + a;
            const float scale = nn__silicon__half_to_float((uint16_t)((uint32_t)luts[r * 2u] | ((uint32_t)luts[r * 2u + 1u] << 8)));
            out[(uint64_t)rows[2u * (j0 + b) + 1u] * out_rows + r] = nn__kernels__zzabi_to_half(sum * scale, over);
        }
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemm4(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                                    uint64_t l_room, const uint16_t* x, const uint32_t* rows, uint64_t count,
                                                                    uint64_t out_rows, uint64_t cols, const float* table, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_gemm_tile(out, weights, w_room, luts, l_room, x, rows, count, out_rows, cols, table, over, 4u);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_gemm8(uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts,
                                                                    uint64_t l_room, const uint16_t* x, const uint32_t* rows, uint64_t count,
                                                                    uint64_t out_rows, uint64_t cols, const float* table, unsigned int* over) {
    khronos_opencl1_igpu__zzprivate_gemm_tile(out, weights, w_room, luts, l_room, x, rows, count, out_rows, cols, table, over, 8u);
}

/* ⭐ A RANGE'S RESIDUAL, ONE QUERY — the tiered cache's decode and a prompt's rows over it (`attention_weights`,
 *   `attention_residual_mix` and their TurboQuant tiers'), nn's arithmetic and order. nn gives a head a block, a lane a
 *   position's whole dot and a value element's whole chain down the cache; here an item a (head, position) for the scores
 *   (the heads innermost, so a group's heads read a key row together), nn's softmax first half a block a head, and an item
 *   a (head, element) for the mix — each element's sum the positions one after another from the first, reading ahead.
 *   The TurboQuant levels as floats (the GEMV's table): an element is `scale · level`, as nn's. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_scores_tq(float* p, const uint16_t* q, const uint8_t* codes, const uint16_t* scales,
                                                                        uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                                        uint64_t d, const float* table) {
    const uint64_t group = q_heads / kv_heads, row_bytes = nn__turboquant__row_bytes(d, head_dim);
    const uint32_t D = (uint32_t)d, per = 16u / D, mask = (1u << D) - 1u, units = (uint32_t)(head_dim / per);
    const float* levels = table + (d == 8u ? 0u : 256u);
    const float inv = 1.0f / nn__silicon__sqrtf((float)(uint32_t)head_dim);
    for (uint64_t it = nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane(); it < q_heads * length;
         it += nn__silicon__blocks() * nn__silicon__lanes()) {
        const uint64_t h = it % q_heads, t = it / q_heads, row = t * kv_heads + h / group;
        const uint16_t* qh = q + h * head_dim;
        const uint16_t* kr = (const uint16_t*)(const void*)(codes + row * row_bytes);
        const float scale = nn__silicon__half_to_float(scales[row]);
        float dot = 0.0f;
        for (uint32_t u = 0u; u < units; ++u) {
            const uint32_t unit = kr[u];
            for (uint32_t s = 0u; s < per; ++s)
                dot += nn__silicon__half_to_float(qh[u * per + s]) * (scale * levels[(unit >> (D * s)) & mask]);
        }
        p[h * length + t] = dot * inv;
    }
}
/* nn's first pass's second half, a block a head: the largest score into `m`, `e^(score − m)` in place, their sum `l`. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_residual_weights(float* p, float* res, uint64_t q_heads, uint64_t head_dim,
                                                                               uint64_t length) {
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    for (uint64_t h = nn__silicon__block(); h < q_heads; h += nn__silicon__blocks()) {
        float* ph = p + h * length;
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
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim)] = top;
            res[h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim) + 1u] = sum;
        }
    }
}
/* Its second pass, `acc`: over the halves' tier (`d` 0) or a TurboQuant one. */
static __device__ inline void khronos_opencl1_igpu__zzprivate_residual_mix_any(float* res, const float* p, const uint16_t* v, const uint8_t* codes,
                                                                               const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads,
                                                                               uint64_t head_dim, uint64_t length, uint64_t d, const float* table) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim;
    const uint64_t row_bytes = d ? nn__turboquant__row_bytes(d, head_dim) : 0u;
    const uint32_t D = (uint32_t)d, per = d ? 16u / D : 1u, mask = d ? (1u << D) - 1u : 0u;
    const float* levels = table + (d == 8u ? 0u : 256u);
    for (uint64_t it = nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane(); it < q_heads * head_dim;
         it += nn__silicon__blocks() * nn__silicon__lanes()) {
        const uint64_t h = it / head_dim, e = it % head_dim, g = h / group;
        const float* ph = p + h * length;
        float sum = 0.0f;
        if (!d) {
            const uint16_t* vh = v + g * head_dim + e;
            for (uint64_t t = 0u; t < length; ++t) {
                sum += *ph * nn__silicon__half_to_float(*vh);
                ++ph;
                vh += row;
            }
        } else {
            /* element `e` of row `t · kv_heads + g`: its container and its code's shift, the same each row */
            const uint64_t at = (e / per) * 2u, stride = kv_heads * row_bytes;
            const uint32_t shift = D * (uint32_t)(e % per);
            const uint8_t* c = codes + g * row_bytes + at;
            const uint16_t* sc = scales + g;
            for (uint64_t t = 0u; t < length; ++t) {
                const uint32_t unit = (uint32_t)c[0] | ((uint32_t)c[1] << 8);
                sum += *ph * (nn__silicon__half_to_float(*sc) * levels[(unit >> shift) & mask]);
                ++ph;
                c += stride;
                sc += kv_heads;
            }
        }
        res[h * NN__ATTENTION__RESIDUAL_FLOATS(head_dim) + 2u + e] = sum;
    }
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_residual_mix16(float* res, const float* p, const uint16_t* v, uint64_t q_heads,
                                                                             uint64_t kv_heads, uint64_t head_dim, uint64_t length) {
    khronos_opencl1_igpu__zzprivate_residual_mix_any(res, p, v, 0, 0, q_heads, kv_heads, head_dim, length, 0u, 0);
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_residual_mix_tq(float* res, const float* p, const uint8_t* codes, const uint16_t* scales,
                                                                              uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length,
                                                                              uint64_t d, const float* table) {
    khronos_opencl1_igpu__zzprivate_residual_mix_any(res, p, 0, codes, scales, q_heads, kv_heads, head_dim, length, d, table);
}

/* ⭐ A ONE-QUERY ATTENTION'S MIX — nn's `attention_mix`, an item a (query head, element): each element's sum the positions
 *   one after another from the first, nn's expression, so nn's bits. nn gives a head a block and its lanes the elements,
 *   each a chain down the whole cache with one read in flight; here `KHRONOS_OPENCL1_IGPU__MIX_AHEAD` positions' weights
 *   and values are read before their products are added, in order. */
#define KHRONOS_OPENCL1_IGPU__MIX_AHEAD 32u
static __device__ inline void khronos_opencl1_igpu__zzprivate_mix(uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads,
                                                                  uint64_t kv_heads, uint64_t head_dim, uint64_t length, unsigned int* over) {
    const uint64_t group = q_heads / kv_heads, row = kv_heads * head_dim;
    for (uint64_t it = nn__silicon__block() * nn__silicon__lanes() + nn__silicon__lane(); it < q_heads * head_dim;
         it += nn__silicon__blocks() * nn__silicon__lanes()) {
        const uint64_t h = it / head_dim, e = it % head_dim;
        const float* ph = p + h * length;
        const uint16_t* vh = v + (h / group) * head_dim + e;
        float sum = 0.0f;
        uint64_t t = 0u;
        /* a position's value a row on: the pointer steps, no index multiplied */
        for (; t + KHRONOS_OPENCL1_IGPU__MIX_AHEAD <= length; t += KHRONOS_OPENCL1_IGPU__MIX_AHEAD) {
            float w[KHRONOS_OPENCL1_IGPU__MIX_AHEAD], x[KHRONOS_OPENCL1_IGPU__MIX_AHEAD];
            for (uint32_t j = 0u; j < KHRONOS_OPENCL1_IGPU__MIX_AHEAD; ++j) {
                w[j] = ph[j];
                x[j] = nn__silicon__half_to_float(*vh);
                vh += row;
            }
            ph += KHRONOS_OPENCL1_IGPU__MIX_AHEAD;
            for (uint32_t j = 0u; j < KHRONOS_OPENCL1_IGPU__MIX_AHEAD; ++j) sum += w[j] * x[j];
        }
        for (; t < length; ++t) {
            sum += *ph * nn__silicon__half_to_float(*vh);
            ++ph;
            vh += row;
        }
        o[h * head_dim + e] = nn__kernels__zzabi_to_half(sum, over);
    }
}

/* ⭐ THE `k` LARGEST OF A LONG VECTOR — a vocabulary's logits, for a sampler: nn's `top_k`, its picks and weights to the bit.
 *   nn's `k` rounds each read every value: 20 rounds over 248,320 logits are 20 passes over half a megabyte, on one block.
 *   Here two passes do it, with no memory shared but the folds' scratch:
 *   ① each lane's largest value (a NaN never one), in its registers;
 *   ② a BAR, the largest half that at least `k` lanes' largest reach — sixteen folds bisecting the halves in order, each
 *     counting the lanes at or above — so `k` values reach it and every pick is at or above it (every valid value where
 *     fewer than `k` lanes hold one);
 *   ③ each lane's values at or above the bar, the `k` largest of them kept largest first and of equal values the first
 *     index first — the order nn's rounds take them in;
 *   ④ nn's `k` rounds over the lanes' heads alone: the largest head and the first index holding it, by nn's two folds, and
 *     the lane that held it moves to its next. A lane's `k` cover every one of its values the rounds can take, since no
 *     round takes more than `k`, so each round is nn's, and so are the picks; the weights are nn's expressions over them
 *     in their order. One block, 256 lanes, as nn's door. */
#define KHRONOS_OPENCL1_IGPU__TOP_K_RUN 16u              /* a lane's reads in flight */
/* A half from its place in the halves' order (the largest last, a NaN past either end). */
static __device__ inline float khronos_opencl1_igpu__zzprivate_top_k_level(uint32_t order) {
    return nn__silicon__half_to_float((uint16_t)((order & 0x8000u) != 0u ? (order & 0x7FFFu) : (~order & 0xFFFFu)));
}
static __device__ inline void khronos_opencl1_igpu__zzprivate_top_k(uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x,
                                                                    uint64_t n, uint64_t k) {
    if (nn__silicon__block() != 0u) return;
    const uint64_t lane = nn__silicon__lane(), lanes = nn__silicon__lanes();
    const uint64_t RUN = KHRONOS_OPENCL1_IGPU__TOP_K_RUN;
    /* ① */
    bool any = false;
    float most = -NN__KERNELS__INFINITY;
    for (uint64_t i0 = lane; i0 < n; i0 += RUN * lanes) {
        float v[16];
        for (uint64_t u = 0u; u < RUN; ++u) v[u] = i0 + u * lanes < n ? nn__silicon__half_to_float(x[i0 + u * lanes]) : 0.0f;
        for (uint64_t u = 0u; u < RUN; ++u)
            if (i0 + u * lanes < n && v[u] == v[u]) {
                if (!any || v[u] > most) most = v[u];
                any = true;
            }
    }
    /* ② the halves' order from -inf (0x03FF) to +inf (0xFC00): the largest place whose half `k` lanes' largest reach */
    uint32_t lo = 0x03FFu, hi = 0xFC00u;
    if (nn__silicon__lanes_sum(any ? 1.0f : 0.0f) >= (float)k)
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo + 1u) / 2u;
            const float level = khronos_opencl1_igpu__zzprivate_top_k_level(mid);
            if (nn__silicon__lanes_sum(any && most >= level ? 1.0f : 0.0f) >= (float)k) lo = mid;
            else hi = mid - 1u;
        }
    const float bar = khronos_opencl1_igpu__zzprivate_top_k_level(lo);
    /* ③ */
    float held_v[64];
    uint32_t held_i[64];
    uint64_t count = 0u;
    if (any && most >= bar)
        for (uint64_t i0 = lane; i0 < n; i0 += RUN * lanes) {
            float v[16];
            for (uint64_t u = 0u; u < RUN; ++u) v[u] = i0 + u * lanes < n ? nn__silicon__half_to_float(x[i0 + u * lanes]) : 0.0f;
            for (uint64_t u = 0u; u < RUN; ++u) {
                if (i0 + u * lanes >= n || v[u] != v[u] || !(v[u] >= bar) || (count == k && !(v[u] > held_v[k - 1u]))) continue;
                uint64_t p = count < k ? count : k - 1u;    /* full, the last is dropped */
                while (p > 0u && v[u] > held_v[p - 1u]) { held_v[p] = held_v[p - 1u]; held_i[p] = held_i[p - 1u]; --p; }
                held_v[p] = v[u];
                held_i[p] = (uint32_t)(i0 + u * lanes);
                if (count < k) ++count;
            }
        }
    /* ④ */
    uint64_t chosen[64];
    float chosen_v[64];
    uint64_t next = 0u;
    for (uint64_t r = 0u; r < k; ++r) {
        const bool has = next < count;
        const float mine = has ? held_v[next] : -NN__KERNELS__INFINITY;
        const float top = nn__silicon__lanes_max(mine);
        const float first = nn__silicon__lanes_max(has && mine == top ? -(float)held_i[next] : -NN__KERNELS__INFINITY);
        chosen[r] = first == -NN__KERNELS__INFINITY ? n : (uint64_t)(-first);
        chosen_v[r] = top;
        if (has && (uint64_t)held_i[next] == chosen[r]) ++next;
    }
    if (lane != 0u) return;
    float sum = 0.0f;
    for (uint64_t r = 0u; r < k; ++r) sum += nn__silicon__expf(chosen_v[r] - chosen_v[0]);
    for (uint64_t r = 0u; r < k; ++r) {
        values[r * stride] = chosen[r];
        weights[r] = nn__kernels__zzabi_to_half(nn__silicon__expf(chosen_v[r] - chosen_v[0]) / sum, 0);   /* in [0, 1]: no overflow */
    }
}

#define KHRONOS_OPENCL1_IGPU__OVERRIDE_ROWS(X) \
    X(khronos_opencl1_igpu__zzprivate_kda_step, (float* S, const uint16_t* conved, const uint16_t* f, const uint16_t* b, const uint16_t* dt, const uint16_t* a_log, const uint16_t* gate, const uint16_t* w, uint16_t* out, uint64_t heads, float lower, float eps, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_hyper_logits, (float* lg, const uint16_t* streams, const uint16_t* fn, uint64_t hidden, uint64_t mult, float norm_eps)) \
    X(khronos_opencl1_igpu__zzprivate_hyper_pre4, (uint16_t* out, float* mix, const uint16_t* streams, const uint16_t* base, const uint16_t* scale, uint64_t hidden, uint64_t iters, float eps, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_hyper_pre4_fp32, (float* out, float* mix, const float* streams, const uint16_t* base, const uint16_t* scale, uint64_t hidden, uint64_t iters, float eps)) \
    X(khronos_opencl1_igpu__zzprivate_scores1, (float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length)) \
    X(khronos_opencl1_igpu__zzprivate_scores2, (float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length)) \
    X(khronos_opencl1_igpu__zzprivate_scores4, (float* p, const uint16_t* q, const uint16_t* k, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length)) \
    X(khronos_opencl1_igpu__zzprivate_gemm4, (uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room, const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t out_rows, uint64_t cols, const float* table, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_gemm8, (uint16_t* out, const uint8_t* weights, uint64_t w_room, const uint8_t* luts, uint64_t l_room, const uint16_t* x, const uint32_t* rows, uint64_t count, uint64_t out_rows, uint64_t cols, const float* table, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_scores_tq, (float* p, const uint16_t* q, const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d, const float* table)) \
    X(khronos_opencl1_igpu__zzprivate_residual_weights, (float* p, float* res, uint64_t q_heads, uint64_t head_dim, uint64_t length)) \
    X(khronos_opencl1_igpu__zzprivate_residual_mix16, (float* res, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length)) \
    X(khronos_opencl1_igpu__zzprivate_residual_mix_tq, (float* res, const float* p, const uint8_t* codes, const uint16_t* scales, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, uint64_t d, const float* table)) \
    X(khronos_opencl1_igpu__zzprivate_mix, (uint16_t* o, const float* p, const uint16_t* v, uint64_t q_heads, uint64_t kv_heads, uint64_t head_dim, uint64_t length, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_hyper_logits_fp32, (float* lg, const float* streams, const uint16_t* fn, uint64_t hidden, uint64_t mult, float norm_eps)) \
    X(khronos_opencl1_igpu__zzprivate_norm, (uint16_t* o, const uint16_t* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_norm_fp32, (uint16_t* o, const float* x, const uint16_t* w, uint64_t n, float eps, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_top_k_scored, (uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x, const uint16_t* bias, uint64_t n, uint64_t k, float scale, uint64_t score)) \
    X(khronos_opencl1_igpu__zzprivate_top_k, (uint64_t* values, uint64_t stride, uint16_t* weights, const uint16_t* x, uint64_t n, uint64_t k)) \
    X(khronos_opencl1_igpu__zzprivate_gemv, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, const float* levels, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_gemv2, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, const float* levels, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_gemv4, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, uint64_t cols, const float* levels, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_gemv_sum, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w, const uint16_t* residual, uint64_t rows, uint64_t cols, const float* levels, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_gemv_sum2, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w, const uint16_t* residual, uint64_t rows, uint64_t cols, const float* levels, unsigned int* over)) \
    X(khronos_opencl1_igpu__zzprivate_gemv_sum4, (uint16_t* out, nn__turboquant__groups g, const uint16_t* x, const uint16_t* w, const uint16_t* residual, uint64_t rows, uint64_t cols, const float* levels, unsigned int* over)) \
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
