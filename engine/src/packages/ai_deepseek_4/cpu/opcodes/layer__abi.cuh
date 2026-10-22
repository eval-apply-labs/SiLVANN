#ifndef SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_LAYER__ABI_CUH
#define SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_LAYER__ABI_CUH
/* ══ DEEPSEEK-V4.1'S LAYER, A VERB A DEVICE SEAM ═════════════════════════════════════════════════════════════
 * The four residual streams go in and come out in place, as floats: a sink token's pass fp16's range (▶ nn's THE STREAMS
 * AS FLOATS). A site's verb writes its own mix of them (`hyper_pre_fp32`), collapses them by the mix the site before it
 * wrote (`hyper_collapse_fp32`), norms the input into halves, runs the sublayer and mixes the streams again with its
 * output by its own mix (`hyper_post_fp32`) — the attention site whole in one verb, the MoE site cut in three where the
 * routed experts run on another worker. Every matrix is a TurboQuant record rotated along its input, so each is handed
 * its input rotated; the router is lossless and reads it as it is.
 * ▶ `contracts/objects/layer.cuh` for the tables. */

#include <math.h>
#include <stdlib.h>                       /* a chunk's pairs, made on the host: malloc, free */
#include "../table.cuh"
#include "../../../nn/cpu/routed__header.cuh"   /* the routed experts, as nn runs them */


/* The operands every site verb starts with: the streams, and its table read. */
static inline bool ai_deepseek_4__layer__zzprivate_open(const sys__heap_node* argv, uint64_t* streams, uint64_t* s_room) {
    return sys__heap_node__carries_reference(argv[1].dtype) && sys__node_array__is(argv[1].args[0])
        && nn__primitives__room(&argv[0], streams, s_room);
}

/* The site's input: its own mix of the streams written into `own`, the streams collapsed by the earlier site's `pre` (in
 * `earlier`) into `xin` (floats, as large as the streams), then normed into `h` (halves). The mix's own collapse lands in
 * `xin` first and is written over. A position's mixes are its table's OWN and EARLIER; a chunk's rows bring their own. */
static inline void ai_deepseek_4__layer__zzprivate_collapse(const nn__doors* doors, unsigned int* over, const uint64_t* at,
                                                            uint64_t own, uint64_t earlier, uint64_t streams, uint64_t xin,
                                                            uint64_t h, uint64_t H, uint64_t M, uint64_t iters, float eps,
                                                            float hc_eps) {
    doors->hyper_logits_fp32((float*)(uintptr_t)own + NN__HYPER__MIX(M), (const float*)(uintptr_t)streams,
                             (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__HC_FN], H, M, eps);
    doors->hyper_pre_fp32((float*)(uintptr_t)xin, (float*)(uintptr_t)own, (const float*)(uintptr_t)streams,
                          (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__HC_BASE], (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__HC_SCALE], H, M,
                          iters, hc_eps);
    doors->hyper_collapse_fp32((float*)(uintptr_t)xin, (const float*)(uintptr_t)earlier, (const float*)(uintptr_t)streams, H, M);
    doors->rmsnorm_fp32((uint16_t*)(uintptr_t)h, (const float*)(uintptr_t)xin, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__NORM], H, eps,
                        over);
}

/* The mixes a site reads and writes, and its streams, each big enough for `mult` streams of `hidden` (floats). */
static inline bool ai_deepseek_4__layer__zzprivate_mixes(const uint64_t* room, uint64_t s_room, uint64_t H, uint64_t M) {
    return H != 0u && M != 0u && M <= NN__HYPER__MULT_MAX && H <= (1ull << 24) && 4ull * M * H <= s_room
        && room[AI_DEEPSEEK_4__OWN] >= AI_DEEPSEEK_4__MIX_BYTES && room[AI_DEEPSEEK_4__EARLIER] >= 4ull * M;
}

static inline void ai_deepseek_4__layer__zzprivate_gemv(const nn__doors* doors, unsigned int* over, uint64_t codes, uint64_t c_room,
                                                        uint64_t luts, uint64_t l_room, uint64_t out, uint64_t x, uint64_t d,
                                                        uint64_t rows, uint64_t cols) {
    doors->turboquant_gemv((uint16_t*)(uintptr_t)out, (const uint8_t*)(uintptr_t)codes, c_room, (const uint8_t*)(uintptr_t)luts, l_room,
                           (const uint16_t*)(uintptr_t)x, d, rows, cols, over);
}

/* A compressing layer's part of its table (▶ AI_DEEPSEEK_4__CMP__*): `ratio` 0 where the table stops at the window's. */
typedef struct ai_deepseek_4__compressed {
    uint64_t ratio, source, cache, ring, zero, gather, signs;
    uint64_t at[AI_DEEPSEEK_4__IDX__PLANES], room[AI_DEEPSEEK_4__IDX__PLANES], d[2];
    /* the indexer, where the table carries it */
    bool indexed;
    uint64_t indexer, heads, dim, di[3], topk;
} ai_deepseek_4__compressed;

/* The compressed part read and checked: 0, or the fault to answer with. */
static uint64_t ai_deepseek_4__attention__zzprivate_compressed(uint64_t table, const uint64_t* at, const uint64_t* room, uint64_t HD,
                                                               uint64_t W, uint64_t pos, ai_deepseek_4__compressed* c) {
    const ai_deepseek_4__compressed none = {};
    *c = none;
    const uint64_t length = sys__node_array__length(table);
    if (length < AI_DEEPSEEK_4__CMP__TABLE) return 0u;
    c->indexed = length >= AI_DEEPSEEK_4__IDX__TABLE;
    sys__node_array_walk w;
    if (!sys__node_array__walk(table, AI_DEEPSEEK_4__ATT__TABLE, &w)) return AI_DEEPSEEK_4__LAYER__FAULT_TABLE;
    uint64_t iv[AI_DEEPSEEK_4__IDX__TABLE];
    const unsigned last = c->indexed ? AI_DEEPSEEK_4__IDX__TABLE : AI_DEEPSEEK_4__CMP__TABLE;
    for (unsigned i = AI_DEEPSEEK_4__ATT__TABLE; i < last; ++i, sys__node_array__next(&w)) {
        const sys__heap_node* n = sys__node_array__walk_cell(&w);
        if (n == 0) return AI_DEEPSEEK_4__LAYER__FAULT_TABLE;
        const bool plane = i < AI_DEEPSEEK_4__CMP__PLANES || (i >= AI_DEEPSEEK_4__IDX__KEYS && i < AI_DEEPSEEK_4__IDX__PLANES);
        if (plane) {
            if (!nn__primitives__room(n, &c->at[i], &c->room[i])) return AI_DEEPSEEK_4__LAYER__FAULT_TABLE;
        } else {
            if (n->dtype != SYS__KIND__VALUE_INT) return AI_DEEPSEEK_4__LAYER__FAULT_TABLE;
            iv[i] = n->args[0];
        }
    }
    const uint64_t ratio = iv[AI_DEEPSEEK_4__CMP__RATIO], source = iv[AI_DEEPSEEK_4__CMP__SOURCE];
    const uint64_t rows = iv[AI_DEEPSEEK_4__CMP__ROWS], topk = iv[AI_DEEPSEEK_4__CMP__TOPK];
    c->d[0] = iv[AI_DEEPSEEK_4__CMP__D]; c->d[1] = iv[AI_DEEPSEEK_4__CMP__D + 1u];
    if (ratio == 0u) return 0u;
    if (ratio > 64u || source > 1u || rows == 0u || topk == 0u) return AI_DEEPSEEK_4__LAYER__FAULT_TABLE;
    const uint64_t latents = (pos + 1u) / ratio;
    if (latents > rows) return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    /* ⛳ past `topk` latents the model's indexer chooses which are attended: without its cells, refused */
    if (latents > topk && !c->indexed) return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    const uint64_t len = pos + 1u < W ? pos + 1u : W, attended = latents < topk ? latents : topk;
    if (!nn__primitives__fits(rows * HD, c->room[AI_DEEPSEEK_4__CMP__CACHE]) || !nn__primitives__fits((len + attended) * HD, c->room[AI_DEEPSEEK_4__CMP__GATHER])
     || (source && (!nn__primitives__fits(2u * ratio * HD, c->room[AI_DEEPSEEK_4__CMP__RING])
                    || !nn__primitives__fits(ratio * HD, c->room[AI_DEEPSEEK_4__CMP__ZERO])
                    || !nn__primitives__fits(HD, c->room[AI_DEEPSEEK_4__CMP__NORM]))))
        return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    if (c->indexed) {
        c->indexer = iv[AI_DEEPSEEK_4__IDX__INDEXER]; c->heads = iv[AI_DEEPSEEK_4__IDX__HEADS]; c->dim = iv[AI_DEEPSEEK_4__IDX__DIM];
        for (unsigned i = 0u; i < 3u; ++i) c->di[i] = iv[AI_DEEPSEEK_4__IDX__D + i];
        if (c->indexer > 1u || c->heads == 0u || c->heads > 1024u || c->dim == 0u || c->dim > 4096u || c->dim < 2u
         || !nn__primitives__fits(rows * c->dim, c->room[AI_DEEPSEEK_4__IDX__KEYS]) || c->room[AI_DEEPSEEK_4__IDX__SELECT] < 4u * (topk + 1u)
         || (c->indexer && c->room[AI_DEEPSEEK_4__IDX__SCORES] / 4u < latents)
         || (source && !nn__primitives__fits(c->dim, c->room[AI_DEEPSEEK_4__IDX__KNORM])))
            return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    }
    c->topk = topk;
    (void)at; (void)room;
    c->ratio = ratio; c->source = source; c->cache = c->at[AI_DEEPSEEK_4__CMP__CACHE]; c->ring = c->at[AI_DEEPSEEK_4__CMP__RING];
    c->zero = c->at[AI_DEEPSEEK_4__CMP__ZERO]; c->gather = c->at[AI_DEEPSEEK_4__CMP__GATHER];
    return 0u;
}

/* Where an attention table's mask begins: past its last part — the window's, the compressed latents' or the indexer's. */
static inline unsigned ai_deepseek_4__attention__zzprivate_mask_first(uint64_t table) {
    const uint64_t length = sys__node_array__length(table);
    return length >= AI_DEEPSEEK_4__IDX__TABLE ? AI_DEEPSEEK_4__IDX__TABLE
         : length >= AI_DEEPSEEK_4__CMP__TABLE ? AI_DEEPSEEK_4__CMP__TABLE : AI_DEEPSEEK_4__ATT__TABLE;
}

/* The source's compressor's maps on the site's rotated input `hr`: its latent into `kv` and, past ratio 1, its gate after
 * it (`HD` halves each) — a position's; a chunk's rows run them as one GEMM (▶ `ai_deepseek_4__attention_rows`). */
static void ai_deepseek_4__attention__zzprivate_project(const nn__doors* doors, unsigned int* over, const ai_deepseek_4__compressed* c,
                                                        uint64_t hr, uint64_t kv, uint64_t H, uint64_t HD) {
    const uint64_t* at = c->at;
    const uint64_t* room = c->room;
    if (c->ratio == 1u) {
        ai_deepseek_4__layer__zzprivate_gemv(doors, over, at[AI_DEEPSEEK_4__CMP__WKV], room[AI_DEEPSEEK_4__CMP__WKV], at[AI_DEEPSEEK_4__CMP__WKV + 1u],
                                             room[AI_DEEPSEEK_4__CMP__WKV + 1u], kv, hr, c->d[0], HD, H);
        return;
    }
    nn__turboquant__groups g = {};
    ai_deepseek_4__table__zzpackage_group(&g, 0u, at, AI_DEEPSEEK_4__CMP__WKV, HD, c->d[0], 0u);
    ai_deepseek_4__table__zzpackage_group(&g, 1u, at, AI_DEEPSEEK_4__CMP__WGATE, HD, c->d[1], HD);
    g.count = 2u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)kv, g, (const uint16_t*)(uintptr_t)hr, H, over);
}

/* The source's step at `pos` with its latent `kv` and, past ratio 1, its gate `gate`: both into the group's ring row
 * `pos % ratio`; the group pooled once it completes (into `work`) — a softmax of the gates a channel (nn's pool, with no
 * place bias) — normed, its tail turned at the group's first position, and written at the cache's row `pos / ratio`.
 * Ratio 1 is the latent of the position itself, normed and turned. `work` holds `2 · HD + dim` halves: the pooled latent,
 * then the index key's rotated latent and the key. */
static void ai_deepseek_4__attention__zzprivate_place(const nn__doors* doors, unsigned int* over, const ai_deepseek_4__compressed* c,
                                                      uint64_t kv, uint64_t gate, uint64_t work, uint64_t HD, uint64_t RD, uint64_t pos,
                                                      uint64_t angles, float eps) {
    const uint64_t r = c->ratio, slot = pos % r, row = c->cache + 2u * (pos / r) * HD, tail = 2u * (HD - RD);
    const uint64_t* at = c->at;
    const uint64_t* room = c->room;
    uint64_t latent = kv;
    if (r != 1u) {
        doors->vector_copy((uint16_t*)(uintptr_t)(c->ring + 2u * slot * HD), (const uint16_t*)(uintptr_t)kv, HD);
        doors->vector_copy((uint16_t*)(uintptr_t)(c->ring + 2u * (r + slot) * HD), (const uint16_t*)(uintptr_t)gate, HD);
        if (slot + 1u != r) return;
        doors->index_pool((uint16_t*)(uintptr_t)work, (const uint16_t*)(uintptr_t)c->ring, (const uint16_t*)(uintptr_t)(c->ring + 2u * r * HD),
                          (const uint16_t*)(uintptr_t)c->zero, 0u, 1u, HD, r, over);
        latent = work;
    }
    doors->rmsnorm((uint16_t*)(uintptr_t)row, (const uint16_t*)(uintptr_t)latent, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__CMP__NORM], HD, eps,
                   over);
    const uint64_t first = pos + 1u - r;
    if (c->indexed) {
        /* the latent's index key, from the latent before its rope: through `wk`, normed, its tail turned at the group's
           first position — at row `pos / r` of the source's keys */
        const uint64_t lr = work + 2u * HD, ik = lr + 2u * HD, key = at[AI_DEEPSEEK_4__IDX__KEYS] + 2u * (pos / r) * c->dim;
        doors->hadamard_rotate((uint16_t*)(uintptr_t)lr, (const uint16_t*)(uintptr_t)row, (const uint16_t*)(uintptr_t)c->signs, HD, over);
        ai_deepseek_4__layer__zzprivate_gemv(doors, over, at[AI_DEEPSEEK_4__IDX__WK], room[AI_DEEPSEEK_4__IDX__WK], at[AI_DEEPSEEK_4__IDX__WK + 1u],
                                             room[AI_DEEPSEEK_4__IDX__WK + 1u], ik, lr, c->di[2], c->dim, HD);
        doors->rmsnorm((uint16_t*)(uintptr_t)key, (const uint16_t*)(uintptr_t)ik, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__IDX__KNORM], c->dim,
                       eps, over);
        doors->rope_interleaved((uint16_t*)(uintptr_t)(key + 2u * (c->dim - RD)), (const uint16_t*)(uintptr_t)(key + 2u * (c->dim - RD)),
                                (const float*)(uintptr_t)(angles + 4u * first * RD), 1u, c->dim, RD, 0u, over);
    }
    doors->rope_interleaved((uint16_t*)(uintptr_t)(row + tail), (const uint16_t*)(uintptr_t)(row + tail),
                            (const float*)(uintptr_t)(angles + 4u * first * RD), 1u, HD, RD, 0u, over);
}

/* The index source's query and a weight a head at a position: `iq` from the query's rank normed and rotated (`qnr`), `iw`
 * from the site's rotated input — a position's; a chunk's rows run them as one GEMM each. */
static void ai_deepseek_4__attention__zzprivate_index_project(const nn__doors* doors, unsigned int* over, const ai_deepseek_4__compressed* c,
                                                              uint64_t qnr, uint64_t hr, uint64_t iq, uint64_t iw, uint64_t QR, uint64_t H) {
    ai_deepseek_4__layer__zzprivate_gemv(doors, over, c->at[AI_DEEPSEEK_4__IDX__WQB], c->room[AI_DEEPSEEK_4__IDX__WQB],
                                         c->at[AI_DEEPSEEK_4__IDX__WQB + 1u], c->room[AI_DEEPSEEK_4__IDX__WQB + 1u], iq, qnr, c->di[0],
                                         c->heads * c->dim, QR);
    ai_deepseek_4__layer__zzprivate_gemv(doors, over, c->at[AI_DEEPSEEK_4__IDX__WP], c->room[AI_DEEPSEEK_4__IDX__WP],
                                         c->at[AI_DEEPSEEK_4__IDX__WP + 1u], c->room[AI_DEEPSEEK_4__IDX__WP + 1u], iw, hr, c->di[1], c->heads, H);
}

/* ⭐ ONE POSITION'S ATTENTION from its query `q` (its tails not yet turned) and its latent `kv` (not yet normed), the
 *   answer into `o` (which may be `q`): the latent normed into the ring's row `pos % window`, the tails of both turned at
 *   `pos`; where the layer compresses, the source's step (its latent `ck`, its gate `cg`, `cwork` its scratch) and the
 *   rows it attends gathered — past `topk` latents the indexer's choice, which an index source makes from its query `iq`
 *   and weights `iw` into `sel` and the layers after it read there; every head over the rows with its sink; the
 *   answer's tails turned back. A position's verb and a chunk's rows, a row at a time in position order, both run it. */
static void ai_deepseek_4__attention__zzprivate_core(const nn__doors* doors, unsigned int* over, const uint64_t* at, const float* r,
                                                     const ai_deepseek_4__compressed* cmp, uint64_t q, uint64_t kv, uint64_t ck, uint64_t cg,
                                                     uint64_t cwork, uint64_t iq, uint64_t iw, uint64_t sel, uint64_t o, uint64_t pos,
                                                     uint64_t heads, uint64_t HD, uint64_t RD, uint64_t W) {
    const uint64_t ring = at[AI_DEEPSEEK_4__ATT__RING], scores = at[AI_DEEPSEEK_4__ATT__SCORES];
    const uint64_t len = pos + 1u < W ? pos + 1u : W;
    const uint64_t row = ring + 2u * (pos % W) * HD, tail = 2u * (HD - RD), cs = at[AI_DEEPSEEK_4__ATT__ANGLES] + 4u * pos * RD;
    /* the position's row of the ring: the latent normed, its tail turned — the key and the value of every head */
    doors->rmsnorm((uint16_t*)(uintptr_t)row, (const uint16_t*)(uintptr_t)kv, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ATT__KV_NORM], HD,
                   r[AI_DEEPSEEK_4__ATT__EPS], over);
    doors->rope_interleaved((uint16_t*)(uintptr_t)(row + tail), (const uint16_t*)(uintptr_t)(row + tail), (const float*)(uintptr_t)cs, 1u, HD,
                            RD, 0u, over);
    doors->rope_interleaved((uint16_t*)(uintptr_t)(q + tail), (const uint16_t*)(uintptr_t)(q + tail), (const float*)(uintptr_t)cs, heads, HD,
                            RD, 0u, over);
    /* the compressed latents: the source writes its group's when it completes; every compressing layer attends them */
    uint64_t rows = ring, attend = len;
    if (cmp->ratio != 0u) {
        if (cmp->source)
            ai_deepseek_4__attention__zzprivate_place(doors, over, cmp, ck, cg, cwork, HD, RD, pos, at[AI_DEEPSEEK_4__ATT__ANGLES],
                                                      r[AI_DEEPSEEK_4__ATT__EPS]);
        const uint64_t latents = (pos + 1u) / cmp->ratio;
        doors->vector_copy((uint16_t*)(uintptr_t)cmp->gather, (const uint16_t*)(uintptr_t)ring, len * HD);
        if (latents <= cmp->topk) {
            if (latents != 0u)
                doors->vector_copy((uint16_t*)(uintptr_t)(cmp->gather + 2u * len * HD), (const uint16_t*)(uintptr_t)cmp->cache, latents * HD);
            attend = len + latents;
        } else {
            /* ⭐ THE INDEXER: an index source scores every latent and keeps the best `topk`; the layers after it reuse them */
            if (cmp->indexer) {
                const uint64_t IH = cmp->heads, ID = cmp->dim;
                doors->rope_interleaved((uint16_t*)(uintptr_t)(iq + 2u * (ID - RD)), (const uint16_t*)(uintptr_t)(iq + 2u * (ID - RD)),
                                        (const float*)(uintptr_t)cs, IH, ID, RD, 0u, over);
                doors->index_scores((float*)(uintptr_t)cmp->at[AI_DEEPSEEK_4__IDX__SCORES], (const uint16_t*)(uintptr_t)iq,
                                    (const uint16_t*)(uintptr_t)iw, (const uint16_t*)(uintptr_t)cmp->at[AI_DEEPSEEK_4__IDX__KEYS], IH, ID, latents,
                                    1.0f / sqrtf((float)ID));
                doors->index_select((uint32_t*)(uintptr_t)sel, (uint32_t*)(uintptr_t)(sel + 4u * cmp->topk),
                                    (const float*)(uintptr_t)cmp->at[AI_DEEPSEEK_4__IDX__SCORES], latents, cmp->topk, 1u, latents, 0u);
            }
            doors->index_gather((uint16_t*)(uintptr_t)(cmp->gather + 2u * len * HD), (const uint16_t*)(uintptr_t)cmp->cache,
                                (const uint32_t*)(uintptr_t)sel, cmp->topk, HD);
            attend = len + cmp->topk;
        }
        rows = cmp->gather;
    }
    /* every head over its rows, one kv head: the scores at 1/√head_dim, the sink, the rows mixed */
    doors->attention_scores_grouped((float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)rows, heads, 1u, HD, attend);
    doors->attention_softmax_sink((float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ATT__SINK], heads, attend);
    doors->attention_mix((uint16_t*)(uintptr_t)o, (const float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)rows, heads, 1u, HD, attend, over);
    doors->rope_interleaved((uint16_t*)(uintptr_t)(o + tail), (const uint16_t*)(uintptr_t)(o + tail), (const float*)(uintptr_t)cs, heads, HD,
                            RD, 1u, over);
}

/* `(ai_deepseek_4__attention streams table pos)` -> `streams`. The attention site at position `pos`: q through its rank, the
 * latent normed into the ring's row `pos % window`, the tails of both turned at `pos`, every head attending the ring's
 * filled rows with its sink, the answer's tails turned back, each group of heads through its own low-rank map, the
 * groups through the output's second map. */
/* An attention table, read and checked for every position up to `pos`: its planes, integers and reals, its shapes, its
 * compressed part. */
typedef struct ai_deepseek_4__attention_site {
    uint64_t at[AI_DEEPSEEK_4__ATT__TABLE], room[AI_DEEPSEEK_4__ATT__TABLE], v[AI_DEEPSEEK_4__ATT__TABLE];
    float r[AI_DEEPSEEK_4__ATT__TABLE];
    uint64_t H, M, heads, HD, RD, QR, OR, G, W, QW, GW, ON;   /* QW the query's width, GW a group's, ON the groups' answer */
    ai_deepseek_4__compressed cmp;
} ai_deepseek_4__attention_site;

/* The attention site's table read and checked for positions up to `pos`, a row of the streams in `s_room`: 0, or the
 * fault to answer with. */
static uint64_t ai_deepseek_4__attention__zzprivate_site(uint64_t table, uint64_t s_room, uint64_t pos, ai_deepseek_4__attention_site* s) {
    if (!ai_deepseek_4__table__zzpackage_read(table, AI_DEEPSEEK_4__ATT__PLANES, AI_DEEPSEEK_4__ATT__EPS, AI_DEEPSEEK_4__ATT__TABLE, s->at,
                                              s->room, s->v, s->r))
        return AI_DEEPSEEK_4__LAYER__FAULT_TABLE;
    const uint64_t* v = s->v;
    const uint64_t* room = s->room;
    const uint64_t H = v[AI_DEEPSEEK_4__ATT__HIDDEN], M = v[AI_DEEPSEEK_4__ATT__MULT], heads = v[AI_DEEPSEEK_4__ATT__HEADS];
    const uint64_t HD = v[AI_DEEPSEEK_4__ATT__HEAD_DIM], RD = v[AI_DEEPSEEK_4__ATT__ROPE], QR = v[AI_DEEPSEEK_4__ATT__Q_RANK];
    const uint64_t OR = v[AI_DEEPSEEK_4__ATT__O_RANK], G = v[AI_DEEPSEEK_4__ATT__GROUPS], W = v[AI_DEEPSEEK_4__ATT__WINDOW];
    if (!ai_deepseek_4__layer__zzprivate_mixes(room, s_room, H, M) || heads == 0u || heads > (1ull << 12) || HD == 0u || HD > (1ull << 16)
     || RD == 0u || RD % 2u != 0u || RD > HD || QR == 0u || QR > (1ull << 20) || OR == 0u || G == 0u || (heads * HD) % G != 0u
     || W == 0u || W > (1ull << 20) || pos >= (1ull << 32))
        return AI_DEEPSEEK_4__LAYER__FAULT_TABLE;
    const uint64_t QW = heads * HD, GW = QW / G, ON = G * OR;
    const uint64_t len = pos + 1u < W ? pos + 1u : W;
    if (!nn__primitives__fits(W * HD, room[AI_DEEPSEEK_4__ATT__RING]) || !nn__primitives__fits(2u * (pos + 1u) * RD, room[AI_DEEPSEEK_4__ATT__ANGLES])
     || !nn__primitives__fits(heads, room[AI_DEEPSEEK_4__ATT__SINK])
     || room[AI_DEEPSEEK_4__ATT__WOA] % ON != 0u || room[AI_DEEPSEEK_4__ATT__WOA + 1u] < 2u * ON)
        return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    const uint64_t cf = ai_deepseek_4__attention__zzprivate_compressed(table, s->at, room, HD, W, pos, &s->cmp);
    if (cf != 0u) return cf;
    const ai_deepseek_4__compressed* cmp = &s->cmp;
    const uint64_t latents_all = cmp->ratio != 0u ? (pos + 1u) / cmp->ratio : 0u;
    if (room[AI_DEEPSEEK_4__ATT__SCORES] / 4u < heads * (len + (latents_all < cmp->topk || cmp->ratio == 0u ? latents_all : cmp->topk)))
        return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    s->cmp.signs = s->at[AI_DEEPSEEK_4__ATT__SIGNS];
    s->H = H; s->M = M; s->heads = heads; s->HD = HD; s->RD = RD; s->QR = QR; s->OR = OR; s->G = G; s->W = W;
    s->QW = QW; s->GW = GW; s->ON = ON;
    return 0u;
}

static uint64_t ai_deepseek_4__attention__zzprivate_row(const sys__heap_node* argv, sys__engine__ctx* ctx, uint64_t pos,
                                                       const uint64_t* tp) {
    uint64_t streams = 0, s_room = 0;
    if (!ai_deepseek_4__layer__zzprivate_open(argv, &streams, &s_room)) return SYS__OPCODES__FAULT_TYPE;
    ai_deepseek_4__attention_site s;
    const uint64_t sf = ai_deepseek_4__attention__zzprivate_site(argv[1].args[0], s_room, pos, &s);
    if (sf != 0u) return sf;
    const uint64_t* at = s.at;
    const uint64_t* room = s.room;
    const uint64_t* d = &s.v[AI_DEEPSEEK_4__ATT__D];          /* wq_a wkv wq_b wo_a wo_b */
    const uint64_t H = s.H, M = s.M, heads = s.heads, HD = s.HD, RD = s.RD, QR = s.QR, OR = s.OR, G = s.G, W = s.W;
    const uint64_t QW = s.QW, GW = s.GW, ON = s.ON;
    const ai_deepseek_4__compressed* cmp = &s.cmp;
    if (tp != 0 && !nn__primitives__fits(H, tp[2])) return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    /* scratch: xin, floats; then halves: h · hr · [qa kv] · qn · qnr · q · o · or · og · ogr · y · [the compressor's kv,
       gate, pooled latent, rotated latent, index key] · [the indexer's query and weights] */
    const uint64_t need = 2u * (5u * H + 3u * QR + HD + 3u * QW + 2u * ON + 4u * HD + cmp->dim + cmp->heads * cmp->dim + cmp->heads);
    if (room[AI_DEEPSEEK_4__ATT__SCRATCH] < need) return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    const uint64_t xin = at[AI_DEEPSEEK_4__ATT__SCRATCH], h = xin + 4u * H, hr = h + 2u * H, qa = hr + 2u * H, kv = qa + 2u * QR;
    const uint64_t qn = kv + 2u * HD, qnr = qn + 2u * QR, q = qnr + 2u * QR, o = q + 2u * QW, orot = o + 2u * QW;
    const uint64_t og = orot + 2u * QW, ogr = og + 2u * ON, y = ogr + 2u * ON;
    const uint64_t ck = y + 2u * H, cg = ck + 2u * HD, cwork = cg + 2u * HD;
    const uint64_t iq = ck + 2u * (4u * HD + cmp->dim), iw = iq + 2u * cmp->heads * cmp->dim;
    const uint64_t signs = at[AI_DEEPSEEK_4__ATT__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    unsigned int* over = ctx->fault_word;

    ai_deepseek_4__layer__zzprivate_collapse(doors, over, at, at[AI_DEEPSEEK_4__OWN], at[AI_DEEPSEEK_4__EARLIER], streams, xin, h, H, M,
                                             s.v[AI_DEEPSEEK_4__ATT__ITERS], s.r[AI_DEEPSEEK_4__ATT__EPS], s.r[AI_DEEPSEEK_4__ATT__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)hr, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, H, over);
    nn__turboquant__groups g = {};
    ai_deepseek_4__table__zzpackage_group(&g, 0u, at, AI_DEEPSEEK_4__ATT__WQA, QR, d[0], 0u);
    ai_deepseek_4__table__zzpackage_group(&g, 1u, at, AI_DEEPSEEK_4__ATT__WKV, HD, d[1], QR);
    g.count = 2u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)qa, g, (const uint16_t*)(uintptr_t)hr, H, over);
    doors->rmsnorm((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qa, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ATT__Q_NORM], QR,
                   s.r[AI_DEEPSEEK_4__ATT__EPS], over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)qnr, (const uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)signs, QR, over);
    ai_deepseek_4__layer__zzprivate_gemv(doors, over, at[AI_DEEPSEEK_4__ATT__WQB], room[AI_DEEPSEEK_4__ATT__WQB], at[AI_DEEPSEEK_4__ATT__WQB + 1u],
                                         room[AI_DEEPSEEK_4__ATT__WQB + 1u], q, qnr, d[2], QW, QR);
    /* the source's compressor's maps, and past `topk` latents the index source's query and weights */
    if (cmp->ratio != 0u && cmp->source) ai_deepseek_4__attention__zzprivate_project(doors, over, cmp, hr, ck, H, HD);
    if (cmp->ratio != 0u && cmp->indexer && (pos + 1u) / cmp->ratio > cmp->topk)
        ai_deepseek_4__attention__zzprivate_index_project(doors, over, cmp, qnr, hr, iq, iw, QR, H);
    ai_deepseek_4__attention__zzprivate_core(doors, over, at, s.r, cmp, q, kv, ck, cg, cwork, iq, iw, cmp->at[AI_DEEPSEEK_4__IDX__SELECT], o,
                                             pos, heads, HD, RD, W);
    /* ⭐ THE OUTPUT, LOW-RANK IN GROUPS: group `j`'s heads (`GW` halves of the answer) through its own `OR` rows of wo_a —
     *   one record of `G · OR` rows, so group `j`'s rows start `j · OR` rows in, its codes and its scales alike */
    doors->hadamard_rotate((uint16_t*)(uintptr_t)orot, (const uint16_t*)(uintptr_t)o, (const uint16_t*)(uintptr_t)signs, QW, over);
    const uint64_t woa = at[AI_DEEPSEEK_4__ATT__WOA], woa_l = at[AI_DEEPSEEK_4__ATT__WOA + 1u], row_bytes = room[AI_DEEPSEEK_4__ATT__WOA] / ON;
    for (uint64_t j = 0u; j < G; ++j)
        ai_deepseek_4__layer__zzprivate_gemv(doors, over, woa + j * OR * row_bytes, OR * row_bytes, woa_l + 2u * j * OR, 2u * OR,
                                             og + 2u * j * OR, orot + 2u * j * GW, d[3], OR, GW);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)ogr, (const uint16_t*)(uintptr_t)og, (const uint16_t*)(uintptr_t)signs, ON, over);
    ai_deepseek_4__layer__zzprivate_gemv(doors, over, at[AI_DEEPSEEK_4__ATT__WOB], room[AI_DEEPSEEK_4__ATT__WOB], at[AI_DEEPSEEK_4__ATT__WOB + 1u],
                                         room[AI_DEEPSEEK_4__ATT__WOB + 1u], y, ogr, d[4], H, ON);
    /* the mask on wo_b's answer, this card's columns of it: the group's sum below adds the cards' parts as it adds wo_b's */
    const uint64_t mf = ai_deepseek_4__table__zzpackage_masked(doors, over, argv[1].args[0],
                                                               ai_deepseek_4__attention__zzprivate_mask_first(argv[1].args[0]), y, ogr, H, ON,
                                                               AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    if (mf != 0u) return mf;
    /* ⭐ UNDER TENSOR PARALLELISM this card's heads and groups gave a partial of the output: the group's sum of them is the
     *   output, in the group's own buffer (▶ `ai_deepseek_4__layer__zzprivate_tp`) */
    uint64_t out = y;
    if (tp != 0) {
        const uint64_t f = nn__partials__reduce_onto(ctx, doors, tp[0], y, tp[1], H, 0u);
        if (f != 0u) return f;
        out = tp[1];
    }
    doors->hyper_post_fp32((float*)(uintptr_t)streams, (const float*)(uintptr_t)streams, (const uint16_t*)(uintptr_t)out,
                           (const float*)(uintptr_t)at[AI_DEEPSEEK_4__OWN], H, M);
    return 0u;
}

/* ⭐ A TENSOR-PARALLEL GROUP's TWO TRAILING ARGUMENTS — its number and the buffer its sums land in, `H` halves at least —
 * into `tp`: the group, the buffer, its room in halves. ⛳ THE SUM HAS A BUFFER OF ITS OWN: between two reduces the group's other cards read it (nn's
 * `contracts/objects/partials.cuh`), so nothing but a reduce may write it. Answers false when they are not there or wrong. */
static inline bool ai_deepseek_4__layer__zzprivate_tp(const sys__heap_node* argv, unsigned first, uint64_t H, uint64_t* tp) {
    uint64_t sum = 0, s_room = 0;
    if (argv[first].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[first + 1u], &sum, &s_room)
     || !nn__primitives__fits(H, s_room))
        return false;
    tp[0] = argv[first].args[0];
    tp[1] = sum;
    tp[2] = s_room;
    return true;
}

/* The MoE site's table, for its three card verbs. */
static inline bool ai_deepseek_4__layer__zzprivate_moe(uint64_t table, uint64_t* at, uint64_t* room, uint64_t* v, float* r) {
    if (!ai_deepseek_4__table__zzpackage_read(table, AI_DEEPSEEK_4__MOE__PLANES, AI_DEEPSEEK_4__MOE__EPS, AI_DEEPSEEK_4__MOE__TABLE,
                                              at, room, v, r))
        return false;
    const uint64_t H = v[AI_DEEPSEEK_4__MOE__HIDDEN], E = v[AI_DEEPSEEK_4__MOE__EXPERTS];
    const uint64_t K = v[AI_DEEPSEEK_4__MOE__TOP_K], I = v[AI_DEEPSEEK_4__MOE__INTER];
    return H != 0u && H <= (1ull << 24) && E != 0u && E <= (1ull << 20) && K != 0u && K <= E && K <= AI_DEEPSEEK_4__TOP_K_MAX
        && I != 0u && I <= (1ull << 24) && room[AI_DEEPSEEK_4__MOE__SCRATCH] >= 2u * (4u * H + E) + 8u * I;
}

/* `(ai_deepseek_4__route streams table hand picks)` -> `hand`. The MoE site to the router: the site's mix written, the
 * streams collapsed and normed, the input rotated into the hand, the router's `k` picks into `picks` and their weights
 * after the input. */
static sys__heap_node ai_deepseek_4__route__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, hand = 0, h_room = 0;
    if (!ai_deepseek_4__layer__zzprivate_open(argv, &streams, &s_room) || !nn__primitives__room(&argv[2], &hand, &h_room)
     || !sys__heap_node__carries_reference(argv[3].dtype) || !sys__node_array__is(argv[3].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_DEEPSEEK_4__MOE__TABLE], room[AI_DEEPSEEK_4__MOE__TABLE], v[AI_DEEPSEEK_4__MOE__TABLE];
    float r[AI_DEEPSEEK_4__MOE__TABLE];
    if (!ai_deepseek_4__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__MOE__HIDDEN], M = v[AI_DEEPSEEK_4__MOE__MULT], E = v[AI_DEEPSEEK_4__MOE__EXPERTS];
    const uint64_t K = v[AI_DEEPSEEK_4__MOE__TOP_K];
    if (!ai_deepseek_4__layer__zzprivate_mixes(room, s_room, H, M) || !nn__primitives__fits(H + K, h_room)
     || !nn__primitives__fits(E, room[AI_DEEPSEEK_4__MOE__BIAS]) || !nn__primitives__fits(E * H, room[AI_DEEPSEEK_4__MOE__ROUTER]))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    /* scratch: the shared expert's answer, kept for `close` · xin (floats) · h · the router's logits */
    const uint64_t scratch = at[AI_DEEPSEEK_4__MOE__SCRATCH], xin = scratch + 2u * H, h = xin + 4u * H, logits = h + 2u * H;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    ai_deepseek_4__layer__zzprivate_collapse(doors, over, at, at[AI_DEEPSEEK_4__OWN], at[AI_DEEPSEEK_4__EARLIER], streams, xin, h, H, M,
                                             v[AI_DEEPSEEK_4__MOE__ITERS], r[AI_DEEPSEEK_4__MOE__EPS], r[AI_DEEPSEEK_4__MOE__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)hand, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__MOE__SIGNS], H, over);
    doors->expert_multiply_fp16((uint16_t*)(uintptr_t)logits, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__MOE__ROUTER], (const uint16_t*)(uintptr_t)h,
                                E, H, over);
    const uint64_t landed = nn__routed__top_k_scored(ctx, doors, argv[3].args[0], hand + 2u * H, logits, at[AI_DEEPSEEK_4__MOE__BIAS],
                                                     r[AI_DEEPSEEK_4__MOE__SCALE], E, K, NN__VECTOR__SCORE_SQRT_SOFTPLUS, 0);
    if (landed != 0u) return sys__engine__abi__error(landed);
    return nn__doors_answer(&argv[2]);
}

/* `(ai_deepseek_4__shared hand table)` -> `hand`. The shared expert on the hand's input, its answer into the scratch's first
 * `hidden` halves — what the card does while the routed experts run elsewhere: gate and up in one grouped launch, the
 * clamped swiglu, rotated, down. */
static sys__heap_node ai_deepseek_4__shared__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hand = 0, h_room = 0;
    if (!nn__primitives__room(&argv[0], &hand, &h_room) || !sys__heap_node__carries_reference(argv[1].dtype)
     || !sys__node_array__is(argv[1].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_DEEPSEEK_4__MOE__TABLE], room[AI_DEEPSEEK_4__MOE__TABLE], v[AI_DEEPSEEK_4__MOE__TABLE];
    float r[AI_DEEPSEEK_4__MOE__TABLE];
    if (!ai_deepseek_4__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__MOE__HIDDEN], I = v[AI_DEEPSEEK_4__MOE__INTER], E = v[AI_DEEPSEEK_4__MOE__EXPERTS];
    const uint64_t* d = &v[AI_DEEPSEEK_4__MOE__D];
    if (!nn__primitives__fits(H, h_room)) return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t sh = at[AI_DEEPSEEK_4__MOE__SCRATCH], gu = sh + 2u * (4u * H + E), act = gu + 4u * I, actr = act + 2u * I;
    const uint64_t signs = at[AI_DEEPSEEK_4__MOE__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    nn__turboquant__groups g = {};
    ai_deepseek_4__table__zzpackage_group(&g, 0u, at, AI_DEEPSEEK_4__MOE__GATE, I, d[0], 0u);
    ai_deepseek_4__table__zzpackage_group(&g, 1u, at, AI_DEEPSEEK_4__MOE__UP, I, d[1], I);
    g.count = 2u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)gu, g, (const uint16_t*)(uintptr_t)hand, H, over);
    doors->swiglu_clamped((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, 1u, I, r[AI_DEEPSEEK_4__MOE__LIMIT], over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)signs, I, over);
    ai_deepseek_4__layer__zzprivate_gemv(doors, over, at[AI_DEEPSEEK_4__MOE__DOWN], room[AI_DEEPSEEK_4__MOE__DOWN], at[AI_DEEPSEEK_4__MOE__DOWN + 1u],
                                         room[AI_DEEPSEEK_4__MOE__DOWN + 1u], sh, actr, d[2], H, I);
    const uint64_t mf = ai_deepseek_4__table__zzpackage_masked(doors, over, argv[1].args[0], AI_DEEPSEEK_4__MOE__MASK, sh, actr, H, I,
                                                               AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    if (mf != 0u) return sys__engine__abi__error(mf);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_deepseek_4__close streams table routed)` -> `streams`. The routed experts' sum added to the shared expert's, and the
 * streams mixed again with it by the site's own mix. */
static sys__heap_node ai_deepseek_4__close__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, routed = 0, r_room = 0;
    if (!ai_deepseek_4__layer__zzprivate_open(argv, &streams, &s_room) || !nn__primitives__room(&argv[2], &routed, &r_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_DEEPSEEK_4__MOE__TABLE], room[AI_DEEPSEEK_4__MOE__TABLE], v[AI_DEEPSEEK_4__MOE__TABLE];
    float r[AI_DEEPSEEK_4__MOE__TABLE];
    if (!ai_deepseek_4__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__MOE__HIDDEN], M = v[AI_DEEPSEEK_4__MOE__MULT];
    if (!ai_deepseek_4__layer__zzprivate_mixes(room, s_room, H, M) || !nn__primitives__fits(H, r_room))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t sh = at[AI_DEEPSEEK_4__MOE__SCRATCH];
    uint64_t y = sh + 2u * H;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    uint64_t tp[3] = {0u, 0u, 0u};
    if (argc == 5u && !ai_deepseek_4__layer__zzprivate_tp(argv, 3u, H, tp)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    doors->vector_add((uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)sh, (const uint16_t*)(uintptr_t)routed, H, ctx->fault_word);
    /* under tensor parallelism: this card's part of the shared expert and its share of the routed — the CPU's sum on one
     *   card, zeros on the others — and the group's sum of them is the site's output */
    if (argc == 5u) {
        const uint64_t f = nn__partials__reduce_onto(ctx, doors, tp[0], y, tp[1], H, 0u);
        if (f != 0u) return sys__engine__abi__error(f);
        y = tp[1];
    }
    doors->hyper_post_fp32((float*)(uintptr_t)streams, (const float*)(uintptr_t)streams, (const uint16_t*)(uintptr_t)y,
                           (const float*)(uintptr_t)at[AI_DEEPSEEK_4__OWN], H, M);
    return nn__doors_answer(&argv[0]);
}

/* The routed experts as nn runs them (▶ nn's `nn__routed`), out of an experts table: its shape, its planes' places in a
 * slot, the experts this worker holds, its arithmetic, the clamped SwiGLU's limit and its scratch. */
static void ai_deepseek_4__experts__zzprivate_routed(const uint64_t* at, const uint64_t* room, const uint64_t* v, const float* r,
                                                     nn__routed* out) {
    const nn__routed none = {};
    *out = none;
    const uint64_t H = v[AI_DEEPSEEK_4__EXP__HIDDEN], K = v[AI_DEEPSEEK_4__EXP__TOP_K];
    out->layer = v[AI_DEEPSEEK_4__EXP__LAYER]; out->type = v[AI_DEEPSEEK_4__EXP__TYPE];
    out->hidden = H; out->inter = v[AI_DEEPSEEK_4__EXP__INTER]; out->experts = v[AI_DEEPSEEK_4__EXP__EXPERTS]; out->top_k = K;
    out->bits = v[AI_DEEPSEEK_4__EXP__D];
    out->up_lut = v[AI_DEEPSEEK_4__EXP__UP_LUT]; out->down = v[AI_DEEPSEEK_4__EXP__DOWN]; out->down_lut = v[AI_DEEPSEEK_4__EXP__DOWN_LUT];
    out->first = v[AI_DEEPSEEK_4__EXP__FIRST]; out->count = v[AI_DEEPSEEK_4__EXP__COUNT];
    out->int8 = v[AI_DEEPSEEK_4__EXP__INT8] == 1u; out->limit = r[AI_DEEPSEEK_4__EXP__LIMIT];
    out->signs = at[AI_DEEPSEEK_4__EXP__SIGNS]; out->zero = at[AI_DEEPSEEK_4__EXP__ZERO];
    out->scratch = at[AI_DEEPSEEK_4__EXP__SCRATCH]; out->scratch_room = room[AI_DEEPSEEK_4__EXP__SCRATCH];
    out->hand_bytes = AI_DEEPSEEK_4__HAND(H, K); out->picks_at = AI_DEEPSEEK_4__HAND_PICKS(H, K); out->batch = AI_DEEPSEEK_4__EXP__BATCH;
}

/* `(ai_deepseek_4__experts hand picks table routed [nn__expert_tier])` -> `routed`: Σ w_j · down_j(swiglu(gate_up_j(x)))
 * over the picks this worker holds, the SwiGLU clamped — as nn runs a position's routed experts (▶ nn's
 * `nn__routed__position`). Handed the binding of the cards' tiers, a card computes the picks its tier holds — in a group
 * the ones the cards' visit shares out to it — and marks them held, so the CPU's experts leave them (NN-99). */
static sys__heap_node ai_deepseek_4__experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hand = 0, h_room = 0, routed = 0, r_room = 0;
    if (!nn__primitives__room(&argv[0], &hand, &h_room) || !nn__primitives__room(&argv[3], &routed, &r_room)
     || !sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_DEEPSEEK_4__EXP__TABLE], room[AI_DEEPSEEK_4__EXP__TABLE], v[AI_DEEPSEEK_4__EXP__TABLE];
    float r[AI_DEEPSEEK_4__EXP__TABLE];
    if (!ai_deepseek_4__table__zzpackage_read(argv[2].args[0], AI_DEEPSEEK_4__EXP__PLANES, AI_DEEPSEEK_4__EXP__LIMIT,
                                              AI_DEEPSEEK_4__EXP__TABLE, at, room, v, r))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__EXP__HIDDEN], K = v[AI_DEEPSEEK_4__EXP__TOP_K];
    if (K > AI_DEEPSEEK_4__TOP_K_MAX || v[AI_DEEPSEEK_4__EXP__INT8] > 1u || !nn__primitives__fits(H + K, h_room)
     || !nn__primitives__fits(H, r_room) || !nn__primitives__fits(H, room[AI_DEEPSEEK_4__EXP__ZERO]))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    nn__routed rt;
    ai_deepseek_4__experts__zzprivate_routed(at, room, v, r, &rt);
    nn__expert_tier tier;
    const bool holds = argc == 5u;
    if (holds && (!sys__heap_node__carries_reference(argv[4].dtype) || !nn__expert_tier__of(argv[4].args[0], rt.layer, &tier)))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t f = nn__routed__position(ctx, doors, &rt, hand, argv[1].args[0], holds ? &tier : 0, routed, 0);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[3]);
}

/* ══ ⭐ A PROMPT AS ROWS — EACH SITE OVER A CHUNK ══════════════════════════════════════════════════════════════════
 * A chunk runs a layer at a time. Every matrix of a site runs ONCE over the chunk, through nn's expert-major GEMM
 * (`expert_groups`, `expert_rows`) with each row its own pair, so its weights are read once a chunk and not once a row;
 * the rotations and the query's norm are one launch each over every row. A row at a time, in position order, runs what
 * it reads of the rows before it (▶ `ai_deepseek_4__attention__zzprivate_core`): its latent into the window's ring, the
 * compressor's group, the indexer's choice, every head over its rows — and the streams' collapse and mix, whose doors
 * take a row of floats, and the router's picks, a row's own hand. The routed experts run ONCE over the chunk on the CPU,
 * EXPERT-MAJOR, each expert the chunk picked read once and run over every row that picked it (▶ nn's
 * `nn__routed__rows`). Row `t` of `streams` is position `first + t`, its mixes a record of AI_DEEPSEEK_4__MIX_BYTES
 * each: the attention site's own in `amixes`, the MoE site's own in `bmixes`.
 * ⛳ THE INDEXER'S CHOICE IS A ROW'S OWN: an index source writes row `t`'s `t` choices into its selection, and the layers
 *   after it read row `t`'s there — the chunk runs a layer over every row before the next layer reads it.
 * ⛳ THE `work` BUFFER holds the chunk's pairs and its rows of each step; the table's scratch keeps what one row needs. */

/* The next `bytes` of the work buffer, 256-aligned; 0 when it has no room left. */
static inline uint64_t ai_deepseek_4__rows__zzprivate_take(uint64_t* next, uint64_t end, uint64_t bytes) {
    const uint64_t at = (*next + 255u) & ~255ull;
    if (at > end || end - at < bytes) return 0u;
    *next = at + bytes;
    return at;
}

/* Where the rows' outputs `y` (`n` rows of `H` halves) are placed from (`out`): `y` itself, or under a group (`tp`) the
 * group's sum of every card's rows, one reduce, in the group's buffer (▶ `ai_deepseek_4__layer__zzprivate_tp`). 0, or the
 * fault. */
static inline uint64_t ai_deepseek_4__rows__zzprivate_sum(sys__engine__ctx* ctx, const nn__doors* doors, uint64_t y, uint64_t n,
                                                          uint64_t H, const uint64_t* tp, uint64_t* out) {
    *out = y;
    if (tp == 0) return 0u;
    if (!nn__primitives__fits(n * H, tp[2])) return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    const uint64_t f = nn__partials__reduce_onto(ctx, doors, tp[0], y, tp[1], n * H, 0u);
    if (f != 0u) return f;
    *out = tp[1];
    return 0u;
}

/* The chunk's pairs, a row of `x` and a row of `out` each: `n` of `(t, t)`; `n` of `(t, 2t)` and `n` of `(t, 2t + 1)` — a
 * gate's answer and an up's side by side a row, as the clamped swiglu reads them; then for each of the attention output's
 * `groups` maps `n` of `(t · groups + j, t · groups + j)` — row `t`'s heads of group `j` into its rows of the groups'
 * answer. */
#define AI_DEEPSEEK_4__ROWS__PAIRS(n, groups)  (8ull * (n) * (3ull + (groups)))
static bool ai_deepseek_4__rows__zzprivate_pairs(sys__engine__ctx* ctx, uint64_t pairs, uint64_t n, uint64_t groups) {
    uint32_t* p = (uint32_t*)malloc(AI_DEEPSEEK_4__ROWS__PAIRS(n, groups));
    if (p == 0) return false;
    for (uint64_t t = 0u; t < n; ++t) {
        p[2u * t] = (uint32_t)t;                  p[2u * t + 1u] = (uint32_t)t;
        p[2u * (n + t)] = (uint32_t)t;            p[2u * (n + t) + 1u] = (uint32_t)(2u * t);
        p[2u * (2u * n + t)] = (uint32_t)t;       p[2u * (2u * n + t) + 1u] = (uint32_t)(2u * t + 1u);
        for (uint64_t j = 0u; j < groups; ++j) {
            const uint64_t i = (3u + j) * n + t;
            p[2u * i] = (uint32_t)(t * groups + j);   p[2u * i + 1u] = (uint32_t)(t * groups + j);
        }
    }
    const bool ok = sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)pairs, p, AI_DEEPSEEK_4__ROWS__PAIRS(n, groups));
    free(p);
    return ok;
}

/* `out = W x` for every row a pair names: a matrix (codes, then scales) over the rows of `x`, `cols` wide. */
static inline void ai_deepseek_4__rows__zzprivate_gemm(const nn__doors* doors, unsigned int* over, uint64_t codes, uint64_t c_room, uint64_t luts,
                                                       uint64_t l_room, uint64_t out, uint64_t x, uint64_t pairs, uint64_t n, uint64_t d,
                                                       uint64_t rows, uint64_t cols) {
    doors->expert_rows((uint16_t*)(uintptr_t)out, (const uint8_t*)(uintptr_t)codes, c_room, (const uint8_t*)(uintptr_t)luts, l_room,
                       (const uint16_t*)(uintptr_t)x, (const uint32_t*)(uintptr_t)pairs, n, d, rows, cols, over);
}

/* A matrix as entry `i` of a grouped launch over the rows its pairs name. */
static inline void ai_deepseek_4__rows__zzprivate_group(nn__expert__groups* g, unsigned i, uint64_t codes, uint64_t luts, uint64_t rows,
                                                        uint64_t d, uint64_t out_at, uint64_t pairs_at, uint64_t pairs) {
    g->codes[i] = codes; g->luts[i] = luts; g->out_rows[i] = rows; g->d[i] = d;
    g->out_at[i] = out_at; g->pairs_at[i] = pairs_at; g->pairs[i] = pairs;
}

/* `(ai_deepseek_4__attention_rows streams table first n amixes bmixes work [group sums])` -> `streams`. The attention site
 * over `n` rows: each row's own mix into its record of `amixes`, its streams collapsed by its record of `bmixes` and normed;
 * the query's rank, the latent, the compressor's maps and the indexer's over every row; a row at a time its attention; the
 * grouped output over every row; each row's streams mixed again — under a group, this card's heads and groups', every
 * card's rows of the output summed first (▶ `ai_deepseek_4__layer__zzprivate_tp`). */
static sys__heap_node ai_deepseek_4__attention_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u && argc != 9u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, amixes = 0, a_room = 0, bmixes = 0, b_room = 0, work = 0, w_room = 0;
    if (!ai_deepseek_4__layer__zzprivate_open(argv, &streams, &s_room) || argv[2].dtype != SYS__KIND__VALUE_INT
     || argv[3].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[4], &amixes, &a_room)
     || !nn__primitives__room(&argv[5], &bmixes, &b_room) || !nn__primitives__room(&argv[6], &work, &w_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t first = argv[2].args[0], n = argv[3].args[0], mix = AI_DEEPSEEK_4__MIX_BYTES;
    if (n == 0u || n > AI_DEEPSEEK_4__ROWS_MAX || first >= (1ull << 32)) return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    uint64_t tp[3] = {0u, 0u, 0u};
    if (argc == 9u && !ai_deepseek_4__layer__zzprivate_tp(argv, 7u, 1u, tp)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_deepseek_4__attention_site s;
    const uint64_t sf = ai_deepseek_4__attention__zzprivate_site(argv[1].args[0], s_room, first + n - 1u, &s);
    if (sf != 0u) return sys__engine__abi__error(sf);
    const uint64_t* at = s.at;
    const uint64_t* room = s.room;
    const uint64_t* d = &s.v[AI_DEEPSEEK_4__ATT__D];          /* wq_a wkv wq_b wo_a wo_b */
    const uint64_t H = s.H, M = s.M, heads = s.heads, HD = s.HD, RD = s.RD, QR = s.QR, OR = s.OR, G = s.G, W = s.W;
    const uint64_t QW = s.QW, GW = s.GW, ON = s.ON, SB = 4u * M * H;
    const ai_deepseek_4__compressed* cmp = &s.cmp;
    const bool source = cmp->ratio != 0u && cmp->source;
    const bool indexing = cmp->ratio != 0u && cmp->indexer && (first + n) / cmp->ratio > cmp->topk;
    const uint64_t sel_bytes = 4u * (cmp->topk + 1u), IH = cmp->heads, ID = cmp->dim;
    if (s_room / SB < n || a_room / mix < n || b_room / mix < n || room[AI_DEEPSEEK_4__ATT__SCRATCH] < 4u * H + 2u * (2u * HD + ID)
     || (cmp->ratio != 0u && cmp->indexed && cmp->room[AI_DEEPSEEK_4__IDX__SELECT] / sel_bytes < n))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_ROOM);
    uint64_t next = work;
    const uint64_t end = work + w_room;
    const uint64_t pairs = ai_deepseek_4__rows__zzprivate_take(&next, end, AI_DEEPSEEK_4__ROWS__PAIRS(n, G));
    const uint64_t h = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * H);         /* then the output, `y` */
    const uint64_t hr = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t qa = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * (QR + HD)); /* wq_a's rows, then wkv's */
    const uint64_t qn = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * QR);
    const uint64_t qnr = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * QR);
    const uint64_t q = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * QW);        /* the answer in place, then `ogr` */
    const uint64_t orot = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * QW);
    const uint64_t og = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * ON);
    const uint64_t ck = source ? ai_deepseek_4__rows__zzprivate_take(&next, end, 4u * n * HD) : 1u;   /* the latents, then the gates */
    const uint64_t iq = indexing ? ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * IH * ID) : 1u;
    const uint64_t iw = indexing ? ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * IH) : 1u;
    if (pairs == 0u || h == 0u || hr == 0u || qa == 0u || qn == 0u || qnr == 0u || q == 0u || orot == 0u || og == 0u || ck == 0u
     || iq == 0u || iw == 0u)
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_ROOM);
    const uint64_t y = h, ogr = q, kv = qa + 2u * n * QR;
    const uint64_t xin = at[AI_DEEPSEEK_4__ATT__SCRATCH], cwork = xin + 4u * H, signs = at[AI_DEEPSEEK_4__ATT__SIGNS];
    const float eps = s.r[AI_DEEPSEEK_4__ATT__EPS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_deepseek_4__rows__zzprivate_pairs(ctx, pairs, n, G)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    for (uint64_t t = 0u; t < n; ++t)
        ai_deepseek_4__layer__zzprivate_collapse(doors, over, at, amixes + t * mix, bmixes + t * mix, streams + t * SB, xin, h + 2u * t * H,
                                                 H, M, s.v[AI_DEEPSEEK_4__ATT__ITERS], eps, s.r[AI_DEEPSEEK_4__ATT__HC_EPS]);
    /* the matrices over every row: wq_a and wkv together, the query's rank normed and rotated, wq_b, and the source's and
       the index source's own */
    doors->hadamard_rotate((uint16_t*)(uintptr_t)hr, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, n * H, over);
    nn__expert__groups g = {};
    ai_deepseek_4__rows__zzprivate_group(&g, 0u, at[AI_DEEPSEEK_4__ATT__WQA], at[AI_DEEPSEEK_4__ATT__WQA + 1u], QR, d[0], 0u, 0u, n);
    ai_deepseek_4__rows__zzprivate_group(&g, 1u, at[AI_DEEPSEEK_4__ATT__WKV], at[AI_DEEPSEEK_4__ATT__WKV + 1u], HD, d[1], n * QR, 0u, n);
    g.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)qa, g, (const uint16_t*)(uintptr_t)hr, (const uint32_t*)(uintptr_t)pairs, H, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qa, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ATT__Q_NORM], QR, n,
                        QR, QR, eps, over);
    doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)qnr, (const uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)signs, QR, n, over);
    ai_deepseek_4__rows__zzprivate_gemm(doors, over, at[AI_DEEPSEEK_4__ATT__WQB], room[AI_DEEPSEEK_4__ATT__WQB], at[AI_DEEPSEEK_4__ATT__WQB + 1u],
                                        room[AI_DEEPSEEK_4__ATT__WQB + 1u], q, qnr, pairs, n, d[2], QW, QR);
    if (source && cmp->ratio == 1u) {
        ai_deepseek_4__rows__zzprivate_gemm(doors, over, cmp->at[AI_DEEPSEEK_4__CMP__WKV], cmp->room[AI_DEEPSEEK_4__CMP__WKV],
                                            cmp->at[AI_DEEPSEEK_4__CMP__WKV + 1u], cmp->room[AI_DEEPSEEK_4__CMP__WKV + 1u], ck, hr, pairs, n,
                                            cmp->d[0], HD, H);
    } else if (source) {
        nn__expert__groups c = {};
        ai_deepseek_4__rows__zzprivate_group(&c, 0u, cmp->at[AI_DEEPSEEK_4__CMP__WKV], cmp->at[AI_DEEPSEEK_4__CMP__WKV + 1u], HD, cmp->d[0], 0u,
                                             0u, n);
        ai_deepseek_4__rows__zzprivate_group(&c, 1u, cmp->at[AI_DEEPSEEK_4__CMP__WGATE], cmp->at[AI_DEEPSEEK_4__CMP__WGATE + 1u], HD, cmp->d[1],
                                             n * HD, 0u, n);
        c.count = 2u;
        doors->expert_groups((uint16_t*)(uintptr_t)ck, c, (const uint16_t*)(uintptr_t)hr, (const uint32_t*)(uintptr_t)pairs, H, over);
    }
    if (indexing) {
        ai_deepseek_4__rows__zzprivate_gemm(doors, over, cmp->at[AI_DEEPSEEK_4__IDX__WQB], cmp->room[AI_DEEPSEEK_4__IDX__WQB],
                                            cmp->at[AI_DEEPSEEK_4__IDX__WQB + 1u], cmp->room[AI_DEEPSEEK_4__IDX__WQB + 1u], iq, qnr, pairs, n,
                                            cmp->di[0], IH * ID, QR);
        ai_deepseek_4__rows__zzprivate_gemm(doors, over, cmp->at[AI_DEEPSEEK_4__IDX__WP], cmp->room[AI_DEEPSEEK_4__IDX__WP],
                                            cmp->at[AI_DEEPSEEK_4__IDX__WP + 1u], cmp->room[AI_DEEPSEEK_4__IDX__WP + 1u], iw, hr, pairs, n,
                                            cmp->di[1], IH, H);
    }
    /* a row at a time, in position order: what it reads of the rows before it */
    for (uint64_t t = 0u; t < n; ++t)
        ai_deepseek_4__attention__zzprivate_core(doors, over, at, s.r, cmp, q + 2u * t * QW, kv + 2u * t * HD, ck + 2u * t * HD,
                                                 ck + 2u * (n + t) * HD, cwork, iq + 2u * t * IH * ID, iw + 2u * t * IH,
                                                 cmp->at[AI_DEEPSEEK_4__IDX__SELECT] + t * sel_bytes, q + 2u * t * QW, first + t, heads, HD,
                                                 RD, W);
    /* ⭐ THE OUTPUT over every row: row `t`'s group `j` of heads, `GW` wide, through group `j`'s `OR` rows of wo_a — the
     *   answer viewed as `n · G` rows of `GW`, the groups' as `n · G` rows of `OR` — then wo_b */
    doors->hadamard_rotate((uint16_t*)(uintptr_t)orot, (const uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)signs, n * QW, over);
    const uint64_t woa = at[AI_DEEPSEEK_4__ATT__WOA], woa_l = at[AI_DEEPSEEK_4__ATT__WOA + 1u], row_bytes = room[AI_DEEPSEEK_4__ATT__WOA] / ON;
    for (uint64_t j0 = 0u; j0 < G; j0 += NN__EXPERT__GROUPS_MAX) {
        nn__expert__groups o = {};
        for (uint64_t j = j0; j < G && j - j0 < NN__EXPERT__GROUPS_MAX; ++j, ++o.count)
            ai_deepseek_4__rows__zzprivate_group(&o, (unsigned)(j - j0), woa + j * OR * row_bytes, woa_l + 2u * j * OR, OR, d[3], 0u,
                                                 (3u + j) * n, n);
        doors->expert_groups((uint16_t*)(uintptr_t)og, o, (const uint16_t*)(uintptr_t)orot, (const uint32_t*)(uintptr_t)pairs, GW, over);
    }
    doors->hadamard_rotate((uint16_t*)(uintptr_t)ogr, (const uint16_t*)(uintptr_t)og, (const uint16_t*)(uintptr_t)signs, n * ON, over);
    ai_deepseek_4__rows__zzprivate_gemm(doors, over, at[AI_DEEPSEEK_4__ATT__WOB], room[AI_DEEPSEEK_4__ATT__WOB], at[AI_DEEPSEEK_4__ATT__WOB + 1u],
                                        room[AI_DEEPSEEK_4__ATT__WOB + 1u], y, ogr, pairs, n, d[4], H, ON);
    const unsigned mask_first = ai_deepseek_4__attention__zzprivate_mask_first(argv[1].args[0]);
    for (uint64_t t = 0u; t < n; ++t) {
        const uint64_t mf = ai_deepseek_4__table__zzpackage_masked(doors, over, argv[1].args[0], mask_first, y + 2u * t * H, ogr + 2u * t * ON, H,
                                                                   ON, AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
        if (mf != 0u) return sys__engine__abi__error(mf);
    }
    uint64_t out = y;
    const uint64_t f = ai_deepseek_4__rows__zzprivate_sum(ctx, doors, y, n, H, argc == 9u ? tp : 0, &out);
    if (f != 0u) return sys__engine__abi__error(f);
    for (uint64_t t = 0u; t < n; ++t)
        doors->hyper_post_fp32((float*)(uintptr_t)(streams + t * SB), (const float*)(uintptr_t)(streams + t * SB),
                               (const uint16_t*)(uintptr_t)(out + 2u * t * H), (const float*)(uintptr_t)(amixes + t * mix), H, M);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_deepseek_4__route_rows streams table hands n amixes bmixes xrows work)` -> `hands`. For each of `n` rows: the
 * site's mix written into its record of `bmixes`, the row's streams collapsed by its attention's (its record of `amixes`)
 * and normed; every row's input rotated at once into `xrows` (a row of `hidden` halves each, what `close_rows`' shared
 * expert reads) and into its hand; the router's `k` picks after the weights in the hand (AI_DEEPSEEK_4__HAND_PICKS) —
 * on the card, where the CPU reads them. */
static sys__heap_node ai_deepseek_4__route_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 8u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, hands = 0, h_room = 0, amixes = 0, a_room = 0, bmixes = 0, b_room = 0, xrows = 0, x_room = 0;
    uint64_t work = 0, w_room = 0;
    if (!ai_deepseek_4__layer__zzprivate_open(argv, &streams, &s_room) || !nn__primitives__room(&argv[2], &hands, &h_room)
     || argv[3].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[4], &amixes, &a_room)
     || !nn__primitives__room(&argv[5], &bmixes, &b_room) || !nn__primitives__room(&argv[6], &xrows, &x_room)
     || !nn__primitives__room(&argv[7], &work, &w_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0];
    uint64_t at[AI_DEEPSEEK_4__MOE__TABLE], room[AI_DEEPSEEK_4__MOE__TABLE], v[AI_DEEPSEEK_4__MOE__TABLE];
    float r[AI_DEEPSEEK_4__MOE__TABLE];
    if (!ai_deepseek_4__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__MOE__HIDDEN], M = v[AI_DEEPSEEK_4__MOE__MULT], E = v[AI_DEEPSEEK_4__MOE__EXPERTS];
    const uint64_t K = v[AI_DEEPSEEK_4__MOE__TOP_K], hand_bytes = AI_DEEPSEEK_4__HAND(H, K), mix = AI_DEEPSEEK_4__MIX_BYTES;
    if (n == 0u || n > AI_DEEPSEEK_4__ROWS_MAX || H == 0u || M == 0u || M > NN__HYPER__MULT_MAX || s_room / (4ull * M * H) < n
     || h_room / hand_bytes < n || a_room / mix < n || b_room / mix < n || !nn__primitives__fits(n * H, x_room)
     || !nn__primitives__fits(n * H, w_room)
     || !nn__primitives__fits(E, room[AI_DEEPSEEK_4__MOE__BIAS]) || !nn__primitives__fits(E * H, room[AI_DEEPSEEK_4__MOE__ROUTER]))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    /* scratch: the shared expert's answer (a position's) · xin (floats) · h · the router's logits; `work` the rows' `h` */
    const uint64_t scratch = at[AI_DEEPSEEK_4__MOE__SCRATCH], xin = scratch + 2u * H, logits = xin + 6u * H;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    for (uint64_t t = 0u; t < n; ++t)
        ai_deepseek_4__layer__zzprivate_collapse(doors, over, at, bmixes + t * mix, amixes + t * mix, streams + t * 4u * M * H, xin,
                                                 work + 2u * t * H, H, M, v[AI_DEEPSEEK_4__MOE__ITERS], r[AI_DEEPSEEK_4__MOE__EPS],
                                                 r[AI_DEEPSEEK_4__MOE__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)xrows, (const uint16_t*)(uintptr_t)work, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__MOE__SIGNS],
                           n * H, over);
    for (uint64_t t = 0u; t < n; ++t) {
        const uint64_t hand = hands + t * hand_bytes;
        doors->vector_copy((uint16_t*)(uintptr_t)hand, (const uint16_t*)(uintptr_t)(xrows + 2u * t * H), H);
        doors->expert_multiply_fp16((uint16_t*)(uintptr_t)logits, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__MOE__ROUTER],
                                    (const uint16_t*)(uintptr_t)(work + 2u * t * H), E, H, over);
        doors->vector_top_k_scored((uint64_t*)(uintptr_t)(hand + AI_DEEPSEEK_4__HAND_PICKS(H, K)), 1ull, (uint16_t*)(uintptr_t)(hand + 2u * H),
                                   (const uint16_t*)(uintptr_t)logits, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__MOE__BIAS], E, K,
                                   r[AI_DEEPSEEK_4__MOE__SCALE], NN__VECTOR__SCORE_SQRT_SOFTPLUS);
    }
    return nn__doors_answer(&argv[2]);
}

/* `(ai_deepseek_4__close_rows streams table xrows routed n bmixes work [group sums])` -> `streams`. The MoE site's end over
 * `n` rows: the shared expert on every row's rotated input (`xrows`, as `route_rows` left it) — gate and up in one grouped
 * launch, the clamped swiglu, rotated, down — each row's routed sum (a row of `routed`) added, and its streams mixed again
 * by its record of `bmixes` — under a group, this card's part of the shared expert and its share of the routed (the CPU's
 * sum on card 0, zeros on the others), every card's rows summed first. */
static sys__heap_node ai_deepseek_4__close_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u && argc != 9u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, xrows = 0, x_room = 0, routed = 0, r_room = 0, bmixes = 0, b_room = 0, work = 0, w_room = 0;
    if (!ai_deepseek_4__layer__zzprivate_open(argv, &streams, &s_room) || !nn__primitives__room(&argv[2], &xrows, &x_room)
     || !nn__primitives__room(&argv[3], &routed, &r_room) || argv[4].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[5], &bmixes, &b_room) || !nn__primitives__room(&argv[6], &work, &w_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[4].args[0];
    uint64_t at[AI_DEEPSEEK_4__MOE__TABLE], room[AI_DEEPSEEK_4__MOE__TABLE], v[AI_DEEPSEEK_4__MOE__TABLE];
    float r[AI_DEEPSEEK_4__MOE__TABLE];
    if (!ai_deepseek_4__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__MOE__HIDDEN], M = v[AI_DEEPSEEK_4__MOE__MULT], I = v[AI_DEEPSEEK_4__MOE__INTER];
    const uint64_t* d = &v[AI_DEEPSEEK_4__MOE__D];
    const uint64_t mix = AI_DEEPSEEK_4__MIX_BYTES, SB = 4u * M * H;
    if (n == 0u || n > AI_DEEPSEEK_4__ROWS_MAX || M == 0u || M > NN__HYPER__MULT_MAX || s_room / SB < n || b_room / mix < n
     || !nn__primitives__fits(n * H, x_room) || !nn__primitives__fits(n * H, r_room))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    uint64_t next = work;
    const uint64_t end = work + w_room;
    const uint64_t pairs = ai_deepseek_4__rows__zzprivate_take(&next, end, AI_DEEPSEEK_4__ROWS__PAIRS(n, 0u));
    const uint64_t gu = ai_deepseek_4__rows__zzprivate_take(&next, end, 4u * n * I);
    const uint64_t act = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * I);
    const uint64_t actr = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * I);
    const uint64_t sh = ai_deepseek_4__rows__zzprivate_take(&next, end, 2u * n * H);
    if (pairs == 0u || gu == 0u || act == 0u || actr == 0u || sh == 0u) return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_deepseek_4__rows__zzprivate_pairs(ctx, pairs, n, 0u)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    nn__expert__groups g = {};
    ai_deepseek_4__rows__zzprivate_group(&g, 0u, at[AI_DEEPSEEK_4__MOE__GATE], at[AI_DEEPSEEK_4__MOE__GATE + 1u], I, d[0], 0u, n, n);
    ai_deepseek_4__rows__zzprivate_group(&g, 1u, at[AI_DEEPSEEK_4__MOE__UP], at[AI_DEEPSEEK_4__MOE__UP + 1u], I, d[1], 0u, 2u * n, n);
    g.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)gu, g, (const uint16_t*)(uintptr_t)xrows, (const uint32_t*)(uintptr_t)pairs, H, over);
    doors->swiglu_clamped((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, n, I, r[AI_DEEPSEEK_4__MOE__LIMIT], over);
    doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__MOE__SIGNS],
                                I, n, over);
    ai_deepseek_4__rows__zzprivate_gemm(doors, over, at[AI_DEEPSEEK_4__MOE__DOWN], room[AI_DEEPSEEK_4__MOE__DOWN], at[AI_DEEPSEEK_4__MOE__DOWN + 1u],
                                        room[AI_DEEPSEEK_4__MOE__DOWN + 1u], sh, actr, pairs, n, d[2], H, I);
    for (uint64_t t = 0u; t < n; ++t) {
        const uint64_t y = sh + 2u * t * H;
        const uint64_t mf = ai_deepseek_4__table__zzpackage_masked(doors, over, argv[1].args[0], AI_DEEPSEEK_4__MOE__MASK, y, actr + 2u * t * I, H,
                                                                   I, AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
        if (mf != 0u) return sys__engine__abi__error(mf);
        doors->vector_add((uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)(routed + 2u * t * H), H, over);
    }
    uint64_t tp[3] = {0u, 0u, 0u};
    if (argc == 9u && !ai_deepseek_4__layer__zzprivate_tp(argv, 7u, 1u, tp)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t out = sh;
    const uint64_t f = ai_deepseek_4__rows__zzprivate_sum(ctx, doors, sh, n, H, argc == 9u ? tp : 0, &out);
    if (f != 0u) return sys__engine__abi__error(f);
    for (uint64_t t = 0u; t < n; ++t)
        doors->hyper_post_fp32((float*)(uintptr_t)(streams + t * SB), (const float*)(uintptr_t)(streams + t * SB),
                               (const uint16_t*)(uintptr_t)(out + 2u * t * H), (const float*)(uintptr_t)(bmixes + t * mix), H, M);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_deepseek_4__experts_rows hands table routed n)` -> `routed`: for each of `n` hand rows, Σ w_j · down_j(swiglu(
 * gate_up_j(x))) over its picks this worker holds, a row of `routed` each — as nn runs a chunk's routed experts (▶ nn's
 * `nn__routed__rows`): expert-major, each pick's down in a sum of its own and the rows added in pick order. */
static sys__heap_node ai_deepseek_4__experts_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hands = 0, h_room = 0, routed = 0, r_room = 0;
    if (!nn__primitives__room(&argv[0], &hands, &h_room) || !nn__primitives__room(&argv[2], &routed, &r_room)
     || !sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0]) || argv[3].dtype != SYS__KIND__VALUE_INT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0];
    uint64_t at[AI_DEEPSEEK_4__EXP__TABLE], room[AI_DEEPSEEK_4__EXP__TABLE], v[AI_DEEPSEEK_4__EXP__TABLE];
    float r[AI_DEEPSEEK_4__EXP__TABLE];
    if (!ai_deepseek_4__table__zzpackage_read(argv[1].args[0], AI_DEEPSEEK_4__EXP__PLANES, AI_DEEPSEEK_4__EXP__LIMIT,
                                              AI_DEEPSEEK_4__EXP__TABLE, at, room, v, r))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__EXP__HIDDEN], K = v[AI_DEEPSEEK_4__EXP__TOP_K];
    if (n == 0u || n > AI_DEEPSEEK_4__ROWS_MAX || K > AI_DEEPSEEK_4__TOP_K_MAX || v[AI_DEEPSEEK_4__EXP__INT8] > 1u
     || h_room / AI_DEEPSEEK_4__HAND(H, K) < n || !nn__primitives__fits(n * H, r_room))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    nn__routed rt;
    ai_deepseek_4__experts__zzprivate_routed(at, room, v, r, &rt);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t f = nn__routed__rows(ctx, doors, &rt, hands, n, 0, routed, 0, 0);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[2]);
}

/* `(ai_deepseek_4__engram streams table)` -> `streams`. The looked-up rows rotated, through their map into a key a stream
 * and a value, each stream gated by its key and given the value. */
static sys__heap_node ai_deepseek_4__engram__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0;
    if (!ai_deepseek_4__layer__zzprivate_open(argv, &streams, &s_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_DEEPSEEK_4__ENG__TABLE], room[AI_DEEPSEEK_4__ENG__TABLE], v[AI_DEEPSEEK_4__ENG__TABLE];
    float r[AI_DEEPSEEK_4__ENG__TABLE];
    if (!ai_deepseek_4__table__zzpackage_read(argv[1].args[0], AI_DEEPSEEK_4__ENG__PLANES, AI_DEEPSEEK_4__ENG__EPS,
                                              AI_DEEPSEEK_4__ENG__TABLE, at, room, v, r))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__ENG__HIDDEN], M = v[AI_DEEPSEEK_4__ENG__MULT], IN = v[AI_DEEPSEEK_4__ENG__IN];
    if (H == 0u || H > (1ull << 24) || M == 0u || M > NN__HYPER__MULT_MAX || IN == 0u || IN > (1ull << 24)
     || 4ull * M * H > s_room || !nn__primitives__fits(IN, room[AI_DEEPSEEK_4__ENG__ROWS])
     || !nn__primitives__fits(M * H, room[AI_DEEPSEEK_4__ENG__W]) || room[AI_DEEPSEEK_4__ENG__SCRATCH] < 2u * (IN + (M + 1u) * H))
        return sys__engine__abi__error(AI_DEEPSEEK_4__LAYER__FAULT_TABLE);
    const uint64_t rr = at[AI_DEEPSEEK_4__ENG__SCRATCH], kv = rr + 2u * IN;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    doors->hadamard_rotate((uint16_t*)(uintptr_t)rr, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ENG__ROWS],
                           (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ENG__SIGNS], IN, over);
    ai_deepseek_4__layer__zzprivate_gemv(doors, over, at[AI_DEEPSEEK_4__ENG__WKV], room[AI_DEEPSEEK_4__ENG__WKV], at[AI_DEEPSEEK_4__ENG__WKV + 1u],
                                         room[AI_DEEPSEEK_4__ENG__WKV + 1u], kv, rr, v[AI_DEEPSEEK_4__ENG__D], (M + 1u) * H, IN);
    doors->engram_gate_fp32((float*)(uintptr_t)streams, (const uint16_t*)(uintptr_t)kv, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ENG__W], H,
                            M, r[AI_DEEPSEEK_4__ENG__EPS]);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_deepseek_4__attention streams table pos [group sum])` -> `streams`. With a tensor-parallel group, the table is this
 * card's heads and groups, and the group's sum of their outputs lands in `sum` before the streams are mixed with it. */
static sys__heap_node ai_deepseek_4__attention__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[2].dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t tp[3] = {0u, 0u, 0u};
    if (argc == 5u && !ai_deepseek_4__layer__zzprivate_tp(argv, 3u, 1u, tp)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t f = ai_deepseek_4__attention__zzprivate_row(argv, ctx, argv[2].args[0], argc == 5u ? tp : 0);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[0]);
}

SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__attention__zzabi_adapter, ai_deepseek_4__attention__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__route__zzabi_adapter,     ai_deepseek_4__route__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__shared__zzabi_adapter,    ai_deepseek_4__shared__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__close__zzabi_adapter,     ai_deepseek_4__close__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__experts__zzabi_adapter,   ai_deepseek_4__experts__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__engram__zzabi_adapter,    ai_deepseek_4__engram__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__route_rows__zzabi_adapter,   ai_deepseek_4__route_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__experts_rows__zzabi_adapter, ai_deepseek_4__experts_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__attention_rows__zzabi_adapter, ai_deepseek_4__attention_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__close_rows__zzabi_adapter,     ai_deepseek_4__close_rows__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_LAYER__ABI_CUH */
