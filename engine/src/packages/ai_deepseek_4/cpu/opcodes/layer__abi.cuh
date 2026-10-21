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
#include "../table.cuh"
#include "../../../nn/cpu/routed__header.cuh"   /* the routed experts, as nn runs them */


/* The operands every site verb starts with: the streams, and its table read. */
static inline bool ai_deepseek_4__layer__zzprivate_open(const sys__heap_node* argv, uint64_t* streams, uint64_t* s_room) {
    return sys__heap_node__carries_reference(argv[1].dtype) && sys__node_array__is(argv[1].args[0])
        && nn__primitives__room(&argv[0], streams, s_room);
}

/* The site's input: its own mix of the streams written into OWN, the streams collapsed by the EARLIER site's `pre` into
 * `xin` (floats, as large as the streams), then normed into `h` (halves). The mix's own collapse lands in `xin` first and
 * is written over. */
static inline void ai_deepseek_4__layer__zzprivate_collapse(const nn__doors* doors, unsigned int* over, const uint64_t* at,
                                                            uint64_t streams, uint64_t xin, uint64_t h, uint64_t H, uint64_t M,
                                                            uint64_t iters, float eps, float hc_eps) {
    const uint64_t own = at[AI_DEEPSEEK_4__OWN];
    doors->hyper_logits_fp32((float*)(uintptr_t)own + NN__HYPER__MIX(M), (const float*)(uintptr_t)streams,
                             (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__HC_FN], H, M, eps);
    doors->hyper_pre_fp32((float*)(uintptr_t)xin, (float*)(uintptr_t)own, (const float*)(uintptr_t)streams,
                          (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__HC_BASE], (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__HC_SCALE], H, M,
                          iters, hc_eps);
    doors->hyper_collapse_fp32((float*)(uintptr_t)xin, (const float*)(uintptr_t)at[AI_DEEPSEEK_4__EARLIER], (const float*)(uintptr_t)streams,
                               H, M);
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
    uint64_t ratio, source, cache, ring, zero, gather, kv, signs;
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

/* The source's compressor at `pos` on the site's rotated input `hr`: its latent (and, past ratio 1, its gate) into the
 * group's ring row `pos % ratio`; the group pooled once it completes — a softmax of the gates a channel (nn's pool, with
 * no place bias) — normed, its tail turned at the group's first position, and written at the cache's row `pos / ratio`.
 * Ratio 1 is the latent of the position itself, normed and turned. */
static void ai_deepseek_4__attention__zzprivate_compress(const nn__doors* doors, unsigned int* over, const ai_deepseek_4__compressed* c,
                                                         uint64_t hr, uint64_t H, uint64_t HD, uint64_t RD, uint64_t pos,
                                                         uint64_t angles, float eps) {
    const uint64_t r = c->ratio, slot = pos % r, row = c->cache + 2u * (pos / r) * HD, tail = 2u * (HD - RD);
    const uint64_t kv = c->kv, gate = kv + 2u * HD;
    const uint64_t* at = c->at;
    const uint64_t* room = c->room;
    if (r == 1u) {
        ai_deepseek_4__layer__zzprivate_gemv(doors, over, at[AI_DEEPSEEK_4__CMP__WKV], room[AI_DEEPSEEK_4__CMP__WKV], at[AI_DEEPSEEK_4__CMP__WKV + 1u],
                                             room[AI_DEEPSEEK_4__CMP__WKV + 1u], kv, hr, c->d[0], HD, H);
    } else {
        nn__turboquant__groups g = {};
        ai_deepseek_4__table__zzpackage_group(&g, 0u, at, AI_DEEPSEEK_4__CMP__WKV, HD, c->d[0], 0u);
        ai_deepseek_4__table__zzpackage_group(&g, 1u, at, AI_DEEPSEEK_4__CMP__WGATE, HD, c->d[1], HD);
        g.count = 2u;
        doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)kv, g, (const uint16_t*)(uintptr_t)hr, H, over);
        doors->vector_copy((uint16_t*)(uintptr_t)(c->ring + 2u * slot * HD), (const uint16_t*)(uintptr_t)kv, HD);
        doors->vector_copy((uint16_t*)(uintptr_t)(c->ring + 2u * (r + slot) * HD), (const uint16_t*)(uintptr_t)gate, HD);
        if (slot + 1u != r) return;
        doors->index_pool((uint16_t*)(uintptr_t)kv, (const uint16_t*)(uintptr_t)c->ring, (const uint16_t*)(uintptr_t)(c->ring + 2u * r * HD),
                          (const uint16_t*)(uintptr_t)c->zero, 0u, 1u, HD, r, over);
    }
    doors->rmsnorm((uint16_t*)(uintptr_t)row, (const uint16_t*)(uintptr_t)kv, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__CMP__NORM], HD, eps, over);
    const uint64_t first = pos + 1u - r;
    if (c->indexed) {
        /* the latent's index key, from the latent before its rope: through `wk`, normed, its tail turned at the group's
           first position — at row `pos / r` of the source's keys */
        const uint64_t lr = gate + 2u * HD, ik = lr + 2u * HD, key = at[AI_DEEPSEEK_4__IDX__KEYS] + 2u * (pos / r) * c->dim;
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

/* `(ai_deepseek_4__attention streams table pos)` -> `streams`. The attention site at position `pos`: q through its rank, the
 * latent normed into the ring's row `pos % window`, the tails of both turned at `pos`, every head attending the ring's
 * filled rows with its sink, the answer's tails turned back, each group of heads through its own low-rank map, the
 * groups through the output's second map. */
static uint64_t ai_deepseek_4__attention__zzprivate_row(const sys__heap_node* argv, sys__engine__ctx* ctx, uint64_t pos,
                                                       const uint64_t* tp) {
    uint64_t streams = 0, s_room = 0;
    if (!ai_deepseek_4__layer__zzprivate_open(argv, &streams, &s_room)) return SYS__OPCODES__FAULT_TYPE;
    uint64_t at[AI_DEEPSEEK_4__ATT__TABLE], room[AI_DEEPSEEK_4__ATT__TABLE], v[AI_DEEPSEEK_4__ATT__TABLE];
    float r[AI_DEEPSEEK_4__ATT__TABLE];
    if (!ai_deepseek_4__table__zzpackage_read(argv[1].args[0], AI_DEEPSEEK_4__ATT__PLANES, AI_DEEPSEEK_4__ATT__EPS,
                                              AI_DEEPSEEK_4__ATT__TABLE, at, room, v, r))
        return AI_DEEPSEEK_4__LAYER__FAULT_TABLE;
    const uint64_t H = v[AI_DEEPSEEK_4__ATT__HIDDEN], M = v[AI_DEEPSEEK_4__ATT__MULT], heads = v[AI_DEEPSEEK_4__ATT__HEADS];
    const uint64_t HD = v[AI_DEEPSEEK_4__ATT__HEAD_DIM], RD = v[AI_DEEPSEEK_4__ATT__ROPE], QR = v[AI_DEEPSEEK_4__ATT__Q_RANK];
    const uint64_t OR = v[AI_DEEPSEEK_4__ATT__O_RANK], G = v[AI_DEEPSEEK_4__ATT__GROUPS], W = v[AI_DEEPSEEK_4__ATT__WINDOW];
    const uint64_t* d = &v[AI_DEEPSEEK_4__ATT__D];            /* wq_a wkv wq_b wo_a wo_b */
    if (!ai_deepseek_4__layer__zzprivate_mixes(room, s_room, H, M) || heads == 0u || heads > (1ull << 12) || HD == 0u || HD > (1ull << 16)
     || RD == 0u || RD % 2u != 0u || RD > HD || QR == 0u || QR > (1ull << 20) || OR == 0u || G == 0u || (heads * HD) % G != 0u
     || W == 0u || W > (1ull << 20) || pos >= (1ull << 32))
        return AI_DEEPSEEK_4__LAYER__FAULT_TABLE;
    const uint64_t QW = heads * HD, GW = QW / G, ON = G * OR;           /* the query's width, a group's, the groups' answer */
    const uint64_t len = pos + 1u < W ? pos + 1u : W;
    if (!nn__primitives__fits(W * HD, room[AI_DEEPSEEK_4__ATT__RING]) || !nn__primitives__fits(2u * (pos + 1u) * RD, room[AI_DEEPSEEK_4__ATT__ANGLES])
     || !nn__primitives__fits(heads, room[AI_DEEPSEEK_4__ATT__SINK])
     || room[AI_DEEPSEEK_4__ATT__WOA] % ON != 0u || room[AI_DEEPSEEK_4__ATT__WOA + 1u] < 2u * ON
     || (tp != 0 && !nn__primitives__fits(H, tp[2])))
        return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    ai_deepseek_4__compressed cmp;
    const uint64_t cf = ai_deepseek_4__attention__zzprivate_compressed(argv[1].args[0], at, room, HD, W, pos, &cmp);
    if (cf != 0u) return cf;
    const uint64_t latents_all = cmp.ratio != 0u ? (pos + 1u) / cmp.ratio : 0u;
    if (room[AI_DEEPSEEK_4__ATT__SCORES] / 4u < heads * (len + (latents_all < cmp.topk || cmp.ratio == 0u ? latents_all : cmp.topk)))
        return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    /* scratch: xin, floats; then halves: h · hr · [qa kv] · qn · qnr · q · o · or · og · ogr · y · [the compressor's kv,
       gate, rotated latent, index key] · [the indexer's query and weights] */
    const uint64_t need = 2u * (5u * H + 3u * QR + HD + 3u * QW + 2u * ON + 4u * HD + cmp.dim + cmp.heads * cmp.dim + cmp.heads);
    if (room[AI_DEEPSEEK_4__ATT__SCRATCH] < need) return AI_DEEPSEEK_4__LAYER__FAULT_ROOM;
    const uint64_t xin = at[AI_DEEPSEEK_4__ATT__SCRATCH], h = xin + 4u * H, hr = h + 2u * H, qa = hr + 2u * H, kv = qa + 2u * QR;
    const uint64_t qn = kv + 2u * HD, qnr = qn + 2u * QR, q = qnr + 2u * QR, o = q + 2u * QW, orot = o + 2u * QW;
    const uint64_t og = orot + 2u * QW, ogr = og + 2u * ON, y = ogr + 2u * ON;
    const uint64_t signs = at[AI_DEEPSEEK_4__ATT__SIGNS], ring = at[AI_DEEPSEEK_4__ATT__RING], scores = at[AI_DEEPSEEK_4__ATT__SCORES];
    cmp.kv = y + 2u * H;
    cmp.signs = signs;
    const uint64_t row = ring + 2u * (pos % W) * HD, tail = 2u * (HD - RD), cs = at[AI_DEEPSEEK_4__ATT__ANGLES] + 4u * pos * RD;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    unsigned int* over = ctx->fault_word;

    ai_deepseek_4__layer__zzprivate_collapse(doors, over, at, streams, xin, h, H, M, v[AI_DEEPSEEK_4__ATT__ITERS],
                                             r[AI_DEEPSEEK_4__ATT__EPS], r[AI_DEEPSEEK_4__ATT__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)hr, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, H, over);
    nn__turboquant__groups g = {};
    ai_deepseek_4__table__zzpackage_group(&g, 0u, at, AI_DEEPSEEK_4__ATT__WQA, QR, d[0], 0u);
    ai_deepseek_4__table__zzpackage_group(&g, 1u, at, AI_DEEPSEEK_4__ATT__WKV, HD, d[1], QR);
    g.count = 2u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)qa, g, (const uint16_t*)(uintptr_t)hr, H, over);
    doors->rmsnorm((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qa, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ATT__Q_NORM], QR,
                   r[AI_DEEPSEEK_4__ATT__EPS], over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)qnr, (const uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)signs, QR, over);
    ai_deepseek_4__layer__zzprivate_gemv(doors, over, at[AI_DEEPSEEK_4__ATT__WQB], room[AI_DEEPSEEK_4__ATT__WQB], at[AI_DEEPSEEK_4__ATT__WQB + 1u],
                                         room[AI_DEEPSEEK_4__ATT__WQB + 1u], q, qnr, d[2], QW, QR);
    /* the position's row of the ring: the latent normed, its tail turned — the key and the value of every head */
    doors->rmsnorm((uint16_t*)(uintptr_t)row, (const uint16_t*)(uintptr_t)kv, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ATT__KV_NORM], HD,
                   r[AI_DEEPSEEK_4__ATT__EPS], over);
    doors->rope_interleaved((uint16_t*)(uintptr_t)(row + tail), (const uint16_t*)(uintptr_t)(row + tail), (const float*)(uintptr_t)cs, 1u, HD,
                            RD, 0u, over);
    doors->rope_interleaved((uint16_t*)(uintptr_t)(q + tail), (const uint16_t*)(uintptr_t)(q + tail), (const float*)(uintptr_t)cs, heads, HD,
                            RD, 0u, over);
    /* the compressed latents: the source writes its group's when it completes; every compressing layer attends them */
    uint64_t rows = ring, attend = len;
    if (cmp.ratio != 0u) {
        if (cmp.source)
            ai_deepseek_4__attention__zzprivate_compress(doors, over, &cmp, hr, H, HD, RD, pos, at[AI_DEEPSEEK_4__ATT__ANGLES],
                                                         r[AI_DEEPSEEK_4__ATT__EPS]);
        const uint64_t latents = (pos + 1u) / cmp.ratio;
        doors->vector_copy((uint16_t*)(uintptr_t)cmp.gather, (const uint16_t*)(uintptr_t)ring, len * HD);
        if (latents <= cmp.topk) {
            if (latents != 0u)
                doors->vector_copy((uint16_t*)(uintptr_t)(cmp.gather + 2u * len * HD), (const uint16_t*)(uintptr_t)cmp.cache, latents * HD);
            attend = len + latents;
        } else {
            /* ⭐ THE INDEXER: an index source scores every latent and keeps the best `topk`; the layers after it reuse them */
            const uint64_t sel = cmp.at[AI_DEEPSEEK_4__IDX__SELECT];
            if (cmp.indexer) {
                const uint64_t IH = cmp.heads, ID = cmp.dim, iq = cmp.kv + 2u * (4u * HD + ID), iw = iq + 2u * IH * ID;
                ai_deepseek_4__layer__zzprivate_gemv(doors, over, cmp.at[AI_DEEPSEEK_4__IDX__WQB], cmp.room[AI_DEEPSEEK_4__IDX__WQB],
                                                     cmp.at[AI_DEEPSEEK_4__IDX__WQB + 1u], cmp.room[AI_DEEPSEEK_4__IDX__WQB + 1u], iq, qnr,
                                                     cmp.di[0], IH * ID, QR);
                doors->rope_interleaved((uint16_t*)(uintptr_t)(iq + 2u * (ID - RD)), (const uint16_t*)(uintptr_t)(iq + 2u * (ID - RD)),
                                        (const float*)(uintptr_t)cs, IH, ID, RD, 0u, over);
                ai_deepseek_4__layer__zzprivate_gemv(doors, over, cmp.at[AI_DEEPSEEK_4__IDX__WP], cmp.room[AI_DEEPSEEK_4__IDX__WP],
                                                     cmp.at[AI_DEEPSEEK_4__IDX__WP + 1u], cmp.room[AI_DEEPSEEK_4__IDX__WP + 1u], iw, hr,
                                                     cmp.di[1], IH, H);
                doors->index_scores((float*)(uintptr_t)cmp.at[AI_DEEPSEEK_4__IDX__SCORES], (const uint16_t*)(uintptr_t)iq,
                                    (const uint16_t*)(uintptr_t)iw, (const uint16_t*)(uintptr_t)cmp.at[AI_DEEPSEEK_4__IDX__KEYS], IH, ID, latents,
                                    1.0f / sqrtf((float)ID));
                doors->index_select((uint32_t*)(uintptr_t)sel, (uint32_t*)(uintptr_t)(sel + 4u * cmp.topk),
                                    (const float*)(uintptr_t)cmp.at[AI_DEEPSEEK_4__IDX__SCORES], latents, cmp.topk, 1u, latents, 0u);
            }
            doors->index_gather((uint16_t*)(uintptr_t)(cmp.gather + 2u * len * HD), (const uint16_t*)(uintptr_t)cmp.cache,
                                (const uint32_t*)(uintptr_t)sel, cmp.topk, HD);
            attend = len + cmp.topk;
        }
        rows = cmp.gather;
    }
    /* every head over its rows, one kv head: the scores at 1/√head_dim, the sink, the rows mixed */
    doors->attention_scores_grouped((float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)rows, heads, 1u, HD, attend);
    doors->attention_softmax_sink((float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__ATT__SINK], heads, attend);
    doors->attention_mix((uint16_t*)(uintptr_t)o, (const float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)rows, heads, 1u, HD, attend, over);
    doors->rope_interleaved((uint16_t*)(uintptr_t)(o + tail), (const uint16_t*)(uintptr_t)(o + tail), (const float*)(uintptr_t)cs, heads, HD,
                            RD, 1u, over);
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
    ai_deepseek_4__layer__zzprivate_collapse(doors, over, at, streams, xin, h, H, M, v[AI_DEEPSEEK_4__MOE__ITERS],
                                             r[AI_DEEPSEEK_4__MOE__EPS], r[AI_DEEPSEEK_4__MOE__HC_EPS]);
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

/* `(ai_deepseek_4__experts hand picks table routed)` -> `routed`: Σ w_j · down_j(swiglu(gate_up_j(x))) over the picks this
 * worker holds, the SwiGLU clamped — as nn runs a position's routed experts (▶ nn's `nn__routed__position`). */
static sys__heap_node ai_deepseek_4__experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
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
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t f = nn__routed__position(ctx, doors, &rt, hand, argv[1].args[0], 0, routed, 0);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[3]);
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

#endif /* SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_LAYER__ABI_CUH */
