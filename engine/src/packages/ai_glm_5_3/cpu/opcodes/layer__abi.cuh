#ifndef SILVANN__PACKAGES_AI_GLM_5_3_CPU_OPCODES_LAYER__ABI_CUH
#define SILVANN__PACKAGES_AI_GLM_5_3_CPU_OPCODES_LAYER__ABI_CUH
/* ══ GLM 5.3 FLASH'S LAYER, A VERB A DEVICE SEAM ═════════════════════════════════════════════════════════════
 * The four residual streams go in and come out in place. A site's verb collapses them into its input
 * (`nn__hyper__pre`), norms it, runs the sublayer and mixes the streams again with its output (`nn__hyper__post`) —
 * the attention site whole in one verb, the MLP site whole when it is dense, and cut in three where the routed
 * experts run on another worker. Every matrix is a TurboQuant record rotated along its input, so each is handed its
 * input rotated; the router is lossless and reads it as it is. ▶ `contracts/objects/layer.cuh` for the tables. */

#include <math.h>
#include <stdlib.h>                       /* a chunk's picks, inverted on the host: malloc, free */
#include "../table.cuh"
#include "../../../nn/cpu/routed__header.cuh"   /* the routed experts, as nn runs them */


/* The operands every site verb starts with: the streams, and its table read. */
static inline bool ai_glm_5_3__layer__zzprivate_open(const sys__heap_node* argv, uint64_t* streams, uint64_t* s_room) {
    return sys__heap_node__carries_reference(argv[1].dtype) && sys__node_array__is(argv[1].args[0])
        && nn__primitives__room(&argv[0], streams, s_room);
}

/* Row `row` of a buffer of rows `width` halves wide: the address moved to it, the room what is left from there. */
static inline bool ai_glm_5_3__layer__zzprivate_row(uint64_t* streams, uint64_t* s_room, uint64_t row, uint64_t width) {
    const uint64_t skip = 2u * width * row;
    if (row >= AI_GLM_5_3__ROWS_MAX || width > (1ull << 32) || *s_room < skip) return false;
    *streams += skip;
    *s_room -= skip;
    return true;
}

/* The site's input: the streams collapsed by the site's map (its weights into the scratch's carry), then normed. */
static inline void ai_glm_5_3__layer__zzprivate_collapse(const nn__doors* doors, unsigned int* over, const uint64_t* at,
                                                         uint64_t scratch, uint64_t streams, uint64_t xin, uint64_t h,
                                                         uint64_t H, uint64_t M, uint64_t iters, float eps, float hc_eps) {
    doors->hyper_logits((float*)(uintptr_t)scratch + NN__HYPER__MIX(M), (const uint16_t*)(uintptr_t)(streams), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__HC_FN]),
                        H, M, eps);
    doors->hyper_pre((uint16_t*)(uintptr_t)(xin), (float*)(uintptr_t)scratch, (const uint16_t*)(uintptr_t)(streams), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__HC_BASE]),
                     (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__HC_SCALE]), H, M, iters, hc_eps, over);
    doors->rmsnorm((uint16_t*)(uintptr_t)(h), (const uint16_t*)(uintptr_t)(xin), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__NORM]), H, eps, over);
}

/* The streams mixed again with the sublayer's output `y`, by the weights the collapse left in the carry. */
static inline void ai_glm_5_3__layer__zzprivate_place(const nn__doors* doors, unsigned int* over, uint64_t scratch,
                                                      uint64_t streams, uint64_t y, uint64_t H, uint64_t M) {
    doors->hyper_post((uint16_t*)(uintptr_t)(streams), (const uint16_t*)(uintptr_t)(streams), (const uint16_t*)(uintptr_t)(y), (const float*)(uintptr_t)scratch, H, M, over);
}

static inline void ai_glm_5_3__layer__zzprivate_gemv(const nn__doors* doors, unsigned int* over, const uint64_t* at,
                                                     const uint64_t* room, unsigned plane, uint64_t out, uint64_t x,
                                                     uint64_t d, uint64_t rows, uint64_t cols) {
    doors->turboquant_gemv((uint16_t*)(uintptr_t)(out), (const uint8_t*)(uintptr_t)(at[plane]), room[plane], (const uint8_t*)(uintptr_t)(at[plane + 1u]), room[plane + 1u],
                           (const uint16_t*)(uintptr_t)(x), d, rows, cols, over);
}

/* `(ai_glm_5_3__kda streams table)` -> `streams`. The attention site of a KDA layer: q, k, v, the forget gate's and the
 * output gate's first projections and β's logits in one grouped launch, the three short convolutions, the gates' second
 * projections on their rotated ranks, the step, the output projection. */
static uint64_t ai_glm_5_3__kda__zzprivate_row(const sys__heap_node* argv, sys__engine__ctx* ctx, uint64_t row) {
    uint64_t streams = 0, s_room = 0;
    if (!ai_glm_5_3__layer__zzprivate_open(argv, &streams, &s_room)) return SYS__OPCODES__FAULT_TYPE;
    uint64_t at[AI_GLM_5_3__KDA__TABLE], room[AI_GLM_5_3__KDA__TABLE], v[AI_GLM_5_3__KDA__TABLE];
    float r[AI_GLM_5_3__KDA__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[1].args[0], AI_GLM_5_3__KDA__PLANES, AI_GLM_5_3__KDA__EPS, AI_GLM_5_3__KDA__TABLE,
                                           at, room, v, r))
        return AI_GLM_5_3__LAYER__FAULT_TABLE;
    const uint64_t H = v[AI_GLM_5_3__KDA__HIDDEN], M = v[AI_GLM_5_3__KDA__MULT], heads = v[AI_GLM_5_3__KDA__HEADS];
    const uint64_t hd = v[AI_GLM_5_3__KDA__HEAD_DIM], KW = heads * hd;
    const uint64_t* d = &v[AI_GLM_5_3__KDA__D];            /* q k v fa ga b fb gb o */
    if (!ai_glm_5_3__layer__zzprivate_row(&streams, &s_room, row, M * H)) return AI_GLM_5_3__LAYER__FAULT_TABLE;
    if (H == 0u || M == 0u || M > NN__HYPER__MULT_MAX || heads == 0u || hd == 0u || H > (1ull << 24) || KW > (1ull << 24)
     || !nn__primitives__fits(M * H, s_room) || room[AI_GLM_5_3__KDA__STATE] < heads * hd * hd * 4u
     || !nn__primitives__fits(9u * KW, room[AI_GLM_5_3__KDA__WINDOW]))
        return AI_GLM_5_3__LAYER__FAULT_TABLE;
    /* scratch, halves after the carry: xin · h · hr · [q k v fa ga fb b] · conved · far · gar · gate · ko · kor · y */
    const uint64_t proj_n = 4u * KW + 2u * hd + heads;
    const uint64_t need = AI_GLM_5_3__CARRY(H) + 2u * (4u * H + proj_n + 3u * KW + 2u * hd + 3u * KW);
    if (room[AI_GLM_5_3__KDA__SCRATCH] < need) return AI_GLM_5_3__LAYER__FAULT_ROOM;
    const uint64_t scratch = at[AI_GLM_5_3__KDA__SCRATCH], signs = at[AI_GLM_5_3__KDA__SIGNS];
    const uint64_t xin = scratch + AI_GLM_5_3__CARRY(H), h = xin + 2u * H, hr = h + 2u * H, proj = hr + 2u * H;
    const uint64_t fa = proj + 2u * 3u * KW, ga = fa + 2u * hd, fb = ga + 2u * hd, b = fb + 2u * KW;
    const uint64_t conved = proj + 2u * proj_n, far = conved + 2u * 3u * KW, gar = far + 2u * hd, gate = gar + 2u * hd;
    const uint64_t ko = gate + 2u * KW, kor = ko + 2u * KW, y = kor + 2u * KW;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    unsigned int* over = ctx->fault_word;

    ai_glm_5_3__layer__zzprivate_collapse(doors, over, at, scratch, streams, xin, h, H, M, v[AI_GLM_5_3__KDA__ITERS],
                                          r[AI_GLM_5_3__KDA__EPS], r[AI_GLM_5_3__KDA__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(hr), (const uint16_t*)(uintptr_t)(h), (const uint16_t*)(uintptr_t)(signs), H, over);
    nn__turboquant__groups g = {};
    ai_glm_5_3__table__zzpackage_group(&g, 0u, at, AI_GLM_5_3__KDA__Q, KW, d[0], 0u);
    ai_glm_5_3__table__zzpackage_group(&g, 1u, at, AI_GLM_5_3__KDA__K, KW, d[1], KW);
    ai_glm_5_3__table__zzpackage_group(&g, 2u, at, AI_GLM_5_3__KDA__V, KW, d[2], 2u * KW);
    ai_glm_5_3__table__zzpackage_group(&g, 3u, at, AI_GLM_5_3__KDA__FA, hd, d[3], 3u * KW);
    ai_glm_5_3__table__zzpackage_group(&g, 4u, at, AI_GLM_5_3__KDA__GA, hd, d[4], 3u * KW + hd);
    ai_glm_5_3__table__zzpackage_group(&g, 5u, at, AI_GLM_5_3__KDA__B, heads, d[5], 4u * KW + 2u * hd);
    g.count = 6u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)(proj), g, (const uint16_t*)(uintptr_t)(hr), H, over);
    const unsigned conv[3] = {AI_GLM_5_3__KDA__Q_CONV, AI_GLM_5_3__KDA__K_CONV, AI_GLM_5_3__KDA__V_CONV};
    for (unsigned i = 0u; i < 3u; ++i)
        doors->deltanet_conv_step((uint16_t*)(uintptr_t)(conved + 2u * i * KW), (uint16_t*)(uintptr_t)(at[AI_GLM_5_3__KDA__WINDOW] + 2u * 3u * i * KW),
                                  (const uint16_t*)(uintptr_t)(at[conv[i]]), (const uint16_t*)(uintptr_t)(proj + 2u * i * KW), KW, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(far), (const uint16_t*)(uintptr_t)(fa), (const uint16_t*)(uintptr_t)(signs), hd, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(gar), (const uint16_t*)(uintptr_t)(ga), (const uint16_t*)(uintptr_t)(signs), hd, over);
    ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, AI_GLM_5_3__KDA__FB, fb, far, d[6], KW, hd);
    ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, AI_GLM_5_3__KDA__GB, gate, gar, d[7], KW, hd);
    doors->kda_step((float*)(uintptr_t)at[AI_GLM_5_3__KDA__STATE], (const uint16_t*)(uintptr_t)(conved), (const uint16_t*)(uintptr_t)(fb), (const uint16_t*)(uintptr_t)(b),
                    (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__KDA__DT_BIAS]), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__KDA__A_LOG]), (const uint16_t*)(uintptr_t)(gate),
                    (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__KDA__O_NORM]), (uint16_t*)(uintptr_t)(ko), heads, hd, r[AI_GLM_5_3__KDA__LOWER],
                    r[AI_GLM_5_3__KDA__EPS], over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(kor), (const uint16_t*)(uintptr_t)(ko), (const uint16_t*)(uintptr_t)(signs), KW, over);
    ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, AI_GLM_5_3__KDA__O, y, kor, d[8], H, KW);
    ai_glm_5_3__layer__zzprivate_place(doors, over, scratch, streams, y, H, M);
    return 0u;
}

/* `(ai_glm_5_3__mla streams table pos)` -> `streams`. The attention site of an MLA layer at position `pos`: q through its
 * rank, the latent normed and written rotated into the cache's row `pos`, the query absorbed into the latent, every head
 * attending rows `0..pos` of the cache as one kv head, the answer expanded by each head's value rows and projected. */
static uint64_t ai_glm_5_3__mla__zzprivate_row(const sys__heap_node* argv, sys__engine__ctx* ctx, uint64_t pos, uint64_t row) {
    uint64_t streams = 0, s_room = 0;
    if (!ai_glm_5_3__layer__zzprivate_open(argv, &streams, &s_room)) return SYS__OPCODES__FAULT_TYPE;
    uint64_t at[AI_GLM_5_3__IDX__TABLE], room[AI_GLM_5_3__IDX__TABLE], v[AI_GLM_5_3__IDX__TABLE];
    float r[AI_GLM_5_3__IDX__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[1].args[0], AI_GLM_5_3__MLA__PLANES, AI_GLM_5_3__MLA__EPS, AI_GLM_5_3__MLA__TABLE,
                                           at, room, v, r))
        return AI_GLM_5_3__LAYER__FAULT_TABLE;
    const uint64_t H = v[AI_GLM_5_3__MLA__HIDDEN], M = v[AI_GLM_5_3__MLA__MULT], heads = v[AI_GLM_5_3__MLA__HEADS];
    const uint64_t QR = v[AI_GLM_5_3__MLA__Q_RANK], LAT = v[AI_GLM_5_3__MLA__LATENT], NOPE = v[AI_GLM_5_3__MLA__NOPE];
    const uint64_t VD = v[AI_GLM_5_3__MLA__V_DIM], len = pos + 1u, stride = NOPE + VD;
    const uint64_t* d = &v[AI_GLM_5_3__MLA__D];            /* q_a kv_a q_b o */
    if (!ai_glm_5_3__layer__zzprivate_row(&streams, &s_room, row, M * H)) return AI_GLM_5_3__LAYER__FAULT_TABLE;
    if (H == 0u || M == 0u || M > NN__HYPER__MULT_MAX || heads == 0u || QR == 0u || LAT == 0u || NOPE == 0u || VD == 0u || H > (1ull << 24)
     || heads > (1ull << 12) || LAT > (1ull << 16) || pos >= (1ull << 32) || !nn__primitives__fits(M * H, s_room)
     || !nn__primitives__fits(heads * stride * LAT, room[AI_GLM_5_3__MLA__KVB]))
        return AI_GLM_5_3__LAYER__FAULT_TABLE;
    /* the indexer, when the table carries it: its planes, its sizes, and which positions this query attends */
    const uint64_t table = argv[1].args[0];
    const bool indexed = sys__node_array__length(table) >= AI_GLM_5_3__IDX__TABLE;
    uint64_t IH = 0u, ID = 0u, KP = 1u, TOPK = 0u, id_[4] = {0u, 0u, 0u, 0u};
    if (indexed) {
        sys__node_array_walk w;
        if (!sys__node_array__walk(table, AI_GLM_5_3__MLA__TABLE, &w)) return AI_GLM_5_3__LAYER__FAULT_TABLE;
        for (unsigned i = AI_GLM_5_3__MLA__TABLE; i < AI_GLM_5_3__IDX__PLANES; ++i, sys__node_array__next(&w)) {
            const sys__heap_node* n = sys__node_array__walk_cell(&w);
            if (n == 0 || !nn__primitives__room(n, &at[i], &room[i])) return AI_GLM_5_3__LAYER__FAULT_TABLE;
        }
        uint64_t iv[AI_GLM_5_3__IDX__TABLE - AI_GLM_5_3__IDX__PLANES];
        for (unsigned i = AI_GLM_5_3__IDX__PLANES; i < AI_GLM_5_3__IDX__TABLE; ++i, sys__node_array__next(&w)) {
            const sys__heap_node* n = sys__node_array__walk_cell(&w);
            if (n == 0 || n->dtype != SYS__KIND__VALUE_INT) return AI_GLM_5_3__LAYER__FAULT_TABLE;
            iv[i - AI_GLM_5_3__IDX__PLANES] = n->args[0];
        }
        IH = iv[0]; ID = iv[1]; KP = iv[2]; TOPK = iv[3];
        for (unsigned i = 0u; i < 4u; ++i) id_[i] = iv[4u + i];
        if (IH == 0u || ID == 0u || KP == 0u || TOPK < KP || IH > 1024u || ID > 4096u
         || !nn__primitives__fits(KP * ID, room[AI_GLM_5_3__IDX__KEYS]) || !nn__primitives__fits(KP * ID, room[AI_GLM_5_3__IDX__GATES])
         || !nn__primitives__fits((len / KP + 1u) * ID, room[AI_GLM_5_3__IDX__POOLED]) || room[AI_GLM_5_3__IDX__SCORES] / 4u < len / KP + 1u
         || room[AI_GLM_5_3__IDX__INDEX] / 4u < TOPK + KP || !nn__primitives__fits((TOPK + KP) * LAT, room[AI_GLM_5_3__IDX__GATHER])
         || !nn__primitives__fits(KP * ID, room[AI_GLM_5_3__IDX__APE]))
            return AI_GLM_5_3__LAYER__FAULT_ROOM;
    }
    const uint64_t pools = len / KP, chosen_pools = indexed ? TOPK / KP : 0u, tail = len - pools * KP;
    const bool sparse = indexed && pools > chosen_pools;
    const uint64_t alen = sparse ? chosen_pools * KP + tail : len;          /* the positions this query attends */
    if (!nn__primitives__fits(len * LAT, room[AI_GLM_5_3__MLA__CACHE]) || room[AI_GLM_5_3__MLA__SCORES] / 4u < heads * alen)
        return AI_GLM_5_3__LAYER__FAULT_ROOM;
    /* scratch, halves after the carry: xin · h · hr · [qa kv] · qn · qnr · ckn · q · qabs · olat · vh · vhr · y · [iq iw ik] */
    const uint64_t need = AI_GLM_5_3__CARRY(H) + 2u * (4u * H + 3u * QR + 2u * LAT + heads * NOPE + 2u * heads * LAT + 2u * heads * VD
                                                       + IH * ID + IH + ID);
    if (room[AI_GLM_5_3__MLA__SCRATCH] < need) return AI_GLM_5_3__LAYER__FAULT_ROOM;
    const uint64_t scratch = at[AI_GLM_5_3__MLA__SCRATCH], signs = at[AI_GLM_5_3__MLA__SIGNS];
    const uint64_t xin = scratch + AI_GLM_5_3__CARRY(H), h = xin + 2u * H, hr = h + 2u * H, qa = hr + 2u * H, kv = qa + 2u * QR;
    const uint64_t qn = kv + 2u * LAT, qnr = qn + 2u * QR, ckn = qnr + 2u * QR, q = ckn + 2u * LAT;
    const uint64_t qabs = q + 2u * heads * NOPE, olat = qabs + 2u * heads * LAT, vh = olat + 2u * heads * LAT;
    const uint64_t vhr = vh + 2u * heads * VD, y = vhr + 2u * heads * VD;
    const uint64_t cache = at[AI_GLM_5_3__MLA__CACHE], kvb = at[AI_GLM_5_3__MLA__KVB];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    unsigned int* over = ctx->fault_word;

    ai_glm_5_3__layer__zzprivate_collapse(doors, over, at, scratch, streams, xin, h, H, M, v[AI_GLM_5_3__MLA__ITERS],
                                          r[AI_GLM_5_3__MLA__EPS], r[AI_GLM_5_3__MLA__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(hr), (const uint16_t*)(uintptr_t)(h), (const uint16_t*)(uintptr_t)(signs), H, over);
    nn__turboquant__groups g = {};
    ai_glm_5_3__table__zzpackage_group(&g, 0u, at, AI_GLM_5_3__MLA__QA, QR, d[0], 0u);
    ai_glm_5_3__table__zzpackage_group(&g, 1u, at, AI_GLM_5_3__MLA__KVA, LAT, d[1], QR);
    g.count = 2u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)(qa), g, (const uint16_t*)(uintptr_t)(hr), H, over);
    doors->rmsnorm((uint16_t*)(uintptr_t)(qn), (const uint16_t*)(uintptr_t)(qa), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__MLA__QA_NORM]), QR, r[AI_GLM_5_3__MLA__EPS], over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(qnr), (const uint16_t*)(uintptr_t)(qn), (const uint16_t*)(uintptr_t)(signs), QR, over);
    ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, AI_GLM_5_3__MLA__QB, q, qnr, d[2], heads * NOPE, QR);
    doors->rmsnorm((uint16_t*)(uintptr_t)(ckn), (const uint16_t*)(uintptr_t)(kv), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__MLA__KVA_NORM]), LAT, r[AI_GLM_5_3__MLA__EPS], over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(cache + 2u * pos * LAT), (const uint16_t*)(uintptr_t)(ckn), (const uint16_t*)(uintptr_t)(signs), LAT, over);
    /* ⛳ the attention's own scale is 1/√latent and the model's 1/√head, so the absorbed query carries their ratio */
    doors->attention_absorb((uint16_t*)(uintptr_t)(qabs), (const uint16_t*)(uintptr_t)(q), (const uint16_t*)(uintptr_t)(kvb), heads, NOPE, LAT, stride,
                            sqrtf((float)LAT / (float)NOPE), over);
    uint64_t src = cache;
    if (indexed) {
        /* the indexer: this position's key and gate kept, a pool finished when it fills, and past `topk` positions the
           top pools and the filling pool's positions chosen, their latents gathered to attend.
           ⛳ A POSITION'S KEY AND GATE ARE READ ONLY TO FINISH ITS POOL — the pool is scored by its pooled key, and the
           attention reads the latents — so they are kept in a ring of `kpool` rows, the pooled keys a row a pool */
        const uint64_t iq = y + 2u * H, iw = iq + 2u * IH * ID, ik = iw + 2u * IH;
        const uint64_t keys = at[AI_GLM_5_3__IDX__KEYS], gates = at[AI_GLM_5_3__IDX__GATES], pooled = at[AI_GLM_5_3__IDX__POOLED];
        ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, AI_GLM_5_3__IDX__WQB, iq, qnr, id_[0], IH * ID, QR);
        ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, AI_GLM_5_3__IDX__WK, ik, hr, id_[1], ID, H);
        ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, AI_GLM_5_3__IDX__WP, iw, hr, id_[2], IH, H);
        const uint64_t ring = 2u * (pos % KP) * ID;
        ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, AI_GLM_5_3__IDX__GATE, gates + ring, hr, id_[3], ID, H);
        doors->index_layernorm((uint16_t*)(uintptr_t)(keys + ring), (const uint16_t*)(uintptr_t)(ik),
                               (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__IDX__KNORM_W]), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__IDX__KNORM_B]),
                               ID, 1e-6f, over);
        if (tail == 0u)
            doors->index_pool((uint16_t*)(uintptr_t)(pooled + 2u * (pools - 1u) * ID), (const uint16_t*)(uintptr_t)(keys),
                              (const uint16_t*)(uintptr_t)(gates), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__IDX__APE]), 0u, 1u, ID, KP, over);
        if (sparse) {
            const uint64_t index = at[AI_GLM_5_3__IDX__INDEX];
            doors->index_scores((float*)(uintptr_t)(at[AI_GLM_5_3__IDX__SCORES]), (const uint16_t*)(uintptr_t)(iq), (const uint16_t*)(uintptr_t)(iw),
                                (const uint16_t*)(uintptr_t)(pooled), IH, ID, pools, 1.0f / sqrtf((float)ID));
            doors->index_select((uint32_t*)(uintptr_t)(index), (uint32_t*)(uintptr_t)(index + 4u * (TOPK + KP)),
                                (const float*)(uintptr_t)(at[AI_GLM_5_3__IDX__SCORES]), pools, chosen_pools, KP, pools * KP, tail);
            doors->index_gather((uint16_t*)(uintptr_t)(at[AI_GLM_5_3__IDX__GATHER]), (const uint16_t*)(uintptr_t)(cache),
                                (const uint32_t*)(uintptr_t)(index), alen, LAT);
            src = at[AI_GLM_5_3__IDX__GATHER];
        }
    }
    /* ⭐ the 64 heads' scores together, the latents read once for all of them (`attention_scores_grouped`), then each
     *   head's softmax — `attention_scores`' answer, ~6-14x sooner on the MI50 (▶ nn's body) */
    bool grouped = true;
    if (grouped) {
        doors->attention_scores_grouped((float*)(uintptr_t)at[AI_GLM_5_3__MLA__SCORES], (const uint16_t*)(uintptr_t)(qabs),
                                        (const uint16_t*)(uintptr_t)(src), heads, 1u, LAT, alen);
        doors->attention_softmax_rows((float*)(uintptr_t)at[AI_GLM_5_3__MLA__SCORES], heads, alen);
    } else
        doors->attention_scores((float*)(uintptr_t)at[AI_GLM_5_3__MLA__SCORES], (const uint16_t*)(uintptr_t)(qabs),
                                (const uint16_t*)(uintptr_t)(src), heads, 1u, LAT, alen);
    doors->attention_mix((uint16_t*)(uintptr_t)(olat), (const float*)(uintptr_t)at[AI_GLM_5_3__MLA__SCORES], (const uint16_t*)(uintptr_t)(src), heads, 1u,
                         LAT, alen, over);
    doors->attention_expand((uint16_t*)(uintptr_t)(vh), (const uint16_t*)(uintptr_t)(olat), (const uint16_t*)(uintptr_t)(kvb), heads, VD, LAT, stride, NOPE, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(vhr), (const uint16_t*)(uintptr_t)(vh), (const uint16_t*)(uintptr_t)(signs), heads * VD, over);
    ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, AI_GLM_5_3__MLA__O, y, vhr, d[3], H, heads * VD);
    ai_glm_5_3__layer__zzprivate_place(doors, over, scratch, streams, y, H, M);
    return 0u;
}

/* A swiglu on the card: gate and up in one grouped launch on `x` (rotated), clamped, rotated, down into `out`. */
static inline void ai_glm_5_3__layer__zzprivate_swiglu(const nn__doors* doors, unsigned int* over, const uint64_t* at,
                                                       const uint64_t* room, unsigned gate, unsigned up, unsigned down,
                                                       const uint64_t* d, uint64_t x, uint64_t gu, uint64_t signs, uint64_t out,
                                                       uint64_t H, uint64_t I, float limit) {
    nn__turboquant__groups g = {};
    ai_glm_5_3__table__zzpackage_group(&g, 0u, at, gate, I, d[0], 0u);
    ai_glm_5_3__table__zzpackage_group(&g, 1u, at, up, I, d[1], I);
    g.count = 2u;
    const uint64_t act = gu + 4u * I, actr = act + 2u * I;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)(gu), g, (const uint16_t*)(uintptr_t)(x), H, over);
    doors->swiglu_clamped((uint16_t*)(uintptr_t)(act), (const uint16_t*)(uintptr_t)(gu), 1u, I, limit, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(actr), (const uint16_t*)(uintptr_t)(act), (const uint16_t*)(uintptr_t)(signs), I, over);
    ai_glm_5_3__layer__zzprivate_gemv(doors, over, at, room, down, out, actr, d[2], H, I);
}

/* `(ai_glm_5_3__mlp streams table)` -> `streams`. The MLP site of a dense layer. */
static uint64_t ai_glm_5_3__mlp__zzprivate_row(const sys__heap_node* argv, sys__engine__ctx* ctx, uint64_t row) {
    uint64_t streams = 0, s_room = 0;
    if (!ai_glm_5_3__layer__zzprivate_open(argv, &streams, &s_room)) return SYS__OPCODES__FAULT_TYPE;
    uint64_t at[AI_GLM_5_3__MLP__TABLE], room[AI_GLM_5_3__MLP__TABLE], v[AI_GLM_5_3__MLP__TABLE];
    float r[AI_GLM_5_3__MLP__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[1].args[0], AI_GLM_5_3__MLP__PLANES, AI_GLM_5_3__MLP__EPS, AI_GLM_5_3__MLP__TABLE,
                                           at, room, v, r))
        return AI_GLM_5_3__LAYER__FAULT_TABLE;
    const uint64_t H = v[AI_GLM_5_3__MLP__HIDDEN], M = v[AI_GLM_5_3__MLP__MULT], I = v[AI_GLM_5_3__MLP__INTER];
    if (!ai_glm_5_3__layer__zzprivate_row(&streams, &s_room, row, M * H)) return AI_GLM_5_3__LAYER__FAULT_TABLE;
    if (H == 0u || M == 0u || M > NN__HYPER__MULT_MAX || I == 0u || H > (1ull << 24) || I > (1ull << 24) || !nn__primitives__fits(M * H, s_room))
        return AI_GLM_5_3__LAYER__FAULT_TABLE;
    /* scratch, halves after the carry: xin · h · hr · gu · act · actr · y */
    if (room[AI_GLM_5_3__MLP__SCRATCH] < AI_GLM_5_3__CARRY(H) + 2u * (4u * H + 4u * I))
        return AI_GLM_5_3__LAYER__FAULT_ROOM;
    const uint64_t scratch = at[AI_GLM_5_3__MLP__SCRATCH], signs = at[AI_GLM_5_3__MLP__SIGNS];
    const uint64_t xin = scratch + AI_GLM_5_3__CARRY(H), h = xin + 2u * H, hr = h + 2u * H, gu = hr + 2u * H, y = gu + 2u * 4u * I;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    unsigned int* over = ctx->fault_word;
    ai_glm_5_3__layer__zzprivate_collapse(doors, over, at, scratch, streams, xin, h, H, M, v[AI_GLM_5_3__MLP__ITERS],
                                          r[AI_GLM_5_3__MLP__EPS], r[AI_GLM_5_3__MLP__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(hr), (const uint16_t*)(uintptr_t)(h), (const uint16_t*)(uintptr_t)(signs), H, over);
    ai_glm_5_3__layer__zzprivate_swiglu(doors, over, at, room, AI_GLM_5_3__MLP__GATE, AI_GLM_5_3__MLP__UP, AI_GLM_5_3__MLP__DOWN,
                                        &v[AI_GLM_5_3__MLP__D], hr, gu, signs, y, H, I, r[AI_GLM_5_3__MLP__LIMIT]);
    ai_glm_5_3__layer__zzprivate_place(doors, over, scratch, streams, y, H, M);
    return 0u;
}

/* The MoE site's table, for its three card verbs. */
static inline bool ai_glm_5_3__layer__zzprivate_moe(uint64_t table, uint64_t* at, uint64_t* room, uint64_t* v, float* r) {
    if (!ai_glm_5_3__table__zzpackage_read(table, AI_GLM_5_3__MOE__PLANES, AI_GLM_5_3__MOE__EPS, AI_GLM_5_3__MOE__TABLE, at, room, v, r))
        return false;
    const uint64_t H = v[AI_GLM_5_3__MOE__HIDDEN], M = v[AI_GLM_5_3__MOE__MULT], E = v[AI_GLM_5_3__MOE__EXPERTS];
    const uint64_t K = v[AI_GLM_5_3__MOE__TOP_K], I = v[AI_GLM_5_3__MOE__INTER];
    return H != 0u && M != 0u && M <= NN__HYPER__MULT_MAX && E != 0u && K != 0u && K <= E && K <= AI_GLM_5_3__TOP_K_MAX && I != 0u
        && H <= (1ull << 24) && E <= (1ull << 20) && I <= (1ull << 24)
        && room[AI_GLM_5_3__MOE__SCRATCH] >= AI_GLM_5_3__CARRY(H) + 2u * (2u * H + E + 4u * I + H);
}

/* `(ai_glm_5_3__route streams table hand picks)` -> `hand`. The MoE site to the router: the streams collapsed and
 * normed, the input rotated into the hand, the router's `k` picks into `picks` and their weights after the input. */
static sys__heap_node ai_glm_5_3__route__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, hand = 0, h_room = 0;
    if (!ai_glm_5_3__layer__zzprivate_open(argv, &streams, &s_room) || !nn__primitives__room(&argv[2], &hand, &h_room)
     || !sys__heap_node__carries_reference(argv[3].dtype) || !sys__node_array__is(argv[3].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__MOE__TABLE], room[AI_GLM_5_3__MOE__TABLE], v[AI_GLM_5_3__MOE__TABLE];
    float r[AI_GLM_5_3__MOE__TABLE];
    if (!ai_glm_5_3__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__MOE__HIDDEN], M = v[AI_GLM_5_3__MOE__MULT], E = v[AI_GLM_5_3__MOE__EXPERTS];
    const uint64_t K = v[AI_GLM_5_3__MOE__TOP_K];
    if (!nn__primitives__fits(M * H, s_room) || !nn__primitives__fits(H + K, h_room) || !nn__primitives__fits(E, room[AI_GLM_5_3__MOE__BIAS])
     || !nn__primitives__fits(E * H, room[AI_GLM_5_3__MOE__ROUTER]))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t scratch = at[AI_GLM_5_3__MOE__SCRATCH], xin = scratch + AI_GLM_5_3__CARRY(H), h = xin + 2u * H, logits = h + 2u * H;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    ai_glm_5_3__layer__zzprivate_collapse(doors, over, at, scratch, streams, xin, h, H, M, v[AI_GLM_5_3__MOE__ITERS],
                                          r[AI_GLM_5_3__MOE__EPS], r[AI_GLM_5_3__MOE__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(hand), (const uint16_t*)(uintptr_t)(h), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__MOE__SIGNS]), H, over);
    doors->expert_multiply_fp16((uint16_t*)(uintptr_t)(logits), (const uint16_t*)(uintptr_t)(at[AI_GLM_5_3__MOE__ROUTER]), (const uint16_t*)(uintptr_t)(h), E, H, over);
    const uint64_t landed = nn__routed__top_k(ctx, doors, argv[3].args[0], hand + 2u * H, logits, at[AI_GLM_5_3__MOE__BIAS],
                                              r[AI_GLM_5_3__MOE__SCALE], E, K, 0);
    if (landed != 0u) return sys__engine__abi__error(landed);
    return nn__doors_answer(&argv[2]);
}

/* `(ai_glm_5_3__shared hand table)` -> `hand`. The shared expert on the hand's input, its answer into the carry — what
 * the card does while the routed experts run elsewhere. */
static sys__heap_node ai_glm_5_3__shared__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hand = 0, h_room = 0;
    if (!nn__primitives__room(&argv[0], &hand, &h_room) || !sys__heap_node__carries_reference(argv[1].dtype)
     || !sys__node_array__is(argv[1].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__MOE__TABLE], room[AI_GLM_5_3__MOE__TABLE], v[AI_GLM_5_3__MOE__TABLE];
    float r[AI_GLM_5_3__MOE__TABLE];
    if (!ai_glm_5_3__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__MOE__HIDDEN], I = v[AI_GLM_5_3__MOE__INTER];
    if (!nn__primitives__fits(H, h_room)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t scratch = at[AI_GLM_5_3__MOE__SCRATCH], sh = scratch + AI_GLM_5_3__CARRY_MIX;
    const uint64_t gu = scratch + AI_GLM_5_3__CARRY(H) + 2u * (2u * H + v[AI_GLM_5_3__MOE__EXPERTS]);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    ai_glm_5_3__layer__zzprivate_swiglu(doors, ctx->fault_word, at, room, AI_GLM_5_3__MOE__GATE, AI_GLM_5_3__MOE__UP,
                                        AI_GLM_5_3__MOE__DOWN, &v[AI_GLM_5_3__MOE__D], hand, gu, at[AI_GLM_5_3__MOE__SIGNS], sh, H, I,
                                        r[AI_GLM_5_3__MOE__LIMIT]);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_glm_5_3__close streams table routed)` -> `streams`. The routed experts' sum added to the shared expert's, and the
 * streams mixed again with it. */
static sys__heap_node ai_glm_5_3__close__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, routed = 0, r_room = 0;
    if (!ai_glm_5_3__layer__zzprivate_open(argv, &streams, &s_room) || !nn__primitives__room(&argv[2], &routed, &r_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__MOE__TABLE], room[AI_GLM_5_3__MOE__TABLE], v[AI_GLM_5_3__MOE__TABLE];
    float r[AI_GLM_5_3__MOE__TABLE];
    if (!ai_glm_5_3__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__MOE__HIDDEN], M = v[AI_GLM_5_3__MOE__MULT];
    if (!nn__primitives__fits(M * H, s_room) || !nn__primitives__fits(H, r_room)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t scratch = at[AI_GLM_5_3__MOE__SCRATCH], sh = scratch + AI_GLM_5_3__CARRY_MIX, y = scratch + AI_GLM_5_3__CARRY(H);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->vector_add((uint16_t*)(uintptr_t)(y), (const uint16_t*)(uintptr_t)(sh), (const uint16_t*)(uintptr_t)(routed), H, ctx->fault_word);
    ai_glm_5_3__layer__zzprivate_place(doors, ctx->fault_word, scratch, streams, y, H, M);
    return nn__doors_answer(&argv[0]);
}

/* A card's tier (▶ nn's `nn__expert_tier`): true with `tier` read as layer `layer` reads it where the verb was handed the
 * binding as its last argument (`argv[at]`), false where it was not; `ok` false when it was and holds no tier for this
 * worker. */
static bool ai_glm_5_3__experts__zzprivate_tier(const sys__heap_node* argv, unsigned argc, unsigned at, uint64_t layer, nn__expert_tier* tier, bool* ok) {
    *ok = true;
    if (argc <= at) return false;
    *ok = sys__heap_node__carries_reference(argv[at].dtype) && nn__expert_tier__of(argv[at].args[0], layer, tier);
    return *ok;
}

/* The experts table's backing, when it has one: none is `file` zero. False when the cell is there and not a backing. */
static bool ai_glm_5_3__experts__zzprivate_backing(uint64_t table, nn__expert__backing* backing) {
    const nn__expert__backing none = {};
    *backing = none;
    if (sys__node_array__length(table) <= AI_GLM_5_3__EXP__BACKING) return true;
    const sys__heap_node n = sys__node_array__borrow(table, AI_GLM_5_3__EXP__BACKING);
    if (n.dtype == SYS__KIND__VALUE_INT && n.args[0] == 0ull) return true;
    return sys__heap_node__carries_reference(n.dtype) && nn__expert__backing_of(n.args[0], backing);
}

/* The routed experts as nn runs them (▶ nn's `nn__routed`), out of an experts table: its shape, its planes' places in a
 * slot, the experts this worker holds, its arithmetic, the clamped SwiGLU's limit, its scratch and its backing. False when
 * the backing cell is not one. */
static bool ai_glm_5_3__experts__zzprivate_routed(uint64_t table, const uint64_t* at, const uint64_t* room, const uint64_t* v,
                                                  const float* r, nn__routed* out) {
    const nn__routed none = {};
    *out = none;
    const uint64_t H = v[AI_GLM_5_3__EXP__HIDDEN], K = v[AI_GLM_5_3__EXP__TOP_K];
    out->layer = v[AI_GLM_5_3__EXP__LAYER]; out->type = v[AI_GLM_5_3__EXP__TYPE];
    out->hidden = H; out->inter = v[AI_GLM_5_3__EXP__INTER]; out->experts = v[AI_GLM_5_3__EXP__EXPERTS]; out->top_k = K;
    out->bits = v[AI_GLM_5_3__EXP__D];
    out->up_lut = v[AI_GLM_5_3__EXP__UP_LUT]; out->down = v[AI_GLM_5_3__EXP__DOWN]; out->down_lut = v[AI_GLM_5_3__EXP__DOWN_LUT];
    out->first = v[AI_GLM_5_3__EXP__FIRST]; out->count = v[AI_GLM_5_3__EXP__COUNT];
    /* ⚖ *"two different instructions so at boot time the lisp can decide"*: the table says which arithmetic */
    out->int8 = v[AI_GLM_5_3__EXP__INT8] == 1u; out->limit = r[AI_GLM_5_3__EXP__LIMIT];
    out->signs = at[AI_GLM_5_3__EXP__SIGNS]; out->zero = at[AI_GLM_5_3__EXP__ZERO];
    out->scratch = at[AI_GLM_5_3__EXP__SCRATCH]; out->scratch_room = room[AI_GLM_5_3__EXP__SCRATCH];
    out->hand_bytes = AI_GLM_5_3__HAND(H, K); out->picks_at = AI_GLM_5_3__HAND_PICKS(H, K); out->batch = AI_GLM_5_3__EXP__BATCH;
    return ai_glm_5_3__experts__zzprivate_backing(table, &out->backing);
}

/* `(ai_glm_5_3__experts hand picks table routed [nn__expert_tier])` -> `routed`: Σ w_j · down_j(swiglu(gate_up_j(x))) over the
 * picks this worker holds, the SwiGLU clamped — as nn runs a position's routed experts (▶ nn's `nn__routed__position`):
 * the ones in memory computed while the rest arrive, a card's tier's picks computed and marked held, and zero where this
 * worker holds none. */
static sys__heap_node ai_glm_5_3__experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hand = 0, h_room = 0, routed = 0, r_room = 0;
    if (!nn__primitives__room(&argv[0], &hand, &h_room) || !nn__primitives__room(&argv[3], &routed, &r_room)
     || !sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__EXP__TABLE], room[AI_GLM_5_3__EXP__TABLE], v[AI_GLM_5_3__EXP__TABLE];
    float r[AI_GLM_5_3__EXP__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[2].args[0], AI_GLM_5_3__EXP__PLANES, AI_GLM_5_3__EXP__LIMIT, AI_GLM_5_3__EXP__TABLE,
                                           at, room, v, r))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__EXP__HIDDEN], K = v[AI_GLM_5_3__EXP__TOP_K];
    if (K > AI_GLM_5_3__TOP_K_MAX || v[AI_GLM_5_3__EXP__INT8] > 1u || !nn__primitives__fits(H + K, h_room) || !nn__primitives__fits(H, r_room)
     || !nn__primitives__fits(H, room[AI_GLM_5_3__EXP__ZERO]))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    nn__routed rt;
    if (!ai_glm_5_3__experts__zzprivate_routed(argv[2].args[0], at, room, v, r, &rt)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    nn__expert_tier tier;
    bool tier_ok = true;
    const bool holds = ai_glm_5_3__experts__zzprivate_tier(argv, argc, 4u, rt.layer, &tier, &tier_ok);
    if (!tier_ok) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t f = nn__routed__position(ctx, doors, &rt, hand, argv[1].args[0], holds ? &tier : 0, routed, 0);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[3]);
}

/* ══ THE SITES AT ONE POSITION, AND AS ROWS ═════════════════════════════════════════════════════════════════════════
 * A site verb takes the streams of one position; its `_rows` form takes `n` rows of them, a position each, in order, and
 * runs the same body on each — the attention in position order, since each row's reads the rows before it. */
static sys__heap_node ai_glm_5_3__layer__zzprivate_answer(const sys__heap_node* argv, uint64_t fault) {
    return fault != 0u ? sys__engine__abi__error(fault) : nn__doors_answer(&argv[0]);
}
static bool ai_glm_5_3__layer__zzprivate_rows(const sys__heap_node* n, uint64_t* rows) {
    if (n->dtype != SYS__KIND__VALUE_INT || n->args[0] == 0u || n->args[0] > AI_GLM_5_3__ROWS_MAX) return false;
    *rows = n->args[0];
    return true;
}

/* `(ai_glm_5_3__kda streams table)` -> `streams`. */
static sys__heap_node ai_glm_5_3__kda__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    return ai_glm_5_3__layer__zzprivate_answer(argv, ai_glm_5_3__kda__zzprivate_row(argv, ctx, 0u));
}
/* `(ai_glm_5_3__mla streams table pos)` -> `streams`. */
static sys__heap_node ai_glm_5_3__mla__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[2].dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return ai_glm_5_3__layer__zzprivate_answer(argv, ai_glm_5_3__mla__zzprivate_row(argv, ctx, argv[2].args[0], 0u));
}
/* `(ai_glm_5_3__mlp streams table)` -> `streams`. */
static sys__heap_node ai_glm_5_3__mlp__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    return ai_glm_5_3__layer__zzprivate_answer(argv, ai_glm_5_3__mlp__zzprivate_row(argv, ctx, 0u));
}
/* ══ ⭐⭐ A PROMPT AS ROWS — each site over `n` positions at once ═══════════════════════════════════════════════
 * Row `t` of `streams` is position `first + t`. Every matrix runs ONCE over the chunk, through nn's expert-major GEMM
 * (`expert_groups`, `expert_rows`) with each row its own pair, so its weights are read once a chunk and not once a row;
 * the collapse, the norms, the rotations, the convolutions and KDA's recurrence are one launch each over every row.
 * What stays a row at a time is MLA's attention — a row attends the rows before it, and the indexer's ring and pools
 * fill in order — and the router's picks, a row's own hand.
 * ⛳ THE `work` BUFFER holds what the chunk's rows need at once: the pairs, each row's weights of the collapse (a record
 * of `AI_GLM_5_3__CARRY_MIX` bytes), and the rows of each step. The table's own scratch keeps what one row needs. */

#define AI_GLM_5_3__ROWS__MIX_FLOATS  (AI_GLM_5_3__CARRY_MIX / 4ull)

/* The next `bytes` of the work buffer, 256-aligned; 0 when it has no room left. */
static inline uint64_t ai_glm_5_3__rows__zzprivate_take(uint64_t* next, uint64_t end, uint64_t bytes) {
    const uint64_t at = (*next + 255u) & ~255ull;
    if (at > end || end - at < bytes) return 0u;
    *next = at + bytes;
    return at;
}

/* The chunk's pairs, a row of `x` and a row of `out` each: `n` of `(t, t)`, then `n` of `(t, 2t)` and `n` of `(t, 2t + 1)` —
 * a gate's answer and an up's side by side for each row, as the clamped swiglu reads them. */
static bool ai_glm_5_3__rows__zzprivate_pairs(sys__engine__ctx* ctx, uint64_t pairs, uint64_t n) {
    uint32_t* p = (uint32_t*)malloc(sizeof(uint32_t) * 6u * n);
    if (p == 0) return false;
    for (uint64_t t = 0u; t < n; ++t) {
        p[2u * t] = (uint32_t)t;                  p[2u * t + 1u] = (uint32_t)t;
        p[2u * (n + t)] = (uint32_t)t;            p[2u * (n + t) + 1u] = (uint32_t)(2u * t);
        p[2u * (2u * n + t)] = (uint32_t)t;       p[2u * (2u * n + t) + 1u] = (uint32_t)(2u * t + 1u);
    }
    const bool ok = sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)pairs, p, 24u * n);
    free(p);
    return ok;
}

/* Every row's site input: its streams collapsed by the site's map — the weights into its record of `mixes` — and normed. */
static inline void ai_glm_5_3__rows__zzprivate_collapse(const nn__doors* doors, unsigned int* over, const uint64_t* at,
                                                        uint64_t mixes, uint64_t streams, uint64_t xin, uint64_t h, uint64_t H,
                                                        uint64_t M, uint64_t n, uint64_t iters, float eps, float hc_eps) {
    doors->hyper_logits_rows((float*)(uintptr_t)mixes + NN__HYPER__MIX(M), (const uint16_t*)(uintptr_t)streams,
                             (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__HC_FN], H, M, eps, n, AI_GLM_5_3__ROWS__MIX_FLOATS);
    doors->hyper_pre_rows((uint16_t*)(uintptr_t)xin, (float*)(uintptr_t)mixes, (const uint16_t*)(uintptr_t)streams,
                          (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__HC_BASE], (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__HC_SCALE],
                          H, M, iters, hc_eps, n, AI_GLM_5_3__ROWS__MIX_FLOATS, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)xin, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__NORM],
                        H, n, H, H, eps, over);
}

/* Every row's streams mixed again with its output, a row of `y`, by the weights in its record of `mixes`. */
static inline void ai_glm_5_3__rows__zzprivate_place(const nn__doors* doors, unsigned int* over, uint64_t mixes, uint64_t streams,
                                                     uint64_t y, uint64_t H, uint64_t M, uint64_t n) {
    doors->hyper_post_rows((uint16_t*)(uintptr_t)streams, (const uint16_t*)(uintptr_t)streams, (const uint16_t*)(uintptr_t)y,
                           (const float*)(uintptr_t)mixes, H, M, n, AI_GLM_5_3__ROWS__MIX_FLOATS, over);
}

/* `out = W x` for every row: a matrix of the table (codes, then scales) over the rows of `x`, `cols` wide. */
static inline void ai_glm_5_3__rows__zzprivate_gemm(const nn__doors* doors, unsigned int* over, const uint64_t* at,
                                                    const uint64_t* room, unsigned plane, uint64_t out, uint64_t x, uint64_t pairs,
                                                    uint64_t n, uint64_t d, uint64_t rows, uint64_t cols) {
    doors->expert_rows((uint16_t*)(uintptr_t)out, (const uint8_t*)(uintptr_t)at[plane], room[plane],
                       (const uint8_t*)(uintptr_t)at[plane + 1u], room[plane + 1u], (const uint16_t*)(uintptr_t)x,
                       (const uint32_t*)(uintptr_t)pairs, n, d, rows, cols, over);
}

/* A matrix of the table as entry `i` of a grouped launch over the rows its pairs name. */
static inline void ai_glm_5_3__rows__zzprivate_group(nn__expert__groups* g, unsigned i, const uint64_t* at, unsigned plane,
                                                     uint64_t rows, uint64_t d, uint64_t out_at, uint64_t pairs_at, uint64_t pairs) {
    g->codes[i] = at[plane]; g->luts[i] = at[plane + 1u]; g->out_rows[i] = rows; g->d[i] = d;
    g->out_at[i] = out_at; g->pairs_at[i] = pairs_at; g->pairs[i] = pairs;
}

/* A swiglu over every row: gate and up in one grouped launch on `x` (rotated), side by side a row, clamped, rotated, down
 * into `out`. `gu` is `n · 2I` halves, `act` and `actr` `n · I`. */
static inline void ai_glm_5_3__rows__zzprivate_swiglu(const nn__doors* doors, unsigned int* over, const uint64_t* at,
                                                      const uint64_t* room, unsigned gate, unsigned up, unsigned down,
                                                      const uint64_t* d, uint64_t x, uint64_t pairs, uint64_t gu, uint64_t act,
                                                      uint64_t actr, uint64_t signs, uint64_t out, uint64_t H, uint64_t I,
                                                      uint64_t n, float limit) {
    nn__expert__groups g = {};
    ai_glm_5_3__rows__zzprivate_group(&g, 0u, at, gate, I, d[0], 0u, n, n);
    ai_glm_5_3__rows__zzprivate_group(&g, 1u, at, up, I, d[1], 0u, 2u * n, n);
    g.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)gu, g, (const uint16_t*)(uintptr_t)x, (const uint32_t*)(uintptr_t)pairs, H, over);
    doors->swiglu_clamped((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, n, I, limit, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)signs, n * I, over);
    ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, down, out, actr, pairs, n, d[2], H, I);
}

/* The operands every rows verb shares: the streams, the table, the count and the work buffer at `w`. */
static inline bool ai_glm_5_3__rows__zzprivate_open(const sys__heap_node* argv, unsigned c, unsigned w, uint64_t* streams,
                                                    uint64_t* s_room, uint64_t* n, uint64_t* work, uint64_t* w_room) {
    return ai_glm_5_3__layer__zzprivate_open(argv, streams, s_room) && ai_glm_5_3__layer__zzprivate_rows(&argv[c], n)
        && nn__primitives__room(&argv[w], work, w_room);
}

/* `(ai_glm_5_3__kda_rows streams table n work)` -> `streams`. The KDA site over `n` rows: the six projections one grouped
 * launch, the three convolutions and the recurrence walking the rows in order, a launch each, the gates' second
 * projections and the output projection one launch each. */
static sys__heap_node ai_glm_5_3__kda_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, n = 0, work = 0, w_room = 0;
    if (!ai_glm_5_3__rows__zzprivate_open(argv, 2u, 3u, &streams, &s_room, &n, &work, &w_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__KDA__TABLE], room[AI_GLM_5_3__KDA__TABLE], v[AI_GLM_5_3__KDA__TABLE];
    float r[AI_GLM_5_3__KDA__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[1].args[0], AI_GLM_5_3__KDA__PLANES, AI_GLM_5_3__KDA__EPS, AI_GLM_5_3__KDA__TABLE,
                                           at, room, v, r))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__KDA__HIDDEN], M = v[AI_GLM_5_3__KDA__MULT], heads = v[AI_GLM_5_3__KDA__HEADS];
    const uint64_t hd = v[AI_GLM_5_3__KDA__HEAD_DIM], KW = heads * hd;
    const uint64_t* d = &v[AI_GLM_5_3__KDA__D];            /* q k v fa ga b fb gb o */
    if (H == 0u || M == 0u || M > NN__HYPER__MULT_MAX || heads == 0u || hd == 0u || H > (1ull << 24) || KW > (1ull << 24)
     || !nn__primitives__fits(n * M * H, s_room) || room[AI_GLM_5_3__KDA__STATE] < heads * hd * hd * 4u
     || !nn__primitives__fits(9u * KW, room[AI_GLM_5_3__KDA__WINDOW]))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    uint64_t next = work;
    const uint64_t end = work + w_room;
    const uint64_t pairs = ai_glm_5_3__rows__zzprivate_take(&next, end, 24u * n);
    const uint64_t mixes = ai_glm_5_3__rows__zzprivate_take(&next, end, AI_GLM_5_3__CARRY_MIX * n);
    const uint64_t xin = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t h = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t hr = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    /* the projections, a matrix's rows after another's: q · k · v · f_a · g_a · b */
    const uint64_t proj = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * (3u * KW + 2u * hd + heads));
    const uint64_t conved = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * 3u * KW);
    const uint64_t far = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * hd);
    const uint64_t gar = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * hd);
    const uint64_t fb = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * KW);
    const uint64_t gate = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * KW);
    const uint64_t ko = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * KW);
    const uint64_t kor = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * KW);
    const uint64_t y = xin;                                /* the collapse's input is read once, by the norm */
    if (pairs == 0u || mixes == 0u || xin == 0u || h == 0u || hr == 0u || proj == 0u || conved == 0u || far == 0u || gar == 0u
     || fb == 0u || gate == 0u || ko == 0u || kor == 0u)
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_ROOM);
    const uint64_t fa = proj + 2u * n * 3u * KW, ga = fa + 2u * n * hd, b = ga + 2u * n * hd;
    const uint64_t signs = at[AI_GLM_5_3__KDA__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_glm_5_3__rows__zzprivate_pairs(ctx, pairs, n)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    ai_glm_5_3__rows__zzprivate_collapse(doors, over, at, mixes, streams, xin, h, H, M, n, v[AI_GLM_5_3__KDA__ITERS],
                                         r[AI_GLM_5_3__KDA__EPS], r[AI_GLM_5_3__KDA__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)hr, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, n * H, over);
    nn__expert__groups g = {};
    ai_glm_5_3__rows__zzprivate_group(&g, 0u, at, AI_GLM_5_3__KDA__Q, KW, d[0], 0u, 0u, n);
    ai_glm_5_3__rows__zzprivate_group(&g, 1u, at, AI_GLM_5_3__KDA__K, KW, d[1], n * KW, 0u, n);
    ai_glm_5_3__rows__zzprivate_group(&g, 2u, at, AI_GLM_5_3__KDA__V, KW, d[2], 2u * n * KW, 0u, n);
    ai_glm_5_3__rows__zzprivate_group(&g, 3u, at, AI_GLM_5_3__KDA__FA, hd, d[3], 3u * n * KW, 0u, n);
    ai_glm_5_3__rows__zzprivate_group(&g, 4u, at, AI_GLM_5_3__KDA__GA, hd, d[4], 3u * n * KW + n * hd, 0u, n);
    ai_glm_5_3__rows__zzprivate_group(&g, 5u, at, AI_GLM_5_3__KDA__B, heads, d[5], 3u * n * KW + 2u * n * hd, 0u, n);
    g.count = 6u;
    doors->expert_groups((uint16_t*)(uintptr_t)proj, g, (const uint16_t*)(uintptr_t)hr, (const uint32_t*)(uintptr_t)pairs, H, over);
    const unsigned conv[3] = {AI_GLM_5_3__KDA__Q_CONV, AI_GLM_5_3__KDA__K_CONV, AI_GLM_5_3__KDA__V_CONV};
    for (unsigned i = 0u; i < 3u; ++i)
        doors->deltanet_conv_steps((uint16_t*)(uintptr_t)(conved + 2u * i * n * KW), (uint16_t*)(uintptr_t)(at[AI_GLM_5_3__KDA__WINDOW] + 2u * 3u * i * KW),
                                   (const uint16_t*)(uintptr_t)at[conv[i]], (const uint16_t*)(uintptr_t)(proj + 2u * i * n * KW), KW, n, over);
    doors->hadamard_blocks((uint16_t*)(uintptr_t)far, (const uint16_t*)(uintptr_t)fa, (const uint16_t*)(uintptr_t)signs, n * hd, hd, 0u, over);
    doors->hadamard_blocks((uint16_t*)(uintptr_t)gar, (const uint16_t*)(uintptr_t)ga, (const uint16_t*)(uintptr_t)signs, n * hd, hd, 0u, over);
    ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, AI_GLM_5_3__KDA__FB, fb, far, pairs, n, d[6], KW, hd);
    ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, AI_GLM_5_3__KDA__GB, gate, gar, pairs, n, d[7], KW, hd);
    doors->kda_steps((float*)(uintptr_t)at[AI_GLM_5_3__KDA__STATE], (const uint16_t*)(uintptr_t)conved, (const uint16_t*)(uintptr_t)fb,
                     (const uint16_t*)(uintptr_t)b, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__KDA__DT_BIAS],
                     (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__KDA__A_LOG], (const uint16_t*)(uintptr_t)gate,
                     (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__KDA__O_NORM], (uint16_t*)(uintptr_t)ko, heads, hd,
                     r[AI_GLM_5_3__KDA__LOWER], r[AI_GLM_5_3__KDA__EPS], n, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)kor, (const uint16_t*)(uintptr_t)ko, (const uint16_t*)(uintptr_t)signs, n * KW, over);
    ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, AI_GLM_5_3__KDA__O, y, kor, pairs, n, d[8], H, KW);
    ai_glm_5_3__rows__zzprivate_place(doors, over, mixes, streams, y, H, M, n);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_glm_5_3__mla_rows streams table first n work)` -> `streams`: row `t` is position `first + t`. Every row's
 * projections, norms and latent — written rotated into the cache's rows `first ..` before any row attends — over the
 * chunk at once, and the indexer's projections too; then a row at a time, in order, the indexer's ring and pools and
 * the attention over the cache up to the row's own position; then the output projection over every row. */
static sys__heap_node ai_glm_5_3__mla_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, n = 0, work = 0, w_room = 0;
    if (argv[2].dtype != SYS__KIND__VALUE_INT || !ai_glm_5_3__rows__zzprivate_open(argv, 3u, 4u, &streams, &s_room, &n, &work, &w_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t first = argv[2].args[0];
    uint64_t at[AI_GLM_5_3__IDX__TABLE], room[AI_GLM_5_3__IDX__TABLE], v[AI_GLM_5_3__IDX__TABLE];
    float r[AI_GLM_5_3__IDX__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[1].args[0], AI_GLM_5_3__MLA__PLANES, AI_GLM_5_3__MLA__EPS, AI_GLM_5_3__MLA__TABLE,
                                           at, room, v, r))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__MLA__HIDDEN], M = v[AI_GLM_5_3__MLA__MULT], heads = v[AI_GLM_5_3__MLA__HEADS];
    const uint64_t QR = v[AI_GLM_5_3__MLA__Q_RANK], LAT = v[AI_GLM_5_3__MLA__LATENT], NOPE = v[AI_GLM_5_3__MLA__NOPE];
    const uint64_t VD = v[AI_GLM_5_3__MLA__V_DIM], stride = NOPE + VD, to = first + n;
    const uint64_t* d = &v[AI_GLM_5_3__MLA__D];            /* q_a kv_a q_b o */
    if (H == 0u || M == 0u || M > NN__HYPER__MULT_MAX || heads == 0u || QR == 0u || LAT == 0u || NOPE == 0u || VD == 0u || H > (1ull << 24)
     || heads > (1ull << 12) || LAT > (1ull << 16) || to >= (1ull << 32) || !nn__primitives__fits(n * M * H, s_room)
     || !nn__primitives__fits(heads * stride * LAT, room[AI_GLM_5_3__MLA__KVB]) || !nn__primitives__fits(to * LAT, room[AI_GLM_5_3__MLA__CACHE]))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    /* the indexer, when the table carries it */
    const uint64_t table = argv[1].args[0];
    const bool indexed = sys__node_array__length(table) >= AI_GLM_5_3__IDX__TABLE;
    uint64_t IH = 0u, ID = 0u, KP = 1u, TOPK = 0u, id_[4] = {0u, 0u, 0u, 0u};
    if (indexed) {
        sys__node_array_walk w;
        if (!sys__node_array__walk(table, AI_GLM_5_3__MLA__TABLE, &w)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
        for (unsigned i = AI_GLM_5_3__MLA__TABLE; i < AI_GLM_5_3__IDX__PLANES; ++i, sys__node_array__next(&w)) {
            const sys__heap_node* p = sys__node_array__walk_cell(&w);
            if (p == 0 || !nn__primitives__room(p, &at[i], &room[i])) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
        }
        uint64_t iv[AI_GLM_5_3__IDX__TABLE - AI_GLM_5_3__IDX__PLANES];
        for (unsigned i = AI_GLM_5_3__IDX__PLANES; i < AI_GLM_5_3__IDX__TABLE; ++i, sys__node_array__next(&w)) {
            const sys__heap_node* p = sys__node_array__walk_cell(&w);
            if (p == 0 || p->dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
            iv[i - AI_GLM_5_3__IDX__PLANES] = p->args[0];
        }
        IH = iv[0]; ID = iv[1]; KP = iv[2]; TOPK = iv[3];
        for (unsigned i = 0u; i < 4u; ++i) id_[i] = iv[4u + i];
        if (IH == 0u || ID == 0u || KP == 0u || TOPK < KP || IH > 1024u || ID > 4096u
         || !nn__primitives__fits(KP * ID, room[AI_GLM_5_3__IDX__KEYS]) || !nn__primitives__fits(KP * ID, room[AI_GLM_5_3__IDX__GATES])
         || !nn__primitives__fits((to / KP + 1u) * ID, room[AI_GLM_5_3__IDX__POOLED]) || room[AI_GLM_5_3__IDX__SCORES] / 4u < to / KP + 1u
         || room[AI_GLM_5_3__IDX__INDEX] / 4u < TOPK + KP || !nn__primitives__fits((TOPK + KP) * LAT, room[AI_GLM_5_3__IDX__GATHER])
         || !nn__primitives__fits(KP * ID, room[AI_GLM_5_3__IDX__APE]))
            return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_ROOM);
    }
    /* a row's own: its query absorbed into the latent, and its answer in the latent — in the table's scratch */
    if (room[AI_GLM_5_3__MLA__SCRATCH] < 2u * 2u * heads * LAT) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_ROOM);
    const uint64_t qabs = at[AI_GLM_5_3__MLA__SCRATCH], olat = qabs + 2u * heads * LAT;
    uint64_t next = work;
    const uint64_t end = work + w_room;
    const uint64_t pairs = ai_glm_5_3__rows__zzprivate_take(&next, end, 24u * n);
    const uint64_t mixes = ai_glm_5_3__rows__zzprivate_take(&next, end, AI_GLM_5_3__CARRY_MIX * n);
    const uint64_t xin = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t h = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t hr = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t qa = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * (QR + LAT));   /* q_a's rows, then kv_a's */
    const uint64_t qn = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * QR);
    const uint64_t qnr = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * QR);
    const uint64_t ckn = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * LAT);
    const uint64_t q = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * heads * NOPE);
    const uint64_t vh = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * heads * VD);
    const uint64_t vhr = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * heads * VD);
    const uint64_t iq = indexed ? ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * IH * ID) : 1u;
    const uint64_t ik = indexed ? ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * ID) : 1u;
    const uint64_t iw = indexed ? ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * IH) : 1u;
    const uint64_t ig = indexed ? ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * ID) : 1u;
    const uint64_t y = xin;
    if (pairs == 0u || mixes == 0u || xin == 0u || h == 0u || hr == 0u || qa == 0u || qn == 0u || qnr == 0u || ckn == 0u || q == 0u
     || vh == 0u || vhr == 0u || iq == 0u || ik == 0u || iw == 0u || ig == 0u)
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_ROOM);
    const uint64_t kv = qa + 2u * n * QR;
    const uint64_t cache = at[AI_GLM_5_3__MLA__CACHE], kvb = at[AI_GLM_5_3__MLA__KVB], signs = at[AI_GLM_5_3__MLA__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_glm_5_3__rows__zzprivate_pairs(ctx, pairs, n)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    ai_glm_5_3__rows__zzprivate_collapse(doors, over, at, mixes, streams, xin, h, H, M, n, v[AI_GLM_5_3__MLA__ITERS],
                                         r[AI_GLM_5_3__MLA__EPS], r[AI_GLM_5_3__MLA__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)hr, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, n * H, over);
    nn__expert__groups g = {};
    ai_glm_5_3__rows__zzprivate_group(&g, 0u, at, AI_GLM_5_3__MLA__QA, QR, d[0], 0u, 0u, n);
    ai_glm_5_3__rows__zzprivate_group(&g, 1u, at, AI_GLM_5_3__MLA__KVA, LAT, d[1], n * QR, 0u, n);
    g.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)qa, g, (const uint16_t*)(uintptr_t)hr, (const uint32_t*)(uintptr_t)pairs, H, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qa, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__MLA__QA_NORM],
                        QR, n, QR, QR, r[AI_GLM_5_3__MLA__EPS], over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)qnr, (const uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)signs, n * QR, over);
    ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, AI_GLM_5_3__MLA__QB, q, qnr, pairs, n, d[2], heads * NOPE, QR);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)ckn, (const uint16_t*)(uintptr_t)kv, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__MLA__KVA_NORM],
                        LAT, n, LAT, LAT, r[AI_GLM_5_3__MLA__EPS], over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(cache + 2u * first * LAT), (const uint16_t*)(uintptr_t)ckn,
                           (const uint16_t*)(uintptr_t)signs, n * LAT, over);
    if (indexed) {
        ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, AI_GLM_5_3__IDX__WQB, iq, qnr, pairs, n, id_[0], IH * ID, QR);
        ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, AI_GLM_5_3__IDX__WK, ik, hr, pairs, n, id_[1], ID, H);
        ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, AI_GLM_5_3__IDX__WP, iw, hr, pairs, n, id_[2], IH, H);
        ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, AI_GLM_5_3__IDX__GATE, ig, hr, pairs, n, id_[3], ID, H);
    }
    /* ⭐ THE ROWS THAT ATTEND EVERY POSITION BEFORE THEM, AND THE ROWS THAT DO NOT. Up to the indexer's `topk` positions
     *   a row attends the whole cache up to itself, which the latents already written make a causal attention over
     *   the chunk — a launch of every (row, head) for its scores and one for its mix, a block of rows at a time. Past
     *   them a row attends the positions the indexer chooses from the pools finished before it, so it runs a row at a
     *   time, in order. Whether a row is past them only grows with its position: the first `dense` rows are the one
     *   kind, the rest the other. The indexer's ring and pools are kept for every row, in order, either way. */
    uint64_t dense = 0u;
    while (dense < n && !(indexed && (first + dense + 1u) / KP > TOPK / KP)) ++dense;
    const uint64_t block = dense < AI_GLM_5_3__MLA__ROWS_BLOCK ? dense : AI_GLM_5_3__MLA__ROWS_BLOCK;
    const uint64_t qabs_rows = block == 0u ? 1u : ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * block * heads * LAT);
    const uint64_t olat_rows = block == 0u ? 1u : ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * block * heads * LAT);
    const uint64_t p_rows = block == 0u ? 1u : ai_glm_5_3__rows__zzprivate_take(&next, end, 4u * block * heads * (first + dense));
    if (qabs_rows == 0u || olat_rows == 0u || p_rows == 0u) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_ROOM);
    const uint64_t keys = indexed ? at[AI_GLM_5_3__IDX__KEYS] : 0u, gates = indexed ? at[AI_GLM_5_3__IDX__GATES] : 0u;
    const uint64_t pooled = indexed ? at[AI_GLM_5_3__IDX__POOLED] : 0u;
    for (uint64_t t = 0u; t < dense && indexed; ++t) {
        /* ⛳ A POSITION'S KEY AND GATE ARE READ ONLY TO FINISH ITS POOL, so they are kept in a ring of `kpool` rows */
        const uint64_t pos = first + t, len = pos + 1u, ring = 2u * (pos % KP) * ID;
        doors->vector_copy((uint16_t*)(uintptr_t)(gates + ring), (const uint16_t*)(uintptr_t)(ig + 2u * t * ID), ID);
        doors->index_layernorm((uint16_t*)(uintptr_t)(keys + ring), (const uint16_t*)(uintptr_t)(ik + 2u * t * ID),
                               (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__IDX__KNORM_W], (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__IDX__KNORM_B],
                               ID, 1e-6f, over);
        if (len % KP == 0u)
            doors->index_pool((uint16_t*)(uintptr_t)(pooled + 2u * (len / KP - 1u) * ID), (const uint16_t*)(uintptr_t)keys,
                              (const uint16_t*)(uintptr_t)gates, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__IDX__APE], 0u, 1u, ID, KP, over);
    }
    for (uint64_t b0 = 0u; b0 < dense; b0 += block) {
        const uint64_t rows = dense - b0 < block ? dense - b0 : block;
        /* ⛳ the attention's own scale is 1/√latent and the model's 1/√head, so the absorbed query carries their ratio */
        for (uint64_t r = 0u; r < rows; ++r)
            doors->attention_absorb((uint16_t*)(uintptr_t)(qabs_rows + 2u * r * heads * LAT),
                                    (const uint16_t*)(uintptr_t)(q + 2u * (b0 + r) * heads * NOPE), (const uint16_t*)(uintptr_t)kvb,
                                    heads, NOPE, LAT, stride, sqrtf((float)LAT / (float)NOPE), over);
        doors->attention_causal_scores_grouped((float*)(uintptr_t)p_rows, (const uint16_t*)(uintptr_t)qabs_rows, (const uint16_t*)(uintptr_t)cache,
                                                       heads, 1u, LAT, first + b0, rows);
        doors->attention_causal_softmax((float*)(uintptr_t)p_rows, heads, first + b0, rows);
        doors->attention_causal_mix((uint16_t*)(uintptr_t)olat_rows, (const float*)(uintptr_t)p_rows, (const uint16_t*)(uintptr_t)cache,
                                    heads, 1u, LAT, first + b0, rows, over);
        for (uint64_t r = 0u; r < rows; ++r)
            doors->attention_expand((uint16_t*)(uintptr_t)(vh + 2u * (b0 + r) * heads * VD), (const uint16_t*)(uintptr_t)(olat_rows + 2u * r * heads * LAT),
                                    (const uint16_t*)(uintptr_t)kvb, heads, VD, LAT, stride, NOPE, over);
    }
    for (uint64_t t = dense; t < n; ++t) {
        const uint64_t pos = first + t, len = pos + 1u;
        const uint64_t pools = len / KP, chosen_pools = indexed ? TOPK / KP : 0u, tail = len - pools * KP;
        const bool sparse = indexed && pools > chosen_pools;
        const uint64_t alen = sparse ? chosen_pools * KP + tail : len;
        if (room[AI_GLM_5_3__MLA__SCORES] / 4u < heads * alen) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_ROOM);
        uint64_t src = cache;
        if (indexed) {
            /* ⛳ A POSITION'S KEY AND GATE ARE READ ONLY TO FINISH ITS POOL, so they are kept in a ring of `kpool` rows */
            const uint64_t ring = 2u * (pos % KP) * ID;
            doors->vector_copy((uint16_t*)(uintptr_t)(gates + ring), (const uint16_t*)(uintptr_t)(ig + 2u * t * ID), ID);
            doors->index_layernorm((uint16_t*)(uintptr_t)(keys + ring), (const uint16_t*)(uintptr_t)(ik + 2u * t * ID),
                                   (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__IDX__KNORM_W], (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__IDX__KNORM_B],
                                   ID, 1e-6f, over);
            if (tail == 0u)
                doors->index_pool((uint16_t*)(uintptr_t)(pooled + 2u * (pools - 1u) * ID), (const uint16_t*)(uintptr_t)keys,
                                  (const uint16_t*)(uintptr_t)gates, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__IDX__APE], 0u, 1u, ID, KP, over);
            if (sparse) {
                const uint64_t index = at[AI_GLM_5_3__IDX__INDEX];
                doors->index_scores((float*)(uintptr_t)at[AI_GLM_5_3__IDX__SCORES], (const uint16_t*)(uintptr_t)(iq + 2u * t * IH * ID),
                                    (const uint16_t*)(uintptr_t)(iw + 2u * t * IH), (const uint16_t*)(uintptr_t)pooled, IH, ID, pools,
                                    1.0f / sqrtf((float)ID));
                doors->index_select((uint32_t*)(uintptr_t)index, (uint32_t*)(uintptr_t)(index + 4u * (TOPK + KP)),
                                    (const float*)(uintptr_t)at[AI_GLM_5_3__IDX__SCORES], pools, chosen_pools, KP, pools * KP, tail);
                doors->index_gather((uint16_t*)(uintptr_t)at[AI_GLM_5_3__IDX__GATHER], (const uint16_t*)(uintptr_t)cache,
                                    (const uint32_t*)(uintptr_t)index, alen, LAT);
                src = at[AI_GLM_5_3__IDX__GATHER];
            }
        }
        /* ⛳ the attention's own scale is 1/√latent and the model's 1/√head, so the absorbed query carries their ratio */
        doors->attention_absorb((uint16_t*)(uintptr_t)qabs, (const uint16_t*)(uintptr_t)(q + 2u * t * heads * NOPE),
                                (const uint16_t*)(uintptr_t)kvb, heads, NOPE, LAT, stride, sqrtf((float)LAT / (float)NOPE), over);
        doors->attention_scores_grouped((float*)(uintptr_t)at[AI_GLM_5_3__MLA__SCORES], (const uint16_t*)(uintptr_t)qabs,
                                        (const uint16_t*)(uintptr_t)src, heads, 1u, LAT, alen);
        doors->attention_softmax_rows((float*)(uintptr_t)at[AI_GLM_5_3__MLA__SCORES], heads, alen);
        doors->attention_mix((uint16_t*)(uintptr_t)olat, (const float*)(uintptr_t)at[AI_GLM_5_3__MLA__SCORES],
                             (const uint16_t*)(uintptr_t)src, heads, 1u, LAT, alen, over);
        doors->attention_expand((uint16_t*)(uintptr_t)(vh + 2u * t * heads * VD), (const uint16_t*)(uintptr_t)olat,
                                (const uint16_t*)(uintptr_t)kvb, heads, VD, LAT, stride, NOPE, over);
    }
    doors->hadamard_rotate((uint16_t*)(uintptr_t)vhr, (const uint16_t*)(uintptr_t)vh, (const uint16_t*)(uintptr_t)signs, n * heads * VD, over);
    ai_glm_5_3__rows__zzprivate_gemm(doors, over, at, room, AI_GLM_5_3__MLA__O, y, vhr, pairs, n, d[3], H, heads * VD);
    ai_glm_5_3__rows__zzprivate_place(doors, over, mixes, streams, y, H, M, n);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_glm_5_3__mlp_rows streams table n work)` -> `streams`. The dense MLP site over `n` rows. */
static sys__heap_node ai_glm_5_3__mlp_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, n = 0, work = 0, w_room = 0;
    if (!ai_glm_5_3__rows__zzprivate_open(argv, 2u, 3u, &streams, &s_room, &n, &work, &w_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__MLP__TABLE], room[AI_GLM_5_3__MLP__TABLE], v[AI_GLM_5_3__MLP__TABLE];
    float r[AI_GLM_5_3__MLP__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[1].args[0], AI_GLM_5_3__MLP__PLANES, AI_GLM_5_3__MLP__EPS, AI_GLM_5_3__MLP__TABLE,
                                           at, room, v, r))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__MLP__HIDDEN], M = v[AI_GLM_5_3__MLP__MULT], I = v[AI_GLM_5_3__MLP__INTER];
    if (H == 0u || M == 0u || M > NN__HYPER__MULT_MAX || I == 0u || H > (1ull << 24) || I > (1ull << 24) || !nn__primitives__fits(n * M * H, s_room))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    uint64_t next = work;
    const uint64_t end = work + w_room;
    const uint64_t pairs = ai_glm_5_3__rows__zzprivate_take(&next, end, 24u * n);
    const uint64_t mixes = ai_glm_5_3__rows__zzprivate_take(&next, end, AI_GLM_5_3__CARRY_MIX * n);
    const uint64_t xin = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t h = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t hr = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t gu = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * 2u * I);
    const uint64_t act = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * I);
    const uint64_t actr = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * I);
    const uint64_t y = xin;
    if (pairs == 0u || mixes == 0u || xin == 0u || h == 0u || hr == 0u || gu == 0u || act == 0u || actr == 0u)
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_ROOM);
    const uint64_t signs = at[AI_GLM_5_3__MLP__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_glm_5_3__rows__zzprivate_pairs(ctx, pairs, n)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    ai_glm_5_3__rows__zzprivate_collapse(doors, over, at, mixes, streams, xin, h, H, M, n, v[AI_GLM_5_3__MLP__ITERS],
                                         r[AI_GLM_5_3__MLP__EPS], r[AI_GLM_5_3__MLP__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)hr, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, n * H, over);
    ai_glm_5_3__rows__zzprivate_swiglu(doors, over, at, room, AI_GLM_5_3__MLP__GATE, AI_GLM_5_3__MLP__UP, AI_GLM_5_3__MLP__DOWN,
                                       &v[AI_GLM_5_3__MLP__D], hr, pairs, gu, act, actr, signs, y, H, I, n, r[AI_GLM_5_3__MLP__LIMIT]);
    ai_glm_5_3__rows__zzprivate_place(doors, over, mixes, streams, y, H, M, n);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_glm_5_3__route_rows streams table hands carries n work)` -> `hands`. The MoE site to the router for `n` rows:
 * every row's input collapsed, normed and rotated into its hand row with the router's weights and picks after it, and
 * the shared expert over every row. `carries` keeps what each row's close needs — its weights of the collapse, a record
 * of `AI_GLM_5_3__CARRY_MIX` bytes a row, then the shared expert's rows — so the rows are all routed before any is closed. */
static sys__heap_node ai_glm_5_3__route_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, hands = 0, h_room = 0, carries = 0, c_room = 0, n = 0u, work = 0, w_room = 0;
    if (!ai_glm_5_3__rows__zzprivate_open(argv, 4u, 5u, &streams, &s_room, &n, &work, &w_room)
     || !nn__primitives__room(&argv[2], &hands, &h_room) || !nn__primitives__room(&argv[3], &carries, &c_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__MOE__TABLE], room[AI_GLM_5_3__MOE__TABLE], v[AI_GLM_5_3__MOE__TABLE];
    float r[AI_GLM_5_3__MOE__TABLE];
    if (!ai_glm_5_3__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__MOE__HIDDEN], M = v[AI_GLM_5_3__MOE__MULT], E = v[AI_GLM_5_3__MOE__EXPERTS];
    const uint64_t K = v[AI_GLM_5_3__MOE__TOP_K], I = v[AI_GLM_5_3__MOE__INTER];
    const uint64_t hand_bytes = AI_GLM_5_3__HAND(H, K);
    if (K > AI_GLM_5_3__TOP_K_MAX || !nn__primitives__fits(n * M * H, s_room) || h_room / hand_bytes < n
     || c_room / AI_GLM_5_3__CARRY(H) < n || !nn__primitives__fits(E, room[AI_GLM_5_3__MOE__BIAS])
     || !nn__primitives__fits(E * H, room[AI_GLM_5_3__MOE__ROUTER]))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    uint64_t next = work;
    const uint64_t end = work + w_room;
    const uint64_t pairs = ai_glm_5_3__rows__zzprivate_take(&next, end, 24u * n);
    const uint64_t xin = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t h = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t hr = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * H);
    const uint64_t gu = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * 2u * I);
    const uint64_t act = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * I);
    const uint64_t actr = ai_glm_5_3__rows__zzprivate_take(&next, end, 2u * n * I);
    if (pairs == 0u || xin == 0u || h == 0u || hr == 0u || gu == 0u || act == 0u || actr == 0u)
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_ROOM);
    const uint64_t mixes = carries, shared = carries + AI_GLM_5_3__CARRY_MIX * n;
    const uint64_t logits = at[AI_GLM_5_3__MOE__SCRATCH], signs = at[AI_GLM_5_3__MOE__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_glm_5_3__rows__zzprivate_pairs(ctx, pairs, n)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    ai_glm_5_3__rows__zzprivate_collapse(doors, over, at, mixes, streams, xin, h, H, M, n, v[AI_GLM_5_3__MOE__ITERS],
                                         r[AI_GLM_5_3__MOE__EPS], r[AI_GLM_5_3__MOE__HC_EPS]);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)hr, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, n * H, over);
    for (uint64_t t = 0u; t < n; ++t) {
        const uint64_t hand = hands + t * hand_bytes;
        doors->vector_copy((uint16_t*)(uintptr_t)hand, (const uint16_t*)(uintptr_t)(hr + 2u * t * H), H);
        doors->expert_multiply_fp16((uint16_t*)(uintptr_t)logits, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__MOE__ROUTER],
                                    (const uint16_t*)(uintptr_t)(h + 2u * t * H), E, H, over);
        doors->vector_top_k_biased((uint64_t*)(uintptr_t)(hand + AI_GLM_5_3__HAND_PICKS(H, K)), 1ull, (uint16_t*)(uintptr_t)(hand + 2u * H),
                                   (const uint16_t*)(uintptr_t)logits, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__MOE__BIAS], E, K,
                                   r[AI_GLM_5_3__MOE__SCALE]);
    }
    ai_glm_5_3__rows__zzprivate_swiglu(doors, over, at, room, AI_GLM_5_3__MOE__GATE, AI_GLM_5_3__MOE__UP, AI_GLM_5_3__MOE__DOWN,
                                       &v[AI_GLM_5_3__MOE__D], hr, pairs, gu, act, actr, signs, shared, H, I, n, r[AI_GLM_5_3__MOE__LIMIT]);
    return nn__doors_answer(&argv[2]);
}

/* `(ai_glm_5_3__close_rows streams table routed carries n work)` -> `streams`. Every row's routed sum (a row of `routed`)
 * added to its shared expert's, and its streams mixed again by the weights its route kept. */
static sys__heap_node ai_glm_5_3__close_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t streams = 0, s_room = 0, routed = 0, r_room = 0, carries = 0, c_room = 0, n = 0u, work = 0, w_room = 0;
    if (!ai_glm_5_3__rows__zzprivate_open(argv, 4u, 5u, &streams, &s_room, &n, &work, &w_room)
     || !nn__primitives__room(&argv[2], &routed, &r_room) || !nn__primitives__room(&argv[3], &carries, &c_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__MOE__TABLE], room[AI_GLM_5_3__MOE__TABLE], v[AI_GLM_5_3__MOE__TABLE];
    float r[AI_GLM_5_3__MOE__TABLE];
    if (!ai_glm_5_3__layer__zzprivate_moe(argv[1].args[0], at, room, v, r)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__MOE__HIDDEN], M = v[AI_GLM_5_3__MOE__MULT];
    if (!nn__primitives__fits(n * M * H, s_room) || !nn__primitives__fits(n * H, r_room) || c_room / AI_GLM_5_3__CARRY(H) < n
     || !nn__primitives__fits(n * H, w_room))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t y = work;
    doors->vector_add((uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)(carries + AI_GLM_5_3__CARRY_MIX * n),
                      (const uint16_t*)(uintptr_t)routed, n * H, ctx->fault_word);
    ai_glm_5_3__rows__zzprivate_place(doors, ctx->fault_word, carries, streams, y, H, M, n);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_glm_5_3__experts_rows hands table routed n [nn__expert_tier])` -> `routed`: for each of `n` hand rows, Σ w_j · down_j(swiglu(
 * gate_up_j(x))) over its picks this worker holds, a row of `routed` each — as nn runs a chunk's routed experts (▶ nn's
 * `nn__routed__rows`): EXPERT-MAJOR, each held expert the chunk used run once over every row that picked it, each pick's
 * down in a sum of its own and the rows added in pick order, whatever order the experts ran in. */
static sys__heap_node ai_glm_5_3__experts_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hands = 0, h_room = 0, routed = 0, r_room = 0, n = 0u;
    if (!nn__primitives__room(&argv[0], &hands, &h_room) || !nn__primitives__room(&argv[2], &routed, &r_room)
     || !sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !ai_glm_5_3__layer__zzprivate_rows(&argv[3], &n))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__EXP__TABLE], room[AI_GLM_5_3__EXP__TABLE], v[AI_GLM_5_3__EXP__TABLE];
    float r[AI_GLM_5_3__EXP__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[1].args[0], AI_GLM_5_3__EXP__PLANES, AI_GLM_5_3__EXP__LIMIT, AI_GLM_5_3__EXP__TABLE,
                                           at, room, v, r))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__EXP__HIDDEN], K = v[AI_GLM_5_3__EXP__TOP_K];
    if (K > AI_GLM_5_3__TOP_K_MAX || v[AI_GLM_5_3__EXP__INT8] > 1u || h_room / AI_GLM_5_3__HAND(H, K) < n
     || !nn__primitives__fits(n * H, r_room))
        return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    nn__routed rt;
    if (!ai_glm_5_3__experts__zzprivate_routed(argv[1].args[0], at, room, v, r, &rt)) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    nn__expert_tier tier;
    bool tier_ok = true;
    const bool holds = ai_glm_5_3__experts__zzprivate_tier(argv, argc, 4u, rt.layer, &tier, &tier_ok);
    if (!tier_ok) return sys__engine__abi__error(AI_GLM_5_3__LAYER__FAULT_TABLE);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t f = nn__routed__rows(ctx, doors, &rt, hands, n, holds ? &tier : 0, routed, 0, 0);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__kda__zzabi_adapter,     ai_glm_5_3__kda__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__mla__zzabi_adapter,     ai_glm_5_3__mla__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__mlp__zzabi_adapter,     ai_glm_5_3__mlp__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__route__zzabi_adapter,   ai_glm_5_3__route__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__shared__zzabi_adapter,  ai_glm_5_3__shared__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__close__zzabi_adapter,   ai_glm_5_3__close__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__experts__zzabi_adapter, ai_glm_5_3__experts__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__kda_rows__zzabi_adapter,     ai_glm_5_3__kda_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__mla_rows__zzabi_adapter,     ai_glm_5_3__mla_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__mlp_rows__zzabi_adapter,     ai_glm_5_3__mlp_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__route_rows__zzabi_adapter,   ai_glm_5_3__route_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__close_rows__zzabi_adapter,   ai_glm_5_3__close_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__experts_rows__zzabi_adapter, ai_glm_5_3__experts_rows__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_GLM_5_3_CPU_OPCODES_LAYER__ABI_CUH */
