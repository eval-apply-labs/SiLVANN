#ifndef SILVANN__PACKAGES_AI_GLM_5_3_CPU_OPCODES_VISION__ABI_CUH
#define SILVANN__PACKAGES_AI_GLM_5_3_CPU_OPCODES_VISION__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../contracts/objects/vision.cuh"    /* the tower's tables */

/* ⭐⭐ A PICTURE THROUGH GLM 5.3'S VISION TOWER, ON A WORKER — three verbs of nn's doors (▶ `contracts/objects/vision.cuh`
 * for their tables): the patch embedding, a block, and the tail (the last norm, the downsampling and the merger). The
 * arithmetic is Hugging Face's (`Glm5NextVisionModel`), as `python/silvann_runtime/vision_glm.py` writes it out in numpy:
 *
 *   embed    x = W·pixels + b                               (W a row of 1176 halves an output: decoded at the boot)
 *   block    h = RMS(x); q, k, v = W·h + b; q, k RMSNormed a head; q, k turned by the patch's row and column;
 *            a = every patch to every patch; x += W·a + b;
 *            h = RMS(x); x += W_d·(silu(min(W_g·h + b_g, L)) · clamp(W_u·h + b_u, ±L)) + b_d
 *   tail     h = RMS(x) a patch; four patches as one row; y = W·h + b (16 bits); y = gelu_erf(LN(W_p·y));
 *            out = W_d·(silu(min(W_g·y, L)) · clamp(W_u·y, ±L))
 *
 * A matrix's codes below 16 bits were rotated row by row in the pack's ladder, so the rows it multiplies are rotated the
 * same way first (`hadamard_ladder_rows`); at 16 bits its rows go in as they are. Every product is nn's `expert_rows`
 * (or `expert_groups`, gate and up together) over the pairs at the front of the scratch: `(t, t)` a row, then `(t, 2t)`
 * and `(t, 2t + 1)`, which leave a row's gate answer and its up answer side by side for the clamped SwiGLU. */

/* The three sets of pairs for `count` rows at `at`, written from the host. False if the card would not take them. */
static bool ai_glm_5_3__vision__zzprivate_pairs(sys__engine__ctx* ctx, uint64_t at, uint64_t count) {
    uint32_t* p = (uint32_t*)malloc(sizeof(uint32_t) * 6u * count);
    if (p == 0) return false;
    for (uint64_t t = 0u; t < count; ++t) {
        p[2u * t] = (uint32_t)t;                       p[2u * t + 1u] = (uint32_t)t;
        p[2u * (count + t)] = (uint32_t)t;             p[2u * (count + t) + 1u] = (uint32_t)(2u * t);
        p[2u * (2u * count + t)] = (uint32_t)t;        p[2u * (2u * count + t) + 1u] = (uint32_t)(2u * t + 1u);
    }
    const bool ok = sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)at, p, (size_t)(24u * count));
    free(p);
    return ok;
}

/* The bytes the pairs of `rows` rows take at the front of a scratch, rounded to 256. */
static uint64_t ai_glm_5_3__vision__zzprivate_pairs_bytes(uint64_t rows) {
    return (24u * rows + 255u) & ~255ull;
}

/* Whether a matrix of `out_rows` rows of `cols` at width `d` fits its planes — its codes, and its scales below 16 bits —
 * and a rotated one's rows are whole rotation blocks. */
static bool ai_glm_5_3__vision__zzprivate_holds(uint64_t codes_room, uint64_t lut_room, uint64_t d, uint64_t out_rows,
                                               uint64_t cols) {
    if (d != 4u && d != 8u && d != 16u) return false;
    if (d != 16u && cols % 128u != 0u) return false;
    return out_rows * nn__turboquant__row_bytes(d, cols) <= codes_room && (d == 16u || 2u * out_rows <= lut_room);
}

/* `out = W·x (+ b)` over `rows` rows of `cols`, every operand a card address: the rows rotated into `rot` first where W's
 * codes are rotated. `bias` 0: none. The `(t, t)` pairs are at `pairs`. */
static void ai_glm_5_3__vision__zzprivate_linear(const nn__doors* doors, uint64_t out, const uint64_t* at, const uint64_t* room,
                                                 unsigned codes, uint64_t d, uint64_t x, uint64_t rot, uint64_t signs, uint64_t pairs,
                                                 uint64_t rows, uint64_t out_rows, uint64_t cols, uint64_t bias, unsigned int* over) {
    uint64_t in = x;
    if (d != 16u) {
        doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)rot, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)signs,
                                    cols, rows, over);
        in = rot;
    }
    doors->expert_rows((uint16_t*)(uintptr_t)out, (const uint8_t*)(uintptr_t)at[codes], room[codes],
                       (const uint8_t*)(uintptr_t)at[codes + 1u], room[codes + 1u], (const uint16_t*)(uintptr_t)in,
                       (const uint32_t*)(uintptr_t)pairs, rows, d, out_rows, cols, over);
    if (bias != 0u)
        doors->vector_bias_rows((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)bias,
                                out_rows, rows, over);
}

/* The clamped SwiGLU over `n` rows of `x` (width `cols`) into `out` (width `outw`): gate and up in one grouped launch on
 * the rows (rotated into `rot` where their codes are), side by side a row in `gu` with their bias `gu_b` (0: none),
 * clamped into `act`, rotated into `actr` where down's codes are, then down and its bias `down_b` (0: none). The pairs'
 * three sets are `count` long at `pairs`; the rows are their first `n`. */
static void ai_glm_5_3__vision__zzprivate_swiglu(const nn__doors* doors, const uint64_t* at, const uint64_t* room,
                                                 unsigned gate, unsigned up, unsigned down, uint64_t d_gu, uint64_t d_down,
                                                 uint64_t x, uint64_t rot, uint64_t gu, uint64_t act, uint64_t actr, uint64_t out,
                                                 uint64_t gu_b, uint64_t down_b, uint64_t signs, uint64_t pairs, uint64_t count,
                                                 uint64_t n, uint64_t cols, uint64_t I, uint64_t outw, float limit,
                                                 unsigned int* over) {
    uint64_t in = x;
    if (d_gu != 16u) {
        doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)rot, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)signs,
                                    cols, n, over);
        in = rot;
    }
    nn__expert__groups g = {};
    g.codes[0] = at[gate]; g.luts[0] = at[gate + 1u]; g.out_rows[0] = I; g.d[0] = d_gu; g.out_at[0] = 0u;
    g.pairs_at[0] = count; g.pairs[0] = n;
    g.codes[1] = at[up]; g.luts[1] = at[up + 1u]; g.out_rows[1] = I; g.d[1] = d_gu; g.out_at[1] = 0u;
    g.pairs_at[1] = 2u * count; g.pairs[1] = n;
    g.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)gu, g, (const uint16_t*)(uintptr_t)in, (const uint32_t*)(uintptr_t)pairs, cols, over);
    if (gu_b != 0u)
        doors->vector_bias_rows((uint16_t*)(uintptr_t)gu, (const uint16_t*)(uintptr_t)gu, (const uint16_t*)(uintptr_t)gu_b,
                                2u * I, n, over);
    doors->swiglu_clamped((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, n, I, limit, over);
    ai_glm_5_3__vision__zzprivate_linear(doors, out, at, room, down, d_down, act, actr, signs, pairs, n, outw, I, down_b, over);
}

/* `(ai_glm_5_3__vision_embed pixels table rows x)` -> `x`. */
static sys__heap_node ai_glm_5_3__vision_embed__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t px = 0, px_room = 0, x = 0, x_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &px, &px_room)
     || !nn__primitives__room(&argv[3], &x, &x_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__VE__TABLE], room[AI_GLM_5_3__VE__TABLE], v[AI_GLM_5_3__VE__TABLE];
    float r[AI_GLM_5_3__VE__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[1].args[0], AI_GLM_5_3__VE__PLANES, AI_GLM_5_3__VE__TABLE, AI_GLM_5_3__VE__TABLE,
                                           at, room, v, r))
        return sys__engine__abi__error(AI_GLM_5_3__VISION__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__VE__HIDDEN], IN = v[AI_GLM_5_3__VE__IN], d = v[AI_GLM_5_3__VE__D], rows = argv[2].args[0];
    if (H == 0u || IN == 0u || IN % 2u != 0u || rows == 0u || rows > (1ull << 20) || d != 16u
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VE__PATCH], room[AI_GLM_5_3__VE__PATCH_LUT], d, H, IN)
     || room[AI_GLM_5_3__VE__PATCH_B] < 2u * H)
        return sys__engine__abi__error(AI_GLM_5_3__VISION__FAULT_TABLE);
    if (!nn__primitives__fits(rows * IN, px_room) || !nn__primitives__fits(rows * H, x_room)
     || room[AI_GLM_5_3__VE__SCRATCH] < ai_glm_5_3__vision__zzprivate_pairs_bytes(rows))
        return sys__engine__abi__error(AI_GLM_5_3__VISION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = at[AI_GLM_5_3__VE__SCRATCH];
    if (!ai_glm_5_3__vision__zzprivate_pairs(ctx, pairs, rows)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    ai_glm_5_3__vision__zzprivate_linear(doors, x, at, room, AI_GLM_5_3__VE__PATCH, d, px, 0u, 0u, pairs, rows, H, IN,
                                         at[AI_GLM_5_3__VE__PATCH_B], ctx->fault_word);
    return nn__doors_answer(&argv[3]);
}

/* `(ai_glm_5_3__vision_block x positions table rows)` -> `x`. */
static sys__heap_node ai_glm_5_3__vision_block__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, ps = 0, ps_room = 0;
    if (!sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || argv[3].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &x, &x_room)
     || !nn__primitives__room(&argv[1], &ps, &ps_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__VB__TABLE], room[AI_GLM_5_3__VB__TABLE], v[AI_GLM_5_3__VB__TABLE];
    float r[AI_GLM_5_3__VB__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[2].args[0], AI_GLM_5_3__VB__PLANES, AI_GLM_5_3__VB__EPS, AI_GLM_5_3__VB__TABLE,
                                           at, room, v, r))
        return sys__engine__abi__error(AI_GLM_5_3__VISION__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__VB__HIDDEN], heads = v[AI_GLM_5_3__VB__HEADS], hd = v[AI_GLM_5_3__VB__HEAD_DIM];
    const uint64_t I = v[AI_GLM_5_3__VB__INTER], theta = v[AI_GLM_5_3__VB__THETA], chunk = v[AI_GLM_5_3__VB__CHUNK];
    const uint64_t d_gu = v[AI_GLM_5_3__VB__D_GATE], d_down = v[AI_GLM_5_3__VB__D_DOWN], rows = argv[3].args[0];
    const float eps = r[AI_GLM_5_3__VB__EPS], limit = r[AI_GLM_5_3__VB__LIMIT];
    if (H == 0u || heads == 0u || hd == 0u || heads * hd != H || hd > NN__ATTENTION__SET_HEAD_MAX || hd % 4u != 0u
     || I == 0u || I > (1ull << 20) || theta == 0u || theta >= (1ull << 32) || chunk == 0u || rows == 0u || rows > (1ull << 20)
     || !(eps > 0.0f) || !(limit > 0.0f)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VB__QKV], room[AI_GLM_5_3__VB__QKV_LUT], v[AI_GLM_5_3__VB__D_QKV], 3u * H, H)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VB__PROJ], room[AI_GLM_5_3__VB__PROJ_LUT], v[AI_GLM_5_3__VB__D_PROJ], H, H)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VB__GATE], room[AI_GLM_5_3__VB__GATE_LUT], d_gu, I, H)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VB__UP], room[AI_GLM_5_3__VB__UP_LUT], d_gu, I, H)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VB__DOWN], room[AI_GLM_5_3__VB__DOWN_LUT], d_down, H, I)
     || room[AI_GLM_5_3__VB__QKV_B] < 6u * H || room[AI_GLM_5_3__VB__PROJ_B] < 2u * H || room[AI_GLM_5_3__VB__GU_B] < 4u * I
     || room[AI_GLM_5_3__VB__DOWN_B] < 2u * H || room[AI_GLM_5_3__VB__NORM1_W] < 2u * H || room[AI_GLM_5_3__VB__NORM2_W] < 2u * H
     || room[AI_GLM_5_3__VB__Q_NORM] < 2u * hd || room[AI_GLM_5_3__VB__K_NORM] < 2u * hd || room[AI_GLM_5_3__VB__SIGNS] < 1024u)
        return sys__engine__abi__error(AI_GLM_5_3__VISION__FAULT_TABLE);
    const uint64_t step = chunk < rows ? chunk : rows;
    /* hrot takes the rows of `h` rotated, or a step's rows of the MLP's width */
    const uint64_t rotn = rows * H > step * I ? rows * H : step * I;
    const uint64_t P = ai_glm_5_3__vision__zzprivate_pairs_bytes(rows);
    const uint64_t h = at[AI_GLM_5_3__VB__SCRATCH] + P, hrot = h + 2u * rows * H, qkv = hrot + 2u * rotn;
    const uint64_t gu = qkv + 6u * rows * H, act = gu + 4u * step * I, actr = act + 2u * step * I, m = actr + 2u * step * I;
    const uint64_t cs = m + 2u * step * H, end = cs + 4u * rows * hd;
    if (!nn__primitives__fits(rows * H, x_room) || ps_room < 8u * rows || room[AI_GLM_5_3__VB__SCRATCH] < end - at[AI_GLM_5_3__VB__SCRATCH])
        return sys__engine__abi__error(AI_GLM_5_3__VISION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = at[AI_GLM_5_3__VB__SCRATCH];
    if (!ai_glm_5_3__vision__zzprivate_pairs(ctx, pairs, rows)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t signs = at[AI_GLM_5_3__VB__SIGNS];

    /* ① the attention: q, k, v for every patch, q and k normed a head and turned by the patch's row and column, every
     *   patch to every patch */
    doors->rope_angles_axial_rows((float*)(uintptr_t)cs, (const uint32_t*)(uintptr_t)ps, (float)(uint32_t)theta, hd, rows);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__VB__NORM1_W],
                        H, rows, H, H, eps, over);
    ai_glm_5_3__vision__zzprivate_linear(doors, qkv, at, room, AI_GLM_5_3__VB__QKV, v[AI_GLM_5_3__VB__D_QKV], h, hrot, signs, pairs,
                                         rows, 3u * H, H, at[AI_GLM_5_3__VB__QKV_B], over);
    /* a head's q of every patch is a column of rows 3H apart: a launch a head, and one for its k */
    for (uint64_t e = 0u; e < heads; ++e) {
        const uint64_t q = qkv + 2u * e * hd, k = qkv + 2u * (H + e * hd);
        doors->rmsnorm_rows((uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__VB__Q_NORM],
                            hd, rows, 3u * H, 3u * H, eps, over);
        doors->rmsnorm_rows((uint16_t*)(uintptr_t)k, (const uint16_t*)(uintptr_t)k, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__VB__K_NORM],
                            hd, rows, 3u * H, 3u * H, eps, over);
    }
    doors->rope_rows((uint16_t*)(uintptr_t)qkv, (const uint16_t*)(uintptr_t)qkv, (const float*)(uintptr_t)cs, rows, heads, hd,
                     3u * H, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)(qkv + 2u * H), (const uint16_t*)(uintptr_t)(qkv + 2u * H), (const float*)(uintptr_t)cs,
                     rows, heads, hd, 3u * H, hd, over);
    doors->attention_set((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)qkv, rows, heads, hd, over);
    /* the projection into the front of qkv's room, which the attention has done with, then onto the residual */
    ai_glm_5_3__vision__zzprivate_linear(doors, qkv, at, room, AI_GLM_5_3__VB__PROJ, v[AI_GLM_5_3__VB__D_PROJ], h, hrot, signs, pairs,
                                         rows, H, H, at[AI_GLM_5_3__VB__PROJ_B], over);
    doors->vector_add((uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)qkv, rows * H, over);

    /* ② the MLP, `step` rows at a time */
    for (uint64_t c0 = 0u; c0 < rows; c0 += step) {
        const uint64_t n = rows - c0 < step ? rows - c0 : step;
        const uint64_t xc = x + 2u * c0 * H;
        doors->rmsnorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)xc, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__VB__NORM2_W],
                            H, n, H, H, eps, over);
        ai_glm_5_3__vision__zzprivate_swiglu(doors, at, room, AI_GLM_5_3__VB__GATE, AI_GLM_5_3__VB__UP, AI_GLM_5_3__VB__DOWN, d_gu,
                                             d_down, h, hrot, gu, act, actr, m, at[AI_GLM_5_3__VB__GU_B], at[AI_GLM_5_3__VB__DOWN_B],
                                             signs, pairs, rows, n, H, I, H, limit, over);
        doors->vector_add((uint16_t*)(uintptr_t)xc, (const uint16_t*)(uintptr_t)xc, (const uint16_t*)(uintptr_t)m, n * H, over);
    }
    return nn__doors_answer(&argv[0]);
}

/* `(ai_glm_5_3__vision_tail x table rows out)` -> `out`, a row of the text model's width a `MERGE` patches. */
static sys__heap_node ai_glm_5_3__vision_tail__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, out = 0, out_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &x, &x_room)
     || !nn__primitives__room(&argv[3], &out, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_GLM_5_3__VT__TABLE], room[AI_GLM_5_3__VT__TABLE], v[AI_GLM_5_3__VT__TABLE];
    float r[AI_GLM_5_3__VT__TABLE];
    if (!ai_glm_5_3__table__zzpackage_read(argv[1].args[0], AI_GLM_5_3__VT__PLANES, AI_GLM_5_3__VT__EPS, AI_GLM_5_3__VT__TABLE,
                                           at, room, v, r))
        return sys__engine__abi__error(AI_GLM_5_3__VISION__FAULT_TABLE);
    const uint64_t H = v[AI_GLM_5_3__VT__HIDDEN], M = v[AI_GLM_5_3__VT__MERGE], OUT = v[AI_GLM_5_3__VT__OUT];
    const uint64_t I = v[AI_GLM_5_3__VT__INTER], chunk = v[AI_GLM_5_3__VT__CHUNK], rows = argv[2].args[0];
    const uint64_t d_gu = v[AI_GLM_5_3__VT__D_GATE], d_down = v[AI_GLM_5_3__VT__D_DOWN];
    const uint64_t W = M * H, tokens = M == 0u ? 0u : rows / M;
    const float eps = r[AI_GLM_5_3__VT__EPS], ln_eps = r[AI_GLM_5_3__VT__LN_EPS], limit = r[AI_GLM_5_3__VT__LIMIT];
    if (H == 0u || M == 0u || OUT == 0u || I == 0u || I > (1ull << 20) || chunk == 0u || rows == 0u || rows % M != 0u
     || rows > (1ull << 20) || W % 2u != 0u || v[AI_GLM_5_3__VT__D_DOWNS] != 16u || !(eps > 0.0f) || !(ln_eps > 0.0f) || !(limit > 0.0f)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VT__DOWNS], room[AI_GLM_5_3__VT__DOWNS_LUT], 16u, OUT, W)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VT__PROJ], room[AI_GLM_5_3__VT__PROJ_LUT], v[AI_GLM_5_3__VT__D_PROJ], OUT, OUT)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VT__GATE], room[AI_GLM_5_3__VT__GATE_LUT], d_gu, I, OUT)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VT__UP], room[AI_GLM_5_3__VT__UP_LUT], d_gu, I, OUT)
     || !ai_glm_5_3__vision__zzprivate_holds(room[AI_GLM_5_3__VT__DOWN], room[AI_GLM_5_3__VT__DOWN_LUT], d_down, OUT, I)
     || room[AI_GLM_5_3__VT__NORM_W] < 2u * H || room[AI_GLM_5_3__VT__DOWNS_B] < 2u * OUT || room[AI_GLM_5_3__VT__LN_W] < 2u * OUT
     || room[AI_GLM_5_3__VT__LN_B] < 2u * OUT || room[AI_GLM_5_3__VT__SIGNS] < 1024u)
        return sys__engine__abi__error(AI_GLM_5_3__VISION__FAULT_TABLE);
    const uint64_t step = chunk < tokens ? chunk : tokens;
    const uint64_t big = rows * H > tokens * OUT ? rows * H : tokens * OUT;
    const uint64_t P = ai_glm_5_3__vision__zzprivate_pairs_bytes(tokens);
    const uint64_t h = at[AI_GLM_5_3__VT__SCRATCH] + P, y = h + 2u * big, yrot = y + 2u * tokens * OUT;
    const uint64_t rotn = tokens * OUT > step * I ? tokens * OUT : step * I;
    const uint64_t gu = yrot + 2u * rotn, act = gu + 4u * step * I, actr = act + 2u * step * I, end = actr + 2u * step * I;
    if (!nn__primitives__fits(rows * H, x_room) || !nn__primitives__fits(tokens * OUT, out_room)
     || room[AI_GLM_5_3__VT__SCRATCH] < end - at[AI_GLM_5_3__VT__SCRATCH])
        return sys__engine__abi__error(AI_GLM_5_3__VISION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = at[AI_GLM_5_3__VT__SCRATCH];
    if (!ai_glm_5_3__vision__zzprivate_pairs(ctx, pairs, tokens)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t signs = at[AI_GLM_5_3__VT__SIGNS];
    /* the last norm a patch; `M` patches side by side are one row of `W` — the rows of `h` as they lie — downsampled */
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__VT__NORM_W],
                        H, rows, H, H, eps, over);
    ai_glm_5_3__vision__zzprivate_linear(doors, y, at, room, AI_GLM_5_3__VT__DOWNS, 16u, h, 0u, signs, pairs, tokens, OUT, W,
                                         at[AI_GLM_5_3__VT__DOWNS_B], over);
    /* the merger: the projection into h's room, LayerNorm back into y, GELU, then the clamped SwiGLU a step at a time */
    ai_glm_5_3__vision__zzprivate_linear(doors, h, at, room, AI_GLM_5_3__VT__PROJ, v[AI_GLM_5_3__VT__D_PROJ], y, yrot, signs, pairs,
                                         tokens, OUT, OUT, 0u, over);
    doors->layernorm_rows((uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__VT__LN_W],
                          (const uint16_t*)(uintptr_t)at[AI_GLM_5_3__VT__LN_B], OUT, tokens, OUT, OUT, ln_eps, over);
    doors->vector_gelu((uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)y, tokens * OUT, 1u, over);
    for (uint64_t c0 = 0u; c0 < tokens; c0 += step) {
        const uint64_t n = tokens - c0 < step ? tokens - c0 : step;
        ai_glm_5_3__vision__zzprivate_swiglu(doors, at, room, AI_GLM_5_3__VT__GATE, AI_GLM_5_3__VT__UP, AI_GLM_5_3__VT__DOWN, d_gu,
                                             d_down, y + 2u * c0 * OUT, yrot, gu, act, actr, out + 2u * c0 * OUT, 0u, 0u, signs, pairs,
                                             tokens, n, OUT, I, OUT, limit, over);
    }
    return nn__doors_answer(&argv[3]);
}

SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__vision_embed__zzabi_adapter, ai_glm_5_3__vision_embed__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__vision_block__zzabi_adapter, ai_glm_5_3__vision_block__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_glm_5_3__vision_tail__zzabi_adapter,  ai_glm_5_3__vision_tail__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_GLM_5_3_CPU_OPCODES_VISION__ABI_CUH */
