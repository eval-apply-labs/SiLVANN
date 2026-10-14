#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_VISION__ABI_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_VISION__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../contracts/objects/vision.cuh"    /* the tower's tables */

/* ⭐⭐ A PICTURE THROUGH QWEN 3.5'S VISION TOWER, ON A WORKER — three verbs of nn's doors (▶ `contracts/objects/vision.cuh`
 * for their tables): the patch embedding, a block, and the merger. The arithmetic is Hugging Face's
 * (`Qwen3_5VisionModel`), as `python/silvann_runtime/vision.py` writes it out in numpy:
 *
 *   embed    x = W·pixels + b + pos                       (W a row of 1536 halves an output: the pack's 16-bit rows)
 *   block    h = LN(x); q, k, v = W·h + b; q, k turned by the patch's row and column; a = every patch to every patch;
 *            x += W·a + b;  h = LN(x); x += W₂·gelu_tanh(W₁·h + b₁) + b₂
 *   merger   h = LN(x) a patch; four patches as one row; out = W₂·gelu_erf(W₁·h + b₁) + b₂
 *
 * A matrix's codes at 8 bits were rotated row by row in the pack's ladder, so the rows it multiplies are rotated the same
 * way first (`hadamard_ladder_rows`); at 16 bits a matrix is as it was, and its rows go in as they are. Every product is
 * nn's `expert_rows` over the rows `(j, j)` — the pairs at the front of the scratch. */

/* The pairs `(j, j)` for `count` rows at `at`, written from the host 256 at a time. False if the card would not take them. */
static bool ai_qwen_3__vision__zzprivate_pairs(sys__engine__ctx* ctx, uint64_t at, uint64_t count) {
    uint32_t pairs[512u];
    for (uint64_t j0 = 0u; j0 < count; j0 += 256u) {
        const uint64_t n = count - j0 < 256u ? count - j0 : 256u;
        for (uint64_t j = 0u; j < n; ++j) { pairs[2u * j] = (uint32_t)(j0 + j); pairs[2u * j + 1u] = (uint32_t)(j0 + j); }
        if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)(at + 8u * j0), pairs, (size_t)(8u * n))) return false;
    }
    return true;
}

/* The bytes the pairs of `rows` rows take at the front of a scratch, rounded to 16. */
static uint64_t ai_qwen_3__vision__zzprivate_pairs_bytes(uint64_t rows) {
    return (8u * rows + 15u) & ~15ull;
}

/* Whether a matrix of `out_rows` rows of `cols` at width `d` fits its planes — the codes, and the scales below 16 bits —
 * and its bias. */
static bool ai_qwen_3__vision__zzprivate_holds(uint64_t codes_room, uint64_t lut_room, uint64_t bias_room, uint64_t d,
                                               uint64_t out_rows, uint64_t cols) {
    if (d != 4u && d != 8u && d != 16u) return false;
    return out_rows * nn__turboquant__row_bytes(d, cols) <= codes_room && (d == 16u || 2u * out_rows <= lut_room)
        && 2u * out_rows <= bias_room;
}

/* `out = W·x + b` over `rows` rows of `cols`, every operand a card address: the rows rotated into `rot` first where W's
 * codes are rotated (below 16 bits). The pairs are already at `pairs`. */
static void ai_qwen_3__vision__zzprivate_linear(const nn__doors* doors, uint64_t out, const uint64_t* at, const uint64_t* room,
                                                unsigned codes, uint64_t d, uint64_t x, uint64_t rot, uint64_t signs,
                                                uint64_t pairs, uint64_t rows, uint64_t out_rows, uint64_t cols, unsigned int* over) {
    uint64_t in = x;
    if (d != 16u) {
        doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)rot, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)signs,
                                    cols, rows, over);
        in = rot;
    }
    doors->expert_rows((uint16_t*)(uintptr_t)out, (const uint8_t*)(uintptr_t)at[codes], room[codes],
                       (const uint8_t*)(uintptr_t)at[codes + 1u], room[codes + 1u], (const uint16_t*)(uintptr_t)in,
                       (const uint32_t*)(uintptr_t)pairs, rows, d, out_rows, cols, over);
    doors->vector_bias_rows((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)at[codes + 2u],
                            out_rows, rows, over);
}

/* `(ai_qwen_3__vision_embed pixels pos table rows x)` -> `x`. */
static sys__heap_node ai_qwen_3__vision_embed__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t px = 0, px_room = 0, pos = 0, pos_room = 0, x = 0, x_room = 0;
    if (!sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || argv[3].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &px, &px_room)
     || !nn__primitives__room(&argv[1], &pos, &pos_room) || !nn__primitives__room(&argv[4], &x, &x_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__VE__TABLE], room[AI_QWEN_3__VE__TABLE], v[AI_QWEN_3__VE__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[2].args[0], AI_QWEN_3__VE__HIDDEN, AI_QWEN_3__VE__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__VISION__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__VE__HIDDEN], IN = v[AI_QWEN_3__VE__IN], d = v[AI_QWEN_3__VE__D], rows = argv[3].args[0];
    if (H == 0u || IN == 0u || rows == 0u || rows > (1ull << 20) || d != 16u
     || !ai_qwen_3__vision__zzprivate_holds(room[AI_QWEN_3__VE__PATCH], room[AI_QWEN_3__VE__PATCH_LUT],
                                            room[AI_QWEN_3__VE__PATCH_B], d, H, IN))
        return sys__engine__abi__error(AI_QWEN_3__VISION__FAULT_TABLE);
    if (!nn__primitives__fits(rows * IN, px_room) || !nn__primitives__fits(rows * H, pos_room)
     || !nn__primitives__fits(rows * H, x_room) || room[AI_QWEN_3__VE__SCRATCH] < ai_qwen_3__vision__zzprivate_pairs_bytes(rows))
        return sys__engine__abi__error(AI_QWEN_3__VISION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = at[AI_QWEN_3__VE__SCRATCH];
    if (!ai_qwen_3__vision__zzprivate_pairs(ctx, pairs, rows)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    ai_qwen_3__vision__zzprivate_linear(doors, x, at, room, AI_QWEN_3__VE__PATCH, d, px, 0u, 0u, pairs, rows, H, IN, over);
    doors->vector_add((uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)pos, rows * H, over);
    return nn__doors_answer(&argv[4]);
}

/* `(ai_qwen_3__vision_block x positions table rows)` -> `x`. */
static sys__heap_node ai_qwen_3__vision_block__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, ps = 0, ps_room = 0;
    if (!sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || argv[3].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &x, &x_room)
     || !nn__primitives__room(&argv[1], &ps, &ps_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__VB__TABLE], room[AI_QWEN_3__VB__TABLE], v[AI_QWEN_3__VB__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[2].args[0], AI_QWEN_3__VB__HIDDEN, AI_QWEN_3__VB__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__VISION__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__VB__HIDDEN], heads = v[AI_QWEN_3__VB__HEADS], hd = v[AI_QWEN_3__VB__HEAD_DIM];
    const uint64_t I = v[AI_QWEN_3__VB__INTER], theta = v[AI_QWEN_3__VB__THETA], chunk = v[AI_QWEN_3__VB__CHUNK];
    const uint64_t rows = argv[3].args[0];
    if (H == 0u || heads == 0u || hd == 0u || heads * hd != H || hd > NN__ATTENTION__SET_HEAD_MAX || hd % 4u != 0u
     || H % 128u != 0u || I == 0u || theta == 0u || theta >= (1ull << 32) || chunk == 0u || rows == 0u || rows > (1ull << 20)
     || !ai_qwen_3__vision__zzprivate_holds(room[AI_QWEN_3__VB__QKV], room[AI_QWEN_3__VB__QKV_LUT], room[AI_QWEN_3__VB__QKV_B],
                                            v[AI_QWEN_3__VB__D_QKV], 3u * H, H)
     || !ai_qwen_3__vision__zzprivate_holds(room[AI_QWEN_3__VB__PROJ], room[AI_QWEN_3__VB__PROJ_LUT], room[AI_QWEN_3__VB__PROJ_B],
                                            v[AI_QWEN_3__VB__D_PROJ], H, H)
     || !ai_qwen_3__vision__zzprivate_holds(room[AI_QWEN_3__VB__FC1], room[AI_QWEN_3__VB__FC1_LUT], room[AI_QWEN_3__VB__FC1_B],
                                            v[AI_QWEN_3__VB__D_FC1], I, H)
     || !ai_qwen_3__vision__zzprivate_holds(room[AI_QWEN_3__VB__FC2], room[AI_QWEN_3__VB__FC2_LUT], room[AI_QWEN_3__VB__FC2_B],
                                            v[AI_QWEN_3__VB__D_FC2], H, I)
     || (v[AI_QWEN_3__VB__D_FC2] != 16u && I % 128u != 0u) || I > (1ull << 20)
     || room[AI_QWEN_3__VB__NORM1_W] < 2u * H || room[AI_QWEN_3__VB__NORM1_B] < 2u * H
     || room[AI_QWEN_3__VB__NORM2_W] < 2u * H || room[AI_QWEN_3__VB__NORM2_B] < 2u * H || room[AI_QWEN_3__VB__SIGNS] < 1024u)
        return sys__engine__abi__error(AI_QWEN_3__VISION__FAULT_TABLE);
    const uint64_t step = chunk < rows ? chunk : rows;
    /* hrot takes a row of `h` rotated, or of the MLP's width where its second matrix is rotated */
    const uint64_t rot = rows * H > step * I ? rows * H : step * I;
    const uint64_t P = ai_qwen_3__vision__zzprivate_pairs_bytes(rows);
    const uint64_t h = at[AI_QWEN_3__VB__SCRATCH] + P, hrot = h + 2u * rows * H, qkv = hrot + 2u * rot;
    const uint64_t m1 = qkv + 6u * rows * H, m2 = m1 + 2u * step * I, cs = m2 + 2u * step * H, end = cs + 4u * rows * hd;
    if (!nn__primitives__fits(rows * H, x_room) || ps_room < 8u * rows || room[AI_QWEN_3__VB__SCRATCH] < end - at[AI_QWEN_3__VB__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_3__VISION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = at[AI_QWEN_3__VB__SCRATCH];
    if (!ai_qwen_3__vision__zzprivate_pairs(ctx, pairs, rows)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t signs = at[AI_QWEN_3__VB__SIGNS];

    /* ① the attention: q, k, v for every patch, turned by its row and column, every patch to every patch */
    doors->rope_angles_axial_rows((float*)(uintptr_t)cs, (const uint32_t*)(uintptr_t)ps, (float)(uint32_t)theta, hd, rows);
    doors->layernorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__VB__NORM1_W],
                          (const uint16_t*)(uintptr_t)at[AI_QWEN_3__VB__NORM1_B], H, rows, H, H, AI_QWEN_3__VISION__EPS, over);
    ai_qwen_3__vision__zzprivate_linear(doors, qkv, at, room, AI_QWEN_3__VB__QKV, v[AI_QWEN_3__VB__D_QKV], h, hrot, signs, pairs,
                                        rows, 3u * H, H, over);
    doors->rope_rows((uint16_t*)(uintptr_t)qkv, (const uint16_t*)(uintptr_t)qkv, (const float*)(uintptr_t)cs, rows, heads, hd,
                     3u * H, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)(qkv + 2u * H), (const uint16_t*)(uintptr_t)(qkv + 2u * H), (const float*)(uintptr_t)cs,
                     rows, heads, hd, 3u * H, hd, over);
    doors->attention_set((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)qkv, rows, heads, hd, over);
    /* the projection into the front of qkv's room, which the attention has done with, then onto the residual */
    ai_qwen_3__vision__zzprivate_linear(doors, qkv, at, room, AI_QWEN_3__VB__PROJ, v[AI_QWEN_3__VB__D_PROJ], h, hrot, signs, pairs,
                                        rows, H, H, over);
    doors->vector_add((uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)qkv, rows * H, over);

    /* ② the MLP, `step` rows at a time */
    for (uint64_t c0 = 0u; c0 < rows; c0 += step) {
        const uint64_t n = rows - c0 < step ? rows - c0 : step;
        const uint64_t xc = x + 2u * c0 * H;
        doors->layernorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)xc, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__VB__NORM2_W],
                              (const uint16_t*)(uintptr_t)at[AI_QWEN_3__VB__NORM2_B], H, n, H, H, AI_QWEN_3__VISION__EPS, over);
        ai_qwen_3__vision__zzprivate_linear(doors, m1, at, room, AI_QWEN_3__VB__FC1, v[AI_QWEN_3__VB__D_FC1], h, hrot, signs, pairs,
                                            n, I, H, over);
        doors->vector_gelu((uint16_t*)(uintptr_t)m1, (const uint16_t*)(uintptr_t)m1, n * I, 0u, over);
        ai_qwen_3__vision__zzprivate_linear(doors, m2, at, room, AI_QWEN_3__VB__FC2, v[AI_QWEN_3__VB__D_FC2], m1, hrot, signs, pairs,
                                            n, H, I, over);
        doors->vector_add((uint16_t*)(uintptr_t)xc, (const uint16_t*)(uintptr_t)xc, (const uint16_t*)(uintptr_t)m2, n * H, over);
    }
    return nn__doors_answer(&argv[0]);
}

/* `(ai_qwen_3__vision_merger x table rows out)` -> `out`, a row of the text model's width a `MERGE` patches. */
static sys__heap_node ai_qwen_3__vision_merger__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, out = 0, out_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &x, &x_room)
     || !nn__primitives__room(&argv[3], &out, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__VM__TABLE], room[AI_QWEN_3__VM__TABLE], v[AI_QWEN_3__VM__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__VM__HIDDEN, AI_QWEN_3__VM__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__VISION__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__VM__HIDDEN], M = v[AI_QWEN_3__VM__MERGE], OUT = v[AI_QWEN_3__VM__OUT], rows = argv[2].args[0];
    const uint64_t W = M * H, tokens = M == 0u ? 0u : rows / M;
    if (H == 0u || M == 0u || OUT == 0u || rows == 0u || rows % M != 0u || rows > (1ull << 20) || W % 128u != 0u
     || !ai_qwen_3__vision__zzprivate_holds(room[AI_QWEN_3__VM__FC1], room[AI_QWEN_3__VM__FC1_LUT], room[AI_QWEN_3__VM__FC1_B],
                                            v[AI_QWEN_3__VM__D_FC1], W, W)
     || !ai_qwen_3__vision__zzprivate_holds(room[AI_QWEN_3__VM__FC2], room[AI_QWEN_3__VM__FC2_LUT], room[AI_QWEN_3__VM__FC2_B],
                                            v[AI_QWEN_3__VM__D_FC2], OUT, W)
     || room[AI_QWEN_3__VM__NORM_W] < 2u * H || room[AI_QWEN_3__VM__NORM_B] < 2u * H || room[AI_QWEN_3__VM__SIGNS] < 1024u)
        return sys__engine__abi__error(AI_QWEN_3__VISION__FAULT_TABLE);
    const uint64_t P = ai_qwen_3__vision__zzprivate_pairs_bytes(tokens);
    const uint64_t h = at[AI_QWEN_3__VM__SCRATCH] + P, hrot = h + 2u * rows * H, m = hrot + 2u * rows * H, mrot = m + 2u * rows * H;
    if (!nn__primitives__fits(rows * H, x_room) || !nn__primitives__fits(tokens * OUT, out_room)
     || room[AI_QWEN_3__VM__SCRATCH] < P + 8u * rows * H)
        return sys__engine__abi__error(AI_QWEN_3__VISION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = at[AI_QWEN_3__VM__SCRATCH];
    if (!ai_qwen_3__vision__zzprivate_pairs(ctx, pairs, tokens)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t signs = at[AI_QWEN_3__VM__SIGNS];
    doors->layernorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__VM__NORM_W],
                          (const uint16_t*)(uintptr_t)at[AI_QWEN_3__VM__NORM_B], H, rows, H, H, AI_QWEN_3__VISION__EPS, over);
    /* `M` patches side by side are one row of `W`: the rows of `h` as they lie */
    ai_qwen_3__vision__zzprivate_linear(doors, m, at, room, AI_QWEN_3__VM__FC1, v[AI_QWEN_3__VM__D_FC1], h, hrot, signs, pairs,
                                        tokens, W, W, over);
    doors->vector_gelu((uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)m, tokens * W, 1u, over);
    ai_qwen_3__vision__zzprivate_linear(doors, out, at, room, AI_QWEN_3__VM__FC2, v[AI_QWEN_3__VM__D_FC2], m, mrot, signs, pairs,
                                        tokens, OUT, W, over);
    return nn__doors_answer(&argv[3]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__vision_embed__zzabi_adapter,  ai_qwen_3__vision_embed__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__vision_block__zzabi_adapter,  ai_qwen_3__vision_block__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__vision_merger__zzabi_adapter, ai_qwen_3__vision_merger__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_VISION__ABI_CUH */
