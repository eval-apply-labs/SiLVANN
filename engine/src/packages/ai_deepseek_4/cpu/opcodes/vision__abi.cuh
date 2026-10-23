#ifndef SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_VISION__ABI_CUH
#define SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_VISION__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdlib.h>                              /* a picture's pairs, made on the host: malloc, free */
#include "../table.cuh"                          /* the plane table every verb reads first */
#include "../../contracts/objects/vision.cuh"    /* the tower's tables */

/* ⭐⭐ A PICTURE THROUGH DEEPSEEK-V4.1'S VISION TOWER, ON A WORKER — three verbs of nn's doors (▶ `contracts/objects/vision.cuh`
 * for their tables): the patch embedding, a block, and the tail (the last norm and the aligner). The arithmetic is the
 * release's `inference/vision.py`:
 *
 *   embed    x = W·pixels + b                               (W a row of 588 halves an output: decoded at the boot)
 *   block    h = RMS(x); q, k, v = W·h + b; q, k turned by the patch's row and column (16 frequencies each, a head's
 *            halves rotated together); a = every patch to every patch; x += W·a + b;
 *            h = RMS(x); g, u = W_gu·h; x += W_d·(silu(g) · u)
 *   tail     h = RMS(x) a patch; a 3 x 3 block of patches as one row (zeros past the picture's edge);
 *            out = W_2·gelu_erf(W_1·row + b_1) + b_2
 *
 * A matrix's codes below 16 bits were rotated row by row in the pack's ladder, so the rows it multiplies are rotated the
 * same way first (`hadamard_ladder_rows`); at 16 bits its rows go in as they are. Every product is nn's `expert_rows` over
 * the pairs `(t, t)` at the front of the scratch. */

/* `(t, t)` for `count` rows at `at`, written from the host. False if the card would not take them. */
static bool ai_deepseek_4__vision__zzprivate_pairs(sys__engine__ctx* ctx, uint64_t at, uint64_t count) {
    uint32_t* p = (uint32_t*)malloc(sizeof(uint32_t) * 2u * count);
    if (p == 0) return false;
    for (uint64_t t = 0u; t < count; ++t) {
        p[2u * t] = (uint32_t)t;
        p[2u * t + 1u] = (uint32_t)t;
    }
    const bool ok = sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)at, p, (size_t)(8u * count));
    free(p);
    return ok;
}

/* The bytes the pairs of `rows` rows take at the front of a scratch, rounded to 256. */
static uint64_t ai_deepseek_4__vision__zzprivate_pairs_bytes(uint64_t rows) {
    return (8u * rows + 255u) & ~255ull;
}

/* Whether a matrix of `out_rows` rows of `cols` at width `d` fits its planes — its codes, and its scales below 16 bits —
 * and a rotated one's rows are whole rotation blocks. */
static bool ai_deepseek_4__vision__zzprivate_holds(uint64_t codes_room, uint64_t lut_room, uint64_t d, uint64_t out_rows,
                                                uint64_t cols) {
    if (d != 4u && d != 8u && d != 16u) return false;
    if (d != 16u && cols % 128u != 0u) return false;
    return out_rows * nn__turboquant__row_bytes(d, cols) <= codes_room && (d == 16u || 2u * out_rows <= lut_room);
}

/* `out = W·x (+ b)` over `rows` rows of `cols`, every operand a card address: the rows rotated into `rot` first where W's
 * codes are rotated. `bias` 0: none. The `(t, t)` pairs are at `pairs`. */
static void ai_deepseek_4__vision__zzprivate_linear(const nn__doors* doors, uint64_t out, const uint64_t* at, const uint64_t* room,
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

/* `(ai_deepseek_4__vision_embed pixels table rows x)` -> `x`. */
static sys__heap_node ai_deepseek_4__vision_embed__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t px = 0, px_room = 0, x = 0, x_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &px, &px_room)
     || !nn__primitives__room(&argv[3], &x, &x_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_DEEPSEEK_4__VE__TABLE], room[AI_DEEPSEEK_4__VE__TABLE], v[AI_DEEPSEEK_4__VE__TABLE];
    float r[AI_DEEPSEEK_4__VE__TABLE];
    if (!ai_deepseek_4__table__zzpackage_read(argv[1].args[0], AI_DEEPSEEK_4__VE__PLANES, AI_DEEPSEEK_4__VE__TABLE,
                                           AI_DEEPSEEK_4__VE__TABLE, at, room, v, r))
        return sys__engine__abi__error(AI_DEEPSEEK_4__VISION__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__VE__HIDDEN], IN = v[AI_DEEPSEEK_4__VE__IN], d = v[AI_DEEPSEEK_4__VE__D], rows = argv[2].args[0];
    if (H == 0u || IN == 0u || IN % 2u != 0u || rows == 0u || rows > (1ull << 20) || d != 16u
     || !ai_deepseek_4__vision__zzprivate_holds(room[AI_DEEPSEEK_4__VE__PATCH], room[AI_DEEPSEEK_4__VE__PATCH_LUT], d, H, IN)
     || room[AI_DEEPSEEK_4__VE__PATCH_B] < 2u * H)
        return sys__engine__abi__error(AI_DEEPSEEK_4__VISION__FAULT_TABLE);
    if (!nn__primitives__fits(rows * IN, px_room) || !nn__primitives__fits(rows * H, x_room)
     || room[AI_DEEPSEEK_4__VE__SCRATCH] < ai_deepseek_4__vision__zzprivate_pairs_bytes(rows))
        return sys__engine__abi__error(AI_DEEPSEEK_4__VISION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = at[AI_DEEPSEEK_4__VE__SCRATCH];
    if (!ai_deepseek_4__vision__zzprivate_pairs(ctx, pairs, rows)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    ai_deepseek_4__vision__zzprivate_linear(doors, x, at, room, AI_DEEPSEEK_4__VE__PATCH, d, px, 0u, 0u, pairs, rows, H, IN,
                                          at[AI_DEEPSEEK_4__VE__PATCH_B], ctx->fault_word);
    return nn__doors_answer(&argv[3]);
}

/* `(ai_deepseek_4__vision_block x positions table rows)` -> `x`. */
static sys__heap_node ai_deepseek_4__vision_block__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, ps = 0, ps_room = 0;
    if (!sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || argv[3].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &x, &x_room)
     || !nn__primitives__room(&argv[1], &ps, &ps_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_DEEPSEEK_4__VB__TABLE], room[AI_DEEPSEEK_4__VB__TABLE], v[AI_DEEPSEEK_4__VB__TABLE];
    float r[AI_DEEPSEEK_4__VB__TABLE];
    if (!ai_deepseek_4__table__zzpackage_read(argv[2].args[0], AI_DEEPSEEK_4__VB__PLANES, AI_DEEPSEEK_4__VB__EPS,
                                           AI_DEEPSEEK_4__VB__TABLE, at, room, v, r))
        return sys__engine__abi__error(AI_DEEPSEEK_4__VISION__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__VB__HIDDEN], heads = v[AI_DEEPSEEK_4__VB__HEADS], hd = v[AI_DEEPSEEK_4__VB__HEAD_DIM];
    const uint64_t I = v[AI_DEEPSEEK_4__VB__INTER], theta = v[AI_DEEPSEEK_4__VB__THETA], chunk = v[AI_DEEPSEEK_4__VB__CHUNK];
    const uint64_t d_gu = v[AI_DEEPSEEK_4__VB__D_GU], d_down = v[AI_DEEPSEEK_4__VB__D_DOWN], rows = argv[3].args[0];
    const float eps = r[AI_DEEPSEEK_4__VB__EPS];
    if (H == 0u || heads == 0u || hd == 0u || heads * hd != H || hd > NN__ATTENTION__SET_HEAD_MAX || hd % 4u != 0u
     || I == 0u || I > (1ull << 20) || theta == 0u || theta >= (1ull << 32) || chunk == 0u || rows == 0u || rows > (1ull << 20)
     || !(eps > 0.0f)
     || !ai_deepseek_4__vision__zzprivate_holds(room[AI_DEEPSEEK_4__VB__QKV], room[AI_DEEPSEEK_4__VB__QKV_LUT], v[AI_DEEPSEEK_4__VB__D_QKV], 3u * H, H)
     || !ai_deepseek_4__vision__zzprivate_holds(room[AI_DEEPSEEK_4__VB__PROJ], room[AI_DEEPSEEK_4__VB__PROJ_LUT], v[AI_DEEPSEEK_4__VB__D_PROJ], H, H)
     || !ai_deepseek_4__vision__zzprivate_holds(room[AI_DEEPSEEK_4__VB__GU], room[AI_DEEPSEEK_4__VB__GU_LUT], d_gu, 2u * I, H)
     || !ai_deepseek_4__vision__zzprivate_holds(room[AI_DEEPSEEK_4__VB__DOWN], room[AI_DEEPSEEK_4__VB__DOWN_LUT], d_down, H, I)
     || room[AI_DEEPSEEK_4__VB__QKV_B] < 6u * H || room[AI_DEEPSEEK_4__VB__PROJ_B] < 2u * H
     || room[AI_DEEPSEEK_4__VB__NORM1_W] < 2u * H || room[AI_DEEPSEEK_4__VB__NORM2_W] < 2u * H || room[AI_DEEPSEEK_4__VB__SIGNS] < 1024u)
        return sys__engine__abi__error(AI_DEEPSEEK_4__VISION__FAULT_TABLE);
    const uint64_t step = chunk < rows ? chunk : rows;
    /* hrot takes the rows of `h` rotated, or a step's rows of the MLP's width */
    const uint64_t rotn = rows * H > step * I ? rows * H : step * I;
    const uint64_t P = ai_deepseek_4__vision__zzprivate_pairs_bytes(rows);
    const uint64_t h = at[AI_DEEPSEEK_4__VB__SCRATCH] + P, hrot = h + 2u * rows * H, qkv = hrot + 2u * rotn;
    const uint64_t gu = qkv + 6u * rows * H, act = gu + 4u * step * I, actr = act + 2u * step * I, m = actr + 2u * step * I;
    const uint64_t cs = m + 2u * step * H, end = cs + 4u * rows * hd;
    if (!nn__primitives__fits(rows * H, x_room) || ps_room < 8u * rows || room[AI_DEEPSEEK_4__VB__SCRATCH] < end - at[AI_DEEPSEEK_4__VB__SCRATCH])
        return sys__engine__abi__error(AI_DEEPSEEK_4__VISION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = at[AI_DEEPSEEK_4__VB__SCRATCH];
    if (!ai_deepseek_4__vision__zzprivate_pairs(ctx, pairs, rows)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t signs = at[AI_DEEPSEEK_4__VB__SIGNS];

    /* ① the attention: q, k, v for every patch, q and k turned by the patch's row and column, every patch to every patch */
    doors->rope_angles_axial_rows((float*)(uintptr_t)cs, (const uint32_t*)(uintptr_t)ps, (float)(uint32_t)theta, hd, rows);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__VB__NORM1_W],
                        H, rows, H, H, eps, over);
    ai_deepseek_4__vision__zzprivate_linear(doors, qkv, at, room, AI_DEEPSEEK_4__VB__QKV, v[AI_DEEPSEEK_4__VB__D_QKV], h, hrot, signs, pairs,
                                          rows, 3u * H, H, at[AI_DEEPSEEK_4__VB__QKV_B], over);
    doors->rope_rows((uint16_t*)(uintptr_t)qkv, (const uint16_t*)(uintptr_t)qkv, (const float*)(uintptr_t)cs, rows, heads, hd,
                     3u * H, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)(qkv + 2u * H), (const uint16_t*)(uintptr_t)(qkv + 2u * H), (const float*)(uintptr_t)cs,
                     rows, heads, hd, 3u * H, hd, over);
    doors->attention_set((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)qkv, rows, heads, hd, over);
    /* the projection into the front of qkv's room, which the attention has done with, then onto the residual */
    ai_deepseek_4__vision__zzprivate_linear(doors, qkv, at, room, AI_DEEPSEEK_4__VB__PROJ, v[AI_DEEPSEEK_4__VB__D_PROJ], h, hrot, signs, pairs,
                                          rows, H, H, at[AI_DEEPSEEK_4__VB__PROJ_B], over);
    doors->vector_add((uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)qkv, rows * H, over);

    /* ② the MLP, `step` rows at a time: gate and up side by side a row, SwiGLU (the clamp past any half), down */
    for (uint64_t c0 = 0u; c0 < rows; c0 += step) {
        const uint64_t n = rows - c0 < step ? rows - c0 : step;
        const uint64_t xc = x + 2u * c0 * H;
        doors->rmsnorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)xc, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__VB__NORM2_W],
                            H, n, H, H, eps, over);
        ai_deepseek_4__vision__zzprivate_linear(doors, gu, at, room, AI_DEEPSEEK_4__VB__GU, d_gu, h, hrot, signs, pairs, n, 2u * I, H, 0u,
                                              over);
        doors->swiglu_clamped((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, n, I, 3.0e38f, over);
        ai_deepseek_4__vision__zzprivate_linear(doors, m, at, room, AI_DEEPSEEK_4__VB__DOWN, d_down, act, actr, signs, pairs, n, H, I, 0u,
                                              over);
        doors->vector_add((uint16_t*)(uintptr_t)xc, (const uint16_t*)(uintptr_t)xc, (const uint16_t*)(uintptr_t)m, n * H, over);
    }
    return nn__doors_answer(&argv[0]);
}

/* `(ai_deepseek_4__vision_tail x table h w out)` -> `out`, a row of the text model's width a block of patches. */
static sys__heap_node ai_deepseek_4__vision_tail__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, out = 0, out_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT || argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[4], &out, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_DEEPSEEK_4__VT__TABLE], room[AI_DEEPSEEK_4__VT__TABLE], v[AI_DEEPSEEK_4__VT__TABLE];
    float r[AI_DEEPSEEK_4__VT__TABLE];
    if (!ai_deepseek_4__table__zzpackage_read(argv[1].args[0], AI_DEEPSEEK_4__VT__PLANES, AI_DEEPSEEK_4__VT__EPS,
                                           AI_DEEPSEEK_4__VT__TABLE, at, room, v, r))
        return sys__engine__abi__error(AI_DEEPSEEK_4__VISION__FAULT_TABLE);
    const uint64_t H = v[AI_DEEPSEEK_4__VT__HIDDEN], R = v[AI_DEEPSEEK_4__VT__RATIO], OUT = v[AI_DEEPSEEK_4__VT__OUT];
    const uint64_t gh = argv[2].args[0], gw = argv[3].args[0], rows = gh * gw;
    const uint64_t th = R == 0u ? 0u : (gh + R - 1u) / R, tw = R == 0u ? 0u : (gw + R - 1u) / R, tokens = th * tw;
    const uint64_t K = R * R * H;
    const float eps = r[AI_DEEPSEEK_4__VT__EPS];
    if (H == 0u || R == 0u || R > 8u || OUT == 0u || gh == 0u || gw == 0u || gh > (1ull << 16) || gw > (1ull << 16)
     || rows > (1ull << 20) || !(eps > 0.0f)
     || !ai_deepseek_4__vision__zzprivate_holds(room[AI_DEEPSEEK_4__VT__W1], room[AI_DEEPSEEK_4__VT__W1_LUT], v[AI_DEEPSEEK_4__VT__D_W1], OUT, K)
     || !ai_deepseek_4__vision__zzprivate_holds(room[AI_DEEPSEEK_4__VT__W2], room[AI_DEEPSEEK_4__VT__W2_LUT], v[AI_DEEPSEEK_4__VT__D_W2], OUT, OUT)
     || room[AI_DEEPSEEK_4__VT__NORM_W] < 2u * H || room[AI_DEEPSEEK_4__VT__W1_B] < 2u * OUT || room[AI_DEEPSEEK_4__VT__W2_B] < 2u * OUT
     || room[AI_DEEPSEEK_4__VT__SIGNS] < 1024u)
        return sys__engine__abi__error(AI_DEEPSEEK_4__VISION__FAULT_TABLE);
    /* u takes the blocks' rows, then the aligner's middle; urot a rotated row of either */
    const uint64_t wide = tokens * K > tokens * OUT ? tokens * K : tokens * OUT;
    const uint64_t P = ai_deepseek_4__vision__zzprivate_pairs_bytes(tokens);
    const uint64_t h = at[AI_DEEPSEEK_4__VT__SCRATCH] + P, u = h + 2u * rows * H, urot = u + 2u * wide, y = urot + 2u * wide;
    const uint64_t end = y + 2u * tokens * OUT;
    if (!nn__primitives__fits(rows * H, x_room) || !nn__primitives__fits(tokens * OUT, out_room)
     || room[AI_DEEPSEEK_4__VT__SCRATCH] < end - at[AI_DEEPSEEK_4__VT__SCRATCH])
        return sys__engine__abi__error(AI_DEEPSEEK_4__VISION__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = at[AI_DEEPSEEK_4__VT__SCRATCH];
    if (!ai_deepseek_4__vision__zzprivate_pairs(ctx, pairs, tokens)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t signs = at[AI_DEEPSEEK_4__VT__SIGNS];
    /* the last norm a patch; each block of patches one row, a patch's channels together, zeros past the edge */
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_DEEPSEEK_4__VT__NORM_W],
                        H, rows, H, H, eps, over);
    doors->unfold_rows((uint16_t*)(uintptr_t)u, (const uint16_t*)(uintptr_t)h, gh, gw, H, R, R, R, 0u, 0u, 1u, tw, 0u, tokens);
    /* the aligner: the first map and its bias into y, GELU, the second map and its bias */
    ai_deepseek_4__vision__zzprivate_linear(doors, y, at, room, AI_DEEPSEEK_4__VT__W1, v[AI_DEEPSEEK_4__VT__D_W1], u, urot, signs, pairs,
                                          tokens, OUT, K, at[AI_DEEPSEEK_4__VT__W1_B], over);
    doors->vector_gelu((uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)y, tokens * OUT, 1u, over);
    ai_deepseek_4__vision__zzprivate_linear(doors, out, at, room, AI_DEEPSEEK_4__VT__W2, v[AI_DEEPSEEK_4__VT__D_W2], y, urot, signs, pairs,
                                          tokens, OUT, OUT, at[AI_DEEPSEEK_4__VT__W2_B], over);
    return nn__doors_answer(&argv[4]);
}

SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__vision_embed__zzabi_adapter, ai_deepseek_4__vision_embed__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__vision_block__zzabi_adapter, ai_deepseek_4__vision_block__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_deepseek_4__vision_tail__zzabi_adapter,  ai_deepseek_4__vision_tail__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_OPCODES_VISION__ABI_CUH */
