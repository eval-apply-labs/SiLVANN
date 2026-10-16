#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_BLOCK__ABI_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_BLOCK__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../contracts/objects/block.cuh"    /* the block's table */
#include "../table.cuh"                          /* reading it */
#include "../rows.cuh"                           /* rows through a matrix */

/* ⭐⭐ A BLOCK OF QWEN-IMAGE 2.1's DIFFUSION TRANSFORMER, ON A WORKER — nn's doors over its planes (▶
 * `contracts/objects/block.cuh` for the table and the verb's form). The arithmetic is diffusers'
 * (`QwenImage21TransformerBlock`), every product ▶ `../rows.cuh`:
 *
 *   h = LayerNorm(x)·(1 + scale₁)                       the step's modulation; the prompt's rows take t = 0's
 *   q, k, v = W·h;  q, k normed a head, then turned     (the rotary of the row's frame, height and width)
 *   a = each row over the keys 0 .. its limit;  x += tanh(gate₁)·W_out·a
 *   h = LayerNorm(x)·(1 + scale₂);  x += tanh(gate₂)·W_down(silu(W_gate·h)·W_up·h) */

/* The first of the rows `r0 .. r1` that is the picture's: the sequence's row `first + r` is the prompt's below `prefix`. */
static uint64_t ai_qwen_image__block__zzprivate_split(uint64_t r0, uint64_t r1, uint64_t first, uint64_t prefix) {
    const uint64_t s = prefix > first ? prefix - first : 0u;
    return s < r0 ? r0 : (s > r1 ? r1 : s);
}

/* `o = LayerNorm(x)·w` over `x`'s rows `r0 .. r1` (`o`'s row 0 is `x`'s row `r0`), `w` the modulation's part `part` of the
 * prompt's row where a row is below `prefix` and of the picture's where not — at most two launches. */
static void ai_qwen_image__block__zzprivate_norm(const nn__doors* doors, uint64_t o, uint64_t x, uint64_t mod, uint64_t zero,
                                                 uint64_t H, uint64_t r0, uint64_t r1, uint64_t first, uint64_t prefix,
                                                 unsigned part, unsigned int* over) {
    const uint64_t s = ai_qwen_image__block__zzprivate_split(r0, r1, first, prefix);
    if (s > r0)
        doors->layernorm_rows((uint16_t*)(uintptr_t)o, (const uint16_t*)(uintptr_t)(x + 2u * r0 * H),
                              (const uint16_t*)(uintptr_t)(mod + 2u * part * H), (const uint16_t*)(uintptr_t)zero, H, s - r0,
                              H, H, AI_QWEN_IMAGE__EPS, over);
    if (r1 > s)
        doors->layernorm_rows((uint16_t*)(uintptr_t)(o + 2u * (s - r0) * H), (const uint16_t*)(uintptr_t)(x + 2u * s * H),
                              (const uint16_t*)(uintptr_t)(mod + 2u * (4u + part) * H), (const uint16_t*)(uintptr_t)zero, H,
                              r1 - s, H, H, AI_QWEN_IMAGE__EPS, over);
}

/* `x += g·y` over `x`'s rows `r0 .. r1` (`y`'s row 0 is `x`'s row `r0`), `g` the modulation's part `part` of the prompt's
 * row or the picture's, as `norm` chooses. */
static void ai_qwen_image__block__zzprivate_gated(const nn__doors* doors, uint64_t x, uint64_t y, uint64_t mod, uint64_t H,
                                                  uint64_t r0, uint64_t r1, uint64_t first, uint64_t prefix, unsigned part,
                                                  unsigned int* over) {
    const uint64_t s = ai_qwen_image__block__zzprivate_split(r0, r1, first, prefix);
    if (s > r0)
        doors->vector_gated_add_rows((uint16_t*)(uintptr_t)(x + 2u * r0 * H), (const uint16_t*)(uintptr_t)(x + 2u * r0 * H),
                                     (const uint16_t*)(uintptr_t)y, (const uint16_t*)(uintptr_t)(mod + 2u * part * H), H,
                                     s - r0, over);
    if (r1 > s)
        doors->vector_gated_add_rows((uint16_t*)(uintptr_t)(x + 2u * s * H), (const uint16_t*)(uintptr_t)(x + 2u * s * H),
                                     (const uint16_t*)(uintptr_t)(y + 2u * (s - r0) * H),
                                     (const uint16_t*)(uintptr_t)(mod + 2u * (4u + part) * H), H, r1 - s, over);
}

/* `(ai_qwen_image__block x mod cs limits table rows first prefix)` -> `x`. */
static sys__heap_node ai_qwen_image__block__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 8u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, md = 0, md_room = 0, cs = 0, cs_room = 0, lim = 0, lim_room = 0;
    if (!sys__heap_node__carries_reference(argv[4].dtype) || !sys__node_array__is(argv[4].args[0])
     || argv[5].dtype != SYS__KIND__VALUE_INT || argv[6].dtype != SYS__KIND__VALUE_INT || argv[7].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[1], &md, &md_room)
     || !nn__primitives__room(&argv[2], &cs, &cs_room) || !nn__primitives__room(&argv[3], &lim, &lim_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_IMAGE__B__TABLE], room[AI_QWEN_IMAGE__B__TABLE], v[AI_QWEN_IMAGE__B__TABLE];
    if (!ai_qwen_image__table__zzpackage_read(argv[4].args[0], AI_QWEN_IMAGE__B__HIDDEN, AI_QWEN_IMAGE__B__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_IMAGE__BLOCK__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_IMAGE__B__HIDDEN], heads = v[AI_QWEN_IMAGE__B__HEADS], hd = v[AI_QWEN_IMAGE__B__HEAD_DIM];
    const uint64_t I = v[AI_QWEN_IMAGE__B__INTER], pmax = v[AI_QWEN_IMAGE__B__PREFIX_MAX], chunk = v[AI_QWEN_IMAGE__B__CHUNK];
    const uint64_t dqkv = v[AI_QWEN_IMAGE__B__D_QKV], dout = v[AI_QWEN_IMAGE__B__D_OUT], dmlp = v[AI_QWEN_IMAGE__B__D_MLP];
    const uint64_t ddown = v[AI_QWEN_IMAGE__B__D_DOWN];
    const uint64_t rows = argv[5].args[0], first = argv[6].args[0], prefix = argv[7].args[0], total = first + rows;
    if (H == 0u || heads == 0u || hd == 0u || heads * hd != H || hd > NN__ATTENTION__SET_HEAD_MAX || hd % 2u != 0u
     || H % 128u != 0u || I == 0u || I % 128u != 0u || I > (1ull << 20) || chunk == 0u || prefix > pmax
     || (first != 0u && first != prefix) || rows == 0u || total > (1ull << 20)
     || !ai_qwen_image__rows__zzpackage_holds(room, AI_QWEN_IMAGE__B__Q, dqkv, H, H)
     || !ai_qwen_image__rows__zzpackage_holds(room, AI_QWEN_IMAGE__B__K, dqkv, H, H)
     || !ai_qwen_image__rows__zzpackage_holds(room, AI_QWEN_IMAGE__B__V, dqkv, H, H)
     || !ai_qwen_image__rows__zzpackage_holds(room, AI_QWEN_IMAGE__B__OUT, dout, H, H)
     || !ai_qwen_image__rows__zzpackage_holds(room, AI_QWEN_IMAGE__B__GATE, dmlp, I, H)
     || !ai_qwen_image__rows__zzpackage_holds(room, AI_QWEN_IMAGE__B__UP, dmlp, I, H)
     || !ai_qwen_image__rows__zzpackage_holds(room, AI_QWEN_IMAGE__B__DOWN, ddown, H, I)
     || room[AI_QWEN_IMAGE__B__NORM_Q] < 2u * hd || room[AI_QWEN_IMAGE__B__NORM_K] < 2u * hd
     || room[AI_QWEN_IMAGE__B__PREFIX_K] < 2u * pmax * H || room[AI_QWEN_IMAGE__B__PREFIX_V] < 2u * pmax * H
     || room[AI_QWEN_IMAGE__B__ZERO] < 2u * H || room[AI_QWEN_IMAGE__B__SIGNS] < 1024u)
        return sys__engine__abi__error(AI_QWEN_IMAGE__BLOCK__FAULT_TABLE);
    const uint64_t step = chunk < rows ? chunk : rows;
    /* hrot takes `rows` rows of HIDDEN rotated, or `step` rows of INTER where down's input is rotated */
    const uint64_t rot = rows * H > step * I ? rows * H : step * I;
    const uint64_t S = at[AI_QWEN_IMAGE__B__SCRATCH], P = ai_qwen_image__rows__zzpackage_pairs_bytes(rows);
    const uint64_t h = S + P, hrot = h + 2u * rows * H, q = hrot + 2u * rot, keys = q + 2u * rows * H;
    const uint64_t vals = keys + 2u * total * H, a = vals + 2u * total * H, m1 = a + 2u * rows * H, m2 = m1 + 2u * step * I;
    const uint64_t end = m2 + 2u * step * I;
    if (!nn__primitives__fits(rows * H, x_room) || md_room < 16u * H || cs_room < 4u * total * hd || lim_room < 4u * total
     || room[AI_QWEN_IMAGE__B__SCRATCH] < end - S)
        return sys__engine__abi__error(AI_QWEN_IMAGE__BLOCK__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t pairs = S;
    if (!ai_qwen_image__rows__zzpackage_pairs(ctx, pairs, rows)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t signs = at[AI_QWEN_IMAGE__B__SIGNS], zero = at[AI_QWEN_IMAGE__B__ZERO];
    const uint64_t k_at = keys + 2u * first * H, v_at = vals + 2u * first * H;

    /* ① the attention: q, k, v for the rows, normed a head and turned; the prompt's keys and values kept from step one */
    ai_qwen_image__block__zzprivate_norm(doors, h, x, md, zero, H, 0u, rows, first, prefix, 0u, over);
    const uint64_t hq = ai_qwen_image__rows__zzpackage_rotated(doors, h, hrot, signs, dqkv, H, rows, over);
    ai_qwen_image__rows__zzpackage_product(doors, q, at, room, AI_QWEN_IMAGE__B__Q, dqkv, hq, pairs, rows, H, H, over);
    ai_qwen_image__rows__zzpackage_product(doors, k_at, at, room, AI_QWEN_IMAGE__B__K, dqkv, hq, pairs, rows, H, H, over);
    ai_qwen_image__rows__zzpackage_product(doors, v_at, at, room, AI_QWEN_IMAGE__B__V, dqkv, hq, pairs, rows, H, H, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)at[AI_QWEN_IMAGE__B__NORM_Q],
                        hd, rows * heads, hd, hd, AI_QWEN_IMAGE__EPS, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)k_at, (const uint16_t*)(uintptr_t)k_at,
                        (const uint16_t*)(uintptr_t)at[AI_QWEN_IMAGE__B__NORM_K], hd, rows * heads, hd, hd, AI_QWEN_IMAGE__EPS, over);
    doors->rope_rows((uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)q, (const float*)(uintptr_t)(cs + 4u * first * hd), rows,
                     heads, hd, H, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)k_at, (const uint16_t*)(uintptr_t)k_at, (const float*)(uintptr_t)(cs + 4u * first * hd),
                     rows, heads, hd, H, hd, over);
    if (prefix > 0u) {
        const uint64_t kp = at[AI_QWEN_IMAGE__B__PREFIX_K], vp = at[AI_QWEN_IMAGE__B__PREFIX_V];
        if (first == 0u) {
            doors->vector_copy((uint16_t*)(uintptr_t)kp, (const uint16_t*)(uintptr_t)keys, prefix * H);
            doors->vector_copy((uint16_t*)(uintptr_t)vp, (const uint16_t*)(uintptr_t)vals, prefix * H);
        } else {
            doors->vector_copy((uint16_t*)(uintptr_t)keys, (const uint16_t*)(uintptr_t)kp, prefix * H);
            doors->vector_copy((uint16_t*)(uintptr_t)vals, (const uint16_t*)(uintptr_t)vp, prefix * H);
        }
    }
    doors->attention_limits((uint16_t*)(uintptr_t)a, (const uint16_t*)(uintptr_t)q, (const uint16_t*)(uintptr_t)keys,
                            (const uint16_t*)(uintptr_t)vals, (const uint32_t*)(uintptr_t)(lim + 4u * first), rows, heads, hd,
                            H, H, over);
    /* the projection into h, which the products have done with, then onto the residual through gate one */
    const uint64_t ar = ai_qwen_image__rows__zzpackage_rotated(doors, a, hrot, signs, dout, H, rows, over);
    ai_qwen_image__rows__zzpackage_product(doors, h, at, room, AI_QWEN_IMAGE__B__OUT, dout, ar, pairs, rows, H, H, over);
    ai_qwen_image__block__zzprivate_gated(doors, x, h, md, H, 0u, rows, first, prefix, 1u, over);

    /* ② the MLP, `step` rows at a time */
    for (uint64_t c0 = 0u; c0 < rows; c0 += step) {
        const uint64_t n = rows - c0 < step ? rows - c0 : step;
        ai_qwen_image__block__zzprivate_norm(doors, h, x, md, zero, H, c0, c0 + n, first, prefix, 2u, over);
        const uint64_t hm = ai_qwen_image__rows__zzpackage_rotated(doors, h, hrot, signs, dmlp, H, n, over);
        ai_qwen_image__rows__zzpackage_product(doors, m1, at, room, AI_QWEN_IMAGE__B__GATE, dmlp, hm, pairs, n, I, H, over);
        ai_qwen_image__rows__zzpackage_product(doors, m2, at, room, AI_QWEN_IMAGE__B__UP, dmlp, hm, pairs, n, I, H, over);
        doors->swiglu_combine((uint16_t*)(uintptr_t)m1, (const uint16_t*)(uintptr_t)m1, (const uint16_t*)(uintptr_t)m2, n * I, over);
        const uint64_t mr = ai_qwen_image__rows__zzpackage_rotated(doors, m1, hrot, signs, ddown, I, n, over);
        ai_qwen_image__rows__zzpackage_product(doors, h, at, room, AI_QWEN_IMAGE__B__DOWN, ddown, mr, pairs, n, H, I, over);
        ai_qwen_image__block__zzprivate_gated(doors, x, h, md, H, c0, c0 + n, first, prefix, 3u, over);
    }
    return nn__doors_answer(&argv[0]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_image__block__zzabi_adapter, ai_qwen_image__block__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_BLOCK__ABI_CUH */
