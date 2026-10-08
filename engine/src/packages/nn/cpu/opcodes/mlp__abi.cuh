#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_MLP__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_MLP__ABI_CUH
/* ══ A DENSE MLP, ITS NORM AND ITS RESIDUAL, ON THE PROGRAM'S SIDE ══════════════════════════════════════════════════
 * The half of a dense model's layer after its mixer, as one word over a plane table (▶ `contracts/objects/mlp.cuh`):
 * the norm, gate and up in one grouped launch, their SwiGLU, the activation rotated, the down with the residual added.
 * The arithmetic is the doors'; here the table is read and the operands checked. */

#include "../doors.cuh"
#include "../../contracts/objects/mlp.cuh"

/* `(nn__mlp__apply h1 table out [residual])` -> `out`: `out = residual + down(swiglu(gate, up)(rmsnorm(h1)))`, the
 * residual `h1` when none is named. */
static sys__heap_node nn__mlp__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u && argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t h1 = 0, h1_room = 0, out = 0, out_room = 0, res = 0, res_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !nn__primitives__room(&argv[0], &h1, &h1_room) || !nn__primitives__room(&argv[2], &out, &out_room)
     || !nn__primitives__room(&argv[argc == 4u ? 3u : 0u], &res, &res_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[NN__MLP__TABLE], room[NN__MLP__TABLE], v[NN__MLP__TABLE];
    float r[NN__MLP__TABLE];
    if (!nn__primitives__table(argv[1].args[0], NN__MLP__HIDDEN, NN__MLP__EPS, NN__MLP__TABLE, at, room, v, r))
        return sys__engine__abi__error(NN__MLP__FAULT_TABLE);
    const uint64_t H = v[NN__MLP__HIDDEN], I = v[NN__MLP__INTER];
    if (H == 0u || I == 0u || H > (1ull << 24) || I > (1ull << 24)
     || !nn__primitives__fits(H, h1_room) || !nn__primitives__fits(H, out_room) || !nn__primitives__fits(H, res_room)
     || !nn__primitives__fits(H, room[NN__MLP__NORM]) || !nn__primitives__fits(1u, room[NN__MLP__ONE]))
        return sys__engine__abi__error(NN__MLP__FAULT_TABLE);
    /* scratch, in halves: h · hrot · [gate · up] · act · actr */
    if (!nn__primitives__fits(2u * H + 4u * I, room[NN__MLP__SCRATCH])) return sys__engine__abi__error(NN__MLP__FAULT_ROOM);
    const uint64_t h = at[NN__MLP__SCRATCH], hrot = h + 2u * H, gu = hrot + 2u * H, act = gu + 2u * 2u * I, actr = act + 2u * I;
    const uint64_t signs = at[NN__MLP__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    const bool rotated = nn__primitives__table_flag(argv[1].args[0], NN__MLP__RESIDUAL_ROTATED);
    doors->rmsnorm((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)at[NN__MLP__NORM], H, r[NN__MLP__EPS], over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, H, over);
    nn__turboquant__groups g = {};
    g.codes[0] = at[NN__MLP__GATE]; g.luts[0] = at[NN__MLP__GATE_SCALES]; g.rows[0] = I; g.d[0] = v[NN__MLP__D_GATE]; g.out_at[0] = 0u;
    g.codes[1] = at[NN__MLP__UP];   g.luts[1] = at[NN__MLP__UP_SCALES];   g.rows[1] = I; g.d[1] = v[NN__MLP__D_UP];   g.out_at[1] = I;
    g.count = 2u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)gu, g, (const uint16_t*)(uintptr_t)(rotated ? h : hrot), H, over);
    doors->swiglu_pairs((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, 1u, I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)signs, I, over);
    nn__turboquant__groups dn = {};
    dn.codes[0] = at[NN__MLP__DOWN]; dn.luts[0] = at[NN__MLP__DOWN_SCALES]; dn.rows[0] = H; dn.d[0] = v[NN__MLP__D_DOWN];
    dn.out_at[0] = 0u; dn.count = 1u;
    doors->turboquant_gemv_groups_sum((uint16_t*)(uintptr_t)out, dn, (const uint16_t*)(uintptr_t)actr,
                                      (const uint16_t*)(uintptr_t)at[NN__MLP__ONE], (const uint16_t*)(uintptr_t)res, H, I, over);
    return nn__doors_answer(&argv[2]);
}

/* `(nn__mlp__rows h1 table out rows)` -> `out`: `out = h1 + mlp(rmsnorm(h1))` for `rows` rows, each row's output where
 * its input is. Gate and up of every row in one grouped launch, their SwiGLU, the rotation, the down rows, the
 * residual. The scratch starts with the rows' pairs (▶ `NN__MLP__ROWS_MAX`). */
static sys__heap_node nn__mlp__zzabi_apply_rows(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, out = 0, out_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[2], &out, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[NN__MLP__TABLE], room[NN__MLP__TABLE], v[NN__MLP__TABLE];
    float r[NN__MLP__TABLE];
    if (!nn__primitives__table(argv[1].args[0], NN__MLP__HIDDEN, NN__MLP__EPS, NN__MLP__TABLE, at, room, v, r))
        return sys__engine__abi__error(NN__MLP__FAULT_TABLE);
    const uint64_t H = v[NN__MLP__HIDDEN], I = v[NN__MLP__INTER], T = argv[3].args[0];
    if (H == 0u || I == 0u || H > (1ull << 24) || I > (1ull << 24) || T == 0u || T > NN__MLP__ROWS_MAX
     || !nn__primitives__fits(T * H, x_room) || !nn__primitives__fits(T * H, out_room) || !nn__primitives__fits(H, room[NN__MLP__NORM]))
        return sys__engine__abi__error(NN__MLP__FAULT_TABLE);
    /* scratch: the pairs, then in halves h · hrot · [gates · ups] · act · actr · ao */
    const uint64_t pairs = at[NN__MLP__SCRATCH], h = pairs + 8u * NN__MLP__ROWS_MAX, hrot = h + 2u * T * H;
    const uint64_t gu = hrot + 2u * T * H, act = gu + 2u * 2u * T * I, actr = act + 2u * T * I, ao = actr + 2u * T * I;
    if (room[NN__MLP__SCRATCH] < ao + 2u * T * H - pairs) return sys__engine__abi__error(NN__MLP__FAULT_ROOM);
    const uint64_t signs = at[NN__MLP__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    /* each row its own pair: row j's input and output both row j */
    uint32_t each[2u * NN__MLP__ROWS_MAX];
    for (uint64_t j = 0u; j < T; ++j) { each[2u * j] = (uint32_t)j; each[2u * j + 1u] = (uint32_t)j; }
    if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)pairs, each, (size_t)(8u * T)))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    const bool rotated = nn__primitives__table_flag(argv[1].args[0], NN__MLP__RESIDUAL_ROTATED);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)(rotated ? hrot : h), (const uint16_t*)(uintptr_t)x,
                        (const uint16_t*)(uintptr_t)at[NN__MLP__NORM], H, T, H, H, r[NN__MLP__EPS], over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, T * H, over);
    nn__expert__groups gr = {};
    gr.codes[0] = at[NN__MLP__GATE]; gr.luts[0] = at[NN__MLP__GATE_SCALES]; gr.out_rows[0] = I; gr.d[0] = v[NN__MLP__D_GATE];
    gr.codes[1] = at[NN__MLP__UP];   gr.luts[1] = at[NN__MLP__UP_SCALES];   gr.out_rows[1] = I; gr.d[1] = v[NN__MLP__D_UP];
    gr.out_at[0] = 0u; gr.out_at[1] = T * I;
    gr.pairs[0] = gr.pairs[1] = T;
    gr.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)gu, gr, (const uint16_t*)(uintptr_t)hrot, (const uint32_t*)(uintptr_t)pairs, H, over);
    doors->swiglu_combine((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, (const uint16_t*)(uintptr_t)(gu + 2u * T * I), T * I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)signs, T * I, over);
    doors->expert_rows((uint16_t*)(uintptr_t)ao, (const uint8_t*)(uintptr_t)at[NN__MLP__DOWN], room[NN__MLP__DOWN],
                       (const uint8_t*)(uintptr_t)at[NN__MLP__DOWN_SCALES], room[NN__MLP__DOWN_SCALES], (const uint16_t*)(uintptr_t)actr,
                       (const uint32_t*)(uintptr_t)pairs, T, v[NN__MLP__D_DOWN], H, I, over);
    doors->vector_add((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)ao, T * H, over);
    return nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(nn__mlp__zzabi_adapter,      nn__mlp__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(nn__mlp__zzabi_adapter_rows, nn__mlp__zzabi_apply_rows)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_MLP__ABI_CUH */
