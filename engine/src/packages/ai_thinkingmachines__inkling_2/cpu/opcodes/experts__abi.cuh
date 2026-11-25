#ifndef SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_EXPERTS__ABI_CUH
#define SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_EXPERTS__ABI_CUH
/* ══ INKLING 2's ROUTED EXPERTS ════════════════════════════════════════════════════════════════════════════════════
 * The routine is nn's (`nn__routed__position`: the picks' gate-and-ups, the activation, its rotation, the downs weighed
 * into the sum, the experts not in memory read while the rest compute); the verb is the model's — it reads the model's
 * table into nn's `nn__routed`, and what it keeps of its own is the hand's layout and the activation: SwiGLU soft-capped,
 * `gate_cap·tanh(g / gate_cap) · σ(g) · up_cap·tanh(u / up_cap)`. ▶ `contracts/objects/experts.cuh`. */

#include "../../../nn/cpu/doors.cuh"            /* nn's doors, a buffer's room, a plane table */
#include "../../../nn/cpu/routed__header.cuh"   /* the routed experts, as nn runs them */

/* The experts table's backing, when it has one: none is `file` zero. False when the cell is there and not a backing. */
static bool ai_thinkingmachines__inkling_2__experts__zzprivate_backing(uint64_t table, nn__expert__backing* backing) {
    const nn__expert__backing none = {};
    *backing = none;
    if (sys__node_array__length(table) <= AI_THINKINGMACHINES__INKLING_2__EXP__BACKING) return true;
    const sys__heap_node n = sys__node_array__borrow(table, AI_THINKINGMACHINES__INKLING_2__EXP__BACKING);
    if (n.dtype == SYS__KIND__VALUE_INT && n.args[0] == 0ull) return true;
    return sys__heap_node__carries_reference(n.dtype) && nn__expert__backing_of(n.args[0], backing);
}

/* `(ai_thinkingmachines__inkling_2__experts hand picks table routed)` -> `routed`: Σ w_j · down_j(act(gate_up_j(x))) over the
 * position's picks, `x` the hand's input and `w` its weights, the activation soft-capped by the table's two caps — as nn
 * runs a position's routed experts. Every expert is this worker's. */
static sys__heap_node ai_thinkingmachines__inkling_2__experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hand = 0, h_room = 0, routed = 0, r_room = 0;
    if (!nn__primitives__room(&argv[0], &hand, &h_room) || !nn__primitives__room(&argv[3], &routed, &r_room)
     || !sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t table = argv[2].args[0];
    uint64_t at[AI_THINKINGMACHINES__INKLING_2__EXP__TABLE], room[AI_THINKINGMACHINES__INKLING_2__EXP__TABLE], v[AI_THINKINGMACHINES__INKLING_2__EXP__TABLE];
    float r[AI_THINKINGMACHINES__INKLING_2__EXP__TABLE];
    if (!nn__primitives__table(table, AI_THINKINGMACHINES__INKLING_2__EXP__PLANES, AI_THINKINGMACHINES__INKLING_2__EXP__GATE_CAP, AI_THINKINGMACHINES__INKLING_2__EXP__TABLE, at, room, v, r))
        return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE);
    const uint64_t H = v[AI_THINKINGMACHINES__INKLING_2__EXP__HIDDEN], I = v[AI_THINKINGMACHINES__INKLING_2__EXP__INTER], K = v[AI_THINKINGMACHINES__INKLING_2__EXP__TOP_K], E = v[AI_THINKINGMACHINES__INKLING_2__EXP__EXPERTS];
    if (H == 0u || I == 0u || K == 0u || K > E || K > AI_THINKINGMACHINES__INKLING_2__HAND__WEIGHTS || K > NN__TURBOQUANT__GROUPS_MAX
     || H > (1ull << 24) || I > (1ull << 24) || v[AI_THINKINGMACHINES__INKLING_2__EXP__INT8] > 1u
     || !(r[AI_THINKINGMACHINES__INKLING_2__EXP__GATE_CAP] > 0.0f) || !(r[AI_THINKINGMACHINES__INKLING_2__EXP__UP_CAP] > 0.0f)
     || sys__node_array__length(argv[1].args[0]) < K
     || !nn__primitives__fits(AI_THINKINGMACHINES__INKLING_2__HAND__HALVES(H), h_room) || !nn__primitives__fits(H, r_room)
     || !nn__primitives__fits(512u, room[AI_THINKINGMACHINES__INKLING_2__EXP__SIGNS]) || !nn__primitives__fits(H, room[AI_THINKINGMACHINES__INKLING_2__EXP__ZERO]))
        return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE);
    const nn__routed none = {};
    nn__routed rt = none;
    rt.layer = v[AI_THINKINGMACHINES__INKLING_2__EXP__LAYER]; rt.type = v[AI_THINKINGMACHINES__INKLING_2__EXP__TYPE];
    rt.hidden = H; rt.inter = I; rt.experts = E; rt.top_k = K; rt.bits = v[AI_THINKINGMACHINES__INKLING_2__EXP__D];
    rt.up_lut = v[AI_THINKINGMACHINES__INKLING_2__EXP__UP_LUT]; rt.down = v[AI_THINKINGMACHINES__INKLING_2__EXP__DOWN]; rt.down_lut = v[AI_THINKINGMACHINES__INKLING_2__EXP__DOWN_LUT];
    rt.first = 0u; rt.count = E; rt.int8 = v[AI_THINKINGMACHINES__INKLING_2__EXP__INT8] == 1u;
    rt.gate_cap = r[AI_THINKINGMACHINES__INKLING_2__EXP__GATE_CAP]; rt.up_cap = r[AI_THINKINGMACHINES__INKLING_2__EXP__UP_CAP];
    rt.signs = at[AI_THINKINGMACHINES__INKLING_2__EXP__SIGNS]; rt.zero = at[AI_THINKINGMACHINES__INKLING_2__EXP__ZERO];
    rt.scratch = at[AI_THINKINGMACHINES__INKLING_2__EXP__SCRATCH]; rt.scratch_room = room[AI_THINKINGMACHINES__INKLING_2__EXP__SCRATCH];
    rt.hand_bytes = AI_THINKINGMACHINES__INKLING_2__HAND__ROW(H, K); rt.picks_at = AI_THINKINGMACHINES__INKLING_2__HAND__PICKS(H); rt.batch = AI_THINKINGMACHINES__INKLING_2__EXP__BATCH;
    if (!ai_thinkingmachines__inkling_2__experts__zzprivate_backing(table, &rt.backing)) return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    nn__routed__done done;
    const uint64_t f = nn__routed__position(ctx, doors, &rt, hand, argv[1].args[0], 0, routed, &done);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[3]);
}

SYS__ENGINE__ABI__BRIDGE(ai_thinkingmachines__inkling_2__experts__zzabi_adapter, ai_thinkingmachines__inkling_2__experts__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_EXPERTS__ABI_CUH */
