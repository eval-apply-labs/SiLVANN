#ifndef SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_EXPERTS__ABI_CUH
#define SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_EXPERTS__ABI_CUH
/* ══ INKLING 2's ROUTED EXPERTS ════════════════════════════════════════════════════════════════════════════════════
 * The routine is nn's (`nn__routed__position`: the picks' gate-and-ups, the activation, its rotation, the downs weighed
 * into the sum, the experts not in memory read while the rest compute); the verb is the model's — it reads the model's
 * table into nn's `nn__routed`, and what it keeps of its own is the hand's layout and the activation: SwiGLU soft-capped,
 * `gate_cap·tanh(g / gate_cap) · σ(g) · up_cap·tanh(u / up_cap)`. ▶ `contracts/objects/experts.cuh`. A card's tier of
 * them (▶ nn's `nn__expert_tier`): the card computes the picks it holds, the CPU the rest, as Qwen's and GLM's. */

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

/* The experts table `table` into nn's `nn__routed`, its shape checked; 0, or the fault. */
static uint64_t ai_thinkingmachines__inkling_2__experts__zzprivate_read(uint64_t table, nn__routed* rt) {
    uint64_t at[AI_THINKINGMACHINES__INKLING_2__EXP__TABLE], room[AI_THINKINGMACHINES__INKLING_2__EXP__TABLE], v[AI_THINKINGMACHINES__INKLING_2__EXP__TABLE];
    float r[AI_THINKINGMACHINES__INKLING_2__EXP__TABLE];
    if (!nn__primitives__table(table, AI_THINKINGMACHINES__INKLING_2__EXP__PLANES, AI_THINKINGMACHINES__INKLING_2__EXP__GATE_CAP, AI_THINKINGMACHINES__INKLING_2__EXP__TABLE, at, room, v, r))
        return AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE;
    const uint64_t H = v[AI_THINKINGMACHINES__INKLING_2__EXP__HIDDEN], I = v[AI_THINKINGMACHINES__INKLING_2__EXP__INTER], K = v[AI_THINKINGMACHINES__INKLING_2__EXP__TOP_K], E = v[AI_THINKINGMACHINES__INKLING_2__EXP__EXPERTS];
    if (H == 0u || I == 0u || K == 0u || K > E || K > AI_THINKINGMACHINES__INKLING_2__HAND__WEIGHTS || K > NN__TURBOQUANT__GROUPS_MAX
     || H > (1ull << 24) || I > (1ull << 24) || v[AI_THINKINGMACHINES__INKLING_2__EXP__INT8] > 1u
     || !(r[AI_THINKINGMACHINES__INKLING_2__EXP__GATE_CAP] > 0.0f) || !(r[AI_THINKINGMACHINES__INKLING_2__EXP__UP_CAP] > 0.0f)
     || !nn__primitives__fits(512u, room[AI_THINKINGMACHINES__INKLING_2__EXP__SIGNS]) || !nn__primitives__fits(H, room[AI_THINKINGMACHINES__INKLING_2__EXP__ZERO]))
        return AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE;
    const nn__routed none = {};
    *rt = none;
    rt->layer = v[AI_THINKINGMACHINES__INKLING_2__EXP__LAYER]; rt->type = v[AI_THINKINGMACHINES__INKLING_2__EXP__TYPE];
    rt->hidden = H; rt->inter = I; rt->experts = E; rt->top_k = K; rt->bits = v[AI_THINKINGMACHINES__INKLING_2__EXP__D];
    rt->up_lut = v[AI_THINKINGMACHINES__INKLING_2__EXP__UP_LUT]; rt->down = v[AI_THINKINGMACHINES__INKLING_2__EXP__DOWN]; rt->down_lut = v[AI_THINKINGMACHINES__INKLING_2__EXP__DOWN_LUT];
    rt->first = 0u; rt->count = E; rt->int8 = v[AI_THINKINGMACHINES__INKLING_2__EXP__INT8] == 1u;
    rt->gate_cap = r[AI_THINKINGMACHINES__INKLING_2__EXP__GATE_CAP]; rt->up_cap = r[AI_THINKINGMACHINES__INKLING_2__EXP__UP_CAP];
    rt->signs = at[AI_THINKINGMACHINES__INKLING_2__EXP__SIGNS]; rt->zero = at[AI_THINKINGMACHINES__INKLING_2__EXP__ZERO];
    rt->scratch = at[AI_THINKINGMACHINES__INKLING_2__EXP__SCRATCH]; rt->scratch_room = room[AI_THINKINGMACHINES__INKLING_2__EXP__SCRATCH];
    rt->hand_bytes = AI_THINKINGMACHINES__INKLING_2__HAND__ROW(H, K); rt->picks_at = AI_THINKINGMACHINES__INKLING_2__HAND__PICKS(H); rt->batch = AI_THINKINGMACHINES__INKLING_2__EXP__BATCH;
    return ai_thinkingmachines__inkling_2__experts__zzprivate_backing(table, &rt->backing) ? 0u : AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE;
}

/* A card's tier (▶ nn's `nn__expert_tier`): true with `tier` read as layer `layer` reads it where the verb was handed the
 * binding at `argv[at]`, false where it was not; `ok` false when it was and holds no tier for this worker. */
static bool ai_thinkingmachines__inkling_2__experts__zzprivate_tier(const sys__heap_node* argv, unsigned argc, unsigned at, uint64_t layer,
                                                                   nn__expert_tier* tier, bool* ok) {
    *ok = true;
    if (argc <= at) return false;
    *ok = sys__heap_node__carries_reference(argv[at].dtype) && nn__expert_tier__of(argv[at].args[0], layer, tier);
    return *ok;
}

/* `(ai_thinkingmachines__inkling_2__experts hand picks table routed [nn__expert_tier [1]])` -> `routed`: Σ w_j ·
 * down_j(act(gate_up_j(x))) over the position's picks this worker holds, `x` the hand's input and `w` its weights, the
 * activation soft-capped by the table's two caps — as nn runs a position's routed experts. Without a tier every expert
 * is this worker's, but a pick a card's tier marked held. With one (▶ nn's `nn__expert_tier`), the picks the card's tier
 * holds, computed and marked held in `picks` for the CPU, the rest left to it; with 1 after the tier, its visit alone —
 * the picks marked and the card waited for, nothing computed, so the CPU starts on its share before the card's experts
 * are queued ahead of its copy of the hand (▶ inkling2.lisp `experts`) — answering how many picks it left the CPU, so
 * a position whose every pick the card holds hands the CPU nothing. */
static sys__heap_node ai_thinkingmachines__inkling_2__experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u && argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    const bool visit_only = argc == 6u;
    if (visit_only && (argv[5].dtype != SYS__KIND__VALUE_INT || argv[5].args[0] != 1u))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t hand = 0, h_room = 0, routed = 0, r_room = 0;
    if (!nn__primitives__room(&argv[0], &hand, &h_room) || !nn__primitives__room(&argv[3], &routed, &r_room)
     || !sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    nn__routed rt;
    const uint64_t read = ai_thinkingmachines__inkling_2__experts__zzprivate_read(argv[2].args[0], &rt);
    if (read != 0u) return sys__engine__abi__error(read);
    if (sys__node_array__length(argv[1].args[0]) < rt.top_k
     || !nn__primitives__fits(AI_THINKINGMACHINES__INKLING_2__HAND__HALVES(rt.hidden), h_room) || !nn__primitives__fits(rt.hidden, r_room))
        return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE);
    nn__expert_tier tier;
    bool tier_ok = true;
    const bool holds = ai_thinkingmachines__inkling_2__experts__zzprivate_tier(argv, argc, 4u, rt.layer, &tier, &tier_ok);
    if (!tier_ok || (visit_only && !holds)) return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE);
    if (visit_only) {
        const uint64_t f = nn__routed__visit(ctx, &rt, argv[1].args[0], &tier);
        if (f != 0u) return sys__engine__abi__error(f);
        if (!sys__gpu__compute_completed(ctx->family)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
        uint64_t left = 0u;                                /* the picks not marked held: the CPU's (▶ nn's `nn__routed__visit`) */
        for (uint64_t j = 0u; j < rt.top_k; ++j) {
            const sys__heap_node n = sys__node_array__borrow(argv[1].args[0], j);
            if (n.dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
            left += n.args[0] < rt.experts ? 1u : 0u;
        }
        sys__heap_node answer = sys__heap_node__nothing();
        answer.dtype = SYS__KIND__VALUE_INT;
        answer.args[0] = left;
        return answer;
    }
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    nn__routed__done done;
    const uint64_t f = nn__routed__position(ctx, doors, &rt, hand, argv[1].args[0], holds ? &tier : 0, routed, &done);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[3]);
}

/* `(ai_thinkingmachines__inkling_2__experts_rows hands n table routed [nn__expert_tier])` -> `routed`: the same for a
 * chunk's `n` positions, `hands` a HAND__ROW a position with its picks in it, `routed` a row a position — nn's rows
 * routine, every expert the chunk picked run once over the rows that picked it. With a tier, the picks its note marked
 * held (▶ nn's `nn__expert_tier__note`) — the card's; without one, the rest. */
static sys__heap_node ai_thinkingmachines__inkling_2__experts_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hands = 0, h_room = 0, routed = 0, r_room = 0;
    if (!nn__primitives__room(&argv[0], &hands, &h_room) || !nn__primitives__room(&argv[3], &routed, &r_room)
     || argv[1].dtype != SYS__KIND__VALUE_INT
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    nn__routed rt;
    const uint64_t read = ai_thinkingmachines__inkling_2__experts__zzprivate_read(argv[2].args[0], &rt);
    if (read != 0u) return sys__engine__abi__error(read);
    const uint64_t n = argv[1].args[0];
    if (n == 0u || n > AI_THINKINGMACHINES__INKLING_2__ROWS_MAX || n * rt.hand_bytes > h_room || !nn__primitives__fits(n * rt.hidden, r_room))
        return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE);
    nn__expert_tier tier;
    bool tier_ok = true;
    const bool holds = ai_thinkingmachines__inkling_2__experts__zzprivate_tier(argv, argc, 4u, rt.layer, &tier, &tier_ok);
    if (!tier_ok) return sys__engine__abi__error(AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t f = nn__routed__rows(ctx, doors, &rt, hands, n, holds ? &tier : 0, routed, 0, 0);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[3]);
}

SYS__ENGINE__ABI__BRIDGE(ai_thinkingmachines__inkling_2__experts__zzabi_adapter, ai_thinkingmachines__inkling_2__experts__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_thinkingmachines__inkling_2__experts_rows__zzabi_adapter, ai_thinkingmachines__inkling_2__experts_rows__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CPU_OPCODES_EXPERTS__ABI_CUH */
