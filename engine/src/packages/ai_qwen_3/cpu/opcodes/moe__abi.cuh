#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MOE__ABI_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MOE__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../../nn/cpu/doors.cuh"            /* nn's doors, launched on the worker's silicon */
#include "../../../nn/cpu/expert__header.cuh"   /* where a routed expert's slot is */
#include "../../contracts/objects/moe.cuh"      /* the plane tables' places and these verbs' faults */
#include "../table.cuh"                          /* reading a table, and landing the picks */

/* ══ ⭐⭐ THE SPARSE MoE BLOCK AS ONE VERB ═════════════════════════════════════════════════════════════════
 * `(ai_qwen_3__moe h1 planes out)` — `out = h1 + MoE(h1)`, the layer's second residual included (⚖ ruled).
 * `planes` is the layer's plane table (`contracts/objects/moe.cuh`); `out` may be `h1`.
 *
 * Nine launches of nn's doors, where the verbs a program writes for the same block make sixty-two:
 *   rmsnorm · the router's gemv · top_k — the host WAITS here and reads the picks · the rotation of the
 *   normed input · every gate_up in one grouped gemv (the routed experts', the shared expert's gate and up,
 *   its gate logit) · every swiglu in one · every activation's rotation in one · the shared gate's sigmoid,
 *   into the weights beside the router's · every down in one grouped gemv, weighted, onto `h1`.
 * ⛳ THE WAIT FOR THE PICKS IS THE ONE PLACE THE HOST MUST SEE THE CARD'S ANSWER: which experts, and so
 *   which slots. In a disjoint setup it is also where the activation would leave for the experts' device.
 * ⛳ The arithmetic is nn's, door for door; what changes from the verbs a program writes is only where a
 *   half is rounded between them. */

static sys__heap_node ai_qwen_3__moe__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t h1_at = 0, h1_room = 0, out_at = 0, out_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !nn__primitives__room(&argv[0], &h1_at, &h1_room) || !nn__primitives__room(&argv[2], &out_at, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t table = argv[1].args[0];
    uint64_t at[AI_QWEN_3__MOE__TABLE], room[AI_QWEN_3__MOE__TABLE], v[AI_QWEN_3__MOE__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(table, AI_QWEN_3__MOE__LAYER, AI_QWEN_3__MOE__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__MOE__HIDDEN], I = v[AI_QWEN_3__MOE__INTER], E = v[AI_QWEN_3__MOE__EXPERTS];
    const uint64_t K = v[AI_QWEN_3__MOE__TOP_K], layer = v[AI_QWEN_3__MOE__LAYER], type = v[AI_QWEN_3__MOE__EXPERT_TYPE];
    /* K routed + the shared gate, up and logit are one table of groups; K routed + the shared down another */
    if (H == 0u || I == 0u || E == 0u || K == 0u || K > E || K > NN__VECTOR__TOP_K_MAX || K + 3u > NN__TURBOQUANT__GROUPS_MAX
     || H > (1ull << 24) || I > (1ull << 24) || E > (1ull << 24)
     || !nn__primitives__fits(H, h1_room) || !nn__primitives__fits(H, out_room) || !nn__primitives__fits(H, room[AI_QWEN_3__MOE__NORM]))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);

    /* the scratch, cut in halves: hm · hmrot · logits · weights · gate_ups · activations · rotated ones */
    const uint64_t n_hm = H, n_logits = E, n_w = 16u, n_gu = (K + 1u) * 2u * I + 2u, n_act = (K + 1u) * I;
    const uint64_t need = 2u * n_hm + n_logits + n_w + n_gu + 2u * n_act;
    if (!nn__primitives__fits(need, room[AI_QWEN_3__MOE__SCRATCH])) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    /* ⛳ CARD ADDRESSES STAY NUMBERS HERE and become pointers only as a door's arguments: nothing on this side
     *   reads them. */
    const uint64_t hm = at[AI_QWEN_3__MOE__SCRATCH], hmrot = hm + 2u * n_hm, logits = hmrot + 2u * n_hm;
    const uint64_t w = logits + 2u * n_logits, gu = w + 2u * n_w, act = gu + 2u * n_gu, actr = act + 2u * n_act;
    const uint64_t signs = at[AI_QWEN_3__MOE__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    /* ① the norm and the router, ② its eight — into a node array made here, which the host then reads. With the
     *   residual rotated the normed input is already what every gate_up reads, so it lands where the rotation would */
    const bool rotated = ai_qwen_3__table__zzpackage_flag(table, AI_QWEN_3__MOE__RESIDUAL_ROTATED);
    const uint64_t normed = rotated ? hmrot : hm;
    doors->rmsnorm((uint16_t*)(uintptr_t)normed, (const uint16_t*)(uintptr_t)h1_at, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__MOE__NORM], H, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)logits, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MOE__ROUTER], room[AI_QWEN_3__MOE__ROUTER],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MOE__ROUTER_LUT], room[AI_QWEN_3__MOE__ROUTER_LUT],
                           (const uint16_t*)(uintptr_t)normed, v[AI_QWEN_3__MOE__ROUTER_D], E, H, over);
    const uint64_t picks = sys__node_array__create(K);
    if (picks == 0ull) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    uint64_t chosen[NN__VECTOR__TOP_K_MAX];
    const bool read = ai_qwen_3__table__zzpackage_top_k(ctx, doors, picks, w, logits, E, K, chosen);
    (void)sys__heap_object__release(picks);
    if (!read) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);

    /* ③ the normed input rotated once, for every gate_up; ④ every gate_up in one launch */
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hmrot, (const uint16_t*)(uintptr_t)hm, (const uint16_t*)(uintptr_t)signs, H, over);
    nn__turboquant__groups up = {};
    for (uint64_t j = 0u; j < K; ++j) {
        sys__heap_node me;
        if (chosen[j] >= E || !nn__expert__slot(layer, type, chosen[j], &me) || me.args[NN__EXPERT__SLOT_AT] == 0ull
         || me.args[NN__EXPERT__SLOT_TYPE] != type) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_EXPERT);
        const uint64_t slot = me.args[NN__EXPERT__SLOT_AT];
        up.codes[j] = slot;
        up.luts[j] = slot + v[AI_QWEN_3__MOE__EXPERT_UP_LUT];
        up.rows[j] = 2u * I; up.d[j] = v[AI_QWEN_3__MOE__EXPERT_D]; up.out_at[j] = j * 2u * I;
        chosen[j] = slot;                              /* from here on, the slot: the downs need it too */
    }
    const uint64_t sd = v[AI_QWEN_3__MOE__SHARED_D];
    up.codes[K] = at[AI_QWEN_3__MOE__SHARED_GATE];      up.luts[K] = at[AI_QWEN_3__MOE__SHARED_GATE_LUT];
    up.rows[K] = I; up.d[K] = sd; up.out_at[K] = K * 2u * I;
    up.codes[K + 1u] = at[AI_QWEN_3__MOE__SHARED_UP];   up.luts[K + 1u] = at[AI_QWEN_3__MOE__SHARED_UP_LUT];
    up.rows[K + 1u] = I; up.d[K + 1u] = sd; up.out_at[K + 1u] = K * 2u * I + I;
    up.codes[K + 2u] = at[AI_QWEN_3__MOE__SHARED_LOGIT]; up.luts[K + 2u] = at[AI_QWEN_3__MOE__SHARED_LOGIT_LUT];
    up.rows[K + 2u] = 1u; up.d[K + 2u] = sd; up.out_at[K + 2u] = (K + 1u) * 2u * I;
    up.count = K + 3u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)gu, up, (const uint16_t*)(uintptr_t)hmrot, H, over);

    /* ⑤ every swiglu, ⑥ every activation rotated, ⑦ the shared gate's sigmoid beside the router's weights */
    doors->swiglu_pairs((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, K + 1u, I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)signs, (K + 1u) * I, over);
    doors->sigmoid((uint16_t*)(uintptr_t)(w + 2u * K), (const uint16_t*)(uintptr_t)(gu + 2u * (K + 1u) * 2u * I), 1u, over);

    /* ⑧ every down in one launch, weighted, onto h1 */
    nn__turboquant__groups down = {};
    for (uint64_t j = 0u; j < K; ++j) {
        down.codes[j] = chosen[j] + v[AI_QWEN_3__MOE__EXPERT_DOWN];
        down.luts[j] = chosen[j] + v[AI_QWEN_3__MOE__EXPERT_DOWN_LUT];
        down.rows[j] = H; down.d[j] = v[AI_QWEN_3__MOE__EXPERT_D]; down.out_at[j] = j;   /* its weight */
    }
    down.codes[K] = at[AI_QWEN_3__MOE__SHARED_DOWN]; down.luts[K] = at[AI_QWEN_3__MOE__SHARED_DOWN_LUT];
    down.rows[K] = H; down.d[K] = sd; down.out_at[K] = K;
    down.count = K + 1u;
    doors->turboquant_gemv_groups_sum((uint16_t*)(uintptr_t)out_at, down, (const uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)w,
                                      (const uint16_t*)(uintptr_t)h1_at, H, I, over);
    return nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__moe__zzabi_adapter, ai_qwen_3__moe__zzabi_apply)

/* ══ ⭐⭐ THE SAME BLOCK CUT AT THE DEVICE SEAMS — ⚖ *"a single word opcode for … the expert, the part before
 * the expert and the part after. this way we can do the disjointed setup for models that are too big"* ═════
 *   pre_expert    the norm, the router, its picks and weights, the normed input rotated, the whole shared
 *                 expert up to its rotated activation — the dense weights every token uses
 *   experts       the routed experts: from the hand-off and the picks to their weighted sum — the sparse
 *                 weights, which can live where there is room for them
 *   post_expert   the shared expert's down, its gate, both sums onto the residual
 * What crosses is the hand-off buffer (`AI_QWEN_3__HAND__HALVES`), the picks and the routed sum: a few KB. */

/* `(ai_qwen_3__pre_expert h1 planes hand picks)` -> `hand`. */
static sys__heap_node ai_qwen_3__pre_expert__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t h1_at = 0, h1_room = 0, hand = 0, hand_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[3].dtype) || !sys__node_array__is(argv[3].args[0])
     || !nn__primitives__room(&argv[0], &h1_at, &h1_room) || !nn__primitives__room(&argv[2], &hand, &hand_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__PRE__TABLE], room[AI_QWEN_3__PRE__TABLE], v[AI_QWEN_3__PRE__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__PRE__HIDDEN, AI_QWEN_3__PRE__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__PRE__HIDDEN], I = v[AI_QWEN_3__PRE__INTER], E = v[AI_QWEN_3__PRE__EXPERTS];
    const uint64_t K = v[AI_QWEN_3__PRE__TOP_K], sd = v[AI_QWEN_3__PRE__SHARED_D];
    if (H == 0u || I == 0u || E == 0u || K == 0u || K > E || K >= AI_QWEN_3__HAND__WEIGHTS || K > NN__VECTOR__TOP_K_MAX
     || H > (1ull << 24) || I > (1ull << 24) || E > (1ull << 24)
     || !nn__primitives__fits(H, h1_room) || !nn__primitives__fits(H, room[AI_QWEN_3__PRE__NORM])
     || !nn__primitives__fits(AI_QWEN_3__HAND__HALVES(H, I), hand_room))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    /* scratch, in halves: hm · logits · the shared gate, up and logit · its activation */
    const uint64_t need = H + E + 2u * I + 2u + I;
    if (!nn__primitives__fits(need, room[AI_QWEN_3__PRE__SCRATCH])) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    const uint64_t hm = at[AI_QWEN_3__PRE__SCRATCH], logits = hm + 2u * H, sgu = logits + 2u * E, sact = sgu + 2u * (2u * I + 2u);
    const uint64_t hmrot = hand, w = hand + 2u * H, sactr = w + 2u * AI_QWEN_3__HAND__WEIGHTS;
    const uint64_t signs = at[AI_QWEN_3__PRE__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    /* ⭐ THE RESIDUAL ROTATED: the normed input goes straight into the hand-off, where the rotated one would have */
    const bool rotated = ai_qwen_3__table__zzpackage_flag(argv[1].args[0], AI_QWEN_3__PRE__RESIDUAL_ROTATED);
    const uint64_t normed = rotated ? hmrot : hm;
    doors->rmsnorm((uint16_t*)(uintptr_t)normed, (const uint16_t*)(uintptr_t)h1_at, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__PRE__NORM], H, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)logits, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__PRE__ROUTER], room[AI_QWEN_3__PRE__ROUTER],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__PRE__ROUTER_LUT], room[AI_QWEN_3__PRE__ROUTER_LUT],
                           (const uint16_t*)(uintptr_t)normed, v[AI_QWEN_3__PRE__ROUTER_D], E, H, over);
    uint64_t chosen[NN__VECTOR__TOP_K_MAX];
    if (!ai_qwen_3__table__zzpackage_top_k(ctx, doors, argv[3].args[0], w, logits, E, K, chosen))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hmrot, (const uint16_t*)(uintptr_t)hm, (const uint16_t*)(uintptr_t)signs, H, over);
    nn__turboquant__groups sh = {};
    sh.codes[0] = at[AI_QWEN_3__PRE__SHARED_GATE];  sh.luts[0] = at[AI_QWEN_3__PRE__SHARED_GATE_LUT];  sh.rows[0] = I; sh.d[0] = sd; sh.out_at[0] = 0u;
    sh.codes[1] = at[AI_QWEN_3__PRE__SHARED_UP];    sh.luts[1] = at[AI_QWEN_3__PRE__SHARED_UP_LUT];    sh.rows[1] = I; sh.d[1] = sd; sh.out_at[1] = I;
    sh.codes[2] = at[AI_QWEN_3__PRE__SHARED_LOGIT]; sh.luts[2] = at[AI_QWEN_3__PRE__SHARED_LOGIT_LUT]; sh.rows[2] = 1u; sh.d[2] = sd; sh.out_at[2] = 2u * I;
    sh.count = 3u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)sgu, sh, (const uint16_t*)(uintptr_t)hmrot, H, over);
    doors->swiglu_pairs((uint16_t*)(uintptr_t)sact, (const uint16_t*)(uintptr_t)sgu, 1u, I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)sactr, (const uint16_t*)(uintptr_t)sact, (const uint16_t*)(uintptr_t)signs, I, over);
    doors->sigmoid((uint16_t*)(uintptr_t)(w + 2u * K), (const uint16_t*)(uintptr_t)(sgu + 2u * 2u * I), 1u, over);
    return nn__doors_answer(&argv[2]);
}

/* The experts table's disk backing — nn's node array at its place, read into what the loader takes — or none:
 * a zero there says every expert is resident. False on anything else. */
static bool ai_qwen_3__moe__zzprivate_backing(uint64_t table, nn__expert__backing* backing) {
    const nn__expert__backing none = {};
    *backing = none;
    const sys__heap_node n = sys__node_array__borrow(table, AI_QWEN_3__EXP__BACKING);
    if (n.dtype == SYS__KIND__VALUE_INT && n.args[0] == 0ull) return true;
    return sys__heap_node__carries_reference(n.dtype) && nn__expert__backing_of(n.args[0], backing);
}

/* Some of the routed experts — `n` of them, pick `which[i]` at slot `slot[i]` — into `routed = residual + Σ`:
 * their gate_ups, swiglus, rotations and downs, each weighted by its own pick's weight. */
static void ai_qwen_3__moe__zzprivate_routed(const nn__doors* doors, unsigned int* over, const uint64_t* v, uint64_t scratch,
                                              uint64_t signs, uint64_t hand, uint64_t routed, uint64_t residual,
                                              const uint64_t* slot, const uint64_t* which, uint64_t n) {
    const uint64_t H = v[AI_QWEN_3__EXP__HIDDEN], I = v[AI_QWEN_3__EXP__INTER];
    const uint64_t gu = scratch, act = gu + 2u * n * 2u * I, actr = act + 2u * n * I;
    const uint64_t hmrot = hand, w = hand + 2u * H;
    nn__turboquant__groups up = {}, down = {};
    for (uint64_t j = 0u; j < n; ++j) {
        up.codes[j] = slot[j]; up.luts[j] = slot[j] + v[AI_QWEN_3__EXP__UP_LUT];
        up.rows[j] = 2u * I; up.d[j] = v[AI_QWEN_3__EXP__EXPERT_D]; up.out_at[j] = j * 2u * I;
        down.codes[j] = slot[j] + v[AI_QWEN_3__EXP__DOWN]; down.luts[j] = slot[j] + v[AI_QWEN_3__EXP__DOWN_LUT];
        down.rows[j] = H; down.d[j] = v[AI_QWEN_3__EXP__EXPERT_D]; down.out_at[j] = which[j];   /* its pick's weight */
    }
    up.count = n; down.count = n;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)gu, up, (const uint16_t*)(uintptr_t)hmrot, H, over);
    doors->swiglu_pairs((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, n, I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)signs, n * I, over);
    doors->turboquant_gemv_groups_sum((uint16_t*)(uintptr_t)routed, down, (const uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)w,
                                      (const uint16_t*)(uintptr_t)residual, H, I, over);
}

/* `(ai_qwen_3__experts hand picks planes routed)` -> `routed`: Σ_j w[j] · down_j(swiglu(gate_up_j(hm))).
 * ⭐ AN EXPERT NOT IN MEMORY IS READ WHILE THE OTHERS COMPUTE (⚖ *"go ahead with the loader pool"*): every
 *   pick is requested first — a resident one is touched, a missing one gets a slot and its read queued —
 *   then the resident ones are computed, then the layer's reads are settled and the rest computed onto
 *   what the first ones summed. With no file every pick must already be resident. */
static sys__heap_node ai_qwen_3__experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hand = 0, hand_room = 0, routed = 0, routed_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || !nn__primitives__room(&argv[0], &hand, &hand_room) || !nn__primitives__room(&argv[3], &routed, &routed_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__EXP__TABLE], room[AI_QWEN_3__EXP__TABLE], v[AI_QWEN_3__EXP__TABLE];
    nn__expert__backing backing = {};
    if (!ai_qwen_3__table__zzpackage_read(argv[2].args[0], AI_QWEN_3__EXP__LAYER, AI_QWEN_3__EXP__BACKING, at, room, v)
     || !ai_qwen_3__moe__zzprivate_backing(argv[2].args[0], &backing))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__EXP__HIDDEN], I = v[AI_QWEN_3__EXP__INTER], E = v[AI_QWEN_3__EXP__EXPERTS];
    const uint64_t K = v[AI_QWEN_3__EXP__TOP_K], layer = v[AI_QWEN_3__EXP__LAYER], type = v[AI_QWEN_3__EXP__EXPERT_TYPE];
    const uint64_t picks = argv[1].args[0];
    if (H == 0u || I == 0u || K == 0u || K > E || K >= AI_QWEN_3__HAND__WEIGHTS || K > NN__TURBOQUANT__GROUPS_MAX
     || H > (1ull << 24) || I > (1ull << 24) || sys__node_array__length(picks) < K
     || !nn__primitives__fits(AI_QWEN_3__HAND__HALVES(H, I), hand_room) || !nn__primitives__fits(H, routed_room)
     || !nn__primitives__fits(H, room[AI_QWEN_3__EXP__ZERO]))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t need = K * 2u * I + 2u * K * I;
    if (!nn__primitives__fits(need, room[AI_QWEN_3__EXP__SCRATCH])) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    uint64_t ids[AI_QWEN_3__HAND__WEIGHTS];
    for (uint64_t j = 0u; j < K; ++j) {
        const sys__heap_node p = sys__node_array__borrow(picks, j);
        if (p.dtype != SYS__KIND__VALUE_INT || p.args[0] >= E) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_EXPERT);
        ids[j] = p.args[0];
    }
    /* ① every pick asked for: the resident ones answered, the others' reads on their way */
    uint64_t ready_slot[AI_QWEN_3__HAND__WEIGHTS], ready_which[AI_QWEN_3__HAND__WEIGHTS], later[AI_QWEN_3__HAND__WEIGHTS];
    uint64_t n_ready = 0u, n_later = 0u;
    for (uint64_t j = 0u; j < K; ++j) {
        uint64_t slot = 0ull;
        const int got = nn__expert__request(ctx->family, layer, type, ids[j], &backing, false, ids, (unsigned)K, &slot);
        if (got == NN__EXPERT__REFUSED) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_EXPERT);
        if (got == NN__EXPERT__RESIDENT) { ready_slot[n_ready] = slot; ready_which[n_ready] = j; ++n_ready; }
        else later[n_later++] = j;
    }
    /* ② the resident ones computed while the reads arrive */
    const uint64_t scratch = at[AI_QWEN_3__EXP__SCRATCH], signs = at[AI_QWEN_3__EXP__SIGNS], zero = at[AI_QWEN_3__EXP__ZERO];
    if (n_ready != 0u)
        ai_qwen_3__moe__zzprivate_routed(doors, over, v, scratch, signs, hand, routed, zero, ready_slot, ready_which, n_ready);
    /* ③ the layer's reads settled — the picked ones and any a prediction queued — and the rest computed onto them */
    if (!nn__expert__settle(layer, type)) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_EXPERT);
    if (n_later != 0u) {
        uint64_t slot[AI_QWEN_3__HAND__WEIGHTS];
        for (uint64_t k = 0u; k < n_later; ++k) {
            sys__heap_node me;
            if (!nn__expert__slot(layer, type, ids[later[k]], &me) || me.args[NN__EXPERT__SLOT_AT] == 0ull)
                return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_EXPERT);
            slot[k] = me.args[NN__EXPERT__SLOT_AT];
        }
        ai_qwen_3__moe__zzprivate_routed(doors, over, v, scratch, signs, hand, routed, n_ready != 0u ? routed : zero,
                                         slot, later, n_later);
    }
    return nn__doors_answer(&argv[3]);
}

/* `(ai_qwen_3__prefetch x pre_planes ex_planes)` -> `x`. ⭐ THE READS OF THE EXPERTS A LAYER WILL PROBABLY PICK,
 * queued before it picks them (⚖ *"and prefetch"*): the layer's router run on `x` — its own input, or an
 * earlier layer's output — and every predicted expert that is not in memory given a slot and a queued read.
 * A predicted one that is resident is touched, so it is not the next to go. Where the program calls it
 * decides the lead and the staleness; the layer's own experts verb settles what it queued. */
static sys__heap_node ai_qwen_3__prefetch__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || !nn__primitives__room(&argv[0], &x, &x_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t pat[AI_QWEN_3__PRE__TABLE], proom[AI_QWEN_3__PRE__TABLE], pv[AI_QWEN_3__PRE__TABLE];
    uint64_t eat[AI_QWEN_3__EXP__TABLE], eroom[AI_QWEN_3__EXP__TABLE], ev[AI_QWEN_3__EXP__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__PRE__HIDDEN, AI_QWEN_3__PRE__TABLE, pat, proom, pv)
     || !ai_qwen_3__table__zzpackage_read(argv[2].args[0], AI_QWEN_3__EXP__LAYER, AI_QWEN_3__EXP__BACKING, eat, eroom, ev))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    nn__expert__backing backing = {};
    if (!ai_qwen_3__moe__zzprivate_backing(argv[2].args[0], &backing)) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = pv[AI_QWEN_3__PRE__HIDDEN], E = pv[AI_QWEN_3__PRE__EXPERTS], K = pv[AI_QWEN_3__PRE__TOP_K];
    if (H == 0u || E == 0u || K == 0u || K > E || K > NN__VECTOR__TOP_K_MAX || H > (1ull << 24) || E > (1ull << 24)
     || !nn__primitives__fits(H, x_room) || !nn__primitives__fits(H + E + 16u, proom[AI_QWEN_3__PRE__SCRATCH]))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    if (backing.file == 0ull) return nn__doors_answer(&argv[0]);            /* all resident: nothing to read */
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t hm = pat[AI_QWEN_3__PRE__SCRATCH], logits = hm + 2u * H, w = logits + 2u * E;
    doors->rmsnorm((uint16_t*)(uintptr_t)hm, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)pat[AI_QWEN_3__PRE__NORM], H, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)logits, (const uint8_t*)(uintptr_t)pat[AI_QWEN_3__PRE__ROUTER], proom[AI_QWEN_3__PRE__ROUTER],
                           (const uint8_t*)(uintptr_t)pat[AI_QWEN_3__PRE__ROUTER_LUT], proom[AI_QWEN_3__PRE__ROUTER_LUT],
                           (const uint16_t*)(uintptr_t)hm, pv[AI_QWEN_3__PRE__ROUTER_D], E, H, over);
    const uint64_t guess = sys__node_array__create(K);
    if (guess == 0ull) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    uint64_t chosen[NN__VECTOR__TOP_K_MAX];
    const bool read = ai_qwen_3__table__zzpackage_top_k(ctx, doors, guess, w, logits, E, K, chosen);
    (void)sys__heap_object__release(guess);
    if (!read) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t layer = ev[AI_QWEN_3__EXP__LAYER], type = ev[AI_QWEN_3__EXP__EXPERT_TYPE];
    for (uint64_t j = 0u; j < K; ++j)
        if (chosen[j] < E) (void)nn__expert__request(ctx->family, layer, type, chosen[j], &backing, true, chosen, (unsigned)K, 0);
    return nn__doors_answer(&argv[0]);
}

/* `(ai_qwen_3__post_expert h1 hand routed planes out)` -> `out = h1 + routed + gate · shared_down(act)`. */
static sys__heap_node ai_qwen_3__post_expert__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t h1 = 0, h1_room = 0, hand = 0, hand_room = 0, routed = 0, routed_room = 0, out = 0, out_room = 0;
    if (!sys__heap_node__carries_reference(argv[3].dtype) || !sys__node_array__is(argv[3].args[0])
     || !nn__primitives__room(&argv[0], &h1, &h1_room) || !nn__primitives__room(&argv[1], &hand, &hand_room)
     || !nn__primitives__room(&argv[2], &routed, &routed_room) || !nn__primitives__room(&argv[4], &out, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__POST__TABLE], room[AI_QWEN_3__POST__TABLE], v[AI_QWEN_3__POST__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[3].args[0], AI_QWEN_3__POST__HIDDEN, AI_QWEN_3__POST__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__POST__HIDDEN], I = v[AI_QWEN_3__POST__INTER], K = v[AI_QWEN_3__POST__TOP_K];
    if (H == 0u || I == 0u || K >= AI_QWEN_3__HAND__WEIGHTS || H > (1ull << 24) || I > (1ull << 24)
     || !nn__primitives__fits(H, h1_room) || !nn__primitives__fits(H, routed_room) || !nn__primitives__fits(H, out_room)
     || !nn__primitives__fits(AI_QWEN_3__HAND__HALVES(H, I), hand_room))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t w = hand + 2u * H, sactr = w + 2u * AI_QWEN_3__HAND__WEIGHTS;
    nn__turboquant__groups sh = {};
    sh.codes[0] = at[AI_QWEN_3__POST__SHARED_DOWN]; sh.luts[0] = at[AI_QWEN_3__POST__SHARED_DOWN_LUT];
    sh.rows[0] = H; sh.d[0] = v[AI_QWEN_3__POST__SHARED_D]; sh.count = 1u;
    doors->vector_add((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)routed, H, over);
    doors->turboquant_gemv_groups_sum((uint16_t*)(uintptr_t)out, sh, (const uint16_t*)(uintptr_t)sactr, (const uint16_t*)(uintptr_t)(w + 2u * K),
                                      (const uint16_t*)(uintptr_t)out, H, I, over);
    return nn__doors_answer(&argv[4]);
}

/* ══ ⭐⭐ THE ROUTED EXPERTS SPLIT BY OUTPUT ROWS — ⚖ *"ok go ahead with the output row split"* ═══════════════════
 * Part `s` of `P` holds, of every expert, I/P of its gate_up's pairs and H/P of its down's rows (▶ `EXP__PART`).
 * The down cannot split its input: it reads the activation rotated over all of I, so every part needs every part's
 * activations. Two verbs, and between them the program moves this part's activations to the others:
 *   experts_up     `(ai_qwen_3__experts_up hand picks planes act)` -> `act`: K × I halves, this part's I/P of each
 *                  expert's activations in their places and ZERO everywhere else
 *   experts_down   `(ai_qwen_3__experts_down hand picks planes act other routed)` -> `routed`: H halves, this
 *                  part's H/P rows of Σ_j w[j] · down_j(rotate(act + other)) in their places and ZERO elsewhere
 * ⭐ THE ZEROS ARE THE GATHER: the parts' activations are added rather than interleaved, and the parts' routed sums
 *   too, so every crossing is a whole buffer and no door needs a stride. Adding a zero is exact, and every row is
 *   the row the one-worker verb computes, so the parts together give its answer bit for bit.
 * ⛳ Every pick must be resident: the split is for experts held in memory (the loader's reads are one pool). */
typedef struct ai_qwen_3__moe__part {
    uint64_t H, I, K, P, s, n, d;               /* n = I / P */
    uint64_t up_lut, down, down_lut;            /* offsets inside a part's slot */
    uint64_t scratch, scratch_room, signs, zero;
    uint64_t slot[AI_QWEN_3__HAND__WEIGHTS];
} ai_qwen_3__moe__part;

/* The part's table and picks read, and every pick's slot found. Zero when all is well, else the fault. */
static uint64_t ai_qwen_3__moe__zzprivate_part(const sys__heap_node* argv, ai_qwen_3__moe__part* p) {
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0]))
        return SYS__OPCODES__FAULT_TYPE;
    uint64_t at[AI_QWEN_3__EXP__PART_TABLE], room[AI_QWEN_3__EXP__PART_TABLE], v[AI_QWEN_3__EXP__PART_TABLE];
    /* ⛳ BACKING IS READ AS AN INTEGER HERE, AND MUST BE 0: every expert resident */
    if (!ai_qwen_3__table__zzpackage_read(argv[2].args[0], AI_QWEN_3__EXP__LAYER, AI_QWEN_3__EXP__PART_TABLE, at, room, v))
        return AI_QWEN_3__MOE__FAULT_TABLE;
    p->H = v[AI_QWEN_3__EXP__HIDDEN]; p->I = v[AI_QWEN_3__EXP__INTER]; p->K = v[AI_QWEN_3__EXP__TOP_K];
    p->P = v[AI_QWEN_3__EXP__PARTS]; p->s = v[AI_QWEN_3__EXP__PART]; p->d = v[AI_QWEN_3__EXP__EXPERT_D];
    p->up_lut = v[AI_QWEN_3__EXP__UP_LUT]; p->down = v[AI_QWEN_3__EXP__DOWN]; p->down_lut = v[AI_QWEN_3__EXP__DOWN_LUT];
    const uint64_t E = v[AI_QWEN_3__EXP__EXPERTS], layer = v[AI_QWEN_3__EXP__LAYER], type = v[AI_QWEN_3__EXP__EXPERT_TYPE];
    const uint64_t picks = argv[1].args[0];
    if (v[AI_QWEN_3__EXP__BACKING] != 0ull
     || p->H == 0u || p->I == 0u || p->K == 0u || p->K > E || p->K >= AI_QWEN_3__HAND__WEIGHTS || p->K > NN__TURBOQUANT__GROUPS_MAX
     || p->H > (1ull << 24) || p->I > (1ull << 24) || p->P == 0u || p->s >= p->P || p->I % p->P != 0u || p->H % p->P != 0u
     || sys__node_array__length(picks) < p->K || !nn__primitives__fits(p->H, room[AI_QWEN_3__EXP__ZERO]))
        return AI_QWEN_3__MOE__FAULT_TABLE;
    p->n = p->I / p->P;
    p->scratch = at[AI_QWEN_3__EXP__SCRATCH]; p->scratch_room = room[AI_QWEN_3__EXP__SCRATCH];
    p->signs = at[AI_QWEN_3__EXP__SIGNS]; p->zero = at[AI_QWEN_3__EXP__ZERO];
    for (uint64_t j = 0u; j < p->K; ++j) {
        const sys__heap_node pick = sys__node_array__borrow(picks, j);
        sys__heap_node me;
        if (pick.dtype != SYS__KIND__VALUE_INT || pick.args[0] >= E || !nn__expert__slot(layer, type, pick.args[0], &me)
         || me.args[NN__EXPERT__SLOT_AT] == 0ull || me.args[NN__EXPERT__SLOT_TYPE] != type)
            return AI_QWEN_3__MOE__FAULT_EXPERT;
        p->slot[j] = me.args[NN__EXPERT__SLOT_AT];
    }
    return 0ull;
}

static sys__heap_node ai_qwen_3__experts_up__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hand = 0, hand_room = 0, act = 0, act_room = 0;
    if (!nn__primitives__room(&argv[0], &hand, &hand_room) || !nn__primitives__room(&argv[3], &act, &act_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_3__moe__part p;
    const uint64_t why = ai_qwen_3__moe__zzprivate_part(argv, &p);
    if (why != 0ull) return sys__engine__abi__error(why);
    if (!nn__primitives__fits(AI_QWEN_3__HAND__HALVES(p.H, p.I), hand_room) || !nn__primitives__fits(p.K * p.I, act_room))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    if (!nn__primitives__fits(p.K * 2u * p.I, p.scratch_room)) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    /* every expert's gate_up laid out as P pairs of (I/P gates, I/P ups), this part's pair written and the rest zero —
     * so one swiglu over K·P pairs puts each activation where the whole expert's would be */
    const uint64_t gu = p.scratch;
    doors->vector_zero((uint16_t*)(uintptr_t)gu, p.K * 2u * p.I);
    nn__turboquant__groups up = {};
    for (uint64_t j = 0u; j < p.K; ++j) {
        up.codes[j] = p.slot[j]; up.luts[j] = p.slot[j] + p.up_lut;
        up.rows[j] = 2u * p.n; up.d[j] = p.d; up.out_at[j] = j * 2u * p.I + p.s * 2u * p.n;
    }
    up.count = p.K;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)gu, up, (const uint16_t*)(uintptr_t)hand, p.H, over);
    doors->swiglu_pairs((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, p.K * p.P, p.n, over);
    return nn__doors_answer(&argv[3]);
}

static sys__heap_node ai_qwen_3__experts_down__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hand = 0, hand_room = 0, act = 0, act_room = 0, other = 0, other_room = 0, routed = 0, routed_room = 0;
    if (!nn__primitives__room(&argv[0], &hand, &hand_room) || !nn__primitives__room(&argv[3], &act, &act_room)
     || !nn__primitives__room(&argv[4], &other, &other_room) || !nn__primitives__room(&argv[5], &routed, &routed_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_3__moe__part p;
    const uint64_t why = ai_qwen_3__moe__zzprivate_part(argv, &p);
    if (why != 0ull) return sys__engine__abi__error(why);
    if (!nn__primitives__fits(AI_QWEN_3__HAND__HALVES(p.H, p.I), hand_room) || !nn__primitives__fits(p.K * p.I, act_room)
     || !nn__primitives__fits(p.K * p.I, other_room) || !nn__primitives__fits(p.H, routed_room))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    if (!nn__primitives__fits(2u * p.K * p.I, p.scratch_room)) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t whole = p.scratch, actr = whole + 2u * p.K * p.I, w = hand + 2u * p.H;
    const uint64_t rows = p.H / p.P, first = p.s * rows;
    /* ⛳ With more than two parts `other` is the others' activations already summed — the program adds them. */
    doors->vector_add((uint16_t*)(uintptr_t)whole, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)other, p.K * p.I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)whole, (const uint16_t*)(uintptr_t)p.signs,
                           p.K * p.I, over);
    doors->vector_zero((uint16_t*)(uintptr_t)routed, p.H);
    nn__turboquant__groups down = {};
    for (uint64_t j = 0u; j < p.K; ++j) {
        down.codes[j] = p.slot[j] + p.down; down.luts[j] = p.slot[j] + p.down_lut;
        down.rows[j] = rows; down.d[j] = p.d; down.out_at[j] = j;   /* its pick's weight */
    }
    down.count = p.K;
    doors->turboquant_gemv_groups_sum((uint16_t*)(uintptr_t)(routed + 2u * first), down, (const uint16_t*)(uintptr_t)actr,
                                      (const uint16_t*)(uintptr_t)w, (const uint16_t*)(uintptr_t)(p.zero + 2u * first),
                                      rows, p.I, over);
    return nn__doors_answer(&argv[5]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__pre_expert__zzabi_adapter,  ai_qwen_3__pre_expert__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__experts__zzabi_adapter,     ai_qwen_3__experts__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__post_expert__zzabi_adapter, ai_qwen_3__post_expert__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__prefetch__zzabi_adapter,    ai_qwen_3__prefetch__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__experts_up__zzabi_adapter,   ai_qwen_3__experts_up__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__experts_down__zzabi_adapter, ai_qwen_3__experts_down__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MOE__ABI_CUH */
