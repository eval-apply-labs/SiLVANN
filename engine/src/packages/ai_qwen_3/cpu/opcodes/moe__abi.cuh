#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MOE__ABI_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MOE__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../../nn/cpu/doors.cuh"            /* nn's doors, launched on the worker's silicon */
#include "../../../nn/cpu/expert__header.cuh"   /* where a routed expert's slot is */
#include "../../../nn/cpu/routed__header.cuh"   /* the routed experts, as nn runs them */
#include "../../contracts/objects/moe.cuh"      /* the plane tables' places and these verbs' faults */
#include "../../contracts/objects/mixer.cuh"    /* the norms' epsilon */
#include "../table.cuh"                          /* reading a table */
#include <stdlib.h>                              /* a chunk's picks, inverted on the host: malloc, free */

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

/* ⭐ THE SHARED EXPERT'S MASK, when the table carries one at `first`: for each of `n` rows, `out_t += r · w_t (v · x_t)`,
 * `x_t` its rotated activation and `w_t` its gate, each `stride` halves after the last row's. 0, or the fault. */
static uint64_t ai_qwen_3__moe__zzprivate_mask_shared(const nn__doors* doors, unsigned int* over, uint64_t table, unsigned first,
                                                      uint64_t out, uint64_t x, uint64_t w, uint64_t n, uint64_t stride,
                                                      uint64_t H, uint64_t I) {
    uint64_t mk[3], mroom[3];
    bool present = false;
    if (!ai_qwen_3__table__zzpackage_mask(table, first, mk, mroom, &present)) return AI_QWEN_3__MOE__FAULT_TABLE;
    if (!present) return 0u;
    if (!nn__primitives__fits(H, mroom[0]) || !nn__primitives__fits(I, mroom[1]) || mroom[2] < 4u * n) return AI_QWEN_3__MOE__FAULT_TABLE;
    doors->rank1_dots((float*)(uintptr_t)mk[2], (const uint16_t*)(uintptr_t)mk[1], 0, (const uint16_t*)(uintptr_t)x,
                      (const uint16_t*)(uintptr_t)w, 0, n, I, stride, stride);
    doors->rank1_add((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)mk[0], (const float*)(uintptr_t)mk[2], n, 1u, H, H, over);
    return 0u;
}

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
    if (H == 0u || I == 0u || E == 0u || K == 0u || K > E || K > AI_QWEN_3__TOP_K_MAX || K + 3u > NN__TURBOQUANT__GROUPS_MAX
     || H > (1ull << 24) || I > (1ull << 24) || E > (1ull << 24)
     || !nn__primitives__fits(H, h1_room) || !nn__primitives__fits(H, out_room) || !nn__primitives__fits(H, room[AI_QWEN_3__MOE__NORM]))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);

    /* the scratch, cut in halves: hm · hmrot · logits · weights · gate_ups · activations · rotated ones
     * ⛳ every cut on 8 halves, so each part starts on 16 bytes: a family's grouped kernels read `x` 16 bytes at a time and
     *   take a call whose `x` is not so aligned to nn's generic body. `MEASURED` on the 35B: the gate_ups' 2 halves for the
     *   shared logit left the rotated activations 4 bytes off, and every routed down ran the generic body, 186 us a layer. */
    const uint64_t n_hm = (H + 7u) / 8u * 8u, n_logits = (E + 7u) / 8u * 8u, n_w = 16u;
    const uint64_t n_gu = (K + 1u) * 2u * I + 8u, n_act = ((K + 1u) * I + 7u) / 8u * 8u;
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
    doors->rmsnorm((uint16_t*)(uintptr_t)normed, (const uint16_t*)(uintptr_t)h1_at, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__MOE__NORM], H, AI_QWEN_3__RMS_NORM_EPS, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)logits, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MOE__ROUTER], room[AI_QWEN_3__MOE__ROUTER],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MOE__ROUTER_LUT], room[AI_QWEN_3__MOE__ROUTER_LUT],
                           (const uint16_t*)(uintptr_t)normed, v[AI_QWEN_3__MOE__ROUTER_D], E, H, over);
    const uint64_t picks = sys__node_array__create(K);
    if (picks == 0ull) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    uint64_t chosen[AI_QWEN_3__TOP_K_MAX];
    uint32_t ids[AI_QWEN_3__TOP_K_MAX];
    const uint64_t landed = nn__routed__top_k(ctx, doors, picks, w, logits, 0u, 1.0f, E, K, chosen);
    (void)sys__heap_object__release(picks);
    if (landed != 0u) return sys__engine__abi__error(landed);

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
        ids[j] = (uint32_t)chosen[j];
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
    /* ⑨ the masks, when the table carries them: the routed experts' (one `r`, a `v` each), then the shared expert's */
    uint64_t mk[3], mroom[3];
    bool masked = false;
    if (!ai_qwen_3__table__zzpackage_mask(table, AI_QWEN_3__MOE__MASK, mk, mroom, &masked)) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    if (masked) {
        if (!nn__primitives__fits(H, mroom[0]) || !nn__primitives__fits(E * I, mroom[1]) || mroom[2] < 8u * K)
            return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
        if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)mk[2], ids, 4u * K)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
        doors->rank1_dots((float*)(uintptr_t)(mk[2] + 4u * K), (const uint16_t*)(uintptr_t)mk[1], (const uint32_t*)(uintptr_t)mk[2],
                          (const uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)w, 0, K, I, I, 1u);
        doors->rank1_add((uint16_t*)(uintptr_t)out_at, (const uint16_t*)(uintptr_t)mk[0], (const float*)(uintptr_t)(mk[2] + 4u * K),
                         1u, K, H, 0u, over);
    }
    const uint64_t f = ai_qwen_3__moe__zzprivate_mask_shared(doors, over, table, AI_QWEN_3__MOE__SHARED_MASK, out_at, actr + 2u * K * I,
                                                             w + 2u * K, 1u, 0u, H, I);
    if (f != 0u) return sys__engine__abi__error(f);
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
    if (H == 0u || I == 0u || E == 0u || K == 0u || K > E || K >= AI_QWEN_3__HAND__WEIGHTS || K > AI_QWEN_3__TOP_K_MAX
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
    doors->rmsnorm((uint16_t*)(uintptr_t)normed, (const uint16_t*)(uintptr_t)h1_at, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__PRE__NORM], H, AI_QWEN_3__RMS_NORM_EPS, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)logits, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__PRE__ROUTER], room[AI_QWEN_3__PRE__ROUTER],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__PRE__ROUTER_LUT], room[AI_QWEN_3__PRE__ROUTER_LUT],
                           (const uint16_t*)(uintptr_t)normed, v[AI_QWEN_3__PRE__ROUTER_D], E, H, over);
    uint64_t chosen[AI_QWEN_3__TOP_K_MAX];
    const uint64_t landed = nn__routed__top_k(ctx, doors, argv[3].args[0], w, logits, 0u, 1.0f, E, K, chosen);
    if (landed != 0u) return sys__engine__abi__error(landed);
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

/* A card's tier (▶ nn's `nn__expert_tier`): true with `tier` read as layer `layer` reads it where the verb was handed the
 * binding as its last argument (`argv[at]`), false where it was not; `ok` false when it was and holds no tier for this
 * worker. */
static bool ai_qwen_3__moe__zzprivate_tier(const sys__heap_node* argv, unsigned argc, unsigned at, uint64_t layer, nn__expert_tier* tier, bool* ok) {
    *ok = true;
    if (argc <= at) return false;
    *ok = sys__heap_node__carries_reference(argv[at].dtype) && nn__expert_tier__of(argv[at].args[0], layer, tier);
    return *ok;
}

/* The routed experts as nn runs them (▶ nn's `nn__routed`), out of an experts table: its shape, its planes' places in a
 * slot, its scratch and its backing; `int8` the products' arithmetic. False when the backing cell is not one. */
static bool ai_qwen_3__moe__zzprivate_routed(uint64_t table, const uint64_t* at, const uint64_t* room, const uint64_t* v, bool int8,
                                             nn__routed* r) {
    const nn__routed none = {};
    *r = none;
    const uint64_t H = v[AI_QWEN_3__EXP__HIDDEN], I = v[AI_QWEN_3__EXP__INTER], K = v[AI_QWEN_3__EXP__TOP_K];
    r->layer = v[AI_QWEN_3__EXP__LAYER]; r->type = v[AI_QWEN_3__EXP__EXPERT_TYPE];
    r->hidden = H; r->inter = I; r->experts = v[AI_QWEN_3__EXP__EXPERTS]; r->top_k = K; r->bits = v[AI_QWEN_3__EXP__EXPERT_D];
    r->up_lut = v[AI_QWEN_3__EXP__UP_LUT]; r->down = v[AI_QWEN_3__EXP__DOWN]; r->down_lut = v[AI_QWEN_3__EXP__DOWN_LUT];
    r->first = 0u; r->count = r->experts; r->int8 = int8; r->limit = 0.0f;               /* every expert; SwiGLU as it is */
    r->signs = at[AI_QWEN_3__EXP__SIGNS]; r->zero = at[AI_QWEN_3__EXP__ZERO];
    r->scratch = at[AI_QWEN_3__EXP__SCRATCH]; r->scratch_room = room[AI_QWEN_3__EXP__SCRATCH];
    r->hand_bytes = AI_QWEN_3__HAND__ROW(H, I, K); r->picks_at = AI_QWEN_3__HAND__PICKS(H, I); r->batch = AI_QWEN_3__EXP__BATCH;
    return ai_qwen_3__moe__zzprivate_backing(table, &r->backing);
}

/* ⭐ THE MASK ON THE ROUTED EXPERTS' DOWNS, for the `n` nn just summed into `routed` (▶ `nn__routed__done`: pick
 * `which[i]`, expert `ids[which[i]]`): `routed += r · Σ_i w_i (V[expert] · actr_i)`. The experts and the weights' places go
 * to the mask's scratch — `2n` words — and the dots after them. 0, or the fault to answer with. */
static uint64_t ai_qwen_3__moe__zzprivate_mask_picks(sys__engine__ctx* ctx, const nn__doors* doors, unsigned int* over,
                                                     const uint64_t* mk, const uint64_t* mroom, const nn__routed* r,
                                                     uint64_t hand, uint64_t routed, const nn__routed__done* done) {
    const uint64_t H = r->hidden, I = r->inter, E = r->experts, n = done->n;
    const uint64_t w = hand + 2u * H, s = mk[2] + 8u * n;
    if (!nn__primitives__fits(H, mroom[0]) || !nn__primitives__fits(E * I, mroom[1]) || mroom[2] < 12u * n)
        return AI_QWEN_3__MOE__FAULT_TABLE;
    uint32_t idx[2u * AI_QWEN_3__HAND__WEIGHTS];
    for (uint64_t i = 0u; i < n; ++i) { idx[i] = (uint32_t)done->ids[done->which[i]]; idx[n + i] = (uint32_t)done->which[i]; }
    if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)mk[2], idx, 8u * n)) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    doors->rank1_dots((float*)(uintptr_t)s, (const uint16_t*)(uintptr_t)mk[1], (const uint32_t*)(uintptr_t)mk[2],
                      (const uint16_t*)(uintptr_t)done->actr, (const uint16_t*)(uintptr_t)w, (const uint32_t*)(uintptr_t)(mk[2] + 4u * n),
                      n, I, I, 0u);
    doors->rank1_add((uint16_t*)(uintptr_t)routed, (const uint16_t*)(uintptr_t)mk[0], (const float*)(uintptr_t)s, 1u, n, H, 0u, over);
    return 0u;
}

/* `(ai_qwen_3__experts hand picks planes routed [nn__expert_tier])` -> `routed`: Σ_j w[j] · down_j(swiglu(gate_up_j(hm))),
 * as nn runs a position's routed experts (▶ nn's `nn__routed__position`): an expert not in memory read while the others
 * compute, a card's tier's picks computed and marked held — and the mask added after, where the table carries one. */
static sys__heap_node ai_qwen_3__experts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hand = 0, hand_room = 0, routed = 0, routed_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || !nn__primitives__room(&argv[0], &hand, &hand_room) || !nn__primitives__room(&argv[3], &routed, &routed_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t table = argv[2].args[0];
    uint64_t at[AI_QWEN_3__EXP__TABLE], room[AI_QWEN_3__EXP__TABLE], v[AI_QWEN_3__EXP__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(table, AI_QWEN_3__EXP__LAYER, AI_QWEN_3__EXP__BACKING, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__EXP__HIDDEN], I = v[AI_QWEN_3__EXP__INTER], E = v[AI_QWEN_3__EXP__EXPERTS];
    const uint64_t K = v[AI_QWEN_3__EXP__TOP_K];
    if (H == 0u || I == 0u || K == 0u || K > E || K >= AI_QWEN_3__HAND__WEIGHTS || K > NN__TURBOQUANT__GROUPS_MAX
     || H > (1ull << 24) || I > (1ull << 24)
     || !nn__primitives__fits(AI_QWEN_3__HAND__HALVES(H, I), hand_room) || !nn__primitives__fits(H, routed_room)
     || !nn__primitives__fits(H, room[AI_QWEN_3__EXP__ZERO]))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    uint64_t mk[3], mroom[3];
    bool masked = false;
    if (!ai_qwen_3__table__zzpackage_mask(table, AI_QWEN_3__EXP__MASK, mk, mroom, &masked))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    /* the products in integers where the table says so (▶ AI_QWEN_3__EXP__INT8) */
    bool int8 = false;
    if (sys__node_array__length(table) > AI_QWEN_3__EXP__INT8) {
        const sys__heap_node f = sys__node_array__borrow(table, AI_QWEN_3__EXP__INT8);
        if (f.dtype != SYS__KIND__VALUE_INT || f.args[0] > 1u) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
        int8 = f.args[0] == 1u;
    }
    nn__routed r;
    if (!ai_qwen_3__moe__zzprivate_routed(table, at, room, v, int8, &r)) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    /* ⭐ A CARD'S TIER: the picks it holds computed here and marked held; the CPU's skip them */
    nn__expert_tier tier;
    bool tier_ok = true;
    const bool holds = ai_qwen_3__moe__zzprivate_tier(argv, argc, 4u, r.layer, &tier, &tier_ok);
    if (!tier_ok || (holds && masked)) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    nn__routed__done done;
    uint64_t f = nn__routed__position(ctx, doors, &r, hand, argv[1].args[0], holds ? &tier : 0, routed, &done);
    if (f == 0u && masked && done.n != 0u)
        f = ai_qwen_3__moe__zzprivate_mask_picks(ctx, doors, ctx->fault_word, mk, mroom, &r, hand, routed, &done);
    if (f != 0u) return sys__engine__abi__error(f);
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
    if (H == 0u || E == 0u || K == 0u || K > E || K > AI_QWEN_3__TOP_K_MAX || H > (1ull << 24) || E > (1ull << 24)
     || !nn__primitives__fits(H, x_room) || !nn__primitives__fits(H + E + 16u, proom[AI_QWEN_3__PRE__SCRATCH]))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    if (backing.file == 0ull) return nn__doors_answer(&argv[0]);            /* all resident: nothing to read */
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const uint64_t hm = pat[AI_QWEN_3__PRE__SCRATCH], logits = hm + 2u * H, w = logits + 2u * E;
    doors->rmsnorm((uint16_t*)(uintptr_t)hm, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)pat[AI_QWEN_3__PRE__NORM], H, AI_QWEN_3__RMS_NORM_EPS, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)logits, (const uint8_t*)(uintptr_t)pat[AI_QWEN_3__PRE__ROUTER], proom[AI_QWEN_3__PRE__ROUTER],
                           (const uint8_t*)(uintptr_t)pat[AI_QWEN_3__PRE__ROUTER_LUT], proom[AI_QWEN_3__PRE__ROUTER_LUT],
                           (const uint16_t*)(uintptr_t)hm, pv[AI_QWEN_3__PRE__ROUTER_D], E, H, over);
    const uint64_t guess = sys__node_array__create(K);
    if (guess == 0ull) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    uint64_t chosen[AI_QWEN_3__TOP_K_MAX];
    const uint64_t landed = nn__routed__top_k(ctx, doors, guess, w, logits, 0u, 1.0f, E, K, chosen);
    (void)sys__heap_object__release(guess);
    if (landed != 0u) return sys__engine__abi__error(landed);
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
    /* the shared expert's mask, weighted by its gate as its down is */
    const uint64_t f = ai_qwen_3__moe__zzprivate_mask_shared(doors, over, argv[3].args[0], AI_QWEN_3__POST__MASK, out, sactr, w + 2u * K,
                                                             1u, 0u, H, I);
    if (f != 0u) return sys__engine__abi__error(f);
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
    sys__node_array_walk pw;
    if (!sys__node_array__walk(picks, 0ull, &pw)) return AI_QWEN_3__MOE__FAULT_EXPERT;
    for (uint64_t j = 0u; j < p->K; ++j, sys__node_array__next(&pw)) {
        const sys__heap_node* pick = sys__node_array__walk_cell(&pw);
        sys__heap_node me;
        if (pick == 0 || pick->dtype != SYS__KIND__VALUE_INT || pick->args[0] >= E || !nn__expert__slot(layer, type, pick->args[0], &me)
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


/* ══ ⭐⭐ A PROMPT AS ROWS WITH THE EXPERTS ON THE CPU — the three words over a chunk ═════════════════════════════
 * The card routes every row into its hand row (`AI_QWEN_3__HAND__ROW`: the hand, then its picks as words) and runs its
 * shared expert's gate and up; the CPU runs each expert the chunk used ONCE over every row that picked it, reading its
 * weights once a chunk, in integers; the card closes every row. ▶ ai_glm_5_3's `experts_rows`, which this follows. */

/* `(ai_qwen_3__pre_expert_rows h1s planes hands n)` -> `hands`: `pre_expert` for each of `n` rows, its picks into its
 * hand row. */
static sys__heap_node ai_qwen_3__pre_expert_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t h1s = 0, h1_room = 0, hands = 0, hands_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &h1s, &h1_room) || !nn__primitives__room(&argv[2], &hands, &hands_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__PRE__TABLE], room[AI_QWEN_3__PRE__TABLE], v[AI_QWEN_3__PRE__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__PRE__HIDDEN, AI_QWEN_3__PRE__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__PRE__HIDDEN], I = v[AI_QWEN_3__PRE__INTER], E = v[AI_QWEN_3__PRE__EXPERTS];
    const uint64_t K = v[AI_QWEN_3__PRE__TOP_K], sd = v[AI_QWEN_3__PRE__SHARED_D], n = argv[3].args[0];
    const uint64_t row = AI_QWEN_3__HAND__ROW(H, I, K);
    if (H == 0u || I == 0u || E == 0u || K == 0u || K > E || K >= AI_QWEN_3__HAND__WEIGHTS || K > AI_QWEN_3__TOP_K_MAX
     || H > (1ull << 24) || I > (1ull << 24) || E > (1ull << 24) || n == 0u || n > AI_QWEN_3__ROWS_MAX
     || !nn__primitives__fits(n * H, h1_room) || !nn__primitives__fits(H, room[AI_QWEN_3__PRE__NORM]) || hands_room / row < n)
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t need = H + E + 2u * I + 2u + I;
    if (!nn__primitives__fits(need, room[AI_QWEN_3__PRE__SCRATCH])) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    const uint64_t hm = at[AI_QWEN_3__PRE__SCRATCH], logits = hm + 2u * H, sgu = logits + 2u * E, sact = sgu + 2u * (2u * I + 2u);
    const uint64_t signs = at[AI_QWEN_3__PRE__SIGNS];
    const bool rotated = ai_qwen_3__table__zzpackage_flag(argv[1].args[0], AI_QWEN_3__PRE__RESIDUAL_ROTATED);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    nn__turboquant__groups sh = {};
    sh.codes[0] = at[AI_QWEN_3__PRE__SHARED_GATE];  sh.luts[0] = at[AI_QWEN_3__PRE__SHARED_GATE_LUT];  sh.rows[0] = I; sh.d[0] = sd; sh.out_at[0] = 0u;
    sh.codes[1] = at[AI_QWEN_3__PRE__SHARED_UP];    sh.luts[1] = at[AI_QWEN_3__PRE__SHARED_UP_LUT];    sh.rows[1] = I; sh.d[1] = sd; sh.out_at[1] = I;
    sh.codes[2] = at[AI_QWEN_3__PRE__SHARED_LOGIT]; sh.luts[2] = at[AI_QWEN_3__PRE__SHARED_LOGIT_LUT]; sh.rows[2] = 1u; sh.d[2] = sd; sh.out_at[2] = 2u * I;
    sh.count = 3u;
    for (uint64_t t = 0u; t < n; ++t) {
        const uint64_t h1 = h1s + 2u * t * H, hand = hands + t * row;
        const uint64_t hmrot = hand, w = hand + 2u * H, sactr = w + 2u * AI_QWEN_3__HAND__WEIGHTS;
        const uint64_t normed = rotated ? hmrot : hm;
        doors->rmsnorm((uint16_t*)(uintptr_t)normed, (const uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__PRE__NORM], H, AI_QWEN_3__RMS_NORM_EPS, over);
        doors->turboquant_gemv((uint16_t*)(uintptr_t)logits, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__PRE__ROUTER], room[AI_QWEN_3__PRE__ROUTER],
                               (const uint8_t*)(uintptr_t)at[AI_QWEN_3__PRE__ROUTER_LUT], room[AI_QWEN_3__PRE__ROUTER_LUT],
                               (const uint16_t*)(uintptr_t)normed, v[AI_QWEN_3__PRE__ROUTER_D], E, H, over);
        doors->vector_top_k((uint64_t*)(uintptr_t)(hand + AI_QWEN_3__HAND__PICKS(H, I)), 1ull, (uint16_t*)(uintptr_t)w,
                            (const uint16_t*)(uintptr_t)logits, E, K);
        if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hmrot, (const uint16_t*)(uintptr_t)hm, (const uint16_t*)(uintptr_t)signs, H, over);
        doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)sgu, sh, (const uint16_t*)(uintptr_t)hmrot, H, over);
        doors->swiglu_pairs((uint16_t*)(uintptr_t)sact, (const uint16_t*)(uintptr_t)sgu, 1u, I, over);
        doors->hadamard_rotate((uint16_t*)(uintptr_t)sactr, (const uint16_t*)(uintptr_t)sact, (const uint16_t*)(uintptr_t)signs, I, over);
        doors->sigmoid((uint16_t*)(uintptr_t)(w + 2u * K), (const uint16_t*)(uintptr_t)(sgu + 2u * 2u * I), 1u, over);
    }
    return nn__doors_answer(&argv[2]);
}

/* `(ai_qwen_3__post_expert_rows h1s hands routed planes outs n)` -> `outs`: `post_expert` for each of `n` rows. */
static sys__heap_node ai_qwen_3__post_expert_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t h1s = 0, h1_room = 0, hands = 0, hands_room = 0, routed = 0, routed_room = 0, outs = 0, out_room = 0;
    if (!sys__heap_node__carries_reference(argv[3].dtype) || !sys__node_array__is(argv[3].args[0]) || argv[5].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &h1s, &h1_room) || !nn__primitives__room(&argv[1], &hands, &hands_room)
     || !nn__primitives__room(&argv[2], &routed, &routed_room) || !nn__primitives__room(&argv[4], &outs, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__POST__TABLE], room[AI_QWEN_3__POST__TABLE], v[AI_QWEN_3__POST__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[3].args[0], AI_QWEN_3__POST__HIDDEN, AI_QWEN_3__POST__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__POST__HIDDEN], I = v[AI_QWEN_3__POST__INTER], K = v[AI_QWEN_3__POST__TOP_K], n = argv[5].args[0];
    const uint64_t row = AI_QWEN_3__HAND__ROW(H, I, K);
    if (H == 0u || I == 0u || K >= AI_QWEN_3__HAND__WEIGHTS || H > (1ull << 24) || I > (1ull << 24) || n == 0u || n > AI_QWEN_3__ROWS_MAX
     || !nn__primitives__fits(n * H, h1_room) || !nn__primitives__fits(n * H, routed_room) || !nn__primitives__fits(n * H, out_room)
     || hands_room / row < n)
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    nn__turboquant__groups sh = {};
    sh.codes[0] = at[AI_QWEN_3__POST__SHARED_DOWN]; sh.luts[0] = at[AI_QWEN_3__POST__SHARED_DOWN_LUT];
    sh.rows[0] = H; sh.d[0] = v[AI_QWEN_3__POST__SHARED_D]; sh.count = 1u;
    for (uint64_t t = 0u; t < n; ++t) {
        const uint64_t out = outs + 2u * t * H, hand = hands + t * row, w = hand + 2u * H, sactr = w + 2u * AI_QWEN_3__HAND__WEIGHTS;
        doors->vector_add((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)(h1s + 2u * t * H), (const uint16_t*)(uintptr_t)(routed + 2u * t * H),
                          H, over);
        doors->turboquant_gemv_groups_sum((uint16_t*)(uintptr_t)out, sh, (const uint16_t*)(uintptr_t)sactr, (const uint16_t*)(uintptr_t)(w + 2u * K),
                                          (const uint16_t*)(uintptr_t)out, H, I, over);
    }
    /* the shared expert's mask over every row: a row's activation and gate are a hand row after the last's */
    const uint64_t w0 = hands + 2u * H;
    const uint64_t f = ai_qwen_3__moe__zzprivate_mask_shared(doors, over, argv[3].args[0], AI_QWEN_3__POST__MASK, outs,
                                                             w0 + 2u * AI_QWEN_3__HAND__WEIGHTS, w0 + 2u * K, n, row / 2u, H, I);
    if (f != 0u) return sys__engine__abi__error(f);
    return nn__doors_answer(&argv[4]);
}

/* The routed experts' mask over a chunk's picks (▶ nn's `nn__routed__rows_mask`): a dot each, in the picks' order, spread
 * onto its pick's sum through the down pairs. */
typedef struct ai_qwen_3__moe__rows_mask { uint64_t mk[3], mroom[3], H, I, E; } ai_qwen_3__moe__rows_mask;
static uint64_t ai_qwen_3__moe__zzprivate_rows_mask(void* model, sys__engine__ctx* ctx, uint64_t P, const uint32_t* expert_of,
                                                    uint64_t actr, uint64_t weights, uint64_t down_pairs, uint64_t sums) {
    const ai_qwen_3__moe__rows_mask* m = (const ai_qwen_3__moe__rows_mask*)model;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    if (!nn__primitives__fits(m->H, m->mroom[0]) || !nn__primitives__fits(m->E * m->I, m->mroom[1]) || m->mroom[2] < 8u * P)
        return AI_QWEN_3__MOE__FAULT_TABLE;
    if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)m->mk[2], expert_of, 4u * P)) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    const uint64_t s = m->mk[2] + 4u * P;
    doors->rank1_dots((float*)(uintptr_t)s, (const uint16_t*)(uintptr_t)m->mk[1], (const uint32_t*)(uintptr_t)m->mk[2],
                      (const uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)weights, 0, P, m->I, m->I, 1u);
    doors->rank1_spread((float*)(uintptr_t)sums, (const uint16_t*)(uintptr_t)m->mk[0], (const float*)(uintptr_t)s,
                        (const uint32_t*)(uintptr_t)(down_pairs + 4u), P, m->H, 2u);
    return 0u;
}

/* `(ai_qwen_3__experts_rows hands planes routed n [nn__expert_tier])` -> `routed`: for each of `n` hand rows, Σ_j w_j · down_j(
 * swiglu(gate_up_j(x))) over its picks, a row of `routed` each, as nn runs a chunk's routed experts (▶ nn's
 * `nn__routed__rows`): EXPERT-MAJOR, each expert the chunk used run once over the rows that picked it — on the CPU in
 * integers, a card's tier's exactly. `planes` is the experts table with the chunk's scratch in place of a position's. */
static sys__heap_node ai_qwen_3__experts_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hands = 0, h_room = 0, routed = 0, r_room = 0;
    if (!nn__primitives__room(&argv[0], &hands, &h_room) || !nn__primitives__room(&argv[2], &routed, &r_room)
     || !sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0]) || argv[3].dtype != SYS__KIND__VALUE_INT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t table = argv[1].args[0];
    uint64_t at[AI_QWEN_3__EXP__TABLE], room[AI_QWEN_3__EXP__TABLE], v[AI_QWEN_3__EXP__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(table, AI_QWEN_3__EXP__LAYER, AI_QWEN_3__EXP__BACKING, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__EXP__HIDDEN], I = v[AI_QWEN_3__EXP__INTER], E = v[AI_QWEN_3__EXP__EXPERTS];
    const uint64_t K = v[AI_QWEN_3__EXP__TOP_K], n = argv[3].args[0], row = AI_QWEN_3__HAND__ROW(H, I, K);
    if (H == 0u || I == 0u || K == 0u || K > E || K >= AI_QWEN_3__HAND__WEIGHTS || H > (1ull << 24) || I > (1ull << 24)
     || n == 0u || n > AI_QWEN_3__ROWS_MAX || h_room / row < n || !nn__primitives__fits(n * H, r_room))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    ai_qwen_3__moe__rows_mask mask = {};
    bool masked = false;
    if (!ai_qwen_3__table__zzpackage_mask(table, AI_QWEN_3__EXP__MASK, mask.mk, mask.mroom, &masked))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    mask.H = H; mask.I = I; mask.E = E;
    nn__expert_tier tier;
    bool tier_ok = true;
    const bool holds = ai_qwen_3__moe__zzprivate_tier(argv, argc, 4u, v[AI_QWEN_3__EXP__LAYER], &tier, &tier_ok);
    if (!tier_ok || (holds && masked)) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    /* a prompt's rows on the CPU in integers, a card's tier's exactly (▶ AI_QWEN_3__EXP__INT8) */
    nn__routed r;
    if (!ai_qwen_3__moe__zzprivate_routed(table, at, room, v, !holds, &r)) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t f = nn__routed__rows(ctx, doors, &r, hands, n, holds ? &tier : 0, routed,
                                        masked ? ai_qwen_3__moe__zzprivate_rows_mask : 0, &mask);
    return f != 0u ? sys__engine__abi__error(f) : nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__pre_expert__zzabi_adapter,  ai_qwen_3__pre_expert__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__experts__zzabi_adapter,     ai_qwen_3__experts__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__post_expert__zzabi_adapter, ai_qwen_3__post_expert__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__prefetch__zzabi_adapter,    ai_qwen_3__prefetch__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__experts_up__zzabi_adapter,   ai_qwen_3__experts_up__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__experts_down__zzabi_adapter, ai_qwen_3__experts_down__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__pre_expert_rows__zzabi_adapter,  ai_qwen_3__pre_expert_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__experts_rows__zzabi_adapter,     ai_qwen_3__experts_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__post_expert_rows__zzabi_adapter, ai_qwen_3__post_expert_rows__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MOE__ABI_CUH */
