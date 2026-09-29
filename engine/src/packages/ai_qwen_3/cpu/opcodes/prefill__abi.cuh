#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_PREFILL__ABI_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_PREFILL__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <time.h>                                 /* the streaming's own clock */
#include "../../../nn/cpu/doors.cuh"            /* nn's doors, launched on the worker's silicon */
#include "../../../nn/cpu/expert__header.cuh"   /* where a routed expert's slot is */
#include "../../contracts/objects/mixer.cuh"    /* the mixers' tables */
#include "../../contracts/objects/moe.cuh"      /* the MoE's table, and the most rows a call takes */
#include "../table.cuh"                          /* reading a table, and landing the picks */

/* ══ ⭐⭐⭐ A PROMPT'S ROWS THROUGH A LAYER AT ONCE — THE SAME THREE WORDS, OVER `rows` POSITIONS ═════════════════
 * ⚖ *"the avx2 bodies, then gemv [GEMM]"* · *"we need expert major verbs"* · *"a for the deltanet"*. Each verb is
 * its one-position namesake with `x` and `h1` as `rows` rows of the hidden width, a row a position, over the
 * SAME plane table — so a program runs a prompt through these, then decodes from the state they leave:
 *   deltanet_rows    the projections as one expert-major GEMM over every row, the conv and the recurrence
 *                    walking the positions in order inside one launch each
 *   attention_rows   keys and values for positions `first ..` into the cache first, then every row's query
 *                    against the cache up to its own position
 *   moe_rows         the router over every row, the picks read once, then each expert used by the batch
 *                    applied to the rows that picked it — its weights read once for all of them
 * The pairs the GEMM doors take (`nn/contracts/abi/gpu.cuh`: a row of `x`, a row of `out`) are written into the
 * front of the scratch; the MoE's are its routing, inverted on the host. */

/* The front of the scratch: `count` pairs `(first_x + j, first_out + j)`, written from the host. False if the
 * card would not take them. */
static bool ai_qwen_3__prefill__zzprivate_run_pairs(sys__engine__ctx* ctx, uint64_t at, uint64_t count, uint64_t first_x,
                                                    uint64_t first_out) {
    uint32_t pairs[2u * AI_QWEN_3__ROWS_MAX];
    for (uint64_t j = 0u; j < count; ++j) { pairs[2u * j] = (uint32_t)(first_x + j); pairs[2u * j + 1u] = (uint32_t)(first_out + j); }
    return sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)at, pairs, (size_t)(8u * count));
}

/* The bytes of `rows` rows of a matrix at width `d` and `cols` columns, and of their scales. */
static uint64_t ai_qwen_3__prefill__zzprivate_codes(uint64_t d, uint64_t rows, uint64_t cols) {
    return rows * nn__turboquant__row_bytes(d, cols);
}

/* `(ai_qwen_3__deltanet_rows x planes h1 rows)` -> `h1`, `rows` positions after the state's. */
static sys__heap_node ai_qwen_3__deltanet_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, h1 = 0, h1_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[2], &h1, &h1_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__DN__TABLE], room[AI_QWEN_3__DN__TABLE], v[AI_QWEN_3__DN__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__DN__HIDDEN, AI_QWEN_3__DN__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__DN__HIDDEN], kh = v[AI_QWEN_3__DN__K_HEADS], vh = v[AI_QWEN_3__DN__V_HEADS];
    const uint64_t hd = v[AI_QWEN_3__DN__HEAD_DIM], T = argv[3].args[0];
    if (H == 0u || kh == 0u || vh == 0u || hd == 0u || vh % kh != 0u || H > (1ull << 24) || vh > (1ull << 16) || hd > (1ull << 16)
     || T == 0u || T > AI_QWEN_3__ROWS_MAX || !nn__primitives__fits(T * H, x_room) || !nn__primitives__fits(T * H, h1_room))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t vdim = vh * hd, ch = 2u * kh * hd + vdim;
    /* scratch: the pairs, then in halves h · hrot · [mixed · z · a · b] · beta · g · conved · core · crot · ao, a row each */
    const uint64_t pairs = at[AI_QWEN_3__DN__SCRATCH], h = pairs + 8u * AI_QWEN_3__ROWS_MAX;
    const uint64_t hrot = h + 2u * T * H, mixed = hrot + 2u * T * H, z = mixed + 2u * T * ch, av = z + 2u * T * vdim;
    const uint64_t bv = av + 2u * T * vh, beta = bv + 2u * T * vh, g = beta + 2u * T * vh, conved = g + 2u * T * vh;
    const uint64_t core = conved + 2u * T * ch, crot = core + 2u * T * vdim, ao = crot + 2u * T * vdim;
    if (!nn__primitives__fits((ao + 2u * T * H - h) / 2u + 4u * AI_QWEN_3__ROWS_MAX, room[AI_QWEN_3__DN__SCRATCH]))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    const uint64_t signs = at[AI_QWEN_3__DN__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_qwen_3__prefill__zzprivate_run_pairs(ctx, pairs, T, 0u, 0u)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    const bool rotated = ai_qwen_3__table__zzpackage_flag(argv[1].args[0], AI_QWEN_3__DN__RESIDUAL_ROTATED);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)(rotated ? hrot : h), (const uint16_t*)(uintptr_t)x,
                        (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__INPUT_NORM], H, T, H, H, over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, T * H, over);
    /* the four projections, each an expert over every row, one launch, their outputs one after another */
    nn__expert__groups gr = {};
    const unsigned codes[4] = {AI_QWEN_3__DN__QKV, AI_QWEN_3__DN__Z, AI_QWEN_3__DN__A, AI_QWEN_3__DN__B};
    const unsigned widths[4] = {AI_QWEN_3__DN__D_QKV, AI_QWEN_3__DN__D_Z, AI_QWEN_3__DN__D_A, AI_QWEN_3__DN__D_B};
    const uint64_t rows[4] = {ch, vdim, vh, vh};
    uint64_t placed = 0u;
    for (unsigned i = 0u; i < 4u; ++i) {
        gr.codes[i] = at[codes[i]]; gr.luts[i] = at[codes[i] + 1u]; gr.out_rows[i] = rows[i]; gr.d[i] = v[widths[i]];
        gr.out_at[i] = placed; gr.pairs_at[i] = 0u; gr.pairs[i] = T;
        placed += T * rows[i];
    }
    gr.count = 4u;
    doors->expert_groups((uint16_t*)(uintptr_t)mixed, gr, (const uint16_t*)(uintptr_t)hrot, (const uint32_t*)(uintptr_t)pairs, H, over);
    doors->deltanet_gates_rows((uint16_t*)(uintptr_t)beta, (uint16_t*)(uintptr_t)g, (const uint16_t*)(uintptr_t)av,
                               (const uint16_t*)(uintptr_t)bv, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__A_LOG],
                               (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__DT_BIAS], vh, T, over);
    doors->deltanet_conv_steps((uint16_t*)(uintptr_t)conved, (uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__WINDOW],
                               (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__CONV_W], (const uint16_t*)(uintptr_t)mixed, ch, T, over);
    doors->deltanet_steps((float*)(uintptr_t)at[AI_QWEN_3__DN__STATE], (const uint16_t*)(uintptr_t)conved, (const uint16_t*)(uintptr_t)z,
                          (const uint16_t*)(uintptr_t)beta, (const uint16_t*)(uintptr_t)g, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__NORM_W],
                          (uint16_t*)(uintptr_t)core, kh, vh, hd, T, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)crot, (const uint16_t*)(uintptr_t)core, (const uint16_t*)(uintptr_t)signs, T * vdim, over);
    doors->expert_rows((uint16_t*)(uintptr_t)ao, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__DN__OUT], room[AI_QWEN_3__DN__OUT],
                       (const uint8_t*)(uintptr_t)at[AI_QWEN_3__DN__OUT_LUT], room[AI_QWEN_3__DN__OUT_LUT],
                       (const uint16_t*)(uintptr_t)crot, (const uint32_t*)(uintptr_t)pairs, T, v[AI_QWEN_3__DN__D_OUT], H, vdim, over);
    doors->vector_add((uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)ao, T * H, over);
    return nn__doors_answer(&argv[2]);
}

/* `(ai_qwen_3__attention_rows x planes first h1 rows)` -> `h1`: positions `first .. first + rows`. Every row's key
 * and value are written into the caches before any query reads them, and row `r` reads rows `0 .. first + r`. */
static sys__heap_node ai_qwen_3__attention_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, h1 = 0, h1_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[3], &h1, &h1_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__AT__ROWS_TABLE], room[AI_QWEN_3__AT__ROWS_TABLE], v[AI_QWEN_3__AT__ROWS_TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__AT__HIDDEN, AI_QWEN_3__AT__ROWS_TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__AT__HIDDEN], qh = v[AI_QWEN_3__AT__Q_HEADS], kvh = v[AI_QWEN_3__AT__KV_HEADS];
    const uint64_t hd = v[AI_QWEN_3__AT__HEAD_DIM], rot = v[AI_QWEN_3__AT__ROT], theta = v[AI_QWEN_3__AT__THETA];
    const uint64_t first = argv[2].args[0], T = argv[4].args[0];
    if (H == 0u || qh == 0u || kvh == 0u || hd == 0u || qh % kvh != 0u || rot == 0u || rot > hd || rot % 2u != 0u
     || H > (1ull << 24) || qh > (1ull << 12) || hd > (1ull << 12) || first >= (1ull << 32) || theta == 0u || theta >= (1ull << 32)
     || T == 0u || T > AI_QWEN_3__ROWS_MAX || !nn__primitives__fits(T * H, x_room) || !nn__primitives__fits(T * H, h1_room))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t kvw = kvh * hd, qw = qh * hd, to = first + T;
    if (!nn__primitives__fits(to * kvw, room[AI_QWEN_3__AT__K_CACHE]) || !nn__primitives__fits(to * kvw, room[AI_QWEN_3__AT__V_CACHE])
     || room[AI_QWEN_3__AT__SCORES] / 4u < T * qh * to || room[AI_QWEN_3__AT__ANGLES] / 4u < T * rot)
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    /* scratch: the query pairs, the value pairs, then in halves h · hrot · [qg · kk] · qn · att · gated · arot · ao */
    const uint64_t pairs = at[AI_QWEN_3__AT__SCRATCH], vpairs = pairs + 8u * AI_QWEN_3__ROWS_MAX, h = vpairs + 8u * AI_QWEN_3__ROWS_MAX;
    const uint64_t hrot = h + 2u * T * H, qg = hrot + 2u * T * H, kk = qg + 2u * T * 2u * qw, qn = kk + 2u * T * kvw;
    const uint64_t att = qn + 2u * T * qw, gated = att + 2u * T * qw, arot = gated + 2u * T * qw, ao = arot + 2u * T * qw;
    if (!nn__primitives__fits((ao + 2u * T * H - h) / 2u + 8u * AI_QWEN_3__ROWS_MAX, room[AI_QWEN_3__AT__SCRATCH]))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    const uint64_t kfirst = at[AI_QWEN_3__AT__K_CACHE] + 2u * first * kvw;
    const uint64_t signs = at[AI_QWEN_3__AT__SIGNS], cs = at[AI_QWEN_3__AT__ANGLES];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_qwen_3__prefill__zzprivate_run_pairs(ctx, pairs, T, 0u, 0u) || !ai_qwen_3__prefill__zzprivate_run_pairs(ctx, vpairs, T, 0u, first))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    const bool rotated = ai_qwen_3__table__zzpackage_flag(argv[1].args[0], AI_QWEN_3__AT__RESIDUAL_ROTATED);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)(rotated ? hrot : h), (const uint16_t*)(uintptr_t)x,
                        (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__INPUT_NORM], H, T, H, H, over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, T * H, over);
    nn__expert__groups gr = {};
    gr.codes[0] = at[AI_QWEN_3__AT__Q]; gr.luts[0] = at[AI_QWEN_3__AT__Q_LUT]; gr.out_rows[0] = 2u * qw; gr.d[0] = v[AI_QWEN_3__AT__D_Q];
    gr.codes[1] = at[AI_QWEN_3__AT__K]; gr.luts[1] = at[AI_QWEN_3__AT__K_LUT]; gr.out_rows[1] = kvw;     gr.d[1] = v[AI_QWEN_3__AT__D_K];
    gr.out_at[0] = 0u; gr.out_at[1] = T * 2u * qw;
    gr.pairs[0] = gr.pairs[1] = T;
    gr.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)qg, gr, (const uint16_t*)(uintptr_t)hrot, (const uint32_t*)(uintptr_t)pairs, H, over);
    /* the values straight into the cache, row `r` at position `first + r` */
    doors->expert_rows((uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__V_CACHE], (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__V], room[AI_QWEN_3__AT__V],
                       (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__V_LUT], room[AI_QWEN_3__AT__V_LUT], (const uint16_t*)(uintptr_t)hrot,
                       (const uint32_t*)(uintptr_t)vpairs, T, v[AI_QWEN_3__AT__D_V], kvw, H, over);
    doors->rope_angles_rows((float*)(uintptr_t)cs, first, (float)(uint32_t)theta, rot, T);
    /* every head's query normed and rotated; every key normed and rotated straight into its cache row */
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qg, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__Q_NORM],
                        hd, T * qh, 2u * hd, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qn, (const float*)(uintptr_t)cs, T, qh, hd, qw, rot, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)kfirst, (const uint16_t*)(uintptr_t)kk, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__K_NORM],
                        hd, T * kvh, hd, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)kfirst, (const uint16_t*)(uintptr_t)kfirst, (const float*)(uintptr_t)cs, T, kvh, hd, kvw, rot, over);
    doors->attention_causal_scores((float*)(uintptr_t)at[AI_QWEN_3__AT__SCORES], (const uint16_t*)(uintptr_t)qn,
                                   (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__K_CACHE], qh, kvh, hd, first, T);
    doors->attention_causal_mix((uint16_t*)(uintptr_t)att, (const float*)(uintptr_t)at[AI_QWEN_3__AT__SCORES],
                                (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__V_CACHE], qh, kvh, hd, first, T, over);
    doors->attention_gate_rows((uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)att, (const uint16_t*)(uintptr_t)qg, T * qh, hd, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)arot, (const uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)signs, T * qw, over);
    doors->expert_rows((uint16_t*)(uintptr_t)ao, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__O], room[AI_QWEN_3__AT__O],
                       (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__O_LUT], room[AI_QWEN_3__AT__O_LUT], (const uint16_t*)(uintptr_t)arot,
                       (const uint32_t*)(uintptr_t)pairs, T, v[AI_QWEN_3__AT__D_O], H, qw, over);
    doors->vector_add((uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)ao, T * H, over);
    return nn__doors_answer(&argv[3]);
}

/* `(ai_qwen_3__attention_tiered_rows xs planes first h1s n)` -> `h1s`: rows `first .. first + n` of a prompt through the
 * tiered cache (▶ `ai_qwen_3__attention_tiered` for the tiers). The projections as the one-cache rows verb makes them;
 * then each row's key and value rotated into its place — the sink, or its slot of the hot ring, the row it displaces
 * cooled to the warm ring first and the warm row that displaces to the cold tier — so the chunk goes in and what it
 * pushes out goes down together. The cache's fp16 positions (the sink, the hot ring in order) are gathered into one
 * window, whose last `n` rows are the chunk: a row attends the window up to its own position, and the warm and cold
 * tiers whole, merged as the one-position verb merges them. ⛳ `n` is at most the hot ring, so the chunk's rows never
 * displace one another. */
static sys__heap_node ai_qwen_3__attention_tiered_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, h1 = 0, h1_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[3], &h1, &h1_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t table = argv[1].args[0];
    uint64_t at[AI_QWEN_3__AT__TIERED_ROWS_TABLE], room[AI_QWEN_3__AT__TIERED_ROWS_TABLE], v[AI_QWEN_3__AT__TIERED_ROWS_TABLE];
    if (!ai_qwen_3__table__zzpackage_read(table, AI_QWEN_3__AT__HIDDEN, AI_QWEN_3__AT__HOT_K, at, room, v)
     || sys__node_array__length(table) < AI_QWEN_3__AT__TIERED_ROWS_TABLE)
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    for (unsigned i = AI_QWEN_3__AT__HOT_K; i < AI_QWEN_3__AT__TIERED_ROWS_TABLE; ++i) {
        const sys__heap_node n = sys__node_array__borrow(table, i);
        const bool plane = i < AI_QWEN_3__AT__SINK || i >= AI_QWEN_3__AT__ROWS_SCRATCH;
        if (plane) { if (!nn__primitives__room(&n, &at[i], &room[i])) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE); }
        else if (n.dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
        else v[i] = n.args[0];
    }
    const uint64_t H = v[AI_QWEN_3__AT__HIDDEN], qh = v[AI_QWEN_3__AT__Q_HEADS], kvh = v[AI_QWEN_3__AT__KV_HEADS];
    const uint64_t hd = v[AI_QWEN_3__AT__HEAD_DIM], rot = v[AI_QWEN_3__AT__ROT], theta = v[AI_QWEN_3__AT__THETA];
    const uint64_t SINK = v[AI_QWEN_3__AT__SINK], HOT = v[AI_QWEN_3__AT__HOT], WARM = v[AI_QWEN_3__AT__WARM];
    const uint64_t first = argv[2].args[0], T = argv[4].args[0], to = first + T;
    if (H == 0u || qh == 0u || kvh == 0u || hd == 0u || qh % kvh != 0u || rot == 0u || rot > hd || rot % 2u != 0u
     || hd > 512u || (hd & (hd - 1u)) != 0u || SINK == 0u || HOT == 0u || WARM == 0u || theta == 0u || theta >= (1ull << 32)
     || H > (1ull << 24) || qh > (1ull << 12) || first >= (1ull << 32) || T == 0u || T > AI_QWEN_3__ROWS_MAX || T > HOT
     || !nn__primitives__fits(T * H, x_room) || !nn__primitives__fits(T * H, h1_room))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t kvw = kvh * hd, qw = qh * hd;
    const uint64_t rb8 = nn__turboquant__row_bytes(8u, hd), rb4 = nn__turboquant__row_bytes(4u, hd);
    const uint64_t past = to > SINK ? to - SINK : 0u;                       /* positions past the sink, the chunk in */
    const uint64_t n_sink = to < SINK ? to : SINK, n_hot = past < HOT ? past : HOT;
    const uint64_t n_warm = past > HOT ? (past - HOT < WARM ? past - HOT : WARM) : 0u;
    const uint64_t n_cold = past > HOT + WARM ? past - HOT - WARM : 0u;
    const uint64_t window = n_sink + n_hot, first_w = window - T;          /* the chunk: the window's last T rows */
    const uint64_t longest = window > WARM ? (window > n_cold ? window : n_cold) : (WARM > n_cold ? WARM : n_cold);
    const uint64_t rw = 4u * qh * NN__ATTENTION__RESIDUAL_FLOATS(hd);     /* a row's residual, in bytes */
    if (!nn__primitives__fits(SINK * kvw, room[AI_QWEN_3__AT__K_CACHE]) || !nn__primitives__fits(SINK * kvw, room[AI_QWEN_3__AT__V_CACHE])
     || !nn__primitives__fits(HOT * kvw, room[AI_QWEN_3__AT__HOT_K]) || !nn__primitives__fits(HOT * kvw, room[AI_QWEN_3__AT__HOT_V])
     || WARM * kvh * rb8 > room[AI_QWEN_3__AT__WARM_KC] || WARM * kvh * rb8 > room[AI_QWEN_3__AT__WARM_VC]
     || !nn__primitives__fits(WARM * kvh, room[AI_QWEN_3__AT__WARM_KS]) || !nn__primitives__fits(WARM * kvh, room[AI_QWEN_3__AT__WARM_VS])
     || n_cold * kvh * rb4 > room[AI_QWEN_3__AT__COLD_KC] || n_cold * kvh * rb4 > room[AI_QWEN_3__AT__COLD_VC]
     || !nn__primitives__fits(n_cold * kvh, room[AI_QWEN_3__AT__COLD_KS]) || !nn__primitives__fits(n_cold * kvh, room[AI_QWEN_3__AT__COLD_VS])
     || room[AI_QWEN_3__AT__SCORES] / 4u < qh * longest || room[AI_QWEN_3__AT__ROWS_ANGLES] / 4u < T * rot
     || !nn__primitives__fits(window * kvw, room[AI_QWEN_3__AT__WINDOW_K]) || !nn__primitives__fits(window * kvw, room[AI_QWEN_3__AT__WINDOW_V])
     || room[AI_QWEN_3__AT__ROWS_RES] < 3u * T * rw || room[AI_QWEN_3__AT__TIER_SCRATCH] < 2u * kvw
     || (n_warm == 0u && n_cold == 0u && room[AI_QWEN_3__AT__ROWS_SCORES] / 4u < T * qh * window))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    /* scratch: the pairs, then in halves h · hrot · [qg · kk] · qn · vt · qr · attr · att · gated · arot · ao */
    const uint64_t pairs = at[AI_QWEN_3__AT__ROWS_SCRATCH], vpairs = pairs + 8u * AI_QWEN_3__ROWS_MAX, h = vpairs + 8u * AI_QWEN_3__ROWS_MAX;
    const uint64_t hrot = h + 2u * T * H, qg = hrot + 2u * T * H, kk = qg + 2u * T * 2u * qw, qn = kk + 2u * T * kvw;
    const uint64_t vt = qn + 2u * T * qw, qr = vt + 2u * T * kvw, attr = qr + 2u * T * qw, att = attr + 2u * T * qw;
    const uint64_t gated = att + 2u * T * qw, arot = gated + 2u * T * qw, ao = arot + 2u * T * qw;
    if (room[AI_QWEN_3__AT__ROWS_SCRATCH] < ao + 2u * T * H - pairs) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    const uint64_t signs = at[AI_QWEN_3__AT__SIGNS], cs = at[AI_QWEN_3__AT__ROWS_ANGLES], p = at[AI_QWEN_3__AT__SCORES];
    const uint64_t wk = at[AI_QWEN_3__AT__WINDOW_K], wv = at[AI_QWEN_3__AT__WINDOW_V], cool = at[AI_QWEN_3__AT__TIER_SCRATCH];
    const uint64_t r0 = at[AI_QWEN_3__AT__ROWS_RES], r1 = r0 + T * rw, r2 = r1 + T * rw;
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_qwen_3__prefill__zzprivate_run_pairs(ctx, pairs, T, 0u, 0u) || !ai_qwen_3__prefill__zzprivate_run_pairs(ctx, vpairs, T, 0u, 0u))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    /* ① the projections, a row each — the values into `vt`, not a cache */
    const bool rotated = ai_qwen_3__table__zzpackage_flag(table, AI_QWEN_3__AT__RESIDUAL_ROTATED);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)(rotated ? hrot : h), (const uint16_t*)(uintptr_t)x,
                        (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__INPUT_NORM], H, T, H, H, over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, T * H, over);
    nn__expert__groups gr = {};
    gr.codes[0] = at[AI_QWEN_3__AT__Q]; gr.luts[0] = at[AI_QWEN_3__AT__Q_LUT]; gr.out_rows[0] = 2u * qw; gr.d[0] = v[AI_QWEN_3__AT__D_Q];
    gr.codes[1] = at[AI_QWEN_3__AT__K]; gr.luts[1] = at[AI_QWEN_3__AT__K_LUT]; gr.out_rows[1] = kvw;     gr.d[1] = v[AI_QWEN_3__AT__D_K];
    gr.out_at[0] = 0u; gr.out_at[1] = T * 2u * qw;
    gr.pairs[0] = gr.pairs[1] = T;
    gr.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)qg, gr, (const uint16_t*)(uintptr_t)hrot, (const uint32_t*)(uintptr_t)pairs, H, over);
    doors->expert_rows((uint16_t*)(uintptr_t)vt, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__V], room[AI_QWEN_3__AT__V],
                       (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__V_LUT], room[AI_QWEN_3__AT__V_LUT], (const uint16_t*)(uintptr_t)hrot,
                       (const uint32_t*)(uintptr_t)vpairs, T, v[AI_QWEN_3__AT__D_V], kvw, H, over);
    doors->rope_angles_rows((float*)(uintptr_t)cs, first, (float)(uint32_t)theta, rot, T);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qg, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__Q_NORM],
                        hd, T * qh, 2u * hd, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qn, (const float*)(uintptr_t)cs, T, qh, hd, qw, rot, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)kk, (const uint16_t*)(uintptr_t)kk, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__K_NORM],
                        hd, T * kvh, hd, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)kk, (const uint16_t*)(uintptr_t)kk, (const float*)(uintptr_t)cs, T, kvh, hd, kvw, rot, over);
    doors->hadamard_blocks((uint16_t*)(uintptr_t)qr, (const uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)signs, T * qw, hd, 0u, over);

    /* ② the chunk in: each row's key and value rotated into the sink or its hot slot, what it displaces cooled first */
    for (uint64_t r = 0u; r < T; ++r) {
        const uint64_t q = first + r;
        uint64_t kdst = at[AI_QWEN_3__AT__K_CACHE] + 2u * q * kvw, vdst = at[AI_QWEN_3__AT__V_CACHE] + 2u * q * kvw;
        if (q >= SINK) {
            const uint64_t j = q - SINK, slot = j % HOT;
            const uint64_t hk = at[AI_QWEN_3__AT__HOT_K] + 2u * slot * kvw, hv = at[AI_QWEN_3__AT__HOT_V] + 2u * slot * kvw;
            if (j >= HOT) {
                const uint64_t out = j - HOT, w = out % WARM;
                if (out >= WARM) {
                    const uint64_t old = out - WARM;
                    const uint64_t sets[2][4] = {{AI_QWEN_3__AT__WARM_KC, AI_QWEN_3__AT__WARM_KS, AI_QWEN_3__AT__COLD_KC, AI_QWEN_3__AT__COLD_KS},
                                                 {AI_QWEN_3__AT__WARM_VC, AI_QWEN_3__AT__WARM_VS, AI_QWEN_3__AT__COLD_VC, AI_QWEN_3__AT__COLD_VS}};
                    for (unsigned s = 0u; s < 2u; ++s) {
                        (void)doors->turboquant_decode((uint16_t*)(uintptr_t)cool, (const uint8_t*)(uintptr_t)(at[sets[s][0]] + w * kvh * rb8),
                                                       kvh * rb8, (const uint8_t*)(uintptr_t)(at[sets[s][1]] + 2u * w * kvh), 2u * kvh, 8u,
                                                       kvh, hd, over);
                        doors->turboquant_encode_rows((uint8_t*)(uintptr_t)(at[sets[s][2]] + old * kvh * rb4),
                                                      (uint16_t*)(uintptr_t)(at[sets[s][3]] + 2u * old * kvh), (const uint16_t*)(uintptr_t)cool,
                                                      kvh, hd, 4u, over);
                    }
                }
                doors->turboquant_encode_rows((uint8_t*)(uintptr_t)(at[AI_QWEN_3__AT__WARM_KC] + w * kvh * rb8),
                                              (uint16_t*)(uintptr_t)(at[AI_QWEN_3__AT__WARM_KS] + 2u * w * kvh), (const uint16_t*)(uintptr_t)hk,
                                              kvh, hd, 8u, over);
                doors->turboquant_encode_rows((uint8_t*)(uintptr_t)(at[AI_QWEN_3__AT__WARM_VC] + w * kvh * rb8),
                                              (uint16_t*)(uintptr_t)(at[AI_QWEN_3__AT__WARM_VS] + 2u * w * kvh), (const uint16_t*)(uintptr_t)hv,
                                              kvh, hd, 8u, over);
            }
            kdst = hk; vdst = hv;
        }
        doors->hadamard_blocks((uint16_t*)(uintptr_t)kdst, (const uint16_t*)(uintptr_t)(kk + 2u * r * kvw), (const uint16_t*)(uintptr_t)signs,
                               kvw, hd, 0u, over);
        doors->hadamard_blocks((uint16_t*)(uintptr_t)vdst, (const uint16_t*)(uintptr_t)(vt + 2u * r * kvw), (const uint16_t*)(uintptr_t)signs,
                               kvw, hd, 0u, over);
    }

    /* ③ the window: the sink's rows, then the hot ring's in position order */
    const uint64_t caches[2] = {at[AI_QWEN_3__AT__K_CACHE], at[AI_QWEN_3__AT__V_CACHE]};
    const uint64_t rings[2] = {at[AI_QWEN_3__AT__HOT_K], at[AI_QWEN_3__AT__HOT_V]}, wins[2] = {wk, wv};
    const uint64_t oldest = past - n_hot, start = n_hot != 0u ? oldest % HOT : 0u;
    const uint64_t seg = n_hot < HOT - start ? n_hot : HOT - start;
    for (unsigned s = 0u; s < 2u; ++s) {
        doors->vector_copy((uint16_t*)(uintptr_t)wins[s], (const uint16_t*)(uintptr_t)caches[s], n_sink * kvw);
        if (seg != 0u)
            doors->vector_copy((uint16_t*)(uintptr_t)(wins[s] + 2u * n_sink * kvw), (const uint16_t*)(uintptr_t)(rings[s] + 2u * start * kvw), seg * kvw);
        if (n_hot > seg)
            doors->vector_copy((uint16_t*)(uintptr_t)(wins[s] + 2u * (n_sink + seg) * kvw), (const uint16_t*)(uintptr_t)rings[s], (n_hot - seg) * kvw);
    }

    /* ④ while the window is the whole cache — nothing warm or cold yet — the rows attend it at once, causally; past it
       each row: the window up to its own position, the warm and the cold tiers whole, a residual each, merged */
    if (n_warm == 0u && n_cold == 0u) {
        doors->attention_causal_scores((float*)(uintptr_t)at[AI_QWEN_3__AT__ROWS_SCORES], (const uint16_t*)(uintptr_t)qr,
                                       (const uint16_t*)(uintptr_t)wk, qh, kvh, hd, first_w, T);
        doors->attention_causal_mix((uint16_t*)(uintptr_t)attr, (const float*)(uintptr_t)at[AI_QWEN_3__AT__ROWS_SCORES],
                                    (const uint16_t*)(uintptr_t)wv, qh, kvh, hd, first_w, T, over);
    }
    for (uint64_t r = 0u; r < T && (n_warm != 0u || n_cold != 0u); ++r) {
        const uint64_t qrr = qr + 2u * r * qw, len = first_w + r + 1u;
        doors->attention_weights((float*)(uintptr_t)p, (float*)(uintptr_t)(r0 + r * rw), (const uint16_t*)(uintptr_t)qrr,
                                 (const uint16_t*)(uintptr_t)wk, qh, kvh, hd, len);
        doors->attention_residual_mix((float*)(uintptr_t)(r0 + r * rw), (const float*)(uintptr_t)p, (const uint16_t*)(uintptr_t)wv, qh, kvh, hd, len);
        if (n_warm != 0u) {
            doors->attention_weights_tq((float*)(uintptr_t)p, (float*)(uintptr_t)(r1 + r * rw), (const uint16_t*)(uintptr_t)qrr,
                                        (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__WARM_KC], (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__WARM_KS],
                                        qh, kvh, hd, n_warm, 8u);
            doors->attention_residual_mix_tq((float*)(uintptr_t)(r1 + r * rw), (const float*)(uintptr_t)p,
                                             (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__WARM_VC], (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__WARM_VS],
                                             qh, kvh, hd, n_warm, 8u);
        }
        if (n_cold != 0u) {
            doors->attention_weights_tq((float*)(uintptr_t)p, (float*)(uintptr_t)(r2 + r * rw), (const uint16_t*)(uintptr_t)qrr,
                                        (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__COLD_KC], (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__COLD_KS],
                                        qh, kvh, hd, n_cold, 4u);
            doors->attention_residual_mix_tq((float*)(uintptr_t)(r2 + r * rw), (const float*)(uintptr_t)p,
                                             (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__COLD_VC], (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__COLD_VS],
                                             qh, kvh, hd, n_cold, 4u);
        }
    }
    if (n_warm != 0u) doors->attention_merge((float*)(uintptr_t)r0, (const float*)(uintptr_t)r0, (const float*)(uintptr_t)r1, T * qh, hd);
    if (n_cold != 0u) doors->attention_merge((float*)(uintptr_t)r0, (const float*)(uintptr_t)r0, (const float*)(uintptr_t)r2, T * qh, hd);
    if (n_warm != 0u || n_cold != 0u)
        doors->attention_finish((uint16_t*)(uintptr_t)attr, (const float*)(uintptr_t)r0, T * qh, hd, over);

    /* ⑤ the answers rotated back a head at a time, then the one-cache rows verb's gate, rotation and out projection */
    doors->hadamard_blocks((uint16_t*)(uintptr_t)att, (const uint16_t*)(uintptr_t)attr, (const uint16_t*)(uintptr_t)signs, T * qw, hd, 1u, over);
    doors->attention_gate_rows((uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)att, (const uint16_t*)(uintptr_t)qg, T * qh, hd, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)arot, (const uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)signs, T * qw, over);
    doors->expert_rows((uint16_t*)(uintptr_t)ao, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__O], room[AI_QWEN_3__AT__O],
                       (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__O_LUT], room[AI_QWEN_3__AT__O_LUT], (const uint16_t*)(uintptr_t)arot,
                       (const uint32_t*)(uintptr_t)pairs, T, v[AI_QWEN_3__AT__D_O], H, qw, over);
    doors->vector_add((uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)ao, T * H, over);
    return nn__doors_answer(&argv[3]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__attention_tiered_rows__zzabi_adapter, ai_qwen_3__attention_tiered_rows__zzabi_apply)

/* `(ai_qwen_3__mlp_rows h1 planes out rows)` -> `out`: `out = h1 + mlp(rmsnorm(h1))` for `rows` rows — the dense MLP's
 * table (▶ `ai_qwen_3__mlp`), its scratch the rows'. Gate and up of every row in one grouped launch, their swiglu, the
 * rotation, the down rows, the residual. */
static sys__heap_node ai_qwen_3__mlp_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, out = 0, out_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[2], &out, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__MLP__TABLE], room[AI_QWEN_3__MLP__TABLE], v[AI_QWEN_3__MLP__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__MLP__HIDDEN, AI_QWEN_3__MLP__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__MLP__HIDDEN], I = v[AI_QWEN_3__MLP__INTER], T = argv[3].args[0];
    if (H == 0u || I == 0u || H > (1ull << 24) || I > (1ull << 24) || T == 0u || T > AI_QWEN_3__ROWS_MAX
     || !nn__primitives__fits(T * H, x_room) || !nn__primitives__fits(T * H, out_room) || !nn__primitives__fits(H, room[AI_QWEN_3__MLP__NORM]))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    /* scratch: the pairs, then in halves h · hrot · [gates · ups] · act · actr · ao */
    const uint64_t pairs = at[AI_QWEN_3__MLP__SCRATCH], h = pairs + 8u * AI_QWEN_3__ROWS_MAX, hrot = h + 2u * T * H;
    const uint64_t gu = hrot + 2u * T * H, act = gu + 2u * 2u * T * I, actr = act + 2u * T * I, ao = actr + 2u * T * I;
    if (room[AI_QWEN_3__MLP__SCRATCH] < ao + 2u * T * H - pairs) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    const uint64_t signs = at[AI_QWEN_3__MLP__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    if (!ai_qwen_3__prefill__zzprivate_run_pairs(ctx, pairs, T, 0u, 0u)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    const bool rotated = ai_qwen_3__table__zzpackage_flag(argv[1].args[0], AI_QWEN_3__MLP__RESIDUAL_ROTATED);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)(rotated ? hrot : h), (const uint16_t*)(uintptr_t)x,
                        (const uint16_t*)(uintptr_t)at[AI_QWEN_3__MLP__NORM], H, T, H, H, over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, T * H, over);
    nn__expert__groups gr = {};
    gr.codes[0] = at[AI_QWEN_3__MLP__GATE]; gr.luts[0] = at[AI_QWEN_3__MLP__GATE_LUT]; gr.out_rows[0] = I; gr.d[0] = v[AI_QWEN_3__MLP__D_GATE];
    gr.codes[1] = at[AI_QWEN_3__MLP__UP];   gr.luts[1] = at[AI_QWEN_3__MLP__UP_LUT];   gr.out_rows[1] = I; gr.d[1] = v[AI_QWEN_3__MLP__D_UP];
    gr.out_at[0] = 0u; gr.out_at[1] = T * I;
    gr.pairs[0] = gr.pairs[1] = T;
    gr.count = 2u;
    doors->expert_groups((uint16_t*)(uintptr_t)gu, gr, (const uint16_t*)(uintptr_t)hrot, (const uint32_t*)(uintptr_t)pairs, H, over);
    doors->swiglu_combine((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, (const uint16_t*)(uintptr_t)(gu + 2u * T * I), T * I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)signs, T * I, over);
    doors->expert_rows((uint16_t*)(uintptr_t)ao, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MLP__DOWN], room[AI_QWEN_3__MLP__DOWN],
                       (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MLP__DOWN_LUT], room[AI_QWEN_3__MLP__DOWN_LUT], (const uint16_t*)(uintptr_t)actr,
                       (const uint32_t*)(uintptr_t)pairs, T, v[AI_QWEN_3__MLP__D_DOWN], H, I, over);
    doors->vector_add((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)ao, T * H, over);
    return nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__mlp_rows__zzabi_adapter, ai_qwen_3__mlp_rows__zzabi_apply)

/* ── ⭐⭐ WHERE A PROMPT'S ROUTED EXPERTS COME FROM — ⚖ *"a gpu even if it ends up not holding the experts on decode
 *   should reserve some space for the prefill as we could compute in batch prefill while the experts are being
 *   loaded"* ────────────────────────────────────────────────────────────────────────────────────────────────
 * The MoE table's last three cells say. With the store an integer 0 every expert is read where the collection
 * holds it. With a STORE — the layer's experts in RAM, expert `e` at `e · SLOT_BYTES`, each laid out as a
 * collection slot — and a STAGING buffer on the card of two halves, each `AI_QWEN_3__STREAM_GROUP` slots, the
 * experts the prompt uses are copied into one half on the side queue while the card computes the other's:
 * the small L1 the prefill streams through. */
#define AI_QWEN_3__STREAM_GROUP  NN__EXPERT__GROUPS_MAX     /* experts a half holds: one grouped launch */

/* What the streaming has cost since boot — ⚖ *"measure the transfer"*: the bytes copied, the time the copies
 * took, and the time the host waited for the card before launching the next round, all in ns. */
static uint64_t ai_qwen_3__stream__zzprivate_counts[3];
static void* ai_qwen_3__stream__zzprivate_channel = 0;
static sys__silicon_family__id ai_qwen_3__stream__zzprivate_family = SYS__SILICON_FAMILY__NONE;

static uint64_t ai_qwen_3__stream__zzprivate_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* The side queue, opened the first time a prompt streams, closed at shutdown. */
static bool ai_qwen_3__stream__zzprivate_open(sys__silicon_family__id family) {
    if (ai_qwen_3__stream__zzprivate_channel != 0) return ai_qwen_3__stream__zzprivate_family == family;
    if (!sys__gpu__side_open(family, &ai_qwen_3__stream__zzprivate_channel)) return false;
    ai_qwen_3__stream__zzprivate_family = family;
    return true;
}

/* The side queue given back — the package's teardown. */
static void ai_qwen_3__stream__zzpackage_close(void) {
    if (ai_qwen_3__stream__zzprivate_channel != 0) sys__gpu__side_close(ai_qwen_3__stream__zzprivate_family, ai_qwen_3__stream__zzprivate_channel);
    ai_qwen_3__stream__zzprivate_channel = 0;
    ai_qwen_3__stream__zzprivate_family = SYS__SILICON_FAMILY__NONE;
}

/* `(ai_qwen_3__stream_counts which)` -> the count: 0 bytes streamed, 1 ns copying, 2 ns waiting on the card. */
static sys__heap_node ai_qwen_3__stream_counts__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[0].dtype != SYS__KIND__VALUE_INT || argv[0].args[0] > 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    sys__heap_node n = sys__heap_node__nothing();
    n.dtype = SYS__KIND__VALUE_INT; n.args[0] = ai_qwen_3__stream__zzprivate_counts[argv[0].args[0]];
    return n;
}

/* One round of routed experts — `u` from `k · GROUP`, their slots in `slot_of` — over their picks: the gate_ups in
 * one launch, the swiglus and rotations over their rows (contiguous: listing order), each down into the sum. */
static void ai_qwen_3__moe__zzprivate_round(const nn__doors* doors, unsigned int* over, const uint64_t* slot_of, const uint64_t* first_at,
                                            const uint64_t* count_of, uint64_t k, uint64_t n_used, uint64_t up_lut, uint64_t dn,
                                            uint64_t dn_lut, uint64_t ed, uint64_t H, uint64_t I, uint64_t gu, uint64_t act, uint64_t actr,
                                            uint64_t acc, uint64_t hmrot, uint64_t wj, uint64_t p_id, uint64_t at_up, uint64_t at_down,
                                            uint64_t signs) {
    const uint64_t u0 = k * AI_QWEN_3__STREAM_GROUP, u1 = u0 + AI_QWEN_3__STREAM_GROUP < n_used ? u0 + AI_QWEN_3__STREAM_GROUP : n_used;
    nn__expert__groups gr = {};
    for (uint64_t u = u0; u < u1; ++u) {
        const uint64_t i = u - u0;
        gr.codes[i] = slot_of[u]; gr.luts[i] = slot_of[u] + up_lut; gr.d[i] = ed; gr.out_rows[i] = 2u * I; gr.out_at[i] = 0u;
        gr.pairs_at[i] = at_up + first_at[u]; gr.pairs[i] = count_of[u];
    }
    gr.count = u1 - u0;
    doors->expert_groups((uint16_t*)(uintptr_t)gu, gr, (const uint16_t*)(uintptr_t)hmrot, (const uint32_t*)(uintptr_t)p_id, H, over);
    const uint64_t j0 = first_at[u0], jn = first_at[u1 - 1u] + count_of[u1 - 1u] - j0;
    doors->swiglu_pairs((uint16_t*)(uintptr_t)(act + 2u * j0 * I), (const uint16_t*)(uintptr_t)(gu + 2u * j0 * 2u * I), jn, I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(actr + 2u * j0 * I), (const uint16_t*)(uintptr_t)(act + 2u * j0 * I),
                           (const uint16_t*)(uintptr_t)signs, jn * I, over);
    const uint64_t down_bytes = ai_qwen_3__prefill__zzprivate_codes(ed, H, I);
    for (uint64_t u = u0; u < u1; ++u)
        doors->expert_rows_sum((float*)(uintptr_t)acc, (const uint8_t*)(uintptr_t)(slot_of[u] + dn), down_bytes,
                               (const uint8_t*)(uintptr_t)(slot_of[u] + dn_lut), 2u * H, (const uint16_t*)(uintptr_t)actr,
                               (const uint32_t*)(uintptr_t)(p_id + 8u * (at_down + first_at[u])), (const uint16_t*)(uintptr_t)wj,
                               count_of[u], ed, H, I);
}

/* `(ai_qwen_3__moe_rows h1 planes out rows)` -> `out = h1 + MoE(h1)`, row by row, over the fused verb's table and
 * its three streaming cells.
 * ⭐ EXPERT-MAJOR: the router's picks for every row are read in one wait and inverted on the host — each expert
 *   the batch uses, and the picks that chose it. Row `j` of the gate_ups and activations is the `j`-th pick in
 *   that order, so an expert's rows are contiguous; the shared expert's row for position `r` is `P + r`. Twelve
 *   experts a round: their gate_ups in one launch, their swiglus and rotations over their rows, each one's down
 *   weighted into the fp32 sum. The sum is rounded once onto `h1`. */
static sys__heap_node ai_qwen_3__moe_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t h1_at = 0, h1_room = 0, out_at = 0, out_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0]) || argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &h1_at, &h1_room) || !nn__primitives__room(&argv[2], &out_at, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t table = argv[1].args[0];
    uint64_t at[AI_QWEN_3__MOE__TABLE], room[AI_QWEN_3__MOE__TABLE], v[AI_QWEN_3__MOE__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(table, AI_QWEN_3__MOE__LAYER, AI_QWEN_3__MOE__TABLE, at, room, v)
     || sys__node_array__length(table) < AI_QWEN_3__MOE__ROWS_TABLE)
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__MOE__HIDDEN], I = v[AI_QWEN_3__MOE__INTER], E = v[AI_QWEN_3__MOE__EXPERTS];
    const uint64_t K = v[AI_QWEN_3__MOE__TOP_K], layer = v[AI_QWEN_3__MOE__LAYER], type = v[AI_QWEN_3__MOE__EXPERT_TYPE];
    const uint64_t T = argv[3].args[0], ed = v[AI_QWEN_3__MOE__EXPERT_D], sd = v[AI_QWEN_3__MOE__SHARED_D];
    if (H == 0u || I == 0u || E == 0u || K == 0u || K > E || K > NN__VECTOR__TOP_K_MAX || H > (1ull << 24) || I > (1ull << 24)
     || E > (1ull << 24) || T == 0u || T > AI_QWEN_3__ROWS_MAX
     || !nn__primitives__fits(T * H, h1_room) || !nn__primitives__fits(T * H, out_room) || !nn__primitives__fits(H, room[AI_QWEN_3__MOE__NORM]))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
    /* where the routed experts come from: the collection, or the store through the staging halves */
    const sys__heap_node store_cell = sys__node_array__borrow(table, AI_QWEN_3__MOE__STORE);
    const sys__heap_node staging_cell = sys__node_array__borrow(table, AI_QWEN_3__MOE__STAGING);
    const sys__heap_node slot_cell = sys__node_array__borrow(table, AI_QWEN_3__MOE__SLOT_BYTES);
    const bool streams = !(store_cell.dtype == SYS__KIND__VALUE_INT && store_cell.args[0] == 0ull);
    uint64_t store = 0, store_room = 0, store_host = 0, staging = 0, staging_room = 0, slot_bytes = 0;
    if (streams) {
        if (!nn__primitives__room(&store_cell, &store, &store_room) || !nn__primitives__room(&staging_cell, &staging, &staging_room)
         || slot_cell.dtype != SYS__KIND__VALUE_INT)
            return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_TABLE);
        (void)nn__primitives__result_slot(&store_cell, &store_host);
        slot_bytes = slot_cell.args[0];
        if (store_host == 0ull || slot_bytes == 0ull || slot_bytes > store_room / E || 2u * AI_QWEN_3__STREAM_GROUP * slot_bytes > staging_room
         || !ai_qwen_3__stream__zzprivate_open(ctx->family))
            return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    }
    const uint64_t P = T * K, R = P + T;               /* the picks, and every row of gate_up: the picks' then the shared's */
    /* scratch: the pairs — T identity, P up, P down, T shared gate, T shared up, T shared down, all counted from
     *   the first — then in halves hm · hmrot · logits · w (the router's weights as top_k leaves them) · wj (the
     *   same in listing order, then the T shared gates) · gu (R pairs of 2I, then T logits) · act · actr, the
     *   fp32 sum, and the picks as the card leaves them */
    const uint64_t n_pairs = T + 2u * P + 3u * T;
    const uint64_t pairs = at[AI_QWEN_3__MOE__SCRATCH], hm = pairs + 8u * n_pairs;
    const uint64_t hmrot = hm + 2u * T * H, logits = hmrot + 2u * T * H, w = logits + 2u * T * E, wj = w + 2u * P, gu = wj + 2u * R;
    const uint64_t act = gu + 2u * (R * 2u * I + T), actr = act + 2u * R * I, accf = actr + 2u * R * I;
    const uint64_t acc = (accf + 7u) & ~7ull, landing = acc + 4u * T * H;   /* the fp32 sum, then the picks' words */
    if (!nn__primitives__fits((landing + 8u * P - pairs + 1u) / 2u, room[AI_QWEN_3__MOE__SCRATCH]))
        return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_ROOM);
    const uint64_t p_id = pairs, at_up = T, at_down = T + P, at_shared = T + 2u * P;   /* in pairs, from `p_id` */
    const uint64_t signs = at[AI_QWEN_3__MOE__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    if (!ai_qwen_3__prefill__zzprivate_run_pairs(ctx, p_id, T, 0u, 0u)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);

    /* ① the norm and the router over every row, ② every row's picks and weights, read in one wait */
    const bool rotated = ai_qwen_3__table__zzpackage_flag(table, AI_QWEN_3__MOE__RESIDUAL_ROTATED);
    const uint64_t normed = rotated ? hmrot : hm;           /* rotated: the normed rows are what every gate_up reads */
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)normed, (const uint16_t*)(uintptr_t)h1_at, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__MOE__NORM],
                        H, T, H, H, over);
    doors->expert_rows((uint16_t*)(uintptr_t)logits, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MOE__ROUTER], room[AI_QWEN_3__MOE__ROUTER],
                       (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MOE__ROUTER_LUT], room[AI_QWEN_3__MOE__ROUTER_LUT],
                       (const uint16_t*)(uintptr_t)normed, (const uint32_t*)(uintptr_t)p_id, T, v[AI_QWEN_3__MOE__ROUTER_D], E, H, over);
    uint64_t chosen[AI_QWEN_3__ROWS_MAX * NN__VECTOR__TOP_K_MAX];
    uint16_t weight[AI_QWEN_3__ROWS_MAX * NN__VECTOR__TOP_K_MAX];
    if (!ai_qwen_3__table__zzpackage_top_k_rows(ctx, doors, landing, w, logits, E, K, T, chosen)
     || !sys__gpu__memory_read(ctx->family, weight, (const void*)(uintptr_t)w, (size_t)(2u * P)))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);

    /* ③ inverted: the experts used, in order of first use, each with its picks in listing order — its up pairs
     *   (row, j), its down pairs (j, row), and the weights in the same order */
    uint64_t used[AI_QWEN_3__ROWS_MAX * NN__VECTOR__TOP_K_MAX], first_at[AI_QWEN_3__ROWS_MAX * NN__VECTOR__TOP_K_MAX];
    uint64_t count_of[AI_QWEN_3__ROWS_MAX * NN__VECTOR__TOP_K_MAX];
    uint32_t all[2u * (2u * AI_QWEN_3__ROWS_MAX * NN__VECTOR__TOP_K_MAX + 4u * AI_QWEN_3__ROWS_MAX)];
    uint16_t wlist[AI_QWEN_3__ROWS_MAX * NN__VECTOR__TOP_K_MAX];
    for (uint64_t r = 0u; r < T; ++r) { all[2u * r] = (uint32_t)r; all[2u * r + 1u] = (uint32_t)r; }
    uint64_t n_used = 0u, j = 0u;
    for (uint64_t p = 0u; p < P; ++p) {
        if (chosen[p] >= E) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_EXPERT);
        bool seen = false;
        for (uint64_t u = 0u; u < n_used && !seen; ++u) seen = used[u] == chosen[p];
        if (!seen) used[n_used++] = chosen[p];
    }
    uint32_t* up = all + 2u * at_up;
    uint32_t* down = all + 2u * at_down;
    for (uint64_t u = 0u; u < n_used; ++u) {
        first_at[u] = j; count_of[u] = 0u;
        for (uint64_t p = 0u; p < P; ++p)
            if (chosen[p] == used[u]) {
                up[2u * j] = (uint32_t)(p / K);   up[2u * j + 1u] = (uint32_t)j;
                down[2u * j] = (uint32_t)j;       down[2u * j + 1u] = (uint32_t)(p / K);
                wlist[j] = weight[p];
                ++j; ++count_of[u];
            }
    }
    uint32_t* shared = all + 2u * at_shared;
    for (uint64_t r = 0u; r < T; ++r) {
        shared[2u * r] = (uint32_t)r;             shared[2u * r + 1u] = (uint32_t)(2u * (P + r));        /* gate: the pair's first half */
        shared[2u * (T + r)] = (uint32_t)r;       shared[2u * (T + r) + 1u] = (uint32_t)(2u * (P + r) + 1u); /* up: its second */
        shared[2u * (2u * T + r)] = (uint32_t)(P + r); shared[2u * (2u * T + r) + 1u] = (uint32_t)r;        /* down: back onto the row */
    }
    if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)p_id, all, (size_t)(8u * n_pairs))
     || !sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)wj, wlist, (size_t)(2u * P)))
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);

    /* ④ the shared expert over every row, its weights resident: gate, up and logit, swiglu, rotation, its gate's
     *   sigmoid after the routed weights, and its down into the sum */
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hmrot, (const uint16_t*)(uintptr_t)hm, (const uint16_t*)(uintptr_t)signs, T * H, over);
    doors->vector_zero((uint16_t*)(uintptr_t)acc, 2u * T * H);
    nn__expert__groups g = {};
    const uint64_t sgate[3] = {AI_QWEN_3__MOE__SHARED_GATE, AI_QWEN_3__MOE__SHARED_UP, AI_QWEN_3__MOE__SHARED_LOGIT};
    for (uint64_t s = 0u; s < 3u; ++s) {
        g.codes[s] = at[sgate[s]]; g.luts[s] = at[sgate[s] + 1u]; g.d[s] = sd;
        g.out_rows[s] = s == 2u ? 1u : I;
        g.out_at[s] = s == 2u ? R * 2u * I : 0u;
        g.pairs_at[s] = s == 2u ? 0u : at_shared + s * T; g.pairs[s] = T;   /* the logit: one out row per row */
    }
    g.count = 3u;
    doors->expert_groups((uint16_t*)(uintptr_t)gu, g, (const uint16_t*)(uintptr_t)hmrot, (const uint32_t*)(uintptr_t)p_id, H, over);
    doors->swiglu_pairs((uint16_t*)(uintptr_t)(act + 2u * P * I), (const uint16_t*)(uintptr_t)(gu + 2u * P * 2u * I), T, I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)(actr + 2u * P * I), (const uint16_t*)(uintptr_t)(act + 2u * P * I),
                           (const uint16_t*)(uintptr_t)signs, T * I, over);
    doors->sigmoid((uint16_t*)(uintptr_t)(wj + 2u * P), (const uint16_t*)(uintptr_t)(gu + 2u * R * 2u * I), T, over);
    doors->expert_rows_sum((float*)(uintptr_t)acc, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MOE__SHARED_DOWN], room[AI_QWEN_3__MOE__SHARED_DOWN],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__MOE__SHARED_DOWN_LUT], room[AI_QWEN_3__MOE__SHARED_DOWN_LUT],
                           (const uint16_t*)(uintptr_t)actr, (const uint32_t*)(uintptr_t)(p_id + 8u * (at_shared + 2u * T)),
                           (const uint16_t*)(uintptr_t)wj, T, sd, H, I);

    /* ⑤ the routed experts, twelve a round. Streaming, each pass of the loop first waits for the card — which
     *   then holds nothing older than the round before this one — then launches round `k − 1`, whose experts
     *   are already in half `(k − 1) % 2`, then copies round `k` into half `k % 2` while that round computes.
     *   ⛔ THE WAIT COMES BEFORE THE COPY: half `k % 2` was last read by round `k − 2`, and only the wait says
     *   it is done — copying first let a round's weights change under the kernels still reading them (`MEASURED`:
     *   the streamed prompt parted from the resident one until this order). */
    const uint64_t up_lut = v[AI_QWEN_3__MOE__EXPERT_UP_LUT], dn = v[AI_QWEN_3__MOE__EXPERT_DOWN], dn_lut = v[AI_QWEN_3__MOE__EXPERT_DOWN_LUT];
    const uint64_t rounds = (n_used + AI_QWEN_3__STREAM_GROUP - 1u) / AI_QWEN_3__STREAM_GROUP;
    uint64_t slot_of[AI_QWEN_3__ROWS_MAX * NN__VECTOR__TOP_K_MAX];
    for (uint64_t u = 0u; u < n_used && !streams; ++u) {
        sys__heap_node me;
        if (!nn__expert__slot(layer, type, used[u], &me) || me.args[NN__EXPERT__SLOT_AT] == 0ull
         || me.args[NN__EXPERT__SLOT_TYPE] != type) return sys__engine__abi__error(AI_QWEN_3__MOE__FAULT_EXPERT);
        slot_of[u] = me.args[NN__EXPERT__SLOT_AT];
    }
    for (uint64_t k = 0u; k <= rounds; ++k) {
        if (streams && k >= 2u) {
            const uint64_t t0 = ai_qwen_3__stream__zzprivate_now();
            if (!sys__gpu__compute_completed(ctx->family)) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
            ai_qwen_3__stream__zzprivate_counts[2] += ai_qwen_3__stream__zzprivate_now() - t0;
        }
        if (k > 0u) ai_qwen_3__moe__zzprivate_round(doors, over, slot_of, first_at, count_of, k - 1u, n_used, up_lut, dn, dn_lut,
                                                    ed, H, I, gu, act, actr, acc, hmrot, wj, p_id, at_up, at_down, signs);
        if (streams && k < rounds) {
            const uint64_t t0 = ai_qwen_3__stream__zzprivate_now();
            for (uint64_t u = k * AI_QWEN_3__STREAM_GROUP; u < n_used && u < (k + 1u) * AI_QWEN_3__STREAM_GROUP; ++u) {
                slot_of[u] = staging + ((k % 2u) * AI_QWEN_3__STREAM_GROUP + (u - k * AI_QWEN_3__STREAM_GROUP)) * slot_bytes;
                if (!sys__gpu__side_memory_write(ctx->family, (void*)(uintptr_t)slot_of[u], (const void*)(uintptr_t)(store_host + used[u] * slot_bytes),
                                                 (size_t)slot_bytes, ai_qwen_3__stream__zzprivate_channel))
                    return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
                ai_qwen_3__stream__zzprivate_counts[0] += slot_bytes;
            }
            ai_qwen_3__stream__zzprivate_counts[1] += ai_qwen_3__stream__zzprivate_now() - t0;
        }
    }
    doors->expert_rows_finish((uint16_t*)(uintptr_t)out_at, (float*)(uintptr_t)acc, (const uint16_t*)(uintptr_t)h1_at, T * H, over);
    return nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__deltanet_rows__zzabi_adapter,  ai_qwen_3__deltanet_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__attention_rows__zzabi_adapter, ai_qwen_3__attention_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__moe_rows__zzabi_adapter,       ai_qwen_3__moe_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__stream_counts__zzabi_adapter,  ai_qwen_3__stream_counts__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_PREFILL__ABI_CUH */
