#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MIXER__ABI_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MIXER__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../../nn/cpu/doors.cuh"            /* nn's doors, launched on the worker's silicon */
#include "../../contracts/objects/mixer.cuh"    /* the mixers' tables and faults */
#include "../table.cuh"                          /* reading a table */

/* ══ ⭐⭐ THE MIXERS — the half of a layer before its MoE, each owning its state ══════════════════════════════
 * ⚖ *"a single word opcode for the attention"*: both answer `h1 = x + mixer(rmsnorm(x))`, the layer's first
 * residual included, over their own plane table (`contracts/objects/mixer.cuh`). The DeltaNet reads and
 * writes its recurrent state and conv window in place; the attention writes this position's key and value
 * into its cache and reads rows `0..pos`. The arithmetic is nn's, door for door. */

/* `(ai_qwen_3__deltanet x planes h1 [residual])` -> `h1`. One grouped gemv for the four projections of the normed input,
 * the gates' small arithmetic, the conv, every head's recurrence in one step, and the out projection. */
static sys__heap_node ai_qwen_3__deltanet__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u && argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, h1 = 0, h1_room = 0, res = 0, res_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[2], &h1, &h1_room)
     || !nn__primitives__room(&argv[argc == 4u ? 3u : 0u], &res, &res_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__DN__TABLE], room[AI_QWEN_3__DN__TABLE], v[AI_QWEN_3__DN__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__DN__HIDDEN, AI_QWEN_3__DN__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__DN__HIDDEN], kh = v[AI_QWEN_3__DN__K_HEADS], vh = v[AI_QWEN_3__DN__V_HEADS];
    const uint64_t hd = v[AI_QWEN_3__DN__HEAD_DIM];
    if (H == 0u || kh == 0u || vh == 0u || hd == 0u || vh % kh != 0u || H > (1ull << 24) || vh > (1ull << 16) || hd > (1ull << 16)
     || !nn__primitives__fits(H, x_room) || !nn__primitives__fits(H, h1_room) || !nn__primitives__fits(H, res_room))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t vdim = vh * hd, ch = 2u * kh * hd + vdim;
    /* scratch, in halves: h · hrot · [mixed · z · a · b] · beta · sumb · spb · eab · prodb · gb · conved · core · crot · ao */
    const uint64_t proj = ch + vdim + 2u * vh;
    const uint64_t need = 2u * H + proj + 6u * vh + ch + 2u * vdim + H;
    if (!nn__primitives__fits(need, room[AI_QWEN_3__DN__SCRATCH])) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    const uint64_t h = at[AI_QWEN_3__DN__SCRATCH], hrot = h + 2u * H, mixed = hrot + 2u * H, z = mixed + 2u * ch;
    const uint64_t av = z + 2u * vdim, bv = av + 2u * vh, beta = bv + 2u * vh, sumb = beta + 2u * vh, spb = sumb + 2u * vh;
    const uint64_t eab = spb + 2u * vh, prodb = eab + 2u * vh, gb = prodb + 2u * vh, conved = gb + 2u * vh;
    const uint64_t core = conved + 2u * ch, crot = core + 2u * vdim, ao = crot + 2u * vdim;
    const uint64_t signs = at[AI_QWEN_3__DN__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    const bool rotated = ai_qwen_3__table__zzpackage_flag(argv[1].args[0], AI_QWEN_3__DN__RESIDUAL_ROTATED);
    doors->rmsnorm((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__INPUT_NORM], H, over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, H, over);
    const uint64_t in = rotated ? h : hrot;
    nn__turboquant__groups g = {};
    const unsigned codes[4] = {AI_QWEN_3__DN__QKV, AI_QWEN_3__DN__Z, AI_QWEN_3__DN__A, AI_QWEN_3__DN__B};
    const unsigned widths[4] = {AI_QWEN_3__DN__D_QKV, AI_QWEN_3__DN__D_Z, AI_QWEN_3__DN__D_A, AI_QWEN_3__DN__D_B};
    const uint64_t rows[4] = {ch, vdim, vh, vh};
    uint64_t placed = 0u;
    for (unsigned i = 0u; i < 4u; ++i) {
        g.codes[i] = at[codes[i]]; g.luts[i] = at[codes[i] + 1u]; g.rows[i] = rows[i]; g.d[i] = v[widths[i]];
        g.out_at[i] = placed; placed += rows[i];
    }
    g.count = 4u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)mixed, g, (const uint16_t*)(uintptr_t)in, H, over);
    /* the gates: β = σ(b) · g = −e^{A_log} · softplus(a + dt_bias) — the step takes e^g itself */
    doors->sigmoid((uint16_t*)(uintptr_t)beta, (const uint16_t*)(uintptr_t)bv, vh, over);
    doors->vector_add((uint16_t*)(uintptr_t)sumb, (const uint16_t*)(uintptr_t)av, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__DT_BIAS], vh, over);
    doors->vector_softplus((uint16_t*)(uintptr_t)spb, (const uint16_t*)(uintptr_t)sumb, vh, over);
    doors->vector_exp((uint16_t*)(uintptr_t)eab, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__A_LOG], vh, over);
    doors->vector_pointwise_mul((uint16_t*)(uintptr_t)prodb, (const uint16_t*)(uintptr_t)spb, (const uint16_t*)(uintptr_t)eab, vh, over);
    doors->vector_scale_at((uint16_t*)(uintptr_t)gb, (const uint16_t*)(uintptr_t)prodb, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__MINUS1], 0u, vh, over);
    doors->deltanet_conv_step((uint16_t*)(uintptr_t)conved, (uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__WINDOW],
                              (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__CONV_W], (const uint16_t*)(uintptr_t)mixed, ch, over);
    doors->deltanet_step((float*)(uintptr_t)at[AI_QWEN_3__DN__STATE], (const uint16_t*)(uintptr_t)conved, (const uint16_t*)(uintptr_t)z,
                         (const uint16_t*)(uintptr_t)beta, (const uint16_t*)(uintptr_t)gb, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__DN__NORM_W],
                         (uint16_t*)(uintptr_t)core, kh, vh, hd, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)crot, (const uint16_t*)(uintptr_t)core, (const uint16_t*)(uintptr_t)signs, vdim, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)ao, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__DN__OUT], room[AI_QWEN_3__DN__OUT],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__DN__OUT_LUT], room[AI_QWEN_3__DN__OUT_LUT],
                           (const uint16_t*)(uintptr_t)crot, v[AI_QWEN_3__DN__D_OUT], H, vdim, over);
    const uint64_t masked = ai_qwen_3__table__zzpackage_masked(doors, over, argv[1].args[0], AI_QWEN_3__DN__MASK, ao, crot, 1u, H, vdim,
                                                               AI_QWEN_3__MIXER__FAULT_TABLE);
    if (masked != 0u) return sys__engine__abi__error(masked);
    doors->vector_add((uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)res, (const uint16_t*)(uintptr_t)ao, H, over);
    return nn__doors_answer(&argv[2]);
}

/* `(ai_qwen_3__attention x planes pos h1 [residual])` -> `h1`. Gated attention with partial RoPE: q_proj interleaves each
 * head's query with its output gate; q and k are RMS-normed a head at a time and rotated in their first `ROT`
 * dims; k and v go into the caches at row `pos`; the query attends rows `0..pos`. */
static sys__heap_node ai_qwen_3__attention__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, h1 = 0, h1_room = 0, res = 0, res_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[3], &h1, &h1_room)
     || !nn__primitives__room(&argv[argc == 5u ? 4u : 0u], &res, &res_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__AT__TABLE], room[AI_QWEN_3__AT__TABLE], v[AI_QWEN_3__AT__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__AT__HIDDEN, AI_QWEN_3__AT__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__AT__HIDDEN], qh = v[AI_QWEN_3__AT__Q_HEADS], kvh = v[AI_QWEN_3__AT__KV_HEADS];
    const uint64_t hd = v[AI_QWEN_3__AT__HEAD_DIM], rot = v[AI_QWEN_3__AT__ROT], pos = argv[2].args[0];
    if (H == 0u || qh == 0u || kvh == 0u || hd == 0u || qh % kvh != 0u || rot == 0u || rot > hd || rot % 2u != 0u
     || H > (1ull << 24) || qh > (1ull << 12) || hd > (1ull << 12) || pos >= (1ull << 32)
     || !nn__primitives__fits(H, x_room) || !nn__primitives__fits(H, h1_room) || !nn__primitives__fits(H, res_room))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t kvw = kvh * hd, qw = qh * hd, to = pos + 1u;
    if (!nn__primitives__fits(to * kvw, room[AI_QWEN_3__AT__K_CACHE]) || !nn__primitives__fits(to * kvw, room[AI_QWEN_3__AT__V_CACHE])
     || room[AI_QWEN_3__AT__SCORES] / 4u < qh * to)
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    /* scratch, in halves: h · hrot · [qg · kk] · qn · gate · att · gated · arot · ao */
    const uint64_t need = 2u * H + 2u * qw + kvw + 5u * qw + H;
    if (!nn__primitives__fits(need, room[AI_QWEN_3__AT__SCRATCH])) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    const uint64_t h = at[AI_QWEN_3__AT__SCRATCH], hrot = h + 2u * H, qg = hrot + 2u * H, kk = qg + 2u * 2u * qw;
    const uint64_t qn = kk + 2u * kvw, gate = qn + 2u * qw, att = gate + 2u * qw, gated = att + 2u * qw;
    const uint64_t arot = gated + 2u * qw, ao = arot + 2u * qw;
    const uint64_t krow = at[AI_QWEN_3__AT__K_CACHE] + 2u * pos * kvw, vrow = at[AI_QWEN_3__AT__V_CACHE] + 2u * pos * kvw;
    const uint64_t signs = at[AI_QWEN_3__AT__SIGNS], cs = at[AI_QWEN_3__AT__ANGLES];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    const bool rotated = ai_qwen_3__table__zzpackage_flag(argv[1].args[0], AI_QWEN_3__AT__RESIDUAL_ROTATED);
    doors->rmsnorm((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__INPUT_NORM], H, over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, H, over);
    const uint64_t in = rotated ? h : hrot;
    nn__turboquant__groups g = {};
    g.codes[0] = at[AI_QWEN_3__AT__Q]; g.luts[0] = at[AI_QWEN_3__AT__Q_LUT]; g.rows[0] = 2u * qw; g.d[0] = v[AI_QWEN_3__AT__D_Q]; g.out_at[0] = 0u;
    g.codes[1] = at[AI_QWEN_3__AT__K]; g.luts[1] = at[AI_QWEN_3__AT__K_LUT]; g.rows[1] = kvw; g.d[1] = v[AI_QWEN_3__AT__D_K]; g.out_at[1] = 2u * qw;
    g.count = 2u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)qg, g, (const uint16_t*)(uintptr_t)in, H, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)vrow, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__V], room[AI_QWEN_3__AT__V],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__V_LUT], room[AI_QWEN_3__AT__V_LUT],
                           (const uint16_t*)(uintptr_t)in, v[AI_QWEN_3__AT__D_V], kvw, H, over);
    /* every head's query normed and rotated in one launch each — a head a row, its query and gate side by side in
     * `qg` — and each gate through a sigmoid; every key the same, straight into its cache row */
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qg, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__Q_NORM],
                        hd, qh, 2u * hd, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qn, (const float*)(uintptr_t)cs, 1u, qh, hd, qw, rot, over);
    for (uint64_t k = 0u; k < qh; ++k)
        doors->sigmoid((uint16_t*)(uintptr_t)(gate + 2u * k * hd), (const uint16_t*)(uintptr_t)(qg + 2u * k * 2u * hd + 2u * hd), hd, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)krow, (const uint16_t*)(uintptr_t)kk, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__K_NORM],
                        hd, kvh, hd, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)krow, (const uint16_t*)(uintptr_t)krow, (const float*)(uintptr_t)cs, 1u, kvh, hd, kvw, rot, over);
    doors->attention_scores((float*)(uintptr_t)at[AI_QWEN_3__AT__SCORES], (const uint16_t*)(uintptr_t)qn,
                            (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__K_CACHE], qh, kvh, hd, to);
    doors->attention_mix((uint16_t*)(uintptr_t)att, (const float*)(uintptr_t)at[AI_QWEN_3__AT__SCORES],
                         (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__V_CACHE], qh, kvh, hd, to, over);
    doors->vector_pointwise_mul((uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)att, (const uint16_t*)(uintptr_t)gate, qw, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)arot, (const uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)signs, qw, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)ao, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__O], room[AI_QWEN_3__AT__O],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__O_LUT], room[AI_QWEN_3__AT__O_LUT],
                           (const uint16_t*)(uintptr_t)arot, v[AI_QWEN_3__AT__D_O], H, qw, over);
    const uint64_t masked = ai_qwen_3__table__zzpackage_masked(doors, over, argv[1].args[0], AI_QWEN_3__AT__MASK, ao, arot, 1u, H, qw,
                                                               AI_QWEN_3__MIXER__FAULT_TABLE);
    if (masked != 0u) return sys__engine__abi__error(masked);
    doors->vector_add((uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)res, (const uint16_t*)(uintptr_t)ao, H, over);
    return nn__doors_answer(&argv[3]);
}

/* `(ai_qwen_3__mlp h1 planes out [residual])` -> `out`: the dense MLP, its norm and its residual — two launches for the
 * three projections, the swiglu, and the activation's rotation between them. ▶ `contracts/objects/mixer.cuh`. */
static sys__heap_node ai_qwen_3__mlp__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 3u && argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t h1 = 0, h1_room = 0, out = 0, out_room = 0, res = 0, res_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || !nn__primitives__room(&argv[0], &h1, &h1_room) || !nn__primitives__room(&argv[2], &out, &out_room)
     || !nn__primitives__room(&argv[argc == 4u ? 3u : 0u], &res, &res_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    uint64_t at[AI_QWEN_3__MLP__TABLE], room[AI_QWEN_3__MLP__TABLE], v[AI_QWEN_3__MLP__TABLE];
    if (!ai_qwen_3__table__zzpackage_read(argv[1].args[0], AI_QWEN_3__MLP__HIDDEN, AI_QWEN_3__MLP__TABLE, at, room, v))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t H = v[AI_QWEN_3__MLP__HIDDEN], I = v[AI_QWEN_3__MLP__INTER];
    if (H == 0u || I == 0u || H > (1ull << 24) || I > (1ull << 24)
     || !nn__primitives__fits(H, h1_room) || !nn__primitives__fits(H, out_room) || !nn__primitives__fits(H, res_room)
     || !nn__primitives__fits(H, room[AI_QWEN_3__MLP__NORM]) || !nn__primitives__fits(1u, room[AI_QWEN_3__MLP__ONE]))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    /* scratch, in halves: h · hrot · [gate · up] · act · actr */
    if (!nn__primitives__fits(2u * H + 4u * I, room[AI_QWEN_3__MLP__SCRATCH])) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    const uint64_t h = at[AI_QWEN_3__MLP__SCRATCH], hrot = h + 2u * H, gu = hrot + 2u * H, act = gu + 2u * 2u * I, actr = act + 2u * I;
    const uint64_t signs = at[AI_QWEN_3__MLP__SIGNS];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    const bool rotated = ai_qwen_3__table__zzpackage_flag(argv[1].args[0], AI_QWEN_3__MLP__RESIDUAL_ROTATED);
    doors->rmsnorm((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__MLP__NORM], H, over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, H, over);
    nn__turboquant__groups g = {};
    g.codes[0] = at[AI_QWEN_3__MLP__GATE]; g.luts[0] = at[AI_QWEN_3__MLP__GATE_LUT]; g.rows[0] = I; g.d[0] = v[AI_QWEN_3__MLP__D_GATE]; g.out_at[0] = 0u;
    g.codes[1] = at[AI_QWEN_3__MLP__UP];   g.luts[1] = at[AI_QWEN_3__MLP__UP_LUT];   g.rows[1] = I; g.d[1] = v[AI_QWEN_3__MLP__D_UP];   g.out_at[1] = I;
    g.count = 2u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)gu, g, (const uint16_t*)(uintptr_t)(rotated ? h : hrot), H, over);
    doors->swiglu_pairs((uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)gu, 1u, I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)actr, (const uint16_t*)(uintptr_t)act, (const uint16_t*)(uintptr_t)signs, I, over);
    nn__turboquant__groups dn = {};
    dn.codes[0] = at[AI_QWEN_3__MLP__DOWN]; dn.luts[0] = at[AI_QWEN_3__MLP__DOWN_LUT]; dn.rows[0] = H; dn.d[0] = v[AI_QWEN_3__MLP__D_DOWN];
    dn.out_at[0] = 0u; dn.count = 1u;
    doors->turboquant_gemv_groups_sum((uint16_t*)(uintptr_t)out, dn, (const uint16_t*)(uintptr_t)actr,
                                      (const uint16_t*)(uintptr_t)at[AI_QWEN_3__MLP__ONE], (const uint16_t*)(uintptr_t)res, H, I, over);
    return nn__doors_answer(&argv[2]);
}

/* `(ai_qwen_3__attention_tiered x planes pos h1 [residual])` -> `h1`: the attention of `ai_qwen_3__attention` over a cache in
 * tiers (▶ `contracts/objects/mixer.cuh`). This position's key and value are rotated a head at a time into the sink, or into
 * the hot ring — the position they displace quantised into the warm ring first, and the one that displaces decoded and
 * quantised again into the cold tier. The query, rotated as they are, attends every tier as a residual, the four merged
 * into one softmax, and the answer is rotated back a head at a time before its gate. */
static sys__heap_node ai_qwen_3__attention_tiered__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, h1 = 0, h1_room = 0, res = 0, res_room = 0;
    if (!sys__heap_node__carries_reference(argv[1].dtype) || !sys__node_array__is(argv[1].args[0])
     || argv[2].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[3], &h1, &h1_room)
     || !nn__primitives__room(&argv[argc == 5u ? 4u : 0u], &res, &res_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t table = argv[1].args[0];
    uint64_t at[AI_QWEN_3__AT__TIERED_TABLE], room[AI_QWEN_3__AT__TIERED_TABLE], v[AI_QWEN_3__AT__TIERED_TABLE];
    if (!ai_qwen_3__table__zzpackage_read(table, AI_QWEN_3__AT__HIDDEN, AI_QWEN_3__AT__HOT_K, at, room, v)
     || sys__node_array__length(table) < AI_QWEN_3__AT__TIERED_TABLE)
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    for (unsigned i = AI_QWEN_3__AT__HOT_K; i < AI_QWEN_3__AT__TIERED_TABLE; ++i) {
        const sys__heap_node n = sys__node_array__borrow(table, i);
        if (i < AI_QWEN_3__AT__SINK) { if (!nn__primitives__room(&n, &at[i], &room[i])) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE); }
        else if (n.dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
        else v[i] = n.args[0];
    }
    const uint64_t H = v[AI_QWEN_3__AT__HIDDEN], qh = v[AI_QWEN_3__AT__Q_HEADS], kvh = v[AI_QWEN_3__AT__KV_HEADS];
    const uint64_t hd = v[AI_QWEN_3__AT__HEAD_DIM], rot = v[AI_QWEN_3__AT__ROT], pos = argv[2].args[0];
    const uint64_t SINK = v[AI_QWEN_3__AT__SINK], HOT = v[AI_QWEN_3__AT__HOT], WARM = v[AI_QWEN_3__AT__WARM];
    if (H == 0u || qh == 0u || kvh == 0u || hd == 0u || qh % kvh != 0u || rot == 0u || rot > hd || rot % 2u != 0u
     || hd > 512u || (hd & (hd - 1u)) != 0u || SINK == 0u || HOT == 0u || WARM == 0u
     || H > (1ull << 24) || qh > (1ull << 12) || pos >= (1ull << 32)
     || !nn__primitives__fits(H, x_room) || !nn__primitives__fits(H, h1_room) || !nn__primitives__fits(H, res_room))
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_TABLE);
    const uint64_t kvw = kvh * hd, qw = qh * hd, to = pos + 1u;
    const uint64_t rb8 = nn__turboquant__row_bytes(8u, hd), rb4 = nn__turboquant__row_bytes(4u, hd);
    const uint64_t past = to > SINK ? to - SINK : 0u;                         /* positions past the sink, this one in */
    const uint64_t n_sink = to < SINK ? to : SINK, n_hot = past < HOT ? past : HOT;
    const uint64_t n_warm = past > HOT ? (past - HOT < WARM ? past - HOT : WARM) : 0u;
    const uint64_t n_cold = past > HOT + WARM ? past - HOT - WARM : 0u;
    const uint64_t longest = n_cold > SINK ? (n_cold > HOT ? n_cold : HOT) : (SINK > HOT ? SINK : HOT);
    const uint64_t rw = 4u * qh * NN__ATTENTION__RESIDUAL_FLOATS(hd);            /* a residual, in bytes */
    if (!nn__primitives__fits(SINK * kvw, room[AI_QWEN_3__AT__K_CACHE]) || !nn__primitives__fits(SINK * kvw, room[AI_QWEN_3__AT__V_CACHE])
     || !nn__primitives__fits(HOT * kvw, room[AI_QWEN_3__AT__HOT_K]) || !nn__primitives__fits(HOT * kvw, room[AI_QWEN_3__AT__HOT_V])
     || WARM * kvh * rb8 > room[AI_QWEN_3__AT__WARM_KC] || WARM * kvh * rb8 > room[AI_QWEN_3__AT__WARM_VC]
     || !nn__primitives__fits(WARM * kvh, room[AI_QWEN_3__AT__WARM_KS]) || !nn__primitives__fits(WARM * kvh, room[AI_QWEN_3__AT__WARM_VS])
     || n_cold * kvh * rb4 > room[AI_QWEN_3__AT__COLD_KC] || n_cold * kvh * rb4 > room[AI_QWEN_3__AT__COLD_VC]
     || !nn__primitives__fits(n_cold * kvh, room[AI_QWEN_3__AT__COLD_KS]) || !nn__primitives__fits(n_cold * kvh, room[AI_QWEN_3__AT__COLD_VS])
     || room[AI_QWEN_3__AT__SCORES] / 4u < qh * (longest > WARM ? longest : WARM)
     || 4u * rw + 2u * (2u * qw + 3u * kvw) > room[AI_QWEN_3__AT__TIER_SCRATCH])
        return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    /* scratch, in halves: h · hrot · [qg · kk] · qn · gate · att · gated · arot · ao — the attention's own */
    const uint64_t need = 2u * H + 2u * qw + kvw + 5u * qw + H;
    if (!nn__primitives__fits(need, room[AI_QWEN_3__AT__SCRATCH])) return sys__engine__abi__error(AI_QWEN_3__MIXER__FAULT_ROOM);
    const uint64_t h = at[AI_QWEN_3__AT__SCRATCH], hrot = h + 2u * H, qg = hrot + 2u * H, kk = qg + 2u * 2u * qw;
    const uint64_t qn = kk + 2u * kvw, gate = qn + 2u * qw, att = gate + 2u * qw, gated = att + 2u * qw;
    const uint64_t arot = gated + 2u * qw, ao = arot + 2u * qw;
    /* the tiers' scratch: four residuals · the query rotated · the answer rotated · this key · this value · a row cooled */
    const uint64_t r0 = at[AI_QWEN_3__AT__TIER_SCRATCH], r1 = r0 + rw, r2 = r1 + rw, r3 = r2 + rw;
    const uint64_t qr = r3 + rw, attr = qr + 2u * qw, kt = attr + 2u * qw, vt = kt + 2u * kvw, cool = vt + 2u * kvw;
    const uint64_t signs = at[AI_QWEN_3__AT__SIGNS], cs = at[AI_QWEN_3__AT__ANGLES], p = at[AI_QWEN_3__AT__SCORES];
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;

    /* ① the projections, as the one-cache verb makes them — the value into `vt` rather than a cache row */
    const bool rotated = ai_qwen_3__table__zzpackage_flag(table, AI_QWEN_3__AT__RESIDUAL_ROTATED);
    doors->rmsnorm((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)x, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__INPUT_NORM], H, over);
    if (!rotated) doors->hadamard_rotate((uint16_t*)(uintptr_t)hrot, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)signs, H, over);
    const uint64_t in = rotated ? h : hrot;
    nn__turboquant__groups g = {};
    g.codes[0] = at[AI_QWEN_3__AT__Q]; g.luts[0] = at[AI_QWEN_3__AT__Q_LUT]; g.rows[0] = 2u * qw; g.d[0] = v[AI_QWEN_3__AT__D_Q]; g.out_at[0] = 0u;
    g.codes[1] = at[AI_QWEN_3__AT__K]; g.luts[1] = at[AI_QWEN_3__AT__K_LUT]; g.rows[1] = kvw; g.d[1] = v[AI_QWEN_3__AT__D_K]; g.out_at[1] = 2u * qw;
    g.count = 2u;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)qg, g, (const uint16_t*)(uintptr_t)in, H, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)vt, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__V], room[AI_QWEN_3__AT__V],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__V_LUT], room[AI_QWEN_3__AT__V_LUT],
                           (const uint16_t*)(uintptr_t)in, v[AI_QWEN_3__AT__D_V], kvw, H, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qg, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__Q_NORM],
                        hd, qh, 2u * hd, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qn, (const float*)(uintptr_t)cs, 1u, qh, hd, qw, rot, over);
    for (uint64_t k = 0u; k < qh; ++k)
        doors->sigmoid((uint16_t*)(uintptr_t)(gate + 2u * k * hd), (const uint16_t*)(uintptr_t)(qg + 2u * k * 2u * hd + 2u * hd), hd, over);
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)kt, (const uint16_t*)(uintptr_t)kk, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__K_NORM],
                        hd, kvh, hd, hd, over);
    doors->rope_rows((uint16_t*)(uintptr_t)kt, (const uint16_t*)(uintptr_t)kt, (const float*)(uintptr_t)cs, 1u, kvh, hd, kvw, rot, over);
    /* ② the query rotated; this key and value rotated into their place, what they displace cooled first */
    doors->hadamard_blocks((uint16_t*)(uintptr_t)qr, (const uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)signs, qw, hd, 0u, over);
    uint64_t kdst = at[AI_QWEN_3__AT__K_CACHE] + 2u * pos * kvw, vdst = at[AI_QWEN_3__AT__V_CACHE] + 2u * pos * kvw;
    if (pos >= SINK) {
        const uint64_t j = pos - SINK, slot = j % HOT;
        const uint64_t hk = at[AI_QWEN_3__AT__HOT_K] + 2u * slot * kvw, hv = at[AI_QWEN_3__AT__HOT_V] + 2u * slot * kvw;
        if (j >= HOT) {
            const uint64_t out = j - HOT, w = out % WARM;                       /* the position leaving the hot ring */
            if (out >= WARM) {                                                   /* and the one leaving the warm ring */
                const uint64_t old = out - WARM;
                const uint64_t sets[2][4] = {{AI_QWEN_3__AT__WARM_KC, AI_QWEN_3__AT__WARM_KS, AI_QWEN_3__AT__COLD_KC, AI_QWEN_3__AT__COLD_KS},
                                             {AI_QWEN_3__AT__WARM_VC, AI_QWEN_3__AT__WARM_VS, AI_QWEN_3__AT__COLD_VC, AI_QWEN_3__AT__COLD_VS}};
                for (unsigned s = 0u; s < 2u; ++s) {
                    (void)doors->turboquant_decode((uint16_t*)(uintptr_t)cool, (const uint8_t*)(uintptr_t)(at[sets[s][0]] + w * kvh * rb8), kvh * rb8,
                                                   (const uint8_t*)(uintptr_t)(at[sets[s][1]] + 2u * w * kvh), 2u * kvh, 8u, kvh, hd, over);
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
    doors->hadamard_blocks((uint16_t*)(uintptr_t)kdst, (const uint16_t*)(uintptr_t)kt, (const uint16_t*)(uintptr_t)signs, kvw, hd, 0u, over);
    doors->hadamard_blocks((uint16_t*)(uintptr_t)vdst, (const uint16_t*)(uintptr_t)vt, (const uint16_t*)(uintptr_t)signs, kvw, hd, 0u, over);
    /* ③ a residual a tier, merged into the first: the sink (never empty), the hot ring, the warm ring, the cold tier */
    doors->attention_weights((float*)(uintptr_t)p, (float*)(uintptr_t)r0, (const uint16_t*)(uintptr_t)qr,
                             (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__K_CACHE], qh, kvh, hd, n_sink);
    doors->attention_residual_mix((float*)(uintptr_t)r0, (const float*)(uintptr_t)p, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__V_CACHE],
                                  qh, kvh, hd, n_sink);
    if (n_hot != 0u) {
        doors->attention_weights((float*)(uintptr_t)p, (float*)(uintptr_t)r1, (const uint16_t*)(uintptr_t)qr,
                                 (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__HOT_K], qh, kvh, hd, n_hot);
        doors->attention_residual_mix((float*)(uintptr_t)r1, (const float*)(uintptr_t)p, (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__HOT_V],
                                      qh, kvh, hd, n_hot);
        doors->attention_merge((float*)(uintptr_t)r0, (const float*)(uintptr_t)r0, (const float*)(uintptr_t)r1, qh, hd);
    }
    if (n_warm != 0u) {
        doors->attention_weights_tq((float*)(uintptr_t)p, (float*)(uintptr_t)r2, (const uint16_t*)(uintptr_t)qr,
                                    (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__WARM_KC], (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__WARM_KS],
                                    qh, kvh, hd, n_warm, 8u);
        doors->attention_residual_mix_tq((float*)(uintptr_t)r2, (const float*)(uintptr_t)p,
                                         (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__WARM_VC], (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__WARM_VS],
                                         qh, kvh, hd, n_warm, 8u);
        doors->attention_merge((float*)(uintptr_t)r0, (const float*)(uintptr_t)r0, (const float*)(uintptr_t)r2, qh, hd);
    }
    if (n_cold != 0u) {
        doors->attention_weights_tq((float*)(uintptr_t)p, (float*)(uintptr_t)r3, (const uint16_t*)(uintptr_t)qr,
                                    (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__COLD_KC], (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__COLD_KS],
                                    qh, kvh, hd, n_cold, 4u);
        doors->attention_residual_mix_tq((float*)(uintptr_t)r3, (const float*)(uintptr_t)p,
                                         (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__COLD_VC], (const uint16_t*)(uintptr_t)at[AI_QWEN_3__AT__COLD_VS],
                                         qh, kvh, hd, n_cold, 4u);
        doors->attention_merge((float*)(uintptr_t)r0, (const float*)(uintptr_t)r0, (const float*)(uintptr_t)r3, qh, hd);
    }
    doors->attention_finish((uint16_t*)(uintptr_t)attr, (const float*)(uintptr_t)r0, qh, hd, over);
    /* ④ the answer rotated back a head at a time, then the one-cache verb's gate, rotation and out projection */
    doors->hadamard_blocks((uint16_t*)(uintptr_t)att, (const uint16_t*)(uintptr_t)attr, (const uint16_t*)(uintptr_t)signs, qw, hd, 1u, over);
    doors->vector_pointwise_mul((uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)att, (const uint16_t*)(uintptr_t)gate, qw, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)arot, (const uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)signs, qw, over);
    doors->turboquant_gemv((uint16_t*)(uintptr_t)ao, (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__O], room[AI_QWEN_3__AT__O],
                           (const uint8_t*)(uintptr_t)at[AI_QWEN_3__AT__O_LUT], room[AI_QWEN_3__AT__O_LUT],
                           (const uint16_t*)(uintptr_t)arot, v[AI_QWEN_3__AT__D_O], H, qw, over);
    const uint64_t masked = ai_qwen_3__table__zzpackage_masked(doors, over, argv[1].args[0], AI_QWEN_3__AT__MASK, ao, arot, 1u, H, qw,
                                                               AI_QWEN_3__MIXER__FAULT_TABLE);
    if (masked != 0u) return sys__engine__abi__error(masked);
    doors->vector_add((uint16_t*)(uintptr_t)h1, (const uint16_t*)(uintptr_t)res, (const uint16_t*)(uintptr_t)ao, H, over);
    return nn__doors_answer(&argv[3]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__attention_tiered__zzabi_adapter, ai_qwen_3__attention_tiered__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__deltanet__zzabi_adapter,  ai_qwen_3__deltanet__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__mlp__zzabi_adapter,       ai_qwen_3__mlp__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_3__attention__zzabi_adapter, ai_qwen_3__attention__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_OPCODES_MIXER__ABI_CUH */
