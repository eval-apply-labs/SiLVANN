#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_STEP__ABI_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_STEP__ABI_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../contracts/objects/step.cuh"     /* the transformer's table */
#include "../table.cuh"                          /* reading it */
#include "../rows.cuh"                           /* rows through a matrix */

/* ⭐ WHAT A STEP OF QWEN-IMAGE 2.1's TRANSFORMER DOES AROUND ITS BLOCKS, ON A WORKER — four verbs of nn's doors over the
 * transformer's own planes (▶ `contracts/objects/step.cuh` for the table and the forms). The arithmetic is diffusers'
 * (`QwenImage21Transformer2DModel.forward`, its `time_text_embed`, `txt_in`, `img_in`, `modulation`, `norm_out` and
 * `proj_out`), every product ▶ `../rows.cuh`. */

/* The table, its sizes, and the doors; false — `fault` set — if the table is not what it reads or the card is gone. */
typedef struct ai_qwen_image__step__table {
    uint64_t at[AI_QWEN_IMAGE__G__TABLE], room[AI_QWEN_IMAGE__G__TABLE], v[AI_QWEN_IMAGE__G__TABLE];
    uint64_t H, C, CH, F, R;
    const nn__doors* doors;
} ai_qwen_image__step__table;

static bool ai_qwen_image__step__zzprivate_read(const sys__heap_node* table, sys__engine__ctx* ctx,
                                                ai_qwen_image__step__table* t, uint64_t* fault) {
    *fault = AI_QWEN_IMAGE__STEP__FAULT_TABLE;
    if (!sys__heap_node__carries_reference(table->dtype) || !sys__node_array__is(table->args[0])) {
        *fault = SYS__OPCODES__FAULT_TYPE;
        return false;
    }
    if (!ai_qwen_image__table__zzpackage_read(table->args[0], AI_QWEN_IMAGE__G__HIDDEN, AI_QWEN_IMAGE__G__TABLE, t->at, t->room,
                                              t->v))
        return false;
    const uint64_t* v = t->v;
    t->H = v[AI_QWEN_IMAGE__G__HIDDEN]; t->C = v[AI_QWEN_IMAGE__G__CONTEXT]; t->CH = v[AI_QWEN_IMAGE__G__CHANNELS];
    t->F = v[AI_QWEN_IMAGE__G__FREQ]; t->R = v[AI_QWEN_IMAGE__G__ROWS];
    const uint64_t H = t->H, C = t->C, CH = t->CH, F = t->F, R = t->R, W = H > C ? H : C;
    if (H == 0u || H % 128u != 0u || C == 0u || CH == 0u || F == 0u || F % 2u != 0u || R == 0u || R > (1ull << 20)
     || !ai_qwen_image__rows__zzpackage_holds(t->room, AI_QWEN_IMAGE__G__IMG, v[AI_QWEN_IMAGE__G__D_IMG], H, CH)
     || !ai_qwen_image__rows__zzpackage_holds(t->room, AI_QWEN_IMAGE__G__TXT_IN, v[AI_QWEN_IMAGE__G__D_TXT_IN], H, C)
     || !ai_qwen_image__rows__zzpackage_holds(t->room, AI_QWEN_IMAGE__G__TXT_OUT, v[AI_QWEN_IMAGE__G__D_TXT_OUT], H, H)
     || !ai_qwen_image__rows__zzpackage_holds(t->room, AI_QWEN_IMAGE__G__T1, v[AI_QWEN_IMAGE__G__D_T1], H, F)
     || !ai_qwen_image__rows__zzpackage_holds(t->room, AI_QWEN_IMAGE__G__T2, v[AI_QWEN_IMAGE__G__D_T2], H, H)
     || !ai_qwen_image__rows__zzpackage_holds(t->room, AI_QWEN_IMAGE__G__MOD, v[AI_QWEN_IMAGE__G__D_MOD], 4u * H, H)
     || !ai_qwen_image__rows__zzpackage_holds(t->room, AI_QWEN_IMAGE__G__NORM_OUT, v[AI_QWEN_IMAGE__G__D_NORM_OUT], H, H)
     || !ai_qwen_image__rows__zzpackage_holds(t->room, AI_QWEN_IMAGE__G__PROJ, v[AI_QWEN_IMAGE__G__D_PROJ], CH, H)
     || t->room[AI_QWEN_IMAGE__G__TXT_NORM] < 2u * C || t->room[AI_QWEN_IMAGE__G__ONES] < 2u * H
     || t->room[AI_QWEN_IMAGE__G__MOD_ONE] < 8u * H || t->room[AI_QWEN_IMAGE__G__ZERO] < 2u * H
     || t->room[AI_QWEN_IMAGE__G__SIGNS] < 1024u
     || t->room[AI_QWEN_IMAGE__G__SCRATCH] < ai_qwen_image__rows__zzpackage_pairs_bytes(R) + 6u * R * W)
        return false;
    t->doors = nn__doors_for(ctx);
    if (t->doors == 0) {
        *fault = NN__PRIMITIVES__FAULT_NO_DEVICE;
        return false;
    }
    if (!ai_qwen_image__rows__zzpackage_pairs(ctx, t->at[AI_QWEN_IMAGE__G__SCRATCH], R)) {
        *fault = NN__PRIMITIVES__FAULT_NO_DEVICE;
        return false;
    }
    return true;
}

/* The scratch's three regions past the pairs, h, hrot and m: ROWS rows each of the wider of HIDDEN and CONTEXT. */
static void ai_qwen_image__step__zzprivate_regions(const ai_qwen_image__step__table* t, uint64_t* h, uint64_t* hrot,
                                                   uint64_t* m) {
    const uint64_t W = t->H > t->C ? t->H : t->C;
    *h = t->at[AI_QWEN_IMAGE__G__SCRATCH] + ai_qwen_image__rows__zzpackage_pairs_bytes(t->R);
    *hrot = *h + 2u * t->R * W;
    *m = *hrot + 2u * t->R * W;
}

/* `o = W·x` for a table's matrix at plane `codes`, its width the integer `d`, `x` rotated into `rot` first where it must be. */
static void ai_qwen_image__step__zzprivate_linear(const ai_qwen_image__step__table* t, unsigned codes, unsigned d,
                                                  uint64_t o, uint64_t x, uint64_t rot, uint64_t rows, uint64_t out_rows,
                                                  uint64_t cols, unsigned int* over) {
    const uint64_t in = ai_qwen_image__rows__zzpackage_rotated(t->doors, x, rot, t->at[AI_QWEN_IMAGE__G__SIGNS], t->v[d], cols,
                                                               rows, over);
    ai_qwen_image__rows__zzpackage_product(t->doors, o, t->at, t->room, codes, t->v[d], in, t->at[AI_QWEN_IMAGE__G__SCRATCH],
                                           rows, out_rows, cols, over);
}

/* `(ai_qwen_image__time sinusoids step mod row scale table)` -> `mod`. */
static sys__heap_node ai_qwen_image__time__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t sn = 0, sn_room = 0, md = 0, md_room = 0, sc = 0, sc_room = 0;
    if (argv[1].dtype != SYS__KIND__VALUE_INT || argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &sn, &sn_room) || !nn__primitives__room(&argv[2], &md, &md_room)
     || !nn__primitives__room(&argv[4], &sc, &sc_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_image__step__table t;
    uint64_t fault = 0;
    if (!ai_qwen_image__step__zzprivate_read(&argv[5], ctx, &t, &fault)) return sys__engine__abi__error(fault);
    const uint64_t step = argv[1].args[0], row = argv[3].args[0], H = t.H, F = t.F;
    if (row > 1u || step > (1ull << 20) || sn_room < 2u * F * (step + 1u) || md_room < 16u * H || sc_room < 2u * H)
        return sys__engine__abi__error(AI_QWEN_IMAGE__STEP__FAULT_ROOM);
    const nn__doors* doors = t.doors;
    unsigned int* over = ctx->fault_word;
    uint64_t h, hrot, m;
    ai_qwen_image__step__zzprivate_regions(&t, &h, &hrot, &m);
    const uint64_t ones = t.at[AI_QWEN_IMAGE__G__ONES], mrow = md + 8u * row * H;
    /* temb = W₂·silu(W₁·sinusoid), then silu(temb): what the modulation and the last norm read */
    ai_qwen_image__step__zzprivate_linear(&t, AI_QWEN_IMAGE__G__T1, AI_QWEN_IMAGE__G__D_T1, h, sn + 2u * F * step, hrot, 1u, H, F,
                                          over);
    doors->swiglu_combine((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)ones, H, over);
    ai_qwen_image__step__zzprivate_linear(&t, AI_QWEN_IMAGE__G__T2, AI_QWEN_IMAGE__G__D_T2, m, h, hrot, 1u, H, H, over);
    doors->swiglu_combine((uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)ones, H, over);
    /* the modulation's row: 1 + the scales, tanh of the gates */
    ai_qwen_image__step__zzprivate_linear(&t, AI_QWEN_IMAGE__G__MOD, AI_QWEN_IMAGE__G__D_MOD, mrow, m, hrot, 1u, 4u * H, H, over);
    doors->vector_bias_rows((uint16_t*)(uintptr_t)mrow, (const uint16_t*)(uintptr_t)mrow,
                            (const uint16_t*)(uintptr_t)t.at[AI_QWEN_IMAGE__G__MOD_ONE], 4u * H, 1u, over);
    for (uint64_t g = 1u; g < 4u; g += 2u)
        doors->vector_tanh((uint16_t*)(uintptr_t)(mrow + 2u * g * H), (const uint16_t*)(uintptr_t)(mrow + 2u * g * H), H, over);
    /* the last norm's scale, 1 + W·silu(temb) */
    ai_qwen_image__step__zzprivate_linear(&t, AI_QWEN_IMAGE__G__NORM_OUT, AI_QWEN_IMAGE__G__D_NORM_OUT, sc, m, hrot, 1u, H, H,
                                          over);
    doors->vector_bias_rows((uint16_t*)(uintptr_t)sc, (const uint16_t*)(uintptr_t)sc, (const uint16_t*)(uintptr_t)ones, H, 1u,
                            over);
    return nn__doors_answer(&argv[2]);
}

/* `(ai_qwen_image__text encoder x table rows from first)` -> `x`. */
static sys__heap_node ai_qwen_image__text__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t en = 0, en_room = 0, x = 0, x_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT || argv[5].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &en, &en_room) || !nn__primitives__room(&argv[1], &x, &x_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_image__step__table t;
    uint64_t fault = 0;
    if (!ai_qwen_image__step__zzprivate_read(&argv[2], ctx, &t, &fault)) return sys__engine__abi__error(fault);
    const uint64_t rows = argv[3].args[0], from = argv[4].args[0], first = argv[5].args[0], H = t.H, C = t.C;
    if (rows == 0u || rows > (1ull << 20) || from > (1ull << 20) || first > (1ull << 20)
     || !nn__primitives__fits((from + rows) * C, en_room) || !nn__primitives__fits((first + rows) * H, x_room))
        return sys__engine__abi__error(AI_QWEN_IMAGE__STEP__FAULT_ROOM);
    unsigned int* over = ctx->fault_word;
    uint64_t h, hrot, m;
    ai_qwen_image__step__zzprivate_regions(&t, &h, &hrot, &m);
    for (uint64_t r0 = 0u; r0 < rows; r0 += t.R) {
        const uint64_t n = rows - r0 < t.R ? rows - r0 : t.R;
        t.doors->rmsnorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)(en + 2u * (from + r0) * C),
                              (const uint16_t*)(uintptr_t)t.at[AI_QWEN_IMAGE__G__TXT_NORM], C, n, C, C, AI_QWEN_IMAGE__EPS, over);
        ai_qwen_image__step__zzprivate_linear(&t, AI_QWEN_IMAGE__G__TXT_IN, AI_QWEN_IMAGE__G__D_TXT_IN, m, h, hrot, n, H, C, over);
        t.doors->vector_gelu((uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)m, n * H, 0u, over);
        ai_qwen_image__step__zzprivate_linear(&t, AI_QWEN_IMAGE__G__TXT_OUT, AI_QWEN_IMAGE__G__D_TXT_OUT,
                                              x + 2u * (first + r0) * H, m, hrot, n, H, H, over);
    }
    return nn__doors_answer(&argv[1]);
}

/* `(ai_qwen_image__latents latents x table rows from first)` -> `x`. */
static sys__heap_node ai_qwen_image__latents__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t la = 0, la_room = 0, x = 0, x_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT || argv[5].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &la, &la_room) || !nn__primitives__room(&argv[1], &x, &x_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_image__step__table t;
    uint64_t fault = 0;
    if (!ai_qwen_image__step__zzprivate_read(&argv[2], ctx, &t, &fault)) return sys__engine__abi__error(fault);
    const uint64_t rows = argv[3].args[0], from = argv[4].args[0], first = argv[5].args[0], H = t.H, CH = t.CH;
    if (rows == 0u || rows > (1ull << 20) || from > (1ull << 20) || first > (1ull << 20)
     || !nn__primitives__fits((from + rows) * CH, la_room) || !nn__primitives__fits((first + rows) * H, x_room))
        return sys__engine__abi__error(AI_QWEN_IMAGE__STEP__FAULT_ROOM);
    unsigned int* over = ctx->fault_word;
    uint64_t h, hrot, m;
    ai_qwen_image__step__zzprivate_regions(&t, &h, &hrot, &m);
    for (uint64_t r0 = 0u; r0 < rows; r0 += t.R) {
        const uint64_t n = rows - r0 < t.R ? rows - r0 : t.R;
        ai_qwen_image__step__zzprivate_linear(&t, AI_QWEN_IMAGE__G__IMG, AI_QWEN_IMAGE__G__D_IMG, x + 2u * (first + r0) * H,
                                              la + 2u * (from + r0) * CH, hrot, n, H, CH, over);
    }
    return nn__doors_answer(&argv[1]);
}

/* `(ai_qwen_image__out x scale prediction latents table rows first dsigma)` -> `latents`. */
static sys__heap_node ai_qwen_image__out__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 8u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, sc = 0, sc_room = 0, pr = 0, pr_room = 0, la = 0, la_room = 0;
    if (argv[5].dtype != SYS__KIND__VALUE_INT || argv[6].dtype != SYS__KIND__VALUE_INT || argv[7].dtype != SYS__KIND__VALUE_FLOAT
     || !nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[1], &sc, &sc_room)
     || !nn__primitives__room(&argv[2], &pr, &pr_room) || !nn__primitives__room(&argv[3], &la, &la_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_image__step__table t;
    uint64_t fault = 0;
    if (!ai_qwen_image__step__zzprivate_read(&argv[4], ctx, &t, &fault)) return sys__engine__abi__error(fault);
    const uint64_t rows = argv[5].args[0], first = argv[6].args[0], H = t.H, CH = t.CH;
    const float dsigma = (float)sys__heap_node__real(argv[7].args[0]);
    if (rows == 0u || rows > (1ull << 20) || first > (1ull << 20) || !nn__primitives__fits((first + rows) * H, x_room)
     || sc_room < 2u * H
     || !nn__primitives__fits(rows * CH, pr_room) || !nn__primitives__fits(rows * CH, la_room))
        return sys__engine__abi__error(AI_QWEN_IMAGE__STEP__FAULT_ROOM);
    unsigned int* over = ctx->fault_word;
    uint64_t h, hrot, m;
    ai_qwen_image__step__zzprivate_regions(&t, &h, &hrot, &m);
    for (uint64_t r0 = 0u; r0 < rows; r0 += t.R) {
        const uint64_t n = rows - r0 < t.R ? rows - r0 : t.R, p = pr + 2u * r0 * CH, l = la + 2u * r0 * CH;
        t.doors->layernorm_rows((uint16_t*)(uintptr_t)h, (const uint16_t*)(uintptr_t)(x + 2u * (first + r0) * H),
                                (const uint16_t*)(uintptr_t)sc,
                                (const uint16_t*)(uintptr_t)t.at[AI_QWEN_IMAGE__G__ZERO], H, n, H, H, AI_QWEN_IMAGE__EPS, over);
        ai_qwen_image__step__zzprivate_linear(&t, AI_QWEN_IMAGE__G__PROJ, AI_QWEN_IMAGE__G__D_PROJ, p, h, hrot, n, CH, H, over);
        /* the flow's step: latents + dsigma·prediction */
        t.doors->vector_scale((uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)p, n * CH, dsigma, over);
        t.doors->vector_add((uint16_t*)(uintptr_t)l, (const uint16_t*)(uintptr_t)l, (const uint16_t*)(uintptr_t)m, n * CH, over);
    }
    return nn__doors_answer(&argv[3]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_image__time__zzabi_adapter, ai_qwen_image__time__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_image__text__zzabi_adapter, ai_qwen_image__text__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_image__latents__zzabi_adapter, ai_qwen_image__latents__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_image__out__zzabi_adapter, ai_qwen_image__out__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_OPCODES_STEP__ABI_CUH */
