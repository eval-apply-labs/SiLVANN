#ifndef SILVANN__PACKAGES_AI_QWEN_4_CPU_OPCODES_LAYER__ABI_CUH
#define SILVANN__PACKAGES_AI_QWEN_4_CPU_OPCODES_LAYER__ABI_CUH
/* ══ QWEN3.8-FLASH-NEXT's LAYER SITES, A VERB EACH ═══════════════════════════════════════════════════════════
 * Each verb launches the doors its site's nn words in `lisp/qwen4_exp.lisp` launch, in their order, so the two give the
 * same bits; what a verb removes is the evaluator's work between them — the ranges, the loops, the table reads, ~31,000
 * applies a position over the words (MEASURED node03 2026-10-31, measurements/2026-10-31_nn141_flash_next_node03.md).
 * A matrix is read plain where its record is exact (d = 16) and rotated otherwise, as the program's `mv` reads it.
 * ▶ `contracts/objects/layer.cuh` for the tables and the forms. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../table.cuh"                                  /* the layer's table and the shape table */
#include <string.h>
#include "../../../nn/contracts/objects/deltanet.cuh"    /* the DeltaNet's window and its output gate */

/* A piece of the scratch for `halves` halves, every piece on a 256-byte boundary. */
static inline uint64_t ai_qwen_4__layer__zzprivate_take(uint64_t* next, uint64_t halves) {
    const uint64_t at = *next;
    *next += (2u * halves + 255u) & ~255ull;
    return at;
}

/* The matrix at cell `c` times `plain` (its record exact) or `rot` (quantised, rotated along its input) into `out`. */
static inline void ai_qwen_4__layer__zzprivate_mv(const nn__doors* doors, unsigned int* over, const ai_qwen_4__table* t,
                                                 unsigned c, uint64_t plain, uint64_t rot, uint64_t out, uint64_t rows,
                                                 uint64_t cols) {
    const uint64_t d = t->v[c + 2u];
    doors->turboquant_gemv((uint16_t*)(uintptr_t)out, (const uint8_t*)(uintptr_t)t->at[c], t->room[c],
                           (const uint8_t*)(uintptr_t)t->at[c + 1u], t->room[c + 1u],
                           (const uint16_t*)(uintptr_t)(d == 16u ? plain : rot), d, rows, cols, over);
}

/* ⭐ SEVERAL MATRICES OVER ONE INPUT AS ONE LAUNCH — nn's groups door, each matrix's answer the one-matrix door's (the door
 *   bench's `own_groups_case`: Flash-Next's DeltaNet and attention maps byte for byte, measurements/2026-11-01_*): the `n`
 *   matrices at cells `c[i]`, their answers at `outs[i]` (each after `outs[0]`, its rows apart), all quantised (4 or 8
 *   bits) over the rotated `rot` — or, where one is not or its record holds fewer rows than it is asked for, each through
 *   `mv` as before. */
static inline void ai_qwen_4__layer__zzprivate_group(const nn__doors* doors, unsigned int* over, const ai_qwen_4__table* t,
                                                    const unsigned* c, const uint64_t* outs, const uint64_t* rows, uint64_t n,
                                                    uint64_t plain, uint64_t rot, uint64_t cols) {
    bool fits = n <= NN__TURBOQUANT__GROUPS_MAX;
    nn__turboquant__groups g;
    memset(&g, 0, sizeof g);
    for (uint64_t i = 0u; fits && i < n; ++i) {
        const uint64_t d = t->v[c[i] + 2u], bytes = rows[i] * (cols * d / 8u);
        fits = (d == 4u || d == 8u) && cols % 32u == 0u && outs[i] >= outs[0] && bytes <= t->room[c[i]]
            && 2u * rows[i] <= t->room[c[i] + 1u];
        g.codes[i] = t->at[c[i]]; g.luts[i] = t->at[c[i] + 1u]; g.rows[i] = rows[i]; g.d[i] = d;
        g.out_at[i] = (outs[i] - outs[0]) / 2u;
    }
    if (!fits) {
        for (uint64_t i = 0u; i < n; ++i) ai_qwen_4__layer__zzprivate_mv(doors, over, t, c[i], plain, rot, outs[i], rows[i], cols);
        return;
    }
    g.count = n;
    doors->turboquant_gemv_groups((uint16_t*)(uintptr_t)outs[0], g, (const uint16_t*)(uintptr_t)rot, cols, over);
}

/* The operands every verb opens with: the layer's table (argument `table_at`) and the shape (the last argument). */
static inline uint64_t ai_qwen_4__layer__zzprivate_open(const sys__heap_node* argv, unsigned argc, unsigned table_at,
                                                       ai_qwen_4__table* t, ai_qwen_4__shape* s) {
    if (!sys__heap_node__carries_reference(argv[table_at].dtype)) return SYS__OPCODES__FAULT_TYPE;
    if (!ai_qwen_4__table__zzpackage_read(argv[table_at].args[0], t) || !ai_qwen_4__table__zzpackage_shape(&argv[argc - 1u], s))
        return AI_QWEN_4__LAYER__FAULT_TABLE;
    return 0ull;
}

/* ── `(ai_qwen_4__gated_pre streams table site into inj shape)` -> `into` ─────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__gated_pre__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t sf = 0, sf_room = 0, into = 0, into_room = 0, inj = 0, inj_room = 0;
    const bool injects = argv[4].dtype != SYS__KIND__VALUE_INT;
    if (argv[2].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &sf, &sf_room)
     || !nn__primitives__room(&argv[3], &into, &into_room)
     || (injects ? !nn__primitives__room(&argv[4], &inj, &inj_room) : argv[4].args[0] != 0ull))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 1u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], M = s.v[AI_QWEN_4__SH__STREAMS], R = s.v[AI_QWEN_4__SH__RANK], S = M * H;
    const uint64_t c = argv[2].args[0];
    if (c + AI_QWEN_4__SITE__INJ + 2u >= AI_QWEN_4__LT__TABLE || R != AI_QWEN_4__RANK_FIRST + AI_QWEN_4__RANK_LAST
     || !ai_qwen_4__table__zzpackage_vector(&t, (unsigned)c, S)
     || !ai_qwen_4__table__zzpackage_matrix(&t, (unsigned)c + AI_QWEN_4__SITE__DOWN)
     || !ai_qwen_4__table__zzpackage_matrix(&t, (unsigned)c + AI_QWEN_4__SITE__UP)
     || (injects && !ai_qwen_4__table__zzpackage_matrix(&t, (unsigned)c + AI_QWEN_4__SITE__INJ)))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t scratch = s.at[AI_QWEN_4__SH__SCRATCH], signs = s.at[AI_QWEN_4__SH__SIGNS];
    uint64_t next = scratch;
    const uint64_t an = ai_qwen_4__layer__zzprivate_take(&next, S), anr = ai_qwen_4__layer__zzprivate_take(&next, S);
    const uint64_t a1 = ai_qwen_4__layer__zzprivate_take(&next, R);
    const uint64_t a3 = ai_qwen_4__layer__zzprivate_take(&next, R), a3r = ai_qwen_4__layer__zzprivate_take(&next, R);
    const uint64_t gt = ai_qwen_4__layer__zzprivate_take(&next, S), ij = ai_qwen_4__layer__zzprivate_take(&next, M);
    if (4u * S > sf_room || !nn__primitives__fits(H, into_room) || (injects && !nn__primitives__fits(M, inj_room))
     || next - scratch > s.room[AI_QWEN_4__SH__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const float eps = s.r[AI_QWEN_4__SH__EPS], mean = (float)(1.0 / (double)M);
    /* each stream normed by its slice of the gain — the M streams as M rows in one launch, row j by slice j, each the bytes
     * the one-row door gives it (nn's `rmsnorm_fp32_rows`) — the mix's down map, silu, the rank rotated as its 256 and its
     * 128 */
    doors->rmsnorm_fp32_rows((uint16_t*)(uintptr_t)an, (const float*)(uintptr_t)sf, (const uint16_t*)(uintptr_t)t.at[c], H, M, M,
                             eps, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)anr, (const uint16_t*)(uintptr_t)an, (const uint16_t*)(uintptr_t)signs, S, over);
    ai_qwen_4__layer__zzprivate_mv(doors, over, &t, (unsigned)c + AI_QWEN_4__SITE__DOWN, an, anr, a1, R, S);
    /* the injection map beside it, over the same normed streams (its weights come with the mean below) */
    if (injects) ai_qwen_4__layer__zzprivate_mv(doors, over, &t, (unsigned)c + AI_QWEN_4__SITE__INJ, an, anr, ij, M, S);
    /* the rank: ¼ of the down map, silu, rotated as its 256 and its 128 — nn's `gated_rank`, one launch for the four doors
     * it answers as */
    doors->gated_rank((uint16_t*)(uintptr_t)a3r, (uint16_t*)(uintptr_t)a3, (const uint16_t*)(uintptr_t)a1,
                      (const uint16_t*)(uintptr_t)signs, R, mean, over);
    /* the up map; its sigmoid times the normed streams, their mean, and the four injection weights 2·σ(¼·W·n) — nn's
     * `gated_mean`, one launch for the eight doors it answers as */
    ai_qwen_4__layer__zzprivate_mv(doors, over, &t, (unsigned)c + AI_QWEN_4__SITE__UP, a3, a3r, gt, S, R);
    doors->gated_mean((uint16_t*)(uintptr_t)into, injects ? (uint16_t*)(uintptr_t)inj : (uint16_t*)0, (const uint16_t*)(uintptr_t)gt,
                      (const uint16_t*)(uintptr_t)an, injects ? (const uint16_t*)(uintptr_t)ij : (const uint16_t*)0, H, M, mean, over);
    return nn__doors_answer(&argv[3]);
}

/* ── `(ai_qwen_4__attention m table kc vc pos y shape)` -> `y` ────────────────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__attention__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t m = 0, m_room = 0, kc = 0, kc_room = 0, vc = 0, vc_room = 0, y = 0, y_room = 0;
    if (argv[4].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &m, &m_room)
     || !nn__primitives__room(&argv[2], &kc, &kc_room) || !nn__primitives__room(&argv[3], &vc, &vc_room)
     || !nn__primitives__room(&argv[5], &y, &y_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 1u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], QH = s.v[AI_QWEN_4__SH__QH], KVH = s.v[AI_QWEN_4__SH__KVH];
    const uint64_t AD = s.v[AI_QWEN_4__SH__AD], ROT = s.v[AI_QWEN_4__SH__ROT], QW = QH * AD, KVW = KVH * AD;
    const uint64_t pos = argv[4].args[0], length = pos + 1u;
    if (QH % KVH != 0u || ROT > AD || ROT % 2u != 0u || pos >= (1ull << 24)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__Q) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__K)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__V) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__O)
     || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__QN, AD) || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__KN, AD))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t scratch = s.at[AI_QWEN_4__SH__SCRATCH], signs = s.at[AI_QWEN_4__SH__SIGNS], cs = s.at[AI_QWEN_4__SH__CS];
    const uint64_t scores = s.at[AI_QWEN_4__SH__SCORES];
    uint64_t next = scratch;
    const uint64_t mr = ai_qwen_4__layer__zzprivate_take(&next, H), qg = ai_qwen_4__layer__zzprivate_take(&next, 2u * QW);
    const uint64_t kk = ai_qwen_4__layer__zzprivate_take(&next, KVW), vv = ai_qwen_4__layer__zzprivate_take(&next, KVW);
    const uint64_t qb = ai_qwen_4__layer__zzprivate_take(&next, QW), gsig = ai_qwen_4__layer__zzprivate_take(&next, QW);
    const uint64_t ao = ai_qwen_4__layer__zzprivate_take(&next, QW), aog = ai_qwen_4__layer__zzprivate_take(&next, QW);
    const uint64_t aogr = ai_qwen_4__layer__zzprivate_take(&next, QW), sg = ai_qwen_4__layer__zzprivate_take(&next, 2u * QW);
    if (!nn__primitives__fits(H, m_room) || !nn__primitives__fits(H, y_room) || !nn__primitives__fits(length * KVW, kc_room)
     || !nn__primitives__fits(length * KVW, vc_room) || !nn__primitives__fits(2u * QH * length, s.room[AI_QWEN_4__SH__SCORES])
     || !nn__primitives__fits(2u * ROT, s.room[AI_QWEN_4__SH__CS]) || next - scratch > s.room[AI_QWEN_4__SH__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const float eps = s.r[AI_QWEN_4__SH__EPS];
    doors->hadamard_rotate((uint16_t*)(uintptr_t)mr, (const uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)signs, H, over);
    const unsigned in_c[3] = { AI_QWEN_4__LT__Q, AI_QWEN_4__LT__K, AI_QWEN_4__LT__V };
    const uint64_t in_out[3] = { qg, kk, vv }, in_rows[3] = { 2u * QW, KVW, KVW };
    ai_qwen_4__layer__zzprivate_group(doors, over, &t, in_c, in_out, in_rows, 3u, m, mr, H);
    /* every query head — its q then its gate, AD halves each — normed and roped on its first ROT in one launch each, the
     * sigmoid over q and gate alike and the gates' halves gathered: a head's arithmetic is the one-head doors' (a row of
     * `rmsnorm_rows` is `rmsnorm`, a head of `rope_rows` is `rope`), 6 launches where a head at a time took 76 */
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)qb, (const uint16_t*)(uintptr_t)qg, (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__QN],
                        AD, QH, 2u * AD, AD, eps, over);
    doors->rope_rows((uint16_t*)(uintptr_t)qb, (const uint16_t*)(uintptr_t)qb, (const float*)(uintptr_t)cs, 1u, QH, AD, QW, ROT, over);
    doors->sigmoid((uint16_t*)(uintptr_t)sg, (const uint16_t*)(uintptr_t)qg, 2u * QW, over);
    doors->vector_copy_rows((uint16_t*)(uintptr_t)gsig, AD, (const uint16_t*)(uintptr_t)(sg + 2u * AD), 2u * AD, AD, QH);
    /* every key head normed and roped into the cache's row for `pos`, the values beside them */
    const uint64_t k = kc + 2u * pos * KVW;
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)k, (const uint16_t*)(uintptr_t)kk, (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__KN],
                        AD, KVH, AD, AD, eps, over);
    doors->rope_rows((uint16_t*)(uintptr_t)k, (const uint16_t*)(uintptr_t)k, (const float*)(uintptr_t)cs, 1u, KVH, AD, KVW, ROT, over);
    doors->vector_copy((uint16_t*)(uintptr_t)(vc + 2u * pos * KVW), (const uint16_t*)(uintptr_t)vv, KVW);
    /* every position 0 … pos, as nn's `decode` attends them */
    doors->attention_scores((float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)qb, (const uint16_t*)(uintptr_t)kc, QH, KVH, AD, length);
    doors->attention_mix((uint16_t*)(uintptr_t)ao, (const float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)vc, QH, KVH, AD, length, over);
    doors->vector_pointwise_mul((uint16_t*)(uintptr_t)aog, (const uint16_t*)(uintptr_t)ao, (const uint16_t*)(uintptr_t)gsig, QW, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)aogr, (const uint16_t*)(uintptr_t)aog, (const uint16_t*)(uintptr_t)signs, QW, over);
    ai_qwen_4__layer__zzprivate_mv(doors, over, &t, AI_QWEN_4__LT__O, aog, aogr, y, H, QW);
    return nn__doors_answer(&argv[5]);
}

/* ── `(ai_qwen_4__deltanet m table s32 cwin y shape)` -> `y` ──────────────────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__deltanet__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t m = 0, m_room = 0, s32 = 0, s32_room = 0, cwin = 0, cwin_room = 0, y = 0, y_room = 0;
    if (!nn__primitives__room(&argv[0], &m, &m_room) || !nn__primitives__room(&argv[2], &s32, &s32_room)
     || !nn__primitives__room(&argv[3], &cwin, &cwin_room) || !nn__primitives__room(&argv[4], &y, &y_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 1u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], KH = s.v[AI_QWEN_4__SH__KH], VH = s.v[AI_QWEN_4__SH__VH];
    const uint64_t LD = s.v[AI_QWEN_4__SH__LD], VAL = VH * LD, CH = 2u * KH * LD + VAL;
    if (VH % KH != 0u || LD > 4096u
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__QKV) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__Z)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__A) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__B)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__OUT)
     || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__CONV, CH * NN__DELTANET__CONV_TAPS)
     || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__ALOG, VH) || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__DTB, VH)
     || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__GNORM, LD))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t scratch = s.at[AI_QWEN_4__SH__SCRATCH], signs = s.at[AI_QWEN_4__SH__SIGNS];
    uint64_t next = scratch;
    const uint64_t mr = ai_qwen_4__layer__zzprivate_take(&next, H), mixed = ai_qwen_4__layer__zzprivate_take(&next, CH);
    const uint64_t z = ai_qwen_4__layer__zzprivate_take(&next, VAL), av = ai_qwen_4__layer__zzprivate_take(&next, VH);
    const uint64_t bv = ai_qwen_4__layer__zzprivate_take(&next, VH), beta = ai_qwen_4__layer__zzprivate_take(&next, VH);
    const uint64_t gb = ai_qwen_4__layer__zzprivate_take(&next, VH), conved = ai_qwen_4__layer__zzprivate_take(&next, CH);
    const uint64_t core = ai_qwen_4__layer__zzprivate_take(&next, VAL), crot = ai_qwen_4__layer__zzprivate_take(&next, VAL);
    if (!nn__primitives__fits(H, m_room) || !nn__primitives__fits(H, y_room) || 4u * VH * LD * LD > s32_room
     || !nn__primitives__fits(CH * NN__DELTANET__CONV_WINDOW, cwin_room) || next - scratch > s.room[AI_QWEN_4__SH__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    doors->hadamard_rotate((uint16_t*)(uintptr_t)mr, (const uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)signs, H, over);
    /* the four input maps over the one rotated input, one launch */
    const unsigned in_c[4] = { AI_QWEN_4__LT__QKV, AI_QWEN_4__LT__Z, AI_QWEN_4__LT__A, AI_QWEN_4__LT__B };
    const uint64_t in_out[4] = { mixed, z, av, bv }, in_rows[4] = { CH, VAL, VH, VH };
    ai_qwen_4__layer__zzprivate_group(doors, over, &t, in_c, in_out, in_rows, 4u, m, mr, H);
    /* β = σ(b), g = −exp(A_log)·softplus(a + dt_bias) — nn's `deltanet_gates_rows` at one row, the six doors' bytes (the door
     * bench's `gates_case`) */
    doors->deltanet_gates_rows((uint16_t*)(uintptr_t)beta, (uint16_t*)(uintptr_t)gb, (const uint16_t*)(uintptr_t)av,
                               (const uint16_t*)(uintptr_t)bv, (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__ALOG],
                               (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__DTB], VH, 1u, over);
    /* the short conv, then every head's step in one launch, its output gated by σ(z) */
    doors->deltanet_conv_step((uint16_t*)(uintptr_t)conved, (uint16_t*)(uintptr_t)cwin, (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__CONV],
                              (const uint16_t*)(uintptr_t)mixed, CH, over);
    doors->deltanet_steps_gate((float*)(uintptr_t)s32, (const uint16_t*)(uintptr_t)conved, (const uint16_t*)(uintptr_t)z,
                               (const uint16_t*)(uintptr_t)beta, (const uint16_t*)(uintptr_t)gb,
                               (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__GNORM], (uint16_t*)(uintptr_t)core, KH, VH, LD, 1u,
                               (unsigned int)NN__DELTANET__GATE_SIGMOID, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)crot, (const uint16_t*)(uintptr_t)core, (const uint16_t*)(uintptr_t)signs, VAL, over);
    ai_qwen_4__layer__zzprivate_mv(doors, over, &t, AI_QWEN_4__LT__OUT, core, crot, y, H, VAL);
    return nn__doors_answer(&argv[4]);
}

/* ── `(ai_qwen_4__shared x xr table out shape)` -> `out` ──────────────────────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__shared__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, xr = 0, xr_room = 0, out = 0, out_room = 0;
    if (!nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[1], &xr, &xr_room)
     || !nn__primitives__room(&argv[3], &out, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 2u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], I = s.v[AI_QWEN_4__SH__INTER];
    if (!ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__SG) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__SU)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__SD) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__SEG))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t scratch = s.at[AI_QWEN_4__SH__SCRATCH], signs = s.at[AI_QWEN_4__SH__SIGNS];
    uint64_t next = scratch;
    const uint64_t sgo = ai_qwen_4__layer__zzprivate_take(&next, I), suo = ai_qwen_4__layer__zzprivate_take(&next, I);
    const uint64_t sact = ai_qwen_4__layer__zzprivate_take(&next, I), sactr = ai_qwen_4__layer__zzprivate_take(&next, I);
    const uint64_t sdo = ai_qwen_4__layer__zzprivate_take(&next, H), segv = ai_qwen_4__layer__zzprivate_take(&next, 1u);
    const uint64_t segs = ai_qwen_4__layer__zzprivate_take(&next, 1u);
    if (!nn__primitives__fits(H, x_room) || !nn__primitives__fits(H, xr_room) || !nn__primitives__fits(H, out_room)
     || next - scratch > s.room[AI_QWEN_4__SH__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    ai_qwen_4__layer__zzprivate_mv(doors, over, &t, AI_QWEN_4__LT__SG, x, xr, sgo, I, H);
    ai_qwen_4__layer__zzprivate_mv(doors, over, &t, AI_QWEN_4__LT__SU, x, xr, suo, I, H);
    doors->swiglu_combine((uint16_t*)(uintptr_t)sact, (const uint16_t*)(uintptr_t)sgo, (const uint16_t*)(uintptr_t)suo, I, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)sactr, (const uint16_t*)(uintptr_t)sact, (const uint16_t*)(uintptr_t)signs, I, over);
    ai_qwen_4__layer__zzprivate_mv(doors, over, &t, AI_QWEN_4__LT__SD, sact, sactr, sdo, H, I);
    ai_qwen_4__layer__zzprivate_mv(doors, over, &t, AI_QWEN_4__LT__SEG, x, xr, segv, 1u, H);
    doors->sigmoid((uint16_t*)(uintptr_t)segs, (const uint16_t*)(uintptr_t)segv, 1u, over);
    doors->vector_scale_at((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)sdo, (const uint16_t*)(uintptr_t)segs, 0u, H, over);
    return nn__doors_answer(&argv[3]);
}

/* ══ ⭐ A PROMPT'S CHUNK, ITS ROWS AT ONCE ═════════════════════════════════════════════════════════════════════
 * The sites over `n` rows of a chunk, a row a position: each matrix one GEMM over every row (nn's `expert_rows`, the rows a
 * pair names), the recurrence and the attention walking the rows in order inside their launches, the streams' norms and
 * mixes — fp32, a row's own — a launch a row. ▶ `contracts/objects/layer.cuh`. The pairs `(r, r)` are written on the
 * worker into the front of the scratch, queued like the launches. */

/* `n`, the chunk's rows: an integer from 1 to the rows' shape's ROWS. */
static inline bool ai_qwen_4__rows__zzprivate_n(const sys__heap_node* cell, const ai_qwen_4__shape* s, uint64_t* n) {
    if (cell->dtype != SYS__KIND__VALUE_INT || cell->args[0] == 0u || cell->args[0] > s->v[AI_QWEN_4__SH__ROWS]) return false;
    *n = cell->args[0];
    return true;
}

/* The matrix at cell `c` over `n` rows of `plain` (its record exact) or `rot` (rotated), the rows `pairs` names. */
static inline void ai_qwen_4__rows__zzprivate_gemm(const nn__doors* doors, unsigned int* over, const ai_qwen_4__table* t, unsigned c,
                                                   uint64_t plain, uint64_t rot, uint64_t out, uint64_t pairs, uint64_t n,
                                                   uint64_t rows, uint64_t cols) {
    const uint64_t d = t->v[c + 2u];
    doors->expert_rows((uint16_t*)(uintptr_t)out, (const uint8_t*)(uintptr_t)t->at[c], t->room[c],
                       (const uint8_t*)(uintptr_t)t->at[c + 1u], t->room[c + 1u],
                       (const uint16_t*)(uintptr_t)(d == 16u ? plain : rot), (const uint32_t*)(uintptr_t)pairs, n, d, rows, cols, over);
}

/* ── `(ai_qwen_4__gated_pre_rows streams table site into inj n shape)` -> `into` ─────────────────────────────────── */
static sys__heap_node ai_qwen_4__gated_pre_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t sf = 0, sf_room = 0, into = 0, into_room = 0, inj = 0, inj_room = 0;
    const bool injects = argv[4].dtype != SYS__KIND__VALUE_INT;
    if (argv[2].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &sf, &sf_room)
     || !nn__primitives__room(&argv[3], &into, &into_room)
     || (injects ? !nn__primitives__room(&argv[4], &inj, &inj_room) : argv[4].args[0] != 0ull))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    uint64_t n = 0u;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 1u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    if (!ai_qwen_4__rows__zzprivate_n(&argv[5], &s, &n)) return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], M = s.v[AI_QWEN_4__SH__STREAMS], R = s.v[AI_QWEN_4__SH__RANK], S = M * H;
    const uint64_t c = argv[2].args[0];
    if (c + AI_QWEN_4__SITE__INJ + 2u >= AI_QWEN_4__LT__TABLE || R != AI_QWEN_4__RANK_FIRST + AI_QWEN_4__RANK_LAST
     || !ai_qwen_4__table__zzpackage_vector(&t, (unsigned)c, S)
     || !ai_qwen_4__table__zzpackage_matrix(&t, (unsigned)c + AI_QWEN_4__SITE__DOWN)
     || !ai_qwen_4__table__zzpackage_matrix(&t, (unsigned)c + AI_QWEN_4__SITE__UP)
     || (injects && !ai_qwen_4__table__zzpackage_matrix(&t, (unsigned)c + AI_QWEN_4__SITE__INJ)))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t scratch = s.at[AI_QWEN_4__SH__SCRATCH], signs = s.at[AI_QWEN_4__SH__SIGNS];
    uint64_t next = scratch;
    const uint64_t pairs = ai_qwen_4__layer__zzprivate_take(&next, 4u * n);
    const uint64_t an = ai_qwen_4__layer__zzprivate_take(&next, n * S), anr = ai_qwen_4__layer__zzprivate_take(&next, n * S);
    const uint64_t a1 = ai_qwen_4__layer__zzprivate_take(&next, n * R), a2 = ai_qwen_4__layer__zzprivate_take(&next, n * R);
    const uint64_t a3 = ai_qwen_4__layer__zzprivate_take(&next, n * R), a3r = ai_qwen_4__layer__zzprivate_take(&next, n * R);
    const uint64_t gt = ai_qwen_4__layer__zzprivate_take(&next, n * S), gs = ai_qwen_4__layer__zzprivate_take(&next, n * S);
    const uint64_t gn = ai_qwen_4__layer__zzprivate_take(&next, n * S), parts = ai_qwen_4__layer__zzprivate_take(&next, M * n * H);
    const uint64_t sum_a = ai_qwen_4__layer__zzprivate_take(&next, n * H), sum_b = ai_qwen_4__layer__zzprivate_take(&next, n * H);
    const uint64_t ij = ai_qwen_4__layer__zzprivate_take(&next, n * M), ij2 = ai_qwen_4__layer__zzprivate_take(&next, n * M);
    const uint64_t ij3 = ai_qwen_4__layer__zzprivate_take(&next, n * M);
    if (4u * n * S > sf_room || !nn__primitives__fits(n * H, into_room) || (injects && !nn__primitives__fits(n * M, inj_room))
     || next - scratch > s.room[AI_QWEN_4__SH__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const float eps = s.r[AI_QWEN_4__SH__EPS], mean = (float)(1.0 / (double)M);
    doors->expert_pairs_rows((uint32_t*)(uintptr_t)pairs, n, 1u, 0u, 1u, 0u);
    /* each row's every stream normed by its slice of the gain — the streams fp32, n·M rows of H in one launch, row k by the
     * gain's slice k mod M, each the bytes the one-row door gives it (nn's `rmsnorm_fp32_rows`) */
    doors->rmsnorm_fp32_rows((uint16_t*)(uintptr_t)an, (const float*)(uintptr_t)sf, (const uint16_t*)(uintptr_t)t.at[c], H, n * M, M,
                             eps, over);
    /* the mix's down map over every row, silu, the rank rotated a row at a time as its 256 and its 128, the up map */
    doors->hadamard_rotate((uint16_t*)(uintptr_t)anr, (const uint16_t*)(uintptr_t)an, (const uint16_t*)(uintptr_t)signs, n * S, over);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, (unsigned)c + AI_QWEN_4__SITE__DOWN, an, anr, a1, pairs, n, R, S);
    doors->vector_scale((uint16_t*)(uintptr_t)a2, (const uint16_t*)(uintptr_t)a1, n * R, mean, over);
    doors->vector_silu((uint16_t*)(uintptr_t)a3, (const uint16_t*)(uintptr_t)a2, n * R, over);
    doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)a3r, (const uint16_t*)(uintptr_t)a3, (const uint16_t*)(uintptr_t)signs, R, n, over);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, (unsigned)c + AI_QWEN_4__SITE__UP, a3, a3r, gt, pairs, n, S, R);
    doors->sigmoid((uint16_t*)(uintptr_t)gs, (const uint16_t*)(uintptr_t)gt, n * S, over);
    doors->vector_pointwise_mul((uint16_t*)(uintptr_t)gn, (const uint16_t*)(uintptr_t)gs, (const uint16_t*)(uintptr_t)an, n * S, over);
    /* the mean over the streams: each stream's rows gathered, summed in the streams' order, scaled */
    for (uint64_t j = 0u; j < M; ++j)
        doors->vector_copy_rows((uint16_t*)(uintptr_t)(parts + 2u * j * n * H), H, (const uint16_t*)(uintptr_t)(gn + 2u * j * H), S, H, n);
    uint64_t acc = parts;
    for (uint64_t j = 1u; j < M; ++j) {
        const uint64_t to = (j % 2u) ? sum_a : sum_b;
        doors->vector_add((uint16_t*)(uintptr_t)to, (const uint16_t*)(uintptr_t)acc, (const uint16_t*)(uintptr_t)(parts + 2u * j * n * H), n * H, over);
        acc = to;
    }
    doors->vector_scale((uint16_t*)(uintptr_t)into, (const uint16_t*)(uintptr_t)acc, n * H, mean, over);
    if (injects) {
        ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, (unsigned)c + AI_QWEN_4__SITE__INJ, an, anr, ij, pairs, n, M, S);
        doors->vector_scale((uint16_t*)(uintptr_t)ij2, (const uint16_t*)(uintptr_t)ij, n * M, mean, over);
        doors->sigmoid((uint16_t*)(uintptr_t)ij3, (const uint16_t*)(uintptr_t)ij2, n * M, over);
        doors->vector_scale((uint16_t*)(uintptr_t)inj, (const uint16_t*)(uintptr_t)ij3, n * M, 2.0f, over);
    }
    return nn__doors_answer(&argv[3]);
}

/* ── `(ai_qwen_4__inject_rows streams y inj n shape)` -> `streams` ─────────────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__inject_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t sf = 0, sf_room = 0, y = 0, y_room = 0, inj = 0, inj_room = 0;
    if (!nn__primitives__room(&argv[0], &sf, &sf_room) || !nn__primitives__room(&argv[1], &y, &y_room)
     || !nn__primitives__room(&argv[2], &inj, &inj_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__shape s;
    uint64_t n = 0u;
    if (!ai_qwen_4__table__zzpackage_shape(&argv[4], &s) || !ai_qwen_4__rows__zzprivate_n(&argv[3], &s, &n))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], M = s.v[AI_QWEN_4__SH__STREAMS], S = M * H, mf = 2u * M + M * M;
    const uint64_t mix = s.at[AI_QWEN_4__SH__MIX];
    if (4u * n * S > sf_room || !nn__primitives__fits(n * H, y_room) || !nn__primitives__fits(n * M, inj_room)
     || 4u * n * mf > s.room[AI_QWEN_4__SH__MIX])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    /* row r's mix: no pre, its injection weights as post, the identity as comb (the boot's) — stream j += inj_j · y */
    for (uint64_t r = 0u; r < n; ++r) {
        doors->vector_widen((float*)(uintptr_t)(mix + 4u * (r * mf + M)), (const uint16_t*)(uintptr_t)(inj + 2u * r * M), M);
        doors->hyper_post_fp32((float*)(uintptr_t)(sf + 4u * r * S), (const float*)(uintptr_t)(sf + 4u * r * S),
                               (const uint16_t*)(uintptr_t)(y + 2u * r * H), (const float*)(uintptr_t)(mix + 4u * r * mf), H, M);
    }
    return nn__doors_answer(&argv[0]);
}

/* ── `(ai_qwen_4__deltanet_rows m table s32 cwin y n shape)` -> `y` ───────────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__deltanet_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t m = 0, m_room = 0, s32 = 0, s32_room = 0, cwin = 0, cwin_room = 0, y = 0, y_room = 0;
    if (!nn__primitives__room(&argv[0], &m, &m_room) || !nn__primitives__room(&argv[2], &s32, &s32_room)
     || !nn__primitives__room(&argv[3], &cwin, &cwin_room) || !nn__primitives__room(&argv[4], &y, &y_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    uint64_t n = 0u;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 1u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    if (!ai_qwen_4__rows__zzprivate_n(&argv[5], &s, &n)) return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], KH = s.v[AI_QWEN_4__SH__KH], VH = s.v[AI_QWEN_4__SH__VH];
    const uint64_t LD = s.v[AI_QWEN_4__SH__LD], VAL = VH * LD, CH = 2u * KH * LD + VAL;
    if (VH % KH != 0u || LD > 4096u
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__QKV) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__Z)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__A) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__B)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__OUT)
     || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__CONV, CH * NN__DELTANET__CONV_TAPS)
     || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__ALOG, VH) || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__DTB, VH)
     || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__GNORM, LD))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t scratch = s.at[AI_QWEN_4__SH__SCRATCH], signs = s.at[AI_QWEN_4__SH__SIGNS];
    uint64_t next = scratch;
    const uint64_t pairs = ai_qwen_4__layer__zzprivate_take(&next, 4u * n);
    const uint64_t mr = ai_qwen_4__layer__zzprivate_take(&next, n * H), mixed = ai_qwen_4__layer__zzprivate_take(&next, n * CH);
    const uint64_t z = ai_qwen_4__layer__zzprivate_take(&next, n * VAL), av = ai_qwen_4__layer__zzprivate_take(&next, n * VH);
    const uint64_t bv = ai_qwen_4__layer__zzprivate_take(&next, n * VH), beta = ai_qwen_4__layer__zzprivate_take(&next, n * VH);
    const uint64_t g = ai_qwen_4__layer__zzprivate_take(&next, n * VH), conved = ai_qwen_4__layer__zzprivate_take(&next, n * CH);
    const uint64_t core = ai_qwen_4__layer__zzprivate_take(&next, n * VAL), crot = ai_qwen_4__layer__zzprivate_take(&next, n * VAL);
    if (!nn__primitives__fits(n * H, m_room) || !nn__primitives__fits(n * H, y_room) || 4u * VH * LD * LD > s32_room
     || !nn__primitives__fits(CH * NN__DELTANET__CONV_WINDOW, cwin_room) || next - scratch > s.room[AI_QWEN_4__SH__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    doors->expert_pairs_rows((uint32_t*)(uintptr_t)pairs, n, 1u, 0u, 1u, 0u);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)mr, (const uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)signs, n * H, over);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__QKV, m, mr, mixed, pairs, n, CH, H);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__Z, m, mr, z, pairs, n, VAL, H);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__A, m, mr, av, pairs, n, VH, H);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__B, m, mr, bv, pairs, n, VH, H);
    /* β = σ(b), g = −exp(A_log)·softplus(a + dt_bias), every row; the conv and every head's step walking the rows */
    doors->deltanet_gates_rows((uint16_t*)(uintptr_t)beta, (uint16_t*)(uintptr_t)g, (const uint16_t*)(uintptr_t)av,
                               (const uint16_t*)(uintptr_t)bv, (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__ALOG],
                               (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__DTB], VH, n, over);
    doors->deltanet_conv_steps((uint16_t*)(uintptr_t)conved, (uint16_t*)(uintptr_t)cwin, (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__CONV],
                               (const uint16_t*)(uintptr_t)mixed, CH, n, over);
    doors->deltanet_steps_gate((float*)(uintptr_t)s32, (const uint16_t*)(uintptr_t)conved, (const uint16_t*)(uintptr_t)z,
                               (const uint16_t*)(uintptr_t)beta, (const uint16_t*)(uintptr_t)g,
                               (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__GNORM], (uint16_t*)(uintptr_t)core, KH, VH, LD, n,
                               (unsigned int)NN__DELTANET__GATE_SIGMOID, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)crot, (const uint16_t*)(uintptr_t)core, (const uint16_t*)(uintptr_t)signs, n * VAL, over);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__OUT, core, crot, y, pairs, n, H, VAL);
    return nn__doors_answer(&argv[4]);
}

/* ── `(ai_qwen_4__attention_rows m table kc vc first n y shape)` -> `y` ───────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__attention_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 8u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t m = 0, m_room = 0, kc = 0, kc_room = 0, vc = 0, vc_room = 0, y = 0, y_room = 0;
    if (argv[4].dtype != SYS__KIND__VALUE_INT || !nn__primitives__room(&argv[0], &m, &m_room)
     || !nn__primitives__room(&argv[2], &kc, &kc_room) || !nn__primitives__room(&argv[3], &vc, &vc_room)
     || !nn__primitives__room(&argv[6], &y, &y_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    uint64_t n = 0u;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 1u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    if (!ai_qwen_4__rows__zzprivate_n(&argv[5], &s, &n)) return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], QH = s.v[AI_QWEN_4__SH__QH], KVH = s.v[AI_QWEN_4__SH__KVH];
    const uint64_t AD = s.v[AI_QWEN_4__SH__AD], ROT = s.v[AI_QWEN_4__SH__ROT], QW = QH * AD, KVW = KVH * AD;
    const uint64_t first = argv[4].args[0], to = first + n;
    if (QH % KVH != 0u || ROT > AD || ROT % 2u != 0u || first >= (1ull << 24)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__Q) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__K)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__V) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__O)
     || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__QN, AD) || !ai_qwen_4__table__zzpackage_vector(&t, AI_QWEN_4__LT__KN, AD))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t scratch = s.at[AI_QWEN_4__SH__SCRATCH], signs = s.at[AI_QWEN_4__SH__SIGNS], cs = s.at[AI_QWEN_4__SH__CS];
    const uint64_t scores = s.at[AI_QWEN_4__SH__SCORES];
    uint64_t next = scratch;
    const uint64_t pairs = ai_qwen_4__layer__zzprivate_take(&next, 4u * n), vpairs = ai_qwen_4__layer__zzprivate_take(&next, 4u * n);
    const uint64_t mr = ai_qwen_4__layer__zzprivate_take(&next, n * H), qg = ai_qwen_4__layer__zzprivate_take(&next, n * 2u * QW);
    const uint64_t kk = ai_qwen_4__layer__zzprivate_take(&next, n * KVW), qn = ai_qwen_4__layer__zzprivate_take(&next, n * QW);
    const uint64_t att = ai_qwen_4__layer__zzprivate_take(&next, n * QW), gated = ai_qwen_4__layer__zzprivate_take(&next, n * QW);
    const uint64_t arot = ai_qwen_4__layer__zzprivate_take(&next, n * QW);
    if (!nn__primitives__fits(n * H, m_room) || !nn__primitives__fits(n * H, y_room) || !nn__primitives__fits(to * KVW, kc_room)
     || !nn__primitives__fits(to * KVW, vc_room) || s.room[AI_QWEN_4__SH__SCORES] / 4u < n * QH * to
     || s.room[AI_QWEN_4__SH__CS] / 4u < n * ROT || next - scratch > s.room[AI_QWEN_4__SH__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    const float eps = s.r[AI_QWEN_4__SH__EPS];
    doors->expert_pairs_rows((uint32_t*)(uintptr_t)pairs, n, 1u, 0u, 1u, 0u);
    doors->expert_pairs_rows((uint32_t*)(uintptr_t)vpairs, n, 1u, 0u, 1u, first);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)mr, (const uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)signs, n * H, over);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__Q, m, mr, qg, pairs, n, 2u * QW, H);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__K, m, mr, kk, pairs, n, KVW, H);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__V, m, mr, vc, vpairs, n, KVW, H);   /* row r at `first + r` */
    /* every head's query normed and roped on its first ROT; every key so, straight into its cache row */
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qg, (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__QN],
                        AD, n * QH, 2u * AD, AD, eps, over);
    doors->rope_rows((uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)qn, (const float*)(uintptr_t)cs, n, QH, AD, QW, ROT, over);
    const uint64_t kfirst = kc + 2u * first * KVW;
    doors->rmsnorm_rows((uint16_t*)(uintptr_t)kfirst, (const uint16_t*)(uintptr_t)kk, (const uint16_t*)(uintptr_t)t.at[AI_QWEN_4__LT__KN],
                        AD, n * KVH, AD, AD, eps, over);
    doors->rope_rows((uint16_t*)(uintptr_t)kfirst, (const uint16_t*)(uintptr_t)kfirst, (const float*)(uintptr_t)cs, n, KVH, AD, KVW, ROT, over);
    /* row r over positions 0 … first + r, gated by σ of its heads' gates, the output projection */
    doors->attention_causal_scores_grouped((float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)qn, (const uint16_t*)(uintptr_t)kc,
                                           QH, KVH, AD, first, n);
    doors->attention_causal_softmax((float*)(uintptr_t)scores, QH, first, n);
    doors->attention_causal_mix((uint16_t*)(uintptr_t)att, (const float*)(uintptr_t)scores, (const uint16_t*)(uintptr_t)vc,
                                QH, KVH, AD, first, n, over);
    doors->attention_gate_rows((uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)att, (const uint16_t*)(uintptr_t)qg, n * QH, AD, over);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)arot, (const uint16_t*)(uintptr_t)gated, (const uint16_t*)(uintptr_t)signs, n * QW, over);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__O, gated, arot, y, pairs, n, H, QW);
    return nn__doors_answer(&argv[6]);
}

/* ── `(ai_qwen_4__shared_rows x xr table out n shape)` -> `out` ───────────────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__shared_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 6u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t x = 0, x_room = 0, xr = 0, xr_room = 0, out = 0, out_room = 0;
    if (!nn__primitives__room(&argv[0], &x, &x_room) || !nn__primitives__room(&argv[1], &xr, &xr_room)
     || !nn__primitives__room(&argv[3], &out, &out_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    uint64_t n = 0u;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 2u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    if (!ai_qwen_4__rows__zzprivate_n(&argv[4], &s, &n)) return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], I = s.v[AI_QWEN_4__SH__INTER];
    if (!ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__SG) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__SU)
     || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__SD) || !ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__SEG))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t scratch = s.at[AI_QWEN_4__SH__SCRATCH], signs = s.at[AI_QWEN_4__SH__SIGNS];
    uint64_t next = scratch;
    const uint64_t pairs = ai_qwen_4__layer__zzprivate_take(&next, 4u * n);
    const uint64_t sgo = ai_qwen_4__layer__zzprivate_take(&next, n * I), suo = ai_qwen_4__layer__zzprivate_take(&next, n * I);
    const uint64_t sact = ai_qwen_4__layer__zzprivate_take(&next, n * I), sactr = ai_qwen_4__layer__zzprivate_take(&next, n * I);
    const uint64_t sdo = ai_qwen_4__layer__zzprivate_take(&next, n * H), segv = ai_qwen_4__layer__zzprivate_take(&next, n);
    const uint64_t segs = ai_qwen_4__layer__zzprivate_take(&next, n);
    if (!nn__primitives__fits(n * H, x_room) || !nn__primitives__fits(n * H, xr_room) || !nn__primitives__fits(n * H, out_room)
     || next - scratch > s.room[AI_QWEN_4__SH__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    doors->expert_pairs_rows((uint32_t*)(uintptr_t)pairs, n, 1u, 0u, 1u, 0u);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__SG, x, xr, sgo, pairs, n, I, H);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__SU, x, xr, suo, pairs, n, I, H);
    doors->swiglu_combine((uint16_t*)(uintptr_t)sact, (const uint16_t*)(uintptr_t)sgo, (const uint16_t*)(uintptr_t)suo, n * I, over);
    doors->hadamard_ladder_rows((uint16_t*)(uintptr_t)sactr, (const uint16_t*)(uintptr_t)sact, (const uint16_t*)(uintptr_t)signs, I, n, over);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__SD, sact, sactr, sdo, pairs, n, H, I);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__SEG, x, xr, segv, pairs, n, 1u, H);
    doors->sigmoid((uint16_t*)(uintptr_t)segs, (const uint16_t*)(uintptr_t)segv, n, over);
    for (uint64_t r = 0u; r < n; ++r)
        doors->vector_scale_at((uint16_t*)(uintptr_t)(out + 2u * r * H), (const uint16_t*)(uintptr_t)(sdo + 2u * r * H),
                               (const uint16_t*)(uintptr_t)segs, r, H, over);
    return nn__doors_answer(&argv[3]);
}

/* ── `(ai_qwen_4__route_rows m mr table logits hands n shape)` -> `logits` ────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__route_rows__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t m = 0, m_room = 0, mr = 0, mr_room = 0, lg = 0, lg_room = 0, hands = 0, h_room = 0;
    if (!nn__primitives__room(&argv[0], &m, &m_room) || !nn__primitives__room(&argv[1], &mr, &mr_room)
     || !nn__primitives__room(&argv[3], &lg, &lg_room) || !nn__primitives__room(&argv[4], &hands, &h_room))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    uint64_t n = 0u;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 2u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    if (!ai_qwen_4__rows__zzprivate_n(&argv[5], &s, &n)) return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], E = s.v[AI_QWEN_4__SH__EXPERTS], HAND = s.v[AI_QWEN_4__SH__HAND];
    if (!ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__GATE) || HAND % 2u != 0u || HAND < 2u * H)
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    const uint64_t scratch = s.at[AI_QWEN_4__SH__SCRATCH], signs = s.at[AI_QWEN_4__SH__SIGNS];
    uint64_t next = scratch;
    const uint64_t pairs = ai_qwen_4__layer__zzprivate_take(&next, 4u * n);
    if (!nn__primitives__fits(n * H, m_room) || !nn__primitives__fits(n * H, mr_room) || !nn__primitives__fits(n * E, lg_room)
     || n * HAND > h_room || next - scratch > s.room[AI_QWEN_4__SH__SCRATCH])
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    doors->expert_pairs_rows((uint32_t*)(uintptr_t)pairs, n, 1u, 0u, 1u, 0u);
    doors->hadamard_rotate((uint16_t*)(uintptr_t)mr, (const uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)signs, n * H, over);
    ai_qwen_4__rows__zzprivate_gemm(doors, over, &t, AI_QWEN_4__LT__GATE, m, mr, lg, pairs, n, E, H);
    doors->vector_copy_rows((uint16_t*)(uintptr_t)hands, HAND / 2u, (const uint16_t*)(uintptr_t)mr, H, H, n);
    return nn__doors_answer(&argv[3]);
}

/* ── `(ai_qwen_4__route m mr table logits picks hand k shape)` -> `picks` ────────────────────────────────────────── */
static sys__heap_node ai_qwen_4__route__zzabi_apply(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 8u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t m = 0, m_room = 0, mr = 0, mr_room = 0, lg = 0, lg_room = 0, hand = 0, h_room = 0;
    if (!nn__primitives__room(&argv[0], &m, &m_room) || !nn__primitives__room(&argv[1], &mr, &mr_room)
     || !nn__primitives__room(&argv[3], &lg, &lg_room) || !nn__primitives__room(&argv[5], &hand, &h_room)
     || !sys__heap_node__carries_reference(argv[4].dtype) || !sys__node_array__is(argv[4].args[0])
     || argv[6].dtype != SYS__KIND__VALUE_INT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    ai_qwen_4__table t;
    ai_qwen_4__shape s;
    const uint64_t why = ai_qwen_4__layer__zzprivate_open(argv, argc, 2u, &t, &s);
    if (why != 0ull) return sys__engine__abi__error(why);
    const uint64_t H = s.v[AI_QWEN_4__SH__HIDDEN], E = s.v[AI_QWEN_4__SH__EXPERTS], k = argv[6].args[0], picks = argv[4].args[0];
    if (!ai_qwen_4__table__zzpackage_matrix(&t, AI_QWEN_4__LT__GATE)) return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_TABLE);
    /* `nn__vector__top_k`'s bounds, as its verb refuses them */
    if (E == 0ull || E > (1ull << 24) || k == 0ull || k > NN__VECTOR__TOP_K_MAX || k > E || sys__node_array__length(picks) < k)
        return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    if (!nn__primitives__fits(H, m_room) || !nn__primitives__fits(H, mr_room) || !nn__primitives__fits(E, lg_room)
     || !nn__primitives__fits(H + k, h_room))
        return sys__engine__abi__error(AI_QWEN_4__LAYER__FAULT_ROOM);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    unsigned int* over = ctx->fault_word;
    doors->hadamard_rotate((uint16_t*)(uintptr_t)mr, (const uint16_t*)(uintptr_t)m, (const uint16_t*)(uintptr_t)s.at[AI_QWEN_4__SH__SIGNS],
                           H, over);
    ai_qwen_4__layer__zzprivate_mv(doors, over, &t, AI_QWEN_4__LT__GATE, m, mr, lg, E, H);
    /* the hand's `mr` queued before the top-k, which waits for its kernel; its weights written into the hand itself */
    doors->vector_copy((uint16_t*)(uintptr_t)hand, (const uint16_t*)(uintptr_t)mr, H);
    const uint64_t fault = nn__routed__top_k_scored(ctx, doors, picks, hand + 2u * H, lg, 0ull, 1.0f, E, k, NN__ROUTED__BIASED, 0);
    if (fault != 0u) return sys__engine__abi__error(fault);
    return nn__doors_answer(&argv[4]);
}

SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__gated_pre__zzabi_adapter, ai_qwen_4__gated_pre__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__attention__zzabi_adapter, ai_qwen_4__attention__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__deltanet__zzabi_adapter,  ai_qwen_4__deltanet__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__shared__zzabi_adapter,    ai_qwen_4__shared__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__gated_pre_rows__zzabi_adapter, ai_qwen_4__gated_pre_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__inject_rows__zzabi_adapter,    ai_qwen_4__inject_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__deltanet_rows__zzabi_adapter,  ai_qwen_4__deltanet_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__attention_rows__zzabi_adapter, ai_qwen_4__attention_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__shared_rows__zzabi_adapter,    ai_qwen_4__shared_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__route_rows__zzabi_adapter,     ai_qwen_4__route_rows__zzabi_apply)
SYS__ENGINE__ABI__BRIDGE(ai_qwen_4__route__zzabi_adapter,          ai_qwen_4__route__zzabi_apply)

#endif /* SILVANN__PACKAGES_AI_QWEN_4_CPU_OPCODES_LAYER__ABI_CUH */
