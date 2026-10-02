#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_EXPERT__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_EXPERT__ABI_CUH
/* ══ THE EXPERT MATVEC AT FULL PRECISION, ON THE PROGRAM'S SIDE ══════════════════════════════════════
 * The checks are its body's in `expert__impl.cuh`, refusing a bad shape with the expert's own word; the
 * arithmetic is the door's. ▶ `vector__abi.cuh` for the shape and the refusals. */

#include "../doors.cuh"

/* ── ⭐⭐⭐ THE fp16 MATVEC — ▶ `expert__header.cuh` for why it is its own verb ──────────────────────── */
/* `(nn__expert__multiply_fp16 W x out rows cols)` — `out = W x`, W rows × cols as stored, every operand
 * fp16 and the accumulator fp32. */
static sys__heap_node nn__expert__zzabi_apply_multiply_fp16(const sys__heap_node* argv, unsigned argc,
                                                     sys__engine__ctx* ctx) {
    if (argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t w_at = 0, x_at = 0, d_at = 0, w_room = 0, x_room = 0, d_room = 0;
    if (argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[0], &w_at, &w_room)
     || !nn__primitives__room(&argv[1], &x_at, &x_room)
     || !nn__primitives__room(&argv[2], &d_at, &d_room)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t rows = argv[3].args[0], cols = argv[4].args[0];
    if (rows == 0ull || cols == 0ull) return sys__engine__abi__error(NN__EXPERT__FAULT_SHAPE);
    if (rows > UINT64_MAX / cols) return sys__engine__abi__error(NN__EXPERT__FAULT_SHAPE);
    if (!nn__primitives__fits(rows * cols, w_room)
     || !nn__primitives__fits(cols, x_room)
     || !nn__primitives__fits(rows, d_room)) return sys__engine__abi__error(NN__EXPERT__FAULT_SHAPE);

    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    doors->expert_multiply_fp16((uint16_t*)(uintptr_t)d_at, (const uint16_t*)(uintptr_t)w_at, (const uint16_t*)(uintptr_t)x_at,
                                rows, cols, ctx->fault_word);
    return nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(nn__expert__zzabi_adapter_multiply_fp16, nn__expert__zzabi_apply_multiply_fp16)

/* `(nn__expert__misses which)` -> the loader's counts: 0 reads picks asked for, 1 reads predictions queued, 2 of
 * those, how many a pick then wanted, 3 the ns layers waited on reads, 4 how many times they waited. */
static sys__heap_node nn__expert__zzabi_apply_misses(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[0].dtype != SYS__KIND__VALUE_INT || argv[0].args[0] >= NN__EXPERT__COUNTS)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    sys__heap_node answer = sys__heap_node__nothing();
    answer.dtype = SYS__KIND__VALUE_INT; answer.args[0] = nn__expert__count((unsigned)argv[0].args[0]);
    return answer;
}
SYS__ENGINE__ABI__BRIDGE(nn__expert__zzabi_adapter_misses, nn__expert__zzabi_apply_misses)

/* ── ⭐⭐ THE LRU'S WORDS ─────────────────────────────────────────────────────────────────────────────────────
 * `(nn__expert__request layer type picks backing)` -> how many are on their way. Each expert of `picks` (a node
 * array of expert numbers, at most 16) is asked for: a resident one is touched, a missing one gets a slot —
 * never another of `picks`' — and its read is queued. Settle the layer before using the ones on their way.
 * `(nn__expert__prefetch layer type picks backing)` -> the same, as a PREDICTION: counted apart, and a pick
 *   asking for it later is what makes it a prediction that came true.
 * `(nn__expert__settle layer type)` -> `#t` when every read of the band has arrived and been admitted. */
#define NN__EXPERT__ASK_MAX 16u
static sys__heap_node nn__expert__zzprivate_ask(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx, bool predicted) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[0].dtype != SYS__KIND__VALUE_INT || argv[1].dtype != SYS__KIND__VALUE_INT
     || !sys__heap_node__carries_reference(argv[2].dtype) || !sys__node_array__is(argv[2].args[0])
     || !sys__heap_node__carries_reference(argv[3].dtype) || !sys__node_array__is(argv[3].args[0]))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    nn__expert__backing backing = {};
    if (!nn__expert__backing_of(argv[3].args[0], &backing)) return sys__engine__abi__error(NN__EXPERT__FAULT_BACKING);
    const uint64_t layer = argv[0].args[0], type = argv[1].args[0], picks = argv[2].args[0];
    const uint64_t n = sys__node_array__length(picks);
    if (n > NN__EXPERT__ASK_MAX) return sys__engine__abi__error(NN__EXPERT__FAULT_SPAN);
    uint64_t ids[NN__EXPERT__ASK_MAX];
    for (uint64_t j = 0u; j < n; ++j) {
        const sys__heap_node p = sys__node_array__borrow(picks, j);
        if (p.dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
        ids[j] = p.args[0];
    }
    uint64_t arriving = 0u;
    for (uint64_t j = 0u; j < n; ++j) {
        const int got = nn__expert__request(ctx->family, layer, type, ids[j], &backing, predicted,
                                            predicted ? 0 : ids, predicted ? 0u : (unsigned)n, 0);
        if (got == NN__EXPERT__REFUSED && !predicted) return sys__engine__abi__error(NN__EXPERT__FAULT_ABSENT);
        if (got == NN__EXPERT__ARRIVING) ++arriving;
    }
    sys__heap_node answer = sys__heap_node__nothing();
    answer.dtype = SYS__KIND__VALUE_INT; answer.args[0] = arriving;
    return answer;
}
static sys__heap_node nn__expert__zzabi_apply_request(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    return nn__expert__zzprivate_ask(argv, argc, ctx, false);
}
static sys__heap_node nn__expert__zzabi_apply_prefetch(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    return nn__expert__zzprivate_ask(argv, argc, ctx, true);
}
static sys__heap_node nn__expert__zzabi_apply_settle(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 2u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[0].dtype != SYS__KIND__VALUE_INT || argv[1].dtype != SYS__KIND__VALUE_INT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    return sys__opcodes__truth(nn__expert__settle(argv[0].args[0], argv[1].args[0]));
}
SYS__ENGINE__ABI__BRIDGE(nn__expert__zzabi_adapter_request,  nn__expert__zzabi_apply_request)
SYS__ENGINE__ABI__BRIDGE(nn__expert__zzabi_adapter_prefetch, nn__expert__zzabi_apply_prefetch)
SYS__ENGINE__ABI__BRIDGE(nn__expert__zzabi_adapter_settle,   nn__expert__zzabi_apply_settle)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_EXPERT__ABI_CUH */
