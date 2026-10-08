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
 * those, how many a pick then wanted, 3 the ns layers waited on reads, 4 how many times they waited — and the tier's:
 * 5 promotions, 6 demotions, 7 failed writes into RAM, 8 the ns promotions waited for a channel, 9 the ns CPUs waited
 * for a write into RAM, 10 promotions refused with every channel held. */
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
    sys__node_array_walk pw;
    if (!sys__node_array__walk(picks, 0ull, &pw)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    for (uint64_t j = 0u; j < n; ++j, sys__node_array__next(&pw)) {
        const sys__heap_node* p = sys__node_array__walk_cell(&pw);
        if (p == 0 || p->dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
        ids[j] = p->args[0];
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

/* ── ⭐ A CARD'S TIER GIVEN ITS SHARE OF A PROMPT CHUNK ─────────────────────────────────────────────────────────────────
 * `(nn__expert_tier__note hands nn__expert_tier layer n row picks line)` -> `hands`. `n` rows of a model's hands, each `row`
 * bytes, a row's picks `top_k` integers at byte `picks` of it — before the CPUs are handed theirs, by the card whose
 * tier the binding holds for this worker. ⚖ *"draw a line on the experts that makes no sense to compute on the gpu and
 * we compute them on the cpu, then we start computing from the ones that are lower hits and move up"*: every call
 * recorded, every expert the card holds taken, and every other one the chunk picked `line` times or more (the
 * program's, measured at boot) copied there from the fewest picks to the most (▶ `nn__expert_tier__chunk`); each pick
 * the card takes marked held in its row, its number plus `experts`, which the CPUs skip. */
static sys__heap_node nn__expert_tier__zzabi_apply_note(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 7u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t hands = 0, h_room = 0;
    if (!nn__primitives__room(&argv[0], &hands, &h_room) || !sys__heap_node__carries_reference(argv[1].dtype)
     || argv[2].dtype != SYS__KIND__VALUE_INT || argv[3].dtype != SYS__KIND__VALUE_INT || argv[4].dtype != SYS__KIND__VALUE_INT
     || argv[5].dtype != SYS__KIND__VALUE_INT || argv[6].dtype != SYS__KIND__VALUE_INT)
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    nn__expert_tier tier;
    if (!nn__expert_tier__of(argv[1].args[0], argv[2].args[0], &tier)) return sys__engine__abi__error(NN__EXPERT__FAULT_TIER);
    const uint64_t n = argv[3].args[0], row = argv[4].args[0], picks = argv[5].args[0], K = tier.top_k;
    if (n == 0u || row == 0u || picks > row || 8u * K > row - picks || n > h_room / row)
        return sys__engine__abi__error(NN__EXPERT__FAULT_SPAN);
    uint64_t* chosen = (uint64_t*)malloc(sizeof(uint64_t) * n * K);
    if (chosen == 0) return sys__engine__abi__error(NN__EXPERT__FAULT_ROOM);
    uint64_t fault = 0u;
    for (uint64_t r = 0u; r < n && fault == 0u; ++r)
        if (!sys__gpu__memory_read(ctx->family, chosen + r * K, (const void*)(uintptr_t)(hands + r * row + picks), 8u * K))
            fault = NN__PRIMITIVES__FAULT_NO_DEVICE;
    if (fault == 0u && !nn__expert_tier__chunk(ctx->family, &tier, chosen, n, argv[6].args[0])) fault = NN__EXPERT__FAULT_ABSENT;
    for (uint64_t r = 0u; r < n && fault == 0u; ++r)
        if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)(hands + r * row + picks), chosen + r * K, 8u * K))
            fault = NN__PRIMITIVES__FAULT_NO_DEVICE;
    free(chosen);
    return fault != 0u ? sys__engine__abi__error(fault) : nn__doors_answer(&argv[0]);
}
SYS__ENGINE__ABI__BRIDGE(nn__expert_tier__zzabi_adapter_note, nn__expert_tier__zzabi_apply_note)

/* ── ⭐ THE CPUS DONE: AN EXCLUSIVE TIER'S HELD WRITES GO ────────────────────────────────────────────────────────────────
 * `(nn__expert_tier__written nn__expert_tier)` -> the binding. Said by the card's program once the CPUs have handed back
 * a layer's experts: on half duplex the writes of experts sent back to RAM wait for this, so they run while the card
 * computes the next attention and not against the CPUs' experts for the memory (▶ `nn__expert_tier__commit`). With
 * nothing held it does nothing. */
static sys__heap_node nn__expert_tier__zzabi_apply_written(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    (void)ctx;
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (!sys__heap_node__carries_reference(argv[0].dtype)) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    nn__expert_tier__written();
    return nn__doors_answer(&argv[0]);
}
SYS__ENGINE__ABI__BRIDGE(nn__expert_tier__zzabi_adapter_written, nn__expert_tier__zzabi_apply_written)

/* ── ⭐ WHETHER THE LINK IS FULL DUPLEX ─────────────────────────────────────────────────────────────────────────────────
 * `(nn__expert_tier__duplex bytes)` -> an integer: how long a copy of `bytes` to the card and one back take run at once,
 * over the copy to the card alone, × 1000 — the best of five each, each copy on its own thread and side channel, from
 * and into pinned memory. ⚖ *"can we make it so that we can see if the system supports full duplex? if so we can launch
 * the read and write together, if not we need to send to vram first and receive to ram later. i think it should be a
 * lisp binding"* — the runtime binds `nn__expert_tier_duplex` from it (1 below ~1.2, the NN-48 rule). `MEASURED` on the
 * R730's MI50 by `test/expert_swap_timing.cpp`: 1.65 — half duplex. */
typedef struct nn__expert_tier__duplex_job { sys__silicon_family__id family; void* to; const void* from; size_t bytes; bool up; bool ok; } nn__expert_tier__duplex_job;
static void* nn__expert_tier__zzprivate_duplex_copy(void* a) {
    nn__expert_tier__duplex_job* j = (nn__expert_tier__duplex_job*)a;
    void* side = 0;
    j->ok = sys__gpu__side_open(j->family, &side)
         && (j->up ? sys__gpu__side_memory_write(j->family, j->to, j->from, j->bytes, side)
                   : sys__gpu__side_memory_read(j->family, j->to, j->from, j->bytes, side));
    if (side != 0) sys__gpu__side_close(j->family, side);
    return 0;
}
static double nn__expert_tier__zzprivate_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static sys__heap_node nn__expert_tier__zzabi_apply_duplex(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 1u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    if (argv[0].dtype != SYS__KIND__VALUE_INT || argv[0].args[0] == 0u || argv[0].args[0] > (1ull << 30))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const size_t bytes = (size_t)argv[0].args[0];
    void *h_up = 0, *h_down = 0, *c_up = 0, *c_down = 0, *d_up = 0, *d_down = 0;
    bool ok = sys__gpu__memory_allocate_host_ram_mapped(ctx->family, &h_up, &c_up, bytes)
           && sys__gpu__memory_allocate_host_ram_mapped(ctx->family, &h_down, &c_down, bytes)
           && sys__gpu__memory_allocate(ctx->family, &d_up, bytes) && sys__gpu__memory_allocate(ctx->family, &d_down, bytes);
    double alone = 1e30, both = 1e30;
    for (int rep = 0; ok && rep < 6; ++rep) {
        nn__expert_tier__duplex_job up = { ctx->family, d_up, h_up, bytes, true, false };
        nn__expert_tier__duplex_job down = { ctx->family, h_down, d_down, bytes, false, false };
        double t0 = nn__expert_tier__zzprivate_seconds();
        nn__expert_tier__zzprivate_duplex_copy(&up);
        const double a = nn__expert_tier__zzprivate_seconds() - t0;
        pthread_t th[2];
        t0 = nn__expert_tier__zzprivate_seconds();
        ok = pthread_create(&th[0], 0, nn__expert_tier__zzprivate_duplex_copy, &up) == 0;
        ok = ok && pthread_create(&th[1], 0, nn__expert_tier__zzprivate_duplex_copy, &down) == 0;
        if (ok) { pthread_join(th[0], 0); pthread_join(th[1], 0); }
        const double b = nn__expert_tier__zzprivate_seconds() - t0;
        ok = ok && up.ok && down.ok;
        if (rep > 0) { if (a < alone) alone = a; if (b < both) both = b; }    /* the first is a warm-up */
    }
    if (d_up != 0) sys__gpu__memory_free(ctx->family, d_up);
    if (d_down != 0) sys__gpu__memory_free(ctx->family, d_down);
    if (h_up != 0) sys__gpu__memory_free_host_ram(ctx->family, h_up);
    if (h_down != 0) sys__gpu__memory_free_host_ram(ctx->family, h_down);
    if (!ok || alone <= 0.0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    sys__heap_node answer = sys__heap_node__nothing();
    answer.dtype = SYS__KIND__VALUE_INT;
    answer.args[0] = (uint64_t)(1000.0 * both / alone + 0.5);
    return answer;
}
SYS__ENGINE__ABI__BRIDGE(nn__expert_tier__zzabi_adapter_duplex, nn__expert_tier__zzabi_apply_duplex)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_EXPERT__ABI_CUH */
