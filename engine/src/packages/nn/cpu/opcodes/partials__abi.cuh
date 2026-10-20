#ifndef SILVANN__PACKAGES_NN_CPU_OPCODES_PARTIALS__ABI_CUH
#define SILVANN__PACKAGES_NN_CPU_OPCODES_PARTIALS__ABI_CUH
/* ══ A TENSOR-PARALLEL GROUP'S PARTIAL SUMS ════════════════════════════════════════════════════════════════════════════
 * Each rank of a group runs on its own worker and its own card. The sums are `vector_sum` launches that read the other
 * cards' memory in place, and the order between cards is sys's marks: the reading card's queue waits for the writing
 * card's mark, so no card waits for the host and the host waits for no card. ▶ `contracts/objects/partials.cuh` for the
 * words, the tree and what a rank may not write between two calls. */
#include "../../../../operating_systems/os.cuh"   /* the clock and a yield: the system's */
#include "../doors.cuh"
#include "../../contracts/objects/partials.cuh"

/* ⛳ A GROUP IS THE HOST'S: its ranks' marks, the addresses each rank published for its latest call, and that call's
 * number — what a rank's thread waits for before its card's queue waits for the mark. A rank's `calls` is touched
 * only by its own thread; the generations are written by their rank's thread and read by the others'. */
typedef struct nn__partials__group {
    uint32_t ranks;
    uint32_t fan;
    uint32_t joined[NN__PARTIALS__RANKS_MAX];
    uint32_t worker[NN__PARTIALS__RANKS_MAX];
    sys__silicon_family__id family[NN__PARTIALS__RANKS_MAX];
    uint32_t device[NN__PARTIALS__RANKS_MAX];
    void*    up_mark[NN__PARTIALS__RANKS_MAX];     /* reached when the rank's partial, with what it added, is written */
    void*    down_mark[NN__PARTIALS__RANKS_MAX];   /* reached when the rank's `out` holds the whole sum */
    uint64_t calls[NN__PARTIALS__RANKS_MAX];
    uint64_t up_at[NN__PARTIALS__RANKS_MAX];
    uint64_t down_at[NN__PARTIALS__RANKS_MAX];
    uint64_t up_gen[NN__PARTIALS__RANKS_MAX];
    uint64_t down_gen[NN__PARTIALS__RANKS_MAX];
} nn__partials__group;

static nn__partials__group nn__partials__zzprivate_groups[NN__PARTIALS__GROUPS_MAX];

/* Closes a rank's marks and empties its place. */
static inline void nn__partials__zzprivate_leave(nn__partials__group* g, uint32_t r) {
    if (!g->joined[r]) return;
    sys__gpu__mark_close(g->family[r], g->up_mark[r]);
    sys__gpu__mark_close(g->family[r], g->down_mark[r]);
    g->up_mark[r] = g->down_mark[r] = 0;
    g->joined[r] = 0u;
}

/* Waits until another rank's thread has published call `want` — it has QUEUED that call's work, not finished it.
 * False after `NN__PARTIALS__PATIENCE` seconds: a rank that never arrives is an error, not a hang. */
static inline bool nn__partials__zzprivate_await(const uint64_t* gen, uint64_t want) {
    if (__atomic_load_n(gen, __ATOMIC_ACQUIRE) >= want) return true;
    struct timespec t0, t;
    sys__os__clock(&t0);
    for (uint32_t spin = 1u;; ++spin) {
        if (__atomic_load_n(gen, __ATOMIC_ACQUIRE) >= want) return true;
        if (spin < 256u) continue;
        sys__os__yield();
        if ((spin & 1023u) != 0u) continue;
        sys__os__clock(&t);
        if ((uint64_t)(t.tv_sec - t0.tv_sec) > (uint64_t)NN__PARTIALS__PATIENCE) return false;
    }
}

/* ── `(nn__partials__join group rank ranks fan)` -> #t ─────────────────────────────────────────────────────────────── */
static sys__heap_node nn__partials__zzabi_apply_join(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    for (unsigned i = 0u; i < 4u; ++i)
        if (argv[i].dtype != SYS__KIND__VALUE_INT) return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t group = argv[0].args[0], rank = argv[1].args[0], ranks = argv[2].args[0], fan = argv[3].args[0];
    if (group >= NN__PARTIALS__GROUPS_MAX || ranks == 0u || ranks > NN__PARTIALS__RANKS_MAX || rank >= ranks
     || fan < 2u || fan > NN__PARTIALS__FAN_MAX) return sys__engine__abi__error(NN__PARTIALS__FAULT_GROUP);
    if (ctx->family == SYS__SILICON_FAMILY__NONE) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    nn__partials__group* g = &nn__partials__zzprivate_groups[group];
    const uint32_t r = (uint32_t)rank;
    /* rank 0 starts the group afresh, and so does a rank that names another shape */
    if (r == 0u || g->ranks != (uint32_t)ranks || g->fan != (uint32_t)fan) {
        for (uint32_t j = 0u; j < NN__PARTIALS__RANKS_MAX; ++j) nn__partials__zzprivate_leave(g, j);
        g->ranks = (uint32_t)ranks;
        g->fan = (uint32_t)fan;
    }
    nn__partials__zzprivate_leave(g, r);
    /* ⚖ "tensor parallelism requires DMA": each pair, both directions, or the rank is refused */
    const sys__silicon_family__id fam = ctx->family;
    const uint32_t here = (uint32_t)ctx->device;
    bool dma = true;
    for (uint32_t j = 0u; j < g->ranks && dma; ++j) {
        if (j == r || !g->joined[j]) continue;
        if (g->family[j] != fam) { dma = false; break; }
        dma = sys__gpu__peer_access(fam, g->device[j]);
        if (dma && g->device[j] != here) {
            dma = sys__silicon__bind_device(fam, g->device[j]) && sys__gpu__peer_access(fam, here);
            dma = sys__silicon__bind_device(fam, here) && dma;
        }
    }
    void* up = 0;
    void* down = 0;
    if (dma && !sys__gpu__mark_open(fam, &up)) dma = false;
    if (dma && !sys__gpu__mark_open(fam, &down)) { sys__gpu__mark_close(fam, up); dma = false; }
    if (!dma) return sys__engine__abi__error(NN__PARTIALS__FAULT_NO_DMA);
    g->worker[r] = sys__silicon__worker();
    g->family[r] = fam;
    g->device[r] = here;
    g->up_mark[r] = up;
    g->down_mark[r] = down;
    g->calls[r] = 0u;
    g->up_at[r] = g->down_at[r] = 0u;
    __atomic_store_n(&g->up_gen[r], 0ull, __ATOMIC_RELEASE);
    __atomic_store_n(&g->down_gen[r], 0ull, __ATOMIC_RELEASE);
    g->joined[r] = 1u;
    return sys__opcodes__truth(true);
}

/* ⭐ THE REDUCE, FOR A VERB THAT ENDS IN ONE — a grouped verb under tensor parallelism computes its partial and calls this
 * where it would have added its residual (`nn__mlp__apply`, ai_qwen_3's mixers). `residual` 0 for none. Answers 0, or the
 * fault to refuse with. */
static inline uint64_t nn__partials__reduce_onto(sys__engine__ctx* ctx, const nn__doors* doors, uint64_t group, uint64_t part,
                                                 uint64_t out, uint64_t n, uint64_t residual) {
    if (group >= NN__PARTIALS__GROUPS_MAX) return NN__PARTIALS__FAULT_GROUP;
    nn__partials__group* g = &nn__partials__zzprivate_groups[group];
    const uint32_t me = sys__silicon__worker(), P = g->ranks, F = g->fan;
    uint32_t r = NN__PARTIALS__RANKS_MAX;
    for (uint32_t j = 0u; j < P; ++j) {
        if (!g->joined[j]) return NN__PARTIALS__FAULT_GROUP;
        if (g->worker[j] == me) r = j;
    }
    if (P == 0u || r == NN__PARTIALS__RANKS_MAX) return NN__PARTIALS__FAULT_GROUP;
    const sys__silicon_family__id fam = ctx->family;
    const uint64_t call = ++g->calls[r];
    nn__vector__parts parts;
    if (P == 1u) {
        parts.count = 0u;
        if (residual != 0u) parts.at[parts.count++] = residual;
        parts.at[parts.count++] = part;
        doors->vector_sum((uint16_t*)(uintptr_t)out, parts, n, ctx->fault_word);
        return 0u;
    }
    /* up: at each level a rank either sends its partial on, or adds its children's onto it */
    uint32_t parent = r;
    for (uint32_t s = 1u; s < P; s *= F) {
        if (r % (s * F) != 0u) { parent = r - r % (s * F); break; }
        const bool last = r == 0u && s * F >= P;
        parts.count = 0u;
        if (last && residual != 0u) parts.at[parts.count++] = residual;
        parts.at[parts.count++] = part;
        for (uint32_t k = 1u; k < F && r + s * k < P; ++k) {
            const uint32_t c = r + s * k;
            if (!nn__partials__zzprivate_await(&g->up_gen[c], call)) return NN__PARTIALS__FAULT_PEER;
            if (!sys__gpu__mark_wait(fam, g->up_mark[c])) return NN__PRIMITIVES__FAULT_NO_DEVICE;
            parts.at[parts.count++] = g->up_at[c];
        }
        if (!last && parts.count == 1u) continue;
        doors->vector_sum((uint16_t*)(uintptr_t)(last ? out : part), parts, n, ctx->fault_word);
    }
    if (r != 0u) {
        g->up_at[r] = part;
        if (!sys__gpu__mark_record(fam, g->up_mark[r])) return NN__PRIMITIVES__FAULT_NO_DEVICE;
        __atomic_store_n(&g->up_gen[r], call, __ATOMIC_RELEASE);
        /* down: the sum, from the rank this one sent its partial to */
        if (!nn__partials__zzprivate_await(&g->down_gen[parent], call)) return NN__PARTIALS__FAULT_PEER;
        if (!sys__gpu__mark_wait(fam, g->down_mark[parent])) return NN__PRIMITIVES__FAULT_NO_DEVICE;
        parts.count = 1u;
        parts.at[0] = g->down_at[parent];
        doors->vector_sum((uint16_t*)(uintptr_t)out, parts, n, ctx->fault_word);
    }
    g->down_at[r] = out;
    if (!sys__gpu__mark_record(fam, g->down_mark[r])) return NN__PRIMITIVES__FAULT_NO_DEVICE;
    __atomic_store_n(&g->down_gen[r], call, __ATOMIC_RELEASE);
    return 0u;
}

/* ── `(nn__partials__reduce group part out n [residual])` -> `out` ─────────────────────────────────────────────────── */
static sys__heap_node nn__partials__zzabi_apply_reduce(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx) {
    if (argc != 4u && argc != 5u) return sys__engine__abi__error(SYS__OPCODES__FAULT_ARITY);
    uint64_t part = 0, part_room = 0, out = 0, out_room = 0, res = 0, res_room = 0;
    if (argv[0].dtype != SYS__KIND__VALUE_INT || argv[3].dtype != SYS__KIND__VALUE_INT
     || !nn__primitives__room(&argv[1], &part, &part_room) || !nn__primitives__room(&argv[2], &out, &out_room)
     || (argc == 5u && !nn__primitives__room(&argv[4], &res, &res_room)))
        return sys__engine__abi__error(SYS__OPCODES__FAULT_TYPE);
    const uint64_t n = argv[3].args[0];
    if (!nn__primitives__fits(n, part_room) || !nn__primitives__fits(n, out_room)
     || (argc == 5u && !nn__primitives__fits(n, res_room))) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_BOUNDS);
    const nn__doors* doors = nn__doors_for(ctx);
    if (doors == 0) return sys__engine__abi__error(NN__PRIMITIVES__FAULT_NO_DEVICE);
    const uint64_t fault = nn__partials__reduce_onto(ctx, doors, argv[0].args[0], part, out, n, argc == 5u ? res : 0u);
    if (fault != 0u) return sys__engine__abi__error(fault);
    return nn__doors_answer(&argv[2]);
}

SYS__ENGINE__ABI__BRIDGE(nn__partials__zzabi_adapter_join,   nn__partials__zzabi_apply_join)
SYS__ENGINE__ABI__BRIDGE(nn__partials__zzabi_adapter_reduce, nn__partials__zzabi_apply_reduce)

#endif /* SILVANN__PACKAGES_NN_CPU_OPCODES_PARTIALS__ABI_CUH */
