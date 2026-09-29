#ifndef SILVANN__PACKAGES_AI_GLM_5_3_CPU_TABLE_CUH
#define SILVANN__PACKAGES_AI_GLM_5_3_CPU_TABLE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../nn/cpu/doors.cuh"            /* nn's doors, and a buffer's room */

/* ══ WHAT EVERY ai_glm_5_3 VERB DOES FIRST — read its plane table ═════════════════════════════════════════════ */

/* The table's `planes` planes as addresses and rooms, its integers up to `reals`, and its reals up to `length`.
 * False on any element that is not what its place says. */
static bool ai_glm_5_3__table__zzpackage_read(uint64_t table, unsigned planes, unsigned reals, unsigned length,
                                              uint64_t* at, uint64_t* room, uint64_t* v, float* r) {
    if (sys__node_array__length(table) < length) return false;
    for (unsigned i = 0u; i < planes; ++i) {
        const sys__heap_node n = sys__node_array__borrow(table, i);
        if (!nn__primitives__room(&n, &at[i], &room[i])) return false;
    }
    for (unsigned i = planes; i < reals; ++i) {
        const sys__heap_node n = sys__node_array__borrow(table, i);
        if (n.dtype != SYS__KIND__VALUE_INT) return false;
        v[i] = n.args[0];
    }
    for (unsigned i = reals; i < length; ++i) {
        const sys__heap_node n = sys__node_array__borrow(table, i);
        if (n.dtype != SYS__KIND__VALUE_FLOAT) return false;
        r[i] = (float)sys__heap_node__real(n.args[0]);
    }
    return true;
}

/* The router's `k` picks — by σ(logit) + bias — into the first `k` elements of `picks`, and their weights into
 * `weights`. The card writes the nodes itself when the heap is registered with it, and a card buffer lands them
 * otherwise. Waits for the card: what the picks are is what the CPU is handed next. */
static bool ai_glm_5_3__table__zzpackage_top_k(sys__engine__ctx* ctx, const nn__doors* doors, uint64_t picks, uint64_t weights,
                                               uint64_t logits, uint64_t bias, uint64_t n, uint64_t k, float scale) {
    if (sys__node_array__length(picks) < k) return false;
    sys__heap_node zero = sys__heap_node__nothing();
    zero.dtype = SYS__KIND__VALUE_INT; zero.args[0] = 0ull;
    for (uint64_t j = 0u; j < k; ++j) {
        const sys__kind kind = sys__node_array__type(picks, j);
        if (kind != SYS__KIND__VALUE_INT && kind != SYS__KIND__VALUE_NULL) return false;   /* holds nothing to lose */
        if (!sys__node_array__set(picks, j, &zero)) return false;
    }
    sys__heap_node* cells = sys__node_array__cells(picks, k);
    if (cells == 0) return false;
    const uint64_t words = (uint64_t)(&cells[1].args[0] - &cells[0].args[0]);
    if (ctx->heap_card != 0 && cells >= ctx->heap_host && (uint64_t)(cells - ctx->heap_host) + k <= ctx->heap_nodes) {
        sys__heap_node* card = ctx->heap_card + (cells - ctx->heap_host);
        doors->vector_top_k_biased(&card[0].args[0], words, (uint16_t*)(uintptr_t)weights, (const uint16_t*)(uintptr_t)logits,
                                   (const uint16_t*)(uintptr_t)bias, n, k, scale);
        return sys__gpu__compute_completed(ctx->family);
    }
    void* landing = 0;
    uint64_t got[AI_GLM_5_3__TOP_K_MAX];
    if (k > AI_GLM_5_3__TOP_K_MAX || !sys__gpu__memory_allocate(ctx->family, &landing, k * sizeof(uint64_t))) return false;
    doors->vector_top_k_biased((uint64_t*)landing, 1ull, (uint16_t*)(uintptr_t)weights, (const uint16_t*)(uintptr_t)logits,
                               (const uint16_t*)(uintptr_t)bias, n, k, scale);
    const bool read = sys__gpu__memory_read(ctx->family, got, landing, k * sizeof(uint64_t));
    sys__gpu__memory_free(ctx->family, landing);
    if (!read) return false;
    for (uint64_t j = 0u; j < k; ++j) cells[j].args[0] = got[j];
    return true;
}

/* A plane pair's codes and scales, and its width, into a grouped launch's entry `i`. */
static inline void ai_glm_5_3__table__zzpackage_group(nn__turboquant__groups* g, unsigned i, const uint64_t* at, unsigned plane,
                                                     uint64_t rows, uint64_t d, uint64_t out_at) {
    g->codes[i] = at[plane]; g->luts[i] = at[plane + 1u]; g->rows[i] = rows; g->d[i] = d; g->out_at[i] = out_at;
}

#endif /* SILVANN__PACKAGES_AI_GLM_5_3_CPU_TABLE_CUH */
