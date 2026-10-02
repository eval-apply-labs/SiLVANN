#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_TABLE_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_TABLE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../nn/cpu/doors.cuh"            /* nn's doors, and a buffer's room */

/* ══ WHAT EVERY ai_qwen_3 VERB DOES FIRST — read its plane table, and land the router's picks ════════════ */

/* The table's planes (`planes` of them, first) as card addresses and rooms, then its integers up to `length`.
 * False on any element that is not what its place says. */
static bool ai_qwen_3__table__zzpackage_read(uint64_t table, unsigned planes, unsigned length,
                                             uint64_t* at, uint64_t* room, uint64_t* v) {
    if (sys__node_array__length(table) < length) return false;
    for (unsigned i = 0u; i < planes; ++i) {
        const sys__heap_node n = sys__node_array__borrow(table, i);
        if (!nn__primitives__room(&n, &at[i], &room[i])) return false;
    }
    for (unsigned i = planes; i < length; ++i) {
        const sys__heap_node n = sys__node_array__borrow(table, i);
        if (n.dtype != SYS__KIND__VALUE_INT) return false;
        v[i] = n.args[0];
    }
    return true;
}

/* An optional integer cell past a table's fixed length: true when the table has it and it is 1. */
static bool ai_qwen_3__table__zzpackage_flag(uint64_t table, unsigned index) {
    if (sys__node_array__length(table) <= index) return false;
    const sys__heap_node n = sys__node_array__borrow(table, index);
    return n.dtype == SYS__KIND__VALUE_INT && n.args[0] == 1ull;
}

/* ⭐ A RANK-1 MASK'S THREE CELLS at `first` — its `r`, its `v` and its scratch, as addresses and rooms — or none. None is
 * a table that stops short of them or an integer 0 in the first; `*present` says which. False on anything else. */
static bool ai_qwen_3__table__zzpackage_mask(uint64_t table, unsigned first, uint64_t* at, uint64_t* room, bool* present) {
    *present = false;
    if (sys__node_array__length(table) <= first) return true;
    const sys__heap_node r = sys__node_array__borrow(table, first);
    if (r.dtype == SYS__KIND__VALUE_INT && r.args[0] == 0ull) return true;
    if (sys__node_array__length(table) < first + 3u) return false;
    for (unsigned i = 0u; i < 3u; ++i) {
        const sys__heap_node n = sys__node_array__borrow(table, first + i);
        if (!nn__primitives__room(&n, &at[i], &room[i])) return false;
    }
    *present = true;
    return true;
}

/* ⭐ THE MASK ON A PROJECTION'S ANSWER, when the table carries one at `first`: `ao` (`T` rows of `H`) += `r · (v · x)`, `x`
 * the projection's input (`cols` a row, rotated). 0, or `fault` when a cell is not what it should be. */
static uint64_t ai_qwen_3__table__zzpackage_masked(const nn__doors* doors, unsigned int* over, uint64_t table, unsigned first,
                                                   uint64_t ao, uint64_t x, uint64_t T, uint64_t H, uint64_t cols, uint64_t fault) {
    uint64_t mk[3], mroom[3];
    bool present = false;
    if (!ai_qwen_3__table__zzpackage_mask(table, first, mk, mroom, &present)) return fault;
    if (!present) return 0u;
    if (!nn__primitives__fits(H, mroom[0]) || !nn__primitives__fits(cols, mroom[1]) || mroom[2] < 4u * T) return fault;
    doors->rank1_dots((float*)(uintptr_t)mk[2], (const uint16_t*)(uintptr_t)mk[1], 0, (const uint16_t*)(uintptr_t)x, 0, 0, T, cols, cols, 0u);
    doors->rank1_add((uint16_t*)(uintptr_t)ao, (const uint16_t*)(uintptr_t)mk[0], (const float*)(uintptr_t)mk[2], T, 1u, H, H, over);
    return 0u;
}

/* The router's `k` largest of `n` logits: their indices into the first `k` elements of `picks` (a node array)
 * and into `chosen`, their softmax into `weights`. The card writes the nodes itself when the heap is
 * registered with it, and a card buffer lands them otherwise — `nn__vector__top_k`'s two paths. Waits for
 * the card: what the picks are is what the next step needs to know. */
static bool ai_qwen_3__table__zzpackage_top_k(sys__engine__ctx* ctx, const nn__doors* doors, uint64_t picks,
                                              uint64_t weights, uint64_t logits, uint64_t n, uint64_t k, uint64_t* chosen) {
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
        doors->vector_top_k(&card[0].args[0], words, (uint16_t*)(uintptr_t)weights, (const uint16_t*)(uintptr_t)logits, n, k);
        if (!sys__gpu__compute_completed(ctx->family)) return false;
        for (uint64_t j = 0u; j < k; ++j) chosen[j] = cells[j].args[0];
        return true;
    }
    void* landing = 0;
    if (!sys__gpu__memory_allocate(ctx->family, &landing, k * sizeof(uint64_t))) return false;
    doors->vector_top_k((uint64_t*)landing, 1ull, (uint16_t*)(uintptr_t)weights, (const uint16_t*)(uintptr_t)logits, n, k);
    const bool read = sys__gpu__memory_read(ctx->family, chosen, landing, k * sizeof(uint64_t));
    sys__gpu__memory_free(ctx->family, landing);
    if (!read) return false;
    for (uint64_t j = 0u; j < k; ++j) cells[j].args[0] = chosen[j];
    return true;
}

/* ⭐ THE SAME FOR `rows` ROWS OF LOGITS (`n` apart), landed in card memory at `landing` (`rows · k` words) rather
 * than in a node array — a prompt's picks are hundreds, and nothing but this verb reads them: row `r`'s `k` picks
 * into `chosen[r·k ..]`, its weights at `weights + r·k`. One launch a row, then one read for them all. */
static bool ai_qwen_3__table__zzpackage_top_k_rows(sys__engine__ctx* ctx, const nn__doors* doors, uint64_t landing,
                                                   uint64_t weights, uint64_t logits, uint64_t n, uint64_t k,
                                                   uint64_t rows, uint64_t* chosen) {
    for (uint64_t r = 0u; r < rows; ++r)
        doors->vector_top_k((uint64_t*)(uintptr_t)(landing + 8u * r * k), 1ull, (uint16_t*)(uintptr_t)(weights + 2u * r * k),
                            (const uint16_t*)(uintptr_t)(logits + 2u * r * n), n, k);
    return sys__gpu__memory_read(ctx->family, chosen, (const void*)(uintptr_t)landing, (size_t)(8u * rows * k));
}

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CPU_TABLE_CUH */
