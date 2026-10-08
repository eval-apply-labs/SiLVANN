#ifndef SILVANN__PACKAGES_AI_QWEN_3_CPU_TABLE_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CPU_TABLE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../nn/cpu/doors.cuh"            /* nn's doors, and a buffer's room */

/* ══ WHAT EVERY ai_qwen_3 VERB DOES FIRST — read its plane table, and land the router's picks ════════════ */

/* The table's planes (`planes` of them, first) as card addresses and rooms, then its integers up to `length`.
 * False on any element that is not what its place says. */
static bool ai_qwen_3__table__zzpackage_read(uint64_t table, unsigned planes, unsigned length,
                                             uint64_t* at, uint64_t* room, uint64_t* v) {
    return nn__primitives__table(table, planes, length, length, at, room, v, 0);
}

/* An optional integer cell past a table's fixed length: true when the table has it and it is 1. */
static bool ai_qwen_3__table__zzpackage_flag(uint64_t table, unsigned index) {
    return nn__primitives__table_flag(table, index);
}

/* ⭐ A RANK-1 MASK'S THREE CELLS at `first` — its `r`, its `v` and its scratch, as addresses and rooms — or none. None is
 * a table that stops short of them or an integer 0 in the first; `*present` says which. False on anything else. */
static bool ai_qwen_3__table__zzpackage_mask(uint64_t table, unsigned first, uint64_t* at, uint64_t* room, bool* present) {
    *present = false;
    if (sys__node_array__length(table) <= first) return true;
    const sys__heap_node r = sys__node_array__borrow(table, first);
    if (r.dtype == SYS__KIND__VALUE_INT && r.args[0] == 0ull) return true;
    if (sys__node_array__length(table) < first + 3u) return false;
    sys__node_array_walk w;
    if (!sys__node_array__walk(table, first, &w)) return false;
    for (unsigned i = 0u; i < 3u; ++i, sys__node_array__next(&w)) {
        const sys__heap_node* n = sys__node_array__walk_cell(&w);
        if (n == 0 || !nn__primitives__room(n, &at[i], &room[i])) return false;
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

/* ⭐ THE ROUTER'S `k` LARGEST OF `n` LOGITS FOR `rows` ROWS (`n` apart), landed in card memory at `landing` (`rows · k`
 * words) rather than in a node array as nn's `nn__routed__top_k` lands a position's — a prompt's picks are hundreds, and nothing but this verb reads them: row `r`'s `k` picks
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
