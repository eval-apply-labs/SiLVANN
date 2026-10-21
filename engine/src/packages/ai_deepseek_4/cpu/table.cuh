#ifndef SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_TABLE_CUH
#define SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_TABLE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../nn/cpu/doors.cuh"            /* nn's doors, and a buffer's room */

/* ══ WHAT EVERY ai_deepseek_4 VERB DOES FIRST — read its plane table ═════════════════════════════════════════════ */

/* The table's `planes` planes as addresses and rooms, its integers up to `reals`, and its reals up to `length`.
 * False on any element that is not what its place says. */
static bool ai_deepseek_4__table__zzpackage_read(uint64_t table, unsigned planes, unsigned reals, unsigned length,
                                              uint64_t* at, uint64_t* room, uint64_t* v, float* r) {
    return nn__primitives__table(table, planes, reals, length, at, room, v, r);
}

/* ⭐ A MASK'S THREE CELLS at `first` — its `r`, its `v` and its scratch, as addresses and rooms — or none. None is a table
 * that stops short of them or an integer 0 in the first; `*present` says which. False on anything else. */
static bool ai_deepseek_4__table__zzpackage_mask(uint64_t table, unsigned first, uint64_t* at, uint64_t* room, bool* present) {
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

/* ⭐ THE MASK ON A MATRIX'S ANSWER at one position, when the table carries one at `first`: `out` (`H` halves) +=
 * `r_i · (v_i · x)` for each of its rows, `x` the matrix's input (`cols` halves, rotated). Its rank is `v`'s rows, and `r`
 * has as many. 0, or `fault` when a cell is not what it should be. */
static uint64_t ai_deepseek_4__table__zzpackage_masked(const nn__doors* doors, unsigned int* over, uint64_t table, unsigned first,
                                                    uint64_t out, uint64_t x, uint64_t H, uint64_t cols, uint64_t fault) {
    uint64_t mk[3], mroom[3];
    bool present = false;
    if (!ai_deepseek_4__table__zzpackage_mask(table, first, mk, mroom, &present)) return fault;
    if (!present) return 0u;
    const uint64_t k = cols != 0u ? mroom[1] / (2u * cols) : 0u;
    if (k == 0u || k > AI_DEEPSEEK_4__MASK__RANK_MAX || mroom[1] != 2u * k * cols || mroom[0] != 2u * k * H || mroom[2] < 4u * k)
        return fault;
    for (uint64_t i = 0u; i < k; ++i)
        doors->rank1_dots((float*)(uintptr_t)(mk[2] + 4u * i), (const uint16_t*)(uintptr_t)(mk[1] + 2u * i * cols), 0,
                          (const uint16_t*)(uintptr_t)x, 0, 0, 1u, cols, cols, 0u);
    for (uint64_t i = 0u; i < k; ++i)
        doors->rank1_add((uint16_t*)(uintptr_t)out, (const uint16_t*)(uintptr_t)(mk[0] + 2u * i * H), (const float*)(uintptr_t)(mk[2] + 4u * i),
                         1u, 1u, H, H, over);
    return 0u;
}

/* A plane pair's codes and scales, and its width, into a grouped launch's entry `i`. */
static inline void ai_deepseek_4__table__zzpackage_group(nn__turboquant__groups* g, unsigned i, const uint64_t* at, unsigned plane,
                                                     uint64_t rows, uint64_t d, uint64_t out_at) {
    g->codes[i] = at[plane]; g->luts[i] = at[plane + 1u]; g->rows[i] = rows; g->d[i] = d; g->out_at[i] = out_at;
}

#endif /* SILVANN__PACKAGES_AI_DEEPSEEK_4_CPU_TABLE_CUH */
