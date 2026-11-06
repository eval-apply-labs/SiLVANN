#ifndef SILVANN__PACKAGES_AI_QWEN_4_CPU_TABLE_CUH
#define SILVANN__PACKAGES_AI_QWEN_4_CPU_TABLE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../nn/cpu/doors.cuh"            /* nn's doors, and a buffer's room */

/* ══ A LAYER'S TABLE, CELL BY CELL ═══════════════════════════════════════════════════════════════════════════
 * Flash-Next's layer table keeps each matrix as its three cells together — codes, scales, bits — and the integer 0 in
 * a cell the layer does not have (an attention layer's DeltaNet cells), so a cell is read as what it holds. */
#define AI_QWEN_4__CELL__INT   0u
#define AI_QWEN_4__CELL__PLANE 1u

typedef struct ai_qwen_4__table {
    uint64_t at[AI_QWEN_4__LT__TABLE], room[AI_QWEN_4__LT__TABLE], v[AI_QWEN_4__LT__TABLE];
    unsigned char kind[AI_QWEN_4__LT__TABLE];
} ai_qwen_4__table;

/* The table's first AI_QWEN_4__LT__TABLE cells: a buffer or a plane as its address and room, an integer as itself. False
 * on a cell that is neither, or a buffer of another worker's. */
static bool ai_qwen_4__table__zzpackage_read(uint64_t table, ai_qwen_4__table* t) {
    if (!sys__node_array__is(table) || sys__node_array__length(table) < AI_QWEN_4__LT__TABLE) return false;
    sys__node_array_walk w;
    if (!sys__node_array__walk(table, 0ull, &w)) return false;
    for (unsigned i = 0u; i < AI_QWEN_4__LT__TABLE; ++i, sys__node_array__next(&w)) {
        const sys__heap_node* n = sys__node_array__walk_cell(&w);
        if (n == 0) return false;
        t->at[i] = 0ull; t->room[i] = 0ull; t->v[i] = 0ull;
        if (n->dtype == SYS__KIND__VALUE_INT) {
            t->v[i] = n->args[0];
            t->kind[i] = AI_QWEN_4__CELL__INT;
        } else {
            if (!nn__primitives__room(n, &t->at[i], &t->room[i])) return false;
            t->kind[i] = AI_QWEN_4__CELL__PLANE;
        }
    }
    return true;
}

/* A matrix at cell `c`: its codes and scales planes and its bits, a width nn's gemv knows. */
static inline bool ai_qwen_4__table__zzpackage_matrix(const ai_qwen_4__table* t, unsigned c) {
    return c + 2u < AI_QWEN_4__LT__TABLE && t->kind[c] == AI_QWEN_4__CELL__PLANE && t->kind[c + 1u] == AI_QWEN_4__CELL__PLANE
        && t->kind[c + 2u] == AI_QWEN_4__CELL__INT && nn__turboquant__rate_is_known(t->v[c + 2u]);
}

/* A vector at cell `c` (a norm's gain, a bias) of at least `n` halves. */
static inline bool ai_qwen_4__table__zzpackage_vector(const ai_qwen_4__table* t, unsigned c, uint64_t n) {
    return c < AI_QWEN_4__LT__TABLE && t->kind[c] == AI_QWEN_4__CELL__PLANE && nn__primitives__fits(n, t->room[c]);
}

/* ══ THE SHAPE TABLE ══════════════════════════════════════════════════════════════════════════════════════════ */
typedef struct ai_qwen_4__shape {
    uint64_t at[AI_QWEN_4__SH__TABLE], room[AI_QWEN_4__SH__TABLE], v[AI_QWEN_4__SH__TABLE];
    float r[AI_QWEN_4__SH__TABLE];
} ai_qwen_4__shape;

/* Its buffers, its integers and its epsilon; false on any cell that is not what its place says, or a shape out of the
 * bounds every verb assumes (each width at most 2^24, the rotation's signs 512 halves). */
static bool ai_qwen_4__table__zzpackage_shape(const sys__heap_node* cell, ai_qwen_4__shape* s) {
    if (!sys__heap_node__carries_reference(cell->dtype) || !sys__node_array__is(cell->args[0])
     || !nn__primitives__table(cell->args[0], AI_QWEN_4__SH__HIDDEN, AI_QWEN_4__SH__EPS, AI_QWEN_4__SH__TABLE,
                               s->at, s->room, s->v, s->r))
        return false;
    for (unsigned i = AI_QWEN_4__SH__HIDDEN; i < AI_QWEN_4__SH__EPS; ++i)
        if (s->v[i] == 0ull || s->v[i] > (1ull << 24)) return false;
    return nn__primitives__fits(512u, s->room[AI_QWEN_4__SH__SIGNS])
        && nn__primitives__fits(s->v[AI_QWEN_4__SH__STREAMS] * s->v[AI_QWEN_4__SH__HIDDEN], s->room[AI_QWEN_4__SH__ONES]);
}

#endif /* SILVANN__PACKAGES_AI_QWEN_4_CPU_TABLE_CUH */
