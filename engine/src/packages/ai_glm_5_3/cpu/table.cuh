#ifndef SILVANN__PACKAGES_AI_GLM_5_3_CPU_TABLE_CUH
#define SILVANN__PACKAGES_AI_GLM_5_3_CPU_TABLE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../nn/cpu/doors.cuh"            /* nn's doors, and a buffer's room */

/* ══ WHAT EVERY ai_glm_5_3 VERB DOES FIRST — read its plane table ═════════════════════════════════════════════ */

/* The table's `planes` planes as addresses and rooms, its integers up to `reals`, and its reals up to `length`.
 * False on any element that is not what its place says. */
static bool ai_glm_5_3__table__zzpackage_read(uint64_t table, unsigned planes, unsigned reals, unsigned length,
                                              uint64_t* at, uint64_t* room, uint64_t* v, float* r) {
    return nn__primitives__table(table, planes, reals, length, at, room, v, r);
}

/* A plane pair's codes and scales, and its width, into a grouped launch's entry `i`. */
static inline void ai_glm_5_3__table__zzpackage_group(nn__turboquant__groups* g, unsigned i, const uint64_t* at, unsigned plane,
                                                     uint64_t rows, uint64_t d, uint64_t out_at) {
    g->codes[i] = at[plane]; g->luts[i] = at[plane + 1u]; g->rows[i] = rows; g->d[i] = d; g->out_at[i] = out_at;
}

#endif /* SILVANN__PACKAGES_AI_GLM_5_3_CPU_TABLE_CUH */
