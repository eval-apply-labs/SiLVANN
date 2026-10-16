#ifndef SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_TABLE_CUH
#define SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_TABLE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../nn/cpu/doors.cuh"            /* nn's doors, and a buffer's room */

/* ══ WHAT THE BLOCK DOES FIRST — read its plane table ══════════════════════════════════════════════════════════ */

/* The table's planes (`planes` of them, first) as card addresses and rooms, then its integers up to `length`.
 * False on any element that is not what its place says. */
static bool ai_qwen_image__table__zzpackage_read(uint64_t table, unsigned planes, unsigned length,
                                                 uint64_t* at, uint64_t* room, uint64_t* v) {
    return nn__primitives__table(table, planes, length, length, at, room, v, 0);
}

#endif /* SILVANN__PACKAGES_AI_QWEN_IMAGE_CPU_TABLE_CUH */
