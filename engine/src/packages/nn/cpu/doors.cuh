#ifndef SILVANN__PACKAGES_NN_CPU_DOORS_CUH
#define SILVANN__PACKAGES_NN_CPU_DOORS_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/abi/gpu.cuh"   /* the shape of nn's door table */

/* ══ nn's DOORS ON A FAMILY — WHAT A VERB LAUNCHES THROUGH ══════════════════════════════════════════════
 *
 * The family loaded for a family carries one door table per package; this is nn's, read as its own
 * type. ⛳ ITS SIZE IS CHECKED BEFORE ANY DOOR IS TRUSTED: a family built from an older list has a
 * shorter table, and reading a door past its end would call whatever pointer happened to lie there.
 * Nothing, rather than a short table, is the answer a verb can act on. */
static inline const nn__doors* nn__doors_of(sys__silicon_family__id family) {
    const nn__doors* doors = (const nn__doors*)sys__silicon__package_doors(family, nn_pkg_id);
    return (doors != 0 && doors->size >= (uint32_t)sizeof(nn__doors)) ? doors : 0;
}

/* The doors a verb may launch through: the calling worker's, and only when it has a fault word to report
 * an overflow into. A context with no bound worker has neither, and a verb given no doors refuses with
 * NN__PRIMITIVES__FAULT_NO_DEVICE rather than launching into nothing. */
static inline const nn__doors* nn__doors_for(const sys__engine__ctx* ctx) {
    return (ctx != 0 && ctx->fault_word != 0) ? nn__doors_of(ctx->family) : 0;
}

/* What a verb answers when its answer is one of its own operands — the out buffer, so a program can chain
 * it. The node comes out of the form borrowed, and a verb hands back a hold it owns, so it is retained on
 * the way out; the bridge gives that hold back once the form has taken its own. */
static inline sys__heap_node nn__doors_answer(const sys__heap_node* operand) {
    if (sys__heap_node__carries_reference(operand->dtype)) (void)sys__heap_object__retain(operand->args[0]);
    return *operand;
}

#endif /* SILVANN__PACKAGES_NN_CPU_DOORS_CUH */
