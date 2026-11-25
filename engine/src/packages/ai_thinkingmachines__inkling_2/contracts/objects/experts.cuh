#ifndef SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CONTRACTS_OBJECTS_EXPERTS_CUH
#define SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CONTRACTS_OBJECTS_EXPERTS_CUH
/* This file needs nothing: it is offsets and fault words. */

#define AI_THINKINGMACHINES__INKLING_2__EXPERTS__FAULT_TABLE 0x49324554ull     /* "I2ET" — the experts table, the hand or the picks are not what the verb reads */

/* ══ `(ai_thinkingmachines__inkling_2__experts hand picks table routed)` — THE ROUTED EXPERTS, WHERE THEY LIVE ════════════════
 * The hand is what the card hands the worker for one position, in halves: [ the experts' input, rotated · H ][ the
 * router's weights · 16 ]. The picks are the router's, a node array of `top_k` integers.
 *
 * The table: three buffers, then integers, then two floats — the activation's soft caps — and, past them, the experts'
 * backing where they are read from a file (nn's node array, `nn/contracts/objects/expert.cuh`), the integer 0 or no cell
 * where every expert is in memory. */
#define AI_THINKINGMACHINES__INKLING_2__HAND__WEIGHTS       16u
#define AI_THINKINGMACHINES__INKLING_2__HAND__HALVES(H)     ((H) + AI_THINKINGMACHINES__INKLING_2__HAND__WEIGHTS)
/* a hand as a row of a chunk's: the hand, then its picks as words — where nn's rows routine finds them */
#define AI_THINKINGMACHINES__INKLING_2__HAND__PICKS(H)      ((2ull * AI_THINKINGMACHINES__INKLING_2__HAND__HALVES(H) + 7ull) & ~7ull)
#define AI_THINKINGMACHINES__INKLING_2__HAND__ROW(H, K)     (AI_THINKINGMACHINES__INKLING_2__HAND__PICKS(H) + 8ull * (K))

#define AI_THINKINGMACHINES__INKLING_2__EXP__SIGNS          0u      /* the rotation's signs, 512 halves */
#define AI_THINKINGMACHINES__INKLING_2__EXP__ZERO           1u      /* H zeros: what the weighted sum starts from */
#define AI_THINKINGMACHINES__INKLING_2__EXP__SCRATCH        2u
#define AI_THINKINGMACHINES__INKLING_2__EXP__PLANES         3u      /* integers from here on */
#define AI_THINKINGMACHINES__INKLING_2__EXP__LAYER          3u
#define AI_THINKINGMACHINES__INKLING_2__EXP__TYPE           4u      /* the experts' collection */
#define AI_THINKINGMACHINES__INKLING_2__EXP__UP_LUT         5u      /* offsets in a slot: the gate-and-up scales, the down codes, its scales */
#define AI_THINKINGMACHINES__INKLING_2__EXP__DOWN           6u
#define AI_THINKINGMACHINES__INKLING_2__EXP__DOWN_LUT       7u
#define AI_THINKINGMACHINES__INKLING_2__EXP__HIDDEN         8u
#define AI_THINKINGMACHINES__INKLING_2__EXP__INTER          9u
#define AI_THINKINGMACHINES__INKLING_2__EXP__EXPERTS        10u
#define AI_THINKINGMACHINES__INKLING_2__EXP__TOP_K          11u
#define AI_THINKINGMACHINES__INKLING_2__EXP__D              12u     /* the experts' width, 4 or 8 */
#define AI_THINKINGMACHINES__INKLING_2__EXP__INT8           13u     /* 1: the products in integers (nn's int8 gemvs), 0: exact */
#define AI_THINKINGMACHINES__INKLING_2__EXP__GATE_CAP       14u     /* floats from here on: the activation's soft caps, each above zero —     */
#define AI_THINKINGMACHINES__INKLING_2__EXP__UP_CAP         15u     /* gate_cap·tanh(g / gate_cap) · σ(g) · up_cap·tanh(u / up_cap)           */
#define AI_THINKINGMACHINES__INKLING_2__EXP__TABLE          16u
#define AI_THINKINGMACHINES__INKLING_2__EXP__BACKING        16u     /* optional, past the table */
#define AI_THINKINGMACHINES__INKLING_2__EXP__BATCH          16u     /* a prompt's experts a batch where the slots are a cache */

#endif /* SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CONTRACTS_OBJECTS_EXPERTS_CUH */
