#ifndef SILVANN__PACKAGES_AI_DEEPSEEK_4_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
#define SILVANN__PACKAGES_AI_DEEPSEEK_4_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
/* This file needs nothing: a macro body names nothing until something expands it. */
/* ══ ai_deepseek_4's VERBS — DeepSeek-V4.1's layer, a verb a device seam ══════════════════════════════════════
 * Each verb is one of the layer's sites over a plane table, launching nn's doors. The four streams go in and come out
 * in place; each site collapses them by the weights the site before it computed. ▶ `contracts/objects/layer.cuh`. */
#define ai_deepseek_4__LANGUAGE_CONTRACT__VERBS(X, PKG)                                                   \
    /* ⭐ THE ATTENTION SITE: the collapse, the norm, the query through its rank, the key into the window's ring, every */ \
    /* head over the ring with its sink, the grouped low-rank output, the streams mixed again                     */ \
    X(PKG, 0, "ai_deepseek_4__attention", ai_deepseek_4__attention__zzabi_adapter,   AI_DEEPSEEK_4__ATTENTION) \
    /* ⭐⭐ THE MLP SITE, CUT WHERE THE CPU TAKES OVER: the router's picks and the rotated input in a hand, the shared */ \
    /* expert while the CPU runs the routed ones, their sum and the streams mixed again                           */ \
    X(PKG, 1, "ai_deepseek_4__route",   ai_deepseek_4__route__zzabi_adapter,         AI_DEEPSEEK_4__ROUTE)     \
    X(PKG, 2, "ai_deepseek_4__shared",  ai_deepseek_4__shared__zzabi_adapter,        AI_DEEPSEEK_4__SHARED)    \
    X(PKG, 3, "ai_deepseek_4__close",   ai_deepseek_4__close__zzabi_adapter,         AI_DEEPSEEK_4__CLOSE)     \
    X(PKG, 4, "ai_deepseek_4__experts", ai_deepseek_4__experts__zzabi_adapter,       AI_DEEPSEEK_4__EXPERTS)    \
    /* ⭐ THE N-GRAM MEMORY: a position's looked-up rows into the streams, gated by each stream's match, before the site */ \
    X(PKG, 5, "ai_deepseek_4__engram",  ai_deepseek_4__engram__zzabi_adapter,        AI_DEEPSEEK_4__ENGRAM) \
    /* ⭐ A PROMPT AS ROWS: each site over a chunk's rows, its matrices once a chunk — the router's picks in the rows'   */ \
    /* hands, the routed experts once over the chunk, expert-major                                               */ \
    X(PKG, 6, "ai_deepseek_4__route_rows", ai_deepseek_4__route_rows__zzabi_adapter,   AI_DEEPSEEK_4__ROUTE_ROWS) \
    X(PKG, 7, "ai_deepseek_4__experts_rows", ai_deepseek_4__experts_rows__zzabi_adapter, AI_DEEPSEEK_4__EXPERTS_ROWS) \
    X(PKG, 8, "ai_deepseek_4__attention_rows", ai_deepseek_4__attention_rows__zzabi_adapter, AI_DEEPSEEK_4__ATTENTION_ROWS) \
    X(PKG, 9, "ai_deepseek_4__close_rows", ai_deepseek_4__close_rows__zzabi_adapter,   AI_DEEPSEEK_4__CLOSE_ROWS) \
    /* ⭐ A PICTURE THROUGH THE VISION TOWER: the patch embedding, a block, the tail (the last norm, the aligner) into */ \
    /* the text model's width. ▶ `cpu/opcodes/vision__abi.cuh`, `contracts/objects/vision.cuh`.                    */ \
    X(PKG, 10, "ai_deepseek_4__vision_embed", ai_deepseek_4__vision_embed__zzabi_adapter, AI_DEEPSEEK_4__VISION_EMBED) \
    X(PKG, 11, "ai_deepseek_4__vision_block", ai_deepseek_4__vision_block__zzabi_adapter, AI_DEEPSEEK_4__VISION_BLOCK) \
    X(PKG, 12, "ai_deepseek_4__vision_tail",  ai_deepseek_4__vision_tail__zzabi_adapter,  AI_DEEPSEEK_4__VISION_TAIL)

#endif /* SILVANN__PACKAGES_AI_DEEPSEEK_4_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH */
