#ifndef SILVANN__PACKAGES_AI_GLM_5_3_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
#define SILVANN__PACKAGES_AI_GLM_5_3_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
/* This file needs nothing: a macro body names nothing until something expands it. */
/* ══ ai_glm_5_3's VERBS — GLM 5.3 Flash's layer, a verb a device seam ═════════════════════════════════════════
 * ⚖ *"lets add the ai_glm_5_3__ package for the fused steps"*. Each verb is one of the layer's blocks over a plane
 * table, launching nn's doors directly — the words a layer program was written in before, as one form each. The four
 * streams go in and come out in place. ▶ `contracts/objects/layer.cuh` for the tables. */
#define ai_glm_5_3__LANGUAGE_CONTRACT__VERBS(X, PKG)                                                   \
    /* ⭐ THE ATTENTION SITE: the hyper-connection's collapse, the norm, the mixer, and the streams mixed again */ \
    X(PKG, 0, "ai_glm_5_3__kda",      ai_glm_5_3__kda__zzabi_adapter,              AI_GLM_5_3__KDA)       \
    X(PKG, 1, "ai_glm_5_3__mla",      ai_glm_5_3__mla__zzabi_adapter,              AI_GLM_5_3__MLA)       \
    /* THE MLP SITE of the first layers, dense                                                          */ \
    X(PKG, 2, "ai_glm_5_3__mlp",      ai_glm_5_3__mlp__zzabi_adapter,              AI_GLM_5_3__MLP)       \
    /* ⭐⭐ THE MLP SITE WITH EXPERTS, CUT WHERE THE CPU TAKES OVER: the router's picks and the rotated input in */ \
    /* a hand, the shared expert while the CPU runs the routed ones, their sum and the streams mixed again   */ \
    X(PKG, 3, "ai_glm_5_3__route",    ai_glm_5_3__route__zzabi_adapter,            AI_GLM_5_3__ROUTE)     \
    X(PKG, 4, "ai_glm_5_3__shared",   ai_glm_5_3__shared__zzabi_adapter,           AI_GLM_5_3__SHARED)    \
    X(PKG, 5, "ai_glm_5_3__close",    ai_glm_5_3__close__zzabi_adapter,            AI_GLM_5_3__CLOSE)     \
    X(PKG, 6, "ai_glm_5_3__experts",  ai_glm_5_3__experts__zzabi_adapter,          AI_GLM_5_3__EXPERTS)  \
    /* ⭐ A PROMPT AS ROWS: the same sites over a chunk of positions, and the CPU's experts expert-major     */ \
    X(PKG, 7, "ai_glm_5_3__kda_rows",     ai_glm_5_3__kda_rows__zzabi_adapter,     AI_GLM_5_3__KDA_ROWS)     \
    X(PKG, 8, "ai_glm_5_3__mla_rows",     ai_glm_5_3__mla_rows__zzabi_adapter,     AI_GLM_5_3__MLA_ROWS)     \
    X(PKG, 9, "ai_glm_5_3__mlp_rows",     ai_glm_5_3__mlp_rows__zzabi_adapter,     AI_GLM_5_3__MLP_ROWS)     \
    X(PKG, 10, "ai_glm_5_3__route_rows",  ai_glm_5_3__route_rows__zzabi_adapter,   AI_GLM_5_3__ROUTE_ROWS)   \
    X(PKG, 11, "ai_glm_5_3__close_rows",  ai_glm_5_3__close_rows__zzabi_adapter,   AI_GLM_5_3__CLOSE_ROWS)   \
    X(PKG, 12, "ai_glm_5_3__experts_rows", ai_glm_5_3__experts_rows__zzabi_adapter, AI_GLM_5_3__EXPERTS_ROWS)

#endif /* SILVANN__PACKAGES_AI_GLM_5_3_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH */
