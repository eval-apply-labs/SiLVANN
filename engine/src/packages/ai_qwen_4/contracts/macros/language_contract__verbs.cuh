#ifndef SILVANN__PACKAGES_AI_QWEN_4_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
#define SILVANN__PACKAGES_AI_QWEN_4_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
/* This file needs nothing: a macro body names nothing until something expands it. */
/* ══ ai_qwen_4's VERBS — Qwen3.8-Flash-Next's layer sites, a verb each ═════════════════════════════════════════
 * ⚖ 2026-10-31: *"then try to speed it up with the verbs"*. Each verb is a site of `lisp/qwen4_exp.lisp`, its nn words
 * launched from one apply (▶ `contracts/objects/layer.cuh` for the tables and the forms). */
#define ai_qwen_4__LANGUAGE_CONTRACT__VERBS(X, PKG)                                                      \
    X(PKG, 0, "ai_qwen_4__gated_pre", ai_qwen_4__gated_pre__zzabi_adapter, AI_QWEN_4__GATED_PRE)         \
    X(PKG, 1, "ai_qwen_4__attention", ai_qwen_4__attention__zzabi_adapter, AI_QWEN_4__ATTENTION)         \
    X(PKG, 2, "ai_qwen_4__deltanet",  ai_qwen_4__deltanet__zzabi_adapter,  AI_QWEN_4__DELTANET)          \
    X(PKG, 3, "ai_qwen_4__shared",    ai_qwen_4__shared__zzabi_adapter,    AI_QWEN_4__SHARED)         \
    X(PKG, 4, "ai_qwen_4__gated_pre_rows", ai_qwen_4__gated_pre_rows__zzabi_adapter, AI_QWEN_4__GATED_PRE_ROWS) \
    X(PKG, 5, "ai_qwen_4__inject_rows",    ai_qwen_4__inject_rows__zzabi_adapter,    AI_QWEN_4__INJECT_ROWS)    \
    X(PKG, 6, "ai_qwen_4__deltanet_rows",  ai_qwen_4__deltanet_rows__zzabi_adapter,  AI_QWEN_4__DELTANET_ROWS)  \
    X(PKG, 7, "ai_qwen_4__attention_rows", ai_qwen_4__attention_rows__zzabi_adapter, AI_QWEN_4__ATTENTION_ROWS) \
    X(PKG, 8, "ai_qwen_4__shared_rows",    ai_qwen_4__shared_rows__zzabi_adapter,    AI_QWEN_4__SHARED_ROWS)    \
    X(PKG, 9, "ai_qwen_4__route_rows",     ai_qwen_4__route_rows__zzabi_adapter,     AI_QWEN_4__ROUTE_ROWS)     \
    X(PKG, 10, "ai_qwen_4__route",         ai_qwen_4__route__zzabi_adapter,          AI_QWEN_4__ROUTE)

#endif /* SILVANN__PACKAGES_AI_QWEN_4_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH */
