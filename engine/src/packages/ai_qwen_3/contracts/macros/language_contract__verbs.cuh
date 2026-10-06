#ifndef SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
#define SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH

/* This file needs nothing: a macro body names nothing until something expands it. */
/* ══ ai_qwen_3's VERBS — the Qwen3-Next / 3.5 / 3.6 layer's blocks, as single opcodes ═════════════════════
 * ⚖ *"we could do a qwen package where we combine nn words into a qwen opcode, and then at compose time we can
 * decide what is the situation"* · ruled: wave-local fusion, a plane table, the residual inside,
 * the package named ai_qwen_3. ▶ `docs/design/pending/qwen_package.md`. */
#define ai_qwen_3__LANGUAGE_CONTRACT__VERBS(X, PKG)                                                    \
    /* ⭐⭐ THE SPARSE MoE BLOCK AND ITS RESIDUAL: `out = h1 + MoE(h1)` in nine launches of nn's doors,  */ \
    /* where the verbs it replaces make sixty-two. ▶ `cpu/opcodes/moe__abi.cuh`.                        */ \
    X(PKG, 0, "ai_qwen_3__moe",       ai_qwen_3__moe__zzabi_adapter,               AI_QWEN_3__MOE)         \
    /* ⭐⭐ THE SAME LAYER CUT AT THE DEVICE SEAMS — ⚖ *"a single word opcode for the attention, the expert, */ \
    /* the part before the expert and the part after. this way we can do the disjointed setup"*: a mixer, */ \
    /* then the MoE in three, each over its own plane table. ▶ `mixer__abi.cuh`, `moe__abi.cuh`.        */ \
    X(PKG, 1, "ai_qwen_3__deltanet",  ai_qwen_3__deltanet__zzabi_adapter,          AI_QWEN_3__DELTANET)    \
    X(PKG, 2, "ai_qwen_3__attention", ai_qwen_3__attention__zzabi_adapter,         AI_QWEN_3__ATTENTION)   \
    X(PKG, 3, "ai_qwen_3__pre_expert", ai_qwen_3__pre_expert__zzabi_adapter,       AI_QWEN_3__PRE_EXPERT)  \
    X(PKG, 4, "ai_qwen_3__experts",   ai_qwen_3__experts__zzabi_adapter,           AI_QWEN_3__EXPERTS)     \
    X(PKG, 5, "ai_qwen_3__post_expert", ai_qwen_3__post_expert__zzabi_adapter,     AI_QWEN_3__POST_EXPERT) \
    /* ⭐ THE READS OF A LAYER'S LIKELY EXPERTS, QUEUED BEFORE IT PICKS — ⚖ *"loader pool and prefetch"*.  */ \
    X(PKG, 6, "ai_qwen_3__prefetch",  ai_qwen_3__prefetch__zzabi_adapter,          AI_QWEN_3__PREFETCH) \
    /* ⭐⭐ A PROMPT'S ROWS THROUGH THE SAME THREE WORDS AT ONCE — the projections and the experts as expert-major */ \
    /* GEMMs, the recurrence walking the positions inside one launch. ▶ `cpu/opcodes/prefill__abi.cuh`.     */ \
    X(PKG, 7, "ai_qwen_3__deltanet_rows",  ai_qwen_3__deltanet_rows__zzabi_adapter,  AI_QWEN_3__DELTANET_ROWS)  \
    X(PKG, 8, "ai_qwen_3__attention_rows", ai_qwen_3__attention_rows__zzabi_adapter, AI_QWEN_3__ATTENTION_ROWS) \
    X(PKG, 9, "ai_qwen_3__moe_rows",       ai_qwen_3__moe_rows__zzabi_adapter,       AI_QWEN_3__MOE_ROWS) \
    /* ⭐ WHAT STREAMING A PROMPT'S EXPERTS FROM RAM HAS COST: bytes, ns copying, ns waiting on the card.     */ \
    X(PKG, 10, "ai_qwen_3__stream_counts", ai_qwen_3__stream_counts__zzabi_adapter,  AI_QWEN_3__STREAM_COUNTS) \
    /* ⭐⭐ THE ROUTED EXPERTS SPLIT BY OUTPUT ROWS OVER SEVERAL WORKERS — each holding its part of every expert:  */ \
    /* its gate_up rows and their swiglu, then, with the other parts' activations added, its down rows.       */ \
    /* ▶ `cpu/opcodes/moe__abi.cuh`.                                                                          */ \
    X(PKG, 11, "ai_qwen_3__experts_up",   ai_qwen_3__experts_up__zzabi_adapter,      AI_QWEN_3__EXPERTS_UP) \
    X(PKG, 12, "ai_qwen_3__experts_down", ai_qwen_3__experts_down__zzabi_adapter,    AI_QWEN_3__EXPERTS_DOWN) \
    /* ⭐ THE DENSE MLP, ITS NORM AND ITS RESIDUAL — a model without experts. ▶ `cpu/opcodes/mixer__abi.cuh`.       */ \
    X(PKG, 13, "ai_qwen_3__mlp",          ai_qwen_3__mlp__zzabi_adapter,             AI_QWEN_3__MLP) \
    /* ⭐⭐ THE ATTENTION OVER A CACHE IN TIERS — sink and hot fp16, warm TurboQuant 8, cold 4, every row rotated a head  */ \
    /* at a time; a position cooled as it ages. ▶ `contracts/objects/mixer.cuh`.                                     */ \
    X(PKG, 14, "ai_qwen_3__attention_tiered", ai_qwen_3__attention_tiered__zzabi_adapter, AI_QWEN_3__ATTENTION_TIERED) \
    /* ⭐⭐ A PROMPT'S ROWS THROUGH THE TIERED CACHE — the chunk into the hot ring, what it displaces cooled, then each row     */ \
    /* attending the cache's fp16 positions in order up to its own, and the warm and cold tiers. ▶ `prefill__abi.cuh`.    */ \
    X(PKG, 15, "ai_qwen_3__attention_tiered_rows", ai_qwen_3__attention_tiered_rows__zzabi_adapter, AI_QWEN_3__ATTENTION_TIERED_ROWS) \
    /* ⭐ THE DENSE MLP OVER A PROMPT'S ROWS, ITS NORM AND ITS RESIDUAL — the 27B's prompts as rows. ▶ `prefill__abi.cuh`. */ \
    X(PKG, 16, "ai_qwen_3__mlp_rows",       ai_qwen_3__mlp_rows__zzabi_adapter,        AI_QWEN_3__MLP_ROWS) \
    /* ⭐ A PROMPT AS ROWS WITH THE EXPERTS ON THE CPU: the MoE's three words over a chunk — the card's route, the CPU's  */ \
    /* experts expert-major, the card's close                                                                        */ \
    X(PKG, 17, "ai_qwen_3__pre_expert_rows",  ai_qwen_3__pre_expert_rows__zzabi_adapter,  AI_QWEN_3__PRE_EXPERT_ROWS) \
    X(PKG, 18, "ai_qwen_3__experts_rows",     ai_qwen_3__experts_rows__zzabi_adapter,     AI_QWEN_3__EXPERTS_ROWS) \
    X(PKG, 19, "ai_qwen_3__post_expert_rows", ai_qwen_3__post_expert_rows__zzabi_adapter, AI_QWEN_3__POST_EXPERT_ROWS)

#endif /* SILVANN__PACKAGES_AI_QWEN_3_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH */
