#ifndef SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
#define SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH
/* This file needs nothing: a macro body names nothing until something expands it. */
/* ══ ai_thinkingmachines__inkling_2's VERBS — Inkling 2's own ═══════════════════════════════════════════════════════
 * ⚖ 2026-11-24: *"each model should implement its own expert verb, so the shared verbs go into nn and the collection of
 * shared verbs used by a model go to their publisher"*. The routed experts are nn's routine; the verb is this model's:
 * its table, its hand, its activation's caps (▶ `contracts/objects/experts.cuh`).
 * ⚖ 2026-11-24: *"the threshold that makes a breakage earn its keep is if we do have a parameter to pass that the
 * evaluator can change during the execution"* — a layer's application is two verbs around its experts', the one place
 * it hands the evaluator a choice: where they run (▶ `contracts/objects/layer.cuh`). */
#define ai_thinkingmachines__inkling_2__LANGUAGE_CONTRACT__VERBS(X, PKG)                                                      \
    X(PKG, 0, "ai_thinkingmachines__inkling_2__experts", ai_thinkingmachines__inkling_2__experts__zzabi_adapter, AI_THINKINGMACHINES__INKLING_2__EXPERTS) \
    X(PKG, 1, "ai_thinkingmachines__inkling_2__before_experts", ai_thinkingmachines__inkling_2__before_experts__zzabi_adapter, AI_THINKINGMACHINES__INKLING_2__BEFORE_EXPERTS) \
    X(PKG, 2, "ai_thinkingmachines__inkling_2__after_experts", ai_thinkingmachines__inkling_2__after_experts__zzabi_adapter, AI_THINKINGMACHINES__INKLING_2__AFTER_EXPERTS)

#endif /* SILVANN__PACKAGES_AI_THINKINGMACHINES__INKLING_2_CONTRACTS_MACROS_LANGUAGE_CONTRACT__VERBS_CUH */
