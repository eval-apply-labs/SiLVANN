#ifndef SILVANN__ENGINE_BINDINGS_CUH
#define SILVANN__ENGINE_BINDINGS_CUH
/* ══ THE BINDINGS OBJECT — "the environment becomes simply a pointer" (architect) ═══════════════════════
 * ⭐ THE ENVIRONMENT IS ONE OBJECT, NOT LOOSE LOCALS: it is `sys__bindings`, an object in the package with
 *   its own header, its own rows and its own verbs, and the evaluator is handed a reference to it.
 * ⭐ AND THE EVALUATOR'S ARGUMENT COUNT IS HELD DOWN. It is
 *   `eng__eval(uint64_t bindings, uint64_t program, sys__heap_node* computing_base)`: three, and the
 *   third is the computing base the block is running on, handed straight through to `apply` so a verb can
 *   reach it. HOLD IT THERE — a field added to the bindings object reaches every verb for nothing, while
 *   an argument added to the evaluator is carried through every step of every program (`REASONED`).
 *   ⛳ The signature IS the claim, so re-derive it: `grep -n 'eng__eval(' engine/eval.cuh`.
 * ⛔⛔ NOTHING NAMES THE STRUCT BELOW. `grep -rn EngineBindings` over the tree finds its definition and
 *   nothing else; `EngineEvalPagedArm` has two hits, both in this file, and the claim gate's
 *   `no_old_tree_borrows` rule holds it there.
 * ⛳ IT IS LEFT STANDING BECAUSE THE ARGUMENT IN IT IS WORTH READING — the argument-count discipline and
 *   the absent return address are decisions about the evaluator, not about this type, and they outlive
 *   their carrier. The struct itself is a question for the architect and not a thing to remove
 *   unattended. */
struct EngineEvalPagedArm;

struct EngineBindings {
    EngineEvalPagedArm* arm;    /* the environment is `sys__bindings` — see above                      */
    /* ⛳ the RETURN ADDRESS is deliberately ABSENT: the caller places the returned value, which is the
     *   architect's *"this prevents us randomly trusting a return address."* */
};

#endif /* SILVANN__ENGINE_BINDINGS_CUH */
