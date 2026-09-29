#ifndef SILVANN__ENGINE_BINDINGS_CUH
#define SILVANN__ENGINE_BINDINGS_CUH
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * The use-site and register counts below are `k_eval` figures, taken while the evaluator ran on the card
 * (⚖ *"sys and the evaluator go host"*): the conclusion may survive on a CPU, the numbers will not
 * reproduce. Kept because a measured number with a dead subject is still evidence of what was tried.
 *   · the old engine spread the interpreter's state across SIX locals with 352 use sites (`unev` 128,
 *     `frame_base` 81, `env_arm` 42, `unev_pos` 50, `unev_valid` 33, `frame_cursor` 18); `MEASURED`
 *     S74: 31 of its 51 real VALUE registers were the paged-list cursor plus the frame machine.
 *   · the argument-count dose curve, `MEASURED` on the old engine: 2 pointer args at 8 call sites
 *     126 -> 125 registers (free); 7 args at 4 call sites 126 -> 139 (+13). Three arguments sat at the
 *     flat end of it, `REASONED` from those two rows.
 *   · a return address would have reintroduced `frame_base`/`frame_cursor`, 13 of those 51 registers.
 *   · the struct below was the port's first scaffold: an opaque pointer at an environment not yet
 *     written, so the tree could build from commit one. `sys__bindings` replaced it.
 *     ITS `SysSpineBase base` FIELD IS GONE: the type was the old tree's, a borrow from `src_old/`, and
 *     nothing read it.
 * ⛳ RETIREMENT: MET — the evaluator runs on the host. Kept until release, when the whole block goes.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
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
