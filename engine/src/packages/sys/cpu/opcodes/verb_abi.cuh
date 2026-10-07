#ifndef SILVANN__PACKAGES_SYS_CPU_OPCODES_VERB_ABI_CUH
#define SILVANN__PACKAGES_SYS_CPU_OPCODES_VERB_ABI_CUH
#include "../../contracts/objects/verb_abi.cuh"      /* its constants, fault words and the layouts it reads and writes */
#include "../../contracts/abi/silicon_family.cuh"    /* the family a verb's context names */
#include "../heap_node__header.cuh"                  /* the node every value in this language is made of */
#include "../heap__header.cuh"                       /* where a verb's room comes from */
#include "../heap_object__header.cuh"                /* holds, and the release the bridge performs */
#include "../error.cuh"                              /* the error value a refusal answers */
/* ══ THE VERB BOUNDARY — ONE BRIDGE, USED BY EVERY VERB IN THE RULED SHAPE ══════════════════════════
 */
/* ══ WHAT A VERB IS ═══════════════════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"the abi will want the proper parameters to be invoked and will return a value
 * (which can be an address or a value) and we can have the fault be the return value of the abi call."*
 *
 *     sys__heap_node verb(const sys__heap_node* argv, unsigned argc, sys__engine__ctx* ctx);
 *
 * The evaluator's dispatch calls `APPLY(base, form)` — a generated switch in `heap_object__impl.cuh`.
 * This file is the one bridge between the two shapes, written ONCE as a macro instead of once per verb.
 *
 * ⭐⭐ AND THE `sys` HALF NEEDS NOTHING ELSE, WHICH IS THE FINDING THAT MADE THIS CHEAP. `MEASURED`
 * over every published `sys` verb body: **NOT ONE OF THEM CALLS `eng__eval`.** There are no special
 * forms in the ABI's sense. `if` does not evaluate an arm — it retags the chosen `QUOTED_LIST` as an
 * ordinary `OBJECT_REFERENCE` and answers it; `eng__eval__zzprivate_first_form` then descends into it
 * because it names a LIST — the test is on the POINTEE's kind, not on a rewrite.
 * ⇒ ★★ **CONTROL FLOW LIVES IN THE EVALUATOR'S SCAN, NOT IN THE VERBS**, so "special form" here means
 *   only "a verb that answers a reference to a list", and that needs no special ABI at all.
 * ⇒ ⛳ THEREFORE NO `sys` VERB NEEDS `ctx` FOR CONTROL — only the two allocators read it, for their
 *   arena. What needs the rest of `ctx` is `nn`: a stream to launch on and a fault word to accumulate into.
 */

/* ⛳ ONE PER EVALUATOR, AND `sys` IGNORES IT ENTIRELY — so the tier needs nobody to supply one. The
 * default below is the shipping one; a harness that owns its device and fault word defines
 * `SYS__ENGINE__ABI__CTX_PROVIDED` and supplies its own.
 * ⛔ AND THE DEFAULT IS A DEFINITION, NOT AN `extern`. A symbol declared and defined nowhere is
 * not a compile error anywhere; it is a link error at the end of the longest step. */
#ifndef SYS__ENGINE__ABI__CTX_PROVIDED
/* ⭐ ON THE CPU THE CONTEXT IS THE WORKER'S: the silicon it bound, its device, and the fault word on that
 * device. The boot fills it in when it binds one (`sys__engine__ctx_bind`); until then the family is NONE
 * and every verb that needs a card refuses with NO_DEVICE. ONE PER WORKER, and a verb is handed the one of
 * the worker its runner is (`sys__silicon__worker`); the bridge hands each call its own copy with the call's
 * arena in it, so nothing here changes under a verb that is reading it. */
/* ⛳ EVERY ONE STARTS ON NO FAMILY — zero is the host's id, so an unbound worker must say NONE, not zero. */
#define SYS__ENGINE__ZZPRIVATE_UNBOUND { SYS__SILICON_FAMILY__NONE, 0u, 0, 0, 0, 0, 0, 0ull }
static_assert(SYS__SILICON__WORKERS_MAX == 8u, "one unbound context below for each worker a machine may have");
static sys__engine__ctx sys__engine__zzprivate_workers[SYS__SILICON__WORKERS_MAX] = {
    SYS__ENGINE__ZZPRIVATE_UNBOUND, SYS__ENGINE__ZZPRIVATE_UNBOUND, SYS__ENGINE__ZZPRIVATE_UNBOUND, SYS__ENGINE__ZZPRIVATE_UNBOUND,
    SYS__ENGINE__ZZPRIVATE_UNBOUND, SYS__ENGINE__ZZPRIVATE_UNBOUND, SYS__ENGINE__ZZPRIVATE_UNBOUND, SYS__ENGINE__ZZPRIVATE_UNBOUND };
#undef SYS__ENGINE__ZZPRIVATE_UNBOUND
static inline sys__engine__ctx* sys__engine__ctx_current(void) { return &sys__engine__zzprivate_workers[sys__silicon__worker()]; }
static inline void sys__engine__ctx_bind(unsigned int worker, sys__silicon_family__id family, unsigned device, uint32_t* fault_word) {
    if (worker >= SYS__SILICON__WORKERS_MAX) return;
    sys__engine__ctx* c = &sys__engine__zzprivate_workers[worker];
    c->family     = family;
    c->device     = device;
    c->stream     = 0;
    c->fault_word = fault_word;
    c->base       = 0;
    c->heap_host  = 0;
    c->heap_card  = 0;
    c->heap_nodes = 0ull;
}
/* The heap as the worker's card sees it, once the boot has registered it there. ▶ the context's fields. */
static inline void sys__engine__ctx_heap(unsigned int worker, sys__heap_node* heap_host, sys__heap_node* heap_card, uint64_t heap_nodes) {
    if (worker >= SYS__SILICON__WORKERS_MAX) return;
    sys__engine__ctx* c = &sys__engine__zzprivate_workers[worker];
    c->heap_host  = heap_host;
    c->heap_card  = heap_card;
    c->heap_nodes = heap_card != 0 ? heap_nodes : 0ull;
}
#else
extern sys__engine__ctx* sys__engine__ctx_current(void);
#endif
/* Worker `w`'s context, or null for a worker there is not — what a verb reads to reach another worker's device.
 * A harness brings one context: worker 0's, and no other. */
static inline sys__engine__ctx* sys__engine__ctx_of(unsigned int worker) {
#ifndef SYS__ENGINE__ABI__CTX_PROVIDED
    return worker < SYS__SILICON__WORKERS_MAX ? &sys__engine__zzprivate_workers[worker] : 0;
#else
    return worker == 0u ? sys__engine__ctx_current() : 0;
#endif
}
/* ══ ⭐⭐⭐ WHO OWNS THE RETURNED NODE — THE RULE, AND THE DEFECT THAT MADE IT NECESSARY ═══════════════
 *
 * ⚖ THE SIGNATURE WAS RULED; THIS IS ITS UNSTATED HALF, AND LEAVING IT UNSTATED COST A LEAK.
 *
 *     ⭐ A VERB RETURNS A NODE CARRYING A HOLD THAT THE VERB OWNS AND HANDS OVER.
 *       The bridge publishes it with `becomes` — which takes the FORM's own hold — and then RELEASES
 *       the verb's. A verb answering something it did not make must therefore RETAIN it first.
 *
 * ⛔⛔ WITHOUT THIS RULE THE TWO KINDS OF ANSWER DISAGREE ABOUT WHO HOLDS WHAT:
 * ```
 *   sys__engine__abi__error(CODE)   sys__error__make -> hold 1, OWNED      ← wants a release
 *   return argv[2];         an operand out of the form, BORROWED   ← must not be released
 * ```
 * A bridge cannot serve both, so the rule picks one and the borrowed case pays a `retain`.
 *
 * ⭐⭐ `MEASURED`, AND IT WAS MEASURED RATHER THAN READ — the A/B is
 * `test_abi_error_holds_balance()`, which refuses down the classic road and the ruled one and compares,
 * because the ABSOLUTE count needs the whole convention to interpret and the DIFFERENCE does not:
 * ```
 *                          classic (sys__begin)   ruled, no release
 *   hold, form alive              1                      2      ⛔ one orphan
 *   hold, form released           0                      1      ⛔ never freed
 *   deallocations over 50       500                    450      ⛔ exactly one short per refusal
 * ```
 * ⇒ ★★ A LEAK IN AN ERROR PATH IS INVISIBLE TO EVERY CORRECTNESS CHECK THERE IS, because the
 *   answer is right — the program refuses, with the right code, and the value it becomes is the right
 *   value. Only the counter that names the event, *"an object's last hold went"*, can see it.
 * ⚳ AND THE ORDER IN THE BRIDGE IS LOAD-BEARING: `becomes` must take its hold BEFORE this release,
 *   or a verb answering one of its own operands frees the thing the form has just been pointed at.
 *   `becomes` is already written that way on purpose — ▶ its comment about "the order is the
 *   correctness, not the saving".
 */

static __device__ inline sys__heap_node sys__engine__abi__error(uint64_t code) {
    const uint64_t at = sys__error__make(code);
    if (at == 0ull) return sys__heap_node__nothing();
    return sys__heap_object__reference_to(at);
}

/* ══ THE BRIDGE ══════════════════════════════════════════════════════════════════════════════════════
 * ⛔ THE RESOLUTION LOOP IS THE BRIDGE'S COST: the operands are lifted out of the form, one `nth` each.
 * ⛔ AND THE ARITY IS CHECKED TWICE ON PURPOSE — THE TWO CHECKS ARE NOT THE SAME CHECK. This one
 * guards the BRIDGE's array; the verb's guards the VERB's contract. Without this one a malformed form
 * is a stack overrun here rather than an error value from there. */
#define SYS__ENGINE__ABI__BRIDGE(ADAPTER_NAME, VERB_FN)                                                      \
    static __device__ __noinline__ void ADAPTER_NAME(sys__heap_node* base, uint64_t form) {          \
        const uint64_t len = sys__sublist__length(form);                                             \
        if (len == 0ull || len - 1ull > (uint64_t)SYS__ENGINE__ABI__MAX_ARGV) {                              \
            sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY);                                    \
            return;                                                                                  \
        }                                                                                            \
        sys__heap_node argv[SYS__ENGINE__ABI__MAX_ARGV];                                                     \
        const unsigned argc = (unsigned)(len - 1ull);                                                \
        for (unsigned i = 0; i < argc; ++i) argv[i] = sys__sublist__nth(form, (uint64_t)(i + 1u));   \
        /* ⛳ A COPY, NOT THE SHARED ONE. The base belongs to THIS dispatch and the context every       \
         * other field describes belongs to the evaluator, so the call gets a per-call view rather     \
         * than the bridge writing into state its caller also reads. Four words on the stack.          \
         * ⛔ AND IT IS WHERE `base` ENTERS THE RULED SHAPE AT ALL: the verb signature does not carry   \
         * it, so without this line `sys__clone` and `sys__create` have no arena to carve from. */     \
        sys__engine__ctx zzprivate_ctx = *sys__engine__ctx_current();                                                  \
        zzprivate_ctx.base = base;                                                                     \
        const sys__heap_node answer = VERB_FN(argv, argc, &zzprivate_ctx);                             \
        /* ⛳ THE PUBLISH IS UNCONDITIONAL, BECAUSE AN ERROR IS A VALUE. No arm here inspects the kind \
         * and takes a different road — there is nothing to correlate: what                             \
         * the program BECAME may be an exception. ▶ `cpu/error.cuh`. */                              \
        sys__opcodes__becomes(form, &answer);                                                        \
        /* ⭐⭐⭐ AND THEN THE VERB'S OWN HOLD GOES — THE SECOND HALF OF THE OWNERSHIP RULE ABOVE.      \
         * `becomes` took the FORM's hold; this gives up the VERB's. Without it every refusal orphans \
         * one error object forever. ▶ the rule, and the measurement that found it, above. */         \
        if (sys__heap_node__carries_reference(answer.dtype))                                         \
            (void)sys__heap_object__release(answer.args[0]);                                         \
    }

#endif /* SILVANN__PACKAGES_SYS_CPU_OPCODES_VERB_ABI_CUH */
