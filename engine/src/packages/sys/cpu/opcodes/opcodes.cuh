#ifndef SILVANN__PACKAGES_SYS_CPU_OPCODES_OPCODES_CUH
#define SILVANN__PACKAGES_SYS_CPU_OPCODES_OPCODES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../heap_object__header.cuh"   /* holds, and what a kind answers when asked */
#include "../heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../error.cuh"                /* what a verb answers with when it refuses */
#include "../bindings__header.cuh"     /* the scope verbs are one line over an environment */
#include "../list__header.cuh"         /* every one of them is handed a form and rewrites it */
#include "../../contracts/objects/opcodes.cuh" /* its constants and fault words */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 *   · The one-read PoC's note cited *"`MEASURED`: a cell costs 8.26 FLAT accesses and a form
 *     450.3"* (`scripts/src_apply_width_census.py`). That is a rocprof count on `eng__abi__k_eval`; the
 *     evaluator has left the card and the figure does not transfer.
 *   · `sys__opcodes__fails` was `zzpackage_` when `nn`'s `getnew` became the first
 *     verb outside `sys` and the `c_subset` gate flagged the call. The float bit move lived here as a
 *     `zzprivate_` pair before it went to `heap_node__header.cuh`.
 * ⛳ RETIREMENT: at release, with the rest.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ the opcodes a first program needs ═══════════════════════════════════════════════════════════════
 *
 * The verbs here and in `opcodes_abi.cuh` are a whole small language between them: names can be given
 * values, taken away and changed, two things can be compared, two numbers added, and a program can
 * choose what to do next. Everything else a lisp has is built out of those. This file holds the helpers
 * every verb finishes with and the evaluator's own control flow — `begin`, `prog1`, `remove_bindings`;
 * the verbs that take their operands as `argv` are in `opcodes_abi.cuh`.
 *
 * ── WHAT AN OPCODE IS HANDED, AND WHAT IT ANSWERS ───────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"pass the form not the cdr … i think it leaves it in place and the opcode decides if it
 * is correct to remove it, so you can have tail call recursion embedded if you want it and you can have
 * recursion proper in the other direction, which is the only way to have a functioning stack in a flat
 * world."*
 *
 * A verb receives the WHOLE FORM — its own name at position 0, its arguments after it — by reference, so
 * what it puts in or takes out is in the list the evaluator is holding. ⭐ **AND THE HEAD STAYING IS THE
 * LOAD-BEARING HALF:** a verb that could not re-emit its own name could not write itself back into the
 * list, and a flat evaluator would have tail recursion in one direction only.
 *
 * ⭐⭐ WHAT IT ANSWERS IS THE FORM, NOT A VALUE. ⚖ *"we either return null constantly and use the list as
 * a & return value or we make the return channel a list … the first saves us an extra round of
 * allocations per opcode."* So each of these rewrites the form and the evaluator reads what is left:
 *     ONE VALUE       that is the answer. The evaluator writes it where the form was and goes on.
 *     ANYTHING ELSE   it is a form. The evaluator keeps scanning it.
 * ⛳ Which is why `if` costs nothing: it does not evaluate a branch, it leaves the branch where the form
 * was and lets the scan meet it — a rewrite, not a call, so nothing recurses and no frame is spent.
 *
 * ── HOW A VERB REFUSES ──────────────────────────────────────────────────────────────────────────────
 * By leaving an ERROR where its value would have gone. There is no second channel and no return code: a
 * caller reads what the form became, and what it became may be an exception. ⛳ The raise still goes to
 * the log, because that is how somebody watching a run finds out at all.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── WHAT EVERY ONE OF THEM DOES TO FINISH ───────────────────────────────────────────────────────────
 * Leave the form holding exactly one thing. Two helpers rather than a copy of the same three lines in
 * every verb, because the three lines are where an off-by-one would be invisible.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* Empty the form and put one value in it. That value is what the form BECAME, so the evaluator writes it
 * where the form was standing. Dropping the arguments afterwards is what releases whatever they held.
 * ⛳ IT IS PUBLIC, because saying what a form became is not something one file's verbs do alone — the
 * bindings, the computing base, the heap object interface and the system registers all finish this way,
 * and a verb in any package would. One name for one act. */
static __device__ inline void sys__opcodes__becomes(uint64_t form, const sys__heap_node* value) {
    const uint64_t was = sys__sublist__length(form);
    if (was == 0ull) { (void)sys__sublist__append(form, value); return; }
    /* ⭐⭐ THE ANSWER GOES WHERE THE VERB WAS, AND THE ARGUMENTS GO AFTERWARDS. A form always has at
     * least the verb in it, so there is always a cell to write into — which means no room has to be
     * found for the answer, and a verb answering therefore cannot fail for want of memory.
     * ⭐ AND THE ORDER IS THE CORRECTNESS, NOT THE SAVING. Replacing takes a hold on what arrives
     * BEFORE letting go of what was there, so an answer that IS one of the arguments — a verb handing
     * back what it was given — is held by the form before the discard reaches the cell it came from.
     * Emptying first and putting the value in afterwards has the opposite order, and every caller
     * passing a value out of its own form had to know that. */
    (void)sys__sublist__replace(form, 0ull, value);
    if (was > 1ull) (void)sys__sublist__discard(form, 1ull, was - 1ull);
}

/* The same, with an error. It raises as well, because the log and the caller are different readers and
 * either one alone leaves half the story with the wrong one.
 *
 * ⭐⭐ IT IS PUBLIC BECAUSE EVERY PACKAGE THAT PUBLISHES A VERB NEEDS IT: a verb that cannot refuse is
 * not a verb, and `nn`'s verbs refuse through this. ⇒ ★ A BOUNDARY IS A CLAIM ABOUT WHO EXISTS.
 * ⛳ It is the twin of `becomes`, public for exactly this reason — the two are how ANY verb answers, and
 * they belong on the same side of the line. The `c_subset` gate is what holds a package outside `sys`
 * to that line. */
static __device__ inline void sys__opcodes__fails(uint64_t form, uint64_t fault) {
    const uint64_t at = sys__error__make(fault);
    if (at == 0ull) return;                              /* the making already raised */
    const sys__heap_node value = sys__heap_object__reference_to(at);
    sys__opcodes__becomes(form, &value);
    (void)sys__heap_object__release(at);                  /* the form holds it now */
}

/* True and false as values. ⭐ There is no payload on either:
 * an answer's identity is its KIND, so a reader tests the tag and never the bits.
 * ⛔ THIS is the one producer. `if` reads both and REFUSES anything else, so there is no third outcome to
 *   argue about and no "not false" standing in for true — a non-boolean test would otherwise take the TRUE
 *   arm silently, which in a loop is a hang. The claim gate's `strict_truth` holds it. */
static __device__ inline sys__heap_node sys__opcodes__truth(bool yes) {
    sys__heap_node v;
    v.dtype    = yes ? SYS__KIND__VALUE_TRUE : SYS__KIND__VALUE_FALSE;
    v.num_args = 0u;
    v.args[0]  = 0ull;
    return v;
}

/* ── THE VERBS ───────────────────────────────────────────────────────────────────────────────────────
 * Each takes the form and leaves it holding what the form became. `base` is the block's computing base,
 * which none of these needs and all of them are handed, because a published verb has one shape.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* `(remove_bindings env 'a 'b 'c)` — leave the innermost scope for each of these names, and it takes as
 * many as you like because a scope ending is almost never one name on its own.
 * ⭐ A CALL'S UNBINDING IS THE CASE THAT DECIDED IT: a procedure of four parameters ends four scopes at
 * once, in ONE list and ONE cycle of the loop, where a form per name would be four of each to call this
 * one function four times. ⚖ ARCHITECT: *"can we do a (remove_bindings (a b c d)) so we only consume one
 * list."*
 * ⛳ AND IT IS WHY THEY DO NOT NEED CHAINING TOGETHER AFTERWARDS: the adjacent unbindings that would have
 * been worth merging are the ones a single scope produces, and this way they are never separate.
 * ⛔ THE NAMES ARE LEFT IN REVERSE, WHICH MATTERS FOR EXACTLY ONE SHAPE AND COSTS NOTHING OTHERWISE. Each
 * name owns its own stack, so for distinct names the order is unobservable — but two scopes entered for
 * the SAME name have to be left innermost-first or the wrong one survives. Mirroring the entry order is
 * correct without having to know whether the caller repeated itself. */
static __device__ __noinline__ void sys__opcodes__zzpackage_apply_remove_bindings(sys__heap_node* base, uint64_t form) {
    (void)base;
    const uint64_t n = sys__sublist__length(form);
    if (n < 3ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const uint64_t env = sys__sublist__nth(form, 1ull).args[0];
    bool ok = true;
    for (uint64_t i = n; i > 2ull; --i) {
        const sys__heap_node name = sys__sublist__nth(form, i - 1ull);
        if (!sys__bindings__remove(env, &name)) ok = false;
    }
    const sys__heap_node said = sys__opcodes__truth(ok);
    sys__opcodes__becomes(form, &said);
}

/* ══ ⭐⭐⭐ REAL NUMBERS — THE BITS, AND THE TWO DOMAINS ══════════════════════════════════════════════
 *
 * ⚖ RULED: a `sys__value_float` is an fp64 held in `args[0]` AS ITS BITS. The pair below is
 * a REINTERPRETATION and not a conversion — nothing rounds, so a float that crosses the C interface and
 * comes back is the same number, bit for bit.
 * ⛳ WRITTEN WITH A UNION AND NOT THROUGH THE SILICON SEAM, for the reason `turboquant__header.cuh` gives about
 * the half-to-float read: the seam carries what DIFFERS between backends, and `expf` differs while a
 * reinterpretation cannot. ⇒ ★ AN EXACT BIT MOVE HAS NO BACKEND.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ⛳ THE BIT MOVE ITSELF LIVES WITH THE NODE — `sys__heap_node__real` / `__real_bits` — because `nn`
 * takes float arguments too and a `zzprivate_` pair here could not reach it. ▶ that file. */

/* ⭐⭐ WIDENING THE SECOND OPERAND, WHICH IS THE WHOLE OF *"a float adding an int still works"*.
 * ⛔ AND IT IS NOT LOSSLESS FOR EVERY `uint64_t`, WHICH IS WORTH SAYING RATHER THAN DISCOVERING: fp64
 * carries 53 bits of mantissa, so an integer above 2^53 (9,007,199,254,740,992) rounds on the way in.
 * That is a real bound and not a hypothetical one — a heap ADDRESS is a 64-bit number and would lose
 * its low bits here. ⇒ ★ "WIDENING IS SAFE" IS TRUE OF THE RANGE PEOPLE MEAN AND FALSE OF THE TYPE.
 * ⛳ It is not refused, because a count or a layer index is nowhere near the bound and refusing would
 * make the ordinary case pay for the exotic one; it is DOCUMENTED, which is what the caller needs. */
static __device__ inline bool sys__opcodes__zzpackage_as_real(const sys__heap_node* n, double* out) {
    if (n->dtype == SYS__KIND__VALUE_FLOAT) { *out = sys__heap_node__real(n->args[0]); return true; }
    if (n->dtype == SYS__KIND__VALUE_INT)   { *out = (double)(int64_t)n->args[0]; return true; }
    return false;
}

/* ══ ⭐⭐ THE ONE-READ OPERAND FETCH — A PROOF OF CONCEPT, ARMED BY `SILVANN_ONE_READ=1` ══════════════
 *
 * ⚖ *"do try a poc about changing a opcode to c parameter assignment so you do
 * only one read, especially if the 99% of the loads get absorbed in the lds. maybe it does not make
 * much difference but if you have some time to spare we can see what kind of annoyances it
 * introduces."* And, earlier: *"even the way we cdr we should probably turn this into a single read
 * from the cdr's pointer, instead of n reads."*
 *
 * ── WHAT A 3-CELL VERB PAYS TODAY, AND WHY IT IS FIVE TRAVERSALS OF ONE LIST ────────────────────────
 * ```
 *   SYS__ENGINE__ABI__BRIDGE                 sys__sublist__length    fields + total
 *                                    sys__sublist__nth(1)    fields + total + locate-from-head + cell
 *                                    sys__sublist__nth(2)    fields + total + locate-from-head + cell
 *   sys__opcodes__becomes            sys__sublist__length    fields + total        (again)
 *                                    sys__sublist__replace   fields + total + locate-from-head + set
 * ```
 * ⭐ EVERY ONE OF THOSE LOCATES STARTS AT THE STORE'S FIRST CHUNK, and cells 0, 1 and 2 of a form are
 * ADJACENT IN ONE CHUNK in every case but a straddle. ⇒ **ONE locate can serve the arity check, both
 * operands and the answer's destination**, because they are the same four nodes.
 *
 * A cell is the PROGRAM's data and ought to cost about one access. On the host that is `UNMEASURED`.
 *
 * ⛔ WHAT IT DOES NOT TOUCH, STATED SO THE RESULT IS NOT OVER-READ: the `discard` of the tail still
 * walks, and the per-form machine cost is mostly elsewhere — descend, dispatch, refcount, pop. This is
 * an experiment about the CELLS and the locates, not about that constant.
 *
 * ⚠ THE ARITY TEST READS `w.total - w.at`, AND THAT IS A REAL DIFFERENCE FROM `length`, NOT A
 * REFACTOR: two words the walk's own open already fetched. Same answer, no read —
 * but it is the store's total rather than a fresh one, so a form mutated between the open and here
 * would be judged on the older value. Nothing mutates a form between those two lines today; that is
 * a premise, and it is the premise a reader should check first if this is ever generalised. */
static __device__ inline bool sys__opcodes__zzprivate_operands3(uint64_t form,
                                                                sys__heap_node** slot0,
                                                                sys__heap_node* a,
                                                                sys__heap_node* b) {
    sys__list_walk w;
    if (!sys__sublist__walk(form, 0ull, &w)) return false;
    if (w.total < w.at || (w.total - w.at) != 3ull) return false;     /* arity, from words already read */
    sys__heap_node* c0 = sys__list__walk_cell(&w);
    if (c0 == 0) return false;
    /* ⭐ THE WHOLE POINT: when all three sit in the chunk the walk already found, the other two are
     * `c0[1]` and `c0[2]` — no second locate, no second guard read, no chunk hop. */
    if ((w.at - w.base) + 3ull <= w.used) {
        *slot0 = c0; *a = c0[1]; *b = c0[2];
        return true;
    }
    /* A form split across two chunks. Stepping is what the cursor is for, and it is still ONE walk. */
    sys__list__walk_step(&w);
    const sys__heap_node* c1 = sys__list__walk_cell(&w);
    if (c1 == 0) return false;
    sys__list__walk_step(&w);
    const sys__heap_node* c2 = sys__list__walk_cell(&w);
    if (c2 == 0) return false;
    *slot0 = c0; *a = *c1; *b = *c2;
    return true;
}

/* ⛳ THE SHIPPING DEFAULT, OUTSIDE THE MARKER ON PURPOSE — the house rule for levers: *"THE ARMING IS
 * INTERNAL, THE DEFAULT IS SHIPPING. A lever is an override, and deleting an override leaves nothing
 * where a value is required."* Stripping the segment below leaves this `0` and the verb it selects. */
#ifndef SILVANN_ONE_READ
#define SILVANN_ONE_READ 0
#endif


/* `(begin a b c)` — do these in order and become the last one. It is what a program IS: things in a row
 * need a head or there is nothing to apply.
 * ⛳ IT NEEDS NO SEQUENCING CODE OF ITS OWN, which is the point of the flat scan — by the time this runs,
 * every one of its arguments has already been reduced, in order, because that is how the scan found them.
 * All that is left is to say which of the answers is the form's. */
static __device__ __noinline__ void sys__opcodes__zzpackage_apply_begin(sys__heap_node* base, uint64_t form) {
    (void)base;
    const uint64_t n = sys__sublist__length(form);
    if (n < 2ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const sys__heap_node last = sys__sublist__nth(form, n - 1ull);
    if (sys__heap_node__carries_reference(last.dtype)) (void)sys__heap_object__retain(last.args[0]);
    sys__opcodes__becomes(form, &last);
    if (sys__heap_node__carries_reference(last.dtype)) (void)sys__heap_object__release(last.args[0]);
}

/* `(while 'cond 'action)` — run the action for as long as the test answers `#t`, in ONE form, whatever the count.
 * ⚖ ARCHITECT: *"the while can also recycle the same execution list when it is true … (while cond action) then
 * evaluates cond and if false it returns and if true it writes action cond in the execution."*
 * ⭐ THE VERB IS HANDED ITS OWN FORM AND LEAVES IT BIGGER, WITH ITS OWN NAME AT THE HEAD, so the scan runs the
 * new cells and comes back to it — and the scan's stack does not grow by a word. What the form holds says
 * where the loop is:
 * ```
 *   (while 'c 'a)          first entry: c and a become PICTURES, and c is laid on the end to run
 *   (while 'c 'a T)        T is the test's answer
 *   (while 'c 'a V T)      V is the action's answer, T the test's that followed it
 *   T = #f                 the form becomes #f — the loop is over
 *   T = #t                 the last cells become a, then c: the action runs first, the test after it
 * ```
 * ⛔⛔ A PICTURE, BECAUSE RUNNING A FORM CONSUMES IT. The scan rewrites what it runs, so a turn runs a list thawed
 * from the picture, never the picture. And the cells laid down are FROZEN, so the scan makes each list when it
 * reaches it — out of the list the previous turn just finished with, when the pool has one.
 * ⛔ THE TEST MUST BE `#t` OR `#f`, as `if`'s must, and anything else is refused. An error from either the test or
 * the action is the loop's answer: it stops there rather than going round again. */
static __device__ inline bool sys__opcodes__zzprivate_while_error(const sys__heap_node* v) {
    return v->dtype == SYS__KIND__OBJECT_REFERENCE && sys__error__is(v->args[0]);
}
static __device__ __noinline__ void sys__opcodes__zzpackage_apply_while(sys__heap_node* base, uint64_t form) {
    (void)base;
    const uint64_t n = sys__sublist__length(form);
    if (n == 3ull) {
        uint64_t pictures[2] = {0ull, 0ull};
        for (uint64_t i = 0ull; i < 2ull; ++i) {
            const sys__heap_node arg = sys__sublist__nth(form, i + 1ull);
            if (arg.dtype == SYS__KIND__QUOTED_LIST) {
                pictures[i] = sys__sublist__deep_copy_to_node_array(arg.args[0]);
            } else if (arg.dtype == SYS__KIND__QUOTED_ARRAY) {
                pictures[i] = arg.args[0];
                (void)sys__heap_object__retain(pictures[i]);
            }
            if (pictures[i] == 0ull) {
                if (i == 1ull) (void)sys__heap_object__release(pictures[0]);
                sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
                return;
            }
        }
        for (uint64_t i = 0ull; i < 2ull; ++i) {
            sys__heap_node pic;
            pic.dtype = SYS__KIND__QUOTED_ARRAY; pic.num_args = 0u; pic.op_code = 0ull; pic.args[0] = pictures[i];
            (void)sys__sublist__replace(form, i + 1ull, &pic);
            (void)sys__heap_object__release(pictures[i]);     /* the cell holds it now */
        }
        sys__heap_node test;
        test.dtype = SYS__KIND__FROZEN_LIST; test.num_args = 0u; test.op_code = 0ull; test.args[0] = pictures[0];
        if (!sys__sublist__append(form, &test)) sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY);
        return;
    }
    if (n != 4ull && n != 5ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    /* The loop ends on an error from the action, an error from the test, or a test of `#f` — and the form
     * becomes that value, held across the write as `begin` holds its answer. */
    sys__heap_node last = sys__sublist__nth(form, n - 1ull);
    if (n == 5ull) {
        const sys__heap_node done = sys__sublist__nth(form, 3ull);
        if (sys__opcodes__zzprivate_while_error(&done)) last = done;
    }
    const bool error = sys__opcodes__zzprivate_while_error(&last);
    if (!error && last.dtype != SYS__KIND__VALUE_TRUE && last.dtype != SYS__KIND__VALUE_FALSE) {
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
        return;
    }
    if (!error && last.dtype == SYS__KIND__VALUE_TRUE) {
        sys__heap_node action = sys__sublist__nth(form, 2ull);
        sys__heap_node test   = sys__sublist__nth(form, 1ull);
        action.dtype = SYS__KIND__FROZEN_LIST;
        test.dtype   = SYS__KIND__FROZEN_LIST;
        (void)sys__sublist__replace(form, 3ull, &action);
        const bool ok = (n == 4ull) ? sys__sublist__append(form, &test) : sys__sublist__replace(form, 4ull, &test);
        if (!ok) sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY);
        return;
    }
    if (sys__heap_node__carries_reference(last.dtype)) (void)sys__heap_object__retain(last.args[0]);
    sys__opcodes__becomes(form, &last);
    if (sys__heap_node__carries_reference(last.dtype)) (void)sys__heap_object__release(last.args[0]);
}

/* `(prog1 a b c)` — do these in order and become the FIRST. It is `begin`'s mirror, and it exists for one
 * shape: a value that has to survive the tidying up after it.
 * ⭐ WHICH IS WHAT A CALL IS. Expanding one binds the arguments where they stand and lays down the body
 * and the unbinding in a row — and `begin` would answer the UNBINDING, throwing away the very thing the
 * call was for. `prog1` keeps the body's answer while everything after it still runs.
 * ⛳ IT IS A PRIMITIVE AND NOT A COMPOSITE, which is the point: nothing here knows what a call is. It says
 * "this one is the answer" and the rest is a program the composer wrote. */
static __device__ __noinline__ void sys__opcodes__zzpackage_apply_prog1(sys__heap_node* base, uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) < 2ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const sys__heap_node first = sys__sublist__nth(form, 1ull);
    if (sys__heap_node__carries_reference(first.dtype)) (void)sys__heap_object__retain(first.args[0]);
    sys__opcodes__becomes(form, &first);
    if (sys__heap_node__carries_reference(first.dtype)) (void)sys__heap_object__release(first.args[0]);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_OPCODES_OPCODES_CUH */
