#ifndef SILVANN__ENGINE_EVAL_CUH
#define SILVANN__ENGINE_EVAL_CUH
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * The rates this file's decisions were taken on were measured on the card evaluator (`k_eval`), and none
 * has been re-run on the host runners. Kept as what was tried, not as what this build does:
 *   · the list pool with the lazy thaw: 1.48x off the whole dispatch at every granularity, the lazy half
 *     alone cost-neutral (commit 15018c7e has the table).
 *   · a FROZEN_LIST tag rather than an `(unthaw arr)` verb: the verb would be a whole apply, ~420 memory
 *     accesses, to save the ~240 a form's birth and death cost.
 *   · the narrow tag read in `resolve_bindings`: the sweep cost 29.0 accesses per apply on a form with no
 *     names, the narrow read recovered 3.8 of them (13%, 0.9% of the rate), because LLVM already
 *     dead-coded seven of the eight `flat_load_dwordx2` a whole-node read looks like.
 *   · the stack of places over recursion: on gfx906 a cycle in the call graph cost 108 registers.
 * ⛳ RETIREMENT: when the pool and the lazy thaw are re-measured on the host runners, or at release.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* What this file needs: the package contract. Every name the evaluator calls is a `sys__` one, and they
 * reach it through `engine/manifest.cuh`, which includes the packages before any engine file; the
 * evaluator names no other engine file. */
#include "../packages/manifest__header.cuh"

/* ══ the evaluator — a flat scan, and a stack of places rather than a stack of calls ═════════════════
 *
 * ⚖ ARCHITECT: *"if the eager evaluator scans the executing list for sublists and always takes the first
 * one it finds and processes it … it just needs to hold a variable where to return the value, and that
 * variable needs to be an array as push/pop with the base of each list."*
 *
 * ```
 *   scan the list left to right
 *   the first FORM it meets        -> descend, remembering where that form's answer goes
 *   a list of nothing but values   -> apply its head, and read what the form became
 * ```
 *
 * ⭐⭐ AND THE STACK IS OF DATA, NOT OF CALLS. Descending is a push and finishing is a pop; nothing here
 * calls itself. A recursive evaluator is a cycle in the call graph and spends a machine frame per level
 * of nesting, so a deep program exhausts the runner's own stack; a stack of places lives on the heap,
 * grows by chunks, and costs one entry per level (`REASONED`). The claim gate's
 * `list_call_graph_acyclic` rule is what keeps the list machinery free of cycles.
 * ⛳ SO IT IS `sys__stack`, ALREADY BUILT: edge access only, which is this package's own rule for
 * choosing one. THE ADT IS FREE AND THE SURFACE IS NOT — the descent uses two verbs of its own,
 * `sys__stack__zzengine_compute_stack_push` and `_pop`, under a titled section of `stack__impl.cuh`
 * written for this one caller: NO LOCK, because nobody but this call of the evaluator can name
 * `places`, and NO HOLD, because the chain of parent cells holds every entry already.
 * ⛔ AND THE ORDINARY `sys__stack__push`/`sys__stack__pop` ARE WRONG HERE RATHER THAN MERELY SLOWER: a
 * push is a transfer PLUS a retain, and the drain at the bottom of this file pops without releasing —
 * so every descent still on the stack when the loop breaks would leak one hold.
 *
 * ── WHY NOTHING NEEDS TO BE MARKED "DO NOT EVALUATE ME" ─────────────────────────────────────────────
 * The scan descends into FORMS. A quoted list is a VALUE, so both arms of an `if` sit untouched while its
 * test is reduced, and a quoted name is a value too, so `set!` is handed a symbol rather than what that
 * symbol currently means. ⇒ **One mechanism — quoting — covers conditional evaluation and assignment,
 * and no verb has to be special.**
 *
 * ── WHAT HAPPENS AFTER A VERB RUNS ──────────────────────────────────────────────────────────────────
 * A verb rewrites the form it was handed, and what the form holds afterwards says what happens next:
 * ```
 *   ONE VALUE       that is the answer. Write it where the form was standing, and pop.
 *   ANYTHING ELSE   it is a form. Keep scanning — nothing was returned and nothing recursed.
 * ```
 * ⭐ WHICH IS WHERE TAIL RECURSION COMES FROM, FOR FREE: a verb that leaves a bigger form with its own
 * name back at the head is simply scanned again, and the stack does not grow by one word.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⛳ THE SHIPPING DEFAULT, OUTSIDE ANY INTERNAL MARKER — this tree's house rule: a lever is an
 * override, and deleting an override must leave the value the engine ships with. ▶ the ablation's own
 * header at its call site below. */
#ifndef SILVANN_ABLATE_RESOLVE
#define SILVANN_ABLATE_RESOLVE 0
#endif

/* ⭐⭐ BOTH ON BY DEFAULT, AND THESE TWO ARE ONE MECHANISM. ⚖ *"make the lazy thaw and pool the
 * default."* `MEASURED` together: deallocations per apply 7.13 -> 1.06, chunk claims 0.15 -> 0.04 —
 * counts of what the evaluator does, which do not depend on where it runs.
 * ⛔ `SILVANN_LIST_POOL` REQUIRES `SILVANN_LAZY_THAW` and the `#error` below says so: with an eager
 * thaw every list is made before the first one runs, so nothing is free at a take and nothing spare at
 * a put — it is the recycling that pays, and the lazy half alone only makes it possible.
 * ⛳ THE VALUE IS THE SHIPPING ONE AND THE OVERRIDE TURNS IT OFF, which is this tree's rule the right
 * way round: `SILVANN_LAZY_THAW=0` in the environment restores the eager evaluator for an A/B, and
 * deleting the override leaves the engine with what it ships. */
#ifndef SILVANN_LAZY_THAW
#define SILVANN_LAZY_THAW 1
#endif
#ifndef SILVANN_LIST_POOL
#define SILVANN_LIST_POOL 1
#endif
#if SILVANN_LIST_POOL && !SILVANN_LAZY_THAW
#error "SILVANN_LIST_POOL needs SILVANN_LAZY_THAW: with an eager thaw there is nothing to recycle into."
#endif

#if SILVANN_LIST_POOL
/* ══ ⭐⭐ THE LIST POOL — ⚖ THE ARCHITECT'S DESIGN ═══════════════════════════════════════════════════
 *
 * ⚖ *"maybe we could have two or three pre-allocated lists so most of the stacks never allocate, and
 * when a computation completes the engine that holds the list reference can decide if it wants to
 * dispose it or feed back one of the three levels if they are orphans … maybe let's have 4 references
 * not 3, and make them into a single heap node.. maybe even 6 since that is the amount of args it
 * holds."*
 *
 * ⭐⭐ WHY SIX SLOTS IS ENOUGH, AND IT IS NOT A GUESS ABOUT PROGRAMS. The evaluator descends and pops,
 * so a form's list is born on the way down and dies on the way up: strict LIFO. A free list that is
 * only ever asked for one at a time therefore needs as many slots as the DEPTH the program reaches,
 * not as many as it has forms. ⇒ `take` is the lowest occupied slot and `put` the lowest free one,
 * which is a bounded stack written flat.
 * ⛔ AND THE LAZINESS IS WHAT MAKES THAT TRUE. An eager thaw builds every list before the first one
 * runs, so nothing is free when the next is wanted — the pool would be empty at every take and full at
 * no put. ⇒ ★ THE TWO ARE ONE MECHANISM; measured apart, the lazy half alone is cost-neutral.
 *
 * ⛳ IT IS `BLOCK_LOCAL` RATHER THAN A HEAP NODE, WHICH IS A DEVIATION FROM THE SKETCH AND CHEAPER.
 * A heap node would be one more object to make, hold and tear down — the very cost this removes — and
 * it would have to be found through a pointer on every take. `BLOCK_LOCAL` is one copy per runner
 * (`static __thread` on the host), which is the scope wanted when each runner evaluates its own program,
 * and it costs no allocation at all.
 * ⛔ IT IS EMPTIED ON EVERY CALL, so the drain at the bottom of `eng__eval` is not tidiness: a list left
 * in a slot when the loop ends is leaked, and nothing else holds it.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
#ifndef SILVANN_LIST_POOL_SLOTS
#define SILVANN_LIST_POOL_SLOTS 6
#endif

/* The lowest occupied slot, emptied — or 0 when every slot is. */
static __device__ inline uint64_t eng__eval__zzprivate_pool_take(uint64_t* pool) {
    for (unsigned int i = 0u; i < (unsigned int)SILVANN_LIST_POOL_SLOTS; ++i) {
        if (pool[i] != 0ull) { const uint64_t got = pool[i]; pool[i] = 0ull; return got; }
    }
    return 0ull;
}

/* Into the lowest free slot. ⛔ ANSWERS FALSE WHEN THERE IS NO ROOM, and the caller must then release
 * what it was handing over — this takes ownership only when it answers true. */
static __device__ inline bool eng__eval__zzprivate_pool_put(uint64_t* pool, uint64_t list) {
    for (unsigned int i = 0u; i < (unsigned int)SILVANN_LIST_POOL_SLOTS; ++i) {
        if (pool[i] == 0ull) { pool[i] = list; return true; }
    }
    return false;
}
#endif

#define ENG__EVAL__FAULT_NOT_A_VERB 0x45564156ull   /* "EVAV" — a form whose head is not a verb        */
#define ENG__EVAL__FAULT_DEPTH      0x45564144ull   /* "EVAD" — a descent the stack could not record  */
/* ⛔ A DESCENT THAT CANNOT BE REMEMBERED STOPS THE EVALUATOR. With no place pushed, the pop on the way
 * back would return to the wrong form, or find the stack empty; so the loop raises `EVAD` and ends
 * instead, after the stack itself has raised why it could not grow. */

/* Resolve every name in a form to what it means now. A QUOTED name is left exactly as it is, which is the
 * whole of why `set!` needs no special treatment. */
static __device__ inline void eng__eval__zzprivate_resolve_bindings(uint64_t bindings, uint64_t form) {
    const uint64_t n = sys__sublist__length(form);
    /* ⭐⭐ ONE WALK, AND IT IS WORTH TWICE WHAT IT LOOKS LIKE: finding a cell with `nth` to read it and
     * again inside `replace` to write it back would locate each cell TWICE. A walk hands back the SLOT,
     * so the read and the write are the same address found once. ▶ `sys__sublist__walk`. */
    sys__list_walk w;
    if (!sys__sublist__walk(form, 0ull, &w)) return;
    for (uint64_t i = 0ull; i < n; ++i, sys__list__walk_step(&w)) {
        sys__heap_node* slot = sys__list__walk_cell(&w);
        if (slot == 0) break;
        /* ⭐ THE TAG IS READ, NOT THE NODE: a cell whose kind names nothing is passed over on one word,
         * and the 64 B cell is copied only when it does. ⚖ *"do the narrow read on resolve_bindings and
         * measure it."* It stays for being free — identical semantics on every program — and it is NOT a
         * compile lever: an `#if` arm declaring `cell` inside an internal segment would orphan its own
         * `#endif` in the strip.
         * ⛳ What is left in this loop is the walk itself — the open, and `walk_cell` re-reading the
         * store's `TOTAL` on every cell as its staleness guard. */
        if (!sys__bindings__mentioned(slot->dtype)) continue;
        const sys__heap_node cell = *slot;
        /* ⭐⭐ THE READ IS BORROWED AND THE CELL IS WHAT MAKES IT SAFE. `zzengine_get_noretain` hands
         * back what the name means without locking the scope stack, without holding it, and without
         * holding the answer; the `replace` on the next line takes the cell's own hold. Nothing happens
         * in between — no allocation, no release, no pop — which is the contiguous path the borrow
         * needs, and it is why this call site is the only one in the tree allowed to make it.
         * ⛔ MOVING ANYTHING BETWEEN THESE TWO LINES BREAKS IT SILENTLY. There is no event to
         * reconcile and no counter that would fail: the failure is a use after free.
         * ⛳ AND THE PAIR IT REMOVES IS TWO PAIRS — the scope stack's hold and the value's — plus the
         * two compare-and-swaps the ordinary peek takes around its read. */
        const sys__heap_node value = sys__bindings__zzengine_get_noretain(bindings, &cell);
        /* ⛳ THE SAME CALL `replace` MAKES, ON THE SLOT THE WALK IS ALREADY STANDING ON — what is going in
         * takes a hold before what was there gives one up, and no adjust runs because nothing moved.
         * There is no `locate` between the read and the write, which is the contiguous path the
         * paragraph above requires. */
        sys__heap_object__set(slot, &value);
    }
}

/* Whether a value names a procedure — a picture of two things, the parameters and the body. The tag is the
 * answer; nothing about the shape is asked. */
static __device__ inline bool eng__eval__zzprivate_is_procedure(sys__heap_node v) {
    return v.dtype == SYS__KIND__PROCEDURE_REFERENCE;
}

/* Where the first form sits, or the length when there is none — and the caller reads the length as "this
 * list is all values now, so apply it".
 *
 * ⭐ ONE TEST IN TWO HALVES, AND WHAT EACH HALF EXCLUDES IS THE INTERESTING PART. A cell is something to
 * descend into when it NAMES A LIST as an ordinary reference. Exactly two things fail on the KIND alone:
 *     a quoted list   is `QUOTED_LIST`, so both arms of an `if` sit untouched while its test reduces —
 *                     which is why no verb needs marking to keep the scanner out of its arguments
 *     a procedure     is named by a `PROCEDURE_REFERENCE` — that is the CELL's tag, where `PROCEDURE` is
 *                     the object's own — so a thing BUILT out of a list is not mistaken for one to run
 * ⛔ AND `sys__sublist__is` IS LOAD-BEARING, NOT A SECOND OPINION ON THE SAME QUESTION. Every OTHER object
 * a form can hold is named by an ordinary `OBJECT_REFERENCE` and passes the kind test untouched; the shape
 * question is the only thing that stops the scan. An ERROR is the live case: `sys__opcodes__fails`
 * leaves the failing form holding `sys__heap_object__reference_to` of one, which is an `OBJECT_REFERENCE`
 * like any other, and it is this second half that keeps the scan from descending into it.
 * ⛳ A PROCEDURE HAD TO BE EXCLUDED BY SHAPE UNTIL IT CARRIED A TAG, and the shape test was a real
 * exposure rather than an inelegance: any two-element list whose first element was a quoted list would
 * have been read as a procedure and silently refused entry. The tag is one word and nothing can satisfy
 * it by accident. */
static __device__ inline uint64_t eng__eval__zzprivate_first_form(uint64_t form, uint64_t from, sys__list_walk* w,
                                                                 bool standing) {
    const uint64_t n = sys__sublist__length(form);
    /* ⭐ `from` MADE THE SCAN RESUME AND THE WALK IS WHAT MAKES RESUMING FREE. The index said "everything
     * before here is already a value", but each `nth` then started at the head of the chunk chain again —
     * so the evaluator re-followed `from/SLOTS` links on every pass over a form it had already crossed.
     * ⭐⭐ AND OPENING THE WALK AT `from` WAS STILL THAT SEARCH, once a pass: a walk opens at the head and
     * steps to its index. So a caller coming back to a place hands the walk it stood up from the place,
     * already `standing` at `from`, and the walk is left on the form found — which is what the descent
     * parks. ▶ `sys__list__zzengine_walk_park`. */
    if (!standing && !sys__sublist__walk(form, from, w)) return n;
    for (uint64_t i = from; i < n; ++i, sys__list__walk_step(w)) {
        const sys__heap_node* cell = sys__list__walk_cell(w);
        if (cell == 0) break;
        if (cell->dtype == SYS__KIND__OBJECT_REFERENCE && sys__sublist__is(cell->args[0])) return i;
        /* ⭐⭐ A FROZEN SUB-FORM IS SOMETHING TO DESCEND INTO, AND IT COSTS ONE COMPARE ON A TAG THIS
         * LOOP IS ALREADY READING. ⚖ architect: *"unthaw could be made to look at a binding for the
         * list instead of instantiating one"* — as a TAG rather than a verb, so nothing is dispatched.
         * ⛔ AND A VERB WOULD HAVE BEEN A NET LOSS: an `(unthaw arr)` form is itself a form, so the
         * scan would descend into it and APPLY it — a whole apply, to save what a form's birth and
         * death cost. The tag spends none.
         * ⛳ NO `sys__sublist__is` TEST BESIDE IT, unlike the line above: what a frozen cell names is a
         * node array and never a list, so the tag alone settles it. */
#if SILVANN_LAZY_THAW
        if (cell->dtype == SYS__KIND__FROZEN_LIST) return i;
#endif
    }
    return n;
}

/* ── CALLING A PROCEDURE ─────────────────────────────────────────────────────────────────────────────
 * A call is not a thing the evaluator does. It is a REWRITE: the form naming a procedure becomes a
 * program that runs the body and unbinds again, and the scan meets that program the way it meets any
 * other. Nothing recurses and no frame is spent.
 * ```
 *   (<proc> a b)  ->  (prog1 <body> (remove_bindings env 'p 'q))
 *                     with p and q bound to a and b before the rewrite lands
 * ```
 * ⭐ THE PARAMETER'S SHADOW STACK IS THE CALL STACK. An inner call pushes its own `n` and the pop puts the
 * outer one back, so recursion needs no frames at all — which is the whole reason a binding is a stack
 * per name rather than a slot.
 * ⭐ AND `prog1` IS WHY THE VALUE SURVIVES: `begin` would answer the last thing, which is the unbinding.
 *
 * ⭐⭐ THE ARGUMENTS ARE ALREADY VALUES WHEN THIS RUNS, SO THE BINDING IS MADE HERE RATHER THAN WRITTEN
 * DOWN TO BE PERFORMED LATER. The scan reduces every sub-form before it applies the form containing it,
 * so by the time a call is expanded its arguments are values and there is nothing left to evaluate about
 * them. Laying down an `(add_bindings env 'p a)` for the scan to descend into, apply and pop would build a
 * list per parameter and spend a full cycle of the loop to call the one function this line calls.
 * ⛳ IT IS THE SAME FUNCTION THE OPCODE CALLS, WHICH IS WHY THE SEMANTICS CANNOT DRIFT: a failed push
 * raises here exactly as it would have there, and the body then meets an unbound name either way.
 * ⛳ AND WITH THE BINDINGS GONE FROM THE PROGRAM, THE CALL FORM IS THE `prog1` — there is no outer `begin`
 * left to hold, so the list it needed goes too. What remains is ONE list per call, whatever the arity,
 * because the unbinding takes every name at once.
 * ⛔ EVERYTHING THAT CAN FAIL IS DONE BEFORE ANYTHING IS BOUND — THE ORDER IS RIGHT AND THE ANSWERS ARE
 * DISCARDED, WHICH IS NOT THE SAME THING (`REASONED`, from the call sites below). The deep copy and the
 * unbinding list are checked; every `append` that fills that list and lands it in the call form is
 * `(void)`-cast, and `sys__sublist__append` answers `false` when there is no room for what it was handed.
 * A call that runs out of room there goes on to bind its parameters anyway — half-entered with no
 * unbinding written down, which is the very thing this paragraph is about — and `sys__bindings__add`'s own
 * answer is discarded beside it, where its header says even a `true` does not mean the value went on.
 * ⛳ `let` BUILDS THE SAME UNBINDING AND CARRIES AN `ok` OVER IT, releases the drop list and fails before
 * the first name enters a scope. That is the shape this wants, and the sibling to read for it.
 *
 * ⛔⛔ AND THE BODY IS COPIED, WHICH IS THE PRICE OF EVALUATING BY REWRITING. A verb rewrites the form it
 * is handed, so a body evaluated in place is a body DESTROYED — the second call would find whatever the
 * first one left. Every call therefore makes a list of its own out of the body's picture first, and that is
 * a real cost on the hottest thing a program does, not a detail.
 * ⭐ AND THE PICTURE IS NEVER WRITTEN, SO READING IT COSTS NOBODY ANYTHING. The copy takes no hold on the
 * picture or on any array inside it — the procedure holds all of them — so any number of blocks can expand
 * one procedure at once, and the only counts they touch are those of the objects the body names.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* Turn a call into the program that performs it, in place. */
static __device__ __noinline__ bool eng__eval__zzprivate_expand(uint64_t bindings, uint64_t form, sys__heap_node proc) {
    /* Both halves are read without a hold: the form holds the procedure and the procedure holds them. A
     * procedure of no parameters has nil where their names would be. */
    sys__heap_node names; sys__heap_node body_cell;
    if (!sys__procedure__zzengine_halves(proc.args[0], &names, &body_cell)) return false;
    const uint64_t params = (names.dtype == SYS__KIND__QUOTED_LIST) ? names.args[0] : 0ull;
    const uint64_t body   = body_cell.args[0];
    const uint64_t arity  = (params == 0ull) ? 0ull : sys__node_array__length(params);
    if (sys__sublist__length(form) != arity + 1ull) return false;

    sys__heap_node verb;
    verb.dtype = SYS__KIND__STANDARD; verb.num_args = 0u; verb.args[0] = 0ull;

    /* ⭐ THE EVALUATOR'S DOOR: a body copied once per call, and under `SILVANN_LAZY_THAW` its
     * sub-forms come back frozen, so a call builds only the branch it takes. */
    const uint64_t run = sys__list__thaw_running(body);
    if (run == 0ull) return false;

    /* ONE unbinding form for every parameter, and it goes on the end where it cannot disturb the
     * arguments still standing in the cells this is about to read. */
    if (arity > 0ull) {
        const uint64_t drop = sys__list__create();
        if (drop == 0ull) { (void)sys__heap_object__release(run); return false; }
        sys__heap_node v = verb; v.op_code = SYS__OPCODES__REMOVE_BINDINGS; (void)sys__sublist__append(drop, &v);
        sys__heap_node e; e.dtype = SYS__KIND__VALUE_INT; e.num_args = 0u; e.args[0] = bindings;
        (void)sys__sublist__append(drop, &e);
        sys__node_array_walk pw;
        (void)sys__node_array__walk(params, 0ull, &pw);
        for (const sys__heap_node* pname = sys__node_array__walk_cell(&pw); pname != 0;
             sys__node_array__next(&pw), pname = sys__node_array__walk_cell(&pw))
            (void)sys__sublist__append(drop, pname);
        const sys__heap_node ref = sys__heap_object__reference_to(drop);
        (void)sys__sublist__append(form, &ref);
        (void)sys__heap_object__release(drop);
    }

    /* Each parameter enters its scope now, reading the argument out of the cell it is still in. */
    if (arity > 0ull) {
        sys__node_array_walk pw;
        (void)sys__node_array__walk(params, 0ull, &pw);
        for (uint64_t i = 0ull; i < arity; ++i, sys__node_array__next(&pw)) {
            const sys__heap_node* pname = sys__node_array__walk_cell(&pw);
            if (pname == 0) break;
            const sys__heap_node arg = sys__sublist__nth(form, i + 1ull);
            (void)sys__bindings__add(bindings, pname, &arg);
        }
    }

    /* ⭐ AND THE CALL FORM BECOMES THE PROGRAM — IT IS NOT REPLACED BY ONE. ⚖ ARCHITECT: *"expanding the
     * defun should reuse the existing execution list chunk, not initiate a new one."* The head stops
     * naming the procedure and names `prog1`, the body goes where the first argument stood, and any
     * arguments after it are dropped in ONE rewrite rather than a cell at a time.
     * ⛳ The arguments may be written over because the scopes above are holding them now. */
    sys__heap_node head = verb; head.op_code = SYS__OPCODES__PROG1;
    (void)sys__sublist__replace(form, 0ull, &head);
    const sys__heap_node runref = sys__heap_object__reference_to(run);
    if (arity == 0ull) {
        (void)sys__sublist__append(form, &runref);
    } else {
        (void)sys__sublist__replace(form, 1ull, &runref);
        if (arity > 1ull) (void)sys__sublist__discard(form, 2ull, arity - 1ull);
    }
    (void)sys__heap_object__release(run);
    return true;
}

/* Run a program and answer what it became.
 *
 * `program` is a view the caller holds and keeps holding; this loop takes ONE hold of its own, on the
 * program, before the first descent. Everything descended into is BORROWED — held by the cell of the
 * level above it, up to that program — so a descent costs no reference operation at all, and the
 * invariant it rests on is the one below. The answer comes back BORROWED in the same sense every read is —
 * good while the program that produced it is. */
static __device__ inline sys__heap_node eng__eval(uint64_t bindings, uint64_t program, sys__heap_node* computing_base) {
    /* The loop-carried state is BLOCK_LOCAL: one copy per runner, because `cpu/silicon/environment.cuh`
     * defines the word `static __thread`, and every runner evaluates with its own.
     * ⛔ THAT IS WHY IT IS SPELLED `SYS__SILICON__BLOCK_LOCAL` AND NOT `static`. These declarations are
     * the evaluator's whole loop-carried state, and plain `static` would hand every runner one shared
     * copy of it, silently. ▶ `silicon__header.cuh`'s BLOCK_LOCAL note for the blast radius.
     * ⚠ One copy per runner also means one evaluation per runner at a time: nothing may call `eng__eval`
     * from inside a verb it is applying — and none can, because the packages are included before the
     * engine and cannot name it (`REASONED`, from `engine/manifest.cuh`'s include order). */
    SYS__SILICON__BLOCK_LOCAL sys__heap_node zzhot_places;
    SYS__SILICON__BLOCK_LOCAL sys__heap_node zzhot_answer;
    SYS__SILICON__BLOCK_LOCAL uint64_t       zzhot_current;
    SYS__SILICON__BLOCK_LOCAL bool           zzhot_entering;
    SYS__SILICON__BLOCK_LOCAL uint64_t       zzhot_resume;
#if SILVANN_LIST_POOL
    /* ⛳ Six words beside the five above; ▶ the pool's own header for why block-local and why six. */
    SYS__SILICON__BLOCK_LOCAL uint64_t       zzhot_pool[SILVANN_LIST_POOL_SLOTS];
    for (unsigned int i = 0u; i < (unsigned int)SILVANN_LIST_POOL_SLOTS; ++i) zzhot_pool[i] = 0ull;
#endif
    sys__heap_node& places   = zzhot_places;
    sys__heap_node& answer   = zzhot_answer;
    uint64_t&       current  = zzhot_current;
    bool&           entering = zzhot_entering;
    uint64_t&       resume   = zzhot_resume;

    places = sys__stack__create();
    if (places.dtype != SYS__KIND__OBJECT_REFERENCE) return sys__heap_node__nothing();

    /* Where the scan of `current` stands, when a return put it back there: a walk stood up from the place,
     * so the next scan starts at the cell just written rather than finding it again. */
    sys__list_walk scan;
    bool           standing = false;

    current  = program;
    answer   = sys__heap_node__nothing();
    entering = true;                 /* nothing has run this yet */
    resume   = 0ull;                 /* and nothing of it has been scanned */
    (void)sys__heap_object__retain(current);

    for (;;) {

        /* ⛔ HAS THIS BECOME A VALUE? ASKED AT THE TOP, AND THE REASON IS A DEFECT THIS COST. Reducing a
         * form and writing the answer into its parent can leave THE PARENT holding a single value too —
         * `if` leaves one element, and once that element reduces the parent is finished as well. Asking
         * only after an apply misses that, and the loop then tries to apply the value: a form whose head
         * is a `#t` is not a verb, and the program dies one step from its answer.
         *
         * ⭐⭐ AND WHAT SEPARATES A FINISHED FORM FROM A CALL IS NOT WHAT IS IN IT — IT IS WHETHER WE ARE
         * ARRIVING OR COMING BACK. `becomes` empties a form and puts the answer in, so a verb that
         * answered a procedure leaves a list of one cell holding a procedure; so does `(seven)`, a call of
         * a procedure of no arguments. The two objects are identical, and no test of the cell can tell
         * them apart, because nothing about the cell differs.
         * ```
         *   ENTERING   a form we have not run yet          whatever is at its head, RUN IT
         *   RETURNING  a form we rewrote, or one a child
         *              just handed its answer back to      one cell and not a form means it is DONE
         * ```
         * ⇒ ⭐ SO THE LOOP CARRIES WHICH OF THE TWO IT IS, and it already knew: descending is arriving,
         * and an apply or a pop is coming back. One word, set where `current` changes, and the question
         * stops being a guess about tags.
         * ⛳ `is_form` STAYS, AND IT IS THE OTHER HALF: coming back to a form holding one SUBLIST means
         * the thing it holds still has to run — which is exactly what `if` leaves behind. */
        if (!entering && sys__sublist__length(current) == 1ull) {
            const sys__heap_node only = sys__sublist__nth(current, 0ull);
            const bool is_form = only.dtype == SYS__KIND__OBJECT_REFERENCE && sys__sublist__is(only.args[0]);
            if (!is_form) {
                if (sys__stack__empty(&places)) { answer = only; break; }
                const sys__heap_node place = sys__stack__zzengine_compute_stack_pop(&places);
#if SILVANN_LIST_POOL
                /* ⭐⭐ THE FORM THAT JUST FINISHED IS ABOUT TO BE DESTROYED BY THE WRITE BELOW — the
                 * parent cell names it, and `set` lets that hold go. Retaining it here is what makes
                 * the difference between a teardown and a recycle, and the ORDER is the correctness:
                 *   retain   so it survives the parent's release
                 *   replace  the parent takes ITS OWN hold on the answer, before the next line
                 *   discard  the finished list gives up the answer it was still holding
                 * ⛔ THE DISCARD MUST COME AFTER THE REPLACE. Emptying first would drop the only hold
                 * on an answer that is itself a reference, and `only` is a borrowed copy — the value
                 * would be freed and then written into the parent.
                 * ⛳ A form is one cell at this point (`becomes` left the answer), so the discard is
                 * a single-chunk shift and the list goes back empty and whole. */
                const uint64_t finished = current;
                (void)sys__heap_object__retain(finished);
                current = place.args[0];
                (void)sys__sublist__zzengine_walk_unpark(current, place.args[1], &place, &scan);
                sys__heap_node* back = sys__list__walk_cell(&scan);
                if (back != 0) sys__heap_object__set(back, &only);
                else (void)sys__sublist__replace(current, place.args[1], &only);
                standing = (back != 0);
                (void)sys__sublist__discard(finished, 0ull, sys__sublist__length(finished));
                if (!eng__eval__zzprivate_pool_put(zzhot_pool, finished))
                    (void)sys__heap_object__release(finished);   /* no room: the ordinary end */
#else
                current = place.args[0];                    /* borrowed: the chain above still holds it */
                (void)sys__sublist__zzengine_walk_unpark(current, place.args[1], &place, &scan);
                sys__heap_node* back = sys__list__walk_cell(&scan);
                if (back != 0) sys__heap_object__set(back, &only);
                else (void)sys__sublist__replace(current, place.args[1], &only);
                standing = (back != 0);
#endif
                resume  = place.args[1];                    /* everything before it is already a value */
                continue;   /* and still coming back — reaching here at all required it */
            }
        }

        const uint64_t at = eng__eval__zzprivate_first_form(current, resume, &scan, standing);
        standing = false;
        if (at < sys__sublist__length(current)) {
            /* Descend. The place is the list and the position in it, and the push is a bare transfer:
             * ⛔ IT TAKES NO HOLD OF THAT LIST. What we came back to is kept by the cell of ITS parent
             * that names it, and that cell is written only by the pop that returns to it — the next
             * block is the whole of the argument. */
            sys__heap_node place;
            place.dtype = SYS__KIND__OBJECT_REFERENCE; place.num_args = 0u;
            place.args[0] = current; place.args[1] = at;
            sys__list__zzengine_walk_park(&scan, &place);     /* and where the scan stands, to come back to */
            if (!sys__stack__zzengine_compute_stack_push(&places, &place)) {
                sys__fault__raise(0ull, ENG__EVAL__FAULT_DEPTH);
                answer = sys__heap_node__nothing();
                break;
            }

            /* ⭐ NOTHING IS RETAINED AND NOTHING RELEASED, AND THE CHAIN IS WHY. The form we are leaving
             * is still named by ITS parent's cell; the one we are entering is named by a cell of the one
             * we are leaving. Every level is held by the level above it, up to the program the engine
             * holds — so the stack borrows those holds instead of adding a second set that would only
             * ever be cancelled on the way back. */
            sys__heap_node* found = sys__list__walk_cell(&scan);
            sys__heap_node into = (found != 0) ? *found : sys__sublist__nth(current, at);
#if SILVANN_LAZY_THAW
            /* ⭐⭐ THE FORM IS MADE HERE, AT THE MOMENT IT IS ENTERED, AND NOWHERE ELSE. The cell named
             * an array in the picture; it now names the list thawed from it, and that list's own
             * sub-forms come back frozen in their turn — so a program is built along the path the
             * evaluator actually walks.
             * ⛔ THE `replace` IS WHAT MAKES THE HOLDS COME OUT RIGHT, AND IT IS NOT AN OPTIMISATION:
             * `set` takes the new list's hold before letting the array's go, and the array is still
             * held by the picture either way. Writing the cell also means a form entered twice — tail
             * recursion re-scanning the same list — finds a list the second time and thaws once.
             * ⛔ AND A REFUSAL HERE CANNOT BE IGNORED: with no list made there is nothing to descend
             * into, and carrying on would apply a form whose head is a frozen tag. */
            if (into.dtype == SYS__KIND__FROZEN_LIST) {
                uint64_t made = 0ull;
#if SILVANN_LIST_POOL
                /* ⭐ A RECYCLED LIST IS FILLED, NOT MADE. Only if every slot is empty is one carved. */
                made = eng__eval__zzprivate_pool_take(zzhot_pool);
                if (made != 0ull && !sys__list__thaw_running_into(into.args[0], made)) {
                    (void)sys__heap_object__release(made);
                    made = 0ull;
                }
                if (made == 0ull)
#endif
                made = sys__list__thaw_running(into.args[0]);
                if (made == 0ull) {
                    sys__fault__raise(0ull, ENG__EVAL__FAULT_NOT_A_VERB);
                    answer = sys__heap_node__nothing();
                    break;
                }
                const sys__heap_node ref = sys__heap_object__reference_to(made);
                if (found != 0) sys__heap_object__set(found, &ref);
                else (void)sys__sublist__replace(current, at, &ref);
                (void)sys__heap_object__release(made);   /* the cell holds it now */
                into = ref;
            }
#endif
            current  = into.args[0];
            entering = true;
            resume   = 0ull;
            continue;
        }

        if (sys__sublist__length(current) == 0ull) { answer = sys__heap_node__nothing(); break; }

        /* ⭐ THE NAMES ARE RESOLVED HERE, WHERE THE FORM IS ABOUT TO BE READ AS A CALL, and not at the
         * top of the loop. A form with something still to evaluate inside it is descended into above and
         * never reaches this line; it resolves its own names when its turn comes. Resolving at the top
         * meant every enclosing form was re-swept on every pass through its children — which is O(N) per
         * step over a list of N forms, and the reason a flat program cost N². */
        eng__eval__zzprivate_resolve_bindings(bindings, current);

        const sys__heap_node head = sys__sublist__nth(current, 0ull);
        /* ⭐ A NAME THAT MEANT A PROCEDURE IS A CALL, and a call is a rewrite rather than a step this loop
         * takes: the form becomes the program that performs it and the scan carries on. */
        if (eng__eval__zzprivate_is_procedure(head)) {
            if (!eng__eval__zzprivate_expand(bindings, current, head)) {
                sys__fault__raise(0ull, ENG__EVAL__FAULT_NOT_A_VERB);
                answer = sys__heap_node__nothing();
                break;
            }
            /* ⛳ NO CONTROL CAN MOVE THIS ONE TODAY, AND IT STAYS ANYWAY. An expansion always leaves a
             * form with the body at index 1, so the next step is always a descent and always writes this
             * word before anything reads it. It is here for the day an expansion answers outright — a
             * call that collapses to a value would then be read as the answer, which is what it is. */
            entering = false;
            resume   = 0ull;        /* an expansion replaces the form; nothing before is known */
            continue;
        }
        if (head.dtype != SYS__KIND__STANDARD) {
            sys__fault__raise(0ull, ENG__EVAL__FAULT_NOT_A_VERB);
            answer = sys__heap_node__nothing();
            break;
        }
        const uint64_t verb = head.op_code;
        (void)sys__heap_object__apply(verb, computing_base, current);

        entering = false;                                   /* whatever it left, we wrote it */
        resume   = 0ull;            /* and a verb may have rewritten the whole form */

        /* ⭐ AND NOTHING STOPS HERE. A program's answer is what it reduces to, and the loop ends when the
         * outermost form is one value — so a verb that rewrote itself into a bigger form is simply scanned
         * again and costs nothing, and whatever finishes last is the answer. */
    }

    /* The one hold this loop took is the one on the program above, and this gives it back — by naming
     * the PROGRAM, not `current`. They are the same list when the loop ends at the top, and different
     * ones when it stops below it: a fault, or a form that comes to nothing, can end the loop with
     * `places` non-empty and `current` a borrowed inner form. Nothing descended into was ever held, so
     * there is nothing else to let go of. The stack goes last and takes what it is still carrying. */
    (void)sys__heap_object__release(program);
    /* ⛳ THE STACK IS DRAINED AND NOTHING IS RELEASED OUT OF IT, because it never held anything: every
     * entry was borrowed from the chain, and releasing the program above lets that chain go from the top. */
    while (!sys__stack__empty(&places)) (void)sys__stack__zzengine_compute_stack_pop(&places);
#if SILVANN_LIST_POOL
    /* ⛔ NOT TIDINESS. The pool is one per runner and is emptied on every call, so a list still in a slot is
     * held by nothing and leaked. Every exit from the loop above reaches here. */
    for (unsigned int i = 0u; i < (unsigned int)SILVANN_LIST_POOL_SLOTS; ++i) {
        if (zzhot_pool[i] != 0ull) { (void)sys__heap_object__release(zzhot_pool[i]); zzhot_pool[i] = 0ull; }
    }
#endif
    (void)sys__heap_object__release(places.args[0]);
    return answer;
}

#endif /* SILVANN__ENGINE_EVAL_CUH */
