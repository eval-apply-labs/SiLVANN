#ifndef SILVANN__PACKAGES_SYS_CPU_BINDINGS__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_BINDINGS__IMPL_CUH
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * FOUR ARGUMENTS IN THIS FILE ARE PRICED IN `k_eval`'s REGISTERS — a device compiler's accounting — and
 * the evaluator runs on the host. RE-DERIVE them there rather than inherit them:
 *   · the reach cache's "block-local storage holds whatever the previous launch left": the hazard is a
 *     RUNNER THREAD's cached value outliving its computation. ▶ `heap__impl.cuh`'s base-cache note.
 *     The ordinary comment in `contracts/objects/bindings.cuh` now says so; nothing is left to move.
 *   · "spills from 2 to 18 ...", "passing the cell costs 1.28% ...", "`create` is inlined into every
 *     kernel ...": the NUMBERS will not survive on a CPU, and "pass the string not the cell" exists
 *     because a device compiler materialises an aggregate argument in memory. **UNMEASURED on a host.**
 *     The figures were taken out of the ordinary comments and LIVE HERE NOW, verbatim:
 *     ① `zzprivate_scope_in_place`: `MEASURED` on `k_eval`, with a number past the width as
 *       the test — the out-of-line calls in the engine's name read and in `add` take its register
 *       spills from 2 to 18, and with both removed they are 2 again, whatever shape the call has. Its
 *       private segment, 576 → 1,536 bytes, is not theirs: it moved with the old bindings files restored
 *       but the dictionary reachable, because a kernel's segment is sized for everything it can reach.
 *       Neither is a rate: that build was 0.50% FASTER on fib(15) — `ENG-9`'s finding again, register
 *       demand and throughput are not coupled on that kernel.
 *     ② `zzprivate_add_spelled`: `MEASURED` on fib(15), passing the cell instead of the
 *       string costs 1.28% (2,070.2 → 2,096.8 ms) and 64 bytes of `k_eval`'s private segment.
 *     ③ `zzprivate_base_fits`: out of line "because `create` is inlined into every kernel that begins a
 *       computation"; `REASONED` there that moving it moved neither the frame nor the spills of `k_eval`.
 * ⛳ RETIREMENT: when the four have been re-derived on the host.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* What this file needs, named where a reader — and an editor — can follow it. */


#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "fault__header.cuh"        /* how it says a caller asked for something that is not there */
#include "heap_object__header.cuh"   /* the head behind a reference, holds, and writing into a cell */
#include "heap__header.cuh"         /* the carve, and turning an offset back into an address */
#include "error.cuh"                /* the value an unbound name answers with */
#include "node_array__header.cuh"  /* both tables are one */
#include "list__header.cuh"         /* what a published verb is handed */
#include "stack__header.cuh"  /* a name's scopes are one, and where a dying object moves things */
#include "opcodes/opcodes.cuh"           /* the two lines every verb ends with */
#include "../contracts/objects/computing_base.cuh" /* where a computation keeps the environment it runs in */
#include "bindings__header.cuh"     /* the verbs it fills in, and the fault words they raise */
#include "dictionary.cuh"           /* where every name past the width is kept */
#include "silicon/silicon__header.cuh"   /* SYS__SILICON__BLOCK_LOCAL, for the reach */
#include "../contracts/objects/bindings.cuh" /* its constants, fault words and layouts */

/* The node holding the two tables and the error. A reference names it, so this is the reference read as
 * what it points at — one line, and it is here rather than written out in every verb below because they
 * would each have had to check the same thing first and get the same address right. It raises where it
 * refuses, which is why none of them does. */
static __device__ inline sys__heap_node* sys__bindings__zzprivate_fields(uint64_t bindings) {
    if (!sys__bindings__is(bindings)) {
        sys__fault__raise(0ull, SYS__BINDINGS__FAULT_KIND);
        return 0;
    }
    return sys__heap__object_full_address(bindings);
}

static __device__ inline sys__bindings_reach* sys__bindings__zzprivate_reach(void) {
    SYS__SILICON__BLOCK_LOCAL sys__bindings_reach remembered;
    return &remembered;
}

/* Take the three answers once. Handed nothing, it empties — which is how a computation that carries no
 * bindings leaves nothing behind for the next one to match against. */
static __device__ inline void sys__bindings__zzpackage_reach_arm(uint64_t bindings) {
    sys__bindings_reach* reach = sys__bindings__zzprivate_reach();
    reach->bindings = 0ull;
    if (bindings == 0ull) return;
    sys__heap_node* fields = sys__bindings__zzprivate_fields(bindings);
    if (fields == 0) return;
    reach->scopes   = fields->args[SYS__BINDINGS__SCOPES];
    reach->width    = sys__node_array__length(reach->scopes) - SYS__BINDINGS__FIRST_NAME;
    reach->bindings = bindings;            /* written LAST, so a half-filled reach is never a match */
}

static __device__ inline void sys__bindings__zzpackage_reach_forget(void) {
    sys__bindings__zzprivate_reach()->bindings = 0ull;
}

/* ── A SPELLED NAME ──────────────────────────────────────────────────────────────────────────────────
 * Both tables keep, at element 0, the dictionary holding every name spelled by its string: stacks in the
 * scopes', values in a view's. Everything that reads one is out of line, so the path that reads a numbered
 * name is the one it always was plus the one to skip the hatch.
 * ⛳ THE KEY IS THE STRING'S OFFSET, as an integer cell — the same key the symbol table files a string's
 * number under, and sound for the same reason: an interned string never dies, so its offset is never
 * anybody else's.
 * ⛳ THE SCOPES' HATCH IS READ WITH `zzpackage_at`, for the argument that array's header states: the scopes
 * are held by this object and were checked when the reach was armed. ⛔ A BASE'S IS READ WITH `borrow` AND
 * NEVER THAT WAY, because `zzpackage_at` hands back a writable cell and the base is the one table several
 * environments share — every use of it here goes straight to a read, which is what the claim gate's
 * `base_never_written` checks. What comes out of it is a SEALED dictionary, so nothing can write that either.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* The dictionary a hatch cell names, or nothing. */
static __device__ inline uint64_t sys__bindings__zzprivate_dictionary_in(const sys__heap_node* hatch) {
    return hatch->dtype == SYS__KIND__OBJECT_REFERENCE ? hatch->args[0] : 0ull;
}

/* The scopes' dictionary, or nothing yet. */
static __device__ inline uint64_t sys__bindings__zzprivate_hatch(uint64_t scopes) {
    return sys__bindings__zzprivate_dictionary_in(sys__node_array__zzpackage_at(scopes, SYS__BINDINGS__HATCH));
}

/* A spelled name: what a dictionary holds for its string, BORROWED, and whether it holds anything.
 * Borrowing is sound for both tables — the scopes' dictionary is this object's alone and is never written
 * while a read is using what it found, and a base's is sealed. */
static __device__ __noinline__ sys__heap_node sys__bindings__zzprivate_spelled_in(uint64_t dictionary,
                                                                                 uint64_t string, bool* found) {
    *found = false;
    if (dictionary == 0ull) return sys__heap_node__nothing();
    const sys__heap_node key = sys__dictionary__number_key(string);
    return sys__dictionary__zzpackage_find(dictionary, &key, found);
}

/* A spelled name: its scope stack, BORROWED, or a null when it has none yet. Out of line so the lookup of
 * a numbered name carries nothing of it. */
static __device__ __noinline__ sys__heap_node sys__bindings__zzprivate_spelled_scope(uint64_t scopes,
                                                                                    uint64_t string) {
    bool found = false;
    return sys__bindings__zzprivate_spelled_in(sys__bindings__zzprivate_hatch(scopes), string, &found);
}

/* Give a spelled name its stack: made, put in the scopes' dictionary — which is made first when this is
 * the first such name — and answered HELD. A refusal answers nothing, having raised. */
static __device__ __noinline__ sys__heap_node sys__bindings__zzprivate_make_spelled(uint64_t scopes,
                                                                                   uint64_t string) {
    uint64_t hatch = sys__bindings__zzprivate_hatch(scopes);
    if (hatch == 0ull) {
        const uint64_t made = sys__dictionary__create();
        if (made == 0ull) return sys__heap_node__nothing();
        const sys__heap_node reference = sys__heap_object__reference_to(made);
        const bool placed = sys__node_array__set(scopes, SYS__BINDINGS__HATCH, &reference);
        (void)sys__heap_object__release(made);     /* the table's hold is the one that stays */
        if (!placed) return sys__heap_node__nothing();
        hatch = made;
    }
    const sys__heap_node stack = sys__stack__create();
    if (stack.dtype != SYS__KIND__OBJECT_REFERENCE) return sys__heap_node__nothing();
    const sys__heap_node key = sys__dictionary__number_key(string);
    if (!sys__dictionary__put(hatch, &key, &stack)) {
        (void)sys__heap_object__release(stack.args[0]);
        return sys__heap_node__nothing();
    }
    return stack;                                  /* the dictionary holds it, and so does the caller */
}

/* A NUMBERED name: its scope stack, BORROWED. Answers false when this is not an environment, when the
 * cell is not a numbered name, and when the number has no cell — raising for all but a spelled name, which
 * a caller then reads out of line. The reach is armed either way, so a caller tells a spelled name from a
 * refusal by asking whether the reach names this environment and the cell is spelled.
 * ⭐⭐ IT IS THE LOOKUP THE EVALUATOR INLINES, SO IT IS KEPT TO a kind test, a bounds test and a read. A
 * caller that gets false for a spelled name goes out of line.
 * ⛔ AND THE CALLS THAT DO THAT ARE WHAT THE EVALUATOR PAYS FOR — the one in the engine's name read and the
 * one in `add` (which binds a call's arguments). `REASONED` on the host: keeping the spelled path out of
 * line keeps it off the numbered path — the price was measured only on the device kernel (▶ the note at
 * the top). */
static __device__ inline bool sys__bindings__zzprivate_scope_in_place(uint64_t bindings, const sys__heap_node* name,
                                                                      sys__heap_node* stack) {
    if (stack == 0) return false;
    sys__bindings_reach* reach = sys__bindings__zzprivate_reach();
    if (reach->bindings != bindings) {
        sys__bindings__zzpackage_reach_arm(bindings);
        if (reach->bindings != bindings) return false;      /* it was not a bindings at all */
    }
    if (!sys__bindings__numbered(name->dtype)) {
        if (!sys__bindings__spelled(name->dtype)) sys__fault__raise(0ull, SYS__BINDINGS__FAULT_NOT_NAME);
        return false;                                      /* a spelled one: the caller asks elsewhere */
    }
    if (name->args[0] >= reach->width) {
        sys__fault__raise(0ull, SYS__BINDINGS__FAULT_RANGE);
        return false;
    }
    /* ⛳ THE HOLD IS TAKEN BY THE WRAPPER BELOW RATHER THAN HERE, because the checks `get` would repeat
     * are the ones the arming already did — and because one caller wants the lookup WITHOUT the hold.
     * ⭐ A SLOT NOTHING HAS BOUND IN HOLDS A NULL, AND THAT IS AN ANSWER, NOT A REFUSAL: the name is this
     *   environment's and has no scope yet — every reader treats it as empty, as it already did a spelled name's. */
    *stack = *sys__node_array__zzpackage_at(reach->scopes, name->args[0] + SYS__BINDINGS__FIRST_NAME);
    return stack->dtype == SYS__KIND__OBJECT_REFERENCE || stack->dtype == SYS__KIND__VALUE_NULL;
}

/* Whether the in-place lookup just said no because the name is spelled, rather than because it refused. */
static __device__ inline bool sys__bindings__zzprivate_spelled_here(uint64_t bindings, const sys__heap_node* name) {
    return sys__bindings__zzprivate_reach()->bindings == bindings && sys__bindings__spelled(name->dtype);
}

/* Any name's scope stack, BORROWED. Answers false when this is not an environment or the cell is refused;
 * a spelled name with no stack yet answers true and leaves `*stack` a null, which every reader treats as
 * empty. */
static __device__ inline bool sys__bindings__zzprivate_scope_borrowed(uint64_t bindings, const sys__heap_node* name,
                                                                      sys__heap_node* stack) {
    if (sys__bindings__zzprivate_scope_in_place(bindings, name, stack)) return true;
    if (stack == 0 || !sys__bindings__zzprivate_spelled_here(bindings, name)) return false;
    *stack = sys__bindings__zzprivate_spelled_scope(sys__bindings__zzprivate_reach()->scopes, name->args[0]);
    return true;
}

/* Whether a scope stack is there and holds something. */
static __device__ inline bool sys__bindings__zzprivate_scoped(const sys__heap_node* stack) {
    return stack->dtype == SYS__KIND__OBJECT_REFERENCE && !sys__stack__empty(stack);
}

/* A spelled name, in the base: whether its sealed dictionary has it, and what — held when `hold`. */
static __device__ __noinline__ bool sys__bindings__zzprivate_spelled_underneath(const sys__heap_node* fields,
                                                                                uint64_t string, bool hold,
                                                                                sys__heap_node* answer) {
    const sys__heap_node hatch = sys__node_array__borrow(fields->args[SYS__BINDINGS__BASE], SYS__BINDINGS__HATCH);
    bool found = false;
    *answer = sys__bindings__zzprivate_spelled_in(sys__bindings__zzprivate_dictionary_in(&hatch), string, &found);
    if (found && hold && sys__heap_node__carries_reference(answer->dtype))
        (void)sys__heap_object__retain(answer->args[0]);
    return found;
}

/* Enter a scope for a spelled name — its stack found or, on its first binding, made — when the in-place
 * lookup said no and the caller has established that this environment's reach is armed.
 * ⛳ IT TAKES THE STRING AND NOT THE CELL, as every out-of-line function here does: these are called from
 * the two functions the evaluator inlines, and a word is the cheapest thing to hand across a call.
 * `REASONED` on the host — the price was measured only on the device kernel (▶ the note at the top). */
static __device__ __noinline__ bool sys__bindings__zzprivate_add_spelled(uint64_t bindings, uint64_t string,
                                                                         const sys__heap_node* value) {
    (void)bindings;
    const uint64_t scopes = sys__bindings__zzprivate_reach()->scopes;
    sys__heap_node stack = sys__bindings__zzprivate_spelled_scope(scopes, string);
    if (stack.dtype == SYS__KIND__OBJECT_REFERENCE) {
        (void)sys__heap_object__retain(stack.args[0]);
    } else {
        stack = sys__bindings__zzprivate_make_spelled(scopes, string);             /* comes back held */
        if (stack.dtype != SYS__KIND__OBJECT_REFERENCE) return false;              /* the making raised */
    }
    const bool went = sys__stack__push(&stack, value);                            /* a refusal raised */
    (void)sys__heap_object__release(stack.args[0]);
    return went;
}

/* What a spelled name means when this environment has no scope for it: the base's value, or this
 * environment's unbound error — HELD when `hold`, and the unbound error is held either way. */
static __device__ __noinline__ sys__heap_node sys__bindings__zzprivate_spelled_below(const sys__heap_node* fields,
                                                                                    uint64_t string, bool hold) {
    sys__heap_node answer;
    if (fields->args[SYS__BINDINGS__BASE] != 0ull &&
        sys__bindings__zzprivate_spelled_underneath(fields, string, hold, &answer)) return answer;
    (void)sys__heap_object__retain(fields->args[SYS__BINDINGS__UNBOUND]);
    return sys__heap_object__reference_to(fields->args[SYS__BINDINGS__UNBOUND]);
}

/* The engine's borrowed read of a spelled name, whole — its own scope, then the base, then unbound. The
 * caller has established that this environment's reach is armed; it takes the string, for the reason above. */
static __device__ __noinline__ sys__heap_node sys__bindings__zzprivate_borrow_spelled(uint64_t bindings,
                                                                                     uint64_t string) {
    const sys__heap_node stack =
        sys__bindings__zzprivate_spelled_scope(sys__bindings__zzprivate_reach()->scopes, string);
    if (sys__bindings__zzprivate_scoped(&stack)) return sys__stack__zzpackage_peek_borrowed(&stack);
    return sys__bindings__zzprivate_spelled_below(sys__heap__object_full_address(bindings), string, false);
}

/* The same lookup, and the scope stack comes back HELD. Every caller that keeps the stack across
 * anything at all uses this one; the borrowing form above is for the single contiguous path that does
 * not. A refusal has nothing to retain, so it is the same refusal. */
static __device__ inline bool sys__bindings__zzprivate_scope(uint64_t bindings, const sys__heap_node* name,
                                                             sys__heap_node* stack) {
    if (!sys__bindings__zzprivate_scope_borrowed(bindings, name, stack)) return false;
    if (sys__heap_node__carries_reference(stack->dtype)) (void)sys__heap_object__retain(stack->args[0]);
    return true;
}

static __device__ inline bool sys__bindings__is(uint64_t bindings) {
    if (!sys__heap_object__zzpackage_addressable(bindings)) return false;
    return sys__heap_object__is_type_from_head(sys__heap_node__zzpackage_head(bindings),
                                              SYS__KIND__BINDINGS);
}

static __device__ inline uint64_t sys__bindings__symbols(uint64_t bindings) {
    sys__heap_node* fields = sys__bindings__zzprivate_fields(bindings);
    if (fields == 0) return 0ull;
    /* The width is the scope table's length less its hatch and is not written down anywhere here. A second
     * copy of a number is a second thing that can be wrong, and the table already had to know. */
    return sys__node_array__length(fields->args[SYS__BINDINGS__SCOPES]) - SYS__BINDINGS__FIRST_NAME;
}

/* A base is somebody else's table, so everything that can be wrong about it is wrong before anything is
 * made: that it is not a table at all, which is the table's own complaint and is raised in its words; that
 * it is the wrong size; and that what it keeps past the width could still be written.
 * ⛳ OUT OF LINE because `create` is inlined into every caller that begins a computation, and this is the
 * part only a snapshot needs. Its cost was reasoned only on the device kernel (▶ the note at the top). */
static __device__ __noinline__ bool sys__bindings__zzprivate_base_fits(uint64_t base, uint64_t width) {
    if (!sys__node_array__is(base)) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_KIND);
        return false;
    }
    if (sys__node_array__length(base) != width + SYS__BINDINGS__FIRST_NAME) {
        sys__fault__raise(0ull, SYS__BINDINGS__FAULT_MISFIT);
        return false;
    }
    const sys__heap_node hatch = sys__node_array__borrow(base, SYS__BINDINGS__HATCH);
    const uint64_t under = sys__bindings__zzprivate_dictionary_in(&hatch);
    if (under != 0ull && !sys__dictionary__readonly(under)) {
        sys__fault__raise(0ull, SYS__BINDINGS__FAULT_HATCH);
        return false;
    }
    return true;
}

static __device__ inline uint64_t sys__bindings__create(uint64_t symbols, uint64_t base) {
    if (symbols == 0ull) {
        sys__fault__raise(0ull, SYS__BINDINGS__FAULT_EMPTY);
        return 0ull;
    }
    const uint64_t width = symbols < SYS__BINDINGS__WIDTH_MAX ? symbols : SYS__BINDINGS__WIDTH_MAX;
    if (base != 0ull && !sys__bindings__zzprivate_base_fits(base, width)) return 0ull;

    sys__heap_node* head = sys__heap__make(SYS__KIND__BINDINGS);
    if (head == 0) return 0ull;                    /* the carve raised, and said which of its reasons */

    /* ⛔ THE THREE SLOTS ARE EMPTIED BEFORE ANYTHING ELSE CAN FAIL, AND THAT IS NOT TIDINESS. A chunk is
     * not zeroed, so until this runs the head holds whatever the last occupant left — and every exit
     * below ends this object, which reads these three and would move whatever those bits happened to
     * look like onto the teardown chain. */
    head[1].args[SYS__BINDINGS__SCOPES]  = 0ull;
    head[1].args[SYS__BINDINGS__BASE]    = 0ull;
    head[1].args[SYS__BINDINGS__UNBOUND] = 0ull;
    const uint64_t bindings = sys__heap__offset(head + 1);

    const uint64_t scopes = sys__node_array__zzpackage_create_run(width + SYS__BINDINGS__FIRST_NAME);
    if (scopes == 0ull) { (void)sys__heap_object__release(bindings); return 0ull; }
    head[1].args[SYS__BINDINGS__SCOPES] = scopes;

    /* ⭐⭐ NO STACK IS MADE UP FRONT: A SLOT GETS ONE WHEN A NAME IS FIRST BOUND IN IT (`add`), and until then holds
     * the null the array was made with, which every reader treats as an empty scope. ⚖ Once, *"does an empty stack
     * consume chunks? if not lets do eager"* — and eager it was, one small object a name. ⚖ *"the scope
     * stack should in reality be one node wide per name … it is only filled when a let/set is actually done"* ·
     * *"a bulk copy of that node array of empty lists"*. The empty table IS that paste: the array is made filled.
     * `MEASURED` before: handing a computation to another block made one stack a slot of the view's width and tore
     * them down — ~0.15 us a slot, 12.7 us at 16 wide and 90 at 508, whatever the names bound. */
    /* Quietly: it is a value this object will answer with, not a failure anybody is reporting. Making it
     * loudly would put an unbound word in the log every time an environment is created, which is a
     * reader being told something went wrong at the one moment nothing has. */
    const uint64_t unbound = sys__error__zzpackage_place(SYS__BINDINGS__FAULT_UNBOUND);
    if (unbound == 0ull) { (void)sys__heap_object__release(bindings); return 0ull; }
    head[1].args[SYS__BINDINGS__UNBOUND] = unbound;

    if (base != 0ull) {
        (void)sys__heap_object__retain(base);
        head[1].args[SYS__BINDINGS__BASE] = base;
    }
    return bindings;
}

static __device__ inline sys__heap_node sys__bindings__get(uint64_t bindings, const sys__heap_node* name) {
    sys__heap_node stack;
    if (name == 0 || !sys__bindings__zzprivate_scope(bindings, name, &stack)) return sys__heap_node__nothing();
    sys__heap_node* fields = sys__heap__object_full_address(bindings);

    sys__heap_node answer;
    if (sys__bindings__zzprivate_scoped(&stack)) {
        answer = sys__stack__peek(&stack);                       /* comes back held */
    } else if (sys__bindings__spelled(name->dtype)) {
        answer = sys__bindings__zzprivate_spelled_below(fields, name->args[0], true);
    } else if (fields->args[SYS__BINDINGS__BASE] != 0ull) {
        answer = sys__node_array__get(fields->args[SYS__BINDINGS__BASE], name->args[0] + SYS__BINDINGS__FIRST_NAME);
    } else {
        /* Nothing above and nothing underneath. The answer is this environment's own error, handed out
         * the way every other answer is — held, so the caller owes it back like anything else and no
         * caller has to know that this one is shared. */
        (void)sys__heap_object__retain(fields->args[SYS__BINDINGS__UNBOUND]);
        answer = sys__heap_object__reference_to(fields->args[SYS__BINDINGS__UNBOUND]);
    }
    if (sys__heap_node__carries_reference(stack.dtype)) (void)sys__heap_object__release(stack.args[0]);
    return answer;
}

/* WHAT A NAME MEANS, BORROWED — no lock, no hold on the scope stack, and no hold on the answer.
 *
 * ⚖ ARCHITECT, S99: *"a private sys__bindings__zzengine_get_noretain could be the solution we are
 * looking for, which does not lock/hold the bindings stack object nor the returned value."*
 *
 * ⛔⛔ THE CALLER OWES A HOLD BEFORE ANYTHING ELSE CAN HAPPEN, AND THAT IS THE WHOLE CONTRACT. The one
 * caller is the evaluator's name resolution, which writes the answer straight into a cell — and the
 * cell takes its OWN hold as it is written. So the borrow lasts from this return to that write, with
 * nothing in between, which is the contiguous path the borrow needs. ⇒ ⛔ A CALLER THAT RETURNS THIS
 * VALUE TO SOMEBODY ELSE HAS BROKEN THE CONTRACT, and nothing will say so: it performs no event, so
 * there is no count that fails to balance and no fault to raise. Its mistake is a USE AFTER FREE.
 *
 * ⭐ WHY IT IS SPELLED `zzengine_` AND NOT `zzpackage_`. ⚖ ARCHITECT: *"this is my engine and i do as i
 * please, as i would be the one having to maintain it and the rules are more about other packages than
 * set in stone."* The marker says exactly that — the engine may call this, another PACKAGE may not —
 * and `src_c_subset_gate.py` enforces it as a third tier beside `zzprivate_` (its own file) and
 * `zzpackage_` (its own package). ⛳ A published verb cannot reach it at all, because only the rows in
 * `language_contract.cuh` are published and this is not one of them.
 *
 * ⛳ WHAT IT DOES NOT DO: the unbound answer still comes back HELD, because it is a shared object this
 * environment hands to everybody and there is no other way to give it out. That path is the one a
 * program takes when the name is not there, which is not the path this exists for. */
static __device__ inline sys__heap_node sys__bindings__zzengine_get_noretain(uint64_t bindings,
                                                                            const sys__heap_node* name) {
    sys__heap_node stack;
    if (!sys__bindings__zzprivate_scope_in_place(bindings, name, &stack)) {
        if (!sys__bindings__zzprivate_spelled_here(bindings, name)) return sys__heap_node__nothing();
        return sys__bindings__zzprivate_borrow_spelled(bindings, name->args[0]);
    }
    sys__heap_node* fields = sys__heap__object_full_address(bindings);

    if (stack.dtype == SYS__KIND__OBJECT_REFERENCE && !sys__stack__empty(&stack)) return sys__stack__zzpackage_peek_borrowed(&stack);
    if (fields->args[SYS__BINDINGS__BASE] != 0ull)
        return sys__node_array__borrow(fields->args[SYS__BINDINGS__BASE], name->args[0] + SYS__BINDINGS__FIRST_NAME);
    (void)sys__heap_object__retain(fields->args[SYS__BINDINGS__UNBOUND]);
    return sys__heap_object__reference_to(fields->args[SYS__BINDINGS__UNBOUND]);
}

/* ⛳ IT COSTS WHAT A READ COSTS, WHICH IS NOT WHAT THE SAME VERB COSTS ON A TABLE. There the type of a
 * cell is one load and no counting; here what a name means may be the top of a stack, so finding it is
 * the read and the only thing saved is the caller's release. That saving is real — a caller asking what
 * is there and owing nothing for the answer is why this exists — but it is the caller's and not the
 * machine's. */
static __device__ inline sys__kind sys__bindings__type(uint64_t bindings, const sys__heap_node* name) {
    const sys__heap_node value = sys__bindings__get(bindings, name);
    if (value.dtype == SYS__KIND__VALUE_NULL && !sys__bindings__is(bindings)) return SYS__KIND__INVALID;
    if (sys__heap_node__carries_reference(value.dtype)) (void)sys__heap_object__release(value.args[0]);
    return value.dtype;
}

static __device__ inline bool sys__bindings__bound(uint64_t bindings, const sys__heap_node* name) {
    sys__heap_node stack;
    if (name == 0 || !sys__bindings__zzprivate_scope(bindings, name, &stack)) return false;
    if (sys__heap_node__carries_reference(stack.dtype)) (void)sys__heap_object__release(stack.args[0]);

    const sys__heap_node value = sys__bindings__get(bindings, name);
    /* ⛔⛔ THE TEST IS NARROWER THAN THE HOLD, WHICH IS WHY THE RELEASE HAS TO COME FIRST. Nothing but an
     * OBJECT_REFERENCE can be the unbound error, so every other kind answers yes without being examined —
     * but `get` hands its answer over HELD on everything `sys__heap_node__carries_reference` names, and
     * that is QUOTED_LIST, PROCEDURE_REFERENCE and QUOTED_ARRAY as well as the one tested below. An arm
     * that answered without giving the hold back would add one to the object's count every time somebody
     * asked whether a name was bound, and a count that only rises is an object that never dies — with no
     * event anywhere: nothing fails to balance, nothing faults, and a suite that does not read counts
     * passes over it.
     * ⇒ ★ SO THE SHORT ARM IS WHERE THE ACCOUNTING IS EASIEST TO LOSE, BEING THE ONE THAT LOOKS LIKE IT
     *   DOES NOTHING. ⛳ The idiom is the function directly above: release on `carries_reference`, then
     *   answer. That is what makes the header's *"takes nothing and holds nothing"* true of a name
     *   `defun` filled, and not only of one bound to a plain value. */
    if (value.dtype != SYS__KIND__OBJECT_REFERENCE) {
        if (sys__heap_node__carries_reference(value.dtype)) (void)sys__heap_object__release(value.args[0]);
        return true;
    }
    /* ⭐ IT ASKS WHAT THE ANSWER SAYS AND NOT WHICH OBJECT IT IS. Comparing against this environment's
     * own error would be one instruction cheaper and would be wrong the moment the answer came out of a
     * base somebody else built, whose unbound error is a different object saying the same thing. */
    const bool unbound = sys__error__is(value.args[0]) &&
                         sys__error__code(value.args[0]) == SYS__BINDINGS__FAULT_UNBOUND;
    (void)sys__heap_object__release(value.args[0]);
    return !unbound;
}

/* ⛳ WHAT IT CANNOT TELL YOU, SAID HERE BECAUSE NOTHING ELSE WILL. A push that could not find room
 * raises and answers nothing, so this reports that the name was reached and the value handed over — not
 * that it is now on the stack. RETIREMENT: the day a push answers, this returns what it answers. */
static __device__ inline bool sys__bindings__add(uint64_t bindings, const sys__heap_node* name,
                                                 const sys__heap_node* value) {
    if (name == 0 || value == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return false;
    }
    /* ⛳ THE IN-PLACE LOOKUP, BECAUSE THE EVALUATOR INLINES THIS: a call binds its arguments here. A
     * spelled name goes out of line whole — see `sys__bindings__zzprivate_scope_in_place`. */
    sys__heap_node stack;
    if (!sys__bindings__zzprivate_scope_in_place(bindings, name, &stack))
        return sys__bindings__zzprivate_spelled_here(bindings, name)
            && sys__bindings__zzprivate_add_spelled(bindings, name->args[0], value);
    if (stack.dtype != SYS__KIND__OBJECT_REFERENCE) {
        /* ⭐ THE FIRST BINDING OF THIS NAME HERE: its stack is made now and put in its slot — ▶ `create` */
        stack = sys__stack__create();
        if (stack.dtype != SYS__KIND__OBJECT_REFERENCE) return false;                      /* the making raised */
        const bool placed = sys__node_array__set(sys__bindings__zzprivate_reach()->scopes,
                                                 name->args[0] + SYS__BINDINGS__FIRST_NAME, &stack);
        (void)sys__heap_object__release(stack.args[0]);                    /* the table's hold is the one kept */
        if (!placed) return false;
    }
    (void)sys__heap_object__retain(stack.args[0]);
    const bool went = sys__stack__push(&stack, value);                            /* a refusal raised */
    (void)sys__heap_object__release(stack.args[0]);
    return went;
}

static __device__ inline bool sys__bindings__remove(uint64_t bindings, const sys__heap_node* name) {
    sys__heap_node stack;
    if (name == 0 || !sys__bindings__zzprivate_scope(bindings, name, &stack)) return false;
    if (!sys__bindings__zzprivate_scoped(&stack)) {
        /* A scope that was never entered is being left. That is a caller whose two ends have come apart,
         * and doing nothing quietly here is how the next one goes wrong somewhere else. */
        sys__fault__raise(0ull, SYS__BINDINGS__FAULT_NO_SCOPE);
        if (sys__heap_node__carries_reference(stack.dtype)) (void)sys__heap_object__release(stack.args[0]);
        return false;
    }
    /* A pop hands over what the stack was holding, so this walks away owning it — and what it does with
     * it is let it go, because leaving a scope is the end of that value's time here. */
    sys__heap_node gone = sys__stack__pop(&stack);
    if (sys__heap_node__carries_reference(gone.dtype)) (void)sys__heap_object__release(gone.args[0]);
    (void)sys__heap_object__release(stack.args[0]);
    return true;
}

static __device__ inline bool sys__bindings__set(uint64_t bindings, const sys__heap_node* name,
                                                 const sys__heap_node* value) {
    if (name == 0 || value == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return false;
    }
    sys__heap_node stack;
    if (!sys__bindings__zzprivate_scope(bindings, name, &stack)) return false;
    if (!sys__bindings__zzprivate_scoped(&stack)) {
        /* Nothing of this environment's holds the name. What is underneath may well, and it is read-only
         * and shared, so there is nothing here to assign into — which is a different mistake from a name
         * nobody has bound anywhere, and a caller that wants to know which asks. */
        sys__fault__raise(0ull, SYS__BINDINGS__FAULT_NO_SCOPE);
        if (sys__heap_node__carries_reference(stack.dtype)) (void)sys__heap_object__release(stack.args[0]);
        return false;
    }
    sys__stack__replace(&stack, value);
    (void)sys__heap_object__release(stack.args[0]);
    return true;
}

/* What every spelled name means now, as a SEALED dictionary at the picture's element 0 — or nothing
 * there, when no spelled name means anything. The base's own spelled names are copied first and this
 * environment's scopes are written over them, which is the read order of a name. The keys go across as
 * they are: a string's offset is the same name in both. */
static __device__ __noinline__ bool sys__bindings__zzprivate_picture_spelled(uint64_t bindings, uint64_t picture) {
    const sys__heap_node* fields = sys__heap__object_full_address(bindings);
    const uint64_t mine = sys__bindings__zzprivate_hatch(fields->args[SYS__BINDINGS__SCOPES]);
    const sys__heap_node hatch = fields->args[SYS__BINDINGS__BASE] != 0ull
        ? sys__node_array__borrow(fields->args[SYS__BINDINGS__BASE], SYS__BINDINGS__HATCH)
        : sys__heap_node__nothing();
    const uint64_t under = sys__bindings__zzprivate_dictionary_in(&hatch);
    if (mine == 0ull && under == 0ull) return true;

    const uint64_t past = under != 0ull ? sys__dictionary__copy(under) : sys__dictionary__create();
    if (past == 0ull) return false;                /* the making raised */
    bool ok = true;
    sys__heap_node at, next, stack;
    bool first = true;
    while (ok && mine != 0ull &&
           sys__dictionary__zzpackage_after(mine, first ? (const sys__heap_node*)0 : &at, &next, &stack)) {
        first = false;
        at = next;
        if (!sys__bindings__zzprivate_scoped(&stack)) continue;
        const sys__heap_node top = sys__stack__peek(&stack);           /* comes back held */
        ok = sys__dictionary__put(past, &at, &top);
        if (sys__heap_node__carries_reference(top.dtype)) (void)sys__heap_object__release(top.args[0]);
    }
    if (ok && sys__dictionary__count(past) != 0ull) {
        const sys__heap_node reference = sys__heap_object__reference_to(past);
        ok = sys__dictionary__seal(past) && sys__node_array__set(picture, SYS__BINDINGS__HATCH, &reference);
    }
    (void)sys__heap_object__release(past);         /* the picture holds it, or nothing does */
    return ok;
}

static __device__ inline uint64_t sys__bindings__create_viewonly_array(uint64_t bindings) {
    const uint64_t symbols = sys__bindings__symbols(bindings);
    if (symbols == 0ull) return 0ull;              /* it raised whichever of the two it was */

    const uint64_t picture = sys__node_array__zzpackage_create_run(symbols + SYS__BINDINGS__FIRST_NAME);
    if (picture == 0ull) return 0ull;

    /* ⭐ THE WALK IS WHY THE OBJECT IS A HEADER OVER TWO TABLES. ⚖ ARCHITECT: *"this should make it
     * easier to build the viewonly object from the main bindings object."* Every name's answer is
     * already what a read of this environment returns, so this is that read once per name — the scopes,
     * then the base, then the unbound error — and nothing here has to know which of the three it got. */
    sys__heap_node name;
    name.dtype = SYS__KIND__QUOTED_NAME; name.num_args = 0u; name.op_code = 0u; name.args[0] = 0ull;
    for (uint64_t i = 0ull; i < symbols; ++i) {
        name.args[0] = i;
        sys__heap_node value = sys__bindings__get(bindings, &name);
        const bool placed = sys__node_array__set(picture, i + SYS__BINDINGS__FIRST_NAME, &value);
        /* The picture took its own hold on whatever went in, so the one the read handed over is given
         * back here — on both paths, because a value that could not be placed is still one this holds. */
        if (sys__heap_node__carries_reference(value.dtype)) (void)sys__heap_object__release(value.args[0]);
        if (!placed) { (void)sys__heap_object__release(picture); return 0ull; }
    }
    if (!sys__bindings__zzprivate_picture_spelled(bindings, picture)) {
        (void)sys__heap_object__release(picture);
        return 0ull;
    }
    return picture;
}

static __device__ __noinline__ void sys__bindings__zzpackage_release_internal(sys__heap_node* head,
                                                                       uint64_t* releaser_stack) {
    if (head == 0) return;
    /* Three references move and none is let go of, for the reason every row here moves rather than
     * releases: letting go would unwind from inside this function, one C frame per level of structure,
     * and a scope table of stacks of environments is exactly that shape. Moving keeps the unwinding one
     * loop at one depth however deep it goes.
     *
     * ⛳ THE BASE MOVES LIKE THE OTHER TWO EVEN THOUGH IT IS SOMEBODY ELSE'S. What this object holds is
     * one hold on it, and giving that up is all that happens — the table itself goes only if this was
     * the last environment naming it, which is the count's business and not this function's. */
    const uint64_t held[3] = { head[1].args[SYS__BINDINGS__SCOPES],
                               head[1].args[SYS__BINDINGS__BASE],
                               head[1].args[SYS__BINDINGS__UNBOUND] };
    head[1].args[SYS__BINDINGS__SCOPES]  = 0ull;
    head[1].args[SYS__BINDINGS__BASE]    = 0ull;
    head[1].args[SYS__BINDINGS__UNBOUND] = 0ull;

    for (unsigned int i = 0u; i < 3u; ++i) {
        if (held[i] == 0ull) continue;
        sys__heap_node reference = sys__heap_object__reference_to(held[i]);
        sys__stack__transfer(releaser_stack, &reference);
    }
}

static __device__ inline sys__heap_node sys__bindings__zzpackage_construct(sys__heap_node* base,
                                                                       const sys__heap_node* parameters) {
    (void)base;
    if (parameters == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return sys__heap_node__nothing();
    }
    /* How many names, and the table to sit over — the second being zero for an environment with nothing
     * underneath it, which is what a program writes when it has no base to name. */
    const uint64_t at = sys__bindings__create(parameters->args[0], parameters->args[1]);
    if (at == 0ull) return sys__heap_node__nothing();          /* the making already raised */
    return sys__heap_object__reference_to(at);
}

/* ⭐ THE ENVIRONMENT A BINDING FORM NAMES — `env`, or when it is 0 the environment of the computation running the form,
 * which its computing base keeps. ⚖ *"ok with your fix"*: a program composed on the host named the HOST's environment
 * in every `let`, so a block handed the picture bound into the host's names — a race with block 0 — and read its own,
 * where nothing was bound. 0 lets a picture bind wherever it runs. */
static __device__ inline uint64_t sys__bindings__zzpackage_named(const sys__heap_node* base, uint64_t env) {
    if (env != 0ull) return env;
    return base == 0 ? 0ull : base->args[SYS__COMPUTING_BASE__BINDINGS];
}

static __device__ __noinline__ void sys__bindings__zzpackage_apply_let(sys__heap_node* base, uint64_t form) {
    /* The verb, the environment, a name and a value for each binding, and the body: an odd length, and at
     * least one pair. */
    const uint64_t n = sys__sublist__length(form);
    if (n < 5ull || (n % 2ull) == 0ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    /* ⛳ RESOLVED ONCE, and the unbinding below is built with what it resolved to, so the scope is left where it
     *   was entered even if the form is read again */
    const uint64_t       env  = sys__bindings__zzpackage_named(base, sys__sublist__nth(form, 1ull).args[0]);
    const sys__heap_node body = sys__sublist__nth(form, n - 1ull);
    if (body.dtype != SYS__KIND__QUOTED_LIST) {
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
        return;
    }
    for (uint64_t i = 2ull; i + 1ull < n - 1ull; i += 2ull)
        if (!sys__bindings__quoted(sys__sublist__nth(form, i).dtype)) {
            sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
            return;
        }

    /* The unbinding, built before the first name is bound. It takes every name at once, so a scope of four
     * names ends in one form and one cycle of the evaluator's loop. */
    const uint64_t drop = sys__list__create();
    if (drop == 0ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE); return; }
    sys__heap_node verb;
    verb.dtype = SYS__KIND__STANDARD; verb.num_args = 0u; verb.args[0] = 0ull;
    verb.op_code = SYS__OPCODES__REMOVE_BINDINGS;
    sys__heap_node where;
    where.dtype = SYS__KIND__VALUE_INT; where.num_args = 0u; where.args[0] = env;
    bool ok = sys__sublist__append(drop, &verb) && sys__sublist__append(drop, &where);
    for (uint64_t i = 2ull; ok && i + 1ull < n - 1ull; i += 2ull) {
        const sys__heap_node name = sys__sublist__nth(form, i);
        ok = sys__sublist__append(drop, &name);
    }
    if (!ok) {
        (void)sys__heap_object__release(drop);
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
        return;
    }

    /* Each name enters its scope now, reading its value out of the cell it is still standing in. */
    for (uint64_t i = 2ull; i + 1ull < n - 1ull; i += 2ull) {
        const sys__heap_node name  = sys__sublist__nth(form, i);
        const sys__heap_node value = sys__sublist__nth(form, i + 1ull);
        (void)sys__bindings__add(env, &name, &value);
    }

    /* And the form becomes what is left to do. ⭐ The body is UNQUOTED WHERE IT STANDS — the same list, now
     * something the scan descends into — so nothing is copied and no room is asked for. The order is the
     * correctness: the body is held by its new cell before the old one is discarded.
     * ⛳ And the values may be written over, because the scopes above are holding them now. */
    sys__heap_node head = verb;
    head.op_code = SYS__OPCODES__PROG1;
    (void)sys__sublist__replace(form, 0ull, &head);
    sys__heap_node run = body;
    run.dtype = SYS__KIND__OBJECT_REFERENCE;
    (void)sys__sublist__replace(form, 1ull, &run);
    const sys__heap_node after = sys__heap_object__reference_to(drop);
    (void)sys__sublist__replace(form, 2ull, &after);
    (void)sys__heap_object__release(drop);          /* the form holds it now */
    (void)sys__sublist__discard(form, 3ull, n - 3ull);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_BINDINGS__IMPL_CUH */
