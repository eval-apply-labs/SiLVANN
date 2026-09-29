#ifndef SILVANN__PACKAGES_SYS_CPU_LIST__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_LIST__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"        /* how it says a caller asked for something that is not there */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "heap_object__header.cuh"   /* the head behind a reference, holds, and what a kind answers */
#include "heap__header.cuh"         /* the carve, and turning an offset back into an address */
#include "node_array__header.cuh"  /* what a read-only picture of a list is */
#include "stack__header.cuh"  /* where a dying object moves what it was holding, and a copy's work */
#include "list__header.cuh"         /* the verbs it fills in, and the three fault words they raise */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * The history behind the lazy thaw below, so nobody re-does it.
 *   · The first `sys__list__thaw_running` delegated to `zzprivate_thaw` and cost 11%; the next had
 *     `thaw_running_into` and `thaw_running` calling each other — mutual recursion, live in the shipping
 *     build, red in `list_call_graph_acyclic`. The work-stack-on-first-need loop replaced both.
 *   · The 11% is a `k_eval` figure, and the evaluator has left the card (⚖ *"sys and the
 *     evaluator go host"*). The shape of the argument — a stack per FORM against one on first need —
 *     survives any processor; the percentage does not. Its artifact:
 *     `measurements/2026-09-23_S107_pricing_the_list_cycle.md`.
 *   · The ordinary comments below carried that 11% as `MEASURED` (the lazy thaw's own paragraph and
 *     `thaw_running_into`'s), with "a 2x turns into a 0.9x"; it lives here now and nowhere else.
 *   · The walk census rewrote `nth`, `replace`, `append`, `room` and `freeze`'s two inner
 *     loops: each had re-walked the chunk chain for a fact it already held. `freeze`'s per-cell loop over
 *     `nth(source, k)` was half its cost, and its `seen` scan went CUBIC through `nth`. `MEASURED`
 *     building a list by appending was the quadratic in `freeze`.
 * ⛳ RETIREMENT: when the evaluator runs on the host and the lazy thaw has been re-priced there.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


/* ⛳ Shipping default, outside any internal marker — a lever is an override, and deleting an override
 * must leave the value the engine ships with. ▶ `sys__list__thaw_running`. */
#ifndef SILVANN_LAZY_THAW
#define SILVANN_LAZY_THAW 1
#endif
#ifndef SILVANN_ABLATE_WALK_GUARD
#define SILVANN_ABLATE_WALK_GUARD 0
#endif

/* ── REACHING THE PARTS ──────────────────────────────────────────────────────────────────────────────
 * A reference names the node after the head, so a chunk's values start where the reference points and
 * its two words are one node behind. These three are here rather than written out at each use because
 * everything below would otherwise have had to get the same two addresses right.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__heap_node* sys__list_chunk__zzprivate_cells(uint64_t chunk) {
    return sys__heap__object_full_address(chunk);
}
static __device__ inline uint64_t sys__list_chunk__zzprivate_used(uint64_t chunk) {
    return sys__heap_node__zzpackage_head(chunk)->args[SYS__LIST_CHUNK__HEAD_USED];
}
static __device__ inline uint64_t sys__list_chunk__zzprivate_next(uint64_t chunk) {
    return sys__heap_node__zzpackage_head(chunk)->args[SYS__LIST_CHUNK__HEAD_NEXT];
}

/* A fresh chunk, empty and linked to nothing. Its cells are whatever the last occupant left, which is
 * safe for exactly one reason: nothing ever reads past `used`, and `used` starts at none. */
static __device__ inline uint64_t sys__list_chunk__zzprivate_create(void) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__LIST_CHUNK);
    if (head == 0) return 0ull;                    /* the carve raised, and said which of its reasons */
    head->args[SYS__LIST_CHUNK__HEAD_USED] = 0ull;
    head->args[SYS__LIST_CHUNK__HEAD_NEXT] = 0ull;
    return sys__heap__offset(head + 1);
}

static __device__ inline bool sys__sublist__is(uint64_t sublist) {
    if (!sys__heap_object__zzpackage_addressable(sublist)) return false;
    return sys__heap_object__is_type_from_head(sys__heap_node__zzpackage_head(sublist),
                                              SYS__KIND__SUBLIST);
}

/* The sublist's own two words, and the store's. Both raise where they refuse, which is why none of the
 * verbs below does. */
static __device__ inline sys__heap_node* sys__sublist__zzprivate_fields(uint64_t sublist) {
    if (!sys__sublist__is(sublist)) {
        sys__fault__raise(0ull, SYS__LIST__FAULT_KIND);
        return 0;
    }
    return sys__heap__object_full_address(sublist);
}
static __device__ inline sys__heap_node* sys__list__zzprivate_store_fields(uint64_t store) {
    return sys__heap__object_full_address(store);
}

/* How many values the whole list holds. ⭐⭐ A READ, NOT A WALK — ▶ `SYS__LIST__STORE_TOTAL` for the
 * invariant and the six sites that keep it. `sys__sublist__length` is this, less a view's start.
 * ⇒ ★ THE CHEAPEST TRAVERSAL IS THE ONE NOBODY MAKES. */
static __device__ inline uint64_t sys__list__zzprivate_total(uint64_t store) {
    return sys__list__zzprivate_store_fields(store)->args[SYS__LIST__STORE_TOTAL];
}

/* Which chunk a position falls in, where in it, and what comes before that chunk — the last of the three
 * because a chunk that empties has to be unlinked, and the only thing that can unlink it is whatever
 * points at it. Answers false for a position the list does not reach; it raises nothing, because two
 * callers want that answer for different reasons and only one of them is a mistake. */
static __device__ inline bool sys__list__zzprivate_locate(uint64_t store, uint64_t position,
                                                          uint64_t* chunk, uint64_t* index, uint64_t* before) {
    uint64_t at   = sys__list__zzprivate_store_fields(store)->args[SYS__LIST__STORE_VALUES];
    uint64_t prev = 0ull;
    uint64_t base = 0ull;
    while (at != 0ull) {
        const uint64_t used = sys__list_chunk__zzprivate_used(at);
        if (position < base + used) {
            *chunk = at; *index = position - base; *before = prev;
            return true;
        }
        base += used;
        prev = at;
        at   = sys__list_chunk__zzprivate_next(at);
    }
    return false;
}

/* ── A WALK ──────────────────────────────────────────────────────────────────────────────────────────
 * ▶ the header for what a walk is for and what ends one. Three functions, and between them they hold the
 * whole difference between reading a list forward and asking it `n` separate questions.
 * ⛳ THE CHUNK IS ENTERED ONCE: `used` is read when the walk arrives in a chunk and not per step, which
 * is the second traversal `nth` made and the one nobody counts — `locate` reads every chunk's `used` on
 * its way past, so a loop over `nth` reads the SAME word `n²/2/SLOTS` times.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* Put the walk in whichever chunk `at` falls in, starting from the chunk it is already in. `from_head`
 * makes it start over, which is what opening one does. Nothing else ever needs to. */
static __device__ inline void sys__list__zzprivate_walk_settle(sys__list_walk* w) {
    while (w->chunk != 0ull && w->at >= w->base + w->used) {
        w->base += w->used;
        w->chunk = sys__list_chunk__zzprivate_next(w->chunk);
        /* ⛳ THE LOOP RATHER THAN AN `if` IS FOR THE EMPTY CHUNK. A chunk emptied by `discard` or
         * `extract` is unlinked — but on a one-slot dial (`SYS__LIST_CHUNK__ALLOC` 2) `split` leaves one
         * holding nothing, and `locate` steps past such a chunk without comment, so a walk that did not
         * would answer nothing in the middle of a list with values in it. */
        w->used  = (w->chunk != 0ull) ? sys__list_chunk__zzprivate_used(w->chunk) : 0ull;
    }
}

static __device__ inline bool sys__sublist__walk(uint64_t sublist, uint64_t from, sys__list_walk* w) {
    if (w == 0) { sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE); return false; }
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);   /* raises "not one" for us */
    if (f == 0) { w->store = 0ull; w->chunk = 0ull; w->used = 0ull; w->at = 0ull; w->base = 0ull;
                  w->total = 0ull; return false; }
    const uint64_t store = f->args[SYS__SUBLIST__STORE];
    w->store = store;
    w->at    = f->args[SYS__SUBLIST__START] + from;
    w->base  = 0ull;
    w->chunk = sys__list__zzprivate_store_fields(store)->args[SYS__LIST__STORE_VALUES];
    w->used  = (w->chunk != 0ull) ? sys__list_chunk__zzprivate_used(w->chunk) : 0ull;
    w->total = sys__list__zzprivate_total(store);
    sys__list__zzprivate_walk_settle(w);
    return true;
}

static __device__ inline sys__heap_node* sys__list__walk_cell(sys__list_walk* w) {
    if (w == 0 || w->chunk == 0ull) return 0;
    /* ⛔ THE GUARD, AND IT IS READ HERE RATHER THAN ON THE STEP because this is the line that would hand
     * back a pointer into a chunk that has moved. A stale walk answers END, which is what every caller
     * already handles — the alternative is a raise, and a walk is engine machinery with no program to
     * tell. ▶ the header for what this catches and the one thing it does not. */
    if (sys__list__zzprivate_total(w->store) != w->total) return 0;
    return &sys__list_chunk__zzprivate_cells(w->chunk)[w->at - w->base];
}

static __device__ inline void sys__list__walk_step(sys__list_walk* w) {
    if (w == 0 || w->chunk == 0ull) return;
    w->at += 1ull;
    /* ⭐ THE ORDINARY STEP TOUCHES NOTHING BUT `at`, AND THAT IS THE WHOLE POINT OF THE THING. Fourteen
     * steps in fifteen are one addition and one compare against words already in registers; the
     * fifteenth follows ONE link. */
    sys__list__zzprivate_walk_settle(w);
}

/* ── THE ROLL ────────────────────────────────────────────────────────────────────────────────────────
 * The store's second chain, holding one entry per view that is looking at it. An entry is a BARE OFFSET
 * and takes no hold, which would otherwise be a cycle — a view holds the store, and a store holding its
 * views back would mean neither ever reached zero.
 * ⛔ WHAT MAKES AN UNCOUNTED ENTRY SAFE IS THAT A VIEW TAKES ITS OWN OFF WHEN IT DIES, and what makes
 * THAT safe without a lock is that a list has one owner, so nothing is walking the roll at that moment.
 * The two facts hold each other up; neither is true on its own.
 * ⛳ A SEAT IS FLAT — chunk ordinal times the slots in one, plus the index — so a view remembers where its
 * entry is with one word and never searches for it.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* Take a seat and answer which one. Reuses one a dead view left behind before it grows the chain, so a
 * walk that takes a million views and drops them does not leave a million entries to step over. */
static __device__ inline bool sys__list__zzprivate_enrol(uint64_t store, uint64_t view, uint64_t* seat) {
    sys__heap_node* fields = sys__list__zzprivate_store_fields(store);
    uint64_t at      = fields->args[SYS__LIST__STORE_ROLL];
    uint64_t prev    = 0ull;
    uint64_t ordinal = 0ull;

    while (at != 0ull) {
        sys__heap_node* head  = sys__heap_node__zzpackage_head(at);
        sys__heap_node* cells = sys__list_chunk__zzprivate_cells(at);
        const uint64_t used = head->args[SYS__LIST_CHUNK__HEAD_USED];
        for (uint64_t i = 0ull; i < used; ++i) {
            if (cells[i].dtype != SYS__KIND__VALUE_NULL) continue;
            cells[i].dtype = SYS__KIND__VALUE_INT;  cells[i].args[0] = view;
            *seat = ordinal * (uint64_t)SYS__LIST_CHUNK__SLOTS + i;
            return true;
        }
        if (used < (uint64_t)SYS__LIST_CHUNK__SLOTS) {   /* room past the high-water mark */
            cells[used].dtype = SYS__KIND__VALUE_INT;  cells[used].num_args = 0u;  cells[used].args[0] = view;
            head->args[SYS__LIST_CHUNK__HEAD_USED] = used + 1ull;
            *seat = ordinal * (uint64_t)SYS__LIST_CHUNK__SLOTS + used;
            return true;
        }
        prev = at;
        at   = sys__list_chunk__zzprivate_next(at);
        ordinal += 1ull;
    }

    const uint64_t fresh = sys__list_chunk__zzprivate_create();
    if (fresh == 0ull) return false;                     /* the carve already raised */
    sys__heap_node* cells = sys__list_chunk__zzprivate_cells(fresh);
    cells[0].dtype = SYS__KIND__VALUE_INT;  cells[0].num_args = 0u;  cells[0].args[0] = view;
    sys__heap_node__zzpackage_head(fresh)->args[SYS__LIST_CHUNK__HEAD_USED] = 1ull;
    if (prev == 0ull) fields->args[SYS__LIST__STORE_ROLL] = fresh;
    else              sys__heap_node__zzpackage_head(prev)->args[SYS__LIST_CHUNK__HEAD_NEXT] = fresh;
    *seat = ordinal * (uint64_t)SYS__LIST_CHUNK__SLOTS;
    return true;
}

/* Give a seat back. The entry becomes a null, which is what `enrol` looks for and what the adjust walk
 * steps over — so an empty seat needs no marker of its own. */
static __device__ inline void sys__list__zzprivate_leave(uint64_t store, uint64_t seat) {
    uint64_t at      = sys__list__zzprivate_store_fields(store)->args[SYS__LIST__STORE_ROLL];
    uint64_t ordinal = seat / (uint64_t)SYS__LIST_CHUNK__SLOTS;
    while (at != 0ull && ordinal != 0ull) { at = sys__list_chunk__zzprivate_next(at); ordinal -= 1ull; }
    if (at == 0ull) return;                              /* a seat on a roll that is gone */
    sys__heap_node* cell = &sys__list_chunk__zzprivate_cells(at)[seat % (uint64_t)SYS__LIST_CHUNK__SLOTS];
    cell->dtype = SYS__KIND__VALUE_NULL;  cell->args[0] = 0ull;
}

/* ⛔⛔ THE WALK THAT IS THE MUTATION. Every view that begins ABOVE where something happened moves by
 * however many went in or came out, so it keeps naming the values it was naming. A view that begins
 * exactly there is left alone, and that is the whole of why removal needs no case of its own:
 * everything above the hole slides down, so an unchanged start already names what came next.
 * ⛔ SKIPPING THIS IS NOT A SLOWER LIST, IT IS A WRONG ONE, and nothing fails when it happens. */
static __device__ inline void sys__list__zzprivate_adjust(uint64_t store, uint64_t at_position, bool up,
                                                          bool displaced, uint64_t count) {
    uint64_t at = sys__list__zzprivate_store_fields(store)->args[SYS__LIST__STORE_ROLL];
    while (at != 0ull) {
        sys__heap_node*    cells = sys__list_chunk__zzprivate_cells(at);
        const uint64_t used  = sys__list_chunk__zzprivate_used(at);
        for (uint64_t i = 0ull; i < used; ++i) {
            if (cells[i].dtype != SYS__KIND__VALUE_INT) continue;      /* an empty seat */
            sys__heap_node* view = sys__heap__object_full_address(cells[i].args[0]);
            const uint64_t start = view->args[SYS__SUBLIST__START];
            /* ⭐ ONE PRINCIPLE, AND THE COMPARISON FOLLOWS FROM IT RATHER THAN BEING CHOSEN: A VIEW KEEPS
             * NAMING THE VALUE IT NAMED.
             *   putting one in WHERE ONE WAS   pushes that value along, and every view standing on it or
             *                                  beyond follows it. Nobody's list gains a value at its front.
             *   putting one in PAST THE END    displaces nothing, because nothing was there — so no view
             *                                  was naming it and none has to move. A view that had reached
             *                                  the end is the rest of the list, and the rest of a list that
             *                                  just grew is the new value.
             *   taking one out                 the view standing on it has nothing left to follow, so it
             *                                  stays where it is — already the next value — and only the
             *                                  ones beyond it move.
             * ⛔ THE MIDDLE CASE IS WHY THIS TAKES A SECOND ARGUMENT, and getting it wrong is not subtle:
             * treating an append as an insert pushes the appender's own view along with everything else,
             * so a list can never be built at all. */
            /* ⛳ A RANGE GENERALISES THE ONE-VALUE RULE RATHER THAN NEEDING ITS OWN. A view standing
             * BEYOND what went follows it back by however many went; a view standing INSIDE the range has
             * nothing left to follow, so it stops where the range began — which is already the next value,
             * exactly as a single removal leaves a view standing on it. At `count == 1` the middle case
             * cannot arise and this is the rule it replaced, unchanged. */
            if (up) {
                if (displaced ? (start >= at_position) : (start > at_position))
                    view->args[SYS__SUBLIST__START] = start + count;
                continue;
            }
            if (start >= at_position + count) view->args[SYS__SUBLIST__START] = start - count;
            else if (start > at_position)     view->args[SYS__SUBLIST__START] = at_position;
        }
        at = sys__list_chunk__zzprivate_next(at);
    }
}

/* ── MAKING ONE, AND MAKING A VIEW OVER ONE ──────────────────────────────────────────────────────────
 * A view is made in one place, so what it takes a hold of and what it puts on the roll cannot come apart.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* A view over `store` at `start`. The store must already be HELD by whoever is asking, and the hold this
 * takes is a second one — so a caller that is handing its own over releases it afterwards. */
static __device__ inline uint64_t sys__list__zzprivate_view(uint64_t store, uint64_t start) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__SUBLIST);
    if (head == 0) return 0ull;
    head[1].args[SYS__SUBLIST__STORE] = 0ull;
    head[1].args[SYS__SUBLIST__START] = start;
    head[1].args[SYS__SUBLIST__SEAT]  = 0ull;
    const uint64_t view = sys__heap__offset(head + 1);

    uint64_t seat = 0ull;
    if (!sys__list__zzprivate_enrol(store, view, &seat)) { (void)sys__heap_object__release(view); return 0ull; }
    head[1].args[SYS__SUBLIST__SEAT]  = seat;
    /* The store is written LAST, because it is what the teardown reads to find the roll it must leave —
     * and a view that failed to take a seat must not be holding a store it never registered with. */
    (void)sys__heap_object__retain(store);
    head[1].args[SYS__SUBLIST__STORE] = store;
    return view;
}

static __device__ inline uint64_t sys__list__create(void) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__LIST);
    if (head == 0) return 0ull;
    head[1].args[SYS__LIST__STORE_VALUES] = 0ull;
    head[1].args[SYS__LIST__STORE_ROLL]   = 0ull;
    head[1].args[SYS__LIST__STORE_TOTAL]  = 0ull;   /* ▶ the invariant note in the header */
    head[1].args[SYS__LIST__STORE_TAIL]   = 0ull;
    const uint64_t store = sys__heap__offset(head + 1);

    const uint64_t view = sys__list__zzprivate_view(store, 0ull);
    /* Whatever happened, the hold the making left is this function's and is given back here: on success
     * the view has taken its own and the store is the view's alone, which is what makes the store die
     * with the last view rather than outliving every one of them. */
    (void)sys__heap_object__release(store);
    return view;
}

static __device__ inline uint64_t sys__list__create_executable(const sys__heap_node* cells, uint64_t count) {
    if (cells == 0 || count == 0ull) {
        sys__fault__raise(0ull, SYS__LIST__FAULT_KIND);
        return 0ull;
    }
    const uint64_t list = sys__list__create();
    if (list == 0ull) return 0ull;
    for (uint64_t i = 0ull; i < count; ++i) {
        if (!sys__sublist__append(list, &cells[i])) {
            /* ⛳ THE LIST GOES AND THE CALLER'S CELLS DO NOT. Whatever was appended is held by the list
             * that is about to die, so letting it go undoes exactly what this call did — and the caller
             * still owns every cell it handed over, on the refusal path as on the other one. There is no
             * half-state to work out, because nothing of the caller's was ever taken. */
            (void)sys__heap_object__release(list);
            return 0ull;
        }
    }
    return list;
}

/* ── WHAT A VIEW IS ──────────────────────────────────────────────────────────────────────────────── */

static __device__ inline uint64_t sys__sublist__length(uint64_t sublist) {
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return 0ull;
    const uint64_t total = sys__list__zzprivate_total(f->args[SYS__SUBLIST__STORE]);
    const uint64_t start = f->args[SYS__SUBLIST__START];
    return (start >= total) ? 0ull : (total - start);
}

static __device__ inline bool sys__sublist__empty(uint64_t sublist) {
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return true;
    uint64_t chunk = 0ull, index = 0ull, before = 0ull;
    /* It asks whether anything is THERE rather than how much, so it stops at the first chunk that
     * reaches the position instead of summing the chain. */
    return !sys__list__zzprivate_locate(f->args[SYS__SUBLIST__STORE], f->args[SYS__SUBLIST__START],
                                        &chunk, &index, &before);
}

/* ── READING ─────────────────────────────────────────────────────────────────────────────────────── */

/* The value at `position`, borrowed. One place, so `car` and `nth` cannot come to different conclusions
 * about the same list and the same distance. */
static __device__ inline sys__heap_node sys__list__zzprivate_at(uint64_t store, uint64_t position) {
    uint64_t chunk = 0ull, index = 0ull, before = 0ull;
    if (!sys__list__zzprivate_locate(store, position, &chunk, &index, &before)) return sys__heap_node__nothing();
    return sys__list_chunk__zzprivate_cells(chunk)[index];
}

static __device__ inline sys__heap_node sys__sublist__car(uint64_t sublist) {
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return sys__heap_node__nothing();
    /* Reaching the end is how a walk finds out it is over, so this refuses nothing and raises nothing. */
    return sys__list__zzprivate_at(f->args[SYS__SUBLIST__STORE], f->args[SYS__SUBLIST__START]);
}

static __device__ inline sys__heap_node sys__sublist__nth(uint64_t sublist, uint64_t n) {
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return sys__heap_node__nothing();
    /* ⭐⭐ THE RANGE TEST IS `locate`'s OWN ANSWER. `locate` walks until the position falls inside a
     * chunk and answers false when it never does — and `n >= length` is true exactly when
     * `START + n >= total`, which is exactly when `locate` fails. ⇒ ★★ ONE WALK ANSWERS BOTH QUESTIONS.
     * Every verb's argument read goes through here (the gather in `verb_abi.cuh`). An index is a claim
     * about how far the list goes, and a wrong one is a mistake that `locate` notices.
     * ⛳ AND IT IS WHAT SEPARATES THIS FROM `car`, which wants the same failure to be silent. */
    uint64_t chunk = 0ull, index = 0ull, before = 0ull;
    if (!sys__list__zzprivate_locate(f->args[SYS__SUBLIST__STORE],
                                     f->args[SYS__SUBLIST__START] + n, &chunk, &index, &before)) {
        sys__fault__raise(0ull, SYS__LIST__FAULT_RANGE);
        return sys__heap_node__nothing();
    }
    return sys__list_chunk__zzprivate_cells(chunk)[index];
}

static __device__ inline uint64_t sys__sublist__cdr(uint64_t sublist) {
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return 0ull;
    /* The rest of nothing is nothing, and it is a view rather than a refusal: a walk that has arrived at
     * the end asks for the rest one more time in the ordinary course of ending. */
    const uint64_t start = f->args[SYS__SUBLIST__START];
    const uint64_t next  = sys__sublist__empty(sublist) ? start : (start + 1ull);
    return sys__list__zzprivate_view(f->args[SYS__SUBLIST__STORE], next);
}

/* ── CHANGING IT ─────────────────────────────────────────────────────────────────────────────────── */

/* ⚖ SPLIT EVENLY (architect). Half stays, half moves to a fresh chunk linked after it, and the values
 * MOVE — a cell copied from one chunk to another changes which chunk owns it and no count moves, so this
 * spends no atomics however many values cross.
 * ⛳ DOWN THE MIDDLE rather than at the insertion point, so there is room on BOTH sides and a second
 * insert near the first does not split again. Nothing depends on where the cut falls. */
static __device__ inline bool sys__list__zzprivate_split(uint64_t store, uint64_t chunk) {
    const uint64_t fresh = sys__list_chunk__zzprivate_create();
    if (fresh == 0ull) return false;
    sys__heap_node* head  = sys__heap_node__zzpackage_head(chunk);
    sys__heap_node* cells = sys__list_chunk__zzprivate_cells(chunk);
    sys__heap_node* to    = sys__list_chunk__zzprivate_cells(fresh);

    const uint64_t used = head->args[SYS__LIST_CHUNK__HEAD_USED];
    const uint64_t keep = used / 2ull;
    for (uint64_t i = keep; i < used; ++i) to[i - keep] = cells[i];

    sys__heap_node__zzpackage_head(fresh)->args[SYS__LIST_CHUNK__HEAD_USED] = used - keep;
    sys__heap_node__zzpackage_head(fresh)->args[SYS__LIST_CHUNK__HEAD_NEXT] = head->args[SYS__LIST_CHUNK__HEAD_NEXT];
    head->args[SYS__LIST_CHUNK__HEAD_USED] = keep;
    head->args[SYS__LIST_CHUNK__HEAD_NEXT] = fresh;
    /* ⛳ THE CELLS MOVED AND THE COUNT DID NOT, so `TOTAL` is untouched — but if the chunk that split
     * was the LAST one, the fresh half is now the last. `fresh.NEXT` inherited `chunk.NEXT`, so a zero
     * there is exactly the test. */
    if (sys__list_chunk__zzprivate_next(fresh) == 0ull)
        sys__list__zzprivate_store_fields(store)->args[SYS__LIST__STORE_TAIL] = fresh;
    return true;
}

/* The chunk a value goes into at `position`, splitting or growing the chain if there is no room. Answers
 * where to write, so the caller does the writing and the accounting in one place. */
/* ⛳ `at_end` IS THE CALLER SAYING IT ALREADY KNOWS `locate` WOULD FAIL. An append's position IS the
 * store's total, so the search below can only walk the whole chain and come back empty — and then the
 * end path walks it AGAIN to find the last chunk. Two traversals to reach a place the caller had already
 * named. ⇒ ★ WHEN A CALLER KNOWS THE ANSWER TO A SEARCH, THE SEARCH IS A COST WITH NO INFORMATION IN IT.
 * ⛔ A caller passing `true` wrongly would insert at the end instead of in the middle, so it is only
 * ever `position == total`, which `place_at` has in hand. */
static __device__ inline bool sys__list__zzprivate_room(uint64_t store, uint64_t position, bool at_end,
                                                        uint64_t* chunk, uint64_t* index) {
    sys__heap_node* fields = sys__list__zzprivate_store_fields(store);
    uint64_t before = 0ull;

    if (at_end || !sys__list__zzprivate_locate(store, position, chunk, index, &before)) {
        /* Past every value: it goes after the last one. An empty list has no last one, so this is also
         * where the first chunk comes from.
         * ⭐⭐ THE LAST CHUNK IS READ, NOT HUNTED — ▶ `SYS__LIST__STORE_TAIL`. A walk to the last chunk on
         * every append is quadratic in a bulk build, which is what `freeze` is. */
        const uint64_t last = fields->args[SYS__LIST__STORE_TAIL];
        if (last != 0ull && sys__list_chunk__zzprivate_used(last) < (uint64_t)SYS__LIST_CHUNK__SLOTS) {
            *chunk = last; *index = sys__list_chunk__zzprivate_used(last);
            return true;
        }
        const uint64_t fresh = sys__list_chunk__zzprivate_create();
        if (fresh == 0ull) return false;
        if (last == 0ull) fields->args[SYS__LIST__STORE_VALUES] = fresh;
        else              sys__heap_node__zzpackage_head(last)->args[SYS__LIST_CHUNK__HEAD_NEXT] = fresh;
        fields->args[SYS__LIST__STORE_TAIL] = fresh;    /* it is linked at the end, so it IS the end */
        *chunk = fresh; *index = 0ull;
        return true;
    }

    if (sys__list_chunk__zzprivate_used(*chunk) < (uint64_t)SYS__LIST_CHUNK__SLOTS) return true;

    /* Full. After the split the position is in one half or the other, and asking again is how we find
     * out which — cheaper to re-walk two chunks than to reason about which side the cut left it on. */
    if (!sys__list__zzprivate_split(store, *chunk)) return false;
    return sys__list__zzprivate_locate(store, position, chunk, index, &before) ||
           (*chunk = 0ull, *index = 0ull, false);
}

/* The work both of them do. It is split out because the two differ only in what they ANSWER, and an
 * append that went through the public insert would make a view and drop it on every value — which a bulk
 * build does once per element, for nothing. */
/* ⭐⭐⭐ THE BODY BOTH ENTRANCES SHARE, TAKING THE POSITION AND THE TOTAL ALREADY KNOWN. Splitting it out
 * is what lets `append` skip a question it is the answer to: appending means `position == total`.
 * ⇒ ★★ AN ENTRY POINT THAT RE-DERIVES WHAT ITS CALLER ALREADY HOLDS IS A LOOP NOBODY WROTE.
 * ▶ `sys__sublist__append`, and the walk census in `NN-15`. */
static __device__ inline bool sys__list__zzprivate_place_at(uint64_t store, uint64_t position,
                                                            uint64_t total, const sys__heap_node* value,
                                                            uint64_t* at) {
    const bool displaced = position < total;

    uint64_t chunk = 0ull, index = 0ull;
    if (!sys__list__zzprivate_room(store, position, !displaced, &chunk, &index)) return false;

    sys__heap_node*    cells = sys__list_chunk__zzprivate_cells(chunk);
    sys__heap_node*    head  = sys__heap_node__zzpackage_head(chunk);
    const uint64_t used  = head->args[SYS__LIST_CHUNK__HEAD_USED];
    for (uint64_t i = used; i > index; --i) cells[i] = cells[i - 1ull];
    cells[index] = *value;
    head->args[SYS__LIST_CHUNK__HEAD_USED] = used + 1ull;
    /* The list is now an owner of what went in. The cell it landed in held nothing that was ours — it was
     * either past the end or a value that shifted up — so nothing gives a hold up here. */
    if (sys__heap_node__carries_reference(value->dtype)) (void)sys__heap_object__retain(value->args[0]);
    /* ⛳ ONE VALUE WENT IN, SO THE STORE HOLDS ONE MORE — ▶ the invariant note in the header. */
    sys__list__zzprivate_store_fields(store)->args[SYS__LIST__STORE_TOTAL] = total + 1ull;

    /* Whether anything stood here is read BEFORE the write, because after it something does. */
    sys__list__zzprivate_adjust(store, position, true, displaced, 1ull);
    if (at != 0) *at = position;
    return true;
}

/* The ordinary entrance: a position RELATIVE to the view, range-checked against the one total it costs. */
static __device__ inline bool sys__list__zzprivate_place(uint64_t sublist, uint64_t n,
                                                         const sys__heap_node* value, uint64_t* at) {
    if (value == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return false;
    }
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return false;
    const uint64_t store    = f->args[SYS__SUBLIST__STORE];
    const uint64_t total    = sys__list__zzprivate_total(store);
    const uint64_t position = f->args[SYS__SUBLIST__START] + n;
    if (position > total) {          /* AT the end is where a list grows; past it is not */
        sys__fault__raise(0ull, SYS__LIST__FAULT_RANGE);
        return false;
    }
    return sys__list__zzprivate_place_at(store, position, total, value, at);
}

/* ⭐ THE VIEW IS THE ANSWER, NOT A CONVENIENCE. The caller's own view moved along with everyone else's,
 * and nothing hands out a view at a lower position — so without this the inserter could not reach what it
 * just inserted. With it, putting something at the front of a list is `cons`, and what comes back is what
 * the caller assigns to the name that held the list. */
static __device__ inline uint64_t sys__sublist__insert(uint64_t sublist, uint64_t n,
                                                       const sys__heap_node* value) {
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return 0ull;
    const uint64_t store = f->args[SYS__SUBLIST__STORE];
    uint64_t position = 0ull;
    if (!sys__list__zzprivate_place(sublist, n, value, &position)) return 0ull;
    return sys__list__zzprivate_view(store, position);
}

/* Appending answers whether it went in and nothing more. A view of the last value is not what a caller
 * would put back where the list was — it would throw the front away — and the appender is already
 * standing before what it added. */
static __device__ inline bool sys__sublist__append(uint64_t sublist, const sys__heap_node* value) {
    if (value == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return false;
    }
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return false;
    /* ⭐⭐ APPENDING IS `position == total`, SO THE TOTAL IS THE WHOLE ANSWER — no index relative to the
     * view is computed, only to be added back to `START`. ⛳ A view that has fallen behind the store
     * appends at the store's end: `START + (total - START) == total`. */
    const uint64_t store = f->args[SYS__SUBLIST__STORE];
    const uint64_t total = sys__list__zzprivate_total(store);
    return sys__list__zzprivate_place_at(store, total, total, value, 0);
}

static __device__ inline bool sys__sublist__replace(uint64_t sublist, uint64_t n, const sys__heap_node* value) {
    if (value == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return false;
    }
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return false;
    /* ⛳ THE RANGE TEST IS `locate`'s OWN ANSWER, as in `nth` — ▶ there. It is the very next line, so
     * the range check costs nothing extra. */
    uint64_t chunk = 0ull, index = 0ull, before = 0ull;
    if (!sys__list__zzprivate_locate(f->args[SYS__SUBLIST__STORE],
                                     f->args[SYS__SUBLIST__START] + n, &chunk, &index, &before)) {
        sys__fault__raise(0ull, SYS__LIST__FAULT_RANGE);
        return false;
    }

    /* ⭐ ONE CALL, AND THE ORDER INSIDE IT IS WHY IT IS ONE: what goes in takes a hold BEFORE what was
     * there gives one up, because they can be the same object and letting go first would take it apart
     * and then store a reference to the room.
     * ⛳ AND NO ADJUST RUNS, WHICH IS THE POINT. Nothing moved, so no view is naming anything different
     * from what it was naming — a replacement is the one change to a list that every holder sees at once
     * and none of them has to be told about. */
    sys__heap_object__set(&sys__list_chunk__zzprivate_cells(chunk)[index], value);
    return true;
}

static __device__ inline uint64_t sys__sublist__discard(uint64_t sublist, uint64_t n, uint64_t count) {
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0 || count == 0ull) return 0ull;
    if (n + count > sys__sublist__length(sublist)) {
        sys__fault__raise(0ull, SYS__LIST__FAULT_RANGE);
        return 0ull;
    }
    const uint64_t store    = f->args[SYS__SUBLIST__STORE];
    const uint64_t position = f->args[SYS__SUBLIST__START] + n;

    /* The walk is per CHUNK and not per value, which is the whole of what this verb is for: a range
     * inside one chunk is one locate and one shift, however many values it covers. A range that spans
     * takes one lap per chunk it reaches into, which is the fewest it can be. */
    uint64_t left = count;
    while (left > 0ull) {
        uint64_t chunk = 0ull, index = 0ull, before = 0ull;
        if (!sys__list__zzprivate_locate(store, position, &chunk, &index, &before)) break;
        sys__heap_node*    cells = sys__list_chunk__zzprivate_cells(chunk);
        sys__heap_node*    head  = sys__heap_node__zzpackage_head(chunk);
        const uint64_t used  = head->args[SYS__LIST_CHUNK__HEAD_USED];
        uint64_t       take  = used - index;
        if (take > left) take = left;

        /* Letting go comes BEFORE the shift, because after it the cells hold something else. */
        for (uint64_t i = 0ull; i < take; ++i)
            if (sys__heap_node__carries_reference(cells[index + i].dtype))
                (void)sys__heap_object__release(cells[index + i].args[0]);
        for (uint64_t i = index + take; i < used; ++i) cells[i - take] = cells[i];
        head->args[SYS__LIST_CHUNK__HEAD_USED] = used - take;
        /* ⛳ THE COUNT COMES DOWN BY WHAT WENT — ▶ the invariant note in the header. */
        sys__heap_node* sf = sys__list__zzprivate_store_fields(store);
        sf->args[SYS__LIST__STORE_TOTAL] -= take;

        if (used - take == 0ull) {
            const uint64_t after = head->args[SYS__LIST_CHUNK__HEAD_NEXT];
            head->args[SYS__LIST_CHUNK__HEAD_NEXT] = 0ull;
            if (before == 0ull) sf->args[SYS__LIST__STORE_VALUES] = after;
            else                sys__heap_node__zzpackage_head(before)->args[SYS__LIST_CHUNK__HEAD_NEXT] = after;
            /* ⛔ A ZERO `after` MEANS THIS WAS THE LAST CHUNK, so the one before it becomes the end —
             * and a zero `before` then means the list is empty, which is the same word. */
            if (after == 0ull) sf->args[SYS__LIST__STORE_TAIL] = before;
            (void)sys__heap_object__release(chunk);
        }
        left -= take;
    }

    /* ⭐ AND EVERY VIEW MOVES ONCE, however many values went. That is the saving the verb exists for, and
     * it is also what makes it CORRECT rather than merely faster: moving them per value would walk a view
     * standing inside the range down through it one step at a time. */
    sys__list__zzprivate_adjust(store, position, false, true, count - left);
    return count - left;
}

static __device__ inline sys__heap_node sys__sublist__extract(uint64_t sublist, uint64_t n) {
    sys__heap_node* f = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return sys__heap_node__nothing();
    if (n >= sys__sublist__length(sublist)) {
        sys__fault__raise(0ull, SYS__LIST__FAULT_RANGE);
        return sys__heap_node__nothing();
    }
    const uint64_t store    = f->args[SYS__SUBLIST__STORE];
    const uint64_t position = f->args[SYS__SUBLIST__START] + n;

    uint64_t chunk = 0ull, index = 0ull, before = 0ull;
    if (!sys__list__zzprivate_locate(store, position, &chunk, &index, &before)) return sys__heap_node__nothing();

    sys__heap_node*    cells = sys__list_chunk__zzprivate_cells(chunk);
    sys__heap_node*    head  = sys__heap_node__zzpackage_head(chunk);
    const uint64_t used  = head->args[SYS__LIST_CHUNK__HEAD_USED];

    /* ⭐ THE VALUE IS TRANSFERRED AND NOT RELEASED, WHICH IS THE WHOLE ACCOUNTING OF A REMOVAL. The list
     * held one reference and the caller walks away with that same one — nothing taken, nothing given up,
     * no count moved. It is also what makes a borrow survivable: the one thing that ends a borrow is the
     * value leaving the list, and this hands ownership over at exactly that moment. */
    const sys__heap_node gone = cells[index];
    for (uint64_t i = index + 1ull; i < used; ++i) cells[i - 1ull] = cells[i];
    head->args[SYS__LIST_CHUNK__HEAD_USED] = used - 1ull;
    /* ⛳ ONE VALUE LEFT, SO THE STORE HOLDS ONE FEWER — ▶ the invariant note in the header. */
    sys__heap_node* sf = sys__list__zzprivate_store_fields(store);
    sf->args[SYS__LIST__STORE_TOTAL] -= 1ull;

    if (used - 1ull == 0ull) {
        /* An empty chunk goes home. The link is cleared BEFORE the release, or the release would move
         * what it names onto the teardown chain and take the whole rest of the list with it. */
        const uint64_t after = head->args[SYS__LIST_CHUNK__HEAD_NEXT];
        head->args[SYS__LIST_CHUNK__HEAD_NEXT] = 0ull;
        if (before == 0ull) sf->args[SYS__LIST__STORE_VALUES] = after;
        else                sys__heap_node__zzpackage_head(before)->args[SYS__LIST_CHUNK__HEAD_NEXT] = after;
        if (after == 0ull) sf->args[SYS__LIST__STORE_TAIL] = before;   /* it was the end; now `before` is */
        (void)sys__heap_object__release(chunk);
    }

    sys__list__zzprivate_adjust(store, position, false, true, 1ull);
    return gone;
}

/* ── GIVING ONE AWAY ─────────────────────────────────────────────────────────────────────────────── */

static __device__ inline uint64_t sys__sublist__clone(uint64_t sublist) {
    const uint64_t length = sys__sublist__length(sublist);
    sys__heap_node*    f      = sys__sublist__zzprivate_fields(sublist);
    if (f == 0) return 0ull;

    const uint64_t copy = sys__list__create();
    if (copy == 0ull) return 0ull;
    const uint64_t store = f->args[SYS__SUBLIST__STORE];
    const uint64_t start = f->args[SYS__SUBLIST__START];
    for (uint64_t i = 0ull; i < length; ++i) {
        /* Read borrowed, put in held — the copy becomes a second owner of every value, which is what
         * makes the two lists changeable without either reaching the other. */
        const sys__heap_node value = sys__list__zzprivate_at(store, start + i);
        if (!sys__sublist__append(copy, &value)) { (void)sys__heap_object__release(copy); return 0ull; }
    }
    return copy;
}

/* Copy one list without looking inside it, and put the source and the copy on the end of the two lists.
 * The copy is let go of here because the list it went on is its owner from that moment. */
static __device__ inline bool sys__list__zzprivate_note(uint64_t seen, uint64_t made, uint64_t source) {
    const uint64_t copy = sys__sublist__clone(source);
    if (copy == 0ull) return false;
    const sys__heap_node s = sys__heap_object__reference_to(source);
    const sys__heap_node d = sys__heap_object__reference_to(copy);
    const bool ok = sys__sublist__append(seen, &s) && sys__sublist__append(made, &d);
    (void)sys__heap_object__release(copy);
    return ok;
}

static __device__ inline uint64_t sys__sublist__deep_clone(uint64_t sublist) {
    const uint64_t seen = sys__list__create();   /* the sources, in the order they were first met */
    const uint64_t made = sys__list__create();   /* the copy of each, at the matching position    */
    uint64_t       answer = 0ull;

    if (seen != 0ull && made != 0ull && sys__list__zzprivate_note(seen, made, sublist)) {
        uint64_t known = 1ull;   /* how many are on them — the two grow together, so one count serves */
        bool     ok    = true;

        for (uint64_t i = 0ull; ok && i < known; ++i) {
            const uint64_t here = sys__sublist__nth(made, i).args[0];
            const uint64_t n    = sys__sublist__length(here);
            /* ⭐ THE SAME WALK AS `deep_copy_to_node_array`, FOR THE SAME REASON AND IN THE SAME SHAPE —
             * ▶ there. `note` appends to `seen` and `made`, never to `here`, so this walk stands for the
             * whole loop; the `j` scan's does not, and is re-opened.
             * ⛳ OF THE FOUR SITES WITH THIS SHAPE THIS ONE IS THE COLDEST — `deep_clone` is the `clone`
             * verb and not the evaluator's way in. It walks anyway, because a defect left standing in
             * one of four is how the next reader learns the pattern. */
            sys__list_walk hw;
            if (!sys__sublist__walk(here, 0ull, &hw)) { ok = false; break; }
            for (uint64_t k = 0ull; k < n; ++k, sys__list__walk_step(&hw)) {
                sys__heap_node* slot = sys__list__walk_cell(&hw);
                if (slot == 0) break;
                sys__heap_node cell = *slot;
                if (!sys__heap_node__is_substructure(cell.dtype) || !sys__sublist__is(cell.args[0])) continue;

                uint64_t j = 0ull;
                sys__list_walk jw;
                if (!sys__sublist__walk(seen, 0ull, &jw)) { ok = false; break; }
                for (; j < known; ++j, sys__list__walk_step(&jw)) {
                    const sys__heap_node* met = sys__list__walk_cell(&jw);
                    if (met == 0) { j = known; break; }
                    if (met->args[0] == cell.args[0]) break;
                }
                if (j == known) {
                    if (!sys__list__zzprivate_note(seen, made, cell.args[0])) { ok = false; break; }
                    ++known;
                }
                /* The kind is kept and only the address changes, so a quoted list stays quoted. Writing it
                 * this way is what hands the hold on the source over to the copy — the same call `replace`
                 * makes, on the slot the walk already found. */
                cell.args[0] = sys__sublist__nth(made, j).args[0];
                sys__heap_object__set(slot, &cell);
            }
        }
        if (ok) {
            answer = sys__sublist__nth(made, 0ull).args[0];
            (void)sys__heap_object__retain(answer);   /* the caller's own hold, before the lists go */
        }
    }

    /* The two lists were holding every source and every copy. Letting them go leaves each copy held by
     * the copy it went into, and the outermost one held by whoever asked for it. */
    if (made != 0ull) (void)sys__heap_object__release(made);
    if (seen != 0ull) (void)sys__heap_object__release(seen);
    return answer;
}

static __device__ inline uint64_t sys__sublist__create_viewonly_array(uint64_t sublist) {
    const uint64_t length = sys__sublist__length(sublist);
    sys__heap_node*    f      = sys__sublist__zzprivate_fields(sublist);
    if (f == 0 || length == 0ull) return 0ull;         /* a table with no elements has nothing to name */

    const uint64_t picture = sys__node_array__create(length);
    if (picture == 0ull) return 0ull;
    const uint64_t store = f->args[SYS__SUBLIST__STORE];
    const uint64_t start = f->args[SYS__SUBLIST__START];
    for (uint64_t i = 0ull; i < length; ++i) {
        const sys__heap_node value = sys__list__zzprivate_at(store, start + i);
        if (!sys__node_array__set(picture, i, &value)) { (void)sys__heap_object__release(picture); return 0ull; }
    }
    return picture;
}

/* ── A PICTURE, AND BACK ─────────────────────────────────────────────────────────────────────────── */

/* Make the array a list is pictured into and note the pair — the list on one side, its array on the other,
 * at the same position. Noting the list holds it, so it cannot go while its picture is being filled. */
static __device__ inline bool sys__list__zzprivate_frame(uint64_t seen, uint64_t made, uint64_t source) {
    const uint64_t picture = sys__node_array__create(sys__sublist__length(source));
    if (picture == 0ull) return false;                   /* the carve said why: nothing in it, or too long */
    const sys__heap_node s = sys__heap_object__reference_to(source);
    const sys__heap_node d = sys__heap_object__reference_to(picture);
    const bool ok = sys__sublist__append(seen, &s) && sys__sublist__append(made, &d);
    (void)sys__heap_object__release(picture);            /* `made` holds it now */
    return ok;
}

static __device__ inline uint64_t sys__sublist__deep_copy_to_node_array(uint64_t sublist) {
    if (!sys__sublist__is(sublist)) {
        sys__fault__raise(0ull, SYS__LIST__FAULT_KIND);
        return 0ull;
    }
    const uint64_t seen = sys__list__create();   /* the lists met, in the order they were first met */
    const uint64_t made = sys__list__create();   /* the array made for each, at the matching position */
    uint64_t       answer = 0ull;

    if (seen != 0ull && made != 0ull && sys__list__zzprivate_frame(seen, made, sublist)) {
        uint64_t known = 1ull;   /* how many are on them — the two grow together, so one count serves */
        bool     ok    = true;
        for (uint64_t i = 0ull; ok && i < known; ++i) {
            const uint64_t source  = sys__sublist__nth(seen, i).args[0];
            const uint64_t picture = sys__sublist__nth(made, i).args[0];
            const uint64_t n       = sys__sublist__length(source);
            /* ⭐ ONE WALK FOR THE WHOLE OF THIS LIST — ▶ `sys__sublist__walk`. Nothing appends to `source`
             * while it is being pictured, so the walk stands for the whole inner loop. `source` is every
             * list met INCLUDING the program's own top-level list, which is the long one, so `nth` per
             * cell here would be the bulk of what `freeze` costs. */
            sys__list_walk sw;
            if (!sys__sublist__walk(source, 0ull, &sw)) { ok = false; break; }
            for (uint64_t k = 0ull; ok && k < n; ++k, sys__list__walk_step(&sw)) {
                const sys__heap_node* found = sys__list__walk_cell(&sw);
                if (found == 0) { ok = false; break; }   /* `n` said it was there and it was not */
                sys__heap_node cell = *found;
                if (sys__heap_node__is_substructure(cell.dtype) && sys__sublist__is(cell.args[0])) {
                    if (sys__sublist__empty(cell.args[0])) {
                        cell = sys__heap_node__nothing();            /* an empty list is nil */
                    } else {
                        /* ⛔ A LIST MET BEFORE IS REFUSED, NOT SHARED: the way back keeps no map, so a shared
                         * list would come back as two. A list that reaches itself has been met before.
                         * ⛔⛔ AND THIS SCAN IS THE COSTLIEST OF THE THREE LOOPS, WHICH READING DOES NOT SHOW
                         * because it is the innermost and looks like a membership test rather than a
                         * traversal. It runs once per SUB-LIST CELL, over every list met so far — through
                         * `nth` that would be `n²/2` scans each following `n/SLOTS` links: CUBIC, in the
                         * verb every program pays on its way in. The walk makes each scan a sequential read,
                         * which leaves `n²` compares — quadratic, and cheap enough that the thing to fix
                         * next is the SCAN, not the access. */
                        sys__list_walk jw;
                        if (!sys__sublist__walk(seen, 0ull, &jw)) { ok = false; break; }
                        for (uint64_t j = 0ull; j < known; ++j, sys__list__walk_step(&jw)) {
                            const sys__heap_node* met = sys__list__walk_cell(&jw);
                            if (met == 0) break;
                            if (met->args[0] == cell.args[0]) { ok = false; break; }
                        }
                        if (!ok) { sys__fault__raise(0ull, SYS__LIST__FAULT_SHARED); break; }
                        if (!sys__list__zzprivate_frame(seen, made, cell.args[0])) { ok = false; break; }
                        cell.args[0] = sys__sublist__nth(made, known).args[0];
                        known += 1ull;
                    }
                } else if (sys__heap_node__is_substructure(cell.dtype) && sys__node_array__is(cell.args[0])) {
                    /* An array the list held stays that array, named so the way back knows it was one. Only a
                     * plain reference may name one: a quoted LIST naming an array would come back a list. */
                    if (cell.dtype != SYS__KIND__OBJECT_REFERENCE) {
                        sys__fault__raise(0ull, SYS__LIST__FAULT_KIND);
                        ok = false;
                        break;
                    }
                    cell.dtype = SYS__KIND__QUOTED_ARRAY;
                }
                if (!sys__node_array__set(picture, k, &cell)) ok = false;
            }
        }
        if (ok) {
            answer = sys__sublist__nth(made, 0ull).args[0];
            (void)sys__heap_object__retain(answer);   /* the caller's own hold, before the lists go */
        }
    }

    /* The two lists were holding every list met and every array made. Letting them go leaves each array
     * held by the array it went into and the outermost one by whoever asked — or, after a refusal, held by
     * nobody, so all of it goes. */
    if (made != 0ull) (void)sys__heap_object__release(made);
    if (seen != 0ull) (void)sys__heap_object__release(seen);
    return answer;
}

/* ⭐⭐ ONE WORKER, TWO DOORS, AND `lazy` IS THE ONLY DIFFERENCE — a sub-form is either MADE now or NAMED
 * for later. ⚖ architect: *"the thawing is shallow and thaws into the already allocated list … only
 * instantiate one if the reference is empty."* This is the first half of that: stop making the ones
 * nobody asked for. The pool is the second half, and it is the evaluator's (`engine/eval.cuh`).
 *
 * ⛔⛔ THE TWO DOORS ARE NOT INTERCHANGEABLE AND THE HOST'S MUST STAY EAGER.
 * ```
 *   eng_abi_thaw              the HOST asks for a list and must get a whole one   EAGER, always
 *   sys__computing_base__begin  a block's private running copy                    lazy
 *   eng__eval__zzprivate_expand a procedure body, copied once PER CALL            lazy
 * ```
 * ⇒ ★ A LAZY COPY IS ONLY SAFE WHERE THE EVALUATOR IS THE READER, because the evaluator is the one
 * thing that knows how to meet a `FROZEN_LIST`. Anything else handed one would see a tag it has no arm
 * for. That is why this is a parameter and not a mode.
 *
 * ⛳ ONLY A PLAIN `OBJECT_REFERENCE` SUB-ARRAY IS FROZEN. A `QUOTED_LIST` is data a verb will be handed,
 * so it is thawed eagerly under both doors and no verb has to learn the tag. */
static __device__ inline uint64_t sys__list__zzprivate_thaw(uint64_t array_base, bool lazy) {
    if (!sys__node_array__is(array_base)) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_KIND);
        return 0ull;
    }
    const uint64_t answer = sys__list__create();
    if (answer == 0ull) return 0ull;

    /* ⭐ WHAT IS LEFT TO DO IS A STACK OF PAIRS, AND NOTHING CALLS ITSELF: an array met is paired with the
     * empty list made for it and waits its turn. A pair is two numbers and not two references — the picture
     * is held by the caller and each list by the list it went into — so the stack owes nothing to anybody
     * and taking it down lets go of nothing. */
    sys__heap_node work = sys__stack__create();
    bool ok = (work.dtype == SYS__KIND__OBJECT_REFERENCE);
    sys__heap_node pair;
    pair.dtype = SYS__KIND__VALUE_INT; pair.num_args = 0u; pair.op_code = 0ull;
    pair.args[0] = array_base; pair.args[1] = answer;
    if (ok) ok = sys__stack__push(&work, &pair);

    while (ok && !sys__stack__empty(&work)) {
        const sys__heap_node next = sys__stack__pop(&work);
        const uint64_t picture = next.args[0];
        const uint64_t list    = next.args[1];
        const uint64_t n       = sys__node_array__length(picture);
        for (uint64_t k = 0ull; ok && k < n; ++k) {
            sys__heap_node cell = sys__node_array__borrow(picture, k);
            if (lazy && cell.dtype == SYS__KIND__OBJECT_REFERENCE
                     && sys__node_array__is(cell.args[0])) {
                /* ⭐ NAMED, NOT MADE. The cell keeps the array it came from and takes a hold of it, the
                 * way any reference cell does — `carries_reference` says so, which is what makes the
                 * ordinary `discard` on an untaken `if` arm let the array go without a special case. */
                sys__heap_node frozen = cell;
                frozen.dtype = SYS__KIND__FROZEN_LIST;
                /* ⛔ NO RETAIN HERE, AND THE EAGER ARM BELOW IS WHY IT LOOKS LIKE ONE IS MISSING.
                 * `append` takes the hold itself — that arm creates `inner` at one, appends it to two
                 * and releases back to one, which is the list's. The array this names is ALREADY held
                 * by the picture, so appending is the whole of what the cell owes: one hold, taken by
                 * the append, given back by the `discard` or the `set` that overwrites the cell. */
                ok = sys__sublist__append(list, &frozen);
                continue;
            }
            if (sys__heap_node__is_substructure(cell.dtype) && sys__node_array__is(cell.args[0])) {
                const uint64_t inner = sys__list__create();
                if (inner == 0ull) { ok = false; break; }
                sys__heap_node named = sys__heap_object__reference_to(inner);
                named.dtype = cell.dtype;                         /* quoted comes back quoted */
                ok = sys__sublist__append(list, &named);
                (void)sys__heap_object__release(inner);           /* the list it went into holds it */
                pair.args[0] = cell.args[0];
                pair.args[1] = inner;
                if (ok) ok = sys__stack__push(&work, &pair);
            } else {
                if (cell.dtype == SYS__KIND__QUOTED_ARRAY) cell.dtype = SYS__KIND__OBJECT_REFERENCE;
                ok = sys__sublist__append(list, &cell);
            }
        }
    }
    if (work.dtype == SYS__KIND__OBJECT_REFERENCE) (void)sys__heap_object__release(work.args[0]);
    if (!ok) { (void)sys__heap_object__release(answer); return 0ull; }
    return answer;
}

/* The host's door, and it is EAGER. `eng_abi_thaw` answers a list somebody outside the evaluator
 * will read, and nothing out there has an arm for a `FROZEN_LIST`. ⛔ Do not "optimise" this one. */
static __device__ inline uint64_t sys__list__deep_copy_from_node_array(uint64_t array_base) {
    return sys__list__zzprivate_thaw(array_base, false);
}

/* The evaluator's door. A running copy is read by `eng__eval` and by nothing else: `first_form` finds a
 * frozen cell and `eng__eval` turns it into a list at the moment it descends into one. ⛳ Its callers are
 * the block's private copy of a scheduled program, a procedure body — copied ONCE PER CALL — and that
 * descent. */
static __device__ inline uint64_t sys__list__thaw_running(uint64_t array_base) {
#if SILVANN_LAZY_THAW
    /* ⭐⭐ NOT A CALL TO `zzprivate_thaw`. That function's first act is `sys__stack__create()` for the
     * tree it is about to walk, and under laziness there IS no tree — one level is the whole job — so
     * delegating makes every materialised sub-form pay for a stack object it pushes one pair onto.
     * ⇒ ★★ A SHALLOW COPY THAT REUSES A DEEP COPY'S MACHINERY IS NOT SHALLOW. The saving this feature
     * exists for is an allocation per form; spending one to avoid one gives the saving back.
     * ⛳ A QUOTED sub-list is built one level deep by `thaw_running_into`, through a work stack made only
     * when one is met — ▶ there. */
    const uint64_t list = sys__list__create();
    if (list == 0ull) return 0ull;
    if (!sys__list__thaw_running_into(array_base, list)) {
        (void)sys__heap_object__release(list);
        return 0ull;
    }
    return list;
#endif
    return sys__list__zzprivate_thaw(array_base, false);
}

#if SILVANN_LAZY_THAW
/* Fill a list that ALREADY EXISTS from one level of a picture. ⭐ IT IS SPLIT OUT SO A RECYCLED LIST
 * CAN BE FILLED WITHOUT BEING MADE — which is the whole of what the pool buys. ⚖ architect: *"the
 * thawing is shallow and thaws into the already allocated list that is held by the engine."* */
static __device__ inline bool sys__list__thaw_running_into(uint64_t array_base, uint64_t list) {
    if (!sys__node_array__is(array_base)) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_KIND);
        return false;
    }

    /* ⭐⭐⭐ THE WORK STACK IS CREATED ON FIRST NEED AND NOT BEFORE. The eager thaw opens
     * `sys__stack__create()` unconditionally, so run once per form it makes a program of N flat forms pay
     * for N stack objects each holding one pair. Here the stack appears only when a QUOTED SUB-LIST is
     * actually met — the rare case — so the common path allocates nothing.
     * ⇒ ★★ THE COST IS A STACK PER *FORM*, NOT THE COST OF HAVING A STACK. Paying for one only where a
     *   quoted sub-list is met is a different trade, and it is what lets this loop stand in for a
     *   recursive call.
     * ⛔ AND THAT IS WHY IT MUST NOT CALL `sys__list__thaw_running` FOR A SUB-LIST: the two would call
     * each other, and the one thing `list__header.cuh` says of this whole file is *"NOTHING HERE CALLS
     * ITSELF"*. ▶ the claim rule `list_call_graph_acyclic`, which fails on exactly that pair. */
    sys__heap_node work = sys__heap_node__nothing();
    bool have_work = false;

    uint64_t picture = array_base;
    uint64_t into    = list;
    bool     ok      = true;

    for (;;) {
        const uint64_t n = sys__node_array__length(picture);
        for (uint64_t k = 0ull; ok && k < n; ++k) {
            sys__heap_node cell = sys__node_array__borrow(picture, k);
            if (cell.dtype == SYS__KIND__OBJECT_REFERENCE && sys__node_array__is(cell.args[0])) {
                cell.dtype = SYS__KIND__FROZEN_LIST;      /* named, not made; `append` takes the hold */
                ok = sys__sublist__append(into, &cell);
                continue;
            }
            if (sys__heap_node__is_substructure(cell.dtype) && sys__node_array__is(cell.args[0])) {
                /* ⭐⭐ A QUOTED LIST IS BUILT, BUT ONLY ONE LEVEL DEEP — AND THE DIFFERENCE IS THE WHOLE
                 * FEATURE. `let` takes its body as a QUOTED_LIST and `if` takes both arms that way, so a
                 * deep eager copy here re-thaws every form of every body: `MEASURED`, with a
                 * deep call on this line nothing in a `(let ... F forms)` program is lazy at all and
                 * the pool never sees a single take.
                 * ⛳ THE VERB GETS A REAL LIST, which is the promise that keeps verbs unchanged —
                 * what comes back frozen is the forms INSIDE it, and those are met by the evaluator
                 * after `if`/`let` unquote the list and the scan descends into it.
                 * ⛳ THE INNER LIST IS MADE AND NAMED HERE, and its filling is pushed onto the work
                 * stack instead of entered through a call. */
                const uint64_t inner = sys__list__create();
                if (inner == 0ull) { ok = false; break; }
                sys__heap_node named = sys__heap_object__reference_to(inner);
                named.dtype = cell.dtype;                 /* quoted comes back quoted */
                ok = sys__sublist__append(into, &named);
                (void)sys__heap_object__release(inner);   /* the list it went into holds it */
                if (!ok) break;
                if (!have_work) {
                    work = sys__stack__create();
                    have_work = (work.dtype == SYS__KIND__OBJECT_REFERENCE);
                    if (!have_work) { ok = false; break; }
                }
                sys__heap_node pair;
                pair.dtype = SYS__KIND__VALUE_INT; pair.num_args = 0u; pair.op_code = 0ull;
                pair.args[0] = cell.args[0];
                pair.args[1] = inner;
                if (!sys__stack__push(&work, &pair)) { ok = false; break; }
                continue;
            }
            if (cell.dtype == SYS__KIND__QUOTED_ARRAY) cell.dtype = SYS__KIND__OBJECT_REFERENCE;
            ok = sys__sublist__append(into, &cell);
        }
        if (!ok) break;
        if (!have_work || sys__stack__empty(&work)) break;
        /* ⛳ THE INNER LIST IS ALREADY HELD BY THE PARENT IT WENT INTO, so the pair carries offsets and
         * no hold of its own — exactly as the eager thaw's does. */
        const sys__heap_node next = sys__stack__pop(&work);
        picture = next.args[0];
        into    = next.args[1];
    }

    if (have_work) (void)sys__heap_object__release(work.args[0]);
    return ok;
}
#endif

/* ── THE ROW METHODS ─────────────────────────────────────────────────────────────────────────────── */

static __device__ __noinline__ void sys__sublist__zzpackage_release_internal(sys__heap_node* head,
                                                                       uint64_t* releaser_stack) {
    if (head == 0) return;
    const uint64_t store = head[1].args[SYS__SUBLIST__STORE];
    if (store == 0ull) return;                          /* it never got as far as taking one */

    /* ⛔ THE SEAT GOES FIRST, AND IT IS THE ONE THING HERE THAT IS NOT ORDINARY. An entry naming a view
     * that has gone would be walked and adjusted by every later mutation, and would hold a freed position
     * in the count of who is looking. Doing it here is safe without a lock for the reason the whole
     * design rests on: one owner, so nothing is walking the roll at this moment. */
    sys__list__zzprivate_leave(store, head[1].args[SYS__SUBLIST__SEAT]);
    head[1].args[SYS__SUBLIST__STORE] = 0ull;

    sys__heap_node reference = sys__heap_object__reference_to(store);
    sys__stack__transfer(releaser_stack, &reference);
}

static __device__ __noinline__ void sys__list__zzpackage_release_internal(sys__heap_node* head,
                                                                    uint64_t* releaser_stack) {
    if (head == 0) return;
    /* Two moves, and each is one object however long the chain behind it — every chunk's own row deals
     * with what is inside it, and with the chunk after it, when its turn comes. */
    const uint64_t held[2] = { head[1].args[SYS__LIST__STORE_VALUES], head[1].args[SYS__LIST__STORE_ROLL] };
    head[1].args[SYS__LIST__STORE_VALUES] = 0ull;
    head[1].args[SYS__LIST__STORE_ROLL]   = 0ull;
    head[1].args[SYS__LIST__STORE_TOTAL]  = 0ull;   /* ▶ the invariant note in the header */
    head[1].args[SYS__LIST__STORE_TAIL]   = 0ull;
    for (unsigned int i = 0u; i < 2u; ++i) {
        if (held[i] == 0ull) continue;
        sys__heap_node reference = sys__heap_object__reference_to(held[i]);
        sys__stack__transfer(releaser_stack, &reference);
    }
}

static __device__ __noinline__ void sys__list_chunk__zzpackage_release_internal(sys__heap_node* head,
                                                                          uint64_t* releaser_stack) {
    if (head == 0) return;
    sys__heap_node*    cells = (sys__heap_node*)(head + 1);
    const uint64_t used  = head->args[SYS__LIST_CHUNK__HEAD_USED];
    /* Nothing beyond `used` is read, which is what makes a chunk safe to hand out without clearing it —
     * and a roll chunk's cells are numbers rather than references, so this walk correctly moves none. */
    for (uint64_t i = 0ull; i < used; ++i)
        if (sys__heap_node__carries_reference(cells[i].dtype))
            sys__stack__transfer(releaser_stack, &cells[i]);

    const uint64_t after = head->args[SYS__LIST_CHUNK__HEAD_NEXT];
    head->args[SYS__LIST_CHUNK__HEAD_NEXT] = 0ull;
    if (after == 0ull) return;
    sys__heap_node reference = sys__heap_object__reference_to(after);
    sys__stack__transfer(releaser_stack, &reference);
}

static __device__ inline sys__heap_node sys__list__zzpackage_construct(sys__heap_node* base,
                                                                   const sys__heap_node* parameters) {
    (void)base; (void)parameters;                       /* a list arrives empty; there is nothing to say */
    const uint64_t at = sys__list__create();
    if (at == 0ull) return sys__heap_node__nothing();    /* the making already raised */
    return sys__heap_object__reference_to(at);
}

/* ▶ `list__header.cuh` for why this adapter exists and what it is for. It is the SUBLIST row's clone
 * column: the `sys__clone` verb's route to the deep copy. */
static __device__ inline uint64_t sys__list__zzpackage_clone_object(sys__heap_node* head,
                                                                     sys__heap_node* computing_base) {
    (void)computing_base;                 /* the copy carves from the CALLER's arena — see the header */
    if (head == 0) return 0ull;
    return sys__sublist__deep_clone(sys__heap__offset(head + 1));
}

#endif /* SILVANN__PACKAGES_SYS_CPU_LIST__IMPL_CUH */
