#ifndef SILVANN__PACKAGES_SYS_CPU_NODE_ARRAY__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_NODE_ARRAY__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "fault__header.cuh"        /* how it says a caller asked for something that is not there */
#include "heap_object__header.cuh"   /* the head behind a reference, and what a kind answers when asked */
#include "heap__header.cuh"         /* the sized carve, and turning an offset back into an address */
#include "stack__header.cuh"  /* where a dying array moves what it was holding */
#include "node_array__header.cuh"  /* the verbs it fills in, and the three fault words they raise */

/* Where an element sits in a run: a reference names element zero, so element `i` is `i` nodes along from
 * it. Inside one run this is the whole of finding an element, and everything below that crosses runs comes
 * back to it once it knows which run. */
static __device__ inline sys__heap_node* sys__node_array__zzprivate_in_run(uint64_t run, uint64_t index) {
    return sys__heap__object_full_address(run + index);
}

/* The cone that ends this run, or nothing when the run is the array's last. Only a full run can end in
 * one, so a shorter run answers off its head and its last cell is never read. */
static __device__ inline sys__heap_node* sys__node_array__zzprivate_cone(uint64_t run) {
    const uint64_t cells = SYS__HEAP_OBJECT__NODES(sys__heap_node__zzpackage_head(run)) - 1ull;
    if (cells != SYS__NODE_ARRAY__RUN_CELLS) return (sys__heap_node*)0;
    sys__heap_node* last = sys__node_array__zzprivate_in_run(run, cells - 1ull);
    return last->dtype == SYS__KIND__NODE_ARRAY_CONE ? last : (sys__heap_node*)0;
}

/* How many elements this run holds: every cell past the head, less the cone if there is one. */
static __device__ inline uint64_t sys__node_array__zzprivate_run_length(uint64_t run, const sys__heap_node* cone) {
    return SYS__HEAP_OBJECT__NODES(sys__heap_node__zzpackage_head(run)) - 1ull - (cone != 0 ? 1ull : 0ull);
}

/* Where an element sits, following the cones from the run named: one hop per run passed. It is what an
 * array without a table pays, and how the table itself is read. */
static __device__ inline sys__heap_node* sys__node_array__zzprivate_at_by_cones(uint64_t run, uint64_t index) {
    for (;;) {
        const sys__heap_node* cone = sys__node_array__zzprivate_cone(run);
        const uint64_t held = sys__node_array__zzprivate_run_length(run, cone);
        if (cone == 0 || index < held) return sys__node_array__zzprivate_in_run(run, index);
        index -= held;
        run = cone->args[0];
    }
}

/* How many elements, following the cones from the run named. */
static __device__ inline uint64_t sys__node_array__zzprivate_length_by_cones(uint64_t run) {
    uint64_t length = 0ull;
    for (;;) {
        const sys__heap_node* cone = sys__node_array__zzprivate_cone(run);
        length += sys__node_array__zzprivate_run_length(run, cone);
        if (cone == 0) return length;
        run = cone->args[0];
    }
}

/* The table of runs a chained array keeps, or nothing — for an array that fits, which has none, and for
 * a table, which is never given one of its own. */
static __device__ inline uint64_t sys__node_array__zzprivate_indexer(uint64_t array_base, const sys__heap_node* cone) {
    if (cone == 0) return 0ull;
    return sys__heap_node__zzpackage_head(array_base)->args[SYS__NODE_ARRAY__INDEXER];
}

/* Which run an index falls in, and how far into it. Every run but the last holds the same number of
 * elements, so the table answers the run by a division and one read; with no table it is the cones. False
 * past the end, with the last run and how far past its start the index lies. */
static __device__ inline bool sys__node_array__zzprivate_locate(uint64_t array_base, uint64_t index,
                                                               uint64_t* run_out, uint64_t* within_out) {
    uint64_t run = array_base;
    const sys__heap_node* cone = sys__node_array__zzprivate_cone(run);
    const uint64_t table = sys__node_array__zzprivate_indexer(run, cone);
    if (table != 0ull) {
        const uint64_t chained = SYS__NODE_ARRAY__RUN_CELLS - 1ull;
        const uint64_t runs = sys__node_array__zzprivate_length_by_cones(table);
        uint64_t r = index / chained;
        if (r >= runs) r = runs - 1ull;                    /* the last run may hold one more than the rest */
        run = sys__node_array__zzprivate_at_by_cones(table, r)->args[0];
        index -= r * chained;
        cone = sys__node_array__zzprivate_cone(run);
    }
    for (;;) {
        const uint64_t held = sys__node_array__zzprivate_run_length(run, cone);
        *run_out = run;
        *within_out = index;
        if (index < held) return true;
        if (cone == 0) return false;
        index -= held;
        run = cone->args[0];
        cone = sys__node_array__zzprivate_cone(run);
    }
}

/* Where an element sits in the whole array: the addition above, in the run the index falls in. */
static __device__ inline sys__heap_node* sys__node_array__zzprivate_at(uint64_t array_base, uint64_t index) {
    uint64_t run = 0ull, within = 0ull;
    (void)sys__node_array__zzprivate_locate(array_base, index, &run, &within);
    return sys__node_array__zzprivate_in_run(run, within);
}

static __device__ inline bool sys__node_array__is(uint64_t array_base) {
    if (!sys__heap_object__zzpackage_addressable(array_base)) return false;
    sys__heap_node* head = sys__heap_node__zzpackage_head(array_base);
    return sys__heap_object__is_type_from_head(head, SYS__KIND__NODE_ARRAY);
}

static __device__ inline uint64_t sys__node_array__length(uint64_t array_base) {
    if (!sys__node_array__is(array_base)) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_KIND);
        return 0ull;
    }
    /* Each run's head is one node and its elements are the rest less a cone, so the length is what the
     * allocations reach. Nothing records it separately, which is the point: a second copy of a number is a
     * second thing that can be wrong, and this one is already written by the carve. An array that fits is
     * one run and answers off its head; a chained one is every run but the last, full, and the last. */
    const sys__heap_node* cone = sys__node_array__zzprivate_cone(array_base);
    const uint64_t table = sys__node_array__zzprivate_indexer(array_base, cone);
    if (table == 0ull) return sys__node_array__zzprivate_length_by_cones(array_base);
    const uint64_t runs = sys__node_array__zzprivate_length_by_cones(table);
    const uint64_t last = sys__node_array__zzprivate_at_by_cones(table, runs - 1ull)->args[0];
    return (runs - 1ull) * (SYS__NODE_ARRAY__RUN_CELLS - 1ull)
         + sys__node_array__zzprivate_run_length(last, sys__node_array__zzprivate_cone(last));
}

/* One question, asked the same way by every verb that reaches an element — the three that read one and the
 * one that writes it — so they cannot come to different conclusions about the same reference and the same
 * index. It raises where it refuses, which is why none of them does. */
static __device__ inline bool sys__node_array__zzprivate_reaches(uint64_t array_base, uint64_t index) {
    if (!sys__node_array__is(array_base)) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_KIND);
        return false;
    }
    if (index >= sys__node_array__length(array_base)) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_RANGE);
        return false;
    }
    return true;
}

/* The cell at an index, with nothing checked and no hold taken. ⛔ IT IS `zzpackage_` AND NOT PUBLIC
 * BECAUSE IT TRUSTS ITS CALLER TWICE OVER: that the base names an array, and that the index is inside it.
 * ⛳ AND INSIDE ITS FIRST RUN, which both callers have by construction: the bindings make their tables
 * through `zzpackage_create_run`, which refuses rather than chains, and a procedure is two elements. An
 * array placed in room somebody else owns is one run however long it is.
 * Two callers have earned that, and they earn it by different arguments. The bindings reach checked both
 * when it armed and holds a base that cannot move — `args[SCOPES]` is written at creation and at teardown
 * and nowhere else. The procedure verbs ask `sys__procedure__is` first, which is the kind, and index with
 * `SYS__PROCEDURE__PARAMETERS` and `SYS__PROCEDURE__BODY` — constants into a thing that is two elements by
 * construction, so the range is settled where it is written rather than checked where it is read.
 * ⛳ A FURTHER CALLER IS A DECISION, NOT AN ADDITION: it owes one of those two arguments in full, or it
 * reaches through a verb that checks. */
static __device__ inline sys__heap_node* sys__node_array__zzpackage_at(uint64_t array_base, uint64_t index) {
    return sys__node_array__zzprivate_in_run(array_base, index);
}

static __device__ inline sys__heap_node* sys__node_array__cells(uint64_t array_base, uint64_t count) {
    if (count == 0ull || !sys__node_array__zzprivate_reaches(array_base, count - 1ull)) return (sys__heap_node*)0;
    /* The answer is one pointer and the caller adds to it, so the cells have to be one run. An array long
     * enough to be chained still answers when the count fits in its first run. */
    if (count > sys__node_array__zzprivate_run_length(array_base, sys__node_array__zzprivate_cone(array_base))) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_SPLIT);
        return (sys__heap_node*)0;
    }
    return sys__node_array__zzprivate_in_run(array_base, 0ull);
}

static __device__ inline sys__heap_node sys__node_array__get(uint64_t array_base, uint64_t index) {
    if (!sys__node_array__zzprivate_reaches(array_base, index)) return sys__heap_node__nothing();
    sys__heap_node value = *sys__node_array__zzprivate_at(array_base, index);
    /* ⭐ A GET HANDS OUT A HOLD, AND THE CONSUMER GIVES IT BACK. ⚖ ARCHITECT: *"doing a get would include
     * a retain, which needs to be decremented by the consumer explicitly ... it is the same logic we
     * applied for the stack after all."* The alternative is a caller holding an offset the array can drop
     * from under it between the read and the use, which is a race nobody can see and everybody can write.
     * ⛳ AND IT IS WHY A LOOK IS NOT FREE: a caller that only wanted to see what is there still owes one
     * release, the same way peeking at a stack does. */
    if (sys__heap_node__carries_reference(value.dtype)) (void)sys__heap_object__retain(value.args[0]);
    return value;
}

static __device__ inline sys__heap_node sys__node_array__borrow(uint64_t array_base, uint64_t index) {
    if (!sys__node_array__zzprivate_reaches(array_base, index)) return sys__heap_node__nothing();
    return *sys__node_array__zzprivate_at(array_base, index);
}

static __device__ inline sys__kind sys__node_array__type(uint64_t array_base, uint64_t index) {
    if (!sys__node_array__zzprivate_reaches(array_base, index)) return SYS__KIND__INVALID;
    return sys__node_array__zzprivate_at(array_base, index)->dtype;
}

static __device__ inline bool sys__node_array__set(uint64_t array_base, uint64_t index,
                                                    const sys__heap_node* value) {
    if (value == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return false;
    }
    /* A cone is the array's own joint and never an element: one stored here would read as the end of a
     * run to every verb that came after it. */
    if (value->dtype == SYS__KIND__NODE_ARRAY_CONE) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_KIND);
        return false;
    }
    if (!sys__node_array__zzprivate_reaches(array_base, index)) return false;
    /* ⭐ THE ARRAY OWNS WHAT IT HOLDS, so storing takes a hold and what was there gives one up — and the
     * order of those two is the whole of why this is one call and not two lines here: they can be the
     * same object, and letting go first would take it apart and then store a reference to the room.
     * ⛳ IT IS SAFE ON A FRESH ELEMENT because `create` writes a null into every one of them before the
     * array is handed out, so there is never poison here to be mistaken for a reference. */
    sys__heap_object__set(sys__node_array__zzprivate_at(array_base, index), value);
    return true;
}

/* ⛔ EVERY ELEMENT IS WRITTEN BEFORE THE ARRAY IS HANDED OUT, AND THAT IS NOT TIDINESS. The room is not
 * zeroed, so a fresh element holds whatever the last occupant left — and a reader asking a fresh array
 * what it holds would be told by that. Filling with the null value makes "nothing has been put here" the
 * answer to the same question every other state is answered by, rather than something a caller has to
 * know not to ask.
 * ⛳ ONE BODY, because both ways of making an array owe it and a second copy is a second thing to keep
 * true — and the one that got missed would be the one nobody tests, since a poisoned element reads as a
 * plausible value rather than as a fault. */
static __device__ inline void sys__node_array__zzprivate_fill(sys__heap_node* head, uint64_t elements) {
    for (uint64_t i = 1ull; i <= elements; ++i) {
        head[i].dtype    = SYS__KIND__VALUE_NULL;
        head[i].num_args = 0u;
        head[i].args[0]  = 0ull;
    }
}

static __device__ inline uint64_t sys__node_array__zzpackage_create_run(uint64_t elements) {
    if (elements == 0ull) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_EMPTY);
        return 0ull;
    }
    /* The head is a node of its own, so the room asked for is one more than the count. Past one chunk the
     * allocator refuses, on geometry, and says so. */
    sys__heap_node* head = sys__heap__zzpackage_make_sized(SYS__KIND__NODE_ARRAY, elements + 1ull);
    if (head == 0) return 0ull;                  /* the carve raised, and said which of its reasons */

    sys__node_array__zzprivate_fill(head, elements);
    /* A reference names the node after the head, which here is element zero — so the thing a caller holds
     * and the thing the arithmetic starts from are the same offset, and neither has to be converted. */
    return sys__heap__offset(head + 1);
}

/* The runs of a chained array, made back to front so each cone is written naming a run that already
 * exists — the hold that run was made with becomes the cone's, and nothing is retained or let go on the
 * way. Every run but the last is full and gives its last cell to the cone; the last takes what is left, up
 * to a full run of elements. When a `table` is handed in, it is told where each run starts and the first
 * run's head names it; without one, every head says there is none.
 * ⛳ A RUN THAT CANNOT BE MADE TAKES THE ONES ALREADY MADE WITH IT: letting go of the newest reaches every
 * run behind it through the cones, by the same teardown any array dies by. The table is the caller's. */
static __device__ inline uint64_t sys__node_array__zzprivate_chain(uint64_t elements, uint64_t table) {
    const uint64_t chained = SYS__NODE_ARRAY__RUN_CELLS - 1ull;          /* elements in a run with a cone */
    uint64_t runs_before = 0ull;
    uint64_t tail_elements = elements;
    while (tail_elements > SYS__NODE_ARRAY__RUN_CELLS) { tail_elements -= chained; ++runs_before; }

    uint64_t run = sys__node_array__zzpackage_create_run(tail_elements);
    uint64_t r = runs_before;
    while (run != 0ull) {
        sys__heap_node__zzpackage_head(run)->args[SYS__NODE_ARRAY__INDEXER] = 0ull;
        if (table != 0ull) {
            sys__heap_node* entry = sys__node_array__zzprivate_at_by_cones(table, r);
            entry->dtype = SYS__KIND__VALUE_INT; entry->num_args = 0u; entry->op_code = 0ull;
            entry->args[0] = run;
        }
        if (r == 0ull) break;
        const uint64_t before = sys__node_array__zzpackage_create_run(SYS__NODE_ARRAY__RUN_CELLS);
        if (before == 0ull) { (void)sys__heap_object__release(run); return 0ull; }
        sys__heap_node* cone = sys__node_array__zzprivate_in_run(before, chained);
        cone->dtype    = SYS__KIND__NODE_ARRAY_CONE;
        cone->num_args = 0u;
        cone->op_code  = 0ull;
        cone->args[0]  = run;
        run = before;
        --r;
    }
    if (run != 0ull) sys__heap_node__zzpackage_head(run)->args[SYS__NODE_ARRAY__INDEXER] = table;
    return run;
}

static __device__ inline uint64_t sys__node_array__create(uint64_t elements) {
    if (elements <= SYS__NODE_ARRAY__RUN_CELLS) return sys__node_array__zzpackage_create_run(elements);

    /* ⭐ TOO LONG FOR ONE RUN, SO IT IS SEVERAL, AND A TABLE OF THEM. The table is itself a node array,
     * one element a run; past a run's width of runs it is chained too, and keeps no table of its own —
     * reading it then costs a hop per run of the table, which is a quarter of a million elements in. */
    const uint64_t chained = SYS__NODE_ARRAY__RUN_CELLS - 1ull;
    uint64_t runs = 1ull;
    for (uint64_t tail = elements; tail > SYS__NODE_ARRAY__RUN_CELLS; tail -= chained) ++runs;
    const uint64_t table = (runs <= SYS__NODE_ARRAY__RUN_CELLS) ? sys__node_array__zzpackage_create_run(runs)
                                                                : sys__node_array__zzprivate_chain(runs, 0ull);
    if (table == 0ull) return 0ull;                 /* the carve raised, and said which of its reasons */
    const uint64_t array = sys__node_array__zzprivate_chain(elements, table);
    if (array == 0ull) { (void)sys__heap_object__release(table); return 0ull; }
    return array;
}

/* ── WALKING ONE ────────────────────────────────────────────────────────────────────────────────────
 * Every element in order, stepping over the cones. A step inside a run is one addition; the step that
 * lands on a cone follows it, so a reader never sees one. */
static __device__ inline void sys__node_array__zzprivate_walk_settle(sys__node_array_walk* w) {
    if (w->cell != w->last) return;
    const sys__heap_node* at = sys__node_array__zzprivate_in_run(w->cell, 0ull);
    if (at->dtype != SYS__KIND__NODE_ARRAY_CONE) return;
    w->cell = at->args[0];
    w->last = w->cell + SYS__HEAP_OBJECT__NODES(sys__heap_node__zzpackage_head(w->cell)) - 2ull;
}

static __device__ inline bool sys__node_array__walk(uint64_t array_base, uint64_t from, sys__node_array_walk* w) {
    if (w == 0) return false;
    w->cell = 0ull; w->last = 0ull;
    if (!sys__node_array__is(array_base)) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_KIND);
        return false;
    }
    /* Into the run `from` falls in — what an index costs, paid once for the walk. Opening at the length is
     * a walk already at its end, as an empty loop over the rest would be. */
    uint64_t run = 0ull, within = 0ull;
    if (!sys__node_array__zzprivate_locate(array_base, from, &run, &within)) {
        if (within == sys__node_array__zzprivate_run_length(run, 0)) return true;
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_RANGE);
        return false;
    }
    from = within;
    w->cell = run + from;
    w->last = run + SYS__HEAP_OBJECT__NODES(sys__heap_node__zzpackage_head(run)) - 2ull;
    sys__node_array__zzprivate_walk_settle(w);
    return true;
}

static __device__ inline sys__heap_node* sys__node_array__walk_cell(const sys__node_array_walk* w) {
    if (w == 0 || w->cell == 0ull) return (sys__heap_node*)0;
    return sys__node_array__zzprivate_in_run(w->cell, 0ull);
}

/* Put a value in the element the walk stands on, exactly as `set` puts one by index: the array takes a hold
 * of what goes in and gives up its hold on what was there. A walk at its end has no element and refuses. */
static __device__ inline bool sys__node_array__walk_set(const sys__node_array_walk* w, const sys__heap_node* value) {
    if (value == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return false;
    }
    if (value->dtype == SYS__KIND__NODE_ARRAY_CONE) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_KIND);
        return false;
    }
    sys__heap_node* cell = sys__node_array__walk_cell(w);
    if (cell == 0) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_RANGE);
        return false;
    }
    sys__heap_object__set(cell, value);
    return true;
}

static __device__ inline void sys__node_array__next(sys__node_array_walk* w) {
    if (w == 0 || w->cell == 0ull) return;
    if (w->cell == w->last) { w->cell = 0ull; return; }       /* the last run's last element: the end */
    w->cell += 1ull;
    sys__node_array__zzprivate_walk_settle(w);
}

/* ── PLACING ONE IN ROOM SOMEBODY ELSE OWNS ──────────────────────────────────────────────────────────
 * The same array, in memory the caller names instead of memory the asking block was carving from.
 *
 * ⭐⭐ WHY THIS EXISTS, AND IT IS ONE CASE RATHER THAN A GENERAL FACILITY. Carving from a block's
 *   provider means asking which chunk that block is carving from, and the answer lives in the block's
 *   computing base. Anything whose ADDRESS has to be known before a computing base can be found cannot
 *   be made that way — it would need the answer in order to be built, and the answer is what it is being
 *   built to hold. The boot path meets exactly that circle once and this is how it gets out of it, the
 *   same way the block's own base gets out of its.
 * ⛔ THE ROOM MUST LIE INSIDE THE MEMORY THE ALLOCATOR WAS PUBLISHED OVER, and that is a real
 *   precondition rather than a preference: what comes back is a heap OFFSET, and an offset is a
 *   subtraction from the base the allocator was handed. Room outside it produces a number that reads
 *   like an offset and names nothing.
 * ⛳ WHICH LEAVES ONE PLACE THAT SATISFIES BOTH HALVES: inside the ALLOCATION and outside the POOL. The
 *   allocator is told how many chunks it has, so memory past them is addressable and unreachable at the
 *   same time — it can never be handed out, and it can still be named. ⇒ ★ AN OFFSET IS A PROPERTY OF
 *   THE ALLOCATION; BEING HANDED OUT IS A PROPERTY OF THE POOL, AND THEY ARE NOT THE SAME BOUNDARY.
 * ⚠ NOTHING RELEASES WHAT IS PLACED HERE, and nothing can: a chunk's occupancy is what a release
 *   decrements, and this room belongs to no chunk. What is placed this way is perennial by construction,
 *   which is correct for the one customer and is why this is not published to the language. */
static __device__ inline uint64_t sys__node_array__zzpackage_place(sys__heap_node* room, uint64_t elements) {
    if (elements == 0ull) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_EMPTY);
        return 0ull;
    }
    if (room == 0 || !sys__heap__zzpackage_published()) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_KIND);
        return 0ull;
    }
    /* The head, written exactly as a carve writes one — every field, because the room was not zeroed.
     * ⛳ AND NO CHUNK IS TOLD, which is the one line a carve has that this does not: there is no chunk
     * holding this, so there is nothing whose occupancy would have to come back down. */
    room[0].dtype    = SYS__KIND__NODE_ARRAY;
    room[0].num_args = 1u;
    room[0].args[SYS__HEAP_OBJECT__LOCK] = 0ull;
    SYS__HEAP_OBJECT__NODES(&room[0]) = elements + 1ull;
    sys__node_array__zzprivate_fill(room, elements);
    return sys__heap__offset(room + 1);
}

static __device__ __noinline__ void sys__node_array__zzpackage_release_internal(sys__heap_node* head,
                                                                          uint64_t* releaser_stack) {
    if (head == 0) return;
    /* Every element that named an object moves onto the stack the teardown is working through, and the
     * test for that is here rather than anywhere downstream: `sys__stack__transfer` filters nothing, so a
     * number or a null handed to it would be carried like anything else — worklist chunks carved to hold
     * cells the drain discards the instant it pops them. ⛳ The drain tests as well, and that is not a copy
     * of this one: it guards against whatever any mover pushes, while this keeps an array from pushing
     * what there is nothing to release.
     *
     * ⛔ IT MOVES AND DOES NOT LET GO, AND THE DIFFERENCE IS NOT VOCABULARY. Releasing here would unwind
     * from inside this function, one C frame for every level of structure below it, and an array of
     * arrays of arrays is then a blown runner-thread stack. Moving keeps the
     * unwinding one loop at one depth however deep the structure is — and it costs no arithmetic at all,
     * because the count does not change when the owner does.
     *
     * ⛳ THE WALK IS OVER THE ALLOCATION'S OWN LENGTH, so it cannot read past the array and cannot stop
     * short of it: the head records how far this one reaches and the elements are all of it but the head.
     * Nothing here needs the length verb, which asks what kind this is — and what kind it is, is how the
     * teardown got here. */
    /* ⛳ A CONE IS MOVED LIKE ANY REFERENCE, which is how the runs after this one die with it: each is
     * an array in its own right, and the drain reaches it the way it reaches any other. */
    const uint64_t nodes = SYS__HEAP_OBJECT__NODES(head);
    /* ⛳ AND THE TABLE OF RUNS GOES WITH THE FIRST, moved the same way. Only an array starts a chain, so a
     * procedure — which shares this teardown — is never asked; the word is cleared as it is moved. */
    if (head->dtype == SYS__KIND__NODE_ARRAY) {
        const uint64_t first = sys__heap__offset(head + 1);
        const uint64_t table = sys__node_array__zzprivate_indexer(first, sys__node_array__zzprivate_cone(first));
        if (table != 0ull) {
            head->args[SYS__NODE_ARRAY__INDEXER] = 0ull;
            sys__heap_node held = sys__heap_object__reference_to(table);
            sys__stack__transfer(releaser_stack, &held);
        }
    }
    for (uint64_t i = 1ull; i < nodes; ++i)
        if (sys__heap_node__carries_reference(head[i].dtype))
            sys__stack__transfer(releaser_stack, &head[i]);
}

static __device__ inline sys__heap_node sys__node_array__zzpackage_construct(sys__heap_node* base,
                                                                          const sys__heap_node* parameters) {
    (void)base;
    if (parameters == 0) {
        sys__fault__raise(0ull, SYS__HEAP_OBJECT__FAULT_NO_VALUE);
        return sys__heap_node__nothing();
    }
    const uint64_t at = sys__node_array__create(parameters->args[0]);
    if (at == 0ull) return sys__heap_node__nothing();          /* the making already raised */
    return sys__heap_object__reference_to(at);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_NODE_ARRAY__IMPL_CUH */
