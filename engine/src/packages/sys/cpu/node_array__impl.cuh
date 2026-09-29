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

/* Where an element sits, which is the whole of what an array is for: a reference names element zero, so
 * element `i` is `i` nodes along from it. One line, and it is here rather than written out five times
 * because the five verbs below would each have had to get the same arithmetic right. */
static __device__ inline sys__heap_node* sys__node_array__zzprivate_at(uint64_t array_base, uint64_t index) {
    return sys__heap__object_full_address(array_base + index);
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
    /* The head is one node and the elements are the rest, so the length is what the allocation reaches
     * minus that one. Nothing records it separately, which is the point: a second copy of a number is a
     * second thing that can be wrong, and this one was already being written by the carve. */
    return SYS__HEAP_OBJECT__NODES(sys__heap_node__zzpackage_head(array_base)) - 1ull;
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
 * Two callers have earned that, and they earn it by different arguments. The bindings reach checked both
 * when it armed and holds a base that cannot move — `args[SCOPES]` is written at creation and at teardown
 * and nowhere else. The procedure verbs ask `sys__procedure__is` first, which is the kind, and index with
 * `SYS__PROCEDURE__PARAMETERS` and `SYS__PROCEDURE__BODY` — constants into a thing that is two elements by
 * construction, so the range is settled where it is written rather than checked where it is read.
 * ⛳ A FURTHER CALLER IS A DECISION, NOT AN ADDITION: it owes one of those two arguments in full, or it
 * reaches through a verb that checks. */
static __device__ inline sys__heap_node* sys__node_array__zzpackage_at(uint64_t array_base, uint64_t index) {
    return sys__node_array__zzprivate_at(array_base, index);
}

static __device__ inline sys__heap_node* sys__node_array__cells(uint64_t array_base, uint64_t count) {
    if (count == 0ull || !sys__node_array__zzprivate_reaches(array_base, count - 1ull)) return (sys__heap_node*)0;
    return sys__node_array__zzprivate_at(array_base, 0ull);
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

static __device__ inline uint64_t sys__node_array__create(uint64_t elements) {
    if (elements == 0ull) {
        sys__fault__raise(0ull, SYS__NODE_ARRAY__FAULT_EMPTY);
        return 0ull;
    }
    /* The head is a node of its own, so the room asked for is one more than the count. */
    sys__heap_node* head = sys__heap__zzpackage_make_sized(SYS__KIND__NODE_ARRAY, elements + 1ull);
    if (head == 0) return 0ull;                  /* the carve raised, and said which of its reasons */

    sys__node_array__zzprivate_fill(head, elements);
    /* A reference names the node after the head, which here is element zero — so the thing a caller holds
     * and the thing the arithmetic starts from are the same offset, and neither has to be converted. */
    return sys__heap__offset(head + 1);
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
    const uint64_t nodes = SYS__HEAP_OBJECT__NODES(head);
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
