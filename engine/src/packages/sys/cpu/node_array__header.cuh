#ifndef SILVANN__PACKAGES_SYS_CPU_NODE_ARRAY__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_NODE_ARRAY__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../contracts/objects/node_array.cuh" /* its constants and fault words */
/* ══ a node array — n contiguous nodes, asked for by how many ════════════════════════════════════════
 *
 * ⚖ ARCHITECT: *"no matter what i do with the allocations of the values [in the system register] they
 * need to be in the heap, so i might as well transform the array create function into something real that
 * can allocate at runtime and takes possession of n contiguous ast nodes from the heap chunk ... since
 * when they become normal ast objects we can use the primitives directly onto them."*
 *
 * ⭐ SO IT IS AN ORDINARY OBJECT AND THE WHOLE POINT IS THAT IT IS. It is made through the allocator,
 * counted like anything else, named by a reference like anything else, and asked what it is by the same
 * verb that asks anything else. What it adds is one idea: the room is CONTIGUOUS and its extent was the
 * caller's to choose, so element `i` is at `base + i` and finding one is an addition rather than a walk.
 *
 * ⛳ WHICH IS ALSO EXACTLY WHAT A LIST IS NOT, AND THE DIFFERENCE IS WHAT PICKS BETWEEN THEM. A list
 * exists for growth, insertion and an extent nobody knows yet, and it pays for those with a walk. An
 * array has none of them and pays for none of them. Where the count is settled once and the access is by
 * position, the list is the wrong container and not merely the slower one.
 *
 * ── WHAT AN ELEMENT IS ──────────────────────────────────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"qty is the only variable and the type is sysastnode."* Every element is a whole node, so
 * an element carries its own type and can be asked what it holds. That is what lets one array hold a
 * number in one element and a reference to an object in the next without anybody having been told which
 * to expect — and it is the reason the type is not a parameter: a program reaching an element has to be
 * able to ask, so the answer has to be in the element.
 *
 * ⛔ THE EXTENT LIVES IN THE ALLOCATION AND NOT IN THE KIND'S ROW, SO THIS KIND HAS NO ONE SIZE. Like a
 * string, it is asked for a count and carved to fit it, and the length is read back off the head that
 * already records how far an allocation reaches.
 * Its row still names a size, and what that size means is the SMALLEST array — one element — because a
 * reference names the node after the head and an array with no elements would hand out a reference to
 * memory it does not own.
 *
 * ── AN ARRAY OWNS WHAT IT HOLDS, AND SO DOES WHOEVER ASKED IT FOR SOMETHING ─────────────────────────
 * ⚖ ARCHITECT: *"an array holding an object should increase that object's arc and doing a get would
 * include a retain, which needs to be decremented by the consumer explicitly ... it is the same logic we
 * applied for the stack after all."* So the three halves of it, and they are one rule read from three
 * sides:
 *     PUTTING ONE IN takes a hold, and whatever was in that element gives one up.
 *     TAKING ONE OUT takes a hold too, and the caller owes it back — a look is not free.
 *     THE ARRAY DYING hands every element it owned to the chain that is draining it.
 *
 * ⭐ WHY A GET RETAINS, WHICH IS THE HALF THAT SURPRISES PEOPLE. Without it a caller holds an offset the
 * array can drop from under it between the read and the use — the element is overwritten, the last hold
 * goes, the room comes back, and the caller is reading somebody else's object. That is a race nobody can
 * see in a diff and anybody can write. Costing every look one release is the price of it being impossible.
 *
 * ⛔ AND THE DYING ARRAY HANDS OVER RATHER THAN LETTING GO, WHICH IS NOT VOCABULARY. A container that
 * released its contents would drive the unwinding from inside itself, one C frame per level of structure,
 * and an array of arrays of arrays is a blown runner-thread stack. Handing over keeps
 * the unwinding one loop at one depth however deep the structure goes — which is what the whole teardown
 * is built around, and why a row may hand over and may not release.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⛔ NARN ANSWERS TWO DIFFERENT MISTAKES AND A LOG SHOWS ONLY THE WORD. Most of its raises are the one
 * the mnemonic names: a reference asked to be an array when it is something else. The other is
 * `zzpackage_place` refusing to place — no room handed in, or no heap published to turn that room into
 * an offset — which is a boot-order mistake rather than a bad reference, and the heap has a word of its
 * own for the unpublished half (`SYS__HEAP__FAULT_NOT_PUBLISHED`) that this raise does not use.
 * ⇒ ⛳ MEETING IT BEFORE THE ALLOCATOR IS PUBLISHED POINTS AT THE PLACEMENT, NOT AT THE CALLER. */

/* Make one, `elements` long, and answer where its first element sits. Zero is refused rather than served,
 * because the reference this hands back names the first element and an array without one has nothing to
 * name. Every element starts as a null value, so the first thing written over is a known state and not
 * whatever the room held before. An array too long for one run is made as several, joined by cones. */
static __device__ inline uint64_t   sys__node_array__create(uint64_t elements);
/* The same, as ONE run or not at all: past a run's width the allocator refuses, on geometry. For a caller
 * that hands out the address of an element and adds to it, which only one run can promise. */
static __device__ inline uint64_t   sys__node_array__zzpackage_create_run(uint64_t elements);
/* The same array in room the CALLER owns, for the one thing whose address must be known before a
 * computing base can be found. The room must lie inside the memory the allocator was published over —
 * see the body, which argues the precondition rather than stating it. */
static __device__ inline uint64_t   sys__node_array__zzpackage_place(sys__heap_node* room, uint64_t elements);

/* How many elements it has, read off the allocation rather than remembered anywhere. */
static __device__ inline uint64_t   sys__node_array__length(uint64_t array_base);

/* ── WALKING ONE, IN ORDER ───────────────────────────────────────────────────────────────────────────
 * `walk` opens a walk on element `from` — at the length it opens already at its end, past it it refuses
 * with NARO — `walk_cell` answers the element it stands on, nothing once it has run off the end, and
 * `next` steps to the following one, following a cone where a run ends. `walk_set` writes the element
 * it stands on with `set`'s holds. Reading this way costs one addition a step however long the array is,
 * where `get` by index costs a hop per run it has to pass. ⛔ A WALK TAKES NO HOLD: what `walk_cell`
 * answers is the cell itself, read in place the way `borrow` reads, and safe where `borrow` is. A value
 * goes in through `walk_set`, never through the pointer, or the holds go wrong. */
static __device__ inline bool            sys__node_array__walk(uint64_t array_base, uint64_t from,
                                                               sys__node_array_walk* w);
static __device__ inline sys__heap_node* sys__node_array__walk_cell(const sys__node_array_walk* w);
static __device__ inline void            sys__node_array__next(sys__node_array_walk* w);
static __device__ inline bool            sys__node_array__walk_set(const sys__node_array_walk* w,
                                                                   const sys__heap_node* value);

/* Whether this names one at all. Asked before the reach by every verb that checks — they share one check,
 * so they cannot come to different conclusions about the same reference — and by `length` before it answers.
 * `zzpackage_at` is the one that does not ask, and it says on its own declaration why it is allowed not to.
 * Worth having on its own for a caller that was handed a reference and has no reason to trust it. */
static __device__ inline bool       sys__node_array__is(uint64_t array_base);

/* The element at `index`, by value — a whole node, so the caller gets the type with it. An element naming
 * an object comes back HELD, and the caller gives that back when it is done. An index past the end answers
 * nothing, having raised, and holds nothing: there was no element to take. */
static __device__ inline sys__heap_node sys__node_array__get(uint64_t array_base, uint64_t index);

/* The cell at an index, with nothing checked and no hold taken. ⛔ `zzpackage_` because it trusts its
 * caller twice over: that the base names an array, and that the index is inside it — inside its FIRST
 * RUN, which an array made by `zzpackage_create_run` or placed is all of. TWO callers have
 * earned that and they earn it by DIFFERENT arguments — the bindings reach by holding a base that cannot
 * move and having checked both when it armed, the procedure verbs by asking the kind first and indexing
 * with constants into a thing that is two elements by construction. The implementation states each in
 * full. ⛳ A FURTHER CALLER IS A DECISION, NOT AN ADDITION: it owes one of those two arguments whole, or
 * it reaches through a verb that checks. */
static __device__ inline sys__heap_node* sys__node_array__zzpackage_at(uint64_t array_base, uint64_t index);

/* ⭐ WHERE THE FIRST `count` ELEMENTS SIT, for a door that writes values into nodes a program made for it —
 * ⚖ *"i can allocate the return nodes beforehand and send the node array base address as an object so the
 * gpu has a base to compute its output slots"*. Element `i` is the answer plus `i`. Zero when this is not
 * an array or it is shorter than `count`, and zero with NARS when the `count` crosses from one run into
 * the next, because one address cannot reach past a cone. ⛔ IT CHECKS BOTH THINGS `zzpackage_at` TRUSTS, and it is for
 * VALUE WORDS ONLY: a caller writes `args[0]` of elements that hold no reference, having asked each one's
 * kind first, so no hold is ever made or lost behind the array's back. */
static __device__ inline sys__heap_node* sys__node_array__cells(uint64_t array_base, uint64_t count);

/* The element at `index`, by value, WITHOUT a hold — for an array nobody writes, read while something
 * already holds it. It is what lets any number of blocks read one array at once touching nothing but the
 * memory: `get` puts an atomic on the element's count for every look, and every block looking at the same
 * element queues on the same word.
 * ⛔ IT IS SAFE ONLY WHERE THE REASON FOR `get`'S HOLD DOES NOT APPLY. That hold exists so an element
 * cannot be dropped between the read and the use, and dropping one takes somebody WRITING the element. An
 * array nothing writes, held by whoever reads it, cannot lose one. A picture of a list is such an array;
 * anything else is read through `get`. */
static __device__ inline sys__heap_node sys__node_array__borrow(uint64_t array_base, uint64_t index);

/* What KIND of thing an element holds, WITHOUT taking a hold of it — the cheap question, for a caller
 * that wants to know what is there rather than to have it. `get` is the other one and it costs a hold;
 * asking through `get` and giving the hold straight back would be two atomics to answer a question that
 * touches no counts at all. An element outside the array answers `SYS__KIND__INVALID`, having raised —
 * a third answer, so a bad index is never read as an empty element. */
static __device__ inline sys__kind sys__node_array__type(uint64_t array_base, uint64_t index);

/* Put a value in an element. The array takes a hold of what goes in and gives up its hold on what was
 * there. Answers whether it went in; the ways it does not are a null value, a bad reference and an index
 * past the end, and all three raise. ⛳ THE FIRST ANSWERS IN A WORD THIS FILE DOES NOT DECLARE —
 * `SYS__HEAP_OBJECT__FAULT_NO_VALUE`, the heap object's word for a cell or value handed in null — so a
 * reader chasing what `set` can put in a log has three words to look for and only two of them here. */
static __device__ inline bool       sys__node_array__set(uint64_t array_base, uint64_t index,
                                                          const sys__heap_node* value);

/* ── THE TWO ROW METHODS ─────────────────────────────────────────────────────────────────────────────
 * What the kind's row names, and neither is called by hand: one runs when the last hold goes, the other
 * when a program asks for one. They are declared here because the row that names them is read in a file
 * that has never heard of this one.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* What an array was holding when it died: every element that named an object, moved onto the chain the
 * release loop drains. It lets go of nothing itself — see the paragraph above for why that distinction is
 * what keeps a deep structure from unwinding through the call stack. */
static __device__ __noinline__ void sys__node_array__zzpackage_release_internal(sys__heap_node* head,
                                                                          uint64_t* releaser_stack);

/* Building one from a program. It takes one argument — how many elements — so it is expressible with the
 * arguments a single node carries and needs nothing that does not exist yet. */
static __device__ inline sys__heap_node sys__node_array__zzpackage_construct(sys__heap_node* base,
                                                                          const sys__heap_node* parameters);

#endif /* SILVANN__PACKAGES_SYS_CPU_NODE_ARRAY__HEADER_CUH */
