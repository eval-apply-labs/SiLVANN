#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_STACK_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_STACK_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "heap.cuh"
#include "heap_node.cuh"

/* ── THE LAYOUT ───────────────────────────────────────────────────────────────────────────────────── */
/* The chunk's state values live in the node its allocation begins with, beside the kind and the count.
 * That node is 64 bytes and those two use eight of them, so there was room and the cells are values and
 * nothing else — which is worth two slots per chunk and, more than that, means a walk over the cells is a
 * walk over values with nothing to skip.
 *
 * ⭐ AND THE LINK IS A PLAIN OFFSET, NOT A TAGGED CELL. Nothing walks a chunk's cells blindly: what takes
 * one apart knows it is a chunk, because the head says so, and follows the previous because it knows
 * where it is. A tag would be a second way of saying the same thing, and two of those can disagree.
 *
 * The cursor counts what is here rather than pointing at the top: zero is empty, and the top is one below
 * the count. That way the empty state needs no reserved index and the first value can live at zero.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
#define SYS__STACK_CHUNK__HEAD_CURSOR  1u   /* args[1] of the head: how many values are in this chunk */
#define SYS__STACK_CHUNK__HEAD_PREVIOUS    2u   /* args[2] of the head: the chunk before this one, 0 for none */

/* A stack chunk is a fixed run of nodes carved out of the chunk the block is carving ARRAYS from, 
 * which also supplies lists and other array chunk consumer.
 * We do need to inherit their size, both to recycle the allocator and to maintain an aligned memory address.
 * A stack that wants to be bigger takes another chunk, which is what the previous is for.
 *
 * ⛔ THE WIDTH IS A CAPACITY AND NOTHING ELSE — how many cells one link of the chain holds before another
 * is needed — AND IT IS NOT AN ALIGNMENT. Nothing is rounded up to it and no address in the heap package
 * is masked: an allocation is placed where the last one ended, and a head is found by subtracting one node
 * from a REFERENCE, which is the only thing any holder has. `REASONED` from `sys__heap_node__head` and the
 * watermark carve, whose own comments say both in as many words.
 * ⇒ THE ALTERNATIVE — a power-of-two width, so that clearing the low bits of any interior address reached
 * its chunk's beginning in one instruction — is the one thing `heap__impl.cuh` forbids putting back, and
 * it gives the reason: an aligned offset inside a chunk whose base is aligned to nothing is an unaligned
 * address. The trick needs the base requirement as well as the placement, so both halves come back
 * together or neither does — and the placement alone delivers none of it while reading as though it did.
 * What packing at the watermark buys back is that a kind can be exactly as wide as it asked for.
 *
 * The first node of the allocation is the count of what refers to this stack. It sits there rather than in
 * a table because that is where the address already is — a holder that has the stack has the count, with
 * no second structure to consult and nothing that can disagree with it.
 *
 * What is left after the count is the reference to the LIFO stack chunk, and it is the stack chunk that 
 * the four private operations speak about, so every index below is relative to that internal stack chunk
 * and none of them has to know the count. */
/* ⚖ A DIAL OF ITS OWN, BECAUSE A STACK AND A LIST WANT DIFFERENT NUMBERS. A binding stack is deep and
 * narrow — one symbol pushed and popped — where an execution list is many SMALL forms, and a form of three
 * values in a wide chunk pays a dozen nodes for every one it uses. One number cannot be both, so there are
 * two. ⚖ ARCHITECT: *"since it is a default value we can change it without too much effort and see how
 * much does the scenery changes."*
 * ⛔⛔ AND THE DIAL DOES NOT REACH THE ALLOCATION, SO IT CANNOT BE TURNED UP AS IT STANDS. What it sets
 * is `SYS__STACK_CHUNK__SLOTS` and the width of `sys__stack_chunk` below, and nothing else. How many nodes
 * the heap carves for a chunk is the size column of the `SYS__KIND__STACK_CHUNK` row in
 * `sys__LANGUAGE_CONTRACT__OBJECTS`, and that column names `SYS__ALLOC__ARRAY_GRANULE` outright — where
 * `SYS__KIND__LIST_CHUNK`'s names `SYS__LIST_CHUNK__ALLOC`, which is the wiring this one wants. So a raised
 * dial lays a wider struct over a granule-sized allocation, and the push that fills it writes past the end
 * of that allocation into whatever the watermark placed next. `REASONED` from those two rows in
 * `language_contract.cuh` and from `sys__heap__object_nodes`, which reads that column; the premise is that
 * `sys__stack_chunk__zzprivate_create` reaches the heap only through `sys__heap__make`.
 * ⛔ AND NEITHER ASSERT BELOW SEES IT, because both are written in the dial's own terms: a width that
 * still divides the heap chunk passes while naming a size the allocator was never told. ⇒ ⭐ AN ASSERT
 * THAT READS THE MACRO THE MISTAKE IS IN CANNOT BE THE ONE THAT CATCHES IT. The repair is one word in that
 * row; until it is written, the quote above says what the dial is FOR rather than what it does.
 * ⛳ THE TRADE IS THE HEAD, and it is why smaller is not simply better: every chunk costs one node of
 * header, so halving the chunk doubles what that header costs per value it holds. */
#ifndef SYS__STACK_CHUNK__ALLOC
#define SYS__STACK_CHUNK__ALLOC   SYS__ALLOC__ARRAY_GRANULE
#endif
#define SYS__STACK_CHUNK__SLOTS   (SYS__STACK_CHUNK__ALLOC - 1u)

/* ── FAILURE ──────────────────────────────────────────────────────────────────────────────────────
 * These functions have no failure return. Push deliberately hands back the chunk it was given when it
 * cannot proceed, so that an assignment can never orphan the chain — which also means the caller has
 * no way to tell that apart from success. Reporting it would need a second output, and every case that
 * could be reported is one the engine cannot continue past:
 *
 *     the heap is full           nothing can be allocated, now or later in this launch
 *     a caller broke a contract  no value, a reference that names something other than a stack, or a
 *                                cursor these functions cannot produce
 *     the stack is empty         a well-formed stack with nothing in it, asked for something
 *
 * ⛳ RUNNING OUT OF ROOM IS NOT AMONG THEM, AND THAT IS THE POINT: it is the HEAP's condition,
 * raised where the cause is known, in the heap's own words. A stack has nothing of its own to run out
 * of — and the list and the dictionary meet the same wall, so one condition should not read as three
 * in a log.
 *
 * So they raise the engine fault instead. A container is called from many sites and cannot know which
 * instruction it is serving, so the program counter is left at zero and the operation word carries the
 * identity — two values rather than one, because "out of memory" and "the caller is wrong" want
 * different responses from whoever reads the log.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ⛳ TWO CODES, BECAUSE THE ANSWERS DIFFER. "You handed me something that is not a stack" is a bug in
 * the caller and the next run will do it again; "your stack is empty" is a program meeting a state
 * it can ask about first, with `sys__stack__empty`. One code for both makes a log unable to say which,
 * and this file already learned that once when a single code carried the heap's conditions too. */
#define SYS__STACK__FAULT_CONTRACT   0x53544B43ull   /* "STKC" */
#define SYS__STACK__FAULT_EMPTY      0x53544B45ull   /* "STKE" */

/* The stack's own fields begin at node 1 of its allocation, and there is one of them: which stack
 * chunk is being pushed into and popped from. Named for WHOSE fields it indexes — the chunk's own
 * head args carry a cursor and a link and are spelled `SYS__STACK_CHUNK__HEAD_*`, and nothing here
 * should ever need a second reading to tell the two apart. */
#define SYS__STACK__CURRENT_STACK_CHUNK 0u

/* A chunk is its own type, not a bare node pointer, because a node pointer already means four other
 * things here — a value, a cell inside a chunk, the heap itself — and the compiler cannot tell any of
 * them apart. Handing a value where a chunk belongs would write a cursor into somebody's payload and
 * compile without complaint. The struct costs nothing: it holds the array and nothing else, so a chunk
 * is laid out exactly as the collector walks it, and the empty stack stays a null pointer with one
 * spelling. The assertion is the tie to the allocator that the paragraph above states in prose. */
struct sys__stack_chunk { sys__heap_node cell[SYS__STACK_CHUNK__SLOTS]; };

/* ⛔ THE ALLOCATION MUST DIVIDE THE HEAP CHUNK, AND THE REASON IS A GUARANTEE ANOTHER FILE BORROWS FROM
 * HERE. `heap__header.cuh` states that a fresh chunk with no room for one array is a geometry that cannot
 * happen, and that a chunk carved into arrays loses its header and its tail and nothing else. Both are
 * exact only while a heap chunk is a whole number of these wide, and neither is asserted there — this is
 * the assert that does it, in a file ordered after it and under the macro that names the width.
 * ⇒ ★ A GUARANTEE HELD BY SOMEBODY ELSE IS ONE THEY CAN LOSE WITHOUT TOUCHING THEIR OWN CODE, which is
 * the reason to say here whose it is.
 * ⚠ THIS DOES NOT SAY THE CHUNK IS FULLY USED, and WHERE the loss sits is worth getting right, because
 * an assert that divides invites reading it as a promise that nothing is stranded. Arrays are carved at
 * the bare watermark from SYS__CHUNK__FIRST, so they sit at 2, 18, 34 … packed against each other, with no
 * boundaries to line up and no low bits cleared to find one. What is stranded is the TAIL — the room after
 * the last array that cannot hold another — which, with the two-node header in front of the first, comes
 * to exactly one array's width. */
static_assert(SYS__HEAP__CHUNK_NODES % SYS__STACK_CHUNK__ALLOC == 0,
              "the allocation must divide a chunk, or the carve geometry in heap__header.cuh is not exact");
static_assert(SYS__STACK_CHUNK__SLOTS >= 1u,
              "a stack chunk must have room for at least one value");

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_STACK_CUH */
