#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_LIST_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_LIST_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "heap.cuh"

/* ── THE LAYOUT, WHICH IS A FACT ABOUT THE SHAPE AND SO BELONGS HERE ─────────────────────────────────
 * A chunk is one allocation on a granule boundary whose first node is its head, exactly as a stack's is —
 * so the thing that would have been a separate spine node is the head this chunk already had. Its two
 * words are how many values are in it and which chunk comes after it.
 * ⛔ THE LINK POINTS FORWARD WHERE A STACK'S POINTS BACK, AND IT IS THE SAME SLOT. A stack chunk handed
 * to a list primitive would therefore compile and be wrong, which is why every verb asks what kind
 * it is holding before it reaches.
 * ⛳ `args[0]` OF ANY HEAD IS THE LOCK and is not ours to use, which is why these start at one.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ⚖ ITS OWN DIAL, SEPARATE FROM THE STACK'S — see that file for the argument. A list here is mostly a
 * FORM: three or four values, made and thrown away by the million. What it wants from a chunk is to be
 * small; what a binding stack wants is to be deep. One number could not be both. */
#ifndef SYS__LIST_CHUNK__ALLOC
#define SYS__LIST_CHUNK__ALLOC      SYS__ALLOC__ARRAY_GRANULE
#endif
#define SYS__LIST_CHUNK__SLOTS      (SYS__LIST_CHUNK__ALLOC - 1u)
#define SYS__LIST_CHUNK__HEAD_USED  1u   /* args[1] of the head: how many values are in this chunk    */
#define SYS__LIST_CHUNK__HEAD_NEXT  2u   /* args[2] of the head: the chunk after this one, 0 for none */

/* The store's payload node: the two chains it holds, and nothing else. */
#define SYS__LIST__NODES         2u
#define SYS__LIST__STORE_VALUES  0u   /* the first chunk of values, 0 while the list is empty        */
#define SYS__LIST__STORE_ROLL    1u   /* the first chunk of the roll of who is looking, 0 for none   */
/* ⭐⭐⭐ TWO WORDS THAT END A TRAVERSAL EACH — ⚖ RULED after the walk census in `NN-15`.
 * Derived, `TOTAL` would be a sum of `used` over every chunk on every `length` and every `append` —
 * a chunk holds 15 cells, so a 1,200-cell program body is an 80-hop dependent pointer chase to answer
 * "how long is this". `TAIL`, derived, would be a walk of the whole chain on every append.
 * ⇒ ★★ A DERIVED FIGURE IS FREE ONLY WHILE NOBODY ASKS FOR IT. These two are asked for on the
 * hottest path in the machine.
 * ⛔⛔ AND THEY ARE INVARIANTS, WHICH IS THE PRICE: every site that changes a VALUES chunk's `used`
 * must move `TOTAL`, and every site that links or unlinks the LAST values chunk must move `TAIL`.
 * The census that makes that checkable — and it is short, which is why this is safe:
 * ```
 *   chunk create   used = 0        no cells       -> TOTAL unchanged
 *   enrol          the ROLL chain, never values   -> TOTAL unchanged  ⛔ do not touch it here
 *   split          keep / used-keep               -> TOTAL unchanged, TAIL moves if it split the last
 *   place_at       used + 1                       -> TOTAL + 1
 *   discard        used - take, may unlink        -> TOTAL - take, TAIL back to `before` if it was last
 *   extract        used - 1,     may unlink        -> TOTAL - 1,    TAIL the same way
 * ```
 * ⇒ ★ SIX SITES, AND THE ROLL IS THE ONE THAT LOOKS LIKE A SEVENTH AND IS NOT.
 * ⇒ ★★ AN INVARIANT IS ONLY AS GOOD AS THE CENSUS OF WHO BREAKS IT, AND A CENSUS WRITTEN FROM READING
 * IS A CLAIM LIKE ANY OTHER. `grep -n "HEAD_USED\] ="` is the command that enumerates them; it answers
 * eight, of which two are the roll's. */
#define SYS__LIST__STORE_TOTAL   2u   /* how many values the chain holds — maintained, never summed  */
#define SYS__LIST__STORE_TAIL    3u   /* the last values chunk, 0 while the list is empty            */

/* A sublist's payload node. The seat is where its own entry sits on the roll, remembered rather than
 * searched for, because the one thing it must do when it dies is take that entry off. */
#define SYS__SUBLIST__NODES  2u
#define SYS__SUBLIST__STORE  0u   /* the store it views. HELD — this is what keeps the list alive     */
#define SYS__SUBLIST__START  1u   /* how far along it begins. Written by the store's own mutations    */
#define SYS__SUBLIST__SEAT   2u   /* its entry on the roll, as a flat seat number                     */

/* Asked of something that is not a sublist, asked for a position the sublist does not reach, and asked
 * for a picture of a list that reaches the same list twice. Each has a word of its own, because a log
 * carries only the word.
 * ⛔ AND THE FIRST WORD IS THREE SITUATIONS AND NOT ONE, WHICH IS WHAT TRIAGING AN "LSTN" IN A LOG HAS
 * TO KNOW. It is raised where a reference handed to a verb is not a sublist, where
 * `sys__list__create_executable` is handed no cells or a count of zero, and where
 * `sys__sublist__deep_copy_to_node_array` meets a quoted list naming an array — every refusal of the shape
 * "that is not what I was asked for". The other two words are one situation each. */
#define SYS__LIST__FAULT_KIND   0x4C53544Eull   /* "LSTN" — Not one                                  */
#define SYS__LIST__FAULT_RANGE  0x4C53544Full   /* "LSTO" — a position Outside this view             */
#define SYS__LIST__FAULT_SHARED 0x4C535453ull   /* "LSTS" — a list reached twice, so no picture of it */

#include <stdint.h>
#include "heap_node.cuh"

struct sys__list_chunk { sys__heap_node cell[SYS__LIST_CHUNK__SLOTS]; };

/* ── A WALK — WHERE A SEQUENTIAL READER IS, SO IT DOES NOT START OVER ────────────────────────────────
 * ⚖ RULED after the walk census in `NN-15`. A list is a chain of 15-cell chunks and finding a
 * position means following that chain from its head. `nth` does exactly that, correctly, and a LOOP over
 * `nth` therefore does it `n` times to read `n` adjacent values:
 * ```
 *   for k in 0..n:  nth(list, k)     ->   O(n^2 / SLOTS) links followed, to read n cells
 *   a walk                           ->   O(n / SLOTS)   links followed, once, in order
 * ```
 * ⭐⭐ SO THE STRUCTURE IS NOT THE PROBLEM — THE ACCESS PATTERN IS. A walk leaves the chunk chain as it
 * is; what it adds is that a reader going forward REMEMBERS WHERE IT GOT TO. `MEASURED` at three of
 * the four callers with this shape: `deep_copy_to_node_array` (which is `freeze`), and the evaluator's
 * own `first_form` and `resolve_bindings`. The fourth, `deep_clone`, walks for the same reason.
 *
 * ── ⛔⛔ WHAT A WALK IS NOT ALLOWED TO OUTLIVE, WHICH IS THE ONLY WAY TO GET THIS WRONG ──────────────
 * A walk holds a CHUNK, and a chunk is exactly the internal the list hands to nobody (▶ `list__header.cuh`:
 * *"No chunk, no slot and no address ever leaves this file"*). The rule holds — a walk is not a
 * value, cannot be put in a variable, and never reaches a program — but the ENGINE may hold one, and a
 * structural change under it would leave it naming a chunk that has been split, emptied or unlinked.
 *     SAFE UNDER A WALK      reading · `walk_cell` writes (a replacement moves nothing) · `replace`
 *     ⛔ ENDS A WALK         `insert` · `append` · `place` · `discard` · `extract` · `split`
 * ⛳ AND IT IS GUARDED RATHER THAN MERELY DOCUMENTED, WITHOUT A NEW INVARIANT TO MAINTAIN: the walk
 * remembers `STORE_TOTAL` as it opens and refuses once that moves. Every verb on the forbidden line
 * changes the total (`split` runs only inside an insert), so the guard is the one already enforced by
 * the six sites in the census above and by 1,596 host checks — a SECOND census here would be one more
 * list to keep true.
 * ⛔ IT IS NOT COMPLETE AND SAYING SO IS THE POINT: a discard and an insert BETWEEN two steps of one
 * walk restore the total and the guard would miss it. Nothing in the tree does that, and a guard that
 * catches the reachable mistake is worth more than a field nobody remembers to bump.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* Where a walk is. It is a VALUE the caller keeps on its own stack — it allocates nothing, takes no hold,
 * and goes away by being forgotten, so nothing has to be closed and nothing leaks when a loop breaks. */
struct sys__list_walk {
    uint64_t store;   /* the store being walked                                                       */
    uint64_t chunk;   /* the chunk `at` falls in, 0 once the walk has run off the end                 */
    uint64_t base;    /* the position of that chunk's first value                                     */
    uint64_t used;    /* how many values that chunk holds — read when the chunk is entered, not per step */
    uint64_t at;      /* how far along the STORE the walk stands, so `at - base` indexes the chunk    */
    uint64_t total;   /* what the store held when this opened — the staleness guard described above   */
};

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_LIST_CUH */
