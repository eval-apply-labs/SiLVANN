#ifndef SILVANN__PACKAGES_NN_CONTRACTS_MACROS_LANGUAGE_CONTRACT__OBJECTS_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_MACROS_LANGUAGE_CONTRACT__OBJECTS_CUH
/* ══ nn — WHAT THIS PACKAGE PUBLISHES INTO THE LANGUAGE ══════════════════════════════════════
 *
 * Three lists. The TYPES are what this package's kinds are CALLED — a number it chooses and a name
 * everybody else uses; the OBJECTS are the subset of those kinds the heap can hold and count; the
 * FUNCTIONS are the words it publishes, one row per verb. A list with nothing in it is still defined
 * here: the machinery that gathers them pastes this package's name onto the macro, so every gather names
 * this file whether or not it has anything to say. A missing definition is a build error at the gather,
 * which is the right place for it — a package cannot quietly register nothing.
 * ⛳ NO LIST HERE IS EMPTY. `nn` publishes one kind, that same kind as allocatable, and two verbs — and a
 * kind appearing in both lists is the ordinary case rather than a duplication: the KINDS row is its name
 * and number, the OBJECTS row is how it is carved, held and let go of.
 *
 * ── WHAT A KIND ROW WILL CARRY ──────────────────────────────────────────────────────────────────────
 * Seven columns: the package forwarded in, THE PUBLISHED KIND, what the object hands over when its
 * last reference goes, which chunk it is carved from, how it is copied, how it is built from the
 * language, and how many nodes one of them occupies.
 * ⛔⛔ AND THE SECOND COLUMN IS NOT A NUMBER — IT IS THE NAME FROM THE TYPE ROW, in
 * `contracts/macros/language_contract__kinds.cuh`, which is the one thing about this file a reader is most
 * likely to get wrong, because the TYPE row's second column IS a number and the two rows sit in two files. Write `NN__KIND__BUFFER`, never `0`. The KINDS gather
 * has already folded this package onto the top byte to build that constant; the OBJECTS arms switch on
 * it bare, so a `0` here would ask the heap about a kind belonging to package 0 — which is `sys`.
 * ⛳ WHY IT IS SHAPED THAT WAY: the tag is applied EXACTLY ONCE per id, and these two lists reach that
 * one application from opposite ends. A VERBS row carries a bare base and its arms tag it; a kind is
 * tagged at the KINDS gather, so by the time an OBJECTS row names it there is nothing left to apply.
 * ⛔ IT WAS APPLIED TWICE and `sys` being package 0 hid it completely — this package's
 * first OBJECTS row is the one that would have found it, silently, by taking every default.
 * ▶ the note at `PACKAGE_DTYPE` in `packages/manifest__header.cuh`, and the `row_tag_at_gather` rule,
 * which now refuses a bare base in this column.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⭐⭐ THE FIRST ALLOCATABLE KIND THIS PACKAGE HAS, and the one column worth reading twice is the second:
 * it is the NAME from the KINDS row and never a number. ⛔ A bare `0` here would ask the heap about a kind
 * belonging to package 0, which is `sys` — and the tag having been applied twice was invisible until this
 * row existed, because `sys` being package 0 made a second application add nothing.
 *
 * ⛳ WHAT EACH COLUMN SAYS, AND WHY:
 *     RELEASE    files the buffer back in its own class's free list. It is NOT a deallocate — the append
 *                takes a hold, so the allocator's own "is it still dead" question answers no and the room
 *                stays. ▶ `buffer__impl.cuh`.
 *     PROVIDER   OBJECTS. The 64-byte reference lives in the heap like every other counted thing; the
 *                BYTES it names are `nn`'s span and are not the heap's at all.
 *     CLONE      the default, which REFUSES. A copy means a second slot and a byte-for-byte move, and
 *                nothing has asked for one — refusing is the honest first answer and the column is here
 *                so that changing the answer stays a one-row edit.
 *     CONSTRUCT  refuses. A program does not build a buffer, it asks `getnew` for one: the pool decides
 *                which class and which slot, and a constructor taking those as parameters would let a
 *                program name a slot it was never given.
 *     NODES      two — the head, and one node holding an address and a class. */
/* ⭐⭐ AND THE SECOND ROW IS WORTH READING AGAINST THE FIRST, because the two differ in exactly one
 * column and the difference is the whole design:
 *     A BUFFER'S RELEASE FILES IT HOME    it is lent from a pool, and going home is what makes the pool
 *                                         a pool. ▶ `buffer__impl.cuh`.
 *     A KV REFERENCE'S RELEASE IS NOTHING it is a sentence about an address. The bytes belong to the
 *                                         cartridge and outlive every reference to them, so there is
 *                                         nowhere to file it and nothing to give back but its own node.
 * ⇒ ★ THIS IS WHY IT IS A SECOND KIND AND NOT A BUFFER CLASS. A release hook is per KIND, so one kind
 * that sometimes owns its bytes and sometimes does not is a bit the reader has to carry at every site —
 * and getting it wrong is a leak with no event, or a free list quietly filling with addresses nobody
 * lent. ⇒ ⛳ THE THREE OTHER COLUMNS ARE THE SAME AS THE BUFFER'S AND FOR THE SAME REASONS: a clone
 * refuses because a second name for one address is not a copy of anything, and a constructor refuses
 * because a program does not build one — it asks `reference` for it, and the resolver decides. */
#define nn__LANGUAGE_CONTRACT__OBJECTS(X, PKG)                                                         \
    X(PKG, NN__KIND__BUFFER,          nn__buffer__zzpackage_release_internal,                          \
      SYS__COMPUTING_BASE__OBJECTS,    sys__heap_object__clone__default,                               \
      sys__heap_object__create_refusal, NN__BUFFER__NODES)                                             \
    X(PKG, NN__KIND__KV_REF,          sys__heap_object__release__default,                              \
      SYS__COMPUTING_BASE__OBJECTS,    sys__heap_object__clone__default,                               \
      sys__heap_object__create_refusal, NN__KV_REF__NODES)                                                       \
    X(PKG, NN__KIND__WEIGHTS,         nn__weights__zzpackage_release_internal,                         \
      SYS__COMPUTING_BASE__OBJECTS,    sys__heap_object__clone__default,                               \
      sys__heap_object__create_refusal, NN__WEIGHTS__NODES)

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_MACROS_LANGUAGE_CONTRACT__OBJECTS_CUH */
