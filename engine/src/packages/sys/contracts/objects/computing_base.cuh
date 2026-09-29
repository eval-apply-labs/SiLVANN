#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_COMPUTING_BASE_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_COMPUTING_BASE_CUH

/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * `RESULT_TRIES` was first priced on the device path: `MEASURED` there, one try cost **820 ns**, timing a
 * collection that finds nothing against a known try count — so 12 million was "about ten seconds". The
 * first figure written was 800 million, reasoned from a guess at the cost of a try; at the real cost that
 * was ELEVEN MINUTES, indistinguishable on a card pinned at 100% from the hang the bound exists to prevent.
 * ⛳ RETIREMENT: when the count is re-derived from a host timing of its own rather than the dispatch's.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../defaults.cuh"

#define SYS__COMPUTING_BASE__OBJECTS   0u   /* the chunk things that LAST are carved from       provider */
#define SYS__COMPUTING_BASE__ARRAYS    1u   /* the chunk things that CHURN are carved from      provider */
/* ⛳ WHY THERE ARE TWO OF THEM IS THE HEAP'S ARGUMENT AND IS MADE THERE, beside the drain-only trade it
 * turns on: a chunk goes home only when everything in it has died, so what LASTS and what CHURNS are
 * carved apart to stop permanence pinning churn. Which of the two a kind uses is not chosen here or by a
 * caller — it is a column in the kind's row. */
/* ⛳ AND HOW MANY OF THEM THERE ARE, WHICH IS NOT THE SAME QUESTION AS HOW MANY SLOTS THERE ARE. The
 * providers are the leading slots and nothing else is one, so a reader wanting "every chunk this block
 * is carving from" counts to here rather than to the end. The heap keeps one cached head per provider
 * and walks this many at a boundary; writing `2` there would be a second place to edit when a third
 * kind of chunk arrives. */
#define SYS__COMPUTING_BASE__PROVIDERS 2u

#define SYS__COMPUTING_BASE__BINDINGS  2u   /* the names the program has bound                  user     */
#define SYS__COMPUTING_BASE__PROGRAM   3u   /* the list being run, then its result              user     */
#define SYS__COMPUTING_BASE__STATUS    4u   /* how the computation is going                 NOT an offset */
/* ⭐⭐ AND WHAT THE RESULT *IS*, WHICH TAKES TWO WORDS BECAUSE A VALUE IS A CELL AND NOT AN OFFSET. A
 * program reduces to a KIND and a word — an integer is a number and names nothing, a list is a reference
 * to one — so a base that carried only the offset would be handing back a room and leaving whoever
 * collected it to guess what was in it. The reducing happens in one block and the collecting in another,
 * and the two have nothing in common except this base, so the kind has to travel with the word.
 * ⛳ IT SHARES THE PROGRAM'S SLOT FOR THE WORD, which is what that slot has always said it was for: the
 * list going in, and what it came to going out. This is the other half of the same answer. */
#define SYS__COMPUTING_BASE__RESULT_KIND 5u /* what the word in PROGRAM is, once there is a result      */

/* A sixth value in the same space, naming no slot: what a kind's row says when it is carved from NO
 * provider. It sits with these because the provider column of the X macro holds SLOT INDICES — `make`
 * takes the answer and reads it out of the base with `zzpackage_get_heap_provider` — so "no slot" belongs
 * beside the slots rather than with the faults, and shares their prefix for the same reason.
 *
 * Two kinds name it, for one reason: neither is carved from a provider. A computing base is placed by hand
 * in `start_block`, before there is anywhere to remember a provider; a chunk provider is taken off the free
 * ring rather than carved out of anything. A row saying "objects" or "arrays" would claim a path neither
 * takes. A kind with no row at all answers this too, so an unregistered kind and a deliberately
 * unprovided one arrive at the same refusal — which is right, since neither can be made.
 *
 * ⛔ IT IS NOT A FAULT WORD, THOUGH IT SITS ONE LINE FROM SOME. `make` RAISES on seeing it, and the word
 * it raises is `SYS__HEAP__FAULT_NO_PROVIDER`, which lives in the heap with the rest of the heap's. This
 * is the value that provokes that; it is not the report. */
#define SYS__COMPUTING_BASE__NO_PROVIDER 0xFFFFFFFFu
/* And the answer for a kind NO package registered, which is a different situation and must not share a
 * value with the one above. `NO_PROVIDER` is a declaration — a kind saying it is placed by hand and never
 * made — and a lookup that could not find a kind at all was returning it too, so a deliberate choice and
 * a hole in the registry came back as one number. Both are above `ARRAYS`, so neither can be mistaken for
 * a real provider slot. */
#define SYS__COMPUTING_BASE__UNREGISTERED 0xFFFFFFFEu

/* How long a collector waits before answering nothing.
 *
 * ⛳ IT IS A BOUND AND NOT A TIMEOUT POLICY. What it buys is that a computation which never finishes
 * cannot park the block that asked for it; `CLAUDE.md`'s own warning is that nothing in this runtime
 * detects a block that never arrives, so the only defence is never to wait forever on purpose.
 *
 * ⛔⛔ AND THE NUMBER IS DERIVED FROM A MEASUREMENT RATHER THAN CHOSEN. One try — two compare-and-swaps and
 * the back-off between them — costs ~1.28 us on the host (`MEASURED`, ▶ `ENG__ABI__DISPATCH_TRIES` in
 * `engine/abi/running.cuh`), so the figure here bounds a wait at about fifteen seconds: generous beside
 * anything this engine has evaluated, and short enough that a bound reached is a bound somebody notices.
 * ⇒ ★ A BOUND NOBODY HAS PRICED IS NOT A BOUND. */
#ifndef SYS__COMPUTING_BASE__RESULT_TRIES
/* ⛳ Priced at the ~1.28 us a try costs the host's dispatch wait, whose count is the same number. */
#define SYS__COMPUTING_BASE__RESULT_TRIES 12000000ull   /* x ~1.28 us = about fifteen seconds */
#endif

/* How long an idle block waits before asking again whether anything has been scheduled for it. Turned in
 * `contracts/defaults.cuh`, guaranteed here — see the note on the lock's pair in `heap_object__header.cuh`
 * for why both files carry a value. */
#ifndef SYS__COMPUTING_BASE__POLL_CYCLES
#define SYS__COMPUTING_BASE__POLL_CYCLES 64u
#endif

/* ⭐⭐ THE STATUS IS THE FIFTH SLOT AND IT IS NOT AN OFFSET BUT A STATE, WHICH IS WHY IT HAS ITS OWN DOOR.
 * (The result kind after it is not an offset either; it is a kind.) The four above are places in the
 * heap — the two providers read and written through
 * `zzpackage_get_heap_provider` and its setter, bindings and program by the verbs that own them; this is a
 * state, and a state that anyone may store into is not a state machine. It moves by
 * compare-and-swap or it does not move.
 *
 * ⛳ FREE IS ZERO ON PURPOSE, and it is the value a base ENDS at rather than the one it starts at. A node
 * fresh from the heap allocation pool holds the last tenant's leftovers in every word, this one included, so `start_block`
 * says who holds it before anything reads it. What zero buys is the other end: `init`, asked for nothing,
 * blanks the word along with the rest and the base is free without a separate store saying so.
 *
 *     FREE       0            nothing is assigned
 *     SCHEDULED  1            a program and bindings are in place and nobody has started
 *     OK         2            it RAN TO THE END — whatever it computed, including an error VALUE
 *     ERROR      3            FATAL: it never started, or something threw that nothing caught
 *     COMPUTING  4 + block    a block has taken it and is running — and WHICH block is the value
 *
 * ⭐⭐ `OK` MEANS IT FINISHED, NOT THAT IT SUCCEEDED, AND THAT IS THE DISTINCTION THE SET TURNS ON.
 * ⚖ ARCHITECT: *"a normal error is an ok, a throwed error is error ... an unhandled error messes with the
 * bindings and leaves stuff allocated, so it needs a signal that we need to wrap up and restart from
 * fresh."* An error is a VALUE, so a program that evaluates to one has evaluated: the scopes unwound, the
 * holds came back, and the caller collects an error the way it collects a number. That needs no state.
 * ⇒ **`ERROR` IS THEREFORE NOT "THE COMPUTATION FAILED". IT IS "DO NOT TRUST THE MACHINE".** Either the
 * base never got as far as running, or a throw unwound nothing — and either way scopes are half-left and
 * objects are held by nobody. A caller that sees it tears the engine down and starts again; there is no
 * partial recovery, because what would be recovered is exactly what is not known.
 *
 * ── AND THE RESULT SLOT TELLS THE FATAL CASES APART, WITHOUT A STATE FOR EACH ────────────────────────
 * ```
 *   OK      an OFFSET      a reference to what it computed. `read_result` transfers the hold.
 *   ERROR   a FAULT CODE   written straight into the slot. NOT an offset, and nothing releases it.
 *   ERROR   ZERO           it could not even say why: the failure was an allocation, so building an
 *                          object to describe it would have failed in the same way.
 * ```
 * ⭐ THE CODE GOES IN THE SLOT RATHER THAN IN AN OBJECT, BECAUSE THE FAILING PATH IS THE ONE THAT CANNOT
 * ALLOCATE. `begin` fails when a thaw could not carve; describing that with a carved error object asks
 * the heap for room at the moment it has just refused. ⚖ ARCHITECT: *"we can rewrite the execution list
 * pointer to be the error message itself ... so we don't even need the heap allocator."* The slot is a
 * word and a fault is a word, so nothing has to be built at all.
 *
 * ⭐⭐ COMPUTING IS NOT A CONSTANT, IT IS THE HOLDER'S INDEX, AND THAT IS THE WHOLE POINT.
 * ⚖ ARCHITECT: *"computing is represented by the atomiccas being the value of the index ... so computing
 * will demand a locked computing base (correctly)."*
 *
 * A lock says HELD. This says held BY WHOM, in the same word, for the same atomic — so "is it being
 * computed" and "who has it" cannot disagree, because they are one fact and not two. And a base that is
 * computing without being held stops being a state anything can represent: taking it IS entering the
 * state, and there is no store that does one without the other.
 *
 * ⛳ WHICH IS WHY THE RESERVED VALUES SIT BELOW THE RANGE. Block ids start at zero and so does FREE, so a
 * bare index would make block 0 indistinguishable from an unassigned base. Four values are spent to keep
 * that from being true, out of a 32-bit word that has no other tenant. */
#define SYS__COMPUTING_BASE__FREE       0u
#define SYS__COMPUTING_BASE__SCHEDULED  1u
#define SYS__COMPUTING_BASE__OK         2u
#define SYS__COMPUTING_BASE__ERROR      3u
#define SYS__COMPUTING_BASE__COMPUTING  4u   /* ...and every value above it: 4 + the block that took it */

/* Raised when a base is asked to hold half a state — a program and no bindings, or the reverse. Its own
 * word, and in this file rather than with the heap's, because each file here owns the faults it raises and
 * this one is about a computing base rather than about memory. */
#define SYS__COMPUTING_BASE__FAULT_HALF 0x43424846ull   /* "CBHF" */

/* Raised when a base that still holds work is asked to become something else — scheduled, running, or
 * finished but not yet cleared. Its own word, because "you asked at the wrong time" and "you asked for
 * half a state" want different answers from a reader of the log. */
#define SYS__COMPUTING_BASE__FAULT_BUSY 0x43424253ull   /* "CBBS" */

/* Raised when a base was begun on a program it could not make its own copy of. A thaw is an allocation,
 * so `begin` can fail, and this is what it says when it does. Its own word rather than the heap's: a
 * caller reading the log wants "there was no room to start this" separated from whichever carve came up
 * short. */
#define SYS__COMPUTING_BASE__FAULT_NO_THAW 0x43424E54ull   /* "CBNT" */

/* Raised when a base was scheduled with a read-only snapshot of an environment and there was no room to
 * build one over it. Its own word beside the thaw's, because they are the two things `begin` makes for
 * the block that will run the work and a reader of the log wants to know which of them came up short. */
#define SYS__COMPUTING_BASE__FAULT_NO_ENV 0x43424E45ull   /* "CBNE" */

/* Raised when a program asked to compute on a block that is not there — an index past the launch, or one
 * whose base was never started. Its own word rather than a type refusal, because a caller naming a cu
 * that does not exist has made a different mistake from one passing the wrong kind of argument, and the
 * repair is different too: one is the program, the other is how many blocks the machine was booted with. */
#define SYS__COMPUTING_BASE__FAULT_NO_BLOCK 0x43424E42ull   /* "CBNB" */

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_COMPUTING_BASE_CUH */
