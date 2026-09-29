#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_BINDINGS_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_BINDINGS_CUH

/* This file includes nothing. Its one borrowed name is the chunk width `WIDTH_MAX` is made of, from
 * `heap.cuh`, which a macro needs only where it is expanded. */

#include <stdint.h>

/* The head, and one node holding what it is made of. */
#define SYS__BINDINGS__NODES 2u

/* ── WHERE THINGS ARE IN EITHER TABLE ────────────────────────────────────────────────────────────── */
#define SYS__BINDINGS__HATCH      0u   /* element 0: the dictionary for names past the width, or nothing */
#define SYS__BINDINGS__FIRST_NAME 1u   /* element 1 holds name 0, so a name's cell is its number plus one */

/* ── WHAT THE PAYLOAD NODE HOLDS ─────────────────────────────────────────────────────────────────────
 * The two tables, and one value. The first and the last are this object's own to give up when it dies;
 * the middle one is a hold on something somebody else made, given up the same way.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
#define SYS__BINDINGS__SCOPES  0u   /* the per-symbol stacks: the width in place, the rest behind the hatch */
#define SYS__BINDINGS__BASE    1u   /* the flat table underneath, shared and read-only, or nothing    */
#define SYS__BINDINGS__UNBOUND 2u   /* the one error every name that nobody has bound answers with    */

#define SYS__BINDINGS__FAULT_EMPTY    0x424E4445ull /* "BNDE" — asked for a bindings with no symbols  */
#define SYS__BINDINGS__FAULT_KIND     0x424E444Eull /* "BNDN" — asked of something that is Not one    */
#define SYS__BINDINGS__FAULT_RANGE    0x424E444Full /* "BNDO" — a numbered name Outside the array     */
#define SYS__BINDINGS__FAULT_NOT_NAME 0x424E444Dull /* "BNDM" — a cell that is not a naMe at all      */
#define SYS__BINDINGS__FAULT_HATCH    0x424E4448ull /* "BNDH" — a base whose Hatch is not read-only   */
#define SYS__BINDINGS__FAULT_UNBOUND  0x424E4455ull /* "BNDU" — nothing is bound to that name         */
#define SYS__BINDINGS__FAULT_MISFIT   0x424E4446ull /* "BNDF" — a base that is not the table's size   */
#define SYS__BINDINGS__FAULT_NO_SCOPE 0x424E4453ull /* "BNDS" — no Scope of this one's own holds it   */
#define SYS__BINDINGS__FAULT_NO_VIEW  0x424E4456ull /* "BNDV" — the View could not be made; log says why */

/* The widest either table can be: one allocation, less its head and the hatch. 508 at the shipping width. */
#define SYS__BINDINGS__WIDTH_MAX  ((uint64_t)SYS__HEAP__CHUNK_NODES - (uint64_t)SYS__CHUNK__FIRST - 2ull)

#include <stdint.h>

/* ── ABOUT `sys__bindings__zzprivate_scope` IN `cpu/bindings__impl.cuh`, WHICH HAS NO DECLARATION HERE ──
 * It answers a name's scope stack, HELD, or nothing when the name is not one this environment has. Every
 * caller that keeps the stack across anything reads through it; the borrowing reads go through
 * `zzprivate_scope_in_place`, where the range test is.
 *
 * ⛳ THE HOLD IS PAID EVEN THOUGH THIS ONE IS PROVABLY SAFE TODAY, and that is deliberate. Nothing takes
 * a stack out of the table — they are made when the environment is and stay for as long as it does — so
 * the reference could be read without counting. What makes it worth paying anyway is that the safety is
 * an invariant of the code around it rather than of the reference itself: the day a verb replaces a
 * name's stack, an uncounted read here becomes a use-after-free with nothing to point at.
 * ⛳ RETIREMENT, AND BOTH ITS HALVES ARE ANSWERED. The accessor that hands out an element without
 * counting is `sys__node_array__zzpackage_at`, which the lookup in `cpu/bindings__impl.cuh` reads through, and
 * `sys__node_array__borrow`; and the reads are `MEASURED` — deleting this retain and the releases that
 * balance it is 7.730 → 5.651 us per name read, −2.079 us, 26.9% of a name read
 * (`measurements/2026-09-14_S98n_what_a_name_read_is_made_of.md` §①). That saving is spent by
 * `sys__bindings__zzengine_get_noretain`, which reads a name through the borrowing form in `cpu/bindings__impl.cuh` and owes
 * nothing for the answer.
 * ⇒ ⭐ WHAT IS LEFT TO DECIDE HERE IS SAFETY AND NOT COST. A borrow is sound only where nothing
 * allocates, releases or pops between the read and the use, so every caller that keeps the stack across
 * anything at all pays the hold. */
/* ── WHERE A NAME'S SCOPE LIVES, REMEMBERED FOR THE LENGTH OF A COMPUTATION ──────────────────────────
 * ⚖ ARCHITECT: *"isn't it only needed to be populated in the beginning of a computation? … it should be
 * inside the inline init method that deals with the other 2."*
 *
 * Three questions stand between a name and the cell it names — where the scopes are, whether that is an
 * array, and how long it is — and answering them one lookup at a time means three reads of the same head
 * before the one that was asked for. None of the three can change while the computation runs:
 * `args[SCOPES]` is written at
 * creation and at teardown and nowhere else in the tree, so what it names outlives every lookup through
 * it. ⇒ the answers are taken once, where the computation is told which bindings it has, and the lookup
 * is then a bounds test and a read.
 *
 * ⛔ WHAT MAKES IT SAFE IS THE BRACKET AND NOT THE KEY. Block-local storage belongs to a runner thread and
 * outlives the computation it was filled for, so a remembered offset that happens to match is the danger
 * rather than a comfort. It
 * cannot arise: a computation is opened by `begin`, which fills this from the bindings that block will
 * read names out of, and closed by `read_result`, which empties it — as does `init`, so a base left for
 * somebody else carries nothing of this one. Nothing reads a name outside that bracket.
 * ⛔ THE ARMING IS IN `begin` AND NOT IN `init`, AND THE REACH BEING BLOCK-LOCAL IS WHY: `init` runs on
 * whoever SCHEDULED, which is not who will read the names, so filling it there fills the wrong block's —
 * and what is scheduled may not be an environment yet, since a snapshot is a plain array until `begin`
 * builds one over it and asking it for scopes raises. The reason in full is at `sys__computing_base__init`.
 * ⛳ THE LOOKUP FILLS IT ITSELF ON A MISS, which is what keeps the harness — and anything else that
 * asks a question without opening a computation — answering correctly rather than quickly. */
typedef struct {
    uint64_t bindings;   /* whose scopes these are, or nothing              */
    uint64_t scopes;     /* the array, which cannot move while it is        */
    uint64_t width;      /* how many names it holds in place, past its hatch */
} sys__bindings_reach;

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_BINDINGS_CUH */
