#ifndef SILVANN__PACKAGES_SYS_CPU_SYSTEM_REGISTER__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_SYSTEM_REGISTER__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../contracts/objects/system_register.cuh" /* its constants and fault words */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * Nothing in it is needed to use the register. It holds what compares this file against the tree it grew
 * out of, so that everything below can be read by somebody who has never seen that tree.
 *
 * WHAT IT CORRESPONDS TO THERE: a getter and a setter over one flat array of `uint64_t`, with a range of
 * rows packages may name and a range above it reserved to the engine. Three differences, each a decision
 * and not an accident:
 *   · A ROW IS A NODE AND NOT A WORD. There, whether a row held a number or an offset was a fact every
 *     caller carried; here the row states it, and a reader of one row needs no second document.
 *   · THERE IS NO RESERVED RANGE. That one had no tenants in it either, and a reserved range inside
 *     somebody else's array is a convention where a second array would be a boundary. Nothing has asked.
 *   · THERE IS NO HOST DOOR. That array is read from outside the device by asking the platform for a
 *     symbol's address. Nothing needs that here: the register is on the host with the rest of the machine.
 *
 * ⛳ AND NOTHING HERE REACHES BACK INTO THAT TREE. The fault channel is this package's own — three words
 * the host owns, handed over by the boot — rather than four rows of that array, and the only
 * include that leaves these files is the container's `manifest__header.cuh`. So this block is a
 * comparison and not a dependency.
 * ⛳ RETIREMENT: **the condition is MET, and that is a note rather than a trigger.** It was *"when the
 * fault channel lands and nothing in the package reaches out"* — the channel is this package's own and
 * the reaches are gone, which is what the paragraph above now says. ⛔ THE BLOCK STILL STANDS: these are
 * a SET and they go as a set, at release, in one act — see the rule in `README.md`. A block deleted
 * because its own condition fell due leaves the other twelve half-referring to it.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ the system register — the one thing a block can name without having been handed it ═══════════════
 *
 * ⚖ ARCHITECT: *"system register is a pure c array, with its getter setter methods and its enum for the
 * indices. the array itself is a pure c object that can fit n ast nodes, so if they are primitives it is
 * fine and if they are object references they instead reference objects that someone else allocated."*
 *
 * ── WHY THERE IS A REGISTER AT ALL, WHICH IS ALSO WHAT DECIDES WHAT MAY GO IN ONE ───────────────────
 * ⚖ ARCHITECT: *"things like locks and semaphores are not shadowable."*
 *
 * A program's names are scoped. Something binds a name, an inner scope may bind it again, and the inner
 * one wins for as long as it lasts — and that is the right home for almost everything precisely because
 * rebinding is harmless: two scopes each holding their own value are two values, and nothing about that
 * is wrong.
 *
 * ⭐ A LOCK IS WHERE IT STOPS BEING HARMLESS, AND THAT IS THE WHOLE TEST. Two scopes each holding their
 * own copy of a lock are two locks, and two locks over one thing protect nothing — the second holder
 * walks straight in past a door the first one is standing behind. The same goes for a semaphore, for a
 * counter the whole grid arrives at, for anything whose meaning is that there is ONE of it. Shadowing
 * such a value is not inelegant, it is incorrect, so a per-scope binding is the WRONG MECHANISM rather
 * than a slower one — and that is a falsifiable test and not a preference.
 * ⇒ ⭐ SO THE QUESTION FOR A NEW ROW IS NOT "does everything need to reach this" BUT "would a second copy
 * of it still BE it". Where the answer is yes it is a binding and belongs in a scope. A row here is what
 * is left over when that question is answered no.
 * ⛔ AND BEING CONVENIENT TO REACH IS NOT THE TEST. A value every block happens to read is shadowable if
 * a scope holding its own would still be correct, and putting it here makes it state that nothing can
 * scope — which is the hardest thing in this file to take back, because by then everybody names it.
 *
 * ⭐ IT IS A PLAIN ARRAY LIKE ANY OTHER, AND ONE STATIC WORD IS WHAT MAKES IT FINDABLE. Everything else
 * here is FOUND: a block reaches what it is working with through its computing base, and it reaches its
 * computing base through a table. The register is where that table is named, so it cannot itself sit at
 * the end of a lookup — what every lookup starts at cannot need one.
 * ⇒ So the storage is an ordinary allocation, counted and shaped like every other, and the ONE thing that
 * is not ordinary is a single word holding where it begins. Reaching a row is that word plus an index:
 * nothing has to have happened first except the boot, and there is no order in which asking is too early
 * once it has. ⛳ THAT WORD IS THE WHOLE OF THE SPECIAL CASE, which is the point — one word rather than a
 * kind of storage nothing else in the package uses.
 *
 * ── WHAT A ROW IS ───────────────────────────────────────────────────────────────────────────────────
 * A whole AST node, so a row carries its type beside its value and says which of the two kinds of thing
 * it holds. A PRIMITIVE is the value: the row is the whole of it. A REFERENCE is a heap offset, and the
 * memory it names belongs to whoever allocated it — the register holds the offset and not the thing.
 * ⭐ THE TYPE IS WHY A ROW IS A NODE AND NOT A WORD. One word can hold either and can say neither, so
 * telling a small number from an offset would be a fact each caller had to carry rather than one the row
 * answers. A row that can be asked what it is can also be asked whether anybody has written it.
 * ⛳ AND THE PRICE IS STATED RATHER THAN HIDDEN: a row costs a whole node, which is 64 bytes here — the
 * width `heap_node__header.cuh` asserts, because nothing else in the tree would say so. The register is
 * exactly as large as the rows declared in it, so what that price buys is never more than what somebody
 * asked for — and it would want re-asking at a thousand rows.
 *
 * ── WHAT IT DOES NOT DO, AND BOTH ARE RULINGS RATHER THAN OMISSIONS ─────────────────────────────────
 * IT ALLOCATES ONCE AND NEVER AGAIN. Its own storage is one array, asked for at boot by the block that
 * stands the world up; after that the register makes nothing. ⛳ THE FIRST ROW'S ARRAY IS PLACED BY THE
 * REGISTER ITSELF, in room handed in for it, because the boot knows how many blocks there will be and
 * this package knows what shape the thing a row names has — the body argues that split. What any OTHER
 * row names is stood up by whoever wanted it there and the register is told the offset, so the thing
 * every lookup starts at never calls the allocator it is on the way to.
 * IT IS NEVER COUNTED. ⚖ ARCHITECT: *"the values inside get their arc +1 but the system register itself
 * and its fields dont need one because they are perennial."* Nothing holds a reference to the register
 * and nothing ever will — it is not made, not handed out and not ended, so there is no count for it to
 * carry and no moment at which one would be read. Its ROWS are the same: a row is a PLACE, and a place
 * does not die.
 * ⛔ AND THAT SAYS NOTHING WHATEVER ABOUT WHAT A ROW HOLDS, which is a distinction worth keeping apart
 * loudly because collapsing it has already cost one wrong design. An object a row names is counted
 * exactly like an object anything else names: putting it in takes a hold, taking it out takes another
 * that the caller gives back, and a write gives up the hold on whatever it displaced. Being perennial is
 * a fact about the REGISTER, not a licence for the register to stop counting.
 *
 * ── WHAT A ROW NOBODY HAS WRITTEN READS AS ──────────────────────────────────────────────────────────
 * A NULL VALUE, because that is what a plain array puts in every element before it hands one out. It is
 * the package's one spelling for nothing, so a reader of a row needs no rule of the register's own to
 * know an empty one when it sees it — and there is no boot pass here to forget, because the filling
 * belongs to the thing that made the array.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── STANDING IT UP ─────────────────────────────────────────────────────────────────────────────────
 * Place the register in room the boot hands in, and remember where it went. Called ONCE, by the boot,
 * and by nobody else: a second call would leave the first array holding rows nothing can reach.
 *
 * ⛳ THE ROOM IS HANDED IN RATHER THAN CARVED, and the reason is the row this register exists for — a
 * block's computing base lives in it, and carving is what asks a block's computing base where to carve.
 * So there is no allocation on this path at all: the room is the tail the boot set aside past the pool,
 * and this runs BEFORE any block is started, which is why there is no base yet to carve against. The
 * body argues it, and the boot says why those two steps cannot be exchanged. The register's only
 * privilege is the word below that says where it went.
 * ⛔ AND IT IS NEVER TAKEN DOWN. The register outlives every block that reads it, so nothing releases the
 * array and nothing may: the hold taken at the making is held by the word, for as long as there is a
 * program. That is what "perennial" means here, said as a fact about lifetime rather than as a mood. */
static __device__ inline bool           sys__system_register__publish(sys__heap_node* room,
                                                                        sys__heap_node* bases_room,
                                                                        uint32_t blocks);
/* Whether that has happened. Asked by every verb before it reaches, and worth having on its own for a
 * caller that has no way of knowing how far the boot got. */
static __device__ inline bool           sys__system_register__published(void);

/* And the other end of it: forget where the register is, so a later `publish` stands a new one up rather
 * than answering that there already is one. The boot's, because the boot is what knows the room is about
 * to stop existing — the argument is beside the definition. */
static __device__ inline void           sys__system_register__zzengine_retire(void);

/* ── THE THREE VERBS ─────────────────────────────────────────────────────────────────────────────────
 * A row is read in two halves because the two questions are different: what is here, and what kind of
 * thing is it. Most callers know the kind — a row's meaning is fixed by its name — and ask only for the
 * value, so the type is a separate question rather than a second return nobody wants.
 *
 * ⚖ AND THESE ARE THE ONLY WAY IN, FROM EITHER SIDE. ARCHITECT: *"the accessors will be via the getters
 * and setters, both here and in lisp."* A program reaching a row names it by the enum above and comes
 * through these, exactly as C does. ⇒ ⛔ WHICH MAKES A ROW'S NUMBER SOMETHING A COMPOSED PROGRAM CAN BE
 * HOLDING: renumbering a row does not break a build, it changes what an already-written program reads.
 * Rows are appended for that reason, and the count is derived so that appending is all it takes.

 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* What a row holds, as a whole node — so a caller is handed the type along with the value and never has
 * to ask a second question to know what it is looking at. A row naming an object comes back HELD, and the
 * caller gives that hold back when it is done. A row that does not exist answers nothing, having raised,
 * and holds nothing: there was no row to take from. */
static __device__ inline sys__heap_node     sys__system_register__get(uint32_t row);

/* What KIND of thing a row holds. `SYS__KIND__VALUE_NULL` means nobody has written it — every element is
 * filled with the null value before the array is handed out, so an empty row answers with the package's
 * one spelling for nothing; anything else is what the writer said it was putting there. Answers
 * `SYS__KIND__INVALID` on a row that does not exist, having raised — a third answer, so a bad row is
 * never mistaken for an empty one. */
static __device__ inline sys__kind sys__system_register__type(uint32_t row);

/* Put something in a row, saying what it is. The register takes a hold of what goes in and gives up its
 * hold on what was there. Answers whether it went in; the only way it does not is a row that does not
 * exist, and that raises. There is no clear: writing `SYS__KIND__VALUE_NULL` is how a
 * row is emptied, because a row that holds nothing is a row holding a value that means nothing, and one
 * spelling of empty is what keeps a reader from having to know two. */
static __device__ inline bool           sys__system_register__set(uint32_t row, sys__kind dtype,
                                                                  uint64_t value);

#endif /* SILVANN__PACKAGES_SYS_CPU_SYSTEM_REGISTER__HEADER_CUH */
