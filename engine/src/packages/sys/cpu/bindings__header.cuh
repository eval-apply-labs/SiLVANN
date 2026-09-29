#ifndef SILVANN__PACKAGES_SYS_CPU_BINDINGS__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_BINDINGS__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../contracts/objects/bindings.cuh" /* its constants and fault words */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * Nothing in it is needed to use the bindings. It holds what compares this file against the tree it grew
 * out of, so everything below can be read by somebody who has never seen that tree.
 *
 * WHAT IT CORRESPONDS TO THERE: an array of keys, each key owning a paged list of the values bound to it,
 * plus a separate per-CU arrangement of three tables — one writable, one a read-only clone, one shared —
 * selected by a predicate re-derived at three call sites. Four differences, each a decision:
 *   · A SYMBOL'S SHADOW STACK IS A STACK. That tree reaches a symbol's current value by walking a paged
 *     list, which is a data-dependent walk on the hottest path in the interpreter. Shadowing is strictly
 *     last-in-first-out, so the walk buys nothing and the edge access is the whole access pattern.
 *   · THE SHARED TABLE IS NOT A BINDINGS. It is a plain array, passed in — so several readers name one
 *     table rather than each being handed a copy, and nothing is cloned per thread.
 *   · THE TIER IS NOT A PREDICATE. Which tier a read lands in is decided by what is in the scope table,
 *     once, at the place that reads it. That tree decided it three times from surrounding state, and the
 *     three disagreed when a binding form straddled a region boundary.
 *   · UNBOUND IS AN OBJECT, NOT A SENTINEL. There, a total miss publishes the same null a bound-to-
 *     nothing symbol has, so the environment cannot report its own failures. Here they are different
 *     values and the answer carries its own reason.
 *
 * ⛳ THE ONE THING TO KNOW BEFORE EDITING: nothing here fills a table with unbound markers on the way
 * in. A missing base is answered from the object's own error, and a snapshot writes the marker only
 * where the source had no value. If you find yourself adding a fill to `create`, that is the shape that
 * has broken — two bindings over one table would each overwrite what the other bound.
 * ⛳ RETIREMENT: this block goes when nothing in `src` reaches into the old environment.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ the bindings — what a name means right now, and what it meant before ═════════════════════════════
 *
 * A program binds names. Something gives a name a value, an inner scope may give the same name another,
 * and the inner one wins for as long as it lasts — then the outer value is back, unchanged. That is the
 * whole of what this holds: for every name, the value it has now, and the values waiting underneath it.
 *
 * ⭐ AND THE OBJECT ITSELF IS A HEADER AND NOTHING MORE. ⚖ ARCHITECT: *"the bindings object should be a
 * header that points at two [tables], the writable one and the view only one."* Everything it is made of
 * is either a primitive or allocated somewhere else and named by reference, which is what lets one of 
 * the two be somebody else's and be shared, and what makes taking a picture of it cheap.
 *
 * ── WHY A NAME IS AN INDEX AND NOT A SEARCH ─────────────────────────────────────────────────────────
 * A name a program was composed with arrives already numbered, so reaching its bindings is an addition. Nothing is compared,
 * nothing is hashed and nothing is walked, which matters here more than anywhere else in the package:
 * resolving a name is what an evaluator does between every two things it does, so a cost paid here is
 * paid more often than any other cost in the language.
 * ⛔ SO A BINDING IS READ IN ONE INDEXED ACCESS WITH NO WALK: the read path holds no loop at all, and the
 *   claim gate's `bindings_no_walk` fails on the first one added.
 * ⭐ AND IT IS WHY THE NAMES A PROGRAM USES MOST ARE AN ARRAY AND NOT A TREE. A tree pays for growth
 * with a chase — one dependent load after another, each waiting on the last.
 * ⚖⚖ AND THE REST ARE A TREE, BEHIND ONE POINTER. ⚖ ARCHITECT: *"a single bindings array to go at the
 * width requested, and the second level done by a dictionary to catch all the values above the
 * bindings_size"* — and where that dictionary lives: *"make position 1 non dispatched and hold the
 * bindings object there."* Numbers are dealt by the machine and never given back, so a program cannot
 * know how high they go when its environment is made, and one allocation holds 508 names at most.
 *     element 0         the DICTIONARY for every name spelled by its string, or nothing until the first
 *     elements 1 …      one cell per numbered name, at the name's number plus one
 * The same layout serves both tables below, and only what the dictionary holds differs.
 *
 * ── A NAME IS A NUMBER OR A STRING, AND ITS CELL SAYS WHICH ─────────────────────────────────────────
 * ⚖ ARCHITECT: *"the overflow bindings dictionary is keyed on the heap node string object
 * reference, not by number. this way we have a immediate way to discriminate what looks into the array
 * and what searches the dictionary."* So every verb below is handed the name's CELL, and its kind decides:
 *     BINDING_REFERENCE · QUOTED_NAME                 a number, read from the array — and a number the
 *                                                     array has no cell for is refused, never looked up
 *     STRING_BINDING_REFERENCE · QUOTED_STRING_NAME   an interned string, read from the dictionary,
 *                                                     keyed by the string's offset
 * ⚖ AND WHO GETS WHICH IS THE COMPOSER'S CALL: *"the dictionary is used for the values known at
 * compile time/binding time"* was said of the array — a name gets its number while the number fits the
 * environment it is composed against, and its string otherwise.
 * ⛳ THE STRING IS THE INTERNED ONE, WHICH IS WHAT MAKES ITS OFFSET A NAME: one text has one interned
 * string, so two cells spelling one name hold one offset. A string cell takes no hold, like a number,
 * because an interned string is held by the symbol table for as long as the machine is up — and the host
 * door refuses a string name whose string is not that one (`sys__symbol__interned`).
 *
 * ── A SYMBOL'S NUMBER IS NEVER GIVEN BACK ───────────────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"we dont make the bindings table release, preventing the whole issue."*
 * Names are registered when a program is composed and nothing returns one, so a slot in these tables
 * belongs to one name for as long as the machine is up.
 * ⛔ RECYCLING A SLOT IS NOT SAFE WITH WHAT IS HERE, AND AN EMPTY SCOPE IS NOT THE SIGNAL. A name cell
 * holds the symbol's NUMBER, or its string without a hold, so nothing counts who still names a symbol — and a scope
 * with nothing in it says nothing about that. A procedure's parameter is unbound between every two calls
 * while its body still names it; a program not yet run names symbols nobody has bound; another block's
 * scopes are not these. Handing the number to a new name while any of those survives makes the old
 * program read the new name's value, and nothing raises and no count is off. `REASONED`, from
 * `sys__heap_node__carries_reference`, which lists no name kind.
 * ⛳ WHAT RECYCLING WOULD TAKE, EITHER OF TWO. COUNT THE NAME CELLS, so a symbol is free when its count
 * falls to the registering table's own hold — an atomic on every name read and every copied body, the
 * evaluator's hottest path, to be measured before anyone chooses it. OR SWEEP WHILE NO BLOCK IS
 * COMPUTING, marking every number a live cell or scope still holds and freeing the rest — nothing on the
 * hot path and one walk of the live heap when asked, `ASSUMED` that a chunk can be walked object by
 * object, which nobody has checked.
 * ⚖ NEITHER IS BUILT, BECAUSE NOTHING NEEDS IT: *"we did not do it cause no one is doing it currently."*
 * ⛳ RETIREMENT CONDITION: the first program that makes symbols while it runs.
 *
 * ── THE TWO TABLES, AND ONLY ONE OF THEM IS THIS OBJECT'S ───────────────────────────────────────────
 *     THE SCOPES  one stack per symbol, and this object's own. Entering a scope pushes, leaving it pops,
 *                 and the top is the value the name has now. It is a stack because shadowing is strictly
 *                 last in, first out — an inner scope always ends before the one holding it. Its
 *                 dictionary is WRITABLE and holds a stack per string-named name, each made on that
 *                 name's first binding.
 *     THE BASE    one value per symbol, flat, and NOT this object's. It is a plain array handed in at
 *                 the making, read and never written, and it is what a name means to somebody who has
 *                 entered no scope of their own. Its dictionary is SEALED — ⚖ *"a readonly dictionary
 *                 too so we dont have to lock when reading, it is necessary for the async compute"* —
 *                 and holds a value per string-named name.
 * A read looks in the scopes and falls through to the base when nothing is bound there.
 *
 * ⭐⭐ THE BASE IS PASSED IN RATHER THAN MADE HERE, AND THAT IS THE WHOLE OF HOW SHARING WORKS.
 * ⚖ ARCHITECT: *"multithreaded execution can refer to the same base binding and dont need to clone
 * multiple times."* Several bindings, each with its own scopes, name ONE table and each takes a hold of
 * it. Nothing is copied per reader, and nothing coordinates them, because the only thing any of them
 * does to that table is read it.
 * ⇒ ⛔ WHICH IS ALSO WHY NOTHING HERE WRITES A BASE. Two readers sharing one table, each initialising
 * it, would each erase what the other put there. The table arrives finished.
 *
 * ── A BINDINGS MAY HAVE NO BASE AT ALL ──────────────────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"receiving a null in the base bindings i think should also be possible ... we can just
 * throw an error when reaching for the missing list, which means the binding was not defined in the main
 * one."*
 * So a bindings with nothing underneath it is an ordinary thing to have, and a name it never bound is
 * unbound — the same answer a name absent from a base gets, arrived at without a table to look in.
 * Nothing is allocated to represent an absence.
 *
 * ── WHAT AN UNBOUND NAME ANSWERS, AND WHY IT IS NOT NOTHING ─────────────────────────────────────────
 * ⚖ ARCHITECT: *"can you bind a let to null? if so it should return null ... i would follow sicp's
 * convention for null."*
 * Binding a name to nothing is legal, so nothing is a value a name can have — which means it cannot also
 * be the answer to "this name has no value". Those are two different facts and a reader has to be able
 * to tell them apart: one is a program that said what it meant, the other is a program reaching for
 * something nobody gave it.
 * ⇒ ⭐ SO AN UNBOUND NAME ANSWERS WITH AN ERROR, WHICH IS A VALUE LIKE ANY OTHER. It comes back the way
 * every other answer comes back, it can be held, passed and asked what it says, and it carries the
 * reason with it — so a caller that did not expect one finds out by looking at what it got rather than
 * by going to look somewhere else.
 * ⭐ AND EACH BINDINGS MAKES ONE, ONCE, WHEN IT IS MADE. It is what a fall-through answers with when
 * there is nothing underneath, and what a picture writes where the source had no value. One object
 * serves every unbound name it will ever have, because an error saying nothing is bound here has no
 * per-name content to carry — and it is held rather than made on demand so that a READ NEVER ALLOCATES,
 * which is the property that decides this and not the saving.
 *
 * ── OWNERSHIP, WHICH IS THE PACKAGE'S RULE AND NOT THIS FILE'S ──────────────────────────────────────
 *     PUTTING A VALUE IN takes a hold, and whatever it displaced gives one up.
 *     TAKING ONE OUT takes a hold too, and the caller owes it back — a look is not free.
 *     THE BINDINGS DYING hands everything it held to the chain that is draining it.
 * ⭐ A READ HANDS BACK A HOLD ON THE VALUE AND NEVER ON THE TABLE IT CAME OUT OF. That is what keeps a
 * shared base from being held alive by one word somebody took out of it — a lifetime set by what is used
 * rather than by what happened to be next to it.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── WHAT A NAME CELL IS ─────────────────────────────────────────────────────────────────────────────
 * Two questions a caller holding a cell asks before it hands the cell over: is it a name at all, and is it
 * one the scan resolves. The verbs below ask the first themselves, and refuse.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* A name read from the array. */
static __device__ inline bool sys__bindings__numbered(sys__kind dtype) {
    return dtype == SYS__KIND__BINDING_REFERENCE || dtype == SYS__KIND__QUOTED_NAME;
}

/* A name read from the dictionary. */
static __device__ inline bool sys__bindings__spelled(sys__kind dtype) {
    return dtype == SYS__KIND__STRING_BINDING_REFERENCE || dtype == SYS__KIND__QUOTED_STRING_NAME;
}

/* A name MENTIONED — the two kinds the evaluator replaces with what they mean. */
static __device__ inline bool sys__bindings__mentioned(sys__kind dtype) {
    return dtype == SYS__KIND__BINDING_REFERENCE || dtype == SYS__KIND__STRING_BINDING_REFERENCE;
}

/* A name QUOTED — the two kinds a verb that binds is handed. */
static __device__ inline bool sys__bindings__quoted(sys__kind dtype) {
    return dtype == SYS__KIND__QUOTED_NAME || dtype == SYS__KIND__QUOTED_STRING_NAME;
}

/* ── MAKING ONE, AND MAKING SOMETHING TO SHARE ───────────────────────────────────────────────────────
 * The two below are not a pair of constructors for one thing. One makes an environment; the other takes
 * a picture of what an environment currently means, so other readers can have it without having a copy
 * of anything — and it is cheap because this object already holds both halves of the answer.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* An environment holding names 0 to `symbols` − 1 in place — at most `SYS__BINDINGS__WIDTH_MAX`, and a
 * wider request gets that many — over the shared table at `base`, or over nothing, which is what `base`
 * being zero says. Every name starts unbound; the numbered ones are ready to be bound without anything
 * further being made, and a string-named one gets its stack when first bound.
 * ⛳ A base is checked for size and not merely for kind: one without a cell for every name in place would
 * answer some of them out of memory that is not its own, so a mismatch is refused rather than trusted —
 * and a base whose dictionary is not sealed is refused too, because it is shared and nobody may write it.
 * Zero names is refused: a table that can answer no question is a caller who has counted wrong.
 * ⛔ WHAT THE CHECK CANNOT DO IS TELL A LIVE TABLE FROM A DEAD ONE, so THE CALLER MUST HOLD THE BASE IT
 * NAMES. A reference is a place, and a place that has been handed back still reads as whatever kind was
 * last there — `MEASURED`: a table whose count has reached zero still answers yes when asked if it is
 * one. Passing a base nobody holds would take a hold on room the pool has already given away, and
 * nothing here or anywhere else would say so. Holding what you pass is the whole of the requirement, and
 * it is not a hard one: a caller with a base got it from somewhere. */
static __device__ inline uint64_t sys__bindings__create(uint64_t symbols, uint64_t base);

/* What `bindings` means right now, as a plain array anybody may read and nobody writes: the same width,
 * one cell per numbered name holding the value that name has or the unbound error where it has none, and
 * at element 0 a SEALED dictionary of what every string-named name means — or nothing, when none does.
 * What comes back is the caller's to hold and to release, and it is what other environments are built
 * over.
 * ⭐ IT IS A PICTURE AND NOT A WINDOW. Nothing the source binds afterwards appears in it, which is what
 * makes it safe to read from anywhere with nothing coordinating the readers. */
static __device__ inline uint64_t sys__bindings__create_viewonly_array(uint64_t bindings);

/* Whether this names one at all. Asked by every verb below before it reaches, and worth having on its
 * own for a caller that was handed a reference and has no reason to trust it. */
static __device__ inline bool sys__bindings__is(uint64_t bindings);

/* How many numbered names it holds — its width — read off the table rather than remembered anywhere. A
 * number at or past it is refused; a name that does not fit is spelled, and found through the dictionary. */
static __device__ inline uint64_t sys__bindings__symbols(uint64_t bindings);

/* ── READING A NAME ──────────────────────────────────────────────────────────────────────────────────
 * The two questions a reader has are what a name means and whether it means anything, and they are
 * separate because the answer to the first is a legal value in both cases. A name bound to nothing and a
 * name nobody bound both come back as something; only the second is an error.
 * ⛳ EVERY VERB FROM HERE ON IS HANDED THE NAME'S CELL, and a cell that is not a name, or a number the
 * array has no cell for, is refused the way a reference that is not a bindings is: raised, and answered
 * with nothing.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* What a name means now: the top of its scope stack, or its value in the base when it has no scope of
 * its own, or the unbound error when there is neither — whether the name is in place or past the width.
 * A value naming an object comes back HELD and the caller gives that hold back when it is done. */
static __device__ inline sys__heap_node sys__bindings__get(uint64_t bindings, const sys__heap_node* name);

/* The same read, BORROWED — no lock, no hold on the scope stack and none on the answer. ⛔ The caller
 * must take its own hold before anything can pop that stack; its mistake is a use after free and nothing
 * will report one. Reachable from this package and from the engine, and from nowhere else.
 * ⛔ ONE ANSWER IS OUTSIDE THAT RULE: the unbound error comes back HELD, because it is one shared object
 * this environment hands to everybody and there is no other way to give it out. So a caller that can
 * meet an unbound name owes a release for that answer and for no other — `REASONED` from the arm that
 * retains it, and the engine's name resolution, which is the only caller, writes the answer into the
 * cell it stands on with `sys__heap_object__set`, which takes the cell's OWN hold and leaves that one
 * unbalanced. */
static __device__ inline sys__heap_node sys__bindings__zzengine_get_noretain(uint64_t bindings,
                                                                            const sys__heap_node* name);

/* What KIND of thing a name means, leaving the caller owing nothing — for one that wants to know what is
 * there rather than to have it. A reference that is not a bindings at all answers `SYS__KIND__INVALID`.
 * ⛳ THE SAVING IS THE CALLER'S RELEASE AND NOT A CHEAPER LOOKUP, which is worth saying because the same
 * verb on a plain table IS the cheaper lookup. What a name means may be the top of a stack, so finding it
 * is the whole of a read; this gives the hold straight back instead of handing it over. */
static __device__ inline sys__kind sys__bindings__type(uint64_t bindings, const sys__heap_node* name);

/* Whether anybody has bound this name — the question `get`'s answer cannot be asked, because an unbound
 * name comes back as an object exactly as a bound one does. It takes nothing and holds nothing.
 * ⛳ It asks what the answer SAYS rather than which object it is, so it is right about a base built by
 * somebody else, whose unbound error is not this object's. */
static __device__ inline bool sys__bindings__bound(uint64_t bindings, const sys__heap_node* name);

/* ── CHANGING WHAT A NAME MEANS ──────────────────────────────────────────────────────────────────────
 * Three verbs, and they are three different events rather than three spellings of one. Entering a scope
 * and leaving it come in pairs and never lose what was underneath; assigning replaces what the innermost
 * scope is holding and leaves the pairing alone.
 * ⛔ NONE OF THEM REACHES THE BASE, WHICH IS WHAT MAKES IT SHARABLE. A write lands in this object's own
 * scopes or it does not land, so no reader of a shared table can be surprised by another one.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* Enter a scope for this name: the value becomes what the name means, and what it meant is kept
 * underneath to come back when the scope ends.
 * ⛔ IT ANSWERS THAT THE NAME WAS REACHED AND THE VALUE HANDED OVER, WHICH IS NOT THAT IT IS ON THE
 * STACK. A push that could not find room raises and answers nothing at all, so the two cannot be told
 * apart here. Said rather than left to be discovered, because a `true` that means less than it looks is
 * worse than no answer. */
static __device__ inline bool sys__bindings__add(uint64_t bindings, const sys__heap_node* name,
                                                 const sys__heap_node* value);

/* Leave the innermost scope for this name: what it means now is given up and what is underneath is what
 * it means again. Answers whether there was a scope to leave — a name with none is a caller whose
 * pairing has come apart, so it raises rather than doing nothing quietly. */
static __device__ inline bool sys__bindings__remove(uint64_t bindings, const sys__heap_node* name);

/* Assign: the innermost scope holding this name is made to hold `value` instead, and what it held gives
 * up its hold. No scope is entered and none is left, so what the name meant before this scope is
 * untouched.
 * ⛔ A NAME WITH NO SCOPE OF ITS OWN CANNOT BE ASSIGNED TO, AND THAT IS THE BASE BEING SHARED RATHER THAN
 * a limitation of this verb. What is underneath belongs to everybody reading it, so there is nothing here
 * to write into — the name may well be bound down there and this still refuses. It is a different mistake
 * from a name nobody bound anywhere, and ⛔ THE WORD DOES NOT SAY WHICH: both raise
 * `SYS__BINDINGS__FAULT_NO_SCOPE`, because in both this object's own scopes are what is holding nothing.
 * A caller that needs the two apart asks `sys__bindings__bound`, which reads through to the base. */
static __device__ inline bool sys__bindings__set(uint64_t bindings, const sys__heap_node* name,
                                                 const sys__heap_node* value);

/* ── THE TWO ROW METHODS ─────────────────────────────────────────────────────────────────────────────
 * What the kind's row names, and neither is called by hand: one runs when the last hold goes, the other
 * when a program asks for one. They are declared here because the row that names them is read in a file
 * that has never heard of this one.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* What a bindings was holding when it died: its scopes, its hold on the base, and its unbound error,
 * moved onto the chain the release loop drains. It lets go of nothing itself — a table's own contents
 * are released when that table is, which is what keeps a structure this deep unwinding at one depth. */
/* The environment a binding form names: `env`, or for 0 the one the computation running it keeps in its base. */
static __device__ inline uint64_t sys__bindings__zzpackage_named(const sys__heap_node* base, uint64_t env);

static __device__ __noinline__ void sys__bindings__zzpackage_release_internal(sys__heap_node* head,
                                                                        uint64_t* releaser_stack);

/* Building one from a program: how many names, and the table to sit over. Two arguments, so it is
 * expressible with what a single node carries. */
static __device__ inline sys__heap_node sys__bindings__zzpackage_construct(sys__heap_node* base,
                                                                       const sys__heap_node* parameters);

/* ── A SCOPE, AS ONE WORD ────────────────────────────────────────────────────────────────────────────
 * `(let env 'name value … 'body)` — bind every name to its value, run the body, and unbind again.
 * ⚖ ARCHITECT: *"i would have preferred to maintain a let implicit so that eval can expand it by doing the
 * bindings inline without allocating new chunk lists."*
 *
 * ⭐⭐ SO IT BINDS WHEN IT RUNS AND REWRITES ITSELF INTO WHAT IS LEFT TO DO. The values are already values
 * by then — the scan reduced them before this form could be applied — so there is nothing to write down
 * for the scan to come back to and the binding is simply made. What the form becomes is
 * `(prog1 <body> (remove_bindings env 'name …))`: the body unquoted WHERE IT STANDS, and one list for the
 * unbinding however many names there are.
 * ⭐ AND REDUCING THE VALUES FIRST IS WHAT MAKES THEM THE ENCLOSING SCOPE'S, which is what a `let`'s values
 * are: `(let env 'x (+ x 1) '…)` reads the `x` that is bound now, not the one it is about to bind.
 * ⭐ `prog1` AND NOT `begin` IS WHY THE VALUE SURVIVES: `begin` answers its last element, which here is the
 * unbinding, and an unbinding says yes whenever a name comes off.
 *
 * ── WHAT IT REFUSES ─────────────────────────────────────────────────────────────────────────────────
 * A length that is not a verb, an environment, whole pairs and a body · a body that is not quoted · a name
 * that is not quoted, numbered or spelled. ⛔ EVERY ONE IS CHECKED BEFORE THE FIRST NAME IS BOUND, so a refused `let` leaves the
 * environment exactly as it found it rather than half-entered with no unbinding written down — which is
 * the rule a call's expansion keeps for the same reason.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ __noinline__ void sys__bindings__zzpackage_apply_let(sys__heap_node* base,
                                                                       uint64_t form);

#endif /* SILVANN__PACKAGES_SYS_CPU_BINDINGS__HEADER_CUH */
