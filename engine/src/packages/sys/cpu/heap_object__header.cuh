#ifndef SILVANN__PACKAGES_SYS_CPU_HEAP_OBJECT__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_HEAP_OBJECT__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../contracts/defaults.cuh"
#include "../contracts/objects/heap_object.cuh" /* its constants, fault words and layouts */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * What the clone paragraphs below replaced, so nobody restores the older wording.
 *   · "WHAT IT DOES NOT DO" read *"It does not COPY anything today. Every clone row refuses"*, and the
 *     `clone` default was justified by *"nothing in this tree can be copied yet"*. Both went false when
 *     SUBLIST's row named its own clone. The prediction — one row edit — was right; "today" aged.
 *   · The lane-contention sentence in "WHAT IT DOES NOT DO" is a DEVICE sentence. With the evaluator on
 *     a host there are no lanes; the contention a caller owes exclusion against is between RUNNER
 *     THREADS. The lock is unchanged — the debt it names is what has to be restated.
 *     RESTATED: "WHAT IT DOES NOT DO" and "no vtable" now speak of runner threads and name no register
 *     budget; the device reasoning they carried (lanes in lockstep, a table costing kernel registers)
 *     is recorded here and nowhere else.
 * ⛳ RETIREMENT: when the evaluator has moved to the host and the lane sentence is restated.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ what you can do with an object, whatever kind it is ═════════════════════════════════════════════
 *
 * The verbs, and NO COUNT OF THEM HERE: the declarations below are the count, and a number written into
 * this sentence goes stale the first time a verb is added. Every counted thing in this package answers
 * all of them, and this file is where that is said — the declarations only, because what a verb DOES for
 * a given kind cannot be written until the kind exists, and a caller needs to name the verb long before
 * that.
 *
 * ── IT IS AN INTERFACE, AND THE PIECES LINE UP WITH THE ONES A LANGUAGE WITH INTERFACES WOULD HAVE ───
 * ⚖ ARCHITECT: *"a nice way to get interfaces like java but expressed into c."*
 *
 *     this file                 the interface — what may be asked of an object
 *     a row in the X macro      the implements clause — which kinds answer, and with what
 *     the generated switch      the dispatch, written once at the end of the package
 *     the dtype in the head     the type tag, which every allocation already carries
 *
 * ⛳ AND THE LIST IS `sys__LANGUAGE_CONTRACT__OBJECTS` IN `language_contract.cuh`, WHICH IS WHERE A KIND IS
 * ADDED. The package token, then six columns — the dtype, its release, which provider it is carved
 * from, its clone, what builds one when a program asks, and how many nodes one occupies:
 *
 *     X(PKG, SYS__KIND__STACK, sys__stack__zzpackage_release_internal, SYS__COMPUTING_BASE__OBJECTS,
 *                          sys__heap_object__clone__default, sys__stack__zzpackage_construct, 2u)
 *
 * It is its own file, ONE PER PACKAGE — `nn` writes its own at the same filename rather than
 * editing this one — and the full contract is in `packages/manifest.cuh`, which belongs to no package so
 * that a package can read what it owes without opening another. The switches that CALL those methods are
 * in `heap_object__impl.cuh`, which asks for the containers' HEADERS and not for the containers: a switch
 * needs the methods a row names DECLARED and not defined, so nothing has to be ordered around it.
 *
 * ⭐ AND TWO THINGS FALL OUT THAT ARE BETTER THAN THE OBJECT-ORIENTED VERSION, NOT WORSE.
 * There is NO POINTER PER OBJECT. A vtable costs every instance a word and every call an indirection; the
 * dtype is already in the head beside the count, already loaded by anything that got this far, so
 * dispatch is a switch on a value in hand. It is also why the teardown is a switch and not a table of
 * function pointers — a switch has no table to load and no indirect call to make.
 * And COMPLIANCE IS CHECKED BY THE COMPILER, which is the part that usually needs a language for. A row
 * supplies one method per column; a row missing one does not expand, and a row naming a method nobody
 * wrote does not link. Nothing has to assert that a kind implements the interface — a kind that does not
 * cannot be registered.
 *
 * ⛔ WHAT IS NOT CHECKED, STATED SO IT IS NOT ASSUMED: a kind with NO ROW AT ALL. It is not a compile
 * error, it is a refusal at run time on whatever path first releases or clones one. That is the one
 * direction the mechanism cannot close, and all three dispatches — create, release, clone — end in a
 * `default` that raises rather than returning quietly, because a silent no-op there is a leak every
 * counter reports as healthy.
 * ⛳ THAT ARM IS NOT THE SAME THING AS A DEFAULT METHOD, AND THE WORDS IT RAISES SAY SO: `UNKNOWN_KIND`
 * means nobody wrote a row, an unfinished kind; a row that takes the default is a finished kind that
 * chose the ordinary behaviour. Collapsing them would put one word on two causes.
 *
 * ── THE CONTRACT ────────────────────────────────────────────────────────────────────────────────────
 *
 * WHAT THE CALLER OWES
 *   a reference     the offset of an object's second node, never its head. `zzpackage_addressable` is
 *                   exactly that test, and every verb that moves a count, takes a lock or takes an
 *                   object apart runs it first. ⛔ `type` and `is_type` DO NOT, deliberately: they are
 *                   what a caller uses to check something it does not trust, so they answer about a
 *                   pointer rather than raising — and the published `type` wrapper steps a program's
 *                   offset back to a head with nothing in between. The owing is real there and nothing
 *                   makes the caller pay it.
 *   one hold each   a caller that took a reference gives it up ONCE. Holding twice and releasing once
 *                   leaks; the reverse ends something somebody else is still inside.
 *   a row           a kind that reaches `release` or `clone` has a row in `sys__LANGUAGE_CONTRACT__OBJECTS`. NOT
 *                   CHECKED at compile time — a kind with no row is a run-time refusal, which is the one
 *                   direction the mechanism cannot close.
 *   the lock, briefly  whatever takes `lock` releases it. It is bounded, so a holder that never returns
 *                   turns into a raised fault at the NEXT caller rather than a hang.
 *
 * WHAT IT PROMISES BACK
 *   one counting    `retain` and `release` are the only things that move a count, and they answer what is
 *                   left, so a caller can tell a first holder from a second without a second read.
 *   a whole teardown  `release` unwinds the entire structure a dying object held, however deep, without
 *                   any kind knowing an unwinding is in progress. A kind hands over what it held; the
 *                   loop does the rest.
 *   a loud unknown  a kind with no row RAISES rather than returning quietly, because a silent no-op there
 *                   is a leak that every counter reports as healthy.
 *   no vtable       dispatch is a switch on the dtype already in the head. No pointer per object and
 *                   no indirection.
 *
 * WHAT IT DOES NOT DO
 *   It does not decide LIFETIME — it observes it. Nothing here frees anything on a schedule; a thing ends
 *   when its last holder lets go, in that caller's own call.
 *   It copies ONE KIND. SUBLIST names `sys__list__zzpackage_clone_object`, so a reader auditing clone
 *   safety must NOT conclude the path is dead and skip it. Every other row takes the refusing default —
 *   and making a kind clonable is one row edit, not new machinery.
 *   It does not protect against contention INSIDE one block. The lock is a block-level tool: it
 *   excludes runner threads from one another, and a block is one runner thread.
 *
 * ── A VERB CAN HAVE A DEFAULT, AND A ROW TAKES IT BY NAMING IT ───────────────────────────────────────
 * ⚖ ARCHITECT: *"the default should be expressed in the caller like we do with java interfaces, so null
 * is default. if you want something different you need to do inheritance, and maybe call super (which in
 * this case will be sys__heap_object__functionname__default())."*
 *
 * `release` has one and it is a real behaviour rather than a refusal: hand nothing over, which is the
 * whole of what a plain value owes. The room is not its line to write — the teardown gives that back once
 * for every kind, after the row's method has returned. `clone` has one too and it REFUSES — that is the
 * DEFAULT'S behaviour, not a statement about the tree: SUBLIST overrides it.
 *
 *     a row that names the default    takes the ordinary behaviour
 *     a row that names its own        overrides it
 *     an override calling the default calls super — and none of `release`'s needs to, the room being
 *                                     the teardown's line rather than any kind's
 *
 * ⛔ A ROW NAMES ITS DEFAULT RATHER THAN LEAVING THE COLUMN EMPTY, AND THAT IS A DELIBERATE DEPARTURE
 * FROM "null is default". An empty macro argument is legal and expands to nothing, so `return CLONE(a,b)`
 * would become `return (a,b)` — the comma operator, which compiles, returns the wrong thing and warns
 * about none of it. Naming the default costs one word per row and makes which kinds override readable
 * from the list itself, without opening a file to find out.
 *
 * ⛳ AND THE ROOM IS GIVEN BACK IN ONE PLACE, WHICH IS NEITHER THE DEFAULT NOR AN OVERRIDE. The teardown
 * gives it back once for every kind, after the row's method has returned, so no kind's method names a
 * chunk at all and there is nowhere for two lines to disagree. A method says what it was HOLDING; where
 * it sat is the allocator's business.
 *
 * ── WHY THE DECLARATIONS ARE HERE AND NOT WITH THEIR FIRST CALLER ────────────────────────────────────
 * A promise that lives with its earliest caller has to be relocated the moment an earlier caller
 * appears, and somebody has to notice. The thing being described is not a fact about who calls it
 * anyway — it is the surface of an object, and a surface belongs in one file that says so.
 *
 * ⛳ AND EVERY ONE OF THEM IS DEFINED IN THE SAME PLACE: `heap_object__impl.cuh`, which asks for every
 * container's HEADER because its three dispatches call every container's method — declared is all a
 * switch needs, which is why the package's implementations are a LIST and not an order. Nothing here
 * needs to know that, which is the point of it.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── ONE PACKAGE-VISIBLE HELPER, PROMISED HERE FOR THE SAME REASON THE VERBS ARE ─────────────────────
 * Not part of the surface a caller outside this package sees, but files ordered before the implementation
 * call it — the error type when it checks a reference, the stack when it looks up a field, and the list,
 * the array, the bindings and a procedure when they check an offset they were handed. A promise belongs
 * where the earliest caller can see it, and for this family that is here rather than in whichever file
 * happened to need one first.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline bool sys__heap_object__zzpackage_addressable(uint64_t offset);

/* ── WHERE THE SHAPE LIVES, AS OPPOSED TO THE BEHAVIOUR ─────────────────────────────────────────────
 * These say where an object keeps what it is and who wants it. They are in the HEADER because they are
 * facts about the layout every kind shares — a container writing a fresh head reads them, and so does the
 * counting — while everything that ACTS on them is in the implementation, ordered last.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ── WHERE AN OBJECT KEEPS WHAT IT IS AND WHO WANTS IT ───────────────────────────────────────────────
 * Both in the first word of the node an allocation begins with: the kind in one half, the count of what
 * refers to it in the other. They are read together far more often than apart — a release has to know
 * whether it was the last, and if it was, what it is now holding the pieces of — so putting them in one
 * word makes that one load instead of two.
 *
 * It also leaves the node's arguments entirely to whatever is being counted.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
#define SYS__HEAP_OBJECT__COUNT(head)  ((head)->num_args)

/* How many nodes this allocation occupies, INCLUDING its head. Written once, when it is made.
 *
 * ⭐ IT IS RECORDED BECAUSE IT CANNOT BE DERIVED. Kinds differ in size — a two-node error beside a
 * whole array chunk — so nothing outside an allocation can work out where it ends. A uniform
 * size would make this a constant anybody could apply and this field unnecessary; the price of a kind
 * choosing its own size is that the size travels with it.
 *
 * ⛳ AND IT COSTS NOTHING. The word it lives in is already in every node and has no other tenant here. */
#define SYS__HEAP_OBJECT__NODES(head)  ((head)->op_code)

/* What an object IS — the third thing its head carries, beside the count and the lock.
 *
 * FOUR VERBS: two questions, and each in two shapes. `_from_head` is the shape axis.
 *
 *     type(object)                    what does it say it is
 *     type_from_head(head)
 *     is_type(object, kind)           is it this
 *     is_type_from_head(head, kind)
 *
 * WHICH SHAPE YOU WANT IS DECIDED BY WHAT YOU ARE HOLDING, which is the only thing a caller ever knows:
 * the plain form takes the OBJECT, one node past the head, which is what every container hands out and
 * steps back for you; `_from_head` takes the ALLOCATION HEAD, which a chunk has because its head is its
 * first node, and so does anything that has just stepped a reference back to its head.
 *
 * ⛳ A NULL ANSWERS `SYS__KIND__INVALID`, which is a kind no allocation ever carries. That is what lets
 * the two questions be the SAME question — `is_type` is exactly `type() == kind`, with no null case of
 * its own — and it is why the dtype exists at all.
 * ⛔ ASKING ABOUT NOTHING IS NOT AN ERROR AND MUST NOT RAISE, which is the alternative that was refused:
 * `is_type` is what a caller uses to VALIDATE something it does not trust, and a validator that faults on
 * bad input cannot be used to check for bad input. `error__is`, `computing_base__is` and the stack's
 * fields lookup all call it for exactly that.
 * ⚠ IT IS A POINTER CHECK AND NOT A VALIDITY ONE. A non-null pointer to garbage answers whatever the
 * garbage says; what tests THAT is `zzpackage_addressable` and the head's own tag.
 *
 * ⛔ NEITHER IS FOR A CELL. `cell->dtype == SYS__KIND__OBJECT_REFERENCE` asks whether a value in a slot is a
 * reference — the cell's own tag, not an allocation's head. The two are written identically and are not
 * the same test: one asks what an object IS, the other what a CELL HOLDS.
 *
 * ⛳ A TYPE MAY ALSO PUBLISH ITS OWN, with the kind baked in — `sys__error__is`, `sys__computing_base__is`.
 * Those are a convenience over these rather than an alternative to them. */
static __device__ inline sys__kind sys__heap_object__type_from_head(const sys__heap_node* head) {
    return (head == 0) ? SYS__KIND__INVALID : head->dtype;
}

static __device__ inline sys__kind sys__heap_object__type(const sys__heap_node* object) {
    /* The null test is here because the ARITHMETIC is here: `&object[-1]` off a null pointer is a small
     * negative address, not a null one. The other three need no guard of their own. */
    return (object == 0) ? SYS__KIND__INVALID : sys__heap_object__type_from_head(&object[-1]);
}

static __device__ inline bool sys__heap_object__is_type_from_head(const sys__heap_node* head, sys__kind kind) {
    return sys__heap_object__type_from_head(head) == kind;
}

static __device__ inline bool sys__heap_object__is_type(const sys__heap_node* object, sys__kind kind) {
    return sys__heap_object__type(object) == kind;
}

/* Make one of a kind FROM THE AST, and hand back a reference to it.
 *
 * ⭐ THIS FILE IS WHAT AN OBJECT LOOKS LIKE TO A PROGRAM, AND THAT IS WHY THE CONSTRUCTOR IS HERE RATHER
 * THAN THE OTHER WAY ROUND. Every kind already has a C constructor of its own — `sys__stack__create()`,
 * and whatever a later kind brings — and those stay exactly as they are, callable directly by anything
 * written in C. This is a WRAPPER over them, reached with a kind and a list of parameters, which is the
 * shape a program has when it asks for something: `create(car(list), cdr(list))`.
 *
 * ⛔⛔ AND NOT EVERY KIND ANSWERS. A stack's CHUNKS are a kind — they are counted, they are torn down by
 * the same dispatch — and no program should ever be able to ask for one: they are a stack's internal
 * storage and exist only because a stack grew. A kind like that names `sys__heap_object__create_refusal`
 * in its row, which raises.
 * ⇒ ★ SO THE ROW ITSELF SAYS WHETHER A KIND SURFACES, and no separate column is needed to record it. What
 * a kind can be asked for from the language is exactly what its constructor column is willing to build.
 *
 * `parameters` is what the kind's own constructor reads its arguments from, and is null for a kind that
 * takes none. It is ONE NODE, and the signature does not change when it becomes the tail of the call —
 * a list handle IS a node, so what gets richer is what it points at.
 * ⛳ WHICH IS ALSO THE LIMIT WORTH KNOWING, AND IT IS THE WRAPPER'S RATHER THAN THE LIST'S. Lists are
 * here; the published `create` SHIFTS a call's arguments into a local node, and a node carries six.
 * Taking the kind out of the first leaves five, so a kind wanting more than five cannot be asked for from
 * a program until that shift is a real `cdr`. */
static __device__ inline sys__heap_node sys__heap_object__create(sys__heap_node* base, sys__kind kind,
                                                            const sys__heap_node* parameters);
/* What a kind names when it must not be reachable from a program. It raises and answers nothing, so the
 * refusal is loud at the moment it is asked for rather than quiet with an object nobody should have.
 * ⛳ IT IS A NAMED VERB AND NOT AN ABSENT ROW, because a kind with no row at all is refused by the
 * dispatch's `default` and reads as an oversight. Naming this one says the absence is a DECISION. */
static __device__ inline sys__heap_node sys__heap_object__create_refusal(sys__heap_node* base,
                                                                    const sys__heap_node* parameters);
/* And what a kind names when creating it is carving it and nothing else — it holds nothing, so there is
 * nothing to fill in. */
static __device__ inline sys__heap_node sys__heap_object__create__plain(sys__heap_node* base,
                                                                   const sys__heap_node* parameters);

/* A reference to what lives at `offset`. Three constructors were writing these three fields out, and the
 * one that matters is `num_args`: a REFERENCE carries no arguments of its own, and the count that governs
 * the object lives in the object's head rather than here. Getting that wrong is a cell that looks like a
 * counted thing. */
static __device__ inline sys__heap_node sys__heap_object__reference_to(uint64_t offset);
/* ── THE TWO QUESTIONS ABOUT A TAG, ASKED IN ONE PLACE AND SIDE BY SIDE ───────────────────
 * A cell holding a heap offset has to be counted when it goes into a container, given up when it leaves,
 * and MOVED when its container dies. For a long time that was one tag and every site tested for it by
 * name — and the moment a second tag carried an offset, every one of those sites was silently wrong.
 * ⛔ `MEASURED`: a QUOTED_LIST inside a list was never retained and never moved, so a deep copy released
 * the only hold and the copied arm was freed before it ran. Nothing failed; the list was simply empty
 * when the evaluator reached it.
 * ⇒ ⭐ SO THE QUESTION IS ASKED HERE AND NOWHERE ELSE. A tag that carries a reference is added to this one
 * function, and every container inherits it — which is the difference between adding a tag and auditing
 * the tree for places that assumed there was only one.
 *
 * ⭐⭐ AND THERE ARE TWO SUCH QUESTIONS, WHICH IS WHY THEY SIT TOGETHER.
 * ⛳ A THIRD ONE WAS LOOKED FOR AND IS NOT THERE, WHICH IS WORTH KNOWING BEFORE SOMEBODY ADDS IT. *"What
 * may stand at the head of a form"* reads like a partition of these same tags and is not one: a list of a
 * single cell holding a procedure is a CALL when the evaluator arrives at it and an ANSWER when the
 * evaluator comes back to it, and the two objects are identical. No tag can separate them, so the
 * evaluator separates them by where it came from and asks nothing here.
 * ⇒ ★ A QUESTION THAT LOOKS LIKE A PROPERTY OF THE VALUE CAN TURN OUT TO BE A PROPERTY OF THE VISIT. The tags that carry a reference
 * and the tags that are PART OF the structure holding them are different sets, and a copy needs the
 * second where the accounting needs the first:
 * ```
 *   OBJECT_REFERENCE  naming a list — a piece of the structure              reference · substructure
 *   QUOTED_LIST    a piece too: inert until something unquotes it, and
 *                  consumed from that moment on exactly like anything else  reference · substructure
 *   PROCEDURE      a shared VALUE the structure POINTS AT. Giving a copy
 *                  its own procedure gives it one that cannot find itself,
 *                  and a procedure that cannot find itself cannot recurse   reference · ---
 *   QUOTED_ARRAY   an array a list held, as it stands inside a PICTURE of
 *                  that list. It is the same array, shared; the tag only
 *                  says it was an array before the picture was made         reference · ---
 *   FROZEN_LIST    a sub-form that has not been thawed yet. It names the
 *                  ARRAY in the picture it would be made from — the SAME
 *                  array, shared, exactly as QUOTED_ARRAY is, and the tag
 *                  only says this one is meant to be RUN rather than read   reference · ---
 * ```
 * ⭐⭐ `FROZEN_LIST` SITS IN THE SAME TWO BOXES AS `QUOTED_ARRAY`, AND THAT IS THE ARGUMENT FOR IT RATHER
 * THAN A COINCIDENCE. It is a reference, because the array must outlive the cell that names it and a hold
 * is what says so. It is NOT substructure, because the array is SHARED with the picture — a copy that
 * followed it would duplicate a thing the picture owns, which is the same reason `QUOTED_ARRAY` is
 * excluded. ⛳ What separates the two is only what the EVALUATOR does on meeting one: it reads past a
 * quoted array and descends into a frozen list.
 * ⛔ AND BOTH LINES ARE WRITTEN IN ONE EDIT, FOR THE REASON THE PARAGRAPH BELOW GIVES.
 * ⛔⛔ BOTH DEFECTS THIS COMPONENT HAS PRODUCED WERE A TAG ADDED TO ONE OF THESE SETS AND NOT THE OTHER,
 * and in both directions: QUOTED_LIST reached the accounting and not the copy, and PROCEDURE reached the
 * copy and not a retain guard. ⇒ ⭐ THEY ARE ADJACENT SO THAT ADDING A TAG IS ONE EDIT WITH BOTH ANSWERS
 * IN SIGHT, rather than one edit and a search for whatever else partitions the same set.
 * ─────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline bool sys__heap_node__carries_reference(sys__kind dtype) {
    return dtype == SYS__KIND__OBJECT_REFERENCE
        || dtype == SYS__KIND__QUOTED_LIST
        || dtype == SYS__KIND__PROCEDURE_REFERENCE
        || dtype == SYS__KIND__QUOTED_ARRAY
        || dtype == SYS__KIND__FROZEN_LIST;
}

/* Whether what this cell names is PART OF the structure rather than something it merely points at — the
 * question a copy asks of every cell it meets. ⛔ IT IS NOT ENOUGH ON ITS OWN: the tag says a heap offset
 * is there and what it means, and the CONTAINER still has to say the offset is one of its own kind. */
static __device__ inline bool sys__heap_node__is_substructure(sys__kind dtype) {
    return dtype == SYS__KIND__OBJECT_REFERENCE
        || dtype == SYS__KIND__QUOTED_LIST;
}

/* Become one of the holders. Answers the count afterwards, so a caller can tell a first holder from a
 * second. */
static __device__ inline uint32_t sys__heap_object__retain(uint64_t offset);

/* Let one hold go, and if it was the last, take the object apart and everything that dies with it.
 * ⛳ NOT INLINED, AND THE ARGUMENT FOR IT IS WHERE IT IS DEFINED. */
static __device__ __noinline__ uint32_t sys__heap_object__release(uint64_t offset);

/* The same, through a value CELL rather than a bare reference — and it clears the cell.
 *
 * A cell is what a container HANDS YOU: `sys__stack__pop`, `sys__stack__peek` and
 * `sys__computing_base__read_result` all answer one. So this is the verb a CALLER wants, and `release`
 * above is the one a container's own teardown uses.
 * ⛔ IT IS NOT HOW A CONTAINER'S CONTENTS ARE FREED. A stack does that through its row in the kind list,
 * which hands its whole chain to the unwinding inside `release`. Nothing walks a container from out here. */
static __device__ inline void sys__heap_object__release_from_reference(sys__heap_node* cell);

/* Write a cell: take the new reference before letting the old one go, since they can be the same object.
 * The assignment a program actually means. */
static __device__ inline void sys__heap_object__set(sys__heap_node* cell, const sys__heap_node* value);

/* The defaults. A row naming one of these takes the ordinary behaviour; a row naming its own method
 * overrides it, and may call the default as super — none of `release`'s needs to, the room going back in
 * the teardown once for every kind rather than inside any method.
 * They are public because a row in another package names them exactly as a row here does. */
static __device__ __noinline__ void sys__heap_object__release__default(sys__heap_node* head,
                                                               uint64_t* releaser_stack);
static __device__ inline uint64_t sys__heap_object__clone__default(sys__heap_node* head,
                                                                  sys__heap_node* computing_base);

/* Make a copy, carved from the block doing the copying rather than the one that owned the original — the
 * argument for which block is beside the definition. Answers an offset, because a value has to survive
 * being copied into a cell some other block will read. */
static __device__ inline uint64_t sys__heap_object__clone(uint64_t offset, sys__heap_node* computing_base);

/* Take the object for the length of a write, and hand it back. The lock is one word in the head, at the
 * same place in every kind. ⛔ YOU MAY ONLY LOCK WHAT YOU HOLD A REFERENCE TO: the lock lives inside the
 * thing it protects, so an object taken apart while somebody waits on its lock leaves that waiter
 * spinning on memory the heap allocation pool has already handed out. */
static __device__ inline bool sys__heap_object__lock(uint64_t offset);
static __device__ inline void sys__heap_object__lock_release(uint64_t offset);

#endif /* SILVANN__PACKAGES_SYS_CPU_HEAP_OBJECT__HEADER_CUH */
