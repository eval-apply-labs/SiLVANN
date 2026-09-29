#ifndef SILVANN__PACKAGES_SYS_CPU_STACK__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_STACK__HEADER_CUH
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 *
 * ⚖ ARCHITECT: *"look at the stack split"*, under the standing preference to *"slice and mechanically
 * copy things ... if you make changes let me know where."*
 *
 * WHAT WAS COPIED AND WHAT WAS WRITTEN:
 *   COPIED VERBATIM  `sys_stack — a chunked stack...` (:60-270) — the operations, the same four over a
 *                    chunk, why the cell tags matter, when a reference is given up, the chunk head
 *                    layout, `ALLOC`/`SLOTS`, the `sys__stack_chunk` struct, both static asserts, the
 *                    failure discussion and the two fault words.
 *   COPIED VERBATIM  `THE SURFACE — A STACK IS AN OBJECT...` (:643-666) — the holder's two nodes, the
 *                    granule it costs, where the lock lives, and `SYS__STACK__CURRENT_STACK_CHUNK`.
 *   DERIVED          the nineteen declarations, from the definitions they stand for.
 *   NEW              the contract. The file argued every clause of it somewhere and never as an
 *                    obligation; nothing was displaced to make room.
 *   NOT MOVED        the implementation's own AI banner, which is a to-do list about ITS code.
 *
 * ⛳ THE TWO COPIED BLOCKS ARE IN SOURCE ORDER — chunk tier first, surface second — because that is how
 * the file was written and reordering them would be a change rather than a slice. A reader meeting a
 * header usually wants the surface first, so this is worth flipping if you agree; it is one move and
 * nothing depends on the order.
 *
 * ⛔ AND THE SPLIT BUYS NO ORDERING HERE, UNLIKE THE OTHER TWO. Both halves must still come after
 * `heap__header.cuh`: the asserts name `SYS__HEAP__CHUNK_NODES` and the layout names `SYS__ALLOC__ARRAY_GRANULE`. The
 * split is for READING, which is the architect's own reason for wanting it, and it is worth saying that
 * the usual second reason does not apply — so nobody later "fixes" the include order expecting a gain.

 * RETIREMENT: delete when the old tree is gone — the block positions this file against the one it was
 * split out of, and neither half of that comparison outlives the port.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════ */
/* What this file needs, named where a reader — and an editor — can follow it. */

/* ══ a stack — a chunked LIFO held in a flat array of AST nodes ══════════════════════════════════════
 *
 * An array-based stack of objects, used for example for the binding lexical scoping.
 * Access is strictly last-in-first-out and only the top value is accessible.
 *
 * The stack is a chain of fixed-size array chunks, sourced from the local array chunk provider
 * (reachable from the block computing base pointer) and accessed through a holder node via LIFO.
 * Each chunk contains SYS__STACK_CHUNK__ALLOC nodes, carved at the allocator's watermark wherever the last
 * allocation ended rather than on any boundary, and the FIRST node is not a value.
 * It is the node saying what the thing is (an stack chunk) and how many things are HOLDING IT — that
 * count is the chunk's own, not a count of what is inside it.
 * The chunk's state fields live in the FIRST node's arguments, so no value slot is spent on them:
 *
 *     head.dtype        SYS__KIND__STACK_CHUNK      what this is, so the counting knows how to end it
 *     head.num_args     the reference count          how many hold this chunk
 *     head.args[0]      the lock                     present in every allocation head, and NOT taken here
 *     head.args[1]      the cursor                   HOW MANY values are in this chunk, 0 for none
 *     head.args[2]      the previous                 the chunk before this one, 0 for none
 *
 *     cell[0 .. SYS__STACK_CHUNK__SLOTS-1]                   the values, all of them
 *
 * ⛔ THE LOCK WORD IS THERE BECAUSE EVERY ALLOCATION HEAD HAS ONE, AND NOTHING IN THIS PACKAGE TAKES IT
 * ON A CHUNK. The exclusion is one tier up: every caller of `sys__heap_object__lock` locks the stack's
 * HOLDER, and the chunk-level verbs underneath run inside a holder that is already locked. ⇒ ★ SO THE
 * SAFETY OF A CHUNK WRITE IS A PROPERTY OF THE CALLER AND NEVER OF THE CHUNK — a cursor bump reached
 * without the holder's lock is a cursor bump racing whatever else is in there, and an unlocked push
 * racing a pop corrupts it. The verbs that reach a chunk holding nothing are named in the implementation
 * and each one says what makes it safe; adding another is a decision, not an addition.
 *
 * ⭐ THE CURSOR IS A COUNT, NOT AN INDEX. Empty is 0 and the top is `cell[cursor - 1]`, which is what
 * lets "how many are here" and "where is the top" be one number instead of two that can disagree.
 * It is local to its chunk, so reading the top is one indexed access with no arithmetic and no walk.
 *
 * A stack holding nothing references a null pointer, and every chunk the holder can hold always has a cursor
 * of at least 1: push initialises a new chunk when the holder has a null address, and pop either hands back 
 * to the holder a chunk with something in it or a null. A referenced chunk with a cursor of 0 cannot be produced,
 * which is why pop raises on one rather than returning empty-handed.
 *
 * This strategy helps both the code being leaner and to release ast chunks when the list is empty, which
 * is especially good if you are going to hold an array of stacks (like in the bindings table) which are
 * expected to operate under a specific lifetime which ends with them sitting as an empty stacks.
 *
 * ── THE OPERATIONS ──────────────────────────────────────────────────────────────────────────────────
 * A stack is reached through a HOLDER — an ordinary cell whose args[0] is a REFERENCE to the stack object,
 * which is the same thing a binding slot already is. These five resolve that reference to the object's own
 * field node — checking on the way that what it names says it is a stack — and write the chunk switch
 * THERE. The caller's cell is read, never written, so nothing is ever handed back for a caller to store
 * and the name a program holds stays put however the top moves.
 *
 *     sys__heap_node  sys__stack__create ();                                            //a reference to a new one
 *     bool        sys__stack__push   (const sys__heap_node* stack, const sys__heap_node* value);
 *     sys__heap_node  sys__stack__pop    (const sys__heap_node* stack);                     //the value
 *     sys__heap_node  sys__stack__peek   (const sys__heap_node* stack);                     //the value, null if empty
 *     bool        sys__stack__empty  (const sys__heap_node* stack);
 *     void        sys__stack__replace(const sys__heap_node* stack, const sys__heap_node* value);
 *
 * ⛳ THERE IS NO PUBLIC `sys__stack__release`, AND THAT IS DELIBERATE. Being finished with a stack is not a
 * stack operation: it is `sys__heap_object__release(stack)`, the same call that ends anything. It is a
 * RELEASE and not a destroy — you never know who else is holding it, so the count goes down by one and
 * the stack ends only if that leaves none. A stack-shaped spelling would be a second verb for an act that
 * is already general, and one a caller would have to know to look for.
 *
 * ⛳ The method that runs when it does reach none is `sys__stack__zzpackage_release_internal`, which the dispatch
 * finds through the dtype. Nobody calls it directly.
 *
 * ⭐ NOTHING IS HANDED ANYTHING BEYOND WHAT IT OPERATES ON, AND THAT IS WHAT MAKES THESE WRITABLE AS
 * OPCODES. The counting is in the objects, so releasing needs no table, and a dying object hands over
 * what it held rather than writing it somewhere, so there is no collection list either — the release
 * loop's own local is the whole of it. What a block is carving from is reached by block index. And the chunk switch
 * goes into the holder rather than into a return value — because an opcode is an expression, and one that
 * also had to be assigned back would be a second form the program must not forget.
 *
 * ⛳ Peek and pop return the VALUE. A node is returned by copy, which on every ABI here is the same
 * hidden-pointer write an out-parameter compiles to, so reading as an expression costs nothing.
 * ⛔ AND THE TWO PART COMPANY ON AN EMPTY STACK, WHICH IS THE EASIEST LINE HERE TO READ PAST. Peek
 * answers with a null value and raises nothing: asking what is on top of nothing is a question with an
 * answer. Pop RAISES `SYS__STACK__FAULT_EMPTY` and hands back nothing — a referenced chunk with a cursor
 * of zero cannot be produced, so a stack with no chunk at all means the caller asked for a value that was
 * never there. `REASONED` from `sys__stack__zzpackage_pop_deferring_at`, which every pop in this file is
 * written in terms of, against `sys__stack_chunk__zzprivate_peek`, which answers null and raises nothing;
 * the FAILURE section below names the empty stack among the conditions that raise, and the contract asks
 * for a question first for this reason and no other.
 * ⇒ ⛳ `while (!empty) pop` is the shape that costs nothing; popping until the answer reads null floods
 * the fault log with an entry per turn of the loop.
 *
 * Replace changes the top without moving the cursor or the chain. It is not sugar over a pop and a push:
 * at a chunk boundary that pair releases a chunk and immediately claims another, and replacing writes a
 * word.
 *
 * Push and pop own the chunk chain. Push allocates a new chunk when the current one fills; pop gives one
 * up when its last value leaves and makes the previous one current. Neither ever leaves a holder naming a
 * half-finished structure. Release ends the whole stack at once, which is what a scope closing wants when
 * it would otherwise pop in a loop.
 *
 * ── THE SAME FOUR, OVER A CHUNK ─────────────────────────────────────────────────────────────────────
 * Underneath is the mechanism the surface drives, and it does not answer uniformly.
 * `sys__stack_chunk__zzpackage_push` and `_pop` take a chunk and hand back the one that is now the top,
 * because those two are the operations that can change which one it is. `sys__stack_chunk__zzprivate_peek`
 * hands back the VALUE, and `sys__stack_chunk__zzprivate_replace` hands back nothing at all — neither of
 * those moves the cursor off the chunk it was given, so there is no address either could have to give.
 * They are what the layer above exists to spare a caller, and they are where the chunk arithmetic is tested.
 *
 * TWO of the package names here are the ones the counting's rows name:
 * `sys__stack__zzpackage_release_internal`, which runs when a stack dies, and `sys__stack_chunk__zzpackage_release_internal`,
 * which runs when one of its chunks does. What is particular about that pair is that NOBODY CALLS EITHER —
 * the teardown's generated switch reaches them through the dtype.
 * ⛳ THEY ARE NOT THE WHOLE OF THE MARKER, AND THE TWO CHUNK VERBS NAMED A PARAGRAPH ABOVE ARE THE PROOF.
 * Every `zzpackage_` name in this file is either declared among the public verbs further down or listed
 * under OUTSIDE THE CONTRACT, which says of each one who reaches it. The marker draws a wall around `sys`
 * and asserts no count: what it promises is that nothing carrying it is reached from outside the PACKAGE.
 *
 * `sys` is written in the C subset, which has no access control, so the marker is the only 
 * thing a reader has at a call site — and the gate is what makes it more than a request.
 *
 * ── WHY THE CELL TAGS MATTER ────────────────────────────────────────────────────────────────────────
 * Every cell in a chunk is a value, and its dtype is what says whether anything is owed when it leaves:
 * a cell tagged as an object reference is released, and anything else is left alone. Releasing an
 * integer would decrement a count indexed by a value, and the damage would land on whatever object
 * happens to live at that offset.
 *
 * A cell is a whole AST node rather than a packed element, because the walk that empties a chunk steps
 * one node at a time. A packed element would be strided past and its references never released.
 *
 * The header needs no tag of its own, because it is not among the cells: the cursor and the previous live in
 * the allocation's head node, which nothing walking values ever touches. There is no step that has to
 * recognise them and skip them, and so no tag that can be forgotten.
 *
 * ── WHEN A REFERENCE IS GIVEN UP, AND WHEN THE MEMORY GOES BACK ─────────────────────────────────────
 * These are two different moments and the file treats them differently on purpose.
 *
 * Releasing objects is EAGER. A value's ARC count drops the moment it leaves the stack, whether that
 * happens one pop at a time or because the stack holding it was disposed of. The alternative — handing the
 * chunk over with a length and letting the cascade do the releasing — is free, and it was refused:
 * it would make the moment an object's count drops depend on whether that particular pop happened to
 * empty a chunk, which is a function of the chunk geometry and of how many pushes preceded it. An
 * object's lifetime should not be readable only by knowing how wide a chunk is.
 * Releasing chunks is also eager: a chunk's ARC count drops the moment it leaves the stack.
 * It has to be that way because we need to always hold the array we can peek, so chunks can't be
 * kept empty, it is the push that will reallocate and we have to eat the churn for better peek performance.
 * One quirk: by default the deallocated values are not removed from the chunk, we simply read up to
 * the cursor, so whatever is beyond that will never be looked at, they just sit there as zombies.
 * Just keep it in mind and avoid to do a second deallocate on them when removing the chunk, as that
 * will cause an ARC mismatch.
 *
 * Releasing does NOT mean deallocating, as the decision is in the hands of the ARC that has the wider
 * visibility and knows if someone else is holding references to the objects we are done with.
 */

#include "heap_object__header.cuh" /* the holds a stack takes on what it carries */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../contracts/objects/stack.cuh" /* its constants, fault words and layouts */
/* ══ THE SURFACE — A STACK IS AN OBJECT, AND WHAT A PROGRAM HOLDS IS A REFERENCE TO IT ═══════════════
 *
 * The stack object is just a stable head node that offers an indirection to the LIFO array chunk 
 * (which actually host its content)
 *
 *     head[0]        dtype = SYS__KIND__STACK, the count, and the lock
 *     head[1].args[0] the chunk that is the top, or zero for a stack with nothing in it
 *
 * ⛳ AND IT COSTS TWO NODES PER STACK, EMPTY OR NOT. That is the price of being nameable: an identity
 * has to be somewhere. What it buys back is that ending one is `sys__heap_object__release(reference)` like
 * anything else — the dispatch reads SYS__KIND__STACK and calls the method below, which nobody calls
 * directly.
 *
 * ⛳ AND THE LOCK IS HERE AND NOWHERE ELSE. 
 * The stack chunks are internal by definition, so the access control is moved at the holder. A chunk is
 * reachable only through the stack that owns it, so holding the stack excludes everybody from every 
 * chunk in it — and the chunk tier takes nothing, which is why there is no pair of these held at 
 * once and no order to get wrong.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── THE CONTRACT ────────────────────────────────────────────────────────────────────────────────────
 * Every clause is argued somewhere above or in the implementation; this gathers them as obligations.
 *
 * WHAT THE CALLER OWES
 *   a reference     the VALUE CELL naming the stack — the thing `create` answered — never the object
 *                   node itself. Three things are checked on every call: that the cell holds a reference
 *                   at all, that the reference could be one of ours, and that what it names is a STACK.
 *   a release for   `pop` TRANSFERS the hold the stack had; `peek` TAKES a new one. Either way the caller
 *   what it takes   leaves holding something and owes `sys__heap_object__release` for it. ⛔ The two are
 *                   symmetric in what they OWE and not in what they DO, which is the easiest thing here
 *                   to get wrong: after a pop no count has moved, because nothing was made or ended.
 *   a question      before `pop` or `peek` on a stack that may be empty. `empty` is that question and it
 *   first           costs nothing; asking anyway and taking the fault is a log entry nobody needs.
 *   the ending      is not a stack verb. Being finished with one is `sys__heap_object__release(stack)`,
 *                   the same call that ends anything, and the dispatch finds the method.
 *
 * WHAT IT PROMISES BACK
 *   a stable name   the reference NEVER MOVES when the top does, so a stack can be named from two places
 *                   at once and neither has to learn that it grew or shrank.
 *   last in, first  out, to the limit of the heap and no fixed depth of its own.
 *   a null, not a   `peek` and `pop` on an empty stack answer the null value rather than reading a chunk
 *   read            that is not there. Both read as expressions, so neither needs an out-parameter.
 *   one lock, at    the holder, held for the length of a push, a pop, a peek or a replace. The chunks
 *   the holder      underneath take nothing, so there is one lock to hold and no order to get wrong.
 *                   ⛔ `transfer` IS THE EXCEPTION AND A DELIBERATE ONE: it takes no lock, because every
 *                   caller hands it a stack whose offset nothing else can name. The argument is beside
 *                   its definition, and what would make this clause false is a caller publishing a stack
 *                   fed through it — a push racing a pop there corrupts the cursor with nothing to
 *                   exclude either of them.
 *   an eager end    a chunk that empties is released as it goes, not left for a collector. ⚖ *"we dont
 *                   want the lifetime of an object to depend on random chance of the detail of the
 *                   length of the stack chunk."*
 *
 * WHAT IT DOES NOT DO
 *   It does not publish a RELEASE of its own, deliberately — see the ending clause above.
 *   It does not COPY a value. What is pushed is the cell, and a reference in it gains a hold.
 *   It does not report RUNNING OUT OF ROOM in its own words. That is the heap's condition, raised where
 *   the cause is known — a stack has nothing of its own to say about it, and the list and the
 *   dictionary meet the same wall, so one condition must not read as three in a log.
 *   It does not divide a block; the lock excludes other blocks, each one runner thread, as everywhere here.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* ── THE CONTRACT, DECLARED ─────────────────────────────────────────────────────────────────────────
 * Eight public verbs, with four `zzpackage_` ones standing among them. ⚖ ARCHITECT: *"the only public
 * methods should be push pop and peek"*, then *"we do need 4 methods after all"* — and `create`, `empty`
 * and `replace` joined them for reasons argued above.
 * ⛔ SO DID `transfer` AND `pop_deferring_disposal`, AND THOSE TWO ARE THE ONES AN AUDIT MOST NEEDS TO
 * SEE: `transfer` takes no lock, and `pop_deferring_disposal` leaves a chunk's disposal owed to whoever
 * called it. A count that stopped at the three the quotes name would skip exactly the pair with the
 * sharpest obligations. Every one of them is defined in `stack__impl.cuh`.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__heap_node sys__stack__create (void);
/* The same thing asked for from the AST: the kind list names this, and it wraps the verb above. */
static __device__ inline sys__heap_node sys__stack__zzpackage_construct(sys__heap_node* base,
                                                                    const sys__heap_node* parameters);
static __device__ inline bool       sys__stack__push   (const sys__heap_node* stack_reference,
                                                        const sys__heap_node* value);
static __device__ inline sys__heap_node sys__stack__pop    (const sys__heap_node* stack_reference);
/* ⭐ THE SAME POP, EXCEPT THAT IT DISPOSES OF NOTHING AND SAYS WHAT IS OWED INSTEAD. Taking the last
 * value out of a chunk leaves that chunk with nothing in it, and the ordinary pop above gives it up on
 * the spot. That is right for every caller but one: the COLLECTOR ITSELF cannot ask for something to be
 * given up, because being asked is what it is currently doing. So this hands the emptied chunk's offset
 * back through `out_spent` — zero when nothing emptied — and the caller owes it a disposal.
 * ⛳ IT IS NOT A SECOND IMPLEMENTATION: the ordinary pop is written in terms of this one. */
static __device__ inline sys__heap_node sys__stack__pop_deferring_disposal(const sys__heap_node* stack_reference,
                                                                       uint64_t* out_spent);
/* Move a value in without taking a hold of it: the owner changes and the count does not. The handle is
 * MUTABLE because a slot holding nothing becomes a stack on the first value — see the body for why a
 * teardown wants that. */
static __device__ inline void       sys__stack__transfer(uint64_t* stack_offset,
                                                        const sys__heap_node* value);
static __device__ inline sys__heap_node sys__stack__peek   (const sys__heap_node* stack_reference);
/* The same read, BORROWED — no lock and no hold. ⛔ Correct ONLY on a contiguous piece of C where
 * nothing can pop this stack before the caller takes its own hold, and its mistake is a use after free
 * that no counter will notice. The argument is beside the definition. */
static __device__ inline sys__heap_node sys__stack__zzpackage_peek_borrowed(const sys__heap_node* stack_reference);
static __device__ inline bool       sys__stack__empty  (const sys__heap_node* stack_reference);

/* The same two an offset apart, for a holder that has an offset and not a node — the teardown's
 * worklist handle is one. */
static __device__ inline bool       sys__stack__zzpackage_empty_at(uint64_t stack_offset);
static __device__ inline sys__heap_node sys__stack__zzpackage_pop_deferring_at(uint64_t stack_offset,
                                                                          uint64_t* out_spent);
static __device__ inline void       sys__stack__replace(const sys__heap_node* stack_reference,
                                                        const sys__heap_node* value);

/* ── OUTSIDE THE CONTRACT ───────────────────────────────────────────────────────────────────────────
 * Reachable from the rest of `sys`, and they divide by WHO reaches them.
 *
 * ⛳ AND ONE TIER IN THIS FILE REACHES FURTHER THAN THE PACKAGE, WHICH A READER ASKING WHAT BREAKS IF
 * THIS CHANGES HAS TO BE TOLD HERE OR NOWHERE. `sys__stack__zzengine_compute_stack_push` and `_pop`, in
 * `stack__impl.cuh`, carry a third marker: `zzengine_`, which `src_c_subset_gate.py` reads as package
 * AND engine. The evaluator calls both — they are the unlocked, unowning pair its descent stack wants,
 * and the argument for each half is beside their definitions. Nothing else here goes past `sys`.
 *
 * The two `release_internal` are the methods this file's rows in `sys__LANGUAGE_CONTRACT__OBJECTS` name. Nobody
 * calls either directly — the teardown's generated switch does, which is why they are package and not
 * public: a caller handed one would decrement, push onto a chain nobody drains, and return with every
 * count balancing.
 *
 * The five chunk verbs — offset, push, transfer, pop and pop_deferring_disposal — are the tier the holder
 * layer above is written out of, and the chunk arithmetic is tested through them.
 *
 * What the OBJECT TEARDOWN in `heap_object__impl.cuh` reaches for is not those: it is the HOLDER-level
 * pair, `sys__stack__zzpackage_empty_at` and `sys__stack__zzpackage_pop_deferring_at`, which take an
 * offset because a worklist handle has one and not a node. The unwinding borrows a stack, whose LIFO
 * shape is what a worklist wants, rather than growing a second kind of list to do the same thing.
 *
 * ⛳ AND THE FILE'S ELEVEN `zzprivate_` VERBS ARE NOT HERE, WHICH IS THE MARKER DOING ITS JOB — the
 * chunk's cursor and previous with a setter each, peek, init, create and replace, and the holder's three
 * field accessors, are reached only from inside the implementation, and `zzprivate` means one FILE. A
 * header that cannot name them is the shape of the rule rather than an omission from it.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ __noinline__ void sys__stack__zzpackage_release_internal(sys__heap_node* head,
                                                                    uint64_t* releaser_stack);
static __device__ __noinline__ void sys__stack_chunk__zzpackage_release_internal(sys__heap_node* head,
                                                                          uint64_t* releaser_stack);

static __device__ inline uint64_t       sys__stack_chunk__zzpackage_offset(const sys__stack_chunk* chunk);
static __device__ inline sys__stack_chunk* sys__stack_chunk__zzpackage_push(sys__stack_chunk* chunk,
                                                                         const sys__heap_node* value);
/* The same store without the hold — what `push` is built out of, and what a dying container moves its
 * contents through. */
static __device__ inline sys__stack_chunk* sys__stack_chunk__zzpackage_transfer(sys__stack_chunk* chunk,
                                                                            const sys__heap_node* value);
static __device__ inline sys__stack_chunk* sys__stack_chunk__zzpackage_pop(sys__stack_chunk* chunk,
                                                                        sys__heap_node* out_value);
/* The same, handing the emptied chunk back rather than giving it up — see the holder-level pair above
 * for whose problem that solves. */
static __device__ inline sys__stack_chunk* sys__stack_chunk__zzpackage_pop_deferring_disposal(
                                                                        sys__stack_chunk* chunk,
                                                                        sys__heap_node* out_value,
                                                                        uint64_t* out_spent);

#endif /* SILVANN__PACKAGES_SYS_CPU_STACK__HEADER_CUH */
