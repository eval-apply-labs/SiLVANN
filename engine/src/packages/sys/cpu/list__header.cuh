#ifndef SILVANN__PACKAGES_SYS_CPU_LIST__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_LIST__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../contracts/objects/list.cuh" /* its constants, fault words and layouts */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * Nothing in it is needed to use a list. It holds what compares this file against the tree it grew out
 * of, so everything below can be read by somebody who has never seen that tree.
 *
 * WHAT IT CORRESPONDS TO THERE: a spine of small nodes, each naming a page of values it does not own,
 * with a caller-supplied page and a caller-supplied node handed to every mutator. The handle is three
 * loose words passed by value. Nothing counts anything. Four differences, each a decision:
 *   · THE HANDLE IS THE OBJECT. There, a view is three words a caller carries and the storage belongs to
 *     nobody. Here the view is the only thing that exists to be held, and the storage dies with the last
 *     one — which is what makes a list a thing you can put in a variable rather than a thing you have to
 *     remember to clean up.
 *   · THE SPINE NODE IS GONE. Its cursor and link live in the chunk's own allocation head, so a chunk is
 *     one object instead of two — the same arrangement the stack's chunks already have.
 *   · A PAGE MAY BE REWRITTEN. There, `page` carries "NEVER MOVES" because `car` hands out a pointer
 *     INTO it. Nothing here hands out an address, so a removal compacts and a position is arithmetic.
 *   · A POSITION IS MAINTAINED, NOT PINNED. Every insert and remove adjusts every live view, so a view
 *     keeps naming the elements it named. There, nothing had to, because nothing could move.
 *
 * WHAT CHANGED ON THIS BRANCH, AND WHY THE COMMENTS BELOW SAY WHAT THEY SAY:
 *   · THE CYCLE PARAGRAPH once read *"a cycle is 108 registers on this hardware"*. That was measured
 *     wrong, not merely stale: removing the one real cycle moved VGPR 128 -> 128, spill got WORSE
 *     (`k_eval` 96 -> 186, SGPR ~+300 on three kernels), rate +2.6%, `uses_dynamic_stack` true -> false
 *     everywhere. `claim_rules/list_call_graph_acyclic.py` keys on "NOTHING HERE CALLS ITSELF" — keep it.
 *     The spill figures are `k_eval`'s, and the evaluator has left the card.
 *   · THE CLONE ADAPTER exists because all 19 object rows in both packages named the refusing default
 *     while the deep copy sat here built and tested; `sys__clone` was also the one published verb no
 *     test had called. ⚖ *"wire deep_clone into the list row and test it."* Do not unwire it.
 *   · The picture paragraph's rate figures are `k_eval` figures; the conclusion may survive on a CPU,
 *     the number will not reproduce.
 *   · `deep_clone`'s paragraph carried the same removal as a `MEASURED` fact — VGPR unchanged, spill no
 *     better, 2.6% faster, `uses_dynamic_stack` false on every kernel. All `k_eval`; moved here.
 *   · `create_executable` once consumed its cells where `append` did not; the ruling made them agree.
 *   · `nth` was once advertised as what a walk should use; it re-walks from the head every call, which
 *     is why `sys__sublist__walk` exists.
 *   · `contracts/objects/list.cuh`: `TOTAL` and `TAIL` were derived (the sum was
 *     `MEASURED` at 78 call sites). The invariant census first said FIVE sites; the suite caught the
 *     sixth (the one-value removal) when NINE rows went red — *"expect 3, got 4"*.
 *
 * ⛳ THE ONE THING TO KNOW BEFORE EDITING: the adjust walk is not an optimisation and it is not optional.
 * A mutation that skips it leaves every view naming the wrong element, silently, and no counter moves.
 * ⛳ RETIREMENT: this block goes when nothing in `src` reaches into the old list.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ a list — the values, and the views that name where in them you are ══════════════════════════════
 *
 * A list holds values in order and lets you say where in it you are. Both halves are objects, and only
 * one of them is ever handed out.
 *
 * ── TWO OBJECTS, AND A CALLER ONLY EVER HAS THE SECOND ──────────────────────────────────────────────
 *     THE STORE    a chain of chunks holding the values, plus the roll of who is looking. Nobody outside
 *                  ever names it, so it has no way of being held wrongly and no surface to get wrong.
 *     A SUBLIST    the store and a position in it. It is what `create` answers, what every verb below
 *                  takes, and the only thing a program can put anywhere.
 * ⭐ SO "THE LIST" IN ORDINARY SPEECH IS A SUBLIST AT POSITION ZERO, and there is no second type for it —
 * the whole list and a part of it are the same kind of thing, which is what makes `cdr` cost nothing to
 * express: it is another sublist, one further along, over the same store.
 * ⭐ AND THE STORE DIES WITH THE LAST SUBLIST. Its count IS the number of views naming it, so "nobody is
 * using this list" is a count reaching zero and the ordinary teardown does the rest. Nothing here has a
 * lifetime rule of its own.
 *
 * ── A POSITION IS A DISTANCE, NEVER AN ADDRESS, AND EVERY MUTATION MAINTAINS IT ─────────────────────
 * A sublist holds how far along it starts, and a mutation moves every view that needs moving:
 *
 *   putting one in WHERE ONE WAS   a view starting AT i or beyond starts one later
 *   putting one in PAST THE END    nothing moves at all
 *   taking one out                 a view starting BEYOND i starts one earlier
 *   all three                      a view starting below i does not move
 *
 * ⭐ ONE PRINCIPLE, AND THE THREE LINES ARE IT APPLIED RATHER THAN THREE RULES: A VIEW KEEPS NAMING THE
 * VALUE IT NAMED.
 *   · Putting one where a value already stood pushes THAT value along, and every view standing on it or
 *     beyond follows it. So nobody's list quietly gains a value at its front.
 *   · Putting one past the end displaces nothing, because nothing was there — so no view was naming it
 *     and none has to move. A view that had reached the end is the rest of the list, and the rest of a
 *     list that just grew is the new value.
 *   · Taking one out leaves the view standing on it with nothing to follow, so it stays where it is —
 *     which is already the next value — and only the ones beyond it move.
 * ⛔ THE MIDDLE LINE IS NOT A SPECIAL CASE, IT IS THE PRINCIPLE APPLIED WHERE THERE WAS NOTHING, and
 * getting it wrong is not subtle: treat an append as an insert and the appender's own view is pushed
 * along with everything else, so a list can never be built at all.
 * ⇒ ⛳ SO AN INSERT IS INVISIBLE TO EVERY VIEW THAT EXISTS, ITS OWN CALLER'S INCLUDED, and the one that
 * asked for it is handed a new view of what it put in — which is `cons`, and is why `insert` answers a
 * view where `append` answers whether it went in.
 * ⛔ THE WALK IS THE MUTATION. There is no version of an insert that skips it: a view left unadjusted
 * names a different element than it did a moment ago, with nothing failing and no count moving.
 *
 * ── ONE OWNER AT A TIME. SHARING IS BY COPY, NOT BY VISITING ────────────────────────────────────────
 * ⚖ ARCHITECT: *"the list can be shared but really shouldn't unless needed ... if you want to share it
 * cloning it into a plain array is most likely better, or you could clone it as a fresh list too. it
 * [the reading of a shared list] is genuinely concurrent access."*
 * A mutation writes values, positions and links at once, so anything reading during one sees a structure
 * partway between two shapes. That is not a thing a lock makes cheap: a reader would have to take it too,
 * and then readers exclude each other. ⇒ **A list belongs to one owner, and somebody else gets a COPY** —
 * a fresh list to change as they like, or a flat read-only table if all they will do is look.
 * ⛳ AND THE SECOND ONE IS THE FAST PATH, not a fallback: nothing is ever written to it, so any number of
 * readers may look at once with nothing between them.
 *
 * ── READING BORROWS ─────────────────────────────────────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"the hold is done at the head level and the chunks are borrowing that hold."*
 * A value read out of a list comes back WITHOUT a hold and the caller owes nothing. That is sound because
 * the standing is already there — the chunk holds the value, the chain holds the chunk, and the caller
 * holds a sublist naming the chain. A read that took its own hold would pay two atomics for standing it
 * already has, on the counter every other reader of that element is also touching.
 * ⛔ AND THE BOUNDARY, WHICH IS THE ONE THING TO GET RIGHT: A BORROWED VALUE IS GOOD FOR AS LONG AS IT IS
 * IN THE LIST. Taking it OUT is what ends it — so a caller that means to keep something past its removal
 * takes a hold of its own first, unless it is the one doing the taking. `extract` is built for that case:
 * the hold the list had MOVES to the caller rather than being let go, so whoever extracted something walks
 * away owning it.
 * ⛔ AND `discard` IS THE OTHER WAY ROUND, WHICH IS WHY THE TWO ARE NAMED APART: it RELEASES every value
 * it takes out, so a borrow of something a list then discards ends with nothing holding it.
 *
 * ── WHAT NOBODY GETS, AND IT IS WHY EVERYTHING ABOVE WORKS ──────────────────────────────────────────
 * ⚖ ARCHITECT: *"by handing out only objects and not internal references we can modify our internals in
 * peace."* No chunk, no slot and no address ever leaves this file. What a caller can hold is a sublist
 * and a value — so pages may be rewritten, chunks split, merged and recycled, and nothing outside can
 * tell. ⛔ IT IS ALSO THE ONE RULE THAT BREAKS SILENTLY: a verb answering "what position is this at" would
 * put a number in somebody's hand that the next mutation makes wrong.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── MAKING ONE, AND MAKING A SECOND ONE TO GIVE AWAY ────────────────────────────────────────────────
 * The three below are not three constructors. One stands up a list; the other two are the two ways of
 * handing what it holds to somebody else, and they differ in what the recipient may then do.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* An empty list, and a view over the whole of it. Nothing is carved for values until the first one is put
 * in — but the view goes on the roll as it is made, and the first entry carves the roll's first chunk, so
 * an empty list costs one store, one sublist and one chunk of `SYS__LIST_CHUNK__ALLOC` nodes, with no room
 * for contents.
 * ⛳ SO WHAT AN EMPTY LIST COSTS IS THE ROLL, not the values — the number to have in mind where this file
 * says lists are made and thrown away by the million, because the dial above sizes that chunk too. */
static __device__ inline uint64_t sys__list__create(void);

/* A list of these cells, in this order, in one call. ⚖ ARCHITECT: *"maybe we need to expose a c method
 * called sys__list__create_executable ... which gives us back a sys__heap_node so it can be used in
 * another sublist."*
 * ⭐⭐ SO A PROGRAM IS BUILT A LIST AT A TIME RATHER THAN A CELL AT A TIME. Everything outside this engine
 * reaches it through one door and every crossing of that door costs the same whatever it carries, so what
 * decides the price of building a program is HOW MANY crossings it takes. A form is one, whatever its
 * width; a caller nests them by putting the answer to one into the cells of the next.
 *
 * ⛔⛔ AND IT TAKES NOTHING FROM THE CALLER, WHICH IS THE SAME CONTRACT `append` HAS AND IS THE POINT.
 * A cell naming an object arrives holding it; this takes a hold OF ITS OWN and the caller keeps the one it
 * came with, so a caller building a program bottom-up gives back each piece as it goes and the outermost
 * list holds everything under it. ⚖ ARCHITECT: *"make them agree, create_executable should not consume."*
 * ⇒ ★ TWO BUILDERS THAT TAKE CELLS AND BUILD A LIST MEAN THE SAME THING BY IT. They read alike at a
 *   call site and they behave alike, which is what stops the difference from being something a caller has
 *   to remember per verb — and remembering it wrongly is quiet: the answer stays right and the object is
 *   freed underneath.
 * ⛳ AND A REFUSAL TAKES NOTHING EITHER: the part-built list dies, which undoes exactly what this call
 * did, and the caller owns every cell it handed over. There is never "some of them".
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline uint64_t sys__list__create_executable(const sys__heap_node* cells, uint64_t count);

/* A list of its own with the same values, for somebody who will change theirs. Every value that names an
 * object is held once more, because the copy is a second owner of it — the values are shared, the
 * STRUCTURE is not, so neither list can be reached through the other.
 * ⛳ IT IS CALLED BY NAME AND NOT THROUGH THE ROW. The SUBLIST row's clone column names
 * `sys__list__zzpackage_clone_object`, which reaches `deep_clone` below; this shallow copier is what
 * `deep_clone` calls, once for each list it meets. */
static __device__ inline uint64_t sys__sublist__clone(uint64_t sublist);

/* A list of its own with the same values AND ITS OWN STRUCTURE, all the way down. Where `clone` shares
 * the sublists its cells name, this copies them too, so nothing in the answer can be reached through the
 * original and changing either one leaves the other exactly as it was.
 *
 * ⭐⭐ ONE PASS, AND WHAT IS BEING BUILT IS ALSO THE QUEUE. Two lists run alongside each other — the
 * sources met, and the copy made for each — and the cursor walks the second one WHILE IT IS STILL
 * GROWING. A list met for the first time is copied shallow and put on the end, so the cell that named it
 * can be pointed at the copy the moment it is read: nothing is collected first and nothing is revisited.
 * ⭐ NOTHING HERE CALLS ITSELF. A copy that descends into what it finds is a cycle in the call graph,
 * and its depth is the depth of the structure it copies. An unbounded recursion in a thaw is worth not
 * having wherever it runs.
 * ⭐ SHARING AND CYCLES NEED NO CASES OF THEIR OWN. Sources are matched BY ADDRESS, so a list named twice
 * is copied ONCE and both cells end up naming the one copy, and a list that reaches itself finds itself
 * already noted and points at its own copy.
 *
 * ⛔ WHAT IT DESCENDS INTO IS NOT THIS FILE'S TO DECIDE: a cell is followed when its tag says the thing it
 * names is part of the structure and this list says the thing it names is a list. The first half is
 * `sys__heap_node__is_substructure` and lives beside the accounting question it is so easily confused
 * with; a tag that names a shared value is left exactly where it is.
 * ⛳ THE LOOKUP IS A SCAN, DELIBERATELY. It is quadratic in the number of lists and the structures that
 * come through here hold a handful — the keyed store that makes it linear is the dictionary
 * (`src/docs/design/dictionary.md`), which this is the second named user of, and it is worth building when a program is long enough to
 * notice rather than to speed up a scan of eight. */
static __device__ inline uint64_t sys__sublist__deep_clone(uint64_t sublist);

/* ⭐⭐ THE LIST'S CLONE COLUMN — the adapter that lets the `sys__clone` VERB reach the deep copy.
 *
 * ⚖ *"sys_clone is there to provide a way to objects like vectors or lists or arrays to be cloned and
 *   handed over, so you can use a local copy"* — so a list answers `clone` with the deep copy above.
 *
 * ⛳ WHY IT IS AN ADAPTER AND NOT THE FUNCTION ITSELF: the column is called as `CLONE(head, base)` and
 * the deep copy takes an OFFSET, because a list is named by where its fields start and not by its
 * allocation head. One `sys__heap__offset(head + 1)` is the whole of the difference.
 * ⛔ AND THE BASE IS UNUSED HERE ON PURPOSE. `sys__list__create()` carves from the caller's own arena,
 * so the copy lands where the CALLER is — which is what "hand it over so you can use a local copy"
 * needs. A column that insisted on passing the base would put the copy somewhere else. */
static __device__ inline uint64_t sys__list__zzpackage_clone_object(sys__heap_node* head,
                                                                     sys__heap_node* computing_base);

/* A flat table of the values, from this view to the end, that anybody may read and nobody may write. It
 * is what to hand a reader in another block: nothing is ever written to it, so any number of them may
 * look at once with nothing coordinating them and nothing to lock.
 * ⛳ IT IS A PICTURE AND NOT A WINDOW — what the list does afterwards does not appear in it, which is
 * exactly what makes it safe to read from anywhere. */
static __device__ inline uint64_t sys__sublist__create_viewonly_array(uint64_t sublist);

/* ── A PICTURE, AND BACK — THE TWO-WAY STREET BETWEEN A LIST AND A PLAIN ARRAY ──────────────────────
 * ⚖ ARCHITECT: *"a list doing a deep copy to plain array and a plain array doing a deep copy into a
 * list, as a two way street."* The picture above is flat; these two are deep. A list becomes an array
 * and every list it holds becomes an array too; coming back, every array that stood for a list becomes a
 * list again.
 *
 * ⭐⭐ WHAT IT IS FOR: SOMETHING MANY READERS COPY AT ONCE. A procedure's body runs by being copied, and a
 * body kept as a list is read through a walk, copied through two maps, and has a hold taken and given back
 * on every list inside it by every caller — so sixty blocks copying one body queue on the same counts. A
 * picture is read by addition, and the copy back takes no hold on it at all: whoever holds the picture
 * holds everything below it. ⚖ *"by not being mutable they require no lock so you could give it to all
 * 60 cus and each cu would do their local deep copy without conflicts."*
 *
 * ── WHAT CHANGES SHAPE ON THE WAY, AND WHAT DOES NOT ────────────────────────────────────────────────
 *     a list a cell names       becomes an array, named by the same tag — so a quoted list comes back
 *                               quoted and one to run comes back to run
 *     an array the list held    is the same array, and inside the picture it is named QUOTED_ARRAY —
 *                               ⚖ *"by tagging original arrays as quoted arrays you can restore them
 *                               properly"* — so the way back does not mistake it for a list
 *     an empty list             becomes nil. A plain array cannot be empty, and nil is what an empty list
 *                               is; it comes back as nil
 *     anything else             is the same value, held once more by whoever holds the copy
 * ⛔ A PICTURE IS A TREE. A list reached twice is refused rather than pictured: the way back keeps no map
 * of where it has been — that is what makes it cheap enough to run on every call — so it would make two
 * lists of one. A list that reaches itself is the same refusal.
 * ⛔ AND A QUOTED LIST NAMING AN ARRAY IS REFUSED, because on the way back it would become a list.
 * ⛔ ONE FORM FITS IN ONE CHUNK. An array is one allocation and an allocation does not span chunks, so a
 * list longer than a chunk holds has no picture; the carve refuses it and says so.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* The picture: a plain array of this view's values, every list among them pictured too. Zero, having
 * raised, for a list reached twice, a list with nothing in it, or one too long for a chunk. */
static __device__ inline uint64_t sys__sublist__deep_copy_to_node_array(uint64_t sublist);

/* And back: a list of its own with the picture's values, every array that stood for a list made a list
 * again. It reads the picture without taking a hold on anything in it, so the caller must be holding the
 * picture — and that is what makes it safe for any number of callers at once. */
static __device__ inline uint64_t sys__list__deep_copy_from_node_array(uint64_t array_base);

/* The same, for a copy the EVALUATOR will run. ⭐ A sub-form may come back as a `FROZEN_LIST` naming the
 * array it would be thawed from, and `eng__eval` makes it a list at the moment it descends — so a form
 * that is never reached, the untaken arm of an `if` among them, is never built at all.
 * ⛔ ONLY THE EVALUATOR MAY BE HANDED ONE. Nothing else in the tree has an arm for that tag. */
static __device__ inline uint64_t sys__list__thaw_running(uint64_t array_base);

/* The same, into a list the caller already has — which is how a RECYCLED list is refilled without
 * being made. ⛔ The list must be EMPTY; nothing here clears one. Answers false having raised. */
static __device__ inline bool sys__list__thaw_running_into(uint64_t array_base, uint64_t list);

/* ── WHAT A VIEW IS ──────────────────────────────────────────────────────────────────────────────────
 * Three questions that take nothing and change nothing.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* Whether this names a sublist at all. Asked by every verb below before it reaches, and worth having on
 * its own for a caller that was handed a reference and has no reason to trust it. */
static __device__ inline bool sys__sublist__is(uint64_t sublist);

/* How many values there are from here to the end: the store's maintained total less this view's start,
 * so it costs a read and not a walk. The view carries no length of its own — one would have to be
 * corrected by every mutation that a position already is. */
static __device__ inline uint64_t sys__sublist__length(uint64_t sublist);

/* Whether there are none — the cheap end test, which is what a walk asks between every two steps. It does
 * not count; it asks whether this view reaches anything at all. */
static __device__ inline bool sys__sublist__empty(uint64_t sublist);

/* ── READING ─────────────────────────────────────────────────────────────────────────────────────────
 * Values come back BORROWED: whole nodes, by value, with no hold and nothing owed. `car` and `nth` cost
 * an addition and a load once the chunk is found, and neither allocates.
 * ⛔⛔ AND THE THREE WORDS "once the chunk is found" ARE THE WHOLE COST. Finding the chunk starts at the
 * head of the chain EVERY TIME, so `nth(list, k)` for `k = 0..n` walks `n²/2/SLOTS` chunk links to read
 * `n` values that are sitting next to each other. A loop over `nth` trades `cdr`'s allocation per step
 * for a traversal per step — which is worse, and silently.
 * ⇒ ⛳ **A SEQUENTIAL READER WANTS `sys__sublist__walk` BELOW.** `nth` is for RANDOM access, which is
 * what its name says and what it is good at.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* The value this view starts at. An empty view answers nothing, without raising — reaching the end is how
 * a walk finds out it is over, not a mistake somebody made. */
static __device__ inline sys__heap_node sys__sublist__car(uint64_t sublist);

/* The value `n` along from here. A position this view does not reach answers nothing, having raised: a
 * walk asks `empty` and an index is a claim about how far the list goes. */
static __device__ inline sys__heap_node sys__sublist__nth(uint64_t sublist, uint64_t n);

/* Open a walk on this view, standing `from` values along it. Answers false only when the thing handed in
 * is not a sublist, having raised — a `from` past the end is not a mistake, it is an empty walk, and the
 * first `walk_cell` answers nothing exactly as `car` does. */
static __device__ inline bool sys__sublist__walk(uint64_t sublist, uint64_t from, sys__list_walk* w);

/* The cell the walk stands on, or 0 at the end — and it is a POINTER rather than a copy, which is what
 * lets a caller that reads and then writes the same cell pay for finding it ONCE. Writing through it is
 * the caller's own business and wants `sys__heap_object__set`, which is what `replace` uses, so the hold
 * that was there is let go and the one going in is taken. */
static __device__ inline sys__heap_node* sys__list__walk_cell(sys__list_walk* w);

/* One further along. Stepping past the end is allowed and leaves the walk ended, so a loop may step
 * unconditionally and ask `walk_cell` whether there was anything there. */
static __device__ inline void sys__list__walk_step(sys__list_walk* w);

/* The rest: another view over the same store, starting one further along. It allocates, because it is an
 * object a program can hold and put somewhere — and it is registered, so what it names keeps meaning the
 * same thing when the list changes underneath it.
 * ⛳ `cdr` of a view is another view over the STORE, never a view of a view. A walk of a thousand steps
 * leaves a thousand views over one list, not a chain a thousand deep. */
static __device__ inline uint64_t sys__sublist__cdr(uint64_t sublist);

/* ── CHANGING IT ─────────────────────────────────────────────────────────────────────────────────────
 * Five verbs, and none of them may be called on a list somebody else is reading.
 * ⭐ FOUR OF THEM ADJUST EVERY LIVE VIEW BEFORE THEY ANSWER, AND `replace` IS THE ONE THAT DOES NOT,
 * BECAUSE NOTHING MOVES: no value changes position, so no view names anything different from what it
 * named and there is no walk to make.
 * ⭐ AND EACH ONE SAYS WHAT BECOMES OF THE VALUE. `insert`, `append` and `replace` take a hold of what
 * goes in; `extract` hands over what comes out; `discard` lets go of everything it takes out. ⛔ THE LAST
 * TWO ARE WHY THEY ARE NAMED APART: treating what a discard took as something the caller now owns is a use
 * of a value nothing is holding.
 *
 * ⚖ AND WHEN A CHUNK IS FULL AN INSERT SPLITS IT EVENLY (architect): half the values stay, half move to a
 * fresh chunk, and the new value goes into whichever half it belongs in. ⛳ Down the middle rather than at
 * the insertion point because it leaves room on BOTH sides, so a second insert near the first does not
 * split again — and it is a policy rather than a law: nothing else depends on where the cut falls.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* Put a value `n` along from here, and answer A VIEW THAT BEGINS AT IT. The list takes a hold of what goes
 * in. A position past the end is a caller counting wrong and raises; at the end is how a list grows.
 *
 * ⭐ THE ANSWER IS THE POINT, AND WITHOUT IT THE VERB WOULD BE UNUSABLE. Everything standing where the
 * value went, the caller's own view included, moves along to keep naming what it was naming — so after
 * an insert nobody is looking at the new value, and nothing hands out a view at a lower position to go
 * back for it. What comes back is that view.
 * ⇒ ⭐ WHICH MAKES INSERTING AT THE FRONT `cons`: what the verb answers is what the caller puts back where
 * the list was, and ⚖ ARCHITECT — *"this way you can set! it back if you want to circulate the new value
 * from a let binding."* */
static __device__ inline uint64_t sys__sublist__insert(uint64_t sublist, uint64_t n, const sys__heap_node* value);

/* Put a value after the last one, which is the common case and needs no counting. */
static __device__ inline bool sys__sublist__append(uint64_t sublist, const sys__heap_node* value);

/* Put a different value in the cell `n` along from here. What was there gives up its hold and what goes
 * in takes one. Answers whether it went in; a position this view does not reach raises.
 * ⭐ NOTHING MOVES, WHICH IS THE WHOLE REASON THIS EXISTS SEPARATELY FROM TAKING ONE OUT AND PUTTING ONE
 * BACK. No position changes, so no view is adjusted and nothing has to be handed back — where an insert
 * at a view's own start is invisible to it by design, a replacement is not, because the view is standing
 * in the same place afterwards. It is how a value is written where a form was without disturbing anybody
 * looking at it. */
static __device__ inline bool sys__sublist__replace(uint64_t sublist, uint64_t n, const sys__heap_node* value);

/* Take `count` values out from `n` along and LET THEM GO. Answers how many went, which is `count` unless
 * the view does not reach that far, and that raises.
 *
 * ⭐ IT IS NAMED FOR THE OWNERSHIP AND NOT FOR THE ARITY, which is the whole reason it is a second verb
 * rather than an argument on the first. ⚖ ARCHITECT: *"remove would be better renamed as extract, because
 * otherwise you might think it also discards, and extract tells you that you still have it in your hand."*
 * ⇒ EXTRACT HANDS YOU SOMETHING; DISCARD MEANS IT IS GONE. A range cannot hand back what it took — there
 * is no one value to answer with — so it releases, and the name says so before a caller has to wonder.
 *
 * ⛳ AND THE POINT OF IT IS THAT A RANGE IS ONE REWRITE. Taking values out one at a time re-derives the
 * length and re-walks to the chunk for every one of them, and a form emptied a cell at a time pays that
 * per cell. `MEASURED` S90: emptying forms one value at a time was 2,586 extractions across fib(0..7)
 * where the same work is ~700 discards.
 * ⛳ Views are moved ONCE, and the rule generalises rather than gaining a case: a view beyond the range
 * follows it back by however many went, and a view INSIDE it stops where the range began — which is what
 * a single removal already does to a view standing on the value it took. */
static __device__ inline uint64_t sys__sublist__discard(uint64_t sublist, uint64_t n, uint64_t count);

/* Take the value `n` along from here out, and answer it. The hold the list had MOVES to the caller — so
 * nothing is taken and nothing given up, and the caller walks away owning what it removed rather than
 * holding something that was let go a line earlier. A position this view does not reach answers nothing,
 * having raised. */
static __device__ inline sys__heap_node sys__sublist__extract(uint64_t sublist, uint64_t n);

/* ── THE TWO ROW METHODS ─────────────────────────────────────────────────────────────────────────────
 * What each kind's row names, and none is called by hand: one runs when the last hold goes, the other
 * when a program asks for one. They are declared here because the rows that name them are read in a file
 * that has never heard of this one.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* What a sublist was holding when it died: its hold on the store, moved onto the chain the release loop
 * drains. ⛔ AND IT TAKES ITSELF OFF THE ROLL FIRST, which is the one thing here that is not ordinary —
 * an entry naming a view that has gone would be adjusted forever by mutations nobody is watching, and
 * would keep a freed position in the count of who is looking. Removing it is safe without a lock for the
 * reason the whole design rests on: one owner, so nothing is walking the roll at that moment. */
static __device__ __noinline__ void sys__sublist__zzpackage_release_internal(sys__heap_node* head,
                                                                       uint64_t* releaser_stack);

/* What a store was holding: its chunk chain and its roll. One move each — a chain is one object however
 * many chunks are behind it, and each chunk's own row deals with what is inside it when its turn comes. */
static __device__ __noinline__ void sys__list__zzpackage_release_internal(sys__heap_node* head,
                                                                    uint64_t* releaser_stack);

/* What a chunk was holding: every value standing in it, and the chunk after it. */
static __device__ __noinline__ void sys__list_chunk__zzpackage_release_internal(sys__heap_node* head,
                                                                          uint64_t* releaser_stack);

/* Building one from a program. It takes no arguments — a list arrives empty and is filled by the verbs
 * above — so it is expressible with what a single node carries and needs nothing that does not exist. */
static __device__ inline sys__heap_node sys__list__zzpackage_construct(sys__heap_node* base,
                                                                   const sys__heap_node* parameters);

#endif /* SILVANN__PACKAGES_SYS_CPU_LIST__HEADER_CUH */
