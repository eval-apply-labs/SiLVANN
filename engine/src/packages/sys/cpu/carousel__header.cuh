#ifndef SILVANN__PACKAGES_SYS_CPU_CAROUSEL__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_CAROUSEL__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/defaults.cuh"
#include "../contracts/objects/carousel.cuh" /* its constants, fault words and layouts */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * Nothing in it is needed to USE a carousel or to change one; it is here because the three things below were each argued the wrong way
 * first, and the corrections are worth more to whoever writes the NEXT contract than to anyone reading
 * this one. ⚖ ARCHITECT: *"mostly i removed retractions or whatever a user looking for the contract
 * would not be interested in."* Right — so they live here instead of in the prose below.
 *
 * ① THE STALEST PART OF A CONTRACT IS THE PART THAT PROMISES ABSENCE.
 *   "WHAT IT DOES NOT DO" read *"it does not allocate, does not free, does not zero anything on the way
 *   out — there is no way out"* for two commits AFTER `create` and `dispose` landed and made every clause
 *   of it false. It survived because nothing rereads a negative claim: a list of what a thing does not do
 *   has no caller, no test and no compiler to notice it aged. Watch for this in the headers still to be
 *   split.
 *
 * ② THE CONTRACT WAS WORTH WRITING FOR WHAT IT REFUTED, NOT FOR WHAT IT RECORDED.
 *   The "loud refusal" clause claimed *"exactly one silent no"* and was FALSE of the code when written —
 *   `add` and `take` answered a null ring with a bare `false`. Writing the sentence is what found them.
 *   A contract is a claim ABOUT the code, so it can be wrong about it, and that is the whole value.
 *
 * ③ A JUSTIFICATION WRITTEN FOR A DECISION ALREADY TAKEN READS AS SETTLED BECAUSE THE DECISION WAS.
 *   The size argument first claimed that rounding STRICTLY past a power of two is what makes
 *   `head == tail` mean empty. It is not, and never was — `add`'s ceiling does that at any size. The
 *   correct reading is below: the power of two is the reason, and the surplus is headroom. Marking it
 *   `MEASURED` would not have caught this; it was reasoning, and reasoning is where this project slips.
 *
 * ⛳ AND ONE THING THAT IS NOT A RETRACTION, KEPT HERE SO IT IS NOT MISTAKEN FOR ONE: the width
 *   parameter's byte-saving argument was priced and REFUTED — 768 B against a 15 MB pool. That refutation
 *   IS live justification and stays in the prose below, because it is why the parameter is a CAPACITY
 *   rather than an optimisation.

 * ⛳ THE BITMASK THIS RING REPLACED, kept for the comparison. It walked the pool from an origin fixed per
 *   block, so "which chunk is free" answered "the first one after the same place, every time".
 *   `MEASURED` on the fixture: claim/free ten times gave index 1 ten times; ten claims without freeing
 *   gave 1..10. Finding a free chunk was O(n) in the BUSY ones. It had NO shared state — every chunk its
 *   own word — so two blocks contended only when reaching for the same chunk.
 *
 * RETIREMENT: delete when the old tree is gone. Everything above compares this ring against the
 * bitmask it replaced, and that comparison has no subject once the bitmask is not there to compare to.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ══ THE CAROUSEL — A RING OF NUMBERS, AND WHAT IS FREE IS ONE ══════════════════════════════════════
 *
 * A ring of numbers with two cursors. Things are added at the head and taken from the tail, both cursors
 * only ever rise, and both are masked to the ring's size when they are used to reach a slot. That is the
 * whole mechanism, and it is deliberately not about chunks: it holds `uint32_t`, so what a number MEANS
 * belongs to whoever fills it.
 *
 * ⛔⛔ IT IS C ONLY, AND THAT IS A RULING WITH A REASON THAT IS NOT PERFORMANCE. 
 * The reason is that Lisp would use an ast-shaped queue, holding ast objects over array chunks.
 * We cannot allocate an array of arbitrary size at runtime, without doing it outside of the constraint
 * of the heap, therefore this is just a support object for C devs
 *
 * ── HOW YOU GET ONE ─────────────────────────────────────────────────────────────────────────────────
 *
 *     sys__carousel* c = sys__carousel__create(slots, SYS__CAROUSEL__U8);   the room comes with it
 *     sys__carousel__dispose(c);                                          and goes back with it
 *
 * ⭐ This is DISPOSED, NOT RELEASED, AND THERE IS NO RELEASE ARM TO RAISE — because there is no KIND. A
 * carousel is a plain C structure and not a heap object: it has no row in the language contract, no
 * dtype, and nothing the counting could reach. The lisp teardown cannot arrive at one by any path, so
 * the case never needs refusing. ⇒ ★ THAT IS THE STRONGER VERSION OF THE SAME GUARANTEE — a refusal is
 * something somebody has to have written and can stop being written; not being a kind is a property of
 * the type. ⛳ It is also why `create` can answer before the heap exists: the room comes from the seam.
 *
 * ── AND THE WIDTH IS AN INPUT TO THE ALLOCATION, NOT A SAVING ────────────────────────────────────────
 *
 * ⭐ THE REASON `create` NEEDS A TYPE IS THAT IT ALLOCATES. It cannot ask for room without knowing how much,
 * and how much is `slots x width`. The width is not there to shave bytes — that argument was priced and
 * REFUTED: a `uint8` ring saves 768 B against a 15 MB pool, which is 0.005% and worth nothing.
 *
 * ⭐⭐ WHAT IT ACTUALLY BUYS IS THE FOOTPRINT, AND IT BUYS IT BY A FACTOR OF FOUR. `create` asks the seam
 * for `sizeof(sys__carousel) + slots x width` bytes and for nothing else, so the same slot count costs four
 * times as much at U32 as it does at U8. That expression in `carousel__impl.cuh` is the whole of the
 * arithmetic — re-derive it there rather than trusting a table of slot counts here.
 *
 * ⛔ AND THERE IS NO PER-WIDTH SLOT LIMIT TO QUOTE. The room comes from the platform, so the only sizes
 * `create` refuses are one that is not a power of two (GEOMETRY), one whose BYTE COUNT would wrap 32 bits
 * (TOO_BIG), and one the platform has no room for (NO_ROOM). `MEASURED` by the suite, which makes a
 * 4096-slot U32 ring and checks it raised nothing. The ceiling the width moves is therefore the MACHINE'S,
 * and it arrives as a NO_ROOM in front of a caller rather than as a constant a build decided.
 *
 * ⇒ ★ THE SAME PARAMETER READ TWO WAYS: as an optimisation it is not worth having, and as a capacity it is
 * the difference between a queue that fits and one that is refused. The second is why it is here.
 *
 * ⛳ THE INTERFACE VALUE STAYS `uint32_t` WHATEVER THE STORAGE IS. A caller adds and takes numbers; the
 * width is the carousel's business. A value too wide for the storage is REFUSED with its own fault rather
 * than truncated, because a truncated chunk index is a valid-looking index to somewhere else.
 *
 * ── THE CONTRACT ────────────────────────────────────────────────────────────────────────────────────
 * It is `create` · `dispose` · `add` · `take` · `count`, and nothing else. The declarations at the foot
 * of this file are those five and no others — there is no verb here that is not a caller's business.
 *
 * WHAT THE CALLER OWES
 *   a width         one of the three. CHECKED.
 *   a size          a power of two, so the wrap is a mask, and a byte count that fits 32 bits —
 *                   `sizeof(sys__carousel) + slots x width`. BOTH CHECKED, as GEOMETRY and as TOO_BIG, and
 *                   REFUSED rather than rounded down, because a caller who asked for a thousand and
 *                   received two hundred meets the shortfall as a FULL refusal much later, somewhere else.
 *   big enough      `slots >= (the most that can ever be waiting) + 1`. NOT CHECKED, and cannot be: the
 *                   carousel does not know how many things its caller owns. `add` catches a caller who
 *                   got it wrong, after the fact.
 *   never a zero    zero is what an unwritten slot says, so it cannot also be a member. CHECKED.
 *   in range        a value the storage width can hold. CHECKED — refused, never truncated.
 *   one meaning     the numbers are opaque here. Nothing reads them, compares them or orders them.
 *   one dispose     per `create`. ⛔ NOT CHECKED, AND NOTHING WILL EVER NOTICE. A carousel is held by one
 *                   C caller and named by no value, so there is no count tracking it and no collector
 *                   that will find it. A missed `dispose` is room the platform gave out and never gets
 *                   back — an ordinary allocator leak, with no pool for it to sit in and no instrument
 *                   that reports it.
 *   an empty ring   to dispose one. CHECKED — a `dispose` with things still waiting is REFUSED, because
 *                   what is waiting is not bytes: for the pool's free list it is chunk indices, and
 *                   abandoning them loses those chunks forever with every count still balancing. Drain
 *                   first; draining is where you find out there was something left.
 *
 * WHAT IT PROMISES BACK
 *   order           `take` answers in the order things were added. That is the entire reason for the
 *                   shape — a stack would answer in the reverse, and would keep reusing the most
 *                   recently freed thing.
 *   no repeats      a value handed out is gone until somebody adds it again.
 *   a quiet empty   `take` on an empty ring answers false and raises NOTHING. Being empty is a state,
 *                   not a fault, and a pool that has run out is its caller's news to break.
 *   a loud refusal  every OTHER false comes with a named fault word. THERE IS EXACTLY ONE SILENT NO. A
 *                   ring that was never created is not an empty one, and answering the same way for both
 *                   would send a boot-order mistake out disguised as memory pressure — a shape nothing
 *                   downstream can tell apart from the real thing. It raises NO_RING.
 *   a safe count    `head - tail`, so it can be asked while others are adding.
 *   a dead handle   ⛔ IS NOT ONE OF THESE, AND IT IS THE ONE A CALLER MOST WANTS. `dispose` hands the
 *                   whole structure — the witness word and the storage pointer with it — back to the
 *                   platform, so a pointer kept past it is DANGLING. `add` and `take` reach their refusal
 *                   by READING that room: until the allocator hands it on they answer NO_RING, and after
 *                   it does they answer whatever the new tenant wrote there. Clearing `live` and `slot`
 *                   before the free costs nothing and catches the near case; it is not a guard, and the
 *                   `one dispose` a caller owes above is what actually covers this.
 *
 * WHAT IT DOES NOT DO
 *   It does not OWE ANYTHING TO THE HEAP, which is a narrower claim than it reads as. `create` asks the
 *   PLATFORM for `sizeof(sys__carousel) + slots x width` bytes through the silicon seam and `dispose` gives
 *   that same address back to it, so a ring is made with no heap base published, no block started and no
 *   chunk claimed. ⇒ ★ OWING NOTHING TO THE HEAP AND OWING NOTHING TO ANY ALLOCATOR ARE DIFFERENT CLAIMS:
 *   the second is what costs a fixed count and a fixed size, and this is the first — so the pair is
 *   allocate and free, and the suite checks the BALANCE rather than the calls.
 *   It does not SCRUB the room it gives back. A disposed ring refuses every verb, and the next `create`
 *   writes each of its own slots before anything can read one.
 *   It does not COUNT its holders. There is no `retain`, and `dispose` is not `release`: a release is a
 *   hold being given up and MAY end a thing, a dispose ends it. One owner, always.
 *   It does not END THROUGH THE LISP TEARDOWN. It has no kind and no row in
 *   `sys__LANGUAGE_CONTRACT__OBJECTS` at all, so the drain has no name to reach it by: a carousel is held
 *   by a C caller and ends at `dispose`. ⇒ ★ HAVING NO ROW IS A STRONGER CLAIM THAN HAVING ONE THAT
 *   REFUSES — a refusal is a door somebody has to keep shut, and this is a wall.
 *   It does not GROW. A ring is the size it was made, and `add` refuses past it rather than reallocating.
 *
 * ── THE WORKED EXAMPLE, WHICH IS THE HEAP ───────────────────────────────────────────────────────────
 *
 * The pool's free list is a carousel and nothing else, and every clause above is met somewhere visible:
 *
 *     static sys__carousel* sys__heap__zzprivate_free;      <- a POINTER: the ring owns its own room,
 *                                                            which `create` allocates from the seam
 *
 *     publish   free = sys__carousel__create(SLOTS_ABOVE(chunks), U32);   sized from the pool it serves,
 *                                                            and both numbers arrive at publish rather
 *                                                            than being written down anywhere
 *               for every chunk except zero:  sys__carousel__add(free, i);
 *     claim     sys__carousel__take(free, &idx)       false here IS the empty pool, and the heap is
 *                                                    what turns it into FAULT_EXHAUSTED, because the
 *                                                    heap is what knows that running out is news
 *     recycle   sys__carousel__add(free, chunk_of(chunk));
 *
 *   a power of two  `SYS__CAROUSEL__SLOTS_ABOVE(chunks)` — the macro at the foot of this file, over the
 *                   count `publish` was handed, so the geometry is decided at publish and not by a build
 *   big enough      the same macro rounds PAST the chunk count, so the ring is larger than it can ever
 *                   need to be and the FULL refusal is unreachable from here. See the size argument below
 *   never a zero     ⭐ NOT BY A CHECK — BY ARITHMETIC. Chunk zero is block zero's and never goes home,
 *                   and its INDEX is zero, which is the one value this ring refuses. The chunk that must
 *                   never be offered is the one whose name cannot be spoken here, and nobody had to write
 *                   a rule about chunk zero anywhere.
 *
 * ⚖ ARCHITECT, on why this exists rather than a table of flags: *"i dont really want the bitmap because i
 * would have to bypass n blocked heap chunks to find a free one ... an array sized as the first 2^n value
 * bigger than the heap chunk, plus two cursors as a head and a tail ... i add the newly returned heap
 * chunks at the top and then top becomes top+1 masked to the size of the array, and when i pick heap
 * chunks i pick them from the tail and increase the tail +1 and then bitmask to the size of the array.
 * this is the clean design that is fast and contains the freemem inside a carousel array."*
 *
 * ⭐ A RING IS UNBIASED, AND BIAS IS THE HARDER FAULT TO SEE. A search for a free chunk from a fixed
 * origin moves only when it is BLOCKED, so one region is reused continuously while the rest of the pool
 * is untouched — the property a stack is avoided for, arrived at without a stack. Taking from the tail
 * cannot do that: a chunk handed back goes behind everything already waiting, and cannot come out again
 * until they have.
 *
 * ⛳ AND THERE IS NO SCAN. A search is O(n) in the number of BUSY chunks, which is worst exactly when the
 * pool is nearly full and a caller can least afford it. Taking from the tail is two reads and a swap
 * whatever the occupancy.
 *
 * ⛔⛔ THE TRADE, WHICH IS REAL AND IS NOT WHAT THE DESIGN WAS CHOSEN AGAINST. A per-chunk flag has NO
 * shared state: every chunk is its own word, so two blocks contend only when they reach for the same
 * chunk. This ring has TWO words that every block touches. `REASONED`, and the premise is the
 * allocation shape: a chunk is claimed only when the one being carved from runs out, so the interval is
 * a chunk's nodes LESS the granule its header strands, over the granule — every allocation but one, per
 * chunk. So the contention is amortised by that factor before it starts, and the factor moves with the
 * dial rather than being a number written here. It has not been
 * measured.
 *
 * ── THE SIZE, AND WHICH HALF OF THE RULE DOES WHICH JOB ─────────────────────────────────────────────
 *
 * The phrase is two rules and they do different jobs. `head == tail` meaning EMPTY is NOT one of them —
 * that is bought in `add`, which refuses at `n-1`, and it would be bought there whatever the size.
 *
 *   a power of two   so the wrap is a mask and not a modulo, on the one path that runs per allocation.
 *                    This half IS the reason, and it is the architect's.
 *   bigger than      HEADROOM, not correctness.
 *
 * A ring of `n` slots can hold `n-1` things, because at `n` the two masked cursors land on the same slot
 * and a full ring would read as an empty one. `add` refuses at `n-1` and that is where the property comes
 * from. So the rule a CALLER must meet is `slots >= (the most that can ever be waiting) + 1` — and for the
 * pool, where at most every chunk but the block's own is ever free, that is `slots >= chunks`.
 *
 * ⛳ WHAT ROUNDING PAST IT BUYS IS THEREFORE NOT THE INVARIANT BUT THE DISTANCE FROM IT: the pool's ring
 * ends up holding at most `chunks - 1` against a ceiling of `slots - 1` that is strictly larger, so the
 * FULL refusal is UNREACHABLE for that ring rather than merely untaken. `>=` would size it to run exactly
 * on its ceiling, where the invariant still holds but every off-by-one anywhere upstream lands on it.
 *
 * ── WHY A SLOT SAYS WHETHER IT HAS BEEN WRITTEN ──────────────────────────────────────────────────────
 *
 * ⛔⛔ THE ONE RACE THE DESIGN AS DESCRIBED DOES NOT COVER, AND IT IS NOT AVOIDABLE BY ORDERING. Adding is
 * two steps — take a slot, write it — and with more than one adder they cannot be one step. Whichever
 * order they are done in, a taker can arrive between them:
 *
 *   write then publish   two adders publish out of order, and the later one's publish exposes the
 *                        earlier one's slot before it has been written. Ordering the stores does not
 *                        help; the problem is that one cursor publishes two slots.
 *   publish then write   the taker reserves a slot whose value has not landed yet, and knows it.
 *
 * The second is the one that can be repaired, because the taker can TELL. A slot is zero until an adder
 * writes it and is set back to zero by the taker that empties it, so "not written yet" is a state the
 * slot itself carries, and the taker waits for its own slot rather than for a cursor.
 *
 * ⭐ ZERO IS AVAILABLE FOR THAT BECAUSE IT IS ALREADY THIS PACKAGE'S WORD FOR NOTHING — a placement
 * returns zero to refuse, offset zero means nothing is being carved from, a null reference is zero. The
 * carousel keeps the convention rather than inventing a flag: ADDING A ZERO IS REFUSED, so a zero in a
 * slot can only ever mean the slot is not ready.
 *
 * ⛳ AND THE WAIT IS BOUNDED, for the reason every wait in this package is: nothing else would end it,
 * and a spin that cannot end is a process that hangs with no suspect. The adder's next instruction after
 * taking its slot is the store, so a real wait is short unless the adder's thread is descheduled
 * between the two; a wait that reaches the bound is a defect, and it says so.
 * ⚠ A take that gives up has CONSUMED its slot and returns nothing, so the value in it is lost. That is
 * the right trade against retrying — the alternative is putting the cursor back, which races every other
 * taker — and it is a leak of one entry on a path that has already raised.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* Round a count UP to the next power of two STRICTLY GREATER than it. The smear fills every bit below the
 * highest set one, so the result is all-ones and adding one carries into the next power — and a value
 * that is ALREADY a power of two smears to twice-itself-less-one, which is why this rounds past it rather
 * than onto it. `MEASURED` by the suite, which checks it directly: 240 → 256, 256 → 512, 16 → 32, 8 → 16.
 * ⛳ Those are VALUES and not geometries — nothing sizes a pool at 240; it is kept because a number
 * that is not already a power of two is the case the rounding exists for. */
#define SYS__CAROUSEL__SMEAR1(v)   ((v) | ((v) >> 1))
#define SYS__CAROUSEL__SMEAR2(v)   (SYS__CAROUSEL__SMEAR1(v) | (SYS__CAROUSEL__SMEAR1(v) >> 2))
#define SYS__CAROUSEL__SMEAR4(v)   (SYS__CAROUSEL__SMEAR2(v) | (SYS__CAROUSEL__SMEAR2(v) >> 4))
#define SYS__CAROUSEL__SMEAR8(v)   (SYS__CAROUSEL__SMEAR4(v) | (SYS__CAROUSEL__SMEAR4(v) >> 8))
#define SYS__CAROUSEL__SMEAR16(v)  (SYS__CAROUSEL__SMEAR8(v) | (SYS__CAROUSEL__SMEAR8(v) >> 16))
#define SYS__CAROUSEL__SLOTS_ABOVE(v)  (SYS__CAROUSEL__SMEAR16((unsigned int)(v)) + 1u)

/* ── THE CONTRACT, DECLARED ─────────────────────────────────────────────────────────────────────────
 * These five ARE the contract above, in the order it describes them. Every one is defined in
 * `carousel__impl.cuh`, which sits BEFORE the heap: `create` takes its room from the silicon seam, and
 * that file names no heap symbol at all — `MEASURED`, and recorded in its own banner. So the heap is a
 * CALLER of this contract rather than something it waits on. Nothing else in this file is a caller's
 * business.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__carousel* sys__carousel__create (unsigned int slots, unsigned int width);
static __device__ inline void         sys__carousel__dispose(sys__carousel* ring);
static __device__ inline bool         sys__carousel__add    (sys__carousel* ring, uint32_t value);
static __device__ inline bool         sys__carousel__take   (sys__carousel* ring, uint32_t* out);
static __device__ inline unsigned int sys__carousel__count  (const sys__carousel* ring);

#endif /* SILVANN__PACKAGES_SYS_CPU_CAROUSEL__HEADER_CUH */
