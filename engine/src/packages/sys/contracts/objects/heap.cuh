#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_HEAP_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_HEAP_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include "heap_node.cuh"

/* Raised when anything here is asked for before a heap base address exists. Its own word rather than a container's,
 * because nothing the caller did is what went wrong — the boot path has not run yet. */
#define SYS__HEAP__FAULT_NOT_PUBLISHED  0x48504E50ull   /* "HPNP" — nothing has published a heap base address       */

/* ── THE HEAP ALLOCATION POOL ────────────────────────────────────────────────────────────────────────────────────────
 * A heap chunk is 512 nodes, which is 32 KB at the shipping node size. Everything below says "chunk" and
 * means that one; a STACK chunk is a different thing and carries its own word wherever it appears.
 *
 * ⭐ WHY 512: THE COST IS FIXED PER HEAP CHUNK TYPE AND THE FRACTION IS WHAT FALLS. There are two costs and
 * they differ by eight times, which is worth knowing before either provider is sized against the other.
 *
 * A heap chunk spends two nodes describing itself and whatever is carved begins after them. For anything
 * that states its own size that is the whole of it: the header, plus whatever does not divide at the end.
 * Two-node objects in a 512-node chunk leave nothing over, so the cost is the two header nodes.
 *
 * ⛔ AN ARRAY CHUNK IS DIFFERENT, AND THE DIFFERENCE IS EXACTLY ONE SLOT. Its width is fixed, so what
 * does not divide is not a remainder but a whole array that will not fit. And the number is not an
 * estimate, and it does not depend on the header's size: a heap chunk is a multiple of the array width,
 * so whatever the header takes, the tail left after the last array makes it back up to exactly one array.
 * ⇒ THE HEADER COSTS PRECISELY ONE ARRAY CHUNK, AT EVERY HEAP CHUNK SIZE AND FOR ANY HEADER SMALLER THAN
 * ONE ARRAY, and no rearrangement recovers it.
 * ⚖ ARCHITECT: *"no matter how you rotate them if your first heap chunk has to be an object for
 * convenience it robs us of an array chunk. it is a cost i have accepted."*
 *
 * ⇒ SO THE SIZE IS ARGUED FROM THE ARRAY SIDE, because that is the expensive one: 3.125% here against
 * 6.25% at half this size, where the same doubling takes an object's loss from 0.78% to 0.39%. Both
 * halve, and only one was ever large enough to argue about.
 *
 * The number is not a preference either way: an offset within a heap chunk fits two bytes at this size
 * and at the largest one a page can hold, and two bytes is what lets the same arithmetic address both
 * without a division anywhere on the path that runs per object.
 *
 * ⭐⭐ WHAT THIS FILE SETTLES IS A MINIMUM, NOT A COUNT, AND THE TWO ARE EASY TO READ AS ONE. How many
 * chunks there are is not written here and cannot be: the number wanted follows the BLOCKS — the runner
 * threads a launch stands up, one per card plus the orchestrator — and that is settled at boot, not
 * when this file is compiled. So `publish` is TOLD, and whoever allocates the heap allocation pool is who
 * knows — the same code, from one computation, rather than two that can disagree about the same memory.
 * ⚠ THE HEADROOM ABOVE THE FLOOR IS NOT YET DERIVED FOR RUNNER THREADS; not refusing is all that has
 * been shown.
 *
 * ⛔ WHAT IS SETTLED HERE IS THE SMALLEST HEAP ALLOCATION POOL THAT DOES NOT REFUSE, AND ASKING FOR IT IS A MISTAKE. A
 * heap allocation pool exactly at the floor starts, hands out every pair it must, and is then empty —
 * so the first block to fill one and ask for another gets nothing. It does not degrade; it fails at the
 * first request that was always going to come. ⇒ THE FLOOR IS WHAT A CALLER MAY NOT GO UNDER, and it is
 * not what a caller should ask for.
 *
 * A block carves from TWO chunks at once, and they are its two PROVIDERS: one holding what LASTS and one
 * holding what CHURNS, kept apart so a single long-lived object cannot pin a chunk full of dead
 * short-lived ones. Its own computing base is not a third chunk — it lives inside the first of the two,
 * and starting a block is what puts it there.
 * ⚠ AND THAT LAST SENTENCE IS THE FLOOR'S PREMISE, SO IT IS THE THING TO WATCH. The bases are one day the
 * system register's to hand out; the register rides with block zero, so on that day every block's base
 * leaves the chunk that block carves from and lands in BLOCK ZERO'S. Two chunks per block still buys two
 * providers, but block zero is then carrying the whole grid's bases inside one of its own — and this
 * number is derived from a sentence that would by then be describing somewhere else. ⇒ WHOEVER MOVES THE BASES
 * RE-DERIVES THIS FLOOR; it will not fail on its own, it will simply stop being the smallest pool that
 * works.
 * ⭐ AND THE SYSTEM REGISTER IS NOT A TERM HERE, WHICH IS A RULING AND NOT AN OVERSIGHT. It rides with
 * block zero: its storage is that block's storage, carved from that block's providers, and block zero is
 * already counted. So the floor asks about blocks and nothing else, and every claimant this heap has is a
 * block or lives inside one.
 * ⇒ TWICE THE BLOCK COUNT, IN CLAIMABLE CHUNKS, IS THE FLOOR — the only number here that is DERIVED
 * rather than chosen: under it a block cannot hold both its providers, and a heap allocation pool that
 * cannot do that does not run slowly, it fails to start. `publish` refuses under exactly this and nothing
 * else.
 * ⛳ CLAIMABLE IS THE LOAD-BEARING WORD. Chunk zero never enters the free ring, so a heap allocation pool of n offers
 * n-1, and a floor written against n accepts a heap allocation pool one short of what its own blocks need.
 *
 * ⚠ ANYTHING ABOVE THE FLOOR IS A CURRENT CONDITION AND NOT A RULE. A block that fills what it started
 * with abandons it and claims another, and the abandoned one comes back only when everything inside it
 * has died — so the headroom worth having depends on how much is alive at once, which is a property of
 * the program being run and not of this allocator. Twice the floor again is a reasonable place to start
 * for a workload nobody has measured; whoever allocates may know better, and the floor is the part they
 * may not go under.
 *
 * At this chunk size the floor is a few megabytes against a machine that has tens of thousands, which is
 * why the floor is worth respecting rather than economising against.
 *
 * The chunk size is overridable so a harness can run a small geometry, and everything below is written in
 * terms of it rather than in terms of numbers.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
#ifndef SYS__HEAP__CHUNK_NODES
#define SYS__HEAP__CHUNK_NODES 512u

#endif


/* How many nodes an ARRAY CHUNK takes. It is a kind's size and not a rule about allocation: nothing else
 * is rounded to it, and every other kind states its own in the same place — its row.
 *
 * ⭐ WHY THE ARRAY KEEPS A FIXED WIDTH WHEN NOTHING ELSE DOES. An array chunk is not one value, it is a
 * run of them with a head in front, so its size is a CAPACITY rather than a shape: how many cells one
 * link of a chain holds before another is needed. A granule of N nodes is N-1 cells because the first is
 * the head, which is why the two differ by one wherever they appear together.
 *
 * ⛳ AND IT IS A CAPACITY AND NOTHING ELSE — NOT AN ALIGNMENT. The heap base address need not sit on a
 * multiple of it: what a node needs is the node's own alignment, the type carries that, and no address
 * in this package is ever masked. One number, one meaning. */
#ifndef SYS__ALLOC__ARRAY_GRANULE
#define SYS__ALLOC__ARRAY_GRANULE    16u
#endif



/* ── WHY MAKING ONE CAN FAIL, AND WHY EACH WAY GETS ITS OWN WORD ───────────────────────────────────────────────
 * They are worth separate words because they want different responses and a reader of the log has only
 * the word to go on: a full heap allocation pool is a CONDITION and may be survivable, no computing base is a CALLER that
 * never started a block, a kind with no row at all is a REGISTRATION nobody wrote or a package left out
 * of the build, a kind whose own row declares no provider is a thing placed by hand and NEVER MEANT TO BE
 * MADE, a heap chunk offered while somebody still holds it is the heap allocation pool DISAGREEING WITH ITSELF, and a
 * chunk with no room for one granule is a GEOMETRY that cannot happen, so seeing it means an assumption
 * somewhere else has already broken. Nothing having published a heap base address has its own word further up,
 * because that is a launch that has not started rather than a making that failed.
 * ⛔ AND WHAT MAKES IT UNABLE TO HAPPEN IS NOT IN THIS FILE, WHICH IS WORTH KNOWING BEFORE RELYING ON IT.
 * There is no assert here at the granule. The one that does the job is in `contracts/objects/stack.cuh`,
 * under a different macro, in a file ordered AFTER this one — so this file's geometry guarantee is
 * BORROWED FROM THE STACK and would leave with it. ⇒ ★ A GUARANTEE HELD BY SOMEBODY ELSE IS ONE YOU CAN
 * LOSE WITHOUT TOUCHING YOUR OWN CODE.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
#define SYS__HEAP__FAULT_EXHAUSTED 0x4850464Cull   /* "HPFL" — the heap allocation pool has no chunk left */
#define SYS__HEAP__FAULT_NO_BASE   0x48504E42ull   /* "HPNB" — nothing has started a block here */
#define SYS__HEAP__FAULT_GEOMETRY  0x48504745ull   /* "HPGE" — a fresh chunk could not hold one granule */
#define SYS__HEAP__FAULT_NO_PROVIDER 0x48504E56ull  /* "HPNV" — a kind that is not carved from a provider */
#define SYS__HEAP__FAULT_NOT_FREE  0x48504E46ull   /* "HPNF" — the carousel offered a chunk that is held */
#define SYS__HEAP__FAULT_UNKNOWN_KIND 0x48504B44ull /* "HPKD" — no package registered this kind at all   */
/* Raised at publish, not at an allocation: the heap allocation pool cannot hold two chunks for every block in the launch,
 * so some block would be unable to start and which one is decided by arrival order. Its own word because
 * it is the only failure here that is settled before anything has run. */
#define SYS__HEAP__FAULT_TOO_SMALL 0x48505348ull   /* "HPSH" — the heap allocation pool is smaller than the grid needs */

/* ── THE CHUNK HEADER ────────────────────────────────────────────────────────────────────────────────
 * The first nodes of every chunk describe the chunk itself, and the rest is what it was claimed for.
 *
 *   [0] the heap chunk head  its kind, how many hold it, its lock, and the watermark — how far into
 *                       this heap chunk anything has been placed, as a node index                  LIVE
 *   [1] the requestor   the computing base of the block currently carving from this heap chunk  LIVE
 *   [2] ...             the end of the header, and where the FIRST object lands — nothing is rounded
 *                       up to anything, so the first free node is the first used one
 *
 * ⛔ NOTHING IN THE HEADER IS RESERVED. `requestor` is written on every claim, and what it buys is this:
 * from an object\'s address, mask to the heap chunk, read the requestor, and the block carving there is
 * found without being handed anything.
 *
 * ⛳ AND `[2]` IS WHERE THINGS BEGIN, WHICH IS NOT THE SAME AS NOTHING BEING LOST. Each allocation starts
 * where the last one ended, so a chunk carved into objects spends only the two nodes it describes
 * itself with. A chunk carved into ARRAY CHUNKS also strands the tail at the end that cannot hold
 * another one — which, with the header, is exactly one array's worth. Room INSIDE an allocation is a
 * third question and belongs to whichever kind asked for it.
 *
 * The watermark is where the next object goes, so allocation is an addition and nothing is searched. It
 * ONLY EVER RISES — nothing releases space back into a chunk, so a release leaves the mark exactly where
 * it was and the room behind it stays unoffered until the whole chunk goes home. That is what makes it a
 * high-water mark in the plain sense: it records how far this chunk was ever written.
 * ⛔ AND THE WORD IN THE POOL IS EXACTLY WHAT IS NOT READ WHILE A BLOCK CARVES HERE. Starting to carve
 * takes the number into that block's own storage and tags the head `HEAP_CHUNK_ACTIVE` so nobody reads
 * the copy left behind; the carving does its arithmetic on what it took, and stopping writes it back.
 * That is why `zzprivate_place` refuses an active chunk outright rather than reading a watermark that
 * stopped moving the moment the carving started.
 *
 * ⛳ THREE THINGS, AND THE SLOT'S NAME ONLY MENTIONS ONE OF THEM.
 * The heap chunk is an object that IS OWNED, by the elements that are allocated into it, and by the
 * COMPUTE BLOCK allocating objects into it. And what the slot holds is neither: a compute block is not
 * something that can be pointed at — it is a number the seam answers when asked — so the reference
 * is to that block's COMPUTING BASE, which is the one object per block that everything else about it
 * hangs from.
 * ⇒ READ `requestor` AS "WHICH BLOCK IS CARVING HERE, BY WAY OF THE ONE OBJECT THAT STANDS FOR IT."
 *
 * A reference and not a copy, because a base is a live thing whose status changes while the chunk sits
 * there; a chunk holding a copy would be answering with yesterday's.
 *
 * ⛔⛔ IT IS TAKEN BACK WHEN THE BLOCK MOVES OFF, NOT WHEN THE COMPUTATION ENDS, AND THAT INTERVAL IS THE
 * WHOLE DESIGN. ⚖ ARCHITECT: *"you then need to remove it when you get a new chunk, not when you exit
 * from compute otherwise you lock the release of that heap chunk."* Held to the end of a computation, a
 * recycled heap chunk would still name a computing base, and giving it back would owe a release into a
 * thing whose method RAISES — so a heap chunk could not go home without faulting. Cleared at the move,
 * a heap chunk in the bin names nobody and nothing is ever owed.
 * ⇒ ★ THE NAME FOLLOWED THE INTERVAL. `owner` claims something permanent; what is stored is true only
 * while this block is carving here, which is what a REQUESTOR is.
 *
 * ⛳ AND IT IS WEAK TODAY — an offset and no hold — because the answer it gives is an ADDRESS and the
 * thing it names outlives everything that could ask: the heap chunks are the block's, and so is the base.
 * Counting it would buy that guarantee as a checked property rather than an argued one, at one retain and
 * one release per claim. ⚠ BOTH WORK: with the reference given back at the move, the count comes down
 * while the base is plainly alive, so the base's raising release is never reached. The choice is open
 * and neither answer is owed.
 *
 * ⛔ AND WHATEVER ARRIVES NEXT DOES NOT ARRIVE HERE. A header whose fields can move is one that every
 * heap chunk in the heap allocation pool would have to be re-read to trust — so a new thing gets a chunk of its own
 * rather than a slot after this one, and the two below keep the positions they have.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
#define SYS__CHUNK__HEAD        0u   /* the node the chunk BEGINS with, which is also its object head */
#define SYS__CHUNK__REQUESTOR   1u   /* the base of the block carving here, while it is carving here   */
#define SYS__CHUNK__FIRST       2u   /* the first node an allocation can use                           */

/* What every verb here that answers with a chunk INDEX says when there is no chunk to name — an address
 * outside the heap allocation pool, a ring with nothing left in it, a chunk somebody is already inside.
 * ⛳ IT IS THE TOP OF THE RANGE AND NOT ZERO, and that is the whole reason it can exist at all: chunk
 * zero is a real chunk — the one the boot hands to the first block — so zero is an ANSWER and cannot
 * also be the absence of one. The carousel makes the opposite choice for its slots, where zero means
 * unwritten, and the two do not collide because neither ever holds the other's number. */
#define SYS__HEAP__NO_CHUNK     0xFFFFFFFFu

/* ⭐⭐ A CHUNK IS AN OBJECT, AND NOTHING HAD TO BE ARRANGED FOR IT TO BE ONE. ⚖ ARCHITECT: *"we can
 * simply make the declaration of 'this is a heap chunk' be a sysastnode ... and one of the arguments of
 * the object itself would be its reference counter, just like an object."*
 *
 * Every piece an object needs, node 0 already has. It carries a DTYPE — `HEAP_CHUNK_ACTIVE` while a block
 * is carving from it, `CHUNK_PROVIDER` while it is a provider whose head in the pool is the one to read,
 * `HEAP_CHUNK` when it is in the bin — so it declares what it is. ⛳ TWO OF THE THREE MEAN PROVIDER, which
 * is why `zzprivate_is_provider` tests both, and why `dtype == CHUNK_PROVIDER` written as the test for "a
 * block is carving here" answers false for every chunk one is carving from. Its `num_args` is
 * where an object keeps its count. And finding it falls out of the geometry rather than being fitted to
 * it: a heap chunk sits at `index × CHUNK_NODES`, so clearing the low bits of an offset anywhere inside
 * one lands on node 0.
 * ⚠ THAT IS NOT HOW ANY OTHER HEAD IS FOUND, and the difference is not a wrinkle — it is the two
 * questions being different. A heap chunk is at a fixed stride, so masking answers "which chunk is this
 * in". An object is packed at whatever width its kind asked for, so nothing recovers ITS head from an
 * interior address and nothing needs to: whoever holds a reference holds the node after the head.
 *
 * ⛔ WHICH IS WHY THE WATERMARK IS AT `args[1]` AND NOT THE `args[0]` A READER WOULD EXPECT. ⚖ ARCHITECT,
 * asked why: *"cause that's the contract."* The lock takes `args[0]` in every kind of object and a heap
 * chunk is not an exception, so the watermark is what moved along.
 * ⛳ THAT IS THE WHOLE ANSWER, and everything after it is only the price of disagreeing: the two
 * overlapping would mean a caller that locked a heap chunk silently rewound its watermark. Room was never
 * the constraint — a node carries six arguments, and nothing was competing for the room.
 *
 * ⚖⚖ AND A CHUNK DOES NOT HAVE TO BE AN OBJECT, WHICH IS THE REASON TO SAY THAT IT IS. ARCHITECT:
 * *"chunks are private by design and they are single owner by design. we can explore if we should make
 * it a property, make it lock-free and count-free — as it is locked by the head object and released by
 * it — but they are an ast object nonetheless in their representation so it might be worth it to keep it
 * aligned for that reason."*
 *
 * Both of the words this costs could be argued away, and it is worth knowing how completely.
 * `MEASURED`: NOTHING IN THE PACKAGE EVER TAKES A CHUNK'S LOCK. All four lock sites are on a stack, in
 * `stack__impl.cuh`, and what protects a chunk is the lock of the object whose head lives in it.
 * The only thing that has ever locked a chunk is the fixture that proves one can be. The count is used —
 * but every holder except one is an object inside the heap chunk, and the exception is the single block
 * carving from it. A thing reached only from inside the package, by one owner at a time, needs neither
 * word in order to be correct.
 *
 * ⇒ ⭐ THEY STAY BECAUSE ONE REPRESENTATION IS WORTH MORE THAN TWO WORDS. Everything that walks a head —
 * the masking, the count, the type test, the teardown — meets a chunk without being told it is meeting
 * one, and the arm that would tell it apart never has to be written. Special-casing the one kind that
 * could afford to be special is how a package ends up holding two shapes for a single idea.
 *
 * ⛳ SO WHAT IS OPEN IS THE PROPERTY AND NOT THE LAYOUT. Private and single-owner are true today because
 * of the way the package is written: nothing states them and nothing checks them. Making them something
 * a chunk DECLARES is the exploration, and this header is not what it would change. */
#define SYS__CHUNK__LOCK       SYS__HEAP_OBJECT__LOCK   /* args[0] of node 0 — the object convention */
#define SYS__CHUNK__WATERMARK  1u                      /* args[1] of node 0 — moved off the lock */


/* ── WHAT THIS ALLOCATOR COUNTS, WHICH IS THREE THINGS AND ITS OWN ──────────────────────────────────
 * A block of plain words the boot path owns and `publish` is handed. It is DIAGNOSTIC: nothing here
 * reads them back and no decision turns on them, which is why the one that fires per DEATH is behind an
 * off-by-default switch and compiles to nothing when it is down.
 * ⛔ THE OTHER TWO ARE NOT SWITCHED, which anyone sizing a shipping build should know: `CHUNK_CLAIMS` and
 * `CHUNK_REFUSED` are counted unconditionally inside the claim, so a build with the switch down still
 * pays their add every time a chunk leaves the free ring.
 * ⛳ THEY ARE `unsigned int` AND THE ADD IS A CAS LOOP, which is a choice about these counters and not about
 * the seam: the seam publishes `add_u32` — bought precisely because a retry loop is a structure the
 * compiler must keep whole and that structure is what costs — and both halves of the reference count are
 * a single instruction through it. A diagnostic sits nowhere near a hot path, so the loop here has never
 * been worth taking out, and a diagnostic wrapping at four billion is a price nothing here reads back. */
#define SYS__HEAP__COUNTER_CHUNK_CLAIMS   0u   /* a heap chunk left the free ring for a block          */
#define SYS__HEAP__COUNTER_CHUNK_REFUSED  1u   /* one was asked for and the ring was empty             */
#define SYS__HEAP__COUNTER_DEALLOCATIONS  2u   /* an object's last hold went                           */
#define SYS__HEAP__COUNTER_N              3u


typedef struct { unsigned int at[SYS__HEAP__COUNTER_N]; } sys__heap_counters;

/* ── HOW MANY THINGS ARE LIVING IN A CHUNK — WHICH IS ALSO WHETHER IT MAY BE HANDED OUT ──────────────────
 * One number answers both, because nothing happens between them. Nothing points at a chunk, so a chunk
 * that empties is free in the same step: there is no unlinking to do first, and therefore no moment in
 * which it could be empty and not yet available.
 *
 *     count == 0     nobody is inside, and the heap allocation pool may hand it out
 *     count == 1     one hold: either the block that is carving from it, or one lone object
 *     count == n     n holds
 *
 * ⭐⭐ AND THE COUNT LIVES IN THE HEAP CHUNK, IN THE SAME WORD EVERY OTHER OBJECT KEEPS ITS COUNT IN.
 * `SYS__HEAP_OBJECT__COUNT` reads a chunk's head exactly as it reads anything else's, because a chunk's
 * head is a head. So there is ONE counting in this package rather than one for objects and another for
 * the memory they sit in — two representations of the same fact, maintained by different code, are two
 * things that can disagree.
 *
 * ⭐ AND THE CLAIM IS LITERALLY THE FIRST HOLD. `claim_chunk` takes a chunk with a compare-and-swap from
 * 0 to 1, and that 1 is the carving block's own hold on it — not a flag that happens to look like a count.
 * The block drops it when it moves on to the next heap chunk, which is what lets the one it left behind
 * reach zero and go home.
 *
 * ⛔ WHICH MAKES ONE STEP LOAD-BEARING THAT IS EASY TO FORGET: a block moving off a full chunk MUST let
 * go of it. Miss that and the chunk sits at 1 forever with nothing in it — a leak that no count is wrong
 * about and nothing reports. `make` does it at the moment the next chunk is taken.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ── THE CHUNK A BLOCK IS CARVING FROM, AND HOW FAR IT HAS GOT ───────────────────────────────────────
 * ⚖ ARCHITECT: *"you not only need to cache the address of the heap, you need to cache its first node
 * too that contains the useful info."*
 *
 * The head of a provider chunk holds four things and they do not belong to the same owner, which is why
 * only two of them are here:
 * ```
 *   dtype       what the chunk is                 the carving block's, and it says so below
 *   watermark   how far the carving has got       the carving block's           ⇒ CACHED
 *   count       how many live things sit here     written by whoever releases one  ⇒ LEFT IN MEMORY
 *   lock        never taken by any path today
 * ```
 * ⛔ THE COUNT IS THE ONE TO BE CAREFUL WITH, and the reason is not where it is written but WHO writes
 * it: taking an object apart lets go of the chunk it lived in, and that runs wherever the last reference
 * to that object happened to be. A copy of it here would be a copy of a number somebody else is moving.
 * The watermark has no such second writer — the four places it is touched are all in `cpu/heap__impl.cuh`, and
 * three of them are the carve.
 *
 * ⭐ SO THE CHUNK SAYS WHICH IT IS. While a head is cached the node in the pool wears `HEAP_CHUNK_ACTIVE`
 * and stops being the thing to read; the write-back in `cpu/heap__impl.cuh` puts the real number
 * back and restores the tag. More than one runner can be live, and the count in that same head has a
 * reader that is not this block, so the tag is what tells that reader the head is not the one to read.
 * ⛔⛔ THE TAG IS WHAT KEEPS THIS HONEST — do not remove it as redundant. ⚖ *"one
 * thread per gpu and one for the orchestrator at id 0."* `MEASURED`: `eng_abi_boot(4)` stands
 * up FOUR runners on a CPU. ⚠ UNMEASURED under real contention — every program so far runs on block 0
 * with the others parked, so two runners have not yet raced here.
 *
 * ⛳ THE WRITE-BACK HAPPENS AT EXACTLY TWO MOMENTS and both already exist as lines in this file: when a
 * chunk fills and the block moves to the next, and when the block stops carving at all. */
/* ⛳ THE TYPE CARRIES NO PRIVACY MARKER BECAUSE NO TYPE IN THIS TREE DOES — `sys__heap_node`,
 * `sys__heap_counters` and `sys__stack_chunk` are all spelled plainly. ⛔ AND NEITHER DOES THE ACCESSOR
 * BELOW: `sys__heap_carving_at` carries no tier marker, so the gate's name pattern does not match it and
 * nothing would refuse a reach from another file. What keeps this one private is that the storage lives
 * inside that accessor and every caller of it is in this file — held by reading rather than by the gate.
 * ⇒ ★ A MARKER IS THE ONLY THING THE GATE CAN SEE, so a name without one is private by habit. */
typedef struct {
    sys__heap_node* chunk;      /* what is being carved from, or nothing         */
    uint64_t        watermark;  /* how far into it, and this is the live copy    */
    uint32_t        appends;    /* holds taken here and not yet written down     */
} sys__heap_carving;

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_OBJECTS_HEAP_CUH */
