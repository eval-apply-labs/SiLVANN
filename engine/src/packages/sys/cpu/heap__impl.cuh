#ifndef SILVANN__PACKAGES_SYS_CPU_HEAP__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_HEAP__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../../manifest__header.cuh"   /* the list of packages, which the gathers below are driven by */
#include "../language_contract.cuh"
#include "heap_object__header.cuh"
#include "bindings__header.cuh"
#include "carousel__header.cuh"
#include "computing_base__header.cuh"
#include "fault__header.cuh"
#include "error.cuh"
#include "heap__header.cuh"
#include "list__header.cuh"
#include "node_array__header.cuh"
#include "silicon/silicon__header.cuh"
#include "system_register__header.cuh"   /* where the blocks' bases are is a row, not a static here */
#include "stack__header.cuh"
#include "procedure.cuh"                 /* the contract row names its size */
#include "string.cuh"                    /* and so does this one */
#include "dictionary.cuh"                /* and these three */
#include "../contracts/objects/heap.cuh" /* the layouts it reads and writes */
/* ════════════════════════════════════════════════════════════════════════════════════════════════════
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT. Nobody maintaining this package reads these; the rules for
 * them are in `README.md`. This one exists so that an agent does not "restore" the allocator this file
 * replaced, or re-add a primitive that was removed for a reason the code no longer shows.
 *
 * ⛳ WHAT THE HOST MOVE (⚖ *"sys and the evaluator go host"*) DOES TO THIS FILE — three notes, none a bug:
 *   · THE REGISTER ARGUMENT near `computing base` INVERTS on a CPU. It refused a register because
 *     registers belong to a wavefront; a CPU has none, and a per-runner value in a callee-saved register
 *     or a `__thread` slot is the right home there. Memory-backed stays correct everywhere, so this is
 *     a cost to reclaim, and the storage choice should be RE-DERIVED on the host, not inherited.
 *   · THE −7.8% for the base cache is a `k_eval` figure. The cache is worth having on either processor;
 *     the number is not transferable and must not be quoted at a host reader.
 *   · THE "EMPTIED AT THE START OF EVERY LAUNCH" HAZARD CHANGES SHAPE rather than going away. With no
 *     launches the stale pointer arrives as a RUNNER THREAD's cached base outliving the computation that
 *     set it — and, `BLOCK_LOCAL` being `static __thread`, invisible to every other runner.
 *     `scripts/src_base_cache_gate.py` enforces the per-kernel form and will have no subjects.
 *     ⚖ WHAT ARMS THE CACHE ON A HOST EVALUATOR IS UNRULED.
 *   · The tag paragraph's *"the evaluator runs one block"* was the reason given for the tag; the tag is
 *     now the reason, since more than one runner can be live.
 *
 * ⛳ DEVICE FIGURES MOVED OUT OF THE ORDINARY COMMENTS, all `k_eval` on gfx906 and none re-measured on
 * the host:
 *   · the base cache: 34.9 -> 32.3 us per carve, -7.8%, and -3.9% on fib(15); the chain was about six
 *     loads rather than the five arrows a diagram of it shows.
 *   · the base asked once per block rather than per allocation: 1.16% of the whole evaluator.
 *   · `sys__heap__make` inline: 2.70% of the whole evaluator, `k_eval`'s frame 896 -> 824 bytes and its
 *     spills 89 -> 86. Inlining `zzpackage_make_sized` too: -2.25% on its own and +9.2% together with
 *     `make`, the frame going to 1,304 bytes with 206 spills.
 *   · the stale-base page fault: arming only the two kernels a probe happened to use killed the device
 *     suite, because `k_list_create`, `k_freeze` and the boot kernels carved too. Those kernels are gone.
 *   · the register argument beside `zzpackage_computing_base`: scalar registers belong to a wavefront (a
 *     block of 512 threads was eight wavefronts), so a SHARED value that changes could not live in one;
 *     `sys__silicon__block_id` was uniform, so the base's address was a scalar load hoisted out of loops.
 *     Its falsifier was a device disassembly, and the script for it retired with `k_eval`. RESTATED in
 *     the ordinary comment for the host: the base is stored `__thread`.
 *
 * ⛳ HISTORY MOVED OUT OF THE ORDINARY HEADER BELOW, so the header can say what the code IS:
 *   · An aligned placement, `place_aligned`, rounded the watermark up to a boundary to allow finding an
 *     object's head by masking its ADDRESS. It could not work — an aligned offset in a chunk whose base
 *     is aligned to nothing is an unaligned address — and the base requirement that would have made it
 *     true was removed too. Adding the verb alone reproduces exactly what was deleted.
 *   · The count was a side array, `chunk_live[]`, indexed by chunk number: two countings with two sets of
 *     primitives. With it went the pool's published pointer and the borrowed `sys::heap_retain` /
 *     `sys::heap_release`, which carried the old tree's 64-chunk bound and would have miscounted every
 *     chunk above the sixty-third. While it existed the heap could be handed out untouched; 0xA5 poison
 *     was harmless then.
 *   · Free chunks were found by a scan from an origin fixed per block, O(n) in busy chunks. `MEASURED` on
 *     the fixture, claim-and-free ten times: `1 1 1 1 1 1 1 1 1 1` against `1 2 3 4 5 6 7 8 9 10` for ten
 *     claims with no frees — one region reused while the rest was never touched.
 *   · The chunk lock was declared the moment the watermark moved off `args[0]` and never written:
 *     `publish` cleared the count only, `init` wrote four other fields, `recycle` two. `MEASURED` by one
 *     added fixture check: a freshly claimed chunk's lock word read `0xa5a5a5a5a5a5a5a5` and
 *     `sys__heap_object__lock` on it returned false. `init` clears it now.
 *   · THE OLD TREE ALLOCATES FROM A BITMASK. `src_old/packages/sys/heap.cuh` claims chunks by setting a bit
 *     in one 64-bit word of the shared parameter array; its two callers are
 *     `src_old/engine/GPU_CMD_LIST.cuh:112` and `src_old/engine/GPU_CMD_MTCOOP_OPEN.cuh:198`. This tree does
 *     not include that directory, so the two allocators never coexist in one binary; the old half is filed
 *     as a defect against the old tree. It was replaced for REDUNDANCY, not performance: the mask was a
 *     shadow of "chunk_live[i] != 0", claiming set the mask without the count, and a claimant that forgot
 *     its retain left a chunk whose next release wrapped from zero. Its rationale — a bitmask avoids ABA —
 *     holds equally for a CAS from zero to one. Its linear scan was argued for 16 chunks and kept at 64,
 *     sized so 58 compute units could each hold one, which is where the scan is worst. Chunk zero: the old
 *     scan took it on the first claim of every launch although both release paths refuse to free it.
 *   · PROVENANCE. ⚖ Architect: "things get assigned from the main arc as a counter, we dont care who owns
 *     it as long as when it takes it gives a +1 to the main arc and once it gets rid of it it does a -1",
 *     and "the bitmask ... does not hold at 120 blocks". An earlier ruling collapsed the object accounting
 *     to one integer per object and left the allocator out of scope: "heap_chunk_live is UNCHANGED and
 *     still a DIFFERENT question ... It is not a reference count." This file is that remainder.
 *   · Other comparisons dropped from the ordinary text: publish's two boundaries "were the same number
 *     until something needed room that outlives every chunk"; `take_up` and `give_up` were CAS retry
 *     loops until the seam published `add_u32`; `make`'s provider "used to be named by a caller" and
 *     nothing checked it; the sized carve did not exist, so "a caller could name the kind or the size,
 *     never both"; the carve cache replaced a slot read, a tag read and two watermark reads in the pool.
 *
 * RETIREMENT: everything here goes when the old tree does — it is all comparison, and the comparison
 * loses its subject. Nothing in this file or anywhere else in the package borrows from `src_old/`: the fault
 * channel is `sys`'s own, declared as a seam in `fault__header.cuh` and filled in
 * `fault__impl.cuh`, which shapes it like `sys__heap_counters` for the reason it gives there.
 * ════════════════════════════════════════════════════════════════════════════════════════════════════
 *
 * ⛔ PLACEMENT IS UNALIGNED, AND AN ALIGNED PLACEMENT ALONE CANNOT BE ADDED. Nothing masks an object's
 * ADDRESS and the base has no alignment requirement, so an aligned offset would still be an unaligned
 * address. ⇒ IF ALIGNMENT IS EVER WANTED, BOTH HALVES COME TOGETHER — a base requirement and the
 * placement verb.
 *
 * ── WHY THE HEAP ALLOCATION POOL IS ZEROED AT PUBLISH ──────────────────────────────────────────────
 * The count lives inside the memory, in each chunk's head, so memory nobody has written reads as OCCUPIED
 * by whatever bytes are there and the allocator would find a full heap allocation pool. ⚠ `MEASURED` by
 * the harness's poison: 0xA5 fill makes every unzeroed heap chunk claim to hold 2,779,096,485 things.
 *
 * ── AND WHY SOMETHING HAS TO PUT CHUNKS ON THE RING ─────────────────────────────────────────────────
 * The carousel is a LIST, so somebody has to add to it, and the only moment that already walks every
 * heap chunk is the same one that zeroes them.
 *
 * ── ONE COUNTING, IN THE HEAD ────────────────────────────────────────────────────────────────────────
 * A chunk's occupancy is the same word every object keeps its count in, moved by the same `take_up` and
 * `give_up`, so there is one set of primitives and one meaning for an underflow. The pool's size is
 * settled at publish and has no compile-time ceiling.
 *
 * ── CHUNK ZERO ──────────────────────────────────────────────────────────────────────────────────────
 * Chunk zero is block zero's and never enters the free ring, so no claim can take it and no release
 * can hand it back: it is permanent by construction.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ── THE THREE THINGS THAT ARE THE SAME FOR EVERY BLOCK ──────────────────────────────────────────────
 * The heap base address, the heap allocation pool's free list and the diagnostic counters are the same
 * three addresses for every block. There is one heap base address, and every offset in every
 * structure is measured from it; there is one free list, and one heap allocation pool for it to describe.
 * Handing them to each function made them look like something a caller might choose, and cost a parameter
 * at every level of a call chain to say the same thing at each one.
 * ⛳ WHERE THE BLOCKS' COMPUTING BASES ARE IS NOT ONE OF THEM, and the include list at the top of this
 * file says so: that array is A ROW IN THE SYSTEM REGISTER, read out of the register on the way to an
 * allocation rather than held in a static here, and the register is stood up by a later boot kernel than
 * the one that publishes the heap. A static would be read before anything had written it.
 * ⛳ WHAT DOES NOT VARY IS WHERE THEY ARE, NOT WHAT IS IN THEM. The free list is taken from and added to
 * all launches, and the counters are written on every hold and every release when they are on. It is the
 * three POINTERS that are settled, and they are settled once, before any block runs.
 * ⛳ ONE OF THE THREE IS WHAT `published()` TESTS — the heap base address — because it is what everything
 * else reaches THROUGH, and the only one of them the rest of the package cannot do without. The bases are
 * deliberately not in that test, and the accessor below says why: the register is built OUT OF the heap,
 * so asking about them here would make the heap unpublished during the window in which it is standing up.
 *
 * ⛔ THE TRADE IS HONEST AND IT IS WORTH KNOWING BEFORE READING THE REST. A parameter announces at the
 * call site what a function touches; these do not. What buys that back is that a parameter can also be
 * passed WRONG, and these cannot — there is exactly one of each, set exactly once, and the setting is
 * the only place it can fail. `REASONED`, not `MEASURED`: the codegen claim usually made for this trade
 * — a scalar load the compiler hoists either way — is not one I have any right to make, because nobody
 * has put the parameter form against the static form.
 *
 * WHO SETS THEM: the boot path, once, before any block runs, through `sys__heap__publish`. NOT a block —
 * every block writing the same value to the same word is benign only by accident, and a
 * thing that is correct by accident is a thing nobody wrote down.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ⭐ WHAT IS FREE, IN THE ORDER IT BECAME FREE. The ring is sized past the number of heap chunks rather
 * than onto it, which is what lets `head == tail` mean EMPTY and nothing else — the argument is in
 * `carousel__header.cuh`, and the half this file owes is a RUNTIME one: `publish` sizes the ring with
 * `SYS__CAROUSEL__SLOTS_ABOVE` over the chunk count it was handed. ⛳ IT CANNOT BE A STATIC ASSERT — the
 * chunk count is decided at boot and arrives as an argument, so there is nothing at compile time to
 * assert about, and the refusal that stands in for one is `publish`'s own floor check.
 *
 * ⛳ CHUNK ZERO CANNOT BE PUT IN IT, AND NOT BY A RULE — BY ARITHMETIC. Its index is zero, and zero is
 * the carousel's word for an unwritten slot, so adding it is refused. The one chunk that must never go
 * home is the one whose name cannot be spoken here. */
static __device__ sys__carousel* sys__heap__zzprivate_free;

static __device__ sys__heap_node* sys__heap__zzprivate_heap_base_address;
/* How big the heap allocation pool is, in chunks and in nodes. ⭐ 
 *     NEITHER IS A BUILD CONSTANT, because neither is knowable when this file is compiled: the boot path
 * decides it and tells `publish`, which refuses anything under twice the block count in claimable
 * chunks. What is stored here is what it was told — the floor is a refusal and never a size. The nodes
 * are derived from the chunks once, here, so the two bounds checks that use them do not have to multiply.
 * */
static __device__ uint32_t    sys__heap__zzprivate_chunks;
static __device__ uint64_t    sys__heap__zzprivate_nodes;
static __device__ sys__heap_counters* sys__heap__zzprivate_counters;

/* ── COUNTING WHAT DIES, WHICH IS OFF UNLESS SOMEBODY ASKS FOR IT ────────────────────────────────────
 * A count reaching zero is the one event in an object's unwinding that nothing else records. What
 * happens after it is decided by kind, and by then the number that got us there is gone — so unless it
 * is counted here, "how many objects ended" can only be answered by walking the heap and inferring it
 * from what is absent.
 *
 * ⛔ ONE COUNT IS A COMPARE-AND-SWAP ON A SINGLE WORD FOR THE WHOLE MACHINE, and every block aims at the
 * same address — so the cost is contention rather than arithmetic, and a CAS pays for contention twice:
 * it serialises like an atomic add would, and it RETRIES when it loses. `REASONED` and not `MEASURED`:
 * the premise is that the loop is contended at all, and nothing has counted how often this CAS loses.
 * That price
 * is what the flag is for — worth paying when something is wrong and worth nothing the rest of the time.
 * ⛳ AND IT IS A RETRY LOOP WHERE ONE INSTRUCTION WOULD DO. The seam publishes `sys__silicon__add_u32`,
 * whose own note says it is there precisely because `cas_u32` alone makes every counter a retry loop, and
 * `sys__heap__zzpackage_take_up` further down is written on it. This counter is not, and what pays for
 * that is that it ships disarmed — the loop is on the arming's bill and not on the allocator's.
 * ⇒ ★ WHOEVER ARMS IT IS THE ONE WHO SHOULD MOVE IT, and moving it is one line.
 *
 * ⛳ AND IT IS THE ONLY PER-OBJECT TALLY, WHICH IS WORTH KNOWING BEFORE ADDING ANOTHER. A tally of holds
 * taken and given up would restate a claim the heap already makes about itself — two identical cycles
 * must leave the same number of nodes live — and makes from the counts that decide deallocation.
 * ⇒ ★ AN INSTRUMENT THAT RESTATES WHAT THE MECHANISM ALREADY PROVES IS A SECOND THING TO KEEP TRUE.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
#ifndef SYS__HEAP__COUNT_DEALLOCATIONS
#define SYS__HEAP__COUNT_DEALLOCATIONS 0
#endif

/* ⛳ THE ADD IS A CAS LOOP AND NOT THE SEAM'S ONE-INSTRUCTION ADD, which is the shape `cas_u32` forces
 * rather than a shape anything here wants: `sys__silicon__add_u32` is published, and
 * `sys__heap__zzpackage_take_up` BELOW is written on it. What keeps this one a loop is that it is
 * disarmed by default, which the paragraph above prices. An `atomicAdd` spelled straight into this file
 * would be a vendor call in package code either way — the seam is the only door.
 * It has no refusal, because a counter has no zero to guard
 * — it climbs until it wraps, and a wrapped diagnostic is a diagnostic and not a fault. */
static __device__ inline void sys__heap__zzprivate_count(uint32_t which) {
    if (sys__heap__zzprivate_counters == 0 || which >= SYS__HEAP__COUNTER_N) return;
    unsigned int* slot = &sys__heap__zzprivate_counters->at[which];
    unsigned int cur = *slot;
    for (;;) {
        const unsigned int old = sys__silicon__cas_u32(slot, cur, cur + 1u);
        if (old == cur) return;
        cur = old;
    }
}

static __device__ inline void sys__heap__zzpackage_record_deallocation(void) {
}

/* How many nodes the heap allocation pool holds — its size and not its occupancy, so it is the bound an
 * offset is checked against and never a measure of what is live. The object tier asks rather than reading
 * a constant because it has no business knowing how the number was arrived at. */
static __device__ inline uint64_t sys__heap__zzpackage_total_nodes(void) {
    return sys__heap__zzprivate_nodes;
}

/* Whether the heap base address has been published. Everything that reaches for it asks this first. */
static __device__ inline bool sys__heap__zzpackage_published(void) {
    /* ⛳ THE BASE ADDRESS ALONE, because it is the only thing publishing sets that the rest of the
     * package cannot do without. Where the blocks' bases live is the system register's answer, and the
     * register stands up AFTER this — so asking about it here would make the heap unpublished during
     * the window in which the register is being built out of it. */
    return sys__heap__zzprivate_heap_base_address != 0;
}

/* ── THE TWO WAYS BETWEEN AN ADDRESS AND AN OFFSET ───────────────────────────────────────────────────
 * ⚖ ARCHITECT: *"i think we should have two sys methods called sys__heap__offset(address) and
 * sys__heap__object_full_address(heap_offset) to improve legibility."*
 *
 * An offset is what a REFERENCE carries, because a value has to survive being copied into a cell that
 * some other block will read, and an address does not. A pointer is what CODE wants, and these two are
 * where the tree crosses between them.
 *
 * ⛔ THEY ARE NOT THE ONLY PLACES THE HEAP BASE ADDRESS IS NAMED, AND SAYING SO WOULD BE THE COMFORTABLE
 * VERSION. Three others do arithmetic on it directly: `chunk_of` compares against it to reject an address
 * below the heap allocation pool, and both `from_object_head` and the two places a chunk index becomes an address mask or
 * stride from it to reach a CHUNK. That is the shape of the exception rather than an oversight — a chunk
 * head sits at the START of a chunk rather than past its header, so `object_full_address` would name it
 * something it is not, which is
 * why the one that lands on a heap chunk is a verb of its own and says as much where it is defined.
 *
 * ⛳ THEY ARE PUBLIC FOR THE REASON `sys__heap_node__head` IS: a caller holding an offset is not doing
 * anything private by turning it into a pointer, and refusing to say how would only mean everyone
 * writes the subtraction out again.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline uint64_t sys__heap__offset(const sys__heap_node* address) {
    return (uint64_t)(address - sys__heap__zzprivate_heap_base_address);
}
static __device__ inline sys__heap_node* sys__heap__object_full_address(uint64_t heap_offset) {
    return sys__heap__zzprivate_heap_base_address + heap_offset;
}

/* ── WHAT THIS BLOCK IS WORKING WITH, REACHED WITHOUT BEING HANDED IT ────────────────────────────────
 * The allocator finds the asking block's computing base rather than taking one, and the reason is where
 * the allocator SITS and not what the language can say. A published verb is handed a base — the opcode
 * shape puts it first, and `clone` spends it carving a copy from the asking block's own chunk. But `make`
 * is reached from many call sites and not one of them is a verb: a stack or a stack chunk growing, a
 * list, a list chunk or a sublist being made, bindings being placed, an error being built, a dictionary
 * or its tree being made, an object being placed. Threading a parameter down every intermediate call,
 * to serve the paths that happen to begin at an opcode, buys nothing on the rest.
 * ⇒ ★ THE VALUE IS A PROPERTY OF THE BLOCK, AND A BLOCK IS THE ONE THING EVERY ONE OF THOSE PATHS HAS
 * IN COMMON. Asking the seam which block is asking is cheaper than carrying the answer down to it.
 *
 * ⭐ AND THE ANSWER IS FIXED FOR AS LONG AS THE BLOCK RUNS. What moves is the chunk the base names, and
 * that is a word inside it; the pointer itself is loop-invariant. On the host a block is one runner
 * thread, `sys__silicon__block_id` is that thread's own id, and the allocator keeps the base it found in
 * `SYS__SILICON__BLOCK_LOCAL` storage — `static __thread` — so each runner sees only its own and nothing
 * has to be an argument. `ASSUMED`, not measured: that a per-runner `__thread` slot is the cheapest home
 * for it on the host; a host rate A/B against passing it would settle it.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__heap_node* sys__heap__zzpackage_computing_base(uint32_t block_id) {
    /* ⭐⭐ THE REGISTER IS ASKED WHERE THE BASES ARE, AND THE ANSWER IS AN OFFSET TO ADD TO. Row zero
     * holds the offset of one allocation holding them end to end, so block N's slot is that offset
     * plus N — which is what makes this an addition rather than a lookup per block.
     * ⛳ THE SLOT IS READ DIRECTLY RATHER THAN THROUGH THE ARRAY'S OWN GET, and the reason is what a get
     * does: it hands out a HOLD, so a caller owes a release. That is right for a row a program reads
     * and wrong on the path to every allocation, where it would put a retain and a release either side
     * of an addition. What is being read is a plain number the boot placed, which nothing counts.
     * ⚠ AND THE COST OF THE ROW READ ITSELF IS UNMEASURED; the remembered base in `make` keeps it off
     * the per-allocation path, and a program worth timing is what would price it. */
    const sys__heap_node where = sys__system_register__get(SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY);
    if (where.dtype != SYS__KIND__VALUE_INT || where.args[0] == 0ull) return (sys__heap_node*)0;
    const uint64_t at = sys__heap__object_full_address(where.args[0] + (uint64_t)block_id)->args[0];
    return (at == 0ull) ? (sys__heap_node*)0 : sys__heap__object_full_address(at);
}

/* Stand the allocator up, from the boot path and nowhere else. Takes the four things the boot path owns —
 * the memory, how many chunks it holds, the counters, and how many nodes past those chunks are reserved
 * for storage the boot owns and no block does — and leaves every chunk counted zero with all but chunk
 * zero offered to the free ring.
 * ⛳ EVERY PATH THAT RETURNS FALSE LEAVES NOTHING PUBLISHED, which is the property the rest of the
 * package rests on: `published()` is the one question its verbs ask, and a half-stood heap would answer
 * it yes. */
static __device__ inline bool sys__heap__publish(sys__heap_node* heap_base_address,
                                                 uint32_t chunks,
                                                 sys__heap_counters* counters,
                                                 uint64_t reserved_nodes) {
    if (heap_base_address == 0 || chunks == 0u)
        return false;
    /* ⛔ A HEAP ALLOCATION POOL TOO SMALL FOR THE LAUNCH IS A LAUNCH THAT DOES NOT START. Every block needs
     * two chunks before it can do anything — one provider for what lasts and one for what churns — and the
     * so a heap allocation pool under twice the grid width cannot serve every block, and WHICH block it
     * fails is decided by arrival order rather than by anything a reader could predict. One branch here
     * turns that into a refusal at boot.
     * ⛳ AND EVERY CLAIMANT IS A BLOCK OR LIVES INSIDE ONE, so this is the whole of the sum. The system
     * register rides with block zero and carves from its providers, which the count already includes.
     *
     * ⭐ AND THE GRID WIDTH IS ASKED OF THE SILICON RATHER THAN OF THE CALLER, WHICH IS THE WHOLE VALUE OF
     * THE CHECK. The count above is what whoever allocated the heap allocation pool BELIEVES; the width
     * below is what the launch ACTUALLY IS. Asking one party for both would compare a number against
     * itself.
     *
     * ⛔ AND IT IS CLAIMABLE CHUNKS THAT ARE COUNTED, NOT CHUNKS. Chunk zero never enters the ring — the
     * pass below skips it — so a heap allocation pool of n hands out n-1, and comparing against n accepts
     * a heap allocation pool that is one short of serving its own launch. The subtraction is the
     * difference between a check that is nearly right and one that is right. */
    const uint64_t claimable = (uint64_t)chunks - 1ull;      /* `chunks` is non-zero: refused above */
    if ((uint64_t)sys__silicon__block_count() * 2ull > claimable) {
        sys__fault__raise(0ull, SYS__HEAP__FAULT_TOO_SMALL);
        return false;
    }
    /* ⛔ AND WHATEVER BASE THIS BLOCK WAS REMEMBERING STOPS MEANING ANYTHING HERE. A remembered base
     * points into the pool it was found in; a new pool makes it a plausible address into somebody else's
     * memory rather than an obviously wrong one. Emptying it where the pool is decided is what keeps any
     * caller from having to know that. `MEASURED`: six host checks about faults failed without this,
     * because the harness publishes more than once and a base outlived the heap it belonged to. */
    sys__heap__forget_base();
    sys__heap__zzprivate_heap_base_address = heap_base_address;
    sys__heap__zzprivate_chunks            = chunks;
    /* ⭐⭐ TWO BOUNDARIES, NOT ONE, AND CONFLATING THEM IS WHAT THIS PARAMETER PREVENTS. What may be
     * HANDED OUT is the chunks; what may be ADDRESSED is everything the base covers. They differ by the
     * room that outlives every chunk and belongs to no block — the boot's own storage, which must carry a
     * real offset and must never reach the free ring.
     * ⇒ ★ AN OFFSET IS A PROPERTY OF THE ALLOCATION; BEING HANDED OUT IS A PROPERTY OF THE POOL.
     * ⛳ AND THE RESERVATION IS COUNTED HERE AND NOWHERE ELSE: the pass below fills the ring from the
     * CHUNKS alone, so nothing has to remember to skip the tail — it was never offered. A caller that
     * reserves nothing passes zero and every number is what it was.
     * ⛳ THE ALLOCATOR IS TOLD, NOT ASKED, and it does not learn what the room is for. Whatever the boot
     * puts there is the boot's business; what this needs to know is only how far the addresses run. */
    sys__heap__zzprivate_nodes             = (uint64_t)chunks * (uint64_t)SYS__HEAP__CHUNK_NODES
                                             + reserved_nodes;

    /* ⛔ AND THE HEAP ALLOCATION POOL HAS TO BE MADE FREE, WHICH IS THE ONE PRICE OF A CHUNK CARRYING ITS
     * OWN COUNT. The count lives in the heap chunk's first node, so memory nobody has written reads as
     * OCCUPIED by whatever bytes happen to be there and the allocator finds a full heap allocation pool.
     * This function is therefore what makes an unwritten heap allocation pool usable, which is a
     * load-bearing property of it and not a detail.
     *
     * ⛳ IT IS ONE WORD PER CHUNK, ONCE, ON A PATH THAT ALREADY RUNS ONCE — one strided store per chunk
     * against a heap allocation pool of a few megabytes. The whole chunk is NOT cleared and does not need
     * to be: everything else in a heap chunk is written by whoever claims it. */
    /* ⛳ AND PUBLISHING TWICE IS A THING A HARNESS DOES, so the previous ring goes back before a new one is
     * taken. Without it a suite that resets more times than the heap allocation pool has heap chunks
     * empties it, and the failure would arrive as NO_ENTRY somewhere unrelated, long after the reset that
     * caused it.
     * ⛳ THE HARNESS PASSES THAT THRESHOLD MANY TIMES OVER, and the count is not written here because it
     * moves every time a section is added.
     *
     * ⛔ THE DRAIN IS NOT TIDINESS: `dispose` REFUSES a ring with things still in it, and there is no
     * reset — `take` in a loop is the only way a carousel empties. So this is what a republish costs.
     *
     * ⛔⛔ AND WHAT MAKES DISCARDING THE INDICES SAFE IS A PRECONDITION ON THE CALLER, WHICH IS WORTH
     * STATING BECAUSE NOTHING HERE CAN CHECK IT. Publishing is a BOOT act: no block is running, so no
     * heap chunk is held, so every chunk in the pool is free whatever the ring happens to say. That is
     * why the values taken out here can be thrown away and the loop below can simply offer all of them —
     * it is not restoring a state, it is asserting the only state a boot can be in.
     * ⇒ ⛔ CALLED MID-RUN IT WOULD DO THE OPPOSITE OF WHAT IT LOOKS LIKE: every chunk would go back on the
     * ring including the ones blocks are carving from, and two claimants would be handed the same one.
     * The precondition is the whole of the protection; there is no guard, and a guard would need to know
     * what is running, which this file cannot ask. */
    if (sys__heap__zzprivate_free != 0) {
        uint32_t stale;
        while (sys__carousel__take(sys__heap__zzprivate_free, &stale)) { }
        sys__carousel__dispose(sys__heap__zzprivate_free);
    }
    /* ⭐ THE RING IS SIZED PAST THE HEAP ALLOCATION POOL, NOT ONTO IT, and it is sized from the same number
     * the heap allocation pool is. A ring of n holds n-1, and the most that can ever be waiting is every
     * chunk but the one a block is carving from — so `slots >= chunks` is what this heap allocation pool
     * actually requires. Rounding to the first power of two ABOVE that gives more, and what the surplus
     * buys is distance: the ring runs at most `chunks - 1` against a ceiling strictly above it, so the
     * carousel's FULL refusal is unreachable from here rather than merely untaken. The argument in full is
     * at the top of `carousel__header.cuh`. */
    sys__heap__zzprivate_free = sys__carousel__create(SYS__CAROUSEL__SLOTS_ABOVE(chunks),
                                                      SYS__CAROUSEL__U32);
    if (sys__heap__zzprivate_free == 0) {
        /* ⛔ THE OLD RING IS ALREADY GONE BY HERE, so what is left is not a heap that lost its free list —
         * it is not a heap. Clearing the two words `published()` asks about says exactly that, and every
         * verb then refuses with NOT_PUBLISHED instead of reaching into a null carousel.
         * ⛳ THE ALTERNATIVE IS QUIETER AND WORSE: leave them and a failed republish answers `published()`
         * TRUE while holding no ring, so the first claim after it walks into the null. Ten call sites ask
         * that question and none of them asks a second one. */
        sys__heap__forget_base();
        sys__heap__zzprivate_heap_base_address = 0;
        return false;
    }
    /* ⛳ ONE PASS ZEROES AND OFFERS, WHICH IS WHERE THE POOL'S CONTENTS ARE DECIDED. A carousel is a LIST
     * and not a search, so what is in the pool is exactly what was put there — and the one moment that
     * already walks every chunk is this one, so it is where they go on.
     * ⛔ CHUNK ZERO IS SKIPPED HERE RATHER THAN REFUSED LATER, AND IT IS BLOCK ZERO'S — the block the
     * boot path starts on, and the one the system register rides with. It is still zeroed, being memory
     * like the rest, and it simply never becomes claimable, so nothing can be handed it twice.
     * ⛳ AND THE ADD CANNOT FAIL HERE, WHICH IS WHY ITS ANSWER IS DROPPED: the ring is sized ABOVE the
     * chunk count on purpose, so a pass that offers at most `chunks - 1` of them cannot fill it. The cast
     * is saying "this has no failing case" and not "I am ignoring one". */
    for (uint32_t i = 0u; i < chunks; ++i) {
        SYS__HEAP_OBJECT__COUNT(&heap_base_address[(uint64_t)i * (uint64_t)SYS__HEAP__CHUNK_NODES]) = 0u;
        if (i != 0u) (void)sys__carousel__add(sys__heap__zzprivate_free, i);
    }

    sys__heap__zzprivate_counters        = counters;
    return true;
}

/* ── AND STANDING THE ALLOCATOR DOWN ─────────────────────────────────────────────────────────────────
 * Give the free ring back and forget where the pool was, so that every verb in this file refuses instead
 * of reaching into memory that has been handed back.
 *
 * ⭐⭐ IT IS NOT THE OPPOSITE OF `publish` AND IT IS NOT MEANT TO BE. Publishing decides what the pool IS;
 * this says there is none. Nothing in the pool is walked, nothing is released and no chunk
 * goes home — the whole pool is about to stop existing, so an allocator tidying its contents first would
 * be arranging furniture in a house being demolished.
 *
 * ⛔ THE RING IS THE ONE THING THAT MUST GO, because it is the one thing that is NOT in the pool: its room
 * comes from the seam and outlives every free the boot path does. Draining it first is not tidiness —
 * `dispose` refuses a ring with things still waiting, and `take` in a loop is the only way one empties.
 *
 * ⛳ AND AFTERWARDS `published()` ANSWERS NO, WHICH IS THE PROPERTY WORTH HAVING. A machine that has been
 * shut down and is asked to allocate now refuses by name instead of carving out of somebody else's
 * memory at a plausible-looking offset. */
static __device__ inline void sys__heap__zzengine_retire(void) {
    if (sys__heap__zzprivate_free != 0) {
        uint32_t stale;
        while (sys__carousel__take(sys__heap__zzprivate_free, &stale)) { }
        sys__carousel__dispose(sys__heap__zzprivate_free);
        sys__heap__zzprivate_free = 0;
    }
    sys__heap__forget_base();
    sys__heap__zzprivate_heap_base_address = 0;
    sys__heap__zzprivate_chunks            = 0u;
    sys__heap__zzprivate_nodes             = 0ull;
    sys__heap__zzprivate_counters          = 0;
}

/* ⭐ WHICH HEAP CHUNK AN ADDRESS IS IN — the one lookup in address space that still needs the heap base
 * address, and the reason it is not a mask.
 *
 * Clearing low bits finds a chunk's BEGINNING; it cannot find a chunk's NUMBER, because a number is
 * measured from the heap base address and a mask has no idea where the heap base address is. The number is
 * what indexes the free list, so this is the conversion the heap allocation pool is built on. Anything
 * outside the heap allocation pool is NIL rather than an index into somebody else's memory. */
/* Where a heap chunk begins, by its number. The inverse of `chunk_of`, and the two are the only places
 * that turn a chunk's index into an address — everything else already holds one.
 * ⛳ IT IS THE PACKAGE'S OWN ARITHMETIC AND NOT A BORROWED ONE, which matters more than one line usually
 * would: the stride IS the geometry, so a version compiled against a different chunk size answers
 * plausible addresses that are wrong by the ratio between them, and every one of them lands inside the
 * pool where no bounds check will object. */
static __device__ inline sys__heap_node* sys__heap__zzprivate_chunk_base(uint32_t idx) {
    return sys__heap__zzprivate_heap_base_address + (uint64_t)idx * (uint64_t)SYS__HEAP__CHUNK_NODES;
}

static __device__ inline uint32_t sys__heap__chunk_of(const sys__heap_node* at) {
    if (!sys__heap__zzpackage_published() || at < sys__heap__zzprivate_heap_base_address) return SYS__HEAP__NO_CHUNK;
    const uint64_t off = sys__heap__offset(at);
    if (off >= sys__heap__zzprivate_nodes) return SYS__HEAP__NO_CHUNK;
    return (uint32_t)(off / (uint64_t)SYS__HEAP__CHUNK_NODES);
}

/* ⭐ ALMOST NOTHING POINTS AT A CHUNK, AND THE TWO EXCEPTIONS BOTH POINT AT A LIVE ONE. The computing base
 * names the chunk its block is carving from, and a stack chunk names the one before it — both of those are
 * chunks in USE. What nothing points at is a chunk that has been moved away from: it is never carved into
 * again, so nothing needs to find it. Whoever is inside one arrives by masking their own address, and
 * nobody outside has any reason to.
 *
 * ⇒ That is what makes a chunk cheap to give back: by the time its count reaches zero, the two references
 * that could exist are gone — the base has moved on and the stack has popped — so there is nothing to
 * repair before it goes. */

/* Where the next object goes. A chunk that has never been written reads zero here, which is below the
 * header and therefore cannot be a real placement, so a fresh heap chunk answers "nothing yet" without
 * anyone having to initialise it — the same reason the heap allocation pool can hand out memory it has not
 * touched. */
static __device__ inline uint64_t sys__heap_chunk__zzprivate_watermark(const sys__heap_node* chunk) {
    const uint64_t w = chunk[SYS__CHUNK__HEAD].args[SYS__CHUNK__WATERMARK];
    return (w < (uint64_t)SYS__CHUNK__FIRST) ? (uint64_t)SYS__CHUNK__FIRST : w;
}

/* Only the payload moves. The tag says what the heap chunk IS and is written when its role changes, not
 * every time something is placed in it. */
static __device__ inline void sys__heap_chunk__zzprivate_set_watermark(sys__heap_node* chunk, uint64_t at) {
    chunk[SYS__CHUNK__HEAD].args[SYS__CHUNK__WATERMARK] = at;
}

/* How much room is left. A request is placeable when it is not larger than this. */
static __device__ inline uint64_t sys__heap_chunk__zzprivate_room(const sys__heap_node* chunk) {
    const uint64_t w = sys__heap_chunk__zzprivate_watermark(chunk);
    return (w >= (uint64_t)SYS__HEAP__CHUNK_NODES) ? 0ull : ((uint64_t)SYS__HEAP__CHUNK_NODES - w);
}

/* ── WHAT A CHUNK IS ─────────────────────────────────────────────────────────────────────────────────
 * A chunk fresh from the heap allocation pool is memory. Giving it the header a block carves from makes it
 * a supplier, and it is turned back into memory before it goes home. The role is written in the first
 * node's tag, so a caller can ask rather than assume — and the chunks it would otherwise land on by
 * mistake are the ones holding somebody else's live data.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* Whether a block is carving from this chunk. ⛳ TWO TAGS ANSWER YES AND THAT IS NOT A LOOSENING: they
 * are the same state, and what they differ about is WHERE THE HEAD IS. `CHUNK_PROVIDER` means the node
 * in the pool is the one to read; `HEAP_CHUNK_ACTIVE` means a block has taken that head into its own
 * storage and the copy here is behind. Neither says anything about whether the chunk is a provider, so
 * neither belongs in the answer to this question. ⇒ ⛔ WHOEVER NEEDS THE OTHER QUESTION — is the head
 * here worth reading — must ask it separately, and `zzprivate_place` below is the one that does. */
static __device__ inline bool sys__heap_chunk__zzprivate_is_provider(const sys__heap_node* chunk) {
    return sys__heap_object__is_type_from_head(&chunk[SYS__CHUNK__HEAD], SYS__KIND__CHUNK_PROVIDER)
        || sys__heap_object__is_type_from_head(&chunk[SYS__CHUNK__HEAD], SYS__KIND__HEAP_CHUNK_ACTIVE);
}

/* Whether this chunk's head is somewhere else. Anything reading the watermark out of the pool has to ask,
 * because for a cached chunk that number stopped moving when the caching started. */
static __device__ inline bool sys__heap_chunk__zzprivate_is_active(const sys__heap_node* chunk) {
    return sys__heap_object__is_type_from_head(&chunk[SYS__CHUNK__HEAD], SYS__KIND__HEAP_CHUNK_ACTIVE);
}

/* Write the reference a chunk carries — the base of the block carving here, which the caller names by
 * slot. It is given as a node offset into the heap rather than as an address, because that is what the
 * counting is kept against and what stays meaningful wherever the pool is mapped. A null is written as
 * a null tag rather than as a zero offset, so "nobody is carving here" and "the block at the start of
 * the heap is carving here" cannot be confused.
 *
 * A claimed chunk is not zeroed, so a recycled one arrives holding whatever its last tenant left. The
 * cell is therefore always written and never assumed, exactly as the stack does with its own header. */
static __device__ inline void sys__heap_chunk__zzprivate_set_ref(sys__heap_node* chunk, uint32_t slot,
                                                           uint64_t offset, bool present) {
    chunk[slot].dtype   = present ? SYS__KIND__OBJECT_REFERENCE : SYS__KIND__VALUE_NULL;
    chunk[slot].args[0] = present ? offset : 0ull;
    chunk[slot].args[1] = present ? (uint64_t)SYS__HEAP__CHUNK_NODES : 0ull;
}

static __device__ inline bool sys__heap_chunk__zzprivate_has_ref(const sys__heap_node* chunk, uint32_t slot) {
    return chunk[slot].dtype == SYS__KIND__OBJECT_REFERENCE;
}

static __device__ inline uint64_t sys__heap_chunk__zzprivate_ref(const sys__heap_node* chunk, uint32_t slot) {
    return chunk[slot].args[0];
}

/* Give a freshly claimed heap chunk its header: empty, owned by nobody in particular yet, indexing
 * nothing. The caller names the owner because the allocator does not know who asked.
 *
 * ⛔⛔ AND IT CLEARS THE LOCK, WHICH IS EASY TO LEAVE OUT AND EXPENSIVE TO LEAVE OUT. The lock word sits
 * inside the header, below where the first tenant begins, so no later one ever writes over it — if it is
 * not cleared here it holds whatever the heap allocation pool held at boot, for the life of the process,
 * and the chunk is DECLARED lockable while being PERMANENTLY unlockable.
 *
 * ⛳ AND THE FAILURE THAT MAKES IS THE WORST-SHAPED ONE AVAILABLE. The lock is bounded, so it does not
 * hang: it spins `LOCK_TRIES` times, backs off, and raises. That is a REFUSAL UNDER APPARENT CONTENTION
 * on a lock nobody else holds — which reads exactly like the thing it is not, and would be chased in the
 * contention rather than in the initialiser.
 *
 * ⇒ ⭐ AND THE PLACE IS HERE, NOT IN `publish`. The count is `publish`'s because the CLAIM READS it —
 * it has to be right before anyone arrives. Every other header field is written by whoever claims,
 * because a claimed chunk is not zeroed and the claimer is the one initialising; the lock is one of
 * those. `make` has said so on its own head for as long as it has existed, one line apart from this
 * one: *"the chunk was not zeroed; every head field is written"*. This is that rule, applied to the
 * head that was not thought of as a head. */
static __device__ inline void sys__heap_chunk__zzprivate_init(sys__heap_node* chunk, uint64_t owner_offset,
                                                              bool owned) {
    chunk[SYS__CHUNK__HEAD].dtype = SYS__KIND__CHUNK_PROVIDER;
    chunk[SYS__CHUNK__HEAD].args[SYS__CHUNK__LOCK] = 0ull;
    sys__heap_chunk__zzprivate_set_watermark(chunk, (uint64_t)SYS__CHUNK__FIRST);
    sys__heap_chunk__zzprivate_set_ref(chunk, SYS__CHUNK__REQUESTOR, owner_offset, owned);
}

/* And the way back: into the bin, with the watermark wound to zero and the tag saying plain memory again.
 *
 * ⚖ **ARCHITECT:** *"the watermark is used only in the write phase, we leave it there and it only
 * increases. once the chunk\'s objects reference counter goes to 0 it goes straight in the recycling bin
 * with the watermark reset to 0."*
 *
 * ⛳ NEITHER WRITE IS LOAD-BEARING, AND IT IS WORTH KNOWING WHICH KIND OF LINE THIS IS. `init`
 * rewrites every field a chunk has at the moment it is claimed, so nothing downstream reads what is
 * left here. It is hygiene: a chunk sitting in the bin still tagged as a supplier, with a watermark from
 * its last tenant, is a confusing thing to meet in a dump, and the two stores cost nothing.
 * ⛳ AND THEY CANNOT RACE A CLAIMER, whatever runs beside this: both come before the carousel add below,
 * and until that add no runner can find the chunk. Were they ever moved after it, the fix would be to
 * remove them, not to lock them, because the claimer is the one initialising. */
static __device__ inline void sys__heap_chunk__zzprivate_recycle(sys__heap_node* chunk) {
    chunk[SYS__CHUNK__HEAD].dtype = SYS__KIND__HEAP_CHUNK;
    sys__heap_chunk__zzprivate_set_watermark(chunk, 0ull);
    /* ⭐ AND THIS IS THE MOMENT IT BECOMES AVAILABLE AGAIN. The carousel is a list, so going home is an
     * act and not something the count says. Putting it here rather than at the two call sites is
     * the whole reason this function exists: one caller is the block moving off a chunk it filled, the
     * other is the last thing inside one dying, and neither should have to know about a free list. */
    (void)sys__carousel__add(sys__heap__zzprivate_free, sys__heap__chunk_of(chunk));
}

/* ── WHAT A CHUNK DOES WHEN ITS LAST OCCUPANT GOES ───────────────────────────────────────────────────
 * The chunk's own row in the kind list, and it is the THIRD answer to what giving the room back means.
 * A plain object hands its granule back to the chunk it was carved from. The computing base hands nothing
 * back and raises, because a release reaching it is a mistake rather than an ending. A chunk has no chunk
 * to hand anything to — it IS the room — so what it owes is to stop being a supplier and go back on the
 * ring, which is what `recycle` does, reached here by kind.
 *
 * ⭐ AND IT HANDS NOTHING OVER, WHICH IS A FACT AND NOT AN OMISSION. What a release method returns is
 * whatever the dying thing was still holding, for the caller's loop to walk. A chunk reaching zero means
 * every occupant has already gone; there is nothing left in it by definition, and a chunk that still held
 * something could not have got here.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ __noinline__ void sys__heap_chunk__zzpackage_release_internal(sys__heap_node* head,
                                                                         uint64_t* releaser_stack) {
    (void)releaser_stack;                 /* a chunk is room, and room holds nothing that can be moved */
    if (head == 0) return;
    sys__heap_chunk__zzprivate_recycle(head);
}

/* Place `nodes` worth of object and return where it went, or zero when it does not fit. Zero is below the
 * header and so cannot be a real placement, which is what lets it mean refusal without a second output.
 *
 * Nothing is searched and nothing is reused: the watermark is where the next one goes. A gap left by an
 * earlier release stays a gap. */
static __device__ inline uint64_t sys__heap_chunk__zzprivate_place(sys__heap_node* chunk, uint64_t nodes) {
    /* ⛔ AND IT REFUSES A CACHED HEAD RATHER THAN READING ONE. Every line below works off the watermark in
     * the pool, which for a cached chunk is where the carving STARTED and not where it has got to — so
     * this would hand out room that is already in use, and would do it without a fault. The one caller
     * is `start_block`, placing by hand into a chunk it has just initialised; the refusal is here
     * for the next one. */
    if (nodes == 0ull || !sys__heap_chunk__zzprivate_is_provider(chunk)) return 0ull;
    if (sys__heap_chunk__zzprivate_is_active(chunk)) return 0ull;
    const uint64_t at = sys__heap_chunk__zzprivate_watermark(chunk);
    if (nodes > sys__heap_chunk__zzprivate_room(chunk)) return 0ull;
    sys__heap_chunk__zzprivate_set_watermark(chunk, at + nodes);
    return at;
}


/* Where a chunk keeps track of how many things are living in it: the same word every object keeps its count
 * in, on the node the chunk begins with. Defined here because the allocator below is its first caller — the
 * claim IS the first hold, so the claim writes this word. */
static __device__ inline unsigned int* sys__heap_chunk__zzprivate_count_word(const sys__heap_node* chunk) {
    return (unsigned int*)&SYS__HEAP_OBJECT__COUNT(&((sys__heap_node*)chunk)[SYS__CHUNK__HEAD]);
}

/* ── THE ALLOCATOR ────────────────────────────────────────────────────────────────────────────────────
 * Take one free heap chunk and return its index, or a nil index when there is none.
 *
 * ⭐⭐ IT ASKS A LIST RATHER THAN SEARCHING THE HEAP ALLOCATION POOL, AND THE REASON IS BIAS BEFORE SPEED.
 * A search from a fixed origin only moves when it is BLOCKED, so a chunk freed and re-claimed comes back
 * as the same chunk every time — one region reused continuously while the rest is never touched, and a
 * cost O(n) in busy chunks on top. Taking from the tail of the carousel cannot do that: a chunk handed
 * back goes behind everything already waiting and cannot come out again until they have.
 *
 * ⭐ THE COMPARE-AND-SWAP ON THE COUNT IS A CHECK RATHER THAN THE MECHANISM.
 * ⚖ ARCHITECT: *"we can also keep it but it is an extra check that it buys for free."* It is: the count is
 * a word this function has to write anyway, so making the write a swap from zero costs nothing and asserts
 * the thing the carousel is trusted for — that what comes off it is genuinely unheld. If it ever is not,
 * the two accountings have diverged, and a claim that proceeded would hand out live memory.
 *
 * ⛳ ONE THREAD, and the guard is not here because it is not this function's to hold — every verb in the
 * package runs under the same condition and the election happens above `sys` entirely. It is worth naming
 * HERE because this is where breaking it costs most: a per-thread claim takes one chunk per thread and
 * empties the heap allocation pool on the first allocation, where breaking it elsewhere costs one wrong
 * count.
 *
 * The name marks it as reachable from THIS FILE and no further — `zzprivate_` is the tightest of the
 * three tiers the gate knows, tighter than the `zzpackage_` the rest of the package is reached by, and
 * the only two callers are the carve and `sys__heap__start_own_block`, both below. Chunks are claimed on
 * the allocator's own paths, never by the containers built on them and never by the code that uses those
 * containers, and the device tier is written in the C subset, which has no access control — so the marker
 * in the name is what a reader has at a call site, and the gate is what makes it more than a request.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline uint32_t sys__heap__zzprivate_claim_chunk(void) {
    if (!sys__heap__zzpackage_published()) return SYS__HEAP__NO_CHUNK;
    uint32_t idx = 0u;
    if (!sys__carousel__take(sys__heap__zzprivate_free, &idx)) {
        sys__heap__zzprivate_count(SYS__HEAP__COUNTER_CHUNK_REFUSED);
        return SYS__HEAP__NO_CHUNK;
    }
    sys__heap_node* taken = sys__heap__zzprivate_chunk_base(idx);
    if (sys__silicon__cas_u32(sys__heap_chunk__zzprivate_count_word(taken), 0u, 1u) != 0u) {
        /* The carousel offered a chunk somebody is inside. Nothing here can put it right — the entry is
         * already spent — and carrying on would hand out live memory, so this refuses and says why. */
        sys__fault__raise(0ull, SYS__HEAP__FAULT_NOT_FREE);
        return SYS__HEAP__NO_CHUNK;
    }
    sys__heap__zzprivate_count(SYS__HEAP__COUNTER_CHUNK_CLAIMS);
    return idx;
}

/* ── GIVING UP A REFERENCE ───────────────────────────────────────────────────────────────────────────
 * Give up one hold and answer with what is left. The decrement and the test that it reached a
 * particular value are one operation, which is the whole reason for the loop: written as a read, a
 * comparison and then a subtraction they are three, and two callers could both act on the same answer.
 *
 * A release at zero is refused rather than allowed to wrap. There is nothing left to give up, so the
 * request is a mistake somewhere further back, and turning a count of zero into four billion would make
 * whatever it guards permanent and the mistake invisible.
 *
 * It takes the word rather than an array and an index because the counts this package keeps are the same
 * operation on different storage, and writing it twice is how they would come to differ. That is also why
 * the name says the package rather than the file: the stack's own count lives in front of each stack
 * rather than in an array here, and it gives it up through this.
 *
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* And taking one, which is the same word moving the other way.
 *
 * ⭐ IT EXISTS BECAUSE THE DECREMENT'S OWN ARGUMENT APPLIES TO IT UNCHANGED. The paragraph above says the
 * counts this package keeps are the same operation on different storage, and that writing one of them out
 * twice is how two of them come to differ. Nothing about taking a hold makes it the exception: same word,
 * same storage, same reason to have one copy of it. Two copies of a thing nobody would ever change on
 * purpose is still the shape in which a change reaches one of them.
 *
 * It answers with the new count, which its callers are free to ignore, because the caller that cares
 * hands it straight back to whoever asked for the hold. */
static __device__ inline uint32_t sys__heap__zzpackage_take_up(unsigned int* slot) {
    /* ⭐ ONE ATOMIC ADD, NOT A RETRY. An increment has no reason to fail and nothing to re-read — see
     * `sys__silicon__add_u32`. It answers the new count, which is the old one the add hands back plus
     * the one we added. */
    return (uint32_t)(sys__silicon__add_u32(slot, 1u) + 1u);
}

static __device__ inline bool sys__heap__zzpackage_give_up(unsigned int* slot, uint32_t* left) {
    /* ⭐⭐ ONE INSTRUCTION, AND THE UNIQUENESS THE LOOP GAVE IS THE ATOMIC'S OWN. ⚖ ARCHITECT: *"what we
     * need to prevent is two decrements to come at the same time or we might have two readers of 0 and a
     * double deallocation — unless the atomic also returns the value, in which case we can remove the
     * lock."* It does return it. Exactly one caller can be handed an old value of 1, so exactly one can
     * see the count reach zero, and the teardown behind this can no more run twice than the hardware can
     * hand the same old value to two callers.
     *
     * ⛳ A DECREMENT IS AN ADD OF ALL-ONES, so the seam needs no second verb for it. Unsigned wrap is
     * defined and is exactly the arithmetic wanted.
     *
     * ⛔ AND IT REFUSES AT ZERO, WHICH IS WHY THIS IS NOT A BARE SUBTRACT. Releasing a count that
     * is already zero is a double release — a caller's bug — and this refuses it rather than
     * wrapping the count to four billion. The undo is what makes the refusal honest: the subtract has
     * already landed by the time we can see it should not have, so it is put back.
     * ⚠ BETWEEN THE TWO THE WORD READS 0xFFFFFFFF, AND THAT WINDOW OPENS ONLY ON A PATH THAT IS ALREADY
     * WRONG. Nothing correct reaches it. One block cannot observe it at all; when more than one runs, a
     * second block reading a bogus count during another's double-release is a second symptom of the
     * first bug rather than a new one. */
    const unsigned int was = sys__silicon__add_u32(slot, 0xFFFFFFFFu);
    if (was == 0u) { (void)sys__silicon__add_u32(slot, 1u); return false; }
    *left = (uint32_t)(was - 1u);
    return true;
}


/* A value that is nothing. It is what an empty container answers with, and what writing over a cell
 * leaves behind — one shape for "there is no value here", so a reader never has to know which absence it
 * is looking at.
 *
 * It is here rather than in whichever container needed it first because absence is not a property of
 * stacks. The count is zero and not one: nothing holds it, because there is nothing to hold. */
static __device__ inline sys__heap_node sys__heap_node__nothing(void) {
    sys__heap_node v;
    v.dtype    = SYS__KIND__VALUE_NULL;
    v.num_args = 0u;
    v.args[0]  = 0ull;
    return v;
}

/* The head of the allocation whose PAYLOAD begins here — the node that says what the thing is, how many
 * hold it, and holds its lock.
 *
 * ⛔ IT IS THE NODE BEFORE, AND THAT IS A NARROWER PROMISE THAN IT SOUNDS. A reference names the node
 * after a head, so subtracting one from a reference is exact. It is NOT a way to find the head from any
 * address inside an allocation: allocations are packed at whatever width their kind asked for, so there
 * is no arithmetic that recovers a beginning from an arbitrary interior node, and asking for one would
 * mean asking every allocation to be a power of two again.
 * ⇒ ★ EVERY CALLER IN THE TREE HOLDS A REFERENCE, which is what makes the subtraction sufficient — and
 * what makes a caller holding something else a bug this cannot detect. There is no check here because
 * there is nothing to check against: one node before an arbitrary node is another arbitrary node.
 *
 * ⛳ The head is SHARED between two tiers, which is why neither owns the whole node: `dtype`, the count and
 * the lock belong to the object layer, and the arguments after them belong to whatever kind this is. That
 * is also why the payload type does not describe the head — a stack chunk declaring that layout would be
 * declaring something it does not own, and two kinds could then disagree about it.
 *
 * ⭐ IT IS PUBLIC, AND `node` RATHER THAN `object` ON PURPOSE. Anything holding a REFERENCE may ask what
 * the thing behind it is, which is a question about MEMORY SHAPE — where an allocation begins — and not
 * about lifetime. The `sys__heap_object__*` verbs are the lifetime family: retain, release, set. This one
 * answers a different question and says so in its second word.
 *
 * ⛳ AND IT TAKES NO REFERENCE AND GIVES NONE. It hands back an address, not ownership, so a caller that
 * wants to KEEP what it found owes a retain — which is exactly what a wrapper exposing this to the
 * language would add, and why that wrapper is a different function rather than a flag on this one. */
static __device__ inline sys__heap_node* sys__heap_node__head(const void* inside) {
    return ((sys__heap_node*)inside) - 1;
}

static __device__ inline sys__heap_node* sys__heap_node__zzpackage_head(uint64_t offset) {
    return sys__heap_node__head(sys__heap__object_full_address(offset));
}

static __device__ inline sys__heap_carving* sys__heap_carving_at(uint32_t from) {
    SYS__SILICON__BLOCK_LOCAL sys__heap_carving remembered[SYS__COMPUTING_BASE__PROVIDERS];
    return &remembered[from < SYS__COMPUTING_BASE__PROVIDERS ? from : 0u];
}

/* ⭐⭐ HOW MANY THINGS HOLD THIS CHUNK, WHICH IS NOT THE SAME AS WHAT THE WORD IN IT SAYS. While a block
 * is carving here the holds it has taken are gathered in that block's own storage and the word carries
 * only the releases — so the word alone answers a question nobody asked, and answers it below zero. The
 * two halves are added here, which is the same sum the boundary performs and the reason this is the verb
 * to ask rather than the field to read.
 * ⛳ AND ONLY THE CARVING BLOCK CAN BE ASKED, which is not a limitation to work around: a block that is
 * not carving here sees a chunk whose word is already whole, and one that is carving elsewhere has no
 * business with this chunk's occupancy. The gathered half is reachable exactly where it is meaningful. */
static __device__ inline uint32_t sys__heap_chunk__zzprivate_occupants(const sys__heap_node* chunk) {
    if (chunk == 0) return 0u;
    uint32_t count = (uint32_t)*sys__heap_chunk__zzprivate_count_word(chunk);
    if (sys__heap_chunk__zzprivate_is_active(chunk))
        for (uint32_t from = 0u; from < SYS__COMPUTING_BASE__PROVIDERS; ++from) {
            const sys__heap_carving* carving = sys__heap_carving_at(from);
            if (carving->chunk == chunk) { count += carving->appends; break; }
        }
    return count;
}

/* ── WHAT GIVING THE ROOM BACK COSTS, AND WHY THERE ARE TWO PROVIDERS ────────────────────────────────
 * The room does not come back in pieces. Freeing an object gives up its chunk's hold and nothing else —
 * the watermark does not move, the space is not reused, and no neighbour is consulted. When the last hold
 * goes the WHOLE heap chunk goes home at once, which is the only moment any of that space becomes
 * available again.
 *
 * ⛳ THAT IS THE TRADE, AND IT IS ABOUT LIFETIME RATHER THAN SIZE. Allocations are packed: each begins
 * where the last ended, so a chunk fills to its own edge and what is left over at the end is at most one
 * object short of fitting. That is the whole of the size cost.
 * ⛔ THINGS ARE STRANDED BY LIFETIME, AND THAT IS THE ONE THAT GROWS WITHOUT BOUND — which is why there
 * are two providers rather than one, argued directly below.
 *
 * ⛔⛔ TWO PROVIDERS, AND THE SECOND ONE IS NOT A CONVENIENCE — IT IS WHAT KEEPS DRAIN-ONLY AFFORDABLE.
 *
 * A chunk goes back to the heap allocation pool only when EVERYTHING in it has died. So a chunk holding
 * one long-lived thing and fourteen short-lived ones never comes home, and the space of the fourteen stays
 * stranded behind a watermark that only rises.
 *
 * `MEASURED`, the selftest's churn scenario at its 256-node geometry: a stack object sitting through six
 * rounds of push-and-pop leaves the chunk it was made in at a watermark of 64 of 256 — itself and nothing
 * else, because the churn is carved elsewhere. ⚠ THE FIGURE IT IS MEASURED AGAINST IS NOT REPRODUCIBLE
 * HERE: the 256 of 256 — full, pinned by that one live thing — comes from a single-provider arrangement
 * the suite does not run, so what can be re-derived today is the 64 and not the comparison.
 *
 * The shipping geometry is DERIVED rather than measured: a chunk is as many granules as it has nodes over
 * the granule, the header strands the first, and the one-provider case pins all the rest behind one live
 * one. The count moves with the dial, so it is stated as a shape rather than as a number.
 *
 * ⇒ So what lasts and what churns are carved from different chunks. It does not remove stranding; it stops
 * PERMANENCE from pinning CHURN, which is the only version of it that grows without bound. The two are
 * told apart by the KIND, in its row, because a kind knows which it is — a stack lasts as long as the
 * name bound to it, its chunks last until the next pop. Nothing asks the caller, which is what makes the
 * split hold rather than merely be observed: putting a stack in the churn chunk is not a thing that can
 * be spelled.
 *
 * The chunk is found by arithmetic — chunks lie end to end at a size that is a power of two — so nothing
 * has to be stored to find the way back.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* An occupant arrives. ⛳ NOBODY OUTSIDE THIS PACKAGE CALLS THIS, AND THAT IS THE POINT OF THE PAIR: a
 * kind is written in terms of creating and releasing an OBJECT, and where that object sits is the
 * allocator's business. The one call is in `start_block`, which places the computing base by hand into a
 * chunk nothing has carved from yet. ⛳ THE CARVE DOES NOT COME THROUGH HERE: it gathers its holds in
 * the carving block's storage as it makes them, and the run is written down once when the block moves
 * off. */
/* Let go of one, on a count that is allowed to go below zero. ⛔ THE ORDINARY `give_up` REFUSES AT ZERO
 * AND IS RIGHT TO, because for anything else zero means nobody is holding it and a further release is a
 * mistake worth raising. A chunk being carved from is the exception the tag names: its increments are
 * gathered in the carving block's storage, so the number here carries only the DECREMENTS and passes
 * through zero on the way down in the ordinary case. Refusing there would raise on correct traffic and
 * lose the release.
 * ⛳ IT ANSWERS NOTHING, which is the other half. There is no "was that the last one" to be had from a
 * provisional number, and a caller that wanted one would be asking the question this tag exists to
 * defer. */
static __device__ inline void sys__heap_chunk__zzprivate_let_go_provisional(sys__heap_node* chunk) {
    unsigned int* slot = sys__heap_chunk__zzprivate_count_word(chunk);
    unsigned int cur = *slot;
    for (;;) {
        const unsigned int old = sys__silicon__cas_u32(slot, cur, cur - 1u);
        if (old == cur) return;
        cur = old;
    }
}

/* Write down a run of holds in one go. The arithmetic wraps, and that is not an accident being tolerated
 * — it is what makes the reconciliation mechanical: a count that went to -3 while three increments were
 * being gathered is `0xFFFFFFFD`, and adding 3 lands on 0 with no case to distinguish. */
static __device__ inline void sys__heap_chunk__zzprivate_add_run(sys__heap_node* chunk, uint32_t n) {
    if (n == 0u) return;
    unsigned int* slot = sys__heap_chunk__zzprivate_count_word(chunk);
    unsigned int cur = *slot;
    for (;;) {
        const unsigned int old = sys__silicon__cas_u32(slot, cur, cur + n);
        if (old == cur) return;
        cur = old;
    }
}

static __device__ inline void sys__heap_chunk__zzpackage_add(sys__heap_node* chunk) {
    if (chunk == 0) return;
    (void)sys__heap__zzpackage_take_up(sys__heap_chunk__zzprivate_count_word(chunk));
}

/* An occupant leaves, and the heap chunk goes home if it was the last one. Answers whether it did.
 *
 * ⭐ THE DECREMENT AND THE GOING-HOME ARE ONE VERB, AND SPLITTING THEM WOULD LEAVE HALF A HAZARD BEHIND.
 * The decrement and the test that it reached zero are fused because two callers both believing they were
 * last would hand the same chunk back twice. Handing it back is what follows that test — so leaving that
 * to the caller gives every caller a second line to remember, and the whole point of fusing the first two
 * was that remembering is not a mechanism.
 *
 * ⛳ AND THE ANSWER IS RETURNED, because a caller may want to know — the harness does — but nothing has
 * to ACT on it.
 *
 * ⛔⛔ AND CHUNK TRAFFIC IS COUNTED APART FROM OBJECT DEATHS, WHICH IS THE ONE PLACE "a chunk is an
 * object" STOPS BEING TRUE. It is true of the STORAGE — same word, same head, same `give_up`. It is not
 * true of the ACCOUNTING: an object is made and ended inside one cycle, while a heap chunk is claimed
 * from a pool that carries state across them, so a chunk coming back is not an object dying and the two
 * numbers answer different questions.
 * ⚠ `MEASURED` in one total: the second cycle reports 44 releases against the first's 42 — a real
 * difference in chunk reuse, and exactly the kind of noise that trains somebody to ignore a check.
 * ⇒ ★ SHARED STORAGE IS NOT SHARED SEMANTICS. Chunks keep `HEAP_CHUNK_CLAIMS` and `_REFUSED`; the object
 * side keeps only what died.
 *
 * ⛳ IT IS THE SAME `give_up` EVERY OTHER COUNT IN THIS PACKAGE USES — the one that REFUSES at zero rather
 * than wrapping. There is one counting, and one answer about what an underflow means. */
static __device__ inline bool sys__heap_chunk__zzpackage_release(sys__heap_node* chunk) {
    if (chunk == 0) return false;
    /* ⛔⛔ A CHUNK SOMEBODY IS CARVING FROM IS LET GO OF AND NEVER GIVEN BACK, and the reason is that the
     * number here cannot answer the question. Its increments are being gathered by the block that is
     * carving; what is written here is the decrements alone, so it reaches zero long before the chunk is
     * empty and would hand a live arena to the free ring. The reconciliation happens when that block
     * moves off, and only from that moment is zero an answer. */
    if (sys__heap_chunk__zzprivate_is_active(chunk)) {
        sys__heap_chunk__zzprivate_let_go_provisional(chunk);
        return false;
    }
    uint32_t left = 0u;
    if (!sys__heap__zzpackage_give_up(sys__heap_chunk__zzprivate_count_word(chunk), &left)) return false;
    if (left != 0u) return false;
    sys__heap_chunk__zzprivate_recycle(chunk);
    return true;
}


/* Which PROVIDER SLOT a kind is carved from, read off its row — `OBJECTS` or `ARRAYS`, an index and not
 * a chunk. ⚠ THE WORD IS LOAD-BEARING AND THE TWO ARE ONE CALL APART. A base holds a chunk OFFSET in
 * each of those slots, so a "chunk" is what `sys__computing_base__zzpackage_get_heap_provider` answers
 * and a "slot" is what this one does. `make` reads this and hands the answer straight to that, which is
 * the one place where saying either word for the other would still compile.
 *
 * ⭐ IT IS THE HEAP'S ANSWER TO GIVE AND IT IS PUBLIC, WHICH IS WHY IT IS NOT IN THE `heap_object` FAMILY.
 * The question is about a KIND, not about an instance — nothing is counted, nothing is held, and there is
 * no object to hand it. What owns chunks owns which one a kind comes from, so this is the heap's, and
 * another package with a data type of its own — ⚖ ARCHITECT, a special dictionary in `nn` — asks
 * it the same way rather than needing a concept of its own.
 *
 * A kind with NO row answers UNREGISTERED, which is a different answer from the NO_PROVIDER a kind gives
 * when its own row declares it is carved from nothing. They are two different mistakes — a row nobody
 * wrote, against a thing placed by hand that was never meant to be made — and `make` refuses them
 * separately, because the log carries only the word. */
static __device__ inline uint32_t sys__heap__provider(sys__kind kind) {
    switch (kind) {
#define SYS__HEAP__ZZPACKAGE_PROVIDER_ARM(PKG, DTYPE, RELEASE, PROVIDER, ...)  case DTYPE: return PROVIDER;
#define SYS__HEAP__ZZPACKAGE_PROVIDER_GATHER(NAME)   NAME##__LANGUAGE_CONTRACT__OBJECTS(SYS__HEAP__ZZPACKAGE_PROVIDER_ARM, NAME)
#define SYS__HEAP__ZZPACKAGE_PROVIDER_GATHER_ROW(ID, NAME, ...)  SYS__HEAP__ZZPACKAGE_PROVIDER_GATHER(NAME)
        /* Gathered by pasting the package's name, the same way the release and clone switches do it, so
         * a package that registers a kind gets an arm HERE as well as there. Naming one package directly
         * would give the other gathers an arm and this one none, and the kind would then be refused at
         * creation with a word that pointed at its row. */
        PACKAGE_LIST(SYS__HEAP__ZZPACKAGE_PROVIDER_GATHER_ROW)
#undef SYS__HEAP__ZZPACKAGE_PROVIDER_ARM
#undef SYS__HEAP__ZZPACKAGE_PROVIDER_GATHER
#undef SYS__HEAP__ZZPACKAGE_PROVIDER_GATHER_ROW
        default: break;
    }
    return SYS__COMPUTING_BASE__UNREGISTERED;
}

/* How much room a kind takes, in nodes, read off its row — the same gather as the provider above and for
 * the same reason: a kind's size is the kind's business and `make` should not hold a table of them.
 *
 * ⛔ ZERO IS AN ANSWER AND NOT A MISSING ONE. A chunk is a registered kind and is never carved: it IS the
 * room. `make` refuses it on the provider column before it ever asks this, so the zero is what a row says
 * when the question does not apply rather than a value anybody places. */
static __device__ inline uint64_t sys__heap__object_nodes(sys__kind kind) {
    switch (kind) {
#define SYS__HEAP__ZZPACKAGE_NODES_ARM(PKG, DTYPE, RELEASE, PROVIDER, CLONE, CONSTRUCT, NODES, ...) \
        case DTYPE: return (uint64_t)(NODES);
#define SYS__HEAP__ZZPACKAGE_NODES_GATHER(NAME)   NAME##__LANGUAGE_CONTRACT__OBJECTS(SYS__HEAP__ZZPACKAGE_NODES_ARM, NAME)
#define SYS__HEAP__ZZPACKAGE_NODES_GATHER_ROW(ID, NAME, ...)  SYS__HEAP__ZZPACKAGE_NODES_GATHER(NAME)
        PACKAGE_LIST(SYS__HEAP__ZZPACKAGE_NODES_GATHER_ROW)
#undef SYS__HEAP__ZZPACKAGE_NODES_ARM
#undef SYS__HEAP__ZZPACKAGE_NODES_GATHER
#undef SYS__HEAP__ZZPACKAGE_NODES_GATHER_ROW
        default: break;
    }
    return 0ull;
}

/* ── MAKING ONE PER KIND ──────────────────────────────────────────────────────────────────────────
 * Carve a granule out of what this block is carving from, tag it, and hand back its head. Every kind of
 * object begins here, and what makes them different begins after.
 *
 * It is general because the carving is: which chunk is current, what to do when it has no room left, and
 * which counts move are the same questions whatever is being made. What a kind supplies is its dtype and
 * whatever it writes into the nodes after the head.
 *
 * ⛔ AND IT RAISES ITS OWN FAULTS RATHER THAN RETURNING A NULL FOR SOMEBODY ELSE TO INTERPRET. Nothing has
 * published a heap base address · this block never started · the kind has no row · the kind's row declares
 * no provider · the heap allocation pool is empty · a fresh chunk could not hold one granule. They are not
 * the same thing and each has its own word; a caller sees one null and cannot tell them apart, so whoever
 * raised on its behalf had to guess — and guessed "the heap is full" for a caller that had simply never
 * started a block. The word belongs where the cause is known.
 *
 * ⛳ AND RAISING AT THE CALLER INSTEAD BUYS NOTHING, WHICH IS WORTH SAYING BECAUSE IT SOUNDS AS IF IT
 * WOULD. The argument for it is that a caller has "the failing instruction's context" and a container does
 * not — but a caller raising here passes a program counter of zero, exactly as this does. There is no
 * context at either end, so the extra hop preserves nothing and loses which of the three failures it was.
 *
 * ⛳ THE FRESH ALLOCATION IS HELD ONCE, BY WHOEVER ASKED. That is the reference the caller stores; when it
 * lets go, the object ends. And the heap chunk it came out of gains an occupant, which is a different
 * count with a different question — see above.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ── THE BLOCK'S OWN BASE, REMEMBERED ────────────────────────────────────────────────────────────────
 *
 * Finding it is a chain: ask the system register where the bases are, read the row, add this block's
 * number, read the slot, and follow it. Five links, and a carve re-walks all of them.
 *
 * ⭐ IT IS THE ONE THING ON THIS PATH THAT CANNOT GO STALE. A block's base is established when the block
 * starts and never changes for as long as it runs, so a remembered one can only be ABSENT — never wrong.
 * That is what makes it worth remembering and the provider chunk a harder question.
 * ⛳ `ASSUMED` on the host that remembering it pays: it was measured on the device evaluator (▶ the note
 * at the top); a host rate A/B with and without the cache would settle it.
 *
 * ⛔⛔ AND IT MUST BE EMPTIED AT THE START OF EVERY LAUNCH, WHICH IS THE WHOLE RISK IN IT. Block-local
 * storage is per runner thread and outlives a launch, so what is here when a launch starts is whatever
 * the last one on that runner left — and a pointer left by a previous launch is not rubbish that fails a
 * test, it is a plausible address that passes one. `REASONED` on the host; the failure was observed on the
 * device (▶ the note at the top). ⇒ EVERY launch that carves arms it, and
 * `scripts/src_base_cache_gate.py` is what makes that a property of the tree rather than a habit. */
static __device__ inline sys__heap_node** sys__heap__zzprivate_base_cache(void) {
    SYS__SILICON__BLOCK_LOCAL sys__heap_node* remembered;
    return &remembered;
}

/* Put the number back and let the head speak for itself again. Answers whether there was anything to
 * put back, which the caller uses to tell "the block was carving here" from "it was not carving at all".
 * ⛳ IT IS SAFE TO CALL ON A CHUNK THAT IS NOT CACHED, because that is the state it leaves behind and a
 * boundary can be crossed twice. */
static __device__ inline bool sys__heap__zzprivate_stop_carving(uint32_t from) {
    sys__heap_carving* carving = sys__heap_carving_at(from);
    sys__heap_node* chunk = carving->chunk;
    if (chunk == 0) return false;
    carving->chunk = (sys__heap_node*)0;
    sys__heap_chunk__zzprivate_set_watermark(chunk, carving->watermark);
    /* ⭐⭐ AND THE TWO HALVES OF THE COUNT MEET HERE, WHICH IS THE ONLY PLACE THEY CAN. What is in the
     * head is every release that arrived while this block was carving; what is in `appends` is every
     * hold it took and did not write down. Adding one to the other is the whole reconciliation, and it
     * needs no cases because the arithmetic wraps: a head sitting at -3 is `0xFFFFFFFD` and three
     * appends land it on 0.
     * ⛔ THE TAG COMES OFF LAST. Until it does, the number is provisional and nothing may recycle on it;
     * after it, the head is an ordinary count again and the next release to reach zero gives the chunk
     * back by the ordinary path. Restoring the tag before adding would open a window where a release
     * could read a settled tag over an unsettled number. */
    sys__heap_chunk__zzprivate_add_run(chunk, carving->appends);
    carving->appends = 0u;
    chunk[SYS__CHUNK__HEAD].dtype = SYS__KIND__CHUNK_PROVIDER;
    return true;
}

/* Take a chunk's head into block-local storage, and mark the one in the pool so nobody reads it. */
static __device__ inline void sys__heap__zzprivate_start_carving(uint32_t from, sys__heap_node* chunk) {
    sys__heap_carving* carving = sys__heap_carving_at(from);
    carving->chunk     = chunk;
    carving->watermark = sys__heap_chunk__zzprivate_watermark(chunk);
    carving->appends   = 0u;
    chunk[SYS__CHUNK__HEAD].dtype = SYS__KIND__HEAP_CHUNK_ACTIVE;
}

/* ⛔⛔ THIS FORGETS AND DOES NOT WRITE BACK, AND THE DIFFERENCE IS THE WHOLE OF WHY THERE ARE TWO VERBS.
 * It is called on the way IN to every launch, where block-local storage holds whatever the last launch
 * left — so a chunk pointer read here is not rubbish that fails a test, it is a plausible address that
 * passes one, and writing a watermark through it would be a write into somebody else's memory. Coming
 * in, there is nothing to put back and nothing may be read; going out, everything must be.
 * ⇒ ★ THE PAIR IS `forget_base` ON ENTRY AND `sys__heap__stop_carving` ON EXIT, and a launch that carves
 *   owes both. */
static __device__ inline void sys__heap__forget_base(void) {
    for (uint32_t from = 0u; from < SYS__COMPUTING_BASE__PROVIDERS; ++from) {
        sys__heap_carving_at(from)->chunk   = (sys__heap_node*)0;
        sys__heap_carving_at(from)->appends = 0u;
    }
    *sys__heap__zzprivate_base_cache() = (sys__heap_node*)0;
}

/* ⛔⛔ OWED BY EVERY LAUNCH THAT CARVES, ON ITS WAY OUT, AND A LAUNCH THAT SKIPS IT CORRUPTS THE NEXT ONE.
 * The watermark a block is carving at lives in block-local storage while it works and the head in the
 * pool is stale for that whole time. If the launch ends without this, the next one arms itself from that
 * stale head and starts handing out room that is already in use — no fault, no refusal, two objects on
 * the same nodes. ⛳ It is idempotent and safe on a block that never carved, so the honest way to owe it
 * is to call it unconditionally rather than to work out whether it is needed. */
static __device__ inline void sys__heap__stop_carving(void) {
    for (uint32_t from = 0u; from < SYS__COMPUTING_BASE__PROVIDERS; ++from)
        (void)sys__heap__zzprivate_stop_carving(from);
}

static __device__ __noinline__ sys__heap_node* sys__heap__zzpackage_make_sized(sys__kind kind, uint64_t nodes) {
    /* Before anything, because the very next line reads through a pointer the publishing sets. */
    if (!sys__heap__zzpackage_published()) {
        sys__fault__raise(0ull, SYS__HEAP__FAULT_NOT_PUBLISHED);
        return (sys__heap_node*)0;
    }
    sys__heap_node** remembered = sys__heap__zzprivate_base_cache();
    sys__heap_node* computing_base = *remembered;
    if (computing_base == 0) {
        computing_base = sys__heap__zzpackage_computing_base(sys__silicon__block_id());
        /* ⭐⭐ ASKED WHERE THE ANSWER IS FOUND AND NOT WHERE IT IS USED. What a block is working with is
         * established once, when the block starts, and nothing can replace it while a launch runs — so
         * this question has the same answer every time, and asking it once spares every allocation a
         * read from memory. `ASSUMED` to pay on the host; the figure is a device one (▶ the note at the
         * top).
         * ⛔ AND WHAT IS GIVEN UP IS REAL, SO IT IS WRITTEN HERE RATHER THAN LEFT TO BE DISCOVERED. The
         * check guards against something CORRUPTING the base, not against change — nothing changes it.
         * Here, a corrupted base is caught when it is picked up and not at every use, so a corruption
         * after that runs on to become an allocation into whatever the pointer names. `REASONED`,
         * premise: nothing in the tree writes over a computing base. */
        if (!sys__computing_base__is(computing_base)) {
            sys__fault__raise(0ull, SYS__HEAP__FAULT_NO_BASE);
            return (sys__heap_node*)0;
        }
        *remembered = computing_base;
    }

    /* Which chunk this kind is carved from is the kind's own business, read off its row, so the two
     * providers are kept apart by the design rather than by whichever call site asks. */
    const uint32_t from = sys__heap__provider(kind);
    /* Two refusals rather than one, because they are two different mistakes and the log carries only the
     * word. A kind that DECLARED it has no provider is placed by hand and was never meant to be made —
     * the caller is wrong. A kind nobody registered means the row is missing or its package is not in the
     * build — the caller may be entirely right and still get nothing. */
    if (from == SYS__COMPUTING_BASE__UNREGISTERED) {
        sys__fault__raise(0ull, SYS__HEAP__FAULT_UNKNOWN_KIND);
        return (sys__heap_node*)0;
    }
    if (from == SYS__COMPUTING_BASE__NO_PROVIDER) {
        sys__fault__raise(0ull, SYS__HEAP__FAULT_NO_PROVIDER);
        return (sys__heap_node*)0;
    }

    /* ⭐ HOW MUCH ROOM IS THE CALLER'S TO SAY, AND IT IS PLACED WHERE THE LAST ONE ENDED. Nothing is
     * rounded up: a two-node error takes two nodes and the next thing starts against it. What that gives
     * up is the trick of finding an allocation's head by clearing low bits — which nothing needed, because
     * everything holding a reference holds the node right after the head and subtracts one. What it buys
     * is that a wide object is expressible at all.
     *
     * ⭐ AND THE SIZE ARRIVES AS AN ARGUMENT RATHER THAN OFF THE KIND'S ROW, WHICH IS WHAT LETS A KIND
     * HAVE NO ONE SIZE. Almost every kind does have one, and for those the wrapper below reads the row and
     * calls this — so the row stays the answer wherever there is an answer to give. A kind whose extent is
     * decided when one is asked for has nothing to write in that column, and this is its way in: the
     * caller names both the kind and the size. */
    if (nodes == 0ull) {
        sys__fault__raise(0ull, SYS__HEAP__FAULT_GEOMETRY);
        return (sys__heap_node*)0;
    }
    /* ⭐⭐ AND THIS IS THE WHOLE OF AN ALLOCATION WHEN THERE IS ROOM: a look at a number in block-local
     * storage, a subtraction, and the same number moved on. Nothing here reaches the pool: which chunk,
     * its tag and its watermark were all read once, when the carving started.
     * ⛳ THE ROOM IS ASKED OF THE CACHED NUMBER rather than of the chunk, which is the same arithmetic the
     * chunk would have done and is why nothing is checked twice. */
    sys__heap_carving* carving = sys__heap_carving_at(from);
    if (carving->chunk == 0) {
        /* Nothing is cached, which is a cold block or the first allocation after a boundary. The walk is
         * paid once and every allocation after it pays none. */
        const uint64_t held = sys__computing_base__zzpackage_get_heap_provider(computing_base, from);
        if (held != 0ull) {
            sys__heap_node* known = sys__heap__object_full_address(held);
            if (sys__heap_chunk__zzprivate_is_provider(known)) sys__heap__zzprivate_start_carving(from, known);
        }
    }
    sys__heap_node* provider = carving->chunk;
    uint64_t at = 0ull;
    if (provider != 0 && carving->watermark < (uint64_t)SYS__HEAP__CHUNK_NODES
                      && nodes <= (uint64_t)SYS__HEAP__CHUNK_NODES - carving->watermark) {
        at = carving->watermark;
        carving->watermark = at + nodes;
    }
    if (at == 0ull) {
        /* Nothing is being carved yet, or what was has no room left. Either way the answer is another
         * chunk, and taking one is a compare-and-swap over the heap allocation pool that depends on
         * nothing — which is what keeps the first allocation of a block's life from waiting on something
         * that does not exist. */
        const uint32_t idx = sys__heap__zzprivate_claim_chunk();
        if (idx == SYS__HEAP__NO_CHUNK) {
            sys__fault__raise(0ull, SYS__HEAP__FAULT_EXHAUSTED);
            return (sys__heap_node*)0;
        }
        /* ⛔ THE HEAD OF THE ONE BEING LEFT GOES BACK FIRST. Everything below reads and writes the chunk
         * this block is moving to; the one behind it is about to be handed to a release that will read
         * its count, and a count sitting next to a watermark this block never wrote is the state the tag
         * exists to prevent. */
        (void)sys__heap__zzprivate_stop_carving(from);
        sys__heap_node* behind = provider;
        provider = sys__heap__zzprivate_chunk_base(idx);
        /* The block that just asked is written into the fresh heap chunk, so an object can be traced back
         * to whoever is carving where it sits without being handed anything. */
        sys__heap_chunk__zzprivate_init(provider, sys__heap__offset(computing_base), true);
        sys__computing_base__zzpackage_set_heap_provider(computing_base, from, sys__heap__offset(provider));

        /* ⛔ AND THE ONE THE BLOCK IS LEAVING HAS TO BE LET GO OF, HERE, OR IT IS A LEAK NOTHING REPORTS.
         * The hold being given up is the block's own — the 1 the heap allocation pool's compare-and-swap
         * took when this chunk was claimed — and it means "I am carving from this". That stops being true
         * at this line. It can be the last hold already, if everything placed there died while it was
         * still current; then the heap chunk goes home now, by the same path as any other emptying.
         *
         * ⛔ AND THE REQUESTOR GOES WITH IT, AT THIS LINE AND NOT AT THE END OF THE COMPUTATION. ⚖
         * ARCHITECT: *"you then need to remove it when you get a new chunk, not when you exit from compute
         * otherwise you lock the release of that heap chunk."* The slot says WHO IS CARVING HERE, and that
         * stops being true on the line above — so a heap chunk left behind must not go on naming a block
         * that has moved on, and one sitting in the bin must name nobody at all.
         * ⇒ ⭐ AND IT IS WHAT WOULD MAKE THE REFERENCE COUNTABLE IF ANYONE EVER WANTED IT TO BE. Held to
         * the end of a computation, a recycled heap chunk would still owe a release into a computing base
         * — whose method raises, because a release reaching one is a mistake by definition — so every
         * recycle would fault. Given back HERE, the count comes down while the base is plainly alive and
         * nothing is ever owed at recycle. The clear is what makes the choice free. */
        /* ⚠ AND `behind` IS NULL ON A BLOCK'S FIRST ALLOCATION, when there is no previous heap chunk to
         * leave. `release` answers false for a null and always has; `set_ref` writes through what it is
         * given and does not, so the guard belongs here rather than in it — a setter that quietly
         * accepts nothing is a setter that hides a caller's mistake. */
        if (behind != 0) {
            sys__heap_chunk__zzprivate_set_ref(behind, SYS__CHUNK__REQUESTOR, 0ull, false);
            (void)sys__heap_chunk__zzpackage_release(behind);
        }
        sys__heap__zzprivate_start_carving(from, provider);
        if (carving->watermark < (uint64_t)SYS__HEAP__CHUNK_NODES
            && nodes <= (uint64_t)SYS__HEAP__CHUNK_NODES - carving->watermark) {
            at = carving->watermark;
            carving->watermark = at + nodes;
        }
        if (at == 0ull) {                          /* a chunk too small to hold one of this kind */
            sys__fault__raise(0ull, SYS__HEAP__FAULT_GEOMETRY);
            return (sys__heap_node*)0;
        }
    }

    sys__heap_node* head = provider + at;
    head[0].dtype    = kind;      /* what this is, so the counting knows how to end it */
    head[0].num_args = 1u;        /* the reference the caller is about to store */
    head[0].args[SYS__HEAP_OBJECT__LOCK] = 0ull;   /* the chunk was not zeroed; every head field is written */
    SYS__HEAP_OBJECT__NODES(&head[0]) = nodes;     /* how far it reaches, which nothing else can work out */
    /* ⭐ THE HOLD IS COUNTED IN BLOCK-LOCAL STORAGE AND NOT IN THE CHUNK, which is the second thing this
     * cache removes from an allocation and the more expensive of the two: writing it down was a
     * compare-and-swap loop over a word in the pool, every single time. Here it is an increment of a
     * number nobody else is touching, and the run of them is written down once when the block moves off.
     * ⛳ THE ONE THAT IS NOT GATHERED IS `start_block`'s, which places the computing base by hand into a
     * chunk it has just made and is carving from nothing. */
    carving->appends += 1u;
    return head;
}

/* Making one of a kind that HAS a size, which is nearly all of them: the row is asked and the answer is
 * handed to the carve above. One line, and it is the line that keeps "a kind's size is a kind's business"
 * true for every kind that has such a business — a caller naming a fixed kind still names nothing else.
 * ⛳ A KIND WITH NO ROW ANSWERS ZERO HERE AND IS REFUSED THERE, on its provider column, which is why this
 * does not test for it twice. */
/* ⭐⭐ AND IT IS `inline` WHERE EVERYTHING AROUND IT IS NOT, AND THE SIZED CARVE IT CALLS STAYS OUT OF
 * LINE. This is one line of work behind a call; removing both boundaries would merge the whole allocation
 * chain into its callers. `ASSUMED` on the host that this pairing (this one inline, the next out of line)
 * is the cheaper one; it was measured on the device evaluator (▶ the note at the top); a host rate A/B of
 * the four combinations would settle it.
 * ⇒ ★ SO NEITHER HALF MAY BE READ AS AN INVITATION TO DO THE OTHER: a boundary is what keeps the other
 *   one affordable. */
static __device__ inline sys__heap_node* sys__heap__make(sys__kind kind) {
    return sys__heap__zzpackage_make_sized(kind, sys__heap__object_nodes(kind));
}

/* ── STARTING A BLOCK OFF ────────────────────────────────────────────────────────────────────────────
 * Turn a fresh heap chunk into the one a block keeps, and put the thing it works with inside it.
 *
 * This is the one allocation that cannot go through the ordinary path, and the reason is a circle: what
 * carves small things out of a chunk is remembered in the computing base, and the computing base is a
 * small thing carved out of a chunk. Something has to be placed before there is anywhere to remember it,
 * so this does that once and everything after it is ordinary.
 *
 * The chunk it is given never comes back, and what makes that true is the computing base itself: it is an
 * object living in it, its row in the teardown frees nothing and raises, so the hold taken for it below is
 * never given up and the heap chunk's count can never reach zero. The one thing that must always be
 * findable ends up in the one place that is always there.
 *
 * ⚠ AND THAT IS TRUE OF WHERE A BASE LIVES TODAY, NOT OF WHAT A BASE IS. ⚖ RULED: a base belongs in the
 * system register's own allocation, indexed by block id. There it provides for nothing, so releasing one
 * would give up its own granule and its holds and nothing else — and the refusal above becomes
 * unnecessary rather than load-bearing. Read the paragraph as a fact about this arrangement, which
 * expires when the base moves.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__heap_node* sys__heap__start_block(sys__heap_node* chunk) {
    /* ⛳ AND WHATEVER WAS REMEMBERED GOES, BECAUSE THIS IS THE ONE PLACE A BLOCK'S BASE IS ESTABLISHED.
     * It runs once per block at boot and there is usually nothing to forget; saying so here is
     * what makes the remembered base's promise — that it can only be absent, never wrong — true of every
     * site rather than of the ones that happened to be thought of. */
    sys__heap__forget_base();
    /* The heap base address is a published static rather than a parameter, so a block running before the
     * boot path has published one is a thing that can happen and has to be refused here. That is the price
     * of not passing it: a parameter cannot be missing, and a static can. One line, and it is the only
     * place the absence is checkable. */
    if (chunk == 0 || !sys__heap__zzpackage_published()) return (sys__heap_node*)0;
    sys__heap_chunk__zzprivate_init(chunk, 0ull, false);
    const uint64_t nodes = sys__heap__object_nodes(SYS__KIND__COMPUTING_BASE);
    const uint64_t at = sys__heap_chunk__zzprivate_place(chunk, nodes);
    if (at == 0ull) return (sys__heap_node*)0;

    sys__heap_node* head = chunk + at;
    head[0].dtype    = SYS__KIND__COMPUTING_BASE;
    head[0].num_args = 1u;                        /* held by the block, for as long as the block runs */
    head[0].args[SYS__HEAP_OBJECT__LOCK] = 0ull;   /* the chunk was not zeroed; every head field is written */
    SYS__HEAP_OBJECT__NODES(&head[0]) = nodes;     /* placed by hand, and it still says how far it reaches */
    /* The state word is poison like the rest until something writes it, and `init` fills only what its
     * caller holds — so a base being carved has to say this block holds it before it can be filled. Both
     * statements are the computing base's business, so they are ONE call rather than two — which is what
     * keeps the raw state store private.
     *
     * ⛳ LATER: ⚖ ARCHITECT — the bases become the SYSTEM REGISTER's to hand out, at which point
     * `start_block` stops placing them at all and an engine-side init becomes the only caller from
     * outside that file. This line goes with it.
     * ⚠ AND THE REGISTER HAS NO ALLOCATION OF ITS OWN TO PUT THEM IN: it rides with block zero and carves
     * from that block's providers. So the bases do not move to a new place, they move to a DIFFERENT
     * BLOCK'S — out of the chunk each block carves from and into block zero's. That is what the heap's
     * floor argument is written against, and it says so there. */
    (void)sys__computing_base__zzpackage_start(head + 1, sys__silicon__block_id());
    /* ⛳ AND THE REQUESTOR IS WRITTEN HERE RATHER THAN AT `init` ABOVE, FOR THE ONE REASON THIS PATH
     * EXISTS: the base a heap chunk would name does not exist until it has been placed IN that heap
     * chunk. Every other claim can name its requestor while initialising; this one is the circle the
     * whole function is for, so it names it on the far side. */
    sys__heap_chunk__zzprivate_set_ref(chunk, SYS__CHUNK__REQUESTOR, sys__heap__offset(head + 1), true);
    sys__computing_base__zzpackage_set_heap_provider(head + 1, SYS__COMPUTING_BASE__OBJECTS, sys__heap__offset(chunk));

    /* ONE hold, for the base living here — and NOT a second one for the block carving from it, which is
     * the hold `claim_chunk` takes when the heap allocation pool hands a heap chunk out.
     *
     * ⛔ THAT IS RIGHT FOR EITHER CHUNK THIS CAN BE GIVEN, BUT FOR DIFFERENT REASONS, AND IT LOOKS LIKE AN
     * OMISSION EITHER WAY. One that came from the heap allocation pool arrived with the carving hold
     * already on it — the 1 the claim's compare-and-swap took — and `make` gives that one up when the
     * block moves off; taking another here would count it twice. Chunk zero never came from the heap
     * allocation pool, because nothing ever put it there — the pass that fills the free list skips it — so
     * it has no carving hold and needs none: its offset is zero, and zero is what the provider slot means
     * by "nothing is being carved from", so the slot cannot name it and `make` always takes a fresh heap
     * chunk instead. A carving hold there would be one nothing ever gives up, on a chunk nothing ever
     * carves from.
     *
     * ⚠ WHICH OF THE TWO THIS IS, IS THE CALLER'S CHOICE, AND THE BOOT MAKES IT BY BLOCK. Chunk zero can
     * be given to one block and no more, so the block that starts first is handed it and every other one
     * comes through the entry below with a chunk of its own. The slot written just above is therefore
     * inert for the first block and live for all the rest.
     *
     * ⭐ AND THE CHUNK IS SAFE EITHER WAY. The computing base's row in the teardown hands nothing back and
     * frees nothing — it raises — so the hold taken here is never given up, this count can never reach
     * zero, and the heap allocation pool can never be handed the chunk back. */
    sys__heap_chunk__zzpackage_add(chunk);

    /* And this is where the block says what it is working with, into the slot the register names. It
     * is written straight into the node rather than through the array's set, for the same reason the
     * read above does not go through its get: a set would release whatever was there and retain what
     * arrives, and what is here is a plain number the boot placed. */
    {
        const sys__heap_node where = sys__system_register__get(SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY);
        if (where.dtype != SYS__KIND__VALUE_INT || where.args[0] == 0ull) {
            sys__fault__raise(0ull, SYS__HEAP__FAULT_NOT_PUBLISHED);
            return (sys__heap_node*)0;
        }
        sys__heap__object_full_address(where.args[0] + (uint64_t)sys__silicon__block_id())->args[0] =
            sys__heap__offset(head + 1);
    }
    return head + 1;
}

/* ── AND A BLOCK STARTING ITSELF ─────────────────────────────────────────────────────────────────────
 * The same thing for a block nobody can hand a chunk to: it takes one from the free ring first and then
 * starts off in it.
 *
 * ⭐⭐ THE TAKING IS WHY THIS EXISTS RATHER THAN THE STARTING. Chunk zero is the only room that can be
 * given away from outside, because it is the only chunk the free ring never holds — anything else offered
 * from outside could be handed to an allocation in the same breath. So a second block cannot be given
 * room; it has to take some, and taking is the allocator's own business. Putting the two steps together
 * here is what lets a caller ask for a block to start without asking for a chunk to exist.
 *
 * ⭐ AND THE HOLD ARRIVES WITH THE CHUNK, WHICH IS WHY NOTHING IS TAKEN HERE. The claim's compare-and-swap
 * puts the count at one, and that is the carving hold — the one `make` gives up when this block fills the
 * chunk and moves on. The paragraph above says why a second one here would be counting it twice.
 *
 * ⛔ A CHUNK TAKEN AND THEN NOT USED IS NOT PUT BACK, AND IT IS NOT MEANT TO BE. Starting off can only
 * fail where the heap base address is not published, which the claim has already checked — so reaching
 * that is a boot happening in an order that cannot work, and the answer is that the boot refuses and the
 * whole pool goes back at once. Winding one chunk out by hand would be a path nothing exercises, for a
 * case where nothing else survives either.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline sys__heap_node* sys__heap__start_own_block(void) {
    const uint32_t idx = sys__heap__zzprivate_claim_chunk();
    if (idx == SYS__HEAP__NO_CHUNK) return (sys__heap_node*)0;
    return sys__heap__start_block(sys__heap__zzprivate_chunk_base(idx));
}

/* The heap chunk an allocation sits in, as a pointer rather than an index.
 *
 * ⭐ IT ANSWERS THE HEAD ITSELF WHEN THE HEAD IS A CHUNK'S OWN FIRST NODE, and that is the whole guard
 * the give-back needs: an object sits IN a chunk, a chunk IS one. Nothing has to ask what kind it is
 * looking at — the arithmetic already distinguishes them, because a chunk begins where the mask lands. */
static __device__ inline sys__heap_node* sys__heap_chunk__zzpackage_from_object_head(const sys__heap_node* head) {
    if (head == 0 || !sys__heap__zzpackage_published()) return (sys__heap_node*)0;
    const uint64_t off = sys__heap__offset(head);
    return sys__heap__zzprivate_heap_base_address + (off & ~((uint64_t)SYS__HEAP__CHUNK_NODES - 1ull));
}

/* ── WHETHER AN ADDRESS RANGE IS THE HEAP'S ──────────────────────────────────────────────────────────────
 * The pool and the room the boot set aside past it, which is everything the published offsets can name.
 * ⛳ Asked by nn's C interface, which answers only outside it; `sys_abi_peek` bounds its own. */
static __device__ inline bool sys__heap__holds(uint64_t address, uint64_t bytes) {
    if (!sys__heap__zzpackage_published() || address == 0ull || bytes == 0ull) return false;
    /* ⛳ THE ENDS AS POINTERS, SO THE NODE WIDTH STAYS THE COMPILER'S: the package counts in nodes. */
    const uint64_t base = (uint64_t)(uintptr_t)sys__heap__zzprivate_heap_base_address;
    const uint64_t end  = (uint64_t)(uintptr_t)(sys__heap__zzprivate_heap_base_address + sys__heap__zzprivate_nodes);
    return address >= base && bytes <= end - base && address - base <= end - base - bytes;
}

/* ── sys's C INTERFACE, THE HEAP'S HALF ── ▶ `contracts/abi/cpu.cuh` for what each promises. ──────────── */
#ifndef SYS__SILICON__HARNESS   /* a real build, not a test harness that brings its own silicon */
#include "../contracts/abi/cpu.cuh"
extern "C" {
int sys_abi_peek(unsigned long long address, sys__heap_node* into) {
    /* ⛳ A HEAP ADDRESS IS HOST MEMORY, so this one reads through it: the node is found as the base plus
     * the whole nodes before it, and an address between two nodes is refused rather than rounded. */
    const sys__heap_node* at = (const sys__heap_node*)(uintptr_t)address;
    if (into == 0 || !sys__heap__zzpackage_published()) return 0;
    const sys__heap_node* base = sys__heap__zzprivate_heap_base_address;
    if (at < base || at >= base + sys__heap__zzprivate_nodes || base + (at - base) != at) return 0;
    *into = *at;
    return 1;
}

unsigned long long sys_abi_full_address(unsigned long long offset) {
    if (offset == 0ull || !sys__heap__zzpackage_published() || offset >= sys__heap__zzprivate_nodes) return 0ull;
    return (unsigned long long)(uintptr_t)sys__heap__object_full_address((uint64_t)offset);
}

unsigned long long sys_abi_base_address(unsigned int block) {
    if (!sys__heap__zzpackage_published()) return 0ull;
    const sys__heap_node where = sys__system_register__get(SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY);
    if (where.dtype != SYS__KIND__VALUE_INT || where.args[0] == 0ull || !sys__node_array__is(where.args[0])) return 0ull;
    if ((uint64_t)block >= sys__node_array__length(where.args[0])) return 0ull;
    return (unsigned long long)(uintptr_t)sys__heap__zzpackage_computing_base((uint32_t)block);
}

/* ⛔⛔ WHETHER THE DEALLOCATION TALLY IS LIVE AT ALL, AND IT HAS TO BE ASKED SEPARATELY. It is behind a
 * compile-time switch that is OFF in a shipping build, so it reads ZERO however much work has been done —
 * and zero is exactly what "nothing was ever freed" looks like. ⇒ ★ A DISABLED INSTRUMENT MUST REPORT THAT
 * IT IS DISABLED, NEVER A READING. */
int sys_abi_counts_deallocations(void) {
#if SYS__HEAP__COUNT_DEALLOCATIONS
    return 1;
#else
    return 0;
#endif
}

int sys_abi_counters(unsigned int* chunk_claims, unsigned int* chunk_refusals, unsigned int* deallocations) {
    const sys__heap_counters* got = sys__heap__zzprivate_counters;
    if (got == 0) return 0;
    if (chunk_claims)   *chunk_claims   = __atomic_load_n(&got->at[SYS__HEAP__COUNTER_CHUNK_CLAIMS], __ATOMIC_RELAXED);
    if (chunk_refusals) *chunk_refusals = __atomic_load_n(&got->at[SYS__HEAP__COUNTER_CHUNK_REFUSED], __ATOMIC_RELAXED);
    if (deallocations)  *deallocations  = __atomic_load_n(&got->at[SYS__HEAP__COUNTER_DEALLOCATIONS], __ATOMIC_RELAXED);
    return 1;
}
}
#endif

#endif /* SILVANN__PACKAGES_SYS_CPU_HEAP__IMPL_CUH */
