#ifndef SILVANN__PACKAGES_SYS_CPU_HEAP__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_HEAP__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/kind.cuh"                /* what a thing IS — the first word of every node */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../contracts/objects/heap.cuh" /* its constants, fault words and layouts */
/* ══ the heap chunk allocator, and the counting ═══════════════════════════════════════════════════════════
 *
 * This file holds three things. The heap allocation pool that hands out CHUNKS. The two halves of a reference count —
 * taking a hold and giving one up — which sit here beside the word they move, because an object carries
 * its count in the node its allocation begins with. And a chunk's own release method, which is the one a
 * reader would least expect in an allocator: a chunk is a registered kind like anything else, so what
 * happens when its last occupant goes is a row in the kind list, and the method that row names belongs
 * to whoever owns heap chunks. What walks a DYING OBJECT releasing what it referred to is a different
 * file — `heap_object__impl.cuh`, reached from that same list.
 *
 * ── WHAT MUST ALREADY EXIST ─────────────────────────────────────────────────────────────────────────
 * The includes at the top name three: `kind.cuh`, for the word every node begins with — what a thing
 * IS, which is what a release method is reached through; `heap_node__header.cuh`, for `sys__heap_node`,
 * which every declaration below is written in; and the heap's own contract, for its constants, fault
 * words, layouts and the counters `publish` is handed. One more must already exist and gets no include
 * of its own here: `SYS__HEAP_OBJECT__LOCK`, the head layout every heap chunk shares, from
 * `heap_object__header.cuh`. The manifest is a list and not an order, so nothing here depends on where
 * this file sits in one.
 *
 * ⛳ AND THE LIST IS SHORT BECAUSE A HEADER DECLARES RATHER THAN CALLS, which is worth saying because
 * the obvious reading is that an allocator must need the carousel it allocates from. It does not, here:
 * the free ring is a static the implementation keeps and every verb that touches it is defined over
 * there, so `heap__impl.cuh` owes that include and this one does not. The silicon seam and the computing
 * base are not named in this file at all; they are what the IMPLEMENTATION needs, and its own includes
 * say so.
 *
 * ── WHAT A BLOCK NEEDS, AND WHY IT IS IN TWO PLACES RATHER THAN ONE ─────────────────────────────────
 * It falls into two kinds, and one structure could not hold them because they are answers to different
 * questions.
 *
 * Some of it CANNOT DIFFER between two calls in the same launch — the heap base address, what is free,
 * the diagnostic counters — and that is published once, through `publish` below, into statics the
 * implementation keeps.
 *
 * The rest differs per block, and that is the computing base: which chunk this block carves objects from
 * and which it carves arrays from. A published verb is HANDED it, the opcode shape putting the base
 * first, and the allocator underneath asks for it instead: `make` takes the block id the seam answers —
 * on the host, the runner thread's own — reads that block's base out of the system register, and
 * remembers it for as long as the block runs.
 * ⇒ ★ THE VALUE IS A PROPERTY OF THE BLOCK, AND A BLOCK IS THE ONE THING EVERY PATH INTO THE ALLOCATOR
 * HAS IN COMMON; threading a parameter down every intermediate call would serve the paths that begin at
 * an opcode and buy nothing on the rest.
 * ⚖ ARCHITECT: *"you should pass base directly then, it is more honest."* So the one thing that genuinely
 * has to travel travels, and nothing rides along with it.
 *
 * ⛳ HOW MANY THINGS REFER TO AN OBJECT IS IN NEITHER OF THOSE PLACES. It lives in the object — first word
 * of its head, beside the kind — so finding it is arithmetic on an address the caller already has. A
 * table indexed by node offset would have to be as long as the heap, and nothing could shorten it: each
 * object begins where the last one ended and nothing is rounded to anything — the array granule further
 * down is one kind's width, asked for by the kinds that want a run of cells and not a rule about where an
 * allocation may land — so EVERY node is one an object may begin at and no entry can be ruled out in
 * advance. `REASONED`, from the placer and from the carve path both: each returns the raw watermark and
 * moves it on by exactly the nodes asked for, and the chunk header below says the same of the first one.
 * ⇒ A WORD PER NODE OF HEAP, STANDING WHETHER ANYTHING IS ALIVE OR NOT, to hold what already rides beside
 * the kind word a release reaches its method through. The same is true of a heap chunk, whose head is a
 * head like any other.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ── THE CONTRACT ────────────────────────────────────────────────────────────────────────────────────
 * Every clause is argued somewhere in the implementation; this is where they are gathered as obligations.
 *
 * WHAT THE CALLER OWES
 *   a published     `publish` before anything else, once, from the boot path — NOT from a block.
 *   heap base address           ⛔ AND WHAT HAPPENS IF IT HAS NOT IS NOT ONE ANSWER, WHICH IS THE PART
 *                   TO READ. `make` tests first and RAISES, because it is about to read through a
 *                   pointer the publishing sets. The block-starting verbs test and answer a quiet null.
 *                   The address arithmetic does not test at all — with no base published, asking for an
 *                   object's full address hands back a plausible-looking offset from zero, which is
 *                   exactly the shape the shutdown retires the base to prevent.
 *                   ⇒ ★ SO A BOOT-ORDER MISTAKE IS DIAGNOSABLE FROM THE LOG ONLY IF IT REACHED `make`.
 *                   Everything earlier is a null to interpret or a wild pointer to chase.
 *                   ⚠ AND THE FAULT WORD IS NOT A RELIABLE WITNESS EITHER: the other place that raises
 *                   it is the block start finding NO SYSTEM-REGISTER ROW, which is a different cause in
 *                   the same boot. Read it as "something the boot owed is missing", not as "the heap
 *                   base address was never published".
 *   a started       a block that allocates must have a computing base, because `make` reads which chunk
 *   block           to carve from out of it. `start_block` and `start_own_block` are what put one there.
 *   a kind with a   `make` derives the provider from the kind's row. A kind with no row is refused at run
 *   row             time, not at compile time — that is the one direction the mechanism cannot close.
 *   one thread      per block, which is not this file's condition but the whole package's — see
 *                   `packages/manifest__header.cuh`. It matters most visibly here: a per-thread `claim_chunk`
 *                   takes one heap chunk per thread and empties the heap allocation pool on the first allocation.
 *   one give-up     per hold. `zzpackage_give_up` REFUSES at zero rather than wrapping, so an extra
 *                   release is reported rather than making the object immortal.
 *   the heap chunk  ⛔ A BLOCK MOVING OFF A FULL CHUNK MUST LET GO OF IT, and `make` does it at the
 *   it leaves       moment the next one is taken. Miss it and the chunk sits at one hold forever with
 *                   nothing in it — a leak no count is wrong about and nothing reports.
 *
 * WHAT IT PROMISES BACK
 *   one counting    an object's holders live in its own head, beside its kind, so finding the count is
 *                   arithmetic on an address the caller already has. A chunk's is the same word in the
 *                   same place, because a chunk's head is a head.
 *   no bias         a chunk freed and re-claimed goes behind everything already waiting, so no region of
 *                   the pool is reused while the rest sits untouched — which is what a free list buys
 *                   over a scan, not speed.
 *   a loud refusal  a word per cause, never one shared between two, because a reader of a log has only
 *                   the word: a full heap allocation pool is a CONDITION, no base is a CALLER, a geometry failure is an
 *                   assumption that has already broken somewhere else.
 *   an offset that  every reference is measured from the heap base address rather than pointing at it, so a value
 *   travels        survives being copied into a cell another block will read.
 *
 * WHAT IT DOES NOT DO
 *   It does not give room back IN PIECES. Freeing an object gives up its chunk's hold and nothing else —
 *   the watermark does not move, the space is not reused, and the WHOLE chunk goes home at once when the
 *   last hold goes. That is why there are two providers: what LASTS and what CHURNS are carved apart so
 *   permanence cannot pin churn.
 *   It does not TAKE APART what it stops counting. Giving an object's room back is one line and it is
 *   somebody else's — walking what that object held is `heap_object__impl.cuh`, reached from the kind's
 *   row, and the give-back happens there too, once, for every kind.
 *   It does not CHECK a kind against its row at compile time — see above.
 *   It does not own the geometry guarantee it relies on. The assert that a chunk can hold one granule is
 *   in `contracts/objects/stack.cuh`, under another macro, in a file ordered after this one.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ── THE CONTRACT, DECLARED ─────────────────────────────────────────────────────────────────────────
 * The public surface, and it is smaller than the file. What carries `zzprivate_` stays in the
 * implementation, which is what that marker means.
 * ⛳ AND THE LIST IS NOT THE WHOLE OF THE SURFACE. `sys__heap__stop_carving` is public, is defined in the
 * implementation, and is called from the engine — `engine/boot.cuh` and the files under `engine/abi/` —
 * with no declaration in any header of this package. It is the exit half of the pair `forget_base`
 * opens, so a reader auditing what the engine may call cannot do it from the declarations below alone.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* ⛳ `reserved_nodes` IS MEMORY THIS MUST ADDRESS AND MUST NEVER HAND OUT — room past the chunks, for
 * storage the boot owns and no block does. Pass 0 to reserve none. The body argues why it is a separate
 * number from the chunk count rather than derivable from it. */
static __device__ inline bool         sys__heap__publish(sys__heap_node* heap_base_address, uint32_t chunks,
                                                         sys__heap_counters* counters,
                                                         uint64_t reserved_nodes);
static __device__ inline uint64_t     sys__heap__offset(const sys__heap_node* address);
static __device__ inline sys__heap_node*  sys__heap__object_full_address(uint64_t heap_offset);
static __device__ inline uint32_t     sys__heap__chunk_of(const sys__heap_node* at);
static __device__ inline uint32_t     sys__heap__provider(sys__kind kind);
/* ⛳ TWO WAYS A BLOCK COMES TO HAVE A BASE, AND THE ONLY DIFFERENCE IS WHERE THE ROOM COMES FROM. One is
 * HANDED a chunk, which is how the first block gets chunk zero — the one chunk that never enters the free
 * ring, and therefore the only one that can be given away from outside without racing an allocation for
 * it. Every other block TAKES its own from the ring, and taking is this file's business rather than its
 * caller's, so that form asks for nothing and answers the same base. */
static __device__ inline sys__heap_node*  sys__heap__start_block(sys__heap_node* chunk);
static __device__ inline sys__heap_node*  sys__heap__start_own_block(void);
/* And the other end of `publish`: there is no pool any more. The boot's, because the boot is what knows
 * the memory is about to be handed back — the argument is beside the definition. */
static __device__ inline void             sys__heap__zzengine_retire(void);

/* Where an allocation BEGINS, from a reference to it, and the value that means nothing. Both are
 * `sys__heap_node__` rather than `sys__heap__` because they answer a question about MEMORY SHAPE and not
 * about the heap allocation pool — the naming argument is beside each definition.
 * ⚠ FROM A REFERENCE, NOT FROM ANY INTERIOR ADDRESS. Allocations are packed at whatever width their kind
 * asked for, so the head is the node before a reference and nothing recovers it from anywhere else. */
static __device__ inline sys__heap_node*  sys__heap_node__head(const void* inside);
static __device__ inline sys__heap_node   sys__heap_node__nothing(void);

/* ── OUTSIDE THE CONTRACT ───────────────────────────────────────────────────────────────────────────
 * Reachable from the rest of `sys` and no further.
 *
 * ⭐⭐ `make` IS DECLARED HERE BUT CARRIES NO PACKAGE MARKER, because a kind's own constructor calls the
 * allocator and kinds live in more than one package — `nn`'s buffer, weights and cartridge reference are
 * made through it too.
 * ⇒ ★ THE RULE THIS SECTION PROTECTS IS NOT ABOUT PACKAGES: *"the rule is about what a PROGRAM can
 * reach"*, and a program reaches only published verbs and the `create` dispatch. A name reachable from
 * another package changes nothing about the language's surface. ⛳ What has to hold instead is that
 * `nn__buffer`'s `create` column REFUSES: a buffer names a slot the pool handed out, so a program that
 * could construct one could name a slot it was never given.
 *
 * `make` is not part of the contract above because a caller says
 * `sys__stack__create()` or `sys__heap_object__release()` and never learns which chunk answered — which is
 * the whole point of deriving the provider from the kind's row. The chunk verbs beside it are package for
 * the same reason from the other end: nobody writing a kind should be naming one.
 * ⛳ AND `make` HAS MANY CALLERS, WHICH IS NOT THE SAME AS MANY DOORS. Most are a kind's own
 * constructor, one per row. The rest carve another heap chunk for a container that has filled the one it
 * had — `sys__stack_chunk__zzprivate_create` for a stack and `sys__list_chunk__zzprivate_create` for a
 * list. `zzpackage_make_sized` serves the kinds whose extent the caller names: the node array and the
 * string.
 *
 * ⭐⭐ THE RULE IS ABOUT WHAT A PROGRAM CAN REACH, NOT ABOUT HOW MANY FUNCTIONS CALL THE ALLOCATOR.
 * `create` is the only entry point from the language that carves a NEW object reference. Everything else
 * carves INSIDE an object that already exists — adding to a stack, removing from one — and a chunk it
 * takes to do that never becomes a value the program can hold.
 * ⛔
 *    THAT IS THE PROPERTY TO PROTECT, and it is the one the growth callers look like they break and do not.
 * ⛳ AND THE REASON IS NOT THAT SUCH A REFERENCE COULD NOT BE ENDED — a heap chunk is a registered kind,
 * so releasing one recycles it like anything else. The reason is what it would be POINTING AT: a
 * reference names the node after a head, and the node after a heap chunk's head is inside that chunk's
 * OWN HEADER. A program treating it as an object would be reading and writing the requestor slot, one
 * node from the watermark and two from the count — corrupting the heap allocation pool through a value it was handed
 * rather than through a mistake it made.
 *
 * `published` and `zzpackage_head` are asked by other files before they reach for anything.
 * `give_up` is the shared decrement — the one that REFUSES at zero rather than wrapping — and every
 * count in this package moves through it, which is what makes "there is one counting" true.
 *
 * ⛳ `zzpackage_computing_base` READS THE REGISTER AND NOT A TABLE OF THE HEAP'S OWN. The per-block array
 * of bases lives in the system register's allocation — one row names the offset the bases lie at, end to
 * end — so this verb reads that row and adds the block id. The heap keeps no bases table and `publish` is
 * handed none; what an allocating block needs from the register it asks for once and then remembers.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline bool         sys__heap__zzpackage_published(void);
/* Whether `bytes` from `address` lie inside the heap — the pool and the room the boot set aside past it. */
static __device__ inline bool         sys__heap__holds(uint64_t address, uint64_t bytes);
/* How far the heap allocation pool reaches, in nodes. Asked rather than read from a constant, because the heap allocation pool's size is
 * settled when it is published and not when this is compiled. */
static __device__ inline uint64_t     sys__heap__zzpackage_total_nodes(void);
/* Record that an object's last hold went, which is the moment nothing else records: what happens next is
 * decided by kind, and by then the count that got there is gone. It takes no argument because there is
 * one thing to say — a verb that named its counter would be offering a choice this package does not have.
 *
 * ⛳ THE ALLOCATOR COUNTS ITS OWN TRAFFIC WITHOUT COMING THROUGH HERE, straight into the same block:
 * `HEAP_CHUNK_CLAIMS` and `_REFUSED` are the chunk's numbers and answer a different question, for the
 * reason set out beside `release`. Worth having when something is wrong and worth nothing the rest of the
 * time, which is what an off-by-default switch is for: disarmed, this compiles to nothing.
 *
 * ⛔ AND IT IS A FUNCTION RATHER THAN A MACRO FOR ONE REASON — that is the only way it can be named from
 * outside the allocator. What it adds to is a static the implementation keeps to itself, so a macro would
 * put a private name into every file that counted anything. The declaration here and the body over there
 * hand out the verb without handing out the storage. */
static __device__ inline void         sys__heap__zzpackage_record_deallocation(void);
/* Carve room for one object of a kind whose extent the CALLER knows, in nodes, head included. A kind
 * whose size is decided when one is asked for has nothing to put in its row's size column, and this is the
 * only way such a kind is made at all. */
static __device__ __noinline__ sys__heap_node*  sys__heap__zzpackage_make_sized(sys__kind kind, uint64_t nodes);
static __device__ inline sys__heap_node*  sys__heap__make(sys__kind kind);
static __device__ inline sys__heap_node*  sys__heap_chunk__zzpackage_from_object_head(const sys__heap_node* head);
static __device__ inline uint32_t     sys__heap__zzpackage_take_up(unsigned int* slot);
static __device__ inline bool         sys__heap__zzpackage_give_up(unsigned int* slot, uint32_t* left);
/* An occupant arrives in a chunk, and an occupant leaves it — the second sends the chunk home when it was
 * the last. They are the whole of what a kind's author never has to think about: objects are created and
 * released, and which heap chunk answered is the allocator's business.
 * ⛳ ONLY THE SECOND IS READ FROM OUTSIDE THE ALLOCATOR, by the one place that gives an object's room
 * back. The first is marked package anyway and on purpose: they are one pair, a caller that can take a
 * hold must be able to give one up in the same vocabulary, and splitting the marker down the middle would
 * say the two halves live at different depths. */
/* ── FORGET THE BLOCK'S REMEMBERED BASE ──────────────────────────────────────────────────────────────
 * The allocator remembers where this block's computing base is, because finding it is a chain of about
 * six loads and it cannot change while the block runs. This is how that memory is emptied.
 *
 * ⛔ EVERY LAUNCH OWES IT BEFORE IT ALLOCATES, and it is public for that reason rather than because
 * anything outside the allocator has an interest in it. Block-local storage is per runner thread and
 * outlives a launch, so what is remembered at the start of one is the PREVIOUS launch's base — a
 * plausible address that passes every test the carve makes, not rubbish that fails one. `REASONED` on
 * the host from that premise; the failure was observed on the device evaluator (▶ the note at the top).
 * ⛳ `scripts/src_base_cache_gate.py` is what keeps that from being a habit. */
static __device__ inline void         sys__heap__forget_base(void);

static __device__ inline void         sys__heap_chunk__zzpackage_add(sys__heap_node* chunk);
static __device__ inline bool         sys__heap_chunk__zzpackage_release(sys__heap_node* chunk);
static __device__ inline sys__heap_node*  sys__heap__zzpackage_computing_base(uint32_t block_id);
static __device__ inline sys__heap_node*  sys__heap_node__zzpackage_head(uint64_t offset);
/* The chunk's release method, named by its row in the kind list and called only by the dispatch. */
static __device__ __noinline__ void sys__heap_chunk__zzpackage_release_internal(sys__heap_node* head,
                                                                         uint64_t* releaser_stack);

#endif /* SILVANN__PACKAGES_SYS_CPU_HEAP__HEADER_CUH */
