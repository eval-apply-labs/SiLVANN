#ifndef SILVANN__PACKAGES_SYS_CPU_COMPUTING_BASE__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_COMPUTING_BASE__HEADER_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "../contracts/defaults.cuh"
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "../contracts/objects/computing_base.cuh" /* its constants and fault words */
/* ══ what a block is working with ════════════════════════════════════════════════════════════════════
 *
 * Everything a block needs to reach from anywhere, in one node accessible at the same address for the
 * whole lifetime of the computation it demands.
 *
 *     the objects      the heap chunk providing memory for objects             a heap offset
 *     the arrays       the heap chunk providing memory for standard arrays     a heap offset
 *     the bindings     the object containing the lexically shadowed            a heap offset
 *                      stacked variables that the ast program may use
 *     the program      the list being run — and, once computing is completed,  a heap offset 
 *                      the value it became
 *     the status       how the computation is going                            a state machine value
 *     the result kind  what the word in the program slot is, once there is     a kind
 *                      a result
 *
 * ⭐ THE FIRST FOUR ARE ALL HEAP OFFSETS, AND THE TWO SHAPES CANNOT BE CONFUSED — WHICH IS WHY THEY SHARE
 * ONE TYPE AND NEED NO TAG BESIDE THEM. A chunk offset is `index × SYS__HEAP__CHUNK_NODES`, so it lands
 * exactly on a chunk's first node; a reference is the node after an allocation's head, so it lands past
 * the chunk's own header. The two sets are disjoint, and `sys__heap_object__zzpackage_addressable` tests
 * exactly that — by POSITION WITHIN THE CHUNK and not by a remainder, because kinds choose their own
 * width. Hand a provider offset where a reference belongs and it is refused, not because somebody
 * remembered to check but because
 * it cannot be one.
 *
 * ⛳ ALL SIX ARE LIVE, AND THEY DIVIDE FOUR WAYS BY WHO MAY WRITE THEM — which is the thing to carry
 * into the rest of the file:
 *
 *     the providers        a RAW chunk offset, nothing counted   the heap, through
 *                                                                `zzpackage_get_heap_provider` and its setter
 *     bindings, program    a counted REFERENCE                   `init` fills them, `read_result` takes them
 *     the status           a state                               the transitions; `zzprivate_set_status`
 *                                                                for the edges nobody can race, and the
 *                                                                engine's C ABI stores it outright
 *     the result kind      a KIND, nothing counted               `finish` writes it, the drain clears it
 *
 * The generic accessors serve the first group and REFUSE the other four, because overwriting a reference
 * drops a hold that is never given up and storing into a state walks around the swap that makes a claim
 * exclusive. ⛳ AND WHAT IS CHECKED IS THE SLOT, NOT THE WRITER: both accessors turn away a `which` above
 * `ARRAYS`, and that is the whole of the enforcement. The other four are reached by direct stores — `init`
 * fills bindings and program, `begin` puts the thawed copy back, `finish` writes the answer and its kind,
 * the drain empties both — and the host fills the same four slots itself through `eng_abi_place` without
 * passing through any of them. ⇒ ★ A REFUSAL KEEPS A GENERIC DOOR OFF A SLOT; IT DOES NOT MAKE ONE WRITER
 * OF IT. That each of those verbs is the only one that should touch what it touches is an argument in this
 * package, and a reader who needs it enforced reads the call sites.
 *
 * ── HOW THEY ARE ORDERED, AND WHAT THE ORDER TELLS YOU ──────────────────────────────────────────────
 * The two providers first, then the two things carved out of them, then the status, then the result kind.
 * A provider names a CHUNK; a user names something INSIDE one — so of the first four, which half a slot
 * sits in answers what kind of offset it holds. The status follows them because it is not an offset at
 * all: it is a state, and the one word two blocks can both reach.
 *
 * ── WHY IT IS ONE NODE ──────────────────────────────────────────────────────────────────────────────
 * Six slots fit in the six words a node carries, so the whole of it is one read on a line the caller is
 * already touching — and that is the whole of the room. ⛔ A NODE IS 64 BYTES UNDER A `static_assert` IN
 * `contracts/objects/heap_node.cuh`; a seventh word makes it 128 and doubles every allocation in the tree,
 * so a seventh thing a base must remember is a change to the node rather than an edit here.
 *
 * The first four are OFFSETS rather than addresses, for the reason everything stored here is: an offset
 * survives the heap being placed at a different address, and it is what the
 * counting is kept against. Zero means "not yet", and can mean it because the heap allocation pool never hands out its
 * first chunk.
 *
 * ── WHERE IT LIVES ──────────────────────────────────────────────────────────────────────────────────
 * ⚖ RULED: in the system register's own allocation, in an array indexed on the block id. One reference
 * reaches every block's base, and it is what gives the language an access point it can name — an opcode
 * written in the lisp has no argument to put a base in, so a base has to be findable rather than passed.
 *
 * ⛳ THE LOOKUP HALF IS BUILT. `SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY` is a row of the register, the
 * boot sets it, and `sys__heap__zzpackage_computing_base` asks the register for that offset and adds the
 * block id — so the access point is a row a program can name rather than a static somebody published.
 * ⛔ WHAT IS OUTSTANDING IS WHERE THE BASE NODE ITSELF SITS: `start_block` carves it out of the chunk the
 * block is handed and writes that offset into the array's slot. Same array, same index, different home —
 * and the finder does not change when the nodes move, because it already goes through the register.
 *
 * ⭐ AND THE MOVE DISSOLVES SOMETHING RATHER THAN JUST TIDYING IT. While a base sits in a chunk it also
 * carves from, freeing it would hand the block's working memory back to the heap allocation pool — which is why the
 * teardown row refuses it, and why that chunk has to be one that never goes home. In the system
 * register's allocation it provides for nothing, so releasing it would give up its own granule and its
 * holds and nothing else. The refusal is a fact about where it lives today, not about its shape.
 *
 * ⛔ SO IT IS NEVER RELEASED, AS THINGS STAND. It outlives everything it names, and handing one to the
 * counting's teardown is a mistake rather than an oversight — refused there loudly rather than quietly
 * doing nothing.
 *
 * ── THE LIFE OF ONE, WHICH IS THE ORDER THIS FILE IS WRITTEN IN ─────────────────────────────────────
 *
 *     claim(me)        FREE          -> COMPUTING+me      one filler wins, everyone else is handed zero
 *     init(me, ...)    COMPUTING+me  -> SCHEDULED         fills it; only what you claimed, while you hold it
 *     begin(owner)     SCHEDULED     -> COMPUTING+owner   the owner thaws its own copy and starts
 *                                    -> ERROR             a thaw allocates, so beginning can fail
 *     finish(owner, x) COMPUTING+own -> OK | ERROR        OK if it ran, ERROR if the machine is unsound
 *     read_result(me)  OK | ERROR    -> FREE              hands the value over and puts the base away
 *                                                         ⛔ answers ZERO at ERROR: there is no object
 *
 * ⭐ THE THAW IS THE RUNNER'S AND NOT THE SCHEDULER'S, AND THAT IS THE ONE PRECISION WORTH WRITING DOWN.
 * A thaw carves from the arena of whoever calls it. `init` is run by whoever SCHEDULES, `begin` by the
 * OWNER — so thawing in `init` would land every block's program in the scheduler's chunks. It would still
 * work, which is exactly the danger: the wrong choice is not an error, it is a silent loss of the locality
 * the whole arrangement exists to get.
 *
 * ⭐ AND A FAILED BEGIN ANSWERS THE WAY A REFUSED `init` DOES — the FAULT CODE in the PROGRAM row and the
 * base at ERROR, which `sys__computing_base__fault` reads without taking anything. So it is one more
 * arrow rather than a state, and it costs the heap nothing at the moment the heap is what failed. ⛔ IT ANSWERS ZERO TO THE OWNER, which is what "not yet" answers too, and
 * that is deliberate: an owner has nothing to run either way and goes back to asking. The base is not
 * stranded, because ERROR is collectable — the requestor drains it and the base returns to FREE.
 *
 * Every edge two callers could race is a compare-and-swap, so each of those has exactly one winner and the
 * loser is told so rather than made to wait: `claim` swaps from FREE, `begin` from SCHEDULED, `finish` from
 * COMPUTING+its own id, `read_result` from OK or ERROR. ⛳ THE REST ARE PLAIN STORES THROUGH
 * `zzprivate_set_status`, AND THAT IS THE SWAP BEING SPENT RATHER THAN SKIPPED — `init` and `begin` reach
 * theirs already holding the base, so there is nobody left to exclude. ⛔ WHAT A STORE DOES NOT CARRY IS
 * THE SWAP'S ORDERING, which is why both publish the fill before the status lands. `status` reads the word
 * without taking anything, which is what lets a caller ask "did it throw?" before it commits to the one
 * collection anybody gets.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
/* ── THE CONTRACT ────────────────────────────────────────────────────────────────────────────────────
 * ⚠ NEW IN THIS FILE — the implementation states every clause below somewhere, but never in one place and
 * never as an obligation. Nothing was moved to write it.
 *
 * WHAT THE CALLER OWES
 *   a base          the OBJECT node — one past the allocation head, which is what `claim`, `begin` and
 *                   `get` answer with and what every verb below takes. NOT the head: `is` steps back for
 *                   itself, so handing it the head asks about whatever sits one node earlier. That test
 *                   runs first in every verb, so a wrong pointer is refused rather than read.
 *   its own id      `claim`, `begin`, `finish` and `read_result` all take a block id. ⛔ NOT CHECKED, AND
 *                   NOT CHECKABLE: the id is the caller's word for who it is. Passing somebody else's is
 *                   claiming or finishing work that is not yours, and every swap below will allow it.
 *   claim then init one without the other is broken in both directions. A claim with no fill leaves a
 *                   base held by nobody; a fill with no claim is the race `init` refuses.
 *   finish what     you began. The FROM state of a swap is the permission being spent, so finishing a
 *   you began       computation you did not begin cannot succeed — but only because the id will not match.
 *   one collection  `read_result` hands the value over ONCE and puts the base away. A second caller gets
 *                   nothing, and that is the surface rather than an implementation detail.
 *   two references  `init` takes a bindings base and a PICTURE of a program, and takes a HOLD OF ITS OWN
 *                   on each. The caller keeps the hold it arrived with and owes nothing.
 *                   ⛔ IT IS A RETAIN AND NOT A TRANSFER, and the reason is the whole shape of a fan-out:
 *                   ONE picture is scheduled onto MANY bases, so a hold that moved could only ever feed
 *                   the first of them. What each base needs is to say "this may not go while I have it",
 *                   which is a count and not a chunk.
 *
 * WHAT IT PROMISES BACK
 *   one winner      every edge two callers could race is a compare-and-swap, so exactly one caller makes
 *                   it and the losers are told so.
 *   a graceful no   a refused `claim` answers ZERO rather than waiting. ⚖ *"my worry is that calling it
 *                   locked means that callers will keep hammering"* — nothing here spins, and a caller
 *                   that wants to try again decides that itself.
 *   a free question `status` and `completed` read the word without swapping it, so asking "did it throw?"
 *                   costs nothing and commits to nothing. That is what makes the pair a try/catch.
 *   an error is a   a computation that failed hands back an ERROR OBJECT in the same slot a value would
 *   VALUE           have used. One result, one collection, no second channel to correlate.
 *   a moved hold    `read_result` TRANSFERS what the base was holding. No count changes, because nothing
 *                   was made or ended — the holding moved.
 *
 * WHAT IT DOES NOT DO
 *   It is never RELEASED. It outlives everything it names, and its row in the kind list refuses rather
 *   than freeing — see `WHERE IT LIVES` above for why that is a fact about its home and not its shape.
 *   It does not WAIT UNBOUNDED, and it waits in only two places, neither of them a question. `begin` backs
 *   off for `SYS__COMPUTING_BASE__POLL_CYCLES` when its swap loses, so an idle block polls instead of
 *   hammering; and the lisp's `sys__result` retries up to `SYS__COMPUTING_BASE__RESULT_TRIES` times with
 *   that same back-off between tries. Every question — `status`, `completed`, `fault` — is a read, and a
 *   refused `claim` answers at once.
 *   It does not COPY in `init`: bindings and program are references handed over and held, not duplicated.
 *   ⛔ `begin` DOES, and it is the one verb here that allocates — it thaws the picture into a list of its
 *   own with `sys__list__deep_copy_from_node_array`, and over a scheduled SNAPSHOT it builds an
 *   environment with `sys__bindings__create`. Both carve, so both can come up short, which is what
 *   `SYS__COMPUTING_BASE__FAULT_NO_THAW` and `SYS__COMPUTING_BASE__FAULT_NO_ENV` say and why the life
 *   cycle above shows `begin` reaching ERROR.
 *   It does not decide WHO. A block id is data here; what makes it true is that one block passes its own.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
/* ── THE CONTRACT, DECLARED ─────────────────────────────────────────────────────────────────────────
 * Ten verbs, in the order the life cycle above runs them. Every one is defined in
 * `computing_base__impl.cuh`, with the argument for it beside the definition.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

static __device__ inline bool        sys__computing_base__is       (const sys__heap_node* computing_base);
static __device__ inline uint32_t    sys__computing_base__status   (const sys__heap_node* computing_base);
static __device__ inline bool        sys__computing_base__completed(const sys__heap_node* computing_base);

/* ── WHY A FATAL BASE IS FATAL, WITHOUT COLLECTING IT ────────────────────────────────────────────────
 * The fault code a failed `init` or `begin` left in the result slot. Answers zero when the base is not at
 * ERROR, and zero ALSO when it is at ERROR and could not say why — which is itself the answer: the
 * failure was an allocation, so describing it would have needed one.
 * ⛳ IT TAKES NOTHING AND CHANGES NOTHING, like `status`, so a caller may ask before deciding what to do
 * — and deciding is the point, because what follows an ERROR is tearing the engine down. */
static __device__ inline uint64_t    sys__computing_base__fault(const sys__heap_node* computing_base);

static __device__ inline sys__heap_node* sys__computing_base__claim (sys__heap_node* computing_base,
                                                                 uint32_t block_id);
static __device__ inline bool        sys__computing_base__init  (sys__heap_node* computing_base,
                                                                 uint32_t block_id,
                                                                 uint64_t bindings_viewonly,
                                                                 uint64_t program_picture);
static __device__ inline sys__heap_node* sys__computing_base__begin (sys__heap_node* computing_base,
                                                                 uint32_t block_id);
/* ⭐ FINISHING TAKES THE ANSWER, BECAUSE PUTTING IT DOWN AND SAYING SO ARE ONE ACT. Whoever is watching
 * this base is watching the status word and nothing else, so a status that lands before the answer is an
 * invitation to collect one that is not there — and two separate verbs would leave that ordering to
 * whoever called them. Handed a value, this places it and then says so; handed nothing, it says so and
 * there is nothing to collect. Either way the running copy of the program is let go of here, which is
 * what the block that ran it owes. */
static __device__ inline bool        sys__computing_base__finish(sys__heap_node* computing_base,
                                                                 uint32_t block_id,
                                                                 uint32_t outcome,
                                                                 sys__heap_node value);
/* ⭐⭐ AND COLLECTING TRANSFERS OWNERSHIP — ⚖ ARCHITECT: *"reading the result transfers ownership, and the
 * method retrieval should set the execution list pointer to null so there is only one holder of that
 * value."* So this hands back the CELL and empties the slot in the same breath: the hold the base was
 * keeping becomes the caller's, and a second collector is handed nothing rather than a second claim on
 * the same thing. Nothing is released on the way out — what would be released is exactly what is being
 * handed over. */
static __device__ inline sys__heap_node sys__computing_base__read_result(sys__heap_node* computing_base,
                                                                         uint32_t block_id);

static __device__ inline sys__heap_node* sys__computing_base__get(uint32_t block_id);

/* ── OUTSIDE THE CONTRACT ───────────────────────────────────────────────────────────────────────────
 * Reachable from the rest of `sys` and no further. The heap reads and writes the two provider slots
 * because it is what carves from them, and `start` is the one placement that cannot go through the
 * ordinary path — a base is a small thing carved out of a chunk, and what carves small things out of a
 * chunk is remembered in a base.
 *
 * ⛳ AND THE FILE'S THREE `zzprivate_` VERBS ARE NOT HERE, WHICH IS THE MARKER DOING ITS JOB. `set_status`,
 * `advance` and `drain` are reached only from inside `computing_base__impl.cuh`, and `zzprivate` means one
 * FILE — so a declaration of any of them in a header would be the violation, not the convenience. A header
 * that cannot name them is the shape of the rule rather than an omission from it.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */
static __device__ inline uint64_t sys__computing_base__zzpackage_get_heap_provider(
                                      const sys__heap_node* computing_base, uint32_t which);
static __device__ inline void     sys__computing_base__zzpackage_set_heap_provider(
                                      sys__heap_node* computing_base, uint32_t which, uint64_t offset);
static __device__ inline bool     sys__computing_base__zzpackage_start(sys__heap_node* computing_base,
                                                                       uint32_t block_id);
/* What the kind list names as this kind's teardown, which refuses rather than freeing. */
static __device__ __noinline__ void sys__computing_base__zzpackage_never_release_internal(sys__heap_node* head,
                                                                                   uint64_t* releaser_stack);

#endif /* SILVANN__PACKAGES_SYS_CPU_COMPUTING_BASE__HEADER_CUH */
