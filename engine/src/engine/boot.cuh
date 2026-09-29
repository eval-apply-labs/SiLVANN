#ifndef SILVANN__ENGINE_BOOT_CUH
#define SILVANN__ENGINE_BOOT_CUH
/* ══ STANDING THE MACHINE UP ═══════════════════════════════════════════════════════════════════════════
 *
 * AI TEMPORARY COMMENT — FOR THE NEXT AGENT; DELETE WHOLE BEFORE RELEASE. Rules in `README.md`.
 * One warning about `ENG_BOOT_BLOCK`, which the ruling ⚖ *"sys and the evaluator go host"* dates.
 *   · The macro used to carry a register-budget table measured on `k_eval`, the evaluator as a kernel
 *     ("21% of the evaluator's time"). The ruling took the evaluator off the card and no kernel named
 *     `k_eval` exists any more, so the table was dropped: on the host there is no occupancy to trade.
 *   · SO 512 MUST BE RE-DERIVED, NOT INHERITED, for what stays on the card: the compute kernels in
 *     `nn/gpu/doors.cuh`. Carrying an evaluator's answer onto a `vector__add` loop because the macro
 *     already said it would be cargo cult.
 * ⛳ RETIREMENT: when the compute kernels have a launch bound of their own, measured.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Everything below this line in the tree assumes an allocator that is already standing: a pool of memory,
 * a table with one computing base per block, and a block that has been started off. This file is what
 * makes those true, and it is the only place that talks to the outside world to do it.
 *
 * ⭐ WHY THE BOOT IS THE ENGINE'S AND NOT A PACKAGE'S. A package is written to be reached from the
 *   language; the engine is what the language wears to reach anything else. Asking a silicon family for
 *   memory and starting runners over it is that and nothing else. A package that could do it would be a
 *   package that had to know which silicon it was running on at all.
 *
 * ⭐⭐ THE ALLOCATOR IS TOLD ITS SIZE, AND NOTHING INSIDE IT COULD WORK THE SIZE OUT. How many chunks a
 *   launch wants follows from two things the allocator cannot know before it exists: how many runners the
 *   launches will start, which is the caller's choice, and how much is alive at the same moment, which is
 *   a property of the program. So the numbers are handed in, and this file is who hands them.
 *
 * ⭐⭐ ZERO IS THE BYPASS, ON EVERY FIELD. A field left zero is one the caller is not claiming and the
 *   boot works out for itself. A field with a number in it is the caller's answer, and the boot either
 *   uses it or refuses it by name — never silently rounds it, and never quietly ignores it.
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* ── WHAT A CALLER ASKS FOR ──────────────────────────────────────────────────────────────────────────
 * ⛳ `chunk_nodes` IS THE FIELD THAT SHOWS WHAT "0 IS THE BYPASS" HAS TO MEAN. It is fixed when the code
 * is compiled: the chunk an object lives in is found by masking that object's own offset, so the width is
 * built into arithmetic on every path and no number arriving at run time could change it. A field that
 * cannot be applied could have been left out — and then a caller sizing its own buffers from a different
 * idea of the chunk would find out by corrupting memory. So the number is taken as a CLAIM and CHECKED,
 * and a caller with the wrong one is told at boot. ⇒ ★ A VALUE YOU CANNOT HONOUR IS STILL WORTH
 * ACCEPTING, IF ACCEPTING IT IS HOW YOU CATCH SOMEBODY BELIEVING IT.
 *
 * ⛔ AND `blocks` IS BOUNDED TWICE, THOUGH ONLY ONE OF THE TWO REFUSES BY NAME. The floor further down
 * asks whether the chunks can seat the blocks and answers ENG__BOOT__POOL_TOO_SMALL when they cannot.
 * The other bound is the reserved tail: the table of computing bases is placed into one chunk-wide slot
 * and takes a head plus one node per block, so the slot seats `SYS__HEAP__CHUNK_NODES -
 * SYS__CHUNK__FIRST - 1` of them, which is 509 today. Nothing asks that question, so a larger grid
 * writes past the allocation instead of being told. `REASONED`, from `reserved_nodes` and the
 * `bases_room` offset at the foot of this file and from `sys__node_array__zzpackage_place`, which
 * writes `elements + 1` nodes. ⇒ ★ A BOUND NOTHING ASKS ABOUT IS ONE THE CALLER FINDS BY CORRUPTING
 * MEMORY — the same ending the `chunk_nodes` claim above is checked to prevent, on a field with no
 * such check. */
typedef struct {
    uint32_t blocks;       /* 0 -> one. Two bounds and only one of them refuses; the note above says.   */
    uint32_t chunks;       /* 0 -> worked out from `blocks`.                                            */
    uint32_t chunk_nodes;  /* 0 -> the compiled width. A number here is checked against it, never used. */
    sys__silicon_family__id family;  /* which silicon to stand up on — the caller's to name, never zero-defaulted,
                                 because zero is a family. With no request at all it is the first offered. */
} EngineBootRequest;

/* ── WHAT IT GOT ─────────────────────────────────────────────────────────────────────────────────────
 * The resolved figures travel back with the pointers rather than being re-derived by whoever needs them.
 * A caller that had to recompute `chunks` from `blocks` would be a second implementation of the rule two
 * dozen lines below, free to disagree with it. */
typedef struct {
    uint32_t         blocks;
    uint32_t         chunks;
    uint32_t         chunk_nodes;
    sys__silicon_family__id   family;    /* the silicon every host operation of this machine names              */
    sys__heap_node*      heap;        /* the pool itself, chunk zero first                                  */
    uint64_t             heap_bytes;  /* and how far it reaches, so a card can be given the whole of it     */
    sys__heap_counters* counters;    /* diagnostic, and the allocator only writes them                     */
    sys__fault_channel*  faults;     /* where a refusal lands, and the only way one reaches the caller     */
    int              status;      /* ENG__BOOT__OK, or the one thing that went wrong                    */
} EngineBoot;

/* Why a boot did not happen. They are separate words rather than one failure because every one of them
 * has a different repair, and a caller that only learns "no" has to guess which. */
#define ENG__BOOT__OK                  0
#define ENG__BOOT__CHUNK_NODES_CLAIM   1   /* the caller believes a chunk is a width it is not          */
/* 2 is a hole. It said "more than one block was asked for", which nothing refuses now that a block can
 * take its own first chunk; the number is left empty rather than closed up, because a word travels out
 * of here and a value meaning one thing in one build and another in the next is worse than a gap. */
#define ENG__BOOT__POOL_TOO_SMALL      3   /* the chunks asked for cannot serve the blocks asked for     */
#define ENG__BOOT__NO_MEMORY           4   /* the silicon family would not give it                     */
#define ENG__BOOT__PUBLISH_REFUSED     5   /* the allocator itself said no, having seen the real launch  */
#define ENG__BOOT__NO_BASE             6   /* the block started and came back without a computing base   */
#define ENG__BOOT__LAUNCH_FAILED       7   /* the work did not run at all                               */
#define ENG__BOOT__REGISTER_REFUSED    8   /* the room was set aside and the register would not take it  */
/* ⛳ THE MACHINE WAS WHOLE AND A PACKAGE WOULD NOT STAND UP ON IT, which is a different failure from
 * every one above: those are the language failing to come up, this is the language up and a package
 * refusing on it. It also covers a deferral that never settles — the driver gives up on a ceiling rather
 * than spinning, so a dependency a package waits for and never gets is a word rather than a hang. */
#define ENG__BOOT__PACKAGES_REFUSED    9   /* the language stood up and a package's init would not run   */
#define ENG__BOOT__NO_SILICON         10   /* no silicon family was offered for the id asked for            */

static inline const char* eng__boot__why(int status) {
    switch (status) {
        case ENG__BOOT__OK:                 return "up";
        case ENG__BOOT__CHUNK_NODES_CLAIM:  return "the chunk width asked for is not the one this binary was compiled with";
        case ENG__BOOT__POOL_TOO_SMALL:     return "the pool cannot hold two chunks for every block in the launch";
        case ENG__BOOT__NO_MEMORY:          return "the silicon family would not allocate the pool";
        case ENG__BOOT__PUBLISH_REFUSED:    return "the allocator refused the pool it was handed";
        case ENG__BOOT__NO_BASE:            return "the block started without a computing base";
        case ENG__BOOT__LAUNCH_FAILED:      return "the boot work did not run";
        case ENG__BOOT__REGISTER_REFUSED:   return "the system register refused the room set aside for it";
        case ENG__BOOT__PACKAGES_REFUSED:   return "the language stood up and a package's init refused or never settled";
        case ENG__BOOT__NO_SILICON:         return "no silicon family was offered for the id asked for";
        default:                            return "unknown";
    }
}

/* ── HOW MANY CHUNKS, WHEN NOBODY SAYS ───────────────────────────────────────────────────────────────
 * There are two different numbers here and reading them as one is the mistake to avoid.
 *
 *   THE FLOOR is what lets a launch START: every block holds two chunks at once, one for what lasts and
 *   one for what churns, and chunk zero never enters the free ring — so a pool of n offers n-1 and the
 *   floor is two per block plus the one that is never offered. Under it the allocator does not run
 *   slowly, it refuses. That number is the allocator's and it checks it itself.
 *
 *   THE HEADROOM is what lets a program FINISH, and it belongs to the program rather than to either of
 *   us. A run that makes many calls in sequence releases as it goes and needs almost none; one that nests
 *   deeply holds every level at once and needs it in proportion to the depth.
 *
 * ⭐ SO THE DEFAULT IS GROUNDED IN WHAT HAS ACTUALLY RUN RATHER THAN DERIVED. `MEASURED`: the deepest
 * program the suite has evaluated peaked at 44 chunks, and every program measured so far has run inside
 * 64. That is where this number comes from, it is roughly half again the deepest peak, and it is a
 * STARTING POINT AND NOT A RULE — which is exactly why the field exists for a caller who knows better.
 * ⛳ WHAT WOULD FALSIFY IT: a program that runs out. The allocator counts refusals, and a boot that hands
 * those counters back is what makes running out visible rather than fatal-and-unexplained. */
#define ENG__BOOT__CHUNKS_PER_BLOCK 64u

/* ⛔ THE GUARD MEANS "A REAL BUILD, NOT THE HARNESS". ▶ `abi.cuh`'s copy of this note for what a
 * guarded-out file does to a probe. */
#ifndef SYS__SILICON__HARNESS   /* a real build, not a test harness that brings its own silicon */

/* ── WHAT THIS FILE ASKS OF A SILICON FAMILY ─────────────────────────────────────────────────────────
 * Memory, a way to reach it, and a way to wait — all of it through `sys__gpu__memory_*` and
 * `sys__gpu__compute_completed`, each taking the family, declared in
 * `packages/sys/cpu/silicon/silicon_family.cuh` and answered by whichever family under `silicon_families/`
 * the caller named. This file names no vendor runtime, and neither does the interface next door.
 * ⛳ THE GUARD ABOVE IS ABOUT AVAILABILITY, NOT CHOICE. Nothing here picks a family — the request names
 * one — and a harness, which brings its own silicon, compiles none of this. */

/* ── THE FIRST PIECE OF WORK: PUBLISHING ─────────────────────────────────────────────────────────────
 * One runner does all of it, and every other runner in the launch is here to be counted.
 *
 * ⭐⭐ THE LAUNCH IS FULL WIDTH THOUGH ONE RUNNER WORKS, AND THAT IS THE POINT OF IT. The allocator
 * checks the pool it is handed against `sys__silicon__block_count()`, which answers how many runners the
 * current launch started. Publishing from a one-runner launch would hand that check a width of one, and it
 * would pass on a pool too small for the blocks started after it. Both widths come from the one `blocks`
 * the host resolved, so this check agrees with the floor `eng__boot` asks first by construction; that
 * floor is the refusal a caller sees, and this one is what the allocator keeps for callers that are not
 * the boot. `REASONED`, from `eng__launch__spread` and `sys__heap__publish`.
 * ⇒ ★ A CHECK THAT COMPARES TWO NUMBERS IS ONLY WORTH RUNNING WHERE BOTH OF THEM ARE TRUE. */
/* ⛳ `static`, LIKE EVERYTHING ELSE HERE. A kernel is launched from the one translation unit that
 * defines it, so it needs no external linkage — and the property being kept is that a definition
 * without `static` is reaching for something outside a build that has no outside. */
/* ── THE BLOCK GEOMETRY, DECLARED ONCE AND THE SAME FOR EVERY KERNEL HERE ────────────────────────────
 * `ENG_BOOT_BLOCK` stands where a launch-geometry hint goes on every kernel in this file and in `abi/`.
 * Nothing in the engine is compiled for a card, and `cpu/silicon/environment.cuh` defines
 * `__launch_bounds__(...)` away, so the macro expands to nothing: a runner is a host thread and has no
 * register budget to declare. `REASONED`, from that definition.
 * ⛔ THE KERNELS THAT DO RUN ON A CARD, in `nn/gpu/doors.cuh`, do not wear this macro, and their
 * occupancy needs are `UNMEASURED`. */
#define ENG_BOOT_BLOCK __launch_bounds__(512, 1)

static __global__ ENG_BOOT_BLOCK void eng__boot__publish(sys__heap_node* heap, uint32_t chunks, sys__heap_counters* counters,
                                   sys__fault_channel* faults, uint64_t reserved_nodes, int* status) {
    if (sys__silicon__block_id() != 0u) return;
    /* ⛔ THE FAULT CHANNEL FIRST, AND IT IS THE ONE ORDERING IN THIS KERNEL THAT IS NOT A PREFERENCE.
     * The call below refuses a pool too small for the launch by RAISING, so a channel handed over after
     * it would miss the only fault the boot can produce on its own — and the refusal would arrive at the
     * host as a status with no word to say what it was. */
    sys__fault__publish(faults);
    sys__heap__forget_base();
    *status = sys__heap__publish(heap, chunks, counters, reserved_nodes) ? ENG__BOOT__OK
                                                                        : ENG__BOOT__PUBLISH_REFUSED;
    sys__heap__stop_carving();
}

/* The runner a launch of more than one block starts it through. */
typedef struct {
    sys__heap_node*     heap;
    uint32_t            chunks;
    sys__heap_counters* counters;
    sys__fault_channel* faults;
    uint64_t            reserved_nodes;
    int*                status;
} eng__boot__publish__args;
static void eng__boot__publish__run(const void* at) {
    const eng__boot__publish__args* a = (const eng__boot__publish__args*)at;
    eng__boot__publish(a->heap, a->chunks, a->counters, a->faults, a->reserved_nodes, a->status);
}

/* ── THE LAST: GIVING EACH BLOCK SOMETHING TO WORK WITH ──────────────────────────────────────────────
 * ⛳ WRITTEN ABOVE THE REGISTER SECTION BELOW AND RUN AFTER IT, which is the one place in this file
 * where reading order and running order part. That section states why the two cannot be exchanged, and
 * the host issues all three in running order at the foot of the file.
 * A block cannot allocate until it has a computing base, and a computing base is itself allocated — so
 * one is placed by hand, in a chunk chosen from outside, and everything after it is ordinary.
 *
 * ⛔⛔ THE FIRST BLOCK IS HANDED ITS ROOM AND EVERY OTHER ONE TAKES ITS OWN, AND WHICH CHUNK IS SAFE TO
 * HAND OVER IS THE WHOLE OF THE REASON. Chunk zero is never offered to the free ring, so it is the one
 * chunk this side can give away knowing nothing else will be given it too. Every other chunk in the pool
 * is in that ring and could be handed to an allocation a moment later — so a block after the first has to
 * TAKE one, from inside, where taking is the allocator's own business. That is what `start_own_block` is,
 * and it is why this kernel names two entries rather than passing a different chunk to one.
 *
 * ⭐ AND THAT IS ALSO WHY THEY ALL RUN AT ONCE RATHER THAN IN TURN. The ring is compare-and-swapped at
 * both cursors, so blocks reaching for a chunk in the same instant is the case it was built for; nothing
 * here orders them and nothing needs to. */
/* ⛔ THE STATUS WORD IS KNOWN GOOD ON THE WAY IN, WHICH IS WHAT LETS ONLY FAILURES WRITE IT. The host runs
 * this only after the launch before it reported success, so what is in the word is that success — and a
 * block that starts correctly has nothing to add to it. A failing block swaps the word from good to its
 * own reason, so the FIRST failure is what a caller reads and later ones do not paper over it. Writing it
 * unconditionally from every block would let a success land after a failure and lose it. */
/* ── AND THE SECOND: THE SYSTEM REGISTER, IN ROOM SET ASIDE FOR IT ───────────────────────────────────
 * ⭐⭐ THE ROOM IS INSIDE THE ALLOCATION AND OUTSIDE THE POOL, which is the one place that satisfies both
 * things the register needs. Its offset has to be a real one, so it must lie within the memory the
 * allocator was published over; and it must never be handed to anybody, so it must lie past the chunks
 * the allocator was told it has. Those are different boundaries, and the gap between them is where this
 * goes. ⇒ ★ AN OFFSET IS A PROPERTY OF THE ALLOCATION; BEING HANDED OUT IS A PROPERTY OF THE POOL.
 * ⛔⛔ AND IT MUST RUN BEFORE THE BLOCKS ARE STARTED, WHICH IS NOT A PREFERENCE. Starting a block is
 * what writes that block's computing base, and where a base goes is the slot this register's first row
 * names — so a block started before this ran would have nowhere to say what it is working with. The
 * ordering is the host's, as everywhere else in this file — each launch joins its runners before the next
 * is issued — and it is the one place in the boot where two steps genuinely cannot be exchanged. */
static __global__ ENG_BOOT_BLOCK void eng__boot__publish_register(sys__heap_node* room, sys__heap_node* bases_room,
                                                   uint32_t blocks, int* status) {
    if (sys__silicon__block_id() != 0u) return;
    sys__heap__forget_base();
    *status = sys__system_register__publish(room, bases_room, blocks) ? ENG__BOOT__OK
                                                                     : ENG__BOOT__REGISTER_REFUSED;
    sys__heap__stop_carving();
}

/* The runner a launch of more than one block starts it through. */
typedef struct {
    sys__heap_node* room;
    sys__heap_node* bases_room;
    uint32_t        blocks;
    int*            status;
} eng__boot__publish_register__args;
static void eng__boot__publish_register__run(const void* at) {
    const eng__boot__publish_register__args* a = (const eng__boot__publish_register__args*)at;
    eng__boot__publish_register(a->room, a->bases_room, a->blocks, a->status);
}

static __global__ ENG_BOOT_BLOCK void eng__boot__start_blocks(sys__heap_node* chunk_zero, int* status) {
    sys__heap__forget_base();
    /* ⛳ ONE CHECK, BECAUSE THERE IS ONLY ONE THING TO ASK NOW. Starting a block places its base AND
     * writes it into the slot the register's first row names, and it raises if that row is not standing
     * — so a non-null answer already means both halves happened. Confirming it a second time would mean
     * reading the slot, and the table that slot is in belongs to the allocator rather than to this
     * tier: the name that reaches it is package-private, and reaching for it to re-check somebody
     * else's work is reaching for the wrong reason. */
    const sys__heap_node* base = (sys__silicon__block_id() == 0u) ? sys__heap__start_block(chunk_zero)
                                                                 : sys__heap__start_own_block();
    if (base == 0)
        (void)sys__silicon__cas_u32((unsigned int*)status, (unsigned int)ENG__BOOT__OK,
                                    (unsigned int)ENG__BOOT__NO_BASE);
    /* ⛔ AND THERE IS ONE WAY OUT FOR EVERY PATH, WHICH IS WHY A FAILURE ABOVE IS A TEST AND NOT A
     * RETURN. Whatever this block was carving from has its head in block-local storage and the copy in
     * the pool is behind until this runs; a launch that ends without it leaves the next one arming from
     * a stale watermark. An early return is the easiest way to owe that and not pay it. */
    sys__heap__stop_carving();
}

/* The runner a launch of more than one block starts it through. */
typedef struct {
    sys__heap_node* chunk_zero;
    int*            status;
} eng__boot__start_blocks__args;
static void eng__boot__start_blocks__run(const void* at) {
    const eng__boot__start_blocks__args* a = (const eng__boot__start_blocks__args*)at;
    eng__boot__start_blocks(a->chunk_zero, a->status);
}

/* ── AND TAKING THE MACHINE DOWN, WHICH IS A THING THE PACKAGES DO ───────────────────────────────────
 * ⛔⛔ FREEING THE MEMORY IS NOT SHUTTING THE ENGINE DOWN. Everything the packages publish — where the
 * pool is, where the register is, which ring is the free list — lives in package statics, and freeing
 * the memory reaches none of them. Without this, a machine whose memory has gone on answering "published" to
 * every question it is asked, holding a base that is an offset into a pool somebody else owns; the next
 * boot then finds a register already standing and every block looks for its computing base through that
 * offset. It presents as "the block started without a computing base", which names the symptom three
 * steps downstream of the cause.
 * ⇒ ★ A TEARDOWN THAT ONLY GIVES THE MEMORY BACK IS HALF A TEARDOWN.
 *
 * ⛳ THE ORDER IS THE REGISTER AND THEN THE ALLOCATOR, and it is the boot's own order reversed. The
 * register's room is inside the pool, so it stops being addressable when the allocator does; standing it
 * down first means nothing ever asks the allocator about a register that is already gone. */
static __global__ ENG_BOOT_BLOCK void eng__boot__stand_down(void) {
    if (sys__silicon__block_id() != 0u) return;
    /* ⛳ THE PAIRING IS HERE THOUGH NOTHING IS CARVED, AND THE ORDER IS WHAT MAKES IT MEAN ANYTHING.
     * Arming empties the block-local carving state, so the flush below has nothing to write; the flush
     * then happens while there is still a pool to write into, which is the last moment it could. Doing
     * it after the retirement would be writing a watermark into memory that is on its way back. Every
     * kernel in this tree arms and flushes, and a kernel claiming an exception would have to be believed
     * by every later reader rather than checked. */
    sys__heap__forget_base();
    sys__heap__stop_carving();
    sys__system_register__zzengine_retire();
    sys__heap__zzengine_retire();
}

/* Hand everything back. It takes a boot that failed halfway as readily as one that worked, because the
 * entry below uses it for exactly that — and a release that only accepted whole things would need its
 * caller to work out how far the boot got before it could ask for the rest to be let go. */
static inline void eng__boot_release(EngineBoot* got) {
    if (got == 0) return;
    /* ⛳ ONLY WHERE THERE WAS A POOL TO PUBLISH. A boot that failed before `publish` ran has nothing
     * standing in the packages, and launching into a machine that was never stood up would be asking the
     * packages to retire something they have never heard of. */
    if (got->heap != 0) {
        ENG__LAUNCH_ONE(eng__boot__stand_down);
        (void)sys__gpu__compute_completed(got->family);
    }
    sys__gpu__memory_free(got->family, got->counters);
    sys__gpu__memory_free(got->family, got->faults);
    sys__gpu__memory_free(got->family, got->heap);
    got->heap = 0; got->counters = 0; got->faults = 0; got->heap_bytes = 0ull;
    got->blocks = 0; got->chunks = 0;
}

/* ── AND THE HOST HALF, WHICH IS THE WHOLE OF THE OUTSIDE WORLD'S ENTRY ──────────────────────────────
 * Resolve what was not asked for, refuse what cannot be given, take the memory, and run the three
 * pieces of work in order.
 *
 * ⭐⭐ THREE LAUNCHES AND NOT ONE, AND IT IS NOT A PREFERENCE. Publishing has to finish before the
 * register is placed, and the register has to be standing before any block starts, and there is no
 * rendezvous inside a launch to arrange either — nothing here waits for the rest of the runners. What
 * orders them is that a launch is synchronous: `ENG__LAUNCH_MANY` joins every runner before it returns, so
 * each piece starts on a machine the one before it has finished with.
 * `REASONED`, from `eng__launch__spread`. ⛳ It also means the shape does not change if the
 * block count later has to come down; only the number does.
 *
 * ⛳ THE POOL IS NOT CLEARED AND MUST NOT NEED TO BE. A chunk carries its own occupancy in its first
 * node, so memory nobody has written reads as full — and making it read empty is exactly what publishing
 * does. Clearing it here would be paying twice for the same property, on a path that is about to write
 * every one of those words anyway. The SMALL allocations beside it are cleared, because nothing else
 * writes them: a counter nobody has touched must read as nothing counted, a status word nobody has
 * failed into must read as the good one — which is what lets only a failing block write it — and a
 * channel nobody has raised into must read as nothing having gone wrong, which is the one of the three
 * a caller acts on.
 * ⛳ AND THE TABLE OF COMPUTING BASES IS NOT AMONG THEM, BECAUSE IT NEEDS NOT TO BE. It is placed in the
 * reserved tail — inside the allocation, past the pool, and cleared by nobody — and the placing writes
 * the null value into every element before the array is handed out. So a block that has not started
 * reads as having no base rather than as having a stale one, and that property belongs to
 * `sys__node_array__zzpackage_place` rather than to anything done here. `REASONED`, from that function
 * and the `zzprivate_fill` it calls. */
static inline bool eng__boot(const EngineBootRequest* want, EngineBoot* got) {
    if (got == 0) return false;
    EngineBootRequest asked = {0u, 0u, 0u, sys__silicon_family__first()};
    if (want) asked = *want;

    got->blocks = 0u; got->chunks = 0u; got->chunk_nodes = 0u;
    got->heap = 0; got->counters = 0; got->faults = 0;
    got->status = ENG__BOOT__OK;

    /* ⛳ THE SILICON FIRST, because every step below names it. A family nobody offered is refused here
     * rather than at the first allocation, where it would read as silicon that had no memory. */
    got->family = asked.family;
    if (sys__silicon_family__of(got->family) == 0) {
        got->status = ENG__BOOT__NO_SILICON;
        return false;
    }

    /* The claim, checked. Nothing is applied — the compiled width is the only one there is. */
    if (asked.chunk_nodes != 0u && asked.chunk_nodes != (uint32_t)SYS__HEAP__CHUNK_NODES) {
        got->status = ENG__BOOT__CHUNK_NODES_CLAIM;
        return false;
    }
    got->chunk_nodes = (uint32_t)SYS__HEAP__CHUNK_NODES;

    got->blocks = (asked.blocks == 0u) ? 1u : asked.blocks;

    got->chunks = (asked.chunks == 0u) ? (got->blocks * ENG__BOOT__CHUNKS_PER_BLOCK) : asked.chunks;
    /* The same floor the allocator enforces, asked here so the answer names the pool rather than
     * arriving as a fault from inside a launch. It checks again on its own account, against the runner
     * count of the publishing launch — which is this same `blocks`, so the two cannot disagree and this
     * is the refusal a caller sees. */
    if (got->chunks == 0u || (uint64_t)(got->chunks - 1u) < (uint64_t)got->blocks * 2ull) {
        got->status = ENG__BOOT__POOL_TOO_SMALL;
        return false;
    }

    /* ⛳ THE ALLOCATION IS THE POOL PLUS THE REGISTER'S ROOM, AND ONLY THE POOL IS PUBLISHED. The
     * allocator is told `chunks` and nothing about the tail, so it can never hand the register's room to
     * anybody — while the room is still inside the base the allocator was given, so its offset is real.
     * ⛳ AND THE SIZE IS THE HEAP'S GEOMETRY RATHER THAN THE REGISTER'S OWN FIGURE: one chunk-wide slot
     * per placed object, worked out below, because what a reservation has to answer to is the chunk
     * arithmetic every offset predicate does and not the row count of what goes in it.
     * ⚠ WHICH LEAVES ONE COUPLING WITH NOTHING WATCHING IT. `SYS__SYSTEM_REGISTER__NODES` is read
     * nowhere — `grep -rn SYS__SYSTEM_REGISTER__NODES src/` finds its own `#define` and nothing else —
     * so a register whose rows outgrew a chunk would be placed in a slot too small to hold them and the
     * boot would say nothing. `REASONED`, from the reservation below and the placing it feeds. */
    const size_t pool_nodes     = (size_t)got->chunks * (size_t)SYS__HEAP__CHUNK_NODES;
    /* ⛳ TWO SLOTS, ONE PER PLACED OBJECT, EACH A WHOLE CHUNK WIDE, THOUGH THE REGISTER NEEDS EIGHT
     * NODES — ONE PER ROW AND ITS HEAD — AND THE SURPLUS IS THE POINT. Every predicate in the package that
     * takes an offset apart masks it to a chunk and asks where it sits inside one; a reservation that is
     * chunk-shaped and chunk-aligned therefore answers those questions the way a carved object does, and
     * removes any need to reason about one straddling a boundary. A chunk is 512 nodes of 64 bytes, so a
     * slot is 32 KB and the pair of them 64 KB: a tighter reservation would give nearly all of that back
     * and buy a class of offset that looks like nothing else. */
    const size_t reserved_nodes = (size_t)SYS__HEAP__CHUNK_NODES * 2u;
    const size_t pool_bytes     = (pool_nodes + reserved_nodes) * sizeof(sys__heap_node);
    const size_t counters_bytes = sizeof(sys__heap_counters);
    const size_t faults_bytes   = sizeof(sys__fault_channel);
    int* reported = 0;

    got->heap_bytes = (uint64_t)pool_bytes;
    if (!sys__gpu__memory_allocate(got->family, (void**)&got->heap,     pool_bytes)     ||
        !sys__gpu__memory_allocate(got->family, (void**)&got->counters, counters_bytes) ||
        !sys__gpu__memory_allocate(got->family, (void**)&got->faults,   faults_bytes)   ||
        !sys__gpu__memory_allocate(got->family, (void**)&reported,      sizeof(int))    ||
        !sys__gpu__memory_zerofill(got->family, got->counters, counters_bytes)         ||
        !sys__gpu__memory_zerofill(got->family, got->faults,   faults_bytes)           ||
        !sys__gpu__memory_zerofill(got->family, reported,      sizeof(int))) {
        sys__gpu__memory_free(got->family, reported);
        eng__boot_release(got);
        got->status = ENG__BOOT__NO_MEMORY;
        return false;
    }

    /* ⛳ THE STATUS IS READ BACK AFTER EACH OF THE THREE PIECES RATHER THAN AFTER THE LAST, AND EACH
     * LAUNCH IS GATED ON WHAT THE ONE BEFORE IT LEFT THERE. A piece is only meaningful if the piece
     * before it worked — placing a register into a pool the allocator refused, or starting a block
     * against that same pool, would fail for a reason that has nothing to do with the register or the
     * block. */
    {
        const eng__boot__publish__args args = { got->heap, got->chunks, got->counters, got->faults,
                                                 (uint64_t)reserved_nodes, reported };
        ENG__LAUNCH_MANY(eng__boot__publish__run, got->blocks, &args);
    }
    if (!sys__gpu__compute_completed(got->family) ||
        !sys__gpu__memory_read(got->family, &got->status, reported, sizeof(int))) {
        got->status = ENG__BOOT__LAUNCH_FAILED;
    }
    if (got->status == ENG__BOOT__OK) {
        /* ⛳ EACH PLACED OBJECT AT THE OFFSET A CARVE WOULD HAVE USED, one chunk-sized slot apart: the
         * first nodes of a chunk describe the chunk, so an allocation begins past them, and a head
         * placed there hands back a reference where every predicate expects one to sit. */
        const eng__boot__publish_register__args args = {
            got->heap + pool_nodes + (size_t)SYS__CHUNK__FIRST,
            got->heap + pool_nodes + (size_t)SYS__HEAP__CHUNK_NODES + (size_t)SYS__CHUNK__FIRST,
            got->blocks, reported };
        ENG__LAUNCH_MANY(eng__boot__publish_register__run, got->blocks, &args);
        if (!sys__gpu__compute_completed(got->family) ||
            !sys__gpu__memory_read(got->family, &got->status, reported, sizeof(int))) {
            got->status = ENG__BOOT__LAUNCH_FAILED;
        }
    }

    /* AND THE BLOCKS LAST, because starting one writes into the slot the register's first row names. */
    if (got->status == ENG__BOOT__OK) {
        const eng__boot__start_blocks__args args = { got->heap, reported };
        ENG__LAUNCH_MANY(eng__boot__start_blocks__run, got->blocks, &args);
        if (!sys__gpu__compute_completed(got->family) ||
            !sys__gpu__memory_read(got->family, &got->status, reported, sizeof(int))) {
            got->status = ENG__BOOT__LAUNCH_FAILED;
        }
    }

    sys__gpu__memory_free(got->family, reported);
    if (got->status != ENG__BOOT__OK) { eng__boot_release(got); return false; }
    return true;
}

#endif  /* a backend is selected */

#endif /* SILVANN__ENGINE_BOOT_CUH */
