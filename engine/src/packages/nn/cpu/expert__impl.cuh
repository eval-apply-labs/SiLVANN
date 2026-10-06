#ifndef SILVANN__PACKAGES_NN_CPU_EXPERT__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_EXPERT__IMPL_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "expert__header.cuh"           /* the contract these definitions answer */
#include "package__header.cuh"          /* this package's id, and its hatch rows */
#include "../../sys/cpu/package__header.cuh"   /* where a package keeps its own device-wide state */
#include "../../sys/cpu/heap_node__header.cuh" /* the node the row is written as */
#include "../../sys/cpu/settings.cuh"          /* the arena's size, read on the card this time */
#include "../../sys/cpu/list__header.cuh"      /* the handed list is an ordinary list, with the ordinary atomics */
#include "../contracts/objects/expert.cuh" /* its constants and fault words */
#define NN__EXPERT__KEY_UP(n)    "nn__expert__type" #n "__up_bytes"
#define NN__EXPERT__KEY_DOWN(n)  "nn__expert__type" #n "__down_bytes"
#define NN__EXPERT__KEY_SLOTS(n) "nn__expert__type" #n "__slots"
/* ⛔ THE ONE OPTIONAL KEY, AND ITS ABSENCE IS THE SAFE ANSWER RATHER THAN THE COMMON ONE — ▶ the polarity
 * note in the header. A config says `anchored:0` to let a collection be evicted; saying nothing, or saying
 * it wrong, anchors. */
#define NN__EXPERT__KEY_ANCHORED(n) "nn__expert__type" #n "__anchored"

static __device__ inline sys__heap_node nn__expert__zzprivate_arena_row(void) {
    return sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_ARENA);
}

static __device__ inline uint64_t nn__expert__arena_at(void) {
    const sys__heap_node row = nn__expert__zzprivate_arena_row();
    if (row.dtype != SYS__KIND__VALUE_INT) return 0ull;
    return row.args[NN__EXPERT__ARENA_AT];
}

static __device__ inline uint64_t nn__expert__arena_bytes(void) {
    const sys__heap_node row = nn__expert__zzprivate_arena_row();
    if (row.dtype != SYS__KIND__VALUE_INT) return 0ull;
    return row.args[NN__EXPERT__ARENA_BYTES];
}

/* ── STANDING THE ARENA UP ───────────────────────────────────────────────────────────────────────────
 *
 * ⭐ IT WRITES A ROW AND TAKES NOTHING. Every other thing this package stands up at init is an OBJECT —
 * a node_array, a list, a buffer — and this is a pair of integers, so its row is a `VALUE_INT` and there
 * is no hold anywhere in it. ⇒ ⛳ WHICH IS WHY IT NEEDS NO TEARDOWN: there is nothing to give back, and a
 * region's identity is arithmetic over memory somebody else owns.
 *
 * ⛔⛔ THE THREE REFUSALS ARE THREE DIFFERENT MISTAKES AND ARE KEPT APART:
 *     the key is unreadable      the two tiers disagree about one text — `FAULT_CONFIG`, the carve's own
 *     the region walks past      the arena does not fit what the buffers left — `FAULT_ROOM`
 *     the row will not write     the hatch is not standing, which is the boot's problem and not this one
 * ⛳ AND A ZERO-BYTE ARENA CANNOT ARRIVE HERE: `sys__settings__size` refuses a figure of 0, so a config
 * that writes `nn__expert_cache_l1__size_mb:0` is refused on BOTH tiers rather than standing up a region
 * with no bytes in it that every later reader would have to have an opinion about. */
static __device__ inline bool nn__expert__zzpackage_stand_arena(uint64_t from,
                                                                uint64_t span_at,
                                                                uint64_t span_bytes) {
    if (span_at == 0ull) return false;

    uint64_t bytes = 0ull;
    if (!sys__settings__size((const uint8_t*)NN__EXPERT__KEY_ARENA,
                             sizeof(NN__EXPERT__KEY_ARENA) - 1ull, &bytes)) return false;

    /* ⛔⛔ TWO COMPARISONS, AND BETWEEN THEM THEY ARE EVERY WAY THIS CAN BE WRONG. The subtraction is what
     * the buffers already spent, so both tests are against the span's own LENGTH and never against an
     * address that could wrap — the same shape the carve uses.
     * ⭐⭐ AND `spent > span_bytes` IS DOING MORE WORK THAN IT LOOKS. A `from` BEFORE the span's start —
     * including the `from` of 0 a failed carve would report — makes that subtraction wrap to a number
     * within a few of 2^64, which is refused by the same line. ⇒ ★ SO THE EXPLICIT GUARDS THIS HAD FOR
     * BOTH CASES ARE GONE: an arm deleting them left the suite GREEN, because no input can distinguish
     * them from this comparison. A check nothing can falsify is a check nobody can see working, and the
     * honest form is one comparison with its reach written down. */
    const uint64_t spent = from - span_at;
    if (spent > span_bytes) return false;                /* past the end — or, by wrapping, before the start */

    /* ⚖⚖ AND THE BASE IS PULLED UP TO A 4 KiB BOUNDARY — RULED.
     * ⛔⛔ THIS HAPPENS AFTER THE GUARD ABOVE AND THE ORDER IS LOAD-BEARING, not a preference. Rounding
     * `from` UP first would let an address BELOW the span round INTO it — `span_at - 8` becomes `span_at`
     * — and the wrapped-subtraction refusal the block above exists for would pass with a plausible base.
     * ⇒ ★ AN ALIGNMENT IS A CHANGE OF ADDRESS, SO IT BELONGS AFTER EVERY TEST ABOUT WHICH ADDRESS IT WAS.
     * ⛳ AND THE ROOM FOR IT IS RESERVED BY THE HOST as a fixed BOUND rather than the same pad computed a
     * second time — ▶ the span sum in `package__impl.cuh`, which says why. */
    uint64_t at = from;
    const uint64_t slack = at % NN__EXPERT__PAGE_ALIGN;
    if (slack != 0ull) {
        const uint64_t add = NN__EXPERT__PAGE_ALIGN - slack;
        if (at > 0xFFFFFFFFFFFFFFFFull - add) return false;
        at += add;
    }
    const uint64_t aligned_spent = at - span_at;
    if (aligned_spent > span_bytes) return false;        /* the pad itself walked past the end */
    if (bytes > span_bytes - aligned_spent) return false; /* the region itself would walk past the end */

    sys__heap_node row;
    row.dtype = SYS__KIND__VALUE_INT; row.num_args = 0u; row.op_code = 0ull;
    row.args[NN__EXPERT__ARENA_AT]    = at;
    row.args[NN__EXPERT__ARENA_BYTES] = bytes;
    return sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_ARENA, &row);
}

/* ══ ⭐⭐⭐ THE TYPE TABLE — THE PAGE LIST DESCRIPTOR ══════════════════════════════════════════════════
 * ▶ `expert__header.cuh` for what a row holds and which ruling put each word there. What is here is the
 * mechanism, and it is small because the collapse did the work: every slot in a collection is the same
 * size, so placement is arithmetic and the table is the only thing that has to be stood.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* Round UP to the slot alignment, answering false rather than wrapping. ⛳ UP, because a run short of its
 * own arithmetic would hand its last slot fewer bytes than the table promised. */
static __device__ inline bool nn__expert__zzprivate_align_up(uint64_t bytes, uint64_t* answer) {
    const uint64_t slack = bytes % NN__EXPERT__PAGE_ALIGN;
    if (slack == 0ull) { *answer = bytes; return true; }
    const uint64_t add = NN__EXPERT__PAGE_ALIGN - slack;
    if (bytes > 0xFFFFFFFFFFFFFFFFull - add) return false;
    *answer = bytes + add;
    return true;
}

static __device__ inline uint64_t nn__expert__zzprivate_types_array(void) {
    const sys__heap_node row = sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_TYPES);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return row.args[0];
}

static __device__ inline uint64_t nn__expert__types(void) {
    const uint64_t array = nn__expert__zzprivate_types_array();
    if (array == 0ull) return 0ull;
    return sys__node_array__length(array);
}

/* One field of one row. ⛳ ONE READER FOR ALL FOUR, because the only thing that differs is the argument —
 * and writing four walks would be four places for the bounds test to be got subtly differently. */
static __device__ inline uint64_t nn__expert__zzprivate_type_word(uint64_t type, unsigned which) {
    const uint64_t array = nn__expert__zzprivate_types_array();
    if (array == 0ull) return 0ull;
    if (type >= sys__node_array__length(array)) return 0ull;
    const sys__heap_node row = sys__node_array__get(array, type);
    if (row.dtype != SYS__KIND__VALUE_INT) return 0ull;
    return row.args[which];
}

static __device__ inline uint64_t nn__expert__type_at(uint64_t type) {
    return nn__expert__zzprivate_type_word(type, NN__EXPERT__TYPE_AT);
}
static __device__ inline uint64_t nn__expert__type_slots(uint64_t type) {
    return nn__expert__zzprivate_type_word(type, NN__EXPERT__TYPE_SLOTS);
}
static __device__ inline uint64_t nn__expert__type_bytes(uint64_t type) {
    return nn__expert__zzprivate_type_word(type, NN__EXPERT__TYPE_BYTES);
}
static __device__ inline uint64_t nn__expert__type_offset(uint64_t type) {
    return nn__expert__zzprivate_type_word(type, NN__EXPERT__TYPE_OFFSET);
}
static __device__ inline uint64_t nn__expert__type_free(uint64_t type) {
    return nn__expert__zzprivate_type_word(type, NN__EXPERT__TYPE_FREE);
}

/* ⭐ NO SPECIAL CASE HERE, AND THAT IS THE POINT. `zzprivate_type_word` answers `0` down every path where
 * it cannot read the word — no table, a type past the end, a row of the wrong dtype — and `0` is what an
 * ANCHORED collection stores. So the safe answer is the one the mechanism produces when it is broken.
 * ▶ the header's polarity note for why the storage is the inverse of the name. */
static __device__ inline bool nn__expert__type_anchored(uint64_t type) {
    return nn__expert__zzprivate_type_word(type, NN__EXPERT__TYPE_EVICTABLE) == 0ull;
}

/* ⭐⭐ THE ADDRESS, AND IT IS ARITHMETIC: every slot in a collection is one size, so where the i-th one
 * begins follows from the run's base and nothing is stored per slot. */
static __device__ inline bool nn__expert__type_slot_at(uint64_t type, uint64_t slot, uint64_t* at) {
    if (at == 0) return false;
    const uint64_t base = nn__expert__type_at(type);
    const uint64_t bytes = nn__expert__type_bytes(type);
    if (base == 0ull || bytes == 0ull) return false;
    if (slot >= nn__expert__type_slots(type)) return false;
    *at = base + slot * bytes;                    /* the stand checked this product against the arena */
    return true;
}

/* And backwards. ⛔ A REMAINDER IS A REFUSAL — an address inside the run but not ON a boundary is not a
 * slot, and answering the slot it lands in would turn a caller's arithmetic mistake into silent aliasing. */
static __device__ inline bool nn__expert__type_slot_of(uint64_t type, uint64_t at, uint64_t* slot) {
    if (slot == 0) return false;
    const uint64_t base = nn__expert__type_at(type);
    const uint64_t bytes = nn__expert__type_bytes(type);
    if (base == 0ull || bytes == 0ull || at < base) return false;
    const uint64_t past = at - base;
    if (past % bytes != 0ull) return false;
    const uint64_t which = past / bytes;
    if (which >= nn__expert__type_slots(type)) return false;
    *slot = which;
    return true;
}

/* ── ⭐⭐⭐ THE FREE LIST, AND IT COSTS NO MEMORY AT ALL ──────────────────────────────────────────────
 *
 * ⚖ *"the pages go in freemem"*, and `freemem` is the architect's own word for the mechanism the buffer
 * pool already has — *"its deallocate is not a real deallocate it just reinserts itself in the freemem
 * list for its buffer type."* One list per collection, and it is the ONLY source of slots: ⚖ *"there is
 * only one lru of the expert and one free list of pages"*, so `take` is a pop with no second path and no
 * branch for a slot that was never handed out.
 *
 * ⭐⭐ AND ITS LINKS LIVE IN THE HEAP, ONE WORD PER SLOT. The head is in the type row; every next link
 * is a word in the collection's link pages (`NN__HATCH__EXPERT_LINKS`, below). ⛳ NOT IN THE FREE SLOT'S
 * OWN BYTES, although those are dead: a slot is card memory and the list is walked by the evaluator on
 * the host, which never reads card memory (the claim rule `card_address_never_read_on_host`) — `REASONED`.
 * ⛔ THE SENTINEL IS `slot + 1` AND NOT `slot`, because slot 0 is an ordinary slot — the same sentinel
 * problem the expert LRU's links have, bought the same way rather than a second way.
 *
 * ⛔⛔ `give` TRUSTS ITS CALLER ABOUT ONE THING AND IT IS WORTH NAMING: that the slot is not ALREADY free.
 * Giving one twice links it to itself, and then two experts are handed one slot with nothing anywhere
 * reporting it. ⛳ WHAT MAKES THAT UNREACHABLE IS UPSTREAM, NOT HERE: `give` is only reached from `evict`
 * and `revoke`, and `evict` refuses an expert that is not resident — so a second eviction never arrives.
 * ⇒ ★ THE GUARD IS ONE LEVEL UP, SO THE TEST FOR IT IS ONE LEVEL UP — a check that a second evict refuses
 * is what holds this invariant, and it is in the suite for that reason and not as a courtesy.
 * ⛔ A SCAN WOULD CATCH IT AND IS REFUSED: the list starts at every slot, so a scan is O(slots) on a path
 * that runs per eviction.
 * ⛳ AND THE WRITES ARE PLAIN: block zero is the sole writer while the workers have returned
 * (`expert_lru_node_array.md` §⑤ⓓ), which is the invariant the whole index is written under. */

/* One word of one row, written back. ⛳ Read-modify-write of the whole node, because that is the only door
 * `node_array` has — and the row is six words, so the cost is the same either way. */
static __device__ inline bool nn__expert__zzprivate_type_word_set(uint64_t type, unsigned which,
                                                                  uint64_t value) {
    const uint64_t array = nn__expert__zzprivate_types_array();
    if (array == 0ull) return false;
    if (type >= sys__node_array__length(array)) return false;
    sys__heap_node row = sys__node_array__get(array, type);
    if (row.dtype != SYS__KIND__VALUE_INT) return false;
    row.args[which] = value;
    return sys__node_array__set(array, type, &row);
}

/* How many links one page of a collection's free list holds, and so the most slots a collection may have.
 * A page is a node_array and must fit one chunk, and so must the array of pages — so it is DERIVED from
 * the chunk, never written as a number: half a chunk is 256 links in the shipping geometry (65,536 slots,
 * ~104 GB of q4 experts in one collection) and 128 in the host tests' smaller one. `MEASURED`: a fixed 256
 * refused as `HPGE` on the tests' 256-node chunks. */
#define NN__EXPERT__LINKS_PER_PAGE  ((uint64_t)SYS__HEAP__CHUNK_NODES / 2ull)
#define NN__EXPERT__MAX_SLOTS       (NN__EXPERT__LINKS_PER_PAGE * NN__EXPERT__LINKS_PER_PAGE)

/* One collection's link for one slot — the free list's next pointer, kept in the heap. Borrowed, never
 * got: a `get` would take a hold on every lookup. ▶ `NN__HATCH__EXPERT_LINKS`. */
static __device__ inline uint64_t nn__expert__zzprivate_link_page(uint64_t type, uint64_t slot) {
    const sys__heap_node row = sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_LINKS);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE || type >= sys__node_array__length(row.args[0])) return 0ull;
    const sys__heap_node pages = sys__node_array__borrow(row.args[0], type);
    const uint64_t p = slot / NN__EXPERT__LINKS_PER_PAGE;
    if (pages.dtype != SYS__KIND__OBJECT_REFERENCE || p >= sys__node_array__length(pages.args[0])) return 0ull;
    const sys__heap_node page = sys__node_array__borrow(pages.args[0], p);
    return page.dtype == SYS__KIND__OBJECT_REFERENCE ? page.args[0] : 0ull;
}
static __device__ inline bool nn__expert__zzprivate_link_get(uint64_t type, uint64_t slot, uint64_t* next) {
    const uint64_t page = nn__expert__zzprivate_link_page(type, slot);
    if (page == 0ull) return false;
    const sys__heap_node n = sys__node_array__borrow(page, slot % NN__EXPERT__LINKS_PER_PAGE);
    if (n.dtype != SYS__KIND__VALUE_INT) return false;
    *next = n.args[0];
    return true;
}
static __device__ inline bool nn__expert__zzprivate_link_set(uint64_t type, uint64_t slot, uint64_t next) {
    const uint64_t page = nn__expert__zzprivate_link_page(type, slot);
    if (page == 0ull) return false;
    sys__heap_node n = sys__heap_node__nothing();
    n.dtype = SYS__KIND__VALUE_INT; n.num_args = 0u; n.op_code = 0ull; n.args[0] = next;
    return sys__node_array__set(page, slot % NN__EXPERT__LINKS_PER_PAGE, &n);
}

/* Take the next free slot, answering which one it is and where. False when the collection is full, which
 * is an ordinary answer and not a fault — ⚖ `place` refuses on a full card by design, with no placeholder
 * eviction policy behind it. */
static __device__ inline bool nn__expert__type_take(uint64_t type, uint64_t* slot, uint64_t* at) {
    const uint64_t head = nn__expert__type_free(type);
    if (head == NN__EXPERT__TYPE_FREE_NONE) return false;
    const uint64_t which = head - 1ull;
    uint64_t addr = 0ull;
    if (!nn__expert__type_slot_at(type, which, &addr)) return false;   /* a head outside its own run */
    uint64_t next = 0ull;
    if (!nn__expert__zzprivate_link_get(type, which, &next)) return false;
    if (!nn__expert__zzprivate_type_word_set(type, NN__EXPERT__TYPE_FREE, next)) return false;
    if (slot != 0) *slot = which;
    if (at != 0)   *at = addr;
    return true;
}

/* And hand one back. ⛔ ▶ the block above for the one thing this trusts and who establishes it. */
static __device__ inline bool nn__expert__type_give(uint64_t type, uint64_t slot) {
    uint64_t addr = 0ull;
    if (!nn__expert__type_slot_at(type, slot, &addr)) return false;
    if (!nn__expert__zzprivate_link_set(type, slot, nn__expert__type_free(type))) return false;
    return nn__expert__zzprivate_type_word_set(type, NN__EXPERT__TYPE_FREE, slot + 1ull);
}

/* ⛔ THE THREE KEYS OF ONE TYPE, AND THEY ARE ALL THREE OR NONE — ⚖ *"absent is fine, partial refuses"*,
 * applied to the triple that they are. A type named in the count and missing a key is an edit that went
 * wrong, never a default anybody chose. */
static __device__ inline bool nn__expert__zzprivate_type_row(uint64_t up_bytes,
                                                             uint64_t down_bytes,
                                                             uint64_t slots,
                                                             bool evictable,
                                                             uint64_t arena_at,
                                                             uint64_t arena_bytes,
                                                             uint64_t* walked,
                                                             sys__heap_node* row,
                                                             uint64_t* link_pages) {
    if (up_bytes == 0ull || down_bytes == 0ull || slots == 0ull) return false;

    uint64_t offset = 0ull, tail = 0ull;
    if (!nn__expert__zzprivate_align_up(up_bytes, &offset)) return false;
    if (!nn__expert__zzprivate_align_up(down_bytes, &tail)) return false;
    if (offset > 0xFFFFFFFFFFFFFFFFull - tail) return false;
    const uint64_t bytes = offset + tail;

    /* The run must land inside the arena the host took for it — the same check the buffers' carve makes,
     * and for the same reason: a run that walks past the end hands out addresses belonging to nobody
     * while every count stays healthy. ⇒ ★ AN IMPLICATION IS NOT A CHECK. */
    if (slots > 0xFFFFFFFFFFFFFFFFull / bytes) return false;
    const uint64_t run = slots * bytes;
    if (run > arena_bytes - *walked) return false;         /* *walked <= arena_bytes, held by the caller */

    const uint64_t base = arena_at + *walked;

    /* ⭐⭐ AND THE RUN IS THREADED INTO ITS FREE LIST HERE, in the heap, one link a slot.
     * ⛳ IT IS DONE AT THE CARVE AND NOT THROUGH THE ACCESSORS, because the hatch rows are not written
     * until every type has been built — so they cannot answer yet.
     * ⛔ EVERY SLOT STARTS FREE, so the list begins at its full length and the head is slot 0. The last
     * slot's link is the sentinel, which is what makes `take` terminate rather than walk off the run. */
    if (slots > NN__EXPERT__MAX_SLOTS) return false;
    const uint64_t page_count = (slots + NN__EXPERT__LINKS_PER_PAGE - 1ull) / NN__EXPERT__LINKS_PER_PAGE;
    *link_pages = sys__node_array__create(page_count);
    if (*link_pages == 0ull) return false;
    for (uint64_t p = 0ull; p < page_count; ++p) {
        const uint64_t first = p * NN__EXPERT__LINKS_PER_PAGE;
        const uint64_t count = (slots - first < NN__EXPERT__LINKS_PER_PAGE) ? slots - first : NN__EXPERT__LINKS_PER_PAGE;
        const uint64_t page = sys__node_array__create(count);
        if (page == 0ull) return false;
        sys__node_array_walk lw;
        bool filled = sys__node_array__walk(page, 0ull, &lw);
        for (uint64_t k = 0ull; filled && k < count; ++k, sys__node_array__next(&lw)) {
            const uint64_t i = first + k;
            sys__heap_node link = sys__heap_node__nothing();
            link.dtype = SYS__KIND__VALUE_INT; link.num_args = 0u; link.op_code = 0ull;
            link.args[0] = (i + 1ull < slots) ? (i + 2ull) : NN__EXPERT__TYPE_FREE_NONE;
            filled = sys__node_array__walk_set(&lw, &link);
        }
        const sys__heap_node held = sys__heap_object__reference_to(page);
        filled = filled && sys__node_array__set(*link_pages, p, &held);
        (void)sys__heap_object__release(page);   /* the page table holds it now, or nothing does */
        if (!filled) return false;
    }

    row->dtype = SYS__KIND__VALUE_INT; row->num_args = 0u; row->op_code = 0ull;
    row->args[NN__EXPERT__TYPE_AT]     = base;
    row->args[NN__EXPERT__TYPE_SLOTS]  = slots;
    row->args[NN__EXPERT__TYPE_BYTES]  = bytes;
    row->args[NN__EXPERT__TYPE_OFFSET] = offset;
    row->args[NN__EXPERT__TYPE_FREE]   = 1ull;             /* slot 0, stored as `slot + 1` */
    row->args[NN__EXPERT__TYPE_EVICTABLE] = evictable ? 1ull : 0ull;   /* ⛳ inverted on purpose — the header */
    *walked += run;
    return true;
}

static __device__ inline bool nn__expert__zzpackage_stand_types(void) {
    uint64_t types = 0ull;
    if (!sys__settings__count((const uint8_t*)NN__EXPERT__KEY_TYPES,
                              sizeof(NN__EXPERT__KEY_TYPES) - 1ull, &types)) {
        return true;                 /* nobody asked: no collections, and that stands. The no-config boot */
    }
    if (types == 0ull || types > (uint64_t)NN__EXPERT__MAX_TYPES) return false;

    const uint64_t arena_at = nn__expert__arena_at();
    const uint64_t arena_bytes = nn__expert__arena_bytes();
    if (arena_at == 0ull || arena_bytes == 0ull) return false;   /* collections with nowhere to put them */

    const uint64_t array = sys__node_array__create(types);
    if (array == 0ull) return false;
    const uint64_t all_links = sys__node_array__create(types);
    if (all_links == 0ull) { (void)sys__heap_object__release(array); return false; }

    bool ok = true;
    uint64_t walked = 0ull;

    /* ⛳ THE SLOTS ARE A GENERATED LIST AND NOT A LOOP, because a settings key is looked up by its BYTES
     * and this tier builds no text — so `type3__up_bytes` has to be a string the compiler wrote. */
#define NN__EXPERT__ZZPRIVATE_TYPE_ROW(n)                                                               \
    if (ok && (uint64_t)(n) < types) {                                                                  \
        uint64_t up = 0ull, down = 0ull, slots = 0ull;                                                  \
        if (!sys__settings__size((const uint8_t*)NN__EXPERT__KEY_UP(n),                                 \
                                 sizeof(NN__EXPERT__KEY_UP(n)) - 1ull, &up)                             \
            || !sys__settings__size((const uint8_t*)NN__EXPERT__KEY_DOWN(n),                            \
                                    sizeof(NN__EXPERT__KEY_DOWN(n)) - 1ull, &down)                      \
            || !sys__settings__count((const uint8_t*)NN__EXPERT__KEY_SLOTS(n),                          \
                                     sizeof(NN__EXPERT__KEY_SLOTS(n)) - 1ull, &slots)) {                \
            ok = false;                                                                                 \
        } else {                                                                                        \
            /* ⛳ THE ONE KEY WHOSE ABSENCE IS NOT A REFUSAL AND NOT A ZERO. A reader that answers false  \
             * cannot say whether the key was missing or malformed, so both land on ANCHORED — which is  \
             * the failure this tier survives. ▶ the header's polarity note. */                          \
            uint64_t anchored = 1ull;                                                                   \
            (void)sys__settings__count((const uint8_t*)NN__EXPERT__KEY_ANCHORED(n),                     \
                                       sizeof(NN__EXPERT__KEY_ANCHORED(n)) - 1ull, &anchored);          \
            sys__heap_node one;                                                                         \
            uint64_t pages = 0ull;                                                                      \
            ok = nn__expert__zzprivate_type_row(up, down, slots, anchored == 0ull,                      \
                                                arena_at, arena_bytes, &walked, &one, &pages)           \
                 && sys__node_array__set(array, (uint64_t)(n), &one);                                   \
            if (pages != 0ull) {                                                                        \
                const sys__heap_node held = sys__heap_object__reference_to(pages);                      \
                ok = ok && sys__node_array__set(all_links, (uint64_t)(n), &held);                       \
                (void)sys__heap_object__release(pages);   /* the table holds it now, or nothing does */ \
            }                                                                                           \
        }                                                                                               \
    }
    NN__EXPERT_TYPE_SLOT_LIST(NN__EXPERT__ZZPRIVATE_TYPE_ROW)
#undef NN__EXPERT__ZZPRIVATE_TYPE_ROW

    if (ok) {
        sys__heap_node row;
        row.dtype = SYS__KIND__OBJECT_REFERENCE; row.num_args = 0u; row.op_code = 0ull;
        row.args[0] = array;
        ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_TYPES, &row);
        row.args[0] = all_links;
        ok = ok && sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_LINKS, &row);
    }
    /* The row took its own hold. This one is the stand's and goes back either way — on the failing path
     * that is what takes a half-built table apart. */
    (void)sys__heap_object__release(array);
    (void)sys__heap_object__release(all_links);
    return ok;
}

/* ══ ⭐⭐ THE EXPERT INDEX — THE TWO ARRAYS, AND THE THREE THINGS THAT HAPPEN TO AN EXPERT ════════════
 * ▶ `expert__header.cuh` for what each word means and which ruling put it there. What is here is the
 * mechanism, and it is small because the design's whole work is done by the indexing.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

static __device__ inline uint64_t nn__expert__zzprivate_index(void) {
    const sys__heap_node row = sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__EXPERTS);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return row.args[0];
}

/* Whose index this is — the worker's own, one a worker — for the loader, which several workers of one program share. */
static __device__ inline uint64_t nn__expert__zzpackage_owner(void) { return nn__expert__zzprivate_index(); }

static __device__ inline uint64_t nn__expert__zzprivate_lru(void) {
    const sys__heap_node row = sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_LRU);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return row.args[0];
}

static __device__ inline uint64_t nn__expert__index_layers(void) {
    const uint64_t index = nn__expert__zzprivate_index();
    if (index == 0ull) return 0ull;
    return sys__node_array__length(index);
}

/* ⭐ BOTH ARRAYS ARE MADE TOGETHER OR NEITHER IS, because a layer's LRU ends and a layer's experts are
 * one fact in two places and a half-built pair would answer one of them.
 *
 * ⚖⚖ THREE LEVELS RULED BY THE ARCHITECT: *"my instinct would have been
 * [layer][type][expert] … i would maintain the layer addressing by index, keeping some null array
 * values is a worthy trade for direct addressing."* ⇒ `index[layer][type][expert]`.
 * ⛳ THE MIDDLE LEVEL IS WHAT ENDS THE COLLISION `NN-5` WAS ABOUT: an expert of layer L is
 * `[L][t][e]` and a dense tensor of layer L is `[L][t'][0]`, and they cannot overwrite each other
 * because the TYPE separates them before the expert index is ever read.
 * ⛳ THE MIDDLE ARRAYS ARE STOOD EAGERLY AND THE INNER ONES ON DEMAND. A middle array is
 * `MAX_TYPES` nodes, so standing all of them makes every `[layer][type]` addressable without asking
 * whether it exists. ⚖ *"keeping some null array values is a worthy trade for direct addressing."*
 * ⛔ NO FIGURE FOR THE COST HERE, AND THE CAP IS NOT SPELLED OUT IN PROSE. **THE COST IS
 * `layers x 2 x MAX_TYPES` NODES** — a band and an ends row apiece — at 64 B a node. Derive it from the
 * constant, which is the one thing that cannot drift away from what the code does.
 * ⇒ ★ A TRADE IS ONLY WORTH ACCEPTING AGAINST A PRICE, so a stale price is worse than none: it argues
 * for the trade using evidence nobody re-checked. ▶ `nn_retired_symbols.md`, `NN__EXPERT__MAX_TYPES`.
 *
 * ⭐⭐ AND THE LRU MOVED DOWN WITH IT — ONE CHAIN PER (LAYER, TYPE), NOT PER LAYER. That is not
 * symmetry for its own sake, it is a CORRECTNESS argument: the LRU exists to choose a victim whose
 * SLOT can then be reused, and a slot only ever serves its own collection. A chain mixing two
 * collections would hand back a victim whose slot is the wrong size for the expert that wants it.
 * ⇒ ★ AN EVICTION POLICY IS ONLY MEANINGFUL OVER A SET OF INTERCHANGEABLE THINGS.
 * ⛳ AND IT KEEPS THE LINKS CHEAP: a link stays a bare expert index, where a per-layer chain would
 * have had to carry `(type, expert)` in one word. */
static __device__ inline bool nn__expert__zzpackage_stand_index(uint64_t layers) {
    if (layers == 0ull) return true;          /* no layer table: no experts, and that stands */

    const uint64_t index = sys__node_array__create(layers);
    if (index == 0ull) return false;
    const uint64_t lru = sys__node_array__create(layers);
    if (lru == 0ull) { (void)sys__heap_object__release(index); return false; }

    bool ok = true;
    for (uint64_t l = 0ull; ok && l < layers; ++l) {
        /* The layer's TYPE bands. Index entries stay null — "this layer holds no experts of that
         * type" needs no writing down — while the LRU's do need writing: a row of zeros IS the
         * empty chain, and zero is what `LINK` reserves. */
        const uint64_t band = sys__node_array__create((uint64_t)NN__EXPERT__MAX_TYPES);
        if (band == 0ull) { ok = false; break; }
        const uint64_t ends = sys__node_array__create((uint64_t)NN__EXPERT__MAX_TYPES);
        if (ends == 0ull) { (void)sys__heap_object__release(band); ok = false; break; }

        sys__node_array_walk ew;
        ok = sys__node_array__walk(ends, 0ull, &ew);
        for (uint64_t t = 0ull; ok && t < (uint64_t)NN__EXPERT__MAX_TYPES; ++t, sys__node_array__next(&ew)) {
            sys__heap_node row;
            row.dtype = SYS__KIND__VALUE_INT; row.num_args = 0u; row.op_code = 0ull;
            row.args[0] = 0ull; row.args[1] = 0ull; row.args[2] = 0ull;
            row.args[3] = 0ull; row.args[4] = 0ull; row.args[5] = 0ull;
            ok = sys__node_array__walk_set(&ew, &row);
        }
        if (ok) {
            sys__heap_node held;
            held.dtype = SYS__KIND__OBJECT_REFERENCE; held.num_args = 0u; held.op_code = 0ull;
            held.args[0] = band;
            ok = sys__node_array__set(index, l, &held);
        }
        if (ok) {
            sys__heap_node held;
            held.dtype = SYS__KIND__OBJECT_REFERENCE; held.num_args = 0u; held.op_code = 0ull;
            held.args[0] = ends;
            ok = sys__node_array__set(lru, l, &held);
        }
        /* The arrays took their own holds; these two are this loop's. */
        (void)sys__heap_object__release(band);
        (void)sys__heap_object__release(ends);
    }

    sys__heap_node held;
    if (ok) {
        held.dtype = SYS__KIND__OBJECT_REFERENCE; held.num_args = 0u; held.op_code = 0ull;
        held.args[0] = index;
        ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__EXPERTS, &held);
    }
    if (ok) {
        held.dtype = SYS__KIND__OBJECT_REFERENCE; held.num_args = 0u; held.op_code = 0ull;
        held.args[0] = lru;
        ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_LRU, &held);
    }

    /* The rows took their own holds; these two are the carve's and go back either way. */
    (void)sys__heap_object__release(index);
    (void)sys__heap_object__release(lru);
    return ok;
}

/* ⭐ THE TWO MIDDLE LOOKUPS, and every accessor below goes through one of them. A layer that is not
 * in this rank's table answers zero QUIETLY — ▶ the cartridge's note on why that is not a refusal. */
static __device__ inline uint64_t nn__expert__zzprivate_band(uint64_t layer) {
    const uint64_t index = nn__expert__zzprivate_index();
    if (index == 0ull) return 0ull;
    if (layer >= sys__node_array__length(index)) return 0ull;
    const sys__heap_node row = sys__node_array__borrow(index, layer);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return row.args[0];
}

static __device__ inline uint64_t nn__expert__zzprivate_ends(uint64_t layer) {
    const uint64_t lru = nn__expert__zzprivate_lru();
    if (lru == 0ull) return 0ull;
    if (layer >= sys__node_array__length(lru)) return 0ull;
    const sys__heap_node row = sys__node_array__borrow(lru, layer);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return row.args[0];
}

/* One (layer, type) band's expert array — the handle, or zero when nothing was stood there. */
static __device__ inline uint64_t nn__expert__zzprivate_inner(uint64_t layer, uint64_t type) {
    const uint64_t band = nn__expert__zzprivate_band(layer);
    if (band == 0ull) return 0ull;
    if (type >= sys__node_array__length(band)) return 0ull;
    const sys__heap_node row = sys__node_array__borrow(band, type);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return row.args[0];
}

static __device__ inline uint64_t nn__expert__layer_experts(uint64_t layer, uint64_t type) {
    const uint64_t inner = nn__expert__zzprivate_inner(layer, type);
    if (inner == 0ull) return 0ull;
    return sys__node_array__length(inner);
}

static __device__ inline bool nn__expert__layer_stand(uint64_t layer, uint64_t type, uint64_t experts) {
    const uint64_t band = nn__expert__zzprivate_band(layer);
    if (band == 0ull) return false;
    if (type >= sys__node_array__length(band)) return false;
    /* ⛔⛔ THE BOUND, AND `MEASURED` BY DELETING IT — IT IS LOAD-BEARING FOR A REASON I HAD
     * NOT WRITTEN DOWN. Without it the over-wide ask still fails, because the allocator cannot fit
     * E+1 nodes in a chunk — but it fails by RAISING, and the raise outlives the call. The suite caught
     * it two rows later, on a check about something else entirely reading a fault count of 1 where it
     * expected 0. ⇒ ★★ THE CHECK IS NOT WHAT MAKES THE REFUSAL HAPPEN; IT IS WHAT MAKES THE REFUSAL AN
     * ANSWER INSTEAD OF AN EVENT. A caller asking for more than a layer can hold has made an ordinary
     * mistake and gets `false`; leaving it to the allocator turns that into a fault on a shared channel
     * that the next unrelated reader inherits. */
    if (experts == 0ull || experts > (uint64_t)NN__EXPERT__MAX_PER_LAYER) return false;
    /* ⛳ AND THE TYPE MUST BE ONE THIS CARD HAS A COLLECTION FOR. Standing a band for a type with no
     * run behind it would build an index nothing could ever be admitted into — every `assign` would
     * refuse later, at a site with no way to say why. ⇒ ★ REFUSE WHERE THE MISTAKE IS, NOT WHERE IT
     * SHOWS. */
    if (type >= nn__expert__types()) return false;
    if (nn__expert__zzprivate_inner(layer, type) != 0ull) return false;  /* stood once, not twice */

    const uint64_t inner = sys__node_array__create(experts);
    if (inner == 0ull) return false;

    /* ⛳ EVERY SLOT IS WRITTEN, not left at the null the array was filled with — because a slot is asked
     * "where is this expert" on the hot path and `VALUE_INT` with a zero address is the answer "nowhere",
     * whereas a null would make the reader distinguish two shapes for one fact. */
    sys__node_array_walk sw;
    bool ok = sys__node_array__walk(inner, 0ull, &sw);
    for (uint64_t e = 0ull; ok && e < experts; ++e, sys__node_array__next(&sw)) {
        sys__heap_node slot;
        slot.dtype = SYS__KIND__VALUE_INT; slot.num_args = 0u; slot.op_code = 0ull;
        slot.args[NN__EXPERT__SLOT_AT]    = 0ull;   slot.args[NN__EXPERT__SLOT_PREV]  = NN__EXPERT__NONE;
        slot.args[NN__EXPERT__SLOT_NEXT]  = NN__EXPERT__NONE;
        slot.args[NN__EXPERT__SLOT_AT_L2] = 0ull;   slot.args[NN__EXPERT__SLOT_TYPE]  = 0ull;
        slot.args[5]                      = 0ull;   /* ⛳ spare — ▶ the header */
        ok = sys__node_array__walk_set(&sw, &slot);
    }
    if (ok) {
        sys__heap_node held;
        held.dtype = SYS__KIND__OBJECT_REFERENCE; held.num_args = 0u; held.op_code = 0ull;
        held.args[0] = inner;
        ok = sys__node_array__set(band, type, &held);     /* the band takes its own hold */
    }
    (void)sys__heap_object__release(inner);
    return ok;
}

static __device__ inline bool nn__expert__slot(uint64_t layer, uint64_t type, uint64_t expert,
                                               sys__heap_node* answer) {
    if (answer == 0) return false;
    const uint64_t inner = nn__expert__zzprivate_inner(layer, type);
    if (inner == 0ull) return false;
    if (expert >= sys__node_array__length(inner)) return false;
    const sys__heap_node slot = sys__node_array__borrow(inner, expert);
    if (slot.dtype != SYS__KIND__VALUE_INT) return false;
    *answer = slot;
    return true;
}

static __device__ inline bool nn__expert__zzprivate_slot_set(uint64_t layer, uint64_t type,
                                                             uint64_t expert,
                                                             const sys__heap_node* slot) {
    const uint64_t inner = nn__expert__zzprivate_inner(layer, type);
    if (inner == 0ull || slot == 0) return false;
    return sys__node_array__set(inner, expert, slot);
}

static __device__ inline bool nn__expert__zzprivate_lru_row(uint64_t layer, uint64_t type,
                                                            sys__heap_node* answer) {
    const uint64_t ends = nn__expert__zzprivate_ends(layer);
    if (ends == 0ull || answer == 0) return false;
    if (type >= sys__node_array__length(ends)) return false;
    const sys__heap_node row = sys__node_array__borrow(ends, type);
    if (row.dtype != SYS__KIND__VALUE_INT) return false;
    *answer = row;
    return true;
}

static __device__ inline bool nn__expert__zzprivate_lru_set(uint64_t layer, uint64_t type,
                                                            const sys__heap_node* row) {
    const uint64_t ends = nn__expert__zzprivate_ends(layer);
    if (ends == 0ull || row == 0) return false;
    return sys__node_array__set(ends, type, row);
}

static __device__ inline bool nn__expert__recent(uint64_t layer, uint64_t type, uint64_t* expert) {
    sys__heap_node row;
    if (expert == 0 || !nn__expert__zzprivate_lru_row(layer, type, &row)) return false;
    const uint64_t link = row.args[NN__EXPERT__LRU_RECENT];
    if (link == NN__EXPERT__NONE) return false;
    *expert = NN__EXPERT__UNLINK(link);
    return true;
}

static __device__ inline bool nn__expert__oldest(uint64_t layer, uint64_t type, uint64_t* expert) {
    sys__heap_node row;
    if (expert == 0 || !nn__expert__zzprivate_lru_row(layer, type, &row)) return false;
    const uint64_t link = row.args[NN__EXPERT__LRU_OLDEST];
    if (link == NN__EXPERT__NONE) return false;
    *expert = NN__EXPERT__UNLINK(link);
    return true;
}

/* ── ⭐⭐ THE CHAIN, AND MEMBERSHIP IS DERIVED FROM POSITION RATHER THAN STORED ────────────────────────
 * ⛔⛔ AN EXPERT OUT OF THE CHAIN AND AN EXPERT ALONE IN IT LOOK IDENTICAL FROM THE SLOT — both have no
 * previous and no next. The head is what tells them apart. ⇒ ★ SO THERE IS NO "in the chain" FLAG: the
 * question is answered by three words already written, which is the same property the page free chains
 * have and the reason neither needs a bit that could go out of step with the links it describes.
 * ⛳ WITHOUT THIS TEST, UNLINKING AN EXPERT THAT WAS NEVER ADMITTED WOULD CLEAR THE LAYER'S HEAD — the
 * whole chain would vanish and every slot in it would still say it was linked. */
static __device__ inline bool nn__expert__zzprivate_in_chain(const sys__heap_node* me, uint64_t expert,
                                                             const sys__heap_node* ends) {
    return me->args[NN__EXPERT__SLOT_PREV] != NN__EXPERT__NONE
        || me->args[NN__EXPERT__SLOT_NEXT] != NN__EXPERT__NONE
        || ends->args[NN__EXPERT__LRU_RECENT] == NN__EXPERT__LINK(expert);
}

/* Take one out. ⛳ A NO-OP ON AN EXPERT THAT IS NOT IN THE CHAIN, deliberately: `touch` and `evict` both
 * begin here and neither should have to ask first. */
static __device__ inline bool nn__expert__zzprivate_unlink(uint64_t layer, uint64_t type,
                                                           uint64_t expert) {
    sys__heap_node me, ends;
    if (!nn__expert__slot(layer, type, expert, &me))        return false;
    if (!nn__expert__zzprivate_lru_row(layer, type, &ends)) return false;
    if (!nn__expert__zzprivate_in_chain(&me, expert, &ends)) return true;

    const uint64_t prev = me.args[NN__EXPERT__SLOT_PREV];
    const uint64_t next = me.args[NN__EXPERT__SLOT_NEXT];

    if (prev != NN__EXPERT__NONE) {
        sys__heap_node p;
        if (!nn__expert__slot(layer, type, NN__EXPERT__UNLINK(prev), &p)) return false;
        p.args[NN__EXPERT__SLOT_NEXT] = next;
        if (!nn__expert__zzprivate_slot_set(layer, type, NN__EXPERT__UNLINK(prev), &p)) return false;
    } else {
        ends.args[NN__EXPERT__LRU_RECENT] = next;
    }
    if (next != NN__EXPERT__NONE) {
        sys__heap_node n;
        if (!nn__expert__slot(layer, type, NN__EXPERT__UNLINK(next), &n)) return false;
        n.args[NN__EXPERT__SLOT_PREV] = prev;
        if (!nn__expert__zzprivate_slot_set(layer, type, NN__EXPERT__UNLINK(next), &n)) return false;
    } else {
        ends.args[NN__EXPERT__LRU_OLDEST] = prev;
    }

    me.args[NN__EXPERT__SLOT_PREV] = NN__EXPERT__NONE;
    me.args[NN__EXPERT__SLOT_NEXT] = NN__EXPERT__NONE;
    if (ends.args[NN__EXPERT__LRU_COUNT] != 0ull) --ends.args[NN__EXPERT__LRU_COUNT];
    return nn__expert__zzprivate_slot_set(layer, type, expert, &me)
        && nn__expert__zzprivate_lru_set(layer, type, &ends);
}

/* Put one at the front. ⛔ THE CALLER MUST HAVE UNLINKED IT FIRST — which is why this is private and why
 * both public callers do exactly that. Re-reads the slot and the ends rather than taking them as
 * arguments, because the unlink it follows has just written both. */
static __device__ inline bool nn__expert__zzprivate_push_front(uint64_t layer, uint64_t type,
                                                               uint64_t expert) {
    sys__heap_node me, ends;
    if (!nn__expert__slot(layer, type, expert, &me))        return false;
    if (!nn__expert__zzprivate_lru_row(layer, type, &ends)) return false;

    const uint64_t was = ends.args[NN__EXPERT__LRU_RECENT];
    me.args[NN__EXPERT__SLOT_PREV] = NN__EXPERT__NONE;
    me.args[NN__EXPERT__SLOT_NEXT] = was;
    if (was != NN__EXPERT__NONE) {
        sys__heap_node r;
        if (!nn__expert__slot(layer, type, NN__EXPERT__UNLINK(was), &r)) return false;
        r.args[NN__EXPERT__SLOT_PREV] = NN__EXPERT__LINK(expert);
        if (!nn__expert__zzprivate_slot_set(layer, type, NN__EXPERT__UNLINK(was), &r)) return false;
    } else {
        /* An empty chain gains its OLDEST end here, and only here. */
        ends.args[NN__EXPERT__LRU_OLDEST] = NN__EXPERT__LINK(expert);
    }
    ends.args[NN__EXPERT__LRU_RECENT] = NN__EXPERT__LINK(expert);
    ++ends.args[NN__EXPERT__LRU_COUNT];
    return nn__expert__zzprivate_slot_set(layer, type, expert, &me)
        && nn__expert__zzprivate_lru_set(layer, type, &ends);
}

/* How many of a (layer, type) band's experts are resident: its chain's length. */
static __device__ inline uint64_t nn__expert__layer_resident(uint64_t layer, uint64_t type) {
    sys__heap_node row;
    return nn__expert__zzprivate_lru_row(layer, type, &row) ? row.args[NN__EXPERT__LRU_COUNT] : 0ull;
}

/* ── ADMIT · TOUCH · EVICT ───────────────────────────────────────────────────────────────────────────
 * ⛳ NONE OF THE THREE ALLOCATES. An admission writes three words and splices; an eviction clears one
 * word and unsplices; a touch only moves the expert. That is the property that makes the structure
 * cheap, and it is the one `expert_lru_node_array.md` §⓪ says the whole design rests on. */
static __device__ inline bool nn__expert__admit(uint64_t layer, uint64_t expert, uint64_t at,
                                                uint64_t type) {
    /* ⛔ AN ADDRESS OF ZERO IS WHAT "EVICTED" MEANS, so admitting one would record a resident expert
     * that every reader agrees is absent. Refused where it can still be told apart from a bug.
     * ⛳ AND THE ADDRESS MUST BE A SLOT OF THIS COLLECTION, which is one division: the run's base and its
     * stride are both in the type's row, so an address that is not on a boundary of it is refused. */
    uint64_t which = 0ull;
    if (at == 0ull || !nn__expert__type_slot_of(type, at, &which)) return false;
    /* ⭐⭐ THE BAND IS CHOSEN BY THE TYPE THE *RESERVATION* RECORDED, not by one the caller re-types.
     * `assign` reads it out of the handed row, so `[layer][type][expert]` is reached with a type that
     * came from the address itself. ⇒ ★ WHERE TWO SOURCES KNOW A FACT, INDEX BY THE ONE NOBODY TYPES —
     * a caller cannot file a slot under a collection that does not own it, because it never says which. */
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, expert, &me)) return false;
    /* ⛔⛔ AN EXPERT THAT IS ALREADY RESIDENT IS REFUSED, AND THIS IS THE `evict` DEFECT RUNNING THE
     * OTHER WAY. Overwriting `SLOT_AT` would leave the OLD slot marked taken by an owner no index
     * names — and `assign` drops the handed row either way, so the outstanding set would read ZERO
     * over a page that has lost room.
     * ⇒ ★★ THE LEAK THE HANDED LIST MAKES ENUMERABLE CAN BE CREATED *THROUGH* IT, by the one door that
     * empties it. A structure that makes one leak impossible does not make leaks impossible, and the
     * second hides behind the first one's guarantee.
     * ⛳ REFUSED RATHER THAN MOVED: a move is `evict` then `admit`, which is two decisions, and a caller
     * that meant to move has said so. Found by a cold review, not by this file's own suite. */
    if (me.args[NN__EXPERT__SLOT_AT] != 0ull) return false;
    if (!nn__expert__zzprivate_unlink(layer, type, expert)) return false;
    if (!nn__expert__slot(layer, type, expert, &me)) return false;   /* the unlink rewrote it */
    me.args[NN__EXPERT__SLOT_AT]   = at;
    me.args[NN__EXPERT__SLOT_TYPE] = type;
    if (!nn__expert__zzprivate_slot_set(layer, type, expert, &me)) return false;
    return nn__expert__zzprivate_push_front(layer, type, expert);
}

/* ⛔ TOUCHING AN EXPERT THAT IS NOT RESIDENT IS A REFUSAL, NOT A PROMOTION. The chain orders what is IN
 * VRAM; an absent expert has nothing to be recent about, and letting it in would put a slot with a zero
 * address at the head — which the evictor would then hand out as its best candidate. */
static __device__ inline bool nn__expert__touch(uint64_t layer, uint64_t type, uint64_t expert) {
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, expert, &me)) return false;
    if (me.args[NN__EXPERT__SLOT_AT] == 0ull)        return false;
    return nn__expert__zzprivate_unlink(layer, type, expert)
        && nn__expert__zzprivate_push_front(layer, type, expert);
}

/* ── THE PROMOTION QUALIFIER ───────────────────────────────────────────────────────────────────────── ▶ the header
 * The two call times live in the slot's last word, the newer in its high half and the older in its low: 32 bits of a
 * clock that moves once a layer's visit is some four billion visits. */
#define NN__EXPERT__ZZPRIVATE_CALLS 5u
static __device__ inline uint64_t nn__expert__tick(void) {
    static uint64_t clock = 0ull;
    return __atomic_add_fetch(&clock, 1ull, __ATOMIC_RELAXED) & 0xFFFFFFFFull;
}
static __device__ inline bool nn__expert__zzprivate_older(uint64_t layer, uint64_t type, uint64_t expert, uint64_t* older) {
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, expert, &me)) return false;
    *older = me.args[NN__EXPERT__ZZPRIVATE_CALLS] & 0xFFFFFFFFull;
    return true;
}
static __device__ inline bool nn__expert__called(uint64_t layer, uint64_t type, uint64_t expert, uint64_t now) {
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, expert, &me)) return false;
    const uint64_t newer = me.args[NN__EXPERT__ZZPRIVATE_CALLS] >> 32;
    me.args[NN__EXPERT__ZZPRIVATE_CALLS] = ((now & 0xFFFFFFFFull) << 32) | newer;
    return nn__expert__zzprivate_slot_set(layer, type, expert, &me);
}
static __device__ inline bool nn__expert__qualifies(uint64_t layer, uint64_t type, uint64_t expert) {
    uint64_t score = 0ull, tail = 0ull, tail_score = 0ull;
    /* ⚖ *"when the cache is empty … the first batch all of the choices go into the cache, then once it fills only those
     *   chosen three times"*: while the layer holds less than its share of the collection and a slot is free, a first
     *   call is enough — RAM keeps every expert, so a slot given to one used once costs nothing to take back */
    if (nn__expert__type_free(type) != NN__EXPERT__TYPE_FREE_NONE) {
        const uint64_t layers = nn__expert__index_layers();
        uint64_t banded = 0ull;
        for (uint64_t l = 0ull; l < layers; ++l) banded += nn__expert__layer_experts(l, type) != 0ull ? 1ull : 0ull;
        if (banded != 0ull && nn__expert__layer_resident(layer, type) < nn__expert__type_slots(type) / banded) return true;
    }
    if (!nn__expert__zzprivate_older(layer, type, expert, &score) || score == 0ull) return false;   /* fewer than three calls */
    if (!nn__expert__oldest(layer, type, &tail)) return false;           /* nothing of this band to give its slot up */
    return nn__expert__zzprivate_older(layer, type, tail, &tail_score) && score > tail_score;
}

/* ⚖ *"an evicted expert gets an offset of 0 and no previous/next."* ⛳ THE TYPE SURVIVES
 * (`expert_lru_node_array.md` §⓪): it says which collection this expert belongs to, and the give below
 * reads it to hand the slot back to the right free list. The L2 address survives too — an expert evicted from VRAM is commonly still in the RAM tier, and
 * that is the state `[0]`-and-`[3]`-together exists to be able to say. */
static __device__ inline bool nn__expert__evict(uint64_t layer, uint64_t type, uint64_t expert) {
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, expert, &me)) return false;
    if (me.args[NN__EXPERT__SLOT_AT] == 0ull)        return false;  /* not resident: nothing to do */

    /* ⛔⛔ AND THE SLOT GOES BACK TO ITS COLLECTION, WHICH IT DID NOT AT ALL. `MEASURED`
     * then: this file held ZERO references to the page layer — eviction cleared the expert's address and
     * told the room nothing, so a card's free space only ever fell. The room was not leaked so much as
     * FORGOTTEN, because the thing that recorded it was a COUNT and a count cannot name a slot.
     * ⇒ ★★ THE TWO INDICES WERE EACH CORRECT AND NOT YET ONE MECHANISM, which is the shape of gap a suite
     * over either one alone cannot see. ⛳ The collapse makes it smaller rather than safer: the slot is a
     * division of the address, and the collection is the one word the expert still carries. */
    /* ⛳ THE SLOT'S OWN TYPE IS WHAT FREES THE RUN, not the band it was reached through. The two agree
     * by `admit`'s construction; using the stored one keeps the give correct even if they ever did not,
     * and it is the word that was written beside the address it describes. */
    const uint64_t held = me.args[NN__EXPERT__SLOT_TYPE];
    uint64_t slot = 0ull;
    /* The slot is WORKED OUT first and GIVEN BACK last — ▶ the ordering note below. */
    if (!nn__expert__type_slot_of(held, me.args[NN__EXPERT__SLOT_AT], &slot)) return false;

    if (!nn__expert__zzprivate_unlink(layer, type, expert)) return false;
    if (!nn__expert__slot(layer, type, expert, &me)) return false;
    me.args[NN__EXPERT__SLOT_AT] = 0ull;
    if (!nn__expert__zzprivate_slot_set(layer, type, expert, &me)) return false;

    /* ⛔⛔ THE GIVE COMES LAST, AND THE ORDER IS THE WHOLE OF IT. Every step above can fail, and until the
     * expert's own slot says it lives nowhere the collection must go on believing that slot is spoken
     * for — otherwise a failure between the two leaves a slot on the free list that the expert still
     * names, and the next `place` hands that address to a SECOND owner.
     * ⇒ ★★ THE TWO FAILURE DIRECTIONS ARE NOT EQUALLY BAD, AND THE ORDER IS HOW YOU CHOOSE BETWEEN THEM.
     * Giving last fails towards a slot nobody hands out — a leak. Giving first fails towards two owners
     * of one address — corruption, which nothing can see. ⛳ SAME RULE `reserve` states for itself below.
     * ⚠ `REASONED`, NOT ARMED: reaching the bad path needs an injected failure in `unlink` or `slot_set`,
     * and nothing in this tree can fail those on demand. What an arm CAN show is that the ordering
     * changes nothing on the succeeding path, which the suite does.
     * ⛔ AND THE TYPE IS *NOT* CLEARED, which is what lets a second evict be refused by the `SLOT_AT == 0`
     * test above rather than by a scan of the free list — ▶ `nn__expert__type_give`, which trusts it. */
    return nn__expert__type_give(held, slot);
}

/* ⭐ A SLOT HANDED FROM ONE EXPERT TO ANOTHER, never through the free list — the exclusive tier's RAM side, where the
 * expert leaving for the card gives its slot to the one arriving from it (`nn__expert_tier__commit`). `from` (of layer
 * `lf`) must be resident and `to` (of `lt`) not; afterwards `to` holds the slot and is its band's most recent, `from`
 * holds none. ⛔ THE SAME ORDER AS `evict`'s, FOR THE SAME REASON: `from` lets go first, `to` takes it second, so a
 * failure between them leaves a slot nobody names — a leak — and never one slot named twice. */
static __device__ inline bool nn__expert__hand_over(uint64_t lf, uint64_t from, uint64_t lt, uint64_t to, uint64_t type) {
    sys__heap_node a, b;
    if (!nn__expert__slot(lf, type, from, &a) || a.args[NN__EXPERT__SLOT_AT] == 0ull) return false;
    if (!nn__expert__slot(lt, type, to, &b) || b.args[NN__EXPERT__SLOT_AT] != 0ull) return false;
    const uint64_t at = a.args[NN__EXPERT__SLOT_AT], held = a.args[NN__EXPERT__SLOT_TYPE];
    if (!nn__expert__zzprivate_unlink(lf, type, from)) return false;
    if (!nn__expert__slot(lf, type, from, &a)) return false;
    a.args[NN__EXPERT__SLOT_AT] = 0ull;
    if (!nn__expert__zzprivate_slot_set(lf, type, from, &a)) return false;
    if (!nn__expert__zzprivate_unlink(lt, type, to)) return false;
    if (!nn__expert__slot(lt, type, to, &b)) return false;
    b.args[NN__EXPERT__SLOT_AT] = at;
    b.args[NN__EXPERT__SLOT_TYPE] = held;
    if (!nn__expert__zzprivate_slot_set(lt, type, to, &b)) return false;
    return nn__expert__zzprivate_push_front(lt, type, to);
}

/* ── FETCH — a request and its settling, for a caller that wants the expert now ─────────────────────── */
static __device__ inline bool nn__expert__fetch(sys__silicon_family__id family, uint64_t layer, uint64_t type, uint64_t expert,
                                                const nn__expert__backing* backing, uint64_t* at, bool* missed) {
    if (at == NULL || missed == NULL) return false;
    *missed = false;
    const int got = nn__expert__request(family, layer, type, expert, backing, false, &expert, 1u, at);
    if (got == NN__EXPERT__RESIDENT) return true;
    if (got != NN__EXPERT__ARRIVING || !nn__expert__settle(layer, type)) return false;
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, expert, &me) || me.args[NN__EXPERT__SLOT_AT] == 0ull) return false;
    *at = me.args[NN__EXPERT__SLOT_AT];
    *missed = true;
    return true;
}

/* ══ ⭐⭐ THE ADMITTER — ONE POP ═══════════════════════════════════════════════════════════════════
 * ▶ `expert__header.cuh` for why there is one place to look and no order to it. */
static __device__ inline bool nn__expert__place(uint64_t type, uint64_t* address, uint64_t* slot) {
    if (address == NULL || slot == NULL) return false;
    if (type >= nn__expert__types()) return false;      /* a collection nobody stood */
    /* ⛔ FALSE HERE IS A FULL COLLECTION AND NOT AN ERROR — ▶ the header: no eviction policy stands
     * behind this yet, so the refusal is the honest answer. */
    return nn__expert__type_take(type, slot, address);
}

static __device__ inline uint64_t nn__expert__zzprivate_handed(void) {
    const sys__heap_node row = sys__package__own_row(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_HANDED);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return row.args[0];
}

static __device__ inline uint64_t nn__expert__handed_count(void) {
    const uint64_t list = nn__expert__zzprivate_handed();
    if (list == 0ull) return 0ull;
    return sys__sublist__length(list);
}

static __device__ inline bool nn__expert__handed_append(uint64_t at, uint64_t type) {
    const uint64_t list = nn__expert__zzprivate_handed();
    if (list == 0ull || at == 0ull) return false;
    sys__heap_node entry = sys__heap_node__nothing();
    entry.dtype = SYS__KIND__VALUE_INT; entry.num_args = 0u; entry.op_code = 0ull;
    entry.args[NN__EXPERT__HANDED_AT]   = at;
    entry.args[NN__EXPERT__HANDED_TYPE] = type;
    return sys__sublist__append(list, &entry);
}

/* ⛳ A WALK, AND IT IS THE RIGHT SHAPE HERE RATHER THAN A CONCESSION. The list holds what is IN FLIGHT —
 * a loader publishes each expert immediately after writing it, so the set is a handful deep and usually
 * one. ⛔ A structure with a lookup would be paying for a search that is never long, in the one place
 * where being able to ENUMERATE the outstanding set is the whole reason the list exists. */
static __device__ inline bool nn__expert__handed_find(uint64_t at, uint64_t* type, uint64_t* where) {
    const uint64_t list = nn__expert__zzprivate_handed();
    if (list == 0ull || at == 0ull) return false;
    const uint64_t n = sys__sublist__length(list);
    for (uint64_t i = 0ull; i < n; ++i) {
        const sys__heap_node entry = sys__sublist__nth(list, i);
        if (entry.dtype != SYS__KIND__VALUE_INT) continue;
        if (entry.args[NN__EXPERT__HANDED_AT] != at) continue;
        if (type)  *type  = entry.args[NN__EXPERT__HANDED_TYPE];
        if (where) *where = i;
        return true;
    }
    return false;
}

static __device__ inline bool nn__expert__handed_drop(uint64_t where) {
    const uint64_t list = nn__expert__zzprivate_handed();
    if (list == 0ull || where >= sys__sublist__length(list)) return false;
    const sys__heap_node gone = sys__sublist__extract(list, where);
    return gone.dtype == SYS__KIND__VALUE_INT;
}

/* Stand the list. ⛳ It holds integers, so it never chases a reference on teardown. */
static __device__ inline bool nn__expert__zzpackage_stand_handed(void) {
    const uint64_t list = sys__list__create();
    if (list == 0ull) return false;
    sys__heap_node row;
    row.dtype = SYS__KIND__OBJECT_REFERENCE; row.num_args = 0u; row.op_code = 0ull;
    row.args[0] = list;
    const bool ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__EXPERT_HANDED, &row);
    (void)sys__heap_object__release(list);      /* the row took its own hold */
    return ok;
}

/* ══ ⭐⭐ THE LOAD PROTOCOL — ▶ `expert__header.cuh` for the four-line shape the host drives ═══════════ */

static __device__ __noinline__ void nn__expert__zzpackage_apply_reserve(sys__heap_node* base,
                                                                        uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) != 2ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const sys__heap_node n = sys__sublist__nth(form, 1ull);
    if (n.dtype != SYS__KIND__VALUE_INT) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE); return; }

    /* ⚖ AND THE ARGUMENT IS A TYPE, RULED — *"reserve should take the type."* ▶ the header
     * for why the old bytes-and-derive argument was right and why its premise is gone. */
    uint64_t at = 0ull, slot = 0ull;
    const uint64_t type = n.args[0];
    if (!nn__expert__place(type, &at, &slot)) {
        sys__opcodes__fails(form, NN__EXPERT__FAULT_ROOMLESS);
        return;
    }
    /* ⛔⛔ THE SLOT IS TAKEN BY NOW, SO A FAILURE TO RECORD IT MUST GIVE IT BACK. Answering an error while
     * holding a slot nobody is listed as holding is the exact leak the handed list exists to make
     * impossible — and it would be a leak the list itself could not see, which is worse than the one it
     * replaces. ⇒ ★ THE UNWIND IS THE PART OF A RESERVATION THAT HAS TO BE RIGHT. */
    if (!nn__expert__handed_append(at, type)) {
        (void)nn__expert__type_give(type, slot);
        sys__opcodes__fails(form, NN__EXPERT__FAULT_LEDGER);
        return;
    }

    sys__heap_node answer = sys__heap_node__nothing();
    answer.dtype = SYS__KIND__VALUE_INT; answer.num_args = 0u; answer.op_code = 0ull;
    answer.args[0] = at;
    sys__opcodes__becomes(form, &answer);
}

static __device__ __noinline__ void nn__expert__zzpackage_apply_assign(sys__heap_node* base,
                                                                       uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) != 5ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const sys__heap_node l = sys__sublist__nth(form, 1ull);
    const sys__heap_node ty = sys__sublist__nth(form, 2ull);
    const sys__heap_node e = sys__sublist__nth(form, 3ull);
    const sys__heap_node a = sys__sublist__nth(form, 4ull);
    if (l.dtype != SYS__KIND__VALUE_INT || ty.dtype != SYS__KIND__VALUE_INT
        || e.dtype != SYS__KIND__VALUE_INT
        || a.dtype != SYS__KIND__VALUE_INT) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE); return; }

    /* ⭐ THE PAGE AND THE CLASS COME FROM THE HANDED ROW, NOT FROM THE CALLER — ▶ the header. An address
     * nobody reserved has nowhere for those to come from, which is why it is refused rather than
     * half-admitted at a page the caller guessed. */
    uint64_t type = 0ull, where = 0ull;
    if (!nn__expert__handed_find(a.args[0], &type, &where)) {
        sys__opcodes__fails(form, NN__EXPERT__FAULT_STRAY);
        return;
    }
    /* ⛔⛔ THE TYPE IS GIVEN *AND* KNOWN, AND A DISAGREEMENT IS A REFUSAL. The caller names the band
     * it means; the handed row says which collection the address actually came out of. Admitting on
     * the handed row alone would let `[L][3][e]` quietly file a type-0 slot, and every later reader
     * of that band would be right about a lie. ⇒ ★ A REDUNDANT ARGUMENT IS WORTH ITS KEYSTROKES WHEN
     * IT CAN BE CHECKED — this one turns a caller's mistake into an error instead of an index. */
    if (ty.args[0] != type) { sys__opcodes__fails(form, NN__EXPERT__FAULT_REFUSED); return; }
    if (!nn__expert__admit(l.args[0], e.args[0], a.args[0], type)) {
        /* ⛳ THE ROW STAYS. A refused admission leaves the slot taken and the handle outstanding, which
         * is TRUE and is what the list is for — the caller can revoke it, and until it does the address
         * is still theirs. Dropping the row here would turn a refusal into a leak. */
        sys__opcodes__fails(form, NN__EXPERT__FAULT_REFUSED);
        return;
    }
    if (!nn__expert__handed_drop(where)) { sys__opcodes__fails(form, NN__EXPERT__FAULT_STRAY); return; }

    const sys__heap_node done = sys__heap_node__nothing();
    sys__opcodes__becomes(form, &done);
}

static __device__ __noinline__ void nn__expert__zzpackage_apply_revoke(sys__heap_node* base,
                                                                       uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) != 2ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const sys__heap_node a = sys__sublist__nth(form, 1ull);
    if (a.dtype != SYS__KIND__VALUE_INT) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE); return; }

    uint64_t type = 0ull, where = 0ull, slot = 0ull;
    if (!nn__expert__handed_find(a.args[0], &type, &where)) {
        sys__opcodes__fails(form, NN__EXPERT__FAULT_STRAY);
        return;
    }
    if (!nn__expert__type_slot_of(type, a.args[0], &slot)
        || !nn__expert__type_give(type, slot)
        || !nn__expert__handed_drop(where)) {
        sys__opcodes__fails(form, NN__EXPERT__FAULT_STRAY);
        return;
    }
    const sys__heap_node done = sys__heap_node__nothing();
    sys__opcodes__becomes(form, &done);
}

/* ⭐ HOW MANY HANDLES ARE OUT. ⛳ A loader that published everything it reserved reads ZERO, so this one
 * number is the whole diagnostic — which is what "the leak is a list anyone can read" buys. */
static __device__ __noinline__ void nn__expert__zzpackage_apply_handed(sys__heap_node* base,
                                                                       uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) != 1ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    sys__heap_node answer = sys__heap_node__nothing();
    answer.dtype = SYS__KIND__VALUE_INT; answer.num_args = 0u; answer.op_code = 0ull;
    answer.args[0] = nn__expert__handed_count();
    sys__opcodes__becomes(form, &answer);
}

/* ⭐⭐ THE WORD THAT MAKES A LAYER EXIST — ▶ the header for why the protocol above could not work
 * without it. */
static __device__ __noinline__ void nn__expert__zzpackage_apply_layer(sys__heap_node* base,
                                                                      uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) != 4ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const sys__heap_node l = sys__sublist__nth(form, 1ull);
    const sys__heap_node t = sys__sublist__nth(form, 2ull);
    const sys__heap_node n = sys__sublist__nth(form, 3ull);
    if (l.dtype != SYS__KIND__VALUE_INT || t.dtype != SYS__KIND__VALUE_INT
        || n.dtype != SYS__KIND__VALUE_INT) {
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
        return;
    }
    if (!nn__expert__layer_stand(l.args[0], t.args[0], n.args[0])) {
        sys__opcodes__fails(form, NN__EXPERT__FAULT_INDEX);
        return;
    }
    const sys__heap_node done = sys__heap_node__nothing();
    sys__opcodes__becomes(form, &done);
}

/* ⭐⭐ THE JOIN — ▶ `expert__header.cuh` for why this is a kind and why it is minted from the index. */
static __device__ __noinline__ void nn__expert__zzpackage_apply_plane(sys__heap_node* base,
                                                                      uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) != 6ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const sys__heap_node l = sys__sublist__nth(form, 1ull);
    const sys__heap_node t = sys__sublist__nth(form, 2ull);
    const sys__heap_node x = sys__sublist__nth(form, 3ull);
    const sys__heap_node o = sys__sublist__nth(form, 4ull);
    const sys__heap_node n = sys__sublist__nth(form, 5ull);
    if (l.dtype != SYS__KIND__VALUE_INT || t.dtype != SYS__KIND__VALUE_INT
        || x.dtype != SYS__KIND__VALUE_INT
        || o.dtype != SYS__KIND__VALUE_INT || n.dtype != SYS__KIND__VALUE_INT) {
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
        return;
    }

    sys__heap_node me;
    if (!nn__expert__slot(l.args[0], t.args[0], x.args[0], &me)) {
        sys__opcodes__fails(form, NN__EXPERT__FAULT_ABSENT);
        return;
    }
    /* ⛔ AN ADDRESS OF ZERO IS WHAT "EVICTED" MEANS, so this is the residency test and not a null check.
     * An expert that has been evicted has no bytes to name, and naming them anyway would hand out a
     * pointer into a slot some other expert now owns. */
    const uint64_t at = me.args[NN__EXPERT__SLOT_AT];
    if (at == 0ull) { sys__opcodes__fails(form, NN__EXPERT__FAULT_ABSENT); return; }

    /* ⛔⛔ AND THE SPAN IS CHECKED AGAINST THIS EXPERT'S OWN SLOT, WHICH IS THE WHOLE REASON THIS VERB
     * EXISTS RATHER THAN A DOOR TAKING AN ADDRESS. Both comparisons are against the slot's LENGTH and
     * never against a computed address, so neither can wrap: the same shape the arena carve uses. */
    const uint64_t width = nn__expert__type_bytes(me.args[NN__EXPERT__SLOT_TYPE]);
    if (width == 0ull || n.args[0] == 0ull) { sys__opcodes__fails(form, NN__EXPERT__FAULT_SPAN); return; }
    if (o.args[0] > width || n.args[0] > width - o.args[0]) {
        sys__opcodes__fails(form, NN__EXPERT__FAULT_SPAN);
        return;
    }

    sys__heap_node* made = sys__heap__make(NN__KIND__WEIGHTS);
    if (made == 0) { sys__opcodes__fails(form, NN__EXPERT__FAULT_HEAP); return; }
    /* ⛔⛔ THE BODY IS WRITTEN WHOLE, NOT JUST THE TWO FIELDS THIS KIND USES. `sys__heap__make` writes
     * the HEAD and leaves the body as whatever the chunk held — *"the chunk was not zeroed"* — so
     * setting only `[0]` and `[1]` would ship four words of somebody else's freed object inside a
     * published object. Nothing reads them today, which is exactly why it would survive to the day
     * something does. ⇒ ★ AN OBJECT'S UNUSED FIELDS ARE PART OF ITS VALUE, and `nn__kv_ref` — which
     * this kind's own comment calls *"the same object pointed at a different owner"* — already does
     * this and zeroes its spare. Consistency with the thing you cite is not decoration. */
    made[1] = sys__heap_node__nothing();
    made[1].dtype = SYS__KIND__VALUE_INT; made[1].num_args = 0u; made[1].op_code = 0ull;
    made[1].args[NN__WEIGHTS__AT]    = at + o.args[0];
    made[1].args[NN__WEIGHTS__BYTES] = n.args[0];
    /* ⛳ A PAGE VIEW HOLDS NOTHING, AND THE ZERO IS WRITTEN RATHER THAN LEFT — ▶ `NN__WEIGHTS__HELD`.
     * A page outlives every name for it, so there is nobody to give a hold back to. */
    made[1].args[NN__WEIGHTS__HELD]  = 0ull;
    made[1].args[NN__WEIGHTS__HOST_AT] = 0ull;   /* a page slot is on the card only */
    made[1].args[NN__WEIGHTS__STAMP]   = 0ull;
    made[1].args[NN__WEIGHTS__OWNER]   = (uint64_t)sys__silicon__worker() + 1ull;   /* the arena is this worker's */

    const uint64_t where = sys__heap__offset(made + 1);
    const sys__heap_node ref = sys__heap_object__reference_to(where);
    /* The form takes its own hold and the one `make` left is this verb's, so it goes back — the same two
     * lines every verb in this tree ends with. */
    sys__opcodes__becomes(form, &ref);
    (void)sys__heap_object__release(where);
}

/* ── ⭐⭐ WHAT A VIEW HANDS OVER WHEN ITS LAST HOLD GOES ──────────────────────────────────────────────
 * ⛔⛔ IT LETS GO OF NOTHING ITSELF. Every release hook in this tree MOVES what it held onto the chain
 * the release loop drains, and never calls release from inside a release — which is what keeps a deep
 * chain from recursing on a device with no stack to spare. ▶ `sys__dictionary__zzpackage_release_internal`,
 * the shape this copies.
 * ⛳ `HELD == 0` IS THE PAGE CASE and is the common one, so the early return is the fast path. */
static __device__ __noinline__ void nn__weights__zzpackage_release_internal(sys__heap_node* head,
                                                                           uint64_t* releaser_stack) {
    if (head == 0) return;
    const uint64_t held = head[1].args[NN__WEIGHTS__HELD];
    head[1].args[NN__WEIGHTS__HELD] = 0ull;   /* ⛳ cleared first: a hook must not hand the same hold twice */
    if (held == 0ull) return;
    sys__heap_node reference = sys__heap_object__reference_to(held);
    sys__stack__transfer(releaser_stack, &reference);
}

#endif /* SILVANN__PACKAGES_NN_CPU_EXPERT__IMPL_CUH */
