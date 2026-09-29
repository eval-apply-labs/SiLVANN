#ifndef SILVANN__PACKAGES_SYS_CPU_PACKAGE__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_PACKAGE__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "package__header.cuh"      /* the contract these definitions answer */
#include "heap_node__header.cuh"   /* the node every value in this language is made of */
#include "list__header.cuh"         /* the list is walked and appended to */
#include "node_array__header.cuh"   /* the hatch is one entry per package, reached by index */
#include "system_register__header.cuh" /* which row the hatch is */
#include "../../manifest__header.cuh"   /* the roster, which the row-count switch below is driven by */
#include "opcodes/opcodes.cuh"              /* how a verb answers, and how it refuses */
#include "heap__header.cuh"         /* a form is a heap object, and so is the list it names */
/* ══ THE INIT LIST, WALKED AND REWRITTEN ═════════════════════════════════════════════════════════════
 * The reasoning is beside this, in the file that declares these. What is here is the three operations
 * every package's init is written out of.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


/* ── THE HATCH ───────────────────────────────────────────────────────────────────────────────────────
 * One entry per package, indexed by the package id, behind a single register row; and one SUB-ARRAY per
 * entry, indexed by that package's own row enum. The row is the whole of what the register learns: a
 * package needing device-wide state of its own puts it behind its own entry and the register's list never
 * grows for anybody. The reasoning is in `package__header.cuh`, and the perennial premise both hops are
 * read under is in `contracts/objects/package.cuh`.
 *
 * ⛳ THE TOP ARRAY IS REACHED THE SAME WAY IN ALL THREE FUNCTIONS THAT NEED IT, so it is one line named
 * once. It answers zero for a hatch that is not standing, which every caller then treats as its own kind
 * of nothing — the entry lookup (and so `own_row`) a null value, `open_entry` a refusal, `hatch_stands`
 * a no. */
/* ⭐⭐ AND ONE HATCH PER WORKER — ⚖ *"a per-device hatch row holds an array indexed by worker id"*: the register row
 * names an array of hatches, one a worker, and a package reads the hatch of the worker its runner is. So a
 * package's own rows describe ITS device on every worker without the package knowing there are several. */
static __device__ inline uint64_t sys__package__zzprivate_hatch(void) {
    const sys__heap_node row = sys__system_register__get(SYS__SYSTEM_REGISTER__EXTENSION_HATCH);
    if (row.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    const sys__heap_node mine = sys__node_array__borrow(row.args[0], (uint64_t)sys__silicon__worker());
    if (mine.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return mine.args[0];
}

/* One package's sub-array, or zero. ⛳ BORROWED AND NOT `get`: the top array is written only by
 * `zzengine_open_entry`, on the stand-up's runner zero, before any init runs — so there is nothing to
 * lose a hold
 * of, and a release path that paid an atomic here would pay one per buffer freed. */
static __device__ inline uint64_t sys__package__zzprivate_entry(uint64_t package_id) {
    const uint64_t hatch = sys__package__zzprivate_hatch();
    if (hatch == 0ull) return 0ull;
    const sys__heap_node entry = sys__node_array__borrow(hatch, package_id);
    if (entry.dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    return entry.args[0];
}

static __device__ inline bool sys__package__zzengine_publish_hatch(uint64_t entries) {
    if (entries == 0ull) return false;
    const uint64_t workers = (uint64_t)sys__silicon__worker_count();
    const uint64_t all = sys__node_array__create(workers);
    if (all == 0ull) return false;
    bool ok = true;
    for (uint64_t w = 0ull; ok && w < workers; ++w) {
        const uint64_t array = sys__node_array__create(entries);
        if (array == 0ull) { ok = false; break; }
        const sys__heap_node held = sys__heap_object__reference_to(array);
        ok = sys__node_array__set(all, w, &held);
        (void)sys__heap_object__release(array);   /* the array of hatches holds it now, or nothing does */
    }
    /* ⛳ THE ROW HOLDS A REFERENCE AND THE ARRAY HOLDS THE HATCHES, which is the shape every other row of
     * the register that names a table already has — so nothing here is a new kind of row. */
    ok = ok && sys__system_register__set(SYS__SYSTEM_REGISTER__EXTENSION_HATCH, SYS__KIND__OBJECT_REFERENCE, all);
    if (!ok) (void)sys__heap_object__release(all);
    return ok;
}

/* ⭐⭐ HOW MANY ROWS A PACKAGE DECLARED, ASKED BY ID. Reached through a switch for the same reason the
 * teardown is — the loop that opens the entries runs over ids at runtime and a macro cannot be indexed —
 * and it lives HERE rather than in the engine so that nobody outside ever states a length.
 * ⇒ ★ THE ONLY FIGURE THAT CANNOT BE WRONG IS THE ONE NOBODY WRITES: `open_entry` looks its own count up,
 * so a caller cannot pass a number that disagrees with the list the rows are numbered from. */
#define SYS__PACKAGE__ZZPRIVATE_ROWS_CASE(ID, NAME, ...)  case (ID): return (uint64_t)NAME##__HATCH__ROWS;
static __device__ inline uint64_t sys__package__zzengine_own_rows(uint64_t package_id) {
    switch (package_id) { PACKAGE_LIST(SYS__PACKAGE__ZZPRIVATE_ROWS_CASE) default: break; }
    return 0ull;
}
#undef SYS__PACKAGE__ZZPRIVATE_ROWS_CASE

/* ⛳ ONE SUB-ARRAY, OPENED FOR A PACKAGE THAT HAS SOMETHING TO PUT IN IT. The length is the room's row
 * plus whatever the package declared, so it is never zero — a package wanting NOTHING is not opened at
 * all and keeps the null entry the ruling asked for. ⇒ the hole is an absence of an array, not an array
 * of absences, and `own_row` cannot tell the difference because both answer nothing. */
static __device__ inline bool sys__package__zzengine_open_entry(uint64_t package_id) {
    const uint64_t hatch = sys__package__zzprivate_hatch();
    if (hatch == 0ull) return false;
    const uint64_t rows = (uint64_t)SYS__PACKAGE__HATCH_OWN_FIRST
                        + sys__package__zzengine_own_rows(package_id);
    const uint64_t array = sys__node_array__create(rows);
    if (array == 0ull) return false;
    sys__heap_node entry;
    entry.dtype = SYS__KIND__OBJECT_REFERENCE; entry.num_args = 0u; entry.op_code = 0ull;
    entry.args[0] = array;
    return sys__node_array__set(hatch, package_id, &entry);
}

static __device__ inline bool sys__package__zzengine_hand_room(uint64_t package_id, uint64_t at,
                                                               uint64_t bytes) {
    /* ⛳ A PACKAGE THAT ASKED FOR NOTHING IS LEFT ALONE rather than handed a zero-length slice: an unwritten
     * row already reads as a null value, so "asked for none" and "has none" are ONE state and not two. */
    if (bytes == 0ull) return true;
    const uint64_t entry = sys__package__zzprivate_entry(package_id);
    if (entry == 0ull) return false;
    sys__heap_node room;
    room.dtype = SYS__KIND__VALUE_INT; room.num_args = 0u; room.op_code = 0ull;
    room.args[SYS__PACKAGE__HATCH_AT]    = at;
    room.args[SYS__PACKAGE__HATCH_BYTES] = bytes;
    room.args[SYS__PACKAGE__HATCH_RAM_AT] = 0ull; room.args[SYS__PACKAGE__HATCH_RAM_CARD_AT] = 0ull;
    room.args[SYS__PACKAGE__HATCH_RAM_BYTES] = 0ull;
    return sys__node_array__set(entry, SYS__PACKAGE__HATCH_ROOM, &room);
}

/* And the room's RAM span, into the same row — pinned RAM the card can write, which a package asks for
 * separately and may ask for alone. The same rule: nothing asked, nothing written. */
static __device__ inline bool sys__package__zzengine_hand_ram(uint64_t package_id, uint64_t ram_at,
                                                              uint64_t ram_card_at, uint64_t ram_bytes) {
    if (ram_bytes == 0ull) return true;
    const uint64_t entry = sys__package__zzprivate_entry(package_id);
    if (entry == 0ull) return false;
    sys__heap_node room = sys__node_array__borrow(entry, SYS__PACKAGE__HATCH_ROOM);
    if (room.dtype != SYS__KIND__VALUE_INT) {
        room = sys__heap_node__nothing();
        room.dtype = SYS__KIND__VALUE_INT; room.num_args = 0u; room.op_code = 0ull;
        room.args[SYS__PACKAGE__HATCH_AT] = 0ull; room.args[SYS__PACKAGE__HATCH_BYTES] = 0ull;
    }
    room.args[SYS__PACKAGE__HATCH_RAM_AT]      = ram_at;
    room.args[SYS__PACKAGE__HATCH_RAM_CARD_AT] = ram_card_at;
    room.args[SYS__PACKAGE__HATCH_RAM_BYTES]   = ram_bytes;
    return sys__node_array__set(entry, SYS__PACKAGE__HATCH_ROOM, &room);
}

static __device__ inline sys__heap_node sys__package__own_row(uint64_t package_id, uint64_t row) {
    const uint64_t entry = sys__package__zzprivate_entry(package_id);
    if (entry == 0ull) return sys__heap_node__nothing();
    return sys__node_array__borrow(entry, row);
}

static __device__ inline bool sys__package__own_row_set(uint64_t package_id, uint64_t row,
                                                        const sys__heap_node* value) {
    /* ⛔ THE ROOM IS NOT A PACKAGE'S TO WRITE. Answering false rather than quietly doing nothing, because a
     * caller that meant row 0 has misunderstood which rows are its own and will misunderstand the next one
     * too — and because the only symptom of a silently-dropped write here is a reader, much later, finding
     * a row that nobody can explain. */
    if (row < (uint64_t)SYS__PACKAGE__HATCH_OWN_FIRST) return false;
    const uint64_t entry = sys__package__zzprivate_entry(package_id);
    if (entry == 0ull) return false;
    return sys__node_array__set(entry, row, value);
}

static __device__ inline sys__heap_node sys__package__own_room(uint64_t package_id) {
    return sys__package__own_row(package_id, SYS__PACKAGE__HATCH_ROOM);
}

static __device__ inline bool sys__package__entry_stands(uint64_t package_id) {
    return sys__package__zzprivate_entry(package_id) != 0ull;
}

static __device__ inline bool sys__package__zzengine_fill_hatch(const uint64_t* slices) {
    if (slices == 0) return false;
    for (uint64_t id = 0ull; id < package_count; ++id) {
        const uint64_t* slice = &slices[id * SYS__PACKAGE__ROOM_SLICE];
        const uint64_t bytes = slice[1];
        /* ⚖ THE HOLE, RULED: *"a package with no system registers will have a null as
         * system_register[package_id] so we can still find thing by index with the package enum. the hole
         * buys us direct referenceability."* ⇒ an entry is opened for a package that has SOMETHING to put
         * in one — rows of its own, or room, because the room is row 0 of that same sub-array and has
         * nowhere else to go. Everything else keeps a null, the array stays `package_count` long, and a
         * package's id stays its index with no search anywhere. */
        if (sys__package__zzengine_own_rows(id) == 0ull && bytes == 0ull && slice[4] == 0ull) continue;
        if (!sys__package__zzengine_open_entry(id)) return false;
        if (!sys__package__zzengine_hand_room(id, slice[0], bytes)) return false;
        if (!sys__package__zzengine_hand_ram(id, slice[2], slice[3], slice[4])) return false;
    }
    return true;
}

/* The table, asked about without asking about a row in it. ⛳ It is the same two lines `own_room` opens
 * with, named — which is the whole repair: the test was always written down, it just had no way to be
 * asked on its own. */
static __device__ inline bool sys__package__hatch_stands(void) {
    return sys__package__zzprivate_hatch() != 0ull;
}

static __device__ inline void sys__package__refuses(uint64_t form, uint64_t why) {
    sys__opcodes__fails(form, why);
}

static __device__ inline uint64_t sys__package__list_of(uint64_t form, uint64_t* why) {
    /* ⛳ THE ARITY IS EXACT RATHER THAN A MINIMUM. An init takes the list and nothing else, so a second
     * argument is a caller that has misunderstood what it is calling — and letting it through would mean
     * the extra was silently ignored by every package in the same way, forever. */
    if (sys__sublist__length(form) != 2ull) {
        *why = SYS__PACKAGE__FAULT_ARITY;
        return 0ull;
    }
    /* ⛳ IT ARRIVES QUOTED, because an object reference naming a list is something the scan REDUCES
     * before it applies the form around it — so an unquoted init list would be evaluated as a program
     * instead of handed over as an argument. */
    const sys__heap_node cell = sys__sublist__nth(form, 1ull);
    if (cell.dtype != SYS__KIND__QUOTED_LIST || cell.args[0] == 0ull) {
        *why = SYS__PACKAGE__FAULT_LIST;
        return 0ull;
    }
    return cell.args[0];
}

static __device__ inline bool sys__package__waiting_for(uint64_t list, uint64_t verb) {
    const uint64_t n = sys__sublist__length(list);
    for (uint64_t i = 0ull; i < n; ++i) {
        const sys__heap_node cell = sys__sublist__nth(list, i);
        /* A call is a plain executable cell carrying the verb, which is what the driver put there and
         * what a deferral appends. Nothing else in this list is anything else. */
        if (cell.dtype == SYS__KIND__STANDARD && cell.op_code == verb) return true;
    }
    return false;
}

static __device__ inline bool sys__package__come_back_later(uint64_t list, uint64_t verb) {
    /* ⛳ A COPY OF THE CALL, NOT A MOVE OF IT. The head is the caller and the driver drops it whatever
     * happens, so putting a fresh cell at the tail and letting the head go is one append rather than a
     * removal and an insertion — and it never leaves the list momentarily without the caller in it. */
    sys__heap_node again;
    again.dtype = SYS__KIND__STANDARD; again.num_args = 0u; again.op_code = verb; again.args[0] = 0ull;
    return sys__sublist__append(list, &again);
}

/* ── sys's HOST INIT: IT TAKES NOTHING, AND SAYS SO ──────────────────────────────────────────────────
 * ⛔ `sys` IS ZERO AND WILL STAY THERE — its memory IS the heap, and the boot takes that already. A
 * second span for the language would be a second allocator nobody asked for.
 * ⛳ IT EXISTS ANYWAY, for the reason every uniform list has one: the loop pastes the name, so the
 * roster names this whether or not the package wants room. Answering "nothing" is an answer, and it is
 * a different one from having no host init at all — which would be a build error naming this package. */
static inline bool sys__package__host_init(const char* config, uint64_t length, sys__package_room* room) {
    (void)config; (void)length;
    if (room == 0) return false;
    room->at = 0; room->bytes = 0ull;
    room->ram_at = 0; room->ram_card_at = 0; room->ram_bytes = 0ull;
    return true;
}

/* And the other half, which has nothing to give back and is written anyway — a teardown that exists only
 * when its init took something is a teardown the loop has to ask about before calling. */
static inline void sys__package__host_teardown(sys__package_room* room) {
    if (room == 0) return;
    room->at = 0; room->bytes = 0ull;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_PACKAGE__IMPL_CUH */
