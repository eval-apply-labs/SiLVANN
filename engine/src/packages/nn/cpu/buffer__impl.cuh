#ifndef SILVANN__PACKAGES_NN_CPU_BUFFER__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_BUFFER__IMPL_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "buffer__header.cuh"           /* the contract these definitions answer */
#include "package__header.cuh"          /* this package's id, and its hatch rows */
#include "../../sys/cpu/package__header.cuh"   /* where a package keeps its own device-wide state */
#include "../../sys/cpu/node_array__header.cuh" /* the two tables, both indexed by the class id */
#include "../../sys/cpu/list__header.cuh"      /* a free list is an ordinary list, with the ordinary atomics */
#include "../../sys/cpu/heap__header.cuh"      /* where a buffer_reference node is carved from */
#include "../../sys/cpu/heap_object__header.cuh" /* holds taken and given back */
#include "../../sys/cpu/settings.cuh"          /* the sizes, read on the card this time */
#include "../contracts/objects/result.cuh"        /* the refusal the evaluator's read gives */
#include "../../sys/cpu/opcodes/opcodes.cuh"           /* how a verb answers, and how it refuses */
#include "../../sys/cpu/opcodes/verb_abi.cuh"          /* the worker's context: which family its card is on */
#include "../../sys/cpu/silicon/silicon_family.cuh"          /* reaching that card */
/* ══ THE ACTIVATION BUFFER POOL ══════════════════════════════════════════════════════════════════════
 *
 * ⚖⚖ THE DESIGN IS THE ARCHITECT'S, and it is the SECOND one — the first was a class table of
 * masks in the heap, and what replaced it is smaller in every dimension:
 * *"i thought that the free mask were going to be a reference to a heap list, and the pinned mask was
 * simply objects that are in the heap and whatever is not in the free list ... its deallocate is not a
 * real deallocate it just reinserts itself in the freemem list for its buffer type."*
 *
 * ⭐⭐ WHAT THAT REMOVED, AND EACH ONE WAS A MECHANISM THIS FILE NOW DOES NOT CONTAIN:
 * ```
 *   THE PINNED MASK   a suspended computation simply keeps holding its buffers. The reference count was
 *                     always the answer — and the pool's own spec had already written the argument
 *                     against a side table: "a side table asked whether anyone holds a thing is a
 *                     reference count that is not one."
 *   THE GENERATION    release fires at zero holders, so no stale reference can exist to be detected.
 *   THE FREE MASK     a list, with the object lock every push and pop in this tree already takes.
 *   THE SLOT COUNT    "available?" is whether the list is empty. Nothing stores a number.
 *   THE RUN OFFSET    a buffer carries its own address. Nothing recomputes one from a base.
 *   THE 64-SLOT CAP   that was a one-word mask's limit wearing a design's clothes. A list has none.
 * ```
 * ⇒ ★ THE SIMPLER DESIGN IS NOT THE ONE WITH FEWER FEATURES, IT IS THE ONE WHOSE INVARIANTS ARE HELD BY
 * A STRUCTURE INSTEAD OF BY A RULE. The first design's §④ read *"a slot is EITHER free in the mask, OR
 * named by exactly one live buffer object"* and needed a walk to check it. Here the buffer IS the thing
 * in the list, so being in two places at once is not something to prevent — it is unsayable.
 *
 * ── WHERE IT ALL LIVES ──────────────────────────────────────────────────────────────────────────────
 * ⚖ *"the free lists are indexed via the system register as
 * system_register[hatch][nn_package_id][buffers][buffer_type]"*, and the node array sits on the buffer
 * types *"because buffer types are static"*.
 * ```
 *   hatch[nn][BUFFERS]        a node_array, one entry per class, each a reference to that class's list
 *   hatch[nn][BUFFER_SIZEOF]  a node_array, one entry per class, each a heap int: bytes per buffer
 * ```
 * ⛳ BOTH ARE COUNTED REFERENCES IN A HATCH ROW rather than words in a device static, which is what makes
 * the whole structure reachable from a release with no context AND owned by something that can let go of
 * it. A raw static holding a heap offset is a reference nothing retains.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


static __device__ inline uint64_t nn__buffer__zzprivate_row(uint64_t row, uint64_t kind) {
    if (kind >= (uint64_t)NN__BUFFER__ALL_CLASSES) return 0ull;
    const sys__heap_node table = sys__package__own_row(nn_pkg_id, row);
    if (table.dtype != SYS__KIND__OBJECT_REFERENCE || table.args[0] == 0ull) return 0ull;
    /* ⛳ BORROWED, NOT `get`. These two arrays are written once by the carve and never again, and the
     * hatch holds them for as long as the machine is up — so there is nothing for a hold to protect
     * against, and a hold here would be an atomic on every allocation and every release. ▶ the perennial
     * premise in `sys/cpu/package__header.cuh`, which this is a tenant of. */
    const sys__heap_node entry = sys__node_array__borrow(table.args[0], kind);
    if (entry.dtype == SYS__KIND__VALUE_NULL) return 0ull;
    return entry.args[0];
}

static __device__ inline uint64_t nn__buffer__zzpackage_freelist(uint64_t kind) {
    return nn__buffer__zzprivate_row((uint64_t)NN__HATCH__BUFFERS, kind);
}

static __device__ inline uint64_t nn__buffer__sizeof(uint64_t kind) {
    return nn__buffer__zzprivate_row((uint64_t)NN__HATCH__BUFFER_SIZEOF, kind);
}

/* ── THE CARVE ───────────────────────────────────────────────────────────────────────────────────────
 *
 * ⭐⭐ IT READS THE SAME KEYS THE HOST SUMMED, AND THAT IS THE POINT RATHER THAN A CONVENIENCE. The host
 * read `nn__buffer_<class>__size_bytes` and `__qty` to decide how big an allocation to take; this reads
 * them again, on the card, to decide how to cut it up. One text, two readers, no total written down
 * anywhere — which is why nn's span keys had to move to `device_params`, the only section that crosses.
 * ⇒ ⛳ SO A DISAGREEMENT BETWEEN THE TWO READERS SHOWS UP HERE AS A REFUSAL RATHER THAN AS A CARVE THAT
 * OVERRUNS ITS SPAN. The host already refused a config it could not read whole; a key this side cannot
 * find means the two parsers do not agree about the same bytes, and that is a fault and not a default.
 * ⛳ AND IT READS THEM THROUGH THE SAME DOORS — `size` here against `host_size` there, `count` against
 * `host_count` — which is newer than this carve is. Until the card had that pair it read a raw figure,
 * correct here only because `__size_bytes` shifts by zero and `__qty` names no unit. ▶ `settings.cuh`.
 *
 * ⛔ THE ORDER IS THE LIST'S ORDER, which is the id order, which is the layout order. `buffer__header.cuh`
 * generates a static_assert that those three are the same thing, because nothing at runtime could tell. */
static __device__ inline bool nn__buffer__zzpackage_carve(uint64_t span_at, uint64_t span_bytes,
                                                          uint64_t ram_at, uint64_t ram_card_at,
                                                          uint64_t ram_bytes, uint64_t* ended) {
    if (span_at == 0ull || ended == 0) return false;

    const uint64_t lists = sys__node_array__create((uint64_t)NN__BUFFER__ALL_CLASSES);
    if (lists == 0ull) return false;
    const uint64_t sizes = sys__node_array__create((uint64_t)NN__BUFFER__ALL_CLASSES);
    if (sizes == 0ull) { (void)sys__heap_object__release(lists); return false; }

    bool ok = true;
    /* ⛳ ONE ROW, TWO PASSES: the card's classes out of the span, then the RAM classes out of the RAM span.
     * Each pass names its keys, where its run starts and how much room it has, and where the program sees
     * its bytes — nowhere, for the card's. */
    uint64_t at = span_at;                       /* the running total, in list order */
    uint64_t run_base = span_at, run_room = span_bytes, host_base = 0ull;
#define NN__BUFFER__ZZPRIVATE_KEY_SIZE(name)  NN__BUFFER__KEY_SIZE(name)
#define NN__BUFFER__ZZPRIVATE_KEY_QTY(name)   NN__BUFFER__KEY_QTY(name)

#define NN__BUFFER__ZZPRIVATE_CARVE_ROW(id, name, konst)                                                \
    if (ok) {                                                                                           \
        uint64_t bytes = 0ull, qty = 0ull;                                                              \
        if (!sys__settings__size((const uint8_t*)NN__BUFFER__ZZPRIVATE_KEY_SIZE(name),                  \
                                 sizeof(NN__BUFFER__ZZPRIVATE_KEY_SIZE(name)) - 1ull, &bytes)           \
            || !sys__settings__count((const uint8_t*)NN__BUFFER__ZZPRIVATE_KEY_QTY(name),               \
                                     sizeof(NN__BUFFER__ZZPRIVATE_KEY_QTY(name)) - 1ull, &qty)) {       \
            ok = false;                                                                                 \
        } else {                                                                                        \
            /* The run must land inside the room the host took for it. A carve that walks past the end  \
             * hands out addresses that belong to nobody, and every count stays healthy. */             \
            if (bytes != 0ull && qty > (0xFFFFFFFFFFFFFFFFull - (at - run_base)) / bytes) ok = false;    \
            if (ok && (at - run_base) + bytes * qty > run_room) ok = false;                              \
        }                                                                                               \
        if (ok) {                                                                                       \
            sys__heap_node one;                                                                         \
            one.dtype = SYS__KIND__VALUE_INT; one.num_args = 0u; one.op_code = 0ull;                     \
            one.args[0] = bytes;                                                                         \
            ok = sys__node_array__set(sizes, (uint64_t)(konst), &one);                                   \
        }                                                                                               \
        if (ok) {                                                                                       \
            const uint64_t list = sys__list__create();                                                   \
            if (list == 0ull) { ok = false; }                                                             \
            else {                                                                                       \
                for (uint64_t i = 0ull; ok && i < qty; ++i) {                                            \
                    sys__heap_node* made = sys__heap__make(NN__KIND__BUFFER);                  \
                    if (made == 0) { ok = false; break; }                                                \
                    made[1].args[NN__BUFFER__AT]    = at + i * bytes;                                    \
                    made[1].args[NN__BUFFER__CLASS] = (uint64_t)(konst);                                 \
                    made[1].args[NN__BUFFER__HOST_AT] = host_base == 0ull ? 0ull                         \
                                                      : host_base + (at + i * bytes - run_base);          \
                    made[1].args[NN__BUFFER__STAMP]   = 0ull;                                              \
                    made[1].args[NN__BUFFER__OWNER]   = (uint64_t)sys__silicon__worker() + 1ull;             \
                    const uint64_t where = sys__heap__offset(made + 1);                                  \
                    const sys__heap_node ref = sys__heap_object__reference_to(where);                    \
                    /* The list takes its own hold; the one `make` left is this loop's and goes back, so \
                     * the list ends up the sole holder and a `getnew` hands that hold straight on. */   \
                    ok = sys__sublist__append(list, &ref);                                               \
                    (void)sys__heap_object__release(where);                                              \
                }                                                                                        \
                sys__heap_node held;                                                                      \
                held.dtype = SYS__KIND__OBJECT_REFERENCE; held.num_args = 0u; held.op_code = 0ull;        \
                held.args[0] = list;                                                                      \
                if (ok) ok = sys__node_array__set(lists, (uint64_t)(konst), &held);                        \
                (void)sys__heap_object__release(list);                                                    \
            }                                                                                            \
        }                                                                                                \
        at += bytes * qty;                                                                                \
    }
    NN__BUFFER_CLASS_LIST(NN__BUFFER__ZZPRIVATE_CARVE_ROW)
    const uint64_t card_end = at;
    /* ⭐ AND THE RAM CLASSES, WHEN THE HOST TOOK A RAM SPAN FOR THEM: a buffer's `AT` is the card's address
     * for its bytes, which is what every door is handed, and its `HOST_AT` the program's. */
    if (ok && ram_bytes != 0ull) {
        at = ram_card_at; run_base = ram_card_at; run_room = ram_bytes; host_base = ram_at;
        if (ram_card_at == 0ull || ram_at == 0ull) ok = false;
#undef NN__BUFFER__ZZPRIVATE_KEY_SIZE
#undef NN__BUFFER__ZZPRIVATE_KEY_QTY
#define NN__BUFFER__ZZPRIVATE_KEY_SIZE(name)  NN__BUFFER__KEY_RAM_SIZE(name)
#define NN__BUFFER__ZZPRIVATE_KEY_QTY(name)   NN__BUFFER__KEY_RAM_QTY(name)
        NN__BUFFER_RAM_CLASS_LIST(NN__BUFFER__ZZPRIVATE_CARVE_ROW)
    }
#undef NN__BUFFER__ZZPRIVATE_KEY_SIZE
#undef NN__BUFFER__ZZPRIVATE_KEY_QTY
#undef NN__BUFFER__ZZPRIVATE_CARVE_ROW

    sys__heap_node row;
    row.dtype = SYS__KIND__OBJECT_REFERENCE; row.num_args = 0u; row.op_code = 0ull;
    if (ok) { row.args[0] = lists; ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__BUFFERS, &row); }
    if (ok) { row.args[0] = sizes; ok = sys__package__own_row_set(nn_pkg_id, (uint64_t)NN__HATCH__BUFFER_SIZEOF, &row); }

    /* The rows took their own holds. These two are the carve's, and they go back whether it worked or
     * not — on the failing path that is what takes the half-built tables apart. */
    (void)sys__heap_object__release(lists);
    (void)sys__heap_object__release(sizes);
    /* ⭐ AND WHERE THE WALK STOPPED IS THE ARENA'S BASE. Written only on success: a refused carve laid
     * some unknown prefix of the runs down, so its `at` is a figure about a half-built layout and naming
     * a region from it would be worse than having no region. ▶ `expert__header.cuh`. */
    if (ok) *ended = card_end;
    return ok;
}

/* ── RELEASE: THE BUFFER GOES HOME ───────────────────────────────────────────────────────────────────
 * ⛳ THE HOOK IS HANDED A HEAD AND THE LIST WANTS A REFERENCE, which is one line of arithmetic — the body
 * node is the head plus one, and its offset is what every other holder of this buffer has been using.
 * ⛔ IF THE FILING FAILS THE BUFFER IS LOST, and it is lost QUIETLY unless this says so: the count stays
 * at zero, the node's room goes back to the heap, and the span it named has no owner and no record. That
 * is the one outcome here with no later symptom, so it raises. */
static __device__ __noinline__ void nn__buffer__zzpackage_release_internal(sys__heap_node* head,
                                                                          uint64_t* releaser_stack) {
    (void)releaser_stack;                     /* a buffer holds no other object, so nothing is handed on */
    if (head == 0) return;
    const uint64_t kind = head[1].args[NN__BUFFER__CLASS];
    const uint64_t list = nn__buffer__zzpackage_freelist(kind);
    if (list == 0ull) { sys__fault__raise(0ull, NN__BUFFER__FAULT_NO_POOL); return; }
    const sys__heap_node ref = sys__heap_object__reference_to(sys__heap__offset(head + 1));
    /* ⭐ AND THIS IS THE WHOLE OF THE RESURRECTION: the append retains, so the count this release just
     * took to zero is one again by the time `zzprivate_deallocate` asks whether the room should go back.
     * Nothing here tells the allocator anything untrue — the object genuinely has a holder. */
    if (!sys__sublist__append(list, &ref)) sys__fault__raise(0ull, NN__BUFFER__FAULT_NO_POOL);
}

/* ── THE VERB ────────────────────────────────────────────────────────────────────────────────────────
 * `(nn__buffer__getnew  size)` — the SMALLEST class that fits, so a request lands where its round-up
 * waste is least. The classes are role-shaped rather than power-of-two, so in a real model program the
 * fit is expected to be exact; taking the smallest sufficient one is what keeps a mis-sized ask from
 * quietly eating the widest class.
 * ⛔ AN EXHAUSTED CLASS RAISES AND MUST NEVER WAIT. A pool that blocks is a grid that hangs with no
 * suspect, which is this tree's opening warning; an error is a value a program can see and recover from.
 */
/* ⛳ ONE CHOICE FOR BOTH ASKS, over the classes each may choose among — the card's for `getnew`, the RAM's
 * for `getnew_ram` — so the two cannot come to refuse differently. It reads the form and answers the
 * buffer taken, or the fault; writing the form is each verb's own last act, where it can be seen. */
static __device__ inline uint64_t nn__buffer__zzprivate_take(uint64_t form, uint64_t first, uint64_t last,
                                                             sys__heap_node* got) {
    if (sys__sublist__length(form) != 2ull) return SYS__OPCODES__FAULT_ARITY;
    const sys__heap_node want = sys__sublist__nth(form, 1ull);
    if (want.dtype != SYS__KIND__VALUE_INT) return SYS__OPCODES__FAULT_TYPE;

    uint64_t chosen = last;
    uint64_t chosen_bytes = 0ull;
    for (uint64_t k = first; k < last; ++k) {
        const uint64_t bytes = nn__buffer__sizeof(k);
        if (bytes == 0ull) return NN__BUFFER__FAULT_NO_POOL;
        if (bytes < want.args[0]) continue;
        if (chosen == last || bytes < chosen_bytes) { chosen = k; chosen_bytes = bytes; }
    }
    if (chosen == last) return NN__BUFFER__FAULT_SIZE;

    const uint64_t list = nn__buffer__zzpackage_freelist(chosen);
    if (list == 0ull) return NN__BUFFER__FAULT_NO_POOL;
    if (sys__sublist__empty(list)) return NN__BUFFER__FAULT_EXHAUSTED;
    *got = sys__sublist__extract(list, 0ull);
    if (got->dtype != SYS__KIND__OBJECT_REFERENCE) return NN__BUFFER__FAULT_EXHAUSTED;
    return 0ull;
}

static __device__ __noinline__ void nn__buffer__zzpackage_apply_getnew(sys__heap_node* base, uint64_t form) {
    (void)base;
    sys__heap_node got;
    const uint64_t why = nn__buffer__zzprivate_take(form, 0ull, (uint64_t)NN__BUFFER__CLASSES, &got);
    if (why != 0ull) { sys__opcodes__fails(form, why); return; }
    sys__opcodes__becomes(form, &got);
    (void)sys__heap_object__release(got.args[0]);
}

static __device__ __noinline__ void nn__buffer__zzpackage_apply_getnew_ram(sys__heap_node* base, uint64_t form) {
    (void)base;
    sys__heap_node got;
    const uint64_t why = nn__buffer__zzprivate_take(form, (uint64_t)NN__BUFFER__CLASSES,
                                                    (uint64_t)NN__BUFFER__ALL_CLASSES, &got);
    if (why != 0ull) { sys__opcodes__fails(form, why); return; }
    sys__opcodes__becomes(form, &got);
    (void)sys__heap_object__release(got.args[0]);
}

/* ── nn's C INTERFACE — THE ORACLE'S TWO ENDS ── ▶ `contracts/abi/cpu.cuh`. ──────────────────────────────
 * ⛳ THROUGH THE WORKER'S FAMILY, AFTER ITS WORK HAS COMPLETED. The worker is the verb context the boot
 * bound; with none there is no card to read, and the answer is a refusal.
 * ⛳ THE WAIT IS KEPT ON THE WRITE TOO, and it is a claim about the seam and not about HIP: `memory_write`
 * promises the copy QUEUES behind the work — an ordering, not a completion — so a family whose write
 * returned early would satisfy the seam and break this door's promise that the bytes are there. `MEASURED`
 * that removing it leaves the suite passing, because this family's write blocks.
 * ⚠ `REASONED`: a read while resident blocks are running can race the verbs writing that buffer — the card
 * work is waited for, the runners are not. A caller reads a buffer after the program that wrote it. */
#ifndef SYS__SILICON__HARNESS   /* a real build, not a test harness that brings its own silicon */
#include "../contracts/abi/cpu.cuh"
extern "C" {
int nn_abi_read(unsigned long long address, unsigned long long bytes, void* into) {
    const sys__engine__ctx* ctx = sys__engine__ctx_current();
    if (into == 0 || address == 0ull || bytes == 0ull || ctx == 0 || ctx->family == SYS__SILICON_FAMILY__NONE) return 0;
    if (sys__heap__holds((uint64_t)address, (uint64_t)bytes)) return 0;   /* the heap is sys's: `sys.peek` */
    if (!sys__gpu__compute_completed(ctx->family)) return 0;              /* the work that wrote them */
    return sys__gpu__memory_read(ctx->family, into, (const void*)(uintptr_t)address, (size_t)bytes) ? 1 : 0;
}

int nn_abi_write(unsigned long long address, unsigned long long bytes, const void* from) {
    const sys__engine__ctx* ctx = sys__engine__ctx_current();
    if (from == 0 || address == 0ull || bytes == 0ull || ctx == 0 || ctx->family == SYS__SILICON_FAMILY__NONE) return 0;
    if (sys__heap__holds((uint64_t)address, (uint64_t)bytes)) return 0;
    if (!sys__gpu__memory_write(ctx->family, (void*)(uintptr_t)address, from, (size_t)bytes)) return 0;
    return sys__gpu__compute_completed(ctx->family) ? 1 : 0;
}

int nn_abi_map_file(unsigned long long address, unsigned long long bytes, const char* path, unsigned long long offset) {
    const sys__engine__ctx* ctx = sys__engine__ctx_current();
    if (path == 0 || address == 0ull || bytes == 0ull || ctx == 0 || ctx->family == SYS__SILICON_FAMILY__NONE) return 0;
    if (sys__heap__holds((uint64_t)address, (uint64_t)bytes)) return 0;
    if (!sys__gpu__compute_completed(ctx->family)) return 0;              /* nothing still reading what is replaced */
    uint64_t handle = 0ull;
    if (!sys__gpu__file_open(ctx->family, path, &handle)) return 0;
    const bool mapped = sys__gpu__memory_map_file(ctx->family, (void*)(uintptr_t)address, (size_t)bytes, handle, (uint64_t)offset);
    sys__gpu__file_close(ctx->family, handle);                           /* the mapping keeps the file */
    return mapped ? 1 : 0;
}
}
#endif

#endif /* SILVANN__PACKAGES_NN_CPU_BUFFER__IMPL_CUH */
