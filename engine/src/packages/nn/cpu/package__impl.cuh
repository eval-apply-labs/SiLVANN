#ifndef SILVANN__PACKAGES_NN_CPU_PACKAGE__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_PACKAGE__IMPL_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "buffer__header.cuh"           /* the classes its span is the sum of, and their ids */
#include "buffer__impl.cuh"             /* and the carve that cuts the span into them */
#include "expert__header.cuh"           /* the arena that begins where that carve ends */
#include "cartridge__header.cuh"        /* the conversation's tiers, text and deltanet state */
#include "cartridge__impl.cuh"          /* and the carve that cuts them out after the arena */
#include "package__header.cuh"          /* the contract this definition answers */
#include "../../sys/cpu/package__header.cuh"   /* the three operations an init is written out of */
#include "../../sys/cpu/opcodes/opcodes.cuh"           /* how a verb answers */
#include "../../sys/cpu/settings.cuh"          /* where its span's size is written down */
#include "../../sys/cpu/silicon/silicon__header.cuh"   /* and who can actually take it */
/* ══ nn — WHAT THIS PACKAGE DOES WHEN IT IS STOOD UP ═════════════════════════════════════════════════
 *
 * ⭐⭐ THE DEPENDENCY IS DECLARED BY BEING NAMED, AND BY NOTHING ELSE. The line below mentions
 * `SYS__PACKAGE__INIT`, and a build in which `sys` were not on the roster would not have that constant —
 * so "does what I depend on exist" is answered by the compiler, at the only moment it could be answered
 * for free. There is no manifest of dependencies to keep in step with the code, and no way for this file
 * to be wrong about what it needs while still building.
 * ⇒ ★ THE TEST FOR *WHETHER IT HAS RUN* IS THEN THE ONLY ONE LEFT, and the list answers that: a call
 * still in it has not happened. Two questions, one of them free and the other already recorded.
 *
 * ⛳ TODAY IT CANNOT ACTUALLY FIRE, AND THAT IS WORTH WRITING DOWN RATHER THAN LEAVING TO BE INFERRED.
 * The roster lists `sys` first, so its call is always ahead of this one and is always gone by the time
 * this runs. The branch below is therefore correct and unexercised in the shipping order — which is
 * exactly the kind of arm this tree has repeatedly found to be broken when it was finally needed, so the
 * suite builds the list in the WRONG order on purpose and requires this to defer.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


static __device__ __noinline__ void nn__package__zzpackage_apply_init(sys__heap_node* base, uint64_t form) {
    (void)base;
    uint64_t why = 0ull;
    const uint64_t list = sys__package__list_of(form, &why);
    if (list == 0ull) { sys__package__refuses(form, why); return; }

    /* ⛳ THE WAIT IS A REWRITE AND NOT A WAIT. If the language has not been stood up yet, this call goes
     * back at the tail and answers nothing; the driver drops the head either way, so the next call runs
     * and this one comes round once the list has turned over. */
    if (sys__package__waiting_for(list, SYS__PACKAGE__INIT)) {
        if (!sys__package__come_back_later(list, NN__PACKAGE__INIT)) {
            /* ⛔ A DEFERRAL THAT DID NOT LAND IS NOT A SKIP. Answering nothing here would leave a package
             * that meant to run later and will now never run at all, with every counter balanced and no
             * event anywhere — so it refuses instead, and the boot stops on a word that names the cause. */
            sys__package__refuses(form, SYS__PACKAGE__FAULT_ROOM);
            return;
        }
        const sys__heap_node later = sys__heap_node__nothing();
        sys__opcodes__becomes(form, &later);
        return;
    }

    /* ⭐ THE ROOM THIS PACKAGE ASKED FOR, READ BY KNOWING WHO IT IS. The engine took it, sliced it and put
     * this package's part behind its own hatch entry; nothing here had to be told where it is.
     * ⚖⚖ AND AN ABSENT SPAN IS NOT A FAULT HERE — RULED, *"absent is fine, partial refuses"*.
     * ⛔⛔ THIS REFUSED ON `VALUE_NULL` UNTIL THE SPAN BECAME A SUM, AND IT WAS RIGHT TO WHILE nn ALWAYS
     * TOOK ITS ONE-MEBIBYTE PLACEHOLDER: there was no such thing as a package that had asked for nothing,
     * so an empty entry could only mean the room went missing. With the placeholder gone, a config naming
     * no nn key legitimately takes no span — and `e.boot()` with no config at all is the case every test
     * in the device suite boots in. ⇒ ★ A CHECK CAN BE CORRECT AND THEN BE MADE WRONG BY A CHANGE
     * ELSEWHERE THAT NEVER TOUCHES IT: what moved was not this code but the set of states reachable by it.
     * ⛳ THE THREE STATES ARE NOW DISTINCT, and the hatch already spells them:
     *     NOTHING            nobody asked. Stand up with no span; a carve is what refuses, later.
     *     AN INT, both set   the room arrived. What every real run has.
     *     anything else      a span was asked for and did not arrive — that is still a fault, and it is
     *                        caught HERE, at the one moment there is something to be done about it. */
    if (!sys__package__hatch_stands()) {
        sys__package__refuses(form, SYS__PACKAGE__FAULT_HATCH);
        return;
    }
    const sys__heap_node room = sys__package__own_room(nn_pkg_id);
    if (room.dtype != SYS__KIND__VALUE_NULL
        && (room.dtype != SYS__KIND__VALUE_INT || room.args[SYS__PACKAGE__HATCH_AT] == 0ull
            || room.args[SYS__PACKAGE__HATCH_BYTES] == 0ull)) {
        sys__package__refuses(form, SYS__PACKAGE__FAULT_HATCH);
        return;
    }

    /* ⛳ THIS INIT DOES NOT RE-READ THE SPAN'S TOTAL, AND THERE IS NO TOTAL ANYWHERE TO RE-READ. What the
     * carve below reads is the SAME PER-CLASS KEYS the host summed — two readers over one text, which is
     * the arrangement `settings.cuh` describes and the differential test holds together. The hatch is what
     * says the room ARRIVED; comparing a re-derived total against it would be the sum checked against
     * itself, since both come from the same lines.
     * ⛔ WHAT THE CARVE DOES CHECK IS THAT ITS RUNS FIT THE ROOM, which is a different question and the one
     * with no later symptom: a run that walks past the end hands out addresses belonging to nobody.
     * ▶ `docs_history_internal/nn_retired_symbols.md`. */
    /* ⭐⭐ AND HERE IS WHAT THE SPAN WAS FOR: the buffer classes are cut out of it and their free lists
     * filled, which is every line of it making a heap object — so it belongs on this side of the boot
     * split and could never have been the host init's.
     * ⛳ A PACKAGE WITH NO SPAN CARVES NOTHING AND STANDS UP ANYWAY. That is the `e.boot()` case the whole
     * suite runs in; what refuses later is an allocation, at the point where there is something to say. */
    uint64_t after_buffers = 0ull;
    if (room.dtype == SYS__KIND__VALUE_INT
        && !nn__buffer__zzpackage_carve(room.args[SYS__PACKAGE__HATCH_AT],
                                        room.args[SYS__PACKAGE__HATCH_BYTES],
                                        room.args[SYS__PACKAGE__HATCH_RAM_AT],
                                        room.args[SYS__PACKAGE__HATCH_RAM_CARD_AT],
                                        room.args[SYS__PACKAGE__HATCH_RAM_BYTES],
                                        &after_buffers)) {
        sys__package__refuses(form, NN__BUFFER__FAULT_CONFIG);
        return;
    }

    /* ⭐⭐ AND THE EXPERT L1 ARENA IS NAMED FROM WHERE THAT CARVE STOPPED. ⚖ *"the arena is a REGION of
     * nn's span, not its own allocation"* — so this takes nothing and computes nothing: the base is the
     * carve's own running total, handed over rather than summed a second time.
     * ⛔ THE ORDER IS A DEPENDENCY AND NOT A STYLE. The arena's base is defined as the first byte after the
     * last buffer run, so it cannot be stood up before the thing that lays those runs down.
     * ⛳ AND IT IS INSIDE THE SAME `VALUE_INT` GUARD, which is the same ruling both halves obey: a package
     * with no span carves nothing and names no arena and stands up anyway — the `e.boot()` case the whole
     * device suite runs in. What refuses later is an allocation, where there is something to say. */
    if (room.dtype == SYS__KIND__VALUE_INT
        && !nn__expert__zzpackage_stand_arena(after_buffers,
                                              room.args[SYS__PACKAGE__HATCH_AT],
                                              room.args[SYS__PACKAGE__HATCH_BYTES])) {
        sys__package__refuses(form, NN__EXPERT__FAULT_CONFIG);
        return;
    }

    /* ⭐⭐ AND THE CARTRIDGE BEGINS WHERE THE ARENA ENDS — THE THIRD LINK OF ONE CHAIN, AND THE SAME MOVE
     * BOTH TIMES. The buffers' carve handed the arena its base; the arena's row already holds its own base
     * and length, so the sum of those two is the cartridge's, and nothing is computed twice or written
     * down anywhere for two readers to disagree about.
     * ⛔ THE ORDER IS A DEPENDENCY AND NOT A STYLE, for the third time: each region is DEFINED as the first
     * byte after the last one, so none can be stood up before its predecessor.
     * ⛳ AND IT IS INSIDE THE SAME `VALUE_INT` GUARD as the two before it, obeying the same ruling: a
     * package with no span carves nothing and holds no conversation and stands up anyway. */
    if (room.dtype == SYS__KIND__VALUE_INT) {
        uint64_t after_cartridge = 0ull;
        if (!nn__cartridge__zzpackage_carve(nn__expert__arena_at() + nn__expert__arena_bytes(),
                                            room.args[SYS__PACKAGE__HATCH_AT],
                                            room.args[SYS__PACKAGE__HATCH_BYTES],
                                            &after_cartridge)) {
            sys__package__refuses(form, NN__CARTRIDGE__FAULT_CONFIG);
            return;
        }
    }

    /* ⭐⭐⭐ AND THE TYPE TABLE — THE PAGE LIST DESCRIPTOR. ⚖ *"one page collection per matrix type"*.
     * ⛔ IT IS INSIDE NO SPAN GUARD AND STILL NEEDS THE ARENA, which is not a contradiction: a config that
     * names no types stands with none, and a config that names types on a card with no arena is asking for
     * collections with nowhere to put them. So the guard is in the stand, where it can tell the two apart,
     * rather than here where it could only skip both.
     * ⛳ IT COMES AFTER THE ARENA BECAUSE IT CARVES IT: each type's run begins where the last one ended,
     * and the walk is checked against the arena's bytes — the same two-readers-of-one-text check the
     * buffers' carve makes against the span. ▶ `expert__header.cuh`. */
    if (!nn__expert__zzpackage_stand_types()) {
        sys__package__refuses(form, NN__EXPERT__FAULT_TYPES);
        return;
    }

    /* ⭐ AND THE HANDED LIST, WHICH IS THE ONE THING LEFT OF THE PAGE LAYER — the ledger of slots given
     * out and not yet published.
     * ⛳ IT STANDS ONLY WHERE THERE IS AN ARENA, which is the same guard the page index carried and for
     * the same reason: a card with no arena hands out nothing, so a ledger of what it handed out is a
     * heap object nobody can ever put a row in. ⛔ AND THAT IS THE `e.boot()` CASE THE WHOLE DEVICE SUITE
     * RUNS IN — a config naming no nn key takes no span, and it is not a fault. `handed_count` answers 0
     * for such a card, which is TRUE rather than a stand-in for "no list".
     * ⛔ THE ONE STATE THAT IS A FAULT is having an arena and failing anyway: that is an allocator that
     * could not stand a list up, which is a heap too small and is worth stopping the boot for. */
    if (nn__expert__arena_bytes() != 0ull && !nn__expert__zzpackage_stand_handed()) {
        sys__package__refuses(form, NN__EXPERT__FAULT_ROOM);
        return;
    }
    /* ⭐⭐ AND THE EXPERT INDEX, WHICH IS SIZED BY THE MODEL'S LAYER COUNT AND SO COMES AFTER THE CARVE.
     * ⚖ *"lets keep the indexes for everyone but they'd point at a null subarray"* — sparse, so the index
     * is the GLOBAL layer id and no rank keeps a translation map.
     * ⛳ THE LAYER TABLE IS THE ONLY THING ON THIS CARD THAT KNOWS HOW MANY LAYERS THE MODEL HAS, which is
     * why this is here and not beside the page index: the two were built the same day and the first is
     * what makes the second sizable at all. A card with no cartridge answers zero and stands up holding
     * no experts, which is the `e.boot()` case and not a fault.
     * ⛔ THE ONE STATE THAT IS A FAULT is having a layer count and failing anyway — that is an allocator
     * that could not stand two arrays up, which is a heap too small and is worth stopping the boot for. */
    if (!nn__expert__zzpackage_stand_index(nn__cartridge__model_layers())) {
        sys__package__refuses(form, NN__EXPERT__FAULT_INDEX);
        return;
    }

    const sys__heap_node done = sys__heap_node__nothing();
    sys__opcodes__becomes(form, &done);
}

/* ── nn's HOST INIT: THE SPAN, SIZED BY THE CONFIG AND TAKEN BY THIS PACKAGE ─────────────────────────
 *
 * ⚖ *"one allocation per package, and we publish the starting point in a register reference. since that
 * is the extent it should be part of the device params, so the packages know what is the size they have
 * to work with."* The EXTENT reaches the package through the hatch, which carries the bytes beside the
 * address — so the span is sized here, once, and read there, once, and there is no second parse of it.
 *
 * ⭐ AND THE EXPERT L1 ARENA IS INSIDE THIS ONE ALLOCATION, NOT BESIDE IT. ⚖ *"l1 expert arena is held by
 * nn, it is its own space in the sense that is not part of the heap"* · *"the offset is from the start of
 * the nn address space."* ⛔ So the arena has its own BASE and not its own ALLOCATION
 * (`expert_lru_node_array.md` §⑤ⓐ), and the base is arithmetic from this span's start. "Its own space" meant NOT CARVED OUT OF
 * THE HEAP, never owned by nobody — which is why "one allocation per package" has no exception in it.
 *
 * ⚖⚖ AND ITS FIGURES LIVE IN `device_params`, RULED — *"device_params are those available to
 * the device, it is like disabled accessible toilets that are not reserved for the disabled."* ⭐ THE
 * SECTION SAYS WHO MAY READ A FIGURE, NOT WHO MAY ONLY READ IT: the host holds the whole file either way,
 * and `settings.cuh` already wrote the reason down — *"that is what lets a package's room size live in
 * exactly one place while both sides use it."*
 * ⛔⛔ AND FOR THESE FOUR TERMS BOTH SIDES GENUINELY DO. The host needs them to SIZE the allocation, which
 * happens before there is an engine; the opcode init needs the same figures to CARVE that allocation into
 * buffer classes, which needs one. A key in `boot_commands` never reaches the card at all — only the
 * `device_params` section is sliced and sent (`abi.cuh`) — so the carve could not have been written.
 * ⇒ ★ THE BOOT SPLIT PLACES THE WORK AND SAYS NOTHING ABOUT THE DATA, and a figure two tiers both act on
 * is not an exception to the axis but a case it does not speak to. `sys__alloc_size_mb` stays where it is
 * because only the host ever acts on it.
 *
 * ⛔ IT TOUCHES ONLY ITS OWN, which is the rule the host loop runs under: this reads the config and takes
 * its span, and nothing else. Anything that has to happen after another package is the opcode init's.
 * ⛳ AND A CONFIG THAT SAYS SOMETHING UNREADABLE IS A REFUSAL RATHER THAN A FALLBACK, which is the rule
 * every reader on this path runs under: a span silently a fraction of what somebody asked for is the kind
 * of wrong that shows up as a performance mystery three weeks later rather than as a failure. */
/* Add, and say so rather than wrapping. A span is the one figure here with no later symptom: a total that
 * wrapped is SMALL, and small allocates fine. */
static inline bool nn__package__zzprivate_add(uint64_t* into, uint64_t more) {
    if (*into > 0xFFFFFFFFFFFFFFFFull - more) return false;
    *into += more;
    return true;
}

/* ── ⭐⭐ THE BUFFERS TERM: COUNTED TWICE, BY TWO ROUTES, AND THE DISAGREEMENT IS THE CHECK ───────────
 *
 * The classes are read by walking `NN__BUFFER_CLASS_LIST` — the set the ENGINE knows — because the id a
 * class is indexed by has to come from the build and not from a file's line order. ▶ `buffer__header.cuh`.
 *
 * ⛔⛔ AND READING ONLY THE LIST OPENS A HOLE THE LIST CANNOT SEE. A class the FILE names and this build
 * does not — `nn__buffer_typoed__size_bytes:4096` with its `__qty:2` — is read by nobody: the walk asks
 * for the five names it knows, the two lines are never visited, and the span is short by exactly the
 * amount nobody can look up. Nothing anywhere says so, and the first symptom is a carve that runs out of
 * room in a package that asked for the right total by its own arithmetic.
 *
 * ⛳ SO BOTH ROUTES RUN AND THEY MUST AGREE:
 *     the LIST walk    size x qty over the classes the ENGINE knows   ⇒ what will actually be carved
 *     the PAIRED SUM   size x qty over the classes the FILE names     ⇒ what somebody asked for
 * A disagreement is a config naming a class this build does not have, and it refuses.
 * ⇒ ★★ ONE MECHANISM CANNOT CHECK ITSELF, and the cheap way to check a SET is to count it twice by
 * different routes. It is the same shape as `host_size` and `host_count` refusing each other's keys, one
 * level up: there, two readers hold each other to a unit; here, two walks hold each other to a set.
 *
 * ⛳ AND IT CATCHES THE MIRROR CASE WITHOUT A SECOND MECHANISM — a class in the list that the file does
 * not name is `ABSENT` on both its keys, so the count of named classes falls short of the list's length
 * and the same refusal applies. Half a set is an edit that went wrong, exactly as half a span is. */
static inline int nn__package__zzprivate_buffers(const char* config, uint64_t length, uint64_t* answer) {
    uint64_t asked = 0ull;
    const int paired = sys__settings__host_paired_sum(config, length, "device_params",
                                                      NN__BUFFER__KEY_PREFIX,
                                                      NN__BUFFER__KEY_TAIL_SIZE,
                                                      NN__BUFFER__KEY_TAIL_QTY, &asked);
    if (paired == SYS__SETTINGS__READ_MALFORMED) return SYS__SETTINGS__READ_MALFORMED;

    uint64_t made = 0ull;
    unsigned named = 0u;
#define NN__PACKAGE__ZZPRIVATE_BUFFER_ROW(id, name, konst)                                              \
    {                                                                                                   \
        uint64_t bytes = 0ull, qty = 0ull;                                                              \
        const int a = sys__settings__host_size(config, length, "device_params",                         \
                                               NN__BUFFER__KEY_SIZE(name), &bytes);                     \
        const int b = sys__settings__host_count(config, length, "device_params",                        \
                                                NN__BUFFER__KEY_QTY(name), &qty);                       \
        /* ⛔ A SIZE WITHOUT ITS COUNT IS NOT A CLASS. Taking one of the two would size a run out of a    \
         * figure whose other half somebody deleted, which allocates fine and is wrong by a factor.  */  \
        if (a == SYS__SETTINGS__READ_MALFORMED || b == SYS__SETTINGS__READ_MALFORMED)                   \
            return SYS__SETTINGS__READ_MALFORMED;                                                       \
        if (a != b) return SYS__SETTINGS__READ_MALFORMED;                                               \
        if (a == SYS__SETTINGS__READ_FOUND) {                                                           \
            named += 1u;                                                                                \
            if (bytes != 0ull && qty > 0xFFFFFFFFFFFFFFFFull / bytes)                                   \
                return SYS__SETTINGS__READ_MALFORMED;                                                   \
            const uint64_t run = bytes * qty;                                                           \
            if (made > 0xFFFFFFFFFFFFFFFFull - run) return SYS__SETTINGS__READ_MALFORMED;               \
            made += run;                                                                                \
        }                                                                                               \
    }
    NN__BUFFER_CLASS_LIST(NN__PACKAGE__ZZPRIVATE_BUFFER_ROW)
#undef NN__PACKAGE__ZZPRIVATE_BUFFER_ROW

    /* Nobody named a buffer class by either route: the term is absent, and absent is fine. */
    if (named == 0u && paired == SYS__SETTINGS__READ_ABSENT) return SYS__SETTINGS__READ_ABSENT;
    /* A class this build has and the file does not name — including the case where the file named
     * nothing but classes this build has never heard of. */
    if (named != NN__BUFFER__CLASSES) return SYS__SETTINGS__READ_MALFORMED;
    /* And the other direction: the file asked for more than the walk will carve. */
    if (paired != SYS__SETTINGS__READ_FOUND || asked != made) return SYS__SETTINGS__READ_MALFORMED;

    *answer = made;
    return SYS__SETTINGS__READ_FOUND;
}

/* ── THE CARTRIDGE'S TIERS — THE SAME TWO-ROUTE WALK THE BUFFER CLASSES GET, AND FOR THE SAME REASON ──
 * `host_paired_sum` answers a TOTAL over whatever tiers a file NAMES, and it never says *"tq6 is absent"*.
 * A config naming two of the four sums perfectly and is short by half a cartridge — so the sum is checked
 * against a walk of the tiers THIS BUILD HAS, in both directions.
 * ⇒ ★★ A SUM OVER A SET AND A COMPLETE COVER OF IT ARE DIFFERENT QUESTIONS. Nothing about the sum is
 * wrong; it was never going to answer the second one.
 * ⛔ AND WITHOUT THIS THE FAILURE LANDS ON THE CARD instead, as the carve's *"the two readers disagree"*
 * refusal — which is true, and names the wrong cause: the readers agree perfectly about a config that is
 * missing a tier. ⇒ ★ A CHECK IN THE WRONG TIER STILL FIRES AND STILL MISDIAGNOSES. */
static inline int nn__package__zzprivate_cartridge(const char* config, uint64_t length, uint64_t* answer) {
    uint64_t asked = 0ull;
    const int paired = sys__settings__host_paired_sum(config, length, "device_params",
                                                      NN__CARTRIDGE__KEY_PREFIX,
                                                      NN__CARTRIDGE__KEY_TAIL_TOKENS,
                                                      NN__CARTRIDGE__KEY_TAIL_BYTES, &asked);
    if (paired == SYS__SETTINGS__READ_MALFORMED) return SYS__SETTINGS__READ_MALFORMED;

    uint64_t made = 0ull;
    unsigned named = 0u;
#define NN__PACKAGE__ZZPRIVATE_TIER_ROW(id, name, konst, rot)                                           \
    {                                                                                                   \
        uint64_t n = 0ull, b = 0ull;                                                                    \
        const int a = sys__settings__host_count(config, length, "device_params",                        \
                                                NN__CARTRIDGE__KEY_TOKENS(name), &n);                   \
        const int c = sys__settings__host_count(config, length, "device_params",                        \
                                                NN__CARTRIDGE__KEY_BYTES(name), &b);                    \
        if (a == SYS__SETTINGS__READ_MALFORMED || c == SYS__SETTINGS__READ_MALFORMED)                   \
            return SYS__SETTINGS__READ_MALFORMED;                                                       \
        if (a != c) return SYS__SETTINGS__READ_MALFORMED;                                               \
        if (a == SYS__SETTINGS__READ_FOUND) {                                                           \
            named += 1u;                                                                                \
            if (n != 0ull && b > 0xFFFFFFFFFFFFFFFFull / n) return SYS__SETTINGS__READ_MALFORMED;        \
            const uint64_t run = n * b;                                                                 \
            if (made > 0xFFFFFFFFFFFFFFFFull - run) return SYS__SETTINGS__READ_MALFORMED;               \
            made += run;                                                                                \
        }                                                                                               \
    }
    NN__CARTRIDGE_TIER_LIST(NN__PACKAGE__ZZPRIVATE_TIER_ROW)
#undef NN__PACKAGE__ZZPRIVATE_TIER_ROW

    if (named == 0u && paired == SYS__SETTINGS__READ_ABSENT) return SYS__SETTINGS__READ_ABSENT;
    if (named != NN__CARTRIDGE__TIERS) return SYS__SETTINGS__READ_MALFORMED;
    if (paired != SYS__SETTINGS__READ_FOUND || asked != made) return SYS__SETTINGS__READ_MALFORMED;

    *answer = made;
    return SYS__SETTINGS__READ_FOUND;
}

/* ── THE DELTANET STATE'S TWO KEYS, WHICH ARE A PAIR AND NOT TWO TERMS ───────────────────────────────
 * ⛳ SAME SHAPE AS A BUFFER CLASS — a per-thing size and a count of the things — and it is a pair for the
 * same reason: taking one of the two would size a region from a figure whose other half somebody deleted,
 * which allocates fine and is wrong by a factor. ⇒ ★ THE TWO ARE ONE TERM BECAUSE THEY ARE ONE DECISION.
 * ⛔ AND THIS IS THE FIFTH TENANT THE FOOT OF THIS FILE COSTED AND DELIBERATELY DID NOT SUM, because no
 * ruling named it. ⚖ names it. The note stays where it is: it is the arithmetic, and this is
 * the act of paying it. */
static inline int nn__package__zzprivate_deltanet(const char* config, uint64_t length, uint64_t* answer,
                                                  uint64_t* counted) {
    if (counted != 0) *counted = 0ull;
    uint64_t bytes = 0ull, layers = 0ull;
    const int a = sys__settings__host_count(config, length, "device_params",
                                            NN__CARTRIDGE__KEY_DELTANET_BYTES, &bytes);
    const int b = sys__settings__host_count(config, length, "device_params",
                                            NN__CARTRIDGE__KEY_DELTANET_LAYERS, &layers);
    if (a == SYS__SETTINGS__READ_MALFORMED || b == SYS__SETTINGS__READ_MALFORMED)
        return SYS__SETTINGS__READ_MALFORMED;
    if (a != b) return SYS__SETTINGS__READ_MALFORMED;
    if (a == SYS__SETTINGS__READ_ABSENT) return SYS__SETTINGS__READ_ABSENT;
    if (bytes != 0ull && layers > 0xFFFFFFFFFFFFFFFFull / bytes) return SYS__SETTINGS__READ_MALFORMED;
    *answer = bytes * layers;
    /* ⛳ THE COUNT IS HANDED BACK AS WELL AS SPENT, because the layer-types array has to be as long as
     * every mechanism's layers put together and this is the one place the deltanet half is read. */
    if (counted != 0) *counted = layers;
    return SYS__SETTINGS__READ_FOUND;
}

/* ── ⭐⭐ THE HOST'S HALF OF THE LAYER-TYPES ARRAY — PRESENCE AND LENGTH, AND DELIBERATELY NO MORE ────
 *
 * ⛔⛔ IT IS HERE AT ALL BECAUSE "ABSENT IS FINE, PARTIAL REFUSES" HAS TO MEAN THE SAME SET OF KEYS ON
 * BOTH SIDES. If the card requires the array and the host does not, then a config carrying the other
 * twelve keys sizes a span here, takes VRAM, and is refused on the card — a refusal arriving at the one
 * place nothing can be done about it, from two readers that disagree about what a whole cartridge is.
 * That disagreement is the exact failure the two-reader arrangement exists to catch, so it cannot be
 * allowed to live inside it.
 *
 * ⛳ AND THE DIVISION OF LABOUR IS BY WHAT EACH SIDE ACTS ON, not by tidiness. The host acts on the
 * COUNTS — it sums `per_layer × fullattn` and `dn_bytes × dn_layers` — so its interest in the array is
 * that it exists and is as long as those counts say. The CARD acts on the letters and the lanes, so it
 * is the card that decodes each letter and tallies them per mechanism. ⇒ ★ NEITHER SIDE CHECKS WHAT IT
 * DOES NOT USE, and between them every field is checked by the reader that would be wrong about it. */
static inline int nn__package__zzprivate_layer_types(const char* config, uint64_t length,
                                                     uint64_t declared, uint64_t* layers) {
    if (layers == 0) return SYS__SETTINGS__READ_MALFORMED;
    *layers = 0ull;
    if (config == 0 || length == 0ull) return SYS__SETTINGS__READ_ABSENT;
    const char* section = 0; uint64_t section_length = 0ull;
    const int found = sys__settings__host_section(config, length, "device_params", &section, &section_length);
    if (found != SYS__SETTINGS__READ_FOUND) return found;
    const char* value = 0; uint64_t value_length = 0ull;
    const int said = sys__settings__host_value(section, section_length,
                                               NN__CARTRIDGE__KEY_LAYER_TYPES, &value, &value_length);
    if (said != SYS__SETTINGS__READ_FOUND) return said;
    /* ⛔ AN EMPTY ARRAY IS MALFORMED, NOT A MODEL WITH NO LAYERS. Somebody wrote the key and then wrote
     * nothing after it, which is an edit that went wrong. */
    if (value_length == 0ull || value_length != declared) return SYS__SETTINGS__READ_MALFORMED;
    *layers = value_length;
    return SYS__SETTINGS__READ_FOUND;
}

/* ⭐⭐ THE SPAN'S TOTAL IS NOT A CONFIG KEY, AND THAT IS THE POINT OF THIS FUNCTION.
 * ⚖ *"the setup could also be used to take the sum of the values"*. A total and its parts are two writable
 * numbers obliged to agree with nothing checking them, and this tree has now applied that rule three times.
 * ⇒ ★ THE ONLY FIGURE THAT CANNOT BE WRONG IS THE ONE NOBODY WRITES. This sums what it is about to carve
 * and asks for THAT, so a config whose parts change cannot leave a stale total behind it.
 *
 * FOUR TERMS, AND EACH IS READ BY THE READER THAT MATCHES ITS KEY'S SHAPE:
 *     the buffers    Σ size × qty over the classes THIS BUILD HAS, checked against the classes the
 *                    file names — ▶ the two-route walk above                `nn__buffer_<class>__…`
 *     the L1 arena   one size                                            `nn__expert_cache_l1__size_mb`
 *     the cartridge  Σ tokens × bytes-per-token-per-layer over tiers     `nn__cartridge_<tier>__…`
 *                    × the FULL-ATTENTION layer count                    `nn__model__kv_cache_layers_…`
 *
 * ⛔ ONLY FULL-ATTENTION LAYERS MULTIPLY THE CARTRIDGE. A deltanet layer's state is constant in the
 * sequence length, so it does not scale with tokens and cannot be part of a per-token product. ⚠ IT IS NOT
 * FREE EITHER — ▶ the note at the end of this file, which is a finding and not a thing this sums.
 * ⛳ SO THE LAYER COUNT HERE IS A SINGLE KEY AND NOT §①ⓐ's PREFIX SUM: that sum answers the model's TOTAL
 * layer count, which is a different question from how many of them hold a per-token cache.
 *
 * ⚖ THE COST OF A QUANTISED TOKEN IS THE CONVERTER'S TO STATE, RULED. A tq6/tq4 token is not
 * geometry — the format stores indices AND fp16 per-block scales — and `src` has no quantised KV path to
 * be consistent with. So the file carries the per-token-per-layer BYTES of each tier and this knows no tier
 * names, no bit widths and no scale layout. A new tier is two lines and no code.
 *
 * ⚖⚖ ABSENT IS FINE, PARTIAL REFUSES — ruled when the alternative was measured against the
 * suites. A config naming NO nn key takes no span and the boot stands, which is the no-config case every
 * test boots in; a config naming SOME of them is a mistake and refuses, because half a span is not a
 * default. ⇒ ★ IT IS THE SAME ABSENT/MALFORMED SPLIT THE READERS THEMSELVES MAKE, applied one level up:
 * "nobody asked" and "somebody asked wrongly" are different facts about a config, and collapsing them is
 * exactly how a machine boots at a size nobody chose. */
/* ── THE RAM CLASSES' TOTAL — the same two routes as the card's, over their own prefix ────────────────
 * ⛳ OPTIONAL WHERE THE CARD'S ARE NOT: absent names no RAM span at all. Half the RAM set is refused for
 * the card's reason — an edit that went wrong. */
static inline int nn__package__zzprivate_ram_buffers(const char* config, uint64_t length, uint64_t* answer) {
    uint64_t asked = 0ull;
    const int paired = sys__settings__host_paired_sum(config, length, "device_params",
                                                      NN__BUFFER__KEY_RAM_PREFIX,
                                                      NN__BUFFER__KEY_TAIL_SIZE,
                                                      NN__BUFFER__KEY_TAIL_QTY, &asked);
    if (paired == SYS__SETTINGS__READ_MALFORMED) return SYS__SETTINGS__READ_MALFORMED;
    uint64_t made = 0ull;
    unsigned named = 0u;
#define NN__PACKAGE__ZZPRIVATE_RAM_ROW(id, name, konst)                                                 \
    {                                                                                                   \
        uint64_t bytes = 0ull, qty = 0ull;                                                              \
        const int a = sys__settings__host_size(config, length, "device_params",                         \
                                               NN__BUFFER__KEY_RAM_SIZE(name), &bytes);                 \
        const int b = sys__settings__host_count(config, length, "device_params",                        \
                                                NN__BUFFER__KEY_RAM_QTY(name), &qty);                   \
        if (a == SYS__SETTINGS__READ_MALFORMED || b == SYS__SETTINGS__READ_MALFORMED || a != b)         \
            return SYS__SETTINGS__READ_MALFORMED;                                                       \
        if (a == SYS__SETTINGS__READ_FOUND) {                                                           \
            named += 1u;                                                                                \
            if (bytes != 0ull && qty > 0xFFFFFFFFFFFFFFFFull / bytes) return SYS__SETTINGS__READ_MALFORMED; \
            if (made > 0xFFFFFFFFFFFFFFFFull - bytes * qty) return SYS__SETTINGS__READ_MALFORMED;       \
            made += bytes * qty;                                                                        \
        }                                                                                               \
    }
    NN__BUFFER_RAM_CLASS_LIST(NN__PACKAGE__ZZPRIVATE_RAM_ROW)
#undef NN__PACKAGE__ZZPRIVATE_RAM_ROW
    if (named == 0u && paired == SYS__SETTINGS__READ_ABSENT) return SYS__SETTINGS__READ_ABSENT;
    if (named != (unsigned)(NN__BUFFER__ALL_CLASSES - NN__BUFFER__CLASSES)) return SYS__SETTINGS__READ_MALFORMED;
    if (paired != SYS__SETTINGS__READ_FOUND || asked != made || made == 0ull) return SYS__SETTINGS__READ_MALFORMED;
    *answer = made;
    return SYS__SETTINGS__READ_FOUND;
}

static inline bool nn__package__host_init(const char* config, uint64_t length, sys__package_room* room) {
    if (room == 0) return false;
    room->at = 0; room->bytes = 0ull;
    room->ram_at = 0; room->ram_card_at = 0; room->ram_bytes = 0ull;

    uint64_t ram = 0ull;
    const int ram_said = nn__package__zzprivate_ram_buffers(config, length, &ram);
    if (ram_said == SYS__SETTINGS__READ_MALFORMED) return false;

    uint64_t buffers = 0ull, arena = 0ull, per_layer = 0ull, fullattn = 0ull;
    uint64_t text = 0ull, deltanet = 0ull, dn_layers = 0ull, model_layers = 0ull;

    /* ⛔ THESE THREE ARE READ BEFORE THE LIST RATHER THAN INSIDE IT, because the array's check needs the
     * two counts already in hand. A braced initialiser does evaluate left to right, but a reader should
     * not have to know that to see that this is correct. */
    const int fullattn_said = sys__settings__host_count(config, length, "device_params",
                                                        NN__CARTRIDGE__KEY_LAYERS, &fullattn);
    const int deltanet_said = nn__package__zzprivate_deltanet(config, length, &deltanet, &dn_layers);
    const int types_said    = nn__package__zzprivate_layer_types(config, length,
                                                                 fullattn + dn_layers, &model_layers);

    const int said[NN__PACKAGE__SPAN_TERMS] = {
        nn__package__zzprivate_buffers(config, length, &buffers),
        sys__settings__host_size(config, length, "device_params",
                                 "nn__expert_cache_l1__size_mb", &arena),
        nn__package__zzprivate_cartridge(config, length, &per_layer),
        fullattn_said,
        /* ⚖ *"add the field for the conversation text, and also its config parameter sizer."* One size,
         * like the arena — a flat region with no interior structure this package knows anything about. */
        sys__settings__host_size(config, length, "device_params",
                                 NN__CARTRIDGE__KEY_TEXT, &text),
        /* ⚖ *"use the max deltanet state for the context window so we have no oom."* ⛳ IT IS O(1) IN THE
         * SEQUENCE LENGTH — a recurrent state, not a per-token cache — so this is the model's own maximum
         * and not a figure that grows with the conversation. Sizing it at that maximum is precisely what
         * turns a mid-run out-of-memory into a refusal at boot, which is the whole ask. */
        deltanet_said,
        /* ⚖ *"an array that specifies what attention type is used by that layer."* It adds no bytes to
         * the span — it is the ORDER of the regions the other terms sized, not a region of its own. */
        types_said,
    };
    (void)model_layers;

    unsigned spoken = 0u;
    for (unsigned i = 0u; i < NN__PACKAGE__SPAN_TERMS; ++i) {
        if (said[i] == SYS__SETTINGS__READ_MALFORMED) return false;
        if (said[i] == SYS__SETTINGS__READ_FOUND)     spoken += 1u;
    }
    /* ⛔ A RAM POOL WITH NO CARD POOL HAS NOTHING TO FEED — RAM classes ride on a span, never alone. */
    if (spoken == 0u) return ram_said != SYS__SETTINGS__READ_FOUND;   /* no nn key anywhere: no span, and that stands */
    if (spoken != NN__PACKAGE__SPAN_TERMS) return false;   /* half a span is a mistake, not a default */

    if (fullattn != 0ull && per_layer > 0xFFFFFFFFFFFFFFFFull / fullattn) return false;
    uint64_t want = 0ull;
    if (!nn__package__zzprivate_add(&want, buffers))              return false;
    if (!nn__package__zzprivate_add(&want, arena))                return false;
    if (!nn__package__zzprivate_add(&want, per_layer * fullattn)) return false;
    if (!nn__package__zzprivate_add(&want, text))                 return false;
    if (!nn__package__zzprivate_add(&want, deltanet))             return false;
    /* ⚖⚖ AND THE ROOM THE ARENA'S 4 KiB ALIGNMENT MIGHT SPEND — RULED. The carve pulls the
     * arena's base up to a boundary, which consumes 0..4,095 bytes the terms above do not name.
     * ⛔⛔ IT IS A FIXED BOUND AND NOT THE SAME PAD COMPUTED A SECOND TIME, and that is the whole point.
     * Mirroring the device's arithmetic here would be two tiers deriving one figure from one text — the
     * exact shape that let `nn__expert_cache_l1__size_mb` be read in MEBIBYTES on one side and BYTES on
     * the other with every total still plausible. A bound cannot disagree with a computation: the device
     * spends what it needs, this reserves the most it could, and no edit to either can make them differ.
     * ⇒ ★ WHERE TWO TIERS MUST AGREE ABOUT A NUMBER, HAVE ONE OF THEM AGREE TO AN INEQUALITY INSTEAD.
     * ⛳ It costs under 4 KiB of a span measured in gibibytes, and only when a config takes a span at all. */
    if (!nn__package__zzprivate_add(&want, (uint64_t)NN__EXPERT__PAGE_ALIGN - 1ull)) return false;
    /* ⛔ A SPAN OF ZERO IS REFUSED RATHER THAN TAKEN. Every term was named, so somebody meant to size this
     * package and arrived at nothing — which no allocation can improve and no later reader can diagnose. */
    if (want == 0ull) return false;

    if (!sys__gpu__memory_allocate(room->family, &room->at, (size_t)want)) return false;
    room->bytes = want;
    /* ⭐ AND THE RAM SPAN, IN PINNED RAM THE WORKER'S CARD CAN WRITE — taken second, so a refusal here gives
     * the span back and leaves nothing half-taken. */
    if (ram_said == SYS__SETTINGS__READ_FOUND) {
        if (!sys__gpu__memory_allocate_host_ram_mapped(room->family, &room->ram_at, &room->ram_card_at, (size_t)ram)) {
            sys__gpu__memory_free(room->family, room->at);
            room->at = 0; room->bytes = 0ull; room->ram_at = 0; room->ram_card_at = 0;
            return false;
        }
        room->ram_bytes = ram;
    }
    return true;
}

/* What it took, given back. ⛔ AND IT IS CALLED ON TWO PATHS, NOT ONE: `shutdown`, and the unwind of a
 * boot that refused AFTER this package had already taken its span. Both are inside a living process,
 * which is the whole reason this exists — a dead process' VRAM comes back on its own (`MEASURED`). */
static inline void nn__package__host_teardown(sys__package_room* room) {
    nn__loader__zzpackage_stop();                   /* the loader threads read into this room: they go first */
    if (room == 0) return;
    if (room->at != 0) sys__gpu__memory_free(room->family, room->at);
    if (room->ram_at != 0) sys__gpu__memory_free_host_ram(room->family, room->ram_at);
    room->at = 0; room->bytes = 0ull;
    room->ram_at = 0; room->ram_card_at = 0; room->ram_bytes = 0ull;
}

/* ══ ⚠ A FIFTH TENANT NOBODY HAS COSTED — RAISED HERE BECAUSE THIS IS WHERE IT WOULD BE PAID ═══════════
 *
 * `nn_span_layout.md` §⓪ names THREE tenants of the span and §①ⓐ excludes deltanet layers from the
 * cartridge arithmetic BY NAME — *"a `deltanet` layer's state is constant in the sequence length, so it
 * does not enter §④'s arithmetic at all."* That sentence is true about the PER-TOKEN product and does not
 * follow to zero: a deltanet layer's state is constant in the sequence length, not absent.
 *
 * `MEASURED` from the old tree, which allocates one per linear layer (`session/alloc.py:476`, and
 * `types.py:113` says the argument is ELEMENTS — the error message spells `num_elements * 4`):
 *     conv_state(qkv_dim × 3) + recurrent_state(num_v_heads × head_dim²) = 1,114,112 fp32
 *                                                                       = 4,456,448 B  ≈ 4.25 MiB / layer
 *     × 36 deltanet layers (Qwen3.5-MoE: 48 total, 12 full-attention)   ≈ 153 MiB / conversation
 *
 * ⇒ ~7.7% ON TOP OF A ~1.98 GiB CARTRIDGE, and it is per-conversation state that travels with it — so it
 * belongs in the cartridge term or beside it, not nowhere. ⛔ IT IS NOT SUMMED ABOVE, deliberately: no key
 * names it, and inventing one would be sizing a span against a figure no ruling stands behind.
 * ⛳ WHAT IT WOULD COST TO CLOSE: one more paired term of exactly the shape already here — the deltanet
 * layer count, which the config file already carries for the layer-total sum of `nn_span_layout.md` §①ⓐ,
 * multiplied by a per-layer state size the converter states the same way it states a quantised token's.
 * `NN__PACKAGE__SPAN_TERMS` becomes 5 and nothing else moves.
 * ⇒ ★ THE EXCLUSION READS AS SETTLED BECAUSE IT IS WRITTEN AS A REASON, and the reason is sound about the
 * thing it was reasoning about. This is the third figure in this arc to be short for that shape of cause —
 * after the layer count and the quantisation scales — and all three were found by asking what a correct
 * sentence did NOT say. ══════════════════════════════════════════════════════════════════════════════ */

#endif /* SILVANN__PACKAGES_NN_CPU_PACKAGE__IMPL_CUH */
