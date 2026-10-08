#ifndef SILVANN__PACKAGES_NN_CPU_PRIMITIVES__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_PRIMITIVES__IMPL_CUH
/* What this file needs, named where a reader — and an editor — can follow it. */

#include "primitives__header.cuh"        /* the contract these definitions answer */
#include "buffer__header.cuh"            /* what a buffer's node holds, and its kind */
#include "../../sys/cpu/list__header.cuh"       /* a form is a list, and the verb reads its arguments */
#include "../../sys/cpu/heap__header.cuh"       /* an offset to a place */
#include "../../sys/cpu/heap_object__header.cuh" /* asking what kind a reference names */
#include "../../sys/cpu/silicon/silicon__header.cuh"    /* the two arithmetic verbs the seam publishes */
#include "../../sys/cpu/opcodes/opcodes.cuh"            /* how a verb answers, and how it refuses */
#include "../../sys/cpu/node_array__header.cuh"         /* a plane table is a node array */
#include "../contracts/objects/primitives.cuh" /* its constants and fault words */
/* ══ nn — THE PRIMITIVES ═════════════════════════════════════════════════════════════════════════════
 * ▶ `primitives__header.cuh` for why these three, and why the resolver is shared.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */


static __device__ inline bool nn__primitives__room(const sys__heap_node* ref,
                                                    uint64_t* at, uint64_t* bytes) {
    /* ⭐ ONLY THIS WORKER'S — ⚖ *"build owner check"*. A buffer of another worker names memory this worker's device
     *   may not reach at all: on node03 one MI50's kernel given another's address is a page fault that ends the
     *   process. Refused here, the one place every verb asks, with a word of its own. */
    const uint64_t owner = nn__primitives__owner(ref);
    if (owner != 0ull && owner != (uint64_t)sys__silicon__worker() + 1ull) {
        sys__fault__raise(0ull, NN__BUFFER__FAULT_OWNER);
        return false;
    }
    return nn__primitives__room_of_any(ref, at, bytes);
}

/* The owning worker plus one, of a buffer or a weights view — 0 for anything else, or one nobody tagged. */
static __device__ inline uint64_t nn__primitives__owner(const sys__heap_node* ref) {
    if (ref->dtype != SYS__KIND__OBJECT_REFERENCE) return 0ull;
    sys__heap_node* object = sys__heap__object_full_address(ref->args[0]);
    if (object == 0) return 0ull;
    if (sys__heap_object__is_type(object, NN__KIND__BUFFER))  return object->args[NN__BUFFER__OWNER];
    if (sys__heap_object__is_type(object, NN__KIND__WEIGHTS)) return object->args[NN__WEIGHTS__OWNER];
    return 0ull;
}

static __device__ inline bool nn__primitives__room_of_any(const sys__heap_node* ref, uint64_t* at, uint64_t* bytes) {
    if (ref->dtype != SYS__KIND__OBJECT_REFERENCE) return false;
    sys__heap_node* object = sys__heap__object_full_address(ref->args[0]);
    if (object == 0) return false;
    /* ⭐⭐ TWO KINDS ANSWER "WHERE, AND HOW MUCH", AND THIS IS THE ONLY PLACE THAT ASKS. ⚖ RULED
     * — weights resident in a page reach the arithmetic through a kind of their own, and
     * teaching THIS FUNCTION about it is the whole of the change: EVERY compute verb in the package
     * asks its operands' address and length here and nowhere else, so a second arm here gives all of
     * them pages at once. ⛔ NO COUNT HERE — this file gains verbs, so a number would rot, and the one
     * that stood here said "nineteen" and was neither of the two figures a re-derivation produces.
     * ⛳ RE-DERIVE:  grep -c 'nn__primitives__room(' packages/nn/*.cuh
     * ⇒ ★ A CHOKEPOINT IS WORTH MORE THAN A CONVENTION — the thing that made this cheap is that nobody
     * ever read a buffer's address directly, which was a discipline until the moment it became leverage.
     * ⛳ AND THE ASYMMETRY IS DELIBERATE: a BUFFER derives its length from its class, because the pool
     * decided that; a WEIGHTS reference carries its own, because the slot it names holds several planes
     * and its length is a fact about the plane rather than about the room. */
    if (sys__heap_object__is_type(object, NN__KIND__BUFFER)) {
        *at    = object->args[NN__BUFFER__AT];
        *bytes = nn__buffer__sizeof(object->args[NN__BUFFER__CLASS]);
    } else if (sys__heap_object__is_type(object, NN__KIND__WEIGHTS)) {
        *at    = object->args[NN__WEIGHTS__AT];
        *bytes = object->args[NN__WEIGHTS__BYTES];
    } else {
        return false;
    }
    return *at != 0ull && *bytes != 0ull;
}

/* The other two words a value verb and `nn__buffer__read` need, from the same two kinds `room()` knows:
 * where the program sees the bytes (0 when only the card can), and the word holding the stamp the next
 * value must carry. Answers 0 for anything that is neither kind. ▶ `contracts/objects/result.cuh`. */
static __device__ inline uint64_t* nn__primitives__result_slot(const sys__heap_node* ref, uint64_t* host_at) {
    *host_at = 0ull;
    if (ref->dtype != SYS__KIND__OBJECT_REFERENCE) return 0;
    sys__heap_node* object = sys__heap__object_full_address(ref->args[0]);
    if (object == 0) return 0;
    if (sys__heap_object__is_type(object, NN__KIND__BUFFER)) {
        *host_at = object->args[NN__BUFFER__HOST_AT];
        return &object->args[NN__BUFFER__STAMP];
    }
    if (sys__heap_object__is_type(object, NN__KIND__WEIGHTS)) {
        *host_at = object->args[NN__WEIGHTS__HOST_AT];
        return &object->args[NN__WEIGHTS__STAMP];
    }
    return 0;
}

static __device__ inline bool nn__primitives__fits(uint64_t n, uint64_t room) {
    if (n != 0ull && n > 0xFFFFFFFFFFFFFFFFull / NN__PRIMITIVES__ELEMENT_BYTES) return false;
    return n * NN__PRIMITIVES__ELEMENT_BYTES <= room;
}

static bool nn__primitives__table(uint64_t table, unsigned planes, unsigned reals, unsigned length,
                                  uint64_t* at, uint64_t* room, uint64_t* v, float* r) {
    if (sys__node_array__length(table) < length) return false;
    sys__node_array_walk w;
    if (!sys__node_array__walk(table, 0ull, &w)) return false;
    for (unsigned i = 0u; i < length; ++i, sys__node_array__next(&w)) {
        const sys__heap_node* n = sys__node_array__walk_cell(&w);
        if (n == 0) return false;
        if (i < planes) { if (!nn__primitives__room(n, &at[i], &room[i])) return false; }
        else if (i < reals) { if (n->dtype != SYS__KIND__VALUE_INT) return false; v[i] = n->args[0]; }
        else if (n->dtype != SYS__KIND__VALUE_FLOAT) return false;
        else r[i] = (float)sys__heap_node__real(n->args[0]);
    }
    return true;
}

static bool nn__primitives__table_flag(uint64_t table, unsigned index) {
    if (sys__node_array__length(table) <= index) return false;
    const sys__heap_node n = sys__node_array__borrow(table, index);
    return n.dtype == SYS__KIND__VALUE_INT && n.args[0] == 1ull;
}
















/* ══ ⭐ THE GATE CHAIN — ▶ `primitives__header` for the transcription this came from ════════════════ */




/* ══ ⭐⭐ READING A BUFFER BY INDEX ═════════════════════════════════════════════════════════════════ */


/* ── `(nn__vector__get_values_at x positions)` -> a list of floats ──────────────────────────────────
 * ⛔⛔ THE POSITIONS ARE WALKED TWICE — ONCE TO CHECK, ONCE TO GATHER — AND THAT IS DELIBERATE. Building
 * the answer as it went would leave a half-filled list to take apart on the first bad position, and a
 * refusal that has already allocated is a refusal with a leak behind it. ⇒ ★ VALIDATE THE WHOLE INPUT
 * BEFORE TAKING ANYTHING FROM THE HEAP, on any path that can still refuse. */
/* One element of `x` as the gather reads it: from the buffer's RAM view when it has one, and otherwise
 * through the worker's card.
 * ⛳ A CARD'S ADDRESS IS NEVER READ FROM THE CPU: it goes into a door or a card transfer, and the only
 * memory read here directly is a buffer's RAM view. Some hosts map a card's memory and a stray read
 * works there, slowly, so nothing at run time would say it happened. */
static inline uint16_t nn__vector__zzprivate_element(uint64_t x_at, uint64_t i, uint64_t host_at,
                                                     sys__silicon_family__id family) {
    if (host_at != 0ull) return ((const uint16_t*)(uintptr_t)host_at)[i];
    uint16_t h = 0x7E00u;                                    /* a NaN, if the card will not answer */
    (void)sys__gpu__memory_read(family, &h, (const void*)(uintptr_t)(x_at + i * sizeof h), sizeof h);
    return h;
}

static __device__ __noinline__ void nn__vector__zzpackage_apply_get_values_at(sys__heap_node* base,
                                                                              uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) != 3ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const sys__heap_node xr = sys__sublist__nth(form, 1ull);
    const sys__heap_node pr = sys__sublist__nth(form, 2ull);

    /* ⭐⭐ BOTH LIST TAGS ARE ACCEPTED, AND THAT IS THE VERB'S CONTRACT RATHER THAN A LENIENCY. The
     * positions arrive one of exactly two ways and this verb is the meeting point of both:
     * ```
     *   QUOTED_LIST       a literal the program wrote   — `(nn__vector__get_values_at v (3 0 7))`
     *   OBJECT_REFERENCE  a list something COMPUTED     — `(nn__vector__get_values_at v (top_k ...))`
     * ```
     * ⛳ THE SECOND IS THE ONE THAT MATTERS: `top_k` answers a list of names, and refusing its tag here
     * would mean the two verbs ruled together on the same day could not be composed. ⛔ AND THE PAYLOAD
     * IS THE SAME OBJECT EITHER WAY — only the tag differs, so `sublist__is` is what actually decides
     * whether this is a list, and it is asked in both cases. ⇒ ★ A TAG SAYS HOW A VALUE GOT HERE; IT IS
     * NOT ALWAYS A CLAIM ABOUT WHAT THE VALUE IS. */
    uint64_t x_at = 0ull, x_room = 0ull;
    if ((pr.dtype != SYS__KIND__OBJECT_REFERENCE && pr.dtype != SYS__KIND__QUOTED_LIST)
        || !sys__sublist__is(pr.args[0])
        || !nn__primitives__room(&xr, &x_at, &x_room)) {
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
        return;
    }
    const uint64_t positions = pr.args[0];
    const uint64_t k = sys__sublist__length(positions);

    for (uint64_t j = 0ull; j < k; ++j) {
        const sys__heap_node p = sys__sublist__nth(positions, j);
        if (p.dtype != SYS__KIND__VALUE_INT) {
            sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
            return;
        }
        if (p.args[0] == 0xFFFFFFFFFFFFFFFFull || !nn__primitives__fits(p.args[0] + 1ull, x_room)) {
            sys__opcodes__fails(form, NN__PRIMITIVES__FAULT_BOUNDS);
            return;
        }
    }

    /* ⭐ ON THE CPU THE ELEMENTS ARE ON THE WORKER'S CARD, SO THEY ARE READ THROUGH IT — after waiting for
     * whatever wrote them — and never dereferenced from here. A buffer in RAM is read where it is. */
    const sys__engine__ctx* ctx = sys__engine__ctx_current();
    uint64_t host_at = 0ull;
    (void)nn__primitives__result_slot(&xr, &host_at);
    if (ctx == 0 || ctx->family == SYS__SILICON_FAMILY__NONE || !sys__gpu__compute_completed(ctx->family)) {
        sys__opcodes__fails(form, NN__PRIMITIVES__FAULT_NO_DEVICE);
        return;
    }
    const uint64_t answer = sys__list__create();
    if (answer == 0ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE); return; }

    bool ok = true;
    for (uint64_t j = 0ull; ok && j < k; ++j) {
        const sys__heap_node p = sys__sublist__nth(positions, j);
        sys__heap_node cell = sys__heap_node__nothing();
        cell.dtype = SYS__KIND__VALUE_FLOAT; cell.num_args = 0u; cell.op_code = 0ull;
        cell.args[0] = sys__heap_node__real_bits(
                           (double)nn__primitives__zzpackage_half_to_float(nn__vector__zzprivate_element(x_at, p.args[0], host_at, ctx->family)));
        ok = sys__sublist__append(answer, &cell);
    }
    if (!ok) {
        (void)sys__heap_object__release(answer);
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
        return;
    }

    /* ⛔⛔⛔ THE ANSWER IS TAGGED `QUOTED_LIST`, AND AN `OBJECT_REFERENCE` HERE IS NOT A STYLE CHOICE —
     * IT IS A VERB THAT DESTROYS ITS OWN ANSWER. `becomes` collapses the form to its one value; the
     * evaluator then reaches `eng__eval__zzprivate_first_form`, whose test for *"something still to
     * evaluate"* is exactly **an `OBJECT_REFERENCE` that names a sublist**. A list answered under that
     * tag satisfies it, so the scan DESCENDS INTO THE ANSWER and applies its first element as a verb.
     * `MEASURED`: `(nn__vector__get_values_at buf '(3 0))` answered `ENG__EVAL__FAULT_NOT_A_VERB`
     * — the evaluator trying to call `3`.
     * ⛳ `eval.cuh:82` STATES THE RULE FROM THE OTHER SIDE — *"a quoted list is `QUOTED_LIST`, so both
     * arms of an `if` sit untouched … which is why no verb needs marking to keep the scanner out of its
     * arguments"*. The same tag that keeps the scanner out of an ARGUMENT is what keeps it out of an
     * ANSWER, and nothing until now had answered with a list for that sentence to apply to.
     * ⇒ ★★ **A LIST IS THE ONE VALUE THAT IS ALSO A PROGRAM, SO RETURNING ONE IS A STATEMENT ABOUT
     * WHETHER IT SHOULD RUN.** Every other kind answers the same under either tag and the choice never
     * came up. ⛔ THIS BINDS `top_k` TOO, and for the same reason — ⚖ it was ruled on the same day to
     * answer *"the list of ints as nodes for the lisp"*, which walks into precisely this.
     * ⛳ AND THE HOLD IS UNCHANGED: `QUOTED_LIST` is one of the reference kinds
     * `sys__heap_node__carries_reference` names, so `becomes` takes its hold exactly as it would have. */
    sys__heap_node out = sys__heap_node__nothing();
    out.dtype = SYS__KIND__QUOTED_LIST; out.num_args = 0u; out.op_code = 0ull;
    out.args[0] = answer;
    sys__opcodes__becomes(form, &out);
    /* ⛳ `becomes` TOOK ITS OWN HOLD, so the creator's goes back here. ▶ the buffer-hold discipline: a
     * verb that hands an object to the form and keeps its hold has leaked one, and the pool recycles
     * instantly so the symptom lands somewhere else entirely. */
    (void)sys__heap_object__release(answer);
}

/* ── ⭐⭐⭐ `(nn__vector__range v start count)` -> an `nn__weights` over the buffer's own bytes ─────── */
static __device__ __noinline__ void nn__vector__zzpackage_apply_range(sys__heap_node* base,
                                                                      uint64_t form) {
    (void)base;
    if (sys__sublist__length(form) != 4ull) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_ARITY); return; }
    const sys__heap_node vr = sys__sublist__nth(form, 1ull);
    const sys__heap_node sc = sys__sublist__nth(form, 2ull);
    const sys__heap_node cc = sys__sublist__nth(form, 3ull);

    uint64_t v_at = 0ull, v_room = 0ull;
    if (sc.dtype != SYS__KIND__VALUE_INT || cc.dtype != SYS__KIND__VALUE_INT
        || !nn__primitives__room(&vr, &v_at, &v_room)) {
        sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE);
        return;
    }
    /* ⛳ A VIEW OF A VIEW IS ORDINARY AND COSTS NOTHING TO ALLOW: `room()` answers for an `nn__weights`
     * exactly as it does for a buffer, so the span below is checked against whatever this one names —
     * which for a view is already the narrowed span. Nesting therefore cannot widen. */
    const uint64_t start = sc.args[0], count = cc.args[0];

    /* ⛔⛔ THE SPAN IS CHECKED IN ELEMENTS AGAINST THE ROOM IN BYTES, AND BOTH COMPARISONS ARE AGAINST A
     * LENGTH RATHER THAN A COMPUTED ADDRESS — the same shape `plane` and the arena carve use, and for
     * the same reason: an address that has already wrapped compares as being inside. */
    if (count == 0ull) { sys__opcodes__fails(form, NN__PRIMITIVES__FAULT_BOUNDS); return; }
    if (!nn__primitives__fits(start, v_room)) { sys__opcodes__fails(form, NN__PRIMITIVES__FAULT_BOUNDS); return; }
    const uint64_t at_byte = start * NN__PRIMITIVES__ELEMENT_BYTES;
    if (!nn__primitives__fits(count, v_room - at_byte)) {
        sys__opcodes__fails(form, NN__PRIMITIVES__FAULT_BOUNDS);
        return;
    }

    sys__heap_node* made = sys__heap__make(NN__KIND__WEIGHTS);
    if (made == 0) { sys__opcodes__fails(form, SYS__OPCODES__FAULT_TYPE); return; }
    /* ⛔ THE BODY IS WRITTEN WHOLE, NOT ONLY THE TWO FIELDS THIS KIND USES — `sys__heap__make` writes the
     * head and leaves the body as whatever the chunk held, so setting two words would publish four of
     * somebody else's freed object. ▶ `nn__expert__plane`, which says the same thing at more length. */
    made[1] = sys__heap_node__nothing();
    made[1].dtype = SYS__KIND__VALUE_INT; made[1].num_args = 0u; made[1].op_code = 0ull;
    made[1].args[NN__WEIGHTS__AT]    = v_at + at_byte;
    made[1].args[NN__WEIGHTS__BYTES] = count * NN__PRIMITIVES__ELEMENT_BYTES;
    made[1].args[NN__WEIGHTS__OWNER] = nn__primitives__owner(&vr);        /* a view is its source's worker's */
    /* ⭐⭐⭐ THE VIEW TAKES A HOLD ON WHAT IT LOOKS AT — ⚖ RULED: *"the base method should
     * hold and it is the developer responsibility to do a nohold only if it is sensible."* One hold,
     * taken once, given back by this kind's release hook when the last reference to the VIEW goes.
     * ⛳ A VIEW OF A VIEW HOLDS THE VIEW IT CAME FROM and not the buffer underneath; the release loop
     * drains the chain, so the buffer's hold goes when the last link does. ⚖ *"other references to the
     * view object are held by the view object itself."*
     * ⇒ ★★ THE DANGEROUS VERSION IS NOW THE ONE SOMEBODY WOULD HAVE TO ASK FOR, which is the whole of
     * the ruling: a borrowed span is a real optimisation and an unsafe default. */
    (void)sys__heap_object__retain(vr.args[0]);
    made[1].args[NN__WEIGHTS__HELD]  = vr.args[0];
    /* A cut of RAM the program can see is still RAM the program can see, at the same offset. */
    uint64_t host_at = 0ull;
    (void)nn__primitives__result_slot(&vr, &host_at);
    made[1].args[NN__WEIGHTS__HOST_AT] = host_at != 0ull ? host_at + at_byte : 0ull;
    made[1].args[NN__WEIGHTS__STAMP]   = 0ull;

    const uint64_t where = sys__heap__offset(made + 1);
    const sys__heap_node ref = sys__heap_object__reference_to(where);
    sys__opcodes__becomes(form, &ref);
    (void)sys__heap_object__release(where);
}

#endif /* SILVANN__PACKAGES_NN_CPU_PRIMITIVES__IMPL_CUH */
