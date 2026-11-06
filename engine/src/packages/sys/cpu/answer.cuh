#ifndef SILVANN__PACKAGES_SYS_CPU_ANSWER_CUH
#define SILVANN__PACKAGES_SYS_CPU_ANSWER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "fault__header.cuh"                 /* how a refusal is said */
#include "heap_node__header.cuh"             /* the node a value is */
#include "heap_object__header.cuh"           /* holds, the lock in the head, and writing a cell */
#include "heap__header.cuh"                  /* where one is carved */
#include "stack__header.cuh"                 /* how a dying one hands over its value */
#include "silicon/silicon__header.cuh"       /* the state's read and swap */
#include "../contracts/objects/answer.cuh"   /* its layout, its three states and its fault words */
/* ══ AN ANSWER — WHERE A COMPUTATION'S VALUE GOES, MADE BEFORE THE COMPUTATION IS ASKED FOR ══════════════
 *
 * ⚖ ARCHITECT 2026-11-05: *"it becomes (sys_compute cu bindings function result), where result is a heap object that
 * contains the result of the computation (and it starts at waiting) so we dissolve the whole issue. it is outside the
 * bindings table so it does not have the same limitations of not being writeable and it allows for a multicast too"* ·
 * *"several readers of one answer"* · *"sys__result r only tells us if the computing is still waiting or if the result
 * is ready"*.
 *
 * A program asks a block for a value and names where the value goes (`sys__scheduler__add`), so nothing has to find it
 * afterwards: whoever holds the answer reads it. It is a value like any other — bound, passed in a picture's names,
 * handed to another computation — and it lives as long as somebody holds it: the asker, each reader, and the request
 * on its way, which keeps it until the value is filed. An asker that walks away leaves nothing behind.
 *
 *     EMPTY     made, asked of nobody
 *     WAITING   asked: a computation will fill it        `arm`, when the request is queued
 *     READY     filled, the value there for every reader  `file`, by the block that finished the computation
 *
 * ⭐ READING THE STATE TAKES NO LOCK. A reader polls one word of the answer's own head, so nothing anybody else waits
 * for makes it look. The writes take the answer's lock — the arming by its asker, the filing by the block that
 * finished, each once a request — and so does reading the value, because a reader takes a hold of it: the value is
 * replaced only when the answer is asked again, and the hold is taken where the replacing cannot run under it.
 * ⭐ READING DOES NOT TAKE THE VALUE AWAY. Every reader is handed a hold of its own, and the answer keeps the value
 * until it is asked again or let go of — so one answer serves any number of readers.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

static __device__ inline unsigned int* sys__answer__zzprivate_state_word(uint64_t answer) {
    return (unsigned int*)&sys__heap_node__zzpackage_head(answer)->args[SYS__ANSWER__STATE];
}

/* An empty answer. The caller holds it; 0 having raised where the heap had no room. */
static __device__ inline uint64_t sys__answer__create(void) {
    sys__heap_node* head = sys__heap__make(SYS__KIND__ANSWER);
    if (head == 0) return 0ull;                                  /* the heap already raised */
    head->args[SYS__ANSWER__STATE]   = (uint64_t)SYS__ANSWER__EMPTY;
    head->args[SYS__ANSWER__WAITERS] = 0ull;
    head[1] = sys__heap_node__nothing();
    return sys__heap__offset(head + 1);
}

/* Whether this names one at all. */
static __device__ inline bool sys__answer__is(uint64_t answer) {
    if (!sys__heap_object__zzpackage_addressable(answer)) return false;
    return sys__heap_object__is_type_from_head(sys__heap_node__zzpackage_head(answer), SYS__KIND__ANSWER);
}

/* What it says — EMPTY, WAITING or READY — read without the lock, as often as anybody likes. */
static __device__ inline uint32_t sys__answer__state(uint64_t answer) {
    if (!sys__answer__is(answer)) return SYS__ANSWER__EMPTY;
    return (uint32_t)sys__silicon__read_u32(sys__answer__zzprivate_state_word(answer));
}

/* Asked: WAITING, the value of the last asking let go of. Refused BUSY while a computation is still on its way to it —
 * two computations filling one answer would each be the other's lost value. Answers whether it is armed. */
static __device__ inline bool sys__answer__zzpackage_arm(uint64_t answer) {
    if (!sys__answer__is(answer)) { sys__fault__raise(0ull, SYS__ANSWER__FAULT_KIND); return false; }
    if (!sys__heap_object__zzpackage_lock_shared(answer)) return false;
    unsigned int* state = sys__answer__zzprivate_state_word(answer);
    if (sys__silicon__read_u32(state) == SYS__ANSWER__WAITING) {
        sys__heap_object__lock_release(answer);
        sys__fault__raise(0ull, SYS__ANSWER__FAULT_BUSY);
        return false;
    }
    sys__heap_node* cell = sys__heap__object_full_address(answer);
    const sys__heap_node old = *cell;
    *cell = sys__heap_node__nothing();
    sys__heap_node__zzpackage_head(answer)->args[SYS__ANSWER__WAITERS] = 0ull;   /* nobody parked on this asking yet */
    (void)sys__silicon__cas_u32(state, sys__silicon__read_u32(state), SYS__ANSWER__WAITING);   /* EMPTY or READY */
    sys__heap_object__lock_release(answer);
    /* ⛳ THE OLD VALUE IS LET GO OF OUTSIDE THE LOCK: letting go can take apart whatever it held, and that is no work to do
     *   while a reader waits on the lock. */
    if (sys__heap_node__carries_reference(old.dtype) && old.args[0] != 0ull) (void)sys__heap_object__release(old.args[0]);
    return true;
}

/* An arming undone, where the request it was armed for could not be queued: EMPTY again. */
static __device__ inline void sys__answer__zzpackage_disarm(uint64_t answer) {
    if (!sys__answer__is(answer)) return;
    (void)sys__silicon__cas_u32(sys__answer__zzprivate_state_word(answer), SYS__ANSWER__WAITING, SYS__ANSWER__EMPTY);
}

/* Filled: `value` in it, the answer taking a hold of its own, then READY — the value is written before the state says
 * so, and a reader that sees READY sees the value. Refused LATE where nothing was waiting on it. */
static __device__ inline bool sys__answer__zzpackage_file(uint64_t answer, const sys__heap_node* value) {
    if (!sys__answer__is(answer) || value == 0) { sys__fault__raise(0ull, SYS__ANSWER__FAULT_KIND); return false; }
    if (!sys__heap_object__zzpackage_lock_shared(answer)) return false;
    unsigned int* state = sys__answer__zzprivate_state_word(answer);
    if (sys__silicon__read_u32(state) != SYS__ANSWER__WAITING) {
        sys__heap_object__lock_release(answer);
        sys__fault__raise(0ull, SYS__ANSWER__FAULT_LATE);
        return false;
    }
    if (sys__heap_node__carries_reference(value->dtype) && value->args[0] != 0ull)
        (void)sys__heap_object__retain(value->args[0]);
    *sys__heap__object_full_address(answer) = *value;            /* the cell was emptied when it was armed */
    sys__silicon__publish();                                     /* the value out before the state says so */
    (void)sys__silicon__cas_u32(state, SYS__ANSWER__WAITING, SYS__ANSWER__READY);
    sys__heap_object__lock_release(answer);
    return true;
}

/* The value, HELD — the caller gives that hold back — once it is READY; nothing while it is not. The answer keeps its
 * own, so the next reader is handed the same value. */
static __device__ inline sys__heap_node sys__answer__read(uint64_t answer) {
    if (sys__answer__state(answer) != SYS__ANSWER__READY) return sys__heap_node__nothing();
    if (!sys__heap_object__zzpackage_lock_shared(answer)) return sys__heap_node__nothing();
    sys__heap_node value = sys__heap_node__nothing();
    if (sys__silicon__read_u32(sys__answer__zzprivate_state_word(answer)) == SYS__ANSWER__READY) {
        value = *sys__heap__object_full_address(answer);
        if (sys__heap_node__carries_reference(value.dtype) && value.args[0] != 0ull)
            (void)sys__heap_object__retain(value.args[0]);
    }
    sys__heap_object__lock_release(answer);
    return value;
}

/* ── WHO IS WAITING FOR IT ──────────────────────────────────────────────────────────────────────────────
 * ⚖ 2026-11-05 *"if we are waiting on a computation and sys_result still says waiting we can shove the computation at the
 * back of the current thread computation queue and make the other progress... after all we are still waiting on a heap
 * object and that does not change shape"*: a block that parks a step until this answer fills says so here, a bit its
 * own, and the block that files the value rings each one's door (▶ `sys__scheduler__zzengine_post`). The bit is set
 * before the parker's last look at the state, and the state is READY before the filer reads the bits — so either the
 * filer rings, or the parker sees it filled. */
static __device__ inline void sys__answer__zzpackage_wait_on(uint64_t answer, uint32_t block) {
    if (!sys__answer__is(answer) || block >= 64u) return;
    unsigned int* word = (unsigned int*)&sys__heap_node__zzpackage_head(answer)->args[SYS__ANSWER__WAITERS] + (block >> 5);
    const unsigned int bit = 1u << (block & 31u);
    for (;;) {
        const unsigned int was = sys__silicon__read_u32(word);
        if ((was & bit) != 0u || sys__silicon__cas_u32(word, was, was | bit) == was) return;
    }
}

static __device__ inline uint64_t sys__answer__zzpackage_waiters(uint64_t answer) {
    if (!sys__answer__is(answer)) return 0ull;
    const unsigned int* word = (const unsigned int*)&sys__heap_node__zzpackage_head(answer)->args[SYS__ANSWER__WAITERS];
    return (uint64_t)sys__silicon__read_u32(word) | ((uint64_t)sys__silicon__read_u32(word + 1) << 32);
}

/* ── THE ROW METHODS ─────────────────────────────────────────────────────────────────────────────────── */

/* What a dying answer hands over: its value, onto the chain the release loop drains. */
static __device__ SYS__NOINLINE void sys__answer__zzpackage_release_internal(sys__heap_node* head,
                                                                           uint64_t* releaser_stack) {
    if (head == 0) return;
    if (sys__heap_node__carries_reference(head[1].dtype) && head[1].args[0] != 0ull)
        sys__stack__transfer(releaser_stack, &head[1]);
    head[1] = sys__heap_node__nothing();
}

/* Building one from a program: it takes nothing and arrives EMPTY. */
static __device__ inline sys__heap_node sys__answer__zzpackage_construct(sys__heap_node* base,
                                                                        const sys__heap_node* parameters) {
    (void)base; (void)parameters;
    const uint64_t at = sys__answer__create();
    if (at == 0ull) return sys__heap_node__nothing();            /* the making already raised */
    return sys__heap_object__reference_to(at);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_ANSWER_CUH */
