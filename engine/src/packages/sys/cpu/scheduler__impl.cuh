#ifndef SILVANN__PACKAGES_SYS_CPU_SCHEDULER__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_SCHEDULER__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "scheduler__header.cuh"         /* what it fills in */
#include "system_register__header.cuh"   /* the queues' row, and the computing bases' row that sizes it */
#include "node_array__header.cuh"        /* the row's array */
#include "list__header.cuh"              /* a queue, shared */
#include "answer.cuh"                    /* where a request's value goes */
#include "computing_base__header.cuh"    /* the base a request is run in, and its send-to */
#include "heap_object__header.cuh"       /* holds */
#include "fault__header.cuh"             /* how a refusal is said */
#include "error.cuh"                     /* an answer that could not be computed, as a value */

/* The row's array: a slot a block, an empty list in each slot, or 0 having let go of what it made. ⛳ NAMED BY
 * OFFSET, as the computing bases' row is: the array is made once and lives as long as the heap, its one hold kept
 * for good, so reading a block's list is two words and no count — the hottest reader is every idle block, every
 * turn of its loop. */
static __device__ inline uint64_t sys__scheduler__zzprivate_row_array(uint64_t blocks) {
    const uint64_t array = sys__node_array__create(blocks);
    if (array == 0ull) return 0ull;
    for (uint64_t b = 0ull; b < blocks; ++b) {
        const uint64_t list = sys__list__create();
        if (list == 0ull) { (void)sys__heap_object__release(array); return 0ull; }
        const sys__heap_node held = sys__heap_object__reference_to(list);
        const bool placed = sys__node_array__set(array, b, &held);
        (void)sys__heap_object__release(list);                 /* the array holds it now, or nothing does */
        if (!placed) { (void)sys__heap_object__release(array); return 0ull; }
    }
    return array;
}

/* ⭐ A BLOCK'S DOOR — ⚖ ARCHITECT *"a semaphore will do"*: a block that has had nothing to do for a while dozes on its
 * own condition (`zzengine_doze`), and whoever queues a request for it rings it (`zzprivate_ring`) — and so does the
 * block that fills an answer a step of its is parked on — so an idle block keeps no core and wakes when there is work. `asleep` says whether to ring: the queuer looks at it once it has
 * queued, behind a fence, and the dozer sets it before its last look at the queue, under the door's lock — so either
 * the queuer sees it asleep and rings, or the dozer sees the request; the lock keeps the ring from passing a doze not
 * yet begun. The wait is bounded, so a block that missed a ring looks again. */
typedef struct {
    unsigned int       asleep;
    sys__os__lock      lock;
    sys__os__condition bell;
} sys__scheduler_door;
static sys__scheduler_door sys__scheduler__zzprivate_doors[SYS__SCHEDULER__DOORS];

static __device__ inline void sys__scheduler__zzprivate_ring(uint32_t cu) {
    if (cu >= SYS__SCHEDULER__DOORS) return;
    sys__scheduler_door* door = &sys__scheduler__zzprivate_doors[cu];
    sys__silicon__publish();                                   /* the request out before the look at the door */
    if (sys__silicon__read_u32(&door->asleep) == 0u) return;
    sys__os__lock_take(&door->lock);
    sys__os__condition_wake_all(&door->bell);
    sys__os__lock_give(&door->lock);
}

/* Whether block `me` has a parked step whose gate has filled (▶ the parked steps, below). */
static __device__ inline bool sys__scheduler__zzprivate_parked_ready(uint32_t me);

static __device__ inline void sys__scheduler__zzengine_doze(uint32_t me, uint64_t ns) {
    if (me >= SYS__SCHEDULER__DOORS) {                        /* no door: a short nap, and a look again */
        const struct timespec nap = { 0, 100000L };
        sys__os__sleep(&nap);
        return;
    }
    sys__scheduler_door* door = &sys__scheduler__zzprivate_doors[me];
    sys__os__lock_take(&door->lock);
    (void)sys__silicon__cas_u32(&door->asleep, 0u, 1u);        /* asleep before the last look, a fence on its own */
    const uint64_t queue = sys__scheduler__queue(me);
    if (queue != 0ull && !sys__list__shared_waiting(queue) && !sys__scheduler__zzprivate_parked_ready(me))
        sys__os__condition_wait_for(&door->bell, &door->lock, ns);
    (void)sys__silicon__cas_u32(&door->asleep, 1u, 0u);
    sys__os__lock_give(&door->lock);
}

static __device__ inline void sys__scheduler__zzengine_ring_all(uint32_t blocks) {
    for (uint32_t b = 0u; b < blocks && b < SYS__SCHEDULER__DOORS; ++b) {
        sys__os__lock_take(&sys__scheduler__zzprivate_doors[b].lock);
        sys__os__condition_wake_all(&sys__scheduler__zzprivate_doors[b].bell);
        sys__os__lock_give(&sys__scheduler__zzprivate_doors[b].lock);
    }
}

static __device__ SYS__NOINLINE bool sys__scheduler__zzpackage_stand_up(void) {
    for (uint32_t b = 0u; b < SYS__SCHEDULER__DOORS; ++b) {
        sys__scheduler__zzprivate_doors[b].asleep = 0u;
        sys__os__lock_init(&sys__scheduler__zzprivate_doors[b].lock);
        sys__os__condition_init(&sys__scheduler__zzprivate_doors[b].bell);
    }
    /* As many as there are computing bases: a block is a base, and the host's slot after them one whose queue nothing
     * serves. */
    const sys__heap_node bases = sys__system_register__get(SYS__SYSTEM_REGISTER__COMPUTING_BASE_ARRAY);
    const uint64_t blocks = (bases.dtype == SYS__KIND__VALUE_INT && bases.args[0] != 0ull)
                          ? sys__node_array__length(bases.args[0]) : 0ull;
    const uint64_t queues = blocks != 0ull ? sys__scheduler__zzprivate_row_array(blocks) : 0ull;
    const bool up = queues != 0ull
                 && sys__system_register__set(SYS__SYSTEM_REGISTER__SCHEDULER_QUEUE_ARRAY, SYS__KIND__VALUE_INT, queues);
    if (!up) {
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_UP);
        return false;
    }
    return true;
}

/* A block's list out of a row the caller read: the slot, borrowed, or 0. */
static __device__ inline uint64_t sys__scheduler__zzprivate_list(sys__heap_node where, uint32_t block) {
    if (where.dtype != SYS__KIND__VALUE_INT || where.args[0] == 0ull) return 0ull;
    if ((uint64_t)block >= sys__node_array__length(where.args[0])) return 0ull;
    const sys__heap_node slot = sys__node_array__borrow(where.args[0], (uint64_t)block);
    return slot.dtype == SYS__KIND__OBJECT_REFERENCE ? slot.args[0] : 0ull;
}

static __device__ inline uint64_t sys__scheduler__queue(uint32_t block) {
    return sys__scheduler__zzprivate_list(sys__system_register__get(SYS__SYSTEM_REGISTER__SCHEDULER_QUEUE_ARRAY), block);
}

/* ── WHAT A COMPUTATION ASKED, BY BLOCK ──────────────────────────────────────────────────────────────────
 * A row a block, a slot a block it may ask: the answer its running computation last asked of that block, held. One
 * computation runs on a block at a time and only its runner touches its row, so the row is that computation's and
 * takes no lock; it is let go of when the computation ends. A block past the row's width keeps no record — its
 * program names its answers. */
#define SYS__SCHEDULER__ASKED_BLOCKS SYS__SCHEDULER__DOORS
static uint64_t sys__scheduler__zzprivate_asked[SYS__SCHEDULER__ASKED_BLOCKS][SYS__SCHEDULER__ASKED_BLOCKS];

static __device__ inline uint64_t sys__scheduler__zzpackage_asked(uint32_t me, uint32_t cu) {
    if (me >= SYS__SCHEDULER__ASKED_BLOCKS || cu >= SYS__SCHEDULER__ASKED_BLOCKS) return 0ull;
    return sys__scheduler__zzprivate_asked[me][cu];
}

static __device__ inline void sys__scheduler__zzpackage_ask(uint32_t me, uint32_t cu, uint64_t answer) {
    if (me >= SYS__SCHEDULER__ASKED_BLOCKS || cu >= SYS__SCHEDULER__ASKED_BLOCKS) return;
    const uint64_t was = sys__scheduler__zzprivate_asked[me][cu];
    if (was == answer) return;
    if (answer != 0ull) (void)sys__heap_object__retain(answer);
    sys__scheduler__zzprivate_asked[me][cu] = answer;
    if (was != 0ull) (void)sys__heap_object__release(was);
}

static __device__ inline void sys__scheduler__zzengine_forget_asked(uint32_t me) {
    if (me >= SYS__SCHEDULER__ASKED_BLOCKS) return;
    for (uint32_t cu = 0u; cu < SYS__SCHEDULER__ASKED_BLOCKS; ++cu) {
        const uint64_t was = sys__scheduler__zzprivate_asked[me][cu];
        if (was == 0ull) continue;
        sys__scheduler__zzprivate_asked[me][cu] = 0ull;
        (void)sys__heap_object__release(was);
    }
}

/* ── STEPS PARKED UNTIL THEIR GATE FILLS ───────────────────────────────────────────────────────────────────
 * ⚖ 2026-11-05 *"if we are waiting on a computation and sys_result still says waiting we can shove the computation at the
 * back of the current thread computation queue and make the other progress... after all we are still waiting on a heap
 * object and that does not change shape"*. A step queued with a gate (`sys__scheduler__continue cu names program
 * '(ra rb)`) runs once every answer its gate names is READY. A block that takes one before then parks it here — a row a block, its runner's alone,
 * so it takes no lock — and goes on to what is queued behind it; each turn it looks at its parked steps' gates, one
 * word each, and runs the oldest that filled before it takes from its queue again. The block that fills the answer rings
 * the parked block's door (▶ `answer.cuh`, `sys__answer__zzpackage_wait_on`), so a block with nothing else to do sleeps.
 * A block past the row's width parks nothing: its gated step goes back on its queue, behind what came after it.
 * `zzprivate_gate`: the gate of the step a block is running, held until it is put away — its names bound in the step's
 * environment meanwhile (`zzengine_bind_gate`). */
static sys__heap_node sys__scheduler__zzprivate_parked[SYS__SCHEDULER__ASKED_BLOCKS][SYS__SCHEDULER__PARKED][SYS__SCHEDULER__REQUEST_CELLS];
static uint32_t sys__scheduler__zzprivate_parked_n[SYS__SCHEDULER__ASKED_BLOCKS];
static uint64_t sys__scheduler__zzprivate_gate[SYS__SCHEDULER__ASKED_BLOCKS];

/* A gate's answers: an array of pairs, the name at 2i and the answer at 2i + 1 (▶ `sys__scheduler__continue`). */
static __device__ inline uint64_t sys__scheduler__zzprivate_gate_pairs(uint64_t gate) {
    return (gate != 0ull && sys__node_array__is(gate)) ? sys__node_array__length(gate) / 2ull : 0ull;
}

static __device__ inline uint64_t sys__scheduler__zzprivate_gate_answer(uint64_t gate, uint64_t i) {
    const sys__heap_node a = sys__node_array__borrow(gate, 2ull * i + 1ull);
    return sys__heap_node__carries_reference(a.dtype) ? a.args[0] : 0ull;
}

/* Whether a request may run now: no gate, or every answer its gate names READY. */
static __device__ inline bool sys__scheduler__zzprivate_open(const sys__heap_node* cells) {
    const sys__heap_node gate = cells[SYS__SCHEDULER__REQUEST_GATE];
    if (!sys__heap_node__carries_reference(gate.dtype) || gate.args[0] == 0ull) return true;
    const uint64_t n = sys__scheduler__zzprivate_gate_pairs(gate.args[0]);
    for (uint64_t i = 0ull; i < n; ++i)
        if (sys__answer__state(sys__scheduler__zzprivate_gate_answer(gate.args[0], i)) != SYS__ANSWER__READY) return false;
    return true;
}

/* The first of block `me`'s parked steps whose gate filled — the oldest — or how many it has parked. */
static __device__ inline uint32_t sys__scheduler__zzprivate_first_open(uint32_t me) {
    if (me >= SYS__SCHEDULER__ASKED_BLOCKS) return 0u;
    const uint32_t n = sys__scheduler__zzprivate_parked_n[me];
    for (uint32_t i = 0u; i < n; ++i)
        if (sys__scheduler__zzprivate_open(sys__scheduler__zzprivate_parked[me][i])) return i;
    return n;
}

static __device__ inline bool sys__scheduler__zzprivate_parked_ready(uint32_t me) {
    return me < SYS__SCHEDULER__ASKED_BLOCKS
        && sys__scheduler__zzprivate_first_open(me) < sys__scheduler__zzprivate_parked_n[me];
}

static __device__ inline uint32_t sys__scheduler__zzengine_parked(uint32_t me) {
    return me < SYS__SCHEDULER__ASKED_BLOCKS ? sys__scheduler__zzprivate_parked_n[me] : 0u;
}

/* A taken request parked, its holds with it, and its gate told this block waits — or false, with no room. */
static __device__ inline bool sys__scheduler__zzprivate_park(uint32_t me, const sys__heap_node* cells) {
    if (me >= SYS__SCHEDULER__ASKED_BLOCKS || sys__scheduler__zzprivate_parked_n[me] >= SYS__SCHEDULER__PARKED) return false;
    sys__heap_node* at = sys__scheduler__zzprivate_parked[me][sys__scheduler__zzprivate_parked_n[me]];
    for (uint32_t k = 0u; k < SYS__SCHEDULER__REQUEST_CELLS; ++k) at[k] = cells[k];
    sys__scheduler__zzprivate_parked_n[me] += 1u;
    const uint64_t gate = cells[SYS__SCHEDULER__REQUEST_GATE].args[0];
    const uint64_t n = sys__scheduler__zzprivate_gate_pairs(gate);
    for (uint64_t i = 0ull; i < n; ++i) sys__answer__zzpackage_wait_on(sys__scheduler__zzprivate_gate_answer(gate, i), me);
    return true;
}

/* Parked step `i` taken out into `out`, its holds with it, the ones after it moved up. */
static __device__ inline void sys__scheduler__zzprivate_unpark(uint32_t me, uint32_t i, sys__heap_node* out) {
    sys__heap_node (*row)[SYS__SCHEDULER__REQUEST_CELLS] = sys__scheduler__zzprivate_parked[me];
    const uint32_t n = sys__scheduler__zzprivate_parked_n[me];
    for (uint32_t k = 0u; k < SYS__SCHEDULER__REQUEST_CELLS; ++k) out[k] = row[i][k];
    for (uint32_t j = i + 1u; j < n; ++j)
        for (uint32_t k = 0u; k < SYS__SCHEDULER__REQUEST_CELLS; ++k) row[j - 1u][k] = row[j][k];
    sys__scheduler__zzprivate_parked_n[me] = n - 1u;
}

/* What a request's cells hold, let go of. */
static __device__ inline void sys__scheduler__zzprivate_let_go(const sys__heap_node* cells) {
    for (uint32_t k = 0u; k < SYS__SCHEDULER__REQUEST_CELLS; ++k)
        if (sys__heap_node__carries_reference(cells[k].dtype) && cells[k].args[0] != 0ull)
            (void)sys__heap_object__release(cells[k].args[0]);
}

static __device__ inline uint64_t sys__scheduler__zzpackage_gate(uint32_t me) {
    return me < SYS__SCHEDULER__ASKED_BLOCKS ? sys__scheduler__zzprivate_gate[me] : 0ull;
}

/* ⭐ THE NAMES A STEP WAITED FOR, BOUND IN ITS ENVIRONMENT TO WHAT THEY CAME TO (⚖ 2026-11-05 *"names also contains the
 * references to ra and rb as bindings in this way, it is the environment both for ra and rb and for second_half"* ·
 * *"a computation replaces the computation result binding with a set to the new result, by removing its hold on the
 * computation pointer and adding one to the result of the computation ... so the let can be set! to the final correct
 * value"*): once `begin` has built the step's environment, each name of its gate is bound there to its answer's VALUE —
 * the result itself, one jump from the name and not two — a scope of its own holding it, left again once the
 * evaluation ends, before `finish` lets the environment go. The answer is the gate's to hold until the step is put
 * away. */
static bool sys__scheduler__zzprivate_gate_bound[SYS__SCHEDULER__ASKED_BLOCKS];

static __device__ inline void sys__scheduler__zzengine_bind_gate(sys__heap_node* base, uint32_t me, bool enter) {
    const uint64_t gate = sys__scheduler__zzpackage_gate(me);
    const uint64_t n = sys__scheduler__zzprivate_gate_pairs(gate);
    if (n == 0ull || base == 0 || me >= SYS__SCHEDULER__ASKED_BLOCKS) return;
    /* ⛳ LEFT ONLY WHERE ENTERED: a step that could not begin has no environment of its own to leave them in. */
    if (sys__scheduler__zzprivate_gate_bound[me] == enter) return;
    sys__scheduler__zzprivate_gate_bound[me] = enter;
    const uint64_t env = base->args[SYS__COMPUTING_BASE__BINDINGS];
    /* ⛔ ONLY AN ENVIRONMENT: the bindings' own lookup would take any other offset for one and read a table not there. */
    if (!sys__bindings__is(env)) return;
    for (uint64_t i = 0ull; i < n; ++i) {
        const sys__heap_node name = sys__node_array__borrow(gate, 2ull * i);
        if (!enter) { (void)sys__bindings__remove(env, &name); continue; }
        const sys__heap_node answer = sys__node_array__borrow(gate, 2ull * i + 1ull);
        /* the value, held for the moment; the scope takes a hold of its own and this one goes back */
        const sys__heap_node value = sys__answer__read(answer.args[0]);
        (void)sys__bindings__add(env, &name, &value);
        if (sys__heap_node__carries_reference(value.dtype) && value.args[0] != 0ull)
            (void)sys__heap_object__release(value.args[0]);
    }
}

static __device__ inline void sys__scheduler__zzprivate_forget_gate(uint32_t me) {
    if (me >= SYS__SCHEDULER__ASKED_BLOCKS) return;
    const uint64_t was = sys__scheduler__zzprivate_gate[me];
    sys__scheduler__zzprivate_gate[me] = 0ull;
    if (was != 0ull) (void)sys__heap_object__release(was);
}

/* ── A REQUEST, ON ITS WAY AND BACK ────────────────────────────────────────────────────────────────────── */

/* A request on block `cu`'s queue, its value for `answer`, already armed. */
static __device__ inline bool sys__scheduler__zzprivate_queue(uint32_t cu, uint64_t bindings, uint64_t program,
                                                             uint64_t answer, uint64_t gate) {
    const uint64_t queue = sys__scheduler__queue(cu);
    if (queue == 0ull) {
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_UP);
        return false;
    }
    /* ⭐ THE REQUEST IS ITS FOUR CELLS ON THE QUEUE, put on under one hold of its lock, so the worker takes all
     * four or none: nothing is made for it, and nothing written on one block is let go on another. */
    sys__heap_node cells[SYS__SCHEDULER__REQUEST_CELLS];
    cells[SYS__SCHEDULER__REQUEST_ANSWER]   = sys__heap_object__reference_to(answer);
    cells[SYS__SCHEDULER__REQUEST_BINDINGS] = sys__heap_object__reference_to(bindings);
    cells[SYS__SCHEDULER__REQUEST_PROGRAM]  = sys__heap_object__reference_to(program);
    cells[SYS__SCHEDULER__REQUEST_GATE]     = gate != 0ull ? sys__heap_object__reference_to(gate) : sys__heap_node__nothing();
    const bool ok = sys__list__shared_append_cells(queue, cells, SYS__SCHEDULER__REQUEST_CELLS);
    if (!ok) sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_QUEUED);
    else sys__scheduler__zzprivate_ring(cu);
    return ok;
}

static __device__ inline bool sys__scheduler__add(uint32_t cu, uint64_t bindings, uint64_t program, uint64_t answer) {
    if (!sys__answer__zzpackage_arm(answer)) return false;
    if (sys__scheduler__zzprivate_queue(cu, bindings, program, answer, 0ull)) return true;
    sys__answer__zzpackage_disarm(answer);
    return false;
}

static __device__ inline bool sys__scheduler__continue(sys__heap_node* base, uint32_t cu, uint64_t names, uint64_t program,
                                                       uint64_t gate) {
    const uint64_t answer = sys__computing_base__zzengine_answer(base);
    if (answer == 0ull || answer == SYS__SCHEDULER__CONTINUED) return false;   /* nothing to fill, or it went on */
    /* ⛔ AN ANSWER NOBODY ASKED INTO WOULD NEVER FILL, so a gate naming one is refused here rather than parked for good. */
    const uint64_t n = sys__scheduler__zzprivate_gate_pairs(gate);
    for (uint64_t i = 0ull; i < n; ++i) {
        const uint64_t a = sys__scheduler__zzprivate_gate_answer(gate, i);
        if (!sys__answer__is(a) || sys__answer__state(a) == SYS__ANSWER__EMPTY) {
            sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_ASKED);
            return false;
        }
    }
    const uint64_t in = names != 0ull ? names : base->args[SYS__COMPUTING_BASE__BINDINGS];
    if (!sys__scheduler__zzprivate_queue(cu, in, program, answer, n != 0ull ? gate : 0ull)) return false;
    /* ⭐ THE STEP FILLS NOTHING NOW: the queue holds the answer for the next step, and this base lets its own go. */
    sys__computing_base__zzengine_set_answer(base, SYS__SCHEDULER__CONTINUED);
    (void)sys__heap_object__release(answer);
    return true;
}

static __device__ inline bool sys__scheduler__zzengine_next(sys__heap_node* base, uint32_t me) {
    const uint64_t queue = sys__scheduler__queue(me);
    /* ⭐ THE LOOKS WRITE NOTHING: every idle turn of every block makes them, and a lock taken to count would bounce
     * the queue's line between the poller and every block appending to it. A parked step's gate is one word read. */
    if (queue == 0ull) return false;
    if (!sys__scheduler__zzprivate_parked_ready(me) && !sys__list__shared_waiting(queue)) return false;
    /* ⛳ THE BASE FIRST, THE REQUEST AFTER: a request is taken only once there is a base to run it in, so none is
     * ever taken with nowhere to go but back to the end of the queue. Only this block takes from its own queue and its
     * own parked steps, so what it saw is still there once the base is held — or the take finds nothing, and the base
     * goes back free. */
    if (sys__computing_base__claim(base, SYS__SCHEDULER__HOLDER) == 0) return false;
    sys__heap_node taken[SYS__SCHEDULER__REQUEST_CELLS];
    bool got = false;
    /* ⭐ A PARKED STEP WHOSE GATE FILLED GOES FIRST, the oldest: it was queued before anything still on the queue. */
    const uint32_t first = sys__scheduler__zzprivate_first_open(me);
    if (me < SYS__SCHEDULER__ASKED_BLOCKS && first < sys__scheduler__zzprivate_parked_n[me]) {
        sys__scheduler__zzprivate_unpark(me, first, taken);
        got = true;
    }
    for (uint32_t tries = 0u; !got && tries <= SYS__SCHEDULER__PARKED; ++tries) {
        if (!sys__list__shared_take_cells(queue, taken, SYS__SCHEDULER__REQUEST_CELLS)) break;
        if (sys__scheduler__zzprivate_open(taken)) { got = true; break; }
        /* ⭐ GATED, AND ITS ANSWER NOT FILLED YET: parked, and the request behind it looked at. Its gate told this block
         * waits, then looked at once more — filled in between, and the filer may have read the bits before this one. */
        if (sys__scheduler__zzprivate_park(me, taken)) {
            if (sys__scheduler__zzprivate_open(taken)) {
                sys__scheduler__zzprivate_unpark(me, sys__scheduler__zzprivate_parked_n[me] - 1u, taken);
                got = true;
            }
            continue;
        }
        /* No room to park it: back on the queue, behind what came after it — the queue takes holds of its own. */
        (void)sys__list__shared_append_cells(queue, taken, SYS__SCHEDULER__REQUEST_CELLS);
        sys__scheduler__zzprivate_let_go(taken);
        break;
    }
    if (!got) {
        (void)sys__computing_base__init(base, SYS__SCHEDULER__HOLDER, 0ull, 0ull);   /* asking for nothing frees it */
        return false;
    }
    const uint64_t names   = taken[SYS__SCHEDULER__REQUEST_BINDINGS].args[0];
    const uint64_t program = taken[SYS__SCHEDULER__REQUEST_PROGRAM].args[0];
    /* ⛳ THE QUEUE'S HOLD ON THE ANSWER IS THE BASE'S NOW, kept until the value is filed; its hold on the gate is the
     *   block's, its names bound in the step's environment until it is put away. */
    sys__computing_base__zzengine_set_answer(base, taken[SYS__SCHEDULER__REQUEST_ANSWER].args[0]);
    sys__scheduler__zzengine_forget_asked(me);                     /* a computation begins with nothing asked */
    sys__scheduler__zzprivate_forget_gate(me);
    const sys__heap_node gate = taken[SYS__SCHEDULER__REQUEST_GATE];
    if (sys__heap_node__carries_reference(gate.dtype) && gate.args[0] != 0ull) {
        if (me < SYS__SCHEDULER__ASKED_BLOCKS) sys__scheduler__zzprivate_gate[me] = gate.args[0];
        else (void)sys__heap_object__release(gate.args[0]);
    }
    /* `init` takes its own holds, so the queue's go back; a half request leaves the base at ERROR, which the post on
     * the next turn says to its answer. */
    (void)sys__computing_base__init(base, SYS__SCHEDULER__HOLDER, names, program);
    if (names != 0ull)   (void)sys__heap_object__release(names);
    if (program != 0ull) (void)sys__heap_object__release(program);
    return true;
}

static __device__ inline bool sys__scheduler__zzengine_post(sys__heap_node* base, uint32_t me, sys__heap_node value) {
    (void)me;
    const uint64_t answer = sys__computing_base__zzengine_answer(base);
    if (answer == 0ull) return false;                          /* nothing to fill */
    if (answer == SYS__SCHEDULER__CONTINUED) return true;      /* it went on: the next step fills it */
    /* ⭐ THE VALUE GOES INTO THE ANSWER, which takes a hold of its own: nothing is made for it on this block. Then every
     * block with a step parked on it is rung, so one asleep wakes to run it. */
    if (!sys__answer__zzpackage_file(answer, &value)) {
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_LOST);
        return true;
    }
    uint64_t waiting = sys__answer__zzpackage_waiters(answer);
    for (uint32_t b = 0u; waiting != 0ull; ++b, waiting >>= 1)
        if ((waiting & 1ull) != 0ull) sys__scheduler__zzprivate_ring(b);
    return true;
}

static __device__ inline void sys__scheduler__zzengine_put_away(sys__heap_node* base, uint32_t me) {
    /* ⛳ THE ANSWER WORD IS CLEARED BEFORE THE BASE GOES BACK FREE, so the next request this block takes finds it blank;
     *   its hold is let go of after, the answer filled while the base still held it. */
    const uint64_t answer = sys__computing_base__zzengine_answer(base);
    sys__computing_base__zzengine_set_answer(base, 0ull);
    sys__scheduler__zzengine_forget_asked(me);
    sys__scheduler__zzprivate_forget_gate(me);
    const sys__heap_node kept = sys__computing_base__read_result(base, me);   /* the base's hold, now ours */
    if (sys__heap_node__carries_reference(kept.dtype) && kept.args[0] != 0ull)
        (void)sys__heap_object__release(kept.args[0]);
    if (answer != 0ull && answer != SYS__SCHEDULER__CONTINUED) (void)sys__heap_object__release(answer);
}

static __device__ inline bool sys__scheduler__zzengine_post_error(sys__heap_node* base, uint32_t me) {
    if (sys__computing_base__zzengine_answer(base) == 0ull) return false;
    if (sys__computing_base__status(base) != SYS__COMPUTING_BASE__ERROR) return false;
    /* ⭐ AN ERROR IS A VALUE: the answer is filled with what went wrong rather than left waiting. */
    const uint64_t why  = sys__computing_base__fault(base);
    const uint64_t made = sys__error__make(why != 0ull ? why : SYS__SCHEDULER__FAULT_LOST);
    const sys__heap_node value = made != 0ull ? sys__heap_object__reference_to(made) : sys__heap_node__nothing();
    (void)sys__scheduler__zzengine_post(base, me, value);
    if (made != 0ull) (void)sys__heap_object__release(made);
    sys__scheduler__zzengine_put_away(base, me);
    return true;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_SCHEDULER__IMPL_CUH */
