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
#include "bindings__header.cuh"          /* an environment that moves is one computation's alone */
#include "garbage_collection__header.cuh" /* the collection mode, which moves with its computation */
#include "silicon/silicon__header.cuh"    /* which block the step runs on */

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
        if (name.dtype == SYS__KIND__VALUE_NULL) continue;      /* a wait's gate: an answer waited on, no name to bind */
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

/* A moved computation's places, emptied where it will not go on — its request refused, or unpacked for a step that never
 * began — as the evaluator empties its own at an evaluation's end: every entry is borrowed from the program, and a stack
 * taken apart with them in it would give up holds it never had. A pop hands its value over with no count moved, so
 * what it hands is dropped. The places are this one's alone: the request, or the resume, held them. */
static __device__ inline void sys__scheduler__zzprivate_drain(uint64_t places) {
    if (places == 0ull) return;
    const sys__heap_node at = sys__heap_object__reference_to(places);
    while (!sys__stack__empty(&at)) (void)sys__stack__pop(&at);
}

/* ── THE PROCESSES (NN-154's kill) ─────────────────────────────────────────────────────
 * ⚖ 2026-11-04 *"maybe we would have to add an hypervisor that undoes every queued compute with the specified pid"* ·
 * 2026-11-06 *"cool"* (the overnight's plan). A request carries the process it is of: the asker's — every request a
 * computation makes and every step it goes on as, moved or continued — or a new one where nothing running asked (the
 * host's doors). `(sys__scheduler__kill pid)` takes the process's requests off every queue, their answers filled with
 * SCKL, and remembers it: a block refuses a request of it taken later and drops its steps parked; a step of it still
 * running is ended at its next move or slice, its answer SCKL, and an answer it fills at its end is SCKL too. What it
 * did is never undone: preemptive, not protected. The last KILLED_KEPT kills are remembered; a process is a 32-bit
 * number. */
static __thread unsigned int sys__scheduler__zzprivate_pid = 0u;   /* the process of the step this runner runs, or 0 */
static unsigned int sys__scheduler__zzprivate_pids = 0u;           /* processes made so far */
static unsigned int sys__scheduler__zzprivate_killed[SYS__SCHEDULER__KILLED_KEPT];
static unsigned int sys__scheduler__zzprivate_kills = 0u;          /* kills made: the next one's slot, modulo KEPT */

static __device__ inline uint64_t sys__scheduler__pid(void) {
    return (uint64_t)sys__scheduler__zzprivate_pid;
}

static __device__ inline bool sys__scheduler__zzpackage_killed(uint64_t pid) {
    if (pid == 0ull) return false;
    const unsigned int made = sys__silicon__read_u32(&sys__scheduler__zzprivate_kills);
    const unsigned int n = made < SYS__SCHEDULER__KILLED_KEPT ? made : SYS__SCHEDULER__KILLED_KEPT;
    for (unsigned int i = 0u; i < n; ++i)
        if ((uint64_t)sys__silicon__read_u32(&sys__scheduler__zzprivate_killed[i]) == pid) return true;
    return false;
}

/* The process a request is made for: the running step's, or a new one where none runs here. */
static __device__ inline unsigned int sys__scheduler__zzprivate_pid_to_give(void) {
    const unsigned int mine = sys__scheduler__zzprivate_pid;
    return mine != 0u ? mine : sys__silicon__add_u32(&sys__scheduler__zzprivate_pids, 1u) + 1u;
}

/* An answer of a process killed filled with SCKL, and every block with a step parked on it rung. */
static __device__ inline void sys__scheduler__zzprivate_killed_into(uint64_t answer) {
    if (answer == 0ull || !sys__answer__is(answer)) return;
    const uint64_t made = sys__error__make(SYS__SCHEDULER__FAULT_KILLED);
    const sys__heap_node value = made != 0ull ? sys__heap_object__reference_to(made) : sys__heap_node__nothing();
    if (sys__answer__zzpackage_file(answer, &value)) {
        uint64_t waiting = sys__answer__zzpackage_waiters(answer);
        for (uint32_t b = 0u; waiting != 0ull; ++b, waiting >>= 1)
            if ((waiting & 1ull) != 0ull) sys__scheduler__zzprivate_ring(b);
    }
    if (made != 0ull) (void)sys__heap_object__release(made);
}

/* A request of a process killed, refused where it waits — on a queue, or parked: its answer SCKL, a moved computation's
 * places emptied, and every hold its cells carry let go — on its answer, its names or the environment it moved with,
 * its program as it stands, its gate, what it carried. Each object whose last hold that was is taken apart by its own
 * kind, and lets go of what it holds: an environment of the values bound in it, a program of the values written into
 * it, the carried array of the places and the answers last asked. */
static __device__ inline void sys__scheduler__zzprivate_refuse(const sys__heap_node* cells) {
    const sys__heap_node a = cells[SYS__SCHEDULER__REQUEST_ANSWER];
    if (sys__heap_node__carries_reference(a.dtype)) sys__scheduler__zzprivate_killed_into(a.args[0]);
    /* ⛔ A MOVED COMPUTATION'S PLACES ARE DRAINED BEFORE THEY GO: what they hold is borrowed from its program */
    const sys__heap_node carried = cells[SYS__SCHEDULER__REQUEST_CARRIED];
    if (sys__heap_node__carries_reference(carried.dtype) && sys__node_array__is(carried.args[0])) {
        const sys__heap_node path = sys__node_array__borrow(carried.args[0], SYS__SCHEDULER__CARRIED_PLACES);
        if (sys__heap_node__carries_reference(path.dtype)) sys__scheduler__zzprivate_drain(path.args[0]);
    }
    sys__scheduler__zzprivate_let_go(cells);
}

static __device__ inline uint64_t sys__scheduler__zzprivate_request_pid(const sys__heap_node* cells) {
    const sys__heap_node p = cells[SYS__SCHEDULER__REQUEST_PID];
    return p.dtype == SYS__KIND__VALUE_INT ? p.args[0] : 0ull;
}

/* ── INLINE ────────────────────────────────────────────────────────────────────────────────────────────────
 * ⚖ 2026-11-07 *"let's add a sys__scheduler__execute_bounded or something similar like sys__scheduler__compute_inline,
 * which is an extra binding that touches sys__scheduler__preemptive and sets it to false and resets it to true once it
 * completes its internal computation. this way we are sure that the evaluator processing them does not detach, but it
 * reopens the issue"*. A computation inside `(sys__scheduler__compute_inline 'form)` is never given up at a slice: a
 * program whose steps keep a worker's scratch between verbs — a model's do — runs them inline, so another program on
 * its block cannot write that scratch between two of its verbs (`MEASURED`: two of DeepSeek's seats on one card,
 * preempted mid-step at a slice of 1 us, and one answered another's tokens — measurements/2026-11-06_kill_node03.md).
 * ⛔ IT REOPENS WHAT THE SLICE CLOSED, INSIDE IT: a form that never ends there holds its block. A kill still ends it
 * at its next slice, as the kill is read before the slice gives anything up (`eval.cuh`). ⛳ A DEPTH, NOT A FLAG, so
 * a section inside another leaves the outer one inline; it goes with a computation that moves, beside its collection
 * mode, and a step begins at 0. */
static __thread unsigned int sys__scheduler__zzprivate_inline = 0u;   /* how many sections the running step is inside */

static __device__ inline void sys__scheduler__zzpackage_inline_enter(void) {
    sys__scheduler__zzprivate_inline += 1u;
}

static __device__ inline void sys__scheduler__zzpackage_inline_leave(void) {
    if (sys__scheduler__zzprivate_inline != 0u) sys__scheduler__zzprivate_inline -= 1u;
}

static __device__ inline bool sys__scheduler__preemptive(void) {
    return sys__scheduler__zzprivate_inline == 0u;
}

/* ── A REQUEST, ON ITS WAY AND BACK ────────────────────────────────────────────────────────────────────── */

/* A request on block `cu`'s queue, its value for `answer`, already armed. */
static __device__ inline bool sys__scheduler__zzprivate_queue(uint32_t cu, uint64_t bindings, uint64_t program,
                                                             uint64_t answer, uint64_t gate, uint64_t carried) {
    const uint64_t queue = sys__scheduler__queue(cu);
    if (queue == 0ull) {
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_UP);
        return false;
    }
    /* ⛔ A PROCESS KILLED ASKS NOTHING MORE: a step of it still running is refused here, its continuation too. */
    const unsigned int pid = sys__scheduler__zzprivate_pid_to_give();
    if (sys__scheduler__zzpackage_killed(pid)) {
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_KILLED);
        return false;
    }
    /* ⭐ THE REQUEST IS ITS SIX CELLS ON THE QUEUE, put on under one hold of its lock, so the worker takes all
     * six or none: nothing is made for it, and nothing written on one block is let go on another. */
    sys__heap_node cells[SYS__SCHEDULER__REQUEST_CELLS];
    cells[SYS__SCHEDULER__REQUEST_ANSWER]   = sys__heap_object__reference_to(answer);
    cells[SYS__SCHEDULER__REQUEST_BINDINGS] = sys__heap_object__reference_to(bindings);
    cells[SYS__SCHEDULER__REQUEST_PROGRAM]  = sys__heap_object__reference_to(program);
    cells[SYS__SCHEDULER__REQUEST_GATE]     = gate != 0ull ? sys__heap_object__reference_to(gate) : sys__heap_node__nothing();
    cells[SYS__SCHEDULER__REQUEST_CARRIED]  = carried != 0ull ? sys__heap_object__reference_to(carried) : sys__heap_node__nothing();
    cells[SYS__SCHEDULER__REQUEST_PID].dtype = SYS__KIND__VALUE_INT; cells[SYS__SCHEDULER__REQUEST_PID].num_args = 0u;
    cells[SYS__SCHEDULER__REQUEST_PID].op_code = 0ull; cells[SYS__SCHEDULER__REQUEST_PID].args[0] = (uint64_t)pid;
    const bool ok = sys__list__shared_append_cells(queue, cells, SYS__SCHEDULER__REQUEST_CELLS);
    if (!ok) sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_QUEUED);
    else sys__scheduler__zzprivate_ring(cu);
    return ok;
}

static __device__ inline bool sys__scheduler__add(uint32_t cu, uint64_t bindings, uint64_t program, uint64_t answer) {
    if (!sys__answer__zzpackage_arm(answer)) return false;
    if (sys__scheduler__zzprivate_queue(cu, bindings, program, answer, 0ull, 0ull)) return true;
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
    /* ⛔ ITS OWN NAMES ON ANOTHER BLOCK ARE TWO PROGRAMS ON ONE ENVIRONMENT — the next step there may start while this
     *   one runs on — so the environment is no longer one computation's, and never moves (▶ `continue_on`). On its own
     *   block the next step starts only once this one has ended. */
    if (names == 0ull && cu != (uint32_t)sys__silicon__block_id()) sys__bindings__zzpackage_set_alone(in, false);
    if (!sys__scheduler__zzprivate_queue(cu, in, program, answer, n != 0ull ? gate : 0ull, 0ull)) return false;
    /* ⭐ THE STEP FILLS NOTHING NOW: the queue holds the answer for the next step, and this base lets its own go. */
    sys__computing_base__zzengine_set_answer(base, SYS__SCHEDULER__CONTINUED);
    (void)sys__heap_object__release(answer);
    return true;
}

/* ── A COMPUTATION MOVED ───────────────────────────────────────────────────
 * ⚖ 2026-11-06 *"as long as we never maintain two programs to reference the same live bindings we are good both to send
 * the rest of the compute somewhere else and to send it on the back of your own queue"* · *"actually program should be
 * non specified in the code, it has to be itself so it has to be (sys__scheduler__continue_on cu) and the bindings and
 * the program are implicit. this guarantees no double live bindings"*.
 * The verb names the block and nothing else; the evaluator, once the verb's form holds its `#t`, hands over what it
 * holds — the program it is rewriting, the places from its root to that form, its environment — and stops. Nothing
 * names a second program, so no second program ever runs in that environment. The block that takes the request runs
 * the same evaluation on from the same form: the evaluator rewrites each form with its value, so the program as it
 * stands is the rest of the computation. What was its block's while it ran there and is its own goes in the same
 * array: its collection mode, and the answer it last asked of each block (`(sys__wait cu)` reads it). The names its
 * gate bound stay bound in its environment, and go with it. */
static __thread uint64_t sys__scheduler__zzprivate_handed = 0ull;          /* the block it moves to, plus one, or 0 */
static __thread uint64_t sys__scheduler__zzprivate_resume_places = 0ull;   /* a moved computation's places, held */
static __thread uint64_t sys__scheduler__zzprivate_resume_current = 0ull;  /* and its form, held by them */

/* ── A WAIT PARKED ──────────────────────────────────────────────────────────────────────────────────────────────
 * ⚖ 2026-11-07 *"the only place that needs attention and that must not wait on things is an opcode, so as long as we
 * specify that and that c opcodes do not wait but return and let the wait be done in lisp then we are good. we can also
 * say that sys__wait sends the computation to the back of the queue if the object is not available"* (NN-158).
 * `(sys__wait a)` on an answer not there yet leaves its form as it stands and hands the computation to its own block's
 * queue, as `continue_on` would, with `a` its gate: the block parks a gated step until its gate fills and runs it then,
 * before what was queued after it (`zzengine_next`), and there the same form is applied again and answers the value.
 * Nothing spins, and the block runs what else waits meanwhile. ⛔ ONLY WHAT MAY MOVE IS PARKED — an answer to fill, an
 * environment of its own, not inside `compute_inline` (a model's step keeps a worker's scratch between verbs) — and a
 * step whose park found no room once waits where it stands for the rest of it: `sys__wait` applied again would only try
 * again. Those wait as they always did, bounded, and a wait that runs out answers ANSR. */
static __thread uint64_t sys__scheduler__zzprivate_park_gate = 0ull;   /* the answer a hand-off parks on, held, or 0 */
static __thread bool     sys__scheduler__zzprivate_park_refused = false;   /* this step's park found no room */

static __device__ inline bool sys__scheduler__zzpackage_may_park(sys__heap_node* base) {
    if (base == 0 || sys__scheduler__zzprivate_inline != 0u || sys__scheduler__zzprivate_park_refused) return false;
    const uint64_t answer = sys__computing_base__zzengine_answer(base);
    if (answer == 0ull || answer == SYS__SCHEDULER__CONTINUED) return false;
    return sys__bindings__zzpackage_alone(base->args[SYS__COMPUTING_BASE__BINDINGS]);
}

static __device__ inline void sys__scheduler__zzpackage_park_on(uint64_t answer) {
    if (sys__scheduler__zzprivate_park_gate != 0ull) (void)sys__heap_object__release(sys__scheduler__zzprivate_park_gate);
    (void)sys__heap_object__retain(answer);
    sys__scheduler__zzprivate_park_gate = answer;
    sys__scheduler__zzprivate_handed = (uint64_t)(uint32_t)sys__silicon__block_id() + 1ull;
}

static __device__ inline bool sys__scheduler__continue_on(sys__heap_node* base, uint32_t cu) {
    const uint64_t answer = sys__computing_base__zzengine_answer(base);
    if (answer == 0ull || answer == SYS__SCHEDULER__CONTINUED) {                /* nothing to fill, or it went on */
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_QUEUED);
        return false;
    }
    /* ⛔⛔ ONLY AN ENVIRONMENT NOBODY ELSE NAMES MOVES: the host's, or one a step still running shares, would be two
     *   programs on one live environment. */
    if (!sys__bindings__zzpackage_alone(base->args[SYS__COMPUTING_BASE__BINDINGS])) {
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_SHARED);
        return false;
    }
    sys__scheduler__zzprivate_handed = (uint64_t)cu + 1ull;
    return true;
}

static __device__ inline uint64_t sys__scheduler__zzengine_handed(void) {
    const uint64_t to = sys__scheduler__zzprivate_handed;
    if (to != 0ull) sys__scheduler__zzprivate_handed = 0ull;
    return to;
}

static __device__ inline bool sys__scheduler__zzengine_move(sys__heap_node* base, uint32_t cu, uint64_t program,
                                                            uint64_t places, uint64_t current) {
    const uint32_t me = (uint32_t)sys__silicon__block_id();
    const uint64_t answer = sys__computing_base__zzengine_answer(base);
    /* ⭐ A WAIT PARKED MOVES WITH ITS ANSWER FOR A GATE: one pair, no name to bind — the step waits on it and binds nothing */
    const uint64_t waits_on = sys__scheduler__zzprivate_park_gate;
    sys__scheduler__zzprivate_park_gate = 0ull;
    uint64_t gate = 0ull;
    if (waits_on != 0ull) {
        gate = sys__node_array__create(2ull);
        const sys__heap_node on = sys__heap_object__reference_to(waits_on);
        if (gate != 0ull && !sys__node_array__set(gate, 1ull, &on)) { (void)sys__heap_object__release(gate); gate = 0ull; }
        (void)sys__heap_object__release(waits_on);              /* the gate holds it now, or nothing does */
        if (gate == 0ull) {
            sys__scheduler__zzprivate_park_refused = true;      /* ⛔ no room: the wait waits where it stands */
            sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_QUEUED);
            return false;
        }
    }
    if (answer == 0ull || answer == SYS__SCHEDULER__CONTINUED || program == 0ull || places == 0ull) {
        if (gate != 0ull) { (void)sys__heap_object__release(gate); sys__scheduler__zzprivate_park_refused = true; }
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_QUEUED);
        return false;
    }
    bool asked = false;
    for (uint32_t b = 0u; me < SYS__SCHEDULER__ASKED_BLOCKS && b < SYS__SCHEDULER__ASKED_BLOCKS && !asked; ++b)
        asked = sys__scheduler__zzprivate_asked[me][b] != 0ull;
    const bool mode = sys__garbage_collection__zzpackage_manual();
    const uint64_t carried = sys__node_array__create((uint64_t)SYS__SCHEDULER__CARRIED_ASKED
                                                     + (asked ? (uint64_t)SYS__SCHEDULER__ASKED_BLOCKS : 0ull));
    if (carried == 0ull) {
        if (gate != 0ull) { (void)sys__heap_object__release(gate); sys__scheduler__zzprivate_park_refused = true; }
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_QUEUED);
        return false;
    }
    sys__heap_node cell;
    cell.dtype = SYS__KIND__VALUE_INT; cell.num_args = 0u; cell.op_code = 0ull;
    cell.args[0] = (mode ? 1ull : 0ull) | ((uint64_t)sys__scheduler__zzprivate_inline << 1);
    bool ok = sys__node_array__set(carried, SYS__SCHEDULER__CARRIED_MODE, &cell);
    cell.args[0] = current;
    ok = ok && sys__node_array__set(carried, SYS__SCHEDULER__CARRIED_CURRENT, &cell);
    const sys__heap_node path = sys__heap_object__reference_to(places);
    ok = ok && sys__node_array__set(carried, SYS__SCHEDULER__CARRIED_PLACES, &path);        /* its own hold */
    for (uint32_t b = 0u; ok && asked && b < SYS__SCHEDULER__ASKED_BLOCKS; ++b) {
        const uint64_t a = sys__scheduler__zzprivate_asked[me][b];
        if (a == 0ull) continue;
        const sys__heap_node ref = sys__heap_object__reference_to(a);
        ok = sys__node_array__set(carried, (uint64_t)SYS__SCHEDULER__CARRIED_ASKED + b, &ref);
    }
    const uint64_t env = base->args[SYS__COMPUTING_BASE__BINDINGS];
    ok = ok && sys__scheduler__zzprivate_queue(cu, env, program, answer, gate, carried);
    (void)sys__heap_object__release(carried);                       /* the request holds its own, if it was queued */
    if (gate != 0ull) (void)sys__heap_object__release(gate);       /* and its gate */
    if (!ok) {
        if (gate != 0ull) sys__scheduler__zzprivate_park_refused = true;
        sys__fault__raise(0ull, SYS__SCHEDULER__FAULT_NOT_QUEUED);
        return false;
    }
    /* ⛔⛔ FROM HERE ON THE PROGRAM, ITS PLACES AND ITS ENVIRONMENT ARE THE NEXT STEP'S, perhaps running already: nothing
     *   below reads or writes them, and the evaluator only lets its holds go. */
    sys__computing_base__zzengine_set_answer(base, SYS__SCHEDULER__CONTINUED);
    (void)sys__heap_object__release(answer);
    if (me < SYS__SCHEDULER__ASKED_BLOCKS) sys__scheduler__zzprivate_gate_bound[me] = false;   /* its names went with it */
    if (mode) (void)sys__garbage_collection__manual(false);                    /* the mode went too */
    sys__scheduler__zzprivate_inline = 0u;                                     /* and how deep inline it stood */
    return true;
}

/* What a moved computation carried, made block `me`'s as its step is taken: its mode, its asked row, and where its
 * evaluation stands, kept for `eng__eval` to take. False for a request that did not move. */
static __device__ inline bool sys__scheduler__zzprivate_unpack(uint32_t me, const sys__heap_node* carried) {
    if (!sys__heap_node__carries_reference(carried->dtype) || !sys__node_array__is(carried->args[0])) return false;
    const uint64_t from = carried->args[0];
    const sys__heap_node mode = sys__node_array__borrow(from, SYS__SCHEDULER__CARRIED_MODE);
    (void)sys__garbage_collection__manual(mode.dtype == SYS__KIND__VALUE_INT && (mode.args[0] & 1ull) == 1ull);
    sys__scheduler__zzprivate_inline = mode.dtype == SYS__KIND__VALUE_INT ? (unsigned int)(mode.args[0] >> 1) : 0u;
    const uint64_t length = sys__node_array__length(from);
    for (uint32_t b = 0u; (uint64_t)SYS__SCHEDULER__CARRIED_ASKED + b < length && b < SYS__SCHEDULER__ASKED_BLOCKS; ++b) {
        const sys__heap_node a = sys__node_array__borrow(from, (uint64_t)SYS__SCHEDULER__CARRIED_ASKED + b);
        if (sys__heap_node__carries_reference(a.dtype) && a.args[0] != 0ull) sys__scheduler__zzpackage_ask(me, b, a.args[0]);
    }
    const sys__heap_node path = sys__node_array__borrow(from, SYS__SCHEDULER__CARRIED_PLACES);
    const sys__heap_node form = sys__node_array__borrow(from, SYS__SCHEDULER__CARRIED_CURRENT);
    if (!sys__heap_node__carries_reference(path.dtype) || path.args[0] == 0ull) return false;
    (void)sys__heap_object__retain(path.args[0]);                   /* held here until the evaluation takes it */
    sys__scheduler__zzprivate_resume_places  = path.args[0];
    sys__scheduler__zzprivate_resume_current = form.args[0];
    return true;
}

static __device__ inline bool sys__scheduler__zzengine_resumed(uint64_t* places, uint64_t* current) {
    if (sys__scheduler__zzprivate_resume_places == 0ull) return false;
    *places  = sys__scheduler__zzprivate_resume_places;             /* the hold is the evaluation's now */
    *current = sys__scheduler__zzprivate_resume_current;
    sys__scheduler__zzprivate_resume_places  = 0ull;
    sys__scheduler__zzprivate_resume_current = 0ull;
    return true;
}

/* A moved computation's places let go of where its request is put away without having run. */
static __device__ inline void sys__scheduler__zzprivate_forget_resume(void) {
    const uint64_t was = sys__scheduler__zzprivate_resume_places;
    sys__scheduler__zzprivate_resume_places  = 0ull;
    sys__scheduler__zzprivate_resume_current = 0ull;
    sys__computing_base__zzpackage_under_way(false);
    if (was != 0ull) {
        sys__scheduler__zzprivate_drain(was);
        (void)sys__heap_object__release(was);
    }
}

/* ── THE SLICE ─────────────────────────────────────────────────────────────
 * ⚖ 2026-11-06 *"the evaluator needs to pause the program after n clock cycles, set as a boot variable, and write to
 * itself its computation queue"* · *"it basically is a free step from cooperative multitasking to preemptive but not
 * protected. it should be behind a flag maybe? and does the counter reset once a new computation starts?"*.
 * `sys__scheduler__slice_cycles` in `boot_commands`, read by sys's host init — or `sys__scheduler__slice_us`, the same
 * slice in microseconds (▶ `ticks_of_us`): 0, or absent, and nothing is preempted, the engine's own default; the
 * runtime boots every machine with a slice of 50 ms (⚖ 2026-11-07 *"set the slice to 0.05s by default"*, python's
 * `Machine.boot`). Otherwise the evaluator reads the clock after every verb, and once a computation has run that
 * many cycles since its evaluation began it does what `(sys__scheduler__continue_on <its own block>)` does: it goes to
 * the back of its block's queue, and the next step there runs. ⭐ THE SLICE STARTS WITH EVERY EVALUATION — a new
 * computation's, and a moved one's where it goes on — so each gets a whole one. ⛳ IT GIVES THE BLOCK UP ONLY TO
 * SOMEBODY: with nothing else waiting there a computation runs a slice more where it is, since moving would only bring
 * it back. And only what may move is preempted: a step with an answer to fill and an environment of its own — never the
 * host's, never a package's init. ⛔ PREEMPTIVE, NOT PROTECTED: a slice is as fine as the longest verb, which runs to its
 * end; and a program whose steps share a worker's scratch between verbs — the models' do, a seat's layer in the
 * worker's buffers — must not run preempted beside another on that block: the flag is a machine's, said at its boot. */
static uint64_t sys__scheduler__zzprivate_slice = 0ull;

static __device__ inline void sys__scheduler__zzpackage_set_slice(uint64_t cycles) {
    sys__scheduler__zzprivate_slice = cycles;
}

static __device__ inline uint64_t sys__scheduler__zzengine_slice(void) {
    return sys__scheduler__zzprivate_slice;
}

/* The clock a slice is counted on: the processor's time-stamp counter, which counts at its nominal clock — or, where
 * there is none, nanoseconds. */
static __device__ inline uint64_t sys__scheduler__zzengine_clock(void) {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_ia32_rdtsc();
#else
    struct timespec now;
    sys__os__clock(&now);
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
#endif
}

/* The clock's ticks in `us` microseconds. ⛳ THE TIME-STAMP COUNTER'S RATE IS WRITTEN NOWHERE THE ENGINE CAN READ — the
 * processor's nominal clock, 2.4 GHz on node03's E5-2680 v4, 2.1 GHz on a Gold 6252 — so it is timed, once a boot,
 * against the system's clock: the ticks over 2 ms of it. Where the slice's clock is that system clock, nanoseconds. */
static inline uint64_t sys__scheduler__zzpackage_ticks_of_us(uint64_t us) {
#if defined(__x86_64__) || defined(__i386__)
    struct timespec from, now;
    sys__os__clock(&from);
    const uint64_t first = __builtin_ia32_rdtsc();
    uint64_t ns = 0ull;
    while (ns < 2000000ull) {
        sys__os__clock(&now);
        ns = (uint64_t)(now.tv_sec - from.tv_sec) * 1000000000ull + (uint64_t)now.tv_nsec - (uint64_t)from.tv_nsec;
    }
    const uint64_t ticks = __builtin_ia32_rdtsc() - first;
    return (uint64_t)((double)ticks / (double)ns * 1000.0 * (double)us);
#else
    return us * 1000ull;
#endif
}

static __device__ inline bool sys__scheduler__zzengine_should_yield(sys__heap_node* base) {
    if (sys__scheduler__zzprivate_inline != 0u) return false;                  /* ⛔ inline: never given up here */
    const uint32_t me = (uint32_t)sys__silicon__block_id();
    const uint64_t queue = sys__scheduler__queue(me);
    if (queue == 0ull) return false;
    if (!sys__list__shared_waiting(queue) && !sys__scheduler__zzprivate_parked_ready(me)) return false;
    const uint64_t answer = sys__computing_base__zzengine_answer(base);
    if (answer == 0ull || answer == SYS__SCHEDULER__CONTINUED) return false;
    return sys__bindings__zzpackage_alone(base->args[SYS__COMPUTING_BASE__BINDINGS]);
}

static __device__ inline uint64_t sys__scheduler__kill(uint64_t pid) {
    if (pid == 0ull || pid > 0xFFFFFFFFull) return 0ull;
    /* ⛳ REMEMBERED FIRST, THE QUEUES AFTER: a request of it put on meanwhile is refused at the queue, or by its block */
    if (!sys__scheduler__zzpackage_killed(pid)) {
        const unsigned int at = sys__silicon__add_u32(&sys__scheduler__zzprivate_kills, 1u) % SYS__SCHEDULER__KILLED_KEPT;
        unsigned int* slot = &sys__scheduler__zzprivate_killed[at];
        (void)sys__silicon__cas_u32(slot, sys__silicon__read_u32(slot), (unsigned int)pid);
    }
    const sys__heap_node where = sys__system_register__get(SYS__SYSTEM_REGISTER__SCHEDULER_QUEUE_ARRAY);
    const uint64_t queues = (where.dtype == SYS__KIND__VALUE_INT && where.args[0] != 0ull) ? sys__node_array__length(where.args[0]) : 0ull;
    uint64_t taken = 0ull;
    for (uint32_t b = 0u; b < queues; ++b) {
        const uint64_t queue = sys__scheduler__zzprivate_list(where, b);
        if (queue == 0ull) continue;
        for (;;) {                                         /* eight at a time; their answers filled outside the lock */
            sys__heap_node got[8u * SYS__SCHEDULER__REQUEST_CELLS];
            const uint64_t k = sys__list__shared_take_matching(queue, SYS__SCHEDULER__REQUEST_CELLS, SYS__SCHEDULER__REQUEST_PID,
                                                               pid, got, 8u);
            for (uint64_t i = 0ull; i < k; ++i) sys__scheduler__zzprivate_refuse(&got[i * SYS__SCHEDULER__REQUEST_CELLS]);
            taken += k;
            if (k < 8u) break;
        }
    }
    return taken;
}

/* ── A PROCESS FOR THE HOST, AND ONE ASKED TO STOP (NN-160) ─────────────────────────────────────────────────
 * ⚖ 2026-11-07 the design agreed: *"1 and it keeps it from a shared heap object that holds the counter, so pids are
 * unique"* · *"the process is the request of doing work, that continues itself until it produces a final result. so
 * stopping will block whatever is in flight, return an error to that process's result as its value"* · *"a stop completes
 * the current token and a pkill tears it down… maybe the stop is a more mild let binding checked once per token?"*.
 * The host takes a process from the counter every process is made from, so no two are one, and makes its requests as it
 * (`process_as`: this thread's, inherited by what they ask and go on as). Two levels: the kill above — whatever of it is
 * in flight torn down, its answers SCKL — and a stop: the process remembered as asked to stop, which its program reads
 * once a token (`(sys__scheduler__stopped)`, ▶ lisp/generate.lisp) and ends at the token it is on, everything it did
 * whole. A killed process reads as stopped too. The last KILLED_KEPT stops are remembered, as the kills are. */
static unsigned int sys__scheduler__zzprivate_stopped[SYS__SCHEDULER__KILLED_KEPT];
static unsigned int sys__scheduler__zzprivate_stops = 0u;          /* stops made: the next one's slot, modulo KEPT */

static __device__ inline uint64_t sys__scheduler__process_new(void) {
    return (uint64_t)(sys__silicon__add_u32(&sys__scheduler__zzprivate_pids, 1u) + 1u);
}

static __device__ inline uint64_t sys__scheduler__process_as(uint64_t pid) {
    const unsigned int was = sys__scheduler__zzprivate_pid;
    sys__scheduler__zzprivate_pid = pid > 0xFFFFFFFFull ? 0u : (unsigned int)pid;
    return (uint64_t)was;
}

static __device__ inline bool sys__scheduler__zzprivate_asked_to_stop(uint64_t pid) {
    if (pid == 0ull) return false;
    const unsigned int made = sys__silicon__read_u32(&sys__scheduler__zzprivate_stops);
    const unsigned int n = made < SYS__SCHEDULER__KILLED_KEPT ? made : SYS__SCHEDULER__KILLED_KEPT;
    for (unsigned int i = 0u; i < n; ++i)
        if ((uint64_t)sys__silicon__read_u32(&sys__scheduler__zzprivate_stopped[i]) == pid) return true;
    return false;
}

static __device__ inline bool sys__scheduler__stop(uint64_t pid) {
    if (pid == 0ull || pid > 0xFFFFFFFFull) return false;
    if (!sys__scheduler__zzprivate_asked_to_stop(pid)) {
        const unsigned int at = sys__silicon__add_u32(&sys__scheduler__zzprivate_stops, 1u) % SYS__SCHEDULER__KILLED_KEPT;
        unsigned int* slot = &sys__scheduler__zzprivate_stopped[at];
        (void)sys__silicon__cas_u32(slot, sys__silicon__read_u32(slot), (unsigned int)pid);
    }
    return true;
}

static __device__ inline bool sys__scheduler__stopped(void) {
    const uint64_t pid = (uint64_t)sys__scheduler__zzprivate_pid;
    return sys__scheduler__zzprivate_asked_to_stop(pid) || sys__scheduler__zzpackage_killed(pid);
}

static __device__ inline bool sys__scheduler__zzengine_killed(sys__heap_node* base) {
    if (!sys__scheduler__zzpackage_killed((uint64_t)sys__scheduler__zzprivate_pid)) return false;
    const uint64_t answer = sys__computing_base__zzengine_answer(base);
    if (answer == 0ull || answer == SYS__SCHEDULER__CONTINUED) return false;
    sys__scheduler__zzprivate_killed_into(answer);
    sys__computing_base__zzengine_set_answer(base, SYS__SCHEDULER__CONTINUED);   /* filled: nothing more posts into it */
    (void)sys__heap_object__release(answer);
    return true;
}

/* A block's parked steps of a process killed, refused: they wait on answers that may never fill. */
static __device__ inline void sys__scheduler__zzprivate_drop_killed_parked(uint32_t me) {
    if (me >= SYS__SCHEDULER__ASKED_BLOCKS || sys__silicon__read_u32(&sys__scheduler__zzprivate_kills) == 0u) return;
    for (uint32_t i = 0u; i < sys__scheduler__zzprivate_parked_n[me];) {
        if (!sys__scheduler__zzpackage_killed(sys__scheduler__zzprivate_request_pid(sys__scheduler__zzprivate_parked[me][i]))) {
            ++i;
            continue;
        }
        sys__heap_node cells[SYS__SCHEDULER__REQUEST_CELLS];
        sys__scheduler__zzprivate_unpark(me, i, cells);
        sys__scheduler__zzprivate_refuse(cells);
    }
}

static __device__ inline bool sys__scheduler__zzengine_next(sys__heap_node* base, uint32_t me) {
    const uint64_t queue = sys__scheduler__queue(me);
    /* ⭐ THE LOOKS WRITE NOTHING: every idle turn of every block makes them, and a lock taken to count would bounce
     * the queue's line between the poller and every block appending to it. A parked step's gate is one word read. */
    if (queue == 0ull) return false;
    sys__scheduler__zzprivate_drop_killed_parked(me);
    if (!sys__scheduler__zzprivate_parked_ready(me) && !sys__list__shared_waiting(queue)) return false;
    /* ⛳ THE BASE FIRST, THE REQUEST AFTER: a request is taken only once there is a base to run it in, so none is
     * ever taken with nowhere to go but back to the end of the queue. Only this block takes its own parked steps; a
     * kill takes from its queue too (▶ `sys__scheduler__kill`), so what it saw may be gone once the base is held —
     * the take then finds nothing, and the base goes back free. */
    if (sys__computing_base__claim(base, SYS__SCHEDULER__HOLDER) == 0) return false;
    sys__heap_node taken[SYS__SCHEDULER__REQUEST_CELLS];
    bool got = false;
    for (uint32_t refused = 0u; ; ++refused) {             /* again where what was taken is of a process killed */
        got = false;
        /* ⭐ A PARKED STEP WHOSE GATE FILLED GOES FIRST, the oldest: it was queued before anything still on the
         * queue. */
        const uint32_t first = sys__scheduler__zzprivate_first_open(me);
        if (me < SYS__SCHEDULER__ASKED_BLOCKS && first < sys__scheduler__zzprivate_parked_n[me]) {
            sys__scheduler__zzprivate_unpark(me, first, taken);
            got = true;
        }
        for (uint32_t tries = 0u; !got && tries <= SYS__SCHEDULER__PARKED; ++tries) {
            if (!sys__list__shared_take_cells(queue, taken, SYS__SCHEDULER__REQUEST_CELLS)) break;
            if (sys__scheduler__zzprivate_open(taken)) { got = true; break; }
            /* ⭐ GATED, AND ITS ANSWER NOT FILLED YET: parked, and the request behind it looked at. Its gate told this
             * block waits, then looked at once more — filled in between, and the filer may have read the bits before
             * this one. */
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
        if (!got || !sys__scheduler__zzpackage_killed(sys__scheduler__zzprivate_request_pid(taken))) break;
        sys__scheduler__zzprivate_refuse(taken);              /* ⛔ a process killed: refused, and the next looked at */
        if (refused > SYS__SCHEDULER__PARKED) { got = false; break; }
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
    sys__scheduler__zzprivate_pid = (unsigned int)sys__scheduler__zzprivate_request_pid(taken);   /* the step's process */
    sys__scheduler__zzprivate_inline = 0u;                         /* inline nowhere, unless it moved here so (unpack) */
    sys__scheduler__zzprivate_park_refused = false;                /* and free to park */
    sys__scheduler__zzengine_forget_asked(me);                     /* a computation begins with nothing asked */
    sys__scheduler__zzprivate_forget_gate(me);
    const sys__heap_node gate = taken[SYS__SCHEDULER__REQUEST_GATE];
    if (sys__heap_node__carries_reference(gate.dtype) && gate.args[0] != 0ull) {
        if (me < SYS__SCHEDULER__ASKED_BLOCKS) sys__scheduler__zzprivate_gate[me] = gate.args[0];
        else (void)sys__heap_object__release(gate.args[0]);
    }
    /* `init` takes its own holds, so the queue's go back; a half request leaves the base at ERROR, which the post on
     * the next turn says to its answer. */
    /* ⭐ A COMPUTATION THAT MOVED HERE BRINGS WHERE IT STANDS AND WHAT WAS ITS LAST BLOCK'S, and both are this one's now:
     *   its program is under way, run as it stands. */
    sys__scheduler__zzprivate_forget_resume();
    const sys__heap_node carried = taken[SYS__SCHEDULER__REQUEST_CARRIED];
    if (sys__scheduler__zzprivate_unpack(me, &carried)) sys__computing_base__zzpackage_under_way(true);
    if (sys__heap_node__carries_reference(carried.dtype) && carried.args[0] != 0ull)
        (void)sys__heap_object__release(carried.args[0]);
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
    /* ⛔ A PROCESS KILLED WHILE THIS STEP RAN: what it came to is not its answer — SCKL is */
    if (sys__scheduler__zzpackage_killed((uint64_t)sys__scheduler__zzprivate_pid)) {
        sys__scheduler__zzprivate_killed_into(answer);
        return true;
    }
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
    sys__scheduler__zzprivate_forget_resume();         /* a moved request that never began leaves nothing behind */
    sys__scheduler__zzprivate_pid = 0u;                 /* no step runs here: a request made now is a new process's */
    sys__scheduler__zzprivate_inline = 0u;              /* and none is inline: a step that ended inside one ended it */
    sys__scheduler__zzprivate_park_refused = false;
    if (sys__scheduler__zzprivate_park_gate != 0ull) {  /* a park set and never handed on: let go */
        (void)sys__heap_object__release(sys__scheduler__zzprivate_park_gate);
        sys__scheduler__zzprivate_park_gate = 0ull;
    }
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
