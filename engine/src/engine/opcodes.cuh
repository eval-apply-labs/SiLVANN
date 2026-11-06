#ifndef SILVANN__ENGINE_OPCODES_CUH
#define SILVANN__ENGINE_OPCODES_CUH
/* ══ WHAT A BLOCK DOES WHEN IT IS NOT EVALUATING ═══════════════════════════════════════════════════════
 * One verb, and it is the other half of the evaluator: `eng__eval` is what a block does with a program,
 * and this is what it does between them.
 *
 * ⭐⭐ THE WHOLE OF THE IDLE STATE IS A COMPUTING BASE'S STATUS WORD AND THE BLOCK'S QUEUE. A block waiting
 *   for work looks at its queue, which every caller appends to (▶ `sys`'s `scheduler__header.cuh`), and fills
 *   its own base from the oldest request; the base saying SCHEDULED is the next turn's work, and the value
 *   goes into the answer the request carried. There is no message and nothing to register with — ⚖ ARCHITECT:
 *   *"a semaphore will do."* — and the look writes nothing.
 *
 * ⭐ AND THE WAITING IS THE `else` ARM OF THIS LOOP, BECAUSE THE IDLE PATH REACHES NOTHING ELSE. `begin`
 *   is called only once the peeked status already says SCHEDULED, so a block with nothing to do never
 *   gets near it — the word reads FREE, the queue is empty, and the last arm's back-off is the whole of
 *   the waiting. ⛔ IT IS NOT A COPY OF `begin`'s AND DELETING IT AS ONE COSTS EXACTLY WHAT THAT VERB'S
 *   OWN COMMENT NAMES: *"a block refused a BEGIN has nowhere else: it IS the idle loop"*, and without a
 *   back-off *"an idle block polls a word at full rate, and the traffic is the entire cost of having
 *   nothing to do"*. `begin` keeps one for the swap that comes up short — a peek that read SCHEDULED off
 *   a word that does not say it, which is the case that swap exists to catch.
 *
 * ⛔⛔ NOTHING HERE MAY EVER BECOME A RENDEZVOUS, AND THE REASON IS THE WORST FAILURE THIS FAMILY HAS. A
 *   block parked in here has arrived nowhere and is expected nowhere: it polls one word that belongs to
 *   it alone, and a block that never gets work simply never gets work. Put a barrier anywhere a parked
 *   block can reach and the grid strands — one participant waiting for work and another waiting for it to
 *   arrive, with nothing in the runtime to notice. The evaluator has no grid sync, no `__syncthreads` and no
 *   fence but the publishing one, and THAT is what makes parking safe rather than any care taken here.
 *   ⛳ A CARD KERNEL'S LANES DO WAIT FOR EACH OTHER, in one place: nn's combine step
 *   (`nn__silicon__lanes_sum` / `lanes_max`, each family's fold). No block of the evaluator can reach it —
 *   it runs on a card, inside one launch, where every lane of the block calls it.
 *
 * ⛳ IT IS `SYS__NOINLINE`, which the host environment keeps as a real `noinline`: the parking loop is
 *   entered once per launch and stays out of line from the evaluator it calls once per computation.
 *   Nothing on the host prices the choice either way (`REASONED`, not measured).
 * ────────────────────────────────────────────────────────────────────────────────────────────────── */

/* ── HOW A PARKED BLOCK IS TOLD TO STOP ──────────────────────────────────────────────────────────────
 * One word, and it is read through the seam rather than plainly: a plain load of the same address in a
 * loop is a data race the compiler may hoist out of it, and the runner then reads the same answer
 * forever. On the host the seam's read is an atomic load (`cpu/silicon/atomic.cuh`).
 * ⇒ ★ A POLL THAT IS ALLOWED TO BE OPTIMISED IS A POLL THAT CANNOT SUCCEED.
 *
 * ⛔⛔ AND IT READS WITHOUT WRITING, WHICH THE OTHER UNCACHEABLE SPELLINGS DO NOT. Adding zero cannot be
 * cached or hoisted either, and it STORES the value it read — safe only while every writer is another
 * runner's atomic, because two atomics on one address serialise and neither can lose the other's write
 * between its own read and its own store. A write from outside the runners is not an atomic
 * read-modify-write and has no such protection, so a waiter that stores what it read will eventually
 * store over it. ⛔⛔ AND THE CALLER OUTSIDE THE RUNNERS WRITES THIS WORD: `eng__abi__zzpackage_stand_down` puts a 1 in the pinned landing room and side
 * sends it into the same word `eng__abi__k_reside` was launched with, which is this argument, and that
 * is the only way a resident launch ends. So the store is not a hazard waiting for its first outside
 * writer — it can lose the very first stop, and the engine is unstoppable, intermittently, with nothing
 * to point at. */
static __device__ inline bool eng__opcodes__zzprivate_told_to_stop(const unsigned int* stop) {
    return stop != 0 && sys__silicon__read_u32(stop) != 0u;
}

/* ── AND HOW AN IDLE BLOCK WAITS ─────────────────────────────────────────────────────────────────────
 * ⭐ IT LOOKS, THEN DOZES ON ITS DOOR. A block with nothing to do looks at its queue and its base every POLL_CYCLES for
 *   ENG__OPCODES__IDLE_SPIN_NS after its last work, then sleeps on its door until a request is queued for it — every
 *   request rings its block's door (▶ `sys`'s `zzengine_doze`) — or ENG__OPCODES__DOZE_NS pass, and looks again. The
 *   blocks are resident for the machine's life (▶ `engine/abi/residency.cuh`), so a machine nobody is asking anything
 *   of keeps no core busy, and work arriving wakes its block at once.
 * ⛔ THE LOOK IS LONG BECAUSE A SLEEPING NEIGHBOUR SLOWS A ROUND TRIP: with 5 ms of it, a program's
 *   `sys__scheduler__add` and `sys__result` between two working blocks took 8.8 us against 6.9 with every block looking, the two blocks
 *   the bench left idle asleep beside them; with 50 ms, 6.6-7.0 (`MEASURED` node03, review/resident's record). So a
 *   block sleeps only once nothing has asked it for anything for longer than any gap a model's programs leave — a
 *   CPU worker is handed a card's experts every layer, ~1 ms apart on the MI50s, and a card its next step after the
 *   host's turnaround. */
#define ENG__OPCODES__IDLE_SPIN_NS 50000000ull    /* 50 ms looking after the last work      */
#define ENG__OPCODES__DOZE_NS     100000000ull    /* then asleep on the door, 100 ms at most */

static __device__ inline uint64_t eng__opcodes__zzprivate_ns_since(const struct timespec* then) {
    struct timespec now;
    sys__os__clock(&now);
    return (uint64_t)(now.tv_sec - then->tv_sec) * 1000000000ull + (uint64_t)now.tv_nsec - (uint64_t)then->tv_nsec;
}

/* Wait for work, do it, say so, wait again.
 *
 * ⭐⭐ THE VALUE GOES INTO THE ANSWER THE REQUEST CARRIED, WHICH IS WHY THIS TAKES NO PLACE TO PUT ONE. The base holds
 * the answer while the work runs, and the value is filed into it at the end — the answer taking a hold of its own,
 * which every reader is lent from (▶ `sys`'s `answer.cuh`).
 *
 * ⛔ AND EVERY ITERATION ARMS THE CARVE STATE AND PUTS IT BACK, rather than the launch doing it once.
 * `eng__abi__k_reside` owes the same pairing and does write the watermark back on its way out — but a
 * parked block's way out is whenever somebody outside says stop, and it runs any number of computations
 * before then. A chunk being carved from is tagged so nobody may read its head, and that head stays
 * provisional until the write-back reconciles it, so a launch-scoped pairing alone would leave the chunk
 * claimed and its count unsettled for the whole residency.
 * The pairing that every kernel in this tree owes per launch, this owes per computation. */
static __device__ SYS__NOINLINE void eng__opcodes__await_instructions(sys__heap_node* here, uint32_t me,
                                                                     unsigned int* stop) {
    if (here == 0) return;
    /* ⭐ STOP MEANS "WHEN NOTHING IS WAITING", NEVER "NOW". Work put on this base before the stop was given
     * is run first: the stop is read BEFORE the status, and whoever stops the machine scheduled its work
     * before it said stop, so a block that has seen the stop is certain to see the work too. `MEASURED`
     * on the CPU's runner threads a block that left at the stop abandoned a computation
     * block 0 had scheduled on it a moment earlier — a card's blocks spin fast enough that it never showed.
     * ⇒ ★ A RACE ONE SILICON ALWAYS WINS IS STILL A RACE. */
    struct timespec busy;                    /* when this block last had something to do */
    sys__os__clock(&busy);
    for (;;) {
        const bool stopping = eng__opcodes__zzprivate_told_to_stop(stop);
        sys__heap__forget_base();
        /* ⭐⭐ ONE READ DECIDES EVERY CASE, AND IT IS A READ RATHER THAN A SWAP: every idle turn makes it, and a
         * swap would put back what it read over a word written in between (▶ `zzengine_peek_status`).
         * ⛳ NOTHING BUT THIS BLOCK MOVES ITS BASE: every computation it runs comes off its own queue — a
         * program's and the host's alike (▶ `sys`'s `scheduler__header.cuh`) — so its value goes into its answer and the
         * base comes back free here, in this loop. */
        const uint32_t state = sys__computing_base__zzengine_peek_status(here);
        if (state == SYS__COMPUTING_BASE__SCHEDULED && sys__computing_base__begin(here, me) != 0) {
            /* A step that waited for answers finds their names bound in the environment `begin` built for it. */
            sys__scheduler__zzengine_bind_gate(here, me, true);
            const uint64_t bindings = here->args[SYS__COMPUTING_BASE__BINDINGS];
            const uint64_t program  = here->args[SYS__COMPUTING_BASE__PROGRAM];
            const sys__heap_node value = eng__eval(bindings, program, here);
            sys__scheduler__zzengine_bind_gate(here, me, false);   /* left before `finish` lets the environment go */
            /* ⭐ THE VALUE GOES INTO ITS ANSWER FIRST, while this block still holds the base, and the base is put away
             * after it finishes. Finishing places the value and says so, in that order and in one act — so there is no
             * moment here where a base says it is done and is not yet holding what it came to. */
            const bool posted = sys__scheduler__zzengine_post(here, me, value);
            (void)sys__computing_base__finish(here, me, SYS__COMPUTING_BASE__OK, value);
            if (posted) sys__scheduler__zzengine_put_away(here, me);
        } else if (state == SYS__COMPUTING_BASE__ERROR && sys__scheduler__zzengine_post_error(here, me)) {
            /* A scheduled computation that could not begin: its answer told so, as a value, and the base free. */
        } else if (state == SYS__COMPUTING_BASE__FREE && sys__scheduler__zzengine_next(here, me)) {
            /* ⭐ THE OLDEST REQUEST ON THIS BLOCK'S QUEUE IS IN ITS BASE NOW, and the next turn runs it — before a
             * stop, as work already scheduled is. */
        } else if (stopping) {
            sys__heap__stop_carving();
            return;                          /* told to stop, and nothing here was waiting */
        } else {
            /* Nothing to do, and nothing was touched finding that out: a short wait, or a nap once quiet. */
            sys__heap__stop_carving();
            if (eng__opcodes__zzprivate_ns_since(&busy) < ENG__OPCODES__IDLE_SPIN_NS) {
                sys__silicon__wait_cycles(SYS__COMPUTING_BASE__POLL_CYCLES);
            } else {
                sys__scheduler__zzengine_doze(me, ENG__OPCODES__DOZE_NS);
            }
            continue;
        }
        sys__os__clock(&busy);
        sys__heap__stop_carving();
    }
}

#endif /* SILVANN__ENGINE_OPCODES_CUH */
