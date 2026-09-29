#ifndef SILVANN__ENGINE_OPCODES_CUH
#define SILVANN__ENGINE_OPCODES_CUH
/* ══ WHAT A BLOCK DOES WHEN IT IS NOT EVALUATING ═══════════════════════════════════════════════════════
 * One verb, and it is the other half of the evaluator: `eng__eval` is what a block does with a program,
 * and this is what it does between them.
 *
 * ⭐⭐ THE WHOLE OF THE IDLE STATE IS A COMPUTING BASE'S STATUS WORD, AND NOTHING ELSE EXISTS. A block
 *   waiting for work is a block asking its own base whether anybody has scheduled any; work arriving is
 *   that word saying SCHEDULED; and the answer going back is the same word saying OK. There is no queue,
 *   no message, no wake-up and nothing to register with — ⚖ ARCHITECT: *"a semaphore will do."* What a
 *   scheduler does to reach a block is fill that block's base, which it can do because the bases are an
 *   array every runner can name.
 *
 * ⭐ AND THE WAITING IS THE `else` ARM OF THIS LOOP, BECAUSE THE IDLE PATH REACHES NOTHING ELSE. `begin`
 *   is called only once the peeked status already says SCHEDULED, so a block with nothing to do never
 *   gets near it — the word reads FREE, the third arm takes it, and that arm's back-off is the whole of
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
 * ⛳ IT IS `__noinline__`, which the host environment keeps as a real `noinline`: the parking loop is
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

/* Wait for work, do it, say so, wait again.
 *
 * ⭐⭐ THE ANSWER GOES INTO THE BASE AND NOWHERE ELSE, WHICH IS WHY THIS TAKES NO PLACE TO PUT ONE.
 * ⚖ ARCHITECT: *"reading the result transfers ownership."* A computation reduces to a CELL, the base
 * carries that cell, and whoever scheduled the work collects it — so the block that ran it needs to know
 * nothing about who asked or where they want it. The alternative was an address the scheduler named,
 * which works and makes every dispatch carry a second channel beside the one it already has.
 *
 * ⛔ AND EVERY ITERATION ARMS THE CARVE STATE AND PUTS IT BACK, rather than the launch doing it once.
 * `eng__abi__k_reside` owes the same pairing and does write the watermark back on its way out — but a
 * parked block's way out is whenever somebody outside says stop, and it runs any number of computations
 * before then. A chunk being carved from is tagged so nobody may read its head, and that head stays
 * provisional until the write-back reconciles it, so a launch-scoped pairing alone would leave the chunk
 * claimed and its count unsettled for the whole residency.
 * The pairing that every kernel in this tree owes per launch, this owes per computation. */
static __device__ __noinline__ void eng__opcodes__await_instructions(sys__heap_node* here, uint32_t me,
                                                                     unsigned int* stop) {
    if (here == 0) return;
    /* ⭐ STOP MEANS "WHEN NOTHING IS WAITING", NEVER "NOW". Work put on this base before the stop was given
     * is run first: the stop is read BEFORE the status, and whoever stops the machine scheduled its work
     * before it said stop, so a block that has seen the stop is certain to see the work too. `MEASURED`
     * on the CPU's runner threads a block that left at the stop abandoned a computation
     * block 0 had scheduled on it a moment earlier — a card's blocks spin fast enough that it never showed.
     * ⇒ ★ A RACE ONE SILICON ALWAYS WINS IS STILL A RACE. */
    for (;;) {
        const bool stopping = eng__opcodes__zzprivate_told_to_stop(stop);
        sys__heap__forget_base();
        /* ⭐⭐ ONE READ DECIDES ALL THREE CASES, AND IT IS A READ RATHER THAN A SWAP FOR A REASON WORTH
         * THE PARAGRAPH. Asking by compare-and-swap is harmless against other blocks, because two
         * atomics on one address serialise — and fatal against a writer outside the runners, because a
         * swap STORES the value it read and a plain write is not an atomic read-modify-write that can
         * hold its place.
         * `MEASURED`: a waiter asking that way lost every SCHEDULED the host sent it, while the two
         * operands written beside it on the same node arrived untouched every time.
         * ⇒ ★★ A POLL THAT WRITES IS A POLL THAT OVERWRITES — invisible until there is a second writer,
         *   because until then it only ever puts back what it found.
         *
         * ⛳ SOMEBODY ELSE TOOK THE LAST ANSWER AND HANDED THE BASE BACK is the first case. A scheduler
         * on another block collects in one call and this block never hears about it. Every collection
         * is made by a runner — a door outside the machine launches one to collect — because putting a
         * computation away is reference counting, and the runners are what do the counting
         * (`REASONED`, from the `read_result` call sites in `engine/abi/running.cuh`). */
        const uint32_t state = sys__computing_base__zzengine_peek_status(here);
        if (state == SYS__COMPUTING_BASE__COMPUTING + me) {
            (void)sys__computing_base__zzengine_release_result(here, me);
        } else if (state == SYS__COMPUTING_BASE__SCHEDULED
                && sys__computing_base__begin(here, me) != 0) {
            const uint64_t bindings = here->args[SYS__COMPUTING_BASE__BINDINGS];
            const uint64_t program  = here->args[SYS__COMPUTING_BASE__PROGRAM];
            const sys__heap_node value = eng__eval(bindings, program, here);
            /* Finishing places the answer and says so, in that order and in one act — so there is no
             * moment here where a base says it is done and is not yet holding what it came to. */
            (void)sys__computing_base__finish(here, me, SYS__COMPUTING_BASE__OK, value);
        } else if (stopping) {
            sys__heap__stop_carving();
            return;                          /* told to stop, and nothing here was waiting */
        } else {
            /* Nothing to do, and nothing was touched finding that out. */
            sys__silicon__wait_cycles(SYS__COMPUTING_BASE__POLL_CYCLES);
        }
        sys__heap__stop_carving();
    }
}

#endif /* SILVANN__ENGINE_OPCODES_CUH */
