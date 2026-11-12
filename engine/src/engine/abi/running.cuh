#ifndef SILVANN__ENGINE_ABI_RUNNING_CUH
#define SILVANN__ENGINE_ABI_RUNNING_CUH

/* ══ RUNNING A PROGRAM ════════════════════════════════════════════════════════════════════════════════
 * ⚖ 2026-11-04 the resident machine: every door that runs a program queues it on a block — the block of the
 * worker the caller acts as (`execute`, `eval`), block 0 (`run_grid`), or the one named (`dispatch`) — into an answer
 * the host makes for it, and reads the value out of that answer. The block runs it as it runs any request: it takes it
 * off its queue, claims its own base, thaws its own copy of the picture, evaluates it and files the value
 * (▶ `engine/opcodes.cuh`, `sys`'s `scheduler__header.cuh` and `answer.cuh`). */

/* ── WAITING FOR AN ANSWER ───────────────────────────────────────────────────────────────────────────
 * The host reads the answer's state without a lock, and reads the value once it is there — so threads waiting at once
 * each wait on their own answer, whichever block computes it.
 * ⛳ IT SPINS, THEN NAPS: a computation of microseconds is read within the spin, and a step of tens of milliseconds is
 *   waited for in naps of ENG__ABI__WAIT_NAP_NS, which the reading is late by at most.
 * ⛔ A PANIC ENDS THE WAIT, ENG__ABI__WAIT_PANIC_NS after it: the block that stopped files what its evaluation left as
 *   it ends, and a block that never began it starts nothing (▶ `eng__eval`) — so the run happened, and the caller is
 *   answered nothing and reads the word with `eng_abi_panic`, as a door that ran a program which stopped always
 *   answered. Every door after it refuses. */
#define ENG__ABI__WAIT_SPIN_NS     200000ull    /* 0.2 ms looking before the first nap        */
#define ENG__ABI__WAIT_NAP_NS       50000ull    /* then 50 us a nap                           */
#define ENG__ABI__WAIT_PANIC_NS 1000000000ull   /* and a second for a stopped block's answer  */

static inline uint64_t eng__abi__zzprivate_ns_since(const struct timespec* then) {
    struct timespec now;
    sys__os__clock(&now);
    return (uint64_t)(now.tv_sec - then->tv_sec) * 1000000000ull + (uint64_t)now.tv_nsec - (uint64_t)then->tv_nsec;
}

/* The value `answer` comes to, its hold the caller's, once it is there; nothing where the machine panicked first. */
static inline bool eng__abi__zzprivate_await(uint64_t answer, sys__heap_node* value) {
    struct timespec since, stopped;
    sys__os__clock(&since);
    bool panicked = false;
    for (;;) {
        if (sys__answer__state(answer) == SYS__ANSWER__READY) {
            ENG__ABI__AS_THE_HOST(*value = sys__answer__read(answer));
            return true;
        }
        if (!panicked && eng__eval__panicked() != 0u) { panicked = true; sys__os__clock(&stopped); }
        if (panicked && eng__abi__zzprivate_ns_since(&stopped) > ENG__ABI__WAIT_PANIC_NS) return true;
        if (eng__abi__zzprivate_ns_since(&since) < ENG__ABI__WAIT_SPIN_NS) {
            sys__silicon__wait_cycles(SYS__COMPUTING_BASE__POLL_CYCLES);
        } else {
            const struct timespec nap = { 0, (long)ENG__ABI__WAIT_NAP_NS };
            sys__os__sleep(&nap);
        }
    }
}

/* ── A PROGRAM ON A BLOCK, ANSWERED TO THE HOST ──────────────────────────────────────────────────────
 * Queued into an answer of the host's own, waited for and read: what the program came to, its hold the caller's — an
 * error object among them, as a request that could not begin is answered one (▶ `sys__scheduler__zzengine_post_error`),
 * and nothing where the machine panicked under it. The answer is let go of once it is read. Answers false where it was
 * not queued. */
static inline bool eng__abi__zzprivate_run_on(uint32_t block, uint64_t bindings, uint64_t program,
                                              sys__heap_node* answer) {
    *answer = sys__heap_node__nothing();
    if (block >= eng__abi__zzpackage_machine.blocks) return false;
    /* ⛳ ASKING FOR NOTHING IS A BLANKING, AND HALF A REQUEST A REFUSAL — answered here as a base answers them
     *   (`init`: nothing asked frees it, one of the two is FAULT_HALF), since a queue is no place for either. */
    if (bindings == 0ull && program == 0ull) return true;
    if (bindings == 0ull || program == 0ull) {
        answer->dtype   = SYS__KIND__VALUE_ERROR;
        answer->args[0] = SYS__COMPUTING_BASE__FAULT_HALF;
        return true;
    }
    uint64_t into = 0ull;
    bool queued = false;
    ENG__ABI__AS_THE_HOST(into = sys__answer__create();
                          queued = into != 0ull && sys__scheduler__add(block, bindings, program, into));
    const bool ran = queued && eng__abi__zzprivate_await(into, answer);
    if (into != 0ull) ENG__ABI__AS_THE_HOST((void)sys__heap_object__release(into));
    return ran;
}

/* ── AND THE MACHINE IDLE AGAIN ──────────────────────────────────────────────────────────────────────
 * A grid's door answers once every block has run what its program handed out: each block's base free, its queue
 * empty and nothing parked waiting for a gate, on two looks in a row — a block files and queues before its base comes
 * back free, so work in flight always shows somewhere, and the second look catches what the first passed while it
 * moved. What a program handed out and
 * never read waits in its answer, for as long as somebody holds it. */
static inline bool eng__abi__zzprivate_idle(void) {
    for (uint32_t b = 0u; b < eng__abi__zzpackage_machine.blocks; ++b) {
        const sys__heap_node* base = eng__abi__zzpackage_base(b);
        const uint64_t queue = sys__scheduler__queue(b);
        if (base != 0 && sys__computing_base__zzengine_peek_status(base) != SYS__COMPUTING_BASE__FREE) return false;
        if (queue != 0ull && sys__list__shared_waiting(queue)) return false;
        if (sys__scheduler__zzengine_parked(b) != 0u) return false;          /* a step waiting for its gate */
    }
    return true;
}

static inline void eng__abi__zzprivate_settle(void) {
    struct timespec since;
    sys__os__clock(&since);
    for (uint32_t clean = 0u; clean < 2u && eng__eval__panicked() == 0u; ) {
        clean = eng__abi__zzprivate_idle() ? clean + 1u : 0u;
        if (clean == 2u) break;
        if (eng__abi__zzprivate_ns_since(&since) < ENG__ABI__WAIT_SPIN_NS) {
            sys__silicon__wait_cycles(SYS__COMPUTING_BASE__POLL_CYCLES);
        } else {
            const struct timespec nap = { 0, (long)ENG__ABI__WAIT_NAP_NS };
            sys__os__sleep(&nap);
        }
    }
}

/* The block a door runs on for its caller: the worker the caller's thread acts as (▶ `eng_abi_act_as`), block 0
 * where it acts as nobody — block `w` runs as worker `w`. */
static inline uint32_t eng__abi__zzprivate_callers_block(void) {
    const int w = sys__silicon__host_acting_as();
    return (w > 0 && (uint32_t)w < eng__abi__zzpackage_machine.blocks) ? (uint32_t)w : 0u;
}

/* ⭐ THE WORKER'S FAULT WORD, CARRIED INTO THE MACHINE'S CHANNEL — ⚖ *"fault channel is per worker and
 * cascades"*. A door flags an overflow on the worker's device and does not stop; when a program's answer
 * is collected, the worker is waited for, the word read, and if it is set it is raised ONCE here and
 * cleared. */
static inline void eng__abi__zzprivate_surface_worker_fault(void) {
    /* ⛳ EVERY WORKER'S, ON ITS OWN DEVICE — a fault cascades from whichever worker flagged it — cleared as it is
     *   read, and raised ONCE after: the first flagged word is the one named. The caller's device is bound again
     *   after: the one of the worker this thread is, or acts as.
     * ⛔ IT WAS WORKER 0'S, ALWAYS. `MEASURED`: a thread acting as MI50 1 ran one program right, and its
     *   next program launched on MI50 0 against MI50 1's buffers — a page fault that killed the process. */
    uint32_t first = 0u;
    for (unsigned int w = 0u; w < eng__abi__zzpackage_worker_count; ++w) {
        eng__abi__worker* me = &eng__abi__zzpackage_workers[w];
        if (me->fault_word == 0) continue;
        uint32_t flagged = 0u;
        if (eng__abi__zzpackage_worker_count > 1u && !sys__silicon__bind_device(me->family, me->device)) continue;
        if (!sys__gpu__compute_completed(me->family)
         || !sys__gpu__memory_read(me->family, &flagged, me->fault_word, sizeof flagged)) continue;
        if (flagged == 0u) continue;
        (void)sys__gpu__memory_zerofill(me->family, me->fault_word, sizeof flagged);
        if (first == 0u) first = flagged;
    }
    if (eng__abi__zzpackage_worker_count > 1u) {
        unsigned int self = sys__silicon__worker();
        if (self >= eng__abi__zzpackage_worker_count) self = 0u;
        (void)sys__silicon__bind_device(eng__abi__zzpackage_workers[self].family, eng__abi__zzpackage_workers[self].device);
    }
    /* The word IS the fault's name when a door wrote one (an overflow writes "NPOV"); a bare flag says only
     * that something did, and is raised as the worker's own word. */
    if (first != 0u) sys__fault__raise(0ull, first == 1u ? SYS__VERB_ABI__FAULT_WORKER : (uint64_t)first);
}

extern "C" {

/* Run a program against an environment, both named the way a program names anything — a cell — and answer
 * one. ⛳ THE RETURN SAYS WHETHER THE CALL HAPPENED AND THE ANSWER SAYS WHAT IT CAME TO, which are two
 * questions and not one. Zero means no answer was produced at all — nothing up, nothing queued, or a machine
 * that panicked. One means a block ran the program, and the answer's KIND is where a refusal and a program
 * that came to nothing part company: `VALUE_ERROR` carries the fault word, `VALUE_NULL` is a value a program is
 * allowed to have.
 * ⭐ THE PROGRAM ARRIVES AS A PICTURE, AND THE CALLER GOES ON HOLDING IT. Running one makes a private copy
 * and reduces THAT, so the picture is untouched and the same one may be run again.
 * ⛳ WHAT COMES BACK IS A VALUE. If it names something, that something is the caller's to release; if it
 * is a number or a boolean it names nothing and there is nothing to give back.
 * ⛳ IT RUNS ON THE CALLER'S BLOCK — the worker its thread acts as, block 0 for nobody — which may hand work to
 *   every other block (`sys__scheduler__add`), all of them resident. */
int eng_abi_execute(sys__heap_node bindings, sys__heap_node program, sys__heap_node* answer) {
    if (answer == 0) return 0;
    memset(answer, 0, sizeof *answer);
    answer->dtype = SYS__KIND__VALUE_NULL;
    if (!eng__abi__zzpackage_live()) return 0;
    const bool ran = eng__abi__zzprivate_run_on(eng__abi__zzprivate_callers_block(), bindings.args[0],
                                                program.args[0], answer);
    eng__abi__zzprivate_surface_worker_fault();
    return ran ? 1 : 0;
}

/* The same door for a caller holding offsets rather than cells, which is what a binding for a language
 * with no structs wants. One line over the one above, so there is one door and not two. */
int eng_abi_eval(unsigned long long bindings, unsigned long long program,
                 unsigned int* dtype, unsigned long long* value) {
    sys__heap_node got;
    if (!eng_abi_execute(eng_abi_cell(SYS__KIND__OBJECT_REFERENCE, 0ull, bindings),
                         eng_abi_cell(SYS__KIND__OBJECT_REFERENCE, 0ull, program), &got)) return 0;
    if (dtype) *dtype = (unsigned int)got.dtype;
    if (value) *value = got.args[0];
    return 1;
}

/* ── AND ON A BLOCK THE CALLER NAMES ─────────────────────────────────────────────────────────────────
 * The same, on block `onto`. `outcome` says what happened, so a caller with no value knows whether the block was
 * never booted, the request refused, or the program answered nothing. */
#define ENG__ABI__DISPATCH_OK        0
#define ENG__ABI__DISPATCH_NO_BASE   1   /* the block named has no computing base: it was never started */
#define ENG__ABI__DISPATCH_REFUSED   3   /* the request was not queued, or the machine panicked waiting */

int eng_abi_dispatch(unsigned long long bindings, unsigned long long program, unsigned int onto,
                     unsigned int* dtype, unsigned long long* value, int* outcome) {
    if (outcome) *outcome = ENG__ABI__DISPATCH_NO_BASE;
    if (dtype) *dtype = (unsigned int)SYS__KIND__VALUE_NULL;
    if (value) *value = 0ull;
    if (!eng__abi__zzpackage_live() || onto >= eng__abi__zzpackage_machine.blocks) return 0;
    sys__heap_node got;
    const bool ran = eng__abi__zzprivate_run_on(onto, bindings, program, &got);
    if (ran) eng__abi__zzprivate_settle();
    eng__abi__zzprivate_surface_worker_fault();
    if (outcome) *outcome = ran ? ENG__ABI__DISPATCH_OK : ENG__ABI__DISPATCH_REFUSED;
    if (!ran) return 0;
    if (dtype) *dtype = (unsigned int)got.dtype;
    if (value) *value = got.args[0];
    return 1;
}

/* ── RUNNING A PROGRAM ON THE WHOLE MACHINE ──────────────────────────────────────────────────────────
 * Block 0's, with every other block resident to be handed work by it: the program says where its own work
 * goes, through `(sys__scheduler__add cu bindings actions)`. ⛔ IT ANSWERS ONCE EVERY BLOCK HAS RUN WHAT IT WAS HANDED
 * (▶ `eng__abi__zzprivate_settle`), so what the program set going is done when the caller reads its buffers. A
 * program that hands work and never reads it can answer the answer, which the caller then holds and reads. */
int eng_abi_run_grid(unsigned long long bindings, unsigned long long program,
                     unsigned int* dtype, unsigned long long* value) {
    if (dtype) *dtype = (unsigned int)SYS__KIND__VALUE_NULL;
    if (value) *value = 0ull;
    if (!eng__abi__zzpackage_live()) return 0;
    sys__heap_node got;
    const bool ran = eng__abi__zzprivate_run_on(0u, bindings, program, &got);
    if (ran) eng__abi__zzprivate_settle();
    eng__abi__zzprivate_surface_worker_fault();
    if (!ran) return 0;
    if (dtype) *dtype = (unsigned int)got.dtype;
    if (value) *value = got.args[0];
    return 1;
}

}  /* extern "C" */

#endif /* SILVANN__ENGINE_ABI_RUNNING_CUH */
