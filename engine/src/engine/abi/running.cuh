#ifndef SILVANN__ENGINE_ABI_RUNNING_CUH
#define SILVANN__ENGINE_ABI_RUNNING_CUH

/* ══ RUNNING A PROGRAM ════════════════════════════════════════════════════════════════════════════════
 * The state machine that owns a computation, and the doors that run a program through it: on the
 * block the host talked to (`execute`, `eval`), on another block (`dispatch`), and on the whole
 * machine (`run_grid`). */

/* ── RUNNING ONE, THROUGH THE STATE MACHINE THAT OWNS A COMPUTATION ──────────────────────────────────
 * ⭐⭐ THE SIX STEPS ARE THE WHOLE OF IT, AND EVERY ONE OF THEM IS A TRANSITION SOMEBODY ELSE COULD BE
 *   RACING FOR. A block CLAIMS a base, is FILLED with work, BEGINS on it — which is where it thaws its own
 *   copy — runs, says how it went, and hands the result over. Nothing here is a convenience wrapper around
 *   the evaluator: the evaluator is one line in the middle, and the rest is what makes a computation a
 *   thing that can be scheduled onto one block out of many rather than simply called.
 *
 * ⭐ WHAT ARRIVES IS A PICTURE AND NOT A LIST, AND THERE IS NO SECOND DOOR FOR ONE-SHOT WORK. `MEASURED`
 *   with the evaluator on a card, for the call form `(fib 12)`: a freeze and a thaw together cost 0.074%
 *   of evaluating it, so a door that skipped them would buy a rounding error and cost a second shape for
 *   every caller to learn. ▶ `measurements/2026-09-12_S94_picture_cost_at_n1.md`.
 *
 * ⛳ THE ANSWER IS TAKEN OUT BEFORE THE COMPUTATION IS PUT AWAY. Evaluation REDUCES the program to what it
 *   became, so the answer is a cell INSIDE the object the base is naming — and collecting the result is
 *   what lets that object go. Anything the answer names is therefore retained first and belongs to the
 *   caller afterwards, which is the same rule everything else this interface hands out follows.
 *
 * ⭐ A REFUSAL ANSWERS THE REASON, AND IT ANSWERS IT THROUGH THE ANSWER RATHER THAN THROUGH THE LOG. A
 *   failed `init` or `begin` leaves a fault CODE in the base's result slot — a word and not an object, so
 *   there is nothing to release and collecting is only how the base gets back to FREE. The code is read
 *   with `sys__computing_base__fault` BEFORE collecting, because a base that is FREE says nothing about
 *   what it was, and it reaches the caller as the answer's KIND: `VALUE_ERROR` with the fault word in
 *   `args[0]`, which is a different answer from the one a program that ran and came to nothing gives.
 * ⇒ ★ AN ERROR IS A VALUE, AND THIS IS THE DOOR WHERE THAT PAYS — a caller reads the kind it was handed
 *   instead of asking a second channel why nothing came back. The refusal arm below carries the ruling. */
/* ⛔⛔ THE TWO HALVES ARE `__noinline__` ON PURPOSE, AND IT IS A RATE DECISION RATHER THAN A STYLE ONE.
 *   Inlined beside the evaluator, the state machine's own values stay live ACROSS the evaluation, for
 *   six transitions that run once per program. Given their own frames, what they need is theirs and the
 *   evaluator's frame is untouched. `REASONED` — the A/B of the two arms on this build is unmeasured, and
 *   a reader checking it measures both against each other on the tree in front of them.
 * ⇒ ★ A COST PAID ONCE CAN STILL BE PAID PER-INSTRUCTION IF IT IS PAID IN REGISTERS. */
static __device__ __noinline__ uint64_t eng__abi__zzprivate_start(sys__heap_node* here, uint32_t me,
                                                                  uint64_t bindings, uint64_t picture,
                                                                  uint64_t* why) {
    *why = 0ull;
    if (sys__computing_base__claim(here, me) == 0) return 0ull;
    if (!sys__computing_base__init(here, me, bindings, picture)
     || sys__computing_base__begin(here, me) == 0) {
        /* Either one leaves the base at ERROR with the reason in its result slot; collecting is how it
         * gets back to FREE, and a base left anywhere else is one no later call can use.
         * ⛔ AND NOTHING IS RELEASED, BECAUSE THERE IS NOTHING TO RELEASE. At ERROR the slot holds a
         * fault CODE and `read_result` answers zero for exactly that reason — so releasing its answer
         * would hand zero to the counting, which is not an address and raises. The reason is read
         * without taking anything, which is what `sys__computing_base__fault` is for — and it has to be
         * read BEFORE collecting, because collecting is what puts the base back to FREE and a free base
         * says nothing about what it was. */
        *why = sys__computing_base__fault(here);
        (void)sys__computing_base__read_result(here, me);   /* nothing comes back at ERROR; this frees it */
        return 0ull;
    }
    return here->args[SYS__COMPUTING_BASE__PROGRAM];
}

/* ⭐ THE ANSWER TRAVELS THROUGH THE BASE RATHER THAN PAST IT, WHICH IS WHAT MAKES THIS ONE SHAPE FOR
 * EVERY CALLER. Finishing places the value and says so; collecting takes it and empties the slot, so the
 * hold the base was keeping becomes this caller's and nothing is retained or released out here. That is
 * the same two steps a scheduler on another block performs, which is the point: the block that ran the
 * work does not know whether it was asked by the host or by another block. */
static __device__ __noinline__ sys__heap_node eng__abi__zzprivate_collect(sys__heap_node* here, uint32_t me,
                                                                         sys__heap_node value) {
    (void)sys__computing_base__finish(here, me, SYS__COMPUTING_BASE__OK, value);
    return sys__computing_base__read_result(here, me);
}

/* The whole of what running a program does, lifted out of the launches that run it.
 * ⭐ IT IS A SEPARATE FUNCTION SO THAT EACH OF THEM HAS EXACTLY ONE WAY OUT. Three of the paths here
 * return early, and a launch owes `sys__heap__stop_carving()` on every one of them — the heap declares
 * it owed by every runner that carves, and a launch that ends without it arms the next one from a stale
 * head — an obligation that is impossible to meet at three `return`s and trivial to meet at one call.
 * The shape is the guarantee. */
static __device__ inline void eng__abi__zzprivate_run(uint64_t bindings, uint64_t picture,
                                                            sys__heap_node* answer) {
    *answer = sys__heap_node__nothing();
    sys__heap_node* here = eng__abi__zzpackage_base((uint32_t)sys__silicon__block_id());
    if (here == 0) return;
    const uint32_t me  = (uint32_t)sys__silicon__block_id();
    uint64_t why = 0ull;
    const uint64_t run = eng__abi__zzprivate_start(here, me, bindings, picture, &why);
    if (run == 0ull) {
        /* ⭐⭐ A REFUSAL ANSWERS THE REASON, AND A PROGRAM THAT ANSWERED NOTHING ANSWERS NOTHING. Those
         * were the same value until now — both `VALUE_NULL` — so a caller could not tell "it did not
         * run" from "it ran and came to nothing". ⚖ ARCHITECT: *"the null answer is null and a refusal
         * will be the error message ... while a computation that returns null returns null."*
         * ⛳ AND THE NODE IS FILLED RATHER THAN ALLOCATED, which is what makes this usable on the one
         * path that fails BECAUSE it could not allocate: the answer is a node this side already owns, a
         * fault is a word, and `ERROR` is in neither the reference set nor the substructure set — so
         * nothing downstream will follow `args[0]` as an offset. It is a value the caller reads. */
        if (why != 0ull) {
            answer->dtype   = SYS__KIND__VALUE_ERROR;
            answer->args[0] = why;
        }
        return;
    }

    const sys__heap_node value = eng__eval(bindings, run, here);
    *answer = eng__abi__zzprivate_collect(here, me, value);
}

/* ── ONE BLOCK PUTTING WORK ON ANOTHER ───────────────────────────────────────────────────────────────
 * Every block but the first parks and waits to be given something; the first gives it to one of them,
 * waits for the answer and then tells everybody to stop.
 *
 * ⭐⭐ THE PROTOCOL IS THE COMPUTING BASE'S AND NOTHING IS ADDED TO IT. `claim` takes a base — any base,
 *   named by whoever is scheduling — `init` fills it and leaves it SCHEDULED, the owner's `begin` takes
 *   SCHEDULED and thaws its own private copy of the program, and `read_result` is how the scheduler puts
 *   the base back to FREE afterwards. Every one of those verbs already names the block it is acting for,
 *   which is what makes a scheduler and an owner two different blocks rather than one block twice.
 *
 * ⭐ AND THE THAW IS WHY THIS IS WORTH DOING RATHER THAN SIMPLY CALLING THE EVALUATOR. What is scheduled
 *   is a PICTURE, which nobody may write; `begin` makes the running copy, carved from the arena of the
 *   block that will run it. So the work and the memory it churns both end up where the work is.
 *
 * ⛔⛔ AND THE FIRST BLOCK WAITS ON A COMPARE-AND-SWAP RATHER THAN ON A READ. `read_result` answers
 *   nothing until the base is finished and takes it when it is, so polling it IS the wait — and because
 *   it is an atomic, neither the compiler nor a cache can turn the poll into the same answer forever. A
 *   plain read of the status word would be both, silently.
 *   ⚠ WHAT IT CANNOT TELL APART is "not finished yet" from "finished with nothing to hand over", since
 *   both answer zero. That is why the loop is BOUNDED: a program that answers nothing would otherwise
 *   spin here forever, and this tree's own warning is that nothing detects it.
 *
 * ⛳ THE BOUND IS MEASURED AND NOT REASONED, because a try's cost is the pause loop's and a guess at it
 *   can be wrong by orders of magnitude. `MEASURED` on `gtasanandrea`, `cpu/silicon/atomic.cuh`'s pause
 *   loop: wait_cycles(1) 33.6 ns · wait_cycles(64) 1282.3 ns · wait_cycles(256) 5142.6 ns. At
 *   `POLL_CYCLES` 64 a try is ~1.28 us, so the figure below is about fifteen seconds — generous beside
 *   anything this engine has evaluated and short enough that reaching it is something somebody notices.
 *   ▶ `measurements/2026-09-23_S107_the_launch_seam.md`.
 * ⇒ ★ A BOUND NOBODY HAS PRICED IS NOT A BOUND. */
#define ENG__ABI__DISPATCH_TRIES 12000000u   /* x ~1.28 us on the host = about fifteen seconds */

#define ENG__ABI__DISPATCH_OK        0
#define ENG__ABI__DISPATCH_NO_BASE   1   /* the block named has no computing base: it was never started */
#define ENG__ABI__DISPATCH_BUSY      2   /* its base is not free — somebody else is already on it       */
#define ENG__ABI__DISPATCH_REFUSED   3   /* the fill was refused, and the base is put back before this  */
#define ENG__ABI__DISPATCH_TIMED_OUT 4   /* the bound above ran out with nothing to collect             */

static __global__ ENG_BOOT_BLOCK void eng__abi__k_dispatch(uint64_t bindings, uint64_t picture,
                                                           uint32_t onto, unsigned int* stop,
                                                           sys__heap_node* answer, int* outcome) {
    const uint32_t me = (uint32_t)sys__silicon__block_id();

    if (me != 0u) {
        /* ⛳ THE PARKED BLOCKS ARM AND PUT BACK INSIDE THE LOOP, once per computation rather than once
         * per launch, because a block that is waiting has not finished and a block that has finished is
         * waiting again. There is nothing for this kernel to pair around. */
        eng__opcodes__await_instructions(eng__abi__zzpackage_base(me), me, stop);
        return;
    }

    sys__heap__forget_base();
    if (answer != 0) *answer = sys__heap_node__nothing();
    int why = ENG__ABI__DISPATCH_NO_BASE;
    sys__heap_node* there = eng__abi__zzpackage_base(onto);
    if (there != 0) {
        why = ENG__ABI__DISPATCH_BUSY;
        if (sys__computing_base__claim(there, me) != 0) {
            if (!sys__computing_base__init(there, me, bindings, picture)) {
                /* Held by us and holding nothing. Ending it at ERROR is what lets it be collected, and
                 * collecting is the only route back to FREE — a base left mid-state is one no later
                 * dispatch can use and nothing would say why. */
                (void)sys__computing_base__finish(there, me, SYS__COMPUTING_BASE__ERROR,
                                                  sys__heap_node__nothing());
                (void)sys__computing_base__read_result(there, me);
                why = ENG__ABI__DISPATCH_REFUSED;
            } else {
                why = ENG__ABI__DISPATCH_TIMED_OUT;
                for (unsigned int k = 0u; k < ENG__ABI__DISPATCH_TRIES; ++k) {
                    const sys__heap_node got = sys__computing_base__read_result(there, me);
                    if (got.dtype != SYS__KIND__VALUE_NULL || got.args[0] != 0ull) {
                        *answer = got;
                        why = ENG__ABI__DISPATCH_OK;
                        break;
                    }
                    sys__silicon__wait_cycles(SYS__COMPUTING_BASE__POLL_CYCLES);
                }
            }
        }
    }
    *outcome = why;
    /* ⛔ AND THE STOP IS THE LAST THING, ON EVERY PATH INCLUDING THE ONES THAT FAILED. A parked block is
     * waiting on this word and on nothing else, so a scheduler that gives up without writing it leaves
     * the launch running with nobody left to end it. */
    if (stop != 0) (void)sys__silicon__cas_u32(stop, 0u, 1u);
    sys__heap__stop_carving();
}

/* The runner a launch of more than one block starts it through. */
typedef struct {
    uint64_t        bindings;
    uint64_t        picture;
    uint32_t        onto;
    unsigned int*   stop;
    sys__heap_node* answer;
    int*            outcome;
} eng__abi__k_dispatch__args;
static void eng__abi__k_dispatch__run(const void* at) {
    const eng__abi__k_dispatch__args* a = (const eng__abi__k_dispatch__args*)at;
    eng__abi__k_dispatch(a->bindings, a->picture, a->onto, a->stop, a->answer, a->outcome);
}

/* ── AND THE SAME GRID, WITH THE PROGRAM DOING THE DISPATCHING ───────────────────────────────────────
 * Block zero runs the program on its OWN base while every other block parks; what the program does with
 * them is the program's business.
 *
 * ⭐⭐ THIS IS THE DOOR THE LANGUAGE ACTUALLY WANTS, and the one above is the scaffolding that proved the
 * protocol. There, the ENGINE decided which block got the work; here a program says `(sys__compute …)`
 * and decides for itself — which is the whole point of the verb, and the only shape that survives into a
 * resident engine where nothing outside is choosing anything.
 *
 * ⛳ BLOCK ZERO'S OWN COMPUTATION IS AN ORDINARY ONE. It claims its base, is filled, begins and finishes
 * exactly as a dispatched block does; the only thing that makes it the first among them is that the host
 * handed it its work rather than another block. */
static __global__ ENG_BOOT_BLOCK void eng__abi__k_grid(uint64_t bindings, uint64_t picture,
                                                        unsigned int* stop, sys__heap_node* answer) {
    const uint32_t me = (uint32_t)sys__silicon__block_id();

    if (me != 0u) {
        eng__opcodes__await_instructions(eng__abi__zzpackage_base(me), me, stop);
        return;
    }

    sys__heap__forget_base();
    eng__abi__zzprivate_run(bindings, picture, answer);
    /* ⛔ AND THE STOP IS WRITTEN WHATEVER HAPPENED, including a program that refused. A parked block is
     * waiting on this word and on nothing else, so a first block that gives up without writing it leaves
     * the launch running with nobody left to end it. */
    if (stop != 0) (void)sys__silicon__cas_u32(stop, 0u, 1u);
    sys__heap__stop_carving();
}

/* The runner a launch of more than one block starts it through. */
typedef struct {
    uint64_t        bindings;
    uint64_t        picture;
    unsigned int*   stop;
    sys__heap_node* answer;
} eng__abi__k_grid__args;
static void eng__abi__k_grid__run(const void* at) {
    const eng__abi__k_grid__args* a = (const eng__abi__k_grid__args*)at;
    eng__abi__k_grid(a->bindings, a->picture, a->stop, a->answer);
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
 * questions and not one. Zero means no answer was produced out here at all — nowhere to put one, or a
 * machine not free to launch. One means a block ran the program, and
 * the answer's KIND is where a refusal and a program that came to nothing part company: `VALUE_ERROR`
 * carries the fault word, `VALUE_NULL` is a value a program is allowed to have. ⇒ ★ the caller reads the
 * kind it was handed instead of asking a second channel why nothing came back, which is the ruling the
 * refusal arm above carries.
 * ⭐ THE PROGRAM ARRIVES AS A PICTURE, AND THE CALLER GOES ON HOLDING IT. Running one makes a private copy
 * and reduces THAT, so the picture is untouched and the same one may be run again — which is the whole
 * difference from handing a list over: a list run twice is gone after the first time.
 * ⛳ WHAT COMES BACK IS A VALUE. If it names something, that something is the caller's to release; if it
 * is a number or a boolean it names nothing and there is nothing to give back. */
int eng_abi_execute(sys__heap_node bindings, sys__heap_node program, sys__heap_node* answer) {
    if (answer == 0) return 0;
    memset(answer, 0, sizeof *answer);
    answer->dtype = SYS__KIND__VALUE_NULL;
    if (!eng__abi__zzpackage_free_to_launch()) return 0;
    ENG__ABI__ON_THE_MACHINE(eng__abi__zzprivate_run(bindings.args[0], program.args[0], answer));
    eng__abi__zzprivate_surface_worker_fault();
    return 1;
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

/* ── AND THE SAME PROGRAM, RUN BY A BLOCK THAT IS NOT THE ONE THE HOST TALKED TO ──────────────────────
 * Launch every block the machine was booted with, have the first one put this program on the block named,
 * and bring back what it answered. `outcome` says which of the five things happened, so a caller that got
 * no value knows whether it was refused, timed out, or simply answered nothing.
 *
 * ⭐ WHAT THIS DOOR IS FOR IS THE DISPATCH AND NOT THE ANSWER. Every program here could be run by the
 *   ordinary door at a fraction of the cost — a runner per block is started to keep one busy. What it
 *   demonstrates is the path a program will take once the dispatch comes from INSIDE a running program
 *   rather than from out here, and at that point there is no launch per piece of work at all.
 *
 * ⛔ THE LAUNCH ENDS, WHICH IS WHY THE ANSWER CAN COME BACK THE ORDINARY WAY. The parked blocks stop when
 *   the first one tells them to, so the runners are joined like every other launch above and the value
 *   is read back after. A launch that OUTLIVES the call is `eng_abi_reside`, and it is read through the
 *   side channel and `sys_abi_peek` while its runners work, because there is no join to read after when
 *   the work has not finished and is not going to.
 *
 * ⛳ THE TWO SMALL WORDS ARE TAKEN AND GIVEN BACK PER CALL rather than held with the machine. They are a
 *   property of one dispatch and not of the engine, and a door used once per program has nothing to gain
 *   from keeping them. */
int eng_abi_dispatch(unsigned long long bindings, unsigned long long program, unsigned int onto,
                     unsigned int* dtype, unsigned long long* value, int* outcome) {
    if (outcome) *outcome = ENG__ABI__DISPATCH_NO_BASE;
    if (dtype) *dtype = (unsigned int)SYS__KIND__VALUE_NULL;
    if (value) *value = 0ull;
    if (!eng__abi__zzpackage_free_to_launch()) return 0;
    if (onto >= eng__abi__zzpackage_machine.blocks) return 0;

    unsigned int* stop = 0;
    int*          said = 0;
    if (!sys__gpu__memory_allocate(eng__abi__zzpackage_machine.family, (void**)&stop, sizeof(unsigned int)) ||
        !sys__gpu__memory_allocate(eng__abi__zzpackage_machine.family, (void**)&said, sizeof(int))          ||
        !sys__gpu__memory_zerofill(eng__abi__zzpackage_machine.family, stop, sizeof(unsigned int))         ||
        !sys__gpu__memory_zerofill(eng__abi__zzpackage_machine.family, said, sizeof(int))) {
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, stop);
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, said);
        return 0;
    }

    const eng__abi__k_dispatch__args args = { bindings, program, onto, stop, eng__abi__zzpackage_answer, said };
    ENG__LAUNCH_MANY(eng__abi__k_dispatch__run, eng__abi__zzpackage_machine.blocks, &args);

    sys__heap_node got;
    int            why = ENG__ABI__DISPATCH_TIMED_OUT;
    const bool ran = eng__abi__zzpackage_collect_answer(&got, sizeof got) &&
                     sys__gpu__memory_read(eng__abi__zzpackage_machine.family, &why, said, sizeof why);
    eng__abi__zzprivate_surface_worker_fault();
    sys__gpu__memory_free(eng__abi__zzpackage_machine.family, stop);
    sys__gpu__memory_free(eng__abi__zzpackage_machine.family, said);
    if (!ran) return 0;

    if (outcome) *outcome = why;
    if (why != ENG__ABI__DISPATCH_OK) return 0;
    if (dtype) *dtype = (unsigned int)got.dtype;
    if (value) *value = got.args[0];
    return 1;
}

/* ── RUNNING A PROGRAM ON THE WHOLE MACHINE ──────────────────────────────────────────────────────────
 * The same as `execute`, except that every block the machine was booted with is launched and the ones
 * this program does not run on wait to be given work by it.
 *
 * ⭐ IT DIFFERS FROM `dispatch` IN WHO CHOOSES. There the engine picks a block and schedules; here the
 * PROGRAM does, through `(sys__compute cu bindings actions)`, and this only makes sure the blocks it
 * might name are there to be named. ⛳ Which is why it takes no block number: it is not scheduling
 * anything.
 *
 * ⛔ THE LAUNCH STILL ENDS WHEN THE PROGRAM DOES, so a program that dispatches and never collects leaves
 * that block's base held and the answer uncollected — the base is put back by `(sys__result cu)` and by
 * nothing else. That is the same obligation the C door above meets by polling, moved to where it belongs
 * once a program can express it. */
int eng_abi_run_grid(unsigned long long bindings, unsigned long long program,
                     unsigned int* dtype, unsigned long long* value) {
    if (dtype) *dtype = (unsigned int)SYS__KIND__VALUE_NULL;
    if (value) *value = 0ull;
    if (!eng__abi__zzpackage_free_to_launch()) return 0;

    unsigned int* stop = 0;
    if (!sys__gpu__memory_allocate(eng__abi__zzpackage_machine.family, (void**)&stop, sizeof(unsigned int)) ||
        !sys__gpu__memory_zerofill(eng__abi__zzpackage_machine.family, stop, sizeof(unsigned int))) {
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, stop);
        return 0;
    }
    const eng__abi__k_grid__args args = { bindings, program, stop, eng__abi__zzpackage_answer };
    ENG__LAUNCH_MANY(eng__abi__k_grid__run, eng__abi__zzpackage_machine.blocks, &args);

    sys__heap_node got;
    const bool ran = eng__abi__zzpackage_collect_answer(&got, sizeof got);
    eng__abi__zzprivate_surface_worker_fault();
    sys__gpu__memory_free(eng__abi__zzpackage_machine.family, stop);
    if (!ran) return 0;
    if (dtype) *dtype = (unsigned int)got.dtype;
    if (value) *value = got.args[0];
    return 1;
}

}  /* extern "C" */

#endif /* SILVANN__ENGINE_ABI_RUNNING_CUH */
