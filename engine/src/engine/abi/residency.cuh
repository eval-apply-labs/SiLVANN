#ifndef SILVANN__ENGINE_ABI_RESIDENCY_CUH
#define SILVANN__ENGINE_ABI_RESIDENCY_CUH

/* ══ STAYING UP — RESIDENCY, AND SCHEDULING FROM OUTSIDE ══════════════════════════════════════════════
 * The launch that does not end, and the doors that start and stop it, place a program on a parked
 * block, poll for its answer and let the answer go. */

/* ── AND THE LAUNCH THAT DOES NOT END ────────────────────────────────────────────────────────────────
 * ⭐⭐ IT IS THE WHOLE POINT OF THE OTHER TWO LAUNCHES, `dispatch` AND `run_grid`. There, one block ran a
 *   program and then told everybody to stop, so the launch lasted exactly as long as the work the host
 *   had already decided on. Here NO block is special and nothing tells anybody to stop, so the launch
 *   lasts until somebody outside says so — and between those two moments the host can put work on any
 *   block, collect it, and put more on.
 * ⛳ BLOCK ZERO PARKS LIKE THE REST, which is what makes a one-block machine still useful: there is no
 *   block reserved to be the scheduler, because the scheduler is the caller, outside the runners. */
/* ⛳ IT ARMS AND FLUSHES AROUND A CALL THAT ALREADY DOES BOTH PER COMPUTATION, and that is not
 * redundant: a block whose base could not be found never enters the loop at all, and a runner that
 * returns without flushing leaves a watermark for the next launch to carve from. The pairing is owed by
 * the entry a launch runs, so that is where it is written. */
static __global__ ENG_BOOT_BLOCK void eng__abi__k_reside(unsigned int* stop) {
    sys__heap__forget_base();
    const uint32_t me = (uint32_t)sys__silicon__block_id();
    eng__opcodes__await_instructions(eng__abi__zzpackage_base(me), me, stop);
    sys__heap__stop_carving();
}

/* The runner a launch of more than one block starts it through. */
typedef struct {
    unsigned int* stop;
} eng__abi__k_reside__args;
static void eng__abi__k_reside__run(const void* at) {
    const eng__abi__k_reside__args* a = (const eng__abi__k_reside__args*)at;
    eng__abi__k_reside(a->stop);
}

extern "C" {

/* ── LOOKING INSIDE IS sys's ─────────────────────────────────────────────────────────────────────────
 * ⚖ *"the looking inside do feel like they are sys"*: a node by address, an offset as an address, where a
 * block's base is, the allocator's counters and the fault channel are `sys`'s C interface
 * (`packages/sys/contracts/abi/cpu.cuh`), and bytes in a buffer are nn's (`nn/contracts/abi/cpu.cuh`).
 * The doors below use `sys_abi_peek` to look at a base they are about to change. */


/* ── STAYING UP ──────────────────────────────────────────────────────────────────────────────────────
 * Start a runner per block into its waiting loop and DO NOT join them. From here until it is told
 * to stop, the engine is a machine somebody talks to rather than a call somebody makes.
 *
 * ⭐⭐ WHAT THIS CHANGES IS NOT SPEED, IT IS WHO DECIDES WHEN WORK ARRIVES. Every door above settles, so
 *   the work a launch would do had to be known before the launch began; a program could dispatch to
 *   another block, but only work the host had already handed in. A launch that outlives the call takes
 *   work that did not exist when it started.
 *
 * ⛔⛔ AND EVERY SETTLING DOOR REFUSES WHILE IT IS UP, WHICH IS A REAL CONSTRAINT AND IS STATED RATHER
 *   THAN DISCOVERED. Build the programs, freeze them, take the addresses — then reside. Asking for a
 *   list while every block's runner is parked would make the caller's thread a second runner zero beside
 *   the one already there, and this file answers no instead. ⛳ What stays open is everything that goes
 *   by the side channel, which is the whole scheduling vocabulary.
 *
 * ⛔ THERE IS NO BOUND ON A RESIDENT LAUNCH AND THERE MUST NOT BE ONE — it is supposed to outlast any
 *   particular piece of work, so a timer would be a machine that stops for no reason. What bounds it is
 *   the caller, and `shutdown` owes the same act so that forgetting is survivable. The stop word is read
 *   without being written, which is what makes a host write to it stick; a waiter that swapped would
 *   overwrite it and the engine would be unstoppable, intermittently. */
int eng_abi_reside(void) {
    if (!eng__abi__zzpackage_free_to_launch()) return 0;
    if (!sys__gpu__memory_allocate(eng__abi__zzpackage_machine.family,
                                 (void**)&eng__abi__zzpackage_stop, sizeof(unsigned int)) ||
        !sys__gpu__memory_zerofill(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_stop, sizeof(unsigned int))) {
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_stop);
        eng__abi__zzpackage_stop = 0;
        return 0;
    }
    /* ⛳ CLEARED BEFORE THE RUNNERS START, and starting a thread orders everything written before it
     * ahead of what the thread reads, so no runner can read a stop word that says stop before anybody has
     * said it. */
    const eng__abi__k_reside__args args = { eng__abi__zzpackage_stop };
    if (!ENG__LAUNCH_RESIDENT(eng__abi__k_reside__run, eng__abi__zzpackage_machine.blocks, &args, sizeof args)) {
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_stop);
        eng__abi__zzpackage_stop = 0;
        return 0;
    }
    eng__abi__zzpackage_resident = 1;
    return 1;
}

int eng_abi_residing(void) { return eng__abi__zzpackage_resident; }

/* Tell it to stop and wait for it. ⛳ IT ANSWERS NO BY CHANGING NOTHING: a word that did not go leaves
 * the launch out there, and the caller may ask again. */
int eng_abi_stop_residing(void) {
    if (!eng__abi__zzpackage_up() || eng__abi__zzpackage_resident == 0) return 0;
    return eng__abi__zzpackage_stand_down() ? 1 : 0;
}

/* ── SCHEDULING FROM OUTSIDE THE RUNNERS ─────────────────────────────────────────────────────────────
 * Three doors, and between them they are the whole of what the host does to a block that is already
 * running: put work on it, ask how it is going, and give the base back when the answer has been taken.
 * They are the same three acts a PROGRAM performs with `(sys__compute …)`, `(sys__completed …)` and
 * `(sys__result …)` — which is the point, and why nothing in the state machine had to grow to take them.
 *
 * ⭐⭐ THEY TAKE AN ADDRESS AND NOT A BLOCK NUMBER. Turning a block number into an address is `sys`'s —
 *   `sys_abi_base_address`, a plain read that answers while the runners work — so these doors take what
 *   it answered and touch nothing but the base itself. Asking it first is not a step a caller can
 *   forget: there is nothing else to pass.
 *
 * ⛔⛔ AND `place` CONSUMES BOTH OF ITS OPERANDS, WHICH IS THE ONE THING A CALLER MUST GET RIGHT. Filling
 *   a base from a runner takes a HOLD of its own on the picture and on the environment — `init` does it
 *   with two atomics — and these doors take none: they run nothing of `sys` while the runners work, only
 *   send words down the side channel. So the hold the caller already has becomes the base's instead,
 *   which costs nothing and balances exactly: `begin` releases the picture when it has thawed its copy,
 *   and putting the computation away releases the environment. `REASONED`. ⇒ ★ WHERE A PARTICIPANT DOES
 *   NOT ACQUIRE, THE SOUND ARRANGEMENT IS TO TRANSFER. A caller that wants to place one picture on two
 *   blocks needs a second picture, and freezing again is how it gets one.
 *
 * ⛔ ONE HOST, AND THAT IS STATED RATHER THAN ASSUMED. Taking a base from a runner is a compare-and-swap,
 *   so two blocks racing for one is decided; a write from out here is a WRITE, so two schedulers racing
 *   for one base is not. A base the host places on is the host's, and a program that also dispatches
 *   there is doing something this design does not yet support. The refusal below catches the ordinary
 *   case — a base still busy — and not that one.
 *
 * ⛳ THE FILL IS ORDERED BEFORE THE FLAG BY THE CHANNEL ITSELF. The side channel fences each write
 *   release, so the four words sent first are visible to a runner that reads the status after them — the
 *   publish a runner does by hand on its side of this hand-over is, out here, a property of sending the
 *   words first. ▶ `silicon_families/host/sys.cuh`. */
static inline void* eng__abi__zzprivate_slot(unsigned long long address, unsigned int slot) {
    return (void*)(uintptr_t)(address + (unsigned long long)offsetof(sys__heap_node, args)
                            + (unsigned long long)slot * (unsigned long long)sizeof(uint64_t));
}

int eng_abi_place(unsigned long long address, unsigned long long bindings, unsigned long long program) {
    if (!eng__abi__zzpackage_up() || address == 0ull) return 0;
    sys__heap_node now;
    if (!sys_abi_peek(address, &now)) return 0;
    if ((unsigned int)now.args[SYS__COMPUTING_BASE__STATUS] != SYS__COMPUTING_BASE__FREE) return 0;
    if ((bindings == 0ull) != (program == 0ull)) return 0;

    /* The landing room doubles as the outgoing buffer: it is a node, the four words being written are
     * four of its own slots, and a second allocation for 32 bytes would be a second thing to get wrong
     * at shutdown. */
    sys__heap_node* out = eng__abi__zzpackage_landing;
    out->args[SYS__COMPUTING_BASE__BINDINGS]    = bindings;
    out->args[SYS__COMPUTING_BASE__PROGRAM]     = program;
    out->args[SYS__COMPUTING_BASE__STATUS]      = (uint64_t)SYS__COMPUTING_BASE__FREE;
    out->args[SYS__COMPUTING_BASE__RESULT_KIND] = 0ull;
    if (!sys__gpu__side_memory_write(eng__abi__zzpackage_machine.family,
                                      eng__abi__zzprivate_slot(address, SYS__COMPUTING_BASE__BINDINGS),
                                        &out->args[SYS__COMPUTING_BASE__BINDINGS],
                                        4u * sizeof(uint64_t), eng__abi__zzpackage_channel))
        return 0;

    out->args[SYS__COMPUTING_BASE__STATUS] = (uint64_t)SYS__COMPUTING_BASE__SCHEDULED;
    return sys__gpu__side_memory_write(eng__abi__zzpackage_machine.family,
                                        eng__abi__zzprivate_slot(address, SYS__COMPUTING_BASE__STATUS),
                                          &out->args[SYS__COMPUTING_BASE__STATUS],
                                          sizeof(uint64_t), eng__abi__zzpackage_channel) ? 1 : 0;
}

/* How it is going, and what it came to if it is over. ⛳ THE STATE IS ANSWERED ALONGSIDE THE VALUE
 * BECAUSE THE VALUE MEANS DIFFERENT THINGS IN TWO OF THEM: at OK it is a reference the caller now has a
 * claim on, and at ERROR it is a fault WORD — a number shaped like an address and belonging to nobody. A
 * door that answered only a pair would hand those back identically. */
int eng_abi_poll(unsigned long long address, unsigned int* state,
                 unsigned int* dtype, unsigned long long* value) {
    if (state) *state = SYS__COMPUTING_BASE__FREE;
    if (dtype) *dtype = (unsigned int)SYS__KIND__VALUE_NULL;
    if (value) *value = 0ull;
    if (!eng__abi__zzpackage_up() || address == 0ull) return 0;
    /* ⛔ THE STATUS IS READ ON ITS OWN, AND FIRST. The answer sits in the slot before it, so a copy of the
     * whole node reads the answer before the status: one taken while the block finishes can pair the
     * program it was running with a status that already says OK. The block publishes the answer before
     * the status, so a status read first and found over means the answer is there; and only
     * `result_release`, from this side, moves a base on from there, so the second read cannot tear. */
    uint64_t status = 0ull;
    if (!sys__gpu__side_memory_read(eng__abi__zzpackage_machine.family, &status,
                                      eng__abi__zzprivate_slot(address, SYS__COMPUTING_BASE__STATUS),
                                      sizeof status, eng__abi__zzpackage_channel))
        return 0;
    if (state) *state = (unsigned int)status;
    if (status != (uint64_t)SYS__COMPUTING_BASE__OK && status != (uint64_t)SYS__COMPUTING_BASE__ERROR) return 1;
    sys__heap_node now;
    if (!sys_abi_peek(address, &now)) return 0;
    if (dtype) *dtype = (unsigned int)now.args[SYS__COMPUTING_BASE__RESULT_KIND];
    if (value) *value = now.args[SYS__COMPUTING_BASE__PROGRAM];
    return 1;
}

/* Give the base back once the answer has been taken. ⭐⭐ IT HANDS THE BASE TO ITS OWNER RATHER THAN
 * FREEING IT, and it has to: what is left is releasing holds, which runs `sys`, and this door runs
 * nothing of `sys` while the runners work — the owner's runner does. The block finds it in the state
 * below on its next turn round its own loop and puts the computation away — the same acts a scheduler
 * on another block performs inside `read_result`.
 * ⛔ AND IT DOES NOT WAIT FOR THAT TO HAPPEN, deliberately. Waiting would need a bound, and a bound
 * nobody has priced is not a bound; asking with `poll` until the base reads FREE costs the caller one
 * line and invents no constant. */
int eng_abi_result_release(unsigned long long address, unsigned int block) {
    if (!eng__abi__zzpackage_up() || address == 0ull) return 0;
    sys__heap_node now;
    if (!sys_abi_peek(address, &now)) return 0;
    const unsigned int was = (unsigned int)now.args[SYS__COMPUTING_BASE__STATUS];
    if (was != SYS__COMPUTING_BASE__OK && was != SYS__COMPUTING_BASE__ERROR) return 0;

    eng__abi__zzpackage_landing->args[SYS__COMPUTING_BASE__STATUS] =
        (uint64_t)(SYS__COMPUTING_BASE__COMPUTING + block);
    return sys__gpu__side_memory_write(eng__abi__zzpackage_machine.family,
                                        eng__abi__zzprivate_slot(address, SYS__COMPUTING_BASE__STATUS),
                                          &eng__abi__zzpackage_landing->args[SYS__COMPUTING_BASE__STATUS],
                                          sizeof(uint64_t), eng__abi__zzpackage_channel) ? 1 : 0;
}

}  /* extern "C" */

#endif /* SILVANN__ENGINE_ABI_RESIDENCY_CUH */
