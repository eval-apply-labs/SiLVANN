#ifndef SILVANN__ENGINE_ABI_RESIDENCY_CUH
#define SILVANN__ENGINE_ABI_RESIDENCY_CUH

/* ══ RESIDENT — THE BLOCKS UP FROM THE BOOT TO THE SHUTDOWN, AND SCHEDULING FROM OUTSIDE ═══════════════
 * ⚖ 2026-11-04 the resident machine: the boot ends with every block in its waiting loop, and the shutdown stands them
 * down. Between the two the host has a slot of its own (▶ `ENG__ABI__AS_THE_HOST`): it builds as the host, and every
 * program it runs — `execute`, `eval`, `run_grid`, `dispatch` — is a request on a block's queue into an answer it
 * makes and reads (▶ `engine/abi/running.cuh`). The three doors at the foot are that vocabulary bare.
 * ⛳ NO BLOCK IS SPECIAL: block zero parks like the rest, and takes the host's work off its queue as the others do. */

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

/* ── LOOKING INSIDE IS sys's ─────────────────────────────────────────────────────────────────────────
 * ⚖ *"the looking inside do feel like they are sys"*: a node by address, an offset as an address, where a
 * block's base is, the allocator's counters and the fault channel are `sys`'s C interface
 * (`packages/sys/contracts/abi/cpu.cuh`), and bytes in a buffer are nn's (`nn/contracts/abi/cpu.cuh`).
 * A caller reads a block's base with `sys_abi_peek` at the address `sys_abi_base_address` answers. */


/* ── STAYING UP ──────────────────────────────────────────────────────────────────────────────────────
 * Start a runner per block into its waiting loop and DO NOT join them — the boot's last act (▶ `eng_abi_boot_with`).
 * ⛔ THERE IS NO BOUND ON THE LAUNCH AND THERE MUST NOT BE ONE: it lasts the machine's life, and the shutdown stands
 *   it down (`eng__abi__zzpackage_stand_down`). The stop word is read without being written, which is what makes the
 *   shutdown's write to it stick; a waiter that swapped would overwrite it and the engine would be unstoppable,
 *   intermittently. */
static bool eng__abi__zzpackage_reside(void) {
    if (!eng__abi__zzpackage_free_to_launch()) return false;
    if (!sys__gpu__memory_allocate(eng__abi__zzpackage_machine.family,
                                 (void**)&eng__abi__zzpackage_stop, sizeof(unsigned int)) ||
        !sys__gpu__memory_zerofill(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_stop, sizeof(unsigned int))) {
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_stop);
        eng__abi__zzpackage_stop = 0;
        return false;
    }
    /* ⛳ CLEARED BEFORE THE RUNNERS START, and starting a thread orders everything written before it
     * ahead of what the thread reads, so no runner can read a stop word that says stop before anybody has
     * said it. */
    const eng__abi__k_reside__args args = { eng__abi__zzpackage_stop };
    if (!ENG__LAUNCH_RESIDENT(eng__abi__k_reside__run, eng__abi__zzpackage_machine.blocks, &args, sizeof args)) {
        sys__gpu__memory_free(eng__abi__zzpackage_machine.family, eng__abi__zzpackage_stop);
        eng__abi__zzpackage_stop = 0;
        return false;
    }
    eng__abi__zzpackage_resident = 1;
    return true;
}

extern "C" {

/* ── SCHEDULING FROM OUTSIDE THE RUNNERS ─────────────────────────────────────────────────────────────
 * Three doors, and between them they are the whole of what the host does to a block that is already
 * running: queue work on it into an answer, ask whether the value is there, and read it. They are a program's
 * `(sys__scheduler__add …)`, `(sys__result …)` and `(sys__wait …)` from the slot past the runners
 * (▶ `ENG__ABI__AS_THE_HOST`):
 * `place` makes the answer and hands it back, and any number of threads may each wait on their own, whichever block
 * computes it — a computation whose steps continue on other blocks (`sys__scheduler__continue`) fills it once. A
 * block runs the host's work as it runs any block's: off its own queue when it is idle, and nothing outside it
 * touches its base.
 *
 * ⭐ `place` TAKES A BLOCK NUMBER, as a program's verbs do, and works whatever another thread is running. Work queued
 *   on a block that is running a program waits until that block parks.
 * ⛳ `place` TAKES HOLDS OF ITS OWN on both operands, as a program's request does, so the caller keeps its own; the
 *   answer it hands back is the caller's to release (`eng_abi_release`) once it has read it.
 * ⛳ `take` LENDS THE VALUE AS `sys__wait` DOES — the answer keeps it, so a second read gets the same — and hands over a
 *   hold of its own: a reference answered is the caller's to release. */
unsigned long long eng_abi_place(unsigned int block, unsigned long long bindings, unsigned long long program) {
    if (!eng__abi__zzpackage_live() || block >= eng__abi__zzpackage_machine.blocks) return 0ull;
    if (bindings == 0ull || program == 0ull) return 0ull;
    uint64_t answer = 0ull;
    ENG__ABI__AS_THE_HOST(answer = sys__answer__create();
                          if (answer != 0ull && !sys__scheduler__add(block, bindings, program, answer)) {
                              (void)sys__heap_object__release(answer);
                              answer = 0ull;
                          });
    return answer;
}

/* Whether `answer`'s value is there, reading its state and nothing else. Answers 0 for no machine or no answer. */
int eng_abi_poll(unsigned long long answer, int* ready) {
    if (ready) *ready = 0;
    if (!eng__abi__zzpackage_up() || !sys__answer__is(answer)) return 0;
    if (ready) *ready = sys__answer__state(answer) == SYS__ANSWER__READY ? 1 : 0;
    return 1;
}

/* `answer`'s value: its kind and its word, the value's hold the caller's. Answers 0, reading nothing, while it is not
 * there. An error is a value, as a program's `sys__wait` is handed one. */
int eng_abi_take(unsigned long long answer, unsigned int* dtype, unsigned long long* value) {
    if (dtype) *dtype = (unsigned int)SYS__KIND__VALUE_NULL;
    if (value) *value = 0ull;
    if (!eng__abi__zzpackage_up() || !sys__answer__is(answer)) return 0;
    if (sys__answer__state(answer) != SYS__ANSWER__READY) return 0;
    sys__heap_node got = sys__heap_node__nothing();
    ENG__ABI__AS_THE_HOST(got = sys__answer__read(answer));
    if (dtype) *dtype = (unsigned int)got.dtype;
    if (value) *value = got.args[0];
    return 1;
}

}  /* extern "C" */

#endif /* SILVANN__ENGINE_ABI_RESIDENCY_CUH */
