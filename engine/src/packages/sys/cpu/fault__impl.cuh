#ifndef SILVANN__PACKAGES_SYS_CPU_FAULT__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_FAULT__IMPL_CUH
#include "../contracts/objects/fault.cuh" /* the layouts it reads and writes */
/* ══ THE FILLING — WHERE A FAULT LANDS IN A REAL BUILD ════════════════════════════════════════════════
 *
 * `fault__header.cuh` declares the seam and defines nothing, so exactly one filling has to arrive or the
 * package links against nothing — the same arrangement the silicon seam has, for the same reason.
 * ⛳ A HARNESS AND THE REAL BUILD WANT OPPOSITE THINGS — a HARNESS fills the verb itself before
 * including the package and asserts on where it went (`test/src_host_shim.cuh`), while the real build
 * wants this filling. The guard below says which.
 *
 * ⭐⭐ WHY THE CHANNEL IS HERE AND NOT IN THE HEADER, WHICH IS THE ONE PLACEMENT DECISION IN THE FILE.
 * The header is what the package says about faults REGARDLESS of what is underneath it — that they are
 * raised, and with what. Three words the host reads after a launch is one tier's ANSWER to that, and a
 * harness answering it with two variables has no channel at all. So the shape travels with the answer.
 * ⛳ THE TYPE ITSELF SITS OUTSIDE THE BACKEND GUARD BELOW, because a struct is not device code: the boot
 * path names it to set the room aside and it is a host function that does so.
 *
 * ⛳ AND IT IS SHAPED LIKE `sys__heap_counters` ON PURPOSE — a plain block of words the boot path owns,
 * handed to a publish, written by whoever faults and read back whole. The allocator already proved that is
 * enough, and a second mechanism for the same job would be a second thing to keep true.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* ⛔⛔ THE GUARD MEANS *"A REAL BUILD, NOT THE HARNESS"*. Nothing here is vendor-specific — it is a
 * pointer and three stores. The host suite gets `test/src_host_shim.cuh`'s own recording stand-in instead.
 * ⇒ ★★ A real build that lands on the wrong side of it gets NO `sys__fault__publish` and NO
 * `sys__fault__raise`, and says so only as *"'sys__fault__publish' was not declared in this scope"*. */
#ifndef SYS__SILICON__HARNESS   /* a real build, not a test harness that brings its own silicon */

/* Where the words are, or nothing. Nothing is the state before the boot path has been anywhere near this
 * — and a raise with no channel is silently dropped rather than refused, because a fault raised before
 * there is anywhere to put it has no better destination and turning it into a second failure would
 * replace a lost report with a crash. */
static __device__ sys__fault_channel* sys__fault__zzprivate_channel;

/* Told once, by the boot path, before anything that can refuse has run. ⛔ IT MUST PRECEDE THE
 * ALLOCATOR'S OWN PUBLISH IN THE SAME BOOT: `sys__heap__publish` refuses a pool too small for the
 * launch by RAISING, so a channel handed over after it would miss the one fault the boot itself can
 * produce. */
static __device__ inline void sys__fault__publish(sys__fault_channel* channel) {
    sys__fault__zzprivate_channel = channel;
}

/* ⭐⭐ KEPT OUT OF LINE. ⛳ WHAT SIZES IT IS CALL SITES, NOT VERBS: inlining makes one copy per CALL, so
 * a verb refusing in four places is four copies. Count the sites with `grep -rn "sys__fault__raise(" src
 * | grep -v "void sys__fault__raise"` — the number climbs with every verb that learns to refuse, so
 * re-derive it rather than read it here. Inlined, each is a pointer load, a test, a branch and three
 * stores, sitting on paths that, between them, almost never run. `REASONED` on the host: out of line,
 * the hot path carries a call it never takes instead of a body it never runs.
 * ⛔ AND IT IS NOT A GENERAL RULE ABOUT THE KEYWORD: the sign follows what the function is on, not what
 * the keyword is. What makes it right here is that the path is COLD: a verb that only runs when
 * something has already gone wrong has nothing to gain from being inlined into the path where everything
 * is going right.
 * ⛳ WHAT WOULD FALSIFY IT: a caller that raises in a loop. **No caller raises in a loop.**
 * ⚠ **A RAISE IS NOT ALWAYS A REFUSAL.** ⚖ *"inf for the value, keep the raise"*: `nn`'s 18
 * fp16-overflow sites raise and **fall through** to publish their answer. ▶ `nn/cpu/primitives__header.cuh`.
 * ⛳ **THAT DOES NOT TOUCH THE CONCLUSION, AND THE REASON IS WORTH BEING PRECISE ABOUT:** what this rests
 * on is that the path is COLD, not that it returns. Those 18 are one raise per VERB CALL — a single flag
 * for the whole loop, not one per element — on a path with `MEASURED` 601x of headroom.
 * ⛔ What WOULD falsify it is a raise that both falls through AND runs hot. There is none, and this line
 * is where to check when somebody adds one. */
static __device__ __noinline__ void sys__fault__raise(uint64_t failed_pc, uint64_t failed_op) {
    sys__fault_channel* channel = sys__fault__zzprivate_channel;
    if (channel == 0) return;
    channel->pc = failed_pc;
    channel->op = failed_op;
    /* ⛳ THE COUNT LAST. A reader that looks at the channel while raisers are running needs the context
     * written before the count it keys on, so the order is the one that stays correct.
     * ⛔⛔ AND THE ORDER ALONE IS NOT ENOUGH. The raisers are RUNNER THREADS in the reader's own address
     * space, running concurrently with it and with each other, so the non-atomic `raised += 1u` can lose
     * a count, two raisers can interleave `pc` and `op`, and a reader would need a volatile view it does
     * not have.
     * ⚠ NOT FIXED HERE, AND IT IS NOT A COMMENT FIX. Making this thread-safe is a design decision —
     * per-runner channels, or an atomic count with a claimed slot — and it belongs with whoever rules
     * on how a multi-runner evaluator reports a fault at all. **UNRULED.** */
    channel->raised += 1u;
}

/* ── sys's C INTERFACE, THE FAULT CHANNEL'S HALF ── ▶ `contracts/abi/cpu.cuh`. ───────────────────────────
 * ⛳ A REFUSAL THAT BELONGS TO ONE PROGRAM LEAVES BY ANOTHER DOOR: a computing base carries its own fault
 * word, which the engine's `poll` and `execute` hand back with the answer. This is the channel for a verb
 * that has nobody to return to.
 * ⛳ THE COUNT IS EXACT AT ZERO AND APPROXIMATE ABOVE IT, and `pc` and `op` belong to the last raiser only —
 * the reasons are at `raise`, above. */
#include "../contracts/abi/cpu.cuh"
extern "C" {
int sys_abi_fault(unsigned int* raised, unsigned long long* failed_pc, unsigned long long* failed_op) {
    const sys__fault_channel* channel = sys__fault__zzprivate_channel;
    if (channel == 0) return 0;
    if (raised)    *raised    = channel->raised;
    if (failed_pc) *failed_pc = (unsigned long long)channel->pc;
    if (failed_op) *failed_op = (unsigned long long)channel->op;
    return 1;
}

/* Back to nothing-went-wrong, so a caller can ask again about the next thing it runs. */
int sys_abi_fault_clear(void) {
    sys__fault_channel* channel = sys__fault__zzprivate_channel;
    if (channel == 0) return 0;
    channel->raised = 0u; channel->pc = 0ull; channel->op = 0ull;
    return 1;
}
}
#endif

#endif /* SILVANN__PACKAGES_SYS_CPU_FAULT__IMPL_CUH */
