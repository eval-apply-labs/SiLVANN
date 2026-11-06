#ifndef SILVANN__PACKAGES_SYS_CPU_GARBAGE_COLLECTION__HEADER_CUH
#define SILVANN__PACKAGES_SYS_CPU_GARBAGE_COLLECTION__HEADER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../contracts/objects/garbage_collection.cuh" /* the mode's fault word */

/* ══ THE MANUAL MODE — A TEARDOWN WAITS UNTIL THE PROGRAM COLLECTS ═══════════════════════════════════
 *
 * ⚖ ARCHITECT 2026-11-03: *"manual garbage collection should be a thing and it would make this language a
 * lot closer to zig/lua"* (NN-151) · *"we should set it to be so as a command rather than a variable, as
 * otherwise every binding gets a diversion path ... it is the responsibility of the programmer to turn
 * its config off"*.
 *
 * The counts still decide WHAT is dead; the mode decides WHEN it is taken apart. A release that takes an
 * object's count to zero ordinarily takes it apart at once (`sys__heap_object__release`). While the thread
 * is in the manual mode, the object waits on the thread's worklist instead, the worklist holding it, and
 * what it holds waits with it:
 *
 *     (sys__garbage_collection__manual #t)    on, from here; it answers the mode it replaced
 *     (sys__garbage_collection__manual #f)    off, and what waits is taken apart now
 *     (sys__garbage_collection__collect)      everything waiting is taken apart now; it answers #t
 *
 * so a program scopes it by hand:
 *
 *     (let ((was (sys__garbage_collection__manual #t))) … (sys__garbage_collection__manual was))
 *
 * WHOSE IT IS. The thread's, set by the verb alone: it stays on across evaluations until the program turns
 * it off, a computation the thread runs for another worker included. One handed to another thread
 * (`sys__scheduler__add`) runs in that thread's mode.
 * WHAT ENDS THE WAIT. A collect, turning the mode off, and an evaluation's end — a nested one's too:
 * nothing waits past an evaluation, so a thread's end (▶ the worklist a thread keeps, `heap_object__impl.cuh`)
 * finds nothing waiting, and the mode staying on costs the next evaluation nothing it did not ask for.
 * WHAT ENDS THE MODE. The verb, and the heap: the mode remembers the publish it was turned on under, and a
 * heap published again — a reset — is a heap that never turned it on.
 * WHAT IT COSTS. Off: a thread's flag read in each release that reaches zero, and a collect as an
 * evaluation ends that finds nothing. On: a waiting object keeps its room, and its heap chunk off the free
 * stack, until it is collected. A release in the mode that finds no room on the worklist takes the object
 * apart at once, as it would off.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */

/* Whether this thread's teardowns wait — what a release that reaches zero asks. */
static __device__ inline bool sys__garbage_collection__zzpackage_manual(void);

/* The mode set: `on` or off, answering the mode it replaced. Turning it off collects. The verb's body. */
static __device__ inline bool sys__garbage_collection__manual(bool on);

/* Everything waiting on this thread's worklist, taken apart now. The verb's body, and what turning the
 * mode off and an evaluation's end call. */
static __device__ inline void sys__garbage_collection__collect(void);

/* An evaluation ending on this thread: what it left waiting is collected, and the mode is left as the
 * program left it. The evaluator's one call. */
static __device__ inline void sys__garbage_collection__zzengine_evaluation_ends(void);

/* The thread ending: out of the mode for good, and what waits taken apart — the worklist's let-go, before it
 * lets go of the worklist's own pieces, which in the mode would wait on a worklist made for them. */
static __device__ inline void sys__garbage_collection__zzpackage_thread_ends(void);

#endif /* SILVANN__PACKAGES_SYS_CPU_GARBAGE_COLLECTION__HEADER_CUH */
