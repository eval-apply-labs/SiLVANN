#ifndef SILVANN__ENGINE_LAUNCH_CUH
#define SILVANN__ENGINE_LAUNCH_CUH
/* ══ ⭐⭐⭐ HOW A KERNEL IS STARTED, WRITTEN ONCE ═══════════════════════════════════════════════════════
 *
 * ⚖ *"do the rearrangement, sys and the evaluator go host."*
 *
 * The engine's kernels are functions the host runs, one runner thread per block, and every launch goes
 * through one of the four macros below. This file is that seam, so what a launch means is stated in ONE
 * place rather than at every site. The sites are counted, not written down here — outside this file,
 * `grep -rn 'ENG__LAUNCH_\(ONE\|MANY\|RESIDENT\|HERE\)(' engine` — and the claim gate's
 * `launch_width` rule counts the same thing and fails a launch that bypasses the seam.
 *
 * ⛔⛔ **AND THE SHAPES ARE NOT THE SAME TRANSLATION, WHICH IS THE WHOLE REASON THERE ARE SEVERAL
 * MACROS.** One block doing one thing is a FUNCTION CALL. Many blocks are not a loop:
 * ```
 *   eng__abi__k_grid   block 0  runs the program, then writes `stop`
 *                      block N  parks in `eng__opcodes__await_instructions` until `stop` is written
 * ```
 * ⇒ ★★ **RUNNING THOSE SEQUENTIALLY DEADLOCKS** (`REASONED`, from `eng__abi__k_grid`). Called in order,
 * block 0 runs first, and a program that schedules work on block 1 waits for a runner that will not
 * start until block 0 returns; called in any other order, a parked block waits for a `stop` that only
 * block 0 writes. The blocks are CONCURRENT PARTICIPANTS, not iterations, so the host form of a
 * many-block launch is a THREAD POOL.
 * ⛳ AND NOTHING WOULD SAY SO: a parked block's wait on the stop word is unbounded, so nothing in the
 * engine detects or breaks a runner that never starts — the process hangs forever with no suspect.
 * A sequential loop here would be a runner that never starts, written on purpose.
 * ⛳ AND A THIRD SHAPE HAS ITS OWN MACRO: a launch whose blocks outlive the call. A pool that joins before
 * returning would never return from one, so `ENG__LAUNCH_RESIDENT` starts its runners and leaves them
 * running, and standing down joins them. ▶ below.
 *
 * ⭐⭐ **AND IT WORKS BECAUSE THE ENGINE NEVER READS `blockIdx` OR `threadIdx`** (`grep -rnE
 * 'blockIdx|threadIdx' engine` finds only this sentence). Every kernel asks `sys__silicon__block_id()`,
 * which is the seam, and the host backend answers it from a `__thread` slot; a runner is one thread, so
 * there is no thread index to ask about. ⇒ ★ THE KERNELS NEED NO GRID, because they never asked about one.
 */


#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ⛳ A KERNEL IS STARTED THROUGH A RUNNER: a function of one pointer that unpacks the kernel's arguments
 * and calls it. A launch that starts more than one block owes one, written beside the kernel, because C
 * has no way to carry a call with its arguments into a thread other than a function and a record. */
typedef void (*eng__launch__runner)(const void* args);

typedef struct {
    unsigned int        block;
    eng__launch__runner run;
    const void*         args;
} eng__launch__runner_start;

/* ⭐ THE DEVICE EACH WORKER'S RUNNER IS ON — a runner is a fresh thread, and a thread starts on device 0 of every
 * family, so a runner binds its worker's device before it runs anything. The boot fills this in. */
static sys__silicon_family__id eng__launch__zzprivate_worker_family[SYS__SILICON__WORKERS_MAX];
static uint32_t                eng__launch__zzprivate_worker_device[SYS__SILICON__WORKERS_MAX];
static unsigned int            eng__launch__zzprivate_workers = 0u;
static inline void eng__launch__set_worker(unsigned int w, sys__silicon_family__id family, uint32_t device) {
    if (w >= SYS__SILICON__WORKERS_MAX) return;
    eng__launch__zzprivate_worker_family[w] = family;
    eng__launch__zzprivate_worker_device[w] = device;
    if (w + 1u > eng__launch__zzprivate_workers) eng__launch__zzprivate_workers = w + 1u;
}
static inline void eng__launch__forget_workers(void) { eng__launch__zzprivate_workers = 0u; }

static void* eng__launch__zzprivate_enter(void* at) {
    const eng__launch__runner_start* start = (const eng__launch__runner_start*)at;
    sys__silicon__host_become_runner(start->block);
    if (eng__launch__zzprivate_workers != 0u) {
        const unsigned int w = sys__silicon__worker();
        (void)sys__silicon__bind_device(eng__launch__zzprivate_worker_family[w], eng__launch__zzprivate_worker_device[w]);
    }
    start->run(start->args);
    return 0;
}

/* ⛔ A RUNNER THAT CANNOT BE STARTED ENDS THE PROCESS. The runners already started may be parked
 * waiting on a stop word only a missing runner would write, and that wait is never broken, so returning
 * would hang with no suspect; a process that stops and says why is the better of the two. */
static void eng__launch__zzprivate_refuse(unsigned int block) {
    fprintf(stderr, "silvann: runner %u of a launch could not be started\n", block);
    abort();
}

/* ⛳ ONE RUNNER PER BLOCK, STARTED TOGETHER AND JOINED. The count is published before any of them runs
 * so that a block asking `sys__silicon__block_count()` inside the body gets the real answer rather than
 * a count that grows as the pool fills.
 * ⛔ AND `n <= 1` TAKES THE CALLER'S OWN THREAD RATHER THAN SPAWNING ONE. Not for speed: a one-block
 * machine is the shape every host suite runs, and a thread that must be joined turns a debugger session
 * into two stacks for no reason. It is also the arm that keeps `become_runner(0)` on the path that a
 * single-block build exercises. */
static inline void eng__launch__spread(unsigned int n, eng__launch__runner run, const void* args) {
    if (n <= 1u) {
        sys__silicon__host_set_runner_count(1u);
        sys__silicon__host_become_runner(0u);
        run(args);
        return;
    }
    sys__silicon__host_set_runner_count(n);
    pthread_t*                    pool   = (pthread_t*)malloc(n * sizeof(pthread_t));
    eng__launch__runner_start* starts =
        (eng__launch__runner_start*)malloc(n * sizeof(eng__launch__runner_start));
    if (pool == 0 || starts == 0) eng__launch__zzprivate_refuse(0u);
    for (unsigned int b = 0; b < n; ++b) {
        starts[b].block = b;
        starts[b].run   = run;
        starts[b].args  = args;
        if (pthread_create(&pool[b], 0, eng__launch__zzprivate_enter, &starts[b]) != 0)
            eng__launch__zzprivate_refuse(b);
    }
    for (unsigned int b = 0; b < n; ++b) pthread_join(pool[b], 0);
    free(starts);
    free(pool);
}

/* ⛳ A ONE-BLOCK LAUNCH NEEDS NO RUNNER: it is the caller's own thread calling the kernel, so the call is
 * written out and the arguments are evaluated once, as a launch evaluates them. It still says how many
 * runners there are, because a launch that only named its runner would leave the count from whatever
 * ran before it, and a block asking `sys__silicon__block_count()` would be told about runners that are
 * not there. */
#define ENG__LAUNCH_ONE(kernel, ...)                                                                   \
    do {                                                                                               \
        sys__silicon__host_set_runner_count(1u);                                                       \
        sys__silicon__host_become_runner(0u);                                                          \
        kernel(__VA_ARGS__);                                                                           \
    } while (0)
/* ⭐ AND A STATEMENT RUN THE SAME WAY — `ENG__LAUNCH_ONE` for work that is one statement and answers its
 * caller directly. The machine is the host's, so a door that needs one runner does not write a kernel and
 * pass its answer through a slot in the machine's memory: it becomes runner zero and runs the statement
 * where it stands, writing into its own locals. */
#define ENG__LAUNCH_HERE(...)                                                                          \
    do {                                                                                               \
        sys__silicon__host_set_runner_count(1u);                                                       \
        sys__silicon__host_become_runner(0u);                                                          \
        __VA_ARGS__;                                                                                   \
    } while (0)
#define ENG__LAUNCH_MANY(run, n, args)       eng__launch__spread((unsigned int)(n), (run), (args))

/* ══ ⭐⭐ A LAUNCH THAT OUTLIVES THE CALL ══════════════════════════════════════════════════════════════
 * A resident launch parks every block until somebody outside writes its stop word, and the only writer
 * is a call the host makes AFTER the launch has returned. `spread` joins its runners before it returns,
 * so a resident launch through it never returns at all — the caller waits on runners that wait on the
 * caller, and nothing faults. So a resident launch has its own pair:
 *     ENG__LAUNCH_RESIDENT(run, n, args, bytes)   starts n runners and returns with them still running
 *     eng__launch__join_resident()                waits for them, once the stop word has been written
 * ⛔ THE RUNNERS ARE THREADS EVEN AT n = 1, because the caller's own thread is the one that must come
 *   back. ⛔ AND THE ARGUMENTS ARE COPIED HERE, because the runners outlive the frame that named them.
 * ⛳ ONE RESIDENT LAUNCH AT A TIME — the engine refuses a second while one is up — so the pool is one
 *   list. ⚠ Standing down is owed before exit, which is what shutting the engine down does. */
#define ENG__LAUNCH__RESIDENT_ARG_BYTES 64u
static pthread_t*                    eng__launch__zzprivate_resident        = 0;
static eng__launch__runner_start* eng__launch__zzprivate_resident_starts = 0;
static unsigned int                  eng__launch__zzprivate_resident_count  = 0;
static unsigned char eng__launch__zzprivate_resident_args[ENG__LAUNCH__RESIDENT_ARG_BYTES];

static inline bool eng__launch__start_resident(unsigned int n, eng__launch__runner run,
                                               const void* args, size_t bytes) {
    if (bytes > sizeof eng__launch__zzprivate_resident_args) return false;
    if (n == 0u) n = 1u;
    memcpy(eng__launch__zzprivate_resident_args, args, bytes);
    eng__launch__zzprivate_resident        = (pthread_t*)malloc(n * sizeof(pthread_t));
    eng__launch__zzprivate_resident_starts =
        (eng__launch__runner_start*)malloc(n * sizeof(eng__launch__runner_start));
    if (eng__launch__zzprivate_resident == 0 || eng__launch__zzprivate_resident_starts == 0)
        eng__launch__zzprivate_refuse(0u);
    sys__silicon__host_set_runner_count(n);
    for (unsigned int b = 0; b < n; ++b) {
        eng__launch__zzprivate_resident_starts[b].block = b;
        eng__launch__zzprivate_resident_starts[b].run   = run;
        eng__launch__zzprivate_resident_starts[b].args  = eng__launch__zzprivate_resident_args;
        if (pthread_create(&eng__launch__zzprivate_resident[b], 0, eng__launch__zzprivate_enter,
                           &eng__launch__zzprivate_resident_starts[b]) != 0)
            eng__launch__zzprivate_refuse(b);
        eng__launch__zzprivate_resident_count = b + 1u;
    }
    return true;
}

static inline bool eng__launch__join_resident(void) {
    for (unsigned int b = 0; b < eng__launch__zzprivate_resident_count; ++b)
        pthread_join(eng__launch__zzprivate_resident[b], 0);
    free(eng__launch__zzprivate_resident_starts);
    free(eng__launch__zzprivate_resident);
    eng__launch__zzprivate_resident        = 0;
    eng__launch__zzprivate_resident_starts = 0;
    eng__launch__zzprivate_resident_count  = 0;
    return true;
}

#define ENG__LAUNCH_RESIDENT(run, n, args, bytes)                                                      \
    eng__launch__start_resident((unsigned int)(n), (run), (args), (bytes))

#endif /* SILVANN__ENGINE_LAUNCH_CUH */
