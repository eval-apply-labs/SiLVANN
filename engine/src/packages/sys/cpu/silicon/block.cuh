#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_BLOCK_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_BLOCK_CUH
#include "silicon__header.cuh"   /* the contract these primitives answer */
/* ══ WHO AM I, AND HOW MANY OF US — ON A CPU ═════════════════════════════════════════════════════════
 *
 * ⚖ On what a block is when the evaluator runs on the CPU: *"… ONE THREAD PER GPU AND ONE FOR THE
 * ORCHESTRATOR AT ID 0, if we want to do tensor parallelism."*
 *
 * ⇒ ⭐ SO A "BLOCK" ON THE HOST IS AN EVALUATOR THREAD, AND ITS ID IS THE DEVICE IT DRIVES. Nothing in
 * `sys` has to learn a new word: `sys__computing_base__get(id)`, the claim protocol and the compute
 * verbs all keep saying "block" and keep meaning "the one runner with this number".
 *
 * ⛔⛔ AND IT IS A THREAD-LOCAL THAT THE EVALUATOR SETS, NOT A HARDCODED ZERO. `test/src_host_shim.cuh`
 * answers `0` and `1` because a single-threaded harness has exactly one runner and no way to be wrong.
 * A shipping host evaluator has several, and an id that is always 0 would make every thread claim the
 * SAME computing base — which the claim protocol would then serialise perfectly and silently, turning
 * four GPUs into one. ⇒ ★★ THE STAND-IN THAT CANNOT BE WRONG IN A TEST IS EXACTLY THE ONE THAT IS
 * ALWAYS WRONG IN PRODUCTION.
 *
 * ⛳ THE DEFAULTS ARE THE HARNESS'S — id 0, count 1 — so a single-threaded caller that never sets them
 * behaves exactly as the harness does, and the host suite's checks mean what they say. */

#ifndef SYS__SILICON__HOST_MAX_RUNNERS
#define SYS__SILICON__HOST_MAX_RUNNERS 64u
#endif

/* ⛳ `__thread` RATHER THAN `thread_local`: this file is included by code the C-subset gate holds to C,
 * and `__thread` is what both front ends have. ▶ `src_c_subset_gate`. */
static __thread unsigned int sys__silicon__zzprivate_runner_id = 0u;
static unsigned int          sys__silicon__zzprivate_runner_count = 1u;

static inline unsigned int sys__silicon__block_id(void) {
    return sys__silicon__zzprivate_runner_id;
}

static inline unsigned int sys__silicon__block_count(void) {
    return sys__silicon__zzprivate_runner_count;
}

/* ── THE TWO DOORS THE EVALUATOR USES TO STAND ITS THREADS UP ──────────────────────────────────────
 * ⛳ THEY ARE NOT PART OF THE SEAM'S DECLARED TEN, and deliberately so: a card has no equivalent —
 * `blockIdx.x` is given to a kernel, it is never assigned. So these are the HOST backend's own, and a
 * caller that reaches for them is a caller that only compiles on the host, which is the honest signal. */
static inline void sys__silicon__host_become_runner(unsigned int id) {
    sys__silicon__zzprivate_runner_id = id;
}
static inline void sys__silicon__host_set_runner_count(unsigned int n) {
    sys__silicon__zzprivate_runner_count = (n == 0u) ? 1u : n;
}

/* ══ ⭐⭐ WHICH WORKER A RUNNER IS — ⚖ *"every worker has an id that corresponds to its array index in the
 * computing_base assigned at boot time"* · *"go ahead with the multithreading"* ════════════════════════════
 * A worker is a device a family drives — a card, or the CPU — with its own package rooms, its own hatch and
 * its own settings. Block `w` runs as worker `w`, so a program puts work on a device by computing onto that
 * block; a block past the last worker runs as worker 0, which is the one a machine of one worker has.
 * ⛳ A RUNNER MAY ACT AS ANOTHER WORKER FOR A WHILE — the boot stands every worker's packages up from
 *   runner zero — and `act_as` says which until it is told `-1`, back to its block's. */
static unsigned int          sys__silicon__zzprivate_worker_count = 1u;
static __thread int          sys__silicon__zzprivate_acting_as = -1;

static inline unsigned int sys__silicon__worker_count(void) { return sys__silicon__zzprivate_worker_count; }
static inline unsigned int sys__silicon__worker(void) {
    if (sys__silicon__zzprivate_acting_as >= 0) return (unsigned int)sys__silicon__zzprivate_acting_as;
    const unsigned int b = sys__silicon__zzprivate_runner_id;
    return b < sys__silicon__zzprivate_worker_count ? b : 0u;
}
static inline void sys__silicon__host_set_worker_count(unsigned int n) {
    sys__silicon__zzprivate_worker_count = (n == 0u) ? 1u : (n > SYS__SILICON__WORKERS_MAX ? SYS__SILICON__WORKERS_MAX : n);
}
static inline void sys__silicon__host_act_as(int worker) { sys__silicon__zzprivate_acting_as = worker; }

#endif /* SILVANN__PACKAGES_SYS_CPU_SILICON_BLOCK_CUH */
