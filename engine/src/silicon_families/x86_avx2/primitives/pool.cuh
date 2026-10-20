#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_POOL_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_POOL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../../operating_systems/os.cuh"   /* threads, locks, the clock and the processors: the system's */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

/* ══ ⭐⭐ THE POOL — ONE THREAD A FREE CORE, AND A LAUNCH IS THEIR BLOCKS ══════════════════════════════
 * A launch of `blocks` blocks is those blocks spread over the pool's threads and the caller's own: each
 * takes the next block number, runs it, and takes another, and the launch returns when every block is
 * done. So the doors are synchronous here — there is no queue to wait on, and `compute_completed` has
 * nothing left to do.
 *
 * ⚖ *"your setup imagines the cpus threads as available, on a consumer pc it might not be the case… if we
 * start thrashing on the scheduler it might even make things slower"*. `MEASURED`
 * (`measurements/2026-09-27_avx2_expert_gemv.md`): with 28 of 56 logical CPUs busy, taking all 56 cost 3x
 * against taking the 28 free ones, because a layer ends at a barrier and one preempted thread stalls all
 * of them. So the pool is sized to what is FREE, once, when the family binds its device:
 *     allowed   the CPUs this process may run on (`sys__os__cpus_of_thread` — `taskset`, a cgroup's cpuset)
 *     cores     one per physical core among them, a sibling hyperthread never counted twice
 *     free      a core whose every CPU was idle more than half of a 50 ms sample of its ticks (`/proc/stat`)
 *     minus     one: the caller's own core — it runs blocks too, and it is the evaluator's runner
 *     capped    by `SILVANN_CPU_THREADS` when it is set above zero
 * and the threads are pinned one to a core, alternating sockets so a small pool still spans both.
 * ⛳ NOT YET: shrinking while running. Every launch ends at the same point, so the pool could time each
 *   thread and drop a core whose thread keeps finishing last; `REASONED`, not built.
 *
 * ══ ⭐⭐ A POOL A DEVICE, AND A SOCKET IS A DEVICE ══════════════════════════════════════════════════════
 * ⚖ *"once we implement the tensor parallelism primitives we can treat them as two independent workers and
 * that should increase the bandwith as it stops choking on NUMA"* · *"one driving thread per cpu"*.
 *     device 0      the whole machine, as it always was: every free core, on every socket
 *     device 1 + s  NUMA node s alone (the s-th that has a CPU this process may use): a thread on every free
 *                   core of it, and its memory on that node (`sys.cuh`)
 * Each device has its own pool, stood up the first time a thread binds it, and a thread launches on the
 * pool of the device IT has bound — so two workers on two sockets run two launches at once, each on its
 * own cores and its own memory.
 * ⛔⛔ ON A NODE THE CALLER RUNS NO BLOCKS AND IS NEVER MOVED — it hands the launch out and waits. `MEASURED`
 * against the first design, which pinned the binding thread to a core of the node: a copy from a
 *   CPU worker binds that worker's device on whatever thread runs it, so a card's driving thread was moved onto
 *   the node too, beside spinning pool threads — a card's steps of a 35B token went 20 -> 42 ms, and with a
 *   second thread left pinned on the same core a token took 7.9 s. A thread is where the scheduler puts it.
 * ⛔ Device 0 and a socket device overlap: a machine that chooses both puts two pools on the same cores. */

#define X86_AVX2__THREADS_MAX 256u
#define X86_AVX2__DEVICES_MAX 9u                  /* the whole machine, and up to eight nodes */

typedef struct x86_avx2__pool {
    uint32_t          workers;                 /* threads beside the caller */
    int               cpu[X86_AVX2__THREADS_MAX];
    sys__os__thread   thread[X86_AVX2__THREADS_MAX];
    sys__os__lock     lock;
    sys__os__condition wake;
    /* the launch in flight */
    void            (*fn)(void*);
    void*             arg;
    uint32_t          blocks;
    uint64_t          next;                    /* the launch's generation above, the next block number below */
    uint32_t          done;                    /* blocks finished */
    uint32_t          generation;              /* one a launch, so a thread knows there is new work */
    int               started;
    int               node;                    /* the NUMA node it and its memory are on, or -1: every one */
    sys__os__cpus     own;                     /* the whole machine's: the CPUs of the caller's core, left free for it */
    sys__os__cpus     worked;                  /* ...and the CPUs of the cores the threads are on, siblings included */
} x86_avx2__pool;

static x86_avx2__pool x86_avx2__zzprivate_pools[X86_AVX2__DEVICES_MAX];
static sys__os__lock x86_avx2__zzprivate_pools_lock = SYS__OS__LOCK_INIT;
/* the device the calling thread has bound — whose pool its launches run on, and whose node its memory is on */
static __thread uint32_t x86_avx2__zzpackage_device = 0u;
/* the block a thread is running, and how many the launch has — what nn's `block()` / `blocks()` answer */
static __thread uint32_t x86_avx2__zzpackage_block  = 0u;
static __thread uint32_t x86_avx2__zzpackage_blocks = 1u;

/* Take blocks of launch `generation` until there are none left.
 * ⛔ THE CLAIM CARRIES THE GENERATION, AND THAT IS NOT DECORATION: a worker that wakes late for one launch
 *   while the caller is setting up the next would otherwise take a block number of the NEW launch from a
 *   counter being reset under it, and a block could run twice — which for an in-place door (a residual
 *   `add` into its own input) is a wrong answer, not wasted work. A claim is a compare-and-swap on
 *   (generation, block), so it succeeds only for the launch the worker woke for. */
static inline void x86_avx2__zzprivate_take(x86_avx2__pool* p, uint32_t generation) {
    /* ⭐ A RUN OF BLOCKS A CLAIM, NOT ONE: a small door launches hundreds of blocks of a few elements each, and a
     *   claim is a compare-and-swap on one line every thread wants. `MEASURED`: zeroing 8192 halves cost
     *   58-91 us that way, most of it the claims. Four runs a thread keep a slow thread from holding the rest. */
    const uint32_t threads = p->workers + 1u;
    for (;;) {
        uint64_t v = __atomic_load_n(&p->next, __ATOMIC_ACQUIRE);
        uint32_t b, blocks, run;
        do {
            if ((uint32_t)(v >> 32) != generation) return;
            /* read only once the counter names this launch: the caller writes it before the launch's size */
            blocks = __atomic_load_n(&p->blocks, __ATOMIC_ACQUIRE);
            run = blocks / (4u * threads) > 1u ? blocks / (4u * threads) : 1u;
            b = (uint32_t)v;
            if (b >= blocks) return;
        } while (!__atomic_compare_exchange_n(&p->next, &v, v + (uint64_t)run, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE));
        const uint32_t end = b + run < blocks ? b + run : blocks;
        for (uint32_t k = b; k < end; ++k) {
            x86_avx2__zzpackage_block = k; x86_avx2__zzpackage_blocks = blocks;
            p->fn(p->arg);
        }
        __atomic_fetch_add(&p->done, end - b, __ATOMIC_RELEASE);
    }
}

/* ⭐ HOW LONG A THREAD STAYS AWAKE AFTER ITS LAST LAUNCH — `SILVANN_CPU_SPIN_US`, 5000 unless set.
 * ⛔⛔ IT WAS ~100 us (20,000 pauses of 5.1 ns on node03), AND ON A MACHINE WHOSE CORES CLOCK DOWN WHEN IDLE THAT IS THE
 *   WHOLE STORY OF A MIXED DEVICE TOKEN. `MEASURED`, the disjoint 35B (two MI50s and this pool, one program a
 *   token): between one layer's expert step and the next the cards work for ~2 ms, the pool's threads slept, their cores
 *   fell from 3.3 to 1.2 GHz (`schedutil`), and every expert step ran at the low clock — 50-55 ms of the token against
 *   ~20 alone. Awake for 1 ms: 76-79 ms a token; 5 ms: 67-72; 25 ms or forever: 66-70; the old ~100 us: 93-101.
 * ⛳ A WINDOW IN TIME, NOT A COUNT OF PAUSES: a pause is 5 ns on this Broadwell and ~40 on later cores. Past it the thread
 *   sleeps, so an idle engine still costs no core. */
static inline uint64_t x86_avx2__zzprivate_now_ns(void) {
    struct timespec t;
    sys__os__clock(&t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static inline uint64_t x86_avx2__zzprivate_spin_ns(void) {
    static uint64_t ns = 0ull;
    if (ns == 0ull) {
        const char* s = getenv("SILVANN_CPU_SPIN_US");
        const long us = s ? strtol(s, 0, 10) : 5000;
        ns = (uint64_t)(us > 0 ? us : 1) * 1000ull;
    }
    return ns;
}

/* A worker: wait for a generation it has not seen, take blocks, wait again. It stays awake a while first — launches
 * come back to back inside a layer, and a mixed token comes back within milliseconds — and then sleeps. */
static void* x86_avx2__zzprivate_worker(void* a) {
    x86_avx2__pool* p = &x86_avx2__zzprivate_pools[(uintptr_t)a >> 16];
    const int i = (int)((uintptr_t)a & 0xFFFFu);
    sys__os__cpu_pin_thread(p->cpu[i]);
    const uint64_t window = x86_avx2__zzprivate_spin_ns();
    uint32_t seen = 0u;
    for (;;) {
        uint32_t g = __atomic_load_n(&p->generation, __ATOMIC_ACQUIRE);
        const uint64_t since = x86_avx2__zzprivate_now_ns();
        for (uint32_t spin = 1u; g == seen; ++spin) {
            _mm_pause();
            g = __atomic_load_n(&p->generation, __ATOMIC_ACQUIRE);
            if ((spin & 255u) == 0u && x86_avx2__zzprivate_now_ns() - since > window) break;   /* the clock, now and then */
        }
        if (g == seen) {
            sys__os__lock_take(&p->lock);
            while ((g = __atomic_load_n(&p->generation, __ATOMIC_ACQUIRE)) == seen) sys__os__condition_wait(&p->wake, &p->lock);
            sys__os__lock_give(&p->lock);
        }
        seen = g;
        x86_avx2__zzprivate_take(p, g);
    }
    return 0;
}

/* ── WHERE: the CPUs this process may use, and the nodes they are on ───────────────────────────────── */
enum { X86_AVX2__ZZPRIVATE_CPUS = 1024, X86_AVX2__ZZPRIVATE_NODES = 64 };
/* ⛳ READ ONCE, BEFORE ANY THREAD IS PINNED: binding a socket device pins the thread that binds it, and a thread's
 *   affinity is what `sys__os__cpus_of_thread` answers — so a later read on a pinned thread would see one core. */
static sys__os__cpus x86_avx2__zzprivate_allowed;
static int       x86_avx2__zzprivate_cpu_node[X86_AVX2__ZZPRIVATE_CPUS];
static int       x86_avx2__zzprivate_node[X86_AVX2__DEVICES_MAX - 1u];   /* socket device 1 + s is node [s] */
static uint32_t  x86_avx2__zzprivate_nodes = 0u;
static sys__os__once x86_avx2__zzprivate_where_once = SYS__OS__ONCE_INIT;
static void x86_avx2__zzprivate_where(void) {
    sys__os__cpus_clear(&x86_avx2__zzprivate_allowed);
    if (!sys__os__cpus_of_thread(&x86_avx2__zzprivate_allowed)) return;
    int seen[X86_AVX2__ZZPRIVATE_NODES] = {0};
    for (int cpu = 0; cpu < X86_AVX2__ZZPRIVATE_CPUS && cpu < SYS__OS__CPUS_MOST; ++cpu) {
        x86_avx2__zzprivate_cpu_node[cpu] = -1;
        if (!sys__os__cpus_has(&x86_avx2__zzprivate_allowed, cpu)) continue;
        const int n = sys__os__cpu_node(cpu);
        if (n >= 0 && n < X86_AVX2__ZZPRIVATE_NODES) { x86_avx2__zzprivate_cpu_node[cpu] = n; seen[n] = 1; }
    }
    for (int n = 0; n < X86_AVX2__ZZPRIVATE_NODES && x86_avx2__zzprivate_nodes < X86_AVX2__DEVICES_MAX - 1u; ++n)
        if (seen[n]) x86_avx2__zzprivate_node[x86_avx2__zzprivate_nodes++] = n;
    /* one node is the whole machine already: no socket devices beside device 0 */
    if (x86_avx2__zzprivate_nodes < 2u) x86_avx2__zzprivate_nodes = 0u;
}
/* How many devices: the whole machine, and a socket device a node when there is more than one. */
static inline uint32_t x86_avx2__zzpackage_devices(void) {
    sys__os__once_run(&x86_avx2__zzprivate_where_once, x86_avx2__zzprivate_where);
    return 1u + x86_avx2__zzprivate_nodes;
}
/* The node device `index` is on, or -1 for the whole machine. */
static inline int x86_avx2__zzpackage_node_of(uint32_t index) {
    sys__os__once_run(&x86_avx2__zzprivate_where_once, x86_avx2__zzprivate_where);
    return index == 0u || index > x86_avx2__zzprivate_nodes ? -1 : x86_avx2__zzprivate_node[index - 1u];
}

/* ── HOW MANY: the free physical cores of the device ───────────────────────────────────────────────── */
/* ⛳ THE TOPOLOGY IN ONE PLACE: a CPU's package and physical core, the system's (`sys__os__cpu_core`). Where they cannot
 *   be read — a container without `/sys` — the CPU counts as a core of its own on package 0, so the pool has a thread
 *   a free CPU; unread, every CPU would match one unknown core and the pool would be the caller alone. */
static inline void x86_avx2__zzprivate_core_of(int cpu, int* pkg, int* id) {
    sys__os__cpu_core(cpu, pkg, id);
    if (*pkg < 0 || *id < 0) { *pkg = 0; *id = X86_AVX2__ZZPRIVATE_CPUS + cpu; }
}
/* The pool's threads' CPUs into `cpus_out`, and their count. On the whole machine the caller's own core is left
 * out, as it always was — it runs blocks too; on a node every free core has a thread, and the caller runs none. */
static inline uint32_t x86_avx2__zzprivate_size(int* cpus_out, int node, sys__os__cpus* own, sys__os__cpus* worked) {
    enum { MOST = X86_AVX2__ZZPRIVATE_CPUS };
    static unsigned long long b0[MOST], t0[MOST], b1[MOST], t1[MOST];
    memset(b0, 0, sizeof b0); memset(t0, 0, sizeof t0); memset(b1, 0, sizeof b1); memset(t1, 0, sizeof t1);
    sys__os__cpu_ticks(b0, t0, MOST);
    const struct timespec nap = { 0, 50 * 1000 * 1000 };
    sys__os__sleep(&nap);
    sys__os__cpu_ticks(b1, t1, MOST);
    /* per physical core (package, core id): its first allowed CPU, and whether every CPU of it is idle */
    int core_cpu[MOST], core_pkg[MOST], core_id[MOST], core_busy[MOST], core_of[MOST], cores = 0;
    for (int cpu = 0; cpu < MOST; ++cpu) core_of[cpu] = -1;
    for (int cpu = 0; cpu < MOST && cpu < SYS__OS__CPUS_MOST; ++cpu) {
        if (!sys__os__cpus_has(&x86_avx2__zzprivate_allowed, cpu)) continue;
        if (node >= 0 && x86_avx2__zzprivate_cpu_node[cpu] != node) continue;
        int pkg, id;
        x86_avx2__zzprivate_core_of(cpu, &pkg, &id);
        const unsigned long long dt = t1[cpu] - t0[cpu], db = b1[cpu] - b0[cpu];
        const int busy = dt > 0ull && db * 2ull > dt;
        int k = 0;
        while (k < cores && !(core_pkg[k] == pkg && core_id[k] == id)) ++k;
        if (k == cores) { core_cpu[k] = cpu; core_pkg[k] = pkg; core_id[k] = id; core_busy[k] = 0; ++cores; }
        core_busy[k] |= busy;
        core_of[cpu] = k;
    }
    /* the free cores, alternating packages: first core of package 0, of package 1, second of 0, … */
    int order[MOST], n = 0, max_pkg = 0;
    for (int k = 0; k < cores; ++k) if (core_pkg[k] > max_pkg) max_pkg = core_pkg[k];
    for (int round = 0; n < cores && round < cores; ++round)
        for (int pkg = 0; pkg <= max_pkg; ++pkg) {
            int seen = 0;
            for (int k = 0; k < cores; ++k)
                if (core_pkg[k] == pkg && !core_busy[k] && seen++ == round) order[n++] = core_cpu[k];
        }
    /* on the whole machine, minus the caller's own core; capped by the setting */
    const int self = node >= 0 ? -1 : sys__os__cpu_now();
    const uint32_t caller = node >= 0 ? 0u : 1u;             /* whether the caller is one of the threads */
    uint32_t workers = 0u;
    const char* cap_s = getenv("SILVANN_CPU_THREADS");
    const long cap = cap_s ? strtol(cap_s, 0, 10) : 0;
    /* ⛔ THE CALLER'S CORE, NOT ITS CPU: `order` holds each core's first CPU, and a caller on the core's second hyperthread
     *   matched none of them — the pool then took the first cores, the caller's among them, and the caller shared a core
     *   with a pool thread for the whole run. `MEASURED` (the 35B on the CPU alone, four cores, the pool's report and the
     *   caller's CPU each run): with the pool on CPUs 0 1 2 and the caller on CPU 4 — core 0's other half — 7.60 tokens/s;
     *   with the caller's core left out, the same caller CPU, 8.98, as every other run. */
    const int self_core = self >= 0 && self < MOST ? core_of[self] : -1;
    for (int k = 0; k < n && workers < X86_AVX2__THREADS_MAX; ++k) {
        if (self_core >= 0 && core_of[order[k]] == self_core) continue;
        if (cap > 0 && (long)(workers + caller) >= cap) break;
        cpus_out[workers++] = order[k];
    }
    if (caller == 1u && n > 0 && workers + 1u > (uint32_t)n) workers = (uint32_t)n - 1u;
    /* the CPUs of the threads' cores, and on the whole machine those of the caller's own core when it is free */
    sys__os__cpus_clear(own); sys__os__cpus_clear(worked);
    for (uint32_t w = 0u; w < workers; ++w)
        for (int cpu = 0; cpu < MOST && cpu < SYS__OS__CPUS_MOST; ++cpu)
            if (core_of[cpu] >= 0 && core_of[cpu] == core_of[cpus_out[w]]) sys__os__cpus_add(worked, cpu);
    if (self >= 0 && self < MOST && core_of[self] >= 0 && !core_busy[core_of[self]] && !sys__os__cpus_has(worked, self))
        for (int cpu = 0; cpu < MOST && cpu < SYS__OS__CPUS_MOST; ++cpu)
            if (core_of[cpu] == core_of[self]) sys__os__cpus_add(own, cpu);
    return workers;
}

/* The pool of the device the calling thread has bound, stood up the first time: sized, a thread pinned to each
 * chosen core. */
static inline x86_avx2__pool* x86_avx2__zzpackage_start(void) {
    const uint32_t device = x86_avx2__zzpackage_device < x86_avx2__zzpackage_devices() ? x86_avx2__zzpackage_device : 0u;
    x86_avx2__pool* p = &x86_avx2__zzprivate_pools[device];
    if (__atomic_load_n(&p->started, __ATOMIC_ACQUIRE)) return p;
    sys__os__lock_take(&x86_avx2__zzprivate_pools_lock);
    if (!p->started) {
        sys__os__lock_init(&p->lock);
        sys__os__condition_init(&p->wake);
        p->node = x86_avx2__zzpackage_node_of(device);
        p->workers = x86_avx2__zzprivate_size(p->cpu, p->node, &p->own, &p->worked);
        for (uint32_t i = 0u; i < p->workers; ++i)
            if (!sys__os__thread_start(&p->thread[i], x86_avx2__zzprivate_worker, (void*)(uintptr_t)((device << 16) | i))) { p->workers = i; break; }
        /* ⛳ `SILVANN_CPU_POOL_REPORT=1` says what was chosen: the size is decided from the machine's load at
         *   the moment of binding, so the one way to know it is to be told. */
        if (getenv("SILVANN_CPU_POOL_REPORT") != 0) {
            fprintf(stderr, "x86_avx2: device %u (node %d), a pool of %u thread(s) %s, on CPUs", device, p->node, p->workers,
                    p->node >= 0 ? "and the caller waiting" : "beside the caller");
            for (uint32_t i = 0u; i < p->workers; ++i) fprintf(stderr, " %d", p->cpu[i]);
            fprintf(stderr, "\n");
        }
        __atomic_store_n(&p->started, 1, __ATOMIC_RELEASE);
    }
    sys__os__lock_give(&x86_avx2__zzprivate_pools_lock);
    return p;
}

/* How many threads a launch on the calling thread's device runs on: its pool's, and on the whole machine its own. */
static inline uint32_t x86_avx2__zzpackage_threads(void) {
    const x86_avx2__pool* p = x86_avx2__zzpackage_start();
    return p->workers + (p->node >= 0 ? 0u : 1u);
}

/* ⭐ BINDING A DEVICE: the thread's launches go to that device's pool from here on. The thread is not moved. */
static inline void x86_avx2__zzpackage_bind(uint32_t device) {
    x86_avx2__zzpackage_device = device;
    (void)x86_avx2__zzpackage_start();
}

/* ⭐ ONE LAUNCH: `blocks` runs of `fn(arg)`, spread over the pool and the caller, returning when all are done.
 * ⛳ One launch at a time a pool: the caller is the worker that bound its device, and nn's doors are called from
 * it in order. Two workers on two devices launch at once, each on its own pool. */
static inline void x86_avx2__zzprivate_run_body(uint32_t blocks, void (*fn)(void*), void* arg);
static inline void x86_avx2__zzprivate_launch(x86_avx2__pool* p, uint32_t blocks, void (*fn)(void*), void* arg);
static inline void x86_avx2__zzpackage_run(uint32_t blocks, void (*fn)(void*), void* arg) {
    x86_avx2__zzprivate_run_body(blocks, fn, arg);
}

/* ⭐ ON THE WHOLE MACHINE THE CALLER RUNS ITS SHARE ON ITS OWN CORE. The pool leaves the caller's core free for it (▶ the
 *   sizing), but the caller is never pinned — a thread is where the scheduler puts it — and the scheduler is free to move it
 *   later onto a sibling of a pool thread's core, where two of a launch's runners share a core while another idles: the
 *   placement a bad sizing gave, 131 against 110 ms a 35B token on the CPU alone (`MEASURED`, ▶ the sizing). So a launch
 *   that finds its caller on a pool thread's core moves it to its own for the launch and hands its affinity back after: a
 *   `sys__os__cpu_now` a launch, and a move only when the caller was misplaced (`REASONED`; the case after a good sizing is not
 *   one this box has been seen to produce). */
static inline bool x86_avx2__zzprivate_caller_to_own(const x86_avx2__pool* p, sys__os__cpus* saved) {
    if (p->node >= 0 || sys__os__cpus_count(&p->own) == 0) return false;
    const int cpu = sys__os__cpu_now();
    if (cpu < 0 || cpu >= SYS__OS__CPUS_MOST || !sys__os__cpus_has(&p->worked, cpu)) return false;
    return sys__os__cpus_of_thread(saved) && sys__os__cpus_to_thread(&p->own);
}
static inline void x86_avx2__zzprivate_run_body(uint32_t blocks, void (*fn)(void*), void* arg) {
    x86_avx2__pool* p = x86_avx2__zzpackage_start();
    if (blocks == 0u) return;
    sys__os__cpus saved;
    const bool moved = p->workers != 0u && blocks > 1u && x86_avx2__zzprivate_caller_to_own(p, &saved);
    x86_avx2__zzprivate_launch(p, blocks, fn, arg);
    if (moved) (void)sys__os__cpus_to_thread(&saved);
}
static inline void x86_avx2__zzprivate_launch(x86_avx2__pool* p, uint32_t blocks, void (*fn)(void*), void* arg) {
    if (p->workers == 0u || blocks == 1u) {
        for (uint32_t b = 0u; b < blocks; ++b) { x86_avx2__zzpackage_block = b; x86_avx2__zzpackage_blocks = blocks; fn(arg); }
        x86_avx2__zzpackage_block = 0u; x86_avx2__zzpackage_blocks = 1u;
        return;
    }
    /* ⛔⛔ THE COUNTER FIRST, NAMING THE NEW GENERATION, THEN THE JOB, THEN THE GENERATION THAT WAKES THEM.
     *   With the job written first, a worker still taking blocks of the LAST launch read the counter's old
     *   generation — still its own — and the NEW launch's `blocks`: a block number past the old launch's end
     *   passed the bound, the claim won, and the worker ran the NEW `fn` on it. That block then ran again in
     *   its own launch, and `done` ran ahead so the caller could return with a block still running.
     *   `MEASURED`: harmless while the doors were slow; once the layer's doors were all AVX2 and
     *   back to back, one 35B prompt in four drifted — layer 32 at 0.199 against 0.133 — because a DeltaNet
     *   step and a residual sum write in place, and running twice is a wrong answer. Counter first: a late
     *   worker sees a generation that is not its own and leaves before it reads anything else. */
    const uint32_t g = __atomic_load_n(&p->generation, __ATOMIC_RELAXED) + 1u;
    __atomic_store_n(&p->next, (uint64_t)g << 32, __ATOMIC_RELEASE);
    p->fn = fn; p->arg = arg; p->blocks = blocks;
    __atomic_store_n(&p->done, 0u, __ATOMIC_RELAXED);
    sys__os__lock_take(&p->lock);
    __atomic_store_n(&p->generation, g, __ATOMIC_RELEASE);
    sys__os__condition_wake_all(&p->wake);
    sys__os__lock_give(&p->lock);
    if (p->node < 0) x86_avx2__zzprivate_take(p, g);         /* on a node the caller only waits */
    while (__atomic_load_n(&p->done, __ATOMIC_ACQUIRE) < blocks) _mm_pause();
    x86_avx2__zzpackage_block = 0u; x86_avx2__zzpackage_blocks = 1u;
}

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_POOL_CUH */
