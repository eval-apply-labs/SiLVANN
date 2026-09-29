#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_POOL_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_POOL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
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
 *     allowed   the CPUs this process may run on (`sched_getaffinity` — `taskset`, a cgroup's cpuset)
 *     cores     one per physical core among them, a sibling hyperthread never counted twice
 *     free      a core whose every CPU was idle more than half of a 50 ms sample of `/proc/stat`
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
    pthread_t         thread[X86_AVX2__THREADS_MAX];
    pthread_mutex_t   lock;
    pthread_cond_t    wake;
    /* the launch in flight */
    void            (*fn)(void*);
    void*             arg;
    uint32_t          blocks;
    uint64_t          next;                    /* the launch's generation above, the next block number below */
    uint32_t          done;                    /* blocks finished */
    uint32_t          generation;              /* one a launch, so a thread knows there is new work */
    int               started;
    int               node;                    /* the NUMA node it and its memory are on, or -1: every one */
} x86_avx2__pool;

static x86_avx2__pool x86_avx2__zzprivate_pools[X86_AVX2__DEVICES_MAX];
static pthread_mutex_t x86_avx2__zzprivate_pools_lock = PTHREAD_MUTEX_INITIALIZER;
/* the device the calling thread has bound — whose pool its launches run on, and whose node its memory is on */
static __thread uint32_t x86_avx2__zzpackage_device = 0u;
/* the block a thread is running, and how many the launch has — what nn's `block()` / `blocks()` answer */
static __thread uint32_t x86_avx2__zzpackage_block  = 0u;
static __thread uint32_t x86_avx2__zzpackage_blocks = 1u;

static inline void x86_avx2__zzprivate_pin(int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

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
    clock_gettime(CLOCK_MONOTONIC, &t);
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
    x86_avx2__zzprivate_pin(p->cpu[i]);
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
            pthread_mutex_lock(&p->lock);
            while ((g = __atomic_load_n(&p->generation, __ATOMIC_ACQUIRE)) == seen) pthread_cond_wait(&p->wake, &p->lock);
            pthread_mutex_unlock(&p->lock);
        }
        seen = g;
        x86_avx2__zzprivate_take(p, g);
    }
    return 0;
}

/* ── WHERE: the CPUs this process may use, and the nodes they are on ───────────────────────────────── */
static inline int x86_avx2__zzprivate_read_int(const char* path) {
    FILE* f = fopen(path, "r"); int v = -1;
    if (f) { if (fscanf(f, "%d", &v) != 1) v = -1; fclose(f); }
    return v;
}
enum { X86_AVX2__ZZPRIVATE_CPUS = 1024, X86_AVX2__ZZPRIVATE_NODES = 64 };
/* ⛳ READ ONCE, BEFORE ANY THREAD IS PINNED: binding a socket device pins the thread that binds it, and a thread's
 *   affinity is what `sched_getaffinity` answers — so a later read on a pinned thread would see one core. */
static cpu_set_t x86_avx2__zzprivate_allowed;
static int       x86_avx2__zzprivate_cpu_node[X86_AVX2__ZZPRIVATE_CPUS];
static int       x86_avx2__zzprivate_node[X86_AVX2__DEVICES_MAX - 1u];   /* socket device 1 + s is node [s] */
static uint32_t  x86_avx2__zzprivate_nodes = 0u;
static pthread_once_t x86_avx2__zzprivate_where_once = PTHREAD_ONCE_INIT;
static void x86_avx2__zzprivate_where(void) {
    CPU_ZERO(&x86_avx2__zzprivate_allowed);
    if (sched_getaffinity(0, sizeof x86_avx2__zzprivate_allowed, &x86_avx2__zzprivate_allowed) != 0) return;
    int seen[X86_AVX2__ZZPRIVATE_NODES] = {0};
    for (int cpu = 0; cpu < X86_AVX2__ZZPRIVATE_CPUS && cpu < CPU_SETSIZE; ++cpu) {
        x86_avx2__zzprivate_cpu_node[cpu] = -1;
        if (!CPU_ISSET(cpu, &x86_avx2__zzprivate_allowed)) continue;
        char path[128];
        for (int n = 0; n < X86_AVX2__ZZPRIVATE_NODES; ++n) {
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/node%d", cpu, n);
            if (access(path, F_OK) == 0) { x86_avx2__zzprivate_cpu_node[cpu] = n; seen[n] = 1; break; }
        }
    }
    for (int n = 0; n < X86_AVX2__ZZPRIVATE_NODES && x86_avx2__zzprivate_nodes < X86_AVX2__DEVICES_MAX - 1u; ++n)
        if (seen[n]) x86_avx2__zzprivate_node[x86_avx2__zzprivate_nodes++] = n;
    /* one node is the whole machine already: no socket devices beside device 0 */
    if (x86_avx2__zzprivate_nodes < 2u) x86_avx2__zzprivate_nodes = 0u;
}
/* How many devices: the whole machine, and a socket device a node when there is more than one. */
static inline uint32_t x86_avx2__zzpackage_devices(void) {
    pthread_once(&x86_avx2__zzprivate_where_once, x86_avx2__zzprivate_where);
    return 1u + x86_avx2__zzprivate_nodes;
}
/* The node device `index` is on, or -1 for the whole machine. */
static inline int x86_avx2__zzpackage_node_of(uint32_t index) {
    pthread_once(&x86_avx2__zzprivate_where_once, x86_avx2__zzprivate_where);
    return index == 0u || index > x86_avx2__zzprivate_nodes ? -1 : x86_avx2__zzprivate_node[index - 1u];
}

/* ── HOW MANY: the free physical cores of the device ───────────────────────────────────────────────── */
/* Each CPU's busy and total ticks from `/proc/stat`, indexed by CPU number. */
static inline void x86_avx2__zzprivate_ticks(unsigned long long* busy, unsigned long long* total, int most) {
    FILE* f = fopen("/proc/stat", "r"); char line[512];
    if (!f) return;
    while (fgets(line, sizeof line, f)) {
        int cpu; unsigned long long u, n, s, idle, io, irq, sirq, steal;
        if (sscanf(line, "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu", &cpu, &u, &n, &s, &idle, &io, &irq, &sirq, &steal) == 9
            && cpu >= 0 && cpu < most) {
            busy[cpu] = u + n + s + irq + sirq + steal; total[cpu] = busy[cpu] + idle + io;
        }
    }
    fclose(f);
}
/* The pool's threads' CPUs into `cpus_out`, and their count. On the whole machine the caller's own core is left
 * out, as it always was — it runs blocks too; on a node every free core has a thread, and the caller runs none. */
static inline uint32_t x86_avx2__zzprivate_size(int* cpus_out, int node) {
    enum { MOST = X86_AVX2__ZZPRIVATE_CPUS };
    static unsigned long long b0[MOST], t0[MOST], b1[MOST], t1[MOST];
    memset(b0, 0, sizeof b0); memset(t0, 0, sizeof t0); memset(b1, 0, sizeof b1); memset(t1, 0, sizeof t1);
    x86_avx2__zzprivate_ticks(b0, t0, MOST);
    const struct timespec nap = { 0, 50 * 1000 * 1000 };
    nanosleep(&nap, 0);
    x86_avx2__zzprivate_ticks(b1, t1, MOST);
    /* per physical core (package, core id): its first allowed CPU, and whether every CPU of it is idle */
    int core_cpu[MOST], core_pkg[MOST], core_id[MOST], core_busy[MOST], cores = 0;
    for (int cpu = 0; cpu < MOST && cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &x86_avx2__zzprivate_allowed)) continue;
        if (node >= 0 && x86_avx2__zzprivate_cpu_node[cpu] != node) continue;
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        const int pkg = x86_avx2__zzprivate_read_int(path);
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        const int id = x86_avx2__zzprivate_read_int(path);
        const unsigned long long dt = t1[cpu] - t0[cpu], db = b1[cpu] - b0[cpu];
        const int busy = dt > 0ull && db * 2ull > dt;
        int k = 0;
        while (k < cores && !(core_pkg[k] == pkg && core_id[k] == id)) ++k;
        if (k == cores) { core_cpu[k] = cpu; core_pkg[k] = pkg; core_id[k] = id; core_busy[k] = 0; ++cores; }
        core_busy[k] |= busy;
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
    const int self = node >= 0 ? -1 : sched_getcpu();
    const uint32_t caller = node >= 0 ? 0u : 1u;             /* whether the caller is one of the threads */
    uint32_t workers = 0u;
    const char* cap_s = getenv("SILVANN_CPU_THREADS");
    const long cap = cap_s ? strtol(cap_s, 0, 10) : 0;
    for (int k = 0; k < n && workers < X86_AVX2__THREADS_MAX; ++k) {
        if (order[k] == self) continue;
        if (cap > 0 && (long)(workers + caller) >= cap) break;
        cpus_out[workers++] = order[k];
    }
    if (caller == 1u && n > 0 && workers + 1u > (uint32_t)n) workers = (uint32_t)n - 1u;
    return workers;
}

/* The pool of the device the calling thread has bound, stood up the first time: sized, a thread pinned to each
 * chosen core. */
static inline x86_avx2__pool* x86_avx2__zzpackage_start(void) {
    const uint32_t device = x86_avx2__zzpackage_device < x86_avx2__zzpackage_devices() ? x86_avx2__zzpackage_device : 0u;
    x86_avx2__pool* p = &x86_avx2__zzprivate_pools[device];
    if (__atomic_load_n(&p->started, __ATOMIC_ACQUIRE)) return p;
    pthread_mutex_lock(&x86_avx2__zzprivate_pools_lock);
    if (!p->started) {
        pthread_mutex_init(&p->lock, 0);
        pthread_cond_init(&p->wake, 0);
        p->node = x86_avx2__zzpackage_node_of(device);
        p->workers = x86_avx2__zzprivate_size(p->cpu, p->node);
        for (uint32_t i = 0u; i < p->workers; ++i)
            if (pthread_create(&p->thread[i], 0, x86_avx2__zzprivate_worker, (void*)(uintptr_t)((device << 16) | i)) != 0) { p->workers = i; break; }
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
    pthread_mutex_unlock(&x86_avx2__zzprivate_pools_lock);
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
static inline void x86_avx2__zzpackage_run(uint32_t blocks, void (*fn)(void*), void* arg) {
    x86_avx2__pool* p = x86_avx2__zzpackage_start();
    if (blocks == 0u) return;
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
    pthread_mutex_lock(&p->lock);
    __atomic_store_n(&p->generation, g, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->lock);
    if (p->node < 0) x86_avx2__zzprivate_take(p, g);         /* on a node the caller only waits */
    while (__atomic_load_n(&p->done, __ATOMIC_ACQUIRE) < blocks) _mm_pause();
    x86_avx2__zzpackage_block = 0u; x86_avx2__zzpackage_blocks = 1u;
}

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_POOL_CUH */
