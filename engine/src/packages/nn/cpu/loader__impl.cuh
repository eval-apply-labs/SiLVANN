#ifndef SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <pthread.h>
#include <time.h>
#include <stdlib.h>
#include "expert__header.cuh"                           /* the index this reads into, and what it publishes */
#include "../../sys/cpu/silicon/silicon__header.cuh"   /* the family's file_read */

/* ══ ⭐⭐ THE LOADER — AN EXPERT'S DISK READ, OFF THE EVALUATOR'S THREAD ════════════════════════════════════
 * ⚖ *"go ahead with the loader pool and prefetch"*. A few host threads do nothing but read: each takes a queued
 * READ — an expert's four slices, from its file into a slot reserved for it — and does it with the worker
 * family's `file_read`. Everything the INDEX knows stays on the evaluator's thread: the slot is taken (or an
 * old expert evicted for it) when the read is queued, and the expert is admitted when the read is settled.
 * ⛳ SO A LOADER THREAD TOUCHES BYTES AND NEVER THE INDEX, and a read in flight is in no LRU chain — nothing
 *   can choose it as a victim while its bytes are arriving.
 * ⛳ A READ IS SETTLED BY WHOEVER NEEDS ITS LAYER: `settle` waits for every read of a (layer, type) band and
 *   admits each, whether a program picked it or a prediction did — so no reserved slot outlives its layer.
 * The threads start the first time a read is queued (`SILVANN_EXPERT_LOADERS`, 4 by default) and are joined
 * at the package's teardown. */

#define NN__LOADER__FREE     0u
#define NN__LOADER__QUEUED   1u
#define NN__LOADER__READING  2u
#define NN__LOADER__DONE     3u
#define NN__LOADER__FAILED   4u
#define NN__LOADER__WAITING  5u   /* half duplex: an evacuation until its gather lands, a write until the CPUs are done — no thread takes it */

typedef struct nn__loader__read {
    uint32_t state;
    bool     predicted;            /* queued by a prediction, not by a pick */
    uint64_t owner;                /* the index of the worker that asked — several workers share these threads */
    sys__silicon_family__id family;
    uint64_t layer, type, expert, slot, room;
    nn__expert__backing backing;
    uint64_t seq;                  /* when it was queued: the threads take the oldest first */
    bool gather;                   /* a promotion: `runs` gathered from memory into `room`, `bytes` long, not a file read */
    uint64_t bytes;
    nn__expert__gather runs;
    uint32_t job;                  /* what it is — ▶ NN__LOADER__JOB_*; `gather` stays true for a GATHER */
    unsigned channel;              /* an EVACUATE's or a WRITE's evacuation channel */
    uint64_t by;                   /* a WRITE: the worker whose commit queued it, the one whose program releases it */
    uint64_t queued_ns;            /* a GATHER: when it was queued, for the time it takes to land */
} nn__loader__read;
/* ⭐ WHAT A JOB IS. A READ (a file into a slot) and a GATHER (a promotion: the CPUs' memory into a card slot) are admitted
 *   by whoever settles their layer. The exclusive tier's two are not: an EVACUATE (a card slot's bytes into an evacuation
 *   channel) waits for the commit that sends its expert to RAM, and a WRITE (a channel into an expert's RAM slot) frees
 *   itself — what waits for it is a CPU about to compute that expert (`nn__expert__wait_written`). So the settles, the
 *   finds and a promotion's in-flight test see only the first two. */
#define NN__LOADER__JOB_READ      0u
#define NN__LOADER__JOB_GATHER    1u
#define NN__LOADER__JOB_EVACUATE  2u
#define NN__LOADER__JOB_WRITE     3u
#define NN__LOADER__ADMITTED(r)   ((r)->job == NN__LOADER__JOB_READ || (r)->job == NN__LOADER__JOB_GATHER)

static nn__loader__read nn__loader__zzprivate_reads[NN__EXPERT__FLIGHT_MAX];
static pthread_mutex_t  nn__loader__zzprivate_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   nn__loader__zzprivate_work = PTHREAD_COND_INITIALIZER;   /* a read was queued */
static pthread_cond_t   nn__loader__zzprivate_done = PTHREAD_COND_INITIALIZER;   /* a read finished   */
static pthread_t        nn__loader__zzprivate_threads[NN__EXPERT__LOADERS_MAX];
static unsigned         nn__loader__zzprivate_started = 0u;
static bool             nn__loader__zzprivate_stopping = false;
static uint64_t         nn__loader__zzprivate_count[NN__EXPERT__COUNTS];
static uint64_t         nn__loader__zzprivate_seq = 0ull;             /* the queue's clock, under the lock */
/* writes to RAM queued or in flight — an exclusive tier's demotions — so a wait for one costs nothing where there are none */
static uint64_t         nn__loader__zzprivate_writes_out = 0ull;
/* The evacuation channels (▶ the exclusive tier, below): pinned memory an expert's bytes wait in, between the card and the
 * RAM slots they are written into. `held` from the evacuation until the commit hands the bytes on; `pending` the writes
 * still reading it — a channel is reused only when both are clear, and its next user waits while it is pending. */
typedef struct nn__loader__channel { void* host; uint64_t bytes; bool held; unsigned pending; } nn__loader__channel;
static nn__loader__channel nn__loader__zzprivate_channels[NN__EXPERT_TIER__CHANNELS];

/* A promotion's bytes, on a loader thread: the runs copied into this thread's pinned staging buffer in the slot's layout,
 * then written to the card's slot on this thread's side channel — which overlaps the kernels on the card's own stream. */
static bool nn__loader__zzprivate_gather(const nn__loader__read* r) {
    static __thread void* stage = 0;
    static __thread void* stage_card = 0;
    static __thread uint64_t stage_bytes = 0ull;
    static __thread void* channel = 0;
    if (stage_bytes < r->bytes) {
        if (stage != 0) sys__gpu__memory_free_host_ram(r->family, stage);
        stage = 0; stage_bytes = 0ull;
        if (!sys__gpu__memory_allocate_host_ram_mapped(r->family, &stage, &stage_card, (size_t)r->bytes)) return false;
        stage_bytes = r->bytes;
    }
    if (channel == 0 && !sys__gpu__side_open(r->family, &channel)) return false;
    /* the runs are this process's own memory — a CPU worker's slots — so they are read through the host family's door */
    static __thread sys__silicon_family__id host = SYS__SILICON_FAMILY__NONE;
    if (host == SYS__SILICON_FAMILY__NONE) host = sys__silicon_family__named("host");
    for (unsigned i = 0u; i < r->runs.runs; ++i) {
        const nn__expert__run* u = &r->runs.run[i];
        for (uint64_t c = 0ull; c < u->count; ++c) {
            if (u->into + c * u->into_stride + u->bytes > r->bytes) return false;
            if (!sys__gpu__memory_read(host, (uint8_t*)stage + u->into + c * u->into_stride,
                                       (const void*)(uintptr_t)(u->from + c * u->from_stride), (size_t)u->bytes))
                return false;
        }
    }
    return sys__gpu__side_memory_write(r->family, (void*)(uintptr_t)r->room, stage, (size_t)r->bytes, channel);
}

/* An EVACUATE: the card slot at `room` read into its channel, on this thread's side channel — beside the kernels. */
static bool nn__loader__zzprivate_evacuate(const nn__loader__read* r) {
    static __thread void* side = 0;
    if (side == 0 && !sys__gpu__side_open(r->family, &side)) return false;
    const nn__loader__channel* c = &nn__loader__zzprivate_channels[r->channel];
    if (r->channel >= NN__EXPERT_TIER__CHANNELS || c->host == 0 || c->bytes < r->bytes) return false;
    return sys__gpu__side_memory_read(r->family, c->host, (const void*)(uintptr_t)r->room, (size_t)r->bytes, side);
}
/* A WRITE: the runs copied out of the channel into an expert's RAM slot — both this process's memory, `from` and `into`
 *   absolute addresses. */
static bool nn__loader__zzprivate_write(const nn__loader__read* r) {
    static __thread sys__silicon_family__id host = SYS__SILICON_FAMILY__NONE;
    if (host == SYS__SILICON_FAMILY__NONE) host = sys__silicon_family__named("host");
    for (unsigned i = 0u; i < r->runs.runs; ++i) {
        const nn__expert__run* u = &r->runs.run[i];
        for (uint64_t c = 0ull; c < u->count; ++c)
            if (!sys__gpu__memory_read(host, (void*)(uintptr_t)(u->into + c * u->into_stride),
                                       (const void*)(uintptr_t)(u->from + c * u->from_stride), (size_t)u->bytes))
                return false;
    }
    return true;
}

static void* nn__loader__zzprivate_run(void* unused) {
    (void)unused;
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    for (;;) {
        /* the oldest queued first — so a layer's promotions land in the order its verb will compute them */
        nn__loader__read* job = 0;
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i)
            if (nn__loader__zzprivate_reads[i].state == NN__LOADER__QUEUED && (job == 0 || nn__loader__zzprivate_reads[i].seq < job->seq))
                job = &nn__loader__zzprivate_reads[i];
        if (job == 0) {
            if (nn__loader__zzprivate_stopping) break;
            pthread_cond_wait(&nn__loader__zzprivate_work, &nn__loader__zzprivate_lock);
            continue;
        }
        job->state = NN__LOADER__READING;
        const nn__loader__read r = *job;
        pthread_mutex_unlock(&nn__loader__zzprivate_lock);
        bool ok = true;
        if (r.job == NN__LOADER__JOB_EVACUATE) ok = nn__loader__zzprivate_evacuate(&r);
        else if (r.job == NN__LOADER__JOB_WRITE) ok = nn__loader__zzprivate_write(&r);
        else if (r.gather) ok = nn__loader__zzprivate_gather(&r);
        for (unsigned i = 0u; ok && r.job == NN__LOADER__JOB_READ && i < NN__EXPERT__SLICES; ++i) {
            if (r.backing.bytes[i] == 0ull) continue;
            ok = sys__gpu__file_read(r.family, r.backing.file, r.backing.from[i] + r.expert * r.backing.bytes[i], r.backing.bytes[i],
                                     (void*)(uintptr_t)(r.room + r.backing.into[i]));
        }
        pthread_mutex_lock(&nn__loader__zzprivate_lock);
        if (r.job == NN__LOADER__JOB_WRITE) {
            /* a write frees itself: its expert is in RAM now, its channel one reader fewer. ⛔ A FAILED WRITE IS LOUD — the
             *   expert's RAM slot holds the wrong bytes, and nothing could tell — so it is counted, and the run stops */
            if (r.channel < NN__EXPERT_TIER__CHANNELS && nn__loader__zzprivate_channels[r.channel].pending > 0u)
                --nn__loader__zzprivate_channels[r.channel].pending;
            if (!ok) ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_WRITE_FAILED];
            job->state = NN__LOADER__FREE;
            __atomic_sub_fetch(&nn__loader__zzprivate_writes_out, 1ull, __ATOMIC_RELEASE);
        } else {
            job->state = ok ? NN__LOADER__DONE : NN__LOADER__FAILED;
        }
        pthread_cond_broadcast(&nn__loader__zzprivate_done);
    }
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return 0;
}

/* The threads, the first time they are wanted. Called with the lock held. */
static bool nn__loader__zzprivate_start(void) {
    if (nn__loader__zzprivate_started != 0u) return true;
    unsigned want = NN__EXPERT__LOADERS_STANDARD;
    const char* said = getenv("SILVANN_EXPERT_LOADERS");
    if (said != 0) { const long n = strtol(said, 0, 10); if (n > 0) want = (unsigned)n; }
    if (want > NN__EXPERT__LOADERS_MAX) want = NN__EXPERT__LOADERS_MAX;
    nn__loader__zzprivate_stopping = false;
    for (unsigned t = 0u; t < want; ++t) {
        if (pthread_create(&nn__loader__zzprivate_threads[t], 0, nn__loader__zzprivate_run, 0) != 0) break;
        ++nn__loader__zzprivate_started;
    }
    return nn__loader__zzprivate_started != 0u;
}

/* The package's teardown: every thread told to stop once the queue is empty, and joined. */
static void nn__loader__zzpackage_stop(void) {
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    const unsigned n = nn__loader__zzprivate_started;
    nn__loader__zzprivate_stopping = true;
    pthread_cond_broadcast(&nn__loader__zzprivate_work);
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    for (unsigned t = 0u; t < n; ++t) pthread_join(nn__loader__zzprivate_threads[t], 0);
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    nn__loader__zzprivate_started = 0u;
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
}

/* This worker's read of `expert` in flight, or none. Called with the lock held. ⛳ A READ IS ITS WORKER'S: two workers of
 *   one program each keep their own index and slots, and the same expert of each is two reads into two slots. */
static nn__loader__read* nn__loader__zzprivate_find(uint64_t owner, uint64_t layer, uint64_t type, uint64_t expert) {
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__FREE && NN__LOADER__ADMITTED(r) && r->owner == owner && r->layer == layer && r->type == type
         && r->expert == expert) return r;
    }
    return 0;
}

/* A slot for a read. ⭐ EVERY LAYER KEEPS ITS FAIR SHARE — the collection's slots over the layers that have experts of the
 * type: a layer holding its share gives up its own least recently used expert (one not `pinned`); one holding less takes
 * a free slot, and with none free, the least recently used of the layer that holds the most.
 * ⛳ WITHOUT THE SHARE, the first layers of a prompt read as rows took every free slot — a chunk wants most of a layer's
 *   experts at once — and a token's misses, each evicting from its own layer, never gave them back: `MEASURED` on the
 *   35B with 8 GB, 83% of reads from memory after a prompt of rows against 94% after one read a position at a time. */
/* Layer `l`'s least recently used expert that is not one of `pinned` (they are the asking layer's), evicted for its slot.
 * ⛳ A PINNED ONE CAN BE THE OLDEST: a batch's resident experts move to the front only as the batch's requests reach them,
 *   so the walk goes on toward the newer ones — at most one step a pinned expert. */
static bool nn__loader__zzprivate_evict_oldest(uint64_t l, uint64_t type, uint64_t layer, const uint64_t* pinned, unsigned npinned,
                                               uint64_t* slot, uint64_t* room) {
    uint64_t victim = 0ull;
    if (nn__expert__layer_experts(l, type) == 0ull || !nn__expert__oldest(l, type, &victim)) return false;
    for (unsigned step = 0u; step <= npinned; ++step) {
        bool held = false;
        for (unsigned p = 0u; l == layer && p < npinned; ++p) held = held || pinned[p] == victim;
        if (!held) return nn__expert__evict(l, type, victim) && nn__expert__type_take(type, slot, room);
        sys__heap_node me;
        if (!nn__expert__slot(l, type, victim, &me) || me.args[NN__EXPERT__SLOT_PREV] == NN__EXPERT__NONE) return false;
        victim = NN__EXPERT__UNLINK(me.args[NN__EXPERT__SLOT_PREV]);
    }
    return false;
}
static bool nn__loader__zzprivate_room(uint64_t layer, uint64_t type, const uint64_t* pinned, unsigned npinned,
                                       uint64_t* slot, uint64_t* room) {
    const uint64_t layers = nn__expert__index_layers();
    uint64_t banded = 0ull;
    for (uint64_t l = 0ull; l < layers; ++l) banded += nn__expert__layer_experts(l, type) != 0ull ? 1ull : 0ull;
    const uint64_t share = banded == 0ull ? 0ull : nn__expert__type_slots(type) / banded;
    if (share != 0ull && nn__expert__layer_resident(layer, type) >= share
     && nn__loader__zzprivate_evict_oldest(layer, type, layer, pinned, npinned, slot, room))
        return true;
    if (nn__expert__type_take(type, slot, room)) return true;
    uint64_t most = layers, held = 0ull;
    for (uint64_t l = 0ull; l < layers; ++l) {
        const uint64_t n = nn__expert__layer_resident(l, type);
        if (n > held) { held = n; most = l; }
    }
    if (most < layers && nn__loader__zzprivate_evict_oldest(most, type, layer, pinned, npinned, slot, room)) return true;
    return nn__loader__zzprivate_evict_oldest(layer, type, layer, pinned, npinned, slot, room);
}


static __device__ inline int nn__expert__request(sys__silicon_family__id family, uint64_t layer, uint64_t type, uint64_t expert,
                                                 const nn__expert__backing* backing, bool predicted,
                                                 const uint64_t* pinned, unsigned npinned, uint64_t* at) {
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, expert, &me)) return NN__EXPERT__REFUSED;
    if (me.args[NN__EXPERT__SLOT_AT] != 0ull) {
        if (at != 0) *at = me.args[NN__EXPERT__SLOT_AT];
        return nn__expert__touch(layer, type, expert) ? NN__EXPERT__RESIDENT : NN__EXPERT__REFUSED;
    }
    const uint64_t owner = nn__expert__zzpackage_owner();
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    nn__loader__read* r = nn__loader__zzprivate_find(owner, layer, type, expert);
    if (r != 0) {
        if (!predicted && r->predicted) { r->predicted = false; ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_PREDICTED_USED]; }
        pthread_mutex_unlock(&nn__loader__zzprivate_lock);
        return NN__EXPERT__ARRIVING;
    }
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    if (backing == 0 || backing->file == 0ull) return NN__EXPERT__REFUSED;

    uint64_t slot = 0ull, room = 0ull;
    if (!nn__loader__zzprivate_room(layer, type, pinned, npinned, &slot, &room)) return NN__EXPERT__REFUSED;
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    nn__loader__read* free_read = 0;
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX && free_read == 0; ++i)
        if (nn__loader__zzprivate_reads[i].state == NN__LOADER__FREE) free_read = &nn__loader__zzprivate_reads[i];
    if (free_read == 0 || !nn__loader__zzprivate_start()) {
        pthread_mutex_unlock(&nn__loader__zzprivate_lock);
        (void)nn__expert__type_give(type, slot);
        return NN__EXPERT__REFUSED;
    }
    free_read->state = NN__LOADER__QUEUED;
    free_read->predicted = predicted;
    free_read->owner = owner;
    free_read->family = family;
    free_read->layer = layer; free_read->type = type; free_read->expert = expert;
    free_read->slot = slot; free_read->room = room;
    free_read->backing = *backing;
    free_read->gather = false;
    free_read->job = NN__LOADER__JOB_READ;
    free_read->seq = ++nn__loader__zzprivate_seq;
    ++nn__loader__zzprivate_count[predicted ? NN__EXPERT__COUNT_PREDICTED : NN__EXPERT__COUNT_MISSES];
    pthread_cond_signal(&nn__loader__zzprivate_work);
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return NN__EXPERT__ARRIVING;
}

/* ⭐ A WAIT FOR A READ, TIMED — the lock held, as `pthread_cond_wait` needs it: the time is what a layer spent waiting on
 *   the disk, the thing a cache's misses cost beyond the compute that overlaps them (`nn__expert__count`). */
/* Now, in ns, for the gathers' time to land. */
static inline uint64_t nn__loader__zzprivate_now_ns(void) {
    struct timespec a;
    clock_gettime(CLOCK_MONOTONIC, &a);
    return (uint64_t)a.tv_sec * 1000000000ull + (uint64_t)a.tv_nsec;
}

/* A gather landed and settled: its time from queued to now counted. Lock held. */
static inline void nn__loader__zzprivate_landed(const nn__loader__read* r) {
    if (r->job != NN__LOADER__JOB_GATHER || r->queued_ns == 0ull) return;
    nn__loader__zzprivate_count[NN__EXPERT__COUNT_LAND_NS] += nn__loader__zzprivate_now_ns() - r->queued_ns;
    ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_LANDED];
}

static inline void nn__loader__zzprivate_wait(void) {
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    pthread_cond_wait(&nn__loader__zzprivate_done, &nn__loader__zzprivate_lock);
    clock_gettime(CLOCK_MONOTONIC, &b);
    nn__loader__zzprivate_count[NN__EXPERT__COUNT_WAIT_NS] += (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull + (uint64_t)(b.tv_nsec - a.tv_nsec);
    ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_WAITS];
}

static __device__ inline bool nn__expert__settle(uint64_t layer, uint64_t type) {
    bool ok = true;
    const uint64_t owner = nn__expert__zzpackage_owner();
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    for (;;) {
        bool waiting = false;
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
            nn__loader__read* r = &nn__loader__zzprivate_reads[i];
            if (r->state == NN__LOADER__FREE || !NN__LOADER__ADMITTED(r) || r->owner != owner || r->layer != layer || r->type != type) continue;
            if (r->state == NN__LOADER__QUEUED || r->state == NN__LOADER__READING) { waiting = true; continue; }
            /* done or failed: the index learns of it here, on this thread */
            const bool read = r->state == NN__LOADER__DONE;
            const uint64_t slot = r->slot, room = r->room, expert = r->expert;
            r->state = NN__LOADER__FREE;
            if (!read || !nn__expert__admit(layer, expert, room, type)) {
                (void)nn__expert__type_give(type, slot);
                ok = false;
            }
        }
        if (!waiting) break;
        nn__loader__zzprivate_wait();
    }
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return ok;
}

static __device__ inline int nn__expert__promote(sys__silicon_family__id family, uint64_t layer, uint64_t type, uint64_t expert,
                                                 const nn__expert__gather* gather, const uint64_t* pinned, unsigned npinned) {
    sys__heap_node me;
    if (gather == 0 || gather->runs == 0u || gather->runs > NN__EXPERT__GATHER_RUNS || !nn__expert__slot(layer, type, expert, &me))
        return NN__EXPERT__REFUSED;
    if (me.args[NN__EXPERT__SLOT_AT] != 0ull) return NN__EXPERT__RESIDENT;
    const uint64_t owner = nn__expert__zzpackage_owner(), bytes = nn__expert__type_bytes(type);
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    nn__loader__read* free_read = 0;
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__FREE && NN__LOADER__ADMITTED(r) && r->owner == owner && r->layer == layer && r->type == type
         && r->expert == expert) {
            pthread_mutex_unlock(&nn__loader__zzprivate_lock);
            return NN__EXPERT__ARRIVING;
        }
        if (r->state == NN__LOADER__FREE && free_read == 0) free_read = r;
    }
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    if (free_read == 0 || bytes == 0ull) return NN__EXPERT__REFUSED;      /* no room in flight: the CPU keeps it this time */
    uint64_t slot = 0ull, room = 0ull;
    if (!nn__loader__zzprivate_room(layer, type, pinned, npinned, &slot, &room)) return NN__EXPERT__REFUSED;
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    if (free_read->state != NN__LOADER__FREE || !nn__loader__zzprivate_start()) {
        pthread_mutex_unlock(&nn__loader__zzprivate_lock);
        (void)nn__expert__type_give(type, slot);
        return NN__EXPERT__REFUSED;
    }
    free_read->state = NN__LOADER__QUEUED;
    free_read->predicted = false;
    free_read->owner = owner;
    free_read->family = family;
    free_read->layer = layer; free_read->type = type; free_read->expert = expert;
    free_read->slot = slot; free_read->room = room;
    free_read->gather = true; free_read->bytes = bytes; free_read->runs = *gather;
    free_read->job = NN__LOADER__JOB_GATHER;
    free_read->seq = ++nn__loader__zzprivate_seq;
    free_read->queued_ns = nn__loader__zzprivate_now_ns();
    ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_PROMOTED];
    pthread_cond_signal(&nn__loader__zzprivate_work);
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return NN__EXPERT__ARRIVING;
}

static __device__ inline unsigned nn__expert__flight_free(void) {
    unsigned n = 0u;
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) n += nn__loader__zzprivate_reads[i].state == NN__LOADER__FREE;
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return n;
}

static __device__ inline bool nn__expert__settle_ready(uint64_t layer, uint64_t type) {
    bool ok = true;
    const uint64_t owner = nn__expert__zzpackage_owner();
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (!NN__LOADER__ADMITTED(r) || r->owner != owner || r->layer != layer || r->type != type) continue;
        if (r->state != NN__LOADER__DONE && r->state != NN__LOADER__FAILED) continue;      /* free, or still on its way */
        const bool landed = r->state == NN__LOADER__DONE;
        const uint64_t slot = r->slot, room = r->room, expert = r->expert;
        if (landed) nn__loader__zzprivate_landed(r);
        r->state = NN__LOADER__FREE;
        if (!landed || !nn__expert__admit(layer, expert, room, type)) {
            (void)nn__expert__type_give(type, slot);
            ok = false;
        }
    }
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return ok;
}

static __device__ inline bool nn__expert__settle_one(uint64_t layer, uint64_t type, uint64_t expert) {
    bool ok = true;
    const uint64_t owner = nn__expert__zzpackage_owner();
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    for (;;) {
        nn__loader__read* r = nn__loader__zzprivate_find(owner, layer, type, expert);
        if (r == 0) break;
        if (r->state == NN__LOADER__QUEUED || r->state == NN__LOADER__READING) {
            nn__loader__zzprivate_wait();
            continue;
        }
        const bool read = r->state == NN__LOADER__DONE;
        const uint64_t slot = r->slot, room = r->room;
        r->state = NN__LOADER__FREE;
        if (!read || !nn__expert__admit(layer, expert, room, type)) {
            (void)nn__expert__type_give(type, slot);
            ok = false;
        }
        break;
    }
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return ok;
}

static __device__ inline bool nn__expert__in_memory(sys__silicon_family__id family, uint64_t type, uint64_t at) {
    return sys__gpu__memory_prefetch(family, (const void*)(uintptr_t)at, (size_t)nn__expert__type_bytes(type));
}

static __device__ inline uint64_t nn__expert__count(unsigned which) {
    if (which >= NN__EXPERT__COUNTS) return 0ull;
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    const uint64_t n = nn__loader__zzprivate_count[which];
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return n;
}

static __device__ inline bool nn__expert__backing_of(uint64_t array, nn__expert__backing* backing) {
    if (backing == 0 || !sys__node_array__is(array) || sys__node_array__length(array) < NN__EXPERT__BACKING__LENGTH) return false;
    uint64_t v[NN__EXPERT__BACKING__LENGTH];
    sys__node_array_walk w;
    if (!sys__node_array__walk(array, 0ull, &w)) return false;
    for (unsigned i = 0u; i < NN__EXPERT__BACKING__LENGTH; ++i, sys__node_array__next(&w)) {
        const sys__heap_node* n = sys__node_array__walk_cell(&w);
        if (n == 0 || n->dtype != SYS__KIND__VALUE_INT) return false;
        v[i] = n->args[0];
    }
    backing->file = v[NN__EXPERT__BACKING__FILE];
    for (unsigned i = 0u; i < NN__EXPERT__SLICES; ++i) {
        backing->from[i] = v[NN__EXPERT__BACKING__FROM(i)];
        backing->bytes[i] = v[NN__EXPERT__BACKING__BYTES(i)];
        backing->into[i] = v[NN__EXPERT__BACKING__INTO(i)];
    }
    return true;
}

/* ── A CARD'S TIER ───────────────────────────────────────────────────────────────────────────────────── ▶ the header */
static __device__ inline bool nn__expert_tier__of(uint64_t binding, uint64_t layer, nn__expert_tier* tier) {
    if (tier == 0 || !sys__node_array__is(binding) || sys__node_array__length(binding) <= (uint64_t)sys__silicon__worker()) return false;
    const sys__heap_node entry = sys__node_array__borrow(binding, (uint64_t)sys__silicon__worker());
    if (!sys__heap_node__carries_reference(entry.dtype) || !sys__node_array__is(entry.args[0])
     || sys__node_array__length(entry.args[0]) < NN__EXPERT_TIER__LENGTH) return false;
    uint64_t v[NN__EXPERT_TIER__LENGTH];
    sys__node_array_walk w;
    if (!sys__node_array__walk(entry.args[0], 0ull, &w)) return false;
    for (unsigned i = 0u; i < NN__EXPERT_TIER__LENGTH; ++i, sys__node_array__next(&w)) {
        const sys__heap_node* n = sys__node_array__walk_cell(&w);
        if (n == 0) return false;
        if (i == NN__EXPERT_TIER__SOURCES && sys__heap_node__carries_reference(n->dtype) && sys__node_array__is(n->args[0])) {
            v[i] = n->args[0];
            continue;
        }
        if (n->dtype != SYS__KIND__VALUE_INT || (i == NN__EXPERT_TIER__SOURCES && n->args[0] != 0ull)) return false;
        v[i] = n->args[0];
    }
    /* this layer's sources, or none */
    uint64_t sources = 0u;
    if (v[NN__EXPERT_TIER__SOURCES] != 0ull && layer < sys__node_array__length(v[NN__EXPERT_TIER__SOURCES])) {
        const sys__heap_node at = sys__node_array__borrow(v[NN__EXPERT_TIER__SOURCES], layer);
        if (sys__heap_node__carries_reference(at.dtype) && sys__node_array__is(at.args[0])) sources = at.args[0];
        else if (at.dtype != SYS__KIND__VALUE_INT || at.args[0] != 0ull) return false;
    }
    uint64_t ex[4] = {0u, 0u, 0u, 0u};                      /* EXCLUSIVE, PART_WORKER, PART_TYPE, DUPLEX — absent: inclusive */
    if (sys__node_array__length(entry.args[0]) >= NN__EXPERT_TIER__LENGTH_EXCLUSIVE)
        for (unsigned i = 0u; i < 4u; ++i) {
            const sys__heap_node n = sys__node_array__borrow(entry.args[0], NN__EXPERT_TIER__EXCLUSIVE + i);
            if (n.dtype != SYS__KIND__VALUE_INT) return false;
            ex[i] = n.args[0];
        }
    nn__expert_tier t = {
        sources, layer, v[NN__EXPERT_TIER__TYPE], v[NN__EXPERT_TIER__EXPERTS], v[NN__EXPERT_TIER__TOP_K],
        v[NN__EXPERT_TIER__HIDDEN], v[NN__EXPERT_TIER__INTER], v[NN__EXPERT_TIER__UP_LUT], v[NN__EXPERT_TIER__DOWN],
        v[NN__EXPERT_TIER__DOWN_LUT], v[NN__EXPERT_TIER__PARTS], v[NN__EXPERT_TIER__P_UP_LUT], v[NN__EXPERT_TIER__P_DOWN],
        v[NN__EXPERT_TIER__P_DOWN_LUT], v[NN__EXPERT_TIER__GATE_ROW], v[NN__EXPERT_TIER__DOWN_ROW], v[NN__EXPERT_TIER__LANDING],
        ex[0], ex[1], ex[2], ex[3], v[NN__EXPERT_TIER__SOURCES], v[NN__EXPERT_TIER__TOP_K] };
    if (t.exclusive > 1u || t.duplex > 1u || (t.exclusive == 1u && t.layers == 0u)) return false;   /* exclusive needs its sources to move */
    if (t.experts == 0u || t.top_k == 0u || t.top_k > t.experts || t.top_k > NN__EXPERT__TIER_K_MAX || t.parts == 0u
     || 5u * t.parts + 1u > NN__EXPERT__GATHER_RUNS || t.inter % t.parts != 0u || t.down_row % t.parts != 0u
     || (sources != 0u && sys__node_array__length(sources) < t.experts * t.parts))
        return false;
    if (sys__node_array__length(entry.args[0]) >= NN__EXPERT_TIER__LENGTH_LINE) {
        const sys__heap_node n = sys__node_array__borrow(entry.args[0], NN__EXPERT_TIER__LANDING_LINE);
        if (n.dtype != SYS__KIND__VALUE_INT) return false;
        t.landing_line = n.args[0];
    }
    *tier = t;
    return true;
}

/* Part `p`'s runs, gathering it from `src` (the part's RAM slot) into a whole card slot — its gate rows, its up rows, their
 * scales, its columns of every down row and, with `scales`, every down row's scale (each part holds them whole). A part
 * that IS a card slot (one part, the same plan) is one run. Answers false when the runs would not fit. */
static __device__ inline bool nn__expert_tier__part_runs(const nn__expert_tier* t, uint64_t p, uint64_t src, bool scales,
                                                         nn__expert__gather* g) {
    const uint64_t P = t->parts, H = t->hidden, I = t->inter, PI = I / P, cols = t->down_row / P;
    if (P == 1u && t->p_up_lut == t->up_lut && t->p_down == t->down && t->p_down_lut == t->down_lut) {
        if (g->runs + 1u > NN__EXPERT__GATHER_RUNS) return false;
        const nn__expert__run whole = { src, 0u, 0u, 0u, t->down_lut + 2u * H, 1u };
        g->run[g->runs++] = whole;
        return true;
    }
    if (g->runs + 5u + (scales ? 1u : 0u) > NN__EXPERT__GATHER_RUNS) return false;
    const nn__expert__run runs[5] = {
        { src,                           0u,   p * PI * t->gate_row,                      0u,          PI * t->gate_row, 1u },
        { src + PI * t->gate_row,        0u,   I * t->gate_row + p * PI * t->gate_row,    0u,          PI * t->gate_row, 1u },
        { src + t->p_up_lut,             0u,   t->up_lut + p * PI * 2u,                   0u,          PI * 2u,          1u },
        { src + t->p_up_lut + PI * 2u,   0u,   t->up_lut + I * 2u + p * PI * 2u,          0u,          PI * 2u,          1u },
        { src + t->p_down,               cols, t->down + p * cols,                        t->down_row, cols,             H  },
    };
    for (unsigned r = 0u; r < 5u; ++r) g->run[g->runs++] = runs[r];
    if (scales) {
        const nn__expert__run sc = { src + t->p_down_lut, 0u, t->down_lut, 0u, H * 2u, 1u };
        g->run[g->runs++] = sc;
    }
    return true;
}

static __device__ inline bool nn__expert_tier__stitch(const nn__expert_tier* t, uint64_t x, nn__expert__gather* g) {
    if (t->sources == 0ull || x >= t->experts) return false;
    g->runs = 0u;
    for (uint64_t p = 0u; p < t->parts; ++p) {
        const sys__heap_node at = sys__node_array__borrow(t->sources, x * t->parts + p);
        if (at.dtype != SYS__KIND__VALUE_INT || at.args[0] == 0ull) return false;
        if (!nn__expert_tier__part_runs(t, p, at.args[0], p == 0u, g)) return false;
    }
    return true;
}

/* ══ ⭐⭐ THE EXCLUSIVE TIER — FREE, OUTGOING, DEMOTION (NN-48) ═════════════════════════════════════════════════════════
 * ⚖ *"we need to have an array of demotion candidates using the pointers, and the move is demotion candidates to recycled
 * landing slots … three arrays, demotion, outgoing, free"* · *"the evacuation channels stay held until they are written to
 * the ram destination slots, so when they are recycled if they are marked as pending the writer does wait"*.
 *   FREE       the card collection's free list: recycled slots, each one whose last expert has left for RAM
 *   OUTGOING   `outgoing` below: a promotion's arriving expert gathered into a FREE slot, and the band's demotion
 *              candidate — its least recently used expert not pinned and not already outgoing — read off the card into an
 *              evacuation channel while the card still computes it
 *   the commit at a layer's visit or a chunk's note, the CPUs idle: the arriving expert, landed, is the card's; the
 *              outgoing one leaves the card (its slot back to FREE) and takes, on every socket, the RAM slot the arriving
 *              one gave up (`nn__expert__hand_over`); the channel is written into them (WRITE jobs) and a CPU about to
 *              compute that expert waits for its write (`nn__expert__wait_written`). The tier's sources follow.
 * ⛳ every expert lives in one place, and is computable everywhere a reader looks for it: an arriving one stays in RAM until
 *   the commit, an outgoing one stays on the card until its bytes are safely in a channel. */
typedef struct nn__expert_tier__out {
    bool used;
    uint64_t owner, lx, x, lv, v;          /* who asked; the arriving expert and its layer; the outgoing one and its */
    unsigned channel;
    nn__loader__read* evacuate;
} nn__expert_tier__out;
static nn__expert_tier__out nn__expert_tier__zzprivate_out[NN__EXPERT_TIER__OUTGOING];

/* Nanoseconds since `a`. */
static inline uint64_t nn__loader__zzprivate_since(const struct timespec* a) {
    struct timespec b;
    clock_gettime(CLOCK_MONOTONIC, &b);
    return (uint64_t)(b.tv_sec - a->tv_sec) * 1000000000ull + (uint64_t)(b.tv_nsec - a->tv_nsec);
}

/* A channel for an evacuation of `bytes`, its pinned memory made the first time. ⛳ A CHANNEL STILL PENDING — written
 *   into RAM by some earlier commit — IS WAITED FOR, as ruled; one HELD (its expert not yet committed) cannot be, because
 *   the commit that frees it runs on this same thread later: with every channel held, the answer is none. Lock held. */
static int nn__loader__zzprivate_channel_take(sys__silicon_family__id family, uint64_t bytes) {
    for (;;) {
        int pending = -1;
        for (unsigned c = 0u; c < NN__EXPERT_TIER__CHANNELS; ++c) {
            nn__loader__channel* ch = &nn__loader__zzprivate_channels[c];
            if (ch->held) continue;
            if (ch->pending != 0u) { pending = (int)c; continue; }
            if (ch->host == 0 || ch->bytes < bytes) {
                void* card = 0;
                if (ch->host != 0) sys__gpu__memory_free_host_ram(family, ch->host);
                ch->host = 0; ch->bytes = 0u;
                if (!sys__gpu__memory_allocate_host_ram_mapped(family, &ch->host, &card, (size_t)bytes)) return -1;
                ch->bytes = bytes;
            }
            ch->held = true;
            return (int)c;
        }
        if (pending < 0) { ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_CHANNEL_FULL]; return -1; }
        /* ⛳ A CHANNEL PENDING ON A HELD WRITE IS LET GO FIRST: the held writes wait for the program to say the CPUs are done,
         *   which cannot come while this visit waits here — the commit just before it may have held them (MEASURED: the
         *   9-layer test hung, 2026-10-05) */
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
            nn__loader__read* r = &nn__loader__zzprivate_reads[i];
            if (r->state == NN__LOADER__WAITING && r->job == NN__LOADER__JOB_WRITE && r->channel == (unsigned)pending) {
                r->state = NN__LOADER__QUEUED;
                r->seq = ++nn__loader__zzprivate_seq;
                pthread_cond_broadcast(&nn__loader__zzprivate_work);
            }
        }
        struct timespec a;
        clock_gettime(CLOCK_MONOTONIC, &a);
        nn__loader__zzprivate_wait();
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_CHANNEL_WAIT_NS] += nn__loader__zzprivate_since(&a);
    }
}

/* Whether expert `e` of band (`l`, `type`) is in a swap not yet committed — outgoing, or ARRIVING: an arriving expert
 * is resident on the card from its settle, but its RAM slots are still its own until the commit hands them on, so a
 * second swap taking it as its victim would evict it with nowhere to write it — and the first swap's commit would
 * then give its RAM slots away, leaving it in neither place (`REASONED`: the whole GLM's decode faulted at token 20
 * once the commit settled arrivals itself, which widened this window). Lock held. */
static bool nn__expert_tier__zzprivate_outgoing(uint64_t owner, uint64_t l, uint64_t e) {
    for (unsigned i = 0u; i < NN__EXPERT_TIER__OUTGOING; ++i) {
        const nn__expert_tier__out* o = &nn__expert_tier__zzprivate_out[i];
        if (o->used && o->owner == owner && ((o->lv == l && o->v == e) || (o->lx == l && o->x == e))) return true;
    }
    return false;
}

/* The demotion candidate: the least recently used expert of `layer`'s band while it holds its share, else of the band
 * holding the most — never one of `pinned` (the asking layer's picks), never one already outgoing. Lock held. */
static bool nn__expert_tier__zzprivate_victim(uint64_t owner, uint64_t layer, uint64_t type, const uint64_t* pinned, unsigned npinned,
                                              uint64_t* vl, uint64_t* v) {
    const uint64_t layers = nn__expert__index_layers();
    uint64_t banded = 0ull, most = layers, held = 0ull;
    for (uint64_t l = 0ull; l < layers; ++l) {
        const uint64_t n = nn__expert__layer_resident(l, type);
        banded += nn__expert__layer_experts(l, type) != 0ull ? 1ull : 0ull;
        if (n > held) { held = n; most = l; }
    }
    const uint64_t share = banded == 0ull ? 0ull : nn__expert__type_slots(type) / banded;
    const uint64_t order[2] = { (share != 0ull && nn__expert__layer_resident(layer, type) >= share) ? layer : most, layer };
    for (unsigned k = 0u; k < 2u; ++k) {
        const uint64_t l = order[k];
        if (l >= layers) continue;
        uint64_t e = 0ull;
        if (!nn__expert__oldest(l, type, &e)) continue;
        for (unsigned step = 0u; step < 64u; ++step) {
            bool skip = nn__expert_tier__zzprivate_outgoing(owner, l, e);
            for (unsigned p = 0u; !skip && l == layer && p < npinned; ++p) skip = pinned[p] == e;
            if (!skip) { *vl = l; *v = e; return true; }
            sys__heap_node me;
            if (!nn__expert__slot(l, type, e, &me) || me.args[NN__EXPERT__SLOT_PREV] == NN__EXPERT__NONE) break;
            e = NN__EXPERT__UNLINK(me.args[NN__EXPERT__SLOT_PREV]);
        }
    }
    return false;
}

/* A free read record, or none. Lock held. */
static nn__loader__read* nn__loader__zzprivate_free_record(void) {
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i)
        if (nn__loader__zzprivate_reads[i].state == NN__LOADER__FREE) return &nn__loader__zzprivate_reads[i];
    return 0;
}

/* Whether a WRITE of expert `e` (band `l`, `type`) is still on its way, for any worker. Lock held. */
static bool nn__loader__zzprivate_writing(uint64_t l, uint64_t type, uint64_t e) {
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        const nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__FREE && r->job == NN__LOADER__JOB_WRITE && r->layer == l && r->type == type && r->expert == e)
            return true;
    }
    return false;
}

/* ⭐ A PROMOTION INTO AN EXCLUSIVE TIER: `x` (in RAM) gathered into a FREE card slot, and a demotion candidate sent
 * OUTGOING beside it. RESIDENT / ARRIVING / REFUSED, as `nn__expert__promote`; REFUSED leaves everything as it was. */
static __device__ inline int nn__expert_tier__promote(sys__silicon_family__id family, const nn__expert_tier* t, uint64_t x,
                                                      const uint64_t* pinned, unsigned npinned) {
    const uint64_t layer = t->layer, type = t->type, owner = nn__expert__zzpackage_owner(), bytes = nn__expert__type_bytes(type);
    sys__heap_node me;
    if (!nn__expert__slot(layer, type, x, &me)) return NN__EXPERT__REFUSED;
    if (me.args[NN__EXPERT__SLOT_AT] != 0ull) return NN__EXPERT__RESIDENT;
    nn__expert__gather g;
    if (bytes == 0ull || !nn__expert_tier__stitch(t, x, &g)) return NN__EXPERT__REFUSED;     /* not in RAM: none to gather */
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    if (nn__loader__zzprivate_find(owner, layer, type, x) != 0) { pthread_mutex_unlock(&nn__loader__zzprivate_lock); return NN__EXPERT__ARRIVING; }
    /* ⛳ AN EXPERT STILL BEING WRITTEN INTO RAM — demoted a moment ago — is not gathered back from bytes not yet there */
    if (nn__loader__zzprivate_writing(layer, t->part_type, x)) { pthread_mutex_unlock(&nn__loader__zzprivate_lock); return NN__EXPERT__REFUSED; }
    /* ⛳ THE CHANNEL FIRST: taking one may wait for a pending one, and a wait lets the lock go — so nothing is reserved
     *   until the last wait is behind */
    const int channel = nn__loader__zzprivate_channel_take(family, bytes);
    if (channel < 0) { pthread_mutex_unlock(&nn__loader__zzprivate_lock); return NN__EXPERT__REFUSED; }
    nn__loader__channel* ch = &nn__loader__zzprivate_channels[channel];
    if (nn__loader__zzprivate_find(owner, layer, type, x) != 0) { ch->held = false; pthread_mutex_unlock(&nn__loader__zzprivate_lock); return NN__EXPERT__ARRIVING; }
    nn__expert_tier__out* o = 0;
    for (unsigned i = 0u; i < NN__EXPERT_TIER__OUTGOING && o == 0; ++i)
        if (!nn__expert_tier__zzprivate_out[i].used) o = &nn__expert_tier__zzprivate_out[i];
    uint64_t vl = 0ull, v = 0ull, slot = 0ull, room = 0ull;
    nn__loader__read *rg = 0, *re = 0;
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX && re == 0; ++i)
        if (nn__loader__zzprivate_reads[i].state == NN__LOADER__FREE) { if (rg == 0) rg = &nn__loader__zzprivate_reads[i]; else re = &nn__loader__zzprivate_reads[i]; }
    /* ⚖ *"if so we can launch the read and write together, if not we need to send to vram first and receive to ram later"*:
     *   half duplex reserves the evacuation's record now and launches it at the commit after the gather lands */
    sys__heap_node vm;
    if (o == 0 || re == 0 || !nn__expert_tier__zzprivate_victim(owner, layer, type, pinned, npinned, &vl, &v)
     || !nn__expert__slot(vl, type, v, &vm) || vm.args[NN__EXPERT__SLOT_AT] == 0ull || !nn__loader__zzprivate_start()
     || !nn__expert__type_take(type, &slot, &room)) {
        ch->held = false;
        pthread_mutex_unlock(&nn__loader__zzprivate_lock);
        return NN__EXPERT__REFUSED;                                   /* no room in flight, no candidate, no FREE slot */
    }
    rg->state = NN__LOADER__QUEUED;
    rg->predicted = false; rg->owner = owner; rg->family = family;
    rg->layer = layer; rg->type = type; rg->expert = x; rg->slot = slot; rg->room = room;
    rg->gather = true; rg->bytes = bytes; rg->runs = g; rg->job = NN__LOADER__JOB_GATHER;
    rg->queued_ns = nn__loader__zzprivate_now_ns();
    rg->seq = ++nn__loader__zzprivate_seq;
    re->state = t->duplex == 1u ? NN__LOADER__QUEUED : NN__LOADER__WAITING;
    re->predicted = false; re->owner = owner; re->family = family;
    re->layer = vl; re->type = type; re->expert = v; re->slot = 0ull; re->room = vm.args[NN__EXPERT__SLOT_AT];
    re->gather = false; re->bytes = bytes; re->job = NN__LOADER__JOB_EVACUATE; re->channel = (unsigned)channel;
    re->seq = ++nn__loader__zzprivate_seq;
    o->used = true; o->owner = owner; o->lx = layer; o->x = x; o->lv = vl; o->v = v; o->channel = (unsigned)channel; o->evacuate = re;
    ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_PROMOTED];
    pthread_cond_broadcast(&nn__loader__zzprivate_work);
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return NN__EXPERT__ARRIVING;
}

/* Every WRITE that worker `by`'s commits held back, let go — the threads take them from here. Lock held. */
static void nn__loader__zzprivate_release_writes(uint64_t by) {
    bool any = false;
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__WAITING || r->job != NN__LOADER__JOB_WRITE || r->by != by) continue;
        r->state = NN__LOADER__QUEUED;
        r->seq = ++nn__loader__zzprivate_seq;
        any = true;
    }
    if (any) pthread_cond_broadcast(&nn__loader__zzprivate_work);
}

/* One cell of the tier's sources — part `p` of expert `e` of layer `l` — set to `at` (0: on the card). */
static bool nn__expert_tier__zzprivate_source(const nn__expert_tier* t, uint64_t l, uint64_t e, uint64_t p, uint64_t at) {
    if (l >= sys__node_array__length(t->layers)) return false;
    const sys__heap_node arr = sys__node_array__borrow(t->layers, l);
    if (!sys__heap_node__carries_reference(arr.dtype) || !sys__node_array__is(arr.args[0])) return false;
    sys__heap_node cell = sys__heap_node__nothing();
    cell.dtype = SYS__KIND__VALUE_INT; cell.args[0] = at;
    return sys__node_array__set(arr.args[0], e * t->parts + p, &cell);
}

/* ⭐ THE COMMIT — every swap of this worker whose arriving expert has landed and whose outgoing one is in its channel:
 * the outgoing one leaves the card and takes the arriving one's RAM slots, the channel is written into them. Called at a
 * layer's visit and a chunk's note, before the CPUs are handed anything, so no CPU is reading a slot this changes. A swap
 * whose gather failed is undone: the outgoing expert stays on the card. False on a step that could not be done. */
static __device__ inline bool nn__expert_tier__commit(sys__silicon_family__id family, const nn__expert_tier* t) {
    const uint64_t owner = nn__expert__zzpackage_owner(), type = t->type, P = t->parts;
    const int acting = sys__silicon__host_acting_as();
    bool ok = true;
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    /* ⛳ writes still held from an earlier commit go now — a program that never says its CPUs are done must not leave
     *   their channels pending for ever */
    nn__loader__zzprivate_release_writes(owner);
    for (unsigned i = 0u; i < NN__EXPERT_TIER__OUTGOING; ++i) {
        nn__expert_tier__out* o = &nn__expert_tier__zzprivate_out[i];
        if (!o->used || o->owner != owner) continue;
        nn__loader__read* rx = nn__loader__zzprivate_find(owner, o->lx, type, o->x);
        if (rx != 0) {
            if (rx->state != NN__LOADER__DONE && rx->state != NN__LOADER__FAILED) continue;      /* still arriving */
            /* ⛳ ITS GATHER SETTLED HERE, WHATEVER LAYER THIS VISIT IS FOR: left to its own layer's visit it waits a token,
             *   holding its channel all that time — MEASURED, 16 channels then carried 15.3 swaps a token, 60 more a
             *   token were refused for want of one, and the card waited 19 ms a token for one to come free. The commit
             *   runs on the card's thread between layers, so no one is reading that layer's index. */
            const bool got = rx->state == NN__LOADER__DONE;
            const uint64_t slot = rx->slot, room = rx->room;
            if (got) nn__loader__zzprivate_landed(rx);
            rx->state = NN__LOADER__FREE;
            if (!got || !nn__expert__admit(o->lx, o->x, room, type)) (void)nn__expert__type_give(type, slot);
        }
        nn__loader__read* re = o->evacuate;
        if (re->state == NN__LOADER__WAITING) {                      /* half duplex: the gather has landed — now the evacuation */
            sys__heap_node lm;
            if (nn__expert__slot(o->lx, type, o->x, &lm) && lm.args[NN__EXPERT__SLOT_AT] != 0ull) {
                re->state = NN__LOADER__QUEUED;
                re->seq = ++nn__loader__zzprivate_seq;
                pthread_cond_broadcast(&nn__loader__zzprivate_work);
                continue;
            }
            re->state = NN__LOADER__FAILED;                          /* the gather failed: undone below */
        }
        if (re->state == NN__LOADER__QUEUED || re->state == NN__LOADER__READING) continue;      /* still going out */
        sys__heap_node xm;
        const bool landed = re->state == NN__LOADER__DONE && nn__expert__slot(o->lx, type, o->x, &xm) && xm.args[NN__EXPERT__SLOT_AT] != 0ull;
        re->state = NN__LOADER__FREE;
        nn__loader__channel* ch = &nn__loader__zzprivate_channels[o->channel];
        o->used = false;
        if (!landed) { ch->held = false; continue; }                                            /* undone: v stays */
        /* the card: the outgoing expert leaves, its slot back to FREE */
        if (!nn__expert__evict(o->lv, type, o->v)) { ch->held = false; ok = false; continue; }
        /* the sockets: the arriving expert's slot handed to the outgoing one, and the channel written into it */
        ch->pending = 0u;
        for (uint64_t p = 0u; p < P; ++p) {
            sys__silicon__host_act_as((int)(t->part_worker + p));
            sys__heap_node pm;
            nn__expert__gather w;
            w.runs = 0u;
            const bool have = nn__expert__slot(o->lx, t->part_type, o->x, &pm) && pm.args[NN__EXPERT__SLOT_AT] != 0ull;
            const uint64_t at = have ? pm.args[NN__EXPERT__SLOT_AT] : 0ull;
            nn__loader__read* rw = nn__loader__zzprivate_free_record();
            /* the write's runs: the gather's for this part, turned round — out of the channel, in the card's layout, into the
             *   part's RAM slot */
            nn__expert__gather gp;
            gp.runs = 0u;
            if (!have || rw == 0 || !nn__expert_tier__part_runs(t, p, at, true, &gp)
             || !nn__expert__hand_over(o->lx, o->x, o->lv, o->v, t->part_type)) { ok = false; continue; }
            for (unsigned r = 0u; r < gp.runs; ++r) {
                const nn__expert__run u = gp.run[r];
                const nn__expert__run back = { (uint64_t)(uintptr_t)ch->host + u.into, u.into_stride, u.from, u.from_stride, u.bytes, u.count };
                w.run[w.runs++] = back;
            }
            const uint64_t writer = nn__expert__zzpackage_owner();
            /* ⚖ *"mark the write to ram after the cpu compute is done (it is the whole point of the half duplex path)"*:
             *   half duplex holds the write until the program says the CPUs are done (▶ `nn__expert_tier__written`), so it
             *   runs while the card computes attention and not against the CPUs' experts for the memory; full duplex
             *   writes at once — which of the two serves best there is the EPYC's to measure */
            rw->state = t->duplex == 1u ? NN__LOADER__QUEUED : NN__LOADER__WAITING;
            rw->by = owner;
            rw->predicted = false; rw->owner = writer; rw->family = family;
            rw->layer = o->lv; rw->type = t->part_type; rw->expert = o->v; rw->slot = 0ull; rw->room = at;
            rw->gather = false; rw->bytes = 0ull; rw->runs = w; rw->job = NN__LOADER__JOB_WRITE; rw->channel = o->channel;
            rw->seq = ++nn__loader__zzprivate_seq;
            __atomic_add_fetch(&nn__loader__zzprivate_writes_out, 1ull, __ATOMIC_RELEASE);
            ++ch->pending;
            ok = nn__expert_tier__zzprivate_source(t, o->lx, o->x, p, 0ull) && ok;
            ok = nn__expert_tier__zzprivate_source(t, o->lv, o->v, p, at) && ok;
        }
        sys__silicon__host_act_as(acting);
        ch->held = false;
        ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_DEMOTED];
        pthread_cond_broadcast(&nn__loader__zzprivate_work);
    }
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return ok;
}

/* ⭐ A CPU ABOUT TO COMPUTE expert `e` of band (`layer`, `type`) waits here for its write, if one is still on its way.
 * ⛳ A WRITE STILL HELD IS LET GO HERE: it is held until the CPUs are done, so a CPU that needs it now would otherwise wait
 *   for a release that comes only after it. */
static __device__ inline void nn__expert__wait_written(uint64_t layer, uint64_t type, uint64_t e) {
    /* ⛳ NO WRITE ON ITS WAY, NO LOCK: a verb asks this once a pick, and the lock is the loader threads' too, busy with a
     *   tier's promotions */
    if (__atomic_load_n(&nn__loader__zzprivate_writes_out, __ATOMIC_ACQUIRE) == 0ull) return;
    const uint64_t owner = nn__expert__zzpackage_owner();
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    for (;;) {
        bool writing = false;
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX && !writing; ++i) {
            nn__loader__read* r = &nn__loader__zzprivate_reads[i];
            writing = r->state != NN__LOADER__FREE && r->job == NN__LOADER__JOB_WRITE && r->owner == owner && r->layer == layer
                   && r->type == type && r->expert == e;
            if (writing && r->state == NN__LOADER__WAITING) {
                r->state = NN__LOADER__QUEUED;
                r->seq = ++nn__loader__zzprivate_seq;
                pthread_cond_broadcast(&nn__loader__zzprivate_work);
            }
        }
        if (!writing) break;
        struct timespec a;
        clock_gettime(CLOCK_MONOTONIC, &a);
        nn__loader__zzprivate_wait();
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_WRITTEN_WAIT_NS] += nn__loader__zzprivate_since(&a);
    }
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
}

/* ⭐ THE CPUS ARE DONE: the writes this worker's commits held back go now, while the card computes what comes next. */
static __device__ inline void nn__expert_tier__written(void) {
    const uint64_t owner = nn__expert__zzpackage_owner();
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    nn__loader__zzprivate_release_writes(owner);
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
}

static __device__ inline bool nn__expert_tier__visit(sys__silicon_family__id family, const nn__expert_tier* t, const uint64_t* ids,
                                                     bool* held) {
    const uint64_t layer = t->layer, type = t->type;
    const unsigned k = (unsigned)t->top_k;
    const bool fed = t->sources != 0ull;
    if (fed) (void)nn__expert__settle_ready(layer, type);
    if (t->exclusive == 1u && !nn__expert_tier__commit(family, t)) return false;   /* the swaps that are ready: ▶ commit */
    const uint64_t now = nn__expert__tick();
    uint64_t pinned[NN__EXPERT__TIER_K_MAX];
    unsigned n_pinned = 0u;
    unsigned hits = 0u;
    for (unsigned j = 0u; j < k; ++j) {
        sys__heap_node me;
        if (!nn__expert__slot(layer, type, ids[j], &me)) return false;
        held[j] = me.args[NN__EXPERT__SLOT_AT] != 0ull;
        if (held[j]) { (void)nn__expert__touch(layer, type, ids[j]); pinned[n_pinned++] = ids[j]; ++hits; }
    }
    {
        const uint64_t owner = nn__expert__zzpackage_owner();
        pthread_mutex_lock(&nn__loader__zzprivate_lock);
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_PICKS] += k;
        nn__loader__zzprivate_count[NN__EXPERT__COUNT_HITS] += hits;
        for (unsigned j = 0u; j < k; ++j)
            if (!held[j] && nn__loader__zzprivate_find(owner, layer, type, ids[j]) != 0)
                ++nn__loader__zzprivate_count[NN__EXPERT__COUNT_PICKED_ARRIVING];
        pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    }
    /* ⚖ 2026-10-06 — HOW MANY TO COPY IS A FUNCTION OF HOW LONG THE CPUS TAKE: above the line of hits one, at or below it
     *   the tier's `landing` — few picks on the card leave the CPUs a long share, and two copies hide behind it.
     * ⚖ AND THE BEST OF THEM, NOT THE FIRST: the qualifying picks in the order of their score (the older of their last two
     *   calls, the most recent first), so a visit that may start one starts the strongest. Each asked before its call
     *   is recorded: the call `qualifies` reads is the one before this. */
    const unsigned land = hits > t->landing_line ? 1u : (unsigned)t->landing;
    unsigned cand[NN__EXPERT__TIER_K_MAX];
    uint64_t score[NN__EXPERT__TIER_K_MAX];
    unsigned n_cand = 0u;
    for (unsigned j = 0u; j < k && fed && land > 0u; ++j) {
        if (held[j] || !nn__expert__qualifies(layer, type, ids[j])) continue;
        uint64_t s = 0ull;
        (void)nn__expert__zzpackage_older(layer, type, ids[j], &s);    /* 0 while the tier fills first-come */
        unsigned at = n_cand++;
        while (at > 0u && score[at - 1u] < s) { cand[at] = cand[at - 1u]; score[at] = score[at - 1u]; --at; }
        cand[at] = j; score[at] = s;
    }
    unsigned started = 0u;
    for (unsigned c = 0u; c < n_cand && started < land; ++c) {
        const unsigned j = cand[c];
        nn__expert__gather g;
        if (t->exclusive == 1u ? nn__expert_tier__promote(family, t, ids[j], pinned, n_pinned) == NN__EXPERT__ARRIVING
                               : (nn__expert_tier__stitch(t, ids[j], &g)
                                  && nn__expert__promote(family, layer, type, ids[j], &g, pinned, n_pinned) == NN__EXPERT__ARRIVING))
            ++started;
    }
    for (unsigned j = 0u; j < k; ++j) (void)nn__expert__called(layer, type, ids[j], now);
    return true;
}

static __device__ inline bool nn__expert_tier__chunk(sys__silicon_family__id family, const nn__expert_tier* t, uint64_t* chosen,
                                                     uint64_t n, uint64_t threshold) {
    const uint64_t layer = t->layer, type = t->type, experts = t->experts;
    const unsigned k = (unsigned)t->top_k;
    const bool fed = t->sources != 0ull;
    if (fed) (void)nn__expert__settle_ready(layer, type);
    if (t->exclusive == 1u && !nn__expert_tier__commit(family, t)) return false;   /* the swaps that are ready: ▶ commit */
    uint32_t* picked = (uint32_t*)calloc(experts, sizeof(uint32_t));
    uint8_t* card = (uint8_t*)calloc(experts, 1u);                 /* 1: the card computes this expert for the chunk */
    uint64_t* list = (uint64_t*)malloc(sizeof(uint64_t) * experts);
    uint64_t* pinned = (uint64_t*)malloc(sizeof(uint64_t) * experts);
    bool ok = picked != 0 && card != 0 && list != 0 && pinned != 0;
    /* every call recorded a row at a time, every resident expert touched */
    for (uint64_t r = 0u; r < n && ok; ++r) {
        const uint64_t now = nn__expert__tick();
        for (unsigned j = 0u; j < k && ok; ++j) {
            const uint64_t id = chosen[r * k + j];
            sys__heap_node me;
            if (id >= experts || !nn__expert__slot(layer, type, id, &me)) { ok = false; break; }
            ++picked[id];
            if (me.args[NN__EXPERT__SLOT_AT] != 0ull) { (void)nn__expert__touch(layer, type, id); card[id] = 1u; }
            (void)nn__expert__called(layer, type, id, now);
        }
    }
    unsigned n_pinned = 0u;
    for (uint64_t e = 0u; e < experts && ok; ++e) if (card[e]) pinned[n_pinned++] = e;
    /* the line: the experts not on the card picked `threshold` times or more, the most picked first, as many as there is room */
    if (ok && fed) {
        uint64_t m = 0u;
        for (uint64_t e = 0u; e < experts; ++e) {
            if (card[e] || picked[e] == 0u || picked[e] < threshold) continue;
            uint64_t q = m++;
            while (q > 0u && picked[list[q - 1u]] < picked[e]) { list[q] = list[q - 1u]; --q; }
            list[q] = e;
        }
        const uint64_t room = nn__expert__flight_free();
        if (m > room) m = room;
        for (uint64_t q = m; q-- > 0u; ) {                         /* queued from the fewest picks to the most */
            nn__expert__gather g;
            if (t->exclusive == 1u ? nn__expert_tier__promote(family, t, list[q], pinned, n_pinned) == NN__EXPERT__ARRIVING
                                   : (nn__expert_tier__stitch(t, list[q], &g)
                                      && nn__expert__promote(family, layer, type, list[q], &g, pinned, n_pinned) == NN__EXPERT__ARRIVING))
                card[list[q]] = 1u;
        }
    }
    for (uint64_t i = 0u; i < n * k && ok; ++i) if (card[chosen[i]]) chosen[i] += experts;
    free(picked); free(card); free(list); free(pinned);
    return ok;
}

#endif /* SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH */
