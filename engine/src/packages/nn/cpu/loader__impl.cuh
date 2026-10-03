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
} nn__loader__read;

static nn__loader__read nn__loader__zzprivate_reads[NN__EXPERT__FLIGHT_MAX];
static pthread_mutex_t  nn__loader__zzprivate_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   nn__loader__zzprivate_work = PTHREAD_COND_INITIALIZER;   /* a read was queued */
static pthread_cond_t   nn__loader__zzprivate_done = PTHREAD_COND_INITIALIZER;   /* a read finished   */
static pthread_t        nn__loader__zzprivate_threads[NN__EXPERT__LOADERS_MAX];
static unsigned         nn__loader__zzprivate_started = 0u;
static bool             nn__loader__zzprivate_stopping = false;
static uint64_t         nn__loader__zzprivate_count[NN__EXPERT__COUNTS];
static uint64_t         nn__loader__zzprivate_seq = 0ull;             /* the queue's clock, under the lock */

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
        if (r.gather) ok = nn__loader__zzprivate_gather(&r);
        for (unsigned i = 0u; ok && !r.gather && i < NN__EXPERT__SLICES; ++i) {
            if (r.backing.bytes[i] == 0ull) continue;
            ok = sys__gpu__file_read(r.family, r.backing.file, r.backing.from[i] + r.expert * r.backing.bytes[i], r.backing.bytes[i],
                                     (void*)(uintptr_t)(r.room + r.backing.into[i]));
        }
        pthread_mutex_lock(&nn__loader__zzprivate_lock);
        job->state = ok ? NN__LOADER__DONE : NN__LOADER__FAILED;
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
        if (r->state != NN__LOADER__FREE && r->owner == owner && r->layer == layer && r->type == type && r->expert == expert) return r;
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
    free_read->seq = ++nn__loader__zzprivate_seq;
    ++nn__loader__zzprivate_count[predicted ? NN__EXPERT__COUNT_PREDICTED : NN__EXPERT__COUNT_MISSES];
    pthread_cond_signal(&nn__loader__zzprivate_work);
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return NN__EXPERT__ARRIVING;
}

/* ⭐ A WAIT FOR A READ, TIMED — the lock held, as `pthread_cond_wait` needs it: the time is what a layer spent waiting on
 *   the disk, the thing a cache's misses cost beyond the compute that overlaps them (`nn__expert__count`). */
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
            if (r->state == NN__LOADER__FREE || r->owner != owner || r->layer != layer || r->type != type) continue;
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
        if (r->state != NN__LOADER__FREE && r->owner == owner && r->layer == layer && r->type == type && r->expert == expert) {
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
    free_read->seq = ++nn__loader__zzprivate_seq;
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
        if (r->owner != owner || r->layer != layer || r->type != type) continue;
        if (r->state != NN__LOADER__DONE && r->state != NN__LOADER__FAILED) continue;      /* free, or still on its way */
        const bool landed = r->state == NN__LOADER__DONE;
        const uint64_t slot = r->slot, room = r->room, expert = r->expert;
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
static __device__ inline bool nn__expert__tier_visit(sys__silicon_family__id family, uint64_t layer, uint64_t type, const uint64_t* ids,
                                                     unsigned k, bool* held, unsigned landing, nn__expert__stitch stitch,
                                                     const void* model) {
    if (k > NN__EXPERT__TIER_K_MAX) return false;
    if (stitch != 0) (void)nn__expert__settle_ready(layer, type);
    const uint64_t now = nn__expert__tick();
    uint64_t pinned[NN__EXPERT__TIER_K_MAX];
    unsigned n_pinned = 0u;
    for (unsigned j = 0u; j < k; ++j) {
        sys__heap_node me;
        if (!nn__expert__slot(layer, type, ids[j], &me)) return false;
        held[j] = me.args[NN__EXPERT__SLOT_AT] != 0ull;
        if (held[j]) { (void)nn__expert__touch(layer, type, ids[j]); pinned[n_pinned++] = ids[j]; }
    }
    /* each asked before its call is recorded: the call `qualifies` reads is the one before this */
    unsigned started = 0u;
    for (unsigned j = 0u; j < k && stitch != 0; ++j) {
        nn__expert__gather g;
        if (!held[j] && started < landing && nn__expert__qualifies(layer, type, ids[j]) && stitch(model, ids[j], &g)
         && nn__expert__promote(family, layer, type, ids[j], &g, pinned, n_pinned) == NN__EXPERT__ARRIVING)
            ++started;
    }
    for (unsigned j = 0u; j < k; ++j) (void)nn__expert__called(layer, type, ids[j], now);
    return true;
}

static __device__ inline bool nn__expert__tier_chunk(sys__silicon_family__id family, uint64_t layer, uint64_t type, uint64_t experts,
                                                     uint64_t* chosen, uint64_t n, unsigned k, uint64_t threshold,
                                                     nn__expert__stitch stitch, const void* model) {
    if (stitch != 0) (void)nn__expert__settle_ready(layer, type);
    uint32_t* picked = (uint32_t*)calloc(experts, sizeof(uint32_t));
    uint8_t* card = (uint8_t*)calloc(experts, 1u);                 /* 1: the card computes this expert for the chunk */
    uint64_t* list = (uint64_t*)malloc(sizeof(uint64_t) * experts);
    uint64_t* pinned = (uint64_t*)malloc(sizeof(uint64_t) * experts);
    bool ok = picked != 0 && card != 0 && list != 0 && pinned != 0;
    /* every call recorded a row at a time, every resident expert touched */
    for (uint64_t t = 0u; t < n && ok; ++t) {
        const uint64_t now = nn__expert__tick();
        for (unsigned j = 0u; j < k && ok; ++j) {
            const uint64_t id = chosen[t * k + j];
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
    if (ok && stitch != 0) {
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
            if (stitch(model, list[q], &g)
             && nn__expert__promote(family, layer, type, list[q], &g, pinned, n_pinned) == NN__EXPERT__ARRIVING)
                card[list[q]] = 1u;
        }
    }
    for (uint64_t i = 0u; i < n * k && ok; ++i) if (card[chosen[i]]) chosen[i] += experts;
    free(picked); free(card); free(list); free(pinned);
    return ok;
}

#endif /* SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH */
