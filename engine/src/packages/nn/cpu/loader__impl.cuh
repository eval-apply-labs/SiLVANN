#ifndef SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH
#define SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <pthread.h>
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
    sys__silicon_family__id family;
    uint64_t layer, type, expert, slot, room;
    nn__expert__backing backing;
} nn__loader__read;

static nn__loader__read nn__loader__zzprivate_reads[NN__EXPERT__FLIGHT_MAX];
static pthread_mutex_t  nn__loader__zzprivate_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   nn__loader__zzprivate_work = PTHREAD_COND_INITIALIZER;   /* a read was queued */
static pthread_cond_t   nn__loader__zzprivate_done = PTHREAD_COND_INITIALIZER;   /* a read finished   */
static pthread_t        nn__loader__zzprivate_threads[NN__EXPERT__LOADERS_MAX];
static unsigned         nn__loader__zzprivate_started = 0u;
static bool             nn__loader__zzprivate_stopping = false;
static uint64_t         nn__loader__zzprivate_count[NN__EXPERT__COUNTS];

static void* nn__loader__zzprivate_run(void* unused) {
    (void)unused;
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    for (;;) {
        nn__loader__read* job = 0;
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX && job == 0; ++i)
            if (nn__loader__zzprivate_reads[i].state == NN__LOADER__QUEUED) job = &nn__loader__zzprivate_reads[i];
        if (job == 0) {
            if (nn__loader__zzprivate_stopping) break;
            pthread_cond_wait(&nn__loader__zzprivate_work, &nn__loader__zzprivate_lock);
            continue;
        }
        job->state = NN__LOADER__READING;
        const nn__loader__read r = *job;
        pthread_mutex_unlock(&nn__loader__zzprivate_lock);
        bool ok = true;
        for (unsigned i = 0u; ok && i < NN__EXPERT__SLICES; ++i) {
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

/* The read of `expert` in flight, or none. Called with the lock held. */
static nn__loader__read* nn__loader__zzprivate_find(uint64_t layer, uint64_t type, uint64_t expert) {
    for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
        nn__loader__read* r = &nn__loader__zzprivate_reads[i];
        if (r->state != NN__LOADER__FREE && r->layer == layer && r->type == type && r->expert == expert) return r;
    }
    return 0;
}

/* A slot for a read: a free one of the collection, or the least recently used expert's — this layer's oldest
 * unless that is one of `pinned`, else the oldest of the first other layer that has one. */
static bool nn__loader__zzprivate_room(uint64_t layer, uint64_t type, const uint64_t* pinned, unsigned npinned,
                                       uint64_t* slot, uint64_t* room) {
    if (nn__expert__type_take(type, slot, room)) return true;
    const uint64_t layers = nn__expert__index_layers();
    for (uint64_t k = 0ull; k < layers; ++k) {
        const uint64_t l = (layer + k) % layers;
        uint64_t victim = 0ull;
        if (nn__expert__layer_experts(l, type) == 0ull || !nn__expert__oldest(l, type, &victim)) continue;
        bool held = false;
        for (unsigned p = 0u; l == layer && p < npinned; ++p) held = held || pinned[p] == victim;
        if (held) continue;
        if (!nn__expert__evict(l, type, victim)) return false;
        return nn__expert__type_take(type, slot, room);
    }
    return false;
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
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    nn__loader__read* r = nn__loader__zzprivate_find(layer, type, expert);
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
    free_read->family = family;
    free_read->layer = layer; free_read->type = type; free_read->expert = expert;
    free_read->slot = slot; free_read->room = room;
    free_read->backing = *backing;
    ++nn__loader__zzprivate_count[predicted ? NN__EXPERT__COUNT_PREDICTED : NN__EXPERT__COUNT_MISSES];
    pthread_cond_signal(&nn__loader__zzprivate_work);
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return NN__EXPERT__ARRIVING;
}

static __device__ inline bool nn__expert__settle(uint64_t layer, uint64_t type) {
    bool ok = true;
    pthread_mutex_lock(&nn__loader__zzprivate_lock);
    for (;;) {
        bool waiting = false;
        for (unsigned i = 0u; i < NN__EXPERT__FLIGHT_MAX; ++i) {
            nn__loader__read* r = &nn__loader__zzprivate_reads[i];
            if (r->state == NN__LOADER__FREE || r->layer != layer || r->type != type) continue;
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
        pthread_cond_wait(&nn__loader__zzprivate_done, &nn__loader__zzprivate_lock);
    }
    pthread_mutex_unlock(&nn__loader__zzprivate_lock);
    return ok;
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
    for (unsigned i = 0u; i < NN__EXPERT__BACKING__LENGTH; ++i) {
        const sys__heap_node n = sys__node_array__borrow(array, i);
        if (n.dtype != SYS__KIND__VALUE_INT) return false;
        v[i] = n.args[0];
    }
    backing->file = v[NN__EXPERT__BACKING__FILE];
    for (unsigned i = 0u; i < NN__EXPERT__SLICES; ++i) {
        backing->from[i] = v[NN__EXPERT__BACKING__FROM(i)];
        backing->bytes[i] = v[NN__EXPERT__BACKING__BYTES(i)];
        backing->into[i] = v[NN__EXPERT__BACKING__INTO(i)];
    }
    return true;
}

#endif /* SILVANN__PACKAGES_NN_CPU_LOADER__IMPL_CUH */
