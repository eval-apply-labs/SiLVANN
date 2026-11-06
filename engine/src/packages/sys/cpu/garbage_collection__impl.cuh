#ifndef SILVANN__PACKAGES_SYS_CPU_GARBAGE_COLLECTION__IMPL_CUH
#define SILVANN__PACKAGES_SYS_CPU_GARBAGE_COLLECTION__IMPL_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "garbage_collection__header.cuh"   /* what it fills in */
#include "heap_object__header.cuh"          /* the worklist's collect */
#include "heap__header.cuh"                 /* the publish the mode was turned on under */

/* The thread's mode: the publish of the heap it was turned on under, or zero for off. A publish counts
 * from one, so zero is never a heap's. */
static __thread uint32_t sys__garbage_collection__zzprivate_manual = 0u;

static __device__ inline bool sys__garbage_collection__zzpackage_manual(void) {
    const uint32_t under = sys__garbage_collection__zzprivate_manual;
    return under != 0u && under == sys__heap__zzengine_generation();
}

static __device__ inline void sys__garbage_collection__collect(void) {
    /* ⛳ OFF WHILE IT RUNS, because collecting is releasing: a release in the mode would only put the next
     * teardown back on the worklist it is draining. */
    const uint32_t was = sys__garbage_collection__zzprivate_manual;
    sys__garbage_collection__zzprivate_manual = 0u;
    sys__heap_object__zzpackage_collect();
    sys__garbage_collection__zzprivate_manual = was;
}

static __device__ inline bool sys__garbage_collection__manual(bool on) {
    const bool was = sys__garbage_collection__zzpackage_manual();
    /* Under no heap there is nothing to wait, and the mode stays off. */
    sys__garbage_collection__zzprivate_manual = on ? sys__heap__zzengine_generation() : 0u;
    if (was && !on) sys__heap_object__zzpackage_collect();
    return was;
}

static __device__ inline void sys__garbage_collection__zzengine_evaluation_ends(void) {
    /* ⭐ NOTHING WAITS PAST AN EVALUATION; the mode does, until the program turns it off. */
    sys__garbage_collection__collect();
}

static __device__ inline void sys__garbage_collection__zzpackage_thread_ends(void) {
    sys__garbage_collection__zzprivate_manual = 0u;
    sys__heap_object__zzpackage_collect();
}

#endif /* SILVANN__PACKAGES_SYS_CPU_GARBAGE_COLLECTION__IMPL_CUH */
