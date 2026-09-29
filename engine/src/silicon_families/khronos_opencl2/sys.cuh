#ifndef SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_SYS_CUH
#define SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_SYS_CUH
/* ══ sys's DOORS — khronos_opencl2 ════════════════════════════════════════════════════════════════════
 *
 * sys's doors through OpenCL 2.0. Where HIP hands out device pointers, this family uses SHARED VIRTUAL
 * MEMORY: `clSVMAlloc` answers a pointer that means the same address on the host and on the card, so
 * everything `sys` does with a card address — offsets, the heap's arithmetic, a pointer in a kernel's
 * arguments — is unchanged. A coarse-grained allocation is card memory the host must not dereference,
 * which is what `sys` already promises; mapped host RAM is a fine-grained one, which both sides may touch.
 *
 * ⛔⛔ ONE CONTEXT PER CARD, AND AN ALLOCATION BELONGS TO THE CARD IT WAS MADE ON — as HIP's does. A context
 *   over several cards makes a coarse-grained allocation reachable from all of them, and the runtime does
 *   that by putting it in HOST RAM: `MEASURED` on the MI50, 129 MB read by one kernel at
 *   ~800 GB/s from a one-card context and at 13 GB/s — PCIe — from a two-card one. The family's first
 *   shape was the second, and every weight was read across the bus.
 * ⛳ And one in-order queue per card, so work on a card runs in the order it was asked for, as a HIP
 *   stream does. A thread's bound card is its own (`__thread`), as the vendor runtimes make it.
 * ⛳ OPENCL HAS NO "CURRENT DEVICE", so the family keeps the thread's and answers device 0 until a thread
 *   binds another — which is what HIP does.
 * ⭐ EVERY LIVE ALLOCATION IS KEPT IN A LIST, WITH ITS CARD, because a kernel may be handed any address
 *   inside any of them, and a coarse-grained allocation is only guaranteed present on the card for a
 *   kernel told it may use it. A launch hands the runtime the bound card's (`nn.cuh`); a copy, a fill and
 *   a free go to the queue of the card that owns the address. */

/* What this file needs, named where a reader — and an editor — can follow it. */
#define CL_TARGET_OPENCL_VERSION 200
#include "../../packages/sys/cpu/silicon/file.cuh"   /* opening and reading a file, written once for every family */
#include <CL/cl.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "includes.cuh"   /* which cards are this family's */

#define KHRONOS_OPENCL2__DEVICES_MAX  16u

typedef struct khronos_opencl2__allocation {
    void*    at;
    size_t   bytes;
    uint32_t card;                                              /* whose context it was made in */
} khronos_opencl2__allocation;

typedef struct khronos_opencl2__state {
    cl_uint          count;                                     /* cards of this family, in platform order */
    cl_device_id     device[KHRONOS_OPENCL2__DEVICES_MAX];
    cl_context       context[KHRONOS_OPENCL2__DEVICES_MAX];     /* one each: ▶ the note above */
    cl_command_queue queue[KHRONOS_OPENCL2__DEVICES_MAX];       /* one in-order queue each */
    pthread_mutex_t  lock;                                      /* the live list, and a kernel's arguments */
    khronos_opencl2__allocation* live;                          /* every allocation not yet freed */
    size_t           live_count, live_room;
} khronos_opencl2__state;

static khronos_opencl2__state khronos_opencl2__zzprivate_state = { 0u, {0}, {0}, {0}, PTHREAD_MUTEX_INITIALIZER, 0, 0u, 0u };
static pthread_once_t         khronos_opencl2__zzprivate_once  = PTHREAD_ONCE_INIT;
static __thread uint32_t      khronos_opencl2__zzprivate_bound = 0u;

/* Whether a device can do what this family asks of it: both kinds of shared virtual memory. */
static inline bool khronos_opencl2__zzprivate_svm(cl_device_id d) {
    cl_device_svm_capabilities svm = 0;
    if (clGetDeviceInfo(d, CL_DEVICE_SVM_CAPABILITIES, sizeof svm, &svm, 0) != CL_SUCCESS) return false;
    return (svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) && (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER);
}

/* Find this family's cards on the first platform that has any, and stand up a context and a queue for
 * each. Once per process; a machine with none leaves the count at zero, and a card whose context or queue
 * will not stand up takes the family's cards with it — a partial set would renumber the rest. */
static void khronos_opencl2__zzprivate_stand_up(void) {
    khronos_opencl2__state* s = &khronos_opencl2__zzprivate_state;
    cl_platform_id platforms[8]; cl_uint np = 0u;
    if (clGetPlatformIDs(8u, platforms, &np) != CL_SUCCESS) return;
    for (cl_uint p = 0u; p < np && p < 8u && s->count == 0u; ++p) {
        cl_device_id all[KHRONOS_OPENCL2__DEVICES_MAX]; cl_uint nd = 0u;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, KHRONOS_OPENCL2__DEVICES_MAX, all, &nd) != CL_SUCCESS) continue;
        for (cl_uint d = 0u; d < nd && d < KHRONOS_OPENCL2__DEVICES_MAX; ++d) {
            char name[256] = {0};
            if (clGetDeviceInfo(all[d], CL_DEVICE_NAME, sizeof name - 1u, name, 0) != CL_SUCCESS) continue;
            if (khronos_opencl2__includes(name) && khronos_opencl2__zzprivate_svm(all[d])) s->device[s->count++] = all[d];
        }
    }
    if (s->count == 0u) return;
    cl_int err = CL_SUCCESS;
    for (cl_uint d = 0u; d < s->count; ++d) {
        s->context[d] = clCreateContext(0, 1u, &s->device[d], 0, 0, &err);
        if (err != CL_SUCCESS) { s->count = 0u; return; }
        s->queue[d] = clCreateCommandQueueWithProperties(s->context[d], s->device[d], 0, &err);
        if (err != CL_SUCCESS) { s->count = 0u; return; }
    }
}

static inline khronos_opencl2__state* khronos_opencl2__zzpackage_up(void) {
    (void)pthread_once(&khronos_opencl2__zzprivate_once, khronos_opencl2__zzprivate_stand_up);
    return khronos_opencl2__zzprivate_state.count > 0u ? &khronos_opencl2__zzprivate_state : 0;
}

/* The queue of the card the calling thread is on. */
static inline cl_command_queue khronos_opencl2__zzpackage_queue(void) {
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    return (s && khronos_opencl2__zzprivate_bound < s->count) ? s->queue[khronos_opencl2__zzprivate_bound] : 0;
}

/* ── THE LIVE LIST ───────────────────────────────────────────────────────────────────────────────────── */
static inline bool khronos_opencl2__zzprivate_keep(khronos_opencl2__state* s, void* at, size_t bytes, uint32_t card) {
    bool kept = true;
    pthread_mutex_lock(&s->lock);
    if (s->live_count == s->live_room) {
        size_t room = s->live_room ? 2u * s->live_room : 64u;
        khronos_opencl2__allocation* grown =
            (khronos_opencl2__allocation*)realloc(s->live, room * sizeof(khronos_opencl2__allocation));
        if (grown) { s->live = grown; s->live_room = room; } else kept = false;
    }
    if (kept) {
        s->live[s->live_count].at = at; s->live[s->live_count].bytes = bytes; s->live[s->live_count].card = card;
        ++s->live_count;
    }
    pthread_mutex_unlock(&s->lock);
    return kept;
}

/* The card that owns `at`: the allocation it falls inside. False for an address this family did not
 * allocate — a host buffer, say. */
static inline bool khronos_opencl2__zzpackage_owner(khronos_opencl2__state* s, const void* at, uint32_t* card) {
    bool found = false;
    pthread_mutex_lock(&s->lock);
    for (size_t i = 0u; i < s->live_count && !found; ++i)
        if ((const char*)at >= (const char*)s->live[i].at && (const char*)at < (const char*)s->live[i].at + s->live[i].bytes) {
            *card = s->live[i].card; found = true;
        }
    pthread_mutex_unlock(&s->lock);
    return found;
}

static inline bool khronos_opencl2__zzprivate_forget(khronos_opencl2__state* s, void* at, uint32_t* card) {
    bool found = false;
    pthread_mutex_lock(&s->lock);
    for (size_t i = 0u; i < s->live_count; ++i)
        if (s->live[i].at == at) { *card = s->live[i].card; s->live[i] = s->live[--s->live_count]; found = true; break; }
    pthread_mutex_unlock(&s->lock);
    return found;
}

/* An allocation of either grain, on the bound card, kept on the list with it. */
static inline void* khronos_opencl2__zzprivate_allocate(size_t bytes, cl_svm_mem_flags grain) {
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    const uint32_t card = khronos_opencl2__zzprivate_bound;
    if (s == 0 || card >= s->count) return 0;
    void* at = clSVMAlloc(s->context[card], CL_MEM_READ_WRITE | grain, bytes ? bytes : 1u, 0u);
    if (at && !khronos_opencl2__zzprivate_keep(s, at, bytes ? bytes : 1u, card)) { clSVMFree(s->context[card], at); at = 0; }
    return at;
}

/* ⛳ ITS CARD'S QUEUE IS DRAINED FIRST: a kernel still queued may be using what is about to go, and
 *   `clSVMFree` does not wait for it. No other card's kernel can be using it — it is not in their context. */
static inline void khronos_opencl2__zzprivate_release(void* at) {
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    uint32_t card = 0u;
    if (s == 0 || at == 0 || !khronos_opencl2__zzprivate_forget(s, at, &card)) return;
    (void)clFinish(s->queue[card]);
    clSVMFree(s->context[card], at);
}

/* The queue a copy or a fill with these two ends goes to: the card that owns either end, the card side
 * first; the bound card's when neither is this family's. */
static inline cl_command_queue khronos_opencl2__zzprivate_queue_for(const void* a, const void* b) {
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    if (s == 0) return 0;
    uint32_t card = 0u;
    if (khronos_opencl2__zzpackage_owner(s, a, &card) || khronos_opencl2__zzpackage_owner(s, b, &card))
        return s->queue[card];
    return khronos_opencl2__zzpackage_queue();
}

/* ── THE DOORS ───────────────────────────────────────────────────────────────────────────────────────── */
static inline bool khronos_opencl2__memory_allocate(void** at, size_t bytes) {
    *at = khronos_opencl2__zzprivate_allocate(bytes, 0);
    return *at != 0;
}

static inline void khronos_opencl2__memory_free(void* at) { khronos_opencl2__zzprivate_release(at); }

static inline bool khronos_opencl2__memory_zerofill(void* at, size_t bytes) {
    cl_command_queue q = khronos_opencl2__zzprivate_queue_for(at, 0);
    const cl_uchar zero = 0u;
    return q && clEnqueueSVMMemFill(q, at, &zero, 1u, bytes, 0u, 0, 0) == CL_SUCCESS && clFinish(q) == CL_SUCCESS;
}

/* ⛳ BLOCKING, AND IN THE QUEUE OF THE CARD THAT OWNS THE CARD END: the copy waits for the work asked for
 *   before it on that card, as `hipMemcpy` does. */
static inline bool khronos_opencl2__memory_read(void* to, const void* from, size_t bytes) {
    cl_command_queue q = khronos_opencl2__zzprivate_queue_for(from, to);
    return q && clEnqueueSVMMemcpy(q, CL_TRUE, to, from, bytes, 0u, 0, 0) == CL_SUCCESS;
}

static inline bool khronos_opencl2__memory_write(void* to, const void* from, size_t bytes) {
    cl_command_queue q = khronos_opencl2__zzprivate_queue_for(to, from);
    return q && clEnqueueSVMMemcpy(q, CL_TRUE, to, from, bytes, 0u, 0, 0) == CL_SUCCESS;
}

static inline bool khronos_opencl2__compute_completed(void) {
    cl_command_queue q = khronos_opencl2__zzpackage_queue();
    return q && clFinish(q) == CL_SUCCESS;
}

/* The side channel: a second queue on the same card, so a copy through it does not wait behind the work
 * in the card's own queue. */
static inline bool khronos_opencl2__side_open(void** channel) {
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    if (s == 0 || khronos_opencl2__zzprivate_bound >= s->count) return false;
    cl_int err = CL_SUCCESS;
    cl_command_queue q = clCreateCommandQueueWithProperties(s->context[khronos_opencl2__zzprivate_bound],
                                                            s->device[khronos_opencl2__zzprivate_bound], 0, &err);
    if (err != CL_SUCCESS) return false;
    *channel = (void*)q;
    return true;
}

static inline void khronos_opencl2__side_close(void* channel) {
    if (channel) (void)clReleaseCommandQueue((cl_command_queue)channel);
}

static inline bool khronos_opencl2__side_memory_read(void* to, const void* from, size_t bytes, void* channel) {
    return clEnqueueSVMMemcpy((cl_command_queue)channel, CL_TRUE, to, from, bytes, 0u, 0, 0) == CL_SUCCESS;
}

static inline bool khronos_opencl2__side_memory_write(void* to, const void* from, size_t bytes, void* channel) {
    return clEnqueueSVMMemcpy((cl_command_queue)channel, CL_TRUE, to, from, bytes, 0u, 0, 0) == CL_SUCCESS;
}

/* Mapped host RAM: one fine-grained allocation, whose address is the same for the host and the card. */
static inline bool khronos_opencl2__memory_allocate_host_ram_mapped(void** at, void** card_at, size_t bytes) {
    *at = khronos_opencl2__zzprivate_allocate(bytes, CL_MEM_SVM_FINE_GRAIN_BUFFER);
    *card_at = *at;
    return *at != 0;
}

static inline void khronos_opencl2__memory_free_host_ram(void* at) { khronos_opencl2__zzprivate_release(at); }

/* ⛔ OpenCL 2.0 cannot hand a kernel a pointer into RAM it did not allocate — that is fine-grained SYSTEM
 *   SVM, which the MI50's runtime does not offer — so this family answers no, and a caller copies. */
static inline bool khronos_opencl2__memory_register_host(void* at, size_t bytes, void** card_at) {
    (void)at; (void)bytes; *card_at = 0;
    return false;
}
static inline void khronos_opencl2__memory_unregister_host(void* at) { (void)at; }

static inline uint32_t khronos_opencl2__device_count(void) {
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    return s ? (uint32_t)s->count : 0u;
}

static inline uint32_t khronos_opencl2__bind_device(uint32_t index) {
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    if (s == 0 || index >= s->count) return 0u;
    khronos_opencl2__zzprivate_bound = index;
    return 1u;
}

static inline uint32_t khronos_opencl2__bound_device(void) {
    khronos_opencl2__state* s = khronos_opencl2__zzpackage_up();
    return (s && khronos_opencl2__zzprivate_bound < s->count) ? khronos_opencl2__zzprivate_bound : 0xFFFFFFFFu;
}

/* ── A FILE, INTO THIS FAMILY'S MEMORY — ⚖ *"sys door"*. ▶ `sys/cpu/silicon/file.cuh`. A card: through a bounce buffer and this family's own copy. */
static inline bool khronos_opencl2__file_open(const char* path, uint64_t* handle) { return sys__file__zzabi_open(path, handle); }
static inline void khronos_opencl2__file_close(uint64_t handle) { sys__file__zzabi_close(handle); }
/* A file mapped as memory: a card's memory is not the machine's, so never here. */
static inline bool khronos_opencl2__memory_map_file(void* at, size_t bytes, uint64_t handle, uint64_t offset) {
    (void)at; (void)bytes; (void)handle; (void)offset;
    return false;
}
static inline bool khronos_opencl2__file_read(uint64_t handle, uint64_t offset, uint64_t bytes, void* to) {
    return sys__file__zzabi_read_through(handle, offset, bytes, to, khronos_opencl2__memory_write);
}

#endif /* SILVANN__SILICON_FAMILIES_KHRONOS_OPENCL2_SYS_CUH */
