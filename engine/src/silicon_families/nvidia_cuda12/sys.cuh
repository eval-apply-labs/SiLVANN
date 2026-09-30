#ifndef SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_SYS_CUH
#define SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_SYS_CUH
/* ══ sys's DOORS — nvidia_cuda12 ══════════════════════════════════════════════════════════════════════
 * What `sys` asks of every family (`sys/contracts/abi/gpu.cuh`), answered for this one. Each of them
 * is a plain host function: no `__device__`, no launch, nothing that runs on the card. What runs on
 * the card is `nn.cuh`, beside it. */
/* ⛳ THE FAMILY'S OWN RUNTIME, NAMED HERE RATHER THAN INHERITED from whatever included this. */
#include "../../packages/sys/cpu/silicon/file.cuh"   /* opening and reading a file, written once for every family */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "includes.cuh"   /* which cards are this family's */


static inline bool nvidia_cuda12__memory_allocate(void** at, size_t bytes) { return cudaMalloc(at, bytes) == cudaSuccess; }
static inline void nvidia_cuda12__memory_free(void* at)           { if (at) (void)cudaFree(at); }
static inline bool nvidia_cuda12__memory_zerofill(void* at, size_t bytes) { return cudaMemset(at, 0, bytes) == cudaSuccess; }

static inline bool nvidia_cuda12__memory_read(void* to, const void* from, size_t bytes) {
    return cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

static inline bool nvidia_cuda12__memory_write(void* to, const void* from, size_t bytes) {
    return cudaMemcpy(to, from, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

static inline bool nvidia_cuda12__compute_completed(void) { return cudaDeviceSynchronize() == cudaSuccess; }

static inline bool nvidia_cuda12__side_open(void** channel) {
    return cudaStreamCreateWithFlags((cudaStream_t*)channel, cudaStreamNonBlocking) == cudaSuccess;
}

static inline void nvidia_cuda12__side_close(void* channel) {
    if (channel) (void)cudaStreamDestroy((cudaStream_t)channel);
}

static inline bool nvidia_cuda12__side_memory_read(void* to, const void* from, size_t bytes, void* channel) {
    return cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)channel) == cudaSuccess
        && cudaStreamSynchronize((cudaStream_t)channel) == cudaSuccess;
}

static inline bool nvidia_cuda12__side_memory_write(void* to, const void* from, size_t bytes, void* channel) {
    return cudaMemcpyAsync(to, from, bytes, cudaMemcpyHostToDevice, (cudaStream_t)channel) == cudaSuccess
        && cudaStreamSynchronize((cudaStream_t)channel) == cudaSuccess;
}

static inline void nvidia_cuda12__memory_free_host_ram(void* at) { if (at) (void)cudaFreeHost(at); }

/* RAM the program already holds, made writable by the bound card: pinned and mapped where it lies. */
static inline bool nvidia_cuda12__memory_register_host(void* at, size_t bytes, void** card_at) {
    if (cudaHostRegister(at, bytes, cudaHostRegisterMapped) != cudaSuccess) return false;
    if (cudaHostGetDevicePointer(card_at, at, 0) == cudaSuccess) return true;
    (void)cudaHostUnregister(at);
    return false;
}
static inline void nvidia_cuda12__memory_unregister_host(void* at) { if (at) (void)cudaHostUnregister(at); }

static inline bool nvidia_cuda12__memory_allocate_host_ram_mapped(void** at, void** card_at, size_t bytes) {
    if (cudaHostAlloc(at, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    if (cudaHostGetDevicePointer(card_at, *at, 0) == cudaSuccess) return true;
    (void)cudaFreeHost(*at); *at = 0;
    return false;
}

/* ── ⭐⭐ WHICH DEVICES — ONLY THE CARDS THIS FAMILY INCLUDES ──────────────────────────────────────────
 * ⚖ *"every worker process detects and takes over its own gpu"* · *"a family's device_count and
 * bind_device count only the cards it includes"*. CUDA numbers every NVIDIA card the driver shows it; this family numbers only the ones it
 * includes, in that order, so device `i` here is the i-th card this family takes. A card of another
 * family in the same box is not a device of this one at all, and cannot be bound through it.
 * ⛳ A machine with no device of this family — or no driver at all — answers zero rather than failing,
 * because a family loaded where it has nothing to drive is an ordinary machine, and zero is how its
 * workers learn there is nothing to take over. */

/* Whether the card the vendor's runtime numbers `ordinal` is this family's, asked by its compute capability, as `sm_86`. */
static inline bool nvidia_cuda12__zzprivate_takes(int ordinal) {
    int major = 0, minor = 0;
    char card[16];
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, ordinal) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, ordinal) != cudaSuccess) return false;
    snprintf(card, sizeof card, "sm_%d%d", major, minor);
    return nvidia_cuda12__includes(card);
}

/* How many cards the vendor's runtime shows, whichever family they are. */
static inline int nvidia_cuda12__zzprivate_shown(void) {
    int n = 0;
    return (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? n : 0;
}

static inline uint32_t nvidia_cuda12__device_count(void) {
    uint32_t mine = 0u;
    for (int ordinal = 0; ordinal < nvidia_cuda12__zzprivate_shown(); ++ordinal)
        if (nvidia_cuda12__zzprivate_takes(ordinal)) ++mine;
    return mine;
}

/* ⛳ PER THREAD, as the vendor's runtime makes it: the device a thread binds is the one ITS allocations,
 * copies and launches go to, so each worker binds its own and none can move another's. */
static inline uint32_t nvidia_cuda12__bind_device(uint32_t index) {
    uint32_t mine = 0u;
    for (int ordinal = 0; ordinal < nvidia_cuda12__zzprivate_shown(); ++ordinal) {
        if (!nvidia_cuda12__zzprivate_takes(ordinal)) continue;
        if (mine++ == index) return (cudaSetDevice(ordinal) == cudaSuccess) ? 1u : 0u;
    }
    return 0u;
}

/* The device of this family the calling thread is on, or `0xFFFFFFFF` when the vendor's runtime cannot
 * say or the thread is on a card this family does not take. */
static inline uint32_t nvidia_cuda12__bound_device(void) {
    int d = -1;
    if (!(cudaGetDevice(&d) == cudaSuccess) || d < 0 || !nvidia_cuda12__zzprivate_takes(d)) return 0xFFFFFFFFu;
    uint32_t mine = 0u;
    for (int ordinal = 0; ordinal < d; ++ordinal)
        if (nvidia_cuda12__zzprivate_takes(ordinal)) ++mine;
    return mine;
}

/* ⛳ THE FUNCTIONS ABOVE ARE THIS FAMILY'S OWN, prefixed with its name; `sys`'s door list names them
 * (`contracts/abi/gpu.cuh`), and the published `sys__gpu__*` names in `cpu/silicon/silicon_family.cuh`
 * reach them through the door table when a caller names this family. So the host side of the engine names
 * no family at all — not by discipline, but because the only thing it can reach is a pointer. */

/* ── A FILE, INTO THIS FAMILY'S MEMORY — ⚖ *"sys door"*. ▶ `sys/cpu/silicon/file.cuh`. A card: through a bounce buffer and this family's own copy. */
static inline bool nvidia_cuda12__file_open(const char* path, uint64_t* handle) { return sys__file__zzabi_open(path, handle); }
static inline void nvidia_cuda12__file_close(uint64_t handle) { sys__file__zzabi_close(handle); }
/* A file mapped as memory: a card's memory is not the machine's, so never here. */
static inline bool nvidia_cuda12__memory_map_file(void* at, size_t bytes, uint64_t handle, uint64_t offset) {
    (void)at; (void)bytes; (void)handle; (void)offset;
    return false;
}
/* A range of this family's memory is always in it: nothing is paged in, so nothing is waited for. */
static inline bool nvidia_cuda12__memory_prefetch(const void* at, size_t bytes) {
    (void)at; (void)bytes;
    return true;
}
static inline bool nvidia_cuda12__file_read(uint64_t handle, uint64_t offset, uint64_t bytes, void* to) {
    return sys__file__zzabi_read_through(handle, offset, bytes, to, nvidia_cuda12__memory_write);
}

#endif /* SILVANN__SILICON_FAMILIES_NVIDIA_CUDA12_SYS_CUH */
