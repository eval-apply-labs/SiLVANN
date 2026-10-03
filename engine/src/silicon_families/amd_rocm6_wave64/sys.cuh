#ifndef SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_SYS_CUH
#define SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_SYS_CUH
/* ══ sys's DOORS — amd_rocm6_wave64 ═══════════════════════════════════════════════════════════════════
 * What `sys` asks of every family (`sys/contracts/abi/gpu.cuh`), answered for this one. Each of them
 * is a plain host function: no `__device__`, no launch, nothing that runs on the card. What runs on
 * the card is `nn.cuh`, beside it. */
/* ⛳ THE FAMILY'S OWN RUNTIME, NAMED HERE RATHER THAN INHERITED from whatever included this. */
#include "../../packages/sys/cpu/silicon/file.cuh"   /* opening and reading a file, written once for every family */
#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include "includes.cuh"   /* which cards are this family's */


static inline bool amd_rocm6_wave64__memory_allocate(void** at, size_t bytes) { return hipMalloc(at, bytes) == hipSuccess; }
static inline void amd_rocm6_wave64__memory_free(void* at)           { if (at) (void)hipFree(at); }
static inline bool amd_rocm6_wave64__memory_zerofill(void* at, size_t bytes) { return hipMemset(at, 0, bytes) == hipSuccess; }

static inline bool amd_rocm6_wave64__memory_read(void* to, const void* from, size_t bytes) {
    return hipMemcpy(to, from, bytes, hipMemcpyDeviceToHost) == hipSuccess;
}

static inline bool amd_rocm6_wave64__memory_write(void* to, const void* from, size_t bytes) {
    return hipMemcpy(to, from, bytes, hipMemcpyHostToDevice) == hipSuccess;
}

/* ⛳ THE CALLER'S WORK, NOT THE DEVICE'S: every kernel nn launches goes on stream 0, and "every operation I asked for has
 * finished" is that stream drained. A side channel's copy is waited for by the thread that issued it (`side_memory_*`), so
 * waiting for the whole device here waited for other threads' copies too — `MEASURED` 2026-10-03 on GLM's card tier
 * (NN-48): each MoE layer's router read stalled behind the promotions on the loader's channels, ~2.5 ms a promotion. */
static inline bool amd_rocm6_wave64__compute_completed(void) { return hipStreamSynchronize(0) == hipSuccess; }

static inline bool amd_rocm6_wave64__side_open(void** channel) {
    return hipStreamCreateWithFlags((hipStream_t*)channel, hipStreamNonBlocking) == hipSuccess;
}

static inline void amd_rocm6_wave64__side_close(void* channel) {
    if (channel) (void)hipStreamDestroy((hipStream_t)channel);
}

static inline bool amd_rocm6_wave64__side_memory_read(void* to, const void* from, size_t bytes, void* channel) {
    return hipMemcpyAsync(to, from, bytes, hipMemcpyDeviceToHost, (hipStream_t)channel) == hipSuccess
        && hipStreamSynchronize((hipStream_t)channel) == hipSuccess;
}

static inline bool amd_rocm6_wave64__side_memory_write(void* to, const void* from, size_t bytes, void* channel) {
    return hipMemcpyAsync(to, from, bytes, hipMemcpyHostToDevice, (hipStream_t)channel) == hipSuccess
        && hipStreamSynchronize((hipStream_t)channel) == hipSuccess;
}

static inline void amd_rocm6_wave64__memory_free_host_ram(void* at) { if (at) (void)hipHostFree(at); }

/* RAM the program already holds, made writable by the bound card: pinned and mapped where it lies. */
static inline bool amd_rocm6_wave64__memory_register_host(void* at, size_t bytes, void** card_at) {
    if (hipHostRegister(at, bytes, hipHostRegisterMapped) != hipSuccess) return false;
    if (hipHostGetDevicePointer(card_at, at, 0) == hipSuccess) return true;
    (void)hipHostUnregister(at);
    return false;
}
static inline void amd_rocm6_wave64__memory_unregister_host(void* at) { if (at) (void)hipHostUnregister(at); }

static inline bool amd_rocm6_wave64__memory_allocate_host_ram_mapped(void** at, void** card_at, size_t bytes) {
    if (hipHostMalloc(at, bytes, hipHostMallocMapped) != hipSuccess) return false;
    if (hipHostGetDevicePointer(card_at, *at, 0) == hipSuccess) return true;
    (void)hipHostFree(*at); *at = 0;
    return false;
}

/* ── ⭐⭐ WHICH DEVICES — ONLY THE CARDS THIS FAMILY INCLUDES ──────────────────────────────────────────
 * ⚖ *"every worker process detects and takes over its own gpu"* · *"a family's device_count and
 * bind_device count only the cards it includes"*. HIP numbers every AMD card the driver shows it; this family numbers only the ones it
 * includes, in that order, so device `i` here is the i-th card this family takes. A card of another
 * family in the same box is not a device of this one at all, and cannot be bound through it.
 * ⛳ A machine with no device of this family — or no driver at all — answers zero rather than failing,
 * because a family loaded where it has nothing to drive is an ordinary machine, and zero is how its
 * workers learn there is nothing to take over. */

/* Whether the card the vendor's runtime numbers `ordinal` is this family's, asked by `gcnArchName`, `gfx906:sramecc+:xnack-`. */
static inline bool amd_rocm6_wave64__zzprivate_takes(int ordinal) {
    hipDeviceProp_t p;
    return hipGetDeviceProperties(&p, ordinal) == hipSuccess && amd_rocm6_wave64__includes(p.gcnArchName);
}

/* How many cards the vendor's runtime shows, whichever family they are. */
static inline int amd_rocm6_wave64__zzprivate_shown(void) {
    int n = 0;
    return (hipGetDeviceCount(&n) == hipSuccess && n > 0) ? n : 0;
}

static inline uint32_t amd_rocm6_wave64__device_count(void) {
    uint32_t mine = 0u;
    for (int ordinal = 0; ordinal < amd_rocm6_wave64__zzprivate_shown(); ++ordinal)
        if (amd_rocm6_wave64__zzprivate_takes(ordinal)) ++mine;
    return mine;
}

/* ⛳ PER THREAD, as the vendor's runtime makes it: the device a thread binds is the one ITS allocations,
 * copies and launches go to, so each worker binds its own and none can move another's. */
static inline uint32_t amd_rocm6_wave64__bind_device(uint32_t index) {
    uint32_t mine = 0u;
    for (int ordinal = 0; ordinal < amd_rocm6_wave64__zzprivate_shown(); ++ordinal) {
        if (!amd_rocm6_wave64__zzprivate_takes(ordinal)) continue;
        if (mine++ == index) return (hipSetDevice(ordinal) == hipSuccess) ? 1u : 0u;
    }
    return 0u;
}

/* The device of this family the calling thread is on, or `0xFFFFFFFF` when the vendor's runtime cannot
 * say or the thread is on a card this family does not take. */
static inline uint32_t amd_rocm6_wave64__bound_device(void) {
    int d = -1;
    if (!(hipGetDevice(&d) == hipSuccess) || d < 0 || !amd_rocm6_wave64__zzprivate_takes(d)) return 0xFFFFFFFFu;
    uint32_t mine = 0u;
    for (int ordinal = 0; ordinal < d; ++ordinal)
        if (amd_rocm6_wave64__zzprivate_takes(ordinal)) ++mine;
    return mine;
}

/* ⛳ THE FUNCTIONS ABOVE ARE THIS FAMILY'S OWN, prefixed with its name; `sys`'s door list names them
 * (`contracts/abi/gpu.cuh`), and the published `sys__gpu__*` names in `cpu/silicon/silicon_family.cuh`
 * reach them through the door table when a caller names this family. So the host side of the engine names
 * no family at all — not by discipline, but because the only thing it can reach is a pointer. */

/* ── A FILE, INTO THIS FAMILY'S MEMORY — ⚖ *"sys door"*. ▶ `sys/cpu/silicon/file.cuh`. A card: through a bounce buffer and this family's own copy. */
static inline bool amd_rocm6_wave64__file_open(const char* path, uint64_t* handle) { return sys__file__zzabi_open(path, handle); }
static inline void amd_rocm6_wave64__file_close(uint64_t handle) { sys__file__zzabi_close(handle); }
/* A file mapped as memory: a card's memory is not the machine's, so never here. */
static inline bool amd_rocm6_wave64__memory_map_file(void* at, size_t bytes, uint64_t handle, uint64_t offset) {
    (void)at; (void)bytes; (void)handle; (void)offset;
    return false;
}
/* A range of this family's memory is always in it: nothing is paged in, so nothing is waited for. */
static inline bool amd_rocm6_wave64__memory_prefetch(const void* at, size_t bytes) {
    (void)at; (void)bytes;
    return true;
}
static inline bool amd_rocm6_wave64__file_read(uint64_t handle, uint64_t offset, uint64_t bytes, void* to) {
    return sys__file__zzabi_read_through(handle, offset, bytes, to, amd_rocm6_wave64__memory_write);
}

#endif /* SILVANN__SILICON_FAMILIES_AMD_ROCM6_WAVE64_SYS_CUH */
