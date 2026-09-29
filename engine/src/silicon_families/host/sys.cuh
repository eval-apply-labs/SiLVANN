#ifndef SILVANN__SILICON_FAMILIES_HOST_SYS_CUH
#define SILVANN__SILICON_FAMILIES_HOST_SYS_CUH
/* ══ ⭐⭐ THE HOST SEAM — CPU. THE CARD THAT IS NOT A CARD ═══════════════════════════════════════════
 *
 * ⚖ *"sys and the evaluator go host"*, and ⚖ *"include all architectures for the cpu
 * end and load to the gpu their specific version … decide at runtime which silicon we are operating on."*
 *
 * The CONTRACT is in `silicon__header.cuh`; the shape is `hip/host.cuh`'s, verb for verb. What is
 * different here is the thing the contract was written around and never had to say out loud:
 *
 * ⛔⛔ **THERE IS NO SECOND ADDRESS SPACE, SO EVERY TRANSFER IS A COPY AND EVERY SYNC IS NOTHING.**
 * `memory_allocate` is `malloc`, `memory_read` and `memory_write` are `memcpy` in the two directions, and
 * `compute_completed` returns
 * true having done nothing at all. ⇒ ★ THE SEAM'S TWELVE VERBS ARE NOT "WHAT A GPU DOES" — they are
 * WHAT A CALLER NEEDS SAID, and on a CPU most of them are already true. That the contract degenerates
 * this gracefully is evidence it was drawn in the right place.
 *
 * ⛔ **AND `compute_completed` RETURNING TRUE IS A CLAIM, NOT A STUB.** Its contract is *"every operation I asked
 * for has finished"*, and on a CPU that is satisfied the moment the call that asked returns — a `memcpy`
 * has already happened, a `malloc` has already happened. A caller that treats a `true` from here as
 * weaker than a `true` from HIP would be wrong; it is STRONGER, because there was never a queue.
 * ⚠ **WHAT WOULD FALSIFY THAT: the moment this build starts launching real kernels through the doors in
 * `nn/gpu/doors.cuh`.** Those ARE queued, and then `compute_completed` owes a `hipDeviceSynchronize` on
 * whichever card the compute went to — which is a DIFFERENT card object from this one. ▶ the open
 * question below.
 *
 * ⛳ THE HOST-RAM PAIR IS ALSO A PLAIN ALLOCATION. "Pinned" means *"a DMA engine may read this without
 * the OS moving it"*, and with no DMA engine in the picture there is nothing to pin against. The pair
 * is kept rather than refused because a caller asking to pin is asking for memory it may hand to a
 * transfer, and it gets exactly that.
 *
 * ── ⚖ TWO SILICONS IN ONE PROGRAM, AND THE CALLER SAYS WHICH ────────────────────────────────────────
 * The engine has two at once: this one, for the lisp heap and the evaluator's own memory, and
 * a real card for the compute the doors launch. ⚖ *"a wrapper that has the arch as an argument"* — every
 * host operation names its family, so an allocation for the language names this one and an allocation for
 * the compute names the card's, and neither can land on the other. ▶ `cpu/silicon/silicon_family.cuh`.
 */
#include "../../packages/sys/cpu/silicon/file.cuh"   /* opening and reading a file, written once for every family */
#include <stdlib.h>
#include <string.h>
#include "../../packages/sys/cpu/silicon/silicon_family.cuh"   /* the table it offers itself into, and sys's doors */
#include "../roster.cuh"                                       /* its number */

static inline bool host__memory_allocate(void** at, size_t bytes) {
    if (at == 0) return false;
    *at = malloc(bytes);
    return *at != 0;
}
static inline void host__memory_free(void* at) { if (at) free(at); }

static inline bool host__memory_zerofill(void* at, size_t bytes) {
    if (at == 0) return false;
    memset(at, 0, bytes);
    return true;
}

/* ⛳ BOTH DIRECTIONS ARE THE SAME CALL, and naming them separately is not redundancy — it is the
 * contract's vocabulary, and a caller reading `memory_read` at a use site learns which way the data went
 * without looking anything up. `memmove` rather than `memcpy` because with one address space a caller
 * CAN hand us overlapping regions, which on a real card is impossible by construction. */
static inline bool host__memory_read(void* to, const void* from, size_t bytes) {
    if (to == 0 || from == 0) return false;
    memmove(to, from, bytes);
    return true;
}
static inline bool host__memory_write(void* to, const void* from, size_t bytes) {
    if (to == 0 || from == 0) return false;
    memmove(to, from, bytes);
    return true;
}

static inline bool host__compute_completed(void) { return true; }

/* ⛳ A SIDE CHANNEL WITH NOTHING TO BE BESIDE. A stream exists so a transfer can be ordered against
 * other work without blocking the caller; here the transfer has already happened by the time the call
 * returns, so an ordering primitive has nothing left to order. The handle is a non-null token rather
 * than a real object so that a caller checking `channel != 0` — which the contract lets it do —
 * still sees an open channel. */
static inline bool host__side_open(void** channel) {
    if (channel == 0) return false;
    *channel = (void*)&host__compute_completed;   /* a stable, non-null, never-dereferenced token */
    return true;
}
static inline void host__side_close(void* channel) { (void)channel; }

/* ⛳ THE SIDE CHANNEL IS WHERE THE HOST TALKS TO RUNNERS THAT ARE STILL GOING — the stop word, and a
 * base's fill followed by its status — so its writes are FENCED and its reads are fenced after. A caller
 * relies on send order: the fill is sent before the status that says it is there, and a runner that sees
 * the status must see the fill. A release fence before each write keeps everything written earlier ahead
 * of it; an acquire fence after each read keeps what the caller reads next behind it. `REASONED`: on
 * x86 both are compiler barriers only and nothing here changes; on a weakly ordered CPU they are what
 * makes the order the callers assume true. ⚠ The copies themselves are still plain, not atomic stores. */
static inline bool host__side_memory_read(void* to, const void* from, size_t bytes, void* channel) {
    (void)channel;
    const bool ok = host__memory_read(to, from, bytes);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    return ok;
}
static inline bool host__side_memory_write(void* to, const void* from, size_t bytes, void* channel) {
    (void)channel;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    return host__memory_write(to, from, bytes);
}

static inline void host__memory_free_host_ram(void* at) { host__memory_free(at); }
/* On a CPU the program and the "card" are one address space, so both views are the same bytes. */
/* On a CPU the memory is already the "card's", at the same address. */
static inline bool host__memory_register_host(void* at, size_t bytes, void** card_at) {
    (void)bytes; *card_at = at;
    return at != 0;
}
static inline void host__memory_unregister_host(void* at) { (void)at; }
static inline bool host__memory_allocate_host_ram_mapped(void** at, void** card_at, size_t bytes) {
    if (!host__memory_allocate(at, bytes)) return false;
    *card_at = *at;
    return true;
}

/* ── A FILE, INTO THIS FAMILY'S MEMORY — ⚖ *"sys door"*. ▶ `sys/cpu/silicon/file.cuh`. The memory is the program's, so the bytes go straight in. */
static inline bool host__file_open(const char* path, uint64_t* handle) { return sys__file__zzabi_open(path, handle); }
static inline void host__file_close(uint64_t handle) { sys__file__zzabi_close(handle); }
/* A file mapped as memory: a card's memory is not the machine's, so never here. */
static inline bool host__memory_map_file(void* at, size_t bytes, uint64_t handle, uint64_t offset) {
    (void)at; (void)bytes; (void)handle; (void)offset;
    return false;
}
static inline bool host__file_read(uint64_t handle, uint64_t offset, uint64_t bytes, void* to) {
    return sys__file__zzabi_read(handle, offset, bytes, to);
}

/* ── THE HOST AS A FAMILY ────────────────────────────────────────────────────────────────────────────
 * ⛳ Same shape as `hip`'s and `cuda`'s, compiled into the program rather than loaded: one device, the
 * process itself, which every thread is already on — and only `sys`'s doors, because nothing is launched here. */
static inline uint32_t host__device_count(void) { return 1u; }
static inline uint32_t host__bind_device(uint32_t index) { return index == 0u ? 1u : 0u; }
static inline uint32_t host__bound_device(void) { return 0u; }

/* sys's doors, filled from sys's own list the way a loaded family fills them. */
#define SYS__SILICON_FAMILY__THIS host
#define HOST__ZZPRIVATE_DOOR(PKG, RET, NAME, FN, PARAMS)  FN,
static const sys__doors host__sys_doors = {
    (uint32_t)sizeof(sys__doors), sys__CONTRACT__DOORS(HOST__ZZPRIVATE_DOOR, sys) };
#undef HOST__ZZPRIVATE_DOOR
#undef SYS__SILICON_FAMILY__THIS

/* ⛳ ONE TABLE, `sys`'s, in slot 0, which is `sys`'s by the package roster: nothing is launched on the host,
 * so no other package has doors here, and a verb that needs a card refuses on it as on any family without
 * that package's doors. */
static const void* const host__package_doors[1] = { (const void*)&host__sys_doors };
static const sys__silicon_family host__silicon_family = {
    SYS__SILICON_FAMILY__ABI_VERSION, (uint32_t)sizeof(sys__silicon_family), host__family_id,
    1u, "host", host__package_doors,
};
static const bool host__family_offered = sys__silicon_family__offer(&host__silicon_family);

#endif /* SILVANN__SILICON_FAMILIES_HOST_SYS_CUH */
