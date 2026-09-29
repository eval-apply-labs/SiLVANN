#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_SYS_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_SYS_CUH
/* ══ sys's DOORS ON THE CPU — ITS OWN MEMORY, AT THE ADDRESS THE PROGRAM HAS ═════════════════════════════
 * The same answers as the host family's (`host/sys.cuh`), because the memory is the same: the program and
 * this "card" share one address space, so a copy is a copy, a registration answers the address itself, and
 * a launch has finished by the time it returns (`pool.cuh`). What differs is the device: there is one, and
 * it is only there when the processors have what this family's bodies were compiled for. */
#include "../../../packages/sys/cpu/silicon/file.cuh"   /* opening and reading a file, written once for every family */
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "../../../packages/sys/cpu/silicon/silicon_family.cuh"   /* the table it offers itself into */
#include "../../roster.cuh"                                       /* its number */
#include "pool.cuh"                                               /* the threads a launch runs on */

/* ⛳ SIXTY-FOUR-BYTE ALIGNED, so a row the bodies load eight or thirty-two bytes at a time starts on a line.
 * ⭐ AND ON THE DEVICE'S NODE: on a socket device the memory is mapped fresh and bound to that node before
 *   anything touches it, so its pages land there whichever thread fills them — the expert loader's readers are
 *   not pinned. On the whole machine it is the process's memory, as it always was, unless it is large.
 * A header just before the address says which of the two it was, and how much was mapped. */
typedef struct x86_avx2__held {
    uint64_t mapped;                           /* the bytes mapped for it, or 0: it came from the process's heap */
    uint64_t bytes;                            /* the bytes asked for */
} x86_avx2__held;
#define X86_AVX2__ZZPRIVATE_PAGE 4096u
/* ⭐ A LARGE ALLOCATION IS MAPPED ON THE WHOLE MACHINE TOO, AND WITHOUT A RESERVATION: an expert arena can be larger
 *   than the machine's memory when a file is mapped over it (`memory_map_file`), and the system refuses such a
 *   mapping outright unless it is told nothing is reserved. Its pages are page-aligned, which the file needs. */
#define X86_AVX2__ZZPRIVATE_LARGE (1ull << 30)
static inline bool x86_avx2__memory_allocate(void** at, size_t bytes) {
    if (at == 0) return false;
    *at = 0;
    if (bytes == 0u) bytes = 64u;
    const int node = x86_avx2__zzpackage_node_of(x86_avx2__zzpackage_device);
    if (node < 0 && bytes < X86_AVX2__ZZPRIVATE_LARGE) {
        void* base = 0;
        if (posix_memalign(&base, 64u, bytes + 64u) != 0 || base == 0) return false;
        ((x86_avx2__held*)base)->mapped = 0u; ((x86_avx2__held*)base)->bytes = bytes;
        *at = (uint8_t*)base + 64u;
        return true;
    }
    const size_t mapped = (bytes + 2u * X86_AVX2__ZZPRIVATE_PAGE - 1u) / X86_AVX2__ZZPRIVATE_PAGE * X86_AVX2__ZZPRIVATE_PAGE;
    void* base = mmap(0, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) return false;
    x86_avx2__held* h = (x86_avx2__held*)((uint8_t*)base + X86_AVX2__ZZPRIVATE_PAGE - sizeof *h);
    if (node < 0) {
        h->mapped = mapped; h->bytes = bytes;
        *at = (uint8_t*)base + X86_AVX2__ZZPRIVATE_PAGE;
        return true;
    }
    unsigned long mask[X86_AVX2__ZZPRIVATE_NODES / (8 * sizeof(unsigned long))] = {0};
    mask[node / (8 * sizeof(unsigned long))] = 1ul << (node % (8 * sizeof(unsigned long)));
    /* MPOL_BIND: the node, and no other — a page that cannot be had there is a failed allocation, not a far one */
    if (syscall(SYS_mbind, base, mapped, 2 /* MPOL_BIND */, mask, (unsigned long)X86_AVX2__ZZPRIVATE_NODES + 1ul, 0u) != 0) {
        munmap(base, mapped);
        return false;
    }
    h->mapped = mapped; h->bytes = bytes;
    *at = (uint8_t*)base + X86_AVX2__ZZPRIVATE_PAGE;
    return true;
}
static inline void x86_avx2__memory_free(void* at) {
    if (at == 0) return;
    const x86_avx2__held* h = (const x86_avx2__held*)((uint8_t*)at - sizeof(x86_avx2__held));
    if (h->mapped != 0u) munmap((uint8_t*)at - X86_AVX2__ZZPRIVATE_PAGE, h->mapped);
    else free((uint8_t*)at - 64u);
}
static inline bool x86_avx2__memory_zerofill(void* at, size_t bytes) {
    if (at == 0) return false;
    memset(at, 0, bytes);
    return true;
}
static inline bool x86_avx2__memory_read(void* to, const void* from, size_t bytes) {
    if (to == 0 || from == 0) return false;
    memmove(to, from, bytes);
    return true;
}
static inline bool x86_avx2__memory_write(void* to, const void* from, size_t bytes) {
    if (to == 0 || from == 0) return false;
    memmove(to, from, bytes);
    return true;
}
/* Every launch has returned by the time anything can ask. */
static inline bool x86_avx2__compute_completed(void) { return true; }
static inline bool x86_avx2__side_open(void** channel) {
    if (channel == 0) return false;
    *channel = (void*)&x86_avx2__compute_completed;   /* a stable, non-null, never-dereferenced token */
    return true;
}
static inline void x86_avx2__side_close(void* channel) { (void)channel; }
static inline bool x86_avx2__side_memory_read(void* to, const void* from, size_t bytes, void* channel) {
    (void)channel;
    const bool ok = x86_avx2__memory_read(to, from, bytes);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    return ok;
}
static inline bool x86_avx2__side_memory_write(void* to, const void* from, size_t bytes, void* channel) {
    (void)channel;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    return x86_avx2__memory_write(to, from, bytes);
}
static inline bool x86_avx2__memory_allocate_host_ram_mapped(void** at, void** card_at, size_t bytes) {
    if (!x86_avx2__memory_allocate(at, bytes)) return false;
    *card_at = *at;
    return true;
}
static inline void x86_avx2__memory_free_host_ram(void* at) { x86_avx2__memory_free(at); }
static inline bool x86_avx2__memory_register_host(void* at, size_t bytes, void** card_at) {
    (void)bytes; *card_at = at;
    return at != 0;
}
static inline void x86_avx2__memory_unregister_host(void* at) { (void)at; }

/* ── THE DEVICE: one, when the processors can run what this family was built for ─────────────────── */
static inline bool x86_avx2__zzprivate_capable(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}
/* ⭐ Device 0 is the whole machine; with more than one NUMA node, device 1 + s is node s alone (`pool.cuh`). */
static inline uint32_t x86_avx2__device_count(void) { return x86_avx2__zzprivate_capable() ? x86_avx2__zzpackage_devices() : 0u; }
/* Binding a device stands its pool up, so its size is decided by the worker that will use it. */
static inline uint32_t x86_avx2__bind_device(uint32_t index) {
    if (!x86_avx2__zzprivate_capable() || index >= x86_avx2__zzpackage_devices()) return 0u;
    x86_avx2__zzpackage_bind(index);
    return 1u;
}
static inline uint32_t x86_avx2__bound_device(void) { return x86_avx2__zzprivate_capable() ? x86_avx2__zzpackage_device : 0xFFFFFFFFu; }

/* ── A FILE, INTO THIS FAMILY'S MEMORY — ⚖ *"sys door"*. ▶ `sys/cpu/silicon/file.cuh`. The memory is the program's, so the bytes go straight in. */
static inline bool x86_avx2__file_open(const char* path, uint64_t* handle) { return sys__file__zzabi_open(path, handle); }
static inline void x86_avx2__file_close(uint64_t handle) { sys__file__zzabi_close(handle); }
/* A file mapped as memory, over memory the program already holds — the pages there are replaced by the file's, read
 * from the disk when first touched and kept as the system keeps a file's pages. `sys__file__zzabi_open`'s handle is
 * the descriptor plus one. The mapping outlives the handle, and `memory_free` unmaps it with the rest. */
static inline bool x86_avx2__memory_map_file(void* at, size_t bytes, uint64_t handle, uint64_t offset) {
    if (at == 0 || bytes == 0u || handle == 0ull || (uintptr_t)at % X86_AVX2__ZZPRIVATE_PAGE != 0u
        || offset % X86_AVX2__ZZPRIVATE_PAGE != 0ull)
        return false;
    void* got = mmap(at, bytes, PROT_READ, MAP_SHARED | MAP_FIXED, (int)(handle - 1ull), (off_t)offset);
    return got == at;
}
static inline bool x86_avx2__file_read(uint64_t handle, uint64_t offset, uint64_t bytes, void* to) {
    return sys__file__zzabi_read(handle, offset, bytes, to);
}

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_SYS_CUH */
