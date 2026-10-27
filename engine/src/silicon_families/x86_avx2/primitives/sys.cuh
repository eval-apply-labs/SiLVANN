#ifndef SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_SYS_CUH
#define SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_SYS_CUH
/* ══ sys's DOORS ON THE CPU — ITS OWN MEMORY, AT THE ADDRESS THE PROGRAM HAS ═════════════════════════════
 * The same answers as the host family's (`host/sys.cuh`), because the memory is the same: the program and
 * this "card" share one address space, so a copy is a copy, a registration answers the address itself, and
 * a launch has finished by the time it returns (`pool.cuh`). What differs is the device: there is one, and
 * it is only there when the processors have what this family's bodies were compiled for. */
#include "../../../packages/sys/cpu/silicon/file.cuh"   /* opening and reading a file, written once for every family */
#include "../../../operating_systems/os.cuh"   /* threads, locks, conditions and pages: the system's */
#include <stdlib.h>
#include <string.h>
#include "../../../packages/sys/cpu/silicon/silicon_family.cuh"   /* the table it offers itself into */
#include "../../roster.cuh"                                       /* its number */
#include "pool.cuh"                                               /* the threads a launch runs on */

/* ⛳ SIXTY-FOUR-BYTE ALIGNED, so a row the bodies load eight or thirty-two bytes at a time starts on a line.
 * ⭐ AND ON THE DEVICE'S NODE: on a socket device the memory is mapped fresh and bound to that node before
 *   anything touches it, so its pages land there whichever thread fills them — the expert loader's readers are
 *   not pinned. On the whole machine it is the process's memory, as it always was, unless it is large.
 * A header just before the address says which of the two it was, and how much was mapped. */
typedef struct x86_avx2__held {
    uint64_t offset;                           /* a guarded allocation's: its mapping starts this far before the address */
    uint64_t mapped;                           /* the bytes mapped for it, or 0: it came from the process's heap */
    uint64_t bytes;                            /* the bytes asked for */
} x86_avx2__held;
#define X86_AVX2__ZZPRIVATE_PAGE 4096u
/* ⭐ A LARGE ALLOCATION IS MAPPED ON THE WHOLE MACHINE TOO, AND WITHOUT A RESERVATION: an expert arena can be larger
 *   than the machine's memory when a file is mapped over it (`memory_map_file`), and the system refuses such a
 *   mapping outright unless it is told nothing is reserved. Its pages are page-aligned, which the file needs. */
#define X86_AVX2__ZZPRIVATE_LARGE (1ull << 30)
#define X86_AVX2__ZZPRIVATE_GUARDED (1ull << 63)   /* in `mapped`: the allocation sits between two guard pages */
static inline bool x86_avx2__memory_allocate(void** at, size_t bytes) {
    if (at == 0) return false;
    *at = 0;
    if (bytes == 0u) bytes = 64u;
    const int node = x86_avx2__zzpackage_node_of(x86_avx2__zzpackage_device);
    if (node < 0 && bytes < X86_AVX2__ZZPRIVATE_LARGE) {
        void* base = sys__os__memory_aligned(64u, bytes + 64u);
        if (base == 0) return false;
        /* ⛔ the header just before the address, where `x86_avx2__memory_free` reads it — not at `base`. Written at `base`
         *   (until 2026-10-07), free read 16 bytes nobody wrote: zero by luck freed it right, anything else munmapped a
         *   garbage length of the process's heap at `at - 4096` — a later large allocation then mapped into the hole,
         *   overlapping live heap, and glibc aborted on a buffer nobody had overrun (`free(): invalid pointer`). */
        x86_avx2__held* h = (x86_avx2__held*)((uint8_t*)base + 64u - sizeof *h);
        h->mapped = 0u; h->bytes = bytes;
        *at = (uint8_t*)base + 64u;
        return true;
    }
    const size_t mapped = (bytes + 2u * X86_AVX2__ZZPRIVATE_PAGE - 1u) / X86_AVX2__ZZPRIVATE_PAGE * X86_AVX2__ZZPRIVATE_PAGE;
    void* base = sys__os__memory_map(mapped);
    if (base == 0) return false;
    /* ⭐ THE WHOLE MACHINE'S LARGE MEMORY INTERLEAVED OVER ITS NODES, a page a node in turn: its threads are on every
     *   socket and each reads the rows its block was given, so a matrix spread evenly draws on every node's memory
     *   controllers at once, where pages left where they were first touched drew unevenly (14 GiB and 22 on this box).
     *   `MEASURED`, the 35B on the CPU alone, node03's two sockets, five pairs: 90.3 ms a token (median) -> 80.9, every
     *   pair faster. A socket device is bound to its own node instead (below). */
    if (node < 0 && x86_avx2__zzpackage_devices() > 2u) {
        int interleave = 1;
        unsigned long all[X86_AVX2__ZZPRIVATE_NODES / (8 * sizeof(unsigned long))] = {0};
        for (uint32_t k = 1u; k < x86_avx2__zzpackage_devices(); ++k) {
            const int n = x86_avx2__zzpackage_node_of(k);
            all[n / (8 * sizeof(unsigned long))] |= 1ul << (n % (8 * sizeof(unsigned long)));
        }
        if (interleave)
            (void)sys__os__memory_interleave(base, mapped, all, (unsigned long)X86_AVX2__ZZPRIVATE_NODES + 1ul);
    }
    x86_avx2__held* h = (x86_avx2__held*)((uint8_t*)base + X86_AVX2__ZZPRIVATE_PAGE - sizeof *h);
    if (node < 0) {
        h->mapped = mapped; h->bytes = bytes;
        *at = (uint8_t*)base + X86_AVX2__ZZPRIVATE_PAGE;
        return true;
    }
    unsigned long mask[X86_AVX2__ZZPRIVATE_NODES / (8 * sizeof(unsigned long))] = {0};
    mask[node / (8 * sizeof(unsigned long))] = 1ul << (node % (8 * sizeof(unsigned long)));
    /* bound: the node, and no other — a page that cannot be had there is a failed allocation, not a far one */
    if (!sys__os__memory_bind(base, mapped, mask, (unsigned long)X86_AVX2__ZZPRIVATE_NODES + 1ul)) {
        sys__os__memory_unmap(base, mapped);
        return false;
    }
    h->mapped = mapped; h->bytes = bytes;
    *at = (uint8_t*)base + X86_AVX2__ZZPRIVATE_PAGE;
    return true;
}
static inline void x86_avx2__memory_free(void* at) {
    if (at == 0) return;
    const x86_avx2__held* h = (const x86_avx2__held*)((uint8_t*)at - sizeof(x86_avx2__held));
    if (h->mapped & X86_AVX2__ZZPRIVATE_GUARDED) sys__os__memory_unmap((uint8_t*)at - h->offset, h->mapped & ~X86_AVX2__ZZPRIVATE_GUARDED);
    else if (h->mapped != 0u) sys__os__memory_unmap((uint8_t*)at - X86_AVX2__ZZPRIVATE_PAGE, h->mapped);
    else sys__os__memory_aligned_free((uint8_t*)at - 64u);
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
/* ⭐ WHERE A FILE IS MAPPED — the ranges `memory_map_file` made, so `memory_prefetch` asks the system of those alone: the
 *   rest is memory this family allocated, in memory already. A runtime maps a file a layer or a part (▶ deepseek4.py,
 *   experts_file.py); past the table's room every range is asked of, as before it. A range freed and allocated again
 *   is still asked of, which is slower and no less right. */
#define X86_AVX2__ZZPRIVATE_FILE_RANGES 256u
static uintptr_t     x86_avx2__zzprivate_file_from[X86_AVX2__ZZPRIVATE_FILE_RANGES];
static uintptr_t     x86_avx2__zzprivate_file_to[X86_AVX2__ZZPRIVATE_FILE_RANGES];
static uint32_t      x86_avx2__zzprivate_file_ranges = 0u;            /* past the room: X86_AVX2__ZZPRIVATE_FILE_RANGES + 1 */
static sys__os__lock x86_avx2__zzprivate_file_lock = SYS__OS__LOCK_INIT;

static inline bool x86_avx2__memory_map_file(void* at, size_t bytes, uint64_t handle, uint64_t offset) {
    if (at == 0 || bytes == 0u || handle == 0ull || (uintptr_t)at % X86_AVX2__ZZPRIVATE_PAGE != 0u
        || offset % X86_AVX2__ZZPRIVATE_PAGE != 0ull)
        return false;
    sys__os__lock_take(&x86_avx2__zzprivate_file_lock);
    const uint32_t n = x86_avx2__zzprivate_file_ranges;
    if (n < X86_AVX2__ZZPRIVATE_FILE_RANGES) {
        x86_avx2__zzprivate_file_from[n] = (uintptr_t)at;
        x86_avx2__zzprivate_file_to[n] = (uintptr_t)at + bytes;
    }
    /* recorded before the mapping is made, so no reader ever takes the range for the family's own */
    __atomic_store_n(&x86_avx2__zzprivate_file_ranges, n < X86_AVX2__ZZPRIVATE_FILE_RANGES ? n + 1u : X86_AVX2__ZZPRIVATE_FILE_RANGES + 1u,
                     __ATOMIC_RELEASE);
    sys__os__lock_give(&x86_avx2__zzprivate_file_lock);
    return sys__os__memory_map_file(at, bytes, sys__file__zzabi_descriptor(handle), offset);
}
/* Whether `[from, to)` may hold a file's pages: it meets a range a file was mapped over, or the table ran out of room. */
static inline bool x86_avx2__zzprivate_file_backed(uintptr_t from, uintptr_t to) {
    const uint32_t n = __atomic_load_n(&x86_avx2__zzprivate_file_ranges, __ATOMIC_ACQUIRE);
    if (n > X86_AVX2__ZZPRIVATE_FILE_RANGES) return true;
    for (uint32_t i = 0u; i < n; ++i)
        if (from < x86_avx2__zzprivate_file_to[i] && x86_avx2__zzprivate_file_from[i] < to) return true;
    return false;
}
/* ⭐ THE PAGERS — a few threads that do nothing but bring a range's pages into memory, by touching them: a page not in
 * memory is read from the disk by the thread that touches it, so several of them keep the disk reading several
 * pieces at once (`MEASURED` on the Optane under node03's test: one thread faulting reads 1.6 GB/s, four 2.3 GB/s, the
 * drive's own rate). A range is handed over and not waited for; a computing thread that reaches a page still on its
 * way waits for that page alone. Started the first time a range is handed over (`SILVANN_PAGERS`, 4 by default). */
#define X86_AVX2__ZZPRIVATE_PAGER_RING    1024u
#define X86_AVX2__ZZPRIVATE_PAGERS_MAX    16u
static uintptr_t       x86_avx2__zzprivate_pager_from[X86_AVX2__ZZPRIVATE_PAGER_RING];
static uintptr_t       x86_avx2__zzprivate_pager_to[X86_AVX2__ZZPRIVATE_PAGER_RING];
static unsigned        x86_avx2__zzprivate_pager_head = 0u;
static unsigned        x86_avx2__zzprivate_pager_tail = 0u;
static unsigned        x86_avx2__zzprivate_pagers = 0u;
static sys__os__lock x86_avx2__zzprivate_pager_lock = SYS__OS__LOCK_INIT;
static sys__os__condition x86_avx2__zzprivate_pager_work = SYS__OS__CONDITION_INIT;
static volatile uint64_t x86_avx2__zzprivate_pager_sink = 0u;

static void* x86_avx2__zzprivate_pager(void* unused) {
    (void)unused;
    for (;;) {
        sys__os__lock_take(&x86_avx2__zzprivate_pager_lock);
        while (x86_avx2__zzprivate_pager_head == x86_avx2__zzprivate_pager_tail)
            sys__os__condition_wait(&x86_avx2__zzprivate_pager_work, &x86_avx2__zzprivate_pager_lock);
        const unsigned k = x86_avx2__zzprivate_pager_tail;
        const uintptr_t from = x86_avx2__zzprivate_pager_from[k], to = x86_avx2__zzprivate_pager_to[k];
        x86_avx2__zzprivate_pager_tail = (k + 1u) % X86_AVX2__ZZPRIVATE_PAGER_RING;
        sys__os__lock_give(&x86_avx2__zzprivate_pager_lock);
        uint64_t sum = 0u;
        for (uintptr_t a = from; a < to; a += X86_AVX2__ZZPRIVATE_PAGE) sum += *(volatile const uint8_t*)a;
        x86_avx2__zzprivate_pager_sink += sum;
    }
    return 0;
}

/* `[from, to)` handed to the pagers, a piece of `X86_AVX2__ZZPRIVATE_PAGE`s each so several share a long range; a full
 * ring drops the rest, which the computing thread then reads itself. */
static void x86_avx2__zzprivate_page_in(uintptr_t from, uintptr_t to) {
    const uintptr_t piece = 64u * X86_AVX2__ZZPRIVATE_PAGE;
    sys__os__lock_take(&x86_avx2__zzprivate_pager_lock);
    if (x86_avx2__zzprivate_pagers == 0u) {
        unsigned want = 4u;
        const char* said = getenv("SILVANN_PAGERS");
        if (said != 0) { const long n = strtol(said, 0, 10); if (n > 0) want = (unsigned)n; }
        if (want > X86_AVX2__ZZPRIVATE_PAGERS_MAX) want = X86_AVX2__ZZPRIVATE_PAGERS_MAX;
        for (unsigned t = 0u; t < want; ++t) {
            sys__os__thread id;
            if (!sys__os__thread_start(&id, x86_avx2__zzprivate_pager, 0)) break;
            sys__os__thread_detach(id);
            ++x86_avx2__zzprivate_pagers;
        }
    }
    for (uintptr_t a = from; a < to; a += piece) {
        const unsigned next = (x86_avx2__zzprivate_pager_head + 1u) % X86_AVX2__ZZPRIVATE_PAGER_RING;
        if (next == x86_avx2__zzprivate_pager_tail) break;
        x86_avx2__zzprivate_pager_from[x86_avx2__zzprivate_pager_head] = a;
        x86_avx2__zzprivate_pager_to[x86_avx2__zzprivate_pager_head] = a + piece < to ? a + piece : to;
        x86_avx2__zzprivate_pager_head = next;
    }
    sys__os__condition_wake_all(&x86_avx2__zzprivate_pager_work);
    sys__os__lock_give(&x86_avx2__zzprivate_pager_lock);
}

/* Whether every page of the range is in memory, from the system's own record (`sys__os__memory_resident`), a page-aligned run at a time;
 * when one is not, the range is handed to the pagers and not waited for. Memory the program allocated is always in it, and
 * is answered at once: asking the system of a routed expert's 4,320 pages took ~0.1 ms of a DeepSeek layer's 2 ms on the
 * CPU, six experts a position — `MEASURED` node03 2026-10-21 (measurements/2026-10-21_cpu_experts_node03.md). */
static inline bool x86_avx2__memory_prefetch(const void* at, size_t bytes) {
    if (at == 0 || bytes == 0u) return true;
    const uintptr_t first = (uintptr_t)at / X86_AVX2__ZZPRIVATE_PAGE * X86_AVX2__ZZPRIVATE_PAGE;
    const uintptr_t end = (uintptr_t)at + bytes;
    if (!x86_avx2__zzprivate_file_backed(first, end)) return true;
    unsigned char in[1024];
    bool all = true;
    for (uintptr_t a = first; a < end && all; a += sizeof in * X86_AVX2__ZZPRIVATE_PAGE) {
        const size_t pages = (end - a + X86_AVX2__ZZPRIVATE_PAGE - 1u) / X86_AVX2__ZZPRIVATE_PAGE;
        const size_t n = pages < sizeof in ? pages : sizeof in;
        if (!sys__os__memory_resident((const void*)a, n * X86_AVX2__ZZPRIVATE_PAGE, in)) return true;   /* not a mapping it can say of */
        for (size_t k = 0; k < n && all; ++k) all = (in[k] & 1u) != 0u;
    }
    if (all) return true;
    x86_avx2__zzprivate_page_in(first, end);
    return false;
}
static inline bool x86_avx2__file_read(uint64_t handle, uint64_t offset, uint64_t bytes, void* to) {
    return sys__file__zzabi_read(handle, offset, bytes, to);
}

/* ── DIRECT ACCESS AND MARKS — every device of this family is RAM of the one machine, which each socket's
 * cores read and write, and a launch is done when the call that asked returns, so there is nothing to wait for. */
static inline bool x86_avx2__peer_access(uint32_t other) { return other < x86_avx2__device_count(); }
static inline bool x86_avx2__mark_open(void** mark) {
    if (mark == 0) return false;
    *mark = (void*)&x86_avx2__compute_completed;   /* a stable, non-null, never-dereferenced token */
    return true;
}
static inline void x86_avx2__mark_close(void* mark) { (void)mark; }
static inline bool x86_avx2__mark_record(void* mark) { return mark != 0; }
static inline bool x86_avx2__mark_wait(void* mark) { return mark != 0; }
/* A queued copy is done when the call returns, so a mark is reached as soon as it is recorded. */
static inline bool x86_avx2__side_memory_write_queued(void* to, const void* from, size_t bytes, void* channel) {
    return x86_avx2__side_memory_write(to, from, bytes, channel);
}
static inline bool x86_avx2__side_memory_read_queued(void* to, const void* from, size_t bytes, void* channel) {
    return x86_avx2__side_memory_read(to, from, bytes, channel);
}
static inline bool x86_avx2__side_mark_record(void* mark, void* channel) { (void)channel; return mark != 0; }
static inline bool x86_avx2__mark_reached(void* mark) { return mark != 0; }

#endif /* SILVANN__SILICON_FAMILIES_X86_AVX2_PRIMITIVES_SYS_CUH */
