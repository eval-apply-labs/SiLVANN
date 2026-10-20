#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_FILE_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_FILE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../../../../operating_systems/os.cuh"   /* opening a file, reading it at an offset: the system's */

/* ══ READING A FILE — WHAT EVERY FAMILY'S FILE DOORS ARE MADE OF ═══════════════════════════════════════
 * ⚖ *"sys door"* (for the expert LRU's disk tier). The doors are the family's — `file_read` puts
 * the bytes in THAT family's memory — but opening and reading a file is the host's, the same for every
 * family on this machine. So it is written once, here, and published as `zzabi_` for the families to call,
 * the way nn publishes what an override may reuse: a family whose memory the program shares reads straight
 * into it; a card's reads through a bounce buffer and its own `memory_write`.
 * ⛳ A HANDLE IS TWO DESCRIPTORS OF THE ONE FILE, EACH PLUS ONE, so zero is never a file: the low half an ordinary
 *   one, the high half one opened direct, past the system's cache — which a read takes whenever its offset, its length and
 *   its memory are whole pages. `MEASURED` on node03's Optane (PCIe 3 x4), 6.3 MB pieces from one thread: 2.6 GB/s direct, the drive's own rate,
 *   against 1.4 buffered — and a direct read leaves no copy in the page cache, so a program that keeps what it read
 *   holds it once. A file system that refuses a direct open leaves the high half zero and every read buffered. The
 *   opening and the reading are the system's (`sys__os__file_*`). */
#define SYS__FILE__ZZPRIVATE_PAGE 4096ull
static inline int sys__file__zzabi_descriptor(uint64_t handle) { return (int)(uint32_t)handle - 1; }

static inline bool sys__file__zzabi_open(const char* path, uint64_t* handle) {
    if (path == 0 || handle == 0) return false;
    const sys__os__file fd = sys__os__file_open(path, false);
    if (fd < 0) return false;
    const sys__os__file direct = sys__os__file_open(path, true);
    *handle = ((uint64_t)(uint32_t)fd + 1ull) | (direct < 0 ? 0ull : ((uint64_t)(uint32_t)direct + 1ull) << 32);
    return true;
}

static inline void sys__file__zzabi_close(uint64_t handle) {
    if (handle == 0ull) return;
    sys__os__file_close(sys__file__zzabi_descriptor(handle));
    if ((handle >> 32) != 0ull) sys__os__file_close((int)(handle >> 32) - 1);
}

/* ⛳ A SMALL READ OFF THE PAGES STILL GOES STRAIGHT TO THE DRIVE: the whole pages around it into a buffer of its own,
 *   and its bytes out of that — a pack read where its experts lie has a few such slices an expert, kilobytes each
 *   (a layer's scales), beside megabytes of whole pages, and through the system's cache each was a copy kept twice
 *   and a slower read. `MEASURED` 2026-10-19 on the precision, before it: the 122B's experts from its pack 2.7% and the
 *   35B's 4% below the same experts from a file of whole pages (NN-97). Past `SYS__FILE__ZZPRIVATE_ROUND` the cache
 *   is used, so the buffer stays small; the buffer is each read's own, as the loader reads from several threads. */
#define SYS__FILE__ZZPRIVATE_ROUND (1ull << 20)
static inline bool sys__file__zzprivate_read_round(int direct, uint64_t offset, uint64_t bytes, void* to) {
    const uint64_t first = offset & ~(SYS__FILE__ZZPRIVATE_PAGE - 1ull);
    const uint64_t span = (offset + bytes + SYS__FILE__ZZPRIVATE_PAGE - 1ull) / SYS__FILE__ZZPRIVATE_PAGE * SYS__FILE__ZZPRIVATE_PAGE - first;
    char* round = (char*)sys__os__memory_aligned((size_t)SYS__FILE__ZZPRIVATE_PAGE, (size_t)span);
    if (round == 0) return false;
    const uint64_t need = offset + bytes - first;            /* the file may end inside the last page */
    uint64_t done = 0ull;
    while (done < need) {
        const int64_t got = sys__os__file_read_at(direct, round + done, (size_t)(span - done), first + done);
        if (got <= 0) break;
        done += (uint64_t)got;
        if (done % SYS__FILE__ZZPRIVATE_PAGE != 0ull) break;   /* a direct read short of a page is the file's end */
    }
    const bool ok = done >= need;
    if (ok) memcpy(to, round + (offset - first), (size_t)bytes);
    sys__os__memory_aligned_free(round);
    return ok;
}

/* `bytes` at `offset` into memory the program can write directly — straight from the drive when every one of the
 * three is whole pages, or through a buffer of whole pages when the read is small. A short read is retried where it
 * stopped; the end of the file before `bytes` is a false. */
static inline bool sys__file__zzabi_read(uint64_t handle, uint64_t offset, uint64_t bytes, void* to) {
    if (handle == 0ull || (to == 0 && bytes != 0ull)) return false;
    const bool whole = (handle >> 32) != 0ull && offset % SYS__FILE__ZZPRIVATE_PAGE == 0ull && bytes % SYS__FILE__ZZPRIVATE_PAGE == 0ull
                    && (uint64_t)(uintptr_t)to % SYS__FILE__ZZPRIVATE_PAGE == 0ull;
    if (!whole && (handle >> 32) != 0ull && bytes != 0ull && bytes <= SYS__FILE__ZZPRIVATE_ROUND)
        return sys__file__zzprivate_read_round((int)(handle >> 32) - 1, offset, bytes, to);
    const int fd = whole ? (int)(handle >> 32) - 1 : sys__file__zzabi_descriptor(handle);
    uint64_t done = 0ull;
    while (done < bytes) {
        const int64_t got = sys__os__file_read_at(fd, (char*)to + done, (size_t)(bytes - done), offset + done);
        if (got <= 0) return false;
        done += (uint64_t)got;
    }
    return true;
}

/* The same bytes into a card's memory: read a piece into the host, hand it to the family's `memory_write`,
 * and go on. ⛳ THE PIECE IS 8 MiB, so the bounce is bounded whatever is asked for. */
typedef bool (*sys__file__zzabi_writer)(void* to, const void* from, size_t bytes);
#define SYS__FILE__BOUNCE_BYTES (8ull << 20)
static inline bool sys__file__zzabi_read_through(uint64_t handle, uint64_t offset, uint64_t bytes, void* to,
                                                 sys__file__zzabi_writer write) {
    if (handle == 0ull || write == 0 || (to == 0 && bytes != 0ull)) return false;
    const uint64_t piece = bytes < SYS__FILE__BOUNCE_BYTES ? bytes : SYS__FILE__BOUNCE_BYTES;
    void* bounce = piece == 0ull ? 0 : malloc((size_t)piece);
    if (piece != 0ull && bounce == 0) return false;
    bool ok = true;
    for (uint64_t done = 0ull; ok && done < bytes; done += piece) {
        const uint64_t n = bytes - done < piece ? bytes - done : piece;
        ok = sys__file__zzabi_read(handle, offset + done, n, bounce) && write((char*)to + done, bounce, (size_t)n);
    }
    free(bounce);
    return ok;
}

#endif /* SILVANN__PACKAGES_SYS_CPU_SILICON_FILE_CUH */
