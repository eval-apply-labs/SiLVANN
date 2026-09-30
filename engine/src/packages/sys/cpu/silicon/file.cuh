#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_FILE_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_FILE_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

/* ══ READING A FILE — WHAT EVERY FAMILY'S FILE DOORS ARE MADE OF ═══════════════════════════════════════
 * ⚖ *"sys door"* (for the expert LRU's disk tier). The doors are the family's — `file_read` puts
 * the bytes in THAT family's memory — but opening and reading a file is the host's, the same for every
 * family on this machine. So it is written once, here, and published as `zzabi_` for the families to call,
 * the way nn publishes what an override may reuse: a family whose memory the program shares reads straight
 * into it; a card's reads through a bounce buffer and its own `memory_write`.
 * ⛳ A HANDLE IS TWO DESCRIPTORS OF THE ONE FILE, EACH PLUS ONE, so zero is never a file: the low half an ordinary
 *   one, the high half one opened `O_DIRECT` — which a read takes whenever its offset, its length and its memory are
 *   whole pages. `MEASURED` on node03's Optane (PCIe 3 x4), 6.3 MB pieces from one thread: 2.6 GB/s direct, the drive's own rate,
 *   against 1.4 buffered — and a direct read leaves no copy in the page cache, so a program that keeps what it read
 *   holds it once. A file system that refuses `O_DIRECT` leaves the high half zero and every read buffered. POSIX;
 *   another host answers these with its own calls. */
#define SYS__FILE__ZZPRIVATE_PAGE 4096ull
static inline int sys__file__zzabi_descriptor(uint64_t handle) { return (int)(uint32_t)handle - 1; }

static inline bool sys__file__zzabi_open(const char* path, uint64_t* handle) {
    if (path == 0 || handle == 0) return false;
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    const int direct = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    *handle = ((uint64_t)(uint32_t)fd + 1ull) | (direct < 0 ? 0ull : ((uint64_t)(uint32_t)direct + 1ull) << 32);
    return true;
}

static inline void sys__file__zzabi_close(uint64_t handle) {
    if (handle == 0ull) return;
    (void)close(sys__file__zzabi_descriptor(handle));
    if ((handle >> 32) != 0ull) (void)close((int)(handle >> 32) - 1);
}

/* `bytes` at `offset` into memory the program can write directly — straight from the drive when every one of the
 * three is whole pages. A short read is retried where it stopped; the end of the file before `bytes` is a false. */
static inline bool sys__file__zzabi_read(uint64_t handle, uint64_t offset, uint64_t bytes, void* to) {
    if (handle == 0ull || (to == 0 && bytes != 0ull)) return false;
    const bool whole = (handle >> 32) != 0ull && offset % SYS__FILE__ZZPRIVATE_PAGE == 0ull && bytes % SYS__FILE__ZZPRIVATE_PAGE == 0ull
                    && (uint64_t)(uintptr_t)to % SYS__FILE__ZZPRIVATE_PAGE == 0ull;
    const int fd = whole ? (int)(handle >> 32) - 1 : sys__file__zzabi_descriptor(handle);
    uint64_t done = 0ull;
    while (done < bytes) {
        const ssize_t got = pread(fd, (char*)to + done, (size_t)(bytes - done), (off_t)(offset + done));
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
