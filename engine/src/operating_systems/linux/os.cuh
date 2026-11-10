#ifndef SILVANN__OPERATING_SYSTEMS_LINUX_OS_CUH
#define SILVANN__OPERATING_SYSTEMS_LINUX_OS_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

/* ══ LINUX'S ANSWERS TO `sys__os__*` — POSIX, AND LINUX'S OWN `/proc` AND `/sys` ══════════════════════════════════
 * Each call is the POSIX or Linux one it names, inline, so a caller moved onto the layer compiles to what it compiled
 * to before.
 * ▶ the contract: `packages/sys/contracts/abi/os.cuh`. */
typedef pthread_t       sys__os__thread;
typedef pthread_mutex_t sys__os__lock;
typedef pthread_cond_t  sys__os__condition;
typedef pthread_once_t  sys__os__once;
typedef pthread_key_t   sys__os__key;
typedef void*           sys__os__library;
typedef int             sys__os__file;
typedef cpu_set_t       sys__os__cpus;
#define SYS__OS__CPUS_MOST       CPU_SETSIZE
#define SYS__OS__LIBRARY_SUFFIX  ".so"
#define SYS__OS__NVIDIA_DRIVER   "libcuda.so.1"
#define SYS__OS__LOCK_INIT       PTHREAD_MUTEX_INITIALIZER
#define SYS__OS__CONDITION_INIT  PTHREAD_COND_INITIALIZER
#define SYS__OS__ONCE_INIT       PTHREAD_ONCE_INIT
#define SYS__OS__EXPORT          __attribute__((visibility("default")))

#include "../../packages/sys/contracts/abi/os.cuh"

static inline bool sys__os__thread_start(sys__os__thread* thread, sys__os__thread_body body, void* arg) {
    return pthread_create(thread, 0, body, arg) == 0;
}
static inline void sys__os__thread_join(sys__os__thread thread)   { (void)pthread_join(thread, 0); }
static inline void sys__os__thread_detach(sys__os__thread thread) { (void)pthread_detach(thread); }

static inline void sys__os__lock_init(sys__os__lock* lock) { (void)pthread_mutex_init(lock, 0); }
static inline void sys__os__lock_take(sys__os__lock* lock) { (void)pthread_mutex_lock(lock); }
static inline bool sys__os__lock_try(sys__os__lock* lock)  { return pthread_mutex_trylock(lock) == 0; }
static inline void sys__os__lock_give(sys__os__lock* lock) { (void)pthread_mutex_unlock(lock); }

static inline void sys__os__condition_init(sys__os__condition* condition) { (void)pthread_cond_init(condition, 0); }
static inline void sys__os__condition_wait(sys__os__condition* condition, sys__os__lock* lock) {
    (void)pthread_cond_wait(condition, lock);
}
/* ⛳ THE DEADLINE ON THE CONDITION'S OWN CLOCK, which is the system's wall clock for one made by the initializer. */
static inline void sys__os__condition_wait_for(sys__os__condition* condition, sys__os__lock* lock, uint64_t ns) {
    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    const uint64_t at = (uint64_t)until.tv_nsec + ns;
    until.tv_sec += (time_t)(at / 1000000000ull);
    until.tv_nsec = (long)(at % 1000000000ull);
    (void)pthread_cond_timedwait(condition, lock, &until);
}
static inline void sys__os__condition_wake_one(sys__os__condition* condition) { (void)pthread_cond_signal(condition); }
static inline void sys__os__condition_wake_all(sys__os__condition* condition) { (void)pthread_cond_broadcast(condition); }

static inline void sys__os__once_run(sys__os__once* once, sys__os__once_body body) { (void)pthread_once(once, body); }

static inline bool sys__os__key_make(sys__os__key* key, sys__os__key_exit at_exit) { return pthread_key_create(key, at_exit) == 0; }
static inline void sys__os__key_set(sys__os__key key, void* value) { (void)pthread_setspecific(key, value); }

static inline sys__os__library sys__os__library_open(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static inline void* sys__os__library_find(sys__os__library library, const char* name) { return dlsym(library, name); }
static inline const char* sys__os__library_why(void) { return dlerror(); }
static inline const char* sys__os__library_path_of(const void* address) {
    Dl_info info;
    return dladdr(address, &info) == 0 ? 0 : info.dli_fname;
}
static inline bool sys__os__path_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}
static inline bool sys__os__path_is_directory(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}
static inline uintptr_t sys__os__library_base_of(const void* address) {
    Dl_info info;
    return dladdr(address, &info) ? (uintptr_t)info.dli_fbase : 0u;
}

static inline void sys__os__clock(struct timespec* now) { clock_gettime(CLOCK_MONOTONIC, now); }
static inline void sys__os__sleep(const struct timespec* how_long) { nanosleep(how_long, 0); }
static inline void sys__os__yield(void) { sched_yield(); }

/* ⛳ ALWAYS INLINE, the three of them: `MEASURED` 2026-10-18 (precision), left to GCC the engine module's package
 *   initialisation compiled differently once `file.cuh` called them — its own inlining weighed one call deeper — and
 *   forced, the module is the file it was before the layer, byte for byte. */
static inline __attribute__((always_inline)) sys__os__file sys__os__file_open(const char* path, bool direct) {
    return open(path, direct ? O_RDONLY | O_CLOEXEC | O_DIRECT : O_RDONLY | O_CLOEXEC);
}
static inline __attribute__((always_inline)) void sys__os__file_close(sys__os__file file) { (void)close(file); }
static inline __attribute__((always_inline)) int64_t sys__os__file_read_at(sys__os__file file, void* to, size_t bytes,
                                                                          uint64_t offset) {
    return pread(file, to, bytes, (off_t)offset);
}

static inline void* sys__os__memory_map(size_t bytes) {
    void* base = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return base == MAP_FAILED ? 0 : base;
}
static inline void sys__os__memory_unmap(void* base, size_t bytes) { (void)munmap(base, bytes); }
static inline bool sys__os__memory_guard(void* at, size_t bytes) { return mprotect(at, bytes, PROT_NONE) == 0; }
static inline bool sys__os__memory_map_file(void* at, size_t bytes, sys__os__file file, uint64_t offset) {
    return mmap(at, bytes, PROT_READ, MAP_SHARED | MAP_FIXED, file, (off_t)offset) == at;
}
static inline bool sys__os__memory_resident(const void* at, size_t bytes, unsigned char* each) {
    return mincore((void*)at, bytes, each) == 0;
}
static inline void sys__os__memory_will_need(void* at, size_t bytes) { (void)madvise(at, bytes, MADV_WILLNEED); }
/* `mbind` by its number: the C library has no wrapper for it, and libnuma is not linked */
static inline bool sys__os__memory_interleave(void* base, size_t bytes, const unsigned long* nodes, unsigned long most) {
    return syscall(SYS_mbind, base, bytes, 3 /* MPOL_INTERLEAVE */, nodes, most, 0u) == 0;
}
static inline bool sys__os__memory_bind(void* base, size_t bytes, const unsigned long* nodes, unsigned long most) {
    return syscall(SYS_mbind, base, bytes, 2 /* MPOL_BIND */, nodes, most, 0u) == 0;
}
static inline void* sys__os__memory_aligned(size_t alignment, size_t bytes) { return aligned_alloc(alignment, bytes); }
static inline void sys__os__memory_aligned_free(void* at) { free(at); }

static inline void sys__os__cpus_clear(sys__os__cpus* set) { CPU_ZERO(set); }
static inline void sys__os__cpus_add(sys__os__cpus* set, int cpu) { CPU_SET(cpu, set); }
static inline bool sys__os__cpus_has(const sys__os__cpus* set, int cpu) { return CPU_ISSET(cpu, set); }
static inline int  sys__os__cpus_count(const sys__os__cpus* set) { return CPU_COUNT(set); }
static inline bool sys__os__cpus_of_thread(sys__os__cpus* set) { return sched_getaffinity(0, sizeof *set, set) == 0; }
static inline bool sys__os__cpus_to_thread(const sys__os__cpus* set) { return sched_setaffinity(0, sizeof *set, set) == 0; }
static inline void sys__os__cpu_pin_thread(int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}
static inline int sys__os__cpu_now(void) { return sched_getcpu(); }
/* ⛳ THE TOPOLOGY IS `/sys/devices/system/cpu`: a CPU's node is the `node<n>` link in its folder, its package and core
 *   two numbers in its `topology/`. A container without `/sys` answers -1 for each. */
static inline int sys__os__cpu_node(int cpu) {
    char path[128];
    for (int n = 0; n < 64; ++n) {
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/node%d", cpu, n);
        if (access(path, F_OK) == 0) return n;
    }
    return -1;
}
static inline int sys__os__zzprivate_read_int(const char* path) {
    FILE* f = fopen(path, "r"); int v = -1;
    if (f) { if (fscanf(f, "%d", &v) != 1) v = -1; fclose(f); }
    return v;
}
static inline void sys__os__cpu_core(int cpu, int* package, int* core) {
    char path[128];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    *package = sys__os__zzprivate_read_int(path);
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
    *core = sys__os__zzprivate_read_int(path);
}
/* each CPU's line of `/proc/stat`: busy is user, nice, system, irq, softirq and steal; total adds idle and iowait */
static inline void sys__os__cpu_ticks(unsigned long long* busy, unsigned long long* total, int most) {
    FILE* f = fopen("/proc/stat", "r"); char line[512];
    if (!f) return;
    while (fgets(line, sizeof line, f)) {
        int cpu; unsigned long long u, n, s, idle, io, irq, sirq, steal;
        if (sscanf(line, "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu", &cpu, &u, &n, &s, &idle, &io, &irq, &sirq, &steal) == 9
            && cpu >= 0 && cpu < most) {
            busy[cpu] = u + n + s + irq + sirq + steal; total[cpu] = busy[cpu] + idle + io;
        }
    }
    fclose(f);
}
static inline long sys__os__cpu_l2_bytes(void) { return sysconf(_SC_LEVEL2_CACHE_SIZE); }

/* Always inline, as the file's calls are: `MEASURED`, left to GCC the engine module's pybind setup compiled differently.
 * Spelled through a name, so a definition below reads as one. */
#define SYS__OS__ZZPRIVATE_ALWAYS __attribute__((always_inline))

/* ⛳ OPENCL'S LOADER, ASKED FOR ONE VENDOR'S GPUS (`pci_vendor`): each named by its OpenCL device name, as Windows'
 *   `cards` names every GPU's — an AMD card's is its `gfx`, Intel's its product — up to `most`, in the order the loader
 *   lists them. Loaded by name and left loaded for the family that opens next. The calls' types are OpenCL's, spelled
 *   here without its headers. */
static inline SYS__OS__ZZPRIVATE_ALWAYS uint32_t sys__os__zzprivate_opencl_gpus(char* names, uint32_t name_bytes, uint32_t most,
                                                                                uint32_t pci_vendor) {
    if (most == 0u) return 0u;
    typedef int32_t (*platforms_fn)(uint32_t, void**, uint32_t*);
    typedef int32_t (*devices_fn)(void*, uint64_t, uint32_t, void**, uint32_t*);
    typedef int32_t (*info_fn)(void*, uint32_t, size_t, void*, size_t*);
    void* cl = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (cl == 0) return 0u;
    const platforms_fn platforms = (platforms_fn)dlsym(cl, "clGetPlatformIDs");
    const devices_fn   devices   = (devices_fn)dlsym(cl, "clGetDeviceIDs");
    const info_fn      info      = (info_fn)dlsym(cl, "clGetDeviceInfo");
    void* platform[16];
    uint32_t platform_count = 0u, n = 0u;
    if (platforms == 0 || devices == 0 || info == 0 || platforms(16u, platform, &platform_count) != 0) return 0u;
    for (uint32_t p = 0u; p < platform_count && p < 16u && n < most; ++p) {
        void* device[16];
        uint32_t device_count = 0u;
        /* (1 << 2) is CL_DEVICE_TYPE_GPU */
        if (devices(platform[p], 1ull << 2, 16u, device, &device_count) != 0) continue;
        for (uint32_t d = 0u; d < device_count && d < 16u && n < most; ++d) {
            uint32_t vendor = 0u;
            char name[256];
            /* 0x1001 is CL_DEVICE_VENDOR_ID, 0x102B CL_DEVICE_NAME */
            if (info(device[d], 0x1001u, sizeof vendor, &vendor, 0) != 0 || vendor != pci_vendor) continue;
            memset(name, 0, sizeof name);
            if (info(device[d], 0x102Bu, sizeof name - 1u, name, 0) != 0 || name[0] == 0) continue;
            char* to = names + (size_t)(n++) * name_bytes;
            strncpy(to, name, name_bytes - 1u);
            to[name_bytes - 1u] = 0;
        }
    }
    return n;
}

/* ⛳ THE KERNEL DRIVER'S TOPOLOGY, `/sys/class/kfd/kfd/topology/nodes/<n>/properties`: every GPU node carries
 *   `gfx_target_version` as a number, major·10000 + minor·100 + stepping, and its name is `gfx` then the major in decimal
 *   and the other two in hex — 90006 is `gfx906`, 90010 `gfx90a`, 110000 `gfx1100`. A CPU node carries 0 and is
 *   skipped. The nodes are read in numeric order, so the answer is the same on every read.
 * ⛳ WSL2 HAS NO KFD: its GPUs are `/dev/dxg`, behind Windows' driver, and the topology names none. There an AMD card is
 *   named as Windows names it, by OpenCL's loader (PCI vendor 0x1002) — ROCm's OpenCL inside WSL2, from the ROCm the AMD
 *   family then opens (`MEASURED` an RX 9070 XT on SaladCloud, ROCm 7.2: `gfx1201`,
 *   measurements/2026-11-10_amd_radeon_wave64.md). NVIDIA's cards are its driver's to name, there as here. */
static inline __attribute__((always_inline)) uint32_t sys__os__cards(char* names, uint32_t name_bytes, uint32_t most) {
    uint32_t n = 0u;
    for (uint32_t node = 0u; node < 256u && n < most; ++node) {
        char path[96];
        snprintf(path, sizeof path, "/sys/class/kfd/kfd/topology/nodes/%u/properties", node);
        FILE* f = fopen(path, "r");
        if (f == 0) break;               /* no kfd at all, or past the last node: they are numbered densely */
        char key[64];
        unsigned long long value = 0ull, version = 0ull;
        while (fscanf(f, "%63s %llu", key, &value) == 2)
            if (strcmp(key, "gfx_target_version") == 0) version = value;
        fclose(f);
        if (version == 0ull) continue;   /* a CPU node */
        snprintf(names + (size_t)(n++) * name_bytes, name_bytes, "gfx%llu%llx%llx",
                 version / 10000ull, (version / 100ull) % 100ull, version % 100ull);
    }
    if (n == 0u && access("/dev/dxg", F_OK) == 0) n = sys__os__zzprivate_opencl_gpus(names, name_bytes, most, 0x1002u);
    return n;
}
/* ⛳ LINUX'S INTEL GPUS, BY OPENCL'S LOADER — the kernel's topology above is AMD's driver's alone. Asked only where the
 *   system shows a display device of Intel's (PCI vendor 0x8086, `/sys/class/drm/card<n>/device/vendor`), read before any
 *   library is loaded; then OpenCL's loader names Intel's devices (above). */
static inline __attribute__((always_inline)) uint32_t sys__os__intel_gpus(char* names, uint32_t name_bytes, uint32_t most) {
    bool intel = false;
    for (int card = 0; card < 16 && !intel; ++card) {
        char path[64];
        snprintf(path, sizeof path, "/sys/class/drm/card%d/device/vendor", card);
        FILE* f = fopen(path, "r");
        if (f == 0) continue;
        unsigned int vendor = 0u;
        if (fscanf(f, "%x", &vendor) == 1 && vendor == 0x8086u) intel = true;
        fclose(f);
    }
    return intel ? sys__os__zzprivate_opencl_gpus(names, name_bytes, most, 0x8086u) : 0u;
}

#endif /* SILVANN__OPERATING_SYSTEMS_LINUX_OS_CUH */
