#ifndef SILVANN__OPERATING_SYSTEMS_WINDOWS_OS_CUH
#define SILVANN__OPERATING_SYSTEMS_WINDOWS_OS_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#include <windows.h>
/* Win32's headers spell plain words as macros — `near` and `far` empty, `IN`, `OUT` and `OPTIONAL` empty — which the
 * tree spells as names; they are those headers' alone, so they end here. */
#undef near
#undef far
#undef IN
#undef OUT
#undef OPTIONAL
#include <malloc.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ══ WINDOWS' ANSWERS TO `sys__os__*` — WIN32 THREADS, SLIM LOCKS, CONDITION VARIABLES ═════════════════════════════
 * `ASSUMED` until a Windows compiler has built it and the checks have run on a Windows PC (NN-94): written against
 * Win32's documentation, never compiled. Each type has a static initializer, as POSIX's has, so a lock or a condition
 * declared at file scope needs no call to make it. A slim lock is given by the thread that took it — the contract's
 * rule, and POSIX's for a default mutex.
 * ▶ the contract: `packages/sys/contracts/abi/os.cuh`. */
typedef HANDLE             sys__os__thread;
typedef SRWLOCK            sys__os__lock;
typedef CONDITION_VARIABLE sys__os__condition;
typedef INIT_ONCE          sys__os__once;
typedef DWORD              sys__os__key;
typedef void*              sys__os__library;
typedef int                sys__os__file;
#define SYS__OS__CPUS_MOST       1024
typedef struct sys__os__cpus { unsigned long long word[SYS__OS__CPUS_MOST / 64]; } sys__os__cpus;
#define SYS__OS__LIBRARY_SUFFIX  ".dll"
#define SYS__OS__NVIDIA_DRIVER   "nvcuda.dll"
#define SYS__OS__LOCK_INIT       SRWLOCK_INIT
#define SYS__OS__CONDITION_INIT  CONDITION_VARIABLE_INIT
#define SYS__OS__ONCE_INIT       INIT_ONCE_STATIC_INIT
#define SYS__OS__EXPORT          __declspec(dllexport)

#include "../../packages/sys/contracts/abi/os.cuh"

/* A Win32 thread runs `DWORD (WINAPI*)(LPVOID)`; the contract's body is `void* (*)(void*)`, so a thread starts on this
 * trampoline, which takes the body and its argument from the heap and frees them before running it. */
typedef struct sys__os__windows_start { sys__os__thread_body body; void* arg; } sys__os__windows_start;
static DWORD WINAPI sys__os__zzprivate_trampoline(LPVOID start) {
    const sys__os__windows_start s = *(const sys__os__windows_start*)start;
    free(start);
    (void)s.body(s.arg);
    return 0u;
}
static inline bool sys__os__thread_start(sys__os__thread* thread, sys__os__thread_body body, void* arg) {
    sys__os__windows_start* s = (sys__os__windows_start*)malloc(sizeof *s);
    if (s == 0) return false;
    s->body = body;
    s->arg = arg;
    const HANDLE h = CreateThread(0, 0, sys__os__zzprivate_trampoline, s, 0, 0);
    if (h == 0) { free(s); return false; }
    *thread = h;
    return true;
}
static inline void sys__os__thread_join(sys__os__thread thread)   { (void)WaitForSingleObject(thread, INFINITE); (void)CloseHandle(thread); }
static inline void sys__os__thread_detach(sys__os__thread thread) { (void)CloseHandle(thread); }

static inline void sys__os__lock_init(sys__os__lock* lock) { InitializeSRWLock(lock); }
static inline void sys__os__lock_take(sys__os__lock* lock) { AcquireSRWLockExclusive(lock); }
static inline bool sys__os__lock_try(sys__os__lock* lock)  { return TryAcquireSRWLockExclusive(lock) != 0; }
static inline void sys__os__lock_give(sys__os__lock* lock) { ReleaseSRWLockExclusive(lock); }

static inline void sys__os__condition_init(sys__os__condition* condition) { InitializeConditionVariable(condition); }
static inline void sys__os__condition_wait(sys__os__condition* condition, sys__os__lock* lock) {
    (void)SleepConditionVariableSRW(condition, lock, INFINITE, 0);
}
static inline void sys__os__condition_wake_one(sys__os__condition* condition) { WakeConditionVariable(condition); }
static inline void sys__os__condition_wake_all(sys__os__condition* condition) { WakeAllConditionVariable(condition); }

/* `InitOnceExecuteOnce` calls `BOOL (CALLBACK*)(PINIT_ONCE, PVOID, PVOID*)`; the body rides in its parameter. */
static BOOL CALLBACK sys__os__zzprivate_once(PINIT_ONCE once, PVOID body, PVOID* context) {
    (void)once;
    (void)context;
    ((sys__os__once_body)body)();
    return TRUE;
}
static inline void sys__os__once_run(sys__os__once* once, sys__os__once_body body) {
    (void)InitOnceExecuteOnce(once, sys__os__zzprivate_once, (PVOID)body, 0);
}

/* A fiber-local slot, whose callback runs as a thread ends, as POSIX's key destructor does. On x64 one calling
 * convention serves both, so the contract's `at_exit` is the callback itself. */
static inline bool sys__os__key_make(sys__os__key* key, sys__os__key_exit at_exit) {
    const DWORD k = FlsAlloc((PFLS_CALLBACK_FUNCTION)at_exit);
    if (k == FLS_OUT_OF_INDEXES) return false;
    *key = k;
    return true;
}
static inline void sys__os__key_set(sys__os__key key, void* value) { (void)FlsSetValue(key, value); }

/* Win32 takes a path as UTF-16: the contract's UTF-8 becomes it here, and what Win32 answers becomes UTF-8 again, its
 * `\` turned to `/` so a caller's `rfind('/')` finds the folder. */
#define SYS__OS__WINDOWS_PATH 4096
static inline bool sys__os__windows_wide(const char* path, wchar_t* wide) {
    return path != 0 && MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, SYS__OS__WINDOWS_PATH) > 0;
}
static inline sys__os__library sys__os__library_open(const char* path) {
    wchar_t wide[SYS__OS__WINDOWS_PATH];
    if (!sys__os__windows_wide(path, wide)) return 0;
    for (wchar_t* c = wide; *c != L'\0'; ++c)   /* the altered search path takes its folder from `\` alone */
        if (*c == L'/') *c = L'\\';
    return (sys__os__library)LoadLibraryExW(wide, 0, LOAD_WITH_ALTERED_SEARCH_PATH);
}
static inline void* sys__os__library_find(sys__os__library library, const char* name) {
    return (void*)GetProcAddress((HMODULE)library, name);
}
static inline const char* sys__os__library_why(void) {
    static __declspec(thread) char why[512];
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, 0, GetLastError(), 0, why,
                                   (DWORD)sizeof why, 0);
    return n == 0u ? 0 : why;
}
static inline const char* sys__os__library_path_of(const void* address) {
    static __declspec(thread) char file[3u * SYS__OS__WINDOWS_PATH];
    HMODULE module = 0;
    wchar_t wide[SYS__OS__WINDOWS_PATH];
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)address, &module)) return 0;
    const DWORD n = GetModuleFileNameW(module, wide, SYS__OS__WINDOWS_PATH);
    if (n == 0u || n >= SYS__OS__WINDOWS_PATH) return 0;
    if (WideCharToMultiByte(CP_UTF8, 0, wide, -1, file, (int)sizeof file, 0, 0) <= 0) return 0;
    for (char* c = file; *c != 0; ++c) if (*c == '\\') *c = '/';
    return file;
}
static inline bool sys__os__path_exists(const char* path) {
    wchar_t wide[SYS__OS__WINDOWS_PATH];
    return sys__os__windows_wide(path, wide) && GetFileAttributesW(wide) != INVALID_FILE_ATTRIBUTES;
}
static inline bool sys__os__path_is_directory(const char* path) {
    wchar_t wide[SYS__OS__WINDOWS_PATH];
    if (!sys__os__windows_wide(path, wide)) return false;
    const DWORD a = GetFileAttributesW(wide);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) != 0u;
}
/* a module's handle is the address it was loaded at */
static inline uintptr_t sys__os__library_base_of(const void* address) {
    HMODULE module = 0;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)address, &module)) return 0u;
    return (uintptr_t)module;
}

/* The performance counter, which never steps back; a sleep in whole milliseconds, rounded up. */
static inline void sys__os__clock(struct timespec* now) {
    LARGE_INTEGER hz, count;
    (void)QueryPerformanceFrequency(&hz);
    (void)QueryPerformanceCounter(&count);
    now->tv_sec = (time_t)(count.QuadPart / hz.QuadPart);
    now->tv_nsec = (long)(count.QuadPart % hz.QuadPart * 1000000000ll / hz.QuadPart);
}
static inline void sys__os__sleep(const struct timespec* how_long) {
    Sleep((DWORD)(how_long->tv_sec * 1000 + (how_long->tv_nsec + 999999L) / 1000000L));
}
static inline void sys__os__yield(void) { (void)SwitchToThread(); }

/* ⛳ A FILE IS OPENED FOR OVERLAPPED READS, each read at its own offset and waited for, so several threads read one
 *   file at once as `pread` lets them; a handle opened without it has its reads taken one at a time. A handle fits an
 *   int: Windows keeps them within 32 bits, and one sign-extends back. */
static inline sys__os__file sys__os__file_open(const char* path, bool direct) {
    wchar_t wide[SYS__OS__WINDOWS_PATH];
    if (!sys__os__windows_wide(path, wide)) return -1;
    const DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0u);
    const HANDLE h = CreateFileW(wide, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0, OPEN_EXISTING,
                                 flags, 0);
    return h == INVALID_HANDLE_VALUE ? -1 : (sys__os__file)(intptr_t)h;
}
static inline void sys__os__file_close(sys__os__file file) { (void)CloseHandle((HANDLE)(intptr_t)file); }
static inline int64_t sys__os__file_read_at(sys__os__file file, void* to, size_t bytes, uint64_t offset) {
    OVERLAPPED at;
    memset(&at, 0, sizeof at);
    at.Offset = (DWORD)offset;
    at.OffsetHigh = (DWORD)(offset >> 32);
    at.hEvent = CreateEventW(0, TRUE, FALSE, 0);
    if (at.hEvent == 0) return -1;
    const HANDLE h = (HANDLE)(intptr_t)file;
    const DWORD ask = bytes > (1u << 30) ? (1u << 30) : (DWORD)bytes;   /* a short read: the caller asks for the rest */
    DWORD got = 0u;
    BOOL ok = ReadFile(h, to, ask, 0, &at);
    if (!ok && GetLastError() == ERROR_IO_PENDING) ok = TRUE;
    if (ok) ok = GetOverlappedResult(h, &at, &got, TRUE);
    const DWORD why = ok ? 0u : GetLastError();
    (void)CloseHandle(at.hEvent);
    if (!ok) return why == ERROR_HANDLE_EOF ? 0 : -1;
    return (int64_t)got;
}

/* ⛳ MAPPED MEMORY IS COMMITTED: Windows has no pages without a reservation behind them, so a map larger than memory
 *   and the page file is refused here where Linux would map it — and a file is not mapped over memory already held
 *   (`memory_map_file` false), so the program reads a file in instead. One system node: no interleave, no binding. */
static inline void* sys__os__memory_map(size_t bytes) { return VirtualAlloc(0, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); }
static inline void sys__os__memory_unmap(void* base, size_t bytes) { (void)bytes; (void)VirtualFree(base, 0, MEM_RELEASE); }
static inline bool sys__os__memory_guard(void* at, size_t bytes) {
    DWORD was = 0u;
    return VirtualProtect(at, bytes, PAGE_NOACCESS, &was) != 0;
}
static inline bool sys__os__memory_map_file(void* at, size_t bytes, sys__os__file file, uint64_t offset) {
    (void)at; (void)bytes; (void)file; (void)offset;
    return false;
}
static inline bool sys__os__memory_resident(const void* at, size_t bytes, unsigned char* each) {
    (void)at; (void)bytes; (void)each;
    return false;
}
static inline void sys__os__memory_will_need(void* at, size_t bytes) { (void)at; (void)bytes; }
static inline bool sys__os__memory_interleave(void* base, size_t bytes, const unsigned long* nodes, unsigned long most) {
    (void)base; (void)bytes; (void)nodes; (void)most;
    return false;
}
static inline bool sys__os__memory_bind(void* base, size_t bytes, const unsigned long* nodes, unsigned long most) {
    (void)base; (void)bytes; (void)nodes; (void)most;
    return false;
}
static inline void* sys__os__memory_aligned(size_t alignment, size_t bytes) { return _aligned_malloc(bytes, alignment); }
static inline void sys__os__memory_aligned_free(void* at) { _aligned_free(at); }

/* ⛳ A CPU'S NUMBER IS ITS GROUP'S TIMES 64 PLUS ITS PLACE IN THE GROUP: Windows counts processors in groups of up to
 *   64, and a thread runs within one group. A set handed to a thread is its lowest group's part. */
static inline void sys__os__cpus_clear(sys__os__cpus* set) { memset(set, 0, sizeof *set); }
static inline void sys__os__cpus_add(sys__os__cpus* set, int cpu) {
    if (cpu >= 0 && cpu < SYS__OS__CPUS_MOST) set->word[cpu / 64] |= 1ull << (cpu % 64);
}
static inline bool sys__os__cpus_has(const sys__os__cpus* set, int cpu) {
    return cpu >= 0 && cpu < SYS__OS__CPUS_MOST && ((set->word[cpu / 64] >> (cpu % 64)) & 1ull) != 0ull;
}
static inline int sys__os__cpus_count(const sys__os__cpus* set) {
    int n = 0;
    for (int w = 0; w < SYS__OS__CPUS_MOST / 64; ++w)
        for (unsigned long long b = set->word[w]; b != 0ull; b &= b - 1ull) ++n;
    return n;
}
static inline bool sys__os__cpus_of_thread(sys__os__cpus* set) {
    GROUP_AFFINITY g;
    memset(&g, 0, sizeof g);
    if (!GetThreadGroupAffinity(GetCurrentThread(), &g) || g.Group >= SYS__OS__CPUS_MOST / 64) return false;
    sys__os__cpus_clear(set);
    set->word[g.Group] = (unsigned long long)g.Mask;
    return true;
}
static inline bool sys__os__cpus_to_thread(const sys__os__cpus* set) {
    for (int w = 0; w < SYS__OS__CPUS_MOST / 64; ++w) {
        if (set->word[w] == 0ull) continue;
        GROUP_AFFINITY g;
        memset(&g, 0, sizeof g);
        g.Group = (WORD)w;
        g.Mask = (KAFFINITY)set->word[w];
        return SetThreadGroupAffinity(GetCurrentThread(), &g, 0) != 0;
    }
    return false;
}
static inline void sys__os__cpu_pin_thread(int cpu) {
    GROUP_AFFINITY g;
    memset(&g, 0, sizeof g);
    g.Group = (WORD)(cpu / 64);
    g.Mask = (KAFFINITY)1 << (cpu % 64);
    (void)SetThreadGroupAffinity(GetCurrentThread(), &g, 0);
}
static inline int sys__os__cpu_now(void) {
    PROCESSOR_NUMBER p;
    GetCurrentProcessorNumberEx(&p);
    return (int)p.Group * 64 + (int)p.Number;
}
static inline int sys__os__cpu_node(int cpu) { (void)cpu; return -1; }
/* The processors' relations, read whole: a core's and a package's CPUs as group masks, a cache's level and size. The
 * caller frees what it answers. */
static inline char* sys__os__windows_relations(LOGICAL_PROCESSOR_RELATIONSHIP which, DWORD* bytes) {
    *bytes = 0u;
    (void)GetLogicalProcessorInformationEx(which, 0, bytes);
    char* info = *bytes == 0u ? 0 : (char*)malloc(*bytes);
    if (info != 0 && !GetLogicalProcessorInformationEx(which, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)info, bytes)) {
        free(info);
        info = 0;
    }
    return info;
}
static inline void sys__os__cpu_core(int cpu, int* package, int* core) {
    *package = -1;
    *core = -1;
    DWORD bytes = 0u;
    char* info = sys__os__windows_relations(RelationAll, &bytes);
    if (info == 0) return;
    int cores = 0, packages = 0;
    for (DWORD at = 0u; at < bytes; ) {
        const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(info + at);
        if (e->Relationship == RelationProcessorCore || e->Relationship == RelationProcessorPackage) {
            const bool is_core = e->Relationship == RelationProcessorCore;
            for (WORD g = 0u; g < e->Processor.GroupCount; ++g) {
                const GROUP_AFFINITY* a = &e->Processor.GroupMask[g];
                if (cpu / 64 == (int)a->Group && ((a->Mask >> (cpu % 64)) & 1u) != 0u) {
                    if (is_core) *core = cores; else *package = packages;
                }
            }
            if (is_core) ++cores; else ++packages;
        }
        at += e->Size;
    }
    free(info);
}
/* Not read here: every CPU counts as idle, so every core is offered. */
static inline void sys__os__cpu_ticks(unsigned long long* busy, unsigned long long* total, int most) {
    (void)busy; (void)total; (void)most;
}
static inline long sys__os__cpu_l2_bytes(void) {
    DWORD bytes = 0u;
    char* info = sys__os__windows_relations(RelationCache, &bytes);
    long l2 = 0;
    for (DWORD at = 0u; info != 0 && at < bytes && l2 == 0; ) {
        const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(info + at);
        if (e->Relationship == RelationCache && e->Cache.Level == 2u) l2 = (long)e->Cache.CacheSize;
        at += e->Size;
    }
    free(info);
    return l2;
}

/* ⛳ OPENCL'S LOADER NAMES THE GPUS: every vendor's driver installs `OpenCL.dll` and its own part behind it, and a card
 *   is named by its OpenCL device name, the name the OpenCL families look for — an AMD card's is its `gfx`. NVIDIA's
 *   are left to its driver, which names them by compute capability. Loaded by name, so nothing links against it, and
 *   left loaded for the family that opens next. The calls' types are OpenCL's, spelled here without its headers. */
static inline uint32_t sys__os__cards(char* names, uint32_t name_bytes, uint32_t most) {
    typedef int32_t (*platforms_fn)(uint32_t, void**, uint32_t*);
    typedef int32_t (*devices_fn)(void*, uint64_t, uint32_t, void**, uint32_t*);
    typedef int32_t (*info_fn)(void*, uint32_t, size_t, void*, size_t*);
    const HMODULE cl = LoadLibraryExW(L"OpenCL.dll", 0, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (cl == 0) return 0u;
    const platforms_fn platforms = (platforms_fn)GetProcAddress(cl, "clGetPlatformIDs");
    const devices_fn   devices   = (devices_fn)GetProcAddress(cl, "clGetDeviceIDs");
    const info_fn      info      = (info_fn)GetProcAddress(cl, "clGetDeviceInfo");
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
            /* 0x1001 is CL_DEVICE_VENDOR_ID, 0x102B CL_DEVICE_NAME; 0x10DE is NVIDIA's PCI vendor */
            if (info(device[d], 0x1001u, sizeof vendor, &vendor, 0) != 0 || vendor == 0x10DEu) continue;
            memset(name, 0, sizeof name);
            if (info(device[d], 0x102Bu, sizeof name - 1u, name, 0) != 0 || name[0] == 0) continue;
            char* to = names + (size_t)(n++) * name_bytes;
            strncpy(to, name, name_bytes - 1u);
            to[name_bytes - 1u] = 0;
        }
    }
    return n;
}

#endif /* SILVANN__OPERATING_SYSTEMS_WINDOWS_OS_CUH */
