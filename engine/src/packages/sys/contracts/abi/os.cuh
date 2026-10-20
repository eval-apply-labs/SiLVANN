#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_ABI_OS_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_ABI_OS_CUH

/* ══ ⭐⭐⭐ WHAT THE TREE ASKS OF AN OPERATING SYSTEM — ONE CONTRACT, A FOLDER A SYSTEM ANSWERING IT ═══════════════════
 *
 * ⚖ 2026-10-18, the architect: *"we can do an os layer so that in the same way of silicon wrapper calls get routed to
 * the correct methods for the system"*. As a silicon family answers nn's doors with its own kernels, a system answers
 * these calls with its own: `operating_systems/linux/os.cuh` and `operating_systems/windows/os.cuh`, the one for the
 * system being built chosen by `operating_systems/os.cuh`. The engine, the packages and the families call `sys__os__*`,
 * never pthread, `dlopen` or `mmap` by name.
 * ⛳ CHOSEN WHEN THE TREE IS BUILT, NOT WHEN IT RUNS: a binary is built for one system, so the calls are the system's
 *   own, inline — the expert loader's locks and the runners' conditions cost what they cost before the layer.
 * ⛳ A SYSTEM'S FILE DEFINES THE TYPES AND THEIR STATIC INITIALIZERS, INCLUDES THIS FILE, THEN ANSWERS EVERY CALL. A
 *   call it leaves unanswered is a declaration with no body, which the compiler reports where it is used.
 * ⛳ AND HOW A LIBRARY OFFERS A FUNCTION BY NAME: `SYS__OS__EXPORT` before it — a family's entry point, the C calls a
 *   host finds in the engine's module by name.
 *   ▶ `docs/design/pending/windows_native.md`. */

/* ── threads ─────────────────────────────────────────────────────────────────────────────────────────────────────
 *   sys__os__thread     a running thread, joined or let go exactly once
 *   A thread runs `body(arg)`; its result is not kept. */
typedef void* (*sys__os__thread_body)(void* arg);
static inline bool sys__os__thread_start(sys__os__thread* thread, sys__os__thread_body body, void* arg);
static inline void sys__os__thread_join(sys__os__thread thread);
static inline void sys__os__thread_detach(sys__os__thread thread);

/* ── locks and conditions ────────────────────────────────────────────────────────────────────────────────────────
 *   sys__os__lock       SYS__OS__LOCK_INIT       taken and given by one thread, never twice by it
 *   sys__os__condition  SYS__OS__CONDITION_INIT  waited on with its lock taken; a wait may end with no wake */
static inline void sys__os__lock_init(sys__os__lock* lock);
static inline void sys__os__lock_take(sys__os__lock* lock);
static inline bool sys__os__lock_try(sys__os__lock* lock);
static inline void sys__os__lock_give(sys__os__lock* lock);
static inline void sys__os__condition_init(sys__os__condition* condition);
static inline void sys__os__condition_wait(sys__os__condition* condition, sys__os__lock* lock);
static inline void sys__os__condition_wake_one(sys__os__condition* condition);
static inline void sys__os__condition_wake_all(sys__os__condition* condition);

/* ── once ────────────────────────────────────────────────────────────────────────────────────────────────────────
 *   sys__os__once       SYS__OS__ONCE_INIT       `body` runs the first time any thread asks; the others wait for it */
typedef void (*sys__os__once_body)(void);
static inline void sys__os__once_run(sys__os__once* once, sys__os__once_body body);

/* ── a thread's key ──────────────────────────────────────────────────────────────────────────────────────────────
 *   sys__os__key        a word each thread holds of its own; `at_exit(value)` runs as a thread that set one ends */
typedef void (*sys__os__key_exit)(void* value);
static inline bool sys__os__key_make(sys__os__key* key, sys__os__key_exit at_exit);
static inline void sys__os__key_set(sys__os__key key, void* value);

/* ── libraries, and paths ────────────────────────────────────────────────────────────────────────────────────────
 *   sys__os__library    a library loaded into the process, 0 when it would not load
 *   SYS__OS__LIBRARY_SUFFIX  a library's file ending on this system; SYS__OS__NVIDIA_DRIVER  the NVIDIA driver's library
 *   A path is UTF-8, its separator `/` (Windows takes it too). `library_why` is the last refusal in words, or 0;
 *   `library_path_of` the file of the library an address is in, or 0 — the string the system's, kept until the next
 *   call on this thread. */
static inline sys__os__library sys__os__library_open(const char* path);
static inline void* sys__os__library_find(sys__os__library library, const char* name);
static inline const char* sys__os__library_why(void);
static inline const char* sys__os__library_path_of(const void* address);
static inline bool sys__os__path_exists(const char* path);
static inline bool sys__os__path_is_directory(const char* path);
/*   `library_base_of` the address the library an address is in was loaded at, or 0. */
static inline uintptr_t sys__os__library_base_of(const void* address);

/* ── the clock, a sleep, a yield ─────────────────────────────────────────────────────────────────────────────────
 *   A time is C's `struct timespec`. `clock` reads a clock that never steps back, from a zero of the system's own;
 *   `sleep` gives the processor up for at least `how_long`; `yield` lets another thread run first. */
static inline void sys__os__clock(struct timespec* now);
static inline void sys__os__sleep(const struct timespec* how_long);
static inline void sys__os__yield(void);

/* ── files ───────────────────────────────────────────────────────────────────────────────────────────────────────
 *   sys__os__file       a file open for reading: an int, never negative — a descriptor, or a handle, which Windows
 *                       keeps within 32 bits — and negative when it would not open
 *   `direct` asks that reads skip the system's cache; each read is then whole pages, at a whole page, into memory on a
 *   whole page. A system or a file system that will not refuses that open, and the file still opens without it.
 *   `read_at` answers the bytes it read, fewer than asked at the end of the file; 0 or less, none. */
static inline sys__os__file sys__os__file_open(const char* path, bool direct);
static inline void sys__os__file_close(sys__os__file file);
static inline int64_t sys__os__file_read_at(sys__os__file file, void* to, size_t bytes, uint64_t offset);

/* ── memory ──────────────────────────────────────────────────────────────────────────────────────────────────────
 *   `memory_map`        `bytes` of zeroed pages to read and write with nothing set aside behind them — a range larger
 *                       than the machine maps, and fails only where it is touched — or 0; `memory_unmap` gives back
 *                       what one map took, whole
 *   `memory_guard`      those pages untouchable: a touch is a fault
 *   `memory_map_file`   `bytes` of a file at `offset`, read only, AS the mapped pages at `at`; false when it will not
 *   `memory_resident`   a byte a page into `each`, bit 0 set when the page is in memory; false when it cannot say
 *   `memory_will_need`  those pages read ahead
 *   `memory_interleave` / `memory_bind`  a map's pages from the nodes in `nodes` (a bit a node, `most` bits) in turn,
 *                       or from those alone
 *   `memory_aligned`    `bytes` of the heap at a multiple of `alignment`, or 0, given back by `memory_aligned_free`
 *                       and by nothing else */
static inline void* sys__os__memory_map(size_t bytes);
static inline void sys__os__memory_unmap(void* base, size_t bytes);
static inline bool sys__os__memory_guard(void* at, size_t bytes);
static inline bool sys__os__memory_map_file(void* at, size_t bytes, sys__os__file file, uint64_t offset);
static inline bool sys__os__memory_resident(const void* at, size_t bytes, unsigned char* each);
static inline void sys__os__memory_will_need(void* at, size_t bytes);
static inline bool sys__os__memory_interleave(void* base, size_t bytes, const unsigned long* nodes, unsigned long most);
static inline bool sys__os__memory_bind(void* base, size_t bytes, const unsigned long* nodes, unsigned long most);
static inline void* sys__os__memory_aligned(size_t alignment, size_t bytes);
static inline void sys__os__memory_aligned_free(void* at);

/* ── the processors ──────────────────────────────────────────────────────────────────────────────────────────────
 *   sys__os__cpus       a set of CPUs, each a number below SYS__OS__CPUS_MOST
 *   `cpus_of_thread` / `cpus_to_thread` the CPUs the calling thread may run on; `cpu_pin_thread` it on one alone;
 *   `cpu_now` the CPU it is on, or -1. A CPU's NUMA node, -1 when the system will not say; its package and its
 *   physical core, -1 each when it will not say; every CPU's busy and total ticks so far, by number below `most`, left
 *   as they were when it will not say; the L2 cache's bytes, 0 or less when it will not say. */
static inline void sys__os__cpus_clear(sys__os__cpus* set);
static inline void sys__os__cpus_add(sys__os__cpus* set, int cpu);
static inline bool sys__os__cpus_has(const sys__os__cpus* set, int cpu);
static inline int  sys__os__cpus_count(const sys__os__cpus* set);
static inline bool sys__os__cpus_of_thread(sys__os__cpus* set);
static inline bool sys__os__cpus_to_thread(const sys__os__cpus* set);
static inline void sys__os__cpu_pin_thread(int cpu);
static inline int  sys__os__cpu_now(void);
static inline int  sys__os__cpu_node(int cpu);
static inline void sys__os__cpu_core(int cpu, int* package, int* core);
static inline void sys__os__cpu_ticks(unsigned long long* busy, unsigned long long* total, int most);
static inline long sys__os__cpu_l2_bytes(void);

/* ── the cards the system names ──────────────────────────────────────────────────────────────────────────────────
 *   `cards` writes each GPU's name the system itself can give, as the families include cards, `name_bytes` a name, up
 *   to `most`, and answers how many: Linux's from the AMD kernel driver's topology (`gfx906`); Windows' from OpenCL's
 *   loader, every GPU but NVIDIA's (which its own driver names) by its OpenCL device name — an AMD card's is its `gfx`.
 *   In the order the system lists them, the same on every call. */
static inline uint32_t sys__os__cards(char* names, uint32_t name_bytes, uint32_t most);

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_ABI_OS_CUH */
