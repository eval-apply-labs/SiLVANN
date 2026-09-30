#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_ABI_GPU_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_ABI_GPU_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/* ══ ⭐⭐⭐ sys's DOORS — WHAT sys ASKS OF EVERY FAMILY ══════════════════════════════════════════════════
 *
 * ⚖ *"i do wonder if card is actually more correctly described as the sys doors"*. Every package lists the
 * functions it needs a family to provide, and this is `sys`'s list: memory, copies, waiting for the work,
 * the side channel, and which device the calling thread is on. The program keeps one family per id and
 * calls through the table the one a machine names. ▶ `cpu/silicon/silicon_family.cuh` for the program's side.
 *
 * ⭐⭐ THIS IS THE MANIFEST FOR A SILICON PORTER. ⚖ *"let's keep it there as a manifest for silicon
 * porters"*. `cpu.cuh` beside it is what a program outside the language may CALL; this is what a family
 * must IMPLEMENT — every package's `contracts/abi/gpu.cuh` together, against the version and table in
 * `silicon_family.cuh` — and the families that do are the tree's `silicon_families/`. Adding a vendor, or a
 * family for a new ROCm or CUDA generation, is a folder there that fills these lists, and nothing else.
 *
 * ⭐⭐ IT NEEDS NO TYPE SURGERY BECAUSE OF A PROPERTY OF THE SEAM: `MEASURED` — of the operations, **not
 * one names a vendor type.** Every parameter is `void*`, `const void*`, `size_t`, `uint32_t` or `bool`, so a
 * table of pointers over them compiles on either side of the boundary; a single `hipStream_t` in a
 * signature would make this a redesign instead.
 * ⭐⭐⭐ AND THAT IS WHAT LETS A HOST TRANSLATION UNIT EXIST AT ALL. `#define __device__` rewrites HIP's own
 * runtime headers — `MEASURED`, 17 of 20 errors inside `amd_device_functions.h` — so a host translation
 * unit cannot include a vendor. Behind this table it never needs to: the implementations live in their
 * own translation unit and arrive as pointers.
 * ⛳ ONE TABLE PER FAMILY, NOT ONE PER DEVICE: which device an operation acts on is the one the calling
 * thread bound with `bind_device`, which answers for the calling thread only, as both vendors' runtimes do.
 * ⚠ `bool` IS NOT YET THE BOUNDARY'S TYPE. A C `_Bool` and a C++ `bool` agree on every compiler this tree
 * builds with, but the boundary's rule is a fixed-width status, and it is owed on the day a door crosses
 * a binary rather than a translation unit. ▶ `docs/design/pending/evaluator_and_providers.md` §②.
 *
 * ── THE ROW ──────────────────────────────────────────────────────────────────────────────────────────
 * Every package's door list has this shape: `X(PKG, RETURN, NAME, FUNCTION, (PARAMETERS))`. The FUNCTION
 * is what the family defines for the door. `sys`'s are each family's own, `<family>__<name>` — `amd_rocm6_wave64__memory_allocate` —
 * named here through `SYS__SILICON_FAMILY__OF_THIS`, which resolves in the translation unit that builds a
 * family — it defines `SYS__SILICON_FAMILY__THIS` first. */
#define SYS__SILICON_FAMILY__ZZPRIVATE_NAME2(FAMILY, WHAT)  FAMILY##__##WHAT
#define SYS__SILICON_FAMILY__ZZPRIVATE_NAME(FAMILY, WHAT)   SYS__SILICON_FAMILY__ZZPRIVATE_NAME2(FAMILY, WHAT)
#define SYS__SILICON_FAMILY__OF_THIS(WHAT) \
    SYS__SILICON_FAMILY__ZZPRIVATE_NAME(SYS__SILICON_FAMILY__THIS, WHAT)

#define sys__CONTRACT__DOORS(X, PKG) \
    X(PKG, bool,     memory_allocate,                 SYS__SILICON_FAMILY__OF_THIS(memory_allocate),                 (void** at, size_t bytes)) \
    X(PKG, void,     memory_free,                     SYS__SILICON_FAMILY__OF_THIS(memory_free),                     (void* at)) \
    X(PKG, bool,     memory_zerofill,                 SYS__SILICON_FAMILY__OF_THIS(memory_zerofill),                 (void* at, size_t bytes)) \
    X(PKG, bool,     memory_read,                     SYS__SILICON_FAMILY__OF_THIS(memory_read),                     (void* to, const void* from, size_t bytes)) \
    X(PKG, bool,     memory_write,                    SYS__SILICON_FAMILY__OF_THIS(memory_write),                    (void* to, const void* from, size_t bytes)) \
    X(PKG, bool,     compute_completed,               SYS__SILICON_FAMILY__OF_THIS(compute_completed),               (void)) \
    X(PKG, bool,     side_open,                       SYS__SILICON_FAMILY__OF_THIS(side_open),                       (void** channel)) \
    X(PKG, void,     side_close,                      SYS__SILICON_FAMILY__OF_THIS(side_close),                      (void* channel)) \
    X(PKG, bool,     side_memory_read,                SYS__SILICON_FAMILY__OF_THIS(side_memory_read),                (void* to, const void* from, size_t bytes, void* channel)) \
    X(PKG, bool,     side_memory_write,               SYS__SILICON_FAMILY__OF_THIS(side_memory_write),               (void* to, const void* from, size_t bytes, void* channel)) \
    X(PKG, bool,     memory_allocate_host_ram_mapped, SYS__SILICON_FAMILY__OF_THIS(memory_allocate_host_ram_mapped), (void** at, void** card_at, size_t bytes)) \
    X(PKG, void,     memory_free_host_ram,            SYS__SILICON_FAMILY__OF_THIS(memory_free_host_ram),            (void* at)) \
    X(PKG, uint32_t, device_count,                    SYS__SILICON_FAMILY__OF_THIS(device_count),                    (void)) \
    X(PKG, uint32_t, bind_device,                     SYS__SILICON_FAMILY__OF_THIS(bind_device),                     (uint32_t index)) \
    X(PKG, uint32_t, bound_device,                    SYS__SILICON_FAMILY__OF_THIS(bound_device),                    (void)) \
    X(PKG, bool,     memory_register_host,            SYS__SILICON_FAMILY__OF_THIS(memory_register_host),            (void* at, size_t bytes, void** card_at)) \
    X(PKG, void,     memory_unregister_host,          SYS__SILICON_FAMILY__OF_THIS(memory_unregister_host),          (void* at)) \
    X(PKG, bool,     file_open,                       SYS__SILICON_FAMILY__OF_THIS(file_open),                       (const char* path, uint64_t* handle)) \
    X(PKG, void,     file_close,                      SYS__SILICON_FAMILY__OF_THIS(file_close),                      (uint64_t handle)) \
    X(PKG, bool,     file_read,                       SYS__SILICON_FAMILY__OF_THIS(file_read),                       (uint64_t handle, uint64_t offset, uint64_t bytes, void* to)) \
    X(PKG, bool,     memory_map_file,                 SYS__SILICON_FAMILY__OF_THIS(memory_map_file),                 (void* at, size_t bytes, uint64_t handle, uint64_t offset)) \
    X(PKG, bool,     memory_prefetch,                 SYS__SILICON_FAMILY__OF_THIS(memory_prefetch),                 (const void* at, size_t bytes))

/* What each door does, in the order of the list:
 *     memory_allocate / memory_free            memory on the card
 *     memory_zerofill                          zero a range of it
 *     memory_read / memory_write               card -> RAM / RAM -> card, and wait
 *     compute_completed                        wait for all work queued on the card
 *     side_open / side_close                   a second queue whose copies do not wait behind compute —
 *     side_memory_read / side_memory_write     how the host talks to a machine that is still working;
 *                                              each copy still waits for itself
 *     memory_allocate_host_ram_mapped          RAM the card can also WRITE: page-locked, so copies through it
 *                                              run at the bus's rate, and mapped, so a kernel can store into
 *                                              it — `*at` is where the program reads it, `*card_at` where a
 *                                              kernel writes it. ~4.7 us for a kernel's answer to land against
 *                                              ~27 us to launch and copy it back (`MEASURED`,
 *                                              `measurements/2026-09-25_S108_readback_split.md`). The only
 *                                              kind of pinned RAM: plain pinned measured the same at every size
 *                                              (`measurements/2026-09-25_S108_pinned_vs_mapped.md`)
 *     memory_free_host_ram                     gives it back
 *     device_count                             how many devices of this family the machine has; zero is how a
 *                                              worker learns there is nothing to take over
 *     bind_device / bound_device               make device `index` the CALLING THREAD's, answering 1 or 0 /
 *                                              which device the calling thread is on
 *     memory_register_host                     make RAM the program already holds writable by the bound card:
 *                                              page-locked and mapped, `*card_at` where a kernel writes it.
 *                                              ⛔ A PLAIN ALLOCATION IS NOT: `MEASURED` on the MI50, a
 *                                              kernel storing into `malloc`ed memory stops the process with
 *                                              "Memory access fault … Page not present"; the same memory
 *                                              registered is written, and the card sees it at the same address.
 *                                              A family that cannot answers false, and a caller copies instead
 *     memory_unregister_host                   gives the registration back; the memory stays the program's
 *     file_open / file_close                   a file, read-only, as a handle (never zero) — ⚖ *"sys door"*,
 *                                              for the expert LRU's disk tier
 *     file_read                                `bytes` of it at `offset` into THIS family's memory at `to`: straight
 *                                              in where the program shares that memory, through a bounce buffer
 *                                              and `memory_write` on a card. ▶ `sys/cpu/silicon/file.cuh`
 *     memory_map_file                          `bytes` of the file at `offset` AS this family's memory at `at`,
 *                                              read-only: a page is read from the disk when first touched and kept
 *                                              for as long as the system keeps a file's pages, so memory smaller
 *                                              than the file holds what is used and the disk the rest. Page-aligned
 *                                              `at` and `offset`. Only where the program's memory is the machine's
 *                                              own; a card answers false.
 *     memory_prefetch                          whether every page of the range is in memory now — and when one is not,
 *                                              its reading STARTED, not waited for, so a caller can compute with what
 *                                              is here while the rest arrives. Memory that is never paged out (a
 *                                              card's, an allocation) answers true. */

#define SYS__DOORS__ZZPRIVATE_FIELD(PKG, RET, NAME, FN, PARAMS)  RET (*NAME) PARAMS;
typedef struct sys__doors {
    uint32_t size;                                   /* sizeof this table as the family built it */
    sys__CONTRACT__DOORS(SYS__DOORS__ZZPRIVATE_FIELD, sys)
} sys__doors;
#undef SYS__DOORS__ZZPRIVATE_FIELD

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_ABI_GPU_CUH */
