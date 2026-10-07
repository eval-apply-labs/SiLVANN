#ifndef SILVANN__PACKAGES_SYS_CPU_SILICON_SILICON_FAMILY_CUH
#define SILVANN__PACKAGES_SYS_CPU_SILICON_SILICON_FAMILY_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../../contracts/abi/silicon_family.cuh"   /* what a family is */
#include "../../contracts/abi/gpu.cuh"       /* sys's doors, which every family carries first */
#include "../../../manifest__header.cuh"     /* sys's package id, where its doors sit in a family */


/* ══ ⭐⭐⭐ WHICH SILICON, NAMED BY THE CALLER ═══════════════════════════════════════════════════════════
 *
 * ⚖ *"we build for all architectures and depending on which one it is we choose the correct fill
 * method."* The program keeps one family per id it was given, and every operation names the family it is
 * for. So two machines on two different cards can live in one process — which is what one worker per GPU
 * needs — and nothing decides for a caller which silicon it meant.
 * ⛳ ONE PER FAMILY, NOT ONE PER DEVICE. Which device of a family an operation reaches is the calling
 * thread's bound one: `sys__silicon__bind_device`, below.
 * ⛳ `sys` ASKS A FAMILY THROUGH ITS OWN DOOR TABLE, as every package does — ⚖ *"card is actually more
 * correctly described as the sys doors"*. A family nobody offered has an empty slot, and every door
 * refuses on it rather than reaching another family's.
 * ⛔ THE COST IS ONE INDIRECT CALL PER `hipMalloc`-SCALE OPERATION, which is not a cost.
 */

/* ══ A FAMILY, LOADED OR COMPILED IN ═══════════════════════════════════════════════════════════════════
 * ⚖ *"the gpu side becomes a separate binary that is loaded at the start of the cpu session."* Whoever
 * loads the binary — the wheel, from Python, which is what keeps the environment out of the program —
 * hands its table here. The host's is offered the same way, compiled in (`silicon_families/host/`).
 * ▶ `contracts/abi/silicon_family.cuh` for what a family is. */
static const sys__silicon_family* sys__silicon_family__zzprivate_table[SYS__SILICON_FAMILY__SLOTS];
static sys__silicon_family__id     sys__silicon_family__zzprivate_first_offered = SYS__SILICON_FAMILY__NONE;

/* A package's door table in a family, untyped, or nothing when the family carries none for it. */
static inline const void* sys__silicon__zzprivate_doors_in(const sys__silicon_family* family, uint64_t package_id) {
    if (family == 0 || family->package_doors == 0 || package_id >= (uint64_t)family->package_count) return 0;
    return family->package_doors[package_id];
}

/* `sys`'s door table on a family, or nothing when the family has none this program can read. */
static inline const sys__doors* sys__silicon__zzprivate_sys_doors(const sys__silicon_family* family) {
    const sys__doors* doors = (const sys__doors*)sys__silicon__zzprivate_doors_in(family, sys_pkg_id);
    return (doors != 0 && doors->size >= (uint32_t)sizeof(sys__doors)) ? doors : 0;
}

/* ⛔ A TABLE OF ANOTHER VERSION, OR SMALLER THAN THIS PROGRAM READS, IS REFUSED — its pointers would not be
 * where this code looks for them. So is one without `sys`'s doors, or without the three that say which
 * device a worker is on. A second family for the same id is refused rather than replacing the first: two
 * claiming one id is a build that loaded something twice, and quietly keeping either would hide it. */
static inline bool sys__silicon_family__offer(const sys__silicon_family* family) {
    if (family == 0 || family->abi_version != SYS__SILICON_FAMILY__ABI_VERSION) return false;
    if (family->size < (uint32_t)sizeof(sys__silicon_family)) return false;
    if (family->id >= (sys__silicon_family__id)SYS__SILICON_FAMILY__SLOTS || family->name == 0) return false;
    const sys__doors* doors = sys__silicon__zzprivate_sys_doors(family);
    if (doors == 0 || doors->device_count == 0 || doors->bind_device == 0 || doors->bound_device == 0) return false;
    if (sys__silicon_family__zzprivate_table[family->id] != 0) return false;
    sys__silicon_family__zzprivate_table[family->id] = family;
    if (sys__silicon_family__zzprivate_first_offered == SYS__SILICON_FAMILY__NONE)
        sys__silicon_family__zzprivate_first_offered = family->id;
    return true;
}

/* The family offered for this id, or nothing. */
static inline const sys__silicon_family* sys__silicon_family__of(sys__silicon_family__id family) {
    return (family < (sys__silicon_family__id)SYS__SILICON_FAMILY__SLOTS) ? sys__silicon_family__zzprivate_table[family] : 0;
}

/* Whether a family was offered for this id. */
static inline bool sys__silicon_family__has(sys__silicon_family__id family) {
    return sys__silicon_family__of(family) != 0;
}

/* The family offered under this name, or `SYS__SILICON_FAMILY__NONE`.
 * ⛳ BY NAME, because what asks is a boot, and a name is what a boot is told — and the name is the one the
 * family's own table gives, so sys keeps no list of them. It refuses a family nobody offered rather than
 * falling back: a caller that asked for one family and silently got another would be told it had what it
 * asked for. */
static inline sys__silicon_family__id sys__silicon_family__named(const char* name) {
    sys__silicon_family__id i;
    if (name == 0) return SYS__SILICON_FAMILY__NONE;
    for (i = 0u; i < (sys__silicon_family__id)SYS__SILICON_FAMILY__SLOTS; ++i) {
        if (sys__silicon_family__zzprivate_table[i] == 0) continue;
        const char *a = sys__silicon_family__zzprivate_table[i]->name, *b = name;
        while (*a && *a == *b) { ++a; ++b; }
        if (*a == 0 && *b == 0) return i;
    }
    return SYS__SILICON_FAMILY__NONE;
}

/* The name of the family offered for this number; null when none was. */
static inline const char* sys__silicon_family__spelling(sys__silicon_family__id family) {
    const sys__silicon_family* p = sys__silicon_family__of(family);
    return p ? p->name : 0;
}

/* The first family offered, which is what a boot that names none stands up on.
 * ⛳ ON A BUILD WITH ONE THIS IS THE ONLY ANSWER THERE IS; on a build with several, a caller that
 * cares names one. */
static inline sys__silicon_family__id sys__silicon_family__first(void) {
    return sys__silicon_family__zzprivate_first_offered;
}

/* ── sys's DOORS, EACH NAMING ITS FAMILY ───────────────────────────────────────────────────────────── */
#define SYS__SILICON__ZZPRIVATE_DOOR(FIELD, FAIL)                                                        \
    const sys__doors* d = sys__silicon__zzprivate_sys_doors(sys__silicon_family__of(family));          \
    if (d == 0 || d->FIELD == 0) return FAIL;

static inline bool sys__gpu__memory_allocate(sys__silicon_family__id family, void** at, size_t bytes)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_allocate, false)          return d->memory_allocate(at, bytes); }
static inline void sys__gpu__memory_free(sys__silicon_family__id family, void* at)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_free, )                   d->memory_free(at); }
static inline bool sys__gpu__memory_zerofill(sys__silicon_family__id family, void* at, size_t bytes)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_zerofill, false)          return d->memory_zerofill(at, bytes); }
static inline bool sys__gpu__memory_read(sys__silicon_family__id family, void* to, const void* from, size_t bytes)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_read, false)              return d->memory_read(to, from, bytes); }
static inline bool sys__gpu__memory_write(sys__silicon_family__id family, void* to, const void* from, size_t bytes)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_write, false)             return d->memory_write(to, from, bytes); }
static inline bool sys__gpu__compute_completed(sys__silicon_family__id family)
    { SYS__SILICON__ZZPRIVATE_DOOR(compute_completed, false)        return d->compute_completed(); }
static inline bool sys__gpu__side_open(sys__silicon_family__id family, void** channel)
    { SYS__SILICON__ZZPRIVATE_DOOR(side_open, false)                return d->side_open(channel); }
static inline void sys__gpu__side_close(sys__silicon_family__id family, void* channel)
    { SYS__SILICON__ZZPRIVATE_DOOR(side_close, )                    d->side_close(channel); }
static inline bool sys__gpu__side_memory_read(sys__silicon_family__id family, void* to, const void* from, size_t bytes,
                                              void* channel)
    { SYS__SILICON__ZZPRIVATE_DOOR(side_memory_read, false)         return d->side_memory_read(to, from, bytes, channel); }
static inline bool sys__gpu__side_memory_write(sys__silicon_family__id family, void* to, const void* from, size_t bytes,
                                               void* channel)
    { SYS__SILICON__ZZPRIVATE_DOOR(side_memory_write, false)        return d->side_memory_write(to, from, bytes, channel); }
static inline void sys__gpu__memory_free_host_ram(sys__silicon_family__id family, void* at)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_free_host_ram, )          d->memory_free_host_ram(at); }
static inline bool sys__gpu__memory_allocate_host_ram_mapped(sys__silicon_family__id family, void** at, void** card_at, size_t bytes)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_allocate_host_ram_mapped, false) return d->memory_allocate_host_ram_mapped(at, card_at, bytes); }
static inline bool sys__gpu__memory_register_host(sys__silicon_family__id family, void* at, size_t bytes, void** card_at)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_register_host, false)    return d->memory_register_host(at, bytes, card_at); }
static inline void sys__gpu__memory_unregister_host(sys__silicon_family__id family, void* at)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_unregister_host, )        d->memory_unregister_host(at); }
static inline bool sys__gpu__file_open(sys__silicon_family__id family, const char* path, uint64_t* handle)
    { SYS__SILICON__ZZPRIVATE_DOOR(file_open, false)                return d->file_open(path, handle); }
static inline void sys__gpu__file_close(sys__silicon_family__id family, uint64_t handle)
    { SYS__SILICON__ZZPRIVATE_DOOR(file_close, )                    d->file_close(handle); }
static inline bool sys__gpu__file_read(sys__silicon_family__id family, uint64_t handle, uint64_t offset, uint64_t bytes, void* to)
    { SYS__SILICON__ZZPRIVATE_DOOR(file_read, false)                return d->file_read(handle, offset, bytes, to); }
static inline bool sys__gpu__memory_map_file(sys__silicon_family__id family, void* at, size_t bytes, uint64_t handle, uint64_t offset)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_map_file, false)          return d->memory_map_file(at, bytes, handle, offset); }
static inline bool sys__gpu__memory_prefetch(sys__silicon_family__id family, const void* at, size_t bytes)
    { SYS__SILICON__ZZPRIVATE_DOOR(memory_prefetch, true)           return d->memory_prefetch(at, bytes); }

/* How many devices of a family this machine has, as its family counts them; zero with no family. */
static inline uint32_t sys__silicon__device_count(sys__silicon_family__id family)
    { SYS__SILICON__ZZPRIVATE_DOOR(device_count, 0u)                return d->device_count(); }
/* ⭐ WHAT A WORKER DOES FIRST: make device `index` of `family` the calling thread's. Every operation it
 * then makes on that family goes to that device, and no other worker's. */
static inline bool sys__silicon__bind_device(sys__silicon_family__id family, uint32_t index)
    { SYS__SILICON__ZZPRIVATE_DOOR(bind_device, false)              return d->bind_device(index) == 1u; }
/* The device of `family` the calling thread is on, or `0xFFFFFFFF` with no family or no answer. */
static inline uint32_t sys__silicon__bound_device(sys__silicon_family__id family)
    { SYS__SILICON__ZZPRIVATE_DOOR(bound_device, 0xFFFFFFFFu)       return d->bound_device(); }
#undef SYS__SILICON__ZZPRIVATE_DOOR

/* ⭐ A PACKAGE'S DOOR TABLE ON A FAMILY — or nothing, with no family for it or no table for that package.
 * Untyped here, because `sys` names no other package's doors; the package reads it as its own type, and
 * checks the table's first word — its size — before trusting any door past it. */
static inline const void* sys__silicon__package_doors(sys__silicon_family__id family, uint64_t package_id) {
    return sys__silicon__zzprivate_doors_in(sys__silicon_family__of(family), package_id);
}

#endif /* SILVANN__PACKAGES_SYS_CPU_SILICON_SILICON_FAMILY_CUH */
