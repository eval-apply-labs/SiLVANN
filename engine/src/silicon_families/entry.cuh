#ifndef SILVANN__SILICON_FAMILIES_ENTRY_CUH
#define SILVANN__SILICON_FAMILIES_ENTRY_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../packages/sys/contracts/abi/silicon_family.cuh"   /* the table this answers */
#include "../packages/sys/contracts/abi/gpu.cuh"              /* sys's doors, the first table it carries */
#include "../packages/manifest__header.cuh"                    /* the packages the door tables are gathered over */
#include "roster.cuh"                                          /* the family's number */

/* ══ ⭐⭐⭐ THE ENTRY POINT EVERY FAMILY EXPORTS — WRITTEN ONCE, FOR EVERY FAMILY ════════════════════════
 *
 * A family's translation unit (`<family>/unit.cuh`) names which one it is — `SYS__SILICON_FAMILY__THIS` as
 * a bare token, its folder's name — and includes its answers to every package before this file. Everything
 * below is then the same for every family: one door table per package, filled from that package's list,
 * gathered into the family's table, and handed out by the one symbol the program looks for.
 * ⛳ `extern "C"`, so the symbol is `silvann_silicon_family_get` in every language a family can be written
 * in, and default visibility, so it survives the build hiding everything else — which it does: a
 * family exports this and nothing more. */
#ifndef SYS__SILICON_FAMILY__THIS
#error "a family says which it is: define SYS__SILICON_FAMILY__THIS (its folder's name) before this file"
#endif

/* ⭐ EACH PACKAGE'S DOOR TABLE, FILLED FROM ITS OWN LIST — gathered over the package roster, so a package
 * gets a table by being listed and no line here names one. A door named in a list and not defined by the
 * family is a compile error here, not a null found at run time. */
#define SILICON_FAMILIES__ZZPRIVATE_DOOR_FILL(PKG, RET, NAME, FN, PARAMS)  FN,
#define SILICON_FAMILIES__ZZPRIVATE_DOOR_TABLE(ID, NAME, ...)                                          \
    static NAME##__doors NAME##__zzprivate_doors = {                                                \
        (uint32_t)sizeof(NAME##__doors), NAME##__CONTRACT__DOORS(SILICON_FAMILIES__ZZPRIVATE_DOOR_FILL, NAME) };
PACKAGE_LIST(SILICON_FAMILIES__ZZPRIVATE_DOOR_TABLE)
#undef SILICON_FAMILIES__ZZPRIVATE_DOOR_TABLE
#undef SILICON_FAMILIES__ZZPRIVATE_DOOR_FILL

#define SILICON_FAMILIES__ZZPRIVATE_DOOR_SLOT(ID, NAME, ...)  (const void*)&NAME##__zzprivate_doors,
static const void* const silicon_families__zzprivate_package_doors[] = {
    PACKAGE_LIST(SILICON_FAMILIES__ZZPRIVATE_DOOR_SLOT)
};
#undef SILICON_FAMILIES__ZZPRIVATE_DOOR_SLOT

#define SILICON_FAMILIES__ZZPRIVATE_TEXT2(T)  #T
#define SILICON_FAMILIES__ZZPRIVATE_TEXT(T)   SILICON_FAMILIES__ZZPRIVATE_TEXT2(T)
static const sys__silicon_family silicon_families__zzprivate_this = {
    SYS__SILICON_FAMILY__ABI_VERSION,
    (uint32_t)sizeof(sys__silicon_family),
    SYS__SILICON_FAMILY__OF_THIS(family_id),
    (uint32_t)package_count,
    SILICON_FAMILIES__ZZPRIVATE_TEXT(SYS__SILICON_FAMILY__THIS),
    silicon_families__zzprivate_package_doors,
};
#undef SILICON_FAMILIES__ZZPRIVATE_TEXT
#undef SILICON_FAMILIES__ZZPRIVATE_TEXT2

/* The table, when asked for the version it speaks; nothing otherwise. ⛳ A version this family does not
 * speak is refused rather than answered with its own, because a program reading a table of another shape
 * would call through whichever pointer happened to sit where it expected one. */
extern "C" {
__attribute__((visibility("default")))
const sys__silicon_family* silvann_silicon_family_get(uint32_t abi_version) {
#ifdef SYS__SILICON_FAMILY__OVERRIDE_DOORS
    /* ⛳ A FAMILY WITH `overrides/` PUTS ITS FASTER DOORS INTO THE TABLES HERE, once, before anyone reads them:
     *   the tables are filled from each package's list first, so a door the family does not override is the
     *   package's own. */
    static bool overridden = false;
    if (!overridden) { SYS__SILICON_FAMILY__OVERRIDE_DOORS(silicon_families__zzprivate_package_doors); overridden = true; }
#endif
    return (abi_version == SYS__SILICON_FAMILY__ABI_VERSION) ? &silicon_families__zzprivate_this : 0;
}
}

#endif /* SILVANN__SILICON_FAMILIES_ENTRY_CUH */
