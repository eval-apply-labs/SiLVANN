#ifndef SILVANN__SILICON_FAMILIES_ROSTER_CUH
#define SILVANN__SILICON_FAMILIES_ROSTER_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../packages/sys/contracts/abi/silicon_family.cuh"   /* what a family is, and the bound on its number */
#include "host/includes.cuh"                                  /* each family's `includes`, in roster order */
#include "x86_avx2/includes.cuh"
#include "khronos_opencl2/includes.cuh"
#include "amd_rocm6_wave64/includes.cuh"
#include "nvidia_cuda12/includes.cuh"
#include "choose.cuh"                                         /* how a card is given to them */

/* ══ ⭐⭐⭐ THE SILICON FAMILIES — WHAT ANSWERS THE PACKAGES ════════════════════════════════════════════
 *
 * ⚖ *"archs become their own packages … it implements the requests … a provider can implement their code
 * autonomously, by looking at the surface requested by the packages he wants to use"*. The packages ASK —
 * each lists, in its `contracts/abi/gpu.cuh`, what it needs a card to do — and this tier ANSWERS: one folder
 * per silicon family, each filling every package's list with its own functions and building to one binary.
 * The host is one of them, compiled into the program rather than loaded.
 * ▶ `add_a_family.md` for what a new family is made of, and how to add one.
 *
 * One row per family: its number, the name its constants are spelled with, its folder — which is also the
 * name a boot asks for it by and the prefix of every function it defines — and its vendor.
 * ⛔ A NUMBER IS NEVER REUSED AND NEVER MOVES, because it is what a machine records about the silicon it
 *   stood up on; the numbers rise down the list, and the check below stops the build if not.
 * ⚖ *"opencl goes to 100, 906 goes to 200 and nvidia to 300"* — AND THE GAPS ARE THE POINT: a family can
 *   be placed between two others without moving either, because a number is a preference and a name, not
 *   a position. The chooser walks the list and answers the row's number.
 * ⭐⭐ AND THE NUMBER IS THE PREFERENCE: A CARD IS OFFERED TO THE HIGHEST FIRST. ⚖ *"if we do by number
 *   (which honestly is valid here) arch get done from the highest family list to the smallest"*. Numbers
 *   only grow, so a higher one is a family added later — a newer runtime, or a build that assumes more of
 *   the silicon — and a card several families include goes to the newest that works, with no list to
 *   reorder when one is added. ⛔ SO A NEW ROW MUST NEVER BE A WORSE CHOICE than an older row that takes
 *   the same cards; if one ever is, the answer is another row, never a renumbering. ▶ `choose.cuh`.
 * ⛳ THE VENDOR IS THERE FOR ONE RULE: two families of one vendor are never opened in one process. ▶ the
 *   same file for why. */
#define SILICON_FAMILIES__LIST(X)                                                                          \
    X(0u,   HOST,             host,             host)                                                    \
    X(50u,  X86_AVX2,         x86_avx2,         x86)                                                     \
    X(100u, KHRONOS_OPENCL2,  khronos_opencl2,  khronos)                                                 \
    X(200u, AMD_ROCM6_WAVE64, amd_rocm6_wave64, amd)                                                     \
    X(300u, NVIDIA_CUDA12,    nvidia_cuda12,    nvidia)

#define SILICON_FAMILIES__ZZPRIVATE_ROW(ID, NAME, FOLDER, VENDOR)    SILICON_FAMILIES__##NAME = (ID),
#define SILICON_FAMILIES__ZZPRIVATE_TALLY(ID, NAME, FOLDER, VENDOR)  + 1u
enum {
    SILICON_FAMILIES__LIST(SILICON_FAMILIES__ZZPRIVATE_ROW)
    SILICON_FAMILIES__COUNT = 0u SILICON_FAMILIES__LIST(SILICON_FAMILIES__ZZPRIVATE_TALLY)
};
#undef SILICON_FAMILIES__ZZPRIVATE_ROW
#undef SILICON_FAMILIES__ZZPRIVATE_TALLY

/* Each number above the one before it: the list is in preference order, lowest first. */
#define SILICON_FAMILIES__ZZPRIVATE_NUMBER(ID, NAME, FOLDER, VENDOR)  (ID),
static constexpr sys__silicon_family__id silicon_families__zzprivate_numbers[] = {
    SILICON_FAMILIES__LIST(SILICON_FAMILIES__ZZPRIVATE_NUMBER)
};
#undef SILICON_FAMILIES__ZZPRIVATE_NUMBER
static constexpr bool silicon_families__zzprivate_rising(unsigned i) {
    return i + 1u >= (unsigned)SILICON_FAMILIES__COUNT
        || (silicon_families__zzprivate_numbers[i] < silicon_families__zzprivate_numbers[i + 1u]
            && silicon_families__zzprivate_rising(i + 1u));
}
static_assert(silicon_families__zzprivate_rising(0u),
              "silicon family numbers must rise down the list — a machine records the number, and the list "
              "order is the preference the chooser walks");
#define SILICON_FAMILIES__ZZPRIVATE_FITS(ID, NAME, FOLDER, VENDOR)                                          \
    static_assert((unsigned)(ID) < (unsigned)SYS__SILICON_FAMILY__SLOTS,                                   \
                  "a silicon family's number is past the program's slots — raise SYS__SILICON_FAMILY__SLOTS");
SILICON_FAMILIES__LIST(SILICON_FAMILIES__ZZPRIVATE_FITS)
#undef SILICON_FAMILIES__ZZPRIVATE_FITS
static_assert((unsigned)SILICON_FAMILIES__COUNT <= (unsigned)SYS__SILICON_FAMILY__SLOTS,
              "more silicon families than the program has slots for — raise SYS__SILICON_FAMILY__SLOTS");

/* Each family's number under its folder's name, which is how a family's own translation unit says which
 * row it is: `amd_rocm6_wave64__family_id`. */
#define SILICON_FAMILIES__ZZPRIVATE_ID(ID, NAME, FOLDER, VENDOR)  \
    static const sys__silicon_family__id FOLDER##__family_id = (ID);
SILICON_FAMILIES__LIST(SILICON_FAMILIES__ZZPRIVATE_ID)
#undef SILICON_FAMILIES__ZZPRIVATE_ID

/* A family's name, from its number; null for a number not on the list. */
static inline const char* silicon_families__name(sys__silicon_family__id family) {
#define SILICON_FAMILIES__ZZPRIVATE_NAME(ID, NAME, FOLDER, VENDOR)  if (family == (ID)) return #FOLDER;
    SILICON_FAMILIES__LIST(SILICON_FAMILIES__ZZPRIVATE_NAME)
#undef SILICON_FAMILIES__ZZPRIVATE_NAME
    return 0;
}

/* ⭐⭐ THE ROSTER AS THE CHOOSER READS IT — every family's name, vendor and `includes`, in number order.
 * ▶ `choose.cuh` for how a card is given to them. */
#define SILICON_FAMILIES__ZZPRIVATE_ENTRY(ID, NAME, FOLDER, VENDOR)  { (ID), #FOLDER, #VENDOR, FOLDER##__includes },
static const silicon_families__row silicon_families__rows[] = {
    SILICON_FAMILIES__LIST(SILICON_FAMILIES__ZZPRIVATE_ENTRY)
};
#undef SILICON_FAMILIES__ZZPRIVATE_ENTRY

#endif /* SILVANN__SILICON_FAMILIES_ROSTER_CUH */
