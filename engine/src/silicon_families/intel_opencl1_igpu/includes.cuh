#ifndef SILVANN__SILICON_FAMILIES_INTEL_OPENCL1_IGPU_INCLUDES_CUH
#define SILVANN__SILICON_FAMILIES_INTEL_OPENCL1_IGPU_INCLUDES_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include "../card.cuh"   /* what a card's name is */
#include "../opencl_versions.cuh"   /* which cards' OpenCL is 1.2 */

/* ══ THE CARDS THIS FAMILY TAKES ═════════════════════════════════════════════════════════════════════
 *
 * ⚖ *"opencl 1.2 will go in the backlog alongside vulkan"* · 2026-11-09 *"old_nvidia does its own thing and does not
 * stomp against the intel old igpu"* · 2026-11-25 *"that is supposed to be the catch-all mode"*: every Intel GPU, named
 * by its OpenCL device name (`Intel(R) …`) — built from `khronos_opencl1_base`. Its number is the lowest of the OpenCL
 * families that take Intel's, so it is asked last: Intel's before Gen9, whose OpenCL is 1.2 (Ivy Bridge, Haswell, Bay
 * Trail — `silicon_families__opencl1_intel`), no other family takes; a later one reaches it where `intel_opencl2` and
 * `khronos_opencl2` open and see none (▶ `choose.cuh`) — Intel's compute runtime from Gen12 is OpenCL 3.0 with
 * coarse-grained SVM alone (`MEASURED`, measurements/2026-11-25_nn214_intel_opencl_phoenixnap.md). An integrated GPU's
 * memory is the PC's, so a span of mapped RAM is one copy there (the base's `sys.cuh`) and the doors take the integrated
 * shape (`CL_DEVICE_HOST_UNIFIED_MEMORY`); an Arc takes the discrete one.
 *
 * ⛳ AN OPENCL BINARY IS NOT BUILT FOR ITS CARDS: the targets row is what the family has been RUN on — none yet, so
 *   every card it takes is a developer preview until nn's verbs pass on one through it. */
#define INTEL_OPENCL1_IGPU__TARGETS(X)

static inline bool intel_opencl1_igpu__includes(const char* card) {
    return silicon_families__name_starts(card, "Intel(R)");
}

#endif /* SILVANN__SILICON_FAMILIES_INTEL_OPENCL1_IGPU_INCLUDES_CUH */
